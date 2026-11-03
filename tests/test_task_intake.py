from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from atlas.agent_factory import AgentTemplate, AgentTemplateRegistry, UnknownAgentRole
from atlas.agents import AgentRequest, AgentTurn
from atlas.api.app import create_app
from atlas.plans.ir import AgentProfile, ToolDefinition, ToolRiskClass
from atlas.plans.planner import PlannedEdge, PlannedNode, PlannerRequest, WorkflowProposal
from atlas.providers.gemini import GeminiAgentConfig
from atlas.providers.gemini_planner import GeminiPlanner
from atlas.runtime.backend import ToolRegistration
from atlas.security.policy import ToolPolicy


class StaticPlanner:
    def __init__(self, proposal: Any) -> None:
        self.proposal = proposal
        self.last_request: PlannerRequest | None = None

    async def plan(self, request: PlannerRequest) -> Any:
        self.last_request = request
        return self.proposal


class ResultAgent:
    def __init__(self, result: str = "ok") -> None:
        self.result = result
        self.closed = False

    async def invoke(self, request: AgentRequest) -> AgentTurn:
        del request
        return AgentTurn(
            action="final", action_summary="Return a result.", final_output=self.result
        )

    async def aclose(self) -> None:
        self.closed = True


class ToolAgent(ResultAgent):
    async def invoke(self, request: AgentRequest) -> AgentTurn:
        if not request.observations:
            return AgentTurn(
                action="act",
                action_summary="Read the authorized verification tool.",
                tool_calls=({"tool_id": "workspace.read", "arguments": {"path": "README.md"}},),
            )
        return AgentTurn(
            action="final", action_summary="Summarize the observed file.", final_output="read"
        )


class SlowClosableAgent(ResultAgent):
    async def invoke(self, request: AgentRequest) -> AgentTurn:
        del request
        await asyncio.sleep(20)
        return AgentTurn(action="final", action_summary="Late result.", final_output="late")


class ReadTool:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        return {"contents": f"authorized:{arguments['path']}"}


class FakeModels:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.aio = SimpleNamespace(models=models)


def make_profile(role: str, *, allowed_tools: tuple[str, ...] = ()) -> AgentProfile:
    return AgentProfile(
        agent_id=role,
        version="1",
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        allowed_tools=allowed_tools,
        max_iterations=2,
    )


def proposal(*, roles: tuple[str, ...] = ("reader", "reviewer")) -> WorkflowProposal:
    nodes = tuple(
        PlannedNode(node_id=f"agent_{index}", role_id=role, task_description=f"Do {role} work")
        for index, role in enumerate(roles)
    )
    edges = (PlannedEdge(source="agent_0", target="agent_1"),) if len(nodes) > 1 else ()
    return WorkflowProposal(nodes=nodes, edges=edges, max_depth=3, max_parallelism=2)


def registry(
    role_factories: dict[str, Any],
    *,
    allowed: dict[str, tuple[str, ...]] | None = None,
    tools: dict[str, ToolRegistration] | None = None,
) -> AgentTemplateRegistry:
    allowed = allowed or {}
    templates = {
        role: AgentTemplate(
            role_id=role,
            description=f"Trusted {role} role",
            profile=make_profile(role, allowed_tools=allowed.get(role, ())),
            adapter_factory=factory,
        )
        for role, factory in role_factories.items()
    }
    return AgentTemplateRegistry(templates, tools=tools or {})


def wait_for_status(client: TestClient, run_id: str, target: str = "succeeded") -> dict[str, Any]:
    for _ in range(300):
        response = client.get(f"/v1/runs/{run_id}")
        payload = response.json()
        if payload["status"] in {target, "failed", "cancelled"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("run did not reach a terminal state")


def test_task_intake_generates_compiles_and_executes_two_trusted_agents(tmp_path: Path) -> None:
    instances: list[ResultAgent] = []

    def create_agent() -> ResultAgent:
        instance = ResultAgent()
        instances.append(instance)
        return instance

    planner = StaticPlanner(proposal())
    templates = registry({"reader": create_agent, "reviewer": create_agent})
    app = create_app(data_dir=tmp_path, planner=planner, agent_templates=templates)
    with TestClient(app) as client:
        accepted = client.post("/v1/tasks", json={"task": "Check release readiness"})
        assert accepted.status_code == 202
        body = accepted.json()
        assert body["plan"]["nodes"][-1]["kind"] == "output"
        assert [node["agent_ref"] for node in body["plan"]["nodes"][:2]] == [
            "reader__agent_0",
            "reviewer__agent_1",
        ]
        assert wait_for_status(client, body["run_id"])["status"] == "succeeded"
    assert len(instances) == 2
    assert instances[0] is not instances[1]
    assert all(instance.closed for instance in instances)
    assert planner.last_request is not None
    assert {role.role_id for role in planner.last_request.available_roles} == {"reader", "reviewer"}


def test_task_intake_rejects_unknown_roles_and_operator_limit_escalation(tmp_path: Path) -> None:
    planner = StaticPlanner(proposal(roles=("attacker",)))
    templates = registry({"reader": ResultAgent})
    app = create_app(data_dir=tmp_path, planner=planner, agent_templates=templates)
    with TestClient(app) as client:
        assert client.post("/v1/tasks", json={"task": "x"}).status_code == 422

    too_large = WorkflowProposal(
        nodes=(PlannedNode(node_id="reader", role_id="reader", task_description="Do work"),),
        max_depth=9,
        max_parallelism=2,
    )
    app = create_app(
        data_dir=tmp_path / "limits",
        planner=StaticPlanner(too_large),
        agent_templates=templates,
    )
    with TestClient(app) as client:
        assert client.post("/v1/tasks", json={"task": "x"}).status_code == 422


def test_template_rejects_unknown_tools_and_planner_cannot_set_permissions() -> None:
    with pytest.raises(ValueError, match="unregistered tools"):
        registry({"reader": ResultAgent}, allowed={"reader": ("attacker.tool",)})
    malformed = proposal().model_dump(mode="json")
    malformed["allowed_tools"] = ["attacker.tool"]
    with pytest.raises(ValueError):
        WorkflowProposal.model_validate(malformed)
    with pytest.raises(UnknownAgentRole):
        registry({"reader": ResultAgent}).resolve("unknown")


def test_dynamic_agent_tool_call_uses_policy_gateway(tmp_path: Path) -> None:
    definition = ToolDefinition(
        tool_id="workspace.read",
        version="1",
        input_schema={"type": "object", "required": ["path"]},
        output_schema={"type": "object", "required": ["contents"]},
        risk_class=ToolRiskClass.READ_ONLY,
    )
    tool = ReadTool()
    tools = {"workspace.read": ToolRegistration(definition, tool)}
    planner = StaticPlanner(proposal(roles=("reader",)))
    templates = registry(
        {"reader": ToolAgent}, allowed={"reader": ("workspace.read",)}, tools=tools
    )
    app = create_app(
        data_dir=tmp_path,
        planner=planner,
        agent_templates=templates,
        tool_registry=tools,
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"workspace.read"})),
    )
    with TestClient(app) as client:
        accepted = client.post("/v1/tasks", json={"task": "Read README"})
        assert accepted.status_code == 202
        assert wait_for_status(client, accepted.json()["run_id"])["status"] == "succeeded"
    assert tool.calls == 1


def test_task_cancellation_closes_ephemeral_agent(tmp_path: Path) -> None:
    instances: list[SlowClosableAgent] = []

    def create_agent() -> SlowClosableAgent:
        instance = SlowClosableAgent()
        instances.append(instance)
        return instance

    app = create_app(
        data_dir=tmp_path,
        planner=StaticPlanner(proposal(roles=("reader",))),
        agent_templates=registry({"reader": create_agent}),
    )
    with TestClient(app) as client:
        accepted = client.post("/v1/tasks", json={"task": "slow task"})
        assert accepted.status_code == 202
        run_id = accepted.json()["run_id"]
        client.post(f"/v1/runs/{run_id}/cancel")
        assert wait_for_status(client, run_id, target="cancelled")["status"] == "cancelled"
    assert len(instances) == 1 and instances[0].closed


def test_gemini_planner_uses_schema_and_rejects_malformed_output() -> None:
    valid = json.dumps(proposal(roles=("reader",)).model_dump(mode="json"))
    models = FakeModels(valid)
    planner = GeminiPlanner(GeminiAgentConfig(), client=FakeClient(models))
    request = PlannerRequest(
        task="Check readiness",
        available_roles=({"role_id": "reader", "description": "Read"},),
    )
    result = asyncio.run(planner.plan(request))
    assert result.nodes[0].role_id == "reader"
    assert models.calls[0]["config"]["response_mime_type"] == "application/json"
    assert "allowed_tools" not in models.calls[0]["config"]["response_schema"]["properties"]

    malformed = GeminiPlanner(
        GeminiAgentConfig(), client=FakeClient(FakeModels('{"nodes": [], "permissions": []}'))
    )
    with pytest.raises(ValueError, match="malformed|schema"):
        asyncio.run(malformed.plan(request))
