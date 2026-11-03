from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.requests import Request

from atlas.agents import AgentRequest, AgentTurn
from atlas.api.app import create_app
from atlas.plans.ir import AgentProfile, ToolDefinition, ToolRiskClass
from atlas.runtime.backend import AgentRegistration, ToolRegistration
from atlas.security.policy import ToolPolicy


class StaticAgent:
    async def invoke(self, request: AgentRequest) -> AgentTurn:
        del request
        return AgentTurn(
            action="final", action_summary="Return a fixture result.", final_output="ok"
        )


class SlowAgent:
    async def invoke(self, request: AgentRequest) -> AgentTurn:
        del request
        await asyncio.sleep(20)
        return AgentTurn(action="final", action_summary="late", final_output="late")


class FixtureTool:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        return {"value": arguments["mapped"]["value"]}  # type: ignore[index]


def simple_plan() -> dict[str, object]:
    return {
        "plan_id": "api-test",
        "task_id": "api-task",
        "nodes": [
            {"node_id": "start", "kind": "join", "name": "Start"},
            {"node_id": "finish", "kind": "output", "name": "Finish", "depends_on": ["start"]},
        ],
    }


def test_health_and_async_run_lifecycle(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {
            "status": "ok",
            "checks": {"process": "alive"},
        }
        assert client.get("/health/ready").status_code == 200
        submitted = client.post("/v1/runs", json={"plan": simple_plan(), "inputs": {"x": 1}})
        assert submitted.status_code == 202
        accepted = submitted.json()
        assert accepted["status"] == "queued"
        run_id = accepted["run_id"]

        for _ in range(100):
            status_response = client.get(f"/v1/runs/{run_id}")
            if status_response.json()["status"] == "succeeded":
                break
            time.sleep(0.01)
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "succeeded"
        events = client.get(f"/v1/runs/{run_id}/events").json()["events"]
        assert any(event["event_type"] == "run.started" for event in events)
        assert events[-1]["event_type"] == "run.completed"
        assert events[0]["event_type"] == "run.queued"
        assert events[0]["artifact_refs"][0]["media_type"] == (
            "application/vnd.atlas.run-request+json"
        )


def test_run_plan_validation_and_unknown_run_errors(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        invalid = client.post("/v1/runs", json={"plan": {**simple_plan(), "unknown": True}})
        assert invalid.status_code == 422
        unregistered_tool_plan = {
            "plan_id": "untrusted-tool",
            "task_id": "untrusted-tool-task",
            "nodes": [
                {
                    "node_id": "tool",
                    "kind": "tool",
                    "name": "Unregistered",
                    "tool_ref": "attacker-supplied",
                },
                {
                    "node_id": "finish",
                    "kind": "output",
                    "name": "Finish",
                    "depends_on": ["tool"],
                },
            ],
        }
        assert client.post("/v1/runs", json={"plan": unregistered_tool_plan}).status_code == 422
        assert client.get("/v1/runs/missing").status_code == 404


def test_static_tool_registry_and_gateway_policy_are_server_owned(tmp_path: Path) -> None:
    definition = ToolDefinition(
        tool_id="fixture",
        version="1",
        input_schema={"type": "object"},
        output_schema={"type": "object", "required": ["value"]},
        risk_class=ToolRiskClass.READ_ONLY,
    )
    tool = FixtureTool()
    app = create_app(
        data_dir=tmp_path,
        tool_registry={"fixture": ToolRegistration(definition, tool)},
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})),
    )
    plan = {
        "plan_id": "tool-api-test",
        "task_id": "tool-api-task",
        "nodes": [
            {
                "node_id": "read",
                "kind": "tool",
                "name": "Read fixture",
                "tool_ref": "fixture",
                "input_mapping": {"value": "hello"},
                "output_schema": {"type": "object", "required": ["value"]},
            },
            {"node_id": "finish", "kind": "output", "name": "Finish", "depends_on": ["read"]},
        ],
    }
    with TestClient(app) as client:
        accepted = client.post("/v1/runs", json={"plan": plan}).json()
        for _ in range(100):
            status_response = client.get(f"/v1/runs/{accepted['run_id']}")
            if status_response.json()["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        assert status_response.json()["status"] == "succeeded"
    assert tool.calls == 1


def test_registered_tool_still_requires_operator_gateway_allowance(tmp_path: Path) -> None:
    definition = ToolDefinition(
        tool_id="fixture",
        version="1",
        input_schema={"type": "object"},
        output_schema={"type": "object", "required": ["value"]},
        risk_class=ToolRiskClass.READ_ONLY,
    )
    tool = FixtureTool()
    app = create_app(
        data_dir=tmp_path,
        tool_registry={"fixture": ToolRegistration(definition, tool)},
        tool_policy=ToolPolicy(),
    )
    plan = {
        "plan_id": "denied-tool-api-test",
        "task_id": "denied-tool-api-task",
        "nodes": [
            {
                "node_id": "read",
                "kind": "tool",
                "name": "Read fixture",
                "tool_ref": "fixture",
                "input_mapping": {"value": "hello"},
                "output_schema": {"type": "object", "required": ["value"]},
            },
            {"node_id": "finish", "kind": "output", "name": "Finish", "depends_on": ["read"]},
        ],
    }
    with TestClient(app) as client:
        accepted = client.post("/v1/runs", json={"plan": plan}).json()
        for _ in range(100):
            response = client.get(f"/v1/runs/{accepted['run_id']}")
            if response.json()["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        assert response.json()["status"] == "failed"
    assert tool.calls == 0


def test_cancellation_and_authentication_interface(tmp_path: Path) -> None:
    profile = AgentProfile(
        agent_id="slow",
        version="1",
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        max_iterations=1,
    )
    plan = {
        "plan_id": "cancel-api-test",
        "task_id": "cancel-api-task",
        "nodes": [
            {
                "node_id": "agent",
                "kind": "agent",
                "name": "Slow fixture",
                "agent_ref": "slow",
                "output_schema": {"type": "string"},
            },
            {"node_id": "finish", "kind": "output", "name": "Finish", "depends_on": ["agent"]},
        ],
    }

    class HeaderAuthenticator:
        async def authenticate(self, request: Request) -> str | None:
            return "test-user" if request.headers.get("authorization") == "Bearer test" else None

    app = create_app(
        data_dir=tmp_path,
        agent_registry={"slow": AgentRegistration(profile, SlowAgent())},
        authenticator=HeaderAuthenticator(),
        require_authentication=True,
    )
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        headers = {"Authorization": "Bearer test"}
        accepted = client.post("/v1/runs", json={"plan": plan}, headers=headers)
        assert accepted.status_code == 202
        run_id = accepted.json()["run_id"]
        cancellation = client.post(f"/v1/runs/{run_id}/cancel", headers=headers)
        assert cancellation.status_code == 200
        assert cancellation.json()["status"] == "cancelled"
        result = client.get(f"/v1/runs/{run_id}", headers=headers).json()
        assert result["status"] == "cancelled"


def test_admission_limit_rejects_excess_queued_work(tmp_path: Path) -> None:
    profile = AgentProfile(
        agent_id="slow",
        version="1",
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        max_iterations=1,
    )
    app = create_app(
        data_dir=tmp_path,
        agent_registry={"slow": AgentRegistration(profile, SlowAgent())},
        max_concurrent_runs=1,
        max_admitted_runs=1,
    )
    plan = {
        "plan_id": "bounded-admission",
        "task_id": "bounded-admission-task",
        "nodes": [
            {
                "node_id": "agent",
                "kind": "agent",
                "name": "Slow fixture",
                "agent_ref": "slow",
                "output_schema": {"type": "string"},
            },
            {"node_id": "finish", "kind": "output", "name": "Finish", "depends_on": ["agent"]},
        ],
    }
    with TestClient(app) as client:
        accepted = client.post("/v1/runs", json={"plan": plan})
        assert accepted.status_code == 202
        rejected = client.post("/v1/runs", json={"plan": plan})
        assert rejected.status_code == 429
        assert rejected.headers["retry-after"] == "1"
        client.post(f"/v1/runs/{accepted.json()['run_id']}/cancel")
