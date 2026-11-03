from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from atlas.agents import AgentRequest, AgentTelemetry, AgentToolCall, AgentTurn, FakeAgent
from atlas.plans.compiler import PlanCompiler
from atlas.plans.ir import AgentProfile, Node, NodeKind, Plan, ToolDefinition
from atlas.runtime.backend import AgentRegistration, ExecutionResult, ToolRegistration
from atlas.runtime.langgraph_backend import LangGraphExecutionBackend
from atlas.security.policy import ToolPolicy


class ScriptedAgent:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[AgentRequest] = []

    async def invoke(self, request: AgentRequest) -> Any:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("agent was invoked beyond its script")
        return self.responses.pop(0)


class RecordingTool:
    def __init__(self, output: dict[str, object] | None = None, *, fail: bool = False) -> None:
        self.output = output or {"ok": True}
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        self.calls.append(dict(arguments))
        if self.fail:
            raise OSError("private backend detail")
        return dict(self.output)


class ModelErrorAgent:
    async def invoke(self, request: AgentRequest) -> str:
        del request
        raise RuntimeError("sensitive provider text")


class CategorizedErrorAgent:
    class ProviderFailure(RuntimeError):
        safe_summary = (
            "Gemini API request failed (HTTP 400 INVALID_ARGUMENT). Bad function response."
        )
        provider_code = 400
        provider_status = "INVALID_ARGUMENT"
        provider_message = "Bad function response."
        validation_errors = [
            {"field": "action_summary", "error_type": "missing", "expected": "required field"}
        ]

    async def invoke(self, request: AgentRequest) -> str:
        del request
        raise self.ProviderFailure("raw secret-bearing provider response")


def tool_definitions(
    tool_ids: tuple[str, ...], *, input_schema: dict[str, Any] | None = None
) -> dict[str, ToolDefinition]:
    return {
        tool_id: ToolDefinition(
            tool_id=tool_id,
            version="1",
            description=f"Fixture tool {tool_id}",
            input_schema=input_schema or {"type": "object"},
            output_schema={"type": "object"},
            risk_class="read_only",
        )
        for tool_id in tool_ids
    }


def make_backend(
    tmp_path: Path,
    agent: Any,
    *,
    allowed_tools: tuple[str, ...] = (),
    tools: dict[str, RecordingTool] | None = None,
    definitions: dict[str, ToolDefinition] | None = None,
    max_iterations: int = 8,
    node_timeout: int = 10,
    max_tool_calls_per_iteration: int = 4,
    max_total_tool_calls_per_agent: int = 16,
) -> tuple[LangGraphExecutionBackend, Any]:
    resolved_tools = definitions or tool_definitions(tuple((tools or {}).keys()))
    profile = AgentProfile(
        agent_id="react-agent",
        version="1",
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        allowed_tools=allowed_tools,
        max_iterations=max_iterations,
    )
    plan = Plan(
        plan_id="react-plan",
        task_id="react-task",
        nodes=(
            Node(
                node_id="agent",
                kind=NodeKind.AGENT,
                name="Complete the requested task",
                agent_ref="react-agent",
                timeout_seconds=node_timeout,
                output_schema={"type": "string"},
            ),
            Node(
                node_id="finish",
                kind=NodeKind.OUTPUT,
                name="Finish",
                depends_on=("agent",),
            ),
        ),
    )
    compiled = PlanCompiler(agents={"react-agent": profile}, tools=resolved_tools).compile(plan)
    registrations = {
        tool_id: ToolRegistration(definition, (tools or {})[tool_id])
        for tool_id, definition in resolved_tools.items()
        if tool_id in (tools or {})
    }
    backend = LangGraphExecutionBackend(
        agent_registry={"react-agent": AgentRegistration(profile, agent)},
        tool_registry=registrations,
        checkpoint_path=str(tmp_path / f"{len(list(tmp_path.iterdir()))}-react.sqlite"),
        max_tool_calls_per_iteration=max_tool_calls_per_iteration,
        max_total_tool_calls_per_agent=max_total_tool_calls_per_agent,
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset(registrations)),
    )
    return backend, compiled


async def run_agent(
    backend: LangGraphExecutionBackend, compiled: Any, run_id: str
) -> ExecutionResult:
    return await backend.execute(compiled, run_id=run_id, inputs={"task": "test"})


@pytest.mark.asyncio
async def test_agent_completes_without_tools(tmp_path: Path) -> None:
    backend, plan = make_backend(tmp_path, FakeAgent("complete"))

    result = await run_agent(backend, plan, "react-no-tools")

    assert result.node_statuses["agent"] == "succeeded"
    assert result.outputs["agent"] == "complete"
    assert [event["event_type"] for event in result.agent_traces["agent"]] == [
        "decision",
        "completion",
        "node_outcome",
    ]


@pytest.mark.asyncio
async def test_provider_telemetry_is_recorded_in_agent_decision_trace(tmp_path: Path) -> None:
    agent = ScriptedAgent(
        [
            AgentTurn(
                action="final",
                action_summary="Return the checked result.",
                final_output="ok",
                telemetry=AgentTelemetry(
                    provider="google_gemini",
                    model_id="gemini-test",
                    latency_ms=12.5,
                    input_tokens=20,
                    output_tokens=5,
                    estimated_cost_usd=0.0001,
                ),
            )
        ]
    )
    backend, plan = make_backend(tmp_path, agent)

    result = await run_agent(backend, plan, "react-provider-telemetry")

    decision = result.agent_traces["agent"][0]
    assert decision["provider"] == "google_gemini"
    assert decision["model_id"] == "gemini-test"
    assert decision["latency_ms"] == 12.5
    assert decision["input_tokens"] == 20
    assert decision["output_tokens"] == 5
    assert decision["estimated_cost_usd"] == 0.0001


@pytest.mark.asyncio
async def test_agent_calls_one_tool_then_finishes(tmp_path: Path) -> None:
    tool = RecordingTool({"answer": 42})
    agent = ScriptedAgent(
        [
            AgentTurn(
                action="act",
                action_summary="Look up the answer.",
                tool_calls=(AgentToolCall(tool_id="lookup", arguments={"q": "answer"}),),
            ),
            AgentTurn(
                action="final", action_summary="Return the observed answer.", final_output="42"
            ),
        ]
    )
    backend, plan = make_backend(
        tmp_path,
        agent,
        allowed_tools=("lookup",),
        tools={"lookup": tool},
    )

    result = await run_agent(backend, plan, "react-one-tool")

    assert result.outputs["agent"] == "42"
    assert len(tool.calls) == 1
    assert agent.requests[1].iteration == 2
    assert agent.requests[1].observations[0].output == {"answer": 42}
    tool_call_trace = next(
        event for event in result.agent_traces["agent"] if event["event_type"] == "tool_call"
    )
    observation_trace = next(
        event for event in result.agent_traces["agent"] if event["event_type"] == "observation"
    )
    assert tool_call_trace["arguments"] == {"q": "answer"}
    assert observation_trace["output"] == {"answer": 42}
    assert tool_call_trace["agent_version"] == "1"


@pytest.mark.asyncio
async def test_agent_calls_multiple_tools_sequentially_in_one_turn(tmp_path: Path) -> None:
    order: list[str] = []

    class OrderedTool(RecordingTool):
        def __init__(self, tool_id: str, output: dict[str, object]) -> None:
            super().__init__(output)
            self.tool_id = tool_id

        async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
            order.append(self.tool_id)
            return await super().invoke(arguments)

    tools = {
        "first": OrderedTool("first", {"value": 1}),
        "second": OrderedTool("second", {"value": 2}),
    }
    agent = ScriptedAgent(
        [
            AgentTurn(
                action="act",
                action_summary="Gather both values.",
                tool_calls=(
                    AgentToolCall(tool_id="first", arguments={}),
                    AgentToolCall(tool_id="second", arguments={}),
                ),
            ),
            AgentTurn(
                action="final", action_summary="Combine both observations.", final_output="1+2"
            ),
        ]
    )
    backend, plan = make_backend(
        tmp_path,
        agent,
        allowed_tools=("first", "second"),
        tools=tools,
    )

    result = await run_agent(backend, plan, "react-multiple-tools")

    assert result.status.value == "succeeded"
    assert order == ["first", "second"]
    assert [obs.output for obs in agent.requests[1].observations] == [{"value": 1}, {"value": 2}]


@pytest.mark.asyncio
async def test_agent_performs_multiple_reason_act_observe_iterations(tmp_path: Path) -> None:
    tool = RecordingTool({"value": "observed"})
    agent = ScriptedAgent(
        [
            AgentTurn(
                action="act",
                action_summary="Collect evidence.",
                tool_calls=(AgentToolCall(tool_id="inspect", arguments={}),),
            ),
            AgentTurn(
                action="act",
                action_summary="Verify the evidence.",
                tool_calls=(AgentToolCall(tool_id="inspect", arguments={"again": True}),),
            ),
            AgentTurn(
                action="final", action_summary="Evidence is sufficient.", final_output="verified"
            ),
        ]
    )
    backend, plan = make_backend(
        tmp_path, agent, allowed_tools=("inspect",), tools={"inspect": tool}
    )

    result = await run_agent(backend, plan, "react-multiple-iterations")

    assert result.status.value == "succeeded"
    assert [request.iteration for request in agent.requests] == [1, 2, 3]
    assert len(agent.requests[2].observations) == 2
    assert result.agent_traces["agent"][-1]["event_type"] == "node_outcome"
    assert result.agent_traces["agent"][-1]["success"] is True


@pytest.mark.asyncio
async def test_agent_observes_tool_failure_and_recovers(tmp_path: Path) -> None:
    tool = RecordingTool(fail=True)
    agent = ScriptedAgent(
        [
            AgentTurn(
                action="act",
                action_summary="Try the lookup.",
                tool_calls=(AgentToolCall(tool_id="lookup", arguments={}),),
            ),
            AgentTurn(
                action="final",
                action_summary="Report lookup failure safely.",
                final_output="unavailable",
            ),
        ]
    )
    backend, plan = make_backend(tmp_path, agent, allowed_tools=("lookup",), tools={"lookup": tool})

    result = await run_agent(backend, plan, "react-tool-recovery")

    assert result.status.value == "succeeded"
    observation = agent.requests[1].observations[0]
    assert observation.success is False
    assert "OSError" in (observation.error or "")
    assert "private backend detail" not in str(result.agent_traces)


@pytest.mark.asyncio
async def test_iteration_limit_fails_without_an_infinite_loop(tmp_path: Path) -> None:
    repeated_action = AgentTurn(
        action="act",
        action_summary="Request an unavailable operation.",
        tool_calls=(AgentToolCall(tool_id="hidden", arguments={}),),
    )
    agent = ScriptedAgent([repeated_action, repeated_action])
    backend, plan = make_backend(tmp_path, agent, max_iterations=2)

    result = await run_agent(backend, plan, "react-iteration-limit")

    assert result.status.value == "failed"
    assert len(agent.requests) == 2
    assert result.errors["agent"] == "AgentExecutionError: maximum agent iterations reached (2)"
    assert result.agent_traces["agent"][-2]["event_type"] == "iteration_limit"
    assert result.agent_traces["agent"][-1]["event_type"] == "node_outcome"
    assert result.agent_traces["agent"][-1]["success"] is False


@pytest.mark.asyncio
async def test_structured_final_turn_terminates_and_does_not_log_thoughts(tmp_path: Path) -> None:
    agent = ScriptedAgent(
        [AgentTurn(action="final", action_summary="The result is ready.", final_output="done")]
    )
    backend, plan = make_backend(tmp_path, agent)

    result = await run_agent(backend, plan, "react-final")

    assert result.outputs["agent"] == "done"
    trace = result.agent_traces["agent"]
    assert [event["event_type"] for event in trace] == [
        "decision",
        "completion",
        "node_outcome",
    ]
    assert all("thought" not in event and "reasoning" not in event for event in trace)


@pytest.mark.asyncio
async def test_invalid_final_output_reports_expected_schema_type(tmp_path: Path) -> None:
    agent = ScriptedAgent(
        [
            AgentTurn(
                action="final",
                action_summary="The result is ready.",
                final_output={"text": "Found it."},
            )
        ]
    )
    backend, plan = make_backend(tmp_path, agent)

    result = await run_agent(backend, plan, "react-invalid-final-output")

    assert result.status.value == "failed"
    event = result.agent_traces["agent"][-2]
    assert event["error_summary"] == "Invalid agent final output."
    assert event["validation_errors"] == [
        {
            "field": "$",
            "error_type": "type",
            "expected": "string",
            "actual_type": "dict",
        }
    ]
    assert "Found it" not in str(result.agent_traces)


@pytest.mark.asyncio
async def test_agent_cannot_call_tool_outside_its_allowlist(tmp_path: Path) -> None:
    forbidden = RecordingTool({"secret": "value"})
    agent = ScriptedAgent(
        [
            AgentTurn(
                action="act",
                action_summary="Try a tool not assigned to this agent.",
                tool_calls=(AgentToolCall(tool_id="secret", arguments={}),),
            ),
            AgentTurn(
                action="final", action_summary="Continue without that tool.", final_output="safe"
            ),
        ]
    )
    definitions = tool_definitions(("safe", "secret"))
    backend, plan = make_backend(
        tmp_path,
        agent,
        allowed_tools=("safe",),
        tools={"safe": RecordingTool(), "secret": forbidden},
        definitions=definitions,
    )

    result = await run_agent(backend, plan, "react-unauthorized-tool")

    assert result.status.value == "succeeded"
    assert forbidden.calls == []
    rejected = next(
        event for event in result.agent_traces["agent"] if event["event_type"] == "tool_call"
    )
    assert rejected["accepted"] is False
    assert agent.requests[1].observations[0].success is False


@pytest.mark.asyncio
async def test_per_iteration_tool_call_limit_prevents_uncontrolled_execution(
    tmp_path: Path,
) -> None:
    first = RecordingTool({"tool": "first"})
    second = RecordingTool({"tool": "second"})
    agent = ScriptedAgent(
        [
            AgentTurn(
                action="act",
                action_summary="Request two tools at once.",
                tool_calls=(
                    AgentToolCall(tool_id="first", arguments={}),
                    AgentToolCall(tool_id="second", arguments={}),
                ),
            ),
            AgentTurn(
                action="final",
                action_summary="Continue with available results.",
                final_output="done",
            ),
        ]
    )
    backend, plan = make_backend(
        tmp_path,
        agent,
        allowed_tools=("first", "second"),
        tools={"first": first, "second": second},
        max_tool_calls_per_iteration=1,
    )

    result = await run_agent(backend, plan, "react-call-limit")

    assert result.status.value == "succeeded"
    assert len(first.calls) == 1
    assert second.calls == []
    assert agent.requests[1].observations[1].error == "Per-iteration tool-call limit reached."


@pytest.mark.asyncio
async def test_invalid_tool_arguments_are_observed_without_invocation(tmp_path: Path) -> None:
    tool = RecordingTool()
    definition = tool_definitions(
        ("lookup",),
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    agent = ScriptedAgent(
        [
            AgentTurn(
                action="act",
                action_summary="Send incomplete arguments.",
                tool_calls=(AgentToolCall(tool_id="lookup", arguments={}),),
            ),
            AgentTurn(
                action="final",
                action_summary="Recover from the validation result.",
                final_output="done",
            ),
        ]
    )
    backend, plan = make_backend(
        tmp_path,
        agent,
        allowed_tools=("lookup",),
        tools={"lookup": tool},
        definitions=definition,
    )

    result = await run_agent(backend, plan, "react-invalid-tool-input")

    assert result.status.value == "succeeded"
    assert tool.calls == []
    assert (
        agent.requests[1].observations[0].error == "Tool arguments failed input schema validation."
    )


@pytest.mark.asyncio
async def test_model_errors_fail_gracefully_and_are_traced(tmp_path: Path) -> None:
    backend, plan = make_backend(tmp_path, ModelErrorAgent())

    result = await run_agent(backend, plan, "react-model-error")

    assert result.status.value == "failed"
    assert result.agent_traces["agent"][-2]["event_type"] == "model_error"
    assert result.agent_traces["agent"][-1]["event_type"] == "node_outcome"
    assert "sensitive provider text" not in str(result.agent_traces)
    assert result.agent_traces["agent"][-2]["input_tokens"] is None
    assert result.agent_traces["agent"][-2]["output_tokens"] is None
    assert result.agent_traces["agent"][-1]["input_tokens"] is None
    assert result.agent_traces["agent"][-1]["output_tokens"] is None


@pytest.mark.asyncio
async def test_safe_provider_error_category_is_preserved_without_raw_error(tmp_path: Path) -> None:
    backend, plan = make_backend(tmp_path, CategorizedErrorAgent())

    result = await run_agent(backend, plan, "react-categorized-error")

    event = result.agent_traces["agent"][-2]
    assert event["error_summary"] == (
        "Gemini API request failed (HTTP 400 INVALID_ARGUMENT). Bad function response."
    )
    assert event["provider_error_code"] == 400
    assert event["provider_status"] == "INVALID_ARGUMENT"
    assert event["provider_message"] == "Bad function response."
    assert event["validation_errors"] == [
        {"field": "action_summary", "error_type": "missing", "expected": "required field"}
    ]
    assert "raw secret-bearing" not in str(result.agent_traces)


@pytest.mark.asyncio
async def test_node_timeout_bounds_the_whole_react_loop(tmp_path: Path) -> None:
    class SlowAgent:
        async def invoke(self, request: AgentRequest) -> str:
            del request
            await asyncio.sleep(10)
            return "too late"

    backend, plan = make_backend(tmp_path, SlowAgent(), node_timeout=1)

    result = await run_agent(backend, plan, "react-timeout")

    assert result.status.value == "failed"
    assert result.agent_traces["agent"][-1]["event_type"] == "node_outcome"
    assert result.agent_traces["agent"][-1]["success"] is False
    assert result.agent_traces["agent"][-1]["error_type"] == "TimeoutError"
