from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from atlas.agents import AgentObservation, AgentRequest, AgentToolSpec, AgentTurn
from atlas.plans.compiler import PlanCompiler
from atlas.plans.ir import AgentProfile, Node, NodeKind, Plan, ToolDefinition
from atlas.providers.gemini import GeminiAgent, GeminiAgentConfig, GeminiProviderError
from atlas.runtime.backend import AgentRegistration, ToolRegistration
from atlas.runtime.langgraph_backend import LangGraphExecutionBackend
from atlas.security.policy import ToolPolicy


class FakeModels:
    def __init__(self, responses: list[Any] | None = None, error: Exception | None = None) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.aio = SimpleNamespace(models=models)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def response(
    *, text: str | None = None, calls: list[Any] | None = None, model_content: Any | None = None
) -> Any:
    return SimpleNamespace(
        text=text,
        function_calls=calls or [],
        usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=40),
        candidates=(SimpleNamespace(content=model_content),) if model_content is not None else [],
    )


class FakeModelContent:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def model_dump(self, *, mode: str, exclude_none: bool) -> dict[str, Any]:
        del mode, exclude_none
        return self.payload


def request(
    *, observations: tuple[AgentObservation, ...] = (), tools: tuple[AgentToolSpec, ...] = ()
) -> AgentRequest:
    return AgentRequest(
        instruction="Find the answer using an authorized tool if needed.",
        input_json=json.dumps({"question": "What is the answer?"}),
        iteration=2 if observations else 1,
        observations=observations,
        available_tools=tools,
    )


def test_gemini_maps_function_call_to_atlas_turn_without_executing_it() -> None:
    models = FakeModels(
        [
            response(
                calls=[
                    SimpleNamespace(name="atlas_tool_0", args={"q": "x"}),
                    SimpleNamespace(name="atlas_tool_1", args={"q": "y"}),
                ]
            )
        ]
    )
    tool = AgentToolSpec(
        tool_id="catalog.lookup",
        description="Look up a catalog item.",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    agent = GeminiAgent(GeminiAgentConfig(), client=FakeClient(models))

    tool2 = AgentToolSpec(
        tool_id="catalog.verify",
        description="Verify a catalog item.",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    turn = asyncio.run(agent.invoke(request(tools=(tool, tool2))))

    assert isinstance(turn, AgentTurn)
    assert turn.action == "act"
    assert [call.tool_id for call in turn.tool_calls] == ["catalog.lookup", "catalog.verify"]
    assert turn.tool_calls[0].arguments == {"q": "x"}
    assert turn.tool_calls[1].arguments == {"q": "y"}
    assert turn.telemetry is not None
    assert turn.telemetry.input_tokens == 100
    assert turn.telemetry.output_tokens == 40
    assert turn.telemetry.estimated_cost_usd is None
    sdk_tools = models.calls[0]["config"]["tools"]
    assert sdk_tools[0]["function_declarations"][0]["name"] == "atlas_tool_0"
    assert sdk_tools[1]["function_declarations"][0]["name"] == "atlas_tool_1"
    assert models.calls[0]["config"]["automatic_function_calling"] == {"disable": True}
    assert all(not callable(tool) for tool in sdk_tools)


def test_final_turn_is_validated_and_usage_cost_is_recorded() -> None:
    models = FakeModels(
        [
            response(
                text='{"action":"final","action_summary":"Answer the task.","final_output":"42"}'
            )
        ]
    )
    agent = GeminiAgent(
        GeminiAgentConfig(
            model="gemini-test",
            temperature=0,
            timeout_seconds=3,
            max_output_tokens=100,
            input_price_per_million_tokens_usd=1.0,
            output_price_per_million_tokens_usd=2.0,
        ),
        client=FakeClient(models),
    )

    turn = asyncio.run(agent.invoke(request()))

    assert isinstance(turn, AgentTurn)
    assert turn.action == "final"
    assert turn.final_output == "42"
    assert turn.telemetry is not None
    assert turn.telemetry.model_id == "gemini-test"
    assert turn.telemetry.provider == "google_gemini"
    assert turn.telemetry.latency_ms >= 0
    assert turn.telemetry.estimated_cost_usd == pytest.approx(0.00018)
    call = models.calls[0]
    assert call["model"] == "gemini-test"
    assert call["config"]["temperature"] == 0
    assert call["config"]["max_output_tokens"] == 100


def test_nonempty_plain_text_maps_to_provider_independent_final_turn() -> None:
    models = FakeModels([response(text="Found the ATLAS item in the demo catalog.")])
    turn = asyncio.run(GeminiAgent(client=FakeClient(models)).invoke(request()))

    assert isinstance(turn, AgentTurn)
    assert turn.action == "final"
    assert turn.final_output == "Found the ATLAS item in the demo catalog."
    assert turn.action_summary == "Gemini returned a final text response."


def test_empty_text_is_rejected() -> None:
    models = FakeModels([response(text="  ")])

    with pytest.raises(GeminiProviderError, match="neither a tool call nor a decision"):
        asyncio.run(GeminiAgent(client=FakeClient(models)).invoke(request()))


def test_missing_agent_turn_fields_include_safe_validation_diagnostics() -> None:
    models = FakeModels([response(text='{"action":"final","final_output":"answer"}')])

    with pytest.raises(GeminiProviderError) as caught:
        asyncio.run(GeminiAgent(client=FakeClient(models)).invoke(request()))

    assert caught.value.validation_errors == [
        {"field": "action_summary", "error_type": "missing", "expected": "required field"}
    ]
    assert "answer" not in str(caught.value.validation_errors)


def test_malformed_structured_output_reports_expected_action() -> None:
    models = FakeModels(
        [response(text='{"action":"finish","action_summary":"Done.","final_output":"x"}')]
    )

    with pytest.raises(GeminiProviderError) as caught:
        asyncio.run(GeminiAgent(client=FakeClient(models)).invoke(request()))

    assert caught.value.validation_errors is not None
    assert caught.value.validation_errors[0]["field"] == "action"
    assert caught.value.validation_errors[0]["error_type"] == "literal_error"
    assert "act" in caught.value.validation_errors[0]["expected"]
    assert "final" in caught.value.validation_errors[0]["expected"]


def test_observations_are_serialized_into_the_next_model_request() -> None:
    models = FakeModels(
        [
            response(
                text=json.dumps(
                    {
                        "action": "final",
                        "action_summary": "Use observed result.",
                        "final_output": "found",
                    }
                )
            )
        ]
    )
    agent = GeminiAgent(client=FakeClient(models))
    observation = AgentObservation(
        call_id="node:1:1",
        tool_id="lookup",
        iteration=1,
        arguments={"q": "x"},
        success=True,
        output={"value": "found"},
    )
    tool = AgentToolSpec(
        tool_id="lookup",
        description="Find a value.",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )

    turn = asyncio.run(agent.invoke(request(observations=(observation,), tools=(tool,))))

    contents = models.calls[0]["contents"]
    assert contents[0]["role"] == "user"
    assert contents[1] == {
        "role": "model",
        "parts": [{"function_call": {"name": "atlas_tool_0", "args": {"q": "x"}}}],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "name": "atlas_tool_0",
                    "response": {"output": {"value": "found"}},
                }
            }
        ],
    }
    assert isinstance(turn, AgentTurn) and turn.action == "final"


def test_multiple_distinct_tool_calls_have_matching_function_responses() -> None:
    models = FakeModels(
        [response(text='{"action":"final","action_summary":"Done.","final_output":"ok"}')]
    )
    tools = (
        AgentToolSpec(tool_id="lookup", description="Look up.", input_schema={"type": "object"}),
        AgentToolSpec(tool_id="verify", description="Verify.", input_schema={"type": "object"}),
    )
    observations = (
        AgentObservation(
            call_id="agent:1:1",
            tool_id="lookup",
            iteration=1,
            arguments={"q": "A"},
            success=True,
            output={"value": "A"},
        ),
        AgentObservation(
            call_id="agent:1:2",
            tool_id="verify",
            iteration=1,
            arguments={"q": "A"},
            success=True,
            output={"valid": True},
        ),
    )

    turn = asyncio.run(
        GeminiAgent(client=FakeClient(models)).invoke(
            request(observations=observations, tools=tools)
        )
    )

    contents = models.calls[0]["contents"]
    assert [part["function_call"]["name"] for part in contents[1]["parts"]] == [
        "atlas_tool_0",
        "atlas_tool_1",
    ]
    assert [part["function_response"]["name"] for part in contents[2]["parts"]] == [
        "atlas_tool_0",
        "atlas_tool_1",
    ]
    assert [part["function_response"]["response"] for part in contents[2]["parts"]] == [
        {"output": {"value": "A"}},
        {"output": {"valid": True}},
    ]
    assert isinstance(turn, AgentTurn) and turn.action == "final"


def test_repeated_identical_tool_requests_remain_distinct_history_turns() -> None:
    models = FakeModels(
        [response(text='{"action":"final","action_summary":"Done.","final_output":"ok"}')]
    )
    tool = AgentToolSpec(tool_id="lookup", description="Look up.", input_schema={"type": "object"})
    observation = lambda iteration, call_id: AgentObservation(  # noqa: E731
        call_id=call_id,
        tool_id="lookup",
        iteration=iteration,
        arguments={"q": "ATLAS"},
        success=True,
        output={"result": "found"},
    )

    asyncio.run(
        GeminiAgent(client=FakeClient(models)).invoke(
            request(
                observations=(observation(1, "agent:1:1"), observation(2, "agent:2:1")),
                tools=(tool,),
            )
        )
    )

    contents = models.calls[0]["contents"]
    assert len(contents) == 5
    for model_turn, response_turn in ((contents[1], contents[2]), (contents[3], contents[4])):
        function_call = model_turn["parts"][0]["function_call"]
        function_response = response_turn["parts"][0]["function_response"]
        assert function_call == {"name": "atlas_tool_0", "args": {"q": "ATLAS"}}
        assert function_response == {
            "name": "atlas_tool_0",
            "response": {"output": {"result": "found"}},
        }


def test_provider_errors_are_sanitized() -> None:
    secret = "never-print-this-key"
    models = FakeModels(error=RuntimeError(f"request failed with key {secret}"))
    agent = GeminiAgent(client=FakeClient(models))

    with pytest.raises(GeminiProviderError) as caught:
        asyncio.run(agent.invoke(request()))

    assert secret not in str(caught.value)
    assert "RuntimeError" in str(caught.value)


def test_provider_http_errors_include_safe_actionable_category() -> None:
    class RateLimitError(RuntimeError):
        code = 429
        status = "RESOURCE_EXHAUSTED"
        message = "quota exhausted"

    secret = "provider-error-secret"
    models = FakeModels(error=RateLimitError(f"raw body with {secret}"))
    agent = GeminiAgent(client=FakeClient(models))

    with pytest.raises(GeminiProviderError) as caught:
        asyncio.run(agent.invoke(request()))

    assert "HTTP 429" in str(caught.value)
    assert caught.value.provider_code == 429
    assert caught.value.provider_status == "RESOURCE_EXHAUSTED"
    assert caught.value.provider_message == "quota exhausted"
    assert secret not in str(caught.value)


def test_provider_400_diagnostics_preserve_sanitized_sdk_fields() -> None:
    class InvalidArgumentError(RuntimeError):
        code = 400
        status = "INVALID_ARGUMENT"
        message = "Invalid function response; api_key=provider-secret-value"

    models = FakeModels(error=InvalidArgumentError("untrusted raw exception text"))
    agent = GeminiAgent(client=FakeClient(models))

    with pytest.raises(GeminiProviderError) as caught:
        asyncio.run(agent.invoke(request()))

    assert caught.value.provider_code == 400
    assert caught.value.provider_status == "INVALID_ARGUMENT"
    assert caught.value.provider_message == "Invalid function response; api_key=[REDACTED]"
    assert "HTTP 400 INVALID_ARGUMENT" in caught.value.safe_summary
    assert "Invalid function response" in caught.value.safe_summary
    assert "provider-secret-value" not in str(caught.value)
    assert "untrusted raw exception text" not in str(caught.value)


def test_bad_model_json_is_reported_without_raw_response() -> None:
    models = FakeModels([response(text='{"action":')])
    agent = GeminiAgent(client=FakeClient(models))

    with pytest.raises(GeminiProviderError) as caught:
        asyncio.run(agent.invoke(request()))

    assert caught.value.validation_errors == [
        {
            "field": "$",
            "error_type": "json_invalid",
            "expected": "valid JSON AgentTurn object",
        }
    ]


def test_configured_key_is_redacted_from_structured_model_output() -> None:
    key = "AIza-secret-test-value"
    models = FakeModels(
        [
            response(
                text=json.dumps(
                    {
                        "action": "final",
                        "action_summary": f"Completed with {key}.",
                        "final_output": {"api_key": key, "value": "safe"},
                    }
                )
            )
        ]
    )
    agent = GeminiAgent(client=FakeClient(models), api_key=key)

    turn = asyncio.run(agent.invoke(request()))

    assert isinstance(turn, AgentTurn)
    assert key not in str(turn.model_dump(mode="json"))
    assert turn.final_output["api_key"] == "[REDACTED]"


def test_api_key_file_and_environment_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_file = tmp_path / "gemini-key"
    secret_file.write_text("file-key\n", encoding="utf-8")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY_FILE", str(secret_file))
    assert GeminiAgent._resolve_api_key(None, None) == "file-key"
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")
    assert GeminiAgent._resolve_api_key(None, None) == "environment-key"
    assert GeminiAgent._resolve_api_key("explicit-key", secret_file) == "explicit-key"


def test_generation_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="temperature"):
        GeminiAgentConfig(temperature=3)
    with pytest.raises(ValueError, match="timeout_seconds"):
        GeminiAgentConfig(timeout_seconds=0)


def test_fake_gemini_function_call_executes_only_through_atlas_gateway(tmp_path: Path) -> None:
    key = "AIza-integration-secret"
    models = FakeModels(
        [
            response(
                calls=[
                    SimpleNamespace(
                        id="gemini-call-123", name="atlas_tool_0", args={"query": "ATLAS"}
                    )
                ],
                model_content=FakeModelContent(
                    {
                        "role": "model",
                        "parts": [
                            {
                                "function_call": {
                                    "id": "gemini-call-123",
                                    "name": "atlas_tool_0",
                                    "args": {"query": "ATLAS"},
                                },
                                "thought_signature": b"opaque-signature",
                            }
                        ],
                    }
                ),
            ),
            response(text="Found the ATLAS item in the demo catalog."),
        ]
    )
    profile = AgentProfile(
        agent_id="gemini-test-agent",
        version="1",
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        allowed_tools=("catalog.lookup",),
    )
    definition = ToolDefinition(
        tool_id="catalog.lookup",
        version="1",
        input_schema={"type": "object", "required": ["query"]},
        output_schema={"type": "object", "required": ["result"]},
        description="Look up a catalog entry.",
        risk_class="read_only",
    )

    class CatalogTool:
        calls = 0

        async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
            self.calls += 1
            return {"result": key}

    catalog = CatalogTool()
    plan = Plan(
        plan_id="gemini-fake-react",
        task_id="gemini-fake-react",
        nodes=(
            Node(
                node_id="agent",
                kind=NodeKind.AGENT,
                name="Find the catalog entry.",
                agent_ref=profile.agent_id,
                output_schema={"type": "string"},
            ),
            Node(node_id="finish", kind=NodeKind.OUTPUT, name="Finish", depends_on=("agent",)),
        ),
    )
    compiled = PlanCompiler(
        agents={profile.agent_id: profile}, tools={definition.tool_id: definition}
    ).compile(plan)
    agent = GeminiAgent(client=FakeClient(models), api_key=key)
    backend = LangGraphExecutionBackend(
        agent_registry={profile.agent_id: AgentRegistration(profile, agent)},
        tool_registry={definition.tool_id: ToolRegistration(definition, catalog)},
        checkpoint_path=str(tmp_path / "gemini-react.sqlite"),
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({definition.tool_id})),
    )

    result = asyncio.run(backend.execute(compiled, run_id="gemini-fake-react"))

    assert result.status.value == "succeeded"
    assert result.outputs["agent"] == "Found the ATLAS item in the demo catalog."
    assert catalog.calls == 1
    history = models.calls[1]["contents"]
    assert history[1]["parts"][0]["function_call"] == {
        "id": "gemini-call-123",
        "name": "atlas_tool_0",
        "args": {"query": "ATLAS"},
    }
    assert history[1]["parts"][0]["thought_signature"] == b"opaque-signature"
    assert history[2]["parts"][0]["function_response"] == {
        "id": "gemini-call-123",
        "name": "atlas_tool_0",
        "response": {"output": {"result": "[REDACTED]"}},
    }
    assert models.calls[0]["config"]["tools"] == models.calls[1]["config"]["tools"]
    assert key not in str(result.agent_traces)
    decision = result.agent_traces["agent"][0]
    assert decision["provider"] == "google_gemini"
    assert decision["model_id"] == "gemini-3.5-flash-lite"
