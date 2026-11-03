"""Bounded, tool-authorized ReAct loop for one Atlas agent node."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Any, Literal

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate
from pydantic import BaseModel, ConfigDict

from atlas.agents import (
    Agent,
    AgentObservation,
    AgentRequest,
    AgentTelemetry,
    AgentToolSpec,
    AgentTurn,
)
from atlas.plans.ir import AgentProfile
from atlas.runtime.backend import ToolRegistration
from atlas.security.policy import ToolDenied, ToolGatewaySession
from atlas.security.redaction import SecretRedactor


class AgentTraceEvent(BaseModel):
    """Structured, chain-of-thought-free evidence for one ReAct event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    node_id: str
    agent_id: str
    agent_version: str
    sequence: int
    iteration: int
    attempt: int
    event_type: Literal[
        "decision",
        "tool_call",
        "observation",
        "completion",
        "model_error",
        "iteration_limit",
        "node_outcome",
    ]
    action_summary: str | None = None
    call_id: str | None = None
    tool_id: str | None = None
    arguments: dict[str, Any] | None = None
    output: Any = None
    success: bool | None = None
    accepted: bool | None = None
    error_type: str | None = None
    error_summary: str | None = None
    provider_error_code: int | None = None
    provider_status: str | None = None
    provider_message: str | None = None
    validation_errors: list[dict[str, str]] | None = None
    provider: str | None = None
    model_id: str | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None


class AgentExecutionError(RuntimeError):
    """A safe summary of an agent failure; raw model reasoning is never attached."""


class ReActAgentRunner:
    """Runs Reason → Act → Observe iterations inside one agent node."""

    def __init__(
        self,
        tool_registry: Mapping[str, ToolRegistration],
        *,
        max_iterations_cap: int = 16,
        max_tool_calls_per_iteration: int = 4,
        max_total_tool_calls: int = 16,
        redactor: SecretRedactor | None = None,
    ) -> None:
        if min(max_iterations_cap, max_tool_calls_per_iteration, max_total_tool_calls) < 1:
            raise ValueError("ReAct iteration and tool-call limits must be positive")
        self.tool_registry = tool_registry
        self.max_iterations_cap = max_iterations_cap
        self.max_tool_calls_per_iteration = max_tool_calls_per_iteration
        self.max_total_tool_calls = max_total_tool_calls
        self.redactor = redactor or SecretRedactor()

    async def run(
        self,
        *,
        run_id: str,
        node_id: str,
        instruction: str,
        input_json: str,
        profile: AgentProfile,
        agent: Agent,
        trace_sink: list[dict[str, Any]],
        attempt: int = 1,
        gateway: ToolGatewaySession | None = None,
    ) -> Any:
        redactor = self.redactor.with_secrets(tuple(getattr(agent, "redaction_secrets", ())))
        available_tools = self._available_tools(profile, gateway)
        max_iterations = min(profile.max_iterations, self.max_iterations_cap)
        observations: list[AgentObservation] = []
        tool_call_count = 0
        sequence = len(trace_sink)

        def emit(
            iteration: int,
            event_type: Literal[
                "decision",
                "tool_call",
                "observation",
                "completion",
                "model_error",
                "iteration_limit",
            ],
            **fields: Any,
        ) -> None:
            nonlocal sequence
            sequence += 1
            event = AgentTraceEvent(
                run_id=run_id,
                node_id=node_id,
                agent_id=profile.agent_id,
                agent_version=profile.version,
                sequence=sequence,
                iteration=iteration,
                attempt=attempt,
                event_type=event_type,
                **fields,
            )
            redacted = redactor.redact(event.model_dump(mode="json"))
            # These are schema-validated integer usage counts, not credential strings.
            # Keep cost/usage telemetry useful while all other secret-like fields remain redacted.
            for metric in ("input_tokens", "output_tokens"):
                value = getattr(event, metric)
                if value is None or isinstance(value, int):
                    redacted[metric] = value
            trace_sink.append(redacted)

        for iteration in range(1, max_iterations + 1):
            request = AgentRequest(
                instruction=instruction,
                input_json=input_json,
                iteration=iteration,
                observations=tuple(observations),
                available_tools=available_tools,
            )
            try:
                response = await agent.invoke(request)
            except Exception as exc:
                emit(
                    iteration,
                    "model_error",
                    action_summary="Agent adapter failed while producing a decision.",
                    error_type=type(exc).__name__,
                    error_summary=redactor.redact(
                        getattr(exc, "safe_summary", "Agent adapter error.")
                    ),
                    provider_error_code=getattr(exc, "provider_code", None),
                    provider_status=redactor.redact(getattr(exc, "provider_status", None)),
                    provider_message=redactor.redact(getattr(exc, "provider_message", None)),
                    validation_errors=redactor.redact(getattr(exc, "validation_errors", None)),
                )
                raise AgentExecutionError(f"agent adapter failed ({type(exc).__name__})") from None

            if isinstance(response, str):
                emit(
                    iteration,
                    "decision",
                    action_summary="Legacy agent response treated as a final answer.",
                )
                final_output: Any = response
                self._validate_final(final_output, profile, iteration, emit)
                emit(
                    iteration,
                    "completion",
                    action_summary="Agent completed the task.",
                    output=final_output,
                )
                return final_output

            try:
                turn = AgentTurn.model_validate(response)
            except Exception as exc:
                emit(
                    iteration,
                    "model_error",
                    action_summary="Agent returned an invalid structured decision.",
                    error_type=type(exc).__name__,
                    error_summary="Invalid agent turn schema.",
                )
                raise AgentExecutionError("agent returned an invalid structured decision") from None

            emit(
                iteration,
                "decision",
                action_summary=turn.action_summary,
                **self._telemetry_fields(turn.telemetry),
            )
            if turn.action == "final":
                self._validate_final(turn.final_output, profile, iteration, emit)
                emit(
                    iteration,
                    "completion",
                    action_summary="Agent completed the task.",
                    output=turn.final_output,
                )
                return turn.final_output

            for call_index, tool_call in enumerate(turn.tool_calls, start=1):
                tool_call_count += 1
                call_id = f"{node_id}:{iteration}:{call_index}"
                arguments = tool_call.arguments
                rejection = self._call_rejection(
                    tool_call.tool_id,
                    arguments,
                    profile,
                    call_index,
                    tool_call_count,
                )
                safe_arguments = self._safe_json_object(arguments)
                if rejection is not None:
                    emit(
                        iteration,
                        "tool_call",
                        action_summary=(
                            "Tool call rejected by the agent capability or resource limit."
                        ),
                        call_id=call_id,
                        tool_id=tool_call.tool_id,
                        arguments=safe_arguments,
                        accepted=False,
                        success=False,
                        error_type="ToolCallRejected",
                        error_summary=rejection,
                    )
                    observation = AgentObservation(
                        call_id=call_id,
                        tool_id=tool_call.tool_id,
                        iteration=iteration,
                        arguments=arguments,
                        provider_call_id=tool_call.call_id,
                        continuation=turn.continuation,
                        success=False,
                        error=rejection,
                    )
                    observations.append(observation)
                    emit(
                        iteration,
                        "observation",
                        action_summary="Rejected tool request was returned to the agent.",
                        call_id=call_id,
                        tool_id=tool_call.tool_id,
                        success=False,
                        error_type="ToolCallRejected",
                        error_summary=rejection,
                    )
                    continue

                try:
                    if gateway is None:
                        raise ToolDenied("Tool gateway is not configured.")
                    output = await gateway.invoke(
                        node_id=node_id,
                        tool_id=tool_call.tool_id,
                        arguments=arguments,
                        profile_allowed=profile.allowed_tools,
                        attempt=attempt,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    error_summary = (
                        str(exc)
                        if isinstance(exc, ToolDenied)
                        else "Tool invocation or output validation failed."
                    )
                    emit(
                        iteration,
                        "tool_call",
                        action_summary="Tool call was rejected or failed at the gateway.",
                        call_id=call_id,
                        tool_id=tool_call.tool_id,
                        arguments=safe_arguments,
                        accepted=False,
                        success=False,
                        error_type=type(exc).__name__,
                        error_summary=error_summary,
                    )
                    observation = AgentObservation(
                        call_id=call_id,
                        tool_id=tool_call.tool_id,
                        iteration=iteration,
                        arguments=arguments,
                        provider_call_id=tool_call.call_id,
                        continuation=turn.continuation,
                        success=False,
                        error=(
                            error_summary
                            if isinstance(exc, ToolDenied)
                            else f"{error_summary} ({type(exc).__name__})"
                        ),
                    )
                    observations.append(observation)
                    emit(
                        iteration,
                        "observation",
                        action_summary="Tool failure was returned to the agent for recovery.",
                        call_id=call_id,
                        tool_id=tool_call.tool_id,
                        success=False,
                        error_type=type(exc).__name__,
                        error_summary=error_summary,
                    )
                    continue

                observation = AgentObservation(
                    call_id=call_id,
                    tool_id=tool_call.tool_id,
                    iteration=iteration,
                    arguments=arguments,
                    provider_call_id=tool_call.call_id,
                    continuation=turn.continuation,
                    success=True,
                    output=redactor.redact(output),
                )
                observations.append(observation)
                emit(
                    iteration,
                    "tool_call",
                    action_summary="Authorized tool call completed at the gateway.",
                    call_id=call_id,
                    tool_id=tool_call.tool_id,
                    arguments=arguments,
                    accepted=True,
                    success=True,
                )
                emit(
                    iteration,
                    "observation",
                    action_summary="Tool result was returned to the agent.",
                    call_id=call_id,
                    tool_id=tool_call.tool_id,
                    output=output,
                    success=True,
                )

        emit(
            max_iterations,
            "iteration_limit",
            action_summary="Agent did not produce a final answer before the iteration limit.",
            error_type="MaxIterationsExceeded",
            error_summary=f"Maximum agent iterations reached ({max_iterations}).",
        )
        raise AgentExecutionError(f"maximum agent iterations reached ({max_iterations})")

    @staticmethod
    def record_node_outcome(
        trace_sink: list[dict[str, Any]],
        *,
        run_id: str,
        node_id: str,
        profile: AgentProfile,
        attempt: int,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        last_iteration = max((event.get("iteration", 1) for event in trace_sink), default=1)
        event = AgentTraceEvent(
            run_id=run_id,
            node_id=node_id,
            agent_id=profile.agent_id,
            agent_version=profile.version,
            sequence=len(trace_sink) + 1,
            iteration=last_iteration,
            attempt=attempt,
            event_type="node_outcome",
            action_summary=(
                "Agent node completed its output validation." if success else "Agent node failed."
            ),
            success=success,
            error_type=error_type,
            error_summary=None if success else "Agent node execution failed.",
        )
        redacted = SecretRedactor().redact(event.model_dump(mode="json"))
        # Preserve validated usage telemetry, including null values, as in emit().
        for metric in ("input_tokens", "output_tokens"):
            value = getattr(event, metric)
            if value is None or isinstance(value, int):
                redacted[metric] = value
        trace_sink.append(redacted)

    def _available_tools(
        self, profile: AgentProfile, gateway: ToolGatewaySession | None
    ) -> tuple[AgentToolSpec, ...]:
        allowed = set(gateway.allowed_tool_ids(profile.allowed_tools)) if gateway else set()
        return tuple(
            AgentToolSpec(
                tool_id=tool_id,
                description=self.tool_registry[tool_id].definition.description,
                input_schema=self.tool_registry[tool_id].definition.model_dump(mode="json")[
                    "input_schema"
                ],
            )
            for tool_id in profile.allowed_tools
            if tool_id in allowed
        )

    @staticmethod
    def _telemetry_fields(telemetry: AgentTelemetry | None) -> dict[str, Any]:
        if telemetry is None:
            return {}
        return telemetry.model_dump(mode="json")

    def _call_rejection(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        profile: AgentProfile,
        call_index: int,
        total_calls: int,
    ) -> str | None:
        if call_index > self.max_tool_calls_per_iteration:
            return "Per-iteration tool-call limit reached."
        if total_calls > self.max_total_tool_calls:
            return "Per-agent total tool-call limit reached."
        try:
            json.dumps(arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            return "Tool arguments are not JSON-serializable."
        return None

    @staticmethod
    def _safe_json_object(value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return {"_invalid": "non_json_arguments"}
        return value

    @staticmethod
    def _validate_final(
        output: Any,
        profile: AgentProfile,
        iteration: int,
        emit: Callable[..., None],
    ) -> None:
        try:
            validate(
                instance=output,
                schema=profile.model_dump(mode="json")["output_schema"],
            )
            json.dumps(output, ensure_ascii=False)
        except Exception as exc:
            emit(
                iteration,
                "model_error",
                action_summary="Agent final output failed schema validation.",
                error_type=type(exc).__name__,
                error_summary="Invalid agent final output.",
                validation_errors=_final_output_validation_errors(exc, output),
            )
            raise AgentExecutionError("agent final output failed schema validation") from None


def _final_output_validation_errors(error: Exception, output: Any) -> list[dict[str, str]]:
    if not isinstance(error, JsonSchemaValidationError):
        return [
            {
                "field": "$",
                "error_type": "serialization_error",
                "expected": "JSON-serializable output",
                "actual_type": type(output).__name__,
            }
        ]
    path = [str(part) for part in error.absolute_path]
    validator = str(error.validator or "schema")
    expected = "valid output"
    if validator == "type":
        expected = str(error.validator_value)
    elif validator == "required":
        required = error.validator_value
        instance = error.instance
        missing = (
            [name for name in required if name not in instance]
            if isinstance(required, (list, tuple)) and isinstance(instance, Mapping)
            else []
        )
        if missing:
            path.append(str(missing[0]))
        expected = "required field"
    elif validator == "enum":
        expected = "one of the configured allowed values"
    else:
        expected = f"{validator} constraint"
    return [
        {
            "field": ".".join(path) if path else "$",
            "error_type": validator,
            "expected": expected,
            "actual_type": type(error.instance).__name__,
        }
    ]
