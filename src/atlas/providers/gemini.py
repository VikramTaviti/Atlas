"""Google Gemini adapter for the provider-independent Atlas Agent protocol."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from atlas.agents import AgentRequest, AgentResponse, AgentTelemetry, AgentToolCall, AgentTurn
from atlas.security.redaction import SecretRedactor


class GeminiProviderError(RuntimeError):
    """Sanitized provider failure; never includes request credentials or raw provider text."""

    def __init__(
        self,
        safe_summary: str,
        *,
        provider_code: int | None = None,
        provider_status: str | None = None,
        provider_message: str | None = None,
        validation_errors: list[dict[str, str]] | None = None,
    ) -> None:
        self.safe_summary = safe_summary
        self.provider_code = provider_code
        self.provider_status = provider_status
        self.provider_message = provider_message
        self.validation_errors = validation_errors
        super().__init__(safe_summary)


@dataclass(frozen=True)
class GeminiAgentConfig:
    model: str = "gemini-3.5-flash-lite"
    temperature: float = 0.2
    timeout_seconds: float = 45.0
    max_output_tokens: int = 2048
    input_price_per_million_tokens_usd: float | None = None
    output_price_per_million_tokens_usd: float | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Gemini model must not be empty")
        if not 0 <= self.temperature <= 2:
            raise ValueError("Gemini temperature must be between 0 and 2")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Gemini timeout_seconds must be positive")
        if self.max_output_tokens < 1:
            raise ValueError("Gemini max_output_tokens must be positive")
        for name, value in (
            ("input_price_per_million_tokens_usd", self.input_price_per_million_tokens_usd),
            ("output_price_per_million_tokens_usd", self.output_price_per_million_tokens_usd),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must not be negative")

    @classmethod
    def from_environment(cls) -> GeminiAgentConfig:
        """Read non-secret generation settings from ATLAS_GEMINI_* environment values."""
        try:
            return cls(
                model=os.environ.get(
                    "ATLAS_GEMINI_MODEL",
                    os.environ.get("ATLAS_DEFAULT_MODEL", "gemini-3.5-flash-lite"),
                ),
                temperature=float(os.environ.get("ATLAS_GEMINI_TEMPERATURE", "0.2")),
                timeout_seconds=float(os.environ.get("ATLAS_GEMINI_TIMEOUT_SECONDS", "45")),
                max_output_tokens=int(os.environ.get("ATLAS_GEMINI_MAX_OUTPUT_TOKENS", "2048")),
                input_price_per_million_tokens_usd=_optional_float(
                    "ATLAS_GEMINI_INPUT_PRICE_PER_MILLION_USD"
                ),
                output_price_per_million_tokens_usd=_optional_float(
                    "ATLAS_GEMINI_OUTPUT_PRICE_PER_MILLION_USD"
                ),
            )
        except ValueError:
            raise ValueError("invalid ATLAS_GEMINI_* generation configuration") from None


def _optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    return float(raw) if raw else None


class GeminiAgent:
    """Async Gemini model adapter that returns decisions; Atlas executes all tools.

    The official Google Gen AI SDK is imported lazily and is an optional dependency.
    Supplying ``client`` enables provider-mocked tests without installing that SDK.
    """

    def __init__(
        self,
        config: GeminiAgentConfig | None = None,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        api_key_file: str | Path | None = None,
    ) -> None:
        self.config = config or GeminiAgentConfig.from_environment()
        self._owned_client = client is None
        self._secret_values = tuple(
            value
            for value in (
                api_key,
                os.environ.get("GOOGLE_API_KEY"),
                os.environ.get("GEMINI_API_KEY"),
            )
            if value
        )
        self._redactor = SecretRedactor(self._secret_values)
        if client is None:
            key = self._resolve_api_key(api_key, api_key_file)
            self._secret_values = (*self._secret_values, key)
            self._redactor = SecretRedactor((*self._secret_values, key))
            try:
                from google import genai
                from google.genai import types
            except ImportError:
                raise RuntimeError(
                    "Gemini SDK is missing; install atlas-agent-runtime[gemini]"
                ) from None
            try:
                client = genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(
                        timeout=max(1, int(self.config.timeout_seconds * 1000))
                    ),
                )
            except Exception as exc:
                raise GeminiProviderError(
                    f"Gemini client initialization failed ({type(exc).__name__})"
                ) from None
        self._client = client
        self._async_client = getattr(client, "aio", client)

    @property
    def redaction_secrets(self) -> tuple[str, ...]:
        """Exact credential values for the surrounding ReAct trace redactor."""
        return self._secret_values

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        function_names = {
            f"atlas_tool_{index}": tool.tool_id
            for index, tool in enumerate(request.available_tools)
        }
        tools = [
            {
                "function_declarations": [
                    {
                        "name": function_name,
                        "description": next(
                            tool.description
                            for tool in request.available_tools
                            if tool.tool_id == tool_id
                        )
                        or f"Request execution of Atlas tool {tool_id}.",
                        "parameters_json_schema": next(
                            tool.input_schema
                            for tool in request.available_tools
                            if tool.tool_id == tool_id
                        ),
                    }
                ]
            }
            for function_name, tool_id in function_names.items()
        ]
        system_instruction = (
            "You are an Atlas agent. Do not reveal private chain-of-thought or hidden reasoning. "
            "Give only a concise operational action_summary. Use a declared function when an "
            "Atlas tool is needed; do not claim to have executed a tool. After receiving tool "
            "results, use those observations to decide whether the task is complete. If they "
            "provide sufficient information, do not call the same tool again. Return the final "
            "user-facing answer as concise plain text, not JSON. Otherwise request only the "
            "additional tool calls needed."
        )
        if tools:
            system_instruction += (
                " For an action, call one or more declared functions and let Atlas execute them."
            )
        try:
            input_value = json.loads(request.input_json)
        except json.JSONDecodeError:
            input_value = request.input_json
        contents = self._conversation_contents(request, input_value, function_names)
        config: dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            "system_instruction": system_instruction,
            "automatic_function_calling": {"disable": True},
        }
        if tools:
            config["tools"] = tools

        started = time.perf_counter()
        try:
            generate_content = cast(Any, self._async_client.models.generate_content)
            async with asyncio.timeout(self.config.timeout_seconds):
                response = await generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=config,
                )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise GeminiProviderError("Gemini request timed out") from None
        except Exception as exc:
            provider_code, provider_status, provider_message, summary = (
                _safe_provider_error_details(exc, self._redactor)
            )
            raise GeminiProviderError(
                summary,
                provider_code=provider_code,
                provider_status=provider_status,
                provider_message=provider_message,
            ) from None
        latency_ms = (time.perf_counter() - started) * 1000
        telemetry = self._telemetry(response, latency_ms)

        function_calls = _read_value(response, "function_calls")
        if function_calls:
            calls: list[AgentToolCall] = []
            for function_call in function_calls:
                function_name = _read_value(function_call, "name")
                arguments = _read_value(function_call, "args") or {}
                if not isinstance(function_name, str) or not isinstance(arguments, Mapping):
                    raise GeminiProviderError("Gemini returned an invalid function call")
                calls.append(
                    AgentToolCall(
                        tool_id=function_names.get(function_name, function_name),
                        arguments=self._redactor.redact(dict(arguments)),
                        call_id=_read_value(function_call, "id"),
                    )
                )
            if not calls:
                raise GeminiProviderError("Gemini returned an empty function-call response")
            try:
                return AgentTurn(
                    action="act",
                    action_summary=f"Gemini requested {len(calls)} Atlas tool call(s).",
                    tool_calls=tuple(calls),
                    continuation=_model_continuation(response),
                    telemetry=telemetry,
                )
            except ValidationError:
                raise GeminiProviderError(
                    "Gemini returned invalid function-call arguments"
                ) from None

        text = _read_value(response, "text")
        if not isinstance(text, str) or not text.strip():
            raise GeminiProviderError("Gemini returned neither a tool call nor a decision")
        cleaned_text = _strip_json_fence(text)
        try:
            payload = json.loads(cleaned_text)
        except json.JSONDecodeError:
            if cleaned_text.lstrip().startswith(("{", "[")):
                raise GeminiProviderError(
                    "Gemini returned malformed structured output.",
                    validation_errors=[
                        {
                            "field": "$",
                            "error_type": "json_invalid",
                            "expected": "valid JSON AgentTurn object",
                        }
                    ],
                ) from None
            final_text = self._redactor.redact(text.strip())
            return AgentTurn(
                action="final",
                action_summary="Gemini returned a final text response.",
                final_output=final_text,
                telemetry=telemetry,
            )
        try:
            turn = AgentTurn.model_validate(payload)
        except ValidationError as exc:
            raise GeminiProviderError(
                "Gemini structured response failed AgentTurn validation.",
                validation_errors=_safe_validation_errors(exc, self._redactor),
            ) from None
        safe_turn = self._redactor.redact(turn.model_dump(mode="json", exclude={"telemetry"}))
        return AgentTurn.model_validate({**safe_turn, "telemetry": telemetry})

    @staticmethod
    def _conversation_contents(
        request: AgentRequest,
        input_value: Any,
        function_names: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        """Rebuild Gemini's required model-call/tool-response turns from Atlas observations."""
        prompt = json.dumps(
            {"instruction": request.instruction, "input": input_value},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": prompt}]}]
        names_by_tool_id = {
            tool_id: function_name for function_name, tool_id in function_names.items()
        }
        observations_by_iteration: dict[int, list[Any]] = {}
        for observation in request.observations:
            observations_by_iteration.setdefault(observation.iteration, []).append(observation)

        for observations in observations_by_iteration.values():
            function_call_parts: list[dict[str, Any]] = []
            function_response_parts: list[dict[str, Any]] = []
            for observation in observations:
                name = names_by_tool_id.get(observation.tool_id, observation.tool_id)
                result = (
                    {"output": observation.output}
                    if observation.success
                    else {"error": observation.error or "Tool call failed."}
                )
                function_response: dict[str, Any] = {"name": name, "response": result}
                if observation.provider_call_id:
                    function_response["id"] = observation.provider_call_id
                function_response_parts.append({"function_response": function_response})
            continuation = observations[0].continuation
            if continuation is not None:
                # Preserve Gemini's original model Content, including call IDs and thought
                # signatures. These opaque provider bytes stay in the in-memory agent turn.
                contents.append(continuation)
            else:
                for observation in observations:
                    name = names_by_tool_id.get(observation.tool_id, observation.tool_id)
                    function_call: dict[str, Any] = {
                        "name": name,
                        "args": observation.arguments,
                    }
                    if observation.provider_call_id:
                        function_call["id"] = observation.provider_call_id
                    function_call_parts.append({"function_call": function_call})
                contents.append({"role": "model", "parts": function_call_parts})
            contents.append({"role": "user", "parts": function_response_parts})
        return contents

    async def aclose(self) -> None:
        """Close SDK resources when this adapter created its client."""
        if not self._owned_client:
            return
        async_client = getattr(self._client, "aio", None)
        async_close = getattr(async_client, "aclose", None)
        if callable(async_close):
            await async_close()
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def _telemetry(self, response: Any, latency_ms: float) -> AgentTelemetry:
        usage = _read_value(response, "usage_metadata")
        input_tokens = _as_int(_read_value(usage, "prompt_token_count"))
        output_tokens = _as_int(_read_value(usage, "candidates_token_count"))
        estimated_cost: float | None = None
        input_rate = self.config.input_price_per_million_tokens_usd
        output_rate = self.config.output_price_per_million_tokens_usd
        if (
            input_tokens is not None
            and output_tokens is not None
            and input_rate is not None
            and output_rate is not None
        ):
            estimated_cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
        return AgentTelemetry(
            provider="google_gemini",
            model_id=self.config.model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
        )

    @staticmethod
    def _resolve_api_key(api_key: str | None, api_key_file: str | Path | None) -> str:
        if api_key and api_key.strip():
            return api_key.strip()
        env_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        secret_path = (
            Path(api_key_file)
            if api_key_file is not None
            else Path(os.environ.get("GEMINI_API_KEY_FILE", "/run/secrets/gemini_api_key"))
        )
        try:
            file_key = secret_path.read_text(encoding="utf-8").strip()
        except OSError:
            file_key = ""
        if file_key:
            return file_key
        raise ValueError("Gemini API key is not configured")


def _read_value(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _safe_provider_error_details(
    error: Exception, redactor: SecretRedactor
) -> tuple[int | None, str | None, str | None, str]:
    """Preserve structured SDK diagnostics while redacting secrets and bounding message size."""
    code = _read_value(error, "code")
    provider_code = code if isinstance(code, int) else None
    raw_status = _read_value(error, "status")
    provider_status = _sanitize_provider_text(raw_status, redactor)
    provider_message = _sanitize_provider_text(_read_value(error, "message"), redactor)
    if provider_code is None:
        return (
            None,
            provider_status,
            provider_message,
            f"Gemini request failed ({type(error).__name__}); check network access and retry.",
        )
    label = f"HTTP {provider_code}"
    if provider_status:
        label += f" {provider_status}"
    summary = f"Gemini API request failed ({label})."
    if provider_message:
        summary += f" {provider_message}"
    return provider_code, provider_status, provider_message, summary


def _sanitize_provider_text(value: Any, redactor: SecretRedactor) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = " ".join(redactor.redact(value).split())
    return cleaned[:500]


def _safe_validation_errors(
    error: ValidationError, redactor: SecretRedactor
) -> list[dict[str, str]]:
    """Expose field/type/expectation only; Pydantic input values may contain user data."""
    details: list[dict[str, str]] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False)[:20]:
        location = item.get("loc", ())
        field = ".".join(str(part) for part in location) if location else "$"
        message = str(redactor.redact(item.get("msg", "Invalid value")))[:200]
        expected = "required field" if message == "Field required" else message
        details.append(
            {
                "field": field[:200],
                "error_type": str(item.get("type", "validation_error"))[:100],
                "expected": expected,
            }
        )
    return details


def _model_continuation(response: Any) -> dict[str, Any] | None:
    """Copy the SDK's exact model Content so Gemini IDs and thought signatures survive."""
    candidates = _read_value(response, "candidates")
    if not isinstance(candidates, (list, tuple)) or not candidates:
        return None
    content = _read_value(candidates[0], "content")
    if content is None:
        return None
    model_dump = getattr(content, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python", exclude_none=True)
    elif isinstance(content, Mapping):
        dumped = dict(content)
    else:
        return None
    if not isinstance(dumped, dict):
        return None
    dumped.setdefault("role", "model")
    return dumped


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).removeprefix("json").strip()
    return stripped
