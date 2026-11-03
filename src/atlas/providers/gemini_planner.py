"""Gemini implementation of the provider-independent bounded Planner protocol."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from pydantic import ValidationError

from atlas.plans.planner import Planner, PlannerRequest, WorkflowProposal
from atlas.providers.gemini import (
    GeminiAgent,
    GeminiAgentConfig,
    GeminiProviderError,
    _read_value,
    _safe_provider_error_details,
)
from atlas.security.redaction import SecretRedactor


class GeminiPlanner:
    """Gemini structured-output planner; it returns graph structure, never capabilities."""

    def __init__(
        self,
        config: GeminiAgentConfig | None = None,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        api_key_file: str | None = None,
    ) -> None:
        self.config = config or GeminiAgentConfig.from_environment()
        self._owned_client = client is None
        self._redactor = SecretRedactor((api_key,) if api_key else ())
        if client is None:
            key = GeminiAgent._resolve_api_key(api_key, api_key_file)
            self._redactor = SecretRedactor((key,))
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
                    f"Gemini planner client initialization failed ({type(exc).__name__})"
                ) from None
        self._client = client
        self._async_client = getattr(client, "aio", client)

    async def plan(self, request: PlannerRequest) -> WorkflowProposal:
        role_catalog = [role.model_dump(mode="json") for role in request.available_roles]
        prompt = json.dumps(
            {
                "task": request.task,
                "context": request.context,
                "trusted_roles": role_catalog,
                "operator_limits": request.limits.model_dump(mode="json"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        system_instruction = (
            "Design a small execution DAG for the supplied task. Return only the schema fields. "
            "Use role_id values exactly from trusted_roles. Never create tools, permissions, "
            "capabilities, credentials, provider settings, or policies. Node descriptions are "
            "bounded task instructions. Edges point from prerequisite source to dependent target. "
            "The graph must be acyclic and fit all operator_limits. Do not include prose outside "
            "the structured response."
        )
        config = {
            "temperature": min(self.config.temperature, 0.2),
            "max_output_tokens": self.config.max_output_tokens,
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": WorkflowProposal.model_json_schema(),
        }
        try:
            generate_content = cast(Any, self._async_client.models.generate_content)
            async with asyncio.timeout(self.config.timeout_seconds):
                response = await generate_content(
                    model=self.config.model,
                    contents=prompt,
                    config=config,
                )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise GeminiProviderError("Gemini planning request timed out") from None
        except Exception as exc:
            code, provider_status, message, summary = _safe_provider_error_details(
                exc, self._redactor
            )
            raise GeminiProviderError(
                summary,
                provider_code=code,
                provider_status=provider_status,
                provider_message=message,
            ) from None

        text = _read_value(response, "text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Gemini planner returned empty structured output")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError("Gemini planner returned malformed JSON") from None
        try:
            return WorkflowProposal.model_validate(payload)
        except ValidationError:
            raise ValueError("Gemini planner output failed the bounded proposal schema") from None

    async def aclose(self) -> None:
        if not self._owned_client:
            return
        async_client = getattr(self._client, "aio", None)
        async_close = getattr(async_client, "aclose", None)
        if callable(async_close):
            await async_close()
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def planner_protocol(planner: GeminiPlanner) -> Planner:
    """Typed helper so application composition can expose the provider-neutral protocol."""
    return planner
