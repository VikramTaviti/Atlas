"""Opt-in live provider smoke test; skipped in standard CI."""

from __future__ import annotations

import asyncio
import os

import pytest

from atlas.agents import AgentRequest, AgentTurn
from atlas.providers.gemini import GeminiAgent

_has_key = bool(
    os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GEMINI_API_KEY_FILE")
)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ATLAS_GEMINI_SMOKE") != "1" or not _has_key,
    reason="set ATLAS_GEMINI_SMOKE=1 and configure Gemini credentials to call the live API",
)
def test_live_gemini_returns_a_structured_final_turn() -> None:
    async def run() -> AgentTurn:
        agent = GeminiAgent()
        try:
            result = await agent.invoke(
                AgentRequest(
                    instruction="Return the requested value.",
                    input_json='{"value":"atlas-smoke-ok"}',
                )
            )
            assert isinstance(result, AgentTurn)
            assert result.action == "final"
            return result
        finally:
            await agent.aclose()

    turn = asyncio.run(run())
    assert turn.final_output is not None
    assert turn.telemetry is not None
    assert turn.telemetry.provider == "google_gemini"
