import asyncio
import logging

from atlas.agents import AgentRequest, FakeAgent
from atlas.config import JsonFormatter, Settings
from atlas.tools import FakeTool


def test_fake_adapters_are_deterministic() -> None:
    agent = FakeAgent('{"ok":true}')
    assert asyncio.run(agent.invoke(AgentRequest("ignored", "{}"))) == '{"ok":true}'
    tool = FakeTool({"ok": True})
    assert asyncio.run(tool.invoke({"anything": 1})) == {"ok": True}


def test_settings_reject_invalid_log_level(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_LOG_LEVEL", "VERBOSE")
    try:
        Settings.from_environment()
    except ValueError as exc:
        assert "standard logging level" in str(exc)
    else:
        raise AssertionError("invalid log level was accepted")


def test_json_formatter_emits_structured_record() -> None:
    record = logging.LogRecord("atlas", logging.INFO, "", 0, "ready", (), None)
    assert (
        JsonFormatter().format(record) == '{"level": "INFO", "logger": "atlas", "message": "ready"}'
    )
