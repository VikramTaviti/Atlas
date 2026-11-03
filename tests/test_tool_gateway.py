from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from atlas.builtin_tools import DockerTestRunner
from atlas.plans.ir import ToolDefinition, ToolRiskClass
from atlas.runtime.backend import ToolRegistration
from atlas.security.policy import (
    ApprovalDecision,
    ApprovalRequest,
    SQLiteToolAuditStore,
    ToolDenied,
    ToolGatewaySession,
    ToolPolicy,
)
from atlas.security.redaction import SecretRedactor


class ToolFixture:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.calls = 0
        self.result = result or {"value": "ok"}

    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        return self.result


def definition(**overrides: Any) -> ToolDefinition:
    values: dict[str, Any] = {
        "tool_id": "fixture",
        "version": "1",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    }
    values.update(overrides)
    return ToolDefinition(**values)


def session(
    tmp_path: Path,
    tool: ToolFixture,
    *,
    policy: ToolPolicy,
    tool_definition: ToolDefinition | None = None,
    approval: Any = None,
) -> ToolGatewaySession:
    return ToolGatewaySession(
        run_id="run",
        tool_registry={"fixture": ToolRegistration(tool_definition or definition(), tool)},
        policy=policy,
        audit_store=SQLiteToolAuditStore(str(tmp_path / "audit.sqlite")),
        approval_provider=approval,
    )


@pytest.mark.asyncio
async def test_denies_by_default_and_audits_without_invoking(tmp_path: Path) -> None:
    tool = ToolFixture()
    gateway = session(tmp_path, tool, policy=ToolPolicy())
    with pytest.raises(ToolDenied, match="operator allowlist"):
        await gateway.invoke(node_id="node", tool_id="fixture", arguments={})
    events = await gateway.audit_store.list_events()
    assert events[0]["decision"] == "denied"
    assert tool.calls == 0


@pytest.mark.asyncio
async def test_invalid_arguments_rejected_before_adapter_and_audited(tmp_path: Path) -> None:
    tool = ToolFixture()
    gateway = session(tmp_path, tool, policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})))
    with pytest.raises(ToolDenied, match="schema"):
        await gateway.invoke(node_id="node", tool_id="fixture", arguments={"unexpected": 1})
    assert tool.calls == 0
    assert (await gateway.audit_store.list_events())[0]["decision"] == "denied"


@pytest.mark.asyncio
async def test_per_run_budget_denies_excess_calls(tmp_path: Path) -> None:
    tool = ToolFixture()
    gateway = session(
        tmp_path,
        tool,
        policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"}), max_invocations_per_run=1),
    )
    await gateway.invoke(node_id="first", tool_id="fixture", arguments={})
    with pytest.raises(ToolDenied, match="Per-run"):
        await gateway.invoke(node_id="second", tool_id="fixture", arguments={})
    assert tool.calls == 1
    events = await gateway.audit_store.list_events()
    assert [event["decision"] for event in events].count("denied") == 1


@pytest.mark.asyncio
async def test_audit_persists_hashes_without_raw_tool_payloads(tmp_path: Path) -> None:
    tool = ToolFixture({"value": "sensitive-result-9281"})
    gateway = session(
        tmp_path,
        tool,
        policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})),
    )
    await gateway.invoke(node_id="node", tool_id="fixture", arguments={"path": "private.txt"})
    serialized = str(await gateway.audit_store.list_events())
    assert "sensitive-result-9281" not in serialized
    assert "private.txt" not in serialized
    assert "input_sha256" in serialized
    assert "output_sha256" in serialized


@pytest.mark.asyncio
async def test_workspace_traversal_and_symlink_escape_blocked(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret"
    outside.write_text("secret")
    (workspace / "escape").symlink_to(outside)
    tool = ToolFixture()
    definition_with_path = definition(workspace_path_argument="path")
    gateway = session(
        tmp_path,
        tool,
        tool_definition=definition_with_path,
        policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"}), workspace_root=str(workspace)),
    )
    for path in ("../secret", "escape"):
        with pytest.raises(ToolDenied, match="escapes"):
            await gateway.invoke(node_id="node", tool_id="fixture", arguments={"path": path})
    assert tool.calls == 0


class Approve:
    async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(
            request_id=request.request_id,
            action_hash=request.action_hash,
            approved=True,
            approver_id="operator:alice",
            expires_at=request.expires_at,
        )


@pytest.mark.asyncio
async def test_side_effecting_action_requires_bound_approval(tmp_path: Path) -> None:
    tool = ToolFixture()
    definition_with_risk = definition(risk_class=ToolRiskClass.SIDE_EFFECTING)
    policy = ToolPolicy(allowed_tool_ids=frozenset({"fixture"}))
    denied = session(tmp_path, tool, policy=policy, tool_definition=definition_with_risk)
    with pytest.raises(ToolDenied, match="approval"):
        await denied.invoke(node_id="node", tool_id="fixture", arguments={})
    accepted = session(
        tmp_path, tool, policy=policy, tool_definition=definition_with_risk, approval=Approve()
    )
    assert await accepted.invoke(node_id="node", tool_id="fixture", arguments={}) == {"value": "ok"}
    assert tool.calls == 1


def test_redactor_hides_secret_keys_and_values() -> None:
    redactor = SecretRedactor(("known-secret",))
    value = redactor.redact({"api_key": "known-secret", "message": "Bearer abc.def"})
    assert value == {"api_key": "[REDACTED]", "message": "Bearer [REDACTED]"}


@pytest.mark.asyncio
async def test_docker_runner_uses_restricted_argv_and_exact_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import atlas.builtin_tools as builtin

    captured: list[str] = []

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"ok", b""

    async def fake_exec(*argv: str, **kwargs: Any) -> Process:
        del kwargs
        captured.extend(argv)
        return Process()

    monkeypatch.setattr(builtin.asyncio, "create_subprocess_exec", fake_exec)
    runner = DockerTestRunner(
        str(tmp_path), image="atlas-tests:locked", commands=(("pytest", "-q"),)
    )
    result = await runner.invoke({"command": ["pytest", "-q"]})
    assert result["exit_code"] == 0
    assert "--network=none" in captured
    assert "--read-only" in captured
    assert "--cap-drop=ALL" in captured
    assert "--security-opt=no-new-privileges" in captured
    assert "--mount" in captured
    assert "--" not in captured
    captured.clear()
    with pytest.raises(PermissionError, match="allowlist"):
        await runner.invoke({"command": ["sh", "-c", "id"]})
    assert captured == []


def test_plan_does_not_accept_policy_fields() -> None:
    from pydantic import ValidationError

    from atlas.plans.ir import Node, NodeKind, Plan

    with pytest.raises(ValidationError):
        Plan(
            plan_id="p",
            task_id="t",
            nodes=(Node(node_id="n", kind=NodeKind.OUTPUT, name="Output"),),
            allowed_tool_ids=("fixture",),
        )
