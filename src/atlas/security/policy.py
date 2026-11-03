"""Operator-owned tool authorization and append-only local audit records."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from jsonschema import ValidationError as SchemaValidationError
from jsonschema import validate
from pydantic import BaseModel, ConfigDict, Field

from atlas.observability import (
    ArtifactReference,
    ArtifactStore,
    RunEvent,
    RunEventStore,
)
from atlas.plans.ir import ToolRiskClass
from atlas.runtime.backend import ToolRegistration


class ToolPolicy(BaseModel):
    """Trusted backend configuration. Plans and agents cannot modify these ceilings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_tool_ids: frozenset[str] = frozenset()
    allowed_capabilities: frozenset[str] = frozenset()
    workspace_root: str | None = None
    allow_network: bool = False
    max_invocations_per_run: int = Field(default=32, ge=1, le=10000)
    max_concurrent_invocations: int = Field(default=4, ge=1, le=256)
    max_tool_timeout_seconds: float = Field(default=30, gt=0, le=3600)
    max_input_bytes: int = Field(default=1_000_000, ge=1)
    max_output_bytes: int = Field(default=1_000_000, ge=1)
    require_approval_for: frozenset[ToolRiskClass] = frozenset(
        {ToolRiskClass.SIDE_EFFECTING, ToolRiskClass.DESTRUCTIVE}
    )


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    action_hash: str
    run_id: str
    node_id: str
    tool_id: str
    expires_at: datetime


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    action_hash: str
    approved: bool
    approver_id: str
    expires_at: datetime


class ApprovalProvider(Protocol):
    async def approve(self, request: ApprovalRequest) -> ApprovalDecision: ...


class DenyApprovalProvider:
    async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(
            request_id=request.request_id,
            action_hash=request.action_hash,
            approved=False,
            approver_id="system:deny-by-default",
            expires_at=request.expires_at,
        )


class ToolDenied(PermissionError):
    """Tool call rejected by gateway policy with a safe reason."""


class ReplayMiss(RuntimeError):
    """No recorded tool result matches this replay request."""


class SQLiteToolAuditStore:
    """Durable insert-only audit log. No raw tool arguments or outputs are stored."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS tool_audit ("
                "sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_json TEXT NOT NULL)"
            )

    async def append(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, sort_keys=True, separators=(",", ":"))
        async with self._lock:
            await asyncio.to_thread(self._insert, payload)

    def _insert(self, payload: str) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("INSERT INTO tool_audit(event_json) VALUES (?)", (payload,))

    async def list_events(self) -> list[dict[str, Any]]:
        async with self._lock:
            return await asyncio.to_thread(self._read)

    def _read(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT event_json FROM tool_audit ORDER BY sequence"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]


class ToolGatewaySession:
    """Per-run gateway session shared by tool nodes and agent ReAct calls."""

    def __init__(
        self,
        *,
        run_id: str,
        plan_hash: str = "",
        tool_registry: Mapping[str, ToolRegistration],
        policy: ToolPolicy,
        audit_store: SQLiteToolAuditStore,
        approval_provider: ApprovalProvider | None = None,
        event_store: RunEventStore | None = None,
        artifact_store: ArtifactStore | None = None,
        replay_from_run_id: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.plan_hash = plan_hash
        self.tool_registry = tool_registry
        self.policy = policy
        self.audit_store = audit_store
        self.approval_provider = approval_provider or DenyApprovalProvider()
        self.event_store = event_store
        self.artifact_store = artifact_store
        self.replay_from_run_id = replay_from_run_id
        self._replay_offsets: dict[tuple[str, str], int] = defaultdict(int)
        self._replay_input_matches: bool | None = None
        self._semaphore = asyncio.Semaphore(policy.max_concurrent_invocations)
        self._invocations = 0
        self._counter_lock = asyncio.Lock()
        self._workspace = Path(policy.workspace_root).resolve() if policy.workspace_root else None

    def allowed_tool_ids(self, profile_allowed: tuple[str, ...] | None = None) -> tuple[str, ...]:
        profile = set(profile_allowed or ())
        return tuple(
            sorted(
                tool_id
                for tool_id in self.policy.allowed_tool_ids
                if tool_id in self.tool_registry and (profile_allowed is None or tool_id in profile)
            )
        )

    async def invoke(
        self,
        *,
        node_id: str,
        tool_id: str,
        arguments: dict[str, Any],
        required_capabilities: tuple[str, ...] = (),
        profile_allowed: tuple[str, ...] | None = None,
        attempt: int | None = None,
    ) -> dict[str, Any]:
        registration = self.tool_registry.get(tool_id)
        definition = registration.definition if registration else None
        reason: str | None = None
        if definition is not None:
            if definition.tool_id != tool_id:
                reason = reason or "Tool registry identity mismatch."
        else:
            reason = reason or "Tool is not registered by the runtime."
        if definition is not None and tool_id not in self.policy.allowed_tool_ids:
            reason = "Tool is not in the operator allowlist."
        elif (
            definition is not None
            and profile_allowed is not None
            and tool_id not in profile_allowed
        ):
            reason = "Tool is not in this agent profile's allowlist."
        elif definition is not None and not set(definition.capabilities).issubset(
            self.policy.allowed_capabilities
        ):
            reason = "Tool requires a capability not granted by operator policy."
        elif definition is not None and not set(required_capabilities).issubset(
            definition.capabilities
        ):
            reason = "Required capability is not declared by the tool."
        elif (
            definition is not None and definition.requires_network and not self.policy.allow_network
        ):
            reason = "Network access is denied by operator policy."
        try:
            encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":")).encode()
        except (TypeError, ValueError):
            encoded = b""
            reason = reason or "Tool arguments are not JSON-serializable."
        if reason is None and len(encoded) > self.policy.max_input_bytes:
            reason = "Tool input exceeds the operator byte limit."
        if reason is None and definition is not None:
            try:
                validate(instance=arguments, schema=self._plain(definition.input_schema))
            except SchemaValidationError:
                reason = "Tool arguments failed input schema validation."
        if reason is None and definition is not None and definition.workspace_path_argument:
            reason = self._check_workspace_path(arguments, definition.workspace_path_argument)
        async with self._counter_lock:
            if reason is None and self._invocations >= self.policy.max_invocations_per_run:
                reason = "Per-run tool invocation limit reached."
            if reason is None:
                self._invocations += 1
        digest = hashlib.sha256(encoded).hexdigest()
        decision = "denied" if reason else "allowed"
        if (
            reason is None
            and definition is not None
            and (
                definition.requires_approval
                or definition.risk_class in self.policy.require_approval_for
            )
        ):
            request = ApprovalRequest(
                request_id=str(uuid4()),
                action_hash=digest,
                run_id=self.run_id,
                node_id=node_id,
                tool_id=tool_id,
                expires_at=datetime.now(UTC) + timedelta(minutes=2),
            )
            approval = await self.approval_provider.approve(request)
            valid = (
                approval.approved
                and approval.request_id == request.request_id
                and approval.action_hash == digest
                and bool(approval.approver_id.strip())
                and approval.expires_at.tzinfo is not None
                and approval.expires_at > datetime.now(UTC)
                and approval.expires_at <= request.expires_at
            )
            decision = "allowed" if valid else "denied"
            if not valid:
                reason = "Required approval was denied or invalid."
        base_event = {
            "event_id": str(uuid4()),
            "call_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "node_id": node_id,
            "tool_id": tool_id,
            "tool_version": definition.version if definition else None,
            "attempt": attempt,
            "decision": decision,
            "reason": reason,
            "input_sha256": digest,
            "output_sha256": None,
        }
        await self.audit_store.append(base_event)
        if reason is not None:
            await self._emit_event(
                event_type="tool.denied",
                node_id=node_id,
                tool_id=tool_id,
                status="denied",
                attempt=attempt,
                attributes={
                    "call_id": base_event["call_id"],
                    "reason": reason,
                    "input_sha256": digest,
                },
            )
            raise ToolDenied(reason)
        assert registration is not None and definition is not None
        timeout = min(definition.timeout_seconds, self.policy.max_tool_timeout_seconds)
        started = time.perf_counter()
        input_ref = (
            await self.artifact_store.put(encoded, media_type="application/json")
            if self.artifact_store
            else None
        )
        await self._emit_event(
            event_type="tool.started",
            node_id=node_id,
            tool_id=tool_id,
            status="started",
            attempt=attempt,
            artifact_refs=(input_ref,) if input_ref else (),
            attributes={
                "call_id": base_event["call_id"],
                "input_sha256": digest,
                "decision": "allowed",
            },
        )
        try:
            if self.replay_from_run_id is not None:
                output, self._replay_input_matches = await self._recorded_output(
                    source_run_id=self.replay_from_run_id,
                    node_id=node_id,
                    tool_id=tool_id,
                    input_digest=digest,
                )
            else:
                async with self._semaphore:
                    with self._span(tool_id, node_id):
                        output = await asyncio.wait_for(
                            registration.adapter.invoke(arguments), timeout
                        )
            validate(instance=output, schema=self._plain(definition.output_schema))
            output_bytes = json.dumps(output, ensure_ascii=False, separators=(",", ":")).encode()
            if len(output_bytes) > self.policy.max_output_bytes:
                raise ValueError("Tool output exceeds the operator byte limit.")
            outcome = "succeeded"
            output_hash = hashlib.sha256(output_bytes).hexdigest()
            output_ref = (
                await self.artifact_store.put(output_bytes, media_type="application/json")
                if self.artifact_store
                else None
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            outcome = "failed"
            output_hash = None
            await self.audit_store.append(
                {
                    **base_event,
                    "event_id": str(uuid4()),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "outcome": outcome,
                    "error_type": type(exc).__name__,
                }
            )
            await self._emit_event(
                event_type="tool.failed",
                node_id=node_id,
                tool_id=tool_id,
                status="failed",
                attempt=attempt,
                duration_ms=(time.perf_counter() - started) * 1000,
                artifact_refs=(input_ref,) if input_ref else (),
                attributes={
                    "call_id": base_event["call_id"],
                    "input_sha256": digest,
                    "error_type": type(exc).__name__,
                },
            )
            raise
        await self.audit_store.append(
            {
                **base_event,
                "event_id": str(uuid4()),
                "timestamp": datetime.now(UTC).isoformat(),
                "outcome": outcome,
                "output_sha256": output_hash,
            }
        )
        refs = tuple(ref for ref in (input_ref, output_ref) if ref is not None)
        await self._emit_event(
            event_type="tool.replayed" if self.replay_from_run_id else "tool.completed",
            node_id=node_id,
            tool_id=tool_id,
            status="succeeded",
            attempt=attempt,
            duration_ms=(time.perf_counter() - started) * 1000,
            artifact_refs=refs,
            attributes={
                "input_sha256": digest,
                "output_sha256": output_hash,
                "call_id": base_event["call_id"],
                "replay_source_run_id": self.replay_from_run_id,
                "replay_input_matches": self._replay_input_matches,
            },
        )
        return output

    async def _recorded_output(
        self, *, source_run_id: str, node_id: str, tool_id: str, input_digest: str
    ) -> tuple[dict[str, Any], bool]:
        if self.event_store is None or self.artifact_store is None:
            raise ReplayMiss("Replay requires event and artifact stores.")
        events = await self.event_store.list_run(source_run_id)
        calls = [
            event
            for event in events
            if event.event_type == "tool.completed"
            and event.node_id == node_id
            and event.tool_id == tool_id
            and len(event.artifact_refs) >= 2
        ]
        key = (node_id, tool_id)
        offset = self._replay_offsets[key]
        if offset >= len(calls):
            raise ReplayMiss("No recorded tool output remains for this replay call.")
        recorded = calls[offset]
        self._replay_offsets[key] += 1
        output_ref: ArtifactReference = recorded.artifact_refs[-1]
        output = json.loads(await self.artifact_store.get(output_ref))
        return output, recorded.attributes.get("input_sha256") == input_digest

    async def _emit_event(
        self,
        *,
        event_type: str,
        node_id: str,
        tool_id: str,
        status: str,
        attributes: dict[str, Any],
        artifact_refs: tuple[ArtifactReference, ...] = (),
        duration_ms: float | None = None,
        attempt: int | None = None,
    ) -> None:
        if self.event_store is None:
            return
        await self.event_store.append(
            RunEvent(
                run_id=self.run_id,
                plan_hash=self.plan_hash,
                event_type=event_type,
                node_id=node_id,
                tool_id=tool_id,
                status=status,
                attempt=attempt,
                duration_ms=duration_ms,
                artifact_refs=artifact_refs,
                attributes=attributes,
            )
        )

    @staticmethod
    def _span(tool_id: str, node_id: str) -> Any:
        from atlas.observability import OpenTelemetrySpanFactory

        return OpenTelemetrySpanFactory().span(
            "atlas.tool", {"atlas.tool_id": tool_id, "atlas.node_id": node_id}
        )

    def _check_workspace_path(self, arguments: dict[str, Any], key: str) -> str | None:
        if self._workspace is None:
            return "Workspace-scoped tool denied because no workspace is configured."
        raw = arguments.get(key)
        if not isinstance(raw, str) or not raw:
            return "Workspace path argument must be a non-empty string."
        candidate = (self._workspace / raw).resolve()
        try:
            candidate.relative_to(self._workspace)
        except ValueError:
            return "Workspace path escapes the configured workspace."
        return None

    @staticmethod
    def _plain(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: ToolGatewaySession._plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [ToolGatewaySession._plain(v) for v in value]
        return value
