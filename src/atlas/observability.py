"""Structured run events and content-addressed local artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    sha256: str
    size_bytes: int = Field(ge=0)
    media_type: str = "application/json"


class ArtifactStore(Protocol):
    async def put(self, data: bytes, *, media_type: str) -> ArtifactReference: ...

    async def get(self, reference: ArtifactReference) -> bytes: ...


class LocalArtifactStore:
    """Local SHA-256-addressed blobs, written atomically with owner-only permissions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    async def put(self, data: bytes, *, media_type: str) -> ArtifactReference:
        digest = hashlib.sha256(data).hexdigest()
        destination = self.root / digest[:2] / digest[2:]
        await asyncio.to_thread(self._write_atomic, destination, data)
        return ArtifactReference(
            artifact_id=digest,
            sha256=digest,
            size_bytes=len(data),
            media_type=media_type,
        )

    async def get(self, reference: ArtifactReference) -> bytes:
        if reference.artifact_id != reference.sha256:
            raise ValueError("artifact reference identity does not match its digest")
        path = self.root / reference.sha256[:2] / reference.sha256[2:]
        data = await asyncio.to_thread(path.read_bytes)
        if hashlib.sha256(data).hexdigest() != reference.sha256:
            raise ValueError("artifact content digest mismatch")
        return data

    @staticmethod
    def _write_atomic(destination: Path, data: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".atlas-")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class RunEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    plan_hash: str
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    node_id: str | None = None
    tool_id: str | None = None
    status: str | None = None
    attempt: int | None = None
    duration_ms: float | None = None
    artifact_refs: tuple[ArtifactReference, ...] = ()
    attributes: dict[str, Any] = Field(default_factory=dict)


class RunEventStore(Protocol):
    async def append(self, event: RunEvent) -> None: ...

    async def list_run(self, run_id: str) -> list[RunEvent]: ...

    async def list_recent(self, limit: int = 100) -> list[RunEvent]: ...


class SQLiteRunEventStore:
    """Append-only event index; payload blobs live in the independent artifact store."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = asyncio.Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS run_events ("
                "sequence INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, "
                "event_json TEXT NOT NULL)"
            )

    async def append(self, event: RunEvent) -> None:
        payload = event.model_dump_json()
        async with self._lock:
            await asyncio.to_thread(self._insert, event.run_id, payload)

    def _insert(self, run_id: str, payload: str) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO run_events(run_id, event_json) VALUES (?, ?)", (run_id, payload)
            )

    async def list_run(self, run_id: str) -> list[RunEvent]:
        async with self._lock:
            return await asyncio.to_thread(self._read, run_id)

    async def list_recent(self, limit: int = 100) -> list[RunEvent]:
        if not 1 <= limit <= 1000:
            raise ValueError("event query limit must be between 1 and 1000")
        async with self._lock:
            return await asyncio.to_thread(self._read_recent, limit)

    def _read(self, run_id: str) -> list[RunEvent]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                "SELECT event_json FROM run_events WHERE run_id=? ORDER BY sequence", (run_id,)
            ).fetchall()
        return [RunEvent.model_validate_json(row[0]) for row in rows]

    def _read_recent(self, limit: int) -> list[RunEvent]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                "SELECT event_json FROM run_events ORDER BY sequence DESC LIMIT ?", (limit,)
            ).fetchall()
        return [RunEvent.model_validate_json(row[0]) for row in reversed(rows)]


class SpanFactory(Protocol):
    def span(self, name: str, attributes: dict[str, str | int | float | bool]) -> Any: ...


class OpenTelemetrySpanFactory:
    """Small adapter over OpenTelemetry API; configure exporters via the SDK extra."""

    def span(self, name: str, attributes: dict[str, str | int | float | bool]) -> Any:
        try:
            trace = importlib.import_module("opentelemetry.trace")
        except ImportError:
            from contextlib import nullcontext

            return nullcontext()
        tracer = trace.get_tracer("atlas.runtime")
        return tracer.start_as_current_span(name, attributes=attributes)


def json_artifact(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
