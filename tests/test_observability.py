from __future__ import annotations

from pathlib import Path

import pytest

from atlas.observability import (
    LocalArtifactStore,
    RunEvent,
    SQLiteRunEventStore,
)


@pytest.mark.asyncio
async def test_artifact_store_round_trip_and_digest_verification(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    reference = await store.put(b'{"value":1}', media_type="application/json")
    assert await store.get(reference) == b'{"value":1}'
    assert reference.size_bytes == 11


@pytest.mark.asyncio
async def test_structured_events_are_correlated_and_ordered(tmp_path: Path) -> None:
    store = SQLiteRunEventStore(tmp_path / "events.sqlite")
    for event_type in ("run.started", "node.started", "node.completed", "run.completed"):
        await store.append(
            RunEvent(run_id="correlation-123", plan_hash="abc", event_type=event_type)
        )
    events = await store.list_run("correlation-123")
    assert [event.event_type for event in events] == [
        "run.started",
        "node.started",
        "node.completed",
        "run.completed",
    ]
    assert {event.run_id for event in events} == {"correlation-123"}
