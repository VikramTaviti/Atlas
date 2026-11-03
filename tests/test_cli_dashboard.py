from __future__ import annotations

from atlas.cli import _render_dashboard_html, _render_run_json
from atlas.observability import RunEvent


def test_dashboard_renders_recent_run_ids_as_escaped_read_only_links() -> None:
    events = [
        RunEvent(run_id="run-<script>", plan_hash="hash", event_type="run.started"),
        RunEvent(run_id="run-2", plan_hash="hash", event_type="run.started"),
    ]
    page = _render_dashboard_html(events)
    assert "run-&lt;script&gt;" in page
    assert "run-%3Cscript%3E" in page
    assert "run-2" in page
    assert "Read-only" in page


def test_dashboard_run_endpoint_serializes_structured_events() -> None:
    event = RunEvent(run_id="correlation-id", plan_hash="plan-hash", event_type="node.failed")
    payload = _render_run_json([event])
    assert '"run_id": "correlation-id"' in payload
    assert '"event_type": "node.failed"' in payload
