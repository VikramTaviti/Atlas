# ADR 0005: Local event index, content-addressed artifacts, and recorded-tool replay

## Status

Accepted for Milestone 4.

## Context

Milestone 3 audit events store hashes, not raw payloads, so they are intentionally insufficient to reconstruct tool calls. Milestone 4 needs run/node/tool correlation, artifacts, inspection, and replay while preserving the no-side-effect replay default.

## Decision

- Add a framework-neutral `RunEventStore` and `ArtifactStore` protocol.
- The local implementation uses append-only SQLite event rows and SHA-256-addressed files with atomic writes and owner-only file mode. A local SQLite/file implementation is used instead of PostgreSQL/object storage to preserve credential-free local development; protocols retain the adapter seam.
- Run start artifacts contain the canonical plan and input; node events point at input/output artifacts. Tool events point at argument/result artifacts and include hashes, decision, status, duration, tool/node/run IDs. Phase 3 audit records remain separate and payload-free.
- `LangGraphExecutionBackend.replay` restores saved inputs and reuses matching recorded tool outputs by source node/tool call order. Agent adapters are re-executed; replay reports say so and compare per-node/run outputs and tool input hashes. Replay never invokes registered tool adapters.
- OpenTelemetry spans are created through the API for runs, nodes, and tool calls. The `observability` extra provides API and SDK; applications configure exporters and providers.
- CLI `inspect-run` prints correlated structured events. `replay-tools` prints the stored exchange transcript without executing tools.

## Consequences

- Replay is deterministic for tool responses only. Re-executed model/provider agents may diverge; the report marks the execution mode and identifies output/input divergence.
- Local artifacts may contain sensitive run/tool data required for replay. They use owner-only permissions but are not encrypted; deployments must choose storage/retention controls appropriate to their data.
- SQLite and local files are single-host prototypes. PostgreSQL/object storage, multi-worker event ordering, retention, artifact encryption, and exporter configuration remain deployment concerns.
- Tool replay matches calls by recorded node/tool order. Extra calls fail with a replay miss; changed inputs still receive the recorded response but are flagged in the report.
