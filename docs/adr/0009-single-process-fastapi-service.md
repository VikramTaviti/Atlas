# ADR 0009: Single-process FastAPI service for persistent run admission

## Status

Accepted for Milestone 8.

## Context

The local deployment previously ran a finite fixture script. Atlas already exposes
framework-neutral plan compilation, a LangGraph execution adapter, a per-run policy
gateway, SQLite event/checkpoint/audit stores, and filesystem artifacts. The service
needs asynchronous HTTP admission without introducing a distributed queue or replacing
those contracts.

## Decision

- Add a FastAPI application factory and Uvicorn entrypoint.
- Configure agent/tool adapters and `ToolPolicy` in trusted application code. Requests
  may submit a plan and JSON inputs but cannot register capabilities or modify policy.
- Compile plans against the server registry before accepting a run.
- Return `202 Accepted` after writing a `run.queued` event and scheduling an in-process
  asyncio task. Execute only through `LangGraphExecutionBackend` and its existing tool
  gateway.
- Persist the accepted Plan and inputs as an artifact referenced from `run.queued`, so
  accepted request evidence survives even if the process exits before execution begins.
- Bound concurrent and total admitted runs within the API process. Persist run events, checkpoints,
  policy audit, and artifacts in the existing single-host stores.
- Expose an injectable authenticator interface. Local default authentication is
  disabled and does not make a security claim.
- Keep liveness/readiness probes accessible without credentials so orchestrators can
  determine process/storage health; all run routes remain subject to configured auth.
- Retain CLI/example entrypoints. Compose starts Uvicorn and probes readiness.

## Consequences

- The app remains single-process/single-instance. In-memory tasks and concurrency
  control are not shared between Uvicorn workers or replicas.
- A restart does not automatically recover queued/running runs. Persisted events and
  LangGraph checkpoints are retained for inspection and future recovery support.
- Shutdown cancels active tasks and awaits cleanup; cancellation during a tool call
  depends on that adapter's cancellation behavior.
- SQLite and local artifact files remain appropriate only for a single-instance
  prototype with local storage protections.
- Real identity verification, authorization scopes, and network exposure controls must
  be supplied by a deployment; this milestone provides only the integration seam.

## Alternatives considered

- **Keep running a finite example:** does not provide a persistent service.
- **Add a distributed queue and workers now:** violates the milestone scope and adds
  coordination/storage dependencies before single-instance behavior is established.
- **Let requests supply tools or permissions:** rejected because untrusted API input
  must not expand trusted runtime capabilities.
