# ADR 0002: Use LangGraph behind a framework-neutral execution backend

- **Status:** Accepted
- **Date:** 2026-11-03
- **Decision owners:** ATLAS maintainers

## Context

Milestone 2 needs a bounded async DAG runtime, checkpoints, restart recovery, cancellation, timeout, retry, and a runnable multi-node example. The project owner selected LangGraph for this milestone while requiring Atlas to remain open to alternative execution engines.

## Decision

1. Define a small Atlas-owned `ExecutionBackend` protocol and `ExecutionResult` contract. Callers submit only an already-validated `CompiledPlan`.
2. Implement `LangGraphExecutionBackend` as an adapter of that protocol. Translate the frozen Atlas graph into a LangGraph `StateGraph` at run time; do not expose LangGraph types in the plan IR or framework-neutral protocol.
3. Publish LangGraph and its SQLite checkpointer as the optional `langgraph` extra. The core compiler and backend protocol remain importable without LangGraph installed.
4. Use LangGraph's SQLite checkpointer for local durable checkpoints in this milestone. The adapter accepts a database path, with a distinct `thread_id` per Atlas run. Use a restricted message-pack serializer allowlist because checkpoint contents are a deserialization boundary.
5. Bound graph execution with `min(plan.max_parallelism, backend.max_parallelism)` and LangGraph's supported run concurrency configuration. Reject conditional edges, condition nodes, and human-approval nodes until their deterministic semantics and approval service exist.
6. Retry within Atlas node wrappers, with a bounded attempt count and node deadline. Tool retries are permitted only when the trusted registered definition classifies the operation as `read_only` or `idempotent`; other tools are rejected until an idempotency-key contract and policy gateway exist.
7. Agent and tool adapter protocols are asynchronous so cancellation and timeouts can propagate through cooperative implementations. Adapter implementations remain responsible for lower-level network timeouts and side-effect idempotency.
8. Until Milestone 3 adds the policy gateway and approval/idempotency enforcement, the LangGraph adapter refuses all tools whose risk class is not `read_only` or `idempotent`.

## Alternatives considered

- **LangGraph as the core runtime and plan representation:** rejected; it would couple Atlas's contracts and future optimizer to one engine.
- **Use LangGraph's SQLite checkpointer as the only persistence abstraction:** accepted for this local milestone, but the adapter boundary allows other backends and future persistence providers. This is not the target production persistence choice.
- **Use an in-memory checkpointer only:** rejected because process restart and durable checkpoint recovery are Milestone 2 acceptance requirements.
- **Enable automatic retry for every failed node:** rejected because tools may have side effects; retries must honor trusted tool risk classification.

## Consequences

The Milestone 2 local setup installs the optional LangGraph and SQLite checkpointer dependencies. SQLite checkpoint state is durable but is not a business event log, distributed worker lease, or production audit store. A cancelled task may be re-executed from the beginning of the currently running node; node side effects must therefore be idempotent or protected. The current M2 backend deliberately rejects conditional routing and human approval graphs rather than interpreting expression strings or bypassing approval.

The synchronous fake implementations become async to make cancellation and concurrency behavior testable; integrations should implement the same async protocols. There is no tool authorization gateway yet, so this adapter is intended for deterministic fixtures and trusted read-only/idempotent adapters only.

## Revisit when

The API/persistence milestone adds Postgres business records, distributed workers, run admission transactions, or the condition and approval services. At that point, verify LangGraph checkpoint recovery against plan-version migrations and evaluate other `ExecutionBackend` implementations without changing Atlas IR.
