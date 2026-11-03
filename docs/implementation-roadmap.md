# ATLAS implementation roadmap and engineering backlog

This backlog turns `ATLAS_IMPLEMENTATION_PLAN.md` into reviewable increments. Work is sequenced by dependency; Milestones 0–7 are implemented and Milestone 8 adds the single-instance persistent API service.

## Repository inspection

Initial state: clean `main` branch with an empty `src/` directory and empty `README.md`, `AGENTS.md`, and `ATLAS_CONTEXT.md`. There was no existing application code, test suite, or packaging configuration to preserve. The design brief targets Python 3.12+, Pydantic v2, and local-first development. The available host interpreter during initial work is Python 3.9.6, so it cannot run this project's declared Python 3.12 target.

## Backlog

| ID | Milestone | Work item | Depends on | Acceptance criteria | Status |
|---|---|---|---|---|---|
| M0-1 | 0 | Establish package metadata, Python target, quality tooling, and source/test layout | — | Fresh Python 3.12 environment can install project and dev extras; configuration is explicit | Done |
| M0-2 | 0 | Add provider-independent agent/tool protocols and deterministic fakes | M0-1 | Fake adapters return fixed results without network or credentials | Done |
| M0-3 | 0 | Add basic validated environment config and JSON structured logging | M0-1 | Invalid log levels fail early; emitted log records are JSON | Done |
| M0-4 | 0 | Add architecture, threat model, ADR, roadmap, and initial tree | Repository inspection | Decisions, trust boundaries, risks, and current scope are documented | Done |
| M1-1 | 1 | Define immutable, versioned Pydantic plan, node, edge, agent, and tool contracts | M0-1 | Extra fields rejected; JSON schemas checked; core models immutable | Done |
| M1-2 | 1 | Compile plans against server-owned agent/tool registries | M1-1 | Unknown refs and malformed graph structure rejected deterministically | Done |
| M1-3 | 1 | Enforce graph size, depth, DAG, output, and ready-node concurrency limits | M1-2 | Cycles, dangling links, duplicate IDs, excess limits, and no-output plans fail | Done |
| M1-4 | 1 | Canonical serialization and SHA-256 plan fingerprint | M1-2 | Equivalent IR serializes identically and has stable hash | Done |
| M1-5 | 1 | Sequential and fan-out/join examples, tests, and validation CLI | M1-2 | Examples validate; regression tests cover acceptance and rejection cases | Done |
| M2-1 | 2 | Framework-neutral backend contract and LangGraph adapter; bounded async DAG execution | M1 | Dependencies gate execution; concurrency respects plan/operator limits; fake examples run | Done |
| M2-2 | 2 | Durable SQLite checkpoints, recovery, retries, deadlines, cancellation | M2-1 | Restart and injected-failure tests demonstrate safe recovery and idempotency | Done |
| M2-3 | 2 | Bounded ReAct loop inside individual agent nodes with structured traces | M2-1 | Tool authorization, observations, iteration/timeout limits, failure handling, trace privacy, and tests pass | Done |
| M3-1 | 3 | Deny-by-default policy engine and centralized tool gateway | M2 | Every invocation is authorized and budget checked before execution | Done |
| M3-2 | 3 | Workspace-scoped tools and isolated command runner | M3-1 | Traversal, command injection, and host escape are blocked by tests | Done |
| M4-1 | 4 | Durable events, artifacts, OpenTelemetry, recorded tool replay | M2, M3 | Replay uses recorded responses by default; trace and audit records correlate | Done |
| M5-1 | 5 | Versioned task/case suite and retained evaluation records | M4 | Suite version, task criteria, environment and report provenance persist | Done |
| M5-2 | 5 | Deterministic structural/test evaluators and optional rubric adapter | M5-1 | Deterministic results and qualitative scores/evaluator provenance are separate | Done |
| M5-3 | 5 | Paired baseline/candidate metrics, uncertainty, and hard safety gates | M5-2 | Same cases/trials/environment; sample count/variance and zero-violation gate | Done |
| M6-1 | 6 | Failure classification, typed proposals, mutable-field allowlist | M5 | Unknown/policy fields reject; skill/tool/token ceilings enforced | Done |
| M6-2 | 6 | Paired isolated candidate experiment and persistent scoped experience | M6-1 | Same cases/budgets; fresh workspace adapter available; provenance, scope, confidence, expiry stored | Done |
| M6-3 | 6 | Explicit approval, version promotion, and rollback | M6-2 | Failed gates cannot promote; approval hash-bound; rollback is traceable | Done |
| M7-1 | 7 | CLI inspection, evaluation/optimizer commands, read-only dashboard | M4-M6 | CLI and dashboard expose local evidence without mutation | Done |
| M7-2 | 7 | Docker Compose quick start and end-to-end fixture demos | M2-M6 | Clean checkout runs both demos without provider credentials | Done |
| M7-3 | 7 | Benchmark methodology, limitations, provider guidance and tagged release process | M5-M7 | Claims labeled; CI tests; tag workflow builds release artifacts | Implemented; release not run |
| M7-4 | 7 | Capture and publish the short end-to-end demo recording | M7-2 | Recording matches reproducible fixture demo and labels illustrative results | Not captured in this environment |
| M8-1 | 8 | Persistent FastAPI service, async in-process runs, health/run/event/cancel API | M2-M4 | Typed API, lifecycle, server-owned tools/policy, cancellation, and integration tests | Done |
| M9-1 | 9 | Gemini adapter on Agent protocol with manual function calls and provider telemetry | M2-M4 | Mocked decision/tool/observation tests, safe credentials, optional live smoke test | Done |

All Milestone 0 and 1 items passed their local checks on Python 3.12.16. See `ATLAS_CONTEXT.md` for exact commands and results.

Milestone 2 uses the adapter and SQLite checkpoint decisions in `docs/adr/0002-langgraph-adapter.md`. The ReAct-in-agent-node boundary and trace contract are in `docs/adr/0003-react-within-agent-nodes.md`. Verification is recorded in `ATLAS_CONTEXT.md`.

Milestone 3 gateway contracts and local audit choice are in `docs/adr/0004-centralized-tool-gateway.md`. Its audit database is distinct from LangGraph checkpoint storage.

Milestone 4 adds local event/artifact protocol implementations and recorded-tool replay; see `docs/adr/0005-local-events-artifacts-and-recorded-tool-replay.md`. The local backend is a prototype adapter chosen for this repository's credential-free workflow; PostgreSQL and object-storage adapters remain production extensions.

Milestone 5 evaluation decisions are in `docs/adr/0006-versioned-paired-evaluation.md`; Milestone 6 optimizer gates and versioning are in `docs/adr/0007-constrained-harness-optimization.md`; Milestone 7 CLI/dashboard/release scope is in `docs/adr/0008-local-dx-and-read-only-dashboard.md`.
Milestone 8's single-process HTTP admission and persistence boundary is recorded in `docs/adr/0009-single-process-fastapi-service.md`.
Milestone 9's Gemini adapter and gateway-owned tool-call boundary are recorded in `docs/adr/0010-gemini-agent-adapter.md`.

## Milestone gates

- **Milestone 0:** project skeleton and documentation exist; installation, lint, typing, and tests run in Python 3.12 CI.
- **Milestone 1:** compiler rejects malformed/untrusted plans, freezes accepted plan IR, fingerprints canonical content, and has runnable fixtures plus tests.
- **Milestones 2–4:** runtime, policy, event and replay contracts have tests and runnable examples.
- **Milestone 5:** benchmark suites and reports are versioned; deterministic checks are distinct from rubric results; paired metrics carry uncertainty and hard safety gates.
- **Milestone 6:** optimizer patches cannot change policy; candidate changes require evidence, same-case experiments, all hard gates, explicit approval, and rollback traceability.
- **Milestone 7:** clean-checkout instructions, Docker Compose, two fixture demos, read-only dashboard, CI, tagged release workflow, methodology and limitations are present. The actual screen recording and live Docker/release runs require tools or infrastructure absent from this environment.
- **Milestone 8:** FastAPI API, in-process asynchronous execution, lifecycle, health/run/event/cancel routes, and local Compose deployment are implemented. TestClient integration coverage verifies admission limits, event persistence, tool authorization, cancellation, validation, and auth-interface behavior. Compose config validation passed; image build needs an environment where Buildx can write its external cache state.
- **Milestone 9:** optional official Google GenAI async adapter, manual function-call mapping, observation feedback, provider-neutral usage telemetry, secret/environment setup, mocked tests, and opt-in live smoke test are implemented. The smoke test is skipped in normal CI unless explicitly enabled with credentials.
