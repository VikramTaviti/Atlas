# ATLAS engineering context

Persistent engineering source of truth. Update this file at each milestone and when a material architecture decision changes.

## Current state — 2026-11-03

### Decisions

- Python 3.12+; Pydantic v2 frozen contracts; JSON Schema Draft 2020-12 via `jsonschema`.
- Framework-neutral plan/runtime/provider/evaluation contracts; LangGraph remains an optional execution adapter.
- Trusted application code owns agent/tool registries, policy, evaluator configuration, and approval identity. Plans and optimizer patches cannot grant permissions or change operator ceilings.
- Tools are deny-by-default through one gateway. Phase 3 audit data remains separate from Phase 4 replay artifacts.
- Local SQLite/filesystem implementations satisfy event, artifact, evaluation, and harness registry interfaces. PostgreSQL/object storage are deployment extensions.
- ADRs 0001–0010 record material decisions.
- ADR 0010 records the Gemini adapter boundary and manual Atlas-mediated function calling.

### Implemented

- **Milestones 0–1:** package/tooling, provider-neutral contracts, immutable Plan IR, deterministic compiler/validation/hash, fixtures, CLI, docs and CI.
- **Milestone 2:** dynamic LangGraph DAG execution, SQLite checkpoints, bounded concurrency/retries/timeouts/cancellation/resume, and fake graph example.
- **ReAct extension:** bounded Reason→Act→Observe loop inside agent nodes, tool observations, limits and structured traces without private chain-of-thought.
- **Milestone 3:** centralized deny-by-default gateway, policy/capability/network/schema/budget/workspace/approval enforcement, hash-only audit, redaction, workspace reader and restricted Docker test runner.
- **Milestone 4:** correlated run/node/tool events, local content-addressed artifacts, optional OpenTelemetry spans, event inspection, recorded-tool replay and divergence reports.
- **Milestone 5:** versioned benchmark suites/cases, deterministic output/node/test checks, optional versioned rubric adapter, paired baseline/candidate metrics, uncertainty, hard safety gates, and persisted suites/reports.
- **Milestone 6:** failure classification, strict mutable-field harness patches, pre-approved skills/authorized tool preferences/token ceilings, paired experiments, disposable per-trial workspace adapter, persistent scoped/expiring experience, explicit hash-bound promotion and versioned rollback.
- **Milestone 7 implementation:** CLI for evaluation/optimization/promotion/rollback, read-only dashboard, Docker Compose, dynamic graph and evidence-gated fixture demos, benchmark methodology/report, provider guidance, limitations, CI and tag-release workflow.
- **Milestone 8:** persistent FastAPI app with lifespan-managed compiler/backend/event/artifact stores, typed health/run/event/cancel endpoints, `202 Accepted` asynchronous run admission, bounded in-process concurrency and admitted-run count, run request artifacts, graceful cancellation/shutdown, injectable authentication interface, structured JSON application logs, and persistent Docker Compose/Uvicorn deployment. Requests cannot add tools or modify operator policy.
- **Milestone 9:** optional `GeminiAgent` uses Google's official async GenAI SDK while implementing the existing provider-independent `Agent` protocol. It parses typed `AgentTurn`s, maps manual function calls to Atlas tool IDs without executing them, and sends ReAct observations into later prompts. Configurable model/temperature/timeout/output tokens, environment or secret-file credentials, sanitized errors, generic model/latency/token/cost telemetry, and API shutdown client cleanup are included. `FakeAgent` remains unchanged.
- **Gemini model compatibility fix:** default model is `gemini-3.5-flash-lite`; `ATLAS_GEMINI_MODEL` takes precedence and `ATLAS_DEFAULT_MODEL` is honored as a fallback. Node outcome traces preserve null token telemetry instead of displaying `[REDACTED]`.
- **Gemini ReAct history fix:** provider-neutral tool observations now retain iteration and arguments. Gemini requests reconstruct chronological user/model conversation turns, with each model function call matched by a user function response. The system instruction directs Gemini to finish when observations suffice; Atlas iteration bounds remain active.
- **Gemini 3 continuation fix:** preserve the SDK's original model Content in memory, including `functionCall.id` and opaque thought signatures, and echo the exact call ID in its `functionResponse`. Structured SDK HTTP code/status/sanitized message are retained in provider diagnostics and trace fields. These continuation bytes are not traced or logged.
- **Gemini final response mapping:** nonempty plain text is wrapped in the common typed `AgentTurn` final response. Structured JSON remains strictly validated; empty text, malformed JSON, and missing/invalid fields expose safe Pydantic field/type/expectation diagnostics. Final values rejected by the agent JSON Schema expose field, failed constraint, expected and actual types without echoing the value.
- **Task-intake vertical slice:** provider-neutral `Planner`, strict bounded `WorkflowProposal`, and operator-owned `PlannerLimits` are in `plans/planner.py`. `GeminiPlanner` uses Gemini structured JSON output, and `AgentTemplateRegistry`/`AgentFactory` resolve only trusted roles and produce one adapter/profile per node. New `POST /v1/tasks` accepts task/context separately from `POST /v1/runs`, validates hard limits, constructs ordinary Plan IR, compiles it, and admits execution via `RunService` on an ephemeral LangGraph backend with the existing gateway and trusted tool/policy registries. The response contains the generated Plan. Ephemeral adapters close after success, failure, and cancellation; the planner closes on app shutdown. Added live example composition in `examples/gemini_task_intake.py`, architecture notes, ADR 0011, and README usage.

### Verification

Environment: CPython 3.12.16 and project dependencies in `/private/tmp/atlas-venv`.

- `/private/tmp/atlas-venv/bin/ruff check .` — **passed**.
- `/private/tmp/atlas-venv/bin/mypy src/atlas` — **passed** (27 source files).
- `/private/tmp/atlas-venv/bin/python -m pytest -q` — **68 passed**.
- After Milestone 8: `/private/tmp/atlas-venv/bin/ruff format .`, `ruff check .`, `mypy src/atlas`, and `python -m pytest -q` — **74 passed**.
- Milestone 9 validation: Google GenAI SDK `1.75.0` installed in the temporary environment; its `GenerateContentConfig` validated Atlas function declarations and automatic-calling disablement offline. `ruff check .` and `mypy src/atlas` passed; `pytest -q` reported **86 passed, 1 skipped**. The skipped test is the opt-in live Gemini smoke; no live provider request was made.
- Gemini conversation regression validation: Ruff passed, mypy passed (27 source files), and pytest reported **88 passed, 1 skipped**. The skipped test is the opt-in live Gemini smoke; no live provider request was made.
- Gemini 3 continuation/error regression validation: Ruff passed, mypy passed (27 source files), and pytest reported **89 passed, 1 skipped**. The skipped test is the opt-in live Gemini smoke; no live provider request was made.
- Gemini final response mapping regression validation: Ruff passed, mypy passed (27 source files), and pytest reported **94 passed, 1 skipped**. The skipped test is the opt-in live Gemini smoke; no live provider request was made.
- Task-intake slice validation: Ruff passed, mypy passed (29 source files), pytest reported **100 passed, 1 skipped**, and `git diff --check` passed. New tests cover bounded structured planner output, malformed/extra planner fields, unknown roles/tools, per-node isolation, gateway-mediated tool use, and cancellation cleanup.
- A live task-intake attempt was blocked before producing a plan: outbound Gemini DNS/network access is restricted in the sandbox. An escalated attempt was rejected by automatic approval review because it would send repository-derived contents to Google's API without explicit approval for that external data transfer. No live generated plan or live multi-agent run is claimed. Docker daemon access is also denied by the current local socket permissions, so the example's `workspace.test` tool cannot run in this environment.
- `python examples/runtime_fake.py`, `python examples/react_fake.py`, and `python examples/evidence_gated_improvement.py --approve` — **passed**.
- Fixture `atlas compare` — **6 paired samples**, baseline success **0.50**, candidate success **1.00**, zero policy violations. Values are illustrative fixture outcomes only.
- CLI `optimize` → `promote` → `rollback` — **passed**; rollback created version 3.
- CLI `inspect-run`, `replay-tools`, and dashboard rendering helpers — **passed**.
- `git diff --check` — **passed**.
- `docker compose config --quiet` passed. Image build was blocked because Buildx attempted to write activity state under the protected user Docker configuration directory. Live Uvicorn socket serving, remote CI/tag release, and screen recording were not run; FastAPI HTTP behavior is covered through TestClient.
- FastAPI TestClient exercises health, 202 admission/completion, persisted events/request artifacts, validation/404, tool gateway authorization, cancellation, and injected authentication. A live Uvicorn socket and Docker Compose container were not run here.

Milestones 0–6 and the Milestone 7 implementation are complete. The required short screen-recorded demo was not captured; `docs/demo-script.md` has the reproducible command sequence.

### Known limitations

- Local SQLite event/audit/evaluation/harness stores and filesystem artifacts are single-host prototypes without encryption, replication, migrations, or host-admin tamper resistance.
- Artifacts contain raw inputs/results for replay. Protect them and define retention before production use.
- Replay re-executes agent adapters; only tool outputs are replayed. Tool exchanges follow recorded node/tool order and report changed inputs/output divergence.
- OpenTelemetry exporter/provider configuration is application-owned. Gemini is available as an optional adapter; OpenAI/Anthropic integrations remain future work.
- Temporary evaluation workspaces are disposable copies, not command sandboxes. Real repository evaluations must use restricted containers and comparable baseline/candidate environments.
- The CLI optimizer's approver ID is caller-supplied and unauthenticated. Production promotion requires an authenticated approval service.
- Fixture benchmark results are illustrative and not claims about model quality or production performance. The single-sample interval is `[0, 1]`; larger stochastic benchmarks need more trials and suitable paired statistical tests.
- Docker Compose and the release workflow are present but Docker image build was blocked by workspace sandbox restrictions on the external Buildx state directory; the actual screen recording remains to be captured.
- Conditional graph execution, production PostgreSQL/object storage, distributed workers, token/cost inference accounting beyond per-decision Gemini telemetry, additional model providers, and online promotion are not implemented.
- API authentication defaults to unauthenticated local development. The app exposes an authenticator interface, but no identity provider, tenant authorization, quotas, or rate limiting is included. Do not expose the default service to untrusted networks.
- Task-intake is opt-in: applications must inject a planner and trusted templates into `create_app`; the default module app remains plan-submission-only. The Gemini example requires network access, credentials, and Docker. The proposal only chooses topology, role IDs and task descriptions; agent results currently aggregate as the normal output node envelope, without semantic synthesis/evaluation.
- The API task registry/semaphore are process-local. Restart does not automatically resume queued/running runs; multiple Uvicorn workers/replicas are unsupported. Accepted plan/input payloads, events, checkpoints, audit data, and artifacts are persisted locally for inspection/recovery development.

### Next

Milestones 8 and 9 plus the initial task-intake vertical slice are implemented. Remaining validation: run the live Gemini task-intake demo after approval for repository-derived data to be sent to Google's API, with working network access; run the Docker-backed verifier where the Docker daemon socket is available; and build/run Compose plus exercise a live Uvicorn socket. The Milestone 7 demo recording also remains uncaptured.
