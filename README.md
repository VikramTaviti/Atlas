# ATLAS — Adaptive Agent Execution & Reliability Engine

ATLAS is a local-first agent execution harness with a typed workflow compiler, LangGraph adapter, bounded ReAct agent nodes, deny-by-default tool gateway, structured evidence, recorded-tool replay, benchmark evaluation, and constrained harness experiments.

## Quick start

Requirements: Python 3.12+, `pip`, and (for Docker Compose) Docker Compose v2.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,langgraph,observability]'
pytest
ruff check .
mypy src/atlas
```

Run the dynamic graph and ReAct demos without provider credentials:

```bash
python examples/runtime_fake.py
python examples/react_fake.py
python examples/evidence_gated_improvement.py
```

The final demo is explicitly illustrative. `--approve` simulates a human approval for the fixture experiment and demonstrates rollback; it is not an authenticated approval service.

## Persistent API service

Install the API extra and start the local FastAPI service:

```bash
python -m pip install -e '.[api,langgraph,observability]'
uvicorn atlas.api.app:app --host 127.0.0.1 --port 8000
```

The service exposes `GET /health/live`, `GET /health/ready`, `POST /v1/runs`,
`GET /v1/runs/{run_id}`, `GET /v1/runs/{run_id}/events`, and
`POST /v1/runs/{run_id}/cancel`. Applications can separately enable `POST /v1/tasks`
by injecting a provider-independent planner and trusted agent templates into
`create_app`; the stock `atlas.api.app:app` does not configure task planning.
`POST /v1/tasks` accepts task text/context and returns `202` with the generated Plan
and run ID. `POST /v1/runs` accepts a validated plan plus JSON
inputs and returns `202` with a run ID. Only agents/tools configured by trusted server
application code can be referenced; callers cannot register adapters or change policy.

A tool-free local smoke request (the default service has no tool or agent adapters):

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/runs \
  -H 'content-type: application/json' \
  -d '{"plan":{"plan_id":"api-smoke","task_id":"api-smoke-task","nodes":[{"node_id":"start","kind":"join","name":"Start"},{"node_id":"finish","kind":"output","name":"Finish","depends_on":["start"]}]},"inputs":{"message":"hello"}}'
```

Use the returned `run_id` with `GET /v1/runs/{run_id}` and
`GET /v1/runs/{run_id}/events` to follow completion and inspect events.

### Live Gemini task-intake example

The separate example composes `GeminiPlanner`, three trusted agent roles, the existing
workspace read tool, and an exact-argv Docker test runner. It makes live Gemini calls
and requires Docker for `workspace.test`:

```bash
python -m pip install -e '.[api,gemini,langgraph,observability]'
set -a
source .env
set +a
uvicorn examples.gemini_task_intake:app --host 127.0.0.1 --port 8000
```

Submit a repository-readiness task and save the accepted plan/run ID:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/tasks \
  -H 'content-type: application/json' \\
  -d '{"task":"Inspect the repository release readiness. Have one agent inspect architecture and tests, another run the configured dependency check, and a reviewer summarize evidence and blockers."}'
```

The response includes the generated `plan` and `run_id`. Poll
`GET /v1/runs/{run_id}` until terminal, then inspect
`GET /v1/runs/{run_id}/events`. This example is an opt-in live demonstration; normal
CI uses mocked planners/agents and does not call Gemini.

Authentication is deliberately **disabled by default** for local development. An
application can inject an `Authenticator` and set `require_authentication=True`; this
interface does not itself provide identity verification. Do not expose the default
service to an untrusted network. The bundled service has no real agent/provider or
tool registry configured, so plans requiring those must be supplied by a deployment
that configures trusted adapters and `ToolPolicy` in its app factory.

Run submission executes asynchronously in the API process with configurable
concurrency and total admitted-run bounds; excess admission receives `429` with a
short `Retry-After`. SQLite checkpoints/events/audits and filesystem
artifacts persist beneath `ATLAS_DATA_DIR` (default `./atlas-data`). On shutdown, active
tasks are cancelled and awaited. A process restart does not automatically resume
admitted runs; multiple API workers do not share task state and are unsupported with
the current local stores.

## Docker Compose

```bash
docker compose up --build -d
curl http://127.0.0.1:${ATLAS_PORT:-8000}/health/ready
```

This runs the persistent API on the host loopback interface and saves local
checkpoints/events/artifacts under `./atlas-data`. Set `ATLAS_PORT` to change the host
and container port. The health check
probes readiness. See [limitations](docs/limitations.md) before using local artifact
storage with sensitive data.

The finite credential-free graph demo remains available with
`docker compose run --rm atlas python examples/runtime_fake.py`.

## Plans, events, and replay

```bash
atlas validate-plan examples/plans/fanout_join.json
atlas inspect-run --events-db /tmp/atlas-react-example.sqlite.events.sqlite --run-id RUN_ID
atlas replay-tools --events-db /tmp/atlas-react-example.sqlite.events.sqlite \
  --artifacts /tmp/atlas-react-example.sqlite.artifacts --run-id RUN_ID
atlas dashboard --events-db /tmp/atlas-react-example.sqlite.events.sqlite
```

The dashboard is read-only and binds to loopback by default. `replay-tools` prints the recorded exchange transcript without executing tools. To replay a workflow, call `await backend.replay(plan, source_run_id=..., replay_run_id=...)`. It restores saved inputs and tool outputs but re-executes agent adapters, which the divergence report marks as nondeterministic.

## Evaluation and constrained optimization

The CLI fixture mode is explicitly labelled illustrative; application code supplies an `EvaluationRunner` for real workflows.

```bash
atlas compare --suite examples/benchmarks/smoke.json \
  --baseline-id baseline --candidate-id candidate --trials 3 --fixture-only

python examples/evidence_gated_improvement.py
```

The optimizer accepts only typed patches to instruction variant, approved skills, already-authorized tool preferences, verification steps, and inference tokens below an operator ceiling. An eligible experiment still requires separate promotion approval. The development CLI flow is:

```bash
atlas optimize --suite examples/benchmarks/smoke.json \
  --baseline examples/harnesses/baseline.json \
  --patch examples/harnesses/add_verification.json \
  --approved-skill repository-inspection --approved-skill verification \
  --authorized-tool workspace.read --authorized-tool workspace.test \
  --inference-budget-ceiling 4096 --fixture-only

atlas promote --experiment-id EXPERIMENT_ID --experiment-hash EXPERIMENT_HASH \
  --approver-id operator:NAME
atlas rollback --harness-id repo-maintenance --version 1 --approver-id operator:NAME
```

CLI approver IDs are caller supplied and not authenticated. Production approval must be wired to a trusted identity provider.

## Telemetry and provider adapters

Install the optional `observability` extra and configure an OpenTelemetry provider/exporter in the application to export run/node/tool spans. Gemini is the first real provider adapter; install the optional extra and run the gateway-mediated ReAct example:

```bash
python -m pip install -e '.[gemini,langgraph]'
# Configure GEMINI_API_KEY or GOOGLE_API_KEY in the environment.
python examples/gemini_react.py
```

The example makes a live provider request and may consume quota. Normal CI uses mocked provider responses; the separate live smoke test is opt-in. Environment settings, Compose secret wiring, usage/cost metadata, and future OpenAI/Anthropic adapter guidance are in [provider adapter guidance](docs/provider-adapters.md). Credentials must never enter plans or harness patches.

## Architecture and project status

Milestones 0–7 are implemented as a local-first MVP. Storage uses SQLite/filesystem adapters behind interfaces; the system is not a distributed worker service. See [architecture](docs/architecture.md), [threat model](docs/threat-model.md), [evaluation methodology](docs/evaluation-methodology.md), [limitations](docs/limitations.md), [release process](docs/release-process.md), [roadmap](docs/implementation-roadmap.md), and [ADRs](docs/adr/). `ATLAS_CONTEXT.md` is the engineering progress source of truth.

The reproducible demo script is documented, but a screen recording was not captured in this development environment. Docker Compose and release automation are present; their live Docker/GitHub execution remains a maintainer release-preparation check.
