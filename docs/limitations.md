# Current limitations

- Local event, evaluation, audit, and harness registries use SQLite; artifact payloads use local files. These are single-host adapters without encryption, replication, migrations, or host-admin tamper resistance.
- Replay re-executes agent adapters while replaying recorded tool outputs. It is not a deterministic replay of provider/model behavior.
- Workflow plans do not support conditional execution or human-approval nodes. Policy approvals are available only at the gateway seam.
- Evaluation reports support deterministic outcomes and an optional rubric adapter. There is no bundled LLM judge or production benchmark execution service.
- The fixture benchmark and evidence-gated improvement demo are illustrative; their results are not claims about model quality or production performance.
- Optimizer proposals currently cover instruction variant, approved skills, existing tool preferences, verification-step selection, and bounded inference tokens. They cannot mutate authorization, sandbox, approval, evaluator, or global budget controls.
- `TemporaryWorkspaceProvider` creates fresh fixture copies per evaluation, but is not a security sandbox. Real repository runners must execute commands in a restricted container and apply comparable environment/budget limits.
- CLI fixture evaluation and optimization require the explicit `--fixture-only` flag. Real workflows must supply an `EvaluationRunner` in application code.
- The dashboard is local and read-only. It has no authentication and should bind to loopback unless protected by a trusted network boundary.
- Docker Compose configuration parses, but the image build was blocked when Buildx tried to write activity state under the protected user Docker configuration directory. A screen-recorded demo was not captured; `docs/demo-script.md` gives the reproducible recording sequence.
- Gemini is the only real provider adapter in this milestone. Token cost is estimated only when the API returns usage and deployment-specific input/output rates are configured; estimates are not billing records. Free-tier quota/model availability depends on Google's account and current terms. No distributed worker, production approval service, or online promotion path is included.
- The FastAPI service is a single-process prototype. Run tasks and concurrency state are in memory; restarting the process does not automatically requeue/resume active runs. Multiple Uvicorn workers/replicas are unsupported with the current task registry and local SQLite/filesystem stores.
- API authentication is disabled by default. An authenticator interface can be injected, but no identity provider, token verification, user/tenant authorization, or API rate limiting is bundled. Keep the default service local and trusted.
