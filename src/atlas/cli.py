"""ATLAS command-line tools for validation, evaluation, inspection, and local demos."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from atlas.evaluation import (
    BenchmarkSuite,
    EvaluationObservation,
    SQLiteEvaluationStore,
    compare_harnesses,
    evaluate_suite,
)
from atlas.observability import LocalArtifactStore, SQLiteRunEventStore
from atlas.optimizer import (
    HarnessConfig,
    HarnessPatch,
    PromotionApproval,
    SQLiteHarnessRegistry,
    run_experiment,
)
from atlas.plans.compiler import PlanCompiler
from atlas.security.redaction import SecretRedactor


class _SuiteFixtureRunner:
    """Explicit fixture-only runner. Real workflows supply an EvaluationRunner adapter."""

    async def __call__(self, case: Any, *, harness_id: str, trial: int) -> EvaluationObservation:
        del trial
        fixtures = case.metadata.get("fixture_observations", {})
        key = harness_id.split("@", maxsplit=1)[0]
        if key not in fixtures and "@" in harness_id:
            key = "candidate" if harness_id.endswith("@2") else "baseline"
        if key not in fixtures:
            raise ValueError(
                f"suite case '{case.case_id}' has no fixture observation for harness '{harness_id}'"
            )
        return EvaluationObservation.model_validate(fixtures[key])


def _load_suite(path: Path) -> BenchmarkSuite:
    return BenchmarkSuite.model_validate_json(path.read_text(encoding="utf-8"))


def _print_model(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(prog="atlas")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-plan", help="validate and fingerprint plan JSON")
    validate.add_argument("path", type=Path)

    inspect = subparsers.add_parser("inspect-run", help="inspect structured events for a run")
    inspect.add_argument("--events-db", type=Path, required=True)
    inspect.add_argument("--run-id", required=True)

    replay = subparsers.add_parser(
        "replay-tools", help="print recorded tool exchanges without invoking tools"
    )
    replay.add_argument("--events-db", type=Path, required=True)
    replay.add_argument("--artifacts", type=Path, required=True)
    replay.add_argument("--run-id", required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate one harness on a versioned suite")
    evaluate.add_argument("--suite", type=Path, required=True)
    evaluate.add_argument("--harness-id", required=True)
    evaluate.add_argument("--environment-id", default="local-fixture")
    evaluate.add_argument("--trials", type=int, default=1)
    evaluate.add_argument("--store", type=Path, default=Path("atlas-evaluations.sqlite"))
    evaluate.add_argument("--fixture-only", action="store_true", required=True)

    compare = subparsers.add_parser("compare", help="paired baseline/candidate suite comparison")
    compare.add_argument("--suite", type=Path, required=True)
    compare.add_argument("--baseline-id", required=True)
    compare.add_argument("--candidate-id", required=True)
    compare.add_argument("--environment-id", default="local-fixture")
    compare.add_argument("--trials", type=int, default=1)
    compare.add_argument("--store", type=Path, default=Path("atlas-evaluations.sqlite"))
    compare.add_argument("--fixture-only", action="store_true", required=True)

    optimize = subparsers.add_parser(
        "optimize", help="run a policy-constrained harness experiment (fixture-only CLI mode)"
    )
    optimize.add_argument("--suite", type=Path, required=True)
    optimize.add_argument("--baseline", type=Path, required=True)
    optimize.add_argument("--patch", type=Path, required=True)
    optimize.add_argument("--state-db", type=Path, default=Path("atlas-harness.sqlite"))
    optimize.add_argument("--evaluation-db", type=Path, default=Path("atlas-evaluations.sqlite"))
    optimize.add_argument("--environment-id", default="local-fixture")
    optimize.add_argument("--trials", type=int, default=3)
    optimize.add_argument("--approved-skill", action="append", default=[])
    optimize.add_argument("--authorized-tool", action="append", default=[])
    optimize.add_argument("--inference-budget-ceiling", type=int, default=4096)
    optimize.add_argument("--fixture-only", action="store_true", required=True)

    promote = subparsers.add_parser(
        "promote", help="explicitly approve and promote an eligible run"
    )
    promote.add_argument("--state-db", type=Path, default=Path("atlas-harness.sqlite"))
    promote.add_argument("--experiment-id", required=True)
    promote.add_argument("--experiment-hash", required=True)
    promote.add_argument("--approver-id", required=True)

    rollback = subparsers.add_parser("rollback", help="create a new version from an older harness")
    rollback.add_argument("--state-db", type=Path, default=Path("atlas-harness.sqlite"))
    rollback.add_argument("--harness-id", required=True)
    rollback.add_argument("--version", type=int, required=True)
    rollback.add_argument("--approver-id", required=True)

    dashboard = subparsers.add_parser("dashboard", help="serve a minimal read-only run dashboard")
    dashboard.add_argument("--events-db", type=Path, required=True)
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()
    try:
        if args.command == "validate-plan":
            payload = json.loads(args.path.read_text(encoding="utf-8"))
            compiled = PlanCompiler().compile(payload)
            _print_model(
                {
                    "plan_id": compiled.plan.plan_id,
                    "content_hash": compiled.content_hash,
                    "topological_order": compiled.topological_order,
                }
            )
            return 0
        if args.command == "inspect-run":
            events = asyncio.run(SQLiteRunEventStore(args.events_db).list_run(args.run_id))
            _print_model(events)
            return 0
        if args.command == "replay-tools":

            async def load_exchanges() -> list[dict[str, object]]:
                store = SQLiteRunEventStore(args.events_db)
                artifacts = LocalArtifactStore(args.artifacts)
                redactor = SecretRedactor()
                events = await store.list_run(args.run_id)
                exchanges: list[dict[str, object]] = []
                for event in events:
                    if event.event_type != "tool.completed" or len(event.artifact_refs) < 2:
                        continue
                    request = json.loads(await artifacts.get(event.artifact_refs[0]))
                    response = json.loads(await artifacts.get(event.artifact_refs[-1]))
                    exchanges.append(
                        redactor.redact(
                            {
                                "node_id": event.node_id,
                                "tool_id": event.tool_id,
                                "request": request,
                                "response": response,
                            }
                        )
                    )
                return exchanges

            _print_model(asyncio.run(load_exchanges()))
            return 0
        if args.command in {"evaluate", "compare"}:
            suite = _load_suite(args.suite)
            store = SQLiteEvaluationStore(args.store)
            if args.command == "evaluate":
                eval_report = asyncio.run(
                    evaluate_suite(
                        suite,
                        harness_id=args.harness_id,
                        runner=_SuiteFixtureRunner(),
                        environment_id=args.environment_id,
                        trials=args.trials,
                        illustrative=True,
                        store=store,
                    )
                )
                _print_model(eval_report)
            else:
                comparison = asyncio.run(
                    compare_harnesses(
                        suite,
                        baseline_id=args.baseline_id,
                        candidate_id=args.candidate_id,
                        runner=_SuiteFixtureRunner(),
                        environment_id=args.environment_id,
                        trials=args.trials,
                        illustrative=True,
                        store=store,
                    )
                )
                _print_model(comparison)
            return 0
        if args.command == "optimize":
            suite = _load_suite(args.suite)
            baseline = HarnessConfig.model_validate_json(args.baseline.read_text(encoding="utf-8"))
            patch = HarnessPatch.model_validate_json(args.patch.read_text(encoding="utf-8"))
            registry = SQLiteHarnessRegistry(args.state_db)
            try:
                registry.get(baseline.harness_id, baseline.version)
            except KeyError:
                registry.register_initial(baseline)
            experiment = asyncio.run(
                run_experiment(
                    suite,
                    baseline=baseline,
                    patch=patch,
                    runner=_SuiteFixtureRunner(),
                    environment_id=args.environment_id,
                    approved_skill_ids=frozenset(args.approved_skill),
                    authorized_tool_ids=frozenset(args.authorized_tool),
                    inference_budget_ceiling=args.inference_budget_ceiling,
                    trials=args.trials,
                    registry=registry,
                    evaluation_store=SQLiteEvaluationStore(args.evaluation_db),
                )
            )
            _print_model(experiment)
            return 0
        if args.command == "promote":
            registry = SQLiteHarnessRegistry(args.state_db)
            experiment = registry.get_experiment(args.experiment_id)
            approval = PromotionApproval(
                experiment_id=args.experiment_id,
                experiment_hash=args.experiment_hash,
                approved=True,
                approver_id=args.approver_id,
                expires_at=datetime.now(UTC) + timedelta(minutes=5),
            )
            _print_model(registry.promote(experiment, approval))
            return 0
        if args.command == "rollback":
            registry = SQLiteHarnessRegistry(args.state_db)
            _print_model(
                registry.rollback(args.harness_id, args.version, approver_id=args.approver_id)
            )
            return 0
        if args.command == "dashboard":
            _serve_dashboard(args.events_db, args.host, args.port)
            return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"atlas: {exc}", file=sys.stderr)
        return 2
    return 2


def _create_dashboard_server(events_db: Path, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            store = SQLiteRunEventStore(events_db)
            if parsed.path.startswith("/api/runs/"):
                run_id = unquote(parsed.path.removeprefix("/api/runs/"))
                payload = _render_run_json(asyncio.run(store.list_run(run_id)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(payload.encode())
                return
            events = asyncio.run(store.list_recent(100))
            page = _render_dashboard_html(events)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page.encode())

        def log_message(self, fmt: str, *args: Any) -> None:
            del fmt, args

    return ThreadingHTTPServer((host, port), Handler)


def _render_dashboard_html(events: list[Any]) -> str:
    run_ids = list(dict.fromkeys(event.run_id for event in events))
    rows = "".join(
        f'<li><a href="/api/runs/{escape(quote(run_id, safe=""))}">{escape(run_id)}</a></li>'
        for run_id in run_ids
    )
    return (
        "<!doctype html><html><head><title>ATLAS Runs</title></head><body>"
        "<h1>ATLAS recent runs</h1><p>Read-only local event view.</p><ul>"
        f"{rows}</ul></body></html>"
    )


def _render_run_json(events: list[Any]) -> str:
    return json.dumps([event.model_dump(mode="json") for event in events])


def _serve_dashboard(events_db: Path, host: str, port: int) -> None:
    server = _create_dashboard_server(events_db, host, port)
    print(f"ATLAS read-only dashboard at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
