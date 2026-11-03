"""Illustrative, deterministic baseline/candidate experiment with explicit approval."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from atlas.evaluation import (
    BenchmarkSuite,
    EvaluationCase,
    EvaluationObservation,
    SQLiteEvaluationStore,
)
from atlas.optimizer import (
    HarnessConfig,
    HarnessPatch,
    PromotionApproval,
    SQLiteHarnessRegistry,
    run_experiment,
)


class SmokeFixtureRunner:
    async def __call__(
        self, case: EvaluationCase, *, harness_id: str, trial: int
    ) -> EvaluationObservation:
        del trial
        fixtures = case.metadata["fixture_observations"]
        name = "candidate" if harness_id.endswith("@2") else "baseline"
        return EvaluationObservation.model_validate(fixtures[name])


async def run_demo(approve: bool) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    suite = BenchmarkSuite.model_validate_json(
        (root / "benchmarks" / "smoke.json").read_text(encoding="utf-8")
    )
    baseline = HarnessConfig.model_validate_json(
        (root / "harnesses" / "baseline.json").read_text(encoding="utf-8")
    )
    patch = HarnessPatch.model_validate_json(
        (root / "harnesses" / "add_verification.json").read_text(encoding="utf-8")
    )
    with tempfile.TemporaryDirectory(prefix="atlas-demo-") as directory:
        registry = SQLiteHarnessRegistry(Path(directory) / "harness.sqlite")
        registry.register_initial(baseline)
        experiment = await run_experiment(
            suite,
            baseline=baseline,
            patch=patch,
            runner=SmokeFixtureRunner(),
            environment_id="python312-deterministic-fixtures",
            approved_skill_ids=frozenset({"repository-inspection", "verification"}),
            authorized_tool_ids=frozenset({"workspace.read", "workspace.test"}),
            inference_budget_ceiling=4096,
            trials=3,
            registry=registry,
            evaluation_store=SQLiteEvaluationStore(Path(directory) / "evaluations.sqlite"),
        )
        result: dict[str, Any] = {
            "illustrative_fixture_only": True,
            "experiment": experiment.model_dump(mode="json"),
            "promotion": "not requested",
        }
        if approve and experiment.gates.passed:
            promoted = registry.promote(
                experiment,
                PromotionApproval(
                    experiment_id=experiment.experiment_id,
                    experiment_hash=experiment.experiment_hash,
                    approved=True,
                    approver_id="demo-operator-explicit-flag",
                    expires_at=datetime.now(UTC) + timedelta(minutes=5),
                ),
            )
            result["promotion"] = f"approved {promoted.harness_id}@{promoted.version}"
            rollback = registry.rollback(
                promoted.harness_id, baseline.version, approver_id="demo-operator-explicit-flag"
            )
            result["rollback"] = (
                f"restored prior config as {rollback.harness_id}@{rollback.version}"
            )
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approve",
        action="store_true",
        help="simulate explicit approval in this fixture-only demo",
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_demo(args.approve)), indent=2))


if __name__ == "__main__":
    main()
