from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.evaluation import BenchmarkSuite, EvaluationCase, EvaluationObservation
from atlas.optimizer import (
    HarnessConfig,
    HarnessPatch,
    PromotionApproval,
    SQLiteHarnessRegistry,
    apply_patch,
    run_experiment,
)


class OptimizerRunner:
    def __init__(self, *, candidate_success: bool = True) -> None:
        self.candidate_success = candidate_success

    async def __call__(
        self, case: EvaluationCase, *, harness_id: str, trial: int
    ) -> EvaluationObservation:
        del case, trial
        candidate = harness_id.endswith("@2")
        success = self.candidate_success if candidate else False
        return EvaluationObservation(
            output="verified" if success else "missed",
            test_results={"required": success},
            cost_usd=0.02 if candidate else 0.03,
            latency_ms=20 if candidate else 25,
            policy_violations=0,
        )


def setup() -> tuple[BenchmarkSuite, HarnessConfig, HarnessPatch]:
    return (
        BenchmarkSuite(
            suite_id="smoke",
            version="1",
            cases=(
                EvaluationCase(
                    case_id="one",
                    task="verify",
                    required_output_fragments=("verified",),
                    expected_test_results={"required": True},
                ),
            ),
        ),
        HarnessConfig(
            harness_id="repo",
            task_scope="repo-maintenance",
            skill_ids=("inspect",),
            tool_preferences=("read",),
        ),
        HarnessPatch(
            base_harness_id="repo",
            rationale="Add an explicit verification pass.",
            evidence_report_ids=("evidence-1",),
            verification_steps=True,
            skill_ids=("inspect", "verify"),
            tool_preferences=("read",),
            inference_budget_tokens=3000,
        ),
    )


@pytest.mark.asyncio
async def test_evidence_gated_experiment_and_manual_promotion_rollback(tmp_path: Path) -> None:
    suite, baseline, patch = setup()
    registry = SQLiteHarnessRegistry(tmp_path / "harness.sqlite")
    registry.register_initial(baseline)
    report = await run_experiment(
        suite,
        baseline=baseline,
        patch=patch,
        runner=OptimizerRunner(),
        environment_id="fixture-py312",
        approved_skill_ids=frozenset({"inspect", "verify"}),
        authorized_tool_ids=frozenset({"read"}),
        inference_budget_ceiling=4096,
        trials=3,
        registry=registry,
    )
    assert report.status == "eligible"
    assert report.gates.passed
    assert report.comparison.paired_sample_count == 3
    assert registry.get_experiment(report.experiment_id).experiment_hash == report.experiment_hash
    assert len(registry.experiences(task_scope="repo-maintenance")) == 1

    with pytest.raises(ValueError, match="approval"):
        registry.promote(
            report,
            PromotionApproval(
                experiment_id=report.experiment_id,
                experiment_hash=report.experiment_hash,
                approved=False,
                approver_id="operator",
                expires_at=datetime.now(UTC) + timedelta(minutes=1),
            ),
        )
    promoted = registry.promote(
        report,
        PromotionApproval(
            experiment_id=report.experiment_id,
            experiment_hash=report.experiment_hash,
            approved=True,
            approver_id="operator:alice",
            expires_at=datetime.now(UTC) + timedelta(minutes=1),
        ),
    )
    assert registry.get("repo").version == 2
    assert promoted.verification_steps
    rolled_back = registry.rollback("repo", 1, approver_id="operator:alice")
    assert rolled_back.version == 3
    assert not rolled_back.verification_steps


@pytest.mark.asyncio
async def test_regression_fails_promotion_gates(tmp_path: Path) -> None:
    suite, baseline, patch = setup()
    registry = SQLiteHarnessRegistry(tmp_path / "harness.sqlite")
    registry.register_initial(baseline)
    report = await run_experiment(
        suite,
        baseline=baseline,
        patch=patch,
        runner=OptimizerRunner(candidate_success=False),
        environment_id="fixture",
        approved_skill_ids=frozenset({"inspect", "verify"}),
        authorized_tool_ids=frozenset({"read"}),
        inference_budget_ceiling=4096,
        registry=registry,
    )
    assert report.status == "rejected"
    assert not report.gates.checks["mandatory_cases"]
    with pytest.raises(ValueError, match="not eligible"):
        registry.promote(
            report,
            PromotionApproval(
                experiment_id=report.experiment_id,
                experiment_hash=report.experiment_hash,
                approved=True,
                approver_id="operator",
                expires_at=datetime.now(UTC) + timedelta(minutes=1),
            ),
        )


def test_patch_rejects_immutable_policy_and_unapproved_capabilities() -> None:
    suite, baseline, _patch = setup()
    del suite
    with pytest.raises(ValidationError):
        HarnessPatch.model_validate(
            {
                "base_harness_id": "repo",
                "rationale": "grant tools",
                "evidence_report_ids": ["e"],
                "allowed_tool_ids": ["delete"],
            }
        )
    unauthorized = HarnessPatch(
        base_harness_id="repo",
        rationale="Prefer a tool",
        evidence_report_ids=("e",),
        tool_preferences=("delete",),
    )
    with pytest.raises(ValueError, match="authorization"):
        apply_patch(
            baseline,
            unauthorized,
            approved_skill_ids=frozenset({"inspect"}),
            authorized_tool_ids=frozenset({"read"}),
            inference_budget_ceiling=4096,
        )
    over_budget = HarnessPatch(
        base_harness_id="repo",
        rationale="Raise budget",
        evidence_report_ids=("e",),
        inference_budget_tokens=9000,
    )
    with pytest.raises(ValueError, match="ceiling"):
        apply_patch(
            baseline,
            over_budget,
            approved_skill_ids=frozenset({"inspect"}),
            authorized_tool_ids=frozenset({"read"}),
            inference_budget_ceiling=4096,
        )


@pytest.mark.asyncio
async def test_candidate_policy_violation_always_fails_promotion_gate() -> None:
    suite, baseline, patch = setup()

    class ViolatingRunner:
        async def __call__(
            self, case: EvaluationCase, *, harness_id: str, trial: int
        ) -> EvaluationObservation:
            del case, trial
            candidate = harness_id.endswith("@2")
            return EvaluationObservation(
                output="verified",
                test_results={"required": True},
                policy_violations=1 if candidate else 0,
            )

    report = await run_experiment(
        suite,
        baseline=baseline,
        patch=patch,
        runner=ViolatingRunner(),
        environment_id="fixture",
        approved_skill_ids=frozenset({"inspect", "verify"}),
        authorized_tool_ids=frozenset({"read"}),
        inference_budget_ceiling=4096,
    )
    assert report.status == "rejected"
    assert not report.gates.checks["hard_safety"]
