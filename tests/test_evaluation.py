from __future__ import annotations

from pathlib import Path

import pytest

from atlas.evaluation import (
    BenchmarkSuite,
    EvaluationCase,
    EvaluationObservation,
    IsolatedEvaluationRunner,
    RubricEvaluator,
    SQLiteEvaluationStore,
    TemporaryWorkspaceProvider,
    compare_harnesses,
    evaluate_suite,
)


class FixtureRunner:
    async def __call__(
        self, case: EvaluationCase, *, harness_id: str, trial: int
    ) -> EvaluationObservation:
        del case, trial
        good = harness_id.startswith("candidate")
        return EvaluationObservation(
            output="verified" if good else "missed",
            node_statuses={"final": "succeeded"},
            test_results={"acceptance": good},
            cost_usd=0.1,
            latency_ms=5,
            retries=1 if not good else 0,
            tool_invocations=2 if not good else 1,
        )


class FixedRubric:
    async def evaluate(self, *, case: EvaluationCase, observation: EvaluationObservation) -> float:
        del case
        return 0.9 if observation.output == "verified" else 0.2


def suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        suite_id="test-suite",
        version="1.2",
        cases=(
            EvaluationCase(
                case_id="case-a",
                task="Verify required output",
                required_output_fragments=("verified",),
                expected_node_statuses={"final": "succeeded"},
                expected_test_results={"acceptance": True},
            ),
        ),
    )


@pytest.mark.asyncio
async def test_paired_comparison_uses_same_cases_and_reports_variance(tmp_path: Path) -> None:
    store = SQLiteEvaluationStore(tmp_path / "evaluations.sqlite")
    report = await compare_harnesses(
        suite(),
        baseline_id="baseline-v1",
        candidate_id="candidate-v2",
        runner=FixtureRunner(),
        environment_id="python312-fixture",
        trials=3,
        store=store,
    )
    assert report.paired_sample_count == 3
    assert report.baseline.metrics.sample_count == report.candidate.metrics.sample_count == 3
    assert report.baseline.metrics.task_success_rate == 0
    assert report.candidate.metrics.task_success_rate == 1
    assert report.candidate.metrics.verified_quality_ci95 == (1.0, 1.0)
    assert report.candidate.metrics.tool_invocations_total == 3
    assert report.hard_safety_passed
    assert len(await store.records("suite")) == 2
    assert len(await store.records("report")) == 2
    persisted = await store.report(report.candidate.report_id)
    assert persisted.harness_id == "candidate-v2"

    single = await evaluate_suite(
        suite(), harness_id="candidate", runner=FixtureRunner(), environment_id="fixture"
    )
    assert single.metrics.verified_quality_ci95 == (0.0, 1.0)


@pytest.mark.asyncio
async def test_safety_violations_fail_hard_gate(tmp_path: Path) -> None:
    class UnsafeRunner:
        async def __call__(
            self, case: EvaluationCase, *, harness_id: str, trial: int
        ) -> EvaluationObservation:
            del case, harness_id, trial
            return EvaluationObservation(output="verified", policy_violations=1)

    report = await evaluate_suite(
        suite(),
        harness_id="unsafe",
        runner=UnsafeRunner(),
        environment_id="local",
        store=SQLiteEvaluationStore(tmp_path / "unsafe.sqlite"),
    )
    assert report.metrics.policy_violations == 1
    assert not report.metrics.hard_safety_passed


@pytest.mark.asyncio
async def test_qualitative_evaluator_is_versioned_separately() -> None:
    report = await evaluate_suite(
        suite(),
        harness_id="candidate",
        runner=FixtureRunner(),
        environment_id="local",
        rubric_evaluator=RubricEvaluator(
            FixedRubric(), evaluator_version="rubric-v3", model_id="fixture-judge"
        ),
    )
    assert report.evaluator_version == "deterministic-v1"
    assert report.rubric_evaluator_version == "rubric-v3"
    assert report.rubric_model_id == "fixture-judge"
    assert report.metrics.rubric_quality_mean == 0.9


@pytest.mark.asyncio
async def test_isolated_runner_gets_fresh_workspace_per_harness_and_trial(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "fixture.txt").write_text("same immutable fixture")
    paths: list[Path] = []

    class WorkspaceRunner:
        async def evaluate_in_workspace(
            self, case: EvaluationCase, *, harness_id: str, trial: int, workspace: Path
        ) -> EvaluationObservation:
            del case, trial
            paths.append(workspace)
            assert (workspace / "fixture.txt").read_text() == "same immutable fixture"
            return EvaluationObservation(output=harness_id)

    runner = IsolatedEvaluationRunner(
        WorkspaceRunner(),
        TemporaryWorkspaceProvider(root=tmp_path, template_directory=template),
    )
    await runner(suite().cases[0], harness_id="baseline", trial=0)
    await runner(suite().cases[0], harness_id="candidate", trial=0)
    assert paths[0] != paths[1]
    assert not paths[0].exists()
    assert not paths[1].exists()


def test_duplicate_suite_case_ids_are_rejected() -> None:
    duplicate = EvaluationCase(case_id="same", task="a")
    with pytest.raises(ValueError, match="unique"):
        BenchmarkSuite(suite_id="bad", version="1", cases=(duplicate, duplicate))
