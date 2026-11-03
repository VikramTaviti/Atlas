"""Versioned deterministic benchmark evaluation and paired comparison."""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import sqlite3
import statistics
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    required_output_fragments: tuple[str, ...] = ()
    expected_node_statuses: dict[str, str] = Field(default_factory=dict)
    expected_test_results: dict[str, bool] = Field(default_factory=dict)
    mandatory: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    suite_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    cases: tuple[EvaluationCase, ...] = Field(min_length=1)
    evaluator_version: str = "deterministic-v1"

    @model_validator(mode="after")
    def unique_case_ids(self) -> BenchmarkSuite:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark case IDs must be unique")
        return self


class EvaluationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output: Any = None
    node_statuses: dict[str, str] = Field(default_factory=dict)
    test_results: dict[str, bool] = Field(default_factory=dict)
    cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    retries: int = Field(default=0, ge=0)
    tool_invocations: int = Field(default=0, ge=0)
    policy_violations: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RubricResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float = Field(ge=0, le=1)
    evaluator_version: str
    model_id: str
    rationale: str = ""


class RubricAdapter(Protocol):
    async def evaluate(
        self, *, case: EvaluationCase, observation: EvaluationObservation
    ) -> float: ...


class RubricEvaluator:
    """Optional qualitative evaluator with explicit evaluator and model provenance."""

    def __init__(self, adapter: RubricAdapter, *, evaluator_version: str, model_id: str) -> None:
        self.adapter = adapter
        self.evaluator_version = evaluator_version
        self.model_id = model_id

    async def evaluate(
        self, case: EvaluationCase, observation: EvaluationObservation
    ) -> RubricResult:
        score = await self.adapter.evaluate(case=case, observation=observation)
        return RubricResult(
            score=score,
            evaluator_version=self.evaluator_version,
            model_id=self.model_id,
        )


class EvaluationRunner(Protocol):
    async def __call__(
        self, case: EvaluationCase, *, harness_id: str, trial: int
    ) -> EvaluationObservation: ...


class WorkspaceEvaluationRunner(Protocol):
    async def evaluate_in_workspace(
        self, case: EvaluationCase, *, harness_id: str, trial: int, workspace: Path
    ) -> EvaluationObservation: ...


class WorkspaceProvider(Protocol):
    def isolated_workspace(
        self, *, harness_id: str, case_id: str, trial: int
    ) -> AbstractAsyncContextManager[Path]: ...


class TemporaryWorkspaceProvider:
    """Creates a fresh disposable workspace for every harness/case/trial invocation."""

    def __init__(
        self, *, root: str | Path | None = None, template_directory: str | Path | None = None
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.template_directory = (
            Path(template_directory).resolve() if template_directory is not None else None
        )
        if self.template_directory is not None and not self.template_directory.is_dir():
            raise ValueError("workspace template must be an existing directory")
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def isolated_workspace(
        self, *, harness_id: str, case_id: str, trial: int
    ) -> AsyncIterator[Path]:
        del harness_id, case_id, trial
        with tempfile.TemporaryDirectory(prefix="atlas-eval-", dir=self.root) as temporary:
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            if self.template_directory is not None:
                await asyncio.to_thread(
                    shutil.copytree,
                    self.template_directory,
                    workspace,
                    dirs_exist_ok=True,
                    symlinks=True,
                )
            yield workspace


class IsolatedEvaluationRunner:
    """Adapt a workspace-aware runner to the ordinary benchmark runner protocol."""

    def __init__(self, runner: WorkspaceEvaluationRunner, workspace_provider: WorkspaceProvider):
        self.runner = runner
        self.workspace_provider = workspace_provider

    async def __call__(
        self, case: EvaluationCase, *, harness_id: str, trial: int
    ) -> EvaluationObservation:
        async with self.workspace_provider.isolated_workspace(
            harness_id=harness_id, case_id=case.case_id, trial=trial
        ) as workspace:
            return await self.runner.evaluate_in_workspace(
                case, harness_id=harness_id, trial=trial, workspace=workspace
            )


class CaseEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    trial: int
    task_success: bool
    verified_quality: float
    cost_usd: float
    latency_ms: float
    retries: int
    tool_invocations: int
    policy_violations: int
    mandatory: bool = True
    rubric_quality: float | None = Field(default=None, ge=0, le=1)
    rubric_evaluator_version: str | None = None
    rubric_model_id: str | None = None
    output: Any = None
    test_results: dict[str, bool] = Field(default_factory=dict)


class AggregateMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_count: int
    task_success_rate: float
    verified_quality_mean: float
    verified_quality_variance: float
    verified_quality_ci95: tuple[float, float]
    rubric_quality_mean: float | None
    total_cost_usd: float
    cost_per_successful_task_usd: float | None
    latency_median_ms: float
    latency_p95_ms: float
    retries_total: int
    tool_invocations_total: int
    policy_violations: int
    hard_safety_passed: bool


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    suite_id: str
    suite_version: str
    evaluator_version: str
    rubric_evaluator_version: str | None = None
    rubric_model_id: str | None = None
    harness_id: str
    environment_id: str
    trial_count: int
    cases: tuple[CaseEvaluation, ...]
    metrics: AggregateMetrics
    illustrative: bool = False


class PairedComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    suite_id: str
    suite_version: str
    environment_id: str
    baseline: EvaluationReport
    candidate: EvaluationReport
    paired_sample_count: int
    quality_delta_mean: float
    quality_delta_variance: float
    cost_delta_usd: float
    hard_safety_passed: bool


class SQLiteEvaluationStore:
    """Append-only persistence of suite snapshots and completed reports."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS evaluation_records ("
                "sequence INTEGER PRIMARY KEY AUTOINCREMENT, record_type TEXT, record_json TEXT)"
            )

    async def append_suite(self, suite: BenchmarkSuite) -> None:
        await asyncio.to_thread(self._append, "suite", suite.model_dump_json())

    async def append_report(self, report: EvaluationReport) -> None:
        await asyncio.to_thread(self._append, "report", report.model_dump_json())

    async def report(self, report_id: str) -> EvaluationReport:
        payload = await asyncio.to_thread(self._read_report, report_id)
        return EvaluationReport.model_validate_json(payload)

    def _append(self, record_type: str, payload: str) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO evaluation_records(record_type, record_json) VALUES (?, ?)",
                (record_type, payload),
            )

    async def records(self, record_type: str | None = None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._read, record_type)

    def _read(self, record_type: str | None) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as db:
            if record_type is None:
                rows = db.execute(
                    "SELECT record_json FROM evaluation_records ORDER BY sequence"
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT record_json FROM evaluation_records WHERE record_type=? "
                    "ORDER BY sequence",
                    (record_type,),
                ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def _read_report(self, report_id: str) -> str:
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                "SELECT record_json FROM evaluation_records WHERE record_type='report' "
                "ORDER BY sequence DESC"
            ).fetchall()
        for (payload,) in rows:
            if json.loads(payload).get("report_id") == report_id:
                if isinstance(payload, str):
                    return payload
        raise KeyError(f"unknown evaluation report: {report_id}")


async def evaluate_suite(
    suite: BenchmarkSuite,
    *,
    harness_id: str,
    runner: EvaluationRunner,
    environment_id: str,
    trials: int = 1,
    illustrative: bool = False,
    store: SQLiteEvaluationStore | None = None,
    rubric_evaluator: RubricEvaluator | None = None,
) -> EvaluationReport:
    if trials < 1:
        raise ValueError("trials must be positive")
    if store:
        await store.append_suite(suite)
    evaluations: list[CaseEvaluation] = []
    for trial in range(trials):
        for case in suite.cases:
            started = time.perf_counter()
            observation = await runner(case, harness_id=harness_id, trial=trial)
            measured_latency = (time.perf_counter() - started) * 1000
            success = deterministic_checks(case, observation)
            rubric = (
                await rubric_evaluator.evaluate(case, observation) if rubric_evaluator else None
            )
            evaluations.append(
                CaseEvaluation(
                    case_id=case.case_id,
                    trial=trial,
                    task_success=success,
                    verified_quality=float(success),
                    cost_usd=observation.cost_usd,
                    latency_ms=max(observation.latency_ms, measured_latency),
                    retries=observation.retries,
                    tool_invocations=observation.tool_invocations,
                    policy_violations=observation.policy_violations,
                    mandatory=case.mandatory,
                    rubric_quality=rubric.score if rubric else None,
                    rubric_evaluator_version=rubric.evaluator_version if rubric else None,
                    rubric_model_id=rubric.model_id if rubric else None,
                    output=observation.output,
                    test_results=observation.test_results,
                )
            )
    report = EvaluationReport(
        suite_id=suite.suite_id,
        suite_version=suite.version,
        evaluator_version=suite.evaluator_version,
        rubric_evaluator_version=(rubric_evaluator.evaluator_version if rubric_evaluator else None),
        rubric_model_id=rubric_evaluator.model_id if rubric_evaluator else None,
        harness_id=harness_id,
        environment_id=environment_id,
        trial_count=trials,
        cases=tuple(evaluations),
        metrics=aggregate_metrics(evaluations),
        illustrative=illustrative,
    )
    if store:
        await store.append_report(report)
    return report


async def compare_harnesses(
    suite: BenchmarkSuite,
    *,
    baseline_id: str,
    candidate_id: str,
    runner: EvaluationRunner,
    environment_id: str,
    trials: int = 1,
    illustrative: bool = False,
    store: SQLiteEvaluationStore | None = None,
) -> PairedComparison:
    baseline = await evaluate_suite(
        suite,
        harness_id=baseline_id,
        runner=runner,
        environment_id=environment_id,
        trials=trials,
        illustrative=illustrative,
        store=store,
    )
    candidate = await evaluate_suite(
        suite,
        harness_id=candidate_id,
        runner=runner,
        environment_id=environment_id,
        trials=trials,
        illustrative=illustrative,
        store=store,
    )
    baseline_by_case = {(row.case_id, row.trial): row for row in baseline.cases}
    candidate_by_case = {(row.case_id, row.trial): row for row in candidate.cases}
    keys = sorted(set(baseline_by_case) & set(candidate_by_case))
    deltas = [
        candidate_by_case[key].verified_quality - baseline_by_case[key].verified_quality
        for key in keys
    ]
    return PairedComparison(
        suite_id=suite.suite_id,
        suite_version=suite.version,
        environment_id=environment_id,
        baseline=baseline,
        candidate=candidate,
        paired_sample_count=len(keys),
        quality_delta_mean=statistics.fmean(deltas) if deltas else 0.0,
        quality_delta_variance=statistics.variance(deltas) if len(deltas) > 1 else 0.0,
        cost_delta_usd=candidate.metrics.total_cost_usd - baseline.metrics.total_cost_usd,
        hard_safety_passed=(
            baseline.metrics.hard_safety_passed and candidate.metrics.hard_safety_passed
        ),
    )


def deterministic_checks(case: EvaluationCase, observation: EvaluationObservation) -> bool:
    output = (
        observation.output
        if isinstance(observation.output, str)
        else json.dumps(observation.output, sort_keys=True, ensure_ascii=False)
    )
    return (
        all(fragment in output for fragment in case.required_output_fragments)
        and all(
            observation.node_statuses.get(key) == value
            for key, value in case.expected_node_statuses.items()
        )
        and all(
            observation.test_results.get(key) is value
            for key, value in case.expected_test_results.items()
        )
    )


def aggregate_metrics(rows: list[CaseEvaluation]) -> AggregateMetrics:
    count = len(rows)
    qualities = [row.verified_quality for row in rows]
    mean = statistics.fmean(qualities) if qualities else 0.0
    variance = statistics.variance(qualities) if len(qualities) > 1 else 0.0
    margin = 1.96 * math.sqrt(variance / count) if count > 1 else None
    successes = sum(row.task_success for row in rows)
    cost = sum(row.cost_usd for row in rows)
    latencies = sorted(row.latency_ms for row in rows)
    return AggregateMetrics(
        sample_count=count,
        task_success_rate=successes / count if count else 0.0,
        verified_quality_mean=mean,
        verified_quality_variance=variance,
        verified_quality_ci95=(
            (max(0.0, mean - margin), min(1.0, mean + margin)) if margin is not None else (0.0, 1.0)
        ),
        rubric_quality_mean=(
            statistics.fmean([row.rubric_quality for row in rows if row.rubric_quality is not None])
            if any(row.rubric_quality is not None for row in rows)
            else None
        ),
        total_cost_usd=cost,
        cost_per_successful_task_usd=cost / successes if successes else None,
        latency_median_ms=statistics.median(latencies) if latencies else 0.0,
        latency_p95_ms=latencies[max(0, math.ceil(0.95 * count) - 1)] if latencies else 0.0,
        retries_total=sum(row.retries for row in rows),
        tool_invocations_total=sum(row.tool_invocations for row in rows),
        policy_violations=sum(row.policy_violations for row in rows),
        hard_safety_passed=all(row.policy_violations == 0 for row in rows),
    )
