"""Constrained harness changes, evidence-gated experiments, and manual promotion."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from atlas.evaluation import (
    BenchmarkSuite,
    EvaluationRunner,
    PairedComparison,
    SQLiteEvaluationStore,
    compare_harnesses,
)


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    harness_id: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    task_scope: str = Field(min_length=1)
    instruction_variant: str = "default"
    skill_ids: tuple[str, ...] = ()
    tool_preferences: tuple[str, ...] = ()
    verification_steps: bool = False
    inference_budget_tokens: int = Field(default=2048, ge=1)


class HarnessPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_harness_id: str
    rationale: str = Field(min_length=1, max_length=2000)
    evidence_report_ids: tuple[str, ...] = Field(min_length=1)
    instruction_variant: str | None = Field(default=None, max_length=1000)
    skill_ids: tuple[str, ...] | None = None
    tool_preferences: tuple[str, ...] | None = None
    verification_steps: bool | None = None
    inference_budget_tokens: int | None = Field(default=None, ge=1)


class FailureClassification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    categories: tuple[str, ...]
    evidence_report_id: str
    task_scope: str


class ExperimentGates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    checks: dict[str, bool]
    reasons: tuple[str, ...]


class ExperimentReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(default_factory=lambda: str(uuid4()))
    experiment_hash: str
    patch: HarnessPatch
    baseline_harness: HarnessConfig
    candidate_harness: HarnessConfig
    comparison: PairedComparison
    gates: ExperimentGates
    status: str


class PromotionApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    experiment_hash: str
    approved: bool
    approver_id: str = Field(min_length=1)
    expires_at: datetime


class ExperienceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    experience_id: str = Field(default_factory=lambda: str(uuid4()))
    experiment_id: str
    task_scope: str
    classification: FailureClassification
    evidence_report_ids: tuple[str, ...]
    confidence: float = Field(ge=0, le=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime


class SQLiteHarnessRegistry:
    """Local persistent version history, experiment decisions, and scoped experience."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.executescript(
                "CREATE TABLE IF NOT EXISTS harness_versions "
                "(harness_id TEXT, version INTEGER, config_json TEXT, experiment_id TEXT, "
                "PRIMARY KEY(harness_id, version));"
                "CREATE TABLE IF NOT EXISTS experiences "
                "(experience_id TEXT PRIMARY KEY, record_json TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS experiments "
                "(experiment_id TEXT PRIMARY KEY, report_json TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS promotion_events "
                "(sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_json TEXT NOT NULL);"
            )

    def register_initial(self, config: HarnessConfig) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO harness_versions VALUES (?, ?, ?, NULL)",
                (config.harness_id, config.version, config.model_dump_json()),
            )

    def get(self, harness_id: str, version: int | None = None) -> HarnessConfig:
        with sqlite3.connect(self.path) as db:
            if version is None:
                row = db.execute(
                    "SELECT config_json FROM harness_versions WHERE harness_id=? "
                    "ORDER BY version DESC LIMIT 1",
                    (harness_id,),
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT config_json FROM harness_versions WHERE harness_id=? AND version=?",
                    (harness_id, version),
                ).fetchone()
        if row is None:
            raise KeyError(f"unknown harness version: {harness_id}@{version or 'latest'}")
        return HarnessConfig.model_validate_json(row[0])

    def promote(self, experiment: ExperimentReport, approval: PromotionApproval) -> HarnessConfig:
        if not experiment.gates.passed or experiment.status != "eligible":
            raise ValueError("experiment is not eligible for promotion")
        stored = self.get_experiment(experiment.experiment_id)
        expected_hash = hashlib.sha256(
            (experiment.patch.model_dump_json() + experiment.comparison.model_dump_json()).encode(
                "utf-8"
            )
        ).hexdigest()
        if (
            stored.model_dump_json() != experiment.model_dump_json()
            or experiment.experiment_hash != expected_hash
        ):
            raise ValueError("experiment report is not the persisted, hash-valid record")
        now = datetime.now(UTC)
        if (
            not approval.approved
            or approval.experiment_id != experiment.experiment_id
            or approval.experiment_hash != experiment.experiment_hash
            or approval.expires_at.tzinfo is None
            or approval.expires_at <= now
            or not approval.approver_id.strip()
        ):
            raise ValueError("promotion approval is missing, stale, or bound to another experiment")
        candidate = experiment.candidate_harness
        with sqlite3.connect(self.path) as db:
            current = db.execute(
                "SELECT MAX(version) FROM harness_versions WHERE harness_id=?",
                (candidate.harness_id,),
            ).fetchone()[0]
            if current != experiment.baseline_harness.version:
                raise ValueError("baseline is stale; harness changed after the experiment")
            db.execute(
                "INSERT INTO harness_versions VALUES (?, ?, ?, ?)",
                (
                    candidate.harness_id,
                    candidate.version,
                    candidate.model_dump_json(),
                    experiment.experiment_id,
                ),
            )
            db.execute(
                "INSERT INTO promotion_events(event_json) VALUES (?)",
                (
                    json.dumps(
                        {
                            "action": "promote",
                            "experiment_id": experiment.experiment_id,
                            "experiment_hash": experiment.experiment_hash,
                            "harness_id": candidate.harness_id,
                            "version": candidate.version,
                            "approver_id": approval.approver_id,
                            "timestamp": now.isoformat(),
                        },
                        sort_keys=True,
                    ),
                ),
            )
        return candidate

    def rollback(self, harness_id: str, version: int, *, approver_id: str) -> HarnessConfig:
        if not approver_id.strip():
            raise ValueError("rollback requires an approver identity")
        target = self.get(harness_id, version)
        current = self.get(harness_id)
        if version >= current.version:
            raise ValueError("rollback target must be an earlier harness version")
        rolled_back = target.model_copy(update={"version": current.version + 1})
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO harness_versions VALUES (?, ?, ?, NULL)",
                (harness_id, rolled_back.version, rolled_back.model_dump_json()),
            )
            db.execute(
                "INSERT INTO promotion_events(event_json) VALUES (?)",
                (
                    json.dumps(
                        {
                            "action": "rollback",
                            "harness_id": harness_id,
                            "from_version": current.version,
                            "to_version": version,
                            "new_version": rolled_back.version,
                            "approver_id": approver_id,
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                        sort_keys=True,
                    ),
                ),
            )
        return rolled_back

    def add_experience(self, record: ExperienceRecord) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO experiences(experience_id, record_json) VALUES (?, ?)",
                (record.experience_id, record.model_dump_json()),
            )

    def record_experiment(self, report: ExperimentReport) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO experiments(experiment_id, report_json) VALUES (?, ?)",
                (report.experiment_id, report.model_dump_json()),
            )

    def get_experiment(self, experiment_id: str) -> ExperimentReport:
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT report_json FROM experiments WHERE experiment_id=?", (experiment_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown experiment: {experiment_id}")
        return ExperimentReport.model_validate_json(row[0])

    def experiences(
        self, *, task_scope: str, now: datetime | None = None
    ) -> list[ExperienceRecord]:
        now = now or datetime.now(UTC)
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT record_json FROM experiences").fetchall()
        return [
            record
            for row in rows
            if (record := ExperienceRecord.model_validate_json(row[0])).task_scope == task_scope
            and record.expires_at > now
        ]


def classify_failure(report: Any, *, task_scope: str) -> FailureClassification:
    metrics = report.metrics
    categories: list[str] = []
    if metrics.policy_violations:
        categories.append("policy_violation")
    if any(not row.task_success for row in report.cases):
        categories.append("mandatory_requirement_failure")
    if metrics.retries_total:
        categories.append("retry_detected")
    if metrics.tool_invocations_total > 2 * max(1, metrics.sample_count):
        categories.append("high_tool_volume")
    if not categories:
        categories.append("no_observed_failure")
    return FailureClassification(
        categories=tuple(categories), evidence_report_id=report.report_id, task_scope=task_scope
    )


def apply_patch(
    baseline: HarnessConfig,
    patch: HarnessPatch,
    *,
    approved_skill_ids: frozenset[str],
    authorized_tool_ids: frozenset[str],
    inference_budget_ceiling: int,
) -> HarnessConfig:
    if patch.base_harness_id != baseline.harness_id:
        raise ValueError("patch base harness does not match the baseline")
    skills = patch.skill_ids if patch.skill_ids is not None else baseline.skill_ids
    tools = (
        patch.tool_preferences if patch.tool_preferences is not None else baseline.tool_preferences
    )
    budget = patch.inference_budget_tokens or baseline.inference_budget_tokens
    if not set(skills).issubset(approved_skill_ids):
        raise ValueError("patch references a skill outside the approved skill set")
    if not set(tools).issubset(authorized_tool_ids):
        raise ValueError("patch references a tool outside existing authorization")
    if budget > inference_budget_ceiling:
        raise ValueError("patch exceeds the operator inference budget ceiling")
    return baseline.model_copy(
        update={
            "version": baseline.version + 1,
            "instruction_variant": patch.instruction_variant or baseline.instruction_variant,
            "skill_ids": skills,
            "tool_preferences": tools,
            "verification_steps": (
                patch.verification_steps
                if patch.verification_steps is not None
                else baseline.verification_steps
            ),
            "inference_budget_tokens": budget,
        }
    )


async def run_experiment(
    suite: BenchmarkSuite,
    *,
    baseline: HarnessConfig,
    patch: HarnessPatch,
    runner: EvaluationRunner,
    environment_id: str,
    approved_skill_ids: frozenset[str],
    authorized_tool_ids: frozenset[str],
    inference_budget_ceiling: int,
    trials: int = 3,
    noninferiority_margin: float = 0.0,
    max_cost_usd: float = float("inf"),
    max_latency_ms: float = float("inf"),
    registry: SQLiteHarnessRegistry | None = None,
    evaluation_store: SQLiteEvaluationStore | None = None,
) -> ExperimentReport:
    if baseline.inference_budget_tokens > inference_budget_ceiling:
        raise ValueError("baseline already exceeds the operator inference budget ceiling")
    candidate = apply_patch(
        baseline,
        patch,
        approved_skill_ids=approved_skill_ids,
        authorized_tool_ids=authorized_tool_ids,
        inference_budget_ceiling=inference_budget_ceiling,
    )
    comparison = await compare_harnesses(
        suite,
        baseline_id=f"{baseline.harness_id}@{baseline.version}",
        candidate_id=f"{candidate.harness_id}@{candidate.version}",
        runner=runner,
        environment_id=environment_id,
        trials=trials,
        illustrative=True,
        store=evaluation_store,
    )
    mandatory_pass = all(row.task_success for row in comparison.candidate.cases if row.mandatory)
    noninferior = (
        comparison.candidate.metrics.verified_quality_ci95[0]
        >= comparison.baseline.metrics.verified_quality_mean - noninferiority_margin
    )
    improved = (
        comparison.quality_delta_mean > 0
        or comparison.candidate.metrics.total_cost_usd < comparison.baseline.metrics.total_cost_usd
        or comparison.candidate.metrics.latency_median_ms
        < comparison.baseline.metrics.latency_median_ms
        or comparison.candidate.metrics.tool_invocations_total
        < comparison.baseline.metrics.tool_invocations_total
    )
    cost_ok = comparison.candidate.metrics.total_cost_usd <= max_cost_usd
    latency_ok = comparison.candidate.metrics.latency_p95_ms <= max_latency_ms
    checks = {
        "hard_safety": comparison.hard_safety_passed,
        "mandatory_cases": mandatory_pass,
        "quality_noninferiority": noninferior,
        "predeclared_improvement": improved,
        "cost_ceiling": cost_ok,
        "latency_ceiling": latency_ok,
    }
    reasons = tuple(key for key, value in checks.items() if not value)
    report_id = str(uuid4())
    patch_json = patch.model_dump_json()
    digest = hashlib.sha256((patch_json + comparison.model_dump_json()).encode("utf-8")).hexdigest()
    gates = ExperimentGates(passed=all(checks.values()), checks=checks, reasons=reasons)
    report = ExperimentReport(
        experiment_id=report_id,
        experiment_hash=digest,
        patch=patch,
        baseline_harness=baseline,
        candidate_harness=candidate,
        comparison=comparison,
        gates=gates,
        status="eligible" if gates.passed else "rejected",
    )
    if registry is not None:
        registry.record_experiment(report)
        registry.add_experience(
            ExperienceRecord(
                experiment_id=report.experiment_id,
                task_scope=baseline.task_scope,
                classification=classify_failure(
                    comparison.baseline, task_scope=baseline.task_scope
                ),
                evidence_report_ids=(
                    comparison.baseline.report_id,
                    comparison.candidate.report_id,
                ),
                confidence=min(1.0, comparison.paired_sample_count / 10),
                expires_at=datetime.now(UTC) + timedelta(days=30),
            )
        )
    return report
