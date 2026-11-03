"""Typed HTTP request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from atlas.observability import RunEvent
from atlas.plans.ir import Plan


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: Plan
    inputs: dict[str, Any] = Field(default_factory=dict)


class TaskCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=10_000)
    context: dict[str, Any] = Field(default_factory=dict)


class RunAcceptedResponse(BaseModel):
    run_id: str
    plan_hash: str
    status: Literal["queued"] = "queued"


class TaskAcceptedResponse(RunAcceptedResponse):
    plan: Plan


class RunStatusResponse(BaseModel):
    run_id: str
    plan_hash: str
    status: Literal["queued", "running", "succeeded", "failed", "timed_out", "cancelled"]
    created_at: datetime
    updated_at: datetime
    node_statuses: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class RunEventsResponse(BaseModel):
    run_id: str
    events: list[RunEvent]


class CancellationResponse(BaseModel):
    run_id: str
    status: Literal["cancelled", "already_finished", "not_active"]


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    checks: dict[str, str] = Field(default_factory=dict)
