"""Framework-neutral execution contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from atlas.agents import Agent
from atlas.plans.compiler import CompiledPlan
from atlas.plans.ir import AgentProfile, ToolDefinition
from atlas.tools import Tool


class ExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class AgentRegistration:
    profile: AgentProfile
    adapter: Agent


@dataclass(frozen=True)
class ToolRegistration:
    definition: ToolDefinition
    adapter: Tool


@dataclass(frozen=True)
class ExecutionResult:
    run_id: str
    plan_hash: str
    status: ExecutionStatus
    outputs: dict[str, Any]
    node_statuses: dict[str, str]
    attempts: dict[str, int]
    errors: dict[str, str]
    agent_traces: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    divergence_report: dict[str, Any] | None = None


class ExecutionBackend(Protocol):
    """Runs an already-compiled plan; implementations own orchestration details."""

    async def execute(
        self,
        plan: CompiledPlan,
        *,
        run_id: str,
        inputs: dict[str, Any] | None = None,
        resume: bool = False,
    ) -> ExecutionResult: ...
