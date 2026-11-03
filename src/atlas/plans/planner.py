"""Provider-independent planning contracts and deterministic fixtures."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atlas.plans.ir import Edge, Node, NodeKind, Plan


class PlannerRole(BaseModel):
    """A trusted role that a planner may reference by ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role_id: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)


class PlannerLimits(BaseModel):
    """Operator ceilings supplied to a planner; returned limits cannot exceed these."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_nodes: int = Field(default=8, ge=2, le=32)
    max_depth: int = Field(default=5, ge=1, le=16)
    max_parallelism: int = Field(default=4, ge=1, le=16)
    max_task_description_chars: int = Field(default=1000, ge=1, le=4000)


class PlannerRequest(BaseModel):
    """Provider-neutral task planning request containing no mutable runtime capabilities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: str = Field(min_length=1, max_length=10_000)
    context: dict[str, Any] = Field(default_factory=dict)
    available_roles: tuple[PlannerRole, ...] = Field(min_length=1, max_length=32)
    limits: PlannerLimits = Field(default_factory=PlannerLimits)


class PlannedNode(BaseModel):
    """A requested node. Its role ID is resolved only through trusted templates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    role_id: str = Field(min_length=1, max_length=80)
    task_description: str = Field(min_length=1, max_length=4000)


class PlannedEdge(BaseModel):
    """A dependency edge between planned node IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    target: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")


class WorkflowProposal(BaseModel):
    """Bounded graph structure returned by a Planner; no tools or permissions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nodes: tuple[PlannedNode, ...] = Field(min_length=1, max_length=31)
    edges: tuple[PlannedEdge, ...] = Field(default=(), max_length=128)
    max_depth: int = Field(ge=1, le=16)
    max_parallelism: int = Field(ge=1, le=16)

    @model_validator(mode="after")
    def unique_ids(self) -> WorkflowProposal:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("planned node IDs must be unique")
        if len({(edge.source, edge.target) for edge in self.edges}) != len(self.edges):
            raise ValueError("planned dependency edges must be unique")
        return self


class Planner(Protocol):
    async def plan(self, request: PlannerRequest) -> WorkflowProposal: ...


class FixturePlanner:
    """Produces a fixed valid graph; this is not an LLM/provider integration."""

    def sequential(self, task_id: str = "fixture-task") -> Plan:
        return Plan(
            plan_id="fixture-sequential-v1",
            task_id=task_id,
            nodes=(
                Node(node_id="start", kind=NodeKind.JOIN, name="Start"),
                Node(node_id="finish", kind=NodeKind.OUTPUT, name="Finish", depends_on=("start",)),
            ),
        )

    def approval_flow(self, task_id: str = "fixture-task") -> Plan:
        return Plan(
            plan_id="fixture-approval-v1",
            task_id=task_id,
            nodes=(
                Node(node_id="review", kind=NodeKind.HUMAN_APPROVAL, name="Approval checkpoint"),
                Node(
                    node_id="gate",
                    kind=NodeKind.CONDITION,
                    name="Approval decision",
                    depends_on=("review",),
                ),
                Node(node_id="finish", kind=NodeKind.OUTPUT, name="Finish", depends_on=("gate",)),
            ),
            edges=(Edge(source="review", target="gate", condition="approved"),),
        )
