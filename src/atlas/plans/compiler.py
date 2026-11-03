"""Deterministic plan validation and immutable compilation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import ValidationError

from atlas.plans.ir import AgentProfile, NodeKind, Plan, ToolDefinition


class PlanValidationError(ValueError):
    """A plan could not be accepted as safe, executable IR."""


@dataclass(frozen=True)
class CompiledPlan:
    plan: Plan
    canonical_json: str
    content_hash: str
    topological_order: tuple[str, ...]


class PlanCompiler:
    def __init__(
        self,
        agents: Mapping[str, AgentProfile] | None = None,
        tools: Mapping[str, ToolDefinition] | None = None,
    ) -> None:
        self.agents = agents or {}
        self.tools = tools or {}

    def compile(self, payload: Plan | dict[str, object]) -> CompiledPlan:
        try:
            plan = payload if isinstance(payload, Plan) else Plan.model_validate(payload)
        except ValidationError as exc:
            raise PlanValidationError(str(exc)) from exc

        ids = [node.node_id for node in plan.nodes]
        if len(ids) != len(set(ids)):
            raise PlanValidationError("duplicate node IDs are not allowed")
        if len(plan.nodes) > plan.max_nodes:
            raise PlanValidationError(
                f"graph has {len(plan.nodes)} nodes; limit is {plan.max_nodes}"
            )

        adjacency: dict[str, set[str]] = defaultdict(set)
        indegree = {node_id: 0 for node_id in ids}
        known = set(ids)
        for node in plan.nodes:
            for dependency in node.depends_on:
                self._add_edge(node.node_id, dependency, known, adjacency, indegree)
        for edge in plan.edges:
            self._add_edge(edge.target, edge.source, known, adjacency, indegree)

        nodes = {node.node_id: node for node in plan.nodes}
        for node in plan.nodes:
            if node.kind is NodeKind.AGENT:
                if not node.agent_ref or node.agent_ref not in self.agents:
                    raise PlanValidationError(
                        f"node '{node.node_id}' references unknown agent '{node.agent_ref}'"
                    )
                profile = self.agents[node.agent_ref]
                if profile.agent_id != node.agent_ref:
                    raise PlanValidationError(
                        f"agent registry key '{node.agent_ref}' does not match profile ID"
                    )
                self._check_capabilities(
                    node.node_id, node.required_capabilities, profile.capabilities
                )
            elif node.kind is NodeKind.TOOL:
                if not node.tool_ref or node.tool_ref not in self.tools:
                    raise PlanValidationError(
                        f"node '{node.node_id}' references unknown tool '{node.tool_ref}'"
                    )
                definition = self.tools[node.tool_ref]
                if definition.tool_id != node.tool_ref:
                    raise PlanValidationError(
                        f"tool registry key '{node.tool_ref}' does not match definition ID"
                    )
                self._check_capabilities(
                    node.node_id, node.required_capabilities, definition.capabilities
                )
            elif node.agent_ref or node.tool_ref:
                raise PlanValidationError(
                    f"node '{node.node_id}' kind '{node.kind}' cannot declare agent/tool refs"
                )
            elif node.required_capabilities:
                raise PlanValidationError(
                    f"node '{node.node_id}' has capabilities but no registered agent/tool"
                )

        order, depth, parallelism = self._topology(ids, adjacency, indegree)
        if len(order) != len(ids):
            raise PlanValidationError("graph contains a cycle")
        if depth > plan.max_depth:
            raise PlanValidationError(f"graph depth {depth} exceeds limit {plan.max_depth}")
        if parallelism > plan.max_parallelism:
            raise PlanValidationError(
                "unbounded parallelism: "
                f"{parallelism} concurrently ready nodes exceeds limit {plan.max_parallelism}"
            )

        outputs = {node_id for node_id, node in nodes.items() if node.kind is NodeKind.OUTPUT}
        if not outputs:
            raise PlanValidationError("graph must contain a reachable output node")
        reachable = self._reachable_from_roots(ids, adjacency, indegree)
        if not outputs.intersection(reachable):
            raise PlanValidationError("no output node is reachable from a graph root")

        canonical = json.dumps(
            plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return CompiledPlan(
            plan=plan, canonical_json=canonical, content_hash=digest, topological_order=tuple(order)
        )

    @staticmethod
    def _check_capabilities(
        node_id: str, requested: tuple[str, ...], declared: tuple[str, ...]
    ) -> None:
        missing = sorted(set(requested) - set(declared))
        if missing:
            raise PlanValidationError(
                f"node '{node_id}' requests undeclared capabilities: {', '.join(missing)}"
            )

    @staticmethod
    def _add_edge(
        target: str,
        source: str,
        known: set[str],
        adjacency: dict[str, set[str]],
        indegree: dict[str, int],
    ) -> None:
        if target not in known:
            raise PlanValidationError(f"dangling edge target '{target}'")
        if source not in known:
            raise PlanValidationError(f"dangling edge source '{source}'")
        if source == target:
            raise PlanValidationError(f"self dependency at node '{source}'")
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1

    @staticmethod
    def _topology(
        ids: list[str], adjacency: dict[str, set[str]], indegree: dict[str, int]
    ) -> tuple[list[str], int, int]:
        remaining = dict(indegree)
        ready = deque(sorted(node_id for node_id, degree in remaining.items() if degree == 0))
        max_ready = len(ready)
        order: list[str] = []
        depths = {node_id: 1 for node_id in ready}
        while ready:
            node_id = ready.popleft()
            order.append(node_id)
            for child in sorted(adjacency[node_id]):
                depths[child] = max(depths.get(child, 1), depths[node_id] + 1)
                remaining[child] -= 1
                if remaining[child] == 0:
                    ready.append(child)
            max_ready = max(max_ready, len(ready))
        return order, max(depths.values(), default=0), max_ready

    @staticmethod
    def _reachable_from_roots(
        ids: list[str], adjacency: dict[str, set[str]], indegree: dict[str, int]
    ) -> set[str]:
        roots = [node_id for node_id in ids if indegree[node_id] == 0]
        seen = set(roots)
        pending = list(roots)
        while pending:
            current = pending.pop()
            for child in adjacency[current] - seen:
                seen.add(child)
                pending.append(child)
        return seen
