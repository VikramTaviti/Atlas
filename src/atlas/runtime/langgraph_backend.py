"""LangGraph implementation of ATLAS's framework-neutral execution backend."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import Annotated, Any, TypedDict, cast

from jsonschema import validate

from atlas.observability import (
    ArtifactStore,
    LocalArtifactStore,
    OpenTelemetrySpanFactory,
    RunEvent,
    RunEventStore,
    SQLiteRunEventStore,
    json_artifact,
)
from atlas.plans.compiler import CompiledPlan
from atlas.plans.ir import Node, NodeKind
from atlas.react import ReActAgentRunner
from atlas.runtime.backend import (
    AgentRegistration,
    ExecutionResult,
    ExecutionStatus,
    ToolRegistration,
)
from atlas.security.policy import (
    ApprovalProvider,
    SQLiteToolAuditStore,
    ToolGatewaySession,
    ToolPolicy,
)
from atlas.security.redaction import SecretRedactor


def _merge_maps(current: dict[str, Any] | None, updates: dict[str, Any] | None) -> dict[str, Any]:
    """Merge independent node updates to LangGraph's shared state channels."""
    return {**(current or {}), **(updates or {})}


class WorkflowState(TypedDict):
    inputs: dict[str, Any]
    outputs: Annotated[dict[str, Any], _merge_maps]
    statuses: Annotated[dict[str, str], _merge_maps]
    attempts: Annotated[dict[str, int], _merge_maps]
    errors: Annotated[dict[str, str], _merge_maps]
    agent_traces: Annotated[dict[str, list[dict[str, Any]]], _merge_maps]


class LangGraphExecutionBackend:
    """Execute compiled ATLAS DAGs with LangGraph and SQLite checkpoints.

    LangGraph is loaded lazily so importing ATLAS's core contracts does not require
    the optional backend extra. One SQLite checkpoint database may serve multiple
    runs; each run uses its own LangGraph thread ID.
    """

    def __init__(
        self,
        *,
        agent_registry: Mapping[str, AgentRegistration] | None = None,
        tool_registry: Mapping[str, ToolRegistration] | None = None,
        checkpoint_path: str = "atlas-checkpoints.sqlite",
        max_parallelism: int = 8,
        run_timeout_seconds: float | None = None,
        max_react_iterations: int = 16,
        max_tool_calls_per_iteration: int = 4,
        max_total_tool_calls_per_agent: int = 16,
        tool_policy: ToolPolicy | None = None,
        approval_provider: ApprovalProvider | None = None,
        audit_path: str | None = None,
        redactor: SecretRedactor | None = None,
        event_path: str | None = None,
        artifact_root: str | None = None,
        event_store: RunEventStore | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        if max_parallelism < 1:
            raise ValueError("max_parallelism must be positive")
        if run_timeout_seconds is not None and run_timeout_seconds <= 0:
            raise ValueError("run_timeout_seconds must be positive")
        self.agent_registry = dict(agent_registry or {})
        self.tool_registry = dict(tool_registry or {})
        self.checkpoint_path = checkpoint_path
        self.max_parallelism = max_parallelism
        self.run_timeout_seconds = run_timeout_seconds
        self.tool_policy = tool_policy or ToolPolicy()
        self.approval_provider = approval_provider
        self.audit_store = SQLiteToolAuditStore(audit_path or f"{checkpoint_path}.audit.sqlite")
        self.event_store = event_store or SQLiteRunEventStore(
            event_path or f"{checkpoint_path}.events.sqlite"
        )
        self.artifact_store = artifact_store or LocalArtifactStore(
            artifact_root or f"{checkpoint_path}.artifacts"
        )
        self.span_factory = OpenTelemetrySpanFactory()
        self.react_runner = ReActAgentRunner(
            self.tool_registry,
            max_iterations_cap=max_react_iterations,
            max_tool_calls_per_iteration=max_tool_calls_per_iteration,
            max_total_tool_calls=max_total_tool_calls_per_agent,
            redactor=redactor,
        )

    async def execute(
        self,
        plan: CompiledPlan,
        *,
        run_id: str,
        inputs: dict[str, Any] | None = None,
        resume: bool = False,
        replay_from_run_id: str | None = None,
    ) -> ExecutionResult:
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        self._validate_supported_graph(plan)
        gateway = ToolGatewaySession(
            run_id=run_id,
            plan_hash=plan.content_hash,
            tool_registry=self.tool_registry,
            policy=self.tool_policy,
            audit_store=self.audit_store,
            approval_provider=self.approval_provider,
            event_store=self.event_store,
            artifact_store=self.artifact_store,
            replay_from_run_id=replay_from_run_id,
        )
        initial_inputs = inputs or {}
        try:
            json.dumps(initial_inputs, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("run inputs must be JSON-serializable") from exc

        run_artifact = await self.artifact_store.put(
            json_artifact({"plan": json.loads(plan.canonical_json), "inputs": initial_inputs}),
            media_type="application/vnd.atlas.run-input+json",
        )
        await self.event_store.append(
            RunEvent(
                run_id=run_id,
                plan_hash=plan.content_hash,
                event_type="run.resumed" if resume else "run.started",
                status="running",
                artifact_refs=(run_artifact,),
                attributes={"replay_from_run_id": replay_from_run_id},
            )
        )

        try:
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph backend dependencies are missing; install atlas-agent-runtime[langgraph]"
            ) from exc

        async with AsyncSqliteSaver.from_conn_string(self.checkpoint_path) as checkpointer:
            checkpointer.serde = JsonPlusSerializer(allowed_msgpack_modules=[])
            await checkpointer.setup()
            graph = self._build_graph(
                plan, run_id, StateGraph, START, END, gateway, plan.content_hash
            ).compile(checkpointer=checkpointer)
            config = {
                "configurable": {"thread_id": run_id},
                "max_concurrency": min(plan.plan.max_parallelism, self.max_parallelism),
                "recursion_limit": max(25, plan.plan.max_depth + 1),
            }
            existing_checkpoint = await checkpointer.aget_tuple(cast(Any, config))
            if resume and existing_checkpoint is None:
                raise ValueError(f"run '{run_id}' has no checkpoint to resume")
            if not resume and existing_checkpoint is not None:
                raise ValueError(f"run '{run_id}' already exists; use resume=True or a new run_id")
            if resume and inputs is not None:
                raise ValueError("inputs cannot be supplied when resuming a checkpointed run")
            graph_input: WorkflowState | None
            if resume:
                graph_input = None
            else:
                graph_input = {
                    "inputs": initial_inputs,
                    "outputs": {},
                    "statuses": {node.node_id: "pending" for node in plan.plan.nodes},
                    "attempts": {},
                    "errors": {},
                    "agent_traces": {},
                }
            try:
                with self.span_factory.span(
                    "atlas.run", {"atlas.run_id": run_id, "atlas.plan_hash": plan.content_hash}
                ):
                    if self.run_timeout_seconds is None:
                        state = await graph.ainvoke(graph_input, config=cast(Any, config))
                    else:
                        async with asyncio.timeout(self.run_timeout_seconds):
                            state = await graph.ainvoke(graph_input, config=cast(Any, config))
            except TimeoutError:
                snapshot = await graph.aget_state(cast(Any, config))
                state = snapshot.values
                if not state:
                    state = graph_input or {}
                result = self._result(run_id, plan, state, ExecutionStatus.TIMED_OUT)
                await self._record_run_result(result)
                return result

        result = self._result(run_id, plan, state)
        await self._record_run_result(result)
        return result

    async def replay(
        self,
        plan: CompiledPlan,
        *,
        source_run_id: str,
        replay_run_id: str,
    ) -> ExecutionResult:
        """Re-run orchestration with tool outputs loaded from artifacts, never tool adapters."""
        source_events = await self.event_store.list_run(source_run_id)
        source_start = next(
            (event for event in source_events if event.event_type == "run.started"), None
        )
        source_finish = next(
            (event for event in reversed(source_events) if event.event_type == "run.completed"),
            None,
        )
        if source_start is None or not source_start.artifact_refs:
            raise ValueError(f"source run '{source_run_id}' has no saved run inputs")
        if source_start.plan_hash != plan.content_hash:
            raise ValueError("replay plan hash does not match the source run")
        saved = json.loads(await self.artifact_store.get(source_start.artifact_refs[0]))
        if source_finish is None or not source_finish.artifact_refs:
            raise ValueError(f"source run '{source_run_id}' has no completed output artifact")
        replayed = await self.execute(
            plan,
            run_id=replay_run_id,
            inputs=saved["inputs"],
            replay_from_run_id=source_run_id,
        )
        original_outputs = json.loads(await self.artifact_store.get(source_finish.artifact_refs[0]))
        divergences = {
            key: {"original": original_outputs.get(key), "replayed": replayed.outputs.get(key)}
            for key in sorted(set(original_outputs) | set(replayed.outputs))
            if original_outputs.get(key) != replayed.outputs.get(key)
        }
        replay_events = await self.event_store.list_run(replay_run_id)
        tool_input_mismatches = [
            {"node_id": event.node_id, "tool_id": event.tool_id}
            for event in replay_events
            if event.event_type == "tool.replayed"
            and not event.attributes.get("replay_input_matches", False)
        ]
        if tool_input_mismatches:
            divergences["tool_inputs"] = {"mismatches": tool_input_mismatches}
        report = {
            "source_run_id": source_run_id,
            "replay_run_id": replay_run_id,
            "mode": "recorded_tool_outputs; agent adapters were re-executed",
            "matches": not divergences,
            "divergences": divergences,
        }
        return replace(replayed, divergence_report=report)

    def _build_graph(
        self,
        plan: CompiledPlan,
        run_id: str,
        state_graph: Any,
        start: str,
        end: str,
        gateway: ToolGatewaySession,
        plan_hash: str,
    ) -> Any:
        builder = state_graph(WorkflowState)
        node_ids = [node.node_id for node in plan.plan.nodes]
        graph_names = {node_id: f"atlas_node_{index}" for index, node_id in enumerate(node_ids)}
        parents: dict[str, set[str]] = defaultdict(set)
        for node in plan.plan.nodes:
            parents[node.node_id].update(node.depends_on)
        for edge in plan.plan.edges:
            parents[edge.target].add(edge.source)

        for node in plan.plan.nodes:

            async def run_node(state: WorkflowState, current: Node = node) -> dict[str, Any]:
                return await self._run_node(
                    current,
                    state,
                    parents[current.node_id],
                    run_id=run_id,
                    gateway=gateway,
                    plan_hash=plan_hash,
                )

            builder.add_node(graph_names[node.node_id], run_node)

        for node_id in node_ids:
            incoming = sorted(parents[node_id])
            if not incoming:
                builder.add_edge(start, graph_names[node_id])
            else:
                builder.add_edge([graph_names[parent] for parent in incoming], graph_names[node_id])
        for node in plan.plan.nodes:
            if node.kind is NodeKind.OUTPUT:
                builder.add_edge(graph_names[node.node_id], end)
        return builder

    def _validate_supported_graph(self, plan: CompiledPlan) -> None:
        for edge in plan.plan.edges:
            if edge.condition is not None:
                raise ValueError("conditional edges are not executable in Milestone 2")
        for node in plan.plan.nodes:
            if node.kind in {NodeKind.CONDITION, NodeKind.HUMAN_APPROVAL}:
                raise ValueError(f"node kind '{node.kind.value}' is not executable in Milestone 2")
            if node.kind is NodeKind.AGENT:
                agent_registration = self.agent_registry.get(node.agent_ref or "")
                if agent_registration is None:
                    raise ValueError(f"agent '{node.agent_ref}' has no runtime adapter")
                if agent_registration.profile.agent_id != node.agent_ref:
                    raise ValueError(f"agent registry mismatch for '{node.agent_ref}'")
                for tool_id in agent_registration.profile.allowed_tools:
                    allowed_tool = self.tool_registry.get(tool_id)
                    if allowed_tool is None:
                        raise ValueError(
                            f"agent '{node.agent_ref}' allows unregistered tool '{tool_id}'"
                        )
                    if allowed_tool.definition.tool_id != tool_id:
                        raise ValueError(f"tool registry mismatch for agent capability '{tool_id}'")
                    if (
                        node.max_retries
                        and allowed_tool.definition.risk_class.value != "idempotent"
                    ):
                        raise ValueError("node retries require an idempotent agent tool")
            elif node.kind is NodeKind.TOOL:
                tool_registration = self.tool_registry.get(node.tool_ref or "")
                if tool_registration is None:
                    raise ValueError(f"tool '{node.tool_ref}' has no runtime adapter")
                if tool_registration.definition.tool_id != node.tool_ref:
                    raise ValueError(f"tool registry mismatch for '{node.tool_ref}'")
                if (
                    node.max_retries
                    and tool_registration.definition.risk_class.value != "idempotent"
                ):
                    raise ValueError("node retries require an idempotent tool")

    async def _run_node(
        self,
        node: Node,
        state: WorkflowState,
        parents: set[str],
        *,
        run_id: str,
        gateway: ToolGatewaySession,
        plan_hash: str,
    ) -> dict[str, Any]:
        blocked_by = sorted(
            parent for parent in parents if state["statuses"].get(parent) != "succeeded"
        )
        if blocked_by:
            await self._append_event(
                run_id=run_id,
                plan_hash=plan_hash,
                event_type="node.blocked",
                node_id=node.node_id,
                status="blocked",
                attributes={"blocked_by": blocked_by},
            )
            return {
                "statuses": {node.node_id: "blocked"},
                "errors": {node.node_id: f"blocked by unsuccessful dependencies: {blocked_by}"},
            }

        dependency_outputs = {parent: state["outputs"][parent] for parent in sorted(parents)}
        mapped = self._plain_json(node.input_mapping)
        node_inputs = {
            "run": state["inputs"],
            "dependencies": dependency_outputs,
            "mapped": mapped,
        }
        trace_events: list[dict[str, Any]] = []
        node_input_ref = await self.artifact_store.put(
            json_artifact(node_inputs), media_type="application/vnd.atlas.node-input+json"
        )
        await self._append_event(
            run_id=run_id,
            plan_hash=plan_hash,
            event_type="node.started",
            node_id=node.node_id,
            status="running",
            artifact_refs=(node_input_ref,),
        )
        last_error: Exception | None = None
        attempt = 0
        try:
            async with asyncio.timeout(node.timeout_seconds):
                for attempt in range(1, node.max_retries + 2):
                    try:
                        with self.span_factory.span(
                            "atlas.node",
                            {
                                "atlas.run_id": run_id,
                                "atlas.node_id": node.node_id,
                                "atlas.node_kind": node.kind.value,
                            },
                        ):
                            output = await self._invoke(
                                node,
                                node_inputs,
                                run_id=run_id,
                                trace_sink=trace_events,
                                attempt=attempt,
                                gateway=gateway,
                            )
                        validate(instance=output, schema=self._plain_json(node.output_schema))
                        json.dumps(output, ensure_ascii=False)
                        output_ref = await self.artifact_store.put(
                            json_artifact(output),
                            media_type="application/vnd.atlas.node-output+json",
                        )
                        await self._append_event(
                            run_id=run_id,
                            plan_hash=plan_hash,
                            event_type="node.completed",
                            node_id=node.node_id,
                            status="succeeded",
                            attempt=attempt,
                            artifact_refs=(output_ref,),
                        )
                        update: dict[str, Any] = {
                            "outputs": {node.node_id: output},
                            "statuses": {node.node_id: "succeeded"},
                            "attempts": {node.node_id: attempt},
                        }
                        if node.kind is NodeKind.AGENT:
                            self.react_runner.record_node_outcome(
                                trace_events,
                                run_id=run_id,
                                node_id=node.node_id,
                                profile=self.agent_registry[node.agent_ref or ""].profile,
                                attempt=attempt,
                                success=True,
                            )
                        if trace_events:
                            update["agent_traces"] = {node.node_id: trace_events}
                        return update
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        last_error = exc
                        if attempt <= node.max_retries and node.retry_delay_seconds:
                            await asyncio.sleep(node.retry_delay_seconds)
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            last_error = exc
        if node.kind is NodeKind.AGENT:
            self.react_runner.record_node_outcome(
                trace_events,
                run_id=run_id,
                node_id=node.node_id,
                profile=self.agent_registry[node.agent_ref or ""].profile,
                attempt=attempt,
                success=False,
                error_type=type(last_error).__name__ if last_error else "AgentExecutionError",
            )
        update = {
            "statuses": {node.node_id: "failed"},
            "attempts": {node.node_id: attempt},
            "errors": {
                node.node_id: f"{type(last_error).__name__}: {last_error}"
                if last_error
                else "node failed without an error"
            },
        }
        await self._append_event(
            run_id=run_id,
            plan_hash=plan_hash,
            event_type="node.failed",
            node_id=node.node_id,
            status="failed",
            attempt=attempt,
            attributes={"error_type": type(last_error).__name__ if last_error else "Unknown"},
        )
        if trace_events:
            update["agent_traces"] = {node.node_id: trace_events}
        return update

    async def _append_event(
        self,
        *,
        run_id: str,
        plan_hash: str,
        event_type: str,
        node_id: str | None = None,
        status: str | None = None,
        attempt: int | None = None,
        duration_ms: float | None = None,
        artifact_refs: tuple[Any, ...] = (),
        attributes: dict[str, Any] | None = None,
    ) -> None:
        await self.event_store.append(
            RunEvent(
                run_id=run_id,
                plan_hash=plan_hash,
                event_type=event_type,
                node_id=node_id,
                status=status,
                attempt=attempt,
                duration_ms=duration_ms,
                artifact_refs=artifact_refs,
                attributes=attributes or {},
            )
        )

    async def _record_run_result(self, result: ExecutionResult) -> None:
        output_ref = await self.artifact_store.put(
            json_artifact(result.outputs),
            media_type="application/vnd.atlas.run-output+json",
        )
        await self._append_event(
            run_id=result.run_id,
            plan_hash=result.plan_hash,
            event_type="run.completed",
            status=result.status.value,
            artifact_refs=(output_ref,),
            attributes={
                "node_statuses": result.node_statuses,
                "attempts": result.attempts,
            },
        )

    async def _invoke(
        self,
        node: Node,
        node_inputs: dict[str, Any],
        *,
        run_id: str,
        trace_sink: list[dict[str, Any]],
        attempt: int,
        gateway: ToolGatewaySession,
    ) -> Any:
        if node.kind is NodeKind.AGENT:
            agent_registration = self.agent_registry[node.agent_ref or ""]
            validate(
                instance=node_inputs,
                schema=self._plain_json(agent_registration.profile.input_schema),
            )
            return await self.react_runner.run(
                run_id=run_id,
                node_id=node.node_id,
                instruction=node.name,
                input_json=json.dumps(node_inputs),
                profile=agent_registration.profile,
                agent=agent_registration.adapter,
                trace_sink=trace_sink,
                attempt=attempt,
                gateway=gateway,
            )
        if node.kind is NodeKind.TOOL:
            output = await gateway.invoke(
                node_id=node.node_id,
                tool_id=node.tool_ref or "",
                arguments=node_inputs,
                required_capabilities=node.required_capabilities,
                attempt=attempt,
            )
            return output
        if node.kind in {NodeKind.JOIN, NodeKind.OUTPUT}:
            return node_inputs
        raise ValueError(f"unsupported node kind '{node.kind.value}'")

    @staticmethod
    def _plain_json(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: LangGraphExecutionBackend._plain_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [LangGraphExecutionBackend._plain_json(item) for item in value]
        return value

    @staticmethod
    def _result(
        run_id: str,
        plan: CompiledPlan,
        state: Mapping[str, Any],
        forced_status: ExecutionStatus | None = None,
    ) -> ExecutionResult:
        statuses = dict(state.get("statuses", {}))
        status = forced_status or (
            ExecutionStatus.FAILED
            if any(value in {"failed", "blocked"} for value in statuses.values())
            else ExecutionStatus.SUCCEEDED
        )
        return ExecutionResult(
            run_id=run_id,
            plan_hash=plan.content_hash,
            status=status,
            outputs=dict(state.get("outputs", {})),
            node_statuses=statuses,
            attempts=dict(state.get("attempts", {})),
            errors=dict(state.get("errors", {})),
            agent_traces=dict(state.get("agent_traces", {})),
        )
