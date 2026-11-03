import asyncio
from pathlib import Path

import pytest

from atlas.agents import FakeAgent
from atlas.plans.compiler import CompiledPlan, PlanCompiler
from atlas.plans.ir import AgentProfile, Node, NodeKind, Plan, ToolDefinition
from atlas.runtime.backend import AgentRegistration, ExecutionStatus, ToolRegistration
from atlas.runtime.langgraph_backend import LangGraphExecutionBackend
from atlas.security.policy import ToolPolicy


def tool_definition(risk_class: str = "read_only") -> ToolDefinition:
    return ToolDefinition(
        tool_id="fixture",
        version="1",
        input_schema={},
        output_schema={"type": "object"},
        risk_class=risk_class,
    )


def fanout_plan(definition: ToolDefinition, *, max_retries: int = 0) -> CompiledPlan:
    nodes = (
        Node(node_id="start", kind=NodeKind.JOIN, name="Start"),
        Node(
            node_id="left",
            kind=NodeKind.TOOL,
            name="Left branch",
            tool_ref="fixture",
            depends_on=("start",),
            input_mapping={"branch": "left"},
            output_schema={"type": "object"},
            max_retries=max_retries,
        ),
        Node(
            node_id="right",
            kind=NodeKind.TOOL,
            name="Right branch",
            tool_ref="fixture",
            depends_on=("start",),
            input_mapping={"branch": "right"},
            output_schema={"type": "object"},
        ),
        Node(
            node_id="merge",
            kind=NodeKind.JOIN,
            name="Join branches",
            depends_on=("left", "right"),
        ),
        Node(node_id="finish", kind=NodeKind.OUTPUT, name="Finish", depends_on=("merge",)),
    )
    plan = Plan(
        plan_id="runtime-fanout",
        task_id="runtime-test",
        nodes=nodes,
        max_parallelism=2,
    )
    return PlanCompiler(tools={"fixture": definition}).compile(plan)


class ConcurrentTool:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.calls = 0

    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        self.active += 1
        self.calls += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.05)
            return {"branch": arguments["mapped"]["branch"]}  # type: ignore[index]
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_fanout_runs_concurrently_and_join_waits(tmp_path: Path) -> None:
    definition = tool_definition()
    tool = ConcurrentTool()
    backend = LangGraphExecutionBackend(
        tool_registry={"fixture": ToolRegistration(definition, tool)},
        checkpoint_path=str(tmp_path / "parallel.sqlite"),
        max_parallelism=2,
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})),
    )

    result = await backend.execute(fanout_plan(definition), run_id="parallel-run")  # type: ignore[arg-type]

    assert result.status is ExecutionStatus.SUCCEEDED
    assert tool.peak == 2
    assert result.node_statuses["merge"] == "succeeded"
    assert result.outputs["merge"]["dependencies"] == {
        "left": {"branch": "left"},
        "right": {"branch": "right"},
    }


@pytest.mark.asyncio
async def test_replay_uses_recorded_tool_outputs_without_invoking_adapters(tmp_path: Path) -> None:
    definition = tool_definition()
    tool = ConcurrentTool()
    backend = LangGraphExecutionBackend(
        tool_registry={"fixture": ToolRegistration(definition, tool)},
        checkpoint_path=str(tmp_path / "replay.sqlite"),
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})),
    )
    plan = fanout_plan(definition)
    original = await backend.execute(plan, run_id="original-run")
    calls_after_original = tool.calls

    replayed = await backend.replay(plan, source_run_id="original-run", replay_run_id="replay-run")

    assert replayed.status is ExecutionStatus.SUCCEEDED
    assert tool.calls == calls_after_original
    assert replayed.outputs == original.outputs
    assert replayed.divergence_report == {
        "source_run_id": "original-run",
        "replay_run_id": "replay-run",
        "mode": "recorded_tool_outputs; agent adapters were re-executed",
        "matches": True,
        "divergences": {},
    }


@pytest.mark.asyncio
async def test_backend_concurrency_ceiling_is_enforced(tmp_path: Path) -> None:
    definition = tool_definition()
    tool = ConcurrentTool()
    backend = LangGraphExecutionBackend(
        tool_registry={"fixture": ToolRegistration(definition, tool)},
        checkpoint_path=str(tmp_path / "bounded.sqlite"),
        max_parallelism=1,
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})),
    )

    result = await backend.execute(fanout_plan(definition), run_id="bounded-run")

    assert result.status is ExecutionStatus.SUCCEEDED
    assert tool.peak == 1


class FailingBranchTool:
    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        mapped = arguments["mapped"]
        if mapped["branch"] == "left":  # type: ignore[index]
            raise RuntimeError("injected failure")
        return {"branch": "right"}


@pytest.mark.asyncio
async def test_failed_node_blocks_dependents(tmp_path: Path) -> None:
    definition = tool_definition()
    backend = LangGraphExecutionBackend(
        tool_registry={"fixture": ToolRegistration(definition, FailingBranchTool())},
        checkpoint_path=str(tmp_path / "failure.sqlite"),
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})),
    )

    result = await backend.execute(fanout_plan(definition), run_id="failure-run")  # type: ignore[arg-type]

    assert result.status is ExecutionStatus.FAILED
    assert result.node_statuses["left"] == "failed"
    assert result.node_statuses["right"] == "succeeded"
    assert result.node_statuses["merge"] == "blocked"
    assert result.node_statuses["finish"] == "blocked"


class RetryOnceTool:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {"left": 0, "right": 0}

    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        mapped = arguments["mapped"]
        branch = mapped["branch"]  # type: ignore[index]
        self.calls[branch] += 1  # type: ignore[index]
        if branch == "left" and self.calls[branch] == 1:  # type: ignore[index]
            raise RuntimeError("transient")
        return {"branch": branch}


@pytest.mark.asyncio
async def test_retry_is_bounded_and_counted_for_idempotent_tool(tmp_path: Path) -> None:
    definition = tool_definition("idempotent")
    tool = RetryOnceTool()
    backend = LangGraphExecutionBackend(
        tool_registry={"fixture": ToolRegistration(definition, tool)},
        checkpoint_path=str(tmp_path / "retry.sqlite"),
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})),
    )

    result = await backend.execute(
        fanout_plan(definition, max_retries=1),
        run_id="retry-run",  # type: ignore[arg-type]
    )

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.attempts["left"] == 2
    assert tool.calls == {"left": 2, "right": 1}


@pytest.mark.asyncio
async def test_side_effecting_tool_cannot_be_configured_for_retry(tmp_path: Path) -> None:
    definition = tool_definition("side_effecting")
    backend = LangGraphExecutionBackend(
        tool_registry={"fixture": ToolRegistration(definition, RetryOnceTool())},
        checkpoint_path=str(tmp_path / "side-effect.sqlite"),
    )

    with pytest.raises(ValueError, match="retries require an idempotent"):
        await backend.execute(
            fanout_plan(definition, max_retries=1),  # type: ignore[arg-type]
            run_id="side-effect-run",
        )


@pytest.mark.asyncio
async def test_side_effecting_tool_is_denied_until_policy_gateway(tmp_path: Path) -> None:
    definition = tool_definition("side_effecting")
    backend = LangGraphExecutionBackend(
        tool_registry={"fixture": ToolRegistration(definition, RetryOnceTool())},
        checkpoint_path=str(tmp_path / "side-effect-denied.sqlite"),
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})),
    )

    result = await backend.execute(fanout_plan(definition), run_id="side-effect-denied-run")
    assert result.node_statuses["left"] == "failed"


class SlowTool:
    def __init__(self) -> None:
        self.cancelled = False

    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return {"late": True}


@pytest.mark.asyncio
async def test_node_timeout_cancels_adapter_call(tmp_path: Path) -> None:
    definition = tool_definition()
    tool = SlowTool()
    nodes = (
        Node(
            node_id="slow",
            kind=NodeKind.TOOL,
            name="Slow",
            tool_ref="fixture",
            timeout_seconds=1,
            output_schema={"type": "object"},
        ),
        Node(node_id="finish", kind=NodeKind.OUTPUT, name="Finish", depends_on=("slow",)),
    )
    plan = Plan(plan_id="timeout", task_id="timeout", nodes=nodes)
    compiled = PlanCompiler(tools={"fixture": definition}).compile(plan)
    backend = LangGraphExecutionBackend(
        tool_registry={"fixture": ToolRegistration(definition, tool)},
        checkpoint_path=str(tmp_path / "timeout.sqlite"),
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})),
    )

    result = await backend.execute(compiled, run_id="timeout-run")

    assert result.status is ExecutionStatus.FAILED
    assert result.node_statuses["slow"] == "failed"
    assert tool.cancelled is True


@pytest.mark.asyncio
async def test_run_deadline_cancels_graph_and_reports_timeout(tmp_path: Path) -> None:
    definition = tool_definition()
    tool = SlowTool()
    plan = Plan(
        plan_id="run-deadline",
        task_id="run-deadline",
        nodes=(
            Node(
                node_id="slow",
                kind=NodeKind.TOOL,
                name="Slow",
                tool_ref="fixture",
                timeout_seconds=10,
                output_schema={"type": "object"},
            ),
            Node(node_id="finish", kind=NodeKind.OUTPUT, name="Finish", depends_on=("slow",)),
        ),
    )
    compiled = PlanCompiler(tools={"fixture": definition}).compile(plan)
    backend = LangGraphExecutionBackend(
        tool_registry={"fixture": ToolRegistration(definition, tool)},
        checkpoint_path=str(tmp_path / "run-deadline.sqlite"),
        run_timeout_seconds=0.05,
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})),
    )

    result = await backend.execute(compiled, run_id="run-deadline")

    assert result.status is ExecutionStatus.TIMED_OUT
    assert tool.cancelled is True


class CrashWindowTool:
    def __init__(self) -> None:
        self.calls = {"left": 0, "right": 0}
        self.left_complete = asyncio.Event()
        self.right_started = asyncio.Event()
        self.release_right = asyncio.Event()

    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        mapped = arguments["mapped"]
        branch = mapped["branch"]  # type: ignore[index]
        self.calls[branch] += 1  # type: ignore[index]
        if branch == "left":
            self.left_complete.set()
            return {"branch": "left"}
        self.right_started.set()
        if self.calls[branch] == 1:  # type: ignore[index]
            await self.release_right.wait()
        return {"branch": "right"}


@pytest.mark.asyncio
async def test_resume_restores_completed_parallel_node_without_repeating_it(
    tmp_path: Path,
) -> None:
    definition = tool_definition()
    tool = CrashWindowTool()
    database = str(tmp_path / "recovery.sqlite")
    backend = LangGraphExecutionBackend(
        tool_registry={"fixture": ToolRegistration(definition, tool)},
        checkpoint_path=database,
        max_parallelism=2,
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})),
    )
    plan = fanout_plan(definition)

    in_flight = asyncio.create_task(backend.execute(plan, run_id="recoverable-run"))
    await asyncio.wait_for(tool.right_started.wait(), timeout=2)
    await asyncio.wait_for(tool.left_complete.wait(), timeout=2)
    await asyncio.sleep(0.1)  # allow the checkpointer to persist left's pending write
    in_flight.cancel()
    with pytest.raises(asyncio.CancelledError):
        await in_flight

    tool.release_right.set()
    restarted_backend = LangGraphExecutionBackend(
        tool_registry={"fixture": ToolRegistration(definition, tool)},
        checkpoint_path=database,
        max_parallelism=2,
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})),
    )
    resumed = await restarted_backend.execute(plan, run_id="recoverable-run", resume=True)

    assert resumed.status is ExecutionStatus.SUCCEEDED
    assert tool.calls == {"left": 1, "right": 2}


@pytest.mark.asyncio
async def test_run_id_cannot_be_reused_for_a_fresh_execution(tmp_path: Path) -> None:
    definition = tool_definition()
    tool = ConcurrentTool()
    backend = LangGraphExecutionBackend(
        tool_registry={"fixture": ToolRegistration(definition, tool)},
        checkpoint_path=str(tmp_path / "run-id.sqlite"),
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"fixture"})),
    )
    plan = fanout_plan(definition)
    await backend.execute(plan, run_id="unique-run")

    with pytest.raises(ValueError, match="already exists"):
        await backend.execute(plan, run_id="unique-run")

    resumed = await backend.execute(plan, run_id="unique-run", resume=True)
    assert resumed.status is ExecutionStatus.SUCCEEDED
    assert tool.calls == 2


@pytest.mark.asyncio
async def test_unsupported_conditionals_fail_closed(tmp_path: Path) -> None:
    definition = tool_definition()
    raw = fanout_plan(definition).plan.model_dump(mode="json")
    raw["edges"] = [{"source": "start", "target": "left", "condition": "ready"}]
    compiled = PlanCompiler(tools={"fixture": definition}).compile(raw)
    backend = LangGraphExecutionBackend(checkpoint_path=str(tmp_path / "conditional.sqlite"))

    with pytest.raises(ValueError, match="conditional edges"):
        await backend.execute(compiled, run_id="conditional-run")


@pytest.mark.asyncio
async def test_agent_adapter_uses_registered_profile_and_validates_output(tmp_path: Path) -> None:
    profile = AgentProfile(
        agent_id="fake-agent",
        version="1",
        input_schema={"type": "object", "required": ["run", "dependencies", "mapped"]},
        output_schema={"type": "string"},
    )
    plan = Plan(
        plan_id="agent-plan",
        task_id="agent-task",
        nodes=(
            Node(
                node_id="agent",
                kind=NodeKind.AGENT,
                name="Summarize",
                agent_ref="fake-agent",
                output_schema={"type": "string"},
            ),
            Node(node_id="finish", kind=NodeKind.OUTPUT, name="Finish", depends_on=("agent",)),
        ),
    )
    compiled = PlanCompiler(agents={"fake-agent": profile}).compile(plan)
    backend = LangGraphExecutionBackend(
        agent_registry={"fake-agent": AgentRegistration(profile, FakeAgent("done"))},
        checkpoint_path=str(tmp_path / "agent.sqlite"),
    )

    result = await backend.execute(compiled, run_id="agent-run", inputs={"task": "hello"})

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.outputs["agent"] == "done"
