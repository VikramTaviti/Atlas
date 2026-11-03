"""In-process asynchronous run admission and execution service."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from atlas.observability import ArtifactStore, RunEvent, SQLiteRunEventStore, json_artifact
from atlas.plans.compiler import CompiledPlan, PlanCompiler
from atlas.plans.ir import Plan
from atlas.runtime.backend import AgentRegistration, ToolRegistration
from atlas.runtime.langgraph_backend import LangGraphExecutionBackend
from atlas.security.policy import ToolPolicy

logger = logging.getLogger("atlas.api")
RunExecutor = Callable[[CompiledPlan, str, dict[str, object]], Awaitable[None]]


class RunNotFound(KeyError):
    """No persisted run exists for this identifier."""


class RunQueueFull(RuntimeError):
    """The service has reached its configured in-process admitted-run limit."""


class RunService:
    def __init__(
        self,
        *,
        data_dir: Path,
        events: SQLiteRunEventStore,
        artifacts: ArtifactStore,
        backend: LangGraphExecutionBackend,
        compiler: PlanCompiler,
        max_concurrent_runs: int,
        max_admitted_runs: int,
    ) -> None:
        self.data_dir = data_dir
        self.events = events
        self.artifacts = artifacts
        self.backend = backend
        self.compiler = compiler
        self._semaphore = asyncio.Semaphore(max_concurrent_runs)
        self._max_admitted_runs = max_admitted_runs
        self._admission_lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    def compile(self, plan: Plan) -> CompiledPlan:
        return self.compiler.compile(plan)

    async def submit(
        self,
        plan: CompiledPlan,
        inputs: dict[str, object],
        *,
        executor: RunExecutor | None = None,
    ) -> tuple[str, datetime]:
        async with self._admission_lock:
            if self._closed:
                raise RuntimeError("API is shutting down")
            if len(self._tasks) >= self._max_admitted_runs:
                raise RunQueueFull("maximum admitted run count reached; retry later")
            run_id = str(uuid4())
            created = datetime.now(UTC)
            request_artifact = await self.artifacts.put(
                json_artifact({"plan": plan.plan.model_dump(mode="json"), "inputs": inputs}),
                media_type="application/vnd.atlas.run-request+json",
            )
            await self.events.append(
                RunEvent(
                    run_id=run_id,
                    plan_hash=plan.content_hash,
                    event_type="run.queued",
                    status="queued",
                    timestamp=created,
                    artifact_refs=(request_artifact,),
                )
            )
            task = asyncio.create_task(
                self._execute(run_id, plan, inputs, executor), name=f"atlas-run-{run_id}"
            )
            self._tasks[run_id] = task

            def on_task_done(completed: asyncio.Task[None]) -> None:
                self._forget_task(run_id, completed)

            task.add_done_callback(on_task_done)
        logger.info("run_admitted", extra={"run_id": run_id, "plan_hash": plan.content_hash})
        return run_id, created

    def _forget_task(self, run_id: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(run_id, None)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error(
                    "run_task_unhandled_error",
                    extra={"run_id": run_id, "error_type": type(error).__name__},
                )

    async def _execute(
        self,
        run_id: str,
        plan: CompiledPlan,
        inputs: dict[str, object],
        executor: RunExecutor | None = None,
    ) -> None:
        try:
            async with self._semaphore:
                logger.info("run_started", extra={"run_id": run_id})
                if executor is None:
                    await self.backend.execute(plan, run_id=run_id, inputs=inputs)
                else:
                    await executor(plan, run_id, inputs)
        except asyncio.CancelledError:
            await self._append_terminal(run_id, plan.content_hash, "run.cancelled", "cancelled")
            logger.info("run_cancelled", extra={"run_id": run_id})
            raise
        except Exception as exc:
            await self._append_terminal(
                run_id,
                plan.content_hash,
                "run.failed",
                "failed",
                attributes={"error_type": type(exc).__name__},
            )
            logger.exception("run_failed", extra={"run_id": run_id})

    async def _append_terminal(
        self,
        run_id: str,
        plan_hash: str,
        event_type: str,
        status: str,
        *,
        attributes: dict[str, str] | None = None,
    ) -> None:
        await self.events.append(
            RunEvent(
                run_id=run_id,
                plan_hash=plan_hash,
                event_type=event_type,
                status=status,
                attributes=attributes or {},
            )
        )

    async def get_events(self, run_id: str) -> list[RunEvent]:
        events = await self.events.list_run(run_id)
        if not events:
            raise RunNotFound(run_id)
        return events

    async def cancel(self, run_id: str) -> str:
        events = await self.get_events(run_id)
        status = self.status_from_events(events)
        if status in {"succeeded", "failed", "timed_out", "cancelled"}:
            return "already_finished"
        task = self._tasks.get(run_id)
        if task is None:
            return "not_active"
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        latest = await self.events.list_run(run_id)
        if self.status_from_events(latest) not in {
            "succeeded",
            "failed",
            "timed_out",
            "cancelled",
        }:
            await self._append_terminal(run_id, latest[-1].plan_hash, "run.cancelled", "cancelled")
        return "cancelled"

    async def shutdown(self) -> None:
        async with self._admission_lock:
            self._closed = True
            tasks = list(self._tasks.items())
        for _, task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
            for run_id, _ in tasks:
                events = await self.events.list_run(run_id)
                if events and self.status_from_events(events) not in {
                    "succeeded",
                    "failed",
                    "timed_out",
                    "cancelled",
                }:
                    await self._append_terminal(
                        run_id, events[-1].plan_hash, "run.cancelled", "cancelled"
                    )

    @staticmethod
    def status_from_events(events: list[RunEvent]) -> str:
        for event in reversed(events):
            if event.event_type in {"run.completed", "run.failed", "run.cancelled"}:
                if event.event_type == "run.completed":
                    return event.status or "succeeded"
                return "failed" if event.event_type == "run.failed" else "cancelled"
        return "running" if any(event.event_type == "run.started" for event in events) else "queued"

    async def status(self, run_id: str) -> dict[str, object]:
        events = await self.get_events(run_id)
        last = events[-1]
        terminal = next(
            (
                event
                for event in reversed(events)
                if event.event_type in {"run.completed", "run.failed", "run.cancelled"}
            ),
            None,
        )
        node_statuses: dict[str, str] = {}
        for event in events:
            if event.node_id and event.event_type in {
                "node.started",
                "node.completed",
                "node.failed",
                "node.blocked",
            }:
                node_statuses[event.node_id] = event.status or "unknown"
        failure = terminal if terminal and terminal.event_type == "run.failed" else None
        return {
            "run_id": run_id,
            "plan_hash": last.plan_hash,
            "status": self.status_from_events(events),
            "created_at": events[0].timestamp,
            "updated_at": last.timestamp,
            "node_statuses": node_statuses,
            "error": failure.attributes.get("error_type") if failure else None,
        }

    async def ready(self) -> bool:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            probe = self.data_dir / ".atlas-ready"
            await asyncio.to_thread(probe.touch, exist_ok=True)
            await asyncio.to_thread(probe.unlink, missing_ok=True)
            await self.events.list_recent(1)
            return True
        except (OSError, sqlite3.Error):
            return False


def make_compiler(
    agents: Mapping[str, AgentRegistration], tools: Mapping[str, ToolRegistration]
) -> PlanCompiler:
    return PlanCompiler(
        agents={key: value.profile for key, value in agents.items()},
        tools={key: value.definition for key, value in tools.items()},
    )


def make_backend(
    *,
    data_dir: Path,
    agents: Mapping[str, AgentRegistration],
    tools: Mapping[str, ToolRegistration],
    tool_policy: ToolPolicy,
) -> LangGraphExecutionBackend:
    return LangGraphExecutionBackend(
        agent_registry=agents,
        tool_registry=tools,
        checkpoint_path=str(data_dir / "checkpoints.sqlite"),
        event_path=str(data_dir / "events.sqlite"),
        artifact_root=str(data_dir / "artifacts"),
        audit_path=str(data_dir / "audit.sqlite"),
        tool_policy=tool_policy,
    )
