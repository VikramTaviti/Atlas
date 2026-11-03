"""FastAPI application factory for the single-process ATLAS service."""

from __future__ import annotations

import inspect
import json
import logging
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from atlas.agent_factory import AgentFactory, AgentTemplateRegistry
from atlas.api.auth import Authenticator, UnauthenticatedLocalAuthenticator
from atlas.api.models import (
    CancellationResponse,
    HealthResponse,
    RunAcceptedResponse,
    RunCreateRequest,
    RunEventsResponse,
    RunStatusResponse,
    TaskAcceptedResponse,
    TaskCreateRequest,
)
from atlas.api.service import (
    RunNotFound,
    RunQueueFull,
    RunService,
    make_backend,
    make_compiler,
)
from atlas.observability import LocalArtifactStore, SQLiteRunEventStore
from atlas.plans.compiler import CompiledPlan, PlanCompiler
from atlas.plans.ir import Edge, Node, NodeKind, Plan
from atlas.plans.planner import Planner, PlannerLimits, PlannerRequest, WorkflowProposal
from atlas.runtime.backend import AgentRegistration, ToolRegistration
from atlas.security.policy import ToolPolicy

logger = logging.getLogger("atlas.api")


class _StructuredFormatter(logging.Formatter):
    """Emit one JSON object per application log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key in ("run_id", "plan_hash", "data_dir", "error_type"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _configure_application_logging() -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_StructuredFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def create_app(
    *,
    data_dir: str | Path | None = None,
    agent_registry: Mapping[str, AgentRegistration] | None = None,
    tool_registry: Mapping[str, ToolRegistration] | None = None,
    tool_policy: ToolPolicy | None = None,
    authenticator: Authenticator | None = None,
    require_authentication: bool = False,
    max_concurrent_runs: int = 4,
    max_admitted_runs: int = 128,
    planner: Planner | None = None,
    agent_templates: AgentTemplateRegistry | None = None,
    planner_limits: PlannerLimits | None = None,
) -> FastAPI:
    """Build the API around trusted, application-configured runtime capabilities.

    Request bodies can submit a Plan and JSON inputs, but cannot register adapters or
    alter the gateway policy. Authentication is disabled by default for local use.
    """
    if max_concurrent_runs < 1 or max_admitted_runs < 1:
        raise ValueError("run limits must be positive")
    if max_admitted_runs < max_concurrent_runs:
        raise ValueError("max_admitted_runs must be at least max_concurrent_runs")
    _configure_application_logging()
    resolved_dir = Path(data_dir or os.environ.get("ATLAS_DATA_DIR", "./atlas-data")).resolve()
    agents = dict(agent_registry or {})
    tools = dict(tool_registry or {})
    policy = tool_policy or ToolPolicy()
    auth = authenticator or UnauthenticatedLocalAuthenticator()
    if (planner is None) != (agent_templates is None):
        raise ValueError("task intake requires both a planner and trusted agent templates")
    limits = planner_limits or PlannerLimits()
    factory = AgentFactory(agent_templates) if agent_templates is not None else None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_dir.mkdir(parents=True, exist_ok=True)
        events = SQLiteRunEventStore(resolved_dir / "events.sqlite")
        artifacts = LocalArtifactStore(resolved_dir / "artifacts")
        backend = make_backend(
            data_dir=resolved_dir,
            agents=agents,
            tools=tools,
            tool_policy=policy,
        )
        application.state.run_service = RunService(
            data_dir=resolved_dir,
            events=events,
            artifacts=artifacts,
            backend=backend,
            compiler=make_compiler(agents, tools),
            max_concurrent_runs=max_concurrent_runs,
            max_admitted_runs=max_admitted_runs,
        )
        if not require_authentication:
            logger.warning("atlas_api_authentication_disabled")
        logger.info("atlas_api_started", extra={"data_dir": str(resolved_dir)})
        try:
            yield
        finally:
            await application.state.run_service.shutdown()
            if planner is not None:
                closer = getattr(planner, "aclose", None)
                if callable(closer):
                    try:
                        result = closer()
                        if inspect.isawaitable(result):
                            await result
                    except Exception as exc:
                        logger.error(
                            "planner_close_failed", extra={"error_type": type(exc).__name__}
                        )
            seen_adapters: set[int] = set()
            for registration in agents.values():
                adapter = registration.adapter
                if id(adapter) in seen_adapters:
                    continue
                seen_adapters.add(id(adapter))
                closer = getattr(adapter, "aclose", None)
                if callable(closer):
                    try:
                        result = closer()
                        if inspect.isawaitable(result):
                            await result
                    except Exception as exc:
                        logger.error(
                            "provider_adapter_close_failed",
                            extra={"error_type": type(exc).__name__},
                        )
            logger.info("atlas_api_stopped")

    application = FastAPI(
        title="ATLAS API",
        version="0.2.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def authenticate_request(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in {"/health/live", "/health/ready"}:
            return await call_next(request)
        try:
            principal = await auth.authenticate(request)
        except Exception:
            logger.exception("authentication_backend_error")
            return JSONResponse(status_code=503, content={"detail": "authentication unavailable"})
        if require_authentication and principal is None:
            return JSONResponse(status_code=401, content={"detail": "authentication required"})
        request.state.principal = principal
        return await call_next(request)

    @application.exception_handler(RunNotFound)
    async def run_not_found(_: Request, exc: RunNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"run '{exc.args[0]}' not found"})

    @application.exception_handler(RunQueueFull)
    async def run_queue_full(_: Request, exc: RunQueueFull) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": str(exc)},
            headers={"Retry-After": "1"},
        )

    @application.exception_handler(ValueError)
    async def invalid_run(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @application.get("/health/live", response_model=HealthResponse)
    async def health_live() -> HealthResponse:
        return HealthResponse(status="ok", checks={"process": "alive"})

    @application.get("/health/ready", response_model=HealthResponse)
    async def health_ready(request: Request, response: Response) -> HealthResponse:
        service: RunService = request.app.state.run_service
        ready = await service.ready()
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="ok" if ready else "not_ready",
            checks={"storage": "writable" if ready else "unavailable"},
        )

    @application.post(
        "/v1/runs", response_model=RunAcceptedResponse, status_code=status.HTTP_202_ACCEPTED
    )
    async def create_run(payload: RunCreateRequest, request: Request) -> RunAcceptedResponse:
        service: RunService = request.app.state.run_service
        compiled = service.compile(payload.plan)
        run_id, _ = await service.submit(compiled, payload.inputs)
        return RunAcceptedResponse(run_id=run_id, plan_hash=compiled.content_hash)

    @application.post(
        "/v1/tasks", response_model=TaskAcceptedResponse, status_code=status.HTTP_202_ACCEPTED
    )
    async def create_task(payload: TaskCreateRequest, request: Request) -> TaskAcceptedResponse:
        if planner is None or agent_templates is None or factory is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="task intake is not configured",
            )
        service: RunService = request.app.state.run_service
        proposal = await planner.plan(
            PlannerRequest(
                task=payload.task,
                context=payload.context,
                available_roles=agent_templates.planner_roles,
                limits=limits,
            )
        )
        # Re-validate even custom planner implementations; provider output is untrusted.
        proposal = WorkflowProposal.model_validate(proposal)
        if len(proposal.nodes) + 1 > limits.max_nodes:
            raise ValueError("planner proposal exceeds operator node limit")
        if proposal.max_depth > limits.max_depth:
            raise ValueError("planner proposal exceeds operator depth limit")
        if proposal.max_parallelism > limits.max_parallelism:
            raise ValueError("planner proposal exceeds operator parallelism limit")
        if any(
            len(node.task_description) > limits.max_task_description_chars
            for node in proposal.nodes
        ):
            raise ValueError("planner node task exceeds operator description limit")

        profiles = factory.profiles_for(proposal.nodes)
        by_node = {node.node_id: node for node in proposal.nodes}
        output_id = "atlas_finish"
        while output_id in by_node:
            output_id += "_"
        plan_nodes = [
            Node(
                node_id=node.node_id,
                kind=NodeKind.AGENT,
                name=node.task_description,
                agent_ref=profiles[node.node_id].agent_id,
                output_schema=profiles[node.node_id].output_schema,
            )
            for node in proposal.nodes
        ]
        outgoing = {edge.source for edge in proposal.edges}
        leaves = tuple(sorted(set(by_node) - outgoing))
        graph_edges = tuple(Edge(source=edge.source, target=edge.target) for edge in proposal.edges)
        plan_nodes.append(
            Node(
                node_id=output_id,
                kind=NodeKind.OUTPUT,
                name="Aggregate task result",
                depends_on=leaves,
            )
        )
        plan = Plan(
            plan_id=f"task-{uuid4()}",
            task_id=f"task-{uuid4()}",
            nodes=tuple(plan_nodes),
            edges=graph_edges,
            max_nodes=limits.max_nodes,
            max_depth=limits.max_depth,
            max_parallelism=proposal.max_parallelism,
        )
        compiled = PlanCompiler(
            agents={profile.agent_id: profile for profile in profiles.values()},
            tools={key: value.definition for key, value in tools.items()},
        ).compile(plan)

        async def execute_dynamic(
            compiled_plan: CompiledPlan, run_id: str, inputs: dict[str, object]
        ) -> None:
            registrations = await factory.instantiate_many(proposal.nodes)
            try:
                dynamic_backend = make_backend(
                    data_dir=resolved_dir,
                    agents=registrations,
                    tools=tools,
                    tool_policy=policy,
                )
                await dynamic_backend.execute(compiled_plan, run_id=run_id, inputs=inputs)
            finally:
                await factory.aclose(registrations.values())

        run_id, _ = await service.submit(
            compiled,
            {"task": payload.task, "context": payload.context},
            executor=execute_dynamic,
        )
        return TaskAcceptedResponse(run_id=run_id, plan_hash=compiled.content_hash, plan=plan)

    @application.get("/v1/runs/{run_id}", response_model=RunStatusResponse)
    async def get_run(run_id: str, request: Request) -> RunStatusResponse:
        service: RunService = request.app.state.run_service
        return RunStatusResponse.model_validate(await service.status(run_id))

    @application.get("/v1/runs/{run_id}/events", response_model=RunEventsResponse)
    async def get_run_events(run_id: str, request: Request) -> RunEventsResponse:
        service: RunService = request.app.state.run_service
        return RunEventsResponse(run_id=run_id, events=await service.get_events(run_id))

    @application.post("/v1/runs/{run_id}/cancel", response_model=CancellationResponse)
    async def cancel_run(run_id: str, request: Request) -> CancellationResponse:
        service: RunService = request.app.state.run_service
        result = await service.cancel(run_id)
        return CancellationResponse(run_id=run_id, status=result)  # type: ignore[arg-type]

    return application


app = create_app()
