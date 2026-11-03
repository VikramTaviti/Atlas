"""Live Gemini task-intake API composition for local repository-readiness work.

Run with ``uvicorn examples.gemini_task_intake:app --host 127.0.0.1 --port 8000``.
This module uses live Gemini requests; the API/tests do not import it by default.
"""

from __future__ import annotations

import os
from pathlib import Path

from atlas.agent_factory import AgentTemplate, AgentTemplateRegistry
from atlas.api.app import create_app
from atlas.builtin_tools import (
    DockerTestRunner,
    WorkspaceReadTool,
    docker_test_runner_definition,
    workspace_read_definition,
)
from atlas.plans.ir import AgentProfile
from atlas.providers.gemini import GeminiAgent
from atlas.providers.gemini_planner import GeminiPlanner
from atlas.runtime.backend import ToolRegistration
from atlas.security.policy import ToolPolicy

workspace = Path(os.environ.get("ATLAS_TASK_WORKSPACE_ROOT", Path.cwd())).resolve()
test_image = os.environ.get("ATLAS_TEST_RUNNER_IMAGE", "atlas-atlas")

tools = {
    "workspace.read": ToolRegistration(
        workspace_read_definition(), WorkspaceReadTool(str(workspace))
    ),
    "workspace.test": ToolRegistration(
        docker_test_runner_definition(),
        DockerTestRunner(
            str(workspace),
            image=test_image,
            commands=(("python", "-m", "pip", "check"),),
        ),
    ),
}


def make_profile(role_id: str, allowed_tools: tuple[str, ...]) -> AgentProfile:
    return AgentProfile(
        agent_id=role_id,
        version="1",
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        allowed_tools=allowed_tools,
        capabilities=allowed_tools,
        max_iterations=5,
    )


templates = AgentTemplateRegistry(
    {
        "repository_inspector": AgentTemplate(
            role_id="repository_inspector",
            description="Inspect repository files and summarize implementation evidence.",
            profile=make_profile("repository_inspector", ("workspace.read",)),
            adapter_factory=GeminiAgent,
        ),
        "test_verifier": AgentTemplate(
            role_id="test_verifier",
            description="Run the configured dependency consistency check and report results.",
            profile=make_profile("test_verifier", ("workspace.test",)),
            adapter_factory=GeminiAgent,
        ),
        "release_reviewer": AgentTemplate(
            role_id="release_reviewer",
            description=(
                "Review gathered repository and verification evidence for release readiness."
            ),
            profile=make_profile("release_reviewer", ("workspace.read", "workspace.test")),
            adapter_factory=GeminiAgent,
        ),
    },
    tools=tools,
)

app = create_app(
    agent_templates=templates,
    planner=GeminiPlanner(),
    tool_registry=tools,
    tool_policy=ToolPolicy(
        allowed_tool_ids=frozenset(tools),
        allowed_capabilities=frozenset({"workspace.read", "workspace.test"}),
        workspace_root=str(workspace),
    ),
)
