"""Run a real Gemini ReAct agent with a local tool mediated by Atlas's gateway.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) and the optional ``gemini`` extra.
This script is intentionally not run by the standard test suite.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from atlas.plans.compiler import PlanCompiler
from atlas.plans.ir import AgentProfile, Node, NodeKind, Plan, ToolDefinition
from atlas.providers.gemini import GeminiAgent
from atlas.runtime.backend import AgentRegistration, ToolRegistration
from atlas.runtime.langgraph_backend import LangGraphExecutionBackend
from atlas.security.policy import ToolPolicy


class CatalogLookupTool:
    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        query = str(arguments["query"])
        return {"matches": [f"Local demo catalog match for: {query}"]}


async def main() -> None:
    profile = AgentProfile(
        agent_id="gemini_catalog_agent",
        version="1",
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        allowed_tools=("catalog.lookup",),
        max_iterations=5,
    )
    tool_definition = ToolDefinition(
        tool_id="catalog.lookup",
        version="1",
        description="Search the local demo catalog using a short query.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={"type": "object", "required": ["matches"]},
        risk_class="read_only",
    )
    plan = Plan(
        plan_id="gemini-react-demo",
        task_id="gemini-react-demo-task",
        nodes=(
            Node(
                node_id="research",
                kind=NodeKind.AGENT,
                name="Use catalog.lookup to find the requested item and return its match.",
                agent_ref=profile.agent_id,
                output_schema={"type": "string"},
            ),
            Node(
                node_id="finish",
                kind=NodeKind.OUTPUT,
                name="Finish",
                depends_on=("research",),
            ),
        ),
    )
    compiled = PlanCompiler(
        agents={profile.agent_id: profile}, tools={tool_definition.tool_id: tool_definition}
    ).compile(plan)
    agent = GeminiAgent()
    data_dir = Path(os.environ.get("ATLAS_DATA_DIR", "/tmp/atlas-gemini"))
    data_dir.mkdir(parents=True, exist_ok=True)
    backend = LangGraphExecutionBackend(
        agent_registry={profile.agent_id: AgentRegistration(profile, agent)},
        tool_registry={
            tool_definition.tool_id: ToolRegistration(tool_definition, CatalogLookupTool())
        },
        checkpoint_path=str(data_dir / "checkpoints.sqlite"),
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({tool_definition.tool_id})),
    )
    try:
        result = await backend.execute(
            compiled,
            run_id=f"gemini-{uuid.uuid4()}",
            inputs={"task": "Find ATLAS in the demo catalog."},
        )
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "output": result.outputs.get("research"),
                    "agent_trace": result.agent_traces.get("research", []),
                },
                indent=2,
            )
        )
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
