"""Run a credential-free agent node that uses one ReAct tool call."""

import asyncio
import json
import os
import uuid
from pathlib import Path

from atlas.agents import AgentRequest, AgentToolCall, AgentTurn
from atlas.plans.compiler import PlanCompiler
from atlas.plans.ir import AgentProfile, Node, NodeKind, Plan, ToolDefinition
from atlas.runtime.backend import AgentRegistration, ToolRegistration
from atlas.runtime.langgraph_backend import LangGraphExecutionBackend
from atlas.security.policy import ToolPolicy


class ScriptedAgent:
    async def invoke(self, request: AgentRequest) -> AgentTurn:
        if request.iteration == 1:
            return AgentTurn(
                action="act",
                action_summary="Look up the requested item.",
                tool_calls=(
                    AgentToolCall(
                        tool_id="catalog_lookup",
                        arguments={"query": "ATLAS"},
                    ),
                ),
            )
        item = request.observations[-1].output["item"]
        return AgentTurn(
            action="final",
            action_summary="Return the item from the tool observation.",
            final_output=item,
        )


class CatalogTool:
    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        query = arguments["query"]
        return {"item": f"result for {query}"}


async def main() -> None:
    profile = AgentProfile(
        agent_id="catalog_agent",
        version="1",
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        allowed_tools=("catalog_lookup",),
        max_iterations=4,
    )
    definition = ToolDefinition(
        tool_id="catalog_lookup",
        version="1",
        description="Look up a catalog item by query.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={"type": "object", "required": ["item"]},
        risk_class="read_only",
    )
    plan = Plan(
        plan_id="react-demo",
        task_id="react-demo-task",
        nodes=(
            Node(
                node_id="research",
                kind=NodeKind.AGENT,
                name="Find the requested catalog item.",
                agent_ref="catalog_agent",
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
        agents={"catalog_agent": profile},
        tools={"catalog_lookup": definition},
    ).compile(plan)
    backend = LangGraphExecutionBackend(
        agent_registry={"catalog_agent": AgentRegistration(profile, ScriptedAgent())},
        tool_registry={"catalog_lookup": ToolRegistration(definition, CatalogTool())},
        checkpoint_path=str(
            Path(os.environ.get("ATLAS_DATA_DIR", "/tmp")) / "atlas-react-example.sqlite"
        ),
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"catalog_lookup"})),
    )
    result = await backend.execute(
        compiled,
        run_id=f"react-example-{uuid.uuid4()}",
        inputs={"task": "Find ATLAS"},
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "output": result.outputs.get("research"),
                "agent_trace": result.agent_traces.get("research", []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
