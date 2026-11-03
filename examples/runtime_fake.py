"""Run a credential-free fan-out/join plan using the LangGraph backend adapter."""

import asyncio
import json
import os
import uuid
from pathlib import Path

from atlas.plans.compiler import PlanCompiler
from atlas.plans.ir import Node, NodeKind, Plan, ToolDefinition
from atlas.runtime.backend import ToolRegistration
from atlas.runtime.langgraph_backend import LangGraphExecutionBackend
from atlas.security.policy import ToolPolicy


class EchoBranchTool:
    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        mapped = arguments["mapped"]
        return {"branch": mapped["branch"]}  # type: ignore[index]


async def main() -> None:
    definition = ToolDefinition(
        tool_id="echo_branch",
        version="1",
        input_schema={},
        output_schema={"type": "object"},
        risk_class="read_only",
    )
    plan = Plan(
        plan_id="example-fanout-join",
        task_id="local-demo",
        max_parallelism=2,
        nodes=(
            Node(node_id="start", kind=NodeKind.JOIN, name="Start"),
            Node(
                node_id="inspect_a",
                kind=NodeKind.TOOL,
                name="Inspect A",
                tool_ref="echo_branch",
                depends_on=("start",),
                input_mapping={"branch": "A"},
                output_schema={"type": "object"},
            ),
            Node(
                node_id="inspect_b",
                kind=NodeKind.TOOL,
                name="Inspect B",
                tool_ref="echo_branch",
                depends_on=("start",),
                input_mapping={"branch": "B"},
                output_schema={"type": "object"},
            ),
            Node(
                node_id="join",
                kind=NodeKind.JOIN,
                name="Join inspections",
                depends_on=("inspect_a", "inspect_b"),
            ),
            Node(node_id="finish", kind=NodeKind.OUTPUT, name="Finish", depends_on=("join",)),
        ),
    )
    compiled = PlanCompiler(tools={"echo_branch": definition}).compile(plan)
    backend = LangGraphExecutionBackend(
        tool_registry={"echo_branch": ToolRegistration(definition, EchoBranchTool())},
        checkpoint_path=str(
            Path(os.environ.get("ATLAS_DATA_DIR", "/tmp")) / "atlas-example-checkpoints.sqlite"
        ),
        tool_policy=ToolPolicy(allowed_tool_ids=frozenset({"echo_branch"})),
    )
    result = await backend.execute(compiled, run_id=f"example-{uuid.uuid4()}")
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.status,
                "node_statuses": result.node_statuses,
                "final_output": result.outputs.get("finish"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
