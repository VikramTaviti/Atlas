import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.plans.compiler import PlanCompiler, PlanValidationError
from atlas.plans.ir import AgentProfile, ToolDefinition
from atlas.plans.planner import FixturePlanner

EXAMPLES = Path(__file__).parents[1] / "examples" / "plans"


@pytest.mark.parametrize("filename", ["sequential.json", "fanout_join.json"])
def test_compiles_examples_and_hash_is_stable(filename: str) -> None:
    payload = json.loads((EXAMPLES / filename).read_text())
    first = PlanCompiler().compile(payload)
    second = PlanCompiler().compile(copy.deepcopy(payload))
    assert first.content_hash == second.content_hash
    assert first.canonical_json == second.canonical_json
    assert first.topological_order[-1] == "finish"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p["nodes"].append(dict(p["nodes"][0])), "duplicate node IDs"),
        (lambda p: p["nodes"][1].update(depends_on=["missing"]), "dangling edge source"),
        (lambda p: p["nodes"][0].update(kind="executable"), "kind"),
        (lambda p: p["nodes"][0].update(depends_on=["finish"]), "cycle"),
        (lambda p: p.update(max_nodes=1), "limit"),
        (lambda p: p["nodes"][1].update(output_schema={"type": "nonsense"}), "invalid JSON Schema"),
    ],
)
def test_rejects_invalid_plans(mutation, message: str) -> None:
    payload = json.loads((EXAMPLES / "sequential.json").read_text())
    mutation(payload)
    with pytest.raises(PlanValidationError, match=message):
        PlanCompiler().compile(payload)


def test_rejects_unbounded_parallelism() -> None:
    payload = json.loads((EXAMPLES / "fanout_join.json").read_text())
    payload["max_parallelism"] = 1
    with pytest.raises(PlanValidationError, match="parallelism"):
        PlanCompiler().compile(payload)


def test_rejects_unreachable_output_in_cyclic_graph() -> None:
    payload = json.loads((EXAMPLES / "sequential.json").read_text())
    payload["nodes"][0]["depends_on"] = ["finish"]
    with pytest.raises(PlanValidationError, match="cycle"):
        PlanCompiler().compile(payload)


def test_compiled_plan_ir_is_immutable() -> None:
    compiled = PlanCompiler().compile(json.loads((EXAMPLES / "sequential.json").read_text()))
    with pytest.raises(ValidationError):
        compiled.plan.plan_id = "changed"  # type: ignore[misc]


def test_compiled_plan_nested_json_is_immutable() -> None:
    payload = json.loads((EXAMPLES / "sequential.json").read_text())
    payload["metadata"] = {"nested": {"values": [1, 2]}}
    compiled = PlanCompiler().compile(payload)
    with pytest.raises(TypeError, match="immutable"):
        compiled.plan.metadata["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        compiled.plan.metadata["nested"]["new"] = True  # type: ignore[index]
    with pytest.raises(AttributeError):
        compiled.plan.metadata["nested"]["values"].append(3)  # type: ignore[attr-defined]


def test_rejects_unknown_agent_or_tool() -> None:
    payload = json.loads((EXAMPLES / "sequential.json").read_text())
    payload["nodes"][0].update(kind="agent", agent_ref="missing")
    with pytest.raises(PlanValidationError, match="unknown agent"):
        PlanCompiler().compile(payload)


def test_agent_registry_resolution_checks_capabilities() -> None:
    payload = json.loads((EXAMPLES / "sequential.json").read_text())
    payload["nodes"][0].update(kind="agent", agent_ref="analyst", required_capabilities=["network"])
    agent = AgentProfile(
        agent_id="analyst",
        version="1",
        input_schema={},
        output_schema={},
        capabilities=("workspace_read",),
    )
    with pytest.raises(PlanValidationError, match="undeclared capabilities"):
        PlanCompiler(agents={"analyst": agent}).compile(payload)


def test_validates_registry_json_schemas() -> None:
    with pytest.raises(ValueError, match="invalid JSON Schema"):
        AgentProfile(
            agent_id="a", version="1", input_schema={"type": "not-a-type"}, output_schema={}
        )
    with pytest.raises(ValueError, match="invalid JSON Schema"):
        ToolDefinition(tool_id="t", version="1", input_schema={}, output_schema={"required": "x"})


def test_rejects_excessive_depth() -> None:
    payload = json.loads((EXAMPLES / "sequential.json").read_text())
    payload["nodes"].insert(
        1, {"node_id": "middle", "kind": "join", "name": "Middle", "depends_on": ["start"]}
    )
    payload["nodes"][2]["depends_on"] = ["middle"]
    payload["max_depth"] = 2
    with pytest.raises(PlanValidationError, match="depth"):
        PlanCompiler().compile(payload)


def test_fixture_planner_is_repeatable_and_approval_graph_compiles() -> None:
    planner = FixturePlanner()
    assert planner.sequential().model_dump(mode="json") == planner.sequential().model_dump(
        mode="json"
    )
    compiled = PlanCompiler().compile(planner.approval_flow())
    assert compiled.topological_order[-1] == "finish"
