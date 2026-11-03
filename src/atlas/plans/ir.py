"""Versioned, serializable plan intermediate representation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, cast

from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrozenDict(dict[str, Any]):
    """A JSON-compatible dict that prevents ordinary mutation after validation."""

    def _immutable(self, *_: Any, **__: Any) -> None:
        raise TypeError("validated plan data is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable  # type: ignore[assignment]
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable  # type: ignore[assignment]


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        frozen = dict.__new__(FrozenDict)
        dict.__init__(frozen, {key: _freeze_json(item) for key, item in value.items()})
        return frozen
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


class NodeKind(StrEnum):
    AGENT = "agent"
    TOOL = "tool"
    CONDITION = "condition"
    JOIN = "join"
    HUMAN_APPROVAL = "human_approval"
    OUTPUT = "output"


class ToolRiskClass(StrEnum):
    READ_ONLY = "read_only"
    IDEMPOTENT = "idempotent"
    SIDE_EFFECTING = "side_effecting"
    DESTRUCTIVE = "destructive"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Edge(FrozenModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    condition: str | None = None


class Node(FrozenModel):
    node_id: str = Field(min_length=1)
    kind: NodeKind
    name: str = Field(min_length=1)
    agent_ref: str | None = None
    tool_ref: str | None = None
    depends_on: tuple[str, ...] = ()
    input_mapping: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=60, gt=0)
    max_retries: int = Field(default=0, ge=0, le=5)
    retry_delay_seconds: float = Field(default=0.0, ge=0, le=30)
    required_capabilities: tuple[str, ...] = ()

    @field_validator("input_mapping", mode="after")
    @classmethod
    def freeze_input_mapping(cls, value: dict[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], _freeze_json(value))

    @field_validator("output_schema")
    @classmethod
    def valid_output_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ValueError(f"invalid JSON Schema: {exc.message}") from exc
        return cast(dict[str, Any], _freeze_json(schema))


class Plan(FrozenModel):
    plan_id: str = Field(min_length=1)
    plan_version: int = Field(default=1, ge=1)
    task_id: str = Field(min_length=1)
    nodes: tuple[Node, ...] = Field(min_length=1)
    edges: tuple[Edge, ...] = ()
    max_nodes: int = Field(default=64, ge=1, le=512)
    max_depth: int = Field(default=16, ge=1, le=128)
    max_parallelism: int = Field(default=8, ge=1, le=64)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="after")
    @classmethod
    def freeze_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], _freeze_json(value))


class AgentProfile(FrozenModel):
    agent_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    allowed_tools: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    max_iterations: int = Field(default=8, ge=1, le=32)

    @field_validator("input_schema", "output_schema")
    @classmethod
    def valid_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ValueError(f"invalid JSON Schema: {exc.message}") from exc
        return cast(dict[str, Any], _freeze_json(schema))


class ToolDefinition(FrozenModel):
    tool_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    risk_class: ToolRiskClass = ToolRiskClass.READ_ONLY
    capabilities: tuple[str, ...] = ()
    description: str = ""
    timeout_seconds: int = Field(default=30, gt=0)
    requires_approval: bool = False
    requires_network: bool = False
    workspace_path_argument: str | None = None

    @field_validator("input_schema", "output_schema")
    @classmethod
    def valid_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ValueError(f"invalid JSON Schema: {exc.message}") from exc
        return cast(dict[str, Any], _freeze_json(schema))
