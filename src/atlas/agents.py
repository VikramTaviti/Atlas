"""Provider-independent agent contract and deterministic test adapter."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentToolSpec(BaseModel):
    """A tool capability the server has explicitly made available to an agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_id: str = Field(min_length=1)
    description: str = ""
    input_schema: dict[str, Any]


class AgentObservation(BaseModel):
    """A tool result or rejection fed into the next agent iteration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    iteration: int = Field(default=1, ge=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    provider_call_id: str | None = None
    continuation: dict[str, Any] | None = None
    success: bool
    output: Any = None
    error: str | None = None


class AgentToolCall(BaseModel):
    """One structured tool request returned by the agent adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_id: str = Field(min_length=1)
    arguments: dict[str, Any]
    call_id: str | None = None


class AgentTelemetry(BaseModel):
    """Provider-neutral measured metadata attached to one agent decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class AgentTurn(BaseModel):
    """A structured ReAct decision; deliberately has no private-thought field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["act", "final"]
    action_summary: str = Field(
        min_length=1,
        max_length=500,
        description="Concise operational summary only; never include private chain-of-thought.",
    )
    tool_calls: tuple[AgentToolCall, ...] = Field(default=(), max_length=64)
    final_output: Any = Field(
        default=None,
        description="Task result only; do not include private chain-of-thought.",
    )
    continuation: dict[str, Any] | None = None
    telemetry: AgentTelemetry | None = None

    @model_validator(mode="after")
    def validate_action_shape(self) -> "AgentTurn":
        if self.action == "act" and not self.tool_calls:
            raise ValueError("act turns must include at least one tool call")
        if self.action == "act" and self.final_output is not None:
            raise ValueError("act turns cannot also provide a final output")
        if self.action == "final" and self.tool_calls:
            raise ValueError("final turns cannot include tool calls")
        return self


AgentResponse = str | AgentTurn | dict[str, Any]


@dataclass(frozen=True)
class AgentRequest:
    instruction: str
    input_json: str
    iteration: int = 1
    observations: tuple[AgentObservation, ...] = ()
    available_tools: tuple[AgentToolSpec, ...] = ()


class Agent(Protocol):
    async def invoke(self, request: AgentRequest) -> AgentResponse: ...


class FakeAgent:
    """Returns a fixed response so compiler/runtime tests need no model credentials."""

    def __init__(self, response: AgentResponse) -> None:
        self.response = response

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        del request
        return self.response
