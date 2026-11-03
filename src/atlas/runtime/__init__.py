"""Framework-independent execution contracts and backend adapters."""

from atlas.runtime.backend import (
    AgentRegistration,
    ExecutionBackend,
    ExecutionResult,
    ExecutionStatus,
    ToolRegistration,
)

__all__ = [
    "AgentRegistration",
    "ExecutionBackend",
    "ExecutionResult",
    "ExecutionStatus",
    "ToolRegistration",
]
