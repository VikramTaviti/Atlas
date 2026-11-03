"""Provider-independent tool contract and deterministic fixture tool."""

from typing import Protocol


class Tool(Protocol):
    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]: ...


class FakeTool:
    """A deterministic tool fixture for contract tests and examples."""

    def __init__(self, result: dict[str, object]) -> None:
        self.result = result

    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        del arguments
        return dict(self.result)
