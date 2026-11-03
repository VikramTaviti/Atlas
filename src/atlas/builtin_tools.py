"""Small workspace tools and a constrained Docker test-runner prototype."""

from __future__ import annotations

import asyncio
from pathlib import Path

from atlas.plans.ir import ToolDefinition, ToolRiskClass


def workspace_read_definition() -> ToolDefinition:
    return ToolDefinition(
        tool_id="workspace.read",
        version="1",
        description="Read a UTF-8 file in the workspace.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
            "additionalProperties": False,
        },
        risk_class=ToolRiskClass.READ_ONLY,
        capabilities=("workspace.read",),
        workspace_path_argument="path",
    )


class WorkspaceReadTool:
    def __init__(self, workspace_root: str, *, max_bytes: int = 1_000_000) -> None:
        self.root = Path(workspace_root).resolve()
        self.max_bytes = max_bytes

    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        path = (self.root / str(arguments["path"])).resolve()
        path.relative_to(self.root)
        data = path.read_bytes()
        if len(data) > self.max_bytes:
            raise ValueError("workspace file exceeds configured size limit")
        return {"content": data.decode("utf-8")}


class DockerTestRunner:
    """Runs explicitly configured test argv in a disposable restricted container."""

    def __init__(
        self,
        workspace_root: str,
        *,
        image: str,
        commands: tuple[tuple[str, ...], ...],
        timeout_seconds: float = 120,
        output_limit_bytes: int = 100_000,
    ) -> None:
        if not image or not commands or any(not command for command in commands):
            raise ValueError("Docker image and at least one non-empty command are required")
        self.root = Path(workspace_root).resolve()
        self.image = image
        self.commands = commands
        self.timeout_seconds = timeout_seconds
        self.output_limit_bytes = output_limit_bytes

    async def invoke(self, arguments: dict[str, object]) -> dict[str, object]:
        command = arguments.get("command")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            raise ValueError("command must be an argv list")
        argv = tuple(command)
        if argv not in self.commands:
            raise PermissionError("command is not in the configured test command allowlist")
        docker_argv = (
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=128",
            "--memory=1g",
            "--cpus=2",
            "--mount",
            f"type=bind,src={self.root},dst=/workspace,readonly",
            "--workdir",
            "/workspace",
            self.image,
            *argv,
        )
        proc = await asyncio.create_subprocess_exec(
            *docker_argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), self.timeout_seconds)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError("Docker test command timed out") from None
        if len(stdout) + len(stderr) > self.output_limit_bytes:
            raise ValueError("Docker test command output exceeded the configured limit")
        return {
            "exit_code": proc.returncode,
            "stdout": stdout.decode("utf-8", "replace"),
            "stderr": stderr.decode("utf-8", "replace"),
        }


def docker_test_runner_definition() -> ToolDefinition:
    return ToolDefinition(
        tool_id="workspace.test",
        version="1",
        description="Run an allowlisted test command in Docker.",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "array", "items": {"type": "string"}}},
            "required": ["command"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "exit_code": {"type": ["integer", "null"]},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
            },
            "required": ["exit_code", "stdout", "stderr"],
            "additionalProperties": False,
        },
        risk_class=ToolRiskClass.READ_ONLY,
        capabilities=("workspace.test",),
        workspace_path_argument=None,
    )
