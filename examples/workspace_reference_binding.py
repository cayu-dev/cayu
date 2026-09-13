"""Run without model credentials: python examples/workspace_reference_binding.py."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pydantic import BaseModel

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    LocalWorkspace,
    Message,
    RunRequest,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    WorkspaceReferenceBinding,
    WorkspaceReferenceBindingError,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent


class Inventory(BaseModel):
    binding: WorkspaceReferenceBinding
    paths: tuple[str, ...]


class InventoryTool(Tool):
    spec = ToolSpec(name="consume_inventory")

    def __init__(self, inventory: Inventory):
        self.inventory = inventory
        self.outcomes: list[str] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        try:
            ctx.require_workspace_binding(self.inventory.binding)
        except WorkspaceReferenceBindingError as error:
            self.outcomes.append(error.code)
            return ToolResult(content=error.code)
        assert ctx.workspace is not None
        # Identity has been checked. File reads still enforce access and existence.
        for path in self.inventory.paths:
            await ctx.workspace.read_bytes(path)
        self.outcomes.append("accepted")
        return ToolResult(content="accepted")


class SyntheticProvider(ModelProvider):
    name = "synthetic"

    def __init__(self):
        self.calls = 0

    async def stream(self, request: ModelRequest):
        self.calls += 1
        if self.calls == 1:
            yield ModelStreamEvent.tool_call(
                id="inventory",
                name="consume_inventory",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        else:
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})


async def consume(environment: Environment, inventory: Inventory) -> list[str]:
    app = CayuApp(enable_logging=False)
    app.register_provider(SyntheticProvider(), default=True)
    app.register_environment(environment, default=True)
    tool = InventoryTool(inventory)
    app.register_agent(AgentSpec(name="reader", model="synthetic"), tools=[tool])
    async for _ in app.run(
        RunRequest(
            agent_name="reader",
            messages=[Message.text("user", "Read the inventory")],
        )
    ):
        pass
    return tool.outcomes


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "note.txt").write_text("same contents")
        workspace = LocalWorkspace(root, workspace_id="independently-assigned-42")
        inventory = Inventory(binding=workspace.reference_binding(), paths=("note.txt",))
        restored = Inventory.model_validate_json(inventory.model_dump_json())
        print(
            await consume(Environment(EnvironmentSpec(name="demo"), workspace=workspace), restored)
        )
        replacement = LocalWorkspace(root, workspace_id=workspace.id)
        print(
            await consume(
                Environment(EnvironmentSpec(name="demo"), workspace=replacement), restored
            )
        )
        # ['accepted'], then ['workspace_binding_mismatch'] despite identical ID/path/content.


if __name__ == "__main__":
    asyncio.run(main())
