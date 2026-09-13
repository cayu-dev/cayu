from __future__ import annotations

import asyncio
import sys

import pytest
from examples.workspace_reference_binding import (
    Inventory,
    InventoryTool,
    SyntheticProvider,
    consume,
)
from pydantic import ValidationError

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    LocalWorkspace,
    Message,
    RunRequest,
    SyncBinding,
    ToolContext,
    WorkspaceReferenceBinding,
    WorkspaceReferenceBindingError,
)
from cayu.environments import EnvironmentFactory, EnvironmentFactoryResult
from cayu.runners import LocalRunner
from cayu.workspaces import RunnerWorkspace


def test_roundtrip_and_replacement(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="assigned-731")
    binding = workspace.reference_binding()
    restored = WorkspaceReferenceBinding.model_validate_json(binding.model_dump_json())
    restored.require_match(workspace.reference_binding())
    replacement = LocalWorkspace(tmp_path, workspace_id=workspace.id)
    with pytest.raises(WorkspaceReferenceBindingError, match="workspace_binding_mismatch"):
        restored.require_match(replacement.reference_binding())
    with pytest.raises(ValidationError):
        WorkspaceReferenceBinding.model_validate({**binding.model_dump(), "schema_version": 2})


def test_unavailable_context_does_not_trust_public_ids(tmp_path):
    workspace = LocalWorkspace(tmp_path)
    ctx = ToolContext(session_id="detached", workspace_id=workspace.id, workspace=workspace)
    for context in (ctx, ToolContext.model_validate_json(ctx.model_dump_json())):
        with pytest.raises(WorkspaceReferenceBindingError, match="workspace_binding_unavailable"):
            context.require_workspace_binding(workspace.reference_binding())


@pytest.mark.parametrize("mode", ["static", "adapter", "factory", "bound", "factory_bound"])
@pytest.mark.parametrize("wrong", [False, True])
def test_invocation_checks_selected_workspace(tmp_path, mode, wrong):
    async def scenario():
        source_root = tmp_path / "source"
        target_root = tmp_path / "target"
        source_root.mkdir()
        target_root.mkdir()
        for root in (source_root, target_root):
            (root / "note.txt").write_text("identical contents")
        source = LocalWorkspace(source_root, workspace_id="builder-name")
        target = (
            RunnerWorkspace(
                LocalRunner(target_root, inherit_env=False),
                workspace_id="adapter-assigned-987",
                python_executable=sys.executable,
            )
            if mode == "adapter"
            else LocalWorkspace(target_root, workspace_id="adapter-assigned-987")
        )
        bound = "bound" in mode
        selected = target if bound or mode == "adapter" else source
        foreign = source if selected is target else target
        inventory = Inventory(
            binding=(foreign if wrong else selected).reference_binding(),
            paths=("note.txt",),
        )
        inventory = Inventory.model_validate_json(inventory.model_dump_json())
        environment = Environment(
            EnvironmentSpec(name="logical-role"),
            workspace=source if bound else selected,
            binding=SyncBinding(target_workspace=target) if bound else None,
        )
        if "factory" not in mode:
            outcomes = await consume(environment, inventory)
        else:

            class Factory(EnvironmentFactory):
                async def create(self, request):
                    return EnvironmentFactoryResult(environment)

            app = CayuApp(enable_logging=False)
            app.register_provider(SyntheticProvider(), default=True)
            app.register_environment_factory(environment.spec, Factory(), default=True)
            tool = InventoryTool(inventory)
            app.register_agent(AgentSpec(name="reader", model="synthetic"), tools=[tool])
            async for _ in app.run(
                RunRequest(
                    agent_name="reader",
                    messages=[Message.text("user", "consume")],
                )
            ):
                pass
            outcomes = tool.outcomes
        assert outcomes == ["workspace_binding_mismatch" if wrong else "accepted"]

    asyncio.run(scenario())


def test_active_capture_and_independent_durable_adapter_recovery(tmp_path):
    # A synthetic adapter's private state, independent of application inventory JSON.
    authority = tmp_path / "adapter-generation"
    authority.write_text("incarnation-1")

    class DurableWorkspace(LocalWorkspace):
        def reference_binding(self):
            observed = super().reference_binding()
            return WorkspaceReferenceBinding(
                identity=observed.identity,
                generation=authority.read_text(),
            )

    class CaptureTool(InventoryTool):
        async def run(self, ctx, args):
            self.inventory = Inventory(binding=ctx.workspace_reference_binding(), paths=())
            return await super().run(ctx, args)

    async def scenario():
        first = DurableWorkspace(tmp_path, workspace_id="durable-assigned-id")
        tool = CaptureTool(Inventory(binding=first.reference_binding(), paths=()))
        app = CayuApp(enable_logging=False)
        app.register_provider(SyntheticProvider(), default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="capture"), workspace=first), default=True
        )
        app.register_agent(AgentSpec(name="reader", model="synthetic"), tools=[tool])
        async for _ in app.run(
            RunRequest(agent_name="reader", messages=[Message.text("user", "capture")])
        ):
            pass
        assert tool.outcomes == ["accepted"]
        serialized = tool.inventory.model_dump_json()
        restored = Inventory.model_validate_json(serialized)
        reopened = DurableWorkspace(tmp_path, workspace_id=first.id)
        environment = Environment(EnvironmentSpec(name="reopened"), workspace=reopened)
        assert await consume(environment, restored) == ["accepted"]
        authority.write_text("incarnation-2")
        assert await consume(environment, restored) == ["workspace_binding_mismatch"]

    asyncio.run(scenario())


def test_binding_does_not_certify_contents_or_existence(tmp_path):
    workspace = LocalWorkspace(tmp_path)
    binding = workspace.reference_binding()
    (tmp_path / "note.txt").write_text("changed after binding")
    binding.require_match(workspace.reference_binding())
    (tmp_path / "note.txt").unlink()
    binding.require_match(workspace.reference_binding())


def test_invocation_without_workspace_is_unavailable(tmp_path):
    binding = LocalWorkspace(tmp_path).reference_binding()
    assert asyncio.run(
        consume(
            Environment(EnvironmentSpec(name="no-workspace")),
            Inventory(binding=binding, paths=("never-read.txt",)),
        )
    ) == ["workspace_binding_unavailable"]
