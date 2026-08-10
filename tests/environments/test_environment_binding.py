from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from cayu.environments import (
    Environment,
    EnvironmentSpec,
    NativeBinding,
    WorkspaceInstructions,
    WorkspaceInstructionsConfig,
    copy_environment,
    load_workspace_instructions,
)
from cayu.workspaces import LocalWorkspace


def test_environment_binding_defaults_to_none() -> None:
    environment = Environment(EnvironmentSpec(name="local"))

    assert environment.binding is None


def test_environment_accepts_workspace_binding() -> None:
    binding = NativeBinding(default_path="/workspace")

    environment = Environment(EnvironmentSpec(name="local"), binding=binding)

    assert environment.binding is binding


def test_environment_rejects_invalid_binding() -> None:
    invalid_binding: Any = object()

    try:
        Environment(EnvironmentSpec(name="local"), binding=invalid_binding)
    except TypeError as exc:
        assert "binding" in str(exc)
    else:
        raise AssertionError("Environment accepted an invalid binding.")


def test_copy_environment_preserves_binding_and_workspace_instructions() -> None:
    binding = NativeBinding(default_path="/workspace")
    workspace_instructions = WorkspaceInstructions(
        content="Use the project test runner.",
        sources=("AGENTS.md",),
    )
    environment = Environment(
        EnvironmentSpec(name="local"),
        binding=binding,
        workspace_instructions=workspace_instructions,
    )

    copied = copy_environment(environment)

    assert copied is not environment
    assert copied.binding is binding
    assert copied.workspace_instructions == workspace_instructions
    assert copied.workspace_instructions is not workspace_instructions


@pytest.mark.parametrize("invalid_path", ["bad\x00path", "bad\ud800path"])
@pytest.mark.parametrize("bypass", ["model_copy", "mutation"])
def test_workspace_instruction_loading_revalidates_config_before_workspace_read(
    tmp_path,
    monkeypatch,
    invalid_path: str,
    bypass: str,
) -> None:
    workspace = LocalWorkspace(tmp_path)
    environment = Environment(
        EnvironmentSpec(name="local"),
        workspace=workspace,
        workspace_instructions=WorkspaceInstructionsConfig(paths=("AGENTS.md",)),
    )
    config = environment.workspace_instructions
    assert isinstance(config, WorkspaceInstructionsConfig)
    if bypass == "model_copy":
        environment.workspace_instructions = config.model_copy(update={"paths": (invalid_path,)})
    else:
        config.paths = (invalid_path,)

    read_bytes = AsyncMock(wraps=workspace.read_bytes)
    monkeypatch.setattr(workspace, "read_bytes", read_bytes)

    with pytest.raises(ValueError):
        asyncio.run(load_workspace_instructions(environment))

    read_bytes.assert_not_awaited()


def test_workspace_instruction_loading_preserves_unicode_path(tmp_path) -> None:
    path = "Zażółć_日本語_😀.md"
    (tmp_path / path).write_text("Use the portable instructions.\n")
    environment = Environment(
        EnvironmentSpec(name="local"),
        workspace=LocalWorkspace(tmp_path),
        workspace_instructions=WorkspaceInstructionsConfig(paths=(path,)),
    )

    instructions = asyncio.run(load_workspace_instructions(environment))

    assert instructions == WorkspaceInstructions(
        content="Use the portable instructions.\n",
        sources=(path,),
    )
