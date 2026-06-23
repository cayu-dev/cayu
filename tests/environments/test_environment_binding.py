from __future__ import annotations

from typing import Any

from cayu.environments import (
    Environment,
    EnvironmentSpec,
    NativeBinding,
    WorkspaceInstructions,
    copy_environment,
)


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
