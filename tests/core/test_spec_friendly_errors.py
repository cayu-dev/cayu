"""Misplaced Spec constructor kwargs must fail with a message that names the fix."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cayu import AgentSpec, EnvironmentSpec


def test_agentspec_tools_kwarg_points_to_register_agent() -> None:
    with pytest.raises(ValidationError, match="register_agent"):
        AgentSpec(name="a", model="m", tools=[])  # type: ignore[call-arg]


def test_agentspec_tool_policy_kwarg_points_to_register_agent() -> None:
    with pytest.raises(ValidationError, match="register_agent"):
        AgentSpec(name="a", model="m", tool_policy=object())  # type: ignore[call-arg]


def test_environmentspec_live_object_kwarg_points_to_environment() -> None:
    with pytest.raises(ValidationError, match="Environment"):
        EnvironmentSpec(name="e", workspace=object())  # type: ignore[call-arg]


def test_valid_specs_still_construct() -> None:
    assert AgentSpec(name="a", model="m").name == "a"
    assert EnvironmentSpec(name="e").name == "e"
