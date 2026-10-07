"""Registration components compose independently."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cayu


def test_registration_components_compose_without_application_controllers() -> None:
    script = """
import importlib.abc
import sys

blocked = {
    "cayu.applications",
    "cayu.runtime._session_engine",
    "cayu.runtime._model_step_executor",
    "cayu.runtime._recovery_coordinator",
}

class RejectControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Registration imported {fullname}")

sys.meta_path.insert(0, RejectControllers())
from cayu import _application_registration as registration
from cayu.agents import AgentSpec
from cayu.environments.base import EnvironmentSpec
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec
from cayu.vaults.redaction import SecretRedactor

class SampleTool(Tool):
    spec = ToolSpec(
        name="sample",
        effect=ToolEffect.NONE,
        input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
    )

    async def run(self, ctx, args):
        return ToolResult(content="ok")

redactor = SecretRedactor()
agent = AgentSpec(name="agent", model="model", metadata={"labels": ["original"]})
agent_copy = registration._validate_agent_spec(agent)
agent.metadata["labels"].append("changed")
assert agent_copy.metadata == {"labels": ["original"]}
environment = EnvironmentSpec(name="environment", metadata={"labels": ["original"]})
environment_copy = registration._validate_environment_spec(environment, redactor=redactor)
environment.metadata["labels"].append("changed")
assert environment_copy.metadata == {"labels": ["original"]}
assert registration._validate_provider_model_patterns(iter(["model-*"])) == ("model-*",)

tool = SampleTool()
registered = registration._validate_registered_tool(tool, redactor=redactor, timeout_seconds=2)
snapshot = registration._copy_registered_tool(registered)
descriptor = registration._registered_tool_descriptor(registered)
assert descriptor == registration._registered_tool_descriptor(snapshot)
assert snapshot.tool is tool
snapshot.schema["properties"]["value"]["type"] = "integer"
snapshot.execution_contract["timeout_strength"] = "none"
assert registered.schema["properties"]["value"]["type"] == "string"
assert registered.execution_contract["timeout_strength"] == "cooperative_in_process"
assert descriptor == registration._registered_tool_descriptor(registered)
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
