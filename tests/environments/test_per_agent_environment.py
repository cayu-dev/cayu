"""Guard for examples/environments/per_agent_environment.py.

The documented pattern is that one ``EnvironmentFactory`` under a single
environment name shapes the environment by ``request.agent_name`` (so a subagent
that inherits the parent's ``environment_name`` can still get a different binding).
This pins that branching so the example cannot silently rot.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

from cayu import EnvironmentFactoryRequest

_EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples" / "environments" / "per_agent_environment.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("per_agent_environment_example", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load()


def _binding_for(agent_name: str, root: Path) -> str:
    factory = mod.WorkbenchFactory(root)
    result = asyncio.run(
        factory.create(
            EnvironmentFactoryRequest(
                session_id=f"s-{agent_name}",
                agent_name=agent_name,
                environment_name="workbench",
            )
        )
    )
    return type(result.environment.binding).__name__


def test_factory_branches_environment_shape_on_agent_name(tmp_path: Path) -> None:
    # Same environment_name, different shapes chosen by agent_name.
    assert _binding_for("qa", tmp_path) == "GitRepositoryBinding"
    assert _binding_for("orchestrator", tmp_path) == "NoWorkspaceBinding"
