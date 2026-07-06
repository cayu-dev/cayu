"""Per-agent environment shaping: one factory, a different binding per agent.

A subagent inherits its parent's ``environment_name`` (see ``SubagentTool``), so
parent and child always resolve the *same* registered environment. To give an
orchestrator agent and a QA agent differently-shaped environments anyway, branch
a single ``EnvironmentFactory`` on ``request.agent_name`` — a required, per-session
field. Here the ``"orchestrator"`` agent gets no checkout (``NoWorkspaceBinding``)
while the ``"qa"`` agent gets a git sandbox, both under one environment name.

Run it (no keys, no network — the git binding is constructed, not cloned here):

    PYTHONPATH=src python examples/environments/per_agent_environment.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from cayu import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    GitRepositoryBinding,
    LocalRunner,
    LocalWorkspace,
    NoWorkspaceBinding,
    WorkspaceBinding,
)

REPO_URL = "https://github.com/octocat/Hello-World.git"


class WorkbenchFactory(EnvironmentFactory):
    """One registered environment name; the binding shape depends on the agent."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        session_root = self._root / request.session_id
        session_root.mkdir(parents=True, exist_ok=True)

        binding: WorkspaceBinding
        if request.agent_name == "qa":
            # The QA agent reviews code, so it gets a real checkout.
            binding = GitRepositoryBinding(repo_url=REPO_URL, ref="master")
        else:
            # The orchestrator only delegates; it pays for no clone.
            binding = NoWorkspaceBinding()

        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(session_root, workspace_id=request.session_id),
                runner=LocalRunner(session_root),
                binding=binding,
            )
        )


async def main() -> None:
    factory = WorkbenchFactory(Path(tempfile.mkdtemp(prefix="cayu_per_agent_env_")))
    for agent_name in ("orchestrator", "qa"):
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id=f"sess-{agent_name}",
                agent_name=agent_name,
                environment_name="workbench",
            )
        )
        binding = type(result.environment.binding).__name__
        print(f"agent={agent_name!r:<14} environment_name='workbench'  binding={binding}")


if __name__ == "__main__":
    asyncio.run(main())
