"""Generated public workflow recovery, without claiming Docker execution."""

from __future__ import annotations

import asyncio
import importlib
import subprocess
from dataclasses import replace

import pytest
from tests.core.test_queued_session_messages import RecordingOneShotProvider

from cayu import (
    AgentSpec,
    CayuApp,
    CodingProductAdmissionError,
    CodingProductArtifactRepository,
    CodingProductState,
    DockerImageIdentity,
    ExecutionProfileBehaviorIdentity,
    LocalArtifactStore,
    LocalWorkspace,
    SQLiteSessionStore,
    SQLiteTaskStore,
)
from cayu.cli import main
from cayu.cli.project import project_context
from cayu.environments import Environment, EnvironmentSpec


def test_generated_workflow_recovers_settled_publication(tmp_path, monkeypatch):
    assert (
        main(
            [
                "new",
                "recovery-coder",
                "--preset",
                "coding",
                "--execution",
                "docker",
                "--coding-toolchain",
                "python",
                "--dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    project = tmp_path / "recovery-coder"
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("value = 1\n", encoding="utf-8")
    for command in (
        ["init"],
        ["add", "example.py"],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "initial fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(source), *command], check=True, capture_output=True)

    with project_context(project):
        workflow = importlib.import_module("workflows.coding_product")
        domain = importlib.import_module("domain.coding_product")
        composition = importlib.import_module("operations.coding")
        profile = composition._python_toolchain_profile(
            DockerImageIdentity(reference="fixture@sha256:" + "a" * 64)
        )
        workspace = LocalWorkspace(
            source,
            workspace_id="coding-source-workspace",
            excluded_directory_names=composition._SOURCE_EXCLUDED_DIRECTORY_NAMES,
            excluded_path_patterns=composition._SOURCE_EXCLUDED_PATH_PATTERNS,
        )
        artifacts = LocalArtifactStore(tmp_path / "artifacts", store_id="fixture-artifacts")
        provider = RecordingOneShotProvider()
        task = domain.CodingProductTask(
            product_run_id="generated-recovery",
            session_id="generated-recovery",
            task_id="fixture-task",
            instruction="inspect the fixture",
        )

        def build_application(store, task_store):
            app = CayuApp(session_store=store, task_store=task_store, enable_logging=False)
            app.register_provider(provider)
            app.register_agent(AgentSpec(name="coder", model="fake-model"))
            app.register_environment(
                Environment(
                    EnvironmentSpec(
                        name="coding",
                        execution_profile_identity=ExecutionProfileBehaviorIdentity(
                            name="tests:generated-recovery-environment",
                            behavior_version="1",
                            implementation_version="1",
                        ),
                    )
                )
            )
            return workflow.CodingProductApplication(
                app,
                source_workspace=workspace,
                artifact_store=artifacts,
                toolchain_profile=profile,
                agent_name="coder",
                project_root=source,
            )

        async def scenario():
            store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
            task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
            try:
                application = build_application(store, task_store)
                with pytest.raises(FileNotFoundError):
                    await application.recover_settled(task)
                assert not provider.requests
                with pytest.raises(FileNotFoundError):
                    await CodingProductArtifactRepository(artifacts).load_request(
                        task.product_run_id,
                        session_id=task.session_id,
                    )

                selected = []

                async def lose_publication(self, candidate):
                    selected.append(candidate.digest)
                    raise RuntimeError("publication process lost")

                with monkeypatch.context() as faults:
                    faults.setattr(
                        CodingProductArtifactRepository, "publish_candidate", lose_publication
                    )
                    with pytest.raises(RuntimeError, match="publication process lost"):
                        await application.run(task)
                assert len(provider.requests) == 1
            finally:
                await store.close()
                await task_store.close()

            store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
            task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
            try:
                replacement = build_application(store, task_store)
                with pytest.raises(CodingProductAdmissionError):
                    await replacement.recover_settled(replace(task, instruction="another task"))
                result = await replacement.recover_settled(task)
                assert result.candidate.state is CodingProductState.RECONSTRUCTION_REQUIRED
                assert selected == [result.candidate.digest]
                assert await replacement.recover_settled(task) == result
                assert len(provider.requests) == 1
            finally:
                await store.close()
                await task_store.close()

        asyncio.run(scenario())
