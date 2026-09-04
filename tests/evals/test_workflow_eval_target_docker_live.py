"""Opt-in real-Docker isolation proof for concurrent workflow-root eval trials."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import NoReturn

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    DockerCodingEnvironmentFactory,
    DockerImageIdentity,
    DockerRunner,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EvalCase,
    EvalStatus,
    EvalSuite,
    ExecCommand,
    FinalOutputContains,
    LocalWorkspace,
    RunRequest,
    ScriptedModelProvider,
    WorkflowBase,
    WorkflowEvalExecution,
    WorkflowEvalInstanceScope,
    WorkflowEvalResult,
    WorkflowEvalTarget,
    WorkflowSpec,
    run_workflow_eval_suite,
)

pytestmark = pytest.mark.process

_IMAGE_ENV = "CAYU_DOCKER_CODING_IMAGE"
_REQUIRE_ENV = "CAYU_REQUIRE_DOCKER_CODING"
_REVISION = "sha256:" + "1" * 64


def _configuration_or_skip() -> tuple[str, str, str]:
    docker_path = os.environ.get("CAYU_DOCKER_PATH") or shutil.which("docker")
    image = os.environ.get(_IMAGE_ENV)
    if docker_path is None or image is None:
        _unavailable(f"docker CLI and {_IMAGE_ENV} are required")
    try:
        subprocess.run(
            [docker_path, "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        image_id = subprocess.run(
            [docker_path, "image", "inspect", "--format", "{{.Id}}", image],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception as exc:
        _unavailable(f"Docker coding image unavailable: {exc}")
    if not image_id.startswith("sha256:") or len(image_id) != 71:
        _unavailable("Docker image inspection did not return an exact content digest")
    return docker_path, image, image_id


def _unavailable(reason: str) -> NoReturn:
    if os.environ.get(_REQUIRE_ENV) == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _app() -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="entry", model="scripted-model"))
    return app


class _DockerIsolationWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="docker-isolation-workflow-eval")

    def __init__(self, app: CayuApp, *, runner, case_id: str, state: dict) -> None:
        super().__init__(app)
        self._runner = runner
        self._case_id = case_id
        self._state = state

    async def run(self, session_id: str):
        context = self.context(session_id)
        yield await context.start()
        self._state["active"] += 1
        self._state["max_active"] = max(self._state["max_active"], self._state["active"])
        if self._state["active"] == 2:
            self._state["overlap"].set()
        try:
            await asyncio.wait_for(self._state["overlap"].wait(), timeout=20)
            probe = await self._runner.exec(
                ExecCommand.process(
                    "python3",
                    "-c",
                    "import pathlib, sys; "
                    "case_id = sys.argv[1]; "
                    "markers = list(pathlib.Path('.').glob('workflow-eval-marker-*')); "
                    "assert not markers, markers; "
                    "pathlib.Path('workflow-eval-marker-' + case_id).write_text(case_id)",
                    self._case_id,
                )
            )
            if probe.exit_code != 0:
                raise RuntimeError("Docker workspace isolation probe failed.")
            yield await context.completed({"answer": self._case_id})
        finally:
            self._state["active"] -= 1


def test_concurrent_workflow_eval_trials_use_isolated_docker_resources_and_remove_them(
    tmp_path: Path,
) -> None:
    docker_path, image, image_id = _configuration_or_skip()
    source_root = tmp_path / "source"
    source_root.mkdir()
    seed_app = _app()
    state = {"active": 0, "max_active": 0, "overlap": asyncio.Event()}
    container_ids: list[str] = []
    workspace_ids: list[str] = []

    async def factory(invocation):
        app = _app()
        created = await DockerCodingEnvironmentFactory(
            source_workspace=LocalWorkspace(source_root),
            image_identity=DockerImageIdentity(reference=image, content_digest=image_id),
            docker_path=docker_path,
        ).create(
            EnvironmentFactoryRequest(
                session_id=invocation.workflow_run_id,
                agent_name="entry",
                environment_name="coding",
            )
        )
        runner = created.environment.runner
        workspace = created.environment.workspace
        if not isinstance(runner, DockerRunner) or runner.container_id is None or workspace is None:
            raise RuntimeError("Docker workflow target allocation was incomplete.")
        release = created.release
        if release is None:
            raise RuntimeError("Docker workflow target allocation was not releasable.")
        container_ids.append(runner.container_id)
        workspace_ids.append(workspace.id)

        async def close() -> None:
            await release(EnvironmentFactoryReleaseAction.DISCARD)

        return WorkflowEvalExecution(
            app=app,
            workflow=_DockerIsolationWorkflow(
                app,
                runner=runner,
                case_id=invocation.case_id,
                state=state,
            ),
            close=close,
        )

    target = WorkflowEvalTarget(
        key="docker-workflow-target",
        app=seed_app,
        request_base=RunRequest(agent_name="entry", messages=[]),
        application_release_id="docker-workflow-live",
        workflow_spec=_DockerIsolationWorkflow.spec,
        implementation_revision=_REVISION,
        result_projector_revision=_REVISION,
        execution_scope_revision=_REVISION,
        instance_scope=WorkflowEvalInstanceScope.PER_TRIAL,
        workflow_factory=factory,
        result_projector=lambda evidence: WorkflowEvalResult(
            final_output=evidence.completion_event.payload["answer"]
        ),
    )
    suite = EvalSuite(
        id="docker-workflow-suite",
        cases=[
            EvalCase(
                id=case_id,
                request=RunRequest(agent_name="entry", messages=[]),
                assertions=[FinalOutputContains(case_id)],
            )
            for case_id in ("first-case", "second-case")
        ],
    )

    result = asyncio.run(run_workflow_eval_suite(target, suite, max_concurrency=2))

    assert result.status is EvalStatus.PASSED
    assert state["max_active"] == 2
    assert len(container_ids) == len(set(container_ids)) == 2
    assert len(workspace_ids) == len(set(workspace_ids)) == 2
    for container_id in container_ids:
        inspected = subprocess.run(
            [docker_path, "container", "inspect", container_id],
            capture_output=True,
            check=False,
            timeout=20,
        )
        assert inspected.returncode != 0
