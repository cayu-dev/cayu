"""Live Docker checks for deferred environments and warm spares (opt-in).

CAYU_DOCKER_DEFERRED_IMAGE=python@sha256:... pytest -q tests/environments/test_docker_deferred_live.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    DockerCodingEnvironmentFactory,
    DockerCodingToolchainProfile,
    DockerImageIdentity,
    EnvironmentSpec,
    ExecCommand,
    ExecutionProfileBehaviorIdentity,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SessionStatus,
    SQLiteSessionStore,
    StaticToolExposurePolicy,
    StaticToolPolicy,
    Tool,
    ToolEffect,
    ToolResult,
    ToolSpec,
    run_to_completion,
)

pytestmark = pytest.mark.process

_IMAGE_ENV = "CAYU_DOCKER_DEFERRED_IMAGE"
_PROBE = (
    "import os, socket\n"
    "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=2); net = 'open'\n"
    "except OSError:\n    net = 'blocked'\n"
    "try:\n    open('/etc/cayu-probe', 'w'); fs = 'writable'\n"
    "except OSError:\n    fs = 'readonly'\n"
    "print(net, 'root' if os.getuid() == 0 else 'nonroot', fs, open('/etc/hostname').read().strip())\n"
)


def _docker() -> str:
    docker = shutil.which("docker")
    image = os.environ.get(_IMAGE_ENV)
    if docker is None or image is None:
        pytest.skip(f"docker CLI and {_IMAGE_ENV} are required")
    return docker


def _architecture(docker: str, image: str) -> str:
    return subprocess.run(
        [docker, "image", "inspect", "--format", "{{.Architecture}}", image],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _factory(root: Path, *, warm_spares: int = 0) -> DockerCodingEnvironmentFactory:
    docker = _docker()
    image = os.environ[_IMAGE_ENV]
    return DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(root, workspace_id="deferred-live"),
        toolchain_profile=DockerCodingToolchainProfile(
            profile_id="deferred-live",
            revision="1",
            image_identity=DockerImageIdentity(reference=image),
            platform_architecture=_architecture(docker, image),
        ),
        docker_path=docker,
        git_baseline=False,
        deferred=True,
        warm_spares=warm_spares,
    )


class Probe(Tool):
    spec = ToolSpec(
        name="probe",
        description="Probe the sandbox.",
        input_schema={"type": "object", "properties": {}},
        effect=ToolEffect.NONE,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="live.probe", behavior_version="1", implementation_version="1"
        ),
    )
    outputs: list[str] = []

    async def run(self, ctx, args) -> ToolResult:
        result = await ctx.runner.exec(ExecCommand(argv=["python3", "-c", _PROBE]), timeout_s=60)
        Probe.outputs.append(result.stdout.strip())
        return ToolResult(content=result.stdout.strip())


def _app(factory, store, *, use_tool: bool) -> CayuApp:
    turns = [
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({"finish_reason": "stop"})]
    ]
    if use_tool:
        turns.insert(
            0,
            [
                ModelStreamEvent.tool_call(id="probe-1", name="probe", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(ScriptedModelProvider(turns), default=True)
    app.register_environment_factory(
        EnvironmentSpec(
            name="sandbox",
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="live.sandbox", behavior_version="1", implementation_version="1"
            ),
        ),
        factory,
        default=True,
    )
    app.register_agent(
        AgentSpec(name="agent", model="scripted"),
        tools=[Probe()],
        tool_exposure_policy=StaticToolExposurePolicy(profile_id="live", tools=("probe",)),
        tool_policy=StaticToolPolicy(),
    )
    return app


def _coding_containers(docker: str) -> set[str]:
    listed = subprocess.run(
        [docker, "ps", "-aq", "--no-trunc", "--filter", "name=cayu-coding-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return set(listed.split())


def test_deferred_runs_create_containers_only_on_use_and_remove_them(tmp_path) -> None:
    docker = _docker()
    factory = _factory(tmp_path)
    before = _coding_containers(docker)

    async def run(use_tool: bool):
        outcome = await run_to_completion(
            _app(factory, InMemorySessionStore(), use_tool=use_tool),
            RunRequest(agent_name="agent", messages=[Message.text("user", "go")]),
        )
        assert outcome.ok, outcome.error

    Probe.outputs.clear()
    asyncio.run(run(False))
    assert Probe.outputs == []
    asyncio.run(run(True))
    network, user, filesystem, _host = Probe.outputs[0].split()
    assert (network, user, filesystem) == ("blocked", "nonroot", "readonly")
    assert _coding_containers(docker) == before


def test_pooled_runs_are_isolated_fresh_containers_and_shutdown_removes_spares(tmp_path) -> None:
    docker = _docker()
    factory = _factory(tmp_path, warm_spares=1)
    before = _coding_containers(docker)

    async def run_all():
        for _ in range(3):
            outcome = await run_to_completion(
                _app(factory, InMemorySessionStore(), use_tool=True),
                RunRequest(agent_name="agent", messages=[Message.text("user", "go")]),
            )
            assert outcome.ok, outcome.error
            await factory.warm_spare_pool.wait_for_refill()
        assert factory.warm_spare_pool.idle_count == 1
        await factory.release_idle_resources()

    Probe.outputs.clear()
    asyncio.run(run_all())

    hosts = []
    for output in Probe.outputs:
        network, user, filesystem, host = output.split()
        assert (network, user, filesystem) == ("blocked", "nonroot", "readonly")
        hosts.append(host)
    assert len(set(hosts)) == 3, "every run gets a container no other run used"
    assert _coding_containers(docker) == before


_SPARE_THEN_DIE = """
import asyncio, os, sys
sys.path[:0] = [{tests!r}, {src!r}]
from tests.environments.test_docker_deferred_live import _factory
from pathlib import Path

async def main():
    factory = _factory(Path({root!r}), warm_spares=1)
    factory.warm_spare_pool.schedule_refill()
    await factory.warm_spare_pool.wait_for_refill()
    print(factory.warm_spare_pool._spares[0].runner.container_id, flush=True)
    os._exit(0)

asyncio.run(main())
"""


def test_a_killed_process_leaves_a_spare_that_the_next_pool_reaps(tmp_path) -> None:
    docker = _docker()
    repository = Path(__file__).resolve().parents[2]
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            _SPARE_THEN_DIE.format(
                tests=str(repository), src=str(repository / "src"), root=str(tmp_path)
            ),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=os.environ.copy(),
    )
    spare_id = child.stdout.strip().splitlines()[-1]
    assert spare_id in _coding_containers(docker)

    pool = _factory(tmp_path, warm_spares=1).warm_spare_pool
    reaped = asyncio.run(pool.reap_stale())

    assert reaped
    assert spare_id not in _coding_containers(docker)


_CRASH_DURING_MATERIALIZATION = """
import asyncio, os, sys
sys.path[:0] = [{tests!r}, {src!r}]
from pathlib import Path
from cayu import Message, RunRequest, SQLiteSessionStore, run_to_completion
from cayu.environments.docker_coding import DockerCodingWorkspaceBinding
from tests.environments.test_docker_deferred_live import _app, _factory

async def crash(self, *args, **kwargs):
    # The container exists; the process dies before the binding completes.
    os._exit(73)

DockerCodingWorkspaceBinding.bind = crash

async def main():
    store = SQLiteSessionStore(Path({database!r}))
    app = _app(_factory(Path({root!r})), store, use_tool=True)
    await run_to_completion(
        app, RunRequest(agent_name="agent", session_id="crashed", messages=[Message.text("user", "go")])
    )

asyncio.run(main())
"""


def test_recovery_alone_reaps_a_container_left_between_creation_and_binding(tmp_path) -> None:
    docker = _docker()
    repository = Path(__file__).resolve().parents[2]
    # Keep the session database outside the synced sandbox folder.
    database = tmp_path / "sessions.sqlite3"
    source = tmp_path / "source"
    source.mkdir()
    before = _coding_containers(docker)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            _CRASH_DURING_MATERIALIZATION.format(
                tests=str(repository),
                src=str(repository / "src"),
                root=str(source),
                database=str(database),
            ),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env=os.environ.copy(),
    )
    assert child.returncode == 73, child.stderr[-2000:]
    leaked = _coding_containers(docker) - before
    assert len(leaked) == 1

    async def recover_only():
        store = SQLiteSessionStore(database)
        try:
            app = _app(_factory(source), store, use_tool=False)
            result = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="crashed", reason="process_killed")
            )
            assert result.status is SessionStatus.INTERRUPTED
            assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION in result.actions
            assert await app.drain_environment_cleanups(timeout_s=60)
        finally:
            await store.close()

    asyncio.run(recover_only())
    assert _coding_containers(docker) == before

    async def resume():
        store = SQLiteSessionStore(database)
        try:
            outcome = await run_to_completion(
                _app(_factory(source), store, use_tool=True),
                ResumeRequest(session_id="crashed", messages=[Message.text("user", "again")]),
            )
            assert outcome.ok, outcome.error
        finally:
            await store.close()

    # The session keeps its allocation and materializes a fresh container.
    Probe.outputs.clear()
    asyncio.run(resume())
    assert len(Probe.outputs) == 1
    assert _coding_containers(docker) == before


def _stable_evidence(runner) -> dict:
    """Evidence minus per-observation times and the per-container fingerprint."""

    def strip(value):
        if isinstance(value, dict):
            return {
                key: strip(item)
                for key, item in value.items()
                if key not in {"observed_at", "valid_until", "environment_fingerprint"}
            }
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return strip(runner.execution_admission_candidate().evidence.to_metadata())


def test_a_pooled_runner_carries_the_same_isolation_evidence_as_a_cold_start(tmp_path) -> None:
    _docker()
    factory = _factory(tmp_path, warm_spares=1)
    pool = factory.warm_spare_pool

    async def compare():
        requirements = pool._backend.base_requirements()
        cold = await factory._create_or_recover_runner(
            "cayu-coding-evidence-cold", immutable_mounts=(), requirements=requirements
        )
        try:
            pool.schedule_refill()
            await pool.wait_for_refill()
            spare = await pool.take("cayu-coding-evidence-pooled", requirements)
            assert spare is not None
            pooled = spare.runner
            try:
                assert pooled.container_id != cold.container_id
                assert _stable_evidence(pooled) == _stable_evidence(cold)
            finally:
                await pooled.close()
        finally:
            await cold.close()
            await factory.release_idle_resources()

    asyncio.run(compare())
