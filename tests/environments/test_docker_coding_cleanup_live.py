from __future__ import annotations

import asyncio
import json
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_docker_coding_live import _configuration_or_skip

from cayu import (
    AgentSpec,
    CayuApp,
    DockerCodingEnvironmentFactory,
    DockerCodingWorkspaceBinding,
    DockerImageIdentity,
    DockerRunner,
    EnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentSpec,
    ExecCommand,
    ExecutionProfileBehaviorIdentity,
    ImmutableInputStore,
    LocalWorkspace,
    Message,
    ModelProvider,
    ModelStreamEvent,
    RunRequest,
    SQLiteSessionStore,
    inspect_local_immutable_input,
)

pytestmark = pytest.mark.process


@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize(
    "mode",
    [
        "clean",
        "scratch",
        "close_failure",
        "close_cancel",
        "ignored_failure",
        "copyback_failure",
        "missing_disposal_hook",
    ],
)
def test_terminal_docker_cleanup_drain_matches_exact_physical_census(
    tmp_path: Path,
    monkeypatch,
    attached: bool,
    mode: str,
) -> None:
    docker_path, image, image_id = _configuration_or_skip()
    architecture = subprocess.check_output(
        [docker_path, "image", "inspect", "--format", "{{.Architecture}}", image],
        text=True,
    ).strip()
    assert architecture in ("amd64", "arm64")
    identity = DockerImageIdentity(reference=image, content_digest=image_id)
    source = tmp_path / "source"
    source.mkdir()
    (source / ".gitignore").write_bytes(b"scratch/\n")
    (source / "code.py").write_bytes(b"value = 1\n")
    inputs = ()
    input_store = None
    if attached:
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "runtime.txt").write_bytes(b"immutable fixture")
        inputs = (
            inspect_local_immutable_input(
                runtime,
                target_path="/opt/cayu/inputs/runtime",
                policy_fingerprint="sha256:" + "a" * 64,
                runtime_compatibility_fingerprint=identity.fingerprint,
                authorization_scope_fingerprint="sha256:" + "c" * 64,
            ),
        )
        input_store = ImmutableInputStore(tmp_path / "managed")
    inner = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(source),
        toolchain_profile=docker_toolchain_profile(
            image_identity=identity,
            platform_architecture="arm64" if architecture == "arm64" else "amd64",
        ),
        immutable_inputs=inputs,
        immutable_input_store=input_store,
        immutable_input_runtime_compatibility_fingerprint=identity.fingerprint
        if attached
        else None,
        docker_path=docker_path,
    )
    results = []
    close_blocked = mode in ("close_failure", "close_cancel")

    class CapturingFactory(EnvironmentFactory):
        @property
        def execution_profile_identity(self):
            return inner.execution_profile_identity

        def construction_admission_candidate(self):
            return inner.construction_admission_candidate()

        def execution_admission_candidate(self, request):
            return inner.execution_admission_candidate(request)

        async def recover_finalization_disposal(self, request, state):
            await inner.recover_finalization_disposal(request, state)

        async def create(self, request):
            result = await inner.create(request)
            binding = result.environment.binding
            assert isinstance(binding, DockerCodingWorkspaceBinding)
            binding.sync_back = (
                "always" if mode in ("ignored_failure", "copyback_failure") else "never"
            )
            results.append(result)
            inspection = json.loads(
                subprocess.check_output(
                    [docker_path, "container", "inspect", result.metadata["container_id"]],
                    text=True,
                )
            )[0]
            assert inspection["HostConfig"]["NetworkMode"] == "none"
            assert all(mount["Type"] != "volume" for mount in inspection["Mounts"])
            runner = result.environment.runner
            assert runner is not None
            original_close = runner.close

            async def close():
                if close_blocked:
                    if mode == "close_cancel":
                        raise asyncio.CancelledError("fixture close cancelled")
                    raise OSError("fixture close unavailable")
                await original_close()

            monkeypatch.setattr(runner, "close", close)
            return result

    original_disposal_hook = CapturingFactory.recover_finalization_disposal
    if mode == "missing_disposal_hook":
        monkeypatch.setattr(
            CapturingFactory,
            "recover_finalization_disposal",
            EnvironmentFactory.recover_finalization_disposal,
        )

    class FixedProvider(ModelProvider):
        name = "cleanup-fixture"

        @property
        def execution_profile_identity(self):
            return ExecutionProfileBehaviorIdentity(
                name="cleanup-fixture",
                behavior_version="1",
                implementation_version="1",
            )

        async def stream(self, request):
            if mode in ("scratch", "ignored_failure"):
                runner = results[-1].environment.runner
                assert runner is not None
                result = await runner.exec(
                    ExecCommand.process(
                        "python3",
                        "-c",
                        "from pathlib import Path; Path('scratch').mkdir(); "
                        "Path('scratch/note.txt').write_text('fixture')",
                    )
                )
                assert result.exit_code == 0
            if mode == "copyback_failure":
                runner = results[-1].environment.runner
                assert runner is not None
                result = await runner.exec(
                    ExecCommand.process(
                        "python3",
                        "-c",
                        "from pathlib import Path; Path('code.py').write_text('candidate')",
                    )
                )
                assert result.exit_code == 0
                (source / "code.py").write_bytes(b"external source edit")
            yield ModelStreamEvent.text_delta("fixture complete")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    def exists(container_id):
        probe = subprocess.run(
            [docker_path, "container", "inspect", container_id],
            capture_output=True,
            text=True,
        )
        if probe.returncode:
            assert "No such container" in probe.stderr or "No such object" in probe.stderr
        return probe.returncode == 0

    async def run():
        nonlocal close_blocked
        store = SQLiteSessionStore(tmp_path / "sessions.db")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FixedProvider(), default=True)
        app.register_agent(AgentSpec(name="probe", model="fixture"))
        app.register_environment_factory(
            EnvironmentSpec(name="coding"), CapturingFactory(), default=True
        )
        session_id = "terminal-cleanup-" + sha256(str(tmp_path).encode()).hexdigest()[:16]
        sibling = None
        if mode == "clean" and not attached:
            sibling = await inner.create(
                EnvironmentFactoryRequest(
                    session_id=session_id + "-sibling",
                    agent_name="probe",
                    environment_name="coding",
                )
            )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="probe",
                        messages=[Message.text("user", "Complete.")],
                    )
                )
            ]
            assert len(results) == 1
            container_id = results[0].metadata["container_id"]
            event_types = {str(event.type) for event in events}
            if mode in (
                "close_failure",
                "close_cancel",
                "ignored_failure",
                "copyback_failure",
                "missing_disposal_hook",
            ):
                assert "session.failed" in event_types
                assert "environment.binding.finalize_failed" in event_types
                assert await app.drain_environment_cleanups(timeout_s=0.2) is False
                assert exists(container_id)
                checkpoint = await store.load_checkpoint(session_id)
                assert checkpoint is not None
                assert "pending_completion_finalization" in checkpoint
                if mode == "missing_disposal_hook":
                    assert "disposal_state" not in checkpoint["pending_completion_finalization"]
                    monkeypatch.setattr(
                        CapturingFactory,
                        "recover_finalization_disposal",
                        original_disposal_hook,
                    )
                if input_store is not None:
                    assert input_store.inspect()[0].reference_count == 1
                close_blocked = False
                if mode in ("ignored_failure", "copyback_failure"):
                    # Preserve the failed output until the owner explicitly changes
                    # its publication policy; no operator removes the container.
                    binding = results[0].environment.binding
                    assert isinstance(binding, DockerCodingWorkspaceBinding)
                    binding.sync_back = "never"
            else:
                assert "session.completed" in event_types, [
                    (event.type, event.payload) for event in events
                ]
                assert "environment.binding.finalize_failed" not in event_types
            assert await app.drain_environment_cleanups(timeout_s=3) is True
            assert not exists(container_id)
            assert await app.drain_environment_cleanups(timeout_s=3) is True
            expected = b"external source edit" if mode == "copyback_failure" else b"value = 1\n"
            assert (source / "code.py").read_bytes() == expected
            if sibling is not None:
                assert exists(sibling.metadata["container_id"])
            binding = results[0].environment.binding
            assert isinstance(binding, DockerCodingWorkspaceBinding)
            assert binding._states == {}
            assert binding._coding_finalize_states == {}
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            assert "pending_completion_finalization" not in checkpoint
            assert not checkpoint.get("environment_factory_allocation_intents")
            assert not (source / "scratch").exists()
            if input_store is not None:
                assert input_store.inspect()[0].reference_count == 0
        finally:
            close_blocked = False
            for result in results:
                assert result.release is not None
                await result.release(EnvironmentFactoryReleaseAction.DISCARD)
            if sibling is not None:
                assert sibling.release is not None
                await sibling.release(EnvironmentFactoryReleaseAction.DISCARD)
            await app.drain_environment_cleanups(timeout_s=3)
            await store.close()

    asyncio.run(run())


def _restart_fixture_app(
    root,
    docker_path,
    image,
    image_id,
    *,
    crash,
    crash_stage="before_close",
    attached=False,
    sync_back="never",
):
    import json
    import os

    architecture = subprocess.check_output(
        [docker_path, "image", "inspect", "--format", "{{.Architecture}}", image],
        text=True,
    ).strip()
    identity = DockerImageIdentity(reference=image, content_digest=image_id)
    inputs = ()
    input_store = None
    if attached:
        inputs = (
            inspect_local_immutable_input(
                root / "runtime",
                target_path="/opt/cayu/inputs/runtime",
                policy_fingerprint="sha256:" + "a" * 64,
                runtime_compatibility_fingerprint=identity.fingerprint,
                authorization_scope_fingerprint="sha256:" + "c" * 64,
            ),
        )
        input_store = ImmutableInputStore(root / "managed")
    inner = DockerCodingEnvironmentFactory(
        immutable_inputs=inputs,
        immutable_input_store=input_store,
        immutable_input_runtime_compatibility_fingerprint=(
            identity.fingerprint if attached else None
        ),
        source_workspace=LocalWorkspace(root / "source"),
        toolchain_profile=docker_toolchain_profile(
            image_identity=DockerImageIdentity(reference=image, content_digest=image_id),
            platform_architecture="arm64" if architecture == "arm64" else "amd64",
        ),
        docker_path=docker_path,
    )
    results = []

    class Factory(EnvironmentFactory):
        @property
        def execution_profile_identity(self):
            return inner.execution_profile_identity

        def construction_admission_candidate(self):
            return inner.construction_admission_candidate()

        def execution_admission_candidate(self, request):
            return inner.execution_admission_candidate(request)

        async def recover_finalization_disposal(self, request, state):
            await inner.recover_finalization_disposal(request, state)

        def allocation_scope(self, request):
            return inner.allocation_scope(request)

        async def create_recoverable(self, request, allocation):
            return self.capture(await inner.create_recoverable(request, allocation))

        async def create(self, request):
            return self.capture(await inner.create(request))

        def capture(self, result):
            results.append(result)
            binding = result.environment.binding
            assert isinstance(binding, DockerCodingWorkspaceBinding)
            binding.sync_back = sync_back
            if crash:
                (root / "allocation.json").write_text(json.dumps(dict(result.metadata)))
                runner = result.environment.runner
                assert runner is not None

                original_close = runner.close

                async def close() -> None:
                    if crash_stage == "after_close":
                        await original_close()
                    os._exit(73)

                pytest.MonkeyPatch().setattr(runner, "close", close)
            return result

    class Provider(ModelProvider):
        name = "restart-fixture"

        @property
        def execution_profile_identity(self):
            return ExecutionProfileBehaviorIdentity(
                name="restart-fixture",
                behavior_version="1",
                implementation_version="1",
            )

        async def stream(self, request):
            assert crash, "Recovery must not redispatch the provider"
            runner = results[-1].environment.runner
            assert runner is not None
            result = await runner.exec(
                ExecCommand.process(
                    "python3",
                    "-c",
                    "from pathlib import Path; Path('code.py').write_text('candidate')",
                )
            )
            assert result.exit_code == 0
            yield ModelStreamEvent.text_delta("complete")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    store = SQLiteSessionStore(root / "sessions.db")
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(Provider(), default=True)
    app.register_agent(AgentSpec(name="probe", model="fixture"))
    app.register_environment_factory(EnvironmentSpec(name="coding"), Factory(), default=True)
    return app, store, results


@pytest.mark.parametrize("crash_stage", ["before_close", "after_close"])
@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("sync_back", ["never", "always"])
def test_real_docker_process_loss_recovers_exact_disposal(
    tmp_path: Path,
    crash_stage: str,
    attached: bool,
    sync_back: str,
):
    import json
    import os
    import sys

    from cayu import IncompleteSessionRecoveryAction, IncompleteSessionRecoveryRequest

    docker_path, image, image_id = _configuration_or_skip()
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "code.py").write_bytes(b"unchanged")
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "input.txt").write_text("immutable")
    (tmp_path / "options.json").write_text(
        json.dumps(
            {
                "crash_stage": crash_stage,
                "attached": attached,
                "sync_back": sync_back,
            }
        )
    )
    worker = """
import asyncio, sys, json
from pathlib import Path
from cayu import Message, RunRequest
from tests.environments.test_docker_coding_cleanup_live import _restart_fixture_app
async def run():
    root = Path(sys.argv[1])
    app, _, _ = _restart_fixture_app(root, *sys.argv[2:], crash=True,
                                    **json.loads((root / 'options.json').read_text()))
    async for _ in app.run(RunRequest(session_id='crashed-cleanup', agent_name='probe',
                                     messages=[Message.text('user', 'Complete.')])):
        pass
asyncio.run(run())
"""
    result = subprocess.run(
        [sys.executable, "-c", worker, str(tmp_path), docker_path, image, image_id],
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src") + os.pathsep + str(Path.cwd())},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 73, result.stderr
    container_id = json.loads((tmp_path / "allocation.json").read_text())["container_id"]
    inspection = subprocess.run(
        [docker_path, "container", "inspect", container_id],
        capture_output=True,
        text=True,
    )
    assert (inspection.returncode == 0) is (crash_stage == "before_close")
    expected_source = b"candidate" if sync_back == "always" else b"unchanged"
    assert (tmp_path / "source" / "code.py").read_bytes() == expected_source
    (tmp_path / "source" / "code.py").write_bytes(b"external edit after publication")

    async def recover():
        app, store, results = _restart_fixture_app(
            tmp_path,
            docker_path,
            image,
            image_id,
            crash=False,
            crash_stage=crash_stage,
            attached=attached,
            sync_back=sync_back,
        )
        checkpoint = await store.load_checkpoint("crashed-cleanup")
        assert checkpoint is not None
        assert "disposal_state" in checkpoint["pending_completion_finalization"]
        assert checkpoint.get("environment_factory_allocation_receipts")
        assert not checkpoint.get("environment_factory_allocation_intents")
        try:
            recovery = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="crashed-cleanup",
                    reason="fixture_process_loss",
                )
            )
            assert recovery.actions == (
                IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_FINALIZATION,
            )
            assert await app.drain_environment_cleanups(timeout_s=3) is True
            checkpoint = await store.load_checkpoint("crashed-cleanup")
            assert checkpoint is not None
            assert "pending_completion_finalization" not in checkpoint
            assert results == [], "Disposal recovery must not recreate or reconnect the guest"
            if attached:
                assert ImmutableInputStore(tmp_path / "managed").inspect()[0].reference_count == 0
            inspection = subprocess.run(
                [docker_path, "container", "inspect", container_id],
                capture_output=True,
                text=True,
            )
            assert inspection.returncode != 0
            assert "No such" in inspection.stderr
            repeated = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="crashed-cleanup",
                    reason="fixture_repeated_recovery",
                )
            )
            assert (
                IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_FINALIZATION
                not in repeated.actions
            )
            assert await app.drain_environment_cleanups(timeout_s=3) is True
        finally:
            for result in results:
                assert result.release is not None
                await result.release(EnvironmentFactoryReleaseAction.DISCARD)
            await store.close()

    try:
        asyncio.run(recover())
    finally:
        # Failed assertions must still clean only the worker's positively
        # identified allocation. Successful acceptance above precedes teardown.
        remaining = subprocess.run(
            [docker_path, "container", "inspect", container_id],
            capture_output=True,
            text=True,
        )
        if remaining.returncode == 0:
            asyncio.run(
                DockerRunner(container_id, close_action="remove", docker_path=docker_path).close()
            )
    assert (tmp_path / "source" / "code.py").read_bytes() == b"external edit after publication"
