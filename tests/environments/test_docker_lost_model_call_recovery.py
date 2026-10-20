"""A lost model call is settled and its session resumed after Docker disposal.

Only the Docker transport is substituted: each container is a local directory
whose commands run on the host. The Docker coding factory, binding, disposal
retirement, model-completion settlement and recovery are the shipped owners.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import cayu.environments.docker_coding as docker_coding
from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.environments import DockerCodingEnvironmentFactory, EnvironmentSpec
from cayu.environments.docker_toolchains import DockerCodingToolchainProfile
from cayu.providers import (
    ModelProvider,
    ModelStreamDeadlineError,
    ModelStreamEvent,
    ProviderDeadlineKind,
    ProviderStreamDeadlineEvidence,
)
from cayu.runners.docker import DockerRunner, _DockerRuntimeEvidence
from cayu.runners.docker_workload import DockerImageIdentity, DockerWorkloadRestrictions
from cayu.runners.local import LocalRunner
from cayu.runtime._session_outcome_publication import SessionOutcomePublication
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    ModelCompletionManualRecoveryRequest,
    ResumeRequest,
)
from cayu.storage import SQLiteSessionStore
from cayu.tasks import TaskCreate, TaskStatus
from cayu.tasks.memory import InMemoryTaskStore
from cayu.workspaces import LocalWorkspace

_IMAGE = DockerImageIdentity(reference="test@sha256:" + "a" * 64)
_PROFILE = DockerCodingToolchainProfile(
    profile_id="lost-call-test",
    revision="1",
    image_identity=_IMAGE,
    platform_architecture="amd64",
)


def _identity(name: str) -> ExecutionProfileBehaviorIdentity:
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


class _LocalContainer(DockerRunner):
    """One Docker container whose guest is a local directory."""

    def __init__(self, root: Path, container_id: str, requirements) -> None:
        now = datetime.now(UTC)
        super().__init__(
            container_id,
            _container_id=container_id,
            docker_path="/usr/bin/true",
            default_cwd=str(root),
            _runtime_evidence=_DockerRuntimeEvidence(
                container_id=container_id,
                image_id=_IMAGE.reference.rsplit("@", 1)[1],
                image_reference=_IMAGE.reference,
                network_mode="none",
                default_cwd=str(root),
                runtime=None,
                seccomp_profile_sha256=None,
                restrictions=DockerWorkloadRestrictions(),
                image_identity=_IMAGE,
                toolchain_profile_fingerprint=_PROFILE.fingerprint,
                required_executables=requirements.executable_names(),
                executable_availability=tuple(
                    (name, True) for name in requirements.executable_names()
                ),
                immutable_input_mounts=(),
                observed_at=now,
                valid_until=now + timedelta(hours=1),
            ),
        )
        self.local = LocalRunner(root, inherit_env=True)
        self.root = root

    def resolve_cwd(self, cwd: str | None = None) -> str:
        del cwd
        return str(self.root)

    async def exec(self, command, **kwargs: Any):
        kwargs["cwd"] = None
        return await self.local.exec(command, **kwargs)

    async def exec_stream(self, command, **kwargs: Any):
        kwargs["cwd"] = None
        return await self.local.exec_stream(command, **kwargs)

    async def exec_system(self, command, **kwargs: Any):
        kwargs["cwd"] = None
        return await self.local.exec_system(command, **kwargs)

    async def collect_execution_admission_candidate(self):
        return self.execution_admission_candidate()

    async def close(self) -> None:
        await self.local.close()
        self._closed = True
        self._container_removed = True


class _LosesFirstCall(ModelProvider):
    """The first call reaches the absolute stream deadline; later calls finish."""

    name = "lost-call"

    def __init__(self) -> None:
        self.requests = 0

    @property
    def execution_profile_identity(self):
        return _identity("tests:lost-call:provider")

    async def stream(self, request):
        self.requests += 1
        if self.requests == 1:
            yield ModelStreamEvent.text_delta("A long answer ...")
            raise ModelStreamDeadlineError(
                provider=self.name,
                evidence=ProviderStreamDeadlineEvidence(
                    deadline_kind=ProviderDeadlineKind.ABSOLUTE,
                    configured_timeout_s=600.0,
                    elapsed_s=600.0,
                    last_progress_kind=None,
                    last_progress_elapsed_s=None,
                    last_progress_at=None,
                ),
            )
        yield ModelStreamEvent.text_delta("Redone.")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _app(root: Path, monkeypatch, tasks):
    # Stand in for a process lost after model.error, before the live run
    # could settle the deadline, which leaves the dispatched call fenced.
    async def keep_fenced(self, context, exc):
        return False

    monkeypatch.setattr(
        SessionOutcomePublication, "_stream_deadline_settles_as_failure", keep_fenced
    )
    (root / "source").mkdir()
    containers: list[_LocalContainer] = []

    async def create(_name, *, requirements, **kwargs):
        guest = root / f"guest-{len(containers)}"
        guest.mkdir()
        container = _LocalContainer(guest, f"{len(containers):x}" * 64, requirements)
        vars(container)["_cayu_execution_environment_authority"] = (
            factory.execution_environment_authority()
        )
        containers.append(container)
        return container

    async def admitted(*args, **kwargs):
        return None

    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(root / "source", workspace_id="source"),
        toolchain_profile=_PROFILE,
        git_baseline=False,
        docker_path="/usr/bin/true",
    )
    monkeypatch.setattr(factory, "_create_or_recover_runner", create)
    monkeypatch.setattr(docker_coding, "_run_toolchain_admission_probes", admitted)
    store = SQLiteSessionStore(root / "sessions.sqlite")
    app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
    provider = _LosesFirstCall()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="builder", model="test-model"))
    app.register_environment_factory(
        EnvironmentSpec(name="coding", execution_profile_identity=_identity("tests:lost-call")),
        factory,
        default=True,
    )
    return app, store, provider, containers


async def _drain(stream) -> list:
    return [event async for event in stream]


@pytest.mark.parametrize("task_bound", [True, False])
def test_manual_settlement_resumes_session_after_run_disposed_its_container(
    tmp_path, monkeypatch, task_bound
):
    async def scenario():
        tasks = InMemoryTaskStore()
        app, store, provider, containers = _app(tmp_path, monkeypatch, tasks)
        try:
            task_fields: dict[str, Any] = {}
            if task_bound:
                await tasks.create_task(TaskCreate(task_id="build", type="build"))
                task = await tasks.claim_task("worker-1")
                task_fields = {
                    "task_id": task.id,
                    "task_worker_id": "worker-1",
                    "task_lease_expires_at": task.lease_expires_at,
                }
            with pytest.raises(ModelStreamDeadlineError):
                await _drain(
                    app.run(
                        RunRequest(
                            agent_name="builder",
                            session_id="build-session",
                            messages=[Message.text("user", "build it")],
                            **task_fields,
                        )
                    )
                )
            # The run's teardown already disposed of the container and retired
            # its allocation while the lost call keeps the session fenced.
            session = await store.load("build-session")
            assert session.status is SessionStatus.RUNNING
            [container] = containers
            assert container.container_removed
            checkpoint = await store.load_checkpoint("build-session")
            assert "environment_factory_retired_disposals" in checkpoint
            assert "environment_factory_allocation_owner" not in checkpoint
            active = await store.load_active_model_completion_stage("build-session")
            assert active is not None and active.stage.state == "in_flight"

            settled = await app.recover_model_completion_stage(
                ModelCompletionManualRecoveryRequest(
                    session_id="build-session",
                    stage_id=active.stage.stage_id,
                    expected_run_epoch=session.run_epoch,
                    terminal_status=SessionStatus.INTERRUPTED,
                )
            )
            assert settled.session.status is SessionStatus.INTERRUPTED
            recovered = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="build-session")
            )
            assert recovered.status is SessionStatus.INTERRUPTED
            assert recovered.actions == (
                IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
            )
            assert provider.requests == 1

            resume_fields: dict[str, Any] = {}
            if task_bound:
                # Recovery neither released nor reassigned the claimed task.
                current = await tasks.load_task("build")
                assert current.status is TaskStatus.RUNNING
                assert current.worker_id == "worker-1"
                assert current.lease_expires_at == task.lease_expires_at
                resume_fields = {
                    "task_worker_id": "worker-1",
                    "task_handoff_id": current.interrupted_handoff_id,
                }
            await _drain(
                app.resume(
                    ResumeRequest(
                        session_id="build-session",
                        messages=[Message.text("user", "Redo the lost step.")],
                        **resume_fields,
                    )
                )
            )
            resumed = await store.load("build-session")
            assert resumed.id == session.id and resumed.instance_id == session.instance_id
            assert resumed.status is SessionStatus.COMPLETED
            assert provider.requests == 2
            # The continuation ran in a fresh container, which it released in turn.
            assert len(containers) == 2 and all(item.container_removed for item in containers)
        finally:
            await app.drain_environment_cleanups(timeout_s=10)
            await store.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))


def test_settlement_refuses_a_retired_disposal_of_another_allocation(tmp_path, monkeypatch):
    async def scenario():
        app, store, _provider, _containers = _app(tmp_path, monkeypatch, None)
        try:
            with pytest.raises(ModelStreamDeadlineError):
                await _drain(
                    app.run(
                        RunRequest(
                            agent_name="builder",
                            session_id="build-session",
                            messages=[Message.text("user", "build it")],
                        )
                    )
                )
            session = await store.load("build-session")
            active = await store.load_active_model_completion_stage("build-session")
            await app.recover_model_completion_stage(
                ModelCompletionManualRecoveryRequest(
                    session_id="build-session",
                    stage_id=active.stage.stage_id,
                    expected_run_epoch=session.run_epoch,
                    terminal_status=SessionStatus.INTERRUPTED,
                )
            )

            def foreign(_session, checkpoint):
                retired = checkpoint["environment_factory_retired_disposals"]["coding"]
                retired["reconnect_metadata"]["allocation_fingerprint"] = "f" * 64
                return checkpoint

            await store.transform_checkpoint("build-session", foreign)
            # Disposal of a different allocation is no evidence for this binding.
            with pytest.raises(RuntimeError, match="session-owned durable allocation identity"):
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="build-session")
                )
            events = await store.load_events("build-session")
            assert not any(event.type is EventType.SESSION_COMPLETED for event in events)
        finally:
            await app.drain_environment_cleanups(timeout_s=10)
            await store.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))


def test_abandoned_sweep_settles_a_task_bound_call_without_taking_the_task(tmp_path, monkeypatch):
    async def scenario():
        tasks = InMemoryTaskStore()
        app, store, provider, containers = _app(tmp_path, monkeypatch, tasks)
        try:
            await tasks.create_task(TaskCreate(task_id="build", type="build"))
            task = await tasks.claim_task("worker-1")
            with pytest.raises(ModelStreamDeadlineError):
                await _drain(
                    app.run(
                        RunRequest(
                            agent_name="builder",
                            session_id="build-session",
                            messages=[Message.text("user", "build it")],
                            task_id=task.id,
                            task_worker_id="worker-1",
                            task_lease_expires_at=task.lease_expires_at,
                        )
                    )
                )
            [result] = await app.recover_abandoned_sessions(inactive_for_seconds=0)
            assert result.status is SessionStatus.INTERRUPTED
            assert IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED in result.actions
            assert await store.load_active_model_completion_stage("build-session") is None
            current = await tasks.load_task("build")
            assert current.status is TaskStatus.RUNNING
            assert current.worker_id == "worker-1"
            assert current.lease_expires_at == task.lease_expires_at
            await _drain(
                app.resume(
                    ResumeRequest(
                        session_id="build-session",
                        task_worker_id="worker-1",
                        task_handoff_id=current.interrupted_handoff_id,
                        messages=[Message.text("user", "Redo the lost step.")],
                    )
                )
            )
            assert (await store.load("build-session")).status is SessionStatus.COMPLETED
            assert provider.requests == 2
            assert len(containers) == 2
        finally:
            await app.drain_environment_cleanups(timeout_s=10)
            await store.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))
