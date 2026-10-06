from __future__ import annotations

import asyncio

import pytest
from tests.core.test_tool_result_projection import _collect, _FakeProvider

from cayu import RunRequest
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.configuration import CayuConfig, ToolExecutionConfig
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.vaults.static import StaticVault


class _OwnedStubbornVault(StaticVault):
    """An owned vault whose lookup finishes even after it is cancelled."""

    def __init__(self) -> None:
        super().__init__({"api_key": "stubborn-secret-value"})
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        self.resolved_after_close: list[bool] = []

    async def resolve(self, ref, *, scope=None):
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            task = asyncio.current_task()
            assert task is not None
            while task.cancelling():
                task.uncancel()
            await self.release.wait()
        self.resolved_after_close.append(self.closed)
        return await super().resolve(ref, scope=scope)

    async def close(self) -> None:
        self.closed = True


class _SecretTool(Tool):
    spec = ToolSpec(
        name="secret_tool",
        description="Resolve a secret.",
        input_schema={"type": "object"},
        effect=ToolEffect.NONE,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.vault is not None
        await ctx.vault.resolve(await ctx.vault.get("api_key"))
        return ToolResult(content="resolved")


@pytest.mark.parametrize("released", ["after_shutdown", "within_budget"])
def test_shutdown_waits_for_a_secret_resolution_its_tool_timed_out_on(released: str) -> None:
    async def scenario() -> None:
        vault = _OwnedStubbornVault()
        provider = _FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call", name="secret_tool", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        app = CayuApp(
            enable_logging=False,
            owned_resources=(vault,),
            config=CayuConfig(tool_execution=ToolExecutionConfig(tool_timeout_seconds=0.2)),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), vault=vault), default=True
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[_SecretTool()])
        await asyncio.wait_for(
            _collect(
                app.run(
                    RunRequest(
                        session_id="secret",
                        agent_name="assistant",
                        messages=[Message.text("user", "run")],
                    )
                )
            ),
            30,
        )
        # The tool timed out while the vault lookup was still running.
        assert vault.started.is_set() and vault.resolved_after_close == []
        if released == "within_budget":
            # The lookup finishes while shutdown waits: the first attempt settles.
            asyncio.get_running_loop().call_later(0.2, vault.release.set)
            outcome = await app.aclose(timeout_s=5)
            assert outcome.settled and vault.closed
            assert vault.resolved_after_close == [False]
            return
        try:
            outcome = await app.aclose(timeout_s=0.5)
            step = outcome.step("environment_cleanups")
            assert step is not None and step.status == "incomplete"
            assert outcome.owned_resources == "retained" and not vault.closed
        finally:
            vault.release.set()
        assert (await app.aclose(timeout_s=5)).settled and vault.closed
        assert vault.resolved_after_close == [False]

    asyncio.run(scenario())


def test_shutdown_waits_for_an_artifact_recovery_read_that_timed_out(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from tests.core.test_workspace_mutation_receipts import (
        _BulkProvider,
        _BulkWriteTool,
        _portable_environment_spec,
        _WorkspaceObservationProcessLoss,
        _WorkspaceObservationProcessLossStore,
        collect_events,
        interrupt_and_release_test_invocation,
    )

    from cayu.artifacts import local as local_artifacts
    from cayu.artifacts.local import LocalArtifactStore
    from cayu.environments.bindings import DeterministicWorkspaceBinding
    from cayu.runtime import _recovery_coordinator
    from cayu.sessions.base import IncompleteSessionRecoveryRequest
    from cayu.workspaces.local import LocalWorkspace

    monkeypatch.setattr(
        _recovery_coordinator, "_WORKSPACE_ARTIFACT_RECOVERY_READ_TIMEOUT_SECONDS", 0.2
    )

    class OwnedResource:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class StallingArtifactStore(LocalArtifactStore):
        """The real store, whose file read runs in a thread nothing can cancel."""

        def __init__(self, root, *, owned: OwnedResource) -> None:
            super().__init__(root, store_id="artifact-store")
            self.owned = owned
            self.started = threading.Event()
            self.gate = threading.Event()
            self.read_after_close: list[bool] = []

    stalling: list[StallingArtifactStore] = []
    read_artifact = local_artifacts._read_artifact

    def blocking_read(*args, **kwargs):
        if stalling:
            store = stalling[0]
            store.started.set()
            store.gate.wait(30)
            store.read_after_close.append(store.owned.closed)
        return read_artifact(*args, **kwargs)

    monkeypatch.setattr(local_artifacts, "_read_artifact", blocking_read)

    def build_app(store, artifacts, *, owned=()) -> CayuApp:
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=owned)
        app.register_provider(_BulkProvider(), default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path / "workspace", workspace_id="workspace"),
                artifact_store=artifacts,
                binding=DeterministicWorkspaceBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"), tools=[_BulkWriteTool()]
        )
        return app

    async def scenario() -> None:
        (tmp_path / "workspace").mkdir()
        store = _WorkspaceObservationProcessLossStore(phase="artifact-revision-after-published")
        first_artifacts = StallingArtifactStore(tmp_path / "artifacts", owned=OwnedResource())
        first_artifacts.gate.set()
        # A process loss leaves a published artifact for recovery to read back.
        with pytest.raises(_WorkspaceObservationProcessLoss):
            await collect_events(
                build_app(store, first_artifacts),
                RunRequest(
                    agent_name="assistant",
                    session_id="artifact-recovery",
                    messages=[Message.text("user", "write")],
                ),
            )
        store.failed = True
        await interrupt_and_release_test_invocation(store, "artifact-recovery")
        owned = OwnedResource()
        artifacts = StallingArtifactStore(tmp_path / "artifacts", owned=owned)
        stalling.append(artifacts)
        app = build_app(store, artifacts, owned=(owned,))
        recovery = asyncio.create_task(
            app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="artifact-recovery")
            )
        )
        try:
            # Recovery stops waiting for the read once it times out.
            await asyncio.wait_for(asyncio.gather(recovery, return_exceptions=True), 20)
            assert artifacts.started.is_set() and artifacts.read_after_close == []
            first = await app.aclose(timeout_s=0.5)
            step = first.step("recovery_cleanups")
            assert step is not None and step.status == "incomplete"
            assert first.owned_resources == "retained" and not owned.closed
        finally:
            artifacts.gate.set()
        assert (await app.aclose(timeout_s=10)).settled and owned.closed
        assert artifacts.read_after_close == [False]

    asyncio.run(scenario())


@pytest.mark.parametrize("closed_first", [False, True])
def test_shutdown_waits_for_stream_acceptance_bookkeeping_after_its_run_ended(
    closed_first: bool,
) -> None:
    import gc
    import warnings

    from cayu.events import Event, EventType
    from cayu.exceptions import TerminalEventPublicationUncertain
    from cayu.runtime import ApplicationAdmissionsSealed
    from cayu.server.routes import _start_detached_event_stream_response
    from cayu.sessions.base import InMemorySessionStore

    class ClosableStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False
            self.writes_after_close: list[bool] = []

        async def close(self) -> None:
            self.closed = True

    async def scenario() -> None:
        store = ClosableStore()
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=(store,))
        terminal = Event(type=EventType.SESSION_FAILED, session_id="stream", payload={})

        async def stream():
            # The run ends, releasing its lease, before any event was delivered.
            raise TerminalEventPublicationUncertain(
                event=terminal,
                publication_failure=OSError("publication"),
                reconciliation_failure=OSError("reconciliation"),
            )
            yield  # pragma: no cover

        async def bookkeeping(_error) -> None:
            # A route-owned store write, such as recording the accepted run.
            await asyncio.sleep(0.3)
            store.writes_after_close.append(store.closed)

        if closed_first:
            assert (await app.aclose(timeout_s=5)).settled and store.closed
        acceptance = asyncio.get_running_loop().create_future()
        _response, pump, _abandon = _start_detached_event_stream_response(
            stream(),
            cayu_app=app,
            session_id="stream",
            acceptance=acceptance,
            after_terminal_publication_uncertain=bookkeeping,
        )
        if closed_first:
            # A closed application refuses the write instead of running it.
            await asyncio.gather(pump, return_exceptions=True)
            error, reason = acceptance.result()
            assert isinstance(error, ApplicationAdmissionsSealed)
            assert reason == "terminal_uncertainty_acceptance_failed"
            assert store.writes_after_close == []
            return
        await asyncio.sleep(0.05)
        outcome = await app.aclose(timeout_s=5)
        assert outcome.settled and store.closed
        assert store.writes_after_close == [False]
        await asyncio.gather(pump, return_exceptions=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(scenario())
        gc.collect()
    # A refused bookkeeping coroutine is closed, not left un-awaited.
    assert not [warning for warning in caught if "never awaited" in str(warning.message)]


class _ClosableResource:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def isolated_cleanups(monkeypatch: pytest.MonkeyPatch):
    from cayu.runtime import _isolated_tool_process

    # The retained cleanups are process-wide; keep this test's own.
    monkeypatch.setattr(_isolated_tool_process, "_RETAINED_ISOLATED_TOOL_OWNERS", {})
    monkeypatch.setattr(_isolated_tool_process, "_RETAINED_ISOLATED_TOOL_APPLICATIONS", {})
    return _isolated_tool_process


async def _retain_from_operation(app: CayuApp, isolated_cleanups, cleanup, **kwargs) -> None:
    """Retain a cleanup the way a tool call of ``app`` does: within its operation."""

    async def step() -> None:
        isolated_cleanups._retain_task(asyncio.create_task(cleanup()), **kwargs)

    await app._run_worker_step(step)


@pytest.mark.parametrize("released", ["after_shutdown", "within_budget"])
def test_shutdown_waits_for_a_retained_isolated_tool_cleanup(
    isolated_cleanups, released: str
) -> None:
    async def scenario() -> None:
        owned = _ClosableResource()
        app = CayuApp(enable_logging=False, owned_resources=(owned,))
        release = asyncio.Event()
        finished_after_close: list[bool] = []

        async def cleanup() -> None:
            await release.wait()
            finished_after_close.append(owned.closed)

        await _retain_from_operation(app, isolated_cleanups, cleanup)
        if released == "within_budget":
            # The cleanup finishes while shutdown waits: the first attempt settles.
            asyncio.get_running_loop().call_later(0.2, release.set)
            assert (await app.aclose(timeout_s=5)).settled and owned.closed
            assert finished_after_close == [False]
            return
        try:
            first = await app.aclose(timeout_s=0.3)
            step = first.step("environment_cleanups")
            assert step is not None and step.status == "incomplete"
            assert first.owned_resources == "retained" and not owned.closed
        finally:
            release.set()
        assert (await app.aclose(timeout_s=5)).settled and owned.closed
        assert finished_after_close == [False]

    asyncio.run(scenario())


def test_shutdown_retries_a_failed_isolated_tool_cleanup_once_per_attempt(
    isolated_cleanups,
) -> None:
    async def scenario() -> None:
        owned = _ClosableResource()
        app = CayuApp(enable_logging=False, owned_resources=(owned,))
        retries: list[int] = []
        still_failing = True

        async def failed_cleanup() -> None:
            raise RuntimeError("process tree not yet reaped")

        async def retry() -> None:
            retries.append(len(retries) + 1)
            if still_failing:
                raise RuntimeError("process tree not yet reaped")

        await _retain_from_operation(app, isolated_cleanups, failed_cleanup, retry_factory=retry)
        # A cleanup that keeps failing leaves shutdown incomplete, resources kept.
        first = await app.aclose(timeout_s=0.5)
        assert not first.settled and first.owned_resources == "retained" and not owned.closed
        assert retries == [1]
        still_failing = False
        assert (await app.aclose(timeout_s=5)).settled and owned.closed
        assert retries == [1, 2]

    asyncio.run(scenario())


def test_an_idle_application_settles_while_another_has_an_isolated_cleanup(
    isolated_cleanups,
) -> None:
    async def scenario() -> None:
        dispatching, idle = _ClosableResource(), _ClosableResource()
        first = CayuApp(enable_logging=False, owned_resources=(dispatching,))
        second = CayuApp(enable_logging=False, owned_resources=(idle,))
        release = asyncio.Event()

        async def cleanup() -> None:
            await release.wait()

        await _retain_from_operation(first, isolated_cleanups, cleanup)
        try:
            # Only the dispatching application waits for its cleanup.
            assert (await second.aclose(timeout_s=0.3)).settled and idle.closed
            assert not (await first.aclose(timeout_s=0.3)).settled and not dispatching.closed
        finally:
            release.set()
        assert (await first.aclose(timeout_s=5)).settled and dispatching.closed

    asyncio.run(scenario())


def test_an_isolated_cleanup_cut_off_by_its_event_loop_is_retried_on_the_next(
    isolated_cleanups,
) -> None:
    owned = _ClosableResource()
    retried: list[bool] = []

    async def build_and_dispatch() -> CayuApp:
        app = CayuApp(enable_logging=False, owned_resources=(owned,))

        async def cleanup() -> None:
            await asyncio.Event().wait()  # still running when its loop ends

        async def retry() -> None:
            retried.append(True)

        await _retain_from_operation(app, isolated_cleanups, cleanup, retry_factory=retry)
        return app

    # One application used across event loops, as some hosts do.
    app = asyncio.run(build_and_dispatch())

    async def shut_down() -> bool:
        return (await app.aclose(timeout_s=5)).settled

    assert asyncio.run(shut_down()) and owned.closed
    assert retried == [True]


@pytest.mark.parametrize("retried_from", ["other_application", "no_operation"])
def test_work_a_retried_isolated_cleanup_starts_stays_with_its_application(
    isolated_cleanups, retried_from: str
) -> None:
    async def scenario() -> None:
        dispatching, other = _ClosableResource(), _ClosableResource()
        first = CayuApp(enable_logging=False, owned_resources=(dispatching,))
        second = CayuApp(enable_logging=False, owned_resources=(other,))
        release = asyncio.Event()

        async def failed_cleanup() -> None:
            raise RuntimeError("temporary directory not yet removed")

        async def slow_removal() -> None:
            await release.wait()

        async def retry() -> None:
            # Like a cleanup retry handing a slow removal to a retained task.
            isolated_cleanups._retain_task(asyncio.create_task(slow_removal()))

        await _retain_from_operation(first, isolated_cleanups, failed_cleanup, retry_factory=retry)

        async def retry_failed_cleanups() -> None:
            isolated_cleanups._retained_isolated_tool_cleanup_pending()
            await asyncio.sleep(0)

        if retried_from == "other_application":
            # Another application's dispatch fence retries it.
            await second._run_worker_step(retry_failed_cleanups)
        else:
            await retry_failed_cleanups()
        try:
            assert not isolated_cleanups.retained_isolated_tool_cleanups_unresolved(
                second._admission
            )
            assert (await second.aclose(timeout_s=0.3)).settled and other.closed
            assert not (await first.aclose(timeout_s=0.3)).settled and not dispatching.closed
        finally:
            release.set()
        assert (await first.aclose(timeout_s=5)).settled and dispatching.closed

    asyncio.run(scenario())


def test_shutdown_waits_for_an_artifact_write_its_tool_timed_out_on(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from cayu.artifacts import local as local_artifacts
    from cayu.artifacts.local import LocalArtifactStore

    owned = _ClosableResource()
    write_started, release_write = threading.Event(), threading.Event()
    writes_after_close: list[bool] = []
    original_write = local_artifacts._write_generated_artifact

    def slow_write(*args, **kwargs):
        # A write thread cannot be cancelled; it outlives the tool's timeout.
        write_started.set()
        release_write.wait(30)
        writes_after_close.append(owned.closed)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(local_artifacts, "_write_generated_artifact", slow_write)

    class ArtifactTool(Tool):
        spec = ToolSpec(
            name="artifact_tool",
            description="Write an artifact.",
            input_schema={"type": "object"},
            effect=ToolEffect.NONE,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.artifact_store is not None
            await ctx.artifact_store.put_bytes(
                b"report", filename="report.txt", session_id=ctx.session_id
            )
            return ToolResult(content="written")

    async def scenario() -> None:
        provider = _FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call", name="artifact_tool", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        app = CayuApp(
            enable_logging=False,
            owned_resources=(owned,),
            config=CayuConfig(tool_execution=ToolExecutionConfig(tool_timeout_seconds=0.2)),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[ArtifactTool()])
        try:
            await asyncio.wait_for(
                _collect(
                    app.run(
                        RunRequest(
                            session_id="artifact-write",
                            agent_name="assistant",
                            messages=[Message.text("user", "run")],
                        )
                    )
                ),
                30,
            )
            assert write_started.is_set() and writes_after_close == []
            first = await app.aclose(timeout_s=0.5)
            step = first.step("environment_cleanups")
            assert step is not None and step.status == "incomplete"
            assert first.owned_resources == "retained" and not owned.closed
        finally:
            release_write.set()
        assert (await app.aclose(timeout_s=5)).settled and owned.closed
        assert writes_after_close == [False]

    asyncio.run(scenario())


def test_shutdown_waits_for_a_receipt_sweep_its_cancelled_worker_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.core.test_dispatch import FakeProvider, _batch

    from cayu.sessions.base import InMemorySessionStore
    from cayu.tasks import InMemoryTaskStore, TaskStoreDispatcher

    async def scenario() -> None:
        owned = _ClosableResource()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        app = CayuApp(
            session_store=InMemorySessionStore(),
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
            owned_resources=(owned,),
        )
        app.register_provider(FakeProvider([_batch("done")]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        coordinator = app._queued_dispatch_coordinator
        list_receipts = coordinator.list_terminal_receipts
        sweeping, release = asyncio.Event(), asyncio.Event()
        reads_after_close: list[bool] = []

        async def slow_list(query):
            sweeping.set()
            await release.wait()
            reads_after_close.append(owned.closed)
            return await list_receipts(query)

        monkeypatch.setattr(coordinator, "list_terminal_receipts", slow_list)
        worker = asyncio.create_task(
            dispatcher.run_worker(
                app, worker_id="worker", stop=asyncio.Event(), poll_interval_s=0.05
            )
        )
        await asyncio.wait_for(sweeping.wait(), 10)
        # The worker is cancelled while its receipt sweep reads the stores.
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        try:
            first = await app.aclose(timeout_s=0.3)
            assert not first.settled and first.owned_resources == "retained"
            assert not owned.closed
        finally:
            release.set()
        assert (await app.aclose(timeout_s=5)).settled and owned.closed
        assert reads_after_close == [False]

    asyncio.run(scenario())


def test_shutdown_waits_for_a_workspace_observation_read_its_caller_abandoned() -> None:
    from cayu.workspaces.observation_recovery import await_workspace_observation_store_read

    async def scenario() -> None:
        owned, idle = _ClosableResource(), _ClosableResource()
        app = CayuApp(enable_logging=False, owned_resources=(owned,))
        other = CayuApp(enable_logging=False, owned_resources=(idle,))
        reading, release = asyncio.Event(), asyncio.Event()
        reads_after_close: list[bool] = []

        async def stubborn_read() -> None:
            # A store read that finishes even after it is cancelled.
            reading.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    assert task is not None
                    while task.cancelling():
                        task.uncancel()
            reads_after_close.append(owned.closed)

        async def recovery_step() -> None:
            await await_workspace_observation_store_read(
                stubborn_read, operation="workspace observation readback"
            )

        recovery = asyncio.create_task(app._run_worker_step(recovery_step))
        await asyncio.wait_for(reading.wait(), 5)
        # Recovery is cancelled while its read is in flight; the read goes on.
        recovery.cancel()
        await asyncio.gather(recovery, return_exceptions=True)
        try:
            # Only the application whose recovery left the read waits for it.
            assert (await other.aclose(timeout_s=0.3)).settled and idle.closed
            first = await app.aclose(timeout_s=0.3)
            step = first.step("recovery_cleanups")
            assert step is not None and step.status == "incomplete"
            assert first.owned_resources == "retained" and not owned.closed
        finally:
            release.set()
        assert (await app.aclose(timeout_s=5)).settled and owned.closed
        assert reads_after_close == [False]

    asyncio.run(scenario())


def test_shutdown_waits_for_an_unaccepted_finalization_handoff_heartbeat() -> None:
    async def scenario() -> None:
        owned = _ClosableResource()
        app = CayuApp(enable_logging=False, owned_resources=(owned,))
        control = app._session_control
        receiver_release, renewal_release = asyncio.Event(), asyncio.Event()
        heartbeat_stop = asyncio.Event()
        renewals_after_close: list[bool] = []

        async def receiver() -> None:
            # The interrupted task the claim is handed to; it exits unclaimed.
            await receiver_release.wait()

        async def heartbeat() -> None:
            await heartbeat_stop.wait()
            # A renewal write already in flight when the heartbeat is stopped.
            await renewal_release.wait()
            renewals_after_close.append(owned.closed)

        receiving = asyncio.create_task(receiver())
        control.register_active_control_task("session", receiving)
        heartbeat_task = asyncio.create_task(heartbeat())
        assert control.register_terminal_finalization_claim_handoff(
            "session",
            session_instance_id="instance",
            run_epoch=1,
            interruption_request_id="interruption",
            expected_interrupt_payload={},
            claim_id="claim",
            heartbeat_stop=heartbeat_stop,
            heartbeat_task=heartbeat_task,
        )
        receiver_release.set()
        await receiving
        control.unregister_active_control_task("session", receiving)
        await asyncio.sleep(0)
        # No task accepted the handoff: its heartbeat was stopped, still renewing.
        assert heartbeat_stop.is_set() and not heartbeat_task.done()
        try:
            first = await app.aclose(timeout_s=0.3)
            step = first.step("session_operations")
            assert step is not None and step.status == "incomplete"
            assert first.owned_resources == "retained" and not owned.closed
        finally:
            renewal_release.set()
        assert (await app.aclose(timeout_s=5)).settled and owned.closed
        assert renewals_after_close == [False]

    asyncio.run(scenario())


def test_shutdown_waits_for_a_task_lease_renewal_its_cancelled_worker_left() -> None:
    from cayu.tasks import TaskCreate
    from cayu.tasks.base import InMemoryTaskStore
    from cayu.tasks.worker import run_task_worker

    class SlowRenewalStore(InMemoryTaskStore):
        supports_interrupted_task_handoffs = True
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.closed = False
            self.handling = False
            self.renewing = asyncio.Event()
            self.release_renewal = asyncio.Event()
            self.renewals_after_close: list[bool] = []

        async def heartbeat(self, *args, **kwargs):
            if self.handling:
                # A periodic renewal while the handler runs.
                self.renewing.set()
                await self.release_renewal.wait()
                self.renewals_after_close.append(self.closed)
            return await super().heartbeat(*args, **kwargs)

        async def close(self) -> None:
            self.closed = True

    async def scenario() -> None:
        store = SlowRenewalStore()
        app = CayuApp(task_store=store, enable_logging=False, owned_resources=(store,))
        await app.create_task(TaskCreate(type="demo", title="long task"))

        handler_release = asyncio.Event()

        async def handler(_app, _task, _worker_id) -> None:
            store.handling = True
            await handler_release.wait()  # still working when its lease renews

        worker = asyncio.create_task(
            run_task_worker(
                app, store, handler, worker_id="worker", lease_seconds=2, poll_interval_s=0.05
            )
        )
        await asyncio.wait_for(store.renewing.wait(), 10)
        # The worker is cancelled while its lease renewal is being written; it
        # drains its handler, but no longer waits for that renewal.
        worker.cancel()
        handler_release.set()
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 10)
        try:
            first = await app.aclose(timeout_s=0.3)
            assert not first.settled and first.owned_resources == "retained"
            assert not store.closed
        finally:
            store.release_renewal.set()
        assert (await app.aclose(timeout_s=5)).settled and store.closed
        assert store.renewals_after_close == [False]

    asyncio.run(scenario())


def test_shutdown_waits_for_a_retained_local_execution_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.runtime import _local_execution_attempt_owner

    # The retained tasks are process-wide; keep this test's own.
    monkeypatch.setattr(_local_execution_attempt_owner, "_RETAINED_LOCAL_EXECUTION_TASKS", {})

    async def scenario() -> None:
        dispatching, idle = _ClosableResource(), _ClosableResource()
        app = CayuApp(enable_logging=False, owned_resources=(dispatching,))
        other = CayuApp(enable_logging=False, owned_resources=(idle,))
        release = asyncio.Event()
        settled_after_close: list[bool] = []

        async def settlement_write() -> None:
            # A settlement write that outlived the attempt that started it.
            await release.wait()
            settled_after_close.append(dispatching.closed)

        async def attempt() -> None:
            # As a local execution attempt run for ``app`` retains its tasks.
            _local_execution_attempt_owner._retain_local_execution_task(
                asyncio.create_task(settlement_write()), application=app._admission
            )

        await attempt()
        try:
            # Only the application that started it waits for it.
            assert (await other.aclose(timeout_s=0.3)).settled and idle.closed
            first = await app.aclose(timeout_s=0.3)
            step = first.step("session_operations")
            assert step is not None and step.status == "incomplete"
            assert first.owned_resources == "retained" and not dispatching.closed
        finally:
            release.set()
        assert (await app.aclose(timeout_s=5)).settled and dispatching.closed
        assert settled_after_close == [False]

    asyncio.run(scenario())


def test_shutdown_waits_for_a_thread_backed_secret_lookup_its_tool_timed_out_on() -> None:
    import threading

    from cayu.vaults.aws_secrets_manager import SecretsManagerVault

    owned = _ClosableResource()
    lookup_started, release_lookup = threading.Event(), threading.Event()
    lookups_after_close: list[bool] = []

    class BlockingSecretsManagerClient:
        def get_secret_value(self, **kwargs):
            # The SDK call runs in a thread: cancelling its task cannot stop it.
            lookup_started.set()
            release_lookup.wait(30)
            lookups_after_close.append(owned.closed)
            return {
                "ARN": "arn:aws:secretsmanager:us-east-1:123:secret:api",
                "Name": kwargs["SecretId"],
                "SecretString": "thread-backed-secret",
                "VersionId": "version-1",
                "VersionStages": ["AWSCURRENT"],
            }

    async def scenario() -> None:
        vault = SecretsManagerVault(
            {"api_key": "prod/cayu/api"}, client=BlockingSecretsManagerClient()
        )
        provider = _FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call", name="secret_tool", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        app = CayuApp(
            enable_logging=False,
            owned_resources=(owned,),
            config=CayuConfig(tool_execution=ToolExecutionConfig(tool_timeout_seconds=0.2)),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), vault=vault), default=True
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[_SecretTool()])
        try:
            await asyncio.wait_for(
                _collect(
                    app.run(
                        RunRequest(
                            session_id="thread-secret",
                            agent_name="assistant",
                            messages=[Message.text("user", "run")],
                        )
                    )
                ),
                30,
            )
            assert lookup_started.is_set() and lookups_after_close == []
            first = await app.aclose(timeout_s=0.5)
            step = first.step("environment_cleanups")
            assert step is not None and step.status == "incomplete"
            assert first.owned_resources == "retained" and not owned.closed
        finally:
            release_lookup.set()
        assert (await app.aclose(timeout_s=5)).settled and owned.closed
        assert lookups_after_close == [False]

    asyncio.run(scenario())


def test_the_session_operations_step_waits_for_detached_store_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.runtime import _local_execution_attempt_owner

    monkeypatch.setattr(_local_execution_attempt_owner, "_RETAINED_LOCAL_EXECUTION_TASKS", {})

    async def scenario() -> None:
        owned = _ClosableResource()
        app = CayuApp(enable_logging=False, owned_resources=(owned,))
        release = asyncio.Event()
        finished_after_close: list[bool] = []

        async def settlement_write() -> None:
            await release.wait()
            finished_after_close.append(owned.closed)

        _local_execution_attempt_owner._retain_local_execution_task(
            asyncio.create_task(settlement_write()), application=app._admission
        )
        # The write finishes while shutdown waits: the step itself must wait for
        # it, so the first attempt settles instead of reporting late work.
        asyncio.get_running_loop().call_later(0.2, release.set)
        outcome = await app.aclose(timeout_s=5)
        assert outcome.settled and owned.closed
        assert finished_after_close == [False]

    asyncio.run(scenario())


def test_an_environment_drain_that_settles_as_its_budget_runs_out_reports_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        app = CayuApp(enable_logging=False)

        async def slow_but_settled(*, timeout_s: float) -> bool:
            # Settles, but only after using the whole budget, as on a slow host.
            await asyncio.sleep(timeout_s + 0.02)
            return True

        monkeypatch.setattr(app._environment_lifecycle, "drain_retained_cleanups", slow_but_settled)
        assert await app.drain_environment_cleanups(timeout_s=0.1) is True

    asyncio.run(scenario())
