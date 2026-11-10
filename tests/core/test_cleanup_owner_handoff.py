"""A cleanup failure that carries its own successor must never lose that owner.

A runner, binding or release may fail while handing its unfinished teardown to an
authenticated cleanup task. Whoever catches that failure and records it as a
diagnostic must hand the task on, so cleanup drains keep waiting for it.
"""

import asyncio

import pytest
from tests.core.test_deferred_environments import DeferredLocalFactory, _app, _turns
from tests.runtime.runtime_environment_support import (
    ProvisionedTestRunner,
    RecordingEnvironmentFactory,
    RecordingWorkspaceBinding,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events
from tests.runtime.test_execution_admission_dispatch import _candidate, _EvidenceRunner

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ResumeRequest,
    RunRequest,
    run_to_completion,
)
from cayu.egress.runtime import _split_cleanup_cancellation, _surviving_cancellation
from cayu.environments import (
    Environment,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.environments import _cleanup_handoff as cleanup_handoff_module
from cayu.environments import deferred as deferred_module
from cayu.environments._cleanup_handoff import (
    KeptTeardowns,
    KeptTeardownStillRunning,
    keep_cleanup_owners,
)
from cayu.environments.deferred import DeferredMaterialization, _PendingClose
from cayu.environments.factory import (
    attach_environment_factory_cleanup_settlement_task,
    environment_factory_cleanup_settlement_task,
    environment_factory_cleanup_settlement_tasks,
    register_environment_factory_cleanup_retry,
    retry_environment_factory_cleanup_settlement_task,
)
from cayu.environments.shared_allocations import AllocationSharingScope
from cayu.environments.warm_spares import WarmSparePool
from cayu.runtime._environment_lifecycle import _release_unclaimed_factory_result
from cayu.runtime._shared_environment_allocations import (
    SharedAllocationKey,
    SharedEnvironmentAllocations,
    _SharedAllocation,
)
from cayu.sessions import InMemorySessionStore, SessionStatus

_RELEASE_TIMEOUT_S = 0.2


async def _stays_pending(awaitable, seconds: float = 0.3) -> None:
    """Fail when ``awaitable`` finishes within ``seconds``."""

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(awaitable, timeout=seconds)


class _Successor:
    """One teardown that keeps running after the call that started it failed."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.completed = asyncio.Event()
        self.task: asyncio.Task[None] | None = None

    def fail_with_handoff(self, message: str) -> RuntimeError:
        async def cleanup() -> None:
            self.started.set()
            await self.finish.wait()
            self.completed.set()

        self.task = asyncio.create_task(cleanup())
        error = RuntimeError(message)
        attach_environment_factory_cleanup_settlement_task(error, self.task)
        return error

    async def settle(self) -> None:
        self.finish.set()
        if self.task is not None:
            await self.task


class _ClosingBinding(RecordingWorkspaceBinding):
    def __init__(self, successor: _Successor) -> None:
        super().__init__()
        self._successor = successor

    async def close(self) -> None:
        raise self._successor.fail_with_handoff("binding teardown continues in its owned task")


@pytest.mark.parametrize("failing", ["runner", "binding", "runner-after-release-timeout"])
def test_fallback_release_keeps_the_teardown_successor(failing):
    async def scenario():
        successor = _Successor()

        class Runner(ProvisionedTestRunner):
            async def close(self):
                if failing == "runner-after-release-timeout":
                    await asyncio.sleep(_RELEASE_TIMEOUT_S * 2)
                if failing.startswith("runner"):
                    raise successor.fail_with_handoff("runner teardown continues in its owned task")
                await super().close()

        environment = Environment(EnvironmentSpec(name="wrong-name"), runner=Runner())
        if failing == "binding":
            environment = Environment(
                EnvironmentSpec(name="wrong-name"),
                runner=Runner(),
                binding=_ClosingBinding(successor),
            )
        # The rejected result has no release callback, so DISCARD falls back to
        # closing its runner and binding directly.
        app = CayuApp(enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            RecordingEnvironmentFactory(environment, release_timeout_s=_RELEASE_TIMEOUT_S),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        try:
            events = await collect_events(
                app,
                RunRequest(
                    session_id=f"fallback-{failing}",
                    agent_name="assistant",
                    messages=[Message.text("user", "run")],
                ),
            )
            assert events[-1].type is EventType.SESSION_FAILED
            await asyncio.wait_for(successor.started.wait(), timeout=2)
            assert not successor.completed.is_set()
            assert not await app.drain_environment_cleanups(timeout_s=0.5), (
                "The drain reported quiescence while the fallback teardown still ran"
            )
        finally:
            await successor.settle()
            assert await app.drain_environment_cleanups(timeout_s=2)

    asyncio.run(scenario())


def test_retried_fallback_release_waits_for_the_teardown_successor():
    async def scenario():
        successor = _Successor()

        class Runner(ProvisionedTestRunner):
            def __init__(self):
                super().__init__()
                self.closes = 0

            async def close(self):
                self.closes += 1
                if self.closes == 1:
                    raise successor.fail_with_handoff("runner teardown continues in its owned task")
                await super().close()

        async def failed_prerequisite():
            raise RuntimeError("probe settlement failed")

        async def settled_prerequisite():
            return None

        prerequisite = asyncio.create_task(failed_prerequisite())
        register_environment_factory_cleanup_retry(
            prerequisite, lambda: asyncio.create_task(settled_prerequisite())
        )
        original = RuntimeError("setup failed")
        attach_environment_factory_cleanup_settlement_task(original, prerequisite)
        runner = Runner()
        result = EnvironmentFactoryResult(
            environment=Environment(EnvironmentSpec(name="env"), runner=runner),
            release_timeout_s=_RELEASE_TIMEOUT_S,
        )
        try:
            # The failed prerequisite stops the fallback before it closes anything.
            payload = await _release_unclaimed_factory_result(
                result, action=EnvironmentFactoryReleaseAction.DISCARD, original_error=original
            )
            assert payload["completed"] is False
            owner = environment_factory_cleanup_settlement_task(original)
            assert owner is not None and owner.done() and runner.closes == 0
            retried = retry_environment_factory_cleanup_settlement_task(owner)
            assert retried is not owner
            await asyncio.wait_for(successor.started.wait(), timeout=2)
            await asyncio.sleep(0)
            assert runner.closes == 1
            assert not retried.done(), "The retried release settled before its teardown did"
            await successor.settle()
            await asyncio.wait_for(retried, timeout=2)
        finally:
            await successor.settle()

    asyncio.run(scenario())


@pytest.mark.parametrize("teardown", ["within-bound", "past-bound"])
def test_deferred_admission_close_keeps_the_teardown_successor(tmp_path, monkeypatch, teardown):
    if teardown == "past-bound":
        monkeypatch.setattr(deferred_module, "KEPT_TEARDOWN_TIMEOUT_S", 0.2)

    async def scenario():
        successor = _Successor()

        class RefusedRunner(_EvidenceRunner):
            def __init__(self):
                super().__init__("deferred-close")
                self.closes = 0
                self.close_overtook_successor = False

            async def collect_execution_admission_candidate(self):
                raise ConnectionError("Admission probe refused")

            async def close(self):
                self.closes += 1
                if self.closes == 1:
                    raise successor.fail_with_handoff("runner teardown continues in its owned task")
                self.close_overtook_successor |= not successor.completed.is_set()
                await super().close()

        runner = RefusedRunner()
        retry_started = asyncio.Event()

        class TracedMaterialization(DeferredMaterialization):
            close_attempts = 0

            async def _close_pending(self, **kwargs):
                self.close_attempts += 1
                if self.close_attempts == 2:
                    retry_started.set()
                await super()._close_pending(**kwargs)

        class Factory(DeferredLocalFactory):
            def execution_admission_candidate(self, request):
                return _candidate("deferred-close", state="available")

            async def create(self, request):
                async def materialize(mode):
                    return runner, None

                state = TracedMaterialization(
                    materialize,
                    default_cwd=str(tmp_path),
                    configured_candidate=lambda requirements: self.execution_admission_candidate(
                        request
                    ),
                )
                return EnvironmentFactoryResult(
                    environment=Environment(
                        EnvironmentSpec(name=request.environment_name),
                        runner=state.runner,
                        binding=state.binding,
                    ),
                    reconnect_metadata={"kind": "deferred-close"},
                )

        app = _app(Factory(tmp_path, tmp_path), InMemorySessionStore(), _turns("write"))

        async def collect():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="deferred-close-successor",
                        messages=[Message.text("user", "Use the runner once")],
                    )
                )
            ]

        run = asyncio.create_task(collect())
        try:
            # The retired runner keeps the successor: its next close (at
            # finalization) waits for it before closing the runner again.
            await asyncio.wait_for(retry_started.wait(), timeout=10)
            assert not successor.completed.is_set()
            assert runner.closes == 1
            if teardown == "past-bound":
                # Finalization gives up at the bound and reports a binding
                # finalization failure; the runner stays pending with its teardown
                # owned instead of being closed beside it.
                events = await asyncio.wait_for(run, timeout=10)
                assert [
                    event.payload["error_type"]
                    for event in events
                    if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
                ] == ["KeptTeardownStillRunning"]
                assert runner.closes == 1
                await successor.settle()
            else:
                await _stays_pending(asyncio.shield(run))
                await successor.settle()
                events = await asyncio.wait_for(run, timeout=10)
                assert runner.closes == 2
            assert any(e.type is EventType.ENVIRONMENT_MATERIALIZATION_FAILED for e in events)
            assert not runner.close_overtook_successor
        finally:
            await successor.settle()
            if not run.done():
                run.cancel()
            await asyncio.gather(run, return_exceptions=True)
            await app.close_idle_environment_resources()

    asyncio.run(scenario())


def test_shared_allocation_close_waits_for_every_teardown_successor():
    async def scenario():
        finalize_successor = _Successor()
        close_successor = _Successor()

        class Binding(RecordingWorkspaceBinding):
            def __init__(self):
                super().__init__()
                self.finalizes = 0
                self.overtook_successor = False

            def abandon(self, bound):
                return False

            async def finalize(self, bound, *, outcome=None, metadata=None):
                self.finalizes += 1
                if self.finalizes == 1:
                    raise finalize_successor.fail_with_handoff("binding teardown continues")
                self.overtook_successor |= not finalize_successor.completed.is_set()
                return None

        class Runner(ProvisionedTestRunner):
            def __init__(self):
                super().__init__()
                self.closes = 0
                self.overtook_successor = False

            async def close(self):
                self.closes += 1
                if self.closes == 1:
                    raise close_successor.fail_with_handoff("runner teardown continues")
                self.overtook_successor |= not close_successor.completed.is_set()
                await super().close()

        runner = Runner()
        binding = Binding()
        environment = Environment(EnvironmentSpec(name="shared"), runner=runner, binding=binding)
        key = SharedAllocationKey(
            scope=AllocationSharingScope.SESSION,
            scope_id="scope",
            owner_session_id="owner",
            environment_name="shared",
        )
        allocations = SharedEnvironmentAllocations(session_store=InMemorySessionStore())
        allocations._live[key] = _SharedAllocation(
            key=key,
            factory=RecordingEnvironmentFactory(environment),
            result=EnvironmentFactoryResult(environment=environment),
            bound=object(),  # type: ignore[arg-type]
        )
        try:
            # The release's finalize fails and hands off; the discard it falls
            # back to must wait for that teardown before finalizing again.
            first = asyncio.create_task(allocations.close())
            await asyncio.wait_for(finalize_successor.started.wait(), timeout=2)
            await _stays_pending(asyncio.shield(first))
            assert binding.finalizes == 1
            await finalize_successor.settle()
            # The discard's runner close fails and hands off too; the next
            # close retries the discard only after that teardown settles.
            assert not await asyncio.wait_for(first, timeout=2)
            assert binding.finalizes == 2
            await asyncio.wait_for(close_successor.started.wait(), timeout=2)
            second = asyncio.create_task(allocations.close())
            await _stays_pending(asyncio.shield(second))
            assert runner.closes == 1
            await close_successor.settle()
            assert await asyncio.wait_for(second, timeout=2)
            assert runner.closes == 2
            assert not binding.overtook_successor
            assert not runner.overtook_successor
        finally:
            await finalize_successor.settle()
            await close_successor.settle()

    asyncio.run(scenario())


def test_keeping_owners_into_a_group_that_holds_them_adds_nothing():
    async def scenario():
        first, second = _Successor(), _Successor()
        source = ExceptionGroup(
            "two teardowns continue",
            [first.fail_with_handoff("first"), second.fail_with_handoff("second")],
        )
        group = BaseExceptionGroup("release failed", [source])
        try:
            keep_cleanup_owners(source, into=group)
            assert environment_factory_cleanup_settlement_task(group) is None
            assert set(environment_factory_cleanup_settlement_tasks(group)) == {
                first.task,
                second.task,
            }
            kept: list[asyncio.Task[None]] = []
            keep_cleanup_owners(source, into=kept)
            keep_cleanup_owners(source, into=kept)
            assert kept == [first.task, second.task]
        finally:
            await first.settle()
            await second.settle()

    asyncio.run(scenario())


def test_warm_spare_pool_close_waits_for_a_failed_creation_teardown_even_after_cancel():
    async def scenario():
        successor = _Successor()

        class Backend:
            async def create_spare(self, name):
                raise successor.fail_with_handoff("spare teardown continues in its owned task")

            async def spare_is_alive(self, spare):
                return True

            async def claim_spare(self, spare, name, requirements):
                return None

            async def discard_spare(self, spare):
                return None

            async def list_spare_names(self, prefix):
                return []

            async def remove_spare_named(self, name):
                return None

        pool = WarmSparePool(Backend(), size=1, name_prefix="cleanup-owner")
        try:
            pool.schedule_refill()
            await pool.wait_for_refill()
            assert successor.task is not None and not successor.task.done()
            closing = asyncio.create_task(pool.close())
            await _stays_pending(asyncio.shield(closing))
            # A cancelled close keeps the teardown; the next close waits again.
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            reclosing = asyncio.create_task(pool.close())
            await _stays_pending(asyncio.shield(reclosing))
            await successor.settle()
            await asyncio.wait_for(reclosing, timeout=2)
        finally:
            await successor.settle()
            await pool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("leaves", ["failure", "cancellation"])
def test_egress_cleanup_split_keeps_owners_attached_to_the_group(leaves):
    async def scenario():
        successor = _Successor()
        leaf = (
            RuntimeError("cleanup failed")
            if leaves == "failure"
            else asyncio.CancelledError("cleanup cancelled")
        )
        successor.fail_with_handoff("teardown continues in its owned task")
        assert successor.task is not None
        group = BaseExceptionGroup("egress cleanup failed", [leaf])
        attach_environment_factory_cleanup_settlement_task(group, successor.task)
        try:
            cancellation, failure = _split_cleanup_cancellation(group)
            kept = failure if leaves == "failure" else cancellation
            assert kept is not None
            assert environment_factory_cleanup_settlement_tasks(kept) == (successor.task,)
        finally:
            await successor.settle()

    asyncio.run(scenario())


@pytest.mark.parametrize("teardown", ["within-bound", "past-bound"])
def test_deferred_crash_cleanup_keeps_its_teardown_for_the_next_materialization(
    tmp_path, monkeypatch, teardown
):
    if teardown == "past-bound":
        monkeypatch.setattr(cleanup_handoff_module, "KEPT_TEARDOWN_TIMEOUT_S", 0.2)

    async def scenario():
        successor = _Successor()
        create_called = asyncio.Event()

        class Factory(DeferredLocalFactory):
            def __init__(self, source, target):
                super().__init__(source, target)
                self.create_overtook_successor = False

            async def create(self, request):
                if successor.task is not None:
                    self.create_overtook_successor |= not successor.completed.is_set()
                    create_called.set()
                return await super().create(request)

            async def release_deferred_materialization(self, request):
                raise successor.fail_with_handoff("leftover teardown continues in its owned task")

        (tmp_path / "source").mkdir()
        (tmp_path / "target").mkdir()
        factory = Factory(tmp_path / "source", tmp_path / "target")
        store = InMemorySessionStore()
        app = _app(factory, store, [*_turns(), *_turns(), *_turns()])

        def resume():
            return run_to_completion(
                app,
                ResumeRequest(
                    session_id="crash-leftover", messages=[Message.text("user", "again")]
                ),
            )

        try:
            await run_to_completion(
                app,
                RunRequest(
                    agent_name="agent",
                    session_id="crash-leftover",
                    messages=[Message.text("user", "go")],
                ),
            )
            session = await store.load("crash-leftover")
            assert session is not None
            # Recovery's best-effort crash cleanup fails while its teardown continues.
            assert not await app._environment_lifecycle.release_deferred_materialization(
                session=session,
                registered_agent=app._get_registered_agent(session.agent_name),
                registered_environment=app._get_registered_environment_for_session(
                    session.environment_name
                ),
            )
            await asyncio.wait_for(successor.started.wait(), timeout=2)
            if teardown == "past-bound":
                # Past the bound the resume fails as an environment setup failure
                # without touching the leftover; a later resume succeeds.
                failed = await asyncio.wait_for(resume(), timeout=10)
                assert failed.status is SessionStatus.FAILED
                assert "teardown is still running; retry later" in (failed.error or "")
                assert not create_called.is_set()
                await successor.settle()
                retried = await asyncio.wait_for(resume(), timeout=10)
                assert retried.status is SessionStatus.COMPLETED
            else:
                run = asyncio.create_task(resume())
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(create_called.wait()), timeout=1)
                assert not run.done()
                await successor.settle()
                await asyncio.wait_for(run, timeout=10)
            assert create_called.is_set()
            assert not factory.create_overtook_successor
        finally:
            await successor.settle()
            await app.close_idle_environment_resources()

    asyncio.run(scenario())


def test_fallback_release_waits_for_every_successor_and_keeps_a_failed_ones_retry():
    async def scenario():
        slow = _Successor()
        retried = asyncio.Event()

        async def failed_teardown():
            raise RuntimeError("runner teardown failed")

        async def retried_teardown():
            retried.set()

        class Runner(ProvisionedTestRunner):
            async def close(self):
                failed = asyncio.create_task(failed_teardown())
                register_environment_factory_cleanup_retry(
                    failed, lambda: asyncio.create_task(retried_teardown())
                )
                error = RuntimeError("runner teardown continues in its owned task")
                attach_environment_factory_cleanup_settlement_task(error, failed)
                raise error

        environment = Environment(
            EnvironmentSpec(name="wrong-name"), runner=Runner(), binding=_ClosingBinding(slow)
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            RecordingEnvironmentFactory(environment, release_timeout_s=5),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        run = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    session_id="two-successors",
                    agent_name="assistant",
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        try:
            # The first successor failed; the release still waits for the second.
            await asyncio.wait_for(slow.started.wait(), timeout=2)
            await _stays_pending(asyncio.shield(run))
            await slow.settle()
            events = await asyncio.wait_for(run, timeout=5)
            assert events[-1].type is EventType.SESSION_FAILED
            # The failed successor's own retry stays reachable from the lifecycle.
            assert await app.drain_environment_cleanups(timeout_s=5)
            assert retried.is_set()
        finally:
            await slow.settle()
            await asyncio.gather(run, return_exceptions=True)

    asyncio.run(scenario())


def test_retiring_a_runner_under_the_lock_leaves_earlier_runners_alone(tmp_path):
    async def scenario():
        retries = []

        async def failed():
            raise RuntimeError("earlier teardown failed")

        earlier_settlement = asyncio.create_task(failed())
        register_environment_factory_cleanup_retry(
            earlier_settlement, lambda: retries.append(1) or asyncio.create_task(asyncio.sleep(10))
        )
        await asyncio.gather(earlier_settlement, return_exceptions=True)
        earlier, retiring = ProvisionedTestRunner(), ProvisionedTestRunner()

        async def materialize(mode):
            return retiring, None

        state = DeferredMaterialization(materialize, default_cwd=str(tmp_path))
        state._pending_close.extend(
            [_PendingClose(earlier, earlier_settlement), _PendingClose(retiring)]
        )
        await asyncio.wait_for(state._close_pending(retiring=retiring), timeout=2)
        assert retiring.is_closed
        assert not earlier.is_closed
        assert retries == []
        assert [pending.runner for pending in state._pending_close] == [earlier]

    asyncio.run(scenario())


def test_kept_teardowns_keep_what_has_not_settled(monkeypatch):
    monkeypatch.setattr(cleanup_handoff_module, "KEPT_TEARDOWN_TIMEOUT_S", 0.1)

    async def scenario():
        successor = _Successor()
        kept = KeptTeardowns()
        keep_cleanup_owners(RuntimeError("no owner"), into=kept.bucket("empty"))
        keep_cleanup_owners(successor.fail_with_handoff("teardown"), into=kept.bucket("key"))
        assert "empty" not in kept._tasks
        try:
            with pytest.raises(KeptTeardownStillRunning):
                await kept.await_settled("key")
            waiting = asyncio.create_task(kept.await_settled("key"))
            await asyncio.sleep(0.01)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert kept._tasks["key"] == [successor.task]
            await successor.settle()
            await kept.await_settled("key")
            assert kept._tasks == {}
        finally:
            await successor.settle()

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_kept_teardown_inventory_harvests_only_settled_owners(outcome):
    async def scenario():
        successor = _Successor()
        kept = KeptTeardowns()
        keep_cleanup_owners(successor.fail_with_handoff("teardown"), into=kept.bucket("pending"))

        async def finish():
            if outcome == "failure":
                raise RuntimeError("teardown failed")

        settled = asyncio.create_task(finish())
        if outcome == "cancelled":
            settled.cancel()
        kept.bucket("settled").append(settled)
        try:
            await asyncio.wait((settled,))
            assert kept.settling() == (successor.task,)
            assert kept._tasks == {"pending": [successor.task]}
            await successor.settle()
            assert kept.settling() == ()
            assert kept._tasks == {}
        finally:
            await successor.settle()

    asyncio.run(scenario())


def test_a_later_egress_cleanup_cancellation_hands_its_owners_to_the_first():
    async def scenario():
        successor = _Successor()
        later = asyncio.CancelledError("later phase")
        successor.fail_with_handoff("teardown continues in its owned task")
        assert successor.task is not None
        attach_environment_factory_cleanup_settlement_task(later, successor.task)
        first = asyncio.CancelledError("first phase")
        try:
            assert _surviving_cancellation(None, later) is later
            assert _surviving_cancellation(first, later) is first
            assert environment_factory_cleanup_settlement_tasks(first) == (successor.task,)
        finally:
            await successor.settle()

    asyncio.run(scenario())


def test_a_dead_earlier_settlement_never_blocks_a_retired_runners_later_close(tmp_path):
    async def scenario():
        async def failed_probe():
            raise RuntimeError("probe failed with no retry")

        async def settled_teardown():
            return None

        probe = asyncio.create_task(failed_probe())
        await asyncio.gather(probe, return_exceptions=True)

        class Runner(ProvisionedTestRunner):
            def __init__(self):
                super().__init__()
                self.closes = 0

            async def close(self):
                self.closes += 1
                if self.closes == 1:
                    successor = asyncio.create_task(settled_teardown())
                    register_environment_factory_cleanup_retry(
                        successor, lambda: asyncio.create_task(settled_teardown())
                    )
                    error = RuntimeError("runner teardown continues in its owned task")
                    attach_environment_factory_cleanup_settlement_task(error, successor)
                    raise error
                await super().close()

        runner = Runner()

        async def materialize(mode):
            return runner, None

        state = DeferredMaterialization(materialize, default_cwd=str(tmp_path))
        state._pending_close.append(_PendingClose(runner, probe))
        with pytest.raises(RuntimeError, match="teardown continues"):
            await state._close_pending()
        await asyncio.wait_for(state._close_pending(), timeout=2)
        assert runner.closes == 2
        assert runner.is_closed
        assert state._pending_close == []

    asyncio.run(scenario())
