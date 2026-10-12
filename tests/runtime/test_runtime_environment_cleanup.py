"""Environment cleanup contracts."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Literal

import pytest
from tests.runtime.runtime_environment_support import (
    ProvisionedTestRunner,
    RecordingEnvironmentFactory,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import _SETTLED_DRAIN_TIMEOUT_S, collect_events

import cayu.runtime._environment_lifecycle as environment_lifecycle_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.factory import (
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.workspaces.local import LocalWorkspace


def test_environment_factory_result_release_is_bounded(tmp_path):
    async def run():
        store = InMemorySessionStore()
        release_started = asyncio.Event()
        finish_release = asyncio.Event()
        release_completed = asyncio.Event()

        async def release(_action: EnvironmentFactoryReleaseAction) -> None:
            release_started.set()
            await finish_release.wait()
            release_completed.set()

        workspace_root = tmp_path / "factory-release-timeout"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="different"),
                workspace=LocalWorkspace(workspace_root),
            ),
            release=release,
            release_timeout_s=0.01,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        captured_cleanup_profiles: list[Any] = []
        original_resolve_factory = app._environment_lifecycle.resolve_factory

        async def capture_resolve_factory(**kwargs):
            captured_cleanup_profiles.append(kwargs["execution_profile"])
            return await original_resolve_factory(**kwargs)

        app._environment_lifecycle.resolve_factory = capture_resolve_factory

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_release_timeout",
                messages=[Message.text("user", "run")],
            ),
        )
        assert release_started.is_set()
        assert not release_completed.is_set()
        assert "sess_factory_release_timeout" in (
            app._environment_lifecycle._deferred_factory_cleanup_tasks
        )
        assert "sess_factory_release_timeout" in (
            app._environment_lifecycle._pending_environment_owner_admissions
        )
        assert (
            app._environment_lifecycle._deferred_factory_cleanup_profiles[
                "sess_factory_release_timeout"
            ]
            is captured_cleanup_profiles[0]
        )
        finish_release.set()
        assert await app.drain_environment_cleanups(timeout_s=_SETTLED_DRAIN_TIMEOUT_S) is True
        assert release_completed.is_set()
        assert app._environment_lifecycle._deferred_factory_cleanup_profiles == {}
        return events

    events = asyncio.run(run())

    expected_release = {
        "action": "discard",
        "callback_provided": True,
        "completed": False,
        "error": "Environment factory result release did not complete within 0.01 seconds.",
        "error_type": "TimeoutError",
        "timeout_s": 0.01,
    }
    factory_failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_FACTORY_FAILED
    )
    session_failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert factory_failed.payload["environment_factory_release"] == expected_release
    assert session_failed.payload["environment_factory_release"] == expected_release


def test_cancellation_waits_for_factory_result_release_then_propagates(tmp_path):
    async def run() -> None:
        store = InMemorySessionStore()
        release_started = asyncio.Event()
        finish_release = asyncio.Event()
        release_completed = asyncio.Event()

        async def release(_action: EnvironmentFactoryReleaseAction) -> None:
            release_started.set()
            await finish_release.wait()
            release_completed.set()

        workspace_root = tmp_path / "factory-release-cancel"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="different"),
                workspace=LocalWorkspace(workspace_root),
            ),
            release=release,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_factory_release_cancel",
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await asyncio.wait_for(release_started.wait(), timeout=10)
        run_task.cancel()
        await asyncio.sleep(0)
        assert run_task.done() is False
        finish_release.set()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        assert release_completed.is_set()

    asyncio.run(run())


def test_environment_factory_fallback_release_is_bounded(tmp_path):
    class HangingRunner(ProvisionedTestRunner):
        def __init__(self) -> None:
            self.close_started = asyncio.Event()
            self.finish_close = asyncio.Event()
            self.close_completed = asyncio.Event()

        async def close(self) -> None:
            self.close_started.set()
            await self.finish_close.wait()
            self.close_completed.set()

    async def run():
        store = InMemorySessionStore()
        runner = HangingRunner()
        workspace_root = tmp_path / "factory-fallback-timeout"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="different"),
                workspace=LocalWorkspace(workspace_root),
                runner=runner,
            ),
            release_timeout_s=0.01,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_fallback_timeout",
                messages=[Message.text("user", "run")],
            ),
        )
        assert runner.close_started.is_set()
        assert not runner.close_completed.is_set()
        assert "sess_factory_fallback_timeout" in (
            app._environment_lifecycle._deferred_factory_cleanup_tasks
        )
        assert "sess_factory_fallback_timeout" in (
            app._environment_lifecycle._pending_environment_owner_admissions
        )
        runner.finish_close.set()
        assert await app.drain_environment_cleanups(timeout_s=_SETTLED_DRAIN_TIMEOUT_S) is True
        assert runner.close_completed.is_set()
        return events

    events = asyncio.run(run())

    factory_failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_FACTORY_FAILED
    )
    assert factory_failed.payload["environment_factory_release"] == {
        "action": "discard",
        "callback_provided": False,
        "completed": False,
        "error": "Environment factory result release did not complete within 0.01 seconds.",
        "error_type": "TimeoutError",
        "timeout_s": 0.01,
    }


@pytest.mark.parametrize(
    "close_fails,cancellation_target", [(False, "caller"), (True, "caller"), (False, "child")]
)
@pytest.mark.parametrize("release_mode", ["fallback", "callback"])
@pytest.mark.parametrize("historical_cancellation", [False, True])
def test_cancellation_waits_for_factory_fallback_release_then_propagates(
    tmp_path, close_fails, cancellation_target, release_mode, historical_cancellation
):
    close_error = RuntimeError("fallback close failed after cancellation")

    class BlockingRunner(ProvisionedTestRunner):
        def __init__(self) -> None:
            self.close_started = asyncio.Event()
            self.finish_close = asyncio.Event()
            self.close_completed = asyncio.Event()
            self.close_calls = 0
            self.close_task = None

        async def close(self) -> None:
            self.close_calls += 1
            self.close_task = asyncio.current_task()
            self.close_started.set()
            await self.finish_close.wait()
            self.close_completed.set()
            if close_fails:
                raise close_error

    async def run() -> None:
        store = InMemorySessionStore()
        runner = BlockingRunner()
        workspace_root = tmp_path / "factory-fallback-cancel"
        workspace_root.mkdir()

        async def release(_action):
            await runner.close()

        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="different"),
                workspace=LocalWorkspace(workspace_root),
                runner=runner,
            ),
            release=release if release_mode == "callback" else None,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def consume():
            if historical_cancellation:
                current = asyncio.current_task()
                current.cancel("previously handled cancellation")
                with pytest.raises(asyncio.CancelledError, match="previously handled cancellation"):
                    await asyncio.sleep(0)
                assert current.cancelling() == 1
            return await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_factory_fallback_cancel",
                    messages=[Message.text("user", "run")],
                ),
            )

        run_task = asyncio.create_task(consume())
        await asyncio.wait_for(runner.close_started.wait(), timeout=10)
        target = run_task if cancellation_target == "caller" else runner.close_task
        assert target is not None
        target.cancel("cancel during factory fallback release")
        expected_count = int(historical_cancellation) + int(cancellation_target == "caller")
        assert target.cancelling() == (expected_count if cancellation_target == "caller" else 1)
        assert run_task.cancelling() == expected_count
        await asyncio.sleep(0)
        assert run_task.done() is False
        runner.finish_close.set()
        if cancellation_target == "child":
            events = await run_task
            assert not run_task.cancelled() and run_task.cancelling() == expected_count
            assert events[-1].type is EventType.SESSION_FAILED
            assert runner.close_task.cancelled() and runner.close_task.cancelling() == 1
            assert runner.close_calls == 1 and not runner.close_completed.is_set()
            assert "sess_factory_fallback_cancel" in (
                app._environment_lifecycle._deferred_factory_cleanup_tasks
            )
            return
        with pytest.raises(asyncio.CancelledError) as raised:
            await run_task
        assert raised.value.args == ("cancel during factory fallback release",)
        if close_fails:
            from cayu._exception_groups import iter_exception_tree

            assert raised.value.__cause__ is not None
            assert (
                sum(item is close_error for item in iter_exception_tree(raised.value.__cause__))
                == 1
            )
        assert run_task.cancelled() is True
        assert run_task.cancelling() == expected_count
        assert runner.close_calls == 1
        assert runner.close_completed.is_set()

    asyncio.run(run())


def test_cancelled_factory_checkpoint_reports_fallback_cleanup_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FailingCloseRunner(ProvisionedTestRunner):
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("runner close failed")

    async def run() -> tuple[asyncio.CancelledError, FailingCloseRunner]:
        store = InMemorySessionStore()
        runner = FailingCloseRunner()
        workspace_root = tmp_path / "factory-checkpoint-cancel"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root),
                runner=runner,
            ),
            reconnect_metadata={"sandbox_id": "sbx_uncommitted"},
        )
        app = CayuApp(session_store=store, enable_logging=False)
        checkpoint_started = asyncio.Event()

        async def block_checkpoint(
            _lifecycle: environment_lifecycle_module.EnvironmentLifecycle,
            **_kwargs: Any,
        ) -> None:
            checkpoint_started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(
            environment_lifecycle_module.EnvironmentLifecycle,
            "_checkpoint_factory_reconnect_metadata",
            block_checkpoint,
        )
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_factory_checkpoint_cancel",
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await asyncio.wait_for(checkpoint_started.wait(), timeout=10)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await run_task
        return exc_info.value, runner

    cancellation, runner = asyncio.run(run())

    assert runner.close_calls == 1
    assert any(
        note
        == (
            "Environment factory fallback release incomplete after discard: "
            "runner: RuntimeError: runner close failed."
        )
        for note in cancellation.__notes__
    )


def test_cancelled_factory_checkpoint_read_discards_uncommitted_allocation(tmp_path):
    class BlockingCheckpointReadStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.block_next_checkpoint_read = False
            self.checkpoint_read_started = asyncio.Event()

        async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
            if self.block_next_checkpoint_read:
                self.block_next_checkpoint_read = False
                self.checkpoint_read_started.set()
                await asyncio.Event().wait()
            return await super().load_checkpoint(session_id)

    async def run() -> tuple[list[EnvironmentFactoryReleaseAction], dict[str, Any] | None]:
        store = BlockingCheckpointReadStore()
        release_actions: list[EnvironmentFactoryReleaseAction] = []

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            release_actions.append(action)

        class BlockingReadFactory(RecordingEnvironmentFactory):
            async def create(
                self,
                request: EnvironmentFactoryRequest,
            ) -> EnvironmentFactoryResult:
                result = await super().create(request)
                store.block_next_checkpoint_read = True
                return result

        workspace_root = tmp_path / "checkpoint-read-workspace"
        workspace_root.mkdir()
        factory = BlockingReadFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root),
            ),
            reconnect_metadata={"sandbox_id": "sandbox_never_checkpointed"},
            release=release,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_cancelled_factory_checkpoint_read",
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await asyncio.wait_for(store.checkpoint_read_started.wait(), timeout=10)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        checkpoint = await store.load_checkpoint("sess_cancelled_factory_checkpoint_read")
        return release_actions, checkpoint

    release_actions, checkpoint = asyncio.run(run())

    assert release_actions == [EnvironmentFactoryReleaseAction.DISCARD]
    assert "environment_factory_reconnect" not in (checkpoint or {})


@pytest.mark.parametrize(
    ("failure_mode", "expected_action", "expected_checkpointed"),
    [
        pytest.param(
            "before_commit",
            EnvironmentFactoryReleaseAction.DISCARD,
            False,
            id="proven-uncommitted",
        ),
        pytest.param(
            "after_commit",
            EnvironmentFactoryReleaseAction.PRESERVE,
            True,
            id="lost-commit-acknowledgement",
        ),
        pytest.param(
            "after_commit_unreadable",
            EnvironmentFactoryReleaseAction.PRESERVE,
            True,
            id="unreconciled-commit",
        ),
    ],
)
def test_factory_checkpoint_write_failure_reconciles_allocation_ownership(
    tmp_path,
    failure_mode: Literal[
        "before_commit",
        "after_commit",
        "after_commit_unreadable",
    ],
    expected_action: EnvironmentFactoryReleaseAction,
    expected_checkpointed: bool,
):
    class HostileCheckpointError(ConnectionError):
        @property
        def _cayu_environment_factory_checkpoint_may_be_committed(self) -> object:
            raise RuntimeError("workload-secret-from-checkpoint-access")

        @_cayu_environment_factory_checkpoint_may_be_committed.setter
        def _cayu_environment_factory_checkpoint_may_be_committed(
            self,
            _value: object,
        ) -> None:
            raise RuntimeError("workload-secret-from-checkpoint-mutation")

        @property
        def _cayu_environment_factory_release(self) -> object:
            raise RuntimeError("workload-secret-from-release-access")

        @_cayu_environment_factory_release.setter
        def _cayu_environment_factory_release(self, _value: object) -> None:
            raise RuntimeError("workload-secret-from-release-mutation")

    class FailingFactoryCheckpointStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.checkpoint_failure_armed = False
            self.reconciliation_read_failure_armed = False

        async def transform_checkpoint(self, session_id, checkpoint_transform) -> None:
            if not self.checkpoint_failure_armed:
                await super().transform_checkpoint(session_id, checkpoint_transform)
                return
            self.checkpoint_failure_armed = False
            if failure_mode == "before_commit":
                raise ConnectionError("checkpoint write rejected before commit")
            await super().transform_checkpoint(session_id, checkpoint_transform)
            if failure_mode == "after_commit_unreadable":
                self.reconciliation_read_failure_armed = True
                raise HostileCheckpointError("checkpoint commit acknowledgement lost")
            raise ConnectionError("checkpoint commit acknowledgement lost")

        async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
            if self.reconciliation_read_failure_armed:
                self.reconciliation_read_failure_armed = False
                raise ConnectionError("checkpoint reconciliation unavailable")
            return await super().load_checkpoint(session_id)

    async def run() -> tuple[
        list[Event],
        Session | None,
        list[EnvironmentFactoryReleaseAction],
        dict[str, Any] | None,
    ]:
        store = FailingFactoryCheckpointStore()
        release_actions: list[EnvironmentFactoryReleaseAction] = []

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            release_actions.append(action)

        class ArmingFactory(RecordingEnvironmentFactory):
            async def create(
                self,
                request: EnvironmentFactoryRequest,
            ) -> EnvironmentFactoryResult:
                result = await super().create(request)
                store.checkpoint_failure_armed = True
                return result

        session_id = f"sess_factory_checkpoint_{failure_mode}"
        workspace_root = tmp_path / failure_mode
        workspace_root.mkdir()
        factory = ArmingFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root),
            ),
            reconnect_metadata={"sandbox_id": f"sandbox_{failure_mode}"},
            release=release,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        return events, session, release_actions, checkpoint

    events, session, release_actions, checkpoint = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.SESSION_FAILED,
    ]
    assert release_actions == [expected_action]
    factory_failure = events[1]
    assert factory_failure.payload["environment_factory_release"]["action"] == str(expected_action)
    assert "workload-secret" not in json.dumps([event.model_dump(mode="json") for event in events])
    if expected_checkpointed:
        assert checkpoint is not None
        assert checkpoint["environment_factory_reconnect"] == {
            "dynamic": {"sandbox_id": f"sandbox_{failure_mode}"}
        }
        assert checkpoint["environment_factory_allocation_owner"] == {
            "dynamic": f"sess_factory_checkpoint_{failure_mode}"
        }
    else:
        assert "environment_factory_reconnect" not in (checkpoint or {})
        assert "environment_factory_allocation_owner" not in (checkpoint or {})


def test_cancelled_sqlite_factory_checkpoint_preserves_committed_allocation(
    tmp_path,
):
    async def run() -> tuple[list[EnvironmentFactoryReleaseAction], dict[str, Any]]:
        transform_started = threading.Event()
        allow_transform_return = threading.Event()
        invocation_release_started = asyncio.Event()
        invocation_release_completed = asyncio.Event()

        class DelayedSQLiteSessionStore(SQLiteSessionStore):
            invocation_lifecycle_command_version = 1

            async def transform_checkpoint(self, session_id, checkpoint_transform) -> None:
                def delayed_transform(session, checkpoint):
                    transformed = checkpoint_transform(session, checkpoint)
                    transform_started.set()
                    if not allow_transform_return.wait(timeout=5):
                        raise TimeoutError("test did not release the checkpoint transform")
                    return transformed

                await super().transform_checkpoint(session_id, delayed_transform)

            async def release_session_invocation(self, command):
                invocation_release_started.set()
                result = await super().release_session_invocation(command)
                invocation_release_completed.set()
                return result

        store = DelayedSQLiteSessionStore(tmp_path / "sessions.sqlite")
        release_actions: list[EnvironmentFactoryReleaseAction] = []

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            release_actions.append(action)

        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root),
            ),
            reconnect_metadata={"sandbox_id": "sandbox_committed_after_cancel"},
            release=release,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_cancelled_sqlite_factory_checkpoint",
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await asyncio.wait_for(asyncio.to_thread(transform_started.wait), timeout=2)
        run_task.cancel()
        await asyncio.sleep(0)

        # Cancellation cannot escape the SQLite ownership boundary while the
        # checkpoint worker still owns the writer connection. In particular,
        # factory release and run-fence cleanup must not race that transaction.
        assert store._lock.locked()
        assert not run_task.done()
        assert release_actions == []
        assert not invocation_release_started.is_set()

        allow_transform_return.set()
        with pytest.raises(asyncio.CancelledError):
            await run_task

        assert invocation_release_started.is_set()
        assert invocation_release_completed.is_set()
        checkpoint = await store.load_checkpoint("sess_cancelled_sqlite_factory_checkpoint")
        await store.close()
        assert checkpoint is not None
        return release_actions, checkpoint

    release_actions, checkpoint = asyncio.run(run())

    assert release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]
    assert checkpoint["environment_factory_reconnect"] == {
        "dynamic": {"sandbox_id": "sandbox_committed_after_cancel"}
    }
    assert checkpoint["environment_factory_allocation_owner"] == {
        "dynamic": "sess_cancelled_sqlite_factory_checkpoint"
    }
