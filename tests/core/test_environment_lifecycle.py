from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

import pytest
from tests.core._workload_secret_support import FakeProvider, collect_events

import cayu.runtime._environment_lifecycle as environment_lifecycle_module
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.environments import (
    BoundWorkspace,
    Environment,
    EnvironmentSpec,
    WorkspaceBinding,
    WorkspaceInstructions,
    WorkspaceSnapshot,
)
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    attach_environment_factory_cleanup_settlement_task,
    register_environment_factory_cleanup_retry,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    ResumeRequest,
    RunRequest,
    SessionIdentity,
    SessionStatus,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._environment_lifecycle import (
    ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY,
    ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY,
    EnvironmentLifecycle,
    render_initial_system_prompt,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime.budgets import InMemoryBudgetStore
from cayu.runtime.sessions import CheckpointTransform, Session
from cayu.workspaces import Workspace


def _preserve_session_control_state(
    checkpoint: dict[str, Any],
) -> CheckpointTransform:
    replacement = deepcopy(checkpoint)

    def transform(_session: Session, current: dict[str, Any] | None) -> dict[str, Any]:
        updated = deepcopy(replacement)
        if current is not None and "pending_session_interrupt" in current:
            updated["pending_session_interrupt"] = deepcopy(current["pending_session_interrupt"])
        return updated

    return transform


def _lifecycle(store: InMemorySessionStore) -> EnvironmentLifecycle:
    return EnvironmentLifecycle(
        session_store=store,
        event_writer=RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=(),
        ),
        checkpoint_transform=_preserve_session_control_state,
    )


class _FakeProvider(ModelProvider):
    name = "fake"

    def __init__(
        self,
        events: list[ModelStreamEvent] | list[list[ModelStreamEvent]],
    ) -> None:
        if events and isinstance(events[0], list):
            self.event_batches = events  # type: ignore[assignment]
        else:
            self.event_batches = [events]  # type: ignore[list-item]
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        batch_index = len(self.requests) - 1
        if batch_index >= len(self.event_batches):
            raise AssertionError(f"No fake provider event batch for request {batch_index}")
        for event in self.event_batches[batch_index]:
            yield event


class _RecordingWorkspaceBinding(WorkspaceBinding):
    def __init__(self) -> None:
        self.finalize_calls: list[dict[str, Any]] = []

    async def bind(
        self,
        workspace: Workspace | None,
        runner: Any,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        del session_id, agent_name, environment_name, metadata
        return BoundWorkspace(
            workspace=workspace,
            source_workspace=workspace,
            runner=runner,
            path="/bound",
            metadata={"binding": "recording"},
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        self.finalize_calls.append(
            {
                "bound": bound,
                "outcome": outcome,
                "metadata": metadata,
            }
        )
        return None


async def _collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


def test_checkpoint_preserves_factory_reconnect_state_and_current_control_state() -> None:
    async def scenario() -> dict[str, Any] | None:
        session_id = "environment_checkpoint_preservation"
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.checkpoint(
            session_id,
            {
                ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY: {
                    "sandbox": {"sandbox_id": "sandbox-1"}
                },
                ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY: {"sandbox": session_id},
                "pending_session_interrupt": {"reason": "current"},
                "stale_context": True,
            },
        )

        await _lifecycle(store).checkpoint_preserving_runtime_state(
            session_id,
            {
                "context_compaction": {"summary": "bounded"},
                "pending_session_interrupt": {"reason": "stale"},
            },
        )
        return await store.load_checkpoint(session_id)

    assert asyncio.run(scenario()) == {
        ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY: {"sandbox": {"sandbox_id": "sandbox-1"}},
        ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY: {
            "sandbox": "environment_checkpoint_preservation"
        },
        "context_compaction": {"summary": "bounded"},
        "pending_session_interrupt": {"reason": "current"},
    }


def test_checkpoint_preservation_rejects_deleting_transform() -> None:
    store = InMemorySessionStore()

    def deleting_transform(_checkpoint: dict[str, Any]) -> CheckpointTransform:
        def transform(
            _session: Session,
            _current: dict[str, Any] | None,
        ) -> None:
            return None

        return transform

    lifecycle = EnvironmentLifecycle(
        session_store=store,
        event_writer=RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=(),
        ),
        checkpoint_transform=deleting_transform,
    )
    transform = lifecycle.checkpoint_transform_preserving_runtime_state(
        {"context_compaction": {"summary": "bounded"}}
    )

    with pytest.raises(
        RuntimeError,
        match="Checkpoint preservation transform unexpectedly deleted the checkpoint",
    ):
        transform(
            Session(
                id="environment_checkpoint_deletion",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            ),
            None,
        )


def test_cancellation_during_factory_resolution_finalizes_before_fence_release() -> None:
    class OrderingStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.lifecycle_order: list[str] = []

        async def publish_interaction_transition(
            self,
            session_id: str,
            *,
            event: Event,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            only_if_no_queued_messages: bool = False,
        ):
            result = await super().publish_interaction_transition(
                session_id,
                event=event,
                from_statuses=from_statuses,
                to_status=to_status,
                only_if_no_queued_messages=only_if_no_queued_messages,
            )
            if to_status == SessionStatus.INTERRUPTED:
                self.lifecycle_order.append("finalized")
            return result

        async def release_run_fence(self, session_id: str) -> None:
            self.lifecycle_order.append("released")
            await super().release_run_fence(session_id)

    class BlockingFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.entered = asyncio.Event()

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            self.entered.set()
            await asyncio.Event().wait()
            raise AssertionError("Blocked factory unexpectedly resumed.")

    async def scenario() -> None:
        session_id = "sess_cancelled_pre_run_factory"
        store = OrderingStore()
        factory = BlockingFactory()
        provider = FakeProvider([])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
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
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await asyncio.wait_for(factory.entered.wait(), timeout=1)
        run_task.cancel("cancel pre-run factory")
        with pytest.raises(asyncio.CancelledError, match="cancel pre-run factory"):
            await run_task

        session = await store.load(session_id)
        assert session is not None
        assert session.status == SessionStatus.INTERRUPTED
        assert store.lifecycle_order == ["finalized", "released"]
        events = await store.load_events(session_id)
        interrupted = [event for event in events if event.type == EventType.SESSION_INTERRUPTED]
        assert len(interrupted) == 1
        assert interrupted[0].payload["abandoned"] is True
        assert provider.requests == []

    asyncio.run(scenario())


def test_cancelled_binding_does_not_repeat_deferred_factory_release() -> None:
    class BlockingBinding(WorkspaceBinding):
        def __init__(self) -> None:
            self.entered = asyncio.Event()

        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            del workspace, runner, kwargs
            self.entered.set()
            await asyncio.Event().wait()
            raise AssertionError("Blocked binding unexpectedly resumed.")

        async def finalize(self, bound, **kwargs):  # type: ignore[no-untyped-def]
            del bound, kwargs
            raise AssertionError("Unbound workspace must not be finalized.")

    class DeferredReleaseFactory(EnvironmentFactory):
        def __init__(self, binding: BlockingBinding) -> None:
            self.binding = binding
            self.release_calls = 0
            self.release_started = asyncio.Event()
            self.allow_release = asyncio.Event()

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            async def release(action: EnvironmentFactoryReleaseAction) -> None:
                assert action is EnvironmentFactoryReleaseAction.PRESERVE
                self.release_calls += 1
                self.release_started.set()
                await self.allow_release.wait()

            return EnvironmentFactoryResult(
                Environment(
                    EnvironmentSpec(name=request.environment_name),
                    binding=self.binding,
                ),
                release=release,
                release_timeout_s=0.01,
            )

    async def scenario() -> tuple[DeferredReleaseFactory, CayuApp]:
        session_id = "sess_cancelled_binding_deferred_release"
        binding = BlockingBinding()
        factory = DeferredReleaseFactory(binding)
        app = CayuApp(enable_logging=False)
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
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await asyncio.wait_for(binding.entered.wait(), timeout=1)
        run_task.cancel("cancel during binding")
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await run_task
        assert any(isinstance(error, asyncio.CancelledError) for error in exc_info.value.exceptions)
        assert any(isinstance(error, TimeoutError) for error in exc_info.value.exceptions)

        assert factory.release_started.is_set()
        assert factory.release_calls == 1
        assert session_id in app._environment_lifecycle._deferred_factory_cleanup_tasks

        factory.allow_release.set()
        assert await app.drain_environment_cleanups(timeout_s=0.2) is True
        return factory, app

    factory, app = asyncio.run(scenario())
    assert factory.release_calls == 1
    assert app._environment_lifecycle._active_environment_setups == {}
    assert app._environment_lifecycle._pending_environment_owner_admissions == set()
    assert app._environment_lifecycle._deferred_factory_cleanup_tasks == {}


def test_render_initial_system_prompt_keeps_agent_and_workspace_provenance() -> None:
    rendered = render_initial_system_prompt(
        agent_system_prompt="  Be careful.  ",
        workspace_instructions=WorkspaceInstructions(
            content="  Use the repository test command.  ",
            sources=("AGENTS.md", "docs/runtime.md"),
        ),
    )

    assert rendered == (
        "[Agent instructions]\n"
        "Be careful.\n\n"
        "[Workspace instructions]\n"
        "Source: AGENTS.md, docs/runtime.md\n"
        "These instructions apply only to the active workspace. If they conflict "
        "with agent, tool, approval, sandbox, or secret policy, follow the "
        "higher-priority runtime policy.\n\n"
        "Use the repository test command."
    )


def test_environment_abort_waits_for_terminal_binding_finalize_quiescence() -> None:
    class BlockingFinalizeBinding(_RecordingWorkspaceBinding):
        def __init__(self) -> None:
            super().__init__()
            self.finalize_started = asyncio.Event()
            self.release_finalize = asyncio.Event()
            self.finalize_finished = False
            self.abandon_calls: list[BoundWorkspace] = []

        async def finalize(
            self,
            bound: BoundWorkspace,
            *,
            outcome: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> WorkspaceSnapshot | None:
            self.finalize_started.set()
            await self.release_finalize.wait()
            self.finalize_finished = True
            return await super().finalize(bound, outcome=outcome, metadata=metadata)

        def abandon(self, bound: BoundWorkspace) -> None:
            assert self.finalize_finished
            self.abandon_calls.append(bound)

    async def run() -> tuple[list[Event], BlockingFinalizeBinding]:
        binding = BlockingFinalizeBinding()
        app = CayuApp(enable_logging=False)
        app.register_provider(
            _FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
            default=True,
        )
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), binding=binding),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        events_task = asyncio.create_task(
            _collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_binding_finalize_abort_race",
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await binding.finalize_started.wait()

        # `cleanup_started` is only an ownership claim. A concurrent abort must
        # leave the setup registered and must not abandon the bound workspace
        # until the terminal finalizer has actually returned.
        await app._environment_lifecycle.abort_environment_setup(
            session_id="sess_binding_finalize_abort_race",
            original_error=None,
        )
        assert binding.abandon_calls == []
        assert (
            "sess_binding_finalize_abort_race"
            in app._environment_lifecycle._active_environment_setups
        )

        binding.release_finalize.set()
        events = await events_task
        assert (
            "sess_binding_finalize_abort_race"
            not in app._environment_lifecycle._active_environment_setups
        )
        return events, binding

    events, binding = asyncio.run(run())
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(binding.finalize_calls) == 1
    assert len(binding.abandon_calls) == 1
    assert binding.abandon_calls[0] is binding.finalize_calls[0]["bound"]


def test_lazy_environment_cleanup_sweep_rotates_failed_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetainedCleanup:
        cleanup_started = True
        cleanup_finished = True
        cleanup_settlement_task = None

    async def run() -> list[str]:
        app = CayuApp(enable_logging=False)
        lifecycle = app._environment_lifecycle
        for index in range(
            environment_lifecycle_module._MAX_LAZY_ENVIRONMENT_CLEANUP_SETTLEMENTS + 1
        ):
            lifecycle._active_environment_setups[f"retained-{index}"] = RetainedCleanup()  # type: ignore[assignment]

        attempts: list[str] = []
        recoverable_session_id = (
            f"retained-{environment_lifecycle_module._MAX_LAZY_ENVIRONMENT_CLEANUP_SETTLEMENTS}"
        )

        async def settle_one(
            *,
            session_id: str,
            original_error: BaseException | None,
            allow_deferred_settlement: bool = False,
        ) -> None:
            del original_error
            assert allow_deferred_settlement
            attempts.append(session_id)
            if session_id == recoverable_session_id:
                del lifecycle._active_environment_setups[session_id]
                return
            raise RuntimeError("cleanup remains unavailable")

        monkeypatch.setattr(lifecycle, "abort_environment_setup", settle_one)
        await lifecycle._settle_retained_environment_cleanups()
        assert recoverable_session_id not in attempts
        await lifecycle._settle_retained_environment_cleanups()
        assert recoverable_session_id not in lifecycle._active_environment_setups
        return attempts

    attempts = asyncio.run(run())
    recoverable_session_id = (
        f"retained-{environment_lifecycle_module._MAX_LAZY_ENVIRONMENT_CLEANUP_SETTLEMENTS}"
    )
    assert recoverable_session_id in attempts


def test_lazy_environment_cleanup_does_not_await_unresolved_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetainedCleanup:
        cleanup_started = True
        cleanup_finished = True
        cleanup_settlement_task = None

    async def run() -> tuple[list[str], int]:
        app = CayuApp(enable_logging=False)
        lifecycle = app._environment_lifecycle
        blocked_owner = RetainedCleanup()
        recoverable_owner = RetainedCleanup()
        lifecycle._active_environment_setups["blocked"] = blocked_owner  # type: ignore[assignment]
        lifecycle._active_environment_setups["recoverable"] = recoverable_owner  # type: ignore[assignment]
        release_blocked = asyncio.Event()
        attempts: list[str] = []

        async def settle_one(
            *,
            session_id: str,
            original_error: BaseException | None,
            allow_deferred_settlement: bool = False,
        ) -> None:
            del original_error
            assert allow_deferred_settlement
            attempts.append(session_id)
            if session_id == "blocked":
                await release_blocked.wait()
            del lifecycle._active_environment_setups[session_id]

        monkeypatch.setattr(lifecycle, "abort_environment_setup", settle_one)
        await asyncio.wait_for(
            lifecycle._settle_retained_environment_cleanups(),
            timeout=1,
        )
        assert "recoverable" not in lifecycle._active_environment_setups
        assert lifecycle._active_environment_setups["blocked"] is blocked_owner
        assert blocked_owner.cleanup_settlement_task is not None
        assert not blocked_owner.cleanup_settlement_task.done()
        assert attempts.count("blocked") == 1

        # Polling an already dispatched owner must neither duplicate nor cancel it.
        await lifecycle._settle_retained_environment_cleanups()
        assert attempts.count("blocked") == 1
        assert not blocked_owner.cleanup_settlement_task.done()

        release_blocked.set()
        await blocked_owner.cleanup_settlement_task
        assert "blocked" not in lifecycle._active_environment_setups
        return attempts, attempts.count("blocked")

    attempts, blocked_attempts = asyncio.run(run())
    assert attempts == ["blocked", "recoverable"]
    assert blocked_attempts == 1


def test_lazy_environment_cleanup_bounds_unresolved_settlement_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetainedCleanup:
        cleanup_started = True
        cleanup_finished = True
        cleanup_settlement_task = None

    async def run() -> tuple[int, int]:
        app = CayuApp(enable_logging=False)
        lifecycle = app._environment_lifecycle
        owner_count = environment_lifecycle_module._MAX_LAZY_ENVIRONMENT_CLEANUP_SETTLEMENTS + 3
        for index in range(owner_count):
            lifecycle._active_environment_setups[f"blocked-{index}"] = RetainedCleanup()  # type: ignore[assignment]
        release = asyncio.Event()
        attempts: list[str] = []

        async def settle_one(
            *,
            session_id: str,
            original_error: BaseException | None,
            allow_deferred_settlement: bool = False,
        ) -> None:
            del original_error
            assert allow_deferred_settlement
            attempts.append(session_id)
            await release.wait()

        monkeypatch.setattr(lifecycle, "abort_environment_setup", settle_one)
        await lifecycle._settle_retained_environment_cleanups()
        await lifecycle._settle_retained_environment_cleanups()
        pending = sum(
            owner.cleanup_settlement_task is not None and not owner.cleanup_settlement_task.done()
            for owner in lifecycle._active_environment_setups.values()
        )
        release.set()
        await asyncio.gather(
            *(
                owner.cleanup_settlement_task
                for owner in lifecycle._active_environment_setups.values()
                if owner.cleanup_settlement_task is not None
            )
        )
        return len(attempts), pending

    attempts, pending = asyncio.run(run())
    expected = environment_lifecycle_module._MAX_LAZY_ENVIRONMENT_CLEANUP_SETTLEMENTS
    assert attempts == expected
    assert pending == expected


def test_lazy_cleanup_retries_internal_cancellation_from_prior_event_loop() -> None:
    class RetainedCleanup:
        cleanup_started = True
        cleanup_finished = True
        cleanup_settlement_task = None

    app = CayuApp(enable_logging=False)
    lifecycle = app._environment_lifecycle
    retained = RetainedCleanup()
    lifecycle._active_environment_setups["prior-loop"] = retained  # type: ignore[assignment]
    attempts = 0

    async def settle_one(
        *,
        session_id: str,
        original_error: BaseException | None,
        allow_deferred_settlement: bool = False,
    ) -> None:
        nonlocal attempts
        del original_error
        assert session_id == "prior-loop"
        assert allow_deferred_settlement
        attempts += 1
        if attempts == 1:
            await asyncio.Event().wait()
        del lifecycle._active_environment_setups[session_id]

    lifecycle.abort_environment_setup = settle_one  # type: ignore[method-assign]
    asyncio.run(lifecycle._settle_retained_environment_cleanups())
    assert retained.cleanup_settlement_task is not None
    assert retained.cleanup_settlement_task.done()

    # The next loop harvests shutdown cancellation as internal state and
    # retries the exact owner instead of cancelling this unrelated caller.
    asyncio.run(lifecycle._settle_retained_environment_cleanups())

    assert attempts == 2
    assert "prior-loop" not in lifecycle._active_environment_setups


def test_lazy_cleanup_child_cancellation_does_not_cancel_unrelated_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetainedCleanup:
        cleanup_started = True
        cleanup_finished = True
        cleanup_settlement_task = None

    async def run() -> tuple[list[Event], list[Event], asyncio.Task[Any], int]:
        app = CayuApp(enable_logging=False)
        app.register_provider(
            _FakeProvider(
                [
                    [ModelStreamEvent.completed({"finish_reason": "stop"})],
                    [ModelStreamEvent.completed({"finish_reason": "stop"})],
                ]
            ),
            default=True,
        )
        app.register_environment(
            Environment(EnvironmentSpec(name="unrelated")),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        lifecycle = app._environment_lifecycle
        retained = RetainedCleanup()
        lifecycle._active_environment_setups["cancelled-cleanup-owner"] = retained  # type: ignore[assignment]
        attempts = 0
        settlement_task: asyncio.Task[Any] | None = None
        allow_retry_completion = asyncio.Event()
        original_abort = lifecycle.abort_environment_setup

        async def settle_one(
            *,
            session_id: str,
            original_error: BaseException | None,
            allow_deferred_settlement: bool = False,
        ) -> None:
            nonlocal attempts, settlement_task
            if session_id != "cancelled-cleanup-owner":
                await original_abort(
                    session_id=session_id,
                    original_error=original_error,
                    allow_deferred_settlement=allow_deferred_settlement,
                )
                return
            assert allow_deferred_settlement
            attempts += 1
            if attempts == 1:
                current = asyncio.current_task()
                assert current is not None
                settlement_task = current
                child = asyncio.create_task(asyncio.Event().wait())
                await asyncio.sleep(0)
                child.cancel("old cleanup child cancelled")
                await child
            await allow_retry_completion.wait()
            del lifecycle._active_environment_setups[session_id]

        monkeypatch.setattr(lifecycle, "abort_environment_setup", settle_one)
        first_admission = asyncio.create_task(
            _collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="unrelated-first",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )
        first_events = await asyncio.wait_for(first_admission, timeout=1)
        assert first_events[-1].type == EventType.SESSION_COMPLETED
        assert first_admission.cancelling() == 0
        assert not first_admission.cancelled()
        assert "cancelled-cleanup-owner" in lifecycle._active_environment_setups
        assert settlement_task is not None
        assert settlement_task.done()
        assert settlement_task.cancelling() == 0
        assert not settlement_task.cancelled()
        assert attempts == 2
        retry_task = retained.cleanup_settlement_task
        assert retry_task is not None
        assert not retry_task.done()

        allow_retry_completion.set()
        second_admission = asyncio.create_task(
            _collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="unrelated-second",
                    messages=[Message.text("user", "continue")],
                ),
            )
        )
        second_events = await asyncio.wait_for(second_admission, timeout=1)
        assert second_events[-1].type == EventType.SESSION_COMPLETED
        assert second_admission.cancelling() == 0
        assert not second_admission.cancelled()
        assert "cancelled-cleanup-owner" not in lifecycle._active_environment_setups
        return first_events, second_events, settlement_task, attempts

    first_events, second_events, settlement_task, attempts = asyncio.run(run())
    assert first_events[-1].type == EventType.SESSION_COMPLETED
    assert second_events[-1].type == EventType.SESSION_COMPLETED
    assert settlement_task.done()
    assert not settlement_task.cancelled()
    assert attempts == 2


def test_unresolved_lazy_cleanup_does_not_block_unrelated_environment_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetainedCleanup:
        cleanup_started = True
        cleanup_finished = True
        cleanup_settlement_task = None

    async def run() -> tuple[list[Event], int]:
        app = CayuApp(enable_logging=False)
        app.register_provider(
            _FakeProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]]),
            default=True,
        )
        app.register_environment(
            Environment(EnvironmentSpec(name="unrelated")),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        lifecycle = app._environment_lifecycle
        retained = RetainedCleanup()
        lifecycle._active_environment_setups["blocked-owner"] = retained  # type: ignore[assignment]
        release = asyncio.Event()
        attempts = 0
        original_abort = lifecycle.abort_environment_setup

        async def settle_one(
            *,
            session_id: str,
            original_error: BaseException | None,
            allow_deferred_settlement: bool = False,
        ) -> None:
            nonlocal attempts
            if session_id != "blocked-owner":
                await original_abort(
                    session_id=session_id,
                    original_error=original_error,
                    allow_deferred_settlement=allow_deferred_settlement,
                )
                return
            assert allow_deferred_settlement
            attempts += 1
            await release.wait()
            del lifecycle._active_environment_setups[session_id]

        monkeypatch.setattr(lifecycle, "abort_environment_setup", settle_one)
        events = await asyncio.wait_for(
            _collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="unrelated-session",
                    messages=[Message.text("user", "continue")],
                ),
            ),
            timeout=1,
        )
        assert events[-1].type == EventType.SESSION_COMPLETED
        assert retained.cleanup_settlement_task is not None
        assert not retained.cleanup_settlement_task.done()
        assert attempts == 1

        release.set()
        await retained.cleanup_settlement_task
        return events, attempts

    events, attempts = asyncio.run(run())
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert attempts == 1


def test_environment_owner_capacity_fails_before_binding_mutation() -> None:
    class RetainedOwner:
        cleanup_started = False
        cleanup_settlement_task = None

    class RecordingBinding(WorkspaceBinding):
        def __init__(self) -> None:
            self.bind_calls = 0

        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            del workspace, runner, kwargs
            self.bind_calls += 1
            return BoundWorkspace()

        async def finalize(self, bound, **kwargs):  # type: ignore[no-untyped-def]
            del bound, kwargs
            return None

    async def run() -> tuple[list[Event], RecordingBinding, CayuApp]:
        binding = RecordingBinding()
        app = CayuApp(
            enable_logging=False,
            max_environment_lifecycle_owners=1,
        )
        app.register_provider(
            _FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
            default=True,
        )
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), binding=binding),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        app._environment_lifecycle._active_environment_setups["retained"] = RetainedOwner()  # type: ignore[assignment]
        events = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="capacity-refused",
                messages=[Message.text("user", "run")],
            ),
        )
        return events, binding, app

    events, binding, app = asyncio.run(run())

    assert events[-1].type == EventType.SESSION_FAILED
    assert "EnvironmentCapacityError" in str(events[-1].payload)
    assert binding.bind_calls == 0
    assert app._environment_lifecycle._pending_environment_owner_admissions == set()


def test_environment_owner_capacity_deduplicates_one_transferred_session() -> None:
    class RetainedOwner:
        cleanup_started = True
        cleanup_settlement_task = None

    app = CayuApp(
        enable_logging=False,
        max_environment_lifecycle_owners=2,
    )
    lifecycle = app._environment_lifecycle
    lifecycle._active_environment_setups["transferred"] = RetainedOwner()  # type: ignore[assignment]
    lifecycle._pending_environment_owner_admissions.add("transferred")

    lifecycle._reserve_environment_owner_admission("contender")

    assert lifecycle._pending_environment_owner_admissions == {
        "transferred",
        "contender",
    }


def test_deferred_factory_failure_retains_capacity_until_cleanup_settles() -> None:
    class DeferredFailureFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.calls = 0
            self.release = asyncio.Event()

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            self.calls += 1
            if self.calls > 1:
                return EnvironmentFactoryResult(
                    Environment(EnvironmentSpec(name=request.environment_name))
                )

            async def settle() -> None:
                await self.release.wait()

            settlement_task = asyncio.create_task(
                settle(),
                name=f"deferred-factory-cleanup-{request.session_id}",
            )
            error = RuntimeError("factory failed with deferred cleanup")
            attach_environment_factory_cleanup_settlement_task(
                error,
                settlement_task,
            )
            raise error

    async def run() -> tuple[
        list[Event],
        list[Event],
        list[Event],
        DeferredFailureFactory,
        CayuApp,
    ]:
        factory = DeferredFailureFactory()
        app = CayuApp(
            enable_logging=False,
            max_environment_lifecycle_owners=1,
        )
        app.register_provider(
            _FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
            default=True,
        )
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        first = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="deferred-owner",
                messages=[Message.text("user", "run")],
            ),
        )
        second = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="capacity-contender",
                messages=[Message.text("user", "run")],
            ),
        )
        assert factory.calls == 1
        assert "deferred-owner" in app._environment_lifecycle._pending_environment_owner_admissions
        assert "deferred-owner" in app._environment_lifecycle._deferred_factory_cleanup_tasks

        factory.release.set()
        assert await app.drain_environment_cleanups(timeout_s=0.2) is True
        with pytest.raises(RuntimeError, match="authoritative initial transcript"):
            _ = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="deferred-owner",
                        messages=[Message.text("user", "continue after cleanup")],
                    )
                )
            ]
        continuation = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="deferred-owner-retry",
                messages=[Message.text("user", "retry after cleanup")],
            ),
        )
        return first, second, continuation, factory, app

    first, second, continuation, factory, app = asyncio.run(run())

    assert first[-1].type == EventType.SESSION_FAILED
    assert second[-1].type == EventType.SESSION_FAILED
    assert "EnvironmentCapacityError" in str(second[-1].payload)
    assert continuation[-1].type == EventType.SESSION_COMPLETED
    assert factory.calls == 2
    assert app._environment_lifecycle._pending_environment_owner_admissions == set()
    assert app._environment_lifecycle._deferred_factory_cleanup_tasks == {}


def test_explicit_drain_retries_same_failed_factory_cleanup_owner() -> None:
    class RecoverableCleanupFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.create_calls = 0
            self.cleanup_calls = 0
            self.allow_cleanup = False

        def cleanup_attempt(self) -> asyncio.Task[None]:
            async def cleanup() -> None:
                self.cleanup_calls += 1
                if not self.allow_cleanup:
                    raise PermissionError("provider cleanup denied")

            return asyncio.create_task(cleanup())

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            self.create_calls += 1
            if self.create_calls > 1:
                return EnvironmentFactoryResult(
                    Environment(EnvironmentSpec(name=request.environment_name))
                )
            cleanup_task = self.cleanup_attempt()
            register_environment_factory_cleanup_retry(
                cleanup_task,
                self.cleanup_attempt,
            )
            error = RuntimeError("factory failed with recoverable cleanup owner")
            attach_environment_factory_cleanup_settlement_task(error, cleanup_task)
            raise error

    async def run() -> tuple[
        list[Event],
        list[Event],
        list[Event],
        RecoverableCleanupFactory,
        CayuApp,
    ]:
        factory = RecoverableCleanupFactory()
        app = CayuApp(
            enable_logging=False,
            max_environment_lifecycle_owners=1,
        )
        app.register_provider(
            _FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
            default=True,
        )
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        first = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="recoverable-cleanup-owner",
                messages=[Message.text("user", "run")],
            ),
        )
        assert await app.drain_environment_cleanups(timeout_s=0.2) is False
        assert factory.cleanup_calls == 2
        assert (
            "recoverable-cleanup-owner"
            in app._environment_lifecycle._pending_environment_owner_admissions
        )

        contender = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="capacity-contender",
                messages=[Message.text("user", "run")],
            ),
        )
        factory.allow_cleanup = True
        assert await app.drain_environment_cleanups(timeout_s=0.2) is True
        with pytest.raises(RuntimeError, match="authoritative initial transcript"):
            _ = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="recoverable-cleanup-owner",
                        messages=[Message.text("user", "continue after recovery")],
                    )
                )
            ]
        continuation = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="recoverable-cleanup-owner-retry",
                messages=[Message.text("user", "retry after recovery")],
            ),
        )
        return first, contender, continuation, factory, app

    first, contender, continuation, factory, app = asyncio.run(run())

    assert first[-1].type == EventType.SESSION_FAILED
    assert contender[-1].type == EventType.SESSION_FAILED
    assert "EnvironmentCapacityError" in str(contender[-1].payload)
    assert continuation[-1].type == EventType.SESSION_COMPLETED
    assert factory.cleanup_calls == 3
    assert factory.create_calls == 2
    assert app._environment_lifecycle._pending_environment_owner_admissions == set()
    assert app._environment_lifecycle._deferred_factory_cleanup_tasks == {}


def test_grouped_factory_leaf_cleanups_retain_capacity_until_all_settle() -> None:
    class GroupedCleanupFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.calls = 0
            self.entered = asyncio.Event()
            self.allow_first_cleanup = asyncio.Event()
            self.allow_second_cleanup = asyncio.Event()
            self.first_cleanup_finished = asyncio.Event()

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            self.calls += 1
            if self.calls > 1:
                return EnvironmentFactoryResult(
                    Environment(EnvironmentSpec(name=request.environment_name))
                )
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:

                async def settle_first() -> None:
                    await self.allow_first_cleanup.wait()
                    self.first_cleanup_finished.set()

                async def settle_second() -> None:
                    await self.allow_second_cleanup.wait()

                first_error = RuntimeError("first grouped cleanup remains active")
                attach_environment_factory_cleanup_settlement_task(
                    first_error,
                    asyncio.create_task(settle_first()),
                )
                second_error = RuntimeError("second grouped cleanup remains active")
                attach_environment_factory_cleanup_settlement_task(
                    second_error,
                    asyncio.create_task(settle_second()),
                )
                raise BaseExceptionGroup(
                    "factory cancellation cleanup",
                    [
                        first_error,
                        BaseExceptionGroup(
                            "nested cleanup",
                            [second_error, cancellation],
                        ),
                    ],
                ) from None

    async def run() -> tuple[list[Event], list[Event], GroupedCleanupFactory, CayuApp]:
        factory = GroupedCleanupFactory()
        app = CayuApp(
            enable_logging=False,
            max_environment_lifecycle_owners=1,
        )
        app.register_provider(
            _FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
            default=True,
        )
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        first_task = asyncio.create_task(
            _collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="grouped-cleanup-owner",
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await factory.entered.wait()
        first_task.cancel("cancel grouped factory")
        with pytest.raises(BaseExceptionGroup):
            await first_task
        # The runtime owns and normalizes the delivered request before
        # propagating the grouped failure to its stream consumer.
        assert first_task.cancelling() == 0
        assert not first_task.cancelled()

        blocked_before = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="grouped-cleanup-contender-before",
                messages=[Message.text("user", "run")],
            ),
        )
        assert factory.calls == 1

        factory.allow_first_cleanup.set()
        await factory.first_cleanup_finished.wait()
        assert await app.drain_environment_cleanups(timeout_s=0.02) is False
        blocked_between = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="grouped-cleanup-contender-between",
                messages=[Message.text("user", "run")],
            ),
        )
        assert factory.calls == 1

        factory.allow_second_cleanup.set()
        assert await app.drain_environment_cleanups(timeout_s=0.2) is True
        completed = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="grouped-cleanup-contender-after",
                messages=[Message.text("user", "run")],
            ),
        )
        assert completed[-1].type is EventType.SESSION_COMPLETED
        return blocked_before, blocked_between, factory, app

    blocked_before, blocked_between, factory, app = asyncio.run(run())
    assert blocked_before[-1].type is EventType.SESSION_FAILED
    assert blocked_between[-1].type is EventType.SESSION_FAILED
    assert "EnvironmentCapacityError" in str(blocked_before[-1].payload)
    assert "EnvironmentCapacityError" in str(blocked_between[-1].payload)
    assert factory.calls == 2
    assert app._environment_lifecycle._pending_environment_owner_admissions == set()
    assert app._environment_lifecycle._deferred_factory_cleanup_tasks == {}


def test_timed_out_factory_release_adopts_late_cleanup_settlement() -> None:
    class FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            del workspace, runner, kwargs
            raise RuntimeError("bind failed")

        async def finalize(self, bound, **kwargs):  # type: ignore[no-untyped-def]
            del bound, kwargs
            raise AssertionError("unbound workspace must not be finalized")

    class LateSettlementFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.calls = 0
            self.release_started = asyncio.Event()
            self.allow_late_failure = asyncio.Event()
            self.allow_settlement = asyncio.Event()
            self.settlement_started = asyncio.Event()

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            self.calls += 1

            async def release(action: EnvironmentFactoryReleaseAction) -> None:
                assert action is EnvironmentFactoryReleaseAction.PRESERVE
                self.release_started.set()
                await self.allow_late_failure.wait()

                async def settle() -> None:
                    self.settlement_started.set()
                    await self.allow_settlement.wait()

                settlement_task = asyncio.create_task(
                    settle(),
                    name=f"late-factory-cleanup-{request.session_id}",
                )
                error = RuntimeError("late release failure")
                attach_environment_factory_cleanup_settlement_task(
                    error,
                    settlement_task,
                )
                raise ExceptionGroup(
                    "late release reported grouped cleanup failure",
                    [error],
                )

            return EnvironmentFactoryResult(
                Environment(
                    EnvironmentSpec(name=request.environment_name),
                    binding=FailingBinding(),
                ),
                release=release,
                release_timeout_s=0.01,
            )

    async def run() -> tuple[list[Event], list[Event], LateSettlementFactory, CayuApp]:
        factory = LateSettlementFactory()
        app = CayuApp(
            enable_logging=False,
            max_environment_lifecycle_owners=1,
        )
        app.register_provider(
            _FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
            default=True,
        )
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        first = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="late-settlement-owner",
                messages=[Message.text("user", "run")],
            ),
        )
        assert factory.release_started.is_set()
        assert "late-settlement-owner" in (
            app._environment_lifecycle._deferred_factory_cleanup_tasks
        )
        factory.allow_late_failure.set()
        await asyncio.wait_for(factory.settlement_started.wait(), timeout=1)

        contender = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="late-settlement-contender",
                messages=[Message.text("user", "run")],
            ),
        )
        assert factory.calls == 1
        factory.allow_settlement.set()
        assert await app.drain_environment_cleanups(timeout_s=0.2) is True
        return first, contender, factory, app

    first, contender, factory, app = asyncio.run(run())

    assert first[-1].type is EventType.SESSION_FAILED
    assert first[-1].payload["error"] == "bind failed"
    assert first[-1].payload["environment_factory_release"]["completed"] is False
    assert contender[-1].type is EventType.SESSION_FAILED
    assert "EnvironmentCapacityError" in str(contender[-1].payload)
    assert factory.calls == 1
    assert app._environment_lifecycle._deferred_factory_cleanup_tasks == {}
    assert app._environment_lifecycle._pending_environment_owner_admissions == set()


def test_timed_out_factory_release_retains_cyclic_cleanup_handoff() -> None:
    class FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            del workspace, runner, kwargs
            raise RuntimeError("bind failed")

        async def finalize(self, bound, **kwargs):  # type: ignore[no-untyped-def]
            del bound, kwargs
            raise AssertionError("unbound workspace must not be finalized")

    class CyclicReleaseFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.calls = 0
            self.release_started = asyncio.Event()
            self.allow_late_failure = asyncio.Event()

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            self.calls += 1

            async def release(action: EnvironmentFactoryReleaseAction) -> None:
                assert action is EnvironmentFactoryReleaseAction.PRESERVE
                self.release_started.set()
                await self.allow_late_failure.wait()
                release_task = asyncio.current_task()
                assert release_task is not None
                error = RuntimeError("late release reported a cyclic cleanup owner")
                attach_environment_factory_cleanup_settlement_task(
                    error,
                    release_task,
                )
                raise error

            return EnvironmentFactoryResult(
                Environment(
                    EnvironmentSpec(name=request.environment_name),
                    binding=FailingBinding(),
                ),
                release=release,
                release_timeout_s=0.01,
            )

    async def run() -> tuple[list[Event], list[Event], CyclicReleaseFactory, CayuApp]:
        factory = CyclicReleaseFactory()
        app = CayuApp(
            enable_logging=False,
            max_environment_lifecycle_owners=1,
        )
        app.register_provider(
            _FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
            default=True,
        )
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        first = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="cyclic-settlement-owner",
                messages=[Message.text("user", "run")],
            ),
        )
        assert factory.release_started.is_set()
        factory.allow_late_failure.set()
        retained = app._environment_lifecycle._deferred_factory_cleanup_tasks[
            "cyclic-settlement-owner"
        ]
        async with asyncio.timeout(0.2):
            while not retained.done():
                await asyncio.sleep(0)

        contender = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="cyclic-settlement-contender",
                messages=[Message.text("user", "run")],
            ),
        )
        return first, contender, factory, app

    first, contender, factory, app = asyncio.run(run())

    assert first[-1].type is EventType.SESSION_FAILED
    assert first[-1].payload["error"] == "bind failed"
    assert first[-1].payload["environment_factory_release"]["completed"] is False
    assert contender[-1].type is EventType.SESSION_FAILED
    assert "EnvironmentCapacityError" in str(contender[-1].payload)
    assert factory.calls == 1
    retained = app._environment_lifecycle._deferred_factory_cleanup_tasks["cyclic-settlement-owner"]
    assert retained.done()
    with pytest.raises(RuntimeError, match="cyclic cleanup owner"):
        retained.result()
    assert "cyclic-settlement-owner" in (
        app._environment_lifecycle._pending_environment_owner_admissions
    )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_cayu_app_rejects_invalid_environment_owner_capacity(value: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="max_environment_lifecycle_owners"):
        CayuApp(
            enable_logging=False,
            max_environment_lifecycle_owners=value,
        )


def test_environment_cleanup_drain_timeout_keeps_mutation_task_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetainedCleanup:
        cleanup_started = True
        cleanup_finished = True
        cleanup_settlement_task = None

    async def run() -> tuple[bool, asyncio.Task[Any]]:
        app = CayuApp(enable_logging=False)
        lifecycle = app._environment_lifecycle
        retained = RetainedCleanup()
        lifecycle._active_environment_setups["blocked"] = retained  # type: ignore[assignment]
        release = asyncio.Event()

        async def settle_one(**kwargs: Any) -> None:
            assert kwargs["session_id"] == "blocked"
            await release.wait()
            del lifecycle._active_environment_setups["blocked"]

        monkeypatch.setattr(lifecycle, "abort_environment_setup", settle_one)
        drained = await app.drain_environment_cleanups(timeout_s=0.01)
        task = retained.cleanup_settlement_task
        assert task is not None
        assert not task.done()
        assert task.cancelling() == 0
        release.set()
        await task
        return drained, task

    drained, task = asyncio.run(run())

    assert drained is False
    assert task.done()
    assert not task.cancelled()


def test_abandoned_started_boundary_releases_unmutated_capacity_admission() -> None:
    async def run() -> set[str]:
        app = CayuApp(
            enable_logging=False,
            max_environment_lifecycle_owners=1,
        )
        lifecycle = app._environment_lifecycle
        lifecycle._reserve_environment_owner_admission("abandoned-before-mutation")
        await lifecycle.abort_environment_setup(
            session_id="abandoned-before-mutation",
            original_error=GeneratorExit(),
        )
        return lifecycle._pending_environment_owner_admissions

    assert asyncio.run(run()) == set()


def test_environment_cleanup_drain_cancellation_keeps_mutation_task_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetainedCleanup:
        cleanup_started = True
        cleanup_finished = True
        cleanup_settlement_task = None

    async def run() -> tuple[asyncio.Task[Any], asyncio.Task[Any]]:
        app = CayuApp(enable_logging=False)
        lifecycle = app._environment_lifecycle
        retained = RetainedCleanup()
        lifecycle._active_environment_setups["blocked"] = retained  # type: ignore[assignment]
        dispatched = asyncio.Event()
        release = asyncio.Event()

        async def settle_one(**kwargs: Any) -> None:
            assert kwargs["session_id"] == "blocked"
            dispatched.set()
            await release.wait()
            del lifecycle._active_environment_setups["blocked"]

        monkeypatch.setattr(lifecycle, "abort_environment_setup", settle_one)
        drain_task = asyncio.create_task(app.drain_environment_cleanups(timeout_s=10))
        await dispatched.wait()
        mutation_task = retained.cleanup_settlement_task
        assert mutation_task is not None

        drain_task.cancel("caller stopped waiting")
        with pytest.raises(asyncio.CancelledError, match="caller stopped waiting"):
            await drain_task
        assert drain_task.cancelling() == 1
        assert drain_task.cancelled()
        assert mutation_task.cancelling() == 0
        assert not mutation_task.done()

        release.set()
        await mutation_task
        return drain_task, mutation_task

    drain_task, mutation_task = asyncio.run(run())

    assert drain_task.cancelled()
    assert mutation_task.done()
    assert not mutation_task.cancelled()


def test_aborted_setup_retained_owner_retries_on_first_cleanup_sweep() -> None:
    cleanup_error = RuntimeError("initial cleanup failed")

    class RetainedOnceBinding(WorkspaceBinding):
        def __init__(self) -> None:
            self.finalize_calls = 0
            self.abandon_calls = 0

        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("bind should not run")

        async def finalize(
            self,
            bound: BoundWorkspace,
            *,
            outcome: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> WorkspaceSnapshot | None:
            self.finalize_calls += 1
            if self.finalize_calls == 1:
                raise cleanup_error
            return None

        def abandon(self, bound: BoundWorkspace) -> bool:
            self.abandon_calls += 1
            return self.abandon_calls > 1

    async def run() -> RetainedOnceBinding:
        app = CayuApp(enable_logging=False)
        lifecycle = app._environment_lifecycle
        binding = RetainedOnceBinding()
        environment = Environment(EnvironmentSpec(name="retained"), binding=binding)
        registered_environment = runtime_records.RegisteredEnvironment(
            spec=environment.spec,
            environment=environment,
            bound_workspace=BoundWorkspace(),
        )
        lifecycle._active_environment_setups["retained-session"] = (
            environment_lifecycle_module._ActiveEnvironmentSetup(registered_environment)
        )

        with pytest.raises(BaseExceptionGroup) as exc_info:
            await lifecycle.abort_environment_setup(
                session_id="retained-session",
                original_error=RuntimeError("setup failed"),
            )
        assert cleanup_error in exc_info.value.exceptions
        retained = lifecycle._active_environment_setups["retained-session"]
        assert retained.cleanup_settlement_deferred
        assert binding.finalize_calls == 1
        assert binding.abandon_calls == 1

        await lifecycle._settle_retained_environment_cleanups()
        assert "retained-session" not in lifecycle._active_environment_setups
        return binding

    binding = asyncio.run(run())
    assert binding.finalize_calls == 2
    assert binding.abandon_calls == 2


def test_lazy_environment_cleanup_retries_fatal_only_finalization() -> None:
    fatal_signal = GeneratorExit("binding finalization interrupted")

    class FatalOnceBinding(_RecordingWorkspaceBinding):
        def __init__(self) -> None:
            super().__init__()
            self.abandon_calls: list[BoundWorkspace] = []

        async def finalize(
            self,
            bound: BoundWorkspace,
            *,
            outcome: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> WorkspaceSnapshot | None:
            result = await super().finalize(bound, outcome=outcome, metadata=metadata)
            if len(self.finalize_calls) == 1:
                raise fatal_signal
            return result

        def abandon(self, bound: BoundWorkspace) -> bool:
            self.abandon_calls.append(bound)
            return True

    async def run() -> FatalOnceBinding:
        binding = FatalOnceBinding()
        app = CayuApp(enable_logging=False)
        app.register_provider(
            _FakeProvider(
                [
                    [ModelStreamEvent.completed({"finish_reason": "stop"})],
                    [ModelStreamEvent.completed({"finish_reason": "stop"})],
                ]
            ),
            default=True,
        )
        app.register_environment(
            Environment(EnvironmentSpec(name="fatal-binding"), binding=binding),
            default=True,
        )
        app.register_environment(Environment(EnvironmentSpec(name="cleanup-trigger")))
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(GeneratorExit, match="binding finalization interrupted"):
            await _collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_fatal_binding_cleanup",
                    messages=[Message.text("user", "run")],
                ),
            )
        assert "sess_fatal_binding_cleanup" in (
            app._environment_lifecycle._active_environment_setups
        )
        assert binding.abandon_calls == []

        events = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_fatal_binding_cleanup_trigger",
                environment_name="cleanup-trigger",
                messages=[Message.text("user", "continue")],
            ),
        )
        assert events[-1].type == EventType.SESSION_COMPLETED
        assert "sess_fatal_binding_cleanup" not in (
            app._environment_lifecycle._active_environment_setups
        )
        return binding

    binding = asyncio.run(run())
    assert len(binding.finalize_calls) == 2
    assert len(binding.abandon_calls) == 1
    assert binding.abandon_calls[0] is binding.finalize_calls[0]["bound"]
