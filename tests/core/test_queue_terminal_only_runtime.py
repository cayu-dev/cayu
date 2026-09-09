from __future__ import annotations

import asyncio
import contextlib
import warnings
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import pytest
from tests.core.test_queued_session_messages import (
    BlockingTwoTurnProvider,
    CommitThenLoseDeliveryAcknowledgementStore,
)

import cayu.runtime._environment_exposure as exposure_module
import cayu.runtime.sessions as session_module
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.events import copy_event
from cayu.environments import Environment, EnvironmentSpec, SyncBinding
from cayu.runtime import (
    CayuApp,
    EnqueueSessionMessageRequest,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    InMemoryTaskStore,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    TaskCreate,
    TaskStatus,
)
from cayu.runtime._environment_exposure import require_environment_exposed
from cayu.runtime._environment_lifecycle import EnvironmentLifecycle
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.runtime.session_message_lifecycle import (
    SessionMessageAccessContext,
    SessionMessageAccessPolicy,
    SessionMessageConditions,
    SessionMessageQuery,
    SessionMessageTarget,
)
from cayu.runtime.sessions import (
    PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY,
    Session,
    SessionStatus,
)
from cayu.workspaces import LocalWorkspace


class _CompletionPublicationBarrierStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1
    terminal_interaction_publication_version = 1

    def __init__(self, pause_at: EventType | None) -> None:
        super().__init__()
        self.pause_at = pause_at
        self.dispatched = asyncio.Event()
        self.release_publication = asyncio.Event()
        self.publication_failure: OSError | None = None

    async def append_event(self, session_id: str, event: Event) -> None:
        if event.type == self.pause_at:
            self.dispatched.set()
            await self.release_publication.wait()
            if self.publication_failure is not None:
                failure, self.publication_failure = self.publication_failure, None
                raise failure
        await super().append_event(session_id, event)


@pytest.mark.parametrize("critical", [False, True])
@pytest.mark.parametrize("boundary", ["task", "turn", "terminal"])
@pytest.mark.parametrize(
    "control",
    [
        "close",
        "cancel",
        "cancel_after_history",
        "cancel_failure",
        "finish",
        "finish_after_history",
        "close_hook",
        "cancel_hook",
        "finish_hook",
    ],
)
def test_rejected_completion_publication_keeps_owner_until_quiescent(
    tmp_path, critical: bool, boundary: str, control: str
) -> None:
    async def run() -> None:
        with_task = boundary == "task"
        cancelling = control.startswith("cancel") and control != "cancel_hook"
        event_type = {
            "task": EventType.TASK_COMPLETED,
            "turn": EventType.TURN_COMPLETED,
            "terminal": EventType.SESSION_COMPLETED,
        }[boundary]
        store = _CompletionPublicationBarrierStore(event_type if cancelling else None)
        publication_failure = OSError("completion publication failed")
        if control == "cancel_failure":
            store.publication_failure = publication_failure
        task_store = InMemoryTaskStore()
        await task_store.create_task(TaskCreate(task_id="completion-task", type="respond"))
        provider = BlockingTwoTurnProvider()

        class BlockingCompletionHook(RuntimeHook):
            def __init__(self) -> None:
                self.entered = asyncio.Event()
                self.release = asyncio.Event()
                self.calls = 0
                self.cancelled = False

            async def after_session_completed(self, context: RuntimeHookContext) -> None:
                self.calls += 1
                self.entered.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

        hook = BlockingCompletionHook()
        app = CayuApp(
            session_store=store,
            task_store=task_store,
            runtime_hooks=[hook] if control.endswith("_hook") else [],
            enable_logging=False,
        )
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        if critical:
            source, target = tmp_path / "source", tmp_path / "target"
            source.mkdir()
            target.mkdir()
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="sync"),
                    workspace=LocalWorkspace(source, workspace_id="queue-source"),
                    binding=SyncBinding(
                        target_workspace=LocalWorkspace(target, workspace_id="queue-target")
                    ),
                ),
                default=True,
            )
        observed: list[Event] = []
        receipt_before = None
        caught_cancellation = False

        async def execute() -> None:
            nonlocal receipt_before, caught_cancellation
            try:
                async with contextlib.aclosing(
                    app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id="terminal-queue",
                            task_id="completion-task" if with_task else None,
                            messages=[Message.text("user", "initial")],
                            max_steps=1,
                        )
                    )
                ) as stream:
                    async for event in stream:
                        observed.append(event)
                        if event.type == EventType.SESSION_MESSAGE_EXPIRED:
                            predecessor = next(
                                event
                                for event in await store.load_events("terminal-queue")
                                if event.type == EventType.INTERACTION_COMPLETED
                            )
                            receipt_before = deepcopy(
                                store._session_operation_records["terminal-queue"][
                                    session_module._interaction_transition_storage_key(
                                        predecessor.id
                                    )
                                ]
                            )
                            assert receipt_before["status_changed"] is False
                            if control.endswith("after_history"):
                                current = asyncio.current_task()
                                assert current is not None
                                current.cancel("previously handled")
                                with contextlib.suppress(asyncio.CancelledError):
                                    await asyncio.sleep(0)
                                assert current.cancelling() == 1
                        if control in {"close", "close_hook"} and event.type == event_type:
                            persisted = await store.load_events("terminal-queue")
                            assert (
                                sum(e.type == EventType.SESSION_COMPLETED for e in persisted) == 1
                            )
                            break
            except asyncio.CancelledError:
                caught_cancellation = True
                raise

        owner = asyncio.create_task(execute())
        try:
            started = asyncio.create_task(provider.first_started.wait())
            await asyncio.wait({owner, started}, timeout=10, return_when=asyncio.FIRST_COMPLETED)
            if owner.done():
                await owner
                pytest.fail(repr([e.model_dump() for e in observed]))
            assert started.done()
            if critical:
                (target / "result.txt").write_text("committed output")
            await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id="terminal-queue",
                    idempotency_key="expired",
                    content="must not reach provider",
                    delivery_mode="on_idle",
                    conditions=SessionMessageConditions(
                        expires_at=datetime.now(UTC) - timedelta(seconds=1)
                    ),
                )
            )
            provider.release_first.set()
            if control in {"cancel_hook", "finish_hook"}:
                entered = asyncio.create_task(hook.entered.wait())
                await asyncio.wait(
                    {owner, entered}, timeout=10, return_when=asyncio.FIRST_COMPLETED
                )
                if owner.done():
                    await owner
                assert entered.done(), "terminal hook was not entered"
                assert any(e.type == EventType.SESSION_COMPLETED for e in observed)
                if control == "cancel_hook":
                    owner.cancel("hook cancelled")
                    done, _ = await asyncio.wait({owner}, timeout=10)
                    assert owner in done, "post-terminal hook must receive caller cancellation"
                    with pytest.raises(asyncio.CancelledError) as cancelled:
                        await owner
                    assert cancelled.value.args == ("hook cancelled",)
                    assert owner.cancelled() and owner.cancelling() == 1
                    assert caught_cancellation and hook.cancelled
                else:
                    hook.release.set()
            if cancelling:
                await asyncio.wait_for(store.dispatched.wait(), 10)
                owner.cancel("publication cancelled")
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not owner.done(), "dispatched completion must reach quiescence"
                pending = await store.load("terminal-queue")
                profile = active_invocation_execution_profile_from_checkpoint(
                    await store.load_checkpoint("terminal-queue")
                )
                assert pending is not None and profile is not None
                assert pending.run_epoch == profile.run_epoch
                store.release_publication.set()
                with pytest.raises(asyncio.CancelledError) as cancelled:
                    await asyncio.wait_for(owner, 20)
                assert cancelled.value.args == ("publication cancelled",)
                if control == "cancel_failure":
                    assert cancelled.value.__cause__ is publication_failure
                assert owner.cancelled() and caught_cancellation
                assert owner.cancelling() == (2 if control == "cancel_after_history" else 1)
            elif control != "cancel_hook":
                await asyncio.wait_for(owner, 20)
                if control == "finish_after_history":
                    assert owner.cancelling() == 1 and not owner.cancelled()
        finally:
            store.release_publication.set()
            hook.release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
        final = await store.load("terminal-queue")
        assert final is not None and final.status is SessionStatus.COMPLETED
        if control == "cancel_failure":
            profile = active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint(final.id)
            )
            assert profile is not None and final.run_epoch == profile.run_epoch
            assert not any(
                e.type == EventType.SESSION_COMPLETED for e in await store.load_events(final.id)
            )
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=final.id, inactive_for_seconds=0)
            )
            final = await store.load(final.id)
            assert final is not None
        checkpoint = await store.load_checkpoint(final.id)
        profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert profile is not None and final.run_epoch == profile.run_epoch + 1
        assert PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY not in (checkpoint or {})
        events = await store.load_events(final.id)
        if control.endswith("_hook"):
            expected_hooks = 0 if control == "close_hook" else 1
            assert hook.calls == expected_hooks
            assert sum(e.type == EventType.HOOK_STARTED for e in events) == expected_hooks
            if expected_hooks:
                durable_types = [event.type for event in events]
                assert durable_types.index(EventType.SESSION_COMPLETED) < durable_types.index(
                    EventType.HOOK_STARTED
                )
            # Cancellation leaves the existing at-most-once reservation, not a
            # fabricated ordinary-failure outcome or a retryable hook slot.
            assert sum(e.type == EventType.HOOK_FAILED for e in events) == 0
            assert sum(e.type == EventType.HOOK_COMPLETED for e in events) == (
                1 if control == "finish_hook" else 0
            )
        for kind in (
            EventType.INTERACTION_STARTED,
            EventType.INTERACTION_COMPLETED,
            EventType.SESSION_MESSAGE_EXPIRED,
            EventType.TASK_COMPLETED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_COMPLETED,
        ):
            expected = 0 if kind == EventType.TASK_COMPLETED and not with_task else 1
            if control == "cancel_failure" and (
                (kind == event_type and kind != EventType.SESSION_COMPLETED)
                or (kind == EventType.TURN_COMPLETED and boundary == "task")
            ):
                expected = 0
            assert sum(event.type == kind for event in events) == expected
        types = [event.type for event in events]
        if with_task and control != "cancel_failure":
            assert types.index(EventType.TASK_COMPLETED) < types.index(EventType.TURN_COMPLETED)
        if EventType.TURN_COMPLETED in types:
            assert types.index(EventType.TURN_COMPLETED) < types.index(EventType.SESSION_COMPLETED)
        if control.startswith("finish") or control == "cancel_hook":
            completion_types = {
                EventType.TASK_COMPLETED,
                EventType.TURN_COMPLETED,
                EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
                EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
                EventType.SESSION_COMPLETED,
            }
            assert [e.type for e in observed if e.type in completion_types] == [
                *([EventType.TASK_COMPLETED] if with_task else []),
                EventType.TURN_COMPLETED,
                *(
                    [
                        EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
                        EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
                    ]
                    if critical
                    else []
                ),
                EventType.SESSION_COMPLETED,
            ]
        predecessor = next(e for e in events if e.type == EventType.INTERACTION_COMPLETED)
        assert (
            store._session_operation_records[final.id][
                session_module._interaction_transition_storage_key(predecessor.id)
            ]
            == receipt_before
        )
        assert len(provider.requests) == 1
        completed_task = await task_store.load_task("completion-task")
        if with_task:
            assert completed_task is not None and completed_task.status is TaskStatus.COMPLETED
        if critical:
            assert (source / "result.txt").read_text() == "committed output"
            for kind in (
                EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
                EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
            ):
                assert sum(e.type == kind for e in events) == 1

    asyncio.run(run())


@pytest.mark.parametrize("rejection", ["stale", "expired"])
def test_queued_exposure_transfer_requires_exact_proof(
    tmp_path, monkeypatch, rejection, capsys, caplog
) -> None:
    original = EnvironmentLifecycle.accept_queued_interaction
    checked = 0

    def inspect_transfer(lifecycle, **kwargs):
        nonlocal checked
        context = kwargs["invocation_context"]
        session = kwargs["session"]
        batch = kwargs["batch"]
        registered = context.registered_environment
        assert registered is not None
        exposure = registered.environment_exposure
        assert exposure is not None
        admission = exposure.admission
        owner = lifecycle._active_environment_setups[session.id]
        assert owner.registered_environment is registered
        original_interaction = exposure.admission.interaction_id
        preserved = (admission.decision, admission.renewal_lock, admission.settlement_task)

        invalid_arguments = [
            {"session": session.model_copy(update={"instance_id": "wrong-instance"})},
            {"session": session.model_copy(update={"run_epoch": session.run_epoch + 1})},
            {"session": session.model_copy(update={"run_epoch": True})},
            {"batch": batch.model_copy(update={"messages": ()})},
            {"batch": batch.model_copy(update={"events": ()})},
            {"batch": batch.model_copy(update={"delivery_id": "wrong-delivery"})},
            {"batch": batch.model_copy(update={"interaction_id": "wrong-interaction"})},
            {
                "batch": batch.model_copy(
                    update={"active_invocation_profile": context.active_profile}
                )
            },
            {
                "predecessor_settlement_event": kwargs["predecessor_settlement_event"].model_copy(
                    update={"id": "wrong-predecessor"}
                )
            },
            {
                "interaction_started_event": kwargs["interaction_started_event"].model_copy(
                    update={"id": "wrong-start"}
                )
            },
            {
                "profile_handoff": kwargs["profile_handoff"].model_copy(
                    update={"expected_session_instance_id": "wrong-instance"}
                )
            },
            {
                "profile_handoff": kwargs["profile_handoff"].model_copy(
                    update={"expected_active_profile": batch.active_invocation_profile}
                )
            },
        ]
        for invalid in invalid_arguments:
            with pytest.raises((TypeError, ValueError, RuntimeError)):
                original(lifecycle, **{**kwargs, **invalid})
            assert exposure.admission.interaction_id == original_interaction
            assert registered.environment_exposure is exposure
            assert exposure.admission is admission
            checked += 1

        class SecretCanary:
            def __repr__(self):
                return "queued-exposure-secret-canary"

            __str__ = __repr__

        for field in ("tool_name", "payload"):
            malformed = copy_event(kwargs["interaction_started_event"])
            if field == "payload":
                malformed.payload["invalid"] = SecretCanary()
            else:
                setattr(malformed, field, SecretCanary())
            with warnings.catch_warnings(record=True) as emitted_warnings:
                warnings.simplefilter("always")
                with pytest.raises((TypeError, ValueError)) as failure:
                    original(
                        lifecycle,
                        **{
                            **kwargs,
                            "batch": batch.model_copy(update={"events": (malformed,)}),
                        },
                    )
            captured = capsys.readouterr()
            assert "queued-exposure-secret-canary" not in (
                captured.out
                + captured.err
                + caplog.text
                + str(failure.value)
                + "".join(str(w.message) for w in emitted_warnings)
            )
            assert not emitted_warnings
            assert exposure.admission.interaction_id == original_interaction
            checked += 1

        # Post-construction conflicting live authority cannot be authenticated
        # merely because the delivery receipt itself is genuine.
        for field, invalid in (
            ("session_instance_id", "wrong-instance"),
            ("run_epoch", session.run_epoch + 1),
            ("execution_profile", context.profile.model_copy(deep=True)),
            ("environment", object()),
            ("runner", object()),
            ("binding_generation_id", "wrong-generation"),
        ):
            before = getattr(exposure, field)
            try:
                object.__setattr__(exposure, field, invalid)
                with pytest.raises(RuntimeError, match="runtime-admitted exposure"):
                    original(lifecycle, **kwargs)
                assert exposure.admission.interaction_id == original_interaction
            finally:
                object.__setattr__(exposure, field, before)
            checked += 1

        successor = original(lifecycle, **kwargs)
        assert successor.registered_environment is registered
        assert owner.registered_environment is registered
        assert registered.environment_exposure is exposure
        assert exposure.admission is admission
        assert admission.decision is preserved[0]
        assert admission.renewal_lock is preserved[1]
        assert admission.settlement_task is preserved[2]
        assert exposure.admission.interaction_id == successor.binding.interaction_id
        require_environment_exposed(
            registered,
            session=session,
            invocation_context=successor,
            registered_agent=context.registered_agent,
            execution_profile=context.profile,
        )
        with pytest.raises(RuntimeError, match="runtime-admitted exposure"):
            require_environment_exposed(
                registered,
                session=session,
                invocation_context=context,
                registered_agent=context.registered_agent,
                execution_profile=context.profile,
            )
        with pytest.raises(RuntimeError, match="runtime-admitted exposure"):
            original(lifecycle, **kwargs)
        return successor

    monkeypatch.setattr(EnvironmentLifecycle, "accept_queued_interaction", inspect_transfer)
    test_terminal_only_queue_uses_existing_session_owner(
        tmp_path, critical=True, mixed=True, ceiling=False, rejection=rejection
    )
    assert checked == 20


@pytest.mark.parametrize("cancel", [False, True])
def test_queued_exposure_preserves_pending_admission_settlement(
    tmp_path, monkeypatch, cancel: bool
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = BlockingTwoTurnProvider()
        second_started = asyncio.Event()
        release_second = asyncio.Event()
        original_stream = provider.stream

        async def stream_with_second_boundary(request):
            if provider.requests:
                second_started.set()
                await release_second.wait()
            async for event in original_stream(request):
                yield event

        provider.stream = stream_with_second_boundary
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        source, target = tmp_path / "source", tmp_path / "target"
        source.mkdir()
        target.mkdir()
        app.register_environment(
            Environment(
                EnvironmentSpec(name="sync"),
                workspace=LocalWorkspace(source, workspace_id="source"),
                binding=SyncBinding(target_workspace=LocalWorkspace(target, workspace_id="target")),
            ),
            default=True,
        )
        release = asyncio.Event()
        waiting = asyncio.Event()
        settlement = None
        captured_owner = None
        captured_admission = None
        original_transfer = EnvironmentLifecycle.accept_queued_interaction
        original_wait = exposure_module._await_exposure_admission_settlement

        async def settle_dispatched_work():
            await release.wait()

        def transfer(lifecycle, **kwargs):
            nonlocal settlement, captured_owner, captured_admission
            context = kwargs["invocation_context"]
            registered = context.registered_environment
            assert registered is not None and registered.environment_exposure is not None
            exposure = registered.environment_exposure
            captured_admission = exposure.admission
            captured_owner = lifecycle._active_environment_setups[kwargs["session"].id]
            settlement = asyncio.create_task(settle_dispatched_work())
            exposure.admission.settlement_task = settlement
            successor = original_transfer(lifecycle, **kwargs)
            assert successor.registered_environment is registered
            assert registered.environment_exposure is exposure
            assert exposure.admission is captured_admission
            assert exposure.admission.settlement_task is settlement
            return successor

        async def observe_settlement(exposure):
            if settlement is not None and exposure.admission.settlement_task is settlement:
                waiting.set()
            await original_wait(exposure)

        monkeypatch.setattr(EnvironmentLifecycle, "accept_queued_interaction", transfer)
        monkeypatch.setattr(
            exposure_module, "_await_exposure_admission_settlement", observe_settlement
        )

        async def execute():
            async with contextlib.aclosing(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="terminal-queue",
                        messages=[Message.text("user", "initial")],
                        max_steps=3,
                    )
                )
            ) as stream:
                return [event async for event in stream]

        owner = asyncio.create_task(execute())
        try:
            await asyncio.wait_for(provider.first_started.wait(), 10)
            for expired in (True, False):
                await app.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id="terminal-queue",
                        idempotency_key="expired" if expired else "valid",
                        content="rejected" if expired else "next interaction",
                        delivery_mode="on_idle",
                        conditions=SessionMessageConditions(
                            expires_at=datetime.now(UTC) - timedelta(seconds=1)
                        )
                        if expired
                        else SessionMessageConditions(),
                    )
                )
            provider.release_first.set()
            await asyncio.wait_for(second_started.wait(), 10)
            # The accepted turn also ends through rejected-only completion, so
            # cancellation exercises that completion owner with the retained
            # admission settlement inherited from its queued handoff.
            await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id="terminal-queue",
                    idempotency_key="expired-successor",
                    content="must not start another interaction",
                    delivery_mode="on_idle",
                    conditions=SessionMessageConditions(
                        expires_at=datetime.now(UTC) - timedelta(seconds=1)
                    ),
                )
            )
            release_second.set()
            await asyncio.wait_for(waiting.wait(), 10)
            # Fresh admission still permits the valid successor's dispatch.
            # Its terminal cleanup must retain and settle this exact owner.
            assert len(provider.requests) == 2
            assert captured_owner is not None and not captured_owner.cleanup_started
            assert settlement is not None and not settlement.done()
            current = await store.load("terminal-queue")
            profile = active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint("terminal-queue")
            )
            assert current is not None and profile is not None
            assert current.run_epoch == profile.run_epoch
            if cancel:
                owner.cancel("queued admission cancelled")
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not owner.done()
                assert not settlement.done()
                assert not captured_owner.cleanup_started
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError) as failure:
                    await asyncio.wait_for(owner, 20)
                assert failure.value.args == ("queued admission cancelled",)
                assert owner.cancelled() and owner.cancelling() == 1
            else:
                await asyncio.wait_for(owner, 20)
            assert settlement.done() and not settlement.cancelled()
            assert captured_admission is not None
            assert captured_admission.settlement_task is None
            assert len(provider.requests) == 2
        finally:
            release.set()
            release_second.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if settlement is not None:
                await settlement
        final = await store.load("terminal-queue")
        profile = active_invocation_execution_profile_from_checkpoint(
            await store.load_checkpoint("terminal-queue")
        )
        assert final is not None and profile is not None
        assert final.run_epoch == profile.run_epoch + 1

    asyncio.run(run())


class _QueueOwnerPolicy(SessionMessageAccessPolicy):
    def authorize(
        self,
        context: SessionMessageAccessContext,
        *,
        session_id: str,
        session_instance_id: str,
        action: Literal["inspect", "enqueue", "source", "withdraw", "quarantine"],
    ) -> bool:
        return context.subject == "queue-owner" and session_id == "terminal-queue"


class _ReceiptCheckingStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1
    receipt_checks = 0

    async def transition_status_if_no_queued_messages(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_mutation: dict[str, Any] | None = None,
    ) -> Session:
        before = await self.load(session_id)
        assert before is not None
        profile = active_invocation_execution_profile_from_checkpoint(
            await self.load_checkpoint(session_id)
        )
        assert profile is not None
        transition = await self.load_invocation_settlement_transition(
            session_id,
            expected_session_instance_id=before.instance_id,
            expected_active_invocation_profile=profile,
        )
        assert transition is not None
        receipt = await self.load_interaction_transition_receipt(session_id, transition=transition)
        assert receipt is not None and receipt.status_changed is False
        result = await super().transition_status_if_no_queued_messages(
            session_id,
            from_statuses=from_statuses,
            to_status=to_status,
            checkpoint_mutation=checkpoint_mutation,
        )
        replay = await self.load_interaction_transition_receipt(session_id, transition=transition)
        assert replay == receipt, "session-only completion must not rewrite predecessor evidence"
        assert result.run_epoch == before.run_epoch
        assert (
            active_invocation_execution_profile_from_checkpoint(
                await self.load_checkpoint(session_id)
            )
            == profile
        )
        self.receipt_checks += 1
        return result


@pytest.mark.parametrize("critical", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("ceiling", [False, True])
@pytest.mark.parametrize("rejection", ["stale", "expired"])
def test_terminal_only_queue_uses_existing_session_owner(
    tmp_path, critical: bool, mixed: bool, ceiling: bool, rejection: str
) -> None:
    async def run() -> None:
        store = _ReceiptCheckingStore()
        provider = BlockingTwoTurnProvider()
        app = CayuApp(
            session_store=store,
            session_message_access_policy=_QueueOwnerPolicy(),
            enable_logging=False,
        )
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        source = tmp_path / "source"
        target = tmp_path / "target"
        if critical:
            source.mkdir()
            target.mkdir()
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="sync"),
                    workspace=LocalWorkspace(source, workspace_id="queue-source"),
                    binding=SyncBinding(
                        target_workspace=LocalWorkspace(target, workspace_id="queue-target")
                    ),
                ),
                default=True,
            )
        events = []

        async def execute() -> None:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="terminal-queue",
                    messages=[Message.text("user", "initial")],
                    max_steps=1 if ceiling else 3,
                )
            ):
                events.append(event)

        task = asyncio.create_task(execute())
        try:
            await asyncio.wait_for(provider.first_started.wait(), 10)
            session = await store.load("terminal-queue")
            assert session is not None
            original_profile = active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint(session.id)
            )
            assert original_profile is not None
            if critical:
                # A real bound workspace mutation must be published by the
                # existing completion-critical tail, never by this test.
                (target / "result.txt").write_text("committed output")
            conditions = (
                SessionMessageConditions(
                    target=SessionMessageTarget(
                        session_instance_id=session.instance_id,
                        run_epoch=session.run_epoch,
                        transcript_cursor=0,
                    )
                )
                if rejection == "stale"
                else SessionMessageConditions(expires_at=datetime.now(UTC) - timedelta(seconds=1))
            )
            await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session.id,
                    idempotency_key="rejected-input",
                    delivery_mode="on_idle",
                    content="must never reach provider",
                    conditions=conditions,
                ),
                context=SessionMessageAccessContext(subject="queue-owner"),
            )
            if mixed:
                await app.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id=session.id,
                        idempotency_key="valid-input",
                        delivery_mode="on_idle",
                        content="valid steering",
                    ),
                    context=SessionMessageAccessContext(subject="queue-owner"),
                )
            provider.release_first.set()
            await asyncio.wait_for(task, 20)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        stopped = mixed and ceiling
        delivered = mixed and not ceiling
        assert len(provider.requests) == (2 if delivered else 1), [
            event.payload for event in events if event.type == EventType.SESSION_FAILED
        ]
        stored = await store.load_events(session.id)
        assert sum(event.type == EventType.INTERACTION_STARTED for event in stored) == (
            2 if delivered else 1
        )
        rejection_type = (
            EventType.SESSION_MESSAGE_STALE
            if rejection == "stale"
            else EventType.SESSION_MESSAGE_EXPIRED
        )
        assert sum(event.type == rejection_type for event in stored) == 1, [
            event.payload for event in events if event.type == EventType.SESSION_FAILED
        ]
        assert sum(event.type == EventType.SESSION_MESSAGE_DELIVERED for event in stored) == (
            1 if delivered else 0
        )
        assert any(event.type == rejection_type for event in events)
        transcript = await store.load_transcript(session.id)
        texts = [
            part.text for message in transcript for part in message.content if hasattr(part, "text")
        ]
        assert "must never reach provider" not in texts
        assert ("valid steering" in texts) is delivered
        inspection = await app.inspect_session_messages(
            SessionMessageQuery(session_id=session.id),
            context=SessionMessageAccessContext(subject="queue-owner"),
        )
        assert [record.status for record in inspection.records] == [
            rejection,
            *(["queued" if ceiling else "delivered"] if mixed else []),
        ]
        final = await store.load(session.id)
        assert final is not None
        assert final.status is (SessionStatus.INTERRUPTED if stopped else SessionStatus.COMPLETED)
        checkpoint = await store.load_checkpoint(session.id)
        profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert profile is not None
        assert final.run_epoch == profile.run_epoch + 1, "terminal cleanup must release the fence"
        assert profile.profile == original_profile.profile
        if not delivered:
            assert profile.interaction_id == original_profile.interaction_id
        if not mixed:
            assert store.receipt_checks == 1
        if critical and not stopped:
            assert (source / "result.txt").read_text() == "committed output"
            assert PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY not in (checkpoint or {})
            assert next(
                i
                for i, event in enumerate(stored)
                if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
            ) < next(
                i for i, event in enumerate(stored) if event.type == EventType.SESSION_COMPLETED
            )

    asyncio.run(run())


@pytest.mark.parametrize("takeover", [False, True])
def test_rejection_yield_rechecks_queue_and_epoch(takeover: bool) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = BlockingTwoTurnProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        rejected = asyncio.Event()
        resume = asyncio.Event()
        events = []

        async def execute() -> None:
            async with contextlib.aclosing(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="terminal-queue",
                        messages=[Message.text("user", "initial")],
                    )
                )
            ) as stream:
                async for event in stream:
                    events.append(event)
                    if event.type == EventType.SESSION_MESSAGE_EXPIRED:
                        rejected.set()
                        await resume.wait()

        task = asyncio.create_task(execute())
        try:
            await asyncio.wait_for(provider.first_started.wait(), 10)
            await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id="terminal-queue",
                    idempotency_key="expired-before-yield",
                    content="expired",
                    delivery_mode="on_idle",
                    conditions=SessionMessageConditions(
                        expires_at=datetime.now(UTC) - timedelta(seconds=1)
                    ),
                )
            )
            provider.release_first.set()
            await asyncio.wait_for(rejected.wait(), 10)
            if takeover:
                successor = await store.fence_run_and_transform_checkpoint(
                    "terminal-queue",
                    statuses={SessionStatus.RUNNING},
                    checkpoint_transform=lambda session, checkpoint: checkpoint,
                )
                successor_checkpoint = await store.load_checkpoint(successor.id)
            else:
                await app.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id="terminal-queue",
                        idempotency_key="arrived-at-yield",
                        content="new input",
                        delivery_mode="on_idle",
                    )
                )
            resume.set()
            try:
                await asyncio.wait_for(task, 20)
            except session_module.SessionRunFenced:
                assert takeover
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        final = await store.load("terminal-queue")
        assert final is not None
        if takeover:
            assert len(provider.requests) == 1
            assert final == successor
            assert await store.load_checkpoint(final.id) == successor_checkpoint
            assert not any(event.type == EventType.SESSION_COMPLETED for event in events)
        else:
            assert len(provider.requests) == 2
            assert final.status is SessionStatus.COMPLETED
            assert sum(event.type == EventType.SESSION_MESSAGE_DELIVERED for event in events) == 1

    asyncio.run(run())


def test_running_handoff_requires_exact_finalization_mutation(tmp_path, monkeypatch) -> None:
    original = session_module._checkpoint_after_queued_interaction_profile_handoff
    checked = 0

    def check_handoff(session, checkpoint, handoff, **kwargs):
        nonlocal checked
        record = kwargs["settlement_record"]
        if record["to_status"] == "running":
            before = deepcopy(checkpoint)
            mutation = record["checkpoint_mutation"]
            invalid_mutations = []
            for field, value in (
                ("key", "unrelated_checkpoint"),
                ("action", "delete"),
                ("expected_value_digest", "0" * 64),
                ("value", "not-a-marker"),
            ):
                invalid = deepcopy(mutation)
                invalid["operations"][0][field] = value
                invalid_mutations.append(invalid)
            invalid_mutations.append({"operations": []})
            for invalid in invalid_mutations:
                conflicting = deepcopy(record)
                conflicting["checkpoint_mutation"] = invalid
                digest_payload = deepcopy(conflicting)
                digest_payload.pop("record_digest")
                # Match the v5 receipt's omission rules, so rejection proves
                # transition-shape validation rather than a digest mismatch.
                for field in (
                    "model_completion_stage_settlement",
                    "recovery_claim_id",
                    "terminal_event",
                    "terminal_decision",
                ):
                    if digest_payload.get(field) is None:
                        digest_payload.pop(field, None)
                conflicting["record_digest"] = session_module._canonical_runtime_publication_digest(
                    digest_payload
                )
                session_module._load_interaction_transition_receipt(conflicting)
                with pytest.raises(
                    session_module.SessionRunFenced, match="completion-finalization"
                ):
                    original(
                        session,
                        checkpoint,
                        handoff,
                        **{**kwargs, "settlement_record": conflicting},
                    )
                assert checkpoint == before
                checked += 1
        return original(session, checkpoint, handoff, **kwargs)

    monkeypatch.setattr(
        session_module, "_checkpoint_after_queued_interaction_profile_handoff", check_handoff
    )
    # Capture real runtime-produced receipt/profile authority at the actual
    # delivery boundary, then allow the authentic mutation to complete normally.
    test_terminal_only_queue_uses_existing_session_owner(
        tmp_path, critical=True, mixed=True, ceiling=False, rejection="stale"
    )
    assert checked == 5


class _RejectionAcknowledgementStore(_ReceiptCheckingStore):
    invocation_lifecycle_command_version = 1
    queued_interaction_profile_handoff_version = 1
    session_message_lifecycle_version = 1

    def __init__(self, lose_ack: bool) -> None:
        super().__init__()
        self.lose_ack = lose_ack
        self.lost_delivery_id: str | None = None
        self.replayed_delivery_id: str | None = None

    async def deliver_queued_session_messages(self, session_id: str, **kwargs):
        batch = await super().deliver_queued_session_messages(session_id, **kwargs)
        if self.lose_ack and batch.events and not batch.messages:
            self.lose_ack = False
            self.lost_delivery_id = batch.delivery_id
            raise ConnectionError("rejection acknowledgement lost")
        if batch.replayed:
            self.replayed_delivery_id = batch.delivery_id
        return batch


@pytest.mark.parametrize("case", ["cross_mode", "many", "ceiling"])
@pytest.mark.parametrize("lose_ack", [False, True])
def test_rejection_batches_keep_first_delivery_identity(case: str, lose_ack: bool) -> None:
    async def run() -> None:
        store = _RejectionAcknowledgementStore(lose_ack)
        provider = BlockingTwoTurnProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def execute() -> list:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="terminal-queue",
                        messages=[Message.text("user", "initial")],
                        max_steps=1 if case == "ceiling" else 3,
                    )
                )
            ]

        task = asyncio.create_task(execute())
        try:
            await asyncio.wait_for(provider.first_started.wait(), 10)
            count = 1 if case == "cross_mode" else 101
            for index in range(count):
                await app.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id="terminal-queue",
                        idempotency_key=f"expired-{index}",
                        content="expired input",
                        delivery_mode="next_turn",
                        conditions=SessionMessageConditions(
                            expires_at=datetime.now(UTC) - timedelta(seconds=1)
                        ),
                    )
                )
            if case == "cross_mode":
                await app.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id="terminal-queue",
                        idempotency_key="valid-idle",
                        content="valid idle",
                        delivery_mode="on_idle",
                    )
                )
            provider.release_first.set()
            events = await asyncio.wait_for(task, 30)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert sum(event.type == EventType.SESSION_MESSAGE_EXPIRED for event in events) == count
        assert len(provider.requests) == (2 if case == "cross_mode" else 1)
        assert sum(event.type == EventType.INTERACTION_STARTED for event in events) == len(
            provider.requests
        )
        final = await store.load("terminal-queue")
        assert final is not None and final.status is SessionStatus.COMPLETED
        profile = active_invocation_execution_profile_from_checkpoint(
            await store.load_checkpoint(final.id)
        )
        assert profile is not None and final.run_epoch == profile.run_epoch + 1
        if lose_ack:
            assert store.lost_delivery_id is not None
            assert store.replayed_delivery_id == store.lost_delivery_id

    asyncio.run(run())


class _PauseCompletionReadStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.pause_next_read = False
        self.read_started = asyncio.Event()

    async def load(self, session_id: str) -> Session | None:
        if self.pause_next_read:
            self.pause_next_read = False
            self.read_started.set()
            await asyncio.Event().wait()
        return await super().load(session_id)


@pytest.mark.parametrize("critical", [False, True])
@pytest.mark.parametrize("control", ["close", "cancel"])
def test_rejection_yield_does_not_commit_or_strand_session(
    tmp_path, critical: bool, control: str
) -> None:
    async def run() -> None:
        store = _PauseCompletionReadStore()
        provider = BlockingTwoTurnProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        if critical:
            source, target = tmp_path / "source", tmp_path / "target"
            source.mkdir()
            target.mkdir()
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="sync"),
                    workspace=LocalWorkspace(source, workspace_id="source"),
                    binding=SyncBinding(
                        target_workspace=LocalWorkspace(target, workspace_id="target")
                    ),
                ),
                default=True,
            )
        caught_cancellation = False
        rejected = False

        async def execute() -> None:
            nonlocal caught_cancellation, rejected
            try:
                async with contextlib.aclosing(
                    app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id="terminal-queue",
                            messages=[Message.text("user", "initial")],
                        )
                    )
                ) as stream:
                    async for event in stream:
                        if event.type == EventType.SESSION_MESSAGE_EXPIRED:
                            rejected = True
                            observed = await store.load("terminal-queue")
                            assert observed is not None and observed.status is SessionStatus.RUNNING
                            assert PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY not in (
                                await store.load_checkpoint(observed.id) or {}
                            )
                            if control == "close":
                                break
                            store.pause_next_read = True
            except asyncio.CancelledError:
                caught_cancellation = True
                raise

        task = asyncio.create_task(execute())
        try:
            await asyncio.wait_for(provider.first_started.wait(), 10)
            await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id="terminal-queue",
                    idempotency_key="expired",
                    content="expired",
                    delivery_mode="on_idle",
                    conditions=SessionMessageConditions(
                        expires_at=datetime.now(UTC) - timedelta(seconds=1)
                    ),
                )
            )
            provider.release_first.set()
            if control == "cancel":
                await asyncio.wait_for(store.read_started.wait(), 10)
                task.cancel()
                assert task.cancelling() == 1
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled() and caught_cancellation
                assert task.cancelling() == 1
            else:
                await asyncio.wait_for(task, 20)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert rejected
        assert len(provider.requests) == 1
        final = await store.load("terminal-queue")
        assert final is not None and final.status is SessionStatus.RUNNING
        profile = active_invocation_execution_profile_from_checkpoint(
            await store.load_checkpoint(final.id)
        )
        assert profile is not None and final.run_epoch == profile.run_epoch + 1
        assert not any(
            event.type == EventType.SESSION_COMPLETED for event in await store.load_events(final.id)
        )

    asyncio.run(run())


def test_unconditioned_custom_store_keeps_ceiling_stop() -> None:
    async def run() -> None:
        store = CommitThenLoseDeliveryAcknowledgementStore()
        provider = BlockingTwoTurnProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def execute() -> list:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="terminal-queue",
                        messages=[Message.text("user", "initial")],
                        max_steps=1,
                    )
                )
            ]

        task = asyncio.create_task(execute())
        try:
            await asyncio.wait_for(provider.first_started.wait(), 10)
            initial_delivery_ids = list(store.attempted_delivery_ids)
            await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id="terminal-queue",
                    idempotency_key="valid",
                    content="keep queued",
                    delivery_mode="next_turn",
                )
            )
            provider.release_first.set()
            events = await asyncio.wait_for(task, 20)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert len(provider.requests) == 1
        assert store.attempted_delivery_ids == initial_delivery_ids
        assert any(event.type == EventType.SESSION_LIMIT_REACHED for event in events)
        final = await store.load("terminal-queue")
        assert final is not None and final.status is SessionStatus.INTERRUPTED
        profile = active_invocation_execution_profile_from_checkpoint(
            await store.load_checkpoint(final.id)
        )
        assert profile is not None and final.run_epoch == profile.run_epoch + 1

    asyncio.run(run())
