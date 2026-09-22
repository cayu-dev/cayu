from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.core.test_context_view_admission import _close, _factory
from tests.core.test_participant_continuation_boundaries import activate, collect
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_provider_operation_offline_recovery import (
    _OfflineOperationAdapter,
    _OfflineOperationProvider,
)

from cayu.agents import AgentSpec
from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration.lifecycle import ParticipantLifecycleChange
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.events import EventType
from cayu.providers.base import ModelStreamEvent
from cayu.providers.operations import ProviderOperationStatus
from cayu.runtime.authority import SessionRunFenced
from cayu.runtime.provider_operations import (
    ProviderOperationResolutionAction,
    ProviderOperationResolutionConflict,
    ProviderOperationResolutionRequest,
    inspect_provider_operation,
    load_pending_provider_operation_disposition,
)
from cayu.sessions.base import IncompleteSessionRecoveryRequest, Message, RunRequest, SessionStatus
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.tasks.base import InMemoryTaskStore, TaskCreate, interrupted_task_handoff_request


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "action,lost_ack,handoff",
    [
        ("fail", False, False),
        ("fail", True, False),
        ("fallback_retry", False, False),
        ("fallback_retry", True, False),
        ("fallback_retry", False, True),
    ],
)
def test_participant_provider_resolution_authority_and_replay(
    backend, action, lost_ack, handoff, tmp_path, request, monkeypatch
):
    factory = _factory(backend, tmp_path, request, [datetime.now(UTC)])

    async def run():
        store = factory()
        collaboration = InMemoryCollaborationStore()
        tasks = InMemoryTaskStore() if handoff else None
        entered = asyncio.Event()

        class Adapter(_OfflineOperationAdapter):
            async def start(self, request):
                if self.start_calls == 0:
                    self.start_calls += 1
                    entered.set()
                    # Actual cancellation after dispatch leaves an unknown start.
                    await asyncio.Event().wait()
                return await super().start(request)

        provider = _OfflineOperationProvider(ProviderOperationStatus.IN_PROGRESS)
        provider.adapter = Adapter(ProviderOperationStatus.IN_PROGRESS)
        provider.adapter.start_events = (
            ModelStreamEvent.text_delta("explicitly authorized fallback"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
        reg = registration()

        def configured_app():
            value = app(collaboration, reg, session_store=store, task_store=tasks)
            value.register_provider(provider, default=True)
            value.register_agent(AgentSpec(name="reviewer", model="model"))
            return value

        value = configured_app()
        try:
            initialized = await value.initialize_collaboration()
            _, created = await create(value, initialized)
            participant = created.participants[0].reference
            if handoff:
                await tasks.create_task(TaskCreate(task_id=str(uuid4()), type="job"))
                original_task = await tasks.claim_task("initial-worker", lease_seconds=300)
                creation = ParticipantSessionCreationRequest(
                    RunRequest(
                        agent_name="reviewer",
                        messages=[Message.text("user", "start")],
                        task_id=original_task.id,
                        task_worker_id="initial-worker",
                        task_lease_expires_at=original_task.lease_expires_at,
                    ),
                    str(uuid4()),
                )
                session, _ = await value.create_participant_session(
                    creation, participant=participant, context=CONTEXT
                )
                execution = ParticipantSessionExecutionRequest(
                    request=creation.request.model_copy(update={"session_id": session.id}),
                    session_instance_id=session.instance_id,
                    execution_key="execute",
                )
            else:
                session, execution = await activate(value, participant)
            task = asyncio.create_task(
                collect(
                    value.execute_participant_session(
                        execution, participant=participant, context=CONTEXT
                    )
                )
            )
            await asyncio.wait_for(entered.wait(), 20)
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            assert provider.adapter.start_calls == 1
            if backend != "memory":
                await store.close()
                store = factory()
            value = configured_app()
            await value.initialize_collaboration()
            current = await store.load(session.id)
            inspection = await inspect_provider_operation(store, session.id)
            assert current.status is SessionStatus.INTERRUPTED
            assert ProviderOperationResolutionAction(action) in inspection.allowed_resolutions
            intent = ProviderOperationResolutionRequest(
                session_id=session.id,
                stage_id=inspection.stage_id,
                expected_run_epoch=current.run_epoch,
                action=ProviderOperationResolutionAction(action),
                reason="operator verified",
            )
            if handoff:
                attached = await tasks.load_task(original_task.id)
                await tasks.release_interrupted_task_worker(
                    interrupted_task_handoff_request(
                        attached,
                        session_run_epoch=current.run_epoch,
                    )
                )
                elected = await tasks.claim_interrupted_task_continuation(
                    "continuation-worker", handoff_id=str(uuid4())
                )
                intent = intent.model_copy(
                    update={
                        "task_worker_id": elected.task.worker_id,
                        "task_handoff_id": elected.task.interrupted_handoff_id,
                    }
                )

            async def snapshot():
                return (
                    await store.load(session.id),
                    await store.load_checkpoint(session.id),
                    await store.load_events(session.id),
                    await store.load_active_model_completion_stage(session.id),
                )

            before = await snapshot()
            with pytest.raises(PermissionError, match="administration"):
                await collect(value.resolve_provider_operation(intent))
            await value.change_participant_lifecycle(
                ParticipantLifecycleChange(
                    operation=initialized.operation("disable"),
                    participant=participant,
                    expected_lifecycle_revision=1,
                    state="disabled",
                ),
                context=CONTEXT,
            )
            with pytest.raises(PermissionError, match="active participants"):
                await collect(value.resolve_provider_operation(intent, context=CONTEXT))
            assert await snapshot() == before
            assert provider.adapter.start_calls == 1
            await value.change_participant_lifecycle(
                ParticipantLifecycleChange(
                    operation=initialized.operation("reactivate"),
                    participant=participant,
                    expected_lifecycle_revision=2,
                    state="active",
                ),
                context=CONTEXT,
            )

            if lost_ack:
                publish = store.publish_session_operation
                committed = False

                async def lose_ack(*args, **kwargs):
                    nonlocal committed
                    result = await publish(*args, **kwargs)
                    if not committed and any(
                        e.type is EventType.PROVIDER_OPERATION_RESOLVED
                        for e in kwargs.get("events", ())
                    ):
                        committed = True
                        raise ConnectionError("resolution acknowledgement lost")
                    return result

                monkeypatch.setattr(store, "publish_session_operation", lose_ack)
                with pytest.raises(ConnectionError, match="acknowledgement lost"):
                    await collect(value.resolve_provider_operation(intent, context=CONTEXT))
                assert committed
                assert provider.adapter.start_calls == 1
                assert (
                    await load_pending_provider_operation_disposition(store, session.id) is not None
                )
                monkeypatch.setattr(store, "publish_session_operation", publish)
                if backend != "memory":
                    await store.close()
                    store = factory()
                value = configured_app()
                await value.initialize_collaboration()
                # Generic recovery cannot borrow the earlier caller's authority.
                before_retry = await snapshot()
                with pytest.raises(PermissionError, match="administration"):
                    await value.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(
                            session_id=session.id,
                            inactive_for_seconds=0,
                        )
                    )
                assert await snapshot() == before_retry

            if handoff:
                stream = value.resolve_provider_operation(intent, context=CONTEXT)
                assert (await anext(stream)).type is EventType.PROVIDER_OPERATION_RESOLVED
                assert (await anext(stream)).type is EventType.INTERACTION_RESUMED
                # This attempt follows a cancelled, terminal invocation. Closing
                # before the replacement dispatch retains its unresolved fence;
                # cleanup must report that instead of silently releasing it.
                with pytest.raises(SessionRunFenced, match="exact durable terminal settlement"):
                    await stream.aclose()
                assert provider.adapter.start_calls == 1
                running = await store.load(session.id)
                assert running.status is SessionStatus.RUNNING
                assert (
                    await load_pending_provider_operation_disposition(store, session.id) is not None
                )
                attached = await tasks.load_task(original_task.id)
                await tasks.release_interrupted_task_worker(
                    interrupted_task_handoff_request(
                        attached,
                        session_run_epoch=running.run_epoch,
                    )
                )
                successor = await tasks.claim_interrupted_task_continuation(
                    "continuation-worker", handoff_id=str(uuid4())
                )
                assert successor.task.interrupted_handoff_id != intent.task_handoff_id
                intent = intent.model_copy(
                    update={"task_handoff_id": successor.task.interrupted_handoff_id}
                )
            events = await collect(value.resolve_provider_operation(intent, context=CONTEXT))
            expected_status = SessionStatus.FAILED if action == "fail" else SessionStatus.COMPLETED
            assert (await store.load(session.id)).status is expected_status
            assert any(
                e.type
                == (EventType.SESSION_FAILED if action == "fail" else EventType.SESSION_COMPLETED)
                for e in events
            )
            assert provider.adapter.start_calls == (1 if action == "fail" else 2)
            assert await load_pending_provider_operation_disposition(store, session.id) is None
            durable = await store.load_events(session.id)
            assert sum(e.type is EventType.PROVIDER_OPERATION_RESOLVED for e in durable) == 1
            if handoff:
                # The task-handoff path has its own terminal task replay protocol.
                # The ordinary cases below qualify exact provider receipt replay.
                return
            if backend != "memory":
                await store.close()
                store = factory()
            value = configured_app()
            await value.initialize_collaboration()
            replay = await collect(value.resolve_provider_operation(intent, context=CONTEXT))
            assert [e.type for e in replay] == [EventType.PROVIDER_OPERATION_RESOLVED]
            assert await store.load_events(session.id) == durable
            with pytest.raises(ProviderOperationResolutionConflict):
                await collect(
                    value.resolve_provider_operation(
                        intent.model_copy(update={"reason": "changed"}), context=CONTEXT
                    )
                )
            with pytest.raises(PermissionError, match="administration"):
                await collect(value.resolve_provider_operation(intent))
            # Cancellation retained the execution permit. Provider receipt replay
            # does not prove that this separate obligation has been settled.
            with pytest.raises(CollaborationConflict):
                await value.change_participant_lifecycle(
                    ParticipantLifecycleChange(
                        operation=initialized.operation("retire"),
                        participant=participant,
                        expected_lifecycle_revision=3,
                        state="retired",
                    ),
                    context=CONTEXT,
                )
            await value.change_participant_lifecycle(
                ParticipantLifecycleChange(
                    operation=initialized.operation("disable-after-recovery"),
                    participant=participant,
                    expected_lifecycle_revision=3,
                    state="disabled",
                ),
                context=CONTEXT,
            )
            with pytest.raises(PermissionError, match="active participants"):
                await collect(value.resolve_provider_operation(intent, context=CONTEXT))
            assert await store.load_events(session.id) == durable
            assert provider.adapter.start_calls == (1 if action == "fail" else 2)
        finally:
            await _close(store)
            await collaboration.close()

    asyncio.run(run())
