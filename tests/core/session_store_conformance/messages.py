"""Shared session-store messages contract cases."""

from __future__ import annotations

import asyncio
import contextvars
import copy
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from tests.core._execution_profile_fixtures import (
    versioned_test_provider_identity,
)
from tests.core.session_store_conformance.support import (
    _close_store,
    _identity,
    _open_store,
    _reopen_store,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
)
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.attachments import FileAttachment, FileAttachmentKind
from cayu.events import (
    Event,
    EventType,
)
from cayu.messages import (
    FilePart,
    Message,
    MessageRole,
)
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime._child_session_notifications import (
    CHILD_SESSION_NOTIFICATION_INTENT_KEY,
    ChildSessionLifecycleQuery,
    ChildSessionNotificationFreshness,
    child_session_notification_storage_key,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._interaction_transitions import (
    QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY,
    InteractionTransitionSpec,
    QueuedInteractionProfileHandoff,
    queued_interaction_profile_handoff_evidence,
)
from cayu.sessions._model_stage import (
    ModelCompletionStageRequest,
    SessionModelCompletionStageConflict,
)
from cayu.sessions._operation_records import (
    SessionOperationPublication,
)
from cayu.sessions.child_context import ChildSessionContextContributor
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectStatus,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.messaging import (
    EnqueueSessionMessageRequest,
    SessionMessageDeliveryMode,
    SessionMessageQueueStatus,
    SessionQueuedMessagesPending,
)
from cayu.sessions.records import Session, SessionStatus, SessionStatusConflict
from cayu.sessions.requests import (
    RunRequest,
)
from cayu.sessions.transcript_queries import TranscriptQuery

pytestmark = pytest.mark.postgres


def test_session_store_conformance_dispatch_consumes_child_terminal_notification(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            parent_id = "child-notification-parent"
            child_id = "child-notification-child"
            await store.create(
                RunRequest(agent_name="assistant", session_id=parent_id, messages=[]),
                identity=_identity(),
            )
            parent = await store.transition_status(
                parent_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=child_id,
                    parent_session_id=parent_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.transition_status(
                child_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.append_event(
                child_id,
                Event(
                    id="child-notification-started",
                    type=EventType.SESSION_STARTED,
                    session_id=child_id,
                ),
            )
            expected_child_result = Message.text(
                MessageRole.ASSISTANT,
                "latest bounded child result",
            )
            await store.append_transcript_messages(
                child_id,
                [
                    Message.text(MessageRole.ASSISTANT, "older child result"),
                    expected_child_result,
                    Message.text(MessageRole.USER, "trailing child input"),
                ],
            )
            latest_child_result = await store.load_latest_transcript_message(
                child_id,
                role=MessageRole.ASSISTANT,
            )
            assert latest_child_result is not None
            assert latest_child_result.index == 1
            assert latest_child_result.message == expected_child_result
            assert await store.load_latest_transcript_text(
                child_id,
                role=MessageRole.ASSISTANT,
                max_chars=6,
            ) == ("latest", True)
            assert await store.load_latest_transcript_text(
                child_id,
                role=MessageRole.ASSISTANT,
                max_chars=len("latest bounded child result"),
            ) == ("latest bounded child result", False)
            await store.append_event(
                child_id,
                Event(
                    id="child-notification-terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id=child_id,
                ),
            )
            await store.transition_status(
                child_id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
            )
            contributor = ChildSessionContextContributor()
            contribution = await contributor.build(session_store=store, session=parent)
            assert contribution.stage_binding is not None
            consumption_key = child_session_notification_storage_key(
                contribution.stage_binding.claims[0].child_session_instance_id,
                "child-notification-terminal",
            )
            with pytest.raises(ValueError, match="reserved child-notification namespace"):
                await store.load_session_operation(parent_id, consumption_key)
            with pytest.raises(ValueError, match="reserved child-notification namespace"):
                await store.publish_session_operation(
                    parent_id,
                    idempotency_key="caller-owned-child-notification-collision",
                    operation_transform=lambda _session, checkpoint, _record: (
                        SessionOperationPublication(
                            checkpoint={} if checkpoint is None else checkpoint,
                            operation_records={consumption_key: {"owner": "caller"}},
                        )
                    ),
                    events=[],
                )
            prepared = await store.prepare_model_completion_stage(
                parent_id,
                request=ModelCompletionStageRequest(
                    stage_id="mstep_child_notification:dispatch:0",
                    logical_step_id="mstep_child_notification",
                    dispatch_ordinal=0,
                    intent={
                        "interaction_id": "interaction-child-notification",
                        "model_step_id": "mstep_child_notification",
                        "model_attempt_id": "matt_child_notification",
                        "request_fingerprint": "c" * 64,
                        CHILD_SESSION_NOTIFICATION_INTENT_KEY: (
                            contribution.stage_binding.model_dump(mode="json")
                        ),
                    },
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=parent.run_epoch,
                expected_transcript_cursor=0,
            )

            await store.mark_model_completion_stage_dispatched(
                parent_id,
                stage=prepared.stage,
                consume_child_session_notifications=False,
            )
            store = await _reopen_store(session_store_case, store)
            pre_count_lifecycle = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=parent_id,
                    max_children_inspected=8,
                )
            )
            assert pre_count_lifecycle.entries[0].freshness is (
                ChildSessionNotificationFreshness.FRESH
            )
            await store.mark_model_completion_stage_dispatched(
                parent_id,
                stage=prepared.stage,
                consume_child_session_notifications=True,
            )
            store = await _reopen_store(session_store_case, store)
            lifecycle = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=parent_id,
                    max_children_inspected=8,
                )
            )
            assert len(lifecycle.entries) == 1
            assert lifecycle.entries[0].freshness is (ChildSessionNotificationFreshness.CONSUMED)
            recovered_parent = await store.load(parent_id)
            assert recovered_parent is not None
            assert (
                await contributor.build(
                    session_store=store,
                    session=recovered_parent,
                )
            ).projection is None

            original_child = await store.load(child_id)
            assert original_child is not None
            await store.delete_session(child_id)

            async def recreate_child() -> Session:
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id=child_id,
                        parent_session_id=parent_id,
                        messages=[],
                    ),
                    identity=_identity(),
                )
                await store.transition_status(
                    child_id,
                    from_statuses={SessionStatus.PENDING},
                    to_status=SessionStatus.RUNNING,
                )
                await store.append_event(
                    child_id,
                    Event(
                        id="child-notification-started",
                        type=EventType.SESSION_STARTED,
                        session_id=child_id,
                    ),
                )
                await store.append_event(
                    child_id,
                    Event(
                        id="child-notification-terminal",
                        type=EventType.SESSION_COMPLETED,
                        session_id=child_id,
                    ),
                )
                return await store.transition_status(
                    child_id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.COMPLETED,
                )

            def recreate_child_task() -> asyncio.Task[Session]:
                return asyncio.create_task(recreate_child())

            replacement_child = await contextvars.Context().run(recreate_child_task)
            assert replacement_child.instance_id != original_child.instance_id
            replacement = await contributor.build(
                session_store=store,
                session=recovered_parent,
            )
            assert replacement.projection is not None
            assert replacement.projection.entries[0].freshness == "fresh"
            assert replacement.stage_binding is not None
            assert replacement.stage_binding.claims[0].child_session_instance_id == (
                replacement_child.instance_id
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_child_notification_priority_and_occurrence_fence(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:

            async def create_running_child(parent_id: str, child_id: str) -> Session:
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id=child_id,
                        parent_session_id=parent_id,
                        messages=[],
                    ),
                    identity=_identity(),
                )
                child = await store.transition_status(
                    child_id,
                    from_statuses={SessionStatus.PENDING},
                    to_status=SessionStatus.RUNNING,
                )
                await store.append_event(
                    child_id,
                    Event(
                        id=f"{child_id}-started",
                        type=EventType.SESSION_STARTED,
                        session_id=child_id,
                    ),
                )
                return child

            async def complete_child(child_id: str, event_id: str) -> None:
                await store.append_event(
                    child_id,
                    Event(
                        id=event_id,
                        type=EventType.SESSION_COMPLETED,
                        session_id=child_id,
                    ),
                )
                await store.transition_status(
                    child_id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.COMPLETED,
                )

            priority_parent_id = "child-notification-priority-parent"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=priority_parent_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            priority_parent = await store.transition_status(
                priority_parent_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            old_child = await create_running_child(
                priority_parent.id,
                "child-notification-old-fresh",
            )
            newer_child = await create_running_child(
                priority_parent.id,
                "child-notification-newer-consumed",
            )
            await complete_child(newer_child.id, "newer-child-terminal")
            contributor = ChildSessionContextContributor(max_children_inspected=8)
            newer_contribution = await contributor.build(
                session_store=store,
                session=priority_parent,
            )
            assert newer_contribution.stage_binding is not None
            prepared = await store.prepare_model_completion_stage(
                priority_parent.id,
                request=ModelCompletionStageRequest(
                    stage_id="mstep_child_priority:dispatch:0",
                    logical_step_id="mstep_child_priority",
                    dispatch_ordinal=0,
                    intent={
                        "interaction_id": "interaction-child-priority",
                        "model_step_id": "mstep_child_priority",
                        "model_attempt_id": "matt_child_priority",
                        "request_fingerprint": "d" * 64,
                        CHILD_SESSION_NOTIFICATION_INTENT_KEY: (
                            newer_contribution.stage_binding.model_dump(mode="json")
                        ),
                    },
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=priority_parent.run_epoch,
                expected_transcript_cursor=0,
            )
            await store.mark_model_completion_stage_dispatched(
                priority_parent.id,
                stage=prepared.stage,
            )
            await complete_child(old_child.id, "old-child-terminal")
            priority_page = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=priority_parent.id,
                    max_children_inspected=1,
                )
            )
            assert priority_page.has_more is True
            assert len(priority_page.entries) == 1
            assert priority_page.entries[0].child_session_id == old_child.id
            assert priority_page.entries[0].freshness is (ChildSessionNotificationFreshness.FRESH)

            fence_parent_id = "child-notification-occurrence-parent"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=fence_parent_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            fence_parent = await store.transition_status(
                fence_parent_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            fenced_child = await create_running_child(
                fence_parent.id,
                "child-notification-occurrence-child",
            )
            await complete_child(fenced_child.id, "child-terminal-original")
            fenced_contribution = await contributor.build(
                session_store=store,
                session=fence_parent,
            )
            assert fenced_contribution.stage_binding is not None
            fenced_stage = await store.prepare_model_completion_stage(
                fence_parent.id,
                request=ModelCompletionStageRequest(
                    stage_id="mstep_child_occurrence:dispatch:0",
                    logical_step_id="mstep_child_occurrence",
                    dispatch_ordinal=0,
                    intent={
                        "interaction_id": "interaction-child-occurrence",
                        "model_step_id": "mstep_child_occurrence",
                        "model_attempt_id": "matt_child_occurrence",
                        "request_fingerprint": "e" * 64,
                        CHILD_SESSION_NOTIFICATION_INTENT_KEY: (
                            fenced_contribution.stage_binding.model_dump(mode="json")
                        ),
                    },
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=fence_parent.run_epoch,
                expected_transcript_cursor=0,
            )
            await store.transition_status(
                fenced_child.id,
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
            )
            await store.append_event(
                fenced_child.id,
                Event(
                    id="child-invocation-resumed",
                    type=EventType.SESSION_RESUMED,
                    session_id=fenced_child.id,
                ),
            )
            await store.transition_status(
                fenced_child.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
            )
            with pytest.raises(SessionModelCompletionStageConflict):
                await store.mark_model_completion_stage_dispatched(
                    fence_parent.id,
                    stage=fenced_stage.stage,
                )
            fence_page = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=fence_parent.id,
                    max_children_inspected=1,
                )
            )
            assert fence_page.entries == ()
            assert fence_page.unavailable_child_count == 1

            await store.append_event(
                fenced_child.id,
                Event(
                    id="child-terminal-replacement",
                    type=EventType.SESSION_COMPLETED,
                    session_id=fenced_child.id,
                ),
            )
            fence_page = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=fence_parent.id,
                    max_children_inspected=1,
                )
            )
            assert fence_page.entries[0].occurrence.source_id == ("child-terminal-replacement")
            assert fence_page.entries[0].freshness is (ChildSessionNotificationFreshness.FRESH)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_child_lifecycle_is_canonical_and_store_deterministic(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:

            async def create_running_child(parent_id: str, child_id: str) -> Session:
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id=child_id,
                        parent_session_id=parent_id,
                        messages=[],
                    ),
                    identity=_identity(),
                )
                child = await store.transition_status(
                    child_id,
                    from_statuses={SessionStatus.PENDING},
                    to_status=SessionStatus.RUNNING,
                )
                await store.append_event(
                    child_id,
                    Event(
                        id=f"{child_id}-started",
                        type=EventType.SESSION_STARTED,
                        session_id=child_id,
                    ),
                )
                return child

            async def complete_child(
                child_id: str,
                event_id: str,
                *,
                timestamp: datetime,
            ) -> None:
                await store.append_event(
                    child_id,
                    Event(
                        id=event_id,
                        type=EventType.SESSION_COMPLETED,
                        session_id=child_id,
                        timestamp=timestamp,
                    ),
                )
                await store.transition_status(
                    child_id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.COMPLETED,
                )

            ordering_parent = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="child-lifecycle-ordering-parent",
                    messages=[],
                ),
                identity=_identity(),
            )
            first_created = await create_running_child(
                ordering_parent.id,
                "child-lifecycle-first-created",
            )
            second_created = await create_running_child(
                ordering_parent.id,
                "child-lifecycle-second-created",
            )
            ordering_time = datetime(2030, 1, 1, tzinfo=UTC)
            await complete_child(
                second_created.id,
                "child-lifecycle-second-terminal",
                timestamp=ordering_time,
            )
            await complete_child(
                first_created.id,
                "child-lifecycle-first-terminal",
                timestamp=ordering_time + timedelta(seconds=1),
            )
            ordering_page = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=ordering_parent.id,
                    max_children_inspected=1,
                )
            )
            assert ordering_page.has_more is True
            assert [entry.child_session_id for entry in ordering_page.entries] == [
                second_created.id
            ]

            transient_parent = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="child-lifecycle-transient-parent",
                    messages=[],
                ),
                identity=_identity(),
            )
            mismatched = await create_running_child(
                transient_parent.id,
                "child-lifecycle-transient-mismatch",
            )
            fresh = await create_running_child(
                transient_parent.id,
                "child-lifecycle-transient-fresh",
            )
            await store.append_event(
                mismatched.id,
                Event(
                    id="child-lifecycle-uncommitted-terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id=mismatched.id,
                ),
            )
            await complete_child(
                fresh.id,
                "child-lifecycle-fresh-terminal",
                timestamp=ordering_time + timedelta(seconds=2),
            )
            transient_page = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=transient_parent.id,
                    max_children_inspected=1,
                )
            )
            assert [entry.child_session_id for entry in transient_page.entries] == [fresh.id]
            assert transient_page.entries[0].freshness is (ChildSessionNotificationFreshness.FRESH)

            conflict_parent = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="child-lifecycle-conflict-parent",
                    messages=[],
                ),
                identity=_identity(),
            )
            conflict_parent = await store.transition_status(
                conflict_parent.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            conflicting = await create_running_child(
                conflict_parent.id,
                "child-lifecycle-conflicting-child",
            )
            await complete_child(
                conflicting.id,
                "child-lifecycle-completed",
                timestamp=ordering_time + timedelta(seconds=3),
            )
            contribution = await ChildSessionContextContributor().build(
                session_store=store,
                session=conflict_parent,
            )
            assert contribution.stage_binding is not None
            prepared = await store.prepare_model_completion_stage(
                conflict_parent.id,
                request=ModelCompletionStageRequest(
                    stage_id="mstep_child_conflict:dispatch:0",
                    logical_step_id="mstep_child_conflict",
                    dispatch_ordinal=0,
                    intent={
                        "interaction_id": "interaction-child-conflict",
                        "model_step_id": "mstep_child_conflict",
                        "model_attempt_id": "matt_child_conflict",
                        "request_fingerprint": "f" * 64,
                        CHILD_SESSION_NOTIFICATION_INTENT_KEY: (
                            contribution.stage_binding.model_dump(mode="json")
                        ),
                    },
                ),
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=conflict_parent.run_epoch,
                expected_transcript_cursor=0,
            )
            await store.append_event(
                conflicting.id,
                Event(
                    id="child-lifecycle-later-failure",
                    type=EventType.SESSION_FAILED,
                    session_id=conflicting.id,
                    timestamp=ordering_time + timedelta(seconds=4),
                ),
            )
            conflict_page = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=conflict_parent.id,
                    max_children_inspected=1,
                )
            )
            assert conflict_page.entries == ()
            assert conflict_page.unavailable_child_count == 1
            with pytest.raises(SessionModelCompletionStageConflict):
                await store.mark_model_completion_stage_dispatched(
                    conflict_parent.id,
                    stage=prepared.stage,
                )

            await store.transition_status(
                conflicting.id,
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
            )
            still_unavailable = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=conflict_parent.id,
                    max_children_inspected=1,
                )
            )
            assert still_unavailable.entries == ()
            assert still_unavailable.unavailable_child_count == 1
            await store.append_event(
                conflicting.id,
                Event(
                    id="child-lifecycle-resumed",
                    type=EventType.SESSION_RESUMED,
                    session_id=conflicting.id,
                    timestamp=ordering_time + timedelta(seconds=5),
                ),
            )
            resumed_page = await store.query_child_session_lifecycle(
                ChildSessionLifecycleQuery(
                    parent_session_id=conflict_parent.id,
                    max_children_inspected=1,
                )
            )
            assert resumed_page.entries[0].state.value == "running"
            assert resumed_page.entries[0].occurrence.source_id == "child-lifecycle-resumed"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_durable_session_message_queue(session_store_case) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_queue_conformance",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            idle_request = EnqueueSessionMessageRequest(
                session_id=created.id,
                idempotency_key="queue-idle",
                content="idle",
                delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
            )
            next_one_request = EnqueueSessionMessageRequest(
                session_id=created.id,
                idempotency_key="queue-next-1",
                content="next one",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
            next_two_request = EnqueueSessionMessageRequest(
                session_id=created.id,
                idempotency_key="queue-next-2",
                content="next two",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
            idle = await store.enqueue_session_message(idle_request)
            next_one = await store.enqueue_session_message(next_one_request)
            next_two = await store.enqueue_session_message(next_two_request)
            replay = await store.enqueue_session_message(next_one_request)
            assert replay.replayed is True
            assert replay.message.queue_id == next_one.message.queue_id
            with pytest.raises(ValueError, match="different request"):
                await store.enqueue_session_message(
                    next_one_request.model_copy(update={"content": "changed"})
                )

            store = await _reopen_store(session_store_case, store)
            reconstructed = await store.enqueue_session_message(next_one_request)
            assert reconstructed.replayed is True
            assert reconstructed.message == next_one.message

            await store.transition_status(
                created.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.transition_status(
                created.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTED,
            )
            with pytest.raises(
                SessionStatusConflict,
                match="delivered only while running",
            ):
                await store.deliver_queued_session_messages(
                    created.id,
                    include_on_idle=True,
                )
            await store.transition_status(
                created.id,
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.RUNNING,
            )
            first = await store.deliver_queued_session_messages(
                created.id,
                include_on_idle=True,
                limit=1,
                interaction_id="interaction-queue-first",
                interaction_started_event=Event(
                    id="evt_interaction_queue_first",
                    type=EventType.INTERACTION_STARTED,
                    session_id=created.id,
                    interaction_id="interaction-queue-first",
                ),
            )
            assert [event.type for event in first.events] == [
                EventType.INTERACTION_STARTED,
                EventType.SESSION_MESSAGE_DELIVERED,
            ]
            assert [message.queue_id for message in first.messages] == [next_one.message.queue_id]
            assert first.has_more is True
            late = await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=created.id,
                    idempotency_key="queue-late",
                    content="late next boundary",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            second = await store.deliver_queued_session_messages(
                created.id,
                include_on_idle=True,
                eligible_through=first.eligible_through,
                limit=1,
                interaction_id="interaction-queue-first",
            )
            third = await store.deliver_queued_session_messages(
                created.id,
                include_on_idle=True,
                eligible_through=first.eligible_through,
                limit=1,
                interaction_id="interaction-queue-first",
            )
            assert [message.queue_id for message in second.messages] == [next_two.message.queue_id]
            assert [message.queue_id for message in third.messages] == [idle.message.queue_id]
            assert third.has_more is False

            with pytest.raises(SessionQueuedMessagesPending):
                await store.transition_status_if_no_queued_messages(
                    created.id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.COMPLETED,
                )
            late_batch = await store.deliver_queued_session_messages(
                created.id,
                include_on_idle=False,
                interaction_id="interaction-queue-late",
                interaction_started_event=Event(
                    id="evt_interaction_queue_late",
                    type=EventType.INTERACTION_STARTED,
                    session_id=created.id,
                    interaction_id="interaction-queue-late",
                ),
            )
            assert [message.queue_id for message in late_batch.messages] == [late.message.queue_id]
            completed = await store.transition_status_if_no_queued_messages(
                created.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
            )
            assert completed.status == SessionStatus.COMPLETED
            transcript = await store.load_transcript(created.id)
            assert [message.content[0].text for message in transcript] == [  # type: ignore[union-attr]
                "next one",
                "next two",
                "idle",
                "late next boundary",
            ]
            transcript_page = await store.query_transcript(
                TranscriptQuery(session_id=created.id, limit=10)
            )
            assert [record.interaction_id for record in transcript_page.records] == [
                "interaction-queue-first",
                "interaction-queue-first",
                "interaction-queue-first",
                "interaction-queue-late",
            ]
            lifecycle = await store.query_events(
                EventQuery(
                    session_id=created.id,
                    event_types=(EventType.INTERACTION_STARTED,),
                )
            )
            assert [record.event.id for record in lifecycle] == [
                "evt_interaction_queue_first",
                "evt_interaction_queue_late",
            ]
            queue_events = [
                event
                for event in await store.load_events(created.id)
                if event.type
                in {EventType.SESSION_MESSAGE_QUEUED, EventType.SESSION_MESSAGE_DELIVERED}
            ]
            assert len(queue_events) == 8
            assert all("content" not in event.payload for event in queue_events)
            deliveries = await store.list_persisted_event_side_effect_deliveries(limit=1000)
            assert {delivery.event_id for delivery in deliveries} == {
                event.id for event in [*queue_events, *(record.event for record in lifecycle)]
            }
            assert all(
                delivery.status is PersistedEventSideEffectStatus.PENDING for delivery in deliveries
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_preserves_typed_queued_messages(session_store_case) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"sess_typed_queue_{session_store_case[0]}"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            message = Message(
                role=MessageRole.USER,
                content=(
                    FilePart(
                        attachment=FileAttachment(
                            artifact_id="artifact:queued-conformance-document",
                            kind=FileAttachmentKind.DOCUMENT,
                            filename="contract.pdf",
                            content_type="application/pdf",
                            size_bytes=128,
                        ).model_dump(mode="json")
                    ),
                ),
            )
            request = EnqueueSessionMessageRequest(
                session_id=session_id,
                idempotency_key="typed-queue-conformance",
                content="Attached file: contract.pdf",
                message=message,
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
            accepted = await store.enqueue_session_message(request)

            store = await _reopen_store(session_store_case, store)
            replay = await store.enqueue_session_message(request)
            assert replay.replayed is True
            assert replay.message == accepted.message
            assert replay.message.message == message

            await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            delivered = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
            )
            assert len(delivered.messages) == 1
            assert delivered.messages[0].message == message
            assert (await store.load_transcript(session_id)) == [message]
        finally:
            await store.release_run_fence(f"sess_typed_queue_{session_store_case[0]}")
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_enqueue_completion_race_is_atomic(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_queue_completion_conformance",
                    messages=[Message.text("user", "create only")],
                ),
                identity=_identity(),
            )
            await store.transition_status(
                created.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            start = asyncio.Event()

            async def enqueue():
                await start.wait()
                return await store.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id=created.id,
                        idempotency_key="completion-race",
                        content="race steering",
                        delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                    )
                )

            async def complete():
                await start.wait()
                return await store.transition_status_if_no_queued_messages(
                    created.id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.COMPLETED,
                )

            enqueue_task = asyncio.create_task(enqueue())
            completion_task = asyncio.create_task(complete())
            start.set()
            enqueue_result, completion_result = await asyncio.gather(
                enqueue_task,
                completion_task,
                return_exceptions=True,
            )

            if isinstance(enqueue_result, Exception):
                assert isinstance(enqueue_result, SessionStatusConflict)
                assert "pending or running" in str(enqueue_result)
                assert not isinstance(completion_result, Exception)
                assert completion_result.status is SessionStatus.COMPLETED
                events = await store.query_events(
                    EventQuery(
                        session_id=created.id,
                        event_type=EventType.SESSION_MESSAGE_QUEUED,
                    )
                )
                assert events == []
            else:
                assert enqueue_result.message.content == "race steering"
                assert isinstance(completion_result, SessionQueuedMessagesPending)
                delivered = await store.deliver_queued_session_messages(
                    created.id,
                    include_on_idle=False,
                )
                assert [message.queue_id for message in delivered.messages] == [
                    enqueue_result.message.queue_id
                ]
                completed = await store.transition_status_if_no_queued_messages(
                    created.id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.COMPLETED,
                )
                assert completed.status is SessionStatus.COMPLETED
            with pytest.raises(SessionStatusConflict, match="pending or running"):
                await store.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id=created.id,
                        idempotency_key="after-completion",
                        content="too late",
                        delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                    )
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reconstructs_queue_delivery_acknowledgement(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_queue_delivery_reconstruction"
            interaction_id = "interaction-queue-delivery-reconstruction"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "initial")],
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="evt_queue_delivery_initial_interaction",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id="interaction-queue-delivery-initial",
                ),
                interaction_source_messages=[Message.text("user", "initial")],
            )
            for index in range(2):
                await store.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id=session_id,
                        idempotency_key=f"queue-delivery-reconstruction-{index}",
                        content=f"queued {index}",
                        delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                    )
                )

            started = Event(
                id="evt_queue_delivery_reconstruction_started",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            first = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                delivery_id=interaction_id,
                limit=1,
                interaction_id=interaction_id,
                interaction_started_event=started,
            )
            assert first.replayed is False
            assert first.delivery_id == interaction_id
            assert first.has_more is True

            store = await _reopen_store(session_store_case, store)
            replayed = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                delivery_id=interaction_id,
                limit=1,
                interaction_id=interaction_id,
                interaction_started_event=started,
            )
            assert replayed.replayed is True
            assert replayed == first.model_copy(update={"replayed": True})

            with pytest.raises(ValueError, match="different queue delivery"):
                await store.deliver_queued_session_messages(
                    session_id,
                    include_on_idle=True,
                    delivery_id=interaction_id,
                    limit=1,
                    interaction_id=interaction_id,
                    interaction_started_event=started,
                )

            second = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                delivery_id=f"{interaction_id}:batch:1",
                eligible_through=first.eligible_through,
                limit=1,
                interaction_id=interaction_id,
            )
            assert second.replayed is False
            assert [message.content for message in second.messages] == ["queued 1"]
            empty_delivery_id = f"{interaction_id}:batch:2"
            empty = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                delivery_id=empty_delivery_id,
                eligible_through=first.eligible_through,
                limit=1,
                interaction_id=interaction_id,
            )
            assert empty.messages == ()
            assert empty.interaction_id == interaction_id
            store = await _reopen_store(session_store_case, store)
            replayed_empty = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                delivery_id=empty_delivery_id,
                eligible_through=first.eligible_through,
                limit=1,
                interaction_id=interaction_id,
            )
            assert replayed_empty == empty.model_copy(update={"replayed": True})
            transcript = await store.load_transcript(session_id)
            assert [
                message.content[0].text  # type: ignore[union-attr]
                for message in transcript
            ] == ["queued 0", "queued 1"]
            lifecycle = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_id=started.id,
                )
            )
            assert [record.event for record in lifecycle] == [started]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_queue_boundary_is_global_and_stable(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            primary = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_queue_global_boundary_primary",
                    messages=[Message.text("user", "primary")],
                ),
                identity=_identity(),
            )
            other = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_queue_global_boundary_other",
                    messages=[Message.text("user", "other")],
                ),
                identity=_identity(),
            )
            for session in (primary, other):
                await store.transition_status(
                    session.id,
                    from_statuses={SessionStatus.PENDING},
                    to_status=SessionStatus.RUNNING,
                )

            primary_request = EnqueueSessionMessageRequest(
                session_id=primary.id,
                idempotency_key="primary-before-boundary",
                content="deliver before boundary",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
            accepted = await store.enqueue_session_message(primary_request)
            other_message = await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=other.id,
                    idempotency_key="other-before-boundary",
                    content="advance global boundary",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )

            first = await store.deliver_queued_session_messages(
                primary.id,
                include_on_idle=False,
            )
            assert [message.queue_id for message in first.messages] == [accepted.message.queue_id]
            assert first.eligible_through >= other_message.message.ordering_key

            replay = await store.enqueue_session_message(primary_request)
            assert replay.replayed is True
            assert replay.message.status is SessionMessageQueueStatus.DELIVERED

            late = await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=primary.id,
                    idempotency_key="primary-after-boundary",
                    content="deliver after boundary",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            fenced = await store.deliver_queued_session_messages(
                primary.id,
                include_on_idle=False,
                eligible_through=first.eligible_through,
            )
            assert fenced.messages == ()

            current = await store.deliver_queued_session_messages(
                primary.id,
                include_on_idle=False,
            )
            assert [message.queue_id for message in current.messages] == [late.message.queue_id]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_interaction_completion_replays_queue_decision(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_interaction_completion_queue_conformance"
            interaction_id = "interaction-completion-queue-conformance"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="evt_interaction_completion_queue_started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                ),
                interaction_source_messages=[Message.text("user", "start")],
            )
            await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key="queued-before-interaction-completion",
                    content="continue",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            completed = Event(
                id="evt_interaction_completion_queue_completed",
                type=EventType.INTERACTION_COMPLETED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            publication = await store.publish_interaction_transition(
                session_id,
                event=completed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
                only_if_no_queued_messages=True,
            )
            assert publication.status_changed is False
            assert publication.session.status is SessionStatus.RUNNING
            receipt = await store.load_interaction_transition_receipt(
                session_id,
                transition=InteractionTransitionSpec(
                    event=completed,
                    from_statuses=(SessionStatus.RUNNING,),
                    to_status=SessionStatus.COMPLETED,
                    only_if_no_queued_messages=True,
                ),
            )
            assert receipt is not None
            assert receipt.status_changed is False
            assert receipt.session.status is SessionStatus.RUNNING

            await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                interaction_id="interaction-after-queued-completion",
                interaction_started_event=Event(
                    id="evt_interaction_after_queued_completion_started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id="interaction-after-queued-completion",
                ),
            )
            later_completed = Event(
                id="evt_interaction_after_queued_completion_completed",
                type=EventType.INTERACTION_COMPLETED,
                session_id=session_id,
                interaction_id="interaction-after-queued-completion",
            )
            later_publication = await store.publish_interaction_transition(
                session_id,
                event=later_completed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
                only_if_no_queued_messages=True,
            )
            assert later_publication.status_changed is True
            assert later_publication.session.status is SessionStatus.COMPLETED

            replayed = await store.publish_interaction_transition(
                session_id,
                event=completed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
                only_if_no_queued_messages=True,
            )
            assert replayed.replayed is True
            assert replayed.status_changed is False
            assert replayed.session == publication.session
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_atomically_rebinds_queued_next_turn(
    session_store_case,
) -> None:
    class BlockingQueuedHandoffProvider(ModelProvider):
        name = "atomic-queued-handoff"

        def __init__(self) -> None:
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.second_started = asyncio.Event()
            self.release_second = asyncio.Event()
            self.requests: list[ModelRequest] = []

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return versioned_test_provider_identity(self)

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                self.first_started.set()
                await self.release_first.wait()
                text = "first"
            else:
                self.second_started.set()
                await self.release_second.wait()
                text = "second"
            yield ModelStreamEvent.text_delta(text)
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> None:
        store = await _open_store(session_store_case)
        provider = BlockingQueuedHandoffProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = f"atomic-queued-handoff-{session_store_case[0]}"

        async def execute() -> list[Event]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "initial")],
                    )
                )
            ]

        run_task = asyncio.create_task(execute())
        try:
            await asyncio.wait_for(provider.first_started.wait(), timeout=10)
            checkpoint_a = await store.load_checkpoint(session_id)
            active_a = active_invocation_execution_profile_from_checkpoint(checkpoint_a)
            assert active_a is not None
            accepted = await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key="atomic-next-turn",
                    content="continue",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            provider.release_first.set()
            await asyncio.wait_for(provider.second_started.wait(), timeout=10)

            checkpoint_b = await store.load_checkpoint(session_id)
            active_b = active_invocation_execution_profile_from_checkpoint(checkpoint_b)
            assert active_b is not None
            assert active_b.interaction_id != active_a.interaction_id
            assert active_b.run_epoch == active_a.run_epoch
            assert active_b.profile == active_a.profile
            durable = await store.load_events(session_id)
            predecessor = next(
                event
                for event in durable
                if event.type is EventType.INTERACTION_COMPLETED
                and event.interaction_id == active_a.interaction_id
            )
            target_start = next(
                event
                for event in durable
                if event.type is EventType.INTERACTION_STARTED
                and event.interaction_id == active_b.interaction_id
            )
            delivery = next(
                event for event in durable if event.type is EventType.SESSION_MESSAGE_DELIVERED
            )
            handoff = QueuedInteractionProfileHandoff(
                expected_session_instance_id=(await store.load(session_id)).instance_id,  # type: ignore[union-attr]
                predecessor_settlement_event_id=predecessor.id,
                expected_active_profile=active_a,
                target_active_profile=active_b,
            )
            assert target_start.payload[
                QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY
            ] == queued_interaction_profile_handoff_evidence(handoff)
            assert delivery.interaction_id == active_b.interaction_id
            assert accepted.message.status is SessionMessageQueueStatus.QUEUED

            with pytest.raises(
                ValueError,
                match="start evidence requires its atomic profile handoff",
            ):
                await store.deliver_queued_session_messages(
                    session_id,
                    include_on_idle=True,
                    delivery_id=active_b.interaction_id,
                    interaction_id=active_b.interaction_id,
                    interaction_started_event=target_start,
                )
            replay = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=True,
                delivery_id=active_b.interaction_id,
                interaction_id=active_b.interaction_id,
                interaction_started_event=target_start,
                profile_handoff=handoff,
            )
            assert replay.replayed is True
            assert [message.queue_id for message in replay.messages] == [accepted.message.queue_id]
            assert replay.active_invocation_profile == active_b

            provider.release_second.set()
            emitted = await asyncio.wait_for(run_task, timeout=10)
            assert emitted[-1].type is EventType.SESSION_COMPLETED

            store = await _reopen_store(session_store_case, store)
            persisted_checkpoint = await store.load_checkpoint(session_id)
            assert (
                active_invocation_execution_profile_from_checkpoint(persisted_checkpoint)
                == active_b
            )
        finally:
            provider.release_first.set()
            provider.release_second.set()
            if not run_task.done():
                run_task.cancel()
                await asyncio.gather(run_task, return_exceptions=True)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_repairs_exact_legacy_queued_handoff(
    session_store_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.runtime import _session_steering as steering

    # Historical writers left the predecessor profile in place and did not
    # persist completion evidence for steering. Simulate both parts of that
    # stored state; current writers fence new work on a closed predecessor.
    monkeypatch.setattr(
        steering,
        "prepare_interaction_completion_steering_record",
        lambda *args, **kwargs: None,
    )

    class BlockingLegacyHandoffProvider(ModelProvider):
        name = "legacy-queued-handoff"

        def __init__(self) -> None:
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.second_started = asyncio.Event()
            self.release_second = asyncio.Event()
            self.calls = 0

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return versioned_test_provider_identity(self)

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.calls += 1
            if self.calls == 1:
                self.first_started.set()
                await self.release_first.wait()
            else:
                self.second_started.set()
                await self.release_second.wait()
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> None:
        store = await _open_store(session_store_case)
        store_type = type(store)
        original_delivery = store_type.deliver_queued_session_messages

        async def legacy_split_delivery(self, session_id: str, **kwargs):
            handoff = kwargs.get("profile_handoff")
            started = kwargs.get("interaction_started_event")
            if handoff is None or started is None:
                return await original_delivery(self, session_id, **kwargs)
            payload = copy.deepcopy(started.payload)
            payload.pop(QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY)
            legacy_started = started.model_copy(update={"payload": payload}, deep=True)
            legacy_kwargs = {
                **kwargs,
                "interaction_started_event": legacy_started,
                "profile_handoff": None,
            }
            batch = await original_delivery(self, session_id, **legacy_kwargs)
            return batch.model_copy(
                update={"active_invocation_profile": handoff.target_active_profile},
                deep=True,
            )

        monkeypatch.setattr(
            store_type,
            "deliver_queued_session_messages",
            legacy_split_delivery,
        )
        provider = BlockingLegacyHandoffProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = f"legacy-queued-handoff-{session_store_case[0]}"

        async def execute() -> list[Event]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "initial")],
                    )
                )
            ]

        run_task = asyncio.create_task(execute())
        try:
            await asyncio.wait_for(provider.first_started.wait(), timeout=10)
            active_a = active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint(session_id)
            )
            assert active_a is not None
            await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key="legacy-next-turn",
                    content="continue",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            provider.release_first.set()
            await asyncio.wait_for(provider.second_started.wait(), timeout=10)

            session = await store.load(session_id)
            checkpoint = await store.load_checkpoint(session_id)
            assert session is not None
            assert active_invocation_execution_profile_from_checkpoint(checkpoint) == active_a
            assert (
                await store.load_session_operation(
                    session_id,
                    steering.steering_operation_key(session.instance_id, active_a.interaction_id),
                )
                is None
            )
            starts = [
                event
                for event in await store.load_events(session_id)
                if event.type is EventType.INTERACTION_STARTED
            ]
            assert len(starts) == 2
            assert QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY not in starts[1].payload

            fresh = CayuApp(session_store=store, enable_logging=False)
            fresh.register_provider(provider, default=True)
            fresh.register_agent(AgentSpec(name="assistant", model="fake-model"))
            repaired = await fresh._execution_profile_continuation.validate(
                session=session,
                checkpoint=checkpoint,
                registered_agent=fresh._get_registered_agent(session.agent_name),
                registered_provider=fresh._get_registered_provider(session.provider_name),
                budget_policy=None,
            )
            assert repaired.interaction_id == starts[1].interaction_id
            assert (
                active_invocation_execution_profile_from_checkpoint(
                    await store.load_checkpoint(session_id)
                )
                == repaired
            )

            provider.release_second.set()
            emitted = await asyncio.wait_for(run_task, timeout=10)
            assert emitted[-1].type is EventType.SESSION_COMPLETED
            monkeypatch.setattr(
                store_type,
                "deliver_queued_session_messages",
                original_delivery,
            )
            store = await _reopen_store(session_store_case, store)
            assert (
                active_invocation_execution_profile_from_checkpoint(
                    await store.load_checkpoint(session_id)
                )
                == repaired
            )
        finally:
            provider.release_first.set()
            provider.release_second.set()
            if not run_task.done():
                run_task.cancel()
                await asyncio.gather(run_task, return_exceptions=True)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_invalid_queue_boundary_before_mutation(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = "sess_invalid_queue_boundary_conformance"
            request = EnqueueSessionMessageRequest(
                session_id=session_id,
                idempotency_key="invalid-queue-boundary-message",
                content="must remain queued",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            accepted = await store.enqueue_session_message(request)
            await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            started = Event(
                id="evt_invalid_queue_boundary_started",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id="interaction-invalid-queue-boundary",
            )

            invalid_boundaries: tuple[object, ...] = (
                True,
                1.5,
                MAX_DURABLE_JSON_INTEGER + 1,
            )
            for index, invalid_boundary in enumerate(invalid_boundaries):
                with pytest.raises(ValueError, match="eligible_through must be an integer"):
                    await store.deliver_queued_session_messages(
                        session_id,
                        include_on_idle=False,
                        eligible_through=invalid_boundary,  # type: ignore[arg-type]
                        delivery_id=f"invalid-queue-boundary-{index}",
                        interaction_id="interaction-invalid-queue-boundary",
                        interaction_started_event=started,
                    )

            replayed = await store.enqueue_session_message(request)
            assert replayed.replayed is True
            assert replayed.message.queue_id == accepted.message.queue_id
            assert replayed.message.status is SessionMessageQueueStatus.QUEUED
            assert await store.load_transcript(session_id) == []
            lifecycle = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=(EventType.INTERACTION_STARTED,),
                )
            )
            assert lifecycle == []
        finally:
            await store.release_run_fence("sess_invalid_queue_boundary_conformance")
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_materializes_deferred_interaction_input_fallback(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_deferred_interaction_fallback",
                    messages=[Message.text("user", "bootstrap")],
                ),
                identity=_identity(),
            )
            await store.append_transcript_messages(
                session.id,
                [Message.text("assistant", "previous response")],
                interaction_id=None,
            )
            await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                interaction_started_event=Event(
                    id="evt_deferred_interaction_fallback",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session.id,
                    interaction_id="interaction-deferred-fallback",
                ),
                interaction_source_messages=[Message.text("user", "deferred source")],
                defer_interaction_source=True,
            )
            store = await _reopen_store(session_store_case, store)

            with pytest.raises(
                RuntimeError,
                match="Deferred interaction input belongs to another interaction",
            ):
                await store.materialize_deferred_interaction_input(
                    session.id,
                    interaction_id="interaction-wrong",
                )
            before = await store.query_transcript(TranscriptQuery(session_id=session.id, limit=10))
            assert [record.interaction_id for record in before.records] == [None]

            assert await store.materialize_deferred_interaction_input(
                session.id,
                interaction_id="interaction-deferred-fallback",
            )
            store = await _reopen_store(session_store_case, store)
            after = await store.query_transcript(TranscriptQuery(session_id=session.id, limit=10))
            assert [record.interaction_id for record in after.records] == [
                None,
                "interaction-deferred-fallback",
            ]
            assert [record.message for record in after.records] == [
                Message.text("assistant", "previous response"),
                Message.text("user", "deferred source"),
            ]
            assert not await store.materialize_deferred_interaction_input(
                session.id,
                interaction_id="interaction-deferred-fallback",
            )
        finally:
            await store.release_run_fence("sess_deferred_interaction_fallback")
            await _close_store(store)

    asyncio.run(run())
