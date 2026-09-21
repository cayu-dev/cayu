from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.core.test_context_view_admission import _close, _factory
from tests.core.test_participant_continuation_boundaries import CountingTool, activate, collect
from tests.core.test_participant_identity import CONTEXT, app, create, registration

from cayu import (
    AlwaysRequireApprovalToolPolicy,
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from cayu.agents import AgentSpec
from cayu.collaboration.lifecycle import ParticipantLifecycleChange
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.authority import SessionRunFenced
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.session_message_lifecycle import SessionMessageQuery
from cayu.sessions.base import (
    EnqueueSessionMessageRequest,
    IncompleteSessionRecoveryRequest,
    Message,
    ResumeRequest,
    RunRequest,
    SessionStatus,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "entrance,state",
    [(e, s) for e in ("root", "resume") for s in ("active", "disabled", "retired")]
    + [("approval", s) for s in ("active", "disabled")]
    + [(e, "cancelled") for e in ("root", "approval")]
    + [("root", "terminal_publication_failure")]
    + [("approval", "tampered_terminal")]
    + [("unbound", "active")],
)
def test_queued_successor_requires_current_participant_authority(
    backend, entrance, state, tmp_path, request, monkeypatch
):
    factory = _factory(backend, tmp_path, request, [datetime.now(UTC)])

    async def run():
        store = factory()
        collaboration = InMemoryCollaborationStore()
        entered, release = asyncio.Event(), asyncio.Event()
        barrier_index = 1 if entrance in {"resume", "approval"} else 0

        class Provider(ScriptedModelProvider):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="participant-queued-test", behavior_version="1", implementation_version="1"
                )

            async def stream(self, request):
                if len(self.requests) == barrier_index:
                    entered.set()
                    await release.wait()
                async for event in super().stream(request):
                    yield event

        scripts = [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for _ in range(5)
        ]
        if entrance == "approval":
            scripts[0] = [
                ModelStreamEvent.tool_call(id="count-1", name="count", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        provider = Provider(scripts)
        reg = registration()

        def configured():
            value = app(collaboration, reg, session_store=store)
            value.register_provider(provider, default=True)
            value.register_agent(
                AgentSpec(name="reviewer", model="model"),
                tools=[CountingTool()] if entrance == "approval" else [],
                tool_policy=AlwaysRequireApprovalToolPolicy(tools=["count"])
                if entrance == "approval"
                else None,
            )
            return value

        value = configured()
        task = None
        try:
            initialized = await value.initialize_collaboration()
            _, created = await create(value, initialized)
            participant = created.participants[0].reference
            if entrance == "unbound":
                session_id = str(uuid4())
                stream = value.run(
                    RunRequest(
                        agent_name="reviewer",
                        session_id=session_id,
                        messages=[Message.text("user", "initial")],
                    )
                )
            else:
                session, execution = await activate(value, participant)
                session_id = session.id
                stream = value.execute_participant_session(
                    execution, participant=participant, context=CONTEXT
                )
                if entrance == "resume":
                    await collect(stream)
                    stream = value.resume(
                        ResumeRequest(
                            session_id=session_id, messages=[Message.text("user", "resume")]
                        ),
                        context=CONTEXT,
                    )
                elif entrance == "approval":
                    initial_events = await collect(stream)
                    approval = next(
                        e
                        for e in initial_events
                        if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                    )
                    stream = value.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id=session_id,
                            approval_id=approval.payload["approval_id"],
                            tool_round_id=approval.payload["tool_round_id"],
                            tool_call_id=approval.payload["tool_call_id"],
                            decision=ToolApprovalDecision.APPROVE,
                        ),
                        context=CONTEXT,
                    )
            observed = []

            async def consume():
                async for event in stream:
                    observed.append(event)
                return observed

            task = asyncio.create_task(consume())
            await asyncio.wait_for(entered.wait(), 10)
            await value.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key="queued-successor",
                    content="queued next interaction",
                    delivery_mode="on_idle",
                )
            )
            authority_entered = asyncio.Event()
            if state in {"cancelled", "terminal_publication_failure"}:

                async def interrupted_inspection(*args, **kwargs):
                    authority_entered.set()
                    await asyncio.Event().wait()

                monkeypatch.setattr(
                    value._participant_coordinator, "inspect", interrupted_inspection
                )
                if state == "terminal_publication_failure":
                    original_emit = value._event_writer.emit
                    failed_terminal_publication = False

                    async def fail_terminal_publication(event):
                        nonlocal failed_terminal_publication
                        if (
                            event.type is EventType.SESSION_INTERRUPTED
                            and not failed_terminal_publication
                        ):
                            failed_terminal_publication = True
                            raise OSError("terminal event publication failed")
                        return await original_emit(event)

                    monkeypatch.setattr(value._event_writer, "emit", fail_terminal_publication)
            elif state != "active":
                await value.change_participant_lifecycle(
                    ParticipantLifecycleChange(
                        operation=initialized.operation("disable-before-handoff"),
                        participant=participant,
                        expected_lifecycle_revision=1,
                        state="disabled" if state == "tampered_terminal" else state,
                    ),
                    context=CONTEXT,
                )
            before = await store.load_events(session_id)
            before_session = await store.load(session_id)
            starts_before = sum(e.type is EventType.INTERACTION_STARTED for e in before)
            tampered = False
            if state == "tampered_terminal":
                query_events = store.query_events

                async def corrupt_candidate(query):
                    nonlocal tampered
                    records = await query_events(query)
                    if (
                        query.limit == 1
                        and query.event_types == (EventType.SESSION_FAILED,)
                        and records
                    ):
                        tampered = True
                        event = records[0].event.model_copy(
                            update={"payload": {"error": "different terminal evidence"}}
                        )
                        return [records[0].model_copy(update={"event": event})]
                    return records

                monkeypatch.setattr(store, "query_events", corrupt_candidate)
            release.set()
            if state in {"cancelled", "terminal_publication_failure"}:
                await asyncio.wait_for(authority_entered.wait(), 10)
                task.cancel()
                assert task.cancelling() == 1
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled()
                assert task.cancelling() == 1
                events = observed
            elif state == "tampered_terminal":
                with pytest.raises(SessionRunFenced, match="exact terminal session evidence"):
                    await asyncio.wait_for(task, 20)
                assert tampered
                assert (await store.load(session_id)).run_epoch == before_session.run_epoch
                events = observed
            else:
                events = await asyncio.wait_for(task, 20)
            durable = await store.load_events(session_id)
            if state == "active":
                assert len(provider.requests) == barrier_index + 2
                assert any(e.type is EventType.SESSION_COMPLETED for e in events)
                assert (
                    sum(e.type is EventType.INTERACTION_STARTED for e in durable)
                    == starts_before + 1
                )
                assert sum(e.type is EventType.SESSION_MESSAGE_DELIVERED for e in durable) == 1
            else:
                assert len(provider.requests) == barrier_index + 1
                assert (
                    sum(e.type is EventType.INTERACTION_STARTED for e in durable) == starts_before
                )
                assert not any(e.type is EventType.SESSION_MESSAGE_DELIVERED for e in durable)
                assert any(e.type is EventType.INTERACTION_COMPLETED for e in events)
                if state in {"cancelled", "terminal_publication_failure"}:
                    # Preserve predecessor success while settling the abandoned
                    # session; cancellation must leave a resumable boundary.
                    retained = await store.load(session_id)
                    assert retained.status is SessionStatus.INTERRUPTED
                    assert retained.run_epoch == (
                        before_session.run_epoch
                        if state == "terminal_publication_failure"
                        else before_session.run_epoch + 1
                    )
                    assert not any(e.type is EventType.SESSION_FAILED for e in durable)
                    if state == "cancelled":
                        assert any(e.type is EventType.SESSION_INTERRUPTED for e in durable)
                    else:
                        assert not any(e.type is EventType.SESSION_INTERRUPTED for e in durable)
                else:
                    assert any(e.type is EventType.SESSION_FAILED for e in durable)
                if backend != "memory":
                    await store.close()
                    store = factory()
                inspection = await store.inspect_session_messages(
                    SessionMessageQuery(session_id=session_id)
                )
                assert len(inspection.records) == 1
                assert inspection.records[0].status == "queued"
                if state == "terminal_publication_failure":
                    value = configured()
                    await value.initialize_collaboration()
                    repaired = await value.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(session_id=session_id)
                    )
                    assert repaired.events
                    assert repaired.events[0].type is EventType.SESSION_INTERRUPTED
                    checkpoint = await store.load_checkpoint(session_id)
                    assert checkpoint is not None
                    assert checkpoint.get("pending_session_interrupt") is None
                    replayed = await value.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(session_id=session_id)
                    )
                    assert replayed.events == ()
                    await collect(
                        value.resume(
                            ResumeRequest(
                                session_id=session_id,
                                messages=[Message.text("user", "continue safely")],
                            ),
                            context=CONTEXT,
                        )
                    )
                    durable = await store.load_events(session_id)
                    assert sum(e.type is EventType.SESSION_MESSAGE_DELIVERED for e in durable) == 1
                elif state in {"disabled", "cancelled"}:
                    value = configured()
                    await value.initialize_collaboration()
                    if state == "disabled":
                        await value.change_participant_lifecycle(
                            ParticipantLifecycleChange(
                                operation=initialized.operation("reactivate"),
                                participant=participant,
                                expected_lifecycle_revision=2,
                                state="active",
                            ),
                            context=CONTEXT,
                        )
                    await collect(
                        value.resume(
                            ResumeRequest(
                                session_id=session_id,
                                messages=[Message.text("user", "continue safely")],
                            ),
                            context=CONTEXT,
                        )
                    )
                    durable = await store.load_events(session_id)
                    assert sum(e.type is EventType.SESSION_MESSAGE_DELIVERED for e in durable) == 1
        finally:
            release.set()
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await _close(store)
            await collaboration.close()

    asyncio.run(run())
