from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    EventQuery,
    IncompleteSessionRecoveryAction,
    IncompleteSessionsRecoveryRequest,
    InMemorySessionStore,
    ResumeRequest,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    SessionIdentity,
    SessionStatus,
    TerminalEventPublicationUncertain,
)
from cayu.runtime import sessions as sessions_module
from cayu.runtime.interactions import InteractionStatus, InteractionSummaryEvidence


class CompletingProvider(ModelProvider):
    name = "completing"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class FailingProvider(ModelProvider):
    name = "failing"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        raise RuntimeError("provider failed")
        yield  # pragma: no cover


class CommitThenLoseInteractionTransitionAcknowledgementStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.lost_acknowledgement = False
        self.attempted_event_ids: list[str] = []

    async def publish_interaction_transition(
        self,
        session_id,
        *,
        event,
        from_statuses,
        to_status,
        only_if_no_queued_messages=False,
    ):
        self.attempted_event_ids.append(event.id)
        result = await super().publish_interaction_transition(
            session_id,
            event=event,
            from_statuses=from_statuses,
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
        )
        if not self.lost_acknowledgement:
            self.lost_acknowledgement = True
            raise ConnectionError("interaction transition acknowledgement lost")
        return result


class CommitThenLoseTerminalPublicationStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.armed = False
        self._unreadable_event_id: str | None = None
        self.publication_failure: ConnectionError | None = None
        self.reconciliation_failure: TimeoutError | None = None

    async def append_event(self, session_id: str, event: Event) -> None:
        await super().append_event(session_id, event)
        if self.armed and event.type == EventType.SESSION_COMPLETED:
            self.armed = False
            self._unreadable_event_id = event.id
            failure = ConnectionError("terminal append acknowledgement lost")
            self.publication_failure = failure
            raise failure

    async def query_events(self, query: EventQuery):
        if self._unreadable_event_id is not None and query.event_id == self._unreadable_event_id:
            self._unreadable_event_id = None
            failure = TimeoutError("terminal reconciliation unavailable")
            self.reconciliation_failure = failure
            raise failure
        return await super().query_events(query)


@pytest.mark.parametrize(
    ("provider", "expected_status", "expected_event_type"),
    [
        (CompletingProvider(), SessionStatus.COMPLETED, EventType.INTERACTION_COMPLETED),
        (FailingProvider(), SessionStatus.FAILED, EventType.INTERACTION_FAILED),
    ],
)
def test_runtime_reconstructs_interaction_transition_after_commit_acknowledgement_loss(
    provider: ModelProvider,
    expected_status: SessionStatus,
    expected_event_type: EventType,
) -> None:
    async def run() -> None:
        store = CommitThenLoseInteractionTransitionAcknowledgementStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"sess_{expected_status}",
                    messages=[Message.text("user", "run")],
                )
            )
        ]

        session = await store.load(f"sess_{expected_status}")
        assert session is not None
        assert session.status is expected_status
        assert store.lost_acknowledgement is True
        assert len(store.attempted_event_ids) == 2
        assert len(set(store.attempted_event_ids)) == 1
        durable = await store.load_events(session.id)
        assert sum(event.type == expected_event_type for event in durable) == 1
        assert sum(event.type == expected_event_type for event in events) == 1
        assert len(provider.requests) == 1  # type: ignore[attr-defined]

    asyncio.run(run())


def test_terminal_publication_uncertainty_preserves_durable_terminal_outcome() -> None:
    async def run() -> None:
        store = CommitThenLoseTerminalPublicationStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_terminal_publication_uncertain"

        initial_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                )
            )
        ]
        assert initial_events[-1].type == EventType.SESSION_COMPLETED

        store.armed = True
        with pytest.raises(TerminalEventPublicationUncertain) as raised:
            [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]

        session = await store.load(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        terminal_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(
                    EventType.SESSION_COMPLETED,
                    EventType.SESSION_FAILED,
                    EventType.SESSION_INTERRUPTED,
                ),
            )
        )

        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert checkpoint is not None
        run_operation = checkpoint["session_run_operation"]
        assert [record.event.type for record in terminal_records] == [
            EventType.SESSION_COMPLETED,
            EventType.SESSION_COMPLETED,
        ]
        assert (
            terminal_records[-1].event.payload["session_run_operation_id"]
            == run_operation["operation_id"]
        )
        assert len(provider.requests) == 2

        uncertainty = raised.value
        assert uncertainty.session_id == session_id
        assert uncertainty.event_id == terminal_records[-1].event.id
        assert uncertainty.event == terminal_records[-1].event
        assert uncertainty.__cause__ is uncertainty.failures
        assert uncertainty.failures.exceptions == (
            store.publication_failure,
            store.reconciliation_failure,
        )

    asyncio.run(run())


def test_initial_run_terminal_publication_uncertainty_preserves_only_completed_outcome() -> None:
    async def run() -> None:
        store = CommitThenLoseTerminalPublicationStore()
        provider = CompletingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_initial_terminal_publication_uncertain"
        store.armed = True

        with pytest.raises(TerminalEventPublicationUncertain) as raised:
            [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "start")],
                    )
                )
            ]

        session = await store.load(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        terminal_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(
                    EventType.SESSION_COMPLETED,
                    EventType.SESSION_FAILED,
                    EventType.SESSION_INTERRUPTED,
                ),
            )
        )

        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert checkpoint is not None
        assert "session_run_operation" not in checkpoint
        assert [record.event.type for record in terminal_records] == [EventType.SESSION_COMPLETED]
        assert len(provider.requests) == 1
        assert raised.value.session_id == session_id
        assert raised.value.event_id == terminal_records[0].event.id
        assert raised.value.event == terminal_records[0].event

    asyncio.run(run())


def test_batch_recovery_attributes_repairs_before_terminal_reconciliation() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    provider = CompletingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    interaction_id = "interaction-batch-recovery"

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_batch_interaction",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name=provider.name, model="fake-model"),
        )
        await store.update_status("sess_batch_interaction", SessionStatus.RUNNING)
        started_at = datetime.now(UTC) - timedelta(seconds=1)
        started = Event(
            id="interaction-batch-recovery-start",
            type=EventType.INTERACTION_STARTED,
            session_id="sess_batch_interaction",
            interaction_id=interaction_id,
            timestamp=started_at,
            payload=InteractionSummaryEvidence(
                status=InteractionStatus.ACTIVE,
                start_event_id="interaction-batch-recovery-start",
                started_at=started_at,
            ).model_dump(mode="json"),
        )
        await store.append_event("sess_batch_interaction", started)
        page = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.RUNNING},
                limit=10,
            )
        )
        return page, await store.load_events("sess_batch_interaction")

    page, events = asyncio.run(setup_and_recover())

    assert page.results[0].actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)
    recovery_events = [event for event in events if event.id != "interaction-batch-recovery-start"]
    assert recovery_events
    assert all(
        event.interaction_id == interaction_id
        for event in recovery_events
        if event.type not in sessions_module.UNASSOCIATED_RUNTIME_EVENT_TYPES
    )
    assert all(
        event.interaction_id is None
        for event in recovery_events
        if event.type in sessions_module.UNASSOCIATED_RUNTIME_EVENT_TYPES
    )
    assert [event.type for event in events].count(EventType.INTERACTION_INTERRUPTED) == 1


def test_batch_recovery_fault_isolates_and_retries_interaction_reconciliation() -> None:
    class FailingInteractionReconciliationStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.interaction_failures_remaining = 2

        async def publish_interaction_transition(self, session_id: str, **kwargs):
            event = kwargs["event"]
            if (
                event.type == EventType.INTERACTION_INTERRUPTED
                and self.interaction_failures_remaining > 0
            ):
                self.interaction_failures_remaining -= 1
                raise ConnectionError("interaction transition unavailable")
            return await super().publish_interaction_transition(session_id, **kwargs)

    async def scenario() -> None:
        store = FailingInteractionReconciliationStore()
        app = CayuApp(session_store=store, enable_logging=False)
        provider = CompletingProvider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_batch_interaction_reconciliation_retry"
        interaction_id = "interaction-batch-reconciliation-retry"
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name=provider.name, model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        started_at = datetime.now(UTC) - timedelta(seconds=1)
        await store.append_event(
            session_id,
            Event(
                id="interaction-batch-reconciliation-retry-start",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
                timestamp=started_at,
                payload=InteractionSummaryEvidence(
                    status=InteractionStatus.ACTIVE,
                    start_event_id="interaction-batch-reconciliation-retry-start",
                    started_at=started_at,
                ).model_dump(mode="json"),
            ),
        )
        request = IncompleteSessionsRecoveryRequest(
            statuses={SessionStatus.INTERRUPTED},
            limit=10,
        )

        failed_page = await app.recover_incomplete_sessions(request)
        assert len(failed_page.results) == 1
        assert failed_page.results[0].actions == (IncompleteSessionRecoveryAction.FAILED,)
        failed_session = await store.load(session_id)
        assert failed_session is not None
        assert failed_session.status is SessionStatus.INTERRUPTED
        first_events = await store.load_events(session_id)
        assert [event.type for event in first_events].count(EventType.INTERACTION_INTERRUPTED) == 0

        retried_page = await app.recover_incomplete_sessions(request)
        assert retried_page.results == ()
        events = await store.load_events(session_id)
        assert [event.type for event in events].count(EventType.SESSION_INTERRUPTED) == 1
        assert [event.type for event in events].count(EventType.INTERACTION_INTERRUPTED) == 1

    asyncio.run(scenario())


def test_terminal_interaction_closes_attribution_before_session_finalization() -> None:
    class TerminalHook(RuntimeHook):
        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            await context.emit_custom_event(
                "custom.after-session-completed",
                payload={"session_id": context.session.id},
            )

    async def run() -> None:
        session_id = "sess_terminal_interaction_attribution"
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            runtime_hooks=[TerminalHook()],
            enable_logging=False,
        )
        app.register_provider(CompletingProvider())
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )

        events = await store.load_events(session_id)
        terminal_index = next(
            index
            for index, event in enumerate(events)
            if event.type == EventType.INTERACTION_COMPLETED
        )
        interaction_id = events[terminal_index].interaction_id
        assert interaction_id is not None
        assert all(event.interaction_id is None for event in events[terminal_index + 1 :])

        turn_completed = next(event for event in events if event.type == EventType.TURN_COMPLETED)
        assert turn_completed.payload["interaction_ids"] == [interaction_id]

    asyncio.run(run())


def test_run_stream_carries_interaction_context_across_consumer_tasks() -> None:
    async def run() -> None:
        session_id = "sess_cross_task_interaction_context"
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            )
        )
        events: list[Event] = []
        while True:
            try:
                events.append(await asyncio.create_task(anext(stream)))
            except StopAsyncIteration:
                break
            assert sessions_module._current_session_interaction_id(session_id) is None

        interaction_id = next(
            event.interaction_id for event in events if event.type == EventType.INTERACTION_STARTED
        )
        assert interaction_id is not None
        for event_type in (
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.INTERACTION_COMPLETED,
        ):
            matching = [event for event in events if event.type == event_type]
            assert matching
            assert {event.interaction_id for event in matching} == {interaction_id}
        assert sessions_module._current_session_interaction_id(session_id) is None

    asyncio.run(run())


def test_in_memory_interaction_query_uses_interaction_scoped_candidates() -> None:
    async def run() -> None:
        session_id = "sess_interaction_candidate_index"
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
        interaction_id = next(
            event.interaction_id
            for event in await store.load_events(session_id)
            if event.type == EventType.INTERACTION_STARTED
        )
        assert interaction_id is not None
        await store.append_events(
            session_id,
            [
                Event(
                    id=f"evt_unrelated_{index}",
                    type="custom.unrelated",
                    session_id=session_id,
                    interaction_id="interaction-unrelated",
                )
                for index in range(32)
            ],
        )

        query = EventQuery(session_id=session_id, interaction_id=interaction_id)
        candidates = store._query_candidate_records(query, frozenset())
        records = await store.query_events(query)

        assert candidates
        assert len(candidates) == len(records)
        assert all(record.event.interaction_id == interaction_id for record in candidates)

    asyncio.run(run())


async def _collect_events(app: CayuApp, request: RunRequest) -> list:
    return [event async for event in app.run(request)]
