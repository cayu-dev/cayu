from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import (
    CayuApp,
    Event,
    EventQuery,
    EventType,
    EventWatcher,
    EventWatcherContext,
    EventWatcherDeliveryStatus,
    RunRequest,
    SQLiteEventWatcherStore,
)
from cayu._validation import MAX_DURABLE_JSON_INTEGER, DurableValueError
from cayu.core.events import event_durable_sequence, event_with_durable_sequence
from cayu.runtime import InMemoryEventWatcherStore, InMemorySessionStore
from cayu.runtime._event_projection import REDACTED_CUSTOM_EVENT_TYPE, public_event_id
from cayu.runtime.event_watchers import (
    EventWatcherClaim,
    EventWatcherDelivery,
    EventWatcherRunResult,
    EventWatcherState,
    EventWatcherStore,
    event_query_after_cursor,
)
from cayu.runtime.sessions import EventRecord, SessionIdentity, SessionStore
from cayu.storage.migrations import SchemaMode
from cayu.vaults import REDACTED_SECRET, SecretRedactor

_POSTGRES_TABLES = (
    "cayu_event_watcher_dead_letters",
    "cayu_event_watcher_state",
    "cayu_budget_reservation_identities",
    "cayu_events",
    "cayu_session_labels",
    "cayu_transcript_messages",
    "cayu_session_message_queue",
    "cayu_persisted_event_side_effects",
    "cayu_mcp_manifest_baselines",
    "cayu_checkpoints",
    "cayu_session_operations",
    "cayu_task_terminalization_receipts",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_eval_results",
    "cayu_eval_runs",
    "cayu_eval_cases",
    "cayu_eval_suites",
    "cayu_eval_corpora",
    "cayu_schema_migrations",
)


class CountingSessionStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.query_event_limits: list[int] = []

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        self.query_event_limits.append(EventQuery().limit if query is None else query.limit)
        return await super().query_events(query)


class LegacyEventWatcherStore(EventWatcherStore):
    async def load_state(self, watcher_name: str) -> EventWatcherState:
        raise NotImplementedError

    async def claim_event(
        self,
        *,
        watcher_name: str,
        record: EventRecord,
        lease_seconds: float,
    ) -> EventWatcherClaim | None:
        raise NotImplementedError

    async def claim_next(
        self,
        *,
        watcher_name: str,
        record: EventRecord,
        lease_seconds: float,
    ) -> EventWatcherClaim | None:
        raise NotImplementedError

    async def mark_success(self, claim: EventWatcherClaim) -> EventWatcherDelivery:
        raise NotImplementedError

    async def mark_failure(
        self,
        claim: EventWatcherClaim,
        *,
        error: str,
        max_attempts: int,
    ) -> EventWatcherDelivery:
        raise NotImplementedError


def test_event_watcher_store_dead_letter_methods_are_optional_for_existing_stores() -> None:
    store = LegacyEventWatcherStore()

    with pytest.raises(NotImplementedError, match="dead letters"):
        asyncio.run(store.list_dead_letters("watcher"))

    with pytest.raises(NotImplementedError, match="dead letters"):
        asyncio.run(store.resolve_dead_letter("watcher", 1))


def test_event_watcher_uses_public_projection_while_claim_keeps_private_identity() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "watcherprojection")
        secret = "watcherlegacycanary"
        legacy = Event(
            id=f"private-{secret}-event",
            type="custom.watcherlegacycanary",
            session_id="watcherprojection",
            payload={
                "watcherlegacycanary": secret,
                "safe": "visible",
            },
        )
        await store.append_event(legacy.session_id, legacy)
        records = await store.query_events(EventQuery(session_id=legacy.session_id))
        observed: list[EventWatcherContext] = []
        watcher_store = InMemoryEventWatcherStore()
        app = CayuApp(
            session_store=store,
            event_watcher_store=watcher_store,
            enable_logging=False,
            secret_redactor=SecretRedactor(secret),
        )
        watcher = EventWatcher(
            name="projection-watcher",
            query=EventQuery(session_id=legacy.session_id),
            handler=observed.append,
        )

        results = await app.run_event_watchers([watcher])
        state = await watcher_store.load_state(watcher.name)
        return legacy, records[0], observed[0], results[0], state

    legacy, raw_record, context, result, state = asyncio.run(scenario())

    assert raw_record.event.id == legacy.id
    assert context.record.sequence == raw_record.sequence
    assert context.record.event.id == public_event_id(raw_record.sequence)
    assert context.record.event.type == REDACTED_CUSTOM_EVENT_TYPE
    assert "watcherlegacycanary" not in repr(context.record.event.model_dump(mode="json"))
    assert result.deliveries[0].event_id == public_event_id(raw_record.sequence)
    assert state.cursor_sequence == raw_record.sequence


async def _create_session(store: InMemorySessionStore, session_id: str = "sess_1") -> None:
    await store.create(
        RunRequest(
            session_id=session_id,
            agent_name="assistant",
            environment_name="local-dev",
            messages=[],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )


async def _append_event(
    store: SessionStore,
    *,
    session_id: str = "sess_1",
    event_type: EventType | str = EventType.BUDGET_LIMIT_REACHED,
    agent_name: str = "assistant",
    payload: dict | None = None,
) -> Event:
    event = Event(
        type=event_type,
        session_id=session_id,
        agent_name=agent_name,
        environment_name="local-dev",
        payload={} if payload is None else payload,
    )
    await store.append_event(session_id, event)
    query = EventQuery(session_id=session_id, event_id=event.id, limit=1)
    records = (
        await InMemorySessionStore.query_events(store, query)
        if isinstance(store, InMemorySessionStore)
        else await store.query_events(query)
    )
    assert len(records) == 1
    return event_with_durable_sequence(event, records[0].sequence)


async def _bounded_event_watcher_app() -> tuple[
    str,
    list[EventRecord],
    InMemoryEventWatcherStore,
    CayuApp,
]:
    session_id = "bounded-watcher"
    session_store = InMemorySessionStore()
    await _create_session(session_store, session_id)
    for number in (1, 2, 3):
        await _append_event(
            session_store,
            session_id=session_id,
            payload={"number": number},
        )
    records = await session_store.query_events(
        EventQuery(
            session_id=session_id,
            event_type=EventType.BUDGET_LIMIT_REACHED,
        )
    )
    watcher_store = InMemoryEventWatcherStore()
    app = CayuApp(
        session_store=session_store,
        event_watcher_store=watcher_store,
        enable_logging=False,
    )
    return session_id, records, watcher_store, app


def _public_id(event: Event) -> str:
    sequence = event_durable_sequence(event)
    assert sequence is not None
    return public_event_id(sequence)


async def _assert_portable_event_watcher_store_text(store: EventWatcherStore) -> None:
    for invalid_text, code in (
        ("workload-secret-value\x00", "nul_character"),
        ("workload-secret-value\ud800", "unicode_surrogate"),
    ):
        with pytest.raises(DurableValueError) as invalid_name:
            await store.load_state(invalid_text)
        assert invalid_name.value.code == code
        assert "workload-secret-value" not in str(invalid_name.value)

    record = EventRecord(
        sequence=1,
        event=Event(
            id="portable-watcher-event",
            type=EventType.SESSION_STARTED,
            session_id="sess_portable_watcher",
        ),
    )
    for invalid_lease_seconds in (1e-12, 10**12):
        with pytest.raises(ValueError, match="lease_seconds"):
            await store.claim_event(
                watcher_name="portable-watcher",
                record=record,
                lease_seconds=invalid_lease_seconds,
            )
        assert (await store.load_state("portable-watcher")).cursor_sequence == 0

    forged_record = record.model_copy(deep=True)
    forged_record.sequence = MAX_DURABLE_JSON_INTEGER + 1
    with pytest.raises(ValidationError):
        await store.claim_event(
            watcher_name="portable-watcher",
            record=forged_record,
            lease_seconds=60,
        )
    assert (await store.load_state("portable-watcher")).cursor_sequence == 0

    claim = await store.claim_event(
        watcher_name="portable-watcher",
        record=record,
        lease_seconds=60,
    )
    assert claim is not None
    forged_claim = claim.model_copy(deep=True)
    forged_claim.event_sequence = MAX_DURABLE_JSON_INTEGER + 1
    with pytest.raises(ValidationError):
        await store.mark_success(forged_claim)
    assert (
        await store.load_state("portable-watcher")
    ).delivery_status is EventWatcherDeliveryStatus.LEASED

    for invalid_text, code in (
        ("workload-secret-value\x00", "nul_character"),
        ("workload-secret-value\ud800", "unicode_surrogate"),
    ):
        with pytest.raises(DurableValueError) as invalid_error:
            await store.mark_failure(claim, error=invalid_text, max_attempts=1)
        assert invalid_error.value.code == code
        assert "workload-secret-value" not in str(invalid_error.value)
        state = await store.load_state("portable-watcher")
        assert state.delivery_status is EventWatcherDeliveryStatus.LEASED

    delivery = await store.mark_failure(claim, error="portable failure", max_attempts=1)
    assert delivery.status is EventWatcherDeliveryStatus.DEAD_LETTERED


def test_memory_and_sqlite_event_watcher_stores_reject_nonportable_text(tmp_path: Path) -> None:
    async def run() -> None:
        memory = InMemoryEventWatcherStore()
        await _assert_portable_event_watcher_store_text(memory)

        sqlite = SQLiteEventWatcherStore(tmp_path / "portable-watcher.sqlite")
        try:
            await _assert_portable_event_watcher_store_text(sqlite)
        finally:
            await sqlite.close()

    asyncio.run(run())


def test_event_watcher_handles_matching_events_once() -> None:
    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        first = await _append_event(session_store, payload={"number": 1})
        await _append_event(session_store, event_type=EventType.MODEL_STARTED)
        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.id)

        app = CayuApp(
            session_store=session_store,
            event_watcher_store=InMemoryEventWatcherStore(),
            enable_logging=False,
        )
        watcher = EventWatcher(
            name="budget-email",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
        )

        first_results = await app.run_event_watchers([watcher])
        second_results = await app.run_event_watchers([watcher])
        return first, handled, first_results, second_results

    event, handled, first_results, second_results = asyncio.run(run())
    assert handled == [_public_id(event)]
    assert first_results[0].deliveries[0].status is EventWatcherDeliveryStatus.SUCCEEDED
    assert first_results[0].deliveries[0].event_id == _public_id(event)
    assert second_results[0].deliveries == []


def test_event_watcher_fetches_matching_events_in_batches() -> None:
    async def run():
        session_store = CountingSessionStore()
        await _create_session(session_store)
        events = [
            await _append_event(session_store, payload={"number": number}) for number in range(3)
        ]
        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.id)

        app = CayuApp(
            session_store=session_store,
            event_watcher_store=InMemoryEventWatcherStore(),
            enable_logging=False,
        )
        watcher = EventWatcher(
            name="budget-email",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
            batch_size=3,
        )

        results = await app.run_event_watchers([watcher], limit=3)
        return events, handled, session_store.query_event_limits, results

    events, handled, query_limits, results = asyncio.run(run())
    assert handled == [_public_id(event) for event in events]
    assert query_limits == [3]
    assert [delivery.event_id for delivery in results[0].deliveries] == [
        _public_id(event) for event in events
    ]


def test_event_watcher_cursor_query_preserves_workflow_filters() -> None:
    async def run() -> list[str]:
        session_id = "workflow-watcher"
        store = InMemorySessionStore()
        await _create_session(store, session_id)
        for event_id, interaction_id, step_id in (
            ("workflow-target-1", "target-interaction", "step-a"),
            ("workflow-other-interaction", "other-interaction", "step-a"),
            ("workflow-other-step", "target-interaction", "step-b"),
            ("workflow-target-2", "target-interaction", "step-a"),
        ):
            await store.append_event(
                session_id,
                Event(
                    id=event_id,
                    type=EventType.WORKFLOW_STEP_COMPLETED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                    workflow_name="maintenance",
                    payload={
                        "attempt_id": "attempt-2",
                        "step_id": step_id,
                        "label": event_id,
                    },
                ),
            )

        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.payload["label"])

        app = CayuApp(
            session_store=store,
            event_watcher_store=InMemoryEventWatcherStore(),
            enable_logging=False,
        )
        watcher = EventWatcher(
            name="workflow-step-completed",
            query=EventQuery(
                session_id=session_id,
                interaction_id="target-interaction",
                event_type=EventType.WORKFLOW_STEP_COMPLETED,
                workflow_name="maintenance",
                workflow_attempt_id="attempt-2",
                workflow_step_id="step-a",
            ),
            handler=handler,
        )

        await app.run_event_watchers([watcher], limit=10)
        return handled

    assert asyncio.run(run()) == ["workflow-target-1", "workflow-target-2"]


def test_event_watcher_cursor_query_preserves_event_id_filter() -> None:
    async def run() -> list[int]:
        store = InMemorySessionStore()
        await _create_session(store)
        await _append_event(store, payload={"number": 1})
        target = await _append_event(store, payload={"number": 2})
        handled: list[int] = []

        app = CayuApp(
            session_store=store,
            event_watcher_store=InMemoryEventWatcherStore(),
            enable_logging=False,
        )
        watcher = EventWatcher(
            name="one-event",
            query=EventQuery(session_id=target.session_id, event_id=target.id),
            handler=lambda context: handled.append(context.record.event.payload["number"]),
        )

        await app.run_event_watchers([watcher], limit=10)
        return handled

    assert asyncio.run(run()) == [2]


def test_event_watcher_preserves_before_sequence_across_cursor_pages(monkeypatch) -> None:
    import cayu.runtime.app as app_module

    monkeypatch.setattr(app_module, "EVENT_WATCHER_QUERY_PAGE_LIMIT", 1)

    async def run() -> list[int]:
        session_id, records, _watcher_store, app = await _bounded_event_watcher_app()
        handled: list[int] = []

        watcher = EventWatcher(
            name="bounded-budget-events",
            query=EventQuery(
                session_id=session_id,
                event_type=EventType.BUDGET_LIMIT_REACHED,
                before_sequence=records[2].sequence,
            ),
            handler=lambda context: handled.append(context.record.event.payload["number"]),
            batch_size=3,
        )

        await app.run_event_watchers([watcher], limit=3)
        return handled

    assert asyncio.run(run()) == [1, 2]


def test_event_watcher_stops_when_existing_cursor_reaches_before_sequence() -> None:
    async def run() -> tuple[list[int], EventWatcherRunResult]:
        session_id, records, watcher_store, app = await _bounded_event_watcher_app()

        # Seed the durable state that a pre-repair worker could leave after consuming
        # the exclusive bound that its reconstructed query dropped.
        claim = await watcher_store.claim_event(
            watcher_name="bounded-budget-events",
            record=records[2],
            lease_seconds=300,
        )
        assert claim is not None
        await watcher_store.mark_success(claim)

        handled: list[int] = []
        results = await app.run_event_watchers(
            [
                EventWatcher(
                    name="bounded-budget-events",
                    query=EventQuery(
                        session_id=session_id,
                        event_type=EventType.BUDGET_LIMIT_REACHED,
                        before_sequence=records[2].sequence,
                    ),
                    handler=lambda context: handled.append(context.record.event.payload["number"]),
                    batch_size=3,
                )
            ],
            limit=3,
        )
        return handled, results[0]

    handled, result = asyncio.run(run())
    assert handled == []
    assert result.deliveries == []


def test_event_watcher_large_batch_uses_capped_event_query_pages() -> None:
    async def run():
        session_store = CountingSessionStore()
        await _create_session(session_store)
        events = [
            await _append_event(session_store, payload={"number": number}) for number in range(5001)
        ]
        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.id)

        app = CayuApp(
            session_store=session_store,
            event_watcher_store=InMemoryEventWatcherStore(),
            enable_logging=False,
        )
        watcher = EventWatcher(
            name="budget-email",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
            batch_size=5001,
        )

        results = await app.run_event_watchers([watcher], limit=5001)
        return events, handled, session_store.query_event_limits, results

    events, handled, query_limits, results = asyncio.run(run())
    assert handled == [_public_id(event) for event in events]
    assert query_limits == [5000, 1]
    assert [delivery.event_id for delivery in results[0].deliveries] == [
        _public_id(event) for event in events
    ]


def test_event_watcher_retries_failed_event_before_later_events() -> None:
    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        first = await _append_event(session_store, payload={"number": 1})
        second = await _append_event(session_store, payload={"number": 2})
        seen: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            seen.append(context.record.event.id)
            if context.record.event.id == _public_id(first) and context.attempt == 1:
                raise RuntimeError("temporary email failure")

        app = CayuApp(
            session_store=session_store,
            event_watcher_store=InMemoryEventWatcherStore(),
            enable_logging=False,
        )
        watcher = EventWatcher(
            name="budget-email",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
        )

        failed = await app.run_event_watchers([watcher], limit=10)
        retried = await app.run_event_watchers([watcher], limit=10)
        return first, second, seen, failed, retried

    first, second, seen, failed, retried = asyncio.run(run())
    assert seen == [_public_id(first), _public_id(first), _public_id(second)]
    assert failed[0].deliveries[0].status is EventWatcherDeliveryStatus.FAILED
    assert failed[0].deliveries[0].attempt == 1
    assert [delivery.status for delivery in retried[0].deliveries] == [
        EventWatcherDeliveryStatus.SUCCEEDED,
        EventWatcherDeliveryStatus.SUCCEEDED,
    ]
    assert [delivery.event_id for delivery in retried[0].deliveries] == [
        _public_id(first),
        _public_id(second),
    ]


def test_event_watcher_dead_letters_after_max_attempts_and_unblocks_cursor() -> None:
    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        first = await _append_event(session_store, payload={"number": 1})
        second = await _append_event(session_store, payload={"number": 2})
        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.id)
            if context.record.event.id == _public_id(first):
                raise RuntimeError("permanent webhook failure")

        app = CayuApp(
            session_store=session_store,
            event_watcher_store=InMemoryEventWatcherStore(),
            enable_logging=False,
        )
        watcher = EventWatcher(
            name="budget-webhook",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
            max_attempts=2,
        )

        first_failure = await app.run_event_watchers([watcher], limit=10)
        dead_letter_then_success = await app.run_event_watchers([watcher], limit=10)
        state = await app.event_watcher_store.load_state("budget-webhook")
        return first, second, handled, first_failure, dead_letter_then_success, state

    first, second, handled, first_failure, second_run, state = asyncio.run(run())
    assert handled == [_public_id(first), _public_id(first), _public_id(second)]
    assert first_failure[0].deliveries[0].status is EventWatcherDeliveryStatus.FAILED
    assert [delivery.status for delivery in second_run[0].deliveries] == [
        EventWatcherDeliveryStatus.DEAD_LETTERED,
        EventWatcherDeliveryStatus.SUCCEEDED,
    ]
    assert [delivery.event_id for delivery in second_run[0].deliveries] == [
        _public_id(first),
        _public_id(second),
    ]
    assert state.cursor_sequence == second_run[0].deliveries[-1].event_sequence
    assert state.dead_lettered_count == 1


@pytest.mark.parametrize(
    "rejected_text",
    [
        "watcher failure\u0000with invalid text",
        "watcher failure\ud800with invalid text",
    ],
)
def test_sqlite_watcher_recovers_nonportable_failure_and_unblocks_later_events(
    tmp_path: Path,
    rejected_text: str,
) -> None:
    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        first = await _append_event(session_store, payload={"number": 1})
        second = await _append_event(session_store, payload={"number": 2})
        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.id)
            if context.record.event.id == _public_id(first):
                raise RuntimeError(rejected_text)

        watcher = EventWatcher(
            name="portable-failure-watcher",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
            max_attempts=2,
        )
        db_path = tmp_path / "nonportable-watcher.sqlite"
        first_store = SQLiteEventWatcherStore(db_path)
        first_app = CayuApp(
            session_store=session_store,
            event_watcher_store=first_store,
            enable_logging=False,
        )
        first_result = await first_app.run_event_watchers([watcher], limit=10)
        failed_state = await first_store.load_state(watcher.name)
        await first_store.close()

        second_store = SQLiteEventWatcherStore(db_path)
        second_app = CayuApp(
            session_store=session_store,
            event_watcher_store=second_store,
            enable_logging=False,
        )
        recovered_result = await second_app.run_event_watchers([watcher], limit=10)
        recovered_state = await second_store.load_state(watcher.name)
        dead_letters = await second_store.list_dead_letters(watcher.name)
        await second_store.close()
        return (
            first,
            second,
            handled,
            first_result,
            failed_state,
            recovered_result,
            recovered_state,
            dead_letters,
        )

    (
        first,
        second,
        handled,
        first_result,
        failed_state,
        recovered_result,
        recovered_state,
        dead_letters,
    ) = asyncio.run(run())

    safe_error = "Event watcher failed with a non-portable diagnostic."
    assert handled == [_public_id(first), _public_id(first), _public_id(second)]
    assert first_result[0].deliveries[0].status is EventWatcherDeliveryStatus.FAILED
    assert first_result[0].deliveries[0].error == safe_error
    assert failed_state.delivery_status is EventWatcherDeliveryStatus.FAILED
    assert failed_state.pending_claim_id is None
    assert failed_state.lease_expires_at is None
    assert [delivery.status for delivery in recovered_result[0].deliveries] == [
        EventWatcherDeliveryStatus.DEAD_LETTERED,
        EventWatcherDeliveryStatus.SUCCEEDED,
    ]
    assert [delivery.event_id for delivery in recovered_result[0].deliveries] == [
        _public_id(first),
        _public_id(second),
    ]
    assert recovered_state.cursor_sequence == recovered_result[0].deliveries[-1].event_sequence
    assert recovered_state.dead_lettered_count == 1
    assert len(dead_letters) == 1
    assert dead_letters[0].error == safe_error


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_event_watcher_redacts_failure_before_all_durable_representations(
    store_kind: str,
    tmp_path: Path,
) -> None:
    secret = "event-watcher-diagnostic-secret-canary"

    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        event = Event(
            id=f"legacy-{secret}-event",
            type=EventType.BUDGET_LIMIT_REACHED,
            session_id="sess_1",
            payload={"number": 1},
        )
        await session_store.append_event(event.session_id, event)
        records = await session_store.query_events(EventQuery(session_id=event.session_id))
        assert len(records) == 1
        watcher_store = (
            InMemoryEventWatcherStore()
            if store_kind == "memory"
            else SQLiteEventWatcherStore(tmp_path / "watcher-redaction.sqlite")
        )
        app = CayuApp(
            session_store=session_store,
            event_watcher_store=watcher_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        async def handler(_context: EventWatcherContext) -> None:
            raise RuntimeError(f"delivery failed with {secret}")

        watcher = EventWatcher(
            name="redacted-watcher",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
            max_attempts=2,
        )
        failed_results = await app.run_event_watchers([watcher])
        dead_letter_results = await app.run_event_watchers([watcher])
        state = await watcher_store.load_state(watcher.name)
        dead_letters = await watcher_store.list_dead_letters(watcher.name)
        if isinstance(watcher_store, SQLiteEventWatcherStore):
            await watcher_store.close()
        return records[0], failed_results, dead_letter_results, state, dead_letters

    raw_record, failed_results, dead_letter_results, state, dead_letters = asyncio.run(run())
    public_id = public_event_id(raw_record.sequence)

    assert failed_results[0].deliveries[0].event_id == public_id
    assert failed_results[0].deliveries[0].status is EventWatcherDeliveryStatus.FAILED
    assert dead_letter_results[0].deliveries[0].event_id == public_id
    assert dead_letter_results[0].deliveries[0].status is EventWatcherDeliveryStatus.DEAD_LETTERED
    assert secret not in repr((failed_results, dead_letter_results, state))
    assert REDACTED_SECRET in repr((failed_results, dead_letter_results, state))
    assert len(dead_letters) == 1
    assert dead_letters[0].event_id == raw_record.event.id
    assert secret not in dead_letters[0].error


def test_event_watcher_records_handler_error_with_broken_stringification() -> None:
    class BrokenStringError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("stringification failed")

    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        await _append_event(session_store)
        watcher_store = InMemoryEventWatcherStore()
        app = CayuApp(
            session_store=session_store,
            event_watcher_store=watcher_store,
            enable_logging=False,
        )

        async def fail(_context: EventWatcherContext) -> None:
            raise BrokenStringError

        watcher = EventWatcher(
            name="broken-string-watcher",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=fail,
        )
        results = await app.run_event_watchers([watcher], limit=1)
        state = await watcher_store.load_state(watcher.name)
        return results, state

    results, state = asyncio.run(run())

    assert results[0].deliveries[0].status is EventWatcherDeliveryStatus.FAILED
    assert results[0].deliveries[0].error == "BrokenStringError: event watcher failed"
    assert state.last_error == "BrokenStringError: event watcher failed"


def test_event_watcher_cancellation_during_failure_storage_drops_handler_error() -> None:
    secret = "event-watcher-cancelled-publication-secret-canary"

    class BlockingFailureStore(InMemoryEventWatcherStore):
        def __init__(self) -> None:
            super().__init__()
            self.failure_started = asyncio.Event()

        async def mark_failure(
            self,
            claim: EventWatcherClaim,
            *,
            error: str,
            max_attempts: int,
        ) -> EventWatcherDelivery:
            assert secret not in error
            self.failure_started.set()
            await asyncio.Event().wait()
            return await super().mark_failure(
                claim,
                error=error,
                max_attempts=max_attempts,
            )

    async def run() -> tuple[asyncio.CancelledError, int, bool]:
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        await _append_event(session_store)
        watcher_store = BlockingFailureStore()
        app = CayuApp(
            session_store=session_store,
            event_watcher_store=watcher_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        async def fail(_context: EventWatcherContext) -> None:
            raise RuntimeError(f"delivery failed with {secret}")

        task = asyncio.create_task(
            app.run_event_watchers(
                [
                    EventWatcher(
                        name="cancelled-publication-watcher",
                        query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
                        handler=fail,
                    )
                ],
                limit=1,
            )
        )
        await watcher_store.failure_started.wait()
        task.cancel("operator cancelled")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(run())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("operator cancelled",)
    assert cancellation.__context__ is None
    traceback = cancellation.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next


def test_event_watcher_rejects_secret_bearing_watcher_authority() -> None:
    secret = "event-watcher-authority-secret-canary"
    app = CayuApp(secret_redactor=SecretRedactor(secret), enable_logging=False)
    watcher = EventWatcher(
        name=f"watcher-{secret}",
        query=EventQuery(),
        handler=lambda _context: None,
    )

    with pytest.raises(ValueError, match="durable watcher authority"):
        asyncio.run(app.run_event_watchers([watcher]))

    with pytest.raises(ValueError, match="Duplicate event watcher name") as exc_info:
        asyncio.run(app.run_event_watchers([watcher, watcher]))
    assert secret not in str(exc_info.value)


def test_event_watcher_active_lease_blocks_duplicate_processing() -> None:
    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        record_event = await _append_event(session_store)
        store = InMemoryEventWatcherStore()
        records = await session_store.query_events(EventQuery(limit=1))
        first_claim = await store.claim_event(
            watcher_name="budget-email",
            record=records[0],
            lease_seconds=300,
        )
        second_claim = await store.claim_event(
            watcher_name="budget-email",
            record=records[0],
            lease_seconds=300,
        )
        return record_event, first_claim, second_claim

    event, first_claim, second_claim = asyncio.run(run())
    assert first_claim is not None
    assert first_claim.event_id == event.id
    assert second_claim is None


def test_event_watcher_expired_lease_can_be_reclaimed() -> None:
    async def run():
        now = {"value": datetime(2026, 1, 1, tzinfo=UTC)}

        def clock() -> datetime:
            return now["value"]

        session_store = InMemorySessionStore()
        await _create_session(session_store)
        record_event = await _append_event(session_store)
        store = InMemoryEventWatcherStore(clock=clock)
        records = await session_store.query_events(EventQuery(limit=1))
        first_claim = await store.claim_event(
            watcher_name="budget-email",
            record=records[0],
            lease_seconds=10,
        )

        now["value"] = now["value"] + timedelta(seconds=11)
        second_claim = await store.claim_event(
            watcher_name="budget-email",
            record=records[0],
            lease_seconds=10,
        )
        return record_event, first_claim, second_claim

    event, first_claim, second_claim = asyncio.run(run())
    assert first_claim is not None
    assert first_claim.event_id == event.id
    assert second_claim is not None
    assert second_claim.event_id == event.id
    assert second_claim.attempt == 2
    assert second_claim.claim_id != first_claim.claim_id


def test_sqlite_event_watcher_store_persists_cursor(tmp_path: Path) -> None:
    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        first = await _append_event(session_store, payload={"number": 1})
        second = await _append_event(session_store, payload={"number": 2})
        db_path = tmp_path / "watchers.sqlite"
        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.id)

        watcher = EventWatcher(
            name="budget-email",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
        )
        first_store = SQLiteEventWatcherStore(db_path)
        first_app = CayuApp(
            session_store=session_store,
            event_watcher_store=first_store,
            enable_logging=False,
        )
        first_result = await first_app.run_event_watchers([watcher], limit=1)
        await first_store.close()

        second_store = SQLiteEventWatcherStore(db_path)
        second_app = CayuApp(
            session_store=session_store,
            event_watcher_store=second_store,
            enable_logging=False,
        )
        second_result = await second_app.run_event_watchers([watcher], limit=10)
        state = await second_store.load_state("budget-email")
        await second_store.close()
        return first, second, handled, first_result, second_result, state

    first, second, handled, first_result, second_result, state = asyncio.run(run())
    assert handled == [_public_id(first), _public_id(second)]
    assert [delivery.event_id for delivery in first_result[0].deliveries] == [_public_id(first)]
    assert [delivery.event_id for delivery in second_result[0].deliveries] == [_public_id(second)]
    assert state.cursor_sequence == second_result[0].deliveries[-1].event_sequence


async def _drop_postgres_tables(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _POSTGRES_TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


def test_postgres_event_watcher_store_rejects_nonportable_text(postgres_dsn: str) -> None:
    async def run() -> None:
        from cayu import PostgresEventWatcherStore

        await _drop_postgres_tables(postgres_dsn)
        store = PostgresEventWatcherStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await _assert_portable_event_watcher_store_text(store)
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_event_watcher_store_persists_cursor(postgres_dsn: str) -> None:
    async def run():
        from cayu import PostgresEventWatcherStore

        await _drop_postgres_tables(postgres_dsn)
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        first = await _append_event(session_store, payload={"number": 1})
        second = await _append_event(session_store, payload={"number": 2})
        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.id)

        watcher = EventWatcher(
            name="budget-email",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
        )
        first_store = PostgresEventWatcherStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            first_app = CayuApp(
                session_store=session_store,
                event_watcher_store=first_store,
                enable_logging=False,
            )
            first_result = await first_app.run_event_watchers([watcher], limit=1)
        finally:
            await first_store.close()

        second_store = PostgresEventWatcherStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            second_app = CayuApp(
                session_store=session_store,
                event_watcher_store=second_store,
                enable_logging=False,
            )
            second_result = await second_app.run_event_watchers([watcher], limit=10)
            state = await second_store.load_state("budget-email")
        finally:
            await second_store.close()
        return first, second, handled, first_result, second_result, state

    first, second, handled, first_result, second_result, state = asyncio.run(run())
    assert handled == [_public_id(first), _public_id(second)]
    assert [delivery.event_id for delivery in first_result[0].deliveries] == [_public_id(first)]
    assert [delivery.event_id for delivery in second_result[0].deliveries] == [_public_id(second)]
    assert state.cursor_sequence == second_result[0].deliveries[-1].event_sequence


def test_postgres_event_watcher_store_serializes_first_claim(postgres_dsn: str) -> None:
    async def run():
        from cayu import PostgresEventWatcherStore

        await _drop_postgres_tables(postgres_dsn)
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        event = await _append_event(session_store)
        records = await session_store.query_events(EventQuery(limit=1))
        setup_store = PostgresEventWatcherStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        await setup_store.load_state("schema-ready")
        await setup_store.close()

        first_store = PostgresEventWatcherStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        second_store = PostgresEventWatcherStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            claim_results = await asyncio.gather(
                first_store.claim_event(
                    watcher_name="budget-email",
                    record=records[0],
                    lease_seconds=300,
                ),
                second_store.claim_event(
                    watcher_name="budget-email",
                    record=records[0],
                    lease_seconds=300,
                ),
            )
        finally:
            await first_store.close()
            await second_store.close()
        return event, claim_results

    event, claim_results = asyncio.run(run())
    claims = [claim for claim in claim_results if claim is not None]
    blocked = [claim for claim in claim_results if claim is None]
    assert len(claims) == 1
    assert len(blocked) == 1
    assert claims[0].event_id == event.id


def test_event_watcher_rejects_cursor_in_query() -> None:
    with pytest.raises(ValueError, match="after_sequence"):
        EventWatcher(
            name="invalid",
            query=EventQuery(after_sequence=10),
            handler=lambda _context: None,
        )


@pytest.mark.parametrize(
    "lease_seconds",
    [float("nan"), float("inf"), float("-inf"), 1e-12, 10**12, 1e308, 10**400],
)
def test_event_watcher_rejects_invalid_lease_seconds(lease_seconds: int | float) -> None:
    with pytest.raises(ValueError, match="lease_seconds"):
        EventWatcher(
            name="invalid-lease",
            query=EventQuery(),
            handler=lambda _context: None,
            lease_seconds=lease_seconds,
        )


def test_inmemory_event_watcher_store_rejects_zero_length_positive_lease() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = InMemoryEventWatcherStore(clock=lambda: now)
    record = EventRecord(
        sequence=1,
        event=Event(
            id="submicrosecond-lease-event",
            type=EventType.SESSION_STARTED,
            session_id="submicrosecond-lease-session",
        ),
    )

    async def scenario() -> None:
        with pytest.raises(ValueError, match="lease_seconds"):
            await store.claim_event(
                watcher_name="submicrosecond-lease-watcher",
                record=record,
                lease_seconds=1e-12,
            )

    asyncio.run(scenario())


def test_event_watcher_cursor_reconstruction_revalidates_the_new_cursor() -> None:
    with pytest.raises(ValidationError, match="after_sequence"):
        event_query_after_cursor(EventQuery(), -1)


async def _dead_letter_first_event(app: CayuApp) -> tuple[Event, Event]:
    session_store = app.session_store
    first = await _append_event(session_store, payload={"number": 1})
    second = await _append_event(session_store, payload={"number": 2})

    async def handler(context: EventWatcherContext) -> None:
        if context.record.event.id == _public_id(first):
            raise RuntimeError("permanent webhook failure")

    watcher = EventWatcher(
        name="budget-webhook",
        query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
        handler=handler,
        max_attempts=1,
    )
    await app.run_event_watchers([watcher], limit=10)
    return first, second


def test_inmemory_event_watcher_store_persists_and_resolves_dead_letters() -> None:
    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        store = InMemoryEventWatcherStore()
        app = CayuApp(
            session_store=session_store,
            event_watcher_store=store,
            enable_logging=False,
        )
        first, _second = await _dead_letter_first_event(app)
        unresolved = await store.list_dead_letters("budget-webhook")
        resolved_record = await store.resolve_dead_letter(
            "budget-webhook", unresolved[0].event_sequence
        )
        after_resolve = await store.list_dead_letters("budget-webhook")
        including = await store.list_dead_letters("budget-webhook", include_resolved=True)
        return first, unresolved, resolved_record, after_resolve, including

    first, unresolved, resolved_record, after_resolve, including = asyncio.run(run())
    assert [record.event_id for record in unresolved] == [first.id]
    assert unresolved[0].watcher_name == "budget-webhook"
    assert unresolved[0].attempts == 1
    assert unresolved[0].error == "permanent webhook failure"
    assert unresolved[0].resolved_at is None
    assert resolved_record.resolved_at is not None
    # A resolved record drops out of the default listing but is still retrievable.
    assert after_resolve == []
    assert [record.event_id for record in including] == [first.id]
    assert including[0].resolved_at is not None


def test_inmemory_event_watcher_store_resolve_missing_dead_letter_raises() -> None:
    async def run():
        store = InMemoryEventWatcherStore()
        await store.resolve_dead_letter("budget-webhook", 7)

    with pytest.raises(ValueError, match="No dead-letter record"):
        asyncio.run(run())


def test_sqlite_event_watcher_store_persists_and_resolves_dead_letters(tmp_path: Path) -> None:
    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        db_path = tmp_path / "dead_letters.sqlite"
        store = SQLiteEventWatcherStore(db_path)
        app = CayuApp(
            session_store=session_store,
            event_watcher_store=store,
            enable_logging=False,
        )
        first, _second = await _dead_letter_first_event(app)
        first_listing = await store.list_dead_letters("budget-webhook")
        await store.close()

        # Records survive a store reopen — they are durable, not in-process state.
        reopened = SQLiteEventWatcherStore(db_path)
        after_reopen = await reopened.list_dead_letters("budget-webhook")
        resolved_record = await reopened.resolve_dead_letter(
            "budget-webhook", after_reopen[0].event_sequence
        )
        # Resolving is idempotent — the second call keeps the first resolved_at.
        resolved_again = await reopened.resolve_dead_letter(
            "budget-webhook", after_reopen[0].event_sequence
        )
        default_after = await reopened.list_dead_letters("budget-webhook")
        including = await reopened.list_dead_letters("budget-webhook", include_resolved=True)
        with pytest.raises(ValueError, match="No dead-letter record"):
            await reopened.resolve_dead_letter("budget-webhook", 999)
        await reopened.close()
        return (
            first,
            first_listing,
            after_reopen,
            resolved_record,
            resolved_again,
            default_after,
            including,
        )

    (
        first,
        first_listing,
        after_reopen,
        resolved_record,
        resolved_again,
        default_after,
        including,
    ) = asyncio.run(run())
    assert [record.event_id for record in first_listing] == [first.id]
    assert [record.event_id for record in after_reopen] == [first.id]
    assert after_reopen[0].attempts == 1
    assert after_reopen[0].error == "permanent webhook failure"
    assert after_reopen[0].resolved_at is None
    assert resolved_record.resolved_at is not None
    assert resolved_again.resolved_at == resolved_record.resolved_at
    assert default_after == []
    assert [record.event_id for record in including] == [first.id]
