"""Shared session-store queries contract cases."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _identity,
    _open_store,
    _reopen_store,
)
from tests.core.transcript_search_conformance import (
    assert_transcript_search_conformance,
)

from cayu.events import (
    Event,
    EventType,
)
from cayu.messages import (
    Message,
)
from cayu.sessions import _terminal_evidence as terminal_event_evidence
from cayu.sessions._terminal_evidence import (
    QueuedDispatchTerminalReceiptQuery,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionQuery
from cayu.sessions.requests import (
    RunRequest,
)
from cayu.sessions.transcript_queries import TranscriptQuery

pytestmark = pytest.mark.postgres


def test_session_store_transcript_search_conformance(session_store_case) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            await assert_transcript_search_conformance(store)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_lists_queued_dispatch_terminal_receipts(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            for session_id in (
                "queued-receipt-a",
                "queued-receipt-b",
                "queued-receipt-unrelated",
            ):
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[],
                    ),
                    identity=_identity(),
                )
            await store.append_event(
                "queued-receipt-a",
                Event(
                    id="terminal-a",
                    type=EventType.SESSION_COMPLETED,
                    session_id="queued-receipt-a",
                    payload={"session_run_operation_id": "operation-a"},
                ),
            )
            await store.checkpoint(
                "queued-receipt-a",
                {
                    "session_run_operation": {
                        "version": 1,
                        "operation_id": "operation-a",
                        "run_epoch": 1,
                        "terminal_event_id": "terminal-a",
                        "queue_task_id": "queue-a",
                    }
                },
            )
            await store.checkpoint(
                "queued-receipt-b",
                {
                    "queued_dispatch_terminal_receipts": {
                        "version": 1,
                        "receipts": {
                            "operation-b-1": {
                                "queue_task_id": "queue-b-1",
                                "terminal_event_id": "terminal-b-1",
                                "run_epoch": 1,
                            },
                            "operation-b-2": {
                                "queue_task_id": "queue-b-2",
                                "terminal_event_id": "terminal-b-2",
                                "run_epoch": 1,
                            },
                        },
                    }
                },
            )
            await store.checkpoint(
                "queued-receipt-unrelated",
                {"unrelated": True},
            )

            first_page = await store.list_queued_dispatch_terminal_receipts(
                QueuedDispatchTerminalReceiptQuery(limit=2)
            )
            assert [(receipt.session_id, receipt.operation_id) for receipt in first_page] == [
                ("queued-receipt-a", "operation-a"),
                ("queued-receipt-b", "operation-b-1"),
            ]

            store = await _reopen_store(session_store_case, store)
            second_page = await store.list_queued_dispatch_terminal_receipts(
                QueuedDispatchTerminalReceiptQuery(
                    after_session_id=first_page[-1].session_id,
                    after_operation_id=first_page[-1].operation_id,
                    limit=2,
                )
            )
            assert [receipt.model_dump(mode="json") for receipt in second_page] == [
                {
                    "session_id": "queued-receipt-b",
                    "queue_task_id": "queue-b-2",
                    "operation_id": "operation-b-2",
                    "terminal_event_id": "terminal-b-2",
                }
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_receipt_query_does_not_serialize_mutated_input(
    session_store_case,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    canary = "mutated-receipt-query-diagnostic-canary"

    class Canary:
        def __repr__(self) -> str:
            return canary

        def __str__(self) -> str:
            return canary

    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            query = QueuedDispatchTerminalReceiptQuery()
            object.__setattr__(query, "limit", Canary())
            with pytest.raises(ValueError):
                await store.list_queued_dispatch_terminal_receipts(query)
        finally:
            await _close_store(store)

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out
    assert canary not in captured.err
    assert all(canary not in str(record.message) for record in recwarn)


def test_in_memory_queued_dispatch_receipt_query_uses_live_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "queued-receipt-indexed"
        await store.create(
            RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
            identity=_identity(),
        )
        await store.checkpoint(
            session_id,
            {
                "session_run_operation": {
                    "version": 1,
                    "operation_id": "operation-indexed",
                    "run_epoch": 1,
                    "terminal_event_id": "terminal-indexed",
                    "queue_task_id": "queue-indexed",
                }
            },
        )
        assert await store.list_queued_dispatch_terminal_receipts() == []
        await store.append_event(
            session_id,
            Event(
                id="terminal-indexed",
                type=EventType.SESSION_COMPLETED,
                session_id=session_id,
                payload={"session_run_operation_id": "operation-indexed"},
            ),
        )
        with pytest.raises(ValueError, match="receipt checkpoint is invalid"):
            await store.checkpoint(
                session_id,
                {
                    "queued_dispatch_terminal_receipts": {
                        "version": 2,
                        "receipts": {},
                    }
                },
            )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["session_run_operation"]["operation_id"] == "operation-indexed"

        def reject_history_projection(*args, **kwargs):
            del args, kwargs
            raise AssertionError("receipt query rescanned checkpoint/event history")

        monkeypatch.setattr(
            terminal_event_evidence,
            "_queued_dispatch_terminal_receipts_for_session",
            reject_history_projection,
        )
        receipts = await store.list_queued_dispatch_terminal_receipts()
        assert [(receipt.session_id, receipt.operation_id) for receipt in receipts] == [
            (session_id, "operation-indexed")
        ]

    asyncio.run(run())


def test_session_store_conformance_declares_usage_aggregate_support(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            assert store.supports_usage_aggregates is True
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_applies_query_filters(session_store_case) -> None:
    async def run() -> None:
        session_store = await _open_store(session_store_case)
        try:
            await session_store.create(
                RunRequest(
                    agent_name="alpha",
                    session_id="sess_query_alpha",
                    causal_budget_id="budget_runtime",
                    environment_name="local",
                    labels={"team": "runtime"},
                    messages=[Message.text("user", "alpha")],
                ),
                identity=_identity(),
            )
            await session_store.create(
                RunRequest(
                    agent_name="beta",
                    session_id="sess_query_beta",
                    causal_budget_id="budget_runtime",
                    environment_name="remote",
                    labels={"team": "review"},
                    messages=[Message.text("user", "beta")],
                ),
                identity=_identity(),
            )
            await session_store.append_events(
                "sess_query_alpha",
                [
                    Event(
                        id="evt_query_alpha",
                        type=EventType.TOOL_CALL_COMPLETED,
                        session_id="sess_query_alpha",
                        agent_name="alpha",
                        environment_name="local",
                        tool_name="read_file",
                        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                    )
                ],
            )
            await session_store.append_events(
                "sess_query_beta",
                [
                    Event(
                        id="evt_query_beta",
                        type=EventType.TOOL_CALL_FAILED,
                        session_id="sess_query_beta",
                        agent_name="beta",
                        environment_name="remote",
                        tool_name="edit_file",
                        timestamp=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
                    )
                ],
            )

            sessions = await session_store.list_sessions(
                SessionQuery(q="ALPHA", labels={"team": "runtime"}, include_total_count=True)
            )
            assert [session.id for session in sessions.sessions] == ["sess_query_alpha"]
            assert sessions.total_count == 1

            records = await session_store.query_events(
                EventQuery(
                    causal_budget_id="budget_runtime",
                    event_types=(EventType.TOOL_CALL_COMPLETED,),
                    agent_name="alpha",
                    tool_name="read_file",
                )
            )
            assert [record.event.id for record in records] == ["evt_query_alpha"]
        finally:
            await _close_store(session_store)

    asyncio.run(run())


def test_session_store_conformance_preserves_interaction_attribution(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interaction_attribution",
                    messages=[Message.text("user", "bootstrap")],
                ),
                identity=_identity(),
            )
            first_id = "interaction-first"
            second_id = "interaction-second"
            await store.append_events(
                session.id,
                [
                    Event(
                        id="evt_interaction_first",
                        type=EventType.MODEL_STARTED,
                        session_id=session.id,
                        interaction_id=first_id,
                    ),
                    Event(
                        id="evt_interaction_legacy",
                        type="custom.legacy",
                        session_id=session.id,
                    ),
                    Event(
                        id="evt_interaction_second",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session.id,
                        interaction_id=second_id,
                    ),
                ],
            )
            await store.append_transcript_messages(
                session.id,
                [Message.text("user", "first"), Message.text("assistant", "one")],
                interaction_id=first_id,
            )
            await store.append_transcript_messages(
                session.id,
                [Message.text("user", "legacy")],
            )
            await store.append_transcript_messages(
                session.id,
                [Message.text("user", "second"), Message.text("assistant", "two")],
                interaction_id=second_id,
            )

            store = await _reopen_store(session_store_case, store)
            first_events = await store.query_events(
                EventQuery(session_id=session.id, interaction_id=first_id)
            )
            assert [(record.event.id, record.event.interaction_id) for record in first_events] == [
                ("evt_interaction_first", first_id)
            ]
            all_events = await store.query_events(EventQuery(session_id=session.id))
            assert [record.event.interaction_id for record in all_events] == [
                first_id,
                None,
                second_id,
            ]

            first_transcript = await store.query_transcript(
                TranscriptQuery(session_id=session.id, interaction_id=first_id)
            )
            assert [record.index for record in first_transcript.records] == [0, 1]
            assert [record.interaction_id for record in first_transcript.records] == [
                first_id,
                first_id,
            ]
            assert first_transcript.total_records == 2

            all_transcript = await store.query_transcript(
                TranscriptQuery(session_id=session.id, limit=10)
            )
            assert [record.index for record in all_transcript.records] == [0, 1, 2, 3, 4]
            assert [record.interaction_id for record in all_transcript.records] == [
                first_id,
                first_id,
                None,
                second_id,
                second_id,
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_paginates_latest_interaction_states(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interaction_pagination",
                    messages=[Message.text("user", "bootstrap")],
                ),
                identity=_identity(),
            )
            await store.append_events(
                session.id,
                [
                    Event(
                        id="interaction_1_started",
                        type=EventType.INTERACTION_STARTED,
                        session_id=session.id,
                        interaction_id="interaction-1",
                    ),
                    Event(
                        id="interaction_2_started",
                        type=EventType.INTERACTION_STARTED,
                        session_id=session.id,
                        interaction_id="interaction-2",
                    ),
                    Event(
                        id="interaction_1_paused",
                        type=EventType.INTERACTION_PAUSED,
                        session_id=session.id,
                        interaction_id="interaction-1",
                    ),
                    Event(
                        id="interaction_3_started",
                        type=EventType.INTERACTION_STARTED,
                        session_id=session.id,
                        interaction_id="interaction-3",
                    ),
                    Event(
                        id="interaction_2_completed",
                        type=EventType.INTERACTION_COMPLETED,
                        session_id=session.id,
                        interaction_id="interaction-2",
                    ),
                ],
            )

            first_page = await store.query_latest_interaction_events(session.id, limit=2)
            assert [record.event.id for record in first_page] == [
                "interaction_2_completed",
                "interaction_3_started",
            ]
            second_page = await store.query_latest_interaction_events(
                session.id,
                before_sequence=first_page[-1].sequence,
                limit=2,
            )
            assert [record.event.id for record in second_page] == ["interaction_1_paused"]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_pages_hundreds_of_interactions(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interaction_scale",
                    messages=[Message.text("user", "bootstrap")],
                ),
                identity=_identity(),
            )
            interaction_count = 250
            await store.append_events(
                session.id,
                [
                    Event(
                        id=f"interaction_{index}_completed",
                        type=EventType.INTERACTION_COMPLETED,
                        session_id=session.id,
                        interaction_id=f"interaction-{index}",
                    )
                    for index in range(interaction_count)
                ],
            )

            observed: list[str] = []
            before_sequence = None
            while True:
                page = await store.query_latest_interaction_events(
                    session.id,
                    before_sequence=before_sequence,
                    limit=37,
                )
                if not page:
                    break
                observed.extend(record.event.interaction_id or "" for record in page)
                before_sequence = page[-1].sequence

            assert observed == [
                f"interaction-{index}" for index in reversed(range(interaction_count))
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_distinguishes_inherited_and_explicit_null_transcript_ids(
    session_store_case,
) -> None:
    from cayu.sessions._invocation_scope import (
        _activate_session_interaction,
        _deactivate_session_interaction,
    )

    async def run() -> None:
        store = await _open_store(session_store_case)
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_transcript_attribution_modes",
                messages=[Message.text("user", "bootstrap")],
            ),
            identity=_identity(),
        )
        try:
            _activate_session_interaction(session.id, "interaction-active")
            await store.append_transcript_messages(
                session.id,
                [Message.text("user", "inherit")],
            )
            await store.append_transcript_messages(
                session.id,
                [Message.text("assistant", "unassociated")],
                interaction_id=None,
            )
            page = await store.query_transcript(TranscriptQuery(session_id=session.id))
            assert [record.interaction_id for record in page.records] == [
                "interaction-active",
                None,
            ]
        finally:
            _deactivate_session_interaction(session.id)
            await _close_store(store)

    asyncio.run(run())
