from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError
from tests.core._session_store_test_doubles import RecordingListSessionsStore

import cayu.runtime.sessions as sessions_module
from cayu.core import Event, EventType, Message
from cayu.runtime import (
    CayuApp,
    EventOrder,
    EventQuery,
    EventRecord,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    IncompleteSessionsRecoveryRequest,
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
    SessionListResult,
    SessionQuery,
    SessionStatus,
)


def test_committed_recovery_survives_bounded_public_linkage_lookup_miss() -> None:
    class BoundedHistoryStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.linkage_queries: list[EventQuery] = []

        async def query_events(self, query: EventQuery) -> list[EventRecord]:
            if query.order_by is EventOrder.SEQUENCE_DESC and query.limit == 5000:
                self.linkage_queries.append(query)
                return [
                    EventRecord(
                        sequence=index + 2,
                        event=Event(
                            type=EventType.HOOK_STARTED,
                            session_id="recovered-session",
                            payload={"index": index},
                        ),
                    )
                    for index in range(5000)
                ]
            return await super().query_events(query)

    async def scenario() -> None:
        store = BoundedHistoryStore()
        app = CayuApp(session_store=store, enable_logging=False)

        async def committed_recovery(
            request: IncompleteSessionRecoveryRequest,
        ) -> IncompleteSessionRecoveryResult:
            assert request.session_id == "recovered-session"
            return IncompleteSessionRecoveryResult(
                session_id="recovered-session",
                previous_status=SessionStatus.RUNNING,
                status=SessionStatus.INTERRUPTED,
                actions=(IncompleteSessionRecoveryAction.PENDING_APPROVAL,),
                pending_approval_id="private-approval-before-bounded-window",
                message="Recovered incomplete session.",
            )

        app._recover_incomplete_session_private = committed_recovery  # type: ignore[method-assign]
        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="recovered-session")
        )

        assert result.status is SessionStatus.INTERRUPTED
        assert result.actions == (IncompleteSessionRecoveryAction.PENDING_APPROVAL,)
        assert result.pending_approval_id is None
        assert "Public linkage unavailable for: pending_approval_id" in result.message
        assert len(store.linkage_queries) == 1

    asyncio.run(scenario())


def test_committed_recovery_survives_public_linkage_lookup_failure() -> None:
    class FailingHistoryStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def query_events(self, query: EventQuery) -> list[EventRecord]:
            if query.order_by is EventOrder.SEQUENCE_DESC and query.limit == 5000:
                raise OSError("history unavailable")
            return await super().query_events(query)

    async def scenario() -> None:
        store = FailingHistoryStore()
        app = CayuApp(session_store=store, enable_logging=False)

        async def committed_recovery(
            request: IncompleteSessionRecoveryRequest,
        ) -> IncompleteSessionRecoveryResult:
            return IncompleteSessionRecoveryResult(
                session_id=request.session_id,
                previous_status=SessionStatus.RUNNING,
                status=SessionStatus.INTERRUPTED,
                actions=(IncompleteSessionRecoveryAction.PENDING_APPROVAL,),
                pending_approval_id="private-approval",
                message="Recovered incomplete session.",
            )

        app._recover_incomplete_session_private = committed_recovery  # type: ignore[method-assign]
        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="recovered-session")
        )

        assert result.status is SessionStatus.INTERRUPTED
        assert result.pending_approval_id is None
        assert "Public linkage unavailable for: pending_approval_id" in result.message

    asyncio.run(scenario())


def test_incomplete_sessions_recovery_request_rejects_empty_statuses():
    with pytest.raises(ValidationError, match="must not be empty"):
        IncompleteSessionsRecoveryRequest(statuses=set(), limit=10)


def test_incomplete_sessions_recovery_request_accepts_all_runtime_statuses():
    statuses = set(SessionStatus)
    request = IncompleteSessionsRecoveryRequest(statuses=statuses, limit=10)

    assert request.statuses == statuses


def test_cayu_app_recover_incomplete_sessions_repairs_requested_terminal_statuses() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        for status in (
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        ):
            session_id = f"sess_batch_terminal_{status.value}"
            await store.create(
                RunRequest(
                    agent_name="removed_agent",
                    session_id=session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status(session_id, status)

        page = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={
                    SessionStatus.COMPLETED,
                    SessionStatus.FAILED,
                    SessionStatus.INTERRUPTED,
                },
                limit=10,
            )
        )
        results = page.results
        assert {result.session_id for result in results} == {
            "sess_batch_terminal_completed",
            "sess_batch_terminal_failed",
            "sess_batch_terminal_interrupted",
        }
        assert all(
            result.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
            for result in results
        )

    asyncio.run(scenario())


def test_cayu_app_terminal_batch_pages_past_healthy_inspection_candidates() -> None:
    async def scenario() -> None:
        store = RecordingListSessionsStore()
        app = CayuApp(session_store=store, enable_logging=False)
        damaged_session_id = "sess_batch_terminal_damaged_old"
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=damaged_session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(damaged_session_id, SessionStatus.COMPLETED)

        for index in range(1000):
            session_id = f"sess_batch_terminal_healthy_{index:03d}"
            await store.create(
                RunRequest(
                    agent_name="removed_agent",
                    session_id=session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            await store.append_event(
                session_id,
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                ),
            )

        first_page = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.COMPLETED},
                limit=1000,
            )
        )
        assert first_page.results == ()
        assert first_page.inspected_session_count == 1000
        assert first_page.next_cursor is not None

        second_page = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.COMPLETED},
                limit=1000,
                cursor=first_page.next_cursor,
            )
        )
        repaired_records = await store.query_events(
            EventQuery(
                session_id=damaged_session_id,
                event_type=EventType.SESSION_COMPLETED,
            )
        )
        assert [result.session_id for result in second_page.results] == [damaged_session_id]
        assert second_page.results[0].actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
        )
        assert second_page.inspected_session_count == 1
        assert second_page.next_cursor is None
        assert len(repaired_records) == 1
        assert len(store.session_queries) == 2
        assert store.session_queries[0].cursor is None
        assert store.session_queries[1].cursor is not None
        assert all(query.limit == 1000 for query in store.session_queries)

    asyncio.run(scenario())


def test_terminal_batch_bounds_store_reads_when_result_capacity_is_small() -> None:
    async def scenario() -> None:
        store = RecordingListSessionsStore()
        app = CayuApp(session_store=store, enable_logging=False)
        damaged_session_id = "sess_batch_bounded_reads_damaged"
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=damaged_session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(damaged_session_id, SessionStatus.COMPLETED)

        for index in range(20):
            session_id = f"sess_batch_bounded_reads_healthy_{index:02d}"
            await store.create(
                RunRequest(
                    agent_name="removed_agent",
                    session_id=session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            await store.append_event(
                session_id,
                Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
            )

        cursor: str | None = None
        pages = []
        query_counts = []
        while True:
            queries_before = len(store.session_queries)
            page = await app.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(
                    statuses={SessionStatus.COMPLETED},
                    limit=1,
                    inspection_limit=10_000,
                    cursor=cursor,
                )
            )
            pages.append(page)
            query_counts.append(len(store.session_queries) - queries_before)
            if page.results:
                break
            assert page.next_cursor is not None
            cursor = page.next_cursor

        assert query_counts == [10, 10, 1]
        assert [page.inspected_session_count for page in pages] == [10, 10, 1]
        assert pages[-1].results[0].session_id == damaged_session_id
        assert pages[-1].results[0].actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
        )
        assert pages[-1].next_cursor is None

    asyncio.run(scenario())


def test_incomplete_session_recovery_cursor_is_bound_to_request_semantics() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        for index in range(2):
            session_id = f"sess_batch_cursor_binding_{index}"
            await store.create(
                RunRequest(
                    agent_name="removed_agent",
                    session_id=session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            await store.append_event(
                session_id,
                Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
            )

        first = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.COMPLETED},
                inspection_limit=1,
                metadata={"deployment": "blue"},
            )
        )
        assert first.results == ()
        assert first.inspected_session_count == 1
        assert first.next_cursor is not None

        with pytest.raises(ValueError, match="does not match the request"):
            await app.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(
                    statuses={SessionStatus.COMPLETED},
                    inspection_limit=1,
                    cursor=first.next_cursor,
                    metadata={"deployment": "green"},
                )
            )

        second = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.COMPLETED},
                limit=10,
                inspection_limit=10,
                cursor=first.next_cursor,
                metadata={"deployment": "blue"},
            )
        )
        assert second.results == ()
        assert second.inspected_session_count == 1
        assert second.next_cursor is None

    asyncio.run(scenario())


def test_incomplete_session_recovery_preserves_opaque_store_cursor() -> None:
    async def scenario() -> None:
        # Exercise the full lower-layer cursor budget with characters that would
        # expand if copied directly into the outer JSON envelope.
        opaque_cursor = '\\"' * (sessions_module.MAX_SESSION_LIST_CURSOR_BYTES // 2)

        class OpaqueCursorRecoveryStore(InMemorySessionStore):
            invocation_lifecycle_command_version = 1

            def __init__(self) -> None:
                super().__init__()
                self.session_ids: list[str] = []
                self.received_cursors: list[str | None] = []

            async def list_sessions(
                self,
                query: SessionQuery | None = None,
            ) -> SessionListResult:
                copied_query = SessionQuery() if query is None else query.model_copy(deep=True)
                self.received_cursors.append(copied_query.cursor)
                if copied_query.cursor is None:
                    session_id = self.session_ids[0]
                    next_cursor = opaque_cursor
                elif copied_query.cursor == opaque_cursor:
                    session_id = self.session_ids[1]
                    next_cursor = None
                else:
                    raise AssertionError("Recovery changed the custom store cursor.")
                session = await self.load(session_id)
                assert session is not None
                return SessionListResult(
                    sessions=[session],
                    next_cursor=next_cursor,
                )

        store = OpaqueCursorRecoveryStore()
        app = CayuApp(session_store=store, enable_logging=False)
        for index in range(2):
            session_id = f"sess_batch_opaque_cursor_{index}"
            store.session_ids.append(session_id)
            await store.create(
                RunRequest(
                    agent_name="removed_agent",
                    session_id=session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            await store.append_event(
                session_id,
                Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
            )

        first = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.COMPLETED},
                limit=1,
                inspection_limit=1,
            )
        )
        assert first.results == ()
        assert first.next_cursor is not None
        assert (
            len(first.next_cursor.encode("utf-8"))
            <= sessions_module.MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES
        )

        second = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.COMPLETED},
                limit=1,
                inspection_limit=1,
                cursor=first.next_cursor,
            )
        )
        assert second.results == ()
        assert second.next_cursor is None
        assert store.received_cursors == [None, opaque_cursor]

    asyncio.run(scenario())
