from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.runtime import (
    CayuApp,
    EventQuery,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    RunRequest,
    RuntimeHookPhase,
    SessionIdentity,
    SessionRunFenced,
    SessionStatus,
    TerminalEventPublicationUncertain,
)
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.runtime.sessions import _checkpoint_with_session_run_operation
from cayu.vaults import REDACTED_SECRET, SecretRedactor


def test_terminal_publication_uncertainty_preserves_both_failure_chains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> TerminalEventPublicationUncertain:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_terminal_publication_reconciliation_failure"
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        session = await store.update_status(session_id, SessionStatus.COMPLETED)

        publication_cause = OSError("database connection failed")
        publication_failure = RuntimeError("publication acknowledgement failed")
        publication_failure.__cause__ = publication_cause
        reconciliation_cause = TimeoutError("read timed out")
        reconciliation_failure = LookupError("reconciliation query failed")
        reconciliation_failure.__cause__ = reconciliation_cause

        async def fail_publication(_event: Event) -> Event:
            raise publication_failure

        async def fail_reconciliation(_event: Event) -> Event | None:
            raise reconciliation_failure

        monkeypatch.setattr(app._event_writer, "emit", fail_publication)
        monkeypatch.setattr(
            app._session_engine,
            "_reconcile_persisted_terminal_event",
            fail_reconciliation,
        )
        stream = app._session_engine._emit_terminal_event_with_hooks(
            event=Event(
                type=EventType.SESSION_COMPLETED,
                session_id=session_id,
                agent_name="assistant",
            ),
            phase=RuntimeHookPhase.AFTER_SESSION_COMPLETED,
            session=session,
            registered_agent=app._get_registered_agent("assistant"),
            registered_environment=None,
        )
        with pytest.raises(TerminalEventPublicationUncertain) as raised:
            await anext(stream)
        return raised.value

    failure = asyncio.run(scenario())

    assert type(failure) is TerminalEventPublicationUncertain
    assert failure.__cause__ is failure.failures
    publication_failure, reconciliation_failure = failure.failures.exceptions
    assert type(publication_failure) is RuntimeError
    assert str(publication_failure) == "publication acknowledgement failed"
    assert type(publication_failure.__cause__) is OSError
    assert str(publication_failure.__cause__) == "database connection failed"
    assert type(reconciliation_failure) is LookupError
    assert str(reconciliation_failure) == "reconciliation query failed"
    assert type(reconciliation_failure.__cause__) is TimeoutError
    assert str(reconciliation_failure.__cause__) == "read timed out"


@pytest.mark.parametrize(
    ("status", "event_type"),
    [
        (SessionStatus.COMPLETED, EventType.SESSION_COMPLETED),
        (SessionStatus.FAILED, EventType.SESSION_FAILED),
        (SessionStatus.INTERRUPTED, EventType.SESSION_INTERRUPTED),
    ],
)
def test_cayu_app_terminal_evidence_repair_is_status_consistent_and_idempotent(
    status: SessionStatus,
    event_type: EventType,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session_id = f"sess_terminal_evidence_{status.value}"
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        terminal = await store.update_status(session_id, status)

        first = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        second = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )

        persisted = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(
                    EventType.SESSION_COMPLETED,
                    EventType.SESSION_FAILED,
                    EventType.SESSION_INTERRUPTED,
                ),
            )
        )
        current = await store.load(session_id)
        assert current is not None
        assert current.status == status
        assert current.run_epoch == terminal.run_epoch + 2
        assert len(persisted) == 1
        assert persisted[0].event.type == event_type
        assert persisted[0].event.timestamp == terminal.updated_at
        assert first.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
        assert first.events == (app.project_event_record_for_exposure(persisted[0]).event,)
        assert second.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
        assert second.events == ()

    asyncio.run(scenario())


def test_cayu_app_terminal_evidence_repair_reconciles_lost_append_acknowledgement() -> None:
    class LostTerminalRepairAcknowledgementStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def append_event(self, session_id: str, event: Event) -> None:
            await super().append_event(session_id, event)
            if event.payload.get("terminal_evidence_repaired") is True and not self.failed:
                self.failed = True
                raise ConnectionError("terminal evidence acknowledgement lost")

    async def scenario() -> None:
        store = LostTerminalRepairAcknowledgementStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_terminal_evidence_lost_ack"
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.FAILED)

        repaired = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        persisted = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_FAILED,
            )
        )
        assert len(persisted) == 1
        assert store.failed is True
        assert repaired.events == (app.project_event_record_for_exposure(persisted[0]).event,)
        assert await store.load_checkpoint(session_id) == {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
        }

    asyncio.run(scenario())


def test_terminal_evidence_repair_reconciles_redacted_lost_append_acknowledgement() -> None:
    class LostRedactedTerminalRepairAcknowledgementStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def append_event(self, session_id: str, event: Event) -> None:
            await super().append_event(session_id, event)
            if event.type == EventType.SESSION_INTERRUPTED and not self.failed:
                self.failed = True
                raise ConnectionError("redacted terminal evidence acknowledgement lost")

    async def scenario() -> None:
        secret = "repair-secret-canary"
        store = LostRedactedTerminalRepairAcknowledgementStore()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        session_id = "sess_terminal_evidence_redacted_lost_ack"
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        await store.checkpoint(
            session_id,
            {
                "pending_session_interrupt": {
                    "reason": f"deploy {secret}",
                    "metadata": {"source": secret},
                    "interruption_type": "operator_requested",
                    "interruption_request_id": "interrupt-redacted-lost-ack",
                }
            },
        )

        repaired = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        persisted = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_INTERRUPTED,
            )
        )
        checkpoint = await store.load_checkpoint(session_id)

        assert store.failed is True
        assert len(persisted) == 1
        assert repaired.events == (app.project_event_record_for_exposure(persisted[0]).event,)
        assert persisted[0].event.payload["reason"] == f"deploy {REDACTED_SECRET}"
        assert persisted[0].event.payload["metadata"] == {"source": REDACTED_SECRET}
        assert secret not in str(persisted[0].event.payload)
        assert checkpoint == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}

    asyncio.run(scenario())


def test_terminal_evidence_repair_rejects_same_id_with_different_timestamp() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_terminal_repair_conflicting_timestamp"
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        persisted = Event(
            id="terminal-repair-conflicting-timestamp",
            type=EventType.SESSION_FAILED,
            session_id=session_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            agent_name="removed_agent",
            payload={
                "error": "Original terminal failure details were not durably recorded.",
                "error_type": "TerminalFailureEvidenceUnavailable",
                "recovered": True,
                "terminal_evidence_repaired": True,
            },
        )
        await store.append_event(session_id, persisted)
        expected = persisted.model_copy(
            update={"timestamp": persisted.timestamp + timedelta(microseconds=1)},
            deep=True,
        )

        with pytest.raises(
            RuntimeError,
            match="identity is already used by different durable evidence",
        ) as raised:
            await app._recovery_coordinator._persist_terminal_evidence_repair_event(expected)

        assert raised.value.__cause__ is not None
        records = await store.query_events(
            EventQuery(session_id=session_id, event_id=persisted.id, limit=1)
        )
        assert [record.event for record in records] == [persisted]

    asyncio.run(scenario())


def test_terminal_evidence_repair_reconciliation_preserves_both_failure_chains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> ExceptionGroup:
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        append_cause = OSError("database write failed")
        append_failure = RuntimeError("append acknowledgement failed")
        append_failure.__cause__ = append_cause
        reconciliation_cause = TimeoutError("database read timed out")
        reconciliation_failure = LookupError("append reconciliation failed")
        reconciliation_failure.__cause__ = reconciliation_cause

        async def fail_append(_event: Event) -> Event:
            raise append_failure

        async def fail_query(_query: EventQuery):
            raise reconciliation_failure

        monkeypatch.setattr(app._event_writer, "persist", fail_append)
        monkeypatch.setattr(app.session_store, "query_events", fail_query)
        with pytest.raises(ExceptionGroup) as raised:
            await app._recovery_coordinator._persist_terminal_evidence_repair_event(
                Event(
                    type=EventType.SESSION_FAILED,
                    session_id="sess_terminal_repair_dual_failure",
                )
            )
        return raised.value

    failure = asyncio.run(scenario())

    assert type(failure) is ExceptionGroup
    append_failure, reconciliation_failure = failure.exceptions
    assert type(append_failure) is RuntimeError
    assert str(append_failure) == "append acknowledgement failed"
    assert type(append_failure.__cause__) is OSError
    assert str(append_failure.__cause__) == "database write failed"
    assert type(reconciliation_failure) is LookupError
    assert str(reconciliation_failure) == "append reconciliation failed"
    assert type(reconciliation_failure.__cause__) is TimeoutError
    assert str(reconciliation_failure.__cause__) == "database read timed out"


def test_cayu_app_terminal_evidence_repair_recognizes_event_before_marker_cleanup() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_terminal_evidence_event_before_cleanup"
        pending_payload = {
            "reason": "deploy",
            "metadata": {},
            "interruption_type": "operator_requested",
            "interruption_request_id": "interrupt-before-cleanup-crash",
        }
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        await store.checkpoint(
            session_id,
            {"pending_session_interrupt": pending_payload},
        )
        existing_event = Event(
            type=EventType.SESSION_INTERRUPTED,
            session_id=session_id,
            payload=pending_payload,
        )
        await store.append_event(session_id, existing_event)

        repaired = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_INTERRUPTED,
            )
        )
        assert repaired.events == (app.project_event_record_for_exposure(records[0]).event,)
        assert [record.event.id for record in records] == [existing_event.id]
        assert await store.load_checkpoint(session_id) == {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
        }

    asyncio.run(scenario())


def test_cayu_app_terminal_evidence_repair_clears_expired_post_publication_claim() -> None:
    async def scenario() -> None:
        now = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now,
        )
        session_id = "sess_terminal_evidence_expired_claim"
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)
        existing_event = Event(
            type=EventType.SESSION_COMPLETED,
            session_id=session_id,
        )
        await store.append_event(session_id, existing_event)
        await store.checkpoint(
            session_id,
            {
                "incomplete_session_recovery_claim": {
                    "version": 1,
                    "claim_id": "crashed-repair",
                    "claimed_at": (now - timedelta(minutes=10)).isoformat(),
                    "claim_expires_at": (now - timedelta(minutes=5)).isoformat(),
                }
            },
        )

        repaired = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_COMPLETED,
            )
        )
        assert repaired.events == (app.project_event_record_for_exposure(records[0]).event,)
        assert repaired.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
        assert await store.load_checkpoint(session_id) == {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
        }

    asyncio.run(scenario())


def test_cayu_app_terminal_evidence_repair_fences_delayed_original_publisher() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_terminal_evidence_delayed_publisher"
        status_committed = asyncio.Event()
        allow_original_append = asyncio.Event()

        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

        async def delayed_original_publisher() -> None:
            await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            status_committed.set()
            await allow_original_append.wait()
            try:
                await store.append_event(
                    session_id,
                    Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
                )
            finally:
                await store.release_run_fence(session_id)

        original_task = asyncio.create_task(delayed_original_publisher())
        await asyncio.wait_for(status_committed.wait(), timeout=5)
        repaired = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        allow_original_append.set()
        with pytest.raises(SessionRunFenced):
            await asyncio.wait_for(original_task, timeout=5)

        records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_COMPLETED,
            )
        )
        assert repaired.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
        assert len(records) == 1
        assert repaired.events == (app.project_event_record_for_exposure(records[0]).event,)

    asyncio.run(scenario())


def test_cayu_app_terminal_evidence_repair_rejects_future_run_operation() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_terminal_evidence_future_operation"
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)
        future_marker = {
            "version": 1,
            "operation_id": "impossible-future-operation",
            "run_epoch": 100,
        }
        await store.checkpoint(
            session_id,
            {"session_run_operation": future_marker},
        )

        with pytest.raises(RuntimeError, match="future run epoch"):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )

        session = await store.load(session_id)
        assert session is not None
        assert session.status == SessionStatus.COMPLETED
        assert session.run_epoch == 0
        assert await store.load_checkpoint(session_id) == {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
            "session_run_operation": future_marker,
        }
        assert await store.query_events(EventQuery(session_id=session_id)) == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("status", "contradictory_type"),
    [
        (SessionStatus.COMPLETED, EventType.SESSION_FAILED),
        (SessionStatus.FAILED, EventType.SESSION_INTERRUPTED),
        (SessionStatus.INTERRUPTED, EventType.SESSION_COMPLETED),
    ],
)
def test_cayu_app_terminal_evidence_repair_rejects_contradictory_events(
    status: SessionStatus,
    contradictory_type: EventType,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session_id = f"sess_terminal_evidence_conflict_{status.value}"
        operation_id = f"terminal-evidence-conflict-{status.value}"
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)
        await store.transition_status_and_checkpoint(
            session_id,
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=lambda current_session, checkpoint: (
                _checkpoint_with_session_run_operation(
                    checkpoint=checkpoint,
                    current_session=current_session,
                    operation_id=operation_id,
                )
            ),
        )
        await store.update_status(session_id, status)
        await store.append_event(
            session_id,
            Event(
                type=contradictory_type,
                session_id=session_id,
                payload={"session_run_operation_id": operation_id},
            ),
        )

        with pytest.raises(RuntimeError, match="Terminal evidence is contradictory"):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
        records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(
                    EventType.SESSION_COMPLETED,
                    EventType.SESSION_FAILED,
                    EventType.SESSION_INTERRUPTED,
                ),
            )
        )
        assert [record.event.type for record in records] == [contradictory_type]
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["session_run_operation"]["operation_id"] == operation_id

    asyncio.run(scenario())


def test_cayu_app_terminal_evidence_repair_rejects_duplicate_matching_events() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_terminal_evidence_duplicate"
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)
        await store.append_events(
            session_id,
            [
                Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
                Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
            ],
        )

        with pytest.raises(RuntimeError, match="more than one terminal event"):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )

    asyncio.run(scenario())


def test_cayu_app_terminal_evidence_repair_scopes_evidence_to_latest_run() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_terminal_evidence_latest_run"
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            session_id,
            [
                Event(type=EventType.SESSION_STARTED, session_id=session_id),
                Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
                Event(type=EventType.SESSION_RESUMED, session_id=session_id),
            ],
        )
        await store.update_status(session_id, SessionStatus.FAILED)

        repaired = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert repaired.events[0].type == EventType.SESSION_FAILED
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
        assert [record.event.type for record in terminal_records] == [
            EventType.SESSION_COMPLETED,
            EventType.SESSION_FAILED,
        ]

    asyncio.run(scenario())


def test_cayu_app_terminal_evidence_repair_serializes_concurrent_workers() -> None:
    class BlockingTerminalClaimStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.claimed = asyncio.Event()
            self.release = asyncio.Event()

        async def fence_run_and_transform_checkpoint(self, session_id: str, **kwargs):
            transitioned = await super().fence_run_and_transform_checkpoint(
                session_id,
                **kwargs,
            )
            if set(kwargs["statuses"]) <= {
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.INTERRUPTED,
            }:
                self.claimed.set()
                await self.release.wait()
            return transitioned

    async def scenario() -> None:
        store = BlockingTerminalClaimStore()
        first_app = CayuApp(session_store=store, enable_logging=False)
        second_app = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_terminal_evidence_concurrent"
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id=session_id,
                messages=[Message.text("user", "finish")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)

        request = IncompleteSessionRecoveryRequest(session_id=session_id)
        first_task = asyncio.create_task(first_app.recover_incomplete_session(request))
        await asyncio.wait_for(store.claimed.wait(), timeout=5)
        second = await asyncio.wait_for(
            second_app.recover_incomplete_session(request),
            timeout=5,
        )
        store.release.set()
        first = await asyncio.wait_for(first_task, timeout=5)

        records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_COMPLETED,
            )
        )
        assert first.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
        assert second.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        assert len(records) == 1
        assert await store.load_checkpoint(session_id) == {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
        }

    asyncio.run(scenario())
