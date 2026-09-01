"""Reusable semantic conformance for the SessionStore publication fault harness."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.core._session_operation_fault_harness import (
    CommitEvidence,
    CommitThenRaise,
    FailBeforeCommit,
    FailBeforeTransform,
    MatchPolicy,
    PauseBeforeTransform,
    PublicationBarrier,
    PublicationFaultActionKind,
    PublicationFaultOutcome,
    SessionOperationFaultHarness,
    SessionOperationFaultRule,
    SessionOperationFaultScheduleError,
    SessionOperationSelector,
)

from cayu.core import Event, EventType
from cayu.core.tools import DurableToolOperationConflict
from cayu.runtime import RunRequest, SessionIdentity
from cayu.runtime.sessions import (
    INTERACTION_TRANSITION_OPERATION_KEY_PREFIX,
    INVOCATION_TERMINAL_EVENT_OPERATION_KEY_PREFIX,
    MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX,
    RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX,
    SessionOperationPublication,
    SessionStore,
)


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fault-conformance", model="fake-model")


async def assert_session_operation_fault_conformance(
    store: SessionStore,
    *,
    session_id_prefix: str,
) -> None:
    """Exercise the complete fault vocabulary against one real store backend."""

    created = await store.create(
        RunRequest(
            agent_name="fault-conformance",
            session_id=f"{session_id_prefix}-publication-faults",
            messages=[],
        ),
        identity=_identity(),
    )
    transform_calls: dict[str, int] = {}

    reserved_keys = tuple(
        prefix + "fault-harness-validation"
        for prefix in (
            RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX,
            INTERACTION_TRANSITION_OPERATION_KEY_PREFIX,
            INVOCATION_TERMINAL_EVENT_OPERATION_KEY_PREFIX,
            MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX,
        )
    )
    reserved_transform_calls = 0

    def reserved_transform(_session, _checkpoint, _current_record):
        nonlocal reserved_transform_calls
        reserved_transform_calls += 1
        return SessionOperationPublication(
            checkpoint={"reserved_key_probe": True},
            operation_records={"reserved-key-probe": {"committed": True}},
        )

    reserved_rules = tuple(
        SessionOperationFaultRule(
            rule_id=f"reserved-key-{index}",
            selector=SessionOperationSelector(idempotency_key=key),
            actions=(FailBeforeCommit(),),
        )
        for index, key in enumerate(reserved_keys)
    )
    with pytest.raises(SessionOperationFaultScheduleError, match="not satisfied"):
        async with SessionOperationFaultHarness(
            store,
            rules=reserved_rules,
        ) as reserved_harness:
            for key in reserved_keys:
                with pytest.raises(ValueError, match="reserved"):
                    await store.publish_session_operation(
                        created.id,
                        idempotency_key=key,
                        operation_transform=reserved_transform,
                        events=[],
                    )

    assert reserved_transform_calls == 0
    assert await store.load(created.id) == created
    assert await store.load_checkpoint(created.id) is None
    assert await store.load_events(created.id) == []
    assert await store.load_session_operation(created.id, "reserved-key-probe") is None
    assert [record.matched_rule_id for record in reserved_harness.trace] == [
        f"reserved-key-{index}" for index in range(len(reserved_keys))
    ]
    assert all(
        record.action is PublicationFaultActionKind.FAIL_BEFORE_COMMIT
        and record.action_reached is False
        and record.transform_started is False
        and record.committed is CommitEvidence.NO
        for record in reserved_harness.trace
    )

    async def publish(
        key: str,
        desired: dict[str, Any],
        *,
        expected: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> None:
        desired_copy = dict(desired)

        def transform(_session, checkpoint, current_record):
            transform_calls[key] = transform_calls.get(key, 0) + 1
            if current_record != expected:
                raise DurableToolOperationConflict("operation compare-and-set lost")
            return SessionOperationPublication(
                checkpoint={"last_operation": key},
                operation_records={key: desired_copy},
            )

        await store.publish_session_operation(
            created.id,
            idempotency_key=key,
            operation_transform=transform,
            events=(
                []
                if event_id is None
                else [
                    Event(
                        id=event_id,
                        type=EventType.SESSION_CHECKPOINTED,
                        session_id=created.id,
                        payload={"operation": key},
                    )
                ]
            ),
        )

    stale_barrier = PublicationBarrier()
    rules = (
        SessionOperationFaultRule(
            rule_id="pre-transform",
            selector=SessionOperationSelector(idempotency_key="fault-pre-transform"),
            actions=(FailBeforeTransform(),),
        ),
        SessionOperationFaultRule(
            rule_id="pre-commit",
            selector=SessionOperationSelector(idempotency_key="fault-pre-commit"),
            actions=(FailBeforeCommit(),),
        ),
        SessionOperationFaultRule(
            rule_id="lost-ack",
            selector=SessionOperationSelector(idempotency_key="fault-lost-ack"),
            actions=(CommitThenRaise(),),
            on_exhausted=MatchPolicy.DELEGATE,
        ),
        SessionOperationFaultRule(
            rule_id="repeat-terminal",
            selector=SessionOperationSelector(idempotency_key="fault-terminal"),
            actions=(FailBeforeTransform(count=2),),
            on_exhausted=MatchPolicy.DELEGATE,
        ),
        SessionOperationFaultRule(
            rule_id="stale-owner",
            selector=SessionOperationSelector(
                session_id=created.id,
                idempotency_key="fault-owner",
                label="stale-owner",
            ),
            actions=(PauseBeforeTransform(stale_barrier),),
        ),
    )

    async with SessionOperationFaultHarness(store, rules=rules) as harness:
        with pytest.raises(ConnectionError, match="before transform"):
            await publish("fault-pre-transform", {"state": "uncommitted"})
        assert transform_calls.get("fault-pre-transform", 0) == 0
        assert await store.load_session_operation(created.id, "fault-pre-transform") is None
        assert await store.load_checkpoint(created.id) is None
        assert await store.load_events(created.id) == []

        with pytest.raises(ConnectionError, match="before commit"):
            await publish(
                "fault-pre-commit",
                {"state": "rolled-back"},
                event_id="fault-pre-commit-event",
            )
        assert transform_calls["fault-pre-commit"] == 1
        assert await store.load_session_operation(created.id, "fault-pre-commit") is None
        assert await store.load_checkpoint(created.id) is None
        assert await store.load_events(created.id) == []

        acknowledged_record = {
            "state": "completed",
            "evidence": ["durable-before-acknowledgement"],
        }
        with pytest.raises(ConnectionError, match="acknowledgement was lost"):
            await publish(
                "fault-lost-ack",
                acknowledged_record,
                event_id="fault-lost-ack-event",
            )
        assert await store.load_session_operation(created.id, "fault-lost-ack") == (
            acknowledged_record
        )
        assert await store.load_checkpoint(created.id) == {"last_operation": "fault-lost-ack"}
        assert [event.id for event in await store.load_events(created.id)] == [
            "fault-lost-ack-event"
        ]

        await publish(
            "fault-lost-ack",
            acknowledged_record,
            expected=acknowledged_record,
        )
        assert transform_calls["fault-lost-ack"] == 2
        assert [event.id for event in await store.load_events(created.id)] == [
            "fault-lost-ack-event"
        ]

        preserved_record = {
            "state": "completed",
            "evidence": ["completed-sibling"],
        }
        await publish(
            "fault-preserved",
            preserved_record,
            event_id="fault-preserved-event",
        )
        terminal_record = {"state": "terminal", "outcome": "gate_failed"}
        for _ in range(2):
            with pytest.raises(ConnectionError, match="before transform"):
                await publish(
                    "fault-terminal",
                    terminal_record,
                    event_id="fault-terminal-event",
                )
        assert transform_calls.get("fault-terminal", 0) == 0
        assert await store.load_checkpoint(created.id) == {"last_operation": "fault-preserved"}
        assert [event.id for event in await store.load_events(created.id)] == [
            "fault-lost-ack-event",
            "fault-preserved-event",
        ]
        await publish(
            "fault-terminal",
            terminal_record,
            event_id="fault-terminal-event",
        )
        assert await store.load_session_operation(created.id, "fault-terminal") == terminal_record
        assert await store.load_session_operation(created.id, "fault-preserved") == (
            preserved_record
        )
        assert await store.load_checkpoint(created.id) == {"last_operation": "fault-terminal"}
        assert [event.id for event in await store.load_events(created.id)] == [
            "fault-lost-ack-event",
            "fault-preserved-event",
            "fault-terminal-event",
        ]

        async def stale_publish() -> None:
            with harness.label("stale-owner"):
                await publish(
                    "fault-owner",
                    {"owner": "stale", "generation": 1},
                    event_id="fault-stale-owner-event",
                )

        stale_task = asyncio.create_task(stale_publish())
        await stale_barrier.wait_until_entered()
        successor_record = {"owner": "successor", "generation": 2}
        await publish(
            "fault-owner",
            successor_record,
            event_id="fault-successor-owner-event",
        )
        stale_barrier.release()
        with pytest.raises(DurableToolOperationConflict, match="compare-and-set lost"):
            await stale_task
        assert await store.load_session_operation(created.id, "fault-owner") == successor_record
        assert await store.load_checkpoint(created.id) == {"last_operation": "fault-owner"}
        assert [event.id for event in await store.load_events(created.id)] == [
            "fault-lost-ack-event",
            "fault-preserved-event",
            "fault-terminal-event",
            "fault-successor-owner-event",
        ]

    by_rule = {
        rule_id: [record for record in harness.trace if record.matched_rule_id == rule_id]
        for rule_id in (
            "pre-transform",
            "pre-commit",
            "lost-ack",
            "repeat-terminal",
            "stale-owner",
        )
    }
    assert by_rule["pre-transform"] == [
        by_rule["pre-transform"][0].__class__(
            sequence=by_rule["pre-transform"][0].sequence,
            matched_rule_id="pre-transform",
            action=PublicationFaultActionKind.FAIL_BEFORE_TRANSFORM,
            outcome=PublicationFaultOutcome.INJECTED_FAILURE,
            action_reached=True,
            transform_started=False,
            transform_completed=False,
            committed=CommitEvidence.NO,
            acknowledgement_returned=False,
        )
    ]
    [pre_commit] = by_rule["pre-commit"]
    assert pre_commit.action is PublicationFaultActionKind.FAIL_BEFORE_COMMIT
    assert pre_commit.outcome is PublicationFaultOutcome.INJECTED_FAILURE
    assert pre_commit.action_reached is True
    assert pre_commit.transform_started is True
    assert pre_commit.transform_completed is True
    assert pre_commit.committed is CommitEvidence.NO
    assert pre_commit.acknowledgement_returned is False

    lost_ack, replay = by_rule["lost-ack"]
    assert lost_ack.action is PublicationFaultActionKind.COMMIT_THEN_RAISE
    assert lost_ack.committed is CommitEvidence.YES
    assert lost_ack.acknowledgement_returned is False
    assert replay.action is PublicationFaultActionKind.EXHAUSTED_DELEGATE
    assert replay.committed is CommitEvidence.YES
    assert replay.acknowledgement_returned is True

    repeated = by_rule["repeat-terminal"]
    assert [record.action for record in repeated] == [
        PublicationFaultActionKind.FAIL_BEFORE_TRANSFORM,
        PublicationFaultActionKind.FAIL_BEFORE_TRANSFORM,
        PublicationFaultActionKind.EXHAUSTED_DELEGATE,
    ]
    assert [record.committed for record in repeated] == [
        CommitEvidence.NO,
        CommitEvidence.NO,
        CommitEvidence.YES,
    ]
    [stale] = by_rule["stale-owner"]
    assert stale.action is PublicationFaultActionKind.PAUSE_BEFORE_TRANSFORM
    assert stale.outcome is PublicationFaultOutcome.DELEGATE_FAILURE
    assert stale.action_reached is True
    assert stale.transform_started is True
    assert stale.committed is CommitEvidence.NO
