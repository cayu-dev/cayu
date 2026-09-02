from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.core import Event, Message
from cayu.runtime import (
    EventQuery,
    RunRequest,
    SessionIdentity,
    SessionOperationPublication,
    SessionQuery,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
)
from cayu.runtime import sessions as sessions_module


async def assert_session_store_time_conformance(
    store: SessionStore,
    *,
    initial_time: datetime,
    set_store_time: Callable[[datetime], None],
    contender_store: SessionStore | None = None,
) -> None:
    """Exercise the shared store-authoritative stalled-run time contract.

    The caller owns the backend clock and advances it explicitly.  The same
    scenario is reusable by built-in and custom stores: query, reservation,
    transformation, and takeover must all observe one store-owned timeline.
    """

    minimum_time = datetime.min.replace(tzinfo=UTC)
    minimum_session_id = "session_store_time_minimum_timestamp"
    set_store_time(minimum_time)
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=minimum_session_id,
            messages=[Message.text("user", "maximum duration must not underflow")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    await store.transition_status(
        minimum_session_id,
        from_statuses={SessionStatus.PENDING},
        to_status=SessionStatus.RUNNING,
    )
    assert (
        await store.list_sessions(
            SessionQuery(
                status=SessionStatus.RUNNING,
                inactive_for_seconds=MAX_DURABLE_JSON_INTEGER,
            )
        )
    ).sessions == []
    assert (
        await store.fence_stalled_run(
            minimum_session_id,
            statuses={SessionStatus.RUNNING},
            inactive_for_seconds=MAX_DURABLE_JSON_INTEGER,
        )
        is None
    )
    minimum_reservation_calls: list[datetime] = []
    assert (
        await store.reserve_stalled_run_recovery(
            minimum_session_id,
            statuses={SessionStatus.RUNNING},
            inactive_for_seconds=MAX_DURABLE_JSON_INTEGER,
            checkpoint_transform=lambda _session, checkpoint, observed_at: (
                minimum_reservation_calls.append(observed_at) or checkpoint or {}
            ),
        )
        is None
    )
    assert minimum_reservation_calls == []
    await store.transition_status(
        minimum_session_id,
        from_statuses={SessionStatus.RUNNING},
        to_status=SessionStatus.COMPLETED,
    )
    set_store_time(initial_time)

    transform_session_id = "session_store_time_transform"
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=transform_session_id,
            messages=[Message.text("user", "transform")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    observed_transform_times: list[datetime] = []
    transform_time = initial_time + timedelta(seconds=7)
    set_store_time(transform_time)
    await store.transform_checkpoint_with_store_time(
        transform_session_id,
        lambda _session, checkpoint, observed_at: (
            observed_transform_times.append(observed_at)
            or {
                **({} if checkpoint is None else checkpoint),
                "store_time_observed_at": observed_at.isoformat(),
            }
        ),
    )
    assert observed_transform_times == [transform_time]

    unchanged_session = await store.load(transform_session_id)
    unchanged_checkpoint = await store.load_checkpoint(transform_session_id)
    assert unchanged_session is not None
    assert unchanged_checkpoint is not None
    noop_transform_times: list[datetime] = []
    noop_transform_time = initial_time + timedelta(seconds=8)
    set_store_time(noop_transform_time)
    await store.transform_checkpoint_with_store_time(
        transform_session_id,
        lambda _session, _checkpoint, observed_at: noop_transform_times.append(observed_at) or None,
    )
    assert noop_transform_times == [noop_transform_time]
    assert await store.load(transform_session_id) == unchanged_session
    assert await store.load_checkpoint(transform_session_id) == unchanged_checkpoint

    transition_times: list[datetime] = []
    transition_time = initial_time + timedelta(seconds=8, milliseconds=500)
    set_store_time(transition_time)
    await store.transition_status_and_checkpoint(
        transform_session_id,
        from_statuses={SessionStatus.PENDING},
        to_status=SessionStatus.COMPLETED,
        store_time_checkpoint_transform=lambda _session, checkpoint, observed_at: (
            transition_times.append(observed_at)
            or {
                **({} if checkpoint is None else checkpoint),
                "store_time_status_transition": observed_at.isoformat(),
            }
        ),
    )
    assert transition_times == [transition_time]
    transitioned_checkpoint = await store.load_checkpoint(transform_session_id)
    assert transitioned_checkpoint is not None
    assert transitioned_checkpoint["store_time_status_transition"] == transition_time.isoformat()

    operation_transform_times: list[datetime] = []
    operation_commit_times: list[datetime] = []
    operation_transform_time = initial_time + timedelta(seconds=9)
    operation_commit_time = initial_time + timedelta(seconds=10)
    set_store_time(operation_transform_time)

    def publish_operation(_session, checkpoint, _record, observed_at):
        operation_transform_times.append(observed_at)
        return SessionOperationPublication(
            checkpoint={
                **({} if checkpoint is None else checkpoint),
                "store_time_session_operation": observed_at.isoformat(),
            },
            operation_records={
                "store-time-operation": {
                    "status": "running",
                    "observed_at": observed_at.isoformat(),
                }
            },
        )

    def advance_during_commit_guard() -> None:
        set_store_time(operation_commit_time)

    await store.publish_session_operation_guarded_with_store_time(
        transform_session_id,
        idempotency_key="store-time-operation",
        operation_transform=publish_operation,
        commit_guard=advance_during_commit_guard,
        commit_time_guard=operation_commit_times.append,
        events=[],
    )
    assert operation_transform_times == [operation_transform_time]
    assert operation_commit_times == [operation_commit_time]
    operation_checkpoint = await store.load_checkpoint(transform_session_id)
    assert operation_checkpoint is not None
    assert operation_checkpoint["store_time_session_operation"] == (
        operation_transform_time.isoformat()
    )
    operation_session = await store.load(transform_session_id)
    assert operation_session is not None
    assert operation_session.last_activity_at == operation_commit_time

    checkpoint_publication_times: list[datetime] = []
    checkpoint_commit_times: list[datetime] = []
    checkpoint_publication_time = initial_time + timedelta(seconds=11)
    set_store_time(checkpoint_publication_time)

    def publish_checkpoint(_session, checkpoint, observed_at):
        checkpoint_publication_times.append(observed_at)
        return {
            **({} if checkpoint is None else checkpoint),
            "store_time_checkpoint_publication": observed_at.isoformat(),
        }

    def advance_checkpoint_commit_guard(observed_at: datetime) -> None:
        checkpoint_commit_times.append(observed_at)

    await store.publish_checkpoint_and_events_with_store_time(
        transform_session_id,
        idempotency_key="store-time-checkpoint-publication",
        checkpoint_transform=publish_checkpoint,
        commit_time_guard=advance_checkpoint_commit_guard,
        events=[
            Event(
                id="store-time-checkpoint-publication-event",
                type="custom.store-time-checkpoint-publication",
                session_id=transform_session_id,
            )
        ],
    )
    assert checkpoint_publication_times == [checkpoint_publication_time]
    assert checkpoint_commit_times == [checkpoint_publication_time]
    checkpoint_publication = await store.load_checkpoint(transform_session_id)
    assert checkpoint_publication is not None
    assert checkpoint_publication["store_time_checkpoint_publication"] == (
        checkpoint_publication_time.isoformat()
    )

    completion_publication_session_id = "session_store_time_completion_publication"
    completion_publication_time = initial_time + timedelta(seconds=12)
    set_store_time(completion_publication_time)
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=completion_publication_session_id,
            messages=[Message.text("user", "completion publication")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    publication_digest = "a" * 64
    publication_id = f"completion-result-publication:v1:{publication_digest}"
    publication_owner_id = f"completion-result-owner:v1:{'b' * 64}"
    observed_completion_publication_times: list[datetime] = []

    def reserve_completion_publication(_session, checkpoint, observed_at):
        observed_completion_publication_times.append(observed_at)
        return sessions_module._reserve_completion_result_event_publication(
            checkpoint,
            publication_id=publication_id,
            authority_sha256=publication_digest,
            owner_id=publication_owner_id,
            owner_expires_at=observed_at + timedelta(seconds=1),
            now=observed_at,
        )

    await store._publish_completion_result_event_publication(
        completion_publication_session_id,
        checkpoint_transform=reserve_completion_publication,
        events=[],
    )
    assert observed_completion_publication_times == [completion_publication_time]

    set_store_time(completion_publication_time + timedelta(seconds=2))
    stale_event = Event(
        id="store-time-stale-completion-publication",
        type="custom.store-time-stale-completion-publication",
        session_id=completion_publication_session_id,
    )
    with pytest.raises(
        ValueError,
        match="Completion-result event publication reservation is missing",
    ):
        await store._publish_completion_result_event_publication(
            completion_publication_session_id,
            checkpoint_transform=lambda _session, checkpoint, observed_at: (
                sessions_module._complete_completion_result_event_publication(
                    checkpoint,
                    publication_id=publication_id,
                    authority_sha256=publication_digest,
                    owner_id=publication_owner_id,
                    require_present=True,
                    now=observed_at,
                )
            ),
            events=[stale_event],
        )
    assert (
        await store.query_events(
            EventQuery(
                session_id=completion_publication_session_id,
                event_id=stale_event.id,
            )
        )
        == []
    )
    await store.delete_session(completion_publication_session_id)
    assert await store.load(completion_publication_session_id) is None

    expired_promotion_session_id = "session_store_time_expired_promotion"
    set_store_time(initial_time)
    expired_promotion_session = await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=expired_promotion_session_id,
            messages=[Message.text("user", "expired recovery cannot transfer ownership")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    await store.checkpoint(
        expired_promotion_session_id,
        {
            "incomplete_session_recovery_claim": {
                "version": 1,
                "claim_id": "expired-promotion-claim",
                "claimed_at": initial_time.isoformat(),
                "claim_expires_at": (initial_time + timedelta(seconds=1)).isoformat(),
            },
        },
    )
    set_store_time(initial_time + timedelta(seconds=2))

    def preserve_checkpoint(_session, checkpoint):
        assert checkpoint is not None
        return checkpoint

    assert (
        await store.fence_stalled_run(
            expired_promotion_session_id,
            statuses={SessionStatus.PENDING},
            inactive_for_seconds=0,
        )
        is None
    )
    with pytest.raises(SessionRunFenced, match="expired before run ownership transfer"):
        await store.transition_status(
            expired_promotion_session_id,
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )
    with pytest.raises(SessionRunFenced, match="expired before run ownership transfer"):
        await store.fence_run_and_transform_checkpoint(
            expired_promotion_session_id,
            statuses={SessionStatus.PENDING},
            checkpoint_transform=preserve_checkpoint,
        )
    with pytest.raises(SessionRunFenced, match="expired before run ownership transfer"):
        await store.transition_status_and_checkpoint(
            expired_promotion_session_id,
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=preserve_checkpoint,
        )
    after_rejected_promotion = await store.load(expired_promotion_session_id)
    assert after_rejected_promotion is not None
    assert after_rejected_promotion.run_epoch == expired_promotion_session.run_epoch
    assert after_rejected_promotion.status is SessionStatus.PENDING

    def replace_expired_claim(_session, checkpoint, observed_at):
        assert checkpoint is not None
        updated = dict(checkpoint)
        updated["incomplete_session_recovery_claim"] = {
            "version": 1,
            "claim_id": "replacement-promotion-claim",
            "claimed_at": observed_at.isoformat(),
            "claim_expires_at": (observed_at + timedelta(minutes=5)).isoformat(),
        }
        return updated

    assert (
        await store.reserve_stalled_run_recovery(
            expired_promotion_session_id,
            statuses={SessionStatus.PENDING},
            inactive_for_seconds=None,
            checkpoint_transform=replace_expired_claim,
        )
        is not None
    )
    promoted = await store.fence_run_and_transform_checkpoint(
        expired_promotion_session_id,
        statuses={SessionStatus.PENDING},
        checkpoint_transform=preserve_checkpoint,
    )
    assert promoted.run_epoch == expired_promotion_session.run_epoch + 1

    session_id = "session_store_time_takeover"
    set_store_time(initial_time)
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "take over only after expiry")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    running = await store.transition_status(
        session_id,
        from_statuses={SessionStatus.PENDING},
        to_status=SessionStatus.RUNNING,
    )

    assert (
        await store.list_sessions(
            SessionQuery(
                status=SessionStatus.RUNNING,
                inactive_for_seconds=MAX_DURABLE_JSON_INTEGER,
            )
        )
    ).sessions == []
    assert (
        await store.fence_stalled_run(
            session_id,
            statuses={SessionStatus.RUNNING},
            inactive_for_seconds=MAX_DURABLE_JSON_INTEGER,
        )
        is None
    )

    reservation_calls: list[datetime] = []

    def reserve(_session, checkpoint, observed_at):
        reservation_calls.append(observed_at)
        return {
            **({} if checkpoint is None else checkpoint),
            "store_time_recovery_reserved_at": observed_at.isoformat(),
        }

    set_store_time(initial_time + timedelta(seconds=59))
    early_page = await store.list_sessions(
        SessionQuery(
            status=SessionStatus.RUNNING,
            inactive_for_seconds=60,
        )
    )
    assert early_page.sessions == []
    assert (
        await store.reserve_stalled_run_recovery(
            session_id,
            statuses={SessionStatus.RUNNING},
            inactive_for_seconds=60,
            checkpoint_transform=reserve,
        )
        is None
    )
    assert reservation_calls == []

    expiry = initial_time + timedelta(seconds=60)
    set_store_time(expiry)
    eligible_page = await store.list_sessions(
        SessionQuery(
            status=SessionStatus.RUNNING,
            inactive_for_seconds=60,
        )
    )
    assert [session.id for session in eligible_page.sessions] == [session_id]
    reserved = await store.reserve_stalled_run_recovery(
        session_id,
        statuses={SessionStatus.RUNNING},
        inactive_for_seconds=60,
        checkpoint_transform=reserve,
    )
    assert reserved is not None
    assert reserved.run_epoch == running.run_epoch
    assert reservation_calls == [expiry]

    contenders = await asyncio.gather(
        store.fence_stalled_run(
            session_id,
            statuses={SessionStatus.RUNNING},
            inactive_for_seconds=60,
        ),
        (contender_store or store).fence_stalled_run(
            session_id,
            statuses={SessionStatus.RUNNING},
            inactive_for_seconds=60,
        ),
    )
    winners = [candidate for candidate in contenders if candidate is not None]
    assert len(winners) == 1
    assert winners[0].run_epoch == running.run_epoch + 1
    assert winners[0].last_activity_at == expiry

    # The context that owned the pre-takeover epoch must remain fenced even
    # though another contender observed the already refreshed record.
    with pytest.raises(SessionRunFenced, match="no longer owns"):
        await store.checkpoint(session_id, {"stale": True})
