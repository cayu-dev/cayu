from __future__ import annotations

import asyncio
import logging
import warnings
from pathlib import Path
from typing import Any, cast

import pytest
from tests.core._session_operation_fault_harness import (
    CommitEvidence,
    CommitThenRaise,
    Delegate,
    FailBeforeTransform,
    MatchPolicy,
    PauseAfterCommit,
    PauseBeforeCommit,
    PauseBeforeTransform,
    PublicationBarrier,
    PublicationBoundary,
    PublicationFaultActionKind,
    PublicationFaultOutcome,
    SessionOperationFaultHarness,
    SessionOperationFaultRule,
    SessionOperationFaultScheduleError,
    SessionOperationSelector,
)
from tests.core.session_operation_fault_conformance import (
    assert_session_operation_fault_conformance,
)

from cayu.core import Event, EventType
from cayu.runtime import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.runtime.sessions import SessionOperationPublication, SessionStore
from cayu.storage import SQLiteSessionStore


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fault-harness", model="fake-model")


async def _create_session(store: SessionStore, session_id: str) -> None:
    await store.create(
        RunRequest(agent_name="fault-harness", session_id=session_id, messages=[]),
        identity=_identity(),
    )


async def _publish_record(
    store: SessionStore,
    session_id: str,
    key: str,
    *,
    value: str = "committed",
    events: list[Event] | None = None,
) -> None:
    desired = {"value": value}

    def transform(_session, checkpoint, _current_record):
        return SessionOperationPublication(
            checkpoint={} if checkpoint is None else checkpoint,
            operation_records={key: desired},
        )

    await store.publish_session_operation(
        session_id,
        idempotency_key=key,
        operation_transform=transform,
        events=[] if events is None else events,
    )


async def _close_store(store: SessionStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_session_operation_fault_harness_store_conformance(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store: SessionStore
        if store_kind == "memory":
            store = InMemorySessionStore()
        else:
            store = SQLiteSessionStore(tmp_path / "publication-faults.sqlite")
        try:
            await assert_session_operation_fault_conformance(
                store,
                session_id_prefix=store_kind,
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_event_append_fault_boundary_distinguishes_precommit_from_lost_ack(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store: SessionStore
        if store_kind == "memory":
            store = InMemorySessionStore()
        else:
            store = SQLiteSessionStore(tmp_path / "event-append-faults.sqlite")
        session_id = f"event-append-{store_kind}"
        await _create_session(store, session_id)
        precommit = Event(
            id=f"event-append-precommit-{store_kind}",
            type=EventType.TURN_COMPLETED,
            session_id=session_id,
        )
        lost_ack = Event(
            id=f"event-append-lost-ack-{store_kind}",
            type=EventType.TURN_COMPLETED,
            session_id=session_id,
        )
        try:
            precommit_rule = SessionOperationFaultRule(
                rule_id="event-precommit",
                selector=SessionOperationSelector(event_types=frozenset({precommit.type})),
                actions=(FailBeforeTransform(),),
                on_exhausted=MatchPolicy.DELEGATE,
            )
            async with SessionOperationFaultHarness(
                store,
                rules=(precommit_rule,),
                boundary=PublicationBoundary.EVENT_APPEND,
            ) as precommit_faults:
                with pytest.raises(ConnectionError, match="before append"):
                    await store.append_event(session_id, precommit)
                assert await store.load_events(session_id) == []
                await store.append_event(session_id, precommit)

            lost_ack_rule = SessionOperationFaultRule(
                rule_id="event-lost-ack",
                selector=SessionOperationSelector(event_types=frozenset({lost_ack.type})),
                actions=(CommitThenRaise(),),
            )
            async with SessionOperationFaultHarness(
                store,
                rules=(lost_ack_rule,),
                boundary=PublicationBoundary.EVENT_APPEND,
            ) as lost_ack_faults:
                with pytest.raises(ConnectionError, match="lost after commit"):
                    await store.append_event(session_id, lost_ack)

            assert [event.id for event in await store.load_events(session_id)] == [
                precommit.id,
                lost_ack.id,
            ]
            assert precommit_faults.trace[0].committed is CommitEvidence.NO
            assert precommit_faults.trace[1].committed is CommitEvidence.YES
            assert lost_ack_faults.trace[0].committed is CommitEvidence.YES
            assert lost_ack_faults.trace[0].acknowledgement_returned is False
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_label_and_event_selectors_do_not_consume_unrelated_publications() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-selector-session"
        await _create_session(store, session_id)
        rule = SessionOperationFaultRule(
            rule_id="selected-event",
            selector=SessionOperationSelector(
                session_id=session_id,
                event_types=frozenset({EventType.SESSION_COMPLETED}),
                label="selected",
            ),
            actions=(FailBeforeTransform(),),
        )
        async with SessionOperationFaultHarness(store, rules=(rule,)) as harness:
            await _publish_record(store, session_id, "unrelated")
            with (
                harness.label("selected"),
                pytest.raises(ConnectionError, match="before transform"),
            ):
                await _publish_record(
                    store,
                    session_id,
                    "selected",
                    events=[
                        Event(
                            type=EventType.SESSION_COMPLETED,
                            session_id=session_id,
                        )
                    ],
                )

        assert await store.load_session_operation(session_id, "unrelated") == {"value": "committed"}
        assert await store.load_session_operation(session_id, "selected") is None
        assert [record.action for record in harness.trace] == [
            PublicationFaultActionKind.UNMATCHED_DELEGATE,
            PublicationFaultActionKind.FAIL_BEFORE_TRANSFORM,
        ]
        assert harness.trace[0].acknowledgement_returned is True
        assert harness.trace[1].matched_rule_id == "selected-event"

    asyncio.run(run())


def test_unsatisfied_and_exhausted_schedules_fail_closed() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-schedule-session"
        await _create_session(store, session_id)
        rule = SessionOperationFaultRule(
            rule_id="one-shot",
            selector=SessionOperationSelector(idempotency_key="selected"),
            actions=(FailBeforeTransform(),),
        )

        with pytest.raises(SessionOperationFaultScheduleError, match="not satisfied"):
            async with SessionOperationFaultHarness(store, rules=(rule,)):
                pass

        async with SessionOperationFaultHarness(store, rules=(rule,)):
            with pytest.raises(ConnectionError, match="before transform"):
                await _publish_record(store, session_id, "selected")
            with pytest.raises(SessionOperationFaultScheduleError, match="exhausted"):
                await _publish_record(store, session_id, "selected")
        assert await store.load_session_operation(session_id, "selected") is None

    asyncio.run(run())


def test_owner_cancellation_before_schedule_preserves_unsatisfied_failure() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-unsatisfied-cancel-session"
        await _create_session(store, session_id)
        original_publish = store.publish_session_operation
        owner_entered = asyncio.Event()
        owner_wait = asyncio.Event()
        rule = SessionOperationFaultRule(
            rule_id="never-reached",
            selector=SessionOperationSelector(idempotency_key="never-reached"),
            actions=(FailBeforeTransform(),),
        )
        harness = SessionOperationFaultHarness(store, rules=(rule,))

        async def own_harness() -> None:
            async with harness:
                owner_entered.set()
                await owner_wait.wait()

        owner = asyncio.create_task(own_harness())
        await asyncio.wait_for(owner_entered.wait(), timeout=5.0)
        owner.cancel()
        assert owner.cancelling() == 1

        caught_cancellation: asyncio.CancelledError | None = None
        try:
            await owner
        except asyncio.CancelledError as error:
            caught_cancellation = error
        assert caught_cancellation is not None
        unsatisfied = caught_cancellation.__cause__
        assert type(unsatisfied) is SessionOperationFaultScheduleError
        assert str(unsatisfied) == (
            "Publication fault schedule was not satisfied (never-reached:1)."
        )
        assert owner.cancelling() == 1
        assert owner.cancelled() is True
        assert harness.trace == ()
        assert store.publish_session_operation == original_publish
        await _publish_record(store, session_id, "after-unsatisfied-cancel")
        assert await store.load_session_operation(
            session_id,
            "after-unsatisfied-cancel",
        ) == {"value": "committed"}

    asyncio.run(run())


def test_nested_install_is_rejected_and_prior_instance_wrapper_is_restored() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-restore-session"
        await _create_session(store, session_id)
        native_publish = store.publish_session_operation
        prior_calls = 0

        async def prior_wrapper(*args: Any, **kwargs: Any):
            nonlocal prior_calls
            prior_calls += 1
            return await native_publish(*args, **kwargs)

        store.__dict__["publish_session_operation"] = prior_wrapper
        outer_rule = SessionOperationFaultRule(
            rule_id="outer",
            selector=SessionOperationSelector(idempotency_key="outer"),
            actions=(FailBeforeTransform(),),
        )
        inner_rule = SessionOperationFaultRule(
            rule_id="inner",
            selector=SessionOperationSelector(idempotency_key="inner"),
            actions=(FailBeforeTransform(),),
        )
        async with SessionOperationFaultHarness(store, rules=(outer_rule,)):
            with pytest.raises(SessionOperationFaultScheduleError, match="already installed"):
                await SessionOperationFaultHarness(store, rules=(inner_rule,)).__aenter__()
            with pytest.raises(ConnectionError, match="before transform"):
                await _publish_record(store, session_id, "outer")

        assert store.__dict__["publish_session_operation"] is prior_wrapper
        await _publish_record(store, session_id, "after-restore")
        assert prior_calls == 1
        assert await store.load_session_operation(session_id, "after-restore") == {
            "value": "committed"
        }

    asyncio.run(run())


def test_ambiguous_schedule_preserves_body_and_teardown_failures() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-ambiguous-session"
        await _create_session(store, session_id)
        rules = tuple(
            SessionOperationFaultRule(
                rule_id=rule_id,
                selector=SessionOperationSelector(idempotency_key="ambiguous"),
                actions=(FailBeforeTransform(),),
            )
            for rule_id in ("first", "second")
        )
        with pytest.raises(BaseExceptionGroup) as caught:
            async with SessionOperationFaultHarness(store, rules=rules):
                await _publish_record(store, session_id, "ambiguous")

        flattened = caught.value.subgroup(SessionOperationFaultScheduleError)
        assert flattened is not None
        messages = [str(error) for error in _leaf_exceptions(flattened)]
        assert messages == [
            "Multiple publication fault rules matched one operation.",
            "Publication fault schedule was not satisfied (first:1, second:1).",
        ]
        assert await store.load_session_operation(session_id, "ambiguous") is None

    asyncio.run(run())


def _leaf_exceptions(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for nested in error.exceptions for leaf in _leaf_exceptions(nested)]
    return [error]


def test_cancellation_before_transform_restores_store_and_cancels_normally() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-cancel-before-session"
        await _create_session(store, session_id)
        original = store.publish_session_operation
        barrier = PublicationBarrier()
        rule = SessionOperationFaultRule(
            rule_id="cancel-before",
            selector=SessionOperationSelector(idempotency_key="cancel-before"),
            actions=(PauseBeforeTransform(barrier),),
        )
        async with SessionOperationFaultHarness(store, rules=(rule,)) as harness:
            publication = asyncio.create_task(_publish_record(store, session_id, "cancel-before"))
            await barrier.wait_until_entered()
            publication.cancel()
            assert publication.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await publication
            assert publication.cancelled() is True

        assert store.publish_session_operation == original
        assert harness.trace[0].outcome is PublicationFaultOutcome.CANCELLED
        assert harness.trace[0].committed is CommitEvidence.NO
        await _publish_record(store, session_id, "after-cancel")
        assert await store.load_session_operation(session_id, "after-cancel") == {
            "value": "committed"
        }

    asyncio.run(run())


def test_delegate_originated_cancellation_uses_real_rollback_evidence() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-delegate-cancel-session"
        await _create_session(store, session_id)
        rule = SessionOperationFaultRule(
            rule_id="delegate-cancel",
            selector=SessionOperationSelector(idempotency_key="delegate-cancel"),
            actions=(Delegate(),),
        )

        def cancel_transform(_session, _checkpoint, _current_record):
            raise asyncio.CancelledError

        async with SessionOperationFaultHarness(store, rules=(rule,)) as harness:
            publication = asyncio.create_task(
                store.publish_session_operation(
                    session_id,
                    idempotency_key="delegate-cancel",
                    operation_transform=cancel_transform,
                    events=[],
                )
            )
            with pytest.raises(asyncio.CancelledError):
                await publication
            assert publication.cancelling() == 0
            assert publication.cancelled() is True

        [trace] = harness.trace
        assert trace.action is PublicationFaultActionKind.DELEGATE
        assert trace.outcome is PublicationFaultOutcome.CANCELLED
        assert trace.transform_started is True
        assert trace.transform_completed is False
        assert trace.committed is CommitEvidence.NO

    asyncio.run(run())


def test_cancelling_harness_owner_releases_and_drains_blocked_publication() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-owner-cancel-session"
        await _create_session(store, session_id)
        original = store.publish_session_operation
        barrier = PublicationBarrier()
        owner_entered = asyncio.Event()
        owner_wait = asyncio.Event()
        publication: asyncio.Task[None] | None = None
        rule = SessionOperationFaultRule(
            rule_id="owner-cancel",
            selector=SessionOperationSelector(idempotency_key="owner-cancel"),
            actions=(PauseBeforeCommit(barrier),),
        )

        async def own_harness() -> None:
            nonlocal publication
            async with SessionOperationFaultHarness(store, rules=(rule,)):
                publication = asyncio.create_task(
                    _publish_record(store, session_id, "owner-cancel")
                )
                await barrier.wait_until_entered()
                owner_entered.set()
                await owner_wait.wait()

        owner = asyncio.create_task(own_harness())
        await owner_entered.wait()
        owner.cancel()
        assert owner.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert owner.cancelled() is True
        assert publication is not None
        assert publication.done() is True
        assert publication.exception() is None
        assert store.publish_session_operation == original
        assert await store.load_session_operation(session_id, "owner-cancel") == {
            "value": "committed"
        }

    asyncio.run(run())


def test_repeated_owner_cancellation_during_drain_remains_cancelled() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-owner-repeat-cancel-session"
        await _create_session(store, session_id)
        native_publish = store.publish_session_operation
        publication_settling = asyncio.Event()
        allow_publication_to_settle = asyncio.Event()

        async def settling_wrapper(*args: Any, **kwargs: Any):
            result = await native_publish(*args, **kwargs)
            publication_settling.set()
            await allow_publication_to_settle.wait()
            return result

        store.__dict__["publish_session_operation"] = settling_wrapper
        barrier = PublicationBarrier()
        owner_entered = asyncio.Event()
        owner_wait = asyncio.Event()
        publication: asyncio.Task[None] | None = None
        rule = SessionOperationFaultRule(
            rule_id="owner-repeat-cancel",
            selector=SessionOperationSelector(idempotency_key="owner-repeat-cancel"),
            actions=(PauseBeforeTransform(barrier),),
        )

        async def own_harness() -> None:
            nonlocal publication
            async with SessionOperationFaultHarness(store, rules=(rule,)):
                publication = asyncio.create_task(
                    _publish_record(store, session_id, "owner-repeat-cancel")
                )
                await barrier.wait_until_entered()
                owner_entered.set()
                await owner_wait.wait()

        owner = asyncio.create_task(own_harness())
        await asyncio.wait_for(owner_entered.wait(), timeout=5.0)
        owner.cancel()
        assert owner.cancelling() == 1
        await asyncio.wait_for(publication_settling.wait(), timeout=5.0)
        assert publication is not None
        assert publication.done() is False

        owner.cancel()
        assert owner.cancelling() == 2
        await asyncio.sleep(0)
        assert owner.done() is False
        owner.cancel()
        assert owner.cancelling() == 3
        allow_publication_to_settle.set()

        caught_cancellation: asyncio.CancelledError | None = None
        try:
            await owner
        except asyncio.CancelledError as error:
            caught_cancellation = error
        assert caught_cancellation is not None
        assert getattr(caught_cancellation, "__notes__", ()) == [
            "Additional cancellation was delivered while the publication "
            "fault harness drained active calls.",
            "1 additional cancellation signal was delivered while the publication "
            "fault harness drained active calls.",
        ]
        assert owner.cancelling() == 3
        assert owner.cancelled() is True
        assert publication.done() is True
        assert publication.exception() is None
        assert store.__dict__["publish_session_operation"] is settling_wrapper
        assert await store.load_session_operation(session_id, "owner-repeat-cancel") == {
            "value": "committed"
        }

    asyncio.run(run())


def test_body_failure_and_cleanup_cancellation_preserve_authority() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-body-failure-cleanup-cancel-session"
        await _create_session(store, session_id)
        native_publish = store.publish_session_operation
        publication_settling = asyncio.Event()
        allow_publication_to_settle = asyncio.Event()

        async def settling_wrapper(*args: Any, **kwargs: Any):
            result = await native_publish(*args, **kwargs)
            publication_settling.set()
            await allow_publication_to_settle.wait()
            return result

        store.__dict__["publish_session_operation"] = settling_wrapper
        barrier = PublicationBarrier()
        publication: asyncio.Task[None] | None = None
        body_failure = AssertionError("body failed before cleanup cancellation")
        rule = SessionOperationFaultRule(
            rule_id="body-failure-cleanup-cancel",
            selector=SessionOperationSelector(idempotency_key="body-failure-cleanup-cancel"),
            actions=(PauseBeforeTransform(barrier),),
        )

        async def own_harness() -> None:
            nonlocal publication
            async with SessionOperationFaultHarness(store, rules=(rule,)):
                publication = asyncio.create_task(
                    _publish_record(
                        store,
                        session_id,
                        "body-failure-cleanup-cancel",
                    )
                )
                await barrier.wait_until_entered()
                raise body_failure

        owner = asyncio.create_task(own_harness())
        await asyncio.wait_for(publication_settling.wait(), timeout=5.0)
        assert owner.done() is False
        owner.cancel()
        assert owner.cancelling() == 1
        allow_publication_to_settle.set()

        caught_cancellation: asyncio.CancelledError | None = None
        try:
            await owner
        except asyncio.CancelledError as error:
            caught_cancellation = error
        assert caught_cancellation is not None
        assert caught_cancellation.__cause__ is body_failure
        assert _leaf_exceptions(caught_cancellation.__cause__) == [body_failure]
        assert owner.cancelling() == 1
        assert owner.cancelled() is True
        assert publication is not None
        assert publication.done() is True
        assert publication.exception() is None
        assert store.__dict__["publish_session_operation"] is settling_wrapper
        assert await store.load_session_operation(
            session_id,
            "body-failure-cleanup-cancel",
        ) == {"value": "committed"}

    asyncio.run(run())


def test_body_cancellation_and_cleanup_failure_preserve_authority() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-body-cancel-cleanup-failure-session"
        await _create_session(store, session_id)
        original_publish = store.publish_session_operation
        barrier = PublicationBarrier()
        provider_failure = ConnectionError("provider request was cancelled")
        nested_failure = RuntimeError("provider cleanup also failed")
        nested_cause = BaseExceptionGroup(
            "provider cancellation failures",
            [provider_failure, BaseExceptionGroup("provider cleanup", [nested_failure])],
        )
        cleanup_failure = RuntimeError("barrier release failed")
        native_release = barrier.release

        def failing_release() -> None:
            native_release()
            raise cleanup_failure

        barrier.__dict__["release"] = failing_release
        publication: asyncio.Task[None] | None = None
        owner_entered = asyncio.Event()
        owner_wait = asyncio.Event()
        rule = SessionOperationFaultRule(
            rule_id="body-cancel-cleanup-failure",
            selector=SessionOperationSelector(idempotency_key="body-cancel-cleanup-failure"),
            actions=(PauseBeforeTransform(barrier),),
        )

        async def own_harness() -> None:
            nonlocal publication
            async with SessionOperationFaultHarness(store, rules=(rule,)):
                publication = asyncio.create_task(
                    _publish_record(
                        store,
                        session_id,
                        "body-cancel-cleanup-failure",
                    )
                )
                await barrier.wait_until_entered()
                owner_entered.set()
                try:
                    await owner_wait.wait()
                except asyncio.CancelledError as cancellation:
                    raise cancellation from nested_cause

        owner = asyncio.create_task(own_harness())
        await asyncio.wait_for(owner_entered.wait(), timeout=5.0)
        owner.cancel()
        assert owner.cancelling() == 1

        caught_cancellation: asyncio.CancelledError | None = None
        try:
            await owner
        except asyncio.CancelledError as error:
            caught_cancellation = error
        assert caught_cancellation is not None
        combined_cause = caught_cancellation.__cause__
        assert isinstance(combined_cause, BaseExceptionGroup)
        assert combined_cause.exceptions == (nested_cause, cleanup_failure)
        assert _leaf_exceptions(combined_cause) == [
            provider_failure,
            nested_failure,
            cleanup_failure,
        ]
        assert owner.cancelling() == 1
        assert owner.cancelled() is True
        assert publication is not None
        assert publication.done() is True
        assert publication.exception() is None
        assert store.publish_session_operation == original_publish
        assert await store.load_session_operation(
            session_id,
            "body-cancel-cleanup-failure",
        ) == {"value": "committed"}

    asyncio.run(run())


def test_repeated_cancellation_at_commit_guard_settles_before_task_finishes() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-cancel-commit-session"
        await _create_session(store, session_id)
        barrier = PublicationBarrier()
        rule = SessionOperationFaultRule(
            rule_id="cancel-at-commit",
            selector=SessionOperationSelector(idempotency_key="cancel-at-commit"),
            actions=(PauseBeforeCommit(barrier),),
        )
        async with SessionOperationFaultHarness(store, rules=(rule,)) as harness:
            publication = asyncio.create_task(
                _publish_record(store, session_id, "cancel-at-commit")
            )
            await barrier.wait_until_entered()
            publication.cancel()
            publication.cancel()
            assert publication.cancelling() == 2
            barrier.release()
            with pytest.raises(asyncio.CancelledError):
                await publication
            assert publication.cancelled() is True

        assert await store.load_session_operation(session_id, "cancel-at-commit") == {
            "value": "committed"
        }
        [trace] = harness.trace
        assert trace.action is PublicationFaultActionKind.PAUSE_BEFORE_COMMIT
        assert trace.action_reached is True
        assert trace.transform_completed is True
        assert trace.committed is CommitEvidence.UNKNOWN
        assert trace.acknowledgement_returned is False

    asyncio.run(run())


def test_cancellation_after_commit_preserves_durable_state_and_cancels_normally() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-cancel-after-session"
        await _create_session(store, session_id)
        barrier = PublicationBarrier()
        rule = SessionOperationFaultRule(
            rule_id="cancel-after",
            selector=SessionOperationSelector(idempotency_key="cancel-after"),
            actions=(PauseAfterCommit(barrier),),
            on_exhausted=MatchPolicy.DELEGATE,
        )
        async with SessionOperationFaultHarness(store, rules=(rule,)) as harness:
            publication = asyncio.create_task(_publish_record(store, session_id, "cancel-after"))
            await barrier.wait_until_entered()
            assert await store.load_session_operation(session_id, "cancel-after") == {
                "value": "committed"
            }
            await _publish_record(store, session_id, "cancel-after")
            publication.cancel()
            assert publication.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await publication
            assert publication.cancelled() is True

        paused, replay = harness.trace
        assert paused.action is PublicationFaultActionKind.PAUSE_AFTER_COMMIT
        assert paused.outcome is PublicationFaultOutcome.CANCELLED
        assert paused.action_reached is True
        assert paused.transform_completed is True
        assert paused.committed is CommitEvidence.YES
        assert paused.acknowledgement_returned is False
        assert replay.action is PublicationFaultActionKind.EXHAUSTED_DELEGATE
        assert replay.committed is CommitEvidence.YES
        assert replay.acknowledgement_returned is True

    asyncio.run(run())


def test_sqlite_lost_acknowledgement_survives_store_reopen(tmp_path: Path) -> None:
    async def run() -> None:
        database_path = tmp_path / "publication-reopen.sqlite"
        session_id = "sqlite-publication-reopen"
        store = SQLiteSessionStore(database_path)
        await _create_session(store, session_id)
        rule = SessionOperationFaultRule(
            rule_id="sqlite-lost-ack",
            selector=SessionOperationSelector(idempotency_key="sqlite-lost-ack"),
            actions=(CommitThenRaise(),),
        )
        async with SessionOperationFaultHarness(store, rules=(rule,)):
            with pytest.raises(ConnectionError, match="acknowledgement was lost"):
                await _publish_record(store, session_id, "sqlite-lost-ack")
        await _close_store(store)

        reopened = SQLiteSessionStore(database_path)
        try:
            assert await reopened.load_session_operation(
                session_id,
                "sqlite-lost-ack",
            ) == {"value": "committed"}
        finally:
            await _close_store(reopened)

    asyncio.run(run())


def test_timeout_and_body_failure_release_barriers_and_drain_calls() -> None:
    async def run() -> None:
        timeout_store = InMemorySessionStore()
        await _create_session(timeout_store, "fault-timeout-session")
        timeout_barrier = PublicationBarrier()
        timeout_rule = SessionOperationFaultRule(
            rule_id="timeout",
            selector=SessionOperationSelector(idempotency_key="timeout"),
            actions=(PauseBeforeTransform(timeout_barrier),),
        )
        async with SessionOperationFaultHarness(timeout_store, rules=(timeout_rule,)):
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.01):
                    await _publish_record(timeout_store, "fault-timeout-session", "timeout")
        assert (
            await timeout_store.load_session_operation("fault-timeout-session", "timeout") is None
        )

        assertion_store = InMemorySessionStore()
        await _create_session(assertion_store, "fault-assertion-session")
        assertion_barrier = PublicationBarrier()
        assertion_rule = SessionOperationFaultRule(
            rule_id="assertion",
            selector=SessionOperationSelector(idempotency_key="assertion"),
            actions=(PauseBeforeTransform(assertion_barrier),),
        )
        publication: asyncio.Task[None] | None = None
        with pytest.raises(AssertionError, match="body failed"):
            async with SessionOperationFaultHarness(
                assertion_store,
                rules=(assertion_rule,),
            ):
                publication = asyncio.create_task(
                    _publish_record(assertion_store, "fault-assertion-session", "assertion")
                )
                await assertion_barrier.wait_until_entered()
                raise AssertionError("body failed")
        assert publication is not None
        assert publication.done() is True
        assert publication.exception() is None
        assert await assertion_store.load_session_operation(
            "fault-assertion-session",
            "assertion",
        ) == {"value": "committed"}

    asyncio.run(run())


def test_trace_is_bounded_and_returned_as_a_defensive_tuple() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id = "fault-trace-session"
        await _create_session(store, session_id)
        rule = SessionOperationFaultRule(
            rule_id="trace",
            selector=SessionOperationSelector(idempotency_key="trace-scheduled"),
            actions=(Delegate(),),
        )
        async with SessionOperationFaultHarness(
            store,
            rules=(rule,),
            trace_limit=1,
        ) as harness:
            await _publish_record(store, session_id, "trace-incidental-1")
            await _publish_record(store, session_id, "trace-incidental-2")
            await _publish_record(store, session_id, "trace-scheduled")

        first_snapshot = harness.trace
        second_snapshot = harness.trace
        assert type(first_snapshot) is tuple
        assert first_snapshot == second_snapshot
        assert first_snapshot is not second_snapshot
        assert len(first_snapshot) == 1
        assert first_snapshot[0].sequence == 3
        assert first_snapshot[0].action is PublicationFaultActionKind.DELEGATE
        assert harness.dropped_trace_entries == 2

        with pytest.raises(
            ValueError,
            match="trace_limit must reserve one entry for every scheduled action",
        ):
            SessionOperationFaultHarness(
                store,
                rules=(
                    SessionOperationFaultRule(
                        rule_id="too-small",
                        selector=SessionOperationSelector(idempotency_key_prefix="trace-"),
                        actions=(Delegate(count=2),),
                    ),
                ),
                trace_limit=1,
            )

    asyncio.run(run())


def test_rejected_values_and_invalid_event_batches_are_diagnostically_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "secret-publication-fault-canary"

    class Hostile:
        def __repr__(self) -> str:
            return canary

        def __str__(self) -> str:
            return canary

    captured_errors: list[BaseException] = []
    with warnings.catch_warnings(record=True) as captured_warnings, caplog.at_level(logging.DEBUG):
        with pytest.raises(TypeError) as wrong_selector:
            SessionOperationSelector(session_id=cast("Any", Hostile()))
        captured_errors.append(wrong_selector.value)
        with pytest.raises(TypeError) as wrong_count:
            FailBeforeTransform(count=cast("Any", True))
        captured_errors.append(wrong_count.value)
        with pytest.raises(TypeError) as wrong_disposition:
            PauseAfterCommit(
                PublicationBarrier(),
                then=cast("Any", Hostile()),
            )
        captured_errors.append(wrong_disposition.value)

        async def run() -> None:
            store = InMemorySessionStore()
            session_id = "fault-diagnostic-session"
            await _create_session(store, session_id)
            rule = SessionOperationFaultRule(
                rule_id="diagnostic",
                selector=SessionOperationSelector(
                    session_id=session_id,
                    idempotency_key="valid-event",
                    event_types=frozenset({EventType.SESSION_COMPLETED}),
                ),
                actions=(FailBeforeTransform(),),
            )
            async with SessionOperationFaultHarness(
                store,
                rules=(rule,),
                on_unmatched=MatchPolicy.FAIL,
            ):
                with pytest.raises(SessionOperationFaultScheduleError) as invalid_batch:
                    await _publish_record(
                        store,
                        session_id,
                        "valid-event",
                        events=cast("list[Event]", [Hostile()]),
                    )
                captured_errors.append(invalid_batch.value)

                mutated_event = Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                )
                object.__setattr__(mutated_event, "type", Hostile())
                object.__setattr__(mutated_event, "session_id", Hostile())
                with pytest.raises(SessionOperationFaultScheduleError) as mutated_batch:
                    await _publish_record(
                        store,
                        session_id,
                        "valid-event",
                        events=[mutated_event],
                    )
                captured_errors.append(mutated_batch.value)

                valid_event = Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                )
                with pytest.raises(SessionOperationFaultScheduleError) as hostile_session:
                    await _publish_record(
                        store,
                        cast("str", Hostile()),
                        "valid-event",
                        events=[valid_event],
                    )
                captured_errors.append(hostile_session.value)
                with pytest.raises(SessionOperationFaultScheduleError) as hostile_key:
                    await _publish_record(
                        store,
                        session_id,
                        cast("str", Hostile()),
                        events=[valid_event],
                    )
                captured_errors.append(hostile_key.value)

                with pytest.raises(ConnectionError):
                    await _publish_record(
                        store,
                        session_id,
                        "valid-event",
                        events=[valid_event],
                    )

        asyncio.run(run())

    output = capsys.readouterr()
    diagnostic_text = "\n".join(
        [
            *(str(error) for error in captured_errors),
            *(repr(error) for error in captured_errors),
            *(str(item.message) for item in captured_warnings),
            caplog.text,
            output.out,
            output.err,
        ]
    )
    assert canary not in diagnostic_text


def test_configuration_rejects_boolean_counts_and_reused_barriers() -> None:
    with pytest.raises(TypeError, match="integer"):
        FailBeforeTransform(count=cast("Any", False))
    barrier = PublicationBarrier()
    with pytest.raises(ValueError, match="distinct barrier"):
        SessionOperationFaultHarness(
            InMemorySessionStore(),
            rules=(
                SessionOperationFaultRule(
                    rule_id="first",
                    selector=SessionOperationSelector(idempotency_key="first"),
                    actions=(PauseBeforeTransform(barrier),),
                ),
                SessionOperationFaultRule(
                    rule_id="second",
                    selector=SessionOperationSelector(idempotency_key="second"),
                    actions=(PauseBeforeTransform(barrier),),
                    on_exhausted=MatchPolicy.DELEGATE,
                ),
            ),
        )
