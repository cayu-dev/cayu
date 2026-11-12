"""Attested lost executions settle promptly without weakening ownership checks."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest
from tests.runtime.test_orphaned_session_reconciliation import (
    _application,
    _Blocking,
    _CrashingProvider,
    _failure,
    _latest_interaction_event,
    _orphan,
    _request,
    _slot,
)
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import EventType, SessionStatus
from cayu.storage import open_application_stores
from cayu.tasks import TaskCreate


def _run_crashing_worker() -> None:
    async def run() -> None:
        stores = open_application_stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"],
            sqlite_path=os.environ["CAYU_ORPHAN_TEST_STORE"],
            event_watchers=False,
        )
        app = _application(
            stores.session_store,
            environment=_slot(),
            task_store=stores.task_store,
            provider=_CrashingProvider if os.environ["CAYU_ORPHAN_TEST_CRASH"] == "model" else None,
        )

        async def run_session():
            async for _ in app.run(_request(os.environ["CAYU_ORPHAN_TEST_SESSION"])):
                pass

        if os.environ["CAYU_ORPHAN_TEST_CRASH"] == "handler":
            from cayu.tasks.worker import run_task_worker

            await stores.task_store.create_task(
                TaskCreate(task_id="job", type="scan", restartable=True)
            )

            async def handler(_app, _task, _worker_id):
                await run_session()

            await run_task_worker(
                app,
                stores.task_store,
                handler,
                worker_id="lost-worker",
                max_tasks=1,
                lease_seconds=1,
                reclaim=False,
            )
        else:
            await run_session()
        raise AssertionError("The worker did not reach the crashing tool.")

    asyncio.run(run())


def _fresh_orphan(path, session_id, database_url, *, crash="tool") -> None:
    _orphan(
        path,
        session_id,
        database_url=database_url,
        crash=crash,
        entry="_run_crashing_worker",
        module=__name__,
    )


def test_default_sweep_reconciles_a_fresh_retired_worker_run(tmp_path, database_url) -> None:
    path = tmp_path / "cayu.db"
    _fresh_orphan(path, "scan", database_url)

    async def scenario() -> None:
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        store = stores.session_store
        reader = _application(
            store,
            task_store=stores.task_store,
            environment=None,
        )
        before = await store.load("scan")
        assert before.status is SessionStatus.RUNNING
        assert (await store.inspect_session_execution("scan")).state == "owner_lost"
        try:
            results = await reader.reconcile_orphaned_sessions()
            assert len(results) == 1
            [result] = results
            assert result.status is SessionStatus.FAILED, result.message
            payload = await _failure(store, "scan")
            assert payload["reason"] == "orphaned_session_reconciled"
            assert [call["tool_call_id"] for call in payload["tool_effect_outcome_unknown"]] == [
                "call-1"
            ]
            assert await _latest_interaction_event(store, "scan") is EventType.INTERACTION_FAILED
            # No reconnection or release of the unavailable environment is invented.
            assert (await store.load("scan")).environment_name == before.environment_name
            events = await store.load_events("scan")
            assert [event.type for event in events].count(EventType.SESSION_FAILED) == 1
            assert await reader.reconcile_orphaned_sessions() == ()
        finally:
            await reader.aclose()
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("ambiguous_task", [False, True])
@pytest.mark.parametrize("crash", ["tool", "model"])
def test_registered_lost_owner_settles_without_task_purpose(
    tmp_path, database_url, ambiguous_task, crash
) -> None:
    path = tmp_path / "cayu.db"
    _fresh_orphan(path, "conversation", database_url, crash=crash)

    async def scenario() -> None:
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        reader = _application(
            stores.session_store, environment=_slot(), task_store=stores.task_store
        )
        try:
            if ambiguous_task:
                # A raw store attachment has no authenticated session incarnation.
                await stores.task_store.create_task(
                    TaskCreate(task_id="finished-job", type="scan", session_id="conversation")
                )
                await stores.task_store.cancel_task("finished-job")
            before = await stores.session_store.load("conversation")
            assert before.status is SessionStatus.RUNNING
            results = await reader.reconcile_orphaned_sessions()
            assert len(results) == 1
            assert results[0].status is SessionStatus.INTERRUPTED
            current = await stores.session_store.load("conversation")
            assert current.instance_id == before.instance_id
            assert current.run_epoch > before.run_epoch
            assert current.status is SessionStatus.INTERRUPTED
            assert await reader.reconcile_orphaned_sessions() == ()
        finally:
            await reader.aclose()
            await stores.close()

    asyncio.run(scenario())


def test_default_sweep_finishes_partial_terminal_publication(tmp_path, database_url, monkeypatch):
    path = tmp_path / "cayu.db"
    _fresh_orphan(path, "scan", database_url)

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        reader = _application(stores.session_store, task_store=stores.task_store)
        finalization = reader._incomplete_recovery.terminal_finalization
        repair = finalization.repair
        calls = 0

        async def lost_once(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("terminal publication lost after durable abandonment")
            return await repair(**kwargs)

        monkeypatch.setattr(finalization, "repair", lost_once)
        try:
            await reader.reconcile_orphaned_sessions()
            assert (await stores.session_store.load("scan")).status is SessionStatus.FAILED
            # No inactivity grace applies to publishing an already committed abandonment.
            assert (
                await _latest_interaction_event(stores.session_store, "scan")
                is EventType.INTERACTION_FAILED
            )
            assert (await _failure(stores.session_store, "scan"))[
                "reason"
            ] == "orphaned_session_reconciled"
            assert await reader.reconcile_orphaned_sessions() == ()
        finally:
            await reader.aclose()
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_remote_lease_expiry_fences_late_tool_result_without_claiming_abort(
    backend, request, sqlite_resources, monkeypatch
):
    from tests.core.test_session_execution_presence import _stores
    from tests.runtime.test_orphaned_session_reconciliation import _Blocking

    from cayu.runtime import _session_execution_presence as presence_module
    from cayu.sessions import SessionExecutionConfig
    from cayu.sessions._process_liveness import ProcessIdentity

    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            now = [datetime.now(UTC)]
            reader_store = reopen()
            if backend == "postgres":
                from cayu.storage import _postgres_locked_reads as locked_reads

                # Both single and pipelined server-clock reads use this decoder.
                # Keep the real queries, locks and writes, controlling only time.
                decode = locked_reads.decode_store_now

                def store_now(row):
                    decode(row)
                    return now[0]

                monkeypatch.setattr(locked_reads, "decode_store_now", store_now)
            else:
                for session_store in (store, reader_store):
                    monkeypatch.setattr(session_store, "_ownership_clock", lambda: now[0])
            tool = _Blocking()
            writer = _application(store, environment=_slot(), tool=tool)
            writer._session_control.execution_presence.config = SessionExecutionConfig(
                heartbeat_interval_seconds=0.05, lease_seconds=0.3
            )
            monkeypatch.setattr(
                presence_module,
                "current_process_identity",
                lambda: ProcessIdentity(
                    host_boot_id="another-container", pid=os.getpid(), start_id="remote"
                ),
            )
            renew = store._renew_session_execution
            outage = False
            partition_observed = asyncio.Event()

            async def renew_unless_partitioned(*args, **kwargs):
                if outage:
                    partition_observed.set()
                    raise OSError("controlled writer partition")
                return await renew(*args, **kwargs)

            monkeypatch.setattr(store, "_renew_session_execution", renew_unless_partitioned)

            async def run():
                return [event async for event in writer.run(_request("remote-scan"))]

            task = asyncio.create_task(run())
            reader = _application(reader_store)
            try:
                try:
                    await asyncio.wait_for(tool.started.wait(), 30)
                except TimeoutError:
                    if task.done():
                        task.result()
                    raise
                before = await store.load("remote-scan")
                assert await reader.reconcile_orphaned_sessions() == ()
                live = await reader.session_store.inspect_session_execution("remote-scan")
                assert live.state == "executing"
                outage = True
                await asyncio.wait_for(partition_observed.wait(), 30)
                # Expiry follows an observed failed renewal. Host scheduling
                # cannot expire the lease during the initial live-owner check.
                now[0] = live.lease_expires_at + timedelta(microseconds=1)
                assert (
                    await reader.session_store.inspect_session_execution("remote-scan")
                ).state == "owner_lost"
                [result] = await reader.reconcile_orphaned_sessions()
                assert result.status is SessionStatus.FAILED, result.message
                assert not task.done(), "Lease loss is not proof that the dispatched tool stopped"
                payload = await _failure(store, "remote-scan")
                assert payload["tool_effect_outcome_unknown"][0]["tool_call_id"] == "call-1"
                settled = await store.load("remote-scan")
                assert settled.environment_name == before.environment_name
                events = await store.load_events("remote-scan")
                outage = False
                tool.release.set()
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 15)
                assert await store.load("remote-scan") == settled
                assert await store.load_events("remote-scan") == events
                assert await reader.reconcile_orphaned_sessions() == ()
            finally:
                outage = False
                tool.release.set()
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 15)
                await writer.aclose()
                await reader.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("race", ["new_task", "successor_epoch"])
def test_fresh_orphan_claim_rechecks_attachment_and_execution(
    tmp_path, database_url, monkeypatch, race
):
    from cayu.sessions import IncompleteSessionRecoveryAction
    from cayu.tasks import TaskStatus

    path = tmp_path / "cayu.db"
    _fresh_orphan(path, "scan", database_url)

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        reader = _application(stores.session_store, task_store=stores.task_store)
        boundary = reader._session_abandonment
        abandon = boundary._abandon_orphan_without_environment
        injected = False

        async def change_after_evidence(session, **kwargs):
            nonlocal injected
            if not injected:
                injected = True
                if race == "new_task":
                    await reader.create_task(
                        TaskCreate(task_id="successor", type="scan", session_id="scan")
                    )
                else:
                    successor = await stores.session_store.fence_stalled_run(
                        "scan", statuses={SessionStatus.RUNNING}, inactive_for_seconds=0
                    )
                    assert successor is not None
            return await abandon(session, **kwargs)

        monkeypatch.setattr(boundary, "_abandon_orphan_without_environment", change_after_evidence)
        try:
            [result] = await reader.reconcile_orphaned_sessions()
            assert injected
            assert result.actions == (IncompleteSessionRecoveryAction.FAILED,)
            assert (await stores.session_store.load("scan")).status is SessionStatus.RUNNING
            assert not any(
                e.type is EventType.SESSION_FAILED
                for e in await stores.session_store.load_events("scan")
            )
            if race == "new_task":
                task = await stores.task_store.load_task("successor")
                assert task.status is TaskStatus.PENDING
                assert task.worker_id is None
                assert await reader.reconcile_orphaned_sessions() == ()
        finally:
            await reader.aclose()
            await stores.close()

    asyncio.run(scenario())


def test_fast_scan_visits_every_keyset_page(monkeypatch):
    from cayu.sessions import InMemorySessionStore
    from cayu.sessions.records import SessionIdentity

    async def scenario():
        store = InMemorySessionStore()
        reader = _application(store)
        expected = set()
        for index in range(1003):
            sid = f"fresh-{index:04}"
            session = await store.create(
                _request(sid), identity=SessionIdentity(provider_name="fake", model="fake-model")
            )
            await store.update_status(session.id, SessionStatus.RUNNING)
            expected.add(sid)
        visited = []
        inspect = reader._session_abandonment._can_reconcile_without_inactivity

        async def record(session):
            visited.append(session.id)
            return await inspect(session)

        monkeypatch.setattr(
            reader._session_abandonment, "_can_reconcile_without_inactivity", record
        )
        try:
            assert await reader.reconcile_orphaned_sessions() == ()
            assert set(visited) == expected
            assert len(visited) == len(expected)
            assert all(
                [(await store.load(sid)).status is SessionStatus.RUNNING for sid in expected]
            )
        finally:
            await reader.aclose()

    asyncio.run(scenario())


def test_default_sweep_retries_previously_committed_abandonment(
    tmp_path, database_url, monkeypatch
):
    path = tmp_path / "cayu.db"
    _fresh_orphan(path, "scan", database_url)

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        first = _application(stores.session_store, task_store=stores.task_store)

        async def unavailable(**kwargs):
            raise OSError("terminal publication unavailable after commit")

        monkeypatch.setattr(first._incomplete_recovery.terminal_finalization, "repair", unavailable)
        try:
            await first.reconcile_orphaned_sessions(inactive_for_seconds=0)
            assert (await stores.session_store.load("scan")).status is SessionStatus.FAILED
            assert (
                await _latest_interaction_event(stores.session_store, "scan")
                is EventType.INTERACTION_STARTED
            )
        finally:
            await first.aclose()
        replacement = _application(stores.session_store, task_store=stores.task_store)
        try:
            await replacement.reconcile_orphaned_sessions()
            assert (
                await _latest_interaction_event(stores.session_store, "scan")
                is EventType.INTERACTION_FAILED
            )
            assert (await _failure(stores.session_store, "scan"))[
                "reason"
            ] == "orphaned_session_reconciled"
            assert await replacement.reconcile_orphaned_sessions() == ()
        finally:
            await replacement.aclose()
            await stores.close()

    asyncio.run(scenario())


def test_registered_terminal_task_is_preserved_by_fast_recovery(tmp_path, database_url):
    path = tmp_path / "cayu.db"
    _fresh_orphan(path, "scan", database_url)

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        reader = _application(
            stores.session_store, environment=_slot(), task_store=stores.task_store
        )
        await reader.create_task(TaskCreate(task_id="finished", type="scan", session_id="scan"))
        await stores.task_store.cancel_task("finished")
        before = await stores.task_store.load_task("finished")
        try:
            [result] = await reader.reconcile_orphaned_sessions()
            assert result.status is SessionStatus.INTERRUPTED, result.message
            assert await stores.task_store.load_task("finished") == before
            events = await stores.session_store.load_events("scan")
            assert [e.type for e in events].count(EventType.SESSION_INTERRUPTED) == 1
            assert await reader.reconcile_orphaned_sessions() == ()
        finally:
            await reader.aclose()
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("resume_successor", [False, True])
def test_registered_recovery_preserves_a_later_task_and_dispatched_successor(
    tmp_path, database_url, monkeypatch, resume_successor
):
    from cayu import Message, ResumeRequest
    from cayu.sessions import IncompleteSessionRecoveryAction

    path = tmp_path / "cayu.db"
    _fresh_orphan(path, "scan", database_url, crash="model")

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        reader = _application(
            stores.session_store,
            environment=_slot(),
            task_store=stores.task_store,
            tool=_Blocking(),
        )
        tool = _Blocking()
        successor = _application(
            stores.session_store, environment=_slot(), task_store=stores.task_store, tool=tool
        )
        await reader.create_task(TaskCreate(task_id="finished", type="scan", session_id="scan"))
        await stores.task_store.cancel_task("finished")
        ownership = reader._incomplete_recovery._recovery_ownership
        claim = ownership.claim
        pending = None
        execution = None

        async def race_after_evidence(**kwargs):
            nonlocal pending, execution
            if pending is None:
                pending = await successor.create_task(
                    TaskCreate(task_id="new-pending", type="scan", session_id="scan")
                )
                if resume_successor:

                    async def resume():
                        return [
                            e
                            async for e in successor.resume(
                                ResumeRequest(
                                    session_id="scan", messages=[Message.text("user", "continue")]
                                )
                            )
                        ]

                    execution = asyncio.create_task(resume())
                    # Real public continuation owns and dispatches its tool before
                    # the old sweep can attempt its native claim.
                    await asyncio.wait_for(tool.started.wait(), 30)
            return await claim(**kwargs)

        monkeypatch.setattr(ownership, "claim", race_after_evidence)
        try:
            [result] = await reader.reconcile_orphaned_sessions()
            assert pending is not None
            assert await stores.task_store.load_task("new-pending") == pending
            if resume_successor:
                assert tool.started.is_set(), (
                    [(event.type, event.payload) for event in execution.result()]
                    if execution.done()
                    else "Successor is still running before its tool dispatch"
                )
                assert not execution.done()
                assert result.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
                assert (await stores.session_store.load("scan")).status is SessionStatus.RUNNING
            else:
                assert result.status is SessionStatus.INTERRUPTED, result.message
        finally:
            tool.release.set()
            if execution is not None:
                await asyncio.wait_for(asyncio.gather(execution, return_exceptions=True), 15)
            await reader.aclose()
            await successor.aclose()
            await stores.close()

    asyncio.run(scenario())


def test_requeued_handler_task_is_not_mutated_when_its_old_session_is_reconciled(
    tmp_path, database_url
):
    from cayu.tasks import TaskStatus
    from cayu.tasks.restart import restart_abandoned_task

    path = tmp_path / "cayu.db"
    _fresh_orphan(path, "scan", database_url, crash="handler")

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        reader = _application(stores.session_store, task_store=stores.task_store)
        try:
            old = await stores.task_store.load_task("job")
            assert old.session_id is None
            assert (await stores.session_store.load("scan")).metadata["cayu:task_id"] == "job"
            # The subprocess has exited; wait for the native task lease before
            # using the existing authenticated worker-loss restart entrance.
            now = await stores.task_store.lease_clock()
            await asyncio.sleep(max(0, (old.lease_expires_at - now).total_seconds()) + 0.05)
            await stores.task_store.reclaim_expired()
            restarted = await restart_abandoned_task(
                stores.task_store,
                "job",
                validator_id="tests.process-exit",
                validator_version="1",
                evidence_id="subprocess-exited-137",
            )
            assert restarted.status is TaskStatus.PENDING
            successor = await stores.task_store.claim_task("replacement-worker")
            assert successor.id == "job"
            assert successor.claim_generation > old.claim_generation
            [result] = await reader.reconcile_orphaned_sessions()
            assert result.status is SessionStatus.FAILED, result.message
            assert await stores.task_store.load_task("job") == successor
            assert await reader.reconcile_orphaned_sessions() == ()
        finally:
            await reader.aclose()
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("publication", ["session_event", "interaction"])
@pytest.mark.parametrize("registered_on_restart", [False, True])
def test_default_sweep_finishes_interruption_its_prior_recovery_committed(
    tmp_path, database_url, monkeypatch, publication, registered_on_restart
):
    from cayu.sessions import IncompleteSessionRecoveryAction
    from cayu.sessions.queries import SessionQuery

    path = tmp_path / "cayu.db"
    _fresh_orphan(path, "scan", database_url)

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        first = _application(
            stores.session_store, environment=_slot(), task_store=stores.task_store
        )
        await first.create_task(TaskCreate(task_id="finished", type="scan", session_id="scan"))
        await stores.task_store.cancel_task("finished")
        writer = first._incomplete_recovery._event_writer
        emit = writer.emit

        async def unavailable(event, *args, **kwargs):
            if event.type is EventType.SESSION_INTERRUPTED:
                raise OSError("interruption committed before terminal event publication")
            return await emit(event, *args, **kwargs)

        async def interaction_unavailable(**kwargs):
            raise OSError("session event durable before interaction publication")

        if publication == "session_event":
            monkeypatch.setattr(writer, "emit", unavailable)
        else:
            monkeypatch.setattr(
                first._session_recovery,
                "_publish_durable_recovery_interaction_transition",
                interaction_unavailable,
            )
        try:
            await first.reconcile_orphaned_sessions(inactive_for_seconds=0)
            assert (await stores.session_store.load("scan")).status is SessionStatus.INTERRUPTED
            checkpoint = await stores.session_store.load_checkpoint("scan")
            assert ("pending_session_interrupt" in checkpoint) == (publication == "session_event")
            # Recovery-owned interruptions retain their own marker, not an ordinary
            # owed outcome. Removing grace from owed-outcome listing cannot repair this.
            owed = await stores.session_store.list_sessions_with_unsettled_invocation_terminal_decision(
                SessionQuery(status=SessionStatus.INTERRUPTED, inactive_for_seconds=0)
            )
            assert not owed.sessions
            before = await stores.session_store.load("scan")
            effect_events = [
                event
                for event in await stores.session_store.load_events("scan")
                if event.type
                in {
                    EventType.TOOL_CALL_STARTED,
                    EventType.TOOL_CALL_COMPLETED,
                    EventType.TOOL_CALL_FAILED,
                    EventType.TOOL_EFFECT_OUTCOME_UNKNOWN,
                }
            ]
            assert (
                sum(event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for event in effect_events)
                == 1
            )
        finally:
            await first.aclose()
        replacement = _application(
            stores.session_store,
            environment=_slot() if registered_on_restart else None,
            task_store=stores.task_store,
        )
        try:
            repaired = await replacement.reconcile_orphaned_sessions()
            assert all(
                IncompleteSessionRecoveryAction.FAILED not in result.actions for result in repaired
            )
            if publication == "session_event":
                assert len(repaired) == 1
                assert repaired[0].status is SessionStatus.INTERRUPTED
                assert repaired[0].actions == (
                    IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
                )
            else:
                assert repaired == ()
            after = await stores.session_store.load("scan")
            assert (after.instance_id, after.status, after.environment_name) == (
                before.instance_id,
                SessionStatus.INTERRUPTED,
                before.environment_name,
            )
            after_checkpoint = await stores.session_store.load_checkpoint("scan")
            assert after_checkpoint["pending_tool_round"] == checkpoint["pending_tool_round"]
            events = await stores.session_store.load_events("scan")
            assert [
                event
                for event in events
                if event.type
                in {
                    EventType.TOOL_CALL_STARTED,
                    EventType.TOOL_CALL_COMPLETED,
                    EventType.TOOL_CALL_FAILED,
                    EventType.TOOL_EFFECT_OUTCOME_UNKNOWN,
                }
            ] == effect_events
            assert [e.type for e in events].count(EventType.SESSION_INTERRUPTED) == 1
            assert (
                await _latest_interaction_event(stores.session_store, "scan")
                is EventType.INTERACTION_PAUSED
            )
            assert "pending_session_interrupt" not in await stores.session_store.load_checkpoint(
                "scan"
            )
            assert await replacement.reconcile_orphaned_sessions() == ()
        finally:
            await replacement.aclose()
            await stores.close()

    asyncio.run(scenario())


def test_interruption_repair_does_not_take_over_a_live_terminal_publisher(
    tmp_path, database_url, monkeypatch
):
    path = tmp_path / "cayu.db"
    _fresh_orphan(path, "scan", database_url)

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        owner = _application(
            stores.session_store, environment=_slot(), task_store=stores.task_store
        )
        observer = _application(
            stores.session_store, environment=_slot(), task_store=stores.task_store
        )
        await owner.create_task(TaskCreate(task_id="finished", type="scan", session_id="scan"))
        await stores.task_store.cancel_task("finished")
        entered, release = asyncio.Event(), asyncio.Event()
        writer = owner._incomplete_recovery._event_writer
        emit = writer.emit

        async def hold(event, *args, **kwargs):
            if event.type is EventType.SESSION_INTERRUPTED:
                entered.set()
                await release.wait()
            return await emit(event, *args, **kwargs)

        monkeypatch.setattr(writer, "emit", hold)
        task = asyncio.create_task(owner.reconcile_orphaned_sessions())
        try:
            await asyncio.wait_for(entered.wait(), 15)
            before = await stores.session_store.load("scan")
            assert before.status is SessionStatus.INTERRUPTED
            await observer.reconcile_orphaned_sessions()
            after = await stores.session_store.load("scan")
            assert (after.instance_id, after.run_epoch, after.status) == (
                before.instance_id,
                before.run_epoch,
                before.status,
            )
            assert not any(
                e.type is EventType.SESSION_INTERRUPTED
                for e in await stores.session_store.load_events("scan")
            )
            assert not task.done()
            release.set()
            await asyncio.wait_for(task, 15)
            assert [e.type for e in await stores.session_store.load_events("scan")].count(
                EventType.SESSION_INTERRUPTED
            ) == 1
            assert await observer.reconcile_orphaned_sessions() == ()
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 15)
            await owner.aclose()
            await observer.aclose()
            await stores.close()

    asyncio.run(scenario())


def test_owed_completion_repair_preserves_a_live_publisher(tmp_path, database_url, monkeypatch):
    from tests.runtime.runtime_provider_support import VersionedFakeProvider

    from cayu.providers.base import ModelStreamEvent
    from cayu.sessions._paired_writes import owed_decision_of_status

    async def scenario():
        stores = open_application_stores(
            database_url, sqlite_path=tmp_path / "cayu.db", event_watchers=False
        )
        owner = _application(
            stores.session_store,
            environment=_slot(),
            task_store=stores.task_store,
            provider=lambda _events: VersionedFakeProvider(
                [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]
            ),
        )
        observer = _application(
            stores.session_store, environment=_slot(), task_store=stores.task_store
        )
        entered, release = asyncio.Event(), asyncio.Event()
        writer = owner._event_writer
        fan_out = writer.fan_out_persisted

        async def hold_committed_completion(events, *args, **kwargs):
            if any(event.type is EventType.INTERACTION_COMPLETED for event in events):
                entered.set()
                await release.wait()
            return await fan_out(events, *args, **kwargs)

        monkeypatch.setattr(writer, "fan_out_persisted", hold_committed_completion)

        async def run():
            return [event async for event in owner.run(_request("live-publisher"))]

        task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(entered.wait(), 30)
            before = await stores.session_store.load("live-publisher")
            checkpoint = await stores.session_store.load_checkpoint("live-publisher")
            assert before.status is SessionStatus.COMPLETED
            assert owed_decision_of_status(checkpoint, SessionStatus.COMPLETED) is not None
            before_events = await stores.session_store.load_events("live-publisher")
            await observer.reconcile_orphaned_sessions()
            assert await stores.session_store.load("live-publisher") == before
            assert await stores.session_store.load_events("live-publisher") == before_events
            assert not task.done()
            release.set()
            await asyncio.wait_for(task, 30)
            assert await observer.reconcile_orphaned_sessions() == ()
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 30)
            await owner.aclose()
            await observer.aclose()
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("record_order", ["invalid_only", "valid_first", "valid_last"])
def test_default_sweep_validates_pending_interruption_before_claim(
    tmp_path, database_url, record_order
):
    async def scenario():
        stores = open_application_stores(
            database_url, sqlite_path=tmp_path / "cayu.db", event_watchers=False
        )
        try:
            await _assert_pending_interruption_validation(stores.session_store, record_order)
        finally:
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("record_order", ["invalid_only", "valid_first", "valid_last"])
def test_memory_default_sweep_validates_pending_interruption_before_claim(record_order):
    from cayu.sessions import InMemorySessionStore

    asyncio.run(_assert_pending_interruption_validation(InMemorySessionStore(), record_order))


async def _assert_pending_interruption_validation(store, record_order):
    from cayu.sessions import IncompleteSessionRecoveryAction
    from cayu.sessions.records import SessionIdentity

    app = _application(store)
    valid_payload = {
        "interruption_type": "runtime_interrupted",
        "interruption_request_id": "interrupt-before-crash",
    }
    invalid = [
        ("type-missing", {"interruption_request_id": "interrupt-before-crash"}),
        *[
            (f"type-{name}", {**valid_payload, "interruption_type": value})
            for name, value in (
                ("null", None),
                ("bool", True),
                ("int", 1),
                ("list", ["runtime_interrupted"]),
                ("empty", ""),
                ("future", "future_interruption"),
            )
        ],
        ("id-missing", {"interruption_type": "runtime_interrupted"}),
        *[
            (f"id-{name}", {**valid_payload, "interruption_request_id": value})
            for name, value in (("null", None), ("bool", True), ("empty", " "))
        ],
    ]
    provider_payload = {
        **valid_payload,
        "provider_cancellation_failures": [
            {
                "phase": "billing_identity_for_request",
                "error": "Provider billing identity cleanup failed during caller cancellation.",
                "error_type": "BillingIdentityCleanupError",
            }
        ],
    }
    valid = [("valid-runtime", valid_payload), ("valid-provider", provider_payload)]
    records = invalid
    if record_order == "valid_first":
        records = valid + invalid
    elif record_order == "valid_last":
        records = invalid + valid
    before = {}
    try:
        for session_id, payload in records:
            await store.create(
                _request(session_id),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status(session_id, SessionStatus.INTERRUPTED)
            await store.checkpoint(session_id, {"pending_session_interrupt": payload})
            before[session_id] = (
                await store.load(session_id),
                await store.load_checkpoint(session_id),
                await store.load_events(session_id),
            )
        results = {result.session_id: result for result in await app.reconcile_orphaned_sessions()}
        assert set(results) == {session_id for session_id, _ in records}
        mutations = {}
        for session_id, _ in invalid:
            session = await store.load(session_id)
            events = await store.load_events(session_id)
            if (
                session != before[session_id][0]
                or await store.load_checkpoint(session_id) != before[session_id][1]
                or events != before[session_id][2]
            ):
                mutations[session_id] = {
                    "epoch": (before[session_id][0].run_epoch, session.run_epoch),
                    "events": [event.type for event in events],
                }
        assert not mutations, mutations
        for session_id, payload in records:
            after = (
                await store.load(session_id),
                await store.load_checkpoint(session_id),
                await store.load_events(session_id),
            )
            if session_id.startswith("valid-"):
                assert results[session_id].actions == (
                    IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
                )
                assert after[0].status is SessionStatus.INTERRUPTED
                assert "pending_session_interrupt" not in after[1]
                [event] = after[2]
                assert event.type is EventType.SESSION_INTERRUPTED
                assert event.payload == payload
            else:
                assert after == before[session_id], session_id
                assert results[session_id].actions == (IncompleteSessionRecoveryAction.FAILED,)
        repeated = await app.reconcile_orphaned_sessions()
        assert {result.session_id for result in repeated} == {
            session_id for session_id, _ in invalid
        }
        for session_id, _ in invalid:
            assert await store.load(session_id) == before[session_id][0]
            assert await store.load_checkpoint(session_id) == before[session_id][1]
            assert await store.load_events(session_id) == before[session_id][2]
    finally:
        await app.aclose()
