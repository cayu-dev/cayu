"""Cancelled tasks retain an elected failure after terminal publication and process loss."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.recovery.test_model_terminal_binding_recovery import _application, _database
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import EventType, Message, RunRequest, SessionStatus
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import IncompleteSessionRecoveryRequest
from cayu.sessions._checkpoint_mutations import SessionRuntimePublicationConflict
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
)
from cayu.sessions._invocation_terminal_decision import (
    invocation_terminal_decision_from_checkpoint,
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.tasks import TaskCreate, TaskStatus, run_task_worker, run_task_worker_until_idle
from cayu.tasks.cancellation import _task_cancellation_requested
from cayu.tasks.records import TaskClaimLost


def _crash_failure_with_cancellation(root: Path, cancellation: str) -> None:
    async def run():
        with pytest.MonkeyPatch.context() as patch:
            app, _store, provider = _application(root, patch)

            async def fail(_request):
                assert (root / "allocated").exists()
                yield ModelStreamEvent.error("original failure wins the session")
                yield ModelStreamEvent.completed({"finish_reason": "error"})

            patch.setattr(provider, "stream", fail)
            await app.create_task(TaskCreate(task_id="linked", type="work"))
            elect = app._session_outcomes.claim_invocation_terminal_decision
            fail_task = app._session_finalization.fail_task

            async def cancel_after_election(**kwargs):
                await elect(**kwargs)
                task = await app.task_store.load_task("linked")
                requested = await app.task_store.request_claimed_task_cancellation(
                    task.id, task.worker_id, claim_generation=task.claim_generation
                )
                assert _task_cancellation_requested(requested)
                if cancellation == "requested":
                    os._exit(137)

            async def die_after_cancelled_task(**kwargs):
                task = await fail_task(**kwargs)
                assert task.status is TaskStatus.CANCELLED
                os._exit(137)

            patch.setattr(
                app._session_outcomes, "claim_invocation_terminal_decision", cancel_after_election
            )
            patch.setattr(app._session_finalization, "fail_task", die_after_cancelled_task)

            async def handler(application, task, worker_id):
                async for _ in application.run(
                    RunRequest(
                        agent_name="coding",
                        session_id="cancelled-failure",
                        task_id=task.id,
                        task_worker_id=worker_id,
                        task_lease_expires_at=task.lease_expires_at,
                        messages=[Message.text("user", "work")],
                    )
                ):
                    pass

            await run_task_worker(
                app,
                app.task_store,
                handler,
                worker_id="original",
                lease_seconds=1,
                max_tasks=1,
                reclaim=False,
            )
            raise AssertionError("Worker did not reach the elected cancellation/failure boundary")

    asyncio.run(run())


def _crash_after_failure_publication(root: Path, _cancellation: str) -> None:
    async def run():
        with pytest.MonkeyPatch.context() as patch:
            # The release which replaces the dead worker can omit its environment.
            # Its elected failure may still settle, but saved allocation cleanup
            # remains owed to a later registered recovery owner.
            app, _store, _provider = _application(root, patch, register_environment=False)
            fanout = app._event_writer.fan_out_persisted

            async def die_after_terminal(events, *args, **kwargs):
                result = await fanout(events, *args, **kwargs)
                if any(event.type is EventType.SESSION_FAILED for event in events):
                    os._exit(137)
                return result

            patch.setattr(app._event_writer, "fan_out_persisted", die_after_terminal)
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="cancelled-failure", inactive_for_seconds=0
                )
            )
            raise AssertionError("Replacement did not publish the original failure")

    asyncio.run(run())


def _crash_process(root: Path, entry: str, cancellation: str) -> None:
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; from {__name__} import {entry}; import sys; {entry}(Path(sys.argv[1]), sys.argv[2])",
            str(root),
            cancellation,
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(item for item in sys.path if item)},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert child.returncode == 137, (child.stdout, child.stderr)


@pytest.mark.parametrize("cancellation", ["requested", "cancelled"])
@pytest.mark.parametrize("entrance", ["public", "sweep"])
def test_cancelled_task_recovers_after_failure_publication_process_loss(
    tmp_path, database_url, cancellation, entrance
):
    _database(tmp_path, database_url)
    _crash_process(tmp_path, "_crash_failure_with_cancellation", cancellation)
    _crash_process(tmp_path, "_crash_after_failure_publication", cancellation)

    async def recover():
        with pytest.MonkeyPatch.context() as patch:
            app, store, provider = _application(tmp_path, patch)
            try:
                session = await store.load("cancelled-failure")
                checkpoint = await store.load_checkpoint(session.id)
                settled = settled_invocation_terminal_decision_from_checkpoint(checkpoint)
                assert invocation_terminal_decision_from_checkpoint(checkpoint) is None
                assert settled is not None and settled.run_epoch == session.run_epoch
                assert session.status is SessionStatus.FAILED
                profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
                assert not active_invocation_execution_profile_is_released(
                    profile, session_id=session.id, run_epoch=session.run_epoch
                )
                task = await app.task_store.load_task("linked")
                assert task.status is (
                    TaskStatus.RUNNING if cancellation == "requested" else TaskStatus.CANCELLED
                )
                if cancellation == "requested":
                    assert _task_cancellation_requested(task)
                before_events = await store.load_events(session.id)
                terminal = [
                    event for event in before_events if event.type is EventType.SESSION_FAILED
                ]
                assert len(terminal) == 1 and terminal[0].id == settled.terminal_event_id
                assert terminal[0].payload["error"] == "original failure wins the session"
                assert not (tmp_path / "disposed").exists()

                replay_args = dict(
                    session_id=session.id,
                    task_id=task.id,
                    task_worker_id=task.worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                    task_handoff_id=task.interrupted_handoff_id,
                    cancelled_by_request=True,
                )
                # Keep identity constant while independently conflicting every
                # decision-bearing field, including provenance and payloads.
                for field in type(settled).model_fields:
                    value = getattr(settled, field)
                    conflicting = None if value is not None else "unexpected"
                    with pytest.raises(TaskClaimLost, match="exact elected decision"):
                        await app._session_tasks.replay_runtime_task_failure_if_needed(
                            **replay_args,
                            recovery_decision=settled.model_copy(update={field: conflicting}),
                        )
                # Public/ordinary replay cannot promote settled evidence into
                # the internal recovery authority, even with the same identity.
                with pytest.raises(TaskClaimLost, match="committed failure's evidence"):
                    await app._session_tasks.replay_runtime_task_failure_if_needed(**replay_args)

                query_events = store.query_events
                for event_type in (EventType.SESSION_FAILED, EventType.TURN_COMPLETED):
                    event = next(event for event in before_events if event.type is event_type)
                    for missing in (False, True):

                        async def broken_readback(query, *, event_id=event.id, missing=missing):
                            records = await query_events(query)
                            if query.event_id != event_id:
                                return records
                            if missing:
                                return []
                            return [
                                record.model_copy(
                                    update={
                                        "event": record.event.model_copy(
                                            update={"payload": {"conflicting": True}}
                                        )
                                    }
                                )
                                for record in records
                            ]

                        with pytest.MonkeyPatch.context() as evidence_patch:
                            evidence_patch.setattr(store, "query_events", broken_readback)
                            with pytest.raises(SessionRuntimePublicationConflict):
                                await app._session_tasks.replay_runtime_task_failure_if_needed(
                                    **replay_args,
                                    recovery_decision=settled,
                                )
                        assert await store.load_checkpoint(session.id) == checkpoint
                        assert await store.load_events(session.id) == before_events
                        assert await app.task_store.load_task("linked") == task
                        assert not (tmp_path / "disposed").exists()

                if entrance == "public":
                    await app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(
                            session_id=session.id, inactive_for_seconds=0
                        )
                    )
                else:
                    await app.reconcile_orphaned_sessions()
                assert (tmp_path / "disposed").exists()
                current = await store.load(session.id)
                current_profile = active_invocation_execution_profile_from_checkpoint(
                    await store.load_checkpoint(session.id)
                )
                assert active_invocation_execution_profile_is_released(
                    current_profile, session_id=current.id, run_epoch=current.run_epoch
                )
                assert current.status is SessionStatus.FAILED
                assert await app.task_store.load_task("linked") == task
                events = await store.load_events(session.id)
                assert [
                    event for event in events if event.type is EventType.SESSION_FAILED
                ] == terminal
                assert not provider.requests
                await app.reconcile_orphaned_sessions()
                assert await store.load_events(session.id) == events
                if cancellation == "requested":
                    handler_calls = []

                    async def no_handler(*_args):
                        handler_calls.append(True)
                        raise AssertionError("Cancellation recovery must not redispatch task work")

                    await run_task_worker_until_idle(
                        app, app.task_store, no_handler, worker_id="replacement", reclaim=False
                    )
                    assert not handler_calls
                    cancelled = await app.task_store.load_task(task.id)
                    assert cancelled.status is TaskStatus.CANCELLED
                    assert not _task_cancellation_requested(cancelled)
                    assert (await store.load(session.id)).status is SessionStatus.FAILED
                    assert await store.load_events(session.id) == events
                    assert not provider.requests
            finally:
                await app.aclose()
                await store.close()
                await app.task_store.close()

    asyncio.run(recover())
