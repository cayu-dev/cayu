"""Explicit recovery after a supervisor has stopped an abandoned harness."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from hashlib import sha256

from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.external_agents._authority import confirm_quiescence
from cayu.external_agents.environment import join_cleanup
from cayu.runtime.application_lifecycle import _admitted_entrance
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.invocation import session_invocation_from_task
from cayu.sessions.records import Session, SessionStatus
from cayu.tasks import TaskQuery, TaskStatus

_SETTLED_TASK_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED})


@_admitted_entrance
async def recover_external_agent_run(
    app: CayuApp,
    session_id: str,
    *,
    quiesce: Callable[[Session], Awaitable[None]],
    evidence_id: str,
) -> Session:
    """Settle an abandoned unleased external run after positive quiescence.

    ``quiesce`` must stop and join the original service process, guest process
    tree and gateway requests, or validate a supervisor receipt proving that
    they are already stopped. Lease expiry is insufficient. It must raise if
    it cannot establish that fact. The receipt identifier enters the audit log.
    Worker-owned tasks keep their separate exact-lease recovery contract; this
    helper refuses them. Schedule a fresh run after successful recovery.
    """
    if not evidence_id.strip() or len(evidence_id) > 512:
        raise ValueError("External recovery requires a bounded quiescence evidence ID.")
    if app.task_store is None:
        raise ValueError("External recovery requires the original task store.")
    session = await app.session_store.load(session_id)
    if session is None or session.runtime_name != "external_agent":
        raise ValueError("External recovery requires an existing external-agent session.")
    tasks = await app.task_store.list_tasks(TaskQuery(session_id=session_id))
    if any(
        task.worker_id is not None
        or task.restartable
        or task.retry_series is not None
        or task.work_contract is not None
        for task in tasks
    ):
        raise ValueError("Worker-owned external tasks require exact-lease task recovery.")
    if session.status in {
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
        SessionStatus.INTERRUPTED,
    } and all(task.status in _SETTLED_TASK_STATUSES for task in tasks):
        return session

    caller = asyncio.current_task()
    if caller in app._session_control._active_control_tasks.get(session_id, ()) or any(
        run.runtime_task is caller for run in app._session_control.active_runs(session_id)
    ):
        raise ValueError("A running external harness cannot recover itself.")

    async def publish(event: Event) -> None:
        # Reuse the durable timestamp on retry. Newly discovered unknown spend
        # belongs to the current budget window, not the session's creation time.
        records = await app.session_store.query_events(
            EventQuery(session_id=session_id, event_id=event.id, limit=1)
        )
        if records:
            event = event.model_copy(update={"timestamp": records[0].event.timestamp})
        try:
            persisted = await app._event_writer.persist_exact_replay(event)
        except Exception:
            # A concurrent recovery may have won the first timestamp. Replay
            # still verifies every other field rather than adopting its payload.
            records = await app.session_store.query_events(
                EventQuery(session_id=session_id, event_id=event.id, limit=1)
            )
            if not records:
                raise
            persisted = await app._event_writer.persist_exact_replay(
                event.model_copy(update={"timestamp": records[0].event.timestamp})
            )
        [delivered] = await app._event_writer.fan_out_persisted([persisted])
        app._session_control.queue_out_of_band_event(delivered)

    async def recover() -> Session:
        assert app.task_store is not None
        await quiesce(session)
        confirm_quiescence(app, session)
        targets = tuple(app._session_control._interrupt_targets(session_id))
        for target in targets:
            target.cancel("External harness quiesced for recovery.")
        await asyncio.gather(*targets, return_exceptions=True)
        current = await app.session_store.load(session_id)
        if current is None or (current.instance_id, current.run_epoch) != (
            session.instance_id,
            session.run_epoch,
        ):
            raise ValueError("External session changed while its harness was quiescing.")
        # A new supervisor receipt is a separate audit event. It must not alter
        # the stable terminal event when recovery retries after a partial write.
        receipt_digest = sha256(evidence_id.encode()).hexdigest()
        await publish(
            Event(
                id=f"external-quiesced-{session.instance_id}-{session.run_epoch}-{receipt_digest}",
                type="custom.external_agent.recovery_quiesced",
                session_id=session_id,
                agent_name=session.agent_name,
                payload=app._secret_redactor.redact_json(
                    {"external_agent": True, "evidence_id": evidence_id}
                ),
            )
        )
        if current.status in {
            SessionStatus.PENDING,
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
            SessionStatus.INTERRUPTED,
            SessionStatus.FAILED,
        }:
            await publish(
                Event(
                    id=f"external-crash-usage-{session.instance_id}-{session.run_epoch}",
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    agent_name=session.agent_name,
                    payload={
                        "provider_name": session.provider_name,
                        "model": session.model,
                        "usage": {},
                        "external_agent": True,
                        "usage_outcome": "unknown_after_crash",
                    },
                )
            )
        if current.status in {
            SessionStatus.PENDING,
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
            SessionStatus.INTERRUPTED,
        }:
            await publish(
                Event(
                    id=f"external-crash-interrupt-{session.instance_id}-{session.run_epoch}",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    agent_name=session.agent_name,
                    payload={
                        "external_agent": True,
                        "reason": "supervisor_recovered_external_harness",
                    },
                )
            )
            if current.status is not SessionStatus.INTERRUPTED:
                current = await app.session_store.transition_status(
                    session_id, from_statuses={current.status}, to_status=SessionStatus.INTERRUPTED
                )
        for task in tasks:
            latest = await app.task_store.load_task(task.id)
            if latest is None or latest.status in _SETTLED_TASK_STATUSES:
                continue
            pre_attachment = (
                latest.session_instance_id is None
                and latest.status is TaskStatus.PENDING
                and latest.type == "external_agent"
                and session_invocation_from_task(latest.invocation, session_id=session.id)
                == session.invocation
            )
            if (
                latest.session_id != session.id
                or latest.worker_id is not None
                or not (latest.session_instance_id == session.instance_id or pre_attachment)
            ):
                raise ValueError("External task ownership changed during recovery.")
            if current.status is SessionStatus.COMPLETED:
                await app.task_store.complete_task(task.id, {"session_id": session.id})
            else:
                await app.task_store.cancel_task(task.id, {"error": "ExternalHarnessAbandoned"})
        return current

    result = await join_cleanup(recover())
    assert result is not None
    return result
