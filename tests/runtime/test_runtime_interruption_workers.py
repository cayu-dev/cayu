"""Interruption workers contracts."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tests.runtime.runtime_provider_support import FakeProvider

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime import _interruption_coordinator as interruption_coordinator_module
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest


def test_background_interruption_shutdown_cancels_shared_workers_and_keeps_marker(monkeypatch):
    async def run():
        monkeypatch.setattr(
            interruption_coordinator_module, "_BACKGROUND_INTERRUPTION_CONCURRENCY", 2
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        parent_id = "sess_parent_shutdown_cascade"
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=identity,
        )
        for index in range(3):
            await store.create(
                RunRequest(
                    agent_name="reviewer",
                    session_id=f"sess_shutdown_child_{index}",
                    parent_session_id=parent_id,
                    messages=[Message.text("user", "child")],
                    metadata={"subagent": {"mode": "background"}},
                ),
                identity=identity,
            )
        started = asyncio.Event()

        async def blocked_interrupt(_request):
            started.set()
            await asyncio.Event().wait()
            yield

        monkeypatch.setattr(
            app._background_interruption_coordinator, "_interrupt_session", blocked_interrupt
        )
        app._session_finalization.schedule_background_interruption_cascade(
            parent_session_id=parent_id,
            interrupt_payload={
                "reason": "operator stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
            create_if_missing=True,
        )
        await asyncio.wait_for(started.wait(), timeout=10)
        drained = await asyncio.wait_for(
            app.drain_background_interruptions(timeout_s=0.001),
            timeout=1,
        )
        return (
            drained,
            app._background_interruption_coordinator._tasks,
            app._background_interruption_coordinator._workers,
            await store.load_checkpoint(parent_id),
        )

    drained, tasks, workers, checkpoint = asyncio.run(run())

    assert drained is False
    assert tasks == set()
    assert workers == set()
    assert "pending_interruption_cascade" in checkpoint


def test_pending_interruption_startup_waits_for_dead_claim_expiry(monkeypatch):
    async def run():
        monkeypatch.setattr(
            interruption_coordinator_module, "_BACKGROUND_INTERRUPTION_LEASE_SECONDS", 0.05
        )
        monkeypatch.setattr(
            interruption_coordinator_module,
            "_BACKGROUND_INTERRUPTION_HEARTBEAT_RETRY_SECONDS",
            0.005,
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            # The durable claim uses the store clock. A frozen worker clock
            # must not prevent deferred takeover after store-owned expiry.
            clock=lambda: datetime(2000, 1, 1, tzinfo=UTC),
        )
        parent_id = "sess_startup_dead_claim"
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(parent_id, SessionStatus.INTERRUPTED)
        await store.checkpoint(
            parent_id,
            {
                "pending_interruption_cascade": {
                    "attempt_id": "cascade-dead-claim-attempt",
                    "interrupt_payload": payload,
                    "created_at": datetime.now(UTC).isoformat(),
                    "generation": 1,
                    "claim_id": "dead-process-claim",
                    "claim_expires_at": (datetime.now(UTC) + timedelta(seconds=0.05)).isoformat(),
                }
            },
        )

        scheduled = await app.resume_pending_interruption_cascades()
        await asyncio.sleep(0.06)
        drained = await app.drain_background_interruptions(timeout_s=1)
        return scheduled, drained, await store.load_checkpoint(parent_id)

    scheduled, drained, checkpoint = asyncio.run(run())

    assert scheduled == 1
    assert drained is True
    assert "pending_interruption_cascade" not in checkpoint


def test_pending_interruption_startup_bounds_healthy_external_claim_recovery():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        root_count = interruption_coordinator_module._BACKGROUND_INTERRUPTION_CONCURRENCY * 3
        for index in range(root_count):
            session_id = f"sess_external_claim_{index}"
            await store.create(
                RunRequest(
                    agent_name="parent",
                    session_id=session_id,
                    messages=[Message.text("user", "parent")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status(session_id, SessionStatus.INTERRUPTED)
            await store.checkpoint(
                session_id,
                {
                    "pending_interruption_cascade": {
                        "attempt_id": f"cascade-external-claim-{index}",
                        "interrupt_payload": payload,
                        "created_at": datetime.now(UTC).isoformat(),
                        "generation": 1,
                        "claim_id": f"external-owner-{index}",
                        "claim_expires_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
                    }
                },
            )

        scheduled = await asyncio.wait_for(
            app.resume_pending_interruption_cascades(),
            timeout=2,
        )
        active_tasks = len(app._background_interruption_coordinator._tasks)
        # Each candidate now performs a bounded root-version guard before
        # projecting the marker. Keep the drain bounded without making
        # scheduler timing, rather than concurrency, the assertion.
        drained = await app.drain_background_interruptions(timeout_s=1)
        return root_count, scheduled, active_tasks, drained

    root_count, scheduled, active_tasks, drained = asyncio.run(run())

    assert scheduled == root_count
    assert active_tasks <= interruption_coordinator_module._BACKGROUND_INTERRUPTION_CONCURRENCY
    assert drained is True


def test_background_interruption_shutdown_drains_locally_queued_roots(monkeypatch):
    async def run():
        monkeypatch.setattr(
            interruption_coordinator_module, "_BACKGROUND_INTERRUPTION_CONCURRENCY", 1
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        old_created_at = datetime.now(UTC) - timedelta(minutes=5)
        for index in range(2):
            session_id = f"sess_locally_queued_{index}"
            await store.create(
                RunRequest(
                    agent_name="parent",
                    session_id=session_id,
                    messages=[Message.text("user", "parent")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status(session_id, SessionStatus.INTERRUPTED)
            await store.checkpoint(
                session_id,
                {
                    "pending_interruption_cascade": {
                        "attempt_id": f"locally-queued-{index}",
                        "interrupt_payload": payload,
                        "created_at": old_created_at.isoformat(),
                    }
                },
            )

        first_started = asyncio.Event()
        release_first = asyncio.Event()
        processed: list[str] = []

        async def controlled_cascade(
            *, parent_session_id, interrupt_payload, create_if_missing=False
        ):
            assert interrupt_payload == payload
            assert create_if_missing is False
            processed.append(parent_session_id)
            if parent_session_id == "sess_locally_queued_0":
                first_started.set()
                await release_first.wait()

        monkeypatch.setattr(
            app._background_interruption_coordinator, "run_cascade", controlled_cascade
        )
        for index in range(2):
            app._session_finalization.schedule_background_interruption_cascade(
                parent_session_id=f"sess_locally_queued_{index}",
                interrupt_payload=payload,
                create_if_missing=False,
            )
        await first_started.wait()
        queued_status = await app.interruption_cascade_status("sess_locally_queued_1")
        drain_task = asyncio.create_task(app.drain_background_interruptions(timeout_s=1))
        await asyncio.sleep(0)
        drain_finished_before_release = drain_task.done()
        release_first.set()
        drained = await drain_task
        return queued_status, drain_finished_before_release, drained, processed

    queued_status, drain_finished_before_release, drained, processed = asyncio.run(run())

    assert queued_status == "pending"
    assert drain_finished_before_release is False
    assert drained is True
    assert processed == ["sess_locally_queued_0", "sess_locally_queued_1"]


def test_background_interruption_shutdown_does_not_restart_external_lease_waiter(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        parent_id = "sess_shutdown_external_lease_race"
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(parent_id, SessionStatus.INTERRUPTED)
        await store.checkpoint(
            parent_id,
            {
                "pending_interruption_cascade": {
                    "attempt_id": "shutdown-external-lease-race",
                    "interrupt_payload": payload,
                    "created_at": datetime.now(UTC).isoformat(),
                    "generation": 1,
                    "claim_id": "external-owner",
                    "claim_expires_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
                }
            },
        )
        claim_started = asyncio.Event()
        release_claim = asyncio.Event()

        async def blocked_claim(*_args, **_kwargs):
            claim_started.set()
            await release_claim.wait()
            return None

        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "claim_pending_interruption_cascade",
            blocked_claim,
        )
        app._session_finalization.schedule_background_interruption_cascade(
            parent_session_id=parent_id,
            interrupt_payload=payload,
            create_if_missing=False,
        )
        await claim_started.wait()
        drain_task = asyncio.create_task(app.drain_background_interruptions(timeout_s=1))
        await asyncio.sleep(0)
        release_claim.set()
        drained = await drain_task
        await asyncio.sleep(0)
        return (
            drained,
            app._background_interruption_coordinator._deferred,
            app._background_interruption_coordinator._deferred_task,
        )

    drained, deferred, deferred_task = asyncio.run(run())

    assert drained is True
    assert deferred == {}
    assert deferred_task is None


def test_background_interruption_drain_cancellation_cleans_up_owned_work(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        parent_id = "sess_cancelled_drain_parent"
        child_id = "sess_cancelled_drain_child"
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id=child_id,
                parent_session_id=parent_id,
                messages=[Message.text("user", "child")],
                metadata={"subagent": {"mode": "background"}},
            ),
            identity=identity,
        )
        child_started = asyncio.Event()

        async def blocked_interrupt(_request):
            child_started.set()
            await asyncio.Event().wait()
            yield

        monkeypatch.setattr(
            app._background_interruption_coordinator, "_interrupt_session", blocked_interrupt
        )
        app._session_finalization.schedule_background_interruption_cascade(
            parent_session_id=parent_id,
            interrupt_payload={
                "reason": "operator stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
            create_if_missing=True,
        )
        await child_started.wait()
        drain_task = asyncio.create_task(app.drain_background_interruptions(timeout_s=10))
        await asyncio.sleep(0)
        drain_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain_task
        checkpoint = await store.load_checkpoint(parent_id)
        return (
            app._background_interruption_coordinator._draining,
            app._background_interruption_coordinator._tasks,
            app._background_interruption_coordinator._workers,
            app._background_interruption_coordinator._deferred_task,
            checkpoint,
        )

    draining, tasks, workers, deferred_task, checkpoint = asyncio.run(run())

    assert draining is False
    assert tasks == set()
    assert workers == set()
    assert deferred_task is None
    marker = checkpoint["pending_interruption_cascade"]
    assert marker["attempt_id"]


def test_background_interruption_shutdown_grace_detaches_cancellation_resistant_worker(
    monkeypatch,
):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        parent_id = "sess_stubborn_shutdown_parent"
        child_id = "sess_stubborn_shutdown_child"
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id=child_id,
                parent_session_id=parent_id,
                messages=[Message.text("user", "child")],
                metadata={"subagent": {"mode": "background"}},
            ),
            identity=identity,
        )
        child_started = asyncio.Event()
        release_child = asyncio.Event()
        child_finished = asyncio.Event()

        async def cancellation_resistant_interrupt(_request):
            child_started.set()
            try:
                while not release_child.is_set():
                    try:
                        await release_child.wait()
                    except asyncio.CancelledError:
                        continue
                yield Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=child_id,
                    agent_name="reviewer",
                )
            finally:
                child_finished.set()

        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "_interrupt_session",
            cancellation_resistant_interrupt,
        )
        app._session_finalization.schedule_background_interruption_cascade(
            parent_session_id=parent_id,
            interrupt_payload={
                "reason": "operator stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
            create_if_missing=True,
        )
        await child_started.wait()
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        drained = await asyncio.wait_for(
            app.drain_background_interruptions(timeout_s=0.01),
            timeout=0.2,
        )
        elapsed = loop.time() - started_at
        tracked_after_drain = (
            app._background_interruption_coordinator._tasks,
            app._background_interruption_coordinator._workers,
            app._background_interruption_coordinator._states,
        )
        release_child.set()
        await asyncio.wait_for(child_finished.wait(), timeout=1)
        await asyncio.sleep(0)
        return drained, elapsed, tracked_after_drain

    drained, elapsed, tracked_after_drain = asyncio.run(run())

    assert drained is False
    assert elapsed < 0.2
    assert tracked_after_drain == (set(), set(), {})


def test_background_interruption_shutdown_fences_cancellation_resistant_claim(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        parent_id = "sess_stubborn_shutdown_claim"
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(parent_id, SessionStatus.INTERRUPTED)

        claim_started = asyncio.Event()
        release_claim = asyncio.Event()
        claim_returned = asyncio.Event()
        claimed_runs: list[str] = []
        original_claim = app._background_interruption_coordinator.claim_pending_interruption_cascade
        original_run_claimed = (
            app._background_interruption_coordinator._run_claimed_background_interruption_cascade
        )

        async def cancellation_resistant_claim(*args, **kwargs):
            claim_started.set()
            while not release_claim.is_set():
                try:
                    await release_claim.wait()
                except asyncio.CancelledError:
                    continue
            marker = await original_claim(*args, **kwargs)
            claim_returned.set()
            return marker

        async def record_claimed_run(state):
            claimed_runs.append(state.parent_session_id)
            await original_run_claimed(state)

        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "claim_pending_interruption_cascade",
            cancellation_resistant_claim,
        )
        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "_run_claimed_background_interruption_cascade",
            record_claimed_run,
        )
        task = app._session_finalization.schedule_background_interruption_cascade(
            parent_session_id=parent_id,
            interrupt_payload=payload,
            create_if_missing=True,
        )
        assert task is not None
        await claim_started.wait()

        drained = await asyncio.wait_for(
            app.drain_background_interruptions(timeout_s=0.01),
            timeout=0.2,
        )
        tracked_after_drain = (
            app._background_interruption_coordinator._tasks,
            app._background_interruption_coordinator._workers,
            app._background_interruption_coordinator._states,
        )
        release_claim.set()
        await asyncio.wait_for(claim_returned.wait(), timeout=1)
        await asyncio.wait_for(task, timeout=1)
        checkpoint = await store.load_checkpoint(parent_id)
        return drained, tracked_after_drain, claimed_runs, checkpoint

    drained, tracked_after_drain, claimed_runs, checkpoint = asyncio.run(run())

    assert drained is False
    assert tracked_after_drain == (set(), set(), {})
    assert claimed_runs == []
    marker = checkpoint["pending_interruption_cascade"]
    assert marker["claim_id"]
    assert marker["claim_expires_at"]


def test_interruption_cascade_status_prefers_cleared_durable_marker_over_stale_defer():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        parent_id = "sess_stale_external_defer"
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(parent_id, SessionStatus.INTERRUPTED)
        await store.checkpoint(
            parent_id,
            {
                "pending_interruption_cascade": {
                    "attempt_id": "stale-external-defer",
                    "interrupt_payload": payload,
                    "created_at": datetime.now(UTC).isoformat(),
                    "generation": 1,
                    "claim_id": "external-owner",
                    "claim_expires_at": (datetime.now(UTC) + timedelta(seconds=30)).isoformat(),
                }
            },
        )
        app._background_interruption_coordinator.defer(
            parent_session_id=parent_id,
            interrupt_payload=payload,
            retry_after_seconds=30,
            drain_required=False,
            retry_request=None,
        )
        await store.checkpoint(parent_id, {})
        status = await app.interruption_cascade_status(parent_id)
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return status

    assert asyncio.run(run()) == "none"


def test_background_interruption_heartbeat_retries_transient_store_error(monkeypatch):
    async def run():
        monkeypatch.setattr(
            interruption_coordinator_module, "_BACKGROUND_INTERRUPTION_LEASE_SECONDS", 0.1
        )
        monkeypatch.setattr(
            interruption_coordinator_module, "_BACKGROUND_INTERRUPTION_HEARTBEAT_SECONDS", 0.01
        )
        monkeypatch.setattr(
            interruption_coordinator_module,
            "_BACKGROUND_INTERRUPTION_HEARTBEAT_RETRY_SECONDS",
            0.005,
        )
        store = InMemorySessionStore()
        fixed_now = datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: fixed_now,
        )
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        parent_id = "sess_transient_heartbeat_parent"
        child_id = "sess_transient_heartbeat_child"
        for session_id, parent_session_id, metadata in (
            (parent_id, None, {}),
            (child_id, parent_id, {"subagent": {"mode": "background"}}),
        ):
            await store.create(
                RunRequest(
                    agent_name="reviewer",
                    session_id=session_id,
                    parent_session_id=parent_session_id,
                    messages=[Message.text("user", "task")],
                    metadata=metadata,
                ),
                identity=identity,
            )

        renewal_succeeded = asyncio.Event()

        async def interrupt_after_renewal(request):
            await asyncio.wait_for(renewal_succeeded.wait(), timeout=1)
            await store.update_status(request.session_id, SessionStatus.INTERRUPTED)
            yield Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=request.session_id,
                agent_name="reviewer",
            )

        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "_interrupt_session",
            interrupt_after_renewal,
        )
        original_renew = (
            app._background_interruption_coordinator.renew_pending_interruption_cascade_claim
        )
        renew_attempts = 0

        async def transient_renew(*args):
            nonlocal renew_attempts
            renew_attempts += 1
            if renew_attempts == 1:
                raise RuntimeError("temporary store outage")
            renewed = await original_renew(*args)
            if renewed:
                renewal_succeeded.set()
            return renewed

        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "renew_pending_interruption_cascade_claim",
            transient_renew,
        )
        await app._background_interruption_coordinator.run_cascade(
            parent_session_id=parent_id,
            interrupt_payload={
                "reason": "operator stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
        )
        drained = await app.drain_background_interruptions(timeout_s=1)
        return drained, renew_attempts, await store.load_checkpoint(parent_id)

    drained, renew_attempts, checkpoint = asyncio.run(run())

    assert drained is True
    assert renew_attempts >= 2
    assert "pending_interruption_cascade" not in checkpoint


def test_background_interruption_does_not_start_after_claim_acknowledgement_expiry(
    monkeypatch,
):
    async def run():
        monkeypatch.setattr(
            interruption_coordinator_module,
            "_BACKGROUND_INTERRUPTION_LEASE_SECONDS",
            0.05,
        )
        monkeypatch.setattr(
            interruption_coordinator_module,
            "_BACKGROUND_INTERRUPTION_HEARTBEAT_SECONDS",
            0.01,
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        parent_id = "sess_delayed_interruption_claim_ack"
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        coordinator = app._background_interruption_coordinator
        original_claim = coordinator.claim_pending_interruption_cascade
        claimed_run_started = False
        deferred: list[dict[str, Any]] = []

        async def delayed_claim(*args, **kwargs):
            marker = await original_claim(*args, **kwargs)
            await asyncio.sleep(0.06)
            return marker

        async def record_claimed_run(_state):
            nonlocal claimed_run_started
            claimed_run_started = True

        monkeypatch.setattr(coordinator, "claim_pending_interruption_cascade", delayed_claim)
        monkeypatch.setattr(
            coordinator,
            "_run_claimed_background_interruption_cascade",
            record_claimed_run,
        )
        monkeypatch.setattr(coordinator, "defer", lambda **kwargs: deferred.append(kwargs))

        await coordinator.run_cascade(
            parent_session_id=parent_id,
            interrupt_payload=payload,
        )
        return claimed_run_started, deferred, await store.load_checkpoint(parent_id)

    claimed_run_started, deferred, checkpoint = asyncio.run(run())

    assert claimed_run_started is False
    assert len(deferred) == 1
    assert "pending_interruption_cascade" in checkpoint


def test_background_interruption_heartbeat_loses_claim_with_frozen_worker_clock(
    monkeypatch,
):
    async def run():
        monkeypatch.setattr(
            interruption_coordinator_module, "_BACKGROUND_INTERRUPTION_LEASE_SECONDS", 0.05
        )
        monkeypatch.setattr(
            interruption_coordinator_module, "_BACKGROUND_INTERRUPTION_HEARTBEAT_SECONDS", 0.01
        )
        monkeypatch.setattr(
            interruption_coordinator_module,
            "_BACKGROUND_INTERRUPTION_HEARTBEAT_RETRY_SECONDS",
            0.005,
        )
        store = InMemorySessionStore()
        fixed_now = datetime(2000, 1, 1, tzinfo=UTC)
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: fixed_now,
        )
        parent_id = "sess_frozen_worker_heartbeat_parent"
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

        renew_attempts = 0

        async def unavailable_renew(*_args):
            nonlocal renew_attempts
            renew_attempts += 1
            raise RuntimeError("store unavailable")

        claim_lost = asyncio.Event()

        async def wait_for_claim_loss(state):
            await state.claim_lost.wait()
            claim_lost.set()

        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "renew_pending_interruption_cascade_claim",
            unavailable_renew,
        )
        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "_run_claimed_background_interruption_cascade",
            wait_for_claim_loss,
        )
        await asyncio.wait_for(
            app._background_interruption_coordinator.run_cascade(
                parent_session_id=parent_id,
                interrupt_payload=payload,
            ),
            timeout=1,
        )
        return claim_lost.is_set(), renew_attempts, await store.load_checkpoint(parent_id)

    claim_lost, renew_attempts, checkpoint = asyncio.run(run())

    assert claim_lost is True
    assert renew_attempts >= 1
    assert "pending_interruption_cascade" in checkpoint


def test_background_interruption_claim_loss_cancels_dispatched_child_interruption(
    monkeypatch,
):
    async def run():
        monkeypatch.setattr(
            interruption_coordinator_module, "_BACKGROUND_INTERRUPTION_LEASE_SECONDS", 0.05
        )
        monkeypatch.setattr(
            interruption_coordinator_module,
            "_BACKGROUND_INTERRUPTION_HEARTBEAT_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            interruption_coordinator_module,
            "_BACKGROUND_INTERRUPTION_HEARTBEAT_RETRY_SECONDS",
            0.005,
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        parent_id = "sess_interruption_claim_loss_parent"
        child_id = "sess_interruption_claim_loss_child"
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id=child_id,
                parent_session_id=parent_id,
                messages=[Message.text("user", "background task")],
                metadata={"subagent": {"mode": "background"}},
            ),
            identity=identity,
        )

        interruption_started = asyncio.Event()
        interruption_cancelled = asyncio.Event()
        late_release = asyncio.Event()
        late_effects: list[str] = []

        async def blocked_interruption(request):
            interruption_started.set()
            try:
                await late_release.wait()
                late_effects.append(request.session_id)
                yield Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=request.session_id,
                    agent_name="reviewer",
                )
            except asyncio.CancelledError:
                interruption_cancelled.set()
                raise

        async def unavailable_renew(*_args):
            raise RuntimeError("store unavailable")

        coordinator = app._background_interruption_coordinator
        monkeypatch.setattr(coordinator, "_interrupt_session", blocked_interruption)
        monkeypatch.setattr(
            coordinator,
            "renew_pending_interruption_cascade_claim",
            unavailable_renew,
        )
        cascade = asyncio.create_task(
            coordinator.run_cascade(
                parent_session_id=parent_id,
                interrupt_payload={
                    "reason": "operator stop",
                    "metadata": {},
                    "requested_by": None,
                    "interruption_type": "operator_requested",
                },
            )
        )
        await asyncio.wait_for(interruption_started.wait(), timeout=10)
        await asyncio.wait_for(interruption_cancelled.wait(), timeout=1)
        await asyncio.wait_for(cascade, timeout=1)
        late_release.set()
        await asyncio.sleep(0)
        return late_effects, await store.load_checkpoint(parent_id)

    late_effects, checkpoint = asyncio.run(run())

    assert late_effects == []
    assert "pending_interruption_cascade" in checkpoint


def test_background_interruption_concurrency_is_app_wide_across_nested_trees(monkeypatch):
    async def run():
        monkeypatch.setattr(
            interruption_coordinator_module, "_BACKGROUND_INTERRUPTION_CONCURRENCY", 2
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        parent_ids = ("sess_nested_parent_a", "sess_nested_parent_b")
        expected_child_ids = set()
        for parent_id in parent_ids:
            await store.create(
                RunRequest(
                    agent_name="parent",
                    session_id=parent_id,
                    messages=[Message.text("user", "parent task")],
                ),
                identity=identity,
            )
            for child_index in range(3):
                child_id = f"{parent_id}_child_{child_index}"
                expected_child_ids.add(child_id)
                await store.create(
                    RunRequest(
                        agent_name="reviewer",
                        session_id=child_id,
                        parent_session_id=parent_id,
                        messages=[Message.text("user", "background task")],
                        metadata={"subagent": {"mode": "background"}},
                    ),
                    identity=identity,
                )
                grandchild_id = f"{child_id}_grandchild"
                expected_child_ids.add(grandchild_id)
                await store.create(
                    RunRequest(
                        agent_name="reviewer",
                        session_id=grandchild_id,
                        parent_session_id=child_id,
                        messages=[Message.text("user", "nested background task")],
                        metadata={"subagent": {"mode": "background"}},
                    ),
                    identity=identity,
                )

        captured_child_ids = set()
        active_requests = 0
        peak_active_requests = 0

        async def capture_interrupt_session(request):
            nonlocal active_requests, peak_active_requests
            active_requests += 1
            peak_active_requests = max(peak_active_requests, active_requests)
            try:
                captured_child_ids.add(request.session_id)
                await asyncio.sleep(0.01)
                yield Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=request.session_id,
                    agent_name="reviewer",
                )
            finally:
                active_requests -= 1

        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "_interrupt_session",
            capture_interrupt_session,
        )
        payload = {
            "reason": "operator stop",
            "metadata": {},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await asyncio.gather(
            *(
                app._background_interruption_coordinator.run_cascade(
                    parent_session_id=parent_id,
                    interrupt_payload=payload,
                )
                for parent_id in parent_ids
            )
        )
        worker_count = len(app._background_interruption_coordinator._workers)
        await app.drain_background_interruptions(timeout_s=1)
        return expected_child_ids, captured_child_ids, peak_active_requests, worker_count

    expected_child_ids, captured_child_ids, peak_active_requests, worker_count = asyncio.run(run())

    assert captured_child_ids == expected_child_ids
    assert peak_active_requests == 2
    assert worker_count == 2


def test_background_interruption_traverses_foreground_nodes_to_background_descendants():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="parent", model="fake-model"))
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        parent_id = "sess_mixed_tree_parent"
        foreground_id = "sess_mixed_tree_foreground"
        background_id = "sess_mixed_tree_background"
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_id,
                messages=[Message.text("user", "parent")],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id=foreground_id,
                parent_session_id=parent_id,
                messages=[Message.text("user", "foreground")],
                metadata={"subagent": {"mode": "foreground"}},
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id=background_id,
                parent_session_id=foreground_id,
                messages=[Message.text("user", "background")],
                metadata={"subagent": {"mode": "background"}},
            ),
            identity=identity,
        )

        await app._background_interruption_coordinator.run_cascade(
            parent_session_id=parent_id,
            interrupt_payload={
                "reason": "operator stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
        )
        return await store.load(foreground_id), await store.load(background_id)

    foreground, background = asyncio.run(run())

    assert foreground.status == SessionStatus.PENDING
    assert background.status == SessionStatus.INTERRUPTED


def test_background_interruption_cascade_paginates_all_children(monkeypatch):
    async def run():
        monkeypatch.setattr(
            interruption_coordinator_module, "_BACKGROUND_INTERRUPTION_CONCURRENCY", 4
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        parent_session_id = "sess_parent_many_background_children"
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id=parent_session_id,
                messages=[Message.text("user", "parent task")],
            ),
            identity=identity,
        )
        expected_child_ids = {
            f"sess_paginated_background_child_{index:04d}" for index in range(1001)
        }
        for child_id in expected_child_ids:
            await store.create(
                RunRequest(
                    agent_name="reviewer",
                    session_id=child_id,
                    parent_session_id=parent_session_id,
                    messages=[Message.text("user", "background task")],
                    metadata={"subagent": {"mode": "background"}},
                ),
                identity=identity,
            )

        captured_child_ids = set()
        active_requests = 0
        peak_active_requests = 0

        async def capture_interrupt_session(request):
            nonlocal active_requests, peak_active_requests
            active_requests += 1
            peak_active_requests = max(peak_active_requests, active_requests)
            try:
                captured_child_ids.add(request.session_id)
                await asyncio.sleep(0)
                yield Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=request.session_id,
                    agent_name="reviewer",
                )
            finally:
                active_requests -= 1

        monkeypatch.setattr(
            app._background_interruption_coordinator,
            "_interrupt_session",
            capture_interrupt_session,
        )
        await app._background_interruption_coordinator.run_cascade(
            parent_session_id=parent_session_id,
            interrupt_payload={
                "reason": "operator stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
        )
        return expected_child_ids, captured_child_ids, peak_active_requests

    expected_child_ids, captured_child_ids, peak_active_requests = asyncio.run(run())

    assert captured_child_ids == expected_child_ids
    assert peak_active_requests == 4
