from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr
from tests.core._execution_profile_fixtures import (
    create_admitted_session,
    versioned_test_provider_identity,
)
from tests.core.task_invocation_fixtures import stored_session_invocation

import cayu.runtime._recovery_plan_coordinator as recovery_plan_coordinator_module
from cayu import SecretRedactor, SQLiteSessionStore, SQLiteTaskStore
from cayu.core import AgentSpec, EventType, ExecutionProfileBehaviorIdentity, Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    InMemoryTaskStore,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    RecoveryBlockerCode,
    RecoveryDecision,
    RecoveryExecutionRequest,
    RecoveryItemExecutionStatus,
    RecoveryPlan,
    RecoveryPlanAction,
    RecoveryPlanBounds,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
    RecoveryRegistrationStatus,
    RecoveryTaskClaimEvidence,
    ResumeRequest,
    RunRequest,
    SessionIdentity,
    SessionStatus,
    TaskCreate,
    TaskInvocationSnapshot,
    TaskQuery,
    TaskStatus,
)
from cayu.runtime.sessions import run_request_with_task_invocation


class _FakeProvider(ModelProvider):
    name = "fake"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return versioned_test_provider_identity(self)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _app(
    store: InMemorySessionStore | SQLiteSessionStore,
    *,
    task_store: InMemoryTaskStore | SQLiteTaskStore | None = None,
    clock: Callable[[], datetime] | None = None,
) -> CayuApp:
    app = CayuApp(
        session_store=store,
        task_store=task_store,
        enable_logging=False,
        clock=clock,
    )
    app.register_provider(_FakeProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return app


async def _create_running_session(
    store: InMemorySessionStore | SQLiteSessionStore,
    app: CayuApp,
    session_id: str,
) -> None:
    async def admit() -> None:
        await create_admitted_session(
            store,
            request=RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "start")],
            ),
            provider_name="fake",
            model="fake-model",
            app=app,
        )

    # The low-level fixture binds invocation authority in its task context;
    # model a restarted operator process by keeping that context out of the
    # planner/executor task.
    await asyncio.create_task(admit())


async def _create_running_task_session(
    store: InMemorySessionStore | SQLiteSessionStore,
    task_store: InMemoryTaskStore | SQLiteTaskStore,
    app: CayuApp,
    *,
    session_id: str,
    task_id: str,
) -> None:
    task = await task_store.create_task(TaskCreate(task_id=task_id, type="recovery-plan"))

    async def admit() -> None:
        await create_admitted_session(
            store,
            request=run_request_with_task_invocation(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    task_id=task_id,
                    messages=[Message.text("user", "start")],
                ),
                TaskInvocationSnapshot(
                    id=task.id,
                    session_id=task.session_id,
                    invocation=task.invocation,
                ),
            ),
            provider_name="fake",
            model="fake-model",
            app=app,
        )

    await asyncio.create_task(admit())
    claimed = await task_store.claim_task(
        "expired-worker",
        TaskQuery(type="recovery-plan"),
        lease_seconds=30,
    )
    assert claimed is not None and claimed.id == task_id
    await task_store.attach_task(
        task_id,
        session_id=session_id,
        session_invocation=await stored_session_invocation(store, session_id),
        worker_id="expired-worker",
        lease_expires_at=claimed.lease_expires_at,
    )


async def assert_recovery_plan_store_conformance(store) -> None:
    """Prove read-only planning, exact execution, successor use, and replay."""

    app = _app(store)
    await _create_running_session(store, app, "sess_recovery_plan")
    before = (
        await store.load("sess_recovery_plan"),
        await store.load_checkpoint("sess_recovery_plan"),
        await store.load_events("sess_recovery_plan"),
    )

    plan = await app.plan_recovery(
        RecoveryPlanRequest(
            selection=RecoveryPlanSelection(
                session_ids=("sess_recovery_plan",),
                inactive_for_seconds=0,
            )
        )
    )

    assert before == (
        await store.load("sess_recovery_plan"),
        await store.load_checkpoint("sess_recovery_plan"),
        await store.load_events("sess_recovery_plan"),
    )
    assert plan.items[0].allowed_actions == (
        RecoveryPlanAction.LEAVE_INTACT,
        RecoveryPlanAction.AUTOMATIC_REPAIR,
    )

    plan = RecoveryPlan.model_validate_json(plan.model_dump_json())
    request = RecoveryExecutionRequest(plan=plan, execution_id="execution-one")
    first = await app.execute_recovery(request)
    assert first.items[0].status is RecoveryItemExecutionStatus.EXECUTED
    assert first.items[0].replayed is False
    assert first.items[0].final_session_status is SessionStatus.INTERRUPTED
    successor_events = [
        event
        async for event in app.resume(
            ResumeRequest(
                session_id="sess_recovery_plan",
                messages=[Message.text("user", "continue")],
            )
        )
    ]
    replay = await app.execute_recovery(request)

    assert successor_events[-1].type is EventType.SESSION_COMPLETED
    assert replay.items[0] == first.items[0].model_copy(update={"replayed": True})
    receipt_events = [
        event
        for event in await store.load_events("sess_recovery_plan")
        if event.type is EventType.RECOVERY_PLAN_ITEM_EXECUTED
    ]
    assert len(receipt_events) == 1


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_recovery_plan_is_read_only_and_execution_receipt_replays(
    backend: str,
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "recovery-plans.sqlite")
        )
        try:
            await assert_recovery_plan_store_conformance(store)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(exercise())


def test_recovery_execution_rejects_state_changed_after_planning() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        app = _app(store)
        await _create_running_session(store, app, "sess_stale_recovery_plan")
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=("sess_stale_recovery_plan",))
            )
        )
        await store.update_status("sess_stale_recovery_plan", SessionStatus.INTERRUPTING)

        receipt = await app.execute_recovery(
            RecoveryExecutionRequest(plan=plan, execution_id="stale-execution")
        )

        assert receipt.items[0].status is RecoveryItemExecutionStatus.BLOCKED
        assert receipt.items[0].error_code == "StaleRecoveryPlanError"
        assert not any(
            event.type is EventType.RECOVERY_PLAN_ITEM_EXECUTED
            for event in await store.load_events("sess_stale_recovery_plan")
        )

    asyncio.run(exercise())


def test_recovery_plan_rejects_registration_drift_before_mutation() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        original_app = _app(store)
        await _create_running_session(store, original_app, "sess_registration_drift")
        original_plan = await original_app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=("sess_registration_drift",))
            )
        )

        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_plan = await replacement_app.plan_recovery(original_plan.request)
        assert replacement_plan.items[0].registration.status is (
            RecoveryRegistrationStatus.MISSING_AGENT
        )
        assert replacement_plan.items[0].allowed_actions == (RecoveryPlanAction.LEAVE_INTACT,)

        before = (
            await store.load("sess_registration_drift"),
            await store.load_checkpoint("sess_registration_drift"),
            await store.load_events("sess_registration_drift"),
        )
        receipt = await replacement_app.execute_recovery(
            RecoveryExecutionRequest(plan=original_plan, execution_id="registration-drift")
        )

        assert receipt.items[0].status is RecoveryItemExecutionStatus.BLOCKED
        assert receipt.items[0].error_code == "StaleRecoveryPlanError"
        assert before == (
            await store.load("sess_registration_drift"),
            await store.load_checkpoint("sess_registration_drift"),
            await store.load_events("sess_registration_drift"),
        )

    asyncio.run(exercise())


def test_recovery_execution_rejects_task_claim_changed_after_planning(monkeypatch) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        app = _app(store)
        await _create_running_session(store, app, "sess_task_claim_changed")
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=("sess_task_claim_changed",))
            )
        )

        async def changed_task_claims(_session_id: str):
            return (
                RecoveryTaskClaimEvidence(
                    task_ref="task:sha256:" + "b" * 64,
                    status=TaskStatus.CLAIMED,
                ),
            )

        monkeypatch.setattr(
            app._recovery_plan_coordinator,
            "_task_claims",
            changed_task_claims,
        )
        receipt = await app.execute_recovery(
            RecoveryExecutionRequest(plan=plan, execution_id="changed-task-execution")
        )

        assert receipt.items[0].status is RecoveryItemExecutionStatus.BLOCKED
        assert receipt.items[0].error_code == "StaleRecoveryPlanError"
        session = await store.load("sess_task_claim_changed")
        assert session is not None
        assert session.status is SessionStatus.RUNNING

    asyncio.run(exercise())


def test_recovery_plan_blocks_live_recovery_and_task_owners_without_mutation(
    monkeypatch,
) -> None:
    async def exercise() -> None:
        now = datetime(2026, 9, 3, tzinfo=UTC)
        store = InMemorySessionStore(ownership_clock=lambda: now)
        app = _app(store, clock=lambda: now)
        await _create_running_session(store, app, "sess_owned_recovery_plan")

        def install_live_claim(
            _session,
            checkpoint: dict[str, object] | None,
        ) -> dict[str, object]:
            return {
                **(checkpoint or {}),
                "incomplete_session_recovery_claim": {
                    "version": 1,
                    "claim_id": "live-recovery-owner",
                    "claimed_at": now.isoformat(),
                    "claim_expires_at": (now + timedelta(minutes=5)).isoformat(),
                },
            }

        await store.transform_checkpoint("sess_owned_recovery_plan", install_live_claim)

        async def active_task_claims(_session_id: str):
            return (
                RecoveryTaskClaimEvidence(
                    task_ref="task:sha256:" + "c" * 64,
                    status=TaskStatus.RUNNING,
                    ownership_status="active",
                    worker_ref="task-worker:sha256:" + "d" * 64,
                    lease_expires_at=now + timedelta(minutes=5),
                ),
            )

        monkeypatch.setattr(
            app._recovery_plan_coordinator,
            "_task_claims",
            active_task_claims,
        )
        before = await store.load_checkpoint("sess_owned_recovery_plan")
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=("sess_owned_recovery_plan",))
            )
        )

        assert {blocker.code for blocker in plan.items[0].blockers} >= {
            RecoveryBlockerCode.ACTIVE_RECOVERY_CLAIM,
            RecoveryBlockerCode.ACTIVE_TASK_CLAIM,
        }
        assert plan.items[0].allowed_actions == (RecoveryPlanAction.LEAVE_INTACT,)
        receipt = await app.execute_recovery(
            RecoveryExecutionRequest(plan=plan, execution_id="leave-owned-intact")
        )
        assert receipt.items[0].status is RecoveryItemExecutionStatus.LEFT_INTACT
        assert await store.load_checkpoint("sess_owned_recovery_plan") == before

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_recovery_plan_hands_off_exact_expired_attached_task_owner(
    backend: str,
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        now = [datetime(2026, 9, 3, tzinfo=UTC)]

        def clock() -> datetime:
            return now[0]

        session_store = (
            InMemorySessionStore(ownership_clock=clock)
            if backend == "memory"
            else SQLiteSessionStore(
                tmp_path / "expired-plan-session.sqlite",
                ownership_clock=clock,
            )
        )
        task_store = (
            InMemoryTaskStore(ownership_clock=clock)
            if backend == "memory"
            else SQLiteTaskStore(
                tmp_path / "expired-plan-task.sqlite",
                ownership_clock=clock,
            )
        )
        app = _app(session_store, task_store=task_store, clock=clock)
        try:
            await _create_running_task_session(
                session_store,
                task_store,
                app,
                session_id="sess_expired_task_plan",
                task_id="expired-task-plan",
            )
            live_plan = await app.plan_recovery(
                RecoveryPlanRequest(
                    selection=RecoveryPlanSelection(session_ids=("sess_expired_task_plan",))
                )
            )
            live_item = live_plan.items[0]
            assert live_item.task_claims[0].ownership_status == "active"
            assert RecoveryBlockerCode.ACTIVE_TASK_CLAIM in {
                blocker.code for blocker in live_item.blockers
            }
            assert live_item.allowed_actions == (RecoveryPlanAction.LEAVE_INTACT,)
            now[0] += timedelta(seconds=31)

            plan = await app.plan_recovery(
                RecoveryPlanRequest(
                    selection=RecoveryPlanSelection(session_ids=("sess_expired_task_plan",))
                )
            )

            item = plan.items[0]
            assert item.task_claims[0].ownership_status == "expired"
            assert RecoveryBlockerCode.ACTIVE_TASK_CLAIM not in {
                blocker.code for blocker in item.blockers
            }
            assert RecoveryPlanAction.AUTOMATIC_REPAIR in item.allowed_actions

            receipt = await app.execute_recovery(
                RecoveryExecutionRequest(
                    plan=plan,
                    execution_id=f"expired-task-{backend}",
                )
            )

            assert receipt.items[0].status is RecoveryItemExecutionStatus.EXECUTED
            session = await session_store.load("sess_expired_task_plan")
            task = await task_store.load_task("expired-task-plan")
            assert session is not None and session.status is SessionStatus.INTERRUPTED
            assert task is not None and task.status is TaskStatus.RUNNING
            assert task.worker_id is None
            assert task.lease_expires_at is None
            assert task.interrupted_handoff_id is not None
            handoff_receipt = await task_store.load_interrupted_task_handoff_receipt(
                task.id,
                task.interrupted_handoff_id,
            )
            assert handoff_receipt is not None
            successor_plan = await app.plan_recovery(
                RecoveryPlanRequest(
                    selection=RecoveryPlanSelection(session_ids=("sess_expired_task_plan",))
                )
            )
            successor_item = successor_plan.items[0]
            assert successor_item.task_claims[0].ownership_status == "unowned"
            assert RecoveryBlockerCode.ACTIVE_TASK_CLAIM not in {
                blocker.code for blocker in successor_item.blockers
            }
        finally:
            if isinstance(session_store, SQLiteSessionStore):
                await session_store.close()
            if isinstance(task_store, SQLiteTaskStore):
                await task_store.close()

    asyncio.run(exercise())


def test_recovery_plan_status_selection_uses_request_bound_cursor() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        app = _app(store)
        for session_id in ("sess_recovery_page_one", "sess_recovery_page_two"):
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
        first = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(statuses={SessionStatus.PENDING}),
                bounds=RecoveryPlanBounds(item_limit=1, inspection_limit=1),
            )
        )
        assert first.next_cursor is not None
        RecoveryPlan.model_validate_json(first.model_dump_json())

        second = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(
                    statuses={SessionStatus.PENDING},
                    cursor=first.next_cursor,
                ),
                bounds=RecoveryPlanBounds(item_limit=1, inspection_limit=1),
            )
        )
        assert {first.items[0].session_id, second.items[0].session_id} == {
            "sess_recovery_page_one",
            "sess_recovery_page_two",
        }

    asyncio.run(exercise())


def test_recovery_plan_serialization_does_not_expose_session_payloads() -> None:
    async def exercise() -> None:
        secret = "recovery-plan-secret-canary"
        store = InMemorySessionStore()
        app = _app(store)
        await _create_running_session(store, app, "sess_safe_recovery_plan")

        def add_private_state(
            _session,
            checkpoint: dict[str, object] | None,
        ) -> dict[str, object]:
            return {**(checkpoint or {}), "private_payload": secret}

        await store.transform_checkpoint("sess_safe_recovery_plan", add_private_state)
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=("sess_safe_recovery_plan",))
            )
        )

        assert secret not in plan.model_dump_json()

    asyncio.run(exercise())


def test_recovery_plan_uses_public_session_alias_for_status_selection() -> None:
    async def exercise() -> None:
        encoded_key = base64.urlsafe_b64encode(bytes([23]) * 32).decode().rstrip("=")
        private_session_id = "private-recovery-session-canary"
        secret = "recovery-session-canary"
        keyring = PublicAuthorityAliasKeyring(
            active_key_id="recovery-test",
            keys={"recovery-test": SecretStr(encoded_key)},
        )
        store = InMemorySessionStore(
            public_authority_alias_codec=PublicAuthorityAliasCodec(keyring)
        )
        app = CayuApp(
            session_store=store,
            public_authority_alias_keyring=keyring,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(_FakeProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=private_session_id,
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(private_session_id, SessionStatus.RUNNING)

        plan = await app.plan_recovery(
            RecoveryPlanRequest(selection=RecoveryPlanSelection(statuses={SessionStatus.RUNNING}))
        )

        assert len(plan.items) == 1
        assert plan.items[0].session_id != private_session_id
        assert private_session_id not in plan.model_dump_json()
        receipt = await app.execute_recovery(
            RecoveryExecutionRequest(plan=plan, execution_id="public-alias-execution")
        )
        assert receipt.items[0].session_id == plan.items[0].session_id
        assert receipt.items[0].status is RecoveryItemExecutionStatus.EXECUTED

    asyncio.run(exercise())


def test_recovery_execution_runs_independent_sessions_concurrently(monkeypatch) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        app = _app(store)
        for session_id in ("sess_parallel_one", "sess_parallel_two"):
            await _create_running_session(store, app, session_id)
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(
                    session_ids=("sess_parallel_one", "sess_parallel_two")
                )
            )
        )

        coordinator = app._recovery_plan_coordinator
        original_recover = coordinator._recover_incomplete_session
        both_started = asyncio.Event()
        active = 0
        peak = 0

        async def synchronized_recover(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)
            try:
                return await original_recover(request)
            finally:
                active -= 1

        monkeypatch.setattr(
            coordinator,
            "_recover_incomplete_session",
            synchronized_recover,
        )
        receipt = await app.execute_recovery(
            RecoveryExecutionRequest(
                plan=plan,
                execution_id="parallel-execution",
                max_concurrency=2,
            )
        )

        assert peak == 2
        assert {item.status for item in receipt.items} == {RecoveryItemExecutionStatus.EXECUTED}

    asyncio.run(exercise())


def test_recovery_execution_renews_its_durable_session_lease(monkeypatch) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        app = _app(store)
        await _create_running_session(store, app, "sess_recovery_heartbeat")
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=("sess_recovery_heartbeat",))
            )
        )
        coordinator = app._recovery_plan_coordinator
        original_recover = coordinator._recover_incomplete_session
        entered = asyncio.Event()
        release = asyncio.Event()

        async def delayed_recover(request):
            entered.set()
            await release.wait()
            return await original_recover(request)

        monkeypatch.setattr(
            recovery_plan_coordinator_module,
            "_RECOVERY_PLAN_EXECUTION_HEARTBEAT_SECONDS",
            0.01,
        )
        monkeypatch.setattr(coordinator, "_recover_incomplete_session", delayed_recover)
        execution = asyncio.create_task(
            app.execute_recovery(
                RecoveryExecutionRequest(plan=plan, execution_id="heartbeat-execution")
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.sleep(0.04)
        checkpoint = await store.load_checkpoint("sess_recovery_heartbeat")
        assert checkpoint is not None
        marker = checkpoint["recovery_plan_execution"]
        ownership = marker["ownership"]
        assert ownership["renewed_at"] > ownership["acquired_at"]
        monkeypatch.setattr(
            recovery_plan_coordinator_module,
            "_RECOVERY_PLAN_EXECUTION_HEARTBEAT_SECONDS",
            10.0,
        )
        await asyncio.sleep(0.02)
        progress = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=("sess_recovery_heartbeat",))
            )
        )
        assert progress.items[0].plan_execution is not None
        assert RecoveryPlanAction.AUTOMATIC_REPAIR not in progress.items[0].allowed_actions
        assert "heartbeat-execution" not in progress.model_dump_json()
        release.set()

        receipt = await asyncio.wait_for(execution, timeout=10)
        assert receipt.items[0].status is RecoveryItemExecutionStatus.EXECUTED

    asyncio.run(exercise())


def test_recovery_execution_settles_one_planned_interruption_cascade() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        app = _app(store)
        await _create_running_session(store, app, "sess_recovery_cascade")
        await store.update_status("sess_recovery_cascade", SessionStatus.INTERRUPTING)
        await store.update_status("sess_recovery_cascade", SessionStatus.INTERRUPTED)

        def add_cascade(_session, checkpoint):
            return {
                **(checkpoint or {}),
                "pending_interruption_cascade": {
                    "attempt_id": "cascade-attempt-one",
                    "interrupt_payload": {
                        "interruption_type": "operator_requested",
                        "reason": "operator recovery test",
                        "metadata": {},
                    },
                    "created_at": datetime.now(UTC).isoformat(),
                },
            }

        await store.transform_checkpoint("sess_recovery_cascade", add_cascade)
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=("sess_recovery_cascade",))
            )
        )
        assert plan.items[0].interruption_cascade is not None
        assert RecoveryPlanAction.AUTOMATIC_REPAIR in plan.items[0].allowed_actions

        receipt = await app.execute_recovery(
            RecoveryExecutionRequest(plan=plan, execution_id="cascade-execution")
        )

        assert receipt.items[0].status is RecoveryItemExecutionStatus.EXECUTED
        checkpoint = await store.load_checkpoint("sess_recovery_cascade")
        assert checkpoint is None or "pending_interruption_cascade" not in checkpoint

    asyncio.run(exercise())


def test_recovery_execution_takes_over_its_expired_claim_after_cancellation(
    monkeypatch,
) -> None:
    async def exercise() -> None:
        now = [datetime(2026, 9, 3, tzinfo=UTC)]

        def clock() -> datetime:
            return now[0]

        store = InMemorySessionStore(ownership_clock=clock)
        app = _app(store, clock=clock)
        await _create_running_session(store, app, "sess_recovery_takeover")
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=("sess_recovery_takeover",))
            )
        )
        request = RecoveryExecutionRequest(plan=plan, execution_id="takeover-execution")

        async def cancel_after_claim(**_kwargs):
            raise asyncio.CancelledError

        monkeypatch.setattr(
            app._recovery_plan_coordinator,
            "_apply_decision",
            cancel_after_claim,
        )
        with pytest.raises(asyncio.CancelledError):
            await app.execute_recovery(request)

        checkpoint = await store.load_checkpoint("sess_recovery_takeover")
        assert checkpoint is not None
        assert checkpoint["recovery_plan_execution"]["execution_id"] == "takeover-execution"

        changed_decision = request.model_copy(
            update={
                "decisions": (
                    RecoveryDecision(
                        item_id=plan.items[0].item_id,
                        action=RecoveryPlanAction.LEAVE_INTACT,
                    ),
                )
            }
        )
        fenced = await app.execute_recovery(changed_decision)
        assert fenced.items[0].status is RecoveryItemExecutionStatus.BLOCKED
        assert fenced.items[0].error_code == "RecoveryPlanExecutionFenced"

        now[0] += timedelta(seconds=901)
        restarted = _app(store, clock=clock)
        resumed = await restarted.execute_recovery(request)

        assert resumed.items[0].status is RecoveryItemExecutionStatus.EXECUTED
        assert resumed.items[0].replayed is False
        checkpoint = await store.load_checkpoint("sess_recovery_takeover")
        assert checkpoint is None or "recovery_plan_execution" not in checkpoint

    asyncio.run(exercise())


def test_recovery_takeover_fails_closed_when_authority_advanced_after_claim(
    monkeypatch,
) -> None:
    async def exercise() -> None:
        now = [datetime(2026, 9, 3, tzinfo=UTC)]

        def clock() -> datetime:
            return now[0]

        store = InMemorySessionStore(ownership_clock=clock)
        app = _app(store, clock=clock)
        session_id = "sess_recovery_takeover_advanced"
        await _create_running_session(store, app, session_id)
        plan = await app.plan_recovery(
            RecoveryPlanRequest(selection=RecoveryPlanSelection(session_ids=(session_id,)))
        )
        request = RecoveryExecutionRequest(plan=plan, execution_id="advanced-execution")

        async def cancel_after_claim(**_kwargs):
            raise asyncio.CancelledError

        monkeypatch.setattr(
            app._recovery_plan_coordinator,
            "_apply_decision",
            cancel_after_claim,
        )
        with pytest.raises(asyncio.CancelledError):
            await app.execute_recovery(request)

        now[0] += timedelta(seconds=901)
        await store.update_status(session_id, SessionStatus.INTERRUPTING)
        restarted = _app(store, clock=clock)
        receipt = await restarted.execute_recovery(request)

        assert receipt.items[0].status is RecoveryItemExecutionStatus.FAILED
        assert receipt.items[0].error_code == "recovery_plan_outcome_unknown"
        current = await store.load(session_id)
        assert current is not None
        assert current.status is SessionStatus.INTERRUPTING
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is None or "recovery_plan_execution" not in checkpoint
        replay = await restarted.execute_recovery(request)
        assert replay.items[0] == receipt.items[0].model_copy(update={"replayed": True})

    asyncio.run(exercise())


def test_recovery_execution_reads_back_receipt_after_publication_ack_loss(
    monkeypatch,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        app = _app(store)
        await _create_running_session(store, app, "sess_recovery_receipt_readback")
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=("sess_recovery_receipt_readback",))
            )
        )
        request = RecoveryExecutionRequest(plan=plan, execution_id="ack-loss-execution")

        coordinator = app._recovery_plan_coordinator
        commit_receipt = coordinator._commit_receipt

        async def lose_ack(**kwargs):
            await commit_receipt(**kwargs)
            raise RuntimeError("simulated event fan-out acknowledgement loss")

        monkeypatch.setattr(coordinator, "_commit_receipt", lose_ack)
        first = await app.execute_recovery(request)
        replay = await app.execute_recovery(request)

        assert first.items[0].status is RecoveryItemExecutionStatus.EXECUTED
        assert first.items[0].replayed is False
        assert replay.items[0] == first.items[0].model_copy(update={"replayed": True})
        receipt_events = [
            event
            for event in await store.load_events("sess_recovery_receipt_readback")
            if event.type is EventType.RECOVERY_PLAN_ITEM_EXECUTED
        ]
        assert len(receipt_events) == 1

    asyncio.run(exercise())
