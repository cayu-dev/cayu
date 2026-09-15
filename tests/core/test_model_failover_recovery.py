"""Public failover recovery with real deadline expiry and process termination."""

from __future__ import annotations

import asyncio
import multiprocessing
from decimal import Decimal

import pytest
from tests.core._execution_profile_fixtures import versioned_test_provider_identity
from tests.core.test_model_failover_stages import _StageMemoryStore, _StageSQLiteStore

from cayu import (
    AgentSpec,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    EventType,
    ExecutionProfileBehaviorIdentity,
    ExecutionProfileMismatchError,
    IncompleteSessionRecoveryRequest,
    Message,
    ModelFailoverPolicy,
    ModelPrice,
    ModelTarget,
    PriceBook,
    RecoveryBlockerCode,
    RecoveryDecision,
    RecoveryExecutionRequest,
    RecoveryItemExecutionStatus,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
    ResumeRequest,
    RunRequest,
    SessionStatusConflict,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolResult,
    ToolSpec,
)
from cayu.providers.base import (
    ModelProvider,
    ModelProviderError,
    ModelStreamDeadlineError,
    ModelStreamEvent,
    _preflight_provider_portable_messages,
)
from cayu.providers.deadlines import ProviderStreamDeadlines
from cayu.runtime.retry_policy import RetryPolicy
from cayu.sessions.base import SessionStore
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy


class _RecoveryProvider(ModelProvider):
    def preflight_portable_messages(self, *, model, messages, tools):
        _preflight_provider_portable_messages(
            model=model,
            messages=messages,
            tools=tools,
            supports_system_messages=True,
            supports_tool_history=True,
            supports_tool_definitions=True,
            supports_file_attachments=True,
        )

    def __init__(self, name, *, entered=None, behavior_version="1"):
        self.name = name
        self.entered = entered
        self.behavior_version = behavior_version
        self.requests = []

    @property
    def execution_profile_identity(self):
        return versioned_test_provider_identity(self, behavior_version=self.behavior_version)

    @property
    def stream_deadlines(self):
        return ProviderStreamDeadlines(semantic_progress_timeout_s=0.1)

    async def stream(self, request):
        self.requests.append(request)
        if self.name == "primary":
            raise ModelProviderError(
                "unavailable", provider=self.name, status_code=503, retryable=True
            )
        if self.entered is not None:
            self.entered.set()
            await asyncio.Event().wait()
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed(
            {"usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1}}
        )


def _application(
    store: SessionStore,
    *,
    entered: asyncio.Event | None = None,
    drift: bool = False,
    ledger: SQLiteBudgetLedger | None = None,
) -> tuple[CayuApp, _RecoveryProvider, _RecoveryProvider]:
    primary = _RecoveryProvider("primary")
    backup = _RecoveryProvider("backup", entered=entered, behavior_version="2" if drift else "1")
    app = CayuApp(
        session_store=store,
        budget_ledger=ledger,
        budget_policy=(
            None
            if ledger is None
            else BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("10"),
                        pricing=PriceBook(
                            prices=tuple(
                                ModelPrice.fixed(
                                    provider_name=name,
                                    model=model,
                                    input_per_million=Decimal(price),
                                    output_per_million=Decimal("0"),
                                )
                                for name, model, price in (
                                    ("primary", "small", "1"),
                                    ("backup", "large", "2"),
                                )
                            )
                        ),
                        reservation=BudgetReservation(
                            max_input_tokens=1_000_000, max_output_tokens=0
                        ),
                    ),
                )
            )
        ),
        enable_logging=False,
    )
    app.register_provider(primary, default=True)
    app.register_provider(backup)
    app.register_agent(AgentSpec(name="agent", model="small"))
    return app, primary, backup


@pytest.mark.parametrize("drift,budgeted", [(False, False), (True, False), (False, True)])
def test_public_recovery_plan_after_fallback_deadline_and_sqlite_reopen(
    monkeypatch, tmp_path, drift, budgeted
):

    async def scenario():
        path = tmp_path / "cancelled-fallback.sqlite"
        store = _StageSQLiteStore(path)
        ledger = (
            SQLiteBudgetLedger(tmp_path / "budget.sqlite", reservation_ttl_seconds=None)
            if budgeted
            else None
        )
        entered = asyncio.Event()
        app, primary, backup = _application(store, entered=entered, ledger=ledger)

        async def collect():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="cancelled-fallback",
                        messages=[Message.text("user", "hello")],
                        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                        ),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        started = asyncio.create_task(entered.wait())
        try:
            done, _ = await asyncio.wait(
                {task, started}, timeout=20, return_when=asyncio.FIRST_COMPLETED
            )
            if started not in done:
                assert task in done, "Provider did not enter before the test deadline"
                result = await task
                pytest.fail(f"Run ended before backup dispatch: {result[-1].payload}")
            with pytest.raises(ModelStreamDeadlineError):
                await asyncio.wait_for(task, timeout=20)
            assert not task.cancelled()
            assert len(primary.requests) == len(backup.requests) == 1
            active = await store.load_active_model_completion_stage("cancelled-fallback")
            assert active is not None and active.stage.intent["provider_name"] == "backup"
            if ledger is not None:
                assert len(active.stage.reservation_ids) == 1
                reservation = await ledger.load_reservation(active.stage.reservation_ids[0])
                assert reservation is not None and reservation.model == "large"
                assert reservation.reserved_amount == Decimal("2")
        finally:
            if not started.done():
                started.cancel()
            await asyncio.gather(started, return_exceptions=True)
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await store.close()
            if ledger is not None:
                await ledger.close()

        reopened = _StageSQLiteStore(path)
        ledger = (
            SQLiteBudgetLedger(tmp_path / "budget.sqlite", reservation_ttl_seconds=None)
            if budgeted
            else None
        )
        try:
            app, primary, backup = _application(reopened, drift=drift, ledger=ledger)
            before_session = await reopened.load("cancelled-fallback")
            before_checkpoint = await reopened.load_checkpoint("cancelled-fallback")
            before_events = await reopened.load_events("cancelled-fallback")
            before_reservation = (
                None
                if ledger is None
                else await ledger.load_reservation(active.stage.reservation_ids[0])
            )
            plan = await app.plan_recovery(
                RecoveryPlanRequest(
                    selection=RecoveryPlanSelection(session_ids=("cancelled-fallback",))
                )
            )
            assert len(plan.items) == 1
            item = plan.items[0]
            assert item.active_model_stage is not None
            assert RecoveryPlanAction.AUTOMATIC_REPAIR not in item.allowed_actions
            if drift:
                assert item.registration.validated_execution_profile_fingerprint is None
            else:
                assert item.registration.validated_execution_profile_fingerprint is not None, item
                assert RecoveryBlockerCode.MODEL_EFFECT_OUTCOME_UNKNOWN in {
                    blocker.code for blocker in item.blockers
                }
                assert RecoveryPlanAction.MODEL_MARK_INTERRUPTED in item.allowed_actions
            assert not primary.requests and not backup.requests
            assert await reopened.load("cancelled-fallback") == before_session
            assert await reopened.load_checkpoint("cancelled-fallback") == before_checkpoint
            assert await reopened.load_events("cancelled-fallback") == before_events
            if ledger is not None:
                assert (
                    await ledger.load_reservation(active.stage.reservation_ids[0])
                    == before_reservation
                )
            if not drift:
                execution = RecoveryExecutionRequest(
                    plan=plan,
                    execution_id="settle-fallback-deadline",
                    decisions=(
                        RecoveryDecision(
                            item_id=item.item_id, action=RecoveryPlanAction.MODEL_MARK_INTERRUPTED
                        ),
                    ),
                )
                receipt = await app.execute_recovery(execution)
                assert receipt.items[0].status is RecoveryItemExecutionStatus.EXECUTED, receipt
                assert (
                    await reopened.load_active_model_completion_stage("cancelled-fallback") is None
                )
                after_events = await reopened.load_events("cancelled-fallback")
                settled_before_replay = (
                    None
                    if ledger is None
                    else await ledger.load_reservation(active.stage.reservation_ids[0])
                )
                replay = await app.execute_recovery(execution)
                assert replay.items[0] == receipt.items[0].model_copy(update={"replayed": True})
                assert await reopened.load_events("cancelled-fallback") == after_events
                assert not primary.requests and not backup.requests
                if ledger is not None:
                    settled = await ledger.load_reservation(active.stage.reservation_ids[0])
                    assert settled is not None and settled.status == "reconciled"
                    assert settled.model == "large" and settled.actual_amount == Decimal("2")
                    assert settled == settled_before_replay
                resumed = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="cancelled-fallback",
                            messages=[Message.text("user", "start a new request")],
                            retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                        )
                    )
                ]
                assert resumed[-1].type is EventType.SESSION_COMPLETED, resumed[-1].payload
                assert not primary.requests and len(backup.requests) == 1
                latest = await reopened.load_checkpoint("cancelled-fallback")
                assert latest is not None and before_checkpoint is not None
                assert latest["model_failover"]["candidate_index"] == 1
                assert (
                    latest["model_failover"]["logical_step_id"]
                    != before_checkpoint["model_failover"]["logical_step_id"]
                )
        finally:
            await reopened.close()
            if ledger is not None:
                await ledger.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "sqlite_drift"])
def test_public_approval_continuation_retains_backup_selection(monkeypatch, tmp_path, backend):

    class ApprovalProvider(_RecoveryProvider):
        emit_tool = True

        async def stream(self, request):
            if self.name == "backup" and self.emit_tool and not self.requests:
                self.requests.append(request)
                yield ModelStreamEvent.tool_call(id="echo-call", name="echo", arguments={})
                yield ModelStreamEvent.completed()
                return
            async for event in super().stream(request):
                yield event

    class EchoTool(Tool):
        spec = ToolSpec(
            name="echo",
            description="Echo",
            input_schema={"type": "object", "properties": {}},
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:failover-recovery:echo",
                behavior_version="1",
                implementation_version="1",
            ),
        )

        def __init__(self):
            self.calls = 0

        async def run(self, ctx, args):
            self.calls += 1
            return ToolResult(content="echoed")

    def application(store, *, restarted=False):
        primary = ApprovalProvider("primary")
        backup = ApprovalProvider(
            "backup", behavior_version="2" if restarted and backend == "sqlite_drift" else "1"
        )
        backup.emit_tool = not restarted
        tool = EchoTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(
            AgentSpec(name="agent", model="small"),
            tools=[tool],
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
        return app, primary, backup, tool

    async def scenario(store):
        app, primary, backup, tool = application(store)
        paused = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="approval-fallback",
                    messages=[Message.text("user", "hello")],
                    retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                    failover=ModelFailoverPolicy(
                        fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                    ),
                )
            )
        ]
        assert paused[-1].type is EventType.SESSION_INTERRUPTED
        assert len(primary.requests) == len(backup.requests) == 1
        assert tool.calls == 0
        pending = next(
            event.payload["approval"]
            for event in paused
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        if backend != "memory":
            await store.close()
            store = _StageSQLiteStore(tmp_path / "approval.sqlite")
            try:
                await finish(store, pending, *application(store, restarted=True))
            finally:
                await store.close()
        else:
            await finish(store, pending, app, primary, backup, tool)

    async def finish(store, pending, app, primary, backup, tool):
        prior_primary = len(primary.requests)
        prior_backup = len(backup.requests)
        before = await store.load_checkpoint("approval-fallback")
        if backend == "sqlite_drift":
            with pytest.raises(ExecutionProfileMismatchError):
                _ = [
                    event
                    async for event in app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id="approval-fallback",
                            approval_id=pending["approval_id"],
                            tool_round_id=pending["tool_round_id"],
                            tool_call_id=pending["tool_call_id"],
                            decision=ToolApprovalDecision.APPROVE,
                        )
                    )
                ]
            assert tool.calls == 0 and not primary.requests and not backup.requests
            assert await store.load_checkpoint("approval-fallback") == before
            return
        completed = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="approval-fallback",
                    approval_id=pending["approval_id"],
                    tool_round_id=pending["tool_round_id"],
                    tool_call_id=pending["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        assert completed[-1].type is EventType.SESSION_COMPLETED, [
            (event.type, event.payload) for event in completed[-5:]
        ]
        assert tool.calls == 1
        assert len(primary.requests) == prior_primary
        assert len(backup.requests) == prior_backup + 1
        assert backup.requests[-1].model == "large"
        checkpoint = await store.load_checkpoint("approval-fallback")
        assert checkpoint is not None and checkpoint["model_failover"]["candidate_index"] == 1
        assert await store.load_active_model_completion_stage("approval-fallback") is None

    async def owned_scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "approval.sqlite")
        )
        try:
            await scenario(store)
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(owned_scenario())


def _process_recovery_store(backend, location, committed=None):
    class PausedPreparation(SessionStore):
        model_failover_stage_version = 1
        invocation_lifecycle_command_version = 1

        async def _prepare_model_completion_stage_atomic(self, prepared):
            result = await super()._prepare_model_completion_stage_atomic(prepared)
            if committed is not None and result.stage.intent.get("provider_name") == "backup":
                committed.set()
                await asyncio.Event().wait()
            return result

    if backend == "postgres":
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        class PausedPostgres(PausedPreparation, PostgresSessionStore):
            model_failover_stage_version = 1
            invocation_lifecycle_command_version = 1

        return PausedPostgres(location, schema_mode=SchemaMode.CREATE)

    class PausedSQLite(PausedPreparation, _StageSQLiteStore):
        model_failover_stage_version = 1
        invocation_lifecycle_command_version = 1

    return PausedSQLite(location)


def _prepared_fallback_worker(backend, location, committed):
    async def run():
        store = _process_recovery_store(backend, location, committed)
        try:
            app, _, _ = _application(store)
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="prepared-fallback",
                        messages=[Message.text("user", "hello")],
                        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                        ),
                    )
                )
            ]
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_process_death_after_backup_preparation_recovers_without_resetting_selection(
    monkeypatch, tmp_path, backend, request
):
    location = (
        request.getfixturevalue("postgres_dsn")
        if backend == "postgres"
        else tmp_path / "prepared-fallback.sqlite"
    )
    context = multiprocessing.get_context("spawn")
    committed = context.Event()
    process = context.Process(target=_prepared_fallback_worker, args=(backend, location, committed))
    process.start()
    try:
        assert committed.wait(30), f"Preparation did not commit; child exit={process.exitcode}"
        process.kill()
        process.join(10)
        assert not process.is_alive() and process.exitcode != 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        process.close()

    async def scenario():
        store = _process_recovery_store(backend, location)
        try:
            app, primary, backup = _application(store)
            active = await store.load_active_model_completion_stage("prepared-fallback")
            assert active is not None and active.stage.intent["provider_name"] == "backup"
            stage_id = active.stage.stage_id
            assert (
                await store.load_model_completion_stage_dispatch("prepared-fallback", stage_id)
                is None
            )
            before = await store.load_checkpoint("prepared-fallback")
            assert before is not None
            before_events = await store.load_events("prepared-fallback")
            # Process loss is not invocation-release authority. A plain resume
            # must remain fenced until the existing recovery owner settles it.
            with pytest.raises(SessionStatusConflict):
                _ = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="prepared-fallback",
                            messages=[Message.text("user", "continue")],
                            retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                        )
                    )
                ]
            assert await store.load_checkpoint("prepared-fallback") == before
            assert await store.load_events("prepared-fallback") == before_events
            assert not primary.requests and not backup.requests
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="prepared-fallback", inactive_for_seconds=0
                )
            )
            assert not primary.requests and not backup.requests
            assert await store.load_active_model_completion_stage("prepared-fallback") is None
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="prepared-fallback",
                        messages=[Message.text("user", "continue")],
                        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                    )
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED, resumed[-1].payload
            assert not primary.requests and len(backup.requests) == 1
            assert (
                await store.load_model_completion_stage_abandonment("prepared-fallback", stage_id)
                is not None
            )
            after = await store.load_checkpoint("prepared-fallback")
            assert after is not None and after["model_failover"]["candidate_index"] == 1
            assert (
                after["model_failover"]["logical_step_id"]
                != before["model_failover"]["logical_step_id"]
            )
            assert after["model_failover"]["attempts_used"] == 1
        finally:
            await store.close()

    asyncio.run(scenario())
