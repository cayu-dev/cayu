"""Compaction budget settlement contracts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from tests.runtime.runtime_compaction_support import compaction_price_book
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import _private_events_for_public_events, collect_events

import cayu.runtime._run_limits as run_limits_module
from cayu._exception_groups import exception_cause, iter_exception_tree
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation, InMemoryBudgetLedger
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime._model_completion_contracts import ModelCompletionManualRecoveryRequired
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.sessions.requests import RunRequest
from cayu.storage.budget_ledger import SQLiteBudgetLedger


def test_automatic_compaction_cancellation_releases_partial_reservation_setup() -> None:
    class BlockingSecondReservationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self.reserve_calls = 0
            self.second_reservation_started = asyncio.Event()

        async def reserve(self, **kwargs):
            self.reserve_calls += 1
            if self.reserve_calls == 2:
                self.second_reservation_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            return await super().reserve(**kwargs)

    async def run() -> None:
        session_id = "sess_compaction_partial_reservation_cancelled"
        store = InMemorySessionStore()
        ledger = BlockingSecondReservationLedger()
        compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
        runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
        reservation = BudgetReservation(max_input_tokens=10, max_output_tokens=0)
        pricing = compaction_price_book()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                    BudgetLimit(
                        scope="agent",
                        key="assistant",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=reservation,
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(runtime_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compactor_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        await asyncio.wait_for(ledger.second_reservation_started.wait(), timeout=10)
        task.cancel("cancel reservation setup")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        records = tuple(ledger._records.values())
        assert exc_info.value.args == ("cancel reservation setup",)
        assert len(records) == 1
        assert records[0].status == "released"
        assert records[0].reason == "reservation setup cancelled"
        assert compactor_provider.requests == []
        assert runtime_provider.requests == []
        stored_events = await store.load_events(session_id)
        reserved = [event for event in stored_events if event.type == EventType.BUDGET_RESERVED]
        assert len(reserved) == 1
        assert reserved[0].payload["reservation_id"] == records[0].reservation_id
        released = [
            event for event in stored_events if event.type == EventType.BUDGET_RESERVATION_RELEASED
        ]
        assert len(released) == 1
        assert released[0].payload["reservation_id"] == records[0].reservation_id
        assert released[0].payload["reason"] == "reservation setup cancelled"

    asyncio.run(run())


def test_automatic_compaction_cancellation_while_persisting_reservation_releases_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        session_id = "sess_compaction_reservation_publish_cancelled"
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
        compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
        runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=compaction_price_book(),
                        reservation=BudgetReservation(
                            max_input_tokens=10,
                            max_output_tokens=0,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(runtime_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compactor_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )
        reservation_publish_started = asyncio.Event()
        original_emit = app._event_writer.emit
        block_reservation_publish = True

        async def block_first_reservation_publish(event: Event) -> Event:
            nonlocal block_reservation_publish
            if block_reservation_publish and event.type == EventType.BUDGET_RESERVED:
                block_reservation_publish = False
                reservation_publish_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", block_first_reservation_publish)
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        await asyncio.wait_for(reservation_publish_started.wait(), timeout=10)
        task.cancel("cancel reservation event persistence")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        records = tuple(ledger._records.values())
        assert exc_info.value.args == ("cancel reservation event persistence",)
        assert len(records) == 1
        assert records[0].status == "released"
        assert records[0].reason == "reservation setup cancelled"
        assert compactor_provider.requests == []
        assert runtime_provider.requests == []
        stored_events = await store.load_events(session_id)
        assert EventType.BUDGET_RESERVED not in [event.type for event in stored_events]
        released = [
            event for event in stored_events if event.type == EventType.BUDGET_RESERVATION_RELEASED
        ]
        assert len(released) == 1
        assert released[0].payload["reservation_id"] == records[0].reservation_id
        assert released[0].payload["reason"] == "reservation setup cancelled"

    asyncio.run(run())


def test_automatic_compaction_cancellation_waits_for_reconciliation() -> None:
    class BlockingReconciliationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self.reconciliation_started = asyncio.Event()
            self.allow_reconciliation = asyncio.Event()

        async def reconcile(self, **kwargs):
            self.reconciliation_started.set()
            await self.allow_reconciliation.wait()
            return await super().reconcile(**kwargs)

    async def run() -> None:
        session_id = "sess_compaction_reconciliation_cancelled"
        store = InMemorySessionStore()
        ledger = BlockingReconciliationLedger()
        compactor_provider = FakeProvider(
            [ModelStreamEvent.completed({"usage": {"input_tokens": 1}})]
        )
        runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=compaction_price_book(),
                        reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(runtime_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compactor_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        await asyncio.wait_for(ledger.reconciliation_started.wait(), timeout=10)
        task.cancel("cancel during compaction reconciliation")
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel("repeated cancellation must not abandon settlement")
        await asyncio.sleep(0)
        assert not task.done()
        ledger.allow_reconciliation.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel during compaction reconciliation",)
        records = tuple(ledger._records.values())
        assert len(records) == 1
        assert records[0].status == "reconciled"
        assert records[0].actual_amount == Decimal("0.000001")
        stored_events = await store.load_events(session_id)
        reconciled = [event for event in stored_events if event.type == EventType.BUDGET_RECONCILED]
        assert len(reconciled) == 1
        assert reconciled[0].payload["reservation_id"] == records[0].reservation_id
        assert len(compactor_provider.requests) == 1
        assert runtime_provider.requests == []

    asyncio.run(run())


def test_automatic_compaction_preserves_provider_cancellation_during_settlement() -> None:
    class BlockingCompactionProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            self.started.set()
            await asyncio.Event().wait()
            yield ModelStreamEvent.completed({})

    class BlockingReconciliationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self.reconciliation_started = asyncio.Event()
            self.allow_reconciliation = asyncio.Event()

        async def reconcile(self, **kwargs):
            self.reconciliation_started.set()
            await self.allow_reconciliation.wait()
            return await super().reconcile(**kwargs)

    async def run() -> None:
        session_id = "sess_compaction_provider_and_settlement_cancelled"
        store = InMemorySessionStore()
        ledger = BlockingReconciliationLedger()
        compactor_provider = BlockingCompactionProvider()
        runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=compaction_price_book(),
                        reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(runtime_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compactor_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        await asyncio.wait_for(compactor_provider.started.wait(), timeout=10)
        task.cancel("first cancellation during provider execution")
        await asyncio.wait_for(ledger.reconciliation_started.wait(), timeout=10)
        task.cancel("later cancellation during settlement")
        await asyncio.sleep(0)
        assert not task.done()
        ledger.allow_reconciliation.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("first cancellation during provider execution",)
        records = tuple(ledger._records.values())
        assert len(records) == 1
        assert records[0].status == "reconciled"
        assert records[0].actual_amount == records[0].reserved_amount
        stored_events = await store.load_events(session_id)
        reconciled = [event for event in stored_events if event.type == EventType.BUDGET_RECONCILED]
        assert len(reconciled) == 1
        assert reconciled[0].payload["reservation_id"] == records[0].reservation_id
        assert len(compactor_provider.requests) == 1
        assert runtime_provider.requests == []

    asyncio.run(run())


def test_automatic_compaction_cancellation_waits_for_reconciliation_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        session_id = "sess_compaction_reconciliation_publish_cancelled"
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
        compactor_provider = FakeProvider(
            [ModelStreamEvent.completed({"usage": {"input_tokens": 1}})]
        )
        runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=compaction_price_book(),
                        reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(runtime_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compactor_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )
        reconciliation_publish_started = asyncio.Event()
        allow_reconciliation_publish = asyncio.Event()
        original_persist_exact_replay = app._event_writer.persist_exact_replay

        async def block_reconciliation_publish(event: Event) -> Event:
            if event.type == EventType.BUDGET_RECONCILED:
                reconciliation_publish_started.set()
                await allow_reconciliation_publish.wait()
            return await original_persist_exact_replay(event)

        monkeypatch.setattr(
            app._event_writer,
            "persist_exact_replay",
            block_reconciliation_publish,
        )
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        await asyncio.wait_for(reconciliation_publish_started.wait(), timeout=10)
        records = tuple(ledger._records.values())
        assert len(records) == 1
        assert records[0].status == "reconciled"
        task.cancel("cancel reconciliation event persistence")
        await asyncio.sleep(0)
        assert not task.done()
        allow_reconciliation_publish.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel reconciliation event persistence",)
        stored_events = await store.load_events(session_id)
        reconciled = [event for event in stored_events if event.type == EventType.BUDGET_RECONCILED]
        assert len(reconciled) == 1
        assert reconciled[0].payload["reservation_id"] == records[0].reservation_id
        assert len(compactor_provider.requests) == 1
        assert runtime_provider.requests == []

    asyncio.run(run())


def test_automatic_compaction_cancellation_stays_authoritative_if_settlement_fails() -> None:
    class BlockingFailingReconciliationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self.reconciliation_started = asyncio.Event()
            self.allow_failure = asyncio.Event()

        async def reconcile(self, **kwargs):
            del kwargs
            self.reconciliation_started.set()
            await self.allow_failure.wait()
            raise RuntimeError("budget ledger unavailable")

    async def run() -> None:
        ledger = BlockingFailingReconciliationLedger()
        compactor_provider = FakeProvider([ModelStreamEvent.completed({})])
        runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
        app = CayuApp(
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=compaction_price_book(),
                        reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
                    ),
                )
            ),
            budget_ledger=ledger,
        )
        app.register_provider(runtime_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=compactor_provider,
                    model="summary-model",
                ),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_compaction_cancelled_settlement_failure",
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                ),
            )
        )
        await asyncio.wait_for(ledger.reconciliation_started.wait(), timeout=10)
        task.cancel("cancel while settlement fails")
        ledger.allow_failure.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task

        assert exc_info.value.args == ("cancel while settlement fails",)
        assert exc_info.value.__notes__ == [
            "Automatic compaction budget settlement also failed: "
            "RuntimeError: budget ledger unavailable",
            "Continuation recovery cleanup failed during cancelled session "
            "finalization: ModelCompletionBudgetSettlementPending. The original "
            "failure remains authoritative.",
        ]
        cleanup_failure = exception_cause(exc_info.value)
        assert isinstance(cleanup_failure, BaseExceptionGroup)
        settlement_failure = exception_cause(cleanup_failure)
        assert type(settlement_failure) is RuntimeError
        assert str(settlement_failure) == "budget ledger unavailable"
        cleanup_failures = tuple(iter_exception_tree(cleanup_failure))
        assert (
            sum(
                isinstance(
                    failure,
                    run_limits_module.ModelCompletionBudgetSettlementPending,
                )
                for failure in cleanup_failures
            )
            == 1
        )
        assert len(compactor_provider.requests) == 1
        assert runtime_provider.requests == []
        active = await app.session_store.load_active_model_completion_stage(
            "sess_compaction_cancelled_settlement_failure"
        )
        assert active is not None
        assert active.stage.state == "completed"
        records = tuple(ledger._records.values())
        assert [record.status for record in records] == ["active"]

    asyncio.run(run())


def test_automatic_compaction_settles_other_limits_after_one_reconciliation_fails() -> None:
    class FailMiddleReconciliationLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self.reconcile_calls = 0

        async def reconcile(self, **kwargs):
            self.reconcile_calls += 1
            if self.reconcile_calls == 2:
                raise RuntimeError("middle reconciliation failed")
            return await super().reconcile(**kwargs)

    session_id = "sess_compaction_partial_settlement_failure"
    ledger = FailMiddleReconciliationLedger()
    reservation = BudgetReservation(max_input_tokens=10, max_output_tokens=0)
    pricing = compaction_price_book()
    compactor_provider = FakeProvider([ModelStreamEvent.completed({"usage": {"input_tokens": 1}})])
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=pricing,
                    reservation=reservation,
                ),
                BudgetLimit(
                    scope="agent",
                    key="assistant",
                    max_estimated_cost=Decimal("1"),
                    pricing=pricing,
                    reservation=reservation,
                ),
                BudgetLimit(
                    scope="causal",
                    key=session_id,
                    max_estimated_cost=Decimal("1"),
                    pricing=pricing,
                    reservation=reservation,
                ),
            )
        ),
        budget_ledger=ledger,
    )
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    async def run():
        events = []
        with pytest.raises(RuntimeError, match="middle reconciliation failed") as exc_info:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                )
            ):
                events.append(event)
        return (
            events,
            exc_info.value,
            await app.session_store.load(session_id),
            await app.session_store.load_active_model_completion_stage(session_id),
        )

    events, failure, session, active = asyncio.run(run())

    records = tuple(ledger._records.values())
    assert ledger.reconcile_calls == 3
    assert [record.status for record in records] == ["reconciled", "active", "reconciled"]
    assert active is not None
    assert active.stage.state == "completed"
    assert set(active.stage.reservation_ids) == {record.reservation_id for record in records}
    assert session is not None and session.status == SessionStatus.RUNNING
    assert EventType.SESSION_FAILED not in [event.type for event in events]
    assert failure.__notes__ == [
        "Automatic compaction budget settlement failed for reservation "
        f"{records[1].reservation_id}."
    ]
    failure_cause = exception_cause(failure)
    assert isinstance(failure_cause, BaseExceptionGroup)
    cause_failures = tuple(iter_exception_tree(failure_cause))
    assert (
        sum(
            isinstance(
                item,
                run_limits_module.ModelCompletionBudgetSettlementPending,
            )
            for item in cause_failures
        )
        == 1
    )

    reconciled = [event for event in events if event.type == EventType.BUDGET_RECONCILED]
    private_reconciled = asyncio.run(
        _private_events_for_public_events(app.session_store, session_id, reconciled)
    )
    assert [event.payload["reservation_id"] for event in private_reconciled] == [
        records[0].reservation_id,
        records[2].reservation_id,
    ]
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []

    with pytest.raises(
        ModelCompletionManualRecoveryRequired,
        match="promoted without a durable context checkpoint",
    ):
        asyncio.run(
            app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_for_seconds=0,
                )
            )
        )

    recovered_records = tuple(ledger._records.values())
    assert ledger.reconcile_calls == 4
    assert [record.status for record in recovered_records] == [
        "reconciled",
        "reconciled",
        "reconciled",
    ]
    assert recovered_records[1].reason == (run_limits_module.UNKNOWN_POST_DISPATCH_BUDGET_REASON)
    assert asyncio.run(app.session_store.load_active_model_completion_stage(session_id)) is None
    recovered_events = asyncio.run(app.session_store.load_events(session_id))
    recovered_completions = [
        event
        for event in recovered_events
        if event.type is EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert len(recovered_completions) == 1
    assert len(compactor_provider.requests) == 1
    assert runtime_provider.requests == []


def test_automatic_compaction_reconciliation_survives_sqlite_ledger_reopen(
    tmp_path,
) -> None:
    ledger_path = tmp_path / "automatic-compaction-budget.sqlite"
    pricing = compaction_price_book()
    limit = BudgetLimit(
        scope="app",
        max_estimated_cost=Decimal("0.000029"),
        pricing=pricing,
        reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=0),
    )
    compactor_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("model summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 20, "output_tokens": 0},
                }
            ),
        ]
    )
    first_runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    first_ledger = SQLiteBudgetLedger(ledger_path)
    first_app = CayuApp(
        budget_policy=BudgetPolicy(limits=(limit,)),
        budget_ledger=first_ledger,
    )
    first_app.register_provider(first_runtime_provider, default=True)
    first_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compactor_provider,
                model="summary-model",
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    asyncio.run(
        collect_events(
            first_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_sqlite_first",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )
    asyncio.run(first_ledger.close())

    second_runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    second_ledger = SQLiteBudgetLedger(ledger_path)
    second_app = CayuApp(
        budget_policy=BudgetPolicy(limits=(limit,)),
        budget_ledger=second_ledger,
    )
    second_app.register_provider(second_runtime_provider, default=True)
    second_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    second_events = asyncio.run(
        collect_events(
            second_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_sqlite_second",
                messages=[Message.text("user", "current")],
            ),
        )
    )
    asyncio.run(second_ledger.close())

    assert first_runtime_provider.requests == []
    assert second_runtime_provider.requests == []
    failure = next(
        event for event in second_events if event.type == EventType.BUDGET_RESERVATION_FAILED
    )
    assert Decimal(failure.payload["actual"]) == Decimal("0.00003")
    assert second_events[-1].type == EventType.SESSION_INTERRUPTED
