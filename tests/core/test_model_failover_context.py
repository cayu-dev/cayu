"""Public selected-target context verification and compaction regressions."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from tests.core.test_model_failover_recovery import _RecoveryProvider

from cayu import (
    AgentSpec,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    CheckpointCompactionContextPolicy,
    DefaultContextPolicy,
    EventType,
    ExecutionProfileBehaviorIdentity,
    Message,
    MessageWindowContextPolicy,
    ModelCompactor,
    ModelFailoverPolicy,
    ModelPrice,
    ModelTarget,
    PriceBook,
    RunRequest,
    UsageTriggeredContextPolicy,
)
from cayu.providers.base import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelProviderError,
    ModelStreamEvent,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.sessions.base import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore


class _CountingProvider(_RecoveryProvider):
    def __init__(self, name, count):
        super().__init__(name)
        self.count = count
        self.count_requests = []
        self.after_count = None

    async def count_input_tokens(self, request):
        self.count_requests.append(request)
        if self.after_count is not None:
            self.after_count()
        return InputTokenCountResult(
            input_tokens=self.count,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        )


class _InspectPrefixPolicy(DefaultContextPolicy):
    def __init__(self):
        super().__init__()
        self.requests = []

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:failover:prefix-policy", behavior_version="1", implementation_version="1"
        )

    async def build(self, request):
        messages = await super().build(request)
        self.requests.append(await request.build_cache_prefix_request(messages))
        return messages


def _request():
    return RunRequest(
        agent_name="agent",
        session_id="context-fallback",
        messages=[
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ],
        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
        failover=ModelFailoverPolicy(
            fallbacks=(ModelTarget(provider_name="backup", model="large"),),
            max_total_attempts=2,
        ),
    )


def _app(store, *, triggered_policy, base_policy=None, budget_policy=None):
    primary = _CountingProvider("primary", 1)
    backup = _CountingProvider("backup", 200)
    app = CayuApp(session_store=store, budget_policy=budget_policy, enable_logging=False)
    app.register_provider(primary, default=True)
    app.register_provider(backup)
    app.register_agent(
        AgentSpec(name="agent", model="small"),
        context_policy=UsageTriggeredContextPolicy(
            base_policy=DefaultContextPolicy() if base_policy is None else base_policy,
            triggered_policy=triggered_policy,
            trigger_estimated_context_tokens=100,
            verify_estimate_with_provider_count=True,
            provider_count_min_delta_tokens=1,
        ),
    )
    return app, primary, backup


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_fallback_context_counter_uses_selected_model(tmp_path, backend):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "context.sqlite")
        )
        try:
            prefix_policy = _InspectPrefixPolicy()
            app, primary, backup = _app(
                store,
                triggered_policy=MessageWindowContextPolicy(max_messages=1),
                base_policy=prefix_policy,
            )
            events = [event async for event in app.run(_request())]
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert [request.model for request in primary.count_requests] == ["small"]
            assert [request.model for request in backup.count_requests] == ["large"]
            assert [request.model for request in backup.requests] == ["large"]
            assert [request.model for request in prefix_policy.requests] == ["small", "large"]
            assert len(primary.requests) == 1
            checkpoint = await store.load_checkpoint("context-fallback")
            assert checkpoint["model_failover"]["candidate_index"] == 1
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("corruption", ["missing", "digest", "step"])
def test_public_fallback_compaction_rejects_changed_predecessor(corruption):
    class Store(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        corrupt = False

        async def load_active_model_completion_stage(self, session_id):
            active = await super().load_active_model_completion_stage(session_id)
            if not self.corrupt or active is None:
                return active
            # Corrupt only the borrowing read, not unrelated terminal settlement.
            self.corrupt = False
            if corruption == "missing":
                return None
            change = (
                {"preparation_digest": "0" * 64}
                if corruption == "digest"
                else {"logical_step_id": "another-model-step"}
            )
            return active.model_copy(update={"stage": active.stage.model_copy(update=change)})

    async def run():
        store = Store()
        summary = _RecoveryProvider("summary")
        app, primary, backup = _app(
            store,
            triggered_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=summary, model="summary-model"),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )
        backup.after_count = lambda: setattr(store, "corrupt", True)
        events = [event async for event in app.run(_request())]
        assert events[-1].type is EventType.SESSION_FAILED
        assert len(primary.requests) == 1
        assert not summary.requests and not backup.requests
        checkpoint = await store.load_checkpoint("context-fallback")
        assert checkpoint["model_failover"]["candidate_index"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel", "budgeted"])
def test_public_fallback_model_compaction_borrows_exact_predecessor(tmp_path, backend, outcome):
    budgeted = outcome == "budgeted"
    if budgeted:
        outcome = "success"

    async def run():
        entered = asyncio.Event()
        predecessor = []
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "compaction.sqlite")
        )

        class SummaryProvider(_RecoveryProvider):
            async def stream(self, request):
                self.requests.append(request)
                active = await store.load_active_model_completion_stage("context-fallback")
                assert active is not None
                assert active.stage.purpose == "assistant-turn"
                assert active.stage.intent["provider_name"] == "primary"
                predecessor.append(active.stage)
                entered.set()
                if outcome == "cancel":
                    await asyncio.Event().wait()
                if outcome == "failure":
                    raise ModelProviderError(
                        "summary refused", provider="summary", retryable=False, status_code=400
                    )
                yield ModelStreamEvent.text_delta("summary of the old conversation")
                yield ModelStreamEvent.completed(
                    {"usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}
                )

        summary = SummaryProvider("summary")
        try:
            app, primary, backup = _app(
                store,
                budget_policy=(
                    BudgetPolicy(
                        limits=(
                            BudgetLimit(
                                scope="app",
                                max_estimated_cost=Decimal("1"),
                                pricing=PriceBook(
                                    prices=tuple(
                                        ModelPrice.fixed(
                                            provider_name=name,
                                            model=model,
                                            input_per_million=Decimal(price),
                                            output_per_million=Decimal(price),
                                        )
                                        for name, model, price in (
                                            ("primary", "small", "1"),
                                            ("backup", "large", "2"),
                                            ("summary", "summary-model", "3"),
                                        )
                                    )
                                ),
                                reservation=BudgetReservation(
                                    max_input_tokens=10, max_output_tokens=10
                                ),
                            ),
                        )
                    )
                    if budgeted
                    else None
                ),
                triggered_policy=CheckpointCompactionContextPolicy(
                    compactor=ModelCompactor(provider=summary, model="summary-model"),
                    max_user_turns=1,
                    compact_after_messages=2,
                ),
            )
            events = []

            async def consume():
                async for event in app.run(_request()):
                    events.append(event)

            task = asyncio.create_task(consume())
            if outcome == "cancel":
                ready = asyncio.create_task(entered.wait())
                try:
                    done, _ = await asyncio.wait(
                        (task, ready), timeout=20, return_when=asyncio.FIRST_COMPLETED
                    )
                    if task in done:
                        await task
                    assert entered.is_set(), "compaction never reached provider dispatch"
                    task.cancel()
                    assert task.cancelling() == 1
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert task.cancelled()
                finally:
                    ready.cancel()
                    await asyncio.gather(ready, return_exceptions=True)
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            elif outcome == "failure":
                await task
                assert events[-1].type is EventType.SESSION_FAILED
                assert "summary refused" in str(events[-1].payload)
            else:
                await task
                assert events[-1].type is EventType.SESSION_COMPLETED
            assert len(summary.requests) == 1
            assert len(primary.requests) == 1
            assert len(backup.requests) == (1 if outcome == "success" else 0)
            assert len(predecessor) == 1
            checkpoint = await store.load_checkpoint("context-fallback")
            assert checkpoint["model_failover"]["candidate_index"] == (
                1 if outcome == "success" else 0
            )
            if outcome == "success":
                assert checkpoint["model_failover"]["attempts_used"] == 2
                assert any(event.type is EventType.CONTEXT_COMPACTION_COMPLETED for event in events)
                assert await store.load_active_model_completion_stage("context-fallback") is None
                if budgeted:
                    assert any(
                        event.type is EventType.BUDGET_RECONCILED
                        and Decimal(event.payload["actual_amount"]) == Decimal("0.000009")
                        for event in events
                    )
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())
