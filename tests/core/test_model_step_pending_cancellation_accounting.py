from __future__ import annotations

import asyncio

import pytest
from tests.core.test_provider_operation_offline_recovery import (
    _budget_policy,
    _LiveCancellableBudgetedOperationAdapter,
    _LiveCancellableBudgetedOperationProvider,
)

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.budgets.base import InMemoryBudgetLedger
from cayu.providers.operations import ProviderOperationSnapshot, ProviderOperationStatus
from cayu.sessions.memory import InMemorySessionStore


@pytest.mark.parametrize("ttl", [None, 3600])
@pytest.mark.parametrize("cancellation_outcome", ["pending", "failed"])
def test_cancellation_keeps_pending_provider_operation_accounting(ttl, cancellation_outcome):
    class PendingCancelAdapter(_LiveCancellableBudgetedOperationAdapter):
        async def cancel(self, state):
            self.cancel_calls.append(state)
            if cancellation_outcome == "failed":
                raise ConnectionError("Cancellation acknowledgement unavailable")
            return ProviderOperationSnapshot(
                state=state, status=ProviderOperationStatus.IN_PROGRESS
            )

    async def scenario():
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=ttl)
        provider = _LiveCancellableBudgetedOperationProvider()
        provider.adapter = PendingCancelAdapter()
        app = CayuApp(
            session_store=store,
            budget_ledger=ledger,
            budget_policy=_budget_policy(provider.name),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "pending-cancel-accounting"

        async def consume():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "Cancel without losing the remote result")],
                    )
                )
            ]

        caller = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(provider.adapter.stream_started.wait(), 10)
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(caller, 10)
            assert provider.adapter.start_calls == 1
            assert len(provider.adapter.cancel_calls) == 1
            durable = await store.load_events(session_id)
            resolved = [
                e for e in durable if e.type is EventType.PROVIDER_OPERATION_CANCEL_RESOLVED
            ]
            assert len(resolved) == 1
            assert resolved[0].payload["cancellation_status"] == cancellation_outcome
            active = await store.load_active_model_completion_stage(session_id)
            assert active is not None, "The original operation must remain recoverable"
            record = next(iter(ledger._records.values()))
            assert record.status == "active", (
                "A pending cancellation must defer accounting until the exact operation resolves",
                record.status,
                record.actual_amount,
            )
            assert record.actual_amount is None
            assert not any(e.type is EventType.BUDGET_RECONCILED for e in durable)
        finally:
            if not caller.done():
                caller.cancel()
            await asyncio.gather(caller, return_exceptions=True)
            await app.aclose()

    asyncio.run(scenario())
