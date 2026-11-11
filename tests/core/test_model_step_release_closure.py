"""Closing a result stream must not interrupt pre-dispatch budget cleanup."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_model_budget_settlement_recovery import (
    _app,
    _budget_policy,
    _CompletedProvider,
)

from cayu.budgets.base import (
    DEFAULT_RESERVATION_TTL_SECONDS,
    BudgetLimit,
    BudgetPolicy,
    InMemoryBudgetLedger,
)
from cayu.events import EventType
from cayu.execution_units import new_model_step_identity
from cayu.messages import Message
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest


class _StartedWriteFailureStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    async def append_event(self, session_id, event):
        if event.type is EventType.MODEL_STARTED:
            raise OSError("model.started write temporarily failed")
        return await super().append_event(session_id, event)


@pytest.mark.parametrize("ttl", [DEFAULT_RESERVATION_TTL_SECONDS, None])
@pytest.mark.parametrize("close_during_release", [False, True])
def test_closing_during_release_settles_all_undispatched_limits(ttl, close_during_release):
    async def scenario():
        store = _StartedWriteFailureStore()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=ttl)
        provider = _CompletedProvider()
        app_limit = _budget_policy().limits[0]
        agent_limit = BudgetLimit.model_validate(
            {**app_limit.model_dump(mode="python"), "scope": "agent", "key": "assistant"}
        )
        app = _app(
            store,
            ledger,
            provider,
            budget_policy=BudgetPolicy(limits=(app_limit, agent_limit)),
        )
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="release-closure",
                messages=[Message.text("user", "answer")],
            )
        )
        events = []
        try:
            async for event in stream:
                events.append(event)
                if close_during_release and event.type is EventType.BUDGET_RESERVATION_RELEASED:
                    break
        finally:
            await stream.aclose()
            await app.aclose()
        assert any(event.type is EventType.BUDGET_RESERVATION_RELEASED for event in events)
        if not close_during_release:
            assert events[-1].type is EventType.SESSION_FAILED
        assert not provider.requests
        assert await store.load_active_model_completion_stage("release-closure") is None
        records = list(ledger._records.values())
        assert len(records) == 2
        assert all(record.dispatch_id is None for record in records)
        next_reservation = await ledger.reserve(
            limit=agent_limit,
            session_id="next-session",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=new_model_step_identity().new_attempt(),
        )
        assert next_reservation.accepted, (
            "The unused agent budget must admit the next model request after cleanup; "
            f"reservation states: {[record.status for record in records]}"
        )
        assert [record.status for record in records] == ["released", "released"], (
            "Consumer closure must not strand the remaining undispatched reservation"
        )

    asyncio.run(scenario())
