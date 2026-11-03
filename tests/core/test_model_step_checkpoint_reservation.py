from __future__ import annotations

import asyncio

import pytest
from tests.core.test_model_budget_settlement_recovery import _app, _CompletedProvider

from cayu.budgets.base import InMemoryBudgetLedger
from cayu.events import EventType
from cayu.messages import Message
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest


class _CheckpointReadFailureStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self):
        super().__init__()
        self.fail_next_checkpoint = False
        self.injected = False

    async def load_checkpoint(self, session_id):
        if self.fail_next_checkpoint:
            self.fail_next_checkpoint = False
            self.injected = True
            raise OSError("temporary checkpoint read failure after reservation")
        return await super().load_checkpoint(session_id)


@pytest.mark.parametrize("ttl", [3600, None])
def test_checkpoint_read_failure_releases_undispatched_model_budget(monkeypatch, ttl):
    async def run():
        store = _CheckpointReadFailureStore()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=ttl)
        provider = _CompletedProvider()
        app = _app(store, ledger, provider)
        controller = app._model_step_executor._run_limit_controller
        original_reserve = controller.reserve_for_model_step

        async def reserve_then_inject(**kwargs):
            setup = await original_reserve(**kwargs)
            assert setup.reservations and setup.error is None and setup.failure is None
            store.fail_next_checkpoint = True
            return setup

        monkeypatch.setattr(controller, "reserve_for_model_step", reserve_then_inject)
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="checkpoint-reservation",
                        messages=[Message.text("user", "answer")],
                    )
                )
            ]
            assert store.injected
            assert provider.requests == []
            assert events[-1].type == EventType.SESSION_FAILED
            assert await store.load_active_model_completion_stage("checkpoint-reservation") is None
            records = list(ledger._records.values())
            assert len(records) == 1
            assert records[0].dispatch_id is None
            assert records[0].status == "released", (
                "A failed checkpoint read before provider dispatch must release its reservation."
            )
        finally:
            await app.aclose()

    asyncio.run(run())
