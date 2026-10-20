"""A stream deadline fails the session and records the provider outcome as unknown."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from tests.runtime.runtime_provider_support import VersionedFakeProvider

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.providers import ModelStreamEvent, ProviderStreamDeadlines
from cayu.runtime._run_limits import UNKNOWN_POST_DISPATCH_BUDGET_REASON
from cayu.sessions import ModelCompletionStageDisposition, run_to_completion
from cayu.storage import SQLiteBudgetLedger, SQLiteSessionStore
from cayu.tasks import TaskCreate, TaskStatus
from cayu.tasks.memory import InMemoryTaskStore


class _StallsAfterText(VersionedFakeProvider):
    """The first call streams text and then stalls; later calls answer."""

    def __init__(self, store, session_id: str) -> None:
        super().__init__([])
        self.store = store
        self.session_id = session_id
        self.calls = 0
        self.stages = []

    @property
    def stream_deadlines(self) -> ProviderStreamDeadlines:
        return ProviderStreamDeadlines(
            transport_idle_timeout_s=5,
            protocol_idle_timeout_s=5,
            semantic_progress_timeout_s=0.05,
            absolute_stream_timeout_s=5,
        )

    async def stream(self, request) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.calls += 1
        active = await self.store.load_active_model_completion_stage(self.session_id)
        self.stages.append(active.stage)
        if self.calls == 1:
            yield ModelStreamEvent.text_delta("A partial answer")
            await asyncio.Event().wait()
        yield ModelStreamEvent.text_delta("Answered.")
        yield ModelStreamEvent.completed(
            {"finish_reason": "stop", "usage": {"input_tokens": 1, "output_tokens": 1}}
        )


def _app(tmp_path, session_id, *, task_store=None):
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    ledger = SQLiteBudgetLedger(tmp_path / "budgets.sqlite", reservation_ttl_seconds=None)
    provider = _StallsAfterText(store, session_id)
    app = CayuApp(
        session_store=store,
        budget_ledger=ledger,
        task_store=task_store,
        enable_logging=False,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=PriceBook(
                        prices=(
                            ModelPrice.fixed(
                                provider_name=provider.name,
                                model="fake-model",
                                input_per_million=1,
                                output_per_million=1,
                            ),
                        )
                    ),
                    reservation=BudgetReservation(max_input_tokens=1000, max_output_tokens=1000),
                ),
            )
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return app, store, ledger, provider


def test_stream_deadline_fails_session_with_unknown_outcome(tmp_path):
    async def scenario():
        app, store, ledger, provider = _app(tmp_path, "stalled")
        try:
            outcome = await run_to_completion(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="stalled",
                    messages=[Message.text("user", "answer")],
                ),
            )
            # The run ends in a stored session.failed instead of raising.
            assert outcome.status is SessionStatus.FAILED
            assert outcome.error_detail is None
            assert outcome.error == "Model provider stream exceeded its semantic_idle deadline."
            assert provider.calls == 1

            session = await store.load("stalled")
            assert session.status is SessionStatus.FAILED
            events = await store.load_events("stalled")
            types = [event.type for event in events]
            assert types.count(EventType.MODEL_ERROR) == 1
            assert EventType.MODEL_RETRY not in types
            assert EventType.INTERACTION_FAILED in types
            [failed] = [event for event in events if event.type is EventType.SESSION_FAILED]
            assert failed.payload["error_type"] == "ModelStreamDeadlineError"
            assert failed.payload["provider_error_code"] == "provider_stream_semantic_idle_timeout"
            assert failed.payload["provider_deadline_kind"] == "semantic_idle"
            assert failed.payload["provider_effect_outcome"] == "unknown"
            assert "A partial answer" not in repr(failed.payload)

            [stage] = provider.stages
            assert await store.load_active_model_completion_stage("stalled") is None
            settlement = await store.load_model_completion_stage_settlement(
                "stalled", stage.stage_id
            )
            assert settlement.disposition is (
                ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN
            )
            assert settlement.reason_code == "provider_stream_deadline"
            # The reserved budget is charged in full as unknown usage.
            charged = [await ledger.load_reservation(rid) for rid in stage.reservation_ids]
            assert charged and all(record.status == "reconciled" for record in charged)
            assert all(record.reason == UNKNOWN_POST_DISPATCH_BUDGET_REASON for record in charged)
            assert all(record.actual_amount == record.reserved_amount for record in charged)

            # The failed conversation can continue; that is a new request, not a replay.
            resumed = await run_to_completion(
                app,
                ResumeRequest(session_id="stalled", messages=[Message.text("user", "Again")]),
            )
            assert resumed.status is SessionStatus.COMPLETED
            assert provider.calls == 2
        finally:
            await store.close()
            await ledger.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))


def test_stream_deadline_fails_the_linked_task(tmp_path):
    async def scenario():
        tasks = InMemoryTaskStore()
        app, store, ledger, provider = _app(tmp_path, "stalled-task", task_store=tasks)
        try:
            await tasks.create_task(TaskCreate(task_id="answer", type="answer"))
            task = await tasks.claim_task("worker-1")
            outcome = await run_to_completion(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="stalled-task",
                    messages=[Message.text("user", "answer")],
                    task_id=task.id,
                    task_worker_id="worker-1",
                    task_lease_expires_at=task.lease_expires_at,
                ),
            )
            assert outcome.status is SessionStatus.FAILED
            assert (await store.load("stalled-task")).status is SessionStatus.FAILED
            assert (await tasks.load_task("answer")).status is TaskStatus.FAILED
            assert provider.calls == 1
        finally:
            await store.close()
            await ledger.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))


@pytest.mark.parametrize("read", ["stage", "operation"])
def test_eligibility_read_failure_preserves_stream_deadline(tmp_path, monkeypatch, read):
    from cayu.providers import ModelStreamDeadlineError
    from cayu.runtime import _session_outcome_publication as publication

    async def scenario():
        app, store, ledger, provider = _app(tmp_path, "read-failure")
        failure = RuntimeError("eligibility store unavailable")
        original = publication.SessionOutcomePublication._stream_deadline_settles_as_failure

        async def broken_read(*args, **kwargs):
            raise failure

        async def check(self, context, exc):
            # Install only at failure publication, after admission and dispatch.
            if read == "stage":
                monkeypatch.setattr(store, "load_active_model_completion_stage", broken_read)
            else:
                monkeypatch.setattr(publication, "load_recoverable_provider_operation", broken_read)
            return await original(self, context, exc)

        monkeypatch.setattr(
            publication.SessionOutcomePublication, "_stream_deadline_settles_as_failure", check
        )
        try:
            with pytest.raises(ModelStreamDeadlineError) as caught:
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="read-failure",
                        messages=[Message.text("user", "answer")],
                    )
                ):
                    pass
            assert isinstance(caught.value.__cause__, BaseExceptionGroup)
            assert failure in caught.value.__cause__.exceptions
            assert provider.calls == 1
        finally:
            await store.close()
            await ledger.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))
