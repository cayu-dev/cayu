"""Public continuation entrances preserve the selected model and root authority."""

from __future__ import annotations

import asyncio
import contextlib
from decimal import Decimal

import pytest
from examples.runtime_auxiliary_inference import MODEL, Summarize
from tests.core.test_model_failover_stages import _StageMemoryStore, _StageSQLiteStore

from cayu import (
    AgentSpec,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    EnqueueSessionMessageRequest,
    EventType,
    Message,
    ModelFailoverPolicy,
    ModelPrice,
    ModelTarget,
    PriceBook,
    RunRequest,
    ScriptedModelProvider,
    SessionMessageDeliveryMode,
    TextPart,
)
from cayu.providers.base import ModelProviderError, ModelStreamEvent
from cayu.runtime.retry_policy import RetryPolicy
from cayu.storage.budget_ledger import SQLiteBudgetLedger


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_queued_interaction_keeps_selected_fallback_and_rebinds_exact_context(
    monkeypatch, tmp_path, backend
):

    async def scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "queued-fallback.sqlite")
        )
        entered, release = asyncio.Event(), asyncio.Event()

        class Backup(ScriptedModelProvider):
            async def stream(self, request):
                self.requests.append(request)
                if len(self.requests) == 1:
                    entered.set()
                    await release.wait()
                yield ModelStreamEvent.text_delta("completed")
                yield ModelStreamEvent.completed()

        def unavailable(_request):
            raise ModelProviderError(
                "unavailable", provider="primary", status_code=503, retryable=True
            )

        primary = ScriptedModelProvider(name="primary", response_factory=unavailable)
        backup = Backup([[ModelStreamEvent.completed()]], name="backup")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="queued-fallback",
                        messages=[Message.text("user", "first")],
                        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),),
                            max_total_attempts=2,
                        ),
                    )
                )
            ]

        task = asyncio.create_task(run())
        barrier = asyncio.create_task(entered.wait())
        try:
            done, _ = await asyncio.wait(
                {task, barrier}, timeout=30, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                pytest.fail(repr(await task))
            assert barrier in done
            before = await store.load_checkpoint("queued-fallback")
            assert before is not None
            initial_progress = before["model_failover"]
            assert initial_progress["candidate_index"] == 1
            request = EnqueueSessionMessageRequest(
                session_id="queued-fallback",
                idempotency_key="second",
                content="second",
                delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
            )
            accepted = await app.enqueue_session_message(request)
            assert not accepted.replayed
            release.set()
            events = await asyncio.wait_for(task, timeout=30)
            assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            assert len(primary.requests) == 1 and len(backup.requests) == 2
            assert all(request.model == "large" for request in backup.requests)
            assert any(
                isinstance(part, TextPart) and part.text == "second"
                for message in backup.requests[-1].messages
                for part in message.content
            )
            checkpoint = await store.load_checkpoint("queued-fallback")
            assert checkpoint is not None
            progress = checkpoint["model_failover"]
            assert progress["candidate_index"] == 1
            assert progress["interaction_id"] != initial_progress["interaction_id"]
            assert progress["source_run_epoch"] == initial_progress["source_run_epoch"]
            assert progress["logical_step_id"] != initial_progress["logical_step_id"]
            assert progress["attempts_used"] == progress["candidate_attempt"] == 1
            assert (
                progress["execution_profile_fingerprint"]
                == initial_progress["execution_profile_fingerprint"]
            )
            session = await store.load("queued-fallback")
            assert (
                session is not None
                and session.provider_name == "primary"
                and session.model == "small"
            )
            assert await store.load_active_model_completion_stage("queued-fallback") is None
            assert (await app.enqueue_session_message(request)).replayed
            assert len(primary.requests) == 1 and len(backup.requests) == 2
        finally:
            release.set()
            for owned in (task, barrier):
                if not owned.done():
                    owned.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await owned
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("cap", [None, "15.456", "15.457", "15.458"])
def test_tool_auxiliary_dispatch_keeps_root_target_while_parent_stays_on_fallback(
    monkeypatch, tmp_path, backend, cap
):

    async def scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "auxiliary-fallback.sqlite")
        )
        primary_calls = 0

        def primary_response(request):
            nonlocal primary_calls
            primary_calls += 1
            assert request.model == MODEL
            if primary_calls == 1:
                raise ModelProviderError(
                    "unavailable", provider="primary", status_code=503, retryable=True
                )
            assert primary_calls == 2
            return [
                ModelStreamEvent.text_delta("Short summary"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 5, "output_tokens": 2}}),
            ]

        primary = ScriptedModelProvider(name="primary", response_factory=primary_response)
        backup = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        name="summarize", id="summary-call", arguments={"text": "Long input"}
                    ),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 2, "output_tokens": 1}}),
                ],
                [
                    ModelStreamEvent.text_delta("Finished"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 2, "output_tokens": 1}}),
                ],
            ],
            name="backup",
        )
        ledger = (
            SQLiteBudgetLedger(tmp_path / "budgets.sqlite")
            if backend == "sqlite" and cap is not None
            else None
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            budget_ledger=ledger,
            budget_policy=None
            if cap is None
            else BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal(cap),
                        pricing=PriceBook(
                            prices=(
                                ModelPrice.fixed(
                                    provider_name="primary",
                                    model=MODEL,
                                    input_per_million=Decimal("1000"),
                                    output_per_million=Decimal("1000"),
                                ),
                                ModelPrice.fixed(
                                    provider_name="backup",
                                    model="backup-model",
                                    input_per_million=Decimal("100000"),
                                    output_per_million=Decimal("100000"),
                                ),
                            )
                        ),
                        # Must cover the reference tool's admitted 100/50 limits.
                        reservation=BudgetReservation(max_input_tokens=100, max_output_tokens=50),
                    ),
                )
            ),
        )
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model=MODEL), tools=[Summarize()])
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="auxiliary-fallback",
                        messages=[Message.text("user", "Please summarize the input")],
                        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="backup-model"),)
                        ),
                    )
                )
            ]
            blocked = cap == "15.456"
            budget_evidence = [
                (
                    str(event.type),
                    {
                        key: value
                        for key, value in event.payload.items()
                        if key
                        in {
                            "actual",
                            "maximum",
                            "requested",
                            "message",
                            "reason",
                            "status",
                            "amount",
                            "reserved_amount",
                            "actual_amount",
                            "error",
                            "auxiliary_outcome",
                        }
                    },
                )
                for event in await store.load_events("auxiliary-fallback")
                if str(event.type).startswith(("budget.", "tool.call.", "model.auxiliary"))
            ]
            if events[-1].type is not (
                EventType.SESSION_INTERRUPTED if blocked else EventType.SESSION_COMPLETED
            ):
                pytest.fail(repr(budget_evidence))
            assert len(primary.requests) == 2, budget_evidence
            assert len(backup.requests) == (1 if blocked else 2)
            assert [request.model for request in primary.requests] == [MODEL, MODEL]
            assert all(request.model == "backup-model" for request in backup.requests)
            [auxiliary] = [
                event for event in events if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            ]
            assert auxiliary.payload["auxiliary_outcome"] == "completed"
            assert auxiliary.payload["provider_name"] == "primary"
            assert auxiliary.payload["requested_model"] == MODEL
            usage = await app.get_session_usage("auxiliary-fallback")
            assert usage.usage.total_tokens == (10 if blocked else 13)
            assert usage.model_steps == (1 if blocked else 2)
            checkpoint = await store.load_checkpoint("auxiliary-fallback")
            assert checkpoint is not None and checkpoint["model_failover"]["candidate_index"] == 1
            assert checkpoint["model_failover"]["attempts_used"] == (2 if blocked else 1)
            assert await store.load_active_model_completion_stage("auxiliary-fallback") is None
            if cap is not None:
                # Failed A owns 0.150 conservatively; first B costs 0.300 and
                # auxiliary A costs 0.007. B's next reservation is 15.000.
                # Admission must combine all three sources once at 15.457.
                reservations = [
                    event
                    for event in await store.load_events("auxiliary-fallback")
                    if event.type is EventType.BUDGET_RESERVED
                ]
                assert len(reservations) == (3 if blocked else 4)
                settled = []
                for event in reservations:
                    record = await app.budget_ledger.load_reservation(
                        event.payload["reservation_id"]
                    )
                    assert record is not None and record.status == "reconciled"
                    assert record.dispatch_id is not None
                    settled.append(
                        (
                            record.provider_name,
                            record.model,
                            record.reserved_amount,
                            record.actual_amount,
                        )
                    )
                expected = [
                    ("primary", MODEL, Decimal("0.150"), Decimal("0.150")),
                    ("backup", "backup-model", Decimal("15"), Decimal("0.300")),
                    ("primary", MODEL, Decimal("0.150"), Decimal("0.007")),
                ]
                if not blocked:
                    expected.append(("backup", "backup-model", Decimal("15"), Decimal("0.300")))
                assert settled == expected
                policy = app.budget_policy
                assert policy is not None
                reported = await app.get_session_cost(
                    "auxiliary-fallback", policy.limits[0].pricing
                )
                assert reported.total_cost == Decimal("0.307" if blocked else "0.607")
                assert (
                    reported.auxiliary_attempts == 1 and reported.unpriced_auxiliary_attempts == 0
                )
        finally:
            if ledger is not None:
                await ledger.close()
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())
