from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from tests.core.test_cost_accounting import _priced_event, _pricing
from tests.core.test_run_limits import _controller

from cayu.core import Event, EventType, Message
from cayu.runtime.budgets import BudgetLimit
from cayu.runtime.sessions import EventQuery, InMemorySessionStore, RunRequest, SessionIdentity
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.migrations import SchemaMode


def _store(backend, tmp_path, request):
    if backend == "memory":
        return InMemorySessionStore()
    if backend == "sqlite":
        return SQLiteSessionStore(tmp_path / "review.sqlite")
    return PostgresSessionStore(
        request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
    )


async def _session(store, *, causal_budget_id=None):
    return await store.create(
        RunRequest(
            session_id=f"review-{uuid4()}",
            agent_name="assistant",
            causal_budget_id=causal_budget_id,
            messages=[Message.text("user", "hi")],
        ),
        identity=SessionIdentity(provider_name="openai", model="gpt-test"),
    )


def test_postgres_cross_session_costs_include_visible_spend_and_late_lower_sequence(postgres_dsn):
    import psycopg

    from cayu.runtime._run_limits import SessionUsageTracker

    async def run():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            causal_id = f"causal-{uuid4()}"
            earlier, later = (
                await _session(store, causal_budget_id=causal_id),
                await _session(store, causal_budget_id=causal_id),
            )
            query = EventQuery(session_ids=(earlier.id, later.id))
            tracker = SessionUsageTracker(store, session_id=later.id)
            await tracker.retain_cost_scopes("review", {"global"})

            async def read(previous):
                return await store.read_cost_accounting(query, _pricing(), previous=previous)

            async with await psycopg.AsyncConnection.connect(postgres_dsn) as held:
                async with held.cursor() as cur:
                    await cur.execute("SELECT pg_current_xact_id()")
                    await store._append_events_with_cursor(
                        cur, earlier.id, [_priced_event(earlier.id)], expected_run_epoch=None
                    )
                    # Commit a later sequence while the earlier insertion is still invisible.
                    await store.append_events(
                        later.id,
                        [
                            _priced_event(later.id).model_copy(update={"agent_name": "assistant"}),
                            Event(type=EventType.MODEL_COMPLETED, session_id=later.id, payload={}),
                        ],
                    )
                    scoped = await store.read_cost_accounting(
                        EventQuery(session_id=later.id), _pricing()
                    )
                    first = await tracker.cost_snapshot("review", "global", read)
                    assert tracker._cost_snapshots == {}
                    assert first.totals.model_steps == 2
                    assert first.totals.unpriced_model_steps == 1
                    assert first.totals.total_cost == scoped.totals.total_cost > 0
                    assert first.cursor is None
                    usage = await store.read_usage_accounting(query)
                    assert usage.summary.model_steps == 2
                    for scope, key in [
                        ("app", None),
                        ("agent", "assistant"),
                        ("causal", causal_id),
                    ]:
                        check = await _controller(
                            store, clock=lambda: datetime.now(UTC)
                        ).evaluate_operation_budgets(
                            session=later,
                            budget_limits=(
                                BudgetLimit(
                                    scope=scope,
                                    key=key,
                                    pricing=_pricing(),
                                    max_estimated_cost=Decimal("0.000001"),
                                ),
                            ),
                            operation_events=[],
                            operation_model_step_id="step",
                            provider_name="openai",
                            model="gpt-test",
                        )
                        assert check[0].check.limit_reached
                await held.commit()
            refreshed = await tracker.cost_snapshot("review", "global", read)
            assert tracker._cost_snapshots == {}
            assert refreshed.totals.model_steps == 3
            assert refreshed.totals.unpriced_model_steps == 1
            assert refreshed.totals.total_cost == first.totals.total_cost * 2
            # The newly visible insertion is below the previous maximum; a
            # sequence-only incremental baseline would permanently lose it.
            assert refreshed.through_sequence == first.through_sequence
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "scope, operation_kind", [("session", "explicit"), ("run", "explicit"), ("run", "automatic")]
)
def test_operation_budget_history_is_durable_and_only_unresolved_tail_is_overlaid(
    backend, scope, operation_kind, tmp_path, request, monkeypatch
):
    async def run():
        store = _store(backend, tmp_path, request)
        try:
            session = await _session(store)
            events = []
            for index in range(300):
                event = _priced_event(session.id, attempt=str(index))
                event.payload.update(
                    model_step_id=f"child-{index}"
                    if operation_kind == "automatic"
                    else "current-step",
                    parent_model_step_id="current-parent",
                    attempt_id="current-attempt",
                )
                events.append(event)
            other_step = _priced_event(session.id)
            other_step.payload.update(
                model_step_id="other-step",
                parent_model_step_id="other-parent",
                attempt_id="current-attempt",
            )
            prior_attempt = _priced_event(session.id)
            prior_attempt.payload.update(
                model_step_id="current-step",
                parent_model_step_id="prior-parent",
                attempt_id="prior-attempt",
            )
            await store.append_events(session.id, [other_step, prior_attempt, *events])
            pending = _priced_event(session.id)
            pending.payload.update(
                model_step_id="current-step",
                parent_model_step_id="current-parent",
                attempt_id="current-attempt",
            )
            original = store.read_cost_accounting
            overlays = []

            async def bounded(query, pricing, **kwargs):
                overlays.append(len(kwargs.get("additional_events", ())))
                return await original(query, pricing, **kwargs)

            monkeypatch.setattr(store, "read_cost_accounting", bounded)
            controller = _controller(store, clock=lambda: datetime.now(UTC))
            limit = BudgetLimit(scope=scope, pricing=_pricing(), max_estimated_cost=Decimal("1"))
            checks = await controller.evaluate_operation_budgets(
                session=session,
                budget_limits=(limit,),
                operation_events=[*events, pending, pending],
                operation_model_step_id="current-step",
                operation_attempt_id="current-attempt",
                operation_parent_model_step_id="current-parent"
                if operation_kind == "automatic"
                else None,
                provider_name="openai",
                model="gpt-test",
            )
            assert overlays == [1]
            expected = await original(
                EventQuery(
                    session_id=session.id,
                    model_step_id="current-step"
                    if scope == "run" and operation_kind == "explicit"
                    else None,
                    parent_model_step_id="current-parent"
                    if scope == "run" and operation_kind == "automatic"
                    else None,
                    operation_attempt_id="current-attempt" if scope == "run" else None,
                ),
                _pricing(),
                additional_events=(pending,),
            )
            assert expected.totals.model_steps == (301 if scope == "run" else 303)
            assert checks[0].check.actual == expected.totals.total_cost
            await store.append_event(session.id, pending)
            again = await controller.evaluate_operation_budgets(
                session=session,
                budget_limits=(limit,),
                operation_events=[*events, pending],
                operation_model_step_id="current-step",
                operation_attempt_id="current-attempt",
                operation_parent_model_step_id="current-parent"
                if operation_kind == "automatic"
                else None,
                provider_name="openai",
                model="gpt-test",
            )
            assert overlays[-1] == 0
            assert again[0].check.actual == checks[0].check.actual
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("scope", ["session", "run"])
def test_explicit_compaction_budget_survives_more_than_256_provider_completions(scope, monkeypatch):
    from tests.core.test_explicit_session_compaction import (
        UsageCompactionProvider,
        _create_profiled_session,
    )

    from cayu.core import AgentSpec
    from cayu.runtime import (
        CayuApp,
        CheckpointCompactionContextPolicy,
        CompactSessionRequest,
        ModelCompactor,
        SessionStatus,
    )
    from cayu.runtime.costs import ModelPrice, PriceBook

    async def run():
        provider = UsageCompactionProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                    max_hierarchy_calls=2000,
                ),
                max_user_turns=1,
            ),
        )
        session = await _create_profiled_session(
            app,
            store,
            RunRequest(agent_name="assistant", session_id=f"long-{uuid4()}", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_transcript_messages(
            session.id, [Message.text("user", "x" * 260000), Message.text("user", "current")]
        )
        session = await store.update_status(session.id, SessionStatus.COMPLETED)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name=provider.name,
                    model="summary-model",
                    input_per_million=Decimal(1),
                    output_per_million=Decimal(1),
                ),
            )
        )
        command = CompactSessionRequest(
            session_id=session.id,
            idempotency_key="long-compaction",
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=2,
            budget_limits=(
                BudgetLimit(scope=scope, max_estimated_cost=Decimal(1), pricing=pricing),
            ),
        )
        original_cost = app._run_limit_controller._cost_for_budget
        observed_steps = []

        async def observe_cost(**kwargs):
            result = await original_cost(**kwargs)
            observed_steps.append(result.model_steps)
            return result

        monkeypatch.setattr(app._run_limit_controller, "_cost_for_budget", observe_cost)
        events = [event async for event in app.compact_session(command)]
        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert provider.calls > 256
        completions = [event for event in events if event.type == EventType.MODEL_COMPLETED]
        assert len(completions) == provider.calls
        assert observed_steps[-1] == provider.calls
        assert max(observed_steps) == provider.calls

    asyncio.run(run())
