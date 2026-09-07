from __future__ import annotations

import asyncio
from datetime import timedelta
from decimal import Decimal

import pytest
from tests.core.test_explicit_session_compaction import _create_profiled_session

from cayu import (
    AgentSpec,
    CayuApp,
    CheckpointCompactionContextPolicy,
    Event,
    EventType,
    Message,
    ModelCompactor,
    RunRequest,
)
from cayu.providers import ModelProvider, ModelProviderError
from cayu.runtime import BudgetLimit, CompactSessionRequest
from cayu.runtime.budgets import budget_check_from_events, budget_check_from_totals
from cayu.runtime.context import ContextBuildError
from cayu.runtime.costs import ModelPrice, PriceBook, estimate_session_cost
from cayu.runtime.sessions import EventQuery, SessionIdentity, SessionStatus, UsageRollupQuery
from cayu.storage import SQLiteSessionStore


class FailingProvider(ModelProvider):
    name = "usage-test"

    async def stream(self, request):
        raise ModelProviderError(
            "SECRET provider detail", provider=self.name, status_code=503, retryable=False
        )
        yield


def pricing():
    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="usage-test",
                model="summary",
                match="exact",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        )
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_failed_compaction_usage_classification_survives_reload(tmp_path, explicit):
    async def run():
        path = tmp_path / "usage.sqlite"
        store = SQLiteSessionStore(path)
        provider = FailingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="summary"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary"),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )
        messages = [
            Message.text("user", "old"),
            Message.text("assistant", "answer"),
            Message.text("user", "new"),
        ]
        if explicit:
            session = await _create_profiled_session(
                app,
                store,
                RunRequest(agent_name="assistant", session_id="usage", messages=[]),
                identity=SessionIdentity(provider_name=provider.name, model="summary"),
            )
            await store.append_transcript_messages(session.id, messages)
            session = await store.update_status(session.id, SessionStatus.COMPLETED)
            events = []
            with pytest.raises(ContextBuildError):
                async for e in app.compact_session(
                    CompactSessionRequest(
                        session_id="usage",
                        idempotency_key="compact",
                        expected_run_epoch=session.run_epoch,
                        expected_transcript_cursor=3,
                    )
                ):
                    events.append(e)
        else:
            events = [
                e
                async for e in app.run(
                    RunRequest(agent_name="assistant", session_id="usage", messages=messages)
                )
            ]
        completed = [e for e in events if e.type == EventType.MODEL_COMPLETED]
        assert len(completed) == 1
        assert completed[0].payload["compaction_outcome"] == "provider_error"
        limit = BudgetLimit(max_estimated_cost=Decimal("10"), pricing=pricing())
        check = budget_check_from_events(limit=limit, events=completed)
        assert check.limit_reached
        assert "missing or invalid completion usage" in check.message
        assert "no matching pricing" not in check.message
        assert "SECRET" not in check.message
        first = await store.read_cost_accounting(EventQuery(session_id="usage"), pricing())
        assert first.totals.missing_usage_model_steps == 1
        await store.close()
        store = SQLiteSessionStore(path)
        try:
            restored = await store.read_cost_accounting(
                EventQuery(session_id="usage"), pricing(), previous=first
            )
            assert restored.totals == first.totals
            from cayu.runtime.aggregates import UsageCostRollup, estimate_usage_rollup_cost

            timestamp = completed[0].timestamp
            rollup = await store.aggregate_usage(
                UsageRollupQuery(
                    start_at=timestamp - timedelta(seconds=1),
                    end_at=timestamp + timedelta(seconds=1),
                    include_pricing_inputs=True,
                )
            )
            grouped = estimate_usage_rollup_cost(rollup, pricing())
            grouped = UsageCostRollup.model_validate_json(grouped.model_dump_json())
            assert grouped.unpriced_reasons[0].unpriced_reason == "missing_usage"
            assert grouped.unpriced_reasons[0].model_steps == 1

            assert (
                budget_check_from_totals(limit=limit, summary=restored.totals).message
                == check.message
            )
            # A later valid completion refreshes totals without erasing the missing usage.
            await store.append_events(
                "usage",
                [
                    Event(
                        type=EventType.MODEL_COMPLETED,
                        session_id="usage",
                        payload={
                            "provider_name": provider.name,
                            "model": "summary",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        },
                    )
                ],
            )
            refreshed = await store.read_cost_accounting(
                EventQuery(session_id="usage"), pricing(), previous=restored
            )
            assert refreshed.totals.missing_usage_model_steps == 1
            assert refreshed.totals.priced_model_steps == 1
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("payload", "prices", "category", "message"),
    [
        (
            {"usage_metrics": {"input_tokens": -1}},
            pricing(),
            "missing_usage",
            "missing or invalid completion usage",
        ),
        (
            {
                "provider_name": "usage-test",
                "model": "summary",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            PriceBook(
                prices=(
                    ModelPrice.fixed(
                        provider_name="other",
                        model="other",
                        input_per_million=Decimal("1"),
                        output_per_million=Decimal("1"),
                    ),
                )
            ),
            "missing_pricing",
            "no matching pricing",
        ),
        (
            {
                "provider_name": "usage-test",
                "model": "summary",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            pricing(),
            "unsupported_pricing",
            "unsupported pricing inputs",
        ),
    ],
)
def test_bounded_unpriced_categories(payload, prices, category, message):
    events = [Event(type=EventType.MODEL_COMPLETED, session_id="usage", payload=payload)]
    currency = "EUR" if category == "unsupported_pricing" else "USD"
    summary = estimate_session_cost(
        session_id="usage", events=events, pricing=prices, currency=currency
    )
    assert summary.line_items[0].unpriced_reason == category
    assert getattr(summary, category + "_model_steps") == 1
    check = budget_check_from_totals(
        limit=BudgetLimit(max_estimated_cost=Decimal("10"), pricing=prices, currency=currency),
        summary=summary,
    )
    assert check.limit_reached
    assert message in check.message
