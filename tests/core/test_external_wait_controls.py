"""The external receiver restores original controls at real continuation admission."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import AgentSpec, CayuApp, Message, RunRequest, ScriptedModelProvider
from cayu.budgets import BudgetLimit
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.external_waits import ExternalEventWaits
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.stop_policy import RunLimits
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions.external_waits import ExternalEventDelivery


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_restarted_event_service_preserves_request_controls(
    backend, tmp_path, request, monkeypatch
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            provider = ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.completed(
                            {
                                "finish_reason": "stop",
                                "usage": {"input_tokens": 1, "output_tokens": 1},
                            }
                        )
                    ]
                    for _ in range(2)
                ]
            )

            def application(native):
                app = CayuApp(session_store=native, enable_logging=False)
                app.register_provider(provider, default=True)
                app.register_agent(AgentSpec(name="root", model="model"))
                return app

            app = application(store)
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            original = RunRequest(
                agent_name="root",
                session_id="controls-" + uuid4().hex,
                messages=[Message.text("user", "submit")],
                max_steps=7,
                limits=RunLimits(scope="session", max_total_tokens=12, max_tool_calls=2),
                budget_limits=(
                    BudgetLimit(
                        scope="session",
                        max_estimated_cost=Decimal("0.001"),
                        pricing=PriceBook(
                            prices=(
                                ModelPrice.fixed(
                                    provider_name="scripted",
                                    model="model",
                                    input_per_million=Decimal("1"),
                                    output_per_million=Decimal("1"),
                                ),
                            )
                        ),
                    ),
                ),
                metadata={"job": "job-controls"},
            )
            await SessionExternalWaitAdapter(app, waits).run_to_wait(
                original, registered, context=CONTEXT
            )
            await app.aclose()
            await waits.aclose()
            restored_store = reopen()
            restored = application(restored_store)
            restored_waits = ExternalEventWaits(store=restored_store, access_policy=Policy())
            await restored_waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="done", payload_json="{}"
                ),
                context=CONTEXT,
            )
            observed = []
            resume = restored._resume_private

            async def observe_resume(candidate, **kwargs):
                observed.append(candidate)
                async for event in resume(candidate, **kwargs):
                    yield event

            monkeypatch.setattr(restored, "_resume_private", observe_resume)
            adapter = SessionExternalWaitAdapter(restored, restored_waits)
            settled = await adapter.service_wait(registered, context=CONTEXT)
            assert settled.wait.handoff == "settled" and len(provider.requests) == 2
            assert len(observed) == 1
            admitted = observed[0]
            assert admitted.max_steps == original.max_steps
            assert admitted.limits == original.limits
            assert admitted.budget_limits == original.budget_limits
            assert admitted.metadata == original.metadata
            assert await adapter.service_wait(registered, context=CONTEXT) == settled
            assert len(observed) == 1 and len(provider.requests) == 2
            await restored.aclose()
            await restored_waits.aclose()

    asyncio.run(scenario())
