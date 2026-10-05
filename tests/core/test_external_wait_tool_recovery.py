"""External waits retain partial tool results and use the native effect receiver."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import SecretStr
from tests.core.test_tool_effect_reconciliation_registration import _spec
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.evals.testing import ScriptedModelProvider
from cayu.external_wait_host import ExternalWaitHost
from cayu.external_waits import ExternalEventWaits
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions.external_waits import ExternalEventDelivery
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_partial_tool_round_requires_exact_effect_reconciliation(backend, tmp_path, request):
    async def scenario():
        codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr("A" * 43)})
        )
        async with stores(
            backend, tmp_path, request, [datetime.now(UTC)], public_authority_alias_codec=codec
        ) as (store, reopen):
            completed, dispatched = asyncio.Event(), asyncio.Event()
            calls, lookups = [], []

            class First(Tool):
                spec = ToolSpec(
                    name="first",
                    effect=ToolEffect.NONE,
                    parallel_safe=False,
                    execution_profile_identity=ExecutionProfileBehaviorIdentity(
                        name="test:external-wait-first",
                        behavior_version="1",
                        implementation_version="1",
                    ),
                )

                async def run(self, ctx, args):
                    calls.append("first")
                    completed.set()
                    return ToolResult(content="first result")

            class External(Tool):
                spec = ToolSpec(
                    name="external",
                    effect=ToolEffect.EXTERNAL,
                    parallel_safe=False,
                    execution_profile_identity=ExecutionProfileBehaviorIdentity(
                        name="test:external-wait-effect",
                        behavior_version="1",
                        implementation_version="1",
                    ),
                )

                async def run(self, ctx, args):
                    await completed.wait()
                    calls.append("external")
                    dispatched.set()
                    await asyncio.Event().wait()

            class Reconciler:
                async def reconcile(self, *, context, receipt):
                    lookups.append(context.idempotency_key)
                    return ToolEffectReconciliationResult(
                        outcome="completed",
                        observation="sent",
                        receipt=ToolEffectReceipt(
                            receipt_id="job-receipt",
                            receipt_schema="deployment",
                            receipt_schema_version=1,
                            tool_call_id=context.tool_call_id,
                            tool_name=context.tool_name,
                            idempotency_key=context.idempotency_key,
                            outcome="completed",
                            message="external result",
                            source="reconciler",
                            observed_at=datetime(2026, 10, 3, tzinfo=UTC),
                        ),
                    )

            provider = ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(id="first", name="first", arguments={}),
                        ModelStreamEvent.tool_call(id="external", name="external", arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("Submitted"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("Received"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            )

            def application(native):
                app = CayuApp(session_store=native, enable_logging=False)
                app.register_provider(provider, default=True)
                app.register_agent(
                    AgentSpec(name="root", model="model"),
                    tools=[First(), External()],
                    tool_effect_reconcilers={
                        "external": ToolEffectReconciliationRegistration(
                            reconciler=Reconciler(), spec=_spec(supports_lookup=True)
                        )
                    },
                )
                return app

            app = application(store)
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            session_id = "partial-wait-" + uuid4().hex
            running = asyncio.create_task(
                SessionExternalWaitAdapter(app, waits).run_to_wait(
                    RunRequest(
                        agent_name="root",
                        session_id=session_id,
                        messages=[Message.text("user", "submit")],
                    ),
                    registered,
                    context=CONTEXT,
                )
            )
            try:
                await asyncio.wait_for(dispatched.wait(), 20)
            finally:
                running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            assert running.cancelled() and running.cancelling() == 1
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="done", payload_json='{"done":true}'
                ),
                context=CONTEXT,
            )
            await app.aclose()
            await waits.aclose()
            restored_store = reopen()
            restored_waits = ExternalEventWaits(store=restored_store, access_policy=Policy())
            restored = application(restored_store)
            adapter = SessionExternalWaitAdapter(restored, restored_waits)
            page = await ExternalWaitHost(adapter, context=CONTEXT).service_once(
                scope=correlation.request.scope, source="renderer"
            )
            assert page.pending == (correlation.request.correlation_key,)
            assert calls == ["first", "external"] and len(provider.requests) == 1
            events = await restored_store.load_events(session_id)
            first_results = [
                event
                for event in events
                if event.type.value == "tool.call.completed" and event.tool_name == "first"
            ]
            assert len(first_results) == 1
            start = next(
                e
                for e in events
                if e.type.value == "tool.call.started" and e.tool_name == "external"
            )
            target = await restored.inspect_tool_effect(
                session_id,
                tool_round_id=start.payload["tool_round_id"],
                tool_call_id=start.payload["tool_call_id"],
            )
            reconciliation = ToolEffectReconciliationRequest(**target.model_dump(), lookup=True)
            async for _ in restored.reconcile_tool_effect(reconciliation):
                pass
            await adapter.recover_to_wait(registered, context=CONTEXT, inactive_for_seconds=0)
            await adapter.service_wait(registered, context=CONTEXT)
            assert calls == ["first", "external"] and len(lookups) == 1
            assert len(provider.requests) == 3
            settled = await adapter.service_wait(registered, context=CONTEXT)
            assert settled.wait.handoff == "settled"
            assert len(provider.requests) == 3
            async for _ in restored.reconcile_tool_effect(reconciliation):
                pass
            assert len(provider.requests) == 3 and len(lookups) == 1
            final_events = await restored_store.load_events(session_id)
            for name in ("first", "external"):
                assert (
                    sum(
                        event.type.value == "tool.call.completed" and event.tool_name == name
                        for event in final_events
                    )
                    == 1
                )
            await restored.aclose()
            await restored_waits.aclose()

    asyncio.run(scenario())
