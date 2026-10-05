"""Native pause continuation must retain an external wait's execution controls."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from tests.core.test_approval_lifecycle_execution_identities import (
    _RecordingTool,
    _RequireApprovalPolicy,
)
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import AgentSpec, CayuApp, Message, RunRequest, ScriptedModelProvider
from cayu.approvals.tools import (
    PendingToolApprovalEventView,
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from cayu.approvals.user_input import UserInputResponse
from cayu.budgets import BudgetLimit
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.events import EventType
from cayu.external_wait_host import ExternalWaitHost
from cayu.external_waits import ExternalEventWaits
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions.external_waits import ExternalEventDelivery, ExternalWaitUnavailable
from cayu.tools.user_input import UserInputTool


class RecordingTool(_RecordingTool):
    spec = _RecordingTool.spec.model_copy(
        update={
            "execution_profile_identity": ExecutionProfileBehaviorIdentity(
                name="tests:external-approval-tool",
                behavior_version="1",
                implementation_version="1",
            )
        }
    )


class ApprovalPolicy(_RequireApprovalPolicy):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:external-approval-policy",
            behavior_version="1",
            implementation_version="1",
        )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("pause", ["approval", "user_input"])
def test_external_event_after_native_pause_restart_preserves_budget(
    backend, pause, tmp_path, request
):
    _native_pause_scenario(backend, pause, tmp_path, request)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("pause", ["approval", "user_input"])
def test_host_preserves_native_pause_until_resolution(backend, pause, tmp_path, request):
    _native_pause_scenario(backend, pause, tmp_path, request, poll=True)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("pause", ["approval", "user_input"])
@pytest.mark.parametrize("control", ["cancel", "retire"])
def test_external_control_can_retire_native_pause(backend, pause, control, tmp_path, request):
    _native_pause_scenario(backend, pause, tmp_path, request, control=control)


class EagerRecoveryAdapter(SessionExternalWaitAdapter):
    async def recover_to_wait(self, registration, *, context, inactive_for_seconds=0):
        # Exercise recovery admission without a wall-clock sleep. Native writer
        # leases and all checkpoint/claim checks remain in force on every store.
        return await super().recover_to_wait(
            registration, context=context, inactive_for_seconds=inactive_for_seconds
        )


def _native_pause_scenario(backend, pause, tmp_path, request, *, poll=False, control=None):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            tool = RecordingTool()
            provider = ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="submit",
                            name=tool.spec.name if pause == "approval" else "ask_user",
                            arguments={"value": "job"}
                            if pause == "approval"
                            else {"question": "Submit the job?"},
                        ),
                        ModelStreamEvent.completed(
                            {
                                "finish_reason": "tool_calls",
                                "usage": {"input_tokens": 1, "output_tokens": 1},
                            }
                        ),
                    ],
                    *[
                        [
                            ModelStreamEvent.text_delta(text),
                            ModelStreamEvent.completed(
                                {
                                    "finish_reason": "stop",
                                    "usage": {"input_tokens": 1, "output_tokens": 1},
                                }
                            ),
                        ]
                        for text in ("Submitted", "Result received")
                    ],
                ]
            )
            apps, owners = [], []

            def application(native):
                app = CayuApp(session_store=native, enable_logging=False)
                app.register_provider(provider, default=True)
                app.register_agent(
                    AgentSpec(name="root", model="model"),
                    tools=[tool] if pause == "approval" else [UserInputTool()],
                    tool_policy=ApprovalPolicy() if pause == "approval" else None,
                )
                waits = ExternalEventWaits(store=native, access_policy=Policy())
                apps.append(app)
                owners.append(waits)
                return app, waits, SessionExternalWaitAdapter(app, waits)

            try:
                app, waits, adapter = application(store)
                correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
                registered = registration(correlation)
                await waits.register(registered, context=CONTEXT)
                session_id = "external-approval-" + uuid4().hex
                with pytest.raises(ExternalWaitUnavailable, match="wait-boundary recovery"):
                    await adapter.run_to_wait(
                        RunRequest(
                            agent_name="root",
                            session_id=session_id,
                            messages=[Message.text("user", "Submit the external job")],
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
                        ),
                        registered,
                        context=CONTEXT,
                    )
                assert not tool.calls and len(provider.requests) == 1
                events = await store.load_events(session_id)
                await app.aclose()
                await waits.aclose()

                restored, restored_waits, _ = application(reopen())
                pending_adapter = EagerRecoveryAdapter(restored, restored_waits)
                if poll or control is not None:
                    if control == "cancel":
                        await restored_waits.cancel(
                            correlation, operation_key="cancel-paused", context=CONTEXT
                        )
                    else:
                        await restored_waits.deliver(
                            ExternalEventDelivery(
                                correlation=correlation,
                                delivery_id="result",
                                payload_json='{"ok":true}',
                            ),
                            context=CONTEXT,
                        )
                    host = ExternalWaitHost(pending_adapter, context=CONTEXT)
                    if control is not None:
                        if control == "retire":
                            await pending_adapter.retire_execution(
                                registered, operation_key="retire-paused", context=CONTEXT
                            )
                        await host.service_once(scope=correlation.request.scope, source="renderer")
                        terminal = await restored_waits.inspect(correlation, context=CONTEXT)
                        assert terminal.handoff == "excluded" and not terminal.pending_handoff
                        assert terminal.outcome.kind == (
                            "cancelled" if control == "cancel" else "event"
                        )
                        assert not tool.calls and len(provider.requests) == 1
                        return
                    native = restored.session_store
                    before = await native.load(session_id)
                    checkpoint = await native.load_checkpoint(session_id)
                    before_events = await native.load_events(session_id)
                    for _ in range(2):
                        page = await host.service_once(
                            scope=correlation.request.scope, source="renderer"
                        )
                        assert page.pending == (correlation.request.correlation_key,)
                        assert not page.settled and not page.failures
                        assert await native.load(session_id) == before
                        assert await native.load_checkpoint(session_id) == checkpoint
                        assert await native.load_events(session_id) == before_events
                    with pytest.raises(ExternalWaitUnavailable):
                        await pending_adapter.recover_to_wait(registered, context=CONTEXT)
                    assert await native.load(session_id) == before
                    assert await native.load_checkpoint(session_id) == checkpoint
                    assert not tool.calls and len(provider.requests) == 1
                if pause == "approval":
                    approval = PendingToolApprovalEventView.from_event(
                        next(
                            event
                            for event in events
                            if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
                        )
                    )
                    stream = restored.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id=session_id,
                            approval_id=approval.approval_id,
                            tool_round_id=approval.tool_round_id,
                            tool_call_id=approval.tool_call_id,
                            decision=ToolApprovalDecision.APPROVE,
                        )
                    )
                else:
                    awaiting = next(
                        event
                        for event in events
                        if event.type == EventType.SESSION_AWAITING_USER_INPUT
                    )
                    stream = restored.resolve_user_input(
                        UserInputResponse(
                            session_id=session_id,
                            input_id=awaiting.payload["input_id"],
                            answer="yes",
                        )
                    )
                resolved = [event async for event in stream]
                assert any(event.type == EventType.SESSION_COMPLETED for event in resolved)
                expected_calls = [{"value": "job"}] if pause == "approval" else []
                assert tool.calls == expected_calls and len(provider.requests) == 2
                await restored.aclose()
                await restored_waits.aclose()

                _, final_waits, final_adapter = application(reopen())
                await final_waits.deliver(
                    ExternalEventDelivery(
                        correlation=correlation, delivery_id="result", payload_json='{"ok":true}'
                    ),
                    context=CONTEXT,
                )
                await final_adapter.recover_to_wait(
                    registered, context=CONTEXT, inactive_for_seconds=0
                )
                assert len(provider.requests) == 2
                settled = await final_adapter.service_wait(registered, context=CONTEXT)
                assert settled.wait.handoff == "settled"
                assert len(provider.requests) == 3 and tool.calls == expected_calls
                assert await final_adapter.service_wait(registered, context=CONTEXT) == settled
                assert len(provider.requests) == 3
            finally:
                for app in apps:
                    await app.aclose()
                for waits in owners:
                    await waits.aclose()

    asyncio.run(scenario())
