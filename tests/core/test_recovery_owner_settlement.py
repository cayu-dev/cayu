"""Each recovered-invocation owner settles an exit once, through the run settlement.

Manual user-input recovery, the manual tool-round supervisor, tool-effect
reconciliation and a provider-operation fallback each admit an invocation and
own its exit. A cancellation or a closed stream once the session runs again must
leave it INTERRUPTED with exactly one terminal event and one terminal hook,
decided by that owner's ledger.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from tests.core._recovery_settlement_observers import InterruptionHook, observe_decisions
from tests.core.test_provider_operation_offline_recovery import (
    _OfflineOperationProvider,
    _prepare_explicit_fallback_resolution,
)
from tests.core.test_tool_effect_reconciliation_registration import _spec
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_tool_support import PureRecordingTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalRecoveryOutcome
from cayu.approvals.user_input import UserInputRecoveryRequest, UserInputResponse
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers import ModelStreamEvent
from cayu.providers.operations import ProviderOperationStatus
from cayu.runtime._tool_effect_state import ToolEffectRecord
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tools.base import Tool, ToolEffect, ToolSpec
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.tools.user_input import UserInputTool

_SESSION = "recovered-owner"


class _Store(InMemorySessionStore):
    """Cancel the n-th checkpoint read once armed."""

    invocation_lifecycle_command_version = 1
    # The overrides delegate, so the store's own protocols still apply.
    supports_atomic_model_completion_stage_release = True

    def __init__(self) -> None:
        super().__init__()
        self.nth = 0
        self.reads: int | None = None
        self.injected = False
        self.fail_tool_completion = False
        self.effect_keys: list[str] = []

    async def load_checkpoint(self, session_id):
        if self.reads is not None:
            self.reads += 1
            if self.reads == self.nth:
                self.reads = None
                self.injected = True
                raise asyncio.CancelledError()
        return await super().load_checkpoint(session_id)

    async def append_events(self, session_id, events):
        if self.fail_tool_completion and any(
            event.type is EventType.TOOL_CALL_COMPLETED for event in events
        ):
            self.fail_tool_completion = False
            raise RuntimeError("terminal tool event unavailable")
        await super().append_events(session_id, events)

    async def publish_session_operation(self, session_id, **kwargs):
        result = await super().publish_session_operation(session_id, **kwargs)
        if kwargs["idempotency_key"].startswith("tool-effect:v1:"):
            self.effect_keys.append(kwargs["idempotency_key"])
        return result


class _External(Tool):
    spec = ToolSpec(
        name="record",
        effect=ToolEffect.EXTERNAL,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:recovered-owner-effect", behavior_version="1", implementation_version="1"
        ),
    )

    async def run(self, ctx, args):
        raise ConnectionError("lost external acknowledgement")


class _Reconciler:
    async def reconcile(self, *, context, receipt):
        return ToolEffectReconciliationResult(
            outcome="completed",
            observation="sent",
            receipt=ToolEffectReceipt(
                receipt_id=f"receipt-{context.tool_call_id}",
                receipt_schema="deployment",
                receipt_schema_version=1,
                tool_call_id=context.tool_call_id,
                tool_name=context.tool_name,
                idempotency_key=context.idempotency_key,
                outcome="completed",
                message="verified",
                source="reconciler",
                observed_at=datetime(2026, 9, 9, tzinfo=UTC),
            ),
        )


def _done() -> list[ModelStreamEvent]:
    return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]


def _calls(*names: str) -> list[ModelStreamEvent]:
    return [
        *(
            ModelStreamEvent.tool_call(
                id=f"call_{name}",
                name=name,
                arguments={"question": "Q?"} if name == "ask_user" else {},
            )
            for name in names
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _app(store: _Store, hook: InterruptionHook, first: list[ModelStreamEvent], tools, **agent):
    app = CayuApp(session_store=store, runtime_hooks=[hook], enable_logging=False)
    app.register_provider(FakeProvider([first, _done(), _done()]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=tools, **agent)
    return app


async def _paused(app: CayuApp) -> list:
    return [
        event
        async for event in app.run(
            RunRequest(
                session_id=_SESSION, agent_name="assistant", messages=[Message.text("user", "go")]
            )
        )
    ]


async def _user_input_recovery(store: _Store, hook: InterruptionHook):
    store.fail_tool_completion = True
    app = _app(
        store, hook, _calls("ask_user", "side_effect"), [UserInputTool(), PureRecordingTool()]
    )
    paused = await _paused(app)
    awaiting = next(e for e in paused if e.type is EventType.SESSION_AWAITING_USER_INPUT)
    resolved = [
        event
        async for event in app.resolve_user_input(
            UserInputResponse(
                session_id=_SESSION, input_id=awaiting.payload["input_id"], answer="yes"
            )
        )
    ]
    assert EventType.SESSION_INTERRUPTED in {event.type for event in resolved}
    return app, app.recover_user_input(
        UserInputRecoveryRequest(
            session_id=_SESSION,
            input_id=awaiting.payload["input_id"],
            answer="yes",
            tool_call_id="call_side_effect",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="recorded outside",
        )
    )


async def _tool_round_recovery(store: _Store, hook: InterruptionHook):
    store.fail_tool_completion = True
    app = _app(store, hook, _calls("side_effect"), [PureRecordingTool()])
    assert EventType.SESSION_FAILED in {event.type for event in await _paused(app)}
    checkpoint = await store.load_checkpoint(_SESSION)
    return app, app.recover_tool_round(
        ToolRoundRecoveryRequest(
            session_id=_SESSION,
            round_id=checkpoint["pending_tool_round"]["tool_round_id"],
            tool_call_id="call_side_effect",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="verified outside",
        )
    )


async def _tool_effect_reconciliation(store: _Store, hook: InterruptionHook):
    app = _app(
        store,
        hook,
        _calls("record"),
        [_External()],
        tool_effect_reconcilers={
            "record": ToolEffectReconciliationRegistration(
                reconciler=_Reconciler(), spec=_spec(supports_lookup=True)
            )
        },
    )
    assert EventType.SESSION_INTERRUPTED in {event.type for event in await _paused(app)}
    record = ToolEffectRecord.model_validate(
        await store.load_session_operation(_SESSION, store.effect_keys[-1])
    )
    session = await store.load(_SESSION)
    return app, app.reconcile_tool_effect(
        ToolEffectReconciliationRequest(
            **{
                name: getattr(record.intent, name)
                for name in (
                    "session_id",
                    "session_instance_id",
                    "tool_round_id",
                    "tool_call_id",
                    "tool_name",
                    "idempotency_key",
                )
            },
            expected_run_epoch=session.run_epoch,
            expected_revision=record.revision,
            lookup=True,
        )
    )


async def _provider_fallback(store: _Store, hook: InterruptionHook):
    # The staged invocation's frozen profile has no runtime hooks, so this
    # owner's exits are counted by their terminal event only.
    provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
    app, request = await _prepare_explicit_fallback_resolution(
        store, session_id=_SESSION, provider=provider
    )
    provider.adapter.start_events = (
        ModelStreamEvent.text_delta("recovered"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    )
    return app, app.resolve_provider_operation(request)


# Each owner and the event after which it holds the admitted invocation: the
# provider fallback owns it only once its replacement dispatch is durable.
_OWNERS = {
    "user-input-recovery": (_user_input_recovery, EventType.SESSION_RESUMED),
    "tool-round-supervisor": (_tool_round_recovery, EventType.SESSION_RESUMED),
    "tool-effect-reconciliation": (_tool_effect_reconciliation, EventType.TOOL_CALL_COMPLETED),
    "provider-fallback": (_provider_fallback, EventType.PROVIDER_OPERATION_STARTING),
}


@pytest.mark.parametrize(("fault", "nth"), [("cancel", 1), ("cancel", 2), ("close", 0)])
@pytest.mark.parametrize("owner", sorted(_OWNERS))
def test_recovered_owner_settles_its_exit_once(owner, fault, nth, monkeypatch):
    decisions = observe_decisions(monkeypatch)

    async def scenario():
        store, hook = _Store(), InterruptionHook()
        store.nth = nth
        open_owner, admitted_event = _OWNERS[owner]
        app, stream = await open_owner(store, hook)
        before = len(await store.load_events(_SESSION))
        hooks_before = hook.calls
        armed = False
        error: BaseException | None = None
        try:
            async for event in stream:
                if event.type is admitted_event and not armed:
                    armed = True
                    if fault == "close":
                        break
                    store.reads = 0
        except asyncio.CancelledError as cancelled:
            error = cancelled
        finally:
            await stream.aclose()
        try:
            phase_events = (await store.load_events(_SESSION))[before:]
            session = await store.load(_SESSION)
            return armed, store, session, phase_events, hook.calls - hooks_before, error
        finally:
            await app.aclose()

    armed, store, session, events, hooks, error = asyncio.run(scenario())
    assert armed
    if fault == "close":
        assert error is None
    else:
        assert store.injected and type(error) is asyncio.CancelledError
    assert session.status is SessionStatus.INTERRUPTED
    assert [e.type for e in events].count(EventType.SESSION_INTERRUPTED) == 1
    assert hooks == (0 if owner == "provider-fallback" else 1)
    # Settled once, and never by a ledger other than the owner's. At the first
    # read after admission, or a close there, the owner itself decides; later
    # the admitted run's own exit handler does.
    owner_decides = (fault, nth) != ("cancel", 2) and owner != "provider-fallback"
    if owner_decides:
        assert decisions == ["recovered"]
    else:
        assert decisions in ([], ["recovered"])
