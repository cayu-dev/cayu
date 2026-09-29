"""Public approval/input continuation publication against the durable backends."""

from __future__ import annotations

import asyncio
import base64
from contextlib import aclosing

import pytest
from pydantic import SecretStr
from tests.core.test_approval_lifecycle_execution_identities import _RecordingTool
from tests.core.test_resource_execution_access import Policy
from tests.core.test_tool_round_publication_failure_matrix import _WATCHDOG_SECONDS, _finish_tasks
from tests.core.test_tool_round_publication_failure_matrix import store_factory as store_factory

from cayu import AgentSpec, CayuApp, Message, ModelStreamEvent, RunRequest, ScriptedModelProvider
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.environments import Environment, EnvironmentSpec
from cayu.events import EventType
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.access import SessionAccessDenied, SessionAccessScope
from cayu.sessions.base import SessionStatus
from cayu.tools.base import ToolEffect
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyResult
from cayu.tools.user_input import UserInputTool
from cayu.vaults import StaticVault

_TERMINALS = {
    EventType.TOOL_CALL_COMPLETED,
    EventType.TOOL_CALL_FAILED,
    EventType.TOOL_CALL_BLOCKED,
    EventType.TOOL_CALL_APPROVAL_DENIED,
}


class _ContinuationFaultStore:
    def __init__(self, *args, fault="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.fault = fault
        self.close_requests = []
        self.close_results = []
        self.boundary_reached = asyncio.Event()
        self.release_publication = asyncio.Event()
        self.lost_terminal_ack = False

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        closing = request.kind in {"approval-close", "user-input-close"}
        first = closing and not self.close_requests
        if closing:
            self.close_requests.append(request.model_copy(deep=True))
        if first and self.fault == "before-close":
            self.boundary_reached.set()
            await self.release_publication.wait()
        result = await super().publish_runtime_publication(session_id, request=request, **kwargs)
        if closing:
            self.close_results.append(result)
        if first and self.fault == "after-close":
            self.boundary_reached.set()
            await self.release_publication.wait()
        if first and self.fault == "close-ack":
            raise ConnectionError("continuation close acknowledgement lost")
        return result

    async def append_events(self, session_id, events):
        await super().append_events(session_id, events)
        if (
            self.fault == "terminal-ack"
            and not self.lost_terminal_ack
            and any(event.type in _TERMINALS for event in events)
        ):
            self.lost_terminal_ack = True
            raise ConnectionError("continuation terminal acknowledgement lost")


@pytest.fixture
def continuation_store_factory(store_factory):
    # Every opener of the module-scoped PostgreSQL database uses the same test key.
    key = base64.urlsafe_b64encode(bytes([43]) * 32).decode().rstrip("=")
    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr(key)})
    )

    def open_store(**options):
        return store_factory(_ContinuationFaultStore, public_authority_alias_codec=codec, **options)

    return open_store


class _PausePolicy(ToolPolicy):
    def __init__(self, entrance):
        self.entrance = entrance

    async def authorize(self, request):
        if request.tool_call_id == "denied":
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="Denied sibling.")
        if self.entrance != "answer" and request.tool_call_id == "pause":
            return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class _PureRecordingTool(_RecordingTool):
    spec = _RecordingTool.spec.model_copy(update={"effect": ToolEffect.NONE})


def _runtime(
    store,
    entrance,
    *,
    dynamic=False,
    access_policy=None,
    include_denied=True,
    tool_type=_RecordingTool,
):
    tool = tool_type()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="pause",
                    name="ask_user" if entrance == "answer" else tool.spec.name,
                    arguments={"question": "Continue?"}
                    if entrance == "answer"
                    else {"value": "first"},
                ),
                ModelStreamEvent.tool_call(
                    id="allowed", name=tool.spec.name, arguments={"value": "second"}
                ),
                *(
                    [
                        ModelStreamEvent.tool_call(
                            id="denied", name=tool.spec.name, arguments={"value": "never"}
                        )
                    ]
                    if include_denied
                    else []
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, resource_access_policy=access_policy, enable_logging=False)
    app.register_provider(provider, default=True)
    if dynamic:
        app.register_environment(
            Environment(EnvironmentSpec(name="dynamic"), vault=StaticVault({"unused": "canary"})),
            default=True,
        )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[tool, *([UserInputTool()] if entrance == "answer" else [])],
        tool_policy=_PausePolicy(entrance),
    )
    return app, provider, tool


async def _drain(stream):
    async with aclosing(stream):
        return [event async for event in stream]


async def _pause(app, entrance, session_id, *, labels=None):
    events = await _drain(
        app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "Run the round.")],
                labels={} if labels is None else labels,
            )
        )
    )
    if entrance == "answer":
        event = next(
            event for event in events if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        return UserInputResponse(
            session_id=session_id, input_id=event.payload["input_id"], answer="yes"
        )
    event = next(event for event in events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
    return ToolApprovalRequest(
        session_id=session_id,
        approval_id=event.payload["approval_id"],
        tool_round_id=event.payload["tool_round_id"],
        tool_call_id=event.payload["tool_call_id"],
        decision=ToolApprovalDecision.DENY if entrance == "deny" else ToolApprovalDecision.APPROVE,
    )


def _resolve(app, request):
    return (
        app.resolve_user_input(request)
        if isinstance(request, UserInputResponse)
        else app.resolve_tool_approval(request)
    )


async def _assert_closed(store, session_id, *, entrance, recovered=False):
    transcript = await store.load_transcript(session_id)
    results = [
        part for message in transcript if message.role.value == "tool" for part in message.content
    ]
    expected_ids = ["pause", "allowed"] + ([] if recovered else ["denied"])
    assert [part.tool_call_id for part in results] == expected_ids
    assert [part.is_error for part in results] == [
        entrance == "deny",
        entrance == "deny" or recovered,
        *([] if recovered else [True]),
    ]
    events = await store.load_events(session_id)
    terminals = [event for event in events if event.type in _TERMINALS]
    assert [event.payload["tool_call_id"] for event in terminals] == expected_ids
    assert len({event.id for event in events}) == len(events)
    fingerprints = {event.payload["execution_profile_fingerprint"] for event in terminals}
    assert len(fingerprints) == 1
    checkpoint = await store.load_checkpoint(session_id)
    assert (
        checkpoint is None
        or not {"pending_tool_round", "pending_tool_approval", "pending_user_input"}
        & checkpoint.keys()
    )
    request = store.close_requests[0]
    receipt = await store.load_runtime_publication_receipt(session_id, request.publication_id)
    assert receipt is not None
    assert {event.id for event in terminals} <= {
        reference.event_id for reference in receipt.referenced_events
    }
    return transcript


@pytest.mark.parametrize("entrance", ["approve", "deny", "answer"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_continuation_close_replays_without_repeating_tools(
    continuation_store_factory, entrance, dynamic
):
    async def scenario():
        async with continuation_store_factory(fault="close-ack") as store:
            app, provider, tool = _runtime(store, entrance, dynamic=dynamic)
            request = await _pause(app, entrance, f"close-ack-{entrance}-{dynamic}")
            assert not tool.calls and len(provider.requests) == 1
            events = await _drain(_resolve(app, request))
            assert events[-1].type is EventType.SESSION_COMPLETED
            expected_calls = (
                []
                if entrance == "deny"
                else (
                    ([{"value": "first"}] if entrance == "approve" else []) + [{"value": "second"}]
                )
            )
            assert tool.calls == expected_calls
            assert len(store.close_requests) == 2
            assert store.close_requests[0] == store.close_requests[1]
            assert not store.close_results[0].replayed and store.close_results[1].replayed
            assert store.close_results[0].receipt == store.close_results[1].receipt
            transcript = await _assert_closed(store, request.session_id, entrance=entrance)
            await _drain(_resolve(app, request))
            assert tool.calls == expected_calls and len(provider.requests) == 2
            assert await store.load_transcript(request.session_id) == transcript
            assert await app.drain_background_interruptions()

    asyncio.run(scenario())


@pytest.mark.parametrize("entrance", ["approve", "answer"])
def test_continuation_retry_preserves_partial_terminal_publication(
    continuation_store_factory, entrance
):
    async def scenario():
        async with continuation_store_factory(fault="terminal-ack") as store:
            app, provider, tool = _runtime(store, entrance, dynamic=True, include_denied=False)
            request = await _pause(app, entrance, f"terminal-ack-{entrance}")
            first = await _drain(_resolve(app, request))
            assert store.lost_terminal_ack
            assert first[-1].type is EventType.SESSION_INTERRUPTED
            calls = list(tool.calls)
            assert calls == ([{"value": "first"}] if entrance == "approve" else []) + [
                {"value": "second"}
            ]
            events = await _drain(_resolve(app, request))
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert tool.calls == calls and len(provider.requests) == 2
            await _assert_closed(store, request.session_id, entrance=entrance, recovered=True)
            assert await app.drain_background_interruptions()

    asyncio.run(scenario())


@pytest.mark.parametrize("entrance", ["approve", "answer"])
@pytest.mark.parametrize("boundary", ["before-close", "after-close"])
def test_continuation_close_retains_repeated_cancellation(
    continuation_store_factory, entrance, boundary
):
    async def scenario():
        async with continuation_store_factory(fault=boundary) as store:
            app, provider, tool = _runtime(store, entrance, dynamic=True)
            request = await _pause(app, entrance, f"cancel-{entrance}-{boundary}")
            running = asyncio.create_task(_drain(_resolve(app, request)))
            try:
                await asyncio.wait_for(store.boundary_reached.wait(), _WATCHDOG_SECONDS)
                running.cancel("continuation cancelled")
                running.cancel("continuation cancelled")
                store.release_publication.set()
                with pytest.raises(asyncio.CancelledError, match="continuation cancelled"):
                    await asyncio.wait_for(running, _WATCHDOG_SECONDS)
            finally:
                store.release_publication.set()
                await _finish_tasks(running)
                assert await app.drain_background_interruptions()
            assert len(provider.requests) == 1
            assert tool.calls == ([{"value": "first"}] if entrance == "approve" else []) + [
                {"value": "second"}
            ]
            await _assert_closed(store, request.session_id, entrance=entrance)

    asyncio.run(scenario())


@pytest.mark.parametrize("entrance", ["approve", "answer"])
def test_revoked_continuation_does_not_dispatch_or_close(continuation_store_factory, entrance):
    async def scenario():
        async with continuation_store_factory() as store:
            policy = Policy()
            app, provider, tool = _runtime(store, entrance, access_policy=policy)
            access = await app.access("alice")
            request = await _pause(
                access, entrance, f"revoked-{entrance}", labels={"organization": "acme"}
            )
            transcript = await store.load_transcript(request.session_id)
            policy.current = SessionAccessScope()
            with pytest.raises(SessionAccessDenied):
                await _drain(_resolve(app, request))
            assert not tool.calls and len(provider.requests) == 1
            assert not store.close_requests
            assert await store.load_transcript(request.session_id) == transcript
            assert (await store.load(request.session_id)).status is SessionStatus.INTERRUPTED
            assert await app.drain_background_interruptions()

    asyncio.run(scenario())


@pytest.mark.parametrize("entrance", ["approve", "deny"])
def test_static_pure_approval_publishes_without_staging(continuation_store_factory, entrance):
    async def scenario():
        async with continuation_store_factory(fault="close-ack") as store:
            app, provider, tool = _runtime(store, entrance, tool_type=_PureRecordingTool)
            request = await _pause(app, entrance, f"pure-{entrance}")
            events = await _drain(_resolve(app, request))
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert tool.calls == (
                [{"value": "first"}, {"value": "second"}] if entrance == "approve" else []
            )
            assert len(provider.requests) == 2
            await _assert_closed(store, request.session_id, entrance=entrance)
            metrics = app.tool_terminal_publication_status()
            assert metrics.maximum_staged_bytes == metrics.active_round_reservations == 0
            assert await app.drain_background_interruptions()

    asyncio.run(scenario())


@pytest.mark.parametrize("entrance", ["approve", "answer"])
def test_closing_continuation_closes_active_result_stream(
    continuation_store_factory, entrance, monkeypatch
):
    async def scenario():
        async with continuation_store_factory() as store:
            app, provider, tool = _runtime(store, entrance, dynamic=True, include_denied=False)
            request = await _pause(app, entrance, f"close-stream-{entrance}")
            emit = app._tool_round_executor.emit_tool_call_result_with_hooks
            closed = []

            async def tracked_emit(**kwargs):
                try:
                    async with aclosing(emit(**kwargs)) as stream:
                        async for item in stream:
                            yield item
                finally:
                    if kwargs.get("terminal_event_emitter") is not None:
                        await asyncio.sleep(0)
                        closed.append(kwargs["tool_call"].id)

            monkeypatch.setattr(
                app._tool_round_executor, "emit_tool_call_result_with_hooks", tracked_emit
            )
            async with aclosing(_resolve(app, request)) as stream:
                async for event in stream:
                    if event.type is EventType.TOOL_CALL_COMPLETED:
                        assert closed == []
                        await stream.aclose()
                        assert closed == ["pause"]
                        break
                else:
                    raise AssertionError("Continuation never published its first terminal.")
            assert len(provider.requests) == 1
            assert tool.calls == ([{"value": "first"}] if entrance == "approve" else []) + [
                {"value": "second"}
            ]
            assert await app.drain_background_interruptions()
            # Unpublished durable evidence keeps its capacity until the pause is recovered.
            metrics = app.tool_terminal_publication_status()
            assert metrics.active_round_reservations == metrics.staged_count == 1
            recovered = await _drain(_resolve(app, request))
            assert recovered[-1].type is EventType.SESSION_COMPLETED
            assert len(provider.requests) == 2
            await _assert_closed(store, request.session_id, entrance=entrance, recovered=True)
            metrics = app.tool_terminal_publication_status()
            assert metrics.active_round_reservations == metrics.staged_count == 0

    asyncio.run(scenario())
