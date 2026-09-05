from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_tool_discovery import _RememberKnowledgeTool

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    ExecutionProfileBehaviorIdentity,
    ForkSessionRequest,
    InMemorySessionStore,
    Message,
    ResumeRequest,
    RunRequest,
    StaticToolExposurePolicy,
    TargetedToolGrant,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime.tool_gateway import (
    DynamicToolReferenceRejection,
    dynamic_tool_reference_rejection,
)
from cayu.runtime.tool_grants import TargetedToolUseRejectionReason
from cayu.storage.sqlite import SQLiteSessionStore


class _CorrectingProvider(ModelProvider):
    name = "correcting-reference"

    def __init__(self, reference: str, reason: str) -> None:
        self.reference = reference
        self.reason = reason
        self.requests: list[ModelRequest] = []
        self.discovered: list[str] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:correcting-reference", behavior_version="1", implementation_version="1"
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        step = len(self.requests) % 4
        self.requests.append(request)
        results = [
            part
            for message in request.messages
            if message.role == "tool"
            for part in message.content
            if part.type == "tool_result"
        ]
        if step == 0:
            yield ModelStreamEvent.tool_call(
                id=f"bad-{len(self.requests)}",
                name="call_tool",
                arguments={"tool_ref": self.reference, "arguments": {}},
            )
        elif step == 1:
            result = results[-1]
            assert result.is_error
            assert result.structured is not None
            rejection = DynamicToolReferenceRejection.model_validate(result.structured)
            assert rejection.reason == self.reason
            assert rejection.next_action == "search_tools"
            assert result.content == rejection.message
            assert "search_tools" in result.content
            assert self.reference not in rejection.model_dump_json()
            assert "remember_knowledge" not in rejection.model_dump_json()
            yield ModelStreamEvent.tool_call(
                id=f"discover-{len(self.requests)}",
                name=rejection.next_action,
                arguments={"query": "remember durable knowledge", "limit": 1},
            )
        elif step == 2:
            assert results[-1].structured is not None
            [match] = results[-1].structured["matches"]
            self.discovered.append(match["tool_ref"])
            yield ModelStreamEvent.tool_call(
                id=f"valid-{len(self.requests)}",
                name="call_tool",
                arguments={"tool_ref": match["tool_ref"], "arguments": {"fact": "recovered"}},
            )
        else:
            assert results[-1].content == "remembered: recovered"
            assert not results[-1].is_error
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("boundary", ["same_session", "other_session", "fork"])
@pytest.mark.parametrize("reason", ["malformed", "unknown"])
def test_bad_reference_rediscovers_and_executes_without_leaking_authority(
    tmp_path,
    backend: str,
    boundary: str,
    reason: str,
    mode: str = "search_tools",
) -> None:
    async def run() -> None:
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "reference.db")
        )
        reference = "cayu_tool_v1_" + ("bad" if reason == "malformed" else "f" * 64)
        provider = _CorrectingProvider(reference, reason)
        tool = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(tool,),
            tool_discovery_mode=mode,
            tool_exposure_policy=StaticToolExposurePolicy(profile_id="discovery", tools=()),
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="parent",
                    agent_name="assistant",
                    messages=[Message.text("user", "Remember")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == [{"fact": "recovered"}]
        if boundary != "same_session":
            provider.reference = provider.discovered[0]
            # Opaque discovery references reveal no cross-session existence information.
            provider.reason = "unknown"
            if boundary == "fork":
                _ = [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id="parent",
                            session_id="child",
                        )
                    )
                ]
                stream = app.resume(
                    ResumeRequest(
                        session_id="child",
                        messages=[Message.text("user", "Remember again")],
                    )
                )
            else:
                stream = app.run(
                    RunRequest(
                        session_id="child",
                        agent_name="assistant",
                        messages=[Message.text("user", "Remember again")],
                    )
                )
            events.extend([event async for event in stream])
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert tool.calls == [{"fact": "recovered"}, {"fact": "recovered"}]
            assert provider.discovered[0] != provider.discovered[1]
        rejections = [event for event in events if event.type is EventType.TOOL_CALL_FAILED]
        assert len(rejections) == (1 if boundary == "same_session" else 2)
        for event in rejections:
            encoded = event.model_dump_json()
            assert reference not in encoded
            assert all(ref not in encoded for ref in provider.discovered)
            result = event.payload["result"]
            assert (
                DynamicToolReferenceRejection.model_validate(result["structured"]).next_action
                == "search_tools"
            )
            assert len(json.dumps(result)) < 1024
        if isinstance(store, SQLiteSessionStore):
            await store.close()
            store = SQLiteSessionStore(tmp_path / "reference.db")
            try:
                durable = await store.load_events("parent")
                [failure] = [event for event in durable if event.type is EventType.TOOL_CALL_FAILED]
                assert failure.payload["result"] == rejections[0].payload["result"]
                transcript = await store.load_transcript("parent")
                [rejected_part] = [
                    part
                    for message in transcript
                    if message.role == "tool"
                    for part in message.content
                    if part.type == "tool_result" and part.is_error
                ]
                assert rejected_part.structured == failure.payload["result"]["structured"]
                assert reference not in rejected_part.model_dump_json()
            finally:
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("reason", list(TargetedToolUseRejectionReason))
def test_corrective_rejections_are_bounded_typed_and_round_trip(reason) -> None:
    result = dynamic_tool_reference_rejection(reason)
    assert result.reason is reason
    assert DynamicToolReferenceRejection.model_validate_json(result.model_dump_json()) == result
    assert len(result.model_dump_json().encode()) < 1024
    assert reason.value in result.message
    assert set(result.model_dump()) == {
        "schema_version",
        "status",
        "reason",
        "next_action",
        "message",
    }
    if reason in {
        TargetedToolUseRejectionReason.CROSS_SESSION,
        TargetedToolUseRejectionReason.CROSS_GENERATION,
        TargetedToolUseRejectionReason.EXPIRED,
        TargetedToolUseRejectionReason.REVOKED,
        TargetedToolUseRejectionReason.CATALOGUE_DRIFT,
        TargetedToolUseRejectionReason.DESCRIPTOR_DRIFT,
    }:
        assert result.next_action == "search_tools"
    if reason in {
        TargetedToolUseRejectionReason.OUT_OF_CEILING,
        TargetedToolUseRejectionReason.TENANT_MISMATCH,
        TargetedToolUseRejectionReason.ALTERED_REPLAY,
    }:
        assert result.next_action == "do_not_retry"
        assert "search_tools" not in result.message


@pytest.mark.parametrize("lifecycle", ["expired", "revoked"])
def test_grant_lifecycle_rejection_can_rediscover_without_reusing_grant(lifecycle) -> None:
    async def run() -> None:
        now = [datetime(2026, 9, 5, tzinfo=UTC)]
        store = InMemorySessionStore()

        class LifecycleProvider(_CorrectingProvider):
            async def stream(self, request):
                if not self.requests:
                    [grant] = await store.list_targeted_tool_grants("parent")
                    self.reference = grant.tool_ref
                    if lifecycle == "expired":
                        now[0] = grant.expires_at + timedelta(seconds=1)
                    else:
                        session = await store.load("parent")
                        assert session is not None
                        await store.revoke_targeted_tool_grant(
                            grant.tool_ref,
                            session_id="parent",
                            expected_run_epoch=session.run_epoch,
                            reason="test revocation",
                            revoked_at=now[0],
                        )
                async for event in super().stream(request):
                    yield event

        provider = LifecycleProvider("unused", lifecycle)
        tool = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False, clock=lambda: now[0])
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(tool,),
            tool_discovery_mode="search_tools",
            targeted_tool_mode="call_tool",
            tool_exposure_policy=StaticToolExposurePolicy(profile_id="discovery", tools=()),
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="parent",
                    agent_name="assistant",
                    messages=[Message.text("user", "Remember")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="initial",
                            tool_id="cayu:remember_knowledge",
                            max_calls=1,
                            lifetime_seconds=60,
                        ),
                    ),
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == [{"fact": "recovered"}]
        [grant] = await store.list_targeted_tool_grants("parent")
        assert grant.used_calls == 0
        assert provider.reference != provider.discovered[0]

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode",
    [
        "openai_tool_search_client_or_search_tools",
        "openai_tool_search_hosted_or_search_tools",
    ],
)
def test_portable_discovery_fallback_provides_executable_recovery(tmp_path, mode: str) -> None:
    test_bad_reference_rediscovers_and_executes_without_leaking_authority(
        tmp_path,
        "memory",
        "same_session",
        "unknown",
        mode=mode,
    )
