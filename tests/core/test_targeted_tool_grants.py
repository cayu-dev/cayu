from __future__ import annotations

import asyncio
import base64
import io
import json
import secrets
import sqlite3
import urllib.request
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError
from tests.core._execution_profile_fixtures import versioned_test_provider_identity

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    ForkSessionRequest,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    PostgresSessionStore,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    RecentTurnsContextPolicy,
    ResumeRequest,
    RetryPolicy,
    RunRequest,
    SessionRunFenced,
    StaticToolExposurePolicy,
    StructuredOutputSpec,
    TargetedToolGrant,
    TargetedToolGrantInspection,
    TargetedToolGrantRecord,
    TargetedToolGrantStateSnapshot,
    TargetedToolUseDisposition,
    TargetedToolUseRejectionReason,
    TargetedToolUseRequest,
    TaskCreate,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolContext,
    ToolEffect,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    ToolResult,
    ToolSpec,
)
from cayu.providers import (
    ModelContextOverflowError,
    ModelProviderError,
    ProviderOperationAdapter,
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
)
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.runtime.tool_gateway import (
    TargetedToolGatewayGrant,
    TargetedToolGatewayProjection,
    gateway_lifecycle_matches_outer_call,
    validate_effective_tool_arguments,
)
from cayu.runtime.tool_grants import (
    TARGETED_TOOL_TRANSCRIPT_REFERENCE,
    PreparedTargetedToolGrant,
    build_targeted_tool_grant_record,
    copy_targeted_tool_grant_record,
    targeted_tool_grant_event,
    targeted_tool_unresolved_rejection_event,
    targeted_tool_use_rejection_event,
    validate_targeted_tool_grant_batch_evidence,
    validate_targeted_tool_grant_revocation_evidence,
    validate_targeted_tool_unresolved_rejection_evidence,
    validate_targeted_tool_use_rejection_evidence,
)
from cayu.storage import SQLiteSessionStore
from cayu.storage.jsonl_export import export_sessions, import_sessions
from cayu.storage.migrations import SchemaMode
from cayu.vaults import SecretRedactor


class _Provider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return versioned_test_provider_identity(self)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _RetryProvider(_Provider):
    name = "retry-fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise ModelProviderError(
                "provider overloaded",
                provider=self.name,
                status_code=503,
                retryable=True,
            )
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _OverflowProvider(_Provider):
    name = "overflow-fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise ModelContextOverflowError(
                "context too large",
                provider=self.name,
                status_code=400,
                error_code="context_length_exceeded",
            )
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _ApprovalProvider(_Provider):
    name = "approval-fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="remember-call",
                name="remember",
                arguments={"fact": "Keep the retry identity stable."},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _RememberTool(Tool):
    spec = ToolSpec(
        name="remember",
        description="Remember a reviewed fact.",
        input_schema={
            "type": "object",
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
        },
        effect=ToolEffect.EXTERNAL,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:targeted-tool-grants:remember",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        raise AssertionError("Targeted grant tests must not execute the tool.")


class _GatewayRememberTool(Tool):
    spec = ToolSpec(
        name="remember",
        description="Remember one reviewed fact.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"fact": {"type": "string", "minLength": 1}},
            "required": ["fact"],
        },
        effect=ToolEffect.NONE,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:targeted-tool-grants:gateway-remember",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        self.calls.append(dict(args))
        return ToolResult(content=f"remembered: {args['fact']}")


class _PrivateArgumentsGatewayRememberTool(_GatewayRememberTool):
    @property
    def _publish_arguments(self) -> bool:
        return False


class _GatewayOtherTool(Tool):
    spec = ToolSpec(
        name="other",
        description="Run another targeted operation.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
        },
        effect=ToolEffect.NONE,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:targeted-tool-grants:gateway-other",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content="other")


class _RemoteRefGatewayTool(_GatewayRememberTool):
    spec = _GatewayRememberTool.spec.model_copy(
        update={"input_schema": {"$ref": "https://schemas.example.invalid/remember.json"}}
    )


class _GatewayProvider(_Provider):
    name = "gateway-fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            assert [tool["name"] for tool in request.tools] == ["call_tool"]
            gateway_message = next(
                message
                for message in request.messages
                if message.role == "user"
                and message.content[0].text.startswith("Cayu runtime targeted-tool context")
            )
            projection = json.loads(gateway_message.content[0].text.rsplit("\n", 1)[1])
            [descriptor] = projection["tools"]
            assert descriptor["tool_id"] == "cayu:remember"
            assert descriptor["name"] == "remember"
            assert descriptor["input_schema"] == _GatewayRememberTool.spec.input_schema
            yield ModelStreamEvent.tool_call(
                id="outer-call",
                name="call_tool",
                arguments={
                    "tool_ref": descriptor["tool_ref"],
                    "arguments": {"fact": "Keep the gateway identity stable."},
                },
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        assistant_call = next(
            part
            for message in request.messages
            if message.role == "assistant"
            for part in message.content
            if part.type == "tool_call"
        )
        tool_result = next(
            part
            for message in request.messages
            if message.role == "tool"
            for part in message.content
            if part.type == "tool_result"
        )
        assert assistant_call.tool_call_id == tool_result.tool_call_id == "outer-call"
        assert assistant_call.tool_name == tool_result.tool_name == "call_tool"
        assert tool_result.content == "remembered: Keep the gateway identity stable."
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def test_private_gateway_lifecycle_match_requires_the_safe_transcript_projection() -> None:
    payload = {
        "dispatch_kind": "gateway",
        "model_tool_name": "call_tool",
        "arguments_state": "unavailable",
        "arguments_sha256": "sha256:" + "1" * 64,
    }
    safe_arguments = {
        "tool_ref": TARGETED_TOOL_TRANSCRIPT_REFERENCE,
        "arguments": {},
    }

    assert gateway_lifecycle_matches_outer_call(
        effective_tool_name="remember",
        event_payload=payload,
        outer_tool_name="call_tool",
        outer_arguments=safe_arguments,
    )
    assert gateway_lifecycle_matches_outer_call(
        effective_tool_name="remember",
        event_payload={**payload, "arguments_state": "quarantined"},
        outer_tool_name="call_tool",
        outer_arguments=safe_arguments,
    )
    for effective_tool_name, candidate_payload, outer_tool_name, outer_arguments in (
        (
            "remember",
            {**payload, "arguments_state": "finalized"},
            "call_tool",
            safe_arguments,
        ),
        (
            "remember",
            payload,
            "call_tool",
            {"tool_ref": "altered-reference", "arguments": {}},
        ),
        (
            "remember",
            payload,
            "call_tool",
            {
                "tool_ref": TARGETED_TOOL_TRANSCRIPT_REFERENCE,
                "arguments": {"fact": "forged"},
            },
        ),
        ("remember", {**payload, "dispatch_kind": "direct"}, "call_tool", safe_arguments),
        ("call_tool", payload, "call_tool", safe_arguments),
        ("remember", payload, "remember", safe_arguments),
    ):
        assert not gateway_lifecycle_matches_outer_call(
            effective_tool_name=effective_tool_name,
            event_payload=candidate_payload,
            outer_tool_name=outer_tool_name,
            outer_arguments=outer_arguments,
        )


class _MultiCallGatewayProvider(_Provider):
    name = "multi-call-gateway-fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            gateway_message = next(
                message
                for message in request.messages
                if message.role == "user"
                and message.content[0].text.startswith("Cayu runtime targeted-tool context")
            )
            projection = json.loads(gateway_message.content[0].text.rsplit("\n", 1)[1])
            [descriptor] = projection["tools"]
            for index in range(2):
                yield ModelStreamEvent.tool_call(
                    id=f"outer-call-{index}",
                    name="call_tool",
                    arguments={
                        "tool_ref": descriptor["tool_ref"],
                        "arguments": {"fact": f"Fact {index}."},
                    },
                )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        assistant_calls = [
            part
            for message in request.messages
            if message.role == "assistant"
            for part in message.content
            if part.type == "tool_call"
        ]
        tool_results = [
            part
            for message in request.messages
            if message.role == "tool"
            for part in message.content
            if part.type == "tool_result"
        ]
        assert [part.tool_call_id for part in assistant_calls] == [
            "outer-call-0",
            "outer-call-1",
        ]
        assert [part.tool_call_id for part in tool_results] == [
            "outer-call-0",
            "outer-call-1",
        ]
        assert all(part.tool_name == "call_tool" for part in (*assistant_calls, *tool_results))
        assert all(
            part.arguments
            == {
                "tool_ref": TARGETED_TOOL_TRANSCRIPT_REFERENCE,
                "arguments": {},
            }
            for part in assistant_calls
        )
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _MixedExpiryGatewayProvider(_Provider):
    name = "mixed-expiry-gateway-fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        gateway_message = next(
            message
            for message in request.messages
            if message.role == "user"
            and message.content[0].text.startswith("Cayu runtime targeted-tool context")
        )
        projection = json.loads(gateway_message.content[0].text.rsplit("\n", 1)[1])
        descriptors_by_name = {descriptor["name"]: descriptor for descriptor in projection["tools"]}
        if len(self.requests) == 1:
            assert set(descriptors_by_name) == {"other", "remember"}
            yield ModelStreamEvent.tool_call(
                id="expiring-remember-call",
                name="call_tool",
                arguments={
                    "tool_ref": descriptors_by_name["remember"]["tool_ref"],
                    "arguments": {"fact": "Keep the remaining grant callable."},
                },
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        assert set(descriptors_by_name) == {"other"}
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _BackgroundGatewayAdapter(ProviderOperationAdapter):
    def __init__(self, provider: _GatewayProvider) -> None:
        self.provider = provider
        self.started = 0

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        self.started += 1
        state = ProviderOperationState(
            operation_id=f"targeted-operation-{self.started}",
            stream_protocol="targeted-test-v1",
            recovery_metadata={"cursor": 0},
        )

        async def events() -> AsyncIterator[ModelStreamEvent]:
            cursor = 0
            async for event in self.provider.stream(request.request):
                cursor += 1
                yield ModelStreamEvent.model_validate(
                    {
                        **event.model_dump(mode="python"),
                        "recovery_metadata": {"cursor": cursor},
                    }
                )

        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        del state
        raise AssertionError("Live targeted-tool test must not retrieve provider operations.")

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        del state
        raise AssertionError("Live targeted-tool test must not reconnect provider operations.")


class _BackgroundGatewayProvider(_GatewayProvider):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = _BackgroundGatewayAdapter(self)

    @property
    def provider_operation_mode(self) -> ProviderOperationMode:
        return ProviderOperationMode.BACKGROUND

    @property
    def provider_operations(self) -> ProviderOperationAdapter:
        return self.adapter


class _MixedStructuredOutputGatewayProvider(_Provider):
    name = "mixed-structured-output-gateway-fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            gateway_message = next(
                message
                for message in request.messages
                if message.role == "user"
                and message.content[0].text.startswith("Cayu runtime targeted-tool context")
            )
            projection = json.loads(gateway_message.content[0].text.rsplit("\n", 1)[1])
            [descriptor] = projection["tools"]
            yield ModelStreamEvent.tool_call(
                id="mixed-targeted-call",
                name="call_tool",
                arguments={
                    "tool_ref": descriptor["tool_ref"],
                    "arguments": {"fact": "must not execute"},
                },
            )
            yield ModelStreamEvent.tool_call(
                id="mixed-structured-output-call",
                name=STRUCTURED_OUTPUT_TOOL_NAME,
                arguments={"output": {"answer": "too early"}},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.tool_call(
            id="valid-structured-output-call",
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            arguments={"output": {"answer": "fixed"}},
        )
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class _RejectedGatewayProvider(_Provider):
    name = "rejected-gateway-fake"

    def __init__(self, case: str, *, unknown_ref: str | None = None) -> None:
        super().__init__()
        self.case = case
        self.unknown_ref = unknown_ref

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            gateway_message = next(
                message
                for message in request.messages
                if message.role == "user"
                and message.content[0].text.startswith("Cayu runtime targeted-tool context")
            )
            projection = json.loads(gateway_message.content[0].text.rsplit("\n", 1)[1])
            [descriptor] = projection["tools"]
            arguments = {
                "malformed": {"tool_ref": descriptor["tool_ref"]},
                "unknown": {
                    "tool_ref": self.unknown_ref or "unknown-reference",
                    "arguments": {"fact": "not reachable"},
                },
                "unknown_secret_collision": {
                    "tool_ref": self.unknown_ref or "unknown-reference",
                    "arguments": {"fact": "not reachable"},
                },
                "invalid_arguments": {
                    "tool_ref": descriptor["tool_ref"],
                    "arguments": {"fact": 42},
                },
                "unresolvable_schema": {
                    "tool_ref": descriptor["tool_ref"],
                    "arguments": {"fact": "must fail locally"},
                },
            }[self.case]
            yield ModelStreamEvent.tool_call(
                id=f"{self.case}-outer-call",
                name="call_tool",
                arguments=arguments,
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        result = next(
            part
            for message in request.messages
            if message.role == "tool"
            for part in message.content
            if part.type == "tool_result"
        )
        assert result.tool_name == "call_tool"
        assert result.tool_call_id == f"{self.case}-outer-call"
        assert result.is_error is True
        expected_reason = {
            "unknown_secret_collision": "malformed",
            "unresolvable_schema": "invalid_arguments",
        }.get(self.case, self.case)
        assert result.structured == {"status": "rejected", "reason": expected_reason}
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _RecordingPolicy(ToolPolicy):
    def __init__(self) -> None:
        self.requests: list[ToolPolicyRequest] = []

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        self.requests.append(request)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def _codec() -> PublicAuthorityAliasCodec:
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={"test": SecretStr(key)},
        )
    )


def _rotation_codec(
    *,
    active_key_id: str,
    include_second: bool,
) -> PublicAuthorityAliasCodec:
    keys = {
        "first": SecretStr(base64.urlsafe_b64encode(bytes([41]) * 32).decode("ascii").rstrip("="))
    }
    if include_second:
        keys["second"] = SecretStr(
            base64.urlsafe_b64encode(bytes([42]) * 32).decode("ascii").rstrip("=")
        )
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id=active_key_id, keys=keys)
    )


@pytest.fixture(params=("memory", "sqlite"))
def targeted_store(request, tmp_path: Path):
    if request.param == "memory":
        store = InMemorySessionStore(public_authority_alias_codec=_codec())
    else:
        store = SQLiteSessionStore(
            tmp_path / "targeted-grants.db",
            public_authority_alias_codec=_codec(),
        )
    try:
        yield store
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            asyncio.run(close())


def _app(store) -> tuple[CayuApp, _Provider]:
    provider = _Provider()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        enable_tool_gateway=True,
        tools=(_RememberTool(),),
    )
    return app, provider


async def _advance_through_grant(
    stream: AsyncIterator[Event],
) -> tuple[list[Event], Event]:
    observed: list[Event] = []
    async for event in stream:
        observed.append(event)
        if event.type is EventType.TARGETED_TOOL_GRANT_ISSUED:
            return observed, event
    raise AssertionError("Run ended without targeted grant evidence.")


def _use_request(
    record: TargetedToolGrantRecord,
    *,
    run_epoch: int,
    **updates: object,
) -> TargetedToolUseRequest:
    values: dict[str, object] = {
        "tool_ref": record.tool_ref,
        "session_id": record.session_id,
        "interaction_id": record.interaction_id,
        "generation_id": record.generation_id,
        "agent_name": record.agent_name,
        "task_id": record.task_id,
        "environment_name": record.environment_name,
        "principal": record.principal,
        "tenant": record.tenant,
        "catalogue_revision": record.catalogue_revision,
        "descriptor_version": record.descriptor_version,
        "schema_fingerprint": record.schema_fingerprint,
        "tool_id": record.tool_id,
        "tool_name": record.tool_name,
        "model_step_id": "model-step",
        "outer_tool_call_id": "outer-call",
        "arguments_sha256": "sha256:" + "1" * 64,
        "invocation_id": "invocation",
        "expected_run_epoch": run_epoch,
    }
    values.update(updates)
    return TargetedToolUseRequest.model_validate(values)


async def _open_targeted_grant(
    store,
    *,
    session_id: str,
    max_calls: int = 2,
    lifetime_seconds: int = 60,
):
    app, provider = _app(store)
    stream = app.run(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "Review this work.")],
            tool_grants=(
                TargetedToolGrant(
                    request_id="review-gotchas",
                    tool_id="cayu:remember",
                    max_calls=max_calls,
                    lifetime_seconds=lifetime_seconds,
                    origin="gotcha-reviewer",
                ),
            ),
        )
    )
    _prefix, _public_issued_event = await _advance_through_grant(stream)
    [record] = await store.list_targeted_tool_grants(session_id)
    issued_event = next(
        event
        for event in await store.load_events(session_id)
        if event.type is EventType.TARGETED_TOOL_GRANT_ISSUED
    )
    session = await store.load(session_id)
    assert session is not None
    return app, provider, stream, issued_event, record, session


def test_targeted_grant_request_is_strict_bounded_and_copy_safe() -> None:
    grant = TargetedToolGrant(
        request_id="review-gotchas",
        tool_id="cayu:remember",
        max_calls=2,
        origin="gotcha-reviewer",
    )
    request = RunRequest(agent_name="assistant", messages=[], tool_grants=(grant,))

    assert request.tool_grants == (grant,)
    assert request.tool_grants[0] is not grant
    with pytest.raises(ValidationError, match="extra"):
        TargetedToolGrant.model_validate({**grant.model_dump(mode="json"), "input_schema": {}})
    with pytest.raises(ValidationError, match="less than or equal to 32"):
        TargetedToolGrant(
            request_id="too-many",
            tool_id="cayu:remember",
            max_calls=33,
        )
    with pytest.raises(ValidationError, match="expire after the issuing interaction"):
        TargetedToolGrant.model_validate(
            {
                **grant.model_dump(mode="python"),
                "expires_after_interaction": 1,
            }
        )
    with pytest.raises(ValidationError, match="unique request_id"):
        RunRequest(
            agent_name="assistant",
            messages=[],
            tool_grants=(grant, grant.model_copy(update={"tool_id": "cayu:other"})),
        )

    record_values = {
        "tool_ref": "r" * 257,
        "session_id": "session",
        "interaction_id": "interaction",
        "generation_id": f"sha256:{'1' * 64}",
        "agent_name": "assistant",
        "catalogue_revision": f"sha256:{'2' * 64}",
        "descriptor_version": f"sha256:{'3' * 64}",
        "schema_fingerprint": f"sha256:{'4' * 64}",
        "tool_id": "cayu:remember",
        "tool_name": "remember",
        "model_step_id": "model-step",
        "outer_tool_call_id": "outer-call",
        "arguments_sha256": f"sha256:{'5' * 64}",
        "invocation_id": "invocation",
        "expected_run_epoch": 1,
    }
    with pytest.raises(ValidationError, match="tool_ref cannot exceed 256 UTF-8 bytes"):
        TargetedToolUseRequest.model_validate(record_values)
    with pytest.raises(TypeError, match="records must be a sequence"):
        TargetedToolGrantStateSnapshot(records=iter(()))


def test_runtime_issues_before_provider_and_store_binds_exact_replay(
    targeted_store,
) -> None:
    async def run() -> None:
        app, provider = _app(targeted_store)
        session_id = "targeted-grant-lifecycle"
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "Review this work.")],
                tool_grants=(
                    TargetedToolGrant(
                        request_id="review-gotchas",
                        tool_id="cayu:remember",
                        max_calls=2,
                        lifetime_seconds=60,
                        origin="gotcha-reviewer",
                    ),
                ),
            )
        )
        prefix, issued_event = await _advance_through_grant(stream)

        assert provider.requests == []
        assert [event.type for event in prefix[:2]] == [
            EventType.INTERACTION_STARTED,
            EventType.TARGETED_TOOL_GRANT_ISSUED,
        ]
        assert prefix[0].payload["targeted_tool_grant_count"] == 1
        assert prefix[0].payload["targeted_tool_grant_batch_fingerprint"].startswith("sha256:")
        assert "tool_ref" not in issued_event.model_dump_json()
        assert "input_schema" not in issued_event.model_dump_json()
        [record] = await targeted_store.list_targeted_tool_grants(session_id)
        session = await targeted_store.load(session_id)
        assert session is not None
        assert record.remaining_calls == 2
        validate_targeted_tool_grant_batch_evidence((record,), prefix[0])
        with pytest.raises(ValueError, match="conflicts with admitted interaction authority"):
            validate_targeted_tool_grant_batch_evidence((), prefix[0])
        assert record.tool_ref.startswith("cayu_authority_v1.")
        [public_record] = await app.inspect_targeted_tool_grants(session_id)
        assert isinstance(public_record, TargetedToolGrantInspection)
        assert public_record.tool_ref == record.tool_ref
        assert public_record.session_id != record.session_id
        assert public_record.interaction_id != record.interaction_id
        assert public_record.grant_id == record.grant_id
        assert "principal" not in public_record.model_dump()
        assert "tenant" not in public_record.model_dump()
        assert "revocation_reason" not in public_record.model_dump()

        base = dict(
            tool_ref=record.tool_ref,
            session_id=session_id,
            interaction_id=record.interaction_id,
            generation_id=record.generation_id,
            agent_name=record.agent_name,
            task_id=record.task_id,
            environment_name=record.environment_name,
            principal=record.principal,
            tenant=record.tenant,
            catalogue_revision=record.catalogue_revision,
            descriptor_version=record.descriptor_version,
            schema_fingerprint=record.schema_fingerprint,
            tool_id=record.tool_id,
            tool_name=record.tool_name,
            model_step_id="model-step-1",
            outer_tool_call_id="outer-call-1",
            arguments_sha256="sha256:" + "1" * 64,
            invocation_id="invocation-1",
            expected_run_epoch=session.run_epoch,
        )
        first = await targeted_store.bind_targeted_tool_grant_use(
            TargetedToolUseRequest(**base),
            observed_at=datetime.now(UTC),
        )
        exact = await targeted_store.bind_targeted_tool_grant_use(
            TargetedToolUseRequest(**base),
            observed_at=datetime.now(UTC) + timedelta(seconds=1),
        )
        altered = await targeted_store.bind_targeted_tool_grant_use(
            TargetedToolUseRequest(**{**base, "arguments_sha256": "sha256:" + "2" * 64}),
            observed_at=datetime.now(UTC) + timedelta(seconds=2),
        )
        second = await targeted_store.bind_targeted_tool_grant_use(
            TargetedToolUseRequest(
                **{
                    **base,
                    "model_step_id": "model-step-2",
                    "outer_tool_call_id": "outer-call-2",
                    "invocation_id": "invocation-2",
                }
            ),
            observed_at=datetime.now(UTC) + timedelta(seconds=3),
        )
        exhausted = await targeted_store.bind_targeted_tool_grant_use(
            TargetedToolUseRequest(
                **{
                    **base,
                    "model_step_id": "model-step-3",
                    "outer_tool_call_id": "outer-call-3",
                    "invocation_id": "invocation-3",
                }
            ),
            observed_at=datetime.now(UTC) + timedelta(seconds=4),
        )

        assert first.disposition is TargetedToolUseDisposition.BOUND
        assert exact.disposition is TargetedToolUseDisposition.REJOINED
        assert exact.binding == first.binding
        assert altered.reason is TargetedToolUseRejectionReason.ALTERED_REPLAY
        assert second.disposition is TargetedToolUseDisposition.BOUND
        assert exhausted.reason is TargetedToolUseRejectionReason.EXHAUSTED
        [updated] = await targeted_store.list_targeted_tool_grants(session_id)
        assert updated.used_calls == 2
        assert updated.remaining_calls == 0
        state = await targeted_store.load_targeted_tool_grant_state(session_id)
        assert state.records == (updated,)
        assert len(state.uses) == 2
        assert {binding.invocation_id for binding in state.uses} == {
            "invocation-1",
            "invocation-2",
        }

        suffix = [event async for event in stream]
        assert provider.requests
        footprint_event = next(
            event for event in suffix if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        )
        assert footprint_event.payload["schema_version"] == 4
        assert footprint_event.payload["targeted_tool_grants"] == {
            "schema_version": 1,
            "generation_id": record.generation_id,
            "catalogue_revision": record.catalogue_revision,
            "grant_count": 1,
            "grant_ids": [record.grant_id],
            "tool_ids": [record.tool_id],
            "max_calls": 2,
            "used_calls": 0,
            "remaining_calls": 2,
            "direct_tool_prefix_changed": False,
        }
        assert "tool_ref" not in footprint_event.model_dump_json()
        assert [tool["name"] for tool in provider.requests[0].tools] == [
            "remember",
            "call_tool",
        ]
        assert suffix[-1].type is EventType.SESSION_COMPLETED

        export = io.StringIO()
        assert await export_sessions(targeted_store, stream=export) == 1
        [imported] = list(import_sessions(io.StringIO(export.getvalue())))
        assert imported.targeted_tool_grant_state == state
        wrong_batch = json.loads(export.getvalue())
        interaction_started = next(
            event
            for event in wrong_batch["events"]
            if event["type"] == EventType.INTERACTION_STARTED
        )
        interaction_started["payload"]["targeted_tool_grant_batch_fingerprint"] = (
            f"sha256:{'0' * 64}"
        )
        with pytest.raises(ValueError, match="admitted interaction authority"):
            list(import_sessions([json.dumps(wrong_batch)]))
        codec = targeted_store.public_authority_alias_codec
        assert codec is not None
        other_session_record = build_targeted_tool_grant_record(
            PreparedTargetedToolGrant(
                request=TargetedToolGrant(
                    request_id=record.request_id,
                    tool_id=record.tool_id,
                    max_calls=record.max_calls,
                    lifetime_seconds=int((record.expires_at - record.issued_at).total_seconds()),
                    origin=record.origin,
                ),
                tool_name=record.tool_name,
                catalogue_revision=record.catalogue_revision,
                descriptor_version=record.descriptor_version,
                schema_fingerprint=record.schema_fingerprint,
            ),
            session_id="different-session",
            interaction_id=record.interaction_id,
            generation_id=record.generation_id,
            agent_name=record.agent_name,
            task_id=record.task_id,
            environment_name=record.environment_name,
            principal=record.principal,
            tenant=record.tenant,
            issued_at=record.issued_at,
            codec=codec,
        )
        wrong_scope = json.loads(export.getvalue())
        wrong_scope["targeted_tool_grant_state"] = TargetedToolGrantStateSnapshot(
            records=(other_session_record,),
        ).model_dump(mode="json")
        with pytest.raises(ValueError, match="belongs to a different session"):
            list(import_sessions([json.dumps(wrong_scope)]))

    asyncio.run(run())


async def _assert_call_tool_routes_outer_identity(
    targeted_store,
    *,
    session_id: str,
    secret_redactor: SecretRedactor | None = None,
    background: bool = False,
) -> None:
    provider = _BackgroundGatewayProvider() if background else _GatewayProvider()
    tool = _GatewayRememberTool()
    policy = _RecordingPolicy()
    app = CayuApp(
        session_store=targeted_store,
        secret_redactor=secret_redactor,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        enable_tool_gateway=True,
        tools=(tool,),
        tool_policy=policy,
        tool_exposure_policy=StaticToolExposurePolicy(
            profile_id="targeted-only",
            tools=(),
        ),
    )

    streamed = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "Review and remember one fact.")],
                tool_grants=(
                    TargetedToolGrant(
                        request_id="remember-reviewed-fact",
                        tool_id="cayu:remember",
                        max_calls=1,
                        lifetime_seconds=60,
                    ),
                ),
            )
        )
    ]

    assert streamed[-1].type is EventType.SESSION_COMPLETED
    assert tool.calls == [{"fact": "Keep the gateway identity stable."}]
    assert len(policy.requests) == 1
    assert policy.requests[0].tool_name == "remember"
    assert policy.requests[0].arguments == {"fact": "Keep the gateway identity stable."}
    assert len(provider.requests) == 2
    [record] = await targeted_store.list_targeted_tool_grants(session_id)
    assert record.used_calls == 1
    assert record.remaining_calls == 0
    assert record.tool_ref not in json.dumps(
        [event.model_dump(mode="json") for event in streamed],
        sort_keys=True,
    )
    durable_events = await targeted_store.load_events(session_id)
    [consumed] = [
        event
        for event in durable_events
        if event.type is EventType.TARGETED_TOOL_REFERENCE_CONSUMED
    ]
    assert consumed.tool_name == "remember"
    assert consumed.payload["outer_tool_call_id"] == "outer-call"
    [started] = [event for event in durable_events if event.type is EventType.TOOL_CALL_STARTED]
    [completed] = [event for event in durable_events if event.type is EventType.TOOL_CALL_COMPLETED]
    for event in (started, completed):
        assert event.tool_name == "remember"
        assert event.payload["dispatch_kind"] == "gateway"
        assert event.payload["model_tool_name"] == "call_tool"
        assert event.payload["grant_id"] == record.grant_id
        assert event.payload["use_id"] == consumed.payload["use_id"]
    transcript = await targeted_store.load_transcript(session_id)
    assistant_call = next(
        part
        for message in transcript
        if message.role == "assistant"
        for part in message.content
        if part.type == "tool_call"
    )
    tool_result = next(
        part
        for message in transcript
        if message.role == "tool"
        for part in message.content
        if part.type == "tool_result"
    )
    assert assistant_call.tool_call_id == tool_result.tool_call_id == "outer-call"
    assert assistant_call.tool_name == tool_result.tool_name == "call_tool"
    assert assistant_call.arguments == {
        "tool_ref": TARGETED_TOOL_TRANSCRIPT_REFERENCE,
        "arguments": {"fact": "Keep the gateway identity stable."},
    }
    assert record.tool_ref not in json.dumps(
        [message.model_dump(mode="json") for message in transcript],
        sort_keys=True,
    )


def test_call_tool_routes_outer_identity_through_the_effective_target(
    targeted_store,
) -> None:
    asyncio.run(
        _assert_call_tool_routes_outer_identity(
            targeted_store,
            session_id="targeted-call-tool",
        )
    )


def test_call_tool_routes_through_postgres(postgres_dsn: str) -> None:
    async def run() -> None:
        store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
            public_authority_alias_codec=_codec(),
        )
        try:
            await _assert_call_tool_routes_outer_identity(
                store,
                session_id="targeted-call-tool-postgres",
            )
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "secret_collision",
    ("cayu_authority_v1.", "sha256:"),
    ids=("public-reference-prefix", "grant-identity-prefix"),
)
def test_call_tool_runtime_reference_survives_short_secret_collision(
    targeted_store,
    secret_collision: str,
) -> None:
    asyncio.run(
        _assert_call_tool_routes_outer_identity(
            targeted_store,
            session_id=f"targeted-call-tool-secret-collision-{len(secret_collision)}",
            secret_redactor=SecretRedactor(secret_collision),
        )
    )


def test_background_call_tool_reference_survives_short_secret_collision(targeted_store) -> None:
    asyncio.run(
        _assert_call_tool_routes_outer_identity(
            targeted_store,
            session_id="targeted-background-call-tool-secret-collision",
            secret_redactor=SecretRedactor("cayu_authority_v1."),
            background=True,
        )
    )


@pytest.mark.parametrize(
    "projection_case",
    ("private_tool", "multi_call_round"),
)
def test_call_tool_preserves_unavailable_argument_projections(
    targeted_store,
    projection_case: str,
) -> None:
    async def run() -> None:
        multi_call = projection_case == "multi_call_round"
        session_id = f"targeted-call-tool-unavailable-{projection_case}"
        provider = _MultiCallGatewayProvider() if multi_call else _GatewayProvider()
        tool = _GatewayRememberTool() if multi_call else _PrivateArgumentsGatewayRememberTool()
        app = CayuApp(session_store=targeted_store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            enable_tool_gateway=True,
            tools=(tool,),
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="targeted-only",
                tools=(),
            ),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Remember the requested facts.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="remember-facts",
                            tool_id="cayu:remember",
                            max_calls=2 if multi_call else 1,
                        ),
                    ),
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == (
            [{"fact": "Fact 0."}, {"fact": "Fact 1."}]
            if multi_call
            else [{"fact": "Keep the gateway identity stable."}]
        )
        terminals = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
        assert len(terminals) == (2 if multi_call else 1)
        assert all(event.payload["arguments_state"] == "unavailable" for event in terminals)
        transcript = await targeted_store.load_transcript(session_id)
        assistant_calls = [
            part
            for message in transcript
            if message.role == "assistant"
            for part in message.content
            if part.type == "tool_call"
        ]
        assert len(assistant_calls) == len(terminals)
        assert all(
            part.arguments
            == {
                "tool_ref": TARGETED_TOOL_TRANSCRIPT_REFERENCE,
                "arguments": {},
            }
            for part in assistant_calls
        )

    asyncio.run(run())


def test_call_tool_schema_validation_supports_local_references() -> None:
    schema = {
        "$defs": {"fact": {"type": "string", "minLength": 1}},
        "type": "object",
        "properties": {"fact": {"$ref": "#/$defs/fact"}},
        "required": ["fact"],
    }

    assert validate_effective_tool_arguments({"fact": "remember this"}, schema) is True
    assert validate_effective_tool_arguments({"fact": ""}, schema) is False


def test_gateway_descriptor_redaction_preserves_only_the_runtime_reference() -> None:
    descriptor_secret = "descriptor-secret-value"
    tool_ref = "cayu_authority_v1.runtime-issued-reference"
    projection = TargetedToolGatewayProjection(
        grants=(
            TargetedToolGatewayGrant(
                tool_ref=tool_ref,
                grant_id=f"sha256:{'a' * 64}",
                tool_id="cayu:remember",
                name="remember",
                description=descriptor_secret,
                input_schema={
                    "type": "object",
                    "description": descriptor_secret,
                    "properties": {"fact": {"const": descriptor_secret}},
                },
                remaining_calls=1,
                expires_at=datetime.now(UTC) + timedelta(minutes=1),
            ),
        )
    )

    instruction = projection.instruction_text(
        redactor=SecretRedactor(["cayu_authority_v1.", descriptor_secret]),
    )
    payload = json.loads(instruction.rsplit("\n", 1)[1])
    [descriptor] = payload["tools"]

    assert descriptor["tool_ref"] == tool_ref
    assert descriptor["description"] == TARGETED_TOOL_TRANSCRIPT_REFERENCE
    assert descriptor["input_schema"]["description"] == TARGETED_TOOL_TRANSCRIPT_REFERENCE
    assert (
        descriptor["input_schema"]["properties"]["fact"]["const"]
        == TARGETED_TOOL_TRANSCRIPT_REFERENCE
    )


def test_mixed_structured_output_round_preserves_private_targeted_selection(
    targeted_store,
) -> None:
    async def run() -> None:
        session_id = "targeted-mixed-structured-output"
        provider = _MixedStructuredOutputGatewayProvider()
        tool = _GatewayRememberTool()
        app = CayuApp(
            session_store=targeted_store,
            secret_redactor=SecretRedactor("cayu_authority_v1."),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            enable_tool_gateway=True,
            tools=(tool,),
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="targeted-only",
                tools=(),
            ),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Use the grant, then answer.")],
                    structured_output=StructuredOutputSpec(
                        json_schema={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                        max_retries=1,
                    ),
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="remember-before-answer",
                            tool_id="cayu:remember",
                            max_calls=1,
                            lifetime_seconds=60,
                        ),
                    ),
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == []
        assert len(provider.requests) == 2
        [record] = await targeted_store.list_targeted_tool_grants(session_id)
        assert record.used_calls == 0
        transcript = await targeted_store.load_transcript(session_id)
        assert record.tool_ref not in json.dumps(
            [message.model_dump(mode="json") for message in transcript],
            sort_keys=True,
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    "redactor_secret",
    (None, "cayu_authority_v1.", "sha256:"),
    ids=("ordinary", "runtime-reference-secret-collision", "grant-identity-secret-collision"),
)
@pytest.mark.parametrize(
    "publish_arguments",
    (True, False),
    ids=("published-arguments", "private-arguments"),
)
def test_call_tool_approval_preserves_bound_gateway_identity(
    targeted_store,
    redactor_secret: str | None,
    publish_arguments: bool,
) -> None:
    async def run() -> None:
        provider = _GatewayProvider()
        tool = (
            _GatewayRememberTool() if publish_arguments else _PrivateArgumentsGatewayRememberTool()
        )
        secret_redactor = SecretRedactor(redactor_secret)
        app = CayuApp(
            session_store=targeted_store,
            secret_redactor=secret_redactor,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            enable_tool_gateway=True,
            tools=(tool,),
            tool_policy=AlwaysRequireApprovalToolPolicy(),
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="targeted-only",
                tools=(),
            ),
        )
        session_id = "targeted-call-tool-approval"

        initial_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Review and remember one fact.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="remember-reviewed-fact",
                            tool_id="cayu:remember",
                            max_calls=1,
                            lifetime_seconds=60,
                        ),
                    ),
                )
            )
        ]
        approval_event = next(
            event
            for event in initial_events
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        assert approval_event.tool_name == "remember"
        assert approval_event.payload["approval"]["tool_name"] == "remember"
        assert "tool_ref" not in approval_event.model_dump_json()
        assert "targeted_tool_grant_id" not in approval_event.model_dump_json()
        assert "targeted_tool_invocation" not in approval_event.model_dump_json()
        assert tool.calls == []
        [bound] = await targeted_store.list_targeted_tool_grants(session_id)
        assert bound.used_calls == 1

        resumed_app = CayuApp(
            session_store=targeted_store,
            secret_redactor=secret_redactor,
            enable_logging=False,
        )
        resumed_app.register_provider(provider, default=True)
        resumed_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            enable_tool_gateway=True,
            tools=(tool,),
            tool_policy=AlwaysRequireApprovalToolPolicy(),
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="targeted-only",
                tools=(),
            ),
        )
        resumed_events = [
            event
            async for event in resumed_app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=approval_event.payload["approval"]["approval_id"],
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id=approval_event.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        assert resumed_events[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == [{"fact": "Keep the gateway identity stable."}]
        [consumed] = await targeted_store.list_targeted_tool_grants(session_id)
        assert consumed.used_calls == 1
        assert consumed.remaining_calls == 0
        assert (
            sum(
                event.type is EventType.TARGETED_TOOL_REFERENCE_REJOINED for event in resumed_events
            )
            == 1
        )
        assert [tool["name"] for tool in provider.requests[1].tools] == ["call_tool"]
        transcript = await targeted_store.load_transcript(session_id)
        result = next(
            part
            for message in transcript
            if message.role == "tool"
            for part in message.content
            if part.type == "tool_result"
        )
        assert result.tool_name == "call_tool"
        assert result.tool_call_id == "outer-call"
        assistant_call = next(
            part
            for message in transcript
            if message.role == "assistant"
            for part in message.content
            if part.type == "tool_call"
        )
        assert assistant_call.arguments == {
            "tool_ref": TARGETED_TOOL_TRANSCRIPT_REFERENCE,
            "arguments": (
                {"fact": "Keep the gateway identity stable."} if publish_arguments else {}
            ),
        }

    asyncio.run(run())


@pytest.mark.parametrize(
    ("case", "reason"),
    (
        ("malformed", TargetedToolUseRejectionReason.MALFORMED),
        ("unknown", TargetedToolUseRejectionReason.UNKNOWN),
        ("invalid_arguments", TargetedToolUseRejectionReason.INVALID_ARGUMENTS),
        ("unresolvable_schema", TargetedToolUseRejectionReason.INVALID_ARGUMENTS),
    ),
)
def test_call_tool_rejects_before_policy_or_execution(
    targeted_store,
    case: str,
    reason: TargetedToolUseRejectionReason,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        session_id = f"targeted-call-tool-{case}"
        codec = targeted_store.public_authority_alias_codec
        assert codec is not None
        provider = _RejectedGatewayProvider(
            case,
            unknown_ref=(
                codec.encode(
                    f"sha256:{'f' * 64}",
                    field_name="tool_ref",
                    session_id=session_id,
                )
                if case == "unknown"
                else None
            ),
        )
        tool = _RemoteRefGatewayTool() if case == "unresolvable_schema" else _GatewayRememberTool()
        policy = _RecordingPolicy()
        app = CayuApp(session_store=targeted_store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            enable_tool_gateway=True,
            tools=(tool,),
            tool_policy=policy,
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="targeted-only",
                tools=(),
            ),
        )
        streamed = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Try the targeted tool.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="remember-reviewed-fact",
                            tool_id="cayu:remember",
                            max_calls=1,
                            lifetime_seconds=60,
                        ),
                    ),
                )
            )
        ]

        assert streamed[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == []
        assert policy.requests == []
        [record] = await targeted_store.list_targeted_tool_grants(session_id)
        assert record.used_calls == 0
        assert record.remaining_calls == 1
        durable_events = await targeted_store.load_events(session_id)
        [rejected] = [
            event
            for event in durable_events
            if event.type is EventType.TARGETED_TOOL_REFERENCE_REJECTED
        ]
        assert rejected.payload["rejection_reason"] == reason.value
        assert "tool_ref" not in rejected.model_dump_json()
        assert not any(event.type is EventType.TOOL_CALL_STARTED for event in durable_events)
        [failed] = [event for event in durable_events if event.type is EventType.TOOL_CALL_FAILED]
        assert failed.tool_name == "call_tool"
        assert failed.payload["reason"] == reason.value
        assert "tool_ref" not in failed.model_dump_json()

    if case == "unresolvable_schema":

        def fail_remote_retrieval(*args, **kwargs):
            del args, kwargs
            raise AssertionError("JSON Schema validation attempted remote retrieval.")

        monkeypatch.setattr(urllib.request, "urlopen", fail_remote_retrieval)
    asyncio.run(run())


def test_call_tool_does_not_trust_an_unissued_secret_bearing_reference(targeted_store) -> None:
    async def run() -> None:
        session_id = "targeted-call-tool-unissued-secret-reference"
        attacker_reference = "cayu_authority_v1.attacker-controlled"
        provider = _RejectedGatewayProvider(
            "unknown_secret_collision",
            unknown_ref=attacker_reference,
        )
        tool = _GatewayRememberTool()
        policy = _RecordingPolicy()
        app = CayuApp(
            session_store=targeted_store,
            secret_redactor=SecretRedactor("cayu_authority_v1."),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            enable_tool_gateway=True,
            tools=(tool,),
            tool_policy=policy,
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="targeted-only",
                tools=(),
            ),
        )

        streamed = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Try an unissued targeted reference.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="remember-reviewed-fact",
                            tool_id="cayu:remember",
                            max_calls=1,
                            lifetime_seconds=60,
                        ),
                    ),
                )
            )
        ]

        assert streamed[-1].type is EventType.SESSION_FAILED
        assert "redaction marker" in streamed[-1].payload["error"]
        assert tool.calls == []
        assert policy.requests == []
        [record] = await targeted_store.list_targeted_tool_grants(session_id)
        assert record.used_calls == 0
        assert record.remaining_calls == 1
        durable_events = await targeted_store.load_events(session_id)
        assert not any(
            event.type
            in {
                EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
                EventType.TARGETED_TOOL_REFERENCE_REJECTED,
                EventType.TOOL_CALL_STARTED,
            }
            for event in durable_events
        )
        transcript = await targeted_store.load_transcript(session_id)
        persisted = json.dumps(
            {
                "events": [event.model_dump(mode="json") for event in durable_events],
                "transcript": [message.model_dump(mode="json") for message in transcript],
            },
            sort_keys=True,
        )
        assert attacker_reference not in persisted
        assert "attacker-controlled" not in persisted

    asyncio.run(run())


def test_provider_retry_reuses_one_prepared_targeted_grant_snapshot(targeted_store) -> None:
    async def run() -> None:
        provider = _RetryProvider()
        app = CayuApp(
            session_store=targeted_store,
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            enable_tool_gateway=True,
            tools=(_RememberTool(),),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="targeted-grant-provider-retry",
                    messages=[Message.text("user", "Review this work.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="review-gotchas",
                            tool_id="cayu:remember",
                            max_calls=2,
                            lifetime_seconds=60,
                        ),
                    ),
                )
            )
        ]

        assert len(provider.requests) == 2
        [record] = await targeted_store.list_targeted_tool_grants("targeted-grant-provider-retry")
        durable_events = await targeted_store.load_events("targeted-grant-provider-retry")
        assert (
            sum(event.type is EventType.TARGETED_TOOL_GRANT_ISSUED for event in durable_events) == 1
        )
        footprints = [
            event.payload["targeted_tool_grants"]
            for event in events
            if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert len(footprints) == 2
        assert footprints[0] == footprints[1]
        assert footprints[0]["grant_ids"] == [record.grant_id]
        assert footprints[0]["used_calls"] == 0
        assert events[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(run())


def test_context_overflow_reuses_one_prepared_targeted_grant_snapshot(targeted_store) -> None:
    async def run() -> None:
        provider = _OverflowProvider()
        app = CayuApp(session_store=targeted_store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            enable_tool_gateway=True,
            tools=(_RememberTool(),),
            context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="targeted-grant-context-overflow",
                    messages=[
                        Message.text("user", "Old review request."),
                        Message.text("user", "Review this work."),
                    ],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="review-gotchas",
                            tool_id="cayu:remember",
                            max_calls=2,
                            lifetime_seconds=60,
                        ),
                    ),
                )
            )
        ]

        assert len(provider.requests) == 2
        [record] = await targeted_store.list_targeted_tool_grants("targeted-grant-context-overflow")
        durable_events = await targeted_store.load_events("targeted-grant-context-overflow")
        assert (
            sum(event.type is EventType.TARGETED_TOOL_GRANT_ISSUED for event in durable_events) == 1
        )
        footprint_events = [
            event for event in events if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert [event.payload["request_variant"] for event in footprint_events] == [
            "initial",
            "context_overflow_recovery",
        ]
        footprints = [event.payload["targeted_tool_grants"] for event in footprint_events]
        assert footprints[0] == footprints[1]
        assert footprints[0]["grant_ids"] == [record.grant_id]
        assert events[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(run())


def test_approval_continuation_reconstructs_the_same_targeted_grant_snapshot(
    targeted_store,
) -> None:
    async def run() -> None:
        provider = _ApprovalProvider()
        app = CayuApp(session_store=targeted_store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            enable_tool_gateway=True,
            tools=(_RememberTool(),),
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )

        initial_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="targeted-grant-approval-continuation",
                    messages=[Message.text("user", "Review this work.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="review-gotchas",
                            tool_id="cayu:remember",
                            lifetime_seconds=60,
                        ),
                    ),
                )
            )
        ]
        approval = next(
            event
            for event in initial_events
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        resumed_events = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="targeted-grant-approval-continuation",
                    approval_id=approval.payload["approval"]["approval_id"],
                    tool_round_id=approval.payload["tool_round_id"],
                    tool_call_id=approval.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]

        assert len(provider.requests) == 2
        [record] = await targeted_store.list_targeted_tool_grants(
            "targeted-grant-approval-continuation"
        )
        durable_events = await targeted_store.load_events("targeted-grant-approval-continuation")
        assert (
            sum(event.type is EventType.TARGETED_TOOL_GRANT_ISSUED for event in durable_events) == 1
        )
        assert (
            sum(
                event.type is EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED
                for event in durable_events
            )
            == 1
        )
        footprint_events = [
            event
            for event in (*initial_events, *resumed_events)
            if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert len(footprint_events) == 2
        footprints = [event.payload["targeted_tool_grants"] for event in footprint_events]
        assert footprints[0] == footprints[1]
        assert footprints[0]["grant_ids"] == [record.grant_id]
        assert resumed_events[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(run())


def test_approval_continuation_omits_a_naturally_expired_targeted_grant(
    targeted_store,
) -> None:
    async def run() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        provider = _ApprovalProvider()
        app = CayuApp(
            session_store=targeted_store,
            enable_logging=False,
            clock=lambda: now[0],
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            enable_tool_gateway=True,
            tools=(_RememberTool(),),
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )

        initial_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="targeted-grant-expired-approval-continuation",
                    messages=[Message.text("user", "Review this work.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="review-gotchas",
                            tool_id="cayu:remember",
                            lifetime_seconds=1,
                        ),
                    ),
                )
            )
        ]
        approval = next(
            event
            for event in initial_events
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        [record] = await targeted_store.list_targeted_tool_grants(
            "targeted-grant-expired-approval-continuation"
        )
        now[0] = record.expires_at

        resumed_events = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="targeted-grant-expired-approval-continuation",
                    approval_id=approval.payload["approval"]["approval_id"],
                    tool_round_id=approval.payload["tool_round_id"],
                    tool_call_id=approval.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]

        assert len(provider.requests) == 2
        footprint_events = [
            event
            for event in (*initial_events, *resumed_events)
            if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert footprint_events[0].payload["targeted_tool_grants"]["grant_ids"] == [record.grant_id]
        assert footprint_events[1].payload.get("targeted_tool_grants") is None
        durable_events = await targeted_store.load_events(record.session_id)
        assert any(event.type is EventType.TARGETED_TOOL_GRANT_EXPIRED for event in durable_events)
        assert any(
            event.type is EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED
            and event.payload["outcome"] == "rejected"
            and event.payload["rejection_reason"] == "expired"
            for event in durable_events
        )
        assert resumed_events[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(run())


def test_approval_continuation_keeps_the_nonexpired_targeted_grant(
    targeted_store,
) -> None:
    async def run() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        provider = _MixedExpiryGatewayProvider()
        remember = _GatewayRememberTool()
        app = CayuApp(
            session_store=targeted_store,
            enable_logging=False,
            clock=lambda: now[0],
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            enable_tool_gateway=True,
            tools=(remember, _GatewayOtherTool()),
            tool_policy=AlwaysRequireApprovalToolPolicy(),
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="targeted-only",
                tools=(),
            ),
        )

        initial_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="targeted-grant-partial-expiry-continuation",
                    messages=[Message.text("user", "Review and remember this work.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="expiring-remember",
                            tool_id="cayu:remember",
                            lifetime_seconds=1,
                        ),
                        TargetedToolGrant(
                            request_id="remaining-other",
                            tool_id="cayu:other",
                            lifetime_seconds=60,
                        ),
                    ),
                )
            )
        ]
        approval = next(
            event
            for event in initial_events
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        records = await targeted_store.list_targeted_tool_grants(
            "targeted-grant-partial-expiry-continuation"
        )
        records_by_name = {record.tool_name: record for record in records}
        now[0] = records_by_name["remember"].expires_at

        resumed_events = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="targeted-grant-partial-expiry-continuation",
                    approval_id=approval.payload["approval"]["approval_id"],
                    tool_round_id=approval.payload["tool_round_id"],
                    tool_call_id=approval.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]

        assert resumed_events[-1].type is EventType.SESSION_COMPLETED
        assert remember.calls == [{"fact": "Keep the remaining grant callable."}]
        assert len(provider.requests) == 2
        footprint_events = [
            event
            for event in (*initial_events, *resumed_events)
            if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert set(footprint_events[0].payload["targeted_tool_grants"]["grant_ids"]) == {
            record.grant_id for record in records
        }
        assert footprint_events[1].payload["targeted_tool_grants"]["grant_ids"] == [
            records_by_name["other"].grant_id
        ]

    asyncio.run(run())


def test_task_backed_targeted_grant_binds_task_scope(targeted_store) -> None:
    async def run() -> None:
        tasks = InMemoryTaskStore()
        task = await tasks.create_task(TaskCreate(task_id="targeted-task", type="run"))
        claimed = await tasks.claim_task("targeted-worker", lease_seconds=300)
        assert claimed is not None
        assert claimed.id == task.id

        provider = _Provider()
        app = CayuApp(
            session_store=targeted_store,
            task_store=tasks,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            enable_tool_gateway=True,
            tools=(_RememberTool(),),
        )
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="task-scoped-targeted-grant",
                task_id=task.id,
                task_worker_id="targeted-worker",
                messages=[Message.text("user", "Review this task.")],
                tool_grants=(
                    TargetedToolGrant(
                        request_id="task-review",
                        tool_id="cayu:remember",
                    ),
                ),
            )
        )
        await _advance_through_grant(stream)
        [record] = await targeted_store.list_targeted_tool_grants("task-scoped-targeted-grant")
        session = await targeted_store.load(record.session_id)
        assert session is not None
        assert record.task_id == task.id
        [inspection] = await app.inspect_targeted_tool_grants(record.session_id)
        assert inspection.task_id is not None
        assert inspection.task_id != task.id
        assert task.id not in inspection.model_dump_json()

        accepted = await targeted_store.bind_targeted_tool_grant_use(
            _use_request(record, run_epoch=session.run_epoch),
            observed_at=record.issued_at + timedelta(seconds=1),
        )
        mismatched = await targeted_store.bind_targeted_tool_grant_use(
            _use_request(
                record,
                run_epoch=session.run_epoch,
                task_id="different-task",
                model_step_id="different-task-step",
                outer_tool_call_id="different-task-call",
                invocation_id="different-task-invocation",
            ),
            observed_at=record.issued_at + timedelta(seconds=2),
        )
        assert accepted.disposition is TargetedToolUseDisposition.BOUND
        assert mismatched.reason is TargetedToolUseRejectionReason.TASK_MISMATCH

        async for _event in stream:
            pass
        assert provider.requests

    asyncio.run(run())


def test_fork_copies_no_grant_authority_and_copied_references_are_inert(
    targeted_store,
) -> None:
    async def run() -> None:
        app, _provider = _app(targeted_store)
        source_id = "targeted-grant-fork-source"
        child_id = "targeted-grant-fork-child"
        run_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "Create source history.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="source-grant",
                            tool_id="cayu:remember",
                        ),
                    ),
                )
            )
        ]
        assert run_events[-1].type is EventType.SESSION_COMPLETED
        [source_record] = await targeted_store.list_targeted_tool_grants(source_id)

        resume_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=source_id,
                    messages=[
                        Message.text(
                            "user",
                            f"Historical inert reference: {source_record.tool_ref}",
                        )
                    ],
                )
            )
        ]
        assert resume_events[-1].type is EventType.SESSION_COMPLETED
        source_state_before = await targeted_store.load_targeted_tool_grant_state(source_id)

        fork_events = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(
                    source_session_id=source_id,
                    session_id=child_id,
                )
            )
        ]
        assert [event.type for event in fork_events] == [EventType.SESSION_FORKED]
        assert await targeted_store.load_targeted_tool_grant_state(child_id) == (
            TargetedToolGrantStateSnapshot()
        )
        assert await targeted_store.load_targeted_tool_grant_state(source_id) == source_state_before
        child_transcript = await targeted_store.load_transcript(child_id)
        assert source_record.tool_ref in " ".join(
            part.text or ""
            for message in child_transcript
            for part in message.content
            if part.type == "text"
        )
        child_events = await targeted_store.load_events(child_id)
        [reset] = [
            event
            for event in child_events
            if event.type is EventType.TARGETED_TOOL_GRANT_FORK_RESET
        ]
        assert reset.payload["inherited_grant_count"] == 0
        assert reset.payload["inherited_reference_count"] == 0

    asyncio.run(run())


def test_invalid_targeted_grant_fails_before_session_or_provider(targeted_store) -> None:
    async def run() -> None:
        app, provider = _app(targeted_store)

        with pytest.raises(ValueError, match="unregistered tool_id"):
            async for _event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="invalid-targeted-grant",
                    messages=[Message.text("user", "Do work")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="unknown",
                            tool_id="cayu:unknown",
                        ),
                    ),
                )
            ):
                pass

        assert provider.requests == []
        assert await targeted_store.load("invalid-targeted-grant") is None

    asyncio.run(run())


def test_targeted_grant_requires_a_registration_wide_stable_gateway(targeted_store) -> None:
    async def run() -> None:
        provider = _Provider()
        app = CayuApp(session_store=targeted_store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(_RememberTool(),),
        )

        with pytest.raises(ValueError, match="enable_tool_gateway=True"):
            async for _event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="gateway-not-enabled",
                    messages=[Message.text("user", "Review this work.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="remember",
                            tool_id="cayu:remember",
                        ),
                    ),
                )
            ):
                pass

        assert provider.requests == []
        assert await targeted_store.load("gateway-not-enabled") is None

    asyncio.run(run())


def test_enabled_gateway_is_stable_without_an_active_grant(targeted_store) -> None:
    async def run() -> None:
        app, provider = _app(targeted_store)
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="stable-gateway-without-grant",
                    messages=[Message.text("user", "Continue ordinary work.")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        [request] = provider.requests
        assert [tool["name"] for tool in request.tools] == ["remember", "call_tool"]
        assert all(
            "cayu.targeted-tool-context.v1" not in part.text
            for message in request.messages
            for part in message.content
            if part.type == "text"
        )

    asyncio.run(run())


def test_call_tool_without_an_active_grant_rejects_without_target_execution(
    targeted_store,
) -> None:
    class UngrantedGatewayProvider(_Provider):
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.tool_call(
                    id="ungranted-call",
                    name="call_tool",
                    arguments={
                        "tool_ref": "unissued-reference",
                        "arguments": {"fact": "must not execute"},
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            yield ModelStreamEvent.text_delta("rejection handled")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> None:
        provider = UngrantedGatewayProvider()
        tool = _GatewayRememberTool()
        app = CayuApp(session_store=targeted_store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(tool,),
            enable_tool_gateway=True,
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="ungranted-call-tool",
                    messages=[Message.text("user", "Continue ordinary work.")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        [rejected] = [
            event for event in events if event.type is EventType.TARGETED_TOOL_REFERENCE_REJECTED
        ]
        assert rejected.payload["rejection_reason"] == "malformed"
        assert tool.calls == []

    asyncio.run(run())


def test_targeted_grant_use_rejects_every_scope_and_descriptor_drift(
    targeted_store,
) -> None:
    async def run() -> None:
        _app_instance, _provider, stream, _issued, record, session = await _open_targeted_grant(
            targeted_store,
            session_id="targeted-grant-rejections",
        )
        cases = (
            (
                {"interaction_id": "other-interaction"},
                TargetedToolUseRejectionReason.CROSS_INTERACTION,
            ),
            (
                {"generation_id": "sha256:" + "2" * 64},
                TargetedToolUseRejectionReason.CROSS_GENERATION,
            ),
            (
                {"principal": "other-principal"},
                TargetedToolUseRejectionReason.PRINCIPAL_MISMATCH,
            ),
            (
                {"tenant": "other-tenant"},
                TargetedToolUseRejectionReason.TENANT_MISMATCH,
            ),
            (
                {"agent_name": "other-agent"},
                TargetedToolUseRejectionReason.AGENT_MISMATCH,
            ),
            (
                {"task_id": "other-task"},
                TargetedToolUseRejectionReason.TASK_MISMATCH,
            ),
            (
                {"environment_name": "other-environment"},
                TargetedToolUseRejectionReason.ENVIRONMENT_MISMATCH,
            ),
            (
                {"tool_id": "cayu:other", "tool_name": "other"},
                TargetedToolUseRejectionReason.OUT_OF_CEILING,
            ),
            (
                {"catalogue_revision": "sha256:" + "3" * 64},
                TargetedToolUseRejectionReason.CATALOGUE_DRIFT,
            ),
            (
                {"descriptor_version": "sha256:" + "4" * 64},
                TargetedToolUseRejectionReason.DESCRIPTOR_DRIFT,
            ),
            (
                {"schema_fingerprint": "sha256:" + "5" * 64},
                TargetedToolUseRejectionReason.DESCRIPTOR_DRIFT,
            ),
        )
        observed_at = record.issued_at + timedelta(seconds=1)
        for index, (updates, expected_reason) in enumerate(cases):
            request = _use_request(
                record,
                run_epoch=session.run_epoch,
                model_step_id=f"model-step-{index}",
                outer_tool_call_id=f"outer-call-{index}",
                invocation_id=f"invocation-{index}",
                **updates,
            )
            result = await targeted_store.bind_targeted_tool_grant_use(
                request,
                observed_at=observed_at,
            )
            assert result.disposition is TargetedToolUseDisposition.REJECTED
            assert result.reason is expected_reason
            assert result.event is not None
            assert result.event.type is EventType.TARGETED_TOOL_REFERENCE_REJECTED
            assert "tool_ref" not in result.event.model_dump_json()

        expired = await targeted_store.bind_targeted_tool_grant_use(
            _use_request(
                record,
                run_epoch=session.run_epoch,
                model_step_id="expired-step",
                outer_tool_call_id="expired-call",
                invocation_id="expired-invocation",
            ),
            observed_at=record.expires_at,
        )
        assert expired.reason is TargetedToolUseRejectionReason.EXPIRED
        assert EventType.TARGETED_TOOL_GRANT_EXPIRED in {
            event.type for event in await targeted_store.load_events(record.session_id)
        }
        async for _event in stream:
            pass

    asyncio.run(run())


def test_unknown_and_malformed_references_are_fenced_and_durably_rejected(
    targeted_store,
) -> None:
    async def run() -> None:
        _app_instance, _provider, stream, _issued, record, session = await _open_targeted_grant(
            targeted_store,
            session_id="targeted-grant-unresolved",
        )
        with pytest.raises(SessionRunFenced, match="run epoch is stale"):
            await targeted_store.bind_targeted_tool_grant_use(
                _use_request(
                    record,
                    run_epoch=session.run_epoch + 1,
                    tool_ref="malformed-reference",
                ),
                observed_at=record.issued_at + timedelta(seconds=1),
            )

        malformed = await targeted_store.bind_targeted_tool_grant_use(
            _use_request(
                record,
                run_epoch=session.run_epoch,
                tool_ref="malformed-reference",
                outer_tool_call_id="malformed-call",
                invocation_id="malformed-invocation",
            ),
            observed_at=record.issued_at + timedelta(seconds=2),
        )
        codec = targeted_store.public_authority_alias_codec
        assert codec is not None
        unknown_ref = codec.encode(
            f"sha256:{'f' * 64}",
            field_name="tool_ref",
            session_id=record.session_id,
        )
        unknown = await targeted_store.bind_targeted_tool_grant_use(
            _use_request(
                record,
                run_epoch=session.run_epoch,
                tool_ref=unknown_ref,
                outer_tool_call_id="unknown-call",
                invocation_id="unknown-invocation",
            ),
            observed_at=record.issued_at + timedelta(seconds=3),
        )
        assert malformed.reason is TargetedToolUseRejectionReason.MALFORMED
        assert unknown.reason is TargetedToolUseRejectionReason.UNKNOWN
        for result in (malformed, unknown):
            assert result.event is not None
            assert result.event.payload.keys() >= {
                "rejection_id",
                "rejection_reason",
                "arguments_sha256",
            }
            assert "tool_ref" not in result.event.model_dump_json()
            assert "cayu_authority_v1" not in result.event.model_dump_json()
        async for _event in stream:
            pass

    asyncio.run(run())


def test_targeted_grant_issue_reuses_exact_authority_and_rejects_tool_conflicts(
    targeted_store,
) -> None:
    async def run() -> None:
        _app_instance, _provider, stream, issued, record, session = await _open_targeted_grant(
            targeted_store,
            session_id="targeted-grant-issue-reuse",
        )
        codec = targeted_store.public_authority_alias_codec
        assert codec is not None
        forged_reference_record = TargetedToolGrantRecord.model_validate(
            record.model_copy(
                update={
                    "tool_ref": codec.encode(
                        f"sha256:{'e' * 64}",
                        field_name="tool_ref",
                        session_id=record.session_id,
                    )
                }
            ).model_dump(mode="python")
        )
        state_before_forged_reference = await targeted_store.load_targeted_tool_grant_state(
            record.session_id
        )
        with pytest.raises(ValueError, match="tool_ref conflicts"):
            await targeted_store.issue_targeted_tool_grants(
                record.session_id,
                expected_run_epoch=session.run_epoch,
                records=(forged_reference_record,),
                events=(issued,),
            )
        assert (
            await targeted_store.load_targeted_tool_grant_state(record.session_id)
            == state_before_forged_reference
        )
        reused = await targeted_store.issue_targeted_tool_grants(
            record.session_id,
            expected_run_epoch=session.run_epoch,
            records=(record,),
            events=(issued,),
        )
        assert reused.outcomes == ("reused",)
        assert reused.records == (record,)
        assert reused.events[0].type is EventType.TARGETED_TOOL_GRANT_REUSED
        bound = await targeted_store.bind_targeted_tool_grant_use(
            _use_request(record, run_epoch=session.run_epoch),
            observed_at=record.issued_at + timedelta(seconds=1),
        )
        assert bound.disposition is TargetedToolUseDisposition.BOUND
        reused_after_use = await targeted_store.issue_targeted_tool_grants(
            record.session_id,
            expected_run_epoch=session.run_epoch,
            records=(record,),
            events=(issued,),
        )
        assert reused_after_use.records[0].used_calls == 1
        assert reused_after_use.events == reused.events

        prepared = PreparedTargetedToolGrant(
            request=TargetedToolGrant(
                request_id="conflicting-request",
                tool_id=record.tool_id,
                max_calls=record.max_calls,
                lifetime_seconds=int((record.expires_at - record.issued_at).total_seconds()),
                origin=record.origin,
            ),
            tool_name=record.tool_name,
            catalogue_revision=record.catalogue_revision,
            descriptor_version=record.descriptor_version,
            schema_fingerprint=record.schema_fingerprint,
        )
        conflicting = build_targeted_tool_grant_record(
            prepared,
            session_id=record.session_id,
            interaction_id=record.interaction_id,
            generation_id=record.generation_id,
            agent_name=record.agent_name,
            task_id=record.task_id,
            environment_name=record.environment_name,
            principal=record.principal,
            tenant=record.tenant,
            issued_at=record.issued_at,
            codec=codec,
        )
        conflicting_event = targeted_tool_grant_event(
            conflicting,
            event_type=EventType.TARGETED_TOOL_GRANT_ISSUED,
            timestamp=record.issued_at,
            outcome="issued",
            event_id_suffix="issued",
        )
        before = await targeted_store.load_targeted_tool_grant_state(record.session_id)
        with pytest.raises(ValueError, match="admitted interaction authority"):
            await targeted_store.issue_targeted_tool_grants(
                record.session_id,
                expected_run_epoch=session.run_epoch,
                records=(conflicting,),
                events=(conflicting_event,),
            )
        assert await targeted_store.load_targeted_tool_grant_state(record.session_id) == before
        async for _event in stream:
            pass

    asyncio.run(run())


def test_targeted_grant_revocation_and_reconstruction_fail_closed(targeted_store) -> None:
    async def run() -> None:
        _app_instance, _provider, stream, _issued, record, session = await _open_targeted_grant(
            targeted_store,
            session_id="targeted-grant-reconstruction",
        )
        descriptors = {
            record.tool_id: (
                record.tool_name,
                record.descriptor_version,
                record.schema_fingerprint,
            )
        }
        reconstructed = await targeted_store.reconstruct_targeted_tool_grants(
            record.session_id,
            expected_run_epoch=session.run_epoch,
            interaction_id=record.interaction_id,
            generation_id=record.generation_id,
            agent_name=record.agent_name,
            task_id=record.task_id,
            environment_name=record.environment_name,
            principal=record.principal,
            tenant=record.tenant,
            catalogue_revision=record.catalogue_revision,
            descriptors_by_id=descriptors,
            capability_ceiling_names=frozenset({record.tool_name}),
            observed_at=record.issued_at + timedelta(seconds=1),
        )
        assert [grant.grant_id for grant in reconstructed.valid] == [record.grant_id]
        assert reconstructed.rejected == ()

        task_drifted = await targeted_store.reconstruct_targeted_tool_grants(
            record.session_id,
            expected_run_epoch=session.run_epoch,
            interaction_id=record.interaction_id,
            generation_id=record.generation_id,
            agent_name=record.agent_name,
            task_id="different-task",
            environment_name=record.environment_name,
            principal=record.principal,
            tenant=record.tenant,
            catalogue_revision=record.catalogue_revision,
            descriptors_by_id=descriptors,
            capability_ceiling_names=frozenset({record.tool_name}),
            observed_at=record.issued_at + timedelta(seconds=2),
        )
        assert task_drifted.valid == ()
        assert task_drifted.rejected == (
            (record.grant_id, TargetedToolUseRejectionReason.TASK_MISMATCH),
        )

        drifted = await targeted_store.reconstruct_targeted_tool_grants(
            record.session_id,
            expected_run_epoch=session.run_epoch,
            interaction_id=record.interaction_id,
            generation_id=record.generation_id,
            agent_name=record.agent_name,
            task_id=record.task_id,
            environment_name=record.environment_name,
            principal=record.principal,
            tenant=record.tenant,
            catalogue_revision="sha256:" + "9" * 64,
            descriptors_by_id=descriptors,
            capability_ceiling_names=frozenset({record.tool_name}),
            observed_at=record.issued_at + timedelta(seconds=3),
        )
        assert drifted.valid == ()
        assert drifted.rejected == (
            (record.grant_id, TargetedToolUseRejectionReason.CATALOGUE_DRIFT),
        )

        revoked_at = record.issued_at + timedelta(seconds=4)
        revoked = await targeted_store.revoke_targeted_tool_grant(
            record.tool_ref,
            session_id=record.session_id,
            expected_run_epoch=session.run_epoch,
            reason="operator-revoked",
            revoked_at=revoked_at,
        )
        assert revoked is not None
        assert revoked.revoked_at == revoked_at
        exact = await targeted_store.revoke_targeted_tool_grant(
            record.tool_ref,
            session_id=record.session_id,
            expected_run_epoch=session.run_epoch,
            reason="operator-revoked",
            revoked_at=revoked_at + timedelta(seconds=1),
        )
        assert exact == revoked
        with pytest.raises(SessionRunFenced):
            await targeted_store.revoke_targeted_tool_grant(
                "not-a-targeted-reference",
                session_id=record.session_id,
                expected_run_epoch=session.run_epoch + 1,
                reason="operator-revoked",
                revoked_at=revoked_at,
            )
        assert (
            await targeted_store.revoke_targeted_tool_grant(
                "not-a-targeted-reference",
                session_id=record.session_id,
                expected_run_epoch=session.run_epoch,
                reason="operator-revoked",
                revoked_at=revoked_at,
            )
            is None
        )
        with pytest.raises(ValueError, match="cannot exceed 512 UTF-8 bytes"):
            await targeted_store.revoke_targeted_tool_grant(
                record.tool_ref,
                session_id=record.session_id,
                expected_run_epoch=session.run_epoch,
                reason="r" * 513,
                revoked_at=revoked_at,
            )
        with pytest.raises(ValueError, match="different reason"):
            await targeted_store.revoke_targeted_tool_grant(
                record.tool_ref,
                session_id=record.session_id,
                expected_run_epoch=session.run_epoch,
                reason="different-reason",
                revoked_at=revoked_at,
            )
        rejected = await targeted_store.bind_targeted_tool_grant_use(
            _use_request(record, run_epoch=session.run_epoch),
            observed_at=revoked_at + timedelta(seconds=1),
        )
        assert rejected.reason is TargetedToolUseRejectionReason.REVOKED
        async for _event in stream:
            pass

    asyncio.run(run())


def test_durable_grant_and_use_identities_reject_copied_corruption(targeted_store) -> None:
    async def run() -> None:
        _app_instance, _provider, stream, _issued, record, session = await _open_targeted_grant(
            targeted_store,
            session_id="targeted-grant-corruption",
        )
        with pytest.raises(ValueError, match="grant_id conflicts"):
            copy_targeted_tool_grant_record(
                TargetedToolGrantRecord.model_validate(
                    {**record.model_dump(mode="python"), "tool_name": "altered-tool"}
                )
            )
        bound = await targeted_store.bind_targeted_tool_grant_use(
            _use_request(record, run_epoch=session.run_epoch),
            observed_at=record.issued_at + timedelta(seconds=1),
        )
        assert bound.binding is not None
        with pytest.raises(ValidationError, match="use_id conflicts"):
            type(bound.binding).model_validate(
                {
                    **bound.binding.model_dump(mode="python"),
                    "arguments_sha256": "sha256:" + "8" * 64,
                }
            )
        with pytest.raises(ValidationError, match="falls outside its grant lifetime"):
            TargetedToolGrantStateSnapshot(
                records=(bound.grant,),
                uses=(
                    bound.binding.model_copy(
                        update={"bound_at": record.issued_at - timedelta(seconds=1)}
                    ),
                ),
            )
        with pytest.raises(ValidationError, match="follows its revocation timestamp"):
            revoked_before_use = TargetedToolGrantRecord.model_validate(
                bound.grant.model_copy(
                    update={
                        "revoked_at": record.issued_at,
                        "revocation_reason": "corrupt chronology",
                    }
                ).model_dump(mode="python")
            )
            TargetedToolGrantStateSnapshot(
                records=(revoked_before_use,),
                uses=(bound.binding,),
            )
        async for _event in stream:
            pass

    asyncio.run(run())


def test_targeted_rejection_and_revocation_evidence_is_exact(targeted_store) -> None:
    async def run() -> None:
        _app_instance, _provider, stream, _issued, record, session = await _open_targeted_grant(
            targeted_store,
            session_id="targeted-evidence-validation",
        )
        request = _use_request(record, run_epoch=session.run_epoch)
        observed_at = record.issued_at + timedelta(seconds=1)
        resolved = targeted_tool_use_rejection_event(
            record,
            request,
            reason=TargetedToolUseRejectionReason.REVOKED,
            timestamp=observed_at,
        )
        validate_targeted_tool_use_rejection_evidence(
            record,
            request,
            reason=TargetedToolUseRejectionReason.REVOKED,
            event=resolved,
        )
        with pytest.raises(ValueError, match="rejection evidence conflicts"):
            validate_targeted_tool_use_rejection_evidence(
                record,
                request,
                reason=TargetedToolUseRejectionReason.REVOKED,
                event=resolved.model_copy(
                    update={"payload": {**resolved.payload, "model_step_id": "altered-step"}}
                ),
            )

        unresolved = targeted_tool_unresolved_rejection_event(
            request,
            reason=TargetedToolUseRejectionReason.UNKNOWN,
            timestamp=observed_at,
            agent_name=record.agent_name,
            environment_name=record.environment_name,
        )
        validate_targeted_tool_unresolved_rejection_evidence(
            request,
            reason=TargetedToolUseRejectionReason.UNKNOWN,
            event=unresolved,
            agent_name=record.agent_name,
            environment_name=record.environment_name,
        )
        with pytest.raises(ValueError, match="Unresolved targeted tool rejection"):
            validate_targeted_tool_unresolved_rejection_evidence(
                request,
                reason=TargetedToolUseRejectionReason.UNKNOWN,
                event=unresolved.model_copy(update={"agent_name": "altered-agent"}),
                agent_name=record.agent_name,
                environment_name=record.environment_name,
            )

        revoked_at = observed_at + timedelta(seconds=1)
        revoked = TargetedToolGrantRecord.model_validate(
            record.model_copy(
                update={
                    "revoked_at": revoked_at,
                    "revocation_reason": "operator revoked",
                }
            ).model_dump(mode="python")
        )
        revocation = targeted_tool_grant_event(
            revoked,
            event_type=EventType.TARGETED_TOOL_GRANT_REVOKED,
            timestamp=revoked_at,
            outcome="revoked",
            event_id_suffix="revoked",
        )
        validate_targeted_tool_grant_revocation_evidence(revoked, revocation)
        with pytest.raises(ValueError, match="revocation time conflicts"):
            validate_targeted_tool_grant_revocation_evidence(
                revoked,
                revocation.model_copy(update={"timestamp": revoked_at + timedelta(seconds=1)}),
            )
        async for _event in stream:
            pass

    asyncio.run(run())


def test_sqlite_targeted_grant_reads_reject_indexed_state_corruption(tmp_path: Path) -> None:
    async def run() -> None:
        database = tmp_path / "targeted-corruption.sqlite"
        store = SQLiteSessionStore(database, public_authority_alias_codec=_codec())
        _app_instance, _provider, stream, _issued, record, _session = await _open_targeted_grant(
            store,
            session_id="targeted-sqlite-corruption",
        )
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                "UPDATE cayu_targeted_tool_grants SET used_calls = 1 WHERE grant_id = ?",
                (record.grant_id,),
            )
            connection.commit()
        finally:
            connection.close()
        with pytest.raises(ValueError, match="conflicts with indexed authority"):
            await store.load_targeted_tool_grant_state(record.session_id)
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                "UPDATE cayu_targeted_tool_grants "
                "SET record_json = json_set(record_json, '$.used_calls', 1) "
                "WHERE grant_id = ?",
                (record.grant_id,),
            )
            connection.commit()
        finally:
            connection.close()
        with pytest.raises(ValueError, match="call counter conflicts with durable uses"):
            await store.list_targeted_tool_grants(record.session_id)
        async for _event in stream:
            pass
        await store.close()

    asyncio.run(run())


def test_sqlite_event_pruning_retains_targeted_grant_retry_authority(tmp_path: Path) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(
            tmp_path / "targeted-pruning.sqlite",
            public_authority_alias_codec=_codec(),
        )
        (
            app_instance,
            _provider,
            stream,
            issued_event,
            record,
            session,
        ) = await _open_targeted_grant(
            store,
            session_id="targeted-sqlite-pruning",
        )
        request = _use_request(record, run_epoch=session.run_epoch)
        first = await store.bind_targeted_tool_grant_use(
            request,
            observed_at=record.issued_at + timedelta(seconds=1),
        )
        assert first.disposition is TargetedToolUseDisposition.BOUND

        await store.prune_events(
            before=record.expires_at + timedelta(days=1),
            session_id=record.session_id,
        )
        retained_types = {event.type for event in await store.load_events(record.session_id)}
        assert EventType.INTERACTION_STARTED in retained_types
        assert EventType.TARGETED_TOOL_GRANT_ISSUED in retained_types
        assert EventType.TARGETED_TOOL_REFERENCE_CONSUMED in retained_types

        reused = await store.issue_targeted_tool_grants(
            record.session_id,
            expected_run_epoch=session.run_epoch,
            records=(record,),
            events=(issued_event,),
        )
        assert reused.records[0].grant_id == record.grant_id
        rejoined = await store.bind_targeted_tool_grant_use(
            request,
            observed_at=record.issued_at + timedelta(seconds=2),
        )
        assert rejoined.disposition is TargetedToolUseDisposition.REJOINED
        revoked = await store.revoke_targeted_tool_grant(
            record.tool_ref,
            session_id=record.session_id,
            expected_run_epoch=session.run_epoch,
            reason="operator revoked",
            revoked_at=record.issued_at + timedelta(seconds=3),
        )
        assert revoked is not None

        await store.prune_events(
            before=record.expires_at + timedelta(days=1),
            session_id=record.session_id,
        )
        repeated = await store.revoke_targeted_tool_grant(
            record.tool_ref,
            session_id=record.session_id,
            expected_run_epoch=session.run_epoch,
            reason="operator revoked",
            revoked_at=record.issued_at + timedelta(seconds=4),
        )
        assert repeated == revoked

        async for _event in stream:
            pass
        await store.prune_events(
            before=record.issued_at + timedelta(seconds=30),
            session_id=record.session_id,
        )
        retained_types = {event.type for event in await store.load_events(record.session_id)}
        assert EventType.INTERACTION_COMPLETED in retained_types
        resume_stream = app_instance.resume(
            ResumeRequest(
                session_id=record.session_id,
                messages=[Message.text("user", "Start a separate interaction.")],
            )
        )
        assert (await anext(resume_stream)).type is EventType.INTERACTION_STARTED
        active_session = await store.load(record.session_id)
        assert active_session is not None
        expired_after_interaction = await store.bind_targeted_tool_grant_use(
            _use_request(record, run_epoch=active_session.run_epoch),
            observed_at=record.issued_at + timedelta(seconds=5),
        )
        assert expired_after_interaction.reason is TargetedToolUseRejectionReason.EXPIRED
        async for _event in resume_stream:
            pass
        await store.close()

    asyncio.run(run())


def test_sqlite_targeted_grant_contention_is_atomic_across_store_handles(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        database = tmp_path / "targeted-multi-handle.sqlite"
        codec = _codec()
        first_store = SQLiteSessionStore(database, public_authority_alias_codec=codec)
        second_store = SQLiteSessionStore(database, public_authority_alias_codec=codec)
        _app_instance, _provider, stream, _issued, record, session = await _open_targeted_grant(
            first_store,
            session_id="targeted-multi-handle",
            max_calls=4,
        )
        observed_at = record.issued_at + timedelta(seconds=1)
        requests = [
            _use_request(
                record,
                run_epoch=session.run_epoch,
                model_step_id=f"step-{index}",
                outer_tool_call_id=f"call-{index}",
                invocation_id=f"invocation-{index}",
                arguments_sha256=f"sha256:{index:064x}",
            )
            for index in range(12)
        ]
        results = await asyncio.gather(
            *(
                (first_store if index % 2 == 0 else second_store).bind_targeted_tool_grant_use(
                    request,
                    observed_at=observed_at,
                )
                for index, request in enumerate(requests)
            )
        )
        assert (
            sum(result.disposition is TargetedToolUseDisposition.BOUND for result in results) == 4
        )
        assert (
            sum(result.reason is TargetedToolUseRejectionReason.EXHAUSTED for result in results)
            == 8
        )
        snapshot = await second_store.load_targeted_tool_grant_state(record.session_id)
        assert snapshot.records[0].used_calls == 4
        assert len(snapshot.uses) == 4
        async for _event in stream:
            pass
        await second_store.close()
        await first_store.close()

    asyncio.run(run())


def test_sqlite_targeted_reference_rotation_backfills_the_new_active_alias(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        database = tmp_path / "targeted-key-rotation.sqlite"
        first_store = SQLiteSessionStore(
            database,
            public_authority_alias_codec=_rotation_codec(
                active_key_id="first",
                include_second=False,
            ),
        )
        _app_instance, _provider, stream, _issued, original, _session = await _open_targeted_grant(
            first_store,
            session_id="targeted-key-rotation",
        )
        async for _event in stream:
            pass
        await first_store.close()

        rotated_codec = _rotation_codec(active_key_id="second", include_second=True)
        rotated_store = SQLiteSessionStore(
            database,
            public_authority_alias_codec=rotated_codec,
        )
        try:
            [rotated] = await rotated_store.list_targeted_tool_grants(original.session_id)
            assert rotated.grant_id == original.grant_id
            assert rotated.tool_ref != original.tool_ref
            assert rotated.tool_ref == rotated_codec.encode(
                original.grant_id,
                field_name="tool_ref",
                session_id=original.session_id,
            )
            assert (
                await rotated_store.resolve_public_authority_alias(
                    rotated.tool_ref,
                    field_name="tool_ref",
                    scope_session_id=original.session_id,
                )
                == original.grant_id
            )
        finally:
            await rotated_store.close()

    asyncio.run(run())
