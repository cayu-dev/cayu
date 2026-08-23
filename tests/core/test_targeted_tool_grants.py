from __future__ import annotations

import asyncio
import base64
import io
import json
import secrets
import sqlite3
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
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    RecentTurnsContextPolicy,
    ResumeRequest,
    RetryPolicy,
    RunRequest,
    SessionRunFenced,
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
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelContextOverflowError, ModelProviderError
from cayu.runtime.tool_grants import (
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
        assert [tool["name"] for tool in provider.requests[0].tools] == ["remember"]
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
