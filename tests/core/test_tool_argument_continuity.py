"""Native audit/continuation separation, including durable store reopening."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from tests.core.test_tool_execution import _collect, _ScriptedProvider, _TestKnowledgeStore

from cayu.core import AgentSpec, EventType, ExecutionProfileBehaviorIdentity, Message, ToolCallPart
from cayu.environments import Environment, EnvironmentSpec
from cayu.runtime import (
    CayuApp,
    ForkSessionRequest,
    InMemorySessionStore,
    ResumeRequest,
    RunRequest,
)
from cayu.runtime._argument_continuity import (
    MAX_CALL_BYTES,
    MAX_ROUNDS,
    STORAGE_KEY,
    ArgumentContinuity,
    append_record,
    capture_arguments,
    materialize,
    private_read_scope,
    redact_continuity,
    require_private_key_access,
)
from cayu.runtime._runtime_records import ToolCallRequest
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.knowledge import RememberKnowledgeTool
from cayu.vaults import REDACTED_SECRET, SecretRedactor


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def continuity_backend(request, tmp_path):
    dsn = request.getfixturevalue("postgres_dsn") if request.param == "postgres" else None
    return request.param, tmp_path / "continuity.sqlite", dsn


@pytest.mark.parametrize("publish_arguments", [False, True])
@pytest.mark.parametrize("retain_arguments", [False, True])
@pytest.mark.parametrize(
    "fault",
    [None, "before_commit", "lost_ack", "invalid_private_commit", "model_promotion_lost_ack"],
)
def test_audit_and_model_argument_policies_are_independent(
    continuity_backend, publish_arguments, retain_arguments, fault
):
    backend, path, dsn = continuity_backend
    canary = "argument-continuity-probe-private-fact"
    arguments = {"text": canary, "kind": "fact"}

    def identity(component):
        return ExecutionProfileBehaviorIdentity(
            name=f"continuity-probe:{component}:{publish_arguments}:{retain_arguments}",
            behavior_version="1",
            implementation_version="1",
        )

    class ProbeProvider(_ScriptedProvider):
        @property
        def execution_profile_identity(self):
            return identity("provider")

        async def stream(self, request):
            from cayu.providers.base import ModelStreamEvent
            from cayu.providers.openai import openai_response_events

            self.requests.append(request)
            if len(self.requests) == 1 and self._tool_calls:
                output = [
                    {
                        "type": "reasoning",
                        "id": "rs_probe",
                        "encrypted_content": "opaque",
                        "summary": [],
                    }
                ]
                output.extend(
                    {
                        "type": "function_call",
                        "id": "fc_probe",
                        "call_id": call_id,
                        "name": name,
                        "arguments": json.dumps(args),
                        "status": "completed",
                    }
                    for call_id, name, args in self._tool_calls
                )
                for event in openai_response_events(
                    {"id": "resp_probe", "status": "completed", "output": output}
                ):
                    yield event
                return
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class ProbeRemember(RememberKnowledgeTool):
        @property
        def retain_arguments_for_model(self):
            return retain_arguments

        @property
        def execution_profile_identity(self):
            return identity("remember")

        @property
        def _publish_arguments(self) -> bool:
            return publish_arguments

    class FaultMixin:
        faulted = False

        async def _promote_model_completion_stage_atomic(self, **kwargs):
            result = await super()._promote_model_completion_stage_atomic(**kwargs)
            if (
                fault == "model_promotion_lost_ack"
                and retain_arguments
                and not publish_arguments
                and not self.faulted
                and not result.replayed
            ):
                checkpoint = await self.load_checkpoint(result.session.id)
                assert (
                    checkpoint["pending_tool_round"]["assistant_publication"]["argument_continuity"]
                    is not None
                )
                self.faulted = True
                raise ConnectionError("injected model promotion acknowledgement loss")
            return result

        async def _publish_runtime_publication_atomic(self, prepared, **kwargs):
            inject = (
                fault in {"before_commit", "lost_ack", "invalid_private_commit"}
                and not self.faulted
                and prepared.request.argument_continuity is not None
            )
            if inject:
                self.faulted = True
                if fault == "before_commit":
                    raise RuntimeError("injected pre-commit failure")
                if fault == "invalid_private_commit":
                    invalid = prepared.request.argument_continuity.model_copy(
                        update={"profile": "f" * 64}
                    )
                    prepared = replace(
                        prepared,
                        request=prepared.request.model_copy(
                            update={"argument_continuity": invalid}
                        ),
                    )
            result = await super()._publish_runtime_publication_atomic(prepared, **kwargs)
            if inject:
                raise RuntimeError("injected acknowledgement loss")
            return result

    class MemoryStore(FaultMixin, InMemorySessionStore):
        invocation_lifecycle_command_version = 1

    class SQLiteStore(FaultMixin, SQLiteSessionStore):
        invocation_lifecycle_command_version = 1

    def make_store():
        if backend == "memory":
            return MemoryStore()
        if backend == "sqlite":
            return SQLiteStore(path)
        from cayu.storage.postgres import PostgresSessionStore

        class PostgresStore(FaultMixin, PostgresSessionStore):
            invocation_lifecycle_command_version = 1

        return PostgresStore(dsn, schema_mode=SchemaMode.CREATE, min_size=1, max_size=2)

    def make_app(store, knowledge, provider):
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="probe", execution_profile_identity=identity("environment")),
                knowledge_store=knowledge,
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[ProbeRemember()])
        return app

    def retained_call(request):
        calls = [
            part
            for message in request.messages
            for part in message.content
            if isinstance(part, ToolCallPart) and part.tool_call_id == "probe-call"
        ]
        assert len(calls) == 1
        return calls[0]

    async def run():
        store = make_store()
        knowledge = _TestKnowledgeStore()
        provider = ProbeProvider([("probe-call", "remember_knowledge", arguments)])
        prior_requests = []
        session_id = f"probe-{backend}-{publish_arguments}-{retain_arguments}-{fault}"
        try:
            app = make_app(store, knowledge, provider)
            events = await _collect(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Run the capture probe.")],
                ),
            )
            if (
                fault in {"before_commit", "invalid_private_commit"}
                and retain_arguments
                and not publish_arguments
            ):
                with private_read_scope():
                    assert await store.load_session_operation(session_id, STORAGE_KEY) is None
                assert "pending_tool_round" in await store.load_checkpoint(session_id)
                prior_requests = list(provider.requests)
                if backend != "memory":
                    await store.close()
                    store = make_store()
                store.faulted = True
                provider = ProbeProvider([])
                app = make_app(store, knowledge, provider)
                events.extend(
                    [
                        event
                        async for event in app.resume(
                            ResumeRequest(
                                session_id=session_id,
                                messages=[
                                    Message.text("user", "Recover the committed tool outcome.")
                                ],
                            )
                        )
                    ]
                )
            assert any(event.type == EventType.SESSION_COMPLETED for event in events)
            requests = [*prior_requests, *provider.requests]
            assert len(requests) == 2
            expected = arguments if publish_arguments or retain_arguments else {}
            assert retained_call(requests[1]).arguments == expected
            from cayu.providers.openai import build_openai_payload

            payload = build_openai_payload(requests[1], stream=True, reasoning_state="inline")
            native_calls = [
                item for item in payload["input"] if item.get("type") == "function_call"
            ]
            assert len(native_calls) == 1
            assert json.loads(native_calls[0]["arguments"]) == expected
            assert any(item.get("encrypted_content") == "opaque" for item in payload["input"])
            transcript = await store.load_transcript(session_id)
            checkpoint = await store.load_checkpoint(session_id)
            assert "pending_tool_round" not in (checkpoint or {})
            assert (canary in repr(transcript)) is publish_arguments
            terminals = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
            assert len(terminals) == 1
            assert (canary in repr(terminals)) is publish_arguments
            assert store.faulted is (
                fault is not None and retain_arguments and not publish_arguments
            )
            assert canary not in repr(checkpoint)
            exported = await store.load_session_export_snapshot(session_id)
            assert (canary in repr(exported)) is publish_arguments
            with pytest.raises(ValueError, match="runtime-owned"):
                await store.load_session_operation(session_id, STORAGE_KEY)
            with private_read_scope():
                private = await store.load_session_operation(session_id, STORAGE_KEY)
            if retain_arguments and not publish_arguments:
                assert len(private["records"]) == 1
            else:
                assert private is None

            # Reconstruct the app/provider and, for durable backends, the store.
            # This is a new interaction after completion, not a simulated crash.
            if backend != "memory":
                await store.close()
                store = make_store()
            store.faulted = True
            resumed_provider = ProbeProvider(
                [("second-call", "remember_knowledge", {"text": "second retained input"})]
            )
            resumed_app = make_app(store, knowledge, resumed_provider)
            resumed_events = [
                event
                async for event in resumed_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "Continue with the next capture.")],
                    )
                )
            ]
            assert any(event.type == EventType.SESSION_COMPLETED for event in resumed_events)
            assert retained_call(resumed_provider.requests[0]).arguments == expected
            assert (canary in repr(await store.load_transcript(session_id))) is publish_arguments
            with private_read_scope():
                after_resume = await store.load_session_operation(session_id, STORAGE_KEY)
            if retain_arguments and not publish_arguments:
                assert len(after_resume["records"]) == 2
            if fault is None and retain_arguments and not publish_arguments:
                child_id = f"{session_id}-child"
                fork_events = [
                    event
                    async for event in resumed_app.fork_session(
                        ForkSessionRequest(source_session_id=session_id, session_id=child_id)
                    )
                ]
                assert fork_events
                with private_read_scope():
                    assert await store.load_session_operation(child_id, STORAGE_KEY) is None
                child_events = [
                    event
                    async for event in resumed_app.resume(
                        ResumeRequest(
                            session_id=child_id,
                            messages=[Message.text("user", "Continue in the fork.")],
                        )
                    )
                ]
                assert child_events[-1].type == EventType.SESSION_COMPLETED
                assert retained_call(resumed_provider.requests[-1]).arguments == {}
                await store.delete_session(child_id)
                await store.delete_session(session_id)
                with private_read_scope(), pytest.raises(KeyError):
                    await store.load_session_operation(session_id, STORAGE_KEY)
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())


def test_later_vault_discovery_redacts_all_private_calls_in_the_round():
    from pydantic import SecretStr

    from cayu.core.tools import Tool, ToolResult, ToolSpec
    from cayu.vaults import ResolvedSecret, SecretRef, Vault

    secret = "late-discovered-continuity-secret"

    class DynamicVault(Vault):
        async def get(self, name, *, scope=None):
            return SecretRef(name=name)

        async def resolve(self, ref, *, scope=None):
            return ResolvedSecret(name=ref.name, value=SecretStr(secret))

    class PrivateTool(Tool):
        spec = ToolSpec(
            name="private_tool",
            description="Exercise late discovery.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}, "resolve": {"type": "boolean"}},
            },
        )

        @property
        def _publish_arguments(self):
            return False

        @property
        def retain_arguments_for_model(self):
            return True

        async def run(self, ctx, args):
            if args.get("resolve"):
                await ctx.vault.resolve(SecretRef(name="api_key"))
            return ToolResult(content="accepted")

    async def run():
        store = InMemorySessionStore()
        provider = _ScriptedProvider(
            [
                ("first", "private_tool", {"text": secret}),
                ("second", "private_tool", {"resolve": True}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="dynamic"), vault=DynamicVault()), default=True
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[PrivateTool()])
        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="late-secret",
                messages=[Message.text("user", "Exercise late secret discovery.")],
            ),
        )
        assert events[-1].type == EventType.SESSION_COMPLETED
        assert secret not in repr(events)
        assert secret not in repr(provider.requests[1].messages)
        calls = [
            part
            for message in provider.requests[1].messages
            for part in message.content
            if isinstance(part, ToolCallPart)
        ]
        assert calls[0].arguments == {"text": REDACTED_SECRET}
        with private_read_scope():
            private = await store.load_session_operation("late-secret", STORAGE_KEY)
        assert private is not None
        assert secret not in repr(private)

    asyncio.run(run())


def _private_fixture(index=0):
    call_id = f"call-{index}"
    message = Message(
        role="assistant",
        content=[
            ToolCallPart(
                tool_call_id=call_id,
                tool_name="private_tool",
                arguments={},
                tool_round_id=f"tround_{index:032x}",
                model_step_id=f"mstep_{index:032x}",
                model_attempt_id=f"matt_{index:032x}",
            )
        ],
    )
    continuity = ArgumentContinuity(
        nonce="a" * 32, profile="b" * 64, arguments={call_id: {"text": f"private-{index}"}}
    )
    return message, continuity


class _PrivateReadProbe:
    def __init__(self, raw):
        self.raw = raw
        self.reads = 0

    async def load_session_operation(self, session_id, key):
        assert session_id == "session"
        assert key == STORAGE_KEY
        require_private_key_access(key, read=True)
        self.reads += 1
        return deepcopy(self.raw)


@pytest.mark.parametrize("case", ["ordinary", "retained", "oversized"])
def test_unsupported_store_stops_only_when_retention_is_needed(case):
    from cayu.tools.knowledge import ListKnowledgeTool

    class UnsupportedStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        supports_private_argument_continuity = False

    async def run():
        store = UnsupportedStore()
        provider = _ScriptedProvider(
            [
                (
                    "call",
                    "remember_knowledge",
                    {"text": "x" * MAX_CALL_BYTES if case == "oversized" else "not-dispatched"},
                )
            ]
            if case != "ordinary"
            else [("call", "list_knowledge", {})]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="unsupported-store-probe"),
                knowledge_store=_TestKnowledgeStore(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RememberKnowledgeTool(), ListKnowledgeTool()],
        )
        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="unsupported",
                messages=[Message.text("user", "Remember.")],
            ),
        )
        assert events[-1].type == (
            EventType.SESSION_FAILED if case == "retained" else EventType.SESSION_COMPLETED
        )
        assert any(event.type == EventType.TOOL_CALL_STARTED for event in events) is (
            case != "retained"
        )
        assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in events) is (
            case != "retained"
        )
        assert not any(event.type == EventType.TOOL_CALL_FAILED for event in events)
        with private_read_scope():
            assert await store.load_session_operation("unsupported", STORAGE_KEY) is None

    asyncio.run(run())


@pytest.mark.parametrize(
    "boundary",
    ["same", "incarnation", "profile", "scope", "call", "compacted", "duplicate", "disabled"],
)
def test_private_materialization_requires_exact_retained_authority(boundary):
    async def run():
        from cayu.storage import KnowledgeAccessScope

        session = SimpleNamespace(id="session", instance_id="incarnation")
        message, continuity = _private_fixture()
        raw = append_record(
            None,
            continuity=continuity,
            session=session,
            messages=[message],
            request_digest="d" * 64,
        )
        store = _PrivateReadProbe(raw)
        profile = continuity.profile
        names = frozenset({"private_tool"})
        scope = None
        if boundary == "incarnation":
            session = SimpleNamespace(id="session", instance_id="new-incarnation")
        if boundary == "profile":
            profile = "c" * 64
        if boundary == "scope":
            scope = KnowledgeAccessScope.privileged()
        if boundary == "call":
            message = message.model_copy(
                update={
                    "content": (
                        message.content[0].model_copy(update={"tool_call_id": "different"}),
                    )
                }
            )
        messages = [] if boundary == "compacted" else [message]
        if boundary == "duplicate":
            messages *= 2
        if boundary == "disabled":
            names = frozenset()
        result = await materialize(
            store=store,
            session=session,
            profile=profile,
            messages=messages,
            names=names,
            redactor=SecretRedactor(),
            scope=scope,
        )
        if boundary == "same":
            assert result[0].content[0].arguments == {"text": "private-0"}
            result[0].content[0].arguments["text"] = "mutated"
            assert "mutated" not in repr(raw)
        else:
            assert result == messages
        assert store.reads == (0 if boundary in {"compacted", "duplicate", "disabled"} else 1)
        assert message.content[0].arguments == {}

    asyncio.run(run())


@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("redact", [False, True])
def test_private_arguments_reach_native_openai_payload_without_changing_audit(retain, redact):
    from cayu.core import ProviderStatePart
    from cayu.providers.base import ModelRequest
    from cayu.providers.openai import build_openai_payload

    async def run():
        session = SimpleNamespace(id="session", instance_id="incarnation")
        message, continuity = _private_fixture()
        reasoning = ProviderStatePart(
            provider="openai",
            state={
                "type": "reasoning",
                "id": "rs_test",
                "encrypted_content": "opaque",
                "summary": [],
            },
        )
        native_call = ProviderStatePart(
            provider="openai",
            state={
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call-0",
                "name": "private_tool",
                "arguments": "{}",
                "status": "completed",
            },
        )
        message = message.model_copy(update={"content": (*message.content, reasoning, native_call)})
        before = message.model_dump(mode="json")
        raw = append_record(
            None,
            continuity=continuity,
            session=session,
            messages=[message],
            request_digest="d" * 64,
        )
        restored = await materialize(
            store=_PrivateReadProbe(raw),
            session=session,
            profile=continuity.profile,
            messages=[message],
            names=frozenset({"private_tool"}) if retain else frozenset(),
            redactor=SecretRedactor("private-0") if redact else SecretRedactor(),
        )
        payload = build_openai_payload(
            ModelRequest(model="test", messages=restored), stream=True, reasoning_state="inline"
        )
        calls = [item for item in payload["input"] if item["type"] == "function_call"]
        expected = {"text": REDACTED_SECRET if redact else "private-0"} if retain else {}
        assert len(calls) == 1
        assert json.loads(calls[0]["arguments"]) == expected
        assert calls[0]["id"] == "fc_test"
        assert payload["input"][0] == reasoning.state
        assert message.model_dump(mode="json") == before
        assert native_call.state["arguments"] == "{}"
        if retain:
            restored[0].content[-1].state["arguments"] = "mutated"
            assert message.model_dump(mode="json") == before
            assert "mutated" not in repr(raw)

    asyncio.run(run())


@pytest.mark.parametrize(
    "mismatch",
    ["provider", "name", "call_id", "type", "arguments", "other_message", "malformed_id"],
)
def test_private_provider_projection_requires_same_message_and_exact_call(mismatch):
    from cayu.core import ProviderStatePart

    async def run():
        session = SimpleNamespace(id="session", instance_id="incarnation")
        message, continuity = _private_fixture()
        state = {
            "type": "function_call",
            "call_id": "call-0",
            "name": "private_tool",
            "arguments": "{}",
        }
        if mismatch in {"name", "call_id", "type", "arguments"}:
            state[mismatch] = "unrelated"
        if mismatch == "malformed_id":
            state["call_id"] = []
        native = ProviderStatePart(
            provider="other" if mismatch == "provider" else "openai", state=state
        )
        if mismatch == "other_message":
            messages = [message, Message(role="assistant", content=[native])]
        else:
            message = message.model_copy(update={"content": (*message.content, native)})
            messages = [message]
        raw = append_record(
            None,
            continuity=continuity,
            session=session,
            messages=[message],
            request_digest="d" * 64,
        )
        restored = await materialize(
            store=_PrivateReadProbe(raw),
            session=session,
            profile=continuity.profile,
            messages=messages,
            names=frozenset({"private_tool"}),
            redactor=SecretRedactor(),
        )
        assert restored[0].content[0].arguments == {"text": "private-0"}
        assert restored[-1].content[-1] == native

    asyncio.run(run())


def test_retention_is_bounded_and_corruption_fails_closed():
    async def run():
        session = SimpleNamespace(id="session", instance_id="incarnation")
        raw = None
        for index in range(MAX_ROUNDS + 2):
            message, continuity = _private_fixture(index)
            raw = append_record(
                raw,
                continuity=continuity,
                session=session,
                messages=[message],
                request_digest="d" * 64,
            )
        assert len(raw["records"]) == MAX_ROUNDS
        oldest, _ = _private_fixture(0)
        store = _PrivateReadProbe(raw)
        kwargs = dict(
            store=store,
            session=session,
            profile="b" * 64,
            messages=[oldest],
            names=frozenset({"private_tool"}),
            redactor=SecretRedactor(),
        )
        assert await materialize(**kwargs) == [oldest]
        store.raw["records"][-1]["continuity"]["arguments"] = {
            "bad": {"secret": "corruption-canary"}
        }
        with pytest.raises(ValueError, match="integrity") as error:
            await materialize(**kwargs)
        assert "corruption-canary" not in str(error.value)

    asyncio.run(run())


def test_capture_is_bounded_original_and_progressively_redacted():
    calls = [ToolCallRequest(id="call", name="private_tool", arguments={"text": "late-secret"})]
    kwargs = dict(names=frozenset({"private_tool"}), profile="b" * 64, redactor=SecretRedactor())
    captured = capture_arguments(calls, **kwargs)
    assert "late-secret" not in repr(captured)
    calls[0].arguments["text"] = "effective-arguments"
    assert captured.arguments == {"call": {"text": "late-secret"}}
    redacted = redact_continuity(captured, SecretRedactor("late-secret"))
    assert redacted.arguments == {"call": {"text": REDACTED_SECRET}}
    assert (
        capture_arguments(
            [
                ToolCallRequest(
                    id="large", name="private_tool", arguments={"text": "x" * MAX_CALL_BYTES}
                )
            ],
            **kwargs,
        )
        is None
    )
    assert capture_arguments(calls, **{**kwargs, "names": frozenset()}) is None


def test_targeted_grant_calls_do_not_retain_resolved_arguments():
    call = ToolCallRequest(
        id="targeted-call",
        name="private_tool",
        arguments={"text": "resolved-inner-input"},
        targeted_tool_grant_id="sha256:" + "a" * 64,
    )
    assert (
        capture_arguments(
            [call],
            names=frozenset({"private_tool"}),
            profile="b" * 64,
            redactor=SecretRedactor(),
        )
        is None
    )


def test_unsealed_secret_scope_discards_private_continuity():
    from cayu.runtime._assistant_tool_round_publication import AssistantToolRoundPublication
    from cayu.runtime._tool_round_recovery import _updated_assistant_publication

    message, continuity = _private_fixture()
    pending = AssistantToolRoundPublication(
        state="pending", message=message, argument_continuity=continuity
    )
    blocked = _updated_assistant_publication(
        pending,
        expected_ids={"call-0"},
        tool_call_id="call-0",
        redactor=SecretRedactor(),
        unsafe_output=True,
        cover_call=True,
    )
    assert blocked.state == "blocked"
    assert blocked.argument_continuity is None
    sealed = _updated_assistant_publication(
        pending,
        expected_ids={"call-0"},
        tool_call_id="call-0",
        redactor=SecretRedactor("private-0"),
        unsafe_output=False,
        cover_call=True,
    )
    assert sealed.state == "ready"
    assert sealed.argument_continuity.arguments["call-0"]["text"] == REDACTED_SECRET


@pytest.mark.parametrize("compact", [False, True])
def test_generic_tool_continuity_is_materialized_after_context_selection(compact):
    from cayu.core.tools import Tool, ToolResult, ToolSpec
    from cayu.runtime.context import ContextPolicy

    class PrivateTool(Tool):
        spec = ToolSpec(
            name="private_tool",
            description="Accept text privately.",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        )

        @property
        def _publish_arguments(self):
            return False

        @property
        def retain_arguments_for_model(self):
            return True

        async def run(self, ctx, args):
            return ToolResult(content="accepted")

    class SelectionPolicy(ContextPolicy):
        async def build(self, request):
            assert "context-selection-canary" not in repr(request.messages)
            if compact:
                return [message for message in request.messages if message.role == "user"]
            return request.messages

    async def run():
        store = InMemorySessionStore()
        provider = _ScriptedProvider(
            [("private-call", "private_tool", {"text": "context-selection-canary"})]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[PrivateTool()],
            context_policy=SelectionPolicy(),
        )
        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="selection",
                messages=[Message.text("user", "Run the private tool.")],
            ),
        )
        assert events[-1].type == EventType.SESSION_COMPLETED
        assert ("context-selection-canary" in repr(provider.requests[1].messages)) is (not compact)
        assert "context-selection-canary" not in repr(events)
        assert "context-selection-canary" not in repr(await store.load_transcript("selection"))
        if not compact:
            from cayu.runtime.request_footprints import analyze_request_context_pressure

            actual = provider.requests[1]
            public = actual.model_copy(
                update={
                    "messages": [
                        message.model_copy(
                            update={
                                "content": tuple(
                                    part.model_copy(update={"arguments": {}})
                                    if isinstance(part, ToolCallPart)
                                    else part
                                    for part in message.content
                                )
                            }
                        )
                        for message in actual.messages
                    ]
                }
            )
            assert (
                analyze_request_context_pressure(
                    actual, provider=provider
                ).estimated_context_input_tokens
                > analyze_request_context_pressure(
                    public, provider=provider
                ).estimated_context_input_tokens
            )

    asyncio.run(run())
