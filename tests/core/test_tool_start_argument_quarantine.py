from __future__ import annotations

import asyncio
import io
import json
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import rebind_test_invocation
from tests.core._workload_secret_support import (
    FakeProvider,
    RequireApprovalPolicy,
    collect_events,
    collect_tool_approval_events,
)

from cayu import (
    ArtifactExternalizingToolResultPolicy,
    CayuConfig,
    LocalArtifactStore,
    PostgresSessionStore,
    SQLiteSessionStore,
    ToolExecutionConfig,
)
from cayu.core import (
    AgentSpec,
    EventType,
    ExecutionProfileBehaviorIdentity,
    Message,
    ToolEffect,
)
from cayu.core.messages import ProviderStatePart, ThinkingPart, ToolCallPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.providers import (
    ModelStreamEvent,
    build_chat_completions_payload,
    build_openai_payload,
)
from cayu.proxies import PassthroughProxy
from cayu.runners import LocalRunner
from cayu.runtime import (
    AfterToolCallDecision,
    BeforeToolCallDecision,
    BeforeToolCallHookContext,
    CayuApp,
    EventQuery,
    EventWatcher,
    ExecutionProfileComponentClass,
    ExecutionProfileMismatchError,
    IncompleteSessionRecoveryRequest,
    InMemoryEventSink,
    InMemoryEventWatcherStore,
    InMemorySessionStore,
    InterruptSessionRequest,
    PendingActionQuery,
    ResumeRequest,
    RunLimits,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    SessionStatus,
    StructuredOutputSpec,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolCallHookContext,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    UserInputResponse,
)
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_support
from cayu.runtime._runtime_records import ToolCallOutcome, ToolCallRequest
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.storage.jsonl_export import export_sessions
from cayu.storage.migrations import SchemaMode
from cayu.tools.commands import (
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    CommandRequest,
    ExecCommandTool,
)
from cayu.tools.user_input import UserInputTool
from cayu.vaults import REDACTED_SECRET, SecretRedactor, SecretRef, StaticVault


def _test_behavior_identity(name: str) -> ExecutionProfileBehaviorIdentity:
    return ExecutionProfileBehaviorIdentity(
        name=f"tests:tool-start-quarantine:{name}",
        behavior_version="1",
        implementation_version="1",
    )


def _portable_environment_spec(name: str) -> EnvironmentSpec:
    return EnvironmentSpec(
        name=name,
        execution_profile_identity=_test_behavior_identity(f"environment:{name}"),
    )


class _ResolveAfterStartTool(Tool):
    spec = ToolSpec(
        name="resolve_after_start",
        description="Resolve an invocation secret after the durable start boundary.",
        input_schema={"type": "object", "additionalProperties": True},
    )

    def __init__(self, *, secret_source: str = "vault") -> None:
        super().__init__(
            self.spec.model_copy(
                update={
                    "execution_profile_identity": _test_behavior_identity(
                        f"{type(self).__name__}:{secret_source}"
                    )
                }
            )
        )
        self.secret_source = secret_source
        self.arguments: list[dict[str, Any]] = []

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        self.arguments.append(args)
        if self.secret_source == "vault":
            assert ctx.vault is not None
            await ctx.vault.resolve(SecretRef(name="api_key"))
        else:
            assert self.secret_source == "proxy"
            assert ctx.proxy is not None
            await ctx.proxy.resolve(SecretRef(name="api_key"))
        return ToolResult(content="done")


class _FailAfterResolutionTool(_ResolveAfterStartTool):
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        await super().run(ctx, args)
        raise RuntimeError("intentional tool failure")


class _TimeoutAfterResolutionTool(_ResolveAfterStartTool):
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        await super().run(ctx, args)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _CancellableAfterResolutionTool(_ResolveAfterStartTool):
    def __init__(self, *, cleanup_fails: bool = False) -> None:
        super().__init__()
        self.cleanup_fails = cleanup_fails
        self.dispatched = asyncio.Event()
        self.never_complete = asyncio.Event()

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        await super().run(ctx, args)
        self.dispatched.set()
        try:
            await self.never_complete.wait()
        finally:
            if self.cleanup_fails:
                raise RuntimeError("intentional cleanup failure")
        raise AssertionError("unreachable")


class _ResolveNamedSecretTool(Tool):
    spec = ToolSpec(
        name="resolve_named_secret",
        description="Resolve the named invocation secret.",
        input_schema={"type": "object", "additionalProperties": True},
        execution_profile_identity=_test_behavior_identity("resolve-named-secret-tool"),
    )

    def __init__(self) -> None:
        super().__init__()
        self.arguments: list[dict[str, Any]] = []

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        self.arguments.append(args)
        assert ctx.vault is not None
        await ctx.vault.resolve(SecretRef(name=args["ref"]))
        return ToolResult(content=f"resolved {args['ref']}")


class _FailFirstTerminalEventStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.failed_terminal = False

    async def append_events(self, session_id: str, events: list[Any]) -> None:
        if not self.failed_terminal and any(
            event.type
            in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
            }
            for event in events
        ):
            self.failed_terminal = True
            raise RuntimeError("simulated terminal append failure")
        await super().append_events(session_id, events)


class _CommitFirstStagedTerminalThenRaiseStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.lost_stage_acknowledgement = False

    async def transform_checkpoint(self, session_id, checkpoint_transform):
        await super().transform_checkpoint(session_id, checkpoint_transform)
        checkpoint = await self.load_checkpoint(session_id)
        pending_payload = None
        if checkpoint is not None:
            pending_payload = checkpoint.get("pending_tool_round") or checkpoint.get(
                "pending_user_input"
            )
        staged_terminals = (
            []
            if not isinstance(pending_payload, dict)
            else pending_payload.get("staged_terminals", [])
        )
        if (
            not self.lost_stage_acknowledgement
            and isinstance(staged_terminals, list)
            and len(staged_terminals) == 1
        ):
            self.lost_stage_acknowledgement = True
            raise RuntimeError("simulated staged-terminal acknowledgement loss")


class _CommitFirstTerminalEventThenRaiseStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.lost_terminal_acknowledgement = False

    async def append_events(self, session_id: str, events: list[Any]) -> None:
        await super().append_events(session_id, events)
        if not self.lost_terminal_acknowledgement and any(
            event.type
            in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
                EventType.TOOL_CALL_APPROVAL_DENIED,
            }
            for event in events
        ):
            self.lost_terminal_acknowledgement = True
            raise RuntimeError("simulated terminal-event acknowledgement loss")


class _FailFirstBlockedAssistantProjectionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.failed_projection = False

    async def transform_checkpoint(self, session_id, checkpoint_transform):
        def fail_before_commit(session, checkpoint):
            transformed = checkpoint_transform(session, checkpoint)
            pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(transformed)
            publication = None if pending_round is None else pending_round.assistant_publication
            if (
                not self.failed_projection
                and publication is not None
                and publication.state == "blocked"
            ):
                self.failed_projection = True
                raise RuntimeError("simulated assistant projection persistence failure")
            return transformed

        await super().transform_checkpoint(session_id, fail_before_commit)


class _RejectBlockedAssistantProjectionSQLiteStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, path) -> None:
        super().__init__(path)
        self.failed_projections = 0

    async def transform_checkpoint(self, session_id, checkpoint_transform):
        def fail_before_commit(session, checkpoint):
            transformed = checkpoint_transform(session, checkpoint)
            pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(transformed)
            publication = None if pending_round is None else pending_round.assistant_publication
            if publication is not None and publication.state == "blocked":
                self.failed_projections += 1
                raise RuntimeError("simulated persistent projection failure")
            return transformed

        await super().transform_checkpoint(session_id, fail_before_commit)


class _ObserveResolvedSecretTool(_ResolveAfterStartTool):
    def __init__(self, *, secret_source: str = "vault") -> None:
        super().__init__(secret_source=secret_source)
        self.resolution_returned = False

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        result = await super().run(ctx, args)
        self.resolution_returned = True
        return result


class _CaptureInterruptedHook(RuntimeHook):
    def __init__(self) -> None:
        self.terminal_events: list[Any] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _test_behavior_identity("capture-interrupted-hook")

    async def after_session_interrupted(self, context: RuntimeHookContext) -> None:
        self.terminal_events.append(context.terminal_event)


class _CaptureAfterToolArgumentsHook(RuntimeHook):
    def __init__(self) -> None:
        self.arguments: list[dict[str, Any]] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _test_behavior_identity("capture-after-tool-arguments-hook")

    async def after_tool_call(self, context: ToolCallHookContext) -> None:
        self.arguments.append(context.arguments)


class _CaptureAfterToolResultsHook(RuntimeHook):
    def __init__(self) -> None:
        self.results: list[ToolResult] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _test_behavior_identity("capture-after-tool-results-hook")

    async def after_tool_call(self, context: ToolCallHookContext) -> None:
        self.results.append(context.result)


async def _run_late_resolved_secret_quarantine_scenario() -> None:
    secret = "late-tool-start-secret-canary"
    arguments = {
        "provided": secret,
        f"prefix-{secret}-suffix": {
            "nested": f"before-{secret}-after",
        },
    }
    store = InMemorySessionStore()
    sink = InMemoryEventSink()
    watcher_store = InMemoryEventWatcherStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_late_secret",
                    name="resolve_after_start",
                    arguments=arguments,
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    tool = _ResolveAfterStartTool()
    app = CayuApp(
        session_store=store,
        event_sinks=[sink],
        event_watcher_store=watcher_store,
        secret_redactor=SecretRedactor("finalized"),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = await collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id="late-secret-quarantine",
            messages=[Message.text("user", "run")],
        ),
    )

    assert tool.arguments == [arguments]
    started = next(event for event in events if event.type is EventType.TOOL_CALL_STARTED)
    assert started.payload["arguments_state"] == "quarantined"
    assert "arguments" not in started.payload

    completed = next(
        (event for event in events if event.type is EventType.TOOL_CALL_COMPLETED),
        None,
    )
    assert completed is not None, ([event.type for event in events], events[-1].payload)
    assert completed.payload["arguments_state"] == "finalized"
    assert secret not in repr(completed.payload)
    assert REDACTED_SECRET in repr(completed.payload["arguments"])

    transcript = await store.load_transcript("late-secret-quarantine")
    tool_call_parts = [
        part for message in transcript for part in message.content if type(part) is ToolCallPart
    ]
    assert len(tool_call_parts) == 1
    assert secret not in repr(tool_call_parts[0].arguments)
    assert secret not in repr(provider.requests[1].messages)
    assert secret not in repr(events)
    assert secret not in repr(sink.events)
    observed = []
    await app.run_event_watchers(
        [
            EventWatcher(
                name="late-secret-start-watcher",
                query=EventQuery(
                    session_id="late-secret-quarantine",
                    event_type=EventType.TOOL_CALL_STARTED,
                ),
                handler=observed.append,
            )
        ]
    )
    assert secret not in repr(observed)


def test_late_resolved_secret_is_quarantined_until_terminal_projection() -> None:
    asyncio.run(_run_late_resolved_secret_quarantine_scenario())


def test_provider_state_uses_complete_round_secret_projection() -> None:
    async def run() -> None:
        secret = "late-openai-provider-state-secret-canary"
        provider_state_target_sha256 = "a" * 64
        provider_state_target = {
            "protocol": "openai-chat-completions",
            "protocol_version": 1,
            "version": 1,
            "sha256": provider_state_target_sha256,
        }
        arguments = {"provided": secret, "nested": {"token": secret}}
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_openai_provider_state",
                        name="resolve_after_start",
                        arguments=arguments,
                    ),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "provider_state": [
                                {
                                    "provider": "openai",
                                    "state": {
                                        "type": "function_call",
                                        "id": "fc_safe",
                                        "call_id": "call_openai_provider_state",
                                        "name": "resolve_after_start",
                                        "arguments": (
                                            '{"nested":{"token":"'
                                            + secret
                                            + '"},"provided":"'
                                            + secret
                                            + '"}'
                                        ),
                                        "status": "completed",
                                    },
                                },
                                {
                                    "provider": "openai",
                                    "state": {"type": "response_ref", "id": "resp_safe"},
                                },
                                {
                                    "provider": "chat_completions",
                                    "state": {
                                        "type": "tool_call_extra_content",
                                        "version": 1,
                                        "target": provider_state_target,
                                        "tool_call_id": "call_openai_provider_state",
                                        "extra_content": {
                                            "google": {"thought_signature": "signature-safe"}
                                        },
                                    },
                                },
                            ],
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        tool = _ResolveAfterStartTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                vault=StaticVault({"api_key": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="late-openai-provider-state",
                messages=[Message.text("user", "run")],
            ),
        )

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert tool.arguments == [arguments]
        transcript = await store.load_transcript("late-openai-provider-state")
        assert secret not in repr(events)
        assert secret not in repr(transcript)
        assert secret not in repr(provider.requests[1].messages)
        assistant_message = next(
            message
            for message in transcript
            if any(type(part) is ToolCallPart for part in message.content)
        )
        provider_states = [
            part for part in assistant_message.content if type(part) is ProviderStatePart
        ]
        function_call_state = next(
            part.state
            for part in provider_states
            if part.provider == "openai" and part.state.get("type") == "function_call"
        )
        assert function_call_state["arguments"] == (
            '{"nested":{"token":"[REDACTED_SECRET]"},"provided":"[REDACTED_SECRET]"}'
        )
        openai_payload = build_openai_payload(provider.requests[1])
        assert secret not in repr(openai_payload)
        outbound_function_call = next(
            item for item in openai_payload["input"] if item.get("type") == "function_call"
        )
        assert outbound_function_call["arguments"] == function_call_state["arguments"]
        assert any(
            part.provider == "openai" and part.state == {"type": "response_ref", "id": "resp_safe"}
            for part in provider_states
        )
        assert any(
            part.provider == "chat_completions"
            and part.state["extra_content"]["google"]["thought_signature"] == "signature-safe"
            for part in provider_states
        )
        chat_payload = build_chat_completions_payload(
            provider.requests[1],
            provider_state_target_sha256=provider_state_target_sha256,
        )
        assert secret not in repr(chat_payload)
        assert "signature-safe" in repr(chat_payload)
        exported = io.StringIO()
        assert await export_sessions(store, stream=exported) == 1
        assert secret not in exported.getvalue()

    asyncio.run(run())


@pytest.mark.parametrize("secret_in_key", [False, True])
def test_secret_bearing_opaque_provider_state_fences_tool_round_publication(
    secret_in_key: bool,
) -> None:
    async def run() -> None:
        secret = "late-opaque-provider-state-secret-canary"
        opaque_state = {secret: "safe"} if secret_in_key else {"opaque": secret}
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_opaque_provider_state",
                        name="resolve_after_start",
                        arguments={"provided": secret},
                    ),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "provider_state": [
                                {
                                    "provider": "vendor",
                                    "state": opaque_state,
                                }
                            ],
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("must not dispatch"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        tool = _ResolveAfterStartTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                vault=StaticVault({"api_key": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"opaque-provider-state-publication-{secret_in_key}",
                messages=[Message.text("user", "run")],
            ),
        )

        assert events[-1].type is EventType.SESSION_FAILED
        assert "opaque provider state" in events[-1].payload["error"]
        assert tool.arguments == [{"provided": secret}]
        assert len(provider.requests) == 1
        assert any(event.type is EventType.TOOL_CALL_COMPLETED for event in events)
        assert secret not in repr(events)
        assert secret not in repr(
            await store.load_transcript(f"opaque-provider-state-publication-{secret_in_key}")
        )

        resumed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=f"opaque-provider-state-publication-{secret_in_key}",
                    messages=[Message.text("user", "continue")],
                )
            )
        ]
        assert resumed[-1].type is EventType.SESSION_FAILED
        assert "opaque provider state" in resumed[-1].payload["error"]
        assert len(provider.requests) == 1
        assert secret not in repr(resumed)

    asyncio.run(run())


def test_secret_is_not_returned_to_tool_before_projection_is_durable() -> None:
    async def run() -> None:
        secret = "projection-write-failure-secret-canary"
        store = _FailFirstBlockedAssistantProjectionStore()
        provider = FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_projection_write_failure",
                    name="resolve_after_start",
                    arguments={"provided": secret},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "provider_state": [{"provider": "vendor", "state": {"opaque": secret}}],
                    }
                ),
            ]
        )
        tool = _ObserveResolvedSecretTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                vault=StaticVault({"api_key": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="projection-write-failure",
                messages=[Message.text("user", "run")],
            ),
        )

        assert store.failed_projection is True
        assert tool.resolution_returned is False
        assert events[-1].type is EventType.SESSION_FAILED
        assert len(provider.requests) == 1
        assert secret not in repr(events)
        assert secret not in repr(await store.load_events("projection-write-failure"))
        assert secret not in repr(await store.load_transcript("projection-write-failure"))

    asyncio.run(run())


@pytest.mark.parametrize("secret_source", ["vault", "proxy"])
def test_restart_blocks_started_call_without_durable_secret_scope(
    tmp_path,
    secret_source: str,
) -> None:
    async def run() -> None:
        secret = "lost-projection-restart-secret-canary"
        session_id = f"lost-projection-restart-{secret_source}"
        database = tmp_path / f"lost-projection-{secret_source}.db"
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_lost_projection",
                        name="resolve_after_start",
                        arguments={"provided": secret},
                    ),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "provider_state": [
                                {
                                    "provider": "vendor",
                                    "state": {"opaque": secret},
                                }
                            ],
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("must not dispatch"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        first_store = _RejectBlockedAssistantProjectionSQLiteStore(database)
        first_tool = _ObserveResolvedSecretTool(secret_source=secret_source)
        first_vault = StaticVault({"api_key": secret})
        first_app = CayuApp(session_store=first_store, enable_logging=False)
        first_app.register_provider(provider, default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                vault=first_vault if secret_source == "vault" else None,
                proxy=PassthroughProxy(first_vault) if secret_source == "proxy" else None,
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[first_tool],
        )

        first_events = await collect_events(
            first_app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )

        assert first_store.failed_projections >= 1
        assert first_tool.resolution_returned is False
        assert first_events[-1].type is EventType.SESSION_FAILED
        assert len(provider.requests) == 1
        await first_store.close()

        recovery_store = SQLiteSessionStore(database)
        recovery_tool = _ResolveAfterStartTool(secret_source=secret_source)
        recovery_app = CayuApp(session_store=recovery_store, enable_logging=False)
        recovery_app.register_provider(provider, default=True)
        recovery_app.register_environment(
            Environment(_portable_environment_spec("local")),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[recovery_tool],
        )

        with pytest.raises(ExecutionProfileMismatchError) as exc_info:
            _ = [
                event
                async for event in recovery_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]

        assert exc_info.value.changed_component_classes == (
            ExecutionProfileComponentClass.EFFECT_AUTHORITY,
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
        )
        assert recovery_tool.arguments == []
        assert len(provider.requests) == 1
        durable_events = await recovery_store.load_events(session_id)
        transcript = await recovery_store.load_transcript(session_id)
        exported = io.StringIO()
        assert await export_sessions(recovery_store, stream=exported) == 1
        assert secret not in repr(first_events)
        assert secret not in repr(exc_info.value)
        assert secret not in repr(durable_events)
        assert secret not in repr(transcript)
        exported_record = json.loads(exported.getvalue())
        # JSONL is a trusted backup and intentionally retains the private
        # quarantined checkpoint. Its public event/transcript representations
        # must nevertheless remain secret-free.
        assert secret not in repr(exported_record["events"])
        assert secret not in repr(exported_record["transcript_records"])
        await recovery_store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("provider_states", "error"),
    [
        (
            [
                {
                    "type": "function_call",
                    "call_id": "different_call",
                    "name": "resolve_after_start",
                    "arguments": "{}",
                }
            ],
            "conflicts with terminal tool evidence",
        ),
        (
            [
                {
                    "type": "function_call",
                    "call_id": "call_projection",
                    "arguments": "{}",
                }
            ],
            "missing its identity",
        ),
        (
            [
                {
                    "type": "function_call",
                    "call_id": "call_projection",
                    "name": "resolve_after_start",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "call_projection",
                    "name": "resolve_after_start",
                    "arguments": "{}",
                },
            ],
            "repeats a tool-call identifier",
        ),
    ],
)
def test_openai_function_call_provider_state_requires_unambiguous_terminal_identity(
    provider_states: list[dict[str, Any]],
    error: str,
) -> None:
    message = Message(
        role="assistant",
        content=(
            ToolCallPart(
                tool_call_id="call_projection",
                tool_name="resolve_after_start",
                arguments={"private": "value"},
            ),
            *(ProviderStatePart(provider="openai", state=state) for state in provider_states),
        ),
    )
    outcomes = (
        ToolCallOutcome(
            call=ToolCallRequest(
                id="call_projection",
                name="resolve_after_start",
                arguments={"private": REDACTED_SECRET},
            ),
            result=ToolResult(content="done"),
        ),
    )

    with pytest.raises(ValueError, match=error):
        transcript_support.assistant_message_with_projected_tool_arguments(
            message,
            outcomes,
        )


def test_durable_recovery_projection_preserves_safe_opaque_assistant_parts() -> None:
    message = Message(
        role="assistant",
        content=(
            ProviderStatePart(provider="vendor", state={"opaque": "unverified"}),
            ToolCallPart(
                tool_call_id="call_recovery_projection",
                tool_name="resolve_after_start",
                arguments={"private": "unverified"},
            ),
        ),
    )
    outcomes = (
        ToolCallOutcome(
            call=ToolCallRequest(
                id="call_recovery_projection",
                name="resolve_after_start",
                arguments={"private": REDACTED_SECRET},
            ),
            result=ToolResult(content="done"),
        ),
    )

    durable_projection = transcript_support.project_assistant_message_for_tool_round_publication(
        message,
        redactor=SecretRedactor(),
    )
    assert durable_projection is not None
    projected = transcript_support.assistant_message_with_projected_tool_arguments(
        durable_projection,
        outcomes,
    )

    assert len(projected.content) == 2
    assert type(projected.content[0]) is ProviderStatePart
    assert projected.content[0].state == {"opaque": "unverified"}
    assert type(projected.content[1]) is ToolCallPart
    assert projected.content[1].arguments == {"private": REDACTED_SECRET}


def test_signed_thinking_state_is_atomic_during_assistant_projection() -> None:
    message = Message(
        role="assistant",
        content=(
            ThinkingPart(
                text="reason safely",
                provider_state={"type": "thinking", "signature": "signed-by-provider"},
            ),
            ToolCallPart(
                tool_call_id="call_signed_thinking",
                tool_name="resolve_after_start",
                arguments={"provided": "private"},
            ),
        ),
    )

    projected = transcript_support.project_assistant_message_for_tool_round_publication(
        message,
        redactor=SecretRedactor("unrelated-secret"),
    )
    assert projected is not None
    assert projected.content[0] == message.content[0]
    assert (
        transcript_support.project_assistant_message_for_tool_round_publication(
            message,
            redactor=SecretRedactor("signed-by-provider"),
        )
        is None
    )


@pytest.mark.parametrize("structural_secret", ["text", "provider", "state", "provider_state"])
def test_quarantined_mixed_assistant_message_preserves_typed_protocol_keys(
    structural_secret: str,
) -> None:
    async def run() -> None:
        late_secret = "late-typed-message-secret-canary"
        arguments = {"provided": late_secret}
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("harmless preface"),
                    ModelStreamEvent.tool_call(
                        id="call_typed_message",
                        name="resolve_after_start",
                        arguments=arguments,
                    ),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "provider_state": [
                                {
                                    "provider": "vendor",
                                    "state": {"opaque": "harmless"},
                                }
                            ],
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        tool = _ResolveAfterStartTool()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(structural_secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                vault=StaticVault({"api_key": late_secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="typed-message-structural-collision",
                messages=[Message.text("user", "run")],
            ),
        )

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert tool.arguments == [arguments]
        assert late_secret not in repr(events)
        transcript = await store.load_transcript("typed-message-structural-collision")
        assert late_secret not in repr(transcript)

    asyncio.run(run())


async def _run_proxy_resolution_scenario() -> None:
    secret = "late-proxy-tool-start-secret-canary"
    arguments = {"provided": secret, f"key-{secret}": {"nested": secret}}
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_proxy_secret",
                    name="resolve_after_start",
                    arguments=arguments,
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    tool = _ResolveAfterStartTool(secret_source="proxy")
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            proxy=PassthroughProxy(StaticVault({"api_key": secret})),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    events = await collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id="late-proxy-secret-quarantine",
            messages=[Message.text("user", "run")],
        ),
    )

    assert tool.arguments == [arguments]
    assert secret not in repr(events)
    assert secret not in repr(await store.load_events("late-proxy-secret-quarantine"))
    assert secret not in repr(await store.load_transcript("late-proxy-secret-quarantine"))
    assert secret not in repr(provider.requests[1].messages)


def test_proxy_resolved_secret_is_quarantined_until_terminal_projection() -> None:
    asyncio.run(_run_proxy_resolution_scenario())


async def _run_approval_resolution_scenario() -> None:
    secret = "late-approved-tool-start-secret-canary"
    arguments = {"provided": secret, "nested": {secret: f"before-{secret}-after"}}
    store = InMemorySessionStore()
    sink = InMemoryEventSink()
    hook = _CaptureInterruptedHook()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_approved_secret",
                    name="resolve_after_start",
                    arguments=arguments,
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    tool = _ResolveAfterStartTool()
    app = CayuApp(
        session_store=store,
        event_sinks=[sink],
        runtime_hooks=[hook],
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            _portable_environment_spec("local"),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    paused = await collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id="late-secret-approval",
            messages=[Message.text("user", "run")],
        ),
    )
    approval = next(
        event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    assert secret not in repr(paused)
    assert secret not in repr(sink.events)
    assert secret not in repr(await store.load_events("late-secret-approval"))
    assert secret not in repr(hook.terminal_events)
    assert len(hook.terminal_events) == 1
    assert "arguments" not in approval.payload["approval"]
    assert approval.payload["approval"]["arguments_state"] == "quarantined"

    resumed_tool = _ResolveAfterStartTool()
    resumed_app = CayuApp(
        session_store=store,
        event_sinks=[sink],
        runtime_hooks=[hook],
        enable_logging=False,
    )
    resumed_app.register_provider(provider, default=True)
    resumed_app.register_environment(
        Environment(
            _portable_environment_spec("local"),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    resumed_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[resumed_tool],
        tool_policy=RequireApprovalPolicy(),
    )
    resumed = await collect_tool_approval_events(
        resumed_app,
        ToolApprovalRequest(
            session_id="late-secret-approval",
            approval_id=approval.payload["approval_id"],
            tool_round_id=approval.payload["tool_round_id"],
            tool_call_id=approval.payload["tool_call_id"],
            decision=ToolApprovalDecision.APPROVE,
            reason=secret,
            metadata={"provided": secret},
        ),
    )

    assert tool.arguments == []
    assert resumed_tool.arguments == [arguments]
    assert resumed[-1].type is EventType.SESSION_COMPLETED, resumed[-1].payload
    approved = next(event for event in resumed if event.type is EventType.TOOL_CALL_APPROVED)
    assert approved.payload["reason"] is None
    assert approved.payload["metadata"] == {}
    assert secret not in repr(resumed)
    assert secret not in repr(sink.events)
    assert secret not in repr(await store.load_transcript("late-secret-approval"))
    assert secret not in repr(provider.requests[1].messages)


def test_approval_pause_keeps_late_secret_arguments_private_until_execution() -> None:
    asyncio.run(_run_approval_resolution_scenario())


def test_multi_call_approval_continuation_uses_round_wide_unavailable_arguments() -> None:
    class ApproveFirstCallPolicy(ToolPolicy):
        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return _test_behavior_identity("approve-first-call-policy")

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            secret = request.arguments["provided"]
            if request.tool_call_id == "call_requires_approval":
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                    reason=f"Approve {secret}",
                    metadata={"provided": secret},
                )
            if request.tool_call_id == "call_denied_before_resolution":
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.DENY,
                    reason=f"Deny {secret}",
                    metadata={"provided": secret},
                )
            return ToolPolicyResult(
                decision=ToolPolicyDecision.ALLOW,
                reason=f"Allow {secret}",
                metadata={"provided": secret},
            )

    class ResolveConditionallyTool(Tool):
        spec = ToolSpec(
            name="resolve_conditionally_after_approval",
            description="Resolve the secret only for the selected sibling call.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.NONE,
            execution_profile_identity=_test_behavior_identity(
                "resolve-conditionally-after-approval-tool"
            ),
        )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            if args["resolve"]:
                assert ctx.vault is not None
                await ctx.vault.resolve(SecretRef(name="shared"))
                return ToolResult(content="resolved")
            # This sibling completes before the later call identifies the same
            # value as a vault-managed secret. Its terminal must remain private
            # until the round-wide registry is finalized.
            return ToolResult(content=args["provided"])

    class CaptureAfterToolResultHook(RuntimeHook):
        def __init__(self) -> None:
            self.results: list[ToolResult] = []

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return _test_behavior_identity("capture-after-tool-result-hook")

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            self.results.append(context.result)

    async def run() -> None:
        secret = "late-approved-sibling-secret-canary"
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        watcher_store = InMemoryEventWatcherStore()
        hook = _CaptureInterruptedHook()
        after_tool_hook = CaptureAfterToolResultHook()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_requires_approval",
                        name="resolve_conditionally_after_approval",
                        arguments={"resolve": False, "provided": secret},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_denied_before_resolution",
                        name="resolve_conditionally_after_approval",
                        arguments={"resolve": False, "provided": secret},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_resolves_after_approval",
                        name="resolve_conditionally_after_approval",
                        arguments={"resolve": True, "provided": secret},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            event_watcher_store=watcher_store,
            runtime_hooks=[hook, after_tool_hook],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                vault=StaticVault({"shared": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[ResolveConditionallyTool()],
            tool_policy=ApproveFirstCallPolicy(),
        )
        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="late-secret-multi-approval",
                messages=[Message.text("user", "run")],
            ),
        )
        approval = next(
            event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        interrupted = next(event for event in paused if event.type is EventType.SESSION_INTERRUPTED)
        for public_approval in (
            approval.payload["approval"],
            interrupted.payload["approval"],
        ):
            assert "reason" not in public_approval
            assert "metadata" not in public_approval
            assert all("reason" not in call for call in public_approval["tool_calls"])
            assert all("metadata" not in call for call in public_approval["tool_calls"])
        pending_actions = await store.query_pending_actions(
            PendingActionQuery(session_id="late-secret-multi-approval", limit=10)
        )
        assert len(pending_actions.actions) == 1
        assert pending_actions.actions[0].detail == "Approval required"
        assert secret not in repr(pending_actions)
        assert secret not in repr(paused)
        assert secret not in repr(sink.events)
        assert secret not in repr(await store.load_events("late-secret-multi-approval"))
        assert secret not in repr(hook.terminal_events)
        observed = []
        for event_type in (
            EventType.TOOL_CALL_APPROVAL_REQUESTED,
            EventType.SESSION_INTERRUPTED,
        ):
            await app.run_event_watchers(
                [
                    EventWatcher(
                        name=f"late-secret-approval-{event_type.value}",
                        query=EventQuery(
                            session_id="late-secret-multi-approval",
                            event_type=event_type,
                        ),
                        handler=observed.append,
                    )
                ]
            )
        assert secret not in repr(observed)

        resumed_app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            runtime_hooks=[hook, after_tool_hook],
            enable_logging=False,
        )
        resumed_app.register_provider(provider, default=True)
        resumed_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                vault=StaticVault({"shared": secret}),
            ),
            default=True,
        )
        resumed_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[ResolveConditionallyTool()],
            tool_policy=ApproveFirstCallPolicy(),
        )
        resumed = await collect_tool_approval_events(
            resumed_app,
            ToolApprovalRequest(
                session_id="late-secret-multi-approval",
                approval_id=approval.payload["approval_id"],
                tool_round_id=approval.payload["tool_round_id"],
                tool_call_id=approval.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )

        terminal_events = [
            event for event in resumed if event.type is EventType.TOOL_CALL_COMPLETED
        ]
        blocked_event = next(
            event for event in resumed if event.type is EventType.TOOL_CALL_BLOCKED
        )
        assert resumed[-1].type is EventType.SESSION_COMPLETED, resumed[-1].payload
        assert len(terminal_events) == 2
        assert all(event.payload["arguments_state"] == "unavailable" for event in terminal_events)
        assert all("arguments" not in event.payload for event in terminal_events)
        assert blocked_event.payload["reason"] == "Tool call denied by policy."
        assert blocked_event.payload["metadata"] == {}
        assert blocked_event.payload["result"] == {
            "content": "Tool call denied by policy.",
            "structured": {
                "decision": "deny",
                "reason": "Tool call denied by policy.",
                "metadata": {},
            },
            "artifacts": [],
            "is_error": True,
        }
        assert secret not in repr(after_tool_hook.results)
        assert secret not in repr(resumed)
        assert secret not in repr(sink.events)
        assert secret not in repr(await store.load_events("late-secret-multi-approval"))
        assert secret not in repr(await store.load_transcript("late-secret-multi-approval"))
        assert secret not in repr(provider.requests[1].messages)
        exported = io.StringIO()
        assert await export_sessions(store, stream=exported) == 1
        assert secret not in exported.getvalue()

    asyncio.run(run())


async def _run_user_input_resolution_scenario() -> None:
    secret = "late-user-input-tool-start-secret-canary"
    sibling_arguments = {"provided": secret, "nested": {"value": secret}}

    class UserInputMixedPolicy(ToolPolicy):
        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return _test_behavior_identity("user-input-mixed-policy")

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            if request.tool_call_id == "call_denied_before_input_resolution":
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.DENY,
                    reason=f"Deny {secret}",
                    metadata={"provided": secret},
                )
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    store = InMemorySessionStore()
    sink = InMemoryEventSink()
    watcher_store = InMemoryEventWatcherStore()
    hook = _CaptureInterruptedHook()
    after_tool_hook = _CaptureAfterToolResultsHook()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_input",
                    name="ask_user",
                    arguments={"question": secret, "options": ["yes", secret]},
                ),
                ModelStreamEvent.tool_call(
                    id="call_denied_before_input_resolution",
                    name="resolve_after_start",
                    arguments=sibling_arguments,
                ),
                ModelStreamEvent.tool_call(
                    id="call_after_input",
                    name="resolve_after_start",
                    arguments=sibling_arguments,
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    tool = _ResolveAfterStartTool()
    app = CayuApp(
        session_store=store,
        event_sinks=[sink],
        event_watcher_store=watcher_store,
        runtime_hooks=[hook, after_tool_hook],
        secret_redactor=SecretRedactor("tool_call_id"),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            _portable_environment_spec("local"),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), tool],
        tool_policy=UserInputMixedPolicy(),
    )

    paused = await collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id="late-secret-user-input",
            messages=[Message.text("user", "run")],
        ),
    )
    awaiting = next(
        event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
    )
    interrupted = next(event for event in paused if event.type is EventType.SESSION_INTERRUPTED)
    assert "question" not in awaiting.payload
    assert "options" not in awaiting.payload
    assert "question" not in interrupted.payload["user_input"]
    assert "options" not in interrupted.payload["user_input"]
    assert secret not in repr(paused)
    assert secret not in repr(sink.events)
    assert secret not in repr(await store.load_events("late-secret-user-input"))
    assert secret not in repr(hook.terminal_events)
    assert len(hook.terminal_events) == 1
    assert all("arguments" not in call for call in awaiting.payload["tool_calls"])
    pending_actions = await store.query_pending_actions(
        PendingActionQuery(session_id="late-secret-user-input", limit=10)
    )
    assert len(pending_actions.actions) == 1
    pending_action = pending_actions.actions[0]
    assert pending_action.detail == "Input required"
    assert pending_action.question is None
    assert pending_action.options == []
    assert secret not in repr(pending_action)
    observed = []
    for event_type in (
        EventType.SESSION_AWAITING_USER_INPUT,
        EventType.SESSION_INTERRUPTED,
    ):
        await app.run_event_watchers(
            [
                EventWatcher(
                    name=f"late-secret-user-input-{event_type.value}",
                    query=EventQuery(
                        session_id="late-secret-user-input",
                        event_type=event_type,
                    ),
                    handler=observed.append,
                )
            ]
        )
    assert len(observed) == 2
    assert secret not in repr(observed)
    private_checkpoint = await store.load_checkpoint("late-secret-user-input")
    assert private_checkpoint is not None
    assert private_checkpoint["pending_user_input"]["question"] == secret
    assert private_checkpoint["pending_user_input"]["options"] == ["yes", secret]

    await rebind_test_invocation(store, "late-secret-user-input")
    restarted_app = CayuApp(
        session_store=store,
        event_sinks=[sink],
        runtime_hooks=[hook, after_tool_hook],
        secret_redactor=SecretRedactor("tool_call_id"),
        enable_logging=False,
    )
    restarted_app.register_provider(provider, default=True)
    restarted_app.register_environment(
        Environment(
            _portable_environment_spec("local"),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    restarted_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), tool],
        tool_policy=UserInputMixedPolicy(),
    )
    recovered = await restarted_app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(session_id="late-secret-user-input")
    )
    recovered_interrupts = [
        event for event in recovered.events if event.type is EventType.SESSION_INTERRUPTED
    ]
    assert recovered_interrupts
    assert "question" not in recovered_interrupts[-1].payload["user_input"]
    assert "options" not in recovered_interrupts[-1].payload["user_input"]
    assert secret not in repr(recovered)

    resumed = [
        event
        async for event in restarted_app.resolve_user_input(
            UserInputResponse(
                session_id="late-secret-user-input",
                input_id=awaiting.payload["input_id"],
                # The injected ask_user result is produced before the later
                # sibling resolves this exact value from the vault.
                answer=secret,
            )
        )
    ]

    assert resumed[-1].type is EventType.SESSION_COMPLETED, resumed[-1].payload
    assert tool.arguments == [sibling_arguments]
    terminal_events = [event for event in resumed if event.type is EventType.TOOL_CALL_COMPLETED]
    blocked_event = next(event for event in resumed if event.type is EventType.TOOL_CALL_BLOCKED)
    assert len(terminal_events) == 2
    assert all(event.payload["arguments_state"] == "unavailable" for event in terminal_events)
    assert all("arguments" not in event.payload for event in terminal_events)
    assert blocked_event.payload["reason"] == "Tool call denied by policy."
    assert blocked_event.payload["metadata"] == {}
    assert secret not in repr(after_tool_hook.results)
    assert secret not in repr(resumed)
    assert secret not in repr(sink.events)
    assert secret not in repr(await store.load_transcript("late-secret-user-input"))
    assert secret not in repr(provider.requests[1].messages)
    exported = io.StringIO()
    assert await export_sessions(store, stream=exported) == 1
    exported_record = json.loads(exported.getvalue())
    assert secret not in repr(exported_record["events"])
    assert secret not in repr(exported_record["transcript_records"])


def test_user_input_pause_keeps_late_secret_arguments_private_until_execution() -> None:
    asyncio.run(_run_user_input_resolution_scenario())


def test_static_user_input_pause_keeps_safe_prompt_public() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_static_input",
                    name="ask_user",
                    arguments={"question": "Deploy now?", "options": ["yes", "no"]},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool()],
        )

        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="static-user-input-prompt",
                messages=[Message.text("user", "run")],
            ),
        )

        awaiting = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        interrupted = next(event for event in paused if event.type is EventType.SESSION_INTERRUPTED)
        assert awaiting.payload["question"] == "Deploy now?"
        assert awaiting.payload["options"] == ["yes", "no"]
        assert interrupted.payload["user_input"]["question"] == "Deploy now?"
        assert interrupted.payload["user_input"]["options"] == ["yes", "no"]
        pending = await store.query_pending_actions(
            PendingActionQuery(session_id="static-user-input-prompt", limit=10)
        )
        assert len(pending.actions) == 1
        assert pending.actions[0].detail == "Deploy now?"
        assert pending.actions[0].question == "Deploy now?"
        assert pending.actions[0].options == ["yes", "no"]

    asyncio.run(run())


def test_user_input_pause_omits_provisional_assistant_publication() -> None:
    async def run() -> None:
        secret = "late-user-input-provider-state-secret-canary"
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        hook = _CaptureInterruptedHook()
        provider = FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_input_projection",
                    name="ask_user",
                    arguments={"question": "Continue?"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_after_input_projection",
                    name="resolve_after_start",
                    arguments={"provided": secret},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "provider_state": [
                            {
                                "provider": "vendor",
                                "state": {"opaque": secret},
                            }
                        ],
                    }
                ),
            ]
        )
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            runtime_hooks=[hook],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                vault=StaticVault({"api_key": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), _ResolveAfterStartTool()],
        )

        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="late-secret-user-input-projection",
                messages=[Message.text("user", "run")],
            ),
        )

        interrupted = next(event for event in paused if event.type is EventType.SESSION_INTERRUPTED)
        assert "assistant_publication" not in interrupted.payload["user_input"]
        assert secret not in repr(paused)
        assert secret not in repr(sink.events)
        assert secret not in repr(await store.load_events("late-secret-user-input-projection"))
        assert secret not in repr(hook.terminal_events)

    asyncio.run(run())


class _DenyToolPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        del request
        return ToolPolicyResult(
            decision=ToolPolicyDecision.DENY,
            reason="denied by test policy",
        )


class _ShortCircuitHook(RuntimeHook):
    name = "short-circuit"

    async def before_tool_call(
        self,
        context: BeforeToolCallHookContext,
    ) -> BeforeToolCallDecision:
        del context
        return BeforeToolCallDecision(
            action="short_circuit",
            synthetic_result=ToolResult(content="cached"),
        )


class _ModifyBeforeNonexecutionHook(RuntimeHook):
    async def before_tool_call(
        self,
        context: BeforeToolCallHookContext,
    ) -> BeforeToolCallDecision:
        modified = dict(context.arguments)
        modified["harmless"] = "modified"
        return BeforeToolCallDecision(
            action="proceed_modified",
            modified_arguments=modified,
        )


class _ObserveAfterNonexecutionHook(RuntimeHook):
    def __init__(self) -> None:
        self.arguments: list[dict[str, Any]] = []

    async def after_tool_call(self, context: ToolCallHookContext) -> None:
        self.arguments.append(context.arguments)


async def _run_nonexecuting_argument_scenario(*, short_circuit: bool) -> None:
    secret = (
        "late-hook-tool-start-secret-canary"
        if short_circuit
        else "late-policy-tool-start-secret-canary"
    )
    arguments = {"provided": secret, secret: {"nested": secret}}
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_nonexecuting_secret",
                    name="resolve_after_start",
                    arguments=arguments,
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    tool = _ResolveAfterStartTool()
    observer = _ObserveAfterNonexecutionHook()
    runtime_hooks: list[RuntimeHook] = [observer]
    if short_circuit:
        runtime_hooks = [
            _ModifyBeforeNonexecutionHook(),
            _ShortCircuitHook(),
            observer,
        ]
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        runtime_hooks=runtime_hooks,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=None if short_circuit else _DenyToolPolicy(),
    )

    events = await collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id=f"late-secret-{'hook' if short_circuit else 'policy'}",
            messages=[Message.text("user", "run")],
        ),
    )

    assert tool.arguments == []
    terminal = next(
        event
        for event in events
        if event.type
        in {
            EventType.TOOL_CALL_BLOCKED,
            EventType.TOOL_CALL_COMPLETED,
        }
    )
    assert terminal.payload["arguments_state"] == "unavailable"
    assert "arguments" not in terminal.payload
    assert "effective_arguments" not in terminal.payload
    assert observer.arguments == [{}]
    assert secret not in repr(events)
    assert secret not in repr(await store.load_transcript(events[0].session_id))
    assert secret not in repr(provider.requests[1].messages)


def test_policy_denial_never_publishes_unresolved_secret_arguments() -> None:
    asyncio.run(_run_nonexecuting_argument_scenario(short_circuit=False))


def test_hook_short_circuit_never_publishes_unresolved_secret_arguments() -> None:
    asyncio.run(_run_nonexecuting_argument_scenario(short_circuit=True))


@pytest.mark.parametrize(
    "secret",
    ["arguments", "arguments_state", "effective_arguments"],
)
def test_terminal_argument_protocol_keys_survive_exact_secret_collisions(secret: str) -> None:
    async def run() -> None:
        arguments = {"provided": secret, "nested": {secret: secret}}
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_protocol_collision",
                        name="resolve_after_start",
                        arguments=arguments,
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        tool = _ResolveAfterStartTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                vault=StaticVault({"api_key": secret}),
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"terminal-key-collision-{secret}",
                messages=[Message.text("user", "run")],
            ),
        )

        assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
        assert tool.arguments == [arguments]
        terminal = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
        assert terminal.payload["arguments_state"] == "finalized"
        assert REDACTED_SECRET not in terminal.payload
        assert terminal.payload["arguments"]["provided"] == REDACTED_SECRET
        assert terminal.payload["arguments"]["nested"] == {REDACTED_SECRET: REDACTED_SECRET}
        transcript = await store.load_transcript(events[0].session_id)
        projected_call = next(
            part for message in transcript for part in message.content if type(part) is ToolCallPart
        )
        assert projected_call.arguments == terminal.payload["arguments"]
        assert (
            provider.requests[1].messages[-2].content[0].arguments == terminal.payload["arguments"]
        )

    asyncio.run(run())


def test_structured_output_rejection_keeps_nonexecuted_arguments_quarantined() -> None:
    async def run() -> None:
        secret = "late-structured-output-secret-canary"
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_nonfinalizer",
                        name="resolve_after_start",
                        arguments={"provided": secret},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_invalid_finalizer",
                        name=STRUCTURED_OUTPUT_TOOL_NAME,
                        arguments={"output": {"answer": "too early"}},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="call_valid_finalizer",
                        name=STRUCTURED_OUTPUT_TOOL_NAME,
                        arguments={"output": {"answer": "safe"}},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
            ]
        )
        tool = _ResolveAfterStartTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                vault=StaticVault({"api_key": secret}),
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="structured-output-argument-quarantine",
                messages=[Message.text("user", "run")],
                structured_output=StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    max_retries=1,
                ),
            ),
        )

        assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
        assert tool.arguments == []
        terminal_events = [
            event
            for event in events
            if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        ]
        assert terminal_events
        assert all(event.payload["arguments_state"] == "unavailable" for event in terminal_events)
        assert all("arguments" not in event.payload for event in terminal_events)
        transcript = await store.load_transcript("structured-output-argument-quarantine")
        assert secret not in repr(events)
        assert secret not in repr(transcript)
        assert secret not in repr(provider.requests[1].messages)
        assert all(
            part.arguments == {}
            for message in transcript
            for part in message.content
            if type(part) is ToolCallPart
        )

    asyncio.run(run())


async def _run_failed_execution_scenario(*, timed_out: bool) -> None:
    secret = (
        "late-timeout-tool-start-secret-canary"
        if timed_out
        else "late-failed-tool-start-secret-canary"
    )
    arguments = {"provided": secret, "nested": [{secret: f"x-{secret}-y"}]}
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_failed_secret",
                    name="resolve_after_start",
                    arguments=arguments,
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    tool = _TimeoutAfterResolutionTool() if timed_out else _FailAfterResolutionTool()
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        config=CayuConfig(
            tool_execution=ToolExecutionConfig(tool_timeout_seconds=0.01 if timed_out else None)
        ),
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    session_id = f"late-secret-{'timeout' if timed_out else 'failure'}"
    events = await collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "run")],
        ),
    )

    terminal = next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)
    assert terminal.payload["arguments_state"] == "finalized"
    assert secret not in repr(terminal.payload["arguments"])
    assert secret not in repr(events)
    assert secret not in repr(await store.load_transcript(session_id))
    assert secret not in repr(provider.requests[1].messages)


def test_failed_tool_finalizes_late_secret_arguments_before_publication() -> None:
    asyncio.run(_run_failed_execution_scenario(timed_out=False))


def test_timed_out_tool_finalizes_late_secret_arguments_before_publication() -> None:
    asyncio.run(_run_failed_execution_scenario(timed_out=True))


@pytest.mark.parametrize("secret_source", ["vault", "proxy"])
def test_tool_timeout_does_not_wait_for_a_nonresponsive_secret_resolver(
    secret_source: str,
) -> None:
    secret = "late-nonresponsive-resolution-secret-canary"

    class CancellationIgnoringVault(StaticVault):
        def __init__(self) -> None:
            super().__init__({"api_key": secret})
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = asyncio.Event()

        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ):
            del scope
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await self.release.wait()
            self.finished.set()
            return await super().resolve(ref)

    async def run() -> None:
        store = InMemorySessionStore()
        vault = CancellationIgnoringVault()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_nonresponsive_resolution",
                        name="resolve_after_start",
                        arguments={"ref": "api_key"},
                    ),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "provider_state": [
                                {
                                    "provider": "vendor",
                                    "state": {"opaque": secret},
                                }
                            ],
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("handled timeout"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        tool = _ResolveAfterStartTool(secret_source=secret_source)
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            config=CayuConfig(tool_execution=ToolExecutionConfig(tool_timeout_seconds=0.01)),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                vault=vault if secret_source == "vault" else None,
                proxy=PassthroughProxy(vault) if secret_source == "proxy" else None,
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

        events = await asyncio.wait_for(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=f"nonresponsive-secret-resolution-{secret_source}",
                    messages=[Message.text("user", "run")],
                ),
            ),
            timeout=1,
        )

        assert vault.started.is_set() is True
        assert vault.finished.is_set() is False
        assert events[-1].type is EventType.SESSION_FAILED
        terminal = next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)
        assert terminal.payload["result"]["content"] == (
            "Tool output was omitted because its secret-redaction scope could not be "
            "finalized safely before publication."
        )
        assert terminal.payload["result"]["structured"]["terminal_outcome"] == (
            "invalid_tool_output"
        )
        assert len(provider.requests) == 1

        vault.release.set()
        await asyncio.wait_for(vault.finished.wait(), timeout=1)
        await asyncio.sleep(0)
        durable_events = await store.load_events(f"nonresponsive-secret-resolution-{secret_source}")
        transcript = await store.load_transcript(f"nonresponsive-secret-resolution-{secret_source}")
        assert secret not in repr(events)
        assert secret not in repr(durable_events)
        assert secret not in repr(transcript)
        assert secret not in repr(provider.requests)

    asyncio.run(run())


async def _run_cancelled_execution_scenario(*, cleanup_fails: bool) -> None:
    secret = (
        "late-cleanup-tool-start-secret-canary"
        if cleanup_fails
        else "late-cancelled-tool-start-secret-canary"
    )
    arguments = {"provided": secret, secret: {"nested": secret}}
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_cancelled_secret",
                name="resolve_after_start",
                arguments=arguments,
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    tool = _CancellableAfterResolutionTool(cleanup_fails=cleanup_fails)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])
    session_id = f"late-secret-{'cleanup' if cleanup_fails else 'cancel'}"
    public_events = []

    async def consume() -> None:
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            )
        ):
            public_events.append(event)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(tool.dispatched.wait(), timeout=1)
    task.cancel("operator cancelled test run")
    if cleanup_fails:
        await task
    else:
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert secret not in repr(raised.value)
    assert secret not in repr(public_events)
    durable_events = await store.load_events(session_id)
    assert secret not in repr(durable_events)
    started = [event for event in durable_events if event.type is EventType.TOOL_CALL_STARTED]
    assert len(started) == 1
    assert started[0].payload["arguments_state"] == "quarantined"


def test_real_task_cancellation_cannot_publish_late_secret_arguments() -> None:
    asyncio.run(_run_cancelled_execution_scenario(cleanup_fails=False))


def test_cleanup_failure_cannot_publish_late_secret_arguments() -> None:
    asyncio.run(_run_cancelled_execution_scenario(cleanup_fails=True))


async def _run_restart_recovery_scenario() -> None:
    secret = "late-restart-tool-start-secret-canary"
    arguments = {"provided": secret, secret: {"nested": secret}}
    store = _FailFirstTerminalEventStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_restart_secret",
                    name="resolve_after_start",
                    arguments=arguments,
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "provider_state": [
                            {
                                "provider": "chat_completions",
                                "state": {
                                    "type": "tool_call_extra_content",
                                    "tool_call_id": "call_restart_secret",
                                    "extra_content": {
                                        "google": {"thought_signature": "restart-signature-safe"}
                                    },
                                },
                            }
                        ],
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    first_tool = _ResolveAfterStartTool()
    first_app = CayuApp(session_store=store, enable_logging=False)
    first_app.register_provider(provider, default=True)
    first_app.register_environment(
        Environment(
            _portable_environment_spec("local"),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    first_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[first_tool],
    )
    first_events = await collect_events(
        first_app,
        RunRequest(
            agent_name="assistant",
            session_id="late-secret-restart-recovery",
            messages=[Message.text("user", "run")],
        ),
    )
    assert first_events[-1].type is EventType.SESSION_FAILED
    assert first_tool.arguments == [arguments]

    recovery_tool = _ResolveAfterStartTool()
    recovery_app = CayuApp(session_store=store, enable_logging=False)
    recovery_app.register_provider(provider, default=True)
    recovery_app.register_environment(
        Environment(
            _portable_environment_spec("local"),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    recovery_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[recovery_tool],
    )
    recovered_events = [
        event
        async for event in recovery_app.resume(
            ResumeRequest(
                session_id="late-secret-restart-recovery",
                messages=[Message.text("user", "continue")],
            )
        )
    ]

    assert recovered_events[-1].type is EventType.SESSION_COMPLETED
    assert recovery_tool.arguments == []
    durable_events = await store.load_events("late-secret-restart-recovery")
    starts = [event for event in durable_events if event.type is EventType.TOOL_CALL_STARTED]
    assert len(starts) == 1
    recovered_terminal = next(
        event
        for event in durable_events
        if event.type is EventType.TOOL_CALL_FAILED
        and event.payload.get("tool_call_id") == "call_restart_secret"
    )
    assert recovered_terminal.payload["arguments_state"] == "unavailable"
    assert secret not in repr(durable_events)
    transcript = await store.load_transcript("late-secret-restart-recovery")
    assert secret not in repr(transcript)
    assert any(
        type(part) is ProviderStatePart
        and part.state["extra_content"]["google"]["thought_signature"] == "restart-signature-safe"
        for message in transcript
        for part in message.content
    )
    assert secret not in repr(provider.requests[1].messages)


def test_restart_recovery_never_republishes_quarantined_arguments() -> None:
    asyncio.run(_run_restart_recovery_scenario())


def test_pre_field_v2_recovery_blocks_missing_invocation_projection() -> None:
    async def run() -> None:
        secret = "legacy-v2-recovery-secret-canary"
        session_id = "legacy-v2-missing-assistant-publication"
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_legacy_projection",
                        name="resolve_after_start",
                        arguments={"provided": secret},
                    ),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "provider_state": [
                                {
                                    "provider": "vendor",
                                    "state": {"opaque": secret},
                                }
                            ],
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("must not dispatch"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        first_tool = _ResolveAfterStartTool()
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(provider, default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                vault=StaticVault({"api_key": secret}),
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[first_tool],
        )

        first_events = await collect_events(
            first_app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
        assert first_events[-1].type is EventType.SESSION_FAILED
        assert first_tool.arguments == [{"provided": secret}]

        def strip_additive_publication(_session, checkpoint):
            assert checkpoint is not None
            copied = dict(checkpoint)
            pending_round = dict(copied["pending_tool_round"])
            pending_round.pop("assistant_publication", None)
            copied["pending_tool_round"] = pending_round
            return copied

        await store.transform_checkpoint(session_id, strip_additive_publication)

        recovery_tool = _ResolveAfterStartTool()
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(provider, default=True)
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                vault=StaticVault({"api_key": secret}),
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[recovery_tool],
        )
        recovered_events = [
            event
            async for event in recovery_app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                )
            )
        ]

        assert recovered_events[-1].type is EventType.SESSION_FAILED
        assert recovery_tool.arguments == []
        assert len(provider.requests) == 1
        assert secret not in repr(recovered_events)
        assert secret not in repr(await store.load_transcript(session_id))

    asyncio.run(run())


async def _run_operator_interruption_scenario() -> None:
    secret = "late-interrupted-tool-start-secret-canary"
    arguments = {"provided": secret, "nested": {secret: secret}}
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_interrupted_secret",
                name="resolve_after_start",
                arguments=arguments,
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    tool = _CancellableAfterResolutionTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])
    run_events: list[Any] = []

    async def consume() -> None:
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="late-secret-interruption",
                messages=[Message.text("user", "run")],
            )
        ):
            run_events.append(event)

    run_task = asyncio.create_task(consume())
    await asyncio.wait_for(tool.dispatched.wait(), timeout=1)
    interrupt_events = [
        event
        async for event in app.interrupt_session(
            InterruptSessionRequest(
                session_id="late-secret-interruption",
                reason="operator interruption regression",
                metadata={
                    "arguments": {"audit": "preserve-this-operator-metadata"},
                    "arguments_state": "operator-supplied-state",
                    "assistant_message_state": "operator-supplied-message-state",
                    "quarantined_assistant_message": "operator-supplied-message",
                },
            )
        )
    ]
    await asyncio.wait_for(run_task, timeout=1)

    assert secret not in repr(run_events)
    assert secret not in repr(interrupt_events)
    assert secret not in repr(await store.load_events("late-secret-interruption"))
    interrupted = next(
        event for event in interrupt_events if event.type is EventType.SESSION_INTERRUPTED
    )
    assert interrupted.payload["metadata"] == {
        "arguments": {"audit": "preserve-this-operator-metadata"},
        "arguments_state": "operator-supplied-state",
        "assistant_message_state": "operator-supplied-message-state",
        "quarantined_assistant_message": "operator-supplied-message",
    }


def test_operator_interruption_cannot_publish_late_secret_arguments() -> None:
    asyncio.run(_run_operator_interruption_scenario())


async def _run_multi_call_secret_scope_scenario(*, max_parallel_tool_calls: int) -> None:
    first_secret = "late-first-tool-start-secret-canary"
    second_secret = "late-second-tool-start-secret-canary"
    calls = [
        {
            "ref": "first",
            "provided": first_secret,
            f"key-{first_secret}": first_secret,
        },
        {
            "ref": "second",
            "provided": second_secret,
            f"key-{second_secret}": second_secret,
        },
    ]
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id=f"call_{index}",
                    name="resolve_named_secret",
                    arguments=arguments,
                )
                for index, arguments in enumerate(calls, start=1)
            ]
            + [
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "provider_state": [
                            {
                                "provider": "vendor",
                                "state": {
                                    "first": "safe-first-state",
                                    "second": "safe-second-state",
                                },
                            }
                        ],
                    }
                )
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    tool = _ResolveNamedSecretTool()
    hook = _CaptureAfterToolArgumentsHook()
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        config=CayuConfig(
            tool_execution=ToolExecutionConfig(max_parallel_tool_calls=max_parallel_tool_calls)
        ),
        runtime_hooks=[hook],
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=StaticVault({"first": first_secret, "second": second_secret}),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    events = await collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id=f"late-secret-multi-{max_parallel_tool_calls}",
            messages=[Message.text("user", "run")],
        ),
    )

    assert sorted(tool.arguments, key=lambda item: item["ref"]) == calls
    hook_arguments = hook.arguments
    assert hook_arguments == [{}, {}]
    assert first_secret not in repr(hook_arguments)
    assert second_secret not in repr(hook_arguments)
    assert len([event for event in events if event.type is EventType.TOOL_CALL_STARTED]) == 2
    terminal_events = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
    assert len(terminal_events) == 2
    assert all(event.payload["arguments_state"] == "finalized" for event in terminal_events)
    assert all(event.payload["arguments_exact"] is False for event in terminal_events)
    assert all(type(event.payload["arguments"]) is dict for event in terminal_events)
    assert first_secret not in repr(events)
    assert second_secret not in repr(events)
    assert first_secret not in repr(provider.requests[1].messages)
    assert second_secret not in repr(provider.requests[1].messages)
    transcript = await store.load_transcript(f"late-secret-multi-{max_parallel_tool_calls}")
    provider_state = next(
        part
        for message in transcript
        for part in message.content
        if type(part) is ProviderStatePart and part.provider == "vendor"
    )
    assert provider_state.state == {
        "first": "safe-first-state",
        "second": "safe-second-state",
    }


@pytest.mark.parametrize("max_parallel_tool_calls", [1, 4])
def test_multi_call_rounds_keep_late_secret_scopes_isolated(
    max_parallel_tool_calls: int,
) -> None:
    asyncio.run(
        _run_multi_call_secret_scope_scenario(
            max_parallel_tool_calls=max_parallel_tool_calls,
        )
    )


@pytest.mark.parametrize("max_parallel_tool_calls", [1, 4])
def test_multi_call_round_redacts_arguments_when_only_a_sibling_resolves_the_secret(
    max_parallel_tool_calls: int,
) -> None:
    class PublishAfterArgumentsHook(RuntimeHook):
        def __init__(self) -> None:
            self.arguments: list[dict[str, Any]] = []

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            self.arguments.append(context.arguments)
            await context.emit_custom_event(
                "custom.sibling_arguments",
                payload={"arguments": context.arguments},
            )

    class ResolveConditionallyTool(Tool):
        spec = ToolSpec(
            name="resolve_conditionally",
            description="Resolve the secret only for the selected sibling call.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.NONE,
        )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            if args["resolve"]:
                assert ctx.vault is not None
                await ctx.vault.resolve(SecretRef(name="shared"))
            return ToolResult(content="done")

    async def run() -> None:
        secret = "late-sibling-tool-start-secret-canary"
        store = InMemorySessionStore()
        hook = PublishAfterArgumentsHook()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_resolves_shared_secret",
                        name="resolve_conditionally",
                        arguments={"resolve": True, "provided": secret},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_does_not_resolve_shared_secret",
                        name="resolve_conditionally",
                        arguments={"resolve": False, "provided": secret},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            config=CayuConfig(
                tool_execution=ToolExecutionConfig(max_parallel_tool_calls=max_parallel_tool_calls)
            ),
            runtime_hooks=[hook],
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                vault=StaticVault({"shared": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[ResolveConditionallyTool()],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"late-secret-sibling-{max_parallel_tool_calls}",
                messages=[Message.text("user", "run")],
            ),
        )

        terminal_events = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
        assert len(terminal_events) == 2
        assert all(event.payload["arguments_state"] == "finalized" for event in terminal_events)
        assert all(event.payload["arguments_exact"] is False for event in terminal_events)
        assert all(type(event.payload["arguments"]) is dict for event in terminal_events)
        assert hook.arguments == [{}, {}]
        durable_events = await store.load_events(f"late-secret-sibling-{max_parallel_tool_calls}")
        custom_events = [
            event for event in durable_events if str(event.type) == "custom.sibling_arguments"
        ]
        assert len(custom_events) == 2
        assert all(event.payload == {"arguments": {}} for event in custom_events)
        assert secret not in repr(events)
        assert secret not in repr(durable_events)
        assert secret not in repr(
            await store.load_transcript(f"late-secret-sibling-{max_parallel_tool_calls}")
        )
        assert secret not in repr(provider.requests[1].messages)

    asyncio.run(run())


@pytest.mark.parametrize("max_parallel_tool_calls", [1, 4])
def test_multi_call_policy_denial_omits_output_before_a_sibling_resolves_its_secret(
    max_parallel_tool_calls: int,
) -> None:
    class MixedPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            secret = request.arguments["provided"]
            if request.tool_call_id == "call_denied_before_resolution":
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.DENY,
                    reason=f"Deny {secret}",
                    metadata={"provided": secret},
                )
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    class ResolveConditionallyTool(Tool):
        spec = ToolSpec(
            name="resolve_for_policy_sibling",
            description="Resolve the shared secret for only one call.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.NONE,
        )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            if args["resolve"]:
                assert ctx.vault is not None
                await ctx.vault.resolve(SecretRef(name="shared"))
            return ToolResult(content="done")

    async def run() -> None:
        secret = "late-policy-sibling-secret-canary"
        session_id = f"late-policy-sibling-{max_parallel_tool_calls}"
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        hook = _CaptureAfterToolResultsHook()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_denied_before_resolution",
                        name="resolve_for_policy_sibling",
                        arguments={"resolve": False, "provided": secret},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_resolves_secret",
                        name="resolve_for_policy_sibling",
                        arguments={"resolve": True, "provided": secret},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            runtime_hooks=[hook],
            enable_logging=False,
            config=CayuConfig(
                tool_execution=ToolExecutionConfig(max_parallel_tool_calls=max_parallel_tool_calls)
            ),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                vault=StaticVault({"shared": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[ResolveConditionallyTool()],
            tool_policy=MixedPolicy(),
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )

        blocked = next(event for event in events if event.type is EventType.TOOL_CALL_BLOCKED)
        assert blocked.payload["reason"] == "Tool call denied by policy."
        assert blocked.payload["metadata"] == {}
        assert blocked.payload["result"] == {
            "content": "Tool call denied by policy.",
            "structured": {
                "decision": "deny",
                "reason": "Tool call denied by policy.",
                "metadata": {},
            },
            "artifacts": [],
            "is_error": True,
        }
        assert secret not in repr(hook.results)
        assert secret not in repr(events)
        assert secret not in repr(sink.events)
        assert secret not in repr(await store.load_events(session_id))
        assert secret not in repr(await store.load_transcript(session_id))
        assert secret not in repr(provider.requests[1].messages)

    asyncio.run(run())


def test_command_policy_denial_redacts_before_a_sibling_resolves_its_secret(
    tmp_path,
) -> None:
    secret = "late-command-policy-sibling-secret-canary"

    class SecretBearingCommandPolicy(CommandPolicy):
        async def evaluate(
            self,
            ctx: ToolContext,
            request: CommandRequest,
        ) -> CommandPolicyResult:
            del ctx
            assert request.env is not None
            return CommandPolicyResult(
                decision=CommandPolicyDecision.DENY,
                reason=f"Deny {request.env['TOKEN']}",
            )

    class ResolveCommandSiblingSecretTool(Tool):
        spec = ToolSpec(
            name="resolve_command_sibling_secret",
            description="Resolve the command sibling's argument value.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.NONE,
        )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del args
            assert ctx.vault is not None
            await ctx.vault.resolve(SecretRef(name="shared"))
            return ToolResult(content="resolved")

    async def run() -> None:
        session_id = "late-command-policy-sibling"
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        hook = _CaptureAfterToolResultsHook()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_command_denied",
                        name="exec_command",
                        arguments={
                            "argv": ["/bin/echo", "safe"],
                            "env": {"TOKEN": secret},
                        },
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_resolves_command_secret",
                        name="resolve_command_sibling_secret",
                        arguments={"provided": secret},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            runtime_hooks=[hook],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=LocalRunner(tmp_path),
                vault=StaticVault({"shared": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[
                ExecCommandTool(policy=SecretBearingCommandPolicy()),
                ResolveCommandSiblingSecretTool(),
            ],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )

        blocked = next(event for event in events if event.type is EventType.TOOL_CALL_BLOCKED)
        assert blocked.payload["denied_by"] == "command_policy"
        assert blocked.payload["reason"] == f"Deny {REDACTED_SECRET}"
        assert blocked.payload["result"]["content"] == (
            f"Command denied by policy. Deny {REDACTED_SECRET}"
        )
        assert blocked.payload["result"]["structured"]["reason"] == (f"Deny {REDACTED_SECRET}")
        assert secret not in repr(hook.results)
        assert secret not in repr(events)
        assert secret not in repr(sink.events)
        assert secret not in repr(await store.load_events(session_id))
        assert secret not in repr(await store.load_transcript(session_id))
        assert secret not in repr(provider.requests[1].messages)

    asyncio.run(run())


@pytest.mark.parametrize("max_parallel_tool_calls", [1, 4])
@pytest.mark.parametrize("result_is_error", [False, True])
@pytest.mark.parametrize(
    "store_kind",
    ["memory", "sqlite", pytest.param("postgres", marks=pytest.mark.postgres)],
)
def test_multi_call_result_redacts_argument_before_a_sibling_resolves_its_secret(
    tmp_path,
    request,
    max_parallel_tool_calls: int,
    result_is_error: bool,
    store_kind: str,
) -> None:
    secret = "late-tool-result-sibling-secret-canary"
    wrapped_argument = f"prefix::{secret}::suffix"

    class EchoOrResolveSiblingSecretTool(Tool):
        spec = ToolSpec(
            name="echo_or_resolve_sibling_secret",
            description="Echo one argument or resolve the shared sibling secret.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.NONE,
        )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            if args["resolve"]:
                assert ctx.vault is not None
                await ctx.vault.resolve(SecretRef(name="shared"))
                return ToolResult(content="resolved")
            parsed = args["provided"].split("::")[1]
            return ToolResult(
                content=parsed,
                structured={"provided": parsed},
                is_error=result_is_error,
            )

    async def run() -> None:
        session_id = f"late-tool-result-sibling-{max_parallel_tool_calls}-{result_is_error}"
        if store_kind == "memory":
            store = InMemorySessionStore()
        elif store_kind == "sqlite":
            store = SQLiteSessionStore(
                tmp_path / f"late-result-{max_parallel_tool_calls}-{result_is_error}.sqlite"
            )
        else:
            store = PostgresSessionStore(
                request.getfixturevalue("postgres_dsn"),
                min_size=1,
                max_size=4,
                schema_mode=SchemaMode.CREATE,
            )
        sink = InMemoryEventSink()
        hook = _CaptureAfterToolResultsHook()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_echoes_before_resolution",
                        name="echo_or_resolve_sibling_secret",
                        arguments={"resolve": False, "provided": wrapped_argument},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_resolves_result_secret",
                        name="echo_or_resolve_sibling_secret",
                        arguments={"resolve": True, "provided": wrapped_argument},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            runtime_hooks=[hook],
            enable_logging=False,
            config=CayuConfig(
                tool_execution=ToolExecutionConfig(max_parallel_tool_calls=max_parallel_tool_calls)
            ),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                vault=StaticVault({"shared": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[EchoOrResolveSiblingSecretTool()],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )

        terminal = [
            event
            for event in events
            if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        ]
        assert len(terminal) == 2
        echoed = terminal[0].payload["result"]
        resolved = terminal[1].payload["result"]
        assert echoed["content"] == REDACTED_SECRET
        assert echoed["structured"] == {"provided": REDACTED_SECRET}
        assert resolved["content"] == "resolved"
        failed = [event for event in terminal if event.type is EventType.TOOL_CALL_FAILED]
        assert len(failed) == (1 if result_is_error else 0)
        assert all(event.payload["result"]["is_error"] for event in failed)
        assert hook.results[0].content == REDACTED_SECRET
        assert hook.results[0].structured == {"provided": REDACTED_SECRET}
        assert hook.results[1].content == "resolved"
        assert secret not in repr(events)
        assert secret not in repr(sink.events)
        assert secret not in repr(await store.load_events(session_id))
        assert secret not in repr(await store.load_transcript(session_id))
        assert secret not in repr(provider.requests[1].messages)
        if isinstance(store, (SQLiteSessionStore, PostgresSessionStore)):
            await store.close()

    asyncio.run(run())


def test_restart_fails_closed_for_a_partial_staged_multi_call_round() -> None:
    secret = "partial-stage-sibling-secret-canary"

    class EchoOrResolveTool(Tool):
        spec = ToolSpec(
            name="partial_stage_echo_or_resolve",
            description="Echo the provided value or resolve its sibling secret.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.NONE,
            parallel_safe=False,
            execution_profile_identity=_test_behavior_identity(
                "partial-stage-echo-or-resolve-tool"
            ),
        )

        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            action = args["action"]
            self.calls.append(action)
            if action == "resolve":
                assert ctx.vault is not None
                await ctx.vault.resolve(SecretRef(name="shared"))
                return ToolResult(content="resolved")
            return ToolResult(content=args["provided"])

    async def run() -> None:
        session_id = "partial-staged-multi-call-restart"
        store = _CommitFirstStagedTerminalThenRaiseStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_echo_before_crash",
                        name="partial_stage_echo_or_resolve",
                        arguments={"action": "echo", "provided": secret},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_resolve_after_crash",
                        name="partial_stage_echo_or_resolve",
                        arguments={"action": "resolve", "provided": secret},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("recovered"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        first_tool = EchoOrResolveTool()
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(provider, default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("dynamic"),
                vault=StaticVault({"shared": secret}),
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[first_tool],
        )

        first_events = await collect_events(
            first_app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )

        assert store.lost_stage_acknowledgement is True
        assert first_events[-1].type is EventType.SESSION_FAILED
        assert first_tool.calls == ["echo"]
        checkpoint = await store.load_checkpoint(session_id)
        pending = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        assert pending is not None
        assert [item.tool_call_id for item in pending.staged_terminals] == [
            "call_echo_before_crash"
        ]

        recovery_tool = EchoOrResolveTool()
        recovery_app = CayuApp(session_store=store, enable_logging=False)
        recovery_app.register_provider(provider, default=True)
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("dynamic"),
                vault=StaticVault({"shared": secret}),
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[recovery_tool],
        )

        recovered_events = [
            event
            async for event in recovery_app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                )
            )
        ]

        assert recovered_events[-1].type is EventType.SESSION_COMPLETED
        assert recovery_tool.calls == []
        durable_events = await store.load_events(session_id)
        echoed_terminal = next(
            event
            for event in durable_events
            if event.payload.get("tool_call_id") == "call_echo_before_crash"
            and event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        )
        assert echoed_terminal.type is EventType.TOOL_CALL_FAILED
        assert echoed_terminal.payload["result"]["structured"]["error"] == ("invalid_tool_output")
        assert secret not in repr(first_events)
        assert secret not in repr(recovered_events)
        assert secret not in repr(durable_events)
        assert secret not in repr(await store.load_transcript(session_id))
        assert secret not in repr(provider.requests[1].messages)

    asyncio.run(run())


@pytest.mark.parametrize("pause_kind", ["approval", "user_input"])
@pytest.mark.parametrize("loss_boundary", ["stage", "terminal"])
def test_pause_retry_reuses_a_staged_terminal_after_acknowledgement_loss(
    pause_kind: str,
    loss_boundary: str,
) -> None:
    class PauseFirstCallPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            if pause_kind == "approval" and request.tool_call_id == "call_pause":
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                    reason="Approve the first call.",
                )
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    class CountingEffectTool(Tool):
        spec = ToolSpec(
            name="staged_ack_effect",
            description="Record executions across a staged-terminal retry.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.EXTERNAL,
            parallel_safe=False,
        )

        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, Any]] = []

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx
            self.calls.append(dict(args))
            return ToolResult(content=f"completed:{args['value']}")

    async def run() -> None:
        session_id = f"staged-ack-{pause_kind}"
        first_call = ModelStreamEvent.tool_call(
            id="call_pause",
            name=("staged_ack_effect" if pause_kind == "approval" else "ask_user"),
            arguments=(
                {"value": "first"}
                if pause_kind == "approval"
                else {"question": "Continue?", "options": ["yes", "no"]}
            ),
        )
        provider = FakeProvider(
            [
                [
                    first_call,
                    ModelStreamEvent.tool_call(
                        id="call_sibling",
                        name="staged_ack_effect",
                        arguments={"value": "second"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        store: InMemorySessionStore
        if loss_boundary == "stage":
            store = _CommitFirstStagedTerminalThenRaiseStore()
        else:
            store = _CommitFirstTerminalEventThenRaiseStore()
        tool = CountingEffectTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="dynamic"),
                vault=StaticVault({"unused": "unused-secret-canary"}),
            ),
            default=True,
        )
        tools: list[Tool] = [tool]
        if pause_kind == "user_input":
            tools.insert(0, UserInputTool())
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=tools,
            tool_policy=PauseFirstCallPolicy(),
        )

        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
        if pause_kind == "approval":
            approval = next(
                event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            resolution: ToolApprovalRequest | UserInputResponse = ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.payload["approval_id"],
                tool_round_id=approval.payload["tool_round_id"],
                tool_call_id=approval.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            )
        else:
            awaiting = next(
                event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            resolution = UserInputResponse(
                session_id=session_id,
                input_id=awaiting.payload["input_id"],
                answer="yes",
            )

        async def resolve() -> list[Any]:
            if isinstance(resolution, ToolApprovalRequest):
                return await collect_tool_approval_events(app, resolution)
            return [event async for event in app.resolve_user_input(resolution)]

        first_attempt = await resolve()
        if isinstance(store, _CommitFirstStagedTerminalThenRaiseStore):
            assert store.lost_stage_acknowledgement is True
        else:
            assert isinstance(store, _CommitFirstTerminalEventThenRaiseStore)
            assert store.lost_terminal_acknowledgement is True
        assert first_attempt[-1].type is EventType.SESSION_INTERRUPTED
        assert first_attempt[-1].payload.get("manual_recovery_required") is not True

        retry = await resolve()
        assert retry[-1].type is EventType.SESSION_COMPLETED, retry[-1].payload
        expected_calls = (
            ([{"value": "first"}] if pause_kind == "approval" else [])
            if loss_boundary == "stage"
            else (
                [{"value": "first"}, {"value": "second"}]
                if pause_kind == "approval"
                else [{"value": "second"}]
            )
        )
        assert tool.calls == expected_calls
        durable_events = await store.load_events(session_id)
        first_call_starts = [
            event
            for event in durable_events
            if event.type is EventType.TOOL_CALL_STARTED
            and event.payload.get("tool_call_id") == "call_pause"
        ]
        assert len(first_call_starts) == 1
        first_call_terminals = [
            event
            for event in durable_events
            if event.type
            in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
                EventType.TOOL_CALL_APPROVAL_DENIED,
            }
            and event.payload.get("tool_call_id") == "call_pause"
        ]
        assert len(first_call_terminals) == 1
        sibling_starts = [
            event
            for event in durable_events
            if event.type is EventType.TOOL_CALL_STARTED
            and event.payload.get("tool_call_id") == "call_sibling"
        ]
        sibling_terminals = [
            event
            for event in durable_events
            if event.type
            in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
                EventType.TOOL_CALL_APPROVAL_DENIED,
            }
            and event.payload.get("tool_call_id") == "call_sibling"
        ]
        assert len(sibling_terminals) == 1
        publication_status = app.tool_terminal_publication_status()
        assert publication_status.maximum_reserved_round_bytes > 0
        assert publication_status.active_round_reservations == 0
        assert publication_status.active_exclusive_rounds == 0
        if loss_boundary == "stage":
            assert sibling_starts == []
            assert sibling_terminals[0].type is EventType.TOOL_CALL_BLOCKED
            assert sibling_terminals[0].payload["result"]["structured"] == {
                "error": "invalid_tool_output",
                "executed": False,
                "outcome_unknown": False,
                "recovered": True,
                "reason": "continuation_secret_scope_unavailable",
            }
        else:
            assert len(sibling_starts) == 1

    asyncio.run(run())


@pytest.mark.parametrize("pause_kind", ["approval", "user_input"])
def test_pause_restart_rejects_static_to_dynamic_secret_scope_upgrade(
    pause_kind: str,
) -> None:
    secret = "restart-capability-secret-canary"
    environment_name = "restart-capability"

    class PausePolicy(ToolPolicy):
        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return _test_behavior_identity(f"pause-policy:{pause_kind}")

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            if pause_kind == "approval" and request.tool_call_id == "call_pause":
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                    reason=f"Approve {secret}",
                    metadata={"provided": secret},
                )
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    class EchoOrResolveAfterRestartTool(Tool):
        spec = ToolSpec(
            name="restart_scope_tool",
            description="Echo a value or resolve the restart secret.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.NONE,
            parallel_safe=False,
            execution_profile_identity=_test_behavior_identity(
                "echo-or-resolve-after-restart-tool"
            ),
        )

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            self.calls.append(dict(args))
            if args.get("resolve"):
                assert ctx.vault is not None
                await ctx.vault.resolve(SecretRef(name="restart_secret"))
            return ToolResult(content=args["provided"])

    async def run() -> None:
        session_id = f"pause-static-to-dynamic-restart-{pause_kind}"
        first_call = (
            ModelStreamEvent.tool_call(
                id="call_pause",
                name="restart_scope_tool",
                arguments={"provided": secret},
            )
            if pause_kind == "approval"
            else ModelStreamEvent.tool_call(
                id="call_pause",
                name="ask_user",
                arguments={"question": secret, "options": ["yes", secret]},
            )
        )
        calls = [first_call]
        if pause_kind == "user_input":
            calls.append(
                ModelStreamEvent.tool_call(
                    id="call_echo",
                    name="restart_scope_tool",
                    arguments={"provided": secret},
                )
            )
        calls.extend(
            [
                ModelStreamEvent.tool_call(
                    id="call_resolve",
                    name="restart_scope_tool",
                    arguments={"provided": "resolved", "resolve": True},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
        provider = FakeProvider(
            [
                calls,
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        store = InMemorySessionStore()
        tool = EchoOrResolveAfterRestartTool()

        def register_runtime(app: CayuApp, *, dynamic: bool) -> None:
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(
                    _portable_environment_spec(environment_name),
                    vault=(StaticVault({"restart_secret": secret}) if dynamic else None),
                ),
                default=True,
            )
            tools: list[Tool] = [tool]
            if pause_kind == "user_input":
                tools.insert(0, UserInputTool())
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=tools,
                tool_policy=PausePolicy(),
            )

        first_app = CayuApp(session_store=store, enable_logging=False)
        register_runtime(first_app, dynamic=False)
        paused = await collect_events(
            first_app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )

        restarted_app = CayuApp(session_store=store, enable_logging=False)
        register_runtime(restarted_app, dynamic=True)
        if pause_kind == "approval":
            approval = next(
                event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            with pytest.raises(ExecutionProfileMismatchError) as exc_info:
                await collect_tool_approval_events(
                    restarted_app,
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.payload["approval_id"],
                        tool_round_id=approval.payload["tool_round_id"],
                        tool_call_id=approval.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    ),
                )
        else:
            awaiting = next(
                event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            with pytest.raises(ExecutionProfileMismatchError) as exc_info:
                _ = [
                    event
                    async for event in restarted_app.resolve_user_input(
                        UserInputResponse(
                            session_id=session_id,
                            input_id=awaiting.payload["input_id"],
                            answer="yes",
                        )
                    )
                ]

        assert exc_info.value.changed_component_classes == (
            ExecutionProfileComponentClass.EFFECT_AUTHORITY,
        )

        assert secret in repr(paused)
        assert tool.calls == []
        assert len(provider.requests) == 1
        session = await store.load(session_id)
        assert session is not None and session.status is SessionStatus.INTERRUPTED

    asyncio.run(run())


def test_restart_preserves_completed_hooks_and_fails_closed_for_pending_hooks() -> None:
    class SafeResultTool(Tool):
        spec = ToolSpec(
            name="staged_hook_result",
            description="Return one safe result for staged-hook recovery.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.NONE,
            execution_profile_identity=_test_behavior_identity("safe-result-tool"),
        )

        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx
            self.calls.append(args["value"])
            return ToolResult(content=args["value"])

    class CountingModifyHook(RuntimeHook):
        def __init__(self) -> None:
            self.calls: list[str] = []

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return _test_behavior_identity("counting-modify-hook")

        async def after_tool_call(
            self,
            context: ToolCallHookContext,
        ) -> AfterToolCallDecision:
            self.calls.append(context.tool_call_id)
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(content=f"hooked:{context.result.content}"),
            )

    async def run() -> None:
        session_id = "staged-hook-completion-restart"
        store = _FailFirstTerminalEventStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_first_hook",
                        name="staged_hook_result",
                        arguments={"value": "first"},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_second_hook",
                        name="staged_hook_result",
                        arguments={"value": "second"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("recovered"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        hook = CountingModifyHook()
        first_tool = SafeResultTool()
        first_app = CayuApp(
            session_store=store,
            runtime_hooks=[hook],
            enable_logging=False,
        )
        first_app.register_provider(provider, default=True)
        first_app.register_environment(
            Environment(
                _portable_environment_spec("dynamic"),
                vault=StaticVault({"unused": "unused-secret-canary"}),
            ),
            default=True,
        )
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[first_tool],
        )

        first_events = await collect_events(
            first_app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
        assert first_events[-1].type is EventType.SESSION_FAILED
        assert sorted(first_tool.calls) == ["first", "second"]
        assert hook.calls == ["call_first_hook"]

        recovery_tool = SafeResultTool()
        recovery_app = CayuApp(
            session_store=store,
            runtime_hooks=[hook],
            enable_logging=False,
        )
        recovery_app.register_provider(provider, default=True)
        recovery_app.register_environment(
            Environment(
                _portable_environment_spec("dynamic"),
                vault=StaticVault({"unused": "unused-secret-canary"}),
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[recovery_tool],
        )

        recovered_events = [
            event
            async for event in recovery_app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                )
            )
        ]

        assert recovered_events[-1].type is EventType.SESSION_COMPLETED
        assert recovery_tool.calls == []
        # The first hook completed durably and is not repeated. The second
        # cannot run after restart because its dynamic invocation redactor was
        # intentionally not persisted with secret values.
        assert hook.calls == ["call_first_hook"]
        completed = [
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.TOOL_CALL_COMPLETED
        ]
        assert [event.payload["result"]["content"] for event in completed] == ["hooked:first"]
        failed = [
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.TOOL_CALL_FAILED
            and event.payload.get("tool_call_id") == "call_second_hook"
        ]
        assert len(failed) == 1
        assert failed[0].payload["result"]["structured"]["reason"] == (
            "recovery_hook_secret_scope_unavailable"
        )

    asyncio.run(run())


def test_mid_round_limit_publishes_completed_stage_before_interrupting(monkeypatch) -> None:
    clock = {"value": 0.0}
    monkeypatch.setattr(
        "cayu.runtime._session_engine.time.monotonic",
        lambda: clock["value"],
    )

    class AdvanceClockTool(Tool):
        spec = ToolSpec(
            name="stage_before_limit",
            description="Complete one call before the next limit check.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.EXTERNAL,
            parallel_safe=False,
        )

        def __init__(self) -> None:
            super().__init__()
            self.calls: list[int] = []

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx
            self.calls.append(args["number"])
            clock["value"] = 1.0
            return ToolResult(content=f"completed:{args['number']}")

    async def run() -> None:
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_completed_before_limit",
                        name="stage_before_limit",
                        arguments={"number": 1},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_skipped_at_limit",
                        name="stage_before_limit",
                        arguments={"number": 2},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("resumed"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        tool = AdvanceClockTool()
        app = CayuApp(
            session_store=store,
            config=CayuConfig(tool_execution=ToolExecutionConfig(max_parallel_tool_calls=1)),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="dynamic"),
                vault=StaticVault({"unused": "unused-secret"}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )

        limited_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="stage-before-limit",
                messages=[Message.text("user", "run")],
                limits=RunLimits(max_elapsed_seconds=1),
            ),
        )

        assert tool.calls == [1]
        assert limited_events[-1].type is EventType.SESSION_INTERRUPTED
        assert EventType.SESSION_FAILED not in {event.type for event in limited_events}
        durable_events = await store.load_events("stage-before-limit")
        skipped = [
            event
            for event in durable_events
            if event.type is EventType.TOOL_CALL_FAILED
            and event.payload.get("result", {}).get("structured", {}).get("skipped") is True
        ]
        assert [event.payload["tool_call_id"] for event in skipped] == ["call_skipped_at_limit"]
        completed = [
            event
            for event in durable_events
            if event.type is EventType.TOOL_CALL_COMPLETED
            and event.payload.get("tool_call_id") == "call_completed_before_limit"
        ]
        assert len(completed) == 1
        checkpoint = await store.load_checkpoint("stage-before-limit")
        assert tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint) is None
        transcript = await store.load_transcript("stage-before-limit")
        assert any(part.type == "tool_result" for message in transcript for part in message.content)

        clock["value"] = 1.0
        resumed_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="stage-before-limit",
                    messages=[Message.text("user", "continue")],
                    limits=RunLimits(max_elapsed_seconds=1),
                )
            )
        ]
        assert resumed_events[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == [1]

    asyncio.run(run())


def test_staged_result_projection_preserves_externalized_artifact_authority(tmp_path) -> None:
    class LargeResultTool(Tool):
        spec = ToolSpec(
            name="large_staged_result",
            description="Return a result that the projection policy externalizes.",
            input_schema={"type": "object", "additionalProperties": True},
        )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            if args["resolve"]:
                assert ctx.vault is not None
                await ctx.vault.resolve(SecretRef(name="artifact_prefix"))
            return ToolResult(content=args["value"] * 1_000)

    async def run() -> None:
        store = InMemorySessionStore()
        artifacts = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_large_one",
                        name="large_staged_result",
                        arguments={"value": "x", "resolve": False},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_large_two",
                        name="large_staged_result",
                        arguments={"value": "y", "resolve": True},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=store,
            tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
                max_inline_bytes=64,
                max_inline_token_estimate=None,
                preview_bytes=10,
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="dynamic"),
                # Every generated artifact id begins with this value. The
                # typed projection authority must survive checkpoint reload
                # so ordinary secret redaction cannot rewrite that identity.
                vault=StaticVault({"artifact_prefix": "art_"}),
                artifact_store=artifacts,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[LargeResultTool()],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="staged-result-projection-authority",
                messages=[Message.text("user", "run")],
            ),
        )

        assert events[-1].type is EventType.SESSION_COMPLETED
        terminal_events = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
        assert len(terminal_events) == 2
        references = [event.payload["result"]["artifacts"][-1] for event in terminal_events]
        assert all(reference["type"] == "cayu.tool_result_artifact.v1" for reference in references)
        assert [
            (await artifacts.read_bytes(reference["artifact_id"])).content
            for reference in references
        ] == [b"x" * 1_000, b"y" * 1_000]

    asyncio.run(run())


def test_observational_stage_tracks_projected_result_after_lost_acknowledgement(
    tmp_path,
) -> None:
    class ExternalLargeResultTool(Tool):
        spec = ToolSpec(
            name="external_large_staged_result",
            description="Externalize an effectful result before observational hooks.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.EXTERNAL,
            parallel_safe=False,
        )

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            self.calls.append(args["value"])
            if args["resolve"]:
                assert ctx.vault is not None
                await ctx.vault.resolve(SecretRef(name="unused"))
            return ToolResult(content=args["value"] * 1_000)

    async def run() -> None:
        session_id = "observational-stage-projection-ack-loss"
        store = _CommitFirstTerminalEventThenRaiseStore()
        artifacts = LocalArtifactStore(tmp_path / "recovery-artifacts", store_id="artifacts")
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_external_one",
                        name="external_large_staged_result",
                        arguments={"value": "x", "resolve": False},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_external_two",
                        name="external_large_staged_result",
                        arguments={"value": "y", "resolve": True},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                [
                    ModelStreamEvent.tool_call(
                        id="call_later_one",
                        name="external_large_staged_result",
                        arguments={"value": "a", "resolve": False},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_later_two",
                        name="external_large_staged_result",
                        arguments={"value": "b", "resolve": True},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        tool = ExternalLargeResultTool()
        app = CayuApp(
            session_store=store,
            tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
                max_inline_bytes=64,
                max_inline_token_estimate=None,
                preview_bytes=10,
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="dynamic"),
                vault=StaticVault({"unused": "unused-secret-canary"}),
                artifact_store=artifacts,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )

        first = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
        assert store.lost_terminal_acknowledgement is True
        assert first[-1].type is EventType.SESSION_FAILED
        assert tool.calls == ["x", "y"]

        resumed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                )
            )
        ]

        assert resumed[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == ["x", "y"]
        durable = await store.load_events(session_id)
        terminals = [
            event
            for event in durable
            if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        ]
        assert len(terminals) == 2
        completed = next(
            event for event in terminals if event.payload["tool_call_id"] == "call_external_one"
        )
        assert completed.type is EventType.TOOL_CALL_COMPLETED
        assert (
            completed.payload["result"]["artifacts"][-1]["type"] == "cayu.tool_result_artifact.v1"
        )
        unavailable = next(
            event for event in terminals if event.payload["tool_call_id"] == "call_external_two"
        )
        assert unavailable.type is EventType.TOOL_CALL_FAILED
        assert unavailable.payload["result"]["structured"] == {
            "error": "invalid_tool_output",
            "reason": "recovery_hook_secret_scope_unavailable",
            "outcome_unknown": True,
            "recovered": True,
        }
        publication_status = app.tool_terminal_publication_status()
        assert publication_status.active_round_reservations == 0
        assert publication_status.active_exclusive_rounds == 0

        later = await asyncio.wait_for(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=f"{session_id}-later",
                    messages=[Message.text("user", "run later")],
                ),
            ),
            timeout=2,
        )
        assert later[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == ["x", "y", "a", "b"]

    asyncio.run(run())


@pytest.mark.parametrize("max_parallel_tool_calls", [1, 4])
@pytest.mark.parametrize("structural_secret", [None, "tool_call_id"])
def test_dynamic_multi_call_preserves_unrelated_results_without_secret_resolution(
    max_parallel_tool_calls: int,
    structural_secret: str | None,
) -> None:
    class UsefulResultTool(Tool):
        spec = ToolSpec(
            name="useful_dynamic_result",
            description="Return useful output without resolving a secret.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.NONE,
        )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx, args
            return ToolResult(
                content="useful result",
                structured={"answer": 42, "status": "useful"},
                artifacts=[{"kind": "note", "content": "safe artifact"}],
            )

    async def run() -> None:
        session_id = f"safe-dynamic-results-{max_parallel_tool_calls}"
        store = InMemorySessionStore()
        hook = _CaptureAfterToolResultsHook()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_safe_first",
                        name="useful_dynamic_result",
                        arguments={"position": "first"},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_safe_second",
                        name="useful_dynamic_result",
                        arguments={"position": "second"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=store,
            runtime_hooks=[hook],
            secret_redactor=(
                SecretRedactor() if structural_secret is None else SecretRedactor(structural_secret)
            ),
            enable_logging=False,
            config=CayuConfig(
                tool_execution=ToolExecutionConfig(max_parallel_tool_calls=max_parallel_tool_calls)
            ),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="dynamic"),
                vault=StaticVault({"unused": "unused-secret-canary"}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UsefulResultTool()],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )

        completed = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
        assert completed, [(event.type, event.payload) for event in events]
        assert [event.payload["result"]["content"] for event in completed] == [
            "useful result",
            "useful result",
        ]
        assert all(
            event.payload["result"]["structured"] == {"answer": 42, "status": "useful"}
            for event in completed
        )
        assert all(
            event.payload["result"]["artifacts"] == [{"kind": "note", "content": "safe artifact"}]
            for event in completed
        )
        assert [result.structured for result in hook.results] == [
            {"answer": 42, "status": "useful"},
            {"answer": 42, "status": "useful"},
        ]
        assert "useful" in repr(await store.load_transcript(session_id))
        assert "useful" in repr(provider.requests[1].messages)

    asyncio.run(run())


def test_dynamic_single_call_preserves_before_hook_result_without_a_sibling() -> None:
    class SafeShortCircuitHook(RuntimeHook):
        async def before_tool_call(
            self,
            context: BeforeToolCallHookContext,
        ) -> BeforeToolCallDecision:
            return BeforeToolCallDecision(
                action="short_circuit",
                synthetic_result=ToolResult(
                    content=f"cached {context.arguments['query']}",
                    structured={"cache": "hit"},
                ),
            )

    async def run() -> None:
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_safe_short_circuit",
                        name="resolve_after_start",
                        arguments={"query": "ordinary input"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        app = CayuApp(
            runtime_hooks=[SafeShortCircuitHook()],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="dynamic"),
                vault=StaticVault({"unused": "unused-secret-canary"}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_ResolveAfterStartTool()],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "run")],
            ),
        )

        completed = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
        assert completed.payload["result"]["content"] == "cached ordinary input"
        assert completed.payload["result"]["structured"] == {"cache": "hit"}

    asyncio.run(run())


@pytest.mark.parametrize("hook_output", ["failure", "custom_event"])
def test_dynamic_single_call_quarantines_before_hook_publication(
    hook_output: str,
) -> None:
    secret = "late-single-hook-publication-secret-canary"

    class PublishingBeforeHook(RuntimeHook):
        async def before_tool_call(
            self,
            context: BeforeToolCallHookContext,
        ) -> None:
            if hook_output == "custom_event":
                await context.emit_custom_event(
                    "custom.before_tool_arguments",
                    payload={"provided": context.arguments["provided"]},
                )
                return
            raise RuntimeError(context.arguments["provided"])

    async def run() -> None:
        session_id = f"single-before-hook-{hook_output}"
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_single_hook",
                        name="resolve_after_start",
                        arguments={"provided": secret},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            runtime_hooks=[PublishingBeforeHook()],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="dynamic"),
                vault=StaticVault({"api_key": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_ResolveAfterStartTool()],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )

        hook_failure = next(event for event in events if event.type is EventType.HOOK_FAILED)
        assert hook_failure.payload["error_type"] == "runtime_hook_failure"
        assert hook_failure.payload["actions"] == []
        assert not any(str(event.type).startswith("custom.") for event in events)
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert secret not in repr(events)
        assert secret not in repr(sink.events)
        assert secret not in repr(await store.load_events(session_id))

    asyncio.run(run())


@pytest.mark.parametrize("max_parallel_tool_calls", [1, 4])
@pytest.mark.parametrize("hook_outcome", ["block", "short_circuit", "failure"])
def test_multi_call_before_hook_output_redacts_before_a_sibling_resolves_its_secret(
    max_parallel_tool_calls: int,
    hook_outcome: str,
) -> None:
    secret = "late-before-hook-sibling-secret-canary"

    class SecretBearingBeforeHook(RuntimeHook):
        def __init__(self) -> None:
            self.after_results: list[ToolResult] = []

        async def before_tool_call(
            self,
            context: BeforeToolCallHookContext,
        ) -> BeforeToolCallDecision | None:
            if context.tool_call_id != "call_hook_output":
                return None
            provided = context.arguments["provided"]
            if hook_outcome == "block":
                return BeforeToolCallDecision(
                    action="block",
                    block_reason=f"Blocked {provided}",
                )
            if hook_outcome == "short_circuit":
                return BeforeToolCallDecision(
                    action="short_circuit",
                    synthetic_result=ToolResult(
                        content=provided,
                        structured={"provided": provided},
                    ),
                )
            raise RuntimeError(provided)

        async def after_tool_call(
            self,
            context: ToolCallHookContext,
        ) -> AfterToolCallDecision | None:
            self.after_results.append(context.result)
            if context.tool_call_id != "call_hook_output":
                return None
            if hook_outcome == "failure":
                raise RuntimeError(secret)
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(
                    content=secret,
                    structured={"provided": secret},
                ),
            )

    class ResolveHookSiblingSecretTool(Tool):
        spec = ToolSpec(
            name="resolve_hook_sibling_secret",
            description="Resolve the shared secret for only one sibling call.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.NONE,
        )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            if args["resolve"]:
                assert ctx.vault is not None
                await ctx.vault.resolve(SecretRef(name="shared"))
            return ToolResult(content="done")

    async def run() -> None:
        session_id = f"late-before-hook-{hook_outcome}-{max_parallel_tool_calls}"
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_hook_output",
                        name="resolve_hook_sibling_secret",
                        arguments={"resolve": False, "provided": secret},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_resolves_hook_secret",
                        name="resolve_hook_sibling_secret",
                        arguments={"resolve": True, "provided": secret},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        hook = SecretBearingBeforeHook()
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            runtime_hooks=[hook],
            enable_logging=False,
            config=CayuConfig(
                tool_execution=ToolExecutionConfig(max_parallel_tool_calls=max_parallel_tool_calls)
            ),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                vault=StaticVault({"shared": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[ResolveHookSiblingSecretTool()],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )

        if hook_outcome == "block":
            terminal = next(event for event in events if event.type is EventType.TOOL_CALL_BLOCKED)
            assert terminal.payload["reason"] == f"Blocked {REDACTED_SECRET}"
        elif hook_outcome == "short_circuit":
            terminal = next(
                event
                for event in events
                if event.type is EventType.TOOL_CALL_COMPLETED
                and event.payload["result"]["structured"] == {"provided": REDACTED_SECRET}
            )
            assert terminal.payload["result"]["content"] == REDACTED_SECRET
        else:
            failed_hooks = [event for event in events if event.type is EventType.HOOK_FAILED]
            assert len(failed_hooks) == 2
            before_failure = next(
                event
                for event in failed_hooks
                if event.payload["error_type"] == "runtime_hook_failure"
            )
            after_failure = next(
                event for event in failed_hooks if event.payload["error_type"] == "RuntimeError"
            )
            assert "error" not in before_failure.payload
            assert after_failure.payload["error"] == REDACTED_SECRET
            assert all(event.payload["actions"] == [] for event in failed_hooks)
            ordinary = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
            assert len(ordinary) == 2
            assert all(event.payload["result"]["content"] == "done" for event in ordinary)
        assert len(hook.after_results) == 2
        assert all(secret not in repr(result) for result in hook.after_results)
        assert secret not in repr(events)
        assert secret not in repr(sink.events)
        assert secret not in repr(await store.load_events(session_id))
        assert secret not in repr(await store.load_transcript(session_id))
        assert secret not in repr(provider.requests[1].messages)

    asyncio.run(run())


@pytest.mark.parametrize("pause_kind", ["approval", "user_input"])
@pytest.mark.parametrize("scope_kind", ["static", "dynamic", "factory_reconnect"])
def test_pause_continuation_reauthorization_uses_durable_policy_output_scope(
    pause_kind: str,
    scope_kind: str,
) -> None:
    dynamic_scope = scope_kind != "static"
    secret = "late-reauthorization-policy-secret-canary"
    diagnostic = secret if dynamic_scope else "safe static reauthorization denial"

    class ContinuationPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            if pause_kind == "approval" and request.tool_call_id == "call_pause":
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                    reason=diagnostic,
                    metadata={"diagnostic": diagnostic},
                )
            if request.arguments.get("mode") == "forbidden":
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.DENY,
                    reason=diagnostic,
                    metadata={"diagnostic": diagnostic},
                )
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    class RewriteContinuationCall(RuntimeHook):
        async def before_tool_call(
            self,
            context: BeforeToolCallHookContext,
        ) -> BeforeToolCallDecision | None:
            if context.tool_call_id != "call_reauthorized":
                return None
            arguments = dict(context.arguments)
            arguments["mode"] = "forbidden"
            return BeforeToolCallDecision(
                action="proceed_modified",
                modified_arguments=arguments,
            )

    class ContinuationTool(Tool):
        spec = ToolSpec(
            name="continuation_tool",
            description="Run after a durable pause.",
            input_schema={"type": "object", "additionalProperties": True},
            effect=ToolEffect.NONE,
        )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            if args.get("resolve"):
                assert ctx.vault is not None
                await ctx.vault.resolve(SecretRef(name="policy_secret"))
            return ToolResult(content="done")

    class ChangingCapabilityFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.operations: list[EnvironmentFactoryOperation] = []

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            self.operations.append(request.operation)
            environment = Environment(
                EnvironmentSpec(name="factory"),
                vault=(
                    StaticVault({"policy_secret": secret})
                    if request.operation is EnvironmentFactoryOperation.RECONNECT
                    else None
                ),
            )
            return EnvironmentFactoryResult(
                environment=environment,
                reconnect_metadata={"allocation": "factory-test"},
            )

    async def run() -> None:
        session_id = f"reauthorization-{pause_kind}-{scope_kind}"
        first_call = (
            ModelStreamEvent.tool_call(
                id="call_pause",
                name="continuation_tool",
                arguments={"mode": "allowed"},
            )
            if pause_kind == "approval"
            else ModelStreamEvent.tool_call(
                id="call_pause",
                name="ask_user",
                arguments={"question": diagnostic, "options": ["yes", "no"]},
            )
        )
        provider = FakeProvider(
            [
                [
                    first_call,
                    ModelStreamEvent.tool_call(
                        id="call_reauthorized",
                        name="continuation_tool",
                        arguments={"mode": "allowed"},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_resolves_policy_secret",
                        name="continuation_tool",
                        arguments={"mode": "allowed", "resolve": dynamic_scope},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        factory: ChangingCapabilityFactory | None = None
        if scope_kind == "dynamic":
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="dynamic"),
                    vault=StaticVault({"policy_secret": secret}),
                ),
                default=True,
            )
        elif scope_kind == "factory_reconnect":
            factory = ChangingCapabilityFactory()
            app.register_environment_factory(
                EnvironmentSpec(name="factory"),
                factory,
                default=True,
            )
        tools: list[Tool] = [ContinuationTool()]
        if pause_kind == "user_input":
            tools.insert(0, UserInputTool())
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=tools,
            tool_policy=ContinuationPolicy(),
            runtime_hooks=[RewriteContinuationCall()],
        )

        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
        if dynamic_scope:
            assert secret not in repr(paused)
            assert secret not in repr(await store.load_events(session_id))
        if pause_kind == "approval":
            approval = next(
                event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            resumed = await collect_tool_approval_events(
                app,
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=approval.payload["approval_id"],
                    tool_round_id=approval.payload["tool_round_id"],
                    tool_call_id=approval.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                ),
            )
        else:
            awaiting = next(
                event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            resumed = [
                event
                async for event in app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id,
                        input_id=awaiting.payload["input_id"],
                        answer="yes",
                    )
                )
            ]

        blocked = next(event for event in resumed if event.type is EventType.TOOL_CALL_BLOCKED)
        expected_reason = "Tool call denied by policy." if dynamic_scope else diagnostic
        assert blocked.payload["blocked_by"] == "tool_policy_reauthorization"
        assert blocked.payload["reason"] == expected_reason
        assert blocked.payload["metadata"] == ({} if dynamic_scope else {"diagnostic": diagnostic})
        assert blocked.payload["result"]["content"] == expected_reason
        if dynamic_scope:
            assert secret not in repr(resumed)
            assert secret not in repr(await store.load_events(session_id))
            assert secret not in repr(await store.load_transcript(session_id))
            assert secret not in repr(provider.requests[1].messages)
        if factory is not None:
            assert factory.operations == [
                EnvironmentFactoryOperation.CREATE,
                EnvironmentFactoryOperation.RECONNECT,
            ]

    asyncio.run(run())
