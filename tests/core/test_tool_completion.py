from __future__ import annotations

import asyncio

import pytest
from tests.core.test_tool_round_publication_failure_matrix import (
    _SimulatedProcessLoss,
    _TwoCallProvider,
)
from tests.core.test_tool_round_publication_failure_matrix import (
    store_factory as store_factory,
)

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    ExecutionProfileBehaviorIdentity,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    RequireFinalTool,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SessionStatus,
    Tool,
    ToolCompletionPolicy,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)


class FinalTool(Tool):
    def __init__(self, effect=ToolEffect.IDEMPOTENT, *, fails=False):
        self.spec = ToolSpec(
            name="ask_customer",
            description="Record a clarification question.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            effect=effect,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:tool-completion",
                behavior_version="1",
                implementation_version="1",
            ),
        )
        super().__init__()
        self.calls = 0
        self.fails = fails

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls += 1
        return ToolResult(
            content="Which order?", structured={"question": "Which order?"}, is_error=self.fails
        )


def call():
    return [
        ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="c1"),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def app_for(store, provider, tool, **options):
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="support", model="scripted-model"), tools=[tool], **options)
    return app


@pytest.mark.parametrize("effect", [ToolEffect.NONE, ToolEffect.IDEMPOTENT])
@pytest.mark.parametrize("max_steps", [1, 8])
def test_success_completes_without_another_provider_request(effect, max_steps):
    async def scenario():
        store = InMemorySessionStore()
        provider = ScriptedModelProvider([call()])
        tool = FinalTool(effect)
        app = app_for(store, provider, tool)
        events = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="support",
                    session_id="s",
                    messages=[Message.text("user", "Help")],
                    max_steps=max_steps,
                    tool_completion=ToolCompletionPolicy(tool_names=["ask_customer"]),
                    loop_policies=(RequireFinalTool(["ask_customer"]),),
                )
            )
        ]
        assert (await store.load("s")).status == SessionStatus.COMPLETED, [
            (e.type, e.payload) for e in events
        ]
        assert tool.calls == 1
        assert len(provider.requests) == 1
        completed = next(e for e in events if e.type == "session.completed")
        assert completed.payload["reason"] == "host_rendered_tool"
        result = completed.payload["tool_completion"]
        assert result["call"]["tool_name"] == "ask_customer"
        assert result["result"]["structured"] == {"question": "Which order?"}
        assert result["effect"] == effect
        assert not [e for e in events if e.type == "custom.loop.before_stop.selected"]
        transcript = await store.load_transcript("s")
        assert [m.role for m in transcript] == ["user", "assistant", "tool"]

    asyncio.run(scenario())


class CrashStore:
    invocation_lifecycle_command_version = 1

    def __init__(self, *args, boundary, **kwargs):
        super().__init__(*args, **kwargs)
        self.boundary = boundary
        self.crashed = False

    def lose_process(self, message):
        self.crashed = True
        raise _SimulatedProcessLoss(message)

    async def append_event(self, session_id, event):
        if (
            self.boundary == "completion"
            and event.type == EventType.SESSION_COMPLETED
            and not self.crashed
        ):
            assert (await self.load(session_id)).status == SessionStatus.COMPLETED
            self.lose_process("lost after successful interaction settlement")
        await super().append_event(session_id, event)
        if (
            self.boundary == "tool-event"
            and event.type == EventType.TOOL_CALL_COMPLETED
            and not self.crashed
        ):
            self.lose_process("lost after successful tool event")

    async def publish_runtime_publication(self, *args, **kwargs):
        result = await super().publish_runtime_publication(*args, **kwargs)
        if (
            self.boundary == "tool-publication"
            and kwargs["request"].kind == "tool-round"
            and not self.crashed
        ):
            self.lose_process("lost after successful tool publication")
        return result


@pytest.mark.parametrize("boundary", ["tool-event", "tool-publication", "completion"])
def test_reconstruction_completes_from_durable_success(store_factory, boundary):
    session_id = f"completion-{boundary}"

    async def scenario():
        async with store_factory(CrashStore, boundary=boundary) as store:
            first = _TwoCallProvider([call()])
            tool = FinalTool()
            app = app_for(store, first, tool)
            with pytest.raises(_SimulatedProcessLoss):
                [
                    e
                    async for e in app.run(
                        RunRequest(
                            agent_name="support",
                            session_id=session_id,
                            messages=[Message.text("user", "Help")],
                            max_steps=1,
                            tool_completion=ToolCompletionPolicy(tool_names=["ask_customer"]),
                        )
                    )
                ]
            assert store.crashed
            assert tool.calls == 1
            second = _TwoCallProvider([])
            reconstructed_tool = FinalTool()
            reconstructed = app_for(store, second, reconstructed_tool)
            recovered = await reconstructed.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id, inactive_for_seconds=0)
            )
            assert recovered.status == SessionStatus.COMPLETED, recovered
            assert reconstructed_tool.calls == 0
            assert second.requests == []
            events = await store.load_events(session_id)
            completion = [e for e in events if e.type == "session.completed"]
            assert len(completion) == 1
            assert completion[0].payload["reason"] == "host_rendered_tool"
            again = await reconstructed.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id, inactive_for_seconds=0)
            )
            assert again.status == SessionStatus.COMPLETED
            assert second.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["tool-event", "tool-publication"])
def test_recovery_checks_execution_permission_before_mutation(boundary, monkeypatch):
    class Store(CrashStore, InMemorySessionStore):
        invocation_lifecycle_command_version = 1

    async def scenario():
        store = Store(boundary=boundary)
        app = app_for(store, _TwoCallProvider([call()]), FinalTool())
        with pytest.raises(_SimulatedProcessLoss):
            [e async for e in app.run(request(tool_completion={"tool_names": ["ask_customer"]}))]
        provider = _TwoCallProvider([])
        tool = FinalTool()
        reconstructed = app_for(store, provider, tool)
        before = await store.load("s")
        checkpoint = await store.load_checkpoint("s")
        events = await store.load_events("s")

        async def denied(*_args):
            raise PermissionError("execution revoked")

        with monkeypatch.context() as patch:
            patch.setattr(
                reconstructed._session_engine._recovery_coordinator,
                "_require_participant_execution",
                denied,
            )
            with pytest.raises(PermissionError, match="execution revoked"):
                await reconstructed.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="s", inactive_for_seconds=0)
                )
        assert await store.load("s") == before
        assert await store.load_checkpoint("s") == checkpoint
        assert await store.load_events("s") == events
        assert not provider.requests
        assert tool.calls == 0
        recovered = await reconstructed.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="s", inactive_for_seconds=0)
        )
        assert recovered.status == SessionStatus.COMPLETED
        assert not provider.requests
        assert tool.calls == 0

    asyncio.run(scenario())


def request(**options):
    return RunRequest(
        agent_name="support",
        session_id=options.pop("session_id", "s"),
        messages=[Message.text("user", "Help")],
        **options,
    )


@pytest.mark.parametrize(
    "names",
    [
        [],
        [""],
        [" "],
        [" ask_customer"],
        ["ask_customer", "ask_customer"],
        "ask_customer",
        [1],
        [True],
    ],
)
@pytest.mark.parametrize("request_type", [RunRequest, ResumeRequest])
def test_invalid_names_fail_at_request_validation(names, request_type):
    with pytest.raises(ValueError):
        request_type(
            **({"agent_name": "support"} if request_type is RunRequest else {"session_id": "s"}),
            messages=[Message.text("user", "Help")],
            tool_completion={"tool_names": names},
        )


@pytest.mark.parametrize("names", [["missing"], ["__cayu_submit_structured_output"]])
def test_unregistered_and_runtime_owned_tools_fail_before_provider_request(names):
    async def scenario():
        provider = _TwoCallProvider([])
        app = app_for(InMemorySessionStore(), provider, FinalTool())
        with pytest.raises(ValueError, match="registered application tools"):
            [e async for e in app.run(request(tool_completion={"tool_names": names}))]
        assert provider.requests == []

    asyncio.run(scenario())


def test_external_tool_is_rejected_before_provider_request():
    async def scenario():
        provider = _TwoCallProvider([])
        app = app_for(InMemorySessionStore(), provider, FinalTool(ToolEffect.EXTERNAL))
        with pytest.raises(ValueError, match="none or idempotent"):
            [e async for e in app.run(request(tool_completion={"tool_names": ["ask_customer"]}))]
        assert provider.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_default_behavior_and_failed_result_keep_the_model_loop(configured, fails):
    async def scenario():
        store = InMemorySessionStore()
        provider = _TwoCallProvider(
            [
                call(),
                [
                    ModelStreamEvent.text_delta("Model reply"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        tool = FinalTool(fails=fails)
        app = app_for(store, provider, tool)
        events = [
            e
            async for e in app.run(
                request(tool_completion={"tool_names": ["ask_customer"]} if configured else None)
            )
        ]
        assert (await store.load("s")).status == SessionStatus.COMPLETED
        assert len(provider.requests) == (1 if configured and not fails else 2)
        assert tool.calls == 1
        completed = next(e for e in events if e.type == EventType.SESSION_COMPLETED)
        assert ("tool_completion" in completed.payload) == (configured and not fails)

    asyncio.run(scenario())


def test_sibling_calls_all_execute_and_follow_the_ordinary_loop():
    async def scenario():
        store = InMemorySessionStore()
        provider = _TwoCallProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="c1"),
                    ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="c2"),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        tool = FinalTool()
        app = app_for(store, provider, tool)
        events = [
            e async for e in app.run(request(tool_completion={"tool_names": ["ask_customer"]}))
        ]
        assert (await store.load("s")).status == SessionStatus.COMPLETED
        assert tool.calls == 2
        assert len(provider.requests) == 2
        assert "tool_completion" not in events[-1].payload
        transcript = await store.load_transcript("s")
        assert len(transcript[2].content) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("enabled_on_resume", [False, True])
def test_a_fresh_resumed_turn_requires_its_own_policy(enabled_on_resume):
    async def scenario():
        store = InMemorySessionStore()
        provider = _TwoCallProvider(
            [
                call(),
                [
                    ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="c2"),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        tool = FinalTool()
        app = app_for(store, provider, tool)
        [e async for e in app.run(request(tool_completion={"tool_names": ["ask_customer"]}))]
        if not enabled_on_resume:
            from cayu import ExecutionProfileMismatchError

            with pytest.raises(ExecutionProfileMismatchError):
                [
                    e
                    async for e in app.resume(
                        ResumeRequest(
                            session_id="s",
                            messages=[Message.text("user", "The second order")],
                        )
                    )
                ]
            assert len(provider.requests) == 1
            assert tool.calls == 1
            return
        events = [
            e
            async for e in app.resume(
                ResumeRequest(
                    session_id="s",
                    messages=[Message.text("user", "The second order")],
                    tool_completion={"tool_names": ["ask_customer"]} if enabled_on_resume else None,
                )
            )
        ]
        assert (await store.load("s")).status == SessionStatus.COMPLETED, [
            (e.type, e.payload) for e in events
        ]
        assert tool.calls == 2
        assert len(provider.requests) == (2 if enabled_on_resume else 3)
        completed = next(e for e in events if e.type == EventType.SESSION_COMPLETED)
        assert ("tool_completion" in completed.payload) == enabled_on_resume

    asyncio.run(scenario())


def test_profile_adoption_can_disable_completion_for_a_fresh_turn():
    from tests.core.test_execution_profiles import RecordingExecutionProfilePolicy

    from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
    from cayu.runtime.execution_profiles import (
        ExecutionProfileAdoptionIntent,
        ExecutionProfileAuthorityDecision,
        ExecutionProfilePolicyAction,
        ExecutionProfilePolicyResult,
    )

    async def scenario():
        store = InMemorySessionStore()
        policy = RecordingExecutionProfilePolicy(
            ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="Reviewed finalization change.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )
        )
        app = CayuApp(session_store=store, execution_profile_policy=policy, enable_logging=False)
        provider = _TwoCallProvider(
            [call(), call(), [ModelStreamEvent.completed({"finish_reason": "stop"})]]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="support", model="scripted-model"), tools=[FinalTool()])
        [e async for e in app.run(request(tool_completion={"tool_names": ["ask_customer"]}))]
        events = [
            e
            async for e in app.resume(
                ResumeRequest(
                    session_id="s",
                    messages=[Message.text("user", "Another order")],
                    tool_completion=None,
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key="ordinary-completion",
                        reason="Use model completion on subsequent turns.",
                        requested_by=ResolutionActor(
                            subject="maintainer", source=ResolutionActorSource.REQUEST
                        ),
                    ),
                )
            )
        ]
        assert (await store.load("s")).status == SessionStatus.COMPLETED
        assert len(provider.requests) == 3
        assert events[-1].type == EventType.SESSION_COMPLETED
        assert "tool_completion" not in events[-1].payload

    asyncio.run(scenario())


def test_require_final_tool_reminds_then_success_completes_immediately():
    async def scenario():
        store = InMemorySessionStore()
        provider = _TwoCallProvider(
            [[ModelStreamEvent.completed({"finish_reason": "stop"})], call()]
        )
        app = app_for(store, provider, FinalTool())
        events = [
            e
            async for e in app.run(
                request(
                    tool_completion={"tool_names": ["ask_customer"]},
                    loop_policies=(RequireFinalTool(["ask_customer"]),),
                )
            )
        ]
        assert (await store.load("s")).status == SessionStatus.COMPLETED
        assert len(provider.requests) == 2
        assert [
            e.payload["action"] for e in events if e.type == "custom.loop.before_stop.selected"
        ] == ["continue"]

    asyncio.run(scenario())


@pytest.mark.parametrize("approve", [False, True])
def test_approval_continuation_retains_policy_and_denial_cannot_complete(store_factory, approve):
    from tests.core.test_require_final_tool import _ApprovalPolicy

    from cayu import ToolApprovalDecision, ToolApprovalRequest

    session_id = f"approval-{approve}"

    async def scenario():
        async with store_factory(CrashStore, boundary="none") as store:
            provider = _TwoCallProvider([call()])
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="support", model="scripted-model"),
                tools=[FinalTool()],
                tool_policy=_ApprovalPolicy(),
            )
            events = [
                e
                async for e in app.run(
                    request(session_id=session_id, tool_completion={"tool_names": ["ask_customer"]})
                )
            ]
            assert (await store.load(session_id)).status == SessionStatus.INTERRUPTED
            approval = next(e for e in events if e.type == EventType.TOOL_CALL_APPROVAL_REQUESTED)
            fresh_provider = _TwoCallProvider(
                [] if approve else [[ModelStreamEvent.completed({"finish_reason": "stop"})]]
            )
            tool = FinalTool()
            fresh = CayuApp(session_store=store, enable_logging=False)
            fresh.register_provider(fresh_provider, default=True)
            fresh.register_agent(
                AgentSpec(name="support", model="scripted-model"),
                tools=[tool],
                tool_policy=_ApprovalPolicy(),
            )
            resolved = [
                e
                async for e in fresh.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.payload["approval"]["approval_id"],
                        tool_call_id=approval.payload["tool_call_id"],
                        tool_round_id=approval.payload["tool_round_id"],
                        decision=ToolApprovalDecision.APPROVE
                        if approve
                        else ToolApprovalDecision.DENY,
                    )
                )
            ]
            assert (await store.load(session_id)).status == SessionStatus.COMPLETED, [
                (e.type, e.payload) for e in resolved
            ]
            assert tool.calls == (1 if approve else 0)
            assert len(fresh_provider.requests) == (0 if approve else 1)
            completed = next(e for e in resolved if e.type == EventType.SESSION_COMPLETED)
            assert ("tool_completion" in completed.payload) == approve

    asyncio.run(scenario())


def test_run_outcome_exposes_a_detached_tool_result():
    from cayu import run_to_completion

    async def scenario():
        store = InMemorySessionStore()
        provider = _TwoCallProvider([call()])
        outcome = await run_to_completion(
            app_for(store, provider, FinalTool()),
            request(tool_completion={"tool_names": ["ask_customer"]}),
        )
        assert outcome.ok
        assert outcome.final_text == ""
        assert outcome.tool_completion.call.tool_name == "ask_customer"
        assert outcome.tool_completion.result.structured == {"question": "Which order?"}
        outcome.tool_completion.result.structured["question"] = "changed"
        completion = next(e for e in outcome.events if e.type == EventType.SESSION_COMPLETED)
        assert completion.payload["tool_completion"]["result"]["structured"] == {
            "question": "Which order?"
        }
        transcript = await store.load_transcript("s")
        assert transcript[2].content[0].structured == {"question": "Which order?"}

    asyncio.run(scenario())


@pytest.mark.parametrize("blocked", [False, True])
def test_malformed_and_policy_blocked_calls_cannot_complete(blocked):
    from cayu import ToolPolicy, ToolPolicyDecision, ToolPolicyResult

    class Deny(ToolPolicy):
        async def authorize(self, request):
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="denied")

    async def scenario():
        store = InMemorySessionStore()
        provider = _TwoCallProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="c1")
                    if blocked
                    else ModelStreamEvent(
                        type="tool_call",
                        payload={
                            "name": "ask_customer",
                            "arguments": "{broken",
                            "id": "c1",
                        },
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        tool = FinalTool()
        app = app_for(store, provider, tool, tool_policy=Deny() if blocked else None)
        events = [
            e async for e in app.run(request(tool_completion={"tool_names": ["ask_customer"]}))
        ]
        assert (await store.load("s")).status == (
            SessionStatus.COMPLETED if blocked else SessionStatus.FAILED
        )
        assert tool.calls == 0
        assert len(provider.requests) == (2 if blocked else 1)
        assert not any(e.payload.get("tool_completion") for e in events)

    asyncio.run(scenario())


def test_completion_does_not_grant_exposure_or_capabilities():
    from cayu import StaticToolExposurePolicy

    async def scenario():
        store = InMemorySessionStore()
        provider = _TwoCallProvider(
            [call(), [ModelStreamEvent.completed({"finish_reason": "stop"})]]
        )
        tool = FinalTool()
        app = app_for(
            store,
            provider,
            tool,
            tool_exposure_policy=StaticToolExposurePolicy(profile_id="hidden", tools=()),
        )
        events = [
            e async for e in app.run(request(tool_completion={"tool_names": ["ask_customer"]}))
        ]
        assert (await store.load("s")).status == SessionStatus.COMPLETED
        assert tool.calls == 0
        assert len(provider.requests) == 2
        assert (
            "tool_completion"
            not in next(e for e in events if e.type == EventType.SESSION_COMPLETED).payload
        )

    asyncio.run(scenario())


def test_structured_output_combination_is_rejected_before_provider_dispatch():
    from cayu import StructuredOutputSpec

    async def scenario():
        provider = _TwoCallProvider([])
        app = app_for(InMemorySessionStore(), provider, FinalTool())
        with pytest.raises(ValueError, match="cannot be combined"):
            [
                e
                async for e in app.run(
                    request(
                        tool_completion={"tool_names": ["ask_customer"]},
                        structured_output=StructuredOutputSpec(json_schema={"type": "object"}),
                    )
                )
            ]
        assert provider.requests == []

    asyncio.run(scenario())


def test_secret_names_are_rejected_before_provider_dispatch():
    from cayu import SecretRedactor

    async def scenario():
        provider = _TwoCallProvider([])
        app = CayuApp(
            session_store=InMemorySessionStore(),
            enable_logging=False,
            secret_redactor=SecretRedactor("ask_customer"),
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="support", model="scripted-model"), tools=[FinalTool()])
        with pytest.raises(ValueError):
            [e async for e in app.run(request(tool_completion={"tool_names": ["ask_customer"]}))]
        assert provider.requests == []

    asyncio.run(scenario())


def test_completion_uses_hook_modified_and_secret_projected_result():
    from cayu import AfterToolCallDecision, RuntimeHook, SecretRedactor

    class Modify(RuntimeHook):
        async def after_tool_call(self, context):
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(
                    content="question private-canary",
                    structured={"question": "question private-canary"},
                ),
            )

    async def scenario():
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            secret_redactor=SecretRedactor("private-canary"),
            runtime_hooks=[Modify()],
        )
        provider = _TwoCallProvider([call()])
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="support", model="scripted-model"), tools=[FinalTool()])
        events = [
            e async for e in app.run(request(tool_completion={"tool_names": ["ask_customer"]}))
        ]
        assert (await store.load("s")).status == SessionStatus.COMPLETED
        completion = next(e for e in events if e.type == EventType.SESSION_COMPLETED)
        basis = completion.payload["tool_completion"]["result"]
        assert "private-canary" not in str(basis)
        assert basis["content"].startswith("question ")
        assert basis["content"] == (await store.load_transcript("s"))[2].content[0].content
        assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_retry_retains_policy_and_success_has_no_followup_request():
    from cayu import RetryPolicy
    from cayu.providers.base import ModelProviderError

    class RetryProvider(_TwoCallProvider):
        async def stream(self, request):
            if not self.requests:
                self.requests.append(request)
                raise ModelProviderError("HTTP 503", provider=self.name, retryable=True)
            async for event in super().stream(request):
                yield event

    async def scenario():
        store = InMemorySessionStore()
        provider = RetryProvider([[], call()])
        tool = FinalTool()
        app = app_for(store, provider, tool)
        events = [
            e
            async for e in app.run(
                request(
                    tool_completion={"tool_names": ["ask_customer"]},
                    retry_policy=RetryPolicy(initial_delay_s=0.0, max_delay_s=0.0, jitter_s=0.0),
                )
            )
        ]
        assert (await store.load("s")).status == SessionStatus.COMPLETED, [
            (e.type, e.payload) for e in events
        ]
        assert len(provider.requests) == 2
        assert tool.calls == 1

    asyncio.run(scenario())


def test_unchecked_configuration_is_revalidated_and_detached():
    from cayu.sessions.base import copy_run_request

    malformed = ToolCompletionPolicy.model_construct(tool_names=("ask_customer", "ask_customer"))
    with pytest.raises(ValueError):
        copy_run_request(request().model_copy(update={"tool_completion": malformed}))
    names = ["ask_customer"]
    invocation = request(tool_completion={"tool_names": names})
    names.append("later")
    assert invocation.tool_completion.tool_names == ("ask_customer",)
    with pytest.raises(ValueError):
        invocation.tool_completion.tool_names = ("changed",)


def test_terminal_and_transcript_result_mismatch_fails_closed():
    class WrongResult(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def query_events(self, query):
            records = await super().query_events(query)
            if query.event_types == (
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
                EventType.TOOL_CALL_APPROVAL_DENIED,
            ):
                for record in records:
                    if record.event.type == EventType.TOOL_CALL_COMPLETED:
                        record.event.payload["result"]["content"] = "forged result"
            return records

    async def scenario():
        store = WrongResult()
        provider = _TwoCallProvider([call()])
        tool = FinalTool()
        app = app_for(store, provider, tool)
        events = [
            e async for e in app.run(request(tool_completion={"tool_names": ["ask_customer"]}))
        ]
        assert (await store.load("s")).status == SessionStatus.FAILED
        assert not [e for e in events if e.type == EventType.SESSION_COMPLETED]
        assert tool.calls == 1
        assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_disabled_policy_keeps_existing_serialization_and_finalization_material():
    from cayu import RetryPolicy, RunLimits
    from cayu.runtime._execution_profile_admission import model_finalization_material

    assert "tool_completion" not in request().model_dump(mode="json")
    assert "tool_completion" not in ResumeRequest(
        session_id="s", messages=[Message.text("user", "Help")]
    ).model_dump(mode="json")
    material = model_finalization_material(
        max_steps=8, limits=RunLimits(), retry_policy=RetryPolicy()
    )
    assert material == {
        "kind": "cayu:model-finalization:v2",
        "max_steps": 8,
        "limits": RunLimits().model_dump(mode="json"),
        "retry_policy": RetryPolicy().model_dump(mode="json"),
    }


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("checkpoint_state", ["not_loaded", "absent", "snapshot"])
def test_recorded_policy_lookup_loads_only_required_evidence(configured, checkpoint_state):
    from scripts.benchmark_tool_round import observe_store

    from cayu import RetryPolicy
    from cayu.runtime._tool_completion import load_recorded_tool_completion_policy
    from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint

    class Store(CrashStore, InMemorySessionStore):
        invocation_lifecycle_command_version = 1

    async def scenario():
        store = Store(boundary="tool-publication")
        policy = ToolCompletionPolicy(tool_names=("ask_customer",)) if configured else None
        options = {"tool_completion": policy} if configured else {}
        invocation = request(retry_policy=RetryPolicy(), **options)
        app = app_for(store, _TwoCallProvider([call()]), FinalTool())
        with pytest.raises(_SimulatedProcessLoss):
            [event async for event in app.run(invocation)]
        session = await store.load("s")
        checkpoint = await store.load_checkpoint("s")
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active is not None
        checkpoint_argument = (
            []
            if checkpoint_state == "not_loaded"
            else [None if checkpoint_state == "absent" else checkpoint]
        )
        with observe_store(store) as (calls, _admissions):
            recorded = await load_recorded_tool_completion_policy(
                store,
                session,
                *checkpoint_argument,
                execution_profile=active.profile,
                max_steps=invocation.max_steps,
                limits=invocation.limits,
                retry_policy=invocation.retry_policy,
            )
        if not configured or checkpoint_state == "absent":
            assert recorded is None
            assert not calls
        else:
            assert recorded == policy
            assert calls["load_checkpoint"] == (checkpoint_state == "not_loaded")

    asyncio.run(scenario())


def test_unconfigured_approval_keeps_checkpoint_read_budget():
    from scripts.benchmark_tool_round import observe_store
    from tests.core.test_tool_round_continuation_backends import _pause, _resolve, _runtime

    async def scenario():
        store = InMemorySessionStore()
        app, provider, tool = _runtime(store, "approve", include_denied=False)
        approval = await _pause(app, "approve", "s")
        with observe_store(store) as (calls, _admissions):
            events = [event async for event in _resolve(app, approval)]
        assert events[-1].type == EventType.SESSION_COMPLETED
        assert tool.calls == [{"value": "first"}, {"value": "second"}]
        assert len(provider.requests) == 2
        # The native continuation now also reads its external-interaction owner
        # in _run_recovered_session. Disabled tool completion still adds none.
        assert calls["load_checkpoint"] <= 32

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["initial", "continuation"])
@pytest.mark.parametrize("policy_state", ["implicit", "none", "configured"])
def test_portable_work_attempt_source_preserves_completion_controls(kind, policy_state):
    from cayu.sessions.base import copy_resume_request, copy_run_request

    engine = CayuApp(enable_logging=False)._session_engine
    model = RunRequest if kind == "initial" else ResumeRequest
    fields = {"agent_name": "support"} if kind == "initial" else {"session_id": "s"}
    policy = ToolCompletionPolicy(tool_names=("ask_customer",))
    if policy_state != "implicit":
        fields["tool_completion"] = policy if policy_state == "configured" else None
    invocation = model(messages=[Message.text("user", "Help")], **fields)
    copied = copy_run_request(invocation) if kind == "initial" else copy_resume_request(invocation)
    assert ("tool_completion" in copied.model_fields_set) is (policy_state != "implicit")
    digest = engine.work_attempt_source_request_sha256(copied, kind=kind)
    snapshot = engine.work_attempt_source_snapshot(copied, kind=kind, source_request_sha256=digest)
    assert snapshot is not None
    restored = model.model_validate(snapshot.request)
    object.__setattr__(restored, "__pydantic_fields_set__", set(snapshot.fields_set))
    assert engine.work_attempt_source_request_sha256(restored, kind=kind) == digest
    assert restored.tool_completion == (policy if policy_state == "configured" else None)


@pytest.mark.parametrize("worker", [False, True])
def test_final_tool_uses_normal_task_completion_owner(worker):
    from cayu import InMemoryTaskStore, TaskCreate, TaskStatus

    async def scenario():
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        await tasks.create_task(TaskCreate(task_id="job", type="support"))
        claim = await tasks.claim_task("worker") if worker else None
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        provider = _TwoCallProvider([call()])
        tool = FinalTool()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="support", model="scripted-model"), tools=[tool])
        events = [
            e
            async for e in app.run(
                request(
                    tool_completion={"tool_names": ["ask_customer"]},
                    task_id="job",
                    task_worker_id="worker" if worker else None,
                    task_lease_expires_at=claim.lease_expires_at if worker else None,
                )
            )
        ]
        assert (await sessions.load("s")).status == SessionStatus.COMPLETED
        assert (await tasks.load_task("job")).status == TaskStatus.COMPLETED
        assert len(provider.requests) == 1
        assert tool.calls == 1
        assert len([e for e in events if e.type == EventType.TASK_COMPLETED]) == 1

    asyncio.run(scenario())


def test_admitted_work_attempt_retains_policy_and_preserves_task_verification():
    from tests.core.test_verified_work_contracts import _contract

    from cayu import InMemoryTaskStore, TaskCreate, TaskStatus
    from cayu.tasks.admission import WorkAttemptExecutionRequest, WorkAttemptRunRequest

    async def scenario():
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        contract = _contract(contract_id="completion-contract")
        await tasks.publish_work_contract(contract)
        await tasks.create_task(
            TaskCreate(task_id="job", type="support", work_contract=contract.reference())
        )
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        provider = _TwoCallProvider([call()])
        tool = FinalTool()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="support", model="scripted-model"), tools=[tool])
        admission = await app.admit_work_attempt(
            request(task_id="job", tool_completion={"tool_names": ["ask_customer"]}),
            execution=WorkAttemptExecutionRequest(
                admission_id="admission",
                claim_id="claim",
                attempt_id="attempt",
                interaction_id="interaction",
                worker_id="worker",
                generation=1,
                lease_seconds=300,
            ),
        )
        assert admission.run_semantics.tool_completion.tool_names == ("ask_customer",)
        events = [
            e
            async for e in app._execute_work_attempt(
                WorkAttemptRunRequest(
                    admission_id=admission.admission_id,
                    claim_id=admission.claim.claim_id,
                    worker_id=admission.claim.worker_id,
                    generation=admission.claim.generation,
                    lease_seconds=300,
                )
            )
        ]
        assert (await sessions.load("s")).status == SessionStatus.COMPLETED
        assert (await tasks.load_task("job")).status == TaskStatus.RUNNING
        assert not any(e.type == EventType.TASK_COMPLETED for e in events)
        assert len(provider.requests) == 1
        assert tool.calls == 1
        assert events[-1].payload["reason"] == "host_rendered_tool"

    asyncio.run(scenario())


def test_queued_turn_keeps_its_admitted_policy_and_completes_from_its_own_tool():
    from cayu import EnqueueSessionMessageRequest

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class BlockingFinalTool(FinalTool):
            async def run(self, ctx, args):
                if self.calls == 0:
                    entered.set()
                    await release.wait()
                return await super().run(ctx, args)

        store = InMemorySessionStore()
        second = [
            ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="c2"),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
        provider = _TwoCallProvider([call(), second])
        tool = BlockingFinalTool()
        app = app_for(store, provider, tool)

        async def collect():
            return [
                e async for e in app.run(request(tool_completion={"tool_names": ["ask_customer"]}))
            ]

        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(entered.wait(), 10)
            await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id="s",
                    idempotency_key="next",
                    content="Help with another order",
                    delivery_mode="on_idle",
                )
            )
            release.set()
            events = await asyncio.wait_for(task, 15)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert (await store.load("s")).status == SessionStatus.COMPLETED, [
            (e.type, e.payload) for e in events
        ]
        assert tool.calls == 2
        assert len(provider.requests) == 2
        completions = [e for e in events if e.type == EventType.INTERACTION_COMPLETED]
        assert len(completions) == 2
        assert [e.payload["tool_completion"]["call"]["tool_call_id"] for e in completions] == [
            "c1",
            "c2",
        ]
        assert events[-1].payload["tool_completion"]["call"]["tool_call_id"] == "c2"

    asyncio.run(scenario())
