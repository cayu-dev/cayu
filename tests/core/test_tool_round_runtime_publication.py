from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

import pytest
from tests.core._event_projection_support import private_events_for_public_events
from tests.core._execution_profile_fixtures import (
    rebind_test_invocation,
)
from tests.core._workload_secret_support import (
    FakeProvider,
    RequireApprovalPolicy,
    SideEffectTool,
    collect_events,
    collect_resume_events,
)

from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    Message,
    ToolResultPart,
)
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
    Session,
    SessionIdentity,
    SessionStatus,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._model_completion_publication import (
    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    model_step_publication_from_checkpoint,
)
from cayu.runtime._tool_round_executor import InterruptedToolRoundRequest
from cayu.runtime.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
)
from cayu.runtime.context import ContextPolicy, ContextRequest, validate_context_messages
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _FailingTerminalToolEventStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.failed_terminal_once = False

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        if not self.failed_terminal_once and any(
            event.type == EventType.TOOL_CALL_COMPLETED for event in events
        ):
            self.failed_terminal_once = True
            raise RuntimeError("terminal tool event unavailable")
        await super().append_events(session_id, events)


class _EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=args["text"],
            structured={"agent": ctx.agent_name, "echoed": args["text"]},
        )


def _tool_round_identity() -> ToolRoundIdentity:
    return ToolRoundIdentity(
        model_step_id=f"mstep_{'1' * 32}",
        model_attempt_id=f"matt_{'2' * 32}",
        tool_round_id=f"tround_{'3' * 32}",
    )


async def _collect_interrupt_events(
    app: CayuApp,
    request: InterruptSessionRequest,
) -> list[Event]:
    return [event async for event in app.interrupt_session(request)]


def _assert_only_model_step_publication_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> None:
    assert checkpoint is not None
    assert set(checkpoint) == {
        ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
        CHECKPOINT_SCHEMA_VERSION_KEY,
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    }
    assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
    assert active_invocation_execution_profile_from_checkpoint(checkpoint) is not None
    assert model_step_publication_from_checkpoint(checkpoint) is not None


def _crashed_tool_round_app(
    session_id: str,
    store: _FailingTerminalToolEventStore | None = None,
) -> tuple[CayuApp, _FailingTerminalToolEventStore, SideEffectTool, dict[str, Any]]:
    store = store if store is not None else _FailingTerminalToolEventStore()
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    initial_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    assert initial_events[-1].type == EventType.SESSION_FAILED
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None and "pending_tool_round" in checkpoint
    assert tool.calls == [{}]
    return app, store, tool, checkpoint


def test_interrupt_close_rejects_missing_marker_before_terminal_publication() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupted_round_missing_marker",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        session = await store.update_status(
            "sess_interrupted_round_missing_marker",
            SessionStatus.RUNNING,
        )
        request = InterruptedToolRoundRequest(
            session=session,
            registered_agent=app._get_registered_agent("assistant"),
            registered_environment=None,
            messages=[],
            tool_calls=[
                runtime_records.ToolCallRequest(
                    id="call_missing_marker",
                    name="missing",
                    arguments={},
                )
            ],
            tool_outcomes=[],
            tool_round_identity=_tool_round_identity(),
            cancellation_artifacts=None,
            cancellation_artifacts_by_id=None,
        )

        with pytest.raises(RuntimeError, match="lost its durable pending marker"):
            async for _event in app._recovery_coordinator.close_interrupted_tool_round(request):
                pass
        assert not [
            event
            for event in await store.load_events(session.id)
            if event.type
            in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
            }
        ]

    asyncio.run(scenario())


def test_pending_tool_round_recovery_replays_commit_after_acknowledgement_loss() -> None:
    class CommitThenRaiseStore(_FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed_publication_ack = False

        async def publish_runtime_publication(
            self,
            session_id: str,
            *,
            request,
            expected_statuses=None,
            expected_run_epoch=None,
            expected_transcript_cursor=None,
        ):
            result = await super().publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses=expected_statuses,
                expected_run_epoch=expected_run_epoch,
                expected_transcript_cursor=expected_transcript_cursor,
            )
            if request.kind == "tool-round" and not self.failed_publication_ack:
                self.failed_publication_ack = True
                raise RuntimeError("tool-round publication acknowledgement lost")
            return result

    session_id = "sess_tool_round_recovery_publication_ack_lost"
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id,
        store=CommitThenRaiseStore(),
    )
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]

    resumed_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert store.failed_publication_ack is True
    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == [{}]
    _assert_only_model_step_publication_checkpoint(asyncio.run(store.load_checkpoint(session_id)))
    assert (
        asyncio.run(
            store.load_runtime_publication_receipt(
                session_id,
                f"tool-round:{round_id}",
            )
        )
        is not None
    )
    transcript = asyncio.run(store.load_transcript(session_id))
    assert sum(message.role == "tool" for message in transcript) == 1
    recovered_terminals = [
        event
        for event in asyncio.run(store.load_events(session_id))
        if event.payload.get("tool_round_id") == round_id
        and event.type
        in {
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
        }
    ]
    assert len(recovered_terminals) == 1


def test_pending_tool_round_recovery_does_not_retry_precommit_rejection() -> None:
    class RejectFirstStore(_FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.publication_attempts = 0

        async def publish_runtime_publication(
            self,
            session_id: str,
            *,
            request,
            expected_statuses=None,
            expected_run_epoch=None,
            expected_transcript_cursor=None,
        ):
            if request.kind == "tool-round":
                self.publication_attempts += 1
                if self.publication_attempts == 1:
                    raise RuntimeError("tool-round publication rejected before commit")
            return await super().publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses=expected_statuses,
                expected_run_epoch=expected_run_epoch,
                expected_transcript_cursor=expected_transcript_cursor,
            )

    session_id = "sess_tool_round_recovery_precommit_rejected"
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id,
        store=RejectFirstStore(),
    )
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]

    first_resume = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
    )

    assert first_resume[-1].type == EventType.SESSION_FAILED
    assert store.publication_attempts == 1
    assert (
        asyncio.run(
            store.load_runtime_publication_receipt(
                session_id,
                f"tool-round:{round_id}",
            )
        )
        is None
    )
    first_checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert first_checkpoint is not None
    assert first_checkpoint["pending_tool_round"]["tool_round_id"] == round_id
    assert Message.text("user", "continue") not in asyncio.run(store.load_transcript(session_id))
    deferred = asyncio.run(store.load_deferred_interaction_input(session_id))
    assert deferred is not None
    assert deferred.source_messages == [Message.text("user", "continue")]

    second_resume = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "retry recovery")],
            ),
        )
    )
    assert second_resume[-1].type == EventType.SESSION_COMPLETED
    assert store.publication_attempts == 2
    assert tool.calls == [{}]
    transcript = asyncio.run(store.load_transcript(session_id))
    assert sum(message.role == "tool" for message in transcript) == 1
    tool_result_index = next(
        index for index, message in enumerate(transcript) if message.role == "tool"
    )
    assert transcript[tool_result_index + 1 : tool_result_index + 3] == [
        Message.text("user", "continue"),
        Message.text("user", "retry recovery"),
    ]


def test_pending_tool_round_recovery_preserves_cancellation_after_exact_replay() -> None:
    class BlockingStore(_FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.publication_committed = asyncio.Event()
            self.allow_publication_return = asyncio.Event()
            self.blocked_publication_once = False

        async def publish_runtime_publication(
            self,
            session_id: str,
            *,
            request,
            expected_statuses=None,
            expected_run_epoch=None,
            expected_transcript_cursor=None,
        ):
            result = await super().publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses=expected_statuses,
                expected_run_epoch=expected_run_epoch,
                expected_transcript_cursor=expected_transcript_cursor,
            )
            if request.kind == "tool-round" and not self.blocked_publication_once:
                self.blocked_publication_once = True
                self.publication_committed.set()
                await self.allow_publication_return.wait()
            return result

    session_id = "sess_tool_round_recovery_cancelled_after_commit"
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id,
        store=BlockingStore(),
    )
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]

    async def scenario() -> None:
        recovery_task = asyncio.create_task(
            collect_resume_events(
                app,
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                ),
            )
        )
        await asyncio.wait_for(store.publication_committed.wait(), timeout=5)
        recovery_task.cancel("cancel after tool-round publication commit")
        assert recovery_task.cancelling() == 1
        store.allow_publication_return.set()
        with pytest.raises(asyncio.CancelledError) as cancellation:
            await asyncio.wait_for(recovery_task, timeout=5)
        assert cancellation.value.args == ("cancel after tool-round publication commit",)
        assert recovery_task.cancelled()

        _assert_only_model_step_publication_checkpoint(await store.load_checkpoint(session_id))
        assert (
            await store.load_runtime_publication_receipt(
                session_id,
                f"tool-round:{round_id}",
            )
            is not None
        )
        transcript = await store.load_transcript(session_id)
        assert sum(message.role == "tool" for message in transcript) == 1
        assert transcript[-1] == Message.text("user", "continue")
        assert transcript.count(Message.text("user", "continue")) == 1
        assert await store.load_deferred_interaction_input(session_id) is None
        assert tool.calls == [{}]

    asyncio.run(scenario())


def test_pending_tool_round_materialization_reapplies_secret_redaction() -> None:
    class RecordingContextPolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:tool-round-materialization-context-policy",
                behavior_version="1",
                implementation_version="1",
            )

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            return request.messages

    async def scenario() -> None:
        session_id = "sess_tool_round_materialization_redaction"
        secret = "dynamic-recovery-secret-canary"
        store = _FailingTerminalToolEventStore()
        initial_provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="side_effect",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        context_policy = RecordingContextPolicy()
        initial_app = CayuApp(session_store=store, enable_logging=False)
        initial_app.register_provider(initial_provider, default=True)
        initial_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[SideEffectTool()],
            context_policy=context_policy,
        )
        initial_events = await collect_events(
            initial_app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", secret)],
            ),
        )
        assert initial_events[-1].type is EventType.SESSION_FAILED
        context_policy.requests.clear()

        resumed_provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("recovered"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        resumed_app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        resumed_app.register_provider(resumed_provider, default=True)
        resumed_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[SideEffectTool()],
            context_policy=context_policy,
        )
        resumed_events = await collect_resume_events(
            resumed_app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", f"continue with {secret}")],
            ),
        )

        assert resumed_events[-1].type is EventType.SESSION_COMPLETED
        assert len(context_policy.requests) == 1
        context_payload = repr(
            [message.model_dump(mode="json") for message in context_policy.requests[0].messages]
        )
        provider_payload = repr(
            [message.model_dump(mode="json") for message in resumed_provider.requests[0].messages]
        )
        assert secret not in context_payload
        assert REDACTED_SECRET in context_payload
        assert secret not in provider_payload
        assert REDACTED_SECRET in provider_payload
        durable_payload = repr(
            [message.model_dump(mode="json") for message in await store.load_transcript(session_id)]
        )
        assert secret in durable_payload

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "corruption",
    [
        pytest.param("nested-id", id="nested-call-id"),
        pytest.param("nested-name", id="nested-tool-name"),
        pytest.param("nested-arguments", id="nested-arguments"),
        pytest.param("target-arguments", id="top-level-target"),
    ],
)
def test_model_boundary_rejects_corrupted_pending_tool_approval(
    corruption: str,
) -> None:
    async def run() -> None:
        session_id = f"sess_corrupt_pending_approval_{corruption}"
        store = InMemorySessionStore()
        tool = SideEffectTool()
        provider = FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "requires approval"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireApprovalPolicy(),
        )
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use tool")],
            ),
        )

        def corrupt_checkpoint(
            _session: Session,
            current: dict[str, Any] | None,
        ) -> dict[str, Any]:
            updated = deepcopy({} if current is None else current)
            approval = updated["pending_tool_approval"]
            if corruption == "nested-id":
                approval["tool_calls"][0]["tool_call_id"] = "call_other"
            elif corruption == "nested-name":
                approval["tool_calls"][0]["tool_name"] = "other_tool"
            elif corruption == "nested-arguments":
                approval["tool_calls"][0]["arguments"] = {"value": "other"}
            elif corruption == "target-arguments":
                approval["arguments"] = {"value": "other"}
            else:  # pragma: no cover - parametrization is exhaustive
                raise AssertionError(f"Unknown corruption: {corruption}")
            return updated

        await store.transform_checkpoint(session_id, corrupt_checkpoint)
        await rebind_test_invocation(store, session_id)
        checkpoint_before = await store.load_checkpoint(session_id)
        transcript_before = await store.load_transcript(session_id)
        events_before = await store.load_events(session_id)
        assert checkpoint_before is not None
        logical_step_id = checkpoint_before["last_model_step_publication"]["logical_step_id"]
        receipt_before = await store.load_runtime_publication_receipt(
            session_id,
            logical_step_id,
        )
        with pytest.raises(
            ValueError,
            match="Pending tool approval checkpoint is invalid and cannot be executed",
        ):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )

        assert await store.load_checkpoint(session_id) == checkpoint_before
        assert await store.load_transcript(session_id) == transcript_before
        assert await store.load_events(session_id) == events_before
        assert (
            await store.load_runtime_publication_receipt(
                session_id,
                logical_step_id,
            )
            == receipt_before
        )
        assert len(provider.requests) == 1
        assert tool.calls == []

    asyncio.run(run())


def test_cross_worker_interrupt_racing_tool_round_publication_finishes_interrupted() -> None:
    class BlockingStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.publication_started: asyncio.Event | None = None
            self.allow_publication: asyncio.Event | None = None
            self.blocked_publication = False

        async def publish_runtime_publication(
            self,
            session_id: str,
            *,
            request,
            expected_statuses=None,
            expected_run_epoch=None,
            expected_transcript_cursor=None,
        ):
            if request.kind == "tool-round" and not self.blocked_publication:
                if self.publication_started is None or self.allow_publication is None:
                    raise AssertionError("Tool-round publication barriers were not initialized.")
                self.blocked_publication = True
                self.publication_started.set()
                await self.allow_publication.wait()
            return await super().publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses=expected_statuses,
                expected_run_epoch=expected_run_epoch,
                expected_transcript_cursor=expected_transcript_cursor,
            )

    session_id = "sess_cross_worker_interrupt_tool_round_publication"
    store = BlockingStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_echo",
                name="echo",
                arguments={"text": "finished"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    worker_app = CayuApp(session_store=store)
    api_app = CayuApp(session_store=store)
    for app in (worker_app, api_app):
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_EchoTool()],
        )

    async def run():
        store.publication_started = asyncio.Event()
        store.allow_publication = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                worker_app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await asyncio.wait_for(store.publication_started.wait(), timeout=5)
        interrupt_task = asyncio.create_task(
            _collect_interrupt_events(
                api_app,
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="operator stop from another worker",
                ),
            )
        )

        async def wait_until_interrupting() -> None:
            while True:
                durable_session = await store.load(session_id)
                if (
                    durable_session is not None
                    and durable_session.status == SessionStatus.INTERRUPTING
                ):
                    return
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_until_interrupting(), timeout=5)
        store.allow_publication.set()
        run_events, interrupt_events = await asyncio.wait_for(
            asyncio.gather(run_task, interrupt_task),
            timeout=5,
        )
        assert await worker_app.drain_background_interruptions(timeout_s=1) is True
        assert await api_app.drain_background_interruptions(timeout_s=1) is True
        return (
            run_events,
            interrupt_events,
            await store.load(session_id),
            await store.load_transcript(session_id),
            await store.load_events(session_id),
            await store.load_checkpoint(session_id),
        )

    (
        run_events,
        interrupt_events,
        durable_session,
        transcript,
        stored_events,
        checkpoint,
    ) = asyncio.run(run())

    assert durable_session is not None
    assert durable_session.status == SessionStatus.INTERRUPTED
    assert run_events[-1].type == EventType.SESSION_INTERRUPTED
    assert interrupt_events == [run_events[-1]]
    assert [event.type for event in stored_events].count(EventType.SESSION_INTERRUPTED) == 1

    terminal_events = [
        event
        for event in stored_events
        if event.payload.get("tool_call_id") == "call_echo"
        and event.type
        in {
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
            EventType.TOOL_CALL_BLOCKED,
            EventType.TOOL_CALL_APPROVAL_DENIED,
        }
    ]
    assert len(terminal_events) == 1
    terminal_event = terminal_events[0]
    assert terminal_event.type == EventType.TOOL_CALL_COMPLETED

    validate_context_messages(transcript)
    tool_messages = [message for message in transcript if message.role == "tool"]
    assert len(tool_messages) == 1
    assert len(tool_messages[0].content) == 1
    tool_result = tool_messages[0].content[0]
    assert isinstance(tool_result, ToolResultPart)
    assert tool_result.tool_call_id == "call_echo"
    assert tool_result.content == terminal_event.payload["result"]["content"] == "finished"
    assert tool_result.structured == terminal_event.payload["result"]["structured"]
    assert tool_result.is_error is terminal_event.payload["result"]["is_error"] is False

    tool_round_id = terminal_event.payload["tool_round_id"]
    assert isinstance(tool_round_id, str)
    assert (
        asyncio.run(
            store.load_runtime_publication_receipt(
                session_id,
                f"tool-round:{tool_round_id}",
            )
        )
        is not None
    )
    assert checkpoint is not None
    assert "pending_tool_round" not in checkpoint
    assert "pending_session_interrupt" not in checkpoint


def test_closing_run_after_first_recovered_tool_event_cannot_strand_interrupting() -> None:
    class BlockingStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.completion_promoted: asyncio.Event | None = None
            self.allow_promotion_return: asyncio.Event | None = None
            self.blocked_promotion = False

        async def promote_model_completion_stage(
            self,
            session_id: str,
            *,
            stage_id: str,
            expected_run_epoch: int,
        ):
            result = await super().promote_model_completion_stage(
                session_id,
                stage_id=stage_id,
                expected_run_epoch=expected_run_epoch,
            )
            if not self.blocked_promotion:
                if self.completion_promoted is None or self.allow_promotion_return is None:
                    raise AssertionError(
                        "Model-completion promotion barriers were not initialized."
                    )
                self.blocked_promotion = True
                self.completion_promoted.set()
                await self.allow_promotion_return.wait()
            return result

    session_id = "sess_close_after_recovered_tool_interrupt"
    store = BlockingStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_echo",
                name="echo",
                arguments={"text": "must not execute"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_EchoTool()],
    )

    async def run():
        store.completion_promoted = asyncio.Event()
        store.allow_promotion_return = asyncio.Event()

        async def consume_through_first_recovered_tool_event():
            stream = app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "use tool")],
                )
            )
            seen_events: list[Event] = []
            try:
                async for event in stream:
                    seen_events.append(event)
                    if event.type == EventType.TOOL_CALL_FAILED:
                        durable_session = await store.load(session_id)
                        checkpoint = await store.load_checkpoint(session_id)
                        stored_events = await store.load_events(session_id)
                        await stream.aclose()
                        return seen_events, durable_session, checkpoint, stored_events
            finally:
                await stream.aclose()
            raise AssertionError("Run stream ended without a recovered tool event.")

        consumer_task = asyncio.create_task(consume_through_first_recovered_tool_event())
        await asyncio.wait_for(store.completion_promoted.wait(), timeout=5)
        interrupt_task = asyncio.create_task(
            _collect_interrupt_events(
                app,
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="operator stop before tool runner creation",
                ),
            )
        )

        async def wait_until_interrupting() -> None:
            while True:
                durable_session = await store.load(session_id)
                if (
                    durable_session is not None
                    and durable_session.status == SessionStatus.INTERRUPTING
                ):
                    return
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_until_interrupting(), timeout=5)
        store.allow_promotion_return.set()
        consumer_result, interrupt_events = await asyncio.wait_for(
            asyncio.gather(consumer_task, interrupt_task),
            timeout=5,
        )
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return (
            consumer_result,
            interrupt_events,
            await store.load(session_id),
            await store.load_checkpoint(session_id),
        )

    (
        (seen_events, session_at_tool_event, checkpoint_at_tool_event, stored_at_tool_event),
        interrupt_events,
        final_session,
        final_checkpoint,
    ) = asyncio.run(run())

    assert seen_events[-1].type == EventType.TOOL_CALL_FAILED
    assert EventType.SESSION_INTERRUPTED not in [event.type for event in seen_events]
    assert session_at_tool_event is not None
    assert session_at_tool_event.status == SessionStatus.INTERRUPTED
    assert checkpoint_at_tool_event is not None
    assert "pending_tool_round" not in checkpoint_at_tool_event
    assert "pending_session_interrupt" not in checkpoint_at_tool_event

    interrupted_events = [
        event for event in stored_at_tool_event if event.type == EventType.SESSION_INTERRUPTED
    ]
    assert len(interrupted_events) == 1
    assert asyncio.run(private_events_for_public_events(store, interrupt_events)) == (
        interrupted_events
    )
    recovered_tool_events = [
        event
        for event in stored_at_tool_event
        if event.type == EventType.TOOL_CALL_FAILED
        and event.payload.get("tool_call_id") == "call_echo"
    ]
    assert len(recovered_tool_events) == 1
    assert recovered_tool_events[0].payload["result"]["content"] == (
        "Tool call interrupted before completion."
    )

    assert final_session is not None
    assert final_session.status == SessionStatus.INTERRUPTED
    assert final_checkpoint is not None
    assert "pending_tool_round" not in final_checkpoint
    assert "pending_session_interrupt" not in final_checkpoint
