from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
    versioned_test_provider_identity,
)

from cayu.artifacts import FileAttachment, FileAttachmentKind
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    FilePart,
    Message,
    MessageRole,
)
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    EnqueueSessionMessageRequest,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
    InMemorySessionStore,
    InMemoryTaskStore,
    InterruptSessionRequest,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RunRequest,
    SessionIdentity,
    SessionMessageDeliveryBatch,
    SessionMessageDeliveryMode,
    SessionStatus,
    TaskCreate,
    TaskHandlerOutcome,
    TaskStatus,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolCapabilityCeiling,
    TranscriptQuery,
    UserInputResponse,
)
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY, public_event_sequence
from cayu.runtime._interruption_coordinator import _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY
from cayu.runtime._invocation_terminal_decision import (
    invocation_terminal_decision_from_checkpoint,
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.runtime.sessions import SESSION_MESSAGE_DELIVERY_BATCH_LIMIT, EventQuery
from cayu.runtime.task_worker import run_task_worker
from cayu.storage import SQLiteSessionStore
from cayu.tools.user_input import UserInputTool
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _ReconstructableQueuedProvider(ModelProvider):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return versioned_test_provider_identity(self)


class _AuthorizeQueuedContinuationPolicy(ExecutionProfilePolicy):
    @property
    def identity(self) -> str:
        return "tests:queued-continuation-authority:v1"

    async def decide(
        self,
        request: ExecutionProfilePolicyRequest,
    ) -> ExecutionProfilePolicyResult:
        assert request.authority_review_required is True
        return ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.ADOPT,
            reason="Authorize the larger model-step limit for queued continuation.",
            authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        )


class BlockingTwoTurnProvider(_ReconstructableQueuedProvider):
    name = "blocking-two-turn"

    def __init__(self) -> None:
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.second_started = asyncio.Event()
        self.release_second = asyncio.Event()
        self.block_second = False
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_started.set()
            await self.release_first.wait()
            text = "first answer"
        else:
            self.second_started.set()
            if self.block_second:
                await self.release_second.wait()
            text = "steered answer"
        yield ModelStreamEvent.text_delta(text)
        yield ModelStreamEvent.completed({})


class RecordingOneShotProvider(_ReconstructableQueuedProvider):
    name = "recording-one-shot"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        yield ModelStreamEvent.text_delta("recovered answer")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class ToolRoundProvider(_ReconstructableQueuedProvider):
    name = "tool-round"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="call-blocking",
                name="blocking_tool",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        text = (
            "finished original response" if len(self.requests) == 2 else "finished after steering"
        )
        yield ModelStreamEvent.text_delta(text)
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class BlockingApprovalProvider(_ReconstructableQueuedProvider):
    name = "blocking-approval"

    def __init__(self) -> None:
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_started.set()
            await self.release_first.wait()
            yield ModelStreamEvent.tool_call(
                id="call-approval",
                name="blocking_tool",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        text = (
            "finished after approved tool" if len(self.requests) == 2 else "finished after steering"
        )
        yield ModelStreamEvent.text_delta(text)
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class BlockingUserInputProvider(_ReconstructableQueuedProvider):
    name = "blocking-user-input"

    def __init__(self) -> None:
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_started.set()
            await self.release_first.wait()
            yield ModelStreamEvent.tool_call(
                id="call-user-input",
                name="ask_user",
                arguments={"question": "Which environment?"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        text = "finished after user input" if len(self.requests) == 2 else "finished after steering"
        yield ModelStreamEvent.text_delta(text)
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class BlockingTool(Tool):
    spec = ToolSpec(
        name="blocking_tool",
        description="Wait until the test releases this tool.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.started.set()
        await self.release.wait()
        return ToolResult(content="tool finished")


class InterruptTrackingStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1
    terminal_interaction_publication_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.interrupting_started = asyncio.Event()

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform=None,
        store_time_checkpoint_transform=None,
        **kwargs,
    ):
        result = await super().transition_status_and_checkpoint(
            session_id,
            from_statuses=from_statuses,
            to_status=to_status,
            checkpoint_transform=checkpoint_transform,
            store_time_checkpoint_transform=store_time_checkpoint_transform,
            **kwargs,
        )
        if to_status is SessionStatus.INTERRUPTING:
            self.interrupting_started.set()
        return result


class CompletionFenceStore(InterruptTrackingStore):
    invocation_lifecycle_command_version = 1
    terminal_interaction_publication_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.completion_started = asyncio.Event()
        self.release_completion = asyncio.Event()

    async def publish_interaction_transition(
        self,
        session_id: str,
        *,
        event: Event,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
        model_completion_stage_settlement=None,
        checkpoint_mutation=None,
        terminal_event: Event | None = None,
        terminal_decision=None,
        expected_session_instance_id: str | None = None,
        expected_active_invocation_profile=None,
        expected_invocation_authority_state="active",
        expected_recovery_claim_id: str | None = None,
    ):
        if only_if_no_queued_messages:
            self.completion_started.set()
            await self.release_completion.wait()
        return await super().publish_interaction_transition(
            session_id,
            event=event,
            from_statuses=from_statuses,
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
            model_completion_stage_settlement=model_completion_stage_settlement,
            checkpoint_mutation=checkpoint_mutation,
            terminal_event=terminal_event,
            terminal_decision=terminal_decision,
            expected_session_instance_id=expected_session_instance_id,
            expected_active_invocation_profile=expected_active_invocation_profile,
            expected_invocation_authority_state=expected_invocation_authority_state,
            expected_recovery_claim_id=expected_recovery_claim_id,
        )


class DispatchAdmissionFenceStore(InterruptTrackingStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.dispatch_admitted = asyncio.Event()
        self.release_dispatch_acknowledgement = asyncio.Event()

    async def mark_model_completion_stage_dispatched(
        self,
        session_id: str,
        *,
        stage,
        consume_child_session_notifications: bool = True,
    ):
        result = await super().mark_model_completion_stage_dispatched(
            session_id,
            stage=stage,
            consume_child_session_notifications=consume_child_session_notifications,
        )
        self.dispatch_admitted.set()
        await self.release_dispatch_acknowledgement.wait()
        return result


class InterruptDecisionTransitionFenceStore(InterruptTrackingStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.interrupt_transition_ready = asyncio.Event()
        self.release_interrupt_transition = asyncio.Event()
        self._block_interrupt_transition = True

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform=None,
        store_time_checkpoint_transform=None,
        **kwargs,
    ):
        if self._block_interrupt_transition and to_status is SessionStatus.INTERRUPTING:
            self._block_interrupt_transition = False
            self.interrupt_transition_ready.set()
            await self.release_interrupt_transition.wait()
        return await super().transition_status_and_checkpoint(
            session_id,
            from_statuses=from_statuses,
            to_status=to_status,
            checkpoint_transform=checkpoint_transform,
            store_time_checkpoint_transform=store_time_checkpoint_transform,
            **kwargs,
        )


class UnsupportedTerminalPublicationStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1
    terminal_interaction_publication_version = None


class UndeclaredTerminalTransitionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.interrupt_transition_called = False

    async def transition_status_and_checkpoint(self, session_id: str, **kwargs):
        if kwargs.get("to_status") is SessionStatus.INTERRUPTING:
            self.interrupt_transition_called = True
        return await super().transition_status_and_checkpoint(session_id, **kwargs)


class DeliveryFenceStore(InterruptTrackingStore):
    invocation_lifecycle_command_version = 1
    terminal_interaction_publication_version = 1

    def __init__(
        self,
        *,
        block_on_call: int,
        lose_closed_terminal_status_acknowledgement: bool = False,
        fail_closed_terminal_status_before_commit: bool = False,
    ) -> None:
        super().__init__()
        self._block_on_call = block_on_call
        self._delivery_calls = 0
        self.delivery_started = asyncio.Event()
        self.release_delivery = asyncio.Event()
        self._lose_closed_terminal_status_acknowledgement = (
            lose_closed_terminal_status_acknowledgement
        )
        self._fail_closed_terminal_status_before_commit = fail_closed_terminal_status_before_commit

    async def transition_status_and_checkpoint(self, session_id: str, **kwargs):
        if (
            self._fail_closed_terminal_status_before_commit
            and kwargs.get("to_status") is SessionStatus.INTERRUPTED
        ):
            self._fail_closed_terminal_status_before_commit = False
            raise ConnectionError("closed terminal status commit failed")
        result = await super().transition_status_and_checkpoint(session_id, **kwargs)
        if (
            self._lose_closed_terminal_status_acknowledgement
            and kwargs.get("to_status") is SessionStatus.INTERRUPTED
        ):
            self._lose_closed_terminal_status_acknowledgement = False
            raise ConnectionError("closed terminal status acknowledgement lost")
        return result

    async def deliver_queued_session_messages(
        self,
        session_id: str,
        *,
        include_on_idle: bool,
        delivery_id: str | None = None,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
        interaction_id: str | None = None,
        interaction_started_event: Event | None = None,
    ) -> SessionMessageDeliveryBatch:
        self._delivery_calls += 1
        if self._delivery_calls == self._block_on_call:
            self.delivery_started.set()
            await self.release_delivery.wait()
        return await super().deliver_queued_session_messages(
            session_id,
            include_on_idle=include_on_idle,
            delivery_id=delivery_id,
            eligible_through=eligible_through,
            limit=limit,
            interaction_id=interaction_id,
            interaction_started_event=interaction_started_event,
        )


class CommitThenLoseDeliveryAcknowledgementStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.lost_acknowledgement = False
        self.lost_delivery_id: str | None = None
        self.attempted_delivery_ids: list[str] = []

    async def deliver_queued_session_messages(
        self,
        session_id: str,
        *,
        include_on_idle: bool,
        delivery_id: str | None = None,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
        interaction_id: str | None = None,
        interaction_started_event: Event | None = None,
    ) -> SessionMessageDeliveryBatch:
        result = await super().deliver_queued_session_messages(
            session_id,
            include_on_idle=include_on_idle,
            delivery_id=delivery_id,
            eligible_through=eligible_through,
            limit=limit,
            interaction_id=interaction_id,
            interaction_started_event=interaction_started_event,
        )
        self.attempted_delivery_ids.append(result.delivery_id)
        if result.messages and not result.replayed and not self.lost_acknowledgement:
            self.lost_acknowledgement = True
            self.lost_delivery_id = result.delivery_id
            raise ConnectionError("queue delivery acknowledgement lost")
        return result


async def _assert_public_delivery_resolves_to_queue(
    store: InMemorySessionStore,
    delivery: Event,
    expected_queue_id: str,
) -> None:
    assert delivery.payload["queue_id"] == PRIVATE_EVENT_AUTHORITY
    delivery_sequence = public_event_sequence(delivery.id)
    assert delivery_sequence is not None
    durable_delivery = await store.query_events(
        EventQuery(
            session_id=delivery.session_id,
            after_sequence=delivery_sequence - 1,
            limit=1,
        )
    )
    assert [record.sequence for record in durable_delivery] == [delivery_sequence]
    assert durable_delivery[0].event.type is EventType.SESSION_MESSAGE_DELIVERED
    assert durable_delivery[0].event.payload["queue_id"] == expected_queue_id


def test_enqueue_session_message_request_validates_public_contract() -> None:
    with pytest.raises(ValidationError):
        EnqueueSessionMessageRequest(  # type: ignore[call-arg]
            session_id="session_1",
            content="hello",
            delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
        )
    with pytest.raises(ValidationError):
        EnqueueSessionMessageRequest(
            session_id="session_1",
            idempotency_key="message-1",
            content="   ",
            delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
        )
    with pytest.raises(ValidationError):
        EnqueueSessionMessageRequest(
            session_id="session_1",
            idempotency_key="message-1",
            content="x" * 65_537,
            delivery_mode="later",  # type: ignore[arg-type]
        )
    for field, value in (("idempotency_key", "message\x00key"), ("content", "hello\x00")):
        values = {
            "session_id": "session_1",
            "idempotency_key": "message-1",
            "content": "hello",
            "delivery_mode": SessionMessageDeliveryMode.NEXT_TURN,
            field: value,
        }
        with pytest.raises(ValidationError, match="NUL"):
            EnqueueSessionMessageRequest(**values)  # type: ignore[arg-type]


def test_enqueue_session_message_redacts_before_durable_queue_write() -> None:
    secret = "queued-steering-boundary-canary"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def run():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_redacted_steering",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        result = await app.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="sess_redacted_steering",
                idempotency_key="steer-redacted",
                content=f"continue with {secret}",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
        )
        await store.update_status("sess_redacted_steering", SessionStatus.RUNNING)
        batch = await store.deliver_queued_session_messages(
            "sess_redacted_steering",
            include_on_idle=True,
        )
        return result, batch

    result, batch = asyncio.run(run())

    assert result.message.content == f"continue with {REDACTED_SECRET}"
    assert batch.messages[0].content == result.message.content
    assert secret not in str(result.model_dump(mode="json"))
    assert secret not in str(batch.model_dump(mode="json"))


def test_cross_process_enqueue_drains_next_turn_before_on_idle_without_event_content() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = BlockingTwoTurnProvider()
        controller = CayuApp(session_store=store, enable_logging=False)
        controller.register_provider(provider)
        controller.register_agent(AgentSpec(name="assistant", model="fake-model"))
        accepting_process = CayuApp(session_store=store, enable_logging=False)
        run_events = []

        async def execute() -> None:
            async for event in controller.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_durable_steering",
                    messages=[Message.text("user", "initial request")],
                )
            ):
                run_events.append(event)

        run_task = asyncio.create_task(execute())
        await provider.first_started.wait()
        idle_result = await accepting_process.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="sess_durable_steering",
                idempotency_key="steer-idle",
                content="idle steering",
                delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
            )
        )
        next_results = await asyncio.gather(
            *(
                accepting_process.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id="sess_durable_steering",
                        idempotency_key=f"steer-next-{index}",
                        content=f"next steering {index}",
                        delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                    )
                )
                for index in range(3)
            )
        )
        provider.release_first.set()
        await run_task

        assert idle_result.replayed is False
        assert all(result.replayed is False for result in next_results)
        assert all(
            result.message.ordering_key > idle_result.message.ordering_key
            for result in next_results
        )
        assert all(
            result.event.type == EventType.SESSION_MESSAGE_QUEUED
            and "content" not in result.event.payload
            for result in next_results
        )
        ordered_next = sorted(next_results, key=lambda result: result.message.ordering_key)
        assert len(provider.requests) == 2
        second_request_text = [
            part.text
            for message in provider.requests[1].messages
            if message.role == MessageRole.USER
            for part in message.content
            if hasattr(part, "text")
        ]
        assert second_request_text[-4:] == [
            *(result.message.content for result in ordered_next),
            "idle steering",
        ]

        transcript = await store.load_transcript("sess_durable_steering")
        assert [message.role for message in transcript] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.USER,
            MessageRole.USER,
            MessageRole.USER,
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        assert [message.content[0].text for message in transcript[2:6]] == [  # type: ignore[union-attr]
            *(result.message.content for result in ordered_next),
            "idle steering",
        ]
        stored_events = await store.load_events("sess_durable_steering")
        delivery_events = [
            event for event in stored_events if event.type == EventType.SESSION_MESSAGE_DELIVERED
        ]
        assert [event.payload["queue_id"] for event in delivery_events] == [
            *(result.message.queue_id for result in ordered_next),
            idle_result.message.queue_id,
        ]
        assert all("content" not in event.payload for event in delivery_events)

        model_started = [event for event in run_events if event.type == EventType.MODEL_STARTED]
        assert len(model_started) == 2
        first_interaction_id = model_started[0].interaction_id
        second_interaction_id = model_started[1].interaction_id
        assert first_interaction_id is not None
        assert second_interaction_id is not None
        assert second_interaction_id != first_interaction_id
        assert {event.interaction_id for event in delivery_events} == {second_interaction_id}
        queued_interaction_start = next(
            event
            for event in stored_events
            if event.type == EventType.INTERACTION_STARTED
            and event.interaction_id == second_interaction_id
        )
        assert stored_events.index(queued_interaction_start) < stored_events.index(
            delivery_events[0]
        )

        first_records = await store.query_transcript(
            TranscriptQuery(
                session_id="sess_durable_steering",
                interaction_id=first_interaction_id,
            )
        )
        second_records = await store.query_transcript(
            TranscriptQuery(
                session_id="sess_durable_steering",
                interaction_id=second_interaction_id,
            )
        )
        assert [record.index for record in first_records.records] == [0, 1]
        assert [record.index for record in second_records.records] == [2, 3, 4, 5, 6]
        terminal_interactions = {
            event.interaction_id
            for event in stored_events
            if event.type
            in {
                EventType.INTERACTION_COMPLETED,
                EventType.INTERACTION_FAILED,
                EventType.INTERACTION_INTERRUPTED,
            }
        }
        assert terminal_interactions == {first_interaction_id, second_interaction_id}

    asyncio.run(run())


def test_next_turn_waits_for_complete_tool_round_before_provider_delivery() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = ToolRoundProvider()
        tool = BlockingTool()
        controller = CayuApp(session_store=store, enable_logging=False)
        controller.register_provider(provider)
        controller.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        accepting_process = CayuApp(session_store=store, enable_logging=False)

        async def execute() -> None:
            async for _event in controller.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_tool_round_steering",
                    messages=[Message.text("user", "use the tool")],
                )
            ):
                pass

        run_task = asyncio.create_task(execute())
        await tool.started.wait()
        accepted = await accepting_process.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="sess_tool_round_steering",
                idempotency_key="during-tool-round",
                content="steer only after the tool result",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
        )
        transcript_during_tool = await store.load_transcript("sess_tool_round_steering")
        assert all(
            message.role is not MessageRole.USER
            or message.content[0].text != "steer only after the tool result"  # type: ignore[union-attr]
            for message in transcript_during_tool
        )

        tool.release.set()
        await run_task

        assert len(provider.requests) == 3
        assert [message.role for message in provider.requests[1].messages] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
        ]
        assert provider.requests[2].messages[-1].content[0].text == (  # type: ignore[union-attr]
            "steer only after the tool result"
        )
        events = await store.load_events("sess_tool_round_steering")
        delivery = next(
            event for event in events if event.type == EventType.SESSION_MESSAGE_DELIVERED
        )
        tool_completed_index = next(
            index
            for index, event in enumerate(events)
            if event.type == EventType.TOOL_CALL_COMPLETED
        )
        assert events.index(delivery) > tool_completed_index
        assert delivery.payload["queue_id"] == accepted.message.queue_id
        model_started = [event for event in events if event.type == EventType.MODEL_STARTED]
        first_interaction_id = model_started[0].interaction_id
        second_interaction_id = model_started[2].interaction_id
        assert first_interaction_id is not None
        assert model_started[1].interaction_id == first_interaction_id
        assert second_interaction_id is not None
        assert second_interaction_id != first_interaction_id
        first_completed_index = next(
            index
            for index, event in enumerate(events)
            if event.type == EventType.INTERACTION_COMPLETED
            and event.interaction_id == first_interaction_id
        )
        second_started_index = next(
            index
            for index, event in enumerate(events)
            if event.type == EventType.INTERACTION_STARTED
            and event.interaction_id == second_interaction_id
        )
        assert first_completed_index < second_started_index < events.index(delivery)
        assert {
            event.interaction_id
            for event in events
            if event.type == EventType.INTERACTION_COMPLETED
        } == {first_interaction_id, second_interaction_id}

    asyncio.run(run())


def test_queued_message_waits_for_pending_tool_approval_resolution() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = BlockingApprovalProvider()
        tool = BlockingTool()
        controller = CayuApp(session_store=store, enable_logging=False)
        controller.register_provider(provider)
        controller.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
        accepting_process = CayuApp(session_store=store, enable_logging=False)
        run_events = []

        async def execute() -> None:
            async for event in controller.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_approval_steering",
                    messages=[Message.text("user", "use the protected tool")],
                )
            ):
                run_events.append(event)

        run_task = asyncio.create_task(execute())
        await provider.first_started.wait()
        accepted = await accepting_process.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="sess_approval_steering",
                idempotency_key="during-pending-approval",
                content="steer only after approval and the tool result",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
        )
        provider.release_first.set()
        await run_task

        approval_event = next(
            event for event in run_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        interrupted = await store.load("sess_approval_steering")
        assert interrupted is not None
        assert interrupted.status is SessionStatus.INTERRUPTED
        assert not tool.started.is_set()
        transcript_before_approval = await store.load_transcript("sess_approval_steering")
        assert all(
            message.role is not MessageRole.USER
            or message.content[0].text != "steer only after approval and the tool result"  # type: ignore[union-attr]
            for message in transcript_before_approval
        )

        tool.release.set()
        resolution_events = [
            event
            async for event in controller.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="sess_approval_steering",
                    approval_id=approval_event.payload["approval"]["approval_id"],
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id=approval_event.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]

        assert len(provider.requests) == 3
        assert [message.role for message in provider.requests[1].messages] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
        ]
        assert provider.requests[2].messages[-1].content[0].text == (  # type: ignore[union-attr]
            "steer only after approval and the tool result"
        )
        delivery = next(
            event
            for event in resolution_events
            if event.type == EventType.SESSION_MESSAGE_DELIVERED
        )
        tool_completed_index = next(
            index
            for index, event in enumerate(resolution_events)
            if event.type == EventType.TOOL_CALL_COMPLETED
        )
        assert resolution_events.index(delivery) > tool_completed_index
        await _assert_public_delivery_resolves_to_queue(
            store,
            delivery,
            accepted.message.queue_id,
        )
        model_started = [
            event
            for event in [*run_events, *resolution_events]
            if event.type == EventType.MODEL_STARTED
        ]
        assert model_started[0].interaction_id == model_started[1].interaction_id
        assert model_started[2].interaction_id != model_started[1].interaction_id

    asyncio.run(run())


def test_queued_message_does_not_bypass_pending_user_input() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = BlockingUserInputProvider()
        controller = CayuApp(session_store=store, enable_logging=False)
        controller.register_provider(provider)
        controller.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool()],
        )
        accepting_process = CayuApp(session_store=store, enable_logging=False)
        run_events = []

        async def execute() -> None:
            async for event in controller.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_user_input_steering",
                    messages=[Message.text("user", "ask before continuing")],
                )
            ):
                run_events.append(event)

        run_task = asyncio.create_task(execute())
        await provider.first_started.wait()
        accepted = await accepting_process.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="sess_user_input_steering",
                idempotency_key="during-user-input",
                content="steer only after the answer",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
        )
        provider.release_first.set()
        await run_task

        awaiting = next(
            event for event in run_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
        )
        assert not any(event.type == EventType.SESSION_MESSAGE_DELIVERED for event in run_events)
        transcript_before_answer = await store.load_transcript("sess_user_input_steering")
        assert all(
            message.role is not MessageRole.USER
            or message.content[0].text != "steer only after the answer"  # type: ignore[union-attr]
            for message in transcript_before_answer
        )

        resolution_events = [
            event
            async for event in controller.resolve_user_input(
                UserInputResponse(
                    session_id="sess_user_input_steering",
                    input_id=awaiting.payload["input_id"],
                    answer="production",
                )
            )
        ]

        delivery = next(
            event
            for event in resolution_events
            if event.type == EventType.SESSION_MESSAGE_DELIVERED
        )
        await _assert_public_delivery_resolves_to_queue(
            store,
            delivery,
            accepted.message.queue_id,
        )
        assert len(provider.requests) == 3
        assert all(
            message.role is not MessageRole.USER
            or message.content[0].text != "steer only after the answer"  # type: ignore[union-attr]
            for message in provider.requests[1].messages
        )
        assert provider.requests[2].messages[-1].content[0].text == (  # type: ignore[union-attr]
            "steer only after the answer"
        )
        model_started = [
            event
            for event in [*run_events, *resolution_events]
            if event.type == EventType.MODEL_STARTED
        ]
        assert model_started[0].interaction_id == model_started[1].interaction_id
        assert model_started[2].interaction_id != model_started[1].interaction_id

    asyncio.run(run())


def test_interrupt_winning_completion_race_does_not_fail_the_session() -> None:
    async def run() -> None:
        store = CompletionFenceStore()
        provider = BlockingTwoTurnProvider()
        provider.release_first.set()
        controller = CayuApp(session_store=store, enable_logging=False)
        controller.register_provider(provider)
        controller.register_agent(AgentSpec(name="assistant", model="fake-model"))
        interrupting_process = CayuApp(session_store=store, enable_logging=False)
        interrupting_process.register_agent(AgentSpec(name="assistant", model="fake-model"))
        run_events = []
        interrupt_events = []

        async def execute() -> None:
            async for event in controller.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interrupt_completion_race",
                    messages=[Message.text("user", "initial")],
                )
            ):
                run_events.append(event)

        async def interrupt() -> None:
            async for event in interrupting_process.interrupt_session(
                InterruptSessionRequest(session_id="sess_interrupt_completion_race")
            ):
                interrupt_events.append(event)

        run_task = asyncio.create_task(execute())
        await store.completion_started.wait()
        interrupt_task = asyncio.create_task(interrupt())
        await store.interrupting_started.wait()
        store.release_completion.set()
        await run_task
        await interrupt_task

        session = await store.load("sess_interrupt_completion_race")
        assert session is not None and session.status is SessionStatus.INTERRUPTED
        assert not any(event.type == EventType.SESSION_FAILED for event in run_events)
        assert run_events[-1].type == EventType.SESSION_INTERRUPTED
        assert [event.id for event in interrupt_events] == [run_events[-1].id]

    asyncio.run(run())


def test_interrupt_after_dispatch_admission_stops_before_provider_entry() -> None:
    async def run() -> None:
        store = DispatchAdmissionFenceStore()
        provider = RecordingOneShotProvider()
        controller = CayuApp(session_store=store, enable_logging=False)
        controller.register_provider(provider)
        controller.register_agent(AgentSpec(name="assistant", model="fake-model"))
        interrupting_process = CayuApp(session_store=store, enable_logging=False)
        interrupting_process.register_provider(provider)
        interrupting_process.register_agent(AgentSpec(name="assistant", model="fake-model"))
        run_events: list[Event] = []
        interrupt_events: list[Event] = []
        session_id = "sess_interrupt_after_dispatch_admission"

        async def execute() -> None:
            async for event in controller.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "initial")],
                )
            ):
                run_events.append(event)

        async def interrupt() -> None:
            async for event in interrupting_process.interrupt_session(
                InterruptSessionRequest(session_id=session_id)
            ):
                interrupt_events.append(event)

        run_task = asyncio.create_task(execute())
        await store.dispatch_admitted.wait()
        interrupt_task = asyncio.create_task(interrupt())
        await store.interrupting_started.wait()
        store.release_dispatch_acknowledgement.set()
        await run_task
        await interrupt_task

        session = await store.load(session_id)
        assert session is not None and session.status is SessionStatus.INTERRUPTED
        assert provider.requests == []
        assert not any(event.type == EventType.SESSION_FAILED for event in run_events)
        terminal_events = [
            event for event in run_events if event.type == EventType.SESSION_INTERRUPTED
        ]
        assert len(terminal_events) == 1
        assert [event.id for event in interrupt_events] == [terminal_events[0].id]

    asyncio.run(run())


@pytest.mark.parametrize(
    "store_type",
    [UnsupportedTerminalPublicationStore, UndeclaredTerminalTransitionStore],
)
def test_non_task_interrupt_preserves_store_without_paired_publication(
    store_type,
) -> None:
    async def run() -> None:
        store = store_type()
        provider = BlockingTwoTurnProvider()
        controller = CayuApp(session_store=store, enable_logging=False)
        controller.register_provider(provider, default=True)
        controller.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_interrupt_unsupported_terminal_publication"
        run_events: list[Event] = []
        interrupt_events: list[Event] = []

        async def execute() -> None:
            async for event in controller.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "initial")],
                )
            ):
                run_events.append(event)

        async def interrupt() -> None:
            async for event in controller.interrupt_session(
                InterruptSessionRequest(session_id=session_id)
            ):
                interrupt_events.append(event)

        run_task = asyncio.create_task(execute())
        await provider.first_started.wait()
        await interrupt()
        await run_task

        current = await store.load(session_id)
        assert current is not None and current.status is SessionStatus.INTERRUPTED
        durable_events = await store.load_events(session_id)
        assert sum(event.type is EventType.INTERACTION_INTERRUPTED for event in durable_events) == 1
        assert sum(event.type is EventType.SESSION_INTERRUPTED for event in durable_events) == 1
        checkpoint = await store.load_checkpoint(session_id)
        assert invocation_terminal_decision_from_checkpoint(checkpoint) is None
        assert settled_invocation_terminal_decision_from_checkpoint(checkpoint) is None
        assert len(provider.requests) == 1
        assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
        assert not any(event.type is EventType.SESSION_FAILED for event in run_events)
        assert getattr(store, "interrupt_transition_called", True)

    asyncio.run(run())


def test_resumed_invocation_does_not_reuse_predecessor_terminal_decision() -> None:
    async def run() -> None:
        store = InterruptTrackingStore()
        provider = BlockingTwoTurnProvider()
        provider.block_second = True
        owner = CayuApp(session_store=store, enable_logging=False)
        owner.register_provider(provider, default=True)
        owner.register_agent(AgentSpec(name="assistant", model="fake-model"))
        interrupter = CayuApp(session_store=store, enable_logging=False)
        interrupter.register_provider(provider, default=True)
        interrupter.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_interrupt_after_profiled_resume"

        async def consume_run() -> list[Event]:
            return [
                event
                async for event in owner.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "initial")],
                    )
                )
            ]

        async def consume_resume() -> list[Event]:
            return [
                event
                async for event in owner.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "resume")],
                    )
                )
            ]

        async def interrupt(reason: str) -> list[Event]:
            return [
                event
                async for event in interrupter.interrupt_session(
                    InterruptSessionRequest(session_id=session_id, reason=reason)
                )
            ]

        first_run = asyncio.create_task(consume_run())
        await provider.first_started.wait()
        first_interrupt_task = asyncio.create_task(interrupt("first interruption"))
        await store.interrupting_started.wait()
        provider.release_first.set()
        first_run_events, first_interrupt = await asyncio.gather(
            first_run,
            first_interrupt_task,
        )
        assert await owner.drain_background_interruptions(timeout_s=1) is True
        assert await interrupter.drain_background_interruptions(timeout_s=1) is True
        first_checkpoint = await store.load_checkpoint(session_id)
        first_settled = settled_invocation_terminal_decision_from_checkpoint(first_checkpoint)
        assert first_settled is not None

        resumed_run = asyncio.create_task(consume_resume())
        await provider.second_started.wait()
        admitted_checkpoint = await store.load_checkpoint(session_id)
        assert settled_invocation_terminal_decision_from_checkpoint(admitted_checkpoint) is None
        store.interrupting_started.clear()
        second_interrupt_task = asyncio.create_task(interrupt("second interruption"))
        await store.interrupting_started.wait()
        provider.release_second.set()
        resumed_events, second_interrupt = await asyncio.gather(
            resumed_run,
            second_interrupt_task,
        )
        assert await owner.drain_background_interruptions(timeout_s=1) is True
        assert await interrupter.drain_background_interruptions(timeout_s=1) is True

        session = await store.load(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        second_settled = settled_invocation_terminal_decision_from_checkpoint(checkpoint)
        durable_interrupts = [
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.SESSION_INTERRUPTED
        ]
        assert session is not None and session.status is SessionStatus.INTERRUPTED
        assert second_settled is not None and second_settled != first_settled
        assert second_settled.run_epoch > first_settled.run_epoch
        assert second_settled.terminal_event_id == durable_interrupts[-1].id
        assert [event.payload["reason"] for event in durable_interrupts] == [
            "first interruption",
            "second interruption",
        ]
        assert first_run_events[-1].id == first_interrupt[-1].id
        assert resumed_events[-1].id == second_interrupt[-1].id
        assert len(provider.requests) == 2

    asyncio.run(run())


def test_interrupt_winning_loop_entry_delivery_race_does_not_fail_the_session() -> None:
    async def run() -> None:
        store = DeliveryFenceStore(block_on_call=1)
        provider = RecordingOneShotProvider()
        controller = CayuApp(session_store=store, enable_logging=False)
        controller.register_provider(provider)
        controller.register_agent(AgentSpec(name="assistant", model="fake-model"))
        interrupting_process = CayuApp(session_store=store, enable_logging=False)
        interrupting_process.register_agent(AgentSpec(name="assistant", model="fake-model"))
        run_events = []
        interrupt_events = []

        async def execute() -> None:
            async for event in controller.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_interrupt_delivery_race",
                    messages=[Message.text("user", "initial")],
                )
            ):
                run_events.append(event)

        async def interrupt() -> None:
            async for event in interrupting_process.interrupt_session(
                InterruptSessionRequest(session_id="sess_interrupt_delivery_race")
            ):
                interrupt_events.append(event)

        run_task = asyncio.create_task(execute())
        await store.delivery_started.wait()
        interrupt_task = asyncio.create_task(interrupt())
        await store.interrupting_started.wait()
        store.release_delivery.set()
        await run_task
        await interrupt_task

        session = await store.load("sess_interrupt_delivery_race")
        assert session is not None and session.status is SessionStatus.INTERRUPTED
        assert provider.requests == []
        assert not any(event.type == EventType.SESSION_FAILED for event in run_events)
        assert run_events[-1].type == EventType.SESSION_INTERRUPTED
        assert [event.id for event in interrupt_events] == [run_events[-1].id]

    asyncio.run(run())


@pytest.mark.parametrize("status_acknowledgement_loss", [False, True])
def test_interrupt_winning_completion_queue_drain_preserves_queued_message(
    status_acknowledgement_loss: bool,
) -> None:
    async def run() -> None:
        store = DeliveryFenceStore(
            block_on_call=2,
            lose_closed_terminal_status_acknowledgement=status_acknowledgement_loss,
        )
        provider = BlockingTwoTurnProvider()
        controller = CayuApp(session_store=store, enable_logging=False)
        controller.register_provider(provider)
        controller.register_agent(AgentSpec(name="assistant", model="fake-model"))
        accepting_process = CayuApp(session_store=store, enable_logging=False)
        interrupting_process = CayuApp(session_store=store, enable_logging=False)
        interrupting_process.register_agent(AgentSpec(name="assistant", model="fake-model"))
        run_events = []
        interrupt_events = []

        request = EnqueueSessionMessageRequest(
            session_id="sess_interrupt_completion_drain_race",
            idempotency_key="completion-drain-race",
            content="preserve this queued message",
            delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
        )

        async def execute() -> None:
            async for event in controller.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=request.session_id,
                    messages=[Message.text("user", "initial")],
                )
            ):
                run_events.append(event)

        async def interrupt() -> None:
            async for event in interrupting_process.interrupt_session(
                InterruptSessionRequest(session_id=request.session_id)
            ):
                interrupt_events.append(event)

        run_task = asyncio.create_task(execute())
        await provider.first_started.wait()
        accepted = await accepting_process.enqueue_session_message(request)
        provider.release_first.set()
        await store.delivery_started.wait()
        interrupt_task = asyncio.create_task(interrupt())
        await store.interrupting_started.wait()
        store.release_delivery.set()
        await run_task
        await interrupt_task

        session = await store.load(request.session_id)
        assert session is not None and session.status is SessionStatus.INTERRUPTED
        assert not any(event.type == EventType.SESSION_FAILED for event in run_events)
        assert not any(event.type == EventType.SESSION_MESSAGE_DELIVERED for event in run_events)
        assert run_events[-1].type == EventType.SESSION_INTERRUPTED
        assert [event.id for event in interrupt_events] == [run_events[-1].id]
        durable_events = await store.load_events(request.session_id)
        interaction_terminals = [
            event
            for event in durable_events
            if event.type
            in {
                EventType.INTERACTION_COMPLETED,
                EventType.INTERACTION_FAILED,
                EventType.INTERACTION_INTERRUPTED,
            }
        ]
        assert len(interaction_terminals) == 1
        checkpoint = await store.load_checkpoint(request.session_id)
        settled = settled_invocation_terminal_decision_from_checkpoint(checkpoint)
        assert settled is not None
        assert settled.interaction_event_id is None
        assert settled.predecessor_interaction_event_id == interaction_terminals[0].id
        replay = await accepting_process.enqueue_session_message(request)
        assert replay.replayed is True
        assert replay.message.queue_id == accepted.message.queue_id
        assert replay.message.status == "queued"

    asyncio.run(run())


def test_closed_interaction_terminal_decision_replays_after_status_commit_failure() -> None:
    async def run() -> None:
        store = DeliveryFenceStore(
            block_on_call=2,
            fail_closed_terminal_status_before_commit=True,
        )
        provider = BlockingTwoTurnProvider()
        owner = CayuApp(session_store=store, enable_logging=False)
        owner.register_provider(provider)
        owner.register_agent(AgentSpec(name="assistant", model="fake-model"))
        interrupter = CayuApp(session_store=store, enable_logging=False)
        interrupter.register_provider(provider)
        interrupter.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "closed-interaction-status-recovery"

        async def execute() -> None:
            async for _event in owner.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "initial")],
                )
            ):
                pass

        async def interrupt() -> list[Event]:
            return [
                event
                async for event in interrupter.interrupt_session(
                    InterruptSessionRequest(session_id=session_id)
                )
            ]

        run_task = asyncio.create_task(execute())
        await provider.first_started.wait()
        await owner.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id=session_id,
                idempotency_key="closed-interaction-status-recovery",
                content="preserve for the next invocation",
                delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
            )
        )
        provider.release_first.set()
        await store.delivery_started.wait()
        interrupt_task = asyncio.create_task(interrupt())
        await store.interrupting_started.wait()
        interrupt_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await interrupt_task
        store.release_delivery.set()
        owner_result = (await asyncio.gather(run_task, return_exceptions=True))[0]
        assert isinstance(owner_result, Exception)

        interrupted_before_retry = await store.load(session_id)
        checkpoint_before_retry = await store.load_checkpoint(session_id)
        assert (
            interrupted_before_retry is not None
            and interrupted_before_retry.status is SessionStatus.INTERRUPTING
        )
        assert (
            settled_invocation_terminal_decision_from_checkpoint(checkpoint_before_retry)
            is not None
        )
        assert checkpoint_before_retry is not None
        assert _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in checkpoint_before_retry

        replayed_events = await interrupt()
        interrupted = await store.load(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        durable_events = await store.load_events(session_id)
        assert interrupted is not None and interrupted.status is SessionStatus.INTERRUPTED
        assert _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY not in (checkpoint or {})
        assert [event.type for event in replayed_events] == [EventType.SESSION_INTERRUPTED]
        assert sum(event.type is EventType.SESSION_INTERRUPTED for event in durable_events) == 1

    asyncio.run(run())


def test_task_linked_interrupt_after_interaction_completion_publishes_one_terminal() -> None:
    async def run() -> None:
        store = DeliveryFenceStore(
            block_on_call=2,
            lose_closed_terminal_status_acknowledgement=True,
        )
        task_store = InMemoryTaskStore()
        provider = BlockingTwoTurnProvider()
        owner = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
        )
        owner.register_provider(provider)
        owner.register_agent(AgentSpec(name="assistant", model="fake-model"))
        interrupter = CayuApp(
            session_store=store,
            task_store=task_store,
            enable_logging=False,
        )
        interrupter.register_provider(provider)
        interrupter.register_agent(AgentSpec(name="assistant", model="fake-model"))
        task = await task_store.create_task(
            TaskCreate(task_id="closed-interaction-task", type="job")
        )
        request = EnqueueSessionMessageRequest(
            session_id="closed-interaction-session",
            idempotency_key="closed-interaction-queued",
            content="preserve this queued message",
            delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
        )

        async def handler(runtime, claimed, worker_id):
            assert claimed.lease_expires_at is not None
            async for _event in runtime.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=request.session_id,
                    task_id=claimed.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=claimed.lease_expires_at,
                    messages=[Message.text("user", "initial")],
                )
            ):
                pass
            return TaskHandlerOutcome.SESSION_INTERRUPTED

        worker = asyncio.create_task(
            run_task_worker(
                owner,
                task_store,
                handler,
                worker_id="closed-interaction-worker",
                reclaim=False,
                poll_interval_s=0.001,
                max_tasks=1,
            )
        )
        await provider.first_started.wait()
        accepted = await owner.enqueue_session_message(request)
        provider.release_first.set()
        await store.delivery_started.wait()

        async def interrupt_session() -> list[Event]:
            return [
                event
                async for event in interrupter.interrupt_session(
                    InterruptSessionRequest(session_id=request.session_id)
                )
            ]

        interrupt = asyncio.create_task(interrupt_session())
        await store.interrupting_started.wait()
        store.release_delivery.set()
        assert await worker == 1
        interrupt_events = await interrupt

        session = await store.load(request.session_id)
        retained_task = await task_store.load_task(task.id)
        events = await store.load_events(request.session_id)
        checkpoint = await store.load_checkpoint(request.session_id)
        settled = settled_invocation_terminal_decision_from_checkpoint(checkpoint)
        assert session is not None and session.status is SessionStatus.INTERRUPTED
        assert retained_task is not None and retained_task.status is TaskStatus.RUNNING
        assert retained_task.worker_id is None
        assert sum(event.type is EventType.SESSION_INTERRUPTED for event in events) == 1
        assert (
            sum(
                event.type
                in {
                    EventType.INTERACTION_COMPLETED,
                    EventType.INTERACTION_FAILED,
                    EventType.INTERACTION_INTERRUPTED,
                }
                for event in events
            )
            == 1
        )
        assert settled is not None and settled.interaction_event_id is None
        assert [
            event.id for event in interrupt_events if event.type is EventType.SESSION_INTERRUPTED
        ]
        replay = await owner.enqueue_session_message(request)
        assert replay.replayed is True
        assert replay.message.queue_id == accepted.message.queue_id

    asyncio.run(run())


def test_interrupt_winning_queued_interaction_uses_current_interaction_authority() -> None:
    async def run() -> None:
        store = InterruptDecisionTransitionFenceStore()
        provider = BlockingTwoTurnProvider()
        provider.block_second = True
        controller = CayuApp(session_store=store, enable_logging=False)
        controller.register_provider(provider)
        controller.register_agent(AgentSpec(name="assistant", model="fake-model"))
        accepting_process = CayuApp(session_store=store, enable_logging=False)
        interrupting_process = CayuApp(session_store=store, enable_logging=False)
        interrupting_process.register_provider(provider)
        interrupting_process.register_agent(AgentSpec(name="assistant", model="fake-model"))
        run_events: list[Event] = []
        interrupt_events: list[Event] = []
        session_id = "sess_interrupt_queued_interaction"

        async def execute() -> None:
            async for event in controller.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "initial")],
                )
            ):
                run_events.append(event)

        async def interrupt() -> None:
            async for event in interrupting_process.interrupt_session(
                InterruptSessionRequest(session_id=session_id)
            ):
                interrupt_events.append(event)

        run_task = asyncio.create_task(execute())
        await provider.first_started.wait()
        accepted = await accepting_process.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id=session_id,
                idempotency_key="interrupt-current-interaction",
                content="deliver before interruption",
                delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
            )
        )
        interrupt_task = asyncio.create_task(interrupt())
        await store.interrupt_transition_ready.wait()
        provider.release_first.set()
        await provider.second_started.wait()
        store.release_interrupt_transition.set()
        await store.interrupting_started.wait()
        provider.release_second.set()
        await run_task
        await interrupt_task

        session = await store.load(session_id)
        assert session is not None and session.status is SessionStatus.INTERRUPTED
        assert not any(event.type == EventType.SESSION_FAILED for event in run_events)
        terminal_events = [
            event for event in run_events if event.type == EventType.SESSION_INTERRUPTED
        ]
        assert len(terminal_events) == 1
        assert [event.id for event in interrupt_events] == [terminal_events[0].id]
        interaction_starts = [
            event for event in run_events if event.type == EventType.INTERACTION_STARTED
        ]
        interaction_interruptions = [
            event for event in run_events if event.type == EventType.INTERACTION_INTERRUPTED
        ]
        assert len(interaction_starts) == 2
        assert len(interaction_interruptions) == 1
        assert interaction_interruptions[0].interaction_id == interaction_starts[1].interaction_id
        replay = await accepting_process.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id=session_id,
                idempotency_key="interrupt-current-interaction",
                content="deliver before interruption",
                delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
            )
        )
        assert replay.replayed is True
        assert replay.message.queue_id == accepted.message.queue_id
        assert replay.message.status == "delivered"

    asyncio.run(run())


def test_queued_message_at_model_step_limit_interrupts_and_survives_resume() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = BlockingTwoTurnProvider()
        controller = CayuApp(
            session_store=store,
            execution_profile_policy=_AuthorizeQueuedContinuationPolicy(),
            enable_logging=False,
        )
        controller.register_provider(provider)
        controller.register_agent(AgentSpec(name="assistant", model="fake-model"))
        accepting_process = CayuApp(session_store=store, enable_logging=False)
        run_events = []

        async def execute() -> None:
            async for event in controller.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_queue_step_limit",
                    messages=[Message.text("user", "initial")],
                    max_steps=1,
                )
            ):
                run_events.append(event)

        run_task = asyncio.create_task(execute())
        await provider.first_started.wait()
        accepted = await accepting_process.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="sess_queue_step_limit",
                idempotency_key="queued-at-step-limit",
                content="continue durably",
                delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
            )
        )
        provider.release_first.set()
        await run_task

        interrupted = await store.load("sess_queue_step_limit")
        assert interrupted is not None and interrupted.status is SessionStatus.INTERRUPTED
        assert EventType.SESSION_FAILED not in [event.type for event in run_events]
        limit_event = next(
            event for event in run_events if event.type == EventType.SESSION_LIMIT_REACHED
        )
        assert limit_event.payload["limit"] == "model_steps"
        assert run_events[-1].type == EventType.SESSION_INTERRUPTED
        assert run_events[-1].payload["interruption_type"] == "limit_reached"
        assert not any(event.type == EventType.SESSION_MESSAGE_DELIVERED for event in run_events)

        resumed = [
            event
            async for event in controller.resume(
                ResumeRequest(
                    session_id="sess_queue_step_limit",
                    messages=[Message.text("user", "resume")],
                    max_steps=2,
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key="queued-step-limit-continuation-v1",
                        reason="Continue the queued message with one additional model step.",
                        requested_by=ResolutionActor(
                            subject="test",
                            source=ResolutionActorSource.SYSTEM,
                        ),
                    ),
                )
            )
        ]
        delivery = next(
            event for event in resumed if event.type == EventType.SESSION_MESSAGE_DELIVERED
        )
        await _assert_public_delivery_resolves_to_queue(
            store,
            delivery,
            accepted.message.queue_id,
        )
        completed = await store.load("sess_queue_step_limit")
        assert completed is not None and completed.status is SessionStatus.COMPLETED

    asyncio.run(run())


def test_enqueue_replay_and_conflict_are_deterministic() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_queue_replay",
                messages=[Message.text("user", "initial")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        request = EnqueueSessionMessageRequest(
            session_id="sess_queue_replay",
            idempotency_key="queue-replay-1",
            content="steer",
            delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
        )

        first = await app.enqueue_session_message(request)
        replay = await app.enqueue_session_message(request)

        assert replay.replayed is True
        assert replay.message.queue_id == first.message.queue_id
        assert replay.event.id == first.event.id
        with pytest.raises(ValueError, match="different request"):
            await app.enqueue_session_message(request.model_copy(update={"content": "changed"}))

    asyncio.run(run())


def test_queued_message_survives_interruption_and_is_delivered_on_resume() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = RecordingOneShotProvider()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_queue_resume",
                messages=[Message.text("user", "original request")],
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
            ),
            identity=profiled_session_identity(
                provider_name="recording-one-shot",
                model="fake-model",
                provider=provider,
            ),
        )
        accepting_process = CayuApp(session_store=store, enable_logging=False)
        accepted = await accepting_process.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="sess_queue_resume",
                idempotency_key="survive-interruption",
                content="durable steering after recovery",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
        )
        await store.transition_status(
            "sess_queue_resume",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.INTERRUPTED,
        )

        recovering_process = CayuApp(session_store=store, enable_logging=False)
        recovering_process.register_provider(provider)
        recovering_process.register_agent(AgentSpec(name="assistant", model="fake-model"))
        events = [
            event
            async for event in recovering_process.resume(
                ResumeRequest(
                    session_id="sess_queue_resume",
                    messages=[Message.text("user", "resume context")],
                )
            )
        ]

        assert len(provider.requests) == 1
        user_text = [
            part.text
            for message in provider.requests[0].messages
            if message.role is MessageRole.USER
            for part in message.content
            if hasattr(part, "text")
        ]
        assert user_text == ["resume context", "durable steering after recovery"]
        deliveries = [
            event for event in events if event.type == EventType.SESSION_MESSAGE_DELIVERED
        ]
        assert len(deliveries) == 1
        await _assert_public_delivery_resolves_to_queue(
            store,
            deliveries[0],
            accepted.message.queue_id,
        )
        session = await store.load("sess_queue_resume")
        assert session is not None and session.status is SessionStatus.COMPLETED

    asyncio.run(run())


def test_enqueue_wins_completion_race_or_is_rejected_without_record() -> None:
    async def run() -> None:
        store = CompletionFenceStore()
        provider = BlockingTwoTurnProvider()
        provider.release_first.set()
        controller = CayuApp(session_store=store, enable_logging=False)
        controller.register_provider(provider)
        controller.register_agent(AgentSpec(name="assistant", model="fake-model"))
        accepting_process = CayuApp(session_store=store, enable_logging=False)

        async def execute() -> None:
            async for _event in controller.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_queue_completion_race",
                    messages=[Message.text("user", "initial")],
                )
            ):
                pass

        run_task = asyncio.create_task(execute())
        await store.completion_started.wait()
        accepted = await accepting_process.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="sess_queue_completion_race",
                idempotency_key="completion-race",
                content="arrived before completion",
                delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
            )
        )
        store.release_completion.set()
        await run_task

        assert accepted.message.status == "queued"
        assert len(provider.requests) == 2
        session = await store.load("sess_queue_completion_race")
        assert session is not None and session.status == SessionStatus.COMPLETED

        with pytest.raises(ValueError, match="pending or running"):
            await accepting_process.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session.id,
                    idempotency_key="too-late",
                    content="late",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )

    asyncio.run(run())


def test_runtime_reconstructs_queued_interaction_after_delivery_acknowledgement_loss() -> None:
    async def run() -> None:
        store = CommitThenLoseDeliveryAcknowledgementStore()
        provider = BlockingTwoTurnProvider()
        controller = CayuApp(session_store=store, enable_logging=False)
        controller.register_provider(provider)
        controller.register_agent(AgentSpec(name="assistant", model="fake-model"))
        accepting_process = CayuApp(session_store=store, enable_logging=False)

        async def execute() -> list[Event]:
            return [
                event
                async for event in controller.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_queue_delivery_ack_loss",
                        messages=[Message.text("user", "initial")],
                    )
                )
            ]

        run_task = asyncio.create_task(execute())
        await provider.first_started.wait()
        accepted = await accepting_process.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="sess_queue_delivery_ack_loss",
                idempotency_key="queue-delivery-ack-loss",
                content="continue after commit",
                delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
            )
        )
        provider.release_first.set()
        events = await run_task

        session = await store.load("sess_queue_delivery_ack_loss")
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        assert store.lost_acknowledgement is True
        assert store.lost_delivery_id is not None
        assert store.attempted_delivery_ids.count(store.lost_delivery_id) == 2
        delivery_event = next(
            event for event in events if event.type == EventType.SESSION_MESSAGE_DELIVERED
        )
        assert store.lost_delivery_id == delivery_event.interaction_id
        assert len(provider.requests) == 2
        replay = await accepting_process.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id=session.id,
                idempotency_key="queue-delivery-ack-loss",
                content="continue after commit",
                delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
            )
        )
        assert replay.replayed is True
        assert replay.message.queue_id == accepted.message.queue_id
        assert replay.message.status == "delivered"
        transcript = await store.load_transcript(session.id)
        assert [
            message.content[0].text  # type: ignore[union-attr]
            for message in transcript
            if message.role == "user"
        ] == ["initial", "continue after commit"]
        assert sum(event.type == EventType.SESSION_MESSAGE_DELIVERED for event in events) == 1
        assert sum(event.type == EventType.INTERACTION_STARTED for event in events) == 2
        assert sum(event.type == EventType.INTERACTION_COMPLETED for event in events) == 2

    asyncio.run(run())


def test_sqlite_queue_reconstructs_and_delivers_once_after_reopen(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "durable-queue.sqlite"
        store = SQLiteSessionStore(path)
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_queue_sqlite",
                messages=[Message.text("user", "initial")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        typed_message = Message(
            role=MessageRole.USER,
            content=(
                FilePart(
                    attachment=FileAttachment(
                        artifact_id="artifact:queued-document",
                        kind=FileAttachmentKind.DOCUMENT,
                        filename="contract.pdf",
                        content_type="application/pdf",
                        size_bytes=128,
                    ).model_dump(mode="json")
                ),
            ),
        )
        request = EnqueueSessionMessageRequest(
            session_id="sess_queue_sqlite",
            idempotency_key="sqlite-queue-1",
            content="Attached file: contract.pdf",
            message=typed_message,
            delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
        )
        accepted = await store.enqueue_session_message(request)
        await store.close()

        reopened = SQLiteSessionStore(path)
        try:
            replay = await reopened.enqueue_session_message(request)
            assert replay.replayed is True
            assert replay.message.queue_id == accepted.message.queue_id
            await reopened.transition_status(
                request.session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            batch = await reopened.deliver_queued_session_messages(
                request.session_id,
                include_on_idle=False,
            )
            retry = await reopened.deliver_queued_session_messages(
                request.session_id,
                include_on_idle=False,
            )

            assert [message.queue_id for message in batch.messages] == [accepted.message.queue_id]
            assert batch.messages[0].message == typed_message
            assert retry.messages == ()
            transcript = await reopened.load_transcript(request.session_id)
            assert transcript[-1] == typed_message
        finally:
            await reopened.close()

    asyncio.run(run())


def test_sqlite_queue_delivery_replay_survives_event_retention(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(tmp_path / "queue-retention.sqlite")
        session_id = "sess_queue_delivery_retention"
        interaction_id = "interaction-queue-delivery-retention"
        try:
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "initial")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            accepted = await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key="queue-delivery-retention",
                    content="survive event pruning",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            await store.transition_status(
                session_id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            started = Event(
                id="evt_queue_delivery_retention_started",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            batch = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                delivery_id=interaction_id,
                interaction_id=interaction_id,
                interaction_started_event=started,
            )
            completed = Event(
                id="evt_queue_delivery_retention_completed",
                type=EventType.INTERACTION_COMPLETED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            publication = await store.publish_interaction_transition(
                session_id,
                event=completed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
                only_if_no_queued_messages=True,
            )
            assert publication.status_changed is True

            for event in (accepted.event, *batch.events, completed):
                claim = await store.claim_persisted_event_side_effect(
                    session_id=session_id,
                    event_id=event.id,
                )
                assert claim is not None
                await store.mark_persisted_event_side_effect_delivered(claim)
            assert (
                await store.prune_events(
                    before=datetime.now(UTC) + timedelta(days=1),
                    session_id=session_id,
                )
                == 4
            )

            replayed = await store.deliver_queued_session_messages(
                session_id,
                include_on_idle=False,
                delivery_id=interaction_id,
                interaction_id=interaction_id,
                interaction_started_event=started,
            )
            assert replayed == batch.model_copy(update={"replayed": True})
        finally:
            await store.close()

    asyncio.run(run())
