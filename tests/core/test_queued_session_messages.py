from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from cayu.core import AgentSpec, EventType, Message, MessageRole
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    EnqueueSessionMessageRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
    SessionIdentity,
    SessionMessageDeliveryBatch,
    SessionMessageDeliveryMode,
    SessionStatus,
    ToolApprovalDecision,
    ToolApprovalRequest,
    UserInputResponse,
)
from cayu.runtime.sessions import SESSION_MESSAGE_DELIVERY_BATCH_LIMIT
from cayu.storage import SQLiteSessionStore
from cayu.tools.user_input import UserInputTool


class BlockingTwoTurnProvider(ModelProvider):
    name = "blocking-two-turn"

    def __init__(self) -> None:
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_started.set()
            await self.release_first.wait()
            text = "first answer"
        else:
            text = "steered answer"
        yield ModelStreamEvent.text_delta(text)
        yield ModelStreamEvent.completed({})


class RecordingOneShotProvider(ModelProvider):
    name = "recording-one-shot"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        yield ModelStreamEvent.text_delta("recovered answer")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class ToolRoundProvider(ModelProvider):
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
        yield ModelStreamEvent.text_delta("finished after steering")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class BlockingApprovalProvider(ModelProvider):
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
        yield ModelStreamEvent.text_delta("finished after approved tool and steering")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class BlockingUserInputProvider(ModelProvider):
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
        yield ModelStreamEvent.text_delta("finished after user input and steering")
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
    def __init__(self) -> None:
        super().__init__()
        self.interrupting_started = asyncio.Event()

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform,
    ):
        result = await super().transition_status_and_checkpoint(
            session_id,
            from_statuses=from_statuses,
            to_status=to_status,
            checkpoint_transform=checkpoint_transform,
        )
        if to_status is SessionStatus.INTERRUPTING:
            self.interrupting_started.set()
        return result


class CompletionFenceStore(InterruptTrackingStore):
    def __init__(self) -> None:
        super().__init__()
        self.completion_started = asyncio.Event()
        self.release_completion = asyncio.Event()

    async def transition_status_if_no_queued_messages(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ):
        self.completion_started.set()
        await self.release_completion.wait()
        return await super().transition_status_if_no_queued_messages(
            session_id,
            from_statuses=from_statuses,
            to_status=to_status,
        )


class DeliveryFenceStore(InterruptTrackingStore):
    def __init__(self, *, block_on_call: int) -> None:
        super().__init__()
        self._block_on_call = block_on_call
        self._delivery_calls = 0
        self.delivery_started = asyncio.Event()
        self.release_delivery = asyncio.Event()

    async def deliver_queued_session_messages(
        self,
        session_id: str,
        *,
        include_on_idle: bool,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    ) -> SessionMessageDeliveryBatch:
        self._delivery_calls += 1
        if self._delivery_calls == self._block_on_call:
            self.delivery_started.set()
            await self.release_delivery.wait()
        return await super().deliver_queued_session_messages(
            session_id,
            include_on_idle=include_on_idle,
            eligible_through=eligible_through,
            limit=limit,
        )


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
        delivery_events = [
            event
            for event in await store.load_events("sess_durable_steering")
            if event.type == EventType.SESSION_MESSAGE_DELIVERED
        ]
        assert [event.payload["queue_id"] for event in delivery_events] == [
            *(result.message.queue_id for result in ordered_next),
            idle_result.message.queue_id,
        ]
        assert all("content" not in event.payload for event in delivery_events)

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

        assert len(provider.requests) == 2
        assert [message.role for message in provider.requests[1].messages] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
            MessageRole.USER,
        ]
        assert provider.requests[1].messages[-1].content[0].text == (  # type: ignore[union-attr]
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
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]

        assert len(provider.requests) == 2
        assert [message.role for message in provider.requests[1].messages] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
            MessageRole.USER,
        ]
        assert provider.requests[1].messages[-1].content[0].text == (  # type: ignore[union-attr]
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
        assert delivery.payload["queue_id"] == accepted.message.queue_id

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
        assert delivery.payload["queue_id"] == accepted.message.queue_id
        assert len(provider.requests) == 2
        assert provider.requests[1].messages[-1].content[0].text == (  # type: ignore[union-attr]
            "steer only after the answer"
        )

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


def test_interrupt_winning_completion_queue_drain_preserves_queued_message() -> None:
    async def run() -> None:
        store = DeliveryFenceStore(block_on_call=2)
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
        replay = await accepting_process.enqueue_session_message(request)
        assert replay.replayed is True
        assert replay.message.queue_id == accepted.message.queue_id
        assert replay.message.status == "queued"

    asyncio.run(run())


def test_queued_message_at_model_step_limit_interrupts_and_survives_resume() -> None:
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
                )
            )
        ]
        delivery = next(
            event for event in resumed if event.type == EventType.SESSION_MESSAGE_DELIVERED
        )
        assert delivery.payload["queue_id"] == accepted.message.queue_id
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
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_queue_resume",
                messages=[Message.text("user", "original request")],
            ),
            identity=SessionIdentity(provider_name="recording-one-shot", model="fake-model"),
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

        provider = RecordingOneShotProvider()
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
        assert [event.payload["queue_id"] for event in deliveries] == [accepted.message.queue_id]
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
        request = EnqueueSessionMessageRequest(
            session_id="sess_queue_sqlite",
            idempotency_key="sqlite-queue-1",
            content="survive restart",
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
            assert retry.messages == ()
            transcript = await reopened.load_transcript(request.session_id)
            assert transcript[-1].content[0].text == "survive restart"  # type: ignore[union-attr]
        finally:
            await reopened.close()

    asyncio.run(run())
