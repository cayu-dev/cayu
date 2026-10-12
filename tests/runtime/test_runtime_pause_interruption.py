"""Pause interruption contracts."""

from __future__ import annotations

import asyncio

import pytest
from tests.runtime.runtime_provider_support import FakeProvider, VersionedFakeProvider
from tests.runtime.runtime_recovery_support import FailingAfterPendingToolRoundCheckpointStore
from tests.runtime.runtime_session_support import (
    collect_events,
    collect_interrupt_events,
    collect_user_input_events,
)
from tests.runtime.runtime_tool_support import RequireApprovalPolicy, SideEffectTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import PendingToolApproval
from cayu.approvals.user_input import UserInputResponse
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime import _session_finalization as session_finalization_module
from cayu.runtime._event_projection import public_event_linkage_id, public_event_sequence
from cayu.runtime._interruption_coordinator import suppress_interruption_cascade
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions._checkpoint_mutations import SessionRuntimePublicationConflict
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryAction, IncompleteSessionRecoveryRequest
from cayu.sessions.requests import InterruptSessionRequest, RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.policy import ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import SecretRedactor


def test_operator_interrupt_after_atomic_user_input_open_supersedes_exact_pause() -> None:
    class BlockingUserInputFanOutStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.awaiting_event_id: str | None = None
            self.user_input_fan_out_started = asyncio.Event()

        async def publish_runtime_publication(self, session_id: str, **kwargs):
            result = await super().publish_runtime_publication(session_id, **kwargs)
            if kwargs["request"].kind == "user-input-open":
                awaiting = next(
                    event
                    for event in kwargs["request"].events
                    if event.type is EventType.SESSION_AWAITING_USER_INPUT
                )
                self.awaiting_event_id = awaiting.id
            return result

        async def claim_persisted_event_side_effect(self, *, session_id: str, event_id: str):
            if event_id == self.awaiting_event_id:
                self.user_input_fan_out_started.set()
                await asyncio.Event().wait()
            return await super().claim_persisted_event_side_effect(
                session_id=session_id,
                event_id=event_id,
            )

    async def run() -> None:
        session_id = "sess_interrupt_after_user_input_open"
        store = BlockingUserInputFanOutStore()
        provider = FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_ask",
                    name="ask_user",
                    arguments={"question": "Continue?"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            secret_redactor=SecretRedactor(["operator_requested"]),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool()],
        )

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "ask")],
                ),
            )
        )
        await asyncio.wait_for(store.user_input_fan_out_started.wait(), timeout=10)
        checkpoint_before = await store.load_checkpoint(session_id)
        assert checkpoint_before is not None
        pending_before = checkpoint_before["pending_user_input"]
        private_input_id = pending_before["input_id"]
        open_receipt = await store.load_runtime_publication_receipt(
            session_id,
            f"user-input-open:{private_input_id}",
        )
        assert open_receipt is not None

        request = InterruptSessionRequest(
            session_id=session_id,
            reason="operator supersedes the question",
        )
        with suppress_interruption_cascade():
            interrupt_task = asyncio.create_task(collect_interrupt_events(app, request))
        run_events, interrupt_events = await asyncio.wait_for(
            asyncio.gather(run_task, interrupt_task),
            timeout=10,
        )
        assert run_events[-1].type is EventType.SESSION_INTERRUPTED
        assert interrupt_events[-1].type is EventType.SESSION_INTERRUPTED

        checkpoint_after = await store.load_checkpoint(session_id)
        assert checkpoint_after is not None
        assert "pending_user_input" not in checkpoint_after
        assert "user_input_resolution_intent" not in checkpoint_after
        records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_INTERRUPTED,
            )
        )
        assert len(records) == 1
        marker = records[0].event.payload["user_input_supersession_intent"]
        assert set(marker) == (set(open_receipt.intent) - {"source_round_digest", "event_ids"}) | {
            "state"
        }
        assert marker["state"] == "active"
        assert all(marker[key] == open_receipt.intent[key] for key in marker if key != "state")
        assert marker["input_id"] == private_input_id
        assert marker["pause_digest"] == open_receipt.intent["pause_digest"]
        assert marker["source_run_epoch"] == open_receipt.source_run_epoch

        repeated = await collect_interrupt_events(app, request)
        assert repeated[-1].id == interrupt_events[-1].id
        awaiting_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_AWAITING_USER_INPUT,
            )
        )
        assert len(awaiting_records) == 1
        public_input_id = public_event_linkage_id(awaiting_records[0].sequence, "input_id")
        with pytest.raises(
            SessionRuntimePublicationConflict,
            match="superseded by an external interruption",
        ):
            _ = [
                event
                async for event in app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id,
                        input_id=public_input_id,
                        answer="stale",
                    )
                )
            ]
        assert await store.load_checkpoint(session_id) == checkpoint_after

    asyncio.run(run())


def test_remote_operator_interrupt_wins_before_atomic_user_input_open() -> None:
    class BlockingUserInputOpenStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.open_started = asyncio.Event()
            self.allow_open = asyncio.Event()
            self.interrupt_claimed = asyncio.Event()
            self.open_committed = False

        async def publish_runtime_publication(self, session_id: str, **kwargs):
            if kwargs["request"].kind == "user-input-open":
                self.open_started.set()
                await self.allow_open.wait()
            result = await super().publish_runtime_publication(session_id, **kwargs)
            if kwargs["request"].kind == "user-input-open":
                self.open_committed = True
            return result

        async def transition_status_and_checkpoint(self, session_id: str, **kwargs):
            result = await super().transition_status_and_checkpoint(session_id, **kwargs)
            if kwargs["to_status"] is SessionStatus.INTERRUPTING:
                self.interrupt_claimed.set()
            return result

    async def run() -> None:
        session_id = "sess_remote_interrupt_before_user_input_open"
        store = BlockingUserInputOpenStore()
        owner = CayuApp(session_store=store, enable_logging=False)
        interrupter = CayuApp(session_store=store, enable_logging=False)
        for app in (owner, interrupter):
            app.register_provider(
                FakeProvider(
                    [
                        ModelStreamEvent.tool_call(
                            id="call_ask",
                            name="ask_user",
                            arguments={"question": "Continue?"},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ]
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )

        run_task = asyncio.create_task(
            collect_events(
                owner,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "ask")],
                ),
            )
        )
        await asyncio.wait_for(store.open_started.wait(), timeout=10)
        request = InterruptSessionRequest(session_id=session_id, reason="remote operator")
        with suppress_interruption_cascade():
            interrupt_task = asyncio.create_task(collect_interrupt_events(interrupter, request))
        await asyncio.wait_for(store.interrupt_claimed.wait(), timeout=10)
        store.allow_open.set()

        run_events, interrupt_events = await asyncio.wait_for(
            asyncio.gather(run_task, interrupt_task),
            timeout=10,
        )
        assert run_events[-1].type is EventType.SESSION_INTERRUPTED
        assert interrupt_events[-1].id == run_events[-1].id
        assert interrupt_events[-1].payload["interruption_type"] == "operator_requested"
        stored_events = await store.load_events(session_id)
        assert EventType.SESSION_FAILED not in [event.type for event in stored_events]
        assert not any(
            event.payload.get("interruption_type") == "user_input_required"
            for event in stored_events
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "pending_user_input" not in checkpoint
        assert "pending_tool_round" not in checkpoint
        assert store.open_committed is False

    asyncio.run(run())


def test_remote_operator_interrupt_after_user_input_open_suppresses_stale_pause_terminal() -> None:
    class BlockingUserInputFanOutStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.awaiting_event_id: str | None = None
            self.fan_out_started = asyncio.Event()
            self.allow_fan_out = asyncio.Event()
            self.interrupt_claimed = asyncio.Event()

        async def publish_runtime_publication(self, session_id: str, **kwargs):
            result = await super().publish_runtime_publication(session_id, **kwargs)
            if kwargs["request"].kind == "user-input-open":
                self.awaiting_event_id = next(
                    event.id
                    for event in kwargs["request"].events
                    if event.type is EventType.SESSION_AWAITING_USER_INPUT
                )
            return result

        async def claim_persisted_event_side_effect(self, *, session_id: str, event_id: str):
            if event_id == self.awaiting_event_id:
                self.fan_out_started.set()
                await self.allow_fan_out.wait()
            return await super().claim_persisted_event_side_effect(
                session_id=session_id,
                event_id=event_id,
            )

        async def transition_status_and_checkpoint(self, session_id: str, **kwargs):
            result = await super().transition_status_and_checkpoint(session_id, **kwargs)
            if kwargs["to_status"] is SessionStatus.INTERRUPTING:
                self.interrupt_claimed.set()
            return result

    async def run() -> None:
        session_id = "sess_remote_interrupt_after_user_input_open"
        store = BlockingUserInputFanOutStore()
        owner = CayuApp(session_store=store, enable_logging=False)
        interrupter = CayuApp(session_store=store, enable_logging=False)
        for app in (owner, interrupter):
            app.register_provider(
                FakeProvider(
                    [
                        ModelStreamEvent.tool_call(
                            id="call_ask",
                            name="ask_user",
                            arguments={"question": "Continue?"},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ]
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )

        run_task = asyncio.create_task(
            collect_events(
                owner,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "ask")],
                ),
            )
        )
        await asyncio.wait_for(store.fan_out_started.wait(), timeout=10)
        request = InterruptSessionRequest(session_id=session_id, reason="remote operator")
        with suppress_interruption_cascade():
            interrupt_task = asyncio.create_task(collect_interrupt_events(interrupter, request))
        await asyncio.wait_for(store.interrupt_claimed.wait(), timeout=10)
        store.allow_fan_out.set()

        run_events, interrupt_events = await asyncio.wait_for(
            asyncio.gather(run_task, interrupt_task),
            timeout=10,
        )
        assert run_events[-1].type is EventType.SESSION_INTERRUPTED
        assert interrupt_events[-1].id == run_events[-1].id
        assert interrupt_events[-1].payload["interruption_type"] == "operator_requested"
        stored_events = await store.load_events(session_id)
        terminals = [
            event for event in stored_events if event.type is EventType.SESSION_INTERRUPTED
        ]
        assert len(terminals) == 1
        assert terminals[0].payload["interruption_type"] == "operator_requested"
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "pending_user_input" not in checkpoint
        assert "user_input_resolution_intent" not in checkpoint

    asyncio.run(run())


def test_operator_interrupt_does_not_replay_a_newer_user_input_pause_as_success() -> None:
    class BlockingStalePauseInterruptStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.block_operator_interrupt = False
            self.operator_transition_started = asyncio.Event()
            self.allow_operator_transition = asyncio.Event()

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform,
            **kwargs,
        ) -> Session:
            if (
                self.block_operator_interrupt
                and to_status is SessionStatus.INTERRUPTING
                and SessionStatus.INTERRUPTED in from_statuses
            ):
                self.block_operator_interrupt = False
                self.operator_transition_started.set()
                await self.allow_operator_transition.wait()
            return await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                **kwargs,
            )

    async def run() -> None:
        session_id = "sess_interrupt_replaced_user_input_pause"
        store = BlockingStalePauseInterruptStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_ask_first",
                        name="ask_user",
                        arguments={"question": "First question?"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="call_ask_second",
                        name="ask_user",
                        arguments={"question": "Second question?"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        interrupt_app = CayuApp(session_store=store, enable_logging=False)
        for candidate in (app, interrupt_app):
            candidate.register_provider(provider, default=True)
            candidate.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[UserInputTool()],
            )

        first_pause = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "ask twice")],
            ),
        )
        first_awaiting = next(
            event for event in first_pause if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        first_checkpoint = await store.load_checkpoint(session_id)
        assert first_checkpoint is not None
        first_private_input_id = first_checkpoint["pending_user_input"]["input_id"]

        store.block_operator_interrupt = True
        with suppress_interruption_cascade():
            interrupting = asyncio.create_task(
                collect_interrupt_events(
                    interrupt_app,
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="operator stop",
                    ),
                )
            )
        await asyncio.wait_for(store.operator_transition_started.wait(), timeout=10)

        second_pause = await collect_user_input_events(
            app,
            UserInputResponse(
                session_id=session_id,
                input_id=first_awaiting.payload["input_id"],
                answer="continue",
            ),
        )
        second_awaiting = next(
            event for event in second_pause if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        checkpoint_before_release = await store.load_checkpoint(session_id)
        assert checkpoint_before_release is not None
        second_private_input_id = checkpoint_before_release["pending_user_input"]["input_id"]
        assert second_private_input_id != first_private_input_id

        store.allow_operator_transition.set()
        outcome = await asyncio.gather(interrupting, return_exceptions=True)
        assert len(outcome) == 1
        assert isinstance(outcome[0], SessionRuntimePublicationConflict)
        assert "pause changed before supersession" in str(outcome[0])

        final_checkpoint = await store.load_checkpoint(session_id)
        assert final_checkpoint is not None
        assert final_checkpoint["pending_user_input"]["input_id"] == second_private_input_id
        assert second_awaiting.payload["input_id"] != first_awaiting.payload["input_id"]
        operator_events = [
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.SESSION_INTERRUPTED
            and event.payload.get("interruption_type") == "operator_requested"
        ]
        assert operator_events == []

    asyncio.run(run())


def test_operator_interrupt_after_answer_claim_supersedes_exact_resolution() -> None:
    class BlockingCommittedAnswerClaimStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.block_answer_claim = False
            self.answer_claim_committed = asyncio.Event()
            self.finish_answer_claim = asyncio.Event()

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform,
            **kwargs,
        ) -> Session:
            session = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                **kwargs,
            )
            if self.block_answer_claim and to_status is SessionStatus.RUNNING:
                checkpoint = await self.load_checkpoint(session_id)
                if checkpoint is not None and "user_input_resolution_intent" in checkpoint:
                    self.block_answer_claim = False
                    self.answer_claim_committed.set()
                    await self.finish_answer_claim.wait()
            return session

    async def run() -> None:
        session_id = "sess_interrupt_after_user_input_answer_claim"
        store = BlockingCommittedAnswerClaimStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_ask",
                        name="ask_user",
                        arguments={"question": "Continue?"},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_side_effect",
                        name="side_effect",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("should not dispatch"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        side_effect = SideEffectTool()
        redactor = SecretRedactor(["answering", "operator_requested"])
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            secret_redactor=redactor,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), side_effect],
        )
        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "ask")],
            ),
        )
        awaiting = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        public_input_id = awaiting.payload["input_id"]
        private_awaiting = next(
            record.event
            for record in await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_AWAITING_USER_INPUT,
                )
            )
        )
        private_input_id = private_awaiting.payload["input_id"]

        store.block_answer_claim = True
        resolution_task = asyncio.create_task(
            collect_user_input_events(
                app,
                UserInputResponse(
                    session_id=session_id,
                    input_id=public_input_id,
                    answer="yes",
                ),
            )
        )
        await asyncio.wait_for(store.answer_claim_committed.wait(), timeout=5)
        claimed_checkpoint = await store.load_checkpoint(session_id)
        assert claimed_checkpoint is not None
        resolution_intent = claimed_checkpoint["user_input_resolution_intent"]

        interrupt_app = CayuApp(
            session_store=store,
            enable_logging=False,
            secret_redactor=redactor,
        )
        interrupt_app.register_provider(provider, default=True)
        interrupt_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), side_effect],
        )
        request = InterruptSessionRequest(
            session_id=session_id,
            reason="operator wins after answer claim",
        )
        with pytest.raises(TimeoutError, match="still finalizing"):
            await collect_interrupt_events(interrupt_app, request)
        checkpoint_after_claim = await store.load_checkpoint(session_id)
        assert checkpoint_after_claim is not None
        assert "pending_user_input" not in checkpoint_after_claim
        assert "user_input_resolution_intent" not in checkpoint_after_claim

        store.finish_answer_claim.set()
        resolved = await asyncio.wait_for(resolution_task, timeout=10)
        assert resolved[-1].type is EventType.SESSION_INTERRUPTED
        assert len(provider.requests) == 1
        assert side_effect.calls == []
        assert not any(
            record.event.type is EventType.TOOL_CALL_STARTED
            and record.event.payload.get("tool_call_id") == "call_side_effect"
            for record in await store.query_events(EventQuery(session_id=session_id))
        )
        assert not any(
            record.event.type is EventType.SESSION_RESUMED
            for record in await store.query_events(EventQuery(session_id=session_id))
        )
        assert (
            await store.load_runtime_publication_receipt(
                session_id=session_id,
                publication_id=f"user-input-close:{private_input_id}",
            )
            is None
        )
        interruption_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_INTERRUPTED,
            )
        )
        operator_records = [
            record
            for record in interruption_records
            if record.event.payload.get("interruption_type") == "operator_requested"
        ]
        assert operator_records, [record.event.payload for record in interruption_records]
        marker = operator_records[-1].event.payload["user_input_supersession_intent"]
        assert marker["state"] == "answering"
        assert marker["input_id"] == private_input_id
        assert marker["claim_run_epoch"] == resolution_intent["claim_run_epoch"]
        assert marker["resolution_request_digest"] == resolution_intent["resolution_request_digest"]
        repeated = await collect_interrupt_events(interrupt_app, request)
        assert public_event_sequence(repeated[-1].id) == operator_records[-1].sequence
        checkpoint_after_interrupt = await store.load_checkpoint(session_id)
        assert checkpoint_after_interrupt is not None
        with pytest.raises(
            SessionRuntimePublicationConflict,
            match="superseded by an external interruption",
        ):
            _ = [
                event
                async for event in app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id,
                        input_id=public_input_id,
                        answer="yes",
                    )
                )
            ]
        assert await store.load_checkpoint(session_id) == checkpoint_after_interrupt

    asyncio.run(run())


def test_operator_interrupt_cannot_supersede_executing_user_input_resolution() -> None:
    class BlockingExecutionAdmissionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.block_execution_admission = False
            self.execution_admission_committed = asyncio.Event()
            self.finish_execution_admission = asyncio.Event()

        async def transform_checkpoint(self, session_id, checkpoint_transform) -> None:
            await super().transform_checkpoint(session_id, checkpoint_transform)
            checkpoint = await self.load_checkpoint(session_id)
            intent = None if checkpoint is None else checkpoint.get("user_input_resolution_intent")
            if (
                self.block_execution_admission
                and type(intent) is dict
                and intent.get("execution_state") == "executing"
            ):
                self.block_execution_admission = False
                self.execution_admission_committed.set()
                await self.finish_execution_admission.wait()

    async def run() -> None:
        session_id = "sess_interrupt_after_user_input_execution_admission"
        store = BlockingExecutionAdmissionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_ask",
                        name="ask_user",
                        arguments={"question": "Continue?"},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call_side_effect",
                        name="side_effect",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        side_effect = SideEffectTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), side_effect],
        )
        paused = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "ask")],
            ),
        )
        input_id = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]

        store.block_execution_admission = True
        resolving = asyncio.create_task(
            collect_user_input_events(
                app,
                UserInputResponse(session_id=session_id, input_id=input_id, answer="yes"),
            )
        )
        await asyncio.wait_for(store.execution_admission_committed.wait(), timeout=10)

        interrupt_app = CayuApp(session_store=store, enable_logging=False)
        interrupt_app.register_provider(provider, default=True)
        interrupt_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), side_effect],
        )
        with pytest.raises(
            SessionRuntimePublicationConflict,
            match="already executing and cannot be superseded",
        ):
            await collect_interrupt_events(
                interrupt_app,
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="too late to supersede admitted continuation",
                ),
            )

        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["user_input_resolution_intent"]["execution_state"] == "executing"
        assert not any(
            event.type is EventType.SESSION_INTERRUPTED
            and event.payload.get("interruption_type") == "operator_requested"
            for event in await store.load_events(session_id)
        )

        store.finish_execution_admission.set()
        resolved = await asyncio.wait_for(resolving, timeout=10)
        assert resolved[-1].type is EventType.SESSION_COMPLETED
        assert side_effect.calls == [{}]

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("with_allowed_sibling", [False, True])
def test_process_loss_after_approval_clear_recovers_exact_interrupt_close_intent(
    tmp_path,
    backend,
    with_allowed_sibling,
):
    class ProcessLoss(BaseException):
        pass

    store_base = InMemorySessionStore if backend == "memory" else SQLiteSessionStore

    class BlockingApprovalFanOutStore(store_base):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            if backend == "memory":
                super().__init__()
            else:
                super().__init__(tmp_path / "approval-close.sqlite")
            self.approval_event_id: str | None = None
            self.approval_fan_out_started = asyncio.Event()

        async def publish_runtime_publication(self, session_id: str, **kwargs):
            published = await super().publish_runtime_publication(session_id, **kwargs)
            for event in kwargs["request"].events:
                if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED:
                    self.approval_event_id = event.id
            return published

        async def claim_persisted_event_side_effect(self, *, session_id: str, event_id: str):
            if event_id == self.approval_event_id:
                self.approval_fan_out_started.set()
                await asyncio.Event().wait()
            return await super().claim_persisted_event_side_effect(
                session_id=session_id,
                event_id=event_id,
            )

    class GatePolicy(RequireApprovalPolicy):
        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:approval-close-gate",
                behavior_version="1",
                implementation_version="1",
            )

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            if request.arguments.get("value") == "allowed sibling":
                return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)
            return await super().authorize(request)

    async def run() -> None:
        session_id = "sess_process_loss_after_approval_clear"
        store = BlockingApprovalFanOutStore()
        provider = VersionedFakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_side_effect",
                    name="side_effect",
                    arguments={"value": "must not execute"},
                ),
                *(
                    [
                        ModelStreamEvent.tool_call(
                            id="call_allowed_sibling",
                            name="side_effect",
                            arguments={"value": "allowed sibling"},
                        )
                    ]
                    if with_allowed_sibling
                    else []
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
        tool = SideEffectTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=GatePolicy(),
        )
        original_close = app._tool_round_executor._close_interrupted_round
        close_started = asyncio.Event()

        async def lose_process_after_close_intent(_request):
            close_started.set()
            raise ProcessLoss("process stopped after approval close intent")
            yield  # pragma: no cover

        app._tool_round_executor._close_interrupted_round = lose_process_after_close_intent
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await asyncio.wait_for(store.approval_fan_out_started.wait(), timeout=10)
        with suppress_interruption_cascade():
            interrupt_task = asyncio.create_task(
                collect_interrupt_events(
                    app,
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="operator stop",
                    ),
                )
            )
        await asyncio.wait_for(close_started.wait(), timeout=10)
        run_outcome = await asyncio.gather(run_task, return_exceptions=True)
        assert isinstance(run_outcome[0], ProcessLoss)
        await asyncio.gather(interrupt_task, return_exceptions=True)
        app._tool_round_executor._close_interrupted_round = original_close

        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "pending_tool_approval" not in checkpoint
        assert "pending_tool_round" in checkpoint
        close_intent = checkpoint["pending_session_interrupt"]["approval_close_intent"]
        assert close_intent["tool_round_id"] == checkpoint["pending_tool_round"]["tool_round_id"]
        assert close_intent["tool_call_id"] == "call_side_effect"
        assert tool.calls == []

        # Rebuild the application, and physically reopen persistent storage:
        # recovery authority must come from the serialized close intent.
        if backend == "sqlite":
            await store.close()
            store = BlockingApprovalFanOutStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=GatePolicy(),
        )
        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in recovered.actions
        final_checkpoint = await store.load_checkpoint(session_id)
        assert "pending_tool_approval" not in (final_checkpoint or {})
        assert "pending_tool_round" not in (final_checkpoint or {})
        assert "pending_session_interrupt" not in (final_checkpoint or {})
        final_session = await store.load(session_id)
        assert final_session is not None
        assert final_session.status is SessionStatus.INTERRUPTED
        assert tool.calls == []
        terminals = [
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.TOOL_CALL_FAILED
        ]
        assert len(terminals) == (2 if with_allowed_sibling else 1)
        assert all(
            event.payload["result"]["structured"]["executed"] is False for event in terminals
        )
        assert all(
            event.payload["result"]["structured"]["outcome_unknown"] is False for event in terminals
        )
        if backend == "sqlite":
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("conflicting_identity", [None, "tool_call_id", "tool_round_id"])
def test_process_loss_after_ambiguous_approval_clear_recovers_exact_interrupt_close_intent(
    conflicting_identity,
):
    async def run() -> None:
        from cayu.runtime import _runtime_records as runtime_records

        session_id = "sess_process_loss_after_ambiguous_approval_clear"
        store = FailingAfterPendingToolRoundCheckpointStore()
        provider = FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_side_effect",
                    name="side_effect",
                    arguments={"value": "must not execute"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
        tool = SideEffectTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        initial_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use tool")],
            ),
        )
        assert initial_events[-1].type is EventType.SESSION_FAILED

        recovered_gate = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert recovered_gate.actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
            IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
            IncompleteSessionRecoveryAction.PENDING_APPROVAL,
        )
        reconciled_failures = [
            event for event in recovered_gate.events if event.type is EventType.SESSION_FAILED
        ]
        assert [event.id for event in reconciled_failures] == [initial_events[-1].id]
        gated_checkpoint = await store.load_checkpoint(session_id)
        assert gated_checkpoint is not None
        approval = PendingToolApproval.model_validate(gated_checkpoint["pending_tool_approval"])
        assert approval.tool_calls[0].policy_evidence == "ambiguous"
        assert tool.calls == []

        # Exercise the exact durable boundary written by a real interrupted
        # approval publication, then simulate process loss before the retained
        # round is closed. The ordinary REQUIRE_APPROVAL test above covers
        # delivery through Task.cancel(); this case isolates the new ambiguous
        # evidence representation at that same restart boundary.
        await store.transition_status_and_checkpoint(
            session_id,
            from_statuses={recovered_gate.status},
            to_status=SessionStatus.INTERRUPTING,
            checkpoint_transform=session_finalization_module._checkpoint_with_pending_session_interrupt(
                {
                    "interruption_type": "operator_requested",
                    "interruption_request_id": "interrupt_ambiguous_approval",
                    "reason": "operator stop",
                }
            ),
        )
        await app._tool_round_executor.clear_pending_tool_approval_for_tool_round(
            session_id,
            [
                runtime_records.ToolCallRequest(
                    id=call.tool_call_id,
                    name=call.tool_name,
                    arguments=call.arguments,
                )
                for call in approval.tool_calls
            ],
        )

        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "pending_tool_approval" not in checkpoint
        assert checkpoint["pending_tool_round"]["tool_calls"][0]["policy_evidence"] == "ambiguous"
        close_intent = checkpoint["pending_session_interrupt"]["approval_close_intent"]
        assert close_intent["tool_call_id"] == "call_side_effect"
        assert tool.calls == []

        if conflicting_identity is not None:

            def conflict_close_intent(_session, current):
                current["pending_session_interrupt"]["approval_close_intent"][
                    conflicting_identity
                ] = "different-identity"
                return current

            await store.transform_checkpoint(session_id, conflict_close_intent)
            with pytest.raises(RuntimeError, match="no authoritative decision"):
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                )
            retained = await store.load_checkpoint(session_id)
            assert "pending_tool_round" in retained
            assert tool.calls == []
            assert not any(
                event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
                for event in await store.load_events(session_id)
            )
            return

        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in recovered.actions
        final_checkpoint = await store.load_checkpoint(session_id)
        assert "pending_tool_approval" not in (final_checkpoint or {})
        assert "pending_tool_round" not in (final_checkpoint or {})
        assert "pending_session_interrupt" not in (final_checkpoint or {})
        final_session = await store.load(session_id)
        assert final_session is not None
        assert final_session.status is SessionStatus.INTERRUPTED
        assert tool.calls == []

    asyncio.run(run())
