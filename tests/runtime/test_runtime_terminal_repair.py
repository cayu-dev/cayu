"""Terminal repair contracts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from tests.core._execution_profile_fixtures import (
    create_admitted_session,
    profiled_session_identity,
    run_request_with_registered_tool_ceiling,
)
from tests.runtime.runtime_environment_support import (
    EnvironmentProvisioningRecorder,
    assert_environment_was_not_provisioned,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_recovery_support import FailingAfterPendingToolRoundCheckpointStore
from tests.runtime.runtime_session_support import (
    _private_record_for_public_event,
    assert_only_model_step_publication_checkpoint,
    checkpoint_without_active_invocation_profile,
    collect_events,
    collect_fork_events,
    collect_interrupt_events,
    collect_resume_events,
    interruption_payload_without_request_id,
)
from tests.runtime.runtime_tool_support import SideEffectTool

import cayu.runtime._session_engine as session_engine_module
import cayu.runtime.execution_profiles as execution_profiles_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY, public_event_linkage_id
from cayu.sessions import _checkpoint_preservation as checkpoint_preservation
from cayu.sessions import _invocation_scope as invocation_scope
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionIdentity, SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryAction, IncompleteSessionRecoveryRequest
from cayu.sessions.requests import (
    ForkSessionRequest,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
)
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import SecretRedactor
from cayu.vaults.static import StaticVault


def test_cayu_app_recover_incomplete_session_interrupts_abandoned_running_session():
    store = InMemorySessionStore()
    later_default_provisioning = EnvironmentProvisioningRecorder("later-default")
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_abandoned_running",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_abandoned_running", SessionStatus.RUNNING)
        app.register_environment_factory(
            EnvironmentSpec(name="later-default"),
            later_default_provisioning.factory,
            default=True,
        )
        return await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="sess_abandoned_running",
                reason="worker restart",
                metadata={"worker": "w1"},
            )
        )

    result = asyncio.run(setup_and_recover())
    session = asyncio.run(store.load("sess_abandoned_running"))
    events = asyncio.run(store.load_events("sess_abandoned_running"))

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert session.environment_name is None
    assert_environment_was_not_provisioned(later_default_provisioning)
    assert result.previous_status == SessionStatus.RUNNING
    assert result.status == SessionStatus.INTERRUPTED
    assert result.actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)
    assert [event.type for event in result.events] == [EventType.SESSION_INTERRUPTED]
    assert result.events[0].environment_name is None
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert interruption_payload_without_request_id(events[-1].payload) == {
        "reason": "worker restart",
        "metadata": {"worker": "w1"},
        "interruption_type": "runtime_interrupted",
        "recovered": True,
    }


def test_cayu_app_recovery_does_not_complete_abandoned_end_turn_false_step():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_abandoned_end_turn_false",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_abandoned_end_turn_false", SessionStatus.RUNNING)
        await store.append_transcript_messages(
            "sess_abandoned_end_turn_false",
            [Message.text("assistant", "I am still working.")],
        )
        await store.append_event(
            "sess_abandoned_end_turn_false",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_abandoned_end_turn_false",
                agent_name="assistant",
                payload={
                    "completion": {
                        "finish_reason": "stop",
                        "raw_finish_reason": None,
                        "status": "completed",
                        "end_turn": False,
                    },
                    "step_classification": {
                        "type": "continue",
                        "reason": "provider requested another model step",
                    },
                },
            ),
        )
        return await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="sess_abandoned_end_turn_false",
                reason="worker restart",
            )
        )

    result = asyncio.run(setup_and_recover())
    session = asyncio.run(store.load("sess_abandoned_end_turn_false"))
    transcript = asyncio.run(store.load_transcript("sess_abandoned_end_turn_false"))

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert result.actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)
    assert [message.role for message in transcript] == ["assistant"]
    assert transcript[-1].content[0].text == "I am still working."


def test_cayu_app_recovery_terminalizes_the_original_durable_interaction():
    from cayu.sessions.interactions import InteractionStatus, InteractionSummaryEvidence

    store = InMemorySessionStore()
    started_at = datetime.now(UTC) - timedelta(seconds=5)
    recovered_at = started_at + timedelta(seconds=10)
    app = CayuApp(session_store=store, clock=lambda: recovered_at)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    interaction_id = "interaction-before-crash"
    start_event = Event(
        id="interaction-before-crash-started",
        type=EventType.INTERACTION_STARTED,
        session_id="sess_recovered_interaction",
        interaction_id=interaction_id,
        timestamp=started_at,
        payload=InteractionSummaryEvidence(
            status=InteractionStatus.ACTIVE,
            start_event_id="interaction-before-crash-started",
            started_at=started_at,
        ).model_dump(mode="json"),
    )

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_recovered_interaction",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_recovered_interaction", SessionStatus.RUNNING)
        await store.append_event("sess_recovered_interaction", start_event)
        await store.append_event(
            "sess_recovered_interaction",
            Event(
                id="interaction-before-crash-model-started",
                type=EventType.MODEL_STARTED,
                session_id="sess_recovered_interaction",
                interaction_id=interaction_id,
                timestamp=started_at + timedelta(seconds=2),
            ),
        )
        return await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_recovered_interaction")
        )

    result = asyncio.run(setup_and_recover())
    lifecycle = asyncio.run(
        store.query_events(
            EventQuery(
                session_id="sess_recovered_interaction",
                interaction_id=interaction_id,
                event_types=(
                    EventType.INTERACTION_STARTED,
                    EventType.INTERACTION_INTERRUPTED,
                ),
            )
        )
    )

    assert [record.event.type for record in lifecycle] == [
        EventType.INTERACTION_STARTED,
        EventType.INTERACTION_INTERRUPTED,
    ]
    assert [event.type for event in result.events] == [
        EventType.INTERACTION_INTERRUPTED,
        EventType.SESSION_INTERRUPTED,
    ]
    evidence = InteractionSummaryEvidence.model_validate(lifecycle[-1].event.payload)
    assert evidence.status == InteractionStatus.INTERRUPTED
    assert evidence.completed_at is not None
    assert evidence.active_duration_ms == 2_000
    assert evidence.wall_duration_ms == 10_000


def test_reactivating_same_interaction_does_not_reset_active_segment(monkeypatch):
    from cayu.sessions.interactions import InteractionStatus, InteractionSummaryEvidence

    monotonic = {"value": 10.0}
    wall = {"value": datetime(2026, 7, 27, tzinfo=UTC)}
    monkeypatch.setattr(session_engine_module.time, "monotonic", lambda: monotonic["value"])
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False, clock=lambda: wall["value"])
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def exercise() -> Event:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_interaction_reactivation_timing",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        session = await store.transition_status(
            session.id,
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )
        invocation_scope._activate_session_interaction(session.id, "interaction-timing")
        registered_agent = app._get_registered_agent("assistant")
        await app._session_outcomes.emit_interaction_started(
            session=session,
            registered_agent=registered_agent,
            environment_name=None,
        )
        monotonic["value"] = 15.0
        invocation_scope._activate_session_interaction(session.id, "interaction-timing")
        monotonic["value"] = 20.0
        wall["value"] += timedelta(seconds=10)
        paused = await app._session_finalization.emit_interaction_state(
            session=session,
            registered_agent=registered_agent,
            environment_name=None,
            event_type=EventType.INTERACTION_PAUSED,
            status=InteractionStatus.PAUSED,
            pending_action_kind="tool_approval",
        )
        assert paused is not None
        await store.release_run_fence(session.id)
        invocation_scope._deactivate_session_interaction(session.id)
        return paused

    evidence = InteractionSummaryEvidence.model_validate(asyncio.run(exercise()).payload)
    assert evidence.active_duration_ms == 10_000


def test_interaction_summary_pages_more_than_5000_usage_events_without_failing():
    from cayu.sessions.interactions import InteractionStatus, InteractionSummaryEvidence

    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def exercise() -> Event:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_large_interaction",
                messages=[Message.text("user", "count tools")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        session = await store.transition_status(
            session.id,
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )
        invocation_scope._activate_session_interaction(session.id, "interaction-large")
        registered_agent = app._get_registered_agent("assistant")
        await app._session_outcomes.emit_interaction_started(
            session=session,
            registered_agent=registered_agent,
            environment_name=None,
        )
        await store.append_events(
            session.id,
            [
                Event(
                    id=f"large-tool-{index}",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session.id,
                    interaction_id="interaction-large",
                )
                for index in range(5001)
            ],
        )
        completed = await app._session_finalization.emit_interaction_state(
            session=session,
            registered_agent=registered_agent,
            environment_name=None,
            event_type=EventType.INTERACTION_COMPLETED,
            status=InteractionStatus.COMPLETED,
        )
        assert completed is not None
        await store.release_run_fence(session.id)
        return completed

    completed = asyncio.run(exercise())
    evidence = InteractionSummaryEvidence.model_validate(completed.payload)
    assert evidence.tool_call_count == 5001


def test_cayu_app_repairs_terminal_evidence_without_registered_agent():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, secret_redactor=SecretRedactor("-"))

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="removed_agent",
                session_id="sess_recover_terminal_missing_agent",
                messages=[Message.text("user", "done")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_recover_terminal_missing_agent", SessionStatus.COMPLETED)
        return await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_recover_terminal_missing_agent")
        )

    result = asyncio.run(setup_and_recover())

    assert result.previous_status == SessionStatus.COMPLETED
    assert result.status == SessionStatus.COMPLETED
    assert result.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
    assert [event.type for event in result.events] == [EventType.SESSION_COMPLETED]
    assert result.events[0].payload == {
        "recovered": True,
        "terminal_evidence_repaired": True,
    }
    assert result.message == "Repaired durable terminal event evidence."
    terminal_records = asyncio.run(
        store.query_events(
            EventQuery(
                session_id="sess_recover_terminal_missing_agent",
                event_type=EventType.SESSION_COMPLETED,
            )
        )
    )
    assert len(terminal_records) == 1
    assert str(UUID(terminal_records[0].event.id)) == terminal_records[0].event.id


def test_terminal_recovery_rejects_receipt_history_without_active_profile() -> None:
    async def scenario() -> None:
        session_id = "sess_terminal_recovery_missing_active_profile"
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        admitted = await create_admitted_session(
            store,
            request=RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "finish without terminal evidence")],
            ),
            provider_name="fake",
            model="fake-model",
            app=app,
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY in checkpoint
        corrupted = dict(checkpoint)
        corrupted.pop(execution_profiles_module.ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY)
        with checkpoint_preservation._invocation_lifecycle_authority_mutation_scope():
            await store.checkpoint(session_id, corrupted)
        before = await store.load_events(session_id)

        with pytest.raises(
            RuntimeError,
            match="lost durable invocation profile authority",
        ):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
        with pytest.raises(
            RuntimeError,
            match="lost durable invocation profile authority",
        ):
            await app._environment_lifecycle._release_quiescent_invocation_fence(
                session_id,
                invocation_context=None,
                terminal_event=None,
            )

        assert await store.load_events(session_id) == before
        current = await store.load(session_id)
        assert current is not None
        assert current.id == admitted.session.id
        assert current.instance_id == admitted.session.instance_id
        assert current.run_epoch == admitted.session.run_epoch
        assert current.status is SessionStatus.COMPLETED
        with pytest.raises(
            SessionRunFenced,
            match="resume lost its durable predecessor invocation authority",
        ):
            await collect_resume_events(
                app,
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "must remain fenced")],
                ),
            )
        after_resume_rejection = await store.load(session_id)
        assert after_resume_rejection == current
        assert await store.load_events(session_id) == before

    asyncio.run(scenario())


def test_cayu_app_resume_repairs_missing_initial_run_terminal_evidence() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            FakeProvider(
                [
                    [
                        ModelStreamEvent.text_delta("continued"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_resume_missing_initial_terminal"
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "start")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        await store.append_transcript_messages(
            session_id,
            [Message.text("user", "start")],
        )
        await store.append_event(
            session_id,
            Event(
                type=EventType.SESSION_STARTED,
                session_id=session_id,
                agent_name="assistant",
            ),
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)

        resumed = await collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
        lifecycle = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(
                    EventType.SESSION_STARTED,
                    EventType.SESSION_RESUMED,
                    EventType.SESSION_COMPLETED,
                ),
            )
        )

        assert resumed[0].type == EventType.SESSION_RESUMED
        assert resumed[-1].type == EventType.SESSION_COMPLETED
        assert [record.event.type for record in lifecycle] == [
            EventType.SESSION_STARTED,
            EventType.SESSION_COMPLETED,
            EventType.SESSION_RESUMED,
            EventType.SESSION_COMPLETED,
        ]
        assert lifecycle[1].event.payload["terminal_evidence_repaired"] is True
        assert "session_run_operation_id" not in lifecycle[1].event.payload
        assert "session_run_operation_id" in lifecycle[-1].event.payload

    asyncio.run(scenario())


def test_cayu_app_resume_cannot_expire_recovery_owner_with_worker_clock() -> None:
    async def scenario() -> None:
        store_now = datetime(2026, 1, 1, tzinfo=UTC)
        worker_now = datetime(2100, 1, 1, tzinfo=UTC)
        store = InMemorySessionStore(ownership_clock=lambda: store_now)
        provider = FakeProvider([])
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: worker_now,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_resume_store_time_recovery_owner"
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "start")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        await store.append_transcript_messages(session_id, [Message.text("user", "start")])
        await store.append_event(
            session_id,
            Event(
                type=EventType.SESSION_STARTED,
                session_id=session_id,
                agent_name="assistant",
            ),
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)

        def install_live_recovery_claim(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            updated = {} if checkpoint is None else dict(checkpoint)
            updated["incomplete_session_recovery_claim"] = {
                "version": 1,
                "claim_id": "store-time-owner",
                "claimed_at": store_now.isoformat(),
                "claim_expires_at": (store_now + timedelta(minutes=5)).isoformat(),
            }
            return updated

        await store.transform_checkpoint(session_id, install_live_recovery_claim)
        before = await store.load(session_id)
        before_events = await store.load_events(session_id)

        with pytest.raises(
            RuntimeError,
            match="active incomplete-session recovery operation",
        ):
            await collect_resume_events(
                app,
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "must remain fenced")],
                ),
            )

        assert await store.load(session_id) == before
        assert await store.load_events(session_id) == before_events
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["incomplete_session_recovery_claim"]["claim_id"] == ("store-time-owner")
        assert provider.requests == []

    asyncio.run(scenario())


def test_cayu_app_resume_cannot_expire_session_operation_with_worker_clock() -> None:
    async def scenario() -> None:
        store_now = datetime(2026, 1, 1, tzinfo=UTC)
        worker_now = datetime(2100, 1, 1, tzinfo=UTC)
        store = InMemorySessionStore(ownership_clock=lambda: store_now)
        provider = FakeProvider([])
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: worker_now,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_resume_store_time_session_operation"
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "start")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        await store.append_transcript_messages(session_id, [Message.text("user", "start")])
        await store.append_event(
            session_id,
            Event(
                type=EventType.SESSION_STARTED,
                session_id=session_id,
                agent_name="assistant",
            ),
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)

        def install_live_operation(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            updated = {} if checkpoint is None else dict(checkpoint)
            updated["session_operations"] = {
                "version": 1,
                "active_operation_id": "live-store-time-operation",
                "records": {
                    "live-store-time-operation-key": {
                        "operation_id": "live-store-time-operation",
                        "status": "running",
                        "claim_expires_at": (store_now + timedelta(minutes=5)).isoformat(),
                    }
                },
            }
            return updated

        await store.transform_checkpoint(session_id, install_live_operation)
        before = await store.load(session_id)
        before_events = await store.load_events(session_id)

        with pytest.raises(RuntimeError, match="active durable operation"):
            await collect_resume_events(
                app,
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "must remain fenced")],
                ),
            )

        assert await store.load(session_id) == before
        assert await store.load_events(session_id) == before_events
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        operations = checkpoint["session_operations"]
        assert operations["active_operation_id"] == "live-store-time-operation"
        assert operations["records"]["live-store-time-operation-key"]["status"] == "running"
        assert provider.requests == []

    asyncio.run(scenario())


def test_cayu_app_recovery_and_resume_preserve_healthy_fork_baseline() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            FakeProvider(
                [
                    [
                        ModelStreamEvent.text_delta("source"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("fork"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_terminal_fork_source",
                messages=[Message.text("user", "start")],
            ),
        )
        await collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_terminal_fork_source",
                session_id="sess_terminal_fork_child",
            ),
        )

        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_terminal_fork_child")
        )
        before_resume = await store.query_events(EventQuery(session_id="sess_terminal_fork_child"))
        resumed = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_terminal_fork_child",
                messages=[Message.text("user", "continue")],
            ),
        )
        lifecycle = await store.query_events(
            EventQuery(
                session_id="sess_terminal_fork_child",
                event_types=(
                    EventType.SESSION_FORKED,
                    EventType.SESSION_RESUMED,
                    EventType.SESSION_COMPLETED,
                ),
            )
        )

        assert recovered.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
        assert [record.event.type for record in before_resume] == [
            EventType.SESSION_FORKED,
            EventType.TARGETED_TOOL_GRANT_FORK_RESET,
        ]
        assert resumed[0].type == EventType.SESSION_RESUMED
        assert resumed[-1].type == EventType.SESSION_COMPLETED
        assert [record.event.type for record in lifecycle] == [
            EventType.SESSION_FORKED,
            EventType.SESSION_RESUMED,
            EventType.SESSION_COMPLETED,
        ]
        assert not any(
            record.event.payload.get("terminal_evidence_repaired") is True for record in lifecycle
        )

    asyncio.run(scenario())


def test_cayu_app_recovery_repairs_missing_pending_approval_terminal_evidence() -> None:
    secret = "late-secret-repaired-approval-pause"

    class ArgumentDerivedApprovalPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            provided = request.arguments["value"]
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason=f"Approve {provided}",
                metadata={"provided": provided},
            )

    class CrashBeforeApprovalPauseEventStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_pause_event = True

        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type == EventType.SESSION_INTERRUPTED and self.fail_pause_event:
                self.fail_pause_event = False
                raise ConnectionError("crash before approval pause event")
            await super().append_event(session_id, event)

    async def scenario() -> None:
        store = CrashBeforeApprovalPauseEventStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            FakeProvider(
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="side_effect",
                        arguments={"value": secret},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                vault=StaticVault({"api_key": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[SideEffectTool()],
            tool_policy=ArgumentDerivedApprovalPolicy(),
        )
        session_id = "sess_missing_approval_pause_terminal"
        with pytest.raises(ConnectionError, match="crash before approval pause event"):
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "use tool")],
                ),
            )

        checkpoint_before = await store.load_checkpoint(session_id)
        assert checkpoint_before is not None
        approval_id = checkpoint_before["pending_tool_approval"]["approval_id"]
        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        terminal_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_INTERRUPTED,
            )
        )
        checkpoint_after = await store.load_checkpoint(session_id)

        assert recovered.actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
            IncompleteSessionRecoveryAction.PENDING_APPROVAL,
        )
        assert len(terminal_records) == 1
        assert recovered.pending_approval_id is not None
        assert (
            await app._resolve_public_action_linkage(
                session_id=session_id,
                value=recovered.pending_approval_id,
                field_name="approval_id",
            )
            == approval_id
        )
        assert recovered.events == (
            app.project_event_record_for_exposure(terminal_records[0]).event,
        )
        assert terminal_records[0].event.payload["interruption_type"] == ("tool_approval_required")
        assert terminal_records[0].event.payload["approval"]["approval_id"] == approval_id
        assert "reason" not in terminal_records[0].event.payload["approval"]
        assert "metadata" not in terminal_records[0].event.payload["approval"]
        assert secret not in repr(terminal_records)
        assert secret not in repr(recovered.events)
        assert checkpoint_after is not None
        assert checkpoint_after["pending_tool_approval"]["approval_id"] == approval_id
        assert "incomplete_session_recovery_claim" not in checkpoint_after

    asyncio.run(scenario())


def test_cayu_app_recovery_repairs_missing_user_input_terminal_evidence() -> None:
    class CrashBeforeUserInputPauseEventStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_pause_event = True

        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type == EventType.SESSION_INTERRUPTED and self.fail_pause_event:
                self.fail_pause_event = False
                raise ConnectionError("crash before user-input pause event")
            await super().append_event(session_id, event)

    async def scenario() -> None:
        store = CrashBeforeUserInputPauseEventStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            FakeProvider(
                [
                    ModelStreamEvent.tool_call(
                        id="call_ask",
                        name="ask_user",
                        arguments={"question": "Proceed?"},
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
        session_id = "sess_missing_user_input_pause_terminal"
        with pytest.raises(ConnectionError, match="crash before user-input pause event"):
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "ask me")],
                ),
            )

        checkpoint_before = await store.load_checkpoint(session_id)
        assert checkpoint_before is not None
        input_id = checkpoint_before["pending_user_input"]["input_id"]
        profile_fingerprint = checkpoint_before["pending_user_input"][
            "execution_profile_fingerprint"
        ]
        assert isinstance(profile_fingerprint, str)
        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        terminal_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_INTERRUPTED,
            )
        )
        checkpoint_after = await store.load_checkpoint(session_id)

        assert recovered.actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
            IncompleteSessionRecoveryAction.PENDING_USER_INPUT,
        )
        public_input_id = public_event_linkage_id(terminal_records[0].sequence, "input_id")
        assert recovered.pending_user_input_id == public_input_id
        assert recovered.events == (
            app.project_event_record_for_exposure(terminal_records[0]).event,
        )
        assert len(terminal_records) == 1
        assert terminal_records[0].event.payload["interruption_type"] == "user_input_required"
        assert (
            terminal_records[0].event.payload["execution_profile_fingerprint"]
            == profile_fingerprint
        )
        assert terminal_records[0].event.payload["user_input"]["input_id"] == input_id
        assert (
            "execution_profile_fingerprint" not in terminal_records[0].event.payload["user_input"]
        )
        assert recovered.events[0].payload["execution_profile_fingerprint"] == profile_fingerprint
        assert recovered.events[0].payload["user_input"]["input_id"] == public_input_id
        assert checkpoint_after is not None
        assert checkpoint_after["pending_user_input"]["input_id"] == input_id
        assert "incomplete_session_recovery_claim" not in checkpoint_after

    asyncio.run(scenario())


def test_cayu_app_recovery_repairs_missing_interrupted_tool_round_terminal_evidence() -> None:
    class CrashAfterPendingToolRoundStore(FailingAfterPendingToolRoundCheckpointStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_terminal_event = True

        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type == EventType.SESSION_FAILED and self.fail_terminal_event:
                self.fail_terminal_event = False
                raise ConnectionError("crash before failed tool-round terminal event")
            await super().append_event(session_id, event)

    async def scenario() -> None:
        store = CrashAfterPendingToolRoundStore()
        tool = SideEffectTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            FakeProvider(
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
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        session_id = "sess_missing_failed_tool_round_terminal"
        with pytest.raises(
            ConnectionError,
            match="crash before failed tool-round terminal event",
        ):
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "use tool")],
                ),
            )

        checkpoint_before = await store.load_checkpoint(session_id)
        assert checkpoint_before is not None
        assert "pending_tool_round" in checkpoint_before
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        terminal_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_INTERRUPTED,
            )
        )
        checkpoint_after = await store.load_checkpoint(session_id)

        assert recovered.actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
            IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
            IncompleteSessionRecoveryAction.PENDING_APPROVAL,
        )
        assert len(terminal_records) == 1
        assert terminal_records[0].event.payload == {
            "model_step_id": checkpoint_before["pending_tool_round"]["model_step_id"],
            "model_attempt_id": checkpoint_before["pending_tool_round"]["model_attempt_id"],
            "tool_round_id": checkpoint_before["pending_tool_round"]["tool_round_id"],
            "interruption_type": "runtime_interrupted",
            "reason": "terminal_event_evidence_repaired",
            "recovered": True,
        }
        assert tool.calls == []
        assert checkpoint_after is not None
        assert (
            checkpoint_after["pending_tool_approval"]["tool_round_id"]
            == (checkpoint_before["pending_tool_round"]["tool_round_id"])
        )
        assert (
            checkpoint_after["pending_tool_round"]["tool_round_id"]
            == (checkpoint_before["pending_tool_round"]["tool_round_id"])
        )
        assert checkpoint_after["pending_tool_round"]["policy_state"] == "planned"
        assert "incomplete_session_recovery_claim" not in checkpoint_after

    asyncio.run(scenario())


def test_cayu_app_repairs_resume_failure_before_lifecycle_boundary() -> None:
    class CrashBeforeResumeFailureEventStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_transcript_load = True
            self.fail_terminal_append = True

        async def load_transcript(self, session_id: str) -> list[Message]:
            if self.fail_transcript_load:
                raise RuntimeError("transcript unavailable")
            return await super().load_transcript(session_id)

        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type == EventType.SESSION_FAILED and self.fail_terminal_append:
                self.fail_terminal_append = False
                raise ConnectionError("worker crashed before failure evidence committed")
            await super().append_event(session_id, event)

    async def scenario() -> None:
        store = CrashBeforeResumeFailureEventStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_resume_failed_before_boundary"
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "start")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)
        previous_terminal = Event(
            type=EventType.SESSION_COMPLETED,
            session_id=session_id,
        )
        await store.append_event(session_id, previous_terminal)

        with pytest.raises(
            ConnectionError,
            match="worker crashed before failure evidence committed",
        ):
            await collect_resume_events(
                app,
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                ),
            )

        failed = await store.load(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        assert failed is not None
        assert failed.status == SessionStatus.FAILED
        assert checkpoint is not None
        assert checkpoint["session_run_operation"]["run_epoch"] == failed.run_epoch - 1

        repaired = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        terminal_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(
                    EventType.SESSION_COMPLETED,
                    EventType.SESSION_FAILED,
                ),
            )
        )
        assert repaired.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
        assert [record.event.type for record in terminal_records] == [
            EventType.SESSION_COMPLETED,
            EventType.SESSION_FAILED,
        ]
        assert repaired.events == (
            app.project_event_record_for_exposure(terminal_records[-1]).event,
        )
        assert terminal_records[-1].event.payload["terminal_evidence_repaired"] is True
        assert (
            str(UUID(terminal_records[-1].event.payload["session_run_operation_id"]))
            == terminal_records[-1].event.payload["session_run_operation_id"]
        )
        assert checkpoint_without_active_invocation_profile(
            await store.load_checkpoint(session_id)
        ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}

    asyncio.run(scenario())


def test_cayu_app_resume_reconciles_terminal_event_before_replacing_run_marker() -> None:
    class LostRunMarkerCleanupAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed_cleanup = False

        async def transform_checkpoint(self, session_id: str, checkpoint_transform) -> None:
            def fail_first_run_marker_cleanup(current_session, checkpoint):
                transformed = checkpoint_transform(current_session, checkpoint)
                if (
                    not self.failed_cleanup
                    and checkpoint is not None
                    and "session_run_operation" in checkpoint
                    and transformed is not None
                    and "session_run_operation" not in transformed
                ):
                    self.failed_cleanup = True
                    raise ConnectionError("run marker cleanup acknowledgement lost")
                return transformed

            await super().transform_checkpoint(
                session_id,
                fail_first_run_marker_cleanup,
            )

    async def scenario() -> None:
        store = LostRunMarkerCleanupAcknowledgementStore()
        provider = FakeProvider(
            [
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_resume_event_before_marker_cleanup"
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "start")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)
        await store.append_event(
            session_id,
            Event(
                type=EventType.SESSION_COMPLETED,
                session_id=session_id,
            ),
        )

        resumed = await collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )

        checkpoint = await store.load_checkpoint(session_id)
        records_before = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_COMPLETED,
            )
        )
        assert store.failed_cleanup is True
        assert resumed[-1].type == EventType.SESSION_COMPLETED
        current = await store.load(session_id)
        assert current is not None
        assert current.status == SessionStatus.COMPLETED
        assert checkpoint is not None
        assert "session_run_operation" in checkpoint
        assert len(records_before) == 2

        resumed_again = await collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue again")],
            ),
        )
        records_after = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_COMPLETED,
            )
        )
        assert resumed_again[-1].type == EventType.SESSION_COMPLETED
        assert len(records_after) == 3
        assert [record.event.id for record in records_after[:2]] == [
            record.event.id for record in records_before
        ]
        assert_only_model_step_publication_checkpoint(await store.load_checkpoint(session_id))

    asyncio.run(scenario())


def test_cayu_app_resume_reconciles_terminal_event_after_side_effect_claim_ack_loss() -> None:
    class LostTerminalSideEffectClaimAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.failures = 0
            self.terminal_event_ids: set[tuple[str, str]] = set()

        async def append_event(self, session_id: str, event: Event) -> None:
            await super().append_event(session_id, event)
            if event.type in {
                EventType.SESSION_COMPLETED,
                EventType.SESSION_FAILED,
                EventType.SESSION_INTERRUPTED,
            }:
                self.terminal_event_ids.add((session_id, event.id))

        async def claim_persisted_event_side_effect(
            self,
            *,
            session_id: str,
            event_id: str,
        ):
            if (
                self.armed
                and (session_id, event_id) in self.terminal_event_ids
                and self.failures < 2
            ):
                self.failures += 1
                raise ConnectionError("terminal side-effect claim acknowledgement lost")
            return await super().claim_persisted_event_side_effect(
                session_id=session_id,
                event_id=event_id,
            )

    async def scenario() -> None:
        store = LostTerminalSideEffectClaimAcknowledgementStore()
        provider = FakeProvider(
            [
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_resume_terminal_side_effect_claim_ack_loss"

        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "start")],
            ),
        )
        store.armed = True

        resumed = await collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )

        current = await store.load(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        terminal_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(
                    EventType.SESSION_COMPLETED,
                    EventType.SESSION_FAILED,
                    EventType.SESSION_INTERRUPTED,
                ),
            )
        )
        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        store.armed = False
        recovered_side_effects = await app.recover_persisted_event_side_effects()

        assert resumed[-1].type == EventType.SESSION_COMPLETED
        assert current is not None
        assert current.status == SessionStatus.COMPLETED
        assert store.failures == 1
        assert [record.event.type for record in terminal_records] == [
            EventType.SESSION_COMPLETED,
            EventType.SESSION_COMPLETED,
        ]
        assert checkpoint is not None
        assert "session_run_operation" not in checkpoint
        assert recovery.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
        assert recovered_side_effects == [resumed[-1]]

    asyncio.run(scenario())


def test_cayu_app_terminal_evidence_repair_preserves_pending_interrupt_identity() -> None:
    class ResumeThenBlockProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.calls = 0
            self.second_started = asyncio.Event()

        async def stream(self, _request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            if self.calls == 1:
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            self.second_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
            yield

    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        provider = ResumeThenBlockProvider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_terminal_evidence_pending_interrupt"
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "start")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        pending_payload = {
            "reason": "deploy",
            "metadata": {"worker": "old"},
            "interruption_type": "operator_requested",
            "interruption_request_id": "interrupt-before-crash",
        }
        await store.checkpoint(
            session_id,
            {"pending_session_interrupt": pending_payload},
        )

        repaired = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        checkpoint = await store.load_checkpoint(session_id)
        terminal_records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_INTERRUPTED,
            )
        )
        assert len(terminal_records) == 1
        assert terminal_records[0].event.payload == pending_payload
        assert repaired.events[0].payload == {
            **pending_payload,
            "interruption_request_id": PRIVATE_EVENT_AUTHORITY,
        }
        assert checkpoint == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}

        resumed = await collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
        assert resumed[-1].type == EventType.SESSION_COMPLETED

        second_resume = asyncio.create_task(
            collect_resume_events(
                app,
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue again")],
                ),
            )
        )
        await asyncio.wait_for(provider.second_started.wait(), timeout=10)
        interrupted = await collect_interrupt_events(
            app,
            InterruptSessionRequest(
                session_id=session_id,
                reason="new operator request",
                metadata={"worker": "new"},
            ),
        )
        second_resume_events = await asyncio.wait_for(second_resume, timeout=5)
        latest = interrupted[-1]
        assert latest.type == EventType.SESSION_INTERRUPTED
        assert second_resume_events[-1].id == latest.id
        assert latest.payload["reason"] == "new operator request"
        assert latest.payload["metadata"] == {"worker": "new"}
        assert latest.payload["interruption_request_id"] != "interrupt-before-crash"
        assert_only_model_step_publication_checkpoint(await store.load_checkpoint(session_id))

    asyncio.run(scenario())


def test_cayu_app_recover_incomplete_session_finalizes_pending_interrupt():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_stale_interrupting",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_stale_interrupting", SessionStatus.INTERRUPTING)
        await store.checkpoint(
            "sess_stale_interrupting",
            {
                "pending_session_interrupt": {
                    "reason": "deploy",
                    "metadata": {"worker": "old"},
                    "interruption_type": "operator_requested",
                }
            },
        )
        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_stale_interrupting")
        )
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return result

    result = asyncio.run(setup_and_recover())
    session = asyncio.run(store.load("sess_stale_interrupting"))
    checkpoint = asyncio.run(store.load_checkpoint("sess_stale_interrupting"))

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert checkpoint == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
    assert result.actions == (IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT,)
    assert [event.type for event in result.events] == [EventType.SESSION_INTERRUPTED]
    assert result.events[0].payload["interruption_request_id"] == PRIVATE_EVENT_AUTHORITY
    durable_result_event = _private_record_for_public_event(
        asyncio.run(store.query_events(EventQuery(session_id="sess_stale_interrupting"))),
        result.events[0],
    ).event
    assert interruption_payload_without_request_id(durable_result_event.payload) == {
        "reason": "deploy",
        "metadata": {"worker": "old"},
        "interruption_type": "operator_requested",
    }


def test_cayu_app_recover_incomplete_session_is_idempotent_after_interrupt_finalized():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def setup_and_recover_twice():
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_recover_interrupt_twice",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("sess_recover_interrupt_twice", SessionStatus.INTERRUPTING)
        await store.checkpoint(
            "sess_recover_interrupt_twice",
            {
                "pending_session_interrupt": {
                    "reason": "deploy",
                    "metadata": {},
                    "interruption_type": "operator_requested",
                }
            },
        )
        first = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_recover_interrupt_twice")
        )
        second = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="sess_recover_interrupt_twice")
        )
        events = await store.load_events("sess_recover_interrupt_twice")
        return first, second, events

    first, second, events = asyncio.run(setup_and_recover_twice())

    interrupted_events = [event for event in events if event.type == EventType.SESSION_INTERRUPTED]
    assert len(interrupted_events) == 1
    assert first.actions == (IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT,)
    assert [event.type for event in first.events] == [EventType.SESSION_INTERRUPTED]
    assert second.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
    assert second.events == ()
    assert second.status == SessionStatus.INTERRUPTED
