"""Fork creation and profile decisions compose independently of turn execution."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_recovery_composition import _run_without_owners

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.evals import ScriptedModelProvider
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import ForkSessionRequest
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.forks import session_fork_profile_relationship


def test_fork_parts_import_without_application_or_session_engine() -> None:
    _run_without_owners(
        ("cayu.applications", "cayu.runtime._session_engine"),
        """
from cayu.runtime._session_fork import SessionFork
from cayu.runtime._execution_profile_classification import ExecutionProfileClassifier
from cayu.runtime._session_tool_discovery import _require_tool_discovery_view
""",
    )


def test_profile_classification_executes_without_application_or_execution_owners() -> None:
    _run_without_owners(
        (
            "cayu.applications",
            "cayu.runtime._session_engine",
            "cayu.runtime._model_step_executor",
            "cayu.runtime._session_fork",
        ),
        """
import asyncio
from datetime import datetime, timezone
from cayu.build_provenance import (
    RuntimeBuildProvenance, RuntimeBuildProvenanceOrigin, RuntimeBuildArtifactKind,
)
from cayu.execution_profiles import ExecutionProfileDecisionKind
from cayu.messages import Message
from cayu.runtime._execution_profile_classification import ExecutionProfileClassifier
from cayu.runtime.execution_profiles import build_execution_profile_identity
from cayu.sessions.base import InMemorySessionStore, RunRequest
from cayu.sessions.records import SessionIdentity
from cayu.sessions.event_queries import EventQuery
from cayu.vaults.redaction import SecretRedactor

async def scenario():
    profile = build_execution_profile_identity(
        runtime_name="cayu", runtime_version="test", provider_name="fake", model="small",
        durable_system_prompt="prompt", direct_tools=(),
        tool_catalogue_revision="sha256:" + "a" * 64,
        runtime_build_provenance=RuntimeBuildProvenance.from_artifact_digest(
            origin=RuntimeBuildProvenanceOrigin.EXPLICIT_MANIFEST,
            artifact_kind=RuntimeBuildArtifactKind.OTHER, artifact_digest="a" * 64,
        ),
    )
    store = InMemorySessionStore()
    session = await store.create(
        RunRequest(agent_name="agent", messages=[Message.text("user", "hello")]),
        identity=SessionIdentity(provider_name="fake", model="small", execution_profile=profile),
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    classifier = ExecutionProfileClassifier(
        policy=None, policy_identity=None, redactor=SecretRedactor(), clock=lambda: now,
    )
    arguments = dict(
        session=session, expected_profile=profile, candidate_profile=profile,
        changed_component_classes=(), intent=None, adoption_request_fingerprint=None,
        target_changed=False, target_provider_name="fake", target_model="small",
    )
    exact = await classifier.classify(**arguments)
    assert exact.kind is ExecutionProfileDecisionKind.EXACT_REUSE
    assert exact.event.timestamp == now
    assert classifier.policy_identity is None
    rejected = await classifier.classify(**arguments, force_authority_review=True)
    assert rejected.kind is ExecutionProfileDecisionKind.REJECTED
    assert await store.query_events(
        EventQuery(session_id=session.id)
    ) == []
asyncio.run(scenario())
""",
    )


@pytest.mark.parametrize("exact_source", [False, True], ids=["ordinary", "exact-source"])
def test_public_fork_and_replay_do_not_need_the_session_engine(monkeypatch, exact_source) -> None:
    async def scenario() -> None:
        app = CayuApp(enable_logging=False)
        app.register_provider(
            ScriptedModelProvider([[ModelStreamEvent.completed()]], name="fake"),
            default=True,
        )
        app.register_agent(AgentSpec(name="agent", model="small", system_prompt="original"))
        source_events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="independent-fork-source",
                    agent_name="agent",
                    messages=[Message.text("user", "hello")],
                )
            )
        ]
        assert source_events[-1].type is EventType.SESSION_COMPLETED
        source_id = source_events[-1].session_id
        # Public snapshot, creation and replay must use their composed owners.
        # Any hidden callback into turn execution now fails immediately.
        with monkeypatch.context() as patch:
            patch.delattr(app, "_session_engine")
            snapshot = await app.snapshot_fork_source(source_id) if exact_source else None
            request = ForkSessionRequest(
                source_session_id=source_id,
                session_id="independent-fork-child",
                expected_source=snapshot,
            )
            first = [event async for event in app.fork_session(request)]
            child = await app.session_store.load("independent-fork-child")
            assert child is not None
            relationship = session_fork_profile_relationship(child)
            assert relationship is not None
            assert relationship.source_session_id == source_id
            records = await app.session_store.query_events(EventQuery(session_id=child.id))
            await app.session_store.delete_session(source_id)
            replay = [event async for event in app.fork_session(request)]
            assert [event.type for event in first] == [EventType.SESSION_FORKED]
            assert [event.id for event in replay] == [event.id for event in first]
            assert await app.session_store.query_events(EventQuery(session_id=child.id)) == records

    asyncio.run(scenario())


def test_fork_snapshot_keeps_its_composed_task_admission_store() -> None:
    from cayu.tasks.memory import InMemoryTaskStore

    class RecordingTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.lookups: list[str] = []

        async def load_active_work_contract_task_for_session(self, session_id):
            self.lookups.append(session_id)
            return await super().load_active_work_contract_task_for_session(session_id)

    async def scenario() -> None:
        configured = RecordingTaskStore()
        replacement = RecordingTaskStore()
        app = CayuApp(task_store=configured, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider([[ModelStreamEvent.completed()]], name="fake"),
            default=True,
        )
        app.register_agent(AgentSpec(name="agent", model="small"))
        events = [
            event
            async for event in app.run(
                RunRequest(agent_name="agent", messages=[Message.text("user", "hello")])
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        session_id = events[-1].session_id
        configured.lookups.clear()
        # The operation retains the store it was composed with, just as creation
        # and publication do; changing an app attribute must not split admission.
        app.task_store = replacement
        await app.snapshot_fork_source(session_id)
        assert configured.lookups == [session_id]
        assert replacement.lookups == []

    asyncio.run(scenario())
