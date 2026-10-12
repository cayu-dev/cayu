"""Forks contracts."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
    run_request_with_registered_tool_ceiling,
)
from tests.runtime.runtime_environment_support import RecordingEnvironmentFactory
from tests.runtime.runtime_provider_support import FakeProvider, VersionedFakeProvider
from tests.runtime.runtime_session_support import (
    _AllowForkProfileAdoption,
    _current_fork_profile_selection,
    collect_events,
    collect_fork_events,
    collect_resume_events,
)
from tests.runtime.runtime_tool_support import EchoTool

import cayu.sessions.base as sessions_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.environments.base import Environment, EnvironmentSpec, WorkspaceInstructionsConfig
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import ExecutionProfileAdoptionRejected
from cayu.sessions import _invocation_scope as invocation_scope
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.forks import ForkSystemPromptPolicy, session_prompt_anatomy_transition
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest, RunRequest
from cayu.sessions.transcript_input import system_prompt_messages_sha256
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.exposure import ToolCapabilityCeiling
from cayu.tools.grants import TargetedToolGrant
from cayu.tools.user_input import UserInputTool
from cayu.workspaces.local import LocalWorkspace


def test_cayu_app_forks_completed_session_and_preserves_source():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("fork answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        targeted_tool_mode="call_tool",
        tools=(EchoTool(),),
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_fork_source",
                labels={"owner": "org_123", "project": "feature_a"},
                messages=[Message.text("user", "first request")],
                tool_grants=(
                    TargetedToolGrant(
                        request_id="fork-source-grant",
                        tool_id="cayu:echo",
                    ),
                ),
            ),
        )
    )
    assert len(asyncio.run(store.list_targeted_tool_grants("sess_fork_source"))) == 1

    async def make_source_checkpoint_versionless() -> None:
        checkpoint = await store.load_checkpoint("sess_fork_source")
        assert checkpoint is not None
        checkpoint.pop(CHECKPOINT_SCHEMA_VERSION_KEY)
        checkpoint["future_additive_field"] = {"kept": True}
        await store.checkpoint("sess_fork_source", checkpoint)

    asyncio.run(make_source_checkpoint_versionless())

    fork_events = asyncio.run(
        collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_fork_source",
                session_id="sess_fork_child",
                metadata={"purpose": "alternate path"},
            ),
        )
    )

    assert [event.type for event in fork_events] == [EventType.SESSION_FORKED]
    assert fork_events[0].session_id == "sess_fork_child"
    assert fork_events[0].payload["source_session_id"] == PRIVATE_EVENT_AUTHORITY
    fork = asyncio.run(store.load("sess_fork_child"))
    source = asyncio.run(store.load("sess_fork_source"))
    assert fork is not None
    assert source is not None
    assert fork.parent_session_id == "sess_fork_source"
    assert fork.status == SessionStatus.COMPLETED
    assert fork.provider_name == source.provider_name == "fake"
    assert fork.model == source.model == "fake-model"
    assert fork.labels == source.labels == {"owner": "org_123", "project": "feature_a"}
    assert sessions_module.session_user_metadata(fork.metadata) == {"purpose": "alternate path"}
    fork_checkpoint = asyncio.run(store.load_checkpoint("sess_fork_child"))
    assert fork_checkpoint is not None
    assert fork_checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
    assert fork_checkpoint["future_additive_field"] == {"kept": True}

    fork_transcript = asyncio.run(store.load_transcript("sess_fork_child"))
    source_transcript = asyncio.run(store.load_transcript("sess_fork_source"))
    assert fork_transcript == source_transcript
    assert asyncio.run(store.list_targeted_tool_grants("sess_fork_child")) == ()
    assert len(asyncio.run(store.list_targeted_tool_grants("sess_fork_source"))) == 1
    fork_reset = asyncio.run(store.load_events("sess_fork_child"))[-1]
    assert fork_reset.type is EventType.TARGETED_TOOL_GRANT_FORK_RESET
    assert fork_reset.payload["inherited_grant_count"] == 0
    assert fork_reset.payload["inherited_reference_count"] == 0

    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_fork_child",
                messages=[Message.text("user", "continue fork")],
            ),
        )
    )
    assert [message.content[0].text for message in provider.requests[1].messages] == [
        "first request",
        "first answer",
        "continue fork",
    ]
    assert [message.content[0].text for message in source_transcript] == [
        "first request",
        "first answer",
    ]


def test_cayu_app_fork_can_install_current_body_prompt_and_preserve_history(
    tmp_path: Path,
) -> None:
    database = tmp_path / "prompt-succession.sqlite"
    parent_workspace = tmp_path / "parent-workspace"
    child_workspace = tmp_path / "child-workspace"
    parent_workspace.mkdir()
    child_workspace.mkdir()
    (parent_workspace / "AGENTS.md").write_text("parent workspace instructions")
    (child_workspace / "AGENTS.md").write_text("child workspace instructions")
    source_store = SQLiteSessionStore(database)
    parent_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("durable parent memory"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    parent_app = CayuApp(session_store=source_store, enable_logging=False)
    parent_app.register_provider(parent_provider, default=True)
    parent_app.register_environment(
        Environment(
            EnvironmentSpec(name="body"),
            workspace=LocalWorkspace(parent_workspace),
            workspace_instructions=WorkspaceInstructionsConfig(paths=("AGENTS.md",)),
        ),
        default=True,
    )
    parent_app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="parent body prompt",
        )
    )
    asyncio.run(
        collect_events(
            parent_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_parent",
                environment_name="body",
                messages=[Message.text("user", "remember this generation")],
            ),
        )
    )

    child_provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("child continued"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    child_store = SQLiteSessionStore(database)
    child_app = CayuApp(
        session_store=child_store,
        execution_profile_policy=_AllowForkProfileAdoption(),
        enable_logging=False,
    )
    child_app.register_provider(child_provider, default=True)
    child_app.register_environment(
        Environment(
            EnvironmentSpec(name="body"),
            workspace=LocalWorkspace(child_workspace),
            workspace_instructions=WorkspaceInstructionsConfig(paths=("AGENTS.md",)),
        ),
        default=True,
    )
    child_app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt="child body prompt",
        ),
        tools=[EchoTool()],
    )
    fork_request = ForkSessionRequest(
        source_session_id="sess_prompt_parent",
        session_id="sess_prompt_child",
        agent_name="assistant",
        copy_checkpoint=False,
        system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
        **_current_fork_profile_selection("prompt-parent-to-child"),
    )

    fork_query_cutoff = datetime.now(UTC)
    first_events = asyncio.run(collect_fork_events(child_app, fork_request))
    replay_events = asyncio.run(collect_fork_events(child_app, fork_request))
    source = asyncio.run(source_store.load("sess_prompt_parent"))
    child = asyncio.run(child_store.load("sess_prompt_child"))
    source_transcript = asyncio.run(source_store.load_transcript("sess_prompt_parent"))
    child_transcript = asyncio.run(child_store.load_transcript("sess_prompt_child"))

    assert child is not None and source is not None
    assert [event.id for event in replay_events] == [event.id for event in first_events]
    assert [event.type for event in asyncio.run(child_store.load_events(child.id))] == [
        EventType.SESSION_EXECUTION_PROFILE_DECIDED,
        EventType.SESSION_FORKED,
        EventType.TARGETED_TOOL_GRANT_FORK_RESET,
    ]
    receipt = session_prompt_anatomy_transition(child)
    assert receipt is not None
    assert receipt.source_prompt_sha256 == system_prompt_messages_sha256(source_transcript)
    assert receipt.child_prompt_sha256 == asyncio.run(
        child_app.current_prompt_anatomy_sha256(
            agent_name="assistant",
            environment_name="body",
        )
    )
    assert receipt.source_session_id == source.id
    assert receipt.descendant_session_id == child.id
    assert receipt.source_system_prompt_retained is False
    assert receipt.source_prompt_sha256 != receipt.child_prompt_sha256
    assert receipt.copy_checkpoint is False
    assert receipt.source_provider_name == receipt.provider_name == "fake"
    assert receipt.source_model == receipt.model == "fake-model"
    assert receipt.model_target_changed is False
    assert receipt.portability_preflight == "context_messages_validated"
    assert receipt.portability_verified is True

    invalid_digest_metadata = json.loads(json.dumps(child.metadata))
    invalid_digest_metadata["cayu:prompt_anatomy_transition"]["transition_id"] = "0" * 64
    with pytest.raises(ValueError, match="prompt-anatomy transition metadata is malformed"):
        session_prompt_anatomy_transition(
            child.model_copy(update={"metadata": invalid_digest_metadata})
        )

    invalid_policy_metadata = json.loads(json.dumps(child.metadata))
    invalid_policy_receipt = invalid_policy_metadata["cayu:prompt_anatomy_transition"]
    invalid_policy_receipt["source_system_prompt_retained"] = True
    invalid_policy_receipt["copy_checkpoint"] = True
    with pytest.raises(ValueError, match="prompt-anatomy transition metadata is malformed"):
        session_prompt_anatomy_transition(
            child.model_copy(update={"metadata": invalid_policy_metadata})
        )
    recent_fork_records = asyncio.run(
        child_store.query_events(
            EventQuery(
                session_id=child.id,
                event_types=(EventType.SESSION_FORKED,),
                since=fork_query_cutoff,
            )
        )
    )
    assert [record.event.type for record in recent_fork_records] == [EventType.SESSION_FORKED]
    assert recent_fork_records[0].event.timestamp >= fork_query_cutoff
    source_text = [message.content[0].text for message in source_transcript]
    child_text = [message.content[0].text for message in child_transcript]
    assert "parent body prompt" in source_text[0]
    assert "parent workspace instructions" in source_text[0]
    assert source_text[1:] == ["remember this generation", "durable parent memory"]
    assert "child body prompt" in child_text[0]
    assert "child workspace instructions" in child_text[0]
    assert "parent workspace instructions" not in child_text[0]
    assert child_text[1:] == ["remember this generation", "durable parent memory"]

    asyncio.run(
        collect_resume_events(
            child_app,
            ResumeRequest(
                session_id=child.id,
                messages=[Message.text("user", "continue as the child")],
            ),
        )
    )
    request_text = [message.content[0].text for message in child_provider.requests[0].messages]
    assert "child body prompt" in request_text[0]
    assert "child workspace instructions" in request_text[0]
    assert request_text[1:] == [
        "remember this generation",
        "durable parent memory",
        "continue as the child",
    ]
    assert child.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=())
    assert child_provider.requests[0].tools == []

    child_event_ids_before_parent_delete = [
        event.id for event in asyncio.run(child_store.load_events(child.id))
    ]
    asyncio.run(source_store.delete_session(source.id))
    detached_child = asyncio.run(child_store.load(child.id))
    assert detached_child is not None
    assert detached_child.parent_session_id is None
    detached_replay = asyncio.run(collect_fork_events(child_app, fork_request))
    assert [event.id for event in detached_replay] == [event.id for event in first_events]
    assert [event.id for event in asyncio.run(child_store.load_events(child.id))] == (
        child_event_ids_before_parent_delete
    )
    with pytest.raises(RuntimeError, match="conflicts with the exact profiled request"):
        asyncio.run(
            collect_fork_events(
                child_app,
                fork_request.model_copy(update={"metadata": {"request": "different"}}),
            )
        )


def test_prompt_anatomy_fork_recovers_after_descendant_commit_before_receipt() -> None:
    class CrashAfterForkStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        crash_after_commit = True

        async def create_profiled_fork(self, *args, **kwargs):
            created = await super().create_profiled_fork(*args, **kwargs)
            if self.crash_after_commit:
                self.crash_after_commit = False
                raise SystemExit("process died after descendant commit")
            return created

    store = CrashAfterForkStore()
    parent_app = CayuApp(session_store=store, enable_logging=False)
    parent_app.register_provider(
        FakeProvider(
            [
                ModelStreamEvent.text_delta("remembered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    parent_app.register_agent(
        AgentSpec(name="assistant", model="fake-model", system_prompt="parent prompt")
    )
    asyncio.run(
        collect_events(
            parent_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_crash_parent",
                messages=[Message.text("user", "start")],
            ),
        )
    )
    request = ForkSessionRequest(
        source_session_id="sess_prompt_crash_parent",
        session_id="sess_prompt_crash_child",
        environment_name="body",
        copy_checkpoint=False,
        system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
        **_current_fork_profile_selection("prompt-crash-child"),
    )

    crashing_app = CayuApp(
        session_store=store,
        execution_profile_policy=_AllowForkProfileAdoption(),
        enable_logging=False,
    )
    crashing_app.register_provider(FakeProvider([]), default=True)
    crashing_app.register_environment(Environment(EnvironmentSpec(name="body")), default=True)
    crashing_app.register_agent(
        AgentSpec(name="assistant", model="fake-model", system_prompt="child prompt")
    )
    with pytest.raises(SystemExit, match="process died"):
        asyncio.run(collect_fork_events(crashing_app, request))
    assert asyncio.run(store.load("sess_prompt_crash_child")) is not None
    assert [event.type for event in asyncio.run(store.load_events("sess_prompt_crash_child"))] == [
        EventType.SESSION_EXECUTION_PROFILE_DECIDED,
        EventType.SESSION_FORKED,
        EventType.TARGETED_TOOL_GRANT_FORK_RESET,
    ]

    restarted_app = CayuApp(
        session_store=store,
        execution_profile_policy=_AllowForkProfileAdoption(),
        enable_logging=False,
    )
    restarted_app.register_provider(FakeProvider([]), default=True)
    restarted_app.register_environment(Environment(EnvironmentSpec(name="body")), default=True)
    restarted_app.register_agent(
        AgentSpec(name="assistant", model="fake-model", system_prompt="child prompt")
    )
    recovered = asyncio.run(collect_fork_events(restarted_app, request))

    assert [event.type for event in recovered] == [EventType.SESSION_FORKED]
    assert len(asyncio.run(store.load_events("sess_prompt_crash_child"))) == 3
    completed_checkpoint = asyncio.run(store.load_checkpoint("sess_prompt_crash_parent"))
    assert completed_checkpoint is not None
    completed_intents = completed_checkpoint["prompt_anatomy_transition_intents"]
    assert next(iter(completed_intents["records"].values()))["status"] == "completed"


def test_prompt_anatomy_fork_exact_retry_survives_source_advance_after_process_death(
    tmp_path: Path,
) -> None:
    class CrashAfterForkStore(SQLiteSessionStore):
        invocation_lifecycle_command_version = 1

        async def create_profiled_fork(self, *args, **kwargs):
            await super().create_profiled_fork(*args, **kwargs)
            raise SystemExit("process died after descendant commit")

    def register_body(
        app: CayuApp,
        provider: FakeProvider,
        *,
        system_prompt: str,
    ) -> None:
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="body",
                    execution_profile_identity=ExecutionProfileBehaviorIdentity(
                        name="tests:runtime:prompt-anatomy-body",
                        behavior_version="1",
                        implementation_version="1",
                    ),
                )
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model", system_prompt=system_prompt)
        )

    async def scenario() -> None:
        database = tmp_path / "prompt-postcommit-recovery.sqlite"
        store = CrashAfterForkStore(database)
        parent_app = CayuApp(session_store=store, enable_logging=False)
        register_body(
            parent_app,
            VersionedFakeProvider(
                [
                    ModelStreamEvent.text_delta("first memory"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ),
            system_prompt="parent prompt",
        )
        await collect_events(
            parent_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_advanced_parent",
                environment_name="body",
                messages=[Message.text("user", "first turn")],
            ),
        )
        request = ForkSessionRequest(
            source_session_id="sess_prompt_advanced_parent",
            session_id="sess_prompt_advanced_child",
            agent_name="assistant",
            environment_name="body",
            copy_checkpoint=False,
            system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
            **_current_fork_profile_selection("prompt-advanced-child"),
        )

        crashing_app = CayuApp(
            session_store=store,
            execution_profile_policy=_AllowForkProfileAdoption(),
            enable_logging=False,
        )
        register_body(crashing_app, VersionedFakeProvider([]), system_prompt="child prompt")
        with pytest.raises(SystemExit, match="descendant commit"):
            await collect_fork_events(crashing_app, request)
        assert await store.load("sess_prompt_advanced_child") is not None
        assert [event.type for event in await store.load_events("sess_prompt_advanced_child")] == [
            EventType.SESSION_EXECUTION_PROFILE_DECIDED,
            EventType.SESSION_FORKED,
            EventType.TARGETED_TOOL_GRANT_FORK_RESET,
        ]
        await store.close()

        advanced_store = SQLiteSessionStore(database)
        advanced_app = CayuApp(session_store=advanced_store, enable_logging=False)
        register_body(
            advanced_app,
            VersionedFakeProvider(
                [
                    ModelStreamEvent.text_delta("second memory"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ),
            system_prompt="parent prompt",
        )
        await collect_resume_events(
            advanced_app,
            ResumeRequest(
                session_id="sess_prompt_advanced_parent",
                messages=[Message.text("user", "second turn")],
            ),
        )
        await advanced_store.close()

        recovered_store = SQLiteSessionStore(database)
        recovered_app = CayuApp(
            session_store=recovered_store,
            execution_profile_policy=_AllowForkProfileAdoption(),
            enable_logging=False,
        )
        register_body(recovered_app, VersionedFakeProvider([]), system_prompt="child prompt")
        recovered = await collect_fork_events(recovered_app, request)

        child = await recovered_store.load("sess_prompt_advanced_child")
        assert child is not None
        assert [event.type for event in recovered] == [EventType.SESSION_FORKED]
        assert len(await recovered_store.load_events(child.id)) == 3
        assert session_prompt_anatomy_transition(child) is not None
        await recovered_store.close()

    asyncio.run(scenario())


def test_prompt_anatomy_fork_recovers_durable_intent_after_process_death(
    tmp_path: Path,
) -> None:
    class CrashBeforeForkStore(SQLiteSessionStore):
        invocation_lifecycle_command_version = 1
        crash_before_commit = True

        async def create_profiled_fork(self, *args, **kwargs):
            if self.crash_before_commit:
                self.crash_before_commit = False
                raise SystemExit("process died before descendant commit")
            return await super().create_profiled_fork(*args, **kwargs)

    async def scenario() -> None:
        database = tmp_path / "prompt-intent-recovery.sqlite"
        store = CrashBeforeForkStore(database)
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_precommit_parent",
                messages=[],
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.append_transcript_messages(
            source.id,
            [Message.text("system", "parent prompt"), Message.text("user", "memory")],
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        request = ForkSessionRequest(
            source_session_id="sess_prompt_precommit_parent",
            session_id="sess_prompt_precommit_child",
            environment_name="body",
            copy_checkpoint=False,
            system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
            **_current_fork_profile_selection("prompt-precommit-child"),
        )

        app = CayuApp(
            session_store=store,
            execution_profile_policy=_AllowForkProfileAdoption(),
            enable_logging=False,
        )
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment(Environment(EnvironmentSpec(name="body")), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model", system_prompt="child prompt")
        )
        with pytest.raises(SystemExit, match="before descendant"):
            await collect_fork_events(app, request)
        assert await store.load("sess_prompt_precommit_child") is None
        checkpoint = await store.load_checkpoint(source.id)
        assert checkpoint is not None
        intents = checkpoint["prompt_anatomy_transition_intents"]
        assert len(intents["records"]) == 1
        assert next(iter(intents["records"].values()))["status"] == "prepared"
        await store.close()

        reopened = SQLiteSessionStore(database)
        restarted_app = CayuApp(
            session_store=reopened,
            execution_profile_policy=_AllowForkProfileAdoption(),
            enable_logging=False,
        )
        restarted_app.register_provider(FakeProvider([]), default=True)
        restarted_app.register_environment(Environment(EnvironmentSpec(name="body")), default=True)
        restarted_app.register_agent(
            AgentSpec(name="assistant", model="fake-model", system_prompt="child prompt")
        )
        with pytest.raises(RuntimeError, match="prepared prompt-anatomy succession intent"):
            await collect_resume_events(
                restarted_app,
                ResumeRequest(
                    session_id=source.id,
                    messages=[Message.text("user", "advance before retry")],
                ),
            )
        fenced_source = await reopened.load(source.id)
        assert fenced_source is not None
        assert fenced_source.status is SessionStatus.COMPLETED
        assert [
            message.content[0].text for message in await reopened.load_transcript(source.id)
        ] == ["parent prompt", "memory"]

        recovered = await collect_fork_events(restarted_app, request)
        child = await reopened.load("sess_prompt_precommit_child")
        completed_checkpoint = await reopened.load_checkpoint(source.id)
        assert completed_checkpoint is not None
        completed_intents = completed_checkpoint["prompt_anatomy_transition_intents"]
        assert next(iter(completed_intents["records"].values()))["status"] == "completed"
        assert [event.type for event in recovered] == [EventType.SESSION_FORKED]
        assert child is not None and session_prompt_anatomy_transition(child) is not None
        await reopened.close()

    asyncio.run(scenario())


def test_prompt_anatomy_fork_requires_exact_intent_inside_atomic_create() -> None:
    class IntentErasingStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def create_profiled_fork(self, *args, **kwargs):
            await self.checkpoint(kwargs["source_session_id"], {})
            return await super().create_profiled_fork(*args, **kwargs)

    store = IntentErasingStore()

    async def scenario() -> None:
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_intent_parent",
                messages=[],
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.append_transcript_messages(
            source.id,
            [Message.text("system", "parent prompt"), Message.text("user", "memory")],
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        app = CayuApp(
            session_store=store,
            execution_profile_policy=_AllowForkProfileAdoption(),
            enable_logging=False,
        )
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment(Environment(EnvironmentSpec(name="body")), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model", system_prompt="child prompt")
        )

        with pytest.raises(RuntimeError, match="intent disappeared"):
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id="sess_prompt_intent_child",
                    environment_name="body",
                    copy_checkpoint=False,
                    system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
                    **_current_fork_profile_selection("prompt-intent-child"),
                ),
            )

        assert await store.load("sess_prompt_intent_child") is None

    asyncio.run(scenario())


def test_prompt_anatomy_fork_rejects_checkpoint_copy() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> None:
        source = await store.create(
            run_request_with_registered_tool_ceiling(app, agent_name="assistant", messages=[]),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        with pytest.raises(ValueError, match="cannot copy checkpoint"):
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id="sess_prompt_checkpoint_child",
                    system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
                    **_current_fork_profile_selection("prompt-checkpoint-child"),
                ),
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("factory_backed", [False, True])
def test_prompt_anatomy_fork_requires_concrete_registered_environment(
    factory_backed: bool,
) -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(FakeProvider([]), default=True)
    factory: RecordingEnvironmentFactory | None = None
    environment_name: str | None = None
    if factory_backed:
        environment_name = "body"
        factory = RecordingEnvironmentFactory(Environment(EnvironmentSpec(name="body")))
        app.register_environment_factory(
            EnvironmentSpec(name="body"),
            factory,
            default=True,
        )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model", system_prompt="current prompt")
    )

    async def scenario() -> None:
        source = await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="concrete registered environment"):
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id=f"sess_prompt_environment_child_{factory_backed}",
                    environment_name=environment_name,
                    copy_checkpoint=False,
                    system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
                    **_current_fork_profile_selection(f"prompt-environment-child-{factory_backed}"),
                ),
            )

        assert await store.load(f"sess_prompt_environment_child_{factory_backed}") is None
        checkpoint = await store.load_checkpoint(source.id)
        assert checkpoint is None or "prompt_anatomy_transition_intents" not in checkpoint

    asyncio.run(scenario())
    if factory is not None:
        assert factory.requests == []


@pytest.mark.parametrize("status", [SessionStatus.RUNNING, SessionStatus.INTERRUPTING])
def test_prompt_anatomy_fork_rejects_live_source_before_effects(
    status: SessionStatus,
) -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(FakeProvider([]), default=True)
    app.register_environment(Environment(EnvironmentSpec(name="body")), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model", system_prompt="current prompt")
    )

    async def scenario() -> None:
        source = await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.update_status(source.id, SessionStatus.RUNNING)
        if status is SessionStatus.INTERRUPTING:
            await store.update_status(source.id, SessionStatus.INTERRUPTING)

        with pytest.raises(ValueError, match="Only completed, failed, or interrupted"):
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id=f"sess_prompt_live_child_{status.value}",
                    environment_name="body",
                    copy_checkpoint=False,
                    system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
                    **_current_fork_profile_selection(f"prompt-live-child-{status.value}"),
                ),
            )

        assert await store.load(f"sess_prompt_live_child_{status.value}") is None
        checkpoint = await store.load_checkpoint(source.id)
        assert checkpoint is None or "prompt_anatomy_transition_intents" not in checkpoint

    asyncio.run(scenario())


def test_prompt_anatomy_fork_rejects_active_recovery_before_effects() -> None:
    store_now = datetime(2026, 1, 1, tzinfo=UTC)
    worker_now = datetime(2100, 1, 1, tzinfo=UTC)
    store = InMemorySessionStore(ownership_clock=lambda: store_now)
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        clock=lambda: worker_now,
        execution_profile_policy=_AllowForkProfileAdoption(),
    )
    app.register_provider(FakeProvider([]), default=True)
    app.register_environment(Environment(EnvironmentSpec(name="body")), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model", system_prompt="current prompt")
    )

    async def scenario() -> None:
        source = await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        checkpoint = {
            "incomplete_session_recovery_claim": {
                "version": 1,
                "claim_id": "active-recovery-owner",
                "claimed_at": store_now.isoformat(),
                "claim_expires_at": (store_now + timedelta(minutes=5)).isoformat(),
            }
        }
        await store.checkpoint(source.id, checkpoint)

        with pytest.raises(RuntimeError, match="active incomplete-session recovery"):
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id="sess_prompt_active_recovery_child",
                    environment_name="body",
                    copy_checkpoint=False,
                    system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
                    **_current_fork_profile_selection("prompt-active-recovery-child"),
                ),
            )

        assert await store.load("sess_prompt_active_recovery_child") is None
        assert await store.load_checkpoint(source.id) == checkpoint

    asyncio.run(scenario())


def test_prompt_anatomy_fork_cannot_expire_session_operation_with_worker_clock() -> None:
    store_now = datetime(2026, 1, 1, tzinfo=UTC)
    worker_now = datetime(2100, 1, 1, tzinfo=UTC)
    store = InMemorySessionStore(ownership_clock=lambda: store_now)
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        clock=lambda: worker_now,
        execution_profile_policy=_AllowForkProfileAdoption(),
    )
    app.register_provider(FakeProvider([]), default=True)
    app.register_environment(Environment(EnvironmentSpec(name="body")), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model", system_prompt="current prompt")
    )

    async def scenario() -> None:
        source = await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        checkpoint = {
            "session_operations": {
                "version": 1,
                "active_operation_id": "active-fork-operation",
                "records": {
                    "active-fork-operation-key": {
                        "operation_id": "active-fork-operation",
                        "status": "running",
                        "claim_expires_at": (store_now + timedelta(minutes=5)).isoformat(),
                    }
                },
            }
        }
        await store.checkpoint(source.id, checkpoint)

        with pytest.raises(RuntimeError, match="active durable operation"):
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id="sess_prompt_active_operation_child",
                    environment_name="body",
                    copy_checkpoint=False,
                    system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
                    **_current_fork_profile_selection("prompt-active-operation-child"),
                ),
            )

        assert await store.load("sess_prompt_active_operation_child") is None
        assert await store.load_checkpoint(source.id) == checkpoint

    asyncio.run(scenario())


def test_prompt_anatomy_fork_rejects_pending_user_input_before_effects() -> None:
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_ask",
                name="ask_user",
                arguments={"question": "Proceed?"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        execution_profile_policy=_AllowForkProfileAdoption(),
    )
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="body")), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model", system_prompt="current prompt"),
        tools=[UserInputTool()],
    )

    async def scenario() -> None:
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_pending_input_source",
                environment_name="body",
                messages=[Message.text("user", "ask me")],
            ),
        )
        await store.update_status("sess_prompt_pending_input_source", SessionStatus.FAILED)
        checkpoint = await store.load_checkpoint("sess_prompt_pending_input_source")
        assert checkpoint is not None and "pending_user_input" in checkpoint

        with pytest.raises(RuntimeError, match="awaiting user input cannot be forked"):
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id="sess_prompt_pending_input_source",
                    session_id="sess_prompt_pending_input_child",
                    environment_name="body",
                    copy_checkpoint=False,
                    system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
                    **_current_fork_profile_selection("prompt-pending-input-child"),
                ),
            )

        assert await store.load("sess_prompt_pending_input_child") is None
        assert await store.load_checkpoint("sess_prompt_pending_input_source") == checkpoint

    asyncio.run(scenario())


@pytest.mark.parametrize("copy_checkpoint", [False, True])
def test_cayu_app_fences_expired_recovery_owner_before_fork(
    copy_checkpoint: bool,
) -> None:
    now = datetime(2026, 1, 1, 1, tzinfo=UTC)

    class InterleavedTakeoverStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.takeover_started = asyncio.Event()
            self.allow_takeover = asyncio.Event()

        async def fence_run_and_transform_checkpoint(self, *args, **kwargs):
            self.takeover_started.set()
            await self.allow_takeover.wait()
            return await super().fence_run_and_transform_checkpoint(*args, **kwargs)

    store = InterleavedTakeoverStore()
    app = CayuApp(session_store=store, enable_logging=False, clock=lambda: now)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> None:
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id="sess_expired_recovery_claim_source",
                messages=[Message.text("user", "fork me")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        session_id = "sess_expired_recovery_claim_source"
        child_id = f"sess_expired_recovery_claim_child_{copy_checkpoint}"
        await store.checkpoint(session_id, {"custom_state": {"value": 1}})
        await store.update_status(session_id, SessionStatus.COMPLETED)
        stale_owner_ready = asyncio.Event()
        release_stale_owner = asyncio.Event()

        async def stale_recovery_owner() -> None:
            fenced = await store.fence_stalled_run(
                session_id,
                statuses={SessionStatus.COMPLETED},
                inactive_for_seconds=0,
            )
            assert fenced is not None

            def add_expired_claim(_session, checkpoint):
                updated = {} if checkpoint is None else dict(checkpoint)
                updated["incomplete_session_recovery_claim"] = {
                    "version": 1,
                    "claim_id": "stale-recovery-owner",
                    "claimed_at": (now - timedelta(minutes=10)).isoformat(),
                    "claim_expires_at": (now - timedelta(minutes=5)).isoformat(),
                }
                return updated

            await store.transform_checkpoint(session_id, add_expired_claim)
            stale_owner_ready.set()
            await store.takeover_started.wait()
            await store.update_metadata(
                session_id,
                {"stale_owner_write_before_atomic_takeover": True},
            )
            store.allow_takeover.set()
            await release_stale_owner.wait()
            try:
                with pytest.raises(SessionRunFenced):
                    await store.update_metadata(session_id, {"stale_owner_write": True})
            finally:
                await store.release_run_fence(session_id)

        stale_owner_task = asyncio.create_task(stale_recovery_owner())
        await asyncio.wait_for(stale_owner_ready.wait(), timeout=10)
        owned = await store.load(session_id)
        assert owned is not None
        stale_epoch = owned.run_epoch

        try:
            with pytest.raises(
                ValueError,
                match="fenced an expired incomplete-session recovery owner",
            ):
                await collect_fork_events(
                    app,
                    ForkSessionRequest(
                        source_session_id=session_id,
                        session_id=child_id,
                        copy_checkpoint=copy_checkpoint,
                    ),
                )

            assert await store.load(child_id) is None
            after_takeover = await store.load(session_id)
            assert after_takeover is not None
            assert after_takeover.run_epoch > stale_epoch
            assert sessions_module.session_user_metadata(after_takeover.metadata) == {
                "stale_owner_write_before_atomic_takeover": True
            }
            source_checkpoint = await store.load_checkpoint(session_id)
            assert source_checkpoint == {
                CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
                "custom_state": {"value": 1},
            }
        finally:
            store.allow_takeover.set()
            release_stale_owner.set()
            await asyncio.wait_for(stale_owner_task, timeout=5)

        await collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id=session_id,
                session_id=child_id,
                copy_checkpoint=copy_checkpoint,
            ),
        )
        child_checkpoint = await store.load_checkpoint(child_id)
        expected_checkpoint = (
            {
                CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
                "custom_state": {"value": 1},
            }
            if copy_checkpoint
            else None
        )
        assert child_checkpoint == expected_checkpoint

    asyncio.run(scenario())


def test_cayu_app_cleans_ambiguous_expired_recovery_takeover_before_retry() -> None:
    now = datetime(2026, 1, 1, 1, tzinfo=UTC)

    class LostTakeoverAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.lose_next_takeover_acknowledgement = False

        async def fence_run_and_transform_checkpoint(self, *args, **kwargs):
            fenced = await super().fence_run_and_transform_checkpoint(*args, **kwargs)
            if self.lose_next_takeover_acknowledgement:
                self.lose_next_takeover_acknowledgement = False
                raise RuntimeError("expired recovery takeover acknowledgement lost")
            return fenced

    store = LostTakeoverAcknowledgementStore()
    app = CayuApp(session_store=store, enable_logging=False, clock=lambda: now)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> None:
        session_id = "sess_ambiguous_expired_recovery_takeover"
        child_id = "sess_ambiguous_expired_recovery_takeover_child"
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "fork me")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        completed = await store.update_status(session_id, SessionStatus.COMPLETED)

        def install_expired_claim(_session, checkpoint):
            updated = {} if checkpoint is None else dict(checkpoint)
            updated["custom_state"] = {"value": 1}
            updated["incomplete_session_recovery_claim"] = {
                "version": 1,
                "claim_id": "expired-recovery-owner",
                "claimed_at": (now - timedelta(minutes=10)).isoformat(),
                "claim_expires_at": (now - timedelta(minutes=5)).isoformat(),
            }
            return updated

        await store.transform_checkpoint(session_id, install_expired_claim)
        store.lose_next_takeover_acknowledgement = True

        with pytest.raises(
            RuntimeError,
            match="expired recovery takeover acknowledgement lost",
        ):
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=session_id,
                    session_id=child_id,
                    copy_checkpoint=True,
                ),
            )

        after_failure = await store.load(session_id)
        assert after_failure is not None
        assert after_failure.run_epoch > completed.run_epoch
        assert await store.load(child_id) is None
        assert await store.load_checkpoint(session_id) == {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
            "custom_state": {"value": 1},
        }

        retry_events = await collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id=session_id,
                session_id=child_id,
                copy_checkpoint=True,
            ),
        )
        assert [event.type for event in retry_events] == [EventType.SESSION_FORKED]
        assert await store.load_checkpoint(child_id) == {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
            "custom_state": {"value": 1},
        }

    asyncio.run(scenario())


def test_cayu_app_does_not_reconcile_an_ambiguous_takeover_as_a_replacement_owner() -> None:
    current_time = {"value": datetime(2026, 1, 1, 1, tzinfo=UTC)}

    class ReplacementDuringReconciliationStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.lose_next_takeover_acknowledgement = False
            self.replace_during_reconciliation = False
            self.replacement_epoch: int | None = None

        async def fence_run_and_transform_checkpoint(self, *args, **kwargs):
            fenced = await super().fence_run_and_transform_checkpoint(*args, **kwargs)
            if self.lose_next_takeover_acknowledgement:
                self.lose_next_takeover_acknowledgement = False
                self.replace_during_reconciliation = True
                raise RuntimeError("expired recovery takeover acknowledgement lost")
            return fenced

        async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
            checkpoint = await super().load_checkpoint(session_id)
            if not self.replace_during_reconciliation:
                return checkpoint
            self.replace_during_reconciliation = False
            current_time["value"] += timedelta(minutes=6)

            def install_replacement_claim(
                _session: Session,
                current: dict[str, Any] | None,
            ) -> dict[str, Any]:
                updated = {} if current is None else dict(current)
                updated["incomplete_session_recovery_claim"] = {
                    "version": 1,
                    "claim_id": "replacement-owner",
                    "claimed_at": current_time["value"].isoformat(),
                    "claim_expires_at": (current_time["value"] + timedelta(minutes=5)).isoformat(),
                }
                return updated

            replacement = await super().fence_run_and_transform_checkpoint(
                session_id,
                statuses={SessionStatus.COMPLETED},
                checkpoint_transform=install_replacement_claim,
            )
            self.replacement_epoch = replacement.run_epoch
            # The replacement represents another worker. Do not leave its
            # process-local fence in this request's context.
            invocation_scope._deactivate_session_run_fence(session_id)
            return checkpoint

    store = ReplacementDuringReconciliationStore()
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        clock=lambda: current_time["value"],
    )
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> None:
        session_id = "sess_ambiguous_takeover_replaced"
        child_id = "sess_ambiguous_takeover_replaced_child"
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "fork me")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)

        def install_expired_claim(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            updated = {} if checkpoint is None else dict(checkpoint)
            updated["incomplete_session_recovery_claim"] = {
                "version": 1,
                "claim_id": "expired-recovery-owner",
                "claimed_at": (current_time["value"] - timedelta(minutes=10)).isoformat(),
                "claim_expires_at": (current_time["value"] - timedelta(minutes=5)).isoformat(),
            }
            return updated

        await store.transform_checkpoint(session_id, install_expired_claim)
        store.lose_next_takeover_acknowledgement = True

        with pytest.raises(
            RuntimeError,
            match="expired recovery takeover acknowledgement lost",
        ):
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=session_id,
                    session_id=child_id,
                    copy_checkpoint=True,
                ),
            )

        assert await store.load(child_id) is None
        replacement_epoch = store.replacement_epoch
        assert replacement_epoch is not None
        current = await store.load(session_id)
        assert current is not None
        assert current.run_epoch == replacement_epoch
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["incomplete_session_recovery_claim"]["claim_id"] == ("replacement-owner")

    asyncio.run(scenario())


def test_cayu_app_requires_authorization_for_fork_to_different_provider():
    class OtherProvider(FakeProvider):
        name = "other"

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("first answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_provider(OtherProvider([]))
    app.register_environment(Environment(EnvironmentSpec(name="body")), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    app.register_agent(AgentSpec(name="other-agent", model="other-model", provider_name="other"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_fork_provider_source",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    with pytest.raises(
        ExecutionProfileAdoptionRejected,
        match="provider_target",
    ):
        asyncio.run(
            collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id="sess_fork_provider_source",
                    session_id="sess_fork_provider_child",
                    agent_name="other-agent",
                    environment_name="body",
                    copy_checkpoint=False,
                    system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
                    **_current_fork_profile_selection("fork-provider-child"),
                ),
            )
        )

    assert asyncio.run(store.load("sess_fork_provider_child")) is None
    source_checkpoint = asyncio.run(store.load_checkpoint("sess_fork_provider_source"))
    assert source_checkpoint is None or "prompt_anatomy_transition_intents" not in source_checkpoint


def test_cayu_app_forks_partial_transcript_without_checkpoint():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_partial_fork_source",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(store.checkpoint("sess_partial_fork_source", {"context_compaction": {}}))

    asyncio.run(
        collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_partial_fork_source",
                session_id="sess_partial_fork_child",
                transcript_cursor=1,
                copy_checkpoint=False,
            ),
        )
    )

    fork_transcript = asyncio.run(store.load_transcript("sess_partial_fork_child"))
    assert [message.content[0].text for message in fork_transcript] == ["first request"]
    assert asyncio.run(store.load_checkpoint("sess_partial_fork_child")) is None

    with pytest.raises(ValueError, match="copy_checkpoint must be false"):
        asyncio.run(
            collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id="sess_partial_fork_source",
                    session_id="sess_invalid_partial_fork",
                    transcript_cursor=1,
                ),
            )
        )
