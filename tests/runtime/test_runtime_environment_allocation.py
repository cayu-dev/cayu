"""Environment allocation contracts."""

from __future__ import annotations

import asyncio

import pytest
from tests.runtime.runtime_compaction_support import RecordingCompactor
from tests.runtime.runtime_environment_support import (
    EnvironmentProvisioningRecorder,
    MemoryWorkspace,
    RecordingEnvironmentFactory,
    RecordingWorkspaceBinding,
    WorkspaceIdTool,
    assert_environment_was_not_provisioned,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _private_record_for_public_event,
    collect_events,
    collect_fork_events,
    collect_resume_events,
)
from tests.runtime.runtime_tool_support import EchoTool

import cayu.runtime._environment_lifecycle as environment_lifecycle_module
import cayu.runtime._session_engine as session_engine_module
import cayu.runtime.execution_profiles as execution_profiles_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.context.base import CheckpointCompactionContextPolicy
from cayu.environments.base import Environment, EnvironmentSpec, WorkspaceInstructionsConfig
from cayu.environments.bindings import BoundWorkspace, WorkspaceBinding
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.observability.hooks import RuntimeHook, RuntimeHookContext
from cayu.providers.base import ModelStreamEvent, ModelStreamEventType
from cayu.runners.base import DEFAULT_EXEC_OUTPUT_LIMIT_BYTES, ExecCommand, ExecResult, Runner
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY, public_event_id
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import EventRecord, Session, SessionIdentity, SessionStatus
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest, RunRequest
from cayu.sessions.transcript_queries import TranscriptQuery
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus
from cayu.workspaces.local import LocalWorkspace


class RecordingSessionFailedHook(RuntimeHook):
    def __init__(self) -> None:
        self.contexts: list[RuntimeHookContext] = []

    async def after_session_failed(self, context: RuntimeHookContext) -> None:
        self.contexts.append(context)


def test_cayu_app_environment_factory_creates_environment_for_session(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        workspace = LocalWorkspace(workspace_root, workspace_id="factory-workspace")
        factory = RecordingEnvironmentFactory(
            Environment(EnvironmentSpec(name="dynamic"), workspace=workspace),
            metadata={"sandbox_id": "sandbox_123"},
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
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
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory",
                labels={"project": "alpha"},
                metadata={"owner": "test"},
                messages=[Message.text("user", "run tool")],
            ),
        )
        session = await store.load("sess_factory")
        return events, session, factory, workspace, store

    events, session, factory, workspace, store = asyncio.run(run())

    assert session is not None
    assert session.environment_name == "dynamic"
    profile_fingerprint = execution_profiles_module.execution_profile_from_session_metadata(
        session.metadata
    ).fingerprint
    assert [event.type for event in events[:3]] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.SESSION_STARTED,
    ]
    assert events[0].payload == {
        "factory_type": "RecordingEnvironmentFactory",
        "requested_environment_name": "dynamic",
        "parent_session_id": None,
        "causal_budget_id": PRIVATE_EVENT_AUTHORITY,
        "execution_profile_fingerprint": profile_fingerprint,
        "labels": {"project": "alpha"},
    }
    assert events[1].payload == {
        "factory_type": "RecordingEnvironmentFactory",
        "requested_environment_name": "dynamic",
        "parent_session_id": None,
        "causal_budget_id": PRIVATE_EVENT_AUTHORITY,
        "execution_profile_fingerprint": profile_fingerprint,
        "labels": {"project": "alpha"},
        "environment_name": "dynamic",
        "result_metadata": {"sandbox_id": "sandbox_123"},
        "reconnect_metadata": {},
    }
    durable_records = asyncio.run(store.query_events(EventQuery(session_id="sess_factory")))
    assert [
        _private_record_for_public_event(durable_records, event).event.payload["causal_budget_id"]
        for event in events[:2]
    ] == ["sess_factory", "sess_factory"]
    assert len(factory.requests) == 1
    assert factory.requests[0].session_id == "sess_factory"
    assert factory.requests[0].agent_name == "assistant"
    assert factory.requests[0].environment_name == "dynamic"
    assert factory.requests[0].execution_profile_fingerprint == profile_fingerprint
    assert factory.requests[0].operation is EnvironmentFactoryOperation.CREATE
    assert factory.requests[0].labels == {"project": "alpha"}
    assert factory.requests[0].metadata == {"owner": "test"}

    tool_events = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    assert tool_events[0].payload["result"]["structured"] == {"workspace_id": workspace.id}


def test_environment_factory_completion_payload_is_isolated_from_started_event():
    async def run():
        factory = RecordingEnvironmentFactory(
            Environment(EnvironmentSpec(name="dynamic")),
            metadata={"sandbox_id": "sandbox_isolated"},
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(
            FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
            default=True,
        )
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_payload_isolation",
                messages=[Message.text("user", "run")],
            )
        )
        interaction_started = await anext(stream)
        assert interaction_started.type == EventType.INTERACTION_STARTED
        started = await anext(stream)
        started.payload["factory_type"] = "consumer-spoof"
        started.payload["consumer_only"] = True
        remaining = [event async for event in stream]
        return started, remaining

    started, remaining = asyncio.run(run())

    assert started.type == EventType.ENVIRONMENT_FACTORY_STARTED
    completed = next(
        event for event in remaining if event.type is EventType.ENVIRONMENT_FACTORY_COMPLETED
    )
    assert completed.type == EventType.ENVIRONMENT_FACTORY_COMPLETED
    assert completed.payload["factory_type"] == "RecordingEnvironmentFactory"
    assert "consumer_only" not in completed.payload


def test_cayu_app_run_without_a_default_does_not_select_a_static_environment():
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="optional"),
                workspace=MemoryWorkspace("optional-workspace"),
            ),
            default=False,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_without_default_static_environment",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_without_default_static_environment")
        return events, session

    events, session = asyncio.run(run())

    assert session is not None
    assert session.environment_name is None
    assert {event.environment_name for event in events} == {None}


def test_cayu_app_run_without_a_default_does_not_invoke_environment_factory():
    async def run():
        store = InMemorySessionStore()
        provisioning = EnvironmentProvisioningRecorder("optional")
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="optional"),
            provisioning.factory,
            default=False,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_without_default_factory",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_without_default_factory")
        checkpoint = await store.load_checkpoint("sess_without_default_factory")
        return events, session, checkpoint, provisioning

    events, session, checkpoint, provisioning = asyncio.run(run())

    assert_environment_was_not_provisioned(provisioning)
    assert session is not None
    assert session.environment_name is None
    assert {event.environment_name for event in events} == {None}
    assert not any(
        event.type
        in {
            EventType.ENVIRONMENT_FACTORY_STARTED,
            EventType.ENVIRONMENT_FACTORY_COMPLETED,
            EventType.ENVIRONMENT_FACTORY_FAILED,
            EventType.ENVIRONMENT_BINDING_STARTED,
            EventType.ENVIRONMENT_BINDING_COMPLETED,
            EventType.ENVIRONMENT_BINDING_FAILED,
            EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
            EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
            EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
        }
        for event in events
    )
    assert checkpoint is None or "environment_factory_reconnect" not in checkpoint


def test_cayu_app_explicitly_named_non_default_factory_is_invoked_once():
    async def run():
        store = InMemorySessionStore()
        provisioning = EnvironmentProvisioningRecorder("optional")
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="optional"),
            provisioning.factory,
            default=False,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                environment_name="optional",
                session_id="sess_explicit_non_default_factory",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_explicit_non_default_factory")
        return events, session, provisioning

    events, session, provisioning = asyncio.run(run())

    assert len(provisioning.factory.requests) == 1
    factory_request = provisioning.factory.requests[0]
    assert factory_request.environment_name == "optional"
    assert provisioning.workspace_allocator.requests == [factory_request]
    assert len(provisioning.workspace_allocator.workspaces) == 1
    assert provisioning.runner_allocator.requests == [factory_request]
    assert len(provisioning.runner_allocator.runners) == 1
    assert provisioning.binding_allocator.requests == [factory_request]
    assert len(provisioning.binding_allocator.bindings) == 1
    assert len(provisioning.binding_allocator.bindings[0].bind_calls) == 1
    assert len(provisioning.binding_allocator.bindings[0].finalize_calls) == 1
    assert session is not None
    assert session.environment_name == "optional"
    assert [event.type for event in events[:2]] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
    ]
    assert events[1].payload["reconnect_metadata"] == {"sandbox_id": "unexpected"}
    assert {event.environment_name for event in events} == {"optional"}


def test_cayu_app_streams_out_of_band_events_from_environment_factory(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        app = CayuApp(session_store=store, enable_logging=False)

        class EmittingFactory(EnvironmentFactory):
            async def create(
                self,
                request: EnvironmentFactoryRequest,
            ) -> EnvironmentFactoryResult:
                await app.emit_event(
                    Event(
                        type=EventType.EGRESS_GRANT_MINTED,
                        session_id=request.session_id,
                        agent_name=request.agent_name,
                        environment_name=request.environment_name,
                        payload={"grant_id": "grant_1"},
                    )
                )
                return EnvironmentFactoryResult(
                    environment=Environment(
                        EnvironmentSpec(name="dynamic"),
                        workspace=LocalWorkspace(
                            workspace_root,
                            workspace_id="factory-workspace",
                        ),
                    ),
                )

        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            EmittingFactory(),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_oob",
                messages=[Message.text("user", "run")],
            ),
        )
        return events, store

    events, store = asyncio.run(run())
    event_types = [event.type for event in events]

    assert EventType.EGRESS_GRANT_MINTED in event_types
    assert event_types.index(EventType.EGRESS_GRANT_MINTED) < event_types.index(
        EventType.SESSION_STARTED
    )
    event = next(event for event in events if event.type == EventType.EGRESS_GRANT_MINTED)
    assert event.agent_name == "assistant"
    assert event.environment_name == "dynamic"
    assert event.payload == {"grant_id": PRIVATE_EVENT_AUTHORITY}
    records = asyncio.run(store.query_events(EventQuery(session_id="sess_factory_oob")))
    assert _private_record_for_public_event(records, event).event.payload == {"grant_id": "grant_1"}


def test_cayu_app_scoped_event_emitter_rejects_unlisted_event_types():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_scoped",
                messages=[Message.text("user", "run")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        emitter = app.scoped_event_emitter(event_types=(EventType.EGRESS_GRANT_MINTED,))

        emitted = await emitter(
            Event(
                type=EventType.EGRESS_GRANT_MINTED,
                session_id="sess_scoped",
                payload={"grant_id": "grant_1"},
            )
        )
        with pytest.raises(ValueError, match="not allowed"):
            await emitter(
                Event(
                    type=EventType.SESSION_STARTED,
                    session_id="sess_scoped",
                )
            )
        return emitted, await store.query_events(EventQuery(session_id="sess_scoped"))

    emitted, records = asyncio.run(run())

    assert emitted.type == EventType.EGRESS_GRANT_MINTED
    assert emitted.id == public_event_id(records[0].sequence)
    assert emitted.id != records[0].event.id
    assert [record.event.type for record in records] == [EventType.EGRESS_GRANT_MINTED]


def test_cayu_app_streams_out_of_band_events_from_binding_finalize():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)

        class FinalizeEmitterBinding(WorkspaceBinding):
            async def bind(
                self,
                workspace,
                runner,
                *,
                session_id,
                agent_name=None,
                environment_name=None,
                metadata=None,
            ):
                return BoundWorkspace(path="/bound")

            async def finalize(self, bound, *, outcome=None, metadata=None):
                await app.emit_event(
                    Event(
                        type=EventType.EGRESS_GRANT_REVOKED,
                        session_id=metadata["session_id"],
                        agent_name="assistant",
                        environment_name="dynamic",
                        payload={"grant_id": "grant_1", "outcome": outcome},
                    )
                )
                return None

        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                binding=FinalizeEmitterBinding(),
            )
        )
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_finalize_oob",
                messages=[Message.text("user", "run")],
            ),
        )
        return events, store

    events, store = asyncio.run(run())
    event_types = [event.type for event in events]

    assert EventType.EGRESS_GRANT_REVOKED in event_types
    assert event_types.index(EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED) < event_types.index(
        EventType.EGRESS_GRANT_REVOKED
    )
    assert event_types.index(EventType.EGRESS_GRANT_REVOKED) < event_types.index(
        EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
    )
    event = next(event for event in events if event.type == EventType.EGRESS_GRANT_REVOKED)
    assert event.payload == {
        "grant_id": PRIVATE_EVENT_AUTHORITY,
        "outcome": "completed",
    }
    records = asyncio.run(store.query_events(EventQuery(session_id="sess_finalize_oob")))
    assert _private_record_for_public_event(records, event).event.payload == {
        "grant_id": "grant_1",
        "outcome": "completed",
    }


def test_cayu_app_environment_factory_failure_fails_session_before_start_event(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
            ),
            fail_create=True,
        )
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("unreached"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_fail",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_factory_fail")
        transcript = await store.query_transcript(
            TranscriptQuery(session_id="sess_factory_fail", interaction_id=None)
        )
        lifecycle = await store.query_events(
            EventQuery(
                session_id="sess_factory_fail",
                event_type=EventType.INTERACTION_FAILED,
            )
        )
        return events, session, factory, provider, transcript, lifecycle

    events, session, factory, provider, transcript, lifecycle = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.SESSION_FAILED,
    ]
    assert events[1].payload["error"] == "factory failed"
    assert events[1].payload["error_type"] == "RuntimeError"
    assert events[2].payload["error"] == "factory failed"
    assert events[2].payload["error_type"] == "RuntimeError"
    assert len(factory.requests) == 1
    assert provider.requests == []
    assert [record.message for record in transcript.records] == [Message.text("user", "run")]
    assert transcript.records[0].interaction_id is not None
    from cayu.sessions.interactions import InteractionSummaryEvidence

    evidence = InteractionSummaryEvidence.model_validate(lifecycle[0].event.payload)
    assert evidence.source_transcript_start == 0
    assert evidence.source_transcript_end == 0


def test_cayu_app_resume_factory_failure_retains_atomic_source_input(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "resume-factory"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(
                    workspace_root,
                    workspace_id="resume-factory-workspace",
                ),
            )
        )
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("first response"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume_factory_fail",
                messages=[Message.text("user", "first request")],
            ),
        )
        factory.fail_create = True
        await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_resume_factory_fail",
                messages=[Message.text("user", "exact resume request")],
            ),
        )
        session = await store.load("sess_resume_factory_fail")
        assert session is not None
        interactions = await store.query_latest_interaction_events(session.id, limit=10)
        latest_interaction_id = interactions[0].event.interaction_id
        assert latest_interaction_id is not None
        transcript = await store.query_transcript(
            TranscriptQuery(
                session_id=session.id,
                interaction_id=latest_interaction_id,
            )
        )
        return session, provider, transcript, interactions[0]

    session, provider, transcript, latest = asyncio.run(run())

    assert session.status is SessionStatus.FAILED
    assert len(provider.requests) == 1
    assert [record.message for record in transcript.records] == [
        Message.text("user", "exact resume request")
    ]
    assert latest.event.type is EventType.INTERACTION_FAILED
    from cayu.sessions.interactions import InteractionSummaryEvidence

    evidence = InteractionSummaryEvidence.model_validate(latest.event.payload)
    assert evidence.source_transcript_start == transcript.records[0].index
    assert evidence.source_transcript_end == transcript.records[0].index


def test_cayu_app_environment_factory_failure_fails_worker_claimed_task(tmp_path):
    async def run():
        store = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        task = await tasks.create_task(TaskCreate(task_id="task_factory_claim", type="run"))
        claimed = await tasks.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        assert claimed.id == task.id
        workspace_root = tmp_path / "factory-claimed-task"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(
                    workspace_root,
                    workspace_id="factory-claimed-task-workspace",
                ),
            ),
            fail_create=True,
        )
        provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app = CayuApp(
            session_store=store,
            task_store=tasks,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_claim_failure",
                task_id=task.id,
                task_worker_id="worker_a",
                task_lease_expires_at=claimed.lease_expires_at,
                messages=[Message.text("user", "run")],
            ),
        )
        return events, await tasks.load_task(task.id)

    events, task = asyncio.run(run())

    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.TASK_STARTED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.TASK_FAILED,
        EventType.SESSION_FAILED,
    ]
    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert task.session_id == "sess_factory_claim_failure"
    assert task.worker_id is None
    assert task.lease_expires_at is None
    assert task.error == {
        "message": "factory failed",
        "type": "RuntimeError",
        "session_id": "sess_factory_claim_failure",
    }


def test_cayu_app_closes_factory_environment_when_post_create_checkpoint_fails():
    class ClosingRunner(Runner):
        isolation = "test"
        default_cwd = "/"

        def __init__(self) -> None:
            self.close_calls = 0

        async def exec(
            self,
            command: ExecCommand,
            *,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_s: int | None = None,
            stdin: str | None = None,
            output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
        ) -> ExecResult:
            raise AssertionError("runner should not execute")

        async def close(self) -> None:
            self.close_calls += 1
            await super().close()

    class FailingReconnectCheckpointStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def transform_checkpoint(self, session_id, checkpoint_transform) -> None:
            def fail_reconnect(session, checkpoint):
                state = checkpoint_transform(session, checkpoint)
                if (
                    state is not None
                    and environment_lifecycle_module.ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY
                    in state
                ):
                    raise RuntimeError("reconnect checkpoint unavailable")
                return state

            await super().transform_checkpoint(session_id, fail_reconnect)

    async def run() -> tuple[list[Event], ClosingRunner, FakeProvider]:
        runner = ClosingRunner()
        factory = RecordingEnvironmentFactory(
            Environment(EnvironmentSpec(name="dynamic"), runner=runner)
        )
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("unreached"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=FailingReconnectCheckpointStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_post_create_fail",
                messages=[Message.text("user", "run")],
            ),
        )
        return events, runner, provider

    events, runner, provider = asyncio.run(run())

    assert runner.close_calls == 1
    assert runner._closed is True
    assert provider.requests == []
    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.SESSION_FAILED,
    ]
    assert events[1].payload["error"] == "reconnect checkpoint unavailable"


def test_cayu_app_closes_factory_environment_when_post_create_checkpoint_is_cancelled():
    class ClosingRunner(Runner):
        isolation = "test"
        default_cwd = "/"

        def __init__(self) -> None:
            self.close_calls = 0

        async def exec(
            self,
            command: ExecCommand,
            *,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_s: int | None = None,
            stdin: str | None = None,
            output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
        ) -> ExecResult:
            raise AssertionError("runner should not execute")

        async def close(self) -> None:
            self.close_calls += 1
            await super().close()

    class CancellingReconnectCheckpointStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def transform_checkpoint(self, session_id, checkpoint_transform) -> None:
            def cancel_reconnect(session, checkpoint):
                state = checkpoint_transform(session, checkpoint)
                if (
                    state is not None
                    and environment_lifecycle_module.ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY
                    in state
                ):
                    raise asyncio.CancelledError()
                return state

            await super().transform_checkpoint(session_id, cancel_reconnect)

    async def run() -> tuple[ClosingRunner, Session, list[Event]]:
        runner = ClosingRunner()
        factory = RecordingEnvironmentFactory(
            Environment(EnvironmentSpec(name="dynamic"), runner=runner)
        )
        store = CancellingReconnectCheckpointStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(asyncio.CancelledError):
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_factory_post_create_cancel",
                    messages=[Message.text("user", "run")],
                ),
            )
        session = await store.load("sess_factory_post_create_cancel")
        assert session is not None
        events = await store.load_events("sess_factory_post_create_cancel")
        return runner, session, events

    runner, session, events = asyncio.run(run())

    assert runner.close_calls == 1
    assert runner._closed is True
    assert session.status == SessionStatus.INTERRUPTED
    assert session.run_epoch == 2
    interrupted = [event for event in events if event.type == EventType.SESSION_INTERRUPTED]
    assert len(interrupted) == 1
    assert interrupted[0].payload["abandoned"] is True


def test_cayu_app_releases_run_fence_when_pre_run_failure_recording_fails(tmp_path):
    class FailingSessionFailedEventStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            if any(event.type == EventType.SESSION_FAILED for event in events):
                raise RuntimeError("terminal event unavailable")
            await super().append_events(session_id, events)

    async def run() -> Session:
        store = FailingSessionFailedEventStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
            ),
            fail_create=True,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(RuntimeError, match="terminal event unavailable"):
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_factory_terminal_failure",
                    messages=[Message.text("user", "run")],
                ),
            )
        session = await store.load("sess_factory_terminal_failure")
        assert session is not None
        return session

    session = asyncio.run(run())

    assert session.status == SessionStatus.FAILED
    assert session.run_epoch == 2


def test_cayu_app_terminalizes_interaction_when_initial_message_projection_fails(monkeypatch):
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    def fail_initial_messages(**kwargs):
        raise RuntimeError("initial message projection unavailable")

    monkeypatch.setattr(
        session_engine_module.transcript_helpers, "initial_messages", fail_initial_messages
    )

    async def run() -> tuple[Session, list[Event], list[Message], list[EventRecord]]:
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_initial_message_failure",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_initial_message_failure")
        assert session is not None
        transcript = await store.load_transcript(session.id)
        lifecycle = await store.query_events(
            EventQuery(
                session_id=session.id,
                event_types=(EventType.INTERACTION_STARTED, EventType.INTERACTION_FAILED),
            )
        )
        return session, events, transcript, lifecycle

    session, events, transcript, lifecycle = asyncio.run(run())

    assert session.status == SessionStatus.FAILED
    assert session.run_epoch == 2
    assert [event.type for event in events] == [EventType.SESSION_FAILED]
    assert transcript == [Message.text("user", "run")]
    assert [record.event.type for record in lifecycle] == [
        EventType.INTERACTION_STARTED,
        EventType.INTERACTION_FAILED,
    ]


def test_cayu_app_environment_factory_failure_runs_failed_session_hooks(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
            ),
            fail_create=True,
        )
        hook = RecordingSessionFailedHook()
        app = CayuApp(
            session_store=store,
            runtime_hooks=[hook],
            enable_logging=False,
        )
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_fail_hook",
                messages=[Message.text("user", "run")],
            ),
        )
        return events, hook

    events, hook = asyncio.run(run())

    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.SESSION_FAILED,
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
    ]
    assert len(hook.contexts) == 1
    assert hook.contexts[0].session.status == SessionStatus.FAILED
    assert hook.contexts[0].terminal_event.type == EventType.SESSION_FAILED
    assert hook.contexts[0].terminal_event.payload["error"] == "factory failed"


def test_cayu_app_environment_factory_workspace_instruction_failure_fails_session(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        (workspace_root / "AGENTS.md").write_text("too long")
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
                workspace_instructions=WorkspaceInstructionsConfig(
                    paths=("AGENTS.md",),
                    max_bytes=3,
                ),
            )
        )
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("unreached"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_instruction_fail",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_factory_instruction_fail")
        return events, session, provider

    events, session, provider = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert "exceeds 3 bytes" in events[2].payload["error"]
    assert provider.requests == []


def test_cayu_app_environment_factory_workspace_instruction_failure_runs_hooks(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        (workspace_root / "AGENTS.md").write_text("too long")
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
                workspace_instructions=WorkspaceInstructionsConfig(
                    paths=("AGENTS.md",),
                    max_bytes=3,
                ),
            )
        )
        hook = RecordingSessionFailedHook()
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("unreached"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(
            session_store=store,
            runtime_hooks=[hook],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_instruction_fail_hook",
                messages=[Message.text("user", "run")],
            ),
        )
        return events, hook, provider

    events, hook, provider = asyncio.run(run())

    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
    ]
    assert len(hook.contexts) == 1
    assert hook.contexts[0].session.status == SessionStatus.FAILED
    assert hook.contexts[0].terminal_event.type == EventType.SESSION_FAILED
    assert "exceeds 3 bytes" in hook.contexts[0].terminal_event.payload["error"]
    assert provider.requests == []


def test_cayu_app_environment_factory_result_name_must_match_registration(tmp_path):
    async def run():
        store = InMemorySessionStore()
        release_actions: list[EnvironmentFactoryReleaseAction] = []

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            release_actions.append(action)

        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="different"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
            ),
            release=release,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_name_mismatch",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_factory_name_mismatch")
        return events, session, release_actions

    events, session, release_actions = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.SESSION_FAILED,
    ]
    assert "different environment name" in events[1].payload["error"]
    assert events[1].payload["environment_factory_release"] == {
        "action": "discard",
        "callback_provided": True,
        "completed": True,
    }
    assert (
        events[2].payload["environment_factory_release"]
        == events[1].payload["environment_factory_release"]
    )
    assert release_actions == [EnvironmentFactoryReleaseAction.DISCARD]


def test_cayu_app_rejects_invalid_allocation_fingerprint_before_checkpoint(tmp_path):
    async def run():
        store = InMemorySessionStore()
        release_actions: list[EnvironmentFactoryReleaseAction] = []

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            release_actions.append(action)

        workspace_root = tmp_path / "factory-invalid-allocation-fingerprint"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
            ),
            reconnect_metadata={"allocation_fingerprint": "A" * 64},
            release=release,
        )
        provider = FakeProvider([])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_invalid_allocation_fingerprint",
                messages=[Message.text("user", "run")],
            ),
        )
        checkpoint = await store.load_checkpoint("sess_factory_invalid_allocation_fingerprint")
        return events, checkpoint, release_actions, provider

    events, checkpoint, release_actions, provider = asyncio.run(run())

    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.SESSION_FAILED,
    ]
    assert "environment_factory_reconnect" not in (checkpoint or {})
    assert release_actions == [EnvironmentFactoryReleaseAction.DISCARD]
    assert provider.requests == []


def test_cayu_app_environment_factory_output_composes_with_workspace_binding(tmp_path):
    async def run():
        store = InMemorySessionStore()
        configured_root = tmp_path / "configured"
        configured_root.mkdir()
        bound_root = tmp_path / "bound"
        bound_root.mkdir()
        configured_workspace = LocalWorkspace(configured_root, workspace_id="configured")
        bound_workspace = LocalWorkspace(bound_root, workspace_id="bound")
        binding = RecordingWorkspaceBinding(bound_workspace=bound_workspace)
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=configured_workspace,
                binding=binding,
            )
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
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
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_binding",
                messages=[Message.text("user", "run")],
            ),
        )
        return events, binding, configured_workspace, bound_workspace

    events, binding, configured_workspace, bound_workspace = asyncio.run(run())

    assert [event.type for event in events[:5]] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.ENVIRONMENT_BINDING_STARTED,
        EventType.ENVIRONMENT_BINDING_COMPLETED,
        EventType.SESSION_STARTED,
    ]
    assert binding.bind_calls[0]["workspace"] is configured_workspace
    assert binding.bind_calls[0]["environment_name"] == "dynamic"
    tool_events = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    assert tool_events[0].payload["result"]["structured"] == {"workspace_id": bound_workspace.id}


def test_cayu_app_resume_uses_stored_factory_backed_environment(tmp_path):
    async def run():
        store = InMemorySessionStore()
        dynamic_root = tmp_path / "dynamic"
        dynamic_root.mkdir()
        static_root = tmp_path / "static"
        static_root.mkdir()
        dynamic_workspace = LocalWorkspace(dynamic_root, workspace_id="dynamic-workspace")
        static_workspace = LocalWorkspace(static_root, workspace_id="static-workspace")
        factory = RecordingEnvironmentFactory(
            Environment(EnvironmentSpec(name="dynamic"), workspace=dynamic_workspace)
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="call_2",
                        name="workspace_id",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done again"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=False,
        )
        app.register_environment(
            Environment(EnvironmentSpec(name="static"), workspace=static_workspace),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
        )

        run_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_resume",
                environment_name="dynamic",
                messages=[Message.text("user", "run tool")],
            ),
        )
        resume_events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_factory_resume",
                messages=[Message.text("user", "run again")],
            ),
        )
        return run_events, resume_events, factory, dynamic_workspace

    run_events, resume_events, factory, dynamic_workspace = asyncio.run(run())

    assert len(factory.requests) == 2
    assert [request.environment_name for request in factory.requests] == [
        "dynamic",
        "dynamic",
    ]
    assert run_events[0].type == EventType.ENVIRONMENT_FACTORY_STARTED
    assert resume_events[0].type == EventType.ENVIRONMENT_FACTORY_STARTED
    tool_events = [
        event
        for event in [*run_events, *resume_events]
        if event.type == EventType.TOOL_CALL_COMPLETED
    ]
    assert [event.payload["result"]["structured"]["workspace_id"] for event in tool_events] == [
        dynamic_workspace.id,
        dynamic_workspace.id,
    ]


def test_cayu_app_fork_preserves_an_explicit_non_default_environment():
    async def run():
        store = InMemorySessionStore()
        factory = RecordingEnvironmentFactory(
            Environment(EnvironmentSpec(name="optional")),
            reconnect_metadata={"sandbox_id": "sandbox_source"},
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("source done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("child done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="optional"),
            factory,
            default=False,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                environment_name="optional",
                session_id="sess_non_default_environment_source",
                messages=[Message.text("user", "run")],
            ),
        )
        fork_events = await collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_non_default_environment_source",
                session_id="sess_non_default_environment_child",
            ),
        )
        child_events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_non_default_environment_child",
                messages=[Message.text("user", "continue")],
            ),
        )
        child = await store.load("sess_non_default_environment_child")
        return fork_events, child_events, child, factory

    fork_events, child_events, child, factory = asyncio.run(run())

    assert len(factory.requests) == 2
    assert factory.requests[1].session_id == "sess_non_default_environment_child"
    assert factory.requests[1].parent_session_id == "sess_non_default_environment_source"
    assert factory.requests[1].environment_name == "optional"
    assert factory.requests[1].operation is EnvironmentFactoryOperation.CREATE
    assert factory.requests[1].reconnect_metadata == {"sandbox_id": "sandbox_source"}
    assert child is not None
    assert child.environment_name == "optional"
    assert fork_events[0].environment_name == "optional"
    assert child_events[0].type == EventType.ENVIRONMENT_FACTORY_STARTED
    assert {event.environment_name for event in child_events} == {"optional"}


def test_fork_reconnects_child_allocation_checkpointed_before_completion_event():
    async def run():
        store = InMemorySessionStore()
        release_actions: list[EnvironmentFactoryReleaseAction] = []

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            release_actions.append(action)

        factory = RecordingEnvironmentFactory(
            Environment(EnvironmentSpec(name="dynamic")),
            reconnect_metadata={"sandbox_id": "sbx_source"},
            release=release,
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("source done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("child done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(EnvironmentSpec(name="dynamic"), factory, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_crash_source",
                messages=[Message.text("user", "run")],
            ),
        )
        await collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_crash_source",
                session_id="sess_crash_child",
            ),
        )
        factory.reconnect_metadata = {"sandbox_id": "sbx_child"}
        original_emit = app._event_writer.emit
        completion_failed = False

        async def fail_child_completion_once(event: Event) -> Event:
            nonlocal completion_failed
            if (
                not completion_failed
                and event.session_id == "sess_crash_child"
                and event.type is EventType.ENVIRONMENT_FACTORY_COMPLETED
            ):
                completion_failed = True
                raise RuntimeError("simulated factory completion persistence failure")
            return await original_emit(event)

        app._event_writer.emit = fail_child_completion_once
        first_resume_events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_crash_child",
                messages=[Message.text("user", "first attempt")],
            ),
        )
        app._event_writer.emit = original_emit
        child_checkpoint = await store.load_checkpoint("sess_crash_child")
        child_events_before_resume = await store.load_events("sess_crash_child")
        assert completion_failed is True
        assert EventType.ENVIRONMENT_FACTORY_FAILED in {event.type for event in first_resume_events}
        factory_failed = next(
            event
            for event in first_resume_events
            if event.type is EventType.ENVIRONMENT_FACTORY_FAILED
        )
        session_failed = next(
            event for event in first_resume_events if event.type is EventType.SESSION_FAILED
        )
        assert factory_failed.payload["environment_factory_release"] == {
            "action": "preserve",
            "callback_provided": True,
            "completed": True,
        }
        assert (
            session_failed.payload["environment_factory_release"]
            == factory_failed.payload["environment_factory_release"]
        )
        assert not any(
            event.type is EventType.ENVIRONMENT_FACTORY_COMPLETED
            for event in child_events_before_resume
        )
        await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_crash_child",
                messages=[Message.text("user", "continue")],
            ),
        )
        return factory, child_checkpoint, release_actions

    factory, child_checkpoint, release_actions = asyncio.run(run())

    assert child_checkpoint is not None
    assert child_checkpoint["environment_factory_reconnect"] == {
        "dynamic": {"sandbox_id": "sbx_child"}
    }
    assert child_checkpoint["environment_factory_allocation_owner"] == {
        "dynamic": "sess_crash_child"
    }
    first_child_request, retried_child_request = factory.requests[-2:]
    assert first_child_request.session_id == "sess_crash_child"
    assert first_child_request.operation is EnvironmentFactoryOperation.CREATE
    assert first_child_request.reconnect_metadata == {"sandbox_id": "sbx_source"}
    assert retried_child_request.session_id == "sess_crash_child"
    assert retried_child_request.operation is EnvironmentFactoryOperation.RECONNECT
    assert retried_child_request.reconnect_metadata == {"sandbox_id": "sbx_child"}
    assert release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]


def test_cayu_app_resume_and_fork_do_not_invent_a_later_default_environment():
    async def run():
        store = InMemorySessionStore()
        provisioning = EnvironmentProvisioningRecorder("later-default")
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("source done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("source resumed"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("child resumed"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        source_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_without_environment_source",
                messages=[Message.text("user", "run")],
            ),
        )
        app.register_environment_factory(
            EnvironmentSpec(name="later-default"),
            provisioning.factory,
            default=True,
        )
        resumed_source_events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_without_environment_source",
                messages=[Message.text("user", "continue")],
            ),
        )
        fork_events = await collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_without_environment_source",
                session_id="sess_without_environment_child",
            ),
        )
        resumed_child_events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_without_environment_child",
                messages=[Message.text("user", "continue")],
            ),
        )
        source = await store.load("sess_without_environment_source")
        child = await store.load("sess_without_environment_child")
        return (
            source_events,
            resumed_source_events,
            fork_events,
            resumed_child_events,
            source,
            child,
            provisioning,
        )

    (
        source_events,
        resumed_source_events,
        fork_events,
        resumed_child_events,
        source,
        child,
        provisioning,
    ) = asyncio.run(run())

    assert_environment_was_not_provisioned(provisioning)
    assert source is not None
    assert source.environment_name is None
    assert child is not None
    assert child.environment_name is None
    assert {
        event.environment_name
        for event in [
            *source_events,
            *resumed_source_events,
            *fork_events,
            *resumed_child_events,
        ]
    } == {None}


def test_cayu_app_environment_factory_reconnect_metadata_round_trips_on_resume(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "dynamic"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="dynamic-workspace"),
            ),
            reconnect_metadata={"sandbox_id": "sandbox_1"},
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("done again"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        run_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_reconnect",
                messages=[Message.text("user", "run")],
            ),
        )
        factory.reconnect_metadata = {"sandbox_id": "sandbox_1", "generation": 2}
        resume_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="sess_factory_reconnect",
                    messages=[Message.text("user", "resume")],
                )
            )
        ]
        checkpoint = await store.load_checkpoint("sess_factory_reconnect")
        return run_events, resume_events, factory, checkpoint

    run_events, resume_events, factory, checkpoint = asyncio.run(run())

    completed_events = [
        event
        for event in [*run_events, *resume_events]
        if event.type == EventType.ENVIRONMENT_FACTORY_COMPLETED
    ]
    assert [request.reconnect_metadata for request in factory.requests] == [
        {},
        {"sandbox_id": "sandbox_1"},
    ]
    assert [request.operation for request in factory.requests] == [
        EnvironmentFactoryOperation.CREATE,
        EnvironmentFactoryOperation.RECONNECT,
    ]
    assert [event.payload["reconnect_metadata"] for event in completed_events] == [
        {"sandbox_id": "sandbox_1"},
        {"sandbox_id": "sandbox_1", "generation": 2},
    ]
    assert checkpoint is not None
    assert checkpoint["environment_factory_reconnect"] == {
        "dynamic": {"sandbox_id": "sandbox_1", "generation": 2}
    }


def test_reconnect_keeps_the_allocation_identity_it_was_checkpointed_with(tmp_path):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "dynamic"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="dynamic-workspace"),
            ),
            reconnect_metadata={"envelope": {"sandbox_id": "sandbox_1"}},
        )
        turns = [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for _ in range(4)
        ]
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider(turns), default=True)
        app.register_environment_factory(EnvironmentSpec(name="dynamic"), factory, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def resume(session_id: str) -> list:
            return [
                event
                async for event in app.resume(
                    ResumeRequest(session_id=session_id, messages=[Message.text("user", "go")])
                )
            ]

        async def identities(session_id: str):
            checkpoint = await store.load_checkpoint(session_id)
            return (checkpoint or {}).get("environment_factory_allocation_identity")

        # Published through the typed field, beside metadata that does not carry it.
        factory.allocation_fingerprint = "a" * 64
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_identity",
                messages=[Message.text("user", "run")],
            ),
        )
        assert await identities("sess_identity") == {"dynamic": "a" * 64}
        await resume("sess_identity")
        assert (await store.load("sess_identity")).status is SessionStatus.COMPLETED
        factory.allocation_fingerprint = "b" * 64
        changed = await resume("sess_identity")
        assert any(event.type is EventType.ENVIRONMENT_FACTORY_FAILED for event in changed)
        assert await identities("sess_identity") == {"dynamic": "a" * 64}

        # An allocation checkpointed without an identity does not gain one later.
        factory.allocation_fingerprint = None
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_no_identity",
                messages=[Message.text("user", "run")],
            ),
        )
        factory.allocation_fingerprint = "c" * 64
        await resume("sess_no_identity")
        assert (await store.load("sess_no_identity")).status is SessionStatus.COMPLETED
        assert await identities("sess_no_identity") is None
        return changed

    changed = asyncio.run(run())
    failed = next(event for event in changed if event.type is EventType.ENVIRONMENT_FACTORY_FAILED)
    assert "changed its allocation fingerprint" in str(failed.payload)


def test_cayu_app_environment_factory_reconnect_metadata_survives_context_checkpoint(
    tmp_path,
):
    async def run():
        store = InMemorySessionStore()
        workspace_root = tmp_path / "dynamic"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="dynamic-workspace"),
            ),
            reconnect_metadata={"sandbox_id": "sandbox_compaction"},
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("done again"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="You are careful.",
            ),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )

        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_reconnect_compaction",
                messages=[
                    Message.text("user", "old one"),
                    Message.text("assistant", "old answer one"),
                    Message.text("user", "current"),
                ],
            ),
        )
        first_checkpoint = await store.load_checkpoint("sess_factory_reconnect_compaction")
        _ = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="sess_factory_reconnect_compaction",
                    messages=[Message.text("user", "resume")],
                )
            )
        ]
        return factory, first_checkpoint

    factory, first_checkpoint = asyncio.run(run())

    assert first_checkpoint is not None
    assert first_checkpoint["environment_factory_reconnect"] == {
        "dynamic": {"sandbox_id": "sandbox_compaction"}
    }
    assert first_checkpoint["environment_factory_allocation_owner"] == {
        "dynamic": "sess_factory_reconnect_compaction"
    }
    assert "context_compaction" in first_checkpoint
    assert [request.reconnect_metadata for request in factory.requests] == [
        {},
        {"sandbox_id": "sandbox_compaction"},
    ]


def test_cayu_app_registers_and_selects_default_environment():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local", metadata={"kind": "dev"})),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_default_environment",
                messages=[Message.text("user", "hi")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_default_environment"))

    assert events[0].payload == {"agent_name": "assistant", "run_epoch": 1}
    assert events[0].environment_name == "local"
    assert events[1].environment_name == "local"
    assert events[-1].environment_name == "local"
    assert provider.requests[0].options["environment_metadata"] == {"kind": "dev"}
    assert session is not None
    assert session.environment_name == "local"
    assert app.get_environment().spec.name == "local"
    assert app.get_environment("local").spec.metadata == {"kind": "dev"}


def test_cayu_app_runs_with_explicit_environment():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="docker", metadata={"kind": "isolated"}))
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                environment_name="docker",
                session_id="sess_explicit_environment",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[0].environment_name == "docker"
    assert provider.requests[0].options["environment_metadata"] == {"kind": "isolated"}


def test_cayu_app_runs_without_environment_when_none_registered():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_no_environment",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[0].environment_name is None
    assert events[-1].environment_name is None
    assert "environment_name" not in events[0].payload
    assert "environment_name" not in events[-1].payload
    assert provider.requests[0].options["environment_metadata"] == {}

    with pytest.raises(RuntimeError, match="No environment registered"):
        app.get_environment()


def test_cayu_app_rejects_unknown_environment_for_run():
    app = CayuApp()
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(KeyError, match="Environment not registered"):
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    environment_name="missing",
                    messages=[Message.text("user", "hi")],
                ),
            )
        )


def test_cayu_app_includes_environment_on_failed_session_event():
    provider = FakeProvider(
        [
            ModelStreamEvent(
                type=ModelStreamEventType.TOOL_CALL,
                payload={"name": "echo", "arguments": "not-an-object"},
            ),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].environment_name == "local"
    assert events[-1].payload["error_type"] == "ValueError"


def test_cayu_app_tags_all_runtime_events_with_environment():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "hello"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events
    assert {event.environment_name for event in events} == {"local"}
    assert all("environment_name" not in event.payload for event in events)


def test_cayu_app_model_events_use_runtime_environment():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_environment",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert events
    model_events = [
        event
        for event in events
        if event.type
        in {
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
        }
    ]
    assert {event.environment_name for event in model_events} == {"local"}


def test_cayu_app_preserves_labels_when_environment_sets_name():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    session_store = InMemorySessionStore()
    app = CayuApp(session_store=session_store)
    app.register_provider(provider, default=True)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_environment_labels",
                labels={"owner": "org_123", "project": "ap_q2"},
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    session = asyncio.run(session_store.load("sess_environment_labels"))
    assert session is not None
    assert session.environment_name == "local"
    assert session.labels == {"owner": "org_123", "project": "ap_q2"}


def test_environment_rejects_invalid_bound_services():
    class WorkspaceLike:
        id = "workspace"

    class RunnerLike:
        isolation = "fake"

    class ArtifactStoreLike:
        id = "artifacts"

    class VaultLike:
        pass

    with pytest.raises(TypeError, match="workspace"):
        Environment(
            EnvironmentSpec(name="workspace_like"),
            workspace=WorkspaceLike(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="runner"):
        Environment(
            EnvironmentSpec(name="runner_like"),
            runner=RunnerLike(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="artifact_store"):
        Environment(
            EnvironmentSpec(name="artifact_store_like"),
            artifact_store=ArtifactStoreLike(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="vault"):
        Environment(
            EnvironmentSpec(name="vault_like"),
            vault=VaultLike(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="mcp_servers"):
        Environment(
            EnvironmentSpec(name="bad_mcp_servers"),
            mcp_servers="not iterable specs",  # type: ignore[arg-type]
        )
