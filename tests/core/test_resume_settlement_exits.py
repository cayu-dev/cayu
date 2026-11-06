"""A resume settles and releases its run fence on every exit, then resumes again."""

import asyncio
import contextlib

import pytest
from tests.core.test_workspace_mutation_receipts import (
    DeterministicWorkspaceBinding,
    Environment,
    LocalWorkspace,
    _portable_environment_spec,
)

import cayu.runtime._model_completion_recovery as model_completion_recovery
from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.evals import ScriptedModelProvider
from cayu.exceptions import InteractionLifecyclePublicationRejected
from cayu.observability.events import EventSink
from cayu.providers import ModelStreamEvent
from cayu.runtime._model_completion_contracts import ModelCompletionBoundaryReconciliation
from cayu.sessions import InMemorySessionStore

_SESSION_ID = "resume-settlement-exits"


def _answers(count):
    return [
        [ModelStreamEvent.text_delta(f"answer {n}"), ModelStreamEvent.completed()]
        for n in range(count)
    ]


def _app(provider, *, event_sinks=(), environment_factory=None):
    app = CayuApp(
        session_store=InMemorySessionStore(),
        event_sinks=list(event_sinks),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    if environment_factory is not None:
        app.register_environment_factory(
            _portable_environment_spec("dynamic"), environment_factory, default=True
        )
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
    return app


async def _run(app, text="first question"):
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=_SESSION_ID,
                messages=[Message.text("user", text)],
            )
        )
    ]
    assert events[-1].type is EventType.SESSION_COMPLETED
    return events


def _resume(app, text):
    return app.resume(ResumeRequest(session_id=_SESSION_ID, messages=[Message.text("user", text)]))


async def _resume_completes(app):
    events = [event async for event in _resume(app, "after the exit")]
    assert events[-1].type is EventType.SESSION_COMPLETED


class _ForgedRefusalSink(EventSink):
    """Extension code that claims the runtime refused this interaction's publication."""

    def __init__(self):
        self.armed = False

    async def emit(self, event):
        if self.armed and event.type is EventType.INTERACTION_STARTED:
            self.armed = False
            forged = InteractionLifecyclePublicationRejected(
                session_id=event.session_id, interaction_id=event.interaction_id
            )
            raise asyncio.CancelledError("sink stopped") from forged


def test_a_forged_publication_refusal_does_not_keep_a_resume_open():
    async def scenario():
        sink = _ForgedRefusalSink()
        app = _app(ScriptedModelProvider(_answers(2)), event_sinks=[sink])
        await _run(app)
        sink.armed = True
        with pytest.raises(asyncio.CancelledError):
            async for _ in _resume(app, "second question"):
                pass
        session = await app.session_store.load(_SESSION_ID)
        assert session.status is SessionStatus.INTERRUPTED
        await _resume_completes(app)

    asyncio.run(scenario())


@pytest.mark.parametrize("state", ["provider_operation_pending", "provider_operation_unavailable"])
def test_closing_a_resume_at_its_provider_pause_releases_the_fence(monkeypatch, state):
    original = model_completion_recovery.ModelCompletionRecovery.reconcile_model_completion_boundary
    paused = True

    async def pending_provider(self, session, **kwargs):
        if not paused:
            return await original(self, session, **kwargs)
        return ModelCompletionBoundaryReconciliation(state=state, session=session)

    monkeypatch.setattr(
        model_completion_recovery.ModelCompletionRecovery,
        "reconcile_model_completion_boundary",
        pending_provider,
    )

    async def scenario():
        nonlocal paused
        app = _app(ScriptedModelProvider(_answers(2)))
        paused = False
        await _run(app)
        paused = True
        seen = []
        async with contextlib.aclosing(_resume(app, "second question")) as stream:
            async for event in stream:
                seen.append(event.type)
                if event.type is EventType.INTERACTION_INTERRUPTED:
                    break
        assert seen[-1] is EventType.INTERACTION_INTERRUPTED
        session = await app.session_store.load(_SESSION_ID)
        assert session.status is SessionStatus.INTERRUPTED
        paused = False
        await _resume_completes(app)

    asyncio.run(scenario())


class _BoundWorkspaceFactory(EnvironmentFactory):
    def __init__(self, root):
        self.root = root

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.root.mkdir(exist_ok=True)
        return EnvironmentFactoryResult(
            Environment(
                _portable_environment_spec(request.environment_name),
                workspace=LocalWorkspace(self.root, workspace_id="factory-workspace"),
                binding=DeterministicWorkspaceBinding(),
            )
        )


def test_closing_a_completed_resume_releases_its_bound_factory_environment(tmp_path):
    async def scenario():
        app = _app(
            ScriptedModelProvider(_answers(3)),
            environment_factory=_BoundWorkspaceFactory(tmp_path / "workspace"),
        )
        await _run(app)
        async with contextlib.aclosing(_resume(app, "second question")) as stream:
            async for event in stream:
                if event.type is EventType.INTERACTION_COMPLETED:
                    break
        session = await app.session_store.load(_SESSION_ID)
        assert session.status is SessionStatus.COMPLETED
        events = await app.session_store.load_events(_SESSION_ID)
        assert events[-1].type is EventType.SESSION_COMPLETED
        await _resume_completes(app)

    asyncio.run(scenario())


class _LostAdmissionStore(InMemorySessionStore):
    """Commits a resume admission, loses its acknowledgement, then fails the replay read."""

    invocation_lifecycle_command_version = 1

    def __init__(self):
        super().__init__()
        self.armed = False
        self.fail_next_checkpoint_read = False

    async def admit_session_invocation(self, *args, **kwargs):
        session = await super().admit_session_invocation(*args, **kwargs)
        if self.armed:
            self.armed = False
            self.fail_next_checkpoint_read = True
            raise OSError("admission acknowledgement lost")
        return session

    async def load_checkpoint(self, session_id):
        if self.fail_next_checkpoint_read:
            self.fail_next_checkpoint_read = False
            raise ConnectionError("receipt replay temporarily unavailable")
        return await super().load_checkpoint(session_id)


def test_a_committed_admission_whose_replay_read_fails_is_settled():
    async def scenario():
        store = _LostAdmissionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider(_answers(2)), default=True)
        app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
        await _run(app)
        before = await store.load(_SESSION_ID)
        store.armed = True
        with pytest.raises(ConnectionError):
            async for _ in _resume(app, "second question"):
                pass
        session = await store.load(_SESSION_ID)
        assert session.run_epoch > before.run_epoch + 1
        assert session.status is SessionStatus.INTERRUPTED
        await _resume_completes(app)

    asyncio.run(scenario())
