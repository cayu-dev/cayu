"""Pre-promotion exit backstops end with their exact allocation ownership."""

from __future__ import annotations

import asyncio
import weakref
from dataclasses import replace

import pytest
from tests.core.test_environment_allocation_recovery import _FakeRemoteFactory, _FakeRemoteProvider
from tests.core.test_environment_lifecycle import _collect_events, _FakeProvider
from tests.environments.test_process_local_allocations import _collect_until_gone

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.environments import Environment, EnvironmentSpec
from cayu.environments.factory import EnvironmentFactory, EnvironmentFactoryResult
from cayu.providers import ModelStreamEvent
from cayu.runners.base import Runner
from cayu.sessions import InMemorySessionStore
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest


class _Runner(Runner):
    async def exec(self, *args, **kwargs):
        raise AssertionError("No commands expected")


class _Factory(EnvironmentFactory):
    def __init__(self):
        self.runners = []
        self.disposed = False

    async def create(self, request):
        # Reconnect deliberately creates a different runner for the same resource.
        runner = _Runner()
        self.runners.append(weakref.ref(runner))
        return EnvironmentFactoryResult(
            Environment(EnvironmentSpec(name=request.environment_name), runner=runner),
            reconnect_metadata={"allocation_id": "exact-allocation"},
        )

    async def is_allocation_disposed(self, request):
        return self.disposed


def _app(monkeypatch, factory=None):
    store = InMemorySessionStore()
    factory = factory or _Factory()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        _FakeProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]] * 2),
        default=True,
    )
    app.register_environment_factory(EnvironmentSpec(name="remote"), factory, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    emit = app._event_writer.emit
    fail = [True]

    async def emit_completion(event):
        if fail[0] and event.type is EventType.ENVIRONMENT_FACTORY_COMPLETED:
            raise ConnectionError("publication unavailable")
        return await emit(event)

    monkeypatch.setattr(app._event_writer, "emit", emit_completion)
    return app, factory, fail


async def _fail_publication(app, fail):
    fail[0] = False
    events = await _collect_events(
        app,
        RunRequest(
            agent_name="assistant", session_id="preserved", messages=[Message.text("user", "go")]
        ),
    )
    assert events[-1].type is EventType.SESSION_COMPLETED
    fail[0] = True
    events = [
        event
        async for event in app.resume(
            ResumeRequest(session_id="preserved", messages=[Message.text("user", "retry")])
        )
    ]
    assert events[-1].type is EventType.SESSION_FAILED
    assert app._environment_lifecycle._preserved_factory_runners
    assert len(app._environment_lifecycle._exit_hook()) == 1


def test_reconnect_promotion_retires_old_preserved_runner(monkeypatch):
    async def scenario():
        app, factory, fail = _app(monkeypatch)
        await _fail_publication(app, fail)
        old_runner = factory.runners[-1]
        events = [
            event
            async for event in app.resume(
                ResumeRequest(session_id="preserved", messages=[Message.text("user", "retry")])
            )
        ]
        assert events[-1].type is EventType.SESSION_FAILED
        assert len(app._environment_lifecycle._exit_hook()) == 2
        fail[0] = False
        events = [
            event
            async for event in app.resume(
                ResumeRequest(session_id="preserved", messages=[Message.text("user", "retry")])
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(factory.runners) == 4
        assert not app._environment_lifecycle._preserved_factory_runners
        return old_runner

    _collect_until_gone(asyncio.run(scenario()))


@pytest.mark.parametrize("retirement", ["pending-disposal", "terminal"])
@pytest.mark.parametrize("exact", [False, True])
def test_disposal_retires_only_exact_preserved_allocation(monkeypatch, retirement, exact):
    async def scenario():
        app, factory, fail = _app(monkeypatch)
        await _fail_publication(app, fail)
        lifecycle = app._environment_lifecycle
        store = lifecycle._session_store
        session = await store.load("preserved")
        checkpoint = await store.load_checkpoint(session.id)
        reconnect = checkpoint["environment_factory_reconnect"]["remote"]
        old_runner = factory.runners[-1]
        if not exact:
            # Another allocation's successful disposal is not evidence about
            # the resource still covered by the original runner.
            reconnect = {"allocation_id": "another-allocation"}
            checkpoint["environment_factory_reconnect"]["remote"] = reconnect
        if retirement == "pending-disposal":
            pending = {"reconnect_metadata": reconnect, "state": {"version": 1}}
            checkpoint["environment_factory_pending_disposals"] = {"remote": pending}
            await store.checkpoint(session.id, checkpoint)
            await lifecycle._retire_disposed_allocation(
                session.id,
                "remote",
                pending,
                session_instance_id=session.instance_id,
                run_epoch=session.run_epoch,
            )
        else:
            await store.checkpoint(session.id, checkpoint)
            factory.disposed = True
            await store.update_status(session.id, SessionStatus.RUNNING)
            session = await store.load(session.id)
            await lifecycle._retire_terminal_allocation(
                session=session,
                registered_agent=app._agents["assistant"],
                environment_name="remote",
                factory=factory,
            )
        assert bool(lifecycle._preserved_factory_runners) is not exact
        assert len(lifecycle._exit_hook()) == (0 if exact else 1)
        return old_runner if exact else None

    old_runner = asyncio.run(scenario())
    if old_runner is not None:
        _collect_until_gone(old_runner)


def test_pending_publication_reaping_retires_preserved_runner(monkeypatch):
    class Factory(_FakeRemoteFactory):
        runner_ref = None

        async def reap_allocation(self, request, allocation):
            assert await allocation.mark_reaping()
            intent = allocation.intent
            self._provider.reap(
                intent.provider_metadata["resource_name"],
                allocation_id=intent.allocation_id,
                session_id=intent.session_id,
                environment_name=intent.environment_name,
                adapter_generation=intent.adapter_generation,
            )
            await allocation.mark_reaped()

        def _result(self, request, resource, *, allocation=None):
            result = super()._result(request, resource, allocation=allocation)
            runner = _Runner()
            self.runner_ref = weakref.ref(runner)
            return replace(result, environment=Environment(result.environment.spec, runner=runner))

    async def scenario():
        remote = _FakeRemoteProvider()
        factory = Factory(remote)
        app, _, _ = _app(monkeypatch, factory)
        events = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="preserved",
                messages=[Message.text("user", "go")],
            ),
        )
        assert events[-1].type is EventType.SESSION_FAILED
        assert app._environment_lifecycle._preserved_factory_runners
        assert len(remote.resources) == 1
        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="preserved")
        )
        assert not remote.resources, recovery
        assert not app._environment_lifecycle._preserved_factory_runners
        assert not app._environment_lifecycle._exit_hook()
        return factory.runner_ref

    _collect_until_gone(asyncio.run(scenario()))
