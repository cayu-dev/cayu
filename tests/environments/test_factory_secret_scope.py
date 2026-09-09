from __future__ import annotations

import asyncio
from typing import Literal

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    EventType,
    ExecutionProfileBehaviorIdentity,
    ExecutionProfileComponentClass,
    Message,
    ModelStreamEvent,
    PassthroughProxy,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    StaticVault,
)
from cayu.runtime._invocation_secrets import (
    registered_environment_secret_resolution_scope,
    require_continuation_secret_resolution_compatibility,
)


class _Factory(EnvironmentFactory):
    scope: Literal["static", "dynamic"] = "dynamic"

    @property
    def secret_resolution_scope(self) -> Literal["static", "dynamic"]:
        return self.scope

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="test-factory", behavior_version="1", implementation_version="1"
        )

    def __init__(self, *, capability: str | None = None) -> None:
        self.capability = capability
        self.operations = []
        self.releases = []

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.operations.append(request.operation)

        async def release(action):
            self.releases.append(action)

        return EnvironmentFactoryResult(
            Environment(
                EnvironmentSpec(name=request.environment_name),
                vault=StaticVault({"token": "private-value"})
                if self.capability == "vault"
                else None,
                proxy=PassthroughProxy(StaticVault({"token": "private-value"}))
                if self.capability == "proxy"
                else None,
            ),
            release=release,
            reconnect_metadata={"allocation": "test"},
        )


def test_factory_scope_is_snapshotted_and_undeclared_factories_stay_dynamic():
    factory = _Factory()
    app = CayuApp(enable_logging=False)
    app.register_environment_factory(EnvironmentSpec(name="managed"), factory)
    registration = app.list_environment_registrations()[0]
    assert registered_environment_secret_resolution_scope(registration) == "dynamic"
    factory.scope = "static"
    assert registered_environment_secret_resolution_scope(registration) == "dynamic"
    with pytest.raises(RuntimeError, match="cannot add dynamic secret resolution"):
        require_continuation_secret_resolution_compatibility("static", registration)


@pytest.mark.parametrize("capability", ["vault", "proxy"])
@pytest.mark.parametrize("reconnect", [False, True])
def test_static_factory_rejects_secret_capability_and_releases_result(capability, reconnect):
    factory = _Factory(capability=None if reconnect else capability)
    factory.scope = "static"
    app = CayuApp(enable_logging=False)
    app.register_environment_factory(EnvironmentSpec(name="managed"), factory, default=True)
    app.register_provider(ScriptedModelProvider([[ModelStreamEvent.completed()]]), default=True)
    app.register_agent(AgentSpec(name="probe", model="scripted"))
    registration = app.list_environment_registrations()[0]
    assert registered_environment_secret_resolution_scope(registration) == "static"

    async def run():
        if reconnect:
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="scope",
                        agent_name="probe",
                        messages=[Message.text("user", "done")],
                    )
                )
            ]
            assert not any(e.type is EventType.ENVIRONMENT_FACTORY_FAILED for e in initial)
            factory.capability = capability
        stream = (
            app.resume(ResumeRequest(session_id="scope", messages=[Message.text("user", "again")]))
            if reconnect
            else app.run(RunRequest(agent_name="probe", messages=[Message.text("user", "done")]))
        )
        events = [event async for event in stream]
        assert await app.drain_environment_cleanups(timeout_s=5)
        return events

    events = asyncio.run(run())
    assert any(event.type is EventType.ENVIRONMENT_FACTORY_FAILED for event in events)
    assert not any(event.type is EventType.ENVIRONMENT_BINDING_STARTED for event in events)
    assert factory.releases == [
        EnvironmentFactoryReleaseAction.PRESERVE
        if reconnect
        else EnvironmentFactoryReleaseAction.DISCARD
    ]
    assert factory.operations[-1] is (
        EnvironmentFactoryOperation.RECONNECT if reconnect else EnvironmentFactoryOperation.CREATE
    )
    assert "private-value" not in repr(events)


def test_factory_secret_scope_changes_durable_execution_profile():
    async def profile(scope):
        factory = _Factory()
        factory.scope = scope
        app = CayuApp(enable_logging=False)
        app.register_environment_factory(
            EnvironmentSpec(
                name="managed", execution_profile_identity=factory.execution_profile_identity
            ),
            factory,
            default=True,
        )
        app.register_provider(ScriptedModelProvider([[ModelStreamEvent.completed()]]), default=True)
        app.register_agent(AgentSpec(name="probe", model="scripted"))
        async for _ in app.run(
            RunRequest(
                session_id="profile", agent_name="probe", messages=[Message.text("user", "done")]
            )
        ):
            pass
        session = await app.session_store.load("profile")
        from cayu.runtime.execution_profiles import execution_profile_from_session_metadata

        return execution_profile_from_session_metadata(session.metadata).component(
            ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT
        )

    static = asyncio.run(profile("static"))
    assert static == asyncio.run(profile("static"))
    assert static != asyncio.run(profile("dynamic"))
