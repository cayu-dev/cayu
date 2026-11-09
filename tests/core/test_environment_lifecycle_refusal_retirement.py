"""Refusal tombstones must not mask a new allocation's live authority."""

import asyncio
from dataclasses import replace

import pytest
from tests.core.test_environment_lifecycle import _lifecycle
from tests.runtime.test_execution_admission_dispatch import (
    _EvidenceRunner,
    _RecordingProvider,
    _RecoveringFactory,
    _requirements,
    _run,
)

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.environments import Environment, EnvironmentSpec
from cayu.environments.factory import (
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryResult,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity


@pytest.mark.parametrize("metadata", [{}, {"allocation_id": "allocation-2"}])
@pytest.mark.parametrize("lose_acknowledgement", [False, True])
def test_repeated_refusals_with_reused_metadata(metadata, lose_acknowledgement):
    async def scenario():
        class ReusingFactory(_RecoveringFactory):
            async def create(self, request):
                assert request.operation is EnvironmentFactoryOperation.CREATE
                self.first_runner = self.second_runner = _EvidenceRunner(
                    "hosted-b" if len(self.requests) < 2 else "hosted-a"
                )
                return replace(await super().create(request), reconnect_metadata=metadata)

        class RetirementStore(InMemorySessionStore):
            invocation_lifecycle_command_version = 1

            def __init__(self):
                super().__init__()
                self.retirements = 0

            async def transform_checkpoint(self, session_id, checkpoint_transform):
                previous = await self.load_checkpoint(session_id)
                await super().transform_checkpoint(session_id, checkpoint_transform)
                checkpoint = await self.load_checkpoint(session_id)
                if (
                    "hosted" in (previous or {}).get("environment_factory_reconnect", {})
                    and (checkpoint or {}).get("environment_factory_retired_disposals")
                    and "environment_factory_reconnect" not in checkpoint
                    and "environment_factory_allocation_owner" not in checkpoint
                ):
                    self.retirements += 1
                    if lose_acknowledgement:
                        # A store may retry the transform after commit, then lose
                        # the acknowledgement. Both paths must remain idempotent.
                        await super().transform_checkpoint(session_id, checkpoint_transform)
                        raise ConnectionError("retirement acknowledgement lost")

        factory = ReusingFactory()
        store = RetirementStore()
        provider = _RecordingProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(EnvironmentSpec(name="hosted"), factory, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        session_id = "reused-refusal-metadata"
        events = await _run(app, session_id)
        for attempt in range(2):
            assert events[-1].type is EventType.SESSION_FAILED
            assert factory.release_actions == [EnvironmentFactoryReleaseAction.DISCARD] * (
                attempt + 1
            )
            checkpoint = await store.load_checkpoint(session_id)
            assert "environment_factory_reconnect" not in checkpoint
            assert "environment_factory_allocation_owner" not in checkpoint
            assert checkpoint["environment_factory_retired_disposals"]["hosted"] == {
                "reason": "admission_refusal",
                "reconnect_metadata": metadata,
            }
            events = [
                event
                async for event in app.resume(
                    ResumeRequest(session_id=session_id, messages=[Message.text("user", "retry")])
                )
            ]

        assert [request.operation.value for request in factory.requests] == ["create"] * 3
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 1
        assert store.retirements == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("conflict", ["metadata", "owner", "instance", "epoch", "no-commit"])
def test_equal_refusal_tombstone_does_not_acknowledge_live_authority(conflict):
    async def scenario():
        class RetirementStore(InMemorySessionStore):
            async def transform_checkpoint(self, session_id, checkpoint_transform):
                if conflict == "no-commit":
                    raise ConnectionError("retirement did not commit")
                await super().transform_checkpoint(session_id, checkpoint_transform)

        store = RetirementStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant", session_id="refusal", messages=[Message.text("user", "go")]
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        metadata = {"allocation_id": "stable"}
        checkpoint = {
            "environment_factory_retired_disposals": {
                "hosted": {"reason": "admission_refusal", "reconnect_metadata": metadata}
            },
            "environment_factory_reconnect": {
                "hosted": {"allocation_id": "successor"} if conflict == "metadata" else metadata
            },
            "environment_factory_allocation_owner": {
                "hosted": "other-session" if conflict == "owner" else session.id
            },
        }
        await store.checkpoint(session.id, checkpoint)
        if conflict == "instance":
            session = session.model_copy(update={"instance_id": "stale-instance"})
        elif conflict == "epoch":
            session = session.model_copy(update={"run_epoch": session.run_epoch + 1})
        spec = EnvironmentSpec(name="hosted")
        environment = Environment(spec)
        registered = runtime_records.RegisteredEnvironment(
            spec=spec,
            environment=environment,
            unclaimed_factory_result=EnvironmentFactoryResult(
                environment=environment, reconnect_metadata=metadata
            ),
        )
        error = RuntimeError("admission refused")
        assert not await _lifecycle(store)._retire_rejected_unbound_allocation(
            session=session, registered_environment=registered, error=error
        )
        assert await store.load_checkpoint(session.id) == checkpoint
        assert any("could not be retired" in note for note in error.__notes__)

    asyncio.run(scenario())
