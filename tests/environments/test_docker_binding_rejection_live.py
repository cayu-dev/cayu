from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_docker_coding_live import _configuration_or_skip

from cayu import (
    AgentSpec,
    CayuApp,
    DockerCodingEnvironmentFactory,
    DockerImageIdentity,
    EnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentSpec,
    ExecutionProfileBehaviorIdentity,
    LocalWorkspace,
    Message,
    ModelProvider,
    ModelStreamEvent,
    RunRequest,
    SQLiteSessionStore,
    SyncBinding,
)

pytestmark = pytest.mark.process


def test_rejected_binding_reaps_only_its_new_docker_allocation(tmp_path: Path) -> None:
    docker, image, digest = _configuration_or_skip()
    architecture = subprocess.check_output(
        [docker, "image", "inspect", "--format", "{{.Architecture}}", image], text=True
    ).strip()
    source = tmp_path / "source"
    source.mkdir()
    (source / "code.py").write_text("value = 1\n")
    unrelated_source = tmp_path / "unrelated"
    unrelated_source.mkdir()
    factories = []

    class Factory(EnvironmentFactory):
        def __init__(self, source):
            self.inner = DockerCodingEnvironmentFactory(
                source_workspace=LocalWorkspace(source),
                toolchain_profile=docker_toolchain_profile(
                    image_identity=DockerImageIdentity(reference=image, content_digest=digest),
                    platform_architecture=architecture,
                ),
                docker_path=docker,
            )
            self.results = []
            self.creates = 0
            self.reaps = 0
            factories.append(self)

        @property
        def execution_profile_identity(self):
            return self.inner.execution_profile_identity

        def construction_admission_candidate(self):
            return self.inner.construction_admission_candidate()

        def execution_admission_candidate(self, request):
            return self.inner.execution_admission_candidate(request)

        def allocation_scope(self, request):
            return self.inner.allocation_scope(request)

        async def recover_finalization_disposal(self, request, state):
            await self.inner.recover_finalization_disposal(request, state)

        async def create(self, request):
            raise AssertionError("Cleanup must not reconnect or provision again")

        async def create_recoverable(self, request, allocation):
            self.creates += 1
            result = await self.inner.create_recoverable(request, allocation)
            assert isinstance(result.environment.binding, SyncBinding)
            result.environment.binding.sync_back = "never"
            self.results.append(result)
            return result

        async def reap_allocation(self, request, allocation):
            self.reaps += 1
            await self.inner.reap_allocation(request, allocation)

    def exists(container_id):
        return (
            subprocess.run(
                [docker, "container", "inspect", container_id], capture_output=True
            ).returncode
            == 0
        )

    async def scenario():
        release_owners = asyncio.Event()
        entered = [asyncio.Event(), asyncio.Event(), asyncio.Event()]
        calls = [0, 0, 0]
        apps = []
        stores = []

        class Provider(ModelProvider):
            name = "binding-rejection-fixture"

            def __init__(self, index):
                self.index = index

            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name=self.name, behavior_version="1", implementation_version="1"
                )

            async def stream(self, request):
                calls[self.index] += 1
                entered[self.index].set()
                await release_owners.wait()
                yield ModelStreamEvent.text_delta("done")
                yield ModelStreamEvent.completed()

        for index, root in enumerate((source, unrelated_source, source)):
            store = SQLiteSessionStore(tmp_path / f"sessions-{index}.db")
            stores.append(store)
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(Provider(index), default=True)
            app.register_agent(AgentSpec(name="probe", model="fixture"))
            app.register_environment_factory(
                EnvironmentSpec(name="docker"), Factory(root), default=True
            )
            apps.append(app)

        async def run(index):
            return [
                event
                async for event in apps[index].run(
                    RunRequest(
                        agent_name="probe",
                        session_id=f"session-{index}",
                        messages=[Message.text("user", "finish")],
                        max_steps=1,
                    )
                )
            ]

        owners = [asyncio.create_task(run(index)) for index in (0, 1)]
        try:
            async with asyncio.timeout(60):
                await asyncio.gather(entered[0].wait(), entered[1].wait())
                await run(2)
            ids = [factory.results[0].metadata["container_id"] for factory in factories]
            events = await stores[2].load_events("session-2")
            types = [event.type for event in events]
            assert "environment.factory.completed" in types
            assert "session.failed" in types
            assert any(
                "already bound by an active session" in str(event.payload) for event in events
            )
            assert calls == [1, 1, 0]
            failure = next(event for event in events if event.type == "environment.binding.failed")
            assert failure.payload["environment_factory_release"]["action"] == "discard"
            assert failure.payload["environment_factory_release"]["completed"] is True
            for _ in range(2):
                for drain in (
                    apps[2].drain_background_interruptions,
                    apps[2].drain_provider_operation_cancellations,
                    apps[2].drain_knowledge_publications,
                    apps[2].drain_recovery_cleanups,
                    apps[2].drain_environment_cleanups,
                ):
                    assert await drain(timeout_s=5)
                assert not exists(ids[2])
                assert exists(ids[0]) and exists(ids[1])
            checkpoint = await stores[2].load_checkpoint("session-2")
            record = checkpoint["environment_factory_allocation_intents"]["docker"]
            assert record["state"] == "reaped"
            assert record["reconnect_metadata"]["container_id"] == ids[2]
            completed = next(
                event for event in events if event.type == "environment.factory.completed"
            )
            assert completed.payload["allocation_id"] == record["intent"]["allocation_id"]
            assert (await stores[2].load("session-2")).status == "failed"
            assert not checkpoint.get("environment_factory_allocation_receipts")
            assert not checkpoint.get("environment_factory_reconnect")
            assert [factory.creates for factory in factories] == [1, 1, 1]
            assert [factory.reaps for factory in factories] == [0, 0, 1]
            release_owners.set()
            await asyncio.gather(*owners)
            for app in apps:
                assert await app.drain_environment_cleanups(timeout_s=5)
            assert all(not exists(container_id) for container_id in ids)
            assert (source / "code.py").read_text() == "value = 1\n"
        finally:
            release_owners.set()
            await asyncio.gather(*owners, return_exceptions=True)
            # Test hygiene only: all physical-absence assertions precede this fallback.
            for factory in factories:
                for result in factory.results:
                    await result.release(EnvironmentFactoryReleaseAction.DISCARD)
            for store in stores:
                await store.close()

    asyncio.run(scenario())
