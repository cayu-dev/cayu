from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_docker_coding_live import _configuration_or_skip

from cayu import (
    AgentSpec,
    CayuApp,
    DockerCodingEnvironmentFactory,
    DockerImageIdentity,
    EnvironmentFactoryOperation,
    EnvironmentSpec,
    EventType,
    ForkSessionRequest,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
)
from cayu.runners import DockerRunner

pytestmark = pytest.mark.process


@pytest.mark.parametrize("copy_checkpoint", [True, False])
def test_completed_docker_fork_allocates_fresh_owned_execution(
    tmp_path: Path, copy_checkpoint: bool
) -> None:
    docker_path, image, image_id = _configuration_or_skip()
    source = tmp_path / "source"
    source.mkdir()
    requests = []
    allocations = []

    class RecordingFactory(DockerCodingEnvironmentFactory):
        def execution_admission_candidate(self, request):
            requests.append(request)
            return super().execution_admission_candidate(request)

        async def create_recoverable(self, request, allocation):
            result = await super().create_recoverable(request, allocation)
            allocations.append((allocation.intent, result.metadata["container_id"]))
            return result

    factory = RecordingFactory(
        source_workspace=LocalWorkspace(source),
        toolchain_profile=docker_toolchain_profile(
            image_identity=DockerImageIdentity(reference=image, content_digest=image_id),
            platform_architecture=subprocess.check_output(
                [docker_path, "image", "inspect", "--format", "{{.Architecture}}", image],
                text=True,
            ).strip(),
        ),
        docker_path=docker_path,
    )
    keyring = PublicAuthorityAliasKeyring(
        active_key_id="test",
        keys={"test": SecretStr("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")},
    )

    async def run():
        store = SQLiteSessionStore(
            tmp_path / "runtime.db", public_authority_alias_codec=PublicAuthorityAliasCodec(keyring)
        )
        app = CayuApp(
            session_store=store, public_authority_alias_keyring=keyring, enable_logging=False
        )
        provider = ScriptedModelProvider(
            [
                [ModelStreamEvent.text_delta("Parent complete."), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("Child complete."), ModelStreamEvent.completed()],
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="scripted"), tools=[])
        app.register_environment_factory(
            EnvironmentSpec(
                name="coding", execution_profile_identity=factory.execution_profile_identity
            ),
            factory,
            default=True,
        )
        try:
            async with asyncio.timeout(120):
                parent_events = [
                    event
                    async for event in app.run(
                        RunRequest(
                            session_id="parent",
                            agent_name="worker",
                            messages=[Message.text("user", "Finish this turn.")],
                            max_steps=1,
                        )
                    )
                ]
                assert EventType.SESSION_COMPLETED in {event.type for event in parent_events}
                assert len(provider.requests) == 1
                assert len(allocations) == 1
                assert not await DockerRunner.container_exists(
                    allocations[0][1], docker_path=docker_path
                )
                parent_checkpoint = await store.load_checkpoint("parent")
                parent = await store.load("parent")
                fork_events = [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id="parent",
                            session_id="child",
                            copy_checkpoint=copy_checkpoint,
                        )
                    )
                ]
                assert [event.type for event in fork_events] == [EventType.SESSION_FORKED]
                child_checkpoint = await store.load_checkpoint("child")
                if copy_checkpoint:
                    assert child_checkpoint is not None and parent_checkpoint is not None
                    for key in (
                        "environment_factory_reconnect",
                        "environment_factory_allocation_receipts",
                    ):
                        assert child_checkpoint[key] == parent_checkpoint[key]
                child_events = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="child",
                            messages=[Message.text("user", "Finish the next turn.")],
                            max_steps=1,
                        )
                    )
                ]
                assert EventType.SESSION_COMPLETED in {event.type for event in child_events}, (
                    child_events
                )
                assert len(provider.requests) == 2
                assert len(allocations) == 2
                assert [intent.session_id for intent, _ in allocations] == ["parent", "child"]
                assert allocations[0][0].allocation_id != allocations[1][0].allocation_id
                assert allocations[0][1] != allocations[1][1]
                child = await store.load("child")
                assert parent is not None and child is not None
                assert parent.status.value == child.status.value == "completed"
                parent_snapshot = await app.snapshot_fork_source("parent")
                child_snapshot = await app.snapshot_fork_source("child")
                assert (
                    child_snapshot.execution_profile_fingerprint
                    == parent_snapshot.execution_profile_fingerprint
                )
                assert await store.load_checkpoint("parent") == parent_checkpoint
                child_requests = [request for request in requests if request.session_id == "child"]
                assert child_requests
                assert all(
                    request.operation is EnvironmentFactoryOperation.CREATE
                    for request in child_requests
                )
                assert all(request.reconnect_metadata == {} for request in child_requests)
        finally:
            try:
                for name in (
                    "drain_background_interruptions",
                    "drain_provider_operation_cancellations",
                    "drain_knowledge_publications",
                    "drain_recovery_cleanups",
                    "drain_environment_cleanups",
                ):
                    assert await getattr(app, name)(timeout_s=20)
                for _, container_id in allocations:
                    assert not await DockerRunner.container_exists(
                        container_id, docker_path=docker_path
                    )
            finally:
                await store.close()

    asyncio.run(run())
