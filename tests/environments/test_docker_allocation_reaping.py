"""Exact-resource disposal and ambiguous-create contracts for Docker recovery."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from tests.docker_toolchain import docker_toolchain_profile

from cayu import (
    DockerCodingEnvironmentFactory,
    DockerImageIdentity,
    EnvironmentAllocationState,
    EnvironmentFactoryRequest,
    ImmutableInputStore,
    InMemorySessionStore,
    LocalWorkspace,
    Message,
    RunRequest,
    inspect_local_immutable_input,
)
from cayu.environments.docker_coding import _docker_coding_reconnect_metadata
from cayu.runners import DockerRunner, ExecResult
from cayu.runtime._environment_allocation import EnvironmentAllocationCoordinator
from cayu.runtime.sessions import SessionIdentity
from cayu.vaults import SecretRedactor

_ID = "a" * 64
_OTHER_ID = "b" * 64


async def _allocation(tmp_path: Path, state: EnvironmentAllocationState):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "input").write_text("immutable")
    image = DockerImageIdentity(reference="fixture:latest", content_digest="sha256:" + "c" * 64)
    projection = inspect_local_immutable_input(
        source,
        target_path="/evidence",
        policy_fingerprint="sha256:" + "d" * 64,
        runtime_compatibility_fingerprint=image.fingerprint,
        authorization_scope_fingerprint="sha256:" + "e" * 64,
    )
    inputs = ImmutableInputStore(tmp_path / "inputs")
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(workspace),
        toolchain_profile=docker_toolchain_profile(image_identity=image),
        docker_path="/usr/bin/docker",
        immutable_inputs=(projection,),
        immutable_input_store=inputs,
        immutable_input_runtime_compatibility_fingerprint=image.fingerprint,
    )
    store = InMemorySessionStore()
    await store.create(
        RunRequest(agent_name="agent", session_id="session", messages=[Message.text("user", "go")]),
        identity=SessionIdentity(provider_name="fixture", model="fixture"),
    )
    request = EnvironmentFactoryRequest(
        session_id="session", agent_name="agent", environment_name="coding"
    )
    coordinator = EnvironmentAllocationCoordinator(
        session_store=store,
        checkpoint_transform=lambda value: lambda _session, _current: value,
        secret_redactor=SecretRedactor(),
    )
    scope = factory.allocation_scope(request)
    allocation = coordinator.context(
        session_id="session",
        environment_name="coding",
        inherited_owner_session_id=None,
        scope=scope,
        existing=None,
    )
    await allocation.prepare(
        {
            "container_name": "cayu-coding-" + allocation.intent.allocation_id,
            "configuration_fingerprint": factory._configuration_fingerprint,
        }
    )
    inputs.attach_sync(
        projection, attachment_id="attachment", owner_id=allocation.intent.allocation_id
    )
    if state is not EnvironmentAllocationState.PREPARED:
        await allocation.mark_dispatched()
    if state is EnvironmentAllocationState.ACKNOWLEDGED:
        await allocation.acknowledge(
            _docker_coding_reconnect_metadata(
                container_id=_ID,
                configuration_fingerprint=factory._configuration_fingerprint,
                image_fingerprint=image.fingerprint,
                toolchain_profile_fingerprint=factory.toolchain_profile.fingerprint,
                allocation_id=allocation.intent.allocation_id,
            )
        )

    async def reload():
        return coordinator.context(
            session_id="session",
            environment_name="coding",
            inherited_owner_session_id=None,
            scope=scope,
            existing=await coordinator.load_record(session_id="session", environment_name="coding"),
        )

    return factory, request, allocation, inputs, reload


@pytest.mark.parametrize("failure", ["unavailable", "cancel_after_delete", "replacement"])
def test_acknowledged_cleanup_retries_only_the_exact_container(
    tmp_path: Path, monkeypatch, failure
):
    async def run():
        factory, request, allocation, inputs, reload = await _allocation(
            tmp_path, EnvironmentAllocationState.ACKNOWLEDGED
        )
        resources = {_ID, _OTHER_ID}
        calls = []

        async def exists(container_id, **_kwargs):
            calls.append(("exists", container_id))
            return container_id in resources

        async def close(runner):
            # The public runner's execution target is permanently bound to the
            # acknowledged ID; a reusable name never participates in disposal.
            calls.append(("close", runner.name))
            assert runner.name == _ID
            if failure == "unavailable" and calls.count(("close", _ID)) == 1:
                raise RuntimeError("daemon unavailable")
            resources.discard(_ID)
            if failure == "cancel_after_delete" and calls.count(("close", _ID)) == 1:
                raise asyncio.CancelledError("original cancellation")

        monkeypatch.setattr(DockerRunner, "container_exists", exists)
        monkeypatch.setattr(DockerRunner, "close", close)
        if failure == "replacement":
            resources.remove(_ID)
        else:
            error = RuntimeError if failure == "unavailable" else asyncio.CancelledError
            with pytest.raises(error):
                await factory.reap_allocation(request, allocation)
            assert inputs.inspect()[0].reference_count == 1
            assert (await reload()).state is EnvironmentAllocationState.REAPING
        await factory.reap_allocation(request, await reload())
        assert resources == {_OTHER_ID}
        assert inputs.inspect()[0].reference_count == 0
        assert (await reload()).state is EnvironmentAllocationState.REAPED
        before = list(calls)
        await factory.reap_allocation(request, await reload())
        assert calls == before

    asyncio.run(run())


def test_prepared_cleanup_fences_dispatch_and_retries_reference_release(
    tmp_path: Path, monkeypatch
):
    async def run():
        factory, request, allocation, inputs, reload = await _allocation(
            tmp_path, EnvironmentAllocationState.PREPARED
        )
        stale_creator = await reload()
        original = inputs.release_allocation_sync

        def unavailable(_allocation_id):
            raise RuntimeError("input ledger unavailable")

        monkeypatch.setattr(inputs, "release_allocation_sync", unavailable)
        with pytest.raises(RuntimeError, match="ledger unavailable"):
            await factory.reap_allocation(request, allocation)
        pending = await reload()
        assert pending.dispatch_precluded
        assert pending.state is EnvironmentAllocationState.REAPING
        with pytest.raises(RuntimeError):
            await stale_creator.mark_dispatched()
        with pytest.raises(RuntimeError):
            await pending.acknowledge({"container_id": _ID})
        assert inputs.inspect()[0].reference_count == 1
        monkeypatch.setattr(inputs, "release_allocation_sync", original)
        await factory.reap_allocation(request, pending)
        assert inputs.inspect()[0].reference_count == 0
        assert (await reload()).state is EnvironmentAllocationState.REAPED
        assert (await reload()).dispatch_precluded

    asyncio.run(run())


@pytest.mark.parametrize("initial_lookup", ["absent", "foreign"])
def test_unacknowledged_cleanup_requires_matching_creation_identity(
    tmp_path: Path, monkeypatch, initial_lookup
):
    async def run():
        factory, request, allocation, inputs, reload = await _allocation(
            tmp_path, EnvironmentAllocationState.DISPATCHED
        )
        present = initial_lookup != "absent"
        owned = False
        calls = []

        async def lookup(_name, **_kwargs):
            return _ID if present else None

        async def identity(container_id, **_kwargs):
            assert container_id == _ID
            if not owned:
                raise RuntimeError("foreign allocation identity")

        async def exists(container_id, **_kwargs):
            return container_id == _ID

        async def close(runner):
            calls.append(runner.name)

        monkeypatch.setattr(DockerRunner, "resolve_container_id", lookup)
        monkeypatch.setattr(DockerRunner, "require_allocation_identity", identity)
        monkeypatch.setattr(DockerRunner, "container_exists", exists)
        monkeypatch.setattr(DockerRunner, "close", close)
        with pytest.raises(RuntimeError):
            await factory.reap_allocation(request, allocation)
        assert not calls
        assert inputs.inspect()[0].reference_count == 1
        assert (await reload()).state is EnvironmentAllocationState.DISPATCHED
        present = owned = True
        await factory.reap_allocation(request, await reload())
        assert calls == [_ID]
        assert inputs.inspect()[0].reference_count == 0
        assert (await reload()).acknowledged_reconnect_metadata["container_id"] == _ID

    asyncio.run(run())


@pytest.mark.parametrize("label", [None, "sha256:" + "e" * 64, "sha256:" + "f" * 64])
def test_docker_allocation_label_is_checked_by_exact_id_without_guest_execution(monkeypatch, label):
    calls = []

    async def subprocess(command, **_kwargs):
        calls.append(command.argv)
        return ExecResult(
            stdout=json.dumps(
                {
                    "Id": _ID,
                    "Config": {
                        "Labels": {} if label is None else {"io.cayu.allocation-identity": label}
                    },
                }
            )
        )

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", subprocess)
    expected = "sha256:" + "f" * 64

    async def run():
        if label == expected:
            await DockerRunner.require_allocation_identity(
                _ID, allocation_identity=expected, docker_path="/usr/bin/docker"
            )
        else:
            with pytest.raises(RuntimeError):
                await DockerRunner.require_allocation_identity(
                    _ID, allocation_identity=expected, docker_path="/usr/bin/docker"
                )

    asyncio.run(run())
    assert calls == [["/usr/bin/docker", "inspect", "--format", "{{json .}}", _ID]]
