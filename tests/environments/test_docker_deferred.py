"""Deferred Docker environments and the warm spare pool, without a Docker daemon."""

from __future__ import annotations

import asyncio
import atexit
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_docker_coding import _image_identity, _TestAllocationContext

import cayu.environments.docker_coding as docker_coding_module
import cayu.environments.warm_spares as warm_spares_module
from cayu.environments.admission import ExecutionRequirements
from cayu.environments.deferred import DeferredRunner, DeferredWorkspaceBinding
from cayu.environments.docker_coding import DockerCodingEnvironmentFactory
from cayu.environments.factory import (
    EnvironmentAllocationIntent,
    EnvironmentAllocationState,
    EnvironmentFactoryOperation,
    EnvironmentFactoryRequest,
)
from cayu.runners.docker import DockerRunner
from cayu.runners.docker_workload import DockerImageIdentity
from cayu.workspaces.local import LocalWorkspace


def _factory(tmp_path: Path, **options) -> DockerCodingEnvironmentFactory:
    return DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path, workspace_id="deferred-source"),
        toolchain_profile=docker_toolchain_profile(image_identity=_image_identity()),
        docker_path="/usr/bin/docker",
        git_baseline=False,
        **options,
    )


def _request(**updates) -> EnvironmentFactoryRequest:
    return EnvironmentFactoryRequest(
        session_id="deferred-session", agent_name="agent", environment_name="sandbox", **updates
    )


def test_deferred_options_are_validated_and_only_deferred_changes_identity(tmp_path) -> None:
    eager = _factory(tmp_path)
    deferred = _factory(tmp_path, deferred=True)

    assert eager.deferred_materialization is False
    assert deferred.deferred_materialization is True
    assert (
        eager.execution_profile_identity.implementation_version
        != deferred.execution_profile_identity.implementation_version
    )
    assert _factory(tmp_path).execution_profile_identity == eager.execution_profile_identity
    with pytest.raises(ValueError, match="requires deferred=True"):
        _factory(tmp_path, warm_spares=1)
    with pytest.raises(ValueError, match="warm_spares"):
        _factory(tmp_path, deferred=True, warm_spares=99)


def test_deferred_allocation_reserves_a_name_without_creating_a_container(
    tmp_path, monkeypatch
) -> None:
    async def must_not_create(*args, **kwargs):
        raise AssertionError("deferred creation must not start a container")

    monkeypatch.setattr(DockerRunner, "create", must_not_create)
    factory = _factory(tmp_path, deferred=True)
    request = _request()
    intent = EnvironmentAllocationIntent(
        allocation_id="ealloc_" + ("e" * 32),
        provider="docker",
        adapter_generation="cayu.docker_coding.v12",
        session_id=request.session_id,
        environment_name=request.environment_name,
        requested_operation=EnvironmentFactoryOperation.CREATE,
    )
    allocation = _TestAllocationContext(intent)

    result = asyncio.run(factory.create_recoverable(request, allocation))

    assert allocation.state is EnvironmentAllocationState.ACKNOWLEDGED
    metadata = allocation.acknowledged_reconnect_metadata
    assert metadata == result.reconnect_metadata
    assert metadata["kind"] == "docker_coding_deferred"
    assert metadata["container_name"] == "cayu-coding-ealloc_" + ("e" * 32)
    assert allocation.intent.provider_metadata["materialization"] == "deferred"
    assert isinstance(result.environment.runner, DeferredRunner)
    assert isinstance(result.environment.binding, DeferredWorkspaceBinding)
    assert result.environment.runner.execution_environment_authority() is (
        factory.execution_environment_authority()
    )

    reconnect = _request(
        operation=EnvironmentFactoryOperation.RECONNECT, reconnect_metadata=metadata
    )
    factory._validate_request(reconnect)
    with pytest.raises(ValueError):
        factory._validate_request(
            _request(
                operation=EnvironmentFactoryOperation.RECONNECT,
                reconnect_metadata={**metadata, "container_name": "someone-else"},
            )
        )


def test_deferred_disposal_and_reaping_follow_the_reserved_name(tmp_path, monkeypatch) -> None:
    factory = _factory(tmp_path, deferred=True)
    present: dict[str, str | None] = {"cayu-coding-x": "f" * 64}
    removed: list[str] = []

    async def resolve(name, *, docker_path=None):
        return present.get(name)

    async def close(self):
        removed.append(self.container_id)
        present["cayu-coding-x"] = None

    monkeypatch.setattr(DockerRunner, "resolve_container_id", resolve)
    monkeypatch.setattr(DockerRunner, "close", close)
    metadata = factory._deferred_reconnect_metadata("cayu-coding-x", None)
    request = _request(operation=EnvironmentFactoryOperation.RECONNECT, reconnect_metadata=metadata)

    async def run():
        before = await factory.is_allocation_disposed(request)
        await factory._remove_named_container("cayu-coding-x")
        after = await factory.is_allocation_disposed(request)
        return before, after

    assert asyncio.run(run()) == (False, True)
    assert removed == ["f" * 64]


class _FakeRunner:
    def __init__(self, container_id: str, *, unavailable: tuple[str, ...] = ()) -> None:
        self.container_id = container_id
        self.is_closed = False
        self.unavailable = unavailable

    async def close(self) -> None:
        self.is_closed = True

    def execution_capability_evidence(self):
        executables = tuple(
            SimpleNamespace(executable=name, state="unavailable") for name in self.unavailable
        )
        return SimpleNamespace(tool_requirements=SimpleNamespace(executables=executables))


class _FakeDocker:
    def __init__(self) -> None:
        self.running: set[str] = set()
        self.renamed: dict[str, str] = {}
        self.fail_rename = False
        self.unreachable = False
        self.removed: list[str] = []
        self.listed: list[str] = []
        self.listings = 0

    async def __call__(self, *arguments: str) -> tuple[int, str]:
        if self.unreachable:
            raise OSError("docker daemon socket unavailable")
        if arguments[0] == "inspect":
            return 0, "true" if arguments[-1] in self.running else "false"
        if arguments[0] == "rename":
            if self.fail_rename:
                return 1, ""
            self.renamed[arguments[1]] = arguments[2]
            return 0, ""
        if arguments[0] == "ps":
            self.listings += 1
            return 0, "\n".join(self.listed)
        if arguments[0] == "rm":
            self.removed.append(arguments[-1])
            return 0, ""
        raise AssertionError(arguments)


def _pool(tmp_path, monkeypatch, *, size=2, fail_creates=0):
    factory = _factory(tmp_path, deferred=True, warm_spares=size)
    pool = factory.warm_spare_pool
    assert pool is factory._warm_pool
    # Fake spares must never reach a real `docker rm` when the test run exits.
    atexit.unregister(pool._exit_hook)
    docker = _FakeDocker()
    monkeypatch.setattr(pool._backend, "_docker", docker)
    created: list[_FakeRunner] = []
    failures = {"left": fail_creates}

    async def create(name, *, immutable_mounts, requirements, allocation_identity=None):
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("docker daemon unavailable")
        runner = _FakeRunner(f"{len(created) + 1:064x}")
        docker.running.add(runner.container_id)
        created.append(runner)
        return runner

    monkeypatch.setattr(factory, "_create_or_recover_runner", create)
    probed: list[str] = []

    async def probe(runner, profile):
        probed.append(runner.container_id)

    monkeypatch.setattr(docker_coding_module, "_run_toolchain_admission_probes", probe)
    factory.probed = probed
    return factory, pool, docker, created


def _requirements(factory) -> ExecutionRequirements:
    return ExecutionRequirements.model_validate(
        {
            **ExecutionRequirements.trusted().model_dump(mode="python", warnings=False),
            "required_executables": tuple(sorted(factory.required_executables)),
        }
    )


def test_pool_hands_each_spare_to_one_run_and_refills(tmp_path, monkeypatch) -> None:
    factory, pool, docker, created = _pool(tmp_path, monkeypatch)
    requirements = _requirements(factory)

    async def run():
        pool.schedule_refill()
        await pool.wait_for_refill()
        assert pool.idle_count == 2
        first = await pool.take("cayu-coding-a", requirements)
        second = await pool.take("cayu-coding-b", requirements)
        await pool.wait_for_refill()
        third = await pool.take("cayu-coding-c", requirements)
        return first, second, third

    first, second, third = asyncio.run(run())

    handed = [spare.runner.container_id for spare in (first, second, third)]
    assert len(set(handed)) == 3, "a container is never handed out twice"
    assert docker.renamed[handed[0]] == "cayu-coding-a"
    assert docker.renamed[handed[1]] == "cayu-coding-b"
    assert len(created) >= 3
    # Spares pass the same toolchain admission probes a cold start runs.
    assert set(handed) <= set(factory.probed)


def test_pool_falls_back_cold_on_dead_spare_or_rename_failure(tmp_path, monkeypatch) -> None:
    factory, pool, docker, created = _pool(tmp_path, monkeypatch, size=1)
    requirements = _requirements(factory)

    async def run():
        pool.schedule_refill()
        await pool.wait_for_refill()
        docker.running.clear()  # the spare died while idle
        dead = await pool.take("cayu-coding-a", requirements)
        await pool.wait_for_refill()
        docker.fail_rename = True
        renamed = await pool.take("cayu-coding-b", requirements)
        return dead, renamed

    dead, renamed = asyncio.run(run())

    assert dead is None and renamed is None
    assert created[0].is_closed and created[1].is_closed  # discarded, never reused


def _with_tool_probes(requirements: ExecutionRequirements) -> ExecutionRequirements:
    return ExecutionRequirements.model_validate(
        {
            **requirements.model_dump(mode="python", warnings=False),
            "required_executables": ("git", *requirements.executable_names()),
        }
    )


def test_a_session_whose_tools_add_executables_claims_a_spare_through_strict_reconnect(
    tmp_path, monkeypatch
) -> None:
    factory, pool, docker, created = _pool(tmp_path, monkeypatch, size=1)
    session = _with_tool_probes(_requirements(factory))
    reconnected: list[tuple[str, tuple[str, ...]]] = []

    async def reconnect(container_id, *, immutable_mounts, requirements):
        reconnected.append((container_id, tuple(requirements.executable_names())))
        return _FakeRunner(container_id)

    monkeypatch.setattr(factory, "_reconnect_runner", reconnect)

    async def run():
        pool.schedule_refill()
        await pool.wait_for_refill()
        return await pool.take("cayu-coding-a", session)

    claimed = asyncio.run(run())

    spare_id = created[0].container_id
    assert claimed is not None and claimed.runner.container_id == spare_id
    assert docker.renamed[spare_id] == "cayu-coding-a"
    # The session's own executables are verified live on that exact container.
    assert reconnected == [(spare_id, tuple(session.executable_names()))]
    assert not created[0].is_closed


@pytest.mark.parametrize("failure", ["executable_unavailable", "daemon_hiccup"])
def test_a_spare_that_fails_the_session_s_extra_checks_is_removed(
    tmp_path, monkeypatch, failure
) -> None:
    factory, pool, _docker, created = _pool(tmp_path, monkeypatch, size=3)
    attempts: list[str] = []

    async def reconnect(container_id, *, immutable_mounts, requirements):
        attempts.append(container_id)
        if failure == "daemon_hiccup":
            raise RuntimeError("docker inspect timed out")
        return _FakeRunner(container_id, unavailable=("git",))

    monkeypatch.setattr(factory, "_reconnect_runner", reconnect)

    async def run():
        pool.schedule_refill()
        await pool.wait_for_refill()
        session = _with_tool_probes(_requirements(factory))
        first = await pool.take("cayu-coding-a", session)
        second = await pool.take("cayu-coding-b", session)
        return first, second

    # Cold start reports the real failure; one claim spends one spare, not the pool.
    assert asyncio.run(run()) == (None, None)
    assert created[0].is_closed
    if failure == "executable_unavailable":
        # The image cannot serve these requirements: later claims go cold at once.
        assert len(attempts) == 1
        assert not created[1].is_closed
    else:
        # A transient error is not remembered: the next claim tries a spare again.
        assert len(attempts) == 2
        assert created[1].is_closed


def test_pool_refill_backs_off_on_docker_errors(tmp_path, monkeypatch) -> None:
    _, pool, _docker, _created = _pool(tmp_path, monkeypatch, size=1, fail_creates=1)

    async def run():
        pool.schedule_refill()
        await pool.wait_for_refill()
        assert pool.idle_count == 0
        assert pool._next_attempt_at > time.monotonic()
        pool.schedule_refill()  # still backing off: no new attempt
        assert pool._refill_task is None or pool._refill_task.done()
        pool._next_attempt_at = 0.0
        pool.schedule_refill()
        await pool.wait_for_refill()
        return pool.idle_count

    assert asyncio.run(run()) == 1


def test_trimming_idle_spares_removes_them_and_the_next_use_refills(tmp_path, monkeypatch) -> None:
    factory, pool, _docker, created = _pool(tmp_path, monkeypatch, size=2)

    async def run():
        pool.schedule_refill()
        await pool.wait_for_refill()
        await pool.release_idle()
        released = pool.idle_count
        pool.schedule_refill()  # trimmed: nothing refills until the pool is used
        await pool.wait_for_refill()
        idle_after_trim = pool.idle_count
        assert await pool.take("cayu-coding-a", _requirements(factory)) is None
        await pool.wait_for_refill()
        return released, idle_after_trim, pool.idle_count

    assert asyncio.run(run()) == (0, 0, 2)
    assert created[0].is_closed and created[1].is_closed


def test_shutdown_release_closes_the_pool_for_good(tmp_path, monkeypatch) -> None:
    factory, pool, _docker, created = _pool(tmp_path, monkeypatch, size=2)

    async def run():
        pool.schedule_refill()
        await pool.wait_for_refill()
        await factory.close_idle_resources()
        taken = await pool.take("cayu-coding-a", _requirements(factory))
        pool.schedule_refill()
        await pool.wait_for_refill()
        return taken, pool.idle_count

    assert asyncio.run(run()) == (None, 0)
    assert pool.closed
    assert [runner.is_closed for runner in created] == [True, True]
    assert pool._nonce not in warm_spares_module._LIVE_POOL_NONCES


def test_pool_errors_fall_back_to_a_cold_start(tmp_path, monkeypatch) -> None:
    factory, pool, docker, created = _pool(tmp_path, monkeypatch, size=1)

    async def run():
        pool.schedule_refill()
        await pool.wait_for_refill()
        docker.unreachable = True
        taken = await pool.take("cayu-coding-a", _requirements(factory))
        # Reaping and refilling with Docker down back off without raising.
        pool._next_reap_at = 0.0
        pool.schedule_refill()
        await pool.wait_for_refill()
        return taken, pool.idle_count

    assert asyncio.run(run()) == (None, 1)
    assert not created[0].is_closed  # an unknown liveness state keeps the spare


def test_refill_rescans_for_stale_spares_after_the_reap_interval(tmp_path, monkeypatch) -> None:
    factory, pool, docker, _created = _pool(tmp_path, monkeypatch, size=1)

    async def run():
        pool.schedule_refill()
        await pool.wait_for_refill()
        first = docker.listings
        await pool.take("cayu-coding-a", _requirements(factory))
        await pool.wait_for_refill()
        within_interval = docker.listings
        pool._next_reap_at = 0.0  # the interval elapsed
        await pool.take("cayu-coding-b", _requirements(factory))
        await pool.wait_for_refill()
        return first, within_interval, docker.listings

    assert asyncio.run(run()) == (1, 1, 2)


def test_stale_spares_are_those_of_dead_processes_or_closed_pools() -> None:
    prefix = "cayu-coding-spare-abcd1234-"
    me = os.getpid()
    names = [
        f"{prefix}{me}-livepool-0001",
        f"{prefix}{me}-oldpool-0002",
        f"{prefix}999999-cafecafe-0003",
        "cayu-coding-spare-ffffffff-999999-cafecafe-0004",
        f"{prefix}not-a-pid",
    ]

    stale = warm_spares_module.stale_spare_names(
        names, prefix=prefix, live_nonces=frozenset({"livepool"})
    )

    assert stale == [names[1], names[2]]


def test_pool_reaps_stale_spares_before_its_first_spare(tmp_path, monkeypatch) -> None:
    _, pool, docker, _created = _pool(tmp_path, monkeypatch, size=1)
    prefix = pool.name_prefix
    assert prefix.startswith("cayu-coding-spare-")
    docker.listed = [f"{prefix}999999-cafecafe-0001", f"{prefix}{os.getpid()}-{pool._nonce}-0002"]
    monkeypatch.setattr(warm_spares_module, "_pid_is_alive", lambda pid: pid == os.getpid())

    async def run():
        pool.schedule_refill()
        await pool.wait_for_refill()

    asyncio.run(run())
    assert docker.removed == [f"{prefix}999999-cafecafe-0001"]


def test_release_deferred_materialization_removes_only_the_reserved_container(
    tmp_path, monkeypatch
) -> None:
    factory = _factory(tmp_path, deferred=True)
    reserved = docker_coding_module._docker_coding_container_name(
        _request(), configuration_fingerprint=factory._configuration_fingerprint
    )
    present: dict[str, str | None] = {reserved: "f" * 64, "cayu-coding-other": "e" * 64}
    removed: list[str] = []

    async def resolve(name, *, docker_path=None):
        return present.get(name)

    async def close(self):
        removed.append(self.container_id)
        present[reserved] = None

    monkeypatch.setattr(DockerRunner, "resolve_container_id", resolve)
    monkeypatch.setattr(DockerRunner, "close", close)
    metadata = factory._deferred_reconnect_metadata(reserved, None)
    request = _request(operation=EnvironmentFactoryOperation.RECONNECT, reconnect_metadata=metadata)

    async def run():
        first = await factory.release_deferred_materialization(request)
        second = await factory.release_deferred_materialization(request)
        eager = await _factory(tmp_path).release_deferred_materialization(request)
        return first, second, eager

    assert asyncio.run(run()) == (True, False, False)
    assert removed == ["f" * 64]
    for forged in (
        # Internally consistent metadata naming another cayu-coding container.
        factory._deferred_reconnect_metadata("cayu-coding-other", None),
        factory._deferred_reconnect_metadata("cayu-coding-spare-abc-1-n-x", None),
        {**metadata, "container_name": "someone-else"},
        {**metadata, "image_fingerprint": "sha256:" + "0" * 64},
    ):
        with pytest.raises(ValueError):
            asyncio.run(
                factory.release_deferred_materialization(
                    _request(
                        operation=EnvironmentFactoryOperation.RECONNECT,
                        reconnect_metadata=forged,
                    )
                )
            )
    # Another session's request cannot release this session's reserved name.
    with pytest.raises(ValueError):
        asyncio.run(
            factory.release_deferred_materialization(
                EnvironmentFactoryRequest(
                    session_id="another-session",
                    agent_name="agent",
                    environment_name="sandbox",
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    reconnect_metadata=metadata,
                )
            )
        )
    assert present["cayu-coding-other"] == "e" * 64


def test_release_reaches_a_container_recorded_before_an_image_repin(tmp_path, monkeypatch) -> None:
    before = _factory(tmp_path, deferred=True)
    repinned = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path, workspace_id="deferred-source"),
        toolchain_profile=docker_toolchain_profile(
            image_identity=DockerImageIdentity(reference="cayu/coding@sha256:" + ("d" * 64))
        ),
        docker_path="/usr/bin/docker",
        git_baseline=False,
        deferred=True,
    )
    assert before.image_identity.fingerprint != repinned.image_identity.fingerprint
    name = "cayu-coding-ealloc_" + ("e" * 32)
    present: dict[str, str | None] = {name: "f" * 64}
    removed: list[str] = []

    async def resolve(container_name, *, docker_path=None):
        return present.get(container_name)

    async def close(self):
        removed.append(self.container_id)
        present[name] = None

    monkeypatch.setattr(DockerRunner, "resolve_container_id", resolve)
    monkeypatch.setattr(DockerRunner, "close", close)
    metadata = before._deferred_reconnect_metadata(name, "ealloc_" + ("e" * 32))
    request = _request(operation=EnvironmentFactoryOperation.RECONNECT, reconnect_metadata=metadata)

    assert asyncio.run(repinned.release_deferred_materialization(request)) is True
    assert removed == ["f" * 64]
    with pytest.raises(ValueError, match="deferred allocation metadata"):
        asyncio.run(
            repinned.release_deferred_materialization(
                _request(
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    reconnect_metadata={**metadata, "container_name": "cayu-coding-other"},
                )
            )
        )


def test_deferred_mode_rejects_immutable_inputs_at_construction(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot be combined with immutable_inputs"):
        _factory(tmp_path, deferred=True, immutable_inputs=[object()])


@pytest.mark.parametrize("mode", ["use", "recover"])
def test_use_mode_replaces_a_crash_leftover_and_only_recovery_adopts(
    tmp_path, monkeypatch, mode
) -> None:
    factory = _factory(tmp_path, deferred=True)
    name = "cayu-coding-ealloc_" + ("e" * 32)
    leftover = "a" * 64
    present: dict[str, str | None] = {name: leftover}
    calls: list[tuple[str, str]] = []

    async def resolve(container_name, *, docker_path=None):
        return present.get(container_name)

    async def remove(container_name):
        calls.append(("remove", present[container_name]))
        present[container_name] = None

    async def create(container_name, *, immutable_mounts, requirements, allocation_identity=None):
        assert present.get(container_name) is None, "a fresh container never reuses the name"
        calls.append(("create", container_name))
        present[container_name] = "b" * 64
        return _FakeRunner("b" * 64)

    async def reconnect(container_id, *, immutable_mounts, requirements):
        calls.append(("adopt", container_id))
        return _FakeRunner(container_id)

    async def probe(runner, profile):
        return None

    monkeypatch.setattr(DockerRunner, "resolve_container_id", resolve)
    monkeypatch.setattr(factory, "_remove_named_container", remove)
    monkeypatch.setattr(factory, "_create_or_recover_runner", create)
    monkeypatch.setattr(factory, "_reconnect_runner", reconnect)
    monkeypatch.setattr(docker_coding_module, "_run_toolchain_admission_probes", probe)
    monkeypatch.setattr(factory, "_require_final_evidence", lambda runner, requirements: None)
    monkeypatch.setattr(factory, "_target_workspace", lambda runner, workspace_id: workspace_id)
    monkeypatch.setattr(
        factory, "create_workspace_binding", lambda request, *, target_workspace: target_workspace
    )

    runner, _binding = asyncio.run(
        factory._materialize_deferred(
            _request(),
            container_name=name,
            effective_requirements=_requirements(factory),
            mode=mode,
        )
    )

    if mode == "use":
        assert calls == [("remove", leftover), ("create", name)]
        assert runner.container_id == "b" * 64
    else:
        assert calls == [("adopt", leftover)]
        assert runner.container_id == leftover


def test_drain_trims_spares_and_the_next_use_refills_while_close_is_final(
    tmp_path, monkeypatch
) -> None:
    factory, pool, _docker, _created = _pool(tmp_path, monkeypatch, size=1)

    async def run():
        pool.schedule_refill()
        await pool.wait_for_refill()
        await factory.release_idle_resources()  # what drain_environment_cleanups calls
        trimmed = pool.idle_count
        await pool.take("cayu-coding-a", _requirements(factory))
        await pool.wait_for_refill()
        refilled = pool.idle_count
        await factory.close_idle_resources()  # what shutdown calls
        await pool.take("cayu-coding-b", _requirements(factory))
        await pool.wait_for_refill()
        return trimmed, refilled, pool.idle_count, pool.closed

    assert asyncio.run(run()) == (0, 1, 0, True)


def _foreign_metadata(factory, name: str, allocation_id: str | None = None) -> dict:
    # Internally consistent metadata from this very factory, naming a container
    # that is not the session's reserved name.
    identity = factory._deferred_reconnect_metadata(name, None)
    if allocation_id is None:
        return identity
    unsigned = {key: value for key, value in identity.items() if key != "allocation_fingerprint"}
    unsigned["allocation_id"] = allocation_id
    return {
        **unsigned,
        "allocation_fingerprint": docker_coding_module.sha256(
            docker_coding_module.canonical_durable_json_bytes(
                unsigned, "docker_coding_deferred_reconnect"
            )
        ).hexdigest(),
    }


@pytest.mark.parametrize(
    ("name", "allocation_id"),
    [("cayu-coding-other", None), ("postgres", None), ("cayu-coding-other", "other")],
)
def test_every_deferred_path_rejects_a_foreign_container_name(
    tmp_path, monkeypatch, name, allocation_id
) -> None:
    factory = _factory(tmp_path, deferred=True)
    docker_calls: list[object] = []

    async def must_not_touch_docker(*args, **kwargs):
        docker_calls.append(args)
        raise AssertionError("a foreign name must never reach Docker")

    monkeypatch.setattr(DockerRunner, "resolve_container_id", must_not_touch_docker)
    monkeypatch.setattr(DockerRunner, "create", must_not_touch_docker)
    monkeypatch.setattr(DockerRunner, "close", must_not_touch_docker)
    metadata = _foreign_metadata(factory, name, allocation_id)
    request = _request(operation=EnvironmentFactoryOperation.RECONNECT, reconnect_metadata=metadata)

    # Reconnect validation, the deferred result (whose use, dispose_unmaterialized
    # and DISCARD release all act on its name) and crash release all refuse.
    with pytest.raises(ValueError):
        factory._validate_request(request)
    with pytest.raises(ValueError):
        asyncio.run(factory.create(request))
    with pytest.raises(ValueError):
        asyncio.run(
            factory._create_deferred(
                request, allocation=None, effective_requirements=_requirements(factory)
            )
        )
    with pytest.raises(ValueError):
        asyncio.run(factory.release_deferred_materialization(request))
    assert docker_calls == []


def test_the_session_s_reserved_name_is_accepted_on_every_path(tmp_path) -> None:
    factory = _factory(tmp_path, deferred=True)
    reserved = docker_coding_module._docker_coding_container_name(
        _request(), configuration_fingerprint=factory._configuration_fingerprint
    )
    request = _request(
        operation=EnvironmentFactoryOperation.RECONNECT,
        reconnect_metadata=factory._deferred_reconnect_metadata(reserved, None),
    )

    factory._validate_request(request)
    result = asyncio.run(
        factory._create_deferred(
            request, allocation=None, effective_requirements=_requirements(factory)
        )
    )
    assert result.reconnect_metadata["container_name"] == reserved


def test_docker_spares_are_removed_synchronously_at_exit(tmp_path, monkeypatch) -> None:
    _factory, pool, _docker, created = _pool(tmp_path, monkeypatch, size=2)
    commands: list[list[str]] = []

    def run(argv, **kwargs):
        commands.append(argv)
        assert kwargs["timeout"] > 0 and kwargs["check"] is False

    monkeypatch.setattr(docker_coding_module.subprocess, "run", run)

    async def fill():
        pool.schedule_refill()
        await pool.wait_for_refill()

    asyncio.run(fill())
    pool._exit_hook()

    assert commands == [
        ["/usr/bin/docker", "rm", "-f", created[0].container_id, created[1].container_id]
    ]
    assert pool.idle_count == 0


def test_pool_docker_commands_are_bounded(tmp_path, monkeypatch) -> None:
    factory = _factory(tmp_path, deferred=True, warm_spares=1)
    atexit.unregister(factory.warm_spare_pool._exit_hook)
    backend = factory.warm_spare_pool._backend

    class _StalledProcess:
        returncode = None
        killed = False

        async def communicate(self):
            await asyncio.Event().wait()  # a daemon that never answers

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            return self.returncode

    process = _StalledProcess()

    async def spawn(*args, **kwargs):
        return process

    monkeypatch.setattr(docker_coding_module, "_WARM_SPARE_DOCKER_TIMEOUT_S", 0.05)
    monkeypatch.setattr(docker_coding_module.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(docker_coding_module, "_require_docker", lambda path: "/usr/bin/docker")

    async def run():
        with pytest.raises(TimeoutError):
            await backend._docker("ps", "-a")

    asyncio.run(run())
    assert process.killed
