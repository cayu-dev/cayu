"""Process-local allocation ownership: exit release and the orphan reaper."""

from __future__ import annotations

import asyncio
import gc
import os
import subprocess
import sys
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
from tests.core.test_deferred_environments import DeferredLocalFactory
from tests.core.test_deferred_environments import _app as _deferred_app
from tests.core.test_deferred_environments import _turns as _deferred_turns
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_docker_coding import _image_identity, _TestAllocationContext

from cayu import CayuApp, EventType, Message, RunRequest
from cayu.approvals import PendingToolApprovalEventView, ToolApprovalDecision, ToolApprovalRequest
from cayu.environments import (
    process_local_owner_exited,
    process_ownership,
    reap_orphaned_docker_coding_containers,
)
from cayu.environments.deferred import DeferredRunner
from cayu.environments.docker_coding import DockerCodingEnvironmentFactory
from cayu.environments.factory import (
    EnvironmentAllocationIntent,
    EnvironmentFactoryOperation,
    EnvironmentFactoryRequest,
    copy_environment_factory_request,
)
from cayu.environments.process_ownership import current_process_local_owner
from cayu.runners import _exit_cleanup
from cayu.runners.base import ExecResult, Runner
from cayu.runners.docker import DockerRunner
from cayu.sessions import InMemorySessionStore
from cayu.storage import SQLiteSessionStore
from cayu.workspaces.local import LocalWorkspace


def _prefix() -> str:
    """``v1:<boot>:<namespace>`` of this process."""

    token = current_process_local_owner()
    assert token is not None
    return token.rsplit(":", 2)[0]


def _token_for(pid: int, start: str | None = None) -> str:
    start = start if start is not None else process_ownership._process_start(pid)
    assert start is not None
    return f"{_prefix()}:{pid}:{start}"


def _waiting_child() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )


def _exited_owner() -> str:
    process = _waiting_child()
    token = _token_for(process.pid)
    process.communicate(b"")
    return token


def test_owner_token_names_boot_namespace_and_exact_process():
    token = current_process_local_owner()
    assert token is not None
    version, boot, namespace, pid, start = token.split(":")
    assert version == "v1" and boot and namespace
    assert int(pid) == os.getpid()
    assert start == process_ownership._process_start(os.getpid())


def test_an_exited_owner_in_this_boot_and_namespace_counts_as_gone():
    assert process_local_owner_exited(_exited_owner())


def test_a_live_owner_is_kept_and_a_reused_pid_is_not_mistaken_for_it():
    process = _waiting_child()
    try:
        assert not process_local_owner_exited(_token_for(process.pid))
        # The same PID with another start time is a different (reused) process.
        assert process_local_owner_exited(_token_for(process.pid, start="1"))
    finally:
        process.communicate(b"")
    assert not process_local_owner_exited(current_process_local_owner() or "")


def test_an_owner_in_another_namespace_or_boot_is_never_reaped():
    # A live owner in another PID namespace (for example the host, seen from a
    # host-network container sharing the daemon) has a PID that does not run
    # here; the namespace or boot mismatch alone must keep its container.
    _, boot, namespace = _prefix().split(":")
    other_namespace = "host" if namespace != "host" else "4026531836"
    other_boot = ("0" if boot[0] != "0" else "1") + boot[1:]
    exited = _exited_owner()
    pid, start = exited.rsplit(":", 2)[1:]
    assert not process_local_owner_exited(f"v1:{boot}:{other_namespace}:{pid}:{start}")
    assert not process_local_owner_exited(f"v1:{other_boot}:{namespace}:{pid}:{start}")
    for malformed in ("", "nonsense", exited.replace("v1:", "v2:"), exited + "x", None):
        assert not process_local_owner_exited(malformed)  # type: ignore[arg-type]


def test_identity_that_cannot_be_proven_fails_closed(monkeypatch):
    exited = _exited_owner()
    process = _waiting_child()
    try:
        live = _token_for(process.pid)
        # A live owner whose start time cannot be read is kept.
        monkeypatch.setattr(process_ownership, "_process_start", lambda pid: None)
        assert not process_local_owner_exited(live)
        assert current_process_local_owner() is None
        # Without this process's own boot and namespace, nothing is provable.
        monkeypatch.setattr(process_ownership, "_boot_and_namespace", lambda: None)
        assert not process_local_owner_exited(exited)
    finally:
        process.communicate(b"")


class _RecordingDeferredFactory(DeferredLocalFactory):
    def __init__(self, source, target) -> None:
        super().__init__(source, target)
        self.requests: list[tuple[EnvironmentFactoryOperation, str | None]] = []

    async def create(self, request):
        self.requests.append((request.operation, request.process_local_owner))
        return await super().create(request)


@pytest.mark.parametrize("process_local", [True, False])
def test_create_and_reconnect_requests_name_a_process_local_owner(tmp_path, process_local):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "input.txt").write_text("hello\n", encoding="utf-8")
    factory = _RecordingDeferredFactory(source, target)
    store = InMemorySessionStore() if process_local else SQLiteSessionStore(tmp_path / "s.db")

    async def run():
        # Pause for approval before the first tool, then approve: the
        # continuation reconnects and only then materializes the container.
        paused = _deferred_app(factory, store, _deferred_turns("write"), approval=True)
        async for _ in paused.run(
            RunRequest(
                agent_name="agent", session_id="paused", messages=[Message.text("user", "go")]
            )
        ):
            pass
        approval = next(
            event
            for event in await store.load_events("paused")
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        pending = PendingToolApprovalEventView.from_event(approval)
        resumed = _deferred_app(factory, store, _deferred_turns(), approval=True)
        async for _ in resumed.resolve_tool_approval(
            ToolApprovalRequest(
                session_id="paused",
                approval_id=pending.approval_id,
                tool_round_id=pending.tool_round_id,
                tool_call_id=pending.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
            )
        ):
            pass

    asyncio.run(run())

    owner = current_process_local_owner() if process_local else None
    assert [operation for operation, _ in factory.requests] == [
        EnvironmentFactoryOperation.CREATE,
        EnvironmentFactoryOperation.RECONNECT,
    ]
    assert [request_owner for _, request_owner in factory.requests] == [owner, owner]
    assert factory.created == 1


def test_process_local_store_registers_exit_release(tmp_path):
    process_local = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    durable = CayuApp(
        session_store=SQLiteSessionStore(tmp_path / "sessions.db"), enable_logging=False
    )

    assert process_local._environment_lifecycle._exit_hook is not None
    assert durable._environment_lifecycle._exit_hook is None


def test_requests_only_accept_owner_tokens():
    with pytest.raises(ValueError, match="process_local_owner"):
        EnvironmentFactoryRequest(
            session_id="s", agent_name="a", environment_name="e", process_local_owner="pid:1"
        )


class _ExitRunner(Runner):
    closed_at_exit: list[_ExitRunner] = []

    async def exec(self, *args, **kwargs):  # pragma: no cover - never executed
        raise AssertionError

    @classmethod
    def _close_all_at_interpreter_exit(cls, entries) -> int:
        assert all(action is None for _, action in entries)
        cls.closed_at_exit.extend(runner for runner, _ in entries)
        return len(entries)


def _owner(runner, *, factory_backed: bool, field: str = "environment"):
    environments = {
        "environment": SimpleNamespace(runner=None),
        "unclaimed_factory_result": None,
        "retained_factory_result": None,
    }
    holder = SimpleNamespace(runner=runner)
    environments[field] = holder if field == "environment" else SimpleNamespace(environment=holder)
    return SimpleNamespace(
        registered_environment=SimpleNamespace(factory_backed=factory_backed, **environments)
    )


@pytest.fixture
def exit_registry(monkeypatch):
    holders: dict = {}
    monkeypatch.setattr(_exit_cleanup, "_holders", holders)
    monkeypatch.setattr(_ExitRunner, "closed_at_exit", [])
    return holders


def test_exit_release_closes_only_factory_runners_still_held(monkeypatch, exit_registry):
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    setups = app._environment_lifecycle._active_environment_setups
    held, unclaimed, registered = _ExitRunner(), _ExitRunner(), _ExitRunner()
    setups.update(
        {
            "held": _owner(held, factory_backed=True),
            "unclaimed": _owner(unclaimed, factory_backed=True, field="unclaimed_factory_result"),
            "registered": _owner(registered, factory_backed=False),
        }
    )
    # The app is dropped without aclose(): what it held is still released at exit.
    del app
    gc.collect()
    assert len(exit_registry) == 1

    # A forked child that exits never releases its parent's allocations.
    monkeypatch.setattr(_exit_cleanup, "_owner_pid", -1)
    _exit_cleanup._release_held_at_exit()
    assert _ExitRunner.closed_at_exit == []

    monkeypatch.setattr(_exit_cleanup, "_owner_pid", os.getpid())
    _exit_cleanup._release_held_at_exit()
    assert set(map(id, _ExitRunner.closed_at_exit)) == {id(held), id(unclaimed)}
    setups.clear()


def test_collected_app_holding_nothing_leaves_no_exit_registration(exit_registry):
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    assert len(exit_registry) == 1
    del app
    gc.collect()
    assert exit_registry == {}


def test_reaper_removes_only_containers_of_exited_process_local_owners(monkeypatch):
    exited = _exited_owner()
    _, boot, namespace = _prefix().split(":")
    other_namespace = "host" if namespace != "host" else "4026531836"
    pid, start = exited.rsplit(":", 2)[1:]
    foreign_owner = f"v1:{boot}:{other_namespace}:{pid}:{start}"
    orphan, live, foreign, spare = "a" * 64, "b" * 64, "c" * 64, "d" * 64
    listing = "\n".join(
        [
            f"{orphan}\tcayu-coding-ealloc_{'1' * 32}\t{exited}",
            f"{live}\tcayu-coding-ealloc_{'2' * 32}\t{current_process_local_owner()}",
            f"{foreign}\tcayu-coding-ealloc_{'3' * 32}\t{foreign_owner}",
            f"{spare}\tcayu-coding-spare-1-2-3\t{exited}",
            f"short\tcayu-coding-ealloc_{'4' * 32}\t{exited}",
            "garbage line",
        ]
    )
    calls: list[list[str]] = []

    async def docker(path, args, **kwargs):
        calls.append(list(args))
        if args[0] == "ps":
            return ExecResult(stdout=listing + "\n")
        return ExecResult()

    monkeypatch.setattr("cayu.environments.docker_coding._run_docker", docker)

    removed = asyncio.run(
        reap_orphaned_docker_coding_containers(docker_path="/usr/bin/docker", label="run=one")
    )

    assert removed == (orphan,)
    listed = calls[0]
    for expected in ("name=cayu-coding-", "label=io.cayu.process-local-owner", "label=run=one"):
        assert listed[listed.index(expected) - 1] == "--filter"
    assert calls[1:] == [["rm", "-f", orphan]]


@pytest.mark.parametrize("operation", ["create", "reconnect"])
def test_deferred_materialization_labels_process_local_containers(tmp_path, monkeypatch, operation):
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path, workspace_id="deferred-source"),
        toolchain_profile=docker_toolchain_profile(image_identity=_image_identity()),
        docker_path="/usr/bin/docker",
        git_baseline=False,
        deferred=True,
    )
    owner = current_process_local_owner()
    request = EnvironmentFactoryRequest(
        session_id="deferred-session",
        agent_name="agent",
        environment_name="sandbox",
        process_local_owner=owner,
    )
    allocation = _TestAllocationContext(
        EnvironmentAllocationIntent(
            allocation_id="ealloc_" + ("e" * 32),
            provider="docker",
            adapter_generation="cayu.docker_coding.v12",
            session_id=request.session_id,
            environment_name=request.environment_name,
            requested_operation=EnvironmentFactoryOperation.CREATE,
        )
    )
    created: list[tuple[str, str | None]] = []

    class Stop(Exception):
        pass

    async def resolve(name, *, docker_path=None):
        return None

    async def create(name, **kwargs):
        created.append((name, kwargs.get("process_local_owner")))
        raise Stop

    monkeypatch.setattr(DockerRunner, "resolve_container_id", resolve)
    monkeypatch.setattr(factory, "_create_or_recover_runner", create)
    result = asyncio.run(factory.create_recoverable(request, allocation))
    if operation == "reconnect":
        # A continuation (for example after an approval) reconnects without an
        # allocation context and only then creates the container.
        reconnect = replace(
            request,
            operation=EnvironmentFactoryOperation.RECONNECT,
            reconnect_metadata=result.reconnect_metadata,
        )
        result = asyncio.run(factory.create(reconnect))
    with pytest.raises(Stop):
        asyncio.run(result.environment.runner.materialization._materialize("use"))

    assert created == [("cayu-coding-ealloc_" + ("e" * 32), owner)]


@pytest.mark.parametrize("process_local", [True, False])
def test_eager_create_without_an_allocation_labels_process_local_containers(
    tmp_path, monkeypatch, process_local
):
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path, workspace_id="eager-source"),
        toolchain_profile=docker_toolchain_profile(image_identity=_image_identity()),
        docker_path="/usr/bin/docker",
        git_baseline=False,
    )
    owner = current_process_local_owner() if process_local else None
    request = EnvironmentFactoryRequest(
        session_id="eager-session",
        agent_name="agent",
        environment_name="sandbox",
        process_local_owner=owner,
    )
    labelled: list[str | None] = []

    class Stop(Exception):
        pass

    async def resolve(name, *, docker_path=None):
        return None

    async def create(name, **kwargs):
        labelled.append(kwargs["_process_local_owner"])
        raise Stop

    monkeypatch.setattr(DockerRunner, "resolve_container_id", resolve)
    monkeypatch.setattr(DockerRunner, "create", create)
    with pytest.raises(Stop):
        asyncio.run(factory.create(request))

    assert labelled == [owner]


def test_exit_release_reaches_a_materialized_deferred_runner():
    removed: list[list] = []

    class Materialized(Runner):
        async def exec(self, *args, **kwargs):  # pragma: no cover - never executed
            raise AssertionError

        @classmethod
        def _close_all_at_interpreter_exit(cls, entries) -> int:
            removed.append(list(entries))
            return len(entries)

    live = Materialized()
    materialized = object.__new__(DeferredRunner)
    materialized._deferred = SimpleNamespace(materialized_runner=live)
    pending = object.__new__(DeferredRunner)
    pending._deferred = SimpleNamespace(materialized_runner=None)

    released = _exit_cleanup.close_runners_at_exit([(materialized, None), (pending, None)])

    assert released == 1
    assert removed == [[(live, None)]]


def test_reaper_reports_a_failed_listing(monkeypatch):
    async def docker(path, args, **kwargs):
        return ExecResult(exit_code=1)

    monkeypatch.setattr("cayu.environments.docker_coding._run_docker", docker)
    with pytest.raises(RuntimeError, match="could not list"):
        asyncio.run(reap_orphaned_docker_coding_containers(docker_path="/usr/bin/docker"))


def test_forked_child_releases_its_own_allocations_but_never_its_parents(monkeypatch):
    monkeypatch.setattr(_exit_cleanup, "_holders", {})
    monkeypatch.setattr(_exit_cleanup, "_inherited", {})
    monkeypatch.setattr(_exit_cleanup, "_unsettled", {})
    monkeypatch.setattr(_ExitRunner, "closed_at_exit", [])
    parent_runner, child_runner = _ExitRunner(), _ExitRunner()
    held = [parent_runner]

    class Owner:
        pass

    owner = Owner()
    _exit_cleanup.release_held_runners_at_exit(owner, lambda: list(held))

    # What os.register_at_fork runs in the child.
    _exit_cleanup._forget_in_child()
    held.append(child_runner)
    _exit_cleanup._release_held_at_exit()

    assert _ExitRunner.closed_at_exit == [child_runner]


def test_copying_a_factory_request_keeps_every_field():
    request = EnvironmentFactoryRequest(
        session_id="s",
        agent_name="a",
        environment_name="e",
        parent_session_id="parent",
        causal_budget_id="budget",
        labels={"team": "x"},
        metadata={"k": "v"},
        reconnect_metadata={"kind": "r"},
        operation=EnvironmentFactoryOperation.RECONNECT,
        execution_profile_fingerprint="a" * 64,
        interaction_id="interaction",
        process_local_owner=current_process_local_owner(),
    )
    copied = copy_environment_factory_request(request)

    # Every dataclass field, so a field added later cannot be dropped silently.
    for field in fields(EnvironmentFactoryRequest):
        assert getattr(copied, field.name) == getattr(request, field.name), field.name
    assert copied.process_local_owner is not None
