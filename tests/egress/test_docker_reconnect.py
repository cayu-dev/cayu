from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cayu.egress._docker_reconnect import (
    OWNER_LABEL,
    DockerEgressReconnectError,
    _OwnedDockerRunner,
    validate_identity,
)
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.egress.errors import InvalidEgressReconnectMetadataError

CID = "a" * 64
NID = "b" * 64
TOKEN = "c" * 32
SID = "d" * 64


class Docker:
    def __init__(self):
        self.commands = []
        self.unavailable = False
        self.container = {
            "Id": CID,
            "Image": "sha256:" + "e" * 64,
            "Config": {"Cmd": ["sleep", "infinity"], "Entrypoint": None, "User": ""},
            "HostConfig": {"NetworkMode": NID, "Privileged": False},
            "Mounts": [],
            "State": {"Running": True, "Paused": False},
            "NetworkSettings": {"Networks": {"fixture": {"NetworkID": NID}}},
        }
        self.network = {"Id": NID, "Internal": True, "Labels": {OWNER_LABEL: TOKEN}}
        self.objects = {CID: self.container, "fixture": self.container, NID: self.network}

    async def __call__(self, args):
        self.commands.append(list(args))
        if self.unavailable:
            return 1, "fixture daemon unavailable"
        if args[:2] == ["context", "inspect"]:
            return 0, "unix:///fixture/docker.sock"
        if args[0] == "inspect":
            item = self.objects.get(args[-1])
            return (1, "") if item is None else (0, json.dumps([item]))
        if args[0] == "ps" or args[:2] == ["network", "ls"]:
            return 0, "\n".join(self.objects)
        if args[0] == "rm" or args[:2] == ["network", "rm"]:
            self.objects.pop(args[-1], None)
        return 0, ""


def setup(tmp_path):
    docker = Docker()
    adapter = DockerEgressAdapter(
        reconnect_state_dir=tmp_path, docker_exec=docker, docker_run=docker
    )
    manager = adapter._reconnect
    identity = validate_identity(
        {
            "version": 1,
            "backend": "docker",
            "allocation_id": TOKEN,
            "container_id": CID,
            "container_name": "fixture",
            "network_id": NID,
            "session_id": "session",
            "environment_name": "environment",
            "image_id": "sha256:" + "e" * 64,
            "configuration": manager.configuration,
            "container_configuration": manager.container_configuration(docker.container),
            "runner_configuration": "f" * 64,
        }
    )
    claim = manager.claim(TOKEN)
    claim.write(state="retained", identity=identity, network_id=NID)
    claim.close()
    return docker, adapter, manager, identity


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("backend", "microsandbox"),
        ("container_id", "fixture"),
        ("allocation_id", "../other"),
        ("unexpected", "secret"),
        ("runner_configuration", "invalid"),
    ],
)
def test_metadata_rejects_malformed_identity_before_backend_access(tmp_path, field, value):
    docker, adapter, _, identity = setup(tmp_path)
    with pytest.raises(InvalidEgressReconnectMetadataError):
        adapter.validate_reconnect_metadata({**identity, field: value})
    assert not docker.commands


@pytest.mark.parametrize("scope", ["session", "environment"])
def test_wrong_scope_never_mutates_allocation(tmp_path, scope):
    async def scenario():
        docker, adapter, _, identity = setup(tmp_path)
        with pytest.raises(InvalidEgressReconnectMetadataError):
            await adapter.prepare_reconnect(
                session_id="foreign" if scope == "session" else "session",
                environment_name="foreign" if scope == "environment" else "environment",
                grants=(),
                broker=None,
                reconnect_metadata=identity,
            )
        assert not docker.commands

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change,code",
    [
        ("name_replaced", "identity_mismatch"),
        ("absent", "allocation_absent"),
        ("daemon", "daemon_unavailable"),
        ("image", "configuration_mismatch"),
        ("network", "identity_mismatch"),
        ("profile", "configuration_mismatch"),
        ("public_network", "identity_mismatch"),
    ],
)
def test_exact_allocation_validation(tmp_path, change, code):
    async def scenario():
        docker, _, manager, identity = setup(tmp_path)
        if change == "name_replaced":
            docker.objects["fixture"] = {**docker.container, "Id": "f" * 64}
        elif change == "absent":
            docker.objects.pop(CID)
        elif change == "daemon":
            docker.unavailable = True
        elif change == "image":
            docker.container["Image"] = "sha256:" + "0" * 64
        elif change == "network":
            docker.container["NetworkSettings"]["Networks"]["second"] = {"NetworkID": "0" * 64}
        elif change == "profile":
            docker.container["HostConfig"]["Privileged"] = True
        elif change == "public_network":
            docker.network["Internal"] = False
        claim = manager.claim(TOKEN)
        claim.read()
        try:
            with pytest.raises(DockerEgressReconnectError) as error:
                await manager.validate_allocation(claim, identity)
            assert error.value.code == code
            assert all(command[0] in {"inspect", "ps", "network"} for command in docker.commands)
        finally:
            claim.close()

    asyncio.run(scenario())


def test_exclusive_process_ownership_and_stale_runner_fence(tmp_path):
    _, _, manager, _ = setup(tmp_path)
    claim = manager.claim(TOKEN)
    claim.read()
    runner = _OwnedDockerRunner("fixture", _container_id=CID)
    runner._owner = claim
    script = """
import sys
from pathlib import Path
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.egress._docker_reconnect import DockerEgressReconnectError
manager = DockerEgressAdapter(reconnect_state_dir=Path(sys.argv[1]))._reconnect
try:
    manager.claim(sys.argv[2])
except DockerEgressReconnectError as error:
    assert error.code == "ownership_conflict"
else:
    raise AssertionError("two owners")
"""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), TOKEN],
        env={**os.environ, "PYTHONPATH": f"{root / 'src'}:{root}"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    claim.close()
    with pytest.raises(DockerEgressReconnectError, match="ownership_uncertain"):
        runner._ensure_exec_open()
    next_owner = manager.claim(TOKEN)
    next_owner.close()


def test_disposal_pending_converges_without_reviving_or_removing_replacement(tmp_path):
    async def scenario():
        docker, adapter, manager, identity = setup(tmp_path)
        claim = manager.claim(TOKEN)
        claim.read()
        claim.write(state="disposal_pending")
        claim.close()
        docker.objects["fixture"] = {**docker.container, "Id": "f" * 64}
        with pytest.raises(DockerEgressReconnectError, match="disposed"):
            await adapter.prepare_reconnect(
                session_id="session",
                environment_name="environment",
                grants=(),
                broker=None,
                reconnect_metadata=identity,
            )
        assert ["rm", "-f", CID] in docker.commands
        assert not any(
            command[-1] == "fixture" and command[0] == "rm" for command in docker.commands
        )
        assert "fixture" in docker.objects
        claim = manager.claim(TOKEN)
        claim.read()
        assert claim.journal["state"] == "disposed"
        claim.close()

    asyncio.run(scenario())


def test_uncertain_daemon_mutation_retains_claim_and_fences_on_restart(tmp_path):
    async def scenario():
        docker, _, manager, identity = setup(tmp_path)
        claim = manager.claim(TOKEN)
        claim.read()
        never = asyncio.Event()

        async def blocked(args):
            await never.wait()
            return 0, ""

        manager.adapter._docker_exec = blocked
        manager.timeout_s = 0.01
        with pytest.raises(DockerEgressReconnectError, match="ownership_uncertain"):
            await manager.run(["exec", CID, "fixture"])
        with pytest.raises(DockerEgressReconnectError, match="ownership_uncertain"):
            claim.close()
        with pytest.raises(DockerEgressReconnectError, match="ownership_conflict"):
            manager.claim(TOKEN)
        # Simulate process death: OS closes its lock; its durable marker survives.
        never.set()
        await asyncio.gather(*manager.pending_commands)
        claim.handle.close()
        claim.closed = True
        adapter2 = DockerEgressAdapter(
            reconnect_state_dir=tmp_path, docker_exec=docker, docker_run=docker
        )
        with pytest.raises(DockerEgressReconnectError, match="ownership_uncertain"):
            await adapter2.prepare_reconnect(
                session_id="session",
                environment_name="environment",
                grants=(),
                broker=None,
                reconnect_metadata=identity,
            )
        assert ["pause", CID] in docker.commands
        claim2 = adapter2._reconnect.claim(TOKEN)
        claim2.read()
        assert claim2.journal["state"] == "ownership_uncertain"
        claim2.close()

    asyncio.run(scenario())


def test_reconnect_is_opt_in(tmp_path):
    assert not DockerEgressAdapter().supports_reconnect
    adapter = DockerEgressAdapter(reconnect_state_dir=tmp_path)
    assert adapter.supports_reconnect
    assert any(
        claim.capability == "reconnect" and claim.state == "declared"
        for claim in adapter.execution_capability_evidence().claims
    )


@pytest.mark.parametrize("endpoint", ["ssh://remote", "tcp://127.0.0.1:2375", ""])
def test_remote_or_unverifiable_context_is_rejected_before_claim(tmp_path, endpoint, monkeypatch):
    async def scenario():
        docker, adapter, manager, identity = setup(tmp_path)
        monkeypatch.delenv("DOCKER_HOST", raising=False)
        monkeypatch.delenv("DOCKER_CONTEXT", raising=False)

        async def run(args):
            if args[:2] == ["context", "inspect"]:
                return 0, endpoint
            return await docker(args)

        adapter._docker_run = run
        with pytest.raises(DockerEgressReconnectError, match="unsupported_host"):
            await adapter.prepare_reconnect(
                session_id="session",
                environment_name="environment",
                grants=(),
                broker=None,
                reconnect_metadata=identity,
            )
        assert not docker.commands
        claim = manager.claim(TOKEN)
        claim.read()
        assert claim.journal["state"] == "retained"
        claim.close()

    asyncio.run(scenario())


def test_container_mount_identity_is_order_independent(tmp_path):
    docker, _, manager, _ = setup(tmp_path)
    docker.container["Mounts"] = [
        {"Destination": "/z", "RW": False},
        {"Destination": "/a", "RW": True},
    ]
    before = manager.container_configuration(docker.container)
    docker.container["Mounts"].reverse()
    assert manager.container_configuration(docker.container) == before
    docker.container["Mounts"][0]["RW"] = False
    assert manager.container_configuration(docker.container) != before


@pytest.mark.parametrize("phase", ["freeze", "remove_sidecar", "listener", "ca", "publication"])
def test_partial_reconnect_never_exposes_runner_and_keeps_truthful_state(
    tmp_path, phase, monkeypatch
):
    from cayu.egress._docker_reconnect import _Claim
    from cayu.egress.adapter import EgressBinding

    async def scenario():
        docker, adapter, manager, identity = setup(tmp_path)
        docker.objects[SID] = {
            "Id": SID,
            "Image": "sha256:" + "1" * 64,
            "Config": {"Labels": {OWNER_LABEL: TOKEN}},
        }
        prepared = []

        async def prepare(**kwargs):
            if phase == "listener":
                raise OSError("fixture listener failure")
            docker.objects[kwargs["reconnect_sidecar"]] = docker.objects[SID]
            prepared.append(True)
            return EgressBinding(network=NID, sidecar=SID, ca_cert_pem=b"public fixture CA")

        monkeypatch.setattr(adapter, "_prepare", prepare)
        if phase in {"freeze", "remove_sidecar"}:
            original = getattr(manager, phase)
            calls = 0

            async def fail_once(claim):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise DockerEgressReconnectError("fencing_failed")
                return await original(claim)

            monkeypatch.setattr(manager, phase, fail_once)
        if phase == "ca":

            def fail_ca(*args):
                raise DockerEgressReconnectError("state_unavailable")

            monkeypatch.setattr(_Claim, "install_ca", fail_ca)
        if phase == "publication":
            original_write = _Claim.write

            def fail_publication(claim, **updates):
                if "sidecar_image_id" in updates:
                    raise DockerEgressReconnectError("state_unavailable")
                return original_write(claim, **updates)

            monkeypatch.setattr(_Claim, "write", fail_publication)
        with pytest.raises((DockerEgressReconnectError, OSError)):
            await adapter.prepare_reconnect(
                session_id="session",
                environment_name="environment",
                grants=(),
                broker=None,
                reconnect_metadata=identity,
            )
        assert not manager.bindings
        assert not any(command[0] == "rm" and command[-1] == CID for command in docker.commands)
        claim = manager.claim(TOKEN)
        claim.read()
        assert claim.journal["state"] == "recovering"
        assert claim.journal["identity"] == identity
        claim.close()

    asyncio.run(scenario())


def test_no_thaw_before_final_admission_and_no_dispatch_after_claim_loss(tmp_path, monkeypatch):
    from cayu.runners import ExecCommand, ExecResult
    from cayu.runners.docker import DockerRunner

    async def scenario():
        _, adapter, manager, _ = setup(tmp_path)
        claim = manager.claim(TOKEN)
        claim.read()
        claim.write(state="active")
        runner = _OwnedDockerRunner("fixture", _container_id=CID)
        runner._owner, runner._manager, runner._frozen = claim, manager, True
        activated, commands = [], []

        async def activate(owner):
            activated.append(owner)

        async def execute(self, command, **kwargs):
            commands.append(command)
            return ExecResult(exit_code=0, stdout="", stderr="")

        monkeypatch.setattr(manager, "activate", activate)
        monkeypatch.setattr(DockerRunner, "_exec", execute)
        await runner._exec(ExecCommand.process("true"))
        assert not activated
        adapter.complete_runner_admission(runner)
        await runner._exec(ExecCommand.process("true"))
        assert activated == [claim]
        claim.close()
        with pytest.raises(DockerEgressReconnectError, match="ownership_uncertain"):
            await runner._exec(ExecCommand.process("false"))
        assert len(commands) == 2

    asyncio.run(scenario())


def test_cancelled_mutation_preserves_cancellation_and_durable_uncertainty(tmp_path):
    async def scenario():
        _, adapter, manager, _ = setup(tmp_path)
        claim = manager.claim(TOKEN)
        claim.read()
        started, finish = asyncio.Event(), asyncio.Event()

        async def mutation(args):
            started.set()
            await finish.wait()
            return 0, ""

        adapter._docker_exec = mutation
        task = asyncio.create_task(manager.run(["exec", CID, "fixture"]))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert claim.journal["pending_mutation"] is True
        with pytest.raises(DockerEgressReconnectError, match="ownership_conflict"):
            manager.claim(TOKEN)
        finish.set()
        await asyncio.gather(*manager.pending_commands)
        assert claim.journal["pending_mutation"] is True
        claim.handle.close()
        claim.closed = True

    asyncio.run(scenario())


def test_terminal_main_removal_keeps_disposal_pending_until_graph_settles(tmp_path, monkeypatch):
    from cayu.runners.docker import DockerRunner

    async def scenario():
        _, _, manager, _ = setup(tmp_path)
        claim = manager.claim(TOKEN)
        claim.read()
        runner = _OwnedDockerRunner("fixture", _container_id=CID)
        runner._owner = claim

        async def close(self):
            assert self.close_action == "remove"

        monkeypatch.setattr(DockerRunner, "close", close)
        await manager.finalize(runner, outcome="completed")
        assert claim.journal["state"] == "disposal_pending"
        claim.close()

    asyncio.run(scenario())


def test_preflight_failure_never_admits_or_thaws_retained_guest(tmp_path, monkeypatch):
    from cayu.egress import EgressBinding, VirtualEgressRunnerRequest, _docker_reconnect

    async def scenario():
        _, _, manager, identity = setup(tmp_path)
        claim = manager.claim(TOKEN)
        claim.read()
        binding = EgressBinding(network=NID)
        request = VirtualEgressRunnerRequest(
            name="fixture",
            runner_kind="docker",
            image="fixture",
            binding=binding,
            env_overlay={},
            ca_cert_host_path="/fixture",
            guest_ca_path="/ca.pem",
            setup_commands=(),
            egress_destinations=(),
            session_id="session",
            environment_name="environment",
        )
        identity["runner_configuration"] = _docker_reconnect._digest(
            ["docker", "fixture", (), None, "/ca.pem"]
        )
        claim.write(identity=identity, image="fixture", state="recovering")
        manager.bindings[id(binding)] = claim

        async def preflight(*args, **kwargs):
            raise RuntimeError("fixture direct network was reachable")

        monkeypatch.setattr(_docker_reconnect, "run_enforcement_preflight", preflight)
        with pytest.raises(DockerEgressReconnectError, match="preflight_failed"):
            await manager.create_runner(request)
        assert claim.journal["state"] == "recovering"
        assert claim.runner._frozen and not claim.runner._admission_complete
        assert claim.runner.close_action == "none"
        claim.close()

    asyncio.run(scenario())


def test_reconnect_refuses_unqualified_colocated_control_server(tmp_path):
    from cayu.egress import UnsupportedEgressError

    with pytest.raises(UnsupportedEgressError, match="colocated control server"):
        DockerEgressAdapter(reconnect_state_dir=tmp_path, control_server_container_id="a" * 64)
