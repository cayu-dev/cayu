"""Exact, claim-owned control attachments for retained Docker allocations."""

import asyncio
import json

import pytest
from tests.egress.test_docker_reconnect import CID, NID, TOKEN, Docker

from cayu.egress._docker_reconnect import OWNER_LABEL
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.egress.errors import DockerEgressReconnectError

CONTROL = "1" * 64
OTHER = "2" * 64


class ControlDocker(Docker):
    def __init__(self):
        super().__init__()
        self.control = {
            "Id": CONTROL,
            "State": {"Running": True},
            "NetworkSettings": {"Networks": {}},
        }
        self.objects[CONTROL] = self.control
        self.network["Containers"] = {CID: {"EndpointID": "browser"}}

    async def __call__(self, args):
        result = await super().__call__(args)
        if args[:2] == ["network", "connect"]:
            network = self.objects[args[-2]]
            network["Containers"][CONTROL] = {"EndpointID": network["Id"]}
            self.control["NetworkSettings"]["Networks"][network["Id"]] = {
                "NetworkID": network["Id"],
                "EndpointID": network["Id"],
                "Aliases": ["cayu-control"],
            }
        if args[:2] == ["network", "disconnect"]:
            self.objects[args[-2]]["Containers"].pop(CONTROL, None)
            self.control["NetworkSettings"]["Networks"].pop(args[-2], None)
        return result


def owned(tmp_path):
    docker = ControlDocker()
    adapter = DockerEgressAdapter(
        reconnect_state_dir=tmp_path,
        control_server_container_id=CONTROL,
        docker_exec=docker,
        docker_run=docker,
    )
    manager = adapter._reconnect
    claim = manager.claim(TOKEN)
    claim.write(
        state="retained",
        network_id=NID,
        control_server_container_id=CONTROL,
        identity={"container_id": CID},
    )
    return docker, manager, claim


def test_configuration_binds_exact_application_container(tmp_path):
    first = DockerEgressAdapter(reconnect_state_dir=tmp_path, control_server_container_id=CONTROL)
    other = DockerEgressAdapter(reconnect_state_dir=tmp_path, control_server_container_id=OTHER)
    assert first.configuration_metadata != other.configuration_metadata


def test_verified_attachment_is_reused_and_cleanup_is_allocation_local(tmp_path):
    async def scenario():
        docker, manager, claim = owned(tmp_path)
        await manager.attach_control_server(TOKEN, NID)
        await manager.attach_control_server(TOKEN, NID)
        assert len([args for args in docker.commands if args[:2] == ["network", "connect"]]) == 1
        other = manager.claim("e" * 32)
        other.write(state="retained", network_id=OTHER, control_server_container_id=CONTROL)
        docker.objects[OTHER] = {
            "Id": OTHER,
            "Internal": True,
            "Labels": {OWNER_LABEL: other.token},
            "Containers": {},
        }
        await manager.attach_control_server(other.token, OTHER)
        await manager.detach_control_server(claim)
        await manager.detach_control_server(claim)
        assert set(docker.control["NetworkSettings"]["Networks"]) == {OTHER}
        assert docker.objects[OTHER]["Containers"][CONTROL]
        await manager.remove_network(claim)
        assert NID not in docker.objects
        assert CONTROL in docker.objects and OTHER in docker.objects
        assert not any(
            args[0] in {"rm", "stop", "pause"} and args[-1] == CONTROL for args in docker.commands
        )
        claim.close()
        other.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fault,code",
    [
        ("missing", "control_server_unavailable"),
        ("replacement", "control_server_unavailable"),
        ("stopped", "control_server_unavailable"),
        ("alias", "control_server_alias_conflict"),
        ("endpoint", "control_server_alias_conflict"),
        ("foreign_network", "identity_mismatch"),
        ("guest_alias", "control_server_alias_conflict"),
        ("foreign_member", "identity_mismatch"),
    ],
)
def test_attachment_conflicts_fail_before_dispatch(tmp_path, fault, code):
    async def scenario():
        docker, manager, claim = owned(tmp_path)
        if fault == "missing":
            del docker.objects[CONTROL]
        elif fault == "replacement":
            docker.control["Id"] = OTHER
        elif fault == "stopped":
            docker.control["State"]["Running"] = False
        elif fault in {"alias", "endpoint"}:
            await manager.attach_control_server(TOKEN, NID)
            endpoint = docker.control["NetworkSettings"]["Networks"][NID]
            endpoint["Aliases" if fault == "alias" else "EndpointID"] = (
                [] if fault == "alias" else "different"
            )
        elif fault == "foreign_network":
            docker.network["Labels"][OWNER_LABEL] = "f" * 32
        elif fault == "guest_alias":
            docker.container["NetworkSettings"]["Networks"]["fixture"]["Aliases"] = ["cayu-control"]
        elif fault == "foreign_member":
            docker.objects[OTHER] = {"Id": OTHER}
            docker.network["Containers"][OTHER] = {}
        docker.commands.clear()
        with pytest.raises(DockerEgressReconnectError) as error:
            await manager.attach_control_server(TOKEN, NID)
        assert error.value.code == code
        assert all(args[0] in {"inspect", "ps"} for args in docker.commands)
        claim.close()

    asyncio.run(scenario())


def test_superseded_owner_cannot_attach_detach_or_remove(tmp_path):
    async def scenario():
        docker, manager, claim = owned(tmp_path)
        await manager.attach_control_server(TOKEN, NID)
        claim.close()
        replacement = manager.claim(TOKEN)
        replacement.read()
        docker.commands.clear()
        for action in (manager.detach_control_server, manager.remove_network):
            with pytest.raises(DockerEgressReconnectError, match="ownership_uncertain"):
                await action(claim)
        assert not docker.commands
        replacement.close()

    asyncio.run(scenario())


def test_late_attachment_cannot_release_claim_or_dispatch_cleanup(tmp_path):
    async def scenario():
        docker, manager, claim = owned(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(args):
            if args[:2] == ["network", "connect"]:
                entered.set()
                await release.wait()
            return await docker(args)

        manager.adapter._docker_exec = delayed
        manager.timeout_s = 0.02
        with pytest.raises(DockerEgressReconnectError, match="ownership_uncertain"):
            await manager.attach_control_server(TOKEN, NID)
        assert entered.is_set()
        with pytest.raises(DockerEgressReconnectError, match="ownership_uncertain"):
            claim.close()
        with pytest.raises(DockerEgressReconnectError, match="ownership_conflict"):
            manager.claim(TOKEN)
        release.set()
        await asyncio.gather(*manager.pending_commands)
        assert CONTROL in docker.network["Containers"]
        with pytest.raises(DockerEgressReconnectError, match="ownership_uncertain"):
            await manager.detach_control_server(claim)
        assert not any(args[:2] == ["network", "disconnect"] for args in docker.commands)
        assert json.loads((tmp_path / f"{TOKEN}.json").read_text())["pending_mutation"]
        # Test-only release after the simulated daemon task has joined. Runtime
        # deliberately retains the fence for explicit exact-allocation disposal.
        claim.write(pending_mutation=False)
        claim.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("replacement", [None, "same-name", OTHER])
def test_cleanup_rejects_changed_journal_control_identity(tmp_path, replacement):
    async def scenario():
        docker, manager, claim = owned(tmp_path)
        await manager.attach_control_server(TOKEN, NID)
        claim.write(control_server_container_id=replacement)
        docker.commands.clear()
        with pytest.raises(DockerEgressReconnectError, match="configuration_mismatch"):
            await manager.detach_control_server(claim)
        assert not docker.commands
        assert CONTROL in docker.network["Containers"]
        claim.close()

    asyncio.run(scenario())
