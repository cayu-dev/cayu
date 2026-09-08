"""Real Docker fault injection around the exact allocation-owned attachment."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from cayu.egress import HttpEgressPolicy, TransparentEgressBroker, VirtualCredentialRegistry
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.egress.errors import DockerEgressReconnectError


async def main():
    root = Path(os.environ["CAYU_DEMO_STATE"])
    control = json.loads((root / "configuration.json").read_text())["control_server_container_id"]
    adapter = DockerEgressAdapter(
        reconnect_state_dir=root / "fault-ownership", control_server_container_id=control
    )
    execute = adapter._docker_exec
    failure = None

    async def fault(args):
        is_attach = args[:2] == ["network", "connect"] and args[-1] == control
        is_detach = args[:2] == ["network", "disconnect"] and args[-1] == control
        if is_attach and failure == "before_attach":
            return 1, "synthetic dispatch refusal"
        result = await execute(args)
        if (is_attach and failure == "after_attach") or (is_detach and failure == "after_detach"):
            return 1, "synthetic lost acknowledgement"
        return result

    adapter._docker_exec = fault

    def broker():
        return TransparentEgressBroker(
            registry=VirtualCredentialRegistry(),
            policies={
                "empty": HttpEgressPolicy(
                    name="empty", allowed_hosts=("fixture.test",), allowed_endpoints=(("GET", "/"),)
                )
            },
        )

    async def networks():
        state = await adapter._reconnect.inspect("container", control)
        return {value["NetworkID"] for value in state["NetworkSettings"]["Networks"].values()}

    original = await networks()
    other = await adapter.prepare(session_id="other", grants=(), broker=broker())
    other_network = other.network
    try:
        assert await networks() == original | {other_network}
        for case in ("before_attach", "after_attach"):
            failure = case
            try:
                await adapter.prepare(session_id=case, grants=(), broker=broker())
            except DockerEgressReconnectError as error:
                assert error.code == "daemon_unavailable"
            else:
                raise AssertionError("Fault did not refuse admission.")
            assert await networks() == original | {other_network}
        failure = None
        first = await adapter.prepare(session_id="first", grants=(), broker=broker())
        assert await networks() == original | {first.network, other_network}
        claim = adapter._reconnect.bindings[id(first)]
        competing = DockerEgressAdapter(
            reconnect_state_dir=root / "fault-ownership", control_server_container_id=control
        )
        try:
            competing._reconnect.claim(claim.token)
        except DockerEgressReconnectError as error:
            assert error.code == "ownership_conflict"
        else:
            raise AssertionError("Duplicate worker acquired a live attachment.")
        failure = "after_detach"
        try:
            await first.close()
        except DockerEgressReconnectError as error:
            assert error.code == "daemon_unavailable"
        else:
            raise AssertionError("Lost detach acknowledgement was treated as success.")
        assert await networks() == original | {other_network}
        assert not claim.closed
        failure = None
        await first.close()
        assert claim.closed
        try:
            await adapter._reconnect.detach_control_server(claim)
        except DockerEgressReconnectError as error:
            assert error.code == "ownership_uncertain"
        else:
            raise AssertionError("Stale owner detached a current endpoint.")
        assert await networks() == original | {other_network}
    finally:
        failure = None
        await other.close()
    assert await networks() == original
    assert (await adapter._reconnect.inspect("container", control))["State"]["Running"]
    print(
        json.dumps(
            {
                "two_allocations": True,
                "lost_attach_ack": True,
                "lost_detach_ack": True,
                "duplicate_owner_denied": True,
                "stale_cleanup_denied": True,
                "server_survived": True,
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
