"""Fresh-process failures before/after a real control attachment mutation."""

import asyncio
import json
import os
import sys
from pathlib import Path

from cayu import ApprovedEgressDestination
from cayu.egress import HttpEgressPolicy
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.egress.errors import DockerEgressReconnectError
from cayu.environments import EnvironmentFactoryOperation, EnvironmentFactoryRequest
from cayu.runtime.egress import VirtualEgressEnvironmentFactory


async def main(mode):
    root = Path(os.environ["CAYU_DEMO_STATE"])
    control = json.loads((root / "configuration.json").read_text())["control_server_container_id"]
    adapter = DockerEgressAdapter(
        reconnect_state_dir=root / "crash-ownership", control_server_container_id=control
    )
    execute = adapter._docker_exec

    async def crash(args):
        target = args[:2] == ["network", "connect"] and args[-1] == control
        if target and mode == "before_attach":
            os._exit(0)
        result = await execute(args)
        if target and mode == "after_attach":
            assert result[0] == 0
            os._exit(0)
        return result

    adapter._docker_exec = crash
    factory = VirtualEgressEnvironmentFactory(
        adapter=adapter,
        credentials=[],
        image="python:3.12-slim",
        policies={
            "fixture": HttpEgressPolicy(
                name="fixture", allowed_hosts=("fixture.test",), allowed_endpoints=(("GET", "/"),)
            )
        },
        approved_destinations=(
            ApprovedEgressDestination(destination="fixture.test", policy_name="fixture"),
        ),
    )
    path = root / "allocation.json"
    metadata = json.loads(path.read_text()) if mode != "create" else {}
    request = EnvironmentFactoryRequest(
        session_id="crash",
        agent_name="fixture",
        environment_name="docker",
        operation=EnvironmentFactoryOperation.CREATE
        if mode == "create"
        else EnvironmentFactoryOperation.RECONNECT,
        reconnect_metadata=metadata,
    )
    if mode == "recover":
        try:
            await factory.create(request)
        except DockerEgressReconnectError as error:
            assert error.code == "ownership_uncertain"
        else:
            raise AssertionError("An acknowledgement-ambiguous attachment was admitted.")
        inspected = await adapter._reconnect.inspect(
            "container", metadata["identity"]["container_id"]
        )
        assert inspected["State"]["Paused"]
        assert (await adapter._reconnect.inspect("container", control))["State"]["Running"]
        print("fenced")
        return
    result = await factory.create(request)
    assert mode == "create", "Crash injection did not execute."
    path.write_text(json.dumps(result.reconnect_metadata))
    await result.environment.runner.finalize(outcome="interrupted")
    print("retained")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
