"""Separate-process Docker reconnect fixture; no paid providers or external accounts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

from tests.egress.microsandbox_reconnect_worker import _factory
from tests.egress_e2e_support import RecordingProviderUpstream, connect_probe_script

from cayu.egress import CapturedRequest
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.environments import EnvironmentFactoryOperation, EnvironmentFactoryRequest
from cayu.runners import ExecCommand


class CapturingDocker(DockerEgressAdapter):
    async def prepare(self, **kwargs):
        self.broker = kwargs["broker"]
        self.grants = kwargs["grants"]
        self.binding = await super().prepare(**kwargs)
        return self.binding

    async def prepare_reconnect(self, **kwargs):
        self.broker = kwargs["broker"]
        self.grants = kwargs["grants"]
        self.binding = await super().prepare_reconnect(**kwargs)
        return self.binding


async def main(mode: str, state_dir: Path) -> None:
    adapter = CapturingDocker(reconnect_state_dir=state_dir)
    upstream = RecordingProviderUpstream("fixture-after-reconnect")
    factory = _factory(adapter=adapter, upstream=upstream, image="python:3.12-slim")
    previous = json.loads(sys.stdin.readline()) if mode == "reconnect" else None
    result = await factory.create(
        EnvironmentFactoryRequest(
            session_id="docker-reconnect-fixture",
            agent_name="fixture",
            environment_name="docker",
            operation=EnvironmentFactoryOperation.RECONNECT
            if previous
            else EnvironmentFactoryOperation.CREATE,
            reconnect_metadata={} if previous is None else previous["metadata"],
        )
    )
    runner = result.environment.runner
    assert runner is not None
    if previous is None:
        write = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                "from pathlib import Path; Path('/workspace/reconnect-sentinel').write_text('same allocation')",
            )
        )
        assert write.exit_code == 0
        evidence = {
            "metadata": result.reconnect_metadata,
            "old_credential": adapter.grants[0].presented_value,
            "ca_sha256": hashlib.sha256(adapter.binding.ca_cert_pem).hexdigest(),
        }
        assert evidence["old_credential"] not in json.dumps(result.reconnect_metadata)
        assert "PRIVATE KEY" not in json.dumps(result.reconnect_metadata)
        if mode == "retain":
            await runner.finalize(outcome="interrupted")
        print(json.dumps(evidence), flush=True)
        if mode == "crash":
            os._exit(0)
        return
    try:
        assert result.reconnect_metadata == previous["metadata"]
        assert hashlib.sha256(adapter.binding.ca_cert_pem).hexdigest() != previous["ca_sha256"]
        read = await runner.exec(ExecCommand.process("cat", "/workspace/reconnect-sentinel"))
        assert read.exit_code == 0 and read.stdout == "same allocation"
        request = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                "import os,urllib.request as u; "
                "r=u.Request('https://api.stripe.com/v1/customers',data=b'x=1',"
                "headers={'Authorization':'Bearer '+os.environ['STRIPE_SECRET_KEY']}); "
                "print(u.urlopen(r,timeout=10).read().decode())",
            ),
            timeout_s=20,
        )
        assert request.exit_code == 0 and "fixture-after-reconnect" in request.stdout
        direct = await runner.exec(
            ExecCommand.process(
                "python3", "-c", connect_probe_script("1.1.1.1", 443, probe_kind="tls")
            ),
            timeout_s=15,
        )
        assert direct.exit_code == 0 and not json.loads(direct.stdout)["tcp_connected"]
        assert adapter.grants[0].presented_value != previous["old_credential"]
        stale = await adapter.broker.handle_request(
            CapturedRequest(
                method="POST",
                host="api.stripe.com",
                path="/v1/customers",
                headers={"Authorization": "Bearer " + previous["old_credential"]},
            )
        )
        assert stale.status_code == 403
        print(
            json.dumps(
                {
                    "same_allocation": True,
                    "sentinel": True,
                    "fresh_tls": True,
                    "stale_credentials_denied": True,
                    "direct_egress_denied": True,
                }
            ),
            flush=True,
        )
    finally:
        await runner.finalize(outcome="completed")
    assert (
        await adapter._reconnect.inspect(
            "container", result.reconnect_metadata["identity"]["container_id"], absent_ok=True
        )
        is None
    )
    assert (
        await adapter._reconnect.inspect(
            "network", result.reconnect_metadata["identity"]["network_id"], absent_ok=True
        )
        is None
    )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
