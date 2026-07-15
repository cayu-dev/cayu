"""Opt-in adversarial virtual-egress test for a real E2B sandbox.

The test needs a raw TCP tunnel command because E2B cannot reach the local
Cayu process directly. The command template receives ``{host}`` and ``{port}``;
``CAYU_E2B_PROXY_URL`` must advertise the tunnel as an IPv4-literal URL.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
from urllib.parse import urlsplit

import pytest
from tests.egress_conformance import (
    EgressScenarioEvidence,
    egress_nightly_failure_boundary,
    emit_egress_nightly_evidence,
    registration_for,
)
from tests.egress_e2e_support import CapturingEgressAdapter, drive_adversarial_egress_contract

from cayu.egress.e2b_adapter import E2BEgressAdapter
from cayu.egress.proxy_exposure import ExposedProxy
from cayu.workspaces import E2BWorkspace

pytest.importorskip("cryptography")
pytest.importorskip("e2b")

_TUNNEL_COMMAND = os.environ.get("CAYU_E2B_PROXY_EXPOSURE_COMMAND")
_PROXY_URL = os.environ.get("CAYU_E2B_PROXY_URL")

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_RUN_E2B_EGRESS_E2E") != "1"
    or not os.environ.get("E2B_API_KEY")
    or not _TUNNEL_COMMAND
    or not _PROXY_URL,
    reason=(
        "Set CAYU_RUN_E2B_EGRESS_E2E=1, E2B_API_KEY, "
        "CAYU_E2B_PROXY_EXPOSURE_COMMAND, and CAYU_E2B_PROXY_URL."
    ),
)

REAL_SECRET = "sk_test_51E2BRealSecretNeverInGuest"


class _CommandExposure:
    def __init__(self, command_template: str, proxy_url: str) -> None:
        self._command_template = command_template
        self._proxy_url = proxy_url

    async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
        command = self._command_template.format(host=local_host, port=local_port)
        process = await asyncio.create_subprocess_exec(
            *shlex.split(command),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await self._wait_until_reachable(process)
        except BaseException:
            process.terminate()
            with contextlib.suppress(Exception):
                await process.wait()
            raise

        async def teardown() -> None:
            if process.returncode is not None:
                return
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()

        return ExposedProxy(proxy_url=self._proxy_url, teardown=teardown)

    async def _wait_until_reachable(self, process: asyncio.subprocess.Process) -> None:
        split = urlsplit(self._proxy_url)
        if split.scheme != "http" or split.hostname is None:
            raise RuntimeError("CAYU_E2B_PROXY_URL must be an absolute HTTP proxy URL.")
        port = split.port or 80
        for _ in range(60):
            if process.returncode is not None:
                stderr = await process.stderr.read() if process.stderr is not None else b""
                raise RuntimeError(f"E2B tunnel exited early: {stderr.decode()[:500]}")
            try:
                _reader, writer = await asyncio.open_connection(split.hostname, port)
            except OSError:
                await asyncio.sleep(0.5)
                continue
            writer.close()
            await writer.wait_closed()
            return
        raise RuntimeError("E2B proxy tunnel did not become reachable within 30 seconds.")


async def _drive() -> tuple[EgressScenarioEvidence, ...]:
    assert _TUNNEL_COMMAND is not None
    assert _PROXY_URL is not None
    adapter = CapturingEgressAdapter(
        E2BEgressAdapter(
            exposure=_CommandExposure(_TUNNEL_COMMAND, _PROXY_URL),
        )
    )
    return await drive_adversarial_egress_contract(
        registration=registration_for("e2b"),
        adapter=adapter,
        real_secret=REAL_SECRET,
        image=os.environ.get("CAYU_E2B_TEMPLATE", "base"),
        search_roots=("/home/user/workspace", "/tmp", "/etc/cayu", "/root"),
        response_id="cus_e2b",
        workspace_factory=E2BWorkspace,
    )


@pytest.fixture(scope="module")
def e2e() -> tuple[EgressScenarioEvidence, ...]:
    with egress_nightly_failure_boundary("e2b"):
        return asyncio.run(_drive())


def test_e2b_shared_real_boundary_security_contract(
    e2e: tuple[EgressScenarioEvidence, ...],
) -> None:
    with egress_nightly_failure_boundary("e2b"):
        assert all(item.adapter == "e2b" for item in e2e)
        assert all(item.status == "verified" for item in e2e)
        assert REAL_SECRET not in repr(e2e)
        emit_egress_nightly_evidence(e2e)
