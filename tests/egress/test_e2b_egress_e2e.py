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
from typing import Any
from urllib.parse import urlsplit

import pytest
from tests.egress_e2e_support import (
    assert_brokered_provider_call,
    assert_direct_egress_blocked,
    assert_guest_non_possession,
    drive_adversarial_egress_contract,
)

from cayu.egress.e2b_adapter import E2BEgressAdapter
from cayu.egress.proxy_exposure import ExposedProxy

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


class _CapturingAdapter(E2BEgressAdapter):
    broker: Any = None
    grant: Any = None

    async def prepare(self, *, session_id, grants, broker):  # type: ignore[no-untyped-def]
        self.broker = broker
        self.grant = grants[0]
        return await super().prepare(session_id=session_id, grants=grants, broker=broker)


async def _drive() -> dict[str, Any]:
    assert _TUNNEL_COMMAND is not None
    assert _PROXY_URL is not None
    adapter = _CapturingAdapter(
        exposure=_CommandExposure(_TUNNEL_COMMAND, _PROXY_URL),
    )
    return await drive_adversarial_egress_contract(
        adapter=adapter,
        real_secret=REAL_SECRET,
        image=os.environ.get("CAYU_E2B_TEMPLATE", "base"),
        session_prefix="e2b-egress",
        search_roots=("/home/user/workspace", "/tmp", "/etc/cayu", "/root"),
        response_id="cus_e2b",
    )


@pytest.fixture(scope="module")
def e2e() -> dict[str, Any]:
    return asyncio.run(_drive())


def test_e2b_guest_possesses_only_virtual_credential(e2e: dict[str, Any]) -> None:
    assert e2e["env"]["HTTPS_PROXY"] == _PROXY_URL
    assert_guest_non_possession(e2e, REAL_SECRET)


def test_e2b_allowed_call_uses_broker_secret(e2e: dict[str, Any]) -> None:
    assert_brokered_provider_call(e2e, real_secret=REAL_SECRET, response_id="cus_e2b")


def test_e2b_direct_and_metadata_egress_are_blocked(e2e: dict[str, Any]) -> None:
    assert_direct_egress_blocked(e2e)
