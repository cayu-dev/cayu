from __future__ import annotations

import asyncio

import pytest

from cayu.egress import UnsupportedEgressError
from cayu.egress.proxy_exposure import HttpProxyEndpoint, MicrosandboxHostProxyExposure


def test_http_proxy_endpoint_centralizes_url_validation() -> None:
    endpoint = HttpProxyEndpoint.parse("http://cayu-egress.example:8443")
    assert endpoint.url == "http://cayu-egress.example:8443"
    assert endpoint.host == "cayu-egress.example"
    assert endpoint.port == 8443

    for invalid in (
        "https://cayu-egress.example:8443",
        "http://user:password@cayu-egress.example:8443",
        "http://cayu-egress.example/path",
        "http://cayu-egress.example:invalid",
    ):
        with pytest.raises(ValueError, match="HTTP proxy URL"):
            HttpProxyEndpoint.parse(invalid)


def test_microsandbox_host_exposure_advertises_the_runtime_host_gateway() -> None:
    async def run() -> str:
        exposed = await MicrosandboxHostProxyExposure().expose(
            local_host="127.0.0.1",
            local_port=8123,
        )
        await exposed.close()
        await exposed.close()
        return exposed.proxy_url

    assert asyncio.run(run()) == "http://host.microsandbox.internal:8123"


def test_microsandbox_host_exposure_rejects_non_loopback_listener() -> None:
    async def run() -> None:
        with pytest.raises(UnsupportedEgressError, match="loopback"):
            await MicrosandboxHostProxyExposure().expose(
                local_host="0.0.0.0",
                local_port=8123,
            )

    asyncio.run(run())
