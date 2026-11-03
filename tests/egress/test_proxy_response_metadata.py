from __future__ import annotations

import asyncio
import ssl

import httpx

from cayu.egress import (
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    HttpxUpstream,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
)
from cayu.egress.proxy_server import TransparentEgressProxyServer


def test_head_does_not_claim_a_nonempty_resource_has_zero_length() -> None:
    async def run() -> None:
        async def origin(request: httpx.Request) -> httpx.Response:
            assert request.method == "HEAD"
            return httpx.Response(200, headers={"Content-Length": "1234"})

        policy = BrowserEgressPolicy(name="files", allowed_hosts=["files.example.test"])
        broker = TransparentEgressBroker(
            registry=VirtualCredentialRegistry(),
            policies={policy.name: policy},
            approved_destinations=[ApprovedEgressDestination("files.example.test", policy.name)],
            upstream=HttpxUpstream(
                transport=httpx.MockTransport(origin),
                routes={"files.example.test": "http://192.0.2.10"},
            ),
        )
        server = TransparentEgressProxyServer(broker, loop=asyncio.get_running_loop())
        port = await server.start()
        try:
            context = ssl.create_default_context(cadata=server.authority.ca_cert_pem().decode())
            async with httpx.AsyncClient(
                proxy=f"http://127.0.0.1:{port}",
                verify=context,
                trust_env=False,
                timeout=5,
            ) as client:
                response = await client.head("https://files.example.test/archive.zip")
            assert response.status_code == 200
            assert response.content == b""
            # For HEAD this field is the GET representation's size, not the
            # number of bytes in the HEAD response. Omitting it is also legal.
            assert response.headers.get("content-length") in {None, "1234"}
        finally:
            await server.close()

    asyncio.run(run())
