from __future__ import annotations

import asyncio
import ssl

import httpx
import pytest

from cayu.egress import (
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    HttpxUpstream,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
)
from cayu.egress.proxy_server import TransparentEgressProxyServer


@pytest.mark.parametrize("filename", ["résumé.csv", "报告.csv"])
def test_origin_header_octets_survive_the_proxy(filename: str) -> None:
    async def run() -> None:
        disposition = f'attachment; filename="{filename}"'.encode()

        async def origin(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers=[(b"Content-Disposition", disposition)],
                content=b"report data",
            )

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
                response = await client.get("https://files.example.test/report.csv")
            assert response.status_code == 200
            assert response.content == b"report data"
            assert [
                value
                for key, value in response.headers.raw
                if key.lower() == b"content-disposition"
            ] == [disposition]
        finally:
            await server.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("secret", "wire_encoded"),
    [("résumé-token", False), ("résumé-token", True), ("报告-token", True)],
)
def test_header_redaction_matches_unicode_and_preserved_wire_bytes(secret, wire_encoded):
    from cayu.egress.broker import CapturedResponse, _scrub_response
    from cayu.vaults import REDACTED_SECRET

    value = secret.encode("utf-8").decode("latin-1") if wire_encoded else secret
    response = CapturedResponse(
        status_code=200,
        headers={"x-token": f"prefix-{value}-suffix"},
        set_cookie_headers=(f"token={value}; Secure",),
        body=b"safe body",
    )
    scrubbed = _scrub_response(response, secrets=(secret,), max_body_bytes=1024)
    assert scrubbed.headers["x-token"] == f"prefix-{REDACTED_SECRET}-suffix"
    assert scrubbed.set_cookie_headers == (f"token={REDACTED_SECRET}; Secure",)
    assert scrubbed.body == b"safe body"
