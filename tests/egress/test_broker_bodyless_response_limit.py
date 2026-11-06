"""Bodyless response metadata must not consume the response-body byte limit."""

import asyncio

import httpx
import pytest
from tests.egress.test_broker_pipeline import _destination_resolver
from tests.egress.test_response_streaming import _with_proxy_client

from cayu.egress import (
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    HttpxUpstream,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
)
from cayu.egress.broker import DEFAULT_EGRESS_BROWSER_MAX_RESPONSE_BYTES


@pytest.mark.parametrize(("method", "status"), [("HEAD", 200), ("GET", 304)])
def test_bodyless_response_can_describe_a_representation_above_the_body_limit(method, status):
    async def run():
        representation_size = DEFAULT_EGRESS_BROWSER_MAX_RESPONSE_BYTES + 1
        received = []

        async def origin(request):
            received.append(request.method)
            # HEAD and 304 have no response body. Their Content-Length is the
            # corresponding GET/200 representation's size (RFC 9110 8.6).
            return httpx.Response(
                status,
                headers={"Content-Length": str(representation_size), "ETag": '"archive-v1"'},
            )

        policy = BrowserEgressPolicy(name="files", allowed_hosts=["files.example.com"])
        broker = TransparentEgressBroker(
            registry=VirtualCredentialRegistry(),
            policies={policy.name: policy},
            approved_destinations=[ApprovedEgressDestination("files.example.com", policy.name)],
            upstream=HttpxUpstream(
                transport=httpx.MockTransport(origin),
                destination_resolver=_destination_resolver("93.184.216.34"),
            ),
        )

        async def use(client):
            response = await client.request(
                method,
                "https://files.example.com/archive.zip",
                headers={"If-None-Match": '"archive-v1"'} if status == 304 else {},
            )
            assert received == [method]
            assert response.status_code == status, response.headers
            assert response.content == b""
            assert response.headers["etag"] == '"archive-v1"'

        await _with_proxy_client(broker, use)

    asyncio.run(run())
