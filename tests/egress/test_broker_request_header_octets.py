"""A response's valid obs-text ETag can be used for a conditional request."""

import asyncio

import httpx
from tests.egress.test_broker_pipeline import _destination_resolver
from tests.egress.test_response_streaming import _with_proxy_client

from cayu.egress import (
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    HttpxUpstream,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
)


def test_conditional_request_preserves_non_ascii_etag_octets():
    async def run():
        # RFC 9110's etagc includes obs-text (0x80-0xff).
        etag = b'"revision-\xff"'
        received = []

        async def origin(request):
            condition = next(
                (v for k, v in request.headers.raw if k.lower() == b"if-none-match"), None
            )
            received.append(condition)
            if condition == etag:
                return httpx.Response(304)
            return httpx.Response(200, headers=[(b"ETag", etag)], content=b"document")

        broker = TransparentEgressBroker(
            registry=VirtualCredentialRegistry(),
            policies={"docs": BrowserEgressPolicy(name="docs", allowed_hosts=["docs.example.com"])},
            approved_destinations=[
                ApprovedEgressDestination(destination="docs.example.com", policy_name="docs")
            ],
            upstream=HttpxUpstream(
                transport=httpx.MockTransport(origin),
                destination_resolver=_destination_resolver("93.184.216.34"),
            ),
        )

        async def use(client):
            first = await client.get("https://docs.example.com/document")
            assert first.status_code == 200
            assert next(v for k, v in first.headers.raw if k.lower() == b"etag") == etag
            # Pass bytes, as a browser/cache does, rather than asking the guest's
            # HTTPX client to encode a non-ASCII Python header string.
            second = await client.get(
                "https://docs.example.com/document", headers=[(b"If-None-Match", etag)]
            )
            assert second.status_code == 304, second.content
            assert received == [None, etag]

        await _with_proxy_client(broker, use)

    asyncio.run(run())
