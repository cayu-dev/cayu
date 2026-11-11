"""Body transformations must not leave byte-range metadata for different bytes."""

import asyncio
import gzip

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


def test_decoding_a_partial_response_preserves_consistent_byte_range_metadata():
    async def run():
        payload = b"downloaded document\n" * 500
        encoded = gzip.compress(payload, mtime=0)
        observed = []

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield encoded

        async def origin(request):
            observed.append((request.headers["range"], request.headers["accept-encoding"]))
            # A stored gzip representation smaller than the requested range.
            # This is a complete, valid 206: the offsets refer to the encoded
            # representation and the wire body has exactly the announced span.
            return httpx.Response(
                206,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Encoding": "gzip",
                    "Content-Length": str(len(encoded)),
                    "Content-Range": f"bytes 0-{len(encoded) - 1}/{len(encoded)}",
                    "Accept-Ranges": "bytes",
                },
                stream=Body(),
                request=request,
            )

        policy = BrowserEgressPolicy(name="downloads", allowed_hosts=["files.example.com"])
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
            async with client.stream(
                "GET", "https://files.example.com/document", headers={"Range": "bytes=0-65535"}
            ) as response:
                # Read wire bytes: a guest decoder must not disguise whether
                # the proxy's Content-Range describes the bytes it transmitted.
                wire_body = b"".join([chunk async for chunk in response.aiter_raw()])
                assert response.status_code in {200, 206, 502}
                if response.status_code == 206:
                    span, _total = (
                        response.headers["content-range"].removeprefix("bytes ").split("/")
                    )
                    first, last = map(int, span.split("-"))
                    assert len(wire_body) == last - first + 1, (
                        response.headers,
                        len(wire_body),
                    )

        await _with_proxy_client(broker, use)
        assert observed == [("bytes=0-65535", "identity")]

    asyncio.run(run())
