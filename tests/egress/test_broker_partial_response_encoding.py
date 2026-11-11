"""Partial responses must keep byte offsets in the delivered representation."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from tests.egress.test_broker_pipeline import _ENCODERS, _destination_resolver

from cayu.egress import CapturedRequest, HttpxUpstream
from cayu.egress.broker import _UpstreamUnsupportedEncodingError


@pytest.mark.parametrize("coding", [None, "identity", *sorted(_ENCODERS)])
def test_partial_response_only_accepts_identity_encoding(coding: str | None) -> None:
    payload = b"downloaded document\n" * 500
    encoded = _ENCODERS[coding](payload) if coding in _ENCODERS else payload
    yielded: list[int] = []
    content_range = f"bytes 0-{len(encoded) - 1}/{len(encoded)}"

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yielded.append(len(encoded))
            yield encoded

    async def origin(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        assert request.headers["range"] == "bytes=0-65535"
        return httpx.Response(
            206,
            headers={
                "Content-Range": content_range,
                "Content-Length": str(len(encoded)),
                **({"Content-Encoding": coding} if coding is not None else {}),
            },
            stream=Body(),
            request=request,
        )

    upstream = HttpxUpstream(
        transport=httpx.MockTransport(origin),
        destination_resolver=_destination_resolver("93.184.216.34"),
    )
    request = CapturedRequest(
        method="GET", host="files.example.com", path="/document", headers={"Range": "bytes=0-65535"}
    )
    if coding in _ENCODERS:
        with pytest.raises(_UpstreamUnsupportedEncodingError):
            asyncio.run(upstream.send(request))
        assert yielded == []
    else:
        response = asyncio.run(upstream.send(request))
        assert response.status_code == 206
        assert response.headers["content-range"] == content_range
        assert response.body == payload
        assert "content-encoding" not in response.headers
