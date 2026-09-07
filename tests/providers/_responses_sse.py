"""Credential-free bounded synthetic Responses SSE framing."""

import json

import httpx


class ChunkedSSE(httpx.AsyncByteStream):
    def __init__(self, events):
        self.body = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)

    async def __aiter__(self):
        for offset in range(0, len(self.body), 23):
            yield self.body[offset : offset + 23]
