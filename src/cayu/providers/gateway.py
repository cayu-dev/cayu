"""Cayu Gateway as a normal Chat Completions provider.

Gateway owns financial admission and settlement. Runtime consumes response
observations and retains its ordinary local execution limits and recovery rules.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.providers._credential_boundary import (
    aclosing_provider_stream,
    detach_provider_call_traceback,
    detach_provider_stream_traceback,
)
from cayu.providers._http import sanitize_provider_cancellation, validate_url
from cayu.providers.base import (
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
)
from cayu.providers.chat_completions import (
    ChatCompletionsProvider,
    ChatCompletionsTransport,
    HttpxChatCompletionsTransport,
)
from cayu.providers.deadlines import ProviderStreamDeadlines

_MAX_LOOKUP_BYTES = 1_048_576


class GatewayTransport(ChatCompletionsTransport, Protocol):
    async def read_json(
        self, *, url: str, headers: Mapping[str, str], timeout_s: float
    ) -> Mapping[str, Any]:
        """Perform one bounded authenticated GET without redirects or retries."""
        ...


class HttpxGatewayTransport(HttpxChatCompletionsTransport):
    async def read_json(
        self, *, url: str, headers: Mapping[str, str], timeout_s: float
    ) -> Mapping[str, Any]:
        url = validate_url(url, "url", provider_label="Gateway", allow_http=self.allow_http)
        async with asyncio.timeout(timeout_s):
            async with self._client.get().stream(
                "GET",
                url,
                headers={**headers, "Accept-Encoding": "identity"},
                timeout=timeout_s,
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    raise ModelProviderError(
                        "Gateway lookup failed.",
                        provider="cayu_gateway",
                        status_code=response.status_code,
                        retryable=False,
                    )
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise ValueError("Encoded lookup responses are unsupported.")
                body = bytearray()
                async for chunk in response.aiter_raw():
                    if len(body) + len(chunk) > _MAX_LOOKUP_BYTES:
                        raise ValueError("Gateway lookup response exceeds its bound.")
                    body.extend(chunk)
        result = json.loads(body)
        if not isinstance(result, dict):
            raise ValueError("Gateway lookup response must be an object.")
        return result


class GatewayProvider(ChatCompletionsProvider):
    """Chat inference plus read-only model discovery and generation lookup.

    ``base_url`` is explicit and includes ``/v1``. Model and usage fields are
    normal provider observations; a reported cost is not a local price estimate.
    Failed POSTs are not automatically retried with a new financial request.
    """

    supports_native_structured_output = True

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        stream_deadlines: ProviderStreamDeadlines | None = None,
        transport: GatewayTransport | None = None,
    ) -> None:
        self.gateway_transport = transport if transport is not None else HttpxGatewayTransport()
        super().__init__(
            name="cayu_gateway",
            base_url=base_url,
            api_key=api_key,
            api_key_env="CAYU_GATEWAY_API_KEY",
            timeout_s=timeout_s,
            stream_deadlines=stream_deadlines,
            transport=self.gateway_transport,
        )
        parts = urlsplit(self.base_url)
        if (
            parts.query
            or parts.fragment
            or parts.username is not None
            or parts.password is not None
        ):
            raise ValueError(
                "Gateway base_url must not contain credentials, a query, or a fragment."
            )

    def preflight_native_structured_output_schema(self, json_schema: dict[str, Any]) -> None:
        from cayu.providers.openai import preflight_openai_native_structured_output_schema

        preflight_openai_native_structured_output_schema(json_schema)

    def _gateway_request(self, request: ModelRequest) -> ModelRequest:
        from cayu.providers.openai import _openai_structured_output_format

        output = _openai_structured_output_format(request.options)
        if output is None:
            return request
        options = copy_json_value(request.options, "options")
        gateway = dict(options.get(self.name, {}))
        if "response_format" in gateway:
            raise ValueError("Native structured output conflicts with response_format.")
        gateway["response_format"] = {
            "type": "json_schema",
            "json_schema": {k: v for k, v in output.items() if k != "type"},
        }
        options[self.name] = gateway
        # model_copy retains the runtime's private peer serialization observer.
        return request.model_copy(update={"options": options})

    def request_footprint_options(self, request: ModelRequest) -> dict[str, Any]:
        return super().request_footprint_options(self._gateway_request(request))

    def request_fingerprint_options(self, request: ModelRequest) -> dict[str, Any]:
        return super().request_fingerprint_options(self._gateway_request(request))

    @detach_provider_stream_traceback
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        events = super().stream(self._gateway_request(request))
        async with aclosing_provider_stream(events):
            async for event in events:
                if event.type is ModelStreamEventType.ERROR:
                    event = event.model_copy(
                        update={"payload": {**event.payload, "retryable": False}}
                    )
                yield event

    @detach_provider_call_traceback
    async def _read(self, path: str) -> dict[str, Any]:
        failure_status = None
        cancellation = None
        try:
            result = await self.gateway_transport.read_json(
                url=f"{self.base_url.rstrip('/')}/{path}",
                headers=self._headers(),
                timeout_s=self.timeout_s,
            )
            return copy_json_value(dict(result), "gateway_lookup")
        except asyncio.CancelledError as exc:
            cancellation = sanitize_provider_cancellation(
                exc, provider_label="Gateway", credential_values=(self.api_key,)
            )
        except Exception as exc:
            if isinstance(exc, ModelProviderError):
                failure_status = exc.status_code
        if cancellation is not None:
            raise cancellation from None
        # Never retain upstream bodies, transport diagnostics or credentials.
        raise ModelProviderError(
            "Gateway lookup failed.",
            provider=self.name,
            status_code=failure_status,
            retryable=False,
        ) from None

    @detach_provider_call_traceback
    async def get_generation(self, request_id: str) -> dict[str, Any]:
        """Observe service-owned status and cost; never replay or settle a request."""
        request_id = require_clean_nonblank(request_id, "request_id")
        if len(request_id) > 128:
            raise ValueError("request_id exceeds 128 characters.")
        result = await self._read("generation?" + urlencode({"id": request_id}))
        data = result.get("data")
        if not isinstance(data, dict) or data.get("id") != request_id:
            raise ModelProviderError(
                "Gateway generation identity mismatch.", provider=self.name, retryable=False
            )
        return data

    @detach_provider_call_traceback
    async def get_models(self) -> list[dict[str, Any]]:
        """Return the models currently visible to this inference key."""
        result = await self._read("models")
        data = result.get("data")
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise ModelProviderError(
                "Gateway model list is invalid.", provider=self.name, retryable=False
            )
        return data
