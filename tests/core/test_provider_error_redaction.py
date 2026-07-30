from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from cayu.providers.anthropic import _safe_error_response_text as anthropic_error_text
from cayu.providers.chat_completions import (
    _safe_error_response_text as chat_completions_error_text,
)
from cayu.providers.openai import _safe_error_response_text as openai_error_text
from cayu.providers.vertex import _safe_error_response_text as vertex_error_text

_BUNDLED_HTTP_ERROR_FORMATTERS: tuple[Callable[[httpx.Response], str], ...] = (
    openai_error_text,
    anthropic_error_text,
    chat_completions_error_text,
    vertex_error_text,
)


@pytest.mark.parametrize("format_error", _BUNDLED_HTTP_ERROR_FORMATTERS)
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(
            500,
            request=httpx.Request("POST", "https://provider.invalid"),
            headers={"content-type": "application/json"},
            content=(
                b'{"error":{"type":"server_error","message":'
                b'"prefix-provider\\u002derror\\u002dsecret-suffix"}}'
            ),
        ),
        httpx.Response(
            500,
            request=httpx.Request("POST", "https://provider.invalid"),
            headers={"content-type": "application/json"},
            text='{"error":{"message":"x' + "provider-error-secret"[:10],
        ),
        httpx.Response(
            500,
            request=httpx.Request("POST", "https://provider.invalid"),
            headers={"content-type": "text/plain"},
            text="x" * 1_995 + "provider-error-secret",
        ),
    ],
    ids=("json-escaped", "malformed-json", "unstructured"),
)
def test_bundled_provider_http_errors_omit_untrusted_bodies(
    format_error: Callable[[httpx.Response], str],
    response: httpx.Response,
) -> None:
    rendered = format_error(response)

    assert rendered == "[provider response body omitted]"
    assert "provider-error-secret" not in rendered
    assert "provider-error" not in rendered
