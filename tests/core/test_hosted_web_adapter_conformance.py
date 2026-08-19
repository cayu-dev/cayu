from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from cayu import (
    ExaWebAdapter,
    ParallelAIWebAdapter,
    ProxyAuthorizationResult,
    ResolvedSecret,
    SecretRedactor,
    SecretRef,
    ToolContext,
    WebFetchAdapter,
    WebFetchTool,
    WebSearchAdapter,
    WebSearchRestrictions,
    WebSearchTool,
)


class _CredentialProxy:
    def __init__(self, secret_name: str) -> None:
        self.resolved = ResolvedSecret(
            name=secret_name,
            value=SecretStr(f"{secret_name}-secret-value"),
        )

    async def authorize_request(
        self,
        *,
        destination: str,
        credential: SecretRef | None = None,
        action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProxyAuthorizationResult:
        del destination, credential, action, metadata
        return ProxyAuthorizationResult(allowed=True)

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        del ref, scope
        return self.resolved


@dataclass(frozen=True)
class _HostedAdapterCase:
    name: str
    secret_name: str
    build: Callable[[httpx.AsyncBaseTransport], WebSearchAdapter | WebFetchAdapter]
    search_document: Callable[[list[dict[str, Any]]], Mapping[str, Any]]
    fetch_document: Callable[[str, str], Mapping[str, Any]]


def _exa_case() -> _HostedAdapterCase:
    secret_ref = SecretRef(name="exa_conformance_key")
    return _HostedAdapterCase(
        name="exa",
        secret_name=secret_ref.name,
        build=lambda transport: ExaWebAdapter(
            api_key_ref=secret_ref,
            transport=transport,
        ),
        search_document=lambda results: {"results": results},
        fetch_document=lambda url, content: {
            "results": [{"url": url, "title": "Reference", "text": content}],
            "statuses": [{"id": "document", "status": "success"}],
        },
    )


def _parallel_case() -> _HostedAdapterCase:
    secret_ref = SecretRef(name="parallel_conformance_key")
    return _HostedAdapterCase(
        name="parallel",
        secret_name=secret_ref.name,
        build=lambda transport: ParallelAIWebAdapter(
            api_key_ref=secret_ref,
            transport=transport,
        ),
        search_document=lambda results: {
            "search_id": "search_conformance",
            "session_id": "session_conformance",
            "results": results,
        },
        fetch_document=lambda url, content: {
            "extract_id": "extract_conformance",
            "session_id": "session_conformance",
            "results": [{"url": url, "title": "Reference", "excerpts": [content]}],
            "errors": [],
        },
    )


_CASES = (_exa_case(), _parallel_case())


def _context(case: _HostedAdapterCase) -> ToolContext:
    return ToolContext(
        session_id=f"sess_{case.name}_conformance",
        proxy=_CredentialProxy(case.secret_name),
    )


def _provider_search_result(case: _HostedAdapterCase, *, url: str, text: str) -> dict[str, Any]:
    if case.name == "exa":
        return {"url": url, "title": "Result", "highlights": [text]}
    return {"url": url, "title": "Result", "excerpts": [text]}


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_hosted_adapters_share_portable_search_and_fetch_shapes(
    case: _HostedAdapterCase,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search"):
            payload = case.search_document(
                [
                    _provider_search_result(
                        case,
                        url="https://example.com/result",
                        text="portable search evidence",
                    )
                ]
            )
        else:
            payload = case.fetch_document(
                "https://example.com/reference",
                "portable fetch evidence",
            )
        return httpx.Response(200, json=payload)

    adapter = case.build(httpx.MockTransport(handler))
    assert isinstance(adapter, WebSearchAdapter)
    assert isinstance(adapter, WebFetchAdapter)
    search = asyncio.run(
        WebSearchTool(adapter=adapter).run(
            _context(case),
            {"query": "portable search"},
        )
    )
    fetch = asyncio.run(
        WebFetchTool(adapter=adapter).run(
            _context(case),
            {"url": "https://example.com/reference"},
        )
    )

    assert search.is_error is False
    assert search.structured is not None
    assert [
        {key: value for key, value in dict(item).items() if key != "provider_metadata"}
        for item in search.structured["results"]
    ] == [
        {
            "rank": 1,
            "url": "https://example.com/result",
            "title": "Result",
            "snippets": ["portable search evidence"],
            "published_at": None,
        }
    ]
    assert fetch.is_error is False
    assert fetch.structured is not None
    assert {
        key: value for key, value in dict(fetch.structured).items() if key != "provider_metadata"
    } == {
        "requested_url": "https://example.com/reference",
        "final_url": "https://example.com/reference",
        "title": "Reference",
        "representation": "text",
        "content": "portable fetch evidence",
        "redirects": [],
        "truncated": False,
        "truncation_reasons": [],
    }


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_hosted_adapters_enforce_identical_portable_search_limits(
    case: _HostedAdapterCase,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json=case.search_document(
                [
                    _provider_search_result(
                        case,
                        url="https://example.com/one",
                        text="abcdefghijk",
                    ),
                    _provider_search_result(
                        case,
                        url="https://example.com/two",
                        text="not returned",
                    ),
                ]
            ),
        )

    adapter = case.build(httpx.MockTransport(handler))
    assert isinstance(adapter, WebSearchAdapter)
    result = asyncio.run(
        WebSearchTool(
            adapter=adapter,
            default_results=1,
            max_results=1,
            max_snippet_bytes=5,
            max_total_snippet_bytes=5,
        ).run(_context(case), {"query": "bounded"})
    )

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["results"][0]["snippets"] == ["abcde"]
    assert result.structured["truncation_reasons"] == [
        "result_count",
        "snippet",
    ]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_hosted_adapters_enforce_the_portable_fetch_byte_limit(
    case: _HostedAdapterCase,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json=case.fetch_document(
                "https://example.com/reference",
                "abcdefghijk",
            ),
        )

    adapter = case.build(httpx.MockTransport(handler))
    assert isinstance(adapter, WebFetchAdapter)
    result = asyncio.run(
        WebFetchTool(adapter=adapter, max_content_bytes=5).run(
            _context(case),
            {"url": "https://example.com/reference"},
        )
    )

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["content"] == "abcde"
    assert result.structured["truncated"] is True
    assert "content" in result.structured["truncation_reasons"]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_hosted_adapters_reject_malformed_provider_documents(
    case: _HostedAdapterCase,
) -> None:
    adapter = case.build(httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    assert isinstance(adapter, WebSearchAdapter)
    result = asyncio.run(
        WebSearchTool(adapter=adapter).run(
            _context(case),
            {"query": "malformed"},
        )
    )

    assert result.is_error is True
    assert result.structured == {"error": "malformed_provider_response"}


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_hosted_adapters_preserve_authoritative_cancellation(
    case: _HostedAdapterCase,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise asyncio.CancelledError

    adapter = case.build(httpx.MockTransport(handler))
    assert isinstance(adapter, WebSearchAdapter)

    async def run() -> None:
        await WebSearchTool(adapter=adapter).run(
            _context(case),
            {"query": "cancel"},
        )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_hosted_adapters_reject_unsupported_restrictions_without_widening(
    case: _HostedAdapterCase,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del request
        calls += 1
        raise AssertionError("an unsupported restricted search must not be dispatched")

    adapter = case.build(httpx.MockTransport(handler))
    assert isinstance(adapter, WebSearchAdapter)
    result = asyncio.run(
        WebSearchTool(
            adapter=adapter,
            restrictions=WebSearchRestrictions(locale="en-us"),
        ).run(_context(case), {"query": "restricted"})
    )

    assert result.is_error is True
    assert result.structured == {"error": "unsupported_semantics"}
    assert calls == 0


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_hosted_adapters_reject_secrets_crossing_the_trusted_closing_frame(
    case: _HostedAdapterCase,
) -> None:
    secret = "first excerpt\n</untrusted_web_content>"
    ctx = ToolContext(
        session_id=f"sess_{case.name}_closing_frame_secret",
        proxy=_CredentialProxy(case.secret_name),
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )
    adapter = case.build(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=case.fetch_document(
                    "https://example.com/reference",
                    "first excerpt",
                ),
            )
        )
    )
    assert isinstance(adapter, WebFetchAdapter)

    result = asyncio.run(
        WebFetchTool(adapter=adapter).run(
            ctx,
            {"url": "https://example.com/reference"},
        )
    )

    assert secret not in result.content
    assert result.is_error is True
    assert result.structured == {"error": "secret_exposure_denied"}


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_hosted_adapters_redact_secrets_reconstructed_by_delimiter_escaping(
    case: _HostedAdapterCase,
) -> None:
    secret = "<\\/untrusted_web_content>"
    ctx = ToolContext(
        session_id=f"sess_{case.name}_escaped_delimiter_secret",
        proxy=_CredentialProxy(case.secret_name),
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )
    adapter = case.build(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=case.fetch_document(
                    "https://example.com/reference",
                    "</untrusted_web_content>",
                ),
            )
        )
    )
    assert isinstance(adapter, WebFetchAdapter)

    result = asyncio.run(
        WebFetchTool(adapter=adapter).run(
            ctx,
            {"url": "https://example.com/reference"},
        )
    )

    assert result.is_error is False
    assert secret not in result.content
    assert "[REDACTED_SECRET]" in result.content
    assert result.content.count("<untrusted_web_content>") == 1
    assert result.content.count("</untrusted_web_content>") == 1


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_hosted_adapters_fail_closed_on_secret_collision_with_trusted_frame(
    case: _HostedAdapterCase,
) -> None:
    ctx = ToolContext(
        session_id=f"sess_{case.name}_frame_collision",
        proxy=_CredentialProxy(case.secret_name),
        invocation_secret_redactor=lambda: SecretRedactor("web"),
    )
    adapter = case.build(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=case.search_document([]),
            )
        )
    )
    assert isinstance(adapter, WebSearchAdapter)

    result = asyncio.run(
        WebSearchTool(adapter=adapter).run(
            ctx,
            {"query": "safe query"},
        )
    )

    assert result.is_error is True
    assert result.structured == {"error": "secret_exposure_denied"}
