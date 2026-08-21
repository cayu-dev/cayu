from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import date
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from cayu import (
    AgentSpec,
    AllowlistProxy,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    ParallelAIWebAdapter,
    ProxyAuthorizationResult,
    ResolvedSecret,
    RunRequest,
    ScriptedModelProvider,
    SecretRedactor,
    SecretRef,
    StaticVault,
    ToolContext,
    WebFetchTool,
    WebSearchRestrictions,
    WebSearchTool,
)
from cayu.tools.web_access import web_destination_fingerprint

_API_KEY = "parallel-test-secret-value"
_API_KEY_REF = SecretRef(name="parallel_api_key")


class _CredentialProxy:
    def __init__(
        self,
        *,
        allowed: bool = True,
        resolved: ResolvedSecret | None = None,
    ) -> None:
        self.allowed = allowed
        self.resolved = resolved or ResolvedSecret(
            name="parallel_api_key",
            value=SecretStr(_API_KEY),
        )
        self.authorizations: list[dict[str, Any]] = []
        self.resolutions: list[tuple[SecretRef, dict[str, Any] | None]] = []

    async def authorize_request(
        self,
        *,
        destination: str,
        credential: SecretRef | None = None,
        action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProxyAuthorizationResult:
        self.authorizations.append(
            {
                "destination": destination,
                "credential": credential,
                "action": action,
                "metadata": metadata,
            }
        )
        return ProxyAuthorizationResult(
            allowed=self.allowed,
            reason=None if self.allowed else "not admitted",
        )

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        self.resolutions.append((ref, scope))
        return self.resolved


class _ByteStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class _UnreadableStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        raise AssertionError("A non-success provider body must remain unread.")
        yield b""  # pragma: no cover - makes this an async generator


class _RegisterSecretOnCloseStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes, register_secret: Callable[[], None]) -> None:
        self.body = body
        self.register_secret = register_secret

    async def __aiter__(self):
        yield self.body

    async def aclose(self) -> None:
        self.register_secret()


def _context(proxy: _CredentialProxy | None = None) -> ToolContext:
    return ToolContext(session_id="sess_parallel", proxy=proxy)


def _json_response(payload: Any, *, status_code: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(
        status_code,
        headers={"content-type": "application/json", **headers},
        json=payload,
    )


def _search_document(*, results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "search_id": "search_123",
        "session_id": "session_123",
        "results": results,
    }


def _extract_document(
    *,
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "extract_id": "extract_123",
        "session_id": "session_123",
        "results": results,
        "errors": [] if errors is None else errors,
    }


def test_parallel_search_translates_fixed_policy_and_returns_portable_results() -> None:
    proxy = _CredentialProxy()
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        assert request.url == "https://api.parallel.ai/v1/search"
        assert request.headers["x-api-key"] == _API_KEY
        assert json.loads(request.content) == {
            "objective": "Prefer primary sources.\n\nSearch request: cayu production runtime",
            "search_queries": ["cayu production runtime"],
            "mode": "fast",
            "max_chars_total": 100,
            "advanced_settings": {
                "excerpt_settings": {"max_chars_per_result": 40},
                "max_results": 2,
                "source_policy": {
                    "include_domains": ["example.com", ".gov"],
                    "exclude_domains": ["archive.example.com"],
                    "after_date": "2025-01-01",
                },
                "location": "us",
                "fetch_policy": {
                    "max_age_seconds": 3_600,
                    "timeout_seconds": 5.0,
                    "disable_cache_fallback": True,
                },
            },
        }
        return _json_response(
            {
                "search_id": "search_123",
                "session_id": "session_123",
                "results": [
                    {
                        "url": "HTTPS://EXAMPLE.COM/runtime#section",
                        "title": "  Cayu runtime  ",
                        "publish_date": "2026-08-17",
                        "excerpts": [" Durable   sessions ", "Bounded recovery"],
                    },
                    {
                        "url": "https://docs.example.com/cayu",
                        "title": None,
                        "publish_date": None,
                        "excerpts": ["Provider-neutral web tools"],
                    },
                ],
                "warnings": [
                    {
                        "type": "input_validation_warning",
                        "message": " provider warning ",
                        "detail": {"private": "not retained"},
                    }
                ],
                "usage": [
                    {"name": "sku_search", "count": 1},
                    {"name": "sku_search_additional_results", "count": 2},
                ],
            }
        )

    adapter = ParallelAIWebAdapter(
        api_key_ref=_API_KEY_REF,
        search_mode="fast",
        search_location="US",
        search_objective=" Prefer   primary sources. ",
        search_fetch_max_age_seconds=3_600,
        search_disable_cache_fallback=True,
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(
        WebSearchTool(
            adapter=adapter,
            default_results=2,
            max_results=2,
            max_snippet_bytes=40,
            max_total_snippet_bytes=100,
            timeout_seconds=5,
            restrictions=WebSearchRestrictions(
                include_domains=("EXAMPLE.COM", ".GOV"),
                exclude_domains=("archive.example.com",),
                published_on_or_after=date(2025, 1, 1),
            ),
        ).run(
            _context(proxy),
            {"query": "cayu production runtime", "num_results": 2},
        )
    )

    assert result.is_error is False
    assert result.structured == {
        "query": "cayu production runtime",
        "results": [
            {
                "rank": 1,
                "url": "https://example.com/runtime",
                "title": "Cayu runtime",
                "snippets": ["Durable sessions", "Bounded recovery"],
                "published_at": "2026-08-17",
            },
            {
                "rank": 2,
                "url": "https://docs.example.com/cayu",
                "title": "https://docs.example.com/cayu",
                "snippets": ["Provider-neutral web tools"],
                "published_at": None,
            },
        ],
        "truncated": False,
        "truncation_reasons": [],
        "provider_metadata": {
            "parallel": {
                "request_id": "search_123",
                "session_id": "session_123",
                "warnings": [
                    {
                        "type": "input_validation_warning",
                        "message": "provider warning",
                    }
                ],
                "usage": [
                    {"name": "sku_search", "count": 1},
                    {"name": "sku_search_additional_results", "count": 2},
                ],
            }
        },
    }
    assert len(seen_requests) == 1
    assert proxy.authorizations == [
        {
            "destination": "https://api.parallel.ai",
            "credential": _API_KEY_REF,
            "action": "parallel.search",
            "metadata": {"method": "POST", "path": "/v1/search"},
        }
    ]
    assert proxy.resolutions == [
        (
            _API_KEY_REF,
            {"destination": "https://api.parallel.ai", "provider": "parallel"},
        )
    ]
    assert _API_KEY not in json.dumps(result.model_dump())


def test_parallel_fetch_uses_extract_excerpts_and_ignores_full_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.parallel.ai/v1/extract"
        assert json.loads(request.content) == {
            "urls": ["https://example.com/reference"],
            "objective": "Retrieve API contract details.",
            "max_chars_total": 64,
            "advanced_settings": {
                "excerpt_settings": {"max_chars_per_result": 64},
                "full_content": False,
                "fetch_policy": {
                    "max_age_seconds": 600,
                    "timeout_seconds": 20.0,
                    "disable_cache_fallback": False,
                },
            },
        }
        return _json_response(
            {
                "extract_id": "extract_123",
                "session_id": "session_123",
                "results": [
                    {
                        "url": "https://example.com/reference",
                        "title": " Reference ",
                        "publish_date": "2026-08-17",
                        "excerpts": ["Hosted   page text.", "Second section."],
                        "full_content": "must never replace bounded excerpts",
                    }
                ],
                "errors": [],
                "usage": [{"name": "sku_extract_excerpts", "count": 1}],
            }
        )

    result = asyncio.run(
        WebFetchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                fetch_objective="Retrieve API contract details.",
                fetch_max_age_seconds=600,
                transport=httpx.MockTransport(handler),
            ),
            max_content_bytes=64,
        ).run(
            _context(_CredentialProxy()),
            {"url": "https://example.com/reference"},
        )
    )

    assert result.is_error is False
    assert result.structured == {
        "requested_url": "https://example.com/reference",
        "final_url": "https://example.com/reference",
        "title": "Reference",
        "representation": "text",
        "content": "Hosted page text.\nSecond section.",
        "redirects": [],
        "truncated": False,
        "truncation_reasons": [],
        "provider_metadata": {
            "parallel": {
                "request_id": "extract_123",
                "session_id": "session_123",
                "usage": [{"name": "sku_extract_excerpts", "count": 1}],
            }
        },
    }
    assert "must never replace" not in result.content


def test_parallel_extract_failure_is_stable_and_does_not_leak_raw_content() -> None:
    secret_detail = "private upstream response body"

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _json_response(
            _extract_document(
                results=[],
                errors=[
                    {
                        "url": "https://example.com/missing",
                        "error_type": "fetch_error",
                        "http_status_code": 404,
                        "content": secret_detail,
                    }
                ],
            )
        )

    result = asyncio.run(
        WebFetchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(
            _context(_CredentialProxy()),
            {"url": "https://example.com/missing"},
        )
    )

    assert result.is_error is True
    assert result.structured == {
        "error": "fetch_failed",
        "access": {
            "schema_version": 1,
            "outcome": "content_unavailable",
            "source": "hosted_provider",
            "signal": "status_code",
            "destination_fingerprint": web_destination_fingerprint("https://example.com/missing"),
            "status_code": 404,
            "retry_after_seconds": None,
            "retry_after_unrepresentable": False,
        },
        "status_code": 404,
    }
    assert secret_detail not in result.content
    assert secret_detail not in json.dumps(result.model_dump()["structured"])


def test_parallel_search_enforces_per_result_and_aggregate_snippet_budgets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _json_response(
            _search_document(
                results=[
                    {
                        "url": "https://example.com/one",
                        "title": "First",
                        "excerpts": ["abcdefghijk", "second snippet"],
                    },
                    {
                        "url": "https://example.com/two",
                        "title": "Second",
                        "excerpts": ["third snippet"],
                    },
                    {
                        "url": "https://example.com/three",
                        "title": "Third",
                        "excerpts": ["must not be returned"],
                    },
                ]
            )
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            ),
            default_results=2,
            max_results=2,
            max_snippet_bytes=8,
            max_total_snippet_bytes=12,
        ).run(_context(_CredentialProxy()), {"query": "bounded"})
    )

    assert result.structured is not None
    assert result.structured["results"] == [
        {
            "rank": 1,
            "url": "https://example.com/one",
            "title": "First",
            "snippets": ["abcdefgh", "seco"],
            "published_at": None,
        },
        {
            "rank": 2,
            "url": "https://example.com/two",
            "title": "Second",
            "snippets": [],
            "published_at": None,
        },
    ]
    assert result.structured["truncation_reasons"] == [
        "result_count",
        "snippet",
        "total_snippet_bytes",
    ]


def test_parallel_fetch_reports_content_and_excerpt_count_truncation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _json_response(
            _extract_document(
                results=[
                    {
                        "url": "https://example.com/large",
                        "title": "Large",
                        "excerpts": ["abcdefghijk", *["extra"] * 16],
                    }
                ]
            )
        )

    result = asyncio.run(
        WebFetchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            ),
            max_content_bytes=8,
        ).run(_context(_CredentialProxy()), {"url": "https://example.com/large"})
    )

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["content"] == "abcdefgh"
    assert result.structured["truncated"] is True
    assert result.structured["truncation_reasons"] == ["content", "excerpt_count"]


@pytest.mark.parametrize(
    "restrictions, label",
    [
        (WebSearchRestrictions(country="us"), "country"),
        (WebSearchRestrictions(locale="en-us"), "locale"),
        (WebSearchRestrictions(content_types=("application/pdf",)), "content-type"),
    ],
)
def test_parallel_rejects_unenforceable_restrictions_without_dispatch(
    restrictions: WebSearchRestrictions,
    label: str,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del request
        calls += 1
        raise AssertionError("unsupported restrictions must fail before dispatch")

    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            ),
            restrictions=restrictions,
        ).run(_context(_CredentialProxy()), {"query": "restricted"})
    )

    assert result.is_error is True
    assert result.structured == {"error": "unsupported_semantics"}
    assert label in result.content
    assert calls == 0


def test_parallel_fetch_fails_explicitly_for_unrepresentable_redirect_provenance() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _json_response(
            _extract_document(
                results=[
                    {
                        "url": "https://example.com/final",
                        "title": "Final",
                        "excerpts": ["content"],
                    }
                ]
            )
        )

    result = asyncio.run(
        WebFetchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"url": "https://example.com/start"})
    )

    assert result.is_error is True
    assert result.structured == {"error": "unsupported_semantics"}


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"search_id": "search_123", "session_id": "session_123"},
        {
            "search_id": "search_123",
            "session_id": "session_123",
            "results": [None],
        },
        {
            "search_id": "search_123",
            "session_id": "session_123",
            "results": [{"url": "https://example.com", "excerpts": None}],
        },
        {
            "search_id": "search_123",
            "session_id": "session_123",
            "results": [],
            "usage": [{"name": "sku", "count": -1}],
        },
    ],
)
def test_parallel_rejects_malformed_search_responses(document: Any) -> None:
    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(lambda request: _json_response(document)),
            )
        ).run(_context(_CredentialProxy()), {"query": "malformed"})
    )

    assert result.is_error is True
    assert result.structured == {"error": "malformed_provider_response"}


@pytest.mark.parametrize(
    "document",
    [
        {},
        {
            "extract_id": "extract_123",
            "session_id": "session_123",
            "results": [],
        },
        _extract_document(results=[]),
        _extract_document(results=[None]),
        _extract_document(
            results=[
                {
                    "url": "https://example.com",
                    "title": "Page",
                    "excerpts": None,
                }
            ]
        ),
        _extract_document(
            results=[
                {
                    "url": "https://example.com",
                    "title": "Page",
                    "excerpts": ["success"],
                }
            ],
            errors=[
                {
                    "url": "https://example.com",
                    "error_type": "fetch_error",
                    "http_status_code": 500,
                    "content": None,
                }
            ],
        ),
    ],
)
def test_parallel_rejects_malformed_fetch_responses(document: Any) -> None:
    result = asyncio.run(
        WebFetchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(lambda request: _json_response(document)),
            )
        ).run(_context(_CredentialProxy()), {"url": "https://example.com"})
    )

    assert result.is_error is True
    assert result.structured == {"error": "malformed_provider_response"}


def test_parallel_rate_limit_preserves_bounded_retry_metadata_without_retrying() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            headers={"retry-after": "15", "x-request-id": "request_429"},
            stream=_UnreadableStream(),
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"query": "rate limited"})
    )

    assert calls == 1
    assert result.is_error is True
    assert result.structured == {
        "error": "rate_limited",
        "status_code": 429,
        "provider_metadata": {
            "parallel": {
                "request_id": "request_429",
                "retry_after_seconds": 15.0,
            }
        },
    }


def test_parallel_rate_limit_does_not_shorten_unrepresentable_retry_timing() -> None:
    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        429,
                        headers={"retry-after": "86401"},
                        stream=_UnreadableStream(),
                    )
                ),
            )
        ).run(_context(_CredentialProxy()), {"query": "rate limited"})
    )

    assert result.structured == {
        "error": "rate_limited",
        "status_code": 429,
        "provider_metadata": {"parallel": {"retry_after_unrepresentable": True}},
    }


def test_parallel_rate_limit_does_not_ignore_overflowing_retry_timing() -> None:
    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        429,
                        headers={"retry-after": "9" * 400},
                        stream=_UnreadableStream(),
                    )
                ),
            )
        ).run(_context(_CredentialProxy()), {"query": "rate limited"})
    )

    assert result.structured == {
        "error": "rate_limited",
        "status_code": 429,
        "provider_metadata": {"parallel": {"retry_after_unrepresentable": True}},
    }


def test_parallel_rejects_duplicate_json_keys() -> None:
    body = b'{"search_id":"one","search_id":"two","session_id":"session","results":[]}'
    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)),
            )
        ).run(_context(_CredentialProxy()), {"query": "duplicates"})
    )

    assert result.structured == {"error": "malformed_provider_response"}


def test_parallel_redacts_resolved_secrets_from_external_output() -> None:
    response_secret = "parallel-response-secret"
    proxy = _CredentialProxy(
        resolved=ResolvedSecret(
            name="parallel_api_key",
            value=SecretStr(response_secret),
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _json_response(
            {
                "search_id": f"search-{response_secret}",
                "session_id": f"session-{response_secret}",
                "results": [
                    {
                        "url": "https://example.com",
                        "title": f"Title {response_secret}",
                        "excerpts": [f"Excerpt {response_secret}"],
                    }
                ],
                "warnings": [{"type": "warning", "message": f"Warning {response_secret}"}],
                "usage": [{"name": f"sku-{response_secret}", "count": 1}],
            }
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(proxy), {"query": "safe query"})
    )

    serialized = json.dumps(result.model_dump())
    assert response_secret not in serialized
    assert "[REDACTED_SECRET]" in serialized


def test_parallel_redacts_secrets_reconstructed_by_text_normalization() -> None:
    secret = "parallel secret"
    ctx = ToolContext(
        session_id="sess_parallel_normalized_secret",
        proxy=_CredentialProxy(),
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(
                    lambda request: _json_response(
                        _search_document(
                            results=[
                                {
                                    "url": "https://example.com",
                                    "title": "parallel   secret",
                                    "excerpts": [],
                                }
                            ]
                        )
                    )
                ),
            )
        ).run(ctx, {"query": "safe query"})
    )

    serialized = json.dumps(result.model_dump())
    assert secret not in serialized
    assert result.structured is not None
    assert result.structured["results"][0]["title"] == "[REDACTED_SECRET]"


def test_parallel_redacts_secrets_reconstructed_across_fetch_excerpts() -> None:
    secret = "first\nsecond"
    ctx = ToolContext(
        session_id="sess_parallel_joined_secret",
        proxy=_CredentialProxy(),
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(
        WebFetchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(
                    lambda request: _json_response(
                        _extract_document(
                            results=[
                                {
                                    "url": "https://example.com/reference",
                                    "title": "Reference",
                                    "excerpts": ["first", "second"],
                                }
                            ]
                        )
                    )
                ),
            )
        ).run(ctx, {"url": "https://example.com/reference"})
    )

    serialized = json.dumps(result.model_dump())
    assert secret not in serialized
    assert result.structured is not None
    assert result.structured["content"] == "[REDACTED_SECRET]"


def test_parallel_redacts_secrets_reconstructed_by_model_projection() -> None:
    secret = "Visible title\nURL: https://example.com/"
    ctx = ToolContext(
        session_id="sess_parallel_projected_secret",
        proxy=_CredentialProxy(),
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(
                    lambda request: _json_response(
                        _search_document(
                            results=[
                                {
                                    "url": "https://example.com",
                                    "title": "Visible title",
                                    "excerpts": [],
                                }
                            ]
                        )
                    )
                ),
            )
        ).run(ctx, {"query": "safe query"})
    )

    assert secret not in result.content
    assert "[REDACTED_SECRET]" in result.content
    assert result.content.count("<untrusted_web_content>") == 1
    assert result.content.count("</untrusted_web_content>") == 1


def test_parallel_redacts_fetch_projection_without_rewriting_trust_boundary() -> None:
    secret = "Reference\n\nfirst excerpt"
    ctx = ToolContext(
        session_id="sess_parallel_fetch_projected_secret",
        proxy=_CredentialProxy(),
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(
        WebFetchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(
                    lambda request: _json_response(
                        _extract_document(
                            results=[
                                {
                                    "url": "https://example.com/reference",
                                    "title": "Reference",
                                    "excerpts": ["first excerpt"],
                                }
                            ]
                        )
                    )
                ),
            )
        ).run(ctx, {"url": "https://example.com/reference"})
    )

    assert secret not in result.content
    assert "[REDACTED_SECRET]" in result.content
    assert result.content.count("<untrusted_web_content>") == 1
    assert result.content.count("</untrusted_web_content>") == 1


def test_parallel_rejects_oversized_composed_search_objective_before_authorization() -> None:
    proxy = _CredentialProxy()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        del request
        raise AssertionError("an oversized provider objective must not be dispatched")

    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                search_objective="x" * 4_990,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(proxy), {"query": "safe query"})
    )

    assert result.is_error is True
    assert result.structured == {"error": "unsupported_semantics"}
    assert proxy.authorizations == []
    assert calls == 0


def test_parallel_rejects_provider_oversized_search_query_before_authorization() -> None:
    proxy = _CredentialProxy()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        del request
        raise AssertionError("an oversized provider query must not be dispatched")

    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(proxy), {"query": "q" * 201})
    )

    assert result.is_error is True
    assert result.structured == {"error": "unsupported_semantics"}
    assert proxy.authorizations == []
    assert proxy.resolutions == []
    assert calls == 0


def test_parallel_omits_secret_collisions_from_typed_usage_metadata() -> None:
    numeric_secret = "731946"
    proxy = _CredentialProxy(
        resolved=ResolvedSecret(
            name="parallel_api_key",
            value=SecretStr(numeric_secret),
        )
    )

    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(
                    lambda request: _json_response(
                        {
                            **_search_document(results=[]),
                            "usage": [
                                {
                                    "name": "sku_search",
                                    "count": int(numeric_secret),
                                }
                            ],
                        }
                    )
                ),
            )
        ).run(_context(proxy), {"query": "safe query"})
    )

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["provider_metadata"]["parallel"]["usage"] == []
    assert result.structured["truncation_reasons"] == ["provider_metadata"]
    assert numeric_secret not in json.dumps(result.model_dump())


def test_parallel_omits_secret_collisions_from_extract_status_metadata() -> None:
    numeric_secret = "418"
    proxy = _CredentialProxy(
        resolved=ResolvedSecret(
            name="parallel_api_key",
            value=SecretStr(numeric_secret),
        )
    )

    result = asyncio.run(
        WebFetchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(
                    lambda request: _json_response(
                        _extract_document(
                            results=[],
                            errors=[
                                {
                                    "url": "https://example.com/missing",
                                    "error_type": "fetch_error",
                                    "http_status_code": int(numeric_secret),
                                    "content": None,
                                }
                            ],
                        )
                    )
                ),
            )
        ).run(_context(proxy), {"url": "https://example.com/missing"})
    )

    assert result.is_error is True
    assert result.structured is not None
    assert "status_code" not in result.structured
    assert numeric_secret not in json.dumps(result.model_dump())


def test_parallel_omits_secret_collisions_from_retry_and_http_status_metadata() -> None:
    retry_secret = "731946"
    retry_proxy = _CredentialProxy(
        resolved=ResolvedSecret(
            name="parallel_api_key",
            value=SecretStr(retry_secret),
        )
    )
    retry_result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        429,
                        headers={"retry-after": retry_secret},
                        stream=_UnreadableStream(),
                    )
                ),
            )
        ).run(_context(retry_proxy), {"query": "safe query"})
    )

    status_secret = "429"
    status_proxy = _CredentialProxy(
        resolved=ResolvedSecret(
            name="parallel_api_key",
            value=SecretStr(status_secret),
        )
    )
    status_result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(429, stream=_UnreadableStream())
                ),
            )
        ).run(_context(status_proxy), {"query": "safe query"})
    )

    assert retry_result.structured == {"error": "rate_limited", "status_code": 429}
    assert retry_secret not in json.dumps(retry_result.model_dump())
    assert status_result.structured == {"error": "rate_limited"}
    assert status_secret not in json.dumps(status_result.model_dump())


def test_parallel_omits_secret_reconstructed_by_retry_hint_parsing() -> None:
    numeric_secret = "86400"
    proxy = _CredentialProxy(
        resolved=ResolvedSecret(
            name="parallel_api_key",
            value=SecretStr(numeric_secret),
        )
    )

    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        429,
                        headers={"retry-after": "8.64e4"},
                        stream=_UnreadableStream(),
                    )
                ),
            )
        ).run(_context(proxy), {"query": "safe query"})
    )

    assert result.structured == {"error": "rate_limited", "status_code": 429}
    assert numeric_secret not in json.dumps(result.model_dump())


def test_parallel_uses_secret_registry_after_response_cleanup() -> None:
    late_secret = "late-parallel-secret"
    proxy = _CredentialProxy()
    current_redactor = [SecretRedactor()]
    ctx = ToolContext(
        session_id="sess_parallel_late_secret",
        proxy=proxy,
        invocation_secret_redactor=lambda: current_redactor[0],
    )
    body = json.dumps(
        _search_document(
            results=[
                {
                    "url": "https://example.com",
                    "title": f"Title {late_secret}",
                    "excerpts": [f"Excerpt {late_secret}"],
                }
            ]
        )
    ).encode()

    def register_secret() -> None:
        current_redactor[0] = SecretRedactor(late_secret)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_RegisterSecretOnCloseStream(body, register_secret),
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(ctx, {"query": "late secret"})
    )

    serialized = json.dumps(result.model_dump())
    assert late_secret not in serialized
    assert "[REDACTED_SECRET]" in serialized


def test_parallel_rejects_secret_collision_in_request_payload_before_dispatch() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        del request
        raise AssertionError("secret-bearing payload must not be dispatched")

    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"query": _API_KEY})
    )

    assert result.is_error is True
    assert result.structured == {"error": "secret_exposure_denied"}
    assert calls == 0


def test_parallel_credential_never_enters_runtime_evidence_or_model_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == _API_KEY
        return _json_response(
            {
                "search_id": f"search-{_API_KEY}",
                "session_id": f"session-{_API_KEY}",
                "results": [
                    {
                        "title": f"title {_API_KEY}",
                        "url": "https://example.com/",
                        "excerpts": [f"snippet {_API_KEY}"],
                    }
                ],
            }
        )

    store = InMemorySessionStore()
    vault = StaticVault({_API_KEY_REF.name: _API_KEY})
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="search",
                    name="web_search",
                    arguments={"query": "cayu"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="parallel"),
            vault=vault,
            proxy=AllowlistProxy(vault, allowed_destinations=["api.parallel.ai"]),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[
            WebSearchTool(
                adapter=ParallelAIWebAdapter(
                    api_key_ref=_API_KEY_REF,
                    transport=httpx.MockTransport(handler),
                )
            )
        ],
    )

    async def run_and_load():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_parallel_runtime_redaction",
                    messages=[Message.text("user", "search")],
                )
            )
        ]
        transcript = await store.load_transcript("sess_parallel_runtime_redaction")
        return events, transcript

    events, transcript = asyncio.run(run_and_load())

    rendered_transcript = repr([message.model_dump(mode="json") for message in transcript])
    assert events[-1].type is EventType.SESSION_COMPLETED
    assert _API_KEY not in repr([event.model_dump(mode="json") for event in events])
    assert _API_KEY not in rendered_transcript
    assert "REDACTED_SECRET" in rendered_transcript
    assert _API_KEY not in repr(
        [message.model_dump(mode="json") for message in provider.requests[1].messages]
    )


def test_parallel_requires_active_authorized_credential_authority() -> None:
    adapter = ParallelAIWebAdapter(
        api_key_ref=_API_KEY_REF,
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(AssertionError("request must not be dispatched"))
        ),
    )

    missing = asyncio.run(WebSearchTool(adapter=adapter).run(_context(), {"query": "missing"}))
    denied = asyncio.run(
        WebSearchTool(adapter=adapter).run(
            _context(_CredentialProxy(allowed=False)),
            {"query": "denied"},
        )
    )

    assert missing.structured == {"error": "credential_authority_unavailable"}
    assert denied.structured == {"error": "credential_denied"}


def test_parallel_oversized_response_fails_boundedly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_ByteStream(b"1234", b"5678"),
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                max_provider_response_bytes=6,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"query": "oversized"})
    )

    assert result.structured == {"error": "oversized_provider_response"}


def test_parallel_transport_cancellation_remains_authoritative() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise asyncio.CancelledError

    async def run() -> None:
        await WebSearchTool(
            adapter=ParallelAIWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"query": "cancel"})

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key_ref": "raw-secret"},
        {"api_key_ref": _API_KEY_REF, "origin": "http://api.parallel.ai"},
        {"api_key_ref": _API_KEY_REF, "origin": "https://api.parallel.ai/v1"},
        {"api_key_ref": _API_KEY_REF, "search_mode": "unknown"},
        {"api_key_ref": _API_KEY_REF, "search_location": "usa"},
        {"api_key_ref": _API_KEY_REF, "search_location": 1},
        {"api_key_ref": _API_KEY_REF, "search_objective": "   "},
        {"api_key_ref": _API_KEY_REF, "search_objective": "x" * 5_001},
        {"api_key_ref": _API_KEY_REF, "fetch_objective": "x" * 5_001},
        {"api_key_ref": _API_KEY_REF, "fetch_max_age_seconds": 599},
        {
            "api_key_ref": _API_KEY_REF,
            "search_disable_cache_fallback": True,
        },
        {"api_key_ref": _API_KEY_REF, "max_provider_response_bytes": 0},
        {"api_key_ref": _API_KEY_REF, "transport": object()},
    ],
)
def test_parallel_configuration_fails_closed(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ParallelAIWebAdapter(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"include_domains": ["example.com"]},
        {"include_domains": ("https://example.com",)},
        {"include_domains": ("example.com", "EXAMPLE.COM")},
        {
            "include_domains": ("example.com",),
            "exclude_domains": ("EXAMPLE.COM",),
        },
        {"country": "usa"},
        {"country": "12"},
        {"country": "éé"},
        {"locale": "en us"},
        {"content_types": ("text/html; charset=utf-8",)},
        {"content_types": ("text/ht@ml",)},
    ],
)
def test_web_search_restrictions_validate_and_normalize(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        WebSearchRestrictions(**kwargs)
