from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
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
    ExaWebAdapter,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    ProxyAuthorizationResult,
    ResolvedSecret,
    RunRequest,
    ScriptedModelProvider,
    SecretRedactor,
    SecretRef,
    StaticVault,
    ToolContext,
    WebFetchTool,
    WebSearchTool,
)

_API_KEY = "exa-test-secret-value"
_API_KEY_REF = SecretRef(name="exa_api_key")


class _CredentialProxy:
    def __init__(
        self,
        *,
        allowed: bool = True,
        resolved: ResolvedSecret | None = None,
    ) -> None:
        self.allowed = allowed
        self.resolved = resolved or ResolvedSecret(
            name="exa_api_key",
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


def _context(
    proxy: _CredentialProxy | None = None,
) -> ToolContext:
    return ToolContext(
        session_id="sess_exa",
        proxy=proxy,
    )


def _json_response(payload: Any, *, status_code: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(
        status_code,
        headers={"content-type": "application/json", **headers},
        json=payload,
    )


def test_exa_search_uses_trusted_credential_and_returns_portable_bounded_results() -> None:
    proxy = _CredentialProxy()
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        assert request.url == "https://api.exa.ai/search"
        assert request.headers["x-api-key"] == _API_KEY
        assert "authorization" not in request.headers
        assert json.loads(request.content) == {
            "query": "cayu production runtime",
            "type": "fast",
            "numResults": 2,
            "moderation": True,
            "contents": {
                "highlights": {
                    "query": "cayu production runtime",
                    "maxCharacters": 40,
                },
                "maxAgeHours": 24,
            },
        }
        return _json_response(
            {
                "requestId": "req-search-1",
                "results": [
                    {
                        "title": "  Cayu runtime  ",
                        "url": "HTTPS://EXAMPLE.COM/runtime#section",
                        "publishedDate": "2026-08-17T12:30:00+02:00",
                        "highlights": [" Durable   sessions ", "Bounded recovery"],
                        "highlightScores": [0.91, 0.75],
                    },
                    {
                        "title": "Reference",
                        "url": "https://docs.example.com/cayu",
                        "highlights": ["Provider-neutral web tools"],
                    },
                ],
                "costDollars": {"total": 0.007, "search": {"neural": 0.005}},
                "warnings": [" provider warning "],
            }
        )

    adapter = ExaWebAdapter(
        api_key_ref=_API_KEY_REF,
        search_type="fast",
        search_max_age_hours=24,
        moderation=True,
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(
        WebSearchTool(
            adapter=adapter,
            default_results=2,
            max_results=2,
            max_snippet_bytes=40,
            max_total_snippet_bytes=100,
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
                "published_at": "2026-08-17T10:30:00Z",
                "provider_metadata": {"exa": {"highlight_scores": [0.91, 0.75]}},
            },
            {
                "rank": 2,
                "url": "https://docs.example.com/cayu",
                "title": "Reference",
                "snippets": ["Provider-neutral web tools"],
                "published_at": None,
            },
        ],
        "truncated": False,
        "truncation_reasons": [],
        "provider_metadata": {
            "exa": {
                "request_id": "req-search-1",
                "warnings": ["provider warning"],
                "usage": {
                    "estimated_cost_usd": 0.007,
                    "estimated_search_cost_usd": 0.005,
                },
            }
        },
    }
    assert "<untrusted_web_content>" in result.content
    assert "https://example.com/runtime" in result.content
    assert len(seen_requests) == 1
    assert proxy.authorizations == [
        {
            "destination": "https://api.exa.ai",
            "credential": _API_KEY_REF,
            "action": "exa.search",
            "metadata": {"method": "POST", "path": "/search"},
        }
    ]
    assert proxy.resolutions == [
        (
            _API_KEY_REF,
            {"destination": "https://api.exa.ai", "provider": "exa"},
        )
    ]
    assert _API_KEY not in json.dumps(result.model_dump())


def test_exa_search_enforces_per_result_and_aggregate_snippet_budgets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _json_response(
            {
                "results": [
                    {
                        "title": "First",
                        "url": "https://example.com/one",
                        "highlights": ["abcdefghijk", "second snippet"],
                    },
                    {
                        "title": "Second",
                        "url": "https://example.com/two",
                        "highlights": ["third snippet"],
                    },
                    {
                        "title": "Third",
                        "url": "https://example.com/three",
                        "highlights": ["must not be returned"],
                    },
                ]
            }
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            ),
            default_results=2,
            max_results=2,
            max_snippet_bytes=8,
            max_total_snippet_bytes=12,
        ).run(
            _context(_CredentialProxy()),
            {"query": "bounded", "num_results": 2},
        )
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
    assert result.structured["truncated"] is True
    assert result.structured["truncation_reasons"] == [
        "result_count",
        "snippet",
        "total_snippet_bytes",
    ]


def test_exa_search_accepts_nullable_title_and_date_only_publication() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _json_response(
            {
                "results": [
                    {
                        "title": None,
                        "url": "https://example.com/untitled",
                        "publishedDate": "2024-01-15",
                        "highlights": ["Documented Exa response shape"],
                    }
                ]
            }
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"query": "documented fields"})
    )

    assert result.is_error is False
    assert result.structured is not None
    assert result.structured["results"] == [
        {
            "rank": 1,
            "url": "https://example.com/untitled",
            "title": "https://example.com/untitled",
            "snippets": ["Documented Exa response shape"],
            "published_at": "2024-01-15",
        }
    ]


def test_exa_search_bounds_snippet_count_per_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _json_response(
            {
                "results": [
                    {
                        "title": "Many snippets",
                        "url": "https://example.com/",
                        "highlights": [f"snippet {index}" for index in range(20)],
                        "highlightScores": [index / 20 for index in range(20)],
                    }
                ]
            }
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            ),
            max_total_snippet_bytes=1_000,
        ).run(_context(_CredentialProxy()), {"query": "many snippets"})
    )

    assert result.structured is not None
    search_result = result.structured["results"][0]
    assert len(search_result["snippets"]) == 8
    assert len(search_result["provider_metadata"]["exa"]["highlight_scores"]) == 8
    assert result.structured["truncation_reasons"] == ["snippet_count"]


def test_exa_fetch_preserves_web_fetch_shape_and_provider_metadata() -> None:
    proxy = _CredentialProxy()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.exa.ai/contents"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        assert "x-api-key" not in request.headers
        assert json.loads(request.content) == {
            "urls": ["https://example.com/reference"],
            "text": {
                "maxCharacters": 64,
                "includeHtmlTags": False,
                "verbosity": "compact",
            },
            "maxAgeHours": 0,
        }
        return _json_response(
            {
                "requestId": "req-fetch-1",
                "results": [
                    {
                        "title": "Reference",
                        "url": "https://example.com/reference",
                        "text": "Hosted   page text.",
                    }
                ],
                "statuses": [
                    {
                        "id": "provider-document-id",
                        "status": "success",
                        "source": "crawled",
                    }
                ],
                "costDollars": {"total": 0.003},
            }
        )

    result = asyncio.run(
        WebFetchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                fetch_max_age_hours=0,
                auth_header="authorization",
                transport=httpx.MockTransport(handler),
            ),
            max_content_bytes=64,
        ).run(_context(proxy), {"url": "https://example.com/reference"})
    )

    assert result.is_error is False
    assert result.structured == {
        "requested_url": "https://example.com/reference",
        "final_url": "https://example.com/reference",
        "title": "Reference",
        "representation": "text",
        "content": "Hosted page text.",
        "redirects": [],
        "truncated": False,
        "truncation_reasons": [],
        "provider_metadata": {
            "exa": {
                "request_id": "req-fetch-1",
                "usage": {"estimated_cost_usd": 0.003},
            }
        },
    }
    assert proxy.authorizations[0]["action"] == "exa.contents"


def test_exa_fetch_maps_per_url_failure_without_exposing_provider_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _json_response(
            {
                "results": [],
                "statuses": [
                    {
                        "id": "https://example.com/missing",
                        "status": "error",
                        "error": {
                            "tag": "private-provider-detail",
                            "httpStatusCode": 404,
                        },
                    }
                ],
            }
        )

    result = asyncio.run(
        WebFetchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(
            _context(_CredentialProxy()),
            {"url": "https://example.com/missing"},
        )
    )

    assert result.structured == {"error": "fetch_failed", "status_code": 404}
    assert result.is_error is True
    assert "private-provider-detail" not in result.content


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": []},
        {"results": [None]},
        {
            "results": [
                {
                    "title": "Missing content",
                    "url": "https://example.com/",
                }
            ]
        },
        {
            "results": [
                {
                    "title": "Page",
                    "url": "https://example.com/",
                    "text": "content",
                }
            ],
            "statuses": [{"id": "doc", "status": "unknown"}],
        },
    ],
)
def test_exa_rejects_malformed_fetch_responses(payload: Any) -> None:
    result = asyncio.run(
        WebFetchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(lambda request: _json_response(payload)),
            )
        ).run(_context(_CredentialProxy()), {"url": "https://example.com/"})
    )

    assert result.structured == {"error": "malformed_provider_response"}


def test_exa_fetch_fails_explicitly_for_unrepresentable_redirect_provenance() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _json_response(
            {
                "results": [
                    {
                        "title": "Moved",
                        "url": "https://example.com/final",
                        "text": "content",
                    }
                ],
                "statuses": [{"id": "doc", "status": "success"}],
            }
        )

    result = asyncio.run(
        WebFetchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(
            _context(_CredentialProxy()),
            {"url": "https://example.com/original"},
        )
    )

    assert result.structured == {"error": "unsupported_semantics"}


def test_exa_fetch_exposes_narrower_provider_content_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["text"]["maxCharacters"] == 10_000
        return _json_response(
            {
                "results": [
                    {
                        "title": "Long page",
                        "url": "https://example.com/",
                        "text": "x" * 10_000,
                    }
                ]
            }
        )

    result = asyncio.run(
        WebFetchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"url": "https://example.com/"})
    )

    assert result.structured is not None
    assert result.structured["truncated"] is True
    assert result.structured["truncation_reasons"] == ["provider_content_limit"]


def test_exa_fetch_does_not_claim_provider_truncation_for_short_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["text"]["maxCharacters"] == 10_000
        return _json_response(
            {
                "results": [
                    {
                        "title": "Short page",
                        "url": "https://example.com/",
                        "text": "Complete short page.",
                    }
                ]
            }
        )

    result = asyncio.run(
        WebFetchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"url": "https://example.com/"})
    )

    assert result.structured is not None
    assert result.structured["truncated"] is False
    assert result.structured["truncation_reasons"] == []


def test_exa_rate_limit_preserves_bounded_retry_metadata_without_retrying() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        del request
        return httpx.Response(
            429,
            headers={
                "content-encoding": "gzip",
                "content-length": str(16 * 1024 * 1024),
                "retry-after": "12.5",
                "x-request-id": "req-rate-limit",
            },
            stream=_UnreadableStream(),
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"query": "rate limit"})
    )

    assert result.structured == {
        "error": "rate_limited",
        "status_code": 429,
        "provider_metadata": {
            "exa": {
                "request_id": "req-rate-limit",
                "retry_after_seconds": 12.5,
            }
        },
    }
    assert calls == 1


def test_exa_transport_failure_is_sanitized_and_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError(
            f"private transport detail {_API_KEY}",
            request=request,
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"query": "transport"})
    )

    assert result.structured == {"error": "provider_unavailable"}
    assert _API_KEY not in result.content
    assert calls == 1


@pytest.mark.parametrize(
    ("payload", "status_code", "expected_error"),
    [
        ({"error": "no"}, 401, "provider_authentication_failed"),
        ({"error": "no"}, 402, "provider_quota_exhausted"),
        ({"error": "no"}, 422, "provider_request_rejected"),
        ({"error": "no"}, 503, "provider_unavailable"),
    ],
)
def test_exa_maps_provider_http_failures(
    payload: dict[str, str],
    status_code: int,
    expected_error: str,
) -> None:
    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(
                    lambda request: _json_response(payload, status_code=status_code)
                ),
            )
        ).run(_context(_CredentialProxy()), {"query": "failure"})
    )

    assert result.structured is not None
    assert result.structured["error"] == expected_error
    assert result.structured["status_code"] == status_code


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"results": "wrong"},
        {"results": [None]},
        {"results": [{"url": "https://example.com/", "highlights": []}]},
        {"results": [{"title": "Missing URL", "highlights": []}]},
        {
            "results": [
                {
                    "title": "Bad URL",
                    "url": "http://example.com",
                    "highlights": [],
                }
            ]
        },
        {
            "results": [
                {
                    "title": "Bad score",
                    "url": "https://example.com/",
                    "highlights": ["snippet"],
                    "highlightScores": [],
                }
            ]
        },
        {
            "results": [
                {
                    "title": "Bad date",
                    "url": "https://example.com/",
                    "publishedDate": "yesterday",
                    "highlights": [],
                }
            ]
        },
        {"results": [], "costDollars": {"total": 10**400}},
        {
            "results": [
                {
                    "title": "Oversized score",
                    "url": "https://example.com/",
                    "highlights": ["snippet"],
                    "highlightScores": [10**400],
                }
            ]
        },
    ],
)
def test_exa_rejects_malformed_search_responses(payload: Any) -> None:
    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(lambda request: _json_response(payload)),
            )
        ).run(_context(_CredentialProxy()), {"query": "malformed"})
    )

    assert result.structured == {"error": "malformed_provider_response"}


def test_exa_maps_nonportable_publication_time_to_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=(
                b'{"results":[{"title":"Nonportable date",'
                b'"url":"https://example.com/","publishedDate":"\\ud800",'
                b'"highlights":[]}]}'
            ),
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"query": "malformed date"})
    )

    assert result.structured == {"error": "malformed_provider_response"}


def test_exa_rejects_duplicate_json_keys() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b'{"results":[],"results":[]}',
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"query": "duplicate"})
    )

    assert result.structured == {"error": "malformed_provider_response"}


@pytest.mark.parametrize(
    "body",
    [
        b'{"results":' + (b"[" * 10_000) + b"0" + (b"]" * 10_000) + b"}",
        b'{"results":[],"unknown":' + (b"9" * 10_000) + b"}",
    ],
)
def test_exa_maps_json_decoder_resource_failures_to_malformed_response(body: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=body,
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"query": "decoder limits"})
    )

    assert result.structured == {"error": "malformed_provider_response"}


def test_exa_redacts_resolved_secrets_from_external_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _json_response(
            {
                "requestId": f"request-{_API_KEY}",
                "warnings": [f"warning-{_API_KEY}"],
                "results": [
                    {
                        "title": f"title {_API_KEY}",
                        "url": "https://example.com/",
                        "highlights": [f"snippet {_API_KEY}"],
                    }
                ],
            }
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(
            _context(_CredentialProxy()),
            {"query": "redaction"},
        )
    )

    rendered = json.dumps(result.model_dump())
    assert _API_KEY not in rendered
    assert "REDACTED_SECRET" in rendered


def test_exa_uses_secret_registry_after_successful_response_cleanup() -> None:
    late_secret = "late-response-secret"
    current_redactor = [SecretRedactor()]

    def register_late_secret() -> None:
        current_redactor[0] = SecretRedactor(late_secret)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        body = json.dumps(
            {
                "results": [
                    {
                        "title": f"title {late_secret}",
                        "url": "https://example.com/",
                        "highlights": [f"snippet {late_secret}"],
                    }
                ]
            }
        ).encode()
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_RegisterSecretOnCloseStream(
                body,
                register_late_secret,
            ),
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(
            ToolContext(
                session_id="sess_exa_late_secret",
                proxy=_CredentialProxy(),
                invocation_secret_redactor=lambda: current_redactor[0],
            ),
            {"query": "late secret"},
        )
    )

    rendered = json.dumps(result.model_dump())
    assert late_secret not in rendered
    assert "REDACTED_SECRET" in rendered


def test_exa_rejects_source_urls_that_contain_a_resolved_secret() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return _json_response(
            {
                "results": [
                    {
                        "title": "Secret-bearing URL",
                        "url": f"https://example.com/?token={_API_KEY}",
                        "highlights": ["snippet"],
                    }
                ]
            }
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(
            _context(_CredentialProxy()),
            {"query": "redaction"},
        )
    )

    assert result.structured == {"error": "malformed_provider_response"}
    assert _API_KEY not in json.dumps(result.model_dump())


@pytest.mark.parametrize("secret", [_API_KEY, 'exa"key\\segment'])
def test_exa_rejects_secret_collision_in_request_payload_before_dispatch(secret: str) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        del request
        return _json_response({"results": []})

    proxy = _CredentialProxy(
        resolved=ResolvedSecret(
            name="exa_api_key",
            value=SecretStr(secret),
        )
    )
    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(proxy), {"query": secret})
    )

    assert result.structured == {"error": "secret_exposure_denied"}
    assert secret not in json.dumps(result.model_dump())
    assert calls == 0


def test_exa_credential_never_enters_runtime_events_transcript_or_model_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == _API_KEY
        return _json_response(
            {
                "requestId": f"request-{_API_KEY}",
                "results": [
                    {
                        "title": f"title {_API_KEY}",
                        "url": "https://example.com/",
                        "highlights": [f"snippet {_API_KEY}"],
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
            EnvironmentSpec(name="exa"),
            vault=vault,
            proxy=AllowlistProxy(vault, allowed_destinations=["api.exa.ai"]),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[
            WebSearchTool(
                adapter=ExaWebAdapter(
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
                    session_id="sess_exa_runtime_redaction",
                    messages=[Message.text("user", "search")],
                )
            )
        ]
        transcript = await store.load_transcript("sess_exa_runtime_redaction")
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


def test_exa_requires_active_authorized_credential_authority() -> None:
    adapter = ExaWebAdapter(
        api_key_ref=_API_KEY_REF,
        transport=httpx.MockTransport(lambda request: _json_response({"results": []})),
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


def test_exa_oversized_response_fails_boundedly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            stream=_ByteStream(b'{"results":[]}', b" " * 100),
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                max_provider_response_bytes=16,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"query": "oversized"})
    )

    assert result.structured == {"error": "oversized_provider_response"}


def test_exa_rejects_encoded_provider_response_before_decoding() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_ByteStream(b"not-consumed-as-json"),
        )

    result = asyncio.run(
        WebSearchTool(
            adapter=ExaWebAdapter(
                api_key_ref=_API_KEY_REF,
                transport=httpx.MockTransport(handler),
            )
        ).run(_context(_CredentialProxy()), {"query": "encoded"})
    )

    assert result.structured == {"error": "unsupported_provider_response"}


def test_exa_transport_cancellation_remains_authoritative() -> None:
    started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def scenario() -> None:
        task = asyncio.create_task(
            WebSearchTool(
                adapter=ExaWebAdapter(
                    api_key_ref=_API_KEY_REF,
                    transport=httpx.MockTransport(handler),
                )
            ).run(_context(_CredentialProxy()), {"query": "cancel"})
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key_ref": "secret"},
        {"api_key_ref": _API_KEY_REF, "origin": "http://api.exa.ai"},
        {"api_key_ref": _API_KEY_REF, "origin": "https://api.exa.ai/path"},
        {"api_key_ref": _API_KEY_REF, "origin": "https://api.exa.ai?tenant=one"},
        {"api_key_ref": _API_KEY_REF, "origin": "https://api.exa.ai#credentials"},
        {"api_key_ref": _API_KEY_REF, "auth_header": "cookie"},
        {"api_key_ref": _API_KEY_REF, "search_type": "magic"},
        {"api_key_ref": _API_KEY_REF, "search_max_age_hours": 721},
        {"api_key_ref": _API_KEY_REF, "fetch_max_age_hours": -2},
        {"api_key_ref": _API_KEY_REF, "moderation": 1},
        {"api_key_ref": _API_KEY_REF, "max_provider_response_bytes": 0},
    ],
)
def test_exa_configuration_fails_closed(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ExaWebAdapter(**kwargs)
