from __future__ import annotations

import pytest

from cayu.egress import BrowserEgressPolicy, EgressRequest, HttpEgressPolicy


def _request(method: str, path: str, *, host: str = "api.stripe.com") -> EgressRequest:
    return EgressRequest(method=method, host=host, path=path)


def test_http_policy_allows_configured_host_method_and_path() -> None:
    policy = HttpEgressPolicy(
        name="provider-example",
        allowed_hosts=["api.example.com"],
        allowed_endpoints=[("POST", "/v1/widgets")],
    )

    result = policy.authorize(_request("POST", "/v1/widgets", host="api.example.com"))

    assert result.allowed is True
    assert result.metadata["policy"] == "provider-example"


def test_http_policy_denies_foreign_destination() -> None:
    policy = HttpEgressPolicy(
        name="provider-example",
        allowed_hosts=["api.example.com"],
        allowed_endpoints=[("POST", "/v1/widgets")],
    )

    result = policy.authorize(_request("POST", "/v1/widgets", host="evil.example.com"))

    assert result.allowed is False
    assert "not allowed" in (result.reason or "")


def test_http_policy_denies_unlisted_endpoint() -> None:
    policy = HttpEgressPolicy(
        name="provider-example",
        allowed_hosts=["api.example.com"],
        allowed_endpoints=[("POST", "/v1/widgets")],
    )

    result = policy.authorize(_request("POST", "/v1/other", host="api.example.com"))

    assert result.allowed is False
    assert "allowlist" in (result.reason or "")


def test_http_policy_denies_wrong_method_on_allowed_path() -> None:
    policy = HttpEgressPolicy(
        name="provider-example",
        allowed_hosts=["api.example.com"],
        allowed_endpoints=[("POST", "/v1/widgets")],
    )

    result = policy.authorize(_request("DELETE", "/v1/widgets", host="api.example.com"))

    assert result.allowed is False


def test_http_policy_denied_prefix_wins_over_allowlist() -> None:
    policy = HttpEgressPolicy(
        name="provider-example",
        allowed_hosts=["api.example.com"],
        allowed_endpoints=[("POST", "/v1/widgets"), ("POST", "/v1/admin")],
        denied_prefixes=["/v1/admin"],
    )

    result = policy.authorize(_request("POST", "/v1/admin", host="api.example.com"))

    assert result.allowed is False
    assert "explicitly denied" in (result.reason or "")


def test_http_policy_denied_prefix_matches_child_routes() -> None:
    policy = HttpEgressPolicy(
        name="provider-example",
        allowed_hosts=["api.example.com"],
        allowed_endpoints=[("POST", "/v1/widgets")],
        denied_prefixes=["/v1/admin"],
    )

    result = policy.authorize(_request("POST", "/v1/admin/key", host="api.example.com"))

    assert result.allowed is False
    assert "explicitly denied" in (result.reason or "")


def test_http_policy_root_denied_prefix_denies_all_paths_even_if_allowed() -> None:
    policy = HttpEgressPolicy(
        name="provider-example",
        allowed_hosts=["api.example.com"],
        allowed_endpoints=[("POST", "/v1/widgets")],
        denied_prefixes=["/"],
    )

    result = policy.authorize(_request("POST", "/v1/widgets", host="api.example.com"))

    assert result.allowed is False
    assert "explicitly denied" in (result.reason or "")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {"name": "bad", "allowed_hosts": [], "allowed_endpoints": [("POST", "/v1/widgets")]},
            "allowed host",
        ),
        (
            {"name": "bad", "allowed_hosts": ["api.example.com"], "allowed_endpoints": []},
            "allowed endpoint",
        ),
        (
            {
                "name": "bad",
                "allowed_hosts": ["api.example.com"],
                "allowed_endpoints": [("POST", "relative")],
            },
            "start with",
        ),
        (
            {
                "name": "bad",
                "allowed_hosts": ["api.example.com"],
                "allowed_endpoints": [("POST", "/v1/widgets")],
                "denied_prefixes": ["relative"],
            },
            "start with",
        ),
    ],
)
def test_http_policy_rejects_invalid_configuration(kwargs, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        HttpEgressPolicy(**kwargs)


def test_http_policy_does_not_interpret_provider_body_semantics() -> None:
    policy = HttpEgressPolicy(
        name="provider-example",
        allowed_hosts=["api.provider.test"],
        allowed_endpoints=[("POST", "/v1/orders")],
    )
    request = EgressRequest(
        method="POST",
        host="api.provider.test",
        path="/v1/orders",
        body=b"plan_id=provider_owned_plan_123",
        content_type="application/x-www-form-urlencoded",
    )

    result = policy.authorize(request)

    assert result.allowed is True


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_browser_policy_allows_read_only_requests_beneath_configured_prefix(
    method: str,
) -> None:
    policy = BrowserEgressPolicy(
        name="public-docs",
        allowed_hosts=["Docs.Example.COM."],
        allowed_path_prefixes=["/guides/"],
    )

    result = policy.authorize(_request(method, "/guides/install", host="docs.example.com"))

    assert result.allowed is True
    assert result.metadata == {"policy": "public-docs"}


@pytest.mark.parametrize(
    ("method", "host", "path", "reason"),
    [
        ("GET", "other.example.com", "/guides/install", "not allowed"),
        ("POST", "docs.example.com", "/guides/install", "Method"),
        ("GET", "docs.example.com", "/other/install", "outside"),
        ("GET", "docs.example.com", "/guides/private/key", "explicitly denied"),
    ],
)
def test_browser_policy_denies_unadmitted_web_request(
    method: str,
    host: str,
    path: str,
    reason: str,
) -> None:
    policy = BrowserEgressPolicy(
        name="public-docs",
        allowed_hosts=["docs.example.com"],
        allowed_path_prefixes=["/guides"],
        denied_prefixes=["/guides/private"],
    )

    result = policy.authorize(_request(method, path, host=host))

    assert result.allowed is False
    assert reason in (result.reason or "")


def test_browser_policy_root_prefix_allows_all_paths_on_configured_hosts_only() -> None:
    policy = BrowserEgressPolicy(
        name="public-site",
        allowed_hosts=["www.example.com"],
    )

    assert policy.authorize(_request("GET", "/anything", host="www.example.com")).allowed
    assert not policy.authorize(_request("GET", "/anything", host="cdn.example.com")).allowed


def test_browser_policy_denies_get_request_bodies() -> None:
    policy = BrowserEgressPolicy(
        name="public-site",
        allowed_hosts=["www.example.com"],
    )
    request = EgressRequest(
        method="GET",
        host="www.example.com",
        path="/anything",
        body=b"unexpected mutation input",
    )

    result = policy.authorize(request)

    assert result.allowed is False
    assert "bodies" in (result.reason or "")


@pytest.mark.parametrize(
    "path",
    [
        "/guides/%70rivate/key",
        "/guides/%2e%2e/private",
        "/guides/%252e%252e/private",
        "/guides//private/key",
        "/guides/%2Fprivate/key",
        "/guides\\private",
        "/guides/private;ignored",
        "/guides/private%3Bignored",
        "/guides/..;/admin",
        "/guides/%2e%2e%3b/admin",
        "/guides/%",
    ],
)
def test_browser_policy_denies_ambiguous_path_spellings(path: str) -> None:
    policy = BrowserEgressPolicy(
        name="public-docs",
        allowed_hosts=["docs.example.com"],
        allowed_path_prefixes=["/guides"],
        denied_prefixes=["/guides/private"],
    )

    result = policy.authorize(_request("GET", path, host="docs.example.com"))

    assert result.allowed is False


def test_browser_policy_matches_percent_decoded_path() -> None:
    policy = BrowserEgressPolicy(
        name="public-docs",
        allowed_hosts=["docs.example.com"],
        allowed_path_prefixes=["/product guides"],
    )

    result = policy.authorize(_request("GET", "/product%20guides/install", host="docs.example.com"))

    assert result.allowed is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"name": "bad", "allowed_hosts": []},
            "allowed host",
        ),
        (
            {
                "name": "bad",
                "allowed_hosts": ["docs.example.com"],
                "allowed_path_prefixes": [],
            },
            "allowed path prefix",
        ),
        (
            {
                "name": "bad",
                "allowed_hosts": ["docs.example.com"],
                "allowed_path_prefixes": ["relative"],
            },
            "start with",
        ),
        (
            {
                "name": "bad",
                "allowed_hosts": ["docs.example.com"],
                "denied_prefixes": ["relative"],
            },
            "start with",
        ),
        (
            {
                "name": "bad",
                "allowed_hosts": ["docs.example.com"],
                "allowed_path_prefixes": ["/guides/%252e%252e"],
            },
            "ambiguous",
        ),
        (
            {
                "name": "bad",
                "allowed_hosts": ["docs.example.com"],
                "allowed_path_prefixes": ["//"],
            },
            "ambiguous",
        ),
        (
            {
                "name": "bad",
                "allowed_hosts": ["docs.example.com"],
                "denied_prefixes": ["/guides//private"],
            },
            "ambiguous",
        ),
        (
            {
                "name": "bad",
                "allowed_hosts": ["docs.example.com"],
                "allowed_path_prefixes": ["/product%20guides"],
            },
            "canonical",
        ),
        (
            {
                "name": "bad",
                "allowed_hosts": ["docs.example.com"],
                "allowed_path_prefixes": ["/guides;audience=public"],
            },
            "ambiguous",
        ),
        (
            {
                "name": "bad",
                "allowed_hosts": ["docs.example.com"],
                "denied_prefixes": ["/guides/private%3Bignored"],
            },
            "ambiguous",
        ),
    ],
)
def test_browser_policy_rejects_invalid_configuration(kwargs, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        BrowserEgressPolicy(**kwargs)
