from __future__ import annotations

import pytest

from cayu.egress import EgressRequest, HttpEgressPolicy


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
