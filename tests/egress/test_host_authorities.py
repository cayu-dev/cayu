from __future__ import annotations

import pytest

from cayu import VirtualCredentialSpec
from cayu.egress import (
    CapturedRequest,
    EgressRequest,
    HttpEgressPolicy,
    HttpxUpstream,
    VirtualCredentialRegistry,
)
from cayu.vaults import SecretRef


@pytest.mark.parametrize(
    "host",
    [
        "api.example.com@evil.example",
        "https://api.example.com",
        "api.example.com/v1",
        "api.example.com?mode=unsafe",
        "api.example.com#fragment",
        "api.example.com:443",
        "api. example.com",
    ],
)
def test_captured_request_rejects_non_bare_host_authorities(host: str) -> None:
    with pytest.raises(ValueError, match="bare hostname"):
        CapturedRequest(method="POST", host=host, path="/v1")


@pytest.mark.parametrize(
    "host",
    [
        "api..example.com",
        "api.example.com..",
        "-api.example.com",
        "api_example.com",
    ],
)
def test_captured_request_rejects_invalid_dns_labels(host: str) -> None:
    with pytest.raises(ValueError, match="valid hostname"):
        CapturedRequest(method="POST", host=host, path="/v1")


def test_captured_request_hides_malformed_host_from_bounded_diagnostics() -> None:
    hostile_host = ("workload-secret-value" * 1000) + "@evil.example"

    with pytest.raises(ValueError) as excinfo:
        CapturedRequest(method="POST", host=hostile_host, path="/v1")

    diagnostic = str(excinfo.value)
    assert "workload-secret-value" not in diagnostic
    assert len(diagnostic) < 500


@pytest.mark.parametrize(
    "constructor",
    [
        pytest.param(
            lambda host: EgressRequest(method="POST", host=host, path="/v1"),
            id="policy-request",
        ),
        pytest.param(
            lambda host: HttpEgressPolicy(
                name="provider",
                allowed_hosts=[host],
                allowed_endpoints=[("POST", "/v1")],
            ),
            id="policy-allowlist",
        ),
        pytest.param(
            lambda host: VirtualCredentialSpec(
                env_name="API_KEY",
                secret=SecretRef(name="api_key"),
                destination=host,
                policy_name="provider",
            ),
            id="credential-spec",
        ),
        pytest.param(
            lambda host: VirtualCredentialRegistry().mint(
                session_id="session",
                env_name="API_KEY",
                secret=SecretRef(name="api_key"),
                destination=host,
                credential_kind="opaque_bearer",
                policy_name="provider",
            ),
            id="credential-grant",
        ),
        pytest.param(
            lambda host: HttpxUpstream(routes={host: "http://receiver.service.local:8080"}),
            id="upstream-route-alias",
        ),
    ],
)
def test_virtual_egress_configuration_rejects_ambiguous_host_authority(constructor) -> None:
    with pytest.raises(ValueError, match="bare hostname"):
        constructor("api.example.com@evil.example")


def test_virtual_egress_host_boundaries_share_canonical_case_and_trailing_dot() -> None:
    raw_host = "API.Example.COM."
    request = CapturedRequest(method="POST", host=raw_host, path="/v1")
    policy_request = EgressRequest(method="POST", host=raw_host, path="/v1")
    policy = HttpEgressPolicy(
        name="provider",
        allowed_hosts=[raw_host],
        allowed_endpoints=[("POST", "/v1")],
    )
    spec = VirtualCredentialSpec(
        env_name="API_KEY",
        secret=SecretRef(name="api_key"),
        destination=raw_host,
        policy_name="provider",
    )
    grant = VirtualCredentialRegistry().mint(
        session_id="session",
        env_name="API_KEY",
        secret=SecretRef(name="api_key"),
        destination=raw_host,
        credential_kind="opaque_bearer",
        policy_name="provider",
    )

    assert request.host == "api.example.com"
    assert policy_request.host == "api.example.com"
    assert policy.allowed_hosts == frozenset({"api.example.com"})
    assert spec.destination == "api.example.com"
    assert grant.destination == "api.example.com"
    assert policy.authorize(policy_request).allowed is True
