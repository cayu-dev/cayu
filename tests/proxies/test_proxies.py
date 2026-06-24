from __future__ import annotations

import asyncio

import pytest

from cayu import CredentialProxy, PassthroughProxy, ProxyAuthorizationResult
from cayu.core.tools import ToolContext
from cayu.environments import Environment, EnvironmentSpec, copy_environment
from cayu.vaults import SecretNotFound, SecretRef, StaticVault


def test_passthrough_proxy_resolves_through_vault() -> None:
    proxy = PassthroughProxy(StaticVault({"api_key": "sk-secret-123"}))

    resolved = asyncio.run(proxy.resolve(SecretRef(name="api_key")))

    assert resolved.name == "api_key"
    assert resolved.value.get_secret_value() == "sk-secret-123"


def test_passthrough_proxy_passes_scope_to_vault() -> None:
    proxy = PassthroughProxy(StaticVault({"api_key": "sk-secret-123"}))

    resolved = asyncio.run(proxy.resolve(SecretRef(name="api_key"), scope={"session_id": "sess_1"}))

    assert resolved.metadata["scope"] == {"session_id": "sess_1"}


def test_passthrough_proxy_raises_vault_errors() -> None:
    proxy = PassthroughProxy(StaticVault({"api_key": "sk-secret-123"}))

    with pytest.raises(SecretNotFound):
        asyncio.run(proxy.resolve(SecretRef(name="missing")))


def test_passthrough_proxy_allows_all_destinations_for_trusted_local_use() -> None:
    proxy = PassthroughProxy(StaticVault({"api_key": "sk-secret-123"}))

    result = asyncio.run(
        proxy.authorize_request(
            destination="https://api.example.com",
            credential=SecretRef(name="api_key"),
            action="send_email",
            metadata={"tenant": "acme"},
        )
    )

    assert result.allowed is True


def test_passthrough_proxy_validates_inputs() -> None:
    with pytest.raises(TypeError, match="PassthroughProxy requires a Vault"):
        PassthroughProxy("not a vault")  # type: ignore[arg-type]

    proxy = PassthroughProxy(StaticVault({"api_key": "sk-secret-123"}))

    with pytest.raises(TypeError, match="Proxy secret refs must be SecretRef"):
        asyncio.run(proxy.resolve("not a ref"))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="destination"):
        asyncio.run(proxy.authorize_request(destination=" "))

    with pytest.raises(TypeError, match="credential"):
        asyncio.run(
            proxy.authorize_request(
                destination="https://api.example.com",
                credential="not a ref",  # type: ignore[arg-type]
            )
        )


def test_proxy_authorization_result_validates_and_copies_metadata() -> None:
    metadata = {"destination": "api.example.com"}
    result = ProxyAuthorizationResult(
        allowed=False,
        reason="destination denied",
        metadata=metadata,
    )
    metadata["destination"] = "mutated.example.com"

    assert result.allowed is False
    assert result.reason == "destination denied"
    assert result.metadata == {"destination": "api.example.com"}

    with pytest.raises(ValueError, match="reason"):
        ProxyAuthorizationResult(allowed=False, reason=" ")

    with pytest.raises(ValueError, match="require a reason"):
        ProxyAuthorizationResult(allowed=False)


def test_credential_proxy_is_abstract() -> None:
    with pytest.raises(TypeError):
        CredentialProxy()


def test_environment_accepts_copies_and_rejects_proxy() -> None:
    proxy = PassthroughProxy(StaticVault({"api_key": "sk-secret-123"}))

    assert Environment(EnvironmentSpec(name="no_proxy")).proxy is None

    environment = Environment(EnvironmentSpec(name="env"), proxy=proxy)
    copied = copy_environment(environment)

    assert environment.proxy is proxy
    assert copied.proxy is proxy

    with pytest.raises(TypeError, match="proxy must be a CredentialProxy"):
        Environment(
            EnvironmentSpec(name="env"),
            proxy="not a proxy",  # type: ignore[arg-type]
        )


def test_tool_context_accepts_and_excludes_proxy_from_serialization() -> None:
    proxy = PassthroughProxy(StaticVault({"api_key": "sk-secret-123"}))

    assert ToolContext(session_id="sess_1").proxy is None

    context = ToolContext(session_id="sess_1", proxy=proxy)

    assert context.proxy is proxy
    assert "proxy" not in context.model_dump()
