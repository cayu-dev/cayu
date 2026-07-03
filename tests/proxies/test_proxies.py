from __future__ import annotations

import asyncio

import pytest

from cayu import AllowlistProxy, CredentialProxy, PassthroughProxy, ProxyAuthorizationResult
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


def test_allowlist_proxy_authorizes_only_listed_destinations() -> None:
    proxy = AllowlistProxy(
        StaticVault({"api_key": "sk-secret-123"}),
        allowed_destinations=["https://api.anthropic.com", "*.internal.example.com"],
    )

    allowed = asyncio.run(proxy.authorize_request(destination="https://api.anthropic.com/v1"))
    allowed_bare_host = asyncio.run(proxy.authorize_request(destination="API.ANTHROPIC.COM:443"))
    wildcard = asyncio.run(
        proxy.authorize_request(destination="https://tools.internal.example.com/x")
    )
    apex_of_wildcard = asyncio.run(
        proxy.authorize_request(destination="https://internal.example.com")
    )
    denied = asyncio.run(
        proxy.authorize_request(
            destination="https://evil.example.net",
            credential=SecretRef(name="api_key"),
            action="exfiltrate",
        )
    )

    assert allowed.allowed is True
    assert allowed.metadata["destination_host"] == "api.anthropic.com"
    assert allowed_bare_host.allowed is True
    assert wildcard.allowed is True
    assert apex_of_wildcard.allowed is False
    assert denied.allowed is False
    assert "evil.example.net" in (denied.reason or "")


def test_allowlist_proxy_resolve_is_fail_closed_on_destination_scope() -> None:
    proxy = AllowlistProxy(
        StaticVault({"api_key": "sk-secret-123"}),
        allowed_destinations=["api.anthropic.com"],
    )

    resolved = asyncio.run(
        proxy.resolve(
            SecretRef(name="api_key"),
            scope={"destination": "https://api.anthropic.com"},
        )
    )
    assert resolved.value.get_secret_value() == "sk-secret-123"

    with pytest.raises(ValueError, match="destination"):
        asyncio.run(proxy.resolve(SecretRef(name="api_key")))

    with pytest.raises(PermissionError, match="denied"):
        asyncio.run(
            proxy.resolve(
                SecretRef(name="api_key"),
                scope={"destination": "https://evil.example.net"},
            )
        )


def test_allowlist_proxy_validates_inputs() -> None:
    vault = StaticVault({"api_key": "sk-secret-123"})

    with pytest.raises(TypeError, match="requires a Vault"):
        AllowlistProxy("not a vault", allowed_destinations=["api.anthropic.com"])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="sequence"):
        AllowlistProxy(vault, allowed_destinations="api.anthropic.com")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="at least one"):
        AllowlistProxy(vault, allowed_destinations=[])

    with pytest.raises(ValueError, match="cannot be blank"):
        AllowlistProxy(vault, allowed_destinations=[" "])

    proxy = AllowlistProxy(vault, allowed_destinations=["api.anthropic.com"])
    assert proxy.allowed_destinations == ("api.anthropic.com",)

    with pytest.raises(TypeError, match="SecretRef"):
        asyncio.run(
            proxy.resolve("not a ref", scope={"destination": "api.anthropic.com"})  # type: ignore[arg-type]
        )
