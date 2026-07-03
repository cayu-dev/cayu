from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.proxies.base import CredentialProxy, ProxyAuthorizationResult
from cayu.vaults import ResolvedSecret, SecretRef, Vault


class PassthroughProxy(CredentialProxy):
    """Credential proxy for trusted local development.

    This adapter delegates secret resolution directly to the configured vault
    and allows every outbound destination. It is not a sandbox security
    boundary; production apps should provide a scoped proxy implementation.
    """

    def __init__(self, vault: Vault) -> None:
        if not isinstance(vault, Vault):
            raise TypeError("PassthroughProxy requires a Vault.")
        self._vault = vault

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        if type(ref) is not SecretRef:
            raise TypeError("Proxy secret refs must be SecretRef instances.")
        return await self._vault.resolve(ref, scope=scope)

    async def authorize_request(
        self,
        *,
        destination: str,
        credential: SecretRef | None = None,
        action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProxyAuthorizationResult:
        _validate_authorize_request_inputs(
            destination=destination,
            credential=credential,
            action=action,
            metadata=metadata,
        )
        return ProxyAuthorizationResult(allowed=True)


class AllowlistProxy(CredentialProxy):
    """Vault-backed credential proxy gated by an explicit destination allowlist.

    ``authorize_request`` allows only destinations whose host matches the
    allowlist (exact hostnames or ``*.example.com`` subdomain wildcards), the
    confused-deputy containment the passthrough tier skips. ``resolve`` is
    fail-closed: it requires ``scope["destination"]`` and refuses to hand out
    a secret bound for a destination that is not on the allowlist.
    """

    def __init__(self, vault: Vault, *, allowed_destinations: Sequence[str]) -> None:
        if not isinstance(vault, Vault):
            raise TypeError("AllowlistProxy requires a Vault.")
        if isinstance(allowed_destinations, str | bytes) or not isinstance(
            allowed_destinations, Sequence
        ):
            raise TypeError("allowed_destinations must be a sequence of destination hosts.")
        hosts: list[str] = []
        for destination in allowed_destinations:
            host = _destination_host(
                require_clean_nonblank(destination, "allowed_destinations entry")
            )
            if host not in hosts:
                hosts.append(host)
        if not hosts:
            raise ValueError("AllowlistProxy requires at least one allowed destination.")
        self._vault = vault
        self._allowed_hosts = tuple(hosts)

    @property
    def allowed_destinations(self) -> tuple[str, ...]:
        return self._allowed_hosts

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        if type(ref) is not SecretRef:
            raise TypeError("Proxy secret refs must be SecretRef instances.")
        destination = (scope or {}).get("destination")
        if type(destination) is not str or not destination.strip():
            raise ValueError("AllowlistProxy.resolve requires scope['destination'].")
        if not self._destination_allowed(destination):
            raise PermissionError(
                f"AllowlistProxy denied secret resolution for destination: {destination}"
            )
        return await self._vault.resolve(ref, scope=scope)

    async def authorize_request(
        self,
        *,
        destination: str,
        credential: SecretRef | None = None,
        action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProxyAuthorizationResult:
        _validate_authorize_request_inputs(
            destination=destination,
            credential=credential,
            action=action,
            metadata=metadata,
        )
        host = _destination_host(destination)
        if self._destination_allowed(destination):
            return ProxyAuthorizationResult(allowed=True, metadata={"destination_host": host})
        return ProxyAuthorizationResult(
            allowed=False,
            reason=f"Destination not in allowlist: {host}",
            metadata={"destination_host": host},
        )

    def _destination_allowed(self, destination: str) -> bool:
        host = _destination_host(destination)
        for allowed in self._allowed_hosts:
            if allowed.startswith("*."):
                if host.endswith(allowed[1:]) and host != allowed[2:]:
                    return True
            elif host == allowed:
                return True
        return False


def _validate_authorize_request_inputs(
    *,
    destination: str,
    credential: SecretRef | None,
    action: str | None,
    metadata: dict[str, Any] | None,
) -> None:
    require_clean_nonblank(destination, "destination")
    if credential is not None and type(credential) is not SecretRef:
        raise TypeError("credential must be a SecretRef.")
    if action is not None:
        require_clean_nonblank(action, "action")
    if metadata is not None:
        copy_json_value(metadata, "metadata")


def _destination_host(destination: str) -> str:
    """Extract the lowercased host from a URL or bare host[:port] destination."""

    value = destination.strip()
    parsed = urlsplit(value if "//" in value else f"//{value}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"Destination has no host: {destination}")
    return host.lower()
