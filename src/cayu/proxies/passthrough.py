from __future__ import annotations

from typing import Any

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
        require_clean_nonblank(destination, "destination")
        if credential is not None and type(credential) is not SecretRef:
            raise TypeError("credential must be a SecretRef.")
        if action is not None:
            require_clean_nonblank(action, "action")
        if metadata is not None:
            copy_json_value(metadata, "metadata")
        return ProxyAuthorizationResult(allowed=True)
