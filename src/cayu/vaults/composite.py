from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cayu._validation import require_clean_nonblank
from cayu.vaults.base import ResolvedSecret, SecretNotFound, SecretRef, Vault


class ChainVault(Vault):
    """Try each vault in order; the first that resolves the secret wins.

    A vault that does not have the secret raises ``SecretNotFound`` and the chain moves
    on. Any other error (for example a network failure from a dynamic vault) propagates,
    so it is not silently masked by a later vault in the chain.

    Note that a child which reports a *configured but currently unavailable* secret as
    ``SecretNotFound`` (for example ``LocalEnvVault`` when the mapped environment variable
    is unset) also triggers fall-through, so a chain will not surface that misconfiguration
    as an error. Use ``RoutedVault`` (or order the authoritative vault so it cannot be
    skipped) when a specific secret must come from one vault.
    """

    def __init__(self, *vaults: Vault) -> None:
        if not vaults:
            raise ValueError("ChainVault requires at least one vault.")
        for vault in vaults:
            if not isinstance(vault, Vault):
                raise TypeError("ChainVault entries must be Vault instances.")
        self._vaults: tuple[Vault, ...] = vaults

    async def get(self, name: str, *, scope: dict[str, Any] | None = None) -> SecretRef:
        name = require_clean_nonblank(name, "name")
        for vault in self._vaults:
            try:
                return await vault.get(name, scope=scope)
            except SecretNotFound:
                continue
        raise SecretNotFound(f"No vault could resolve secret: {name}")

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        if type(ref) is not SecretRef:
            raise TypeError("Vault refs must be SecretRef instances.")
        for vault in self._vaults:
            try:
                return await vault.resolve(ref, scope=scope)
            except SecretNotFound:
                continue
        raise SecretNotFound(f"No vault could resolve secret: {ref.name}")


class RoutedVault(Vault):
    """Route secret names to specific vaults, with an optional fallback for the rest.

    Unlike ``ChainVault``, routing is explicit: each name maps to exactly one vault, so
    a dynamic vault is only called for the names it owns. Names not in ``routes`` go to
    ``fallback`` (typically a static vault); with no route and no fallback, the lookup
    raises ``SecretNotFound`` without calling any vault.
    """

    def __init__(
        self,
        routes: Mapping[str, Vault],
        *,
        fallback: Vault | None = None,
    ) -> None:
        if not isinstance(routes, Mapping):
            raise TypeError("RoutedVault routes must be a mapping.")
        if fallback is not None and not isinstance(fallback, Vault):
            raise TypeError("RoutedVault fallback must be a Vault or None.")
        self._routes: dict[str, Vault] = {}
        for name, vault in routes.items():
            secret_name = require_clean_nonblank(name, "route name")
            if not isinstance(vault, Vault):
                raise TypeError("RoutedVault route values must be Vault instances.")
            self._routes[secret_name] = vault
        if not self._routes and fallback is None:
            raise ValueError("RoutedVault requires at least one route or a fallback.")
        self._fallback = fallback

    def _vault_for(self, name: str) -> Vault:
        vault = self._routes.get(name)
        if vault is None:
            vault = self._fallback
        if vault is None:
            raise SecretNotFound(f"No vault configured for secret: {name}")
        return vault

    async def get(self, name: str, *, scope: dict[str, Any] | None = None) -> SecretRef:
        secret_name = require_clean_nonblank(name, "name")
        return await self._vault_for(secret_name).get(secret_name, scope=scope)

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        if type(ref) is not SecretRef:
            raise TypeError("Vault refs must be SecretRef instances.")
        secret_name = require_clean_nonblank(ref.name, "name")
        return await self._vault_for(secret_name).resolve(ref, scope=scope)
