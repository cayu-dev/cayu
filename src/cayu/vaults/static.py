from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import SecretStr

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.vaults.base import ResolvedSecret, SecretNotFound, SecretRef, Vault


class StaticVault(Vault):
    """In-memory vault for tests and trusted local development."""

    def __init__(
        self,
        secrets: Mapping[str, str | SecretStr],
        *,
        metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if not isinstance(secrets, Mapping):
            raise TypeError("StaticVault secrets must be a mapping.")
        self._secrets: dict[str, SecretStr] = {}
        for name, value in secrets.items():
            secret_name = require_clean_nonblank(name, "secret name")
            if type(value) is SecretStr:
                require_nonblank(value.get_secret_value(), "secret value")
                secret_value = value
            elif type(value) is str:
                require_nonblank(value, "secret value")
                secret_value = SecretStr(value)
            else:
                raise TypeError("StaticVault secret values must be strings or SecretStr.")
            self._secrets[secret_name] = secret_value

        self._metadata: dict[str, dict[str, Any]] = {}
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("StaticVault metadata must be a mapping.")
            for name, value in metadata.items():
                secret_name = require_clean_nonblank(name, "metadata secret name")
                if secret_name not in self._secrets:
                    raise ValueError(f"Metadata provided for unknown secret: {secret_name}")
                if not isinstance(value, Mapping):
                    raise TypeError("StaticVault metadata values must be mappings.")
                self._metadata[secret_name] = copy_json_value(value, "metadata")

    async def get(self, name: str, *, scope: dict[str, Any] | None = None) -> SecretRef:
        secret_name = require_clean_nonblank(name, "name")
        if secret_name not in self._secrets:
            raise SecretNotFound(f"Secret not found: {secret_name}")
        return SecretRef(
            name=secret_name,
            handle=f"static:{secret_name}",
            metadata=self._metadata_for(secret_name, scope),
        )

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        if type(ref) is not SecretRef:
            raise TypeError("Vault refs must be SecretRef instances.")
        secret_name = require_clean_nonblank(ref.name, "name")
        value = self._secrets.get(secret_name)
        if value is None:
            raise SecretNotFound(f"Secret not found: {secret_name}")
        return ResolvedSecret(
            name=secret_name,
            value=value,
            metadata=self._metadata_for(secret_name, scope),
        )

    def _metadata_for(
        self,
        name: str,
        scope: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = copy_json_value(self._metadata.get(name, {}), "metadata")
        if scope:
            metadata["scope"] = copy_json_value(scope, "scope")
        return metadata
