from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from pydantic import SecretStr

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.vaults.base import ResolvedSecret, SecretNotFound, SecretRef, Vault


class LocalEnvVault(Vault):
    """Resolve secret refs from environment variables in the trusted app process."""

    def __init__(
        self,
        mapping: Mapping[str, str],
        *,
        metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if not isinstance(mapping, Mapping):
            raise TypeError("LocalEnvVault mapping must be a mapping.")
        self._mapping: dict[str, str] = {}
        for name, env_name in mapping.items():
            secret_name = require_clean_nonblank(name, "secret name")
            environment_name = require_clean_nonblank(env_name, "environment variable name")
            self._mapping[secret_name] = environment_name

        self._metadata: dict[str, dict[str, Any]] = {}
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("LocalEnvVault metadata must be a mapping.")
            for name, value in metadata.items():
                secret_name = require_clean_nonblank(name, "metadata secret name")
                if secret_name not in self._mapping:
                    raise ValueError(f"Metadata provided for unknown secret: {secret_name}")
                if not isinstance(value, Mapping):
                    raise TypeError("LocalEnvVault metadata values must be mappings.")
                self._metadata[secret_name] = copy_json_value(value, "metadata")

    async def get(self, name: str, *, scope: dict[str, Any] | None = None) -> SecretRef:
        secret_name = require_clean_nonblank(name, "name")
        if secret_name not in self._mapping:
            raise SecretNotFound(f"Secret not found: {secret_name}")
        return SecretRef(
            name=secret_name,
            handle=f"env:{self._mapping[secret_name]}",
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
        env_name = self._mapping.get(secret_name)
        if env_name is None:
            raise SecretNotFound(f"Secret not found: {secret_name}")
        value = os.environ.get(env_name)
        if value is None:
            raise SecretNotFound(f"Environment variable not set for secret: {secret_name}")
        try:
            require_nonblank(value, "secret value")
        except ValueError as exc:
            raise SecretNotFound(
                f"Environment variable is blank for secret: {secret_name}"
            ) from exc
        return ResolvedSecret(
            name=secret_name,
            value=SecretStr(value),
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
