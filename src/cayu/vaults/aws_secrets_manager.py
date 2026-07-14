from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping
from typing import Any

from pydantic import SecretStr

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.vaults.base import ResolvedSecret, SecretNotFound, SecretRef, Vault, VaultError

_HANDLE_PREFIX = "aws-secretsmanager:"


class SecretsManagerVault(Vault):
    """Resolve allowlisted logical names from AWS Secrets Manager.

    The mapping is intentionally required: callers can request only application-
    defined logical names, never pass an arbitrary Secrets Manager identifier
    through the model or sandbox boundary.
    """

    def __init__(
        self,
        mapping: Mapping[str, str],
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        endpoint_url: str | None = None,
        version_stage: str = "AWSCURRENT",
        client: Any | None = None,
        metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if not isinstance(mapping, Mapping):
            raise TypeError("SecretsManagerVault mapping must be a mapping.")
        self._mapping: dict[str, str] = {}
        for name, secret_id in mapping.items():
            logical_name = require_clean_nonblank(name, "secret name")
            self._mapping[logical_name] = require_clean_nonblank(secret_id, "secret id")
        self._version_stage = require_clean_nonblank(version_stage, "version_stage")
        self._region_name = _optional_clean_string(region_name, "region_name")
        self._profile_name = _optional_clean_string(profile_name, "profile_name")
        self._endpoint_url = _optional_clean_string(endpoint_url, "endpoint_url")
        if client is not None and (
            self._profile_name is not None or self._endpoint_url is not None
        ):
            raise ValueError(
                "An injected client cannot be combined with profile_name or endpoint_url."
            )
        self._client = client
        self._client_lock = asyncio.Lock()

        self._metadata: dict[str, dict[str, Any]] = {}
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("SecretsManagerVault metadata must be a mapping.")
            for name, value in metadata.items():
                logical_name = require_clean_nonblank(name, "metadata secret name")
                if logical_name not in self._mapping:
                    raise ValueError(f"Metadata provided for unknown secret: {logical_name}")
                if not isinstance(value, Mapping):
                    raise TypeError("SecretsManagerVault metadata values must be mappings.")
                self._metadata[logical_name] = copy_json_value(dict(value), "metadata")

    async def get(self, name: str, *, scope: dict[str, Any] | None = None) -> SecretRef:
        logical_name, secret_id = self._target(name)
        return SecretRef(
            name=logical_name,
            handle=f"{_HANDLE_PREFIX}{secret_id}",
            metadata=self._metadata_for(logical_name, scope),
        )

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        if type(ref) is not SecretRef:
            raise TypeError("Vault refs must be SecretRef instances.")
        logical_name, secret_id = self._target(ref.name)
        expected_handle = f"{_HANDLE_PREFIX}{secret_id}"
        if ref.handle is not None and ref.handle != expected_handle:
            raise VaultError(
                f"Secret reference handle does not match configured target: {logical_name}"
            )
        client = await self._get_client()
        try:
            response = await asyncio.to_thread(
                client.get_secret_value,
                SecretId=secret_id,
                VersionStage=self._version_stage,
            )
        except Exception as exc:
            if _aws_error_code(exc) == "ResourceNotFoundException":
                raise SecretNotFound(f"Secret not found: {logical_name}") from exc
            raise VaultError(f"Secrets Manager could not resolve secret: {logical_name}") from exc
        if not isinstance(response, Mapping):
            raise VaultError(f"Secrets Manager returned an invalid response for: {logical_name}")
        value = response.get("SecretString")
        if type(value) is not str:
            if "SecretBinary" in response:
                raise VaultError(f"Secrets Manager secret must be a text secret: {logical_name}")
            raise SecretNotFound(f"Secret value not found: {logical_name}")
        try:
            require_nonblank(value, "secret value")
        except ValueError as exc:
            raise SecretNotFound(f"Secret value is blank: {logical_name}") from exc

        metadata = self._metadata_for(logical_name, scope)
        arn = response.get("ARN")
        if type(arn) is str and arn.strip():
            metadata["secret_arn"] = arn
        version_id = response.get("VersionId")
        if type(version_id) is str and version_id.strip():
            metadata["version_id"] = version_id
        stages = response.get("VersionStages")
        if isinstance(stages, list) and all(type(stage) is str for stage in stages):
            metadata["version_stages"] = list(stages)
        return ResolvedSecret(
            name=logical_name,
            value=SecretStr(value),
            metadata=metadata,
        )

    def _target(self, name: str) -> tuple[str, str]:
        logical_name = require_clean_nonblank(name, "name")
        secret_id = self._mapping.get(logical_name)
        if secret_id is None:
            raise SecretNotFound(f"Secret not found: {logical_name}")
        return logical_name, secret_id

    def _metadata_for(
        self,
        name: str,
        scope: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = copy_json_value(self._metadata.get(name, {}), "metadata")
        metadata["provider"] = "aws-secrets-manager"
        if scope:
            metadata["scope"] = copy_json_value(scope, "scope")
        return metadata

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = await asyncio.to_thread(self._create_client)
        return self._client

    def _create_client(self) -> Any:
        boto3 = _boto3_module()
        session_options: dict[str, Any] = {}
        if self._profile_name is not None:
            session_options["profile_name"] = self._profile_name
        session = boto3.Session(**session_options)
        client_options: dict[str, Any] = {}
        if self._region_name is not None:
            client_options["region_name"] = self._region_name
        if self._endpoint_url is not None:
            client_options["endpoint_url"] = self._endpoint_url
        return session.client("secretsmanager", **client_options)


def _optional_clean_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return require_clean_nonblank(value, field_name)


def _aws_error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return None
    error = response.get("Error")
    if not isinstance(error, Mapping):
        return None
    code = error.get("Code")
    return code if type(code) is str else None


def _boto3_module() -> Any:
    try:
        return importlib.import_module("boto3")
    except ModuleNotFoundError as exc:
        if exc.name != "boto3":
            raise
        raise RuntimeError(
            "SecretsManagerVault requires the optional AWS dependencies; install cayu[aws]."
        ) from exc
