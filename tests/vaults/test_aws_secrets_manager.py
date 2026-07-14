from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cayu import SecretNotFound, SecretRef, SecretsManagerVault, VaultError


class _ClientError(RuntimeError):
    def __init__(self, code: str, message: str = "backend detail") -> None:
        self.response = {"Error": {"Code": code, "Message": message}}
        super().__init__(message)


class _SecretsManagerClient:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def get_secret_value(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        value = self.responses[kwargs["SecretId"]]
        if isinstance(value, BaseException):
            raise value
        return value


def test_secrets_manager_vault_resolves_allowlisted_logical_name() -> None:
    client = _SecretsManagerClient(
        {
            "prod/cayu/github": {
                "ARN": "arn:aws:secretsmanager:us-east-1:123:secret:github",
                "Name": "prod/cayu/github",
                "SecretString": "ghp_real_secret",
                "VersionId": "version-1",
                "VersionStages": ["AWSCURRENT"],
            }
        }
    )
    vault = SecretsManagerVault(
        {"github_token": "prod/cayu/github"},
        client=client,
        metadata={"github_token": {"owner": "platform"}},
    )

    ref = asyncio.run(vault.get("github_token", scope={"session_id": "sess_1"}))
    resolved = asyncio.run(vault.resolve(ref, scope={"session_id": "sess_1"}))

    assert ref == SecretRef(
        name="github_token",
        handle="aws-secretsmanager:prod/cayu/github",
        metadata={
            "owner": "platform",
            "provider": "aws-secrets-manager",
            "scope": {"session_id": "sess_1"},
        },
    )
    assert resolved.name == "github_token"
    assert resolved.value.get_secret_value() == "ghp_real_secret"
    assert "ghp_real_secret" not in repr(resolved)
    assert resolved.metadata == {
        "owner": "platform",
        "provider": "aws-secrets-manager",
        "secret_arn": "arn:aws:secretsmanager:us-east-1:123:secret:github",
        "version_id": "version-1",
        "version_stages": ["AWSCURRENT"],
        "scope": {"session_id": "sess_1"},
    }
    assert client.calls == [{"SecretId": "prod/cayu/github", "VersionStage": "AWSCURRENT"}]


def test_secrets_manager_vault_rejects_unknown_or_mismatched_refs_before_aws() -> None:
    client = _SecretsManagerClient({})
    vault = SecretsManagerVault({"github_token": "prod/cayu/github"}, client=client)

    with pytest.raises(SecretNotFound, match="missing"):
        asyncio.run(vault.get("missing"))
    with pytest.raises(SecretNotFound, match="missing"):
        asyncio.run(vault.resolve(SecretRef(name="missing")))
    with pytest.raises(VaultError, match="does not match"):
        asyncio.run(
            vault.resolve(
                SecretRef(
                    name="github_token",
                    handle="aws-secretsmanager:prod/cayu/other",
                )
            )
        )

    assert client.calls == []


def test_secrets_manager_vault_maps_backend_errors_without_secret_material() -> None:
    missing = _ClientError("ResourceNotFoundException", "secret id was absent")
    denied = _ClientError("AccessDeniedException", "sensitive backend diagnostic")
    client = _SecretsManagerClient({"missing-id": missing, "denied-id": denied})
    vault = SecretsManagerVault({"missing": "missing-id", "denied": "denied-id"}, client=client)

    with pytest.raises(SecretNotFound, match="missing") as missing_error:
        asyncio.run(vault.resolve(SecretRef(name="missing")))
    with pytest.raises(VaultError, match="denied") as denied_error:
        asyncio.run(vault.resolve(SecretRef(name="denied")))

    assert "backend diagnostic" not in str(denied_error.value)
    assert "secret id" not in str(missing_error.value)


def test_secrets_manager_vault_rejects_binary_and_blank_values() -> None:
    client = _SecretsManagerClient(
        {
            "binary": {"SecretBinary": b"bytes"},
            "blank": {"SecretString": "  "},
        }
    )
    vault = SecretsManagerVault({"binary": "binary", "blank": "blank"}, client=client)

    with pytest.raises(VaultError, match="text secret"):
        asyncio.run(vault.resolve(SecretRef(name="binary")))
    with pytest.raises(SecretNotFound, match="blank"):
        asyncio.run(vault.resolve(SecretRef(name="blank")))


def test_secrets_manager_vault_rejects_client_configuration_conflicts() -> None:
    client = _SecretsManagerClient({})

    with pytest.raises(ValueError, match="injected client"):
        SecretsManagerVault({"token": "token-id"}, client=client, profile_name="prod")
