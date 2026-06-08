from __future__ import annotations

import asyncio

import pytest
from pydantic import SecretStr, ValidationError

from cayu import (
    REDACTED_SECRET,
    Environment,
    EnvironmentSpec,
    LocalEnvVault,
    ResolvedSecret,
    SecretEnv,
    SecretNotFound,
    SecretRedactor,
    SecretRef,
    StaticVault,
    VaultError,
    copy_secret_env,
)


def test_secret_env_is_reference_only_and_owns_metadata() -> None:
    metadata = {"scope": {"project": "alpha"}}
    ref = SecretRef(name="github_token", handle="vault://github", metadata=metadata)
    secret_env = SecretEnv(name="GITHUB_TOKEN", ref=ref, metadata=metadata)

    metadata["scope"]["project"] = "mutated"
    ref.metadata["scope"]["project"] = "mutated-ref"
    dumped = secret_env.model_dump()

    assert dumped == {
        "name": "GITHUB_TOKEN",
        "ref": {
            "name": "github_token",
            "handle": "vault://github",
            "metadata": {"scope": {"project": "alpha"}},
        },
        "metadata": {"scope": {"project": "alpha"}},
    }
    assert "value" not in dumped


def test_secret_env_rejects_invalid_boundary_data() -> None:
    with pytest.raises(ValidationError, match="cannot be blank"):
        SecretEnv(name=" ", ref=SecretRef(name="github_token"))

    with pytest.raises(ValidationError, match="extra"):
        SecretEnv(name="GITHUB_TOKEN", ref=SecretRef(name="github_token"), value="secret")  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="JSON-compatible"):
        SecretEnv(
            name="GITHUB_TOKEN",
            ref=SecretRef(name="github_token"),
            metadata={"bad": object()},
        )


def test_copy_secret_env_rejects_subclasses_before_attribute_access() -> None:
    class BadSecretEnv(SecretEnv):
        def __getattribute__(self, name):
            if name == "name":
                raise RuntimeError("secret env name access should not run")
            return super().__getattribute__(name)

    secret_env = BadSecretEnv.model_construct(
        name="TOKEN",
        ref=SecretRef(name="token"),
        metadata={},
    )

    with pytest.raises(TypeError, match="SecretEnv"):
        copy_secret_env(secret_env)


def test_static_vault_gets_and_resolves_secret_refs() -> None:
    vault = StaticVault(
        {"github_token": "ghp_test"},
        metadata={"github_token": {"owner": "user_1"}},
    )

    ref = asyncio.run(vault.get("github_token", scope={"session_id": "sess_1"}))
    resolved = asyncio.run(vault.resolve(ref, scope={"session_id": "sess_1"}))

    assert ref == SecretRef(
        name="github_token",
        handle="static:github_token",
        metadata={"owner": "user_1", "scope": {"session_id": "sess_1"}},
    )
    assert resolved.name == "github_token"
    assert str(resolved.value) == "**********"
    assert resolved.value.get_secret_value() == "ghp_test"
    assert resolved.metadata == {"owner": "user_1", "scope": {"session_id": "sess_1"}}


def test_static_vault_rejects_missing_and_blank_secrets() -> None:
    with pytest.raises(ValueError, match="cannot be blank"):
        StaticVault({"github_token": " "})

    with pytest.raises(ValueError, match="cannot be blank"):
        StaticVault({"github_token": SecretStr(" ")})

    vault = StaticVault({"github_token": "ghp_test"})
    with pytest.raises(SecretNotFound, match="missing"):
        asyncio.run(vault.get("missing"))

    with pytest.raises(SecretNotFound, match="missing"):
        asyncio.run(vault.resolve(SecretRef(name="missing")))


def test_local_env_vault_resolves_trusted_process_env(monkeypatch) -> None:
    monkeypatch.setenv("CAYU_TEST_GITHUB_TOKEN", "ghp_from_env")
    vault = LocalEnvVault(
        {"github_token": "CAYU_TEST_GITHUB_TOKEN"},
        metadata={"github_token": {"source": "env"}},
    )

    ref = asyncio.run(vault.get("github_token"))
    resolved = asyncio.run(vault.resolve(ref))

    assert ref == SecretRef(
        name="github_token",
        handle="env:CAYU_TEST_GITHUB_TOKEN",
        metadata={"source": "env"},
    )
    assert resolved.value.get_secret_value() == "ghp_from_env"
    assert resolved.metadata == {"source": "env"}


def test_local_env_vault_rejects_missing_mapping_and_unset_env(monkeypatch) -> None:
    monkeypatch.delenv("CAYU_TEST_MISSING_TOKEN", raising=False)
    monkeypatch.setenv("CAYU_TEST_BLANK_TOKEN", " ")
    vault = LocalEnvVault({"github_token": "CAYU_TEST_MISSING_TOKEN"})

    with pytest.raises(SecretNotFound, match="missing"):
        asyncio.run(vault.get("missing"))

    with pytest.raises(SecretNotFound, match="not set"):
        asyncio.run(vault.resolve(SecretRef(name="github_token")))

    blank_vault = LocalEnvVault({"github_token": "CAYU_TEST_BLANK_TOKEN"})
    with pytest.raises(SecretNotFound, match="blank"):
        asyncio.run(blank_vault.resolve(SecretRef(name="github_token")))


def test_environment_resolves_secret_through_attached_vault() -> None:
    environment = Environment(
        EnvironmentSpec(name="local"),
        vault=StaticVault({"github_token": "ghp_test"}),
    )

    resolved = asyncio.run(environment.resolve_secret(SecretRef(name="github_token")))

    assert resolved.value.get_secret_value() == "ghp_test"


def test_environment_requires_vault_for_secret_resolution() -> None:
    environment = Environment(EnvironmentSpec(name="local"))

    with pytest.raises(VaultError, match="no vault"):
        asyncio.run(environment.resolve_secret(SecretRef(name="github_token")))


def test_secret_redactor_redacts_strings_and_json_values() -> None:
    resolved = ResolvedSecret(name="github_token", value=SecretStr("ghp_secret"))
    redactor = SecretRedactor([resolved]).with_secret("npm_secret")

    assert redactor.redact_text("tokens: ghp_secret npm_secret") == (
        f"tokens: {REDACTED_SECRET} {REDACTED_SECRET}"
    )
    assert redactor.redact_json(
        {
            "stdout": "ghp_secret",
            "nested": ["npm_secret", {"safe": "ok"}],
        }
    ) == {
        "stdout": REDACTED_SECRET,
        "nested": [REDACTED_SECRET, {"safe": "ok"}],
    }


def test_secret_redactor_accepts_single_secret_values() -> None:
    assert SecretRedactor("token").redact_text("token total") == (f"{REDACTED_SECRET} total")
    assert SecretRedactor(SecretStr("token")).redact_text("token total") == (
        f"{REDACTED_SECRET} total"
    )
    assert (
        SecretRedactor(ResolvedSecret(name="github_token", value=SecretStr("token"))).redact_text(
            "token total"
        )
        == f"{REDACTED_SECRET} total"
    )


def test_secret_redactor_rejects_non_json_values() -> None:
    redactor = SecretRedactor(["secret"])

    with pytest.raises(ValueError, match="JSON-compatible"):
        redactor.redact_json(object())


def test_secret_redactor_rejects_blank_secrets() -> None:
    with pytest.raises(ValueError, match="cannot be blank"):
        SecretRedactor([" "])

    with pytest.raises(ValueError, match="cannot be blank"):
        SecretRedactor().with_secret(SecretStr(" "))
