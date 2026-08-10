from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal

import pytest
from pydantic import SecretStr

from cayu import (
    ChainVault,
    LocalEnvVault,
    ResolvedSecret,
    RoutedVault,
    SecretEnv,
    SecretRef,
    SecretsManagerVault,
    StaticVault,
    Vault,
    extract_durable_value_error,
    secret_env_refs,
)
from cayu._validation import MAX_DURABLE_JSON_NESTING

_SECRET_CANARY = "vault-private-secret-value-canary"
_TEXT_CANARY = "vault-private-text-canary"

TextBoundary = Literal[
    "secret_ref_name",
    "secret_ref_handle",
    "secret_env_name",
    "resolved_secret_name",
    "secret_env_mapping_key",
    "secret_env_mutated_sequence_name",
    "local_logical_name",
    "local_environment_name",
    "local_metadata_name",
    "static_logical_name",
    "static_metadata_name",
    "aws_logical_name",
    "aws_provider_identifier",
    "aws_metadata_name",
    "routed_name",
]

_TEXT_BOUNDARIES: tuple[TextBoundary, ...] = (
    "secret_ref_name",
    "secret_ref_handle",
    "secret_env_name",
    "resolved_secret_name",
    "secret_env_mapping_key",
    "secret_env_mutated_sequence_name",
    "local_logical_name",
    "local_environment_name",
    "local_metadata_name",
    "static_logical_name",
    "static_metadata_name",
    "aws_logical_name",
    "aws_provider_identifier",
    "aws_metadata_name",
    "routed_name",
)

VaultKind = Literal["local", "static", "aws"]
_VAULT_KINDS: tuple[VaultKind, ...] = ("local", "static", "aws")
ResolverKind = Literal["local", "static", "aws", "chain", "routed"]
_RESOLVER_KINDS: tuple[ResolverKind, ...] = (
    "local",
    "static",
    "aws",
    "chain",
    "routed",
)


class _NoCallSecretsManagerClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_secret_value(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        raise AssertionError("portable configuration validation must precede provider access")


def _exercise_text_boundary(boundary: TextBoundary, value: str) -> None:
    ref = SecretRef(name="token")
    if boundary == "secret_ref_name":
        SecretRef(name=value)
    elif boundary == "secret_ref_handle":
        SecretRef(name="token", handle=value)
    elif boundary == "secret_env_name":
        SecretEnv(name=value, ref=ref)
    elif boundary == "resolved_secret_name":
        ResolvedSecret(name=value, value=SecretStr(_SECRET_CANARY))
    elif boundary == "secret_env_mapping_key":
        secret_env_refs({value: ref})
    elif boundary == "secret_env_mutated_sequence_name":
        entry = SecretEnv(name="TOKEN", ref=ref)
        entry.name = value
        secret_env_refs([entry])
    elif boundary == "local_logical_name":
        LocalEnvVault({value: "CAYU_TOKEN"})
    elif boundary == "local_environment_name":
        LocalEnvVault({"token": value})
    elif boundary == "local_metadata_name":
        LocalEnvVault({"token": "CAYU_TOKEN"}, metadata={value: {}})
    elif boundary == "static_logical_name":
        StaticVault({value: _SECRET_CANARY})
    elif boundary == "static_metadata_name":
        StaticVault({"token": _SECRET_CANARY}, metadata={value: {}})
    elif boundary == "aws_logical_name":
        SecretsManagerVault({value: "prod/token"}, client=_NoCallSecretsManagerClient())
    elif boundary == "aws_provider_identifier":
        SecretsManagerVault({"token": value}, client=_NoCallSecretsManagerClient())
    elif boundary == "aws_metadata_name":
        SecretsManagerVault(
            {"token": "prod/token"},
            client=_NoCallSecretsManagerClient(),
            metadata={value: {}},
        )
    else:
        RoutedVault({value: StaticVault({"token": _SECRET_CANARY})})


@pytest.mark.parametrize("boundary", _TEXT_BOUNDARIES)
@pytest.mark.parametrize(
    ("value", "expected_code"),
    [
        (f"{_TEXT_CANARY}\x00", "nul_character"),
        (f"{_TEXT_CANARY}\ud800", "unicode_surrogate"),
    ],
)
def test_vault_text_boundaries_reject_nonportable_text_without_disclosure(
    boundary: TextBoundary,
    value: str,
    expected_code: str,
) -> None:
    with pytest.raises(ValueError) as raised:
        _exercise_text_boundary(boundary, value)

    durable_error = extract_durable_value_error(raised.value)
    assert durable_error is not None
    assert durable_error.code == expected_code
    rendered = f"{raised.value!s} {raised.value!r}"
    assert _TEXT_CANARY not in rendered
    assert _SECRET_CANARY not in rendered


def _too_deep_value() -> dict[str, Any]:
    value: Any = "leaf"
    for _ in range(MAX_DURABLE_JSON_NESTING + 1):
        value = {"child": value}
    return value


def _circular_value() -> list[Any]:
    value: list[Any] = []
    value.append(value)
    return value


@pytest.mark.parametrize("vault_kind", _VAULT_KINDS)
@pytest.mark.parametrize(
    ("value_factory", "expected_code"),
    [
        (lambda: 2**63, "integer_out_of_range"),
        (lambda: -(2**63) - 1, "integer_out_of_range"),
        (lambda: float(2**63), "integral_float_out_of_range"),
        (lambda: float("nan"), "non_finite_number"),
        (lambda: f"{_TEXT_CANARY}\x00", "nul_character"),
        (lambda: f"{_TEXT_CANARY}\ud800", "unicode_surrogate"),
        (lambda: object(), "invalid_json_type"),
        (_circular_value, "circular_reference"),
        (_too_deep_value, "nesting_too_deep"),
    ],
)
def test_builtin_vault_metadata_uses_the_complete_durable_json_contract(
    vault_kind: VaultKind,
    value_factory: Callable[[], Any],
    expected_code: str,
) -> None:
    client = _NoCallSecretsManagerClient()
    metadata = {"token": {"probe": value_factory()}}

    with pytest.raises(ValueError) as raised:
        _build_vault(vault_kind, metadata=metadata, client=client)

    durable_error = extract_durable_value_error(raised.value)
    assert durable_error is not None
    assert durable_error.code == expected_code
    rendered = f"{raised.value!s} {raised.value!r}"
    assert _TEXT_CANARY not in rendered
    assert _SECRET_CANARY not in rendered
    assert client.calls == []


@pytest.mark.parametrize("vault_kind", _VAULT_KINDS)
@pytest.mark.parametrize(
    ("key", "expected_code"),
    [
        (f"{_TEXT_CANARY}\x00", "nul_character"),
        (f"{_TEXT_CANARY}\ud800", "unicode_surrogate"),
        (1, "invalid_json_key"),
    ],
)
def test_builtin_vault_metadata_rejects_nonportable_object_keys(
    vault_kind: VaultKind,
    key: Any,
    expected_code: str,
) -> None:
    client = _NoCallSecretsManagerClient()

    with pytest.raises(ValueError) as raised:
        _build_vault(
            vault_kind,
            metadata={"token": {key: "value"}},
            client=client,
        )

    durable_error = extract_durable_value_error(raised.value)
    assert durable_error is not None
    assert durable_error.code == expected_code
    assert _TEXT_CANARY not in f"{raised.value!s} {raised.value!r}"
    assert client.calls == []


def _build_vault(
    vault_kind: VaultKind,
    *,
    metadata: dict[str, dict[str, Any]],
    client: _NoCallSecretsManagerClient,
) -> Vault:
    if vault_kind == "local":
        return LocalEnvVault({"token": "CAYU_TOKEN"}, metadata=metadata)
    if vault_kind == "static":
        return StaticVault({"token": _SECRET_CANARY}, metadata=metadata)
    return SecretsManagerVault(
        {"token": "prod/token"},
        client=client,
        metadata=metadata,
    )


@pytest.mark.parametrize("vault_kind", _VAULT_KINDS)
def test_builtin_vault_metadata_is_normalized_and_defensively_copied(
    vault_kind: VaultKind,
) -> None:
    client = _NoCallSecretsManagerClient()
    metadata: dict[str, dict[str, Any]] = {
        "token": {
            "integral": 42.0,
            "negative_zero": -0.0,
            "fractional": 1.25,
            "minimum": -(2**63),
            "maximum": 2**63 - 1,
            "unicode": "Zażółć 😀",
            "nested": {"value": "original", "items": [1.0, True, None]},
        }
    }
    vault = _build_vault(vault_kind, metadata=metadata, client=client)

    metadata["token"]["nested"]["value"] = "mutated"
    ref = asyncio.run(vault.get("token"))

    assert ref.metadata["integral"] == 42
    assert type(ref.metadata["integral"]) is int
    assert ref.metadata["negative_zero"] == 0
    assert type(ref.metadata["negative_zero"]) is int
    assert ref.metadata["fractional"] == 1.25
    assert ref.metadata["minimum"] == -(2**63)
    assert ref.metadata["maximum"] == 2**63 - 1
    assert ref.metadata["unicode"] == "Zażółć 😀"
    assert ref.metadata["nested"] == {
        "value": "original",
        "items": [1, True, None],
    }
    assert client.calls == []


def test_public_secret_values_preserve_valid_unicode_and_owned_metadata() -> None:
    metadata = {"nested": {"value": 1.0}}
    ref = SecretRef(name="秘密😀", handle="vault://秘密😀", metadata=metadata)
    secret_env = SecretEnv(name="VALID_TOKEN", ref=ref, metadata=metadata)
    resolved = ResolvedSecret(
        name="秘密😀",
        value=SecretStr(_SECRET_CANARY),
        metadata=metadata,
    )

    metadata["nested"]["value"] = 2

    assert ref.metadata == {"nested": {"value": 1}}
    assert secret_env.metadata == {"nested": {"value": 1}}
    assert resolved.metadata == {"nested": {"value": 1}}
    assert _SECRET_CANARY not in repr(resolved)


def test_builtin_vaults_preserve_supported_unicode_names_and_provider_identifiers() -> None:
    client = _NoCallSecretsManagerClient()
    aws = SecretsManagerVault({"秘密😀": "prod/秘密😀"}, client=client)
    static = StaticVault({"秘密😀": _SECRET_CANARY})
    routed = RoutedVault({"秘密😀": static})

    aws_ref = asyncio.run(aws.get("秘密😀"))
    routed_ref = asyncio.run(routed.get("秘密😀"))

    assert aws_ref.handle == "aws-secretsmanager:prod/秘密😀"
    assert routed_ref.name == "秘密😀"
    assert client.calls == []


@pytest.mark.parametrize("resolver_kind", _RESOLVER_KINDS)
@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("name", "nul_character"),
        ("handle", "unicode_surrogate"),
        ("metadata_bigint", "integer_out_of_range"),
        ("metadata_unsupported", "invalid_json_type"),
    ],
)
def test_builtin_vault_resolvers_revalidate_mutated_secret_refs_before_side_effects(
    resolver_kind: ResolverKind,
    mutation: str,
    expected_code: str,
) -> None:
    client = _NoCallSecretsManagerClient()
    leaf = _build_vault(
        "aws" if resolver_kind == "aws" else "static",
        metadata={"token": {}},
        client=client,
    )
    if resolver_kind == "local":
        vault: Vault = _build_vault("local", metadata={"token": {}}, client=client)
    elif resolver_kind == "chain":
        vault = ChainVault(leaf)
    elif resolver_kind == "routed":
        vault = RoutedVault({"token": leaf})
    else:
        vault = leaf
    ref = SecretRef(name="token")
    if mutation == "name":
        ref.name = f"{_TEXT_CANARY}\x00"
    elif mutation == "handle":
        ref.handle = f"{_TEXT_CANARY}\ud800"
    elif mutation == "metadata_bigint":
        ref.metadata["probe"] = 2**63
    else:
        ref.metadata[_TEXT_CANARY] = object()

    with pytest.raises(ValueError) as raised:
        asyncio.run(vault.resolve(ref))

    durable_error = extract_durable_value_error(raised.value)
    assert durable_error is not None
    assert durable_error.code == expected_code
    assert _TEXT_CANARY not in f"{raised.value!s} {raised.value!r}"
    assert client.calls == []
