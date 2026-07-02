from __future__ import annotations

import asyncio

import pytest
from pydantic import SecretStr, ValidationError

from cayu import (
    REDACTED_SECRET,
    ChainVault,
    Environment,
    EnvironmentSpec,
    LocalEnvVault,
    ResolvedSecret,
    RoutedVault,
    SecretEnv,
    SecretNotFound,
    SecretRedactor,
    SecretRef,
    StaticVault,
    Vault,
    VaultError,
    copy_resolved_secret,
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


def test_copy_resolved_secret_owns_value_and_metadata() -> None:
    metadata = {"scope": {"project": "alpha"}}
    secret = ResolvedSecret(
        name="github_token",
        value=SecretStr("ghp_secret"),
        metadata=metadata,
    )

    copied = copy_resolved_secret(secret)
    secret.value = SecretStr("mutated_secret")
    secret.metadata["scope"]["project"] = "mutated"
    metadata["scope"]["project"] = "external-mutated"

    assert copied == ResolvedSecret(
        name="github_token",
        value=SecretStr("ghp_secret"),
        metadata={"scope": {"project": "alpha"}},
    )


def test_copy_resolved_secret_rejects_subclasses_before_attribute_access() -> None:
    class BadResolvedSecret(ResolvedSecret):
        def __getattribute__(self, name):
            if name == "name":
                raise RuntimeError("secret name access should not run")
            return super().__getattribute__(name)

    secret = BadResolvedSecret.model_construct(
        name="token",
        value=SecretStr("secret"),
        metadata={},
    )

    with pytest.raises(TypeError, match="ResolvedSecret"):
        copy_resolved_secret(secret)


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


def test_secret_redactor_exposes_whether_it_has_values() -> None:
    assert SecretRedactor().has_values is False
    assert SecretRedactor("token").has_values is True


def test_secret_redactor_rejects_non_json_values() -> None:
    redactor = SecretRedactor(["secret"])

    with pytest.raises(ValueError, match="JSON-compatible"):
        redactor.redact_json(object())


def test_secret_redactor_rejects_blank_secrets() -> None:
    with pytest.raises(ValueError, match="cannot be blank"):
        SecretRedactor([" "])

    with pytest.raises(ValueError, match="cannot be blank"):
        SecretRedactor().with_secret(SecretStr(" "))


class _StubVault(Vault):
    """Configurable vault: records calls, optionally raises a non-SecretNotFound error."""

    def __init__(
        self, secrets: dict[str, str] | None = None, *, error: Exception | None = None
    ) -> None:
        self._secrets = dict(secrets or {})
        self._error = error
        self.get_calls: list[tuple[str, dict | None]] = []
        self.resolve_calls: list[tuple[str, dict | None]] = []

    async def get(self, name: str, *, scope: dict | None = None) -> SecretRef:
        self.get_calls.append((name, scope))
        if self._error is not None:
            raise self._error
        if name not in self._secrets:
            raise SecretNotFound(f"Secret not found: {name}")
        return SecretRef(name=name, handle=f"stub:{name}")

    async def resolve(self, ref: SecretRef, *, scope: dict | None = None) -> ResolvedSecret:
        self.resolve_calls.append((ref.name, scope))
        if self._error is not None:
            raise self._error
        if ref.name not in self._secrets:
            raise SecretNotFound(f"Secret not found: {ref.name}")
        return ResolvedSecret(name=ref.name, value=SecretStr(self._secrets[ref.name]))


# --- ChainVault -----------------------------------------------------------------------


def test_chain_vault_first_success_wins_and_short_circuits() -> None:
    first = _StubVault({"token": "one"})
    second = _StubVault({"token": "two"})
    chain = ChainVault(first, second)

    ref = asyncio.run(chain.get("token"))
    resolved = asyncio.run(chain.resolve(ref))

    assert resolved.value.get_secret_value() == "one"
    assert second.get_calls == [] and second.resolve_calls == []  # short-circuited


def test_chain_vault_falls_through_to_next_on_secret_not_found() -> None:
    first = _StubVault({})  # knows nothing
    second = _StubVault({"token": "two"})
    chain = ChainVault(first, second)

    resolved = asyncio.run(chain.resolve(SecretRef(name="token")))

    assert resolved.value.get_secret_value() == "two"
    assert first.resolve_calls == [("token", None)]  # was tried first


def test_chain_vault_raises_when_no_vault_resolves() -> None:
    chain = ChainVault(_StubVault({}), _StubVault({}))
    with pytest.raises(SecretNotFound, match="No vault could resolve"):
        asyncio.run(chain.get("missing"))
    with pytest.raises(SecretNotFound, match="No vault could resolve"):
        asyncio.run(chain.resolve(SecretRef(name="missing")))


def test_chain_vault_propagates_non_secret_not_found_errors() -> None:
    # A real failure (e.g. a network error) must NOT be swallowed by a later vault.
    failing = _StubVault({}, error=VaultError("nango down"))
    backup = _StubVault({"token": "two"})
    chain = ChainVault(failing, backup)

    with pytest.raises(VaultError, match="nango down"):
        asyncio.run(chain.resolve(SecretRef(name="token")))
    assert backup.resolve_calls == []  # not reached — the error stopped the chain

    with pytest.raises(VaultError, match="nango down"):
        asyncio.run(chain.get("token"))
    assert backup.get_calls == []  # get honors the same propagation contract


def test_chain_vault_passes_scope_through() -> None:
    stub = _StubVault({"token": "one"})
    chain = ChainVault(stub)
    asyncio.run(chain.get("token", scope={"tenant": "t1"}))
    assert stub.get_calls == [("token", {"tenant": "t1"})]


def test_chain_vault_validates_construction() -> None:
    with pytest.raises(ValueError, match="at least one vault"):
        ChainVault()
    with pytest.raises(TypeError, match="Vault instances"):
        ChainVault("not-a-vault")  # type: ignore[arg-type]


# --- RoutedVault ----------------------------------------------------------------------


def test_routed_vault_routes_by_name_with_fallback() -> None:
    dynamic = _StubVault({"gmail": "oauth"})
    static = _StubVault({"openai_key": "sk-static"})
    routed = RoutedVault(routes={"gmail": dynamic}, fallback=static)

    routed_result = asyncio.run(routed.resolve(SecretRef(name="gmail")))
    fallback_result = asyncio.run(routed.resolve(SecretRef(name="openai_key")))

    assert routed_result.value.get_secret_value() == "oauth"
    assert fallback_result.value.get_secret_value() == "sk-static"
    assert dynamic.resolve_calls == [("gmail", None)]  # static never hit for gmail
    assert static.resolve_calls == [("openai_key", None)]


def test_routed_vault_get_routes_by_name() -> None:
    dynamic = _StubVault({"gmail": "oauth"})
    static = _StubVault({"openai_key": "sk-static"})
    routed = RoutedVault(routes={"gmail": dynamic}, fallback=static)

    ref = asyncio.run(routed.get("gmail"))
    assert ref.handle == "stub:gmail"
    assert static.get_calls == []


def test_routed_vault_unrouted_without_fallback_raises_without_calling_vaults() -> None:
    dynamic = _StubVault({"gmail": "oauth"})
    routed = RoutedVault(routes={"gmail": dynamic})
    with pytest.raises(SecretNotFound, match="No vault configured"):
        asyncio.run(routed.get("slack"))
    assert dynamic.get_calls == []  # never called for an unrouted name


def test_routed_vault_passes_scope_through() -> None:
    dynamic = _StubVault({"gmail": "oauth"})
    routed = RoutedVault(routes={"gmail": dynamic})
    asyncio.run(routed.resolve(SecretRef(name="gmail"), scope={"connection_id": "org_123"}))
    assert dynamic.resolve_calls == [("gmail", {"connection_id": "org_123"})]


def test_routed_vault_validates_construction() -> None:
    with pytest.raises(ValueError, match="route or a fallback"):
        RoutedVault({})
    with pytest.raises(TypeError, match="must be a mapping"):
        RoutedVault(["gmail"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be blank"):
        RoutedVault({" ": _StubVault({})})
    with pytest.raises(TypeError, match="Vault instances"):
        RoutedVault({"gmail": "nope"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="fallback must be a Vault"):
        RoutedVault({}, fallback="nope")  # type: ignore[arg-type]


def test_composites_work_with_real_vaults() -> None:
    # End-to-end against real StaticVault/LocalEnvVault, exercising the real SecretNotFound
    # fall-through, and usable as an Environment's single vault.
    static = StaticVault({"openai_key": "sk-real"})
    other = StaticVault({"anthropic_key": "ak-real"})

    chained = ChainVault(other, static)
    routed = RoutedVault(routes={"openai_key": static}, fallback=other)

    assert (
        asyncio.run(
            chained.resolve(asyncio.run(chained.get("openai_key")))
        ).value.get_secret_value()
        == "sk-real"
    )
    assert (
        asyncio.run(
            routed.resolve(asyncio.run(routed.get("anthropic_key")))
        ).value.get_secret_value()
        == "ak-real"
    )
    # composites are a drop-in Vault for an Environment
    Environment(EnvironmentSpec(name="prod"), vault=chained)
    Environment(EnvironmentSpec(name="prod"), vault=routed)


def test_chain_vault_validates_lookup_inputs() -> None:
    # ChainVault honors the same input contract as the rest of the vault family.
    chain = ChainVault(_StubVault({"token": "one"}))
    with pytest.raises(ValueError, match="cannot be blank"):
        asyncio.run(chain.get(" "))
    with pytest.raises(TypeError, match="SecretRef instances"):
        asyncio.run(chain.resolve("not-a-ref"))  # type: ignore[arg-type]


def test_routed_vault_validates_lookup_inputs() -> None:
    # RoutedVault honors the same input contract as the rest of the vault family.
    routed = RoutedVault(routes={"token": _StubVault({"token": "one"})})
    with pytest.raises(ValueError, match="cannot be blank"):
        asyncio.run(routed.get(" "))
    with pytest.raises(TypeError, match="SecretRef instances"):
        asyncio.run(routed.resolve("not-a-ref"))  # type: ignore[arg-type]


def test_routed_vault_route_wins_over_fallback() -> None:
    # A name present in BOTH a route and the fallback resolves via the route; fallback untouched.
    routed_vault = _StubVault({"token": "routed"})
    fallback = _StubVault({"token": "fallback"})
    routed = RoutedVault(routes={"token": routed_vault}, fallback=fallback)

    resolved = asyncio.run(routed.resolve(SecretRef(name="token")))

    assert resolved.value.get_secret_value() == "routed"
    assert fallback.resolve_calls == []


def test_routed_vault_propagates_non_secret_not_found_errors() -> None:
    # A routed vault's real error propagates; it does NOT fall through to the fallback.
    failing = _StubVault({}, error=VaultError("nango down"))
    fallback = _StubVault({"gmail": "static"})
    routed = RoutedVault(routes={"gmail": failing}, fallback=fallback)

    with pytest.raises(VaultError, match="nango down"):
        asyncio.run(routed.resolve(SecretRef(name="gmail")))
    assert fallback.resolve_calls == []


def test_composites_nest() -> None:
    # Composites are themselves Vaults, so they nest: a miss in the inner composite falls
    # through to the outer chain.
    inner = RoutedVault(routes={"gmail": _StubVault({"gmail": "oauth"})})
    outer = ChainVault(inner, _StubVault({"openai_key": "sk-static"}))

    gmail = asyncio.run(outer.resolve(SecretRef(name="gmail")))
    openai = asyncio.run(outer.resolve(SecretRef(name="openai_key")))

    assert gmail.value.get_secret_value() == "oauth"
    assert openai.value.get_secret_value() == "sk-static"
