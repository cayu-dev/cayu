from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu.egress import (
    VirtualCredentialError,
    VirtualCredentialGrant,
    VirtualCredentialRegistry,
)
from cayu.vaults import SecretRef

_MANUAL_STRIPE_VIRTUAL = "sk_test_cayu_vc_" + ("a" * 48)


class _Clock:
    """Manually advanced clock for deterministic expiry tests."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def _registry(start: datetime | None = None) -> tuple[VirtualCredentialRegistry, _Clock]:
    clock = _Clock(start or datetime(2026, 7, 6, tzinfo=UTC))
    return VirtualCredentialRegistry(clock=clock), clock


def test_mint_produces_stripe_shaped_virtual_value() -> None:
    registry, _ = _registry()

    grant = registry.mint(
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
        policy_name="provider-example",
    )

    assert grant.presented_value.startswith("sk_test_cayu_vc_")
    assert grant.env_name == "STRIPE_SECRET_KEY"
    assert grant.destination == "api.stripe.com"
    assert grant.policy_name == "provider-example"


def test_grant_holds_only_a_secret_ref_never_a_value() -> None:
    registry, _ = _registry()

    grant = registry.mint(
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
    )

    assert isinstance(grant.secret, SecretRef)
    # The real secret value must be absent everywhere in the grant's serialized form.
    dumped = grant.model_dump_json()
    assert "sk_test_cayu_vc_" in grant.presented_value
    assert "get_secret_value" not in dumped
    # A grant carries a pointer (name) only; there is no field holding a raw value.
    assert "value" not in grant.model_dump()


def test_generic_kind_uses_generic_prefix() -> None:
    registry, _ = _registry()

    grant = registry.mint(
        session_id="sess_1",
        env_name="API_KEY",
        secret=SecretRef(name="generic_key"),
        destination="api.example.com",
        credential_kind="opaque_bearer",
    )

    assert grant.presented_value.startswith("cayu_vc_")


def test_opaque_token_kind_uses_generic_prefix() -> None:
    registry, _ = _registry()

    grant = registry.mint(
        session_id="sess_1",
        env_name="GH_TOKEN",
        secret=SecretRef(name="github_token"),
        destination="api.github.com",
        credential_kind="opaque_token",
    )

    assert grant.presented_value.startswith("cayu_vc_")


def test_unsupported_kind_is_rejected() -> None:
    registry, _ = _registry()

    with pytest.raises(ValueError, match="Unsupported credential kind"):
        registry.mint(
            session_id="sess_1",
            env_name="API_KEY",
            secret=SecretRef(name="generic_key"),
            destination="api.example.com",
            credential_kind="mystery_kind",
        )


def test_lookup_returns_live_grant() -> None:
    registry, _ = _registry()
    grant = registry.mint(
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
    )

    found = registry.lookup(grant.presented_value)

    assert found.grant_id == grant.grant_id


def test_lookup_unknown_value_raises() -> None:
    registry, _ = _registry()

    with pytest.raises(VirtualCredentialError):
        registry.lookup("sk_test_cayu_vc_does_not_exist")


def test_expired_credential_is_rejected() -> None:
    registry, clock = _registry()
    grant = registry.mint(
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
        ttl_seconds=60,
    )

    registry.lookup(grant.presented_value)  # still valid
    clock.advance(61)

    with pytest.raises(VirtualCredentialError):
        registry.lookup(grant.presented_value)


def test_active_grants_excludes_expired_grants() -> None:
    registry, clock = _registry()
    grant = registry.mint(
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
        ttl_seconds=60,
    )

    assert registry.active_grants() == (grant,)
    clock.advance(61)

    assert registry.active_grants() == ()


def test_revoke_rejects_further_use() -> None:
    registry, _ = _registry()
    grant = registry.mint(
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
    )

    assert registry.revoke(grant.presented_value) is True
    assert registry.was_revoked(grant.grant_id) is True
    with pytest.raises(VirtualCredentialError):
        registry.lookup(grant.presented_value)


def test_revoke_and_wait_marks_active_lease_revoked_before_waiting() -> None:
    async def run() -> None:
        registry, _ = _registry()
        grant = registry.mint(
            session_id="sess_1",
            env_name="STRIPE_SECRET_KEY",
            secret=SecretRef(name="stripe_test_key"),
            destination="api.stripe.com",
            credential_kind="stripe_bearer",
        )
        lease = registry.acquire(grant.presented_value)

        revoke_task = asyncio.create_task(registry.revoke_and_wait(grant.presented_value))
        await asyncio.sleep(0)

        assert revoke_task.done() is False
        with pytest.raises(VirtualCredentialError, match="revoked"):
            lease.ensure_active()

        lease.close()
        assert await revoke_task is True

    asyncio.run(run())


def test_revoke_session_and_wait_revokes_all_grants_before_waiting() -> None:
    async def run() -> None:
        registry, _ = _registry()
        first = registry.mint(
            session_id="sess_1",
            env_name="STRIPE_SECRET_KEY",
            secret=SecretRef(name="stripe_test_key"),
            destination="api.stripe.com",
            credential_kind="stripe_bearer",
        )
        second = registry.mint(
            session_id="sess_1",
            env_name="OTHER_KEY",
            secret=SecretRef(name="other_key"),
            destination="api.example.com",
            credential_kind="opaque_bearer",
        )
        first_lease = registry.acquire(first.presented_value)

        revoke_task = asyncio.create_task(registry.revoke_session_and_wait("sess_1"))
        await asyncio.sleep(0)

        assert revoke_task.done() is False
        with pytest.raises(VirtualCredentialError, match="revoked"):
            first_lease.ensure_active()
        with pytest.raises(VirtualCredentialError, match="revoked"):
            registry.lookup(second.presented_value)

        first_lease.close()
        assert await revoke_task == 2

    asyncio.run(run())


def test_revoked_grant_cannot_be_bound_again() -> None:
    registry, _ = _registry()
    grant = registry.mint(
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
    )

    assert registry.revoke(grant.presented_value) is True
    with pytest.raises(VirtualCredentialError, match="revoked"):
        registry.bind(grant)


def test_lookup_rejects_grant_whose_id_was_revoked() -> None:
    registry, _ = _registry()
    grant = registry.mint(
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
    )

    assert registry.revoke(grant.presented_value) is True
    # Simulate a stale persisted binding being reintroduced below the public API:
    # lookup still treats the revoked grant id as terminal.
    registry._by_value[grant.presented_value] = grant

    with pytest.raises(VirtualCredentialError, match="revoked"):
        registry.lookup(grant.presented_value)
    assert registry.active_grants() == ()


def test_revoke_session_revokes_all_session_grants() -> None:
    registry, _ = _registry()
    first = registry.mint(
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
    )
    second = registry.mint(
        session_id="sess_1",
        env_name="OTHER_KEY",
        secret=SecretRef(name="other_key"),
        destination="api.example.com",
        credential_kind="opaque_bearer",
    )
    survivor = registry.mint(
        session_id="sess_2",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
    )

    count = registry.revoke_session("sess_1")

    assert count == 2
    with pytest.raises(VirtualCredentialError):
        registry.lookup(first.presented_value)
    with pytest.raises(VirtualCredentialError):
        registry.lookup(second.presented_value)
    assert registry.lookup(survivor.presented_value).grant_id == survivor.grant_id


def test_bind_registers_external_grant_and_rejects_duplicates() -> None:
    registry, clock = _registry()
    grant = VirtualCredentialGrant(
        grant_id="grant_1",
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        presented_value=_MANUAL_STRIPE_VIRTUAL,
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
        created_at=clock.now,
    )

    registry.bind(grant)
    assert registry.lookup(_MANUAL_STRIPE_VIRTUAL).grant_id == "grant_1"
    with pytest.raises(ValueError, match="already registered"):
        registry.bind(grant)


def test_bind_accepts_virtual_credential_grant_subclasses() -> None:
    class CustomGrant(VirtualCredentialGrant):
        pass

    registry, clock = _registry()
    grant = CustomGrant(
        grant_id="grant_1",
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        presented_value=_MANUAL_STRIPE_VIRTUAL,
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
        created_at=clock.now,
    )

    registry.bind(grant)

    assert registry.lookup(_MANUAL_STRIPE_VIRTUAL).grant_id == "grant_1"


def test_bind_rejects_external_grant_with_raw_provider_secret_presented_value() -> None:
    registry, clock = _registry()
    grant = VirtualCredentialGrant(
        grant_id="grant_1",
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        presented_value="sk_test_51ActuallyRealProviderSecret",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
        created_at=clock.now,
    )

    with pytest.raises(ValueError, match="raw provider credential"):
        registry.bind(grant)


def test_mint_rejects_caller_supplied_raw_provider_secret_as_presented_value() -> None:
    registry, _ = _registry()

    with pytest.raises(ValueError, match="raw provider credential"):
        registry.mint(
            session_id="sess_1",
            env_name="STRIPE_SECRET_KEY",
            secret=SecretRef(name="stripe_test_key"),
            destination="api.stripe.com",
            credential_kind="stripe_bearer",
            presented_value="sk_test_51ActuallyRealProviderSecret",
        )


def test_mint_accepts_caller_supplied_cayu_virtual_presented_value() -> None:
    registry, _ = _registry()

    grant = registry.mint(
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
        presented_value=_MANUAL_STRIPE_VIRTUAL,
    )

    assert grant.presented_value == _MANUAL_STRIPE_VIRTUAL


def test_grant_is_immutable() -> None:
    registry, _ = _registry()
    grant = registry.mint(
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
    )

    with pytest.raises(Exception):
        grant.presented_value = "sk_test_cayu_vc_tampered"  # type: ignore[misc]


def test_zero_ttl_is_rejected() -> None:
    registry, _ = _registry()

    with pytest.raises(ValueError, match="ttl_seconds"):
        registry.mint(
            session_id="sess_1",
            env_name="STRIPE_SECRET_KEY",
            secret=SecretRef(name="stripe_test_key"),
            destination="api.stripe.com",
            credential_kind="stripe_bearer",
            ttl_seconds=0,
        )
