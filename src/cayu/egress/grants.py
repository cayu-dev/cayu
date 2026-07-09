from __future__ import annotations

import asyncio
import secrets
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.egress.credential_kinds import (
    credential_kind_descriptor,
    validate_presented_value,
    virtual_credential_entropy_bytes,
)
from cayu.egress.errors import VirtualCredentialError
from cayu.vaults import SecretRef, copy_secret_ref


def _utcnow() -> datetime:
    return datetime.now(UTC)


class VirtualCredentialGrant(BaseModel):
    """A per-session binding of a virtual credential to a real secret reference.

    The grant is what the sandbox is allowed to *present*; it carries a
    ``SecretRef`` (a pointer), never a resolved secret value. Only the broker,
    outside the sandbox, exchanges the presented value for the real secret. A
    grant is therefore safe to log, store, and hand to the sandbox.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    grant_id: str
    session_id: str
    env_name: str
    presented_value: str
    secret: SecretRef
    destination: str
    credential_kind: str
    policy_name: str | None = None
    created_at: datetime
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("grant_id", "session_id", "env_name", "credential_kind", "presented_value")
    @classmethod
    def validate_nonblank(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name).lower()

    @field_validator("policy_name")
    @classmethod
    def validate_policy_name(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("secret")
    @classmethod
    def copy_ref(cls, value: SecretRef) -> SecretRef:
        return copy_secret_ref(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at is not None and now >= self.expires_at


class VirtualCredentialRegistry:
    """Mints, binds, and revokes per-session virtual credentials.

    The registry holds **no real secret material** — only grants, which carry
    ``SecretRef`` pointers. It is the broker's source of truth for whether a
    presented value is currently a live, unexpired, unrevoked credential.
    """

    def __init__(self, *, clock: Callable[[], datetime] = _utcnow) -> None:
        self._clock = clock
        self._by_value: dict[str, VirtualCredentialGrant] = {}
        self._revoked_ids: set[str] = set()
        self._active_counts: dict[str, int] = {}
        self._inactive_waiters: dict[str, list[asyncio.Future[None]]] = {}

    def mint(
        self,
        *,
        session_id: str,
        env_name: str,
        secret: SecretRef,
        destination: str,
        credential_kind: str,
        policy_name: str | None = None,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
        presented_value: str | None = None,
    ) -> VirtualCredentialGrant:
        """Create and register a new virtual credential grant."""
        now = self._clock()
        value = presented_value or self._generate_value(credential_kind)
        if presented_value is not None:
            validate_presented_value(credential_kind, value)
        if value in self._by_value:
            raise ValueError("Virtual credential value collision; retry minting.")
        expires_at = self._expiry(now, ttl_seconds)
        grant = VirtualCredentialGrant(
            grant_id=uuid4().hex,
            session_id=session_id,
            env_name=env_name,
            presented_value=value,
            secret=secret,
            destination=destination,
            credential_kind=credential_kind,
            policy_name=policy_name,
            created_at=now,
            expires_at=expires_at,
            metadata=metadata or {},
        )
        self._by_value[grant.presented_value] = grant
        return grant

    def bind(self, grant: VirtualCredentialGrant) -> VirtualCredentialGrant:
        """Register an externally-constructed grant."""
        if not isinstance(grant, VirtualCredentialGrant):
            raise TypeError("Grants must be VirtualCredentialGrant instances.")
        validate_presented_value(grant.credential_kind, grant.presented_value)
        if grant.grant_id in self._revoked_ids:
            raise VirtualCredentialError("Virtual credential grant has been revoked.")
        if grant.presented_value in self._by_value:
            raise ValueError("A grant with this presented value is already registered.")
        self._by_value[grant.presented_value] = grant
        return grant

    def acquire(self, presented_value: str) -> VirtualCredentialLease:
        """Acquire a live grant for one broker request.

        The lease lets teardown mark a grant revoked while an already-started
        request is still resolving, and lets the broker re-check liveness before
        the real credential is sent upstream.
        """
        grant = self.lookup(presented_value)
        self._active_counts[grant.grant_id] = self._active_counts.get(grant.grant_id, 0) + 1
        return VirtualCredentialLease(self, grant)

    def lookup(
        self,
        presented_value: str,
        *,
        now: datetime | None = None,
    ) -> VirtualCredentialGrant:
        """Return the live grant for a presented value or raise.

        Raises ``VirtualCredentialError`` for unknown, revoked, or expired
        values. The broker calls this before any policy check or vault resolve.
        """
        now = now or self._clock()
        grant = self._by_value.get(presented_value)
        if grant is None:
            raise VirtualCredentialError("Unknown or revoked virtual credential.")
        if grant.grant_id in self._revoked_ids:
            self._by_value.pop(presented_value, None)
            raise VirtualCredentialError("Unknown or revoked virtual credential.")
        if grant.is_expired(now):
            raise VirtualCredentialError("Virtual credential has expired.")
        return grant

    def revoke(self, presented_value: str) -> bool:
        """Revoke by presented value. Returns whether a grant was removed."""
        grant = self._by_value.pop(presented_value, None)
        if grant is None:
            return False
        self._revoked_ids.add(grant.grant_id)
        return True

    async def revoke_and_wait(self, presented_value: str) -> bool:
        """Revoke by presented value and wait for active broker leases to drain."""
        return bool(await self.revoke_values_and_wait((presented_value,)))

    async def revoke_values_and_wait(self, presented_values: Sequence[str]) -> int:
        """Revoke every presented value, then wait for all active leases to drain.

        Revocation is intentionally two-phase: first mark every matching grant
        revoked without awaiting, then wait for in-flight broker leases. That
        prevents teardown of a multi-credential session from leaving later
        credentials live while an earlier credential's active request drains.
        """
        grant_ids: list[str] = []
        count = 0
        seen_values: set[str] = set()
        for value in presented_values:
            if value in seen_values:
                continue
            seen_values.add(value)
            grant = self._by_value.get(value)
            if self.revoke(value):
                count += 1
                if grant is not None:
                    grant_ids.append(grant.grant_id)
        await self.wait_for_inactive_grants(grant_ids)
        return count

    async def wait_for_inactive_grants(self, grant_ids: Sequence[str]) -> None:
        """Wait until every listed grant has no active broker request leases."""
        for grant_id in grant_ids:
            await self._wait_for_inactive(grant_id)

    def revoke_session(self, session_id: str) -> int:
        """Revoke every grant bound to ``session_id``. Returns the count."""
        values = [
            grant.presented_value
            for grant in self._by_value.values()
            if grant.session_id == session_id
        ]
        for value in values:
            self.revoke(value)
        return len(values)

    async def revoke_session_and_wait(self, session_id: str) -> int:
        """Revoke every session grant and wait for active broker leases to drain."""
        values = [
            grant.presented_value
            for grant in self._by_value.values()
            if grant.session_id == session_id
        ]
        return await self.revoke_values_and_wait(values)

    def was_revoked(self, grant_id: str) -> bool:
        return grant_id in self._revoked_ids

    def active_grants(self) -> tuple[VirtualCredentialGrant, ...]:
        now = self._clock()
        return tuple(
            grant
            for grant in self._by_value.values()
            if grant.grant_id not in self._revoked_ids and not grant.is_expired(now)
        )

    def _generate_value(self, credential_kind: str) -> str:
        descriptor = credential_kind_descriptor(credential_kind)
        token = secrets.token_hex(virtual_credential_entropy_bytes())
        return f"{descriptor.virtual_prefix}{token}"

    def _assert_lease_active(self, grant: VirtualCredentialGrant) -> None:
        if grant.grant_id in self._revoked_ids:
            raise VirtualCredentialError("Virtual credential grant has been revoked.")
        if grant.is_expired(self._clock()):
            raise VirtualCredentialError("Virtual credential has expired.")

    def _release_lease(self, grant: VirtualCredentialGrant) -> None:
        current = self._active_counts.get(grant.grant_id, 0)
        if current <= 1:
            self._active_counts.pop(grant.grant_id, None)
            waiters = self._inactive_waiters.pop(grant.grant_id, [])
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_result(None)
            return
        self._active_counts[grant.grant_id] = current - 1

    async def _wait_for_inactive(self, grant_id: str) -> None:
        while self._active_counts.get(grant_id, 0) > 0:
            future = asyncio.get_running_loop().create_future()
            self._inactive_waiters.setdefault(grant_id, []).append(future)
            await future

    @staticmethod
    def _expiry(now: datetime, ttl_seconds: float | None) -> datetime | None:
        if ttl_seconds is None:
            return None
        if ttl_seconds <= 0:
            raise ValueError("`ttl_seconds` must be positive.")
        return now + timedelta(seconds=ttl_seconds)


class VirtualCredentialLease:
    """An active broker request's lease on a live virtual credential grant."""

    def __init__(
        self,
        registry: VirtualCredentialRegistry,
        grant: VirtualCredentialGrant,
    ) -> None:
        self._registry = registry
        self.grant = grant
        self._closed = False

    def ensure_active(self) -> None:
        if self._closed:
            raise VirtualCredentialError("Virtual credential lease is closed.")
        self._registry._assert_lease_active(self.grant)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._registry._release_lease(self.grant)

    def __enter__(self) -> VirtualCredentialLease:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
