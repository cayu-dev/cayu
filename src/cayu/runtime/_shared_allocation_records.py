"""Durable records of the allocations a scope's sessions share.

A shared allocation is recorded in its owner's checkpoint, under one key per
allocation scope and environment. The owner is the workflow run (the workflow
journal anchor) for workflow steps, or the agent session whose subagents share
its sandbox, when the session store outlives the process. The record is written before the factory is called, so the
allocation's identity (the ``session_id`` its request names) is durable before
the allocation exists, and a resumed run in another process asks for the same
one. It then records what the factory and binding returned: the factory's
reconnect metadata, and the binding's private recovery state, which lets
another process rebuild the bound workspace without copying the source in
again.

Every record carries a claim: the owner's attempt that holds it (a workflow
attempt, or one run epoch of an agent session), and a generation that goes up
with every change of owner. Every write is a
compare-and-set inside one ``transform_checkpoint`` call, so two processes
never both believe they own one record, and a superseded attempt that is still
running finds out at its next write that it lost the allocation instead of
disposing of it.

A record that can't be reconnected (its factory published no reconnect
metadata, or its binding can't be rebuilt) is moved to ``orphans`` when another
attempt takes the scope over, and the new attempt gets a fresh allocation.
``recover_incomplete_sessions`` disposes of orphans, and of records whose
owner has ended.

A record whose factory creates recoverably (it declares an
``allocation_scope``) also holds that creation's allocation intent, the same
``EnvironmentAllocationRecord`` an ordinary session keeps in its own
checkpoint. ``cayu.runtime._shared_allocation_intents`` moves it through its
states with the same compare-and-set, so a crash during the factory's create
leaves something the next attempt can recover and recovery can reap.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
from collections.abc import AsyncIterator, Callable, Collection
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_metadata,
)
from cayu.runners._cleanup import preserve_sandbox_on_command_cleanup
from cayu.runtime._environment_allocation import EnvironmentAllocationRecord

if TYPE_CHECKING:
    from cayu.sessions.records import Session
    from cayu.sessions.store import SessionStore

SHARED_ALLOCATIONS_CHECKPOINT_KEY = "shared_environment_allocations"
#: How long a disposal lease (recovery's reaping, or a hand release) lasts
#: without renewal, and how often its holder renews it while it works.
DISPOSAL_LEASE_SECONDS = 300.0
DISPOSAL_RENEW_SECONDS = 60.0
_SCHEMA_VERSION = 1
_LOGGER = logging.getLogger(__name__)

RecordState = Literal["creating", "created", "live", "reaping"]
_STATES = frozenset({"creating", "created", "live", "reaping"})
# How many step sessions a record remembers having copied back after their step.
MAX_COPIED_BACK_SESSIONS = 64


_IDENTITY_SEPARATOR = ".shared-"


def shared_allocation_session_id(owner_session_id: str) -> str:
    """Return a fresh identity for one shared allocation, named after its owner."""

    return f"{owner_session_id}{_IDENTITY_SEPARATOR}{secrets.token_hex(8)}"


def shared_allocation_owner(allocation_session_id: str) -> str | None:
    """The owner session an allocation identity names, or None when it isn't one."""

    owner, separator, suffix = allocation_session_id.rpartition(_IDENTITY_SEPARATOR)
    if not owner or not separator or len(suffix) != 16:
        return None
    if any(character not in "0123456789abcdef" for character in suffix):
        return None
    return owner


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SharedAllocationRecord:
    """What one process knows durably about one shared allocation."""

    allocation_session_id: str
    scope: str
    scope_id: str
    owner_session_id: str
    environment_name: str
    agent_name: str
    attempt_id: str
    claim_generation: int
    state: RecordState
    attempt_sequence: int = 0
    binding_required: bool = True
    disposal_state: dict[str, Any] | None = None
    reconnect_metadata: dict[str, Any] = field(default_factory=dict)
    binding_state: dict[str, Any] | None = None
    result_metadata: dict[str, Any] = field(default_factory=dict)
    # The recoverable creation's ``EnvironmentAllocationRecord`` payload, for a
    # factory that declares an allocation scope; None otherwise.
    allocation_intent: dict[str, Any] | None = None
    # The latest step sessions whose copy-back after their step completed,
    # written together with the binding state that copy-back left.
    copied_back: tuple[str, ...] = ()
    # Step sessions whose copy-back after their step failed, so the allocation
    # holds work the source hasn't got; a release that wouldn't copy back
    # refuses to dispose of it.
    pending_copy_back: tuple[str, ...] = ()
    # When this identity was first recorded; None for records written before
    # the field existed.
    created_at: str | None = None
    # A hand release in progress: {"token": ..., "expires_at": <ISO time>}. While
    # it hasn't expired, nothing else releases or reaps the allocation.
    release_lease: dict[str, str] | None = None
    # The owner's interrupted release left the allocation in place (a suspended
    # Lambda MicroVM) for its resume, with the binding state that release left.
    # A resume that can't reconnect to it then starts afresh and leaves this
    # record as an orphan to reap, rather than failing on it for good.
    kept_for_resume: bool = False

    def release_leased(self, now: datetime) -> bool:
        """Whether a hand release holds this record and its lease hasn't expired."""

        lease = self.release_lease
        if lease is None:
            return False
        try:
            return datetime.fromisoformat(lease["expires_at"]) > now
        except (KeyError, TypeError, ValueError):
            return False

    @property
    def reconnectable(self) -> bool:
        """Whether another process can attach to this allocation without copying in again.

        A created allocation was never bound, so reconnecting it and binding it
        loses nothing; a live one needs the binding's recovery state too.
        """

        if not self.reconnect_metadata:
            return False
        if self.state == "created":
            return True
        return self.disposal_state is not None or (
            self.state == "live" and (not self.binding_required or self.binding_state is not None)
        )

    def payload(self) -> dict[str, Any]:
        intent = (
            {}
            if self.allocation_intent is None
            else {
                "allocation_intent": copy_durable_json_object(
                    self.allocation_intent, "allocation_intent"
                )
            }
        )
        return {
            **intent,
            "version": _SCHEMA_VERSION,
            "allocation_session_id": self.allocation_session_id,
            "scope": self.scope,
            "scope_id": self.scope_id,
            "owner_session_id": self.owner_session_id,
            "environment_name": self.environment_name,
            "agent_name": self.agent_name,
            "attempt_id": self.attempt_id,
            "claim_generation": self.claim_generation,
            "state": self.state,
            "attempt_sequence": self.attempt_sequence,
            "binding_required": self.binding_required,
            "disposal_state": self.disposal_state,
            "reconnect_metadata": copy_durable_metadata(
                self.reconnect_metadata, "reconnect_metadata"
            ),
            "binding_state": (
                None
                if self.binding_state is None
                else copy_durable_json_object(self.binding_state, "binding_state")
            ),
            "result_metadata": copy_durable_metadata(self.result_metadata, "result_metadata"),
            **({"copied_back": list(self.copied_back)} if self.copied_back else {}),
            **(
                {"pending_copy_back": list(self.pending_copy_back)}
                if self.pending_copy_back
                else {}
            ),
            "created_at": self.created_at,
            "release_lease": None if self.release_lease is None else dict(self.release_lease),
            **({"kept_for_resume": True} if self.kept_for_resume else {}),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> SharedAllocationRecord:
        if type(payload) is not dict or payload.get("version") != _SCHEMA_VERSION:
            raise ValueError("Shared allocation record has an unsupported format.")
        texts = (
            "allocation_session_id",
            "scope",
            "scope_id",
            "owner_session_id",
            "environment_name",
            "agent_name",
            "attempt_id",
        )
        for name in texts:
            if type(payload.get(name)) is not str or not payload[name]:
                raise ValueError(f"Shared allocation record field {name!r} is malformed.")
        generation = payload.get("claim_generation")
        state = payload.get("state")
        if type(generation) is not int or generation < 1 or state not in _STATES:
            raise ValueError("Shared allocation record claim is malformed.")
        attempt_sequence = payload.get("attempt_sequence", 0)
        binding_required = payload.get("binding_required", True)
        disposal_state = payload.get("disposal_state")
        if type(attempt_sequence) is not int or attempt_sequence < 0:
            raise ValueError("Shared allocation attempt sequence is malformed.")
        if type(binding_required) is not bool:
            raise ValueError("Shared allocation binding requirement is malformed.")
        if disposal_state is not None:
            disposal_state = copy_durable_json_object(disposal_state, "disposal_state")
        binding_state = payload.get("binding_state")
        if binding_state is not None and type(binding_state) is not dict:
            raise ValueError("Shared allocation record binding state is malformed.")
        allocation_intent = payload.get("allocation_intent")
        if allocation_intent is not None:
            # Parsed for validation only; the record keeps the payload.
            EnvironmentAllocationRecord.from_payload(allocation_intent)
            allocation_intent = copy_durable_json_object(allocation_intent, "allocation_intent")
        copied_back = _session_ids(payload, "copied_back")
        pending_copy_back = _session_ids(payload, "pending_copy_back")
        release_lease = payload.get("release_lease")
        if release_lease is not None and (
            type(release_lease) is not dict
            or set(release_lease) != {"token", "expires_at"}
            or any(type(value) is not str for value in release_lease.values())
        ):
            raise ValueError("Shared allocation record release lease is malformed.")
        created_at = payload.get("created_at")
        if created_at is not None and type(created_at) is not str:
            raise ValueError("Shared allocation record creation time is malformed.")
        kept_for_resume = payload.get("kept_for_resume", False)
        if type(kept_for_resume) is not bool:
            raise ValueError("Shared allocation record resume flag is malformed.")
        return cls(
            allocation_session_id=payload["allocation_session_id"],
            scope=payload["scope"],
            scope_id=payload["scope_id"],
            owner_session_id=payload["owner_session_id"],
            environment_name=payload["environment_name"],
            agent_name=payload["agent_name"],
            attempt_id=payload["attempt_id"],
            claim_generation=generation,
            state=state,
            attempt_sequence=attempt_sequence,
            binding_required=binding_required,
            disposal_state=disposal_state,
            reconnect_metadata=copy_durable_metadata(
                payload.get("reconnect_metadata") or {}, "reconnect_metadata"
            ),
            binding_state=(
                None
                if binding_state is None
                else copy_durable_json_object(binding_state, "binding_state")
            ),
            result_metadata=copy_durable_metadata(
                payload.get("result_metadata") or {}, "result_metadata"
            ),
            allocation_intent=allocation_intent,
            copied_back=copied_back,
            pending_copy_back=pending_copy_back,
            created_at=created_at,
            release_lease=None if release_lease is None else dict(release_lease),
            kept_for_resume=kept_for_resume,
        )

    def same_claim(self, other: SharedAllocationRecord | None) -> bool:
        return (
            other is not None
            and other.allocation_session_id == self.allocation_session_id
            and other.attempt_id == self.attempt_id
            and other.claim_generation == self.claim_generation
        )


def _session_ids(payload: dict[str, Any], name: str) -> tuple[str, ...]:
    value = payload.get(name, [])
    if (
        type(value) is not list
        or len(value) > MAX_COPIED_BACK_SESSIONS
        or any(type(item) is not str or not item for item in value)
    ):
        raise ValueError(f"Shared allocation record field {name!r} is malformed.")
    return tuple(value)


class DisposalHold:
    """The record a disposal lease holder holds, as its writes and renewals change it."""

    def __init__(self, record: SharedAllocationRecord, timeout: asyncio.Timeout) -> None:
        self.record = record
        self._timeout = timeout
        self._renewal_lock = asyncio.Lock()
        self._finished = False

    def expired(self) -> bool:
        deadline = self._timeout.when()
        assert deadline is not None
        return self._timeout.expired() or asyncio.get_running_loop().time() >= deadline

    def check_active(self) -> None:
        if self.expired():
            raise SharedAllocationClaimLost("The disposal lease was lost or expired.")

    async def remove_record(self, records: SharedAllocationRecords) -> None:
        # Renewal must not interpret our own committed deletion as lease loss
        # while the store is still returning its successful removal response.
        async with self._renewal_lock:
            self.check_active()
            await records.remove_reaped(self.record)
            self._finished = True


@contextlib.asynccontextmanager
async def keeping_disposal_lease(
    records: SharedAllocationRecords, record: SharedAllocationRecord
) -> AsyncIterator[DisposalHold]:
    """Renew the lease, cancelling the guarded work if ownership is lost or expires."""

    loop = asyncio.get_running_loop()

    def deadline(record: SharedAllocationRecord) -> float:
        lease = record.release_lease
        if lease is None:
            raise SharedAllocationClaimLost("The disposal lease is missing.")
        remaining = (
            datetime.fromisoformat(lease["expires_at"]) - datetime.now(UTC)
        ).total_seconds()
        return loop.time() + remaining

    expires_at = deadline(record)
    if expires_at <= loop.time():
        raise SharedAllocationClaimLost("The disposal lease has expired.")
    lease_timeout = asyncio.timeout_at(expires_at)
    hold = DisposalHold(record, lease_timeout)

    async def renew() -> None:
        while True:
            await asyncio.sleep(DISPOSAL_RENEW_SECONDS)
            async with hold._renewal_lock:
                if hold._finished:
                    return
                try:
                    renewed = await records.renew_release(
                        hold.record, lease_seconds=DISPOSAL_LEASE_SECONDS
                    )
                except Exception as error:
                    _LOGGER.warning(
                        "Renewing the disposal lease of shared allocation %s failed (%s).",
                        hold.record.allocation_session_id,
                        type(error).__name__,
                    )
                    # The deadline still cancels work if renewal fails or stalls.
                    continue
                if hold.expired():
                    # A late response must never revive authority after expiry.
                    return
                if renewed is None:
                    lease_timeout.reschedule(loop.time())
                    return
                hold.record = renewed
                lease_timeout.reschedule(deadline(renewed))

    preservation = preserve_sandbox_on_command_cleanup.set(True)
    try:
        async with lease_timeout:
            renewal = asyncio.ensure_future(renew())
            try:
                yield hold
            finally:
                renewal.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await renewal
    except TimeoutError as error:
        if not lease_timeout.expired():
            raise
        raise SharedAllocationClaimLost("The disposal lease was lost or expired.") from error
    finally:
        preserve_sandbox_on_command_cleanup.reset(preservation)
    if lease_timeout.expired():
        # Even a binding that swallowed cancellation cannot report success or
        # cause its caller to forget a still-owned allocation.
        raise SharedAllocationClaimLost("The disposal lease was lost or expired.")


def shared_allocation_record_id(scope: str, scope_id: str, environment_name: str) -> str:
    """The checkpoint key of one scope's record for one environment."""

    return sha256(
        canonical_durable_json_bytes(
            {"scope": scope, "scope_id": scope_id, "environment_name": environment_name},
            "shared_allocation_record_id",
        )
    ).hexdigest()[:32]


@dataclass(frozen=True)
class SharedAllocationSnapshot:
    """Every record one owner's checkpoint holds."""

    records: dict[str, SharedAllocationRecord]
    orphans: dict[str, SharedAllocationRecord]

    def all(self) -> tuple[SharedAllocationRecord, ...]:
        return (*self.records.values(), *self.orphans.values())


@dataclass(frozen=True)
class SharedAllocationClaim:
    """The record an attempt holds after claiming its scope, and what it replaced."""

    record: SharedAllocationRecord
    previous: SharedAllocationRecord | None
    orphaned_previous: bool


class SharedAllocationClaimLost(RuntimeError):
    """Raised when another attempt of the owner took a shared allocation over."""


def _read(checkpoint: dict[str, Any] | None) -> tuple[dict[str, Any], SharedAllocationSnapshot]:
    current = {} if checkpoint is None else checkpoint
    raw = current.get(SHARED_ALLOCATIONS_CHECKPOINT_KEY)
    if raw is None:
        return current, SharedAllocationSnapshot({}, {})
    if type(raw) is not dict:
        raise ValueError("Shared allocation records must be an object.")
    records = raw.get("records") or {}
    orphans = raw.get("orphans") or {}
    if type(records) is not dict or type(orphans) is not dict:
        raise ValueError("Shared allocation records are malformed.")
    return current, SharedAllocationSnapshot(
        {key: SharedAllocationRecord.from_payload(value) for key, value in records.items()},
        {key: SharedAllocationRecord.from_payload(value) for key, value in orphans.items()},
    )


def _write(current: dict[str, Any], snapshot: SharedAllocationSnapshot) -> dict[str, Any]:
    if not snapshot.records and not snapshot.orphans:
        current.pop(SHARED_ALLOCATIONS_CHECKPOINT_KEY, None)
        return current
    current[SHARED_ALLOCATIONS_CHECKPOINT_KEY] = {
        "records": {key: record.payload() for key, record in snapshot.records.items()},
        "orphans": {key: record.payload() for key, record in snapshot.orphans.items()},
    }
    return current


class SharedAllocationRecords:
    """Compare-and-set access to the shared allocation records of owner sessions."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    async def load(self, owner_session_id: str) -> SharedAllocationSnapshot | None:
        """Return the owner's records, or None when the owner session doesn't exist."""

        if await self._store.load(owner_session_id) is None:
            return None
        _, snapshot = _read(await self._store.load_checkpoint(owner_session_id))
        return snapshot

    async def _mutate(
        self,
        owner_session_id: str,
        change: Callable[[SharedAllocationSnapshot], tuple[bool, Any]],
    ) -> Any:
        """Apply ``change`` to the owner's records atomically and return its value.

        ``change`` edits the snapshot's dictionaries in place and returns
        ``(changed, value)``; nothing is written unless ``changed``.
        """

        outcome: list[Any] = []

        def transform(
            _session: Session, checkpoint: dict[str, Any] | None
        ) -> dict[str, Any] | None:
            outcome.clear()
            current, snapshot = _read(checkpoint)
            changed, value = change(snapshot)
            outcome.append(value)
            return _write(current, snapshot) if changed else None

        await self._store.transform_checkpoint(owner_session_id, transform)
        if not outcome:
            raise RuntimeError("Shared allocation record transform did not run.")
        return outcome[0]

    async def claim(
        self,
        *,
        record_id: str,
        scope: str,
        scope_id: str,
        owner_session_id: str,
        environment_name: str,
        agent_name: str,
        attempt_id: str,
        attempt_sequence: int,
        replace_unreachable: bool = True,
        keep_allocation_session_id: str | None = None,
        expected_generation: int | None = None,
    ) -> SharedAllocationClaim | None:
        """Claim the scope's record for ``attempt_id``, creating or taking it over.

        A record another attempt holds is taken over: kept when another process
        can attach to it (or it never got past creation, so a CREATE under the
        same identity adopts or replaces it), otherwise moved to the orphans
        and replaced by a fresh one. With ``replace_unreachable=False`` nothing
        new is created: the result is None when there is no record this
        attempt can reconnect to. With ``keep_allocation_session_id`` (a
        process that holds that allocation live) the record is taken over only
        while it still names that allocation, whatever it is, and the result
        is None otherwise.
        """

        def change(snapshot: SharedAllocationSnapshot) -> Any:
            current = snapshot.records.get(record_id)
            if current is not None and current.owner_session_id != owner_session_id:
                # A fork starts from a copy of its source's checkpoint; the
                # source's record names the source's allocation, never the fork's.
                del snapshot.records[record_id]
                current = None
            if current is not None and current.attempt_sequence > attempt_sequence:
                raise SharedAllocationClaimLost("A newer workflow attempt owns this allocation.")
            if expected_generation is not None and expected_generation != (
                0 if current is None else current.claim_generation
            ):
                # Claimed since the claimant last saw it, so possibly by an
                # attempt newer than the claimant's last fence check.
                raise SharedAllocationClaimLost(
                    "The record changed since this attempt checked that it is the newest."
                )
            reachable = current is not None and current.state != "reaping"
            if keep_allocation_session_id is not None:
                if (
                    current is None
                    or not reachable
                    or current.allocation_session_id != keep_allocation_session_id
                ):
                    return False, None
                if current.attempt_id == attempt_id:
                    return False, SharedAllocationClaim(current, current, False)
                kept = replace(
                    current,
                    attempt_id=attempt_id,
                    attempt_sequence=attempt_sequence,
                    claim_generation=current.claim_generation + 1,
                )
                snapshot.records[record_id] = kept
                return True, SharedAllocationClaim(kept, current, False)
            if reachable and current is not None and current.attempt_id == attempt_id:
                return False, SharedAllocationClaim(current, current, False)
            if not replace_unreachable and (
                current is None or not reachable or not current.reconnectable
            ):
                return False, None
            generation = 1 if current is None else current.claim_generation + 1
            orphaned = False
            if (
                reachable
                and current is not None
                and (current.state == "creating" or current.reconnectable)
            ):
                claimed = replace(
                    current,
                    attempt_id=attempt_id,
                    attempt_sequence=attempt_sequence,
                    claim_generation=generation,
                )
            else:
                if current is not None:
                    snapshot.orphans[current.allocation_session_id] = current
                    orphaned = True
                claimed = SharedAllocationRecord(
                    allocation_session_id=shared_allocation_session_id(owner_session_id),
                    scope=scope,
                    scope_id=scope_id,
                    owner_session_id=owner_session_id,
                    environment_name=environment_name,
                    agent_name=agent_name,
                    attempt_id=attempt_id,
                    attempt_sequence=attempt_sequence,
                    claim_generation=generation,
                    state="creating",
                    created_at=_now(),
                )
            snapshot.records[record_id] = claimed
            return True, SharedAllocationClaim(claimed, current, orphaned)

        return await self._mutate(owner_session_id, change)

    async def restore(self, record_id: str, claim: SharedAllocationClaim) -> None:
        """Undo ``claim`` if nothing changed since, as when its attempt turned out superseded."""

        record = claim.record

        def change(snapshot: SharedAllocationSnapshot) -> Any:
            if not record.same_claim(snapshot.records.get(record_id)):
                return False, None
            previous = claim.previous
            if previous is None:
                del snapshot.records[record_id]
                return True, None
            if claim.orphaned_previous:
                snapshot.orphans.pop(previous.allocation_session_id, None)
            snapshot.records[record_id] = previous
            return True, None

        if claim.previous is not None and record.same_claim(claim.previous):
            return
        await self._mutate(record.owner_session_id, change)

    async def update(
        self, record_id: str, record: SharedAllocationRecord, **changes: Any
    ) -> SharedAllocationRecord:
        """Replace the record ``record`` with ``changes`` applied, while it is still held.

        Raises ``SharedAllocationClaimLost`` when another attempt holds it now.
        """

        def change(snapshot: SharedAllocationSnapshot) -> Any:
            current = snapshot.records.get(record_id)
            if not record.same_claim(current):
                return False, None
            updated = replace(record, **changes)
            snapshot.records[record_id] = updated
            return True, updated

        updated = await self._mutate(record.owner_session_id, change)
        if updated is None:
            raise SharedAllocationClaimLost(
                f"Another workflow attempt took over the shared allocation of environment "
                f"{record.environment_name!r}."
            )
        return updated

    async def replace_with_fresh(
        self, record_id: str, record: SharedAllocationRecord
    ) -> SharedAllocationRecord:
        """Give the held record a fresh identity, after its allocation proved gone."""

        return await self.update(
            record_id,
            record,
            allocation_session_id=shared_allocation_session_id(record.owner_session_id),
            claim_generation=record.claim_generation + 1,
            state="creating",
            reconnect_metadata={},
            binding_state=None,
            result_metadata={},
            disposal_state=None,
            binding_required=True,
            allocation_intent=None,
            copied_back=(),
            pending_copy_back=(),
            created_at=_now(),
            kept_for_resume=False,
        )

    async def orphan_and_replace_with_fresh(
        self, record_id: str, record: SharedAllocationRecord
    ) -> SharedAllocationRecord:
        """Move the held record to the orphans and hold a fresh identity in its place.

        For an allocation a resume couldn't reconnect to that isn't proven gone:
        the run continues on a new one, and recovery or a hand release still
        owns the old one through its orphan.
        """

        fresh = replace(
            record,
            allocation_session_id=shared_allocation_session_id(record.owner_session_id),
            claim_generation=record.claim_generation + 1,
            state="creating",
            reconnect_metadata={},
            binding_state=None,
            result_metadata={},
            disposal_state=None,
            binding_required=True,
            allocation_intent=None,
            copied_back=(),
            pending_copy_back=(),
            created_at=_now(),
            release_lease=None,
            kept_for_resume=False,
        )

        def change(snapshot: SharedAllocationSnapshot) -> Any:
            current = snapshot.records.get(record_id)
            if current is None or not record.same_claim(current):
                return False, None
            snapshot.orphans[current.allocation_session_id] = replace(
                current, kept_for_resume=False
            )
            snapshot.records[record_id] = fresh
            return True, fresh

        updated = await self._mutate(record.owner_session_id, change)
        if updated is None:
            raise SharedAllocationClaimLost(
                f"Another workflow attempt took over the shared allocation of environment "
                f"{record.environment_name!r}."
            )
        return updated

    async def transition_intent(
        self,
        record: SharedAllocationRecord,
        *,
        expected: dict[str, Any] | None,
        desired: dict[str, Any],
    ) -> SharedAllocationRecord:
        """Move the record's allocation intent from ``expected`` to ``desired``.

        The record may be held or orphaned, and may be fenced for reaping; the
        claim must be unchanged. Repeating a transition that already happened
        returns the record as it is. Raises ``SharedAllocationClaimLost`` when
        another attempt holds the record now, and ``RuntimeError`` when the
        intent moved on some other way.
        """

        conflict = object()

        def change(snapshot: SharedAllocationSnapshot) -> Any:
            for table in (snapshot.records, snapshot.orphans):
                for key, current in tuple(table.items()):
                    if current.allocation_session_id != record.allocation_session_id:
                        continue
                    if not record.same_claim(current):
                        return False, None
                    if current.allocation_intent == desired:
                        return False, current
                    if current.allocation_intent != expected:
                        return False, conflict
                    updated = replace(current, allocation_intent=desired)
                    table[key] = updated
                    return True, updated
            return False, None

        updated = await self._mutate(record.owner_session_id, change)
        if updated is None:
            raise SharedAllocationClaimLost(
                f"Another attempt took over the shared allocation of environment "
                f"{record.environment_name!r}."
            )
        if updated is conflict:
            raise RuntimeError(
                "The allocation intent of a shared allocation changed while it was in use."
            )
        return updated

    async def orphan(self, record_id: str, record: SharedAllocationRecord) -> bool:
        """Move the held record to the orphans; False when another attempt holds it."""

        def change(snapshot: SharedAllocationSnapshot) -> Any:
            current = snapshot.records.get(record_id)
            if current is None or not record.same_claim(current):
                return False, False
            del snapshot.records[record_id]
            snapshot.orphans[current.allocation_session_id] = current
            return True, True

        return await self._mutate(record.owner_session_id, change)

    async def holds(
        self, owner_session_id: str, allocation_session_id: str, attempts: Collection[str]
    ) -> bool:
        """Whether one of ``attempts`` still holds the allocation (as a record or an orphan)."""

        snapshot = await self.load(owner_session_id)
        if snapshot is None:
            return False
        return any(
            record.allocation_session_id == allocation_session_id
            and record.attempt_id in attempts
            and record.state != "reaping"
            for record in snapshot.all()
        )

    async def remove(
        self, owner_session_id: str, allocation_session_id: str, attempts: Collection[str]
    ) -> bool:
        """Forget a disposed allocation one of ``attempts`` held; False when none did."""

        def change(snapshot: SharedAllocationSnapshot) -> Any:
            removed = False
            for table in (snapshot.records, snapshot.orphans):
                for key, record in tuple(table.items()):
                    if (
                        record.allocation_session_id == allocation_session_id
                        and record.attempt_id in attempts
                    ):
                        del table[key]
                        removed = True
            return removed, removed

        return await self._mutate(owner_session_id, change)

    async def mark_reaping(self, record: SharedAllocationRecord) -> SharedAllocationRecord | None:
        """Fence ``record`` for recovery's disposal; None when someone else has it.

        A reaping record is never reconnected: a claim moves it to the orphans
        and starts a fresh allocation, so disposal can't race a reattach. The
        fence is the same disposal lease a hand release takes
        (``claim_release``), so recovery and a release never both hold the
        allocation. None when the record changed since it was read, or another
        reaper or release holds a lease on it that hasn't expired. The reaper
        gives the lease up with ``drop_release`` if it can't finish.
        """

        return await self.claim_release(
            record, token=f"reaping-{secrets.token_hex(8)}", lease_seconds=DISPOSAL_LEASE_SECONDS
        )

    async def claim_release(
        self, record: SharedAllocationRecord, *, token: str, lease_seconds: float
    ) -> SharedAllocationRecord | None:
        """Fence ``record`` as reaping and hold it under a disposal lease named ``token``.

        Both a hand release and recovery's reaping (``mark_reaping``) take this
        lease, and only one can hold it at a time. The claim generation goes
        up, so a reaper or release that read the record before this can't
        fence it any more, and none can while the lease is live. A record that
        is already reaping can be taken once its holder gave the lease up or
        let it expire. None when the record changed since it was read, or
        another holder's lease on it hasn't expired.
        """

        now = datetime.now(UTC)
        lease = {"token": token, "expires_at": (now + timedelta(seconds=lease_seconds)).isoformat()}

        def change(snapshot: SharedAllocationSnapshot) -> Any:
            for table in (snapshot.records, snapshot.orphans):
                for key, current in tuple(table.items()):
                    if current.allocation_session_id != record.allocation_session_id:
                        continue
                    if not record.same_claim(current) or current.release_leased(now):
                        return False, None
                    held = replace(
                        current,
                        state="reaping",
                        claim_generation=current.claim_generation + 1,
                        release_lease=lease,
                    )
                    table[key] = held
                    return True, held
            return False, None

        return await self._mutate(record.owner_session_id, change)

    async def renew_release(
        self, record: SharedAllocationRecord, *, lease_seconds: float
    ) -> SharedAllocationRecord | None:
        """Extend the holder's disposal lease; None when it no longer holds it."""

        expires_at = (datetime.now(UTC) + timedelta(seconds=lease_seconds)).isoformat()
        return await self._change_release(record, expires_at=expires_at)

    async def drop_release(self, record: SharedAllocationRecord) -> None:
        """Give up a disposal lease that couldn't finish, keeping the record fenced as reaping."""

        await self._change_release(record, expires_at=None)

    async def _change_release(
        self, record: SharedAllocationRecord, *, expires_at: str | None
    ) -> SharedAllocationRecord | None:
        lease = record.release_lease
        assert lease is not None

        def change(snapshot: SharedAllocationSnapshot) -> Any:
            for table in (snapshot.records, snapshot.orphans):
                for key, current in tuple(table.items()):
                    held = current.release_lease
                    if not record.same_claim(current) or held is None:
                        continue
                    if held.get("token") != lease["token"]:
                        continue
                    if expires_at is not None and not current.release_leased(datetime.now(UTC)):
                        # An expired holder cannot revive its lease, even if no
                        # replacement has claimed the allocation yet.
                        continue
                    updated = replace(
                        current,
                        release_lease=(
                            None if expires_at is None else {**lease, "expires_at": expires_at}
                        ),
                    )
                    table[key] = updated
                    return True, updated
            return False, None

        return await self._mutate(record.owner_session_id, change)

    async def remove_reaped(self, record: SharedAllocationRecord) -> None:
        def change(snapshot: SharedAllocationSnapshot) -> Any:
            removed = False
            for table in (snapshot.records, snapshot.orphans):
                for key, current in tuple(table.items()):
                    if current.state == "reaping" and record.same_claim(current):
                        del table[key]
                        removed = True
            return removed, None

        await self._mutate(record.owner_session_id, change)
