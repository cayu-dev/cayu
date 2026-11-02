"""Recoverable creation for shared allocations.

A factory that declares an ``allocation_scope`` creates an ordinary session's
allocation through ``create_recoverable``, with an ``EnvironmentAllocationContext``
whose intent lives in that session's checkpoint. A shared allocation belongs to
no session of its own, so its intent lives in its shared record instead
(``SharedAllocationRecord.allocation_intent``), in the owner's checkpoint. The
intent and its states are the same ``EnvironmentAllocationRecord`` the
per-session coordinator uses, and every transition is a compare-and-set that
also checks the record's claim, so a superseded attempt can't advance it.

The intent names the allocation's own identity (the ``.shared-<hex>``
``session_id`` its requests carry), so the factory sees the request and intent
agree as it does for a session. A crash anywhere in the factory's create then
leaves a prepared, dispatched or acknowledged intent: the owner's next attempt
passes it back to ``create_recoverable`` to recover, and recovery can hand it
to ``reap_allocation`` to clean up.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from cayu._validation import copy_durable_json_object
from cayu.environments.factory import (
    EnvironmentAllocationContext,
    EnvironmentAllocationIntent,
    EnvironmentAllocationScope,
    EnvironmentAllocationState,
    EnvironmentFactoryOperation,
)
from cayu.runtime._environment_allocation import (
    EnvironmentAllocationRecord,
    require_bounded_reconnect_metadata,
)
from cayu.runtime._shared_allocation_records import (
    SharedAllocationRecord,
    SharedAllocationRecords,
)


class SharedAllocationIntentContext(EnvironmentAllocationContext):
    """An allocation context whose durable state is one shared record's intent."""

    def __init__(
        self,
        records: SharedAllocationRecords,
        record: SharedAllocationRecord,
        *,
        intent: EnvironmentAllocationIntent,
        persisted: EnvironmentAllocationRecord | None,
    ) -> None:
        self._records = records
        self._record = record
        self._intent = intent
        self._persisted = persisted

    @classmethod
    def open(
        cls,
        records: SharedAllocationRecords,
        record: SharedAllocationRecord,
        scope: EnvironmentAllocationScope | None = None,
    ) -> SharedAllocationIntentContext:
        """Resume the record's intent, or start one for ``scope`` when it has none.

        Raises when the recorded intent belongs to another allocation or, with
        ``scope``, to another provider or adapter generation.
        """

        persisted = (
            None
            if record.allocation_intent is None
            else EnvironmentAllocationRecord.from_payload(record.allocation_intent)
        )
        if persisted is None:
            if scope is None:
                raise RuntimeError("The shared allocation has no allocation intent.")
            intent = EnvironmentAllocationIntent(
                allocation_id=f"ealloc_{uuid4().hex}",
                provider=scope.provider,
                adapter_generation=scope.adapter_generation,
                session_id=record.allocation_session_id,
                environment_name=record.environment_name,
                requested_operation=EnvironmentFactoryOperation.CREATE,
            )
        else:
            intent = persisted.intent
            if (
                intent.session_id != record.allocation_session_id
                or intent.environment_name != record.environment_name
            ):
                raise RuntimeError("The allocation intent belongs to another shared allocation.")
            if scope is not None and intent.scope != scope:
                raise RuntimeError(
                    "The shared allocation's intent belongs to another provider or adapter "
                    "generation than the factory now declares."
                )
        return cls(records, record, intent=intent, persisted=persisted)

    @property
    def record(self) -> SharedAllocationRecord:
        """The shared record as this context last wrote or read it."""

        return self._record

    @property
    def intent(self) -> EnvironmentAllocationIntent:
        return EnvironmentAllocationIntent.from_payload(self._intent.to_payload())

    @property
    def state(self) -> EnvironmentAllocationState:
        if self._persisted is None:
            return EnvironmentAllocationState.UNPREPARED
        return self._persisted.state

    @property
    def dispatch_precluded(self) -> bool:
        return self._persisted is not None and self._persisted.dispatch_precluded

    @property
    def acknowledged_reconnect_metadata(self) -> dict[str, Any] | None:
        persisted = self._persisted
        if persisted is None or persisted.reconnect_metadata is None:
            return None
        return copy_durable_json_object(persisted.reconnect_metadata, "reconnect_metadata")

    async def _transition(self, desired: EnvironmentAllocationRecord) -> None:
        expected = None if self._persisted is None else self._persisted.to_payload()
        self._record = await self._records.transition_intent(
            self._record, expected=expected, desired=desired.to_payload()
        )
        self._persisted = desired
        self._intent = desired.intent

    async def prepare(self, provider_metadata: Mapping[str, Any]) -> EnvironmentAllocationIntent:
        if not isinstance(provider_metadata, Mapping):
            raise TypeError("provider_metadata must be a mapping.")
        prepared = self._intent.with_provider_metadata(
            copy_durable_json_object(dict(provider_metadata), "provider_metadata")
        )
        if self.state is not EnvironmentAllocationState.UNPREPARED:
            if prepared != self._intent:
                raise RuntimeError(
                    "Environment allocation provider metadata changed after preparation."
                )
            return self.intent
        await self._transition(
            EnvironmentAllocationRecord(intent=prepared, state=EnvironmentAllocationState.PREPARED)
        )
        return self.intent

    async def mark_dispatched(self) -> None:
        if self.state is EnvironmentAllocationState.DISPATCHED:
            return
        if self.state is not EnvironmentAllocationState.PREPARED:
            raise RuntimeError("Environment allocation must be prepared before provider dispatch.")
        await self._transition(
            EnvironmentAllocationRecord(
                intent=self._intent, state=EnvironmentAllocationState.DISPATCHED
            )
        )

    async def acknowledge(self, reconnect_metadata: Mapping[str, Any]) -> None:
        if not isinstance(reconnect_metadata, Mapping):
            raise TypeError("reconnect_metadata must be a mapping.")
        copied = copy_durable_json_object(dict(reconnect_metadata), "reconnect_metadata")
        require_bounded_reconnect_metadata(copied)
        state = self.state
        if state is EnvironmentAllocationState.ACKNOWLEDGED or (
            state is EnvironmentAllocationState.REAPING
            and self.acknowledged_reconnect_metadata is not None
        ):
            if self.acknowledged_reconnect_metadata != copied:
                raise RuntimeError(
                    "Environment allocation acknowledgement changed after publication."
                )
            return
        if (
            state not in {EnvironmentAllocationState.DISPATCHED, EnvironmentAllocationState.REAPING}
            or self.dispatch_precluded
        ):
            raise RuntimeError("Environment allocation must be dispatched before acknowledgement.")
        await self._transition(
            EnvironmentAllocationRecord(
                intent=self._intent,
                state=(
                    EnvironmentAllocationState.REAPING
                    if state is EnvironmentAllocationState.REAPING
                    else EnvironmentAllocationState.ACKNOWLEDGED
                ),
                reconnect_metadata=copied,
            )
        )

    async def mark_reaping(self) -> bool:
        # A shared allocation is published to its scope only by the attempt
        # that holds its claim, and the claim is checked on every transition,
        # so cleanup never races a publication it would have to yield to.
        state = self.state
        if state in {EnvironmentAllocationState.REAPING, EnvironmentAllocationState.REAPED}:
            return True
        if state not in {
            EnvironmentAllocationState.PREPARED,
            EnvironmentAllocationState.DISPATCHED,
            EnvironmentAllocationState.ACKNOWLEDGED,
        }:
            raise RuntimeError(
                "Only a prepared or dispatched allocation can acquire the cleanup fence."
            )
        await self._transition(
            EnvironmentAllocationRecord(
                intent=self._intent,
                state=EnvironmentAllocationState.REAPING,
                reconnect_metadata=self.acknowledged_reconnect_metadata,
                dispatch_precluded=state is EnvironmentAllocationState.PREPARED,
            )
        )
        return True

    async def mark_reaped(self) -> None:
        if self.state is EnvironmentAllocationState.REAPED:
            return
        if self.state is not EnvironmentAllocationState.REAPING:
            raise RuntimeError(
                "Environment allocation cleanup must be durably fenced before it is marked reaped."
            )
        await self._transition(
            EnvironmentAllocationRecord(
                intent=self._intent,
                state=EnvironmentAllocationState.REAPED,
                reconnect_metadata=self.acknowledged_reconnect_metadata,
                dispatch_precluded=self.dispatch_precluded,
            )
        )
