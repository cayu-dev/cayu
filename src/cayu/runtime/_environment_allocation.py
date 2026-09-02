"""Crash-recoverable remote environment allocation coordination."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from cayu._exception_groups import add_exception_note_safely
from cayu._task_wait import await_shielded_task_outcome
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_json_value,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.environments import (
    EnvironmentAllocationContext,
    EnvironmentAllocationIntent,
    EnvironmentAllocationScope,
    EnvironmentAllocationState,
    EnvironmentFactoryOperation,
)
from cayu.runtime._binding_cleanup import binding_finalize_fatal_signal
from cayu.runtime._diagnostics import (
    _attach_runtime_exception_payload,
    _runtime_exception_payload,
)
from cayu.runtime.sessions import (
    CheckpointTransform,
    Session,
    SessionStore,
    session_fork_profile_relationship,
)
from cayu.vaults import SecretRedactor

ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY = "environment_factory_reconnect"
ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY = "environment_factory_allocation_owner"
ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY = "environment_factory_allocation_intents"
ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY = "environment_factory_allocation_receipts"
ENVIRONMENT_FACTORY_ALLOCATION_RECORD_SCHEMA_VERSION = 1
ENVIRONMENT_FACTORY_RECONNECT_METADATA_MAX_BYTES = 32 * 1024
_ENVIRONMENT_FACTORY_CHECKPOINT_MAY_BE_COMMITTED_ATTRIBUTE = (
    "_cayu_environment_factory_checkpoint_may_be_committed"
)

CheckpointTransformFactory = Callable[[dict[str, Any]], CheckpointTransform]


def environment_allocation_parent_session_id(session: Session) -> str | None:
    """Return authenticated lineage even after a fork source row is deleted."""

    if type(session) is not Session:
        raise TypeError("Environment allocation lineage requires a Session.")
    relationship = session_fork_profile_relationship(session)
    if relationship is not None:
        return relationship.source_session_id
    return session.parent_session_id


def environment_allocation_source_owner_session_id(
    session: Session,
    *,
    environment_name: str,
) -> str | None:
    """Return the authenticated owner of copied state for one environment."""

    if type(session) is not Session:
        raise TypeError("Environment allocation ownership requires a Session.")
    environment_name = require_clean_nonblank(environment_name, "environment_name")
    relationship = session_fork_profile_relationship(session)
    if relationship is None:
        return session.parent_session_id
    return next(
        (
            owner.owner_session_id
            for owner in relationship.source_environment_allocation_owners
            if owner.environment_name == environment_name
        ),
        None,
    )


def require_authorized_environment_allocation_owners(
    session: Session,
    owners_by_environment: Mapping[str, str],
) -> None:
    """Reject checkpoint owners outside each environment's immutable lineage."""

    if type(session) is not Session:
        raise TypeError("Environment allocation ownership requires a Session.")
    if not isinstance(owners_by_environment, Mapping):
        raise TypeError("Environment allocation owners must be a mapping.")
    for environment_name, owner_session_id in owners_by_environment.items():
        environment_name = require_clean_nonblank(environment_name, "environment_name")
        owner_session_id = require_clean_nonblank(owner_session_id, "owner_session_id")
        inherited_owner_session_id = environment_allocation_source_owner_session_id(
            session,
            environment_name=environment_name,
        )
        if owner_session_id != session.id and owner_session_id != inherited_owner_session_id:
            raise ValueError(
                "Environment allocation checkpoint owner conflicts with immutable "
                f"session lineage for environment {environment_name!r}."
            )


@dataclass(frozen=True)
class EnvironmentAllocationRecord:
    intent: EnvironmentAllocationIntent
    state: EnvironmentAllocationState
    reconnect_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.state is EnvironmentAllocationState.UNPREPARED:
            raise ValueError("Unprepared environment allocation state is not durable.")
        reconnect_metadata = self.reconnect_metadata
        if reconnect_metadata is not None:
            reconnect_metadata = copy_durable_json_object(
                reconnect_metadata,
                "reconnect_metadata",
            )
            require_bounded_reconnect_metadata(reconnect_metadata)
        if self.state is EnvironmentAllocationState.ACKNOWLEDGED:
            if reconnect_metadata is None:
                raise ValueError("Acknowledged environment allocation requires reconnect metadata.")
        elif reconnect_metadata is not None and self.state not in {
            EnvironmentAllocationState.REAPING,
            EnvironmentAllocationState.REAPED,
        }:
            raise ValueError(
                "Environment allocation reconnect metadata requires acknowledged, reaping, "
                "or reaped state."
            )
        object.__setattr__(
            self,
            "intent",
            EnvironmentAllocationIntent.from_payload(self.intent.to_payload()),
        )
        object.__setattr__(self, "reconnect_metadata", reconnect_metadata)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": ENVIRONMENT_FACTORY_ALLOCATION_RECORD_SCHEMA_VERSION,
            "state": self.state.value,
            "intent": self.intent.to_payload(),
        }
        if self.reconnect_metadata is not None:
            payload["reconnect_metadata"] = copy_durable_json_object(
                self.reconnect_metadata,
                "reconnect_metadata",
            )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EnvironmentAllocationRecord:
        if not isinstance(payload, Mapping):
            raise TypeError("Environment allocation record payload must be a mapping.")
        copied = copy_durable_json_object(dict(payload), "environment_allocation_record")
        expected = {"schema_version", "state", "intent"}
        if "reconnect_metadata" in copied:
            expected.add("reconnect_metadata")
        if set(copied) != expected:
            raise ValueError("Environment allocation record has an invalid schema.")
        if (
            type(copied["schema_version"]) is not int
            or copied["schema_version"] != ENVIRONMENT_FACTORY_ALLOCATION_RECORD_SCHEMA_VERSION
        ):
            raise ValueError("Environment allocation record schema version is unsupported.")
        try:
            state = EnvironmentAllocationState(copied["state"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Environment allocation record state is invalid.") from exc
        return cls(
            intent=EnvironmentAllocationIntent.from_payload(copied["intent"]),
            state=state,
            reconnect_metadata=copied.get("reconnect_metadata"),
        )


@dataclass(frozen=True)
class EnvironmentAllocationReceipt:
    intent: EnvironmentAllocationIntent
    reconnect_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        reconnect_metadata = copy_durable_json_object(
            self.reconnect_metadata,
            "reconnect_metadata",
        )
        require_bounded_reconnect_metadata(reconnect_metadata)
        object.__setattr__(
            self,
            "intent",
            EnvironmentAllocationIntent.from_payload(self.intent.to_payload()),
        )
        object.__setattr__(self, "reconnect_metadata", reconnect_metadata)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": ENVIRONMENT_FACTORY_ALLOCATION_RECORD_SCHEMA_VERSION,
            "intent": self.intent.to_payload(),
            "reconnect_metadata": copy_durable_json_object(
                self.reconnect_metadata,
                "reconnect_metadata",
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EnvironmentAllocationReceipt:
        if not isinstance(payload, Mapping):
            raise TypeError("Environment allocation receipt payload must be a mapping.")
        copied = copy_durable_json_object(dict(payload), "environment_allocation_receipt")
        if set(copied) != {
            "schema_version",
            "intent",
            "reconnect_metadata",
        }:
            raise ValueError("Environment allocation receipt has an invalid schema.")
        if (
            type(copied["schema_version"]) is not int
            or copied["schema_version"] != ENVIRONMENT_FACTORY_ALLOCATION_RECORD_SCHEMA_VERSION
        ):
            raise ValueError("Environment allocation receipt schema version is unsupported.")
        return cls(
            intent=EnvironmentAllocationIntent.from_payload(copied["intent"]),
            reconnect_metadata=copied["reconnect_metadata"],
        )


class EnvironmentAllocationTransitionConflict(RuntimeError):
    """Another lifecycle owner advanced the same allocation state."""


class EnvironmentAllocationCoordinator:
    """Own allocation records and their atomic checkpoint transitions."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        checkpoint_transform: CheckpointTransformFactory,
        secret_redactor: SecretRedactor,
    ) -> None:
        self._session_store = session_store
        self._checkpoint_transform = checkpoint_transform
        self._secret_redactor = secret_redactor

    async def load_record(
        self,
        *,
        session_id: str,
        environment_name: str,
    ) -> EnvironmentAllocationRecord | None:
        checkpoint = await self._session_store.load_checkpoint(session_id)
        return self.record_from_checkpoint(
            checkpoint,
            environment_name=environment_name,
        )

    def record_from_checkpoint(
        self,
        checkpoint: Mapping[str, Any] | None,
        *,
        environment_name: str,
    ) -> EnvironmentAllocationRecord | None:
        if checkpoint is None:
            return None
        record = allocation_record_from_checkpoint(
            checkpoint,
            environment_name=environment_name,
        )
        if record is not None:
            self._require_secret_free_intent(
                record.intent,
                field_name="environment allocation intent",
            )
            if record.reconnect_metadata is not None:
                self._require_secret_free_value(
                    record.reconnect_metadata,
                    field_name="environment allocation acknowledgement",
                )
        return record

    def receipt_from_checkpoint(
        self,
        checkpoint: Mapping[str, Any] | None,
        *,
        environment_name: str,
    ) -> EnvironmentAllocationReceipt | None:
        if checkpoint is None:
            return None
        receipt = allocation_receipt_from_checkpoint(
            checkpoint,
            environment_name=environment_name,
        )
        if receipt is not None:
            self._require_secret_free_intent(
                receipt.intent,
                field_name="environment allocation receipt intent",
            )
            self._require_secret_free_value(
                receipt.reconnect_metadata,
                field_name="environment allocation receipt reconnect metadata",
            )
        return receipt

    def context(
        self,
        *,
        session_id: str,
        inherited_owner_session_id: str | None,
        environment_name: str,
        scope: EnvironmentAllocationScope,
        existing: EnvironmentAllocationRecord | None,
    ) -> DurableEnvironmentAllocationContext:
        if existing is None or (
            inherited_owner_session_id is not None
            and existing.intent.session_id == inherited_owner_session_id
        ):
            intent = EnvironmentAllocationIntent(
                allocation_id=f"ealloc_{uuid4().hex}",
                provider=scope.provider,
                adapter_generation=scope.adapter_generation,
                session_id=session_id,
                environment_name=environment_name,
                requested_operation=EnvironmentFactoryOperation.CREATE,
            )
            state = EnvironmentAllocationState.UNPREPARED
            existing = None
        else:
            intent = existing.intent
            state = existing.state
            if intent.session_id != session_id or intent.environment_name != environment_name:
                raise RuntimeError("Environment allocation intent belongs to a different owner.")
            if intent.scope != scope:
                raise RuntimeError(
                    "Environment allocation intent belongs to a different provider "
                    "or adapter generation."
                )
        return DurableEnvironmentAllocationContext(
            coordinator=self,
            intent=intent,
            state=state,
            record=existing,
        )

    async def transition(
        self,
        *,
        session_id: str,
        environment_name: str,
        expected: EnvironmentAllocationRecord | None,
        desired: EnvironmentAllocationRecord,
    ) -> EnvironmentAllocationRecord:
        if (
            desired.intent.session_id != session_id
            or desired.intent.environment_name != environment_name
        ):
            raise RuntimeError("Environment allocation transition owner changed.")
        self._require_secret_free_intent(
            desired.intent,
            field_name="environment allocation intent",
        )

        def transform(session: Session, current: dict[str, Any] | None) -> dict[str, Any]:
            checkpoint = {} if current is None else copy_json_value(current, "checkpoint")
            existing = allocation_record_from_checkpoint(
                checkpoint,
                environment_name=environment_name,
            )
            if existing == desired and desired.state in {
                EnvironmentAllocationState.ACKNOWLEDGED,
                EnvironmentAllocationState.REAPING,
                EnvironmentAllocationState.REAPED,
            }:
                # A store acknowledgement can be lost after the atomic write,
                # and two recovery workers can independently reconstruct the
                # same provider result, cleanup fence, or cleanup completion.
                # Repeating those exact idempotent transitions is success.
                # PREPARED and DISPATCHED deliberately remain single-winner
                # fences.
                return checkpoint
            receipt = allocation_receipt_from_checkpoint(
                checkpoint,
                environment_name=environment_name,
            )
            desired_receipt = (
                EnvironmentAllocationReceipt(
                    intent=desired.intent,
                    reconnect_metadata=desired.reconnect_metadata,
                )
                if desired.state is EnvironmentAllocationState.ACKNOWLEDGED
                and desired.reconnect_metadata is not None
                else None
            )
            if (
                existing is None
                and desired_receipt is not None
                and receipt == desired_receipt
                and _publication_payload_matches(
                    checkpoint,
                    environment_name=environment_name,
                    session_id=session_id,
                    receipt=desired_receipt,
                )
            ):
                # Another worker may publish the acknowledgement between this
                # worker's provider lookup and its durable acknowledgement.
                # The immutable receipt is stronger evidence than the pending
                # record, so reconstruct the exact completed transition.
                return checkpoint
            if existing != expected and not _is_inherited_parent_record(
                session=session,
                environment_name=environment_name,
                existing=existing,
                desired=desired,
                expected=expected,
            ):
                raise EnvironmentAllocationTransitionConflict(
                    "Environment allocation transition lost its expected durable state."
                )
            if receipt is not None:
                if not _is_inherited_parent_receipt(
                    session=session,
                    checkpoint=checkpoint,
                    environment_name=environment_name,
                    receipt=receipt,
                    desired=desired,
                ):
                    raise RuntimeError(
                        "Environment allocation intent conflicts with a published receipt."
                    )
                receipts = checkpoint_object_map(
                    checkpoint,
                    ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY,
                )
                receipts.pop(environment_name, None)
                if receipts:
                    checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY] = receipts
                else:
                    checkpoint.pop(
                        ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY,
                        None,
                    )
            records = checkpoint_object_map(
                checkpoint,
                ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY,
            )
            records[environment_name] = desired.to_payload()
            checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY] = records
            transformed = self._checkpoint_transform(checkpoint)(session, current)
            if transformed is None:
                raise RuntimeError(
                    "Environment allocation transition unexpectedly deleted the checkpoint."
                )
            if (
                allocation_record_from_checkpoint(
                    transformed,
                    environment_name=environment_name,
                )
                != desired
                or allocation_receipt_from_checkpoint(
                    transformed,
                    environment_name=environment_name,
                )
                is not None
            ):
                raise RuntimeError(
                    "Environment allocation transition was removed by the checkpoint transform."
                )
            return transformed

        try:
            await self._session_store.transform_checkpoint(session_id, transform)
        except EnvironmentAllocationTransitionConflict:
            raise
        except BaseException as exc:
            reconciled = await self._reconcile_record(
                session_id=session_id,
                environment_name=environment_name,
                expected=desired,
                original_error=exc,
            )
            if reconciled is not None:
                return reconciled
            raise
        persisted = await self._load_transition_result(
            session_id=session_id,
            environment_name=environment_name,
            expected=desired,
        )
        if persisted != desired:
            raise RuntimeError("Environment allocation transition was not durably observable.")
        if persisted is None:
            raise RuntimeError("Environment allocation transition disappeared.")
        return persisted

    async def _reconcile_record(
        self,
        *,
        session_id: str,
        environment_name: str,
        expected: EnvironmentAllocationRecord,
        original_error: BaseException,
    ) -> EnvironmentAllocationRecord | None:
        current_task = asyncio.current_task()
        caller_cancellation = (
            original_error
            if isinstance(original_error, asyncio.CancelledError)
            and current_task is not None
            and current_task.cancelling() > 0
            else None
        )
        outcome = await await_shielded_task_outcome(
            asyncio.create_task(
                self._load_transition_result(
                    session_id=session_id,
                    environment_name=environment_name,
                    expected=expected,
                )
            ),
            cancellation=caller_cancellation,
        )
        if outcome.error is not None:
            fatal_signal = binding_finalize_fatal_signal(outcome.error)
            if fatal_signal is not None:
                add_exception_note_safely(
                    fatal_signal,
                    "The environment allocation transition also failed; its commit "
                    "outcome could not be reconciled.",
                )
                raise fatal_signal from original_error
            add_exception_note_safely(
                original_error,
                "Could not reconcile the durable environment allocation transition.",
            )
            if outcome.cancellation is not None and outcome.cancellation is not original_error:
                raise BaseExceptionGroup(
                    "Environment allocation transition failed during caller cancellation.",
                    [original_error, outcome.cancellation],
                ) from outcome.error
            raise original_error from outcome.error
        if outcome.result == expected:
            fatal_signal = binding_finalize_fatal_signal(original_error)
            if fatal_signal is not None:
                raise fatal_signal from original_error
            if outcome.cancellation is not None:
                raise outcome.cancellation from original_error
            return outcome.result
        if outcome.cancellation is not None and outcome.cancellation is not original_error:
            raise BaseExceptionGroup(
                "Environment allocation transition failed during caller cancellation.",
                [original_error, outcome.cancellation],
            ) from original_error
        return None

    async def _load_transition_result(
        self,
        *,
        session_id: str,
        environment_name: str,
        expected: EnvironmentAllocationRecord,
    ) -> EnvironmentAllocationRecord | None:
        persisted = await self.load_record(
            session_id=session_id,
            environment_name=environment_name,
        )
        if persisted is not None:
            return persisted
        if (
            expected.state is EnvironmentAllocationState.ACKNOWLEDGED
            and expected.reconnect_metadata is not None
            and await self._publication_matches(
                session_id=session_id,
                environment_name=environment_name,
                receipt=EnvironmentAllocationReceipt(
                    intent=expected.intent,
                    reconnect_metadata=expected.reconnect_metadata,
                ),
            )
        ):
            return expected
        return None

    async def publish(
        self,
        *,
        session_id: str,
        environment_name: str,
        expected_record: EnvironmentAllocationRecord,
    ) -> None:
        if (
            expected_record.intent.session_id != session_id
            or expected_record.intent.environment_name != environment_name
        ):
            raise RuntimeError("Environment allocation publication owner changed.")
        self._require_secret_free_intent(
            expected_record.intent,
            field_name="environment allocation receipt intent",
        )
        if expected_record.state is not EnvironmentAllocationState.ACKNOWLEDGED:
            raise RuntimeError("Only an acknowledged environment allocation can be published.")
        reconnect_metadata = expected_record.reconnect_metadata
        if reconnect_metadata is None:
            raise RuntimeError("Environment allocation acknowledgement is missing.")
        receipt = EnvironmentAllocationReceipt(
            intent=expected_record.intent,
            reconnect_metadata=reconnect_metadata,
        )

        def transform(session: Session, current: dict[str, Any] | None) -> dict[str, Any]:
            checkpoint = {} if current is None else copy_json_value(current, "checkpoint")
            existing_record = allocation_record_from_checkpoint(
                checkpoint,
                environment_name=environment_name,
            )
            existing_receipt = allocation_receipt_from_checkpoint(
                checkpoint,
                environment_name=environment_name,
            )
            if existing_receipt is not None:
                if existing_receipt != receipt or existing_record is not None:
                    raise RuntimeError(
                        "Environment allocation publication conflicts with an existing receipt."
                    )
            elif existing_record != expected_record:
                raise RuntimeError(
                    "Environment allocation publication lost its acknowledged intent."
                )

            reconnect_state = checkpoint_object_map(
                checkpoint,
                ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY,
            )
            owner_state = checkpoint_object_map(
                checkpoint,
                ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY,
            )
            receipts = checkpoint_object_map(
                checkpoint,
                ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY,
            )
            records = checkpoint_object_map(
                checkpoint,
                ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY,
            )
            reconnect_state[environment_name] = copy_durable_json_object(
                reconnect_metadata,
                "reconnect_metadata",
            )
            owner_state[environment_name] = session_id
            receipts[environment_name] = receipt.to_payload()
            records.pop(environment_name, None)
            checkpoint[ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY] = reconnect_state
            checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY] = owner_state
            checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY] = receipts
            if records:
                checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY] = records
            else:
                checkpoint.pop(
                    ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY,
                    None,
                )
            transformed = self._checkpoint_transform(checkpoint)(session, current)
            if transformed is None:
                raise RuntimeError(
                    "Environment allocation publication unexpectedly deleted the checkpoint."
                )
            if not _publication_payload_matches(
                transformed,
                environment_name=environment_name,
                session_id=session_id,
                receipt=receipt,
            ):
                raise RuntimeError(
                    "Environment allocation publication was removed by the checkpoint transform."
                )
            return transformed

        try:
            await self._session_store.transform_checkpoint(session_id, transform)
        except BaseException as exc:
            if await self._reconcile_publication_failure(
                session_id=session_id,
                environment_name=environment_name,
                receipt=receipt,
                original_error=exc,
            ):
                return
            raise
        try:
            publication_matches = await self._publication_matches(
                session_id=session_id,
                environment_name=environment_name,
                receipt=receipt,
            )
        except BaseException as exc:
            mark_environment_factory_checkpoint_may_be_committed(exc)
            raise
        if not publication_matches:
            error = RuntimeError("Environment allocation publication was not durably observable.")
            mark_environment_factory_checkpoint_may_be_committed(error)
            raise error

    async def _reconcile_publication_failure(
        self,
        *,
        session_id: str,
        environment_name: str,
        receipt: EnvironmentAllocationReceipt,
        original_error: BaseException,
    ) -> bool:
        current_task = asyncio.current_task()
        caller_cancellation = (
            original_error
            if isinstance(original_error, asyncio.CancelledError)
            and current_task is not None
            and current_task.cancelling() > 0
            else None
        )
        outcome = await await_shielded_task_outcome(
            asyncio.create_task(
                self._publication_matches(
                    session_id=session_id,
                    environment_name=environment_name,
                    receipt=receipt,
                )
            ),
            cancellation=caller_cancellation,
        )
        if outcome.error is None and outcome.result and outcome.cancellation is None:
            fatal_signal = binding_finalize_fatal_signal(original_error)
            if fatal_signal is not None:
                mark_environment_factory_checkpoint_may_be_committed(fatal_signal)
                raise fatal_signal from original_error
            return True
        checkpoint_may_be_committed = outcome.error is not None or bool(outcome.result)
        if outcome.error is not None:
            fatal_signal = binding_finalize_fatal_signal(outcome.error)
            if fatal_signal is not None:
                mark_environment_factory_checkpoint_may_be_committed(fatal_signal)
                add_exception_note_safely(
                    fatal_signal,
                    "The environment allocation publication also failed; its commit "
                    "outcome could not be reconciled.",
                )
                raise fatal_signal from original_error
            add_exception_note_safely(
                original_error,
                "Could not reconcile whether the environment allocation publication "
                "committed; the exact allocation will be preserved.",
            )
        propagated_error: BaseException = original_error
        if outcome.cancellation is not None and outcome.cancellation is not original_error:
            propagated_error = BaseExceptionGroup(
                "Environment allocation publication failed during caller cancellation.",
                [original_error, outcome.cancellation],
            )
        if checkpoint_may_be_committed:
            mark_environment_factory_checkpoint_may_be_committed(propagated_error)
        if outcome.error is not None:
            raise propagated_error from outcome.error
        if propagated_error is original_error:
            raise original_error
        raise propagated_error from original_error

    async def _publication_matches(
        self,
        *,
        session_id: str,
        environment_name: str,
        receipt: EnvironmentAllocationReceipt,
    ) -> bool:
        checkpoint = await self._session_store.load_checkpoint(session_id)
        if checkpoint is None:
            return False
        return _publication_payload_matches(
            checkpoint,
            environment_name=environment_name,
            session_id=session_id,
            receipt=receipt,
        )

    def _require_secret_free_value(
        self,
        value: Mapping[str, Any],
        *,
        field_name: str,
    ) -> None:
        self._secret_redactor.require_no_secret_keys(
            value,
            field_name=field_name,
            match_short_substrings=True,
        )
        if self._secret_redactor.redact_json_values(dict(value)) != dict(value):
            raise ValueError(f"{field_name} contains a workload secret.")

    def _require_secret_free_intent(
        self,
        intent: EnvironmentAllocationIntent,
        *,
        field_name: str,
    ) -> None:
        self._require_secret_free_value(
            {
                "provider": intent.provider,
                "adapter_generation": intent.adapter_generation,
                "provider_metadata": intent.provider_metadata,
            },
            field_name=field_name,
        )


class DurableEnvironmentAllocationContext(EnvironmentAllocationContext):
    """Exact checkpoint-backed state machine for one factory allocation."""

    def __init__(
        self,
        *,
        coordinator: EnvironmentAllocationCoordinator,
        intent: EnvironmentAllocationIntent,
        state: EnvironmentAllocationState,
        record: EnvironmentAllocationRecord | None,
    ) -> None:
        self._coordinator = coordinator
        self._intent = EnvironmentAllocationIntent.from_payload(intent.to_payload())
        self._state = state
        self._record = record

    @property
    def intent(self) -> EnvironmentAllocationIntent:
        return EnvironmentAllocationIntent.from_payload(self._intent.to_payload())

    @property
    def state(self) -> EnvironmentAllocationState:
        return self._state

    @property
    def acknowledged_reconnect_metadata(self) -> dict[str, Any] | None:
        record = self._record
        if record is None or record.reconnect_metadata is None:
            return None
        return copy_durable_json_object(
            record.reconnect_metadata,
            "reconnect_metadata",
        )

    async def prepare(
        self,
        provider_metadata: Mapping[str, Any],
    ) -> EnvironmentAllocationIntent:
        if not isinstance(provider_metadata, Mapping):
            raise TypeError("provider_metadata must be a mapping.")
        copied = copy_durable_json_object(
            dict(provider_metadata),
            "provider_metadata",
        )
        self._coordinator._require_secret_free_value(
            copied,
            field_name="environment allocation provider metadata",
        )
        prepared_intent = self._intent.with_provider_metadata(copied)
        if self._state is not EnvironmentAllocationState.UNPREPARED:
            if prepared_intent != self._intent:
                raise RuntimeError(
                    "Environment allocation provider metadata changed after preparation."
                )
            return self.intent
        desired = EnvironmentAllocationRecord(
            intent=prepared_intent,
            state=EnvironmentAllocationState.PREPARED,
        )
        persisted = await self._coordinator.transition(
            session_id=self._intent.session_id,
            environment_name=self._intent.environment_name,
            expected=None,
            desired=desired,
        )
        self._adopt_record(persisted)
        return self.intent

    async def mark_dispatched(self) -> None:
        if self._state is EnvironmentAllocationState.DISPATCHED:
            return
        if self._state is not EnvironmentAllocationState.PREPARED or self._record is None:
            raise RuntimeError("Environment allocation must be prepared before provider dispatch.")
        desired = EnvironmentAllocationRecord(
            intent=self._intent,
            state=EnvironmentAllocationState.DISPATCHED,
        )
        persisted = await self._coordinator.transition(
            session_id=self._intent.session_id,
            environment_name=self._intent.environment_name,
            expected=self._record,
            desired=desired,
        )
        self._adopt_record(persisted)

    async def acknowledge(self, reconnect_metadata: Mapping[str, Any]) -> None:
        if not isinstance(reconnect_metadata, Mapping):
            raise TypeError("reconnect_metadata must be a mapping.")
        copied = copy_durable_json_object(
            dict(reconnect_metadata),
            "reconnect_metadata",
        )
        self._coordinator._require_secret_free_value(
            copied,
            field_name="environment allocation acknowledgement",
        )
        require_bounded_reconnect_metadata(copied)
        if self._state is EnvironmentAllocationState.ACKNOWLEDGED:
            if self.acknowledged_reconnect_metadata != copied:
                raise RuntimeError(
                    "Environment allocation acknowledgement changed after publication."
                )
            return
        if self._state is not EnvironmentAllocationState.DISPATCHED or self._record is None:
            raise RuntimeError("Environment allocation must be dispatched before acknowledgement.")
        desired = EnvironmentAllocationRecord(
            intent=self._intent,
            state=EnvironmentAllocationState.ACKNOWLEDGED,
            reconnect_metadata=copied,
        )
        persisted = await self._coordinator.transition(
            session_id=self._intent.session_id,
            environment_name=self._intent.environment_name,
            expected=self._record,
            desired=desired,
        )
        self._adopt_record(persisted)

    async def mark_reaped(self) -> None:
        if self._state is EnvironmentAllocationState.REAPED:
            return
        if self._state is not EnvironmentAllocationState.REAPING or self._record is None:
            raise RuntimeError(
                "Environment allocation cleanup must be durably fenced before it is marked reaped."
            )
        desired = EnvironmentAllocationRecord(
            intent=self._intent,
            state=EnvironmentAllocationState.REAPED,
            reconnect_metadata=self.acknowledged_reconnect_metadata,
        )
        persisted = await self._coordinator.transition(
            session_id=self._intent.session_id,
            environment_name=self._intent.environment_name,
            expected=self._record,
            desired=desired,
        )
        self._adopt_record(persisted)

    async def mark_reaping(self) -> bool:
        if self._state is EnvironmentAllocationState.REAPING:
            return True
        if self._state is EnvironmentAllocationState.REAPED:
            return True
        if (
            self._state
            not in {
                EnvironmentAllocationState.DISPATCHED,
                EnvironmentAllocationState.ACKNOWLEDGED,
            }
            or self._record is None
        ):
            raise RuntimeError(
                "Only a dispatched environment allocation can acquire the cleanup fence."
            )
        expected = self._record
        desired = EnvironmentAllocationRecord(
            intent=self._intent,
            state=EnvironmentAllocationState.REAPING,
            reconnect_metadata=self.acknowledged_reconnect_metadata,
        )
        try:
            persisted = await self._coordinator.transition(
                session_id=self._intent.session_id,
                environment_name=self._intent.environment_name,
                expected=expected,
                desired=desired,
            )
        except EnvironmentAllocationTransitionConflict:
            reconnect_metadata = expected.reconnect_metadata
            if reconnect_metadata is None:
                raise
            receipt = EnvironmentAllocationReceipt(
                intent=expected.intent,
                reconnect_metadata=reconnect_metadata,
            )
            if not await self._coordinator._publication_matches(
                session_id=self._intent.session_id,
                environment_name=self._intent.environment_name,
                receipt=receipt,
            ):
                raise
            return False
        self._adopt_record(persisted)
        return True

    async def publish(self) -> None:
        record = self._record
        if record is None:
            raise RuntimeError(
                "Recoverable environment factory lost its durable allocation record "
                "before publication."
            )
        await self._coordinator.publish(
            session_id=self._intent.session_id,
            environment_name=self._intent.environment_name,
            expected_record=record,
        )

    def _adopt_record(self, record: EnvironmentAllocationRecord) -> None:
        self._record = record
        self._intent = EnvironmentAllocationIntent.from_payload(record.intent.to_payload())
        self._state = record.state


def checkpoint_object_map(
    checkpoint: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    candidate = checkpoint.get(key)
    if candidate is None:
        return {}
    if type(candidate) is not dict:
        raise ValueError(f"{key} checkpoint state must be an object.")
    return copy_durable_json_object(candidate, key)


def allocation_record_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    environment_name: str,
) -> EnvironmentAllocationRecord | None:
    records = checkpoint_object_map(
        checkpoint,
        ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY,
    )
    payload = records.get(environment_name)
    if payload is None:
        return None
    if type(payload) is not dict:
        raise ValueError("Environment allocation record must be an object.")
    return EnvironmentAllocationRecord.from_payload(payload)


def allocation_receipt_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    environment_name: str,
) -> EnvironmentAllocationReceipt | None:
    receipts = checkpoint_object_map(
        checkpoint,
        ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY,
    )
    payload = receipts.get(environment_name)
    if payload is None:
        return None
    if type(payload) is not dict:
        raise ValueError("Environment allocation receipt must be an object.")
    return EnvironmentAllocationReceipt.from_payload(payload)


def environment_allocation_owners_from_checkpoint(
    checkpoint: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Resolve every allocation owner carried by one fork checkpoint."""

    if checkpoint is None:
        return {}
    records = checkpoint_object_map(
        checkpoint,
        ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY,
    )
    receipts = checkpoint_object_map(
        checkpoint,
        ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY,
    )
    owner_state = checkpoint_object_map(
        checkpoint,
        ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY,
    )
    reconnect_state = checkpoint_object_map(
        checkpoint,
        ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY,
    )
    duplicate_environments = records.keys() & receipts.keys()
    if duplicate_environments:
        raise ValueError(
            "Environment allocation checkpoint contains both a pending intent and a receipt."
        )
    owners: dict[str, str] = {}
    environment_names = sorted(
        require_clean_nonblank(environment_name, "environment_name")
        for environment_name in records.keys() | receipts.keys()
    )
    for environment_name in environment_names:
        if environment_name in records:
            record_payload = records[environment_name]
            if type(record_payload) is not dict:
                raise ValueError("Environment allocation record must be an object.")
            record = EnvironmentAllocationRecord.from_payload(record_payload)
            if record.intent.environment_name != environment_name:
                raise ValueError("Environment allocation intent conflicts with its checkpoint key.")
            owners[environment_name] = record.intent.session_id
            continue
        if environment_name not in receipts:
            raise AssertionError("Environment allocation owner discovery lost its state.")
        receipt_payload = receipts[environment_name]
        if type(receipt_payload) is not dict:
            raise ValueError("Environment allocation receipt must be an object.")
        receipt = EnvironmentAllocationReceipt.from_payload(receipt_payload)
        if receipt.intent.environment_name != environment_name:
            raise ValueError("Environment allocation receipt conflicts with its checkpoint key.")
        if (
            owner_state.get(environment_name) != receipt.intent.session_id
            or reconnect_state.get(environment_name) != receipt.reconnect_metadata
        ):
            raise ValueError("Environment allocation receipt has inconsistent ownership state.")
        owners[environment_name] = receipt.intent.session_id
    return owners


def _publication_payload_matches(
    checkpoint: Mapping[str, Any],
    *,
    environment_name: str,
    session_id: str,
    receipt: EnvironmentAllocationReceipt,
) -> bool:
    reconnect_state = checkpoint_object_map(
        checkpoint,
        ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY,
    )
    owner_state = checkpoint_object_map(
        checkpoint,
        ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY,
    )
    persisted_receipt = allocation_receipt_from_checkpoint(
        checkpoint,
        environment_name=environment_name,
    )
    pending = allocation_record_from_checkpoint(
        checkpoint,
        environment_name=environment_name,
    )
    return (
        persisted_receipt == receipt
        and pending is None
        and owner_state.get(environment_name) == session_id
        and reconnect_state.get(environment_name) == receipt.reconnect_metadata
    )


def require_bounded_reconnect_metadata(reconnect_metadata: Mapping[str, Any]) -> None:
    if (
        len(
            canonical_durable_json_bytes(
                dict(reconnect_metadata),
                "reconnect_metadata",
            )
        )
        > ENVIRONMENT_FACTORY_RECONNECT_METADATA_MAX_BYTES
    ):
        raise ValueError("Environment factory reconnect metadata exceeds the durable byte limit.")


def _is_inherited_parent_receipt(
    *,
    session: Session,
    checkpoint: Mapping[str, Any],
    environment_name: str,
    receipt: EnvironmentAllocationReceipt,
    desired: EnvironmentAllocationRecord,
) -> bool:
    source_owner_session_id = environment_allocation_source_owner_session_id(
        session,
        environment_name=environment_name,
    )
    if (
        desired.state is not EnvironmentAllocationState.PREPARED
        or source_owner_session_id is None
        or receipt.intent.session_id != source_owner_session_id
        or receipt.intent.environment_name != environment_name
    ):
        return False
    owners = checkpoint_object_map(
        checkpoint,
        ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY,
    )
    reconnect = checkpoint_object_map(
        checkpoint,
        ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY,
    )
    return (
        owners.get(environment_name) == receipt.intent.session_id
        and reconnect.get(environment_name) == receipt.reconnect_metadata
    )


def _is_inherited_parent_record(
    *,
    session: Session,
    environment_name: str,
    existing: EnvironmentAllocationRecord | None,
    desired: EnvironmentAllocationRecord,
    expected: EnvironmentAllocationRecord | None,
) -> bool:
    source_owner_session_id = environment_allocation_source_owner_session_id(
        session,
        environment_name=environment_name,
    )
    return (
        expected is None
        and existing is not None
        and desired.state is EnvironmentAllocationState.PREPARED
        and source_owner_session_id is not None
        and existing.intent.session_id == source_owner_session_id
        and existing.intent.environment_name == environment_name
    )


def mark_environment_factory_checkpoint_may_be_committed(
    error: BaseException,
) -> None:
    _attach_runtime_exception_payload(
        error,
        attribute_name=_ENVIRONMENT_FACTORY_CHECKPOINT_MAY_BE_COMMITTED_ATTRIBUTE,
        payload={"checkpoint_may_be_committed": True},
    )


def environment_factory_checkpoint_may_be_committed(
    error: BaseException,
) -> bool:
    return _runtime_exception_payload(
        error,
        attribute_name=_ENVIRONMENT_FACTORY_CHECKPOINT_MAY_BE_COMMITTED_ATTRIBUTE,
    ) == {"checkpoint_may_be_committed": True}
