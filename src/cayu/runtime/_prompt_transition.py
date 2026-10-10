"""Durable prompt transition intent publication and reconciliation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_record,
    copy_json_value,
    require_clean_nonblank,
)
from cayu.sessions.base import (
    SessionStore,
)
from cayu.sessions.forks import (
    PromptAnatomyTransitionReceipt,
    session_prompt_anatomy_transition,
)
from cayu.sessions.records import (
    Session,
)

_PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY = "prompt_anatomy_transition_intents"

_PROMPT_ANATOMY_TRANSITION_INTENT_LIMIT = 256


class _PromptTransitionIntentStatus(StrEnum):
    PREPARED = "prepared"
    COMPLETED = "completed"


class _PromptTransitionIntent(BaseModel):
    """Typed durable authority for one prompt-succession effect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transition_id: str
    request_sha256: str
    source_session_id: str
    descendant_session_id: str
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_snapshot_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    status: _PromptTransitionIntentStatus
    requested_at: datetime
    descendant_created_at: datetime | None = None
    completed_at: datetime | None = None

    @classmethod
    def prepared(
        cls,
        *,
        receipt: PromptAnatomyTransitionReceipt,
        source_run_epoch: int,
        source_snapshot_cursor: int,
        requested_at: datetime,
    ) -> _PromptTransitionIntent:
        return cls(
            transition_id=receipt.transition_id,
            request_sha256=receipt.request_sha256,
            source_session_id=receipt.source_session_id,
            descendant_session_id=receipt.descendant_session_id,
            source_run_epoch=source_run_epoch,
            source_transcript_cursor=receipt.source_transcript_cursor,
            source_snapshot_cursor=source_snapshot_cursor,
            status=_PromptTransitionIntentStatus.PREPARED,
            requested_at=requested_at,
        )

    @field_validator(
        "transition_id",
        "request_sha256",
        "source_session_id",
        "descendant_session_id",
    )
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if info.field_name in {"transition_id", "request_sha256"} and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("requested_at", "descendant_created_at", "completed_at")
    @classmethod
    def validate_timestamps(cls, value: datetime | None, info) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> _PromptTransitionIntent:
        completed = self.status is _PromptTransitionIntentStatus.COMPLETED
        if completed != (self.descendant_created_at is not None and self.completed_at is not None):
            raise ValueError("Prompt-transition intent completion evidence is inconsistent.")
        return self

    def same_identity(self, other: _PromptTransitionIntent) -> bool:
        return (
            self.transition_id,
            self.request_sha256,
            self.source_session_id,
            self.descendant_session_id,
            self.source_run_epoch,
            self.source_transcript_cursor,
            self.source_snapshot_cursor,
        ) == (
            other.transition_id,
            other.request_sha256,
            other.source_session_id,
            other.descendant_session_id,
            other.source_run_epoch,
            other.source_transcript_cursor,
            other.source_snapshot_cursor,
        )


class _PromptTransitionIntentLedger(BaseModel):
    """Bounded parser and state machine for prompt-transition checkpoint authority."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    records: dict[str, _PromptTransitionIntent] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_records(self) -> _PromptTransitionIntentLedger:
        if len(self.records) > _PROMPT_ANATOMY_TRANSITION_INTENT_LIMIT:
            raise ValueError("Prompt-transition intent ledger exceeds its bounded record limit.")
        if any(key != record.transition_id for key, record in self.records.items()):
            raise ValueError("Prompt-transition intent ledger key conflicts with its record.")
        return self

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: dict[str, Any] | None,
        *,
        required: bool,
    ) -> _PromptTransitionIntentLedger:
        raw = (
            None
            if checkpoint is None
            else checkpoint.get(_PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY)
        )
        if raw is None:
            if required:
                raise RuntimeError("Prompt-anatomy transition intent disappeared.")
            return cls()
        try:
            return cls.model_validate(
                copy_json_value(raw, _PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY)
            )
        except ValidationError as exc:
            raise ValueError("Prompt-anatomy transition intent ledger is malformed.") from exc

    def store_in(self, checkpoint: dict[str, Any]) -> None:
        checkpoint[_PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY] = self.model_dump(mode="json")

    def require_exact(self, candidate: _PromptTransitionIntent) -> _PromptTransitionIntent:
        existing = self.records.get(candidate.transition_id)
        if existing is None or not existing.same_identity(candidate):
            raise RuntimeError("Prompt-anatomy transition intent conflicts with the exact request.")
        return existing

    def prepare(self, candidate: _PromptTransitionIntent) -> None:
        for record in self.records.values():
            if (
                record.descendant_session_id == candidate.descendant_session_id
                or record.request_sha256 == candidate.request_sha256
            ) and not record.same_identity(candidate):
                raise RuntimeError(
                    "Existing prompt-anatomy transition intent conflicts with the exact request."
                )
        existing = self.records.get(candidate.transition_id)
        if existing is not None:
            self.require_exact(candidate)
            return
        if len(self.records) >= _PROMPT_ANATOMY_TRANSITION_INTENT_LIMIT:
            completed = sorted(
                (
                    record
                    for record in self.records.values()
                    if record.status is _PromptTransitionIntentStatus.COMPLETED
                ),
                key=lambda record: record.completed_at or record.requested_at,
            )
            if completed:
                del self.records[completed[0].transition_id]
        if len(self.records) >= _PROMPT_ANATOMY_TRANSITION_INTENT_LIMIT:
            raise RuntimeError("Prompt-transition intent ledger has no bounded capacity.")
        self.records[candidate.transition_id] = candidate

    def complete(
        self,
        candidate: _PromptTransitionIntent,
        *,
        descendant_created_at: datetime,
        completed_at: datetime,
    ) -> None:
        existing = self.require_exact(candidate)
        if (
            existing.status is _PromptTransitionIntentStatus.COMPLETED
            and existing.descendant_created_at == descendant_created_at
        ):
            return
        self.records[candidate.transition_id] = existing.model_copy(
            update={
                "status": _PromptTransitionIntentStatus.COMPLETED,
                "descendant_created_at": descendant_created_at,
                "completed_at": completed_at,
            }
        )

    def complete_from_receipt(
        self,
        receipt: PromptAnatomyTransitionReceipt,
        *,
        descendant_created_at: datetime,
        completed_at: datetime,
    ) -> bool:
        """Complete surviving prepared authority from the durable descendant receipt."""

        existing = self.records.get(receipt.transition_id)
        if existing is None:
            return False
        if (
            existing.request_sha256,
            existing.source_session_id,
            existing.descendant_session_id,
            existing.source_transcript_cursor,
        ) != (
            receipt.request_sha256,
            receipt.source_session_id,
            receipt.descendant_session_id,
            receipt.source_transcript_cursor,
        ):
            raise RuntimeError(
                "Prompt-anatomy transition receipt conflicts with its durable intent."
            )
        if existing.status is _PromptTransitionIntentStatus.COMPLETED:
            if existing.descendant_created_at != descendant_created_at:
                raise RuntimeError(
                    "Prompt-anatomy transition completion time conflicts with its descendant."
                )
            return False
        self.records[receipt.transition_id] = existing.model_copy(
            update={
                "status": _PromptTransitionIntentStatus.COMPLETED,
                "descendant_created_at": descendant_created_at,
                "completed_at": completed_at,
            }
        )
        return True

    def has_prepared_intent(self) -> bool:
        return any(
            record.status is _PromptTransitionIntentStatus.PREPARED
            for record in self.records.values()
        )


def _reject_prepared_prompt_transition_intent(
    checkpoint: dict[str, Any] | None,
) -> None:
    ledger = _PromptTransitionIntentLedger.from_checkpoint(
        checkpoint,
        required=False,
    )
    if ledger.has_prepared_intent():
        raise RuntimeError(
            "Session has a prepared prompt-anatomy succession intent. Retry the exact "
            "fork request before continuing or compacting the source session."
        )


async def _reconcile_committed_prompt_transition_intents(
    *,
    session_store: SessionStore,
    source_session_id: str,
    checkpoint: dict[str, Any] | None,
    clock: Callable[[], datetime],
    persist: bool = True,
) -> dict[str, Any] | None:
    """Complete prepared intents whose descendants already prove the durable effect."""

    ledger = _PromptTransitionIntentLedger.from_checkpoint(
        checkpoint,
        required=False,
    )
    committed_receipts: list[tuple[PromptAnatomyTransitionReceipt, datetime]] = []
    for record in ledger.records.values():
        if record.status is not _PromptTransitionIntentStatus.PREPARED:
            continue
        descendant = await session_store.load(record.descendant_session_id)
        if descendant is None:
            continue
        receipt = session_prompt_anatomy_transition(descendant)
        if receipt is None:
            raise RuntimeError(
                "Prepared prompt-anatomy descendant has no exact transition receipt."
            )
        committed_receipts.append((receipt, descendant.created_at))
    if not committed_receipts:
        return checkpoint

    reconciled_checkpoint: dict[str, Any] | None = None

    def reconciled_value(current_checkpoint: dict[str, Any] | None) -> dict[str, Any]:
        if current_checkpoint is None:
            raise RuntimeError("Prompt-anatomy transition intent disappeared.")
        updated = copy_durable_record(current_checkpoint, "checkpoint")
        current_ledger = _PromptTransitionIntentLedger.from_checkpoint(
            updated,
            required=True,
        )
        for receipt, descendant_created_at in committed_receipts:
            current_ledger.complete_from_receipt(
                receipt,
                descendant_created_at=descendant_created_at,
                completed_at=clock(),
            )
        current_ledger.store_in(updated)
        return updated

    if not persist:
        return reconciled_value(checkpoint)

    def reconcile(
        _current_source: Session,
        current_checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        nonlocal reconciled_checkpoint
        reconciled_checkpoint = reconciled_value(current_checkpoint)
        return reconciled_checkpoint

    await session_store.publish_checkpoint_and_events(
        source_session_id,
        checkpoint_transform=reconcile,
        events=[],
    )
    return reconciled_checkpoint
