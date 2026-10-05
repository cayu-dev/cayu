"""Typed, bounded external-wait records. Values carry identity, not authority."""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.sessions._session_continuation import ContinuationPreparation, ContinuationService

EXTERNAL_WAIT_VERSION = 1
EXTERNAL_WAIT_PAYLOAD_BYTES = 64 * 1024
# Each canonical payload is embedded as a JSON string in the record. Reserve
# encoding headroom for both maximum-sized values plus bounded identity/evidence.
EXTERNAL_WAIT_RECORD_BYTES = 512 * 1024
EXTERNAL_WAIT_CORRELATIONS = 4096
EXTERNAL_WAIT_SCOPE_BYTES = 32 * 1024 * 1024
EXTERNAL_WAIT_DELIVERIES = 32
EXTERNAL_WAIT_PAGE_SIZE = 32
EXTERNAL_WAIT_PAGE_LIMIT = 256
EXTERNAL_WAIT_RETENTION_SECONDS = 24 * 60 * 60
EXTERNAL_WAIT_MAX_RETENTION_SECONDS = 30 * 24 * 60 * 60
EXTERNAL_WAIT_MAX_HORIZON_SECONDS = 366 * 24 * 60 * 60

Identifier = Annotated[StrictStr, Field(min_length=1, max_length=256)]
Digest = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
OutcomeKind = Literal["event", "timeout", "cancelled", "unavailable"]


class ExternalWaitConflict(ValueError):
    """An exact operation conflicts with retained evidence."""


class ExternalWaitUnavailable(RuntimeError):
    """No positive receiving evidence is available; this is not exclusion."""


class ExternalWaitCapacityExceeded(RuntimeError):
    """Optional work cannot fit without consuming reserved settlement capacity."""


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)

    @field_validator("*", mode="after")
    @classmethod
    def clean_strings(cls, value: object) -> object:
        if type(value) is str:
            require_durable_clean_nonblank(value, "external wait value")
        return value


class ExternalWaitScope(_Value):
    application_scope: Identifier
    generation: StrictInt = Field(ge=1, le=9007199254740991)


class ExternalWaitLimits(_Value):
    correlations: StrictInt = Field(default=EXTERNAL_WAIT_CORRELATIONS, ge=1, le=4096)
    retained_bytes: StrictInt = Field(default=EXTERNAL_WAIT_SCOPE_BYTES, ge=131072, le=33554432)
    deliveries: StrictInt = Field(default=EXTERNAL_WAIT_DELIVERIES, ge=1, le=32)
    payload_bytes: StrictInt = Field(default=EXTERNAL_WAIT_PAYLOAD_BYTES, ge=1, le=65536)
    projection_bytes: StrictInt = Field(default=EXTERNAL_WAIT_PAYLOAD_BYTES, ge=1, le=65536)


class ExternalCorrelationRequest(_Value):
    scope: ExternalWaitScope
    correlation_key: Identifier
    source: Identifier
    # Absolute instants are frozen by the caller once, never recomputed on retry.
    deadline: datetime | None = None
    early_event_retention_seconds: StrictInt = Field(
        default=EXTERNAL_WAIT_RETENTION_SECONDS, ge=1, le=EXTERNAL_WAIT_MAX_RETENTION_SECONDS
    )

    @field_validator("deadline")
    @classmethod
    def aware_deadline(cls, value: datetime | None) -> datetime | None:
        from datetime import UTC

        if value is not None:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("External wait deadline must be timezone-aware.")
            return value.astimezone(UTC)
        return None


class ExternalCorrelation(_Value):
    request: ExternalCorrelationRequest
    incarnation: Identifier
    created_at_ms: StrictInt = Field(ge=0)
    early_event_expires_at_ms: StrictInt = Field(ge=0)
    limits: ExternalWaitLimits


class ExternalWaitRegistration(_Value):
    correlation: ExternalCorrelation
    operation_key: Identifier
    projector_id: Identifier
    projector_version: StrictInt = Field(ge=1, le=9007199254740991)


class ExternalWaitRetirementRequest(_Value):
    scope: ExternalWaitScope
    operation_key: Identifier
    limits: ExternalWaitLimits


class ExternalWaitRetirement(_Value):
    request: ExternalWaitRetirementRequest
    retired_at_ms: StrictInt = Field(ge=0, le=9007199254740991)
    correlation_count: StrictInt = Field(ge=0, le=EXTERNAL_WAIT_CORRELATIONS)
    records_sha256: Digest


class ExternalWaitPruneResult(_Value):
    retirement: ExternalWaitRetirement
    removed: StrictInt = Field(ge=0, le=EXTERNAL_WAIT_PAGE_LIMIT)
    remaining: StrictInt = Field(ge=0, le=EXTERNAL_WAIT_CORRELATIONS)


class ExternalWaitTimer(_Value):
    """Immutable managed-task hint; never authority to elect or execute a wait."""

    schema_version: Literal[1] = 1
    scheduler_id: Identifier
    correlation: ExternalCorrelation
    registration_sha256: Digest
    task_id: Identifier


def external_wait_timer(
    registration: ExternalWaitRegistration, scheduler_id: str
) -> ExternalWaitTimer:
    if registration.correlation.request.deadline is None:
        raise ValueError("An event-only wait needs no timer.")
    commitment = external_wait_digest(registration)
    identity = sha256(
        canonical_durable_json_bytes(
            {"scheduler": scheduler_id, "registration": commitment, "version": 1}, "external timer"
        )
    ).hexdigest()
    return ExternalWaitTimer(
        scheduler_id=scheduler_id,
        correlation=registration.correlation,
        registration_sha256=commitment,
        task_id="external-wait-" + identity,
    )


class ExternalEventDelivery(_Value):
    correlation: ExternalCorrelation
    delivery_id: Identifier
    # Store only canonical, validated JSON. It never contains executable objects.
    payload_json: StrictStr

    @field_validator("payload_json")
    @classmethod
    def bounded_payload(cls, value: str) -> str:
        return canonical_payload(value, EXTERNAL_WAIT_PAYLOAD_BYTES)


class ExternalDeliveryReceipt(_Value):
    delivery_id: Identifier
    content_sha256: Digest
    accepted_at_ms: StrictInt = Field(ge=0)
    disposition: Literal["accepted", "additional", "late", "settled", "unavailable"]


class ExternalWaitOutcome(_Value):
    kind: OutcomeKind
    selected_at_ms: StrictInt = Field(ge=0)
    delivery_id: Identifier | None = None
    payload_json: StrictStr | None = None
    content_sha256: Digest | None = None

    @model_validator(mode="after")
    def coherent_payload(self) -> ExternalWaitOutcome:
        values = (self.delivery_id, self.payload_json, self.content_sha256)
        if self.kind == "event":
            if any(value is None for value in values):
                raise ValueError("Event outcomes require exact delivery evidence.")
            assert self.payload_json is not None
            if (
                canonical_payload(self.payload_json, EXTERNAL_WAIT_PAYLOAD_BYTES)
                != self.payload_json
            ):
                raise ValueError("Event outcome payload is not canonical.")
        elif any(value is not None for value in values):
            raise ValueError("Non-event outcomes cannot contain delivery evidence.")
        return self


class ExternalWaitExecutionIntent(_Value):
    """Resolved runtime preparation, not a new execution permission."""

    mode: Literal["run", "resume"]
    request_sha256: Digest
    source_request_sha256: Digest | None = None
    profile_sha256: Digest
    session_id: Identifier
    expected_session_instance_id: Identifier | None = None
    expected_run_epoch: StrictInt | None = Field(default=None, ge=0, le=9007199254740991)
    admission_sha256: Digest | None = None

    @model_validator(mode="after")
    def exact_destination(self) -> ExternalWaitExecutionIntent:
        if self.mode == "run":
            if self.admission_sha256 is not None:
                raise ValueError("Initial execution cannot carry resume admission evidence.")
            if self.expected_session_instance_id is not None or self.expected_run_epoch is not None:
                raise ValueError(
                    "Initial external execution cannot select an existing incarnation."
                )
        elif (
            self.expected_session_instance_id is None
            or self.expected_run_epoch is None
            or self.admission_sha256 is None
        ):
            raise ValueError("External resume requires the exact existing session frontier.")
        return self


class ExternalWaitExecution(_Value):
    intent: ExternalWaitExecutionIntent
    preparation_owner_id: Identifier
    session_instance_id: Identifier
    interaction_id: Identifier
    interaction_event_id: Identifier
    prepared_at_ms: StrictInt = Field(ge=0, le=9007199254740991)

    @model_validator(mode="after")
    def matching_incarnation(self) -> ExternalWaitExecution:
        if (
            self.intent.mode == "resume"
            and self.session_instance_id != self.intent.expected_session_instance_id
        ):
            raise ValueError("External resume incarnation conflicts with its intent.")
        return self


class ExternalWaitExecutionRetirement(_Value):
    """Exact host request for native retirement, not proof of execution exclusion."""

    operation_key: Identifier
    prepared_at_ms: StrictInt = Field(ge=0, le=9007199254740991)


class ExternalWaitRecord(_Value):
    schema_version: Literal[1] = 1
    correlation: ExternalCorrelation
    registration: ExternalWaitRegistration | None = None
    revision: StrictInt = Field(ge=1, le=9007199254740991)
    deliveries: tuple[ExternalDeliveryReceipt, ...] = Field(default=(), max_length=32)
    early_event: ExternalWaitOutcome | None = None
    outcome: ExternalWaitOutcome | None = None
    cancel_key: Identifier | None = None
    execution: ExternalWaitExecution | None = None
    execution_excluded: StrictBool = False
    execution_retirement: ExternalWaitExecutionRetirement | None = None
    # Runtime-owned handoffs are committed independently of election. Their
    # durable stage is discoverable even when the original observer disappears.
    continuation: ContinuationPreparation | None = None
    service: ContinuationService | None = None
    service_stage_id: Identifier | None = None
    projection_json: StrictStr | None = None
    timer: ExternalWaitTimer | None = None
    timer_published: StrictBool = False
    handoff: Literal["unbound", "pending", "settled", "excluded"] = "unbound"
    handoff_receipt_sha256: Digest | None = None
    retirement_complete: StrictBool = False

    @property
    def pending_handoff(self) -> bool:
        return (
            (
                self.execution is not None
                and self.continuation is None
                and not self.execution_excluded
            )
            or self.handoff == "pending"
            or (self.handoff == "excluded" and not self.retirement_complete)
        )

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != EXTERNAL_WAIT_VERSION:
            raise ValueError("Unsupported external wait record version.")
        return value

    @model_validator(mode="after")
    def coherent_record(self) -> ExternalWaitRecord:
        if self.execution_retirement is not None and (
            self.continuation is None or self.outcome is None
        ):
            raise ValueError("External retirement requires a bound terminal wait.")
        if (self.service is None) != (self.service_stage_id is None):
            raise ValueError("External service requires its retained model-stage identity.")
        if self.timer is not None:
            if self.registration is None or self.timer != external_wait_timer(
                self.registration, self.timer.scheduler_id
            ):
                raise ValueError("External timer conflicts with its exact registration.")
        elif self.timer_published:
            raise ValueError("External timer publication has no retained intent.")
        if self.execution_excluded and (self.execution is None or self.continuation is not None):
            raise ValueError("External execution exclusion requires an unbound preparation.")
        if self.retirement_complete and self.handoff != "excluded":
            raise ValueError("External retirement completion requires excluded handoff evidence.")
        if (self.handoff in {"settled", "excluded"}) != (self.handoff_receipt_sha256 is not None):
            raise ValueError("External terminal handoff lacks its exact native evidence.")
        if self.registration is not None and self.registration.correlation != self.correlation:
            raise ValueError("External registration does not match its correlation.")
        if self.execution is not None and self.registration is None:
            raise ValueError("External execution has no registered wait.")
        if self.continuation is None:
            if self.handoff != "unbound":
                raise ValueError("External handoff has no native continuation binding.")
        else:
            if self.registration is None or self.handoff == "unbound":
                raise ValueError("External continuation lacks its registered handoff.")
            from cayu.sessions._external_wait_records import require_binding_identity

            try:
                require_binding_identity(self.registration, self.continuation)
            except PermissionError:
                raise ValueError("External continuation registration identity conflicts.") from None
            if self.execution is not None:
                execution = self.execution
                ticket = self.continuation.intent
                epoch = 1
                if execution.intent.mode == "resume":
                    assert execution.intent.expected_run_epoch is not None
                    epoch = execution.intent.expected_run_epoch + 1
                if (
                    ticket.session_id != execution.intent.session_id
                    or ticket.session_instance_id != execution.session_instance_id
                    or ticket.interaction_id != execution.interaction_id
                    or ticket.writer_generation != epoch
                ):
                    raise ValueError("External continuation conflicts with its retained execution.")
        if len(self.deliveries) > self.correlation.limits.deliveries:
            raise ValueError("External delivery count exceeds its retained limit.")
        receipts = {receipt.delivery_id: receipt for receipt in self.deliveries}
        if len(receipts) != len(self.deliveries):
            raise ValueError("External delivery identities must be unique.")
        if sum(receipt.disposition == "accepted" for receipt in self.deliveries) > 1:
            raise ValueError("An external correlation cannot accept multiple winning deliveries.")
        if self.outcome is not None:
            if self.outcome.kind == "timeout":
                deadline = self.correlation.request.deadline
                if deadline is None or self.outcome.selected_at_ms != int(
                    deadline.timestamp() * 1000
                ):
                    raise ValueError("External timeout lacks its exact deadline.")
            elif self.outcome.kind == "cancelled" and self.cancel_key is None:
                raise ValueError("External cancellation lacks its exact operation key.")
        if self.early_event is not None and (
            self.early_event.kind != "event"
            or self.registration is not None
            or self.outcome is not None
        ):
            raise ValueError("Early external delivery state conflicts.")
        for event in (self.early_event, self.outcome):
            if event is None or event.kind != "event":
                continue
            assert event.delivery_id is not None
            receipt = receipts.get(event.delivery_id)
            if receipt is None or (
                receipt.disposition != "accepted"
                or receipt.accepted_at_ms != event.selected_at_ms
                or receipt.content_sha256 != event.content_sha256
            ):
                raise ValueError("External event lacks its exact acceptance receipt.")
            assert event.payload_json is not None
            canonical_payload(event.payload_json, self.correlation.limits.payload_bytes)
            deadline = self.correlation.request.deadline
            if deadline is not None and event.selected_at_ms >= int(deadline.timestamp() * 1000):
                raise ValueError("External event was not accepted before its deadline.")
        if self.projection_json is not None:
            if (
                self.registration is None
                or self.outcome is None
                or self.outcome.kind not in {"event", "timeout"}
            ):
                raise ValueError("External projection has no eligible registered outcome.")
            canonical_payload(self.projection_json, self.correlation.limits.projection_bytes)
        if self.service is not None:
            from cayu.sessions._external_wait_records import elected_external_latch
            from cayu.sessions._session_continuation import (
                require_latch_identity,
                require_ticket_identity,
            )

            if self.continuation is None:
                raise ValueError("External service lacks a continuation binding.")
            require_ticket_identity(self.continuation.intent, self.service.ticket)
            # A later exclusion changes lifecycle state, not the immutable
            # projected service that must remain reconstructable for diagnosis.
            projected = (
                self
                if self.handoff != "excluded"
                else self.model_copy(update={"handoff": "pending"})
            )
            require_latch_identity(elected_external_latch(projected), self.service.latch)
        return self

    @property
    def reserved_bytes(self) -> int:
        return self.correlation.limits.payload_bytes + self.correlation.limits.projection_bytes


def canonical_payload(value: str, limit: int) -> str:
    if type(value) is not str or len(value.encode("utf-8")) > limit:
        raise ValueError("External wait payload exceeds its byte limit.")
    try:
        raw = json.loads(value, object_pairs_hook=_unique_object)
        encoded = canonical_durable_json_bytes(raw, "external wait payload")
    except (ValueError, TypeError, RecursionError):
        raise ValueError("External wait payload must be bounded durable JSON.") from None
    if len(encoded) > limit:
        raise ValueError("External wait payload exceeds its byte limit.")
    return encoded.decode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate external wait payload key.")
        result[key] = value
    return result


def external_wait_digest(value: _Value) -> str:
    return sha256(
        canonical_durable_json_bytes(value.model_dump(mode="json"), "external wait")
    ).hexdigest()


def encode_record(record: ExternalWaitRecord) -> str:
    value = record.model_dump_json()
    if len(value.encode("utf-8")) > EXTERNAL_WAIT_RECORD_BYTES:
        raise ExternalWaitCapacityExceeded("External wait record exceeds its durable bound.")
    return value
