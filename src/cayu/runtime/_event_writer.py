from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import suppress
from itertools import islice
from typing import cast

from cayu.core.events import Event, EventType
from cayu.core.tools import _POLICY_DENIAL_TRUNCATION_MARKER
from cayu.runtime import _tool_results as tool_results
from cayu.runtime.budgets import BudgetStore
from cayu.runtime.event_sinks import EventSink
from cayu.runtime.model_steps import StepClassificationType
from cayu.runtime.sessions import (
    EventQuery,
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectStatus,
    SessionStore,
    attribute_event_to_current_interaction,
    attribute_events_to_current_interaction,
    portable_persisted_event_side_effect_error,
)
from cayu.vaults.redaction import SecretRedactor

_PERSISTED_SIDE_EFFECT_MAX_ATTEMPTS = 3
_PERSISTED_SIDE_EFFECT_RETRY_DELAY_SECONDS = 30.0
_MAX_AGGREGATED_FAILURES = 16
_MAX_EXCEPTION_NOTES = 16
_MODEL_STEP_CLASSIFICATION_TYPES = frozenset(item.value for item in StepClassificationType)
_EVENT_CONTROL_STRING_FIELDS = frozenset(
    {
        "action",
        "agent_name",
        "approval_id",
        "budget_limit_id",
        "checkpoint",
        "decision",
        "denied_by",
        "effect",
        "environment_name",
        "error_type",
        "event_id",
        "event_type",
        "finish_reason",
        "idempotency_key",
        "input_id",
        "pending_action_kind",
        "interruption_type",
        "is_error",
        "model_attempt_id",
        "model_step_id",
        "model",
        "operation",
        "outcome",
        "phase",
        "policy",
        "policy_name",
        "provider",
        "reservation_id",
        "role",
        "scope",
        "session_id",
        "source",
        "start_event_id",
        "started_at",
        "status",
        "completed_at",
        "step_classification",
        "strategy",
        "terminal_outcome",
        "task_id",
        "tool_call_id",
        "tool_name",
        "tool_round_id",
        "type",
        "workflow_name",
    }
)
_EVENT_IDENTITY_STRING_FIELDS = frozenset(
    {
        "agent_name",
        "environment_name",
        "tool_name",
        "workflow_name",
    }
)
_EVENT_SECRET_FREE_AUTHORITY_FIELDS = frozenset(
    {
        "approval_id",
        "budget_limit_id",
        "checkpoint",
        "event_id",
        "input_id",
        "interaction_id",
        "model_attempt_id",
        "model_step_id",
        "model",
        "policy_name",
        "provider",
        "reservation_id",
        "session_id",
        "task_id",
        "tool_call_id",
        "tool_round_id",
    }
)
_EVENT_PUBLIC_STRUCTURE_KEYS = _EVENT_CONTROL_STRING_FIELDS | {
    "active_duration_ms",
    "arguments",
    "cache",
    "cached_input_tokens",
    "completion",
    "content",
    "details",
    "error",
    "estimated_tool_schema_input_tokens",
    "failures",
    "interaction_ids",
    "input_tokens",
    "metadata",
    "model_step_count",
    "models",
    "provider_name",
    "provider_names",
    "raw_finish_reason",
    "reason",
    "result",
    "reasoning_output_tokens",
    "read_tokens",
    "result_transcript_end",
    "result_transcript_start",
    "source_transcript_end",
    "source_transcript_start",
    "start_event_sequence",
    "structured",
    "token_usage",
    "total_tokens",
    "tool_call_count",
    "uncached_input_tokens",
    "usage",
    "wall_duration_ms",
    "write_1h_tokens",
    "write_5m_tokens",
    "write_tokens",
    "write_unknown_ttl_tokens",
    "output_tokens",
}

logger = logging.getLogger(__name__)
_TOOL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_REQUESTED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.TOOL_CALL_APPROVAL_EXPIRED,
    }
)
_TOOL_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
    }
)


class RuntimeEventWriter:
    """Persist runtime events and fan them out to configured sinks."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        budget_store: BudgetStore,
        event_sinks: Iterable[EventSink],
        secret_redactor: SecretRedactor | None = None,
    ) -> None:
        if secret_redactor is not None and not isinstance(secret_redactor, SecretRedactor):
            raise TypeError("secret_redactor must be a SecretRedactor.")
        self._session_store = session_store
        self._budget_store = budget_store
        self._event_sinks = tuple(event_sinks)
        self._secret_redactor = secret_redactor or SecretRedactor()

    async def emit(self, event: Event) -> Event:
        event = self.prepare(attribute_event_to_current_interaction(event))
        await self._session_store.append_event(event.session_id, event)
        claim = await self._session_store.claim_persisted_event_side_effect(
            session_id=event.session_id,
            event_id=event.id,
        )
        if claim is None:
            await self._handle_unclaimed_persisted_side_effect(event)
            return event.model_copy(deep=True)
        delivered_event, _ = await self._deliver_persisted_side_effect_claim(claim)
        return delivered_event

    async def persist(self, event: Event) -> Event:
        """Commit an event to the durable side-effect handoff without delivering it.

        This is reserved for failure evidence that must become durable before
        caller cancellation is redelivered. The store atomically creates the
        pending side-effect record, so normal recovery retains ownership of
        budget and sink delivery.
        """

        event = self.prepare(attribute_event_to_current_interaction(event))
        await self._session_store.append_event(event.session_id, event)
        return event.model_copy(deep=True)

    async def persist_exact_replay(self, event: Event) -> Event:
        """Persist one stable event or verify its exact acknowledgement-loss replay."""

        prepared = self.prepare(event)
        try:
            await self._session_store.append_event(prepared.session_id, prepared)
        except Exception as append_error:
            try:
                records = await self._session_store.query_events(
                    EventQuery(
                        session_id=prepared.session_id,
                        event_id=prepared.id,
                        limit=1,
                    )
                )
            except Exception as verification_error:
                append_error.add_note(
                    "Exact event replay verification also failed: "
                    f"{type(verification_error).__name__}: {verification_error}"
                )
                raise append_error from verification_error
            if len(records) != 1 or records[0].event != prepared:
                raise append_error
        return prepared.model_copy(deep=True)

    async def is_persisted(self, event: Event) -> bool:
        """Return whether this exact event reached the durable event handoff."""

        records = await self._session_store.query_events(
            EventQuery(session_id=event.session_id, event_id=event.id, limit=1)
        )
        return any(record.event.id == event.id for record in records)

    async def emit_many(self, session_id: str, events: list[Event]) -> list[Event]:
        """Persist and fan out a defensive copy of one event batch.

        Batch events use the same durable budget and sink handoff as ``emit``.
        This matters for store-atomic publications that include cost-bearing
        events, such as explicit compaction recovery.
        """
        copied_events = await self.persist_many(session_id, events)
        await self.fan_out_persisted(copied_events)
        return copied_events

    async def persist_many(self, session_id: str, events: list[Event]) -> list[Event]:
        """Persist a defensive event batch without delivering its side effects.

        Callers that must distinguish a failed durable append from failed
        post-commit delivery can persist first and then call
        :meth:`fan_out_persisted`. The store-owned side-effect handoff keeps a
        committed batch recoverable if fan-out is interrupted or fails.
        """
        if type(events) is not list:
            raise TypeError("Runtime events must be a list.")
        copied_events: list[Event] = []
        for event in attribute_events_to_current_interaction(events):
            if type(event) is not Event:
                raise TypeError("Runtime events must be Event instances.")
            if event.session_id != session_id:
                raise ValueError("Event session_id does not match target session.")
            copied_events.append(self.prepare(event))

        await self._session_store.append_events(session_id, copied_events)
        return copied_events

    async def fan_out_persisted(self, events: list[Event]) -> list[Event]:
        """Apply budget/sink side effects after a store-owned atomic publication."""

        if type(events) is not list:
            raise TypeError("Runtime events must be a list.")
        copied_events: list[Event] = []
        for event in events:
            if type(event) is not Event:
                raise TypeError("Runtime events must be Event instances.")
            copied = self.prepare(event)
            claim = await self._session_store.claim_persisted_event_side_effect(
                session_id=copied.session_id,
                event_id=copied.id,
            )
            if claim is None:
                await self._handle_unclaimed_persisted_side_effect(copied)
            else:
                await self._deliver_persisted_side_effect_claim(claim)
            copied_events.append(copied)
        return copied_events

    async def recover_persisted_side_effects(self, *, limit: int = 100) -> list[Event]:
        """Deliver a bounded batch of committed event side effects after a crash."""

        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000.")
        candidates = await self._session_store.list_persisted_event_side_effect_deliveries(
            statuses={
                PersistedEventSideEffectStatus.PENDING,
                PersistedEventSideEffectStatus.FAILED,
                PersistedEventSideEffectStatus.LEASED,
            },
            claimable_only=True,
            limit=limit,
        )
        recovered: list[Event] = []
        for candidate in candidates:
            claim = await self._session_store.claim_persisted_event_side_effect(
                session_id=candidate.session_id,
                event_id=candidate.event_id,
            )
            if claim is None:
                continue
            try:
                event, delivered = await self._deliver_persisted_side_effect_claim(claim)
            except Exception as exc:
                logger.error(
                    "Persisted event side-effect recovery failed: "
                    "session_id=%s event_id=%s event_type=%s error_type=%s",
                    claim.session_id,
                    claim.event_id,
                    claim.event.type,
                    _object_type_name(exc),
                )
                continue
            if delivered:
                recovered.append(event)
        return recovered

    async def _deliver_persisted_side_effect_claim(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> tuple[Event, bool]:
        try:
            event = self.prepare(claim.event)
            await self._forward_budget_event_if_required(event)
        except Exception as exc:
            try:
                await self._mark_claim_failed(
                    claim,
                    error=_exception_summary(exc, redactor=self._secret_redactor),
                )
            except Exception as bookkeeping_exc:
                _add_exception_note(
                    exc,
                    "Persisted event side-effect failure bookkeeping also failed: "
                    f"{_exception_summary(bookkeeping_exc, redactor=self._secret_redactor)}",
                )
            raise
        sink_failures = await self._emit_to_sinks(event)
        if sink_failures:
            failure_summary = _failure_summary(
                sink_failures,
                redactor=self._secret_redactor,
            )
            try:
                await self._mark_claim_failed(
                    claim,
                    error=failure_summary,
                )
            except Exception as bookkeeping_exc:
                primary_failure = sink_failures[0]
                _add_exception_note(
                    primary_failure,
                    "Persisted event side-effect failure bookkeeping also failed: "
                    f"{_exception_summary(bookkeeping_exc, redactor=self._secret_redactor)}",
                )
                if len(sink_failures) > 1:
                    _add_exception_note(
                        primary_failure,
                        "Additional persisted event sink failures: "
                        + _failure_summary(
                            sink_failures,
                            start=1,
                            redactor=self._secret_redactor,
                        ),
                    )
                raise primary_failure from bookkeeping_exc
            return event, False
        try:
            await self._session_store.mark_persisted_event_side_effect_delivered(claim)
        except PersistedEventSideEffectClaimLost:
            return event, False
        except Exception as exc:
            logger.error(
                "Persisted event side-effect delivery acknowledgement failed; "
                "leaving the durable claim for recovery: "
                "session_id=%s event_id=%s event_type=%s error_type=%s",
                claim.session_id,
                claim.event_id,
                claim.event.type,
                _object_type_name(exc),
            )
            return event, False
        return event, True

    async def _forward_budget_event_if_required(self, event: Event) -> None:
        if event.type == EventType.MODEL_COMPLETED:
            await self._budget_store.append_event(event.model_copy(deep=True))

    async def _handle_unclaimed_persisted_side_effect(self, event: Event) -> None:
        delivery = await self._session_store.get_persisted_event_side_effect_delivery(
            session_id=event.session_id,
            event_id=event.id,
        )
        if delivery is None:
            raise RuntimeError("Persisted event side-effect delivery was not found.")
        if delivery.status in {
            PersistedEventSideEffectStatus.PENDING,
            PersistedEventSideEffectStatus.LEASED,
        }:
            # PostgreSQL can expose the pre-claim PENDING row while another
            # transaction owns its update. This idempotent fallback closes the
            # accounting race; the durable handoff still owns delivery/retry.
            try:
                await self._forward_budget_event_if_required(event)
            except Exception:
                return
            return
        if delivery.status in {
            PersistedEventSideEffectStatus.FAILED,
            PersistedEventSideEffectStatus.DELIVERED,
            PersistedEventSideEffectStatus.DEAD_LETTERED,
        }:
            return
        raise RuntimeError(
            "Persisted event side-effect claim unexpectedly returned no claim "
            f"for {delivery.status.value} delivery."
        )

    async def _mark_claim_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
    ) -> PersistedEventSideEffectDelivery | None:
        try:
            delivery = await self._session_store.mark_persisted_event_side_effect_failed(
                claim,
                error=portable_persisted_event_side_effect_error(
                    self._secret_redactor.redact_text(error)
                ),
                max_attempts=_PERSISTED_SIDE_EFFECT_MAX_ATTEMPTS,
                retry_delay_seconds=_PERSISTED_SIDE_EFFECT_RETRY_DELAY_SECONDS,
            )
        except PersistedEventSideEffectClaimLost:
            return None
        if delivery.status is PersistedEventSideEffectStatus.DEAD_LETTERED:
            logger.error(
                "Persisted event side effect dead-lettered: "
                "session_id=%s event_id=%s event_type=%s attempts=%s budget_effect=%s",
                claim.session_id,
                claim.event_id,
                claim.event.type,
                delivery.attempts,
                claim.event.type == EventType.MODEL_COMPLETED,
            )
        return delivery

    async def _emit_to_sinks(self, event: Event) -> list[Exception]:
        failures: list[Exception] = []
        for sink in self._event_sinks:
            try:
                await sink.emit(event.model_copy(deep=True))
            except Exception as exc:
                try:
                    await self._session_store.append_event(
                        event.session_id,
                        self.prepare(
                            Event(
                                type=EventType.RUNTIME_SINK_FAILED,
                                session_id=event.session_id,
                                interaction_id=event.interaction_id,
                                agent_name=event.agent_name,
                                environment_name=event.environment_name,
                                payload={
                                    "sink": _object_type_name(sink),
                                    "error": _exception_message(
                                        exc,
                                        redactor=self._secret_redactor,
                                    ),
                                    "error_type": _object_type_name(exc),
                                    "event_id": event.id,
                                    "event_type": str(event.type),
                                },
                            )
                        ),
                    )
                except Exception as diagnostic_exc:
                    _add_exception_note(
                        exc,
                        "runtime.sink.failed persistence failed: "
                        f"{_exception_summary(diagnostic_exc, redactor=self._secret_redactor)}",
                    )
                failures.append(exc)
        return failures

    def prepare(self, event: Event) -> Event:
        """Return the event shape safe for durable or external publication."""

        return prepare_runtime_event(
            attribute_event_to_current_interaction(event),
            redactor=self._secret_redactor,
        )

    def prepare_exact_replay(self, event: Event) -> Event:
        """Validate that publication policy preserves an immutable replay event."""

        prepared = self.prepare(event)
        if prepared != event:
            raise ValueError("Exact replay event requires redaction before publication.")
        return prepared

    def prepare_many(self, events: list[Event]) -> list[Event]:
        """Return publication-safe defensive copies of one event batch."""

        if type(events) is not list:
            raise TypeError("Runtime events must be a list.")
        return [self.prepare(event) for event in events]


def prepare_runtime_event(
    event: Event,
    *,
    redactor: SecretRedactor,
    reject_authority_secrets: bool = True,
) -> Event:
    """Return one event safe for durable or external publication."""

    if type(event) is not Event:
        raise TypeError("Runtime events must be Event instances.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    if type(reject_authority_secrets) is not bool:
        raise TypeError("reject_authority_secrets must be a bool.")
    if reject_authority_secrets:
        authority_fields = [
            ("session_id", event.session_id),
            ("event_id", event.id),
        ]
        if event.interaction_id is not None:
            authority_fields.append(("interaction_id", event.interaction_id))
        if not isinstance(event.type, EventType):
            authority_fields.append(("event_type", str(event.type)))
        for field_name, value in authority_fields:
            if redactor.redact_text(value) != value:
                raise ValueError(
                    f"event.{field_name} contains a workload secret and cannot be "
                    "used as durable event authority."
                )
    runtime_controls = (
        tool_results.runtime_tool_event_controls(
            event.payload,
            include_terminal_controls=event.type in _TOOL_TERMINAL_EVENT_TYPES,
        )
        if event.type in _TOOL_EVENT_TYPES
        else {}
    )
    if reject_authority_secrets:
        for field_name, value in runtime_controls.items():
            if (
                field_name in _EVENT_SECRET_FREE_AUTHORITY_FIELDS
                and type(value) is str
                and redactor.redact_text(value) != value
            ):
                raise ValueError(
                    f"event.payload.{field_name} contains a workload secret and cannot "
                    "be used as durable event authority."
                )
    runtime_controls.pop("tool_name", None)
    redactor.require_no_secret_keys(
        event.payload,
        field_name="event.payload",
        preserve_keys=_EVENT_PUBLIC_STRUCTURE_KEYS,
        untrusted_container_keys={
            "arguments",
            "details",
            "metadata",
            "provider_state",
        },
        match_short_substrings=True,
    )
    if reject_authority_secrets:
        for field_name in _EVENT_SECRET_FREE_AUTHORITY_FIELDS:
            value = event.payload.get(field_name)
            if type(value) is str and redactor.redact_text(value) != value:
                raise ValueError(
                    f"event.payload.{field_name} contains a workload secret and cannot "
                    "be used as durable event authority."
                )
    # Identity values are descriptive event data, not protocol discriminators.
    # Redact them in payload duplicates just as we do on the event envelope.
    control_fields = _EVENT_CONTROL_STRING_FIELDS - _EVENT_IDENTITY_STRING_FIELDS
    control_fields = control_fields | runtime_controls.keys()
    if not reject_authority_secrets:
        control_fields = control_fields - _EVENT_SECRET_FREE_AUTHORITY_FIELDS
    redacted_payload = redactor.redact_json_values(
        event.payload,
        preserve_string_fields=control_fields,
    )
    redacted_payload.update(runtime_controls)
    if event.type == EventType.MODEL_COMPLETED:
        classification = event.payload.get("step_classification")
        redacted_classification = redacted_payload.get("step_classification")
        classification_type = classification.get("type") if type(classification) is dict else None
        if (
            type(redacted_classification) is dict
            and type(classification_type) is str
            and classification_type in _MODEL_STEP_CLASSIFICATION_TYPES
        ):
            # The discriminator is runtime-owned protocol state. Other nested
            # strings, including the explanatory reason, remain redacted.
            redacted_classification["type"] = classification_type
    terminal_controls = {
        key: value
        for key, value in runtime_controls.items()
        if key
        in {
            "terminal_outcome",
            "tool_effect",
            "outcome_unknown",
            "manual_reconciliation_required",
            "durable_value_error_code",
            "durable_value_error_path",
        }
    }
    if terminal_controls:
        result = redacted_payload.get("result")
        if type(result) is dict:
            structured = result.get("structured")
            if type(structured) is dict:
                structured.update(terminal_controls)
    if event.type == EventType.TOOL_CALL_BLOCKED and "denied_by" in event.payload:
        _restore_policy_denial_truncation_markers(
            original=event.payload,
            redacted=redacted_payload,
            redactor=redactor,
        )
    redacted_identities = {
        field_name: (None if value is None else redactor.redact_text(value))
        for field_name, value in (
            ("agent_name", event.agent_name),
            ("environment_name", event.environment_name),
            ("workflow_name", event.workflow_name),
            ("tool_name", event.tool_name),
        )
    }
    if not reject_authority_secrets:
        redacted_identities["session_id"] = redactor.redact_text(event.session_id)
    return event.model_copy(
        update={"payload": redacted_payload, **redacted_identities},
        deep=True,
    )


def _restore_policy_denial_truncation_markers(
    *,
    original: dict[str, object],
    redacted: dict[str, object],
    redactor: SecretRedactor,
) -> None:
    """Keep the public bound marker stable while still scrubbing its prefix."""

    reason = original.get("reason")
    if type(reason) is str:
        redacted["reason"] = _redact_policy_denial_text(reason, redactor=redactor)
    original_result = original.get("result")
    redacted_result = redacted.get("result")
    if type(original_result) is not dict or type(redacted_result) is not dict:
        return
    original_result = cast("dict[str, object]", original_result)
    redacted_result = cast("dict[str, object]", redacted_result)
    content = original_result.get("content")
    if type(content) is str:
        redacted_result["content"] = _redact_policy_denial_text(content, redactor=redactor)
    original_structured = original_result.get("structured")
    redacted_structured = redacted_result.get("structured")
    if type(original_structured) is not dict or type(redacted_structured) is not dict:
        return
    original_structured = cast("dict[str, object]", original_structured)
    redacted_structured = cast("dict[str, object]", redacted_structured)
    structured_reason = original_structured.get("reason")
    if type(structured_reason) is str:
        redacted_structured["reason"] = _redact_policy_denial_text(
            structured_reason,
            redactor=redactor,
        )
    for field_name in ("decision", "error"):
        value = original_structured.get(field_name)
        if type(value) is str:
            redacted_structured[field_name] = value


def _redact_policy_denial_text(value: str, *, redactor: SecretRedactor) -> str:
    if not value.endswith(_POLICY_DENIAL_TRUNCATION_MARKER):
        return redactor.redact_text(value)
    prefix = value[: -len(_POLICY_DENIAL_TRUNCATION_MARKER)]
    return redactor.redact_text(prefix) + _POLICY_DENIAL_TRUNCATION_MARKER


def _exception_summary(
    exc: Exception,
    *,
    redactor: SecretRedactor | None = None,
) -> str:
    resolved_redactor = redactor or SecretRedactor()
    parts = [f"{_object_type_name(exc)}: {_exception_message(exc, redactor=resolved_redactor)}"]
    try:
        notes = object.__getattribute__(exc, "__notes__")
    except BaseException:
        notes = ()
    if type(notes) is list:
        parts.extend(
            portable_persisted_event_side_effect_error(resolved_redactor.redact_text(note))
            for note in islice(notes, _MAX_EXCEPTION_NOTES)
            if type(note) is str
        )
        if len(notes) > _MAX_EXCEPTION_NOTES:
            parts.append(f"{len(notes) - _MAX_EXCEPTION_NOTES} additional notes omitted")
    return portable_persisted_event_side_effect_error("; ".join(parts))


def _failure_summary(
    failures: list[Exception],
    *,
    start: int = 0,
    redactor: SecretRedactor | None = None,
) -> str:
    """Bound diagnostic aggregation independently of configured sink count."""

    resolved_redactor = redactor or SecretRedactor()
    parts = [
        _exception_summary(failure, redactor=resolved_redactor)
        for failure in islice(failures, start, start + _MAX_AGGREGATED_FAILURES)
    ]
    omitted = len(failures) - start - len(parts)
    if omitted > 0:
        parts.append(f"{omitted} additional failures omitted")
    return portable_persisted_event_side_effect_error("; ".join(parts))


def _exception_message(
    exc: Exception,
    *,
    redactor: SecretRedactor | None = None,
) -> str:
    try:
        message = str(exc)
    except BaseException:
        message = None
    if message is not None:
        message = (redactor or SecretRedactor()).redact_text(message)
    return portable_persisted_event_side_effect_error(message)


def _object_type_name(value: object) -> str:
    """Return a portable type name without invoking an extension metaclass."""

    try:
        name = type.__getattribute__(type(value), "__name__")
    except BaseException:
        name = None
    return portable_persisted_event_side_effect_error(name)


def _add_exception_note(exc: BaseException, note: str) -> None:
    """Attach optional diagnostics without relying on extension overrides."""

    portable_note = portable_persisted_event_side_effect_error(note)
    with suppress(BaseException):
        BaseException.add_note(exc, portable_note)
