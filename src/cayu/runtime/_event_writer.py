from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import suppress
from itertools import islice

from cayu.core.events import (
    Event,
    EventType,
    copy_event,
    event_with_durable_sequence,
    event_with_runtime_envelope_authority,
)
from cayu.runtime._event_projection import (
    prepare_budget_settlement_event_template,
    prepare_new_runtime_event,
    project_persisted_runtime_event,
)
from cayu.runtime.budgets import BudgetStore
from cayu.runtime.event_sinks import (
    EventSink,
    InMemoryEventSink,
    _emit_in_memory_delivery,
    _EventSinkDelivery,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.runtime.sessions import (
    EventQuery,
    EventRecord,
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
logger = logging.getLogger(__name__)


def _reconcile_exact_persisted_event(
    expected: Event,
    records: list[EventRecord],
    *,
    conflict_message: str,
) -> Event | None:
    """Return one exact durable event, fail closed on reused identity."""

    if not records:
        return None
    persisted = records[0].event
    if len(records) != 1 or persisted != expected:
        raise RuntimeError(conflict_message)
    return copy_event(persisted)


class RuntimeEventWriter:
    """Persist runtime events and fan them out to configured sinks."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        budget_store: BudgetStore,
        event_sinks: Iterable[EventSink],
        secret_redactor: SecretRedactor | None = None,
        public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
    ) -> None:
        if secret_redactor is not None and not isinstance(secret_redactor, SecretRedactor):
            raise TypeError("secret_redactor must be a SecretRedactor.")
        if public_authority_alias_codec is not None and not isinstance(
            public_authority_alias_codec,
            PublicAuthorityAliasCodec,
        ):
            raise TypeError("public_authority_alias_codec must be a PublicAuthorityAliasCodec.")
        self._session_store = session_store
        self._budget_store = budget_store
        self._event_sinks = tuple(event_sinks)
        self._secret_redactor = secret_redactor or SecretRedactor()
        store_alias_codec = session_store.public_authority_alias_codec
        if (
            public_authority_alias_codec is not None
            and public_authority_alias_codec != store_alias_codec
        ):
            raise ValueError(
                "session_store and RuntimeEventWriter must use the same public "
                "authority alias keyring."
            )
        self._public_authority_alias_codec = store_alias_codec
        if self._secret_redactor.has_values and (
            self._public_authority_alias_codec is None
            or not session_store.supports_public_authority_aliases
        ):
            raise ValueError(
                "A secret-redacting RuntimeEventWriter requires a durable indexed "
                "public-authority alias store and codec before publication."
            )

    async def emit(self, event: Event) -> Event:
        event = self.prepare(attribute_event_to_current_interaction(event))
        await self._session_store.append_event(event.session_id, event)
        claim = await self._session_store.claim_persisted_event_side_effect(
            session_id=event.session_id,
            event_id=event.id,
        )
        if claim is None:
            sequence = await self._handle_unclaimed_persisted_side_effect(event)
            return event_with_durable_sequence(event, sequence)
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
        """Return whether this event identity reached the durable event handoff."""

        records = await self._session_store.query_events(
            EventQuery(session_id=event.session_id, event_id=event.id, limit=1)
        )
        return any(record.event.id == event.id for record in records)

    async def is_exact_persisted(self, event: Event) -> bool:
        """Return whether one exact prepared event reached the durable handoff."""

        records = await self._session_store.query_events(
            EventQuery(session_id=event.session_id, event_id=event.id, limit=2)
        )
        return (
            _reconcile_exact_persisted_event(
                event,
                records,
                conflict_message="Persisted event identity conflicts with exact readback.",
            )
            is not None
        )

    async def emit_many(self, session_id: str, events: list[Event]) -> list[Event]:
        """Persist and fan out a defensive copy of one event batch.

        Batch events use the same durable budget and sink handoff as ``emit``.
        This matters for store-atomic publications that include cost-bearing
        events, such as explicit compaction recovery.
        """
        copied_events = await self.persist_many(session_id, events)
        return await self.fan_out_persisted(copied_events)

    async def reserve_workflow_step_started(
        self,
        event: Event,
        *,
        workflow_name: str,
        attempt_id: str,
    ) -> bool:
        """Atomically fence and publish one workflow step reservation."""

        prepared = self.prepare(attribute_event_to_current_interaction(event))
        reserved = await self._session_store.append_workflow_step_started(
            prepared.session_id,
            prepared,
            workflow_name=workflow_name,
            attempt_id=attempt_id,
        )
        if not reserved:
            return False
        await self.fan_out_persisted([prepared])
        return True

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
            # This method accepts only already-persisted records. Reapplying
            # new-write authority validation here would strand imported legacy
            # rows and lose generated-ID provenance after deserialization.
            copied = copy_event(event)
            claim = await self._session_store.claim_persisted_event_side_effect(
                session_id=copied.session_id,
                event_id=copied.id,
            )
            if claim is None:
                sequence = await self._handle_unclaimed_persisted_side_effect(copied)
                delivered_event = event_with_durable_sequence(copied, sequence)
            else:
                delivered_event, _ = await self._deliver_persisted_side_effect_claim(claim)
            copied_events.append(delivered_event)
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
                _, delivered = await self._deliver_persisted_side_effect_claim(claim)
            except Exception as exc:
                public = project_persisted_runtime_event(
                    claim.event,
                    sequence=claim.event_sequence,
                    redactor=self._secret_redactor,
                    public_authority_alias_codec=self._public_authority_alias_codec,
                )
                logger.error(
                    "Persisted event side-effect recovery failed: "
                    "session_id=%s event_id=%s event_type=%s error_type=%s",
                    public.session_id,
                    public.id,
                    public.type,
                    _object_type_name(exc),
                )
                continue
            if delivered:
                recovered.append(
                    project_persisted_runtime_event(
                        claim.event,
                        sequence=claim.event_sequence,
                        redactor=self._secret_redactor,
                        public_authority_alias_codec=self._public_authority_alias_codec,
                    )
                )
        return recovered

    async def _deliver_persisted_side_effect_claim(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> tuple[Event, bool]:
        try:
            # Accounting and durable claim ownership intentionally use the
            # original record. Legacy exposure is projected only after these
            # private authority decisions have completed.
            private_event = claim.event.model_copy(deep=True)
            await self._forward_budget_event_if_required(private_event)
            public_event = project_persisted_runtime_event(
                private_event,
                sequence=claim.event_sequence,
                redactor=self._secret_redactor,
                public_authority_alias_codec=self._public_authority_alias_codec,
            )
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
        sink_failures = await self._emit_to_sinks(
            _EventSinkDelivery(
                event=public_event,
                event_sequence=claim.event_sequence,
                private_session_id=claim.session_id,
                private_event_id=claim.event_id,
                private_interaction_id=private_event.interaction_id,
                private_tool_call_id=_private_payload_string(
                    private_event.payload,
                    "tool_call_id",
                ),
                private_parent_session_id=_private_payload_string(
                    private_event.payload,
                    "parent_session_id",
                ),
            )
        )
        delivered_event = event_with_durable_sequence(private_event, claim.event_sequence)
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
            return delivered_event, False
        try:
            await self._session_store.mark_persisted_event_side_effect_delivered(claim)
        except PersistedEventSideEffectClaimLost:
            return delivered_event, False
        except Exception as exc:
            public = public_event
            logger.error(
                "Persisted event side-effect delivery acknowledgement failed; "
                "leaving the durable claim for recovery: "
                "session_id=%s event_id=%s event_type=%s error_type=%s",
                public.session_id,
                public.id,
                public.type,
                _object_type_name(exc),
            )
            return delivered_event, False
        return delivered_event, True

    async def _forward_budget_event_if_required(self, event: Event) -> None:
        if event.type == EventType.MODEL_COMPLETED:
            await self._budget_store.append_event(event.model_copy(deep=True))

    async def _handle_unclaimed_persisted_side_effect(self, event: Event) -> int:
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
                return delivery.event_sequence
            return delivery.event_sequence
        if delivery.status in {
            PersistedEventSideEffectStatus.FAILED,
            PersistedEventSideEffectStatus.DELIVERED,
            PersistedEventSideEffectStatus.DEAD_LETTERED,
        }:
            return delivery.event_sequence
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
            public = project_persisted_runtime_event(
                claim.event,
                sequence=claim.event_sequence,
                redactor=self._secret_redactor,
                public_authority_alias_codec=self._public_authority_alias_codec,
            )
            logger.error(
                "Persisted event side effect dead-lettered: "
                "session_id=%s event_id=%s event_type=%s attempts=%s budget_effect=%s",
                public.session_id,
                public.id,
                public.type,
                delivery.attempts,
                claim.event.type == EventType.MODEL_COMPLETED,
            )
        return delivery

    async def _emit_to_sinks(self, delivery: _EventSinkDelivery) -> list[Exception]:
        failures: list[Exception] = []
        event = delivery.event
        for sink in self._event_sinks:
            try:
                await _emit_event_sink(sink, delivery)
            except Exception as exc:
                try:
                    diagnostic = Event(
                        type=EventType.RUNTIME_SINK_FAILED,
                        session_id=delivery.private_session_id,
                        interaction_id=_safe_sink_failure_interaction_id(
                            delivery,
                            redactor=self._secret_redactor,
                        ),
                        agent_name=event.agent_name,
                        environment_name=event.environment_name,
                        payload={
                            "sink": _object_type_name(sink),
                            "error": _exception_message(
                                exc,
                                redactor=self._secret_redactor,
                            ),
                            "error_type": _object_type_name(exc),
                            "event_sequence": delivery.event_sequence,
                            "event_type": str(event.type),
                        },
                    )
                    envelope_fields = ["session_id"]
                    if diagnostic.interaction_id is not None:
                        envelope_fields.append("interaction_id")
                    diagnostic = event_with_runtime_envelope_authority(
                        diagnostic,
                        *envelope_fields,
                    )
                    await self._session_store.append_event(
                        delivery.private_session_id,
                        prepare_new_runtime_event(
                            diagnostic,
                            redactor=self._secret_redactor,
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
        """Return an event validated for its first durable append."""

        return prepare_new_runtime_event(
            attribute_event_to_current_interaction(event),
            redactor=self._secret_redactor,
        )

    def prepare_budget_settlement_template(self, event: Event) -> Event:
        """Prepare causal metadata retained until ledger settlement."""

        return prepare_budget_settlement_event_template(
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
    """Backward-compatible new-write validation entry point.

    Legacy/public projection requires a durable sequence and therefore uses
    :func:`project_runtime_event` directly.
    """

    if reject_authority_secrets is not True:
        raise ValueError(
            "Legacy event projection requires project_runtime_event() and a durable event sequence."
        )
    return prepare_new_runtime_event(event, redactor=redactor)


async def _emit_event_sink(
    sink: EventSink,
    delivery: _EventSinkDelivery,
) -> None:
    """Expose private correlation only through exact runtime-owned adapters."""

    if type(sink) is InMemoryEventSink:
        await _emit_in_memory_delivery(sink, delivery)
        return
    # The optional telemetry module imports no OpenTelemetry package until the
    # concrete sink is constructed, so this lazy exact-type check keeps normal
    # runtime imports lightweight while preventing subclass overrides.
    from cayu.observability.otel import (
        OpenTelemetryEventSink,
        _emit_opentelemetry_delivery,
    )

    if type(sink) is OpenTelemetryEventSink:
        await _emit_opentelemetry_delivery(sink, delivery)
        return
    await sink.emit(copy_event(delivery.event))


def _safe_sink_failure_interaction_id(
    delivery: _EventSinkDelivery,
    *,
    redactor: SecretRedactor,
) -> str | None:
    """Avoid turning a public redaction marker into durable interaction authority."""

    interaction_id = delivery.private_interaction_id
    if interaction_id is None or redactor.redact_text(interaction_id) != interaction_id:
        return None
    return interaction_id


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


def _private_payload_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    return value if type(value) is str and value else None


def _add_exception_note(exc: BaseException, note: str) -> None:
    """Attach optional diagnostics without relying on extension overrides."""

    portable_note = portable_persisted_event_side_effect_error(note)
    with suppress(BaseException):
        BaseException.add_note(exc, portable_note)
