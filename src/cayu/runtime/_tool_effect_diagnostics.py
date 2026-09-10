"""Nonterminal cleanup evidence; never effect settlement or retry authority."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes, copy_durable_json_value
from cayu.core.events import Event, EventType, event_with_runtime_generated_id
from cayu.runtime._diagnostics import MAX_DIAGNOSTIC_UTF8_BYTES
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._tool_effect_state import ToolEffectRecord
from cayu.runtime.sessions import EventQuery, SessionStore
from cayu.runtime.tool_effects import ToolEffectConflict
from cayu.vaults import SecretRedactor


async def persist_cleanup_diagnostics(
    *,
    store: SessionStore,
    writer: RuntimeEventWriter,
    records: tuple[ToolEffectRecord, ...],
    artifacts: list[dict],
    redactor: SecretRedactor,
    attributed: bool,
) -> Event | None:
    """Persist detached diagnostics against positively resolved dispatches.

    An unattributed observation remains round-scoped even if only one dispatch
    was discovered. Exact replay preserves the first publication timestamp.
    """
    if type(artifacts) is not list or any(type(item) is not dict for item in artifacts):
        raise TypeError("Cleanup artifacts must be a list of objects.")
    if not artifacts:
        return None
    if not records or (attributed and len(records) != 1):
        raise ToolEffectConflict("Cleanup diagnostics require an exact dispatch scope.")
    first = records[0].intent
    if any(
        record.dispatch_id is None
        or record.intent.session_id != first.session_id
        or record.intent.session_instance_id != first.session_instance_id
        or record.intent.tool_round_id != first.tool_round_id
        for record in records
    ):
        raise ToolEffectConflict("Cleanup diagnostic dispatch scope conflicts.")
    scope = [
        {"intent": record.intent.model_dump(mode="json"), "dispatch_id": record.dispatch_id}
        for record in records
    ]
    scope_digest = sha256(canonical_durable_json_bytes(scope, "cleanup_scope")).hexdigest()
    copied = copy_durable_json_value(artifacts, "cleanup_artifacts")
    safe = redactor.redact_json(copied)
    if type(safe) is not list:
        raise TypeError("Cleanup artifacts must be a list.")
    artifacts_digest = sha256(canonical_durable_json_bytes(safe, "cleanup_artifacts")).hexdigest()
    bounded = []
    for artifact in safe:
        candidate = [*bounded, artifact]
        if (
            len(canonical_durable_json_bytes(candidate, "cleanup_artifacts"))
            > MAX_DIAGNOSTIC_UTF8_BYTES
        ):
            break
        bounded.append(artifact)
    payload = {
        "schema_version": 1,
        "scope_digest": scope_digest,
        "artifacts": bounded,
        "truncated": len(bounded) != len(safe),
        "model_step_id": first.model_step_id,
        "model_attempt_id": first.model_attempt_id,
        "tool_round_id": first.tool_round_id,
        **({"tool_call_id": first.tool_call_id} if attributed else {}),
    }
    event = event_with_runtime_generated_id(
        Event(
            id="tool-effect-cleanup:v1:"
            + sha256(
                canonical_durable_json_bytes(
                    {"payload": payload, "artifacts_digest": artifacts_digest}, "cleanup_event"
                )
            ).hexdigest(),
            type=EventType.TOOL_EFFECT_CLEANUP_OBSERVED,
            timestamp=datetime.now(UTC),
            session_id=first.session_id,
            interaction_id=first.interaction_id,
            agent_name=first.agent_name,
            environment_name=first.environment_name,
            tool_name=first.tool_name if attributed else None,
            payload=payload,
        )
    )
    prepared = writer.prepare(event)
    existing = await store.query_events(
        EventQuery(session_id=first.session_id, event_id=event.id, limit=1)
    )
    if existing:
        stored = existing[0].event
        if prepared.model_copy(update={"timestamp": stored.timestamp}) != stored:
            raise ToolEffectConflict("Cleanup diagnostic replay conflicts with stored evidence.")
        prepared = prepared.model_copy(update={"timestamp": stored.timestamp})
    try:
        return await writer.persist_exact_replay(prepared)
    except Exception as append_error:
        # Concurrent first publishers may both observe absence and choose
        # different timestamps. Only the first committed time is authoritative;
        # every other field must still match this exact diagnostic episode.
        try:
            rows = await store.query_events(
                EventQuery(session_id=first.session_id, event_id=event.id, limit=1)
            )
        except Exception as readback_error:
            raise ExceptionGroup(
                "Cleanup diagnostic append and reconciliation failed.",
                [append_error, readback_error],
            ) from None
        if (
            len(rows) != 1
            or prepared.model_copy(update={"timestamp": rows[0].event.timestamp}) != rows[0].event
        ):
            raise
        return rows[0].event.model_copy(deep=True)
