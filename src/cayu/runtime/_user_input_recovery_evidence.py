"""Durable user-input receipt evidence shared by execution and recovery."""

from __future__ import annotations

from typing import Any

from cayu._validation import copy_json_value
from cayu.approvals.user_input import (
    AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY,
    PENDING_USER_INPUT_CHECKPOINT_KEY,
    USER_INPUT_SUPERSESSION_INTENT_KEY,
    AmbiguousUserInputSupersessionIntent,
    PendingUserInput,
    UserInputSupersessionIntent,
    pending_user_input_identity,
)
from cayu.events import Event, EventType
from cayu.sessions._terminal_evidence import (
    _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
    interruption_request_id_from_payload,
)
from cayu.sessions.base import (
    RuntimePublicationReceipt,
    SessionRuntimePublicationConflict,
    SessionStore,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import EventRecord, Session, SessionStatus


class UserInputRecoveryEvidence:
    """Authenticate exact user-input opening, closure and supersession evidence."""

    def __init__(self, session_store: SessionStore) -> None:
        self._session_store = session_store

    async def require_exact_user_input_open_receipt(
        self,
        *,
        session: Session,
        pending: PendingUserInput | None = None,
        input_id: str | None = None,
    ) -> RuntimePublicationReceipt:
        """Load and authenticate the complete publication that opened one pause."""

        if pending is not None:
            if input_id is not None and input_id != pending.input_id:
                raise SessionRuntimePublicationConflict(
                    "Pending user input conflicts with its requested opening receipt."
                )
            input_id = pending.input_id
        if type(input_id) is not str or not input_id:
            raise SessionRuntimePublicationConflict(
                "User-input opening receipt lookup has no exact pause identity."
            )
        receipt = await self._session_store.load_runtime_publication_receipt(
            session.id,
            f"user-input-open:{input_id}",
        )
        identity_fields = {
            "schema_version",
            "session_id",
            "session_instance_id",
            "source_interaction_id",
            "source_run_epoch",
            "input_id",
            "tool_call_id",
            "tool_round_id",
            "model_step_id",
            "model_attempt_id",
            "execution_profile_fingerprint",
            "pause_digest",
        }
        required_fields = identity_fields | {"source_round_digest", "event_ids"}
        expected_identity = pending_user_input_identity(pending) if pending is not None else None
        if (
            receipt is None
            or receipt.session_id != session.id
            or receipt.publication_id != f"user-input-open:{input_id}"
            or receipt.kind != "user-input-open"
            or receipt.source_status is not SessionStatus.RUNNING
            or set(receipt.intent) != required_fields
            or receipt.intent.get("schema_version") != 1
            or receipt.intent.get("session_id") != session.id
            or receipt.intent.get("session_instance_id") != session.instance_id
            or receipt.intent.get("input_id") != input_id
            or receipt.source_run_epoch != receipt.intent.get("source_run_epoch")
            or receipt.interaction_id != receipt.intent.get("source_interaction_id")
            or receipt.transcript_start_cursor != receipt.transcript_end_cursor
            or receipt.referenced_events
            or (
                expected_identity is not None
                and any(
                    receipt.intent.get(key) != value for key, value in expected_identity.items()
                )
            )
            or receipt.intent.get("event_ids") != list(receipt.appended_event_ids)
            or len(receipt.appended_event_ids) != 2
        ):
            raise SessionRuntimePublicationConflict(
                "Pending user input has no exact durable opening receipt."
            )
        for field_name in (
            "execution_profile_fingerprint",
            "pause_digest",
            "source_round_digest",
        ):
            value = receipt.intent.get(field_name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SessionRuntimePublicationConflict(
                    "User-input opening receipt contains malformed digest authority."
                )
        records: list[EventRecord] = []
        for event_id in receipt.appended_event_ids:
            candidates = await self._session_store.query_events(
                EventQuery(session_id=session.id, event_id=event_id, limit=2)
            )
            if len(candidates) != 1 or candidates[0].event.id != event_id:
                raise SessionRuntimePublicationConflict(
                    "User-input opening event is missing from durable history."
                )
            records.append(candidates[0])
        if [record.event.type for record in records] != [
            EventType.SESSION_CHECKPOINTED,
            EventType.SESSION_AWAITING_USER_INPUT,
        ]:
            raise SessionRuntimePublicationConflict(
                "User-input opening event sequence conflicts with its receipt."
            )
        for record in records:
            event = record.event
            if (
                event.session_id != session.id
                or event.interaction_id != receipt.intent["source_interaction_id"]
                or any(
                    event.payload.get(field_name) != receipt.intent[field_name]
                    for field_name in (
                        "input_id",
                        "tool_call_id",
                        "tool_round_id",
                        "model_step_id",
                        "model_attempt_id",
                        "source_run_epoch",
                        "pause_digest",
                    )
                )
            ):
                raise SessionRuntimePublicationConflict(
                    "User-input opening event conflicts with its pause authority."
                )
        return receipt

    async def validated_user_input_supersession_interrupt_payload(
        self,
        *,
        session: Session,
        pending_interrupt_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Authenticate one retained exact or ambiguous user-input supersession."""

        supersession_payload = pending_interrupt_payload.get(USER_INPUT_SUPERSESSION_INTENT_KEY)
        ambiguous_supersession_payload = pending_interrupt_payload.get(
            AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY
        )
        if supersession_payload is None and ambiguous_supersession_payload is None:
            return None
        if supersession_payload is not None and ambiguous_supersession_payload is not None:
            raise SessionRuntimePublicationConflict(
                "Retained user-input supersession has conflicting authority."
            )
        try:
            interruption_request_id = interruption_request_id_from_payload(
                pending_interrupt_payload
            )
        except ValueError as exc:
            raise SessionRuntimePublicationConflict(
                "Retained user-input supersession conflicts with its session."
            ) from exc
        if (
            interruption_request_id is None
            or pending_interrupt_payload.get("interruption_type")
            != _INTERRUPTION_TYPE_OPERATOR_REQUESTED
        ):
            raise SessionRuntimePublicationConflict(
                "Retained user-input supersession conflicts with its session."
            )
        if supersession_payload is not None:
            try:
                supersession_intent = UserInputSupersessionIntent.model_validate(
                    supersession_payload
                )
            except (TypeError, ValueError) as exc:
                raise SessionRuntimePublicationConflict(
                    "Retained user-input supersession evidence is malformed."
                ) from exc
            if (
                supersession_intent.session_id != session.id
                or supersession_intent.session_instance_id != session.instance_id
            ):
                raise SessionRuntimePublicationConflict(
                    "Retained user-input supersession conflicts with its session."
                )
            open_receipt = await self.require_exact_user_input_open_receipt(
                session=session,
                input_id=supersession_intent.input_id,
            )
            supersession_identity = supersession_intent.model_dump(
                mode="json",
                exclude={
                    "state",
                    "claim_run_epoch",
                    "resolution_request_digest",
                },
            )
            if any(
                open_receipt.intent.get(field_name) != value
                for field_name, value in supersession_identity.items()
            ):
                raise SessionRuntimePublicationConflict(
                    "Retained user-input supersession conflicts with its opening receipt."
                )
        else:
            try:
                ambiguous_supersession_intent = AmbiguousUserInputSupersessionIntent.model_validate(
                    ambiguous_supersession_payload
                )
            except (TypeError, ValueError) as exc:
                raise SessionRuntimePublicationConflict(
                    "Retained ambiguous user-input supersession evidence is malformed."
                ) from exc
            if (
                ambiguous_supersession_intent.session_id != session.id
                or ambiguous_supersession_intent.session_instance_id != session.instance_id
            ):
                raise SessionRuntimePublicationConflict(
                    "Retained ambiguous user-input supersession conflicts with its session."
                )
        return copy_json_value(
            pending_interrupt_payload,
            "pending_session_interrupt",
        )

    async def exact_user_input_close_event(
        self,
        *,
        session: Session,
        input_id: str,
        receipt: RuntimePublicationReceipt,
        expected_resolution_request_digest: str | None = None,
    ) -> Event:
        """Authenticate an exact answered receipt and return its durable close event."""

        open_receipt = await self.require_exact_user_input_open_receipt(
            session=session,
            input_id=input_id,
        )
        intent = receipt.intent
        referenced_ids = [reference.event_id for reference in receipt.referenced_events]
        required_fields = {
            "schema_version",
            "session_id",
            "session_instance_id",
            "source_interaction_id",
            "source_run_epoch",
            "input_id",
            "tool_call_id",
            "tool_round_id",
            "model_step_id",
            "model_attempt_id",
            "execution_profile_fingerprint",
            "pause_digest",
            "claim_run_epoch",
            "answer_request_digest",
            "execution_state",
            "resolution_request_digest",
            "tool_call_ids",
            "event_ids",
            "referenced_event_ids",
        }
        if "post_action_continuation_digest" in intent:
            required_fields.add("post_action_continuation_digest")
            digest = intent["post_action_continuation_digest"]
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise SessionRuntimePublicationConflict("Invalid action-close continuation digest.")
        if "foreground_parent_continuation" in intent:
            required_fields.add("foreground_parent_continuation")
        if (
            receipt.session_id != session.id
            or receipt.publication_id != f"user-input-close:{input_id}"
            or receipt.kind != "user-input-close"
            or receipt.source_status is not SessionStatus.RUNNING
            or set(intent) != required_fields
            or intent.get("schema_version") != 1
            or intent.get("session_id") != session.id
            or intent.get("session_instance_id") != session.instance_id
            or intent.get("source_interaction_id") != receipt.interaction_id
            or intent.get("input_id") != input_id
            or intent.get("claim_run_epoch") != receipt.source_run_epoch
            or intent.get("execution_state") != "executing"
            or intent.get("event_ids") != list(receipt.appended_event_ids)
            or intent.get("referenced_event_ids") != referenced_ids
            or len(receipt.appended_event_ids) != 1
            or (
                expected_resolution_request_digest is not None
                and intent.get("resolution_request_digest") != expected_resolution_request_digest
            )
        ):
            raise SessionRuntimePublicationConflict(
                "User input was already closed with conflicting resolution authority."
            )
        immutable_identity_fields = (
            "schema_version",
            "session_id",
            "session_instance_id",
            "source_interaction_id",
            "source_run_epoch",
            "input_id",
            "tool_call_id",
            "tool_round_id",
            "model_step_id",
            "model_attempt_id",
            "execution_profile_fingerprint",
            "pause_digest",
        )
        if any(
            intent.get(field_name) != open_receipt.intent.get(field_name)
            for field_name in immutable_identity_fields
        ):
            raise SessionRuntimePublicationConflict(
                "User-input closure does not belong to its exact opening publication."
            )
        for field_name in (
            "execution_profile_fingerprint",
            "pause_digest",
            "answer_request_digest",
            "resolution_request_digest",
        ):
            value = intent.get(field_name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SessionRuntimePublicationConflict(
                    "User-input closure receipt contains malformed digest authority."
                )
        event_id = receipt.appended_event_ids[0]
        records = await self._session_store.query_events(
            EventQuery(session_id=session.id, event_id=event_id, limit=2)
        )
        if len(records) != 1 or records[0].event.id != event_id:
            raise SessionRuntimePublicationConflict(
                "User-input closure event is missing from durable history."
            )
        event = records[0].event
        if (
            event.type is not EventType.SESSION_CHECKPOINTED
            or event.session_id != session.id
            or event.interaction_id != intent.get("source_interaction_id")
            or event.payload.get("checkpoint") != PENDING_USER_INPUT_CHECKPOINT_KEY
            or event.payload.get("transition") != "answered"
            or any(
                event.payload.get(field_name) != intent.get(field_name)
                for field_name in (
                    "source_run_epoch",
                    "input_id",
                    "tool_call_id",
                    "tool_round_id",
                    "model_step_id",
                    "model_attempt_id",
                    "pause_digest",
                    "resolution_request_digest",
                )
            )
        ):
            raise SessionRuntimePublicationConflict(
                "User-input closure event conflicts with its receipt."
            )
        return event
