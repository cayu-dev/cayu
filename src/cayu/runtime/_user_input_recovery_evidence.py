"""Durable user-input receipt evidence shared by execution and recovery."""

from __future__ import annotations

from typing import Any, cast

from cayu._validation import (
    copy_json_value,
)
from cayu.approvals.user_input import (
    AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY,
    PENDING_USER_INPUT_CHECKPOINT_KEY,
    USER_INPUT_SUPERSESSION_INTENT_KEY,
    AmbiguousUserInputSupersessionIntent,
    PendingUserInput,
    UserInputPauseState,
    UserInputSupersessionIntent,
    pending_user_input_identity,
    user_input_lifecycle_authority_from_checkpoint,
)
from cayu.events import (
    Event,
    EventType,
)
from cayu.runtime._interruption_coordinator import (
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
)
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
from cayu.sessions.records import (
    EventRecord,
    Session,
    SessionStatus,
)
from cayu.vaults.redaction import SecretRedactor


class UserInputRecoveryEvidence:
    """Authenticate exact user-input opening, closure and supersession evidence."""

    def __init__(self, session_store: SessionStore, secret_redactor: SecretRedactor) -> None:
        self._session_store = session_store
        self._secret_redactor = secret_redactor

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

    async def classify_pause(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        input_id: str,
        _refresh_supersession_conflict: bool = True,
    ) -> UserInputPauseState:
        """Classify one exact pause from positive durable lifecycle evidence."""

        close_receipt = await self._session_store.load_runtime_publication_receipt(
            session.id,
            f"user-input-close:{input_id}",
        )
        if close_receipt is not None:
            # The caller's checkpoint read may have raced the atomic close.
            # Read it again only after the receipt is observable so an
            # acknowledgement-loss retry cannot mistake the pre-close pause
            # for contradictory durable state.
            checkpoint = await self._session_store.load_checkpoint(session.id)
        try:
            pending, resolution_intent = user_input_lifecycle_authority_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
                current_run_epoch=session.run_epoch,
                runtime_session=session,
            )
        except (TypeError, ValueError, RuntimeError):
            return UserInputPauseState.AMBIGUOUS
        interrupt_marker: object | None = None
        if checkpoint is not None:
            interrupt_payload = checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            if type(interrupt_payload) is dict:
                interrupt_marker = interrupt_payload.get(USER_INPUT_SUPERSESSION_INTENT_KEY)

        try:
            supersession_events = await self._session_store.load_user_input_supersession_events(
                session.id,
                input_id,
            )
        except (TypeError, ValueError):
            return UserInputPauseState.AMBIGUOUS
        pending_conflicts_with_supersession = pending is not None and (
            supersession_events
            or (type(interrupt_marker) is dict and interrupt_marker.get("input_id") == input_id)
        )
        if pending_conflicts_with_supersession:
            # The caller's checkpoint read may have preceded an atomic
            # supersession whose terminal event is already visible. Never let
            # that mixed snapshot reclassify the retired pause as active.
            if not _refresh_supersession_conflict:
                return UserInputPauseState.AMBIGUOUS
            refreshed_session = await self._session_store.load(session.id)
            if refreshed_session is None:
                return UserInputPauseState.AMBIGUOUS
            refreshed_checkpoint = await self._session_store.load_checkpoint(session.id)
            return await self.classify_pause(
                session=refreshed_session,
                checkpoint=refreshed_checkpoint,
                input_id=input_id,
                _refresh_supersession_conflict=False,
            )
        if close_receipt is not None:
            if (
                (pending is not None and pending.input_id == input_id)
                or (resolution_intent is not None and resolution_intent.input_id == input_id)
                or (type(interrupt_marker) is dict and interrupt_marker.get("input_id") == input_id)
            ):
                return UserInputPauseState.AMBIGUOUS
            if supersession_events:
                return UserInputPauseState.AMBIGUOUS
            try:
                await self.exact_user_input_close_event(
                    session=session,
                    input_id=input_id,
                    receipt=close_receipt,
                )
            except SessionRuntimePublicationConflict:
                return UserInputPauseState.AMBIGUOUS
            else:
                return UserInputPauseState.ANSWERED

        if pending is not None:
            if (
                pending.input_id != input_id
                or pending.session_id != session.id
                or pending.session_instance_id != session.instance_id
            ):
                return UserInputPauseState.AMBIGUOUS
            try:
                await self.require_exact_user_input_open_receipt(
                    session=session,
                    pending=pending,
                )
            except SessionRuntimePublicationConflict:
                return UserInputPauseState.AMBIGUOUS
            return (
                UserInputPauseState.ANSWERING
                if resolution_intent is not None
                else UserInputPauseState.ACTIVE
            )

        if resolution_intent is not None:
            return UserInputPauseState.AMBIGUOUS

        marker_candidates: list[object] = []
        if interrupt_marker is not None:
            marker_candidates.append(interrupt_marker)
        marker_candidates.extend(
            event.payload.get(USER_INPUT_SUPERSESSION_INTENT_KEY) for event in supersession_events
        )
        matching_markers: list[dict[str, object]] = []
        for marker in marker_candidates:
            if type(marker) is not dict:
                continue
            typed_marker = cast("dict[str, object]", marker)
            if typed_marker.get("input_id") == input_id:
                matching_markers.append(typed_marker)
        if not matching_markers:
            return UserInputPauseState.AMBIGUOUS
        if len(supersession_events) > 1:
            return UserInputPauseState.AMBIGUOUS
        if any(
            event.type is not EventType.SESSION_INTERRUPTED
            or event.session_id != session.id
            or event.payload.get("interruption_type") != "operator_requested"
            for event in supersession_events
        ):
            return UserInputPauseState.AMBIGUOUS
        try:
            open_receipt = await self.require_exact_user_input_open_receipt(
                session=session,
                input_id=input_id,
            )
        except SessionRuntimePublicationConflict:
            return UserInputPauseState.AMBIGUOUS
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
        parsed_markers: list[UserInputSupersessionIntent] = []
        for marker in matching_markers:
            try:
                parsed = UserInputSupersessionIntent.model_validate(marker)
            except (TypeError, ValueError):
                return UserInputPauseState.AMBIGUOUS
            if parsed.session_id != session.id or parsed.session_instance_id != session.instance_id:
                return UserInputPauseState.AMBIGUOUS
            parsed_payload = parsed.model_dump(mode="json", exclude_none=True)
            if any(
                parsed_payload.get(field_name) != open_receipt.intent.get(field_name)
                for field_name in immutable_identity_fields
            ):
                return UserInputPauseState.AMBIGUOUS
            parsed_markers.append(parsed)
        if any(marker != parsed_markers[0] for marker in parsed_markers[1:]):
            return UserInputPauseState.AMBIGUOUS
        return UserInputPauseState.SUPERSEDED
