from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, StrictBool, field_validator, model_validator

from cayu.core.events import EventType
from cayu.core.messages import Message, MessageRole, TextPart
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_MESSAGES_PER_CASE,
    CorpusUserMessageSpec,
    RunInputSpec,
    _bounded_durable_text,
    _ordered_sequence_input,
    _SchemaV1PortableModel,
)
from cayu.evals.models import (
    Trajectory,
    _model_instance_python_input,
    _trajectory_promotion_capture_sha256,
    _validate_trajectory_record_contract,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import SessionStatus, session_input_messages_sha256

PROMOTABLE_RUN_INPUT_SCHEMA_VERSION = 1


class PromotableRunInputV1(_SchemaV1PortableModel):
    """Sanitized, text-only caller input proven to belong to one fresh invocation."""

    schema_version: Literal[1] = PROMOTABLE_RUN_INPUT_SCHEMA_VERSION
    messages: tuple[CorpusUserMessageSpec, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_MESSAGES_PER_CASE,
    )
    redactions_applied: StrictBool = False

    @field_validator("messages", mode="before")
    @classmethod
    def validate_messages_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_run_input_contract(self) -> PromotableRunInputV1:
        RunInputSpec(messages=self.messages)
        return self

    def to_run_input_spec(self) -> RunInputSpec:
        return RunInputSpec.model_validate(
            {"messages": [message.model_dump(mode="json") for message in self.messages]}
        )


class SessionPromotionErrorCode(StrEnum):
    """Stable reason one captured trajectory cannot become a portable eval case."""

    INVALID_TRAJECTORY = "invalid_trajectory"
    SOURCE_AGENT_MISMATCH = "source_agent_mismatch"
    ROOT_STATUS_UNSUPPORTED = "root_status_unsupported"
    DESCENDANT_EVIDENCE_UNSUPPORTED = "descendant_evidence_unsupported"
    APPROVAL_CONTINUATION_UNSUPPORTED = "approval_continuation_unsupported"
    SESSION_RESUME_UNSUPPORTED = "session_resume_unsupported"
    QUEUED_INPUT_UNSUPPORTED = "queued_input_unsupported"
    LATER_INTERACTION_UNSUPPORTED = "later_interaction_unsupported"
    STRUCTURED_OUTPUT_UNSUPPORTED = "structured_output_unsupported"
    INPUT_EVIDENCE_UNAVAILABLE = "input_evidence_unavailable"
    INPUT_EVIDENCE_INCONSISTENT = "input_evidence_inconsistent"
    INPUT_MESSAGE_COUNT_UNSUPPORTED = "input_message_count_unsupported"
    INPUT_ROLE_UNSUPPORTED = "input_role_unsupported"
    INPUT_PART_UNSUPPORTED = "input_part_unsupported"
    INPUT_LIMIT_EXCEEDED = "input_limit_exceeded"
    INPUT_REDACTION_FAILED = "input_redaction_failed"


_PROMOTION_ERROR_MESSAGES = {
    SessionPromotionErrorCode.INVALID_TRAJECTORY: (
        "The captured trajectory violates its durable evidence contract."
    ),
    SessionPromotionErrorCode.SOURCE_AGENT_MISMATCH: (
        "The captured root agent does not match the configured promotion source."
    ),
    SessionPromotionErrorCode.ROOT_STATUS_UNSUPPORTED: (
        "Promotion requires a completed or failed root session."
    ),
    SessionPromotionErrorCode.DESCENDANT_EVIDENCE_UNSUPPORTED: (
        "Promotion requires a complete completed/failed descendant tree."
    ),
    SessionPromotionErrorCode.APPROVAL_CONTINUATION_UNSUPPORTED: (
        "Portable corpus v1 does not support approval continuations."
    ),
    SessionPromotionErrorCode.SESSION_RESUME_UNSUPPORTED: (
        "Portable corpus v1 does not support resumed sessions."
    ),
    SessionPromotionErrorCode.QUEUED_INPUT_UNSUPPORTED: (
        "Portable corpus v1 does not support queued later input."
    ),
    SessionPromotionErrorCode.LATER_INTERACTION_UNSUPPORTED: (
        "Portable corpus v1 requires exactly one initial interaction."
    ),
    SessionPromotionErrorCode.STRUCTURED_OUTPUT_UNSUPPORTED: (
        "Portable corpus v1 does not support structured-output runs."
    ),
    SessionPromotionErrorCode.INPUT_EVIDENCE_UNAVAILABLE: (
        "The session has no runtime-attested fresh-input boundary."
    ),
    SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT: (
        "The runtime promotion attestation does not match the captured trajectory."
    ),
    SessionPromotionErrorCode.INPUT_MESSAGE_COUNT_UNSUPPORTED: (
        "Portable corpus v1 requires one or more bounded initial user messages."
    ),
    SessionPromotionErrorCode.INPUT_ROLE_UNSUPPORTED: (
        "Portable corpus v1 accepts only caller-supplied user messages."
    ),
    SessionPromotionErrorCode.INPUT_PART_UNSUPPORTED: (
        "Portable corpus v1 requires exactly one text part per caller-supplied message."
    ),
    SessionPromotionErrorCode.INPUT_LIMIT_EXCEEDED: (
        "The initial input exceeds a portable corpus v1 limit."
    ),
    SessionPromotionErrorCode.INPUT_REDACTION_FAILED: (
        "The initial input could not cross the application redaction boundary."
    ),
}


class SessionPromotionError(RuntimeError):
    """Typed fail-closed rejection from the pure promotion contract."""

    def __init__(self, code: SessionPromotionErrorCode) -> None:
        if not isinstance(code, SessionPromotionErrorCode):
            raise TypeError("code must be a SessionPromotionErrorCode.")
        self.code = code
        super().__init__(_PROMOTION_ERROR_MESSAGES[code])


_APPROVAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_APPROVAL_REQUESTED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.TOOL_CALL_APPROVAL_EXPIRED,
    }
)
_QUEUED_INPUT_EVENT_TYPES = frozenset(
    {
        EventType.SESSION_MESSAGE_QUEUED,
        EventType.SESSION_MESSAGE_DELIVERED,
    }
)
_STRUCTURED_OUTPUT_EVENT_TYPES = frozenset(
    {
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
    }
)


def _promotion_error(code: SessionPromotionErrorCode) -> SessionPromotionError:
    return SessionPromotionError(code)


def _validate_eligible_tree(trajectory: Trajectory) -> None:
    if trajectory.children_incomplete:
        raise _promotion_error(SessionPromotionErrorCode.DESCENDANT_EVIDENCE_UNSUPPORTED)
    for child in trajectory.children:
        if child.session is None or child.session.status not in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
        }:
            raise _promotion_error(SessionPromotionErrorCode.DESCENDANT_EVIDENCE_UNSUPPORTED)
        _validate_eligible_tree(child)


def _validate_caller_replay_phases(trajectory: Trajectory) -> None:
    """Reject caller-driven phases that a one-input portable corpus cannot replay."""

    event_types = tuple(event.type for event in trajectory.events)
    if any(event_type in _APPROVAL_EVENT_TYPES for event_type in event_types):
        raise _promotion_error(SessionPromotionErrorCode.APPROVAL_CONTINUATION_UNSUPPORTED)
    if EventType.SESSION_RESUMED in event_types:
        raise _promotion_error(SessionPromotionErrorCode.SESSION_RESUME_UNSUPPORTED)
    if any(event_type in _QUEUED_INPUT_EVENT_TYPES for event_type in event_types):
        raise _promotion_error(SessionPromotionErrorCode.QUEUED_INPUT_UNSUPPORTED)
    interaction_starts = event_types.count(EventType.INTERACTION_STARTED)
    if (
        interaction_starts != 1
        or EventType.INTERACTION_RESUMED in event_types
        or EventType.INTERACTION_PAUSED in event_types
    ):
        raise _promotion_error(SessionPromotionErrorCode.LATER_INTERACTION_UNSUPPORTED)
    for child in trajectory.children:
        _validate_caller_replay_phases(child)


def _initial_source_messages(
    trajectory: Trajectory,
    count: int,
    start_index: int,
) -> tuple[Message, ...]:
    transcript = trajectory.transcript
    end = start_index + count
    if end > len(transcript):
        raise _promotion_error(SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT)
    return transcript[start_index:end]


def _validated_trajectory_for_promotion(trajectory: Trajectory) -> Trajectory:
    """Rebuild untrusted public state while retaining exact private capture facts."""

    if type(trajectory) is not Trajectory:
        raise TypeError("trajectory must be an exact Trajectory.")
    try:
        attestation = (
            trajectory.initial_input_message_start_index,
            trajectory.initial_input_message_count,
            trajectory.initial_input_messages_sha256,
            trajectory.input_redactions_applied,
            trajectory.structured_output_requested,
        )
        capture_sha256 = trajectory._promotion_capture_sha256
        start_index, count, messages_sha256, redactions_applied, structured_output = attestation
        attestation_present = tuple(value is not None for value in attestation)
        if start_index is not None and type(start_index) is not int:
            raise TypeError("initial input message start index is malformed")
        if count is not None and type(count) is not int:
            raise TypeError("initial input message count is malformed")
        if messages_sha256 is not None and (
            type(messages_sha256) is not str
            or len(messages_sha256) != 64
            or any(character not in "0123456789abcdef" for character in messages_sha256)
        ):
            raise TypeError("initial input message digest is malformed")
        if redactions_applied is not None and type(redactions_applied) is not bool:
            raise TypeError("initial input redaction evidence is malformed")
        if structured_output is not None and type(structured_output) is not bool:
            raise TypeError("initial structured-output evidence is malformed")
        if capture_sha256 is not None and (
            type(capture_sha256) is not str
            or len(capture_sha256) != 64
            or any(character not in "0123456789abcdef" for character in capture_sha256)
        ):
            raise TypeError("promotion capture digest is malformed")

        validated = Trajectory.model_validate(_model_instance_python_input(trajectory))
        _validate_trajectory_record_contract(validated)
        validated._initial_input_message_start_index = start_index
        validated._initial_input_message_count = count
        validated._initial_input_messages_sha256 = messages_sha256
        validated._input_redactions_applied = redactions_applied
        validated._structured_output_requested = structured_output
        validated_capture_sha256 = (
            _trajectory_promotion_capture_sha256(validated)
            if capture_sha256 is not None and all(attestation_present)
            else None
        )
    except (AttributeError, KeyError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise _promotion_error(SessionPromotionErrorCode.INVALID_TRAJECTORY) from exc

    if capture_sha256 is None:
        if any(attestation_present):
            raise _promotion_error(SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT)
    elif not all(attestation_present) or validated_capture_sha256 != capture_sha256:
        raise _promotion_error(SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT)

    validated._initial_input_message_start_index = start_index
    validated._initial_input_message_count = count
    validated._initial_input_messages_sha256 = messages_sha256
    validated._input_redactions_applied = redactions_applied
    validated._structured_output_requested = structured_output
    validated._promotion_capture_sha256 = capture_sha256
    return validated


def _sanitized_user_messages(
    app: CayuApp,
    messages: tuple[Message, ...],
) -> tuple[tuple[CorpusUserMessageSpec, ...], bool]:
    sanitized: list[CorpusUserMessageSpec] = []
    redactions_applied = False
    for message in messages:
        if message.role != MessageRole.USER:
            raise _promotion_error(SessionPromotionErrorCode.INPUT_ROLE_UNSUPPORTED)
        # Corpus v1 has one text field per user message. Multiple provider input
        # blocks are not portably equivalent: some adapters preserve the blocks,
        # while others insert separators. Reject instead of silently collapsing
        # a production prompt into different replay input.
        if len(message.content) != 1 or type(message.content[0]) is not TextPart:
            raise _promotion_error(SessionPromotionErrorCode.INPUT_PART_UNSUPPORTED)
        text = message.content[0].text
        try:
            redacted = app.redact_json(text)
        except Exception as exc:
            raise _promotion_error(SessionPromotionErrorCode.INPUT_REDACTION_FAILED) from exc
        if type(redacted) is not str:
            raise _promotion_error(SessionPromotionErrorCode.INPUT_REDACTION_FAILED)
        redactions_applied = redactions_applied or redacted != text
        try:
            sanitized.append(CorpusUserMessageSpec(text=redacted))
        except ValueError as exc:
            raise _promotion_error(SessionPromotionErrorCode.INPUT_LIMIT_EXCEEDED) from exc
    return tuple(sanitized), redactions_applied


def _promotable_run_input_from_validated(
    app: CayuApp,
    trajectory: Trajectory,
    *,
    source_agent_name: str,
) -> PromotableRunInputV1:
    source_agent_name = _bounded_durable_text(
        source_agent_name,
        "source_agent_name",
        max_chars=256,
        nonblank=True,
        clean=True,
    )
    session = trajectory.session
    if session is None or session.status not in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
        raise _promotion_error(SessionPromotionErrorCode.ROOT_STATUS_UNSUPPORTED)
    if session.agent_name != source_agent_name:
        raise _promotion_error(SessionPromotionErrorCode.SOURCE_AGENT_MISMATCH)
    _validate_eligible_tree(trajectory)
    _validate_caller_replay_phases(trajectory)

    event_types = tuple(event.type for event in trajectory.events)
    if (
        trajectory.structured_output_requested is None
        or trajectory.input_redactions_applied is None
    ):
        raise _promotion_error(SessionPromotionErrorCode.INPUT_EVIDENCE_UNAVAILABLE)
    if trajectory.structured_output_requested or any(
        event_type in _STRUCTURED_OUTPUT_EVENT_TYPES for event_type in event_types
    ):
        raise _promotion_error(SessionPromotionErrorCode.STRUCTURED_OUTPUT_UNSUPPORTED)

    input_count = trajectory.initial_input_message_count
    input_start_index = trajectory.initial_input_message_start_index
    if input_count is None or input_start_index is None:
        raise _promotion_error(SessionPromotionErrorCode.INPUT_EVIDENCE_UNAVAILABLE)
    if not 1 <= input_count <= EVAL_CORPUS_MAX_MESSAGES_PER_CASE:
        raise _promotion_error(SessionPromotionErrorCode.INPUT_MESSAGE_COUNT_UNSUPPORTED)
    if input_start_index < 0:
        raise _promotion_error(SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT)
    source_messages = _initial_source_messages(trajectory, input_count, input_start_index)
    input_messages_sha256 = trajectory.initial_input_messages_sha256
    if input_messages_sha256 is None:
        raise _promotion_error(SessionPromotionErrorCode.INPUT_EVIDENCE_UNAVAILABLE)
    if session_input_messages_sha256(source_messages) != input_messages_sha256:
        raise _promotion_error(SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT)
    messages, redactions_applied = _sanitized_user_messages(app, source_messages)
    try:
        return PromotableRunInputV1(
            messages=messages,
            redactions_applied=(redactions_applied or bool(trajectory.input_redactions_applied)),
        )
    except ValueError as exc:
        raise _promotion_error(SessionPromotionErrorCode.INPUT_LIMIT_EXCEEDED) from exc


def promotable_run_input(
    app: CayuApp,
    trajectory: Trajectory,
    *,
    source_agent_name: str,
) -> PromotableRunInputV1:
    """Return safe initial input only when one trajectory is honestly replayable in v1."""

    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp.")
    validated_trajectory = _validated_trajectory_for_promotion(trajectory)
    return _promotable_run_input_from_validated(
        app,
        validated_trajectory,
        source_agent_name=source_agent_name,
    )
