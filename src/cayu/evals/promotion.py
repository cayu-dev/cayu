from __future__ import annotations

import hashlib
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, Field, StrictBool, StrictStr, field_validator, model_validator

from cayu._validation import canonical_durable_json_bytes, json_utf8_size_within_limit
from cayu.core.events import EventType
from cayu.core.messages import Message, MessageRole, TextPart
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_MESSAGES_PER_CASE,
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    EvaluationSourceIdentityV1,
    PricingProfileIdentityV1,
    RootStatusAssertionSpec,
    RunInputSpec,
    _bounded_durable_text,
    _content_revision,
    _model_python_input,
    _ordered_sequence_argument,
    _ordered_sequence_input,
    _portable_id,
    _SchemaV1PortableModel,
    _sha256_hex,
    _sha256_revision,
    pricing_profile_identity,
)
from cayu.evals.models import (
    Trajectory,
    _model_instance_python_input,
    _trajectory_promotion_capture_sha256,
    _validate_trajectory_record_contract,
)
from cayu.evals.evidence import AssertionEvidenceView, project_assertion_evidence_view
from cayu.runtime.app import CayuApp
from cayu.runtime.costs import PriceBook
from cayu.runtime.sessions import SessionStatus, session_input_messages_sha256

PROMOTABLE_RUN_INPUT_SCHEMA_VERSION = 1
PROMOTION_SOURCE_SCHEMA_VERSION = 1
PROMOTION_CANDIDATE_SCHEMA_VERSION = 1
PROMOTION_CANDIDATE_MAX_BYTES = 16 << 20

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _validate_exact_model(value: _ModelT, model_type: type[_ModelT], field_name: str) -> _ModelT:
    if type(value) is not model_type:
        raise TypeError(f"{field_name} must be an exact {model_type.__name__}.")
    return model_type.model_validate(_model_python_input(value))


class PromotableRunInputV1(_SchemaV1PortableModel):
    """Sanitized, text-only caller input proven to belong to one fresh invocation."""

    schema_version: Literal[1] = PROMOTABLE_RUN_INPUT_SCHEMA_VERSION
    revision: StrictStr
    messages: tuple[CorpusUserMessageSpec, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_MESSAGES_PER_CASE,
    )
    redactions_applied: StrictBool = False

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("messages", mode="before")
    @classmethod
    def validate_messages_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_run_input_contract(self) -> PromotableRunInputV1:
        RunInputSpec(messages=self.messages)
        expected = _content_revision(self.model_dump(mode="json"), "promotable run input")
        if self.revision != expected:
            raise ValueError("Promotable run input revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        messages: Sequence[CorpusUserMessageSpec],
        redactions_applied: bool = False,
    ) -> PromotableRunInputV1:
        ordered_messages = _ordered_sequence_argument(messages, "messages")
        validated_messages = tuple(
            _validate_exact_model(message, CorpusUserMessageSpec, "messages")
            for message in ordered_messages
        )
        RunInputSpec(messages=validated_messages)
        if type(redactions_applied) is not bool:
            raise TypeError("redactions_applied must be a bool.")
        document = {
            "schema_version": PROMOTABLE_RUN_INPUT_SCHEMA_VERSION,
            "messages": [message.model_dump(mode="json") for message in validated_messages],
            "redactions_applied": redactions_applied,
        }
        return cls(
            schema_version=PROMOTABLE_RUN_INPUT_SCHEMA_VERSION,
            revision=_content_revision(document, "promotable run input"),
            messages=validated_messages,
            redactions_applied=redactions_applied,
        )

    def to_run_input_spec(self) -> RunInputSpec:
        return RunInputSpec.model_validate(
            {"messages": [message.model_dump(mode="json") for message in self.messages]}
        )


class PromotionWarningCode(StrEnum):
    """Stable, non-blocking fact surfaced with one editable candidate."""

    INPUT_REDACTED = "input_redacted"
    SOURCE_RUN_FAILED = "source_run_failed"


class PromotionSourceV1(_SchemaV1PortableModel):
    """Safe diagnostic capture provenance without executable or session authority."""

    schema_version: Literal[1] = PROMOTION_SOURCE_SCHEMA_VERSION
    source_agent_name: StrictStr
    application_release_id: StrictStr
    app_manifest_schema_version: StrictStr
    app_manifest_fingerprint: StrictStr
    input_revision: StrictStr
    input_redactions_applied: StrictBool
    evidence_revision: StrictStr
    evidence_policy_revision: StrictStr
    pricing_profile_fingerprint: StrictStr | None = None
    source_label: StrictStr | None = None

    @field_validator("source_agent_name", "application_release_id")
    @classmethod
    def validate_source_identity(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("app_manifest_schema_version")
    @classmethod
    def validate_manifest_schema_version(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=32,
            nonblank=True,
            clean=True,
        )

    @field_validator("source_label")
    @classmethod
    def validate_source_label(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("app_manifest_fingerprint")
    @classmethod
    def validate_manifest_fingerprint(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator(
        "evidence_revision",
        "evidence_policy_revision",
        "input_revision",
        "pricing_profile_fingerprint",
    )
    @classmethod
    def validate_content_revision(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    def case_source(self) -> EvaluationSourceIdentityV1:
        """Return the corpus-native subset persisted with the promoted case."""

        return EvaluationSourceIdentityV1(
            application_release_id=self.application_release_id,
            app_manifest_schema_version=self.app_manifest_schema_version,
            app_manifest_fingerprint=self.app_manifest_fingerprint,
            evidence_revision=self.evidence_revision,
        )


def _promotion_warnings(
    source: PromotionSourceV1,
    evidence: AssertionEvidenceView,
) -> tuple[PromotionWarningCode, ...]:
    warnings: list[PromotionWarningCode] = []
    if source.input_redactions_applied:
        warnings.append(PromotionWarningCode.INPUT_REDACTED)
    if evidence.root_status == SessionStatus.FAILED.value:
        warnings.append(PromotionWarningCode.SOURCE_RUN_FAILED)
    return tuple(sorted(warnings, key=lambda warning: warning.value))


class PromotionCandidateV1(_SchemaV1PortableModel):
    """One deterministic, editable case candidate and its public-safe evidence."""

    schema_version: Literal[1] = PROMOTION_CANDIDATE_SCHEMA_VERSION
    revision: StrictStr
    target_key: StrictStr
    source: PromotionSourceV1
    evidence_policy: EvaluationEvidencePolicySpec
    pricing_profile: PricingProfileIdentityV1 | None = None
    evidence: AssertionEvidenceView
    suite: EvalSuiteSpec
    case: EvalCaseSpec
    warnings: tuple[PromotionWarningCode, ...] = Field(max_length=2)

    @field_validator("target_key")
    @classmethod
    def validate_target_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("warnings", mode="before")
    @classmethod
    def validate_warnings_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> PromotionCandidateV1:
        if self.source.evidence_revision != self.evidence.revision:
            raise ValueError("Promotion source and evidence revisions do not match.")
        if self.source.evidence_policy_revision != self.evidence_policy.revision:
            raise ValueError("Promotion source and evidence-policy revisions do not match.")
        if self.evidence.policy_revision != self.evidence_policy.revision:
            raise ValueError("Promotion evidence and evidence-policy revisions do not match.")
        expected_pricing_fingerprint = (
            None if self.pricing_profile is None else self.pricing_profile.fingerprint
        )
        if self.source.pricing_profile_fingerprint != expected_pricing_fingerprint:
            raise ValueError("Promotion source and pricing-profile identities do not match.")
        if self.evidence.pricing_profile_fingerprint not in {
            None,
            expected_pricing_fingerprint,
        }:
            raise ValueError("Promotion evidence uses a different pricing profile.")
        if self.case.source != self.source.case_source():
            raise ValueError("Promotion case source does not match candidate provenance.")
        if self.case.suite_id != self.suite.id:
            raise ValueError("Promotion case must reference the candidate suite.")
        expected_warnings = _promotion_warnings(self.source, self.evidence)
        if self.warnings != expected_warnings:
            raise ValueError("Promotion warnings do not match captured source facts.")
        if not json_utf8_size_within_limit(self, PROMOTION_CANDIDATE_MAX_BYTES):
            raise ValueError(
                f"Promotion candidate exceeds {PROMOTION_CANDIDATE_MAX_BYTES} canonical JSON bytes."
            )
        expected = _content_revision(self.model_dump(mode="json"), "promotion candidate")
        if self.revision != expected:
            raise ValueError("Promotion candidate revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        target_key: str,
        source: PromotionSourceV1,
        evidence_policy: EvaluationEvidencePolicySpec,
        evidence: AssertionEvidenceView,
        suite: EvalSuiteSpec,
        case: EvalCaseSpec,
        pricing_profile: PricingProfileIdentityV1 | None = None,
    ) -> PromotionCandidateV1:
        validated_target_key = _portable_id(target_key, "target_key")
        validated_source = _validate_exact_model(source, PromotionSourceV1, "source")
        validated_policy = _validate_exact_model(
            evidence_policy,
            EvaluationEvidencePolicySpec,
            "evidence_policy",
        )
        validated_evidence = _validate_exact_model(evidence, AssertionEvidenceView, "evidence")
        validated_suite = _validate_exact_model(suite, EvalSuiteSpec, "suite")
        validated_case = _validate_exact_model(case, EvalCaseSpec, "case")
        validated_pricing = (
            None
            if pricing_profile is None
            else _validate_exact_model(
                pricing_profile,
                PricingProfileIdentityV1,
                "pricing_profile",
            )
        )
        warnings = _promotion_warnings(validated_source, validated_evidence)
        document = {
            "schema_version": PROMOTION_CANDIDATE_SCHEMA_VERSION,
            "target_key": validated_target_key,
            "source": validated_source.model_dump(mode="json"),
            "evidence_policy": validated_policy.model_dump(mode="json"),
            "pricing_profile": (
                None if validated_pricing is None else validated_pricing.model_dump(mode="json")
            ),
            "evidence": validated_evidence.model_dump(mode="json"),
            "suite": validated_suite.model_dump(mode="json"),
            "case": validated_case.model_dump(mode="json"),
            "warnings": [warning.value for warning in warnings],
        }
        return cls(
            schema_version=PROMOTION_CANDIDATE_SCHEMA_VERSION,
            revision=_content_revision(document, "promotion candidate"),
            target_key=validated_target_key,
            source=validated_source,
            evidence_policy=validated_policy,
            pricing_profile=validated_pricing,
            evidence=validated_evidence,
            suite=validated_suite,
            case=validated_case,
            warnings=warnings,
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
        return PromotableRunInputV1.create(
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


def _safe_candidate_text(
    app: CayuApp,
    value: str,
    field_name: str,
    *,
    max_chars: int,
) -> str:
    validated = _bounded_durable_text(
        value,
        field_name,
        max_chars=max_chars,
        nonblank=True,
        clean=True,
    )
    try:
        redacted = app.redact_json(validated)
    except Exception as exc:
        raise ValueError(
            f"{field_name} could not cross the application redaction boundary."
        ) from exc
    if type(redacted) is not str or redacted != validated:
        raise ValueError(f"{field_name} contains a workload secret.")
    return validated


def _safe_portable_id(app: CayuApp, value: str, field_name: str) -> str:
    validated = _portable_id(value, field_name)
    return _safe_candidate_text(
        app,
        validated,
        field_name,
        max_chars=128,
    )


def _safe_pricing_profile_identity(
    app: CayuApp,
    identity: PricingProfileIdentityV1 | None,
) -> PricingProfileIdentityV1 | None:
    if identity is None:
        return None
    validated = _validate_exact_model(
        identity,
        PricingProfileIdentityV1,
        "pricing_profile",
    )
    _safe_candidate_text(
        app,
        validated.price_book_version,
        "pricing_profile.price_book_version",
        max_chars=256,
    )
    _safe_candidate_text(
        app,
        validated.generated_at,
        "pricing_profile.generated_at",
        max_chars=256,
    )
    for index, currency in enumerate(validated.currencies):
        _safe_candidate_text(
            app,
            currency,
            f"pricing_profile.currencies[{index}]",
            max_chars=16,
        )
    return validated


def _default_suite_id(target_key: str) -> str:
    readable = f"{target_key}.regressions"
    if len(readable) <= 128:
        return readable
    digest = hashlib.sha256(target_key.encode("ascii")).hexdigest()
    return f"regressions-{digest}"


def _default_case_id(
    target_key: str,
    source: EvaluationSourceIdentityV1,
    input_revision: str,
) -> str:
    identity = {
        "target_key": target_key,
        "source": source.model_dump(mode="json"),
        "input_revision": _sha256_revision(input_revision, "input_revision"),
    }
    digest = hashlib.sha256(
        canonical_durable_json_bytes(identity, "promotion case identity")
    ).hexdigest()
    return f"case-{digest}"


def build_promotion_candidate(
    app: CayuApp,
    trajectory: Trajectory,
    *,
    target_key: str,
    source_agent_name: str,
    application_release_id: str,
    evidence_policy: EvaluationEvidencePolicySpec,
    pricing: PriceBook | None = None,
    source_label: str | None = None,
    project_root: str | Path | None = None,
) -> PromotionCandidateV1:
    """Build one deterministic, public-safe candidate from eligible terminal evidence."""

    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp.")
    validated_trajectory = _validated_trajectory_for_promotion(trajectory)
    validated_target_key = _safe_portable_id(app, target_key, "target_key")
    safe_source_agent_name = _safe_candidate_text(
        app,
        source_agent_name,
        "source_agent_name",
        max_chars=256,
    )
    validated_policy = _validate_exact_model(
        evidence_policy,
        EvaluationEvidencePolicySpec,
        "evidence_policy",
    )
    safe_release_id = _safe_candidate_text(
        app,
        application_release_id,
        "application_release_id",
        max_chars=256,
    )
    safe_source_label = (
        None
        if source_label is None
        else _safe_candidate_text(
            app,
            source_label,
            "source_label",
            max_chars=256,
        )
    )
    run_input = _promotable_run_input_from_validated(
        app,
        validated_trajectory,
        source_agent_name=safe_source_agent_name,
    )
    evidence = project_assertion_evidence_view(
        app,
        validated_trajectory,
        evidence_policy=validated_policy,
    )
    manifest = app.describe(project_root=project_root)
    pricing_profile = _safe_pricing_profile_identity(
        app,
        None if pricing is None else pricing_profile_identity(pricing),
    )
    source = PromotionSourceV1(
        source_agent_name=safe_source_agent_name,
        application_release_id=safe_release_id,
        app_manifest_schema_version=manifest.schema_version,
        app_manifest_fingerprint=manifest.fingerprint,
        input_revision=run_input.revision,
        input_redactions_applied=run_input.redactions_applied,
        evidence_revision=evidence.revision,
        evidence_policy_revision=validated_policy.revision,
        pricing_profile_fingerprint=(
            None if pricing_profile is None else pricing_profile.fingerprint
        ),
        source_label=safe_source_label,
    )
    case_source = source.case_source()
    suite_id = _safe_portable_id(
        app,
        _default_suite_id(validated_target_key),
        "suite.id",
    )
    suite_name = _safe_candidate_text(
        app,
        f"{validated_target_key} regressions",
        "suite.name",
        max_chars=256,
    )
    suite = EvalSuiteSpec.create(
        id=suite_id,
        name=suite_name,
        description="Regression cases promoted from bounded Cayu production evidence.",
    )
    case_id = _default_case_id(
        validated_target_key,
        case_source,
        source.input_revision,
    )
    case = EvalCaseSpec.create(
        id=case_id,
        suite_id=suite.id,
        name=(
            safe_source_label
            if safe_source_label is not None
            else f"Captured regression {case_id.removeprefix('case-')[:12]}"
        ),
        description="Editable regression candidate derived from public-safe captured evidence.",
        source=case_source,
        input=run_input.to_run_input_spec(),
        assertions=(
            RootStatusAssertionSpec(
                id="session-completed",
                description="The regression must complete successfully.",
                expected="completed",
            ),
        ),
    )
    return PromotionCandidateV1.create(
        target_key=validated_target_key,
        source=source,
        evidence_policy=validated_policy,
        pricing_profile=pricing_profile,
        evidence=evidence,
        suite=suite,
        case=case,
    )
