from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

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

from cayu._validation import canonical_durable_json_bytes, require_durable_text
from cayu.artifacts import (
    ArtifactScope,
    ArtifactStore,
    ArtifactStoreUnavailableError,
    InvalidArtifactIdError,
    copy_artifact_read_result,
)
from cayu.artifacts.attachments import (
    MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY,
    FileAttachment,
)
from cayu.core.events import Event, EventType, event_payload_authority_is_runtime_generated
from cayu.core.messages import FilePart, Message, MessageRole, TextPart
from cayu.evals.corpus import EvaluationSourceIdentityV1, _portable_id
from cayu.evals.scenario import (
    EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS,
    EVAL_SCENARIO_MAX_EVENTS,
    EVAL_SCENARIO_MAX_TOTAL_ARTIFACT_BYTES,
    EvalScenarioDocumentV2,
    ScenarioApprovalCheckpointEventV2,
    ScenarioArtifactRequirementV2,
    ScenarioEventV2,
    ScenarioFilePartV2,
    ScenarioInitialInputEventV2,
    ScenarioInputPartV2,
    ScenarioInputV2,
    ScenarioQueuedInputEventV2,
    ScenarioResumedInputEventV2,
    ScenarioTextPartV2,
    ScenarioUserMessageV2,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import (
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    SessionInputContractEvidence,
    SessionMessageDeliveryMode,
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    TerminalSessionEvidenceLimits,
    TranscriptRecord,
    parse_session_input_contract_evidence,
    session_input_messages_sha256,
)

SCENARIO_CAPTURE_MAX_DIAGNOSTICS = 1_024
SCENARIO_CAPTURE_ARTIFACT_READ_CONCURRENCY = 8


class ScenarioCaptureDiagnosticCode(StrEnum):
    """Stable factual reason why retained stimuli cannot become one scenario."""

    TERMINAL_EVIDENCE_NOT_SUPPORTED = "terminal_evidence_not_supported"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_TERMINAL = "session_not_terminal"
    CAPTURED_EVIDENCE_INCOMPLETE = "captured_evidence_incomplete"
    CAPTURED_EVIDENCE_CONTRADICTORY = "captured_evidence_contradictory"
    CAPTURE_LIMIT_EXCEEDED = "capture_limit_exceeded"
    SOURCE_AGENT_MISMATCH = "source_agent_mismatch"
    SOURCE_PAYLOAD_UNAVAILABLE = "source_payload_unavailable"
    SOURCE_PAYLOAD_REDACTED = "source_payload_redacted"
    SOURCE_PAYLOAD_INCONSISTENT = "source_payload_inconsistent"
    SOURCE_INPUT_ROLE_UNSUPPORTED = "source_input_role_unsupported"
    SOURCE_INPUT_PART_UNSUPPORTED = "source_input_part_unsupported"
    ARTIFACT_NOT_RETAINED = "artifact_not_retained"
    ARTIFACT_ACCESS_DENIED = "artifact_access_denied"
    ARTIFACT_STORE_UNAVAILABLE = "artifact_store_unavailable"
    ARTIFACT_CONTENT_INCONSISTENT = "artifact_content_inconsistent"
    SCENARIO_LIMIT_EXCEEDED = "scenario_limit_exceeded"
    SOURCE_CHANGED = "source_changed"


_DIAGNOSTIC_COPY: dict[ScenarioCaptureDiagnosticCode, tuple[str, str]] = {
    ScenarioCaptureDiagnosticCode.TERMINAL_EVIDENCE_NOT_SUPPORTED: (
        "The session store cannot provide bounded terminal evidence.",
        "Use a SessionStore that supports terminal-session evidence.",
    ),
    ScenarioCaptureDiagnosticCode.SESSION_NOT_FOUND: (
        "The source session does not exist.",
        "Choose a retained session that is visible to the current operator.",
    ),
    ScenarioCaptureDiagnosticCode.SESSION_NOT_TERMINAL: (
        "The source session has not reached a supported terminal state.",
        "Wait for the session to complete or fail, then capture it again.",
    ),
    ScenarioCaptureDiagnosticCode.CAPTURED_EVIDENCE_INCOMPLETE: (
        "The retained session evidence is incomplete.",
        "Repair or restore the missing durable evidence before capturing a scenario.",
    ),
    ScenarioCaptureDiagnosticCode.CAPTURED_EVIDENCE_CONTRADICTORY: (
        "The retained session evidence contains contradictory boundaries.",
        "Inspect and repair the durable session evidence before capturing a scenario.",
    ),
    ScenarioCaptureDiagnosticCode.CAPTURE_LIMIT_EXCEEDED: (
        "The retained session evidence exceeds the configured capture limit.",
        "Raise the bounded capture limit or select a smaller source session.",
    ),
    ScenarioCaptureDiagnosticCode.SOURCE_AGENT_MISMATCH: (
        "The source session does not belong to the selected eval target agent.",
        "Select the eval target published for the source session's agent.",
    ),
    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_UNAVAILABLE: (
        "A retained input boundary does not contain enough material for exact reconstruction.",
        "Author the missing stimulus explicitly or capture a newer session with input evidence.",
    ),
    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_REDACTED: (
        "A source stimulus crossed a redaction boundary and its original value is unavailable.",
        "Review and author a safe replacement value before saving the scenario.",
    ),
    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_INCONSISTENT: (
        "A retained input boundary contradicts its transcript material.",
        "Inspect the source session and repair the inconsistent durable evidence.",
    ),
    ScenarioCaptureDiagnosticCode.SOURCE_INPUT_ROLE_UNSUPPORTED: (
        "A caller-input boundary contains a non-user message.",
        "Author an equivalent user stimulus explicitly.",
    ),
    ScenarioCaptureDiagnosticCode.SOURCE_INPUT_PART_UNSUPPORTED: (
        "A caller-input boundary contains a part that scenario v2 cannot represent.",
        "Replace the unsupported part with text, JSON, or a retained file reference.",
    ),
    ScenarioCaptureDiagnosticCode.ARTIFACT_NOT_RETAINED: (
        "A referenced input artifact is no longer retained.",
        "Restore the artifact or bind an immutable replacement fixture.",
    ),
    ScenarioCaptureDiagnosticCode.ARTIFACT_ACCESS_DENIED: (
        "The source artifact cannot be read under the current access boundary.",
        "Obtain artifact read access or select an authorized replacement fixture.",
    ),
    ScenarioCaptureDiagnosticCode.ARTIFACT_STORE_UNAVAILABLE: (
        "The source artifact store is temporarily unavailable.",
        "Restore the artifact store and retry scenario capture.",
    ),
    ScenarioCaptureDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT: (
        "A retained artifact no longer matches the source attachment metadata.",
        "Restore the exact artifact or select an immutable replacement fixture.",
    ),
    ScenarioCaptureDiagnosticCode.SCENARIO_LIMIT_EXCEEDED: (
        "The reconstructed stimuli exceed a scenario-v2 safety limit.",
        "Reduce or split the scenario before saving it.",
    ),
    ScenarioCaptureDiagnosticCode.SOURCE_CHANGED: (
        "The source session changed while its scenario was being captured.",
        "Retry capture against the current terminal session revision.",
    ),
}


class _ScenarioCaptureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ScenarioCaptureDiagnosticV2(_ScenarioCaptureModel):
    code: ScenarioCaptureDiagnosticCode
    message: StrictStr = Field(min_length=1, max_length=512)
    remediation: StrictStr = Field(min_length=1, max_length=512)
    event_sequence: StrictInt | None = Field(default=None, ge=1)
    transcript_index: StrictInt | None = Field(default=None, ge=0)
    artifact_requirement_id: StrictStr | None = Field(default=None, max_length=128)

    @field_validator("message", "remediation")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_durable_text(value, info.field_name)

    @field_validator("artifact_requirement_id")
    @classmethod
    def validate_artifact_requirement_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _portable_id(value, "artifact_requirement_id")

    @model_validator(mode="after")
    def validate_copy(self) -> ScenarioCaptureDiagnosticV2:
        if (self.message, self.remediation) != _DIAGNOSTIC_COPY[self.code]:
            raise ValueError("Scenario capture diagnostic copy does not match its code.")
        return self


class ScenarioCaptureResultV2(_ScenarioCaptureModel):
    available: StrictBool
    scenario: EvalScenarioDocumentV2 | None = None
    diagnostics: tuple[ScenarioCaptureDiagnosticV2, ...] = Field(
        default_factory=tuple,
        max_length=SCENARIO_CAPTURE_MAX_DIAGNOSTICS,
    )

    @model_validator(mode="after")
    def validate_availability(self) -> ScenarioCaptureResultV2:
        if self.available != (self.scenario is not None and not self.diagnostics):
            raise ValueError("Scenario capture availability contradicts its result material.")
        if not self.available and not self.diagnostics:
            raise ValueError("Unavailable scenario capture requires at least one diagnostic.")
        return self


@dataclass(frozen=True, slots=True)
class _InputStimulus:
    anchor_sequence: int
    materialized_sequence: int
    kind: Literal["initial", "queued", "resumed"]
    messages: tuple[Message, ...]
    delivery_mode: Literal["next_turn", "on_idle"] | None = None


@dataclass(frozen=True, slots=True)
class _ApprovalStimulus:
    anchor_sequence: int
    tool_name: str
    occurrence: int


_Stimulus = _InputStimulus | _ApprovalStimulus


def _diagnostic(
    code: ScenarioCaptureDiagnosticCode,
    *,
    event_sequence: int | None = None,
    transcript_index: int | None = None,
    artifact_requirement_id: str | None = None,
) -> ScenarioCaptureDiagnosticV2:
    message, remediation = _DIAGNOSTIC_COPY[code]
    return ScenarioCaptureDiagnosticV2(
        code=code,
        message=message,
        remediation=remediation,
        event_sequence=event_sequence,
        transcript_index=transcript_index,
        artifact_requirement_id=artifact_requirement_id,
    )


def _unavailable(
    *diagnostics: ScenarioCaptureDiagnosticV2,
) -> ScenarioCaptureResultV2:
    bounded = tuple(diagnostics[:SCENARIO_CAPTURE_MAX_DIAGNOSTICS])
    if not bounded:
        bounded = (_diagnostic(ScenarioCaptureDiagnosticCode.CAPTURED_EVIDENCE_INCOMPLETE),)
    return ScenarioCaptureResultV2(available=False, diagnostics=bounded)


def _terminal_diagnostic(exc: TerminalSessionEvidenceError) -> ScenarioCaptureDiagnosticV2:
    code = exc.code
    if code is TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND:
        mapped = ScenarioCaptureDiagnosticCode.SESSION_NOT_FOUND
    elif code in {
        TerminalSessionEvidenceErrorCode.SESSION_NOT_TERMINAL,
        TerminalSessionEvidenceErrorCode.SESSION_INTERRUPTED,
    }:
        mapped = ScenarioCaptureDiagnosticCode.SESSION_NOT_TERMINAL
    elif code in {
        TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
        TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED,
        TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
        TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
        TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
    }:
        mapped = ScenarioCaptureDiagnosticCode.CAPTURE_LIMIT_EXCEEDED
    elif code in {
        TerminalSessionEvidenceErrorCode.INITIAL_TRANSCRIPT_INCOMPLETE,
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_MISSING,
    }:
        mapped = ScenarioCaptureDiagnosticCode.CAPTURED_EVIDENCE_INCOMPLETE
    else:
        mapped = ScenarioCaptureDiagnosticCode.CAPTURED_EVIDENCE_CONTRADICTORY
    return _diagnostic(mapped)


def _input_contract(event: Event) -> SessionInputContractEvidence | None:
    raw = event.payload.get(SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY)
    if type(raw) is not str or not event_payload_authority_is_runtime_generated(
        event,
        field_name=SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
        value=raw,
    ):
        return None
    try:
        return parse_session_input_contract_evidence(raw)
    except ValueError:
        return None


def _records_for_contract(
    transcript_by_index: Mapping[int, TranscriptRecord],
    contract: SessionInputContractEvidence,
) -> tuple[TranscriptRecord, ...] | None:
    records = tuple(
        transcript_by_index.get(index)
        for index in range(
            contract.message_start_index,
            contract.message_start_index + contract.message_count,
        )
    )
    if any(record is None for record in records):
        return None
    selected = tuple(record for record in records if record is not None)
    if session_input_messages_sha256(tuple(record.message for record in selected)) != (
        contract.messages_sha256
    ):
        return None
    return selected


def _capture_input_stimuli(
    evidence: TerminalSessionEvidence,
) -> tuple[tuple[_Stimulus, ...], tuple[ScenarioCaptureDiagnosticV2, ...]]:
    diagnostics: list[ScenarioCaptureDiagnosticV2] = []
    stimuli: list[_Stimulus] = []
    claimed_indexes: set[int] = set()
    events = tuple(evidence.events)
    transcript_by_index = {record.index: record for record in evidence.transcript}

    started = tuple(record for record in events if record.event.type == EventType.SESSION_STARTED)
    if len(started) != 1:
        return (), (_diagnostic(ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_INCONSISTENT),)
    initial_record = started[0]
    initial_contract = _input_contract(initial_record.event)
    if initial_contract is None or initial_contract.message_count < 1:
        diagnostics.append(
            _diagnostic(
                ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_UNAVAILABLE,
                event_sequence=initial_record.sequence,
            )
        )
    else:
        initial_records = _records_for_contract(transcript_by_index, initial_contract)
        if initial_records is None:
            diagnostics.append(
                _diagnostic(
                    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_INCONSISTENT,
                    event_sequence=initial_record.sequence,
                    transcript_index=initial_contract.message_start_index,
                )
            )
        else:
            claimed_indexes.update(record.index for record in initial_records)
            stimuli.append(
                _InputStimulus(
                    anchor_sequence=0,
                    materialized_sequence=initial_record.sequence,
                    kind="initial",
                    messages=tuple(record.message for record in initial_records),
                )
            )
            if initial_contract.redactions_applied:
                diagnostics.append(
                    _diagnostic(
                        ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_REDACTED,
                        event_sequence=initial_record.sequence,
                        transcript_index=initial_contract.message_start_index,
                    )
                )

    queued_by_id: dict[str, list[tuple[int, Event]]] = {}
    delivered_queue_ids: set[str] = set()
    for record in events:
        event = record.event
        if event.type != EventType.SESSION_MESSAGE_QUEUED:
            continue
        queue_id = event.payload.get("queue_id")
        if type(queue_id) is not str or not queue_id:
            diagnostics.append(
                _diagnostic(
                    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_INCONSISTENT,
                    event_sequence=record.sequence,
                )
            )
            continue
        queued_by_id.setdefault(queue_id, []).append((record.sequence, event))

    for record in events:
        event = record.event
        if event.type != EventType.SESSION_MESSAGE_DELIVERED:
            continue
        queue_id = event.payload.get("queue_id")
        contract = _input_contract(event)
        transcript_cursor = event.payload.get("transcript_cursor")
        delivery_mode = event.payload.get("delivery_mode")
        accepted = queued_by_id.get(queue_id) if type(queue_id) is str else None
        if contract is None or accepted is None or len(accepted) != 1:
            diagnostics.append(
                _diagnostic(
                    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_UNAVAILABLE,
                    event_sequence=record.sequence,
                )
            )
            continue
        accepted_sequence, accepted_event = accepted[0]
        accepted_contract = _input_contract(accepted_event)
        accepted_transcript_cursor = accepted_event.payload.get("transcript_cursor")
        if accepted_contract is None:
            diagnostics.append(
                _diagnostic(
                    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_UNAVAILABLE,
                    event_sequence=accepted_sequence,
                )
            )
            continue
        if (
            type(queue_id) is not str
            or not queue_id
            or contract.message_count != 1
            or type(transcript_cursor) is not int
            or contract.message_start_index != transcript_cursor - 1
            or delivery_mode
            not in {
                SessionMessageDeliveryMode.NEXT_TURN.value,
                SessionMessageDeliveryMode.ON_IDLE.value,
            }
            or accepted_sequence >= record.sequence
            or accepted_event.payload.get("delivery_mode") != delivery_mode
            or accepted_event.payload.get("ordering_key") != event.payload.get("ordering_key")
            or type(accepted_transcript_cursor) is not int
            or accepted_contract.message_count != 1
            or accepted_contract.message_start_index != accepted_transcript_cursor
        ):
            diagnostics.append(
                _diagnostic(
                    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_INCONSISTENT,
                    event_sequence=record.sequence,
                )
            )
            continue
        selected = _records_for_contract(transcript_by_index, contract)
        if (
            selected is None
            or selected[0].index in claimed_indexes
            or selected[0].interaction_id != event.interaction_id
            or accepted_contract.messages_sha256 != contract.messages_sha256
        ):
            diagnostics.append(
                _diagnostic(
                    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_INCONSISTENT,
                    event_sequence=record.sequence,
                    transcript_index=contract.message_start_index,
                )
            )
            continue
        claimed_indexes.add(selected[0].index)
        delivered_queue_ids.add(queue_id)
        stimuli.append(
            _InputStimulus(
                anchor_sequence=accepted_sequence,
                materialized_sequence=record.sequence,
                kind="queued",
                messages=(selected[0].message,),
                delivery_mode=delivery_mode,
            )
        )
        if accepted_contract.redactions_applied:
            diagnostics.append(
                _diagnostic(
                    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_REDACTED,
                    event_sequence=accepted_sequence,
                    transcript_index=contract.message_start_index,
                )
            )

    for queue_id, accepted in queued_by_id.items():
        if queue_id in delivered_queue_ids:
            continue
        diagnostics.append(
            _diagnostic(
                ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_UNAVAILABLE,
                event_sequence=accepted[0][0],
            )
        )

    interaction_starts = tuple(
        record
        for record in events
        if record.event.type == EventType.INTERACTION_STARTED
        and record.event.interaction_id is not None
    )
    for record in events:
        event = record.event
        if event.type != EventType.SESSION_RESUMED:
            continue
        appended_messages = event.payload.get("appended_messages")
        if appended_messages is None:
            if event.payload.get("interruption_type") == "user_input_required":
                diagnostics.append(
                    _diagnostic(
                        ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_UNAVAILABLE,
                        event_sequence=record.sequence,
                    )
                )
            continue
        contract = _input_contract(event)
        if (
            type(appended_messages) is not int
            or appended_messages < 1
            or contract is None
            or contract.message_count != appended_messages
        ):
            diagnostics.append(
                _diagnostic(
                    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_UNAVAILABLE,
                    event_sequence=record.sequence,
                )
            )
            continue
        selected = _records_for_contract(transcript_by_index, contract)
        selected_interaction_ids = (
            set() if selected is None else {item.interaction_id for item in selected}
        )
        interaction_id = (
            next(iter(selected_interaction_ids)) if len(selected_interaction_ids) == 1 else None
        )
        candidate_starts = tuple(
            candidate
            for candidate in interaction_starts
            if candidate.sequence < record.sequence
            and candidate.event.interaction_id == interaction_id
        )
        if (
            selected is None
            or interaction_id is None
            or len(candidate_starts) != 1
            or any(item.index in claimed_indexes for item in selected)
            or any(item.interaction_id != interaction_id for item in selected)
        ):
            diagnostics.append(
                _diagnostic(
                    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_INCONSISTENT,
                    event_sequence=record.sequence,
                    transcript_index=contract.message_start_index,
                )
            )
            continue
        claimed_indexes.update(item.index for item in selected)
        stimuli.append(
            _InputStimulus(
                anchor_sequence=record.sequence,
                materialized_sequence=record.sequence,
                kind="resumed",
                messages=tuple(item.message for item in selected),
            )
        )
        if contract.redactions_applied:
            diagnostics.append(
                _diagnostic(
                    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_REDACTED,
                    event_sequence=record.sequence,
                    transcript_index=contract.message_start_index,
                )
            )
    unclaimed_user_records = tuple(
        record
        for record in evidence.transcript
        if record.message.role == MessageRole.USER and record.index not in claimed_indexes
    )
    diagnostics.extend(
        _diagnostic(
            ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_UNAVAILABLE,
            transcript_index=record.index,
        )
        for record in unclaimed_user_records
    )

    approval_ids: set[str] = set()
    approval_occurrences: dict[str, int] = {}
    for record in events:
        event = record.event
        if event.type != EventType.TOOL_CALL_APPROVAL_REQUESTED:
            continue
        approval_id = event.payload.get("approval_id")
        tool_name = event.tool_name
        if type(approval_id) is not str or not approval_id or type(tool_name) is not str:
            diagnostics.append(
                _diagnostic(
                    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_INCONSISTENT,
                    event_sequence=record.sequence,
                )
            )
            continue
        if approval_id in approval_ids:
            continue
        approval_ids.add(approval_id)
        occurrence = approval_occurrences.get(tool_name, 0) + 1
        approval_occurrences[tool_name] = occurrence
        stimuli.append(
            _ApprovalStimulus(
                anchor_sequence=record.sequence,
                tool_name=tool_name,
                occurrence=occurrence,
            )
        )

    if len(stimuli) > EVAL_SCENARIO_MAX_EVENTS:
        diagnostics.append(_diagnostic(ScenarioCaptureDiagnosticCode.SCENARIO_LIMIT_EXCEEDED))
    return tuple(sorted(stimuli, key=lambda item: item.anchor_sequence)), tuple(diagnostics)


def _artifact_store_for_session(
    app: CayuApp, evidence: TerminalSessionEvidence
) -> ArtifactStore | None:
    environment_name = evidence.session.environment_name
    if environment_name is None:
        return None
    matches = tuple(
        registration.environment.artifact_store
        for registration in app.list_environment_registrations()
        if registration.spec.name == environment_name
        and isinstance(registration.environment.artifact_store, ArtifactStore)
    )
    return matches[0] if len(matches) == 1 else None


def _artifact_requirement_id(attachment: FileAttachment) -> str:
    identity = "\0".join(
        (
            attachment.artifact_id,
            attachment.filename,
            attachment.content_type,
            str(attachment.size_bytes),
        )
    ).encode("utf-8")
    return f"artifact.{hashlib.sha256(identity).hexdigest()}"


async def _resolve_artifact_requirements(
    app: CayuApp,
    evidence: TerminalSessionEvidence,
    attachments: Mapping[str, FileAttachment],
    *,
    expected_content_sha256: Mapping[str, str] | None = None,
) -> tuple[
    dict[str, ScenarioArtifactRequirementV2],
    tuple[ScenarioCaptureDiagnosticV2, ...],
]:
    if not attachments:
        return {}, ()
    if (
        len(attachments) > EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS
        or sum(attachment.size_bytes for attachment in attachments.values())
        > EVAL_SCENARIO_MAX_TOTAL_ARTIFACT_BYTES
    ):
        return {}, (_diagnostic(ScenarioCaptureDiagnosticCode.SCENARIO_LIMIT_EXCEEDED),)
    try:
        store = _artifact_store_for_session(app, evidence)
    except Exception:
        store = None
    if store is None:
        return {}, tuple(
            _diagnostic(
                ScenarioCaptureDiagnosticCode.ARTIFACT_STORE_UNAVAILABLE,
                artifact_requirement_id=_artifact_requirement_id(attachment),
            )
            for attachment in attachments.values()
        )

    semaphore = asyncio.Semaphore(SCENARIO_CAPTURE_ARTIFACT_READ_CONCURRENCY)

    async def resolve(
        attachment: FileAttachment,
    ) -> tuple[ScenarioArtifactRequirementV2 | None, ScenarioCaptureDiagnosticV2 | None]:
        requirement_id = _artifact_requirement_id(attachment)
        try:
            async with semaphore:
                read = copy_artifact_read_result(
                    await store.read_bytes(
                        attachment.artifact_id,
                        max_bytes=attachment.size_bytes,
                    ),
                    expected_artifact_id=attachment.artifact_id,
                    max_content_bytes=attachment.size_bytes,
                )
        except (FileNotFoundError, InvalidArtifactIdError):
            return None, _diagnostic(
                ScenarioCaptureDiagnosticCode.ARTIFACT_NOT_RETAINED,
                artifact_requirement_id=requirement_id,
            )
        except PermissionError:
            return None, _diagnostic(
                ScenarioCaptureDiagnosticCode.ARTIFACT_ACCESS_DENIED,
                artifact_requirement_id=requirement_id,
            )
        except (ArtifactStoreUnavailableError, OSError):
            return None, _diagnostic(
                ScenarioCaptureDiagnosticCode.ARTIFACT_STORE_UNAVAILABLE,
                artifact_requirement_id=requirement_id,
            )
        except (TypeError, ValueError):
            return None, _diagnostic(
                ScenarioCaptureDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT,
                artifact_requirement_id=requirement_id,
            )
        except Exception:
            return None, _diagnostic(
                ScenarioCaptureDiagnosticCode.ARTIFACT_STORE_UNAVAILABLE,
                artifact_requirement_id=requirement_id,
            )

        metadata = read.metadata
        owner_matches = (
            metadata.scope == ArtifactScope.ENVIRONMENT
            and metadata.environment_name == evidence.session.environment_name
        ) or (
            metadata.scope == ArtifactScope.SESSION and metadata.session_id == evidence.session.id
        )
        if (
            read.truncated
            or read.total_bytes != attachment.size_bytes
            or len(read.content) != attachment.size_bytes
            or metadata.id != attachment.artifact_id
            or metadata.filename != attachment.filename
            or metadata.content_type != attachment.content_type
            or metadata.size_bytes != attachment.size_bytes
            or not owner_matches
        ):
            return None, _diagnostic(
                ScenarioCaptureDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT,
                artifact_requirement_id=requirement_id,
            )
        try:
            reference = app.redact_json(attachment.artifact_id)
            filename = app.redact_json(attachment.filename)
            content_type = app.redact_json(attachment.content_type)
        except Exception:
            return None, _diagnostic(
                ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_REDACTED,
                artifact_requirement_id=requirement_id,
            )
        if (
            type(reference) is not str
            or reference != attachment.artifact_id
            or type(filename) is not str
            or filename != attachment.filename
            or type(content_type) is not str
            or content_type != attachment.content_type
        ):
            return None, _diagnostic(
                ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_REDACTED,
                artifact_requirement_id=requirement_id,
            )
        content_sha256 = hashlib.sha256(read.content).hexdigest()
        expected_digest = (
            None
            if expected_content_sha256 is None
            else expected_content_sha256.get(attachment.artifact_id)
        )
        if expected_digest is not None and content_sha256 != expected_digest:
            return None, _diagnostic(
                ScenarioCaptureDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT,
                artifact_requirement_id=requirement_id,
            )
        try:
            requirement = ScenarioArtifactRequirementV2(
                id=requirement_id,
                source="artifact_reference",
                reference=reference,
                content_sha256=content_sha256,
                filename=filename,
                content_type=content_type,
                size_bytes=attachment.size_bytes,
            )
        except (TypeError, ValueError):
            return None, _diagnostic(
                ScenarioCaptureDiagnosticCode.SCENARIO_LIMIT_EXCEEDED,
                artifact_requirement_id=requirement_id,
            )
        return requirement, None

    resolved = await asyncio.gather(*(resolve(item) for item in attachments.values()))
    requirements: dict[str, ScenarioArtifactRequirementV2] = {}
    diagnostics: list[ScenarioCaptureDiagnosticV2] = []
    for requirement, diagnostic in resolved:
        if diagnostic is not None:
            diagnostics.append(diagnostic)
        elif requirement is not None:
            requirements[requirement.reference or ""] = requirement
    return requirements, tuple(diagnostics)


def _attachments(stimuli: Sequence[_Stimulus]) -> dict[str, FileAttachment]:
    attachments: dict[str, FileAttachment] = {}
    for stimulus in stimuli:
        if not isinstance(stimulus, _InputStimulus):
            continue
        for message in stimulus.messages:
            for part in message.content:
                if type(part) is not FilePart:
                    continue
                attachment = FileAttachment.model_validate(part.attachment)
                if attachment.metadata:
                    raise TypeError("source_input_part_unsupported")
                existing = attachments.get(attachment.artifact_id)
                if existing is not None and existing != attachment:
                    raise ValueError("One artifact id has contradictory attachment metadata.")
                attachments[attachment.artifact_id] = attachment
    return attachments


def _attachment_source_digests(
    evidence: TerminalSessionEvidence,
    stimuli: Sequence[_Stimulus],
    attachments: Mapping[str, FileAttachment],
) -> tuple[dict[str, str], tuple[ScenarioCaptureDiagnosticV2, ...]]:
    if not attachments:
        return {}, ()
    source_sequences: dict[str, list[int]] = {artifact_id: [] for artifact_id in attachments}
    for stimulus in stimuli:
        if not isinstance(stimulus, _InputStimulus):
            continue
        stimulus_artifact_ids: set[str] = set()
        for message in stimulus.messages:
            for part in message.content:
                if type(part) is FilePart:
                    attachment = FileAttachment.model_validate(part.attachment)
                    stimulus_artifact_ids.add(attachment.artifact_id)
        for artifact_id in stimulus_artifact_ids:
            source_sequences[artifact_id].append(stimulus.materialized_sequence)

    attested: dict[str, list[tuple[int, str]]] = {}
    malformed_sequence: int | None = None
    for record in evidence.events:
        event = record.event
        if event.type != EventType.MODEL_STARTED:
            continue
        raw = event.payload.get(MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY)
        if raw is None:
            continue
        if type(raw) is not str or not event_payload_authority_is_runtime_generated(
            event,
            field_name=MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY,
            value=raw,
        ):
            malformed_sequence = record.sequence
            break
        if not raw.startswith("v1:"):
            malformed_sequence = record.sequence
            break
        try:
            decoded = json.loads(raw.removeprefix("v1:"))
        except (json.JSONDecodeError, RecursionError):
            malformed_sequence = record.sequence
            break
        try:
            canonical_marker = "v1:" + canonical_durable_json_bytes(
                decoded,
                "model file attachment attestations",
            ).decode("utf-8")
        except (TypeError, ValueError):
            malformed_sequence = record.sequence
            break
        if type(decoded) is not list or canonical_marker != raw:
            malformed_sequence = record.sequence
            break
        seen_ids: set[str] = set()
        for item in decoded:
            if type(item) is not dict or set(item) != {"artifact_id", "content_sha256"}:
                malformed_sequence = record.sequence
                break
            artifact_id = item.get("artifact_id")
            digest = item.get("content_sha256")
            if (
                type(artifact_id) is not str
                or not artifact_id
                or artifact_id in seen_ids
                or type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                malformed_sequence = record.sequence
                break
            seen_ids.add(artifact_id)
            if artifact_id in attachments:
                attested.setdefault(artifact_id, []).append((record.sequence, digest))
        if malformed_sequence is not None:
            break
    if malformed_sequence is not None:
        return {}, (
            _diagnostic(
                ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_INCONSISTENT,
                event_sequence=malformed_sequence,
            ),
        )

    expected: dict[str, str] = {}
    diagnostics: list[ScenarioCaptureDiagnosticV2] = []
    for artifact_id, attachment in attachments.items():
        requirement_id = _artifact_requirement_id(attachment)
        observations = attested.get(artifact_id, [])
        boundaries = source_sequences[artifact_id]
        if not boundaries or any(
            not any(sequence > boundary for sequence, _ in observations) for boundary in boundaries
        ):
            diagnostics.append(
                _diagnostic(
                    ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_UNAVAILABLE,
                    artifact_requirement_id=requirement_id,
                )
            )
            continue
        relevant = tuple(digest for sequence, digest in observations if sequence > min(boundaries))
        if len(set(relevant)) != 1:
            diagnostics.append(
                _diagnostic(
                    ScenarioCaptureDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT,
                    artifact_requirement_id=requirement_id,
                )
            )
            continue
        expected[artifact_id] = relevant[0]
    return expected, tuple(diagnostics)


def _scenario_message(
    app: CayuApp,
    message: Message,
    artifact_requirements: Mapping[str, ScenarioArtifactRequirementV2],
) -> ScenarioUserMessageV2:
    if message.role != MessageRole.USER:
        raise TypeError("source_input_role_unsupported")
    parts: list[ScenarioInputPartV2] = []
    for part in message.content:
        if type(part) is TextPart:
            try:
                redacted = app.redact_json(part.text)
            except Exception as exc:
                raise ValueError("source_payload_redacted") from exc
            if type(redacted) is not str or redacted != part.text:
                raise ValueError("source_payload_redacted")
            parts.append(ScenarioTextPartV2(text=redacted))
        elif type(part) is FilePart:
            attachment = FileAttachment.model_validate(part.attachment)
            if attachment.metadata:
                raise TypeError("source_input_part_unsupported")
            requirement = artifact_requirements.get(attachment.artifact_id)
            if requirement is None:
                raise ValueError("artifact_not_retained")
            parts.append(ScenarioFilePartV2(artifact_requirement_id=requirement.id))
        else:
            raise TypeError("source_input_part_unsupported")
    return ScenarioUserMessageV2.create(parts)


def _scenario_events(
    app: CayuApp,
    stimuli: Sequence[_Stimulus],
    artifact_requirements: Mapping[str, ScenarioArtifactRequirementV2],
) -> tuple[ScenarioEventV2, ...]:
    events: list[ScenarioEventV2] = []
    for sequence, stimulus in enumerate(stimuli):
        event_id = "initial" if sequence == 0 else f"event-{sequence:04d}"
        if isinstance(stimulus, _ApprovalStimulus):
            try:
                tool_name = app.redact_json(stimulus.tool_name)
            except Exception as exc:
                raise ValueError("source_payload_redacted") from exc
            if type(tool_name) is not str or tool_name != stimulus.tool_name:
                raise ValueError("source_payload_redacted")
            events.append(
                ScenarioApprovalCheckpointEventV2(
                    sequence=sequence,
                    id=event_id,
                    tool_name=tool_name,
                    occurrence=stimulus.occurrence,
                )
            )
            continue
        scenario_input = ScenarioInputV2.create(
            tuple(
                _scenario_message(app, message, artifact_requirements)
                for message in stimulus.messages
            )
        )
        if stimulus.kind == "initial":
            events.append(
                ScenarioInitialInputEventV2(
                    sequence=sequence,
                    id=event_id,
                    input=scenario_input,
                )
            )
        elif stimulus.kind == "queued":
            if stimulus.delivery_mode is None:
                raise ValueError("source_payload_inconsistent")
            events.append(
                ScenarioQueuedInputEventV2(
                    sequence=sequence,
                    id=event_id,
                    delivery_mode=stimulus.delivery_mode,
                    input=scenario_input,
                )
            )
        else:
            events.append(
                ScenarioResumedInputEventV2(
                    sequence=sequence,
                    id=event_id,
                    input=scenario_input,
                )
            )
    return tuple(events)


def _scenario_id(
    target_key: str,
    source: EvaluationSourceIdentityV1,
    events: Sequence[ScenarioEventV2],
    artifact_requirements: Sequence[ScenarioArtifactRequirementV2],
) -> str:
    material = canonical_durable_json_bytes(
        {
            "target_key": target_key,
            "source": source.model_dump(mode="json"),
            "events": [event.model_dump(mode="json") for event in events],
            "artifact_requirements": [
                requirement.model_dump(mode="json")
                for requirement in sorted(artifact_requirements, key=lambda item: item.id)
            ],
        },
        "captured scenario identity",
    )
    return f"scenario.{hashlib.sha256(material).hexdigest()}"


async def capture_eval_scenario_from_session(
    app: CayuApp,
    session_id: str,
    *,
    target_key: str,
    source_agent_name: str,
    source: EvaluationSourceIdentityV1,
    scenario_id: str | None = None,
    name: str = "Captured production scenario",
    description: str | None = None,
    limits: TerminalSessionEvidenceLimits | None = None,
) -> ScenarioCaptureResultV2:
    """Compile exact retained external stimuli without executing application work.

    The operation reads one bounded terminal snapshot and, for file inputs, the
    exact referenced artifact bytes. It never calls providers, tools,
    environments, hooks, recovery, or application mutation paths.
    """

    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp.")
    session_id = require_durable_text(session_id, "session_id")
    if not session_id.strip():
        raise ValueError("session_id must be non-empty.")
    target_key = _portable_id(target_key, "target_key")
    source_agent_name = require_durable_text(source_agent_name, "source_agent_name")
    if not source_agent_name.strip():
        raise ValueError("source_agent_name must be non-empty.")
    if type(source) is not EvaluationSourceIdentityV1:
        raise TypeError("source must be an exact EvaluationSourceIdentityV1.")
    source = EvaluationSourceIdentityV1.model_validate(source.model_dump(mode="python"))
    selected_limits = (
        None
        if limits is None
        else TerminalSessionEvidenceLimits.model_validate(limits.model_dump(mode="python"))
    )
    if not app.session_store.supports_terminal_session_evidence:
        return _unavailable(
            _diagnostic(ScenarioCaptureDiagnosticCode.TERMINAL_EVIDENCE_NOT_SUPPORTED)
        )
    try:
        evidence = await app.session_store.load_terminal_session_evidence(
            session_id,
            limits=selected_limits,
        )
    except TerminalSessionEvidenceError as exc:
        return _unavailable(_terminal_diagnostic(exc))
    except Exception:
        return _unavailable(_diagnostic(ScenarioCaptureDiagnosticCode.CAPTURED_EVIDENCE_INCOMPLETE))
    if evidence.session.agent_name != source_agent_name:
        return _unavailable(_diagnostic(ScenarioCaptureDiagnosticCode.SOURCE_AGENT_MISMATCH))

    stimuli, diagnostics = _capture_input_stimuli(evidence)
    if diagnostics:
        return _unavailable(*diagnostics)
    try:
        attachments = _attachments(stimuli)
    except TypeError:
        return _unavailable(
            _diagnostic(ScenarioCaptureDiagnosticCode.SOURCE_INPUT_PART_UNSUPPORTED)
        )
    except ValueError:
        return _unavailable(_diagnostic(ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_INCONSISTENT))
    try:
        expected_content_sha256, attestation_diagnostics = _attachment_source_digests(
            evidence,
            stimuli,
            attachments,
        )
    except (TypeError, ValueError):
        return _unavailable(_diagnostic(ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_INCONSISTENT))
    if attestation_diagnostics:
        return _unavailable(*attestation_diagnostics)
    artifact_requirements, artifact_diagnostics = await _resolve_artifact_requirements(
        app,
        evidence,
        attachments,
        expected_content_sha256=expected_content_sha256,
    )
    if artifact_diagnostics:
        return _unavailable(*artifact_diagnostics)

    try:
        events = _scenario_events(app, stimuli, artifact_requirements)
    except TypeError as exc:
        code = (
            ScenarioCaptureDiagnosticCode.SOURCE_INPUT_ROLE_UNSUPPORTED
            if str(exc) == "source_input_role_unsupported"
            else ScenarioCaptureDiagnosticCode.SOURCE_INPUT_PART_UNSUPPORTED
        )
        return _unavailable(_diagnostic(code))
    except ValueError as exc:
        code = (
            ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_REDACTED
            if str(exc) == "source_payload_redacted"
            else ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_INCONSISTENT
        )
        return _unavailable(_diagnostic(code))

    try:
        sorted_artifact_requirements = tuple(
            sorted(artifact_requirements.values(), key=lambda item: item.id)
        )
        scenario = EvalScenarioDocumentV2.create(
            id=(
                _scenario_id(
                    target_key,
                    source,
                    events,
                    sorted_artifact_requirements,
                )
                if scenario_id is None
                else scenario_id
            ),
            target_key=target_key,
            name=name,
            description=description,
            source=source,
            events=events,
            artifact_requirements=sorted_artifact_requirements,
        )
        try:
            projected = app.redact_json(scenario.model_dump(mode="json"))
        except Exception:
            return _unavailable(_diagnostic(ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_REDACTED))
        if type(projected) is not dict or projected != scenario.model_dump(mode="json"):
            return _unavailable(_diagnostic(ScenarioCaptureDiagnosticCode.SOURCE_PAYLOAD_REDACTED))
    except (TypeError, ValueError):
        return _unavailable(_diagnostic(ScenarioCaptureDiagnosticCode.SCENARIO_LIMIT_EXCEEDED))

    try:
        current = await app.session_store.load_terminal_session_evidence(
            session_id,
            limits=selected_limits,
        )
    except Exception:
        return _unavailable(_diagnostic(ScenarioCaptureDiagnosticCode.SOURCE_CHANGED))
    if current != evidence:
        return _unavailable(_diagnostic(ScenarioCaptureDiagnosticCode.SOURCE_CHANGED))
    return ScenarioCaptureResultV2(available=True, scenario=scenario)


__all__ = [
    "SCENARIO_CAPTURE_ARTIFACT_READ_CONCURRENCY",
    "SCENARIO_CAPTURE_MAX_DIAGNOSTICS",
    "ScenarioCaptureDiagnosticCode",
    "ScenarioCaptureDiagnosticV2",
    "ScenarioCaptureResultV2",
    "capture_eval_scenario_from_session",
]
