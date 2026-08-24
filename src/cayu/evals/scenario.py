from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    Field,
    StrictInt,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_object,
    copy_durable_json_value,
    durable_json_object_from_pairs,
    freeze_json_value,
    json_utf8_size_within_limit,
    parse_durable_json_integer_literal,
    reject_nonportable_json_constant,
    thaw_json_value,
)
from cayu.evals.corpus import (
    EvalCorpusDocument,
    EvaluationSourceIdentityV1,
    _bounded_durable_text,
    _content_revision,
    _model_content_revision,
    _model_python_input,
    _ordered_sequence_argument,
    _ordered_sequence_input,
    _portable_id,
    _PortableModel,
    _pretty_json_size_within_limit,
    _sha256_hex,
    _sha256_revision,
    _validated_model_document,
)

EVAL_SCENARIO_SCHEMA_VERSION = 2
EVAL_SCENARIO_MAX_BYTES = 8 << 20
EVAL_SCENARIO_MAX_EVENTS = 1_024
EVAL_SCENARIO_MAX_MESSAGES_PER_EVENT = 32
EVAL_SCENARIO_MAX_PARTS_PER_MESSAGE = 32
EVAL_SCENARIO_MAX_TEXT_CHARS = 65_536
EVAL_SCENARIO_MAX_TOTAL_TEXT_CHARS = 1 << 20
EVAL_SCENARIO_MAX_JSON_PART_BYTES = 256 << 10
EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS = 128
EVAL_SCENARIO_MAX_SECRET_REQUIREMENTS = 128
EVAL_SCENARIO_MAX_TOTAL_ARTIFACT_BYTES = 1 << 30


class _SchemaV2PortableModel(_PortableModel):
    schema_version: Literal[2] = EVAL_SCENARIO_SCHEMA_VERSION

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 2.")
        return value


class ScenarioTextPartV2(_PortableModel):
    """One literal caller-authored text part."""

    type: Literal["text"] = "text"
    text: StrictStr

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_SCENARIO_MAX_TEXT_CHARS,
            nonblank=True,
            clean=False,
        )


class ScenarioJsonPartV2(_PortableModel):
    """One structured caller input preserved as durable JSON."""

    type: Literal["json"] = "json"
    value: Any

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: object) -> object:
        copied = copy_durable_json_value(value, "scenario JSON input")
        if not json_utf8_size_within_limit(copied, EVAL_SCENARIO_MAX_JSON_PART_BYTES):
            raise ValueError(
                "Scenario JSON input exceeds "
                f"{EVAL_SCENARIO_MAX_JSON_PART_BYTES} canonical JSON bytes."
            )
        return copied

    @field_validator("value")
    @classmethod
    def freeze_value(cls, value: Any) -> Any:
        return freeze_json_value(value)

    @field_serializer("value")
    def serialize_value(self, value: Any) -> Any:
        return thaw_json_value(value)


class ScenarioFilePartV2(_PortableModel):
    """A file input by immutable requirement id; it never carries file bytes."""

    type: Literal["file"] = "file"
    artifact_requirement_id: StrictStr

    @field_validator("artifact_requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)


ScenarioInputPartV2: TypeAlias = Annotated[
    ScenarioTextPartV2 | ScenarioJsonPartV2 | ScenarioFilePartV2,
    Field(discriminator="type"),
]

_SCENARIO_INPUT_PART_TYPES = (
    ScenarioTextPartV2,
    ScenarioJsonPartV2,
    ScenarioFilePartV2,
)


def _validated_input_part(part: ScenarioInputPartV2) -> ScenarioInputPartV2:
    part_type = type(part)
    if part_type not in _SCENARIO_INPUT_PART_TYPES:
        raise TypeError("Scenario input parts must use an exact built-in part type.")
    return part_type.model_validate(_model_python_input(part))


class ScenarioUserMessageV2(_PortableModel):
    """One external user message with ordered text, JSON, or file-reference parts."""

    role: Literal["user"] = "user"
    content: tuple[ScenarioInputPartV2, ...] = Field(
        min_length=1,
        max_length=EVAL_SCENARIO_MAX_PARTS_PER_MESSAGE,
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content_is_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @classmethod
    def create(cls, parts: Sequence[ScenarioInputPartV2]) -> ScenarioUserMessageV2:
        ordered = _ordered_sequence_argument(parts, "parts")
        return cls(content=tuple(_validated_input_part(part) for part in ordered))


class ScenarioInputV2(_PortableModel):
    """One bounded batch of caller messages delivered at an explicit lifecycle point."""

    messages: tuple[ScenarioUserMessageV2, ...] = Field(
        min_length=1,
        max_length=EVAL_SCENARIO_MAX_MESSAGES_PER_EVENT,
    )

    @field_validator("messages", mode="before")
    @classmethod
    def validate_messages_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @classmethod
    def create(cls, messages: Sequence[ScenarioUserMessageV2]) -> ScenarioInputV2:
        ordered = _ordered_sequence_argument(messages, "messages")
        validated: list[ScenarioUserMessageV2] = []
        for message in ordered:
            if type(message) is not ScenarioUserMessageV2:
                raise TypeError("messages must contain exact ScenarioUserMessageV2 instances.")
            validated.append(ScenarioUserMessageV2.model_validate(_model_python_input(message)))
        return cls(messages=tuple(validated))


class _ScenarioEventBaseV2(_PortableModel):
    sequence: StrictInt = Field(ge=0, le=EVAL_SCENARIO_MAX_EVENTS - 1)
    id: StrictStr

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)


class ScenarioInitialInputEventV2(_ScenarioEventBaseV2):
    """Input used to create the fresh scenario session."""

    kind: Literal["initial"] = "initial"
    input: ScenarioInputV2


class ScenarioQueuedInputEventV2(_ScenarioEventBaseV2):
    """Input queued while the scenario session is active."""

    kind: Literal["queued"] = "queued"
    delivery_mode: Literal["next_turn", "on_idle"]
    input: ScenarioInputV2


class ScenarioResumedInputEventV2(_ScenarioEventBaseV2):
    """Fresh caller input supplied when a scenario session is paused.

    ``manual_recovery`` is the scenario-v2 wire name for an ordinary explicit
    ``CayuApp.resume(...)`` interaction. It does not assert or synthesize the
    outcome of a runtime tool call whose external effect is unknown.
    """

    kind: Literal["resumed"] = "resumed"
    resume_kind: Literal["user_input", "manual_recovery"] = "user_input"
    input: ScenarioInputV2


class ScenarioApprovalCheckpointEventV2(_ScenarioEventBaseV2):
    """Require a new approval decision for a matching current tool call.

    The checkpoint identifies an expected pause but contains no approve/deny
    choice, approval id, actor, or reusable authorization from the source run.
    """

    kind: Literal["approval_checkpoint"] = "approval_checkpoint"
    tool_name: StrictStr
    occurrence: StrictInt = Field(default=1, ge=1, le=EVAL_SCENARIO_MAX_EVENTS)
    resolution: Literal["fresh_decision"] = "fresh_decision"

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )


ScenarioEventV2: TypeAlias = Annotated[
    ScenarioInitialInputEventV2
    | ScenarioQueuedInputEventV2
    | ScenarioResumedInputEventV2
    | ScenarioApprovalCheckpointEventV2,
    Field(discriminator="kind"),
]

_SCENARIO_EVENT_TYPES = (
    ScenarioInitialInputEventV2,
    ScenarioQueuedInputEventV2,
    ScenarioResumedInputEventV2,
    ScenarioApprovalCheckpointEventV2,
)


def _validated_event(event: ScenarioEventV2) -> ScenarioEventV2:
    event_type = type(event)
    if event_type not in _SCENARIO_EVENT_TYPES:
        raise TypeError("Scenario events must use an exact built-in event type.")
    return event_type.model_validate(_model_python_input(event))


class ScenarioArtifactRequirementV2(_PortableModel):
    """An immutable file requirement resolved and authorized only at launch."""

    id: StrictStr
    source: Literal["fixture_digest", "artifact_reference"]
    reference: StrictStr | None = None
    content_sha256: StrictStr
    filename: StrictStr
    content_type: StrictStr
    size_bytes: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("content_sha256")
    @classmethod
    def validate_content_sha256(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("reference")
    @classmethod
    def validate_reference(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=512,
            nonblank=True,
            clean=True,
        )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=512,
            nonblank=True,
            clean=False,
        )

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @model_validator(mode="after")
    def validate_source_reference(self) -> ScenarioArtifactRequirementV2:
        if (self.source == "artifact_reference") != (self.reference is not None):
            raise ValueError(
                "Artifact requirements need `reference` exactly when source is artifact_reference."
            )
        return self


class ScenarioSecretRequirementV2(_PortableModel):
    """A named server-side dependency; no secret value or vault handle is portable."""

    id: StrictStr
    usage: Literal["provider", "tool", "environment", "artifact", "other"]
    purpose: StrictStr

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("purpose")
    @classmethod
    def validate_purpose(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=512,
            nonblank=True,
            clean=True,
        )


class EvalScenarioInspectionV2(_SchemaV2PortableModel):
    revision: StrictStr
    id: StrictStr
    target_key: StrictStr
    event_count: StrictInt = Field(ge=1, le=EVAL_SCENARIO_MAX_EVENTS)
    input_event_count: StrictInt = Field(ge=1, le=EVAL_SCENARIO_MAX_EVENTS)
    approval_checkpoint_count: StrictInt = Field(ge=0, le=EVAL_SCENARIO_MAX_EVENTS)
    message_count: StrictInt = Field(
        ge=1,
        le=EVAL_SCENARIO_MAX_EVENTS * EVAL_SCENARIO_MAX_MESSAGES_PER_EVENT,
    )
    part_count: StrictInt = Field(
        ge=1,
        le=(
            EVAL_SCENARIO_MAX_EVENTS
            * EVAL_SCENARIO_MAX_MESSAGES_PER_EVENT
            * EVAL_SCENARIO_MAX_PARTS_PER_MESSAGE
        ),
    )
    artifact_requirement_count: StrictInt = Field(
        ge=0,
        le=EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS,
    )
    secret_requirement_count: StrictInt = Field(
        ge=0,
        le=EVAL_SCENARIO_MAX_SECRET_REQUIREMENTS,
    )

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("id", "target_key")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @model_validator(mode="after")
    def validate_counts(self) -> EvalScenarioInspectionV2:
        if self.input_event_count + self.approval_checkpoint_count != self.event_count:
            raise ValueError("Eval scenario inspection event counts are inconsistent.")
        if self.message_count < self.input_event_count:
            raise ValueError("Eval scenario inspection message count is impossible.")
        if self.part_count < self.message_count:
            raise ValueError("Eval scenario inspection part count is impossible.")
        return self


class EvalScenarioDocumentV2(_SchemaV2PortableModel):
    """One canonical external-stimulus sequence with no executable authority."""

    revision: StrictStr
    id: StrictStr
    target_key: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    source: EvaluationSourceIdentityV1 | None = None
    events: tuple[ScenarioEventV2, ...] = Field(
        min_length=1,
        max_length=EVAL_SCENARIO_MAX_EVENTS,
    )
    artifact_requirements: tuple[ScenarioArtifactRequirementV2, ...] = Field(
        default_factory=tuple,
        max_length=EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS,
    )
    secret_requirements: tuple[ScenarioSecretRequirementV2, ...] = Field(
        default_factory=tuple,
        max_length=EVAL_SCENARIO_MAX_SECRET_REQUIREMENTS,
    )

    @field_validator("events", "artifact_requirements", "secret_requirements", mode="before")
    @classmethod
    def validate_collections_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("id", "target_key")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=2_048,
            nonblank=True,
            clean=True,
        )

    @model_validator(mode="after")
    def validate_contract(self) -> EvalScenarioDocumentV2:
        if tuple(event.sequence for event in self.events) != tuple(range(len(self.events))):
            raise ValueError("Scenario event sequences must be contiguous and start at zero.")
        event_ids = tuple(event.id for event in self.events)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Scenario event IDs must be unique.")
        if type(self.events[0]) is not ScenarioInitialInputEventV2 or any(
            type(event) is ScenarioInitialInputEventV2 for event in self.events[1:]
        ):
            raise ValueError("A scenario requires exactly one initial event at sequence zero.")

        artifact_ids = tuple(item.id for item in self.artifact_requirements)
        secret_ids = tuple(item.id for item in self.secret_requirements)
        if artifact_ids != tuple(sorted(set(artifact_ids))):
            raise ValueError("Artifact requirements must be unique and sorted by id.")
        if secret_ids != tuple(sorted(set(secret_ids))):
            raise ValueError("Secret requirements must be unique and sorted by id.")
        referenced_artifacts = {
            part.artifact_requirement_id
            for event in self.events
            if isinstance(
                event,
                ScenarioInitialInputEventV2
                | ScenarioQueuedInputEventV2
                | ScenarioResumedInputEventV2,
            )
            for message in event.input.messages
            for part in message.content
            if isinstance(part, ScenarioFilePartV2)
        }
        missing_artifacts = sorted(referenced_artifacts - set(artifact_ids))
        if missing_artifacts:
            raise ValueError(
                "Scenario file parts reference undeclared artifact requirements: "
                + ", ".join(missing_artifacts)
                + "."
            )
        total_artifact_bytes = sum(item.size_bytes for item in self.artifact_requirements)
        if total_artifact_bytes > EVAL_SCENARIO_MAX_TOTAL_ARTIFACT_BYTES:
            raise ValueError(
                "Scenario artifact requirements exceed "
                f"{EVAL_SCENARIO_MAX_TOTAL_ARTIFACT_BYTES} total bytes."
            )

        approval_occurrences: dict[str, list[int]] = {}
        for event in self.events:
            if isinstance(event, ScenarioApprovalCheckpointEventV2):
                approval_occurrences.setdefault(event.tool_name, []).append(event.occurrence)
        for tool_name, occurrences in approval_occurrences.items():
            if occurrences != sorted(set(occurrences)):
                raise ValueError(
                    f"Approval checkpoints for tool {tool_name!r} require unique ascending "
                    "occurrences."
                )

        total_text_chars = sum(
            len(part.text)
            for event in self.events
            if isinstance(
                event,
                ScenarioInitialInputEventV2
                | ScenarioQueuedInputEventV2
                | ScenarioResumedInputEventV2,
            )
            for message in event.input.messages
            for part in message.content
            if isinstance(part, ScenarioTextPartV2)
        )
        if total_text_chars > EVAL_SCENARIO_MAX_TOTAL_TEXT_CHARS:
            raise ValueError(
                f"Scenario text exceeds {EVAL_SCENARIO_MAX_TOTAL_TEXT_CHARS} total characters."
            )
        if not json_utf8_size_within_limit(self, EVAL_SCENARIO_MAX_BYTES):
            raise ValueError(
                f"Eval scenario exceeds {EVAL_SCENARIO_MAX_BYTES} canonical JSON bytes."
            )
        if not _pretty_json_size_within_limit(self, EVAL_SCENARIO_MAX_BYTES):
            raise ValueError(
                f"Eval scenario exceeds {EVAL_SCENARIO_MAX_BYTES} serialized JSON bytes."
            )
        if self.revision != _model_content_revision(self, "eval scenario"):
            raise ValueError("Eval scenario revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        id: str,
        target_key: str,
        name: str,
        events: Sequence[ScenarioEventV2],
        description: str | None = None,
        source: EvaluationSourceIdentityV1 | None = None,
        artifact_requirements: Sequence[ScenarioArtifactRequirementV2] = (),
        secret_requirements: Sequence[ScenarioSecretRequirementV2] = (),
    ) -> EvalScenarioDocumentV2:
        ordered_events = _ordered_sequence_argument(events, "events")
        validated_events = tuple(_validated_event(event) for event in ordered_events)
        ordered_artifacts = _ordered_sequence_argument(
            artifact_requirements,
            "artifact_requirements",
        )
        validated_artifacts: list[ScenarioArtifactRequirementV2] = []
        for requirement in ordered_artifacts:
            if type(requirement) is not ScenarioArtifactRequirementV2:
                raise TypeError(
                    "artifact_requirements must contain exact "
                    "ScenarioArtifactRequirementV2 instances."
                )
            validated_artifacts.append(
                ScenarioArtifactRequirementV2.model_validate(_model_python_input(requirement))
            )
        ordered_secrets = _ordered_sequence_argument(secret_requirements, "secret_requirements")
        validated_secrets: list[ScenarioSecretRequirementV2] = []
        for requirement in ordered_secrets:
            if type(requirement) is not ScenarioSecretRequirementV2:
                raise TypeError(
                    "secret_requirements must contain exact ScenarioSecretRequirementV2 instances."
                )
            validated_secrets.append(
                ScenarioSecretRequirementV2.model_validate(_model_python_input(requirement))
            )
        if source is None:
            validated_source = None
        elif type(source) is EvaluationSourceIdentityV1:
            validated_source = EvaluationSourceIdentityV1.model_validate(
                _model_python_input(source)
            )
        else:
            raise TypeError("source must be an exact EvaluationSourceIdentityV1 or None.")
        document: dict[str, Any] = {
            "schema_version": EVAL_SCENARIO_SCHEMA_VERSION,
            "id": _portable_id(id, "id"),
            "target_key": _portable_id(target_key, "target_key"),
            "name": name,
            "description": description,
            "source": (
                None if validated_source is None else validated_source.model_dump(mode="json")
            ),
            "events": [event.model_dump(mode="json") for event in validated_events],
            "artifact_requirements": [
                item.model_dump(mode="json")
                for item in sorted(validated_artifacts, key=lambda item: item.id)
            ],
            "secret_requirements": [
                item.model_dump(mode="json")
                for item in sorted(validated_secrets, key=lambda item: item.id)
            ],
        }
        if not json_utf8_size_within_limit(
            {"revision": "sha256:" + "0" * 64, **document},
            EVAL_SCENARIO_MAX_BYTES,
        ):
            raise ValueError(
                f"Eval scenario exceeds {EVAL_SCENARIO_MAX_BYTES} canonical JSON bytes."
            )
        return cls(revision=_content_revision(document, "eval scenario"), **document)


def inspect_eval_scenario(scenario: EvalScenarioDocumentV2) -> EvalScenarioInspectionV2:
    validated, _ = _validated_model_document(
        scenario,
        model_type=EvalScenarioDocumentV2,
        field_name="eval scenario",
    )
    input_events = tuple(
        event
        for event in validated.events
        if isinstance(
            event,
            ScenarioInitialInputEventV2 | ScenarioQueuedInputEventV2 | ScenarioResumedInputEventV2,
        )
    )
    return EvalScenarioInspectionV2(
        revision=validated.revision,
        id=validated.id,
        target_key=validated.target_key,
        event_count=len(validated.events),
        input_event_count=len(input_events),
        approval_checkpoint_count=sum(
            isinstance(event, ScenarioApprovalCheckpointEventV2) for event in validated.events
        ),
        message_count=sum(len(event.input.messages) for event in input_events),
        part_count=sum(
            len(message.content) for event in input_events for message in event.input.messages
        ),
        artifact_requirement_count=len(validated.artifact_requirements),
        secret_requirement_count=len(validated.secret_requirements),
    )


@dataclass(frozen=True, slots=True)
class CompiledEvalScenarioV2:
    """Validated authority-free template consumed by later launch preflight."""

    document: EvalScenarioDocumentV2
    initial: ScenarioInitialInputEventV2
    steps: tuple[
        ScenarioQueuedInputEventV2
        | ScenarioResumedInputEventV2
        | ScenarioApprovalCheckpointEventV2,
        ...,
    ]

    @property
    def revision(self) -> str:
        return self.document.revision

    @property
    def target_key(self) -> str:
        return self.document.target_key

    def artifact_requirement(self, requirement_id: str) -> ScenarioArtifactRequirementV2:
        requirement_id = _portable_id(requirement_id, "requirement_id")
        for requirement in self.document.artifact_requirements:
            if requirement.id == requirement_id:
                return requirement
        raise KeyError(f"Scenario artifact requirement not found: {requirement_id}")

    def secret_requirement(self, requirement_id: str) -> ScenarioSecretRequirementV2:
        requirement_id = _portable_id(requirement_id, "requirement_id")
        for requirement in self.document.secret_requirements:
            if requirement.id == requirement_id:
                return requirement
        raise KeyError(f"Scenario secret requirement not found: {requirement_id}")


def compile_eval_scenario(scenario: EvalScenarioDocumentV2) -> CompiledEvalScenarioV2:
    """Compile portable stimuli without resolving any execution authority."""

    validated, _ = _validated_model_document(
        scenario,
        model_type=EvalScenarioDocumentV2,
        field_name="eval scenario",
    )
    initial = validated.events[0]
    if type(initial) is not ScenarioInitialInputEventV2:  # validated invariant
        raise AssertionError("Validated eval scenario has no initial event.")
    return CompiledEvalScenarioV2(
        document=validated,
        initial=initial,
        steps=tuple(validated.events[1:]),
    )


def scenario_from_corpus_case(
    corpus: EvalCorpusDocument,
    case_id: str,
    *,
    scenario_id: str | None = None,
) -> EvalScenarioDocumentV2:
    """Lift one runnable corpus-v1 case into the scenario-v2 stimulus contract."""

    validated, _ = _validated_model_document(
        corpus,
        model_type=EvalCorpusDocument,
        field_name="eval corpus",
    )
    case_id = _portable_id(case_id, "case_id")
    case = next((item for item in validated.cases if item.id == case_id), None)
    if case is None:
        raise ValueError(f"Eval corpus does not contain case {case_id!r}.")
    if case.input is None:
        raise ValueError("Captured-only corpus cases need authored scenario stimuli.")
    messages = tuple(
        ScenarioUserMessageV2.create((ScenarioTextPartV2(text=message.text),))
        for message in case.input.messages
    )
    return EvalScenarioDocumentV2.create(
        id=case.id if scenario_id is None else scenario_id,
        target_key=validated.target_key,
        name=case.name,
        description=case.description,
        source=case.source,
        events=(
            ScenarioInitialInputEventV2(
                sequence=0,
                id="initial",
                input=ScenarioInputV2.create(messages),
            ),
        ),
    )


def eval_scenario_to_json(scenario: EvalScenarioDocumentV2) -> str:
    """Return deterministic, human-readable scenario-v2 JSON."""

    _, document = _validated_model_document(
        scenario,
        model_type=EvalScenarioDocumentV2,
        field_name="eval scenario",
    )
    encoder = json.JSONEncoder(ensure_ascii=False, indent=2, sort_keys=True)
    chunks: list[str] = []
    total_bytes = 1
    for chunk in encoder.iterencode(document):
        total_bytes += len(chunk.encode("utf-8"))
        if total_bytes > EVAL_SCENARIO_MAX_BYTES:
            raise ValueError(f"Eval scenario JSON exceeds {EVAL_SCENARIO_MAX_BYTES} bytes.")
        chunks.append(chunk)
    return "".join(chunks) + "\n"


def eval_scenario_from_json(source: str) -> EvalScenarioDocumentV2:
    """Load one bounded scenario-v2 JSON document from text."""

    if type(source) is not str:
        raise TypeError("eval_scenario_from_json requires text.")
    if len(source) > EVAL_SCENARIO_MAX_BYTES:
        raise ValueError(f"Eval scenario JSON exceeds {EVAL_SCENARIO_MAX_BYTES} bytes.")
    try:
        raw = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Eval scenario JSON must contain valid Unicode scalar text.") from exc
    if len(raw) > EVAL_SCENARIO_MAX_BYTES:
        raise ValueError(f"Eval scenario JSON exceeds {EVAL_SCENARIO_MAX_BYTES} bytes.")
    try:
        decoded = json.loads(
            source,
            parse_int=partial(
                parse_durable_json_integer_literal,
                field_name="eval scenario JSON",
            ),
            parse_constant=partial(
                reject_nonportable_json_constant,
                field_name="eval scenario JSON",
            ),
            object_pairs_hook=partial(
                durable_json_object_from_pairs,
                field_name="eval scenario JSON",
            ),
        )
    except RecursionError as exc:
        raise ValueError("Eval scenario JSON nesting exceeds the supported depth.") from exc
    document = copy_durable_json_object(decoded, "eval scenario JSON")
    raw_version = document.get("schema_version")
    if type(raw_version) is not int or raw_version != EVAL_SCENARIO_SCHEMA_VERSION:
        raise ValueError(
            f"Eval scenario has unsupported schema_version {raw_version!r}; this Cayu "
            f"version supports only {EVAL_SCENARIO_SCHEMA_VERSION}."
        )
    return EvalScenarioDocumentV2.model_validate(document)


def load_eval_scenario(path: str | Path) -> EvalScenarioDocumentV2:
    """Read at most the scenario hard limit before decoding or validating JSON."""

    resolved = Path(path)
    with resolved.open("rb") as handle:
        raw = handle.read(EVAL_SCENARIO_MAX_BYTES + 1)
    if len(raw) > EVAL_SCENARIO_MAX_BYTES:
        raise ValueError(f"Eval scenario JSON exceeds {EVAL_SCENARIO_MAX_BYTES} bytes.")
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Eval scenario JSON must be UTF-8.") from exc
    return eval_scenario_from_json(source)


__all__ = [
    "EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS",
    "EVAL_SCENARIO_MAX_BYTES",
    "EVAL_SCENARIO_MAX_EVENTS",
    "EVAL_SCENARIO_MAX_JSON_PART_BYTES",
    "EVAL_SCENARIO_MAX_MESSAGES_PER_EVENT",
    "EVAL_SCENARIO_MAX_PARTS_PER_MESSAGE",
    "EVAL_SCENARIO_MAX_SECRET_REQUIREMENTS",
    "EVAL_SCENARIO_MAX_TEXT_CHARS",
    "EVAL_SCENARIO_MAX_TOTAL_ARTIFACT_BYTES",
    "EVAL_SCENARIO_MAX_TOTAL_TEXT_CHARS",
    "EVAL_SCENARIO_SCHEMA_VERSION",
    "CompiledEvalScenarioV2",
    "EvalScenarioDocumentV2",
    "EvalScenarioInspectionV2",
    "ScenarioApprovalCheckpointEventV2",
    "ScenarioArtifactRequirementV2",
    "ScenarioEventV2",
    "ScenarioFilePartV2",
    "ScenarioInitialInputEventV2",
    "ScenarioInputPartV2",
    "ScenarioInputV2",
    "ScenarioJsonPartV2",
    "ScenarioQueuedInputEventV2",
    "ScenarioResumedInputEventV2",
    "ScenarioSecretRequirementV2",
    "ScenarioTextPartV2",
    "ScenarioUserMessageV2",
    "compile_eval_scenario",
    "eval_scenario_from_json",
    "eval_scenario_to_json",
    "inspect_eval_scenario",
    "load_eval_scenario",
    "scenario_from_corpus_case",
]
