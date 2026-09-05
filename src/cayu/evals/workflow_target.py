"""Typed application-owned workflow execution contracts for Cayu Evals."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, TypeAlias
from uuid import UUID, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    json_utf8_size_within_limit,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, detach_message
from cayu.core.workflows import WorkflowSpec, copy_workflow_spec
from cayu.evals.capture_policy import SessionTrajectoryBounds
from cayu.runtime.app import CayuApp
from cayu.workflows import WorkflowBase

WORKFLOW_EVAL_MAX_FINAL_OUTPUT_CHARS = 65_536
WORKFLOW_EVAL_MAX_STRUCTURED_OUTPUT_BYTES = 256 << 10
WORKFLOW_EVAL_MAX_APPLICATION_CONTEXT_BYTES = 64 << 10
WORKFLOW_EVAL_MAX_INPUT_MESSAGES = 32
WORKFLOW_EVAL_MAX_INPUT_BYTES = 8 << 20
WORKFLOW_EVAL_DEFAULT_CLOSE_TIMEOUT_SECONDS = 30.0
WORKFLOW_EVAL_MAX_CLOSE_TIMEOUT_SECONDS = 300.0

_WORKFLOW_TARGET_REVISION_DOMAIN = b"cayu-workflow-eval-target-v1\0"
_WORKFLOW_SPEC_REVISION_DOMAIN = b"cayu-workflow-eval-spec-v1\0"
_WORKFLOW_APPLICATION_CONTEXT_REVISION_DOMAIN = b"cayu-workflow-eval-context-v1\0"
_WORKFLOW_INPUT_REVISION_DOMAIN = b"cayu-workflow-eval-input-v1\0"
_WORKFLOW_OUTPUT_REVISION_DOMAIN = b"cayu-workflow-eval-output-v1\0"
_WORKFLOW_TRIAL_SESSION_NAMESPACE = UUID("16ffd157-53db-42a4-8cda-73b0e224849d")


def _sha256_revision(value: str, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must be a sha256 revision.")
    return value


def workflow_spec_revision(spec: WorkflowSpec) -> str:
    """Return the canonical public revision of one workflow specification."""

    copied = copy_workflow_spec(spec)
    digest = hashlib.sha256(
        _WORKFLOW_SPEC_REVISION_DOMAIN
        + canonical_durable_json_bytes(copied.model_dump(mode="json"), "workflow spec")
    ).hexdigest()
    return f"sha256:{digest}"


class WorkflowEvalInstanceScope(StrEnum):
    """How a target constructs application/workflow state for concrete trials."""

    SHARED = "shared"
    PER_TRIAL = "per_trial"


class WorkflowEvalTargetIdentityV1(BaseModel):
    """Portable behavior identity for a trusted workflow-root eval target."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    schema_version: Literal[1] = 1
    revision: StrictStr
    workflow_name: StrictStr
    workflow_spec_revision: StrictStr
    implementation_revision: StrictStr
    result_projector_revision: StrictStr
    execution_scope_revision: StrictStr
    application_context_revision: StrictStr
    evidence_policy_revision: StrictStr
    capture_bounds: SessionTrajectoryBounds | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    instance_scope: WorkflowEvalInstanceScope
    close_timeout_seconds: float = Field(
        gt=0,
        le=WORKFLOW_EVAL_MAX_CLOSE_TIMEOUT_SECONDS,
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("workflow_name")
    @classmethod
    def validate_workflow_name(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator(
        "revision",
        "workflow_spec_revision",
        "implementation_revision",
        "result_projector_revision",
        "execution_scope_revision",
        "evidence_policy_revision",
        "application_context_revision",
    )
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("close_timeout_seconds", mode="before")
    @classmethod
    def validate_close_timeout(cls, value: object) -> object:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= WORKFLOW_EVAL_MAX_CLOSE_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "close_timeout_seconds must be a finite positive number no greater than "
                f"{WORKFLOW_EVAL_MAX_CLOSE_TIMEOUT_SECONDS}."
            )
        return value

    @model_validator(mode="after")
    def validate_revision(self) -> WorkflowEvalTargetIdentityV1:
        material = self.model_dump(mode="json", exclude={"revision"})
        expected = (
            "sha256:"
            + hashlib.sha256(
                _WORKFLOW_TARGET_REVISION_DOMAIN
                + canonical_durable_json_bytes(material, "workflow eval target identity")
            ).hexdigest()
        )
        if self.revision != expected:
            raise ValueError("Workflow eval target revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        workflow_spec: WorkflowSpec,
        implementation_revision: str,
        result_projector_revision: str,
        execution_scope_revision: str,
        application_context: dict[str, Any],
        evidence_policy_revision: str,
        instance_scope: WorkflowEvalInstanceScope | str,
        close_timeout_seconds: float,
        capture_bounds: SessionTrajectoryBounds | None = None,
    ) -> WorkflowEvalTargetIdentityV1:
        copied = copy_workflow_spec(workflow_spec)
        material = {
            "schema_version": 1,
            "workflow_name": copied.name,
            "workflow_spec_revision": workflow_spec_revision(copied),
            "implementation_revision": _sha256_revision(
                implementation_revision, "implementation_revision"
            ),
            "result_projector_revision": _sha256_revision(
                result_projector_revision, "result_projector_revision"
            ),
            "execution_scope_revision": _sha256_revision(
                execution_scope_revision, "execution_scope_revision"
            ),
            "application_context_revision": "sha256:"
            + hashlib.sha256(
                _WORKFLOW_APPLICATION_CONTEXT_REVISION_DOMAIN
                + canonical_durable_json_bytes(
                    copy_durable_json_object(application_context, "application_context"),
                    "application_context",
                )
            ).hexdigest(),
            "evidence_policy_revision": _sha256_revision(
                evidence_policy_revision, "evidence_policy_revision"
            ),
            "instance_scope": WorkflowEvalInstanceScope(instance_scope).value,
            "close_timeout_seconds": close_timeout_seconds,
        }
        if capture_bounds is not None and capture_bounds != SessionTrajectoryBounds():
            material["capture_bounds"] = capture_bounds.model_dump(mode="json")
        revision = (
            "sha256:"
            + hashlib.sha256(
                _WORKFLOW_TARGET_REVISION_DOMAIN
                + canonical_durable_json_bytes(material, "workflow eval target identity")
            ).hexdigest()
        )
        return cls.model_validate({"revision": revision, **material})


class WorkflowEvalInvocation(BaseModel):
    """Bounded candidate input and runtime identities supplied to one trusted factory."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    run_id: StrictStr = Field(max_length=256)
    suite_id: StrictStr = Field(max_length=128)
    case_id: StrictStr = Field(max_length=128)
    trial_number: StrictInt = Field(ge=1, le=100)
    workflow_run_id: StrictStr = Field(max_length=128)
    idempotency_key: StrictStr = Field(max_length=256)
    messages: tuple[Message, ...]
    application_context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "suite_id", "case_id", "workflow_run_id", "idempotency_key")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("messages", mode="before")
    @classmethod
    def copy_messages(cls, value: object) -> tuple[Message, ...]:
        if not isinstance(value, list | tuple):
            raise TypeError("messages must be an ordered list or tuple.")
        if len(value) > WORKFLOW_EVAL_MAX_INPUT_MESSAGES:
            raise ValueError(
                f"messages cannot contain more than {WORKFLOW_EVAL_MAX_INPUT_MESSAGES} values."
            )
        copied: list[Message] = []
        for message in value:
            if type(message) is not Message:
                raise TypeError("messages must contain exact Message values.")
            copied.append(detach_message(message))
        if not json_utf8_size_within_limit(
            [message.model_dump(mode="json") for message in copied],
            WORKFLOW_EVAL_MAX_INPUT_BYTES,
        ):
            raise ValueError("messages exceed the workflow eval input byte limit.")
        return tuple(copied)

    @field_validator("application_context", mode="before")
    @classmethod
    def copy_application_context(cls, value: object) -> dict[str, Any]:
        copied = copy_durable_json_object(value, "application_context")
        if not json_utf8_size_within_limit(
            copied,
            WORKFLOW_EVAL_MAX_APPLICATION_CONTEXT_BYTES,
        ):
            raise ValueError("application_context exceeds its canonical JSON byte limit.")
        return copied


class WorkflowEvalResult(BaseModel):
    """Typed bounded candidate output returned by an application-owned projector."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    final_output: StrictStr
    structured_output: dict[str, Any] | None = None

    @field_validator("final_output")
    @classmethod
    def validate_final_output(cls, value: str, info) -> str:
        value = require_durable_text(value, info.field_name)
        if len(value) > WORKFLOW_EVAL_MAX_FINAL_OUTPUT_CHARS:
            raise ValueError("final_output exceeds the workflow eval character limit.")
        return value

    @field_validator("structured_output", mode="before")
    @classmethod
    def copy_structured_output(cls, value: object) -> dict[str, Any] | None:
        if value is None:
            return None
        copied = copy_durable_json_object(value, "structured_output")
        if not json_utf8_size_within_limit(
            copied,
            WORKFLOW_EVAL_MAX_STRUCTURED_OUTPUT_BYTES,
        ):
            raise ValueError("structured_output exceeds its canonical JSON byte limit.")
        return copied


class WorkflowEvalTerminalEvidence(BaseModel):
    """Exact current-attempt completion evidence visible to the result projector."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    workflow_run_id: StrictStr
    workflow_name: StrictStr
    attempt_id: StrictStr
    completion_event: Event

    @field_validator("workflow_run_id", "workflow_name", "attempt_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("completion_event", mode="before")
    @classmethod
    def copy_completion_event(cls, value: object) -> object:
        if type(value) is Event:
            return value.model_dump(mode="python", round_trip=True, warnings="none")
        if isinstance(value, BaseModel):
            raise TypeError("completion_event must be an exact Event or JSON object.")
        return value

    @model_validator(mode="after")
    def validate_completion_boundary(self) -> WorkflowEvalTerminalEvidence:
        completion = self.completion_event
        if (
            completion.type != EventType.WORKFLOW_COMPLETED
            or completion.session_id != self.workflow_run_id
            or completion.workflow_name != self.workflow_name
            or completion.payload.get("attempt_id") != self.attempt_id
        ):
            raise ValueError(
                "completion_event must match the exact workflow run, name, and attempt."
            )
        return self


class WorkflowEvalOutputEvidenceV1(BaseModel):
    """Replay-safe binding from projected output to one exact workflow completion."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    schema_version: Literal[1] = 1
    target_revision: StrictStr
    projector_revision: StrictStr
    workflow_name: StrictStr
    attempt_id: StrictStr
    completion_event_id: StrictStr
    input_message_count: StrictInt = Field(ge=0, le=WORKFLOW_EVAL_MAX_INPUT_MESSAGES)
    input_messages_sha256: StrictStr
    final_output_sha256: StrictStr
    structured_output: dict[str, Any] | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("target_revision", "projector_revision")
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("input_messages_sha256", "final_output_sha256")
    @classmethod
    def validate_sha256(cls, value: str, info) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("workflow_name", "attempt_id", "completion_event_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("structured_output", mode="before")
    @classmethod
    def copy_structured_output(cls, value: object) -> dict[str, Any] | None:
        if value is None:
            return None
        copied = copy_durable_json_object(value, "structured_output")
        if not json_utf8_size_within_limit(
            copied,
            WORKFLOW_EVAL_MAX_STRUCTURED_OUTPUT_BYTES,
        ):
            raise ValueError("structured_output exceeds its canonical JSON byte limit.")
        return copied


@dataclass(frozen=True, slots=True)
class WorkflowEvalExecution:
    """The exact application/workflow pair and optional quiescence callback for one trial."""

    app: CayuApp
    workflow: WorkflowBase
    close: Callable[[], Awaitable[None]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.app, CayuApp):
            raise TypeError("WorkflowEvalExecution app must be a CayuApp.")
        if not isinstance(self.workflow, WorkflowBase):
            raise TypeError("WorkflowEvalExecution workflow must be a WorkflowBase.")
        if self.workflow.app is not self.app:
            raise ValueError("WorkflowEvalExecution workflow must belong to its declared app.")
        if self.close is not None and not callable(self.close):
            raise TypeError("WorkflowEvalExecution close must be callable or None.")


WorkflowEvalFactory: TypeAlias = Callable[
    [WorkflowEvalInvocation],
    WorkflowEvalExecution | Awaitable[WorkflowEvalExecution],
]
WorkflowEvalResultProjector: TypeAlias = Callable[
    [WorkflowEvalTerminalEvidence],
    WorkflowEvalResult | Awaitable[WorkflowEvalResult],
]


class WorkflowEvalFailureCode(StrEnum):
    """Internal stable classifications mapped onto public trial diagnostics."""

    TARGET_FAILED = "workflow_target_failed"
    EXECUTION_FAILED = "workflow_execution_failed"
    COMPLETION_MISSING = "workflow_completion_missing"
    COMPLETION_CONFLICT = "workflow_completion_conflict"
    ATTEMPT_SUPERSEDED = "workflow_attempt_superseded"
    PROJECTOR_FAILED = "workflow_projector_failed"
    OUTPUT_INVALID = "workflow_output_invalid"
    QUIESCENCE_FAILED = "workflow_quiescence_failed"


class WorkflowEvalFailure(RuntimeError):
    """Fail-closed workflow driver error with a public-safe diagnostic."""

    def __init__(self, code: WorkflowEvalFailureCode, message: str) -> None:
        self.code = WorkflowEvalFailureCode(code)
        super().__init__(require_durable_clean_nonblank(message, "message"))


def workflow_eval_trial_session_id(
    *,
    target_revision: str,
    run_id: str,
    suite_id: str,
    case_id: str,
    trial_number: int,
) -> str:
    """Derive the recovery-stable root identity for one concrete workflow trial."""

    target_revision = _sha256_revision(target_revision, "target_revision")
    for value, field_name in (
        (run_id, "run_id"),
        (suite_id, "suite_id"),
        (case_id, "case_id"),
    ):
        require_durable_clean_nonblank(value, field_name)
    if type(trial_number) is not int or trial_number < 1:
        raise ValueError("trial_number must be a positive integer.")
    material = "\0".join((target_revision, run_id, suite_id, case_id, str(trial_number)))
    return str(uuid5(_WORKFLOW_TRIAL_SESSION_NAMESPACE, material))


def workflow_eval_output_sha256(final_output: str) -> str:
    """Bind one projected text result without duplicating it in evidence metadata."""

    final_output = require_durable_text(final_output, "final_output")
    return hashlib.sha256(
        _WORKFLOW_OUTPUT_REVISION_DOMAIN + final_output.encode("utf-8")
    ).hexdigest()


def workflow_eval_input_messages_sha256(messages: tuple[Message, ...]) -> str:
    """Bind the exact candidate input retained on a workflow-root trajectory."""

    if not isinstance(messages, tuple):
        raise TypeError("messages must be a tuple of exact Message values.")
    copied: list[Message] = []
    for message in messages:
        if type(message) is not Message:
            raise TypeError("messages must contain exact Message values.")
        copied.append(detach_message(message))
    return hashlib.sha256(
        _WORKFLOW_INPUT_REVISION_DOMAIN
        + canonical_durable_json_bytes(
            [message.model_dump(mode="json") for message in copied],
            "workflow eval input messages",
        )
    ).hexdigest()


__all__ = [
    "WORKFLOW_EVAL_DEFAULT_CLOSE_TIMEOUT_SECONDS",
    "WORKFLOW_EVAL_MAX_APPLICATION_CONTEXT_BYTES",
    "WORKFLOW_EVAL_MAX_CLOSE_TIMEOUT_SECONDS",
    "WORKFLOW_EVAL_MAX_FINAL_OUTPUT_CHARS",
    "WORKFLOW_EVAL_MAX_INPUT_BYTES",
    "WORKFLOW_EVAL_MAX_INPUT_MESSAGES",
    "WORKFLOW_EVAL_MAX_STRUCTURED_OUTPUT_BYTES",
    "WorkflowEvalExecution",
    "WorkflowEvalFactory",
    "WorkflowEvalFailure",
    "WorkflowEvalFailureCode",
    "WorkflowEvalInstanceScope",
    "WorkflowEvalInvocation",
    "WorkflowEvalOutputEvidenceV1",
    "WorkflowEvalResult",
    "WorkflowEvalResultProjector",
    "WorkflowEvalTargetIdentityV1",
    "WorkflowEvalTerminalEvidence",
    "workflow_eval_input_messages_sha256",
    "workflow_eval_output_sha256",
    "workflow_eval_trial_session_id",
    "workflow_spec_revision",
]
