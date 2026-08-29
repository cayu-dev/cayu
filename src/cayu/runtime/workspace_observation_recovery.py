"""Durable workspace-observation lifecycle and exact transition publication."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict, StrictInt, field_validator, model_validator

from cayu._exception_groups import (
    add_exception_note_safely,
    exception_tree_contains,
    iter_exception_tree,
)
from cayu._exception_state import exception_state, set_exception_state
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    ShieldedTaskOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_clean_nonblank,
)
from cayu.artifacts import ArtifactMetadata, ArtifactScope
from cayu.core.events import Event, copy_event
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY,
)
from cayu.runtime.public_authority import (
    PublicAuthorityAliasCodec,
    public_authority_alias_is_reserved,
)
from cayu.runtime.sessions import (
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
    RuntimePublicationReceipt,
    RuntimePublicationRequest,
    RuntimePublicationResult,
    Session,
    SessionStatus,
    SessionStore,
    copy_session,
    runtime_publication_checkpoint_value_digest,
    runtime_publication_request_digest,
)
from cayu.vaults import SecretRedactor

if TYPE_CHECKING:
    from cayu.runtime._event_writer import RuntimeEventWriter

WORKSPACE_OBSERVATION_RECORD_TYPE = "cayu.workspace-observation"
WORKSPACE_OBSERVATION_SCHEMA_VERSION = 1
WORKSPACE_OBSERVATION_MAX_ACTIVE = 256
WORKSPACE_OBSERVATION_ARTIFACT_MAX_BYTES = 16 * 1024 * 1024
_WORKSPACE_OBSERVATION_MAX_ABANDONED_READS = 256
_WORKSPACE_OBSERVATION_ABANDONED_READS: set[asyncio.Task[Any]] = set()
_ReadT = TypeVar("_ReadT")
_MutationT = TypeVar("_MutationT")
WorkspaceObservationEvidenceKind = Literal[
    "revision-before",
    "revision-after",
    "revision-delta",
]
WorkspaceObservationObserverAuthority = Literal["runtime_builtin", "configured"]

_WORKSPACE_OBSERVATION_WORKSPACE_ALIAS_FIELD = "workspace_observation_workspace_id"
_WORKSPACE_OBSERVATION_OBSERVER_ALIAS_FIELD = "workspace_observation_observer"
_WORKSPACE_OBSERVATION_ARTIFACT_STORE_ALIAS_FIELD = "workspace_observation_artifact_store_id"
_WORKSPACE_OBSERVATION_PENDING_CANCELLATION_ATTRIBUTE = (
    "_cayu_workspace_observation_pending_cancellation"
)
_WORKSPACE_OBSERVATION_PENDING_CANCELLATION_AUTHORITY = object()
_WORKSPACE_OBSERVATION_RECOVERY_REJECTED_AUTHORITY = object()


class _WorkspaceObservationRecoveryRejected(RuntimeError):
    """Private proof that durable observation authority cannot be recovered."""

    def __init__(self, message: str, *, _authority: object) -> None:
        if _authority is not _WORKSPACE_OBSERVATION_RECOVERY_REJECTED_AUTHORITY:
            raise TypeError("Workspace observation recovery rejection is runtime-owned.")
        super().__init__(require_clean_nonblank(message, "message"))


_WORKSPACE_OBSERVATION_RECOVERY_REJECTION_PROVENANCE: WeakKeyDictionary[
    _WorkspaceObservationRecoveryRejected,
    bool,
] = WeakKeyDictionary()


def workspace_observation_recovery_rejected(message: str) -> RuntimeError:
    """Create authenticated evidence that deterministic recovery must stop retrying."""

    signal = _WorkspaceObservationRecoveryRejected(
        message,
        _authority=_WORKSPACE_OBSERVATION_RECOVERY_REJECTED_AUTHORITY,
    )
    _WORKSPACE_OBSERVATION_RECOVERY_REJECTION_PROVENANCE[signal] = True
    return signal


def is_workspace_observation_recovery_rejected(error: BaseException) -> bool:
    """Recognize only runtime-created permanent observation recovery rejection."""

    return (
        type(error) is _WorkspaceObservationRecoveryRejected
        and error in _WORKSPACE_OBSERVATION_RECOVERY_REJECTION_PROVENANCE
    )


def restore_workspace_observation_cancellation_requests(count: int) -> None:
    """Restore shield-consumed requests immediately before control escapes."""

    if type(count) is not int or count < 0:
        raise ValueError("Consumed cancellation request count must be a non-negative int.")
    current_task = asyncio.current_task()
    if current_task is not None:
        for _request in range(count):
            current_task.cancel()


@dataclass(frozen=True, slots=True)
class _WorkspaceObservationAuthorityProjection:
    """Private raw authority paired with its secret-safe durable representation."""

    configured_workspace_id: str | None
    configured_observer: str
    configured_artifact_store_id: str | None
    workspace_id: str
    observer: str
    observer_authority: WorkspaceObservationObserverAuthority
    artifact_store_id: str | None

    @property
    def configured_identity(self) -> tuple[str, str]:
        return (
            self.configured_workspace_id or "workspace-unavailable",
            self.configured_observer,
        )


class _WorkspaceObservationIntentAdmission:
    """Private positive authority for one secret-safe initial lifecycle intent."""

    __slots__ = ("__lifecycle", "__token")

    def __init__(self, lifecycle: WorkspaceObservationLifecycle, *, _token: object) -> None:
        if _token is not _WORKSPACE_OBSERVATION_ADMISSION_TOKEN:
            raise TypeError("Workspace observation intent admission is runtime-owned.")
        self.__lifecycle = lifecycle
        self.__token = _token

    def matches(self, lifecycle: WorkspaceObservationLifecycle) -> bool:
        return (
            self.__token is _WORKSPACE_OBSERVATION_ADMISSION_TOKEN and self.__lifecycle == lifecycle
        )


_WORKSPACE_OBSERVATION_ADMISSION_TOKEN = object()


def _workspace_observation_captured_task_outcome(
    outcome: ShieldedTaskOutcome[CapturedAwaitableOutcome[_ReadT]],
    *,
    operation: str,
) -> tuple[_ReadT | None, BaseException | None]:
    """Detach one extension outcome captured inside an owned child task."""

    if outcome.error is not None:
        return None, _workspace_observation_operation_error(
            outcome.error,
            operation=operation,
        )
    captured = outcome.result
    if type(captured) is not CapturedAwaitableOutcome:
        return None, RuntimeError(f"{operation} returned an invalid owned outcome.")
    return captured.result, _workspace_observation_operation_error(
        captured.error,
        operation=operation,
    )


def _workspace_observation_operation_error(
    error: BaseException | None,
    *,
    operation: str,
) -> BaseException | None:
    """Classify extension-owned cancellation without inventing caller control."""

    if isinstance(error, asyncio.CancelledError):
        return unexpected_child_cancellation_error(error, operation=operation)
    if (
        isinstance(error, BaseExceptionGroup)
        and exception_tree_contains(error, asyncio.CancelledError)
        and not exception_tree_contains(error, (KeyboardInterrupt, SystemExit, GeneratorExit))
    ):
        classified = RuntimeError(f"{operation} reported multiple operational failures.")
        classified.__cause__ = error
        return classified
    return error


def _workspace_observation_failure_group(
    message: str,
    *failures: BaseException,
) -> BaseExceptionGroup:
    """Preserve ordered, identity-distinct failures without flattening their trees."""

    ordered: list[BaseException] = []
    for failure in failures:
        if all(failure is not existing for existing in ordered):
            ordered.append(failure)
    if all(isinstance(failure, Exception) for failure in ordered):
        return ExceptionGroup(message, cast("list[Exception]", ordered))
    return BaseExceptionGroup(message, ordered)


class _WorkspaceObservationConcurrentControl(BaseExceptionGroup):
    """Concurrent observer control plus exact restored task-cancellation count."""

    cancellation_requests_pending: int

    def __new__(
        cls,
        message: str,
        failures: Sequence[BaseException],
        cancellation_requests_pending: int,
    ) -> _WorkspaceObservationConcurrentControl:
        grouped = super().__new__(cls, message, failures)
        grouped.cancellation_requests_pending = cancellation_requests_pending
        return grouped

    def __init__(
        self,
        message: str,
        failures: Sequence[BaseException],
        cancellation_requests_pending: int,
    ) -> None:
        super().__init__(message, failures)


def workspace_observation_pending_cancellation_requests(error: BaseException) -> int:
    """Return restored cancellation requests authenticated by observation control."""

    return max(
        (
            max(
                _workspace_observation_concurrent_cancellation_requests(candidate),
                _workspace_observation_retained_cancellation_requests(candidate),
            )
            for candidate in iter_exception_tree(error)
        ),
        default=0,
    )


def _workspace_observation_concurrent_cancellation_requests(error: BaseException) -> int:
    if type(error) is not _WorkspaceObservationConcurrentControl:
        return 0
    count = exception_state(error, "cancellation_requests_pending")
    return count if type(count) is int and count > 0 else 0


def retain_workspace_observation_pending_cancellation_requests(
    error: BaseException,
    count: int,
) -> None:
    """Retain authenticated cancellation ownership across a runtime rebuild."""

    if not isinstance(error, BaseException):
        raise TypeError("error must be a BaseException.")
    if type(count) is not int or count <= 0:
        raise ValueError("Pending cancellation request count must be a positive int.")
    retained = max(count, workspace_observation_pending_cancellation_requests(error))
    if not set_exception_state(
        error,
        _WORKSPACE_OBSERVATION_PENDING_CANCELLATION_ATTRIBUTE,
        (
            _WORKSPACE_OBSERVATION_PENDING_CANCELLATION_AUTHORITY,
            retained,
        ),
    ):
        raise RuntimeError("Could not retain workspace observation cancellation authority.")


def copy_workspace_observation_pending_cancellation_requests(
    source: BaseException,
    target: BaseException,
) -> None:
    """Copy authenticated cancellation ownership to a sanitized failure."""

    count = workspace_observation_pending_cancellation_requests(source)
    if count:
        retain_workspace_observation_pending_cancellation_requests(target, count)


def _workspace_observation_retained_cancellation_requests(error: BaseException) -> int:
    authority = exception_state(
        error,
        _WORKSPACE_OBSERVATION_PENDING_CANCELLATION_ATTRIBUTE,
    )
    if (
        type(authority) is tuple
        and len(authority) == 2
        and authority[0] is _WORKSPACE_OBSERVATION_PENDING_CANCELLATION_AUTHORITY
        and type(authority[1]) is int
        and authority[1] > 0
    ):
        return authority[1]
    return 0


def raise_workspace_observation_concurrent_control(
    *,
    cancellation: asyncio.CancelledError | None,
    error: BaseException | None,
    operation: str,
    cancellation_requests_pending: int = 0,
) -> None:
    """Preserve process control that settles alongside caller cancellation.

    Observation and artifact adapters are extension boundaries.  Their ordinary
    failures remain content-free when caller cancellation is authoritative, but
    a real process-control signal must neither be rewritten nor discarded.
    """

    if cancellation is None:
        return
    operation = require_clean_nonblank(operation, "operation")
    if type(cancellation_requests_pending) is not int or cancellation_requests_pending < 0:
        raise ValueError("Pending cancellation request count must be a non-negative int.")
    if error is None:
        raise cancellation
    if exception_tree_contains(error, (GeneratorExit, KeyboardInterrupt, SystemExit)):
        raise _WorkspaceObservationConcurrentControl(
            f"{operation} received concurrent process control and caller cancellation.",
            [cancellation, error],
            cancellation_requests_pending,
        ) from None
    # Preserve normal ``except CancelledError`` compatibility without attaching
    # an extension-owned traceback or diagnostic to the caller's control
    # signal.  The fixed cause records that another outcome was present.
    concurrent_failure = RuntimeError(f"{operation} failed concurrently with caller cancellation.")
    raise cancellation from concurrent_failure


class WorkspaceObservationPhase(StrEnum):
    INTENT = "intent"
    BEFORE_CAPTURED = "before_captured"
    TOOL_OUTCOME_STAGED = "tool_outcome_staged"
    AFTER_CAPTURED = "after_captured"
    DELTA_PUBLISHED = "delta_published"


class WorkspaceObservationEvidenceState(StrEnum):
    PENDING = "pending"
    CAPTURED_PRIVATE = "captured_private"
    PUBLISHED = "published"
    QUARANTINED = "quarantined"
    FAILED = "failed"
    MISSING = "missing"


class WorkspaceObservationTerminalStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"


WORKSPACE_OBSERVATION_TERMINAL_CONTROLS = frozenset(
    {
        (WorkspaceObservationTerminalStatus.COMPLETE.value, None),
        (
            WorkspaceObservationTerminalStatus.INCOMPLETE.value,
            "receipt_publication_interrupted",
        ),
        (
            WorkspaceObservationTerminalStatus.INCOMPLETE.value,
            "referenced_workspace_artifact_missing",
        ),
        (
            WorkspaceObservationTerminalStatus.INCOMPLETE.value,
            "worker_lost_before_workspace_observation_completed",
        ),
        (
            WorkspaceObservationTerminalStatus.INCOMPLETE.value,
            "workspace_artifact_verification_failed",
        ),
        (
            WorkspaceObservationTerminalStatus.INCOMPLETE.value,
            "workspace_delta_evidence_missing",
        ),
        (
            WorkspaceObservationTerminalStatus.INCOMPLETE.value,
            "workspace_revision_evidence_incomplete",
        ),
        (
            WorkspaceObservationTerminalStatus.AMBIGUOUS.value,
            "durable_tool_outcome_evidence_missing",
        ),
        (
            WorkspaceObservationTerminalStatus.AMBIGUOUS.value,
            "worker_lost_before_tool_outcome_was_durable",
        ),
        (
            WorkspaceObservationTerminalStatus.AMBIGUOUS.value,
            "workspace_delta_evidence_conflict",
        ),
        (
            WorkspaceObservationTerminalStatus.FAILED.value,
            "mutation_settlement_unproven",
        ),
        (
            WorkspaceObservationTerminalStatus.FAILED.value,
            "receipt_publication_failed",
        ),
        (
            WorkspaceObservationTerminalStatus.FAILED.value,
            "workspace_revision_comparison_failed",
        ),
    }
)


def workspace_observation_terminal_from_delta_status(
    status: str,
    *,
    detail_code: str | None = None,
) -> tuple[WorkspaceObservationTerminalStatus, str | None]:
    """Map durable delta status to one fixed terminal lifecycle classification."""

    if type(status) is not str:
        raise TypeError("Workspace mutation status must be a string.")
    if detail_code is not None:
        detail_code = require_clean_nonblank(detail_code, "detail_code")
    if status in {"changed", "no_change"}:
        return WorkspaceObservationTerminalStatus.COMPLETE, None
    if status == "failed":
        if detail_code in {
            "manifest_artifact_reference_invalid",
            "manifest_redaction_failed",
        }:
            return (
                WorkspaceObservationTerminalStatus.INCOMPLETE,
                "workspace_revision_evidence_incomplete",
            )
        return (
            WorkspaceObservationTerminalStatus.FAILED,
            "workspace_revision_comparison_failed",
        )
    if status in {"unsupported", "incomplete", "truncated"}:
        return (
            WorkspaceObservationTerminalStatus.INCOMPLETE,
            "workspace_revision_evidence_incomplete",
        )
    raise ValueError("Workspace mutation status is not recognized.")


class WorkspaceObservationArtifactState(StrEnum):
    INTENT = "intent"
    PUBLISHED = "published"
    REFERENCED = "referenced"
    FAILED = "failed"
    ORPHANED = "orphaned"
    MISSING = "missing"


class WorkspaceObservationArtifact(BaseModel):
    """Content-bound artifact state retained until its event reference is durable."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    evidence_kind: WorkspaceObservationEvidenceKind
    artifact_id: str
    sha256: str
    size_bytes: StrictInt
    state: WorkspaceObservationArtifactState

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "artifact_id")

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("sha256 must be a lowercase SHA-256 digest.")
        return value

    @field_validator("size_bytes")
    @classmethod
    def validate_size_bytes(cls, value: int) -> int:
        if type(value) is not int or value < 1 or value > WORKSPACE_OBSERVATION_ARTIFACT_MAX_BYTES:
            raise ValueError("size_bytes must be a positive bounded integer.")
        return value


class WorkspaceObservationLifecycle(BaseModel):
    """Bounded active state for one exact workspace mutation window."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.workspace-observation"] = WORKSPACE_OBSERVATION_RECORD_TYPE
    schema_version: StrictInt = WORKSPACE_OBSERVATION_SCHEMA_VERSION
    session_id: str
    interaction_id: str | None = None
    window_id: str
    source_run_epoch: StrictInt
    binding_generation_id: str
    workspace_id: str
    observer: str
    observer_authority: WorkspaceObservationObserverAuthority
    artifact_store_id: str | None = None
    agent_name: str
    environment_name: str | None = None
    tool_name: str
    tool_call_id: str
    model_step_id: str
    model_attempt_id: str
    tool_round_id: str
    model_step: StrictInt | None = None
    phase: WorkspaceObservationPhase = WorkspaceObservationPhase.INTENT
    before_state: WorkspaceObservationEvidenceState = WorkspaceObservationEvidenceState.PENDING
    before_observation_id: str | None = None
    tool_outcome_event_id: str | None = None
    tool_outcome_event_digest: str | None = None
    after_state: WorkspaceObservationEvidenceState = WorkspaceObservationEvidenceState.PENDING
    after_observation_id: str | None = None
    delta_state: WorkspaceObservationEvidenceState = WorkspaceObservationEvidenceState.PENDING
    mutation_event_id: str | None = None
    mutation_event_digest: str | None = None
    artifacts: tuple[WorkspaceObservationArtifact, ...] = ()

    @field_validator(
        "session_id",
        "window_id",
        "binding_generation_id",
        "workspace_id",
        "observer",
        "agent_name",
        "tool_name",
        "tool_call_id",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "interaction_id",
        "environment_name",
        "artifact_store_id",
        "before_observation_id",
        "tool_outcome_event_id",
        "after_observation_id",
        "mutation_event_id",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        return None if value is None else require_clean_nonblank(value, info.field_name)

    @field_validator("source_run_epoch")
    @classmethod
    def validate_source_run_epoch(cls, value: int) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("source_run_epoch must be a non-negative integer.")
        return value

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: int) -> int:
        if value != WORKSPACE_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("schema_version is not supported.")
        return value

    @field_validator("model_step")
    @classmethod
    def validate_model_step(cls, value: int | None) -> int | None:
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("model_step must be a positive integer or None.")
        return value

    @field_validator("tool_outcome_event_digest", "mutation_event_digest")
    @classmethod
    def validate_optional_digest(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def validate_phase_material(self) -> WorkspaceObservationLifecycle:
        phase_order = {
            WorkspaceObservationPhase.INTENT: 0,
            WorkspaceObservationPhase.BEFORE_CAPTURED: 1,
            WorkspaceObservationPhase.TOOL_OUTCOME_STAGED: 2,
            WorkspaceObservationPhase.AFTER_CAPTURED: 3,
            WorkspaceObservationPhase.DELTA_PUBLISHED: 4,
        }
        if (
            phase_order[self.phase] >= 1
            and self.before_state is WorkspaceObservationEvidenceState.PENDING
        ):
            raise ValueError("A captured-before phase requires bounded before evidence state.")
        if self.phase is WorkspaceObservationPhase.INTENT and (
            self.before_state is not WorkspaceObservationEvidenceState.PENDING
            or self.before_observation_id is not None
            or self.artifacts
        ):
            raise ValueError("An observation intent cannot contain captured evidence.")
        if (self.tool_outcome_event_id is None) != (self.tool_outcome_event_digest is None):
            raise ValueError("Tool outcome event identity and digest must be present together.")
        if phase_order[self.phase] >= 2 and self.tool_outcome_event_id is None:
            raise ValueError("A staged-outcome phase requires content-bound terminal evidence.")
        if phase_order[self.phase] < 2 and self.tool_outcome_event_id is not None:
            raise ValueError("Tool outcome evidence cannot precede its staged-outcome phase.")
        if (
            phase_order[self.phase] >= 3
            and self.after_state is WorkspaceObservationEvidenceState.PENDING
        ):
            raise ValueError("An after-captured phase requires bounded after evidence state.")
        if phase_order[self.phase] < 3 and (
            self.after_state is not WorkspaceObservationEvidenceState.PENDING
            or self.after_observation_id is not None
        ):
            raise ValueError("After-capture evidence cannot precede its lifecycle phase.")
        if phase_order[self.phase] >= 3 and (
            self.before_observation_id is None or self.after_observation_id is None
        ):
            raise ValueError("An after-captured phase requires both observation event identities.")
        if (self.mutation_event_id is None) != (self.mutation_event_digest is None):
            raise ValueError("Mutation event identity and digest must be present together.")
        if phase_order[self.phase] < 4 and (
            self.delta_state is not WorkspaceObservationEvidenceState.PENDING
            or self.mutation_event_id is not None
        ):
            raise ValueError("Delta evidence cannot precede its published lifecycle phase.")
        if phase_order[self.phase] >= 4 and (
            self.delta_state is WorkspaceObservationEvidenceState.PENDING
            or self.mutation_event_id is None
            or self.mutation_event_digest is None
        ):
            raise ValueError("A delta-published phase requires bounded terminal evidence state.")
        kinds = tuple(artifact.evidence_kind for artifact in self.artifacts)
        if len(kinds) > 3 or len(set(kinds)) != len(kinds):
            raise ValueError("Workspace observation artifacts must be unique and bounded.")
        if self.artifacts and self.artifact_store_id is None:
            raise ValueError(
                "Workspace observation artifacts require an authoritative artifact store."
            )
        return self

    def authority_tuple(self) -> tuple[Any, ...]:
        return (
            self.session_id,
            self.interaction_id,
            self.window_id,
            self.source_run_epoch,
            self.binding_generation_id,
            self.workspace_id,
            self.observer,
            self.observer_authority,
            self.artifact_store_id,
            self.agent_name,
            self.environment_name,
            self.tool_name,
            self.tool_call_id,
            self.model_step_id,
            self.model_attempt_id,
            self.tool_round_id,
            self.model_step,
        )


def _project_workspace_observation_authority(
    *,
    session_id: str,
    configured_workspace_id: str | None,
    configured_observer: str,
    configured_artifact_store_id: str | None,
    observer_is_runtime_owned: bool,
    secret_resolution_scope: Literal["static", "dynamic"],
    redactor: SecretRedactor,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
) -> _WorkspaceObservationAuthorityProjection:
    """Project raw observation authority before its first durable write.

    A dynamic environment can resolve a value only after the intent is durable,
    so configurable identities cannot be proven non-secret at this boundary.
    They are retained only through keyed, field-scoped aliases. Runtime-owned
    observer names keep their structural provenance and are never authenticated
    merely because an extension returned the same string.
    """

    session_id = require_clean_nonblank(session_id, "session_id")
    configured_observer = require_clean_nonblank(
        configured_observer,
        "configured_observer",
    )
    if type(observer_is_runtime_owned) is not bool:
        raise TypeError("observer_is_runtime_owned must be a boolean.")
    if secret_resolution_scope not in {"static", "dynamic"}:
        raise ValueError("secret_resolution_scope must be static or dynamic.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    if configured_workspace_id is not None:
        configured_workspace_id = require_clean_nonblank(
            configured_workspace_id,
            "configured_workspace_id",
        )
    if configured_artifact_store_id is not None:
        configured_artifact_store_id = require_clean_nonblank(
            configured_artifact_store_id,
            "configured_artifact_store_id",
        )

    configurable_values = (
        configured_workspace_id,
        configured_artifact_store_id,
        None if observer_is_runtime_owned else configured_observer,
    )
    if any(
        value is not None and public_authority_alias_is_reserved(value)
        for value in configurable_values
    ):
        raise ValueError(
            "Workspace observation configured identity uses the reserved authority-alias namespace."
        )

    if secret_resolution_scope == "static":
        if any(
            value is not None and redactor.redact_text(value) != value
            for value in configurable_values
        ):
            raise ValueError(
                "Workspace observation configured identity contains a workload secret; "
                "refusing durable observation intent."
            )
        return _WorkspaceObservationAuthorityProjection(
            configured_workspace_id=configured_workspace_id,
            configured_observer=configured_observer,
            configured_artifact_store_id=configured_artifact_store_id,
            workspace_id=configured_workspace_id or "workspace-unavailable",
            observer=configured_observer,
            observer_authority=("runtime_builtin" if observer_is_runtime_owned else "configured"),
            artifact_store_id=configured_artifact_store_id,
        )

    if not isinstance(public_authority_alias_codec, PublicAuthorityAliasCodec):
        raise RuntimeError(
            "Dynamic workspace observation authority requires a configured public "
            "authority alias keyring."
        )

    def alias(value: str, field_name: str) -> str:
        return public_authority_alias_codec.encode(
            value,
            field_name=field_name,
            session_id=session_id,
        )

    return _WorkspaceObservationAuthorityProjection(
        configured_workspace_id=configured_workspace_id,
        configured_observer=configured_observer,
        configured_artifact_store_id=configured_artifact_store_id,
        workspace_id=(
            "workspace-unavailable"
            if configured_workspace_id is None
            else alias(
                configured_workspace_id,
                _WORKSPACE_OBSERVATION_WORKSPACE_ALIAS_FIELD,
            )
        ),
        observer=(
            configured_observer
            if observer_is_runtime_owned
            else alias(
                configured_observer,
                _WORKSPACE_OBSERVATION_OBSERVER_ALIAS_FIELD,
            )
        ),
        observer_authority=("runtime_builtin" if observer_is_runtime_owned else "configured"),
        artifact_store_id=(
            None
            if configured_artifact_store_id is None
            else alias(
                configured_artifact_store_id,
                _WORKSPACE_OBSERVATION_ARTIFACT_STORE_ALIAS_FIELD,
            )
        ),
    )


def workspace_observation_authority_matches(
    durable_value: str | None,
    configured_value: str | None,
    *,
    field_name: Literal["workspace_id", "observer", "artifact_store_id"],
    session_id: str,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
) -> bool:
    """Match raw current authority against a raw or keyed durable identity."""

    if durable_value is None or configured_value is None:
        return durable_value is None and configured_value is None
    if type(durable_value) is not str or type(configured_value) is not str:
        return False
    alias_field = {
        "workspace_id": _WORKSPACE_OBSERVATION_WORKSPACE_ALIAS_FIELD,
        "observer": _WORKSPACE_OBSERVATION_OBSERVER_ALIAS_FIELD,
        "artifact_store_id": _WORKSPACE_OBSERVATION_ARTIFACT_STORE_ALIAS_FIELD,
    }[field_name]
    if public_authority_alias_is_reserved(durable_value):
        return isinstance(
            public_authority_alias_codec,
            PublicAuthorityAliasCodec,
        ) and public_authority_alias_codec.matches(
            durable_value,
            configured_value,
            field_name=alias_field,
            session_id=session_id,
        )
    return durable_value == configured_value


def workspace_observation_observer_authority_matches(
    durable_observer: str,
    durable_authority: WorkspaceObservationObserverAuthority,
    configured_observer: str,
    *,
    configured_observer_is_runtime_owned: bool,
    session_id: str,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
) -> bool:
    """Match an observer without allowing class-name equality to invent provenance."""

    if (
        type(durable_observer) is not str
        or type(configured_observer) is not str
        or type(configured_observer_is_runtime_owned) is not bool
    ):
        return False
    if durable_authority == "runtime_builtin":
        return configured_observer_is_runtime_owned and durable_observer == configured_observer
    if durable_authority != "configured" or configured_observer_is_runtime_owned:
        return False
    return workspace_observation_authority_matches(
        durable_observer,
        configured_observer,
        field_name="observer",
        session_id=session_id,
        public_authority_alias_codec=public_authority_alias_codec,
    )


def _admit_workspace_observation_intent(
    lifecycle: WorkspaceObservationLifecycle,
    *,
    redactor: SecretRedactor,
    configured_workspace_id: str | None,
    configured_artifact_store_id: str | None,
    authority_projection: _WorkspaceObservationAuthorityProjection | None = None,
) -> _WorkspaceObservationIntentAdmission:
    """Admit configured identities before their first durable checkpoint write."""

    if type(lifecycle) is not WorkspaceObservationLifecycle:
        raise TypeError("lifecycle must be a WorkspaceObservationLifecycle.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    if authority_projection is None:
        authority_projection = _project_workspace_observation_authority(
            session_id=lifecycle.session_id,
            configured_workspace_id=configured_workspace_id,
            configured_observer=lifecycle.observer,
            configured_artifact_store_id=configured_artifact_store_id,
            observer_is_runtime_owned=(lifecycle.observer_authority == "runtime_builtin"),
            secret_resolution_scope="static",
            redactor=redactor,
            public_authority_alias_codec=None,
        )
    if type(authority_projection) is not _WorkspaceObservationAuthorityProjection:
        raise TypeError(
            "authority_projection must be runtime-owned workspace observation authority."
        )
    if authority_projection.configured_workspace_id != configured_workspace_id:
        raise ValueError("Workspace observation admission changed its configured workspace ID.")
    if authority_projection.configured_artifact_store_id != configured_artifact_store_id:
        raise ValueError(
            "Workspace observation admission changed its configured artifact-store ID."
        )
    if lifecycle.workspace_id != authority_projection.workspace_id:
        raise ValueError("Workspace observation intent changed its configured workspace identity.")
    if lifecycle.observer != authority_projection.observer:
        raise ValueError("Workspace observation intent changed its observer identity.")
    if lifecycle.observer_authority != authority_projection.observer_authority:
        raise ValueError("Workspace observation intent changed its observer authority.")
    if lifecycle.artifact_store_id != authority_projection.artifact_store_id:
        raise ValueError("Workspace observation intent changed its artifact-store identity.")
    return _WorkspaceObservationIntentAdmission(
        lifecycle,
        _token=_WORKSPACE_OBSERVATION_ADMISSION_TOKEN,
    )


def workspace_observation_event_digest(event: Event) -> str:
    copied = copy_event(event)
    payload = dict(copied.payload)
    # Capture controls are finalized after the authoritative tool outcome is
    # staged.  They are observation metadata, not part of the tool outcome's
    # content identity.
    payload.pop("workspace_mutation_capture_status", None)
    payload.pop("workspace_mutation_capture_detail_code", None)
    copied = copied.model_copy(update={"payload": payload}, deep=True)
    encoded = json.dumps(
        copied.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def workspace_observations_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> dict[str, WorkspaceObservationLifecycle]:
    if checkpoint is None:
        return {}
    if type(checkpoint) is not dict:
        raise TypeError("Workspace observation checkpoint must be an object or None.")
    if WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY not in checkpoint:
        return {}
    raw = checkpoint[WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY]
    if type(raw) is not dict or len(raw) > WORKSPACE_OBSERVATION_MAX_ACTIVE:
        raise ValueError(
            "Workspace observation checkpoint state is malformed or exceeds its bound."
        )
    observations: dict[str, WorkspaceObservationLifecycle] = {}
    for raw_window_id, raw_record in raw.items():
        if type(raw_window_id) is not str or type(raw_record) is not dict:
            raise ValueError("Workspace observation checkpoint entries are malformed.")
        record = WorkspaceObservationLifecycle.model_validate(raw_record)
        if record.window_id != raw_window_id:
            raise ValueError("Workspace observation checkpoint key conflicts with its record.")
        if any(
            artifact.state
            in {
                WorkspaceObservationArtifactState.ORPHANED,
                WorkspaceObservationArtifactState.MISSING,
            }
            for artifact in record.artifacts
        ):
            raise ValueError("Active workspace observation contains terminal-only artifact state.")
        observations[raw_window_id] = record
    return observations


def workspace_observation_checkpoint_value(
    observations: dict[str, WorkspaceObservationLifecycle],
) -> dict[str, Any]:
    if len(observations) > WORKSPACE_OBSERVATION_MAX_ACTIVE:
        raise ValueError("Too many active workspace observations.")
    return {
        window_id: observations[window_id].model_dump(mode="json")
        for window_id in sorted(observations)
    }


def validate_workspace_observation_transition(
    *,
    previous: WorkspaceObservationLifecycle | None,
    current: WorkspaceObservationLifecycle | None,
    phase: str,
    terminal_status: WorkspaceObservationTerminalStatus | None = None,
    terminal_artifacts: tuple[WorkspaceObservationArtifact, ...] | None = None,
) -> None:
    """Validate one legal, exact lifecycle edge before durable publication."""

    phase = require_clean_nonblank(phase, "phase")
    if previous is None and current is None:
        raise ValueError("Workspace observation transition requires lifecycle state.")
    if previous is not None and type(previous) is not WorkspaceObservationLifecycle:
        raise TypeError("previous must be a WorkspaceObservationLifecycle or None.")
    if current is not None and type(current) is not WorkspaceObservationLifecycle:
        raise TypeError("current must be a WorkspaceObservationLifecycle or None.")
    if previous is None:
        if phase != "intent" or current is None:
            raise ValueError("Only an intent transition may create workspace observation state.")
        if current.phase is not WorkspaceObservationPhase.INTENT:
            raise ValueError("Workspace observation intent has an invalid target phase.")
        return
    if current is None:
        if phase != "terminal":
            raise ValueError("Only a terminal transition may remove workspace observation state.")
        if terminal_status is None:
            raise ValueError("A terminal transition requires a terminal classification.")
        if (
            terminal_status is WorkspaceObservationTerminalStatus.COMPLETE
            and previous.phase is not WorkspaceObservationPhase.DELTA_PUBLISHED
        ):
            raise ValueError("Only a published workspace delta may complete its lifecycle.")
        _validate_terminal_artifact_transition(previous.artifacts, terminal_artifacts)
        return
    if terminal_status is not None:
        raise ValueError("A nonterminal transition cannot carry terminal classification.")
    if previous.authority_tuple() != current.authority_tuple():
        raise ValueError("Workspace observation authority cannot change across a transition.")

    changed = {
        field_name
        for field_name in WorkspaceObservationLifecycle.model_fields
        if getattr(previous, field_name) != getattr(current, field_name)
    }
    if phase == "before-capture":
        _require_workspace_transition_shape(
            previous,
            current,
            changed=changed,
            allowed={"phase", "before_state"},
            source_phase=WorkspaceObservationPhase.INTENT,
            target_phase=WorkspaceObservationPhase.BEFORE_CAPTURED,
        )
        if current.before_state not in {
            WorkspaceObservationEvidenceState.CAPTURED_PRIVATE,
            WorkspaceObservationEvidenceState.FAILED,
        }:
            raise ValueError("Before capture has an invalid evidence state.")
        return
    if phase in {"tool-outcome", "recovered-tool-outcome"}:
        _require_workspace_transition_shape(
            previous,
            current,
            changed=changed,
            allowed={"phase", "tool_outcome_event_id", "tool_outcome_event_digest"},
            source_phase=WorkspaceObservationPhase.BEFORE_CAPTURED,
            target_phase=WorkspaceObservationPhase.TOOL_OUTCOME_STAGED,
        )
        return
    if phase.startswith("artifact-"):
        if changed != {"artifacts"} or previous.phase is not current.phase:
            raise ValueError("Workspace artifact publication changed unrelated lifecycle state.")
        raw_evidence_kind, artifact_state = phase.removeprefix("artifact-").rsplit("-", 1)
        if raw_evidence_kind not in {
            "revision-before",
            "revision-after",
            "revision-delta",
        }:
            raise ValueError("Workspace artifact publication has an invalid evidence kind.")
        evidence_kind = cast("WorkspaceObservationEvidenceKind", raw_evidence_kind)
        expected_phase = (
            WorkspaceObservationPhase.AFTER_CAPTURED
            if evidence_kind == "revision-delta"
            else WorkspaceObservationPhase.TOOL_OUTCOME_STAGED
        )
        if previous.phase is not expected_phase:
            raise ValueError("Workspace artifact publication has an invalid evidence phase.")
        if evidence_kind == "revision-after" and (
            previous.before_observation_id is None
            or previous.before_state
            not in {
                WorkspaceObservationEvidenceState.PUBLISHED,
                WorkspaceObservationEvidenceState.QUARANTINED,
                WorkspaceObservationEvidenceState.FAILED,
            }
        ):
            raise ValueError("After-observation artifact preceded durable before evidence.")
        _validate_artifact_transition(
            previous.artifacts,
            current.artifacts,
            evidence_kind=evidence_kind,
            target_state=WorkspaceObservationArtifactState(artifact_state),
        )
        return
    if phase == "before-evidence":
        _require_workspace_transition_shape(
            previous,
            current,
            changed=changed,
            allowed={"before_state", "before_observation_id", "artifacts"},
            source_phase=WorkspaceObservationPhase.TOOL_OUTCOME_STAGED,
            target_phase=WorkspaceObservationPhase.TOOL_OUTCOME_STAGED,
        )
        _validate_optional_reference_transition(
            previous.artifacts,
            current.artifacts,
            evidence_kind="revision-before",
        )
        if (
            current.before_state
            not in {
                WorkspaceObservationEvidenceState.PUBLISHED,
                WorkspaceObservationEvidenceState.QUARANTINED,
                WorkspaceObservationEvidenceState.FAILED,
            }
            or current.before_observation_id is None
        ):
            raise ValueError("Before evidence publication is incomplete.")
        return
    if phase == "after-capture":
        _require_workspace_transition_shape(
            previous,
            current,
            changed=changed,
            allowed={"phase", "after_state", "after_observation_id", "artifacts"},
            source_phase=WorkspaceObservationPhase.TOOL_OUTCOME_STAGED,
            target_phase=WorkspaceObservationPhase.AFTER_CAPTURED,
        )
        _validate_optional_reference_transition(
            previous.artifacts,
            current.artifacts,
            evidence_kind="revision-after",
        )
        if current.after_state not in {
            WorkspaceObservationEvidenceState.PUBLISHED,
            WorkspaceObservationEvidenceState.QUARANTINED,
            WorkspaceObservationEvidenceState.FAILED,
        }:
            raise ValueError("After evidence publication is incomplete.")
        return
    if phase == "delta-publication":
        _require_workspace_transition_shape(
            previous,
            current,
            changed=changed,
            allowed={
                "phase",
                "delta_state",
                "mutation_event_id",
                "mutation_event_digest",
                "artifacts",
            },
            source_phase=WorkspaceObservationPhase.AFTER_CAPTURED,
            target_phase=WorkspaceObservationPhase.DELTA_PUBLISHED,
        )
        _validate_optional_reference_transition(
            previous.artifacts,
            current.artifacts,
            evidence_kind="revision-delta",
        )
        if current.delta_state not in {
            WorkspaceObservationEvidenceState.PUBLISHED,
            WorkspaceObservationEvidenceState.QUARANTINED,
            WorkspaceObservationEvidenceState.FAILED,
        }:
            raise ValueError("Workspace delta publication is incomplete.")
        return
    raise ValueError("Workspace observation lifecycle transition phase is invalid.")


def _require_workspace_transition_shape(
    previous: WorkspaceObservationLifecycle,
    current: WorkspaceObservationLifecycle,
    *,
    changed: set[str],
    allowed: set[str],
    source_phase: WorkspaceObservationPhase,
    target_phase: WorkspaceObservationPhase,
) -> None:
    if previous.phase is not source_phase or current.phase is not target_phase:
        raise ValueError("Workspace observation lifecycle transition has an invalid phase edge.")
    if not changed or not changed <= allowed:
        raise ValueError("Workspace observation lifecycle transition changed unrelated state.")


def _validate_artifact_transition(
    previous: tuple[WorkspaceObservationArtifact, ...],
    current: tuple[WorkspaceObservationArtifact, ...],
    *,
    evidence_kind: WorkspaceObservationEvidenceKind,
    target_state: WorkspaceObservationArtifactState,
) -> None:
    previous_by_kind = {artifact.evidence_kind: artifact for artifact in previous}
    current_by_kind = {artifact.evidence_kind: artifact for artifact in current}
    changed_kinds = {
        kind
        for kind in previous_by_kind.keys() | current_by_kind.keys()
        if previous_by_kind.get(kind) != current_by_kind.get(kind)
    }
    if changed_kinds != {evidence_kind}:
        raise ValueError("Workspace artifact publication changed unrelated evidence.")
    before = previous_by_kind.get(evidence_kind)
    after = current_by_kind.get(evidence_kind)
    if after is None or after.state is not target_state:
        raise ValueError("Workspace artifact publication has an invalid target state.")
    if before is None:
        if target_state is not WorkspaceObservationArtifactState.INTENT:
            raise ValueError("Workspace artifact must begin with durable intent.")
        return
    if before.model_dump(exclude={"state"}) != after.model_dump(exclude={"state"}):
        raise ValueError("Workspace artifact identity changed across its transition.")
    if before.state is not WorkspaceObservationArtifactState.INTENT or target_state not in {
        WorkspaceObservationArtifactState.PUBLISHED,
        WorkspaceObservationArtifactState.FAILED,
    }:
        raise ValueError("Workspace artifact lifecycle transition is invalid.")


def _validate_optional_reference_transition(
    previous: tuple[WorkspaceObservationArtifact, ...],
    current: tuple[WorkspaceObservationArtifact, ...],
    *,
    evidence_kind: WorkspaceObservationEvidenceKind,
) -> None:
    if previous == current:
        return
    previous_by_kind = {artifact.evidence_kind: artifact for artifact in previous}
    current_by_kind = {artifact.evidence_kind: artifact for artifact in current}
    if previous_by_kind.keys() != current_by_kind.keys():
        raise ValueError("Workspace evidence publication changed its artifact set.")
    changed = [kind for kind in previous_by_kind if previous_by_kind[kind] != current_by_kind[kind]]
    if changed != [evidence_kind]:
        raise ValueError("Workspace evidence publication changed unrelated artifacts.")
    before = previous_by_kind[evidence_kind]
    after = current_by_kind[evidence_kind]
    if (
        before.model_dump(exclude={"state"}) != after.model_dump(exclude={"state"})
        or before.state is not WorkspaceObservationArtifactState.PUBLISHED
        or after.state is not WorkspaceObservationArtifactState.REFERENCED
    ):
        raise ValueError("Workspace evidence references an invalid artifact transition.")


def _validate_terminal_artifact_transition(
    previous: tuple[WorkspaceObservationArtifact, ...],
    terminal: tuple[WorkspaceObservationArtifact, ...] | None,
) -> None:
    terminal = previous if terminal is None else terminal
    previous_by_kind = {artifact.evidence_kind: artifact for artifact in previous}
    terminal_by_kind = {artifact.evidence_kind: artifact for artifact in terminal}
    if previous_by_kind.keys() != terminal_by_kind.keys():
        raise ValueError("Terminal publication changed the workspace artifact set.")
    allowed = {
        WorkspaceObservationArtifactState.INTENT: {
            WorkspaceObservationArtifactState.INTENT,
            WorkspaceObservationArtifactState.ORPHANED,
        },
        WorkspaceObservationArtifactState.PUBLISHED: {
            WorkspaceObservationArtifactState.ORPHANED,
            WorkspaceObservationArtifactState.FAILED,
        },
        WorkspaceObservationArtifactState.REFERENCED: {
            WorkspaceObservationArtifactState.REFERENCED,
            WorkspaceObservationArtifactState.FAILED,
            WorkspaceObservationArtifactState.MISSING,
        },
        WorkspaceObservationArtifactState.FAILED: {
            WorkspaceObservationArtifactState.FAILED,
        },
    }
    for kind, before in previous_by_kind.items():
        after = terminal_by_kind[kind]
        if before.model_dump(exclude={"state"}) != after.model_dump(
            exclude={"state"}
        ) or after.state not in allowed.get(before.state, set()):
            raise ValueError("Terminal workspace artifact lifecycle transition is invalid.")


def workspace_observation_artifact_metadata_matches(
    metadata: object,
    *,
    artifact: WorkspaceObservationArtifact,
    session_id: str,
    agent_name: str,
    environment_name: str | None,
    window_id: str,
) -> bool:
    """Validate the complete store-owned identity of workspace evidence."""

    if type(metadata) is not ArtifactMetadata:
        return False
    text_fields = (
        metadata.id,
        metadata.filename,
        metadata.content_type,
        metadata.session_id,
        metadata.agent_name,
        metadata.environment_name,
    )
    if any(value is not None and type(value) is not str for value in text_fields):
        return False
    if type(metadata.size_bytes) is not int or type(metadata.scope) is not ArtifactScope:
        return False
    try:
        artifact_metadata = copy_durable_json_object(
            metadata.metadata,
            "workspace_observation_artifact_metadata",
        )
    except Exception:
        return False
    return (
        metadata.id == artifact.artifact_id
        and metadata.filename == f"workspace-{artifact.evidence_kind}.json"
        and metadata.content_type == "application/json"
        and metadata.size_bytes == artifact.size_bytes
        and metadata.scope is ArtifactScope.SESSION
        and metadata.session_id == session_id
        and metadata.agent_name == agent_name
        and metadata.environment_name == environment_name
        and artifact_metadata
        == {
            "schema_version": 1,
            "kind": artifact.evidence_kind,
            "sha256": artifact.sha256,
            "window_id": window_id,
        }
    )


def _phase_publication_id(window_id: str, phase: str) -> str:
    phase = require_clean_nonblank(phase, "phase")
    return f"workspace-observation:{window_id}:{phase}"


async def await_workspace_observation_store_read(
    operation_factory: Callable[[], Awaitable[_ReadT]],
    *,
    operation: str,
) -> _ReadT:
    """Own one extension store read without fabricating caller cancellation."""

    if not callable(operation_factory):
        raise TypeError("operation_factory must be callable.")
    operation = require_clean_nonblank(operation, "operation")
    _discard_completed_workspace_observation_reads()
    if len(_WORKSPACE_OBSERVATION_ABANDONED_READS) >= (_WORKSPACE_OBSERVATION_MAX_ABANDONED_READS):
        raise RuntimeError("Workspace observation store-read capacity is exhausted.")

    task = asyncio.create_task(
        capture_awaitable_outcome(operation_factory),
        name="cayu-workspace-observation-store-read",
    )
    try:
        outcome = await await_shielded_task_outcome(
            task,
            timeout_after_cancellation_s=0.0,
        )
    except BaseException:
        if not task.done():
            _retain_workspace_observation_read(task)
        raise
    if outcome.timed_out:
        _retain_workspace_observation_read(task)
        if outcome.cancellation is not None:
            restore_workspace_observation_cancellation_requests(
                outcome.cancellation_requests_consumed
            )
            raise outcome.cancellation
        raise RuntimeError(f"{operation} did not settle.") from None
    result, error = _workspace_observation_captured_task_outcome(
        outcome,
        operation=operation,
    )
    if outcome.cancellation is not None:
        if error is None:
            restore_workspace_observation_cancellation_requests(
                outcome.cancellation_requests_consumed
            )
            raise outcome.cancellation
        if isinstance(error, Exception):
            restore_workspace_observation_cancellation_requests(
                outcome.cancellation_requests_consumed
            )
            raise outcome.cancellation from error
        restore_workspace_observation_cancellation_requests(outcome.cancellation_requests_consumed)
        raise BaseExceptionGroup(
            f"{operation} failed concurrently with caller cancellation.",
            [outcome.cancellation, error],
        ) from None
    if error is not None:
        raise error
    return cast("_ReadT", result)


async def await_workspace_observation_store_mutation(
    operation_factory: Callable[[], Awaitable[_MutationT]],
    *,
    operation: str,
) -> _MutationT:
    """Own one store mutation until settlement and authenticate cancellation."""

    if not callable(operation_factory):
        raise TypeError("operation_factory must be callable.")
    operation = require_clean_nonblank(operation, "operation")

    task = asyncio.create_task(
        capture_awaitable_outcome(operation_factory),
        name="cayu-workspace-observation-store-mutation",
    )
    outcome = await await_shielded_task_outcome(task)
    result, error = _workspace_observation_captured_task_outcome(
        outcome,
        operation=operation,
    )
    if outcome.cancellation is not None:
        if error is None:
            restore_workspace_observation_cancellation_requests(
                outcome.cancellation_requests_consumed
            )
            raise outcome.cancellation
        if isinstance(error, Exception):
            restore_workspace_observation_cancellation_requests(
                outcome.cancellation_requests_consumed
            )
            raise outcome.cancellation from error
        restore_workspace_observation_cancellation_requests(outcome.cancellation_requests_consumed)
        raise BaseExceptionGroup(
            f"{operation} failed concurrently with caller cancellation.",
            [error, outcome.cancellation],
        ) from None
    if error is not None:
        raise error
    return cast("_MutationT", result)


def _retain_workspace_observation_read(task: asyncio.Task[Any]) -> None:
    _WORKSPACE_OBSERVATION_ABANDONED_READS.add(task)
    task.add_done_callback(_consume_workspace_observation_read)


def _consume_workspace_observation_read(task: asyncio.Task[Any]) -> None:
    _WORKSPACE_OBSERVATION_ABANDONED_READS.discard(task)
    with contextlib.suppress(BaseException):
        task.exception()


def _discard_completed_workspace_observation_reads() -> None:
    for task in tuple(_WORKSPACE_OBSERVATION_ABANDONED_READS):
        if task.done():
            _consume_workspace_observation_read(task)


async def publish_workspace_observation_transition(
    *,
    session_store: SessionStore,
    event_writer: RuntimeEventWriter,
    session: Session,
    previous: WorkspaceObservationLifecycle | None,
    current: WorkspaceObservationLifecycle | None,
    phase: str,
    terminal_status: WorkspaceObservationTerminalStatus | None = None,
    terminal_detail_code: str | None = None,
    terminal_artifacts: tuple[WorkspaceObservationArtifact, ...] | None = None,
    events: tuple[Event, ...] = (),
    intent_admission: _WorkspaceObservationIntentAdmission | None = None,
) -> tuple[Event, ...]:
    """CAS one lifecycle transition and idempotently fan out its events."""

    if previous is None and current is None:
        raise ValueError("A workspace observation transition requires previous or current state.")
    owner = current or previous
    assert owner is not None
    if previous is None:
        if (
            type(intent_admission) is not _WorkspaceObservationIntentAdmission
            or current is None
            or not intent_admission.matches(current)
        ):
            raise ValueError(
                "Initial workspace observation state requires matching runtime admission."
            )
    elif intent_admission is not None:
        raise ValueError("Workspace observation admission applies only to initial intent.")
    if previous is not None and current is not None:
        if previous.authority_tuple() != current.authority_tuple():
            raise ValueError("Workspace observation authority cannot change across a transition.")
        if previous.window_id != current.window_id:
            raise ValueError("Workspace observation transition changed its window id.")
    if terminal_status is None and current is None:
        raise ValueError("Removing active workspace observation state requires terminal status.")
    if terminal_status is not None and current is not None:
        raise ValueError("A terminal workspace observation transition must remove active state.")
    if terminal_artifacts is not None and terminal_status is None:
        raise ValueError("Terminal artifact state requires a terminal transition.")
    publication_artifacts = owner.artifacts
    if terminal_artifacts is not None:
        if type(terminal_artifacts) is not tuple or any(
            type(artifact) is not WorkspaceObservationArtifact for artifact in terminal_artifacts
        ):
            raise TypeError(
                "terminal_artifacts must be a tuple of WorkspaceObservationArtifact values."
            )
        owner_by_kind = {artifact.evidence_kind: artifact for artifact in owner.artifacts}
        terminal_by_kind = {artifact.evidence_kind: artifact for artifact in terminal_artifacts}
        if set(owner_by_kind) != set(terminal_by_kind) or any(
            owner_by_kind[kind].model_dump(exclude={"state"})
            != terminal_by_kind[kind].model_dump(exclude={"state"})
            for kind in owner_by_kind
        ):
            raise ValueError(
                "Terminal artifact state conflicts with the active lifecycle identity."
            )
        publication_artifacts = terminal_artifacts
    if terminal_detail_code is not None:
        terminal_detail_code = require_clean_nonblank(
            terminal_detail_code,
            "terminal_detail_code",
        )
    validate_workspace_observation_transition(
        previous=previous,
        current=current,
        phase=phase,
        terminal_status=terminal_status,
        terminal_artifacts=terminal_artifacts,
    )

    prepared_events = tuple(event_writer.prepare(event) for event in events)
    intent: dict[str, Any] = {
        "schema_version": WORKSPACE_OBSERVATION_SCHEMA_VERSION,
        "window_id": owner.window_id,
        "phase": phase,
        "source_run_epoch": owner.source_run_epoch,
        "binding_generation_id": owner.binding_generation_id,
        "workspace_id": owner.workspace_id,
        "observer": owner.observer,
        "observer_authority": owner.observer_authority,
        "artifact_store_id": owner.artifact_store_id,
        "agent_name": owner.agent_name,
        "environment_name": owner.environment_name,
        "tool_name": owner.tool_name,
        "tool_call_id": owner.tool_call_id,
        "model_step_id": owner.model_step_id,
        "model_attempt_id": owner.model_attempt_id,
        "tool_round_id": owner.tool_round_id,
        "model_step": owner.model_step,
        "before_observation_id": owner.before_observation_id,
        "tool_outcome_event_id": owner.tool_outcome_event_id,
        "tool_outcome_event_digest": owner.tool_outcome_event_digest,
        "after_observation_id": owner.after_observation_id,
        "mutation_event_id": owner.mutation_event_id,
        "mutation_event_digest": owner.mutation_event_digest,
        "artifacts": [artifact.model_dump(mode="json") for artifact in publication_artifacts],
        "terminal_status": None if terminal_status is None else terminal_status.value,
        "terminal_detail_code": terminal_detail_code,
    }

    checkpoint = await await_workspace_observation_store_read(
        lambda: session_store.load_checkpoint(session.id),
        operation="Workspace observation checkpoint read",
    )
    raw_schema_version = (
        None if checkpoint is None else checkpoint.get(CHECKPOINT_SCHEMA_VERSION_KEY)
    )
    if raw_schema_version is not None and (
        type(raw_schema_version) is not int
        or not 1 <= raw_schema_version <= CURRENT_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("Workspace observation checkpoint schema is unsupported.")
    active = workspace_observations_from_checkpoint(checkpoint)
    durable_previous = active.get(owner.window_id)
    already_applied = False
    if previous is None:
        if durable_previous is not None:
            if current == durable_previous:
                already_applied = True
            else:
                raise RuntimeError("Workspace observation intent already has conflicting state.")
    elif durable_previous != previous:
        if durable_previous == current:
            already_applied = True
        else:
            raise RuntimeError("Workspace observation state changed before its transition.")

    source = dict(active)
    if already_applied:
        if previous is None:
            source.pop(owner.window_id, None)
        else:
            source[owner.window_id] = previous
        updated = dict(active)
    else:
        updated = dict(active)
        if current is None:
            updated.pop(owner.window_id, None)
        else:
            updated[owner.window_id] = current
    source_value = workspace_observation_checkpoint_value(source) if source else None
    target_value = workspace_observation_checkpoint_value(updated) if updated else None
    intent.update(
        {
            "source_observations": (
                {}
                if source_value is None
                else copy_durable_json_object(
                    source_value,
                    "workspace_observation_source",
                )
            ),
            "source_root_digest": runtime_publication_checkpoint_value_digest(source_value),
            "target_root_digest": runtime_publication_checkpoint_value_digest(target_value),
        }
    )
    expected_digest = (
        None if source_value is None else runtime_publication_checkpoint_value_digest(source_value)
    )
    if updated:
        operation = RuntimePublicationCheckpointOperation(
            key=WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY,
            expected_value_digest=expected_digest,
            action="set",
            value=target_value,
        )
    else:
        if expected_digest is None:
            raise RuntimeError(
                "Workspace observation terminal transition lost its checkpoint root."
            )
        operation = RuntimePublicationCheckpointOperation(
            key=WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY,
            expected_value_digest=expected_digest,
            action="delete",
        )

    schema_operation = RuntimePublicationCheckpointOperation(
        key=CHECKPOINT_SCHEMA_VERSION_KEY,
        expected_value_digest=(
            None
            if raw_schema_version is None
            else runtime_publication_checkpoint_value_digest(raw_schema_version)
        ),
        action="set",
        value=CURRENT_CHECKPOINT_SCHEMA_VERSION,
    )
    request = RuntimePublicationRequest(
        publication_id=_phase_publication_id(owner.window_id, phase),
        kind="workspace-observation",
        interaction_id=owner.interaction_id,
        intent=intent,
        mutation=RuntimePublicationMutation(operations=(schema_operation, operation)),
        transcript_messages=(),
        events=prepared_events,
    )
    expected_request_digests = frozenset({runtime_publication_request_digest(session.id, request)})
    expected_statuses = {
        SessionStatus.PENDING,
        SessionStatus.RUNNING,
        SessionStatus.INTERRUPTING,
        SessionStatus.INTERRUPTED,
        SessionStatus.FAILED,
        SessionStatus.COMPLETED,
    }

    async def publish_exact() -> RuntimePublicationResult:
        return await session_store.publish_runtime_publication(
            session.id,
            request=request,
            expected_statuses=expected_statuses,
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=None,
        )

    publication_task = asyncio.create_task(capture_awaitable_outcome(publish_exact))
    outcome = await await_shielded_task_outcome(publication_task)
    cancellation = outcome.cancellation
    cancellation_requests_consumed = outcome.cancellation_requests_consumed
    result, publication_error = _workspace_observation_captured_task_outcome(
        outcome,
        operation="Workspace observation publication",
    )
    # The custom-store task and its wait outcome both retain the raw returned
    # object.  Sever those references before any bounded validation failure is
    # propagated so traceback-local capture cannot recover rejected extension
    # output from this frame.
    del outcome
    del publication_task
    if publication_error is None:
        validated_result, validation_error = _validate_workspace_observation_publication_result(
            result,
            session=session,
            request=request,
            prepared_events=prepared_events,
            expected_statuses=expected_statuses,
            expected_request_digests=expected_request_digests,
        )
        publication_error = validation_error
        del validated_result
    del result

    preserved_fatal_error = (
        publication_error
        if publication_error is not None and not isinstance(publication_error, Exception)
        else None
    )
    if publication_error is not None:

        async def reconcile() -> RuntimePublicationResult | None:
            receipt = await await_workspace_observation_store_read(
                lambda: session_store.load_runtime_publication_receipt(
                    session.id,
                    request.publication_id,
                ),
                operation="Workspace observation receipt read",
            )
            receipt_exists = receipt is not None
            del receipt
            if not receipt_exists:
                return None
            replayed = await publish_exact()
            validated_replay, replay_error = _validate_workspace_observation_publication_result(
                replayed,
                session=session,
                request=request,
                prepared_events=prepared_events,
                expected_statuses=expected_statuses,
                expected_request_digests=expected_request_digests,
            )
            del replayed
            if replay_error is not None:
                raise replay_error
            if validated_replay is None:  # pragma: no cover - paired invariant
                raise RuntimeError(
                    "Workspace observation publication returned invalid acknowledgement."
                )
            return validated_replay

        reconciliation_outcome = await await_shielded_task_outcome(
            asyncio.create_task(capture_awaitable_outcome(reconcile)),
            cancellation=cancellation,
        )
        cancellation = reconciliation_outcome.cancellation
        cancellation_requests_consumed += reconciliation_outcome.cancellation_requests_consumed
        reconciliation_result, reconciliation_error = _workspace_observation_captured_task_outcome(
            reconciliation_outcome,
            operation="Workspace observation publication reconciliation",
        )
        if reconciliation_error is not None:
            combined_failure = _workspace_observation_failure_group(
                "Workspace observation publication and reconciliation failed.",
                publication_error,
                reconciliation_error,
            )
            if cancellation is not None:
                if not isinstance(combined_failure, Exception):
                    restore_workspace_observation_cancellation_requests(
                        cancellation_requests_consumed
                    )
                    raise _workspace_observation_failure_group(
                        "Workspace observation publication failed with caller cancellation.",
                        publication_error,
                        reconciliation_error,
                        cancellation,
                    ) from None
                add_exception_note_safely(
                    cancellation,
                    "Workspace observation publication reconciliation failed.",
                )
                restore_workspace_observation_cancellation_requests(cancellation_requests_consumed)
                raise cancellation from combined_failure
            raise combined_failure
        if reconciliation_result is None:
            if cancellation is not None and preserved_fatal_error is not None:
                restore_workspace_observation_cancellation_requests(cancellation_requests_consumed)
                raise BaseExceptionGroup(
                    "Workspace observation publication remained ambiguous after control signals.",
                    [preserved_fatal_error, cancellation],
                ) from None
            if cancellation is not None:
                restore_workspace_observation_cancellation_requests(cancellation_requests_consumed)
                raise cancellation from publication_error
            raise publication_error

    if prepared_events:
        fan_out_outcome = await await_shielded_task_outcome(
            asyncio.create_task(
                capture_awaitable_outcome(
                    lambda: event_writer.fan_out_persisted(list(prepared_events))
                )
            ),
            cancellation=cancellation,
        )
        cancellation = fan_out_outcome.cancellation
        cancellation_requests_consumed += fan_out_outcome.cancellation_requests_consumed
        _fan_out_result, fan_out_error = _workspace_observation_captured_task_outcome(
            fan_out_outcome,
            operation="Workspace observation event fan-out",
        )
        if fan_out_error is not None:
            if preserved_fatal_error is not None:
                failures = [preserved_fatal_error]
                if cancellation is not None:
                    failures.append(cancellation)
                failures.append(fan_out_error)
                restore_workspace_observation_cancellation_requests(cancellation_requests_consumed)
                raise BaseExceptionGroup(
                    "Workspace observation publication and event fan-out failed.",
                    failures,
                ) from None
            if cancellation is not None:
                if not isinstance(fan_out_error, Exception):
                    restore_workspace_observation_cancellation_requests(
                        cancellation_requests_consumed
                    )
                    raise BaseExceptionGroup(
                        "Workspace observation event fan-out failed with caller cancellation.",
                        [fan_out_error, cancellation],
                    ) from None
                add_exception_note_safely(
                    cancellation,
                    "Workspace observation event fan-out failed.",
                )
                restore_workspace_observation_cancellation_requests(cancellation_requests_consumed)
                raise cancellation from fan_out_error
            raise fan_out_error
    if cancellation is not None and preserved_fatal_error is not None:
        restore_workspace_observation_cancellation_requests(cancellation_requests_consumed)
        raise BaseExceptionGroup(
            "Workspace observation publication completed with control signals.",
            [preserved_fatal_error, cancellation],
        ) from None
    if preserved_fatal_error is not None:
        raise preserved_fatal_error
    if cancellation is not None:
        restore_workspace_observation_cancellation_requests(cancellation_requests_consumed)
        raise cancellation
    return tuple(copy_event(event) for event in prepared_events)


def _copy_workspace_observation_publication_receipt(
    receipt: RuntimePublicationReceipt,
) -> RuntimePublicationReceipt:
    if type(receipt) is not RuntimePublicationReceipt:
        raise TypeError("Workspace observation publication returned an invalid receipt.")
    detached = RuntimePublicationReceipt.model_validate(
        {
            field_name: getattr(receipt, field_name)
            for field_name in RuntimePublicationReceipt.model_fields
        }
    )
    digest_payload = detached.model_dump(mode="json", exclude={"publication_digest"})
    if (
        hashlib.sha256(
            canonical_durable_json_bytes(
                digest_payload,
                "workspace_observation_publication_receipt",
            )
        ).hexdigest()
        != detached.publication_digest
    ):
        raise ValueError("Workspace observation publication receipt digest is invalid.")
    return detached


def _copy_workspace_observation_publication_result(
    result: object,
) -> tuple[RuntimePublicationResult | None, BaseException | None]:
    try:
        if type(result) is not RuntimePublicationResult:
            raise TypeError
        detached = RuntimePublicationResult(
            session=copy_session(result.session),
            receipt=_copy_workspace_observation_publication_receipt(result.receipt),
            replayed=result.replayed,
        )
    except BaseException as error:
        if exception_tree_contains(error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            preserved_error = error
            del result
            return None, preserved_error
        del result
        return (
            None,
            RuntimeError("Workspace observation publication returned invalid acknowledgement."),
        )
    del result
    return detached, None


def _validate_workspace_observation_publication_result(
    result: object,
    *,
    session: Session,
    request: RuntimePublicationRequest,
    prepared_events: tuple[Event, ...],
    expected_statuses: set[SessionStatus],
    expected_request_digests: frozenset[str],
) -> tuple[RuntimePublicationResult | None, BaseException | None]:
    detached, copy_error = _copy_workspace_observation_publication_result(result)
    del result
    if copy_error is not None:
        return None, copy_error
    if detached is None:  # pragma: no cover - paired invariant
        return (
            None,
            RuntimeError("Workspace observation publication returned invalid acknowledgement."),
        )
    receipt = detached.receipt
    expected_transcript_digest = hashlib.sha256(
        canonical_durable_json_bytes([], "workspace_observation_transcript")
    ).hexdigest()
    expected_events_digest = hashlib.sha256(
        canonical_durable_json_bytes(
            [event.model_dump(mode="json") for event in prepared_events],
            "workspace_observation_events",
        )
    ).hexdigest()
    if (
        detached.session.id != session.id
        or detached.session.run_epoch != session.run_epoch
        or receipt.session_id != session.id
        or receipt.publication_id != request.publication_id
        or receipt.kind != request.kind
        or receipt.interaction_id != request.interaction_id
        or receipt.intent != request.intent
        or receipt.request_digest not in expected_request_digests
        or receipt.source_status not in expected_statuses
        or receipt.source_run_epoch != session.run_epoch
        or receipt.transcript_digest != expected_transcript_digest
        or receipt.events_digest != expected_events_digest
        or receipt.appended_event_ids != tuple(event.id for event in prepared_events)
        or receipt.referenced_events
        or receipt.transcript_end_cursor != receipt.transcript_start_cursor
    ):
        del receipt
        del detached
        return (
            None,
            RuntimeError(
                "Workspace observation publication acknowledgement conflicts with its request."
            ),
        )
    return detached, None


def copy_workspace_observation_lifecycle(
    value: WorkspaceObservationLifecycle,
) -> WorkspaceObservationLifecycle:
    if type(value) is not WorkspaceObservationLifecycle:
        raise TypeError("value must be a WorkspaceObservationLifecycle.")
    return WorkspaceObservationLifecycle.model_validate(
        copy_durable_json_object(value.model_dump(mode="json"), "workspace_observation")
    )
