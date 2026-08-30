"""Durable authority records for task-backed subagent execution."""

from __future__ import annotations

import asyncio
from hashlib import sha256
from typing import Any, Literal
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._exception_groups import exception_group_children
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
)
from cayu.core.messages import Message
from cayu.runtime._child_session_identity import ChildSessionKind, generate_child_session_id
from cayu.runtime.execution_profiles import ExecutionProfileIdentity
from cayu.runtime.sessions import RunRequest, copy_run_request
from cayu.vaults import SecretRedactor

DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY = "durable_subagent_submissions"
DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY = "durable_subagent_submission_seeds"
DURABLE_SUBAGENT_SEED_RECORD_TYPE = "cayu.durable-subagent-submission-seed"
DURABLE_SUBAGENT_SEED_SCHEMA_VERSION = 1
DURABLE_SUBAGENT_INTENT_RECORD_TYPE = "cayu.durable-subagent-submission"
DURABLE_SUBAGENT_INTENT_SCHEMA_VERSION = 2
DURABLE_SUBAGENT_RECEIPT_RECORD_TYPE = "cayu.durable-subagent-submission-receipt"
DURABLE_SUBAGENT_RECEIPT_SCHEMA_VERSION = 1

_DURABLE_SUBAGENT_UNSETTLED_AUTHORITY = object()
_DURABLE_SUBAGENT_COMMITTED_CANCELLATION_AUTHORITY = object()
_DURABLE_SUBAGENT_UNSETTLED_CANCELLATION_AUTHORITY = object()
_DURABLE_SUBAGENT_WORKER_INCOMPATIBLE_AUTHORITY = object()
_DURABLE_SUBAGENT_AUTHORITY_REJECTED_AUTHORITY = object()
_DURABLE_SUBAGENT_PREPARATION_REJECTED_AUTHORITY = object()


class _DurableSubagentPreparationRejected(RuntimeError):
    """Private proof that child preparation failed before child publication."""

    def __init__(self, *, _authority: object) -> None:
        if _authority is not _DURABLE_SUBAGENT_PREPARATION_REJECTED_AUTHORITY:
            raise TypeError("Durable subagent preparation rejection is runtime-owned.")
        super().__init__("Durable subagent preparation was rejected before dispatch.")


_DURABLE_SUBAGENT_PREPARATION_REJECTED_PROVENANCE: WeakKeyDictionary[
    _DurableSubagentPreparationRejected,
    bool,
] = WeakKeyDictionary()


def durable_subagent_preparation_rejected() -> RuntimeError:
    """Create authenticated permanent pre-dispatch rejection evidence."""

    signal = _DurableSubagentPreparationRejected(
        _authority=_DURABLE_SUBAGENT_PREPARATION_REJECTED_AUTHORITY,
    )
    _DURABLE_SUBAGENT_PREPARATION_REJECTED_PROVENANCE[signal] = True
    return signal


def is_durable_subagent_preparation_rejected(error: BaseException) -> bool:
    """Recognize only runtime-created permanent preparation rejection."""

    return (
        type(error) is _DurableSubagentPreparationRejected
        and error in _DURABLE_SUBAGENT_PREPARATION_REJECTED_PROVENANCE
    )


class _DurableSubagentWorkerIncompatible(RuntimeError):
    """Private signal that this worker cannot admit an otherwise valid child."""

    def __init__(self, *, _authority: object) -> None:
        if _authority is not _DURABLE_SUBAGENT_WORKER_INCOMPATIBLE_AUTHORITY:
            raise TypeError("Durable subagent worker compatibility is runtime-owned.")
        super().__init__(
            "Durable subagent worker does not provide the frozen child execution profile."
        )


_DURABLE_SUBAGENT_WORKER_INCOMPATIBLE_PROVENANCE: WeakKeyDictionary[
    _DurableSubagentWorkerIncompatible,
    bool,
] = WeakKeyDictionary()


def durable_subagent_worker_incompatible() -> RuntimeError:
    """Create authenticated evidence that another worker must claim the child."""

    signal = _DurableSubagentWorkerIncompatible(
        _authority=_DURABLE_SUBAGENT_WORKER_INCOMPATIBLE_AUTHORITY,
    )
    _DURABLE_SUBAGENT_WORKER_INCOMPATIBLE_PROVENANCE[signal] = True
    return signal


def is_durable_subagent_worker_incompatible(error: BaseException) -> bool:
    """Recognize only runtime-created worker-incompatibility evidence."""

    return (
        type(error) is _DurableSubagentWorkerIncompatible
        and error in _DURABLE_SUBAGENT_WORKER_INCOMPATIBLE_PROVENANCE
    )


class _DurableSubagentAuthorityRejected(RuntimeError):
    """Private signal that prepared child authority is permanently invalid."""

    def __init__(self, *, _authority: object) -> None:
        if _authority is not _DURABLE_SUBAGENT_AUTHORITY_REJECTED_AUTHORITY:
            raise TypeError("Durable subagent authority rejection is runtime-owned.")
        super().__init__("Prepared durable subagent authority is invalid.")


_DURABLE_SUBAGENT_AUTHORITY_REJECTED_PROVENANCE: WeakKeyDictionary[
    _DurableSubagentAuthorityRejected,
    bool,
] = WeakKeyDictionary()


def durable_subagent_authority_rejected() -> RuntimeError:
    """Create authenticated evidence that a prepared child must not be retried."""

    signal = _DurableSubagentAuthorityRejected(
        _authority=_DURABLE_SUBAGENT_AUTHORITY_REJECTED_AUTHORITY,
    )
    _DURABLE_SUBAGENT_AUTHORITY_REJECTED_PROVENANCE[signal] = True
    return signal


def is_durable_subagent_authority_rejected(error: BaseException) -> bool:
    """Recognize only runtime-created permanent child-authority rejection."""

    return (
        type(error) is _DurableSubagentAuthorityRejected
        and error in _DURABLE_SUBAGENT_AUTHORITY_REJECTED_PROVENANCE
    )


class _DurableSubagentSubmissionUnsettled(RuntimeError):
    """Runtime-owned signal that a durable spawn needs exact reconciliation."""

    def __init__(
        self,
        *,
        parent_session_id: str,
        tool_name: str,
        idempotency_key: str,
        failure: Exception,
        _authority: object,
    ) -> None:
        if _authority is not _DURABLE_SUBAGENT_UNSETTLED_AUTHORITY:
            raise TypeError("Durable subagent unsettled state is runtime-owned.")
        self.parent_session_id = require_durable_clean_nonblank(
            parent_session_id,
            "parent_session_id",
        )
        self.tool_name = require_durable_clean_nonblank(tool_name, "tool_name")
        self.idempotency_key = require_durable_clean_nonblank(
            idempotency_key,
            "idempotency_key",
        )
        if not isinstance(failure, Exception):
            raise TypeError("Durable subagent unsettled failure must be an Exception.")
        self.failure = failure
        super().__init__(
            "Durable subagent submission outcome is unsettled; the parent tool "
            "round remains recoverable."
        )


_DURABLE_SUBAGENT_UNSETTLED_PROVENANCE: WeakKeyDictionary[
    _DurableSubagentSubmissionUnsettled,
    tuple[str, str, str],
] = WeakKeyDictionary()


class _DurableSubagentSubmissionCommittedDuringCancellation(RuntimeError):
    """Private handoff from durable publication to the timeout-owning boundary."""

    def __init__(self, *, _authority: object) -> None:
        if _authority is not _DURABLE_SUBAGENT_COMMITTED_CANCELLATION_AUTHORITY:
            raise TypeError("Durable subagent cancellation settlement is runtime-owned.")
        super().__init__("Durable subagent submission committed while cancellation was pending.")


_DURABLE_SUBAGENT_COMMITTED_CANCELLATION_PROVENANCE: WeakKeyDictionary[
    _DurableSubagentSubmissionCommittedDuringCancellation,
    tuple[
        Any,
        asyncio.CancelledError,
        asyncio.CancelledError | None,
        int,
    ],
] = WeakKeyDictionary()


class _DurableSubagentSubmissionUnsettledDuringCancellation(RuntimeError):
    """Private handoff for ambiguous publication observed during cancellation."""

    def __init__(self, *, _authority: object) -> None:
        if _authority is not _DURABLE_SUBAGENT_UNSETTLED_CANCELLATION_AUTHORITY:
            raise TypeError("Durable subagent cancellation settlement is runtime-owned.")
        super().__init__("Durable subagent submission remained unsettled during cancellation.")


_DURABLE_SUBAGENT_UNSETTLED_CANCELLATION_PROVENANCE: WeakKeyDictionary[
    _DurableSubagentSubmissionUnsettledDuringCancellation,
    tuple[
        _DurableSubagentSubmissionUnsettled,
        asyncio.CancelledError,
        asyncio.CancelledError | None,
        int,
    ],
] = WeakKeyDictionary()


def durable_subagent_submission_committed_during_cancellation(
    *,
    result: Any,
    cancellation: asyncio.CancelledError,
    subsequent_cancellation: asyncio.CancelledError | None,
    cancellation_requests_consumed: int,
) -> RuntimeError:
    """Carry a committed result to the boundary that owns cancellation classification."""

    if not isinstance(cancellation, asyncio.CancelledError):
        raise TypeError("Durable subagent cancellation settlement requires CancelledError.")
    if subsequent_cancellation is not None and not isinstance(
        subsequent_cancellation,
        asyncio.CancelledError,
    ):
        raise TypeError("Subsequent durable subagent cancellation must be CancelledError.")
    if type(cancellation_requests_consumed) is not int or cancellation_requests_consumed < 1:
        raise ValueError(
            "Durable subagent cancellation settlement requires consumed cancellation evidence."
        )
    signal = _DurableSubagentSubmissionCommittedDuringCancellation(
        _authority=_DURABLE_SUBAGENT_COMMITTED_CANCELLATION_AUTHORITY,
    )
    _DURABLE_SUBAGENT_COMMITTED_CANCELLATION_PROVENANCE[signal] = (
        result,
        cancellation,
        subsequent_cancellation,
        cancellation_requests_consumed,
    )
    return signal


def durable_subagent_committed_cancellation_outcome(
    error: BaseException,
) -> tuple[Any, asyncio.CancelledError, asyncio.CancelledError | None, int] | None:
    """Return only the outcome attached to an exact runtime-created private signal."""

    if type(error) is not _DurableSubagentSubmissionCommittedDuringCancellation:
        return None
    try:
        return _DURABLE_SUBAGENT_COMMITTED_CANCELLATION_PROVENANCE.get(error)
    except (TypeError, ValueError):
        return None


def durable_subagent_submission_unsettled_during_cancellation(
    *,
    unsettled: BaseException,
    cancellation: asyncio.CancelledError,
    subsequent_cancellation: asyncio.CancelledError | None,
    cancellation_requests_consumed: int,
) -> RuntimeError:
    """Carry an unsettled result to the boundary that owns timeout classification."""

    if type(unsettled) is not _DurableSubagentSubmissionUnsettled:
        raise TypeError("Durable subagent cancellation requires an exact unsettled signal.")
    try:
        provenance = _DURABLE_SUBAGENT_UNSETTLED_PROVENANCE.get(unsettled)
    except (TypeError, ValueError):
        provenance = None
    if provenance is None:
        raise TypeError("Durable subagent cancellation requires runtime-owned unsettled evidence.")
    if not isinstance(cancellation, asyncio.CancelledError):
        raise TypeError("Durable subagent cancellation settlement requires CancelledError.")
    if subsequent_cancellation is not None and not isinstance(
        subsequent_cancellation,
        asyncio.CancelledError,
    ):
        raise TypeError("Subsequent durable subagent cancellation must be CancelledError.")
    if type(cancellation_requests_consumed) is not int or cancellation_requests_consumed < 1:
        raise ValueError(
            "Durable subagent cancellation settlement requires consumed cancellation evidence."
        )
    signal = _DurableSubagentSubmissionUnsettledDuringCancellation(
        _authority=_DURABLE_SUBAGENT_UNSETTLED_CANCELLATION_AUTHORITY,
    )
    _DURABLE_SUBAGENT_UNSETTLED_CANCELLATION_PROVENANCE[signal] = (
        unsettled,
        cancellation,
        subsequent_cancellation,
        cancellation_requests_consumed,
    )
    return signal


def durable_subagent_unsettled_cancellation_outcome(
    error: BaseException,
) -> (
    tuple[
        RuntimeError,
        asyncio.CancelledError,
        asyncio.CancelledError | None,
        int,
    ]
    | None
):
    """Return only runtime-owned unsettled cancellation evidence."""

    if type(error) is not _DurableSubagentSubmissionUnsettledDuringCancellation:
        return None
    try:
        return _DURABLE_SUBAGENT_UNSETTLED_CANCELLATION_PROVENANCE.get(error)
    except (TypeError, ValueError):
        return None


def durable_subagent_submission_unsettled(
    *,
    parent_session_id: str,
    tool_name: str,
    idempotency_key: str,
    failure: Exception,
) -> RuntimeError:
    """Create the private control signal after durable seed ownership exists."""

    signal = _DurableSubagentSubmissionUnsettled(
        parent_session_id=parent_session_id,
        tool_name=tool_name,
        idempotency_key=idempotency_key,
        failure=failure,
        _authority=_DURABLE_SUBAGENT_UNSETTLED_AUTHORITY,
    )
    _DURABLE_SUBAGENT_UNSETTLED_PROVENANCE[signal] = (
        signal.parent_session_id,
        signal.tool_name,
        signal.idempotency_key,
    )
    return signal


def is_durable_subagent_submission_unsettled(
    error: BaseException,
    *,
    parent_session_id: str | None = None,
    tool_name: str | None = None,
    idempotency_key: str | None = None,
) -> bool:
    """Recognize only exact runtime-created unsettled signals, including groups."""

    pending = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if type(candidate) is _DurableSubagentSubmissionUnsettled:
            signal = candidate
            try:
                provenance = _DURABLE_SUBAGENT_UNSETTLED_PROVENANCE.get(signal)
            except (TypeError, ValueError):
                continue
            if provenance is None:
                continue
            candidate_parent, candidate_tool, candidate_key = provenance
            if parent_session_id is not None and candidate_parent != parent_session_id:
                continue
            if tool_name is not None and candidate_tool != tool_name:
                continue
            if idempotency_key is not None and candidate_key != idempotency_key:
                continue
            return True
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                pending.extend(
                    child for child in reversed(children) if isinstance(child, BaseException)
                )
    return False


def _require_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _durable_run_request(value: object) -> RunRequest:
    if type(value) is RunRequest:
        return copy_run_request(value)
    if type(value) is not dict:
        raise TypeError("Durable subagent request must be a RunRequest or object.")
    copied = dict(value)
    raw_messages = copied.get("messages")
    if type(raw_messages) is not list:
        raise ValueError("Durable subagent request messages must be an array.")
    copied["messages"] = [
        message if type(message) is Message else Message.model_validate(message)
        for message in raw_messages
    ]
    return RunRequest.model_validate(copied)


def durable_subagent_request_sha256(request: RunRequest) -> str:
    request = copy_run_request(request)
    if request.loop_policies:
        raise ValueError("Durable subagent requests cannot carry process-local loop policies.")
    return sha256(
        canonical_durable_json_bytes(
            request.model_dump(mode="json", warnings=False),
            "durable_subagent.request",
        )
    ).hexdigest()


def durable_subagent_dispatch_id(*, parent_session_id: str, idempotency_key: str) -> str:
    material = [
        "cayu-durable-subagent-dispatch-v1",
        require_durable_clean_nonblank(parent_session_id, "parent_session_id"),
        require_durable_clean_nonblank(idempotency_key, "idempotency_key"),
    ]
    return (
        "cayu-subagent-dispatch:"
        + sha256(
            canonical_durable_json_bytes(material, "durable_subagent.dispatch_identity")
        ).hexdigest()
    )


def durable_dispatch_queue_task_id(*, task_type: str, dispatch_id: str) -> str:
    material = {
        "schema": "cayu.queued-dispatch-task.v1",
        "task_type": require_durable_clean_nonblank(task_type, "task_type"),
        "dispatch_id": require_durable_clean_nonblank(dispatch_id, "dispatch_id"),
    }
    return (
        "cayu-dispatch-"
        + sha256(
            canonical_durable_json_bytes(material, "durable_subagent.queue_task_identity")
        ).hexdigest()
    )


def durable_subagent_interaction_id(*, child_session_id: str, idempotency_key: str) -> str:
    material = [
        "cayu-durable-subagent-interaction-v1",
        require_durable_clean_nonblank(child_session_id, "child_session_id"),
        require_durable_clean_nonblank(idempotency_key, "idempotency_key"),
    ]
    return (
        "cayu-subagent-interaction:"
        + sha256(
            canonical_durable_json_bytes(material, "durable_subagent.interaction_identity")
        ).hexdigest()
    )


def durable_subagent_interaction_event_id(interaction_id: str) -> str:
    interaction_id = require_durable_clean_nonblank(interaction_id, "interaction_id")
    return "cayu-subagent-interaction-started:" + sha256(interaction_id.encode()).hexdigest()


class DurableSubagentAuthority(BaseModel):
    """Exact identity shared by preparation seeds and finalized submissions."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    parent_session_id: str
    parent_session_instance_fingerprint: str
    parent_task_id: str | None
    parent_run_epoch: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    parent_execution_profile_fingerprint: str
    causal_budget_id: str
    model_step_id: str
    model_attempt_id: str
    tool_round_id: str
    tool_call_id: str
    tool_name: str
    idempotency_key: str
    effective_arguments_sha256: str
    agent_alias: str
    agent_name: str
    environment_name: str | None
    spawn_fingerprint: str
    child_session_id: str
    dispatch_id: str
    queue_task_id: str
    queue_task_type: str
    interaction_id: str
    interaction_started_event_id: str
    request_sha256: str
    request: RunRequest

    @field_validator(
        "parent_session_id",
        "parent_task_id",
        "causal_budget_id",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "tool_call_id",
        "tool_name",
        "idempotency_key",
        "agent_alias",
        "agent_name",
        "environment_name",
        "spawn_fingerprint",
        "child_session_id",
        "dispatch_id",
        "queue_task_id",
        "queue_task_type",
        "interaction_id",
        "interaction_started_event_id",
    )
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator(
        "parent_session_instance_fingerprint",
        "parent_execution_profile_fingerprint",
        "effective_arguments_sha256",
        "request_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str, info) -> str:
        return _require_sha256(value, info.field_name)

    @field_validator("request", mode="before")
    @classmethod
    def copy_request(cls, value: object) -> RunRequest:
        return _durable_run_request(value)

    @model_validator(mode="after")
    def validate_authority(self) -> DurableSubagentAuthority:
        if self.request.session_id != self.child_session_id:
            raise ValueError("Durable subagent authority targets another child session.")
        if self.request.parent_session_id != self.parent_session_id:
            raise ValueError("Durable subagent authority has conflicting parent linkage.")
        if self.request.causal_budget_id != self.causal_budget_id:
            raise ValueError("Durable subagent authority has conflicting causal-budget linkage.")
        if self.request.agent_name != self.agent_name:
            raise ValueError("Durable subagent authority has conflicting agent linkage.")
        if self.request.environment_name != self.environment_name:
            raise ValueError("Durable subagent authority has conflicting environment linkage.")
        if self.request.task_id is not None or self.request.task_worker_id is not None:
            raise ValueError("Durable subagent child execution cannot reuse the parent task.")
        if self.request.loop_policies:
            raise ValueError("Durable subagent requests cannot carry loop policies.")
        expected_child = generate_child_session_id(
            kind=ChildSessionKind.SUBAGENT,
            parent_session_id=self.parent_session_id,
            logical_spawn_id=self.idempotency_key,
        )
        if self.child_session_id != expected_child:
            raise ValueError("Durable subagent child identity conflicts with its spawn.")
        if self.dispatch_id != durable_subagent_dispatch_id(
            parent_session_id=self.parent_session_id,
            idempotency_key=self.idempotency_key,
        ):
            raise ValueError("Durable subagent dispatch identity conflicts with its spawn.")
        if self.queue_task_id != durable_dispatch_queue_task_id(
            task_type=self.queue_task_type,
            dispatch_id=self.dispatch_id,
        ):
            raise ValueError("Durable subagent queue identity conflicts with its spawn.")
        if self.interaction_id != durable_subagent_interaction_id(
            child_session_id=self.child_session_id,
            idempotency_key=self.idempotency_key,
        ):
            raise ValueError("Durable subagent interaction conflicts with its spawn.")
        if self.interaction_started_event_id != durable_subagent_interaction_event_id(
            self.interaction_id
        ):
            raise ValueError("Durable subagent interaction event identity conflicts.")
        if self.request_sha256 != durable_subagent_request_sha256(self.request):
            raise ValueError("Durable subagent request digest conflicts with its request.")
        return self


class _DurableSubagentAuthorityRecord(BaseModel):
    """Common typed view over a record's single embedded authority value."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    authority: DurableSubagentAuthority

    @field_validator("authority", mode="before")
    @classmethod
    def copy_authority(cls, value: object) -> DurableSubagentAuthority:
        if type(value) is DurableSubagentAuthority:
            value = value.model_dump(mode="json", warnings=False)
        return DurableSubagentAuthority.model_validate(value)

    @property
    def parent_session_id(self) -> str:
        return self.authority.parent_session_id

    @property
    def parent_session_instance_fingerprint(self) -> str:
        return self.authority.parent_session_instance_fingerprint

    @property
    def parent_task_id(self) -> str | None:
        return self.authority.parent_task_id

    @property
    def parent_run_epoch(self) -> int:
        return self.authority.parent_run_epoch

    @property
    def parent_execution_profile_fingerprint(self) -> str:
        return self.authority.parent_execution_profile_fingerprint

    @property
    def causal_budget_id(self) -> str:
        return self.authority.causal_budget_id

    @property
    def model_step_id(self) -> str:
        return self.authority.model_step_id

    @property
    def model_attempt_id(self) -> str:
        return self.authority.model_attempt_id

    @property
    def tool_round_id(self) -> str:
        return self.authority.tool_round_id

    @property
    def tool_call_id(self) -> str:
        return self.authority.tool_call_id

    @property
    def tool_name(self) -> str:
        return self.authority.tool_name

    @property
    def idempotency_key(self) -> str:
        return self.authority.idempotency_key

    @property
    def effective_arguments_sha256(self) -> str:
        return self.authority.effective_arguments_sha256

    @property
    def agent_alias(self) -> str:
        return self.authority.agent_alias

    @property
    def agent_name(self) -> str:
        return self.authority.agent_name

    @property
    def environment_name(self) -> str | None:
        return self.authority.environment_name

    @property
    def spawn_fingerprint(self) -> str:
        return self.authority.spawn_fingerprint

    @property
    def child_session_id(self) -> str:
        return self.authority.child_session_id

    @property
    def dispatch_id(self) -> str:
        return self.authority.dispatch_id

    @property
    def queue_task_id(self) -> str:
        return self.authority.queue_task_id

    @property
    def queue_task_type(self) -> str:
        return self.authority.queue_task_type

    @property
    def interaction_id(self) -> str:
        return self.authority.interaction_id

    @property
    def interaction_started_event_id(self) -> str:
        return self.authority.interaction_started_event_id

    @property
    def request_sha256(self) -> str:
        return self.authority.request_sha256

    @property
    def request(self) -> RunRequest:
        return self.authority.request


class DurableSubagentSubmissionSeed(_DurableSubagentAuthorityRecord):
    """Immutable pre-preparation authority persisted before any submission await."""

    record_type: Literal["cayu.durable-subagent-submission-seed"] = (
        DURABLE_SUBAGENT_SEED_RECORD_TYPE
    )
    schema_version: Literal[1] = DURABLE_SUBAGENT_SEED_SCHEMA_VERSION
    effective_arguments: dict[str, Any]
    seed_sha256: str

    @field_validator("seed_sha256")
    @classmethod
    def validate_seed_sha256(cls, value: str) -> str:
        return _require_sha256(value, "seed_sha256")

    @field_validator("effective_arguments", mode="before")
    @classmethod
    def copy_effective_arguments(cls, value: object) -> dict[str, Any]:
        if type(value) is not dict:
            raise TypeError("Durable subagent effective arguments must be an object.")
        return copy_durable_json_object(value, "effective_arguments")

    @model_validator(mode="after")
    def validate_seed(self) -> DurableSubagentSubmissionSeed:
        if (
            self.effective_arguments_sha256
            != sha256(
                canonical_durable_json_bytes(
                    self.effective_arguments,
                    "effective_arguments",
                )
            ).hexdigest()
        ):
            raise ValueError("Durable subagent effective-argument digest conflicts.")
        if self.seed_sha256 != durable_subagent_seed_sha256(self, include_digest=False):
            raise ValueError("Durable subagent seed digest conflicts with its authority.")
        return self


def durable_subagent_seed_sha256(
    seed: DurableSubagentSubmissionSeed,
    *,
    include_digest: bool = True,
) -> str:
    if type(seed) is not DurableSubagentSubmissionSeed:
        raise TypeError("Durable subagent seed digest requires an exact seed.")
    payload: dict[str, Any] = seed.model_dump(mode="json", warnings=False)
    if not include_digest:
        payload.pop("seed_sha256", None)
    return sha256(
        canonical_durable_json_bytes(payload, "durable_subagent.submission_seed")
    ).hexdigest()


def durable_subagent_effective_arguments_sha256(arguments: object) -> str:
    """Return the canonical digest for one owned effective-argument object."""

    if type(arguments) is not dict:
        raise TypeError("Durable subagent effective arguments must be an object.")
    owned = copy_durable_json_object(arguments, "effective_arguments")
    return sha256(canonical_durable_json_bytes(owned, "effective_arguments")).hexdigest()


def new_durable_subagent_submission_seed(**authority: Any) -> DurableSubagentSubmissionSeed:
    """Construct one validated seed while deriving its content digest."""

    payload = dict(authority)
    shared = {
        field_name: payload.pop(field_name) for field_name in DurableSubagentAuthority.model_fields
    }
    candidate = DurableSubagentSubmissionSeed.model_construct(
        authority=DurableSubagentAuthority.model_validate(shared),
        **payload,
        seed_sha256="0" * 64,
    )
    payload = candidate.model_dump(mode="json", warnings=False)
    payload["seed_sha256"] = durable_subagent_seed_sha256(
        candidate,
        include_digest=False,
    )
    return DurableSubagentSubmissionSeed.model_validate(payload)


def copy_durable_subagent_submission_seed(
    seed: DurableSubagentSubmissionSeed,
) -> DurableSubagentSubmissionSeed:
    if type(seed) is not DurableSubagentSubmissionSeed:
        raise TypeError("Durable subagent submission seed must be exact.")
    return DurableSubagentSubmissionSeed.model_validate(
        seed.model_dump(mode="json", warnings=False)
    )


class DurableSubagentSubmissionIntent(_DurableSubagentAuthorityRecord):
    """Immutable mapping persisted before a durable child queue task is published."""

    record_type: Literal["cayu.durable-subagent-submission"] = DURABLE_SUBAGENT_INTENT_RECORD_TYPE
    schema_version: Literal[2] = DURABLE_SUBAGENT_INTENT_SCHEMA_VERSION
    child_provider_name: str
    child_model: str
    child_runtime_name: str
    child_runtime_version: str | None
    seed_sha256: str
    child_execution_profile: ExecutionProfileIdentity
    submission_sha256: str

    @field_validator("child_provider_name", "child_model", "child_runtime_name")
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("child_runtime_version")
    @classmethod
    def validate_optional_runtime_version(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator(
        "seed_sha256",
        "submission_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str, info) -> str:
        return _require_sha256(value, info.field_name)

    @field_validator("child_execution_profile", mode="before")
    @classmethod
    def copy_profile(cls, value: object) -> ExecutionProfileIdentity:
        if type(value) is ExecutionProfileIdentity:
            value = value.model_dump(mode="json")
        return ExecutionProfileIdentity.model_validate(value)

    @model_validator(mode="after")
    def validate_intent(self) -> DurableSubagentSubmissionIntent:
        if self.parent_execution_profile_fingerprint == self.child_execution_profile.fingerprint:
            # Equality is allowed; this branch documents that the two authorities remain
            # distinct even when their content happens to match.
            pass
        expected_submission = durable_subagent_submission_sha256(self, include_digest=False)
        if self.submission_sha256 != expected_submission:
            raise ValueError("Durable subagent submission digest conflicts with its authority.")
        return self


class DurableSubagentSubmissionReceipt(BaseModel):
    """Bounded parent-side proof retained after spawn-result publication."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.durable-subagent-submission-receipt"] = (
        DURABLE_SUBAGENT_RECEIPT_RECORD_TYPE
    )
    schema_version: Literal[1] = DURABLE_SUBAGENT_RECEIPT_SCHEMA_VERSION
    outcome: Literal["submitted", "rejected"]
    parent_session_id: str
    parent_session_instance_fingerprint: str
    parent_run_epoch: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    causal_budget_id: str
    tool_round_id: str
    tool_call_id: str
    tool_name: str
    idempotency_key: str
    agent_alias: str
    agent_name: str
    spawn_fingerprint: str
    child_session_id: str
    dispatch_id: str
    queue_task_id: str
    queue_task_type: str
    effective_arguments_sha256: str
    seed_sha256: str
    submission_sha256: str | None = None
    failure_code: Literal["preparation_rejected"] | None = None
    receipt_sha256: str

    @field_validator(
        "parent_session_id",
        "causal_budget_id",
        "tool_round_id",
        "tool_call_id",
        "tool_name",
        "idempotency_key",
        "agent_alias",
        "agent_name",
        "spawn_fingerprint",
        "child_session_id",
        "dispatch_id",
        "queue_task_id",
        "queue_task_type",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator(
        "parent_session_instance_fingerprint",
        "effective_arguments_sha256",
        "seed_sha256",
        "submission_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _require_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_authority(self) -> DurableSubagentSubmissionReceipt:
        if self.outcome == "submitted":
            if self.submission_sha256 is None or self.failure_code is not None:
                raise ValueError("Submitted durable subagent receipt is incomplete.")
        elif self.submission_sha256 is not None or self.failure_code != "preparation_rejected":
            raise ValueError("Rejected durable subagent receipt is incomplete.")
        if self.child_session_id != generate_child_session_id(
            kind=ChildSessionKind.SUBAGENT,
            parent_session_id=self.parent_session_id,
            logical_spawn_id=self.idempotency_key,
        ):
            raise ValueError("Durable subagent receipt child identity conflicts.")
        if self.dispatch_id != durable_subagent_dispatch_id(
            parent_session_id=self.parent_session_id,
            idempotency_key=self.idempotency_key,
        ):
            raise ValueError("Durable subagent receipt dispatch identity conflicts.")
        if self.queue_task_id != durable_dispatch_queue_task_id(
            task_type=self.queue_task_type,
            dispatch_id=self.dispatch_id,
        ):
            raise ValueError("Durable subagent receipt queue identity conflicts.")
        if self.receipt_sha256 != durable_subagent_receipt_sha256(
            self,
            include_digest=False,
        ):
            raise ValueError("Durable subagent receipt digest conflicts.")
        return self


def durable_subagent_receipt_sha256(
    receipt: DurableSubagentSubmissionReceipt,
    *,
    include_digest: bool = True,
) -> str:
    if type(receipt) is not DurableSubagentSubmissionReceipt:
        raise TypeError("Durable subagent receipt digest requires an exact receipt.")
    payload = receipt.model_dump(mode="json", warnings=False)
    if not include_digest:
        payload.pop("receipt_sha256", None)
    return sha256(canonical_durable_json_bytes(payload, "durable_subagent.receipt")).hexdigest()


def _new_durable_subagent_submission_receipt(
    **authority: Any,
) -> DurableSubagentSubmissionReceipt:
    candidate = DurableSubagentSubmissionReceipt.model_construct(
        **authority,
        receipt_sha256="0" * 64,
    )
    payload = candidate.model_dump(mode="json", warnings=False)
    payload["receipt_sha256"] = durable_subagent_receipt_sha256(
        candidate,
        include_digest=False,
    )
    return DurableSubagentSubmissionReceipt.model_validate(payload)


def durable_subagent_submission_receipt_from_intent(
    intent: DurableSubagentSubmissionIntent,
) -> DurableSubagentSubmissionReceipt:
    intent = copy_durable_subagent_submission_intent(intent)
    return _new_durable_subagent_submission_receipt(
        outcome="submitted",
        parent_session_id=intent.parent_session_id,
        parent_session_instance_fingerprint=intent.parent_session_instance_fingerprint,
        parent_run_epoch=intent.parent_run_epoch,
        causal_budget_id=intent.causal_budget_id,
        tool_round_id=intent.tool_round_id,
        tool_call_id=intent.tool_call_id,
        tool_name=intent.tool_name,
        idempotency_key=intent.idempotency_key,
        agent_alias=intent.agent_alias,
        agent_name=intent.agent_name,
        spawn_fingerprint=intent.spawn_fingerprint,
        child_session_id=intent.child_session_id,
        dispatch_id=intent.dispatch_id,
        queue_task_id=intent.queue_task_id,
        queue_task_type=intent.queue_task_type,
        effective_arguments_sha256=intent.effective_arguments_sha256,
        seed_sha256=intent.seed_sha256,
        submission_sha256=intent.submission_sha256,
        failure_code=None,
    )


def durable_subagent_submission_rejection_receipt(
    seed: DurableSubagentSubmissionSeed,
) -> DurableSubagentSubmissionReceipt:
    seed = copy_durable_subagent_submission_seed(seed)
    return _new_durable_subagent_submission_receipt(
        outcome="rejected",
        parent_session_id=seed.parent_session_id,
        parent_session_instance_fingerprint=seed.parent_session_instance_fingerprint,
        parent_run_epoch=seed.parent_run_epoch,
        causal_budget_id=seed.causal_budget_id,
        tool_round_id=seed.tool_round_id,
        tool_call_id=seed.tool_call_id,
        tool_name=seed.tool_name,
        idempotency_key=seed.idempotency_key,
        agent_alias=seed.agent_alias,
        agent_name=seed.agent_name,
        spawn_fingerprint=seed.spawn_fingerprint,
        child_session_id=seed.child_session_id,
        dispatch_id=seed.dispatch_id,
        queue_task_id=seed.queue_task_id,
        queue_task_type=seed.queue_task_type,
        effective_arguments_sha256=seed.effective_arguments_sha256,
        seed_sha256=seed.seed_sha256,
        submission_sha256=None,
        failure_code="preparation_rejected",
    )


def durable_subagent_submission_sha256(
    intent: DurableSubagentSubmissionIntent,
    *,
    include_digest: bool = True,
) -> str:
    if type(intent) is not DurableSubagentSubmissionIntent:
        raise TypeError("Durable subagent submission digest requires an intent.")
    payload: dict[str, Any] = intent.model_dump(mode="json", warnings=False)
    if not include_digest:
        payload.pop("submission_sha256", None)
    return sha256(canonical_durable_json_bytes(payload, "durable_subagent.submission")).hexdigest()


def new_durable_subagent_submission_intent(
    **authority: Any,
) -> DurableSubagentSubmissionIntent:
    """Construct and validate one intent while deriving its content digest."""

    payload = dict(authority)
    shared = {
        field_name: payload.pop(field_name) for field_name in DurableSubagentAuthority.model_fields
    }
    candidate = DurableSubagentSubmissionIntent.model_construct(
        authority=DurableSubagentAuthority.model_validate(shared),
        **payload,
        submission_sha256="0" * 64,
    )
    payload = candidate.model_dump(mode="json", warnings=False)
    payload["submission_sha256"] = durable_subagent_submission_sha256(
        candidate,
        include_digest=False,
    )
    return DurableSubagentSubmissionIntent.model_validate(payload)


def copy_durable_subagent_submission_intent(
    intent: DurableSubagentSubmissionIntent,
) -> DurableSubagentSubmissionIntent:
    if type(intent) is not DurableSubagentSubmissionIntent:
        raise TypeError("Durable subagent submission must be an exact intent.")
    return DurableSubagentSubmissionIntent.model_validate(
        intent.model_dump(mode="json", warnings=False)
    )


def require_durable_subagent_intent_matches_seed(
    intent: DurableSubagentSubmissionIntent,
    seed: DurableSubagentSubmissionSeed,
) -> None:
    """Reject a finalized record that does not extend the exact preparation seed."""

    if type(intent) is not DurableSubagentSubmissionIntent:
        raise TypeError("Durable subagent submission intent must be exact.")
    if type(seed) is not DurableSubagentSubmissionSeed:
        raise TypeError("Durable subagent submission seed must be exact.")
    if intent.authority != seed.authority or intent.seed_sha256 != seed.seed_sha256:
        raise RuntimeError("Durable subagent intent conflicts with its preparation seed.")


def require_durable_subagent_receipt_matches_intent(
    receipt: DurableSubagentSubmissionReceipt,
    intent: DurableSubagentSubmissionIntent,
) -> None:
    """Reject a compact parent receipt that does not authenticate one intent."""

    if type(receipt) is not DurableSubagentSubmissionReceipt:
        raise TypeError("Durable subagent receipt must be exact.")
    if type(intent) is not DurableSubagentSubmissionIntent:
        raise TypeError("Durable subagent submission intent must be exact.")
    expected = durable_subagent_submission_receipt_from_intent(intent)
    if receipt != expected:
        raise RuntimeError("Durable subagent receipt conflicts with its submission intent.")


def require_durable_subagent_receipt_matches_seed(
    receipt: DurableSubagentSubmissionReceipt,
    seed: DurableSubagentSubmissionSeed,
) -> None:
    """Reject a compact receipt that does not retain one seed's exact identity."""

    if type(receipt) is not DurableSubagentSubmissionReceipt:
        raise TypeError("Durable subagent receipt must be exact.")
    if type(seed) is not DurableSubagentSubmissionSeed:
        raise TypeError("Durable subagent submission seed must be exact.")
    shared_fields = (
        "parent_session_id",
        "parent_session_instance_fingerprint",
        "parent_run_epoch",
        "causal_budget_id",
        "tool_round_id",
        "tool_call_id",
        "tool_name",
        "idempotency_key",
        "agent_alias",
        "agent_name",
        "spawn_fingerprint",
        "child_session_id",
        "dispatch_id",
        "queue_task_id",
        "queue_task_type",
        "effective_arguments_sha256",
        "seed_sha256",
    )
    if any(getattr(receipt, field) != getattr(seed, field) for field in shared_fields):
        raise RuntimeError("Durable subagent receipt conflicts with its preparation seed.")


def require_durable_subagent_rejection_receipt_matches_seed(
    receipt: DurableSubagentSubmissionReceipt,
    seed: DurableSubagentSubmissionSeed,
) -> None:
    """Reject a permanent-rejection receipt that does not extend one exact seed."""

    if type(receipt) is not DurableSubagentSubmissionReceipt:
        raise TypeError("Durable subagent receipt must be exact.")
    if type(seed) is not DurableSubagentSubmissionSeed:
        raise TypeError("Durable subagent submission seed must be exact.")
    require_durable_subagent_receipt_matches_seed(receipt, seed)
    expected = durable_subagent_submission_rejection_receipt(seed)
    if receipt != expected:
        raise RuntimeError("Durable subagent rejection receipt conflicts with its seed.")


def checkpoint_with_durable_subagent_submission_seed(
    checkpoint: dict[str, Any] | None,
    *,
    seed: DurableSubagentSubmissionSeed,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    from cayu._validation import copy_durable_json_object
    from cayu.runtime._checkpoint_redaction import require_secret_free_durable_object

    seed = copy_durable_subagent_submission_seed(seed)
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    raw = updated.get(DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY, {})
    if type(raw) is not dict:
        raise ValueError("Durable subagent submission-seed checkpoint is malformed.")
    seeds = copy_durable_json_object(raw, "durable_subagent_submission_seeds")
    existing = seeds.get(seed.idempotency_key)
    submission = durable_subagent_submission_receipt_from_checkpoint(
        updated,
        idempotency_key=seed.idempotency_key,
    )
    if submission is not None:
        require_durable_subagent_receipt_matches_seed(submission, seed)
        if submission.outcome == "submitted":
            if existing is not None:
                if type(existing) is not dict:
                    raise RuntimeError(
                        "Committed durable subagent receipt has malformed recovery authority."
                    )
                existing_seed = DurableSubagentSubmissionSeed.model_validate(existing)
                if existing_seed != seed:
                    raise RuntimeError(
                        "Committed durable subagent receipt has conflicting recovery authority."
                    )
            return updated
    safe_seed_map = require_secret_free_durable_object(
        {seed.idempotency_key: seed.model_dump(mode="json", warnings=False)},
        redactor=redactor,
        field_name="durable_subagent_submission_seed",
        schema_root=DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY,
    )
    payload = safe_seed_map[seed.idempotency_key]
    if type(payload) is not dict:  # pragma: no cover - exact construction above
        raise AssertionError("Durable subagent seed validation returned a non-object.")
    if existing is not None and existing != payload:
        raise RuntimeError("Durable subagent submission seed conflicts with its retry.")
    seeds[seed.idempotency_key] = payload
    updated[DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY] = seeds
    return updated


def durable_subagent_submission_seed_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    idempotency_key: str,
) -> DurableSubagentSubmissionSeed | None:
    if checkpoint is None:
        return None
    raw = checkpoint.get(DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY)
    if raw is None:
        return None
    if type(raw) is not dict:
        raise ValueError("Durable subagent submission-seed checkpoint is malformed.")
    value = raw.get(require_durable_clean_nonblank(idempotency_key, "idempotency_key"))
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("Durable subagent submission seed is malformed.")
    return DurableSubagentSubmissionSeed.model_validate(value)


def checkpoint_with_durable_subagent_submission(
    checkpoint: dict[str, Any] | None,
    *,
    intent: DurableSubagentSubmissionIntent,
) -> dict[str, Any]:
    from cayu._validation import copy_durable_json_object

    intent = copy_durable_subagent_submission_intent(intent)
    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    raw = updated.get(DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY, {})
    if type(raw) is not dict:
        raise ValueError("Durable subagent submission checkpoint is malformed.")
    submissions = copy_durable_json_object(raw, "durable_subagent_submissions")
    existing = submissions.get(intent.idempotency_key)
    payload = intent.model_dump(mode="json", warnings=False)
    if existing is not None and existing != payload:
        if type(existing) is not dict:
            raise RuntimeError("Durable subagent submission identity conflicts with its retry.")
        if existing.get("record_type") != DURABLE_SUBAGENT_RECEIPT_RECORD_TYPE:
            raise RuntimeError("Durable subagent submission identity conflicts with its retry.")
        receipt = DurableSubagentSubmissionReceipt.model_validate(existing)
        require_durable_subagent_receipt_matches_intent(receipt, intent)
        return updated
    submissions[intent.idempotency_key] = payload
    updated[DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY] = submissions
    return updated


def durable_subagent_submission_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    idempotency_key: str,
) -> DurableSubagentSubmissionIntent | None:
    if checkpoint is None:
        return None
    raw = checkpoint.get(DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY)
    if raw is None:
        return None
    if type(raw) is not dict:
        raise ValueError("Durable subagent submission checkpoint is malformed.")
    value = raw.get(require_durable_clean_nonblank(idempotency_key, "idempotency_key"))
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("Durable subagent submission intent is malformed.")
    if value.get("record_type") == DURABLE_SUBAGENT_RECEIPT_RECORD_TYPE:
        DurableSubagentSubmissionReceipt.model_validate(value)
        return None
    return DurableSubagentSubmissionIntent.model_validate(value)


def durable_subagent_submission_receipt_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    idempotency_key: str,
) -> DurableSubagentSubmissionReceipt | None:
    if checkpoint is None:
        return None
    raw = checkpoint.get(DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY)
    if raw is None:
        return None
    if type(raw) is not dict:
        raise ValueError("Durable subagent submission checkpoint is malformed.")
    value = raw.get(require_durable_clean_nonblank(idempotency_key, "idempotency_key"))
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("Durable subagent submission receipt is malformed.")
    if value.get("record_type") != DURABLE_SUBAGENT_RECEIPT_RECORD_TYPE:
        DurableSubagentSubmissionIntent.model_validate(value)
        return None
    return DurableSubagentSubmissionReceipt.model_validate(value)


def checkpoint_with_durable_subagent_submission_rejection(
    checkpoint: dict[str, Any] | None,
    *,
    seed: DurableSubagentSubmissionSeed,
) -> dict[str, Any]:
    """Persist one bounded permanent rejection while retaining recovery arguments."""

    seed = copy_durable_subagent_submission_seed(seed)
    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    persisted_seed = durable_subagent_submission_seed_from_checkpoint(
        updated,
        idempotency_key=seed.idempotency_key,
    )
    if persisted_seed != seed:
        raise RuntimeError("Durable subagent rejection lost its exact preparation seed.")
    raw = updated.get(DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY, {})
    if type(raw) is not dict:
        raise ValueError("Durable subagent submission checkpoint is malformed.")
    submissions = copy_durable_json_object(raw, "durable_subagent_submissions")
    receipt = durable_subagent_submission_rejection_receipt(seed)
    payload = receipt.model_dump(mode="json", warnings=False)
    existing = submissions.get(seed.idempotency_key)
    if existing is not None and existing != payload:
        raise RuntimeError("Durable subagent rejection conflicts with durable authority.")
    submissions[seed.idempotency_key] = payload
    updated[DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY] = submissions
    return updated


def checkpoint_with_committed_durable_subagent_submission(
    checkpoint: dict[str, Any] | None,
    *,
    intent: DurableSubagentSubmissionIntent,
) -> dict[str, Any]:
    """Compact a confirmed handoff when the pending call retains its arguments."""

    intent = copy_durable_subagent_submission_intent(intent)
    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    seed = durable_subagent_submission_seed_from_checkpoint(
        updated,
        idempotency_key=intent.idempotency_key,
    )
    persisted_intent = durable_subagent_submission_from_checkpoint(
        updated,
        idempotency_key=intent.idempotency_key,
    )
    persisted_receipt = durable_subagent_submission_receipt_from_checkpoint(
        updated,
        idempotency_key=intent.idempotency_key,
    )
    expected_receipt = durable_subagent_submission_receipt_from_intent(intent)
    if seed is None and persisted_intent is None and persisted_receipt == expected_receipt:
        return updated
    if seed is None or persisted_intent is None or persisted_receipt is not None:
        raise RuntimeError("Durable subagent handoff has incomplete parent authority.")
    require_durable_subagent_intent_matches_seed(persisted_intent, seed)
    if persisted_intent != intent:
        raise RuntimeError("Durable subagent handoff conflicts with its parent intent.")

    raw_pending_round = updated.get("pending_tool_round")
    if type(raw_pending_round) is not dict:
        raise RuntimeError("Durable subagent handoff has no pending tool round.")
    if raw_pending_round.get("tool_round_id") != intent.tool_round_id:
        raise RuntimeError("Durable subagent handoff targets another pending tool round.")
    raw_tool_calls = raw_pending_round.get("tool_calls")
    if type(raw_tool_calls) is not list:
        raise RuntimeError("Durable subagent pending tool calls are malformed.")
    matching_calls = [
        call
        for call in raw_tool_calls
        if type(call) is dict and call.get("tool_call_id") == intent.tool_call_id
    ]
    if len(matching_calls) != 1:
        raise RuntimeError("Durable subagent pending tool-call authority is ambiguous.")
    pending_arguments = matching_calls[0].get("arguments")
    if type(pending_arguments) is not dict:
        raise RuntimeError("Durable subagent pending tool arguments are malformed.")
    if (
        durable_subagent_effective_arguments_sha256(pending_arguments)
        != intent.effective_arguments_sha256
    ):
        # A before-tool hook changed the arguments. The original pending call
        # cannot reconstruct those effective arguments, so retain the exact seed
        # until terminal tool-round publication compacts both records together.
        raw_submissions = updated.get(DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY)
        if type(raw_submissions) is not dict:
            raise RuntimeError("Durable subagent handoff authority map is malformed.")
        submissions = copy_durable_json_object(
            raw_submissions,
            "durable_subagent_submissions",
        )
        submissions[intent.idempotency_key] = expected_receipt.model_dump(
            mode="json",
            warnings=False,
        )
        updated[DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY] = submissions
        return updated

    raw_seeds = updated.get(DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY)
    raw_submissions = updated.get(DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY)
    if type(raw_seeds) is not dict or type(raw_submissions) is not dict:
        raise RuntimeError("Durable subagent handoff authority maps are malformed.")
    seeds = copy_durable_json_object(raw_seeds, "durable_subagent_submission_seeds")
    submissions = copy_durable_json_object(
        raw_submissions,
        "durable_subagent_submissions",
    )
    seeds.pop(intent.idempotency_key)
    submissions[intent.idempotency_key] = expected_receipt.model_dump(
        mode="json",
        warnings=False,
    )
    if seeds:
        updated[DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY] = seeds
    else:
        updated.pop(DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY, None)
    updated[DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY] = submissions
    return updated


def checkpoint_with_compacted_durable_subagent_submissions(
    checkpoint: dict[str, Any],
    *,
    tool_round_id: str,
    committed_handoffs: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    """Replace one published round's full parent authority with bounded receipts."""

    updated = copy_durable_json_object(checkpoint, "checkpoint")
    tool_round_id = require_durable_clean_nonblank(tool_round_id, "tool_round_id")
    if type(committed_handoffs) is not dict or any(
        type(tool_call_id) is not str
        or type(handoff) is not tuple
        or len(handoff) != 2
        or any(type(value) is not str for value in handoff)
        for tool_call_id, handoff in committed_handoffs.items()
    ):
        raise TypeError("Committed durable-subagent handoffs must have exact string authority.")
    raw_seeds = updated.get(DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY)
    if raw_seeds is None:
        return updated
    if type(raw_seeds) is not dict:
        raise ValueError("Durable subagent submission-seed checkpoint is malformed.")
    raw_submissions = updated.get(DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY)
    if type(raw_submissions) is not dict:
        raise RuntimeError("Durable subagent seed has no submission authority map.")
    seeds = copy_durable_json_object(raw_seeds, "durable_subagent_submission_seeds")
    submissions = copy_durable_json_object(
        raw_submissions,
        "durable_subagent_submissions",
    )
    compacted = False
    for idempotency_key in tuple(seeds):
        seed_payload = seeds[idempotency_key]
        if type(seed_payload) is not dict:
            raise ValueError("Durable subagent submission seed is malformed.")
        seed = DurableSubagentSubmissionSeed.model_validate(seed_payload)
        if seed.idempotency_key != idempotency_key:
            raise ValueError("Durable subagent submission seed map key conflicts.")
        if seed.tool_round_id != tool_round_id:
            continue
        record = submissions.get(idempotency_key)
        if type(record) is not dict:
            raise RuntimeError("Published durable subagent has no terminal submission record.")
        if record.get("record_type") == DURABLE_SUBAGENT_RECEIPT_RECORD_TYPE:
            receipt = DurableSubagentSubmissionReceipt.model_validate(record)
            if receipt.outcome == "submitted":
                require_durable_subagent_receipt_matches_seed(receipt, seed)
            else:
                require_durable_subagent_rejection_receipt_matches_seed(receipt, seed)
        else:
            intent = DurableSubagentSubmissionIntent.model_validate(record)
            require_durable_subagent_intent_matches_seed(intent, seed)
            if committed_handoffs.get(seed.tool_call_id) != (
                seed.child_session_id,
                seed.queue_task_id,
            ):
                continue
            receipt = durable_subagent_submission_receipt_from_intent(intent)
        submissions[idempotency_key] = receipt.model_dump(mode="json", warnings=False)
        seeds.pop(idempotency_key)
        compacted = True
    if not compacted:
        return updated
    if seeds:
        updated[DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY] = seeds
    else:
        updated.pop(DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY, None)
    updated[DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY] = submissions
    return updated


def durable_subagent_submissions_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> tuple[DurableSubagentSubmissionIntent, ...]:
    """Return every keyed submission after validating map-key authority."""

    if checkpoint is None:
        return ()
    raw = checkpoint.get(DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY)
    if raw is None:
        return ()
    if type(raw) is not dict:
        raise ValueError("Durable subagent submission checkpoint is malformed.")
    if any(type(key) is not str for key in raw):
        raise ValueError("Durable subagent submission intent is malformed.")
    intents: list[DurableSubagentSubmissionIntent] = []
    for key in sorted(raw):
        value = raw[key]
        if type(value) is not dict:
            raise ValueError("Durable subagent submission intent is malformed.")
        intent = DurableSubagentSubmissionIntent.model_validate(value)
        if intent.idempotency_key != key:
            raise ValueError("Durable subagent submission map key conflicts with its intent.")
        intents.append(intent)
    return tuple(intents)
