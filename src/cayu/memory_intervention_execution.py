"""Durable application-owned execution boundary for fixed memory interventions.

The portable declarations in :mod:`cayu.memory_interventions` intentionally do
not execute candidate effects.  This module owns the bounded execution
identity, durable phase journal, and extension seams used to open an isolated
AgentSnapshot memory view.  Runtime integration is deliberately expressed as
one typed runner boundary so ordinary Cayu runs cannot acquire intervention
authority from caller metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self
from uuid import uuid4

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

from cayu._clock import utc_clock
from cayu._exception_groups import (
    exception_cause,
    exception_tree_contains,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._task_wait import (
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    compact_json_utf8_size,
    copy_durable_json_object,
    require_durable_clean_nonblank,
    revalidate_model_input,
)
from cayu.agent_snapshots import (
    AGENT_SNAPSHOT_TRIAL_METADATA_KEY,
    AgentSnapshot,
    AgentSnapshotAccess,
    AgentSnapshotCoordinator,
    AgentSnapshotExecutionProfileRef,
    AgentSnapshotMaterialization,
    AgentSnapshotMaterializationRequest,
    AgentSnapshotResultBinding,
    AgentSnapshotTerminalDisposition,
    AgentSnapshotTrialBinding,
    execution_profile_snapshot_ref,
)
from cayu.core.events import event_durable_sequence
from cayu.evals._memory_attribution import (
    eval_memory_attribution_evidence_from_runtime_source,
)
from cayu.evals.memory_attribution import (
    EvalMemoryAttributionEvidenceV1,
    EvalMemoryEvidenceLimitation,
    EvalMemorySourceAliasV1,
    eval_memory_source_alias,
    standard_eval_memory_attribution_bounds,
)
from cayu.evals.models import (
    EvalCaseContractV1,
    EvalTrialResult,
    _memory_source_expected_counts,
)
from cayu.evals.revisions import eval_trial_result_revision
from cayu.memory import AutomaticRecallPolicy
from cayu.memory_attribution import (
    MemoryAttribution,
    MemoryAttributionBounds,
    MemoryAttributionStatus,
    MemoryAttributionUnavailableReason,
)
from cayu.memory_interventions import (
    MemoryInterventionEffectStatus,
    MemoryInterventionOperation,
    MemoryInterventionReceipt,
    MemoryInterventionSpec,
    MemoryInterventionTrialBinding,
    memory_attribution_fingerprint,
)
from cayu.runtime._durable_operation_ownership import (
    DurableOperationOwnership,
    DurableOperationOwnershipAction,
    DurableOperationOwnershipDisposition,
    DurableOperationOwnershipResult,
    DurableOperationOwnershipState,
    DurableOperationOwnershipTransition,
    transition_durable_operation_ownership,
)
from cayu.runtime._memory_attribution import (
    MemoryAttributionCaptureBudget,
    project_memory_attribution,
)
from cayu.runtime._memory_evidence import memory_evidence_key
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileIdentity,
)
from cayu.runtime.sessions import (
    IncompleteSessionRecoveryRequest,
    RunnerObservedEventIdentity,
    RunRequest,
    RuntimeSessionCreateClaimAuthenticationDisposition,
    RuntimeSessionCreateClaimReference,
    RuntimeSessionCreateClaimReferenceKey,
    Session,
    SessionStatus,
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    authenticate_runtime_session_create_claim_reference,
    copy_run_request,
    copy_session,
    copy_terminal_session_evidence,
    run_request_with_runtime_generated_authority,
    run_request_with_runtime_session_create_claim_reference,
    runtime_session_create_claim_reference,
)
from cayu.storage.memory import (
    KnowledgeAccessScope,
    KnowledgeStore,
    copy_knowledge_access_scope,
)

MEMORY_INTERVENTION_EXECUTION_SCHEMA_VERSION = 1
MEMORY_INTERVENTION_EXECUTION_RECORD_SCHEMA_VERSION = 2
MEMORY_INTERVENTION_EXECUTION_MAX_TIMEOUT_SECONDS = 3_600
MEMORY_INTERVENTION_EXECUTION_MAX_RECORD_BYTES = 1 << 20
MEMORY_INTERVENTION_ATTRIBUTION_MAX_PROJECTION_BYTES = 1 << 19
MEMORY_INTERVENTION_RUNTIME_LEASE_SECONDS = 30
MEMORY_INTERVENTION_RUNTIME_HEARTBEAT_SECONDS = 10.0
MEMORY_INTERVENTION_RUNTIME_OWNERSHIP_WAIT_SECONDS = 0.25

_HMAC_CONTEXT = b"cayu.memory-intervention-execution-request.v1\x00"
_SHA256_HEX = frozenset("0123456789abcdef")
_MAX_ID_CHARS = 256
_EXECUTION_GATES_LOCK = threading.Lock()
_EXECUTION_GATES: dict[str, Future[None]] = {}
_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    hide_input_in_errors=True,
    revalidate_instances="always",
    validate_default=True,
)


async def _acquire_execution_gate(execution_id: str) -> Future[None]:
    """Serialize one process-local identity across executors and event loops."""

    while True:
        with _EXECUTION_GATES_LOCK:
            existing = _EXECUTION_GATES.get(execution_id)
            if existing is None:
                owned: Future[None] = Future()
                _EXECUTION_GATES[execution_id] = owned
                return owned
        await asyncio.shield(asyncio.wrap_future(existing))


def _release_execution_gate(execution_id: str, owned: Future[None]) -> None:
    with _EXECUTION_GATES_LOCK:
        if _EXECUTION_GATES.get(execution_id) is not owned:
            raise AssertionError("Memory-intervention execution gate ownership changed.")
        del _EXECUTION_GATES[execution_id]
    owned.set_result(None)


def _clean(value: str, field_name: str, *, max_chars: int = _MAX_ID_CHARS) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > max_chars:
        raise ValueError(f"{field_name} must be at most {max_chars} characters.")
    return value


def _sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _content_revision(value: str, field_name: str) -> str:
    if not value.startswith("sha256:"):
        raise ValueError(f"{field_name} must be a sha256 content revision.")
    _sha256(value.removeprefix("sha256:"), field_name)
    return value


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


def _content_sha256(value: object, field_name: str) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


class _ExecutionModel(BaseModel):
    model_config = _MODEL_CONFIG


class MemoryInterventionExecutionPhase(StrEnum):
    PREPARED = "prepared"
    TRIAL_BOUND = "trial_bound"
    EFFECT_RESOLVED = "effect_resolved"
    SESSION_BOUND = "session_bound"
    RUNTIME_TERMINAL = "runtime_terminal"
    EVALUATED = "evaluated"
    FINALIZED = "finalized"


_PHASE_ORDER = {phase: index for index, phase in enumerate(MemoryInterventionExecutionPhase)}


class MemoryInterventionExecutionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    OUTCOME_UNKNOWN = "outcome_unknown"
    CONFLICTING = "conflicting"
    INDETERMINATE = "indeterminate"


class MemoryInterventionIsolationAuthority(_ExecutionModel):
    """Positive authority that one store is an intervention-only overlay.

    The application-owned overlay provider creates this record only after it
    has opened a store that cannot write through to the production knowledge
    store.  Runtime adapters validate the complete materialization authority
    before exposing the store to Cayu.
    """

    materialization_fingerprint: StrictStr
    memory_overlay_fingerprint: StrictStr
    state_scope_id: StrictStr = Field(max_length=_MAX_ID_CHARS)

    @field_validator("materialization_fingerprint", "memory_overlay_fingerprint")
    @classmethod
    def validate_fingerprints(cls, value: str, info) -> str:
        return _sha256(value, info.field_name)

    @field_validator("state_scope_id")
    @classmethod
    def validate_scope(cls, value: str, info) -> str:
        return _clean(value, info.field_name)


class MemoryInterventionTrialRequest(_ExecutionModel):
    """One bounded fixed-candidate execution request.

    Session, task, parent, and causal-budget identities are runtime-owned for
    this API.  Rejecting caller values prevents a candidate trial from joining
    unrelated durable ownership or accounting scopes.
    """

    schema_version: Literal[1] = MEMORY_INTERVENTION_EXECUTION_SCHEMA_VERSION
    spec: MemoryInterventionSpec
    candidate_id: StrictStr = Field(max_length=_MAX_ID_CHARS)
    trial_id: StrictStr = Field(max_length=_MAX_ID_CHARS)
    case: EvalCaseContractV1
    run_request: RunRequest
    timeout_seconds: StrictInt = Field(
        default=300,
        ge=1,
        le=MEMORY_INTERVENTION_EXECUTION_MAX_TIMEOUT_SECONDS,
    )

    @field_validator("schema_version", "timeout_seconds", mode="before")
    @classmethod
    def validate_integer_literals(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be a JSON integer.")
        return value

    @field_validator("spec", mode="before")
    @classmethod
    def copy_spec(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionSpec)

    @field_validator("case", mode="before")
    @classmethod
    def copy_case(cls, value: object) -> object:
        return revalidate_model_input(value, EvalCaseContractV1)

    @field_validator("run_request")
    @classmethod
    def copy_request(cls, value: RunRequest) -> RunRequest:
        if type(value) is not RunRequest:
            raise TypeError("run_request must be an exact RunRequest.")
        runtime_authority = value._runtime_generated_authority
        if type(runtime_authority) is not frozenset or runtime_authority:
            raise ValueError(
                "Memory intervention execution does not accept prepared runtime authority."
            )
        if type(value._input_redactions_applied) is not bool or value._input_redactions_applied:
            raise ValueError(
                "Memory intervention execution does not accept prepared runtime authority."
            )
        if any(
            authority is not None
            for authority in (
                value._runtime_session_create_claim,
                value._verified_invocation_origin,
                value._runtime_invocation_source,
                value._runtime_task_invocation,
                value._runtime_prepared_session_authority,
            )
        ):
            raise ValueError(
                "Memory intervention execution does not accept prepared runtime authority."
            )
        return copy_run_request(value)

    @field_validator("candidate_id", "trial_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @model_validator(mode="after")
    def validate_isolated_request(self) -> Self:
        request = self.run_request
        owned_values = {
            "session_id": request.session_id,
            "parent_session_id": request.parent_session_id,
            "causal_budget_id": request.causal_budget_id,
            "task_id": request.task_id,
            "task_worker_id": request.task_worker_id,
            "invocation_origin": request.invocation_origin,
        }
        supplied = tuple(name for name, value in owned_values.items() if value is not None)
        if supplied:
            raise ValueError(
                "Memory intervention execution owns these RunRequest fields: "
                + ", ".join(supplied)
                + "."
            )
        if request.loop_policies:
            raise ValueError(
                "Memory intervention execution does not accept request-local loop policies."
            )
        if AGENT_SNAPSHOT_TRIAL_METADATA_KEY in request.metadata:
            raise ValueError("Memory intervention execution owns AgentSnapshot trial metadata.")
        return self

    def identity_material(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "spec_fingerprint": self.spec.fingerprint,
            "candidate_id": self.candidate_id,
            "trial_id": self.trial_id,
            "case": self.case.model_dump(mode="json"),
            "run_request": self.run_request.model_dump(mode="json"),
            "timeout_seconds": self.timeout_seconds,
        }

    @property
    def execution_id(self) -> str:
        return _content_sha256(
            {
                "spec_fingerprint": self.spec.fingerprint,
                "candidate_id": self.candidate_id,
                "trial_id": self.trial_id,
                "case": self.case.model_dump(mode="json"),
            },
            "memory intervention execution identity",
        )

    @property
    def session_id(self) -> str:
        return f"cayu_memtrial_{self.execution_id[:40]}"

    @property
    def causal_budget_id(self) -> str:
        return f"cayu_membudget_{self.execution_id[:40]}"


@dataclass(frozen=True, slots=True, repr=False)
class MemoryInterventionRequestFingerprintKey:
    """Detached restart-stable key for secret-safe exact request identity."""

    key_id: str
    secret: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_id", _clean(self.key_id, "key_id"))
        if type(self.secret) is not bytes or len(self.secret) < 32:
            raise ValueError("Memory intervention request keys require at least 32 bytes.")

    def __repr__(self) -> str:
        return f"MemoryInterventionRequestFingerprintKey(key_id={self.key_id!r})"

    def fingerprint(self, request: MemoryInterventionTrialRequest) -> str:
        if type(request) is not MemoryInterventionTrialRequest:
            raise TypeError("request must be an exact MemoryInterventionTrialRequest.")
        return hmac.new(
            self.secret,
            _HMAC_CONTEXT
            + canonical_durable_json_bytes(
                request.identity_material(),
                "memory intervention execution request",
            ),
            hashlib.sha256,
        ).hexdigest()

    def runtime_session_create_reference_key(self) -> RuntimeSessionCreateClaimReferenceKey:
        """Derive a restricted runtime-reference key from this rotated root key."""

        return RuntimeSessionCreateClaimReferenceKey(
            key_id=self.key_id,
            secret=hmac.digest(
                self.secret,
                b"cayu.memory-intervention.runtime-session-create-reference-key.v1",
                "sha256",
            ),
        )


class MemoryInterventionExecutionRecord(_ExecutionModel):
    """One CAS-owned durable execution journal row."""

    record_type: Literal["cayu.memory-intervention-execution"] = (
        "cayu.memory-intervention-execution"
    )
    schema_version: Literal[2] = MEMORY_INTERVENTION_EXECUTION_RECORD_SCHEMA_VERSION
    execution_id: StrictStr
    request_key_id: StrictStr = Field(max_length=_MAX_ID_CHARS)
    request_fingerprint: StrictStr
    spec_fingerprint: StrictStr
    candidate_id: StrictStr = Field(max_length=_MAX_ID_CHARS)
    trial_id: StrictStr = Field(max_length=_MAX_ID_CHARS)
    case_id: StrictStr = Field(max_length=_MAX_ID_CHARS)
    case_revision: StrictStr
    required_execution_profile_fingerprint: StrictStr
    runtime_execution_profile_fingerprint: StrictStr
    overlay_provider_id: StrictStr = Field(max_length=_MAX_ID_CHARS)
    overlay_provider_fingerprint: StrictStr
    runtime_runner_fingerprint: StrictStr
    evaluator_fingerprint: StrictStr
    session_id: StrictStr = Field(max_length=512)
    causal_budget_id: StrictStr = Field(max_length=512)
    phase: MemoryInterventionExecutionPhase
    status: MemoryInterventionExecutionStatus = MemoryInterventionExecutionStatus.ACTIVE
    revision: StrictInt = Field(ge=0)
    runtime_cancellation_observed: bool = False
    runtime_timeout_observed: bool = False
    runtime_session_create_claim: RuntimeSessionCreateClaimReference | None = None
    runtime_deadline_at: datetime | None = None
    runtime_dispatch_ownership: DurableOperationOwnership | None = None
    created_at: datetime
    updated_at: datetime
    materialization_fingerprint: StrictStr | None = None
    trial_binding_fingerprint: StrictStr | None = None
    operation_fingerprint: StrictStr | None = None
    receipt_fingerprint: StrictStr | None = None
    runtime_evidence_fingerprint: StrictStr | None = None
    runtime_result_fingerprint: StrictStr | None = None
    runtime_result_payload: dict[str, Any] | None = None
    eval_result_revision: StrictStr | None = None
    snapshot_result_fingerprint: StrictStr | None = None
    final_binding_fingerprint: StrictStr | None = None
    failure_code: (
        Literal[
            "intervention_conflicting",
            "intervention_indeterminate",
            "runtime_cancelled",
            "runtime_failed",
            "runtime_outcome_unknown",
            "runtime_timed_out",
        ]
        | None
    ) = None

    @field_validator("schema_version", "revision", mode="before")
    @classmethod
    def validate_integer_literals(cls, value: object, info) -> object:
        if type(value) is not int:
            raise ValueError(f"{info.field_name} must be a JSON integer.")
        return value

    @field_validator(
        "runtime_cancellation_observed",
        "runtime_timeout_observed",
        mode="before",
    )
    @classmethod
    def validate_runtime_control_observed(cls, value: object, info) -> object:
        if type(value) is not bool:
            raise ValueError(f"{info.field_name} must be a JSON boolean.")
        return value

    @field_validator(
        "execution_id",
        "request_fingerprint",
        "spec_fingerprint",
        "required_execution_profile_fingerprint",
        "runtime_execution_profile_fingerprint",
        "overlay_provider_fingerprint",
        "runtime_runner_fingerprint",
        "evaluator_fingerprint",
        "materialization_fingerprint",
        "trial_binding_fingerprint",
        "operation_fingerprint",
        "receipt_fingerprint",
        "runtime_evidence_fingerprint",
        "runtime_result_fingerprint",
        "eval_result_revision",
        "snapshot_result_fingerprint",
        "final_binding_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256(value, info.field_name)

    @field_validator("case_revision")
    @classmethod
    def validate_case_revision(cls, value: str, info) -> str:
        return _content_revision(value, info.field_name)

    @field_validator("runtime_result_payload", mode="before")
    @classmethod
    def copy_runtime_result_payload(cls, value: object) -> object:
        if value is None:
            return None
        return copy_durable_json_object(value, "runtime_result_payload")

    @field_validator(
        "request_key_id",
        "overlay_provider_id",
        "candidate_id",
        "trial_id",
        "case_id",
        "session_id",
        "causal_budget_id",
    )
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean(
            value,
            info.field_name,
            max_chars=(512 if info.field_name in {"session_id", "causal_budget_id"} else 256),
        )

    @field_validator("runtime_dispatch_ownership", mode="before")
    @classmethod
    def copy_runtime_dispatch_ownership(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, DurableOperationOwnership)

    @field_validator("runtime_session_create_claim", mode="before")
    @classmethod
    def copy_runtime_session_create_claim(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, RuntimeSessionCreateClaimReference)

    @field_validator(
        "created_at",
        "updated_at",
        "runtime_deadline_at",
    )
    @classmethod
    def validate_timestamps(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return _utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_phase(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at.")
        required_by_phase = {
            MemoryInterventionExecutionPhase.TRIAL_BOUND: (
                "materialization_fingerprint",
                "trial_binding_fingerprint",
                "operation_fingerprint",
            ),
            MemoryInterventionExecutionPhase.EFFECT_RESOLVED: ("receipt_fingerprint",),
            MemoryInterventionExecutionPhase.SESSION_BOUND: (
                "runtime_session_create_claim",
                "runtime_deadline_at",
            ),
            MemoryInterventionExecutionPhase.RUNTIME_TERMINAL: (
                "runtime_dispatch_ownership",
                "runtime_evidence_fingerprint",
                "runtime_result_fingerprint",
                "runtime_result_payload",
            ),
            MemoryInterventionExecutionPhase.EVALUATED: ("eval_result_revision",),
            MemoryInterventionExecutionPhase.FINALIZED: (
                "snapshot_result_fingerprint",
                "final_binding_fingerprint",
            ),
        }
        for phase, fields in required_by_phase.items():
            if _PHASE_ORDER[self.phase] >= _PHASE_ORDER[phase]:
                missing = tuple(field for field in fields if getattr(self, field) is None)
                if missing:
                    raise ValueError(
                        f"{self.phase.value} execution is missing " + ", ".join(missing) + "."
                    )
        evidence_phase = {
            "materialization_fingerprint": MemoryInterventionExecutionPhase.TRIAL_BOUND,
            "trial_binding_fingerprint": MemoryInterventionExecutionPhase.TRIAL_BOUND,
            "operation_fingerprint": MemoryInterventionExecutionPhase.TRIAL_BOUND,
            "receipt_fingerprint": MemoryInterventionExecutionPhase.EFFECT_RESOLVED,
            "runtime_session_create_claim": MemoryInterventionExecutionPhase.SESSION_BOUND,
            "runtime_deadline_at": MemoryInterventionExecutionPhase.SESSION_BOUND,
            "runtime_dispatch_ownership": MemoryInterventionExecutionPhase.SESSION_BOUND,
            "runtime_evidence_fingerprint": MemoryInterventionExecutionPhase.RUNTIME_TERMINAL,
            "runtime_result_fingerprint": MemoryInterventionExecutionPhase.RUNTIME_TERMINAL,
            "runtime_result_payload": MemoryInterventionExecutionPhase.RUNTIME_TERMINAL,
            "eval_result_revision": MemoryInterventionExecutionPhase.EVALUATED,
            "snapshot_result_fingerprint": MemoryInterventionExecutionPhase.FINALIZED,
            "final_binding_fingerprint": MemoryInterventionExecutionPhase.FINALIZED,
        }
        premature = tuple(
            field
            for field, first_phase in evidence_phase.items()
            if getattr(self, field) is not None
            and _PHASE_ORDER[self.phase] < _PHASE_ORDER[first_phase]
        )
        if premature:
            raise ValueError(
                f"{self.phase.value} execution carries premature " + ", ".join(premature) + "."
            )
        if (self.runtime_cancellation_observed or self.runtime_timeout_observed) and (
            _PHASE_ORDER[self.phase] < _PHASE_ORDER[MemoryInterventionExecutionPhase.SESSION_BOUND]
        ):
            raise ValueError(
                "Runtime terminal-control authority cannot precede the durable session claim."
            )
        if (
            self.runtime_dispatch_ownership is not None
            and self.runtime_dispatch_ownership.operation_id != self.execution_id
        ):
            raise ValueError("Runtime dispatch ownership conflicts with execution identity.")
        if self.runtime_session_create_claim is not None and (
            self.runtime_session_create_claim.session_id != self.session_id
            or self.runtime_session_create_claim.operation_id != self.execution_id
        ):
            raise ValueError("Runtime session create reference conflicts with execution identity.")
        if self.runtime_result_payload is not None:
            runtime_result = MemoryInterventionRuntimeResult.model_validate(
                self.runtime_result_payload
            )
            if (
                self.runtime_result_fingerprint
                != memory_intervention_runtime_result_fingerprint(runtime_result)
                or self.runtime_evidence_fingerprint != runtime_result.runtime_evidence_fingerprint
                or self.session_id != runtime_result.session_id
            ):
                raise ValueError(
                    "Durable runtime result payload conflicts with execution evidence."
                )
            if (
                runtime_result.terminal_disposition is AgentSnapshotTerminalDisposition.TIMED_OUT
            ) is not self.runtime_timeout_observed:
                raise ValueError(
                    "Durable runtime timeout evidence conflicts with timeout authority."
                )
        if self.status is MemoryInterventionExecutionStatus.ACTIVE:
            if self.phase is MemoryInterventionExecutionPhase.FINALIZED:
                raise ValueError("Finalized execution cannot remain active.")
            if self.failure_code is not None:
                raise ValueError("Active execution cannot carry a failure code.")
        elif self.status is MemoryInterventionExecutionStatus.COMPLETED:
            if self.phase is not MemoryInterventionExecutionPhase.FINALIZED:
                raise ValueError("Completed execution must be finalized.")
            if self.failure_code is not None:
                raise ValueError("Completed execution cannot carry a failure code.")
        else:
            expected_failure_code = {
                MemoryInterventionExecutionStatus.FAILED: "runtime_failed",
                MemoryInterventionExecutionStatus.CANCELLED: "runtime_cancelled",
                MemoryInterventionExecutionStatus.TIMED_OUT: "runtime_timed_out",
                MemoryInterventionExecutionStatus.OUTCOME_UNKNOWN: "runtime_outcome_unknown",
                MemoryInterventionExecutionStatus.CONFLICTING: "intervention_conflicting",
                MemoryInterventionExecutionStatus.INDETERMINATE: "intervention_indeterminate",
            }[self.status]
            if self.failure_code != expected_failure_code:
                raise ValueError(
                    "Terminal execution status requires its exact bounded failure code."
                )
        encoded = canonical_durable_json_bytes(
            self.model_dump(mode="json"),
            "memory intervention execution record",
        )
        if len(encoded) > MEMORY_INTERVENTION_EXECUTION_MAX_RECORD_BYTES:
            raise ValueError("Memory intervention execution record exceeds its byte bound.")
        return self

    @classmethod
    def prepare(
        cls,
        request: MemoryInterventionTrialRequest,
        *,
        key: MemoryInterventionRequestFingerprintKey,
        overlay_provider_fingerprint: str,
        overlay_provider_id: str,
        runtime_runner_fingerprint: str,
        runtime_execution_profile_fingerprint: str,
        evaluator_fingerprint: str,
        created_at: datetime,
    ) -> MemoryInterventionExecutionRecord:
        if type(request) is not MemoryInterventionTrialRequest:
            raise TypeError("request must be an exact MemoryInterventionTrialRequest.")
        if type(key) is not MemoryInterventionRequestFingerprintKey:
            raise TypeError("key must be an exact MemoryInterventionRequestFingerprintKey.")
        timestamp = _utc(created_at, "created_at")
        return cls(
            execution_id=request.execution_id,
            request_key_id=key.key_id,
            request_fingerprint=key.fingerprint(request),
            spec_fingerprint=request.spec.fingerprint,
            candidate_id=request.candidate_id,
            trial_id=request.trial_id,
            case_id=request.case.case_id,
            case_revision=request.case.case_revision,
            required_execution_profile_fingerprint=(request.spec.execution_profile_fingerprint),
            runtime_execution_profile_fingerprint=runtime_execution_profile_fingerprint,
            overlay_provider_id=overlay_provider_id,
            overlay_provider_fingerprint=overlay_provider_fingerprint,
            runtime_runner_fingerprint=runtime_runner_fingerprint,
            evaluator_fingerprint=evaluator_fingerprint,
            session_id=request.session_id,
            causal_budget_id=request.causal_budget_id,
            phase=MemoryInterventionExecutionPhase.PREPARED,
            revision=0,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def replay_identity(self) -> tuple[object, ...]:
        return (
            self.record_type,
            self.schema_version,
            self.execution_id,
            self.request_key_id,
            self.request_fingerprint,
            self.spec_fingerprint,
            self.candidate_id,
            self.trial_id,
            self.case_id,
            self.case_revision,
            self.required_execution_profile_fingerprint,
            self.runtime_execution_profile_fingerprint,
            self.overlay_provider_id,
            self.overlay_provider_fingerprint,
            self.runtime_runner_fingerprint,
            self.evaluator_fingerprint,
            self.session_id,
            self.causal_budget_id,
        )

    def immutable_identity(self) -> tuple[object, ...]:
        return (
            *self.replay_identity(),
            self.created_at,
        )


class MemoryInterventionTrialOutcome(_ExecutionModel):
    """Bounded executor result assembled from existing evidence contracts."""

    execution: MemoryInterventionExecutionRecord
    receipt: MemoryInterventionReceipt | None = None
    eval_result: EvalTrialResult | None = None
    snapshot_result: AgentSnapshotResultBinding | None = None
    binding: MemoryInterventionTrialBinding | None = None

    @field_validator("execution", mode="before")
    @classmethod
    def copy_execution(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionExecutionRecord)

    @field_validator("receipt", mode="before")
    @classmethod
    def copy_receipt(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, MemoryInterventionReceipt)

    @field_validator("eval_result", mode="before")
    @classmethod
    def copy_eval_result(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, EvalTrialResult)

    @field_validator("snapshot_result", mode="before")
    @classmethod
    def copy_snapshot_result(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, AgentSnapshotResultBinding)

    @field_validator("binding", mode="before")
    @classmethod
    def copy_binding(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, MemoryInterventionTrialBinding)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        expected = self.execution
        links = (
            (self.receipt, expected.receipt_fingerprint),
            (self.snapshot_result, expected.snapshot_result_fingerprint),
            (self.binding, expected.final_binding_fingerprint),
        )
        for value, fingerprint in links:
            if (value is None) != (fingerprint is None):
                raise ValueError("Execution outcome evidence is incomplete.")
            if value is not None and value.fingerprint != fingerprint:
                raise ValueError("Execution outcome evidence fingerprint changed.")
        if self.eval_result is not None:
            if expected.eval_result_revision is None:
                raise ValueError("Eval result has no durable execution revision.")
            if (
                memory_intervention_eval_result_revision(self.eval_result)
                != expected.eval_result_revision
            ):
                raise ValueError("Eval result revision changed after durable publication.")
        if expected.status is MemoryInterventionExecutionStatus.COMPLETED and self.binding is None:
            raise ValueError("Completed execution requires its final trial binding.")
        return self


class MemoryInterventionExecutionConflict(RuntimeError):
    pass


class MemoryInterventionRuntimeOwnershipResult(_ExecutionModel):
    """One store-authenticated runtime-dispatch ownership observation."""

    execution: MemoryInterventionExecutionRecord
    ownership: DurableOperationOwnershipResult

    @field_validator("execution", mode="before")
    @classmethod
    def copy_execution(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionExecutionRecord)

    @field_validator("ownership", mode="before")
    @classmethod
    def copy_ownership(cls, value: object) -> object:
        return revalidate_model_input(value, DurableOperationOwnershipResult)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if (
            self.ownership.disposition is not DurableOperationOwnershipDisposition.INDETERMINATE
            and self.execution.runtime_dispatch_ownership != self.ownership.ownership
        ):
            raise ValueError("Ownership result conflicts with the execution record.")
        return self


def _transition_runtime_dispatch_ownership(
    current: MemoryInterventionExecutionRecord,
    request: DurableOperationOwnershipTransition,
    *,
    store_now: datetime,
) -> MemoryInterventionRuntimeOwnershipResult:
    outcome = transition_durable_operation_ownership(
        current.runtime_dispatch_ownership,
        request,
        store_now=store_now,
        operation_active=(
            current.status is MemoryInterventionExecutionStatus.ACTIVE
            and current.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
        ),
    )
    desired = current
    if outcome.ownership != current.runtime_dispatch_ownership:
        desired = MemoryInterventionExecutionRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "revision": current.revision + 1,
                # Feature timestamps remain monotonic bookkeeping, not lease
                # authority. A skewed worker timestamp must not stop the store
                # from deciding ownership with its own transaction time.
                "updated_at": max(current.updated_at, outcome.observed_at),
                "runtime_dispatch_ownership": outcome.ownership,
            }
        )
        _validate_transition(
            current,
            desired,
            runtime_ownership_transition=True,
        )
    return MemoryInterventionRuntimeOwnershipResult(
        execution=desired,
        ownership=outcome,
    )


def _runtime_dispatch_ownership_matches(
    actual: DurableOperationOwnership | None,
    expected: DurableOperationOwnership | DurableOperationOwnershipTransition,
) -> bool:
    """Match the exact logical claim while allowing a store-stamped lease."""

    return (
        type(actual) is DurableOperationOwnership
        and type(expected)
        in {
            DurableOperationOwnership,
            DurableOperationOwnershipTransition,
        }
        and actual.state is DurableOperationOwnershipState.ACTIVE
        and actual.operation_id == expected.operation_id
        and actual.claim_id == expected.claim_id
        and actual.owner_id == expected.owner_id
        and (expected.generation is None or actual.generation == expected.generation)
    )


class _MemoryInterventionRuntimeTimeoutObserved(Exception):
    """In-process handoff that durably records timeout before evidence reads."""

    def __init__(
        self,
        completion: Callable[[], Awaitable[MemoryInterventionRuntimeResult]],
    ) -> None:
        super().__init__("Canonical memory-intervention runtime timed out.")
        self._completion = completion

    async def complete(self) -> MemoryInterventionRuntimeResult:
        return await self._completion()


class MemoryInterventionExecutionStore(ABC):
    @abstractmethod
    async def begin(
        self,
        record: MemoryInterventionExecutionRecord,
    ) -> MemoryInterventionExecutionRecord:
        """Create one execution identity or return its exact replay."""

    @abstractmethod
    async def load(self, execution_id: str) -> MemoryInterventionExecutionRecord | None:
        pass

    @abstractmethod
    async def compare_and_set(
        self,
        expected: MemoryInterventionExecutionRecord,
        desired: MemoryInterventionExecutionRecord,
    ) -> MemoryInterventionExecutionRecord:
        """Advance one exact execution revision atomically.

        A session-bound control or runtime-result publication must also verify
        that the embedded dispatch ownership is live using time sampled inside
        this same write boundary.
        """

    async def transition_runtime_dispatch_ownership(
        self,
        execution_id: str,
        request: DurableOperationOwnershipTransition,
    ) -> MemoryInterventionRuntimeOwnershipResult:
        """Apply one ownership transition using time sampled by this store.

        Custom stores must compare and persist the embedded ownership in the
        same transaction that samples ``store_now``.  There is deliberately no
        lossy compare-and-set fallback: an unported store fails before runtime
        dispatch while remaining constructible for an orderly upgrade.
        """

        raise NotImplementedError(
            "The execution store does not implement atomic runtime ownership."
        )


def _copy_record(record: MemoryInterventionExecutionRecord) -> MemoryInterventionExecutionRecord:
    return MemoryInterventionExecutionRecord.model_validate(record.model_dump(mode="json"))


def _validated_store_record(
    value: object,
    *,
    operation: str,
    allow_missing: bool = False,
) -> MemoryInterventionExecutionRecord | None:
    if value is None and allow_missing:
        return None
    if type(value) is not MemoryInterventionExecutionRecord:
        raise MemoryInterventionExecutionConflict(
            f"Execution store returned invalid {operation} evidence."
        )
    return _copy_record(value)


def _validate_transition(
    expected: MemoryInterventionExecutionRecord,
    desired: MemoryInterventionExecutionRecord,
    *,
    runtime_ownership_transition: bool = False,
) -> None:
    if expected.execution_id != desired.execution_id:
        raise MemoryInterventionExecutionConflict("Execution identity changed.")
    if expected.immutable_identity() != desired.immutable_identity():
        raise MemoryInterventionExecutionConflict("Immutable execution authority changed.")
    if desired.revision != expected.revision + 1:
        raise MemoryInterventionExecutionConflict("Execution revision is not the next revision.")
    if expected.status is not MemoryInterventionExecutionStatus.ACTIVE:
        raise MemoryInterventionExecutionConflict("Terminal execution cannot advance.")
    phase_delta = _PHASE_ORDER[desired.phase] - _PHASE_ORDER[expected.phase]
    if phase_delta not in {0, 1}:
        raise MemoryInterventionExecutionConflict(
            "Execution transition must retain or advance exactly one phase."
        )
    cancellation_authority_advanced = (
        expected.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
        and desired.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
        and not expected.runtime_cancellation_observed
        and desired.runtime_cancellation_observed
    )
    timeout_authority_advanced = (
        expected.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
        and desired.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
        and not expected.runtime_timeout_observed
        and desired.runtime_timeout_observed
    )
    dispatch_ownership_changed = (
        desired.runtime_dispatch_ownership != expected.runtime_dispatch_ownership
    )
    dispatch_authority_advanced = (
        expected.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
        and desired.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
        and dispatch_ownership_changed
    )
    if dispatch_ownership_changed and not runtime_ownership_transition:
        raise MemoryInterventionExecutionConflict(
            "Runtime dispatch ownership must use the atomic ownership transition."
        )
    if phase_delta == 1 and dispatch_ownership_changed:
        raise MemoryInterventionExecutionConflict(
            "Runtime dispatch ownership cannot change while execution phase advances."
        )
    if (
        phase_delta == 0
        and desired.status is MemoryInterventionExecutionStatus.ACTIVE
        and not cancellation_authority_advanced
        and not timeout_authority_advanced
        and not dispatch_authority_advanced
    ):
        raise MemoryInterventionExecutionConflict(
            "An active execution cannot publish an empty same-phase transition."
        )
    if expected.runtime_cancellation_observed and not desired.runtime_cancellation_observed:
        raise MemoryInterventionExecutionConflict(
            "Durable runtime cancellation authority cannot be removed."
        )
    if expected.runtime_timeout_observed and not desired.runtime_timeout_observed:
        raise MemoryInterventionExecutionConflict(
            "Durable runtime timeout authority cannot be removed."
        )
    if (
        not expected.runtime_cancellation_observed
        and desired.runtime_cancellation_observed
        and not cancellation_authority_advanced
    ):
        raise MemoryInterventionExecutionConflict(
            "Runtime cancellation authority must be recorded at the session boundary."
        )
    if (
        not expected.runtime_timeout_observed
        and desired.runtime_timeout_observed
        and not timeout_authority_advanced
    ):
        raise MemoryInterventionExecutionConflict(
            "Runtime timeout authority must be recorded at the session boundary."
        )
    if phase_delta == 1 and (
        desired.status is not MemoryInterventionExecutionStatus.ACTIVE
        and desired.phase is not MemoryInterventionExecutionPhase.FINALIZED
    ):
        raise MemoryInterventionExecutionConflict(
            "Only a finalized execution may become terminal while advancing phases."
        )
    evidence_fields = (
        "materialization_fingerprint",
        "trial_binding_fingerprint",
        "operation_fingerprint",
        "receipt_fingerprint",
        "runtime_session_create_claim",
        "runtime_deadline_at",
        "runtime_evidence_fingerprint",
        "runtime_result_fingerprint",
        "runtime_result_payload",
        "eval_result_revision",
        "snapshot_result_fingerprint",
        "final_binding_fingerprint",
    )
    if any(
        getattr(expected, field) is not None and getattr(desired, field) != getattr(expected, field)
        for field in evidence_fields
    ):
        raise MemoryInterventionExecutionConflict(
            "Previously committed execution evidence cannot change."
        )


def _is_runtime_dispatch_publication(
    expected: MemoryInterventionExecutionRecord,
    desired: MemoryInterventionExecutionRecord,
) -> bool:
    publishes_control_authority = (
        expected.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
        and desired.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
        and (
            desired.runtime_cancellation_observed != expected.runtime_cancellation_observed
            or desired.runtime_timeout_observed != expected.runtime_timeout_observed
        )
    )
    publishes_runtime_result = (
        expected.phase is MemoryInterventionExecutionPhase.SESSION_BOUND
        and desired.phase is MemoryInterventionExecutionPhase.RUNTIME_TERMINAL
    )
    return publishes_control_authority or publishes_runtime_result


def _validate_runtime_dispatch_publication(
    expected: MemoryInterventionExecutionRecord,
    desired: MemoryInterventionExecutionRecord,
    *,
    store_now: datetime,
) -> None:
    """Fence owner-authored evidence at the store's atomic write boundary."""

    if not _is_runtime_dispatch_publication(expected, desired):
        return
    ownership = expected.runtime_dispatch_ownership
    if (
        ownership is None
        or ownership.state is not DurableOperationOwnershipState.ACTIVE
        or ownership.lease_expires_at is None
        or store_now >= ownership.lease_expires_at
    ):
        raise MemoryInterventionExecutionConflict(
            "Runtime publication requires live store-authoritative dispatch ownership."
        )


class InMemoryMemoryInterventionExecutionStore(MemoryInterventionExecutionStore):
    def __init__(self, *, ownership_clock: Callable[[], datetime] | None = None) -> None:
        self._records: dict[str, MemoryInterventionExecutionRecord] = {}
        self._lock = asyncio.Lock()
        self._ownership_clock = utc_clock(ownership_clock)

    async def begin(
        self,
        record: MemoryInterventionExecutionRecord,
    ) -> MemoryInterventionExecutionRecord:
        candidate = _copy_record(record)
        async with self._lock:
            existing = self._records.get(candidate.execution_id)
            if existing is None:
                self._records[candidate.execution_id] = candidate
                return _copy_record(candidate)
            if existing.replay_identity() != candidate.replay_identity():
                raise MemoryInterventionExecutionConflict(
                    "Execution identity is already bound to another request."
                )
            return _copy_record(existing)

    async def load(self, execution_id: str) -> MemoryInterventionExecutionRecord | None:
        execution_id = _sha256(execution_id, "execution_id")
        async with self._lock:
            existing = self._records.get(execution_id)
            return None if existing is None else _copy_record(existing)

    async def compare_and_set(
        self,
        expected: MemoryInterventionExecutionRecord,
        desired: MemoryInterventionExecutionRecord,
    ) -> MemoryInterventionExecutionRecord:
        expected = _copy_record(expected)
        desired = _copy_record(desired)
        _validate_transition(expected, desired)
        async with self._lock:
            existing = self._records.get(expected.execution_id)
            if existing is None or existing != expected:
                raise MemoryInterventionExecutionConflict("Execution revision changed.")
            if _is_runtime_dispatch_publication(expected, desired):
                _validate_runtime_dispatch_publication(
                    expected,
                    desired,
                    store_now=self._ownership_clock(),
                )
            self._records[expected.execution_id] = desired
            return _copy_record(desired)

    async def transition_runtime_dispatch_ownership(
        self,
        execution_id: str,
        request: DurableOperationOwnershipTransition,
    ) -> MemoryInterventionRuntimeOwnershipResult:
        execution_id = _sha256(execution_id, "execution_id")
        if type(request) is not DurableOperationOwnershipTransition:
            raise TypeError("request must be a DurableOperationOwnershipTransition.")
        async with self._lock:
            current = self._records.get(execution_id)
            if current is None:
                raise MemoryInterventionExecutionConflict("Execution is unavailable.")
            result = _transition_runtime_dispatch_ownership(
                current,
                request,
                store_now=self._ownership_clock(),
            )
            if result.execution != current:
                self._records[execution_id] = result.execution
            return MemoryInterventionRuntimeOwnershipResult.model_validate(
                result.model_dump(mode="python")
            )


class SQLiteMemoryInterventionExecutionStore(MemoryInterventionExecutionStore):
    """SQLite execution journal using one exact JSON document per CAS revision."""

    def __init__(
        self,
        path: str | Path,
        *,
        ownership_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path)
        self._schema_lock = threading.Lock()
        self._schema_ready = False
        self._ownership_clock = utc_clock(ownership_clock)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cayu_memory_intervention_executions (
                        execution_id TEXT PRIMARY KEY,
                        revision INTEGER NOT NULL,
                        document TEXT NOT NULL
                    )
                    """
                )
            self._schema_ready = True

    async def begin(
        self,
        record: MemoryInterventionExecutionRecord,
    ) -> MemoryInterventionExecutionRecord:
        candidate = _copy_record(record)
        document = candidate.model_dump_json()
        return await asyncio.to_thread(self._begin_sync, candidate, document)

    def _begin_sync(
        self,
        candidate: MemoryInterventionExecutionRecord,
        document: str,
    ) -> MemoryInterventionExecutionRecord:
        self._ensure_schema()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision, document FROM cayu_memory_intervention_executions "
                "WHERE execution_id = ?",
                (candidate.execution_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO cayu_memory_intervention_executions "
                    "(execution_id, revision, document) VALUES (?, ?, ?)",
                    (candidate.execution_id, candidate.revision, document),
                )
                return candidate
            existing = MemoryInterventionExecutionRecord.model_validate_json(row[1])
            if row[0] != existing.revision:
                raise MemoryInterventionExecutionConflict(
                    "Stored execution revision conflicts with its durable document."
                )
            if existing.replay_identity() != candidate.replay_identity():
                raise MemoryInterventionExecutionConflict(
                    "Execution identity is already bound to another request."
                )
            return existing

    async def load(self, execution_id: str) -> MemoryInterventionExecutionRecord | None:
        execution_id = _sha256(execution_id, "execution_id")
        return await asyncio.to_thread(self._load_sync, execution_id)

    def _load_sync(self, execution_id: str) -> MemoryInterventionExecutionRecord | None:
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT revision, document FROM cayu_memory_intervention_executions "
                "WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
        if row is None:
            return None
        record = MemoryInterventionExecutionRecord.model_validate_json(row[1])
        if row[0] != record.revision:
            raise MemoryInterventionExecutionConflict(
                "Stored execution revision conflicts with its durable document."
            )
        return record

    async def compare_and_set(
        self,
        expected: MemoryInterventionExecutionRecord,
        desired: MemoryInterventionExecutionRecord,
    ) -> MemoryInterventionExecutionRecord:
        expected = _copy_record(expected)
        desired = _copy_record(desired)
        _validate_transition(expected, desired)
        return await asyncio.to_thread(self._compare_and_set_sync, expected, desired)

    def _compare_and_set_sync(
        self,
        expected: MemoryInterventionExecutionRecord,
        desired: MemoryInterventionExecutionRecord,
    ) -> MemoryInterventionExecutionRecord:
        self._ensure_schema()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision, document FROM cayu_memory_intervention_executions "
                "WHERE execution_id = ?",
                (expected.execution_id,),
            ).fetchone()
            if row is None:
                raise MemoryInterventionExecutionConflict("Execution is unavailable.")
            existing = MemoryInterventionExecutionRecord.model_validate_json(row[1])
            if row[0] != expected.revision or existing != expected:
                raise MemoryInterventionExecutionConflict("Execution revision changed.")
            if _is_runtime_dispatch_publication(expected, desired):
                _validate_runtime_dispatch_publication(
                    expected,
                    desired,
                    store_now=self._ownership_clock(),
                )
            updated = connection.execute(
                "UPDATE cayu_memory_intervention_executions "
                "SET revision = ?, document = ? "
                "WHERE execution_id = ? AND revision = ?",
                (
                    desired.revision,
                    desired.model_dump_json(),
                    desired.execution_id,
                    expected.revision,
                ),
            )
            if updated.rowcount != 1:
                raise MemoryInterventionExecutionConflict("Execution revision changed.")
        return desired

    async def transition_runtime_dispatch_ownership(
        self,
        execution_id: str,
        request: DurableOperationOwnershipTransition,
    ) -> MemoryInterventionRuntimeOwnershipResult:
        execution_id = _sha256(execution_id, "execution_id")
        if type(request) is not DurableOperationOwnershipTransition:
            raise TypeError("request must be a DurableOperationOwnershipTransition.")
        return await asyncio.to_thread(
            self._transition_runtime_dispatch_ownership_sync,
            execution_id,
            request,
        )

    def _transition_runtime_dispatch_ownership_sync(
        self,
        execution_id: str,
        request: DurableOperationOwnershipTransition,
    ) -> MemoryInterventionRuntimeOwnershipResult:
        self._ensure_schema()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision, document FROM cayu_memory_intervention_executions "
                "WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
            if row is None:
                raise MemoryInterventionExecutionConflict("Execution is unavailable.")
            current = MemoryInterventionExecutionRecord.model_validate_json(row[1])
            if row[0] != current.revision:
                raise MemoryInterventionExecutionConflict(
                    "Stored execution revision conflicts with its durable document."
                )
            result = _transition_runtime_dispatch_ownership(
                current,
                request,
                store_now=self._ownership_clock(),
            )
            if result.execution != current:
                updated = connection.execute(
                    "UPDATE cayu_memory_intervention_executions "
                    "SET revision = ?, document = ? "
                    "WHERE execution_id = ? AND revision = ?",
                    (
                        result.execution.revision,
                        result.execution.model_dump_json(),
                        execution_id,
                        current.revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise MemoryInterventionExecutionConflict("Execution revision changed.")
        return result


@dataclass(frozen=True, slots=True)
class MemoryInterventionRuntimeView:
    """Ephemeral executable view authenticated by one durable materialization."""

    materialization_fingerprint: str
    memory_overlay_fingerprint: str
    state_scope_id: str
    knowledge_store: KnowledgeStore
    knowledge_access_scope: KnowledgeAccessScope
    isolation_authority: MemoryInterventionIsolationAuthority
    trial_recall_policy: AutomaticRecallPolicy
    receipt: MemoryInterventionReceipt

    def __post_init__(self) -> None:
        _sha256(self.materialization_fingerprint, "materialization_fingerprint")
        _sha256(self.memory_overlay_fingerprint, "memory_overlay_fingerprint")
        _clean(self.state_scope_id, "state_scope_id")
        if not isinstance(self.knowledge_store, KnowledgeStore):
            raise TypeError("knowledge_store must be a KnowledgeStore.")
        if type(self.knowledge_access_scope) is not KnowledgeAccessScope:
            raise TypeError("knowledge_access_scope must be an exact KnowledgeAccessScope.")
        if type(self.isolation_authority) is not MemoryInterventionIsolationAuthority:
            raise TypeError(
                "isolation_authority must be an exact MemoryInterventionIsolationAuthority."
            )
        if type(self.trial_recall_policy) is not AutomaticRecallPolicy:
            raise TypeError("trial_recall_policy must be an exact AutomaticRecallPolicy.")
        if type(self.receipt) is not MemoryInterventionReceipt:
            raise TypeError("receipt must be an exact MemoryInterventionReceipt.")
        scope = copy_knowledge_access_scope(self.knowledge_access_scope)
        bound_scope = self.knowledge_store.bound_access_scope()
        if bound_scope is not None and bound_scope != scope:
            raise ValueError("The intervention store is bound to another access scope.")
        receipt = MemoryInterventionReceipt.model_validate(self.receipt.model_dump(mode="python"))
        isolation = MemoryInterventionIsolationAuthority.model_validate(
            self.isolation_authority.model_dump(mode="python")
        )
        policy = AutomaticRecallPolicy.model_validate(
            self.trial_recall_policy.model_dump(mode="python")
        )
        if (
            receipt.materialization_fingerprint != self.materialization_fingerprint
            or receipt.memory_overlay_fingerprint != self.memory_overlay_fingerprint
            or receipt.state_scope_id != self.state_scope_id
            or isolation.materialization_fingerprint != self.materialization_fingerprint
            or isolation.memory_overlay_fingerprint != self.memory_overlay_fingerprint
            or isolation.state_scope_id != self.state_scope_id
        ):
            raise ValueError("The intervention view conflicts with its effect receipt.")
        object.__setattr__(self, "knowledge_access_scope", scope)
        object.__setattr__(self, "isolation_authority", isolation)
        object.__setattr__(self, "trial_recall_policy", policy)
        object.__setattr__(self, "receipt", receipt)


class MemoryInterventionOverlayProvider(ABC):
    """Application-owned opener and effect authority for one memory overlay."""

    provider_id: str
    execution_profile_fingerprint: str

    @abstractmethod
    async def apply(
        self,
        *,
        spec: MemoryInterventionSpec,
        operation: MemoryInterventionOperation,
        materialization: AgentSnapshotMaterialization,
        trial: AgentSnapshotTrialBinding,
    ) -> MemoryInterventionRuntimeView:
        """Apply one precommitted effect exactly once and open its runtime view."""

    @abstractmethod
    async def recover(
        self,
        *,
        spec: MemoryInterventionSpec,
        operation: MemoryInterventionOperation,
        materialization: AgentSnapshotMaterialization,
        trial: AgentSnapshotTrialBinding,
    ) -> MemoryInterventionRuntimeView:
        """Complete or reconcile the exact precommitted effect operation.

        Recovery can begin after the durable operation claim but before the
        first application call.  Implementations must therefore use the
        operation identity as their idempotency authority: apply the effect if
        no receipt exists, otherwise return the exact prior receipt, without
        repeating a candidate effect.
        """


class MemoryInterventionRuntimeRunner(ABC):
    """Typed internal runtime entrance; caller metadata cannot construct it.

    The executor supplies only a domain-derived session-reference key. A runner
    never receives the root key that authenticates the durable trial request.
    """

    execution_profile_fingerprint: str

    def required_execution_profile_fingerprint(
        self,
        spec: MemoryInterventionSpec,
    ) -> str:
        if type(spec) is not MemoryInterventionSpec:
            raise TypeError("spec must be an exact MemoryInterventionSpec.")
        return spec.execution_profile_fingerprint

    @abstractmethod
    async def run(
        self,
        *,
        request: MemoryInterventionTrialRequest,
        execution: MemoryInterventionExecutionRecord,
        starting_execution_profile: AgentSnapshotExecutionProfileRef,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
        view: MemoryInterventionRuntimeView,
        reference_key: RuntimeSessionCreateClaimReferenceKey,
    ) -> MemoryInterventionRuntimeResult:
        pass

    @abstractmethod
    async def recover(
        self,
        *,
        request: MemoryInterventionTrialRequest,
        execution: MemoryInterventionExecutionRecord,
        starting_execution_profile: AgentSnapshotExecutionProfileRef,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
        view: MemoryInterventionRuntimeView,
        reference_key: RuntimeSessionCreateClaimReferenceKey,
    ) -> MemoryInterventionRuntimeResult:
        """Complete or reconcile the exact precommitted runtime session.

        The session claim is durable before ``run`` is entered, so recovery
        must safely start an undispatched session or reconnect an already
        dispatched one using the runtime-owned session identity.
        """


class MemoryInterventionRuntimeApplicationFactory(ABC):
    """Application-owned constructor for one isolated canonical Cayu runtime.

    Implementations register the frozen agent/provider/evaluation authority and
    must install ``view.knowledge_store`` as the selected environment's exact
    knowledge store.  The concrete runner validates those facts and the full
    runtime execution profile before dispatch.
    """

    factory_id: str
    execution_profile_fingerprint: str

    def expected_execution_profile_fingerprint(
        self,
        spec: MemoryInterventionSpec,
    ) -> str:
        """Return the app-owned exact profile for this authorized recall policy.

        Unchanged recall uses the verified snapshot profile. Applications that
        support ``automatic_recall_off`` or another policy-changing fixed
        variant must override this method and bind that mapping into
        ``execution_profile_fingerprint``.
        """

        if type(spec) is not MemoryInterventionSpec:
            raise TypeError("spec must be an exact MemoryInterventionSpec.")
        if spec.trial_recall_policy_fingerprint != spec.starting_recall_policy_fingerprint:
            raise MemoryInterventionExecutionConflict(
                "The runtime factory has no profile for the trial recall policy."
            )
        return spec.execution_profile_fingerprint

    @abstractmethod
    async def create(
        self,
        *,
        request: MemoryInterventionTrialRequest,
        execution: MemoryInterventionExecutionRecord,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
        view: MemoryInterventionRuntimeView,
    ) -> object:
        """Return a newly configured ``CayuApp`` for the exact runtime view."""


def _require_recall_only_execution_profile_change(
    *,
    starting: AgentSnapshotExecutionProfileRef,
    runtime: AgentSnapshotExecutionProfileRef,
    recall_policy_changed: bool,
) -> None:
    starting = AgentSnapshotExecutionProfileRef.model_validate(starting.model_dump(mode="json"))
    runtime = AgentSnapshotExecutionProfileRef.model_validate(runtime.model_dump(mode="json"))
    if starting.schema_version != runtime.schema_version:
        raise MemoryInterventionExecutionConflict(
            "Memory intervention changed the execution-profile schema."
        )
    starting_by_name = {component.name: component for component in starting.components}
    runtime_by_name = {component.name: component for component in runtime.components}
    if starting_by_name.keys() != runtime_by_name.keys():
        raise MemoryInterventionExecutionConflict(
            "Memory intervention changed the execution-profile component set."
        )
    changed = tuple(
        name for name in sorted(starting_by_name) if starting_by_name[name] != runtime_by_name[name]
    )
    expected = (
        (ExecutionProfileComponentClass.AUTOMATIC_RECALL.value,) if recall_policy_changed else ()
    )
    if changed != expected:
        raise MemoryInterventionExecutionConflict(
            "Memory intervention changed execution authority outside automatic recall."
        )


def _interrupted_runtime_disposition(
    *,
    cancellation_observed: bool,
) -> AgentSnapshotTerminalDisposition:
    if type(cancellation_observed) is not bool:
        raise TypeError("cancellation_observed must be a bool.")
    if cancellation_observed:
        return AgentSnapshotTerminalDisposition.CANCELLED
    return AgentSnapshotTerminalDisposition.OUTCOME_UNKNOWN


def _memory_intervention_runtime_request(
    request: MemoryInterventionTrialRequest,
    execution: MemoryInterventionExecutionRecord,
    trial: AgentSnapshotTrialBinding,
) -> RunRequest:
    """Rebuild the exact runtime-owned request represented by the journal."""

    metadata = dict(request.run_request.metadata)
    metadata.update(trial.session_metadata())
    runtime_request = copy_run_request(request.run_request).model_copy(
        update={
            "session_id": execution.session_id,
            "causal_budget_id": execution.causal_budget_id,
            "metadata": metadata,
        }
    )
    return run_request_with_runtime_generated_authority(
        runtime_request,
        "session_id",
        "causal_budget_id",
    )


class CayuMemoryInterventionRuntimeRunner(MemoryInterventionRuntimeRunner):
    """Run a fixed intervention through Cayu's ordinary runtime pipeline."""

    def __init__(
        self,
        factory: MemoryInterventionRuntimeApplicationFactory,
        *,
        attribution_bounds: MemoryAttributionBounds | None = None,
    ) -> None:
        if not isinstance(factory, MemoryInterventionRuntimeApplicationFactory):
            raise TypeError("factory must implement MemoryInterventionRuntimeApplicationFactory.")
        self.factory = factory
        self._factory_id = _clean(factory.factory_id, "factory.factory_id")
        self._factory_fingerprint = _sha256(
            factory.execution_profile_fingerprint,
            "factory.execution_profile_fingerprint",
        )
        requested_bounds = MemoryAttributionBounds.model_validate(
            (attribution_bounds or MemoryAttributionBounds()).model_dump(mode="python")
        )
        policy_bounds = standard_eval_memory_attribution_bounds()
        self._attribution_bounds = MemoryAttributionBounds(
            max_receipts=min(requested_bounds.max_receipts, policy_bounds.max_receipts),
            max_exposures=min(requested_bounds.max_exposures, policy_bounds.max_exposures),
            max_items=min(requested_bounds.max_items, policy_bounds.max_items),
            max_source_bytes=min(
                requested_bounds.max_source_bytes,
                policy_bounds.max_source_bytes,
            ),
            max_projection_bytes=min(
                requested_bounds.max_projection_bytes,
                policy_bounds.max_projection_bytes,
                MEMORY_INTERVENTION_ATTRIBUTION_MAX_PROJECTION_BYTES,
            ),
        )
        self.execution_profile_fingerprint = _content_sha256(
            {
                "kind": "cayu_memory_intervention_runtime",
                "version": 2,
                "factory_id": self._factory_id,
                "factory_fingerprint": self._factory_fingerprint,
                "attribution_bounds": self._attribution_bounds.model_dump(mode="json"),
            },
            "memory intervention runtime runner",
        )

    def required_execution_profile_fingerprint(
        self,
        spec: MemoryInterventionSpec,
    ) -> str:
        return _sha256(
            self.factory.expected_execution_profile_fingerprint(spec),
            "factory expected execution profile",
        )

    async def run(
        self,
        *,
        request: MemoryInterventionTrialRequest,
        execution: MemoryInterventionExecutionRecord,
        starting_execution_profile: AgentSnapshotExecutionProfileRef,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
        view: MemoryInterventionRuntimeView,
        reference_key: RuntimeSessionCreateClaimReferenceKey,
    ) -> MemoryInterventionRuntimeResult:
        (
            app,
            runtime_request,
            expected_profile,
            registered_environment,
            policy,
            session_create_claim,
        ) = await self._prepare_application(
            request=request,
            execution=execution,
            starting_execution_profile=starting_execution_profile,
            trial=trial,
            operation=operation,
            view=view,
            reference_key=reference_key,
        )
        return await self._execute(
            app=app,
            request=request,
            runtime_request=runtime_request,
            expected_profile=expected_profile,
            expected_registered_environment=registered_environment,
            expected_context_policy=policy,
            session_create_claim=session_create_claim,
            execution=execution,
            reference_key=reference_key,
        )

    async def recover(
        self,
        *,
        request: MemoryInterventionTrialRequest,
        execution: MemoryInterventionExecutionRecord,
        starting_execution_profile: AgentSnapshotExecutionProfileRef,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
        view: MemoryInterventionRuntimeView,
        reference_key: RuntimeSessionCreateClaimReferenceKey,
    ) -> MemoryInterventionRuntimeResult:
        (
            app,
            runtime_request,
            expected_profile,
            registered_environment,
            policy,
            session_create_claim,
        ) = await self._prepare_application(
            request=request,
            execution=execution,
            starting_execution_profile=starting_execution_profile,
            trial=trial,
            operation=operation,
            view=view,
            reference_key=reference_key,
        )
        session = await app.session_store.load(execution.session_id)
        if session is None:
            if execution.runtime_timeout_observed:
                return _runtime_timeout_before_session_result(
                    execution,
                    effective_attribution_bounds=self._attribution_bounds,
                    source_alias=_memory_intervention_source_alias(
                        app,
                        execution.session_id,
                    ),
                )
            return await self._execute(
                app=app,
                request=request,
                runtime_request=runtime_request,
                expected_profile=expected_profile,
                expected_registered_environment=registered_environment,
                expected_context_policy=policy,
                session_create_claim=session_create_claim,
                execution=execution,
                reference_key=reference_key,
            )
        session = await _authenticate_intervention_session(
            app,
            session,
            session_create_claim,
            reference=execution.runtime_session_create_claim,
            request=runtime_request,
            operation_id=execution.execution_id,
            reference_key=reference_key,
        )
        if session.status not in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }:
            await app._recover_incomplete_session_private(
                IncompleteSessionRecoveryRequest(
                    session_id=execution.session_id,
                    reason="memory_intervention_runtime_recovery",
                    metadata={
                        "execution_id": execution.execution_id,
                        "operation_id": operation.operation_id,
                    },
                )
            )
            session = await app.session_store.load(execution.session_id)
            session = await _authenticate_intervention_session(
                app,
                session,
                session_create_claim,
                reference=execution.runtime_session_create_claim,
                request=runtime_request,
                operation_id=execution.execution_id,
                reference_key=reference_key,
            )
        return await self._collect_result(
            app=app,
            expected_session=session,
            observed_events=(),
            timeout_expired=execution.runtime_timeout_observed,
            cancellation_observed=execution.runtime_cancellation_observed,
        )

    async def _prepare_application(
        self,
        *,
        request: MemoryInterventionTrialRequest,
        execution: MemoryInterventionExecutionRecord,
        starting_execution_profile: AgentSnapshotExecutionProfileRef,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
        view: MemoryInterventionRuntimeView,
        reference_key: RuntimeSessionCreateClaimReferenceKey,
    ):
        from cayu.runtime.app import CayuApp
        from cayu.runtime.memory_context import AutomaticRecallContextPolicy

        if type(reference_key) is not RuntimeSessionCreateClaimReferenceKey:
            raise TypeError("reference_key must be a RuntimeSessionCreateClaimReferenceKey.")
        if reference_key.key_id != execution.request_key_id:
            raise MemoryInterventionExecutionConflict(
                "Runtime session reference key conflicts with execution authority."
            )
        if (
            _clean(self.factory.factory_id, "factory.factory_id") != self._factory_id
            or _sha256(
                self.factory.execution_profile_fingerprint,
                "factory.execution_profile_fingerprint",
            )
            != self._factory_fingerprint
        ):
            raise MemoryInterventionExecutionConflict(
                "Memory intervention runtime factory identity changed."
            )
        runtime_request = _memory_intervention_runtime_request(
            request,
            execution,
            trial,
        )
        session_create_reference = execution.runtime_session_create_claim
        if session_create_reference is None:
            raise MemoryInterventionExecutionConflict(
                "Runtime execution is missing its durable session claim reference."
            )
        try:
            runtime_request, session_create_claim = (
                run_request_with_runtime_session_create_claim_reference(
                    runtime_request,
                    session_create_reference,
                    operation_id=execution.execution_id,
                    key=reference_key,
                )
            )
        except ValueError:
            raise MemoryInterventionExecutionConflict(
                "Runtime session claim reference conflicts with execution authority."
            ) from None
        value = await self.factory.create(
            request=request,
            execution=execution,
            trial=trial,
            operation=operation,
            view=view,
        )
        if type(value) is not CayuApp:
            raise MemoryInterventionExecutionConflict(
                "Memory intervention runtime factory returned an invalid application."
            )
        app = value
        registered_environment = app._get_registered_environment(
            request.run_request.environment_name
        )
        if registered_environment is None or registered_environment.factory is not None:
            raise MemoryInterventionExecutionConflict(
                "Memory intervention runtime requires one concrete isolated environment."
            )
        environment = registered_environment.environment
        if (
            environment.knowledge_store is not view.knowledge_store
            or environment.knowledge_access_scope != view.knowledge_access_scope
        ):
            raise MemoryInterventionExecutionConflict(
                "Memory intervention runtime did not install the exact isolated overlay."
            )
        registered_agent = app._get_registered_agent(request.run_request.agent_name)
        policy = registered_agent.context_policy
        if (
            type(policy) is not AutomaticRecallContextPolicy
            or policy.admission_policy != view.trial_recall_policy
        ):
            raise MemoryInterventionExecutionConflict(
                "Memory intervention runtime did not install the exact canonical recall policy."
            )
        expected_profile = _sha256(
            self.factory.expected_execution_profile_fingerprint(request.spec),
            "factory expected execution profile",
        )
        if expected_profile != execution.runtime_execution_profile_fingerprint:
            raise MemoryInterventionExecutionConflict(
                "Runtime execution profile differs from durable execution authority."
            )
        prepared = await app._session_engine._prepare_initial_run(
            runtime_request,
            admit_session=False,
            store_resolved_existing_session_id=execution.session_id,
        )
        if (
            request.spec.trial_recall_policy_fingerprint
            == request.spec.starting_recall_policy_fingerprint
            and expected_profile != request.spec.execution_profile_fingerprint
        ):
            raise MemoryInterventionExecutionConflict(
                "Unchanged recall cannot replace the verified snapshot profile."
            )
        if prepared is None or prepared.execution_profile.fingerprint != expected_profile:
            raise MemoryInterventionExecutionConflict(
                "Canonical runtime execution profile differs from the verified snapshot."
            )
        _require_recall_only_execution_profile_change(
            starting=starting_execution_profile,
            runtime=execution_profile_snapshot_ref(prepared.execution_profile),
            recall_policy_changed=(
                request.spec.trial_recall_policy_fingerprint
                != request.spec.starting_recall_policy_fingerprint
            ),
        )
        return (
            app,
            runtime_request,
            prepared.execution_profile,
            registered_environment,
            policy,
            session_create_claim,
        )

    async def _execute(
        self,
        *,
        app,
        request,
        runtime_request,
        expected_profile: ExecutionProfileIdentity,
        expected_registered_environment,
        expected_context_policy,
        session_create_claim,
        execution: MemoryInterventionExecutionRecord,
        reference_key: RuntimeSessionCreateClaimReferenceKey,
    ) -> MemoryInterventionRuntimeResult:
        observed: list[RunnerObservedEventIdentity] = []
        remaining_timeout = _remaining_runtime_timeout(request, execution)
        if remaining_timeout <= 0:
            result = _runtime_timeout_before_session_result(
                execution,
                effective_attribution_bounds=self._attribution_bounds,
                source_alias=_memory_intervention_source_alias(
                    app,
                    execution.session_id,
                ),
            )
            raise _MemoryInterventionRuntimeTimeoutObserved(lambda: asyncio.sleep(0, result=result))
        try:
            async with asyncio.timeout(remaining_timeout):
                async for event in app._run_private(
                    runtime_request,
                    expected_execution_profile=expected_profile,
                    expected_registered_environment=expected_registered_environment,
                    expected_context_policy=expected_context_policy,
                ):
                    observed.append(
                        RunnerObservedEventIdentity(
                            session_id=event.session_id,
                            sequence=event_durable_sequence(event),
                            event_type=event.type,
                        )
                    )
        except TimeoutError:
            observed_events = tuple(observed)
            raise _MemoryInterventionRuntimeTimeoutObserved(
                lambda: self._collect_timeout_result(
                    app=app,
                    execution=execution,
                    runtime_request=runtime_request,
                    session_create_claim=session_create_claim,
                    observed_events=observed_events,
                    reference_key=reference_key,
                )
            ) from None
        session = await app.session_store.load(execution.session_id)
        session = await _authenticate_intervention_session(
            app,
            session,
            session_create_claim,
            reference=execution.runtime_session_create_claim,
            request=runtime_request,
            operation_id=execution.execution_id,
            reference_key=reference_key,
        )
        return await self._collect_result(
            app=app,
            expected_session=session,
            observed_events=tuple(observed),
            timeout_expired=False,
            cancellation_observed=False,
        )

    async def _collect_timeout_result(
        self,
        *,
        app,
        execution: MemoryInterventionExecutionRecord,
        runtime_request: RunRequest,
        session_create_claim: object,
        observed_events: tuple[RunnerObservedEventIdentity, ...],
        reference_key: RuntimeSessionCreateClaimReferenceKey,
    ) -> MemoryInterventionRuntimeResult:
        session = await app.session_store.load(execution.session_id)
        if session is None:
            return _runtime_timeout_before_session_result(
                execution,
                effective_attribution_bounds=self._attribution_bounds,
                source_alias=_memory_intervention_source_alias(
                    app,
                    execution.session_id,
                ),
            )
        session = await _authenticate_intervention_session(
            app,
            session,
            session_create_claim,
            reference=execution.runtime_session_create_claim,
            request=runtime_request,
            operation_id=execution.execution_id,
            reference_key=reference_key,
        )
        return await self._collect_result(
            app=app,
            expected_session=session,
            observed_events=observed_events,
            timeout_expired=True,
            cancellation_observed=False,
        )

    async def _collect_result(
        self,
        *,
        app,
        expected_session: Session,
        observed_events: tuple[RunnerObservedEventIdentity, ...],
        timeout_expired: bool,
        cancellation_observed: bool,
    ) -> MemoryInterventionRuntimeResult:
        session = copy_session(expected_session)
        session_id = session.id
        evidence: TerminalSessionEvidence | None = None
        terminal_evidence_limitation: EvalMemoryEvidenceLimitation | None = None
        try:
            evidence = await app.session_store.load_terminal_session_evidence(session_id)
        except TerminalSessionEvidenceError as error:
            if session.status is SessionStatus.INTERRUPTED and observed_events:
                try:
                    evidence = await app.session_store.load_runner_owned_interrupted_evidence(
                        session_id,
                        observed_events=observed_events,
                    )
                except (NotImplementedError, TerminalSessionEvidenceError) as fallback_error:
                    evidence = None
                    terminal_evidence_limitation = _terminal_evidence_limitation(fallback_error)
                except Exception:
                    evidence = None
                    terminal_evidence_limitation = EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED
            else:
                terminal_evidence_limitation = _terminal_evidence_limitation(error)
        except NotImplementedError as error:
            terminal_evidence_limitation = _terminal_evidence_limitation(error)
        except Exception:
            terminal_evidence_limitation = EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED
        if evidence is not None:
            evidence = _validated_memory_intervention_terminal_evidence(
                evidence,
                expected_session=session,
                observed_events=observed_events,
            )
            terminal_evidence_limitation = None
            runtime_material: object = evidence.model_dump(mode="json")
        else:
            terminal_evidence_limitation = (
                terminal_evidence_limitation or EvalMemoryEvidenceLimitation.MISSING
            )
            runtime_material = {
                "session": session.model_dump(mode="json"),
                "terminal_evidence": "unavailable",
                "terminal_evidence_limitation": terminal_evidence_limitation.value,
            }
        usage = await app.get_session_usage(session_id)
        evidence_key = memory_evidence_key(app._request_footprint)
        source_alias = _memory_intervention_source_alias(app, session_id)
        if evidence is None:
            expected_receipt_count = None
            expected_exposure_count = None
            attribution = _unavailable_terminal_memory_attribution()
        else:
            expected_receipt_count, expected_exposure_count = _memory_source_expected_counts(
                record.event for record in evidence.events
            )
            attribution = await project_memory_attribution(
                app.session_store,
                session_id,
                key=evidence_key,
                budget=MemoryAttributionCaptureBudget(bounds=self._attribution_bounds),
            )
        disposition = (
            AgentSnapshotTerminalDisposition.TIMED_OUT
            if timeout_expired
            else {
                SessionStatus.COMPLETED: AgentSnapshotTerminalDisposition.COMPLETED,
                SessionStatus.FAILED: AgentSnapshotTerminalDisposition.FAILED,
                SessionStatus.INTERRUPTED: _interrupted_runtime_disposition(
                    cancellation_observed=cancellation_observed,
                ),
            }.get(session.status, AgentSnapshotTerminalDisposition.OUTCOME_UNKNOWN)
        )
        return MemoryInterventionRuntimeResult(
            session_id=session_id,
            terminal_disposition=disposition,
            runtime_evidence_fingerprint=_content_sha256(
                runtime_material,
                "memory intervention runtime evidence",
            ),
            terminal_evidence_available=evidence is not None,
            terminal_evidence_limitation=terminal_evidence_limitation,
            expected_receipt_count=expected_receipt_count,
            expected_exposure_count=expected_exposure_count,
            effective_attribution_bounds=self._attribution_bounds,
            source_alias=source_alias,
            attribution=attribution,
            usage_fingerprint=_content_sha256(
                usage.model_dump(mode="json"),
                "memory intervention usage evidence",
            ),
        )


class MemoryInterventionRuntimeResult(_ExecutionModel):
    session_id: StrictStr = Field(max_length=512)
    terminal_disposition: AgentSnapshotTerminalDisposition
    runtime_evidence_fingerprint: StrictStr
    terminal_evidence_available: StrictBool
    terminal_evidence_limitation: EvalMemoryEvidenceLimitation | None = None
    expected_receipt_count: StrictInt | None = Field(default=None, ge=0)
    expected_exposure_count: StrictInt | None = Field(default=None, ge=0)
    effective_attribution_bounds: MemoryAttributionBounds
    source_alias: EvalMemorySourceAliasV1 | None = None
    attribution: MemoryAttribution
    usage_fingerprint: StrictStr | None = None
    cost_fingerprint: StrictStr | None = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=512)

    @field_validator("runtime_evidence_fingerprint", "usage_fingerprint", "cost_fingerprint")
    @classmethod
    def validate_evidence(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256(value, info.field_name)

    @field_validator("attribution", mode="before")
    @classmethod
    def copy_attribution(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryAttribution)

    @field_validator("effective_attribution_bounds", mode="before")
    @classmethod
    def copy_effective_attribution_bounds(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryAttributionBounds)

    @field_validator("source_alias", mode="before")
    @classmethod
    def copy_source_alias(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, EvalMemorySourceAliasV1)

    @model_validator(mode="after")
    def validate_terminal_memory_authority(self) -> Self:
        if (self.expected_receipt_count is None) is not (self.expected_exposure_count is None):
            raise ValueError("Terminal memory counts must be present or absent together.")
        counts_available = self.expected_receipt_count is not None
        if self.terminal_evidence_available is not counts_available:
            raise ValueError(
                "Terminal memory counts must be present exactly when terminal evidence is available."
            )
        if self.terminal_evidence_available:
            if self.terminal_evidence_limitation is not None:
                raise ValueError("Available terminal evidence cannot carry a limitation.")
        elif self.terminal_evidence_limitation not in {
            EvalMemoryEvidenceLimitation.STORE_UNSUPPORTED,
            EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED,
            EvalMemoryEvidenceLimitation.MISSING,
            EvalMemoryEvidenceLimitation.CONTRADICTORY_LINEAGE,
            EvalMemoryEvidenceLimitation.DEADLINE_EXPIRED,
        }:
            raise ValueError("Unavailable terminal evidence requires a terminal limitation.")
        policy_bounds = standard_eval_memory_attribution_bounds()
        if self.effective_attribution_bounds.max_receipts > policy_bounds.max_receipts:
            raise ValueError("Effective receipt bound exceeds the eval capture policy.")
        if self.effective_attribution_bounds.max_exposures > policy_bounds.max_exposures:
            raise ValueError("Effective exposure bound exceeds the eval capture policy.")
        if self.effective_attribution_bounds.max_items > policy_bounds.max_items:
            raise ValueError("Effective item bound exceeds the eval capture policy.")
        if self.effective_attribution_bounds.max_source_bytes > policy_bounds.max_source_bytes:
            raise ValueError("Effective source-byte bound exceeds the eval capture policy.")
        if (
            self.effective_attribution_bounds.max_projection_bytes
            > policy_bounds.max_projection_bytes
        ):
            raise ValueError("Effective projection-byte bound exceeds the eval capture policy.")
        return self


def _validated_memory_intervention_terminal_evidence(
    value: object,
    *,
    expected_session: Session,
    observed_events: tuple[RunnerObservedEventIdentity, ...],
) -> TerminalSessionEvidence:
    """Detach store authority and bind it to the exact runtime session and live census."""

    try:
        expected = copy_session(expected_session)
        if type(value) is not TerminalSessionEvidence:
            raise TypeError
        evidence = copy_terminal_session_evidence(value)
        if evidence.session != expected or evidence.boundary.run_epoch != expected.run_epoch:
            raise ValueError
        if observed_events:
            retained_identities = {
                (record.event.session_id, record.sequence): record.event.type
                for record in evidence.events
            }
            if any(
                observed.sequence is not None
                and retained_identities.get((observed.session_id, observed.sequence))
                is not observed.event_type
                for observed in observed_events
            ):
                raise ValueError
        return evidence
    except (TypeError, ValueError):
        raise MemoryInterventionExecutionConflict(
            "Canonical runtime returned invalid or foreign terminal evidence."
        ) from None


async def _authenticate_intervention_session(
    app,
    session: Session | None,
    claim: object,
    *,
    reference: RuntimeSessionCreateClaimReference | None,
    request: RunRequest,
    operation_id: str,
    reference_key: RuntimeSessionCreateClaimReferenceKey,
) -> Session:
    """Translate the shared typed proof into this feature's fixed failure."""

    if reference is None:
        raise MemoryInterventionExecutionConflict(
            "Canonical runtime session claim reference is unavailable."
        )
    deferred_input = (
        None
        if session is None
        else await app.session_store.load_deferred_interaction_input(session.id)
    )
    authentication = authenticate_runtime_session_create_claim_reference(
        session,
        deferred_input,
        claim,
        reference,
        request=request,
        operation_id=operation_id,
        parent_session=None,
        key=reference_key,
    )
    if (
        authentication.disposition
        is not RuntimeSessionCreateClaimAuthenticationDisposition.MATCHING_SESSION
        or session is None
    ):
        raise MemoryInterventionExecutionConflict(
            "Canonical runtime session does not match its intervention create claim."
        )
    return session


def _remaining_runtime_timeout(
    request: MemoryInterventionTrialRequest,
    execution: MemoryInterventionExecutionRecord,
) -> float:
    deadline = execution.runtime_deadline_at
    if deadline is None:
        raise MemoryInterventionExecutionConflict(
            "Runtime execution is missing its durable deadline."
        )
    return max(
        0.0,
        min(
            float(request.timeout_seconds),
            (deadline - datetime.now(UTC)).total_seconds(),
        ),
    )


def _runtime_timeout_before_session_result(
    execution: MemoryInterventionExecutionRecord,
    *,
    effective_attribution_bounds: MemoryAttributionBounds,
    source_alias: EvalMemorySourceAliasV1 | None,
) -> MemoryInterventionRuntimeResult:
    return MemoryInterventionRuntimeResult(
        session_id=execution.session_id,
        terminal_disposition=AgentSnapshotTerminalDisposition.TIMED_OUT,
        runtime_evidence_fingerprint=_content_sha256(
            {
                "kind": "memory_intervention_runtime_deadline_expired_before_session",
                "version": 1,
                "execution_id": execution.execution_id,
                "session_id": execution.session_id,
                "runtime_deadline_at": (
                    None
                    if execution.runtime_deadline_at is None
                    else execution.runtime_deadline_at.isoformat()
                ),
                "runtime_session_create_claim_id": (
                    None
                    if execution.runtime_session_create_claim is None
                    else execution.runtime_session_create_claim.claim_id
                ),
            },
            "memory intervention pre-session timeout evidence",
        ),
        terminal_evidence_available=False,
        terminal_evidence_limitation=EvalMemoryEvidenceLimitation.DEADLINE_EXPIRED,
        expected_receipt_count=None,
        expected_exposure_count=None,
        effective_attribution_bounds=effective_attribution_bounds,
        source_alias=source_alias,
        attribution=_unavailable_terminal_memory_attribution(),
    )


def _terminal_evidence_limitation(
    error: NotImplementedError | TerminalSessionEvidenceError,
) -> EvalMemoryEvidenceLimitation:
    if isinstance(error, NotImplementedError):
        return EvalMemoryEvidenceLimitation.STORE_UNSUPPORTED
    if error.code is TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND:
        return EvalMemoryEvidenceLimitation.MISSING
    if error.code in {
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_CONFLICT,
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_DUPLICATE,
        TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID,
        TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_CONFLICT,
        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT,
    }:
        return EvalMemoryEvidenceLimitation.CONTRADICTORY_LINEAGE
    return EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED


def _memory_intervention_source_alias(app, session_id: str) -> EvalMemorySourceAliasV1 | None:
    key = memory_evidence_key(app._request_footprint)
    if key is None:
        return None
    return eval_memory_source_alias(
        session_id=session_id,
        key_id=key.key_id,
        key=key.key,
    )


def _unavailable_terminal_memory_attribution() -> MemoryAttribution:
    return MemoryAttribution(
        status=MemoryAttributionStatus.UNAVAILABLE,
        truncated=False,
        reason=MemoryAttributionUnavailableReason.EVIDENCE_READ_FAILED,
        observed_receipt_count=0,
        observed_exposure_count=0,
        observed_item_count=0,
        omitted_receipt_count_at_least=0,
        omitted_exposure_count_at_least=0,
        omitted_item_count_at_least=0,
    )


def _runtime_result_with_compact_attribution(
    result: MemoryInterventionRuntimeResult,
) -> MemoryInterventionRuntimeResult:
    attribution = result.attribution
    compact = MemoryAttribution(
        status=MemoryAttributionStatus.TRUNCATED,
        truncated=True,
        observed_receipt_count=attribution.observed_receipt_count,
        observed_exposure_count=attribution.observed_exposure_count,
        observed_item_count=attribution.observed_item_count,
        omitted_receipt_count_at_least=attribution.observed_receipt_count,
        omitted_exposure_count_at_least=attribution.observed_exposure_count,
        omitted_item_count_at_least=attribution.observed_item_count,
    )
    return MemoryInterventionRuntimeResult.model_validate(
        result.model_copy(update={"attribution": compact}).model_dump(mode="json")
    )


def _runtime_result_exceeds_effective_attribution_bounds(
    result: MemoryInterventionRuntimeResult,
) -> bool:
    bounds = result.effective_attribution_bounds
    retained_records = (*result.attribution.receipts, *result.attribution.exposures)
    return (
        len(result.attribution.receipts) > bounds.max_receipts
        or len(result.attribution.exposures) > bounds.max_exposures
        or sum(len(record.items) for record in retained_records) > bounds.max_items
        or sum(
            compact_json_utf8_size(record.model_dump(mode="json")) for record in retained_records
        )
        > bounds.max_projection_bytes
    )


class MemoryInterventionEvaluator(ABC):
    """Thin adapter over an application-owned existing evaluator identity."""

    evaluator_fingerprint: str

    @abstractmethod
    async def evaluate(
        self,
        *,
        operation_id: str,
        case: EvalCaseContractV1,
        runtime: MemoryInterventionRuntimeResult,
    ) -> EvalTrialResult:
        pass

    @abstractmethod
    async def recover(
        self,
        *,
        operation_id: str,
        case: EvalCaseContractV1,
        runtime: MemoryInterventionRuntimeResult,
    ) -> EvalTrialResult:
        """Complete or reconcile evaluation for the exact operation identity."""


def memory_intervention_runtime_result_fingerprint(
    result: MemoryInterventionRuntimeResult,
) -> str:
    if type(result) is not MemoryInterventionRuntimeResult:
        raise TypeError("result must be an exact MemoryInterventionRuntimeResult.")
    return _content_sha256(
        result.model_dump(mode="json"),
        "memory intervention runtime result",
    )


def _runtime_result_from_record(
    record: MemoryInterventionExecutionRecord,
) -> MemoryInterventionRuntimeResult:
    payload = record.runtime_result_payload
    if payload is None:
        raise MemoryInterventionExecutionConflict("Durable runtime result evidence is unavailable.")
    result = MemoryInterventionRuntimeResult.model_validate(payload)
    if (
        record.runtime_result_fingerprint != memory_intervention_runtime_result_fingerprint(result)
        or record.runtime_evidence_fingerprint != result.runtime_evidence_fingerprint
        or record.session_id != result.session_id
    ):
        raise MemoryInterventionExecutionConflict(
            "Durable runtime result evidence conflicts with its execution record."
        )
    return result


class MemoryInterventionExecutor:
    """Execute one fixed intervention through durable, exact phase boundaries.

    The injected overlay provider owns idempotent candidate-local effects.  The
    runtime runner owns the real Cayu session and must not return from timeout,
    cancellation, or interruption until any dispatched mutation is settled or
    durably recoverable.  The evaluator is an adapter over the evaluator already
    frozen in the starting :class:`AgentSnapshot`.
    """

    def __init__(
        self,
        *,
        snapshots: AgentSnapshotCoordinator,
        executions: MemoryInterventionExecutionStore,
        overlay_provider: MemoryInterventionOverlayProvider,
        runtime_runner: MemoryInterventionRuntimeRunner,
        evaluator: MemoryInterventionEvaluator,
        request_keys: Mapping[str, MemoryInterventionRequestFingerprintKey],
        current_request_key_id: str,
        clock=None,
    ) -> None:
        if type(snapshots) is not AgentSnapshotCoordinator:
            raise TypeError("snapshots must be an exact AgentSnapshotCoordinator.")
        if not isinstance(executions, MemoryInterventionExecutionStore):
            raise TypeError("executions must implement MemoryInterventionExecutionStore.")
        if not isinstance(overlay_provider, MemoryInterventionOverlayProvider):
            raise TypeError("overlay_provider must implement MemoryInterventionOverlayProvider.")
        if not isinstance(runtime_runner, MemoryInterventionRuntimeRunner):
            raise TypeError("runtime_runner must implement MemoryInterventionRuntimeRunner.")
        if not isinstance(evaluator, MemoryInterventionEvaluator):
            raise TypeError("evaluator must implement MemoryInterventionEvaluator.")
        provider_id = _clean(overlay_provider.provider_id, "overlay_provider.provider_id")
        provider_fingerprint = _sha256(
            overlay_provider.execution_profile_fingerprint,
            "overlay_provider.execution_profile_fingerprint",
        )
        runner_fingerprint = _sha256(
            runtime_runner.execution_profile_fingerprint,
            "runtime_runner.execution_profile_fingerprint",
        )
        evaluator_fingerprint = _sha256(
            evaluator.evaluator_fingerprint,
            "evaluator.evaluator_fingerprint",
        )
        copied_keys: dict[str, MemoryInterventionRequestFingerprintKey] = {}
        for raw_key_id, key in request_keys.items():
            key_id = _clean(raw_key_id, "request key id")
            if type(key) is not MemoryInterventionRequestFingerprintKey or key.key_id != key_id:
                raise TypeError("request_keys must contain exact, consistently keyed values.")
            copied_keys[key_id] = key
        current_key_id = _clean(current_request_key_id, "current_request_key_id")
        if current_key_id not in copied_keys:
            raise ValueError("The current memory-intervention request key is unavailable.")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable.")
        self.snapshots = snapshots
        self.executions = executions
        self.overlay_provider = overlay_provider
        self.runtime_runner = runtime_runner
        self.evaluator = evaluator
        self._provider_id = provider_id
        self._provider_fingerprint = provider_fingerprint
        self._runner_fingerprint = runner_fingerprint
        self._evaluator_fingerprint = evaluator_fingerprint
        self._request_keys = copied_keys
        self._current_request_key_id = current_key_id
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self._runtime_clock = lambda: datetime.now(UTC)
        self._runtime_dispatch_owner_id = f"memory-intervention-worker:{uuid4().hex}"

    def _now(self, *, at_least: datetime | None = None) -> datetime:
        value = _utc(self._clock(), "clock result")
        if at_least is not None and value < at_least:
            return at_least
        return value

    async def execute_trial(
        self,
        request: MemoryInterventionTrialRequest,
    ) -> MemoryInterventionTrialOutcome:
        """Execute or exactly recover one deterministic candidate trial."""

        request = _copy_trial_request(request)
        self._validate_live_owners()
        execution_id = request.execution_id
        gate = await _acquire_execution_gate(execution_id)
        try:
            return await self._execute_trial(request)
        finally:
            _release_execution_gate(execution_id, gate)

    async def _execute_trial(
        self,
        request: MemoryInterventionTrialRequest,
    ) -> MemoryInterventionTrialOutcome:
        record = _validated_store_record(
            await self.executions.load(request.execution_id),
            operation="load",
            allow_missing=True,
        )
        if record is None:
            key = self._request_keys[self._current_request_key_id]
            prepared = MemoryInterventionExecutionRecord.prepare(
                request,
                key=key,
                overlay_provider_fingerprint=self._provider_fingerprint,
                overlay_provider_id=self._provider_id,
                runtime_runner_fingerprint=self._runner_fingerprint,
                runtime_execution_profile_fingerprint=(
                    self.runtime_runner.required_execution_profile_fingerprint(request.spec)
                ),
                evaluator_fingerprint=self._evaluator_fingerprint,
                created_at=self._now(),
            )
            record = _validated_store_record(
                await self.executions.begin(prepared),
                operation="begin",
            )
            if record is None or record.replay_identity() != prepared.replay_identity():
                raise MemoryInterventionExecutionConflict(
                    "Execution store substituted another begin identity."
                )
        self._validate_request_authority(request, record)
        snapshot = await self._verified_snapshot(request)

        materialization, trial, operation, just_bound_trial = await self._trial_lineage(
            request,
            record,
            snapshot,
        )
        if just_bound_trial:
            desired = self._successor(
                record,
                phase=MemoryInterventionExecutionPhase.TRIAL_BOUND,
                materialization_fingerprint=materialization.fingerprint,
                trial_binding_fingerprint=trial.fingerprint,
                operation_fingerprint=operation.fingerprint,
            )
            record, just_bound_trial = await self._advance(record, desired)
        self._require_lineage(record, materialization, trial, operation)

        apply_effect = just_bound_trial
        view = await self._open_view(
            request,
            record,
            materialization,
            trial,
            operation,
            apply=apply_effect,
        )
        if record.phase is MemoryInterventionExecutionPhase.TRIAL_BOUND:
            desired = self._successor(
                record,
                phase=MemoryInterventionExecutionPhase.EFFECT_RESOLVED,
                receipt_fingerprint=view.receipt.fingerprint,
            )
            record, effect_won = await self._advance(record, desired)
            if not effect_won:
                view = await self._open_view(
                    request,
                    record,
                    materialization,
                    trial,
                    operation,
                    apply=False,
                )
        if record.receipt_fingerprint != view.receipt.fingerprint:
            raise MemoryInterventionExecutionConflict(
                "Recovered intervention receipt differs from durable execution evidence."
            )

        terminal_effect = _terminal_effect_status(view.receipt.status)
        if terminal_effect is not None:
            if record.status is MemoryInterventionExecutionStatus.ACTIVE:
                desired = self._successor(
                    record,
                    phase=MemoryInterventionExecutionPhase.EFFECT_RESOLVED,
                    status=terminal_effect,
                    failure_code=f"intervention_{view.receipt.status.value}",
                )
                record, _ = await self._advance(record, desired)
            if record.status is not terminal_effect:
                raise MemoryInterventionExecutionConflict(
                    "Terminal intervention evidence conflicts with execution state."
                )
            return MemoryInterventionTrialOutcome(execution=record, receipt=view.receipt)
        if record.status is not MemoryInterventionExecutionStatus.ACTIVE and (
            record.phase is not MemoryInterventionExecutionPhase.FINALIZED
        ):
            raise MemoryInterventionExecutionConflict(
                "A non-terminal intervention receipt cannot resume a terminal execution."
            )

        just_bound_session = False
        if record.phase is MemoryInterventionExecutionPhase.EFFECT_RESOLVED:
            deadline_started_at = self._runtime_clock()
            runtime_request = _memory_intervention_runtime_request(
                request,
                record,
                trial,
            )
            reference_key = memory_intervention_request_key(
                self._request_keys,
                record.request_key_id,
            ).runtime_session_create_reference_key()
            desired = self._successor(
                record,
                phase=MemoryInterventionExecutionPhase.SESSION_BOUND,
                runtime_session_create_claim=runtime_session_create_claim_reference(
                    runtime_request,
                    operation_id=record.execution_id,
                    key=reference_key,
                ),
                runtime_deadline_at=(
                    deadline_started_at + timedelta(seconds=request.timeout_seconds)
                ),
            )
            record, just_bound_session = await self._advance(record, desired)
        run_runtime = False
        runtime_dispatch_ownership: DurableOperationOwnership | None = None
        just_recorded_runtime = False
        if record.phase is MemoryInterventionExecutionPhase.SESSION_BOUND:
            record, run_runtime, runtime_dispatch_ownership = await self._claim_runtime_dispatch(
                record,
                fresh_if_acquired=just_bound_session,
            )
        if record.phase is MemoryInterventionExecutionPhase.SESSION_BOUND:
            if runtime_dispatch_ownership is None:
                raise MemoryInterventionExecutionConflict(
                    "Runtime dispatch ownership disappeared before execution."
                )
            if self._runtime_deadline_expired(record):
                record = await self._record_runtime_timeout(record)
            current_task = asyncio.current_task()
            historical_cancellation_requests = (
                0 if current_task is None else current_task.cancelling()
            )
            try:
                runtime, record = await self._run_runtime(
                    request,
                    record,
                    snapshot.execution_profile,
                    trial,
                    operation,
                    view,
                    run=run_runtime,
                    ownership=runtime_dispatch_ownership,
                )
            except BaseException as runtime_failure:
                cancellation = next(
                    (
                        candidate
                        for candidate in iter_exception_tree(runtime_failure)
                        if isinstance(candidate, asyncio.CancelledError)
                    ),
                    None,
                )
                if cancellation is None:
                    raise
                await self._record_runtime_cancellation(
                    record,
                    runtime_failure,
                    cancellation,
                    historical_requests=historical_cancellation_requests,
                )
                raise AssertionError(
                    "Cancellation recording must propagate cancellation."
                ) from None
            runtime_result_fingerprint = memory_intervention_runtime_result_fingerprint(runtime)
            try:
                desired = self._successor(
                    record,
                    phase=MemoryInterventionExecutionPhase.RUNTIME_TERMINAL,
                    runtime_evidence_fingerprint=runtime.runtime_evidence_fingerprint,
                    runtime_result_fingerprint=runtime_result_fingerprint,
                    runtime_result_payload=runtime.model_dump(mode="json"),
                )
            except ValueError as error:
                if "Memory intervention execution record exceeds its byte bound." not in str(error):
                    raise
                runtime = _runtime_result_with_compact_attribution(runtime)
                runtime_result_fingerprint = memory_intervention_runtime_result_fingerprint(runtime)
                desired = self._successor(
                    record,
                    phase=MemoryInterventionExecutionPhase.RUNTIME_TERMINAL,
                    runtime_evidence_fingerprint=runtime.runtime_evidence_fingerprint,
                    runtime_result_fingerprint=runtime_result_fingerprint,
                    runtime_result_payload=runtime.model_dump(mode="json"),
                )
            record, just_recorded_runtime = await self._advance(record, desired)
        runtime = _runtime_result_from_record(record)

        eval_result = await self._evaluate(
            request,
            runtime,
            evaluate=just_recorded_runtime,
        )
        eval_revision = memory_intervention_eval_result_revision(eval_result)
        if record.phase is MemoryInterventionExecutionPhase.RUNTIME_TERMINAL:
            desired = self._successor(
                record,
                phase=MemoryInterventionExecutionPhase.EVALUATED,
                eval_result_revision=eval_revision,
            )
            record, eval_won = await self._advance(record, desired)
            if not eval_won:
                eval_result = await self._evaluate(
                    request,
                    runtime,
                    evaluate=False,
                )
                eval_revision = memory_intervention_eval_result_revision(eval_result)
        if record.eval_result_revision != eval_revision:
            raise MemoryInterventionExecutionConflict(
                "Recovered evaluation differs from durable execution evidence."
            )

        snapshot_result = AgentSnapshotResultBinding.create(
            trial=trial,
            session_id=record.session_id,
            terminal_disposition=runtime.terminal_disposition,
            runtime_evidence_fingerprint=runtime.runtime_evidence_fingerprint,
            eval_result_revision=eval_revision,
            recorded_at=record.updated_at,
            memory_evidence_fingerprint=memory_attribution_fingerprint(runtime.attribution),
            usage_fingerprint=runtime.usage_fingerprint,
            cost_fingerprint=runtime.cost_fingerprint,
        )
        snapshot_result = await self.snapshots.record_result(snapshot_result)
        binding = MemoryInterventionTrialBinding.create(
            spec=request.spec,
            operation=operation,
            receipt=view.receipt,
            trial=trial,
            result=snapshot_result,
            attribution=runtime.attribution,
            terminal_evidence_available=runtime.terminal_evidence_available,
            expected_receipt_count=runtime.expected_receipt_count,
            expected_exposure_count=runtime.expected_exposure_count,
        )
        final_status = MemoryInterventionExecutionStatus(runtime.terminal_disposition.value)
        failure_code = (
            None
            if final_status is MemoryInterventionExecutionStatus.COMPLETED
            else f"runtime_{final_status.value}"
        )
        if record.phase is MemoryInterventionExecutionPhase.EVALUATED:
            desired = self._successor(
                record,
                phase=MemoryInterventionExecutionPhase.FINALIZED,
                status=final_status,
                snapshot_result_fingerprint=snapshot_result.fingerprint,
                final_binding_fingerprint=binding.fingerprint,
                failure_code=failure_code,
            )
            record, finalized = await self._advance(record, desired)
            if not finalized:
                stored_result = await self.snapshots.store.load_result(
                    record.snapshot_result_fingerprint or ""
                )
                if type(stored_result) is not AgentSnapshotResultBinding:
                    raise MemoryInterventionExecutionConflict(
                        "Finalized execution result is unavailable."
                    )
                snapshot_result = AgentSnapshotResultBinding.model_validate(
                    stored_result.model_dump(mode="json")
                )
                binding = MemoryInterventionTrialBinding.create(
                    spec=request.spec,
                    operation=operation,
                    receipt=view.receipt,
                    trial=trial,
                    result=snapshot_result,
                    attribution=runtime.attribution,
                    terminal_evidence_available=runtime.terminal_evidence_available,
                    expected_receipt_count=runtime.expected_receipt_count,
                    expected_exposure_count=runtime.expected_exposure_count,
                )
        if (
            record.status is not final_status
            or record.failure_code != failure_code
            or record.snapshot_result_fingerprint != snapshot_result.fingerprint
            or record.final_binding_fingerprint != binding.fingerprint
        ):
            raise MemoryInterventionExecutionConflict(
                "Final execution lineage differs from its durable evidence."
            )
        return MemoryInterventionTrialOutcome(
            execution=record,
            receipt=view.receipt,
            eval_result=eval_result,
            snapshot_result=snapshot_result,
            binding=binding,
        )

    def _runtime_deadline_expired(self, record: MemoryInterventionExecutionRecord) -> bool:
        deadline = record.runtime_deadline_at
        if deadline is None:
            raise MemoryInterventionExecutionConflict(
                "Runtime execution is missing its durable deadline."
            )
        return record.runtime_timeout_observed or self._runtime_clock() >= deadline

    async def _claim_runtime_dispatch(
        self,
        record: MemoryInterventionExecutionRecord,
        *,
        fresh_if_acquired: bool,
    ) -> tuple[
        MemoryInterventionExecutionRecord,
        bool,
        DurableOperationOwnership | None,
    ]:
        """Claim through store time, waiting without interpreting a worker clock."""

        current = record
        claim_id = "memory-intervention-runtime-claim:" + _content_sha256(
            {
                "execution_id": record.execution_id,
                "owner_id": self._runtime_dispatch_owner_id,
            },
            "memory intervention runtime claim",
        )
        request = DurableOperationOwnershipTransition(
            operation_id=record.execution_id,
            claim_id=claim_id,
            owner_id=self._runtime_dispatch_owner_id,
            action=DurableOperationOwnershipAction.CLAIM,
            lease_seconds=MEMORY_INTERVENTION_RUNTIME_LEASE_SECONDS,
        )
        while current.phase is MemoryInterventionExecutionPhase.SESSION_BOUND:
            raw_result = await self.executions.transition_runtime_dispatch_ownership(
                current.execution_id,
                request,
            )
            if type(raw_result) is not MemoryInterventionRuntimeOwnershipResult:
                raise MemoryInterventionExecutionConflict(
                    "Execution store returned invalid runtime ownership evidence."
                )
            result = MemoryInterventionRuntimeOwnershipResult.model_validate(
                raw_result.model_dump(mode="python")
            )
            current = result.execution
            if current.immutable_identity() != record.immutable_identity():
                raise MemoryInterventionExecutionConflict(
                    "Runtime dispatch ownership lost its execution identity."
                )
            disposition = result.ownership.disposition
            ownership = result.ownership.ownership
            if disposition in {
                DurableOperationOwnershipDisposition.ACQUIRED,
                DurableOperationOwnershipDisposition.EQUIVALENT_LIVE_OWNER,
                DurableOperationOwnershipDisposition.EXPIRED_TAKEN_OVER,
            }:
                if not _runtime_dispatch_ownership_matches(ownership, request):
                    raise MemoryInterventionExecutionConflict(
                        "Execution store substituted runtime ownership authority."
                    )
                return (
                    current,
                    fresh_if_acquired
                    and disposition
                    in {
                        DurableOperationOwnershipDisposition.ACQUIRED,
                        DurableOperationOwnershipDisposition.EQUIVALENT_LIVE_OWNER,
                    },
                    ownership,
                )
            if disposition is DurableOperationOwnershipDisposition.OPERATION_ADVANCED:
                return current, False, None
            if disposition is DurableOperationOwnershipDisposition.INDETERMINATE:
                raise MemoryInterventionExecutionConflict(
                    "Runtime dispatch ownership outcome is indeterminate."
                )
            if disposition is DurableOperationOwnershipDisposition.IDENTITY_CONFLICT:
                raise MemoryInterventionExecutionConflict(
                    "Runtime dispatch ownership conflicts with execution identity."
                )
            if (
                disposition is not DurableOperationOwnershipDisposition.FENCED
                or ownership is None
                or ownership.lease_expires_at is None
            ):
                raise MemoryInterventionExecutionConflict(
                    "Execution store returned contradictory runtime ownership evidence."
                )
            await asyncio.sleep(MEMORY_INTERVENTION_RUNTIME_OWNERSHIP_WAIT_SECONDS)
        return current, False, None

    async def _heartbeat_runtime_dispatch(
        self,
        execution_id: str,
        ownership: DurableOperationOwnership,
        stop: asyncio.Event,
    ) -> None:
        current_ownership = ownership
        while True:
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=MEMORY_INTERVENTION_RUNTIME_HEARTBEAT_SECONDS,
                )
                return
            except TimeoutError:
                pass
            result = await self.executions.transition_runtime_dispatch_ownership(
                execution_id,
                DurableOperationOwnershipTransition(
                    operation_id=current_ownership.operation_id,
                    claim_id=current_ownership.claim_id,
                    owner_id=current_ownership.owner_id,
                    generation=current_ownership.generation,
                    action=DurableOperationOwnershipAction.RENEW,
                    lease_seconds=MEMORY_INTERVENTION_RUNTIME_LEASE_SECONDS,
                ),
            )
            if type(result) is not MemoryInterventionRuntimeOwnershipResult:
                raise MemoryInterventionExecutionConflict(
                    "Execution store returned invalid heartbeat ownership evidence."
                )
            disposition = result.ownership.disposition
            renewed = result.ownership.ownership
            if (
                disposition is not DurableOperationOwnershipDisposition.RENEWED
                or not _runtime_dispatch_ownership_matches(renewed, current_ownership)
            ):
                raise MemoryInterventionExecutionConflict(
                    "Runtime dispatch ownership changed during heartbeat publication."
                )
            assert renewed is not None
            current_ownership = renewed

    async def _request_runtime_dispatch_renewal(
        self,
        record: MemoryInterventionExecutionRecord,
        ownership: DurableOperationOwnership,
        *,
        operation: str,
    ) -> MemoryInterventionRuntimeOwnershipResult:
        """Revalidate one exact generation using store time before publication."""

        if not _runtime_dispatch_ownership_matches(
            record.runtime_dispatch_ownership,
            ownership,
        ):
            raise MemoryInterventionExecutionConflict(
                f"{operation} has no exact runtime dispatch ownership."
            )
        raw_result = await self.executions.transition_runtime_dispatch_ownership(
            record.execution_id,
            DurableOperationOwnershipTransition(
                operation_id=ownership.operation_id,
                claim_id=ownership.claim_id,
                owner_id=ownership.owner_id,
                generation=ownership.generation,
                action=DurableOperationOwnershipAction.RENEW,
                lease_seconds=MEMORY_INTERVENTION_RUNTIME_LEASE_SECONDS,
            ),
        )
        if type(raw_result) is not MemoryInterventionRuntimeOwnershipResult:
            raise MemoryInterventionExecutionConflict(
                f"Execution store returned invalid {operation} ownership evidence."
            )
        result = MemoryInterventionRuntimeOwnershipResult.model_validate(
            raw_result.model_dump(mode="python")
        )
        if result.execution.immutable_identity() != record.immutable_identity():
            raise MemoryInterventionExecutionConflict(f"{operation} lost its execution identity.")
        return result

    async def _record_runtime_cancellation(
        self,
        record: MemoryInterventionExecutionRecord,
        runtime_failure: BaseException,
        cancellation: asyncio.CancelledError,
        *,
        historical_requests: int,
    ) -> None:
        if type(historical_requests) is not int or historical_requests < 0:
            raise ValueError("historical_requests must be a non-negative int.")
        current_task = asyncio.current_task()
        if current_task is None or current_task.cancelling() <= historical_requests:
            raise MemoryInterventionExecutionConflict(
                "Runtime runner cancelled without authenticated caller cancellation."
            ) from None
        publication_task = asyncio.create_task(
            capture_awaitable_outcome(
                lambda: self._publish_runtime_control_authority(
                    record,
                    control="cancellation",
                )
            ),
            name="cayu-memory-intervention-cancellation-authority",
        )
        outcome = await await_shielded_task_outcome(
            publication_task,
            cancellation=cancellation,
        )
        if outcome.timed_out:
            raise AssertionError("Cancellation-authority publication cannot time out.")
        publication_error = outcome.error
        publication_result = None
        if outcome.result is not None:
            publication_error = outcome.result.error or publication_error
            publication_result = outcome.result.result
        if publication_error is not None and exception_tree_contains(
            publication_error,
            (GeneratorExit, KeyboardInterrupt, SystemExit),
        ):
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed,
                cancellation=cancellation,
            )
            raise BaseExceptionGroup(
                "Runtime cancellation authority failed during process control.",
                (runtime_failure, publication_error),
            ) from None
        additional_failure = publication_error
        if additional_failure is None and (
            publication_result is None or not publication_result[0].runtime_cancellation_observed
        ):
            additional_failure = MemoryInterventionExecutionConflict(
                "Runtime cancellation authority was not durably recorded."
            )
        if additional_failure is not None:
            existing_cause = exception_cause(runtime_failure)
            cause = (
                additional_failure
                if existing_cause is None
                else BaseExceptionGroup(
                    "Runtime cancellation retained multiple boundary failures.",
                    (existing_cause, additional_failure),
                )
            )
            if not set_exception_cause(runtime_failure, cause):
                restore_task_cancellation_requests(
                    outcome.cancellation_requests_consumed,
                    cancellation=cancellation,
                )
                raise BaseExceptionGroup(
                    "Runtime cancellation authority could not retain boundary failure.",
                    (runtime_failure, cause),
                ) from None
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed,
            cancellation=cancellation,
        )
        raise runtime_failure

    async def _record_runtime_timeout(
        self,
        record: MemoryInterventionExecutionRecord,
    ) -> MemoryInterventionExecutionRecord:
        publication_task = asyncio.create_task(
            capture_awaitable_outcome(
                lambda: self._publish_runtime_control_authority(
                    record,
                    control="timeout",
                )
            ),
            name="cayu-memory-intervention-timeout-authority",
        )
        outcome = await await_shielded_task_outcome(publication_task)
        if outcome.timed_out:
            raise AssertionError("Timeout-authority publication cannot time out.")
        publication_error = outcome.error
        publication_result = None
        if outcome.result is not None:
            publication_error = outcome.result.error or publication_error
            publication_result = outcome.result.result
        if outcome.cancellation is not None:
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed,
                cancellation=outcome.cancellation,
            )
            if publication_error is not None:
                raise BaseExceptionGroup(
                    "Runtime timeout authority failed during caller cancellation.",
                    (outcome.cancellation, publication_error),
                ) from None
            if publication_result is None or not publication_result[0].runtime_timeout_observed:
                raise BaseExceptionGroup(
                    "Runtime timeout authority was not durably recorded.",
                    (
                        outcome.cancellation,
                        MemoryInterventionExecutionConflict(
                            "Runtime timeout authority was not durably recorded."
                        ),
                    ),
                ) from None
            raise outcome.cancellation
        if publication_error is not None:
            raise publication_error
        if publication_result is None or not publication_result[0].runtime_timeout_observed:
            raise MemoryInterventionExecutionConflict(
                "Runtime timeout authority was not durably recorded."
            )
        return publication_result[0]

    async def _publish_runtime_control_authority(
        self,
        record: MemoryInterventionExecutionRecord,
        *,
        control: Literal["cancellation", "timeout"],
    ) -> tuple[MemoryInterventionExecutionRecord, bool]:
        field_name = f"runtime_{control}_observed"
        ownership = record.runtime_dispatch_ownership
        if ownership is None:
            raise MemoryInterventionExecutionConflict(
                f"Runtime {control} authority has no dispatch ownership."
            )
        renewal = await self._request_runtime_dispatch_renewal(
            record,
            ownership,
            operation=f"runtime {control} authority",
        )
        current = renewal.execution
        exact_current_ownership = _runtime_dispatch_ownership_matches(
            current.runtime_dispatch_ownership,
            ownership,
        )
        if (
            getattr(current, field_name)
            and exact_current_ownership
            and renewal.ownership.disposition
            in {
                DurableOperationOwnershipDisposition.RENEWED,
                DurableOperationOwnershipDisposition.FENCED,
                DurableOperationOwnershipDisposition.OPERATION_ADVANCED,
            }
        ):
            return current, False
        if (
            renewal.ownership.disposition is not DurableOperationOwnershipDisposition.RENEWED
            or not exact_current_ownership
            or current.phase is not MemoryInterventionExecutionPhase.SESSION_BOUND
        ):
            raise MemoryInterventionExecutionConflict(
                f"Runtime {control} authority lost its live dispatch ownership."
            )
        desired = (
            self._successor(
                current,
                phase=MemoryInterventionExecutionPhase.SESSION_BOUND,
                runtime_cancellation_observed=True,
            )
            if control == "cancellation"
            else self._successor(
                current,
                phase=MemoryInterventionExecutionPhase.SESSION_BOUND,
                runtime_timeout_observed=True,
            )
        )
        return await self._advance(current, desired)

    def _validate_request_authority(
        self,
        request: MemoryInterventionTrialRequest,
        record: MemoryInterventionExecutionRecord,
    ) -> None:
        if record.execution_id != request.execution_id:
            raise MemoryInterventionExecutionConflict("Execution identity changed.")
        key = memory_intervention_request_key(self._request_keys, record.request_key_id)
        if not hmac.compare_digest(record.request_fingerprint, key.fingerprint(request)):
            raise MemoryInterventionExecutionConflict(
                "Execution identity is already bound to another request."
            )
        expected = (
            request.spec.fingerprint,
            request.candidate_id,
            request.trial_id,
            request.case.case_id,
            request.case.case_revision,
            request.spec.execution_profile_fingerprint,
            self.runtime_runner.required_execution_profile_fingerprint(request.spec),
            request.session_id,
            request.causal_budget_id,
            self._provider_fingerprint,
            self._provider_id,
            self._runner_fingerprint,
            self._evaluator_fingerprint,
        )
        actual = (
            record.spec_fingerprint,
            record.candidate_id,
            record.trial_id,
            record.case_id,
            record.case_revision,
            record.required_execution_profile_fingerprint,
            record.runtime_execution_profile_fingerprint,
            record.session_id,
            record.causal_budget_id,
            record.overlay_provider_fingerprint,
            record.overlay_provider_id,
            record.runtime_runner_fingerprint,
            record.evaluator_fingerprint,
        )
        if actual != expected:
            raise MemoryInterventionExecutionConflict(
                "Execution authority differs from its exact request or application owners."
            )

    async def _verified_snapshot(
        self,
        request: MemoryInterventionTrialRequest,
    ) -> AgentSnapshot:
        loaded = await self.snapshots.store.load_snapshot(request.spec.snapshot_fingerprint)
        if type(loaded) is not AgentSnapshot:
            raise MemoryInterventionExecutionConflict("Starting snapshot is unavailable.")
        snapshot = await self.snapshots.verify(loaded)
        if (
            snapshot.fingerprint != request.spec.snapshot_fingerprint
            or snapshot.memory_state is None
            or snapshot.memory_state.fingerprint != request.spec.memory_state_fingerprint
            or snapshot.execution_profile.fingerprint != request.spec.execution_profile_fingerprint
            or snapshot.authority_scope_fingerprint != request.spec.authority_scope_fingerprint
            or snapshot.subject.agent_id != request.run_request.agent_name
            or snapshot.evaluator is None
            or snapshot.evaluator.identity.fingerprint != self._evaluator_fingerprint
        ):
            raise MemoryInterventionExecutionConflict(
                "The verified snapshot conflicts with the fixed intervention request."
            )
        return snapshot

    async def _trial_lineage(
        self,
        request: MemoryInterventionTrialRequest,
        record: MemoryInterventionExecutionRecord,
        snapshot: AgentSnapshot,
    ) -> tuple[
        AgentSnapshotMaterialization,
        AgentSnapshotTrialBinding,
        MemoryInterventionOperation,
        bool,
    ]:
        if record.phase is MemoryInterventionExecutionPhase.PREPARED:
            materialization = await self.snapshots.materialize(
                AgentSnapshotMaterializationRequest(
                    access=AgentSnapshotAccess(
                        snapshot=snapshot.ref,
                        binding_id=snapshot.identity_binding.binding_id,
                        authority_scope_fingerprint=snapshot.authority_scope_fingerprint,
                    ),
                    candidate_id=request.candidate_id,
                    trial_id=request.trial_id,
                    state_mode=request.spec.trial_state_mode,
                    state_partition_fingerprint=request.spec.fingerprint,
                )
            )
            trial = await self.snapshots.begin_trial(
                materialization,
                case_id=request.case.case_id,
                trial_id=request.trial_id,
                evaluator_fingerprint=self._evaluator_fingerprint,
            )
            operation = MemoryInterventionOperation.create(
                spec=request.spec,
                materialization=materialization,
                trial=trial,
            )
            return materialization, trial, operation, True
        if (
            record.materialization_fingerprint is None
            or record.trial_binding_fingerprint is None
            or record.operation_fingerprint is None
        ):
            raise MemoryInterventionExecutionConflict("Execution trial lineage is incomplete.")
        materialization = await self.snapshots.recover_materialization(
            record.materialization_fingerprint,
            access=AgentSnapshotAccess(
                snapshot=snapshot.ref,
                binding_id=snapshot.identity_binding.binding_id,
                authority_scope_fingerprint=snapshot.authority_scope_fingerprint,
            ),
        )
        loaded_trial = await self.snapshots.store.load_trial(record.trial_binding_fingerprint)
        if type(loaded_trial) is not AgentSnapshotTrialBinding:
            raise MemoryInterventionExecutionConflict("Execution trial binding is unavailable.")
        trial = AgentSnapshotTrialBinding.model_validate(loaded_trial.model_dump(mode="json"))
        operation = MemoryInterventionOperation.create(
            spec=request.spec,
            materialization=materialization,
            trial=trial,
        )
        return materialization, trial, operation, False

    @staticmethod
    def _require_lineage(
        record: MemoryInterventionExecutionRecord,
        materialization: AgentSnapshotMaterialization,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
    ) -> None:
        if (
            record.materialization_fingerprint != materialization.fingerprint
            or record.trial_binding_fingerprint != trial.fingerprint
            or record.operation_fingerprint != operation.fingerprint
        ):
            raise MemoryInterventionExecutionConflict(
                "Recovered trial lineage differs from durable execution evidence."
            )

    async def _open_view(
        self,
        request: MemoryInterventionTrialRequest,
        record: MemoryInterventionExecutionRecord,
        materialization: AgentSnapshotMaterialization,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
        *,
        apply: bool,
    ) -> MemoryInterventionRuntimeView:
        self._validate_live_owners()
        method = self.overlay_provider.apply if apply else self.overlay_provider.recover
        value = await method(
            spec=request.spec,
            operation=operation,
            materialization=materialization,
            trial=trial,
        )
        if type(value) is not MemoryInterventionRuntimeView:
            raise MemoryInterventionExecutionConflict(
                "Overlay provider returned an invalid runtime view."
            )
        view = MemoryInterventionRuntimeView(
            materialization_fingerprint=value.materialization_fingerprint,
            memory_overlay_fingerprint=value.memory_overlay_fingerprint,
            state_scope_id=value.state_scope_id,
            knowledge_store=value.knowledge_store,
            knowledge_access_scope=value.knowledge_access_scope,
            isolation_authority=value.isolation_authority,
            trial_recall_policy=value.trial_recall_policy,
            receipt=value.receipt,
        )
        if (
            view.materialization_fingerprint != materialization.fingerprint
            or view.memory_overlay_fingerprint != operation.memory_overlay_fingerprint
            or view.state_scope_id != operation.state_scope_id
            or view.receipt.spec_fingerprint != request.spec.fingerprint
            or view.receipt.operation_fingerprint != operation.fingerprint
            or view.trial_recall_policy.fingerprint()
            != request.spec.trial_recall_policy_fingerprint
        ):
            raise MemoryInterventionExecutionConflict(
                "Overlay provider returned a view outside the precommitted operation."
            )
        if (
            record.receipt_fingerprint is not None
            and view.receipt.fingerprint != record.receipt_fingerprint
        ):
            raise MemoryInterventionExecutionConflict(
                "Recovered overlay receipt differs from durable execution evidence."
            )
        return view

    async def _run_runtime(
        self,
        request: MemoryInterventionTrialRequest,
        record: MemoryInterventionExecutionRecord,
        starting_execution_profile: AgentSnapshotExecutionProfileRef,
        trial: AgentSnapshotTrialBinding,
        operation: MemoryInterventionOperation,
        view: MemoryInterventionRuntimeView,
        *,
        run: bool,
        ownership: DurableOperationOwnership,
    ) -> tuple[MemoryInterventionRuntimeResult, MemoryInterventionExecutionRecord]:
        self._validate_live_owners()
        method = self.runtime_runner.run if run else self.runtime_runner.recover
        current = record
        stop_heartbeat = asyncio.Event()
        runtime_task = asyncio.create_task(
            method(
                request=request,
                execution=record,
                starting_execution_profile=starting_execution_profile,
                trial=trial,
                operation=operation,
                view=view,
                reference_key=memory_intervention_request_key(
                    self._request_keys,
                    record.request_key_id,
                ).runtime_session_create_reference_key(),
            ),
            name=f"cayu-memory-intervention-runtime:{record.execution_id[:12]}",
        )
        heartbeat_task = asyncio.create_task(
            self._heartbeat_runtime_dispatch(
                record.execution_id,
                ownership,
                stop_heartbeat,
            ),
            name=f"cayu-memory-intervention-heartbeat:{record.execution_id[:12]}",
        )
        try:
            done, _ = await asyncio.wait(
                {runtime_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                heartbeat_error = heartbeat_task.exception()
                if heartbeat_error is not None:
                    runtime_task.cancel()
                    await asyncio.gather(runtime_task, return_exceptions=True)
                    raise heartbeat_error
            value = await runtime_task
        except _MemoryInterventionRuntimeTimeoutObserved as timeout:
            stop_heartbeat.set()
            await heartbeat_task
            current = await self._record_runtime_timeout(record)
            value = await timeout.complete()
        except BaseException:
            stop_heartbeat.set()
            if not runtime_task.done():
                runtime_task.cancel()
            await asyncio.gather(runtime_task, heartbeat_task, return_exceptions=True)
            raise
        else:
            stop_heartbeat.set()
            await heartbeat_task
        renewal = await self._request_runtime_dispatch_renewal(
            record,
            ownership,
            operation="runtime dispatch completion",
        )
        current = renewal.execution
        renewed_ownership = renewal.ownership.ownership
        if (
            renewal.ownership.disposition is not DurableOperationOwnershipDisposition.RENEWED
            or current.phase is not MemoryInterventionExecutionPhase.SESSION_BOUND
            or not _runtime_dispatch_ownership_matches(renewed_ownership, ownership)
        ):
            raise MemoryInterventionExecutionConflict(
                "Runtime dispatch completed without its durable ownership."
            )
        if type(value) is not MemoryInterventionRuntimeResult:
            raise MemoryInterventionExecutionConflict(
                "Runtime runner returned invalid terminal evidence."
            )
        result = MemoryInterventionRuntimeResult.model_validate(value.model_dump(mode="json"))
        if result.session_id != record.session_id:
            raise MemoryInterventionExecutionConflict(
                "Runtime result belongs to another intervention session."
            )
        if _runtime_result_exceeds_effective_attribution_bounds(result):
            result = _runtime_result_with_compact_attribution(result)
        if result.terminal_disposition is AgentSnapshotTerminalDisposition.TIMED_OUT:
            current = await self._record_runtime_timeout(current)
        return result, current

    async def _evaluate(
        self,
        request: MemoryInterventionTrialRequest,
        runtime: MemoryInterventionRuntimeResult,
        *,
        evaluate: bool,
    ) -> EvalTrialResult:
        self._validate_live_owners()
        method = self.evaluator.evaluate if evaluate else self.evaluator.recover
        value = await method(
            operation_id=request.execution_id,
            case=request.case,
            runtime=runtime,
        )
        if type(value) is not EvalTrialResult:
            raise MemoryInterventionExecutionConflict("Evaluator returned an invalid result.")
        result_payload = value.model_dump(mode="python")
        result_payload["trajectory"] = value.trajectory
        result = EvalTrialResult.model_validate(result_payload)
        if result.session_id != runtime.session_id:
            raise MemoryInterventionExecutionConflict(
                "Evaluator result belongs to another intervention session."
            )
        terminal_status: Literal["completed", "failed", "interrupted"]
        if runtime.terminal_disposition is AgentSnapshotTerminalDisposition.COMPLETED:
            terminal_status = "completed"
        elif runtime.terminal_disposition is AgentSnapshotTerminalDisposition.FAILED:
            terminal_status = "failed"
        else:
            terminal_status = "interrupted"
        expected_attribution = eval_memory_attribution_evidence_from_runtime_source(
            terminal_status=terminal_status,
            attribution=runtime.attribution,
            terminal_evidence_available=runtime.terminal_evidence_available,
            terminal_evidence_limitation=runtime.terminal_evidence_limitation,
            expected_receipt_count=runtime.expected_receipt_count,
            expected_exposure_count=runtime.expected_exposure_count,
            effective_bounds=runtime.effective_attribution_bounds,
            source_alias=runtime.source_alias,
        )
        default_attribution = EvalMemoryAttributionEvidenceV1.unavailable(
            EvalMemoryEvidenceLimitation.MISSING
        )
        supplied_attribution = result.memory_attribution
        if supplied_attribution != default_attribution:
            supplied_fingerprints = tuple(
                source.attribution_fingerprint for source in supplied_attribution.sources
            )
            expected_fingerprints = tuple(
                source.attribution_fingerprint for source in expected_attribution.sources
            )
            if supplied_fingerprints != expected_fingerprints:
                raise MemoryInterventionExecutionConflict(
                    "Evaluator result conflicts with runtime-owned memory attribution."
                )
        result_payload = result.model_dump(mode="python")
        result_payload["trajectory"] = result.trajectory
        result_payload["memory_attribution"] = expected_attribution
        return EvalTrialResult.model_validate(result_payload)

    def _validate_live_owners(self) -> None:
        if (
            _clean(self.overlay_provider.provider_id, "overlay_provider.provider_id")
            != self._provider_id
            or _sha256(
                self.overlay_provider.execution_profile_fingerprint,
                "overlay_provider.execution_profile_fingerprint",
            )
            != self._provider_fingerprint
            or _sha256(
                self.runtime_runner.execution_profile_fingerprint,
                "runtime_runner.execution_profile_fingerprint",
            )
            != self._runner_fingerprint
            or _sha256(
                self.evaluator.evaluator_fingerprint,
                "evaluator.evaluator_fingerprint",
            )
            != self._evaluator_fingerprint
        ):
            raise MemoryInterventionExecutionConflict(
                "Application-owned intervention execution identity changed."
            )

    def _successor(
        self,
        record: MemoryInterventionExecutionRecord,
        *,
        phase: MemoryInterventionExecutionPhase,
        status: MemoryInterventionExecutionStatus | None = None,
        **updates: object,
    ) -> MemoryInterventionExecutionRecord:
        values = record.model_dump(mode="python")
        values.update(updates)
        values.update(
            {
                "phase": phase,
                "status": record.status if status is None else status,
                "revision": record.revision + 1,
                "updated_at": self._now(at_least=record.updated_at),
            }
        )
        return MemoryInterventionExecutionRecord.model_validate(values)

    async def _advance(
        self,
        expected: MemoryInterventionExecutionRecord,
        desired: MemoryInterventionExecutionRecord,
    ) -> tuple[MemoryInterventionExecutionRecord, bool]:
        try:
            committed = _validated_store_record(
                await self.executions.compare_and_set(expected, desired),
                operation="compare-and-set",
            )
            if committed is None:
                raise AssertionError("Compare-and-set validation returned no record.")
            if committed != desired:
                raise MemoryInterventionExecutionConflict(
                    "Execution store substituted another compare-and-set result."
                )
            return committed, True
        except MemoryInterventionExecutionConflict:
            current = _validated_store_record(
                await self.executions.load(expected.execution_id),
                operation="conflict readback",
                allow_missing=True,
            )
            if current is None or current.immutable_identity() != expected.immutable_identity():
                raise
            if _PHASE_ORDER[current.phase] < _PHASE_ORDER[desired.phase]:
                raise
            evidence_fields = (
                "materialization_fingerprint",
                "trial_binding_fingerprint",
                "operation_fingerprint",
                "receipt_fingerprint",
                "runtime_session_create_claim",
                "runtime_deadline_at",
                "runtime_evidence_fingerprint",
                "runtime_result_fingerprint",
                "runtime_result_payload",
                "eval_result_revision",
                "snapshot_result_fingerprint",
                "final_binding_fingerprint",
            )
            if any(
                getattr(desired, field) is not None
                and getattr(current, field) != getattr(desired, field)
                for field in evidence_fields
            ):
                raise MemoryInterventionExecutionConflict(
                    "Concurrent execution published conflicting evidence."
                ) from None
            if desired.runtime_cancellation_observed and not current.runtime_cancellation_observed:
                raise MemoryInterventionExecutionConflict(
                    "Concurrent execution omitted runtime cancellation authority."
                ) from None
            if desired.runtime_timeout_observed and not current.runtime_timeout_observed:
                raise MemoryInterventionExecutionConflict(
                    "Concurrent execution omitted runtime timeout authority."
                ) from None
            if desired.status is not MemoryInterventionExecutionStatus.ACTIVE and (
                current.status is not desired.status or current.failure_code != desired.failure_code
            ):
                raise MemoryInterventionExecutionConflict(
                    "Concurrent execution published another terminal state."
                ) from None
            return current, False


def _copy_trial_request(
    request: MemoryInterventionTrialRequest,
) -> MemoryInterventionTrialRequest:
    if type(request) is not MemoryInterventionTrialRequest:
        raise TypeError("request must be an exact MemoryInterventionTrialRequest.")
    return MemoryInterventionTrialRequest(
        schema_version=request.schema_version,
        spec=request.spec,
        candidate_id=request.candidate_id,
        trial_id=request.trial_id,
        case=request.case,
        run_request=request.run_request,
        timeout_seconds=request.timeout_seconds,
    )


def _terminal_effect_status(
    status: MemoryInterventionEffectStatus,
) -> MemoryInterventionExecutionStatus | None:
    if status is MemoryInterventionEffectStatus.CONFLICTING:
        return MemoryInterventionExecutionStatus.CONFLICTING
    if status is MemoryInterventionEffectStatus.INDETERMINATE:
        return MemoryInterventionExecutionStatus.INDETERMINATE
    return None


def memory_intervention_eval_result_revision(result: EvalTrialResult) -> str:
    """Compatibility name for the canonical eval trial-result identity."""

    return eval_trial_result_revision(result)


def memory_intervention_request_key(
    keys: Mapping[str, MemoryInterventionRequestFingerprintKey],
    key_id: str,
) -> MemoryInterventionRequestFingerprintKey:
    key_id = _clean(key_id, "key_id")
    key = keys.get(key_id)
    if type(key) is not MemoryInterventionRequestFingerprintKey or key.key_id != key_id:
        raise MemoryInterventionExecutionConflict(
            "The execution request fingerprint key is unavailable."
        )
    return key


__all__ = [
    "MEMORY_INTERVENTION_EXECUTION_MAX_RECORD_BYTES",
    "MEMORY_INTERVENTION_EXECUTION_MAX_TIMEOUT_SECONDS",
    "MEMORY_INTERVENTION_EXECUTION_RECORD_SCHEMA_VERSION",
    "MEMORY_INTERVENTION_EXECUTION_SCHEMA_VERSION",
    "CayuMemoryInterventionRuntimeRunner",
    "InMemoryMemoryInterventionExecutionStore",
    "MemoryInterventionEvaluator",
    "MemoryInterventionExecutionConflict",
    "MemoryInterventionExecutionPhase",
    "MemoryInterventionExecutionRecord",
    "MemoryInterventionExecutionStatus",
    "MemoryInterventionExecutionStore",
    "MemoryInterventionExecutor",
    "MemoryInterventionIsolationAuthority",
    "MemoryInterventionOverlayProvider",
    "MemoryInterventionRequestFingerprintKey",
    "MemoryInterventionRuntimeApplicationFactory",
    "MemoryInterventionRuntimeResult",
    "MemoryInterventionRuntimeRunner",
    "MemoryInterventionRuntimeView",
    "MemoryInterventionTrialOutcome",
    "MemoryInterventionTrialRequest",
    "SQLiteMemoryInterventionExecutionStore",
    "memory_intervention_eval_result_revision",
    "memory_intervention_request_key",
    "memory_intervention_runtime_result_fingerprint",
]
