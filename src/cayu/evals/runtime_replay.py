from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SkipValidation,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import canonical_durable_json_bytes, copy_durable_json_value
from cayu.core.billing import (
    BillingIdentity,
    completed_billing_identity,
    copy_billing_identity,
)
from cayu.core.events import Event, EventType
from cayu.core.messages import (
    Message,
    MessageRole,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.evals.models import (
    Trajectory,
    _trajectory_public_sha256,
    _validate_trajectory_record_contract,
)
from cayu.evals.promotion import _validated_trajectory_for_promotion
from cayu.evals.trajectory import final_output_text
from cayu.providers.base import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ProviderOperationMode,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_argument_publication as tool_argument_publication
from cayu.runtime._runtime_replay_profile import bind_runtime_replay_profile_source
from cayu.runtime._tool_identity import tool_idempotency_key
from cayu.runtime.app import CayuApp
from cayu.runtime.budgets import budget_limits_for_session, has_deferred_contextual_price
from cayu.runtime.context import (
    DefaultContextPolicy,
    MessageWindowContextPolicy,
    RecentTurnsContextPolicy,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileIdentity,
    changed_execution_profile_components,
    execution_profile_from_session_metadata,
)
from cayu.runtime.request_footprints import (
    RequestFingerprint,
    RequestFingerprintAvailability,
    RequestFootprint,
    RequestVariant,
)
from cayu.runtime.sessions import (
    InMemorySessionStore,
    ModelTarget,
    RunRequest,
    SessionStatus,
    session_input_messages_sha256,
    session_user_metadata,
)
from cayu.runtime.tool_exposure import (
    AllRegisteredToolsExposurePolicy,
    StaticToolExposurePolicy,
    ToolCapabilityCeiling,
    resolve_tool_capability_ceiling,
)
from cayu.runtime.tool_policy import AllowAllToolPolicy, StaticToolPolicy

RUNTIME_REPLAY_SCHEMA_VERSION = 1
RUNTIME_REPLAY_DEFAULT_MAX_EVENTS = 10_000
RUNTIME_REPLAY_HARD_MAX_EVENTS = 100_000
RUNTIME_REPLAY_DEFAULT_MAX_TRANSCRIPT_MESSAGES = 1_000
RUNTIME_REPLAY_HARD_MAX_TRANSCRIPT_MESSAGES = 10_000
RUNTIME_REPLAY_DEFAULT_MAX_MODEL_STEPS = 2
RUNTIME_REPLAY_HARD_MAX_MODEL_STEPS = 16
RUNTIME_REPLAY_DEFAULT_MAX_TOOL_CALLS = 16
RUNTIME_REPLAY_HARD_MAX_TOOL_CALLS = 256
RUNTIME_REPLAY_HARD_MAX_SUPPORTING_SOURCE_EVENTS = 2 * RUNTIME_REPLAY_HARD_MAX_TOOL_CALLS + 6
RUNTIME_REPLAY_DEFAULT_TIMEOUT_SECONDS = 30.0
RUNTIME_REPLAY_HARD_TIMEOUT_SECONDS = 300.0


def _runtime_replay_clock(instant: datetime) -> Callable[[], datetime]:
    replay_instant = instant.astimezone(UTC)

    def clock() -> datetime:
        return replay_instant

    return clock


class RuntimeReplayDisposition(StrEnum):
    MATCHED = "matched"
    DIVERGED = "diverged"
    UNAVAILABLE = "unavailable"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    FAILED = "failed"


class RuntimeReplayBoundaryKind(StrEnum):
    PREFLIGHT = "preflight"
    EXECUTION_PROFILE = "execution_profile"
    MODEL_REQUEST = "model_request"
    TOOL_POLICY = "tool_policy"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TERMINAL_OUTCOME = "terminal_outcome"


class RuntimeReplayDivergenceKind(StrEnum):
    EXECUTION_PROFILE_MISMATCH = "execution_profile_mismatch"
    REQUEST_FOOTPRINT_MISMATCH = "request_footprint_mismatch"
    POLICY_DECISION_MISMATCH = "policy_decision_mismatch"
    TOOL_CALL_MISMATCH = "tool_call_mismatch"
    RECORDED_OUTCOME_MISMATCH = "recorded_outcome_mismatch"
    TERMINAL_OUTCOME_MISMATCH = "terminal_outcome_mismatch"


class RuntimeReplayReason(StrEnum):
    SOURCE_SESSION_MISSING = "source_session_missing"
    SOURCE_SESSION_NOT_COMPLETED = "source_session_not_completed"
    SOURCE_TRAJECTORY_INVALID = "source_trajectory_invalid"
    SOURCE_CHILDREN_UNSUPPORTED = "source_children_unsupported"
    SOURCE_INPUT_EVIDENCE_UNAVAILABLE = "source_input_evidence_unavailable"
    SOURCE_INPUT_EVIDENCE_INCONSISTENT = "source_input_evidence_inconsistent"
    SOURCE_SHAPE_UNSUPPORTED = "source_shape_unsupported"
    SOURCE_MODEL_EVIDENCE_UNAVAILABLE = "source_model_evidence_unavailable"
    SOURCE_TOOL_EVIDENCE_UNAVAILABLE = "source_tool_evidence_unavailable"
    SOURCE_TOOL_ARGUMENT_EVIDENCE_UNAVAILABLE = "source_tool_argument_evidence_unavailable"
    SOURCE_REQUEST_FOOTPRINT_UNAVAILABLE = "source_request_footprint_unavailable"
    SOURCE_EXECUTION_PROFILE_UNAVAILABLE = "source_execution_profile_unavailable"
    SOURCE_BILLING_EVIDENCE_UNAVAILABLE = "source_billing_evidence_unavailable"
    SOURCE_INVOCATION_EVIDENCE_UNAVAILABLE = "source_invocation_evidence_unavailable"
    REQUEST_FINGERPRINT_INCOMPARABLE = "request_fingerprint_incomparable"
    CANDIDATE_AGENT_UNAVAILABLE = "candidate_agent_unavailable"
    CANDIDATE_MODEL_TARGET_UNAVAILABLE = "candidate_model_target_unavailable"
    CANDIDATE_ENVIRONMENT_UNSUPPORTED = "candidate_environment_unsupported"
    CANDIDATE_KNOWLEDGE_CONTEXT_UNSUPPORTED = "candidate_knowledge_context_unsupported"
    CANDIDATE_HOOKS_UNSUPPORTED = "candidate_hooks_unsupported"
    CANDIDATE_LOOP_POLICIES_UNSUPPORTED = "candidate_loop_policies_unsupported"
    CANDIDATE_CONTEXT_POLICY_UNSUPPORTED = "candidate_context_policy_unsupported"
    CANDIDATE_TOOL_POLICY_UNSUPPORTED = "candidate_tool_policy_unsupported"
    CANDIDATE_COMMAND_POLICY_UNSUPPORTED = "candidate_command_policy_unsupported"
    CANDIDATE_TOOL_EXPOSURE_UNSUPPORTED = "candidate_tool_exposure_unsupported"
    CANDIDATE_TOOL_LIFECYCLE_UNSUPPORTED = "candidate_tool_lifecycle_unsupported"
    CANDIDATE_TOOL_DISCOVERY_UNSUPPORTED = "candidate_tool_discovery_unsupported"
    CANDIDATE_MCP_UNSUPPORTED = "candidate_mcp_unsupported"
    CANDIDATE_HOSTED_TOOLS_UNSUPPORTED = "candidate_hosted_tools_unsupported"
    CANDIDATE_TOOL_RESULT_PROJECTION_UNSUPPORTED = "candidate_tool_result_projection_unsupported"
    CANDIDATE_PROVIDER_OPERATION_UNSUPPORTED = "candidate_provider_operation_unsupported"
    CANDIDATE_PROFILE_UNAVAILABLE = "candidate_profile_unavailable"
    CANDIDATE_EXECUTION_FAILED = "candidate_execution_failed"
    EVENT_BOUND_EXCEEDED = "event_bound_exceeded"
    TRANSCRIPT_BOUND_EXCEEDED = "transcript_bound_exceeded"
    MODEL_STEP_BOUND_EXCEEDED = "model_step_bound_exceeded"
    TOOL_CALL_BOUND_EXCEEDED = "tool_call_bound_exceeded"
    WALL_CLOCK_BOUND_EXCEEDED = "wall_clock_bound_exceeded"


class RuntimeReplayWarning(StrEnum):
    SOURCE_USAGE_UNAVAILABLE = "source_usage_unavailable"
    PROVIDER_WIRE_IDENTITY_UNAVAILABLE = "provider_wire_identity_unavailable"


class RuntimeReplayBounds(BaseModel):
    """Hard limits for one isolated replay attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_events: StrictInt = Field(
        default=RUNTIME_REPLAY_DEFAULT_MAX_EVENTS,
        ge=1,
        le=RUNTIME_REPLAY_HARD_MAX_EVENTS,
    )
    max_transcript_messages: StrictInt = Field(
        default=RUNTIME_REPLAY_DEFAULT_MAX_TRANSCRIPT_MESSAGES,
        ge=1,
        le=RUNTIME_REPLAY_HARD_MAX_TRANSCRIPT_MESSAGES,
    )
    max_model_steps: StrictInt = Field(
        default=RUNTIME_REPLAY_DEFAULT_MAX_MODEL_STEPS,
        ge=1,
        le=RUNTIME_REPLAY_HARD_MAX_MODEL_STEPS,
    )
    max_tool_calls: StrictInt = Field(
        default=RUNTIME_REPLAY_DEFAULT_MAX_TOOL_CALLS,
        ge=0,
        le=RUNTIME_REPLAY_HARD_MAX_TOOL_CALLS,
    )
    timeout_seconds: StrictFloat = Field(
        default=RUNTIME_REPLAY_DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        le=RUNTIME_REPLAY_HARD_TIMEOUT_SECONDS,
    )


class RuntimeReplayRequest(BaseModel):
    """Candidate app and immutable promoted trajectory selected for replay."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        frozen=True,
        hide_input_in_errors=True,
    )

    trajectory: SkipValidation[Trajectory] = Field(exclude=True, repr=False)
    agent_name: str | None = None
    bounds: RuntimeReplayBounds = Field(default_factory=RuntimeReplayBounds)

    @field_validator("trajectory")
    @classmethod
    def validate_trajectory(cls, value: object) -> Trajectory:
        # Retain the runtime-owned private promotion attestation. Revalidating an
        # exact Trajectory here would intentionally discard that non-serializable
        # authority before replay can verify it.
        if type(value) is not Trajectory:
            raise TypeError("trajectory must be an exact Trajectory.")
        return value

    @field_validator("agent_name")
    @classmethod
    def validate_agent_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if type(value) is not str or not value.strip() or value != value.strip():
            raise ValueError("agent_name must be a clean nonblank string.")
        return value


class RuntimeReplayFingerprintIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["hmac-sha256"] = "hmac-sha256"
    canonicalization_version: Literal[1] = 1
    key_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$",
    )
    value: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeReplayAttemptComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: StrictInt = Field(ge=1)
    source_event_id: str = Field(min_length=1, max_length=512)
    source_footprint_schema_version: StrictInt = Field(ge=1)
    candidate_footprint_schema_version: StrictInt = Field(ge=1)
    source_provider_name: str = Field(min_length=1, max_length=256)
    source_model: str = Field(min_length=1, max_length=512)
    candidate_provider_name: str = Field(min_length=1, max_length=256)
    candidate_model: str = Field(min_length=1, max_length=512)
    source_provider_neutral: RuntimeReplayFingerprintIdentity
    candidate_provider_neutral: RuntimeReplayFingerprintIdentity
    source_provider_wire: RuntimeReplayFingerprintIdentity | None = None
    candidate_provider_wire: RuntimeReplayFingerprintIdentity | None = None
    matched: StrictBool

    @model_validator(mode="after")
    def validate_match(self) -> RuntimeReplayAttemptComparison:
        expected = (
            self.source_footprint_schema_version == self.candidate_footprint_schema_version
            and self.source_provider_name == self.candidate_provider_name
            and self.source_model == self.candidate_model
            and self.source_provider_neutral == self.candidate_provider_neutral
            and self.source_provider_wire == self.candidate_provider_wire
        )
        if self.matched is not expected:
            raise ValueError("Replay request comparison match state is inconsistent.")
        return self


class RuntimeReplayDivergence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: RuntimeReplayDivergenceKind
    boundary: RuntimeReplayBoundaryKind
    index: StrictInt | None = Field(default=None, ge=1)
    source_event_id: str | None = Field(default=None, min_length=1, max_length=512)


class RuntimeReplayReport(BaseModel):
    """Secret-safe, versioned proof boundary for one runtime-contract replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = RUNTIME_REPLAY_SCHEMA_VERSION
    disposition: RuntimeReplayDisposition
    reason: RuntimeReplayReason | None = None
    source_session_identity: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    trajectory_identity: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    source_execution_profile: ExecutionProfileIdentity | None = None
    candidate_execution_profile: ExecutionProfileIdentity | None = None
    changed_execution_profile_components: tuple[str, ...] = Field(
        default=(),
        max_length=32,
    )
    request_attempts: tuple[RuntimeReplayAttemptComparison, ...] = Field(
        default=(),
        max_length=RUNTIME_REPLAY_HARD_MAX_MODEL_STEPS,
    )
    compared_model_steps: StrictInt = Field(default=0, ge=0)
    compared_tool_rounds: StrictInt = Field(default=0, ge=0)
    first_divergence: RuntimeReplayDivergence | None = None
    supporting_source_event_ids: tuple[str, ...] = Field(
        default=(),
        max_length=RUNTIME_REPLAY_HARD_MAX_SUPPORTING_SOURCE_EVENTS,
    )
    warnings: tuple[RuntimeReplayWarning, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_disposition(self) -> RuntimeReplayReport:
        if self.disposition is RuntimeReplayDisposition.MATCHED:
            if self.reason is not None or self.first_divergence is not None:
                raise ValueError("A matched replay cannot carry failure or divergence state.")
            if (
                self.source_session_identity is None
                or self.trajectory_identity is None
                or self.source_execution_profile is None
                or self.candidate_execution_profile is None
                or self.source_execution_profile != self.candidate_execution_profile
                or self.changed_execution_profile_components
                or len(self.request_attempts) != RUNTIME_REPLAY_DEFAULT_MAX_MODEL_STEPS
                or not all(item.matched for item in self.request_attempts)
                or self.compared_tool_rounds != 1
            ):
                raise ValueError("A matched replay requires complete agreeing V1 evidence.")
        elif self.disposition is RuntimeReplayDisposition.DIVERGED:
            if self.reason is not None or self.first_divergence is None:
                raise ValueError("A divergent replay requires one typed divergence only.")
            if (
                self.source_session_identity is None
                or self.trajectory_identity is None
                or self.source_execution_profile is None
                or self.candidate_execution_profile is None
            ):
                raise ValueError("A divergent replay requires both compared identities.")
        elif self.reason is None or self.first_divergence is not None:
            raise ValueError("A non-success replay requires one typed reason only.")
        attempt_indexes = tuple(item.index for item in self.request_attempts)
        if attempt_indexes != tuple(range(1, len(attempt_indexes) + 1)):
            raise ValueError("Replay request comparisons must use contiguous indexes.")
        if self.compared_model_steps != len(self.request_attempts):
            raise ValueError("Replay model-step count must match its request comparisons.")
        if self.compared_tool_rounds not in {0, 1}:
            raise ValueError("Runtime replay V1 compares at most one tool round.")
        if len(self.supporting_source_event_ids) != len(set(self.supporting_source_event_ids)):
            raise ValueError("Replay supporting event identities must be unique.")
        if len(self.warnings) != len(set(self.warnings)):
            raise ValueError("Replay warnings must be unique.")
        if len(self.changed_execution_profile_components) != len(
            set(self.changed_execution_profile_components)
        ):
            raise ValueError("Replay profile component changes must be unique.")
        if (
            self.source_execution_profile is not None
            and self.candidate_execution_profile is not None
        ):
            expected_changes = tuple(
                item.value
                for item in changed_execution_profile_components(
                    self.source_execution_profile,
                    self.candidate_execution_profile,
                )
            )
            if self.changed_execution_profile_components != expected_changes:
                raise ValueError("Replay profile changes do not match the compared identities.")
        elif self.changed_execution_profile_components:
            raise ValueError("Replay profile changes require both compared identities.")
        if (
            self.first_divergence is not None
            and self.first_divergence.source_event_id is not None
            and self.first_divergence.source_event_id not in self.supporting_source_event_ids
        ):
            raise ValueError("Replay divergence evidence must be listed as supporting evidence.")
        return self


@dataclass(frozen=True)
class _RecordedToolOutcome:
    call: ToolCallPart
    result: ToolResult
    source_effect: str
    source_idempotency_key: str
    source_tool_round_id: str
    source_model_step_id: str
    source_model_attempt_id: str
    source_terminal_event_type: EventType
    source_event_id: str | None


@dataclass
class _RecordedToolTracker:
    outcomes: Mapping[str, _RecordedToolOutcome]
    consumed: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    mismatched: bool = False
    mismatched_call_id: str | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def resolve(
        self,
        *,
        tool_name: str,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        call_id = context.metadata.get("tool_call_id")
        async with self._lock:
            if type(call_id) is not str or call_id in self.consumed:
                self.mismatched = True
                self.mismatched_call_id = call_id if type(call_id) is str else None
                raise RuntimeError("Recorded replay tool-call identity is unavailable.")
            outcome = self.outcomes.get(call_id)
            if (
                outcome is None
                or outcome.call.tool_name != tool_name
                or outcome.call.arguments != arguments
            ):
                self.mismatched = True
                self.mismatched_call_id = call_id
                raise RuntimeError("Recorded replay tool call does not match its fixture.")
            effect = context.metadata.get("tool_effect")
            if type(effect) is not str or effect != outcome.source_effect:
                self.mismatched = True
                self.mismatched_call_id = call_id
                raise RuntimeError("Recorded replay tool effect is unavailable.")
            idempotency_key = context.metadata.get("idempotency_key")
            if type(idempotency_key) is not str or not idempotency_key:
                self.mismatched = True
                self.mismatched_call_id = call_id
                raise RuntimeError("Recorded replay tool idempotency identity is unavailable.")
            self.consumed[call_id] = (tool_name, effect, idempotency_key)
            return ToolResult.model_validate(outcome.result.model_dump(mode="python"))


class _RecordedTool(Tool):
    def __init__(
        self,
        *,
        registered_tool: runtime_records.RegisteredTool,
        tracker: _RecordedToolTracker,
    ) -> None:
        self._tracker = tracker
        bind_runtime_replay_profile_source(self, registered_tool.tool)
        super().__init__(
            ToolSpec(
                name=registered_tool.name,
                description=registered_tool.description,
                input_schema=registered_tool.schema,
                parallel_safe=registered_tool.parallel_safe,
                effect=registered_tool.effect,
                workspace_mutation=registered_tool.workspace_mutation,
                execution_profile_identity=registered_tool.execution_profile_identity,
            )
        )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return await self._tracker.resolve(
            tool_name=self.name,
            context=ctx,
            arguments=args,
        )


class _RecordedProvider(ModelProvider):
    """Delegate pure request shaping while replacing every dispatch boundary."""

    def __init__(
        self,
        source: ModelProvider,
        batches: tuple[tuple[ModelStreamEvent, ...], ...],
        request_billing_identities: tuple[BillingIdentity | None, ...],
        completion_billing_identities: tuple[BillingIdentity | None, ...],
    ) -> None:
        if len(request_billing_identities) != len(batches) or len(
            completion_billing_identities
        ) != len(batches):
            raise ValueError("Recorded replay billing evidence must align with model batches.")
        self._source = source
        self._batches = batches
        self._request_index = 0
        self._billing_request_index = 0
        self._billing_completion_index = 0
        self._request_billing_identities = tuple(
            copy_billing_identity(identity) for identity in request_billing_identities
        )
        self._completion_billing_identities = tuple(
            copy_billing_identity(identity) for identity in completion_billing_identities
        )
        bind_runtime_replay_profile_source(self, source)
        self.name = source.name
        self.billing_provider_name = source.billing_provider_name
        self.usage_dialect = source.usage_dialect
        self.supports_native_structured_output = source.supports_native_structured_output

    @property
    def execution_profile_identity(self):
        return self._source.execution_profile_identity

    @property
    def context_pressure_profile(self):
        return self._source.context_pressure_profile

    @property
    def stream_deadlines(self):
        return self._source.stream_deadlines

    @property
    def provider_operation_mode(self) -> ProviderOperationMode:
        return ProviderOperationMode.SYNCHRONOUS

    @property
    def provider_operations(self):
        return None

    def request_cache_policy(self, request: ModelRequest):
        return self._source.request_cache_policy(request)

    def request_cache_projection(self, request: ModelRequest):
        return self._source.request_cache_projection(request)

    def request_footprint_options(self, request: ModelRequest) -> dict[str, Any]:
        return self._source.request_footprint_options(request)

    def request_fingerprint_options(self, request: ModelRequest) -> dict[str, Any]:
        return self._source.request_fingerprint_options(request)

    def preflight_model_target(self, *, model: str) -> None:
        self._source.preflight_model_target(model=model)

    def preflight_portable_messages(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        self._source.preflight_portable_messages(model=model, messages=messages, tools=tools)

    def preflight_hosted_tools(self, *, model: str, hosted_tools, options: dict[str, Any]) -> None:
        self._source.preflight_hosted_tools(
            model=model,
            hosted_tools=hosted_tools,
            options=options,
        )

    def preflight_native_structured_output_schema(self, json_schema: dict[str, Any]) -> None:
        self._source.preflight_native_structured_output_schema(json_schema)

    async def billing_identity_for_request(self, request: ModelRequest):
        del request
        index = self._billing_request_index
        self._billing_request_index += 1
        if index >= len(self._request_billing_identities):
            raise RuntimeError("Recorded replay provider exhausted its billing fixtures.")
        return copy_billing_identity(self._request_billing_identities[index])

    def billing_identity_for_completion(self, identity, payload: dict[str, Any]):
        del payload
        index = self._billing_completion_index
        self._billing_completion_index += 1
        if index >= len(self._completion_billing_identities):
            raise RuntimeError("Recorded replay provider exhausted its billing fixtures.")
        expected_request = self._request_billing_identities[index]
        if identity != expected_request:
            raise RuntimeError("Recorded replay request billing identity changed in flight.")
        return copy_billing_identity(self._completion_billing_identities[index])

    async def count_input_tokens(self, request: ModelRequest):
        del request
        return None

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        index = self._request_index
        self._request_index += 1
        if index >= len(self._batches):
            raise RuntimeError("Recorded replay provider exhausted its bounded fixtures.")
        for event in self._batches[index]:
            yield event.model_copy(deep=True)


@dataclass(frozen=True)
class _ReplayEvidence:
    trajectory: Trajectory
    replay_instant: datetime
    initial_messages: tuple[Message, ...]
    assistant_messages: tuple[Message, ...]
    model_completed_events: tuple[Event, ...]
    source_footprint_events: tuple[Event, ...]
    source_footprints: tuple[RequestFootprint, ...]
    request_billing_identities: tuple[BillingIdentity | None, ...]
    completion_billing_identities: tuple[BillingIdentity | None, ...]
    tool_outcomes: Mapping[str, _RecordedToolOutcome]
    source_execution_profile: ExecutionProfileIdentity
    supporting_source_event_ids: tuple[str, ...]
    warnings: tuple[RuntimeReplayWarning, ...]


class _ReplayUnavailable(RuntimeError):
    def __init__(self, reason: RuntimeReplayReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


class _ReplayBoundsExceeded(RuntimeError):
    def __init__(self, reason: RuntimeReplayReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


class _ReplayFailed(RuntimeError):
    def __init__(self, reason: RuntimeReplayReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _trajectory_identity(trajectory: Trajectory) -> str | None:
    try:
        if (
            trajectory.children
            or trajectory.children_incomplete
            or len(trajectory.events) > RUNTIME_REPLAY_HARD_MAX_EVENTS
            or len(trajectory.transcript) > RUNTIME_REPLAY_HARD_MAX_TRANSCRIPT_MESSAGES
        ):
            return None
        return f"sha256:{_trajectory_public_sha256(trajectory)}"
    except Exception:
        return None


def _session_identity(session_id: str) -> str:
    return f"sha256:{hashlib.sha256(session_id.encode('utf-8')).hexdigest()}"


def _base_report(
    trajectory: Trajectory,
    *,
    disposition: RuntimeReplayDisposition,
    reason: RuntimeReplayReason | None = None,
    evidence: _ReplayEvidence | None = None,
    candidate_profile: ExecutionProfileIdentity | None = None,
) -> RuntimeReplayReport:
    report_trajectory = trajectory if evidence is None else evidence.trajectory
    try:
        session_id = None if report_trajectory.session is None else report_trajectory.session.id
    except Exception:
        session_id = None
    source_profile = None if evidence is None else evidence.source_execution_profile
    profile_changes = (
        ()
        if source_profile is None or candidate_profile is None
        else tuple(
            item.value
            for item in changed_execution_profile_components(source_profile, candidate_profile)
        )
    )
    return RuntimeReplayReport(
        disposition=disposition,
        reason=reason,
        source_session_identity=(
            None if type(session_id) is not str else _session_identity(session_id)
        ),
        trajectory_identity=_trajectory_identity(report_trajectory),
        source_execution_profile=source_profile,
        candidate_execution_profile=candidate_profile,
        changed_execution_profile_components=profile_changes,
        supporting_source_event_ids=(
            () if evidence is None else evidence.supporting_source_event_ids
        ),
        warnings=() if evidence is None else evidence.warnings,
    )


def _require_bounds(trajectory: Trajectory, bounds: RuntimeReplayBounds) -> None:
    if len(trajectory.events) > bounds.max_events:
        raise _ReplayBoundsExceeded(RuntimeReplayReason.EVENT_BOUND_EXCEEDED)
    if len(trajectory.transcript) > bounds.max_transcript_messages:
        raise _ReplayBoundsExceeded(RuntimeReplayReason.TRANSCRIPT_BOUND_EXCEEDED)


def _available_identity(value: RequestFingerprint) -> RuntimeReplayFingerprintIdentity:
    if (
        value.availability is not RequestFingerprintAvailability.AVAILABLE
        or value.algorithm is None
        or value.key_id is None
        or value.value is None
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_REQUEST_FOOTPRINT_UNAVAILABLE)
    return RuntimeReplayFingerprintIdentity(
        algorithm=value.algorithm,
        canonicalization_version=value.canonicalization_version,
        key_id=value.key_id,
        value=value.value,
    )


def _recorded_billing_identities(
    *,
    events: tuple[Event, ...],
    footprints: tuple[RequestFootprint, ...],
    model_completed_events: tuple[Event, ...],
) -> tuple[
    tuple[BillingIdentity | None, ...],
    tuple[BillingIdentity | None, ...],
    tuple[str, ...],
]:
    request_identities: list[BillingIdentity | None] = []
    completion_identities: list[BillingIdentity | None] = []
    supporting_event_ids: list[str] = []
    for footprint, completed_event in zip(
        footprints,
        model_completed_events,
        strict=True,
    ):
        reservations = tuple(
            event
            for event in events
            if event.type is EventType.BUDGET_RESERVED
            and event.payload.get("model_step_id") == footprint.model_step_id
            and event.payload.get("model_attempt_id") == footprint.model_attempt_id
        )
        raw_request_identities = tuple(
            event.payload.get("billing_identity") for event in reservations
        )
        if any(
            event.payload.get("provider_name") != footprint.provider_name
            or event.payload.get("model") != footprint.model
            for event in reservations
        ) or (
            raw_request_identities
            and any(value is None for value in raw_request_identities)
            and any(value is not None for value in raw_request_identities)
        ):
            raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_BILLING_EVIDENCE_UNAVAILABLE)
        try:
            parsed_request_identities = tuple(
                BillingIdentity.model_validate(value)
                for value in raw_request_identities
                if value is not None
            )
            request_identity = (
                None if not parsed_request_identities else parsed_request_identities[0]
            )
            if any(identity != request_identity for identity in parsed_request_identities):
                raise ValueError("Conflicting recorded request billing identities.")
            raw_completion_identity = completed_event.payload.get("billing_identity")
            completion_identity = (
                None
                if request_identity is None
                else BillingIdentity.model_validate(raw_completion_identity)
            )
            if request_identity is not None:
                completion_identity = completed_billing_identity(
                    request_identity,
                    completion_identity,
                )
        except (TypeError, ValueError) as exc:
            raise _ReplayUnavailable(
                RuntimeReplayReason.SOURCE_BILLING_EVIDENCE_UNAVAILABLE
            ) from exc
        request_identities.append(copy_billing_identity(request_identity))
        completion_identities.append(copy_billing_identity(completion_identity))
        if request_identity is not None:
            supporting_event_ids.append(reservations[0].id)
    return (
        tuple(request_identities),
        tuple(completion_identities),
        tuple(supporting_event_ids),
    )


def _require_source_event_boundaries(
    *,
    events: tuple[Event, ...],
    footprint_events: tuple[Event, ...],
    footprints: tuple[RequestFootprint, ...],
    model_completed_events: tuple[Event, ...],
    first_calls: tuple[ToolCallPart, ...],
    started_tool_event_items: tuple[tuple[object, Event], ...],
    terminal_tool_event_items: tuple[tuple[object, Event], ...],
    source_profile: ExecutionProfileIdentity,
) -> None:
    model_started_events = tuple(event for event in events if event.type is EventType.MODEL_STARTED)
    interaction_started_events = tuple(
        event for event in events if event.type is EventType.INTERACTION_STARTED
    )
    interaction_completed_events = tuple(
        event for event in events if event.type is EventType.INTERACTION_COMPLETED
    )
    session_started_events = tuple(
        event for event in events if event.type is EventType.SESSION_STARTED
    )
    session_completed_events = tuple(
        event for event in events if event.type is EventType.SESSION_COMPLETED
    )
    if any(
        len(group) != expected
        for group, expected in (
            (model_started_events, 2),
            (interaction_started_events, 1),
            (interaction_completed_events, 1),
            (session_started_events, 1),
            (session_completed_events, 1),
        )
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_MODEL_EVIDENCE_UNAVAILABLE)
    if len({event.id for event in events}) != len(events):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TRAJECTORY_INVALID)

    event_positions = {event.id: index for index, event in enumerate(events)}
    started_tool_events = tuple(event for _, event in started_tool_event_items)
    terminal_tool_events = tuple(event for _, event in terminal_tool_event_items)
    first_footprint_event, second_footprint_event = footprint_events
    first_model_started, second_model_started = model_started_events
    first_model_completed, second_model_completed = model_completed_events
    first_call = first_calls[0]
    if (
        not (
            event_positions[interaction_started_events[0].id]
            < event_positions[session_started_events[0].id]
            < event_positions[first_footprint_event.id]
            < event_positions[first_model_started.id]
            < event_positions[first_model_completed.id]
            < min(event_positions[event.id] for event in started_tool_events)
        )
        or max(event_positions[event.id] for event in terminal_tool_events)
        >= event_positions[second_footprint_event.id]
        or not (
            event_positions[second_footprint_event.id]
            < event_positions[second_model_started.id]
            < event_positions[second_model_completed.id]
            < event_positions[interaction_completed_events[0].id]
            < event_positions[session_completed_events[0].id]
        )
        or any(
            event_positions[start_event.id] >= event_positions[terminal_event.id]
            for call_id, start_event in started_tool_event_items
            for terminal_call_id, terminal_event in terminal_tool_event_items
            if call_id == terminal_call_id
        )
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_MODEL_EVIDENCE_UNAVAILABLE)

    for index, (footprint, started, completed) in enumerate(
        zip(footprints, model_started_events, model_completed_events, strict=True),
        start=1,
    ):
        if (
            footprint.execution_profile_fingerprint != source_profile.fingerprint
            or footprint.step != index
            or footprint.attempt != 1
            or footprint.request_variant is not RequestVariant.INITIAL
            or started.payload.get("step") != index
            or started.payload.get("attempt") != 1
            or completed.payload.get("step") != index
            or completed.payload.get("attempt") != 1
            or started.payload.get("model_step_id") != footprint.model_step_id
            or started.payload.get("model_attempt_id") != footprint.model_attempt_id
            or completed.payload.get("model_step_id") != footprint.model_step_id
            or completed.payload.get("model_attempt_id") != footprint.model_attempt_id
            or started.payload.get("provider") != footprint.provider_name
            or completed.payload.get("provider_name") != footprint.provider_name
            or started.payload.get("model") != footprint.model
            or completed.payload.get("model") != footprint.model
            or started.payload.get("execution_profile_fingerprint") != source_profile.fingerprint
            or completed.payload.get("execution_profile_fingerprint") != source_profile.fingerprint
        ):
            raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_MODEL_EVIDENCE_UNAVAILABLE)

    if (
        first_call.model_step_id != footprints[0].model_step_id
        or first_call.model_attempt_id != footprints[0].model_attempt_id
        or first_model_completed.payload.get("tool_round_id") != first_call.tool_round_id
        or second_model_completed.payload.get("tool_round_id") is not None
        or any(
            event.payload.get("execution_profile_fingerprint") != source_profile.fingerprint
            for event in (*started_tool_events, *terminal_tool_events)
        )
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TOOL_EVIDENCE_UNAVAILABLE)


def _extract_evidence(
    trajectory: Trajectory,
    *,
    bounds: RuntimeReplayBounds,
) -> _ReplayEvidence:
    _require_bounds(trajectory, bounds)
    try:
        if trajectory.children or trajectory.children_incomplete:
            raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_CHILDREN_UNSUPPORTED)
    except _ReplayUnavailable:
        raise
    except Exception as exc:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TRAJECTORY_INVALID) from exc
    try:
        validated = _validated_trajectory_for_promotion(trajectory)
        _validate_trajectory_record_contract(validated)
    except Exception as exc:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TRAJECTORY_INVALID) from exc
    session = validated.session
    if session is None:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_SESSION_MISSING)
    if session.status is not SessionStatus.COMPLETED:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_SESSION_NOT_COMPLETED)
    if session.parent_session_id is not None:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_CHILDREN_UNSUPPORTED)
    if session.created_at.tzinfo is None or session.created_at.utcoffset() is None:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TRAJECTORY_INVALID)
    replay_instant = session.created_at.astimezone(UTC)

    start = validated.initial_input_message_start_index
    count = validated.initial_input_message_count
    expected_digest = validated.initial_input_messages_sha256
    if type(start) is not int or type(count) is not int or type(expected_digest) is not str:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_INPUT_EVIDENCE_UNAVAILABLE)
    if validated.input_redactions_applied is not False:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_INPUT_EVIDENCE_UNAVAILABLE)
    if validated.structured_output_requested is not False:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_SHAPE_UNSUPPORTED)
    end = start + count
    if start < 0 or count < 1 or end > len(validated.transcript):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_INPUT_EVIDENCE_INCONSISTENT)
    initial_messages = tuple(validated.transcript[start:end])
    if session_input_messages_sha256(initial_messages) != expected_digest:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_INPUT_EVIDENCE_INCONSISTENT)
    if any(
        message.role is not MessageRole.USER
        or not message.content
        or any(type(part) is not TextPart for part in message.content)
        for message in initial_messages
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_SHAPE_UNSUPPORTED)
    if any(
        message.role is MessageRole.USER and not start <= index < end
        for index, message in enumerate(validated.transcript)
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_SHAPE_UNSUPPORTED)

    unsupported_source_events = {
        EventType.SESSION_RESUMED,
        EventType.SESSION_INTERRUPTED,
        EventType.SESSION_AWAITING_USER_INPUT,
        EventType.SESSION_FORKED,
        EventType.SESSION_MESSAGE_QUEUED,
        EventType.SESSION_MESSAGE_DELIVERED,
        EventType.MODEL_RETRY,
        EventType.MODEL_ATTEMPT_DISCARDED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
        EventType.HOOK_FAILED,
        EventType.TOOL_CALL_APPROVAL_REQUESTED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.TOOL_CALL_APPROVAL_EXPIRED,
    }
    if any(event.type in unsupported_source_events for event in validated.events):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_SHAPE_UNSUPPORTED)

    assistant_messages = tuple(
        message for message in validated.transcript if message.role is MessageRole.ASSISTANT
    )
    if len(assistant_messages) > bounds.max_model_steps:
        raise _ReplayBoundsExceeded(RuntimeReplayReason.MODEL_STEP_BOUND_EXCEEDED)
    if len(assistant_messages) != 2:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_SHAPE_UNSUPPORTED)
    if any(
        type(part) not in {TextPart, ToolCallPart}
        for message in assistant_messages
        for part in message.content
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_MODEL_EVIDENCE_UNAVAILABLE)
    assistant_indexes = tuple(
        index
        for index, message in enumerate(validated.transcript)
        if message.role is MessageRole.ASSISTANT
    )
    tool_message_indexes = tuple(
        index
        for index, message in enumerate(validated.transcript)
        if message.role is MessageRole.TOOL
    )
    system_message_indexes = tuple(
        index
        for index, message in enumerate(validated.transcript)
        if message.role is MessageRole.SYSTEM
    )
    allowed_indexes = {
        *range(start, end),
        *assistant_indexes,
        *tool_message_indexes,
        *system_message_indexes,
    }
    if (
        len(assistant_indexes) != 2
        or not tool_message_indexes
        or any(
            not assistant_indexes[0] < index < assistant_indexes[1]
            for index in tool_message_indexes
        )
        or any(index >= start for index in system_message_indexes)
        or end > assistant_indexes[0]
        or allowed_indexes != set(range(len(validated.transcript)))
        or final_output_text(validated.transcript) != validated.final_output
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_SHAPE_UNSUPPORTED)
    first_calls = tuple(
        part for part in assistant_messages[0].content if type(part) is ToolCallPart
    )
    second_calls = tuple(
        part for part in assistant_messages[1].content if type(part) is ToolCallPart
    )
    if not first_calls or second_calls:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_SHAPE_UNSUPPORTED)
    if len(first_calls) > bounds.max_tool_calls:
        raise _ReplayBoundsExceeded(RuntimeReplayReason.TOOL_CALL_BOUND_EXCEEDED)
    call_ids = tuple(call.tool_call_id for call in first_calls)
    if len(call_ids) != len(set(call_ids)):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TOOL_EVIDENCE_UNAVAILABLE)
    source_round_identities = {
        (call.tool_round_id, call.model_step_id, call.model_attempt_id) for call in first_calls
    }
    if len(source_round_identities) != 1 or any(
        any(identity is None for identity in round_identity)
        for round_identity in source_round_identities
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TOOL_EVIDENCE_UNAVAILABLE)

    all_result_parts = tuple(
        part
        for message in validated.transcript
        for part in message.content
        if type(part) is ToolResultPart
    )
    result_parts = tuple(part for part in all_result_parts if part.tool_call_id in set(call_ids))
    if len(all_result_parts) != len(first_calls) or len(result_parts) != len(first_calls):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TOOL_EVIDENCE_UNAVAILABLE)
    results_by_id = {part.tool_call_id: part for part in result_parts}
    if len(results_by_id) != len(result_parts):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TOOL_EVIDENCE_UNAVAILABLE)

    started_tool_event_items = tuple(
        (event.payload.get("tool_call_id"), event)
        for event in validated.events
        if event.type == EventType.TOOL_CALL_STARTED
        and type(event.payload.get("tool_call_id")) is str
    )
    terminal_tool_event_items = tuple(
        (event.payload.get("tool_call_id"), event)
        for event in validated.events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        and type(event.payload.get("tool_call_id")) is str
    )
    started_tool_events = dict(started_tool_event_items)
    terminal_tool_events = dict(terminal_tool_event_items)
    if (
        len(started_tool_event_items) != len(first_calls)
        or len(terminal_tool_event_items) != len(first_calls)
        or len(started_tool_events) != len(started_tool_event_items)
        or len(terminal_tool_events) != len(terminal_tool_event_items)
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TOOL_EVIDENCE_UNAVAILABLE)
    tool_outcomes: dict[str, _RecordedToolOutcome] = {}
    for call in first_calls:
        result = results_by_id.get(call.tool_call_id)
        if result is None or result.tool_name != call.tool_name:
            raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TOOL_EVIDENCE_UNAVAILABLE)
        terminal = terminal_tool_events.get(call.tool_call_id)
        started = started_tool_events.get(call.tool_call_id)
        if terminal is None or started is None:
            raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TOOL_EVIDENCE_UNAVAILABLE)
        source_effect = started.payload.get("effect")
        started_idempotency = started.payload.get("idempotency_key")
        terminal_idempotency = terminal.payload.get("idempotency_key")
        terminal_arguments = terminal.payload.get("arguments")
        terminal_arguments_exact = terminal.payload.get(
            tool_argument_publication.ARGUMENTS_EXACT_FIELD
        )
        terminal_result = terminal.payload.get("result")
        source_tool_round_id = call.tool_round_id
        source_model_step_id = call.model_step_id
        source_model_attempt_id = call.model_attempt_id
        try:
            ToolEffect(source_effect)
            recorded_result = ToolResult(
                content=result.content,
                structured=result.structured,
                artifacts=result.artifacts,
                is_error=result.is_error,
            )
        except (TypeError, ValueError) as exc:
            raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TOOL_EVIDENCE_UNAVAILABLE) from exc
        if terminal_arguments_exact is not True:
            raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TOOL_ARGUMENT_EVIDENCE_UNAVAILABLE)
        if (
            type(source_effect) is not str
            or type(started_idempotency) is not str
            or type(source_tool_round_id) is not str
            or type(source_model_step_id) is not str
            or type(source_model_attempt_id) is not str
            or started_idempotency != terminal_idempotency
            or started_idempotency
            != tool_idempotency_key(
                session_id=session.id,
                tool_call_id=call.tool_call_id,
                tool_round_id=source_tool_round_id,
            )
            or started.tool_name != call.tool_name
            or terminal.tool_name != call.tool_name
            or any(
                event.payload.get(field_name) != expected
                for event in (started, terminal)
                for field_name, expected in (
                    ("tool_round_id", source_tool_round_id),
                    ("model_step_id", source_model_step_id),
                    ("model_attempt_id", source_model_attempt_id),
                )
            )
            or result.tool_round_id != source_tool_round_id
            or result.model_step_id != source_model_step_id
            or result.model_attempt_id != source_model_attempt_id
            or terminal.type
            is not (
                EventType.TOOL_CALL_FAILED
                if recorded_result.is_error
                else EventType.TOOL_CALL_COMPLETED
            )
            or terminal_arguments != call.arguments
            or terminal_result != recorded_result.model_dump(mode="json")
        ):
            raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_TOOL_EVIDENCE_UNAVAILABLE)
        tool_outcomes[call.tool_call_id] = _RecordedToolOutcome(
            call=call,
            result=recorded_result,
            source_effect=source_effect,
            source_idempotency_key=started_idempotency,
            source_tool_round_id=source_tool_round_id,
            source_model_step_id=source_model_step_id,
            source_model_attempt_id=source_model_attempt_id,
            source_terminal_event_type=terminal.type,
            source_event_id=terminal.id,
        )

    model_completed_events = tuple(
        event for event in validated.events if event.type == EventType.MODEL_COMPLETED
    )
    footprint_events = tuple(
        event for event in validated.events if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
    )
    if len(model_completed_events) != 2:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_MODEL_EVIDENCE_UNAVAILABLE)
    if len(footprint_events) != 2:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_REQUEST_FOOTPRINT_UNAVAILABLE)
    try:
        footprints = tuple(
            RequestFootprint.model_validate(event.payload) for event in footprint_events
        )
    except (TypeError, ValueError) as exc:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_REQUEST_FOOTPRINT_UNAVAILABLE) from exc
    for footprint in footprints:
        _available_identity(footprint.fingerprints.provider_neutral_request)

    try:
        source_profile = execution_profile_from_session_metadata(session.metadata)
    except (TypeError, ValueError) as exc:
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_EXECUTION_PROFILE_UNAVAILABLE) from exc

    _require_source_event_boundaries(
        events=validated.events,
        footprint_events=footprint_events,
        footprints=footprints,
        model_completed_events=model_completed_events,
        first_calls=first_calls,
        started_tool_event_items=started_tool_event_items,
        terminal_tool_event_items=terminal_tool_event_items,
        source_profile=source_profile,
    )
    (
        request_billing_identities,
        completion_billing_identities,
        billing_event_ids,
    ) = _recorded_billing_identities(
        events=validated.events,
        footprints=footprints,
        model_completed_events=model_completed_events,
    )

    warning_values: list[RuntimeReplayWarning] = []
    if any(event.payload.get("usage_metrics") is None for event in model_completed_events):
        warning_values.append(RuntimeReplayWarning.SOURCE_USAGE_UNAVAILABLE)
    if any(
        footprint.fingerprints.provider_wire_request.availability
        is not RequestFingerprintAvailability.AVAILABLE
        for footprint in footprints
    ):
        warning_values.append(RuntimeReplayWarning.PROVIDER_WIRE_IDENTITY_UNAVAILABLE)
    supporting = tuple(
        dict.fromkeys(
            [
                *(event.id for event in footprint_events),
                *(event.id for event in model_completed_events),
                *(event.id for _, event in started_tool_event_items),
                *(event.id for _, event in terminal_tool_event_items),
                *billing_event_ids,
            ]
        )
    )
    return _ReplayEvidence(
        trajectory=validated,
        replay_instant=replay_instant,
        initial_messages=initial_messages,
        assistant_messages=assistant_messages,
        model_completed_events=model_completed_events,
        source_footprint_events=footprint_events,
        source_footprints=footprints,
        request_billing_identities=request_billing_identities,
        completion_billing_identities=completion_billing_identities,
        tool_outcomes=MappingProxyType(tool_outcomes),
        source_execution_profile=source_profile,
        supporting_source_event_ids=supporting,
        warnings=tuple(warning_values),
    )


def _require_candidate_boundary(
    app: CayuApp,
    agent_name: str,
    *,
    provider_name: str,
) -> runtime_records.RegisteredAgentState:
    registered_agent = app._agents.get(agent_name)
    if registered_agent is None:
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_AGENT_UNAVAILABLE)
    if provider_name not in app._providers:
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_MODEL_TARGET_UNAVAILABLE)
    if app._default_environment_name is not None or app._environments:
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_ENVIRONMENT_UNSUPPORTED)
    if app.knowledge_store is not None:
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_KNOWLEDGE_CONTEXT_UNSUPPORTED)
    if app._runtime_hooks or registered_agent.runtime_hooks:
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_HOOKS_UNSUPPORTED)
    if app._loop_policies or registered_agent.loop_policies:
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_LOOP_POLICIES_UNSUPPORTED)
    if (
        type(registered_agent.context_policy)
        not in {
            DefaultContextPolicy,
            MessageWindowContextPolicy,
            RecentTurnsContextPolicy,
        }
        or registered_agent.context_overflow_policy is not None
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_CONTEXT_POLICY_UNSUPPORTED)
    if type(registered_agent.tool_policy) not in {AllowAllToolPolicy, StaticToolPolicy}:
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_TOOL_POLICY_UNSUPPORTED)
    if any(
        getattr(registered_tool.tool, "command_policy", None) is not None
        for registered_tool in registered_agent.tools.values()
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_COMMAND_POLICY_UNSUPPORTED)
    if any(
        registered_tool.child_session_recovery is not None
        or registered_tool.durable_tool_recovery is not None
        or getattr(registered_tool.tool, "pauses_session", False) is not False
        for registered_tool in registered_agent.tools.values()
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_TOOL_LIFECYCLE_UNSUPPORTED)
    if type(registered_agent.tool_exposure_policy) not in {
        AllRegisteredToolsExposurePolicy,
        StaticToolExposurePolicy,
    }:
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_TOOL_EXPOSURE_UNSUPPORTED)
    if (
        registered_agent.targeted_tool_mode is not None
        or registered_agent.tool_discovery_mode is not None
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_TOOL_DISCOVERY_UNSUPPORTED)
    if registered_agent.mcp_toolsets or registered_agent.runtime_tools:
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_MCP_UNSUPPORTED)
    if registered_agent.hosted_tools:
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_HOSTED_TOOLS_UNSUPPORTED)
    if app._tool_result_projection_policy is not None:
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_TOOL_RESULT_PROJECTION_UNSUPPORTED)
    if any(
        registered.provider.provider_operation_mode is not ProviderOperationMode.SYNCHRONOUS
        or registered.provider.provider_operations is not None
        for registered in app._providers.values()
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_PROVIDER_OPERATION_UNSUPPORTED)
    return registered_agent


def _require_candidate_billing_evidence(
    app: CayuApp,
    evidence: _ReplayEvidence,
    *,
    agent_name: str,
) -> None:
    session = evidence.trajectory.session
    if session is None:  # pragma: no cover - evidence preflight owns this invariant
        raise AssertionError("Replay evidence lost its source session.")
    registered_provider = app._providers.get(session.provider_name)
    if registered_provider is None:  # pragma: no cover - candidate preflight owns this invariant
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_MODEL_TARGET_UNAVAILABLE)
    commercial_provider = (
        registered_provider.provider.billing_provider_name or session.provider_name
    )
    limits = budget_limits_for_session(
        policy=app.budget_policy,
        agent_name=agent_name,
        causal_budget_id=session.causal_budget_id,
    )
    requires_identity = any(
        has_deferred_contextual_price(
            limit.pricing,
            provider_name=commercial_provider,
            model=session.model,
        )
        for limit in limits
    )
    if requires_identity and any(
        identity is None for identity in evidence.request_billing_identities
    ):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_BILLING_EVIDENCE_UNAVAILABLE)


_UNATTESTED_INVOCATION_COMPONENTS = frozenset(
    {
        ExecutionProfileComponentClass.INVOCATION_POLICIES,
        ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
        ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
        ExecutionProfileComponentClass.FINALIZATION,
    }
)


def _require_attributable_invocation_profile(
    evidence: _ReplayEvidence,
    candidate_profile: ExecutionProfileIdentity,
) -> None:
    """Reject profile differences that retained source evidence cannot attribute.

    A terminal trajectory retains the resulting execution-profile identities but
    not every caller-owned ``RunRequest`` object that produced them. Rebuilding
    those settings from the candidate application's defaults can therefore prove
    equality, but a difference is ambiguous: it may be candidate drift or a
    source invocation override that replay cannot reconstruct. Never report that
    ambiguity as a regression.
    """

    changed = changed_execution_profile_components(
        evidence.source_execution_profile,
        candidate_profile,
    )
    if any(component in _UNATTESTED_INVOCATION_COMPONENTS for component in changed):
        raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_INVOCATION_EVIDENCE_UNAVAILABLE)


def _model_completion_fixture_payload(event: Event) -> dict[str, Any]:
    allowed = {
        "completion",
        "end_turn",
        "finish_reason",
        "incomplete_details",
        "model",
        "raw_finish_reason",
        "reason",
        "status",
        "stop_reason",
        "usage",
        "usage_metrics",
    }
    return copy_durable_json_value(
        {key: value for key, value in event.payload.items() if key in allowed},
        "runtime replay completion fixture",
    )


def _provider_batches(evidence: _ReplayEvidence) -> tuple[tuple[ModelStreamEvent, ...], ...]:
    batches: list[tuple[ModelStreamEvent, ...]] = []
    for message, completion_event in zip(
        evidence.assistant_messages,
        evidence.model_completed_events,
        strict=True,
    ):
        events: list[ModelStreamEvent] = []
        for part in message.content:
            if type(part) is TextPart:
                events.append(ModelStreamEvent.text_delta(part.text))
            elif type(part) is ToolCallPart:
                events.append(
                    ModelStreamEvent.tool_call(
                        id=part.tool_call_id,
                        name=part.tool_name,
                        arguments=part.arguments,
                    )
                )
            else:  # pragma: no cover - preflight owns this invariant
                raise AssertionError("Unsupported assistant replay fixture escaped preflight.")
        events.append(
            ModelStreamEvent.completed(_model_completion_fixture_payload(completion_event))
        )
        batches.append(tuple(events))
    return tuple(batches)


def _isolated_app(
    app: CayuApp,
    *,
    provider_batches: tuple[tuple[ModelStreamEvent, ...], ...] | None,
    request_billing_identities: tuple[BillingIdentity | None, ...] | None,
    completion_billing_identities: tuple[BillingIdentity | None, ...] | None,
    tool_tracker: _RecordedToolTracker | None,
    clock_instant: datetime,
) -> CayuApp:
    store = InMemorySessionStore(public_authority_alias_codec=app._public_authority_alias_codec)
    isolated = CayuApp(
        session_store=store,
        budget_policy=app.budget_policy,
        retry_policy=app._default_retry_policy,
        loop_policies=(),
        context_counting=app._context_counting,
        request_footprint=app._request_footprint,
        enable_logging=False,
        secret_redactor=app._secret_redactor,
        max_file_attachment_bytes=app._max_file_attachment_bytes,
        max_total_file_attachment_bytes=app._max_total_file_attachment_bytes,
        max_file_attachments_per_request=app._max_file_attachments_per_request,
        tool_timeout_seconds=app._tool_timeout_seconds,
        max_parallel_tool_calls=app._max_parallel_tool_calls,
        max_environment_lifecycle_owners=app._max_environment_lifecycle_owners,
        clock=_runtime_replay_clock(clock_instant),
    )
    isolated._execution_profile_process_identity = app._execution_profile_process_identity
    isolated._session_engine._execution_profile_process_identity = (
        app._execution_profile_process_identity
    )
    isolated._default_provider_name = app._default_provider_name
    isolated._default_environment_name = None
    if provider_batches is not None and (
        request_billing_identities is None or completion_billing_identities is None
    ):
        raise AssertionError("Replay execution requires recorded billing evidence.")
    isolated._providers = {
        name: (
            registered
            if provider_batches is None
            else replace(
                registered,
                provider=_RecordedProvider(
                    registered.provider,
                    provider_batches,
                    request_billing_identities or (),
                    completion_billing_identities or (),
                ),
            )
        )
        for name, registered in app._providers.items()
    }
    if provider_batches is None:
        isolated._agents = dict(app._agents)
        return isolated
    if tool_tracker is None:
        raise AssertionError("Replay execution requires a tool tracker.")
    replay_agents: dict[str, runtime_records.RegisteredAgentState] = {}
    for name, registered_agent in app._agents.items():
        replay_tools = {
            tool_name: replace(
                registered_tool,
                tool=_RecordedTool(
                    registered_tool=registered_tool,
                    tracker=tool_tracker,
                ),
                child_session_recovery=None,
                durable_tool_recovery=None,
            )
            for tool_name, registered_tool in registered_agent.tools.items()
        }
        replay_agents[name] = replace(
            registered_agent,
            tools=MappingProxyType(replay_tools),
            runtime_tools=MappingProxyType({}),
            mcp_toolsets=(),
        )
    isolated._agents = replay_agents
    return isolated


def _run_request(
    evidence: _ReplayEvidence,
    *,
    agent_name: str,
    session_id: str,
    tool_capability_ceiling: ToolCapabilityCeiling,
) -> RunRequest:
    session = evidence.trajectory.session
    if session is None:  # pragma: no cover - evidence preflight owns this invariant
        raise AssertionError("Replay evidence lost its source session.")
    return RunRequest(
        agent_name=agent_name,
        messages=list(evidence.initial_messages),
        session_id=session_id,
        causal_budget_id=session.causal_budget_id,
        target=ModelTarget(
            provider_name=session.provider_name,
            model=session.model,
        ),
        tool_capability_ceiling=tool_capability_ceiling,
        labels=session.labels,
        metadata=session_user_metadata(session.metadata),
    )


async def _candidate_profile(
    app: CayuApp,
    evidence: _ReplayEvidence,
    *,
    agent_name: str,
    tool_capability_ceiling: ToolCapabilityCeiling,
) -> ExecutionProfileIdentity:
    isolated = _isolated_app(
        app,
        provider_batches=None,
        request_billing_identities=None,
        completion_billing_identities=None,
        tool_tracker=None,
        clock_instant=evidence.replay_instant,
    )
    prepared = await isolated._session_engine._prepare_initial_run(
        _run_request(
            evidence,
            agent_name=agent_name,
            session_id=f"replay-profile-{uuid4()}",
            tool_capability_ceiling=tool_capability_ceiling,
        ),
        admit_session=False,
        store_resolved_existing_session_id=None,
    )
    if prepared is None:
        raise _ReplayUnavailable(RuntimeReplayReason.CANDIDATE_PROFILE_UNAVAILABLE)
    return prepared.execution_profile


def _attempt_comparisons(
    evidence: _ReplayEvidence,
    candidate_events: tuple[Event, ...],
    *,
    require_complete: bool = True,
) -> tuple[RuntimeReplayAttemptComparison, ...]:
    candidate_footprint_events = tuple(
        event for event in candidate_events if event.type == EventType.REQUEST_FOOTPRINT_RECORDED
    )
    if (
        not candidate_footprint_events
        or len(candidate_footprint_events) > len(evidence.source_footprints)
        or (require_complete and len(candidate_footprint_events) != len(evidence.source_footprints))
    ):
        raise _ReplayFailed(RuntimeReplayReason.CANDIDATE_EXECUTION_FAILED)
    try:
        candidate_footprints = tuple(
            RequestFootprint.model_validate(event.payload) for event in candidate_footprint_events
        )
    except (TypeError, ValueError) as exc:
        raise _ReplayFailed(RuntimeReplayReason.CANDIDATE_EXECUTION_FAILED) from exc

    comparisons: list[RuntimeReplayAttemptComparison] = []
    compared_count = len(candidate_footprints)
    for index, (source_event, source, candidate) in enumerate(
        zip(
            evidence.source_footprint_events[:compared_count],
            evidence.source_footprints[:compared_count],
            candidate_footprints,
            strict=True,
        ),
        start=1,
    ):
        source_neutral = _available_identity(source.fingerprints.provider_neutral_request)
        source_wire = (
            _available_identity(source.fingerprints.provider_wire_request)
            if source.fingerprints.provider_wire_request.availability
            is RequestFingerprintAvailability.AVAILABLE
            else None
        )
        try:
            candidate_neutral = _available_identity(candidate.fingerprints.provider_neutral_request)
            candidate_wire = (
                _available_identity(candidate.fingerprints.provider_wire_request)
                if source_wire is not None
                else None
            )
        except _ReplayUnavailable as exc:
            raise _ReplayUnavailable(RuntimeReplayReason.REQUEST_FINGERPRINT_INCOMPARABLE) from exc
        if source_neutral.key_id != candidate_neutral.key_id or (
            source_wire is not None
            and candidate_wire is not None
            and source_wire.key_id != candidate_wire.key_id
        ):
            raise _ReplayUnavailable(RuntimeReplayReason.REQUEST_FINGERPRINT_INCOMPARABLE)
        matched = (
            source.schema_version == candidate.schema_version
            and source.provider_name == candidate.provider_name
            and source.model == candidate.model
            and source_neutral == candidate_neutral
            and source_wire == candidate_wire
        )
        comparisons.append(
            RuntimeReplayAttemptComparison(
                index=index,
                source_event_id=source_event.id,
                source_footprint_schema_version=source.schema_version,
                candidate_footprint_schema_version=candidate.schema_version,
                source_provider_name=source.provider_name,
                source_model=source.model,
                candidate_provider_name=candidate.provider_name,
                candidate_model=candidate.model,
                source_provider_neutral=source_neutral,
                candidate_provider_neutral=candidate_neutral,
                source_provider_wire=source_wire,
                candidate_provider_wire=candidate_wire,
                matched=matched,
            )
        )
    return tuple(comparisons)


def _terminal_fingerprint(trajectory: Trajectory) -> str:
    session = trajectory.session
    material = {
        "status": None if session is None else session.status.value,
        "final_output_sha256": hashlib.sha256(trajectory.final_output.encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(
        canonical_durable_json_bytes(material, "runtime_replay_terminal")
    ).hexdigest()


def _source_tool_boundary(
    evidence: _ReplayEvidence,
    tool_call_id: object,
) -> tuple[int, _RecordedToolOutcome]:
    for index, (source_call_id, outcome) in enumerate(
        evidence.tool_outcomes.items(),
        start=1,
    ):
        if source_call_id == tool_call_id:
            return index, outcome
    return 1, next(iter(evidence.tool_outcomes.values()))


@dataclass(frozen=True)
class _CandidateToolDivergence:
    kind: RuntimeReplayDivergenceKind
    index: int
    source_event_id: str | None


def _candidate_tool_divergence(
    evidence: _ReplayEvidence,
    *,
    candidate_events: tuple[Event, ...],
    candidate_transcript: tuple[Message, ...],
    replay_session_id: str,
    tracker: _RecordedToolTracker,
) -> _CandidateToolDivergence | None:
    started_items = tuple(
        (event.payload.get("tool_call_id"), event)
        for event in candidate_events
        if event.type is EventType.TOOL_CALL_STARTED
        and type(event.payload.get("tool_call_id")) is str
    )
    terminal_items = tuple(
        (event.payload.get("tool_call_id"), event)
        for event in candidate_events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        and type(event.payload.get("tool_call_id")) is str
    )
    started_by_id = dict(started_items)
    terminal_by_id = dict(terminal_items)
    expected_ids = set(evidence.tool_outcomes)
    if (
        len(started_items) != len(expected_ids)
        or len(terminal_items) != len(expected_ids)
        or len(started_by_id) != len(started_items)
        or len(terminal_by_id) != len(terminal_items)
        or set(started_by_id) != expected_ids
        or set(terminal_by_id) != expected_ids
    ):
        for index, (call_id, outcome) in enumerate(evidence.tool_outcomes.items(), start=1):
            if call_id not in started_by_id or call_id not in terminal_by_id:
                return _CandidateToolDivergence(
                    kind=RuntimeReplayDivergenceKind.TOOL_CALL_MISMATCH,
                    index=index,
                    source_event_id=outcome.source_event_id,
                )
        first_outcome = next(iter(evidence.tool_outcomes.values()))
        return _CandidateToolDivergence(
            kind=RuntimeReplayDivergenceKind.TOOL_CALL_MISMATCH,
            index=1,
            source_event_id=first_outcome.source_event_id,
        )

    candidate_calls = tuple(
        part
        for message in candidate_transcript
        for part in message.content
        if type(part) is ToolCallPart
    )
    candidate_results = tuple(
        part
        for message in candidate_transcript
        for part in message.content
        if type(part) is ToolResultPart
    )
    calls_by_id = {part.tool_call_id: part for part in candidate_calls}
    results_by_id = {part.tool_call_id: part for part in candidate_results}
    if (
        len(candidate_calls) != len(expected_ids)
        or len(candidate_results) != len(expected_ids)
        or len(calls_by_id) != len(candidate_calls)
        or len(results_by_id) != len(candidate_results)
        or set(calls_by_id) != expected_ids
        or set(results_by_id) != expected_ids
    ):
        for index, (call_id, outcome) in enumerate(evidence.tool_outcomes.items(), start=1):
            if call_id not in calls_by_id or call_id not in results_by_id:
                return _CandidateToolDivergence(
                    kind=RuntimeReplayDivergenceKind.TOOL_CALL_MISMATCH,
                    index=index,
                    source_event_id=outcome.source_event_id,
                )
        first_outcome = next(iter(evidence.tool_outcomes.values()))
        return _CandidateToolDivergence(
            kind=RuntimeReplayDivergenceKind.TOOL_CALL_MISMATCH,
            index=1,
            source_event_id=first_outcome.source_event_id,
        )

    for call_id, outcome in evidence.tool_outcomes.items():
        started = started_by_id[call_id]
        terminal = terminal_by_id[call_id]
        candidate_call = calls_by_id[call_id]
        candidate_result = results_by_id[call_id]
        round_identity = (
            candidate_call.tool_round_id,
            candidate_call.model_step_id,
            candidate_call.model_attempt_id,
        )
        if any(type(value) is not str for value in round_identity):
            index, _ = _source_tool_boundary(evidence, call_id)
            return _CandidateToolDivergence(
                kind=RuntimeReplayDivergenceKind.TOOL_CALL_MISMATCH,
                index=index,
                source_event_id=outcome.source_event_id,
            )
        tool_round_id, model_step_id, model_attempt_id = round_identity
        candidate_idempotency = tool_idempotency_key(
            session_id=replay_session_id,
            tool_call_id=call_id,
            tool_round_id=tool_round_id,
        )
        consumed = tracker.consumed.get(call_id)
        if (
            candidate_call.tool_name != outcome.call.tool_name
            or candidate_call.arguments != outcome.call.arguments
            or started.tool_name != outcome.call.tool_name
            or terminal.tool_name != outcome.call.tool_name
            or terminal.type is not outcome.source_terminal_event_type
            or started.payload.get("effect") != outcome.source_effect
            or consumed != (outcome.call.tool_name, outcome.source_effect, candidate_idempotency)
            or any(
                event.payload.get(field_name) != expected
                for event in (started, terminal)
                for field_name, expected in (
                    ("tool_round_id", tool_round_id),
                    ("model_step_id", model_step_id),
                    ("model_attempt_id", model_attempt_id),
                    ("idempotency_key", candidate_idempotency),
                )
            )
            or candidate_result.tool_name != outcome.call.tool_name
            or candidate_result.tool_round_id != tool_round_id
            or candidate_result.model_step_id != model_step_id
            or candidate_result.model_attempt_id != model_attempt_id
        ):
            index, _ = _source_tool_boundary(evidence, call_id)
            return _CandidateToolDivergence(
                kind=RuntimeReplayDivergenceKind.TOOL_CALL_MISMATCH,
                index=index,
                source_event_id=outcome.source_event_id,
            )
        if (
            ToolResult(
                content=candidate_result.content,
                structured=candidate_result.structured,
                artifacts=candidate_result.artifacts,
                is_error=candidate_result.is_error,
            )
            != outcome.result
        ):
            index, _ = _source_tool_boundary(evidence, call_id)
            return _CandidateToolDivergence(
                kind=RuntimeReplayDivergenceKind.RECORDED_OUTCOME_MISMATCH,
                index=index,
                source_event_id=outcome.source_event_id,
            )
    return None


async def _execute_replay(
    app: CayuApp,
    evidence: _ReplayEvidence,
    *,
    agent_name: str,
    bounds: RuntimeReplayBounds,
    candidate_profile: ExecutionProfileIdentity,
    candidate_tool_capability_ceiling: ToolCapabilityCeiling,
) -> RuntimeReplayReport:
    tracker = _RecordedToolTracker(evidence.tool_outcomes)
    isolated = _isolated_app(
        app,
        provider_batches=_provider_batches(evidence),
        request_billing_identities=evidence.request_billing_identities,
        completion_billing_identities=evidence.completion_billing_identities,
        tool_tracker=tracker,
        clock_instant=evidence.replay_instant,
    )
    replay_session_id = f"runtime-replay-{uuid4()}"
    candidate_events: list[Event] = []
    try:
        async for event in isolated.run(
            _run_request(
                evidence,
                agent_name=agent_name,
                session_id=replay_session_id,
                tool_capability_ceiling=candidate_tool_capability_ceiling,
            )
        ):
            if len(candidate_events) >= bounds.max_events:
                raise _ReplayBoundsExceeded(RuntimeReplayReason.EVENT_BOUND_EXCEEDED)
            candidate_events.append(event)
    except _ReplayBoundsExceeded:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise _ReplayFailed(RuntimeReplayReason.CANDIDATE_EXECUTION_FAILED) from exc

    stored_session = await isolated.session_store.load(replay_session_id)
    candidate_records = tuple(await isolated.session_store.load_events(replay_session_id))
    if stored_session is not None:
        try:
            replay_execution_profile = execution_profile_from_session_metadata(
                stored_session.metadata
            )
        except (TypeError, ValueError) as exc:
            raise _ReplayFailed(RuntimeReplayReason.CANDIDATE_EXECUTION_FAILED) from exc
        if replay_execution_profile != candidate_profile:
            raise _ReplayFailed(RuntimeReplayReason.CANDIDATE_EXECUTION_FAILED)
    if stored_session is None or stored_session.status is not SessionStatus.COMPLETED:
        comparisons = _attempt_comparisons(
            evidence,
            candidate_records,
            require_complete=False,
        )
        first_request_mismatch = next(
            (comparison for comparison in comparisons if not comparison.matched),
            None,
        )
        if first_request_mismatch is not None:
            return _report_with_divergence(
                evidence,
                candidate_profile=candidate_profile,
                comparisons=comparisons,
                kind=RuntimeReplayDivergenceKind.REQUEST_FOOTPRINT_MISMATCH,
                boundary=RuntimeReplayBoundaryKind.MODEL_REQUEST,
                index=first_request_mismatch.index,
                source_event_id=first_request_mismatch.source_event_id,
            )
        policy_event = next(
            (
                event
                for event in candidate_records
                if event.type
                in {
                    EventType.TOOL_CALL_BLOCKED,
                    EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    EventType.TOOL_CALL_APPROVAL_DENIED,
                }
            ),
            None,
        )
        if policy_event is not None:
            policy_index, policy_outcome = _source_tool_boundary(
                evidence,
                policy_event.payload.get("tool_call_id"),
            )
            return _report_with_divergence(
                evidence,
                candidate_profile=candidate_profile,
                comparisons=comparisons,
                kind=RuntimeReplayDivergenceKind.POLICY_DECISION_MISMATCH,
                boundary=RuntimeReplayBoundaryKind.TOOL_POLICY,
                index=policy_index,
                source_event_id=policy_outcome.source_event_id,
            )
        raise _ReplayFailed(RuntimeReplayReason.CANDIDATE_EXECUTION_FAILED)

    comparisons = _attempt_comparisons(evidence, candidate_records)
    if comparisons and not comparisons[0].matched:
        first_request_mismatch = comparisons[0]
        return _report_with_divergence(
            evidence,
            candidate_profile=candidate_profile,
            comparisons=comparisons,
            kind=RuntimeReplayDivergenceKind.REQUEST_FOOTPRINT_MISMATCH,
            boundary=RuntimeReplayBoundaryKind.MODEL_REQUEST,
            index=first_request_mismatch.index,
            source_event_id=first_request_mismatch.source_event_id,
        )

    if tracker.mismatched:
        mismatch_index, mismatch_outcome = _source_tool_boundary(
            evidence,
            tracker.mismatched_call_id,
        )
        return _report_with_divergence(
            evidence,
            candidate_profile=candidate_profile,
            comparisons=comparisons,
            kind=RuntimeReplayDivergenceKind.TOOL_CALL_MISMATCH,
            boundary=RuntimeReplayBoundaryKind.TOOL_CALL,
            index=mismatch_index,
            source_event_id=mismatch_outcome.source_event_id,
        )
    if set(tracker.consumed) != set(evidence.tool_outcomes):
        missing_call_id = next(
            call_id for call_id in evidence.tool_outcomes if call_id not in tracker.consumed
        )
        missing_index, missing_outcome = _source_tool_boundary(evidence, missing_call_id)
        return _report_with_divergence(
            evidence,
            candidate_profile=candidate_profile,
            comparisons=comparisons,
            kind=RuntimeReplayDivergenceKind.POLICY_DECISION_MISMATCH,
            boundary=RuntimeReplayBoundaryKind.TOOL_POLICY,
            index=missing_index,
            source_event_id=missing_outcome.source_event_id,
        )

    candidate_transcript = tuple(await isolated.session_store.load_transcript(replay_session_id))
    tool_divergence = _candidate_tool_divergence(
        evidence,
        candidate_events=candidate_records,
        candidate_transcript=candidate_transcript,
        replay_session_id=replay_session_id,
        tracker=tracker,
    )
    if tool_divergence is not None:
        return _report_with_divergence(
            evidence,
            candidate_profile=candidate_profile,
            comparisons=comparisons,
            kind=tool_divergence.kind,
            boundary=(
                RuntimeReplayBoundaryKind.TOOL_RESULT
                if tool_divergence.kind is RuntimeReplayDivergenceKind.RECORDED_OUTCOME_MISMATCH
                else RuntimeReplayBoundaryKind.TOOL_CALL
            ),
            index=tool_divergence.index,
            source_event_id=tool_divergence.source_event_id,
        )

    later_request_mismatch = next((item for item in comparisons[1:] if not item.matched), None)
    if later_request_mismatch is not None:
        return _report_with_divergence(
            evidence,
            candidate_profile=candidate_profile,
            comparisons=comparisons,
            kind=RuntimeReplayDivergenceKind.REQUEST_FOOTPRINT_MISMATCH,
            boundary=RuntimeReplayBoundaryKind.MODEL_REQUEST,
            index=later_request_mismatch.index,
            source_event_id=later_request_mismatch.source_event_id,
        )

    candidate_final_output = final_output_text(candidate_transcript)
    candidate_trajectory = Trajectory(
        session=stored_session,
        final_output=candidate_final_output,
    )
    if _terminal_fingerprint(candidate_trajectory) != _terminal_fingerprint(evidence.trajectory):
        return _report_with_divergence(
            evidence,
            candidate_profile=candidate_profile,
            comparisons=comparisons,
            kind=RuntimeReplayDivergenceKind.TERMINAL_OUTCOME_MISMATCH,
            boundary=RuntimeReplayBoundaryKind.TERMINAL_OUTCOME,
            index=2,
            source_event_id=evidence.model_completed_events[-1].id,
        )

    profile_changes = changed_execution_profile_components(
        evidence.source_execution_profile,
        candidate_profile,
    )
    if profile_changes:
        return _report_with_divergence(
            evidence,
            candidate_profile=candidate_profile,
            comparisons=comparisons,
            kind=RuntimeReplayDivergenceKind.EXECUTION_PROFILE_MISMATCH,
            boundary=RuntimeReplayBoundaryKind.EXECUTION_PROFILE,
            index=None,
            source_event_id=None,
        )
    return RuntimeReplayReport(
        disposition=RuntimeReplayDisposition.MATCHED,
        source_session_identity=(
            _session_identity(evidence.trajectory.session.id)
            if evidence.trajectory.session
            else None
        ),
        trajectory_identity=_trajectory_identity(evidence.trajectory),
        source_execution_profile=evidence.source_execution_profile,
        candidate_execution_profile=candidate_profile,
        request_attempts=comparisons,
        compared_model_steps=len(comparisons),
        compared_tool_rounds=1,
        supporting_source_event_ids=evidence.supporting_source_event_ids,
        warnings=evidence.warnings,
    )


def _report_with_divergence(
    evidence: _ReplayEvidence,
    *,
    candidate_profile: ExecutionProfileIdentity,
    comparisons: tuple[RuntimeReplayAttemptComparison, ...],
    kind: RuntimeReplayDivergenceKind,
    boundary: RuntimeReplayBoundaryKind,
    index: int | None,
    source_event_id: str | None,
    compared_tool_rounds: Literal[0, 1] | None = None,
) -> RuntimeReplayReport:
    changes = changed_execution_profile_components(
        evidence.source_execution_profile,
        candidate_profile,
    )
    return RuntimeReplayReport(
        disposition=RuntimeReplayDisposition.DIVERGED,
        source_session_identity=(
            _session_identity(evidence.trajectory.session.id)
            if evidence.trajectory.session
            else None
        ),
        trajectory_identity=_trajectory_identity(evidence.trajectory),
        source_execution_profile=evidence.source_execution_profile,
        candidate_execution_profile=candidate_profile,
        changed_execution_profile_components=tuple(item.value for item in changes),
        request_attempts=comparisons,
        compared_model_steps=len(comparisons),
        compared_tool_rounds=(
            (1 if evidence.tool_outcomes else 0)
            if compared_tool_rounds is None
            else compared_tool_rounds
        ),
        first_divergence=RuntimeReplayDivergence(
            kind=kind,
            boundary=boundary,
            index=index,
            source_event_id=source_event_id,
        ),
        supporting_source_event_ids=evidence.supporting_source_event_ids,
        warnings=evidence.warnings,
    )


async def replay_session(app: CayuApp, request: RuntimeReplayRequest) -> RuntimeReplayReport:
    """Replay one promoted two-step tool round through an isolated ordinary runtime.

    Recorded model completions and tool results are fixtures. The operation never
    calls a live provider, registered tool, environment, runner, vault, or source
    session store. A match proves only the versioned runtime/request contracts
    represented by the retained identities.
    """

    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp.")
    if type(request) is not RuntimeReplayRequest:
        raise TypeError("request must be an exact RuntimeReplayRequest.")
    trajectory = request.trajectory
    if type(trajectory) is not Trajectory:  # pragma: no cover - request validation owns this
        raise TypeError("request.trajectory must be an exact Trajectory.")
    evidence: _ReplayEvidence | None = None
    candidate_profile: ExecutionProfileIdentity | None = None
    try:
        evidence = _extract_evidence(trajectory, bounds=request.bounds)
        source_session = evidence.trajectory.session
        if source_session is None:
            raise _ReplayUnavailable(RuntimeReplayReason.SOURCE_SESSION_MISSING)
        agent_name = request.agent_name or source_session.agent_name
        registered_agent = _require_candidate_boundary(
            app,
            agent_name,
            provider_name=source_session.provider_name,
        )
        candidate_tool_capability_ceiling = resolve_tool_capability_ceiling(
            None,
            registered_agent.tool_capabilities,
            maximum=source_session.tool_capability_ceiling,
        )
        _require_candidate_billing_evidence(app, evidence, agent_name=agent_name)
        replay_timeout = asyncio.timeout(request.bounds.timeout_seconds)
        try:
            async with replay_timeout:
                candidate_profile = await _candidate_profile(
                    app,
                    evidence,
                    agent_name=agent_name,
                    tool_capability_ceiling=candidate_tool_capability_ceiling,
                )
                if candidate_tool_capability_ceiling != source_session.tool_capability_ceiling:
                    return _report_with_divergence(
                        evidence,
                        candidate_profile=candidate_profile,
                        comparisons=(),
                        kind=RuntimeReplayDivergenceKind.EXECUTION_PROFILE_MISMATCH,
                        boundary=RuntimeReplayBoundaryKind.EXECUTION_PROFILE,
                        index=None,
                        source_event_id=None,
                        compared_tool_rounds=0,
                    )
                _require_attributable_invocation_profile(evidence, candidate_profile)
                return await _execute_replay(
                    app,
                    evidence,
                    agent_name=agent_name,
                    bounds=request.bounds,
                    candidate_profile=candidate_profile,
                    candidate_tool_capability_ceiling=candidate_tool_capability_ceiling,
                )
        except TimeoutError as exc:
            if not replay_timeout.expired():
                raise _ReplayFailed(RuntimeReplayReason.CANDIDATE_EXECUTION_FAILED) from exc
            raise _ReplayBoundsExceeded(RuntimeReplayReason.WALL_CLOCK_BOUND_EXCEEDED) from exc
    except _ReplayBoundsExceeded as exc:
        return _base_report(
            trajectory,
            disposition=RuntimeReplayDisposition.BOUNDS_EXCEEDED,
            reason=exc.reason,
            evidence=evidence,
            candidate_profile=candidate_profile,
        )
    except _ReplayUnavailable as exc:
        return _base_report(
            trajectory,
            disposition=RuntimeReplayDisposition.UNAVAILABLE,
            reason=exc.reason,
            evidence=evidence,
            candidate_profile=candidate_profile,
        )
    except _ReplayFailed as exc:
        return _base_report(
            trajectory,
            disposition=RuntimeReplayDisposition.FAILED,
            reason=exc.reason,
            evidence=evidence,
            candidate_profile=candidate_profile,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        return _base_report(
            trajectory,
            disposition=RuntimeReplayDisposition.FAILED,
            reason=RuntimeReplayReason.CANDIDATE_EXECUTION_FAILED,
            evidence=evidence,
            candidate_profile=candidate_profile,
        )
