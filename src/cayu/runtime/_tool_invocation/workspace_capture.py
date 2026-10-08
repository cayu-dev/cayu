"""Bounded workspace observation, mutation receipts and artifact capture."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from cayu._exception_groups import (
    iter_exception_tree,
)
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
)
from cayu._validation import (
    require_clean_nonblank,
)
from cayu._workspace_mutation import workspace_mutation_task_settlement_probe
from cayu.artifacts.base import ArtifactScope
from cayu.environments.bindings import BoundWorkspace, _runtime_owned_workspace_observer_name
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_nested_payload_authority,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.execution_units import ToolRoundIdentity
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._event_writer import RuntimeEventWriter, prepare_runtime_event
from cayu.runtime._tool_invocation.cancellation import (
    _contains_process_signal,
)
from cayu.runtime._tool_invocation.context import (
    _artifact_store_id,
    _workspace_id,
)
from cayu.runtime._tool_round_staging import (
    _event_with_tool_round_authority,
)
from cayu.sessions.base import (
    Session,
    SessionStore,
)
from cayu.tools._operation_boundary import (
    BoundedInvocationOperationRegistry,
    await_invocation_operation,
)
from cayu.tools._resources import (
    WorkspaceMutationSettlementError,
)
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces._revision_records import (
    _WORKSPACE_PATH_REVISION_AUTHORITY_FIELDS,
    _WORKSPACE_PATH_REVISION_DELTA_AUTHORITY_FIELDS,
)
from cayu.workspaces.mutation_attribution import (
    DirectWorkspaceMutationCollector,
    WorkspaceMutationWindow,
    classify_workspace_mutation_attribution,
    direct_workspace_mutation_payload,
    observed_pre_window_change,
    reconcile_direct_workspace_mutations,
)
from cayu.workspaces.observation_recovery import (
    WORKSPACE_OBSERVATION_TERMINAL_CONTROLS,
    WorkspaceObservationArtifact,
    WorkspaceObservationArtifactState,
    WorkspaceObservationEvidenceState,
    WorkspaceObservationLifecycle,
    WorkspaceObservationPhase,
    WorkspaceObservationTerminalStatus,
    _WorkspaceObservationAuthorityProjection,
    publish_workspace_observation_transition,
    raise_workspace_observation_concurrent_control,
    restore_workspace_observation_cancellation_requests,
    workspace_observation_artifact_metadata_matches,
    workspace_observation_terminal_from_delta_status,
)
from cayu.workspaces.revisions import (
    WorkspaceDirectMutationReconciliation,
    WorkspaceIdentity,
    WorkspaceMutationAttribution,
    WorkspaceMutationAttributionConfidence,
    WorkspaceRevisionDelta,
    WorkspaceRevisionDeltaStatus,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationLimitExceeded,
    WorkspaceRevisionObservationLimits,
    WorkspaceRevisionObservationStatus,
    WorkspaceWriterIsolationEvidence,
    compare_workspace_revisions,
    copy_bounded_workspace_revision_observation,
    unsupported_workspace_revision,
)


def _event_with_workspace_observation_authority(
    event: Event,
    identity: ToolRoundIdentity,
    interaction_id: str | None,
    *additional_fields: str,
) -> Event:
    """Attest an observation event only from its typed lifecycle owner."""

    event = _event_with_tool_round_authority(event, identity, *additional_fields)
    if (
        event.type
        in {
            EventType.WORKSPACE_REVISION_OBSERVED,
            EventType.WORKSPACE_MUTATION_RECORDED,
        }
        and "paths" in event.payload
    ):
        authority_fields = (
            _WORKSPACE_PATH_REVISION_AUTHORITY_FIELDS
            if event.type is EventType.WORKSPACE_REVISION_OBSERVED
            else _WORKSPACE_PATH_REVISION_DELTA_AUTHORITY_FIELDS
        )
        event = event_with_runtime_nested_payload_authority(
            event,
            *(("paths", "*", field_name) for field_name in sorted(authority_fields)),
        )
    if interaction_id is None:
        if event.interaction_id is not None:
            raise ValueError("Workspace observation event changed its interaction identity.")
        return event
    interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
    if event.interaction_id != interaction_id:
        raise ValueError("Workspace observation event conflicts with its interaction owner.")
    return event_with_runtime_envelope_authority(event, "interaction_id")


@dataclass(frozen=True)
class _WorkspaceEvidenceProjection:
    paths: list[dict[str, Any]]
    status: str
    detail_code: str | None
    manifest_artifact_id: str | None = None
    manifest_artifact_sha256: str | None = None
    manifest_artifact_size_bytes: int | None = None
    evidence_available: bool = True


@dataclass(frozen=True)
class _WorkspaceCaptureResult:
    lifecycle: WorkspaceObservationLifecycle
    delta_lifecycle: WorkspaceObservationLifecycle
    events: tuple[Event, ...]
    receipt_event: Event
    terminal_status: WorkspaceObservationTerminalStatus
    terminal_detail_code: str | None


_WORKSPACE_RECEIPT_INLINE_PATH_LIMIT = 32


_WORKSPACE_RECEIPT_INLINE_BYTES = 8 * 1024


_WORKSPACE_OBSERVATION_RUNTIME_LIMITS = WorkspaceRevisionObservationLimits()


_WORKSPACE_OBSERVATION_MAX_TOTAL_PATHS = (1 << 63) - 1


_WORKSPACE_OBSERVATION_TIMEOUT_SECONDS = 30.0


_WORKSPACE_ARTIFACT_WRITE_TIMEOUT_SECONDS = 30.0


_MAX_RETAINED_WORKSPACE_CAPTURE_OPERATIONS = 64


def _workspace_binding_generation_id(
    registered_environment: runtime_records.RegisteredEnvironment,
) -> str:
    value = registered_environment.binding_generation_id
    value = require_clean_nonblank(value, "binding_generation_id")
    if not value.startswith("wbind_") or len(value) != 38:
        raise ValueError("Workspace binding generation identity is malformed.")
    return value


def _workspace_revision_observer_name(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str:
    if registered_environment is None:
        return "UnconfiguredEnvironment"
    binding = registered_environment.environment.binding
    if binding is None:
        return "UnconfiguredWorkspaceBinding"
    return type(binding).__name__


def _workspace_revision_observer_is_runtime_owned(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> bool:
    if registered_environment is None:
        return True
    binding = registered_environment.environment.binding
    if binding is None:
        return True
    return _runtime_owned_workspace_observer_name(binding) is not None


def _workspace_writer_isolation(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> WorkspaceWriterIsolationEvidence:
    """Copy adapter isolation evidence without treating malformed claims as proof."""

    if registered_environment is None:
        return WorkspaceWriterIsolationEvidence()
    binding = registered_environment.environment.binding
    if binding is None:
        return WorkspaceWriterIsolationEvidence()
    bound = registered_environment.bound_workspace
    if bound is None:
        workspace = registered_environment.environment.workspace
        bound = BoundWorkspace(
            workspace=workspace,
            source_workspace=workspace,
            runner=registered_environment.environment.runner,
        )
    try:
        evidence = binding.observe_writer_isolation(bound)
        if type(evidence) is not WorkspaceWriterIsolationEvidence:
            raise TypeError("Workspace binding returned invalid writer-isolation evidence.")
        return WorkspaceWriterIsolationEvidence.model_validate(evidence.model_dump(mode="python"))
    except Exception:
        return WorkspaceWriterIsolationEvidence(detail_code="writer_isolation_observer_failed")


async def _observe_workspace_revision(
    registered_environment: runtime_records.RegisteredEnvironment | None,
    *,
    operation_registry: BoundedInvocationOperationRegistry,
    require_quiescence_before_return: bool = False,
    authority_projection: _WorkspaceObservationAuthorityProjection | None = None,
) -> WorkspaceRevisionObservation:
    if type(require_quiescence_before_return) is not bool:
        raise TypeError("require_quiescence_before_return must be a boolean.")
    workspace_id = _workspace_id(registered_environment) or "workspace-unavailable"
    observer = _workspace_revision_observer_name(registered_environment)
    expected_identity = WorkspaceIdentity(workspace_id=workspace_id, observer=observer)
    durable_identity = (
        expected_identity
        if authority_projection is None
        else WorkspaceIdentity(
            workspace_id=authority_projection.workspace_id,
            observer=authority_projection.observer,
        )
    )
    if authority_projection is not None and (
        authority_projection.configured_identity
        != (expected_identity.workspace_id, expected_identity.observer)
    ):
        raise RuntimeError("Workspace observation authority changed before capture.")
    if registered_environment is None:
        return unsupported_workspace_revision(
            workspace_id=durable_identity.workspace_id,
            observer=durable_identity.observer,
        )
    binding = registered_environment.environment.binding
    bound = registered_environment.bound_workspace
    if binding is None:
        return unsupported_workspace_revision(
            workspace_id=durable_identity.workspace_id,
            observer=durable_identity.observer,
        )
    if bound is None:
        workspace = registered_environment.environment.workspace
        bound = BoundWorkspace(
            workspace=workspace,
            source_workspace=workspace,
            runner=registered_environment.environment.runner,
        )
    if not operation_registry.reserve():
        return WorkspaceRevisionObservation(
            identity=durable_identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            detail_code="revision_observer_capacity_exhausted",
        )

    try:
        observation_task = asyncio.create_task(
            capture_awaitable_outcome(lambda: binding.observe_revision(bound))
        )
    except BaseException:
        operation_registry.release_reservation()
        raise
    operation_registry.track(observation_task)
    try:
        outcome = await await_shielded_task_outcome(
            observation_task,
            timeout_s=_WORKSPACE_OBSERVATION_TIMEOUT_SECONDS,
            timeout_after_cancellation_s=0.0,
        )
    except BaseException:
        if not observation_task.done():
            registered_environment.workspace_mutation_fence.fail_closed(
                workspace_mutation_task_settlement_probe(observation_task)
            )
        raise
    if observation_task.done():
        operation_registry.release(observation_task)
    observed: object = None
    observer_error = outcome.error
    if observer_error is None:
        captured = outcome.result
        if type(captured) is not CapturedAwaitableOutcome:
            observer_error = RuntimeError(
                "Workspace revision observer returned an invalid owned outcome."
            )
        else:
            observed = captured.result
            observer_error = captured.error
    observer_settlement_unproven = observer_error is not None and any(
        isinstance(candidate, asyncio.CancelledError)
        for candidate in iter_exception_tree(observer_error)
    )
    if observer_settlement_unproven:
        # A child-originated cancellation proves only that the observer
        # coroutine stopped.  It does not prove that thread-, executor-, SDK-,
        # subprocess-, or remote-backed work dispatched by that observer also
        # stopped.  Retain a fail-closed environment owner even when an
        # independent caller cancellation arrived in the same scheduling turn.
        registered_environment.workspace_mutation_fence.fail_closed(
            workspace_mutation_task_settlement_probe(observation_task)
        )
    if outcome.cancellation is not None and not observation_task.done():
        registered_environment.workspace_mutation_fence.fail_closed(
            workspace_mutation_task_settlement_probe(observation_task)
        )
    if outcome.cancellation is not None:
        restore_workspace_observation_cancellation_requests(outcome.cancellation_requests_consumed)
    raise_workspace_observation_concurrent_control(
        cancellation=outcome.cancellation,
        error=observer_error,
        operation="Workspace revision observation",
        cancellation_requests_pending=outcome.cancellation_requests_consumed,
    )
    if outcome.timed_out:
        if not observation_task.done():
            registered_environment.workspace_mutation_fence.fail_closed(
                workspace_mutation_task_settlement_probe(observation_task)
            )
            # The tool must not begin mutating while a cancellation-opaque
            # before-observer may still be reading the same workspace.  The
            # retained fence permits a later invocation only after the exact
            # observer task settles.
            if require_quiescence_before_return:
                raise WorkspaceMutationSettlementError(
                    "Workspace revision observation did not settle after its deadline."
                ) from None
        return WorkspaceRevisionObservation(
            identity=durable_identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            detail_code="revision_observer_timeout",
        )
    if observer_error is not None and any(
        isinstance(candidate, (GeneratorExit, KeyboardInterrupt, SystemExit))
        for candidate in iter_exception_tree(observer_error)
    ):
        if observer_error is None:  # pragma: no cover - narrowed by the helper
            raise AssertionError("Process-signal classification lost its error.")
        raise observer_error
    if observer_settlement_unproven and require_quiescence_before_return:
        raise WorkspaceMutationSettlementError(
            "Workspace revision observation did not prove mutation quiescence."
        ) from None
    if observer_error is not None:
        return WorkspaceRevisionObservation(
            identity=durable_identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            detail_code="revision_observer_failed",
        )
    try:
        validated = _bounded_workspace_observation_copy(
            observed,
            expected_identity=expected_identity,
        )
    except WorkspaceRevisionObservationLimitExceeded:
        return WorkspaceRevisionObservation(
            identity=durable_identity,
            status=WorkspaceRevisionObservationStatus.TRUNCATED,
            detail_code="revision_observer_limit_exceeded",
        )
    except Exception:
        return WorkspaceRevisionObservation(
            identity=durable_identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            detail_code="revision_observer_failed",
        )
    return validated.model_copy(update={"identity": durable_identity})


def _bounded_workspace_observation_copy(
    observed: object,
    *,
    expected_identity: WorkspaceIdentity,
) -> WorkspaceRevisionObservation:
    """Detach public observer output only after enforcing runtime hard limits."""
    return copy_bounded_workspace_revision_observation(
        observed,
        expected_identity=expected_identity,
        limits=_WORKSPACE_OBSERVATION_RUNTIME_LIMITS,
        max_total_paths=_WORKSPACE_OBSERVATION_MAX_TOTAL_PATHS,
    )


async def _record_workspace_mutation_after(
    *,
    session_store: SessionStore,
    event_writer: RuntimeEventWriter,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    artifact_store: Any,
    artifact_unavailable_detail_code: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    tool_call: runtime_records.ToolCallRequest,
    tool_round_identity: ToolRoundIdentity,
    execution_profile: ExecutionProfileIdentity | None,
    model_step: int | None,
    window_id: str,
    lifecycle: WorkspaceObservationLifecycle,
    before_observation: WorkspaceRevisionObservation,
    authority_projection: _WorkspaceObservationAuthorityProjection,
    attribution_window: WorkspaceMutationWindow,
    writer_isolation_before: WorkspaceWriterIsolationEvidence,
    direct_mutations: DirectWorkspaceMutationCollector,
    redactor: SecretRedactor,
    evidence_available: bool,
    operation_registry: BoundedInvocationOperationRegistry,
) -> _WorkspaceCaptureResult:
    """Persist reconstructible evidence up to, but not including, terminal closure."""

    if registered_environment is None:
        raise RuntimeError("Workspace capture requires its registered environment owner.")
    if lifecycle.phase is not WorkspaceObservationPhase.TOOL_OUTCOME_STAGED:
        raise RuntimeError("Workspace capture requires a durably staged tool outcome.")
    if lifecycle.window_id != window_id or lifecycle.session_id != session.id:
        raise RuntimeError("Workspace capture lifecycle conflicts with its invocation.")
    if authority_projection.configured_artifact_store_id != _artifact_store_id(
        registered_environment
    ):
        raise RuntimeError("Workspace artifact-store authority changed before capture.")
    if (
        before_observation.identity.workspace_id != lifecycle.workspace_id
        or before_observation.identity.observer != lifecycle.observer
    ):
        raise RuntimeError("Workspace capture before evidence conflicts with its authority.")

    active_lifecycle = lifecycle

    async def record_artifact_state(artifact: WorkspaceObservationArtifact) -> None:
        nonlocal active_lifecycle
        existing_by_kind = {item.evidence_kind: item for item in active_lifecycle.artifacts}
        existing = existing_by_kind.get(artifact.evidence_kind)
        if existing is not None and (
            existing.artifact_id != artifact.artifact_id
            or existing.sha256 != artifact.sha256
            or existing.size_bytes != artifact.size_bytes
        ):
            raise RuntimeError("Workspace artifact identity changed across publication.")
        existing_by_kind[artifact.evidence_kind] = artifact
        updated = WorkspaceObservationLifecycle.model_validate(
            {
                **active_lifecycle.model_dump(mode="json"),
                "artifacts": [
                    existing_by_kind[key].model_dump(mode="json")
                    for key in sorted(existing_by_kind)
                ],
            }
        )
        await publish_workspace_observation_transition(
            session_store=session_store,
            event_writer=event_writer,
            session=session,
            previous=active_lifecycle,
            current=updated,
            phase=f"artifact-{artifact.evidence_kind}-{artifact.state.value}",
        )
        active_lifecycle = updated

    def lifecycle_with_referenced_artifact(
        value: WorkspaceObservationLifecycle,
        projection: _WorkspaceEvidenceProjection,
    ) -> WorkspaceObservationLifecycle:
        if projection.manifest_artifact_id is None:
            return value
        artifacts = []
        matched = False
        for artifact in value.artifacts:
            if artifact.artifact_id != projection.manifest_artifact_id:
                artifacts.append(artifact)
                continue
            matched = True
            if artifact.state is not WorkspaceObservationArtifactState.PUBLISHED:
                raise RuntimeError("Workspace event references an unpublished artifact.")
            artifacts.append(
                artifact.model_copy(update={"state": WorkspaceObservationArtifactState.REFERENCED})
            )
        if not matched:
            raise RuntimeError("Workspace event references an unknown artifact.")
        return WorkspaceObservationLifecycle.model_validate(
            {
                **value.model_dump(mode="json"),
                "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
            }
        )

    before_projection = await _workspace_evidence_projection(
        paths=[path.model_dump(mode="json") for path in before_observation.paths],
        status=before_observation.status.value,
        detail_code=before_observation.detail_code,
        artifact_store=artifact_store,
        artifact_unavailable_detail_code=artifact_unavailable_detail_code,
        session=session,
        registered_agent=registered_agent,
        environment_name=environment_name,
        window_id=window_id,
        evidence_kind="revision-before",
        redactor=redactor,
        evidence_available=evidence_available,
        operation_registry=operation_registry,
        artifact_state_recorder=record_artifact_state,
        registered_environment=registered_environment,
    )
    before_event = prepare_runtime_event(
        _workspace_revision_observed_event(
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            tool_call=tool_call,
            tool_round_identity=tool_round_identity,
            execution_profile=execution_profile,
            model_step=model_step,
            window_id=window_id,
            binding_generation_id=lifecycle.binding_generation_id,
            artifact_store_id=lifecycle.artifact_store_id,
            observer_is_runtime_owned=(lifecycle.observer_authority == "runtime_builtin"),
            interaction_id=lifecycle.interaction_id,
            phase="before",
            observation=before_observation,
            projection=before_projection,
        ),
        redactor=redactor,
    )
    before_state = (
        WorkspaceObservationEvidenceState.QUARANTINED
        if not before_projection.evidence_available
        else (
            WorkspaceObservationEvidenceState.FAILED
            if before_observation.status is WorkspaceRevisionObservationStatus.FAILED
            or before_projection.status != before_observation.status.value
            else WorkspaceObservationEvidenceState.PUBLISHED
        )
    )
    before_published = WorkspaceObservationLifecycle.model_validate(
        {
            **lifecycle_with_referenced_artifact(
                active_lifecycle,
                before_projection,
            ).model_dump(mode="json"),
            "before_state": before_state.value,
            "before_observation_id": before_event.id,
        }
    )
    (before_event,) = await publish_workspace_observation_transition(
        session_store=session_store,
        event_writer=event_writer,
        session=session,
        previous=active_lifecycle,
        current=before_published,
        phase="before-evidence",
        events=(before_event,),
    )
    active_lifecycle = before_published
    try:
        after_observation = await _observe_workspace_revision(
            registered_environment,
            operation_registry=operation_registry,
            authority_projection=authority_projection,
        )
        writer_isolation_after = _workspace_writer_isolation(registered_environment)
    except BaseException:
        attribution_window.close(discard_history=not evidence_available)
        raise
    if (
        after_observation.identity.workspace_id != lifecycle.workspace_id
        or after_observation.identity.observer != lifecycle.observer
    ):
        attribution_window.close(discard_history=not evidence_available)
        raise RuntimeError("Workspace capture after evidence conflicts with its authority.")
    try:
        try:
            delta = compare_workspace_revisions(before_observation, after_observation)
        except Exception:
            delta = WorkspaceRevisionDelta(
                identity=before_observation.identity,
                status=WorkspaceRevisionDeltaStatus.FAILED,
                before_revision=before_observation.revision,
                after_revision=after_observation.revision,
                detail_code="revision_comparison_failed",
            )
        direct_reconciliation = (
            reconcile_direct_workspace_mutations(
                before=before_observation,
                after=after_observation,
                collector=direct_mutations,
            )
            if evidence_available
            else WorkspaceDirectMutationReconciliation.NOT_OBSERVED
        )
        attribution = classify_workspace_mutation_attribution(
            window=attribution_window,
            isolation_before=writer_isolation_before,
            isolation_after=writer_isolation_after,
            direct_reconciliation=direct_reconciliation,
        )
        if not evidence_available:
            attribution = WorkspaceMutationAttribution(
                confidence=(
                    WorkspaceMutationAttributionConfidence.CONCURRENT_AMBIGUITY
                    if attribution_window.overlap_detected
                    else WorkspaceMutationAttributionConfidence.EXTERNAL_OR_UNKNOWN
                ),
                writer_isolation=attribution.writer_isolation,
                overlap_detected=attribution_window.overlap_detected,
                direct_reconciliation=WorkspaceDirectMutationReconciliation.NOT_OBSERVED,
                detail_code=(
                    "overlapping_workspace_mutation_windows"
                    if attribution_window.overlap_detected
                    else "workspace_evidence_quarantined"
                ),
            )
        pre_window_change = (
            observed_pre_window_change(
                attribution_window,
                before_observation,
            )
            if evidence_available
            else None
        )
    finally:
        attribution_window.close(
            after_observation if evidence_available else None,
            discard_history=not evidence_available,
        )
    after_projection = await _workspace_evidence_projection(
        paths=[path.model_dump(mode="json") for path in after_observation.paths],
        status=after_observation.status.value,
        detail_code=after_observation.detail_code,
        artifact_store=artifact_store,
        artifact_unavailable_detail_code=artifact_unavailable_detail_code,
        session=session,
        registered_agent=registered_agent,
        environment_name=environment_name,
        window_id=window_id,
        evidence_kind="revision-after",
        redactor=redactor,
        evidence_available=evidence_available,
        operation_registry=operation_registry,
        artifact_state_recorder=record_artifact_state,
        registered_environment=registered_environment,
    )
    after_event = prepare_runtime_event(
        _workspace_revision_observed_event(
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            tool_call=tool_call,
            tool_round_identity=tool_round_identity,
            execution_profile=execution_profile,
            model_step=model_step,
            window_id=window_id,
            binding_generation_id=lifecycle.binding_generation_id,
            artifact_store_id=lifecycle.artifact_store_id,
            observer_is_runtime_owned=(lifecycle.observer_authority == "runtime_builtin"),
            interaction_id=lifecycle.interaction_id,
            phase="after",
            observation=after_observation,
            projection=after_projection,
        ),
        redactor=redactor,
    )
    after_state = (
        WorkspaceObservationEvidenceState.QUARANTINED
        if not after_projection.evidence_available
        else (
            WorkspaceObservationEvidenceState.FAILED
            if after_observation.status is WorkspaceRevisionObservationStatus.FAILED
            or after_projection.status != after_observation.status.value
            else WorkspaceObservationEvidenceState.PUBLISHED
        )
    )
    after_captured = WorkspaceObservationLifecycle.model_validate(
        {
            **lifecycle_with_referenced_artifact(
                active_lifecycle,
                after_projection,
            ).model_dump(mode="json"),
            "phase": WorkspaceObservationPhase.AFTER_CAPTURED.value,
            "after_state": after_state.value,
            "after_observation_id": after_event.id,
        }
    )
    (after_event,) = await publish_workspace_observation_transition(
        session_store=session_store,
        event_writer=event_writer,
        session=session,
        previous=active_lifecycle,
        current=after_captured,
        phase="after-capture",
        events=(after_event,),
    )
    active_lifecycle = after_captured
    delta_projection = await _workspace_evidence_projection(
        paths=[path.model_dump(mode="json") for path in delta.paths],
        status=delta.status.value,
        detail_code=delta.detail_code,
        artifact_store=artifact_store,
        artifact_unavailable_detail_code=artifact_unavailable_detail_code,
        session=session,
        registered_agent=registered_agent,
        environment_name=environment_name,
        window_id=window_id,
        evidence_kind="revision-delta",
        redactor=redactor,
        evidence_available=evidence_available,
        operation_registry=operation_registry,
        artifact_state_recorder=record_artifact_state,
        registered_environment=registered_environment,
    )
    receipt_event = prepare_runtime_event(
        _workspace_mutation_recorded_event(
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            tool_call=tool_call,
            tool_round_identity=tool_round_identity,
            execution_profile=execution_profile,
            model_step=model_step,
            window_id=window_id,
            binding_generation_id=lifecycle.binding_generation_id,
            artifact_store_id=lifecycle.artifact_store_id,
            observer_is_runtime_owned=(lifecycle.observer_authority == "runtime_builtin"),
            interaction_id=lifecycle.interaction_id,
            tool_outcome_event_id=(lifecycle.tool_outcome_event_id or ""),
            tool_outcome_event_digest=(lifecycle.tool_outcome_event_digest or ""),
            before_observation_id=before_event.id,
            after_observation_id=after_event.id,
            delta=delta,
            projection=delta_projection,
            attribution=attribution,
            writer_isolation_before=writer_isolation_before,
            writer_isolation_after=writer_isolation_after,
            direct_mutations=direct_workspace_mutation_payload(
                direct_mutations,
                window_id=window_id,
                evidence_available=evidence_available,
            ),
            pre_window_change=pre_window_change,
        ),
        redactor=redactor,
    )
    terminal_status, terminal_detail_code = workspace_observation_terminal_from_delta_status(
        delta_projection.status,
        detail_code=delta_projection.detail_code,
    )
    projection_changed_evidence = (
        before_projection.status != before_observation.status.value
        or after_projection.status != after_observation.status.value
        or delta_projection.status != delta.status.value
        or not before_projection.evidence_available
        or not after_projection.evidence_available
        or not delta_projection.evidence_available
    )
    if projection_changed_evidence:
        terminal_status = WorkspaceObservationTerminalStatus.INCOMPLETE
        terminal_detail_code = "workspace_revision_evidence_incomplete"
    return _WorkspaceCaptureResult(
        lifecycle=active_lifecycle,
        delta_lifecycle=lifecycle_with_referenced_artifact(
            active_lifecycle,
            delta_projection,
        ),
        events=(before_event, after_event),
        receipt_event=receipt_event,
        terminal_status=terminal_status,
        terminal_detail_code=terminal_detail_code,
    )


def _workspace_mutation_window_id(
    *,
    session_id: str,
    session_run_epoch: int,
    tool_round_id: str,
    tool_call_id: str,
) -> str:
    material = json.dumps(
        [session_id, session_run_epoch, tool_round_id, tool_call_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "wmut_" + hashlib.sha256(material).hexdigest()


def _workspace_observation_event_id(*, window_id: str, phase: str) -> str:
    material = json.dumps(
        [window_id, phase],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "wevt_" + hashlib.sha256(material).hexdigest()


def _workspace_revision_observed_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    tool_call: runtime_records.ToolCallRequest,
    tool_round_identity: ToolRoundIdentity,
    execution_profile: ExecutionProfileIdentity | None,
    model_step: int | None,
    window_id: str,
    binding_generation_id: str,
    artifact_store_id: str | None,
    observer_is_runtime_owned: bool,
    interaction_id: str | None,
    phase: Literal["before", "after"],
    observation: WorkspaceRevisionObservation,
    projection: _WorkspaceEvidenceProjection,
) -> Event:
    payload: dict[str, Any] = {
        "phase": phase,
        "window_id": window_id,
        "session_run_epoch": session.run_epoch,
        "binding_generation_id": binding_generation_id,
        "artifact_store_id": artifact_store_id,
        "tool_call_id": tool_call.id,
        "workspace_id": observation.identity.workspace_id,
        "observer": observation.identity.observer,
        "status": projection.status,
        "revision": observation.revision if projection.evidence_available else None,
        "head_revision": (observation.head_revision if projection.evidence_available else None),
        "branch": observation.branch if projection.evidence_available else None,
        "path_scope": observation.path_scope,
        "paths": projection.paths,
        "total_paths": observation.total_paths if projection.evidence_available else 0,
        "detail_code": projection.detail_code,
        **tool_round_identity.payload(),
    }
    if model_step is not None:
        payload["model_step"] = model_step
    _add_workspace_manifest_artifact_fields(payload, projection)
    event = event_with_runtime_generated_id(
        Event(
            id=_workspace_observation_event_id(window_id=window_id, phase=phase),
            type=EventType.WORKSPACE_REVISION_OBSERVED,
            session_id=session.id,
            interaction_id=interaction_id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            tool_name=tool_call.name,
            payload=payload,
        )
    )
    return _event_with_workspace_observation_authority(
        event_with_execution_profile_authority(event, execution_profile),
        tool_round_identity,
        interaction_id,
        "tool_call_id",
        "window_id",
        "binding_generation_id",
        "workspace_id",
        *(("observer",) if observer_is_runtime_owned else ()),
        *(() if artifact_store_id is None else ("artifact_store_id",)),
        "manifest_artifact_id",
        "manifest_artifact_sha256",
    )


def _workspace_mutation_recorded_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    tool_call: runtime_records.ToolCallRequest,
    tool_round_identity: ToolRoundIdentity,
    execution_profile: ExecutionProfileIdentity | None,
    model_step: int | None,
    window_id: str,
    binding_generation_id: str,
    artifact_store_id: str | None,
    observer_is_runtime_owned: bool,
    interaction_id: str | None,
    tool_outcome_event_id: str,
    tool_outcome_event_digest: str,
    before_observation_id: str,
    after_observation_id: str,
    delta: WorkspaceRevisionDelta,
    projection: _WorkspaceEvidenceProjection,
    attribution: WorkspaceMutationAttribution,
    writer_isolation_before: WorkspaceWriterIsolationEvidence,
    writer_isolation_after: WorkspaceWriterIsolationEvidence,
    direct_mutations: dict[str, Any],
    pre_window_change: WorkspaceRevisionDelta | None,
) -> Event:
    payload: dict[str, Any] = {
        "window_id": window_id,
        "before_observation_id": before_observation_id,
        "after_observation_id": after_observation_id,
        "session_run_epoch": session.run_epoch,
        "binding_generation_id": binding_generation_id,
        "artifact_store_id": artifact_store_id,
        "tool_call_id": tool_call.id,
        "tool_outcome_event_id": tool_outcome_event_id,
        "tool_outcome_event_digest": tool_outcome_event_digest,
        "workspace_id": delta.identity.workspace_id,
        "observer": delta.identity.observer,
        "status": projection.status,
        "before_revision": delta.before_revision if projection.evidence_available else None,
        "after_revision": delta.after_revision if projection.evidence_available else None,
        "paths": projection.paths,
        "total_paths": delta.total_paths if projection.evidence_available else 0,
        "head_changed": delta.head_changed if projection.evidence_available else False,
        "branch_changed": delta.branch_changed if projection.evidence_available else False,
        "detail_code": projection.detail_code,
        "attribution": attribution.model_dump(mode="json"),
        "writer_isolation": {
            "before": writer_isolation_before.model_dump(mode="json"),
            "after": writer_isolation_after.model_dump(mode="json"),
        },
        "direct_mutations": direct_mutations,
        **tool_round_identity.payload(),
    }
    if pre_window_change is not None:
        payload["pre_window_change"] = _pre_window_change_payload(
            pre_window_change,
            window_id=window_id,
        )
    if model_step is not None:
        payload["model_step"] = model_step
    _add_workspace_manifest_artifact_fields(payload, projection)
    event = event_with_runtime_generated_id(
        Event(
            id=_workspace_observation_event_id(window_id=window_id, phase="delta"),
            type=EventType.WORKSPACE_MUTATION_RECORDED,
            session_id=session.id,
            interaction_id=interaction_id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            tool_name=tool_call.name,
            payload=payload,
        )
    )
    event = _event_with_workspace_observation_authority(
        event_with_execution_profile_authority(event, execution_profile),
        tool_round_identity,
        interaction_id,
        "tool_call_id",
        "window_id",
        "binding_generation_id",
        "tool_outcome_event_id",
        "tool_outcome_event_digest",
        "workspace_id",
        *(("observer",) if observer_is_runtime_owned else ()),
        *(() if artifact_store_id is None else ("artifact_store_id",)),
        "before_observation_id",
        "after_observation_id",
        "manifest_artifact_id",
        "manifest_artifact_sha256",
    )
    nested_authority_paths: list[tuple[str, ...]] = [
        ("attribution", "confidence"),
        ("attribution", "detail_code"),
        ("attribution", "direct_reconciliation"),
        ("attribution", "writer_isolation"),
        ("writer_isolation", "before", "status"),
        ("writer_isolation", "after", "status"),
        ("direct_mutations", "operations", "*", "method"),
        ("direct_mutations", "operations", "*", "path_sha256"),
        ("direct_mutations", "operations", "*", "result_operation"),
        ("direct_mutations", "operations", "*", "result_evidence_sha256"),
    ]
    if pre_window_change is not None:
        nested_authority_paths.extend(
            (
                ("pre_window_change", "attribution_confidence"),
                ("pre_window_change", "status"),
                ("pre_window_change", "paths", "*", "change"),
            )
        )
    return event_with_runtime_nested_payload_authority(
        event,
        *nested_authority_paths,
    )


def _pre_window_change_payload(
    delta: WorkspaceRevisionDelta,
    *,
    window_id: str,
) -> dict[str, Any]:
    """Project a bounded gap delta without exposing an unredacted path."""

    retained_paths = delta.paths[:_MAX_RETAINED_WORKSPACE_CAPTURE_OPERATIONS]
    return {
        "attribution_confidence": "external_or_unknown",
        "status": delta.status.value,
        "before_revision": delta.before_revision,
        "after_revision": delta.after_revision,
        "paths": [
            {
                "path_sha256": hashlib.sha256(f"{window_id}\0{path.path}".encode()).hexdigest(),
                "change": path.change,
            }
            for path in retained_paths
        ],
        "retained_paths": len(retained_paths),
        "total_paths": delta.total_paths,
        "truncated": len(retained_paths) < delta.total_paths,
        "head_changed": delta.head_changed,
        "branch_changed": delta.branch_changed,
        "detail_code": delta.detail_code,
    }


def _workspace_mutation_incomplete_event(
    *,
    lifecycle: WorkspaceObservationLifecycle,
    session: Session,
    execution_profile: ExecutionProfileIdentity | None,
    status: WorkspaceObservationTerminalStatus,
    detail_code: str | None,
) -> Event:
    normalized_detail_code = (
        None if detail_code is None else require_clean_nonblank(detail_code, "detail_code")
    )
    if (status.value, normalized_detail_code) not in WORKSPACE_OBSERVATION_TERMINAL_CONTROLS:
        raise ValueError("Workspace observation terminal classification is invalid.")
    identity = ToolRoundIdentity(
        model_step_id=lifecycle.model_step_id,
        model_attempt_id=lifecycle.model_attempt_id,
        tool_round_id=lifecycle.tool_round_id,
    )
    artifact_payload: dict[str, Any] = {}
    artifact_authority_fields: list[str] = []
    for artifact in lifecycle.artifacts:
        prefix = artifact.evidence_kind.replace("-", "_")
        id_field = f"{prefix}_artifact_id"
        artifact_payload[id_field] = artifact.artifact_id
        artifact_payload[f"{prefix}_artifact_sha256"] = artifact.sha256
        artifact_payload[f"{prefix}_artifact_size_bytes"] = artifact.size_bytes
        artifact_payload[f"{prefix}_artifact_state"] = artifact.state.value
        artifact_authority_fields.extend(
            (
                id_field,
                f"{prefix}_artifact_sha256",
            )
        )
    payload: dict[str, Any] = {
        "window_id": lifecycle.window_id,
        "session_run_epoch": lifecycle.source_run_epoch,
        "recovery_run_epoch": session.run_epoch,
        "binding_generation_id": lifecycle.binding_generation_id,
        "workspace_id": lifecycle.workspace_id,
        "observer": lifecycle.observer,
        "artifact_store_id": lifecycle.artifact_store_id,
        "tool_call_id": lifecycle.tool_call_id,
        "status": status.value,
        "paths": [],
        "total_paths": 0,
        "head_changed": False,
        "branch_changed": False,
        "detail_code": normalized_detail_code,
        "referenced_artifact_count": sum(
            artifact.state is WorkspaceObservationArtifactState.REFERENCED
            for artifact in lifecycle.artifacts
        ),
        "failed_artifact_count": sum(
            artifact.state is WorkspaceObservationArtifactState.FAILED
            for artifact in lifecycle.artifacts
        ),
        **artifact_payload,
        **identity.payload(),
    }
    if lifecycle.mutation_event_id is None:
        payload["attribution"] = {
            "confidence": (
                "concurrent_ambiguity"
                if status is WorkspaceObservationTerminalStatus.AMBIGUOUS
                else "external_or_unknown"
            ),
            "writer_isolation": "unknown",
            "overlap_detected": False,
            "direct_reconciliation": "not_observed",
            "detail_code": "workspace_attribution_recovery_incomplete",
        }
    if lifecycle.model_step is not None:
        payload["model_step"] = lifecycle.model_step
    if lifecycle.tool_outcome_event_id is not None:
        payload["tool_outcome_event_id"] = lifecycle.tool_outcome_event_id
    if lifecycle.tool_outcome_event_digest is not None:
        payload["tool_outcome_event_digest"] = lifecycle.tool_outcome_event_digest
    if lifecycle.before_observation_id is not None:
        payload["before_observation_id"] = lifecycle.before_observation_id
    if lifecycle.after_observation_id is not None:
        payload["after_observation_id"] = lifecycle.after_observation_id
    if lifecycle.mutation_event_id is not None:
        payload["mutation_event_id"] = lifecycle.mutation_event_id
    if lifecycle.mutation_event_digest is not None:
        payload["mutation_event_digest"] = lifecycle.mutation_event_digest
    event = event_with_runtime_generated_id(
        Event(
            id=_workspace_observation_event_id(
                window_id=lifecycle.window_id,
                phase="terminal",
            ),
            type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
            session_id=lifecycle.session_id,
            interaction_id=lifecycle.interaction_id,
            agent_name=lifecycle.agent_name,
            environment_name=lifecycle.environment_name,
            tool_name=lifecycle.tool_name,
            payload=payload,
        )
    )
    return _event_with_workspace_observation_authority(
        event_with_execution_profile_authority(event, execution_profile),
        identity,
        lifecycle.interaction_id,
        "tool_call_id",
        "window_id",
        "binding_generation_id",
        "workspace_id",
        *(("observer",) if lifecycle.observer_authority == "runtime_builtin" else ()),
        *(() if lifecycle.artifact_store_id is None else ("artifact_store_id",)),
        "tool_outcome_event_id",
        "tool_outcome_event_digest",
        "before_observation_id",
        "after_observation_id",
        "mutation_event_id",
        "mutation_event_digest",
        *artifact_authority_fields,
    )


def _workspace_observation_terminal_view(
    lifecycle: WorkspaceObservationLifecycle,
) -> WorkspaceObservationLifecycle:
    """Classify successfully written but unreferenced terminal artifacts."""

    artifacts = tuple(
        artifact.model_copy(update={"state": WorkspaceObservationArtifactState.ORPHANED})
        if artifact.state is WorkspaceObservationArtifactState.PUBLISHED
        else artifact
        for artifact in lifecycle.artifacts
    )
    if artifacts == lifecycle.artifacts:
        return lifecycle
    return WorkspaceObservationLifecycle.model_validate(
        {
            **lifecycle.model_dump(mode="json"),
            "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
        }
    )


def _add_workspace_manifest_artifact_fields(
    payload: dict[str, Any],
    projection: _WorkspaceEvidenceProjection,
) -> None:
    if projection.manifest_artifact_id is None:
        return
    payload.update(
        {
            "manifest_artifact_id": projection.manifest_artifact_id,
            "manifest_artifact_sha256": projection.manifest_artifact_sha256,
            "manifest_artifact_size_bytes": projection.manifest_artifact_size_bytes,
        }
    )


async def _workspace_evidence_projection(
    *,
    paths: list[dict[str, Any]],
    status: str,
    detail_code: str | None,
    artifact_store: Any,
    artifact_unavailable_detail_code: str = "manifest_artifact_store_unavailable",
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    window_id: str,
    evidence_kind: Literal["revision-before", "revision-after", "revision-delta"],
    redactor: SecretRedactor,
    evidence_available: bool = True,
    operation_registry: BoundedInvocationOperationRegistry,
    registered_environment: runtime_records.RegisteredEnvironment,
    artifact_state_recorder: (
        Callable[[WorkspaceObservationArtifact], Awaitable[None]] | None
    ) = None,
) -> _WorkspaceEvidenceProjection:
    if not evidence_available:
        return _WorkspaceEvidenceProjection(
            paths=[],
            status="truncated",
            detail_code="workspace_evidence_quarantined",
            evidence_available=False,
        )
    redacted_paths = redactor.redact_json_values(paths)
    if type(redacted_paths) is not list or any(type(path) is not dict for path in redacted_paths):
        return _WorkspaceEvidenceProjection(
            paths=[],
            status="failed",
            detail_code="manifest_redaction_failed",
        )
    content = json.dumps(
        {
            "schema_version": 1,
            "kind": evidence_kind,
            "paths": redacted_paths,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if (
        len(redacted_paths) <= _WORKSPACE_RECEIPT_INLINE_PATH_LIMIT
        and len(content) <= _WORKSPACE_RECEIPT_INLINE_BYTES
    ):
        return _WorkspaceEvidenceProjection(
            paths=redacted_paths,
            status=status,
            detail_code=detail_code,
        )
    if artifact_store is None:
        return _WorkspaceEvidenceProjection(
            paths=[],
            status="truncated",
            detail_code=artifact_unavailable_detail_code,
        )

    digest = hashlib.sha256(content).hexdigest()
    artifact_identity = json.dumps(
        [session.id, window_id, evidence_kind, digest],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact_id = f"art_{hashlib.sha256(artifact_identity).hexdigest()[:32]}"
    filename = f"workspace-{evidence_kind}.json"
    artifact_record = WorkspaceObservationArtifact(
        evidence_kind=evidence_kind,
        artifact_id=artifact_id,
        sha256=digest,
        size_bytes=len(content),
        state=WorkspaceObservationArtifactState.INTENT,
    )
    if artifact_state_recorder is not None:
        await artifact_state_recorder(artifact_record)
    try:
        async with asyncio.timeout(_WORKSPACE_ARTIFACT_WRITE_TIMEOUT_SECONDS):
            artifact_outcome = await await_invocation_operation(
                lambda: artifact_store.put_bytes(
                    content,
                    artifact_id=artifact_id,
                    filename=filename,
                    content_type="application/json",
                    scope=ArtifactScope.SESSION,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    metadata={
                        "schema_version": 1,
                        "kind": evidence_kind,
                        "sha256": digest,
                        "window_id": window_id,
                    },
                ),
                request_child_cancellation=False,
                abandon_on_caller_cancellation=True,
                operation_registry=operation_registry,
                on_abandoned_caller_cancellation=(
                    registered_environment.workspace_mutation_fence.fail_closed
                ),
            )
            raise_workspace_observation_concurrent_control(
                cancellation=artifact_outcome.cancellation,
                error=artifact_outcome.error,
                operation="Workspace observation artifact publication",
            )
    except TimeoutError:
        # The store call was dispatched with cancellation-opaque semantics and
        # remains retained by ``operation_registry``.  Timeout proves only
        # that Cayu stopped waiting; it does not prove that publication failed.
        # Keep the durable intent so a late successful write remains an
        # explicitly reconstructible possible orphan rather than being
        # misreported as a settled failure.
        return _WorkspaceEvidenceProjection(
            paths=[],
            status="truncated",
            detail_code="manifest_artifact_write_unsettled",
        )
    if _contains_process_signal(artifact_outcome.error):
        if artifact_outcome.error is None:  # pragma: no cover - narrowed by the helper
            raise AssertionError("Process-signal classification lost its error.")
        raise artifact_outcome.error
    if artifact_outcome.error is not None:
        # Once the extension call started, an exception is not authoritative
        # evidence that the content-bound write did not commit.  Preserve the
        # durable intent so restart reconciliation can identify a successful
        # late or acknowledgement-lost write as an orphan.  Only positive
        # no-dispatch evidence may close the intent as failed.
        if not artifact_outcome.operation_started and artifact_state_recorder is not None:
            await artifact_state_recorder(
                artifact_record.model_copy(
                    update={"state": WorkspaceObservationArtifactState.FAILED}
                )
            )
        return _WorkspaceEvidenceProjection(
            paths=[],
            status="truncated",
            detail_code=(
                "manifest_artifact_write_failed"
                if not artifact_outcome.operation_started
                else "manifest_artifact_write_unsettled"
            ),
        )
    metadata = artifact_outcome.result
    if not workspace_observation_artifact_metadata_matches(
        metadata,
        artifact=artifact_record,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        window_id=window_id,
    ):
        return _WorkspaceEvidenceProjection(
            paths=[],
            status="failed",
            detail_code="manifest_artifact_reference_invalid",
        )
    if artifact_state_recorder is not None:
        await artifact_state_recorder(
            artifact_record.model_copy(
                update={"state": WorkspaceObservationArtifactState.PUBLISHED}
            )
        )
    return _WorkspaceEvidenceProjection(
        paths=[],
        status=status,
        detail_code=detail_code,
        manifest_artifact_id=artifact_id,
        manifest_artifact_sha256=digest,
        manifest_artifact_size_bytes=len(content),
    )
