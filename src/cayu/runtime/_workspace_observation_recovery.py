"""Repair durable workspace observations without redispatching effects."""

from __future__ import annotations

import asyncio
from hashlib import sha256
from typing import Any, Literal, cast

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
from cayu.artifacts.base import ArtifactReadResult, copy_artifact_read_result
from cayu.environments.bindings import _runtime_owned_workspace_observer_name
from cayu.events import (
    Event,
    EventType,
    copy_event,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
)
from cayu.execution_units import (
    ToolRoundIdentity,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._event_writer import (
    RuntimeEventWriter,
    prepare_runtime_event,
)
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
)
from cayu.runtime._recovery_claims import (
    _IncompleteRecoveryClaimLost,
)
from cayu.runtime._tool_effect_state import (
    ToolEffectStateOwner,
    validate_tool_effect_uncertainty_event,
)
from cayu.runtime._tool_invocation.workspace_capture import _workspace_mutation_incomplete_event
from cayu.runtime._tool_round_staging import (
    restore_staged_terminal_authority,
)
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions import _staged_tool_terminal_reader as staged_terminal_reader
from cayu.sessions._checkpoint_preservation import (
    _workspace_observation_authority_mutation_scope,
)
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions.base import (
    SessionStore,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import (
    Session,
)
from cayu.tools._operation_boundary import BoundedInvocationOperationRegistry
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.observation_recovery import (
    WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY,
    WorkspaceObservationArtifactState,
    WorkspaceObservationEvidenceState,
    WorkspaceObservationLifecycle,
    WorkspaceObservationPhase,
    WorkspaceObservationTerminalStatus,
    await_workspace_observation_store_mutation,
    await_workspace_observation_store_read,
    publish_workspace_observation_transition,
    raise_workspace_observation_concurrent_control,
    restore_workspace_observation_cancellation_requests,
    workspace_observation_artifact_metadata_matches,
    workspace_observation_authority_matches,
    workspace_observation_checkpoint_value,
    workspace_observation_event_digest,
    workspace_observation_observer_authority_matches,
    workspace_observation_recovery_rejected,
    workspace_observation_terminal_from_delta_status,
    workspace_observations_from_checkpoint,
)

_WORKSPACE_ARTIFACT_RECOVERY_READ_TIMEOUT_SECONDS = 30.0


class WorkspaceObservationRecovery:
    """Repair durable workspace observations without redispatching effects."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        secret_redactor: SecretRedactor,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._secret_redactor = secret_redactor
        self._workspace_artifact_recovery_operations = BoundedInvocationOperationRegistry(
            max_operations=64
        )

    @property
    def detached_recovery_work(self) -> set[asyncio.Future[Any]]:
        """Retain reads that outlive their timeout until application shutdown."""
        return self._workspace_artifact_recovery_operations.running()

    async def settle_tool_round_workspace_observations(
        self,
        *,
        session: Session,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None,
        checkpoint: dict[str, Any] | None,
        pending_round: pending_rounds.PendingToolRound,
        staged_only: bool = False,
    ) -> tuple[dict[str, Any] | None, pending_rounds.PendingToolRound, tuple[Event, ...]]:
        """Settle the exact workspace-bound stage before changing publication timing.

        Live interruption and later recovery obey the same ordering and authority
        checks. No tool or model work is dispatched by observation recovery.
        """
        if not workspace_observations_from_checkpoint(checkpoint):
            return checkpoint, pending_round, ()
        snapshot = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if snapshot is None or execution_profile is None or snapshot.profile != execution_profile:
            raise RuntimeError("Workspace recovery lost the admitted execution profile.")
        snapshot = snapshot.model_copy(update={"profile": execution_profile})
        events = await self.recover_workspace_observations(
            session=session,
            registered_environment=registered_environment,
            execution_profile_snapshot=snapshot,
            invocation_context=invocation_context,
            staged_only=staged_only,
        )
        checkpoint, recovered_pending = await pending_round_reader.load_pending_tool_round(
            self._session_store,
            session.id,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            runtime_session=session,
        )
        if recovered_pending is None or (
            pending_rounds.pending_tool_round_identity(recovered_pending)
            != pending_rounds.pending_tool_round_identity(pending_round)
        ):
            raise RuntimeError("Workspace recovery lost its pending tool round.")
        return checkpoint, recovered_pending, events

    async def recover_workspace_observations(
        self,
        *,
        session: Session,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile | None,
        invocation_context: InvocationContext | None = None,
        staged_only: bool = False,
    ) -> tuple[Event, ...]:
        """Close crash-interrupted observation state without redispatching effects."""

        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_environment is not invocation_context.registered_environment
            or execution_profile_snapshot is None
            or invocation_context.profile is not execution_profile_snapshot.profile
        ):
            raise RuntimeError("Workspace recovery substituted frozen invocation authority.")

        recovered_events: list[Event] = []
        checkpoint = await await_workspace_observation_store_read(
            lambda: self._session_store.load_checkpoint(session.id),
            operation="Workspace observation recovery checkpoint read",
        )
        observations = workspace_observations_from_checkpoint(checkpoint)
        pending_round = pending_round_reader.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            runtime_session=session,
        )
        # Revalidate the complete aggregate after the recovery claim. Otherwise
        # a valid record ordered before a foreign record could be terminalized
        # (and could trigger artifact-store reads) before the malformed
        # aggregate is rejected.
        environment_authority_available = self.validate_workspace_observation_recovery_authority(
            session=session,
            observations=observations,
            pending_round=pending_round,
            execution_profile_snapshot=execution_profile_snapshot,
            registered_environment=registered_environment,
        )
        for window_id in sorted(observations):
            durable_lifecycle = observations[window_id]
            if staged_only and (
                pending_round is None
                or not any(
                    stage.tool_call_id == durable_lifecycle.tool_call_id
                    for stage in pending_round.staged_terminals
                )
            ):
                # Live interruption only settles observations that own a staged
                # result awaiting publication. An unfinished call, including a
                # supervisory exit during capture, retains its recovery owner.
                continue
            reconstructed_stage = self._reconstruct_workspace_observation_staged_outcome(
                session=session,
                checkpoint=checkpoint,
                lifecycle=durable_lifecycle,
            )
            if reconstructed_stage is not None:
                await publish_workspace_observation_transition(
                    session_store=self._session_store,
                    event_writer=self._event_writer,
                    session=session,
                    previous=durable_lifecycle,
                    current=reconstructed_stage,
                    phase="recovered-tool-outcome",
                )
                durable_lifecycle = reconstructed_stage
                checkpoint = await await_workspace_observation_store_read(
                    lambda: self._session_store.load_checkpoint(session.id),
                    operation="Workspace observation recovery checkpoint read",
                )
            tool_outcome_evidence_valid = True
            if durable_lifecycle.tool_outcome_event_id is not None:
                tool_outcome_evidence_valid = (
                    await self._workspace_observation_tool_outcome_evidence_valid(
                        session=session,
                        checkpoint=checkpoint,
                        lifecycle=durable_lifecycle,
                    )
                )

            delta_evidence_valid = False
            delta_evidence_conflict = False
            terminal_status: WorkspaceObservationTerminalStatus | None = None
            terminal_detail: str | None = None
            if durable_lifecycle.phase is WorkspaceObservationPhase.DELTA_PUBLISHED:
                (
                    delta_evidence_valid,
                    delta_evidence_conflict,
                    terminal_status,
                    terminal_detail,
                ) = await self._workspace_observation_delta_evidence(
                    session=session,
                    lifecycle=durable_lifecycle,
                )

            if (
                environment_authority_available
                and tool_outcome_evidence_valid
                and not delta_evidence_conflict
            ):
                lifecycle = await self._reconcile_workspace_observation_artifacts(
                    durable_lifecycle,
                    registered_environment=registered_environment,
                )
            else:
                # An extension-owned ArtifactStore must not be entered until the
                # content-bound tool and delta evidence prove that this lifecycle
                # owns the requested artifacts. Retain their exact identities but
                # fail verification closed for the terminal diagnostic.
                lifecycle = self._workspace_observation_unverified_artifacts(durable_lifecycle)
            if lifecycle.phase is WorkspaceObservationPhase.DELTA_PUBLISHED:
                if terminal_status is None:
                    raise AssertionError("Published workspace delta lost its classification.")
                if not environment_authority_available and terminal_status not in {
                    WorkspaceObservationTerminalStatus.AMBIGUOUS,
                    WorkspaceObservationTerminalStatus.FAILED,
                }:
                    terminal_status = WorkspaceObservationTerminalStatus.INCOMPLETE
                    terminal_detail = "workspace_revision_evidence_incomplete"
                if terminal_status not in {
                    WorkspaceObservationTerminalStatus.AMBIGUOUS,
                    WorkspaceObservationTerminalStatus.FAILED,
                }:
                    if any(
                        artifact.state is WorkspaceObservationArtifactState.MISSING
                        for artifact in lifecycle.artifacts
                    ):
                        terminal_status = WorkspaceObservationTerminalStatus.INCOMPLETE
                        terminal_detail = "referenced_workspace_artifact_missing"
                    elif any(
                        artifact.state is WorkspaceObservationArtifactState.FAILED
                        for artifact in lifecycle.artifacts
                    ):
                        terminal_status = WorkspaceObservationTerminalStatus.INCOMPLETE
                        terminal_detail = "workspace_artifact_verification_failed"
                    elif (
                        any(
                            artifact.state
                            in {
                                WorkspaceObservationArtifactState.INTENT,
                                WorkspaceObservationArtifactState.ORPHANED,
                            }
                            for artifact in lifecycle.artifacts
                        )
                        or lifecycle.before_state is not WorkspaceObservationEvidenceState.PUBLISHED
                        or lifecycle.after_state is not WorkspaceObservationEvidenceState.PUBLISHED
                        or lifecycle.delta_state is not WorkspaceObservationEvidenceState.PUBLISHED
                    ):
                        terminal_status = WorkspaceObservationTerminalStatus.INCOMPLETE
                        terminal_detail = "workspace_revision_evidence_incomplete"
                repaired_lifecycle = await self._repair_workspace_observation_terminal_stage(
                    session=session,
                    durable_lifecycle=durable_lifecycle,
                    lifecycle=lifecycle,
                    capture_status=("recorded" if delta_evidence_valid else "failed"),
                    capture_detail_code=(None if delta_evidence_valid else terminal_detail),
                )
                if repaired_lifecycle is None:
                    terminal_status = WorkspaceObservationTerminalStatus.AMBIGUOUS
                    terminal_detail = "durable_tool_outcome_evidence_missing"
                else:
                    durable_lifecycle = repaired_lifecycle
                    lifecycle = lifecycle.model_copy(
                        update={
                            "tool_outcome_event_digest": (
                                repaired_lifecycle.tool_outcome_event_digest
                            ),
                        },
                        deep=True,
                    )
                final_event = prepare_runtime_event(
                    _workspace_mutation_incomplete_event(
                        lifecycle=lifecycle,
                        session=session,
                        execution_profile=(
                            None
                            if execution_profile_snapshot is None
                            else execution_profile_snapshot.profile
                        ),
                        status=terminal_status,
                        detail_code=terminal_detail,
                    ),
                    redactor=self._secret_redactor,
                )
                published = await publish_workspace_observation_transition(
                    session_store=self._session_store,
                    event_writer=self._event_writer,
                    session=session,
                    previous=durable_lifecycle,
                    current=None,
                    phase="terminal",
                    terminal_status=terminal_status,
                    terminal_detail_code=terminal_detail,
                    terminal_artifacts=lifecycle.artifacts,
                    events=(final_event,),
                )
                recovered_events.extend(published)
                checkpoint = await await_workspace_observation_store_read(
                    lambda: self._session_store.load_checkpoint(session.id),
                    operation="Workspace observation recovery checkpoint read",
                )
                observations = workspace_observations_from_checkpoint(checkpoint)
                continue

            terminal_status = (
                WorkspaceObservationTerminalStatus.INCOMPLETE
                if lifecycle.phase is WorkspaceObservationPhase.TOOL_OUTCOME_STAGED
                or lifecycle.phase is WorkspaceObservationPhase.AFTER_CAPTURED
                else WorkspaceObservationTerminalStatus.AMBIGUOUS
            )
            detail_code = (
                "worker_lost_before_workspace_observation_completed"
                if terminal_status is WorkspaceObservationTerminalStatus.INCOMPLETE
                else "worker_lost_before_tool_outcome_was_durable"
            )

            if lifecycle.tool_outcome_event_id is not None:
                repaired_lifecycle = await self._repair_workspace_observation_terminal_stage(
                    session=session,
                    durable_lifecycle=durable_lifecycle,
                    lifecycle=lifecycle,
                    capture_status="failed",
                    capture_detail_code=detail_code,
                )
                if repaired_lifecycle is None:
                    terminal_status = WorkspaceObservationTerminalStatus.AMBIGUOUS
                    detail_code = "durable_tool_outcome_evidence_missing"
                else:
                    durable_lifecycle = repaired_lifecycle
                    lifecycle = lifecycle.model_copy(
                        update={
                            "tool_outcome_event_digest": (
                                repaired_lifecycle.tool_outcome_event_digest
                            ),
                        },
                        deep=True,
                    )

            terminal_event = prepare_runtime_event(
                _workspace_mutation_incomplete_event(
                    lifecycle=lifecycle,
                    session=session,
                    execution_profile=(
                        None
                        if execution_profile_snapshot is None
                        else execution_profile_snapshot.profile
                    ),
                    status=terminal_status,
                    detail_code=detail_code,
                ),
                redactor=self._secret_redactor,
            )
            published = await publish_workspace_observation_transition(
                session_store=self._session_store,
                event_writer=self._event_writer,
                session=session,
                previous=durable_lifecycle,
                current=None,
                phase="terminal",
                terminal_status=terminal_status,
                terminal_detail_code=detail_code,
                terminal_artifacts=lifecycle.artifacts,
                events=(terminal_event,),
            )
            recovered_events.extend(published)
            checkpoint = await await_workspace_observation_store_read(
                lambda: self._session_store.load_checkpoint(session.id),
                operation="Workspace observation recovery checkpoint read",
            )
            observations = workspace_observations_from_checkpoint(checkpoint)
        return tuple(recovered_events)

    def validate_workspace_observation_recovery_authority(
        self,
        *,
        session: Session,
        observations: dict[str, WorkspaceObservationLifecycle],
        pending_round: pending_rounds.PendingToolRound | None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile | None,
        registered_environment: runtime_records.RegisteredEnvironment | None = None,
    ) -> bool:
        """Reject lifecycle authority conflicts before recovery side effects."""

        if not observations:
            return True
        if pending_round is None:
            raise workspace_observation_recovery_rejected(
                "Workspace observation has no authoritative pending tool round."
            )
        if execution_profile_snapshot is None:
            raise workspace_observation_recovery_rejected(
                "Workspace observation has no authoritative active invocation profile."
            )
        if (
            pending_round.source_run_epoch is None
            or pending_round.execution_profile_fingerprint is None
        ):
            raise workspace_observation_recovery_rejected(
                "Workspace observation pending tool round has incomplete execution authority."
            )
        if (
            pending_round.execution_profile_fingerprint
            != execution_profile_snapshot.profile.fingerprint
            or (
                pending_round.interaction_id is not None
                and pending_round.interaction_id != execution_profile_snapshot.interaction_id
            )
        ):
            raise workspace_observation_recovery_rejected(
                "Workspace observation conflicts with its active invocation profile."
            )
        pending_identity = pending_rounds.pending_tool_round_identity(pending_round)
        pending_calls = {call.tool_call_id: call for call in pending_round.tool_calls}
        current_workspace_id: str | None = None
        current_observer = "UnconfiguredEnvironment"
        current_observer_is_runtime_owned = True
        current_artifact_store_id: str | None = None
        factory_template_unavailable = (
            registered_environment is not None
            and registered_environment.factory_backed
            and registered_environment.factory is not None
        )
        if registered_environment is not None and not factory_template_unavailable:
            try:
                workspace = registered_environment.environment.workspace
                workspace_id = None if workspace is None else getattr(workspace, "id", None)
                current_workspace_id = (
                    None
                    if workspace_id is None
                    else require_clean_nonblank(workspace_id, "workspace.id")
                )
                binding = registered_environment.environment.binding
                current_observer = (
                    "UnconfiguredWorkspaceBinding" if binding is None else type(binding).__name__
                )
                current_observer_is_runtime_owned = (
                    binding is None or _runtime_owned_workspace_observer_name(binding) is not None
                )
                artifact_store = registered_environment.environment.artifact_store
                artifact_store_id = (
                    None if artifact_store is None else getattr(artifact_store, "id", None)
                )
                current_artifact_store_id = (
                    None
                    if artifact_store_id is None
                    else require_clean_nonblank(artifact_store_id, "artifact_store.id")
                )
            except (AttributeError, TypeError, ValueError):
                raise workspace_observation_recovery_rejected(
                    "Workspace observation current environment authority is invalid."
                ) from None
        claimed_tool_calls: set[tuple[str, str]] = set()
        for lifecycle in observations.values():
            if lifecycle.session_id != session.id:
                raise workspace_observation_recovery_rejected(
                    "Workspace observation belongs to a different session."
                )
            if (
                lifecycle.agent_name != session.agent_name
                or lifecycle.agent_name != pending_round.agent_name
                or lifecycle.environment_name != session.environment_name
                or lifecycle.environment_name != pending_round.environment_name
            ):
                raise workspace_observation_recovery_rejected(
                    "Workspace observation conflicts with its invocation scope."
                )
            if lifecycle.source_run_epoch > session.run_epoch:
                raise workspace_observation_recovery_rejected(
                    "Workspace observation belongs to a future run epoch."
                )
            # ``binding_generation_id`` identifies the historical concrete
            # in-process binding that owned the observation window. A fresh
            # worker necessarily registers a new generation, so equality with
            # the current process would make restart repair impossible. The
            # frozen lifecycle/pending-round tuple authenticates the historical
            # owner; only stable workspace, observer, and artifact-store
            # authority can be rebound positively across processes.
            if (
                registered_environment is not None
                and not factory_template_unavailable
                and (
                    not workspace_observation_authority_matches(
                        lifecycle.workspace_id,
                        current_workspace_id or "workspace-unavailable",
                        field_name="workspace_id",
                        session_id=lifecycle.session_id,
                        public_authority_alias_codec=(
                            self._session_store.public_authority_alias_codec
                        ),
                    )
                    or not workspace_observation_observer_authority_matches(
                        lifecycle.observer,
                        lifecycle.observer_authority,
                        current_observer,
                        configured_observer_is_runtime_owned=current_observer_is_runtime_owned,
                        session_id=lifecycle.session_id,
                        public_authority_alias_codec=(
                            self._session_store.public_authority_alias_codec
                        ),
                    )
                    or not workspace_observation_authority_matches(
                        lifecycle.artifact_store_id,
                        current_artifact_store_id,
                        field_name="artifact_store_id",
                        session_id=lifecycle.session_id,
                        public_authority_alias_codec=(
                            self._session_store.public_authority_alias_codec
                        ),
                    )
                )
            ):
                raise workspace_observation_recovery_rejected(
                    "Workspace observation conflicts with its registered environment authority."
                )
            if lifecycle.interaction_id != execution_profile_snapshot.interaction_id:
                raise workspace_observation_recovery_rejected(
                    "Workspace observation conflicts with its active invocation profile."
                )
            # Recovery leases may rebind the active invocation profile to a
            # newer run epoch while the pending round and workspace effect
            # retain the epoch in which the effect was originally dispatched.
            # Authenticate the historical effect against that durable round,
            # while the checks above independently authenticate the round to
            # the current immutable invocation profile.
            if lifecycle.source_run_epoch != pending_round.source_run_epoch:
                raise workspace_observation_recovery_rejected(
                    "Workspace observation conflicts with its pending tool round."
                )
            if (
                lifecycle.model_step_id != pending_identity.model_step_id
                or lifecycle.model_attempt_id != pending_identity.model_attempt_id
                or lifecycle.tool_round_id != pending_identity.tool_round_id
                or lifecycle.model_step != pending_round.model_step
            ):
                raise workspace_observation_recovery_rejected(
                    "Workspace observation conflicts with its pending tool round."
                )
            pending_call = pending_calls.get(lifecycle.tool_call_id)
            if pending_call is None or pending_call.tool_name != lifecycle.tool_name:
                raise workspace_observation_recovery_rejected(
                    "Workspace observation conflicts with its pending tool call."
                )
            tool_call_owner = (lifecycle.tool_round_id, lifecycle.tool_call_id)
            if tool_call_owner in claimed_tool_calls:
                raise workspace_observation_recovery_rejected(
                    "Workspace observation has duplicate active lifecycles for one tool call."
                )
            claimed_tool_calls.add(tool_call_owner)
        # Factory registrations deliberately retain only an unmaterialized
        # template.  A fresh worker must not call the factory merely to inspect
        # a crashed mutation window, and the template's placeholder workspace
        # and binding are not evidence that the historical concrete authority
        # conflicts.  The authenticated lifecycle/profile/pending-round tuple
        # above is sufficient to close the durable lifecycle, but not to enter
        # extension-owned observers or artifact stores.
        return registered_environment is not None and not factory_template_unavailable

    def _reconstruct_workspace_observation_staged_outcome(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        lifecycle: WorkspaceObservationLifecycle,
    ) -> WorkspaceObservationLifecycle | None:
        """Bind a pre-crash private terminal stage to its exact observation owner."""

        if (
            lifecycle.phase is not WorkspaceObservationPhase.BEFORE_CAPTURED
            or lifecycle.tool_outcome_event_id is not None
        ):
            return None
        pending_round = pending_round_reader.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            runtime_session=session,
        )
        if pending_round is None:
            return None
        identity = ToolRoundIdentity(
            model_step_id=lifecycle.model_step_id,
            model_attempt_id=lifecycle.model_attempt_id,
            tool_round_id=lifecycle.tool_round_id,
        )
        if pending_rounds.pending_tool_round_identity(pending_round) != identity:
            return None
        matches = [
            item
            for item in pending_round.staged_terminals
            if item.tool_call_id == lifecycle.tool_call_id
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise workspace_observation_recovery_rejected(
                "Workspace observation has duplicate staged tool outcomes."
            )
        staged_event = self._validated_workspace_observation_tool_outcome(
            matches[0].event,
            lifecycle=lifecycle,
            require_bound_identity=False,
        )
        if staged_event is None:
            raise workspace_observation_recovery_rejected(
                "Workspace observation staged tool outcome conflicts with its owner."
            )
        return WorkspaceObservationLifecycle.model_validate(
            {
                **lifecycle.model_dump(mode="json"),
                "phase": WorkspaceObservationPhase.TOOL_OUTCOME_STAGED.value,
                "tool_outcome_event_id": staged_event.id,
                "tool_outcome_event_digest": workspace_observation_event_digest(staged_event),
            }
        )

    async def _repair_workspace_observation_terminal_stage(
        self,
        *,
        session: Session,
        durable_lifecycle: WorkspaceObservationLifecycle,
        lifecycle: WorkspaceObservationLifecycle,
        capture_status: Literal["recorded", "failed"],
        capture_detail_code: str | None,
    ) -> WorkspaceObservationLifecycle | None:
        """Repair one staged tool outcome and return its exact lifecycle binding."""

        if lifecycle.tool_outcome_event_id is None or lifecycle.tool_outcome_event_digest is None:
            return None
        if await self._workspace_observation_uncertainty_evidence_valid(
            session=session, lifecycle=lifecycle
        ):
            # An immutable unknown event has no tool result or capture metadata
            # to rewrite. Workspace finalization owns the observation status.
            return durable_lifecycle
        checkpoint = await await_workspace_observation_store_read(
            lambda: self._session_store.load_checkpoint(session.id),
            operation="Workspace observation terminal-stage checkpoint read",
        )
        pending_round = pending_round_reader.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            runtime_session=session,
        )
        matching_raw_stages = []
        matching_safe_stages = []
        if pending_round is not None:
            matching_raw_stages = [
                item
                for item in pending_round.staged_terminals
                if item.tool_call_id == lifecycle.tool_call_id
            ]
            matching_safe_stages = [
                item
                for item in staged_terminal_reader.staged_terminal_records(pending_round)
                if item.tool_call_id == lifecycle.tool_call_id
            ]
        if len(matching_raw_stages) > 1 or len(matching_safe_stages) > 1:
            raise workspace_observation_recovery_rejected(
                "Workspace observation has duplicate staged tool outcomes."
            )
        if bool(matching_raw_stages) != bool(matching_safe_stages):
            raise workspace_observation_recovery_rejected(
                "Workspace observation staged tool outcome projection is incomplete."
            )
        if not matching_raw_stages:
            durable = await await_workspace_observation_store_read(
                lambda: self._session_store.query_events(
                    EventQuery(
                        session_id=session.id,
                        event_id=lifecycle.tool_outcome_event_id,
                        limit=2,
                    )
                ),
                operation="Workspace observation tool-outcome event read",
            )
            available = (
                len(durable) == 1
                and self._validated_workspace_observation_tool_outcome(
                    durable[0].event,
                    lifecycle=lifecycle,
                )
                is not None
            )
            return durable_lifecycle if available else None
        authenticated_event = self._validated_workspace_observation_tool_outcome(
            matching_raw_stages[0].event,
            lifecycle=lifecycle,
        )
        if authenticated_event is None:
            raise workspace_observation_recovery_rejected(
                "Workspace observation tool outcome conflicts with its stage."
            )
        staged_event = self._validated_workspace_observation_tool_outcome(
            matching_safe_stages[0].event,
            lifecycle=lifecycle,
            require_bound_identity=False,
        )
        if staged_event is None or staged_event.id != authenticated_event.id:
            raise workspace_observation_recovery_rejected(
                "Workspace observation safe tool outcome conflicts with its stage."
            )
        identity = ToolRoundIdentity(
            model_step_id=lifecycle.model_step_id,
            model_attempt_id=lifecycle.model_attempt_id,
            tool_round_id=lifecycle.tool_round_id,
        )
        payload = dict(staged_event.payload)
        payload["workspace_mutation_capture_status"] = capture_status
        if capture_detail_code is None:
            payload.pop("workspace_mutation_capture_detail_code", None)
        else:
            payload["workspace_mutation_capture_detail_code"] = capture_detail_code
        staged_event = staged_event.model_copy(update={"payload": payload}, deep=True)
        projected_lifecycle = durable_lifecycle.model_copy(
            update={
                "tool_outcome_event_digest": workspace_observation_event_digest(staged_event),
            },
            deep=True,
        )
        stage_transform = tool_round_recovery.projected_staged_terminal_transform(
            tool_round_identity=identity,
            event=staged_event,
        )

        def guarded_stage_transform(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            # A peer recovery may validly advance the epoch or lifecycle after
            # this worker's read. Losing that race is retryable; it is not proof
            # that the durable authority tuple itself is corrupt.
            if current_session.run_epoch != session.run_epoch:
                raise _IncompleteRecoveryClaimLost(
                    "Workspace observation recovery ownership changed."
                )
            current = workspace_observations_from_checkpoint(current_checkpoint).get(
                durable_lifecycle.window_id
            )
            if current != durable_lifecycle:
                raise _IncompleteRecoveryClaimLost(
                    "Workspace observation changed before stage repair."
                )
            updated_checkpoint = stage_transform(current_session, current_checkpoint)
            observations = workspace_observations_from_checkpoint(updated_checkpoint)
            if observations.get(durable_lifecycle.window_id) != durable_lifecycle:
                raise workspace_observation_recovery_rejected(
                    "Workspace observation changed during stage repair."
                )
            observations[durable_lifecycle.window_id] = projected_lifecycle
            updated_checkpoint[WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY] = (
                workspace_observation_checkpoint_value(observations)
            )
            return updated_checkpoint

        with _workspace_observation_authority_mutation_scope():
            await await_workspace_observation_store_mutation(
                lambda: self._session_store.transform_checkpoint(
                    session.id,
                    guarded_stage_transform,
                ),
                operation="Workspace observation terminal-stage repair",
            )
        return projected_lifecycle

    @staticmethod
    def _validated_workspace_observation_tool_outcome(
        event: Event,
        *,
        lifecycle: WorkspaceObservationLifecycle,
        require_bound_identity: bool = True,
    ) -> Event | None:
        """Return one detached terminal event only when its complete owner matches."""

        if type(require_bound_identity) is not bool:
            raise TypeError("require_bound_identity must be a boolean.")

        identity = ToolRoundIdentity(
            model_step_id=lifecycle.model_step_id,
            model_attempt_id=lifecycle.model_attempt_id,
            tool_round_id=lifecycle.tool_round_id,
        )
        try:
            staged_event = restore_staged_terminal_authority(
                event,
                session_id=lifecycle.session_id,
                tool_round_identity=identity,
            )
        except Exception:
            return None
        if (
            staged_event.type
            not in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
            }
            or staged_event.session_id != lifecycle.session_id
            or staged_event.interaction_id != lifecycle.interaction_id
            or staged_event.agent_name != lifecycle.agent_name
            or staged_event.environment_name != lifecycle.environment_name
            or staged_event.tool_name != lifecycle.tool_name
            or staged_event.payload.get("tool_call_id") != lifecycle.tool_call_id
            or not identity.matches_payload(staged_event.payload)
        ):
            return None
        if require_bound_identity and (
            staged_event.id != lifecycle.tool_outcome_event_id
            or workspace_observation_event_digest(staged_event)
            != lifecycle.tool_outcome_event_digest
        ):
            return None
        return staged_event

    async def _workspace_observation_uncertainty_evidence_valid(
        self,
        *,
        session: Session,
        lifecycle: WorkspaceObservationLifecycle,
    ) -> bool:
        if lifecycle.tool_outcome_event_id is None:
            return False
        rows = await await_workspace_observation_store_read(
            lambda: self._session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_id=lifecycle.tool_outcome_event_id,
                    limit=2,
                )
            ),
            operation="Workspace observation uncertainty evidence read",
        )
        if len(rows) != 1 or rows[0].event.type is not EventType.TOOL_EFFECT_OUTCOME_UNKNOWN:
            return False
        event = rows[0].event
        record = await ToolEffectStateOwner(self._session_store).resolve_call(
            session,
            tool_round_id=lifecycle.tool_round_id,
            tool_call_id=lifecycle.tool_call_id,
        )
        if (
            record is None
            or event.session_id != lifecycle.session_id
            or event.interaction_id != lifecycle.interaction_id
            or event.agent_name != lifecycle.agent_name
            or event.environment_name != lifecycle.environment_name
            or event.tool_name != lifecycle.tool_name
            or any(
                event.payload.get(name) != getattr(lifecycle, name)
                for name in ("model_step_id", "model_attempt_id", "tool_round_id", "tool_call_id")
            )
            or workspace_observation_event_digest(event) != lifecycle.tool_outcome_event_digest
        ):
            raise workspace_observation_recovery_rejected(
                "Workspace observation uncertainty conflicts with its owner."
            )
        validate_tool_effect_uncertainty_event(event, record)
        return True

    async def _workspace_observation_tool_outcome_evidence_valid(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        lifecycle: WorkspaceObservationLifecycle,
    ) -> bool:
        """Validate exact tool evidence before any extension-owned artifact read."""

        if await self._workspace_observation_uncertainty_evidence_valid(
            session=session, lifecycle=lifecycle
        ):
            return True
        pending_round = pending_round_reader.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            runtime_session=session,
        )
        matching_raw_stages = (
            []
            if pending_round is None
            else [
                item
                for item in pending_round.staged_terminals
                if item.tool_call_id == lifecycle.tool_call_id
            ]
        )
        matching_safe_stages = (
            []
            if pending_round is None
            else [
                item
                for item in staged_terminal_reader.staged_terminal_records(pending_round)
                if item.tool_call_id == lifecycle.tool_call_id
            ]
        )
        if len(matching_raw_stages) > 1 or len(matching_safe_stages) > 1:
            raise workspace_observation_recovery_rejected(
                "Workspace observation has duplicate staged tool outcomes."
            )
        if bool(matching_raw_stages) != bool(matching_safe_stages):
            raise workspace_observation_recovery_rejected(
                "Workspace observation staged tool outcome projection is incomplete."
            )
        if matching_raw_stages:
            if (
                self._validated_workspace_observation_tool_outcome(
                    matching_raw_stages[0].event,
                    lifecycle=lifecycle,
                )
                is None
            ):
                raise workspace_observation_recovery_rejected(
                    "Workspace observation tool outcome conflicts with its stage."
                )
            if (
                self._validated_workspace_observation_tool_outcome(
                    matching_safe_stages[0].event,
                    lifecycle=lifecycle,
                    require_bound_identity=False,
                )
                is None
            ):
                raise workspace_observation_recovery_rejected(
                    "Workspace observation safe tool outcome conflicts with its stage."
                )
            return True
        if lifecycle.tool_outcome_event_id is None:
            return False
        durable = await await_workspace_observation_store_read(
            lambda: self._session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_id=lifecycle.tool_outcome_event_id,
                    limit=2,
                )
            ),
            operation="Workspace observation tool-outcome event read",
        )
        return (
            len(durable) == 1
            and self._validated_workspace_observation_tool_outcome(
                durable[0].event,
                lifecycle=lifecycle,
            )
            is not None
        )

    async def _workspace_observation_delta_evidence(
        self,
        *,
        session: Session,
        lifecycle: WorkspaceObservationLifecycle,
    ) -> tuple[
        bool,
        bool,
        WorkspaceObservationTerminalStatus,
        str | None,
    ]:
        """Classify exact durable delta evidence before artifact reconciliation."""

        if lifecycle.mutation_event_id is None or lifecycle.mutation_event_digest is None:
            raise workspace_observation_recovery_rejected(
                "Published workspace delta lost its event identity."
            )
        records = await await_workspace_observation_store_read(
            lambda: self._session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_id=lifecycle.mutation_event_id,
                    limit=2,
                )
            ),
            operation="Workspace observation delta event read",
        )
        if not records:
            return (
                False,
                False,
                WorkspaceObservationTerminalStatus.INCOMPLETE,
                "workspace_delta_evidence_missing",
            )
        if len(records) != 1 or (
            workspace_observation_event_digest(records[0].event) != lifecycle.mutation_event_digest
        ):
            return (
                False,
                True,
                WorkspaceObservationTerminalStatus.AMBIGUOUS,
                "workspace_delta_evidence_conflict",
            )
        delta_event = self._validated_workspace_observation_delta_event(
            records[0].event,
            lifecycle=lifecycle,
        )
        if delta_event is None:
            return (
                False,
                True,
                WorkspaceObservationTerminalStatus.AMBIGUOUS,
                "workspace_delta_evidence_conflict",
            )
        delta_status = delta_event.payload.get("status")
        delta_detail_code = delta_event.payload.get("detail_code")
        if type(delta_status) is not str or (
            delta_detail_code is not None and type(delta_detail_code) is not str
        ):
            return (
                False,
                True,
                WorkspaceObservationTerminalStatus.AMBIGUOUS,
                "workspace_delta_evidence_conflict",
            )
        try:
            terminal_status, terminal_detail = workspace_observation_terminal_from_delta_status(
                delta_status,
                detail_code=delta_detail_code,
            )
        except (TypeError, ValueError):
            return (
                False,
                True,
                WorkspaceObservationTerminalStatus.AMBIGUOUS,
                "workspace_delta_evidence_conflict",
            )
        return True, False, terminal_status, terminal_detail

    @staticmethod
    def _workspace_observation_unverified_artifacts(
        lifecycle: WorkspaceObservationLifecycle,
    ) -> WorkspaceObservationLifecycle:
        """Retain exact artifact identities without entering an unproven store owner."""

        artifacts = tuple(
            artifact
            if artifact.state
            in {
                WorkspaceObservationArtifactState.INTENT,
                WorkspaceObservationArtifactState.FAILED,
            }
            else artifact.model_copy(update={"state": WorkspaceObservationArtifactState.FAILED})
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

    @staticmethod
    def _validated_workspace_observation_delta_event(
        event: Event,
        *,
        lifecycle: WorkspaceObservationLifecycle,
    ) -> Event | None:
        """Return a detached delta event only when its complete owner matches."""

        try:
            delta_event = copy_event(event)
        except Exception:
            return None
        payload = delta_event.payload
        if (
            delta_event.type is not EventType.WORKSPACE_MUTATION_RECORDED
            or delta_event.id != lifecycle.mutation_event_id
            or workspace_observation_event_digest(delta_event) != lifecycle.mutation_event_digest
            or delta_event.session_id != lifecycle.session_id
            or delta_event.interaction_id != lifecycle.interaction_id
            or delta_event.agent_name != lifecycle.agent_name
            or delta_event.environment_name != lifecycle.environment_name
            or delta_event.tool_name != lifecycle.tool_name
            or payload.get("window_id") != lifecycle.window_id
            or payload.get("session_run_epoch") != lifecycle.source_run_epoch
            or payload.get("binding_generation_id") != lifecycle.binding_generation_id
            or payload.get("workspace_id") != lifecycle.workspace_id
            or payload.get("observer") != lifecycle.observer
            or payload.get("artifact_store_id") != lifecycle.artifact_store_id
            or payload.get("tool_call_id") != lifecycle.tool_call_id
            or payload.get("model_step_id") != lifecycle.model_step_id
            or payload.get("model_attempt_id") != lifecycle.model_attempt_id
            or payload.get("tool_round_id") != lifecycle.tool_round_id
            or payload.get("model_step") != lifecycle.model_step
            or payload.get("before_observation_id") != lifecycle.before_observation_id
            or payload.get("after_observation_id") != lifecycle.after_observation_id
            or payload.get("tool_outcome_event_id") != lifecycle.tool_outcome_event_id
            or payload.get("tool_outcome_event_digest") != lifecycle.tool_outcome_event_digest
        ):
            return None
        delta_artifacts = tuple(
            artifact
            for artifact in lifecycle.artifacts
            if artifact.evidence_kind == "revision-delta"
            and artifact.state is WorkspaceObservationArtifactState.REFERENCED
        )
        manifest_fields = (
            payload.get("manifest_artifact_id"),
            payload.get("manifest_artifact_sha256"),
            payload.get("manifest_artifact_size_bytes"),
        )
        if not delta_artifacts:
            if any(value is not None for value in manifest_fields):
                return None
        elif len(delta_artifacts) != 1 or manifest_fields != (
            delta_artifacts[0].artifact_id,
            delta_artifacts[0].sha256,
            delta_artifacts[0].size_bytes,
        ):
            return None
        return delta_event

    async def _reconcile_workspace_observation_artifacts(
        self,
        lifecycle: WorkspaceObservationLifecycle,
        *,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> WorkspaceObservationLifecycle:
        if not lifecycle.artifacts:
            return lifecycle
        artifact_store = None
        if registered_environment is not None:
            candidate_store = registered_environment.environment.artifact_store
            try:
                candidate_store_id = (
                    None if candidate_store is None else getattr(candidate_store, "id", None)
                )
            except Exception:
                candidate_store_id = None
            if type(candidate_store_id) is str and workspace_observation_authority_matches(
                lifecycle.artifact_store_id,
                candidate_store_id,
                field_name="artifact_store_id",
                session_id=lifecycle.session_id,
                public_authority_alias_codec=(self._session_store.public_authority_alias_codec),
            ):
                artifact_store = candidate_store
        reconciled = []
        for artifact in lifecycle.artifacts:
            if artifact.state is WorkspaceObservationArtifactState.FAILED:
                reconciled.append(artifact)
                continue
            exists = False
            valid = False
            verification_failed = artifact_store is None
            if artifact_store is not None:
                if self._workspace_artifact_recovery_operations.reserve():
                    try:
                        read_task = asyncio.create_task(
                            capture_awaitable_outcome(
                                lambda artifact_id=artifact.artifact_id, artifact_size=artifact.size_bytes: (
                                    artifact_store.read_bytes(
                                        artifact_id,
                                        max_bytes=artifact_size,
                                    )
                                )
                            )
                        )
                    except BaseException:
                        self._workspace_artifact_recovery_operations.release_reservation()
                        raise
                    self._workspace_artifact_recovery_operations.track(read_task)
                    outcome = await await_shielded_task_outcome(
                        read_task,
                        timeout_s=_WORKSPACE_ARTIFACT_RECOVERY_READ_TIMEOUT_SECONDS,
                        timeout_after_cancellation_s=0.0,
                    )
                    if read_task.done():
                        self._workspace_artifact_recovery_operations.release(read_task)
                    if outcome.cancellation is not None and not read_task.done():
                        read_task.cancel("workspace artifact recovery abandoned")
                    read_result: object = None
                    read_error = outcome.error
                    if read_error is None:
                        captured = outcome.result
                        if type(captured) is not CapturedAwaitableOutcome:
                            read_error = RuntimeError(
                                "Workspace artifact recovery returned an invalid owned outcome."
                            )
                        else:
                            read_result = captured.result
                            read_error = captured.error
                    if outcome.cancellation is not None:
                        restore_workspace_observation_cancellation_requests(
                            outcome.cancellation_requests_consumed
                        )
                    raise_workspace_observation_concurrent_control(
                        cancellation=outcome.cancellation,
                        error=read_error,
                        operation="Workspace observation artifact recovery",
                        cancellation_requests_pending=(outcome.cancellation_requests_consumed),
                    )
                    if outcome.timed_out:
                        verification_failed = True
                        read_task.cancel("workspace artifact recovery timed out")
                    elif read_error is not None and any(
                        isinstance(candidate, (KeyboardInterrupt, SystemExit, GeneratorExit))
                        for candidate in iter_exception_tree(read_error)
                    ):
                        raise read_error
                    elif read_error is not None and not isinstance(
                        read_error,
                        FileNotFoundError,
                    ):
                        verification_failed = True
                    elif read_error is None:
                        try:
                            result = copy_artifact_read_result(
                                cast("ArtifactReadResult", read_result),
                                expected_artifact_id=artifact.artifact_id,
                                max_content_bytes=artifact.size_bytes,
                            )
                            metadata = result.metadata
                            exists = True
                            valid = (
                                not result.truncated
                                and not result.redaction_truncated
                                and result.total_bytes == artifact.size_bytes
                                and result.source_bytes_read == artifact.size_bytes
                                and metadata.size_bytes == artifact.size_bytes
                                and len(result.content) == artifact.size_bytes
                                and sha256(result.content).hexdigest() == artifact.sha256
                                and workspace_observation_artifact_metadata_matches(
                                    metadata,
                                    artifact=artifact,
                                    session_id=lifecycle.session_id,
                                    agent_name=lifecycle.agent_name,
                                    environment_name=lifecycle.environment_name,
                                    window_id=lifecycle.window_id,
                                )
                            )
                        except Exception:
                            verification_failed = True
                else:
                    verification_failed = True
            if artifact.state is WorkspaceObservationArtifactState.REFERENCED:
                if verification_failed:
                    state = WorkspaceObservationArtifactState.FAILED
                elif exists and valid:
                    state = WorkspaceObservationArtifactState.REFERENCED
                else:
                    state = WorkspaceObservationArtifactState.MISSING
            elif exists and valid:
                state = WorkspaceObservationArtifactState.ORPHANED
            elif artifact.state is WorkspaceObservationArtifactState.INTENT:
                # An unacknowledged cancellation-opaque put may still finish
                # after this read. Absence is not positive evidence of failure;
                # retain the exact content identity as a possible late orphan.
                state = WorkspaceObservationArtifactState.INTENT
            else:
                state = WorkspaceObservationArtifactState.FAILED
            reconciled.append(artifact.model_copy(update={"state": state}))
        return WorkspaceObservationLifecycle.model_validate(
            {
                **lifecycle.model_dump(mode="json"),
                "artifacts": [artifact.model_dump(mode="json") for artifact in reconciled],
            }
        )
