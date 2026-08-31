"""Orchestration for task-backed durable subagent submission and recovery."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from cayu._validation import canonical_durable_json_bytes, copy_json_value
from cayu.core.events import Event
from cayu.core.tools import ToolContext, ToolResult, _runtime_tool_invocation_authority
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._checkpoint_store import load_runtime_session_checkpoint_snapshot
from cayu.runtime._durable_subagents import (
    DurableSubagentSubmissionIntent,
    DurableSubagentSubmissionReceipt,
    DurableSubagentSubmissionSeed,
    checkpoint_with_committed_durable_subagent_submission,
    checkpoint_with_durable_subagent_submission,
    checkpoint_with_durable_subagent_submission_rejection,
    checkpoint_with_durable_subagent_submission_seed,
    durable_dispatch_queue_task_id,
    durable_subagent_dispatch_id,
    durable_subagent_effective_arguments_sha256,
    durable_subagent_interaction_event_id,
    durable_subagent_interaction_id,
    durable_subagent_preparation_rejected,
    durable_subagent_request_sha256,
    durable_subagent_submission_from_checkpoint,
    durable_subagent_submission_receipt_from_checkpoint,
    durable_subagent_submission_receipt_from_intent,
    durable_subagent_submission_rejection_receipt,
    durable_subagent_submission_seed_from_checkpoint,
    durable_subagent_submission_unsettled,
    is_durable_subagent_preparation_rejected,
    is_durable_subagent_submission_unsettled,
    new_durable_subagent_submission_intent,
    new_durable_subagent_submission_seed,
    require_durable_subagent_intent_matches_seed,
    require_durable_subagent_receipt_matches_intent,
    require_durable_subagent_receipt_matches_seed,
    require_durable_subagent_rejection_receipt_matches_seed,
)
from cayu.runtime.build_provenance import RuntimeBuildProvenance
from cayu.runtime.dispatch import (
    DispatchHandle,
    DispatchStatus,
    TaskStoreDispatcher,
    _new_prepared_subagent_dispatch_envelope,
    _PreparedSubagentAlreadyAdmitted,
    _QueuedDispatchAuthorityRejected,
    _QueuedDispatchEnvelope,
    _QueuedDispatchSettlement,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileIdentity,
    active_invocation_execution_profile_from_checkpoint,
    execution_profile_from_session_metadata,
)
from cayu.runtime.invocation import (
    SessionExecutionSource,
    SessionInvocationBinding,
    inherited_session_invocation,
)
from cayu.runtime.sessions import (
    InterruptSessionRequest,
    QueuedDispatchTerminalReceipt,
    RunRequest,
    Session,
    SessionIdentity,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
    _invocation_lifecycle_authority_read_scope,
    _queued_dispatch_session_instance_fingerprint,
    _session_run_operation_from_checkpoint,
    copy_run_request,
    run_request_with_prepared_session_authority,
    run_request_with_runtime_generated_authority,
    run_request_with_runtime_invocation,
)
from cayu.runtime.tasks import TaskStatus, TaskStore
from cayu.runtime.tool_discovery import (
    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
    current_tool_discovery_view,
    initial_tool_discovery_operation_records,
    tool_discovery_generation_id,
)
from cayu.runtime.tool_exposure import (
    resolve_tool_capability_ceiling,
    tool_capability_ceiling_from_session_metadata,
)


@dataclass(frozen=True, slots=True)
class DurableSubagentPreparedRun:
    """Secret-free child preparation projected from the session engine."""

    request: RunRequest
    provider_name: str
    model: str
    runtime_name: str
    runtime_version: str | None
    runtime_build_provenance: RuntimeBuildProvenance
    execution_profile: ExecutionProfileIdentity


class DurableSubagentCoordinator:
    """Own the complete staged child/session/task durable handoff."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        runtime_session_store: SessionStore,
        task_store: TaskStore | None,
        dispatcher: object,
        prepare_initial_run: Callable[
            [RunRequest],
            Awaitable[DurableSubagentPreparedRun],
        ],
        resolve_registered_agent: Callable[
            [str],
            runtime_records.RegisteredAgentState,
        ],
        resolve_registered_provider: Callable[
            [str | None],
            runtime_records.RegisteredProvider,
        ],
        route_registered_provider_for_model: Callable[
            [str],
            runtime_records.RegisteredProvider | None,
        ],
        resolve_registered_environment: Callable[
            [str | None],
            runtime_records.RegisteredEnvironment | None,
        ],
        interrupt_session: Callable[
            ...,
            AsyncIterator[Event],
        ],
        load_session_invocation: Callable[
            [str],
            Awaitable[SessionInvocationBinding],
        ],
        classify_dispatch_settlement: Callable[
            [_QueuedDispatchEnvelope],
            Awaitable[_QueuedDispatchSettlement],
        ],
        acknowledge_dispatch: Callable[..., Awaitable[None]],
    ) -> None:
        self.session_store = session_store
        self._runtime_session_store = runtime_session_store
        self.task_store = task_store
        self.dispatcher = dispatcher
        self._prepare_initial_run = prepare_initial_run
        self._get_registered_agent = resolve_registered_agent
        self._get_registered_provider = resolve_registered_provider
        self._route_registered_provider_for_model_callback = route_registered_provider_for_model
        self._get_registered_environment = resolve_registered_environment
        self._interrupt_session_private = interrupt_session
        self._load_session_invocation = load_session_invocation
        self._classify_dispatch_settlement = classify_dispatch_settlement
        self._acknowledge_dispatch = acknowledge_dispatch

    def _route_registered_provider_for_model(
        self,
        *,
        model: str,
    ) -> runtime_records.RegisteredProvider | None:
        return self._route_registered_provider_for_model_callback(model)

    async def session_invocation_for_dispatch(
        self,
        session_id: str,
    ) -> SessionInvocationBinding:
        return await self._load_session_invocation(session_id)

    async def _queued_dispatch_settlement_state(
        self,
        envelope: _QueuedDispatchEnvelope,
    ) -> _QueuedDispatchSettlement:
        return await self._classify_dispatch_settlement(envelope)

    async def _acknowledge_queued_dispatch(
        self,
        envelope: _QueuedDispatchEnvelope,
        *,
        dispatch_status: DispatchStatus,
        receipt: QueuedDispatchTerminalReceipt | None = None,
    ) -> None:
        await self._acknowledge_dispatch(
            envelope,
            dispatch_status=dispatch_status,
            receipt=receipt,
        )

    async def _load_queued_dispatch_session_snapshot(
        self,
        session_id: str,
    ) -> tuple[Session, dict[str, Any] | None]:
        return await load_runtime_session_checkpoint_snapshot(
            self._runtime_session_store,
            session_id,
        )

    async def submit(
        self,
        *,
        context: ToolContext,
        request: RunRequest,
        agent_alias: str,
        tool_name: str,
        spawn_fingerprint: str,
        effective_arguments: dict[str, Any],
    ) -> DispatchHandle:
        """Persist one exact child and its queue task before acknowledging a spawn."""

        if type(context) is not ToolContext or type(request) is not RunRequest:
            raise TypeError("Durable subagent submission received invalid runtime input.")
        if not isinstance(self.dispatcher, TaskStoreDispatcher) or self.task_store is None:
            raise RuntimeError("Durable subagents require TaskStoreDispatcher(task_store).")
        if self.dispatcher.task_store is not self.task_store:
            raise RuntimeError(
                "Durable subagents require the coordinator and TaskStoreDispatcher to share "
                "the exact TaskStore instance."
            )
        if not self.session_store.supports_pending_session_initial_checkpoint:
            raise NotImplementedError(
                "Durable subagents require atomic PENDING-session checkpoint creation."
            )
        authority = _runtime_tool_invocation_authority(context)
        if authority is None:
            raise RuntimeError("Durable subagents require runtime-owned tool invocation authority.")
        idempotency_key = context.idempotency_key
        if type(idempotency_key) is not str:
            raise RuntimeError("Durable subagents require a runtime-owned idempotency key.")
        copied_effective_arguments = copy_json_value(
            effective_arguments,
            "effective_arguments",
        )
        if type(copied_effective_arguments) is not dict:
            raise TypeError("Durable subagent effective arguments must be an object.")
        effective_arguments_sha256 = sha256(
            canonical_durable_json_bytes(
                copied_effective_arguments,
                "effective_arguments",
            )
        ).hexdigest()
        if (
            authority.idempotency_key != idempotency_key
            or authority.effective_arguments_sha256 != effective_arguments_sha256
            or authority.tool_name != tool_name
            or request.parent_session_id != context.session_id
            or request.causal_budget_id != (context.causal_budget_id or context.session_id)
        ):
            raise RuntimeError("Durable subagent invocation authority conflicts with its request.")
        publication_snapshot = authority.secret_publication_sealer()
        if type(publication_snapshot) is not invocation_secrets.InvocationPublicationSnapshot:
            raise RuntimeError(
                "Durable subagent invocation secret publication returned invalid evidence."
            )
        if publication_snapshot.unsafe_output or publication_snapshot.secret_scope_incomplete:
            raise RuntimeError(
                "Durable subagent submission requires a complete invocation secret scope."
            )
        invocation_redactor = publication_snapshot.redactor
        parent, parent_checkpoint = await self._load_queued_dispatch_session_snapshot(
            context.session_id
        )
        registered_parent = self._get_registered_agent(parent.agent_name)
        registered_tool = registered_parent.tools.get(tool_name)
        parent_profile = active_invocation_execution_profile_from_checkpoint(parent_checkpoint)
        if (
            parent.status is not SessionStatus.RUNNING
            or parent.run_epoch != authority.parent_run_epoch
            or parent_profile is None
            or parent_profile.run_epoch != parent.run_epoch
            or parent_profile.profile.fingerprint != authority.execution_profile_fingerprint
            or registered_tool is None
            or registered_tool.child_session_recovery is None
        ):
            raise SessionRunFenced(
                "Durable subagent parent invocation no longer owns its execution profile."
            )

        request = copy_run_request(request)
        if request.session_id is None:
            raise ValueError("Durable subagent request requires a deterministic child session.")
        registered_child = self._get_registered_agent(request.agent_name)
        request = request.model_copy(
            update={
                "tool_capability_ceiling": resolve_tool_capability_ceiling(
                    request.tool_capability_ceiling,
                    registered_child.tool_capabilities,
                )
            }
        )
        dispatch_id = durable_subagent_dispatch_id(
            parent_session_id=parent.id,
            idempotency_key=idempotency_key,
        )
        queue_task_id = durable_dispatch_queue_task_id(
            task_type=self.dispatcher.prepared_subagent_task_type,
            dispatch_id=dispatch_id,
        )
        metadata = copy_json_value(request.metadata, "durable_subagent.metadata")
        subagent_metadata = metadata.get("subagent")
        if type(subagent_metadata) is not dict:
            raise ValueError("Durable subagent request has no subagent identity metadata.")
        expected_subagent_metadata = {
            "agent": agent_alias,
            "agent_name": request.agent_name,
            "context_mode": "task_only",
            "mode": "durable",
            "parent_session_id": parent.id,
            "tool_call_id": authority.tool_call_id,
            "idempotency_key": authority.idempotency_key,
            "spawn_fingerprint": spawn_fingerprint,
        }
        if (
            any(
                subagent_metadata.get(key) != value
                for key, value in expected_subagent_metadata.items()
            )
            or request.causal_budget_id != parent.causal_budget_id
            or request.environment_name != parent.environment_name
        ):
            raise RuntimeError(
                "Durable subagent request conflicts with its parent invocation authority."
            )
        subagent_metadata["durable_dispatch"] = {
            "record_type": "cayu.durable-subagent-linkage",
            "schema_version": 1,
            "parent_task_id": authority.parent_task_id,
            "parent_run_epoch": authority.parent_run_epoch,
            "model_step_id": authority.model_step_id,
            "model_attempt_id": authority.model_attempt_id,
            "tool_round_id": authority.tool_round_id,
            "tool_call_id": authority.tool_call_id,
            "dispatch_id": dispatch_id,
            "queue_task_id": queue_task_id,
            "queue_task_type": self.dispatcher.prepared_subagent_task_type,
        }
        request = request.model_copy(update={"metadata": metadata}, deep=True)
        child_session_id = request.session_id
        if child_session_id is None:  # pragma: no cover - preserved by the validated copy
            raise AssertionError("Durable subagent child session identity disappeared.")
        interaction_id = durable_subagent_interaction_id(
            child_session_id=child_session_id,
            idempotency_key=idempotency_key,
        )
        interaction_event_id = durable_subagent_interaction_event_id(interaction_id)
        seed = new_durable_subagent_submission_seed(
            parent_session_id=parent.id,
            parent_session_instance_fingerprint=(
                _queued_dispatch_session_instance_fingerprint(parent)
            ),
            parent_task_id=authority.parent_task_id,
            parent_run_epoch=authority.parent_run_epoch,
            parent_execution_profile_fingerprint=parent_profile.profile.fingerprint,
            causal_budget_id=request.causal_budget_id,
            model_step_id=authority.model_step_id,
            model_attempt_id=authority.model_attempt_id,
            tool_round_id=authority.tool_round_id,
            tool_call_id=authority.tool_call_id,
            tool_name=tool_name,
            idempotency_key=authority.idempotency_key,
            effective_arguments=copied_effective_arguments,
            effective_arguments_sha256=authority.effective_arguments_sha256,
            agent_alias=agent_alias,
            agent_name=request.agent_name,
            environment_name=request.environment_name,
            spawn_fingerprint=spawn_fingerprint,
            child_session_id=request.session_id,
            dispatch_id=dispatch_id,
            queue_task_id=queue_task_id,
            queue_task_type=self.dispatcher.prepared_subagent_task_type,
            interaction_id=interaction_id,
            interaction_started_event_id=interaction_event_id,
            request_sha256=durable_subagent_request_sha256(request),
            request=request,
        )

        def persist_parent_seed(
            current_parent: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            active = active_invocation_execution_profile_from_checkpoint(checkpoint)
            if (
                current_parent.id != parent.id
                or current_parent.run_epoch != authority.parent_run_epoch
                or active is None
                or active.profile.fingerprint != authority.execution_profile_fingerprint
            ):
                raise SessionRunFenced(
                    "Durable subagent parent ownership changed before submission."
                )
            return checkpoint_with_durable_subagent_submission_seed(
                checkpoint,
                seed=seed,
                redactor=invocation_redactor,
            )

        try:
            with _invocation_lifecycle_authority_read_scope():
                await self._runtime_session_store.transform_checkpoint(
                    parent.id,
                    persist_parent_seed,
                )
        except Exception as publication_failure:
            try:
                parent_checkpoint = await self._runtime_session_store.load_checkpoint(parent.id)
                persisted_seed = durable_subagent_submission_seed_from_checkpoint(
                    parent_checkpoint,
                    idempotency_key=seed.idempotency_key,
                )
            except Exception as reconciliation_failure:
                unsettled_failure = ExceptionGroup(
                    "Durable subagent seed publication and reconciliation failed.",
                    [publication_failure, reconciliation_failure],
                )
                raise durable_subagent_submission_unsettled(
                    parent_session_id=seed.parent_session_id,
                    tool_name=seed.tool_name,
                    idempotency_key=seed.idempotency_key,
                    failure=unsettled_failure,
                ) from unsettled_failure
            if persisted_seed is None:
                raise
            if persisted_seed != seed:
                unsettled_failure = ExceptionGroup(
                    "Durable subagent seed publication conflicted with durable state.",
                    [
                        publication_failure,
                        RuntimeError("Durable subagent seed conflicts with its exact retry."),
                    ],
                )
                raise durable_subagent_submission_unsettled(
                    parent_session_id=seed.parent_session_id,
                    tool_name=seed.tool_name,
                    idempotency_key=seed.idempotency_key,
                    failure=unsettled_failure,
                ) from unsettled_failure

        try:
            intent = await self._finalize_durable_subagent_submission_seed(
                seed,
                recovery_parent=None,
            )
            _child, handle = await self.ensure_submission(intent)
            await self._persist_committed_durable_subagent_submission(intent)
        except Exception as submission_failure:
            if is_durable_subagent_preparation_rejected(submission_failure):
                try:
                    await self._persist_durable_subagent_preparation_rejection(seed)
                except Exception as publication_failure:
                    unsettled_failure = ExceptionGroup(
                        "Durable subagent rejection publication failed.",
                        [submission_failure, publication_failure],
                    )
                    raise durable_subagent_submission_unsettled(
                        parent_session_id=seed.parent_session_id,
                        tool_name=seed.tool_name,
                        idempotency_key=seed.idempotency_key,
                        failure=unsettled_failure,
                    ) from unsettled_failure
                raise
            if is_durable_subagent_submission_unsettled(
                submission_failure,
                parent_session_id=seed.parent_session_id,
                tool_name=seed.tool_name,
                idempotency_key=seed.idempotency_key,
            ):
                raise
            raise durable_subagent_submission_unsettled(
                parent_session_id=seed.parent_session_id,
                tool_name=seed.tool_name,
                idempotency_key=seed.idempotency_key,
                failure=submission_failure,
            ) from submission_failure
        return handle

    async def _persist_durable_subagent_preparation_rejection(
        self,
        seed: DurableSubagentSubmissionSeed,
        *,
        recovery_parent: Session | None = None,
    ) -> DurableSubagentSubmissionReceipt:
        """Commit or reconstruct one exact permanent pre-dispatch rejection."""

        expected = durable_subagent_submission_rejection_receipt(seed)

        def persist_rejection(
            current_parent: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            active = active_invocation_execution_profile_from_checkpoint(checkpoint)
            normal_owner_matches = (
                recovery_parent is None
                and current_parent.run_epoch == seed.parent_run_epoch
                and active is not None
                and active.profile.fingerprint == seed.parent_execution_profile_fingerprint
            )
            recovery_owner_matches = (
                recovery_parent is not None
                and recovery_parent.id == current_parent.id
                and recovery_parent.run_epoch == current_parent.run_epoch
                and recovery_parent.status is current_parent.status
                and _queued_dispatch_session_instance_fingerprint(recovery_parent)
                == seed.parent_session_instance_fingerprint
            )
            if _queued_dispatch_session_instance_fingerprint(
                current_parent
            ) != seed.parent_session_instance_fingerprint or not (
                normal_owner_matches or recovery_owner_matches
            ):
                raise SessionRunFenced(
                    "Durable subagent rejection no longer owns the parent invocation."
                )
            return checkpoint_with_durable_subagent_submission_rejection(
                checkpoint,
                seed=seed,
            )

        try:
            with _invocation_lifecycle_authority_read_scope():
                await self._runtime_session_store.transform_checkpoint(
                    seed.parent_session_id,
                    persist_rejection,
                )
        except Exception as publication_failure:
            try:
                checkpoint = await self._runtime_session_store.load_checkpoint(
                    seed.parent_session_id
                )
                receipt = durable_subagent_submission_receipt_from_checkpoint(
                    checkpoint,
                    idempotency_key=seed.idempotency_key,
                )
            except Exception as reconciliation_failure:
                publication_failure.add_note(
                    "Durable subagent rejection reconciliation also failed: "
                    f"{type(reconciliation_failure).__name__}."
                )
                raise publication_failure from reconciliation_failure
            if receipt != expected:
                raise
        return expected

    async def _persist_committed_durable_subagent_submission(
        self,
        intent: DurableSubagentSubmissionIntent,
        *,
        recovery_parent: Session | None = None,
    ) -> None:
        """Compact one positively confirmed child/task handoff in its parent."""

        expected = durable_subagent_submission_receipt_from_intent(intent)

        def persist_receipt(
            current_parent: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            active = active_invocation_execution_profile_from_checkpoint(checkpoint)
            normal_owner_matches = (
                recovery_parent is None
                and current_parent.run_epoch == intent.parent_run_epoch
                and active is not None
                and active.profile.fingerprint == intent.parent_execution_profile_fingerprint
            )
            recovery_owner_matches = (
                recovery_parent is not None
                and recovery_parent.id == current_parent.id
                and recovery_parent.run_epoch == current_parent.run_epoch
                and recovery_parent.status is current_parent.status
                and _queued_dispatch_session_instance_fingerprint(recovery_parent)
                == intent.parent_session_instance_fingerprint
            )
            if _queued_dispatch_session_instance_fingerprint(
                current_parent
            ) != intent.parent_session_instance_fingerprint or not (
                normal_owner_matches or recovery_owner_matches
            ):
                raise SessionRunFenced(
                    "Durable subagent handoff no longer owns the parent invocation."
                )
            return checkpoint_with_committed_durable_subagent_submission(
                checkpoint,
                intent=intent,
            )

        try:
            with _invocation_lifecycle_authority_read_scope():
                await self._runtime_session_store.transform_checkpoint(
                    intent.parent_session_id,
                    persist_receipt,
                )
        except Exception as publication_failure:
            try:
                checkpoint = await self._runtime_session_store.load_checkpoint(
                    intent.parent_session_id
                )
                receipt = durable_subagent_submission_receipt_from_checkpoint(
                    checkpoint,
                    idempotency_key=intent.idempotency_key,
                )
                seed = durable_subagent_submission_seed_from_checkpoint(
                    checkpoint,
                    idempotency_key=intent.idempotency_key,
                )
                persisted_intent = durable_subagent_submission_from_checkpoint(
                    checkpoint,
                    idempotency_key=intent.idempotency_key,
                )
            except Exception as reconciliation_failure:
                publication_failure.add_note(
                    "Durable subagent handoff-receipt reconciliation also failed: "
                    f"{type(reconciliation_failure).__name__}."
                )
                raise publication_failure from reconciliation_failure
            compacted = receipt == expected and seed is None and persisted_intent is None
            retained_intent = (
                receipt is None
                and seed is not None
                and persisted_intent is not None
                and persisted_intent == intent
            )
            if retained_intent:
                require_durable_subagent_intent_matches_seed(persisted_intent, seed)
            retained_arguments = (
                receipt is not None
                and receipt == expected
                and seed is not None
                and persisted_intent is None
            )
            if retained_arguments:
                require_durable_subagent_intent_matches_seed(intent, seed)
                require_durable_subagent_receipt_matches_seed(receipt, seed)
            if not compacted and not retained_intent and not retained_arguments:
                raise

    async def _finalize_durable_subagent_submission_seed(
        self,
        seed: DurableSubagentSubmissionSeed,
        *,
        recovery_parent: Session | None,
    ) -> DurableSubagentSubmissionIntent:
        """Prepare and freeze one marker-backed child execution profile."""

        seed = DurableSubagentSubmissionSeed.model_validate(
            seed.model_dump(mode="json", warnings=False)
        )
        parent, parent_checkpoint = await self._load_queued_dispatch_session_snapshot(
            seed.parent_session_id
        )
        active = active_invocation_execution_profile_from_checkpoint(parent_checkpoint)
        if (
            _queued_dispatch_session_instance_fingerprint(parent)
            != seed.parent_session_instance_fingerprint
            or active is None
            or active.profile.fingerprint != seed.parent_execution_profile_fingerprint
            or (
                recovery_parent is None
                and (
                    parent.status is not SessionStatus.RUNNING
                    or parent.run_epoch != seed.parent_run_epoch
                )
            )
            or (
                recovery_parent is not None
                and (
                    parent.id != recovery_parent.id or parent.run_epoch != recovery_parent.run_epoch
                )
            )
        ):
            raise SessionRunFenced(
                "Durable subagent preparation no longer owns the parent profile."
            )
        persisted_seed = durable_subagent_submission_seed_from_checkpoint(
            parent_checkpoint,
            idempotency_key=seed.idempotency_key,
        )
        existing_intent = durable_subagent_submission_from_checkpoint(
            parent_checkpoint,
            idempotency_key=seed.idempotency_key,
        )
        existing_receipt = durable_subagent_submission_receipt_from_checkpoint(
            parent_checkpoint,
            idempotency_key=seed.idempotency_key,
        )
        if existing_receipt is not None:
            require_durable_subagent_receipt_matches_seed(existing_receipt, seed)
            if existing_receipt.outcome == "rejected":
                if persisted_seed != seed:
                    raise RuntimeError(
                        "Rejected durable subagent receipt lost its preparation seed."
                    )
                require_durable_subagent_rejection_receipt_matches_seed(
                    existing_receipt,
                    seed,
                )
                raise durable_subagent_preparation_rejected()
            if existing_intent is not None:
                raise RuntimeError(
                    "Committed durable subagent receipt has duplicate parent authority."
                )
            if persisted_seed is not None and persisted_seed != seed:
                raise RuntimeError(
                    "Committed durable subagent receipt has conflicting recovery authority."
                )
            child_checkpoint = await self._runtime_session_store.load_checkpoint(
                existing_receipt.child_session_id
            )
            child_intent = durable_subagent_submission_from_checkpoint(
                child_checkpoint,
                idempotency_key=seed.idempotency_key,
            )
            if child_intent is None:
                raise RuntimeError(
                    "Committed durable subagent receipt has no child submission intent."
                )
            require_durable_subagent_receipt_matches_intent(
                existing_receipt,
                child_intent,
            )
            return child_intent
        if persisted_seed != seed:
            raise RuntimeError("Durable subagent preparation seed is missing or conflicting.")
        if existing_intent is not None:
            require_durable_subagent_intent_matches_seed(existing_intent, seed)
            return existing_intent

        try:
            registered_child = self._get_registered_agent(seed.request.agent_name)
            if seed.request.target is not None:
                child_model = seed.request.target.model
                self._get_registered_provider(seed.request.target.provider_name)
            else:
                child_model = registered_child.spec.model
                if registered_child.spec.provider_name is not None:
                    self._get_registered_provider(registered_child.spec.provider_name)
                else:
                    self._route_registered_provider_for_model(
                        model=child_model
                    ) or self._get_registered_provider(None)
            self._get_registered_environment(seed.request.environment_name)
        except (KeyError, RuntimeError, ValueError) as rejection:
            raise durable_subagent_preparation_rejected() from rejection
        finally:
            registered_child = None

        prepared = await self._prepare_initial_run(seed.request)
        request = prepared.request
        child_provider_name = prepared.provider_name
        child_model = prepared.model
        child_runtime_name = prepared.runtime_name
        child_runtime_version = prepared.runtime_version
        child_runtime_build_provenance = prepared.runtime_build_provenance
        child_execution_profile = prepared.execution_profile
        # Do not retain the provider-bearing preparation bundle across the durable
        # checkpoint publication below. Store failures may escape with frame locals.
        del prepared
        if (
            request.session_id != seed.child_session_id
            or request.parent_session_id != seed.parent_session_id
            or request.causal_budget_id != seed.causal_budget_id
            or request.agent_name != seed.agent_name
            or request.environment_name != seed.environment_name
        ):
            raise RuntimeError("Durable subagent preparation changed immutable submission linkage.")
        intent = new_durable_subagent_submission_intent(
            parent_session_id=seed.parent_session_id,
            parent_session_instance_fingerprint=(seed.parent_session_instance_fingerprint),
            parent_task_id=seed.parent_task_id,
            parent_run_epoch=seed.parent_run_epoch,
            parent_execution_profile_fingerprint=(seed.parent_execution_profile_fingerprint),
            causal_budget_id=seed.causal_budget_id,
            model_step_id=seed.model_step_id,
            model_attempt_id=seed.model_attempt_id,
            tool_round_id=seed.tool_round_id,
            tool_call_id=seed.tool_call_id,
            tool_name=seed.tool_name,
            idempotency_key=seed.idempotency_key,
            effective_arguments_sha256=seed.effective_arguments_sha256,
            agent_alias=seed.agent_alias,
            agent_name=seed.agent_name,
            child_provider_name=child_provider_name,
            child_model=child_model,
            child_runtime_name=child_runtime_name,
            child_runtime_version=child_runtime_version,
            child_runtime_build_provenance=child_runtime_build_provenance,
            environment_name=seed.environment_name,
            spawn_fingerprint=seed.spawn_fingerprint,
            child_session_id=seed.child_session_id,
            dispatch_id=seed.dispatch_id,
            queue_task_id=seed.queue_task_id,
            queue_task_type=seed.queue_task_type,
            interaction_id=seed.interaction_id,
            interaction_started_event_id=seed.interaction_started_event_id,
            seed_sha256=seed.seed_sha256,
            request_sha256=durable_subagent_request_sha256(request),
            request=request,
            child_execution_profile=child_execution_profile,
        )
        require_durable_subagent_intent_matches_seed(intent, seed)

        def persist_parent_intent(
            current_parent: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            current_seed = durable_subagent_submission_seed_from_checkpoint(
                checkpoint,
                idempotency_key=seed.idempotency_key,
            )
            current_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            if (
                current_parent.id != parent.id
                or current_parent.run_epoch != parent.run_epoch
                or current_seed != seed
                or current_profile is None
                or current_profile.profile.fingerprint != seed.parent_execution_profile_fingerprint
            ):
                raise SessionRunFenced(
                    "Durable subagent parent ownership changed during preparation."
                )
            return checkpoint_with_durable_subagent_submission(checkpoint, intent=intent)

        try:
            with _invocation_lifecycle_authority_read_scope():
                await self._runtime_session_store.transform_checkpoint(
                    parent.id,
                    persist_parent_intent,
                )
        except Exception as publication_failure:
            try:
                reconciled_checkpoint = await self._runtime_session_store.load_checkpoint(parent.id)
                persisted_intent = durable_subagent_submission_from_checkpoint(
                    reconciled_checkpoint,
                    idempotency_key=seed.idempotency_key,
                )
            except Exception as reconciliation_failure:
                publication_failure.add_note(
                    "Durable subagent parent-intent reconciliation also failed: "
                    f"{type(reconciliation_failure).__name__}."
                )
                raise publication_failure from reconciliation_failure
            if persisted_intent != intent:
                raise
        return intent

    async def ensure_submission(
        self,
        intent: DurableSubagentSubmissionIntent,
    ) -> tuple[Session, DispatchHandle]:
        """Reconcile the recoverable child-first publication state machine."""

        if (
            not isinstance(self.dispatcher, TaskStoreDispatcher)
            or self.task_store is None
            or self.dispatcher.task_store is not self.task_store
        ):
            raise RuntimeError("Durable subagent reconciliation requires a task dispatcher.")
        intent = DurableSubagentSubmissionIntent.model_validate(
            intent.model_dump(mode="json", warnings=False)
        )
        parent = await self.session_store.load(intent.parent_session_id)
        if (
            parent is None
            or _queued_dispatch_session_instance_fingerprint(parent)
            != intent.parent_session_instance_fingerprint
        ):
            raise RuntimeError("Durable subagent parent session identity changed.")
        # Queue publication is strictly child-first across the two stores. Read the
        # queue side first so observing a task establishes that a subsequent child
        # read must also observe its already-committed prerequisite. The opposite
        # order can combine two individually valid READ COMMITTED snapshots into a
        # false task-without-child contradiction during concurrent reconciliation.
        queue_task = await self.task_store.load_task(intent.queue_task_id)
        child = await self.session_store.load(intent.child_session_id)
        if queue_task is not None and child is None:
            raise RuntimeError("Durable subagent task exists without its referenced child session.")

        registered_child = self._get_registered_agent(intent.agent_name)
        discovery_initializer = None
        if registered_child.tool_discovery_mode is not None:
            if not self.session_store.supports_atomic_session_operation_initialization:
                raise RuntimeError(
                    "Tool discovery requires atomic session operation initialization."
                )
            discovery_ceiling = intent.request.tool_capability_ceiling
            if discovery_ceiling is None:
                raise RuntimeError("Durable subagent discovery has no capability ceiling.")

            def initialize_discovery_view(
                session: Session,
            ) -> dict[str, dict[str, Any]]:
                return initial_tool_discovery_operation_records(
                    session_id=session.id,
                    root_invocation_id=session.invocation.root_invocation_id,
                    agent_name=registered_child.spec.name,
                    catalogue=registered_child.tool_catalogue,
                    ceiling=discovery_ceiling,
                )

            discovery_initializer = initialize_discovery_view

        if child is None:
            request = copy_run_request(intent.request)
            request = run_request_with_runtime_generated_authority(
                request,
                "session_id",
                "parent_session_id",
                "causal_budget_id",
            )
            request = run_request_with_runtime_invocation(
                request,
                source=SessionExecutionSource.SUBAGENT,
            )

            def persist_child_submission(
                current_child: Session,
                checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any]:
                if current_child.id != intent.child_session_id:
                    raise RuntimeError("Durable subagent child identity changed during creation.")
                return checkpoint_with_durable_subagent_submission(
                    checkpoint,
                    intent=intent,
                )

            try:
                child = await self._runtime_session_store.create(
                    request,
                    identity=SessionIdentity(
                        provider_name=intent.child_provider_name,
                        model=intent.child_model,
                        runtime_name=intent.child_runtime_name,
                        runtime_version=intent.child_runtime_version,
                        runtime_build_provenance=(intent.child_runtime_build_provenance),
                        execution_profile=intent.child_execution_profile,
                    ),
                    checkpoint_transform=persist_child_submission,
                    operation_initializer=discovery_initializer,
                )
            except Exception as publication_failure:
                try:
                    child = await self.session_store.load(intent.child_session_id)
                except Exception as reconciliation_failure:
                    publication_failure.add_note(
                        "Durable subagent child publication reconciliation also failed: "
                        f"{type(reconciliation_failure).__name__}."
                    )
                    raise publication_failure from reconciliation_failure
                if child is None:
                    raise
        child_checkpoint = await self._runtime_session_store.load_checkpoint(child.id)
        child_intent = durable_subagent_submission_from_checkpoint(
            child_checkpoint,
            idempotency_key=intent.idempotency_key,
        )
        try:
            child_tool_capability_ceiling = tool_capability_ceiling_from_session_metadata(
                child.metadata
            )
        except ValueError as exc:
            raise RuntimeError(
                "Existing durable subagent child has no tool capability ceiling."
            ) from exc
        if registered_child.tool_discovery_mode is not None:
            current_tool_discovery_view(
                await self.session_store.load_session_operation(
                    child.id,
                    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
                ),
                session_id=child.id,
                generation_id=tool_discovery_generation_id(
                    session_id=child.id,
                    root_invocation_id=child.invocation.root_invocation_id,
                ),
                agent_name=registered_child.spec.name,
                catalogue=registered_child.tool_catalogue,
                ceiling=child_tool_capability_ceiling,
            )
        if (
            child_intent != intent
            or intent.request.tool_capability_ceiling is None
            or child_tool_capability_ceiling != intent.request.tool_capability_ceiling
            or child.parent_session_id != parent.id
            or child.causal_budget_id != intent.causal_budget_id
            or child.agent_name != intent.agent_name
            or child.provider_name != intent.child_provider_name
            or child.model != intent.child_model
            or child.runtime_name != intent.child_runtime_name
            or child.runtime_version != intent.child_runtime_version
            or child.runtime_build_provenance != intent.child_runtime_build_provenance
            or child.environment_name != intent.environment_name
            or child.invocation
            != inherited_session_invocation(
                parent.invocation,
                source=SessionExecutionSource.SUBAGENT,
            )
            or execution_profile_from_session_metadata(child.metadata)
            != intent.child_execution_profile
        ):
            raise RuntimeError(
                "Existing durable subagent child conflicts with its submission authority."
            )
        envelope = _new_prepared_subagent_dispatch_envelope(
            intent=intent,
            session_instance_fingerprint=_queued_dispatch_session_instance_fingerprint(child),
        )
        handle = await self.dispatcher._submit_prepared_subagent(self, envelope)
        return child, handle

    async def reconcile(
        self,
        *,
        parent_session: Session,
        tool_name: str,
        tool_round_id: str,
        tool_call_id: str,
        idempotency_key: str,
        effective_arguments: dict[str, Any],
    ) -> Session | ToolResult | None:
        """Finish a marker-backed submission during pending-tool-round recovery."""

        copied_effective_arguments = copy_json_value(
            effective_arguments,
            "effective_arguments",
        )
        if type(copied_effective_arguments) is not dict:
            raise TypeError("Durable subagent effective arguments must be an object.")
        effective_arguments_sha256 = durable_subagent_effective_arguments_sha256(
            copied_effective_arguments
        )

        checkpoint = await self._runtime_session_store.load_checkpoint(parent_session.id)
        intent = durable_subagent_submission_from_checkpoint(
            checkpoint,
            idempotency_key=idempotency_key,
        )
        seed = durable_subagent_submission_seed_from_checkpoint(
            checkpoint,
            idempotency_key=idempotency_key,
        )
        receipt = durable_subagent_submission_receipt_from_checkpoint(
            checkpoint,
            idempotency_key=idempotency_key,
        )
        if intent is None and seed is None and receipt is None:
            return None
        if intent is not None and receipt is not None:
            raise RuntimeError("Durable subagent recovery has duplicate submission authority.")
        authority = receipt if receipt is not None else (intent if intent is not None else seed)
        if authority is None:  # pragma: no cover - narrowed above
            raise AssertionError("Durable subagent recovery authority disappeared.")
        if (
            authority.parent_session_id != parent_session.id
            or authority.tool_name != tool_name
            or authority.tool_round_id != tool_round_id
            or authority.tool_call_id != tool_call_id
            or authority.effective_arguments_sha256 != effective_arguments_sha256
        ):
            raise RuntimeError(
                "Durable subagent recovery authority conflicts with its pending tool call."
            )
        if receipt is not None:
            if receipt.outcome == "rejected":
                if seed is None:
                    raise RuntimeError(
                        "Rejected durable subagent receipt has no recovery arguments."
                    )
                require_durable_subagent_rejection_receipt_matches_seed(receipt, seed)
                if seed.effective_arguments != copied_effective_arguments:
                    raise RuntimeError(
                        "Durable subagent recovery arguments conflict with its preparation seed."
                    )
            else:
                child_checkpoint = await self._runtime_session_store.load_checkpoint(
                    receipt.child_session_id
                )
                intent = durable_subagent_submission_from_checkpoint(
                    child_checkpoint,
                    idempotency_key=idempotency_key,
                )
                if intent is None:
                    raise RuntimeError(
                        "Committed durable subagent receipt has no child submission intent."
                    )
                require_durable_subagent_receipt_matches_intent(receipt, intent)
                if seed is not None:
                    require_durable_subagent_intent_matches_seed(intent, seed)
                    require_durable_subagent_receipt_matches_seed(receipt, seed)
                    if seed.effective_arguments != copied_effective_arguments:
                        raise RuntimeError(
                            "Durable subagent recovery arguments conflict with its "
                            "preparation seed."
                        )
        elif intent is not None:
            if seed is None:
                raise RuntimeError("Durable subagent intent has no preparation seed.")
            require_durable_subagent_intent_matches_seed(intent, seed)
            if seed.effective_arguments != copied_effective_arguments:
                raise RuntimeError(
                    "Durable subagent recovery arguments conflict with its preparation seed."
                )
        else:
            if seed is None:  # pragma: no cover - narrowed above
                raise AssertionError("Durable subagent preparation seed disappeared.")
            if seed.effective_arguments != copied_effective_arguments:
                raise RuntimeError(
                    "Durable subagent recovery arguments conflict with its preparation seed."
                )

        registered_parent = self._get_registered_agent(parent_session.agent_name)
        registered_tool = registered_parent.tools.get(tool_name)
        matcher = None if registered_tool is None else registered_tool.child_session_recovery
        if matcher is None:
            raise RuntimeError("Durable subagent recovery matcher is unavailable.")
        matched = matcher.matches_recoverable_submission(
            parent_session=parent_session,
            tool_name=tool_name,
            tool_round_id=tool_round_id,
            tool_call_id=tool_call_id,
            idempotency_key=idempotency_key,
            arguments=copied_effective_arguments,
            spawn_fingerprint=authority.spawn_fingerprint,
        )
        if type(matched) is not bool:
            raise TypeError("Child-session submission recovery matchers must return bool.")
        if not matched:
            raise RuntimeError(
                "Durable subagent recovery no longer matches its registered tool contract."
            )
        if receipt is not None and receipt.outcome == "rejected":
            return durable_subagent_preparation_rejection_result(receipt)
        if intent is None:
            if seed is None:  # pragma: no cover - authority branches above narrow this
                raise AssertionError("Durable subagent preparation seed disappeared.")
            try:
                intent = await self._finalize_durable_subagent_submission_seed(
                    seed,
                    recovery_parent=parent_session,
                )
            except Exception as preparation_failure:
                if not is_durable_subagent_preparation_rejected(preparation_failure):
                    raise
                receipt = await self._persist_durable_subagent_preparation_rejection(
                    seed,
                    recovery_parent=parent_session,
                )
                return durable_subagent_preparation_rejection_result(receipt)
        child, _handle = await self.ensure_submission(intent)
        await self._persist_committed_durable_subagent_submission(
            intent,
            recovery_parent=parent_session,
        )
        if self.task_store is None:
            raise RuntimeError("Durable subagent reconciliation lost its task store.")
        queue_task = await self.task_store.load_task(intent.queue_task_id)
        if queue_task is None:
            raise RuntimeError("Durable subagent reconciliation lost its queue task.")
        terminal_child_statuses = {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }
        if queue_task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            refreshed = await self.session_store.load(child.id)
            if refreshed is None:
                raise RuntimeError("Durable subagent task outlived its child session.")
            child = refreshed
        if queue_task.status is TaskStatus.COMPLETED and (
            child.status not in terminal_child_statuses
        ):
            raise RuntimeError("Durable subagent task completed without a terminal child session.")
        if queue_task.status in {TaskStatus.FAILED, TaskStatus.CANCELLED} and (
            child.status not in terminal_child_statuses
        ):
            async for _event in self._interrupt_session_private(
                InterruptSessionRequest(
                    session_id=child.id,
                    reason="durable_subagent_queue_task_terminal",
                    metadata={
                        "queue_task_id": intent.queue_task_id,
                        "queue_task_status": queue_task.status.value,
                    },
                ),
                store_resolved_session_id=child.id,
            ):
                pass
            refreshed = await self.session_store.load(child.id)
            if refreshed is None or refreshed.status not in terminal_child_statuses:
                raise RuntimeError(
                    "Durable subagent queue task is terminal but its child did not settle."
                )
            child = refreshed
        return child

    def prepare_queued_child_run(
        self,
        *,
        envelope: _QueuedDispatchEnvelope,
        session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> RunRequest:
        """Validate a prepared child and return its runtime-owned run request."""

        intent = envelope.prepared_subagent
        if envelope.operation_kind != "prepared_subagent" or intent is None:
            raise _QueuedDispatchAuthorityRejected(
                "Prepared subagent dispatch lost its submission authority."
            )
        try:
            child_intent = durable_subagent_submission_from_checkpoint(
                checkpoint,
                idempotency_key=intent.idempotency_key,
            )
            child_tool_capability_ceiling = tool_capability_ceiling_from_session_metadata(
                session.metadata
            )
        except (TypeError, ValueError) as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Prepared subagent session authority is malformed."
            ) from exc
        if (
            child_intent != intent
            or intent.request.tool_capability_ceiling is None
            or child_tool_capability_ceiling != intent.request.tool_capability_ceiling
            or session.parent_session_id != intent.parent_session_id
            or session.causal_budget_id != intent.causal_budget_id
            or session.runtime_name != intent.child_runtime_name
            or session.runtime_version != intent.child_runtime_version
            or session.runtime_build_provenance != intent.child_runtime_build_provenance
            or execution_profile_from_session_metadata(session.metadata)
            != intent.child_execution_profile
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Prepared subagent session conflicts with its queue authority."
            )
        if session.status is not SessionStatus.PENDING:
            try:
                run_operation = _session_run_operation_from_checkpoint(checkpoint)
            except (TypeError, ValueError) as exc:
                raise _QueuedDispatchAuthorityRejected(
                    "Prepared subagent run-operation authority is malformed."
                ) from exc
            if session.status in {
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.INTERRUPTED,
            } and (
                run_operation is None
                or run_operation.operation_id != envelope.dispatch_operation_id
                or run_operation.queue_task_id != envelope.queue_task_id
                or run_operation.terminal_event_id != envelope.terminal_event_id
            ):
                raise _QueuedDispatchAuthorityRejected(
                    "Prepared subagent was terminalized before queue admission."
                )
            raise _PreparedSubagentAlreadyAdmitted(
                "Prepared durable child was already admitted by an earlier worker, "
                "but exact terminal replay is not yet available."
            )
        run_request = copy_run_request(intent.request)
        run_request = run_request_with_runtime_generated_authority(
            run_request,
            "session_id",
            "parent_session_id",
            "causal_budget_id",
        )
        run_request = run_request_with_runtime_invocation(
            run_request,
            source=SessionExecutionSource.SUBAGENT,
        )
        return run_request_with_prepared_session_authority(
            run_request,
            session_id=intent.child_session_id,
            queue_task_id=intent.queue_task_id,
            dispatch_operation_id=envelope.dispatch_operation_id,
            terminal_event_id=envelope.terminal_event_id,
            interaction_id=intent.interaction_id,
            interaction_started_event_id=intent.interaction_started_event_id,
            idempotency_key=intent.idempotency_key,
            submission_sha256=intent.submission_sha256,
        )

    async def require_prepared_subagent_parent_authority(
        self,
        envelope: _QueuedDispatchEnvelope,
    ) -> None:
        """Validate a prepared child's queue intent against its durable parent seed."""

        intent = envelope.prepared_subagent
        if envelope.operation_kind != "prepared_subagent" or intent is None:
            raise _QueuedDispatchAuthorityRejected(
                "Prepared subagent dispatch lost its submission authority."
            )
        try:
            parent, parent_checkpoint = await self._load_queued_dispatch_session_snapshot(
                intent.parent_session_id
            )
        except KeyError as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Prepared subagent dispatch has no durable parent session."
            ) from exc
        try:
            seed = durable_subagent_submission_seed_from_checkpoint(
                parent_checkpoint,
                idempotency_key=intent.idempotency_key,
            )
            parent_intent = durable_subagent_submission_from_checkpoint(
                parent_checkpoint,
                idempotency_key=intent.idempotency_key,
            )
            parent_receipt = durable_subagent_submission_receipt_from_checkpoint(
                parent_checkpoint,
                idempotency_key=intent.idempotency_key,
            )
        except (TypeError, ValueError) as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Prepared subagent parent submission authority is malformed."
            ) from exc
        if seed is not None and parent_intent is not None and parent_receipt is None:
            try:
                require_durable_subagent_intent_matches_seed(parent_intent, seed)
            except (RuntimeError, TypeError) as exc:
                raise _QueuedDispatchAuthorityRejected(
                    "Prepared subagent dispatch conflicts with its durable parent seed."
                ) from exc
            if parent_intent != intent:
                raise _QueuedDispatchAuthorityRejected(
                    "Prepared subagent queue intent conflicts with its durable parent intent."
                )
        elif parent_intent is None and parent_receipt is not None:
            try:
                require_durable_subagent_receipt_matches_intent(parent_receipt, intent)
                if seed is not None:
                    require_durable_subagent_intent_matches_seed(intent, seed)
                    require_durable_subagent_receipt_matches_seed(parent_receipt, seed)
            except (RuntimeError, TypeError) as exc:
                raise _QueuedDispatchAuthorityRejected(
                    "Prepared subagent dispatch conflicts with its compact parent receipt."
                ) from exc
        else:
            raise _QueuedDispatchAuthorityRejected(
                "Prepared subagent dispatch has incomplete durable parent authority."
            )
        if (
            _queued_dispatch_session_instance_fingerprint(parent)
            != intent.parent_session_instance_fingerprint
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Prepared subagent parent session instance changed."
            )


def durable_subagent_preparation_rejection_result(
    receipt: DurableSubagentSubmissionReceipt,
) -> ToolResult:
    """Project one authenticated permanent rejection without exposing its cause."""

    if type(receipt) is not DurableSubagentSubmissionReceipt or receipt.outcome != "rejected":
        raise TypeError("Durable subagent rejection result requires an exact rejected receipt.")
    return ToolResult(
        content=f"Durable subagent {receipt.agent_alias} could not be submitted.",
        structured={
            "agent": receipt.agent_alias,
            "agent_name": receipt.agent_name,
            "context_mode": "task_only",
            "mode": "durable",
            "parent_session_id": receipt.parent_session_id,
            "child_session_id": receipt.child_session_id,
            "causal_budget_id": receipt.causal_budget_id,
            "status": "submission_failed",
            "error_type": "DurableSubagentPreparationRejected",
            "failure_code": "preparation_rejected",
        },
        is_error=True,
    )
