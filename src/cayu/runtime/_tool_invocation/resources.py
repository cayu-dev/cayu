"""Bind tool resources and durable invocation authority without dispatch."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from cayu._task_wait import (
    await_shielded_task_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import (
    copy_durable_json_object,
)
from cayu.artifacts._images import ImageDecodePolicy
from cayu.budgets.base import BudgetLimit, _copy_budget_limit_definition
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
)
from cayu.execution_units import ToolRoundIdentity
from cayu.knowledge._publication import KnowledgePublicationScope
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._browser_control_bootstrap import BrowserGuestBootstrap
from cayu.runtime._browser_control_model import (
    browser_model_control_admission,
    browser_terminal_checkpoint_mutation,
    validate_browser_model_publication,
)
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._tool_invocation.context import (
    _artifact_store,
    _artifact_store_id,
    _knowledge_access_scope,
    _knowledge_store,
    _mcp_servers,
    _runner,
    _workspace,
    _workspace_id,
    _workspace_receipt_artifact_store,
)
from cayu.sessions._browser_control_checkpoint import (
    browser_control_checkpoint_mutation_scope,
    browser_control_checkpoint_read_scope,
)
from cayu.sessions.base import (
    Session,
    SessionOperationPublication,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
)
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._resources import (
    InvocationWorkspaceMutationOwner,
    invocation_artifact_store_handle,
    invocation_workspace_handle,
)
from cayu.tools._runner import (
    invocation_runner_handle,
)
from cayu.tools.base import (
    DurableToolOperationConflict,
    ToolContext,
    _bind_runtime_tool_invocation_authority,
    _RuntimeBrowserAllocationAuthority,
)
from cayu.tools.catalogue import SEARCH_TOOLS_NAME
from cayu.tools.discovery import (
    _bind_runtime_tool_discovery_authority,
    tool_discovery_generation_id,
)
from cayu.tools.exposure import (
    ResolvedToolExposureAuthority,
    tool_capability_ceiling_from_session_metadata,
)
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.mutation_attribution import (
    DirectWorkspaceMutationCollector,
)


@dataclass(frozen=True, slots=True)
class ToolInvocationCall:
    """One admitted call and the frozen authority shared by its invocation phases."""

    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    registered_tool: runtime_records.RegisteredTool
    tool_call: runtime_records.ToolCallRequest
    effective_tool_call: runtime_records.ToolCallRequest
    tool_round_identity: ToolRoundIdentity
    execution_profile: ExecutionProfileIdentity | None
    invocation_context: InvocationContext | None
    task_id: str | None
    budget_limits: tuple[BudgetLimit, ...]
    tool_exposure: ResolvedToolExposureAuthority | None
    environment_name: str | None
    idempotency_key: str
    model_step: int | None
    approval_id: str | None
    input_id: str | None


@dataclass(frozen=True, slots=True)
class BoundInvocationResources:
    context: ToolContext
    raw_workspace: Any
    direct_workspace_mutations: DirectWorkspaceMutationCollector
    workspace_mutation_owner: InvocationWorkspaceMutationOwner | None
    workspace_receipt_artifact_store: Any
    workspace_receipt_artifact_unavailable_detail: str


class ToolInvocationResources:
    """Create scoped resource handles and retain abandoned secret resolutions."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        image_decode_policy: ImageDecodePolicy | None,
        knowledge_publication_scope: KnowledgePublicationScope,
        browser_control_service: BrowserControlService | None,
        clock: Callable[[], datetime],
    ) -> None:
        self._session_store = session_store
        self._image_decode_policy = image_decode_policy
        self._knowledge_publication_scope = knowledge_publication_scope
        self._browser_control_service = browser_control_service
        self._clock = clock
        self._detached_secret_resolutions: set[asyncio.Task[Any]] = set()

    def detached_environment_work(self) -> set[asyncio.Future[Any]]:
        return set(self._detached_secret_resolutions)

    def _retain_detached_secret_resolution(self, task: asyncio.Task[Any]) -> None:
        # The resolver consumes its own outcome; this only keeps it owned.
        self._detached_secret_resolutions.add(task)
        task.add_done_callback(self._detached_secret_resolutions.discard)

    def bind(
        self,
        call: ToolInvocationCall,
        *,
        ctx_metadata: dict[str, Any],
        invocation_secret_scope: invocation_secrets.InvocationSecretTracker,
        proxy_authorizations: list[invocation_secrets.ProxyAuthorizationRecord],
        redactor_provider: Callable[[], SecretRedactor],
        observe_runner_execution: Callable[
            [Literal["started", "completed"], dict[str, Any], int], Awaitable[None]
        ],
        persist_resolved_secret_projection: Callable[[InvocationRedactorSnapshot], Awaitable[None]],
    ) -> BoundInvocationResources:
        session = call.session
        registered_agent = call.registered_agent
        registered_environment = call.registered_environment
        registered_tool = call.registered_tool
        effective_tool_call = call.effective_tool_call
        tool_round_identity = call.tool_round_identity
        execution_profile = call.execution_profile
        invocation_context = call.invocation_context
        task_id = call.task_id
        budget_limits = call.budget_limits
        tool_exposure = call.tool_exposure
        environment_name = call.environment_name
        idempotency_key = call.idempotency_key

        raw_workspace = _workspace(registered_environment)
        raw_artifact_store = _artifact_store(registered_environment)
        from cayu.resource_access import current_binding

        if current_binding() is not None:
            for store, capability in (
                (raw_artifact_store, "artifact_access_version"),
                (_knowledge_store(registered_environment), "resource_knowledge_access_version"),
            ):
                if store is not None and type(store).__dict__.get(capability) != 1:
                    raise NotImplementedError("Environment store cannot enforce resource access.")
        direct_workspace_mutations = DirectWorkspaceMutationCollector()
        workspace_mutation_owner = (
            InvocationWorkspaceMutationOwner(
                on_settlement_unproven=(
                    registered_environment.workspace_mutation_fence.fail_closed
                ),
            )
            if (
                registered_environment is not None
                and registered_tool.workspace_mutation
                and raw_workspace is not None
            )
            else None
        )
        (
            workspace_receipt_artifact_store,
            workspace_receipt_artifact_unavailable_detail,
        ) = _workspace_receipt_artifact_store(registered_environment)
        from cayu.tools.browser_session import BrowserSessionTool, _RunnerBrowserSessionBackend

        allow_private_browser_profile_io = (
            type(registered_tool.tool) is BrowserSessionTool
            and registered_tool.tool.browser_profile is not None
        )
        tool_context = ToolContext(
            image_decode_limits=(
                self._image_decode_policy.as_dict() if self._image_decode_policy else None
            ),
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            causal_budget_id=session.causal_budget_id,
            workspace_id=_workspace_id(registered_environment),
            artifact_store_id=_artifact_store_id(registered_environment),
            idempotency_key=idempotency_key,
            workspace=invocation_workspace_handle(
                raw_workspace,
                redactor_snapshot_provider=invocation_secret_scope.snapshot,
                capture_observer=invocation_secret_scope.record_ambiguous_output_capture,
                mutation_owner=workspace_mutation_owner,
                direct_mutation_observer=direct_workspace_mutations.record,
            ),
            artifact_store=invocation_artifact_store_handle(
                raw_artifact_store,
                redactor_snapshot_provider=invocation_secret_scope.snapshot,
                capture_observer=invocation_secret_scope.record_ambiguous_output_capture,
            ),
            runner=invocation_runner_handle(
                _runner(registered_environment),
                redactor_snapshot_provider=invocation_secret_scope.snapshot,
                ambiguous_capture_observer=(
                    invocation_secret_scope.record_ambiguous_output_capture
                ),
                mutation_owner=workspace_mutation_owner,
                execution_observer=observe_runner_execution,
                publish_execution_arguments=registered_tool.publish_arguments,
                allow_private_browser_profile_io=allow_private_browser_profile_io,
                allow_private_browser_control_io=(
                    type(registered_tool.tool) is BrowserSessionTool
                    and invocation_context is not None
                    and registered_environment is not None
                    and registered_environment.live_allocation_fingerprint is not None
                ),
            ),
            invocation_secret_redactor=redactor_provider,
            invocation_secret_snapshot_provider=invocation_secret_scope.snapshot,
            invocation_secret_capture_observer=(
                invocation_secret_scope.record_ambiguous_output_capture
            ),
            vault=invocation_secrets.vault_for_environment(
                registered_environment,
                tracker=invocation_secret_scope,
                on_redactor_change=persist_resolved_secret_projection,
                retain_abandoned=self._retain_detached_secret_resolution,
            ),
            proxy=invocation_secrets.proxy_for_environment(
                registered_environment,
                tracker=invocation_secret_scope,
                on_authorize=proxy_authorizations.append,
                on_redactor_change=persist_resolved_secret_projection,
                retain_abandoned=self._retain_detached_secret_resolution,
            ),
            knowledge_store=_knowledge_store(registered_environment),
            knowledge_access_scope=_knowledge_access_scope(registered_environment),
            mcp_servers=_mcp_servers(registered_environment),
            metadata=ctx_metadata,
        )
        tool_context._bind_runtime_causal_budget_limits(
            tuple(
                _copy_budget_limit_definition(limit)
                for limit in budget_limits
                if limit.scope == "causal" and limit.key == session.causal_budget_id
            )
            if registered_tool.child_session_recovery is not None
            else ()
        )
        tool_context._bind_runtime_resource_authorities(
            workspace=raw_workspace,
            artifact_store=raw_artifact_store,
        )
        tool_context._bind_runtime_knowledge_publication_scope(self._knowledge_publication_scope)
        if effective_tool_call.name == SEARCH_TOOLS_NAME:
            if registered_agent.tool_discovery_mode is None:
                raise RuntimeError("search_tools execution requires enabled tool discovery.")
            _bind_runtime_tool_discovery_authority(
                tool_context,
                generation_id=tool_discovery_generation_id(
                    session_id=session.id,
                    root_invocation_id=session.invocation.root_invocation_id,
                ),
                catalogue=registered_agent.tool_catalogue,
                ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
                directly_exposed_names=(
                    registered_agent.tools if tool_exposure is None else tool_exposure.tool_names
                ),
                model_step_id=tool_round_identity.model_step_id,
                created_at=self._clock(),
            )
        if execution_profile is not None:

            async def load_durable_operation(storage_key: str) -> dict[str, Any] | None:
                return await self._session_store.load_session_operation(
                    session.id,
                    storage_key,
                )

            async def authorize_shared_artifact(
                reference: dict[str, Any],
                policy_fingerprint: str,
                observed_at: str,
            ) -> dict[str, Any]:
                from cayu.tools.shared_artifacts import authorize_shared_artifact_materialization

                return await authorize_shared_artifact_materialization(
                    session_store=self._session_store,
                    caller_session_id=session.id,
                    caller_session_instance_id=session.instance_id,
                    reference=reference,
                    policy_fingerprint=policy_fingerprint,
                    observed_at=observed_at,
                )

            async def compare_and_set_durable_operation(
                storage_key: str,
                expected: dict[str, Any] | None,
                desired: dict[str, Any],
                secondary_records: Mapping[str, dict[str, Any]],
            ) -> dict[str, Any]:
                expected_copy = (
                    None
                    if expected is None
                    else copy_durable_json_object(expected, "durable_tool_operation.expected")
                )
                desired_copy = copy_durable_json_object(
                    desired,
                    "durable_tool_operation.desired",
                )
                secondary_copy = {
                    key: copy_durable_json_object(
                        value,
                        f"durable_tool_operation.secondary[{key!r}]",
                    )
                    for key, value in secondary_records.items()
                }
                if storage_key in secondary_copy:
                    raise ValueError("A durable tool operation cannot duplicate its primary key.")
                control_mutation = None
                if type(
                    registered_tool.tool
                ) is BrowserSessionTool and effective_tool_call.arguments.get("operation") in {
                    "observe",
                    "close",
                }:
                    with browser_control_checkpoint_read_scope(session.id):
                        control_checkpoint = await self._session_store.load_checkpoint(session.id)
                    control_mutation = browser_terminal_checkpoint_mutation(
                        control_checkpoint,
                        session_id=session.id,
                        operation_records={storage_key: desired_copy, **secondary_copy},
                        operation_name=effective_tool_call.arguments.get("operation"),
                    )

                def publish(
                    current_session: Session,
                    checkpoint: dict[str, Any] | None,
                    current: dict[str, Any] | None,
                ) -> SessionOperationPublication:
                    if (
                        current_session.id != session.id
                        or current_session.run_epoch != session.run_epoch
                    ):
                        raise SessionRunFenced(
                            "Durable tool operation lost its parent run authority."
                        )
                    if current != expected_copy:
                        raise DurableToolOperationConflict(
                            "Durable tool operation changed before publication."
                        )
                    records = {storage_key: desired_copy, **secondary_copy}
                    if type(registered_tool.tool) is BrowserSessionTool:
                        validate_browser_model_publication(
                            checkpoint,
                            session=current_session,
                            operation_records=records,
                            operation_name=effective_tool_call.arguments.get("operation"),
                        )
                    published_checkpoint = {} if checkpoint is None else dict(checkpoint)
                    if control_mutation is not None:
                        from cayu.sessions.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY

                        published_checkpoint[BROWSER_CONTROLS_CHECKPOINT_KEY] = (
                            control_mutation.desired.model_dump(mode="json")
                        )
                    return SessionOperationPublication(
                        checkpoint=published_checkpoint,
                        operation_records=records,
                    )

                with (
                    browser_control_checkpoint_read_scope(session.id),
                    browser_control_checkpoint_mutation_scope(control_mutation)
                    if control_mutation is not None
                    else nullcontext(),
                ):
                    publication = asyncio.create_task(
                        self._session_store.publish_session_operation(
                            session.id,
                            idempotency_key=storage_key,
                            operation_transform=publish,
                            events=[],
                            expected_statuses={SessionStatus.RUNNING},
                            expected_run_epoch=session.run_epoch,
                        )
                    )
                outcome = await await_shielded_task_outcome(publication)
                publication_error = outcome.error
                if publication_error is not None:
                    persisted = await self._session_store.load_session_operation(
                        session.id,
                        storage_key,
                    )
                    if persisted != desired_copy:
                        if outcome.cancellation is not None:
                            outcome.cancellation.add_note(
                                "Durable tool operation publication also failed: "
                                f"{type(publication_error).__name__}."
                            )
                            restore_task_cancellation_requests(
                                outcome.cancellation_requests_consumed,
                                cancellation=outcome.cancellation,
                            )
                            raise outcome.cancellation from publication_error
                        raise publication_error
                    for key, value in secondary_copy.items():
                        if (
                            await self._session_store.load_session_operation(
                                session.id,
                                key,
                            )
                            != value
                        ):
                            inconsistency = RuntimeError(
                                "Durable tool operation acknowledgement is inconsistent."
                            )
                            if outcome.cancellation is not None:
                                restore_task_cancellation_requests(
                                    outcome.cancellation_requests_consumed,
                                    cancellation=outcome.cancellation,
                                )
                                raise outcome.cancellation from inconsistency
                            raise inconsistency from publication_error
                if outcome.cancellation is not None:
                    restore_task_cancellation_requests(
                        outcome.cancellation_requests_consumed,
                        cancellation=outcome.cancellation,
                    )
                    raise outcome.cancellation
                return copy_durable_json_object(
                    desired_copy,
                    "durable_tool_operation.result",
                )

            def seal_durable_output(record: dict[str, Any]) -> dict[str, Any]:
                snapshot = invocation_secret_scope.seal_for_publication()
                if snapshot.unsafe_output or snapshot.secret_scope_incomplete:
                    raise RuntimeError(
                        "Durable tool output requires a complete invocation secret scope."
                    )
                redacted = snapshot.redactor.redact_json_values(
                    copy_durable_json_object(record, "durable_tool_output")
                )
                if type(redacted) is not dict:
                    raise TypeError("Durable tool output must remain an object after redaction.")
                return redacted

            bootstrap_browser_control = None
            browser_control_admission = None
            if (
                self._browser_control_service is not None
                and type(registered_tool.tool) is BrowserSessionTool
                and invocation_context is not None
                and environment_name is not None
                and registered_environment is not None
                and registered_environment.live_allocation_fingerprint is not None
            ):
                browser_service = self._browser_control_service
                browser_backend = registered_tool.tool._backend
                browser_arguments = copy_durable_json_object(
                    effective_tool_call.arguments, "browser_control.arguments"
                )

                async def bootstrap_browser_control(browser_session_id: str) -> None:
                    if type(browser_backend) is not _RunnerBrowserSessionBackend:
                        raise RuntimeError("Browser control requires the built-in runner backend.")
                    await browser_service.bootstrap(
                        tool_context,
                        backend=browser_backend,
                        browser_session_id=browser_session_id,
                        arguments=browser_arguments,
                    )

                async def browser_control_admission(browser_session_id: str, operation_name: str):
                    allocation = BrowserGuestBootstrap.allocation_for_invocation(
                        tool_context,
                        purpose=browser_service.purpose,
                        browser_session_id=browser_session_id,
                        arguments=browser_arguments,
                    )
                    with browser_control_checkpoint_read_scope(session.id):
                        checkpoint = await self._session_store.load_checkpoint(session.id)
                    if checkpoint is not None and "browser_controls" in checkpoint:
                        from cayu.tools.browser_control import (
                            BrowserControlCheckpoint,
                            BrowserControlIdentity,
                            rebound_browser_control_successor,
                        )

                        controls = BrowserControlCheckpoint.model_validate(
                            checkpoint["browser_controls"]
                        )
                        prior = next(
                            (
                                record
                                for record in controls.records
                                if record.identity.browser_session_id == browser_session_id
                            ),
                            None,
                        )
                        if prior is not None and prior.identity.run_epoch != allocation.run_epoch:
                            # Validate the strict view-only transition before issuing a
                            # private capability. The owned runner/native guest must
                            # settle old transport and acknowledge the new fence first.
                            rebound_browser_control_successor(
                                prior,
                                BrowserControlIdentity(
                                    **allocation.model_dump(),
                                    worker_instance_id=prior.identity.worker_instance_id,
                                ),
                            )
                            await bootstrap_browser_control(browser_session_id)
                            with browser_control_checkpoint_read_scope(session.id):
                                checkpoint = await self._session_store.load_checkpoint(session.id)
                    return browser_model_control_admission(
                        checkpoint, allocation=allocation, operation_name=operation_name
                    )

            _bind_runtime_tool_invocation_authority(
                tool_context,
                parent_task_id=task_id,
                parent_run_epoch=session.run_epoch,
                model_step_id=tool_round_identity.model_step_id,
                model_attempt_id=tool_round_identity.model_attempt_id,
                tool_round_id=tool_round_identity.tool_round_id,
                tool_call_id=effective_tool_call.id,
                tool_name=effective_tool_call.name,
                idempotency_key=idempotency_key,
                effective_arguments=effective_tool_call.arguments,
                execution_profile_fingerprint=execution_profile.fingerprint,
                environment_allocation_fingerprint=(
                    None
                    if registered_environment is None
                    else registered_environment.live_allocation_fingerprint
                ),
                environment_allocation_generation=(
                    None
                    if registered_environment is None
                    else registered_environment.allocation_generation
                ),
                current_session_lineage={
                    "session_id": session.id,
                    "session_instance_id": session.instance_id,
                    "parent_session_id": session.parent_session_id,
                    "causal_budget_id": session.causal_budget_id,
                    "invocation": session.invocation.model_dump(mode="json"),
                },
                load_durable_operation=load_durable_operation,
                authorize_shared_artifact=authorize_shared_artifact,
                compare_and_set_durable_operation=(compare_and_set_durable_operation),
                seal_durable_output=seal_durable_output,
                secret_publication_sealer=invocation_secret_scope.seal_for_publication,
                bootstrap_browser_control=bootstrap_browser_control,
                browser_control_admission=browser_control_admission,
                browser_allocation=(
                    _RuntimeBrowserAllocationAuthority(
                        session_id=session.id,
                        session_instance_id=session.instance_id,
                        run_epoch=session.run_epoch,
                        interaction_id=invocation_context.active_profile.interaction_id,
                        execution_profile_fingerprint=execution_profile.fingerprint,
                        environment_name=environment_name,
                        allocation_fingerprint=registered_environment.live_allocation_fingerprint,
                        profile_checkpoint_policy=(
                            "unavailable"
                            if registered_tool.tool.browser_profile is None
                            else registered_tool.tool.browser_profile.checkpoint_policy.value
                        ),
                    )
                    if type(registered_tool.tool) is BrowserSessionTool
                    and invocation_context is not None
                    and environment_name is not None
                    and registered_environment is not None
                    and registered_environment.live_allocation_fingerprint is not None
                    else None
                ),
            )
        return BoundInvocationResources(
            context=tool_context,
            raw_workspace=raw_workspace,
            direct_workspace_mutations=direct_workspace_mutations,
            workspace_mutation_owner=workspace_mutation_owner,
            workspace_receipt_artifact_store=workspace_receipt_artifact_store,
            workspace_receipt_artifact_unavailable_detail=workspace_receipt_artifact_unavailable_detail,
        )
