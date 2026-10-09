"""Tool dispatch under frozen authority and durable external-effect intent."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable

from cayu._validation import (
    canonical_durable_json_bytes,
)
from cayu.environments.deferred import materialization_trigger
from cayu.events import (
    Event,
)
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime._auxiliary_inference import AuxiliaryInferenceOwner
from cayu.runtime._auxiliary_invocation import AuxiliaryInvocationPolicy
from cayu.runtime._environment_exposure import (
    refresh_and_require_environment_exposed,
)
from cayu.runtime._tool_effect_state import (
    ToolEffectRecord,
    ToolEffectStateOwner,
)
from cayu.runtime._tool_invocation.context import (
    _targeted_tool_invocation_payload,
)
from cayu.runtime._tool_invocation.resources import ToolInvocationCall
from cayu.sessions._tool_effect_intent import ToolEffectIntent
from cayu.sessions.base import (
    SessionStore,
)
from cayu.tools.base import (
    ToolContext,
    ToolEffect,
    ToolResult,
)


class ToolInvocationDispatch:
    """One dispatch and its effect record, retained even if admission refuses it."""

    def __init__(
        self,
        *,
        call: ToolInvocationCall,
        context: ToolContext,
        secret_scope: invocation_secrets.InvocationSecretTracker,
        session_store: SessionStore,
        auxiliary_inference: AuxiliaryInferenceOwner,
        auxiliary_policy: AuxiliaryInvocationPolicy | None,
        tool_timeout_seconds: float | None,
        strict_common_budget_admission: bool,
        runner_events: list[Event],
        settle_workspace: Callable[[], Awaitable[tuple[Event, ...]]] | None,
        on_invoke: Callable[[], None] | None = None,
    ) -> None:
        self._call = call
        self._on_invoke = on_invoke
        self._context = context
        self._secret_scope = secret_scope
        self._session_store = session_store
        self._auxiliary_inference = auxiliary_inference
        self._auxiliary_policy = auxiliary_policy
        self._tool_timeout_seconds = tool_timeout_seconds
        self._strict_common_budget_admission = strict_common_budget_admission
        self._runner_events = runner_events
        self._settle_workspace = settle_workspace
        self.effect: ToolEffectRecord | None = None
        self.events: list[Event] = []

    async def run(self) -> tool_execution.ToolExecutionOutcome:
        session = self._call.session
        registered_agent = self._call.registered_agent
        registered_environment = self._call.registered_environment
        registered_tool = self._call.registered_tool
        effective_tool_call = self._call.effective_tool_call
        tool_round_identity = self._call.tool_round_identity
        execution_profile = self._call.execution_profile
        invocation_context = self._call.invocation_context
        budget_limits = self._call.budget_limits
        environment_name = self._call.environment_name
        idempotency_key = self._call.idempotency_key
        approval_id = self._call.approval_id
        input_id = self._call.input_id

        async def require_live_environment_exposure() -> None:
            if registered_environment is None:
                return
            assert invocation_context is not None
            assert execution_profile is not None
            await refresh_and_require_environment_exposed(
                registered_environment,
                session=session,
                invocation_context=invocation_context,
                registered_agent=registered_agent,
                execution_profile=execution_profile,
                redactor=self._secret_scope.redactor,
            )

        # Refuse missing/stale environment authority before consuming a
        # protected effect. The exact dispatch seam below still performs
        # main's independent freshness check after durable preparation.
        await require_live_environment_exposure()
        from cayu.resource_access import require_dispatch

        await require_dispatch()
        inference_scope = None
        if registered_tool.auxiliary_inference is not None:
            if invocation_context is None or self._auxiliary_policy is None:
                raise RuntimeError("Auxiliary inference requires frozen invocation authority.")
            inference_scope = self._auxiliary_inference.create_scope(
                session=session,
                invocation=invocation_context,
                policy=self._auxiliary_policy,
                registered_tool=registered_tool,
                parent=tool_round_identity,
                tool_call_id=effective_tool_call.id,
                idempotency_key=idempotency_key,
                budget_limits=budget_limits,
                budget_binding=self._auxiliary_policy.budget_binding,
                redactor=lambda: self._secret_scope.redactor,
                refresh=require_live_environment_exposure,
                observe_event=self.events.append,
            )
            self._context._bind_runtime_inference(inference_scope)
        if registered_tool.effect is ToolEffect.EXTERNAL:
            if self._strict_common_budget_admission:
                raise RuntimeError(
                    "Opaque external tool adapters are not qualified for common-root "
                    "budget admission and are refused before dispatch."
                )
            if invocation_context is None or execution_profile is None:
                raise RuntimeError("External tool dispatch requires frozen invocation authority.")
            targeted_material = _targeted_tool_invocation_payload(effective_tool_call)
            effect_intent = ToolEffectIntent(
                session_id=session.id,
                session_instance_id=session.instance_id,
                source_run_epoch=session.run_epoch,
                interaction_id=invocation_context.active_profile.interaction_id,
                model_step_id=tool_round_identity.model_step_id,
                model_attempt_id=tool_round_identity.model_attempt_id,
                tool_round_id=tool_round_identity.tool_round_id,
                agent_name=registered_agent.spec.name,
                tool_name=effective_tool_call.name,
                tool_call_id=effective_tool_call.id,
                idempotency_key=idempotency_key,
                execution_profile_fingerprint=execution_profile.fingerprint,
                schema_digest=hashlib.sha256(
                    canonical_durable_json_bytes(
                        registered_tool.schema,
                        "effect_schema",
                    )
                ).hexdigest(),
                arguments_digest=hashlib.sha256(
                    canonical_durable_json_bytes(
                        effective_tool_call.arguments,
                        "effect_arguments",
                    )
                ).hexdigest(),
                approval_id=approval_id,
                pause_id=input_id,
                environment_name=environment_name,
                allocation_fingerprint=(
                    None
                    if registered_environment is None
                    else registered_environment.live_allocation_fingerprint
                ),
                reconciler_fingerprint=(
                    None
                    if registered_tool.effect_reconciler is None
                    else registered_tool.effect_reconciler.fingerprint
                ),
                targeted_invocation_digest=(
                    None
                    if not targeted_material
                    else hashlib.sha256(
                        canonical_durable_json_bytes(
                            targeted_material,
                            "effect_targeted_invocation",
                        )
                    ).hexdigest()
                ),
            )
            self.effect = await ToolEffectStateOwner(self._session_store).begin(
                effect_intent,
                run_epoch=session.run_epoch,
                child_recovery_arguments=(
                    effective_tool_call.arguments
                    if registered_tool.child_session_recovery is not None
                    else None
                ),
            )

        async def reconcile_child_result() -> ToolResult | None:
            if self.effect is None or registered_tool.child_session_recovery is None:
                return None
            from cayu.runtime._foreground_child_wait import (
                ForegroundChildActionRequired,
                observe_foreground_child_wait,
                project_current_foreground_child_result,
                retain_foreground_child_wait,
            )

            child_wait = await observe_foreground_child_wait(
                self._session_store,
                parent=session,
                intent=self.effect.intent,
                matcher=registered_tool.child_session_recovery,
                arguments=effective_tool_call.arguments,
            )
            if child_wait is not None:
                if self._settle_workspace is not None:
                    await self._settle_workspace()
                await retain_foreground_child_wait(
                    self._session_store,
                    parent=session,
                    effect=self.effect,
                    wait=child_wait,
                )
                raise ForegroundChildActionRequired(child_wait)
            return await project_current_foreground_child_result(
                self._session_store,
                parent=session,
                intent=self.effect.intent,
                matcher=registered_tool.child_session_recovery,
                arguments=effective_tool_call.arguments,
            )

        async def require_resource_dispatch():
            await require_live_environment_exposure()
            await require_dispatch()

        # A deferred environment attributes its materialization to this call.
        with materialization_trigger(
            tool_call_id=effective_tool_call.id,
            tool_name=effective_tool_call.name,
            # Yielded with the call's runner evidence before its terminal.
            event_sink=self._runner_events.append,
        ):
            execution_outcome = await tool_execution.run_tool(
                tool=registered_tool.tool,
                effect=registered_tool.effect,
                ctx=self._context,
                arguments=effective_tool_call.arguments,
                redactor=lambda: self._secret_scope.redactor,
                registered_schema=registered_tool.schema,
                registered_execution_contract=registered_tool.execution_contract,
                finalize_publication=self._secret_scope.seal_for_publication,
                timeout_seconds=self._tool_timeout_seconds,
                before_dispatch=require_resource_dispatch,
                reconcile_result=reconcile_child_result,
                inference_scope=inference_scope,
                on_invoke=self._on_invoke,
            )

        return execution_outcome
