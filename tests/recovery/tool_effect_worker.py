"""Real process-loss barriers using the counterfactual example's public adapter."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

from examples.counterfactual_approval.deployment import (
    DeploymentState,
    DeployServiceTool,
    deployment_reconciliation,
)
from worker_harness import _append_json_line, _public_authority_alias_codec, _write_json_atomic

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    IncompleteSessionRecoveryRequest,
    Message,
    ResumeRequest,
    RunRequest,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolEffectReconciliationRequest,
)
from cayu.core import ToolResultPart
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.storage.sqlite import SQLiteSessionStore


async def run_tool_effect_worker(config):
    phase = config["crash_phase"] if config["action"] == "start" else None
    now = datetime.now(UTC) + timedelta(seconds=0 if phase is not None else 600)
    session_id = config["session_id"]
    state = DeploymentState(
        Path(config["external_path"]),
        lose_acknowledgement_once=phase in {"receipt_transaction", "before_continuation"},
    )
    replay_request = None

    def announce(**values):
        _write_json_atomic(
            Path(config["phase_path"]),
            {
                "phase": phase,
                "replay_request": replay_request,
                **values,
            },
        )

    async def pause(**values):
        announce(**values)
        await asyncio.Future()

    class BarrierStore(SQLiteSessionStore):
        invocation_lifecycle_command_version = 1

        async def publish_session_operation(self, target_session_id, **kwargs):
            receipt_batch = any(
                event.type.value == "tool.effect.receipt.validated"
                for event in kwargs.get("events", ())
            )
            if phase == "receipt_transaction" and receipt_batch:

                def before_commit():
                    # Guard executes after the actual SQL record/event inserts,
                    # inside BEGIN IMMEDIATE, before connection.commit().
                    announce()
                    threading.Event().wait()

                return await super().publish_session_operation_guarded(
                    target_session_id,
                    commit_guard=before_commit,
                    **kwargs,
                )
            result = await super().publish_session_operation(target_session_id, **kwargs)
            key = kwargs["idempotency_key"]
            if key.startswith("tool-effect:v1:"):
                record = await self.load_session_operation(target_session_id, key)
                if phase == "after_preparation" and record["state"] == "prepared":
                    await pause(idempotency_key=record["intent"]["idempotency_key"])
                if phase == "before_invocation" and record["state"] == "executing":
                    await pause(idempotency_key=record["intent"]["idempotency_key"])
                if phase == "before_continuation" and receipt_batch:
                    await pause()
            return result

    class Deployment(DeployServiceTool):
        async def run(self, ctx, args):
            if phase == "during_execution":
                rejected = state.begin(
                    **args,
                    idempotency_key=ctx.idempotency_key,
                    tool_call_id=ctx.metadata["tool_call_id"],
                )
                assert rejected is None
                await pause(idempotency_key=ctx.idempotency_key)
            result = await super().run(ctx, args)
            if phase == "after_external_completion":
                await pause(idempotency_key=ctx.idempotency_key)
            return result

    class Provider(ModelProvider):
        name = "tool-effect-recovery"

        @property
        def execution_profile_identity(self):
            return ExecutionProfileBehaviorIdentity(
                name="tests:tool-effect-recovery",
                behavior_version="1",
                implementation_version="1",
            )

        async def stream(self, request):
            terminal = any(
                isinstance(part, ToolResultPart)
                for message in request.messages
                for part in message.content
            )
            _append_json_line(Path(config["marker_path"]), {"model_continuation": terminal})
            if terminal:
                yield ModelStreamEvent.text_delta("Deployment reconciled.")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return
            if config["action"] != "start":
                raise AssertionError("Recovery invoked the model without a terminal tool result.")
            yield ModelStreamEvent.tool_call(
                id="deploy-call",
                name="deploy_service",
                arguments={"service": "payments", "release": "2026.07.11", "expected_version": 7},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    store = BarrierStore(
        config["backend"]["session_path"],
        ownership_clock=lambda: now,
        public_authority_alias_codec=_public_authority_alias_codec(),
    )
    try:
        app = CayuApp(session_store=store, clock=lambda: now, enable_logging=False)
        app.register_provider(Provider(), default=True)
        app.register_agent(
            AgentSpec(name="effect-agent", model="scripted"),
            tools=[Deployment(state)],
            tool_policy=AlwaysRequireApprovalToolPolicy(tools=["deploy_service"])
            if config["approval_gate"]
            else None,
            tool_effect_reconcilers={"deploy_service": deployment_reconciliation(state)},
        )
        if config["action"] == "start":
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="effect-agent",
                        messages=[Message.text("user", "deploy")],
                    )
                )
            ]
            if config["approval_gate"]:
                approval = next(e for e in events if e.type.value == "tool.call.approval_requested")
                events = [
                    event
                    async for event in app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id=session_id,
                            approval_id=approval.payload["approval"]["approval_id"],
                            tool_round_id=approval.payload["tool_round_id"],
                            tool_call_id=approval.payload["tool_call_id"],
                            decision=ToolApprovalDecision.APPROVE,
                        )
                    )
                ]
            assert events[-1].type.value == "session.interrupted"
        else:
            # Retire the dead invocation through its existing recovery owner
            # even when receipt selection committed. Exact receipt replay is
            # not authority to bypass an incomplete invocation's terminal fence.
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_for_seconds=0,
                )
            )

        if config["action"] == "recover" and config["crash_phase"] == "after_preparation":
            before = await store.load_events(session_id)
            if config["approval_gate"]:
                approval = next(e for e in before if e.type.value == "tool.call.approval_requested")
                stream = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.payload["approval"]["approval_id"],
                        tool_round_id=approval.payload["tool_round_id"],
                        tool_call_id=approval.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            else:
                stream = app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "Continue after recovery.")],
                    )
                )
            events = [e async for e in stream]
            before_replay = await store.load_events(session_id)
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_for_seconds=0,
                )
            )
            assert await store.load_events(session_id) == before_replay
            session = await store.load(session_id)
            return {
                "status": session.status.value,
                "events": [e.type.value for e in events],
                "mutation_count": state.mutation_count,
                "receipt_count": len(state.receipts),
            }

        if config.get("replay_request") is not None:
            request = ToolEffectReconciliationRequest.model_validate(config["replay_request"])
        else:
            started = next(
                e
                for e in await store.load_events(session_id)
                if e.type.value == "tool.call.started"
            )
            current_session = await store.load(session_id)
            effect = await ToolEffectStateOwner(store).resolve_call(
                current_session,
                tool_round_id=started.payload["tool_round_id"],
                tool_call_id=started.payload["tool_call_id"],
            )
            assert effect is not None
            assert effect.state == "outcome_unknown", effect.state
            target = await app.inspect_tool_effect(
                session_id,
                tool_round_id=started.payload["tool_round_id"],
                tool_call_id=started.payload["tool_call_id"],
            )
            request = ToolEffectReconciliationRequest(**target.model_dump(), lookup=True)
        replay_request = request.model_dump(mode="json")
        events = [event async for event in app.reconcile_tool_effect(request)]
        session = await store.load(session_id)
        if config.get("replay_request") is not None and session.status.value == "interrupted":
            # Ordinary incomplete recovery can already publish the selected
            # tool round without starting the model. Receipt replay must stay
            # read-only once consumed; normal resume owns the remaining loop.
            events.extend(
                [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id=session_id,
                            messages=[
                                Message.text("user", "Continue from the reconciled deployment.")
                            ],
                        )
                    )
                ]
            )
            session = await store.load(session_id)
        if session.status.value == "completed":
            before = await store.load_events(session_id)
            _ = [event async for event in app.reconcile_tool_effect(request)]
            assert await store.load_events(session_id) == before
        return {
            "status": session.status.value,
            "events": [e.type.value for e in events],
            "mutation_count": state.mutation_count,
            "receipt_count": len(state.receipts),
        }
    finally:
        await store.close()
