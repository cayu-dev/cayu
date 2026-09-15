"""Explicit recovery policy requests cannot replace or bypass frozen authority."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_model_failover_recovery import _RecoveryProvider

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    ExecutionProfileBehaviorIdentity,
    Message,
    ModelFailoverPolicy,
    ModelTarget,
    ResumeRequest,
    RunRequest,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.retry_policy import RetryPolicy
from cayu.sessions.base import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.policy import AllowAllToolPolicy


class _PendingProvider(_RecoveryProvider):
    def __init__(self, name, *, first=False):
        super().__init__(name)
        self.first = first

    async def stream(self, request):
        if self.name == "backup" and self.first:
            self.first = False
            self.requests.append(request)
            yield ModelStreamEvent.tool_call(id="read-call", name="read", arguments={})
            yield ModelStreamEvent.completed()
            return
        async for event in super().stream(request):
            yield event


class _ReadTool(Tool):
    spec = ToolSpec(
        name="read",
        description="Read an in-memory value.",
        input_schema={"type": "object", "properties": {}},
        effect=ToolEffect.NONE,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:failover-recovery-policy:read",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self):
        self.calls = 0

    async def run(self, ctx, args):
        self.calls += 1
        return ToolResult(content="value")


def _policy(*, cap=20, model="large"):
    return ModelFailoverPolicy(
        fallbacks=(ModelTarget(provider_name="backup", model=model),),
        max_total_attempts=cap,
    )


def _app(store, *, first=False):
    primary = _PendingProvider("primary")
    backup = _PendingProvider("backup", first=first)
    tool = _ReadTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(primary, default=True)
    app.register_provider(backup)
    app.register_agent(
        AgentSpec(name="agent", model="small"), tools=[tool], tool_policy=AllowAllToolPolicy()
    )
    return app, primary, backup, tool


class _HoldPromotion:
    invocation_lifecycle_command_version = 1
    hold_promotion = False

    async def promote_model_completion_stage(self, *args, **kwargs):
        if self.hold_promotion:
            raise RuntimeError("test holds committed model completion before promotion")
        return await super().promote_model_completion_stage(*args, **kwargs)


class _MemoryStore(_HoldPromotion, InMemorySessionStore):
    model_failover_stage_version = 1
    invocation_lifecycle_command_version = 1


class _SQLiteStore(_HoldPromotion, SQLiteSessionStore):
    model_failover_stage_version = 1
    invocation_lifecycle_command_version = 1


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "boundary,policy_kind",
    [("tool", kind) for kind in ("omitted", "identical", "cap", "chain")]
    + [("model", kind) for kind in ("cap", "chain")],
)
def test_public_pending_round_resume_checks_explicit_failover_policy(
    tmp_path, backend, boundary, policy_kind
):
    async def run():
        path = tmp_path / "pending.sqlite"
        store = _MemoryStore() if backend == "memory" else _SQLiteStore(path)
        retry = RetryPolicy(max_attempts=1, initial_delay_s=0)
        try:
            app, primary, backup, tool = _app(store, first=boundary == "tool")
            store.hold_promotion = boundary == "model"
            stream = app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="pending-fallback",
                    messages=[Message.text("user", "read")],
                    retry_policy=retry,
                    failover=_policy(),
                )
            )

            async def consume_initial():
                try:
                    async for event in stream:
                        if boundary == "tool" and event.type is EventType.MODEL_COMPLETED:
                            break
                finally:
                    await stream.aclose()

            initial_task = asyncio.create_task(consume_initial())
            await initial_task
            assert len(primary.requests) == len(backup.requests) == 1
            assert tool.calls == 0
            checkpoint = await store.load_checkpoint("pending-fallback")
            if boundary == "tool":
                assert checkpoint.get("pending_tool_round") is not None
            else:
                active = await store.load_active_model_completion_stage("pending-fallback")
                assert active is not None and active.stage.state == "completed"
            store.hold_promotion = False
            if isinstance(store, SQLiteSessionStore):
                await store.close()
                store = _SQLiteStore(path)
            app, primary, backup, tool = _app(store)
            before_session = await store.load("pending-fallback")
            before_checkpoint = await store.load_checkpoint("pending-fallback")
            before_events = await store.load_events("pending-fallback")
            before_transcript = await store.load_transcript("pending-fallback")
            before_active = await store.load_active_model_completion_stage("pending-fallback")
            policy = {
                "omitted": None,
                "identical": _policy(),
                "cap": _policy(cap=1),
                "chain": _policy(model="different-model"),
            }[policy_kind]

            async def resume():
                return [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="pending-fallback",
                            messages=[Message.text("user", "continue")],
                            retry_policy=retry,
                            failover=policy,
                        )
                    )
                ]

            if policy_kind in {"cap", "chain"}:
                with pytest.raises(RuntimeError, match="failover policy cannot change"):
                    await resume()
                assert await store.load("pending-fallback") == before_session
                assert await store.load_checkpoint("pending-fallback") == before_checkpoint
                assert await store.load_events("pending-fallback") == before_events
                assert await store.load_transcript("pending-fallback") == before_transcript
                assert (
                    await store.load_active_model_completion_stage("pending-fallback")
                    == before_active
                )
                assert tool.calls == 0
                assert not primary.requests and not backup.requests
            else:
                events = await resume()
                # Closing before tool dispatch quarantines the retained arguments.
                # Matching policy permits recovery, not bypassing that approval.
                assert events[-1].type is EventType.SESSION_INTERRUPTED
                assert events[-1].payload["interruption_type"] == "tool_approval_required"
                approval = events[-1].payload["approval"]
                assert not primary.requests and not backup.requests and tool.calls == 0
                events = [
                    event
                    async for event in app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id="pending-fallback",
                            approval_id=approval["approval_id"],
                            tool_round_id=approval["tool_round_id"],
                            tool_call_id=approval["tool_call_id"],
                            decision=ToolApprovalDecision.APPROVE,
                        )
                    )
                ]
                assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
                # Approval does not reconstruct unavailable quarantined arguments.
                # The call is resolved without executing it; the parent may continue.
                assert tool.calls == 0
                assert not primary.requests and len(backup.requests) == 1
                assert backup.requests[0].model == "large"
                checkpoint = await store.load_checkpoint("pending-fallback")
                assert checkpoint["model_failover"]["candidate_index"] == 1
                assert checkpoint["model_failover"]["plan"]["max_total_attempts"] == 20
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())
