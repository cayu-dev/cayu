"""Legacy recovered outcomes cannot prove the assistant's invocation secret scope."""

from datetime import UTC, datetime, timedelta

import pytest
from tests.core._workload_secret_support import (
    FakeProvider,
    RequireApprovalPolicy,
    collect_events,
    collect_tool_approval_events,
)
from tests.core.test_tool_start_argument_quarantine import (
    _portable_environment_spec,
    _ResolveAfterStartTool,
)

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.budgets.run_limits import RunLimits
from cayu.environments.base import Environment
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.vaults.static import StaticVault


class _FailApprovalCloseStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1
    fail_close = True

    async def publish_runtime_publication(self, session_id, **kwargs):
        if self.fail_close and kwargs["request"].kind == "approval-close":
            raise OSError("simulated approval-close publication failure")
        return await super().publish_runtime_publication(session_id, **kwargs)


@pytest.mark.anyio
@pytest.mark.parametrize("recover_completed", [False, True])
async def test_legacy_approval_limit_requires_scope_for_recovered_calls(recover_completed):
    secret = "legacy-approval-limit-invocation-secret-canary"
    session_id = "legacy-approval-limit"
    now = [datetime.now(UTC)]
    store = _FailApprovalCloseStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="first", name="resolve_after_start", arguments={"provided": secret}
            ),
            ModelStreamEvent.tool_call(
                id="second", name="resolve_after_start", arguments={"provided": secret}
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )

    def make_app():
        tool = _ResolveAfterStartTool()
        app = CayuApp(session_store=store, clock=lambda: now[0], enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                vault=StaticVault({"api_key": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireApprovalPolicy(),
        )
        return app, tool

    app, first_tool = make_app()
    await collect_events(
        app,
        RunRequest(
            session_id=session_id,
            agent_name="assistant",
            messages=[Message.text("user", "run")],
            limits=RunLimits(max_elapsed_seconds=600, scope="session"),
        ),
    )
    pending = (await store.load_checkpoint(session_id))["pending_tool_approval"]
    request = ToolApprovalRequest(
        session_id=session_id,
        approval_id=pending["approval_id"],
        tool_round_id=pending["tool_round_id"],
        tool_call_id=pending["tool_call_id"],
        decision=ToolApprovalDecision.APPROVE,
    )
    if recover_completed:
        failed = await collect_tool_approval_events(app, request)
        assert failed[-1].type is EventType.SESSION_INTERRUPTED
        assert first_tool.arguments == [{"provided": secret}, {"provided": secret}]
        checkpoint = await store.load_checkpoint(session_id)
        assert len(checkpoint["pending_tool_round"]["staged_terminals"]) == 2

    def strip_additive_publication(_session, checkpoint):
        # Pre-field checkpoints also lack positive approval secret-scope evidence.
        checkpoint["pending_tool_round"].pop("assistant_publication", None)
        checkpoint["pending_tool_approval"].pop("secret_resolution_scope", None)
        return checkpoint

    await store.transform_checkpoint(session_id, strip_additive_publication)
    store.fail_close = False
    now[0] = (await store.load(session_id)).created_at + timedelta(seconds=601)
    recovery_app, recovery_tool = make_app()
    recovered = await collect_tool_approval_events(recovery_app, request)
    assert recovery_tool.arguments == []
    assert len(provider.requests) == 1
    checkpoint = await store.load_checkpoint(session_id)
    if recover_completed:
        publication = checkpoint["pending_tool_round"]["assistant_publication"]
        assert publication["state"] == "blocked"
        assert publication["reason"] == "incomplete_invocation_secret_scope"
        assert recovered[-1].type is EventType.SESSION_INTERRUPTED
        assert "pending_tool_approval" in checkpoint
        assert secret not in repr(await store.load_transcript(session_id))
        assert secret not in repr(recovered)
    else:
        assert recovered[-1].type is EventType.SESSION_INTERRUPTED
        assert "pending_tool_approval" not in checkpoint
