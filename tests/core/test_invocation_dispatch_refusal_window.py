"""Dispatch revocation after before-capture must close process-local attribution."""

import asyncio

import pytest
from tests.core.test_resource_execution_access import Policy, scope
from tests.core.test_workspace_mutation_receipts import (
    _ExclusiveWriterBinding,
    _portable_environment_spec,
    _PublicWorkspaceWriteTool,
    _SingleToolProvider,
)

import cayu.runtime._tool_invocation.invocation as invocation_module
from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment
from cayu.sessions import InMemorySessionStore
from cayu.sessions._access_rules import SessionAccessScope
from cayu.workspaces import LocalWorkspace


@pytest.mark.parametrize("revoke", [False, True])
def test_pre_dispatch_refusal_closes_workspace_attribution(tmp_path, monkeypatch, revoke):
    async def scenario():
        policy = Policy()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, resource_access_policy=policy, enable_logging=False)
        app.register_provider(
            _SingleToolProvider(tool_name="public_workspace_write", arguments={"path": "out.txt"}),
            default=True,
        )
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                workspace=LocalWorkspace(tmp_path, workspace_id="owned"),
                binding=_ExclusiveWriterBinding(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="worker", model="scripted-model"), tools=[_PublicWorkspaceWriteTool()]
        )
        windows = []
        original_begin = invocation_module.begin_workspace_mutation_window
        original_publish = invocation_module.publish_workspace_observation_transition
        reached = []

        def begin(*args, **kwargs):
            window = original_begin(*args, **kwargs)
            windows.append(window)
            return window

        async def publish(**kwargs):
            result = await original_publish(**kwargs)
            if kwargs.get("phase") == "before-capture":
                reached.append(True)
                if revoke:
                    policy.current = SessionAccessScope()
            return result

        monkeypatch.setattr(invocation_module, "begin_workspace_mutation_window", begin)
        monkeypatch.setattr(invocation_module, "publish_workspace_observation_transition", publish)
        access = await app.access("alice")
        try:
            try:
                _ = [
                    event
                    async for event in access.run(
                        RunRequest(
                            session_id="refused-workspace",
                            agent_name="worker",
                            labels={"organization": "acme"},
                            messages=[Message.text("user", "write")],
                        )
                    )
                ]
            except PermissionError:
                if not revoke:
                    raise
            assert reached == [True]
            assert (tmp_path / "out.txt").exists() is not revoke
            policy.current = scope("acme")
            # A fresh invocation has no unfinished work and no checkpoint from
            # the refused session, so its receipt must not claim concurrent writers.
            followup = CayuApp(session_store=store, enable_logging=False)
            followup.register_provider(
                _SingleToolProvider(
                    tool_name="public_workspace_write", arguments={"path": "next.txt"}
                ),
                default=True,
            )
            followup.register_environment(
                Environment(
                    _portable_environment_spec("local"),
                    workspace=LocalWorkspace(tmp_path, workspace_id="owned"),
                    binding=_ExclusiveWriterBinding(),
                ),
                default=True,
            )
            followup.register_agent(
                AgentSpec(name="worker", model="scripted-model"),
                tools=[_PublicWorkspaceWriteTool()],
            )
            try:
                events = [
                    event
                    async for event in followup.run(
                        RunRequest(
                            session_id="healthy-followup",
                            agent_name="worker",
                            messages=[Message.text("user", "write")],
                        )
                    )
                ]
                assert events[-1].type is EventType.SESSION_COMPLETED
                receipt = next(
                    event for event in events if event.type is EventType.WORKSPACE_MUTATION_RECORDED
                )
                assert receipt.payload["attribution"]["overlap_detected"] is False, receipt.payload
                assert all(window.closed for window in windows), (
                    "A refused, undispatched tool left a process-local active writer",
                    windows,
                )
            finally:
                await followup.aclose()
        finally:
            policy.current = scope("acme")
            for window in windows:
                window.close(discard_history=True)
            await app.aclose()

    asyncio.run(scenario())
