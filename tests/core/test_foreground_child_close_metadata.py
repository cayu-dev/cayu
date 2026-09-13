"""Resolution metadata survives a committed close and real process loss."""

from __future__ import annotations

import asyncio
import os
import select
import subprocess
import sys
from pathlib import Path

import pytest
from tests.core.test_foreground_child_restart import _RestartRecordingTool
from tests.core.test_foreground_subagent_recovery import _identity, _Provider

from cayu import (
    AgentSpec,
    CayuApp,
    IncompleteSessionRecoveryRequest,
    Message,
    RunRequest,
    SessionQuery,
    SessionStatus,
    SQLiteSessionStore,
    SubagentSpec,
    SubagentTool,
    ToolApprovalDecision,
)
from cayu.providers import ModelStreamEvent
from cayu.runtime import ToolApprovalRequest, UserInputResponse
from cayu.runtime.approvals import ResolutionActor, ResolutionActorSource
from cayu.runtime.invocation import InvocationOriginClaim
from cayu.runtime.tool_policy import ToolPolicy, ToolPolicyDecision, ToolPolicyResult
from cayu.tools.user_input import UserInputTool

_METADATA = {
    "resolution": {"route": "reviewed-雪", "enabled": True},
    "cayu:taint_labels": ["untrusted_web"],
}
_ACTOR = ResolutionActor(
    subject="reviewer-雪", tenant="tenant", source=ResolutionActorSource.REQUEST
)
_KEY = "foreground_child_post_action_continuation"


class _MetadataPolicy(ToolPolicy):
    def __init__(self):
        self.observed = []

    @property
    def execution_profile_identity(self):
        return _identity("close-metadata-policy")

    async def authorize(self, request):
        if request.tool_name == "record" and request.arguments["value"] == 7:
            return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)
        if request.tool_name == "record":
            self.observed.append(request.metadata)
            return ToolPolicyResult(
                decision=ToolPolicyDecision.ALLOW
                if request.metadata == _METADATA
                else ToolPolicyDecision.DENY
            )
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


async def _worker(path, action, phase, delayed="no"):
    delayed = delayed == "yes"
    store = SQLiteSessionStore(path)
    opening = [
        [
            ModelStreamEvent.tool_call(
                id="spawn", name="subagent", arguments={"agent": "child", "task": "work"}
            ),
            ModelStreamEvent.completed(),
        ],
        [
            ModelStreamEvent.tool_call(
                id="action",
                name="ask_user" if action == "input" else "record",
                arguments={"question": "Continue?"} if action == "input" else {"value": 7},
            ),
            ModelStreamEvent.completed(),
        ],
    ]
    remaining = [
        [
            ModelStreamEvent.tool_call(id="after", name="record", arguments={"value": 8}),
            ModelStreamEvent.completed(),
        ],
        [ModelStreamEvent.text_delta("child done"), ModelStreamEvent.completed()],
        [ModelStreamEvent.text_delta("parent done"), ModelStreamEvent.completed()],
    ]
    if delayed:
        opening.append(opening[-1])
    provider = _Provider(remaining if phase == "recover" else opening + remaining)
    tool = _RestartRecordingTool()
    policy = _MetadataPolicy()
    app = CayuApp(
        session_store=store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="test"),
        tools=[
            SubagentTool(
                app,
                agents={"child": SubagentSpec(agent_name="child")},
                execution_profile_identity=_identity("close-metadata-subagent"),
            )
        ],
    )
    app.register_agent(
        AgentSpec(name="child", model="test"), tools=[tool, UserInputTool()], tool_policy=policy
    )
    try:
        if phase != "recover":
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="parent",
                        agent_name="parent",
                        messages=[Message.text("user", "go")],
                        metadata={
                            "original": "must not replace resolution metadata",
                            "cayu:taint_labels": ["untrusted_web"],
                        },
                        invocation_origin=InvocationOriginClaim(
                            subject="requester", tenant="tenant"
                        ),
                    )
                )
            ]
        children = await store.list_sessions(SessionQuery(parent_session_id="parent"))
        assert len(children.sessions) == 1
        child = children.sessions[0]
        parent = await store.load("parent")
        assert child.invocation.origin == parent.invocation.origin
        assert child.invocation.origin.subject == "requester"
        assert child.invocation.root_invocation_id == parent.invocation.root_invocation_id
        assert child.parent_session_id == parent.id
        assert child.metadata["cayu:taint_labels"] == ["untrusted_web"]
        if delayed and phase != "recover":
            # Defer only discovery delivery. The second public child resolution
            # must close independently of that still-pending parent update.
            deliver = app._event_writer._continue_foreground_parent

            async def delay_discovery(claim):
                if claim.session_id == child.id and claim.event.type == "session.interrupted":
                    return False
                return await deliver(claim)

            app._event_writer._continue_foreground_parent = delay_discovery
            checkpoint = await store.load_checkpoint(child.id)
            if action == "input":
                first = app.resolve_user_input(
                    UserInputResponse(
                        session_id=child.id,
                        input_id=checkpoint["pending_user_input"]["input_id"],
                        answer="first",
                        metadata=_METADATA,
                        resolved_by=_ACTOR,
                    )
                )
            else:
                approval = checkpoint["pending_tool_approval"]
                first = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=child.id,
                        approval_id=approval["approval_id"],
                        tool_round_id=approval["tool_round_id"],
                        tool_call_id=approval["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                        metadata=_METADATA,
                        resolved_by=_ACTOR,
                    )
                )
            _ = [event async for event in first]
        if phase == "recover":
            checkpoint = await store.load_checkpoint(child.id)
            marker = checkpoint[_KEY]
            assert marker["request_metadata"] == _METADATA
            if delayed:
                assert marker["action_id"] != marker["wait"]["child_action_id"]

            # An otherwise identical marker with altered metadata cannot borrow
            # the immutable close receipt's authority.
            def corrupt(_session, current):
                current[_KEY]["request_metadata"] = {"different": True}
                return current

            await store.transform_checkpoint(child.id, corrupt)
            before = await store.load_checkpoint(child.id)
            with pytest.raises(RuntimeError, match="close"):
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id=child.id,
                        inactive_for_seconds=0,
                    )
                )
            assert await store.load_checkpoint(child.id) == before
            assert provider.requests == [] and tool.values == []

            def restore(_session, current):
                current[_KEY] = marker
                return current

            await store.transform_checkpoint(child.id, restore)
            if delayed:
                # The earlier action's valid lineage is not authority to close
                # that action again under the newer receipt.
                def rewind_action(_session, current):
                    current[_KEY]["action_id"] = marker["wait"]["child_action_id"]
                    prefix = "approval-close" if action == "approval" else "user-input-close"
                    current[_KEY]["close_publication_id"] = (
                        f"{prefix}:{marker['wait']['child_action_id']}"
                    )
                    return current

                await store.transform_checkpoint(child.id, rewind_action)
                before = await store.load_checkpoint(child.id)
                with pytest.raises(RuntimeError, match="close"):
                    await app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(
                            session_id=child.id, inactive_for_seconds=0
                        )
                    )
                assert await store.load_checkpoint(child.id) == before
                assert provider.requests == [] and tool.values == []
                await store.transform_checkpoint(child.id, restore)
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=child.id,
                    inactive_for_seconds=0,
                )
            )
        else:
            if phase == "kill":
                publish = app._runtime_session_store.publish_runtime_publication

                async def stop_after_close(session_id, **kwargs):
                    request = kwargs["request"]
                    if request.kind in {"approval-close", "user-input-close"}:
                        changed = request.model_copy(deep=True)
                        operation = next(op for op in changed.mutation.operations if op.key == _KEY)
                        operation.value["request_metadata"] = {"different": True}
                        before = await store.load_checkpoint(session_id)
                        with pytest.raises(ValueError, match="continuation conflicts"):
                            await publish(session_id, **{**kwargs, "request": changed})
                        assert await store.load_checkpoint(session_id) == before
                        assert (
                            await store.load_runtime_publication_receipt(
                                session_id, request.publication_id
                            )
                            is None
                        )
                    result = await publish(session_id, **kwargs)
                    if request.kind in {"approval-close", "user-input-close"}:
                        assert (await store.load_checkpoint(session_id))[_KEY][
                            "request_metadata"
                        ] == _METADATA
                        print("CLOSED", flush=True)
                        await asyncio.Event().wait()
                    return result

                app._runtime_session_store.publish_runtime_publication = stop_after_close
            events = await store.load_events(child.id)
            if action == "input":
                pending = next(
                    event
                    for event in reversed(events)
                    if event.type == "session.awaiting_user_input"
                )
                stream = app.resolve_user_input(
                    UserInputResponse(
                        session_id=child.id,
                        input_id=pending.payload["input_id"],
                        answer="yes",
                        metadata=_METADATA,
                        resolved_by=_ACTOR,
                    )
                )
            else:
                pending = next(
                    event
                    for event in reversed(events)
                    if event.type == "tool.call.approval_requested"
                )
                stream = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=child.id,
                        approval_id=pending.payload["approval"]["approval_id"],
                        tool_round_id=pending.payload["tool_round_id"],
                        tool_call_id=pending.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                        metadata=_METADATA,
                        resolved_by=_ACTOR,
                    )
                )
            _ = [event async for event in stream]
        assert await app.drain_background_interruptions(timeout_s=10)
        await app.recover_persisted_event_side_effects()
        assert policy.observed == [_METADATA]
        assert tool.values == (
            ([7] * (2 if delayed else 1)) + [8]
            if phase == "control" and action == "approval"
            else [8]
        )
        assert len(provider.requests) == (3 if phase == "recover" else 5 + int(delayed))
        assert (await store.load(child.id)).status is SessionStatus.COMPLETED
        assert (await store.load(child.id)).invocation == child.invocation
        assert (await store.load("parent")).metadata["cayu:taint_labels"] == ["untrusted_web"]
        actor_events = [
            event
            for event in await store.load_events(child.id)
            if event.payload.get("resolved_by") is not None
        ]
        assert actor_events
        assert all(
            event.payload["resolved_by"]
            == {"subject": "reviewer-雪", "tenant": "tenant", "source": "request"}
            for event in actor_events
        )
        assert (await store.load("parent")).status is SessionStatus.COMPLETED
        assert (
            sum(event.type == "tool.call.completed" for event in await store.load_events("parent"))
            == 1
        )
    finally:
        assert await app.drain_background_interruptions(timeout_s=10)
        await store.close()


@pytest.mark.parametrize("action", ["approval", "input"])
@pytest.mark.parametrize("restart", [False, True], ids=["uninterrupted", "sigkill"])
@pytest.mark.parametrize("delayed", ["no", "yes"], ids=["current-discovery", "stale-discovery"])
def test_resolution_metadata_reaches_later_policy(tmp_path, action, restart, delayed):
    command = [
        sys.executable,
        "-m",
        "tests.core.test_foreground_child_close_metadata",
        str(tmp_path / "metadata.sqlite"),
        action,
    ]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd()))),
    }
    if restart:
        process = subprocess.Popen(
            [*command, "kill", delayed],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        try:
            assert process.stdout is not None
            ready, _, _ = select.select([process.stdout], [], [], 60)
            assert ready, "Close worker did not reach its barrier"
            line = process.stdout.readline()
            assert line.strip() == "CLOSED", line + (
                process.communicate(timeout=10)[0] if process.poll() is not None else ""
            )
            process.kill()
            process.communicate(timeout=10)
            assert process.returncode == -9
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    result = subprocess.run(
        [*command, "recover" if restart else "control", delayed],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    asyncio.run(_worker(*sys.argv[1:]))
