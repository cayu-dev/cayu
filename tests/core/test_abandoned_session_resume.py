from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from tests.core.test_runtime import RequireApprovalPolicy, VersionedFakeProvider
from tests.core.test_session_execution_presence import _app, _consume, _stores

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ModelStreamEvent,
    PostgresSessionStore,
    ResumeRequest,
    RunRequest,
    SessionExecutionConfig,
    SessionExecutionInProgress,
    SessionStatus,
    SQLiteSessionStore,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolEffect,
    ToolResult,
    ToolSpec,
    UserInputResponse,
    UserInputTool,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.storage.migrations import SchemaMode


class _Proposal(Tool):
    def __init__(self, effect, receipt, *, mode="return"):
        self.spec = ToolSpec(
            name="propose",
            description="Record the proposal exactly once.",
            input_schema={"type": "object", "properties": {}},
            effect=effect,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:abandoned-proposal", behavior_version="1", implementation_version="1"
            ),
        )
        self.receipt = Path(receipt)
        self.mode = mode

    async def run(self, arguments, context):
        with self.receipt.open("a") as receipt:
            receipt.write("committed\n")
        if self.mode == "crash":
            os._exit(137)
        if self.mode == "block":
            print("committed", flush=True)
            await asyncio.Event().wait()
        return ToolResult(content="recorded")


class _ProposalApprovalPolicy(RequireApprovalPolicy):
    execution_profile_identity = ExecutionProfileBehaviorIdentity(
        name="tests:proposal-approval", behavior_version="1", implementation_version="1"
    )


def _proposal_app(store, tool, provider, gate="none"):
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        session_execution=SessionExecutionConfig(heartbeat_interval_seconds=0.05, lease_seconds=60),
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool, UserInputTool()] if gate == "input" else [tool],
        tool_policy=_ProposalApprovalPolicy() if gate == "approval" else None,
    )
    return app


def _run_child():
    async def scenario():
        store = (
            SQLiteSessionStore(os.environ["CAYU_CRASH_TEST_STORE"])
            if os.environ["CAYU_CRASH_TEST_BACKEND"] == "sqlite"
            else PostgresSessionStore(
                os.environ["CAYU_CRASH_TEST_STORE"],
                min_size=1,
                max_size=2,
                schema_mode=SchemaMode.CREATE,
            )
        )
        tool = _Proposal(
            ToolEffect(os.environ["CAYU_CRASH_TEST_EFFECT"]),
            os.environ["CAYU_CRASH_TEST_RECEIPT"],
            mode=os.environ["CAYU_CRASH_TEST_MODE"],
        )
        gate = os.environ["CAYU_CRASH_TEST_GATE"]
        model_events = [
            ModelStreamEvent.tool_call(id="proposal", name="propose", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
        if gate == "input":
            model_events.insert(
                0,
                ModelStreamEvent.tool_call(
                    id="question", name="ask_user", arguments={"question": "Continue?"}
                ),
            )
        app = _proposal_app(store, tool, VersionedFakeProvider(model_events), gate)
        if os.environ["CAYU_CRASH_TEST_MODE"] == "legacy":
            store.supports_session_execution = False
            tool.mode = "crash"
        events = await _consume(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=os.environ["CAYU_CRASH_TEST_SESSION"],
                    messages=[Message.text("user", "Please propose a replacement")],
                )
            )
        )
        if gate == "approval":
            pending = next(
                event for event in events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            resolution = ToolApprovalRequest(
                session_id=os.environ["CAYU_CRASH_TEST_SESSION"],
                approval_id=pending.payload["approval"]["approval_id"],
                tool_round_id=pending.payload["tool_round_id"],
                tool_call_id=pending.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            )
            stream = app.resolve_tool_approval(resolution)
        elif gate == "input":
            pending = next(
                event for event in events if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            resolution = UserInputResponse(
                session_id=os.environ["CAYU_CRASH_TEST_SESSION"],
                input_id=pending.payload["input_id"],
                answer="yes",
            )
            stream = app.resolve_user_input(resolution)
        else:
            raise AssertionError("The proposal fixture did not stop inside its tool.")
        tool.receipt.with_suffix(".resolution.json").write_text(resolution.model_dump_json())
        await _consume(stream)

    asyncio.run(scenario())


def _child(store, backend, request, receipt, sid, effect, mode, *, gate="none"):
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from tests.core.test_abandoned_session_resume import _run_child; _run_child()",
        ],
        env={
            **os.environ,
            "CAYU_CRASH_TEST_BACKEND": backend,
            "CAYU_CRASH_TEST_STORE": str(store.path)
            if backend == "sqlite"
            else request.getfixturevalue("postgres_dsn"),
            "CAYU_CRASH_TEST_EFFECT": effect.value,
            "CAYU_CRASH_TEST_RECEIPT": str(receipt),
            "CAYU_CRASH_TEST_SESSION": sid,
            "CAYU_CRASH_TEST_MODE": mode,
            "CAYU_CRASH_TEST_GATE": gate,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("effect", [ToolEffect.NONE, ToolEffect.IDEMPOTENT, ToolEffect.EXTERNAL])
def test_resume_recovers_process_death_without_repeating_tool(
    backend, effect, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "proposal-receipt"
            sid = "crashed-" + uuid4().hex
            child = _child(store, backend, request, receipt, sid, effect, "crash")
            try:
                stdout, stderr = await asyncio.wait_for(asyncio.to_thread(child.communicate), 30)
                assert child.returncode == 137, (stdout, stderr)
            finally:
                if child.poll() is None:
                    child.kill()
                    await asyncio.to_thread(child.wait)
            observer_store = reopen()
            original = await observer_store.load(sid)
            assert original.status is SessionStatus.RUNNING
            app = _app(
                observer_store,
                VersionedFakeProvider(
                    [
                        ModelStreamEvent.text_delta("Your proposal is recorded."),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ),
                tools=[_Proposal(effect, receipt)],
            )
            execution = await app.inspect_session_execution(sid)
            assert execution.state == "owner_lost"
            assert execution.lease_expires_at > datetime.now(UTC)
            events = await _consume(
                app.resume(
                    ResumeRequest(session_id=sid, messages=[Message.text("user", "Any update?")])
                )
            )
            current = await observer_store.load(sid)
            assert current.run_epoch > original.run_epoch
            assert receipt.read_text() == "committed\n"
            assert any(
                event.type is EventType.SESSION_RUN_FENCED
                and event.payload["reason"] == "continuation_recovered_abandoned_execution"
                for event in events
            )
            if effect is ToolEffect.EXTERNAL:
                assert current.status is SessionStatus.INTERRUPTED
                assert any(event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for event in events)
            else:
                assert current.status is SessionStatus.COMPLETED
                assert any(event.type is EventType.SESSION_COMPLETED for event in events)
                interrupted = next(e for e in events if e.type is EventType.TOOL_CALL_FAILED)
                guidance = interrupted.payload["result"]["content"]
                assert "calling it again for the same operation is safe" in guidance
                assert "inspect external state" not in guidance

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_resume_refuses_live_executor_in_another_process(
    backend, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "proposal-receipt"
            sid = "live-" + uuid4().hex
            child = _child(store, backend, request, receipt, sid, ToolEffect.IDEMPOTENT, "block")
            try:
                assert (
                    await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 30)
                    == "committed\n"
                )
                app = _app(
                    reopen(),
                    VersionedFakeProvider([]),
                    tools=[_Proposal(ToolEffect.IDEMPOTENT, receipt)],
                )
                original = await store.load(sid)
                await asyncio.sleep(1.6)
                assert (await app.inspect_session_execution(sid)).state == "executing"
                with pytest.raises(SessionExecutionInProgress, match="recover_incomplete_session"):
                    await _consume(
                        app.resume(
                            ResumeRequest(
                                session_id=sid, messages=[Message.text("user", "Any update?")]
                            )
                        )
                    )
                current = await store.load(sid)
                assert current == original
                assert receipt.read_text() == "committed\n"
            finally:
                if child.poll() is None:
                    child.kill()
                await asyncio.to_thread(child.wait)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("gate", ["approval", "input"])
def test_human_continuation_recovers_crash_during_accepted_tool(
    backend, gate, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            sid = "resolution-" + uuid4().hex
            receipt = tmp_path / "proposal-receipt"
            child = _child(
                store, backend, request, receipt, sid, ToolEffect.IDEMPOTENT, "crash", gate=gate
            )
            try:
                stdout, stderr = await asyncio.wait_for(asyncio.to_thread(child.communicate), 30)
                assert child.returncode == 137, (stdout, stderr)
            finally:
                if child.poll() is None:
                    child.kill()
                    await asyncio.to_thread(child.wait)
            app = _proposal_app(
                reopen(),
                _Proposal(ToolEffect.IDEMPOTENT, receipt),
                VersionedFakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
                gate,
            )
            retained = json.loads(receipt.with_suffix(".resolution.json").read_text())
            stream = (
                app.resolve_tool_approval(ToolApprovalRequest(**retained))
                if gate == "approval"
                else app.resolve_user_input(UserInputResponse(**retained))
            )
            await _consume(stream)
            assert receipt.read_text() == "committed\n"
            assert (await store.load(sid)).status in {
                SessionStatus.COMPLETED,
                SessionStatus.INTERRUPTED,
            }

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_uninstrumented_execution_requires_explicit_recovery(
    backend, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            sid = "legacy-" + uuid4().hex
            receipt = tmp_path / "proposal-receipt"
            child = _child(store, backend, request, receipt, sid, ToolEffect.IDEMPOTENT, "legacy")
            try:
                stdout, stderr = await asyncio.wait_for(asyncio.to_thread(child.communicate), 30)
                assert child.returncode == 137, (stdout, stderr)
            finally:
                if child.poll() is None:
                    child.kill()
                    await asyncio.to_thread(child.wait)
            app = _app(
                reopen(),
                VersionedFakeProvider([]),
                tools=[_Proposal(ToolEffect.IDEMPOTENT, receipt)],
            )
            original = await store.load(sid)
            assert (await app.inspect_session_execution(sid)).state == "unknown"
            with pytest.raises(SessionExecutionInProgress, match="recover_incomplete_session"):
                await _consume(
                    app.resume(
                        ResumeRequest(
                            session_id=sid, messages=[Message.text("user", "Any update?")]
                        )
                    )
                )
            assert await store.load(sid) == original
            assert receipt.read_text() == "committed\n"

    asyncio.run(scenario())
