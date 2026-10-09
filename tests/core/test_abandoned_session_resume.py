from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from tests.core.test_runtime import RequireApprovalPolicy, VersionedFakeProvider
from tests.core.test_session_execution_presence import _app, _consume, _stores

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    IncompleteSessionRecoveryRequest,
    Message,
    ModelStreamEvent,
    ModelTarget,
    PostgresSessionStore,
    ResumeRequest,
    RunLimits,
    RunRequest,
    SessionExecutionConfig,
    SessionExecutionInProgress,
    SessionStatus,
    SQLiteSessionStore,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolEffect,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyResult,
    ToolResult,
    ToolSpec,
    UserInputResponse,
    UserInputTool,
)
from cayu.observability.hooks import BeforeToolCallDecision, RuntimeHook
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.storage.migrations import SchemaMode

_PROPOSAL_ARGUMENTS = {"body": "Replacement proposal with private customer details"}


class _Proposal(Tool):
    def __init__(
        self,
        effect,
        receipt,
        *,
        mode="return",
        publish_arguments=True,
        keyed=False,
        sequential=False,
    ):
        self.spec = ToolSpec(
            name="propose",
            description="Record the proposal exactly once.",
            input_schema={"type": "object", "properties": {"body": {"type": "string"}}},
            effect=effect,
            parallel_safe=not sequential,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:abandoned-proposal", behavior_version="1", implementation_version="1"
            ),
        )
        self.receipt = Path(receipt)
        self.mode = mode
        self.publish_arguments = publish_arguments
        self.keyed = keyed
        self.sequential = sequential
        self.entered = 0

    @property
    def _publish_arguments(self):
        return self.publish_arguments

    async def run(self, context, arguments):
        with self.receipt.with_suffix(".calls").open("a") as calls:
            calls.write(context.idempotency_key + "\n")
        key = context.idempotency_key if self.keyed else "committed"
        prior = self.receipt.read_text().splitlines() if self.receipt.exists() else []
        if not self.keyed or key not in prior:
            with self.receipt.open("a") as receipt:
                receipt.write(key + "\n")
        if self.mode == "crash_batch":
            self.entered += 1
            if self.entered < 2:
                await asyncio.Event().wait()
            os._exit(137)
        if self.mode == "crash" and (
            not self.sequential or arguments["body"] == _PROPOSAL_ARGUMENTS["body"]
        ):
            os._exit(137)
        if self.mode == "block":
            print("committed", flush=True)
            await asyncio.Event().wait()
        return ToolResult(content="recorded")


class _ProposalApprovalPolicy(RequireApprovalPolicy):
    execution_profile_identity = ExecutionProfileBehaviorIdentity(
        name="tests:proposal-approval", behavior_version="1", implementation_version="1"
    )


class _FlagPolicy(ToolPolicy):
    """Allows until its flag file exists, then denies; one stable declaration."""

    execution_profile_identity = ExecutionProfileBehaviorIdentity(
        name="tests:proposal-flag-policy", behavior_version="1", implementation_version="1"
    )

    def __init__(self, flag):
        self.flag = Path(flag)

    async def authorize(self, request):
        if not self.flag.exists():
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)
        if self.flag.read_text() == "raise":
            raise RuntimeError("Policy service unavailable.")
        if self.flag.read_text() == "approve":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL, reason="Proposals need review."
            )
        return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="Proposals paused.")


class _FlagHook(RuntimeHook):
    """Proceeds until its flag file names a ``before_tool_call`` action to take."""

    execution_profile_identity = ExecutionProfileBehaviorIdentity(
        name="tests:proposal-flag-hook", behavior_version="1", implementation_version="1"
    )

    def __init__(self, flag):
        self.flag = Path(flag)

    async def before_tool_call(self, context):
        action = self.flag.read_text() if self.flag.exists() else "proceed"
        if action == "block":
            return BeforeToolCallDecision(action="block", block_reason="Proposals paused.")
        if action == "short_circuit":
            return BeforeToolCallDecision(
                action="short_circuit", synthetic_result=ToolResult(content="skipped")
            )
        return None


def _gate_policy(gate, receipt):
    if gate == "approval":
        return _ProposalApprovalPolicy()
    if gate == "flag":
        return _FlagPolicy(Path(receipt).with_suffix(".deny"))
    return None


def _proposal_app(store, tool, provider, gate="none", *, replay=True, lease_seconds=60, clock=None):
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        clock=clock,
        runtime_hooks=[_FlagHook(tool.receipt.with_suffix(".hook"))] if gate == "hook" else None,
        session_execution=SessionExecutionConfig(
            heartbeat_interval_seconds=0.05,
            lease_seconds=lease_seconds,
            replay_interrupted_tool_calls=replay,
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool, UserInputTool()] if gate == "input" else [tool],
        tool_policy=_gate_policy(gate, tool.receipt),
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
            publish_arguments=os.environ["CAYU_CRASH_TEST_PUBLISH_ARGUMENTS"] == "1",
            keyed=os.environ["CAYU_CRASH_TEST_KEYED"] == "1",
            sequential=os.environ.get("CAYU_CRASH_TEST_SEQUENTIAL") == "1",
        )
        gate = os.environ["CAYU_CRASH_TEST_GATE"]
        model_events = [
            ModelStreamEvent.tool_call(
                id="proposal", name="propose", arguments=_PROPOSAL_ARGUMENTS
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
        if tool.mode == "crash_batch":
            model_events.insert(
                -1,
                ModelStreamEvent.tool_call(
                    id="sibling", name="propose", arguments={"body": "sibling"}
                ),
            )
        if tool.sequential:
            model_events.insert(
                0,
                ModelStreamEvent.tool_call(
                    id="before", name="propose", arguments={"body": "before"}
                ),
            )
            model_events.insert(
                -1,
                ModelStreamEvent.tool_call(id="after", name="propose", arguments={"body": "after"}),
            )
        if gate == "input":
            model_events.insert(
                0,
                ModelStreamEvent.tool_call(
                    id="question", name="ask_user", arguments={"question": "Continue?"}
                ),
            )
        app = _proposal_app(store, tool, VersionedFakeProvider(model_events), gate)
        if os.environ["CAYU_CRASH_TEST_PHASE"] == "resume":
            await _consume(
                app.resume(
                    ResumeRequest(
                        session_id=os.environ["CAYU_CRASH_TEST_SESSION"],
                        messages=[Message.text("user", "Any update?")],
                    )
                )
            )
            raise AssertionError("The proposal fixture did not stop inside its replay.")
        if os.environ["CAYU_CRASH_TEST_MODE"] == "legacy":
            store.supports_session_execution = False
            tool.mode = "crash"
        events = await _consume(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    limits=RunLimits.model_validate_json(
                        os.environ.get("CAYU_CRASH_TEST_LIMITS", "{}")
                    ),
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


def _child(
    store,
    backend,
    request,
    receipt,
    sid,
    effect,
    mode,
    *,
    gate="none",
    publish_arguments=True,
    keyed=False,
    phase="run",
    sequential=False,
    limits=None,
):
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
            "CAYU_CRASH_TEST_PUBLISH_ARGUMENTS": "1" if publish_arguments else "0",
            "CAYU_CRASH_TEST_KEYED": "1" if keyed else "0",
            "CAYU_CRASH_TEST_PHASE": phase,
            "CAYU_CRASH_TEST_SEQUENTIAL": "1" if sequential else "0",
            "CAYU_CRASH_TEST_LIMITS": (limits or RunLimits()).model_dump_json(),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("effect", [ToolEffect.NONE, ToolEffect.IDEMPOTENT, ToolEffect.EXTERNAL])
@pytest.mark.parametrize("publish_arguments", [True, False])
def test_resume_recovers_process_death_without_repeating_tool(
    backend, effect, publish_arguments, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "proposal-receipt"
            sid = "crashed-" + uuid4().hex
            child = _child(
                store,
                backend,
                request,
                receipt,
                sid,
                effect,
                "crash",
                publish_arguments=publish_arguments,
            )
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
            provider = VersionedFakeProvider(
                [
                    ModelStreamEvent.text_delta("Your proposal is recorded."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            )
            app = _proposal_app(
                observer_store,
                _Proposal(effect, receipt, publish_arguments=publish_arguments),
                provider,
                replay=False,
                lease_seconds=1.5,
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
                # Static scope does not override a tool's private-argument contract.
                [call] = [
                    part
                    for message in provider.requests[0].messages
                    for part in message.content
                    if getattr(part, "tool_call_id", None) == "proposal"
                    and getattr(part, "arguments_state", None) is not None
                ]
                assert call.arguments_state == ("finalized" if publish_arguments else "unavailable")
                assert call.arguments == (_PROPOSAL_ARGUMENTS if publish_arguments else {})
                assert interrupted.payload["arguments_state"] == call.arguments_state
                if not publish_arguments:
                    assert _PROPOSAL_ARGUMENTS["body"] not in json.dumps(
                        [event.model_dump(mode="json") for event in events]
                    )
                    assert _PROPOSAL_ARGUMENTS["body"] not in json.dumps(
                        [
                            message.model_dump(mode="json")
                            for message in provider.requests[0].messages
                        ]
                    )
                guidance = interrupted.payload["result"]["content"]
                if effect is ToolEffect.IDEMPOTENT:
                    assert "same downstream idempotency identity is preserved" in guidance
                    assert "identical arguments alone do not guarantee deduplication" in guidance
                elif publish_arguments:
                    assert "calling it again with the same arguments is safe" in guidance
                else:
                    assert "original arguments are not shown" in guidance
                assert "inspect external state" not in guidance

    asyncio.run(scenario())


def test_recovered_idempotent_guidance_accounts_for_new_call_identity(
    request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores("sqlite", request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "keyed-proposal-receipt"
            sid = "keyed-" + uuid4().hex
            child = _child(
                store, "sqlite", request, receipt, sid, ToolEffect.IDEMPOTENT, "crash", keyed=True
            )
            try:
                stdout, stderr = await asyncio.wait_for(asyncio.to_thread(child.communicate), 30)
                assert child.returncode == 137, (stdout, stderr)
            finally:
                if child.poll() is None:
                    child.kill()
                    await asyncio.to_thread(child.wait)
            [original_key] = receipt.read_text().splitlines()
            provider = VersionedFakeProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="retry", name="propose", arguments=_PROPOSAL_ARGUMENTS
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("Done."),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            )
            app = _proposal_app(
                reopen(),
                _Proposal(ToolEffect.IDEMPOTENT, receipt, keyed=True),
                provider,
                replay=False,
                lease_seconds=1.5,
            )
            events = await _consume(
                app.resume(ResumeRequest(session_id=sid, messages=[Message.text("user", "Retry")]))
            )
            # Identical arguments do not preserve a ToolContext-based downstream key.
            keys = receipt.read_text().splitlines()
            assert keys[0] == original_key and len(keys) == len(set(keys)) == 2
            recovered = next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)
            guidance = recovered.payload["result"]["content"]
            assert "same downstream idempotency identity is preserved" in guidance
            assert "A new tool call receives a different ToolContext.idempotency_key" in guidance
            assert "calling it again with the same arguments is safe" not in guidance

    asyncio.run(scenario())


def test_recovered_arguments_stay_hidden_after_dynamic_secret_scope(
    request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores("sqlite", request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "proposal-receipt"
            sid = "dynamic-" + uuid4().hex
            child = _child(store, "sqlite", request, receipt, sid, ToolEffect.IDEMPOTENT, "crash")
            try:
                stdout, stderr = await asyncio.wait_for(asyncio.to_thread(child.communicate), 30)
                assert child.returncode == 137, (stdout, stderr)
            finally:
                if child.poll() is None:
                    child.kill()
                    await asyncio.to_thread(child.wait)
            observer_store = reopen()

            # A round whose model completion could resolve invocation secrets has no
            # sealed redactor after the crash, so its arguments must not be published.
            def dynamic(_session, checkpoint):
                checkpoint = json.loads(json.dumps(checkpoint))
                checkpoint["pending_tool_round"]["assistant_publication"][
                    "secret_resolution_scope"
                ] = "dynamic"
                return checkpoint

            await observer_store.transform_checkpoint(sid, dynamic)
            provider = VersionedFakeProvider(
                [
                    ModelStreamEvent.text_delta("Your proposal is recorded."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            )
            app = _app(observer_store, provider, tools=[_Proposal(ToolEffect.IDEMPOTENT, receipt)])
            await _consume(
                app.resume(
                    ResumeRequest(session_id=sid, messages=[Message.text("user", "Any update?")])
                )
            )
            assert (await observer_store.load(sid)).status is SessionStatus.COMPLETED
            assert not any(
                getattr(part, "tool_call_id", None) == "proposal"
                and getattr(part, "arguments_state", None) == "finalized"
                for message in provider.requests[0].messages
                for part in message.content
            )

    asyncio.run(scenario())


async def _crash_child(child):
    try:
        stdout, stderr = await asyncio.wait_for(asyncio.to_thread(child.communicate), 30)
        assert child.returncode == 137, (stdout, stderr)
    finally:
        if child.poll() is None:
            child.kill()
            await asyncio.to_thread(child.wait)


def _replay_provider():
    return VersionedFakeProvider(
        [
            ModelStreamEvent.text_delta("Your proposal is recorded."),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )


def _tool_results(provider, tool_call_id="proposal"):
    return [
        part
        for message in provider.requests[0].messages
        for part in message.content
        if getattr(part, "tool_call_id", None) == tool_call_id
        and getattr(part, "arguments_state", None) is None
    ]


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("effect", [ToolEffect.NONE, ToolEffect.IDEMPOTENT, ToolEffect.EXTERNAL])
def test_resume_replays_interrupted_tool_call_once(
    backend, effect, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "replay-receipt"
            sid = "replay-" + uuid4().hex
            await _crash_child(
                _child(store, backend, request, receipt, sid, effect, "crash", keyed=True)
            )
            [original_key] = receipt.with_suffix(".calls").read_text().splitlines()
            provider = _replay_provider()
            app = _proposal_app(
                reopen(),
                _Proposal(effect, receipt, keyed=True),
                provider,
                lease_seconds=1.5,
            )
            events = await _consume(
                app.resume(
                    ResumeRequest(session_id=sid, messages=[Message.text("user", "Any update?")])
                )
            )
            calls = receipt.with_suffix(".calls").read_text().splitlines()
            current = await app.session_store.load(sid)
            if effect is ToolEffect.EXTERNAL:
                assert calls == [original_key]
                assert current.status is SessionStatus.INTERRUPTED
                assert any(event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for event in events)
                return
            # One replay, under the original idempotency key, deduplicated downstream.
            assert calls == [original_key, original_key]
            assert receipt.read_text().splitlines() == [original_key]
            assert current.status is SessionStatus.COMPLETED
            stored = await app.session_store.load_events(sid)
            [terminal] = [
                event
                for event in stored
                if event.type is EventType.TOOL_CALL_COMPLETED
                and event.payload["tool_call_id"] == "proposal"
            ]
            assert terminal.payload["replayed_after_recovery"] is True
            assert terminal.payload["idempotency_key"] == original_key
            assert not any(event.type is EventType.TOOL_CALL_FAILED for event in stored)
            [result] = _tool_results(provider)
            assert result.content == "recorded"

    asyncio.run(scenario())


def test_replay_does_not_execute_unstarted_sibling(request, sqlite_resources, tmp_path):
    async def scenario():
        async with _stores("sqlite", request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "sequential-receipt"
            sid = "sequential-" + uuid4().hex
            await _crash_child(
                _child(
                    store,
                    "sqlite",
                    request,
                    receipt,
                    sid,
                    ToolEffect.IDEMPOTENT,
                    "crash",
                    sequential=True,
                    keyed=True,
                )
            )
            original_calls = receipt.with_suffix(".calls").read_text().splitlines()
            assert len(original_calls) == 2
            provider = _replay_provider()
            app = _proposal_app(
                reopen(),
                _Proposal(ToolEffect.IDEMPOTENT, receipt, keyed=True, sequential=True),
                provider,
                lease_seconds=1.5,
            )
            await _consume(
                app.resume(
                    ResumeRequest(session_id=sid, messages=[Message.text("user", "Any update?")])
                )
            )
            assert receipt.with_suffix(".calls").read_text().splitlines() == [
                *original_calls,
                original_calls[1],
            ]
            assert receipt.read_text().splitlines() == original_calls
            stored = await app.session_store.load_events(sid)
            assert [
                e.payload["tool_call_id"] for e in stored if e.type is EventType.TOOL_CALL_STARTED
            ] == ["before", "proposal"]
            assert [
                e.payload["tool_call_id"] for e in stored if e.type is EventType.TOOL_CALL_COMPLETED
            ] == ["before", "proposal"]
            [failed] = [e for e in stored if e.type is EventType.TOOL_CALL_FAILED]
            assert failed.payload["tool_call_id"] == "after"
            assert (await app.session_store.load(sid)).status is SessionStatus.COMPLETED

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", ["elapsed", "tool_calls"])
def test_replay_obeys_session_limits(limit, request, sqlite_resources, tmp_path):
    async def scenario():
        async with _stores("sqlite", request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "limited-receipt"
            sid = "limited-" + uuid4().hex
            limits = (
                RunLimits(max_elapsed_seconds=60, scope="session")
                if limit == "elapsed"
                else RunLimits(max_tool_calls=1, scope="session")
            )
            await _crash_child(
                _child(
                    store,
                    "sqlite",
                    request,
                    receipt,
                    sid,
                    ToolEffect.IDEMPOTENT,
                    "crash",
                    keyed=True,
                    limits=limits,
                )
            )
            original_calls = receipt.with_suffix(".calls").read_text().splitlines()
            assert len(original_calls) == 1
            session = await store.load(sid)
            now = session.created_at + timedelta(seconds=120)
            provider = _replay_provider()
            app = _proposal_app(
                reopen(),
                _Proposal(ToolEffect.IDEMPOTENT, receipt, keyed=True),
                provider,
                lease_seconds=1.5,
                clock=lambda: now,
            )
            events = await _consume(
                app.resume(
                    ResumeRequest(session_id=sid, messages=[Message.text("user", "Any update?")])
                )
            )
            if limit == "tool_calls":
                # This is the same logical call, already counted by its start;
                # replay must not charge a second call against the one-call limit.
                assert receipt.with_suffix(".calls").read_text().splitlines() == original_calls * 2
                assert (await app.session_store.load(sid)).status is SessionStatus.COMPLETED
                [result] = _tool_results(provider)
                assert result.content == "recorded"
                return
            assert receipt.with_suffix(".calls").read_text().splitlines() == original_calls
            assert not provider.requests
            assert (await app.session_store.load(sid)).status is SessionStatus.INTERRUPTED
            [reached] = [e for e in events if e.type is EventType.SESSION_LIMIT_REACHED]
            assert reached.payload["limit"] == "elapsed_seconds"
            stored = await app.session_store.load_events(sid)
            assert not any(e.type is EventType.TOOL_CALL_COMPLETED for e in stored)
            [failed] = [e for e in stored if e.type is EventType.TOOL_CALL_FAILED]
            assert failed.payload["result"]["structured"]["outcome_unknown"] is True

    asyncio.run(scenario())


def test_replay_rechecks_limit_between_started_calls(request, sqlite_resources, tmp_path):
    async def scenario():
        async with _stores("sqlite", request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "batch-receipt"
            sid = "batch-" + uuid4().hex
            await _crash_child(
                _child(
                    store,
                    "sqlite",
                    request,
                    receipt,
                    sid,
                    ToolEffect.IDEMPOTENT,
                    "crash_batch",
                    keyed=True,
                    limits=RunLimits(max_elapsed_seconds=60, scope="session"),
                )
            )
            original_calls = receipt.with_suffix(".calls").read_text().splitlines()
            assert len(original_calls) == 2
            session = await store.load(sid)
            now = [session.created_at]

            class AdvanceClockProposal(_Proposal):
                async def run(self, context, arguments):
                    result = await super().run(context, arguments)
                    now[0] += timedelta(seconds=120)
                    return result

            provider = _replay_provider()
            app = _proposal_app(
                reopen(),
                AdvanceClockProposal(ToolEffect.IDEMPOTENT, receipt, keyed=True),
                provider,
                lease_seconds=1.5,
                clock=lambda: now[0],
            )
            events = await _consume(
                app.resume(
                    ResumeRequest(session_id=sid, messages=[Message.text("user", "Any update?")])
                )
            )
            assert len(receipt.with_suffix(".calls").read_text().splitlines()) == 3
            assert not provider.requests
            assert any(e.type is EventType.SESSION_LIMIT_REACHED for e in events)
            stored = await app.session_store.load_events(sid)
            [completed] = [e for e in stored if e.type is EventType.TOOL_CALL_COMPLETED]
            assert completed.payload["tool_call_id"] == "proposal"
            assert completed.payload["replayed_after_recovery"] is True
            [failed] = [e for e in stored if e.type is EventType.TOOL_CALL_FAILED]
            assert failed.payload["tool_call_id"] == "sibling"
            assert failed.payload["result"]["structured"]["outcome_unknown"] is True
            assert (await app.session_store.load(sid)).status is SessionStatus.INTERRUPTED

    asyncio.run(scenario())


def test_crash_during_replay_does_not_replay_again(request, sqlite_resources, tmp_path):
    async def scenario():
        async with _stores("sqlite", request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "replay-crash-receipt"
            sid = "replay-crash-" + uuid4().hex
            await _crash_child(
                _child(store, "sqlite", request, receipt, sid, ToolEffect.IDEMPOTENT, "crash")
            )
            await _crash_child(
                _child(
                    store,
                    "sqlite",
                    request,
                    receipt,
                    sid,
                    ToolEffect.IDEMPOTENT,
                    "crash",
                    phase="resume",
                )
            )
            assert len(receipt.with_suffix(".calls").read_text().splitlines()) == 2
            provider = _replay_provider()
            app = _proposal_app(
                reopen(), _Proposal(ToolEffect.IDEMPOTENT, receipt), provider, lease_seconds=1.5
            )
            await _consume(
                app.resume(
                    ResumeRequest(session_id=sid, messages=[Message.text("user", "Any update?")])
                )
            )
            assert len(receipt.with_suffix(".calls").read_text().splitlines()) == 2
            assert (await app.session_store.load(sid)).status is SessionStatus.COMPLETED
            stored = await app.session_store.load_events(sid)
            failed = next(e for e in stored if e.type is EventType.TOOL_CALL_FAILED)
            assert failed.payload["result"]["structured"]["outcome_unknown"] is True

    asyncio.run(scenario())


def test_replay_that_current_policy_denies_is_not_replayed(request, sqlite_resources, tmp_path):
    async def scenario():
        async with _stores("sqlite", request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "replay-policy-receipt"
            sid = "replay-policy-" + uuid4().hex
            await _crash_child(
                _child(
                    store,
                    "sqlite",
                    request,
                    receipt,
                    sid,
                    ToolEffect.IDEMPOTENT,
                    "crash",
                    gate="flag",
                )
            )
            receipt.with_suffix(".deny").write_text("paused")
            provider = _replay_provider()
            app = _proposal_app(
                reopen(),
                _Proposal(ToolEffect.IDEMPOTENT, receipt),
                provider,
                gate="flag",
                lease_seconds=1.5,
            )
            await _consume(
                app.resume(
                    ResumeRequest(session_id=sid, messages=[Message.text("user", "Any update?")])
                )
            )
            # The original attempt started and its outcome is unknown; current policy
            # refusing a replay doesn't make it "not executed", so the round closes
            # with the unknown-outcome result rather than a replayed denial.
            assert len(receipt.with_suffix(".calls").read_text().splitlines()) == 1
            assert (await app.session_store.load(sid)).status is SessionStatus.COMPLETED
            stored = await app.session_store.load_events(sid)
            failed = next(e for e in stored if e.type is EventType.TOOL_CALL_FAILED)
            assert failed.payload["result"]["structured"]["outcome_unknown"] is True
            assert "replayed_after_recovery" not in failed.payload
            assert not any(e.type is EventType.TOOL_CALL_BLOCKED for e in stored)

    asyncio.run(scenario())


def test_replay_that_now_needs_approval_is_not_replayed(request, sqlite_resources, tmp_path):
    async def scenario():
        async with _stores("sqlite", request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "replay-approval-receipt"
            sid = "replay-approval-" + uuid4().hex
            await _crash_child(
                _child(
                    store,
                    "sqlite",
                    request,
                    receipt,
                    sid,
                    ToolEffect.IDEMPOTENT,
                    "crash",
                    gate="flag",
                )
            )
            receipt.with_suffix(".deny").write_text("approve")
            app = _proposal_app(
                reopen(),
                _Proposal(ToolEffect.IDEMPOTENT, receipt),
                _replay_provider(),
                gate="flag",
                lease_seconds=1.5,
            )
            events = await _consume(
                app.resume(
                    ResumeRequest(session_id=sid, messages=[Message.text("user", "Any update?")])
                )
            )
            # Current policy asks for approval, so the call isn't replayed: the round
            # closes with the unknown-outcome result and the session continues.
            assert len(receipt.with_suffix(".calls").read_text().splitlines()) == 1
            assert (await app.session_store.load(sid)).status is SessionStatus.COMPLETED
            stored = await app.session_store.load_events(sid)
            failed = next(e for e in stored if e.type is EventType.TOOL_CALL_FAILED)
            assert failed.payload["result"]["structured"]["outcome_unknown"] is True
            assert not any(event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED for event in events)

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["block", "short_circuit"])
def test_replay_that_a_hook_skips_is_not_marked_replayed(
    action, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores("sqlite", request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "replay-hook-receipt"
            sid = "replay-hook-" + uuid4().hex
            await _crash_child(
                _child(
                    store,
                    "sqlite",
                    request,
                    receipt,
                    sid,
                    ToolEffect.IDEMPOTENT,
                    "crash",
                    gate="hook",
                )
            )
            receipt.with_suffix(".hook").write_text(action)
            app = _proposal_app(
                reopen(),
                _Proposal(ToolEffect.IDEMPOTENT, receipt),
                _replay_provider(),
                gate="hook",
                lease_seconds=1.5,
            )
            await _consume(
                app.resume(
                    ResumeRequest(session_id=sid, messages=[Message.text("user", "Any update?")])
                )
            )
            assert len(receipt.with_suffix(".calls").read_text().splitlines()) == 1
            assert (await app.session_store.load(sid)).status is SessionStatus.COMPLETED
            stored = await app.session_store.load_events(sid)
            [terminal] = [
                e
                for e in stored
                if e.type
                in {
                    EventType.TOOL_CALL_COMPLETED,
                    EventType.TOOL_CALL_FAILED,
                    EventType.TOOL_CALL_BLOCKED,
                }
            ]
            assert terminal.type is (
                EventType.TOOL_CALL_BLOCKED if action == "block" else EventType.TOOL_CALL_COMPLETED
            )
            assert "replayed_after_recovery" not in terminal.payload

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["before_run", "during_replay"])
@pytest.mark.parametrize("next_step", ["resume", "operator_recovery"])
def test_failed_resume_after_takeover_keeps_round_for_next_continuation(
    failure, next_step, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores("sqlite", request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "failed-resume-receipt"
            sid = "failed-resume-" + uuid4().hex
            await _crash_child(
                _child(
                    store,
                    "sqlite",
                    request,
                    receipt,
                    sid,
                    ToolEffect.IDEMPOTENT,
                    "crash",
                    gate="flag",
                )
            )
            app = _proposal_app(
                reopen(),
                _Proposal(ToolEffect.IDEMPOTENT, receipt),
                _replay_provider(),
                gate="flag",
                lease_seconds=1.5,
            )
            # The takeover elects the replay, then this resume fails before the
            # replay dispatches: either before its run starts, or inside the run
            # when the policy check for the replay raises.
            if failure == "before_run":
                with pytest.raises(KeyError, match="Provider not registered"):
                    await _consume(
                        app.resume(
                            ResumeRequest(
                                session_id=sid,
                                messages=[Message.text("user", "Any update?")],
                                target=ModelTarget(provider_name="missing", model="fake-model"),
                            )
                        )
                    )
                expected_status = SessionStatus.INTERRUPTED
            else:
                receipt.with_suffix(".deny").write_text("raise")
                await _consume(
                    app.resume(
                        ResumeRequest(
                            session_id=sid, messages=[Message.text("user", "Any update?")]
                        )
                    )
                )
                receipt.with_suffix(".deny").unlink()
                expected_status = SessionStatus.FAILED
            assert (await app.session_store.load(sid)).status is expected_status
            stored = await app.session_store.load_events(sid)
            assert not any(
                e.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
                for e in stored
            )
            if next_step == "resume":
                # The next continuation replays the elected call once.
                await _consume(
                    app.resume(
                        ResumeRequest(session_id=sid, messages=[Message.text("user", "Again?")])
                    )
                )
                assert len(receipt.with_suffix(".calls").read_text().splitlines()) == 2
                assert (await app.session_store.load(sid)).status is SessionStatus.COMPLETED
                stored = await app.session_store.load_events(sid)
                [completed] = [e for e in stored if e.type is EventType.TOOL_CALL_COMPLETED]
                assert completed.payload["replayed_after_recovery"] is True
                return
            # Operator recovery doesn't dispatch tools; it closes the round as before.
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=sid, inactive_for_seconds=0, reason="operator"
                )
            )
            assert len(receipt.with_suffix(".calls").read_text().splitlines()) == 1
            stored = await app.session_store.load_events(sid)
            [failed] = [e for e in stored if e.type is EventType.TOOL_CALL_FAILED]
            assert failed.payload["result"]["structured"]["outcome_unknown"] is True
            assert "replayed_after_recovery" not in failed.payload

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
