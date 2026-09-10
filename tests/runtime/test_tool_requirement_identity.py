from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_active_invocation_execution_profiles import RequireApprovalPolicy
from tests.runtime.test_execution_admission_dispatch import _EvidenceRunner

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    ExecutionAdmissionCandidate,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ExecutionProfileBehaviorIdentity,
    ExecutionProfileComponentClass,
    ExecutionProfileMismatchError,
    InMemorySessionStore,
    Message,
    ResumeRequest,
    RunRequest,
    SQLiteSessionStore,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    ToolExecutableRequirement,
    ToolExecutionRequirement,
    ToolResult,
    ToolRoundRecoveryRequest,
    ToolRunnerCapabilityRequirement,
    ToolSpec,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers import ModelStreamEvent


def _identity(name):
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


def _requirement(change):
    executable = ToolExecutableRequirement(
        executable="other" if change == "executable" else "rg",
        probe_arguments=(
            None if change == "availability" else ("--version",) if change == "arguments" else ()
        ),
        accepted_exit_codes=(0, 1) if change == "exit_codes" else (0,),
    )
    native = ToolRunnerCapabilityRequirement(
        capability="other_search" if change == "native_capability" else "workspace_text_search_v1",
        minimum_evidence="available" if change == "native_minimum" else "live_verified",
    )
    return ToolExecutionRequirement(
        name="other_clause" if change == "clause_name" else "search",
        alternatives=(executable, native)
        if change == "alternative_order"
        else (native, executable),
    )


class _IdentityTool(Tool):
    def __init__(self, change):
        self.calls = []
        super().__init__(
            ToolSpec(
                name="search",
                execution_profile_identity=_identity("requirement-identity-tool"),
                execution_requirements=(_requirement(change),),
            )
        )

    async def run(self, ctx, args):
        self.calls.append(args)
        return ToolResult(content="done")


class _IdentityRunner(_EvidenceRunner):
    def __init__(self):
        super().__init__("hosted")
        self.snapshots = 0

    @property
    def execution_profile_identity(self):
        return _identity("requirement-identity-runner")

    def execution_admission_candidate(self):
        self.snapshots += 1
        now = datetime.now(UTC)
        return ExecutionAdmissionCandidate(
            candidate="hosted",
            evidence=ExecutionCapabilityEvidence(
                subject="hosted",
                environment_fingerprint="sha256:" + "1" * 64,
                claims=(
                    ExecutionCapabilityClaim.live_verified(
                        "workspace_text_search_v1",
                        observation="supported",
                        observed_at=now,
                        valid_until=now + timedelta(seconds=60),
                    ),
                ),
            ),
        )


def _composition(store, change, stream, entrance):
    tool, runner = _IdentityTool(change), _IdentityRunner()
    provider = ScriptedModelProvider(stream, name="fake")
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(
                name="hosted", execution_profile_identity=_identity("requirement-identity-env")
            ),
            runner=runner,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy()
        if entrance in {"approval", "manual_approval"}
        else None,
    )
    return app, tool, runner, provider


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("entrance", ["resume", "approval", "manual_approval", "manual_round"])
@pytest.mark.parametrize(
    "change",
    [
        "unchanged",
        "executable",
        "availability",
        "arguments",
        "exit_codes",
        "native_capability",
        "native_minimum",
        "clause_name",
        "alternative_order",
    ],
)
def test_reconstructed_requirement_identity_guards_continuation(
    backend, entrance, change, tmp_path, monkeypatch
):
    async def collect(stream):
        return [event async for event in stream]

    async def run():
        database = tmp_path / "requirement-identity.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(database)
        try:
            manual = entrance.startswith("manual_")
            failed_terminal = False
            append_events = store.append_events

            async def lose_terminal_result(session_id, events):
                nonlocal failed_terminal
                if not failed_terminal and any(
                    event.type is EventType.TOOL_CALL_COMPLETED for event in events
                ):
                    failed_terminal = True
                    raise RuntimeError("completed tool result persistence unavailable")
                await append_events(session_id, events)

            if manual:
                monkeypatch.setattr(store, "append_events", lose_terminal_result)
            initial_stream = (
                [
                    ModelStreamEvent.tool_call(id="call-1", name="search", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
                if entrance != "resume"
                else [ModelStreamEvent.completed({"finish_reason": "stop"})]
            )
            app, initial_tool, _, _ = _composition(store, "unchanged", initial_stream, entrance)
            initial = await collect(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="identity",
                        messages=[Message.text("user", "start")],
                    )
                )
            )
            approval = None
            if entrance in {"approval", "manual_approval"}:
                approval = next(
                    event.payload["approval"]
                    for event in initial
                    if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                )
            elif entrance == "resume":
                assert initial[-1].type is EventType.SESSION_COMPLETED
            if entrance == "manual_approval":
                initial = await collect(
                    app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id="identity",
                            approval_id=approval["approval_id"],
                            tool_round_id=approval["tool_round_id"],
                            tool_call_id=approval["tool_call_id"],
                            decision=ToolApprovalDecision.APPROVE,
                        )
                    )
                )
            if manual:
                assert failed_terminal
                assert initial_tool.calls == [{}]
                assert initial[-1].type is (
                    EventType.SESSION_INTERRUPTED
                    if entrance == "manual_approval"
                    else EventType.SESSION_FAILED
                )
                monkeypatch.setattr(store, "append_events", append_events)
            round_id = None
            if entrance == "manual_round":
                round_id = next(
                    event.payload["tool_round_id"]
                    for event in await store.load_events("identity")
                    if event.type is EventType.TOOL_CALL_STARTED
                )
            if isinstance(store, SQLiteSessionStore):
                await store.close()
                store = SQLiteSessionStore(database)
            before = await store.load("identity")
            checkpoint = await store.load_checkpoint("identity")
            prior_event_ids = {event.id for event in await store.load_events("identity")}
            assert before is not None
            app, tool, runner, provider = _composition(
                store,
                change,
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                entrance,
            )
            if entrance == "manual_round":
                continuation = app.recover_tool_round(
                    ToolRoundRecoveryRequest(
                        session_id="identity",
                        round_id=round_id,
                        tool_call_id="call-1",
                        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                        message="The original tool completed before persistence failed.",
                    )
                )
            elif entrance == "manual_approval":
                continuation = app.recover_tool_approval(
                    ToolApprovalRecoveryRequest(
                        session_id="identity",
                        approval_id=approval["approval_id"],
                        tool_round_id=approval["tool_round_id"],
                        tool_call_id=approval["tool_call_id"],
                        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                        message="The original tool completed before persistence failed.",
                    )
                )
            elif approval is not None:
                continuation = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id="identity",
                        approval_id=approval["approval_id"],
                        tool_round_id=approval["tool_round_id"],
                        tool_call_id=approval["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            else:
                continuation = app.resume(
                    ResumeRequest(
                        session_id="identity", messages=[Message.text("user", "continue")]
                    )
                )
            if change == "unchanged":
                events = await collect(continuation)
                assert events[-1].type is EventType.SESSION_COMPLETED
                assert len(provider.requests) == 1
                assert len(tool.calls) == int(entrance == "approval")
                assert runner.snapshots > 0
            else:
                with pytest.raises(ExecutionProfileMismatchError) as caught:
                    await collect(continuation)
                assert caught.value.changed_component_classes == (
                    ExecutionProfileComponentClass.DIRECT_TOOLS,
                )
                assert tool.calls == []
                assert provider.requests == []
                assert runner.snapshots == 0
                after = await store.load("identity")
                assert after is not None
                assert after.status is before.status
                assert after.run_epoch == before.run_epoch
                assert after.metadata == before.metadata
                assert await store.load_checkpoint("identity") == checkpoint
                new_events = [
                    event
                    for event in await store.load_events("identity")
                    if event.id not in prior_event_ids
                ]
                assert [event.type for event in new_events] == [
                    EventType.SESSION_EXECUTION_PROFILE_REJECTED
                ]
                decision = new_events[0].payload
                assert decision["decision"] == "rejected"
                assert decision["changed_component_classes"] == [
                    ExecutionProfileComponentClass.DIRECT_TOOLS.value
                ]
                assert decision["expected_profile"]["fingerprint"] == (
                    caught.value.expected_profile_fingerprint
                )
                assert decision["candidate_profile"]["fingerprint"] == (
                    caught.value.candidate_profile_fingerprint
                )
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())
