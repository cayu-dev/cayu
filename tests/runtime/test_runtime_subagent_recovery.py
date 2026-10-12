"""Subagent recovery contracts."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from tests.core._execution_profile_fixtures import (
    create_admitted_session,
    profiled_session_identity,
    tool_capability_ceiling_for_app,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import _tool_round_identity

import cayu.sessions.base as sessions_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import _invocation_scope as invocation_scope
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions.invocation import SessionExecutionSource
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryAction, IncompleteSessionRecoveryRequest
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.tools.base import Tool, ToolResult
from cayu.tools.exposure import ToolCapabilityCeiling
from cayu.tools.policy import ToolPolicyDecision, ToolPolicyResult
from cayu.tools.subagents import SubagentExecutionMode, SubagentSpec, SubagentTool


def _interaction_started_event(
    session_id: str,
    *,
    interaction_id: str,
    agent_name: str,
) -> Event:
    from cayu.sessions.interactions import InteractionStatus, InteractionSummaryEvidence

    event_id = f"{session_id}:interaction-started"
    started_at = datetime.now(UTC)
    return Event(
        id=event_id,
        type=EventType.INTERACTION_STARTED,
        session_id=session_id,
        interaction_id=interaction_id,
        timestamp=started_at,
        agent_name=agent_name,
        payload=InteractionSummaryEvidence(
            status=InteractionStatus.ACTIVE,
            start_event_id=event_id,
            started_at=started_at,
        ).model_dump(mode="json"),
    )


async def _seed_crashed_spawn_parent(
    store,
    *,
    parent_tool: Tool,
    parent_identity: SessionIdentity,
    parent_tool_capability_ceiling: ToolCapabilityCeiling,
    child_status,
    mode: str = "background",
    linkage: str = "correct",
    conflicting_linkage: bool = False,
    generated_identity_conflict: bool = False,
):
    """Reproduce the AGT-02 crash window: a parent with an in-progress ``subagent`` spawn tool round
    (STARTED, no terminal event) and a durably-created child.

    ``mode`` is the child's subagent execution mode.
    ``linkage`` controls the child's stamped idempotency_key: "correct" stamps the child's real key,
    "none" omits it, "wrong_round" stamps a key for a different tool round (both must fall back)."""
    from hashlib import sha256

    from cayu._validation import canonical_durable_json_bytes
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime import _tool_execution as tool_execution
    from cayu.runtime import _tool_round_recovery as tool_round_recovery
    from cayu.runtime._tool_effect_state import ToolEffectStateOwner
    from cayu.sessions._tool_effect_intent import ToolEffectIntent

    child_identity = SessionIdentity(provider_name="fake", model="fake-model")
    parent_interaction_id = "interaction-parent-crashed-spawn"
    parent_messages = [Message.text("user", "spawn a reviewer")]

    assert parent_identity.execution_profile is not None
    await create_admitted_session(
        store,
        request=RunRequest(
            agent_name="parent",
            session_id="parent",
            messages=parent_messages,
            tool_capability_ceiling=parent_tool_capability_ceiling,
        ),
        provider_name=parent_identity.provider_name,
        model=parent_identity.model,
        execution_profile=parent_identity.execution_profile,
        interaction_id=parent_interaction_id,
    )
    # Build the pending round first so the child can stamp the round-scoped idempotency_key.
    spawn_call = runtime_records.ToolCallRequest(
        id="call_spawn",
        name="subagent",
        arguments={"agent": "reviewer", "task": "review"},
    )
    active_checkpoint = await store.load_checkpoint("parent")
    assert active_checkpoint is not None
    checkpoint, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        active_checkpoint,
        agent_name="parent",
        environment_name=None,
        task_id=None,
        tool_calls=[spawn_call],
        policy_outcomes=[
            runtime_records.ToolCallPolicyOutcome(
                call=spawn_call,
                result=ToolPolicyResult(decision=ToolPolicyDecision.ALLOW),
                evidence=runtime_records.ToolPolicyEvidence.AUTHORITATIVE,
            )
        ],
        policy_state="planned",
        policy_context_version=1,
        structured_output=None,
        tool_round_identity=_tool_round_identity(),
    )

    subagent_meta = {
        "agent": "reviewer",
        "agent_name": "reviewer",
        "context_mode": "task_only",
        "mode": mode,
        "parent_session_id": "parent",
        "tool_call_id": "call_spawn",
    }
    if linkage == "correct":
        subagent_meta["idempotency_key"] = tool_execution.tool_idempotency_key(
            session_id="parent",
            tool_round_id=pending_round.tool_round_id,
            tool_call_id="call_spawn",
        )
    elif linkage == "wrong_round":
        subagent_meta["idempotency_key"] = tool_execution.tool_idempotency_key(
            session_id="parent", tool_round_id="a-different-round", tool_call_id="call_spawn"
        )
    child_session_id = "child"
    if generated_identity_conflict:
        from cayu.runtime._child_session_identity import (
            ChildSessionKind,
            generate_child_session_id,
        )

        child_session_id = generate_child_session_id(
            kind=ChildSessionKind.SUBAGENT,
            parent_session_id="parent",
            logical_spawn_id=subagent_meta["idempotency_key"],
        )
        subagent_meta["spawn_fingerprint"] = "sha256:" + ("0" * 64)
    await store.create(
        sessions_module.run_request_with_runtime_invocation(
            RunRequest(
                agent_name="reviewer",
                session_id=child_session_id,
                parent_session_id="parent",
                causal_budget_id="parent",
                messages=[Message.text("user", "review")],
                metadata={"subagent": subagent_meta},
            ),
            source=SessionExecutionSource.SUBAGENT,
        ),
        identity=child_identity,
    )
    await store.append_transcript_messages(
        child_session_id,
        [Message.text("user", "review"), Message.text("assistant", "looks good")],
    )
    await store.update_status(child_session_id, child_status)
    if mode == "foreground" and child_status in {
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
        SessionStatus.INTERRUPTED,
    }:
        await store.append_events(
            child_session_id,
            [
                Event(
                    type={
                        SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
                        SessionStatus.FAILED: EventType.SESSION_FAILED,
                        SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
                    }[child_status],
                    session_id=child_session_id,
                    agent_name="reviewer",
                    payload={},
                )
            ],
        )
    if conflicting_linkage:
        await store.create(
            sessions_module.run_request_with_runtime_invocation(
                RunRequest(
                    agent_name="reviewer",
                    session_id="conflicting-child",
                    parent_session_id="parent",
                    causal_budget_id="parent",
                    messages=[Message.text("user", "different review")],
                    metadata={"subagent": dict(subagent_meta)},
                ),
                source=SessionExecutionSource.SUBAGENT,
            ),
            identity=child_identity,
        )
        await store.update_status("conflicting-child", child_status)

    await store.checkpoint("parent", checkpoint)
    parent = await store.load("parent")
    assert parent is not None
    await ToolEffectStateOwner(store).begin(
        ToolEffectIntent(
            session_id=parent.id,
            session_instance_id=parent.instance_id,
            source_run_epoch=parent.run_epoch,
            interaction_id=parent_interaction_id,
            model_step_id=pending_round.model_step_id,
            model_attempt_id=pending_round.model_attempt_id,
            tool_round_id=pending_round.tool_round_id,
            tool_call_id=spawn_call.id,
            agent_name="parent",
            tool_name=spawn_call.name,
            idempotency_key=tool_execution.tool_idempotency_key(
                session_id=parent.id,
                tool_round_id=pending_round.tool_round_id,
                tool_call_id=spawn_call.id,
            ),
            execution_profile_fingerprint=parent_identity.execution_profile.fingerprint,
            schema_digest=sha256(
                canonical_durable_json_bytes(parent_tool.spec.input_schema, "effect_schema")
            ).hexdigest(),
            arguments_digest=sha256(
                canonical_durable_json_bytes(spawn_call.arguments, "effect_arguments")
            ).hexdigest(),
        ),
        run_epoch=parent.run_epoch,
    )
    await store.append_events(
        "parent",
        [
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="parent",
                interaction_id=parent_interaction_id,
                agent_name="parent",
                tool_name="subagent",
                payload={
                    "model_step_id": pending_round.model_step_id,
                    "model_attempt_id": pending_round.model_attempt_id,
                    "tool_round_id": pending_round.tool_round_id,
                    "tool_call_id": "call_spawn",
                    "idempotency_key": tool_execution.tool_idempotency_key(
                        session_id="parent",
                        tool_round_id=pending_round.tool_round_id,
                        tool_call_id="call_spawn",
                    ),
                    "arguments": {"agent": "reviewer", "task": "review"},
                },
            )
        ],
    )
    return pending_round


def _recovered_parent_tool_result(store):
    transcript = asyncio.run(store.load_transcript("parent"))
    tool_message = next(message for message in reversed(transcript) if message.role == "tool")
    return tool_message.content[0]


def _recovered_tool_event(result):
    return next(
        event
        for event in result.events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        and event.payload.get("recovered") is True
    )


def _recover_parent(
    child_status,
    *,
    mode: str = "background",
    linkage: str = "correct",
    conflicting_linkage: bool = False,
    generated_identity_conflict: bool = False,
    resume: bool = False,
):
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        subagent_tool = SubagentTool(
            app,
            agents={
                "reviewer": SubagentSpec(
                    agent_name="reviewer",
                    description="Review delegated work.",
                    mode=SubagentExecutionMode(mode),
                )
            },
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name=f"tests:runtime:recovery-subagent-tool:{mode}",
                behavior_version="1",
                implementation_version="1",
            ),
        )
        app.register_agent(AgentSpec(name="parent", model="fake-model"), tools=[subagent_tool])
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
        parent_identity = profiled_session_identity(
            provider_name="fake",
            model="fake-model",
            agent_name="parent",
            tools=[subagent_tool],
            causal_budget_id="parent",
            app=app,
        )
        await _seed_crashed_spawn_parent(
            store,
            parent_tool=subagent_tool,
            parent_identity=parent_identity,
            parent_tool_capability_ceiling=tool_capability_ceiling_for_app(
                app,
                agent_name="parent",
            ),
            child_status=child_status,
            mode=mode,
            linkage=linkage,
            conflicting_linkage=conflicting_linkage,
            generated_identity_conflict=generated_identity_conflict,
        )
        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="parent", reason="worker restart")
        )
        if resume:
            result = [
                event
                async for event in app.resume(
                    ResumeRequest(session_id="parent", messages=[Message.text("user", "continue")])
                )
            ]
        return store, result

    return asyncio.run(run())


@pytest.mark.parametrize("mode", ["foreground", "background"])
def test_recovery_selects_durable_effect_terminal_for_completed_child(mode):
    from cayu.runtime._tool_effect_state import ToolEffectStateOwner

    store, result = _recover_parent(SessionStatus.COMPLETED, mode=mode)
    event = _recovered_tool_event(result)

    async def load_record():
        session = await store.load("parent")
        return await ToolEffectStateOwner(store).resolve_call(
            session,
            tool_round_id=_tool_round_identity().tool_round_id,
            tool_call_id="call_spawn",
        )

    record = asyncio.run(load_record())
    assert record is not None
    assert record.state == "completed"
    assert record.terminal is not None
    durable_events = asyncio.run(store.load_events("parent"))
    terminal_events = [
        item
        for item in durable_events
        if item.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    ]
    assert len(terminal_events) == 1
    assert record.terminal.event_id == terminal_events[0].id
    assert terminal_events[0].type is event.type


def test_recovery_reattaches_completed_subagent_child():
    store, result = _recover_parent(SessionStatus.COMPLETED)
    recovered = _recovered_parent_tool_result(store)

    assert recovered.tool_call_id == "call_spawn"
    assert recovered.is_error is False
    assert recovered.structured["recovery_reason"] == "pending_tool_round_reattached_subagent"
    assert recovered.structured["child_session_id"] == "child"
    assert recovered.structured["parent_session_id"] == "parent"
    assert recovered.structured["status"] == "completed"
    assert recovered.structured["outcome_unknown"] is False
    # The retrieval guidance must name the real tool (subagent_result), not a nonexistent one.
    assert "subagent_result" in recovered.content
    assert "get_subagent_result" not in recovered.content
    # A completed re-attach must emit a COMPLETED (not FAILED) event, matching normal execution.
    assert _recovered_tool_event(result).type == EventType.TOOL_CALL_COMPLETED


def test_recovery_reattaches_interrupted_subagent_child_as_error():
    store, result = _recover_parent(SessionStatus.INTERRUPTED)
    recovered = _recovered_parent_tool_result(store)

    assert recovered.tool_call_id == "call_spawn"
    assert recovered.is_error is True
    assert recovered.structured["recovery_reason"] == "pending_tool_round_reattached_subagent"
    assert recovered.structured["child_session_id"] == "child"
    assert recovered.structured["status"] == "interrupted"
    # A terminal child is a known (if failed) outcome, not an unknown one.
    assert recovered.structured["outcome_unknown"] is False
    assert _recovered_tool_event(result).type == EventType.TOOL_CALL_FAILED


def test_recovery_without_child_linkage_preserves_unknown_effect():
    # A child without the exact spawn linkage cannot settle the parent's effect.
    store, result = _recover_parent(SessionStatus.COMPLETED, linkage="none")
    _assert_recovered_parent_effect_unknown(store, result)


def test_recovery_reattaches_completed_foreground_child():
    store, result = _recover_parent(SessionStatus.COMPLETED, mode="foreground")
    recovered = _recovered_parent_tool_result(store)

    assert recovered.structured["child_session_id"] == "child"
    assert recovered.structured["mode"] == "foreground"
    assert recovered.structured["status"] == "session.completed"
    assert recovered.structured["result_truncated"] is False
    assert recovered.content == "looks good"
    assert _recovered_tool_event(result).type == EventType.TOOL_CALL_COMPLETED


@pytest.mark.parametrize("mode", ["foreground", "background"])
@pytest.mark.parametrize(
    "child_status", [SessionStatus.PENDING, SessionStatus.RUNNING, SessionStatus.INTERRUPTING]
)
def test_recovery_keeps_nonterminal_child_effect_unknown(mode, child_status):
    store, result = _recover_parent(child_status, mode=mode)
    _assert_recovered_parent_effect_unknown(store, result)
    if mode == "foreground":
        assert IncompleteSessionRecoveryAction.PENDING_SUBAGENT in result.actions
        assert IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT not in result.actions
        assert result.pending_subagent_session_ids == ("child",)
        assert "before parent continuation" in result.message
    else:
        assert IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT in result.actions
        assert result.pending_subagent_session_ids == ()


@pytest.mark.parametrize(
    "child_status", [SessionStatus.PENDING, SessionStatus.RUNNING, SessionStatus.INTERRUPTING]
)
def test_resume_reports_exact_nonterminal_foreground_child(child_status):
    store, events = _recover_parent(child_status, mode="foreground", resume=True)
    assert events[-1].type is EventType.SESSION_INTERRUPTED
    assert events[-1].payload["recovery_required"] == "foreground_subagent"
    assert events[-1].payload["pending_subagent_session_ids"] == ["child"]
    # Public streams alias execution identifiers; durable evidence keeps the
    # exact internal call/round rather than granting public bearer authority.
    stored = next(
        event
        for event in reversed(asyncio.run(store.load_events("parent")))
        if event.type is EventType.SESSION_INTERRUPTED
    )
    assert stored.payload["recovery_required"] == "foreground_subagent"
    assert stored.payload["pending_subagent_session_ids"] == ["child"]
    assert stored.payload["tool_call_id"] == "call_spawn"
    assert stored.payload["tool_round_id"] == _tool_round_identity().tool_round_id
    assert not any(
        event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        for event in events
    )
    checkpoint = asyncio.run(store.load_checkpoint("parent"))
    from cayu.sessions._pending_tool_round_reader import pending_tool_round_from_checkpoint

    assert pending_tool_round_from_checkpoint(checkpoint) is not None


def _assert_recovered_parent_effect_unknown(store, result):
    from cayu.runtime._tool_effect_state import ToolEffectStateOwner

    async def verify():
        session = await store.load("parent")
        assert session.status is SessionStatus.INTERRUPTED
        record = await ToolEffectStateOwner(store).resolve_call(
            session,
            tool_round_id=_tool_round_identity().tool_round_id,
            tool_call_id="call_spawn",
        )
        assert record is not None
        assert record.state == "outcome_unknown"
        assert record.terminal is None
        checkpoint = await store.load_checkpoint("parent")
        assert pending_round_reader.pending_tool_round_from_checkpoint(checkpoint) is not None
        transcript = await store.load_transcript("parent")
        assert not any(message.role == "tool" for message in transcript)
        events = await store.load_events("parent")
        assert any(event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for event in events)
        assert not any(
            event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            for event in events
        )

    asyncio.run(verify())
    assert not any(
        event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        for event in result.events
    )


def test_recovery_ignores_child_stamped_for_a_different_round():
    # Round-scoping: a child whose idempotency_key encodes a different tool round (e.g. a same-tool_call_id
    # child from an earlier round) must NOT be re-attached to this round's pending spawn call.
    store, result = _recover_parent(SessionStatus.COMPLETED, linkage="wrong_round")
    _assert_recovered_parent_effect_unknown(store, result)


@pytest.mark.parametrize("mode", ["foreground", "background"])
def test_recovery_fails_closed_when_multiple_children_claim_one_spawn_identity(mode):
    store, result = _recover_parent(
        SessionStatus.COMPLETED,
        mode=mode,
        conflicting_linkage=True,
    )
    _assert_recovered_parent_effect_unknown(store, result)


@pytest.mark.parametrize("mode", ["foreground", "background"])
def test_recovery_rejects_single_new_identity_with_contradictory_spawn_evidence(mode):
    store, result = _recover_parent(
        SessionStatus.COMPLETED,
        mode=mode,
        generated_identity_conflict=True,
    )
    _assert_recovered_parent_effect_unknown(store, result)


async def _reattach_interrupted_spawn(*, tool_round_id, child_round_id):
    """Seed a parent + a background child stamped for `child_round_id`, then run the interrupt-close
    re-attach against a generic interrupted outcome for tool round `tool_round_id`."""
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime import _tool_execution as tool_execution

    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    subagent_tool = SubagentTool(
        app,
        agents={
            "reviewer": SubagentSpec(
                agent_name="reviewer",
                mode=SubagentExecutionMode.BACKGROUND,
            )
        },
    )
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[subagent_tool],
    )
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
    identity = SessionIdentity(provider_name="fake", model="fake-model")
    await store.create(
        RunRequest(agent_name="parent", session_id="parent", messages=[Message.text("user", "go")]),
        identity=identity,
    )
    key = tool_execution.tool_idempotency_key(
        session_id="parent", tool_round_id=child_round_id, tool_call_id="call_spawn"
    )
    await store.create(
        sessions_module.run_request_with_runtime_invocation(
            RunRequest(
                agent_name="reviewer",
                session_id="child",
                parent_session_id="parent",
                causal_budget_id="parent",
                messages=[Message.text("user", "review")],
                metadata={
                    "subagent": {
                        "agent": "reviewer",
                        "agent_name": "reviewer",
                        "context_mode": "task_only",
                        "mode": "background",
                        "parent_session_id": "parent",
                        "tool_call_id": "call_spawn",
                        "idempotency_key": key,
                    }
                },
            ),
            source=SessionExecutionSource.SUBAGENT,
        ),
        identity=identity,
    )
    await store.update_status("child", SessionStatus.INTERRUPTED)
    interrupted = runtime_records.ToolCallOutcome(
        call=runtime_records.ToolCallRequest(
            id="call_spawn",
            name="subagent",
            arguments={"agent": "reviewer", "task": "review"},
        ),
        result=ToolResult(
            content="Tool call interrupted before completion.",
            structured={"interrupted": True, "tool_call_id": "call_spawn"},
            is_error=True,
        ),
    )
    parent = await store.load("parent")
    assert parent is not None
    outcomes = await app._pending_tool_round_recovery.reattach_subagent_children_in_outcomes(
        session=parent,
        registered_agent=app._get_registered_agent("parent"),
        tool_round_id=tool_round_id,
        outcomes=[interrupted],
        source_checkpoint=await store.load_checkpoint(parent.id),
    )
    return outcomes[0].result


def test_interrupt_close_reattaches_background_subagent_child():
    # Factor-5 sweep: the live-interrupt close path records the parent->child linkage like crash recovery.
    result = asyncio.run(
        _reattach_interrupted_spawn(tool_round_id="round-1", child_round_id="round-1")
    )

    assert result.structured["recovery_reason"] == "pending_tool_round_reattached_subagent"
    assert result.structured["child_session_id"] == "child"
    assert result.structured["status"] == "interrupted"
    assert result.is_error is True


def test_interrupt_close_ignores_child_from_a_different_round():
    # Non-vacuity + round-scoping: a child stamped for a different round is not re-attached; the generic
    # interrupted outcome is returned unchanged.
    result = asyncio.run(
        _reattach_interrupted_spawn(tool_round_id="round-2", child_round_id="round-1")
    )

    assert result.structured == {"interrupted": True, "tool_call_id": "call_spawn"}
    assert "child_session_id" not in result.structured


class _CountSplitInterruptionCheckpointReadsStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self._children_queried = False
        self.checkpoint_reads_after_child_query: list[str] = []

    async def list_sessions(self, query=None):
        page = await super().list_sessions(query)
        if query is not None and query.parent_session_id is not None:
            self._children_queried = True
        return page

    async def load_checkpoint(self, session_id):
        if self._children_queried:
            self.checkpoint_reads_after_child_query.append(session_id)
        return await super().load_checkpoint(session_id)


async def _close_interrupted_spawn_and_collect_events(
    child_status,
    *,
    store: InMemorySessionStore | None = None,
):
    """Drive coordinator round closure for an interrupted `subagent` spawn whose background child is
    in `child_status`, returning the emitted events. Exercises the event-type derivation on that path."""
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime import _tool_execution as tool_execution
    from cayu.runtime import _tool_round_recovery as tool_round_recovery
    from cayu.runtime._tool_round_executor import InterruptedToolRoundRequest

    store = InMemorySessionStore() if store is None else store
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(FakeProvider([]), default=True)
    subagent_tool = SubagentTool(
        app,
        agents={
            "reviewer": SubagentSpec(
                agent_name="reviewer",
                mode=SubagentExecutionMode.BACKGROUND,
            )
        },
    )
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[subagent_tool],
    )
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
    identity = SessionIdentity(provider_name="fake", model="fake-model")
    interaction_id = "interaction-parent-interrupted-spawn"
    parent_messages = [Message.text("user", "go")]
    await store.create(
        RunRequest(agent_name="parent", session_id="parent", messages=parent_messages),
        identity=identity,
        interaction_started_event=_interaction_started_event(
            "parent",
            interaction_id=interaction_id,
            agent_name="parent",
        ),
        interaction_source_messages=parent_messages,
    )
    await store.replace_initial_transcript_messages(
        "parent",
        parent_messages,
        parent_messages,
        interaction_id=interaction_id,
    )
    round_identity = _tool_round_identity()
    tool_call = runtime_records.ToolCallRequest(
        id="call_spawn",
        name="subagent",
        arguments={"agent": "reviewer", "task": "review"},
    )
    checkpoint, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="parent",
        environment_name=None,
        task_id=None,
        tool_calls=[tool_call],
        policy_outcomes=None,
        structured_output=None,
        tool_round_identity=round_identity,
    )
    await store.checkpoint("parent", checkpoint)
    key = tool_execution.tool_idempotency_key(
        session_id="parent",
        tool_round_id=pending_round.tool_round_id,
        tool_call_id="call_spawn",
    )
    await store.create(
        sessions_module.run_request_with_runtime_invocation(
            RunRequest(
                agent_name="reviewer",
                session_id="child",
                parent_session_id="parent",
                causal_budget_id="parent",
                messages=[Message.text("user", "review")],
                metadata={
                    "subagent": {
                        "agent": "reviewer",
                        "agent_name": "reviewer",
                        "context_mode": "task_only",
                        "mode": "background",
                        "parent_session_id": "parent",
                        "tool_call_id": "call_spawn",
                        "idempotency_key": key,
                    }
                },
            ),
            source=SessionExecutionSource.SUBAGENT,
        ),
        identity=identity,
    )
    await store.append_transcript_messages(
        "child", [Message.text("user", "review"), Message.text("assistant", "done")]
    )
    await store.update_status("child", child_status)
    parent = await store.load("parent")
    invocation_scope._activate_session_interaction("parent", interaction_id)
    events = []
    async for event in app._pending_tool_round_recovery.close_interrupted_tool_round(
        InterruptedToolRoundRequest(
            session=parent,
            registered_agent=app._get_registered_agent("parent"),
            registered_environment=None,
            messages=list(parent_messages),
            tool_calls=[tool_call],
            tool_outcomes=[],
            tool_round_identity=round_identity,
            cancellation_artifacts=None,
            cancellation_artifacts_by_id=None,
        )
    ):
        events.append(event)
    return [event for event in events if event.payload.get("tool_call_id") == "call_spawn"]


def test_interrupt_close_emits_completed_event_for_reattached_completed_child():
    # A re-attached child that already COMPLETED must emit TOOL_CALL_COMPLETED, not a FAILED event whose
    # payload contradicts it (the crash path already derives the type from result.is_error).
    spawn_events = asyncio.run(_close_interrupted_spawn_and_collect_events(SessionStatus.COMPLETED))

    assert len(spawn_events) == 1
    event = spawn_events[0]
    assert event.type == EventType.TOOL_CALL_COMPLETED
    assert event.payload["result"]["is_error"] is False
    assert event.payload["result"]["structured"]["child_session_id"] == "child"
    assert (
        event.payload["result"]["structured"]["recovery_reason"]
        == "pending_tool_round_reattached_subagent"
    )


def test_interrupt_close_emits_failed_event_for_reattached_interrupted_child():
    spawn_events = asyncio.run(
        _close_interrupted_spawn_and_collect_events(SessionStatus.INTERRUPTED)
    )

    assert len(spawn_events) == 1
    event = spawn_events[0]
    assert event.type == EventType.TOOL_CALL_FAILED
    assert event.payload["result"]["is_error"] is True
    assert event.payload["result"]["structured"]["child_session_id"] == "child"


def test_interrupt_close_reuses_validated_checkpoint_for_subagent_reattachment():
    store = _CountSplitInterruptionCheckpointReadsStore()

    spawn_events = asyncio.run(
        _close_interrupted_spawn_and_collect_events(SessionStatus.INTERRUPTED, store=store)
    )

    assert len(spawn_events) == 1
    assert spawn_events[0].payload["result"]["structured"]["child_session_id"] == "child"
    # Assistant and final recovery publication each read the parent once;
    # background child reattachment reuses the supplied parent checkpoint.
    assert store.checkpoint_reads_after_child_query == ["parent", "parent"]
