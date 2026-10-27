"""The run summaries and human touches an agent reports to Cayu Cloud."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from tests.core._execution_profile_fixtures import profiled_session_identity
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    _reopen_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_foreground_child_restart import _RestartRecordingTool
from tests.core.test_foreground_subagent_recovery import _identity
from tests.core.test_foreground_subagent_recovery import _Provider as _SequencedProvider
from tests.core.test_session_fork_overlays import (
    CONTROLLER,
    HOST_AGENT,
    SUBJECT,
    ScriptedProvider,
    _completed,
    _host_app,
    _run_trunk,
)
from tests.core.test_user_input import _CountingTool, _crashed_user_input_resume_events

from cayu import AgentSpec, CayuApp, Message, ResumeRequest, RunRequest, cloud_reporting
from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.approvals import (
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRequest,
    UserInputRecoveryRequest,
    UserInputResponse,
)
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.budgets.pricing import default_price_book
from cayu.cloud_reporting import (
    CLOUD_ACTIVITY_MAX_LIMIT,
    CloudActivityCursorInvalid,
    CloudActivityCursorStale,
    CloudActivityPage,
    CloudRunStatus,
    read_cloud_activity,
    read_cloud_contract,
)
from cayu.evals import EvalCase, ScriptedModelProvider, run_eval_case
from cayu.messages import MessageRole
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.releases import ReleaseIdentity
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import ForkOverlay, ForkSessionRequest, InterruptSessionRequest
from cayu.sessions._run_identity import (
    ATTENTION_HANDOFF_METADATA_KEY,
    EVAL_TRIAL_METADATA_KEY,
    attention_withheld,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.messaging import (
    EnqueueSessionMessageRequest,
    SessionMessageActionRequest,
    SessionMessageConditions,
    SessionMessageDeliveryMode,
    SessionMessageQuery,
)
from cayu.sessions.pending_action_contracts import PendingActionQuery, PendingActionRecord
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import SecretRedactor

__all__ = ["conformance_postgres_dsn", "session_store_case"]

QUESTION = "Which ledger holds the question-canary accrual?"
ARGUMENT = "journal-argument-canary"
ANSWER = "answer-canary ledger 6200"
REVIEWER = ResolutionActor(subject="priya", source=ResolutionActorSource.REQUEST)
# A model the bundled price book prices, so summaries carry a real estimate.
PRICE = next(
    price
    for price in default_price_book().prices
    if price.provider_name == "anthropic" and price.match == "exact"
)
USAGE = {"input_tokens": 1_000, "output_tokens": 200}
T0 = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)


async def _deny(app: CayuApp, session_id: str) -> None:
    action = await _pending(app, session_id)
    await _drain(
        app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=action.approval_id,
                tool_round_id=action.round_id,
                tool_call_id=action.tool_call_id,
                decision=ToolApprovalDecision.DENY,
                resolved_by=REVIEWER,
            )
        )
    )


@pytest.mark.store_semantics
def test_reusing_a_session_id_retains_separate_runs_and_touches(session_store_case) -> None:
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "reused", "approver")
            await _deny(app, "reused")
            original = await read_cloud_activity(app)
            first_session = await store.load("reused")
            await store.delete_session("reused")
            await _start(app, "reused", "writer")
            store = await _reopen_store(session_store_case, store)
            app = _app(store)
            replacement = await read_cloud_activity(app, cursor=original.next_cursor)
            replay = await read_cloud_activity(app)
            second_session = await store.load("reused")
            assert first_session is not None and second_session is not None
            assert first_session.instance_id != second_session.instance_id
            assert first_session.instance_id not in original.model_dump_json()
            assert second_session.instance_id not in replacement.model_dump_json()
            return original, replacement, replay
        finally:
            await _close_store(store)

    original, replacement, replay = asyncio.run(scenario())
    (old,) = original.runs
    (new,) = replacement.runs
    (denial,) = original.touches
    assert old.session_id == new.session_id == "reused"
    assert old.run_id != new.run_id
    assert old.root_run_id == old.run_id == denial.run_id == denial.root_run_id
    assert new.root_run_id == new.run_id
    assert old.parent_run_id is new.parent_run_id is denial.parent_run_id is None
    assert denial.kind == "deny"
    assert old.cost.amount > 0 and new.cost.amount > 0
    assert replacement.touches == ()
    retained = {run.run_id: run for page in (original, replacement) for run in page.runs}
    assert len(retained) == 2
    assert sum(run.cost.amount for run in retained.values()) == old.cost.amount + new.cost.amount
    assert replay.runs == replacement.runs
    assert replay.touches == ()


@pytest.mark.store_semantics
def test_deleted_ancestors_never_rebind_children_or_touches(session_store_case) -> None:
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "root", "writer")
            for name, parent, agent in (
                ("middle", "root", "writer"),
                ("leaf", "middle", "approver"),
            ):
                await _drain(
                    app.run(
                        RunRequest(
                            session_id=name,
                            parent_session_id=parent,
                            agent_name=agent,
                            messages=[Message.text("user", "go")],
                        )
                    )
                )
            before = await read_cloud_activity(app)
            await store.delete_session("middle")
            # The replacement intermediate parent shares the same root invocation.
            await _drain(
                app.run(
                    RunRequest(
                        session_id="middle",
                        parent_session_id="root",
                        agent_name="writer",
                        messages=[Message.text("user", "go")],
                    )
                )
            )
            replaced_middle = await read_cloud_activity(app, cursor=before.next_cursor)
            await store.delete_session("root")
            await _start(app, "root", "writer")
            replacement = await read_cloud_activity(app, cursor=replaced_middle.next_cursor)
            store = await _reopen_store(session_store_case, store)
            app = _app(store)
            await _deny(app, "leaf")
            settled = await read_cloud_activity(app, cursor=replacement.next_cursor)
            await _drain(
                app.fork_session(ForkSessionRequest(source_session_id="leaf", session_id="copy"))
            )
            forked = await read_cloud_activity(app, cursor=settled.next_cursor)
            return before, replaced_middle, replacement, settled, forked
        finally:
            await _close_store(store)

    before, replaced_middle, replacement, settled, forked = asyncio.run(scenario())
    old = {run.session_id: run for run in before.runs}
    middle = next(run for run in replaced_middle.runs if run.session_id == "middle")
    (new_root,) = replacement.runs
    (leaf,) = settled.runs
    (denial,) = settled.touches
    (fork,) = forked.runs
    assert middle.run_id != old["middle"].run_id
    assert middle.root_run_id == old["root"].run_id != new_root.run_id
    assert leaf.run_id == denial.run_id == old["leaf"].run_id
    assert leaf.root_run_id == denial.root_run_id == old["root"].run_id
    assert leaf.parent_run_id == denial.parent_run_id == old["middle"].run_id
    assert leaf.parent_session_id == "middle"  # Historical alias survives detachment.
    assert fork.parent_run_id == leaf.run_id
    assert fork.root_run_id == old["root"].run_id


@pytest.mark.store_semantics
def test_root_and_child_lineage_survives_protocol_word_secrets(session_store_case) -> None:
    from cayu.sessions._run_identity import session_run_id, session_run_lineage
    from cayu.sessions.records import SessionStatus

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(
                store, secret_redactor=SecretRedactor(["cayu", "session", "run_", "is_root"])
            )
            await _start(app, "root", "writer")
            await _drain(
                app.run(
                    RunRequest(
                        session_id="child",
                        parent_session_id="root",
                        agent_name="writer",
                        messages=[Message.text("user", "go")],
                        metadata={"note": "user data remains present"},
                    )
                )
            )
            parent = await store.load("root")
            child = await store.load("child")
            assert parent is not None and child is not None
            assert parent.status is child.status is SessionStatus.COMPLETED
            assert session_run_lineage(parent) == (session_run_id(parent), None)
            assert session_run_lineage(child) == (session_run_id(parent), session_run_id(parent))
            assert child.metadata["note"] == "user data remains present"
        finally:
            await _close_store(store)

    asyncio.run(scenario())


def test_legacy_missing_lineage_is_unknown_even_after_parent_reuse(monkeypatch) -> None:
    from cayu.sessions._run_identity import RUN_LINEAGE_METADATA_KEY

    async def scenario():
        app = _app()
        await _start(app, "root", "writer")
        await _drain(
            app.run(
                RunRequest(
                    session_id="child",
                    parent_session_id="root",
                    agent_name="writer",
                    messages=[Message.text("user", "go")],
                )
            )
        )
        await app.session_store.delete_session("root")
        await _start(app, "root", "writer")
        load = app.session_store.load

        async def legacy_load(session_id):
            session = await load(session_id)
            if session is not None:
                session.metadata.pop(RUN_LINEAGE_METADATA_KEY, None)
            return session

        monkeypatch.setattr(app.session_store, "load", legacy_load)
        return await read_cloud_activity(app)

    page = asyncio.run(scenario())
    assert {run.session_id for run in page.runs} == {"root", "child"}
    assert all(
        run.run_id and run.root_run_id is None and run.parent_run_id is None for run in page.runs
    )


@pytest.mark.parametrize("fork", [False, True])
def test_callers_cannot_supply_run_lineage(fork: bool) -> None:
    from cayu.sessions._run_identity import RUN_LINEAGE_METADATA_KEY

    with pytest.raises(ValueError, match="runtime-owned run-lineage"):
        if fork:
            ForkSessionRequest(source_session_id="root", metadata={RUN_LINEAGE_METADATA_KEY: {}})
        else:
            RunRequest(agent_name="writer", messages=[], metadata={RUN_LINEAGE_METADATA_KEY: {}})


@pytest.mark.store_semantics
def test_direct_store_forks_bind_lineage_inside_the_creation_fence(session_store_case) -> None:
    from cayu.sessions._run_identity import session_run_id, session_run_lineage
    from cayu.sessions.forks import fork_session_invocation
    from cayu.sessions.records import Session, SessionIdentity, SessionStatus

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            source = await store.create(
                RunRequest(agent_name="writer", session_id="source", messages=[]),
                identity=SessionIdentity(provider_name="test", model="test"),
            )
            fork = Session(
                id="fork",
                agent_name=source.agent_name,
                provider_name=source.provider_name,
                model=source.model,
                parent_session_id=source.id,
                causal_budget_id=source.causal_budget_id,
                invocation=fork_session_invocation(source),
                # A native caller can copy the source, including its root marker.
                # The store must derive the fork's lineage from its fenced source.
                metadata=source.metadata,
            )
            created = await store.create_fork(
                source_session_id=source.id,
                fork=fork,
                source_statuses={SessionStatus.PENDING},
                transcript_cursor=None,
                checkpoint_transform=None,
                system_prompt_replacement=None,
                expected_source_run_epoch=source.run_epoch,
            )
            await store.delete_session(source.id)
            replacement = await store.create(
                RunRequest(agent_name="writer", session_id="source", messages=[]),
                identity=SessionIdentity(provider_name="test", model="test"),
            )
            store = await _reopen_store(session_store_case, store)
            restored = await store.load(created.id)
            assert restored is not None
            assert session_run_id(restored) == session_run_id(created)
            assert session_run_lineage(restored) == (session_run_id(source), session_run_id(source))
            assert session_run_id(replacement) != session_run_id(source)
            assert session_run_lineage(replacement) == (session_run_id(replacement), None)
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["classification", "accounting"])
def test_replacement_during_read_never_mixes_incarnations(monkeypatch, phase: str) -> None:
    async def scenario():
        app = _app()
        await _start(app, "reused", "approver")
        await _deny(app, "reused")
        before = await read_cloud_activity(app)
        replaced = False

        async def replace():
            nonlocal replaced
            if not replaced:
                replaced = True
                await app.session_store.delete_session("reused")
                await _start(app, "reused", "writer")

        if phase == "classification":
            classify = cloud_reporting.classify_supervision_anchors

            async def classify_after_replacement(store, records):
                await replace()
                return await classify(store, records)

            monkeypatch.setattr(
                cloud_reporting, "classify_supervision_anchors", classify_after_replacement
            )
        else:
            accounting = app.session_store.read_usage_accounting

            async def accounting_after_replacement(*args, **kwargs):
                await replace()
                return await accounting(*args, **kwargs)

            monkeypatch.setattr(
                app.session_store, "read_usage_accounting", accounting_after_replacement
            )
        raced = await read_cloud_activity(app)
        after = await read_cloud_activity(app, cursor=raced.next_cursor)
        return before, raced, after

    before, raced, after = asyncio.run(scenario())
    (old,) = before.runs
    (new,) = after.runs
    assert old.run_id != new.run_id
    assert all(touch.run_id == old.run_id for touch in raced.touches)
    assert all(run.run_id == new.run_id for run in raced.runs)
    if phase == "classification":
        assert raced.touches == ()
    else:
        assert raced.touches == before.touches
        assert raced.runs == ()


@pytest.mark.store_semantics
@pytest.mark.parametrize("phase", ["usage", "cost", "outcome"])
@pytest.mark.parametrize("replace", [False, True])
def test_disappearance_during_summary_omits_only_that_incarnation(
    session_store_case, monkeypatch, phase: str, replace: bool
) -> None:
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "removed", "writer")
            await _start(app, "survivor", "writer")
            before = await read_cloud_activity(app)
            changed = False
            method = {
                "usage": "read_usage_accounting",
                "cost": "read_cost_accounting",
                "outcome": "summarize_outcome",
            }[phase]
            original = getattr(store, method)

            async def while_reading(*args, **kwargs):
                nonlocal changed
                if not changed and (phase != "outcome" or args[0] == "removed"):
                    changed = True
                    await store.delete_session("removed")
                    if replace:
                        await _start(app, "removed", "writer")
                return await original(*args, **kwargs)

            monkeypatch.setattr(store, method, while_reading)
            raced = await read_cloud_activity(app)
            assert changed
            assert [run.session_id for run in raced.runs] == ["survivor"]
            assert raced.runs[0] == next(run for run in before.runs if run.session_id == "survivor")
            next_page = await read_cloud_activity(app, cursor=raced.next_cursor)
            if replace:
                (new,) = next_page.runs
                old = next(run for run in before.runs if run.session_id == "removed")
                assert new.session_id == old.session_id and new.run_id != old.run_id
            else:
                assert next_page.runs == ()
        finally:
            await _close_store(store)

    asyncio.run(scenario())


def test_summary_does_not_hide_unrelated_outcome_errors(monkeypatch) -> None:
    async def scenario():
        app = _app()
        await _start(app, "retained", "writer")

        async def broken_outcome(session_id):
            raise KeyError("broken outcome index")

        monkeypatch.setattr(app.session_store, "summarize_outcome", broken_outcome)
        with pytest.raises(KeyError, match="broken outcome index"):
            await read_cloud_activity(app)

    asyncio.run(scenario())


class _Provider(ScriptedModelProvider):
    # The same identity in every process, so a restarted agent continues its sessions.
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:cloud-reporting:provider", behavior_version="1", implementation_version="1"
        )


class _Record(Tool):
    spec = ToolSpec(
        name="record",
        description="Record a journal entry.",
        input_schema={"type": "object", "properties": {"note": {"type": "string"}}},
        effect=ToolEffect.IDEMPOTENT,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:cloud-reporting:record", behavior_version="1", implementation_version="1"
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="Recorded.")


class _ApproveRecords(ToolPolicy):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:cloud-reporting:approval-policy",
            behavior_version="1",
            implementation_version="1",
        )

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)


def _respond(request: ModelRequest) -> list[ModelStreamEvent]:
    tools = {tool["name"] for tool in request.tools}
    if any(message.role is MessageRole.TOOL for message in request.messages) or not tools:
        return [
            ModelStreamEvent.text_delta("Done."),
            ModelStreamEvent.completed({"finish_reason": "stop", "usage": USAGE}),
        ]
    call = (
        ModelStreamEvent.tool_call(id="call-1", name="ask_user", arguments={"question": QUESTION})
        if "ask_user" in tools
        else ModelStreamEvent.tool_call(id="call-1", name="record", arguments={"note": ARGUMENT})
    )
    return [call, ModelStreamEvent.completed({"finish_reason": "tool_calls", "usage": USAGE})]


def _app(store=None, *, release: ReleaseIdentity | None = None, secret_redactor=None) -> CayuApp:
    app = CayuApp(
        session_store=store,
        release_identity=release,
        enable_logging=False,
        secret_redactor=secret_redactor,
    )
    app.register_provider(
        _Provider(name=PRICE.provider_name, response_factory=_respond), default=True
    )
    app.register_agent(AgentSpec(name="writer", model=PRICE.model))
    app.register_agent(AgentSpec(name="asker", model=PRICE.model), tools=[UserInputTool()])
    app.register_agent(
        AgentSpec(name="approver", model=PRICE.model),
        tools=[_Record()],
        tool_policy=_ApproveRecords(),
    )
    return app


async def _drain(stream) -> None:
    _ = [event async for event in stream]


async def _start(app: CayuApp, session_id: str, agent: str) -> None:
    await _drain(
        app.run(
            RunRequest(
                agent_name=agent, session_id=session_id, messages=[Message.text("user", "go")]
            )
        )
    )


async def _pending(app: CayuApp, session_id: str) -> PendingActionRecord:
    page = await app.session_store.query_pending_actions(PendingActionQuery(session_id=session_id))
    (action,) = page.actions
    return action


def _no_content(page: CloudActivityPage) -> None:
    text = page.model_dump_json()
    for canary in (QUESTION, ARGUMENT, ANSWER):
        assert canary not in text


def test_a_page_reports_each_run_and_who_stepped_in_without_content() -> None:
    async def scenario():
        app = _app()
        await _start(app, "write", "writer")
        await _start(app, "approve", "approver")
        await _start(app, "ask", "asker")
        await _start(app, "stop", "approver")
        waiting = await read_cloud_activity(app)
        expected_cost = (await app.get_session_cost("write", default_price_book())).total_cost

        approval = await _pending(app, "approve")
        question = await _pending(app, "ask")
        await _drain(
            app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="approve",
                    approval_id=approval.approval_id,
                    tool_round_id=approval.round_id,
                    tool_call_id=approval.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                    reason="Matches the invoice.",
                    resolved_by=REVIEWER,
                )
            )
        )
        await _drain(
            app.resolve_user_input(
                UserInputResponse(session_id="ask", input_id=question.input_id, answer=ANSWER)
            )
        )
        await _drain(
            app.interrupt_session(
                InterruptSessionRequest(
                    session_id="stop", reason="Wrong source file.", requested_by=REVIEWER
                )
            )
        )
        settled = await read_cloud_activity(app, cursor=waiting.next_cursor)
        idle = await read_cloud_activity(app, cursor=settled.next_cursor)
        return waiting, settled, idle, expected_cost, approval, question

    waiting, settled, idle, expected_cost, approval, question = asyncio.run(scenario())

    runs = {run.session_id: run for run in waiting.runs}
    assert list(runs) == ["write", "approve", "ask", "stop"]
    assert waiting.touches == ()
    assert waiting.has_more is False
    write = runs["write"]
    assert (write.status, write.reason, write.agent_name) == (
        CloudRunStatus.COMPLETED,
        "completed",
        "writer",
    )
    assert (write.root_session_id, write.parent_session_id, write.source) == (
        "write",
        None,
        "sdk_run",
    )
    assert (write.forked, write.fork_controller, write.started_by) == (False, None, None)
    assert write.ended_at is not None and write.ended_at >= write.started_at
    assert (write.model_steps, write.tool_calls, write.waiting_for) == (1, 0, ())
    assert write.cost.amount == expected_cost > 0
    assert (write.cost.currency, write.cost.complete) == ("USD", True)
    assert write.release_id is None

    approve = runs["approve"]
    assert (approve.status, approve.reason, approve.ended_at) == (
        CloudRunStatus.WAITING_APPROVAL,
        "tool_approval_required",
        None,
    )
    (item,) = approve.waiting_for
    assert (item.kind, item.tool_name, item.attention_id, item.child_session_id) == (
        "tool_approval",
        "record",
        approval.attention_id,
        None,
    )
    ask = runs["ask"]
    assert ask.status is CloudRunStatus.WAITING_INPUT
    assert [(item.kind, item.attention_id) for item in ask.waiting_for] == [
        ("user_input", question.attention_id)
    ]
    _no_content(waiting)

    # Settling each item reports the touch and the run's new state; the
    # untouched run is not reported again.
    settled_runs = {run.session_id: run for run in settled.runs}
    assert set(settled_runs) == {"approve", "ask", "stop"}
    assert settled_runs["approve"].status is CloudRunStatus.COMPLETED
    assert settled_runs["approve"].waiting_for == ()
    assert settled_runs["approve"].tool_calls == 1
    assert settled_runs["ask"].status is CloudRunStatus.COMPLETED
    stop = settled_runs["stop"]
    assert (stop.status, stop.reason) == (CloudRunStatus.INTERRUPTED, "operator_requested")
    assert stop.ended_at is not None
    touches = {touch.session_id: touch for touch in settled.touches}
    assert [touch.kind for touch in settled.touches] == ["approve", "answer", "interrupt"]
    approved = touches["approve"]
    assert (approved.outcome, approved.reason, approved.tool_name) == (
        "approved",
        "Matches the invoice.",
        "record",
    )
    assert approved.by is not None
    assert (approved.by.subject, approved.by.source) == ("priya", "request")
    assert approved.attention_id == approval.attention_id
    assert approved.requested_at is not None and approved.settled_at >= approved.requested_at
    assert touches["ask"].attention_id == question.attention_id
    assert (touches["stop"].reason, touches["stop"].outcome) == (
        "Wrong source file.",
        "interrupted",
    )
    assert all(touch.stepped_in for touch in settled.touches)
    assert len({touch.touch_id for touch in settled.touches}) == 3
    _no_content(settled)

    assert (idle.runs, idle.touches, idle.has_more) == ((), (), False)
    assert idle.next_cursor == settled.next_cursor


def test_an_expired_approval_is_reported_but_nobody_stepped_in() -> None:
    clock = {"now": T0}

    async def scenario():
        app = CayuApp(clock=lambda: clock["now"], enable_logging=False)
        app.register_provider(
            _Provider(name=PRICE.provider_name, response_factory=_respond), default=True
        )
        app.register_agent(
            AgentSpec(name="approver", model=PRICE.model),
            tools=[_Record()],
            tool_policy=AlwaysRequireApprovalToolPolicy(expires_in_seconds=60),
        )
        await _start(app, "late", "approver")
        approval = await _pending(app, "late")
        clock["now"] = T0 + timedelta(minutes=5)
        await _drain(
            app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="late",
                    approval_id=approval.approval_id,
                    tool_round_id=approval.round_id,
                    tool_call_id=approval.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                    resolved_by=REVIEWER,
                )
            )
        )
        return await read_cloud_activity(app)

    page = asyncio.run(scenario())

    (touch,) = page.touches
    assert (touch.kind, touch.stepped_in) == ("expire", False)
    assert touch.by is not None and touch.by.source == "system"


def test_small_pages_cover_every_run_once_and_reject_foreign_cursors() -> None:
    async def scenario():
        app = _app()
        for index in range(5):
            await _start(app, f"run-{index}", "writer")
        pages: list[CloudActivityPage] = []
        cursor = None
        while True:
            page = await read_cloud_activity(app, cursor=cursor, limit=2)
            pages.append(page)
            cursor = page.next_cursor
            if not page.has_more:
                break
        for invalid in ("-1", "01", "1.0", " 1", "seven", str(MAX_DURABLE_JSON_INTEGER + 1)):
            with pytest.raises(CloudActivityCursorInvalid, match="Invalid activity cursor"):
                await read_cloud_activity(app, cursor=invalid)
        for limit in (0, CLOUD_ACTIVITY_MAX_LIMIT + 1):
            with pytest.raises(ValueError, match="limit"):
                await read_cloud_activity(app, limit=limit)
        return pages

    pages = asyncio.run(scenario())

    assert all(len(page.runs) <= 2 for page in pages)
    reported = [run.session_id for page in pages for run in page.runs]
    assert reported == [f"run-{index}" for index in range(5)]
    assert [page.has_more for page in pages] == [True, True, False]
    assert all(run.status is CloudRunStatus.COMPLETED for page in pages for run in page.runs)


def test_a_page_that_reaches_its_scan_budget_continues_on_the_next(monkeypatch) -> None:
    # Each run here writes three events the read collects; four fit a page.
    monkeypatch.setattr(cloud_reporting, "_SCAN_BUDGET", 4)
    monkeypatch.setattr(cloud_reporting, "_SCAN_BATCH", 2)

    async def scenario():
        app = _app()
        await _start(app, "first", "writer")
        await _start(app, "second", "writer")
        full = await read_cloud_activity(app)
        rest = await read_cloud_activity(app, cursor=full.next_cursor)
        return full, rest

    full, rest = asyncio.run(scenario())

    assert ([run.session_id for run in full.runs], full.has_more) == (["first", "second"], True)
    assert ([run.session_id for run in rest.runs], rest.has_more) == (["second"], False)


@pytest.mark.store_semantics
def test_a_restarted_agent_continues_from_the_stored_cursor(session_store_case) -> None:
    first = ReleaseIdentity(id="release-14")
    second = ReleaseIdentity(id="release-15")

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            before = _app(store, release=first)
            await _start(before, "continued", "writer")
            stored = await read_cloud_activity(before)
            # A new process, on the next Release, over the same database.
            store = await _reopen_store(session_store_case, store)
            after = _app(store, release=second)
            await _start(after, "new", "writer")
            await _drain(
                after.resume(
                    ResumeRequest(
                        session_id="continued", messages=[Message.text("user", "once more")]
                    )
                )
            )
            resumed = await read_cloud_activity(after, cursor=stored.next_cursor)
            caught_up = await read_cloud_activity(after, cursor=resumed.next_cursor)
            replayed = await read_cloud_activity(after)
            return stored, resumed, caught_up, replayed
        finally:
            await _close_store(store)

    stored, resumed, caught_up, replayed = asyncio.run(scenario())

    assert [(run.session_id, run.release_id) for run in stored.runs] == [
        ("continued", "release-14")
    ]
    assert [(run.session_id, run.release_id, run.model_steps) for run in resumed.runs] == [
        ("new", "release-15", 1),
        ("continued", "release-15", 2),
    ]
    assert (caught_up.runs, caught_up.has_more) == ((), False)
    # Reading from the start again gives the same current summaries, once each.
    assert sorted(run.session_id for run in replayed.runs) == ["continued", "new"]


def test_forks_are_reported_when_created_and_name_their_controller() -> None:
    async def scenario():
        app, _journal = _host_app(
            InMemorySessionStore(),
            ScriptedProvider([_completed("closed"), _completed("finding")]),
            host_tools=("post_journal",),
        )
        await _run_trunk(app)
        trunk = await read_cloud_activity(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(
            controller.fork(
                ForkSessionRequest(
                    source_session_id="trunk",
                    session_id="compass-copy",
                    overlay=ForkOverlay(controller=CONTROLLER, instructions="Look again."),
                )
            )
        )
        await _drain(
            app.fork_session(ForkSessionRequest(source_session_id="trunk", session_id="host-copy"))
        )
        # Neither fork has run; creating it is enough to report it.
        created = await read_cloud_activity(app, cursor=trunk.next_cursor)
        await _drain(
            controller.resume(
                ResumeRequest(session_id="compass-copy", messages=[Message.text("user", "go")])
            )
        )
        ran = await read_cloud_activity(app, cursor=created.next_cursor)
        return trunk, created, ran

    trunk_page, created, ran = asyncio.run(scenario())

    (trunk,) = trunk_page.runs
    assert (trunk.session_id, trunk.forked, trunk.fork_controller, trunk.agent_name) == (
        "trunk",
        False,
        None,
        HOST_AGENT,
    )
    forks = {run.session_id: run for run in created.runs}
    assert list(forks) == ["compass-copy", "host-copy"]
    for run in forks.values():
        assert (run.source, run.forked) == ("fork", True)
        assert (run.parent_session_id, run.root_session_id) == ("trunk", "trunk")
        assert run.root_run_id == run.parent_run_id == trunk.run_id
        assert run.run_id != trunk.run_id
        # A fork that has not run carries its source's status and has not ended itself.
        assert (run.status, run.ended_at, run.model_steps) == (CloudRunStatus.COMPLETED, None, 0)
    assert forks["compass-copy"].fork_controller == CONTROLLER
    assert forks["host-copy"].fork_controller is None
    (copy,) = ran.runs
    assert (copy.session_id, copy.fork_controller, copy.model_steps) == (
        "compass-copy",
        CONTROLLER,
        1,
    )
    assert copy.status is CloudRunStatus.COMPLETED and copy.ended_at is not None


def _delegating_app(store=None) -> CayuApp:
    """``root`` waits on ``middle``, which waits on ``leaf``; ``leaf`` asks for approval, then input."""

    def call(name: str, arguments: dict) -> list[ModelStreamEvent]:
        return [
            ModelStreamEvent.tool_call(id="call-1", name=name, arguments=arguments),
            ModelStreamEvent.completed(),
        ]

    def finished(name: str) -> list[ModelStreamEvent]:
        return [ModelStreamEvent.text_delta(f"{name} finished"), ModelStreamEvent.completed()]

    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        _SequencedProvider(
            [
                call("subagent", {"agent": "middle", "task": "work"}),
                call("subagent", {"agent": "leaf", "task": "work"}),
                call("record", {"value": 7}),
                call("ask_user", {"question": QUESTION}),
                finished("leaf"),
                finished("middle"),
                finished("root"),
            ]
        ),
        default=True,
    )
    for parent, child in (("root", "middle"), ("middle", "leaf")):
        app.register_agent(
            AgentSpec(name=parent, model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={child: SubagentSpec(agent_name=child)},
                    execution_profile_identity=_identity(f"cloud-{parent}"),
                )
            ],
        )
    app.register_agent(
        AgentSpec(name="leaf", model="test"),
        tools=[_RestartRecordingTool(), UserInputTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=["record"]),
    )
    return app


@pytest.mark.store_semantics
@pytest.mark.parametrize("wait_identity", ["exact", "wrong_parent", "wrong_child"])
def test_legacy_ancestors_refresh_only_from_exact_retained_waits(
    session_store_case, monkeypatch, wait_identity: str
) -> None:
    from cayu.sessions._run_identity import RUN_LINEAGE_METADATA_KEY

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _delegating_app(store)
            await _start(app, "root", "root")
            load = store.load

            async def legacy_load(session_id):
                session = await load(session_id)
                if session is not None:
                    metadata = dict(session.metadata)
                    metadata.pop(RUN_LINEAGE_METADATA_KEY, None)
                    session = session.model_copy(update={"metadata": metadata})
                return session

            monkeypatch.setattr(store, "load", legacy_load)
            waiting = await read_cloud_activity(app)
            leaf = next(run.session_id for run in waiting.runs if run.agent_name == "leaf")
            # The adapter changes only the feed's legacy read view. Runtime
            # execution requires the store's concrete unmodified protocol.
            monkeypatch.delattr(store, "load")
            approval = await _pending(app, leaf)
            await _drain(
                app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=leaf,
                        approval_id=approval.approval_id,
                        tool_round_id=approval.round_id,
                        tool_call_id=approval.tool_call_id,
                        decision=ToolApprovalDecision.APPROVE,
                        resolved_by=REVIEWER,
                    )
                )
            )
            monkeypatch.setattr(store, "load", legacy_load)
            checkpoint = store.load_checkpoint

            async def retained_checkpoint(session_id):
                value = await checkpoint(session_id)
                if session_id == "root" and wait_identity != "exact":
                    assert value is not None
                    wait = value["foreground_child_wait"]
                    if wait_identity == "wrong_parent":
                        wait["parent_effect"]["session_instance_id"] = "different-parent-instance"
                    else:
                        wait["child_session_instance_id"] = "different-child-instance"
                return value

            monkeypatch.setattr(store, "load_checkpoint", retained_checkpoint)
            asking = await read_cloud_activity(app, cursor=waiting.next_cursor)
            expected = (
                {"root", "middle", "leaf"} if wait_identity == "exact" else {"middle", "leaf"}
            )
            assert {run.agent_name for run in asking.runs} == expected
            assert all(run.status is CloudRunStatus.WAITING_INPUT for run in asking.runs)
            assert all(run.root_run_id is None and run.parent_run_id is None for run in asking.runs)
            # A dependency refresh does not manufacture legacy ancestry, and a
            # one-item page still carries every exactly proved dependent run.
            cursor, more, latest = waiting.next_cursor, True, {}
            while more:
                page = await read_cloud_activity(app, cursor=cursor, limit=1)
                latest.update((run.agent_name, run.status) for run in page.runs)
                cursor, more = page.next_cursor, page.has_more
            assert set(latest) == expected
            assert set(latest.values()) == {CloudRunStatus.WAITING_INPUT}
            assert cursor == asking.next_cursor
        finally:
            await _close_store(store)

    asyncio.run(scenario())


def test_a_change_two_levels_down_refreshes_every_run_waiting_on_it() -> None:
    async def scenario():
        app = _delegating_app()
        await _start(app, "root", "root")
        waiting = await read_cloud_activity(app)
        leaf = next(run.session_id for run in waiting.runs if run.agent_name == "leaf")
        approval = await _pending(app, leaf)
        # Approving the leaf lets it run on until it asks a question. Only the
        # leaf and the session waiting directly on it record that.
        await _drain(
            app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=leaf,
                    approval_id=approval.approval_id,
                    tool_round_id=approval.round_id,
                    tool_call_id=approval.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                    resolved_by=REVIEWER,
                )
            )
        )
        asking = await read_cloud_activity(app, cursor=waiting.next_cursor)
        # One run per page still reports every run above each change.
        latest: dict[str, CloudRunStatus] = {}
        cursor, more = waiting.next_cursor, True
        while more:
            page = await read_cloud_activity(app, cursor=cursor, limit=1)
            latest.update((run.agent_name, run.status) for run in page.runs)
            cursor, more = page.next_cursor, page.has_more
        assert cursor == asking.next_cursor
        question = await _pending(app, leaf)
        await _drain(
            app.resolve_user_input(
                UserInputResponse(session_id=leaf, input_id=question.input_id, answer=ANSWER)
            )
        )
        settled = await read_cloud_activity(app, cursor=asking.next_cursor)
        idle = await read_cloud_activity(app, cursor=settled.next_cursor)
        return waiting, asking, latest, settled, idle, approval, question

    waiting, asking, latest, settled, idle, approval, question = asyncio.run(scenario())

    def statuses(page: CloudActivityPage) -> dict[str, CloudRunStatus]:
        return {run.agent_name: run.status for run in page.runs}

    assert statuses(waiting) == dict.fromkeys(
        ("root", "middle", "leaf"), CloudRunStatus.WAITING_APPROVAL
    )
    runs = {run.agent_name: run for run in waiting.runs}
    middle = runs["middle"].session_id
    (root_item,) = runs["root"].waiting_for
    assert (root_item.kind, root_item.child_session_id, root_item.attention_id) == (
        "delegated_action",
        middle,
        None,
    )
    (leaf_item,) = runs["leaf"].waiting_for
    assert (leaf_item.kind, leaf_item.attention_id) == ("tool_approval", approval.attention_id)
    assert (runs["middle"].parent_session_id, runs["leaf"].parent_session_id) == ("root", middle)
    assert {run.root_session_id for run in waiting.runs} == {"root"}
    assert {run.root_run_id for run in waiting.runs} == {runs["root"].run_id}
    assert runs["middle"].parent_run_id == runs["root"].run_id
    assert runs["leaf"].parent_run_id == runs["middle"].run_id
    assert root_item.child_run_id == runs["middle"].run_id

    # The root records nothing itself, yet its summary changes and is reported again.
    assert statuses(asking) == dict.fromkeys(
        ("root", "middle", "leaf"), CloudRunStatus.WAITING_INPUT
    )
    assert [(touch.kind, touch.attention_id) for touch in asking.touches] == [
        ("approve", approval.attention_id)
    ]
    assert latest == statuses(asking)
    assert statuses(settled) == dict.fromkeys(("root", "middle", "leaf"), CloudRunStatus.COMPLETED)
    assert [(touch.kind, touch.attention_id) for touch in settled.touches] == [
        ("answer", question.attention_id)
    ]
    _no_content(settled)
    assert (idle.runs, idle.touches, idle.has_more) == ((), (), False)


def test_steering_counts_as_stepping_in_unless_a_session_or_peer_sent_it() -> None:
    async def scenario():
        app = _app()
        store = app.session_store
        identity = profiled_session_identity(
            provider_name=PRICE.provider_name, model=PRICE.model, agent_name="writer"
        )
        for session_id in ("target", "source"):
            await store.create(
                RunRequest(
                    agent_name="writer",
                    session_id=session_id,
                    messages=[Message.text("user", "go")],
                ),
                identity=identity,
            )
        source = await store.snapshot_session_message_source("source")
        for key, conditions, requested_by in (
            ("from-person", SessionMessageConditions(), REVIEWER),
            ("unattributed", SessionMessageConditions(), None),
            ("from-session", SessionMessageConditions(source=source), None),
        ):
            await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id="target",
                    idempotency_key=key,
                    content="private steering",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                    conditions=conditions,
                    requested_by=requested_by,
                )
            )
        inspection = await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        for record in inspection.records:
            await store.apply_session_message_action(
                SessionMessageActionRequest(
                    session_id=inspection.session_id,
                    session_instance_id=inspection.session_instance_id,
                    queue_id=record.queue_id,
                    expected_revision=record.revision,
                    idempotency_key="withdraw-" + record.queue_id,
                    action="withdraw",
                )
            )
        return await read_cloud_activity(app)

    page = asyncio.run(scenario())

    assert [
        (touch.kind, touch.stepped_in, None if touch.by is None else touch.by.subject)
        for touch in page.touches
    ] == [("steer", True, "priya"), ("steer", True, None), ("steer", False, None)]
    assert "private steering" not in page.model_dump_json()


def test_a_run_that_changes_while_its_page_is_read_is_reported_as_it_ends(monkeypatch) -> None:
    # One event per scan batch: the page meets the paused session's first event
    # long before the approval and completion that land while it is still reading.
    monkeypatch.setattr(cloud_reporting, "_SCAN_BATCH", 1)

    async def scenario():
        app = _app()
        await _start(app, "approve", "approver")
        approval = await _pending(app, "approve")
        store = app.session_store
        scan = store.query_events
        batches = 0

        async def query_events(query=None):
            nonlocal batches
            if query is not None and query.event_types == cloud_reporting._ACTIVITY_EVENT_TYPES:
                batches += 1
                if batches == 2:
                    await _drain(
                        app.resolve_tool_approval(
                            ToolApprovalRequest(
                                session_id="approve",
                                approval_id=approval.approval_id,
                                tool_round_id=approval.round_id,
                                tool_call_id=approval.tool_call_id,
                                decision=ToolApprovalDecision.APPROVE,
                            )
                        )
                    )
            return await scan(query)

        monkeypatch.setattr(store, "query_events", query_events)
        pages: list[CloudActivityPage] = []
        cursor = None
        while not pages or pages[-1].has_more:
            pages.append(await read_cloud_activity(app, cursor=cursor))
            cursor = pages[-1].next_cursor
        idle = await read_cloud_activity(app, cursor=cursor)
        return pages, idle, batches

    pages, idle, batches = asyncio.run(scenario())

    assert batches > 2
    latest = [run for page in pages for run in page.runs][-1]
    assert (latest.session_id, latest.status, latest.reason) == (
        "approve",
        CloudRunStatus.COMPLETED,
        "completed",
    )
    assert latest.ended_at is not None and latest.waiting_for == ()
    assert latest.tool_calls == 1
    assert [touch.kind for page in pages for touch in page.touches] == ["approve"]
    assert (idle.runs, idle.touches, idle.has_more) == ((), (), False)


def test_a_run_waiting_on_a_subagent_reports_what_the_subagent_waits_on_now() -> None:
    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_provider(
            _SequencedProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="delegate",
                            name="subagent",
                            arguments={"agent": "child", "task": "go"},
                        ),
                        ModelStreamEvent.completed(),
                    ],
                    [
                        ModelStreamEvent.tool_call(id="call_1", name="count", arguments={}),
                        ModelStreamEvent.tool_call(
                            id="call_2", name="ask_user", arguments={"question": QUESTION}
                        ),
                        ModelStreamEvent.completed(),
                    ],
                    [ModelStreamEvent.text_delta("child done"), ModelStreamEvent.completed()],
                    [ModelStreamEvent.text_delta("parent done"), ModelStreamEvent.completed()],
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("cloud-recovery-parent"),
                )
            ],
        )
        app.register_agent(
            AgentSpec(name="child", model="test"), tools=[_CountingTool(), UserInputTool()]
        )
        store = app.session_store
        await _start(app, "parent", "parent")
        asking = await read_cloud_activity(app)
        child = next(run.session_id for run in asking.runs if run.agent_name == "child")
        question = await _pending(app, child)
        # The child's answer finds a sibling call that started and never
        # finished, so the child now needs manual recovery. The parent's own
        # wait still names the question it delegated.
        await store.append_events(
            child,
            _crashed_user_input_resume_events(
                await store.load_events(child), session_id=child, tool_call_id="call_1"
            ),
        )
        await _drain(
            app.resolve_user_input(
                UserInputResponse(session_id=child, input_id=question.input_id, answer=ANSWER)
            )
        )
        (parent_wait,) = (
            await store.query_pending_actions(PendingActionQuery(session_id="parent"))
        ).actions
        assert parent_wait.delegated_action is not None
        assert parent_wait.delegated_action.action_kind == "user_input"
        stuck = await read_cloud_activity(app, cursor=asking.next_cursor)
        await _drain(
            app.recover_user_input(
                UserInputRecoveryRequest(
                    session_id=child,
                    input_id=question.input_id,
                    answer=ANSWER,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="Verified in the ledger.",
                )
            )
        )
        # The server's side-effect recovery loop hands the child's end to its
        # parent; run it once here.
        await app.recover_persisted_event_side_effects()
        recovered = await read_cloud_activity(app, cursor=stuck.next_cursor)
        return asking, stuck, recovered

    asking, stuck, recovered = asyncio.run(scenario())

    def statuses(page: CloudActivityPage) -> dict[str, CloudRunStatus]:
        return {run.agent_name: run.status for run in page.runs}

    assert statuses(asking) == dict.fromkeys(("parent", "child"), CloudRunStatus.WAITING_INPUT)
    assert statuses(stuck) == dict.fromkeys(("parent", "child"), CloudRunStatus.NEEDS_RECOVERY)
    assert statuses(recovered) == dict.fromkeys(("parent", "child"), CloudRunStatus.COMPLETED)
    assert [(touch.kind, touch.stepped_in) for touch in recovered.touches] == [("recover", True)]


def test_a_cursor_from_a_replaced_store_is_stale(tmp_path) -> None:
    async def scenario():
        # An in-memory store does not outlive its process, so a restart replaces it.
        before = _app()
        await _start(before, "before", "writer")
        stored = await read_cloud_activity(before)
        after = _app()
        await _start(after, "after", "writer")
        # A new database file under the same agent.
        first = SQLiteSessionStore(tmp_path / "first.sqlite")
        second = SQLiteSessionStore(tmp_path / "second.sqlite")
        try:
            await _start(_app(first), "first", "writer")
            on_first = await read_cloud_activity(_app(first))
            contracts = [
                await read_cloud_contract(app) for app in (before, after, _app(first), _app(second))
            ]
            for app, cursor in (
                (after, stored.next_cursor),
                (_app(second), on_first.next_cursor),
                # Issued before cursors named their store.
                (after, "7"),
            ):
                with pytest.raises(CloudActivityCursorStale):
                    await read_cloud_activity(app, cursor=cursor)
            restarted = await read_cloud_activity(after)
        finally:
            await _close_store(first)
            await _close_store(second)
        return stored, contracts, restarted

    stored, contracts, restarted = asyncio.run(scenario())

    store_ids = [contract.store_id for contract in contracts]
    assert all(store_id is not None and len(store_id) == 32 for store_id in store_ids)
    assert len(set(store_ids)) == 4
    assert stored.next_cursor.startswith(f"{store_ids[0]}.")
    assert [run.session_id for run in restarted.runs] == ["after"]


def test_a_store_restored_from_a_backup_refuses_a_cursor_past_the_backup(tmp_path) -> None:
    path, backup = tmp_path / "sessions.sqlite", tmp_path / "backup.sqlite"

    async def scenario():
        store = SQLiteSessionStore(path)
        app = _app(store)
        await _start(app, "kept", "writer")
        kept = await read_cloud_activity(app)
        identity = await store.read_store_identity()
        with sqlite3.connect(path) as source, sqlite3.connect(backup) as target:
            source.backup(target)
        await _start(app, "lost", "approver")
        await _deny(app, "lost")
        stored = await read_cloud_activity(app, cursor=kept.next_cursor)
        await _close_store(store)
        for leftover in (
            path,
            path.with_name(f"{path.name}-wal"),
            path.with_name(f"{path.name}-shm"),
        ):
            leftover.unlink(missing_ok=True)
        backup.rename(path)
        store = SQLiteSessionStore(path)
        try:
            app = _app(store)
            restored_identity = await store.read_store_identity()
            # The restored store reissues the sequences the reader already read.
            for session_id in ("new", "newer"):
                await _start(app, session_id, "approver")
                await _deny(app, session_id)
            with pytest.raises(CloudActivityCursorStale):
                await read_cloud_activity(app, cursor=stored.next_cursor)
            # A cursor from before the backup still continues.
            continued = await read_cloud_activity(app, cursor=kept.next_cursor)
            restarted = await read_cloud_activity(app)
        finally:
            await _close_store(store)
        return identity, restored_identity, stored, continued, restarted

    identity, restored_identity, stored, continued, restarted = asyncio.run(scenario())

    # A restore keeps the store's identity; the cursor's event tells the histories apart.
    assert restored_identity == identity
    assert [touch.session_id for touch in stored.touches] == ["lost"]
    assert [run.session_id for run in continued.runs] == ["new", "newer"]
    assert sorted(run.session_id for run in restarted.runs) == ["kept", "new", "newer"]
    assert [touch.session_id for touch in restarted.touches] == ["new", "newer"]


def test_a_database_without_a_store_identity_serves_no_activity(tmp_path) -> None:
    path = tmp_path / "sessions.sqlite"

    async def scenario():
        store = SQLiteSessionStore(path)
        try:
            await _start(_app(store), "run", "writer")
            with sqlite3.connect(path) as connection:
                # As a database migrated only to revision 124 has it.
                connection.execute("DROP TABLE cayu_store_identity")
            app = _app(store)
            with pytest.raises(NotImplementedError, match="cayu storage migrate"):
                await read_cloud_activity(app)
            return await read_cloud_contract(app)
        finally:
            await _close_store(store)

    assert asyncio.run(scenario()).store_id is None


def _record_when_asked_again(request: ModelRequest) -> list[ModelStreamEvent]:
    """Finish a first turn; on a later one, record an entry once."""

    users = sum(message.role is MessageRole.USER for message in request.messages)
    if users > 1 and not any(message.role is MessageRole.TOOL for message in request.messages):
        return [
            ModelStreamEvent.tool_call(id="call-1", name="record", arguments={"note": ARGUMENT}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls", "usage": USAGE}),
        ]
    return [
        ModelStreamEvent.text_delta("Done."),
        ModelStreamEvent.completed({"finish_reason": "stop", "usage": USAGE}),
    ]


_FINISHED = {CloudRunStatus.COMPLETED, CloudRunStatus.FAILED, CloudRunStatus.INTERRUPTED}


def _measures(pages: list[CloudActivityPage]) -> tuple[int, int, Decimal]:
    """Finished runs, correct runs and their cost, as the contract's Measures section says."""

    runs = {run.run_id: run for page in pages for run in page.runs}
    touches = {touch.touch_id: touch for page in pages for touch in page.touches}
    roots = {
        run_id: run
        for run_id, run in runs.items()
        if run.root_run_id == run_id and not (run.forked or run.eval) and run.status in _FINISHED
    }
    # A forked or eval session counts toward no run, even though it names a root.
    counted = {
        run_id: run
        for run_id, run in runs.items()
        if not (run.forked or run.eval) and run.root_run_id in roots
    }
    corrected = {
        touch.root_run_id
        for touch in touches.values()
        if touch.run_id in counted and touch.kind in {"edit", "deny"}
    }
    correct = sum(
        run.status is CloudRunStatus.COMPLETED and run_id not in corrected
        for run_id, run in roots.items()
    )
    return len(roots), correct, sum((run.cost.amount for run in counted.values()), Decimal(0))


def test_a_fork_counts_toward_none_of_its_source_runs_measures() -> None:
    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_provider(
            _Provider(name=PRICE.provider_name, response_factory=_record_when_asked_again),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="reviewed", model=PRICE.model),
            tools=[_Record()],
            tool_policy=_ApproveRecords(),
        )
        await _start(app, "trunk", "reviewed")
        trunk = await read_cloud_activity(app)
        # A fork, such as one Compass makes to try a change, that a person corrects.
        await _drain(
            app.fork_session(ForkSessionRequest(source_session_id="trunk", session_id="trial"))
        )
        await _drain(
            app.resume(ResumeRequest(session_id="trial", messages=[Message.text("user", "again")]))
        )
        await _deny(app, "trial")
        forked = await read_cloud_activity(app, cursor=trunk.next_cursor)
        return trunk, forked

    trunk_page, forked = asyncio.run(scenario())

    (trunk,) = trunk_page.runs
    (trial,) = forked.runs
    (denial,) = forked.touches
    # Lineage still names the source's root, and the fork says it is one.
    assert trial.root_run_id == denial.root_run_id == trunk.run_id
    assert trial.forked and trial.cost.amount > 0
    assert denial.run_id == trial.run_id and denial.kind == "deny"
    assert _measures([trunk_page, forked]) == (1, 1, trunk.cost.amount)


def test_eval_trials_and_every_session_under_them_are_reported_as_eval() -> None:
    def call(name: str, arguments: dict) -> list[ModelStreamEvent]:
        return [
            ModelStreamEvent.tool_call(id="call-1", name=name, arguments=arguments),
            ModelStreamEvent.completed(),
        ]

    def finished(name: str) -> list[ModelStreamEvent]:
        return [ModelStreamEvent.text_delta(f"{name} finished"), ModelStreamEvent.completed()]

    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_provider(
            _SequencedProvider(
                [
                    call("subagent", {"agent": "helper", "task": "work"}),
                    finished("helper"),
                    finished("lead"),
                    finished("ordinary"),
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="lead", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"helper": SubagentSpec(agent_name="helper")},
                    execution_profile_identity=_identity("cloud-eval-lead"),
                )
            ],
        )
        app.register_agent(AgentSpec(name="helper", model="test"))
        # A server runs the deployed application's eval cases in that application.
        case = EvalCase(
            id="delegates",
            request=RunRequest(agent_name="lead", messages=[Message.text("user", "go")]),
        )
        result = await run_eval_case(app, case, suite_id="suite")
        (trial,) = result.trials
        assert trial.session_id is not None
        await _start(app, "ordinary", "lead")
        await _drain(
            app.fork_session(
                ForkSessionRequest(source_session_id=trial.session_id, session_id="trial-copy")
            )
        )
        return trial.session_id, await read_cloud_activity(app)

    trial_session_id, page = asyncio.run(scenario())

    runs = {run.session_id: run for run in page.runs}
    (child,) = [run for run in page.runs if run.agent_name == "helper"]
    assert child.parent_session_id == trial_session_id
    # The trial, its subagent and a fork of the trial are eval; the application's own run is not.
    assert [
        (runs[trial_session_id].eval, child.eval, runs["trial-copy"].eval, runs["ordinary"].eval)
    ] == [(True, True, True, False)]
    assert all(run["eval"] in (True, False) for run in page.model_dump(mode="json")["runs"])


def test_a_caller_cannot_mark_its_own_run_as_an_eval_trial() -> None:
    marked = {EVAL_TRIAL_METADATA_KEY: True}
    go = [Message.text("user", "go")]
    # The body of POST /runs and POST /fork, so a remote caller is refused too.
    with pytest.raises(ValueError, match="eval-trial authority"):
        RunRequest(agent_name="writer", messages=go, metadata=marked)
    with pytest.raises(ValueError, match="eval-trial authority"):
        ForkSessionRequest(source_session_id="write", metadata=marked)

    async def scenario():
        app = _app()
        unchecked = RunRequest(agent_name="writer", session_id="hidden", messages=go)
        with pytest.raises(ValueError, match="eval-trial"):
            await _drain(app.run(unchecked.model_copy(update={"metadata": marked})))
        await _start(app, "write", "writer")
        with pytest.raises(ValueError, match="runtime-owned"):
            await app.session_store.update_metadata("write", marked)
        return await app.session_store.load("hidden"), await read_cloud_activity(app)

    hidden, page = asyncio.run(scenario())

    assert hidden is None
    assert [(run.session_id, run.eval) for run in page.runs] == [("write", False)]


def test_deleting_a_session_in_between_keeps_the_marks_of_the_sessions_above_it() -> None:
    async def scenario():
        app = _app()
        await _drain(
            app.run(
                RunRequest(
                    agent_name="writer",
                    session_id="quiet",
                    messages=[Message.text("user", "go")],
                    metadata={ATTENTION_HANDOFF_METADATA_KEY: False},
                )
            )
        )
        case = EvalCase(
            id="writes",
            request=RunRequest(agent_name="writer", messages=[Message.text("user", "go")]),
        )
        (trial,) = (await run_eval_case(app, case, suite_id="suite")).trials
        assert trial.session_id is not None
        for source, middle, last in (
            ("quiet", "quiet-b", "quiet-c"),
            (trial.session_id, "trial-b", "trial-c"),
        ):
            await _drain(
                app.fork_session(ForkSessionRequest(source_session_id=source, session_id=middle))
            )
            await _drain(
                app.fork_session(
                    ForkSessionRequest(
                        source_session_id=middle,
                        session_id=last,
                        # A fork cannot turn back on what a session above it turned off.
                        metadata={ATTENTION_HANDOFF_METADATA_KEY: True},
                    )
                )
            )
            await app.session_store.delete_session(middle)
        sessions = {name: await app.session_store.load(name) for name in ("quiet-c", "trial-c")}
        return sessions, await read_cloud_activity(app)

    sessions, page = asyncio.run(scenario())

    quiet, trial = sessions["quiet-c"], sessions["trial-c"]
    assert quiet is not None and trial is not None
    assert attention_withheld(quiet) and attention_withheld(trial)
    runs = {run.session_id: run for run in page.runs}
    assert (runs["quiet-c"].eval, runs["trial-c"].eval) == (False, True)


def test_a_page_counts_tasks_fenced_after_their_worker_was_lost(tmp_path) -> None:
    from cayu.storage.sqlite import SQLiteTaskStore
    from cayu.tasks.creation import TaskCreate
    from cayu.tasks.queries import TaskQuery

    async def scenario() -> None:
        assert (await read_cloud_activity(CayuApp(enable_logging=False))).task_health is None
        clock = {"value": datetime.now(UTC) - timedelta(hours=1)}
        tasks = SQLiteTaskStore(tmp_path / "tasks.db", ownership_clock=lambda: clock["value"])
        app = CayuApp(task_store=tasks, enable_logging=False)
        try:
            empty = await read_cloud_activity(app)
            assert empty.task_health is not None
            assert empty.task_health.fenced_after_worker_loss == 0
            for task_id in ("scan-1", "scan-2"):
                await tasks.create_task(TaskCreate(task_id=task_id, type="scan"))
            claimed = await tasks.claim_task("worker-old", TaskQuery(type="scan"), lease_seconds=30)
            assert claimed is not None and claimed.lease_expires_at is not None
            await tasks.mark_claimed_task_execution_started(
                claimed.id, "worker-old", claimed.lease_expires_at
            )
            await tasks.claim_task("worker-live", TaskQuery(type="scan"), lease_seconds=3600)
            clock["value"] = claimed.lease_expires_at + timedelta(seconds=1)
            await tasks.reclaim_expired(query=TaskQuery(type="scan"))

            page = await read_cloud_activity(app)
            assert page.task_health is not None
            assert page.task_health.fenced_after_worker_loss == 1
            assert page.task_health.complete is True
            assert page.model_dump(mode="json")["task_health"] == {
                "fenced_after_worker_loss": 1,
                "complete": True,
            }
            assert "task_health" in (await read_cloud_contract(app)).features
        finally:
            await app.aclose()
            await tasks.close()

    asyncio.run(scenario())
