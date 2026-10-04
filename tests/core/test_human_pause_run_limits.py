from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_runtime import SideEffectTool, VersionedFakeProvider

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    EventType,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    PostgresSessionStore,
    ResumeRequest,
    RunLimits,
    RunRequest,
    SQLiteSessionStore,
    ToolApprovalDecision,
    ToolApprovalRequest,
    UserInputResponse,
    UserInputTool,
)
from cayu.budgets._run_limit_accounting import (
    RunLimitAccountingContext,
    pause_run_limit_accounting_context,
    resume_run_limit_accounting_context,
)
from cayu.budgets.usage import SessionUsageSummary
from cayu.runtime._run_limit_accounting import restore_run_limit_accounting_context
from cayu.storage.migrations import SchemaMode


@asynccontextmanager
async def _stores(backend, request, sqlite_resources):
    async with sqlite_resources as resources:
        if backend == "memory":
            store = InMemorySessionStore()
            yield store, lambda: store
        elif backend == "sqlite":
            path = resources.path("pause.sqlite")

            def reopen():
                return resources.own(SQLiteSessionStore(path))

            yield reopen(), reopen
        else:
            dsn = request.getfixturevalue("postgres_dsn")
            stores = []

            def reopen():
                store = PostgresSessionStore(
                    dsn, min_size=1, max_size=2, schema_mode=SchemaMode.CREATE
                )
                stores.append(store)
                return store

            try:
                yield reopen(), reopen
            finally:
                for store in stores:
                    await store.close()


def _app(store, now, tool, pause_kind, responses):
    app = CayuApp(session_store=store, clock=lambda: now[0], enable_logging=False)
    app.register_provider(VersionedFakeProvider(responses), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool, UserInputTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy() if pause_kind == "approval" else None,
    )
    return app


async def _resolve(app, session_id, pause_kind):
    checkpoint = await app.session_store.load_checkpoint(session_id)
    if pause_kind == "approval":
        pending = checkpoint["pending_tool_approval"]
        return [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=pending["approval_id"],
                    tool_round_id=pending["tool_round_id"],
                    tool_call_id=pending["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
    pending = checkpoint["pending_user_input"]
    return [
        event
        async for event in app.resolve_user_input(
            UserInputResponse(
                session_id=session_id,
                input_id=pending["input_id"],
                answer="Continue",
            )
        )
    ]


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("pause_kind", ["approval", "user_input"])
def test_three_day_human_wait_does_not_consume_run_elapsed_limit(
    backend, pause_kind, request, sqlite_resources
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            now = [datetime(2026, 9, 29, tzinfo=UTC)]
            tool = SideEffectTool()
            calls = [
                ModelStreamEvent.tool_call(
                    id="call-effect",
                    name="side_effect",
                    arguments={"value": "approved"},
                )
            ]
            if pause_kind == "user_input":
                calls.insert(
                    0,
                    ModelStreamEvent.tool_call(
                        id="call-input",
                        name="ask_user",
                        arguments={"question": "Continue?"},
                    ),
                )
            app = _app(
                store,
                now,
                tool,
                pause_kind,
                [[*calls, ModelStreamEvent.completed({"finish_reason": "tool_calls"})]],
            )
            session_id = "three-day-" + pause_kind
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "Act after my decision")],
                        limits=RunLimits(max_elapsed_seconds=10, scope="run"),
                    )
                )
            ]
            assert not tool.calls
            assert any(event.type == EventType.SESSION_INTERRUPTED for event in events)
            now[0] += timedelta(days=3)
            if backend != "memory":
                await store.close()
            restored = _app(
                reopen(),
                now,
                tool,
                pause_kind,
                [[ModelStreamEvent.completed({"finish_reason": "stop"})]],
            )
            events = await _resolve(restored, session_id, pause_kind)
            assert tool.calls == [{"value": "approved"}]
            assert any(event.type == EventType.SESSION_COMPLETED for event in events)
            assert not any(event.type == EventType.SESSION_LIMIT_REACHED for event in events)
            durable = await restored.session_store.load_events(session_id)
            effect_events = [
                event.type
                for event in durable
                if event.tool_name == "side_effect"
                and event.type in {EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_COMPLETED}
            ]
            assert effect_events == [EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_COMPLETED]

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_legacy_shaped_approval_limit_skip_resumes_after_restart(
    backend, request, sqlite_resources
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            now = [datetime(2026, 9, 29, tzinfo=UTC)]
            tool = SideEffectTool()
            app = _app(
                store,
                now,
                tool,
                "approval",
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="call-skip", name="side_effect", arguments={}
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ]
                ],
            )
            session_id = "legacy-limit-skip"
            limits = RunLimits(max_elapsed_seconds=10, scope="session")
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "Wait for approval")],
                        limits=limits,
                    )
                )
            ]
            pending = (await store.load_checkpoint(session_id))["pending_tool_approval"]
            # Session-scope elapsed time starts at the store-owned creation
            # timestamp, not the application's independently injected clock.
            now[0] = (await store.load(session_id)).created_at + timedelta(days=3)
            _ = await _resolve(app, session_id, "approval")
            before = await store.load_events(session_id)
            skip = next(event for event in before if event.type == EventType.TOOL_CALL_FAILED)
            assert skip.payload["reason"] == "limit_reached"
            assert skip.payload["approval_id"] == pending["approval_id"]
            assert skip.payload["result"]["structured"]["skipped"] is True
            assert not any(event.type == EventType.TOOL_CALL_STARTED for event in before)
            assert await store.load_runtime_publication_receipt(
                session_id, "approval-close:" + pending["approval_id"]
            )
            assert (
                await store.load_runtime_publication_receipt(
                    session_id, "tool-round:" + pending["tool_round_id"]
                )
                is None
            )
            if backend != "memory":
                await store.close()
            restored = _app(reopen(), now, tool, "approval", [])
            events = [
                event
                async for event in restored.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "Continue after the recorded skip")],
                        limits=limits,
                    )
                )
            ]
            assert not any(event.type == EventType.SESSION_FAILED for event in events), [
                event.payload for event in events if event.type == EventType.SESSION_FAILED
            ]
            assert not tool.calls
            after = await restored.session_store.load_events(session_id)
            assert after[: len(before)] == before

    asyncio.run(scenario())


def test_pause_accounting_preserves_active_time_usage_and_retry_origin(monkeypatch):
    now = datetime(2026, 9, 29, tzinfo=UTC)
    context = RunLimitAccountingContext(
        started_at=now - timedelta(seconds=7),
        baseline=SessionUsageSummary(session_id="session", tool_calls=2),
    )
    paused = pause_run_limit_accounting_context(context, now=now)
    resolved_at = now + timedelta(days=3)
    restored = resume_run_limit_accounting_context(paused, resolved_at=resolved_at)
    assert restored.baseline == context.baseline
    assert restored.started_at == resolved_at - timedelta(seconds=7)
    assert restored.pause_started_at is None
    # Reloading the original pause after a crash closes the same interval;
    # five active seconds after the first decision remain charged.
    repeated = resume_run_limit_accounting_context(
        RunLimitAccountingContext.model_validate_json(paused.model_dump_json()),
        resolved_at=resolved_at,
    )
    monkeypatch.setattr("cayu.runtime._run_limit_accounting.time.monotonic", lambda: 100.0)
    started, baseline, _ = restore_run_limit_accounting_context(
        repeated, session_id="session", budget_limits=(), now=resolved_at + timedelta(seconds=5)
    )
    assert started == 88.0
    assert baseline.tool_calls == 2


def test_pause_behind_resumed_origin_clamps_to_empty_interval(monkeypatch):
    origin = datetime(2026, 9, 29, tzinfo=UTC)
    # A resolver ahead of the next publisher shifted the origin past its clock.
    context = RunLimitAccountingContext(
        started_at=origin,
        baseline=SessionUsageSummary(session_id="session"),
    )
    paused = pause_run_limit_accounting_context(context, now=origin - timedelta(seconds=30))
    assert paused.pause_started_at == origin
    resolved_at = origin + timedelta(hours=1)
    restored = resume_run_limit_accounting_context(paused, resolved_at=resolved_at)
    assert restored.started_at == resolved_at
    monkeypatch.setattr("cayu.runtime._run_limit_accounting.time.monotonic", lambda: 100.0)
    started, _, _ = restore_run_limit_accounting_context(
        restored, session_id="session", budget_limits=(), now=resolved_at + timedelta(seconds=5)
    )
    assert started == 95.0
    # A durable record whose pause precedes its origin is still malformed.
    with pytest.raises(ValueError, match="must not precede"):
        RunLimitAccountingContext.model_validate(
            {
                **context.model_dump(mode="python"),
                "pause_started_at": origin - timedelta(seconds=30),
            }
        )


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "public-legacy",
        "model-steps",
        "started",
        "idempotency",
        "approval",
        "duplicate",
        "not-skipped",
        "wrong-call",
    ],
)
def test_receiptless_limit_skip_requires_exact_never_dispatched_provenance(case):
    from tests.core.test_model_completion_recovery import _receiptless_pause_tail, _register_runtime

    from cayu import ToolResult

    async def scenario():
        tail = await _receiptless_pause_tail(
            pause_kind="approval", session_id="strict-limit-" + case
        )
        original = tail.terminal_events[0]
        structured = {
            "skipped": True,
            "reason": "limit_reached",
            "limit": "elapsed_seconds",
            "maximum": 10,
            "actual": 20,
            "tool_call_id": original.payload["tool_call_id"],
            "tool_name": original.tool_name,
            **{
                key: original.payload[key]
                for key in ("tool_round_id", "model_step_id", "model_attempt_id")
            },
        }
        if case == "model-steps":
            structured["limit"] = "model_steps"
        if case == "not-skipped":
            structured["skipped"] = False
        if case == "wrong-call":
            structured["tool_call_id"] = "another-call"
        result = ToolResult(
            content="Tool call skipped because a run limit was reached.",
            structured=structured,
            is_error=True,
        )
        payload = {
            **original.payload,
            "reason": "limit_reached",
            "limit": structured["limit"],
            "result": result.model_dump(mode="json"),
        }
        if case == "idempotency":
            payload["idempotency_key"] = "another-key"
        if case == "approval":
            payload["approval_id"] = "another-approval"
        terminal = original.model_copy(
            update={"type": EventType.TOOL_CALL_FAILED, "payload": payload}
        )
        events = [tail.interruption_event, tail.resume_event]
        if case == "started":
            events.extend(tail.started_events)
        events.append(terminal)
        if case == "duplicate":
            events.append(terminal.model_copy(update={"id": terminal.id + "-duplicate"}))
        message = tail.tool_result_message.model_copy(
            update={
                "content": [
                    tail.tool_result_message.content[0].model_copy(
                        update={
                            "content": result.content,
                            "structured": structured,
                            "is_error": True,
                        }
                    )
                ]
            }
        )
        await tail.store.append_events(tail.staged.session.id, events)
        await tail.store.append_transcript_messages(tail.staged.session.id, [message])
        app = _register_runtime(tail.store, tail.provider)
        if case in {"valid", "model-steps"}:
            reconciliation = await app._recovery_coordinator.reconcile_model_completion_boundary(
                tail.promoted_session
            )
            assert reconciliation.state == "already_promoted"
        elif case == "public-legacy":
            from tests.core._execution_profile_fixtures import interrupt_and_release_test_invocation
            from tests.core.test_model_completion_recovery import _test_tool

            # The historical fixture has no pause-accounting timestamps or close
            # receipt, and no started event for the never-dispatched effect.
            before = await tail.store.load_events(tail.staged.session.id)
            await interrupt_and_release_test_invocation(tail.store, tail.staged.session.id)
            tail.provider._responses = [
                [
                    ModelStreamEvent.text_delta("continued"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
            app = _register_runtime(tail.store, tail.provider, tool=_test_tool("echo"))
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=tail.staged.session.id,
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]
            assert resumed[-1].type == EventType.SESSION_COMPLETED, [
                (event.type, event.payload) for event in resumed
            ]
            assert (await tail.store.load_events(tail.staged.session.id))[: len(before)] == before
            assert not any(event.type == EventType.TOOL_CALL_STARTED for event in resumed)
            assert len(tail.provider.requests) == 1
            return
        else:
            with pytest.raises(RuntimeError):
                await app._recovery_coordinator.reconcile_model_completion_boundary(
                    tail.promoted_session
                )
        assert tail.provider.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("pause_kind", ["approval", "user_input"])
def test_resolution_claim_retries_keep_first_durable_pause_end(pause_kind):
    from cayu.approvals.tools import PendingToolApproval
    from cayu.approvals.user_input import (
        PendingUserInput,
        checkpoint_with_user_input_resolution_intent,
    )
    from cayu.runtime import _approval_support as approvals
    from cayu.vaults.redaction import SecretRedactor

    async def scenario():
        store = InMemorySessionStore()
        now = [datetime(2026, 9, 29, tzinfo=UTC)]
        app = _app(
            store,
            now,
            SideEffectTool(),
            pause_kind,
            [
                [
                    ModelStreamEvent.tool_call(
                        id="claim",
                        name="side_effect" if pause_kind == "approval" else "ask_user",
                        arguments={} if pause_kind == "approval" else {"question": "Continue?"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ],
        )
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="claim-retry",
                    messages=[Message.text("user", "Wait")],
                    limits=RunLimits(max_elapsed_seconds=10),
                )
            )
        ]
        checkpoint = await store.load_checkpoint("claim-retry")
        first = now[0] + timedelta(days=3)
        later = first + timedelta(days=2)
        redactor = SecretRedactor()
        if pause_kind == "approval":
            pending = PendingToolApproval.model_validate(checkpoint["pending_tool_approval"])
            first_checkpoint = approvals.checkpoint_with_approval_resolution_intent(
                checkpoint,
                approval=pending,
                decision=ToolApprovalDecision.APPROVE,
                resolution_request_digest="a" * 64,
                redactor=redactor,
                pause_resolved_at=first,
            )
            second_checkpoint = approvals.checkpoint_with_approval_resolution_intent(
                first_checkpoint,
                approval=pending,
                decision=ToolApprovalDecision.APPROVE,
                resolution_request_digest="a" * 64,
                redactor=redactor,
                pause_resolved_at=later,
            )
            intent = approvals.approval_resolution_intent_from_checkpoint(second_checkpoint)
        else:
            pending = PendingUserInput.model_validate(checkpoint["pending_user_input"])
            kwargs = dict(
                pending=pending,
                answer_request_digest="a" * 64,
                resolution_stage="answer",
                resolution_request_digest="b" * 64,
                redactor=redactor,
            )
            first_checkpoint, _ = checkpoint_with_user_input_resolution_intent(
                checkpoint,
                **kwargs,
                claim_run_epoch=pending.source_run_epoch + 1,
                pause_resolved_at=first,
            )
            _, intent = checkpoint_with_user_input_resolution_intent(
                first_checkpoint,
                **kwargs,
                claim_run_epoch=pending.source_run_epoch + 2,
                pause_resolved_at=later,
            )
        assert intent.pause_resolved_at == first

    asyncio.run(scenario())
