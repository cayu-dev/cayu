"""Create-or-resume convenience must retain native admission and recovery rules."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_require_final_tool import _ApprovalPolicy
from tests.core.test_tool_completion import FinalTool, app_for, call
from tests.core.test_tool_round_publication_failure_matrix import store_factory as store_factory

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    RunOutcome,
    RunRequest,
    SessionStatus,
    TurnRequest,
    send_turn,
)
from cayu.approvals import ToolApprovalDecision, ToolApprovalRequest, UserInputResponse
from cayu.deadlines import ExecutionDeadline, bind_execution_deadline
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime import ToolCompletionPolicy
from cayu.sessions import SessionStatusConflict
from cayu.sessions.requests import ResumeRequest, copy_resume_request


class Store:
    """Use the shared factory's real memory, SQLite and Postgres backends."""


def text_response(text):
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def request(session_id=None, **options):
    return TurnRequest(
        agent_name="support",
        session_id=session_id,
        messages=[Message.text("user", "Help")],
        **options,
    )


@pytest.mark.parametrize("supplied_id", [None, "chosen"])
def test_create_then_resume_preserves_transcript_and_reports_only_current_turn(
    store_factory, supplied_id
):
    async def scenario():
        async with store_factory(Store) as store:
            provider = ScriptedModelProvider([text_response("First"), text_response("Second")])
            app = app_for(store, provider, FinalTool())
            first = await send_turn(app, request(supplied_id))
            assert first.ok, first.error
            assert first.session_id
            if supplied_id is not None:
                assert first.session_id == supplied_id
            second = await send_turn(app, request(first.session_id))
            assert second.ok, second.error
            assert second.final_text == "Second"
            assert second.model_steps == 1
            assert second.session_id == first.session_id
            assert second.interaction_id != first.interaction_id
            assert [m.role for m in await store.load_transcript(first.session_id)] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]
            assert len(provider.requests) == 2

    asyncio.run(scenario())


def test_existing_agent_and_creation_settings_are_checked_before_execution(store_factory):
    async def scenario():
        async with store_factory(Store) as store:
            provider = ScriptedModelProvider([text_response("done")])
            app = app_for(store, provider, FinalTool())
            assert (await send_turn(app, request("existing"))).ok
            before = await store.load_transcript("existing")
            with pytest.raises(ValueError, match="agent_name"):
                await send_turn(
                    app,
                    TurnRequest(
                        agent_name="other",
                        session_id="existing",
                        messages=[Message.text("user", "help")],
                    ),
                )
            for setting in (
                {"labels": {}},
                {"environment_name": None},
                {"parent_session_id": None},
                {"execution_deadline": ExecutionDeadline()},
            ):
                with pytest.raises(ValueError, match="creation-only"):
                    await send_turn(app, request("existing", **setting))
            assert await store.load_transcript("existing") == before
            assert len(provider.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status", [SessionStatus.PENDING, SessionStatus.RUNNING, SessionStatus.INTERRUPTING]
)
def test_active_sessions_raise_conflict_without_mutation(store_factory, status):
    async def scenario():
        async with store_factory(Store) as store:
            provider = ScriptedModelProvider([text_response("done")])
            app = app_for(store, provider, FinalTool())
            assert (await send_turn(app, request(f"active-{status.value}"))).ok
            await store.update_status(f"active-{status.value}", status)
            before = await store.load(f"active-{status.value}")
            transcript = await store.load_transcript(f"active-{status.value}")
            with pytest.raises(SessionStatusConflict):
                await send_turn(app, request(f"active-{status.value}"))
            assert await store.load(f"active-{status.value}") == before
            assert await store.load_transcript(f"active-{status.value}") == transcript
            assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_lost_creation_race_does_not_retry_as_resume(store_factory, monkeypatch):
    async def scenario():
        async with store_factory(Store) as store:
            provider = ScriptedModelProvider([text_response("winner")])
            app = app_for(store, provider, FinalTool())
            native_run = app.run

            async def race(selected):
                # Another creator wins after the helper chose creation.
                winner = RunRequest(
                    agent_name="support",
                    session_id=selected.session_id,
                    messages=[Message.text("user", "winner input")],
                )
                async for _ in native_run(winner):
                    pass
                async for value in native_run(selected):
                    yield value

            monkeypatch.setattr(app, "run", race)
            outcome = await send_turn(app, request("race"))
            assert outcome.status is SessionStatus.FAILED
            assert outcome.error_detail is not None
            assert len(provider.requests) == 1
            transcript = await store.load_transcript("race")
            assert [m.role for m in transcript] == ["user", "assistant"]
            assert transcript[0].content[0].text == "winner input"

    asyncio.run(scenario())


def test_replaced_session_cannot_receive_the_selected_resume(store_factory, monkeypatch):
    async def scenario():
        async with store_factory(Store) as store:
            provider = ScriptedModelProvider(
                [text_response("original"), text_response("replacement")]
            )
            app = app_for(store, provider, FinalTool())
            assert (await send_turn(app, request("reused"))).ok
            original = await store.load("reused")
            native_resume = app.resume

            async def race(selected):
                await store.delete_session("reused")
                replacement = await send_turn(app, request("reused"))
                assert replacement.ok, replacement.error
                assert (await store.load("reused")).instance_id != original.instance_id
                async for value in native_resume(selected):
                    yield value

            monkeypatch.setattr(app, "resume", race)
            outcome = await send_turn(app, request("reused"))
            assert outcome.status is SessionStatus.FAILED
            assert "SessionStatusConflict" in outcome.error
            assert len(provider.requests) == 2
            assert len(await store.load_transcript("reused")) == 2

    asyncio.run(scenario())


def test_step_limit_is_resumable_and_omitted_resume_limit_is_inherited(store_factory):
    async def scenario():
        async with store_factory(Store) as store:
            provider = ScriptedModelProvider(
                [text_response("initial"), call(), text_response("finished")]
            )
            tool = FinalTool()
            app = app_for(store, provider, tool)
            assert (await send_turn(app, request("limited", max_steps=1))).ok
            limited = await send_turn(app, request("limited"))
            assert limited.status is SessionStatus.INTERRUPTED, limited.error
            assert limited.resumable is True
            assert limited.interruption.kind == "limit"
            assert limited.interruption.limit == "model_steps"
            assert limited.tool_call_count == 1
            assert limited.tool_calls[0].result.content == "Which order?"
            assert tool.calls == 1
            final = await send_turn(app, request("limited"))
            assert final.ok, final.error
            assert final.final_text == "finished"
            assert final.model_steps == 1
            assert final.tool_call_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("approve", [True, False])
def test_approval_requires_dedicated_resolution_and_uses_the_same_outcome(store_factory, approve):
    async def scenario():
        async with store_factory(Store) as store:
            provider = ScriptedModelProvider([call(), text_response("denied")])
            tool = FinalTool()
            app = app_for(store, provider, tool, tool_policy=_ApprovalPolicy())
            paused = await send_turn(
                app,
                request(
                    f"approval-{approve}",
                    tool_completion=ToolCompletionPolicy(tool_names=["ask_customer"]),
                ),
            )
            assert paused.interruption.kind == "tool_approval"
            assert paused.interruption.approval_id
            assert paused.resumable is False
            before = await store.load_checkpoint(f"approval-{approve}")
            refused = await send_turn(app, request(f"approval-{approve}"))
            assert not refused.ok
            assert await store.load_checkpoint(f"approval-{approve}") == before
            assert tool.calls == 0
            approval = next(e for e in paused.events if e.type == "tool.call.approval_requested")
            resolved = RunOutcome.from_events(
                [
                    e
                    async for e in app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id=f"approval-{approve}",
                            approval_id=paused.interruption.approval_id,
                            tool_call_id=approval.payload["tool_call_id"],
                            tool_round_id=approval.payload["tool_round_id"],
                            decision=ToolApprovalDecision.APPROVE
                            if approve
                            else ToolApprovalDecision.DENY,
                        )
                    )
                ]
            )
            assert resolved.ok, resolved.error
            assert resolved.tool_call_count == 1
            assert resolved.tool_calls[0].disposition == ("completed" if approve else "blocked")
            assert resolved.model_steps == (0 if approve else 1)
            assert tool.calls == int(approve)

    asyncio.run(scenario())


def test_user_input_requires_dedicated_resolution_and_projects_its_result(store_factory):
    from tests.core.test_user_input import _build

    async def scenario():
        async with store_factory(Store) as store:
            app, _ = _build([("ask", "ask_user", {"question": "Which environment?"})], store=store)
            paused = await send_turn(
                app,
                TurnRequest(
                    agent_name="assistant",
                    session_id="input",
                    messages=[Message.text("user", "deploy")],
                ),
            )
            assert paused.interruption.kind == "user_input"
            assert paused.interruption.input_id
            assert paused.resumable is False
            before = await store.load_checkpoint("input")
            refused = await send_turn(
                app,
                TurnRequest(
                    agent_name="assistant",
                    session_id="input",
                    messages=[Message.text("user", "prod")],
                ),
            )
            assert not refused.ok
            assert await store.load_checkpoint("input") == before
            resolved = RunOutcome.from_events(
                [
                    e
                    async for e in app.resolve_user_input(
                        UserInputResponse(
                            session_id="input",
                            input_id=paused.interruption.input_id,
                            answer="prod",
                        )
                    )
                ]
            )
            assert resolved.ok, resolved.error
            assert resolved.tool_call_count == 1
            assert resolved.tool_calls[0].result.content == "prod"
            assert resolved.model_steps == 1

    asyncio.run(scenario())


def test_captured_default_deadline_survives_native_request_conversion(monkeypatch):
    from cayu.sessions import InMemorySessionStore

    deadline = ExecutionDeadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
    with bind_execution_deadline(deadline):
        turn = request()
    app = app_for(
        InMemorySessionStore(), ScriptedModelProvider([text_response("done")]), FinalTool()
    )
    native_run = app.run
    observed = []

    async def run(selected):
        observed.append(selected.execution_deadline)
        async for value in native_run(selected):
            yield value

    monkeypatch.setattr(app, "run", run)
    outcome = asyncio.run(send_turn(app, turn))
    assert outcome.ok, outcome.error
    assert observed == [deadline]


def test_resume_copy_preserves_restrictive_instance_guard_and_omitted_controls():
    original = ResumeRequest(
        session_id="s",
        expected_session_instance_id="instance",
        messages=[Message.text("user", "hello")],
    )
    copied = copy_resume_request(original)
    assert copied.expected_session_instance_id == "instance"
    assert "max_steps" not in copied.model_fields_set
    assert (
        ResumeRequest.model_validate_json(copied.model_dump_json()).expected_session_instance_id
        == "instance"
    )
    assert (
        "expected_session_instance_id"
        not in ResumeRequest(session_id="s", messages=original.messages).model_dump()
    )


def test_mutated_default_metadata_and_loop_policy_survive_conversion(store_factory):
    from cayu.runtime import RequireFinalTool

    async def scenario():
        async with store_factory(Store) as store:
            provider = ScriptedModelProvider([text_response("draft"), call()])
            app = app_for(store, provider, FinalTool())
            turn = request(
                "controls",
                max_steps=2,
                tool_completion=ToolCompletionPolicy(tool_names=["ask_customer"]),
                loop_policies=(RequireFinalTool(["ask_customer"]),),
            )
            turn.metadata["product"] = "support"
            outcome = await send_turn(app, turn)
            assert outcome.ok, outcome.error
            assert (await store.load("controls")).metadata["product"] == "support"
            assert outcome.model_steps == 2
            assert outcome.tool_completion.result.content == "Which order?"
            assert outcome.summary()["tool_completion"]["content"] == "Which order?"
            assert "arguments" not in outcome.summary()["tool_completion"]
            assert outcome.summary(include_structured=True)["tool_completion"]["structured"] == {
                "question": "Which order?",
            }

    asyncio.run(scenario())


def test_future_checkpoint_is_rejected_before_new_messages_or_provider_calls(
    store_factory, monkeypatch
):
    from cayu.sessions.checkpoints import (
        CHECKPOINT_SCHEMA_VERSION_KEY,
        CURRENT_CHECKPOINT_SCHEMA_VERSION,
    )

    async def scenario():
        async with store_factory(Store) as store:
            provider = ScriptedModelProvider([text_response("done")])
            app = app_for(store, provider, FinalTool())
            assert (await send_turn(app, request("future"))).ok
            before = await store.load_transcript("future")
            load_checkpoint = store.load_checkpoint

            async def future_checkpoint(session_id):
                value = await load_checkpoint(session_id)
                return {
                    **(value or {}),
                    CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
                }

            monkeypatch.setattr(store, "load_checkpoint", future_checkpoint)
            outcome = await send_turn(app, request("future"))
            assert not outcome.ok
            assert "CheckpointCompatibilityError" in outcome.error
            assert await store.load_transcript("future") == before
            assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_manual_recovery_projects_one_terminal_call_without_reexecution(store_factory):
    from tests.core.test_tool_round_publication_failure_matrix import _SimulatedProcessLoss

    from cayu import ToolEffect
    from cayu.tools import ToolRoundRecoveryRequest

    class LostEffect(FinalTool):
        async def run(self, ctx, args):
            self.calls += 1
            raise _SimulatedProcessLoss("process lost before tool result publication")

    async def scenario():
        async with store_factory(Store) as store:
            provider = ScriptedModelProvider([call(), text_response("recovered")])
            tool = LostEffect(ToolEffect.NONE)
            app = app_for(store, provider, tool)
            events = []
            with pytest.raises(_SimulatedProcessLoss):
                async for value in app.run(
                    RunRequest(
                        agent_name="support",
                        session_id="recover",
                        messages=[Message.text("user", "Help")],
                    )
                ):
                    events.append(value)
            failed = RunOutcome.from_events(events)
            assert not failed.ok
            assert tool.calls == 1
            started = next(e for e in failed.events if e.type == "tool.call.started")
            assert failed.tool_call_count == 1
            assert failed.tool_calls[0].disposition == "incomplete"
            recovered = RunOutcome.from_events(
                [
                    e
                    async for e in app.recover_tool_round(
                        ToolRoundRecoveryRequest(
                            session_id="recover",
                            round_id=started.payload["tool_round_id"],
                            tool_call_id=started.payload["tool_call_id"],
                            outcome="completed",
                            message="recorded",
                        )
                    )
                ]
            )
            assert recovered.ok, recovered.error
            assert recovered.tool_call_count == 1
            assert recovered.tool_calls[0].disposition == "completed"
            assert recovered.tool_calls[0].start_event_id is None
            assert recovered.tool_calls[0].correlation_id == failed.tool_calls[0].correlation_id
            assert recovered.final_text == "recovered"
            assert tool.calls == 1

    asyncio.run(scenario())


def test_summary_uses_effective_arguments_and_final_redacted_hook_result():
    from cayu import Tool, ToolResult, ToolSpec
    from cayu.observability import AfterToolCallDecision, BeforeToolCallDecision, RuntimeHook
    from cayu.sessions import InMemorySessionStore
    from cayu.vaults import SecretRedactor

    class Echo(Tool):
        spec = ToolSpec(
            name="echo",
            description="Echo",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

        async def run(self, ctx, args):
            assert args == {"text": "changed"}
            return ToolResult(content="original result")

    class Modify(RuntimeHook):
        async def before_tool_call(self, context):
            return BeforeToolCallDecision(
                action="proceed_modified", modified_arguments={"text": "changed"}
            )

        async def after_tool_call(self, context):
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(
                    content="final private-canary",
                    structured={"value": "private-canary"},
                ),
            )

    app = CayuApp(
        session_store=InMemorySessionStore(),
        enable_logging=False,
        runtime_hooks=[Modify()],
        secret_redactor=SecretRedactor("private-canary"),
    )
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="echo", name="echo", arguments={"text": "requested"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="support", model="scripted-model"), tools=[Echo()])
    outcome = asyncio.run(send_turn(app, request(max_steps=1)))
    assert outcome.status is SessionStatus.INTERRUPTED, outcome.error
    assert outcome.tool_call_count == 1
    summary = outcome.tool_calls[0]
    assert summary.requested_arguments == {"text": "requested"}
    assert summary.effective_arguments == {"text": "changed"}
    assert summary.result.content.startswith("final ")
    assert "private-canary" not in str(summary)
    assert "private-canary" not in str(
        outcome.summary(include_events=True, include_structured=True)
    )


@pytest.mark.parametrize("guard", ["omitted", None, "instance"])
@pytest.mark.parametrize("epoch", [None, 7])
def test_portable_resume_source_preserves_guard_without_changing_omission_semantics(guard, epoch):
    from cayu.runtime._session_request_boundary import prepare_resume_request
    from cayu.vaults import SecretRedactor

    fields = {} if guard == "omitted" else {"expected_session_instance_id": guard}
    if epoch is not None:
        fields["expected_run_epoch"] = epoch
    original = ResumeRequest(session_id="s", messages=[Message.text("user", "help")], **fields)
    copied = copy_resume_request(original)
    assert ("expected_session_instance_id" in copied.model_fields_set) == (guard != "omitted")
    engine = CayuApp(enable_logging=False)._work_attempt_execution
    digest = engine.work_attempt_source_request_sha256(copied, kind="continuation")
    snapshot = engine.work_attempt_source_snapshot(
        copied, kind="continuation", source_request_sha256=digest
    )
    restored = ResumeRequest.model_validate(snapshot.request)
    assert restored.expected_run_epoch == epoch
    assert restored.expected_session_instance_id == original.expected_session_instance_id
    assert ("expected_session_instance_id" in snapshot.fields_set) == (guard != "omitted")
    if guard == "instance":
        with pytest.raises(ValueError, match="expected_session_instance_id"):
            prepare_resume_request(original, redactor=SecretRedactor("instance"))


@pytest.mark.parametrize("boundary", ["selection", "preparation"])
def test_concurrent_turn_cannot_be_taken_over_after_selection(store_factory, monkeypatch, boundary):
    async def scenario():
        async with store_factory(Store) as store:
            provider = ScriptedModelProvider([text_response("initial"), text_response("winner")])
            app = app_for(store, provider, FinalTool())
            assert (await send_turn(app, request(f"epoch-{boundary}"))).ok
            session_id = f"epoch-{boundary}"
            before = await store.load(session_id)
            native_resume = app.resume
            other_stream = None
            started = False

            async def start_other():
                nonlocal other_stream, started
                if started:
                    return
                started = True
                other_stream = native_resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "winner input")],
                    )
                )
                async for value in other_stream:
                    if value.type == "session.resumed":
                        break
                else:
                    raise AssertionError("Concurrent turn did not start.")
                current = await store.load(session_id)
                assert current.status is SessionStatus.RUNNING
                assert current.run_epoch > before.run_epoch

            if boundary == "selection":

                async def resume(selected):
                    await start_other()
                    async for value in native_resume(selected):
                        yield value

                monkeypatch.setattr(app, "resume", resume)
            else:
                prepare_resume = app._session_engine.prepare_resume

                async def prepare(**options):
                    await start_other()
                    return await prepare_resume(**options)

                monkeypatch.setattr(app._session_engine, "prepare_resume", prepare)
            try:
                outcome = await send_turn(
                    app,
                    TurnRequest(
                        agent_name="support",
                        session_id=session_id,
                        messages=[Message.text("user", "loser input")],
                    ),
                )
                assert not outcome.ok
                assert "SessionStatusConflict" in outcome.error
                assert (await store.load(session_id)).status is SessionStatus.RUNNING
                transcript = await store.load_transcript(session_id)
                assert all("loser input" not in str(message) for message in transcript)
            finally:
                if other_stream is not None:
                    await other_stream.aclose()

    asyncio.run(scenario())
