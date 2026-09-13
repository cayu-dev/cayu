"""A child's human-action pause must remain a pending parent delegation."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest
from pydantic import SecretStr
from tests.core.test_foreground_subagent_recovery import _identity, _Provider
from tests.core.test_tool_round_execution_identities import _RecordingTool

from cayu import (
    AgentSpec,
    CayuApp,
    InMemorySessionStore,
    Message,
    ResumeRequest,
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
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.tool_policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.user_input import UserInputTool
from cayu.vaults import SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("action", ["approve", "deny", "input"])
@pytest.mark.parametrize("replace_child_after_selection", [False, True])
@pytest.mark.parametrize("redacted_child_id", [False, True], ids=["ordinary-id", "redacted-id"])
def test_foreground_child_action_suspends_and_automatically_continues_parent(
    tmp_path, monkeypatch, backend, action, replace_child_after_selection, redacted_child_id
):
    async def scenario():
        codec = (
            PublicAuthorityAliasCodec(
                PublicAuthorityAliasKeyring(
                    active_key_id="test", keys={"test": SecretStr("A" * 43)}
                )
            )
            if redacted_child_id
            else None
        )
        store = (
            InMemorySessionStore(public_authority_alias_codec=codec)
            if backend == "memory"
            else SQLiteSessionStore(
                tmp_path / "child-pauses.sqlite", public_authority_alias_codec=codec
            )
        )
        protected = _RecordingTool()
        child_call = ModelStreamEvent.tool_call(
            id="child-action",
            name="ask_user" if action == "input" else "record",
            arguments={"question": "Which value?"} if action == "input" else {"value": 7},
        )
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "work"}
                    ),
                    ModelStreamEvent.completed(),
                ],
                [child_call, ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("child finished"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("parent finished"), ModelStreamEvent.completed()],
            ]
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            secret_redactor=SecretRedactor("cayu-child") if redacted_child_id else None,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("paused-subagent"),
                )
            ],
        )
        app.register_agent(
            AgentSpec(name="child", model="test"),
            tools=[UserInputTool()] if action == "input" else [protected],
            tool_policy=None
            if action == "input"
            else AlwaysRequireApprovalToolPolicy(tools=["record"]),
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="parent",
                        agent_name="parent",
                        messages=[Message.text("user", "delegate")],
                    )
                )
            ]
            assert events[-1].type == "session.interrupted", events[-1].payload
            assert len(provider.requests) == 2
            assert protected.values == []
            assert not any(
                event.type in {"tool.call.completed", "tool.call.failed", "interaction.completed"}
                for event in events
            )
            checkpoint = await store.load_checkpoint("parent")
            assert checkpoint is not None and "pending_tool_round" in checkpoint
            children = await store.list_sessions(SessionQuery(parent_session_id="parent"))
            assert len(children.sessions) == 1
            child = children.sessions[0]
            from cayu.runtime._foreground_child_wait import ForegroundChildWait
            from cayu.runtime.execution_profiles import (
                active_invocation_execution_profile_from_checkpoint,
                active_invocation_execution_profile_is_released,
            )

            for paused_id in ("parent", child.id):
                paused = await store.load(paused_id)
                assert paused is not None
                profile = active_invocation_execution_profile_from_checkpoint(
                    await store.load_checkpoint(paused_id)
                )
                assert profile is not None
                assert active_invocation_execution_profile_is_released(
                    profile, session_id=paused.id, run_epoch=paused.run_epoch
                )
                assert not app._session_control.has_active_tasks(paused_id)

            wait = ForegroundChildWait.model_validate(checkpoint["foreground_child_wait"])
            assert wait.parent_effect.session_id == "parent"
            assert wait.child_session_id == child.id
            assert wait.child_session_instance_id == child.instance_id
            assert wait.child_action_kind == (
                "user_input" if action == "input" else "tool_approval"
            )
            assert checkpoint.get("pending_tool_approval") is None
            assert checkpoint.get("pending_user_input") is None
            parent_pauses = [e for e in events if e.type == "interaction.paused"]
            assert len(parent_pauses) == 1
            assert parent_pauses[0].interaction_id == wait.parent_effect.interaction_id
            child_events = await store.load_events(child.id)
            if action == "input":
                pending = next(e for e in child_events if e.type == "session.awaiting_user_input")
                resolution = app.resolve_user_input(
                    UserInputResponse(
                        session_id=app.project_session_id_for_exposure(child.id),
                        input_id=pending.payload["input_id"],
                        answer="7",
                    )
                )
            else:
                pending = next(e for e in child_events if e.type == "tool.call.approval_requested")
                resolution = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=app.project_session_id_for_exposure(child.id),
                        approval_id=pending.payload["approval"]["approval_id"],
                        tool_round_id=pending.payload["tool_round_id"],
                        tool_call_id=pending.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE
                        if action == "approve"
                        else ToolApprovalDecision.DENY,
                    )
                )
            attached_delivery_results = []
            checked_staged_identity = []
            if not replace_child_after_selection:
                from cayu.runtime._foreground_child_continuation import (
                    deliver_foreground_child_terminal,
                )

                publish = app._runtime_session_store.publish_runtime_publication

                async def unexpected_resume(_terminal):
                    pytest.fail("An attached child result must not be attached again.")

                async def unexpected_refresh(_wait, _event):
                    pytest.fail("An attached child result must not refresh a pending action.")

                async def inspect_after_parent_attachment(session_id, **kwargs):
                    if (
                        redacted_child_id
                        and action != "input"
                        and session_id == child.id
                        and not checked_staged_identity
                    ):
                        from cayu.runtime._tool_round_recovery import (
                            pending_tool_round_from_checkpoint,
                        )

                        staged_checkpoint = await store.load_checkpoint(child.id)
                        staged_round = (staged_checkpoint or {}).get("pending_tool_round")
                        if staged_round and staged_round.get("staged_terminals"):
                            assert staged_checkpoint is not None
                            current_child = await store.load(child.id)
                            assert current_child is not None
                            redactor = SecretRedactor("cayu-child")
                            with pytest.raises(ValueError, match="workload secret"):
                                pending_tool_round_from_checkpoint(
                                    staged_checkpoint, redactor=redactor
                                )
                            trusted_round = pending_tool_round_from_checkpoint(
                                staged_checkpoint, redactor=redactor, runtime_session=current_child
                            )
                            assert trusted_round is not None
                            wrong_owner = current_child.model_copy(
                                update={"id": "different-session"}
                            )
                            with pytest.raises(ValueError, match="workload secret"):
                                pending_tool_round_from_checkpoint(
                                    staged_checkpoint,
                                    redactor=redactor,
                                    runtime_session=wrong_owner,
                                )
                            # The exact envelope allowance must not extend into
                            # arbitrary result content, even with the right owner.
                            unsafe_checkpoint = deepcopy(staged_checkpoint)
                            unsafe_checkpoint["pending_tool_round"]["staged_terminals"][0]["event"][
                                "payload"
                            ]["untrusted_note"] = "cayu-child"
                            with pytest.raises(ValueError, match="workload secret"):
                                pending_tool_round_from_checkpoint(
                                    unsafe_checkpoint,
                                    redactor=redactor,
                                    runtime_session=current_child,
                                )
                            checked_staged_identity.append(True)
                    result = await publish(session_id, **kwargs)
                    if session_id == "parent" and kwargs["request"].kind == "tool-round":
                        outcome = await store.summarize_outcome(child.id)
                        assert outcome.terminal_event is not None
                        attached_delivery_results.append(
                            await deliver_foreground_child_terminal(
                                outcome.terminal_event.event,
                                store=store,
                                has_active_tasks=app._session_control.has_active_tasks,
                                resume=unexpected_resume,
                                refresh=unexpected_refresh,
                                settle=unexpected_refresh,
                            )
                        )
                    return result

                monkeypatch.setattr(
                    app._runtime_session_store,
                    "publish_runtime_publication",
                    inspect_after_parent_attachment,
                )
            if replace_child_after_selection:
                selection_ready = asyncio.Event()
                proceed = asyncio.Event()
                apply_command = app._runtime_session_store.apply_invocation_lifecycle_command

                async def pause_after_selection(command):
                    result = await apply_command(command)
                    if command.session_id == "parent" and not selection_ready.is_set():
                        state = await store.load_checkpoint("parent")
                        if state is not None and "foreground_child_terminal" in state:
                            selection_ready.set()
                            await proceed.wait()
                    return result

                monkeypatch.setattr(
                    app._runtime_session_store,
                    "apply_invocation_lifecycle_command",
                    pause_after_selection,
                )

                async def resolve():
                    return [event async for event in resolution]

                resolution_task = asyncio.create_task(resolve())
                try:
                    await asyncio.wait_for(selection_ready.wait(), timeout=20)
                    replacement = [
                        event
                        async for event in app.resume(
                            ResumeRequest(
                                session_id=app.project_session_id_for_exposure(child.id),
                                messages=[Message.text("user", "a different interaction")],
                            )
                        )
                    ]
                    assert replacement[-1].type == "session.completed", replacement[-1].payload
                finally:
                    proceed.set()
                    await asyncio.wait_for(resolution_task, timeout=20)
                assert len(provider.requests) == 4
                parent = await store.load("parent")
                assert parent is not None and parent.status is SessionStatus.FAILED
                assert not any(
                    event.type == "tool.call.completed"
                    for event in await store.load_events("parent")
                )
                return
            _ = [event async for event in resolution]
            resolved_child = await store.load(child.id)
            assert (
                resolved_child is not None and resolved_child.status is SessionStatus.COMPLETED
            ), "\n".join(
                str(event.payload.get("error") or event.payload.get("reason") or event.type)
                for event in (await store.load_events(child.id))[-3:]
            )
            # No unrelated parent resume call is required to notice its child's result.
            parent = await store.load("parent")
            assert parent is not None and parent.status is SessionStatus.COMPLETED, {
                "child_terminal": [
                    (event.type, event.payload.get("error"), event.payload.get("reason"))
                    for event in await store.load_events(child.id)
                    if event.type in {"session.failed", "session.interrupted"}
                ],
                "delivery_errors": [
                    (delivery.event_id, delivery.status, delivery.last_error)
                    for delivery in await store.list_persisted_event_side_effect_deliveries(
                        limit=100
                    )
                    if delivery.last_error is not None
                ],
                "parent_failures": [
                    event.payload
                    for event in await store.load_events("parent")
                    if event.type == "session.failed"
                ],
            }
            assert len(provider.requests) == 4
            assert protected.values == ([7] if action == "approve" else [])
            if redacted_child_id and action != "input":
                assert checked_staged_identity == [True]
            parent_events = await store.load_events("parent")
            terminal_tools = [
                event
                for event in parent_events
                if event.type in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(terminal_tools) == 1
            assert terminal_tools[0].type == "tool.call.completed"
            assert "child finished" in terminal_tools[0].payload["result"]["content"]
            assert len([m for m in await store.load_transcript("parent") if m.role == "tool"]) == 1
            receipt = await store.load_runtime_publication_receipt(
                "parent", f"tool-round:{wait.parent_effect.tool_round_id}"
            )
            assert receipt is not None
            from cayu.runtime._foreground_child_wait import (
                FOREGROUND_PARENT_CONTINUATION_KEY,
                ForegroundChildTerminal,
                ForegroundParentContinuation,
            )

            selected = ForegroundChildTerminal.model_validate(
                receipt.intent["foreground_child_terminal"]
            )
            assert selected.wait == wait
            child_outcome = await store.summarize_outcome(child.id)
            assert child_outcome.terminal_event is not None
            assert selected.event_id == child_outcome.terminal_event.event.id
            settled_checkpoint = await store.load_checkpoint("parent")
            assert settled_checkpoint is not None
            assert "foreground_child_wait" not in settled_checkpoint
            assert "foreground_child_terminal" not in settled_checkpoint
            continuation = ForegroundParentContinuation.model_validate(
                receipt.intent[FOREGROUND_PARENT_CONTINUATION_KEY]
            )
            assert continuation.terminal == selected
            assert continuation.publication_id == receipt.publication_id
            assert continuation.request.session_id == "parent"
            assert continuation.request.messages == []
            assert attached_delivery_results == [False]
            assert await deliver_foreground_child_terminal(
                child_outcome.terminal_event.event,
                store=store,
                has_active_tasks=lambda _: False,
                resume=unexpected_resume,
                refresh=unexpected_refresh,
                settle=unexpected_refresh,
            )
            assert (
                continuation.model_dump(mode="json")
                == settled_checkpoint[FOREGROUND_PARENT_CONTINUATION_KEY]
            )
            from cayu.runtime._foreground_child_continuation import (
                load_attached_foreground_continuation,
            )

            assert await load_attached_foreground_continuation(parent, store=store) == continuation
            # A valid-looking reconstructed request is not authority to change
            # the original invocation. Keep the real insert-only receipt while
            # independently replacing decision-bearing checkpoint fields.
            for field in ("max_steps", "metadata", "event_digest"):
                altered_checkpoint = deepcopy(settled_checkpoint)
                altered = altered_checkpoint[FOREGROUND_PARENT_CONTINUATION_KEY]
                if field == "max_steps":
                    altered["request"][field] += 1
                elif field == "metadata":
                    altered["request"][field] = {"replacement": "different invocation"}
                else:
                    altered["terminal"][field] = "0" * 64
                try:
                    await store.checkpoint("parent", altered_checkpoint)
                    with pytest.raises(RuntimeError, match="exact publication receipt"):
                        await load_attached_foreground_continuation(parent, store=store)
                finally:
                    await store.checkpoint("parent", settled_checkpoint)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())
