"""Public terminal/progress journeys after a human-gate epoch transfer."""

import asyncio
import json
import sys
from contextlib import aclosing, asynccontextmanager

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu import (
    ProducerProgressOccurrence,
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
)
from cayu.collaboration.requests import RequestControl
from cayu.events import EventType
from cayu.providers.base import ModelStreamEvent
from cayu.tools.user_input import UserInputTool


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("termination", "gate", "closure_kind"),
    [(kind, "input", "cancel") for kind in ("failure", "stop", "cancel")]
    + [
        ("stop_paused", gate, kind)
        for gate in ("input", "approval")
        for kind in ("cancel", "expire")
    ],
)
async def test_rebound_producer_terminal_and_progress(
    native_stores, monkeypatch, termination, gate, closure_kind
):
    from tests.core import test_prepared_admission_public as preparation

    setup = preparation.setup

    async def stop_disposition(store):
        values = list(await setup(store))
        values[4] = values[4].model_copy(update={"cancellation": "stop"})
        return tuple(values)

    if termination in ("stop", "stop_paused"):
        monkeypatch.setattr(preparation, "setup", stop_disposition)
    entered, release = asyncio.Event(), asyncio.Event()

    class Barrier(Tool):
        spec = ToolSpec(name="barrier", effect=ToolEffect.NONE, input_schema={"type": "object"})

        async def run(self, ctx, args):
            entered.set()
            await release.wait()
            return ToolResult(content="done")

    class Approval(ToolPolicy):
        async def authorize(self, request):
            return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL, reason="Review")

    values = await output_scenario(
        native_stores,
        with_exports=True,
        tools=(UserInputTool(), Barrier()),
        tool_policy=Approval() if gate == "approval" else None,
    )
    app, resolver, admission, provider, session, initialized, command, execution = values
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    original = resolver.recipient.resolution
    actions = (*original.principal.actions, "execute", "publish")
    resolver.recipient.resolution = original.model_copy(
        update={
            "principal": original.principal.model_copy(update={"actions": actions}),
            "chain": original.chain.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(update={"actions": actions})
                        for entry in original.chain.entries
                    )
                }
            ),
        }
    )
    provider._batches = (
        (
            ModelStreamEvent.tool_call(
                name="ask_user" if gate == "input" else "barrier",
                id="question",
                arguments={"question": "Continue?"} if gate == "input" else {},
            ),
            ModelStreamEvent.completed(),
        ),
        (
            ModelStreamEvent.tool_call(name="barrier", id="barrier", arguments={}),
            ModelStreamEvent.completed(),
        ),
        (
            ModelStreamEvent.error("qualified failure", cause=ValueError("qualified failure")),
            ModelStreamEvent.completed(),
        ),
    )
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
    paused = [
        event
        async for event in app.execute_producer_output(
            command, execution, context=CONTEXT, producer_context=resolver.recipient.context
        )
    ]
    question = next(
        event
        for event in paused
        if event.type
        is (
            EventType.SESSION_AWAITING_USER_INPUT
            if gate == "input"
            else EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
    )
    cancellations = []

    async def resume():
        try:
            if gate == "input":
                stream = app.resolve_user_input(
                    UserInputResponse(
                        session_id=session.id, input_id=question.payload["input_id"], answer="yes"
                    ),
                    context=CONTEXT,
                )
            else:
                stream = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session.id,
                        approval_id=question.payload["approval_id"],
                        tool_round_id=question.payload["tool_round_id"],
                        tool_call_id=question.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    ),
                    context=CONTEXT,
                )
            async with aclosing(stream) as events:
                return [event async for event in events]
        except asyncio.CancelledError:
            cancellations.append(True)
            raise

    async def progress(sequence, kind):
        prior = await app.inspect_collaboration_request(
            admission.expected, context=resolver.recipient.context
        )
        occurrence = ProducerProgressOccurrence(
            operation=initialized.operation("rebound-" + kind),
            expected_revision=prior.revision,
            sequence=sequence,
            kind=kind,
        )
        result = await app.record_producer_progress(
            command, occurrence, context=resolver.recipient.context
        )
        assert (
            await app.record_producer_progress(
                command, occurrence, context=resolver.recipient.context
            )
            == result
        )
        return result

    async def close():
        prior = await app.inspect_collaboration_request(
            admission.expected, context=resolver.sender.context
        )
        original_transaction = native_stores[0]._transaction

        @asynccontextmanager
        async def expired(scope, *, write):
            async with original_transaction(scope, write=write) as tx:

                async def now_ms():
                    return admission.expected.intent.selection.expires_at_ms

                tx.now_ms = now_ms
                yield tx

        if closure_kind == "expire":
            monkeypatch.setattr(native_stores[0], "_transaction", expired)
        control = RequestControl(
            operation=initialized.operation("close-rebound"),
            expected=admission.expected,
            expected_revision=prior.revision,
            kind=closure_kind,
        )
        if termination == "stop_paused" and closure_kind == "cancel":
            from tests.core.producer_stop_observation import close_with_native_observation_failure

            await close_with_native_observation_failure(
                app,
                control,
                resolver.sender.context,
                "cancel" if gate == "input" else "ack",
                monkeypatch,
            )
        else:
            await app.control_collaboration_request(control, context=resolver.sender.context)
        monkeypatch.setattr(native_stores[0], "_transaction", original_transaction)

    if termination == "stop_paused":
        from cayu.runtime._invocation_lifecycle import RebindInvocationCommand

        transition_owner = app._runtime_session_store
        original_apply = transition_owner.apply_invocation_lifecycle_command
        claimed, proceed = asyncio.Event(), asyncio.Event()

        async def competing_rebind(command):
            if type(command) is RebindInvocationCommand:
                claimed.set()
                await proceed.wait()
            return await original_apply(command)

        competing = None
        if closure_kind == "cancel":
            monkeypatch.setattr(
                transition_owner, "apply_invocation_lifecycle_command", competing_rebind
            )
            competing = asyncio.create_task(resume())
            await asyncio.wait_for(claimed.wait(), 90)
        await close()
        accepted = await app.service_producer_disposition(command, context=CONTEXT)
        assert accepted.disposition == "stop_accepted"
        assert await app.service_producer_disposition(command, context=CONTEXT) == accepted
        from cayu.sessions.base import SessionRunFenced, SessionStatusConflict

        rejection = SessionRunFenced if gate == "input" else SessionStatusConflict

        proceed.set()
        if competing is not None:
            with pytest.raises(rejection):
                await asyncio.wait_for(competing, 90)
        monkeypatch.setattr(transition_owner, "apply_invocation_lifecycle_command", original_apply)
        backend, address = native_stores[3]
        if backend != "memory":
            reader = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "tests.recovery.producer_paused_stop_reader",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(
                    reader.communicate(
                        json.dumps(
                            {
                                "backend": backend,
                                "address": address,
                                "command": command.model_dump(mode="json"),
                            }
                        ).encode()
                    ),
                    90,
                )
                assert reader.returncode == 0, err.decode()
                assert out.strip() == b"paused-stop-reconstructed"
            finally:
                if reader.returncode is None:
                    reader.kill()
                    await reader.wait()
        with pytest.raises(rejection):
            await resume()
        assert not entered.is_set()
        assert len(provider.requests) == 1
        completion = await app.retain_producer_completion(command, context=CONTEXT)
        assert completion.output.disposition == "stopped"
        final = await app.settle_producer_output(command, context=CONTEXT)
        assert await app.settle_producer_output(command, context=CONTEXT) == final
        assert not (
            await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
        ).items
        await app.drain_collaboration_requests()
        return

    task = asyncio.create_task(resume())
    try:
        await asyncio.wait_for(entered.wait(), 90)
        producing = await progress(1, "producing")
        assert producing.command.evidence.run_epoch > 1
        if termination == "stop":
            await close()
            accepted = await app.service_producer_disposition(command, context=CONTEXT)
            assert accepted.disposition == "stop_accepted"
            assert accepted.native_stop.run_epoch == producing.command.evidence.run_epoch
            assert await app.service_producer_disposition(command, context=CONTEXT) == accepted
            release.set()
            await asyncio.wait_for(task, 90)
        elif termination == "cancel":
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 90)
            assert task.cancelled() and task.cancelling() == 1 and cancellations == [True]
        else:
            release.set()
            await asyncio.wait_for(task, 90)
        completion = await app.retain_producer_completion(command, context=CONTEXT)
        assert completion.output.disposition == (
            "failed" if termination == "failure" else "stopped"
        )
        assert completion.output.run_epoch == producing.command.evidence.run_epoch
        if termination != "stop":
            published = await progress(2, "published")
            assert published.command.evidence.run_epoch == completion.output.run_epoch
            await close()
        final = await app.settle_producer_output(command, context=CONTEXT)
        assert await app.settle_producer_output(command, context=CONTEXT) == final
        assert len(provider.requests) == (3 if termination == "failure" else 2)
        assert not (
            await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
        ).items
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await app.drain_collaboration_requests()
