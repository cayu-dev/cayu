"""Closed requests control only their frozen native producer invocation."""

import asyncio

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._request_store import retained_request
from cayu.collaboration.requests import RequestControl
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._producer_stop import accept_native_producer_stop
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec


@pytest.mark.anyio
@pytest.mark.parametrize("observation_failure", ["ordinary", "cancel", "ack"])
async def test_closed_producer_uses_exact_native_stop_scope(
    native_stores, monkeypatch, observation_failure
):
    entered, release = asyncio.Event(), asyncio.Event()

    class BarrierTool(Tool):
        spec = ToolSpec(name="barrier", effect=ToolEffect.NONE, input_schema={"type": "object"})

        @property
        def execution_profile_identity(self):
            return ExecutionProfileBehaviorIdentity(
                name="tests:planned-producer-barrier",
                behavior_version="1",
                implementation_version="1",
            )

        async def run(self, ctx, args):
            entered.set()
            await release.wait()
            return ToolResult(content="complete")

    (
        app,
        resolver,
        admission,
        provider,
        session,
        initialized,
        proposal,
        execution,
    ) = await output_scenario(
        native_stores, with_exports=True, tools=(BarrierTool(),), planned=True
    )
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    await app.register_producer_output(proposal, execution, context=resolver.recipient.context)
    original = resolver.recipient.resolution
    actions = (*original.principal.actions, "execute")
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
            ModelStreamEvent.tool_call(name="barrier", id="barrier-call", arguments={}),
            ModelStreamEvent.completed(),
        ),
        (ModelStreamEvent.text_delta("late completed answer"), ModelStreamEvent.completed()),
    )

    async def execute():
        return [
            event
            async for event in app.execute_producer_output(
                proposal,
                execution,
                context=CONTEXT,
                producer_context=resolver.recipient.context,
            )
        ]

    task = asyncio.create_task(execute())
    try:
        await asyncio.wait_for(entered.wait(), 60)
        async with native_stores[0]._transaction(
            initialized.owner.application_scope, write=False
        ) as tx:
            snapshot = await retained_request(
                native_stores[0],
                tx,
                initialized,
                admission.expected.intent.request,
                admission.expected.initiator,
                app._secret_redactor,
            )
        assert snapshot is not None
        from tests.core.producer_stop_observation import close_with_native_observation_failure

        control = await close_with_native_observation_failure(
            app,
            RequestControl(
                operation=initialized.operation("close-running-producer"),
                expected=admission.expected,
                expected_revision=snapshot.revision,
                kind="cancel",
            ),
            resolver.sender.context,
            observation_failure,
            monkeypatch,
        )
        async with native_stores[0]._transaction(
            initialized.owner.application_scope, write=False
        ) as tx:
            registration = await read_output_registration(
                tx, proposal, redactor=app._secret_redactor
            )
        with pytest.raises(PermissionError):
            await accept_native_producer_stop(
                native_stores[1], registration, control, authority=None
            )
        # Closure changes no native state until the registered owner services it.
        assert (await native_stores[1].load(session.id)).status == "running"
        accepted = await app.service_producer_disposition(proposal, context=CONTEXT)
        assert accepted.disposition == "stop_accepted"
        assert await app.service_producer_disposition(proposal, context=CONTEXT) == accepted
        assert not task.done()
        pending = await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
        assert any(item.recovery.registration == proposal.operation for item in pending.items)
        release.set()
        await asyncio.wait_for(task, 90)
        assert len(provider.requests) == 1
        assert (await native_stores[1].load(session.id)).status == "interrupted"
        assert await app.service_producer_disposition(proposal, context=CONTEXT) == accepted
        pending = await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
        assert any(item.recovery.registration == proposal.operation for item in pending.items)
        completion = await app.retain_producer_completion(proposal, context=CONTEXT)
        assert completion.output.disposition == "stopped"
        final = await app.settle_producer_output(proposal, context=CONTEXT)
        assert final.delivery == "excluded"
        async with native_stores[0]._transaction(
            initialized.owner.application_scope, write=False
        ) as tx:
            closed = await retained_request(
                native_stores[0],
                tx,
                initialized,
                admission.expected.intent.request,
                admission.expected.initiator,
                app._secret_redactor,
            )
            settled = await read_output_registration(tx, proposal, redactor=app._secret_redactor)
        assert closed is not None
        assert closed.state == "cancelled" and closed.outcome is None
        assert closed.terminal == control and closed.producer_settlement == final.operation
        assert settled.cleanup.evidence.terminal_kind == "closure"
        assert settled.cleanup.evidence.terminal == control.expected.operation
        assert (settled.reserved_operations, settled.reserved_events, settled.reserved_bytes) == (
            0,
            0,
            0,
        )
        await native_stores[1].delete_session(session.id)
        assert await app.settle_producer_output(proposal, context=CONTEXT) == final
        assert await app.retain_producer_completion(proposal, context=CONTEXT) == completion
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
