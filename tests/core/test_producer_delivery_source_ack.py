"""Committed source delivery and cleanup receipts survive lost acknowledgements."""

from contextlib import asynccontextmanager

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration._producer_cleanup_finalization import finalization_operation
from cayu.collaboration._producer_cleanup_store import cleanup_operation
from cayu.collaboration._producer_delivery_store import delivery_operation
from cayu.collaboration._request_store import operation_key
from cayu.collaboration._session_export_store import digest
from cayu.collaboration.exports import SessionExportSettlementRequest
from cayu.collaboration.participants import CollaborationUnavailable


@pytest.mark.anyio
@pytest.mark.parametrize("phase", ["delivery", "acceptance", "cleanup", "finalization"])
async def test_source_delivery_cleanup_ack_loss_replays_exactly(native_stores, monkeypatch, phase):
    values, context = await completed_export_scenario(native_stores, monkeypatch)
    app, resolver, _, provider, session, _, command, _ = values
    destination = command.destinations[0]
    exported = await app.export_producer_output(command, destination.operation, context=context)
    await app.publish_producer_outcome(command, destination=destination.operation, context=context)
    source = await app.lookup_session_export(exported.request, context=context)
    assert isinstance(source, ExactMatch)
    policy = app._session_export_coordinator.registration.policy
    policy.register_export(
        source.receipt,
        payload_sha256=digest({"text": "retained answer", "artifact_commitments": []}),
        consumer_id=destination.recipient.participant_id,
    )
    policy.allowed_receipts.add(command.operation.caller_key)
    append = app.append_peer_content
    appends = []

    async def count_append(*args, **kwargs):
        receipt = await append(*args, **kwargs)
        appends.append(receipt)
        return receipt

    monkeypatch.setattr(app, "append_peer_content", count_append)
    delivery = phase in {"delivery", "acceptance"}
    operation = (
        delivery_operation(destination, app._secret_redactor)
        if delivery
        else (cleanup_operation if phase == "cleanup" else finalization_operation)(
            command, app._secret_redactor
        )
    )
    key = operation_key(operation)
    transaction = native_stores[0]._transaction
    lost = []

    @asynccontextmanager
    async def lose_committed_ack(scope, *, write):
        committed = False
        async with transaction(scope, write=write) as tx:
            yield tx
            if write and not lost:
                raw = await tx.get("operations", key)
                committed = isinstance(raw, dict) and (
                    not delivery or (raw.get("receipt") is not None) == (phase == "acceptance")
                )
        if committed:
            lost.append(phase)
            raise ConnectionError("Source responsibility commit acknowledgement lost")

    async def dispatch():
        if delivery:
            return await app.deliver_producer_output(
                command, destination.operation, context=context
            )
        return await app.settle_producer_output(command, context=CONTEXT)

    try:
        if not delivery:
            await app.deliver_producer_output(command, destination.operation, context=context)
            resolution = resolver.recipient.resolution
            actions = tuple(dict.fromkeys((*resolution.principal.actions, "release")))
            resolver.recipient.resolution = resolution.model_copy(
                update={
                    "principal": resolution.principal.model_copy(update={"actions": actions}),
                    "chain": resolution.chain.model_copy(
                        update={
                            "entries": tuple(
                                entry.model_copy(update={"actions": actions})
                                for entry in resolution.chain.entries
                            )
                        }
                    ),
                }
            )
            await app.settle_session_export(
                SessionExportSettlementRequest(
                    request=exported.request,
                    mode="release",
                    operation=exported.request.ref.operation.model_copy(
                        update={"caller_key": "release-before-producer-cleanup"}
                    ),
                ),
                context=context,
            )
        with monkeypatch.context() as fault:
            fault.setattr(native_stores[0], "_transaction", lose_committed_ack)
            with pytest.raises(CollaborationUnavailable):
                await dispatch()
        assert lost == [phase]
        result = await dispatch()
        assert await dispatch() == result
        assert len(provider.requests) == 1 and len(appends) == 1
        if delivery:
            assert result.receipt.status == "appended"
        else:
            assert result.delivery == "published"
            await native_stores[1].delete_session(session.id)
            assert await dispatch() == result
    finally:
        await app.drain_collaboration_requests()
