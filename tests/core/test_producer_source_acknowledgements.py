"""Source export intent and publication commits survive acknowledgement loss."""

from contextlib import asynccontextmanager

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.collaboration._producer_export_store import export_operation, read_export
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.participants import CollaborationUnavailable


@pytest.mark.anyio
@pytest.mark.parametrize("phase", ["prepared", "published"])
async def test_source_export_commit_reconciles_without_repeating_production(
    native_stores, monkeypatch, phase
):
    values, context = await completed_export_scenario(native_stores, monkeypatch)
    app, _, _, provider, _, initialized, command, _ = values
    destination = command.destinations[0]
    store = native_stores[0]
    transaction = store._transaction
    key = operation_key(export_operation(destination, app._secret_redactor))
    lost = []

    @asynccontextmanager
    async def lose_committed_ack(scope, *, write):
        committed = False
        async with transaction(scope, write=write) as tx:
            yield tx
            if write and not lost:
                raw = await tx.get("operations", key)
                committed = isinstance(raw, dict) and raw.get("state") == phase
        if committed:
            lost.append(phase)
            raise ConnectionError("Source export commit acknowledgement lost")

    try:
        with monkeypatch.context() as fault:
            fault.setattr(store, "_transaction", lose_committed_ack)
            with pytest.raises(CollaborationUnavailable):
                await app.export_producer_output(command, destination.operation, context=context)
        assert lost == [phase]
        async with transaction(initialized.owner.application_scope, write=False) as tx:
            retained = await read_export(tx, command, destination, redactor=app._secret_redactor)
        assert retained is not None and retained.state == phase
        exported = await app.export_producer_output(command, destination.operation, context=context)
        assert exported.state == "published"
        assert (
            await app.export_producer_output(command, destination.operation, context=context)
            == exported
        )
        assert len(provider.requests) == 1
        projector = app._session_export_coordinator.projectors[destination.projector]
        assert projector.calls == 1
    finally:
        await app.drain_collaboration_requests()
