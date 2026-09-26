"""Real multiple producers share one bounded, atomic native reclamation batch."""

from dataclasses import replace

import pytest
from tests.core import test_prepared_admission_public as preparations
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_collaboration_namespace import rotate
from tests.core.test_participant_identity import CONTEXT, registration
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.budgets import InMemoryBudgetLedger
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestControl
from cayu.storage import _producer_retirement


@pytest.mark.anyio
async def test_public_reclamation_validates_whole_batch_before_removing_any_receipt(
    native_stores, monkeypatch
):
    reg = registration()
    reg = replace(
        reg,
        bootstrap=reg.bootstrap.model_copy(
            update={
                "limits": reg.bootstrap.limits.model_copy(
                    update={"retained_bytes": 16 * 1024 * 1024, "operations": 512, "events": 1024}
                )
            }
        ),
    )
    setup = preparations.setup

    async def shared_setup(store):
        return await setup(store, reg=reg)

    monkeypatch.setattr(preparations, "setup", shared_setup)
    ledger = InMemoryBudgetLedger()
    commands, providers, retained = [], [], []
    app = None
    try:
        for number in range(2):
            values, _ = await completed_export_scenario(
                native_stores,
                monkeypatch,
                operation_prefix=f"batch-{number}:",
                budget_ledger=ledger,
            )
            app, resolver, admission, provider, _, initialized, command, _ = values
            snapshot = await app.inspect_collaboration_request(
                admission.expected, context=resolver.sender.context
            )
            await app.control_collaboration_request(
                RequestControl(
                    operation=initialized.operation(f"batch-close-{number}"),
                    expected=admission.expected,
                    expected_revision=snapshot.revision,
                    kind="cancel",
                ),
                context=resolver.sender.context,
            )
            await app.settle_producer_output(command, context=CONTEXT)
            async with native_stores[0]._transaction(
                initialized.owner.application_scope, write=False
            ) as tx:
                retained.append(
                    await read_output_registration(tx, command, redactor=app._secret_redactor)
                )
            commands.append(command)
            providers.append(provider)
        assert commands[0].operation != commands[1].operation
        assert commands[0].operation.application_scope == commands[1].operation.application_scope
        _, rotated = await rotate(native_stores[0], initialized)
        await app.retire_collaboration_namespace(
            NamespaceRetire(
                operation=rotated.successor.reference.operation("retire-two-producers"),
                namespace=rotated.namespace.reference,
                expected_revision=rotated.namespace.revision,
                expected_retired_through=0,
            ),
            context=CONTEXT,
        )
        for number in range(32):
            state = await app.inspect_collaboration_namespace(context=CONTEXT)
            pruned = await app.prune_collaboration_namespace(
                NamespacePrune(
                    operation=rotated.successor.reference.operation(f"prune-two-{number}"),
                    namespace=rotated.namespace.reference,
                    expected_retention_revision=state.retention_revision,
                    max_records=32,
                ),
                context=CONTEXT,
            )
            if pruned.complete:
                break
        else:
            pytest.fail("Bounded source reclamation did not finish")
        before = [
            await app.session_store._read_completed_native_producer_cleanup(record)
            for record in retained
        ]
        assert all(receipt is not None for receipt in before)
        validate = _producer_retirement.validate_retiring_receipt
        observed = []

        def fail_second(*args):
            receipt = validate(*args)
            observed.append(receipt.registration)
            if len(observed) == 2:
                raise RuntimeError("Second native receipt validation unavailable")
            return receipt

        with monkeypatch.context() as fault:
            fault.setattr(_producer_retirement, "validate_retiring_receipt", fail_second)
            with pytest.raises(CollaborationUnavailable):
                await app.reclaim_producer_cleanup(
                    rotated.namespace.reference, context=CONTEXT, limit=2
                )
        assert len(observed) == 2
        assert [
            await app.session_store._read_completed_native_producer_cleanup(record)
            for record in retained
        ] == before
        first = await app.reclaim_producer_cleanup(
            rotated.namespace.reference, context=CONTEXT, limit=1
        )
        assert first.removed == 1 and first.remaining
        second = await app.reclaim_producer_cleanup(
            rotated.namespace.reference, context=CONTEXT, limit=1
        )
        assert second.removed == 1 and not second.remaining
        empty = await app.reclaim_producer_cleanup(
            rotated.namespace.reference, context=CONTEXT, limit=32
        )
        assert empty.removed == 0 and not empty.remaining
        assert [len(provider.requests) for provider in providers] == [1, 1]
    finally:
        if app is not None:
            await app.drain_collaboration_requests()
