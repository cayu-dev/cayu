"""Producer cleanup qualifies its ledger before work and isolates native incarnations."""

from uuid import uuid4

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_budget_binding import _binding
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_budget_refusal import authorize_execution
from tests.core.test_producer_budget_refusal import refusal_ledger as refusal_ledger
from tests.core.test_producer_output_contracts import output_scenario

from cayu import ProducerDeliveryRecovery
from cayu.budgets.base import InMemoryBudgetLedger
from cayu.collaboration.participants import CollaborationUnavailable


@pytest.mark.anyio
async def test_replacement_producer_settles_with_old_budget_history(
    native_stores, refusal_ledger, monkeypatch
):
    public_id = "producer-reused-" + uuid4().hex
    instances = []
    counts = []
    for generation in ("first", "replacement"):

        def binding(scope, generation=generation):
            return _binding(application_scope=scope, binding_id=public_id + generation)

        values, context = await completed_export_scenario(
            native_stores,
            monkeypatch,
            requested_session_id=public_id,
            operation_prefix=generation + ":",
            budget_ledger=refusal_ledger[0],
            budget_binding_factory=binding,
        )
        app, _, _, provider, session, _, command, _ = values
        instances.append(session.instance_id)
        destination = command.destinations[0]
        await app.export_producer_output(command, destination.operation, context=context)
        await app.publish_producer_outcome(
            command, destination=destination.operation, context=context
        )
        pending = await app.pending_producer_outputs(
            command.admission.prepared.recipient, context=CONTEXT
        )
        token = next(
            item.recovery
            for item in pending.items
            if item.recovery.registration == command.operation
        )
        await app.reconcile_producer_delivery(
            ProducerDeliveryRecovery(**token.model_dump(), destination=destination.operation),
            context=CONTEXT,
            exclude=True,
        )
        await app.retire_producer_export(command, destination.operation, context=CONTEXT)
        rows = await refusal_ledger[0]._scan_reservation_records(session_id=session.id)
        assert all(
            row.settlement_event_payload.get("session_instance_id") in instances for row in rows
        ), [row.settlement_event_payload for row in rows]
        accounting = await app._run_limit_controller._read_producer_budget_settlement(command)
        counts.append(accounting.reservation_count)
        reopened = refusal_ledger[1]()
        if reopened is not refusal_ledger[0]:
            from cayu.applications import CayuApp

            try:
                recovered = CayuApp(
                    session_store=app.session_store, budget_ledger=reopened, enable_logging=False
                )
                assert (
                    await recovered._run_limit_controller._read_producer_budget_settlement(command)
                    == accounting
                )
            finally:
                await reopened.close()
        if native_stores[3][0] == "memory":
            # Ambiguous history and conflicting current authority are not
            # silently filtered out merely because another valid row exists.
            for row in rows:
                for field, replacement in (
                    ("session_instance_id", None),
                    ("budget_binding_id", "foreign"),
                ):
                    payload = dict(row.settlement_event_payload)
                    payload[field] = replacement
                    with monkeypatch.context() as patch:
                        patch.setitem(
                            refusal_ledger[0]._records,
                            row.reservation_id,
                            row.model_copy(update={"settlement_event_payload": payload}),
                        )
                        with pytest.raises((ValueError, CollaborationUnavailable)):
                            await app.settle_producer_output(command, context=CONTEXT)
        settled = await app.settle_producer_output(command, context=CONTEXT)
        assert settled.delivery == "excluded"
        assert len(provider.requests) == 1
        assert not (
            await app.pending_producer_outputs(
                command.admission.prepared.recipient, context=CONTEXT
            )
        ).items
        if generation == "first":
            await app.session_store.delete_session(session.id)
    assert instances[0] != instances[1]
    assert counts[0] == counts[1] > 0
    rows = await refusal_ledger[0]._scan_reservation_records(session_id=public_id)
    assert len(rows) == sum(counts)
    assert {row.settlement_event_payload["session_instance_id"] for row in rows} == set(instances)


@pytest.mark.anyio
@pytest.mark.parametrize("phase", ["registration", "launch"])
async def test_unqualified_producer_ledger_refused_before_dispatch(
    native_stores, phase, monkeypatch
):
    class UnqualifiedLedger(InMemoryBudgetLedger):
        async def _scan_reservation_records(self, *, session_id, after=None, limit=128):
            return await super()._scan_reservation_records(
                session_id=session_id, after=after, limit=limit
            )

    ledger = UnqualifiedLedger() if phase == "registration" else InMemoryBudgetLedger()
    app, resolver, _, provider, _, _, command, execution = await output_scenario(
        native_stores, with_exports=True, budget_ledger=ledger
    )
    if phase == "registration":
        with pytest.raises(CollaborationUnavailable, match="accounting readback"):
            await app.register_producer_output(
                command, execution, context=resolver.recipient.context
            )
        assert not (
            await app.pending_producer_outputs(
                command.admission.prepared.recipient, context=CONTEXT
            )
        ).items
    else:
        await app.register_producer_output(command, execution, context=resolver.recipient.context)
        authorize_execution(resolver)
        with monkeypatch.context() as patch:
            patch.setattr(app._run_limit_controller, "_budget_ledger", UnqualifiedLedger())
            with pytest.raises(CollaborationUnavailable) as refusal:
                async for _ in app.execute_producer_output(
                    command, execution, context=CONTEXT, producer_context=resolver.recipient.context
                ):
                    pass
            assert str(refusal.value.__cause__) == "Producer accounting readback is not qualified."
        assert (
            await app.pending_producer_outputs(
                command.admission.prepared.recipient, context=CONTEXT
            )
        ).items
    assert not provider.requests
