"""A first-dispatch budget refusal must not strand registered output cleanup."""

from decimal import Decimal

import pytest
from tests.core.test_budget_binding import _binding, _limit
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration.exports import SessionExportAccessContext
from cayu.events import EventType


def authorize_execution(resolver):
    resolution = resolver.recipient.resolution
    actions = (*resolution.principal.actions, "execute", "publish")
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


@pytest.fixture
async def refusal_ledger(native_stores, tmp_path):
    from cayu.budgets.base import InMemoryBudgetLedger
    from cayu.storage.budget_ledger import SQLiteBudgetLedger
    from cayu.storage.budget_postgres import PostgresBudgetLedger
    from cayu.storage.migrations import SchemaMode

    backend, address = native_stores[3]
    if backend == "memory":
        ledger = InMemoryBudgetLedger()

        def factory():
            return ledger
    elif backend == "sqlite":

        def factory():
            return SQLiteBudgetLedger(tmp_path / "refusal-ledger.sqlite")

        ledger = factory()
    else:

        def factory():
            return PostgresBudgetLedger(address, schema_mode=SchemaMode.CREATE)

        ledger = factory()
    try:
        yield ledger, factory
    finally:
        if hasattr(ledger, "close"):
            await ledger.close()


@pytest.mark.anyio
async def test_public_producer_first_reservation_refusal_settles_without_dispatch(
    native_stores, refusal_ledger, monkeypatch
):
    def insufficient_binding(scope):
        root = "refusal-root:" + scope
        return _binding(
            binding_id="refusal-binding:" + scope,
            application_scope=scope,
            root_budget_id=root,
            limits=(
                _limit().model_copy(
                    update={"key": root, "max_estimated_cost": Decimal("0.000001")}
                ),
            ),
        )

    app, resolver, _, provider, session, _, command, execution = await output_scenario(
        native_stores,
        with_exports=True,
        budget_binding_factory=insufficient_binding,
        budget_ledger=refusal_ledger[0],
    )
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
    authorize_execution(resolver)
    events = []
    failure = None
    try:
        async for event in app.execute_producer_output(
            command, execution, context=CONTEXT, producer_context=resolver.recipient.context
        ):
            events.append(event)
    except Exception as error:
        failure = error
    assert not provider.requests
    assert any(event.type is EventType.BUDGET_RESERVATION_FAILED for event in events), failure
    assert not await app.budget_ledger._scan_reservation_records(session_id=session.id)
    completion = await app.retain_producer_completion(command, context=CONTEXT)
    assert completion.output.disposition != "answer"
    accounting = await app._run_limit_controller._read_producer_budget_settlement(command)
    assert accounting.reservation_count == 0
    assert accounting.native_failure_commitment is not None
    # A replacement controller has no provider or binding resolver. Only exact
    # native release, failure, and the original durable accounting owner suffice.
    from cayu.applications import CayuApp
    from cayu.budgets.base import InMemoryBudgetLedger

    missing_accounting = CayuApp(
        session_store=native_stores[1],
        budget_ledger=InMemoryBudgetLedger(),
        enable_logging=False,
    )
    with pytest.raises(LookupError, match="Original budget binding registration"):
        await missing_accounting._run_limit_controller._read_producer_budget_settlement(command)

    reopened_ledger = refusal_ledger[1]()
    try:
        reopened = CayuApp(
            session_store=native_stores[1], budget_ledger=reopened_ledger, enable_logging=False
        )
        assert (
            await reopened._run_limit_controller._read_producer_budget_settlement(command)
            == accounting
        )
    finally:
        if reopened_ledger is not refusal_ledger[0]:
            await reopened_ledger.close()
    disclosure = SessionExportAccessContext(
        principal=resolver.recipient.context.principal, mandate=resolver.recipient.context
    )
    elected = await app.publish_producer_outcome(command, context=disclosure)
    assert elected.command.outcome == "failed" and elected.command.export is None
    finalized = await app.settle_producer_output(command, context=CONTEXT)
    assert finalized.delivery == "excluded"
    assert not (
        await app.pending_producer_outputs(command.admission.prepared.recipient, context=CONTEXT)
    ).items
    await app.session_store.delete_session(session.id)
    assert await app.settle_producer_output(command, context=CONTEXT) == finalized
    assert not provider.requests


@pytest.mark.anyio
@pytest.mark.parametrize("disposition", ["answer", "failed"])
async def test_dispatched_producer_cannot_settle_against_empty_budget_inventory(
    monkeypatch, disposition
):
    from cayu.budgets.base import InMemoryBudgetLedger
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.providers.base import ModelStreamEvent
    from cayu.sessions.base import InMemorySessionStore

    collaboration = InMemoryCollaborationStore()
    stores = collaboration, InMemorySessionStore(), lambda: collaboration, ("memory", None)
    app, resolver, _, provider, session, _, command, execution = await output_scenario(
        stores, with_exports=True
    )
    if disposition == "answer":
        provider._batches = ((ModelStreamEvent.text_delta("answer"), ModelStreamEvent.completed()),)
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
    authorize_execution(resolver)
    async for _ in app.execute_producer_output(
        command, execution, context=CONTEXT, producer_context=resolver.recipient.context
    ):
        pass
    assert len(provider.requests) == 1
    completion = await app.retain_producer_completion(command, context=CONTEXT)
    assert completion.output.disposition == disposition
    ledger = app.budget_ledger
    assert type(ledger) is InMemoryBudgetLedger
    accounting = await app._run_limit_controller._read_producer_budget_settlement(command)
    assert accounting.reservation_count > 0
    # Model lost inventory while keeping the valid binding and native release.
    # Neither successful native output nor empty accounting is no-work evidence.
    inventory = ledger._reservation_ids_by_session.pop(session.id)
    try:
        with pytest.raises(ValueError, match="native accounting evidence"):
            await app._run_limit_controller._read_producer_budget_settlement(command)
    finally:
        ledger._reservation_ids_by_session[session.id] = inventory
    assert await app._run_limit_controller._read_producer_budget_settlement(command) == accounting
