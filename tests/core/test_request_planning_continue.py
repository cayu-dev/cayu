"""Prepared CONTINUE through the real deterministic planner and native receiver."""

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from hashlib import sha256

import pytest
from tests.core.test_participant_identity import app
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_recipient_continuation_selection import prepared_continue_scenario
from tests.core.test_request_planning_contracts import _policy
from tests.core.test_request_planning_public import complete_plan

from cayu.collaboration._contracts import CollaborationConflict, ExactMatch, ExactNotFound
from cayu.collaboration._preparation import contract_bytes
from cayu.collaboration.planning import (
    RequestPlanningContinue,
    RequestPlanningControl,
    RequestPlanningRequest,
    planning_policy_commitment,
)
from cayu.collaboration.prepared_admission import prepared_budget
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio


async def test_prepared_plan_terminal_ack_loss_reconstructs_without_receiver(
    native_stores, monkeypatch
):
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.collaboration.request_access import RequestRegistration

    application, resolver, command, expected, provider, session = await scenario(native_stores)
    store = native_stores[0]
    original = store._transaction
    lost = False

    @asynccontextmanager
    async def lose_terminal_ack(scope, *, write):
        nonlocal lost
        terminal = False
        async with original(scope, write=write) as tx:
            yield tx
            if write and not lost:
                stages = await tx.scan_request_plan_stages(command.operation, limit=2)
                terminal = bool(stages and stages[0]["state"] == "settled")
        if terminal and not lost:
            lost = True
            raise RuntimeError("committed prepared-plan acknowledgement lost")

    monkeypatch.setattr(store, "_transaction", lose_terminal_ack)
    try:
        with suppress(CollaborationUnavailable):
            await application.plan_collaboration_request(
                command, context=resolver.recipient.context
            )
        async with asyncio.timeout(180):
            await asyncio.gather(
                *tuple(application._request_coordinator._owners.pending), return_exceptions=True
            )
        assert lost
    finally:
        monkeypatch.setattr(store, "_transaction", original)

    # A new application/connection has neither the proposal policy nor the native
    # selection receiver. Only authenticated committed evidence can prove replay.
    reopened = app(
        native_stores[2](),
        application._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=60000),
    )
    await reopened.initialize_collaboration()
    found = await reopened.lookup_collaboration_plan(command, context=resolver.recipient.context)
    assert isinstance(found, ExactMatch)
    assert found.receipt.state == "admitted" and found.receipt.pending_stages == 0
    assert found.receipt.reserved_bytes == found.receipt.reserved_events == 0
    admission = await reopened.lookup_collaboration_admission(
        expected, context=resolver.sender.context
    )
    assert isinstance(admission, ExactMatch) and admission.receipt.command == expected
    assert (
        await reopened.reconcile_collaboration_plan(command, context=resolver.recipient.context)
        == found.receipt
    )
    assert (
        await reopened.plan_collaboration_request(command, context=resolver.recipient.context)
        == found.receipt
    )
    assert len(provider.requests) == 1
    assert (await application.session_store.load(session.id)).status == "completed"


async def scenario(native_stores):
    (
        original,
        resolver,
        admission,
        provider,
        session,
        initialized,
    ) = await prepared_continue_scenario(native_stores)
    proposal = RequestPlanningContinue(prepared=admission.prepared)
    policy = _policy()
    policy = policy.model_copy(
        update={
            "reference": policy.reference.model_copy(update={"owner": initialized.owner}),
            "rules": (),
            "default": proposal,
        }
    )

    class BudgetReceiver:
        async def resolve_budget_binding(self, *, request):
            return prepared_budget(admission.prepared.budget_binding_json)

    application = app(
        native_stores[0],
        original._participant_coordinator._registration,
        session_store=native_stores[1],
        collaboration_requests=replace(
            original._request_coordinator._registration,
            receiving_owner=None,
            planning_policies=(policy,),
        ),
        budget_binding_receiver=BudgetReceiver(),
        enable_common_root_budget_binding=True,
    )
    await application.initialize_collaboration()
    command = RequestPlanningRequest(
        operation=initialized.operation("continue-plan"),
        expected=admission.expected,
        expected_revision=admission.expected_revision,
        expected_input_revision=admission.expected_input_revision,
        expected_input_sha256=admission.expected_input_sha256,
        planning_generation=1,
        admission_operation=admission.operation,
        admission_generation=admission.generation,
        initiator=admission.initiator,
        policy=policy.reference,
        policy_sha256=planning_policy_commitment(policy, redactor=SecretRedactor()),
        limits=policy.limits,
        deadline_at_ms=admission.expected.intent.selection.expires_at_ms,
        predecessor=None,
    )
    expected_admission = admission.model_copy(
        update={
            "proposal_commitment": sha256(
                contract_bytes(proposal, redactor=SecretRedactor())
            ).hexdigest()
        }
    )
    return application, resolver, command, expected_admission, provider, session


async def test_preparation_publication_failure_rolls_back_stage_and_accounting(
    native_stores, monkeypatch
):
    from cayu.collaboration import _planning_stages
    from cayu.collaboration.participants import CollaborationUnavailable

    application, resolver, command, expected, provider, _ = await scenario(native_stores)
    application._request_coordinator._owners.observation_timeout = 180
    original = _planning_stages._write_transition
    before = None

    async def fail_after_write(store, tx, initialized, prior, updated, event, redactor):
        nonlocal before
        assert event.type == "plan_preparing"
        await original(store, tx, initialized, prior, updated, event, redactor)
        # A real first-stage write and its event have happened in this transaction.
        assert updated.stage_count == 1 and updated.pending_stages == 1
        before = prior.receipt.command
        raise RuntimeError("preparation publication interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(_planning_stages, "_write_transition", fail_after_write)
        with pytest.raises(CollaborationUnavailable):
            await application.plan_collaboration_request(
                command, context=resolver.recipient.context
            )
    assert before == command
    found = await application.lookup_collaboration_plan(command, context=resolver.recipient.context)
    assert isinstance(found, ExactMatch)
    retained = found.receipt
    assert retained.state == "decided" and retained.stage_count == retained.pending_stages == 0
    initialized = await application.initialize_collaboration()
    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        assert await tx.scan_request_plan_stages(command.operation, limit=2) == []
    assert isinstance(
        await application.lookup_collaboration_admission(expected, context=resolver.sender.context),
        ExactNotFound,
    )
    # Exact replay may now commit one stage/admission; no partial reservation
    # from the failed transaction remains to be charged or settled twice.
    result = await complete_plan(application, command, resolver.recipient.context)
    assert result.state == "admitted" and result.stage_count == 1
    assert result.pending_stages == result.reserved_bytes == result.reserved_events == 0
    assert len(provider.requests) == 1


async def test_public_planner_continue_admits_once_without_launch(native_stores):
    application, resolver, command, expected, provider, session = await scenario(native_stores)
    before = (
        await application.session_store.load(session.id),
        await application.session_store.load_checkpoint(session.id),
        await application.session_store.load_transcript(session.id),
    )
    record = await complete_plan(application, command, resolver.recipient.context)
    assert record.state == "admitted"
    assert record.stage_count == 1 and record.pending_stages == 0
    assert record.reserved_bytes == record.reserved_events == 0
    found = await application.lookup_collaboration_admission(
        expected, context=resolver.sender.context
    )
    assert isinstance(found, ExactMatch) and found.receipt.command == expected
    assert (
        await application.plan_collaboration_request(command, context=resolver.recipient.context)
        == record
    )
    assert (
        await application.reconcile_collaboration_plan(command, context=resolver.recipient.context)
        == record
    )
    assert before == (
        await application.session_store.load(session.id),
        await application.session_store.load_checkpoint(session.id),
        await application.session_store.load_transcript(session.id),
    )
    assert len(provider.requests) == 1
    # Characterize the exact individual-row bound using genuine admitted native
    # material, in addition to the public no-partial-mutation tests below.
    from cayu.collaboration._planning_preflight import preflight_stage_terminal
    from cayu.collaboration._planning_records import RequestPlanningStageRecord

    initialized = await application.initialize_collaboration()
    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        rows = await tx.scan_request_plan_stages(command.operation, limit=2)
    assert len(rows) == 1
    stage = RequestPlanningStageRecord.model_validate(rows[0]).model_copy(
        update={
            "state": "pending",
            "receipt": None,
            "settled_at_ms": None,
            "settlement_event": None,
        }
    )
    required = preflight_stage_terminal(
        stage, initialized, ceiling=65536, redactor=SecretRedactor()
    )
    with pytest.raises(ValueError, match="terminal evidence"):
        preflight_stage_terminal(
            stage, initialized, ceiling=required - 1, redactor=SecretRedactor()
        )
    for ceiling in (required, required + 1):
        assert (
            preflight_stage_terminal(stage, initialized, ceiling=ceiling, redactor=SecretRedactor())
            == required
        )


@pytest.mark.parametrize("cancel_observer", [False, True])
async def test_prepared_plan_exclusion_fences_delayed_receiver_and_raw_admission(
    native_stores, monkeypatch, cancel_observer
):
    application, resolver, command, expected, provider, session = await scenario(native_stores)
    store = application.session_store
    # This test delivers cancellation at the real receiver barrier, rather than
    # mistaking the separately covered finite observation timeout for that signal.
    application._request_coordinator._owners.observation_timeout = 180
    original = store.capture_recipient_continuation
    entered, release = asyncio.Event(), asyncio.Event()
    held = False

    async def held_capture(session_id):
        nonlocal held
        if not held:
            held = True
            entered.set()
            await release.wait()
        return await original(session_id)

    monkeypatch.setattr(store, "capture_recipient_continuation", held_capture)
    caller = asyncio.create_task(
        application.plan_collaboration_request(command, context=resolver.recipient.context)
    )
    try:
        await asyncio.wait_for(entered.wait(), 90)
        if cancel_observer and not caller.done():
            caller.cancel()
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert caller.cancelled() and caller.cancelling() == 2
        elif cancel_observer:
            # A finite observer timeout is not cancellation coverage. Keep this
            # regression explicit rather than accepting an already-ended waiter.
            await caller
            pytest.fail("Observer ended before cancellation barrier.")
        found = await application.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(found, ExactMatch)
        assert found.receipt.state == "preparing" and found.receipt.pending_stages == 1
        with pytest.raises(CollaborationConflict):
            await application.admit_collaboration_request(
                expected, context=resolver.recipient.context
            )
        control = RequestPlanningControl(
            expected=command,
            expected_revision=found.receipt.revision,
            kind="cancelled",
            initiator=command.initiator,
        )
        cleaned = await complete_plan(
            application, control, resolver.recipient.context, control=True
        )
        assert cleaned.state == "cancelled" and cleaned.pending_stages == 0
        release.set()
        if not cancel_observer:
            from cayu.collaboration.participants import CollaborationUnavailable

            with pytest.raises((CollaborationConflict, CollaborationUnavailable)):
                await caller
        pending = tuple(application._request_coordinator._owners.pending)
        async with asyncio.timeout(90):
            await asyncio.gather(*pending, return_exceptions=True)
        assert isinstance(
            await application.lookup_collaboration_admission(
                expected, context=resolver.sender.context
            ),
            ExactNotFound,
        )
        final = await application.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(final, ExactMatch) and final.receipt == cleaned
        assert len(provider.requests) == 1
        assert (await store.load(session.id)).status == "completed"
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
