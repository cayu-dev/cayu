"""Registered deterministic planning through the real application entrance."""

import asyncio
import warnings
from contextlib import asynccontextmanager

import pytest
from examples.collaboration.planning import plan_once
from tests.core.test_collaboration_request_foundation import setup
from tests.core.test_participant_identity import CONTEXT, app, create, registration, stores
from tests.core.test_participant_lifecycle import change
from tests.core.test_prepared_admission_public import PreparationResolver
from tests.core.test_request_planning_contracts import _policy

from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ExactConflict,
    ExactMatch,
)
from cayu.collaboration._request_coordinator import _initiator
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.participants import CollaborationCapacityExceeded, CollaborationUnavailable
from cayu.collaboration.planning import (
    RequestPlanningControl,
    RequestPlanningDecline,
    RequestPlanningDefer,
    RequestPlanningPredecessor,
    RequestPlanningRequest,
    RequestPlanningTimer,
    planning_policy_commitment,
)
from cayu.collaboration.request_access import RequestRegistration
from cayu.evals.testing import ScriptedModelProvider
from cayu.vaults.redaction import SecretRedactor

__all__ = ["stores"]
pytestmark = pytest.mark.anyio
REDACTOR = SecretRedactor()


async def lookup_plan(application, expected, context):
    """Observe the exact public read worker, not a new mutation or guessed result."""
    owners = application._request_coordinator._owners
    before = set(owners.pending)
    try:
        return await application.lookup_collaboration_plan(expected, context=context)
    except CollaborationUnavailable:
        pending = set(owners.pending) - before
        if len(pending) != 1:
            raise
        done, remaining = await asyncio.wait(pending, timeout=180)
        assert not remaining, "Exact planning read remains owned after bounded observation."
        return next(iter(done)).result()


async def complete_plan(application, command, context, *, control=False, reconcile=False):
    """Observe existing ownership, then use authenticated exact public readback."""
    try:
        if control:
            return await application.control_collaboration_plan(command, context=context)
        if reconcile:
            return await application.reconcile_collaboration_plan(command, context=context)
        return await plan_once(application, command, context=context)
    except CollaborationUnavailable:
        pending = tuple(application._request_coordinator._owners.pending)
        if not pending:
            raise
        # This is an observation bound, not an execution-cancellation test.
        # Keep the real owned mutations alive if this diagnostic wait expires.
        expected = command.expected if control else command
        lifetime = (
            expected.deadline_at_ms - expected.expected.intent.selection.accepted_at_ms
        ) / 1000
        done, remaining = await asyncio.wait(
            pending,
            timeout=max(
                180, lifetime + application._request_coordinator._owners.observation_timeout
            ),
        )
        assert not remaining, "Planning mutations remain owned after bounded observation."
        for task in done:
            # Observation timing must not change the public error contract.
            await application._request_coordinator._observe(task)
        found = await lookup_plan(application, command.expected if control else command, context)
        assert isinstance(found, ExactMatch)
        if control:
            assert found.receipt.control == command
        return found.receipt


async def test_public_plan_cleanup_after_disablement_is_exact_and_releases_reserves(stores):
    store = stores()
    application, resolver, command, _, provider = await scenario(store, "defer")
    first = await complete_plan(application, command, resolver.recipient.context)
    control = RequestPlanningControl(
        expected=command,
        expected_revision=first.revision,
        kind="cancelled",
        initiator=_initiator(resolver.recipient.context),
    )
    with pytest.raises(CollaborationConflict):
        await application.control_collaboration_plan(
            control.model_copy(update={"kind": "expired"}), context=resolver.recipient.context
        )
    selected = command.expected.intent.selection.recipient
    current = await application.inspect_participant(selected.reference, context=CONTEXT)
    initialized = await application.initialize_collaboration()
    await application.change_participant_lifecycle(
        change(
            initialized,
            selected.reference,
            key="planning-disabled-cleanup",
            revision=current.participant.lifecycle_revision,
            state="disabled",
        ),
        context=CONTEXT,
    )
    result = await application.control_collaboration_plan(
        control, context=resolver.recipient.context
    )
    assert result.state == "cancelled" and result.control == control
    assert result.reserved_bytes == result.reserved_events == result.pending_stages == 0
    assert (
        await application.plan_collaboration_request(command, context=resolver.recipient.context)
        == result
    )
    assert (
        await application.control_collaboration_plan(control, context=resolver.recipient.context)
        == result
    )
    with pytest.raises(CollaborationConflict):
        await application.control_collaboration_plan(
            control.model_copy(update={"expected_revision": first.revision + 1}),
            context=resolver.recipient.context,
        )
    found = await application.lookup_collaboration_plan(command, context=resolver.recipient.context)
    assert isinstance(found, ExactMatch) and found.receipt == result
    assert provider.requests == []


@pytest.mark.parametrize("decline", [False, True])
async def test_public_timer_successor_is_explicit_atomic_and_owner_timed(
    stores, monkeypatch, decline
):
    store = stores()
    application, resolver, command, policy, provider = await scenario(store, "defer")
    first = await application.plan_collaboration_request(
        command, context=resolver.recipient.context
    )
    successor = command.model_copy(
        update={
            "operation": command.operation.model_copy(update={"caller_key": "second-plan"}),
            "admission_operation": command.admission_operation.model_copy(
                update={"caller_key": "second-admission"}
            ),
            "expected_revision": 2,
            "planning_generation": 2,
            "admission_generation": 2,
            "predecessor": RequestPlanningPredecessor(
                operation=command.operation, revision=first.revision
            ),
        }
    )
    if decline:
        next_policy = policy.model_copy(
            update={
                "reference": policy.reference.model_copy(update={"revision": 2}),
                "default": RequestPlanningDecline(reason="unsupported_request"),
            }
        )
        successor = successor.model_copy(
            update={
                "policy": next_policy.reference,
                "policy_sha256": planning_policy_commitment(next_policy, redactor=REDACTOR),
            }
        )
        application = app(
            store,
            application._participant_coordinator._registration,
            collaboration_requests=RequestRegistration(
                mandates=resolver, max_ttl_ms=60000, planning_policies=(next_policy,)
            ),
        )
        await application.initialize_collaboration()
    with pytest.raises(CollaborationConflict):
        await application.plan_collaboration_request(successor, context=resolver.recipient.context)
    assert (
        await application.lookup_collaboration_plan(command, context=resolver.recipient.context)
    ).receipt == first
    original = store._transaction

    @asynccontextmanager
    async def due_owner(scope, *, write):
        async with original(scope, write=write) as tx:

            async def now_ms():
                return policy.default.prerequisite.not_before_ms

            tx.now_ms = now_ms
            yield tx

    monkeypatch.setattr(store, "_transaction", due_owner)
    second = await complete_plan(application, successor, resolver.recipient.context)
    assert second.state == ("declined" if decline else "deferred")
    assert second.receipt.command == successor
    prior = await application.lookup_collaboration_plan(command, context=resolver.recipient.context)
    assert isinstance(prior, ExactMatch) and prior.receipt.state == "superseded"
    assert prior.receipt.reserved_bytes == 0 and prior.receipt.pending_stages == 0
    assert (
        await application.plan_collaboration_request(successor, context=resolver.recipient.context)
        == second
    )
    assert provider.requests == []
    if decline:
        request = await application.inspect_collaboration_request(
            command.expected, context=resolver.recipient.context
        )
        assert request is not None and request.state == "declined"
        participant = await application.inspect_participant(
            command.expected.intent.selection.recipient.reference, context=CONTEXT
        )
        assert participant.outstanding_obligations == 0


async def scenario(store, decision, *, limits=None, reg=None):
    original, initial, _, recipient, request, _ = await setup(store, reg=reg)
    registration = original._participant_coordinator._registration
    resolver = PreparationResolver(request, recipient.reference)
    accepting = app(
        store,
        registration,
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=60000),
    )
    await accepting.initialize_collaboration()
    accepted = await accepting.accept_collaboration_request(
        request, context=resolver.sender.context
    )
    deadline = accepted.expected.intent.selection.expires_at_ms
    proposal = (
        RequestPlanningDecline(reason="unsupported_request")
        if decision == "decline"
        else RequestPlanningDefer(
            prerequisite=RequestPlanningTimer(
                not_before_ms=deadline - 1000, deadline_at_ms=deadline
            )
        )
    )
    policy = _policy()
    if limits is not None:
        policy = policy.model_copy(update={"limits": limits})
    policy = policy.model_copy(
        update={
            "reference": policy.reference.model_copy(update={"owner": initial.owner}),
            "rules": (),
            "default": proposal,
        }
    )
    provider = ScriptedModelProvider([])
    application = app(
        store,
        registration,
        collaboration_requests=RequestRegistration(
            mandates=resolver, max_ttl_ms=60000, planning_policies=(policy,)
        ),
    )
    application.register_provider(provider, default=True)
    await application.initialize_collaboration()
    command = RequestPlanningRequest(
        operation=initial.operation("public-plan"),
        expected=accepted.expected,
        expected_revision=1,
        expected_input_revision=0,
        expected_input_sha256=clarification_commitment(accepted.expected, REDACTOR),
        planning_generation=1,
        admission_operation=initial.operation("public-plan-admission"),
        admission_generation=1,
        initiator=_initiator(resolver.recipient.context),
        policy=policy.reference,
        policy_sha256=planning_policy_commitment(policy, redactor=REDACTOR),
        limits=policy.limits,
        deadline_at_ms=deadline,
        predecessor=None,
    )
    return application, resolver, command, policy, provider


@pytest.mark.parametrize("dimension", ["operations", "events", "retained_bytes"])
async def test_public_planning_cleanup_survives_full_aggregate_capacity(stores, dimension):
    limits = registration().bootstrap.limits.model_copy(
        update={dimension: {"operations": 24, "events": 32, "retained_bytes": 2_000_000}[dimension]}
    )
    store = stores()
    application, resolver, command, _, provider = await scenario(
        store, "defer", reg=registration(limits=limits)
    )
    planned = await complete_plan(application, command, resolver.recipient.context)
    initialized = await application.initialize_collaboration()

    async def anchor():
        async with store._transaction(initialized.binding.application_scope, write=False) as tx:
            return await store._anchor(tx, initialized, REDACTOR)

    # Real accepted requests consume both retained evidence and future settlement
    # reservations. A rejected request must not leave either charge behind.
    for index in range(32):
        before = await anchor()
        additional = command.expected.intent.request.model_copy(
            update={"operation": initialized.operation(f"capacity-request-{index}")}
        )
        try:
            await application.accept_collaboration_request(
                additional, context=resolver.sender.context
            )
        except CollaborationCapacityExceeded:
            assert await anchor() == before
            break
    else:
        pytest.fail("Aggregate request capacity was not enforced")
    # Consume the smaller remaining slots without inventing ledger charges.
    for index in range(64):
        before = await anchor()
        try:
            await create(application, initialized, f"capacity-fill-{index}")
        except CollaborationCapacityExceeded:
            assert await anchor() == before
            break
    else:
        pytest.fail("Ordinary publication capacity was not enforced")
    full = await anchor()
    if dimension == "operations":
        assert (
            full.operation_count + full.reserved_operations
            == limits.operations - limits.control_operations
        )
    elif dimension == "events":
        assert full.event_count + full.reserved_events == limits.events - limits.control_events
    else:
        assert full.retained_bytes < limits.retained_bytes
        assert full.reserved_bytes < limits.retained_bytes
    # Replay is not another reservation, even at the admission ceiling.
    assert await complete_plan(application, command, resolver.recipient.context) == planned
    assert await anchor() == full
    control = RequestPlanningControl(
        expected=command,
        expected_revision=planned.revision,
        kind="cancelled",
        initiator=_initiator(resolver.recipient.context),
    )
    settled = await complete_plan(application, control, resolver.recipient.context, control=True)
    assert settled.state == "cancelled"
    assert settled.reserved_bytes == settled.reserved_events == 0
    after = await anchor()
    assert after.reserved_bytes == full.reserved_bytes - planned.reserved_bytes
    assert after.reserved_events == full.reserved_events - planned.reserved_events
    assert after.event_count == full.event_count + 1
    assert after.operation_count == full.operation_count
    assert (
        await complete_plan(application, control, resolver.recipient.context, control=True)
        == settled
    )
    assert await anchor() == after
    assert provider.requests == []


async def test_public_successors_cannot_widen_the_retained_generation_ceiling(stores, monkeypatch):
    store = stores()
    limits = _policy().limits.model_copy(update={"max_generations": 2})
    application, resolver, command, policy, provider = await scenario(store, "defer", limits=limits)
    transaction = store._transaction

    @asynccontextmanager
    async def due_owner(scope, *, write):
        async with transaction(scope, write=write) as tx:

            async def now_ms():
                return policy.default.prerequisite.not_before_ms

            tx.now_ms = now_ms
            yield tx

    monkeypatch.setattr(store, "_transaction", due_owner)
    current = await complete_plan(application, command, resolver.recipient.context)
    for generation in (2, 3):
        previous = current.receipt.command
        next_command = previous.model_copy(
            update={
                "operation": previous.operation.model_copy(
                    update={"caller_key": f"bounded-plan-{generation}"}
                ),
                "admission_operation": previous.admission_operation.model_copy(
                    update={"caller_key": f"bounded-admission-{generation}"}
                ),
                "expected_revision": generation,
                "planning_generation": generation,
                "admission_generation": generation,
                "predecessor": RequestPlanningPredecessor(
                    operation=previous.operation, revision=current.revision
                ),
            }
        )
        if generation == 2:
            current = await complete_plan(application, next_command, resolver.recipient.context)
            assert current.state == "deferred"
            continue
        wider = policy.model_copy(
            update={
                "reference": policy.reference.model_copy(update={"revision": 2}),
                "limits": limits.model_copy(update={"max_generations": 3}),
            }
        )
        next_command = next_command.model_copy(
            update={
                "limits": wider.limits,
                "policy": wider.reference,
                "policy_sha256": planning_policy_commitment(wider, redactor=REDACTOR),
            }
        )
        replacement = app(
            store,
            application._participant_coordinator._registration,
            collaboration_requests=RequestRegistration(
                mandates=resolver, max_ttl_ms=60000, planning_policies=(wider,)
            ),
        )
        await replacement.initialize_collaboration()
        with pytest.raises(CollaborationConflict):
            await replacement.plan_collaboration_request(
                next_command, context=resolver.recipient.context
            )
        assert (
            await replacement.lookup_collaboration_plan(
                previous, context=resolver.recipient.context
            )
        ).receipt == current
    assert provider.requests == []


@pytest.mark.parametrize("decision", ["defer", "decline"])
async def test_public_non_dispatching_plan_and_exact_restart_readback(stores, decision):
    application, resolver, command, policy, provider = await scenario(stores(), decision)
    result = await application.plan_collaboration_request(
        command, context=resolver.recipient.context
    )
    assert result.state == ("deferred" if decision == "defer" else "declined")
    assert result.decision == policy.default
    assert result.pending_stages == 0
    assert provider.requests == []
    found = await application.lookup_collaboration_plan(command, context=resolver.sender.context)
    assert isinstance(found, ExactMatch) and found.receipt == result
    # Replay reconstructs committed evidence without the old policy registration.
    reopened = app(
        stores(),
        application._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=60000),
    )
    await reopened.initialize_collaboration()
    page = await reopened.list_pending_collaboration_plans(
        context=resolver.recipient.context, limit=1
    )
    assert page.records == ((result,) if decision == "defer" else ())
    if page.next_cursor is not None:
        assert not (
            await reopened.list_pending_collaboration_plans(
                context=resolver.recipient.context, after=page.next_cursor, limit=1
            )
        ).records
    access = reopened._participant_coordinator._registration.access_policy
    allowed = access.allowed
    access.allowed = (command.expected.intent.selection.recipient.reference,)
    try:
        with pytest.raises(CollaborationAccessDenied):
            await reopened.list_pending_collaboration_plans(context=resolver.recipient.context)
    finally:
        access.allowed = allowed
    assert (
        await reopened.plan_collaboration_request(command, context=resolver.recipient.context)
        == result
    )
    changed = command.model_copy(update={"deadline_at_ms": command.deadline_at_ms - 1})
    assert isinstance(
        await reopened.lookup_collaboration_plan(changed, context=resolver.sender.context),
        ExactConflict,
    )
    assert provider.requests == []


async def test_public_deactivation_orders_before_plan_retention(stores, monkeypatch):
    store = stores()
    application, resolver, command, _, provider = await scenario(store, "decline")
    other = app(stores(), application._participant_coordinator._registration)
    initial = await other.initialize_collaboration()
    entered, release = asyncio.Event(), asyncio.Event()
    original = store._transaction
    pause = True

    @asynccontextmanager
    async def before_write(scope, *, write):
        nonlocal pause
        if write and pause:
            pause = False
            entered.set()
            await release.wait()
        async with original(scope, write=write) as tx:
            yield tx

    monkeypatch.setattr(store, "_transaction", before_write)
    task = asyncio.create_task(
        application.plan_collaboration_request(command, context=resolver.recipient.context)
    )
    await asyncio.wait_for(entered.wait(), timeout=10)
    try:
        await other.change_participant_lifecycle(
            change(
                initial,
                command.expected.intent.selection.recipient.reference,
                key="disable-before-planning",
                revision=1,
                state="disabled",
            ),
            context=CONTEXT,
        )
    finally:
        release.set()
    with pytest.raises(CollaborationAccessDenied):
        await task
    async with original(initial.binding.application_scope, write=False) as tx:
        assert await tx.scan_pending_request_plans(after=None, limit=1) == []
    assert provider.requests == []


async def test_public_retention_acknowledgement_loss_recovers_frozen_policy(stores, monkeypatch):
    store = stores()
    application, resolver, command, _, provider = await scenario(store, "decline")
    original = store._transaction
    lose = True

    @asynccontextmanager
    async def lose_first_commit(scope, *, write):
        nonlocal lose
        async with original(scope, write=write) as tx:
            yield tx
        if write and lose:
            lose = False
            raise RuntimeError("retention acknowledgement lost")

    monkeypatch.setattr(store, "_transaction", lose_first_commit)
    with pytest.raises(CollaborationUnavailable):
        await application.plan_collaboration_request(command, context=resolver.recipient.context)
    monkeypatch.setattr(store, "_transaction", original)
    reopened = app(
        stores(),
        application._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=60000),
    )
    await reopened.initialize_collaboration()
    found = await reopened.lookup_collaboration_plan(command, context=resolver.sender.context)
    assert isinstance(found, ExactMatch) and found.receipt.state == "evaluating"
    result = await reopened.reconcile_collaboration_plan(
        command, context=resolver.recipient.context
    )
    assert result.state == "declined"
    assert result.receipt == found.receipt.receipt
    assert provider.requests == []


async def test_public_reconciliation_cannot_create_or_change_the_retained_operation(stores):
    store = stores()
    application, resolver, command, _, provider = await scenario(store, "defer")
    before = await application.inspect_collaboration_request(
        command.expected, context=resolver.recipient.context
    )
    with pytest.raises(CollaborationUnavailable):
        await application.reconcile_collaboration_plan(command, context=resolver.recipient.context)
    initialized = await application.initialize_collaboration()
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        assert not await tx.scan_request_plans(command.expected.intent.selection.reference, limit=1)
    assert (
        await application.inspect_collaboration_request(
            command.expected, context=resolver.recipient.context
        )
        == before
    )
    planned = await complete_plan(application, command, resolver.recipient.context)
    changed = command.model_copy(update={"deadline_at_ms": command.deadline_at_ms - 1})
    with pytest.raises(CollaborationUnavailable):
        await application.reconcile_collaboration_plan(changed, context=resolver.recipient.context)
    assert (
        await application.reconcile_collaboration_plan(command, context=resolver.recipient.context)
        == planned
    )
    assert provider.requests == []


async def test_public_fixed_planning_key_compares_each_added_authority_field(stores):
    store = stores()
    application, resolver, command, _, provider = await scenario(store, "defer")
    record = await complete_plan(application, command, resolver.recipient.context)
    initialized = await application.initialize_collaboration()

    async def anchor():
        async with store._transaction(initialized.binding.application_scope, write=False) as tx:
            return await store._anchor(tx, initialized, REDACTOR)

    before = await anchor()
    changes = [
        {"expected_revision": command.expected_revision + 1},
        {"expected_input_revision": command.expected_input_revision + 1},
        {"expected_input_sha256": "f" * 64},
        {"admission_generation": command.admission_generation + 1},
        {
            "admission_operation": command.admission_operation.model_copy(
                update={"caller_key": "different-admission"}
            )
        },
        {"policy": command.policy.model_copy(update={"revision": command.policy.revision + 1})},
        {"policy_sha256": "f" * 64},
        {"deadline_at_ms": command.deadline_at_ms - 1},
        # Generation and predecessor form one structurally coupled tuple. Change
        # both to a valid successor shape while retaining the original stable key.
        {
            "planning_generation": 2,
            "predecessor": RequestPlanningPredecessor(
                operation=command.operation.model_copy(
                    update={"caller_key": "different-predecessor"}
                ),
                revision=3,
            ),
        },
    ]
    for field in ("principal", "invocation_id", "interaction_id"):
        changes.append({"initiator": command.initiator.model_copy(update={field: "different"})})
    changes.append(
        {
            "initiator": command.initiator.model_copy(
                update={"mandate": command.initiator.mandate.model_copy(update={"revision": 2})}
            )
        }
    )
    for field in (
        "max_generations",
        "max_stages",
        "max_resources",
        "max_record_bytes",
        "max_recovery_items",
    ):
        changes.append(
            {
                "limits": command.limits.model_copy(
                    update={field: getattr(command.limits, field) - 1}
                )
            }
        )
    for altered_fields in changes:
        different = command.model_copy(update=altered_fields)
        assert isinstance(
            await application.lookup_collaboration_plan(
                different, context=resolver.recipient.context
            ),
            ExactConflict,
        ), tuple(altered_fields)
        with pytest.raises(CollaborationConflict):
            await application.plan_collaboration_request(
                different, context=resolver.recipient.context
            )
        assert await anchor() == before
    found = await application.lookup_collaboration_plan(command, context=resolver.recipient.context)
    assert isinstance(found, ExactMatch) and found.receipt == record
    assert provider.requests == []


async def test_public_planning_rejects_mutated_values_before_diagnostics_or_mutation(
    stores, caplog, capsys
):
    class Hostile:
        def __repr__(self):
            raise AssertionError("planning-diagnostic-canary")

        def __str__(self):
            raise AssertionError("planning-diagnostic-canary")

    store = stores()
    application, resolver, command, _, provider = await scenario(store, "defer")
    initialized = await application.initialize_collaboration()
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        before = await store._anchor(tx, initialized, REDACTOR)
    invalid = (
        command.model_copy(update={"expected_revision": Hostile()}),
        command.model_copy(update={"expected_input_revision": True}),
        command.model_copy(
            update={"limits": command.limits.model_copy(update={"max_stages": Hostile()})}
        ),
        command.model_copy(
            update={
                "initiator": command.initiator.model_copy(
                    update={"principal": "planning-diagnostic-canary", "invocation_id": Hostile()}
                )
            }
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for value in invalid:
            for entrance in (
                application.plan_collaboration_request,
                application.lookup_collaboration_plan,
                application.reconcile_collaboration_plan,
            ):
                with pytest.raises(CollaborationContractError) as error:
                    await entrance(value, context=resolver.recipient.context)
                assert "planning-diagnostic-canary" not in str(error.value)
                assert "planning-diagnostic-canary" not in repr(error.value)
    assert not caught
    output = capsys.readouterr()
    assert "planning-diagnostic-canary" not in caplog.text + output.out + output.err
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        assert await store._anchor(tx, initialized, REDACTOR) == before
        assert await tx.scan_pending_request_plans(after=None, limit=1) == []
    assert provider.requests == []


async def test_public_competing_applications_commit_one_plan_and_native_admission(stores):
    store = stores()
    first, resolver, command, policy, provider = await scenario(store, "decline")
    second = app(
        stores(),
        first._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(
            mandates=resolver, max_ttl_ms=60000, planning_policies=(policy,)
        ),
    )
    initialized = await second.initialize_collaboration()
    results = await asyncio.gather(
        complete_plan(first, command, resolver.recipient.context),
        complete_plan(second, command, resolver.recipient.context),
    )
    assert results[0] == results[1] and results[0].state == "declined"
    parent = await first.inspect_collaboration_request(
        command.expected, context=resolver.recipient.context
    )
    assert parent is not None and parent.state == "declined" and parent.admission_generation == 1
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        assert (
            len(await tx.scan_request_plans(command.expected.intent.selection.reference, limit=2))
            == 1
        )
        events = [
            await tx.get("request_events", (sequence,)) for sequence in parent.event_sequences
        ]
    assert sum(event["type"] == "request_admission" for event in events) == 1
    assert provider.requests == []


async def test_public_plan_reserves_individual_cleanup_room_before_retaining_debt(
    stores, monkeypatch
):
    from cayu.collaboration import _planning_preflight
    from cayu.collaboration._contracts import CollaborationContractError
    from cayu.collaboration._preparation import contract_bytes

    store = stores()
    application, resolver, command, _, provider = await scenario(store, "defer")
    initialized = await application.initialize_collaboration()
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        before = await store._anchor(tx, initialized, REDACTOR)
    measured = []

    def interrupt_before_retention(record):
        measured.append(
            (
                len(contract_bytes(record, redactor=REDACTOR)),
                _planning_preflight.control_record_ceiling(record),
            )
        )
        raise OSError("pre-retention sizing interruption")

    with monkeypatch.context() as patch:
        patch.setattr(_planning_preflight, "preflight_plan_control", interrupt_before_retention)
        with pytest.raises(CollaborationUnavailable):
            await application.plan_collaboration_request(
                command, context=resolver.recipient.context
            )
    assert len(measured) == 1
    retained_size, cleanup_size = measured[0]
    limit = retained_size + 512
    assert limit < cleanup_size
    constrained = command.model_copy(
        update={"limits": command.limits.model_copy(update={"max_record_bytes": limit})}
    )
    with pytest.raises(CollaborationContractError) as refused:
        await application.plan_collaboration_request(
            constrained, context=resolver.recipient.context
        )
    assert "mandatory bounded cleanup" in str(refused.value.__cause__)
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        assert await store._anchor(tx, initialized, REDACTOR) == before
        assert not await tx.scan_pending_request_plans(after=None, limit=1)
    assert provider.requests == []
