"""Public FRESH staging retains the existing native creation responsibility."""

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import replace

import pytest
from examples.collaboration.planning import decline_policy, fresh_policy
from tests.core.test_participant_identity import CONTEXT, app
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_prepared_admission_public import prepared_scenario
from tests.core.test_request_planning_contracts import _policy
from tests.core.test_request_planning_public import complete_plan

from cayu import Message, RunRequest
from cayu.agents import AgentSpec
from cayu.collaboration._contracts import CollaborationConflict, ExactMatch, ExactNotFound
from cayu.collaboration.planning import (
    RequestPlanningControl,
    RequestPlanningPredecessor,
    RequestPlanningRequest,
    planning_policy_commitment,
)
from cayu.collaboration.prepared_admission import prepared_budget
from cayu.sessions.context_views import RecipientSessionCreationRequest
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio


async def test_narrow_request_stage_limit_rejects_before_decision_and_remains_cleanable(
    native_stores,
):
    from cayu.collaboration._contracts import CollaborationContractError
    from cayu.collaboration.participants import CollaborationUnavailable

    application, resolver, command, provider, initialized, _, preparation = await scenario(
        native_stores
    )
    command = command.model_copy(
        update={"limits": command.limits.model_copy(update={"max_stages": 1})}
    )
    with pytest.raises((CollaborationUnavailable, CollaborationContractError)):
        await application.plan_collaboration_request(command, context=resolver.recipient.context)
    pending = tuple(application._request_coordinator._owners.pending)
    if pending:
        async with asyncio.timeout(90):
            await asyncio.gather(*pending, return_exceptions=True)
    found = await application.lookup_collaboration_plan(command, context=resolver.recipient.context)
    assert isinstance(found, ExactMatch)
    assert found.receipt.state == "evaluating"
    assert found.receipt.decision is None
    assert found.receipt.stage_count == found.receipt.pending_stages == 0
    assert (
        await native_stores[0]._lookup_registered_permit(
            initialized, preparation.creation.permit.operation, redactor=SecretRedactor()
        )
        is None
    )
    assert isinstance(
        await native_stores[1].read_session_creation_decision(preparation.creation), ExactNotFound
    )
    cancelled = await application.control_collaboration_plan(
        RequestPlanningControl(
            expected=command,
            expected_revision=found.receipt.revision,
            kind="cancelled",
            initiator=command.initiator,
        ),
        context=resolver.recipient.context,
    )
    assert cancelled.state == "cancelled"
    assert cancelled.pending_stages == cancelled.reserved_bytes == cancelled.reserved_events == 0
    assert provider.requests == []


async def reopen_planner(native_stores, application, preparation):
    class BudgetReceiver:
        async def resolve_budget_binding(self, *, request):
            return prepared_budget(preparation.budget_binding_json)

    reopened = app(
        native_stores[2](),
        application._participant_coordinator._registration,
        session_store=native_stores[1],
        collaboration_requests=replace(
            application._request_coordinator._registration, receiving_owner=None
        ),
        budget_binding_receiver=BudgetReceiver(),
        enable_common_root_budget_binding=True,
    )
    await reopened.initialize_collaboration()
    return reopened


async def declined_successor(native_stores, application, command, *, revision):
    policy = decline_policy(
        command.policy.model_copy(update={"object_id": "after-settlement"}), command.limits
    )
    successor_app = app(
        native_stores[2](),
        application._participant_coordinator._registration,
        collaboration_requests=replace(
            application._request_coordinator._registration,
            receiving_owner=None,
            planning_policies=(policy,),
        ),
    )
    initialized = await successor_app.initialize_collaboration()
    successor = command.model_copy(
        update={
            "operation": initialized.operation("settled-successor"),
            "admission_operation": initialized.operation("settled-successor-admission"),
            "planning_generation": 2,
            "policy": policy.reference,
            "policy_sha256": planning_policy_commitment(policy, redactor=SecretRedactor()),
            "predecessor": RequestPlanningPredecessor(
                operation=command.operation, revision=revision
            ),
        }
    )
    return successor_app, successor


async def scenario(native_stores, *, max_recovery_items=32):
    # This witness qualifies the complete multi-owner journey, not expiry. Keep
    # its real deadline finite and within the registration's existing ceiling.
    original, resolver, admission, provider, _, initialized = await prepared_scenario(
        native_stores, request_ttl_ms=300_000
    )
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "input")]),
        creation_key="planned-child:" + initialized.owner.application_scope,
        recipient=admission.prepared.recipient,
    )
    preparation = await original.prepare_recipient_creation(creation, context=CONTEXT)
    policy = fresh_policy(
        _policy().reference.model_copy(update={"owner": initialized.owner}),
        _policy().limits.model_copy(update={"max_recovery_items": max_recovery_items}),
        preparation,
    )

    class BudgetReceiver:
        async def resolve_budget_binding(self, *, request):
            return prepared_budget(preparation.budget_binding_json)

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
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    await application.initialize_collaboration()
    command = RequestPlanningRequest(
        operation=initialized.operation("fresh-plan"),
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
    return application, resolver, command, provider, initialized, creation, preparation


async def test_public_fresh_plan_creates_one_inert_child_and_admits_exactly(native_stores):
    application, resolver, command, provider, _, creation, _ = await scenario(native_stores)
    result = await complete_plan(application, command, resolver.recipient.context)
    assert result.state == "admitted" and result.stage_count == 2 and result.pending_stages == 0
    assert result.reserved_bytes == result.reserved_events == 0
    child = await application.lookup_recipient_session(creation, context=CONTEXT)
    assert child is not None and child[0].status == "pending" and child[0].run_epoch == 0
    assert await complete_plan(application, command, resolver.recipient.context) == result
    assert (
        await complete_plan(application, command, resolver.recipient.context, reconcile=True)
        == result
    )
    assert await application.lookup_recipient_session(creation, context=CONTEXT) == child
    assert provider.requests == []

    # Individual terminal rows, including the future admission envelope, must
    # fit before creation is permitted. Characterize the exact inclusive bound
    # using the real public journey's retained intent.
    from cayu.collaboration._planning_preflight import preflight_stage_terminal
    from cayu.collaboration._planning_records import RequestPlanningStageRecord

    initialized = await application.initialize_collaboration()
    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        stages = await tx.scan_request_plan_stages(command.operation, limit=3)
    assert len(stages) == 2
    stage = RequestPlanningStageRecord.model_validate(stages[0]).model_copy(
        update={
            "state": "pending",
            "receipt": None,
            "settled_at_ms": None,
            "settlement_event": None,
        }
    )

    def size(ceiling):
        return preflight_stage_terminal(
            stage,
            initialized,
            ceiling=ceiling,
            redactor=SecretRedactor(),
            expected_plan=command,
        )

    required = size(command.limits.max_record_bytes)
    with pytest.raises(ValueError, match="terminal evidence"):
        size(required - 1)
    assert size(required) == size(required + 1) == required


async def test_fresh_root_closure_prunes_planning_before_creation_permit(native_stores):
    from tests.core.test_collaboration_namespace import rotate
    from tests.core.test_request_planning_fork import complete_maintenance

    from cayu.collaboration._contracts import ExactUnavailable
    from cayu.collaboration._planning_records import RequestPlanningEvent, RequestPlanningRecord
    from cayu.collaboration._request_store import operation_key
    from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.collaboration.requests import RequestControl

    application, resolver, command, provider, initialized, creation, preparation = await scenario(
        native_stores
    )
    admitted = await complete_plan(application, command, resolver.recipient.context)
    assert admitted.state == "admitted"
    child = await application.lookup_recipient_session(creation, context=CONTEXT)
    closed = await application.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("close-fresh-request"),
            expected=command.expected,
            expected_revision=2,
            kind="cancel",
        ),
        context=resolver.sender.context,
    )
    assert closed.state == "cancelled"
    _, rotated = await rotate(native_stores[0], initialized)
    retirement = NamespaceRetire(
        operation=rotated.successor.reference.operation("retire-fresh"),
        namespace=rotated.namespace.reference,
        expected_revision=rotated.namespace.revision,
        expected_retired_through=0,
    )
    await complete_maintenance(
        native_stores[0],
        lambda: application.retire_collaboration_namespace(retirement, context=CONTEXT),
    )
    reconstructed = False
    for index in range(24):
        before = await application.inspect_collaboration_namespace(context=CONTEXT)
        pruning = NamespacePrune(
            operation=rotated.successor.reference.operation(f"prune-fresh-{index}"),
            namespace=rotated.namespace.reference,
            expected_retention_revision=before.retention_revision,
            max_records=2,
        )
        result = await complete_maintenance(
            native_stores[0],
            lambda pruning=pruning, application=application: (
                application.prune_collaboration_namespace(pruning, context=CONTEXT)
            ),
        )
        assert result.removed_records <= 2
        assert await application.prune_collaboration_namespace(pruning, context=CONTEXT) == result
        async with native_stores[0]._transaction(
            initialized.owner.application_scope, write=False
        ) as tx:
            raw = await tx.get("request_plans", operation_key(command.operation))
        if raw is not None and RequestPlanningRecord.model_validate(raw).pruned_stages:
            assert isinstance(
                await application.lookup_collaboration_plan(
                    command, context=resolver.recipient.context
                ),
                ExactUnavailable,
            )
            if not reconstructed:
                # The initial batch's successful validation is not permission
                # to delete divergent evidence found in a later transaction.
                record = RequestPlanningRecord.model_validate(raw)
                event_key = (record.event_sequences[-1],)
                scope = initialized.owner.application_scope
                async with native_stores[0]._transaction(scope, write=True) as tx:
                    event = RequestPlanningEvent.model_validate(
                        await tx.get("request_plan_events", event_key)
                    )
                    remaining_stages = await tx.scan_request_plan_stages(
                        command.operation, limit=command.limits.max_stages + 1
                    )
                    await tx.put(
                        "request_plan_events",
                        event_key,
                        event.model_copy(
                            update={
                                "operation": event.operation.model_copy(
                                    update={"caller_key": "another-plan"}
                                )
                            }
                        ),
                        insert=False,
                    )
                before_rejection = await application.inspect_collaboration_namespace(
                    context=CONTEXT
                )
                rejected = NamespacePrune(
                    operation=rotated.successor.reference.operation("reject-divergent-event"),
                    namespace=rotated.namespace.reference,
                    expected_retention_revision=before_rejection.retention_revision,
                    max_records=2,
                )
                with pytest.raises(CollaborationUnavailable):
                    await complete_maintenance(
                        native_stores[0],
                        lambda application=application, rejected=rejected: (
                            application.prune_collaboration_namespace(rejected, context=CONTEXT)
                        ),
                    )
                assert (
                    await application.inspect_collaboration_namespace(context=CONTEXT)
                    == before_rejection
                )
                async with native_stores[0]._transaction(scope, write=True) as tx:
                    assert await tx.get("request_plans", operation_key(command.operation)) == raw
                    assert (
                        await tx.scan_request_plan_stages(
                            command.operation, limit=command.limits.max_stages + 1
                        )
                        == remaining_stages
                    )
                    await tx.put("request_plan_events", event_key, event, insert=False)
                application = await reopen_planner(native_stores, application, preparation)
                assert (
                    await application.prune_collaboration_namespace(pruning, context=CONTEXT)
                    == result
                )
                reconstructed = True
        if result.complete:
            break
    else:
        pytest.fail("Native creation permit or planning stages stranded request reclamation")
    assert reconstructed, "The complete plan must not fit in this pruning batch."
    assert isinstance(
        await application.lookup_collaboration_plan(command, context=resolver.recipient.context),
        ExactUnavailable,
    )
    assert child is not None and await native_stores[1].load(child[0].id) == child[0]
    assert provider.requests == []


@pytest.mark.parametrize("phase", ["created", "adopted", "admitted"])
async def test_fresh_lost_ack_reconciles_exact_child_in_reconstructed_application(
    native_stores, monkeypatch, phase
):
    from cayu.collaboration.participants import CollaborationUnavailable

    application, resolver, command, provider, _, creation, preparation = await scenario(
        native_stores, max_recovery_items=1 if phase == "created" else 32
    )
    lost = False
    create_calls = 0
    real_create = application.create_recipient_session

    async def create_then_lose_ack(*args, **kwargs):
        nonlocal lost, create_calls
        result = await real_create(*args, **kwargs)
        create_calls += 1
        if phase == "created" and not lost:
            lost = True
            raise RuntimeError("native creation committed; acknowledgement lost")
        return result

    monkeypatch.setattr(application, "create_recipient_session", create_then_lose_ack)
    store = native_stores[0]
    transaction = store._transaction

    @asynccontextmanager
    async def lose_local_ack(scope, *, write):
        nonlocal lost
        committed = False
        async with transaction(scope, write=write) as tx:
            put = tx.put

            async def note_put(family, key, value, *, insert):
                nonlocal committed
                await put(family, key, value, insert=insert)
                if phase == "adopted" and family == "request_plan_stages":
                    committed |= value.intent.ordinal == 1 and value.state == "settled"
                if phase == "admitted" and family == "request_plans":
                    committed |= value.state == "admitted"

            if write:
                tx.put = note_put
            yield tx
        if committed and not lost:
            lost = True
            raise RuntimeError("planning transaction committed; acknowledgement lost")

    monkeypatch.setattr(store, "_transaction", lose_local_ack)
    try:
        with suppress(CollaborationUnavailable):
            await application.plan_collaboration_request(
                command, context=resolver.recipient.context
            )
        async with asyncio.timeout(180):
            await asyncio.gather(
                *tuple(application._request_coordinator._owners.pending), return_exceptions=True
            )
        assert lost and create_calls == 1
    finally:
        monkeypatch.setattr(store, "_transaction", transaction)
    child = await application.lookup_recipient_session(creation, context=CONTEXT)
    assert child is not None and child[0].status == "pending"
    reopened = await reopen_planner(native_stores, application, preparation)

    async def forbid_new_creation(*args, **kwargs):
        pytest.fail("reconciliation must adopt the committed child, not create again")

    monkeypatch.setattr(reopened, "create_recipient_session", forbid_new_creation)
    result = await complete_plan(reopened, command, resolver.recipient.context, reconcile=True)
    if phase == "created":
        assert result.state == "preparing" and result.stage_count == 1
        assert result.pending_stages == 0
        result = await complete_plan(reopened, command, resolver.recipient.context, reconcile=True)
    assert result.state == "admitted" and result.pending_stages == 0
    assert result.stage_count == 2 and result.reserved_bytes == result.reserved_events == 0
    assert await complete_plan(reopened, command, resolver.recipient.context) == result
    assert await application.lookup_recipient_session(creation, context=CONTEXT) == child
    assert provider.requests == []


async def test_fresh_cancelled_observer_retains_creation_for_competing_reconciliation(
    native_stores, monkeypatch
):
    application, resolver, command, provider, _, creation, preparation = await scenario(
        native_stores
    )
    entered, release = asyncio.Event(), asyncio.Event()
    real_create = application.create_recipient_session
    created = None

    async def hold_committed_creation(*args, **kwargs):
        nonlocal created
        created = await real_create(*args, **kwargs)
        entered.set()
        await release.wait()
        return created

    monkeypatch.setattr(application, "create_recipient_session", hold_committed_creation)
    application._request_coordinator._owners.observation_timeout = 180
    caller = asyncio.create_task(
        application.plan_collaboration_request(command, context=resolver.recipient.context)
    )
    try:
        await asyncio.wait_for(entered.wait(), 120)
        caller.cancel()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert caller.cancelled() and caller.cancelling() == 2
        assert application._request_coordinator._owners.pending
        before = await application.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(before, ExactMatch)
        assert before.receipt.state == "preparing" and before.receipt.pending_stages == 1

        reopened = await reopen_planner(native_stores, application, preparation)

        async def forbid_creation(*args, **kwargs):
            pytest.fail("another observer must reconcile the same native creation")

        monkeypatch.setattr(reopened, "create_recipient_session", forbid_creation)
        admitted = await complete_plan(
            reopened, command, resolver.recipient.context, reconcile=True
        )
        assert admitted.state == "admitted" and admitted.pending_stages == 0
        assert application._request_coordinator._owners.pending
        pending = tuple(application._request_coordinator._owners.pending)
        release.set()
        async with asyncio.timeout(180):
            await asyncio.gather(*pending)
        assert not application._request_coordinator._owners.pending
        final = await reopened.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(final, ExactMatch) and final.receipt == admitted
        assert await application.lookup_recipient_session(creation, context=CONTEXT) == created
        assert provider.requests == []
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        await asyncio.gather(
            *tuple(application._request_coordinator._owners.pending), return_exceptions=True
        )


async def test_fresh_created_child_is_retained_while_cancelled_plan_is_replaced(
    native_stores, monkeypatch
):
    application, resolver, command, provider, _, creation, _ = await scenario(native_stores)
    # Keep the test observer attached while another operation deliberately holds
    # creation. The generic fixture's timeout recovery joins all pending owners,
    # including that held operation; this race needs independent control readback.
    application._request_coordinator._owners.observation_timeout = 300
    entered, release = asyncio.Event(), asyncio.Event()
    real_create = application.create_recipient_session
    created = None

    async def hold_created(*args, **kwargs):
        nonlocal created
        created = await real_create(*args, **kwargs)
        entered.set()
        await release.wait()
        return created

    monkeypatch.setattr(application, "create_recipient_session", hold_created)
    caller = asyncio.create_task(complete_plan(application, command, resolver.recipient.context))
    try:
        await asyncio.wait_for(entered.wait(), 120)
        found = await application.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(found, ExactMatch) and found.receipt.pending_stages == 1
        cancelled = await complete_plan(
            application,
            RequestPlanningControl(
                expected=command,
                expected_revision=found.receipt.revision,
                kind="cancelled",
                initiator=command.initiator,
            ),
            resolver.recipient.context,
            control=True,
        )
        assert cancelled.state == "cancelled" and cancelled.pending_stages == 0
        successor_app, successor = await declined_successor(
            native_stores, application, command, revision=cancelled.revision
        )
        replacement = await complete_plan(successor_app, successor, resolver.recipient.context)
        assert replacement.state == "declined"
        release.set()
        assert await caller == cancelled
        original = await application.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(original, ExactMatch) and original.receipt == cancelled
        assert await application.lookup_recipient_session(creation, context=CONTEXT) == created
        assert created is not None and created[0].status == "pending"
        assert provider.requests == []
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        await asyncio.gather(
            *tuple(application._request_coordinator._owners.pending), return_exceptions=True
        )


@pytest.mark.parametrize("created_first", [False, True])
@pytest.mark.parametrize("recovery_only", [False, True])
async def test_real_fresh_deadline_settles_exact_native_responsibility(
    native_stores, monkeypatch, created_first, recovery_only
):
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.sessions.creation_fence import SessionCreationExcluded

    application, resolver, command, provider, initialized, creation, preparation = await scenario(
        native_stores
    )
    store = native_stores[0]
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        now = await tx.now_ms()
    command = command.model_copy(update={"deadline_at_ms": now + 45_000})
    other = await reopen_planner(native_stores, application, preparation)
    application._request_coordinator._owners.observation_timeout = 180
    other._request_coordinator._owners.observation_timeout = 180
    entered, release = asyncio.Event(), asyncio.Event()
    created = None
    if created_first:
        native_create = application.create_recipient_session

        async def hold_created(*args, **kwargs):
            nonlocal created
            created = await native_create(*args, **kwargs)
            entered.set()
            await release.wait()
            return created

        monkeypatch.setattr(application, "create_recipient_session", hold_created)
    else:
        native_prepare = native_stores[1]._prepare_session_creation_target

        async def hold_prepare(*args, **kwargs):
            entered.set()
            await release.wait()
            return await native_prepare(*args, **kwargs)

        monkeypatch.setattr(native_stores[1], "_prepare_session_creation_target", hold_prepare)
    caller = asyncio.create_task(
        application.plan_collaboration_request(command, context=resolver.recipient.context)
    )
    barrier = asyncio.create_task(entered.wait())
    try:
        done, _ = await asyncio.wait(
            (caller, barrier), timeout=90, return_when=asyncio.FIRST_COMPLETED
        )
        if caller in done:
            await caller
            pytest.fail("Planning completed before the held creation boundary")
        assert barrier in done
        found = await other.lookup_collaboration_plan(command, context=resolver.recipient.context)
        assert isinstance(found, ExactMatch) and found.receipt.pending_stages == 1
        control = RequestPlanningControl(
            expected=command,
            expected_revision=found.receipt.revision,
            kind="expired",
            initiator=command.initiator,
        )
        # Expiry is delivered by real owner time, not by throwing TimeoutError or
        # replacing the transaction clock. The foreign worker stays held.
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            remaining = command.deadline_at_ms - await tx.now_ms()
        if remaining > 0:
            await asyncio.sleep(remaining / 1000 + 0.01)
        if recovery_only:

            async def reject_new_creation(*args, **kwargs):
                pytest.fail("Expired recovery must not dispatch new recipient creation")

            monkeypatch.setattr(other, "create_recipient_session", reject_new_creation)
            unchanged = await other.lookup_collaboration_plan(
                command, context=resolver.recipient.context
            )
            assert unchanged == found
        recovery_command = command if recovery_only else control
        settled = await complete_plan(
            other,
            recovery_command,
            resolver.recipient.context,
            reconcile=recovery_only,
            control=not recovery_only,
        )
        assert settled.state == "expired" and settled.pending_stages == 0
        assert settled.reserved_bytes == settled.reserved_events == 0
        decision = await native_stores[1].read_session_creation_decision(preparation.creation)
        assert isinstance(decision, ExactMatch)
        assert decision.receipt.state == ("created" if created_first else "excluded")
        assert decision.receipt.settlement_acknowledged
        release.set()
        if created_first:
            assert await caller == settled
        else:
            with pytest.raises((CollaborationUnavailable, SessionCreationExcluded)):
                await caller
        assert (
            await complete_plan(
                other,
                recovery_command,
                resolver.recipient.context,
                reconcile=recovery_only,
                control=not recovery_only,
            )
            == settled
        )
        assert await application.lookup_recipient_session(creation, context=CONTEXT) == created
        assert provider.requests == []
    finally:
        release.set()
        barrier.cancel()
        await asyncio.gather(caller, barrier, return_exceptions=True)
        await asyncio.gather(
            *tuple(application._request_coordinator._owners.pending), return_exceptions=True
        )


@pytest.mark.parametrize("permit_first", [False, True])
async def test_fresh_disablement_orders_against_permit_and_final_admission(
    native_stores, monkeypatch, permit_first
):
    from tests.core.test_participant_lifecycle import change

    from cayu.collaboration.access import CollaborationAccessDenied

    application, resolver, command, provider, initialized, creation, preparation = await scenario(
        native_stores
    )
    other = await reopen_planner(native_stores, application, preparation)
    entered, release = asyncio.Event(), asyncio.Event()
    real_prepare = native_stores[1]._prepare_session_creation_target

    async def hold_receiving_prepare(target, *, authority):
        entered.set()
        await release.wait()
        return await real_prepare(target, authority=authority)

    monkeypatch.setattr(
        native_stores[1], "_prepare_session_creation_target", hold_receiving_prepare
    )
    application._request_coordinator._owners.observation_timeout = 180
    caller = None
    try:
        if permit_first:
            caller = asyncio.create_task(
                complete_plan(application, command, resolver.recipient.context)
            )
            await asyncio.wait_for(entered.wait(), 120)
            assert (
                await native_stores[0]._lookup_registered_permit(
                    initialized, preparation.creation.permit.operation, redactor=SecretRedactor()
                )
                is not None
            )
        selected = command.expected.intent.selection.recipient
        snapshot = await other.inspect_participant(selected.reference, context=CONTEXT)
        await other.change_participant_lifecycle(
            change(
                initialized,
                selected.reference,
                key="disable-planned-recipient",
                revision=snapshot.participant.lifecycle_revision,
                state="disabled",
            ),
            context=CONTEXT,
        )
        release.set()
        if caller is None:
            caller = asyncio.create_task(
                complete_plan(application, command, resolver.recipient.context)
            )
        with pytest.raises(CollaborationAccessDenied):
            await caller
        found = await other.lookup_collaboration_plan(command, context=resolver.recipient.context)
        if permit_first:
            # Prior native permission remains valid for its exact creation, but
            # does not renew authority for the distinct request admission.
            child = await application.lookup_recipient_session(creation, context=CONTEXT)
            assert child is not None and child[0].status == "pending"
            assert isinstance(found, ExactMatch)
            assert found.receipt.state == "preparing"
            assert found.receipt.stage_count == 1 and found.receipt.pending_stages == 0
            cancelled = await complete_plan(
                other,
                RequestPlanningControl(
                    expected=command,
                    expected_revision=found.receipt.revision,
                    kind="cancelled",
                    initiator=command.initiator,
                ),
                resolver.recipient.context,
                control=True,
            )
            assert cancelled.pending_stages == 0
            assert cancelled.reserved_bytes == cancelled.reserved_events == 0
            assert await application.lookup_recipient_session(creation, context=CONTEXT) == child
        else:
            assert not entered.is_set()
            assert (
                await native_stores[0]._lookup_registered_permit(
                    initialized, preparation.creation.permit.operation, redactor=SecretRedactor()
                )
                is None
            )
            assert isinstance(
                await native_stores[1].read_session_creation_decision(preparation.creation),
                ExactNotFound,
            )
        # No final admission command exists without a created child; assert
        # durable absence at the exact already-chosen admission operation.
        async with native_stores[0]._transaction(
            initialized.binding.application_scope, write=False
        ) as tx:
            from cayu.collaboration._request_store import operation_key

            assert await tx.get("operations", operation_key(command.admission_operation)) is None
        assert provider.requests == []
    finally:
        release.set()
        if caller is not None:
            await asyncio.gather(caller, return_exceptions=True)


@pytest.mark.parametrize("fail_stage_fit", [False, True])
async def test_fresh_stage_and_existing_permit_are_retained_before_receiving_write(
    native_stores, monkeypatch, fail_stage_fit
):
    application, resolver, command, provider, initialized, creation, preparation = await scenario(
        native_stores
    )
    if fail_stage_fit:
        from cayu.collaboration import _permit_store, _planning_preflight
        from cayu.collaboration._contracts import CollaborationContractError
        from cayu.collaboration._planning_creation_types import RequestCreationStageCommand
        from cayu.collaboration.participants import CollaborationUnavailable

        reached = False
        permit_written = False
        original_preflight = _planning_preflight.preflight_stage_terminal
        original_registration = _permit_store.register_permit_in_transaction

        async def record_real_registration(*args, **kwargs):
            nonlocal permit_written
            receipt = await original_registration(*args, **kwargs)
            permit_written = True
            return receipt

        def reject_terminal_size(stage, *args, **kwargs):
            nonlocal reached
            if isinstance(stage.intent.command, RequestCreationStageCommand):
                assert permit_written
                reached = True
                # Exercise the real envelope validator after permit insertion;
                # rejection must roll the shared transaction back completely.
                return original_preflight(stage, *args, **{**kwargs, "ceiling": 1})
            return original_preflight(stage, *args, **kwargs)

        monkeypatch.setattr(_planning_preflight, "preflight_stage_terminal", reject_terminal_size)
        monkeypatch.setattr(
            _permit_store, "register_permit_in_transaction", record_real_registration
        )
        with pytest.raises((CollaborationUnavailable, CollaborationContractError)):
            await complete_plan(application, command, resolver.recipient.context)
        assert reached
        assert (
            await native_stores[0]._lookup_registered_permit(
                initialized, preparation.creation.permit.operation, redactor=SecretRedactor()
            )
            is None
        )
        found = await application.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(found, ExactMatch)
        assert found.receipt.stage_count == found.receipt.pending_stages == 0
        assert isinstance(
            await native_stores[1].read_session_creation_decision(preparation.creation),
            ExactNotFound,
        )
        assert provider.requests == []
        return
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.sessions.creation_fence import SessionCreationExcluded

    entered, release = asyncio.Event(), asyncio.Event()
    original_prepare = native_stores[1]._prepare_session_creation_target

    async def hold_receiving_prepare(target, *, authority):
        entered.set()
        await release.wait()
        return await original_prepare(target, authority=authority)

    monkeypatch.setattr(
        native_stores[1], "_prepare_session_creation_target", hold_receiving_prepare
    )
    application._request_coordinator._owners.observation_timeout = 180
    caller = asyncio.create_task(
        application.plan_collaboration_request(command, context=resolver.recipient.context)
    )
    try:
        await asyncio.wait_for(entered.wait(), 90)
        found = await application.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(found, ExactMatch)
        staged = found.receipt
        assert staged.state == "preparing" and staged.pending_stages == staged.stage_count == 1
        assert isinstance(
            await native_stores[1].read_session_creation_decision(preparation.creation),
            ExactNotFound,
        )
        permit = await native_stores[0]._lookup_registered_permit(
            initialized, preparation.creation.permit.operation, redactor=SecretRedactor()
        )
        assert permit is not None and permit.expected == preparation.creation.permit
        successor_app, successor = await declined_successor(
            native_stores, application, command, revision=staged.revision
        )
        with pytest.raises(CollaborationConflict):
            await complete_plan(successor_app, successor, resolver.recipient.context)
        cancelled = await complete_plan(
            application,
            RequestPlanningControl(
                expected=command,
                expected_revision=staged.revision,
                kind="cancelled",
                initiator=command.initiator,
            ),
            resolver.recipient.context,
            control=True,
        )
        assert cancelled.state == "cancelled" and cancelled.pending_stages == 0
        excluded = await native_stores[1].read_session_creation_decision(preparation.creation)
        assert isinstance(excluded, ExactMatch) and excluded.receipt.state == "excluded"
        assert excluded.receipt.settlement_acknowledged
        successor = successor.model_copy(
            update={
                "predecessor": RequestPlanningPredecessor(
                    operation=command.operation, revision=cancelled.revision
                )
            }
        )
        replacement = await complete_plan(successor_app, successor, resolver.recipient.context)
        assert replacement.state == "declined"
        assert (
            await complete_plan(
                successor_app, cancelled.control, resolver.recipient.context, control=True
            )
            == cancelled
        )
        release.set()
        with pytest.raises((CollaborationUnavailable, SessionCreationExcluded)):
            await caller
        found = await application.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(found, ExactMatch) and found.receipt == cancelled
        assert (
            await native_stores[1].lookup_participant_session_creation(creation.participant_request)
            is None
        )
        assert provider.requests == []
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        await asyncio.gather(
            *tuple(application._request_coordinator._owners.pending), return_exceptions=True
        )
