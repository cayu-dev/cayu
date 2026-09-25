"""Public durable FORK planning through real native creation and admission owners."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import uuid4

import pytest
from examples.collaboration.planning import fork_policy
from tests.core.test_context_selection_exclusion import reopened_store
from tests.core.test_participant_identity import CONTEXT, app, create
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_prepared_admission_public import prepared_scenario
from tests.core.test_request_planning_contracts import _policy
from tests.core.test_request_planning_public import complete_plan
from tests.core.test_request_planning_view_owner import source_scenario

from cayu.agents import AgentSpec
from cayu.collaboration._contracts import ExactMatch, ExactNotFound
from cayu.collaboration._request_coordinator import _initiator
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import (
    RequestPlanningControl,
    RequestPlanningRequest,
    planning_policy_commitment,
)
from cayu.collaboration.prepared_admission import prepared_budget
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import RunRequest
from cayu.sessions._planning_fork_owner import NativePlanningForkOwner
from cayu.sessions.context_views import RecipientSessionCreationRequest
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("ordering", ["full", "last_slot", "cancel_race", "reservation_lost_ack"])
async def test_public_fork_reserves_cleanup_before_permits(
    native_stores, tmp_path, monkeypatch, ordering
):
    import cayu.sessions._context_selection_fence as fence
    import cayu.storage._context_selection_fence as persistent_fence
    from cayu.collaboration._contracts import CollaborationConflict
    from cayu.collaboration._planning_records import RequestPlanningStageRecord
    from cayu.collaboration._request_store import operation_key
    from cayu.sessions._planning_view_owner import NativePlanningViewOwner

    application, resolver, command, provider, initialized, blueprint = await scenario(native_stores)
    native = native_stores[1]
    independent = (
        native if native_stores[3][0] == "memory" else reopened_store(native_stores, tmp_path)
    )
    monkeypatch.setattr(fence, "CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER", 1)
    monkeypatch.setattr(persistent_fence, "CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER", 1)
    competing = blueprint.view.request.model_copy(
        update={"selection_key": "competing-last-slot-" + uuid4().hex}
    )
    reserve = native._prepare_context_view_selection_target
    reached, release = asyncio.Event(), asyncio.Event()

    async def held(target, *, authority):
        if ordering == "reservation_lost_ack":
            await reserve(target, authority=authority)
            raise OSError("lost reservation acknowledgement")
        reached.set()
        await release.wait()
        return await reserve(target, authority=authority)

    if ordering == "full":
        await independent._reserve_context_view_selection_control(
            competing, authority=fence._CONTEXT_SELECTION_AUTHORITY
        )
    else:
        monkeypatch.setattr(native, "_prepare_context_view_selection_target", held)
    application._request_coordinator._owners.observation_timeout = 180
    observer = asyncio.create_task(
        application.plan_collaboration_request(command, context=resolver.recipient.context)
    )
    other = None
    try:
        if ordering in {"last_slot", "cancel_race"}:
            async with asyncio.timeout(180):
                await reached.wait()
            await independent._reserve_context_view_selection_control(
                competing, authority=fence._CONTEXT_SELECTION_AUTHORITY
            )
        if ordering == "cancel_race":
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert observer.cancelled() and observer.cancelling() == 1
        else:
            release.set()
            with pytest.raises(CollaborationUnavailable):
                await observer
            assert not application._request_coordinator._owners.pending

        # Reconstruct with independent persistent connections, not the original
        # receiver's handles or a process-local capacity promise.
        other = app(
            native_stores[2](),
            application._participant_coordinator._registration,
            session_store=independent,
            collaboration_requests=application._request_coordinator._registration,
        )
        await other.initialize_collaboration()
        retained = await other.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(retained, ExactMatch)
        assert retained.receipt.state == "preparing" and retained.receipt.pending_stages == 1
        control = RequestPlanningControl(
            expected=command,
            expected_revision=retained.receipt.revision,
            kind="cancelled",
            initiator=_initiator(resolver.recipient.context),
        )
        settled = await complete_plan(other, control, resolver.recipient.context, control=True)
        assert settled.state == "cancelled" and settled.pending_stages == 0
        assert settled.reserved_bytes == settled.reserved_events == 0
        release.set()
        await asyncio.gather(
            *tuple(application._request_coordinator._owners.pending), return_exceptions=True
        )
        assert (
            await complete_plan(other, command, resolver.recipient.context, reconcile=True)
            == settled
        )
        assert (
            await complete_plan(other, control, resolver.recipient.context, control=True) == settled
        )
        async with native_stores[0]._transaction(
            initialized.owner.application_scope, write=False
        ) as tx:
            rows = await tx.scan_request_plan_stages(command.operation, limit=2)
            assert len(rows) == 1
            stage = RequestPlanningStageRecord.model_validate(rows[0])
            assert stage.state == "excluded" and stage.receipt is None
            for permit in blueprint.view.permits:
                assert await tx.get("operations", operation_key(permit.operation)) is None
                assert await tx.get("permits", operation_key(permit.operation)) is None

        # Even if optional capacity later becomes available, the durable local
        # fence—not the quota or a missing native receipt—rejects late selection.
        monkeypatch.setattr(fence, "CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER", 2)
        monkeypatch.setattr(persistent_fence, "CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER", 2)
        monkeypatch.setattr(native, "_prepare_context_view_selection_target", reserve)
        with pytest.raises(CollaborationConflict):
            await NativePlanningViewOwner(other).select(blueprint.view, context=CONTEXT)
        assert (
            await independent.lookup_context_view_selection(blueprint.view.request.selection_key)
            is None
        )
        assert isinstance(
            await independent.read_session_creation_decision(blueprint.base.creation), ExactNotFound
        )
        assert (
            await complete_plan(other, command, resolver.recipient.context, reconcile=True)
            == settled
        )
        for participant in (
            blueprint.view.permit.intent.request.participant,
            blueprint.view.recipient,
        ):
            _, pending = await native_stores[0].scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=True,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert not any(
                item.effect_scope in {"context_view_selection", "context_view_retention"}
                for item in pending
            )
        if ordering == "full":
            from tests.core.test_request_planning_fresh import declined_successor

            successor_app, successor = await declined_successor(
                native_stores, other, command, revision=settled.revision
            )
            replacement = await complete_plan(successor_app, successor, resolver.recipient.context)
            assert replacement.state == "declined" and replacement.pending_stages == 0
        assert len(provider.requests) == 1
    finally:
        release.set()
        await asyncio.gather(observer, return_exceptions=True)
        await asyncio.gather(
            *tuple(application._request_coordinator._owners.pending), return_exceptions=True
        )
        if independent is not native:
            await independent.close()


async def test_expired_fork_recovery_excludes_unstarted_creation(
    native_stores, tmp_path, monkeypatch
):
    from cayu.sessions._planning_creation_owner import NativePlanningCreationOwner

    application, resolver, command, provider, _, blueprint = await scenario(native_stores)
    creation_target = None

    async def interrupted_before_creation(owner, expected, *, context):
        nonlocal creation_target
        creation_target = expected.preparation.creation
        raise OSError("interrupted before native creation")

    monkeypatch.setattr(NativePlanningCreationOwner, "create", interrupted_before_creation)
    application._request_coordinator._owners.observation_timeout = 180
    with pytest.raises(CollaborationUnavailable):
        await application.plan_collaboration_request(command, context=resolver.recipient.context)
    assert not application._request_coordinator._owners.pending
    retained = await application.lookup_collaboration_plan(
        command, context=resolver.recipient.context
    )
    assert isinstance(retained, ExactMatch) and retained.receipt.pending_stages == 2
    assert isinstance(
        await native_stores[1].read_session_creation_decision(blueprint.base.creation),
        ExactNotFound,
    )
    ledger = native_stores[2]()
    transaction = ledger._transaction

    @asynccontextmanager
    async def expired_owner_time(scope, *, write):
        async with transaction(scope, write=write) as tx:

            async def now_ms():
                return command.deadline_at_ms

            tx.now_ms = now_ms
            yield tx

    monkeypatch.setattr(ledger, "_transaction", expired_owner_time)
    native = native_stores[1]
    reopened = (
        native if native_stores[3][0] == "memory" else reopened_store(native_stores, tmp_path)
    )
    try:
        other = app(
            ledger,
            application._participant_coordinator._registration,
            session_store=reopened,
            collaboration_requests=application._request_coordinator._registration,
            budget_binding_receiver=application.budget_binding_receiver,
            enable_common_root_budget_binding=True,
        )
        await other.initialize_collaboration()
        result = await complete_plan(other, command, resolver.recipient.context, reconcile=True)
        assert result.state == "expired" and result.pending_stages == 0
        assert result.reserved_bytes == result.reserved_events == 0
        assert creation_target is not None
        decision = await reopened.read_session_creation_decision(creation_target)
        assert isinstance(decision, ExactMatch) and decision.receipt.state == "excluded"
        assert decision.receipt.settlement_acknowledged
        retention = await reopened._read_context_view_retention(blueprint.view)
        assert isinstance(retention, ExactMatch) and retention.receipt.state == "released"
        assert (
            await complete_plan(other, command, resolver.recipient.context, reconcile=True)
            == result
        )
        assert len(provider.requests) == 1  # Source publication only; no child dispatch.
    finally:
        if reopened is not native:
            await reopened.close()


@pytest.mark.parametrize("batch_size", [2, 32])
async def test_terminal_fork_plan_prunes_before_both_view_permits(native_stores, batch_size):
    from tests.core.test_collaboration_namespace import rotate

    from cayu.collaboration._contracts import ExactUnavailable
    from cayu.collaboration._planning_retention import creation_permit_planning_parent
    from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
    from cayu.collaboration.requests import RequestControl

    application, resolver, command, provider, initialized, blueprint = await scenario(native_stores)
    result = await complete_plan(application, command, resolver.recipient.context)
    assert result.state == "admitted"
    # Locate the parent independently from either permit, including the
    # destination retention permit whose key is not the view-stage key.
    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        for permit in reversed(blueprint.view.permits):
            assert await creation_permit_planning_parent(tx, permit, redactor=SecretRedactor()) == (
                command.expected
            )
    closed = await application.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("close-fork-request"),
            expected=command.expected,
            expected_revision=2,
            kind="cancel",
        ),
        context=resolver.sender.context,
    )
    assert closed.state == "cancelled"
    _, rotated = await rotate(native_stores[0], initialized)
    retirement = NamespaceRetire(
        operation=rotated.successor.reference.operation("retire-fork"),
        namespace=rotated.namespace.reference,
        expected_revision=rotated.namespace.revision,
        expected_retired_through=0,
    )
    await complete_maintenance(
        native_stores[0],
        lambda: application.retire_collaboration_namespace(retirement, context=CONTEXT),
    )
    for index in range(32):
        before = await application.inspect_collaboration_namespace(context=CONTEXT)
        pruning = NamespacePrune(
            operation=rotated.successor.reference.operation(f"prune-fork-{index}"),
            namespace=rotated.namespace.reference,
            expected_retention_revision=before.retention_revision,
            max_records=batch_size,
        )
        pruned = await complete_maintenance(
            native_stores[0],
            lambda pruning=pruning: application.prune_collaboration_namespace(
                pruning, context=CONTEXT
            ),
        )
        assert pruned.removed_records <= batch_size
        assert await application.prune_collaboration_namespace(pruning, context=CONTEXT) == pruned
        if pruned.complete:
            break
    else:
        pytest.fail("FORK view or creation permits stranded settled planning reclamation")
    assert isinstance(
        await application.lookup_collaboration_plan(command, context=resolver.recipient.context),
        ExactUnavailable,
    )
    assert len(provider.requests) == 1


async def complete_maintenance(store, operation):
    try:
        return await operation()
    except CollaborationUnavailable:
        pending = tuple(store._owners.pending)
        if not pending:
            raise
        done, remaining = await asyncio.wait(pending, timeout=180)
        assert not remaining, "Namespace maintenance remains owned after bounded observation."
        for task in done:
            task.result()
        return await operation()


async def test_lost_selection_ack_recovers_one_step_at_a_time_after_reconstruction(
    native_stores, tmp_path, monkeypatch
):
    application, resolver, command, provider, _, blueprint = await scenario(
        native_stores, max_recovery_items=1
    )
    prepare = NativePlanningForkOwner.prepare

    async def lost_ack(owner, expected, *, context):
        await prepare(owner, expected, context=context)
        raise OSError("selection acknowledgement lost")

    monkeypatch.setattr(NativePlanningForkOwner, "prepare", lost_ack)
    application._request_coordinator._owners.observation_timeout = 180
    with pytest.raises(CollaborationUnavailable):
        await application.plan_collaboration_request(command, context=resolver.recipient.context)
    assert not application._request_coordinator._owners.pending
    before = await application.lookup_collaboration_plan(
        command, context=resolver.recipient.context
    )
    assert isinstance(before, ExactMatch)
    assert before.receipt.stage_count == before.receipt.pending_stages == 1
    selected = await native_stores[1]._read_context_view_retention(blueprint.view)
    assert isinstance(selected, ExactMatch) and selected.receipt.state == "adopted"
    monkeypatch.setattr(NativePlanningForkOwner, "prepare", prepare)
    native = native_stores[1]
    reopened = native
    if native_stores[3][0] != "memory":
        await native.close()
        reopened = reopened_store(native_stores, tmp_path)
    try:
        other = app(
            native_stores[2](),
            application._participant_coordinator._registration,
            session_store=reopened,
            collaboration_requests=application._request_coordinator._registration,
            budget_binding_receiver=application.budget_binding_receiver,
            enable_common_root_budget_binding=True,
        )
        other.register_provider(provider, default=True)
        other.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
        await other.initialize_collaboration()
        for stage_count, pending in ((2, 2), (2, 1), (2, 0)):
            result = await complete_plan(other, command, resolver.recipient.context, reconcile=True)
            assert result.state == "preparing"
            assert (result.stage_count, result.pending_stages) == (stage_count, pending)
        result = await complete_plan(other, command, resolver.recipient.context, reconcile=True)
        assert result.state == "admitted" and result.stage_count == 3
        assert result.pending_stages == result.reserved_events == result.reserved_bytes == 0
        assert await complete_plan(other, command, resolver.recipient.context) == result
        after = await reopened._read_context_view_retention(blueprint.view)
        assert isinstance(after, ExactMatch) and after.receipt.state == "released"
        assert after.receipt.selection == selected.receipt.selection
        assert len(provider.requests) == 1
    finally:
        if reopened is not native:
            await reopened.close()


async def test_cancelled_observer_and_other_instance_cleanup_fence_late_fork_worker(
    native_stores, monkeypatch
):
    application, resolver, command, provider, _, blueprint = await scenario(native_stores)
    reached, release = asyncio.Event(), asyncio.Event()
    prepare = NativePlanningForkOwner.prepare

    async def paused(owner, expected, *, context):
        resolved = await prepare(owner, expected, context=context)
        reached.set()
        await release.wait()
        return resolved

    monkeypatch.setattr(NativePlanningForkOwner, "prepare", paused)
    application._request_coordinator._owners.observation_timeout = 180
    observer = asyncio.create_task(
        application.plan_collaboration_request(command, context=resolver.recipient.context)
    )
    try:
        async with asyncio.timeout(90):
            await reached.wait()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled() and observer.cancelling() == 1
        retained = await application.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(retained, ExactMatch)
        assert retained.receipt.stage_count == retained.receipt.pending_stages == 1
        other = app(
            native_stores[2](),
            application._participant_coordinator._registration,
            session_store=native_stores[1],
            collaboration_requests=application._request_coordinator._registration,
        )
        await other.initialize_collaboration()
        control = RequestPlanningControl(
            expected=command,
            expected_revision=retained.receipt.revision,
            kind="cancelled",
            initiator=_initiator(resolver.recipient.context),
        )
        terminal = await complete_plan(other, control, resolver.recipient.context, control=True)
        assert terminal.state == "cancelled" and terminal.pending_stages == 0
        pending = tuple(application._request_coordinator._owners.pending)
        release.set()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        final = await other.lookup_collaboration_plan(command, context=resolver.recipient.context)
        assert isinstance(final, ExactMatch) and final.receipt == terminal
        assert isinstance(
            await native_stores[1].read_session_creation_decision(blueprint.base.creation),
            ExactNotFound,
        )
        pin = await native_stores[1]._read_context_view_retention(blueprint.view)
        assert isinstance(pin, ExactMatch) and pin.receipt.state == "released"
        assert len(provider.requests) == 1
    finally:
        release.set()
        if not observer.done():
            observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
        await asyncio.gather(
            *tuple(application._request_coordinator._owners.pending), return_exceptions=True
        )


async def scenario(native_stores, *, max_recovery_items=32):
    original, resolver, admission, provider, _, initialized = await prepared_scenario(
        native_stores,
        request_ttl_ms=300_000,
        provider_events=[
            [
                ModelStreamEvent.text_delta("historical answer"),
                ModelStreamEvent.completed(
                    {"finish_reason": "stop", "usage": {"input_tokens": 1, "output_tokens": 1}}
                ),
            ]
        ],
    )
    _, created = await create(original, initialized, key="fork-planning-source")
    source = created.participants[0].reference
    _, _, _, selection, _ = await source_scenario(
        native_stores, configured=(original, initialized, source), provider=provider
    )
    creation = RecipientSessionCreationRequest(
        request=RunRequest(
            agent_name="reviewer", messages=[Message.text("user", "new child input")]
        ),
        creation_key="planned-fork-" + uuid4().hex,
        recipient=admission.prepared.recipient,
    )
    blueprint = await original.prepare_recipient_fork(
        creation,
        selection,
        source_participant=source,
        context=CONTEXT,
        deadline_at_ms=admission.expected.intent.selection.expires_at_ms,
    )
    policy = fork_policy(
        _policy().reference.model_copy(update={"owner": initialized.owner}),
        _policy().limits.model_copy(update={"max_recovery_items": max_recovery_items}),
        blueprint,
    )

    class BudgetReceiver:
        async def resolve_budget_binding(self, *, request):
            return prepared_budget(blueprint.base.budget_binding_json)

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
        operation=initialized.operation("fork-plan"),
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
    return application, resolver, command, provider, initialized, blueprint


async def test_public_fork_plan_creates_exact_inert_child_and_settles_retention(native_stores):
    application, resolver, command, provider, initialized, blueprint = await scenario(native_stores)
    result = await complete_plan(application, command, resolver.recipient.context)
    assert result.state == "admitted" and result.stage_count == 3 and result.pending_stages == 0
    assert result.reserved_bytes == result.reserved_events == 0
    assert await complete_plan(application, command, resolver.recipient.context) == result
    assert (
        await complete_plan(application, command, resolver.recipient.context, reconcile=True)
        == result
    )
    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        stages = await tx.scan_request_plan_stages(command.operation, limit=4)
    from cayu.collaboration._planning_records import RequestPlanningStageRecord

    view, creation, admission = [RequestPlanningStageRecord.model_validate(row) for row in stages]
    from cayu.collaboration._preparation import contract_bytes

    assert all(
        len(contract_bytes(stage, redactor=SecretRedactor())) <= command.limits.max_record_bytes
        for stage in (view, creation, admission)
    )
    assert all(stage.state == "settled" for stage in (view, creation, admission))
    assert view.receipt.state == "released"
    assert creation.receipt.decision.state == "created"
    assert admission.receipt.command.prepared.target.kind == "fork"
    session = await native_stores[1].load(creation.receipt.decision.session_id)
    assert session.status == "pending" and session.run_epoch == 0
    transcript = await native_stores[1].load_transcript(session.id)
    text = "\n".join(message.model_dump_json() for message in transcript)
    assert "historical answer" in text and "new child input" in text
    retention = await native_stores[1]._read_context_view_retention(blueprint.view)
    assert isinstance(retention, ExactMatch) and retention.receipt.state == "released"
    for participant in (blueprint.view.permit.intent.request.participant, blueprint.view.recipient):
        _, pending = await native_stores[0].scan_obligations(
            initialized,
            participant,
            after=0,
            limit=64,
            pending_only=True,
            retention_revision=None,
            redactor=SecretRedactor(),
        )
        assert not any(
            item.effect_scope
            in {"context_view_selection", "context_view_retention", "recipient_session_creation"}
            for item in pending
        )
    assert len(provider.requests) == 1
