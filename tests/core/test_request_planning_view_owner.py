"""Real source turns and native selection handoff; planner composition is separate."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tests.core.test_context_selection_exclusion import reopened_store
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.agents import AgentSpec
from cayu.collaboration._contracts import CollaborationConflict, ExactMatch
from cayu.collaboration.lifecycle import ParticipantLifecycleChange
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import RunRequest
from cayu.sessions._context_selection_fence import ContextViewSelectionConflict
from cayu.sessions._model_completion_publication import model_step_publication_from_checkpoint
from cayu.sessions._planning_view_owner import NativePlanningViewOwner
from cayu.sessions.context_views import (
    ContextViewLimits,
    ContextViewOwnershipRequest,
    ContextViewPublicationRequest,
    ContextViewSelectionRequest,
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)

pytestmark = pytest.mark.anyio


async def test_exclusion_fences_registration_paused_after_native_reservation(
    native_stores, tmp_path, monkeypatch
):
    from cayu.vaults.redaction import SecretRedactor

    value, initialized, source, request, _ = await source_scenario(native_stores)
    _, created = await create(value, initialized, key="delayed-selection-recipient")
    recipient = created.participants[0].reference
    owner = NativePlanningViewOwner(value)
    target = await owner.prepare(
        request,
        participant=source,
        recipient=recipient,
        context=CONTEXT,
        deadline_at_ms=2**53 - 1,
    )
    native = native_stores[1]

    async def assert_no_obligations():
        for participant in (source, recipient):
            _, pending = await native_stores[0].scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=True,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert pending == ()

    original = native._prepare_context_view_selection_target
    entered, release = asyncio.Event(), asyncio.Event()

    async def paused(*args, **kwargs):
        decision = await original(*args, **kwargs)
        entered.set()
        await release.wait()
        return decision

    monkeypatch.setattr(native, "_prepare_context_view_selection_target", paused)
    caller = asyncio.create_task(owner.select(target, context=CONTEXT))
    reopened = (
        native if native_stores[3][0] == "memory" else reopened_store(native_stores, tmp_path)
    )
    try:
        async with asyncio.timeout(30):
            await entered.wait()
        application = app(
            native_stores[2](),
            value._participant_coordinator._registration,
            session_store=reopened,
        )
        await application.initialize_collaboration()
        receiver = NativePlanningViewOwner(application)
        excluded = await receiver.exclude(target, context=CONTEXT)
        assert excluded.state == "excluded"
        release.set()
        with pytest.raises(CollaborationConflict):
            await caller
        await assert_no_obligations()
        monkeypatch.setattr(native, "_prepare_context_view_selection_target", original)
        if reopened is not native:
            await reopened.close()
            reopened = reopened_store(native_stores, tmp_path)
        application = app(
            native_stores[2](),
            value._participant_coordinator._registration,
            session_store=reopened,
        )
        await application.initialize_collaboration()
        receiver = NativePlanningViewOwner(application)
        assert await receiver.exclude(target, context=CONTEXT) == excluded
        with pytest.raises(CollaborationConflict):
            await receiver.select(target, context=CONTEXT)
        await assert_no_obligations()
        assert await receiver.exclude(target, context=CONTEXT) == excluded
        await assert_no_obligations()
        assert await reopened.lookup_context_view_selection(request.selection_key) is None
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        if reopened is not native:
            await reopened.close()


@pytest.mark.parametrize("phase", ["unselected", "selected", "adopted"])
async def test_discharge_reconstructs_exact_pin_cleanup_after_lost_ack(
    native_stores, tmp_path, monkeypatch, phase
):
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.vaults.redaction import SecretRedactor

    value, initialized, source, request, provider = await source_scenario(native_stores)
    _, created = await create(value, initialized, key="cleanup-recipient")
    recipient = created.participants[0].reference
    receiver = NativePlanningViewOwner(value)
    target = await receiver.prepare(
        request,
        participant=source,
        recipient=recipient,
        context=CONTEXT,
        deadline_at_ms=2**53 - 1,
    )
    if phase != "unselected":
        await receiver.select(target, context=CONTEXT)
    if phase == "adopted":
        await receiver.adopt(target, context=CONTEXT)
    collaboration, native = native_stores[:2]
    method = "_exclude_permit" if phase == "unselected" else "_settle_permit"
    original = getattr(collaboration, method)

    async def lost_ack(*args, **kwargs):
        await original(*args, **kwargs)
        raise OSError("native cleanup source acknowledgement lost")

    monkeypatch.setattr(collaboration, method, lost_ack)
    with pytest.raises(CollaborationUnavailable):
        await receiver.discharge(target, context=CONTEXT)
    monkeypatch.setattr(collaboration, method, original)
    reopened = native
    if native_stores[3][0] != "memory":
        await native.close()
        reopened = reopened_store(native_stores, tmp_path)
    try:
        application = app(
            native_stores[2](), value._participant_coordinator._registration, session_store=reopened
        )
        await application.initialize_collaboration()
        receiver = NativePlanningViewOwner(application)
        result = await receiver.discharge(target, context=CONTEXT)
        assert isinstance(result, ExactMatch)
        assert result.receipt.state == ("excluded" if phase == "unselected" else "released")
        assert await receiver.discharge(target, context=CONTEXT) == result
        for participant in (source, recipient):
            _, pending = await collaboration.scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=True,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert pending == ()
        if phase == "unselected":
            with pytest.raises(CollaborationConflict):
                await receiver.select(target, context=CONTEXT)
        else:
            events = await reopened.read_context_view_lifecycle_events(
                result.receipt.selection.view_id
            )
            assert sum(event.state == "released" for event in events) == 1
        assert len(provider.requests) == 1
    finally:
        if reopened is not native:
            await reopened.close()


@pytest.mark.parametrize("committed_before_deadline", [False, True])
async def test_adoption_deadline_fences_new_effect_but_preserves_exact_replay(
    native_stores, monkeypatch, committed_before_deadline
):
    value, _, participant, request, _ = await source_scenario(native_stores)
    native = native_stores[1]
    clock_name = "_clock" if native_stores[3][0] == "postgres" else "_ownership_clock"
    now = datetime.now(UTC)
    monkeypatch.setattr(native, clock_name, lambda: now)
    receiver = NativePlanningViewOwner(value)
    target = await receiver.prepare(
        request,
        participant=participant,
        context=CONTEXT,
        deadline_at_ms=int(now.timestamp() * 1000) + 60_000,
    )
    await receiver.select(target, context=CONTEXT)
    before = await native._read_context_view_retention(target)
    if committed_before_deadline:
        before = await receiver.adopt(target, context=CONTEXT)
    now += timedelta(seconds=60)
    if committed_before_deadline:
        assert await receiver.adopt(target, context=CONTEXT) == before
    else:
        with pytest.raises(PermissionError, match="expired"):
            await receiver.adopt(target, context=CONTEXT)
        assert await native._read_context_view_retention(target) == before
        assert (
            await native.read_context_view_lifecycle_events(before.receipt.selection.view_id) == ()
        )


async def source_scenario(native_stores, *, configured=None, provider=None):
    collaboration, sessions, _, _ = native_stores
    if configured is None:
        reg = registration()
        value = app(collaboration, reg, session_store=sessions)
        initialized = await value.initialize_collaboration()
        _, receipt = await create(value, initialized)
        participant = receipt.participants[0].reference
    else:
        value, initialized, participant = configured
    if provider is None:
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("historical answer"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        value.register_provider(provider, default=True)
    if configured is None:
        value.register_agent(AgentSpec(name="reviewer", model="model"))
    identity = uuid4().hex
    creation = ParticipantSessionCreationRequest(
        RunRequest(agent_name="reviewer", messages=[Message.text("user", "source turn")]), identity
    )
    session, _ = await value.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    events = [
        event
        async for event in value.execute_participant_session(
            ParticipantSessionExecutionRequest(
                request=creation.request.model_copy(update={"session_id": session.id}),
                session_instance_id=session.instance_id,
                execution_key=identity,
            ),
            participant=participant,
            context=CONTEXT,
        )
    ]
    pointer = model_step_publication_from_checkpoint(await sessions.load_checkpoint(session.id))
    assert pointer is not None, "\n".join(
        event.model_dump_json()
        for event in events
        if "fail" in event.type.value or "error" in event.type.value
    )
    assert any(event.type.value == "session.completed" for event in events)
    completion = next(
        event
        for event in await sessions.load_events(session.id)
        if event.id == pointer.completion_event_id
    )
    manifest = await value.publish_completed_context_view(
        ContextViewPublicationRequest(
            source_session_id=session.id,
            source_session_instance_id=session.instance_id,
            view_id=identity,
            interaction_id=completion.interaction_id,
            boundary_id=pointer.logical_step_id,
            projection_schema="whole-turn.v1",
            publication_key=identity,
        ),
        participant=participant,
        context=CONTEXT,
    )
    request = ContextViewSelectionRequest(
        source_owner=participant.owner,
        source_session_id=session.id,
        source_session_instance_id=session.instance_id,
        selector="latest",
        projection_schema=manifest.projection_schema,
        extension_set_commitment=manifest.extension_set_commitment,
        limits=ContextViewLimits(),
        selection_key=identity,
    )
    return value, initialized, participant, request, provider


@pytest.mark.parametrize("registered_first", [False, True])
async def test_selection_permit_orders_two_instance_disablement(
    native_stores, monkeypatch, registered_first
):
    value, initialized, participant, request, provider = await source_scenario(native_stores)
    other = app(
        native_stores[2](),
        value._participant_coordinator._registration,
        session_store=native_stores[1],
    )
    await other.initialize_collaboration()
    owner = NativePlanningViewOwner(value)
    entered, release = asyncio.Event(), asyncio.Event()
    native = native_stores[1]
    before = (
        await native.load(request.source_session_id),
        await native.load_events(request.source_session_id),
    )
    if registered_first:
        original = native._register_context_view_selection_target

        async def held(target, *, authority):
            entered.set()
            await release.wait()
            return await original(target, authority=authority)

        monkeypatch.setattr(native, "_register_context_view_selection_target", held)
    else:
        original = value._participant_coordinator.inspect

        async def held(*args, **kwargs):
            result = await original(*args, **kwargs)
            entered.set()
            await release.wait()
            return result

        monkeypatch.setattr(value._participant_coordinator, "inspect", held)
    targets = []

    async def select():
        target = await owner.prepare(
            request, participant=participant, context=CONTEXT, deadline_at_ms=2**53 - 1
        )
        targets.append(target)
        return await owner.select(target, context=CONTEXT)

    task = asyncio.create_task(select())
    try:
        async with asyncio.timeout(20):
            await entered.wait()
        await other.change_participant_lifecycle(
            ParticipantLifecycleChange(
                operation=initialized.operation("disable-during-selection"),
                participant=participant,
                expected_lifecycle_revision=1,
                state="disabled",
            ),
            context=CONTEXT,
        )
        release.set()
        if registered_first:
            found = await task
            assert isinstance(found, ExactMatch)
            assert found.receipt.state == "selected" and found.receipt.responsibility_registered
            assert found.receipt.target == targets[0]
            assert await owner.select(targets[0], context=CONTEXT) == found
        else:
            with pytest.raises(CollaborationConflict):
                await task
            assert await native.lookup_context_view_selection(request.selection_key) is None
            excluded = await owner.exclude(targets[0], context=CONTEXT)
            assert excluded.state == "excluded"
            assert await owner.exclude(targets[0], context=CONTEXT) == excluded
        assert (
            await native.load(request.source_session_id),
            await native.load_events(request.source_session_id),
        ) == before
        assert len(provider.requests) == 1
        # Identical raw ordinary selection data cannot borrow the registered
        # internal responsibility or bypass its negative receiving decision.
        with pytest.raises(ContextViewSelectionConflict):
            await native.select_context_view(request)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("phase", ["registration", "selection"])
@pytest.mark.parametrize("signal", ["cancel", "lost_ack"])
async def test_selection_handoff_reconstructs_after_interruption(
    native_stores, tmp_path, monkeypatch, phase, signal
):
    value, initialized, participant, request, provider = await source_scenario(native_stores)
    owner = NativePlanningViewOwner(value)
    target = await owner.prepare(
        request, participant=participant, context=CONTEXT, deadline_at_ms=2**53 - 1
    )
    native = native_stores[1]
    method = (
        "_register_context_view_selection_target"
        if phase == "registration"
        else "_select_context_view_target"
    )
    original = getattr(native, method)
    entered, release = asyncio.Event(), asyncio.Event()

    async def interrupted(*args, **kwargs):
        result = await original(*args, **kwargs)
        entered.set()
        if signal == "lost_ack":
            raise OSError("selection handoff acknowledgement lost")
        await release.wait()
        return result

    monkeypatch.setattr(native, method, interrupted)
    caller = asyncio.create_task(owner.select(target, context=CONTEXT))
    try:
        async with asyncio.timeout(20):
            await entered.wait()
        if signal == "cancel":
            caller.cancel()
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert caller.cancelled() and caller.cancelling() == 2
        else:
            with pytest.raises(OSError, match="acknowledgement lost"):
                await caller
        found = await owner.read(target)
        assert isinstance(found, ExactMatch)
        assert found.receipt.responsibility_registered
        assert found.receipt.state == ("reserved" if phase == "registration" else "selected")
        # Register-before-disable remains valid for this exact operation even
        # when observer interruption and persistent reconstruction intervene.
        await value.change_participant_lifecycle(
            ParticipantLifecycleChange(
                operation=initialized.operation("disable-after-interruption"),
                participant=participant,
                expected_lifecycle_revision=1,
                state="disabled",
            ),
            context=CONTEXT,
        )
        monkeypatch.setattr(native, method, original)
        reopened = native
        if native_stores[3][0] != "memory":
            await native.close()
            reopened = reopened_store(native_stores, tmp_path)
        try:
            application = app(
                native_stores[2](),
                value._participant_coordinator._registration,
                session_store=reopened,
            )
            await application.initialize_collaboration()
            receiver = NativePlanningViewOwner(application)
            assert await receiver.read(target) == found
            selected = await receiver.select(target, context=CONTEXT)
            assert isinstance(selected, ExactMatch) and selected.receipt.state == "selected"
            assert await receiver.select(target, context=CONTEXT) == selected
            assert selected.receipt.target == target
            assert len(provider.requests) == 1
            # Same selection identity cannot be reused with changed authority.
            changed = target.model_copy(
                update={
                    "permit": target.permit.model_copy(
                        update={
                            "initiator": target.permit.initiator.model_copy(
                                update={"principal": "other"}
                            )
                        }
                    )
                }
            )
            from cayu.collaboration._contracts import ExactConflict

            assert isinstance(await receiver.read(changed), ExactConflict)
        finally:
            if reopened is not native:
                await reopened.close()
    finally:
        release.set()
        if not caller.done():
            caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)


@pytest.mark.parametrize("lost_ack", [False, True])
async def test_selected_pin_settles_only_after_exact_release_even_when_disabled(
    native_stores, tmp_path, monkeypatch, lost_ack
):
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.sessions._planning_view_owner import _ViewSettlementReader
    from cayu.vaults.redaction import SecretRedactor

    value, initialized, participant, request, _ = await source_scenario(native_stores)
    receiver = NativePlanningViewOwner(value)
    target = await receiver.prepare(
        request, participant=participant, context=CONTEXT, deadline_at_ms=2**53 - 1
    )
    found = await receiver.select(target, context=CONTEXT)
    assert isinstance(found, ExactMatch)
    selection = found.receipt
    native, collaboration = native_stores[1], native_stores[0]
    reader = _ViewSettlementReader(native, target)
    assert not isinstance(await reader.lookup(target.permit), ExactMatch)
    retention = await native._read_context_view_retention(target)
    assert isinstance(retention, ExactMatch) and retention.receipt.state == "selected"
    _, pending = await collaboration.scan_obligations(
        initialized,
        participant,
        after=0,
        limit=64,
        pending_only=True,
        retention_revision=None,
        redactor=SecretRedactor(),
    )
    assert len(pending) == 1
    release = ContextViewOwnershipRequest(
        selection_key=request.selection_key,
        view_id=selection.view_id,
        pin_commitment=selection.pin_commitment,
        expected_state="selected",
        expected_revision=retention.receipt.revision,
        operation="release",
        current_owner=participant.owner,
        current_participant=participant,
        operation_key="release-" + request.selection_key,
    )
    await value.change_participant_lifecycle(
        ParticipantLifecycleChange(
            operation=initialized.operation("disable-before-release"),
            participant=participant,
            expected_lifecycle_revision=1,
            state="disabled",
        ),
        context=CONTEXT,
    )
    original = collaboration._settle_permit

    async def commit_then_raise(*args, **kwargs):
        await original(*args, **kwargs)
        raise OSError("release settlement acknowledgement lost")

    if lost_ack:
        monkeypatch.setattr(collaboration, "_settle_permit", commit_then_raise)
        with pytest.raises(CollaborationUnavailable):
            await receiver.release(target, release, context=CONTEXT)
        monkeypatch.setattr(collaboration, "_settle_permit", original)
    else:
        result = await receiver.release(target, release, context=CONTEXT)
        assert isinstance(result, ExactMatch) and result.receipt.state == "released"
    reopened = native
    if native_stores[3][0] != "memory":
        await native.close()
        reopened = reopened_store(native_stores, tmp_path)
    try:
        application = app(
            native_stores[2](), value._participant_coordinator._registration, session_store=reopened
        )
        await application.initialize_collaboration()
        receiver = NativePlanningViewOwner(application)
        result = await receiver.release(target, release, context=CONTEXT)
        assert isinstance(result, ExactMatch) and result.receipt.state == "released"
        assert await receiver.release(target, release, context=CONTEXT) == result
        _, pending = await collaboration.scan_obligations(
            initialized,
            participant,
            after=0,
            limit=64,
            pending_only=True,
            retention_revision=None,
            redactor=SecretRedactor(),
        )
        assert pending == ()
        events = await reopened.read_context_view_lifecycle_events(selection.view_id, limit=64)
        assert len(events) == 1 and events[0].state == "released"
        with pytest.raises(ValueError):
            await receiver.release(
                target, release.model_copy(update={"expected_revision": 2}), context=CONTEXT
            )
        assert await receiver.read(target) == found
    finally:
        if reopened is not native:
            await reopened.close()


@pytest.mark.parametrize("disable", ["before_registration", "after_registration", "none"])
async def test_exact_cross_participant_adoption_and_partial_settlement(
    native_stores, monkeypatch, disable
):
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.sessions._context_selection_fence import selection_adoption_request
    from cayu.vaults.redaction import SecretRedactor

    value, initialized, source, request, provider = await source_scenario(native_stores)
    _, created = await create(value, initialized, key="recipient")
    recipient = created.participants[0].reference
    other = app(
        native_stores[2](),
        value._participant_coordinator._registration,
        session_store=native_stores[1],
    )
    await other.initialize_collaboration()
    receiver = NativePlanningViewOwner(value)
    target = await receiver.prepare(
        request, participant=source, recipient=recipient, context=CONTEXT, deadline_at_ms=2**53 - 1
    )
    assert len(target.permits) == 2 and target.recipient == recipient

    async def deactivate():
        await other.change_participant_lifecycle(
            ParticipantLifecycleChange(
                operation=initialized.operation("disable-recipient"),
                participant=recipient,
                expected_lifecycle_revision=1,
                state="disabled",
            ),
            context=CONTEXT,
        )

    native, collaboration = native_stores[1], native_stores[0]
    if disable == "before_registration":
        await deactivate()
        with pytest.raises(CollaborationConflict):
            await receiver.select(target, context=CONTEXT)
        assert await native.lookup_context_view_selection(request.selection_key) is None
        # Source registration may have committed before destination refusal.
        # Both exact responsibilities are fenced and settled, without new access.
        assert (await receiver.exclude(target, context=CONTEXT)).state == "excluded"
    else:
        selected = await receiver.select(target, context=CONTEXT)
        assert isinstance(selected, ExactMatch)
        adoption = selection_adoption_request(target, selected.receipt)
        # Equal data at the ordinary entrance cannot consume the private handoff.
        with pytest.raises(PermissionError):
            await value.transition_context_view_ownership(
                adoption, participant=source, destination_participant=recipient, context=CONTEXT
            )
        if disable == "after_registration":
            await deactivate()
        adopted = await receiver.adopt(target, context=CONTEXT)
        assert isinstance(adopted, ExactMatch)
        assert adopted.receipt.state == "adopted" and adopted.receipt.participant == recipient
        assert await receiver.adopt(target, context=CONTEXT) == adopted
        release = ContextViewOwnershipRequest(
            selection_key=request.selection_key,
            view_id=selected.receipt.view_id,
            pin_commitment=selected.receipt.pin_commitment,
            expected_state="adopted",
            expected_revision=adopted.receipt.revision,
            operation="release",
            current_owner=recipient.owner,
            current_participant=recipient,
            operation_key="release-" + request.selection_key,
        )
        original = collaboration._settle_permit
        once = False

        async def partial_ack(*args, **kwargs):
            nonlocal once
            result = await original(*args, **kwargs)
            if not once:
                once = True
                raise OSError("first of two permit settlements acknowledgement lost")
            return result

        monkeypatch.setattr(collaboration, "_settle_permit", partial_ack)
        with pytest.raises(CollaborationUnavailable):
            await receiver.release(target, release, context=CONTEXT)
        assert (
            await receiver.release(target, release, context=CONTEXT)
        ).receipt.state == "released"
    for participant in (source, recipient):
        _, pending = await collaboration.scan_obligations(
            initialized,
            participant,
            after=0,
            limit=64,
            pending_only=True,
            retention_revision=None,
            redactor=SecretRedactor(),
        )
        assert pending == ()
    assert len(provider.requests) == 1
