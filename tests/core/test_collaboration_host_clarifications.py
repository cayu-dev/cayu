"""Real clarification service and native return under a one-slot host owner."""

import asyncio

import pytest

from cayu.collaboration._host import CollaborationHost, _HostRegistration
from cayu.collaboration._host_clarification_maintenance import (
    _ClarificationMaintenanceSource,
    acknowledge_clarification_maintenance,
    start_clarification_maintenance,
)
from cayu.collaboration._host_clarifications import _ClarificationRule
from cayu.collaboration._host_ownership import HostOwnership, HostOwnershipLimits


async def drive_pending_maintenance(app, *, context):
    host = CollaborationHost(
        app,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 4, 262144),
            producer_sources=(),
            producer_rules=(),
            clarification_maintenance_sources=(
                _ClarificationMaintenanceSource("deliveries", context),
            ),
            observation_timeout_s=30,
            shutdown_timeout_s=30,
        ),
    )
    async with host:
        async with asyncio.timeout(60):
            while True:
                observed = await host.service_once()
                assert not observed.source_failures
                for outcome in host._owned.inspect().completed:
                    if outcome.error is not None:
                        raise outcome.error
                if observed.serviced:
                    break
                await asyncio.sleep(0.01)
    assert not host.inspect().uncertain


async def drive_expiry(
    application,
    factory,
    initialized,
    command,
    registration,
    monkeypatch,
    *,
    cancellation,
    acknowledgement_loss=False,
    competing_expiry=False,
    **_,
):
    from tests.core.test_participant_identity import CONTEXT, app

    from cayu.collaboration.memory import InMemoryCollaborationStore

    store = factory()
    reopened = app(store, registration)
    try:
        assert await reopened.initialize_collaboration() == initialized
        before = await reopened.list_pending_clarification_deliveries(context=CONTEXT)
        assert len(before.items) == 1
        host = CollaborationHost(
            reopened,
            _HostRegistration(
                limits=HostOwnershipLimits(1, 1, 4, 262144),
                producer_sources=(),
                producer_rules=(),
                clarification_maintenance_sources=(
                    _ClarificationMaintenanceSource("questions", CONTEXT),
                ),
                observation_timeout_s=30,
                shutdown_timeout_s=0.01 if cancellation else 30,
            ),
        )
        if competing_expiry:
            from cayu.collaboration._clarification_question_recovery import (
                _QuestionExpirySuperseded,
            )

            coordinator = reopened._clarification_coordinator
            expire = coordinator.expire_question
            inspect = coordinator._inspect_maintenance_owned
            failures = []
            supersessions = []

            async def inspect_supersession(*args, **kwargs):
                result = await inspect(*args, **kwargs)
                if isinstance(result, _QuestionExpirySuperseded):
                    assert result.state == "expired" and len(result.decision_sha256) == 64
                    assert "Which API version?" not in repr(result)
                    assert "private-state" not in repr(result)
                    supersessions.append(result)
                return result

            async def lose_to_distinct_expiry(expected, **kwargs):
                await expire(
                    expected.model_copy(
                        update={"operation": initialized.operation("other-expiry")}
                    ),
                    **kwargs,
                )
                try:
                    return await expire(expected, **kwargs)
                except Exception as error:
                    failures.append(error)
                    raise

            monkeypatch.setattr(coordinator, "expire_question", lose_to_distinct_expiry)
            monkeypatch.setattr(coordinator, "_inspect_maintenance_owned", inspect_supersession)
            async with asyncio.timeout(60):
                while not host.inspect().failed:
                    await host.service_once()
            assert len(failures) == 1
            reported = []
            async with asyncio.timeout(30):
                while True:
                    try:
                        if not (await host.aclose()).pending:
                            break
                    except Exception as error:
                        reported.append(error)
                    await asyncio.sleep(0.01)
            assert reported == failures
            assert len(supersessions) == 1
        elif acknowledgement_loss:
            from tests.core.test_collaboration_host_role_reconciliation import recover_same_host

            coordinator = reopened._clarification_coordinator
            expire, inspect = coordinator.expire_question, coordinator._inspect_maintenance_owned
            primary = ConnectionError("question expiry acknowledgement lost")
            entered, release = asyncio.Event(), asyncio.Event()
            effects = reads = 0

            async def lost_ack(*args, **kwargs):
                nonlocal effects
                await expire(*args, **kwargs)
                effects += 1
                raise primary

            async def unavailable_then_blocked(*args, **kwargs):
                nonlocal reads
                reads += 1
                if reads == 1:
                    return None
                receipt = await inspect(*args, **kwargs)
                assert receipt is not None
                entered.set()
                await release.wait()
                return receipt

            monkeypatch.setattr(coordinator, "expire_question", lost_ack)
            monkeypatch.setattr(coordinator, "_inspect_maintenance_owned", unavailable_then_blocked)
            await recover_same_host(host, primary, entered, release)
            assert effects == 1 and reads == 2
        elif cancellation:
            from cayu.collaboration import _clarification_question_recovery as native

            original = native.expire_question
            entered, release = asyncio.Event(), asyncio.Event()

            async def commit_then_wait(*args, **kwargs):
                result = await original(*args, **kwargs)
                entered.set()
                await release.wait()
                return result

            monkeypatch.setattr(native, "expire_question", commit_then_wait)
            observer = asyncio.create_task(host.run())
            try:
                await asyncio.wait_for(entered.wait(), 45)
                observer.cancel()
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                assert observer.cancelled() and observer.cancelling() == 2
                assert (await host.aclose()).uncertain == 1
                assert not (await reopened.list_due_clarification_questions(context=CONTEXT)).items
            finally:
                release.set()
                async with asyncio.timeout(30):
                    while (await host.aclose()).pending:
                        for outcome in host._owned.inspect().completed:
                            if outcome.error is not None:
                                raise outcome.error
                        await asyncio.sleep(0.01)
                if not observer.done():
                    observer.cancel()
                    await asyncio.gather(observer, return_exceptions=True)
        else:
            async with host:
                async with asyncio.timeout(60):
                    while True:
                        observed = await host.service_once()
                        assert not observed.source_failures
                        for outcome in host._owned.inspect().completed:
                            if outcome.error is not None:
                                raise outcome.error
                        if observed.serviced:
                            break
                        await asyncio.sleep(0.05)
        assert not host.inspect().pending
        assert not (await reopened.list_due_clarification_questions(context=CONTEXT)).items
        assert await reopened.list_pending_clarification_deliveries(context=CONTEXT) == before
    finally:
        if not isinstance(store, InMemoryCollaborationStore):
            await store.close()


async def drive_service(
    app,
    request,
    *,
    context,
    delivery_context,
    recovery_context,
    timeout,
    verify_recovery_identity=False,
    maintenance_ack_loss=None,
    **_,
):
    host = CollaborationHost(
        app,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 4, 262144),
            producer_sources=(),
            producer_rules=(),
            clarification_rules=(_ClarificationRule(request, context, delivery_context),),
            observation_timeout_s=30,
            shutdown_timeout_s=30,
        ),
    )
    async with host:
        async with asyncio.timeout(timeout):
            while True:
                observed = await host.service_once()
                for outcome in host._owned.inspect().completed:
                    if outcome.error is not None:
                        raise outcome.error
                if observed.serviced:
                    inventory = await app.inspect_clarification_services(
                        request.ticket,
                        context=recovery_context,
                    )
                    recovery = next(
                        item.recovery
                        for item in inventory.items
                        if item.recovery.operation == request.operation
                    )
                    assert (recovery.session_id, recovery.session_instance_id) == (
                        request.ticket.session_id,
                        request.ticket.session_instance_id,
                    )
                    if verify_recovery_identity:
                        await reject_foreign_service_recovery(
                            app, request, recovery, inventory, context=recovery_context
                        )
                    if maintenance_ack_loss is not None:
                        return await recover_service_maintenance(
                            app,
                            inventory,
                            recovery,
                            context=recovery_context,
                            monkeypatch=maintenance_ack_loss,
                        )
                    maintenance = HostOwnership(HostOwnershipLimits(1, 1, 4, 262144))
                    start_clarification_maintenance(
                        app,
                        maintenance,
                        recovery,
                        context=recovery_context,
                        observation_deadline=asyncio.get_running_loop().time() + timeout,
                    )
                    while True:
                        settled = await maintenance.observe(0.1)
                        for outcome in settled.completed:
                            if outcome.error is not None:
                                raise outcome.error
                            assert acknowledge_clarification_maintenance(maintenance, outcome)
                            assert not maintenance.inspect().uncertain
                            receipt = outcome.value.receipt
                            target = request.delivery.append.append_key
                            assert (receipt.session_id, receipt.session_instance_id) == (
                                target.target_session_id,
                                target.target_session_instance_id,
                            )
                            assert receipt.question == request.delivery.question.operation
                            return outcome.value.receipt
                await asyncio.sleep(0.01)


async def reject_foreign_service_recovery(app, request, recovery, inventory, *, context):
    from cayu.collaboration._contracts import CollaborationConflict
    from cayu.collaboration.participants import CollaborationUnavailable

    for field in (
        "session_id",
        "session_instance_id",
        "selection_sha256",
        "dispatch_sha256",
    ):
        value = getattr(recovery, field)
        changed = ("0" if value[0] != "0" else "1") + value[1:]
        ownership = HostOwnership(HostOwnershipLimits(1, 1, 4, 262144))
        start_clarification_maintenance(
            app,
            ownership,
            recovery.model_copy(update={field: changed}),
            context=context,
            observation_deadline=asyncio.get_running_loop().time() + 60,
        )
        async with asyncio.timeout(60):
            while True:
                observed = await ownership.observe(0.1)
                if observed.completed:
                    (outcome,) = observed.completed
                    assert isinstance(
                        outcome.error, (CollaborationConflict, CollaborationUnavailable)
                    )
                    assert not acknowledge_clarification_maintenance(ownership, outcome)
                    assert ownership.pending == 1
                    break
        assert (
            await app.inspect_clarification_services(request.ticket, context=context) == inventory
        )
        with pytest.raises((CollaborationConflict, CollaborationUnavailable)):
            await app._clarification_coordinator._inspect_maintenance_owned(
                app, recovery.model_copy(update={field: changed}), context=context
            )


async def recover_service_maintenance(app, inventory, recovery, *, context, monkeypatch):
    from tests.core.test_collaboration_host_role_reconciliation import recover_same_host

    from cayu.collaboration._clarification_recovery_types import (
        ClarificationPendingService,
        ClarificationPendingServicePage,
    )

    coordinator = app._clarification_coordinator
    reconcile, inspect = coordinator.reconcile_service, coordinator._inspect_maintenance_owned
    # A source scan may finish after another worker has settled the item. The
    # native exact receiving record, not this stale hint, must authorize release.
    item = next(item for item in inventory.items if item.recovery == recovery)
    page = ClarificationPendingServicePage(
        items=(
            ClarificationPendingService(
                recovery=item.recovery,
                question=item.question,
                deadline_at_ms=item.deadline_at_ms,
            ),
        ),
        next_cursor=None,
    )
    receipts = []
    reads = 0
    primary = ConnectionError("service maintenance acknowledgement lost")
    entered, release = asyncio.Event(), asyncio.Event()

    async def stale_scan(**kwargs):
        return page

    async def lost_ack(*args, **kwargs):
        receipts.append(await reconcile(*args, **kwargs))
        raise primary

    async def unavailable_then_blocked(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 1:
            return None
        receipt = await inspect(*args, **kwargs)
        assert receipt == receipts[0]
        entered.set()
        await release.wait()
        return receipt

    with monkeypatch.context() as patch:
        patch.setattr(coordinator, "pending_services", stale_scan)
        patch.setattr(coordinator, "reconcile_service", lost_ack)
        patch.setattr(coordinator, "_inspect_maintenance_owned", unavailable_then_blocked)
        host = CollaborationHost(
            app,
            _HostRegistration(
                limits=HostOwnershipLimits(1, 1, 2, 262144),
                producer_sources=(),
                producer_rules=(),
                clarification_maintenance_sources=(
                    _ClarificationMaintenanceSource("services", context),
                ),
                observation_timeout_s=0.1,
            ),
        )
        await recover_same_host(host, primary, entered, release)
    assert reads == 2 and len(receipts) == 1
    return receipts[0]


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_host_side_session_maintenance_preserves_exact_waiting_identity(
    backend, tmp_path, request, monkeypatch
):
    from functools import partial

    from tests.core.test_clarification_public import test_public_question_uses_real_assistant_export

    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=True,
        one_slot=True,
        service_driver=partial(drive_service, verify_recovery_identity=True),
        maintenance_driver=drive_pending_maintenance,
        journey_ttl_ms=900_000,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("side_session", [False, True])
@pytest.mark.parametrize("successive", [False, True])
async def test_host_native_clarification_service(
    backend, tmp_path, request, monkeypatch, side_session, successive
):
    from tests.core.test_clarification_public import test_public_question_uses_real_assistant_export

    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=side_session,
        one_slot=True,
        multiple_questions=successive,
        finish_request=successive,
        service_driver=drive_service,
        maintenance_driver=drive_pending_maintenance,
        journey_ttl_ms=900_000,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("side_session", [False, True])
@pytest.mark.parametrize("timing", ["before", "during_return", "foreign_settlement"])
async def test_host_final_latch_arbitration(
    backend, tmp_path, request, monkeypatch, side_session, timing
):
    from tests.core.test_clarification_public import test_public_question_uses_real_assistant_export

    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=side_session,
        one_slot=True,
        finish_request=True,
        final_latch_timing=timing,
        service_driver=drive_service,
        maintenance_driver=drive_pending_maintenance,
        journey_ttl_ms=900_000,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("cancel_observer", [False, True])
async def test_host_discovers_due_question_after_reopen(
    backend, tmp_path, request, monkeypatch, cancel_observer
):
    from tests.core.test_clarification_public import test_public_question_uses_real_assistant_export

    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=False,
        public_reply=False,
        post_admission=False,
        side_session=False,
        question_recovery="cancel" if cancel_observer else True,
        question_driver=drive_expiry,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_host_question_expiry_recovers_late_readback(backend, tmp_path, request, monkeypatch):
    from functools import partial

    from tests.core.test_clarification_public import test_public_question_uses_real_assistant_export

    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=False,
        public_reply=False,
        post_admission=False,
        side_session=False,
        question_recovery=True,
        question_driver=partial(drive_expiry, acknowledgement_loss=True),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_host_question_expiry_reconciles_competing_terminal_decision(
    backend, tmp_path, request, monkeypatch
):
    from functools import partial

    from tests.core.test_clarification_public import test_public_question_uses_real_assistant_export

    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=False,
        public_reply=False,
        post_admission=False,
        side_session=False,
        question_recovery=True,
        question_driver=partial(drive_expiry, competing_expiry=True),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_host_side_service_maintenance_recovers_late_readback(
    backend, tmp_path, request, monkeypatch
):
    from functools import partial

    from tests.core.test_clarification_public import test_public_question_uses_real_assistant_export

    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=True,
        one_slot=True,
        journey_ttl_ms=900_000,
        service_driver=partial(drive_service, maintenance_ack_loss=monkeypatch),
        maintenance_driver=drive_pending_maintenance,
    )
