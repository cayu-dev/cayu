"""Receiving exclusion retires retained exports without renewing disclosure."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import app as make_app
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu import ProducerDeliveryRecovery
from cayu.collaboration._producer_export_cleanup import ProducerExportCleanupStatus
from cayu.collaboration._session_export_store import digest
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.events import EventType
from cayu.sessions.base import EventQuery


async def process_readback(backend, address, command, *, failure_resolution=None):
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.recovery.producer_export_retirement_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            child.communicate(
                json.dumps(
                    {
                        "backend": backend,
                        "address": address,
                        "expected": command.model_dump(mode="json"),
                        "failure_resolution": None
                        if failure_resolution is None
                        else failure_resolution.model_dump(mode="json"),
                    }
                ).encode()
            ),
            90,
        )
        assert child.returncode == 0, stderr.decode()
        if failure_resolution is not None:
            from cayu.collaboration.requests import RequestOutcomeReceipt

            return RequestOutcomeReceipt.model_validate_json(stdout)
        return ProducerExportCleanupStatus.model_validate_json(stdout)
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "rejection",
    [
        "validator_rejected",
        "projection_too_large",
        "projection_wrapped_envelope_overflow",
        "projection_envelope_overflow",
        "source_visible_overflow",
        "source_private_overflow",
    ],
)
async def test_output_validation_failure_elects_failed_and_cleans_up(
    native_stores, monkeypatch, rejection
):
    values, context = await completed_export_scenario(
        native_stores,
        monkeypatch,
        visible_text="x" * (40 * 1024)
        if rejection == "source_visible_overflow"
        else "retained answer",
        private_text="private-canary" * 6000 if rejection == "source_private_overflow" else None,
        output_bytes=65536,
    )
    app, resolver, _, provider, _, _, command, _ = values
    destination = command.destinations[0]
    projector = app._session_export_coordinator.projectors[destination.projector]
    if rejection == "validator_rejected":
        monkeypatch.setattr(projector, "validate", lambda *args: False)
    elif not rejection.startswith("source_"):
        size = {
            "projection_too_large": 8192,
            "projection_wrapped_envelope_overflow": 65520,
            "projection_envelope_overflow": 65536,
        }[rejection]
        monkeypatch.setattr(projector, "project", lambda *args: {"text": "x" * size})
    with pytest.raises(CollaborationUnavailable):
        await app.export_producer_output(command, destination.operation, context=context)

    from cayu.collaboration._contracts import ExactUnavailable
    from cayu.collaboration._producer_export_store import read_export

    store, initialized = app._participant_coordinator._ready()
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        retained = await read_export(tx, command, destination, redactor=app._secret_redactor)
    assert retained is not None
    assert isinstance(
        await app.lookup_session_export(retained.request, context=context), ExactUnavailable
    )

    def no_projection(*args):
        raise AssertionError("Retained rejection must not reproject")

    monkeypatch.setattr(projector, "project", no_projection)
    with pytest.raises(CollaborationUnavailable):
        await app.export_producer_output(command, destination.operation, context=context)
    backend, address = native_stores[3]
    elected = (
        await app.publish_producer_outcome(command, context=context)
        if backend == "memory"
        else await process_readback(
            backend, address, command, failure_resolution=resolver.recipient.resolution
        )
    )
    assert elected.command.outcome == "failed" and elected.command.output_failure is not None
    assert await app.publish_producer_outcome(command, context=context) == elected
    pending = await app.pending_producer_outputs(
        command.admission.prepared.recipient, context=CONTEXT
    )
    token = next(
        item.recovery for item in pending.items if item.recovery.registration == command.operation
    )
    recovery = ProducerDeliveryRecovery(**token.model_dump(), destination=destination.operation)
    resolver.recipient.denied = True
    excluded = await app.reconcile_producer_delivery(recovery, context=CONTEXT, exclude=True)
    assert excluded.state == "excluded"
    retired = await app.retire_producer_export(command, destination.operation, context=CONTEXT)
    assert retired.state == "excluded"
    final = await app.settle_producer_output(command, context=CONTEXT)
    assert final.delivery == "excluded"
    assert len(provider.requests) == 1
    assert not (
        await app.pending_producer_outputs(command.admission.prepared.recipient, context=CONTEXT)
    ).items


@pytest.mark.anyio
@pytest.mark.parametrize("delayed_prepare", [False, True, "expired"])
async def test_unprepared_delivery_is_excluded_without_disclosure(
    native_stores, monkeypatch, delayed_prepare
):
    values, context = await completed_export_scenario(native_stores, monkeypatch)
    original, resolver, _, provider, session, _, command, _ = values
    destination = command.destinations[0]
    await original.export_producer_output(command, destination.operation, context=context)
    elected = await original.publish_producer_outcome(
        command, destination=destination.operation, context=context
    )
    entered, release = asyncio.Event(), asyncio.Event()
    task = None
    if delayed_prepare is True:
        from contextlib import asynccontextmanager

        from cayu.collaboration import _producer_delivery as delivery_module

        acquire = delivery_module.acquire_clarification_source

        @asynccontextmanager
        async def paused_source(*args, **kwargs):
            async with acquire(*args, **kwargs) as source:
                entered.set()
                await release.wait()
                yield source

        monkeypatch.setattr(delivery_module, "acquire_clarification_source", paused_source)
        task = asyncio.create_task(
            original.deliver_producer_output(
                command, destination.operation, context=context, prepare_only=True
            )
        )
        await asyncio.wait_for(entered.wait(), 60)
    elif delayed_prepare == "expired":
        from cayu.collaboration._contracts import CollaborationConflict

        store, initialized = original._participant_coordinator._ready()
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            repository_type = type(tx)

        async def expired_now(self):
            return destination.attempt.deadline_at_ms + 1

        monkeypatch.setattr(repository_type, "now_ms", expired_now)
        with pytest.raises((CollaborationConflict, CollaborationUnavailable)):
            await original.deliver_producer_output(command, destination.operation, context=context)
    else:
        resolver.recipient.denied = True
        original._session_export_coordinator.registration.policy.denied.update(
            {"source", "readback", "export", "expose", "append", "retire", "release"}
        )
    app = make_app(
        native_stores[2](),
        original._participant_coordinator._registration,
        session_store=original.session_store,
        budget_ledger=original.budget_ledger,
        budget_binding_receiver=original.budget_binding_receiver,
        enable_common_root_budget_binding=True,
        collaboration_requests=original._request_coordinator._registration,
        session_exports=original._session_export_coordinator.registration,
    )
    await app.initialize_collaboration()
    app._request_coordinator._owners.observation_timeout = 60
    app._session_export_coordinator.owners.observation_timeout = 60
    pending = await app.pending_producer_outputs(
        command.admission.prepared.recipient, context=CONTEXT
    )
    token = next(
        item.recovery for item in pending.items if item.recovery.registration == command.operation
    )
    recovery = ProducerDeliveryRecovery(**token.model_dump(), destination=destination.operation)
    try:
        excluded = await app.reconcile_producer_delivery(recovery, context=CONTEXT, exclude=True)
    finally:
        release.set()
        if task is not None:
            result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 60)
            assert isinstance(result[0], Exception)
    resolver.recipient.denied = True
    assert excluded.state == "excluded"
    assert await app.reconcile_producer_delivery(recovery, context=CONTEXT) == excluded
    retired = await app.retire_producer_export(command, destination.operation, context=CONTEXT)
    assert retired.state == "retired"
    backend, address = native_stores[3]
    if backend != "memory":
        assert await process_readback(backend, address, command) == retired
    final = await app.settle_producer_output(command, context=CONTEXT)
    assert final.delivery == "excluded"
    assert (
        await app.retire_producer_export(command, destination.operation, context=CONTEXT) == retired
    )
    await app.session_store.delete_session(session.id)
    assert (
        await app.retire_producer_export(command, destination.operation, context=CONTEXT) == retired
    )
    historical = await app.inspect_collaboration_request(
        command.admission.expected, context=resolver.sender.context
    )
    assert historical.state == "answered" and historical.outcome == elected
    assert len(provider.requests) == 1
    assert not (
        await app.pending_producer_outputs(command.admission.prepared.recipient, context=CONTEXT)
    ).items


@pytest.mark.anyio
async def test_excluded_export_cleanup_survives_revocation_and_owner_reconstruction(
    native_stores, monkeypatch
):
    values, context = await completed_export_scenario(native_stores, monkeypatch)
    original, resolver, _, provider, session, _, command, _ = values
    destination = command.destinations[0]
    exported = await original.export_producer_output(
        command, destination.operation, context=context
    )
    elected = await original.publish_producer_outcome(
        command, destination=destination.operation, context=context
    )
    source = await original.lookup_session_export(exported.request, context=context)
    policy = original._session_export_coordinator.registration.policy
    policy.register_export(
        source.receipt,
        payload_sha256=digest({"text": "retained answer", "artifact_commitments": []}),
        consumer_id=destination.recipient.participant_id,
    )
    policy.allowed_receipts.add(command.operation.caller_key)
    prepared = await original.deliver_producer_output(
        command, destination.operation, context=context, prepare_only=True
    )
    with pytest.raises(CollaborationUnavailable):
        await original.retire_producer_export(command, destination.operation, context=CONTEXT)
    pending = await original.pending_producer_outputs(
        command.admission.prepared.recipient, context=CONTEXT
    )
    token = next(
        item.recovery for item in pending.items if item.recovery.registration == command.operation
    )
    recovery = ProducerDeliveryRecovery(**token.model_dump(), destination=destination.operation)
    resolver.recipient.denied = True
    policy.denied.update({"source", "readback", "export", "expose", "append", "retire", "release"})
    excluded = await original.reconcile_producer_delivery(recovery, context=CONTEXT, exclude=True)
    assert excluded.state == "excluded"

    backend, address = native_stores[3]
    sessions = original.session_store
    if backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        sessions = SQLiteSessionStore(Path(address).with_name("sessions.sqlite"))
    elif backend == "postgres":
        from cayu.storage.postgres import PostgresSessionStore

        sessions = PostgresSessionStore(address)
    app = make_app(
        native_stores[2](),
        original._participant_coordinator._registration,
        session_store=sessions,
        budget_ledger=original.budget_ledger,
        budget_binding_receiver=original.budget_binding_receiver,
        enable_common_root_budget_binding=True,
        collaboration_requests=original._request_coordinator._registration,
        session_exports=original._session_export_coordinator.registration,
    )
    try:
        await app.initialize_collaboration()
        assert not app._provider_registry.registrations
        app._request_coordinator._owners.observation_timeout = 60
        app._session_export_coordinator.owners.observation_timeout = 60
        read = type(sessions).read_peer_content_attempt
        for wrong in ("missing", "foreign"):

            async def unavailable(self, request, wrong=wrong):
                actual = await read(self, request)
                assert actual is not None and actual.status == "excluded"
                return (
                    None
                    if wrong == "missing"
                    else actual.model_copy(update={"operation_key": "foreign"})
                )

            with monkeypatch.context() as fault:
                fault.setattr(type(sessions), "read_peer_content_attempt", unavailable)
                assert sessions._supports_producer_attachment_protocol()
                with pytest.raises(CollaborationUnavailable):
                    await app.retire_producer_export(
                        command, destination.operation, context=CONTEXT
                    )
            with pytest.raises(CollaborationUnavailable):
                await app.settle_producer_output(command, context=CONTEXT)
            assert await sessions.load(session.id) is not None
        receiver = app._request_coordinator._registration.receiving_owner
        retire = receiver._retire_producer_export
        entered, release = asyncio.Event(), asyncio.Event()
        results, cancellations = [], []

        async def delayed_ack(*args, **kwargs):
            entered.set()
            await release.wait()
            results.append(await retire(*args, **kwargs))
            raise ConnectionError("Export retirement committed; acknowledgement lost")

        async def observe():
            try:
                return await app.retire_producer_export(
                    command, destination.operation, context=CONTEXT
                )
            except asyncio.CancelledError:
                cancellations.append(True)
                raise

        task, owners = None, ()
        try:
            with monkeypatch.context() as fault:
                fault.setattr(receiver, "_retire_producer_export", delayed_ack)
                task = asyncio.create_task(observe())
                await asyncio.wait_for(entered.wait(), 60)
                task.cancel()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled() and task.cancelling() == 2 and cancellations == [True]
                owners = tuple(app._request_coordinator._owners.pending)
                assert owners and not any(owner.done() for owner in owners)
                with pytest.raises(CollaborationUnavailable):
                    await app.settle_producer_output(command, context=CONTEXT)
                release.set()
                outcomes = await asyncio.wait_for(
                    asyncio.gather(*owners, return_exceptions=True), 60
                )
                assert any(isinstance(outcome, BaseException) for outcome in outcomes)
                assert len(results) == 1 and results[0].state == "retired"
        finally:
            release.set()
            await asyncio.gather(
                *owners, *((task,) if task is not None else ()), return_exceptions=True
            )
        retired = await app.retire_producer_export(command, destination.operation, context=CONTEXT)
        assert retired.state == "retired"
        assert retired.closure is None and retired.delivery == prepared.operation
        if backend != "memory":
            assert await process_readback(backend, address, command) == retired
        assert (
            await app.retire_producer_export(command, destination.operation, context=CONTEXT)
            == retired
        )
        retired_events = await sessions.query_events(
            EventQuery(
                session_id=session.id, event_types=(EventType.SESSION_EXPORT_RETIRED,), limit=2
            )
        )
        assert len(retired_events) == 1
        final = await app.settle_producer_output(command, context=CONTEXT)
        assert final.delivery == "excluded"
        assert await app.reconcile_producer_delivery(recovery, context=CONTEXT) == excluded
        # Native deletion does not erase authenticated retirement or change the
        # original answered election into an invented cancellation.
        await sessions.delete_session(session.id)
        if backend != "memory":
            assert await process_readback(backend, address, command) == retired
        assert (
            await app.retire_producer_export(command, destination.operation, context=CONTEXT)
            == retired
        )
        assert await app.settle_producer_output(command, context=CONTEXT) == final
        with pytest.raises(CollaborationAccessDenied):
            await app.publish_producer_outcome(
                command, destination=destination.operation, context=context
            )
        historical = await app.inspect_collaboration_request(
            command.admission.expected, context=resolver.sender.context
        )
        assert historical.state == "answered" and historical.outcome == elected
        assert len(provider.requests) == 1
    finally:
        await app.drain_collaboration_requests()
        if sessions is not original.session_store:
            await sessions.close()
