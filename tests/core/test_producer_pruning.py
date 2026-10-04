"""Retired producer responsibility is reclaimed through bounded public maintenance."""

import asyncio
import json
import sys

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.producer_pruning_observation import prune_to_receipt, retire_to_receipt
from tests.core.test_collaboration_namespace import rotate
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import app as make_app
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu import ProducerCleanupReclamation, ProducerProgressOccurrence
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._request_store import operation_key
from cayu.collaboration._session_export_store import digest
from cayu.collaboration.exports import SessionExportSettlementRequest
from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.request_access import PreparedAdmissionRegistration, RequestRegistration
from cayu.collaboration.requests import RequestControl
from cayu.runtime._producer_cleanup_receipt import _accepted_source_cleanup


@pytest.mark.anyio
@pytest.mark.parametrize("produced", [False, True, "delivered", "rejected"])
async def test_public_producer_pruning_restarts_between_bounded_batches(
    native_stores, monkeypatch, produced
):
    if produced:
        values, disclosure = await completed_export_scenario(
            native_stores, monkeypatch, planned=True
        )
    else:
        values = await output_scenario(native_stores, with_exports=True, planned=True)
    app, resolver, admission, provider, _, initialized, registration, execution = values
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    if not produced:
        await app.register_producer_output(
            registration, execution, context=resolver.recipient.context
        )
    if produced:
        prior = await app.inspect_collaboration_request(
            admission.expected, context=resolver.recipient.context
        )
        await app.record_producer_progress(
            registration,
            ProducerProgressOccurrence(
                operation=initialized.operation("progress-before-pruning"),
                expected_revision=prior.revision,
                sequence=1,
                kind="published",
            ),
            context=resolver.recipient.context,
        )
    if produced == "delivered":
        destination = registration.destinations[0].operation
        exported = await app.export_producer_output(registration, destination, context=disclosure)
        await app.publish_producer_outcome(
            registration, destination=destination, context=disclosure
        )
        source = await app.lookup_session_export(exported.request, context=disclosure)
        policy = app._session_export_coordinator.registration.policy
        policy.register_export(
            source.receipt,
            payload_sha256=digest({"text": "retained answer", "artifact_commitments": []}),
            consumer_id=registration.destinations[0].recipient.participant_id,
        )
        policy.allowed_receipts.add(registration.operation.caller_key)
        delivered = await app.deliver_producer_output(registration, destination, context=disclosure)
        assert delivered.receipt.status == "appended"
        resolution = resolver.recipient.resolution
        actions = (*resolution.principal.actions, "release")
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
        await app.settle_session_export(
            SessionExportSettlementRequest(
                request=exported.request,
                mode="release",
                operation=exported.request.ref.operation.model_copy(
                    update={"caller_key": "release-before-pruning"}
                ),
            ),
            context=disclosure,
        )
    elif produced == "rejected":
        from cayu import ProducerDeliveryRecovery

        destination = registration.destinations[0]
        projector = app._session_export_coordinator.projectors[destination.projector]
        monkeypatch.setattr(projector, "validate", lambda *args: False)
        with pytest.raises(CollaborationUnavailable):
            await app.export_producer_output(
                registration, destination.operation, context=disclosure
            )
        await app.publish_producer_outcome(registration, context=disclosure)
        pending = await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
        token = next(
            item.recovery
            for item in pending.items
            if item.recovery.registration == registration.operation
        )
        recovery = ProducerDeliveryRecovery(**token.model_dump(), destination=destination.operation)
        await app.reconcile_producer_delivery(recovery, context=CONTEXT, exclude=True)
        await app.retire_producer_export(registration, destination.operation, context=CONTEXT)
    else:
        prior = await app.inspect_collaboration_request(
            admission.expected, context=resolver.sender.context
        )
        await app.control_collaboration_request(
            RequestControl(
                operation=initialized.operation("close-before-pruning"),
                expected=admission.expected,
                expected_revision=prior.revision,
                kind="cancel",
            ),
            context=resolver.sender.context,
        )
    if produced:
        await app.settle_producer_output(registration, context=CONTEXT)
    assert len(provider.requests) == int(bool(produced))
    assert not (
        await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
    ).items
    store = native_stores[0]
    native_app = app
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        final_record = await read_output_registration(
            tx, registration, redactor=app._secret_redactor
        )
    _, rotated = await rotate(store, initialized)
    with pytest.raises(CollaborationUnavailable):
        await app.reclaim_producer_cleanup(rotated.namespace.reference, context=CONTEXT)
    await retire_to_receipt(
        app,
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire-producer"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        ),
    )
    owner_registration = app._participant_coordinator._registration
    cursor_seen = False
    for index in range(64):
        state = await app.inspect_collaboration_namespace(context=CONTEXT)
        batch = NamespacePrune(
            operation=rotated.successor.reference.operation(f"prune-producer-{index}"),
            namespace=rotated.namespace.reference,
            expected_retention_revision=state.retention_revision,
            max_records=2,
        )
        receipt = await prune_to_receipt(app, batch)
        assert 0 < receipt.removed_records <= 2
        assert await prune_to_receipt(app, batch) == receipt
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            cursor_seen |= (
                await tx.get("request_pruning", operation_key(admission.expected.operation))
                is not None
            )
        app = make_app(
            native_stores[2](),
            owner_registration,
            collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
        )
        assert await app.initialize_collaboration() == initialized
        assert await prune_to_receipt(app, batch) == receipt
        if receipt.complete:
            break
    else:
        pytest.fail("Settled producer history was not reclaimed in bounded batches")
    assert cursor_seen
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        assert not await tx.scan_operations(initialized.namespace_incarnation, 1, limit=1)
        assert await tx.get("requests", operation_key(admission.expected.operation)) is None
        assert await tx.get("request_pruning", operation_key(admission.expected.operation)) is None
    assert len(provider.requests) == int(bool(produced))
    if produced:
        # Restart with a newer registered receiving configuration on the same
        # physical native owner. Namespace cleanup is not renewed acquisition.
        native_app = make_app(
            native_stores[2](),
            owner_registration,
            session_store=native_stores[1],
            collaboration_requests=RequestRegistration(
                mandates=resolver,
                prepared_admission=PreparedAdmissionRegistration(
                    receiver=registration.receiver.model_copy(
                        update={"revision": registration.receiver.revision + 1}
                    )
                ),
                max_ttl_ms=300_000,
            ),
        )
        assert await native_app.initialize_collaboration() == initialized
        monkeypatch.setattr(native_app._request_coordinator._owners, "observation_timeout", 60)
        assert not native_app._provider_registry.registrations
    reclaimed = await native_app.reclaim_producer_cleanup(
        rotated.namespace.reference, context=CONTEXT, limit=1
    )
    assert reclaimed.removed == 1 and not reclaimed.remaining
    with pytest.raises(PermissionError):
        await native_stores[1]._retire_native_producer_cleanup(
            reclaimed.retirement, authority=None, limit=1
        )
    assert (
        await native_app.reclaim_producer_cleanup(
            rotated.namespace.reference, context=CONTEXT, limit=1
        )
    ).removed == 0
    if produced:
        # A previously genuine source-acceptance handoff cannot resurrect native
        # history after its namespace fence was durably installed.
        with pytest.raises(ValueError, match="retired"):
            await native_stores[1]._complete_native_producer_cleanup(
                final_record, authority=_accepted_source_cleanup(final_record)
            )
        with pytest.raises(ValueError, match="retired"):
            await native_stores[1]._read_completed_native_producer_cleanup(final_record)
        assert (
            await native_app.reclaim_producer_cleanup(
                rotated.namespace.reference, context=CONTEXT, limit=1
            )
        ).removed == 0
        backend, address = native_stores[3]
        if backend != "memory":
            worker = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "tests.recovery.producer_retirement_reader_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                output, error = await asyncio.wait_for(
                    worker.communicate(
                        json.dumps(
                            {
                                "backend": backend,
                                "address": address,
                                "record": final_record.model_dump(mode="json"),
                                "namespace": rotated.namespace.reference.model_dump(mode="json"),
                            }
                        ).encode()
                    ),
                    90,
                )
                assert worker.returncode == 0, error.decode()
                restored = ProducerCleanupReclamation.model_validate_json(output)
                assert restored.removed == 0 and not restored.remaining
                assert restored.retirement.namespace == rotated.namespace.reference
            finally:
                if worker.returncode is None:
                    worker.kill()
                    await worker.wait()
    await app.drain_collaboration_requests()
