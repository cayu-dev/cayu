"""Closed producer cleanup excludes late publication without restoring disclosure."""

import asyncio
import threading

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.collaboration._producer_cleanup import settle_producer_output
from cayu.collaboration._producer_export import export_producer_output
from cayu.collaboration._producer_export_cleanup import retire_unneeded_producer_export
from cayu.collaboration._producer_export_retirement import retire_producer_export_native
from cayu.collaboration._producer_export_store import read_export
from cayu.collaboration._producer_recovery import pending_producer_outputs
from cayu.collaboration._request_store import retained_request
from cayu.collaboration._session_export_store import ExportPreparation, ExportRecord
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestControl


@pytest.mark.anyio
@pytest.mark.parametrize("phase", ["unregistered", "prepared", "published"])
async def test_retire_closed_export_after_permanent_disclosure_revocation(
    native_stores, monkeypatch, phase
):
    values, context = await completed_export_scenario(native_stores, monkeypatch)
    app, resolver, admission, provider, session, initialized, proposal, _ = values
    destination = proposal.destinations[0]
    exports = app._session_export_coordinator
    projector = exports.projectors[destination.projector]
    entered, release = threading.Event(), threading.Event()
    original = projector.project
    registration_entered, registration_release = asyncio.Event(), asyncio.Event()

    def blocked(source):
        entered.set()
        if not release.wait(180):
            raise AssertionError("Retirement test did not release projection")
        return original(source)

    observer = None
    try:
        if phase in ("unregistered", "prepared"):
            if phase == "prepared":
                monkeypatch.setattr(projector, "project", blocked)
            else:
                original_register = exports.participants.register

                async def delayed_registration(admission):
                    registration_entered.set()
                    await registration_release.wait()
                    return await original_register(admission)

                monkeypatch.setattr(exports.participants, "register", delayed_registration)
            observer = asyncio.create_task(
                export_producer_output(app, proposal, destination.operation, context=context)
            )
            if phase == "prepared":
                assert await asyncio.to_thread(entered.wait, 60)
            else:
                await asyncio.wait_for(registration_entered.wait(), 60)
            observer.cancel()
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert observer.cancelled() and observer.cancelling() == 2
        else:
            await export_producer_output(app, proposal, destination.operation, context=context)
        async with native_stores[0]._transaction(
            initialized.owner.application_scope, write=False
        ) as tx:
            request = await retained_request(
                native_stores[0],
                tx,
                initialized,
                admission.expected.intent.request,
                admission.expected.initiator,
                app._secret_redactor,
            )
            intent = await read_export(tx, proposal, destination, redactor=app._secret_redactor)
        assert request is not None and intent is not None
        if phase == "unregistered":
            from cayu.collaboration._permit_store import registered_receipt

            preparation = await exports._record(
                await app.session_store.load(session.id), intent.request
            )
            assert isinstance(preparation, ExportPreparation)
            async with native_stores[0]._transaction(
                initialized.owner.application_scope, write=False
            ) as tx:
                assert (
                    await registered_receipt(tx, preparation.admission.permit, app._secret_redactor)
                    is None
                )
        closure = await app.control_collaboration_request(
            RequestControl(
                operation=initialized.operation("close-before-export-delivery"),
                expected=admission.expected,
                expected_revision=request.revision,
                kind="cancel",
            ),
            context=resolver.sender.context,
        )
        resolver.recipient.denied = True
        exports.registration.policy.denied.update(
            ("readback", "source", "export", "expose", "retire", "release")
        )
        with pytest.raises(PermissionError):
            await retire_producer_export_native(exports, proposal, closure, intent, authority=None)
        with pytest.raises(CollaborationUnavailable):
            await settle_producer_output(app, proposal)
        status = await retire_unneeded_producer_export(
            app, proposal, destination.operation, context=CONTEXT
        )
        assert status.state == ("retired" if phase == "published" else "excluded")
        assert (
            await retire_unneeded_producer_export(
                app, proposal, destination.operation, context=CONTEXT
            )
            == status
        )
        release.set()
        registration_release.set()
        await asyncio.wait_for(
            asyncio.gather(
                *tuple(exports.owners.pending),
                *tuple(app._request_coordinator._owners.pending),
                return_exceptions=True,
            ),
            60,
        )
        native = await exports._record(await app.session_store.load(session.id), intent.request)
        if phase != "published":
            assert isinstance(native, ExportPreparation) and native.state == "excluded"
            assert native.admission.settled
            if phase == "unregistered":
                from cayu.collaboration._permit_store import prepare_permit_record
                from cayu.collaboration._permits import PermitExclusion
                from cayu.collaboration._request_store import operation_key

                assert projector.calls == 0
                async with native_stores[0]._transaction(
                    initialized.owner.application_scope, write=False
                ) as tx:
                    exclusion = prepare_permit_record(
                        await tx.get(
                            "operations", operation_key(native.admission.permit.operation)
                        ),
                        app._secret_redactor,
                    )
                    assert isinstance(exclusion, PermitExclusion)
                    assert exclusion.expected == native.admission.permit
                    assert exclusion.receiving_receipt.proves_exclusion
        else:
            assert isinstance(native, ExportRecord) and native.state == "retired"
            assert native.admission.settled
        with pytest.raises(CollaborationUnavailable):
            await export_producer_output(app, proposal, destination.operation, context=context)
        pending = await pending_producer_outputs(app, admission.prepared.recipient, context=CONTEXT)
        assert any(item.recovery.registration == proposal.operation for item in pending.items)
        with pytest.raises(ValueError, match="producer output responsibility"):
            await app.session_store.delete_session(session.id)
        final = await settle_producer_output(app, proposal)
        assert final.delivery == "excluded"
        await app.session_store.delete_session(session.id)
        assert await settle_producer_output(app, proposal) == final
        assert (
            await retire_unneeded_producer_export(
                app, proposal, destination.operation, context=CONTEXT
            )
            == status
        )
        assert len(provider.requests) == 1
    finally:
        release.set()
        registration_release.set()
        if observer is not None:
            if not observer.done():
                observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
        await exports.owners.drain()
        await app._request_coordinator._owners.drain()
