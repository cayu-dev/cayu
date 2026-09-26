"""Optional export capacity cannot consume pre-admitted producer cleanup control."""

from dataclasses import replace

import pytest
from tests.core import test_producer_output_contracts as scenarios
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.collaboration._contracts import OperationRef, OwnerRef
from cayu.collaboration._producer_cleanup import settle_producer_output
from cayu.collaboration._producer_completion import retain_producer_completion
from cayu.collaboration._producer_export import export_producer_output
from cayu.collaboration._producer_export_cleanup import retire_unneeded_producer_export
from cayu.collaboration._request_store import retained_request
from cayu.collaboration.exports import (
    ExportLimits,
    SessionExportConflict,
    SessionExportRef,
    SessionExportRequest,
    SessionExportSettlementRequest,
)
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestControl


@pytest.mark.anyio
@pytest.mark.parametrize("retained_bytes", [65535, 65536])
async def test_producer_launch_rejects_impossible_export_capacity(
    native_stores, monkeypatch, retained_bytes
):
    from cayu.collaboration._producer_registration import register_producer_output
    from cayu.collaboration._producer_store import read_output_registration
    from cayu.runtime._producer_execution import _ProducerExecution

    registration = scenarios.output_exports

    def bounded_exports(*args, **kwargs):
        return replace(
            registration(*args, **kwargs),
            limits=ExportLimits(max_exports=1, max_pending=1, max_retained_bytes=retained_bytes),
        )

    monkeypatch.setattr(scenarios, "output_exports", bounded_exports)
    values = await scenarios.output_scenario(native_stores, with_exports=True)
    app, resolver, admission, provider, _, initialized, proposal, execution = values
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    registered = await register_producer_output(
        app, proposal, execution, context=resolver.recipient.context
    )
    original = resolver.recipient.resolution
    actions = (*original.principal.actions, "execute")
    resolver.recipient.resolution = original.model_copy(
        update={
            "principal": original.principal.model_copy(update={"actions": actions}),
            "chain": original.chain.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(update={"actions": actions})
                        for entry in original.chain.entries
                    )
                }
            ),
        }
    )

    async def launch():
        async for _ in app._execute_participant_session(
            execution,
            participant=admission.prepared.recipient,
            context=CONTEXT,
            producer_output=_ProducerExecution(app, proposal, resolver.recipient.context),
        ):
            pass

    if retained_bytes < 65536:
        with pytest.raises(CollaborationUnavailable):
            await launch()
    else:
        await launch()
    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        record = await read_output_registration(tx, proposal, redactor=app._secret_redactor)
    assert record is not None
    if retained_bytes < 65536:
        assert record == registered
    assert len(provider.requests) == (1 if retained_bytes == 65536 else 0)


@pytest.mark.anyio
async def test_full_optional_export_pool_preserves_producer_exclusion_and_unrelated_work(
    native_stores, monkeypatch
):
    registration = scenarios.output_exports

    def one_optional(*args, **kwargs):
        return replace(
            registration(*args, **kwargs),
            limits=ExportLimits(
                max_exports=1,
                max_pending=1,
                max_retained_bytes=65536,
            ),
        )

    monkeypatch.setattr(scenarios, "output_exports", one_optional)
    values, context = await completed_export_scenario(native_stores, monkeypatch)
    app, resolver, admission, provider, session, initialized, proposal, _ = values
    destination = proposal.destinations[0]
    exports = app._session_export_coordinator
    completion = await retain_producer_completion(app, proposal)
    namespace = await exports.initialize(session.id, context=context)
    optional = SessionExportRequest(
        ref=SessionExportRef(
            session_id=session.id,
            session_instance_id=session.instance_id,
            operation=OperationRef(
                application_scope=namespace.owner.application_scope,
                namespace_incarnation=namespace.namespace_incarnation,
                generation=namespace.generation,
                caller_key="independent-optional-export",
            ),
        ),
        source_indices=completion.output.source_indices,
        source_selection="assistant_visible_text_v1",
        audience=OwnerRef(
            application_scope=destination.recipient.owner.application_scope,
            owner_id=destination.recipient.participant_id,
            incarnation=destination.recipient.incarnation,
        ),
        projector=destination.projector,
        policy=destination.disclosure_policy,
    )
    receipt = await app.export_session(optional, context=context)
    with pytest.raises(CollaborationUnavailable):
        await export_producer_output(app, proposal, destination.operation, context=context)
    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        current = await retained_request(
            native_stores[0],
            tx,
            initialized,
            admission.expected.intent.request,
            admission.expected.initiator,
            app._secret_redactor,
        )
    assert current is not None
    await app.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("close-at-full-export-capacity"),
            expected=admission.expected,
            expected_revision=current.revision,
            kind="cancel",
        ),
        context=resolver.sender.context,
    )
    status = await retire_unneeded_producer_export(
        app, proposal, destination.operation, context=CONTEXT
    )
    assert status.state == "excluded"
    root = await exports.root(await app.session_store.load(session.id))
    assert (root.export_count, root.pending_count, root.retained_bytes) == (1, 1, 65536)
    assert (root.producer_exclusion_count, root.producer_exclusion_bytes) == (1, 65536)
    found = await app.lookup_session_export(optional, context=context)
    assert found.status == "match" and found.receipt == receipt
    final = await settle_producer_output(app, proposal)
    assert final.delivery == "excluded"
    with pytest.raises(SessionExportConflict):
        await app.session_store.delete_session(session.id)
    # Mandatory cleanup did not cancel the independent export or release its pin.
    original = resolver.recipient.resolution
    resolver.recipient.resolution = original.model_copy(
        update={
            "principal": original.principal.model_copy(
                update={"actions": (*original.principal.actions, "retire")}
            ),
            "chain": original.chain.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(update={"actions": (*entry.actions, "retire")})
                        for entry in original.chain.entries
                    )
                }
            ),
        }
    )
    await app.settle_session_export(
        SessionExportSettlementRequest(
            request=optional,
            mode="retire",
            operation=optional.ref.operation.model_copy(
                update={"caller_key": "retire-independent-export"}
            ),
        ),
        context=context,
    )
    await app.session_store.delete_session(session.id)
    assert await settle_producer_output(app, proposal) == final
    assert len(provider.requests) == 1
