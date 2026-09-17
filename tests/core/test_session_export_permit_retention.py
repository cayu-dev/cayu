"""Native permit pruning must not strand a published source's settlement ACK."""

import pytest
from tests.core.test_participant_identity import configuration, registration
from tests.core.test_session_export_content_release import run_async
from tests.core.test_session_export_mandates import Resolver
from tests.core.test_session_export_mandates import participant_backend as participant_backend
from tests.core.test_session_exports import CONTEXT, OWNER, harness, published
from tests.core.test_session_exports import backend as backend

from cayu.collaboration._session_export_store import ROOT_KEY, read_scope
from cayu.collaboration.access import (
    CollaborationAccessContext,
    CollaborationAccessGrant,
    CollaborationAccessPolicy,
)
from cayu.collaboration.exports import (
    SessionExportConflict,
    SessionExportSettlementRequest,
    SessionExportUnavailable,
)
from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire, NamespaceRotate
from cayu.collaboration.mandates import MandateChain
from cayu.collaboration.participants import CollaborationUnavailable, ParticipantCreate


@pytest.mark.parametrize(
    "maintenance", ["retired", "pruned", "pruned_during_settlement", "missing", "wrong_namespace"]
)
@run_async
async def test_published_admission_reconciles_after_native_settlement_pruning(
    backend, participant_backend, maintenance, monkeypatch
):
    class Access(CollaborationAccessPolicy):
        def authorize(self, context, *, application_scope, action):
            assert context.principal == CONTEXT.principal
            return CollaborationAccessGrant(application_scope=application_scope, participants=None)

    participants = participant_backend
    try:
        async with harness(backend) as case:
            plain, store, _, _ = case.app()
            await case.create(store)
            request = await case.request(plain)
            resolver = Resolver(request)
            reg = registration(policy=Access(), scope=OWNER.application_scope)
            app, _, _, projector = case.app(
                mandates=resolver, collaboration=reg, collaboration_store=participants
            )
            await app.initialize_collaboration()
            access = CollaborationAccessContext(principal=CONTEXT.principal)
            current_namespace = await app.inspect_collaboration_namespace(context=access)
            created = await app.create_participant(
                ParticipantCreate(
                    operation=current_namespace.current.reference.operation(case.session_id),
                    configuration=configuration(),
                ),
                context=access,
            )
            participant = created.participants[0].reference
            root = resolver.resolution.chain.entries[0].model_copy(
                update={"participant": participant}
            )
            resolver.resolution = resolver.resolution.model_copy(
                update={
                    "principal": resolver.resolution.principal.model_copy(
                        update={"participants": (participant,)}
                    ),
                    "chain": MandateChain(entries=(root,)),
                }
            )
            resolver.context = resolver.context.model_copy(
                update={
                    "mandate": resolver.context.mandate.model_copy(
                        update={"participant": participant}
                    )
                }
            )
            owner = app._session_export_coordinator
            original_publish = owner.publish

            async def fail_source_ack(session, before, after, key, record, *args, **kwargs):
                if "receipt" in record and record.get("admission", {}).get("settled") is True:
                    raise RuntimeError("Source acknowledgement unavailable after native settlement")
                return await original_publish(session, before, after, key, record, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(owner, "publish", fail_source_ack)
                with pytest.raises(SessionExportUnavailable):
                    await app.export_session(request, context=resolver.context)
            receipt = (await plain.lookup_session_export(request, context=CONTEXT)).receipt
            assert projector.calls == 1
            assert (
                await app.inspect_participant(participant, context=access)
            ).outstanding_obligations == 0

            async def assert_fenced():
                with read_scope(case.session_id):
                    checkpoint = await store.load_checkpoint(case.session_id)
                assert checkpoint[ROOT_KEY]["admission_count"] == 1
                with pytest.raises(SessionExportConflict):
                    await store.delete_session(case.session_id)

            await assert_fenced()
            if maintenance != "missing":
                before = await app.inspect_collaboration_namespace(context=access)
                rotated = await app.rotate_collaboration_namespace(
                    NamespaceRotate(
                        operation=before.current.reference.operation(case.session_id + "-rotate"),
                        namespace=before.current.reference,
                        expected_revision=before.current.revision,
                    ),
                    context=access,
                )
                await app.retire_collaboration_namespace(
                    NamespaceRetire(
                        operation=rotated.successor.reference.operation(
                            case.session_id + "-retire"
                        ),
                        namespace=rotated.namespace.reference,
                        expected_revision=rotated.namespace.revision,
                        expected_retired_through=before.retired_through,
                    ),
                    context=access,
                )

                async def prune():
                    retained = await app.inspect_collaboration_namespace(context=access)
                    result = await app.prune_collaboration_namespace(
                        NamespacePrune(
                            operation=rotated.successor.reference.operation(
                                case.session_id + "-prune"
                            ),
                            namespace=rotated.namespace.reference,
                            expected_retention_revision=retained.retention_revision,
                            max_records=32,
                        ),
                        context=access,
                    )
                    assert result.complete

                if maintenance in {"pruned", "wrong_namespace"}:
                    await prune()

            # A reconstructed export coordinator must use the native owner's
            # retained authority, not a cached settlement receipt or projector.
            reopened, _, _, _ = case.app(
                collaboration=reg, collaboration_store=participants, projectors=()
            )
            await reopened.initialize_collaboration()
            if maintenance == "pruned_during_settlement":
                original_settle = participants._settle_permit

                async def racing_settle(*args, **kwargs):
                    await prune()
                    return await original_settle(*args, **kwargs)

                with monkeypatch.context() as patch:
                    patch.setattr(participants, "_settle_permit", racing_settle)
                    reconciled = await reopened.reconcile_session_export(request, context=CONTEXT)
            else:
                if maintenance in {"missing", "wrong_namespace"}:
                    original_inspect = participants.inspect_retirement

                    async def unavailable(*args, **kwargs):
                        raise CollaborationUnavailable("Exact registration is unavailable")

                    async def wrong_namespace(*args, **kwargs):
                        result = await original_inspect(*args, **kwargs)
                        return result.model_copy(
                            update={
                                "namespace": result.namespace.model_copy(
                                    update={"namespace_incarnation": "other"}
                                )
                            }
                        )

                    with monkeypatch.context() as patch:
                        if maintenance == "missing":
                            patch.setattr(participants, "_settle_permit", unavailable)
                        else:
                            patch.setattr(participants, "inspect_retirement", wrong_namespace)
                        with pytest.raises(
                            SessionExportUnavailable
                            if maintenance == "missing"
                            else SessionExportConflict
                        ):
                            await reopened.reconcile_session_export(request, context=CONTEXT)
                    await assert_fenced()
                reconciled = await reopened.reconcile_session_export(request, context=CONTEXT)
            assert reconciled.receipt == receipt
            if maintenance in {"pruned", "pruned_during_settlement", "wrong_namespace"}:
                retained = await reopened.inspect_collaboration_namespace(context=access)
                assert retained.pruned_through >= rotated.namespace.reference.generation
            assert await reopened.reconcile_session_export(request, context=CONTEXT) == reconciled
            with read_scope(case.session_id):
                checkpoint = await store.load_checkpoint(case.session_id)
            assert checkpoint[ROOT_KEY]["admission_count"] == 0
            assert projector.calls == 1
            assert [event.id for event in await published(store, case.session_id)] == [
                receipt.event_id
            ]
            await reopened.settle_session_export(
                SessionExportSettlementRequest(
                    request=request,
                    operation=request.ref.operation.model_copy(
                        update={"caller_key": "retire-content"}
                    ),
                    mode="retire",
                ),
                context=CONTEXT,
            )
            await store.delete_session(case.session_id)
            if maintenance == "retired":
                await prune()
    finally:
        await participants.close()
