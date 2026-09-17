"""Failed admission cannot consume the namespace's recovery reserve."""

import pytest
from tests.core.test_participant_identity import configuration, registration
from tests.core.test_session_export_content_release import run_async
from tests.core.test_session_export_mandates import Resolver
from tests.core.test_session_export_mandates import participant_backend as participant_backend
from tests.core.test_session_exports import CONTEXT, OWNER, harness, published
from tests.core.test_session_exports import backend as backend

from cayu.collaboration.access import (
    CollaborationAccessContext,
    CollaborationAccessGrant,
    CollaborationAccessPolicy,
)
from cayu.collaboration.exports import SessionExportCapacityExceeded, SessionExportConflict
from cayu.collaboration.lifecycle import NamespaceRetire, NamespaceRotate
from cayu.collaboration.mandates import MandateChain
from cayu.collaboration.participants import ParticipantConfigure, ParticipantCreate


@run_async
async def test_unregistered_export_keeps_maintenance_capacity_and_recovers(
    backend, participant_backend
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
            reg = registration(
                policy=Access(),
                scope=OWNER.application_scope,
                limits=reg.bootstrap.limits.model_copy(
                    update={"operations": 10, "control_operations": 8}
                ),
            )
            app, _, _, projector = case.app(
                mandates=resolver, collaboration=reg, collaboration_store=participants
            )
            initialized = await app.initialize_collaboration()
            context = CollaborationAccessContext(principal=CONTEXT.principal)
            created = await app.create_participant(
                ParticipantCreate(
                    operation=initialized.operation("participant"), configuration=configuration()
                ),
                context=context,
            )
            participant = created.participants[0].reference
            await app.configure_participant(
                ParticipantConfigure(
                    operation=initialized.operation("configuration"),
                    participant=participant,
                    expected_configuration_revision=1,
                    configuration=configuration(2),
                ),
                context=context,
            )
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
            with pytest.raises(SessionExportCapacityExceeded):
                await app.export_session(request, context=resolver.context)
            administrator, _, _, _ = case.app(collaboration=reg, collaboration_store=participants)
            await administrator.initialize_collaboration()
            # Source exclusion is reserved and commits, but native registration
            # remains fenced until exact exclusion or permanent retirement wins.
            # Creating a negative registration must not steal maintenance slots.
            with pytest.raises(SessionExportCapacityExceeded):
                await administrator.reconcile_session_export(request, context=CONTEXT)
            with pytest.raises(SessionExportConflict):
                await store.delete_session(case.session_id)
            assert projector.calls == 0
            assert not await published(store, case.session_id)
            inspected = await app.inspect_participant(participant, context=context)
            assert inspected.outstanding_obligations == 0
            assert inspected.issued_permit_frontier == 0
            namespace = (await app.inspect_collaboration_namespace(context=context)).current
            rotated = await app.rotate_collaboration_namespace(
                NamespaceRotate(
                    operation=namespace.reference.operation("rotate"),
                    namespace=namespace.reference,
                    expected_revision=namespace.revision,
                ),
                context=context,
            )
            await app.retire_collaboration_namespace(
                NamespaceRetire(
                    operation=rotated.successor.reference.operation("retire"),
                    namespace=rotated.namespace.reference,
                    expected_revision=rotated.namespace.revision,
                    expected_retired_through=0,
                ),
                context=context,
            )
            result = await administrator.reconcile_session_export(request, context=CONTEXT)
            assert result.state == "excluded" and result.receipt is None
            assert await administrator.reconcile_session_export(request, context=CONTEXT) == result
            await store.delete_session(case.session_id)
    finally:
        await participants.close()
