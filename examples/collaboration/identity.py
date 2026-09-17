"""Administer participant identities without starting a model or worker."""

from __future__ import annotations

import asyncio

from cayu import (
    CayuApp,
    CollaborationAccessContext,
    CollaborationAccessDenied,
    CollaborationAccessGrant,
    CollaborationAccessPolicy,
    CollaborationBootstrap,
    CollaborationLimits,
    CollaborationRegistration,
    InMemoryCollaborationStore,
    NamespacePrune,
    NamespaceRetire,
    NamespaceRotate,
    ParticipantConfiguration,
    ParticipantConfigurationRef,
    ParticipantCreate,
    ParticipantLifecycleChange,
)


class ApplicationPolicy(CollaborationAccessPolicy):
    def authorize(self, context, *, application_scope, action):
        if context.principal != "application-admin":
            raise CollaborationAccessDenied("Participant administration is not authorized.")
        return CollaborationAccessGrant(application_scope=application_scope, participants=None)


async def main() -> None:
    configuration = ParticipantConfiguration(
        definition=ParticipantConfigurationRef(name="security-reviewer", version=1),
        routing=ParticipantConfigurationRef(name="review-routing", version=1),
        admission=ParticipantConfigurationRef(name="review-admission", version=1),
    )
    store = InMemoryCollaborationStore()
    registration = CollaborationRegistration(
        bootstrap=CollaborationBootstrap(
            application_scope="engineering",
            provisioning_scope="development",
            owner_name="participants",
            limits=CollaborationLimits(
                participants=64,
                aliases=64,
                operations=256,
                events=512,
                retained_bytes=4 * 1024 * 1024,
                control_operations=16,
                control_events=16,
                control_bytes=65536,
                namespaces=4,
                generations=8,
                obligations=64,
            ),
        ),
        access_policy=ApplicationPolicy(),
        configurations=(configuration,),
    )
    app = CayuApp(collaboration_store=store, collaboration=registration, enable_logging=False)
    try:
        initialized = await app.initialize_collaboration()
        # The host application supplies this from its authentication boundary.
        context = CollaborationAccessContext(principal="application-admin")
        request = ParticipantCreate(
            operation=initialized.operation("create-reviewer"),
            configuration=configuration,
            alias="security",
        )
        receipt = await app.create_participant(request, context=context)
        assert await app.create_participant(request, context=context) == receipt
        resolved = await app.resolve_participant_alias("security", context=context)
        assert resolved is not None and resolved.target == receipt.participants[0].reference
        disabled = await app.change_participant_lifecycle(
            ParticipantLifecycleChange(
                operation=initialized.operation("disable-reviewer"),
                participant=resolved.target,
                expected_lifecycle_revision=1,
                state="disabled",
            ),
            context=context,
        )
        # Accepted disable and settled disable are separate observations. This
        # identity-only example has no receiving-owner permits to settle.
        inspected = await app.inspect_participant(resolved.target, context=context)
        assert inspected.settlement == "settled"
        assert inspected.participant == disabled.participant
        current = await app.inspect_collaboration_namespace(context=context)
        rotated = await app.rotate_collaboration_namespace(
            NamespaceRotate(
                operation=current.current.reference.operation("rotate"),
                namespace=current.current.reference,
                expected_revision=current.current.revision,
            ),
            context=context,
        )
        assert rotated.successor is not None and rotated.namespace is not None
        await app.retire_collaboration_namespace(
            NamespaceRetire(
                operation=rotated.successor.reference.operation("retire-old-generation"),
                namespace=rotated.namespace.reference,
                expected_revision=rotated.namespace.revision,
                expected_retired_through=0,
            ),
            context=context,
        )
        # Old receipts remain exact until the application explicitly prunes them.
        assert await app.create_participant(request, context=context) == receipt
        retention = await app.inspect_collaboration_namespace(context=context)
        pruned = await app.prune_collaboration_namespace(
            NamespacePrune(
                operation=rotated.successor.reference.operation("prune-old-generation"),
                namespace=rotated.namespace.reference,
                expected_retention_revision=retention.retention_revision,
            ),
            context=context,
        )
        assert pruned.complete
        assert (
            await app.lookup_participant_operation(receipt.expected, context=context)
        ).status == "unavailable"
        evidence = await app.inspect_collaboration_retirement(
            rotated.namespace.reference, context=context
        )
        assert evidence is not None and evidence.content == "pruned"
        print(resolved.target.participant_id)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
