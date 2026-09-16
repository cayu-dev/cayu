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
    ParticipantConfiguration,
    ParticipantConfigurationRef,
    ParticipantCreate,
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
        print(resolved.target.participant_id)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
