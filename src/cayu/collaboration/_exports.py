"""Lazy public participant-administration exports."""

EXPORTS: dict[str, tuple[str, str]] = {
    "ParticipantCommand": ("cayu.collaboration.participants", "ParticipantCommand"),
    "CollaborationConflict": ("cayu.collaboration._contracts", "CollaborationConflict"),
    "CollaborationContractError": ("cayu.collaboration._contracts", "CollaborationContractError"),
    "CollaborationCapabilityUnavailable": (
        "cayu.collaboration._capabilities",
        "CollaborationCapabilityUnavailable",
    ),
    "CollaborationAccessContext": ("cayu.collaboration.access", "CollaborationAccessContext"),
    "CollaborationAccessDenied": ("cayu.collaboration.access", "CollaborationAccessDenied"),
    "CollaborationAccessGrant": ("cayu.collaboration.access", "CollaborationAccessGrant"),
    "CollaborationAccessPolicy": ("cayu.collaboration.access", "CollaborationAccessPolicy"),
    "CollaborationRegistration": ("cayu.collaboration.access", "CollaborationRegistration"),
    "CollaborationStore": ("cayu.collaboration.base", "CollaborationStore"),
    "InMemoryCollaborationStore": ("cayu.collaboration.memory", "InMemoryCollaborationStore"),
    "CollaborationBootstrap": ("cayu.collaboration.participants", "CollaborationBootstrap"),
    "CollaborationCapacityExceeded": (
        "cayu.collaboration.participants",
        "CollaborationCapacityExceeded",
    ),
    "CollaborationInitialization": (
        "cayu.collaboration.participants",
        "CollaborationInitialization",
    ),
    "CollaborationLimits": ("cayu.collaboration.participants", "CollaborationLimits"),
    "CollaborationNotInitialized": (
        "cayu.collaboration.participants",
        "CollaborationNotInitialized",
    ),
    "CollaborationUnavailable": ("cayu.collaboration.participants", "CollaborationUnavailable"),
    "ParticipantAlias": ("cayu.collaboration.participants", "ParticipantAlias"),
    "ParticipantAliasChange": ("cayu.collaboration.participants", "ParticipantAliasChange"),
    "ParticipantConfiguration": ("cayu.collaboration.participants", "ParticipantConfiguration"),
    "ParticipantConfigurationRef": (
        "cayu.collaboration.participants",
        "ParticipantConfigurationRef",
    ),
    "ParticipantConfigure": ("cayu.collaboration.participants", "ParticipantConfigure"),
    "ParticipantCreate": ("cayu.collaboration.participants", "ParticipantCreate"),
    "ParticipantCursor": ("cayu.collaboration.participants", "ParticipantCursor"),
    "ParticipantEvent": ("cayu.collaboration.participants", "ParticipantEvent"),
    "ParticipantEventCursor": ("cayu.collaboration.participants", "ParticipantEventCursor"),
    "ParticipantEventPage": ("cayu.collaboration.participants", "ParticipantEventPage"),
    "ParticipantInspection": ("cayu.collaboration.participants", "ParticipantInspection"),
    "ParticipantIntent": ("cayu.collaboration.participants", "ParticipantIntent"),
    "ParticipantPage": ("cayu.collaboration.participants", "ParticipantPage"),
    "ParticipantReceipt": ("cayu.collaboration.participants", "ParticipantReceipt"),
    "ParticipantRef": ("cayu.collaboration.participants", "ParticipantRef"),
    "ParticipantSnapshot": ("cayu.collaboration.participants", "ParticipantSnapshot"),
    "PostgresCollaborationStore": (
        "cayu.storage.collaboration_postgres",
        "PostgresCollaborationStore",
    ),
    "SQLiteCollaborationStore": ("cayu.storage.collaboration_sqlite", "SQLiteCollaborationStore"),
}

PUBLIC_NAMES = tuple(EXPORTS)
