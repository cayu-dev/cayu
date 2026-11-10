"""Explicit public exports; implementations load on first access."""

EXPORTS: dict[str, tuple[str, str]] = {
    "CloudConnectionBroker": ("cayu.connections.client", "CloudConnectionBroker"),
    "ConnectionCallError": ("cayu.connections.client", "ConnectionCallError"),
    "ConnectionClient": ("cayu.connections.client", "ConnectionClient"),
    "ConnectionResponse": ("cayu.connections.client", "ConnectionResponse"),
    "DEFAULT_TIMEOUT_SECONDS": ("cayu.connections.client", "DEFAULT_TIMEOUT_SECONDS"),
    "LocalConnections": ("cayu.connections.client", "LocalConnections"),
    "LocalSlot": ("cayu.connections.client", "LocalSlot"),
    "MAX_BODY_BYTES": ("cayu.connections.client", "MAX_BODY_BYTES"),
    "MAX_RETRY_AFTER_SECONDS": ("cayu.connections.client", "MAX_RETRY_AFTER_SECONDS"),
    "ModelSlot": ("cayu.connections.models", "ModelSlot"),
    "ModelSlotUnavailable": ("cayu.connections.models", "ModelSlotUnavailable"),
    "ModelSlots": ("cayu.connections.models", "ModelSlots"),
    "Query": ("cayu.connections.client", "Query"),
    "TOKEN_ENVIRONMENT": ("cayu.connections.client", "TOKEN_ENVIRONMENT"),
    "URL_ENVIRONMENT": ("cayu.connections.client", "URL_ENVIRONMENT"),
    "retry_after_seconds": ("cayu.connections.client", "retry_after_seconds"),
}

PUBLIC_NAMES = sorted(EXPORTS)
