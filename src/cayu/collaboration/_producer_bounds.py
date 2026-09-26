"""Shared finite producer retention and mandatory control bounds."""

MAX_OUTPUT_DESTINATIONS = 16
MAX_OUTPUT_BYTES = 64 * 1024
MAX_OUTPUT_PROGRESS = 64


def supports_peer_delivery(store) -> bool:
    """Declared receiving protocol only; not authenticated append/cleanup evidence."""
    version = store.peer_content_version
    return type(version) is int and version == 1
