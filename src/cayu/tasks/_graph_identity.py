"""Task graph identity validation and size limits."""

from __future__ import annotations

from cayu._validation import require_durable_clean_nonblank

# The byte bound is what keeps a graph document small; node and edge counts are
# sized so a fan-out over a realistic batch (one task per file or record) fits.
TASK_GRAPH_MAX_NODES = 1024
TASK_GRAPH_MAX_EDGES = 4096
TASK_GRAPH_MAX_BYTES = 4 * 1024 * 1024
TASK_GRAPH_ID_MAX_BYTES = 256


def graph_identifier(value: str) -> str:
    value = require_durable_clean_nonblank(value, "graph identity")
    if len(value.encode("utf-8")) > TASK_GRAPH_ID_MAX_BYTES:
        raise ValueError("Graph identity exceeds its byte limit.")
    return value
