"""Task graph identity validation and size limits."""

from __future__ import annotations

from cayu._validation import require_durable_clean_nonblank

TASK_GRAPH_MAX_NODES = 128
TASK_GRAPH_MAX_EDGES = 1024
TASK_GRAPH_MAX_BYTES = 1024 * 1024
TASK_GRAPH_ID_MAX_BYTES = 256


def graph_identifier(value: str) -> str:
    value = require_durable_clean_nonblank(value, "graph identity")
    if len(value.encode("utf-8")) > TASK_GRAPH_ID_MAX_BYTES:
        raise ValueError("Graph identity exceeds its byte limit.")
    return value
