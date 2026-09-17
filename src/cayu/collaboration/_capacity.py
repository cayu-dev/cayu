"""One aggregate admission over retained evidence and pending settlement reserves."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES
from cayu.collaboration.participants import CollaborationCapacityExceeded

if TYPE_CHECKING:
    from cayu.collaboration.base import _Anchor

# Settlement can replace two envelopes, append an event and grow bounded counters.
# Admission reserves their worst-case sizes, not an estimate of a future receipt.
PERMIT_SETTLEMENT_BYTES = 4 * MAX_ENVELOPE_BYTES


def require_capacity(anchor: _Anchor, *, ordinary: bool) -> None:
    limits = anchor.initialization.binding.limits
    if (
        anchor.participant_count > limits.participants
        or anchor.alias_count > limits.aliases
        or anchor.permit_count > limits.obligations
        or anchor.retained_generations > limits.generations
        or anchor.operation_count
        > limits.operations - (limits.control_operations if ordinary else 0)
        or anchor.event_count + anchor.reserved_events
        > limits.events - (limits.control_events if ordinary else 0)
        or anchor.retained_bytes + anchor.reserved_bytes
        > limits.retained_bytes - (limits.control_bytes if ordinary else 0)
    ):
        raise CollaborationCapacityExceeded(
            "Collaboration retained evidence and reservations exceed capacity."
        )
