"""Finite identity allowance within an export's mandatory future envelopes."""

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.collaboration._contracts import InitiatorBinding, ObjectRef, OwnerRef, snapshot_input

MAX_INITIATOR_BYTES = 8 * 1024


def initiator_bytes(value: InitiatorBinding) -> bytes:
    return canonical_bounded_durable_json_bytes(
        snapshot_input(value),
        "session export initiator",
        max_bytes=MAX_INITIATOR_BYTES,
        max_nodes=128,
        max_nesting=8,
    )


def probe_initiator() -> InitiatorBinding:
    """Maximal shape, minimal strings; callers reserve the remaining byte allowance.

    This is a size probe, never authenticated authority. Fully populated object
    references cover the greatest node count and depth of a later identity.
    """
    owner = OwnerRef(application_scope="x", owner_id="x", incarnation="x")
    reference = ObjectRef(
        owner=owner, kind="participant", object_id="x", incarnation="x", revision=2**53 - 1
    )
    return InitiatorBinding(
        issuer=owner,
        principal="x",
        participant=reference,
        mandate=reference,
        invocation_id=None,
        interaction_id=None,
    )
