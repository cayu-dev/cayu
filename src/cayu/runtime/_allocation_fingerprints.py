"""Validate allocation continuity and checkpoint identity evidence."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from cayu._validation import canonical_durable_json_bytes
from cayu.environments.factory import EnvironmentFactoryResult
from cayu.runtime._environment_allocation import (
    ENVIRONMENT_FACTORY_ALLOCATION_IDENTITY_BINDING_CHECKPOINT_KEY,
    ENVIRONMENT_FACTORY_ALLOCATION_IDENTITY_CHECKPOINT_KEY,
    DurableEnvironmentAllocationContext,
    EnvironmentAllocationReceipt,
    allocation_identity_binding,
)


def _live_allocation_fingerprint(
    allocation: DurableEnvironmentAllocationContext | None,
    receipt: EnvironmentAllocationReceipt | None,
) -> str | None:
    """Return content-free continuity authority for one acknowledged allocation."""

    if allocation is None and receipt is None:
        return None
    reconnect_metadata = (
        receipt.reconnect_metadata
        if allocation is None and receipt is not None
        else allocation.acknowledged_reconnect_metadata
        if allocation is not None
        else None
    )
    if reconnect_metadata is None:
        return None
    if allocation is not None:
        intent = allocation.intent
    elif receipt is not None:
        intent = receipt.intent
    else:  # pragma: no cover - guarded above
        return None
    return sha256(
        canonical_durable_json_bytes(
            {
                "record_type": "cayu.live-environment-allocation",
                "schema_version": 1,
                "allocation_id": intent.allocation_id,
                "provider": intent.provider,
                "adapter_generation": intent.adapter_generation,
                "session_id": intent.session_id,
                "environment_name": intent.environment_name,
                "reconnect_metadata": reconnect_metadata,
            },
            "live_environment_allocation",
        )
    ).hexdigest()


def _validated_allocation_fingerprint(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "Environment factory allocation fingerprint must be a lowercase SHA-256 digest."
        )
    return value


def _reconnect_allocation_fingerprint(reconnect_metadata: dict[str, Any]) -> str | None:
    """Read the legacy top-level identity some factories put in reconnect metadata."""

    value = reconnect_metadata.get("allocation_fingerprint")
    if value is None:
        return None
    return _validated_allocation_fingerprint(value)


def _published_allocation_fingerprint(
    result: EnvironmentFactoryResult,
    reconnect_metadata: dict[str, Any],
) -> str | None:
    """Return the backend identity a factory result publishes, typed field first."""

    legacy = _reconnect_allocation_fingerprint(reconnect_metadata)
    published = result.allocation_fingerprint
    if published is None:
        return legacy
    published = _validated_allocation_fingerprint(published)
    if legacy is not None and legacy != published:
        raise ValueError(
            "Environment factory allocation fingerprint conflicts with its reconnect metadata."
        )
    return published


def _checkpoint_allocation_fingerprint(
    checkpoint: dict[str, Any] | None,
    *,
    environment_name: str,
    reconnect_metadata: dict[str, Any],
) -> str | None:
    """Return the checkpointed factory-published identity of one live allocation.

    Checkpoints written before the identity had its own record carry it only as
    a top-level key of the reconnect metadata.

    Releases from before that record (down to the compatibility floor) dispose
    of and re-create an allocation without removing it, so an identity can
    outlive the allocation it named. Each identity is therefore bound to the
    reconnect metadata it was published beside. One whose binding doesn't
    match the current metadata belongs to a replaced allocation and is
    ignored. One written before bindings existed is trusted only where it
    agrees with the metadata's own top-level identity: that one is rewritten
    with every allocation, so it wins a disagreement.
    """

    legacy = _reconnect_allocation_fingerprint(reconnect_metadata)
    identities = (checkpoint or {}).get(ENVIRONMENT_FACTORY_ALLOCATION_IDENTITY_CHECKPOINT_KEY)
    if identities is None:
        return legacy
    if type(identities) is not dict:
        raise ValueError("Environment factory allocation identities must be an object.")
    value = identities.get(environment_name)
    if value is None:
        return legacy
    value = _validated_allocation_fingerprint(value)
    bindings = (checkpoint or {}).get(
        ENVIRONMENT_FACTORY_ALLOCATION_IDENTITY_BINDING_CHECKPOINT_KEY
    )
    if bindings is not None and type(bindings) is not dict:
        raise ValueError("Environment factory allocation identity bindings must be an object.")
    binding = None if bindings is None else bindings.get(environment_name)
    if binding is None:
        return value if legacy is None else legacy
    if binding != allocation_identity_binding(reconnect_metadata):
        return legacy
    if legacy is not None and legacy != value:
        raise ValueError(
            "Environment factory allocation identity conflicts with its reconnect metadata."
        )
    return value


def _retired_allocation_fingerprint(retired: dict[str, Any]) -> str | None:
    """Return the allocation identity a disposal tombstone retired."""

    value = retired.get("allocation_fingerprint")
    if value is not None:
        return _validated_allocation_fingerprint(value)
    # Tombstones written before the identity was recorded beside them.
    return _reconnect_allocation_fingerprint(retired["reconnect_metadata"])
