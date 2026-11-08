"""Where SQL stores keep a session's invocation lifecycle receipt ledger.

The ledger is part of the logical checkpoint, under
``INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY``, and every checkpoint a store
hands out carries it whole. The SQL stores keep it out of the checkpoint row:
the row holds a reference that names the ledger's content digest, and the
ledger itself sits in its own row, written only when it changes.

So a write that leaves the ledger as it was (every runtime publication and
generic checkpoint transform) neither validates nor rewrites it, and the
checkpoint row it decodes and checks stays small. A reference in a checkpoint
that was not joined with its ledger fails closed: every ledger reader requires
the complete ledger record type, so it is never mistaken for an empty ledger.
"""

from __future__ import annotations

from typing import Any

from cayu._validation import DURABLE_DOCUMENT_LIMITS, DurableValueError, measure_durable_json
from cayu.sessions.checkpoints import (
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_LIFECYCLE_RECEIPT_LEDGER_RECORD_TYPE,
)

LEDGER_REFERENCE_RECORD_TYPE = "cayu.invocation-lifecycle-command-receipt-ledger-reference"
_LEDGER_REFERENCE_SCHEMA_VERSION = 1
_LEDGER_REFERENCE_KEYS = frozenset(
    {"record_type", "schema_version", "record_sha256", "ledger_bytes", "ledger_nodes"}
)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def ledger_reference(record_sha256: str, ledger: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint row's stand-in for a ledger stored in its own row.

    It records the ledger's size and node count as the document limits count
    them, so a row writer can check the logical checkpoint against those
    limits without reading the ledger.
    """

    if not _is_sha256(record_sha256):
        raise ValueError("A receipt ledger reference needs the ledger's SHA-256 digest.")
    ledger_bytes, ledger_nodes = measure_durable_json(ledger, "invocation_lifecycle_receipt")
    return {
        "record_type": LEDGER_REFERENCE_RECORD_TYPE,
        "schema_version": _LEDGER_REFERENCE_SCHEMA_VERSION,
        "record_sha256": record_sha256,
        "ledger_bytes": ledger_bytes,
        "ledger_nodes": ledger_nodes,
    }


def _ledger_reference(checkpoint: dict[str, Any] | None) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    value = checkpoint.get(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY)
    if type(value) is not dict or value.get("record_type") != LEDGER_REFERENCE_RECORD_TYPE:
        return None
    if (
        set(value) != _LEDGER_REFERENCE_KEYS
        or value["schema_version"] != _LEDGER_REFERENCE_SCHEMA_VERSION
        or not _is_sha256(value["record_sha256"])
        or any(
            type(value[key]) is not int or value[key] < 1
            for key in ("ledger_bytes", "ledger_nodes")
        )
    ):
        raise RuntimeError("Durable invocation lifecycle receipt ledger reference is malformed.")
    return value


def ledger_reference_digest(checkpoint: dict[str, Any] | None) -> str | None:
    """The ledger digest a checkpoint row refers to, or ``None`` if it holds no reference."""

    reference = _ledger_reference(checkpoint)
    return None if reference is None else reference["record_sha256"]


def require_logical_checkpoint_limits(row: dict[str, Any], serialized: str) -> None:
    """Check the logical checkpoint a row stands for against the document limits.

    A row that refers to its ledger is checked as if the ledger were in it, so
    no write can store a checkpoint that a read could not return whole.
    ``serialized`` is the row's compact JSON. Its length bounds the row's size
    and node count from above, so the row is walked only when that bound
    alone could exceed a limit.
    """

    reference = _ledger_reference(row)
    if reference is None:
        return
    upper = len(serialized.encode("utf-8"))
    if (
        upper + reference["ledger_bytes"] <= DURABLE_DOCUMENT_LIMITS.max_bytes // 2
        and upper + reference["ledger_nodes"] <= DURABLE_DOCUMENT_LIMITS.max_nodes
    ):
        return
    row_bytes, row_nodes = measure_durable_json(row, "checkpoint")
    reference_bytes, reference_nodes = measure_durable_json(reference, "checkpoint")
    logical_bytes = row_bytes - reference_bytes + reference["ledger_bytes"]
    logical_nodes = row_nodes - reference_nodes + reference["ledger_nodes"]
    if logical_bytes > DURABLE_DOCUMENT_LIMITS.max_bytes:
        raise DurableValueError(
            "json_value_too_large",
            "checkpoint",
            limit=DURABLE_DOCUMENT_LIMITS.max_bytes,
            observed_lower_bound=logical_bytes,
        )
    if logical_nodes > DURABLE_DOCUMENT_LIMITS.max_nodes:
        raise DurableValueError(
            "too_many_json_nodes",
            "checkpoint",
            limit=DURABLE_DOCUMENT_LIMITS.max_nodes,
            observed_lower_bound=logical_nodes,
        )


def split_receipt_ledger(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The checkpoint row to store and, if the checkpoint holds one, its ledger.

    A ledger is replaced by its reference; the checkpoint itself is not
    modified. A checkpoint that already holds a reference, or no ledger, is
    returned as it is.
    """

    value = checkpoint.get(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY)
    if type(value) is not dict or value.get("record_type") != (
        INVOCATION_LIFECYCLE_RECEIPT_LEDGER_RECORD_TYPE
    ):
        return checkpoint, None
    from cayu.sessions._invocation_lifecycle import _shared_receipt_ledger_from_checkpoint

    # The ledger is checked whole before it is stored on its own. This reuses
    # the validated ledger an unchanged write already read.
    ledger = _shared_receipt_ledger_from_checkpoint(checkpoint)
    row = dict(checkpoint)
    row[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = ledger_reference(ledger.record_sha256, value)
    return row, value


def join_receipt_ledger(
    checkpoint: dict[str, Any],
    stored_digest: str | None,
    ledger: dict[str, Any] | None,
) -> dict[str, Any]:
    """The logical checkpoint: the row with the ledger it refers to put back.

    ``checkpoint`` is the decoded row, ``stored_digest`` and ``ledger`` the
    session's ledger row, if it has one. ``checkpoint`` is modified in place
    and must be a private copy.
    """

    digest = ledger_reference_digest(checkpoint)
    if digest is None:
        return checkpoint
    if ledger is None or stored_digest != digest or ledger.get("record_sha256") != digest:
        raise RuntimeError("Durable invocation lifecycle receipt ledger is missing or stale.")
    checkpoint[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = ledger
    return checkpoint
