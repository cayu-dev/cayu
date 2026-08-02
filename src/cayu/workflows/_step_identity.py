from __future__ import annotations

from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes

GATED_LOOP_STEP_ID_PREFIX = "gated-loop:"
GATED_LOOP_STEP_ID_VERSION = 2
_GATED_LOOP_V2_PREFIX = f"{GATED_LOOP_STEP_ID_PREFIX}v{GATED_LOOP_STEP_ID_VERSION}:"


def gated_loop_step_id(loop_name: str, item_key: str) -> str:
    """Return the versioned collision-resistant identity for one loop item."""

    source = canonical_durable_json_bytes(
        [loop_name, item_key],
        "gated_loop step identity",
    )
    return f"{_GATED_LOOP_V2_PREFIX}{sha256(source).hexdigest()}"


def upgraded_legacy_gated_loop_step_id(
    step_id: str,
    item_key: str,
) -> str | None:
    """Map an unversioned v1 delimiter ID using its journaled item key."""

    if not step_id.startswith(GATED_LOOP_STEP_ID_PREFIX):
        return None
    source = step_id[len(GATED_LOOP_STEP_ID_PREFIX) :]
    item_suffix = f":{item_key}"
    if not source.endswith(item_suffix):
        return None
    loop_name = source[: -len(item_suffix)]
    if not loop_name:
        return None
    return gated_loop_step_id(loop_name, item_key)
