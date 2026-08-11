from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank

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
    step_id: object,
    item_key: object,
    *,
    kind: object,
) -> str | None:
    """Map one canonical unversioned v1 identity to its v2 equivalent."""

    if type(kind) is not str or kind != "gated_loop":
        return None
    if type(step_id) is not str or type(item_key) is not str:
        return None
    try:
        item_key = require_durable_clean_nonblank(item_key, "legacy gated_loop item_key")
    except ValueError:
        return None
    if not step_id.startswith(GATED_LOOP_STEP_ID_PREFIX):
        return None
    source = step_id[len(GATED_LOOP_STEP_ID_PREFIX) :]
    item_suffix = f":{item_key}"
    if not source.endswith(item_suffix):
        return None
    loop_name = source[: -len(item_suffix)]
    try:
        loop_name = require_durable_clean_nonblank(loop_name, "legacy gated_loop name")
    except ValueError:
        return None
    return gated_loop_step_id(loop_name, item_key)


def validated_modern_gated_loop_step_id(
    step_id: object,
    item_key: object,
    *,
    kind: object,
    loop_name: object,
    step_id_version: object,
) -> str | None:
    """Return one coherent current gated-loop identity, or fail closed."""

    if type(step_id_version) is not int or step_id_version != GATED_LOOP_STEP_ID_VERSION:
        return None
    if type(kind) is not str or kind != "gated_loop":
        return None
    if type(step_id) is not str or type(loop_name) is not str or type(item_key) is not str:
        return None
    try:
        loop_name = require_durable_clean_nonblank(loop_name, "gated_loop name")
        item_key = require_durable_clean_nonblank(item_key, "gated_loop item_key")
    except ValueError:
        return None
    expected_step_id = gated_loop_step_id(loop_name, item_key)
    if step_id != expected_step_id:
        return None
    return step_id


def replay_eligible_completed_step_id(payload: Mapping[str, object]) -> str | None:
    """Return the canonical identity one completed event may satisfy on replay."""

    step_id = payload.get("step_id")
    if type(step_id) is not str or not step_id:
        return None
    if not step_id.startswith(GATED_LOOP_STEP_ID_PREFIX):
        return step_id
    item_key = payload.get("item_key")
    if "step_id_version" in payload:
        return validated_modern_gated_loop_step_id(
            step_id,
            item_key,
            kind=payload.get("kind"),
            loop_name=payload.get("loop_name"),
            step_id_version=payload.get("step_id_version"),
        )
    return upgraded_legacy_gated_loop_step_id(
        step_id,
        item_key,
        kind=payload.get("kind"),
    )
