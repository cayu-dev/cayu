"""Shared durable-pin owner validation for built-in artifact stores."""

from __future__ import annotations

import hashlib

from cayu._validation import require_clean_nonblank, require_unicode_scalar_text

MAX_PIN_OWNER_BYTES = 1024


def pin_owner_digest(owner: str) -> str:
    """Validate a pin owner and return its stable non-secret storage identity."""

    owner = require_unicode_scalar_text(require_clean_nonblank(owner, "pin.owner"), "pin.owner")
    encoded = owner.encode("utf-8")
    if len(encoded) > MAX_PIN_OWNER_BYTES:
        raise ValueError("Artifact pin owner is too long.")
    return hashlib.sha256(encoded).hexdigest()
