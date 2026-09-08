"""Exact, private authority for a single browser-control checkpoint publication.

This is a store transaction capability, not operator authentication. Its owner
must authenticate and authorize outside the scope and perform only the matching
store publication inside it. No user callback or remote operation belongs here.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from cayu._validation import canonical_durable_json_bytes
from cayu.runtime.browser_control import BrowserControlCheckpoint, BrowserControlConflict
from cayu.runtime.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY as _KEY

BROWSER_CONTROL_OPERATION_PREFIX = "browser-control:"


def _authority_digest(value: Any) -> str:
    return sha256(canonical_durable_json_bytes(value, "browser control authority")).hexdigest()


def browser_control_receipt_key(mutation: BrowserControlCheckpointMutation) -> str:
    before = () if mutation.expected is None else mutation.expected.records
    record = next(record for record in mutation.desired.records if record not in before)
    return (
        f"{BROWSER_CONTROL_OPERATION_PREFIX}v1:"
        f"{_authority_digest(record.identity.model_dump(mode='json'))}:{record.revision}"
    )


def browser_control_receipt(mutation: BrowserControlCheckpointMutation) -> dict[str, Any]:
    return {
        "record_type": "cayu.browser-control-publication",
        "schema_version": 1,
        "source_sha256": _authority_digest(
            None if mutation.expected is None else mutation.expected.model_dump(mode="json")
        ),
        "desired_sha256": _authority_digest(mutation.desired.model_dump(mode="json")),
    }


def require_browser_control_operation_owner(key: str, record: Any = None) -> None:
    if not key.startswith(BROWSER_CONTROL_OPERATION_PREFIX):
        return
    mutation = _MUTATION.get()
    if mutation is None:
        raise BrowserControlConflict("Browser control receipts require their exact runtime owner.")
    owned = BrowserControlCheckpointMutation(
        mutation.session_id, mutation.expected, mutation.desired
    )
    if key != browser_control_receipt_key(owned):
        raise BrowserControlConflict("Browser control receipt belongs to another publication.")
    if record is not None and _authority_digest(record) != _authority_digest(
        browser_control_receipt(owned)
    ):
        raise BrowserControlConflict("Browser control receipt differs from its publication.")


@dataclass(frozen=True)
class BrowserControlCheckpointMutation:
    session_id: str
    expected: BrowserControlCheckpoint | None
    desired: BrowserControlCheckpoint

    def __post_init__(self) -> None:
        expected = (
            None
            if self.expected is None
            else BrowserControlCheckpoint.model_validate(self.expected)
        )
        desired = BrowserControlCheckpoint.model_validate(self.desired)
        if not desired.records or any(
            record.identity.session_id != self.session_id for record in desired.records
        ):
            raise BrowserControlConflict("Browser control publication belongs to another session.")
        before = BrowserControlCheckpoint() if expected is None else expected
        old = {record.identity.browser_session_id: record for record in before.records}
        changed = [
            record
            for record in desired.records
            if old.get(record.identity.browser_session_id) != record
        ]
        if len(changed) != 1:
            raise BrowserControlConflict("Browser control publication requires one exact change.")
        record = changed[0]
        reconstructed = before.replace_record(
            expected=old.get(record.identity.browser_session_id), desired=record
        )
        if reconstructed != desired:
            raise BrowserControlConflict("Browser control publication cannot discard other owners.")
        object.__setattr__(self, "expected", expected)
        object.__setattr__(self, "desired", desired)


_MUTATION: ContextVar[BrowserControlCheckpointMutation | None] = ContextVar(
    "cayu_exact_browser_control_checkpoint_mutation", default=None
)
_READ_SESSION: ContextVar[str | None] = ContextVar(
    "cayu_browser_control_read_session", default=None
)


@contextmanager
def browser_control_checkpoint_read_scope(session_id: str) -> Iterator[None]:
    """Expose private control evidence to one runtime-owned admission callback."""

    token = _READ_SESSION.set(session_id)
    try:
        yield
    finally:
        _READ_SESSION.reset(token)


@contextmanager
def browser_control_checkpoint_mutation_scope(
    mutation: BrowserControlCheckpointMutation,
) -> Iterator[None]:
    # Reconstruct without serialization, including nested post-construction edits.
    owned = BrowserControlCheckpointMutation(
        mutation.session_id, mutation.expected, mutation.desired
    )
    if _MUTATION.get() is not None:
        raise BrowserControlConflict("Browser control publication scopes cannot be nested.")
    token = _MUTATION.set(owned)
    try:
        yield
    finally:
        _MUTATION.reset(token)


def browser_control_checkpoint_visible(*, session_id: str) -> bool:
    mutation = _MUTATION.get()
    return (mutation is not None and mutation.session_id == session_id) or (
        _READ_SESSION.get() == session_id
    )


def project_browser_control_checkpoint(
    current: dict[str, Any] | None,
    replacement: dict[str, Any],
    *,
    session_id: str,
) -> dict[str, Any] | None:
    """Return the sole permitted root, checking exact state inside the store."""

    raw = None if current is None else current.get(_KEY)
    if current is not None and _KEY in current and raw is None:
        raise BrowserControlConflict("Browser control checkpoint is malformed.")
    actual = None if raw is None else BrowserControlCheckpoint.model_validate(raw)
    if actual is not None and any(
        record.identity.session_id != session_id for record in actual.records
    ):
        raise BrowserControlConflict("Browser control checkpoint belongs to another session.")
    mutation = _MUTATION.get()
    if mutation is None:
        return None if actual is None else actual.model_dump(mode="json")
    if mutation.session_id != session_id or mutation.expected != actual:
        raise BrowserControlConflict("Browser control changed before publication.")
    proposed = BrowserControlCheckpoint.model_validate(replacement.get(_KEY))
    if proposed != mutation.desired:
        raise BrowserControlConflict("Browser control publication differs from its exact command.")
    return proposed.model_dump(mode="json")
