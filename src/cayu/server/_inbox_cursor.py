"""Sealed, principal-bound positions for inbox scans and task keysets.

A caller an inbox access rule limits must not learn where other people's items
are, so a scan that ends among them continues from a sealed position: encrypted
and authenticated with AES-256-GCM, bound to the caller and the listing, and
valid for 15 minutes. A caller no rule limits may see every item, so its
positions stay plain keysets and need no key.

Task keysets use the same seal with a separate authenticated purpose.

The key is the deployment's ``inbox_cursor_key`` when one is configured, shared
by every worker. Without one, each server generates its own key, so an inbox
pages in every way of running a server without configuration. A sealed position
the key can't open (malformed, expired, someone else's, or, with a generated
key, from before a restart or from another worker) is refused with 422, so the
reader restarts the listing instead of silently reading items it already has.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import time
from collections.abc import Callable
from typing import Any, Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException

from cayu.knowledge.search import decode_knowledge_list_cursor
from cayu.sessions.queries import decode_session_cursor
from cayu.tasks.index import decode_task_index_cursor

SealedPurpose = Literal["knowledge", "actions", "pending-actions", "tasks"]

_LIFETIME_SECONDS = 900
_CLOCK_SKEW_SECONDS = 60
_INVALID = "Invalid or expired inbox cursor. Restart the inbox listing."
# A generated key is this server's own, so name the usual cause for operators.
_INVALID_GENERATED = (
    _INVALID + " Workers that serve one inbox need the same configured inbox_cursor_key."
)
_NO_STORE = {"Cache-Control": "private, no-store"}
# What each sealed position names: a knowledge list position, or a session-store
# position among pending actions the caller may not see.
_PURPOSES: dict[SealedPurpose, tuple[str, Callable[[str], object]]] = {
    "tasks": ("cayu.tasks.index.v1", decode_task_index_cursor),
    "knowledge": ("cayu.inbox.knowledge.v1", decode_knowledge_list_cursor),
    "actions": ("cayu.inbox.actions.v1", decode_session_cursor),
    "pending-actions": ("cayu.pending-actions.v1", decode_session_cursor),
}


class InboxCursorSeal:
    """Seals scan positions with the deployment's key, or a key of this server's own."""

    def __init__(self, key: bytes | None) -> None:
        if key is not None and (type(key) is not bytes or len(key) != 32):
            raise ValueError("inbox_cursor_key must contain exactly 32 bytes.")
        self.configured = key is not None
        self._cipher = AESGCM(os.urandom(32) if key is None else key)

    @staticmethod
    def _aad(purpose: SealedPurpose, binding: Any) -> bytes:
        return json.dumps(
            [_PURPOSES[purpose][0], binding],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def seal(self, purpose: SealedPurpose, position: str, binding: Any) -> str:
        nonce = os.urandom(12)
        payload = json.dumps([int(time.time()), position], separators=(",", ":")).encode("utf-8")
        encrypted = self._cipher.encrypt(nonce, payload, self._aad(purpose, binding))
        return base64.urlsafe_b64encode(nonce + encrypted).decode("ascii").rstrip("=")

    def open(self, purpose: SealedPurpose, value: str, binding: Any) -> str:
        """The sealed position; a cursor this key can't open is refused with 422."""

        try:
            if len(value) > 6144:
                raise ValueError
            raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
            payload = json.loads(
                self._cipher.decrypt(raw[:12], raw[12:], self._aad(purpose, binding))
            )
            if type(payload) is not list or len(payload) != 2:
                raise ValueError
            issued, position = payload
            now = time.time()
            # Replicas share a key, but their clocks may differ slightly. Keep
            # expiry strict while allowing a bounded future issuance timestamp.
            if (
                type(issued) is not int
                or not -_CLOCK_SKEW_SECONDS <= now - issued <= _LIFETIME_SECONDS
            ):
                raise ValueError
            _PURPOSES[purpose][1](position)
            return position
        except (ValueError, TypeError, binascii.Error, InvalidTag):
            detail = _INVALID if self.configured else _INVALID_GENERATED
            if purpose == "tasks":
                detail = "Invalid or expired task cursor. Restart the task listing."
            raise HTTPException(status_code=422, detail=detail, headers=_NO_STORE) from None
