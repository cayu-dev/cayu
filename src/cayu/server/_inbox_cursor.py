"""Private, principal-bound positions for bounded knowledge inbox scans."""

from __future__ import annotations

import base64
import binascii
import json
import os
import time
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException

from cayu.knowledge.search import decode_knowledge_list_cursor

_LIFETIME_SECONDS = 900
_CLOCK_SKEW_SECONDS = 60
_INVALID = "Invalid or expired inbox cursor. Restart the inbox listing."
_NO_STORE = {"Cache-Control": "private, no-store"}


class KnowledgeInboxCursor:
    """An install-owned key is required: cursors must work across server workers."""

    def __init__(self, key: bytes | None) -> None:
        if key is not None and (type(key) is not bytes or len(key) != 32):
            raise ValueError("inbox_cursor_key must contain exactly 32 bytes.")
        self._cipher = None if key is None else AESGCM(key)

    @staticmethod
    def _aad(binding: Any) -> bytes:
        return json.dumps(
            ["cayu.inbox.knowledge.v1", binding],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def encode(self, position: str, binding: Any) -> str:
        if self._cipher is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Knowledge inbox pagination requires a configured inbox_cursor_key "
                    "shared by this deployment's server workers."
                ),
                headers=_NO_STORE,
            )
        nonce = os.urandom(12)
        payload = json.dumps([int(time.time()), position], separators=(",", ":")).encode("utf-8")
        encrypted = self._cipher.encrypt(nonce, payload, self._aad(binding))
        return base64.urlsafe_b64encode(nonce + encrypted).decode("ascii").rstrip("=")

    def decode(self, value: str, binding: Any) -> str:
        try:
            if self._cipher is None or len(value) > 6144:
                raise ValueError
            raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
            payload = json.loads(self._cipher.decrypt(raw[:12], raw[12:], self._aad(binding)))
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
            decode_knowledge_list_cursor(position)
            return position
        except (ValueError, TypeError, binascii.Error, InvalidTag):
            raise HTTPException(status_code=422, detail=_INVALID, headers=_NO_STORE) from None
