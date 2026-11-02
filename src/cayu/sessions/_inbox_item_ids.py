"""The inbox's id for a session's pending approval, question or recovery item.

``GET /api/inbox`` lists each pause under this id, and every inbox route takes it back. It
is built from public parts only (the public session id, the pause's event sequence and its
human-attention id), so a reader of the session store can name the same item without the
server: ``cayu session show`` lists the ids of a session's pending actions this way.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Literal

from cayu.sessions.records import PendingActionKind

InboxSessionItemKind = Literal["approval", "question", "recovery"]

INBOX_ITEM_ID_PREFIX = "inbox_"
INBOX_ITEM_ID_VERSION = 1
INBOX_ITEM_KIND_CODES: dict[str, str] = {
    "approval": "a",
    "question": "q",
    "recovery": "r",
    "knowledge": "k",
}
#: The pending actions the inbox lists, by the item kind it lists them as. A delegated
#: parent is navigation to its child, which is listed itself.
PENDING_INBOX_KINDS: dict[PendingActionKind, InboxSessionItemKind] = {
    PendingActionKind.TOOL_APPROVAL: "approval",
    PendingActionKind.USER_INPUT: "question",
    PendingActionKind.MANUAL_RECOVERY: "recovery",
}


def encode_inbox_id(prefix: str, document: list[Any]) -> str:
    raw = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return prefix + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def inbox_session_item_id(
    kind: str, public_session_id: str, sequence: int, attention_id: str
) -> str:
    """The id the inbox lists one session pause under."""

    return encode_inbox_id(
        INBOX_ITEM_ID_PREFIX,
        [
            INBOX_ITEM_ID_VERSION,
            INBOX_ITEM_KIND_CODES[kind],
            public_session_id,
            sequence,
            attention_id,
        ],
    )
