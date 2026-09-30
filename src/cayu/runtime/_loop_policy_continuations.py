from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from cayu.messages import Message, MessageRole

if TYPE_CHECKING:
    from cayu.runtime.loop_policies import LoopPolicy
    from cayu.sessions.base import Session, TranscriptSnapshot


BEFORE_STOP_CONTINUATIONS_CHECKPOINT_KEY = "before_stop_continuations"


def before_stop_policy_key(policy: LoopPolicy, *, scope: str, position: int) -> str:
    identity = policy.execution_profile_identity
    material = [
        scope,
        position,
        policy.name,
        None if identity is None else identity.model_dump(mode="json"),
    ]
    return sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


def _message_digest(message: Message) -> str:
    material = json.dumps(message.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return sha256(material.encode("utf-8")).hexdigest()


def _continuation_anchors(
    checkpoint: dict[str, Any] | None,
    *,
    session_id: str,
    policy_key: str,
) -> dict[int, str]:
    marker = (checkpoint or {}).get(BEFORE_STOP_CONTINUATIONS_CHECKPOINT_KEY)
    if marker is None:
        return {}
    if type(marker) is not dict or marker.get("version") != 1:
        raise ValueError("Invalid before-stop continuation provenance.")
    # Copied checkpoints must not grant a fork ownership of the source's reminders.
    if (
        marker.get("session_sha256") != sha256(session_id.encode("utf-8")).hexdigest()
        or marker.get("policy_sha256") != policy_key
    ):
        return {}
    anchors = marker.get("runtime_authored_anchors")
    if type(anchors) is not list:
        raise ValueError("Invalid before-stop continuation anchors.")
    result: dict[int, str] = {}
    for anchor in anchors:
        if type(anchor) is not dict:
            raise ValueError("Invalid before-stop continuation anchor.")
        index, digest = anchor.get("anchor_transcript_index"), anchor.get("user_message_sha256")
        if type(index) is not int or index < 0 or type(digest) is not str or index in result:
            raise ValueError("Invalid before-stop continuation anchor.")
        result[index] = digest
    return result


def before_stop_continuation_indices(
    checkpoint: dict[str, Any] | None,
    *,
    snapshot: TranscriptSnapshot,
    session_id: str,
    policy_key: str,
) -> frozenset[int]:
    """Map durable absolute anchors to positions in the supplied retained messages."""

    anchors = _continuation_anchors(checkpoint, session_id=session_id, policy_key=policy_key)
    return frozenset(
        position
        for position, record in enumerate(snapshot.records)
        if record.index in anchors
        and record.message.role == MessageRole.USER
        and anchors.get(record.index) == _message_digest(record.message)
    )


def before_stop_continuation_checkpoint_transform(
    *,
    snapshot: TranscriptSnapshot,
    message: Message,
    policy_key: str,
    checkpoint_transform: Callable[[Session, dict[str, Any] | None], dict[str, Any]],
) -> Callable[[Session, dict[str, Any] | None], dict[str, Any]]:
    """Receipt a selected continuation in the same write as its transcript append.

    Retain only this policy's consecutive runtime-authored user messages. A real
    user message or another policy's continuation begins a new scan boundary.
    """

    anchor = {
        "anchor_transcript_index": snapshot.cursor,
        "user_message_sha256": _message_digest(message),
    }

    def transform(session: Session, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
        checkpoint = checkpoint_transform(session, checkpoint)
        owned = before_stop_continuation_indices(
            checkpoint, snapshot=snapshot, session_id=session.id, policy_key=policy_key
        )
        previous: list[dict[str, Any]] = []
        for position in range(len(snapshot.records) - 1, -1, -1):
            record = snapshot.records[position]
            if record.message.role != MessageRole.USER:
                continue
            if position not in owned:
                break
            previous.append(
                {
                    "anchor_transcript_index": record.index,
                    "user_message_sha256": _message_digest(record.message),
                }
            )
        updated = dict(checkpoint or {})
        updated[BEFORE_STOP_CONTINUATIONS_CHECKPOINT_KEY] = {
            "version": 1,
            "session_sha256": sha256(session.id.encode("utf-8")).hexdigest(),
            "policy_sha256": policy_key,
            "runtime_authored_anchors": [*reversed(previous), anchor],
        }
        return updated

    return transform
