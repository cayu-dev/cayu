"""Content-free incarnation identities and immutable ancestry for reporting."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from hashlib import sha256
from typing import TYPE_CHECKING, Any, TypedDict

from cayu._validation import canonical_durable_json_bytes

if TYPE_CHECKING:
    from cayu.sessions.records import Session

RUN_LINEAGE_METADATA_KEY = "cayu:run_lineage"
#: Runtime-owned session metadata, ``True`` on every session created while the eval
#: runner runs a trial and on every session created under one. Callers cannot set it.
EVAL_TRIAL_METADATA_KEY = "cayu:eval_trial"
#: Session metadata that, set to ``False``, keeps the session and the sessions created
#: under it from the human-attention receiver.
ATTENTION_HANDOFF_METADATA_KEY = "cayu:attention_handoff"
_EVAL_TRIAL: ContextVar[bool] = ContextVar("cayu_eval_trial", default=False)


class RunLineageMetadata(TypedDict):
    is_root: bool
    root_run_id: str | None
    parent_run_id: str | None
    parent_session_id: str | None


def validated_run_lineage_metadata(metadata: dict[str, Any]) -> RunLineageMetadata | None:
    """Validate the closed runtime record before treating any part as protocol."""

    from cayu.sessions.records import _require_bounded_session_id

    if RUN_LINEAGE_METADATA_KEY not in metadata:
        return None
    raw = metadata[RUN_LINEAGE_METADATA_KEY]
    if (
        type(raw) is not dict
        or set(raw) != {"is_root", "root_run_id", "parent_run_id", "parent_session_id"}
        or type(raw["is_root"]) is not bool
    ):
        raise ValueError("Invalid runtime run-lineage metadata.")
    root, parent, parent_session = (
        raw["root_run_id"],
        raw["parent_run_id"],
        raw["parent_session_id"],
    )
    for identity in (root, parent):
        if identity is not None and (
            type(identity) is not str
            or len(identity) != 68
            or not identity.startswith("run_")
            or any(char not in "0123456789abcdef" for char in identity[4:])
        ):
            raise ValueError("Invalid runtime run-lineage identity.")
    if raw["is_root"]:
        if root is not None or parent is not None or parent_session is not None:
            raise ValueError("A root run cannot retain parent lineage.")
    else:
        if parent is None or type(parent_session) is not str:
            raise ValueError("A child run requires retained parent lineage.")
        _require_bounded_session_id(parent_session, "run-lineage parent session")
    return {
        "is_root": raw["is_root"],
        "root_run_id": root,
        "parent_run_id": parent,
        "parent_session_id": parent_session,
    }


def session_run_id(session: Session) -> str:
    """Name an incarnation without exposing its private instance UUID."""

    return (
        "run_"
        + sha256(
            canonical_durable_json_bytes([session.id, session.instance_id], "run identity")
        ).hexdigest()
    )


def session_run_lineage(session: Session) -> tuple[str | None, str | None]:
    """Return original root and parent run identities, never reusable session IDs.

    Older rows have no retained proof of their ancestors' incarnations, or
    whether a missing parent was deleted. Their links stay unknown.
    """

    marker = session.metadata.get(RUN_LINEAGE_METADATA_KEY)
    if isinstance(marker, dict):
        if marker.get("is_root") is True:
            return session_run_id(session), None
        return marker["root_run_id"], marker["parent_run_id"]
    return None, None


@asynccontextmanager
async def eval_trial_scope() -> AsyncIterator[None]:
    """Mark every session created inside the scope as an eval trial; the eval runner's own."""

    token = _EVAL_TRIAL.set(True)
    try:
        yield
    finally:
        _EVAL_TRIAL.reset(token)


def is_eval_trial(session: Session) -> bool:
    """Whether the session is an eval trial or was created under one, as a child or fork."""

    return session.metadata.get(EVAL_TRIAL_METADATA_KEY) is True


def attention_withheld(session: Session) -> bool:
    """Whether the session, or a session it was created under, opted out or is an eval."""

    return session.metadata.get(ATTENTION_HANDOFF_METADATA_KEY) is False or is_eval_trial(session)


def run_marks_for_creation(parent: Session | None) -> dict[str, bool]:
    """The runtime marks a new session takes from its parent and the eval runner.

    They are copied at the creation boundary so that deleting or replacing an
    ancestor later can never change them.
    """

    marks: dict[str, bool] = {}
    if _EVAL_TRIAL.get() or (parent is not None and is_eval_trial(parent)):
        marks[EVAL_TRIAL_METADATA_KEY] = True
    if parent is not None and parent.metadata.get(ATTENTION_HANDOFF_METADATA_KEY) is False:
        marks[ATTENTION_HANDOFF_METADATA_KEY] = False
    return marks


def run_lineage_for_creation(parent: Session | None) -> dict[str, str | bool | None]:
    """Capture ancestry at the creation boundary while the parent is fenced."""

    root, _ = (None, None) if parent is None else session_run_lineage(parent)
    return {
        "is_root": parent is None,
        "root_run_id": root,
        "parent_run_id": None if parent is None else session_run_id(parent),
        "parent_session_id": None if parent is None else parent.id,
    }
