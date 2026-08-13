from __future__ import annotations

from hashlib import sha256
from uuid import UUID

from cayu.runtime.invocation import (
    InvocationOrigin,
    InvocationOriginTrust,
    SessionExecutionSource,
    SessionInvocation,
)
from cayu.runtime.sessions import Session


def unattributed_session_invocation(
    root_session_id: str,
    *,
    source: SessionExecutionSource = SessionExecutionSource.SDK_RUN,
) -> SessionInvocation:
    """Build explicit provenance for isolated Session value-object fixtures."""

    return SessionInvocation(
        origin=InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED),
        root_invocation_id=_fixture_root_invocation_id(root_session_id),
        root_session_id=root_session_id,
        source=source,
    )


def _fixture_root_invocation_id(root_session_id: str) -> str:
    """Build a deterministic RFC 4122 UUIDv4-shaped fixture identity."""

    return str(UUID(bytes=sha256(root_session_id.encode()).digest()[:16], version=4))


def fixture_session_invocation(
    session_id: str,
    *,
    parent_session_id: str | None = None,
) -> SessionInvocation:
    """Build internally consistent provenance for a standalone Session fixture."""

    return unattributed_session_invocation(
        parent_session_id or session_id,
        source=(
            SessionExecutionSource.SUBAGENT
            if parent_session_id is not None
            else SessionExecutionSource.SDK_RUN
        ),
    )


def session_fixture(**values: object) -> Session:
    """Construct an isolated Session fixture with explicit invocation provenance."""

    session_id = values.get("id")
    if type(session_id) is not str:
        raise TypeError("Session fixtures must provide an explicit string id.")
    parent_session_id = values.get("parent_session_id")
    if parent_session_id is not None and type(parent_session_id) is not str:
        raise TypeError("Session fixture parent_session_id must be a string or None.")
    values.setdefault(
        "invocation",
        fixture_session_invocation(
            session_id,
            parent_session_id=parent_session_id,
        ),
    )
    return Session.model_validate(values)
