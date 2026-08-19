from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank

if TYPE_CHECKING:
    from cayu.core.tools import ToolResult
    from cayu.runtime.invocation import SessionInvocation
    from cayu.runtime.sessions import Session


class ChildSessionRecoveryMatcher(ABC):
    """Explicit registration capability for authenticating recovered children."""

    @abstractmethod
    def matches_recoverable_child(
        self,
        child: Session,
        *,
        parent_invocation: SessionInvocation,
        parent_session_id: str,
        causal_budget_id: str,
        environment_name: str | None,
        tool_call_id: str,
        idempotency_key: str,
        arguments: dict[str, Any],
        require_fingerprint: bool,
    ) -> bool:
        """Return exact authority evidence for one recovered child candidate."""

    def matches_recoverable_submission(
        self,
        *,
        parent_session: Session,
        tool_name: str,
        tool_round_id: str,
        tool_call_id: str,
        idempotency_key: str,
        arguments: dict[str, Any],
        spawn_fingerprint: str,
    ) -> bool:
        """Validate marker-backed effective arguments before creating a missing child."""

        del (
            parent_session,
            tool_name,
            tool_round_id,
            tool_call_id,
            idempotency_key,
            arguments,
            spawn_fingerprint,
        )
        return False

    async def reconcile_recoverable_child(
        self,
        child: Session | None,
        *,
        parent_session: Session,
        tool_name: str,
        tool_round_id: str,
        tool_call_id: str,
        idempotency_key: str,
        arguments: dict[str, Any],
    ) -> Session | ToolResult | None:
        """Optionally repair a durable child or settle its rejected submission."""

        del parent_session, tool_name, tool_round_id, tool_call_id, idempotency_key, arguments
        return child


class ChildSessionKind(StrEnum):
    """Runtime-owned namespaces for generated child-session identities."""

    SUBAGENT = "subagent"
    WORKFLOW_STEP = "workflow-step"


def child_session_id_prefix(kind: ChildSessionKind) -> str:
    if type(kind) is not ChildSessionKind:
        raise TypeError("kind must be a ChildSessionKind.")
    return f"cayu-child:v1:{kind.value}:"


def generate_child_session_id(
    *,
    kind: ChildSessionKind,
    parent_session_id: str,
    logical_spawn_id: str | None = None,
) -> str:
    """Return one opaque, collision-safe identity for a runtime-created child.

    Idempotent spawn paths supply their durable logical identity and receive the
    same parent-scoped child identity after retry or reconstruction. Other paths
    retain the complete UUID4 identity rather than a shortened display suffix.
    """

    if type(kind) is not ChildSessionKind:
        raise TypeError("kind must be a ChildSessionKind.")
    parent_session_id = require_durable_clean_nonblank(
        parent_session_id,
        "parent_session_id",
    )
    if logical_spawn_id is None:
        identity = uuid4().hex
    else:
        logical_spawn_id = require_durable_clean_nonblank(
            logical_spawn_id,
            "logical_spawn_id",
        )
        identity = sha256(
            canonical_durable_json_bytes(
                [
                    "cayu-child-session-v1",
                    kind.value,
                    parent_session_id,
                    logical_spawn_id,
                ],
                "child_session_identity",
            )
        ).hexdigest()
    return child_session_id_prefix(kind) + identity
