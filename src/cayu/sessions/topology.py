"""Bounded session topology contracts, cursors and result construction."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from itertools import islice

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.build_provenance import RuntimeBuildProvenance, legacy_runtime_build_provenance
from cayu.sessions.records import Session, SessionStatus

SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH = 32
SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS = 50
SESSION_TOPOLOGY_DEFAULT_CHILD_LIMIT = 25
SESSION_TOPOLOGY_MAX_CHILD_LIMIT = 100
SESSION_TOPOLOGY_MAX_NODES = 500
SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES = 1024
SESSION_TOPOLOGY_MAX_CURSOR_BYTES = 4096


class SessionTopologyDepthExceeded(ValueError):
    """A focus session has more ancestors than the bounded topology contract."""


class SessionTopologyCycle(ValueError):
    """Durable parent-session records contain a lineage cycle."""


def _bounded_session_topology_text(
    value: str,
    field_name: str,
    *,
    max_bytes: int,
) -> str:
    value = require_clean_nonblank(value, field_name)
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must contain portable Unicode text.") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{field_name} exceeds the session topology byte limit.")
    return value


class SessionTopologyNode(BaseModel):
    """Bounded metadata-free identity for one session topology node."""

    model_config = ConfigDict(extra="forbid")

    id: str
    agent_name: str
    provider_name: str
    model: str
    parent_session_id: str | None
    causal_budget_id: str
    runtime_name: str
    runtime_version: str | None
    runtime_build_provenance: RuntimeBuildProvenance = Field(
        default_factory=legacy_runtime_build_provenance
    )
    environment_name: str | None
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime

    @classmethod
    def from_session(cls, session: Session) -> SessionTopologyNode:
        if type(session) is not Session:
            raise TypeError("Session topology nodes require Session instances.")
        return cls(
            id=session.id,
            agent_name=session.agent_name,
            provider_name=session.provider_name,
            model=session.model,
            parent_session_id=session.parent_session_id,
            causal_budget_id=session.causal_budget_id,
            runtime_name=session.runtime_name,
            runtime_version=session.runtime_version,
            runtime_build_provenance=session.runtime_build_provenance,
            environment_name=session.environment_name,
            status=session.status,
            created_at=session.created_at,
            updated_at=session.updated_at,
            last_activity_at=session.last_activity_at,
        )

    @field_validator(
        "id",
        "agent_name",
        "provider_name",
        "model",
        "causal_budget_id",
        "runtime_name",
    )
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("id", "causal_budget_id")
    @classmethod
    def validate_bounded_ids(cls, value: str, info) -> str:
        return _bounded_session_topology_text(
            value,
            info.field_name,
            max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @field_validator("parent_session_id")
    @classmethod
    def validate_parent_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_session_topology_text(
            value,
            "parent_session_id",
            max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @field_validator("runtime_version", "environment_name")
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("created_at", "updated_at", "last_activity_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)


class SessionTopologyQuery(BaseModel):
    """One bounded ancestry plus batched child-branch read."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    focus_session_id: str
    expanded_parent_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    child_cursors: dict[str, str] = Field(
        default_factory=dict,
        max_length=SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    ancestor_depth_limit: StrictInt = Field(
        default=SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH,
        ge=1,
        le=SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH,
    )
    child_limit: StrictInt = Field(
        default=SESSION_TOPOLOGY_DEFAULT_CHILD_LIMIT,
        ge=1,
        le=SESSION_TOPOLOGY_MAX_CHILD_LIMIT,
    )

    @field_validator("focus_session_id")
    @classmethod
    def validate_focus_session_id(cls, value: str) -> str:
        return _bounded_session_topology_text(
            value,
            "focus_session_id",
            max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @field_validator("expanded_parent_ids", mode="before")
    @classmethod
    def copy_expanded_parent_ids(cls, value) -> tuple[str, ...]:
        if value is None:
            return ()
        if type(value) is str:
            raise ValueError("expanded_parent_ids must be a sequence of strings.")
        copied: list[str] = []
        for index, item in enumerate(tuple(value)):
            parent_id = _bounded_session_topology_text(
                item,
                f"expanded_parent_ids[{index}]",
                max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
            )
            if parent_id in copied:
                raise ValueError("expanded_parent_ids must not contain duplicates.")
            copied.append(parent_id)
        return tuple(copied)

    @field_validator("child_cursors", mode="before")
    @classmethod
    def copy_child_cursors(cls, value) -> dict[str, str]:
        if value is None:
            return {}
        if type(value) is not dict:
            raise ValueError("child_cursors must be an object.")
        copied: dict[str, str] = {}
        for raw_parent_id, raw_cursor in value.items():
            parent_id = _bounded_session_topology_text(
                raw_parent_id,
                "child_cursors key",
                max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
            )
            copied[parent_id] = _bounded_session_topology_text(
                raw_cursor,
                f"child_cursors[{parent_id!r}]",
                max_bytes=SESSION_TOPOLOGY_MAX_CURSOR_BYTES,
            )
        return copied

    @model_validator(mode="after")
    def validate_cursor_parents(self) -> SessionTopologyQuery:
        unknown = set(self.child_cursors).difference(self.expanded_parent_ids)
        if unknown:
            raise ValueError("child_cursors keys must also appear in expanded_parent_ids.")
        return self


class SessionTopologyBranch(BaseModel):
    """One independently pageable direct-child branch."""

    model_config = ConfigDict(extra="forbid")

    parent_session_id: str
    children: tuple[SessionTopologyNode, ...] = Field(
        default=(),
        max_length=SESSION_TOPOLOGY_MAX_CHILD_LIMIT,
    )
    next_cursor: str | None = None
    has_more: StrictBool = False

    @field_validator("parent_session_id")
    @classmethod
    def validate_parent_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "parent_session_id")

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "next_cursor")

    @model_validator(mode="after")
    def validate_continuation(self) -> SessionTopologyBranch:
        if self.has_more and self.next_cursor is None:
            raise ValueError("A topology branch with more children requires a cursor.")
        if not self.has_more and self.next_cursor is not None:
            raise ValueError("A complete topology branch cannot expose a cursor.")
        return self


class SessionTopologyStoreResult(BaseModel):
    """Backend-neutral bounded topology projection."""

    model_config = ConfigDict(extra="forbid")

    focus: SessionTopologyNode
    ancestors: tuple[SessionTopologyNode, ...] = Field(
        default=(),
        max_length=SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH,
    )
    expanded_parents: tuple[SessionTopologyNode, ...] = Field(
        default=(),
        max_length=SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    branches: tuple[SessionTopologyBranch, ...] = Field(
        default=(),
        max_length=SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )

    @model_validator(mode="after")
    def validate_shape(self) -> SessionTopologyStoreResult:
        ancestor_ids = [node.id for node in self.ancestors]
        if len(set(ancestor_ids)) != len(ancestor_ids) or self.focus.id in ancestor_ids:
            raise ValueError("Topology ancestors must form an acyclic path to the focus.")
        if self.ancestors:
            if self.ancestors[0].parent_session_id is not None:
                raise ValueError("Topology ancestry must start at the durable root session.")
            for parent, child in zip(self.ancestors, self.ancestors[1:], strict=False):
                if child.parent_session_id != parent.id:
                    raise ValueError("Topology ancestors contain a contradictory parent edge.")
            if self.focus.parent_session_id != self.ancestors[-1].id:
                raise ValueError("Topology ancestry does not terminate at the focus parent.")
        elif self.focus.parent_session_id is not None:
            raise ValueError("Topology omitted the focus session's durable ancestors.")
        expanded_ids = [node.id for node in self.expanded_parents]
        if len(set(expanded_ids)) != len(expanded_ids):
            raise ValueError("Topology expanded parents must not contain duplicates.")
        if len(self.branches) != len(self.expanded_parents):
            raise ValueError("Every expanded parent requires exactly one branch.")
        if [branch.parent_session_id for branch in self.branches] != [
            parent.id for parent in self.expanded_parents
        ]:
            raise ValueError("Topology branches must preserve expanded-parent order.")
        for branch in self.branches:
            child_ids = [child.id for child in branch.children]
            if len(set(child_ids)) != len(child_ids):
                raise ValueError("A topology branch must not repeat a child session.")
            if any(
                child.parent_session_id != branch.parent_session_id for child in branch.children
            ):
                raise ValueError("A topology branch contains a contradictory parent edge.")
            if list(branch.children) != sorted(
                branch.children,
                key=lambda child: (child.created_at, child.id),
            ):
                raise ValueError("Topology branch children must use stable creation ordering.")
            if branch.next_cursor is not None:
                cursor_created_at, cursor_id = decode_session_topology_cursor(
                    branch.next_cursor,
                    parent_session_id=branch.parent_session_id,
                )
                if not branch.children or (
                    cursor_created_at,
                    cursor_id,
                ) != (
                    branch.children[-1].created_at,
                    branch.children[-1].id,
                ):
                    raise ValueError(
                        "A topology continuation cursor must identify the last returned child."
                    )
        all_nodes = (
            self.focus,
            *self.ancestors,
            *self.expanded_parents,
            *(child for branch in self.branches for child in branch.children),
        )
        nodes_by_id: dict[str, SessionTopologyNode] = {}
        for node in all_nodes:
            prior = nodes_by_id.setdefault(node.id, node)
            if prior != node:
                raise ValueError("Topology contains contradictory representations of one session.")
        _reject_loaded_session_topology_cycles(nodes_by_id)
        if len(nodes_by_id) > SESSION_TOPOLOGY_MAX_NODES:
            raise ValueError(
                f"Session topology cannot retain more than {SESSION_TOPOLOGY_MAX_NODES} nodes."
            )
        return self


def encode_session_topology_cursor(
    parent_session_id: str,
    node: SessionTopologyNode,
) -> str:
    """Encode a parent-bound direct-child cursor."""

    parent_session_id = _bounded_session_topology_text(
        parent_session_id,
        "parent_session_id",
        max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    )
    if type(node) is not SessionTopologyNode:
        raise TypeError("Session topology cursors require SessionTopologyNode values.")
    raw = json.dumps(
        [
            parent_session_id,
            node.created_at.astimezone(UTC).isoformat(),
            node.id,
        ],
        separators=(",", ":"),
    )
    encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")
    if len(encoded) > SESSION_TOPOLOGY_MAX_CURSOR_BYTES:
        raise ValueError("Session topology cursor exceeds its byte limit.")
    return encoded


def decode_session_topology_cursor(
    cursor: str,
    *,
    parent_session_id: str,
) -> tuple[datetime, str]:
    """Decode a cursor and reject reuse against a different parent branch."""

    parent_session_id = _bounded_session_topology_text(
        parent_session_id,
        "parent_session_id",
        max_bytes=SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    )
    try:
        cursor = _bounded_session_topology_text(
            cursor,
            "cursor",
            max_bytes=SESSION_TOPOLOGY_MAX_CURSOR_BYTES,
        )
        encoded = cursor.encode("ascii")
        raw = base64.b64decode(encoded, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw) != encoded:
            raise ValueError("Non-canonical session topology cursor.")
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ValueError("Invalid session topology cursor.") from exc
    if (
        type(decoded) is not list
        or len(decoded) != 3
        or type(decoded[0]) is not str
        or type(decoded[1]) is not str
        or type(decoded[2]) is not str
        or decoded[0] != parent_session_id
        or not decoded[2]
    ):
        raise ValueError("Invalid session topology cursor.")
    try:
        child_session_id = require_clean_nonblank(decoded[2], "cursor child_session_id")
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid session topology cursor.") from exc
    try:
        created_at = datetime.fromisoformat(decoded[1])
    except ValueError as exc:
        raise ValueError("Invalid session topology cursor.") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("Invalid session topology cursor.")
    return created_at.astimezone(UTC), child_session_id


def build_session_topology_result(
    *,
    focus: SessionTopologyNode,
    ancestors: Iterable[SessionTopologyNode],
    expanded_parents: Iterable[SessionTopologyNode],
    branch_candidates: Iterable[Iterable[SessionTopologyNode]],
    child_limit: int,
) -> SessionTopologyStoreResult:
    """Apply the shared response-node ceiling without losing branch continuation."""

    if type(focus) is not SessionTopologyNode:
        raise TypeError("focus must be a SessionTopologyNode.")
    if (
        type(child_limit) is not int
        or child_limit < 1
        or child_limit > SESSION_TOPOLOGY_MAX_CHILD_LIMIT
    ):
        raise ValueError("child_limit is outside the session topology bounds.")
    ancestor_nodes = tuple(islice(ancestors, SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH + 1))
    if len(ancestor_nodes) > SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH:
        raise ValueError("Session topology exceeds the ancestor-node bound.")
    expanded_nodes = tuple(islice(expanded_parents, SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS + 1))
    if len(expanded_nodes) > SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS:
        raise ValueError("Session topology exceeds the expanded-parent bound.")
    candidate_pages = tuple(
        tuple(islice(page, child_limit + 1))
        for page in islice(branch_candidates, len(expanded_nodes) + 1)
    )
    if len(expanded_nodes) != len(candidate_pages):
        raise ValueError("Every expanded parent requires one candidate page.")

    retained_ids = {
        focus.id,
        *(node.id for node in ancestor_nodes),
        *(node.id for node in expanded_nodes),
    }
    nonempty_after = [
        sum(bool(later) for later in candidate_pages[index + 1 :])
        for index in range(len(candidate_pages))
    ]
    branches: list[SessionTopologyBranch] = []
    for parent, candidates, remaining_nonempty in zip(
        expanded_nodes,
        candidate_pages,
        nonempty_after,
        strict=True,
    ):
        available = SESSION_TOPOLOGY_MAX_NODES - len(retained_ids)
        branch_capacity = max(0, available - remaining_nonempty)
        retained = candidates[: min(child_limit, branch_capacity)]
        if candidates and not retained:
            raise RuntimeError("Session topology node allocation could not retain a branch cursor.")
        retained_ids.update(child.id for child in retained)
        has_more = len(candidates) > len(retained)
        branches.append(
            SessionTopologyBranch(
                parent_session_id=parent.id,
                children=retained,
                next_cursor=(
                    encode_session_topology_cursor(parent.id, retained[-1]) if has_more else None
                ),
                has_more=has_more,
            )
        )
    loaded_nodes = (
        focus,
        *ancestor_nodes,
        *expanded_nodes,
        *(child for branch in branches for child in branch.children),
    )
    loaded_nodes_by_id: dict[str, SessionTopologyNode] = {}
    for node in loaded_nodes:
        loaded_nodes_by_id.setdefault(node.id, node)
    _reject_loaded_session_topology_cycles(loaded_nodes_by_id)
    return SessionTopologyStoreResult(
        focus=focus,
        ancestors=ancestor_nodes,
        expanded_parents=expanded_nodes,
        branches=tuple(branches),
    )


def _reject_loaded_session_topology_cycles(
    nodes_by_id: Mapping[str, SessionTopologyNode],
) -> None:
    """Reject every parent cycle visible in the returned bounded projection."""

    complete: set[str] = set()
    for start_id in nodes_by_id:
        if start_id in complete:
            continue
        path: list[str] = []
        path_positions: dict[str, int] = {}
        current_id: str | None = start_id
        while current_id is not None and current_id in nodes_by_id:
            if current_id in complete:
                break
            if current_id in path_positions:
                raise SessionTopologyCycle(
                    "Session topology contains a cycle among loaded session nodes."
                )
            path_positions[current_id] = len(path)
            path.append(current_id)
            current_id = nodes_by_id[current_id].parent_session_id
        complete.update(path)
