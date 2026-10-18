"""Host-enforced read limit for one enrolled fork controller.

The bound travels on the same task-local slot as scoped session access, so every
store read that already honors a session bound evaluates this one too. Stores
compile it into SQL or apply it as a predicate; nothing filters after a read.
A session is readable when its ``agent_name`` is an enrolled agent or the
controller's own body for one and its fork lineage settles on a session whose
overlay record names this controller, whose lineage marker names this
controller or no controller, or on an unforked root. A session without either
entry, such as one written before lineage markers existed, settles through its
parents; one whose parents do not settle it is never readable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from cayu._resource_access_errors import ResourceAccessDenied
from cayu.sessions.fork_overlays import (
    FORK_LINEAGE_METADATA_KEY,
    FORK_OVERLAY_METADATA_KEY,
    ForkLineage,
    resolve_fork_lineage,
)
from cayu.sessions.records import Session


def overlay_record_admits(metadata: Mapping[str, Any], controller: str) -> bool:
    """Admit a session without an overlay record, or one whose record names ``controller``.

    The check is structural so stores evaluate it identically in SQL: a record
    that is present but does not name exactly this controller as a string is
    never readable, however malformed it is.
    """

    raw = metadata.get(FORK_OVERLAY_METADATA_KEY)
    if raw is None:
        return True
    if not isinstance(raw, Mapping):
        return False
    overlay = raw.get("overlay")
    if not isinstance(overlay, Mapping):
        return False
    named = overlay.get("controller")
    return type(named) is str and named == controller


def lineage_marker_admits(metadata: Mapping[str, Any], controller: str) -> bool:
    """Admit a session without a lineage marker, or one whose marker allows ``controller``.

    A marker naming no controller belongs to a host fork lineage and admits any
    reader the session's agent admits. A marker that is present but is not an
    object with a boolean ``forked`` and an admitted ``controller`` is never
    readable. An unforked marker must name no controller.
    """

    raw = metadata.get(FORK_LINEAGE_METADATA_KEY)
    if raw is None:
        return True
    if (
        not isinstance(raw, Mapping)
        or type(raw.get("forked")) is not bool
        or "controller" not in raw
    ):
        return False
    named = raw["controller"]
    if raw["forked"] is False:
        return named is None
    return named is None or (type(named) is str and named == controller)


def fork_controller_lineage_admits(metadata: Mapping[str, Any], controller: str) -> bool:
    """Whether neither the overlay record nor the lineage marker keeps ``controller`` out."""

    return overlay_record_admits(metadata, controller) and lineage_marker_admits(
        metadata, controller
    )


def fork_lineage_admits(lineage: ForkLineage, controller: str) -> bool:
    """Whether a resolved lineage lets ``controller`` read the session it was resolved for."""

    terminal = lineage.terminal
    return terminal is not None and fork_controller_lineage_admits(terminal.metadata, controller)


@dataclass(frozen=True, slots=True)
class _ForkControllerReadBounds:
    """What one controller may read; built only by ``ForkControllerAccess``.

    ``agent_names`` holds the enrolled agents and the controller's body for each.
    ``exclude_forked`` narrows multi-session event reads to sessions outside any
    forked lineage: no ``session.forked`` event, and a lineage that settles on
    an unforked root without passing a fork. It never widens what is readable.

    ``parent_of`` gives the predicate the session's parents, as a store's SQL
    reads them. Without it, a session that needs its parents to settle its
    lineage is denied.
    """

    controller: str
    agent_names: tuple[str, ...]
    exclude_forked: bool = False

    def __post_init__(self) -> None:
        if type(self.controller) is not str or not self.controller:
            raise TypeError("controller must be a non-empty string.")
        if type(self.agent_names) is not tuple or not self.agent_names:
            raise TypeError("agent_names must be a non-empty tuple.")
        if any(type(name) is not str for name in self.agent_names):
            raise TypeError("agent_names must contain strings.")
        if type(self.exclude_forked) is not bool:
            raise TypeError("exclude_forked must be a bool.")

    def admits(
        self, session: Session, parent_of: Callable[[str], Session | None] | None = None
    ) -> bool:
        return session.agent_name in self.agent_names and fork_lineage_admits(
            resolve_fork_lineage(session, parent_of), self.controller
        )

    def admits_row(self, session: Session) -> bool:
        """Check only the row's own agent, overlay record and lineage marker.

        A store's SQL clause also walks the row's parents; this rechecks the
        part of that limit a loaded row can show.
        """

        return session.agent_name in self.agent_names and fork_controller_lineage_admits(
            session.metadata, self.controller
        )

    def excludes(
        self, session: Session, parent_of: Callable[[str], Session | None] | None = None
    ) -> bool:
        """Whether ``exclude_forked`` drops this session for its lineage alone."""

        return self.exclude_forked and resolve_fork_lineage(session, parent_of).forked

    def require_read(
        self, session: Session | None, parent_of: Callable[[str], Session | None] | None = None
    ) -> Session:
        if session is None or not self.admits(session, parent_of):
            raise ResourceAccessDenied()
        return session

    def require_action(
        self,
        session: Session | None,
        action: str,
        parent_of: Callable[[str], Session | None] | None = None,
    ) -> Session:
        # A controller bound is read-only: any other action is denied outright.
        session = self.require_read(session, parent_of)
        if action != "read":
            raise ResourceAccessDenied()
        return session

    def cursor_audience(self) -> str:
        return "fork_controller:" + self.controller
