"""Who belongs where: memberships, the snapshot a policy reads, and the directory seam.

The application owns its organizations, departments and memberships. Cayu keeps none of its
own: a :class:`MembershipDirectory` reports a person's current memberships, and
:class:`~cayu.memberships.MembershipAccessPolicy` turns them into resource-access grants each
time Cayu asks. :class:`InMemoryMembershipDirectory` serves tests and trying the policy without
a database; :class:`~cayu.memberships.PostgresMemberships` reads the application's own tables.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["member", "admin"]
ROLES: tuple[Role, ...] = ("member", "admin")


@dataclass(frozen=True, slots=True)
class Membership:
    """A membership of ``organization``, and of at most one of its departments.

    A department ID starts with its organization's ID and a ``/``, as in ``acme/finance``,
    so it can never name another organization's department.
    """

    organization: str
    department: str | None = None


@dataclass(frozen=True, slots=True)
class AccessSnapshot:
    """A person's current memberships, read together with the revision they belong to.

    Every membership change raises the revision, so Cayu refuses a decision read before it.
    """

    memberships: tuple[Membership, ...]
    revision: int


@dataclass(frozen=True, slots=True)
class Member:
    """One row of an organization's members: who, where and with which role."""

    subject: str
    organization: str
    department: str | None
    role: str

    @property
    def membership(self) -> Membership:
        return Membership(self.organization, self.department)


class MembershipDirectory(Protocol):
    """Where a person's current memberships come from: the application's tables, an
    identity provider's groups or another service."""

    async def current(self, subject: str) -> AccessSnapshot: ...


class InMemoryMembershipDirectory:
    """Memberships kept in memory, for tests and for trying the policy without PostgreSQL."""

    def __init__(self, memberships: dict[str, Iterable[Membership]] | None = None) -> None:
        self._memberships = {
            subject: tuple(items) for subject, items in (memberships or {}).items()
        }
        self._revision = 0

    async def current(self, subject: str) -> AccessSnapshot:
        return AccessSnapshot(self._memberships.get(subject, ()), self._revision)

    def set_memberships(self, subject: str, memberships: Iterable[Membership]) -> None:
        """Replace ``subject``'s memberships and raise the revision."""

        self._memberships[subject] = tuple(memberships)
        self._revision += 1
