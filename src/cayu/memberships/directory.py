"""Who belongs where: memberships, the snapshot a policy reads, and the directory seam.

The application owns its organizations and memberships. Cayu keeps none of its own: a
:class:`MembershipDirectory` reports a person's current memberships, and
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
    """A membership of ``organization``."""

    organization: str


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
    role: str

    @property
    def membership(self) -> Membership:
        return Membership(self.organization)


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
        from cayu.memberships.knowledge_grants import InMemoryKnowledgeGrants

        #: The knowledge operators granted to work admitted before the policy granted any.
        self.knowledge_grants = InMemoryKnowledgeGrants()

    async def current(self, subject: str) -> AccessSnapshot:
        return AccessSnapshot(self._memberships.get(subject, ()), self._revision)

    def set_memberships(self, subject: str, memberships: Iterable[Membership]) -> None:
        """Replace ``subject``'s memberships and raise the revision."""

        self._memberships[subject] = tuple(memberships)
        self._revision += 1
