"""Changes to an organization's members, and why a store refuses one."""

from __future__ import annotations

from dataclasses import dataclass

from cayu.memberships.directory import Role


@dataclass(frozen=True, slots=True)
class SetMember:
    """Add ``subject`` to the organization, or give them ``role`` if already a member."""

    subject: str
    role: Role = "member"


@dataclass(frozen=True, slots=True)
class RemoveMember:
    """Remove ``subject`` from the organization."""

    subject: str


MemberChange = SetMember | RemoveMember


class MembershipChangeRefused(Exception):
    """A membership change the store refused; nothing was changed."""


class NotAnAdmin(MembershipChangeRefused):
    """The caller isn't an admin of the organization they tried to change."""


class LastAdmin(MembershipChangeRefused):
    """The change would leave an organization that has an admin without one."""


class UnknownOrganization(MembershipChangeRefused):
    """The organization doesn't exist."""
