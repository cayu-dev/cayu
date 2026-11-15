"""Who may see and continue which conversations, from current membership.

Each person sees their own personal conversations, those shared with a department they
belong to, and those shared with an organization they belong to, and nothing of any other
organization. Only the owner continues, changes or deletes a conversation: seeing a shared
one doesn't let anyone else run it. Cayu applies this to sessions, tasks and artifacts on
every read, listing and background step, asking the policy again each time, so a removed
member loses access at their next step.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cayu.memberships.directory import MembershipDirectory
from cayu.memberships.sharing import SHARING_LABELS
from cayu.resource_access import (
    ResourceAccessDecision,
    ResourceAccessGrant,
    ResourceAccessPolicy,
)
from cayu.sessions import SessionAccessRule, SessionAccessScope, SessionAccessSelector

# Cayu stores the authority with every piece of work it admits and refuses a stored binding
# whose authority differs, so this keeps the string generated projects have always used.
DEFAULT_AUTHORITY = "b2b-membership-v1"
# The longest Cayu may use one decision. It asks the policy again at every protected step.
DEFAULT_DECISION_LIFETIME = timedelta(seconds=60)
# Cayu stores each admitted grant with the work it admits, up to 64 KiB. With the longest IDs
# (64 characters) a person's grant fits up to 20 memberships; this leaves room to spare.
DEFAULT_MAX_MEMBERSHIPS = 16


def _rule(**labels: list[str]) -> SessionAccessRule:
    return SessionAccessRule(
        selectors=tuple(
            SessionAccessSelector(key=key, values=tuple(values)) for key, values in labels.items()
        )
    )


class MembershipAccessPolicy(ResourceAccessPolicy):
    """Cayu resource-access grants from the memberships ``directory`` reports.

    ``authority`` names the policy in every binding Cayu stores; changing it strands work
    admitted under the old name. ``decision_lifetime`` bounds how long Cayu may use one
    decision, and ``max_memberships`` how many memberships one person's grant may carry.
    """

    def __init__(
        self,
        directory: MembershipDirectory,
        *,
        authority: str = DEFAULT_AUTHORITY,
        decision_lifetime: timedelta = DEFAULT_DECISION_LIFETIME,
        max_memberships: int = DEFAULT_MAX_MEMBERSHIPS,
    ) -> None:
        if not isinstance(authority, str) or not authority.strip():
            raise ValueError("A membership policy's authority is a non-blank string.")
        if not isinstance(decision_lifetime, timedelta) or decision_lifetime <= timedelta(0):
            raise ValueError("decision_lifetime must be a positive timedelta.")
        if type(max_memberships) is not int or max_memberships < 1:
            raise ValueError("max_memberships must be a positive integer.")
        self._directory = directory
        self._authority = authority
        self._decision_lifetime = decision_lifetime
        self._max_memberships = max_memberships

    @property
    def authority(self) -> str:
        return self._authority

    async def resolve(self, subject: str) -> ResourceAccessDecision:
        snapshot = await self._directory.current(subject)
        organizations = sorted({item.organization for item in snapshot.memberships})
        if not organizations:
            return ResourceAccessDecision(ResourceAccessGrant(), revision=snapshot.revision)
        if len(snapshot.memberships) > self._max_memberships:
            raise ValueError(
                f"{subject!r} has more than {self._max_memberships} memberships, "
                "more than one grant holds."
            )
        for item in snapshot.memberships:
            if item.department is not None and not item.department.startswith(
                f"{item.organization}/"
            ):
                raise ValueError(
                    f"{subject!r} has a membership of department {item.department!r}, "
                    f"which isn't one of {item.organization!r}'s: a department ID starts "
                    "with its organization's ID and '/'."
                )
        departments = sorted(
            {item.department for item in snapshot.memberships if item.department is not None}
        )
        owner = [subject]
        visible = [
            _rule(organization=organizations, owner=owner, sharing=["personal"]),
            _rule(organization=organizations, sharing=["organization"]),
        ]
        # A new conversation is always the person's own, inside their memberships.
        creatable = [
            _rule(organization=organizations, owner=owner, sharing=["personal", "organization"])
        ]
        own = list(creatable)
        if departments:
            # A department conversation is decided by its department alone: a department ID
            # names its organization (`acme/finance`), so whatever organization label it
            # carries grants nothing.
            visible.append(_rule(department=departments, sharing=["department"]))
            creatable.append(
                _rule(
                    organization=organizations,
                    department=departments,
                    owner=owner,
                    sharing=["department"],
                )
            )
            own.append(_rule(department=departments, owner=owner, sharing=["department"]))
        scope = SessionAccessScope(
            read=tuple(visible),
            create=tuple(creatable),
            execute=tuple(own),
            modify=tuple(own),
            delete=tuple(own),
            update_labels=tuple(own),
            protected_label_keys=SHARING_LABELS,
        )
        return ResourceAccessDecision(
            ResourceAccessGrant(sessions=scope, tasks=scope, artifacts=scope),
            revision=snapshot.revision,
            valid_until=datetime.now(UTC) + self._decision_lifetime,
        )
