"""Who may see and continue which conversations, from current membership.

Each person sees their own conversations and those shared with an organization they belong
to, and nothing of any other organization. Only the owner continues, changes or deletes a
conversation: seeing a shared one doesn't let anyone else run it. Cayu applies this to
sessions, tasks and artifacts on every read, listing and background step, asking the policy
again each time, so a removed member loses access at their next step.

Grouping people inside an organization is the application's: a policy of its own can add
rules for its groups to this one's grant.

With ``knowledge=True`` each person also reads, and the agent remembers into, the knowledge of
the organizations they belong to, and nothing of any other organization's. Work admitted
before that was turned on keeps a grant without knowledge until an operator grants it
(``cayu memberships grant-knowledge``, :mod:`cayu.memberships.knowledge_grants`).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from cayu._resource_access_binding import ResourceExecutionBinding
from cayu.memberships.directory import MembershipDirectory
from cayu.memberships.knowledge_grants import KnowledgeGrants
from cayu.memberships.sharing import SHARING_LABELS
from cayu.resource_access import (
    ResourceAccessDecision,
    ResourceAccessGrant,
    ResourceAccessPolicy,
    admitted_scope,
    binding_sha256,
)
from cayu.sessions import SessionAccessRule, SessionAccessScope, SessionAccessSelector

# Cayu stores the authority with every piece of work it admits and refuses a stored binding
# whose authority differs, so this keeps the string generated projects have always used.
DEFAULT_AUTHORITY = "b2b-membership-v1"
# The longest Cayu may use one decision. It asks the policy again at every protected step.
DEFAULT_DECISION_LIFETIME = timedelta(seconds=60)
# Cayu stores each admitted grant with the work it admits, up to 64 KiB. With the longest IDs
# (64 characters) a person's grant fits many more memberships; this leaves room to spare.
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

    ``knowledge=True`` grants knowledge too: a member reads, creates and revises entries
    labelled with one of their organizations (``organization`` can't be changed), and nothing
    else. Pair it with a knowledge scope whose ``session_label_keys`` names ``organization``,
    so inside a conversation, or any session labelled with an organization, only that
    organization's knowledge is used. Off by default, which denies all knowledge, as before.

    Work admitted before ``knowledge=True`` keeps the grant it was admitted with, which has no
    knowledge. ``knowledge_grants`` is where an operator's decision to grant it is recorded
    (``cayu memberships grant-knowledge``); such work then gets the knowledge of the
    organizations it was admitted in. It defaults to the directory's own
    (``PostgresMemberships.knowledge_grants``), when it has one.
    """

    def __init__(
        self,
        directory: MembershipDirectory,
        *,
        authority: str = DEFAULT_AUTHORITY,
        decision_lifetime: timedelta = DEFAULT_DECISION_LIFETIME,
        max_memberships: int = DEFAULT_MAX_MEMBERSHIPS,
        knowledge: bool = False,
        knowledge_grants: KnowledgeGrants | None = None,
    ) -> None:
        if not isinstance(authority, str) or not authority.strip():
            raise ValueError("A membership policy's authority is a non-blank string.")
        if not isinstance(decision_lifetime, timedelta) or decision_lifetime <= timedelta(0):
            raise ValueError("decision_lifetime must be a positive timedelta.")
        if type(max_memberships) is not int or max_memberships < 1:
            raise ValueError("max_memberships must be a positive integer.")
        if type(knowledge) is not bool:
            raise ValueError("knowledge must be a boolean.")
        self._directory = directory
        self._authority = authority
        self._decision_lifetime = decision_lifetime
        self._max_memberships = max_memberships
        self._knowledge = knowledge
        self._knowledge_grants = (
            getattr(directory, "knowledge_grants", None)
            if knowledge_grants is None
            else knowledge_grants
        )

    @property
    def authority(self) -> str:
        return self._authority

    async def knowledge_admission(
        self, binding: ResourceExecutionBinding
    ) -> SessionAccessScope | None:
        """The knowledge of the organizations ``binding``'s work was admitted in, once an
        operator has granted it; ``None`` until then, or without ``knowledge=True``."""

        if not self._knowledge or self._knowledge_grants is None:
            return None
        if binding.authority != self._authority:
            return None
        organizations = admitted_organizations(binding)
        if not organizations or not await self._knowledge_grants.granted(binding_sha256(binding)):
            return None
        return _knowledge_within(organizations)

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
        owner = [subject]
        # A person's own conversations in their organizations, however they are shared:
        # one labelled with a sharing this policy doesn't grant on, such as a grouping the
        # application used to have, stays its owner's alone.
        own = (_rule(organization=organizations, owner=owner),)
        visible = (*own, _rule(organization=organizations, sharing=["organization"]))
        # A conversation a person starts is theirs, inside their memberships, and shared only
        # in a way this policy knows.
        conversations = SessionAccessScope(
            read=visible,
            create=(
                _rule(
                    organization=organizations, owner=owner, sharing=["personal", "organization"]
                ),
            ),
            execute=own,
            modify=own,
            delete=own,
            update_labels=own,
            protected_label_keys=SHARING_LABELS,
        )
        # Tasks and artifacts take the labels of the conversation that makes them, so the
        # owner creates them in any conversation of their own. Whatever their sharing, nobody
        # else sees one unless it is shared with an organization they belong to.
        outputs = replace(conversations, create=own)
        knowledge = _knowledge_within(organizations) if self._knowledge else None
        return ResourceAccessDecision(
            ResourceAccessGrant(
                sessions=conversations, tasks=outputs, artifacts=outputs, knowledge=knowledge
            ),
            revision=snapshot.revision,
            valid_until=datetime.now(UTC) + self._decision_lifetime,
        )


def _knowledge_within(organizations: list[str]) -> SessionAccessScope:
    # Knowledge is shared by a whole organization, whichever conversation it came from.
    within = (_rule(organization=organizations),)
    return SessionAccessScope(read=within, create=within, modify=within)


def admitted_organizations(binding: ResourceExecutionBinding) -> list[str]:
    """The organizations ``binding``'s work was admitted in: those its stored grant lets it
    read conversations of, sorted."""

    organizations: set[str] = set()
    for rule in admitted_scope(binding, "sessions").read:
        for selector in rule.selectors:
            if selector.key == "organization" and selector.operator == "in":
                organizations.update(selector.values)
    return sorted(organizations)
