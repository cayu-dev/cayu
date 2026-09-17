"""Bounded mandate evidence and registered owner boundaries.

These values describe authority; constructing or deserializing them grants none.
Only a trusted registered resolver's held guard supplies current permission.
Receiving owners must still admit the exact effect and enforce its own contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.collaboration._contracts import (
    ContractValue,
    Generation,
    Identifier,
    ObjectRef,
    OwnerRef,
)
from cayu.collaboration.participants import ParticipantRef

MandateAction = Literal[
    "consult",
    "source",
    "prepare",
    "execute",
    "publish",
    "commission",
    "accept_responsibility",
    "accept_result",
    "administer",
    "readback",
    "expose",
    "release",
    "retire",
]
InputChannel = Literal["prompt", "context", "tool", "retrieval", "artifact", "source"]
Expiry = Annotated[StrictInt, Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)]


class MandateDenied(PermissionError):
    def __init__(self) -> None:
        super().__init__("Collaboration mandate does not authorize the requested operation.")


class ResourceSelector(ContractValue):
    """An exact revision or a subtree understood only by its registered owner.

    A bounded union is a tuple of these selectors. Subtree names are not paths
    that Cayu compares by prefix; their interpretation belongs to the owner.
    """

    resource: ObjectRef
    mode: Literal["exact", "subtree"] = "exact"

    @model_validator(mode="after")
    def pinned_revision(self) -> ResourceSelector:
        if self.resource.revision is None:
            raise ValueError("Resource selectors require an exact owner revision.")
        return self


class ResourceSelectorOwner(ABC):
    @property
    @abstractmethod
    def owner(self) -> OwnerRef:
        """Exact registered resource owner/configuration incarnation."""

    @abstractmethod
    def canonicalize(self, selector: ResourceSelector) -> ResourceSelector:
        """Resolve aliases to a pinned, bounded canonical selector, without effects."""

    @abstractmethod
    def contains(self, parent: ResourceSelector, child: ResourceSelector) -> bool:
        """Positively validate owner-defined subtree containment, not string prefixes."""


class MandateRestrictions(ContractValue):
    unordered_fields = frozenset({"channels", "excluded_sources"})

    channels: tuple[InputChannel, ...] = Field(max_length=6)
    excluded_sources: tuple[ObjectRef, ...] = Field(max_length=32)
    independence_policy: ObjectRef
    disclosure_policy: ObjectRef

    @model_validator(mode="after")
    def pinned_policy_and_exclusions(self) -> MandateRestrictions:
        if any(
            ref.revision is None
            for ref in (
                self.independence_policy,
                self.disclosure_policy,
                *self.excluded_sources,
            )
        ):
            raise ValueError("Restrictions require pinned policy and source revisions.")
        return self


class CollaborationMandate(ContractValue):
    """One frozen step in an authenticated root-to-leaf delegation chain."""

    unordered_fields = frozenset({"audiences", "scopes", "actions", "resources", "budgets"})

    reference: ObjectRef
    root: ObjectRef
    parent: ObjectRef | None
    issuer: OwnerRef
    principal: Identifier
    participant: ParticipantRef | None
    audiences: tuple[OwnerRef, ...] = Field(max_length=16)
    scopes: tuple[Identifier, ...] = Field(max_length=16)
    actions: tuple[MandateAction, ...] = Field(max_length=16)
    resources: tuple[ResourceSelector, ...] = Field(max_length=32)
    remaining_delegations: StrictInt = Field(ge=0, le=8)
    sponsor: ObjectRef | None
    budgets: tuple[ObjectRef, ...] = Field(max_length=16)
    restrictions: MandateRestrictions
    expires_at_ms: Expiry
    revocation_generation: Generation

    @model_validator(mode="after")
    def exact_authority(self) -> CollaborationMandate:
        scope = self.reference.owner.application_scope
        if (
            self.reference.owner != self.issuer
            or self.reference.revision is None
            or self.root.revision is None
            or self.root.owner.application_scope != scope
            or (self.parent is not None and self.parent.revision is None)
            or (self.participant is not None and self.participant.owner.application_scope != scope)
            or (self.parent is None and self.root != self.reference)
            or (self.parent is not None and self.root == self.reference)
            or any(ref.revision is None for ref in self.budgets)
            or (self.sponsor is not None and self.sponsor.revision is None)
        ):
            raise ValueError("Mandate authority is inconsistent.")
        return self


class MandateChain(ContractValue):
    """Ordered history authenticated together under current revocation policy."""

    entries: tuple[CollaborationMandate, ...] = Field(min_length=1, max_length=9)

    @model_validator(mode="after")
    def exact_lineage(self) -> MandateChain:
        root = self.entries[0]
        if root.parent is not None or len({entry.reference for entry in self.entries}) != len(
            self.entries
        ):
            raise ValueError("Mandate chain must have one nonrepeating root.")
        for parent, child in pairwise(self.entries):
            if (
                child.parent != parent.reference
                or child.root != root.reference
                or child.reference.owner.application_scope != root.issuer.application_scope
            ):
                raise ValueError("Mandate delegation lineage conflicts.")
        return self


class PrincipalResolution(ContractValue):
    """Fresh resolver output, not a portable credential or participant owner claim."""

    unordered_fields = frozenset({"participants", "audiences", "actions", "scopes"})

    resolver: ObjectRef
    issuer: OwnerRef
    principal: Identifier
    participants: tuple[ParticipantRef, ...] = Field(max_length=16)
    audiences: tuple[OwnerRef, ...] = Field(max_length=16)
    actions: tuple[MandateAction, ...] = Field(max_length=16)
    scopes: tuple[Identifier, ...] = Field(max_length=16)
    expires_at_ms: Expiry

    @model_validator(mode="after")
    def registered_issuer(self) -> PrincipalResolution:
        if self.resolver.owner != self.issuer or self.resolver.revision is None:
            raise ValueError("Principal resolution requires an exact issuer configuration.")
        return self


class MandateAccessContext(ContractValue):
    """Authenticated host input, separate from any public operation proposal."""

    issuer: OwnerRef
    principal: Identifier
    participant: ParticipantRef | None = None
    mandate: ObjectRef | None = None


class MandateResolution(ContractValue):
    principal: PrincipalResolution
    chain: MandateChain


class MandateResolver(ABC):
    @property
    @abstractmethod
    def ref(self) -> ObjectRef:
        """Stable versioned implementation/configuration, never an ambient default."""

    @abstractmethod
    def acquire(
        self, context: MandateAccessContext
    ) -> AbstractAsyncContextManager[MandateResolution]:
        """Authenticate identity and the whole chain, holding current revocation.

        Deny replaced/missing issuers, mandates or ancestor generations. Keep the
        guard held across owned effects/exposure; historical values alone cannot
        renew it. This extension is trusted application code, not an auth service.
        """
