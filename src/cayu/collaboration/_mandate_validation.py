"""Validate already-authenticated mandate evidence; never mint live authority."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise

from pydantic import Field

from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.collaboration._contracts import ContractValue, Identifier, ObjectRef, OwnerRef
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.mandates import (
    CollaborationMandate,
    InputChannel,
    MandateAccessContext,
    MandateAction,
    MandateDenied,
    MandateResolution,
    ResourceSelector,
    ResourceSelectorOwner,
)
from cayu.vaults.redaction import SecretRedactor


class MandateInput(ContractValue):
    source: ObjectRef
    channel: InputChannel


class MandateUse(ContractValue):
    unordered_fields = frozenset({"actions", "resources", "inputs"})

    audience: OwnerRef
    scope: Identifier
    actions: tuple[MandateAction, ...] = Field(min_length=1, max_length=16)
    resources: tuple[ResourceSelector, ...] = Field(max_length=32)
    inputs: tuple[MandateInput, ...] = Field(max_length=32)


def validate_mandate_resolution(
    resolution: MandateResolution,
    *,
    context: MandateAccessContext,
    resolver: ObjectRef,
    use: MandateUse,
    now_ms: int,
    resource_owners: Mapping[OwnerRef, ResourceSelectorOwner],
    redactor: SecretRedactor,
) -> MandateResolution:
    """Check every ancestor and the proposed use under a held resolver guard.

    Callers obtain ``now_ms`` from the receiving store. Repeat expiry validation
    at commit/exposure; this preparation is not a replacement for that check.
    Resource-owner callbacks execute outside store transactions and must not be
    invoked on a daemon event loop when implemented with blocking dependencies.
    """
    resolution = prepare_contract(MandateResolution, resolution, redactor=redactor)
    context = prepare_contract(MandateAccessContext, context, redactor=redactor)
    resolver = prepare_contract(ObjectRef, resolver, redactor=redactor)
    use = prepare_contract(MandateUse, use, redactor=redactor)
    if type(now_ms) is not int or not 0 <= now_ms <= MAX_PORTABLE_JSON_INTEGER:
        raise MandateDenied()
    principal = resolution.principal
    chain = resolution.chain.entries
    leaf = chain[-1]
    if (
        principal.resolver != resolver
        or principal.issuer != context.issuer
        or principal.principal != context.principal
        or principal.expires_at_ms <= now_ms
        or leaf.issuer != context.issuer
        or leaf.principal != context.principal
        or leaf.participant != context.participant
        or leaf.reference != context.mandate
        or (context.participant is not None and context.participant not in principal.participants)
        or use.audience not in principal.audiences
        or use.scope not in principal.scopes
        or not set(use.actions) <= set(principal.actions)
    ):
        raise MandateDenied()

    def canonical(selector: ResourceSelector) -> None:
        owner = resource_owners.get(selector.resource.owner)
        if owner is None:
            # Exact IDs have no alias/path interpretation. Subtrees always need
            # positive registered-owner evidence, even for identical values.
            if selector.mode != "exact":
                raise MandateDenied()
            return
        if (
            not isinstance(owner, ResourceSelectorOwner)
            or prepare_contract(OwnerRef, owner.owner, redactor=redactor) != selector.resource.owner
            or prepare_contract(
                ResourceSelector,
                owner.canonicalize(selector.model_copy(deep=True)),
                redactor=redactor,
            )
            != selector
        ):
            raise MandateDenied()

    def contains(parent: ResourceSelector, child: ResourceSelector) -> bool:
        if parent.resource.owner != child.resource.owner:
            return False
        if parent.mode == "exact":
            return parent == child
        owner = resource_owners.get(parent.resource.owner)
        if owner is None:
            return False
        parent_input = parent.model_copy(deep=True)
        child_input = child.model_copy(deep=True)
        result = owner.contains(parent_input, child_input)
        if (
            prepare_contract(ResourceSelector, parent_input, redactor=redactor) != parent
            or prepare_contract(ResourceSelector, child_input, redactor=redactor) != child
        ):
            raise MandateDenied()
        return result is True

    def covers(parent: CollaborationMandate, children: tuple[ResourceSelector, ...]) -> bool:
        return all(
            any(contains(allowed, child) for allowed in parent.resources) for child in children
        )

    # A bounded chain, bounded union and aggregate 64 KiB contract bound limit the
    # comparison work. Validate all selectors before invoking containment.
    for entry in chain:
        if entry.expires_at_ms <= now_ms:
            raise MandateDenied()
        for selector in entry.resources:
            canonical(selector)
        for excluded in entry.restrictions.excluded_sources:
            canonical(ResourceSelector(resource=excluded))
    for selector in use.resources:
        canonical(selector)
    for item in use.inputs:
        canonical(ResourceSelector(resource=item.source))
    for parent, child in pairwise(chain):
        if (
            child.remaining_delegations >= parent.remaining_delegations
            or child.expires_at_ms > parent.expires_at_ms
            or child.sponsor != parent.sponsor
            or not set(parent.budgets) <= set(child.budgets)
            or not set(child.audiences) <= set(parent.audiences)
            or not set(child.scopes) <= set(parent.scopes)
            or not set(child.actions) <= set(parent.actions)
            or not set(child.restrictions.channels) <= set(parent.restrictions.channels)
            or not set(parent.restrictions.excluded_sources)
            <= set(child.restrictions.excluded_sources)
            or child.restrictions.independence_policy != parent.restrictions.independence_policy
            or child.restrictions.disclosure_policy != parent.restrictions.disclosure_policy
            or not covers(parent, child.resources)
        ):
            raise MandateDenied()
    if (
        use.audience not in leaf.audiences
        or use.scope not in leaf.scopes
        or not set(use.actions) <= set(leaf.actions)
        or not covers(leaf, use.resources)
    ):
        raise MandateDenied()
    for item in use.inputs:
        if (
            item.channel not in leaf.restrictions.channels
            or item.source in leaf.restrictions.excluded_sources
            or not covers(leaf, (ResourceSelector(resource=item.source),))
        ):
            raise MandateDenied()
    return resolution
