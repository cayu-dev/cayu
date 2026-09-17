"""Contract characterization; public receiver qualification lives in export tests."""

from __future__ import annotations

import pytest

from cayu.collaboration._contracts import ObjectRef, OwnerRef
from cayu.collaboration._mandate_validation import (
    MandateInput,
    MandateUse,
    validate_mandate_resolution,
)
from cayu.collaboration._preparation import contract_bytes
from cayu.collaboration.mandates import (
    CollaborationMandate,
    MandateAccessContext,
    MandateChain,
    MandateDenied,
    MandateResolution,
    MandateRestrictions,
    PrincipalResolution,
    ResourceSelector,
    ResourceSelectorOwner,
)
from cayu.vaults.redaction import SecretRedactor

OWNER = OwnerRef(application_scope="project", owner_id="authority", incarnation="one")
AUDIENCE = OwnerRef(application_scope="project", owner_id="reviewer", incarnation="one")
REDACTOR = SecretRedactor()


def ref(name):
    return ObjectRef(owner=OWNER, kind="test", object_id=name, incarnation="one", revision=1)


def evidence():
    resource = ResourceSelector(resource=ref("source"))
    root = CollaborationMandate(
        reference=ref("root"),
        root=ref("root"),
        parent=None,
        issuer=OWNER,
        principal="alice",
        participant=None,
        audiences=(AUDIENCE,),
        scopes=("project",),
        actions=("source", "publish", "readback", "expose"),
        resources=(resource,),
        remaining_delegations=2,
        sponsor=ref("sponsor"),
        budgets=(ref("common-budget"),),
        restrictions=MandateRestrictions(
            channels=("source", "tool"),
            excluded_sources=(ref("private"),),
            independence_policy=ref("blind"),
            disclosure_policy=ref("disclosure"),
        ),
        expires_at_ms=2000,
        revocation_generation=1,
    )
    child = root.model_copy(
        update={
            "reference": ref("child"),
            "parent": root.reference,
            "remaining_delegations": 1,
            "expires_at_ms": 1500,
        }
    )
    resolution = MandateResolution(
        principal=PrincipalResolution(
            resolver=ref("resolver"),
            issuer=OWNER,
            principal="alice",
            participants=(),
            audiences=(AUDIENCE,),
            scopes=("project",),
            actions=root.actions,
            expires_at_ms=2000,
        ),
        chain=MandateChain(entries=(root, child)),
    )
    context = MandateAccessContext(issuer=OWNER, principal="alice", mandate=child.reference)
    use = MandateUse(
        audience=AUDIENCE,
        scope="project",
        actions=("source", "publish"),
        resources=(resource,),
        inputs=(MandateInput(source=ref("source"), channel="source"),),
    )
    return resolution, context, use


def check(resolution, context, use, **kwargs):
    return validate_mandate_resolution(
        resolution,
        context=context,
        resolver=ref("resolver"),
        use=use,
        now_ms=kwargs.pop("now_ms", 1000),
        resource_owners=kwargs.pop("resource_owners", {}),
        redactor=REDACTOR,
        **kwargs,
    )


def test_exact_chain_validates_after_json_reconstruction():
    resolution, context, use = evidence()
    reconstructed = MandateResolution.model_validate_json(resolution.model_dump_json())
    assert check(reconstructed, context, use) == resolution
    assert contract_bytes(check(reconstructed, context, use), redactor=REDACTOR) == contract_bytes(
        resolution, redactor=REDACTOR
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("remaining_delegations", 2),
        ("expires_at_ms", 2001),
        ("sponsor", ref("other-sponsor")),
        ("budgets", ()),
        ("audiences", (OWNER, AUDIENCE)),
        ("scopes", ("project", "other")),
        ("actions", ("source", "publish", "execute")),
        ("resources", (ResourceSelector(resource=ref("private")),)),
    ],
)
def test_delegation_cannot_widen_one_field(field, value):
    resolution, context, use = evidence()
    root, child = resolution.chain.entries
    modified = resolution.model_copy(
        update={"chain": MandateChain(entries=(root, child.model_copy(update={field: value})))}
    )
    with pytest.raises(MandateDenied):
        check(modified, context, use)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("channels", ("source", "tool", "artifact")),
        ("excluded_sources", ()),
        ("independence_policy", ref("not-blind")),
        ("disclosure_policy", ref("other-disclosure")),
    ],
)
def test_delegation_cannot_drop_restrictions(field, value):
    resolution, context, use = evidence()
    root, child = resolution.chain.entries
    child = child.model_copy(
        update={"restrictions": child.restrictions.model_copy(update={field: value})}
    )
    resolution = resolution.model_copy(update={"chain": MandateChain(entries=(root, child))})
    with pytest.raises(MandateDenied):
        check(resolution, context, use)


@pytest.mark.parametrize("now", [1500, 2000, True, -1, 2**53])
def test_expiry_equality_and_invalid_clock_fail_closed(now):
    with pytest.raises(MandateDenied):
        check(*evidence(), now_ms=now)


@pytest.mark.parametrize("channel", ["prompt", "context", "retrieval", "artifact"])
def test_input_restrictions_apply_to_each_channel(channel):
    resolution, context, use = evidence()
    use = use.model_copy(update={"inputs": (MandateInput(source=ref("source"), channel=channel),)})
    with pytest.raises(MandateDenied):
        check(resolution, context, use)


def test_extra_budget_ceiling_and_fewer_actions_are_narrowing():
    resolution, context, use = evidence()
    root, child = resolution.chain.entries
    child = child.model_copy(
        update={"budgets": (*child.budgets, ref("extra-budget")), "actions": use.actions}
    )
    resolution = resolution.model_copy(update={"chain": MandateChain(entries=(root, child))})
    assert check(resolution, context, use).chain.entries[-1].budgets == child.budgets


def test_unordered_permissions_have_one_identity():
    resolution, _, _ = evidence()
    root = resolution.chain.entries[0]
    reordered = CollaborationMandate.model_validate(
        root.model_copy(update={"actions": tuple(reversed(root.actions))})
    )
    assert contract_bytes(root, redactor=REDACTOR) == contract_bytes(reordered, redactor=REDACTOR)


class ResourceOwner(ResourceSelectorOwner):
    def __init__(self):
        self.result = True

    @property
    def owner(self):
        return OWNER

    def canonicalize(self, selector):
        return selector

    def contains(self, parent, child):
        return self.result if child.resource == ref("source") else False


def test_subtree_requires_positive_owner_evidence_not_prefix():
    resolution, context, use = evidence()
    root, child = resolution.chain.entries
    root = root.model_copy(
        update={"resources": (ResourceSelector(resource=ref("namespace"), mode="subtree"),)}
    )
    resolution = resolution.model_copy(update={"chain": MandateChain(entries=(root, child))})
    with pytest.raises(MandateDenied):
        check(resolution, context, use)
    owner = ResourceOwner()
    assert check(resolution, context, use, resource_owners={OWNER: owner}) == resolution
    owner.result = 1
    with pytest.raises(MandateDenied):
        check(resolution, context, use, resource_owners={OWNER: owner})


def test_same_principal_different_issuer_does_not_match():
    resolution, context, use = evidence()
    context = context.model_copy(update={"issuer": OWNER.model_copy(update={"incarnation": "two"})})
    with pytest.raises(MandateDenied):
        check(resolution, context, use)
