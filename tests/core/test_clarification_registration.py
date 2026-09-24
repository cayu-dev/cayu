"""Registration/copying characterization, not public service qualification."""

from contextlib import asynccontextmanager

import pytest
from tests.core.test_clarification_contracts import policy, reference

from cayu.collaboration._contracts import CollaborationConflict, CollaborationContractError
from cayu.collaboration._coordinator import ParticipantCoordinator
from cayu.collaboration._request_coordinator import RequestCoordinator
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.mandates import MandateResolver
from cayu.collaboration.request_access import RequestRegistration
from cayu.vaults.redaction import SecretRedactor


class UnusedResolver(MandateResolver):
    @property
    def ref(self):
        return reference("resolver")

    @asynccontextmanager
    async def acquire(self, context):
        pytest.fail("Registration must not acquire execution or disclosure authority")
        yield


def coordinator(policies):
    redactor = SecretRedactor()
    return RequestCoordinator(
        participants=ParticipantCoordinator(store=None, registration=None, redactor=redactor),
        registration=RequestRegistration(
            mandates=UnusedResolver(), max_ttl_ms=60000, clarification_policies=policies
        ),
        redactor=redactor,
    )


def test_policy_is_explicit_and_defensively_copied():
    with pytest.raises(CollaborationAccessDenied):
        coordinator(())._require_clarification_policy(policy())
    supplied = policy()
    owner = coordinator((supplied,))
    returned = owner._require_clarification_policy(supplied)
    object.__setattr__(supplied, "max_questions", 1)
    object.__setattr__(returned, "max_questions", 2)
    assert owner._require_clarification_policy(policy()) == policy()
    with pytest.raises(CollaborationConflict):
        owner._require_clarification_policy(supplied)


@pytest.mark.parametrize(
    "policies",
    [
        [policy()],
        (policy(), policy()),
        tuple(policy() for _ in range(33)),
        (policy(reference=reference("policy").model_copy(update={"revision": None})),),
        (
            policy(
                reference=reference("policy").model_copy(
                    update={
                        "owner": reference("policy").owner.model_copy(
                            update={"application_scope": "other"}
                        )
                    }
                )
            ),
        ),
        (policy().model_copy(update={"max_questions": True}),),
    ],
)
def test_policy_registration_rejects_invalid_or_unbounded_values(policies):
    with pytest.raises(CollaborationContractError):
        coordinator(policies)


@pytest.mark.parametrize("count", [1, 31, 32])
def test_policy_registration_finite_boundary(count):
    policies = tuple(
        policy(reference=reference("policy").model_copy(update={"object_id": str(index)}))
        for index in range(count)
    )
    owner = coordinator(policies)
    for registered in policies:
        assert owner._require_clarification_policy(registered) == registered
