"""A durable admission dependency is resolved by the registered production reader."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.test_participant_identity import app, stores
from tests.core.test_request_planning_public import complete_plan, scenario

from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import ExactMatch, ObjectRef
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import (
    RequestPlanningDecline,
    RequestPlanningDefer,
    RequestPlanningPredecessor,
    RequestPlanningPrerequisite,
    planning_policy_commitment,
)
from cayu.collaboration.request_access import RequestPlanningAdmissionReader, RequestRegistration
from cayu.collaboration.requests import RequestAdmissionCommand
from cayu.vaults.redaction import SecretRedactor

__all__ = ["stores"]
pytestmark = pytest.mark.anyio
REDACTOR = SecretRedactor()


async def test_public_registered_prerequisite_requires_positive_exact_readback(stores, monkeypatch):
    store = stores()
    original, resolver, request, policy, provider = await scenario(store, "defer")
    initial = await original.initialize_collaboration()
    registered = original._participant_coordinator._registration
    acquire = resolver.acquire
    lock = asyncio.Lock()

    @asynccontextmanager
    async def held(context):
        async with lock, acquire(context) as resolution:
            yield resolution

    monkeypatch.setattr(resolver, "acquire", held)
    dependency_request = request.expected.intent.request.model_copy(
        update={"operation": initial.operation("dependency-request")}
    )
    accepted = await original.accept_collaboration_request(
        dependency_request, context=resolver.sender.context
    )
    decline = RequestPlanningDecline(reason="dependency_resolved")
    dependency_policy = policy.model_copy(update={"default": decline})
    dependency_plan = request.model_copy(
        update={
            "operation": initial.operation("dependency-plan"),
            "expected": accepted.expected,
            "expected_input_sha256": clarification_commitment(accepted.expected, REDACTOR),
            "admission_operation": initial.operation("dependency-admission"),
            "policy_sha256": planning_policy_commitment(dependency_policy, redactor=REDACTOR),
            "deadline_at_ms": accepted.expected.intent.selection.expires_at_ms,
        }
    )
    receiving = app(
        store,
        registered,
        collaboration_requests=RequestRegistration(
            mandates=resolver,
            max_ttl_ms=60000,
            planning_policies=(dependency_policy,),
        ),
    )
    await receiving.initialize_collaboration()
    admission = RequestAdmissionCommand(
        operation=dependency_plan.admission_operation,
        expected=accepted.expected,
        expected_revision=1,
        expected_input_revision=0,
        expected_input_sha256=dependency_plan.expected_input_sha256,
        generation=1,
        decision="decline",
        evidence=(),
        initiator=request.initiator,
        proposal_commitment=clarification_commitment(decline, REDACTOR),
    )
    reader_ref = ObjectRef(
        owner=initial.owner,
        kind="admission_reader",
        object_id="native",
        incarnation="one",
        revision=1,
    )
    policy = policy.model_copy(
        update={
            "default": RequestPlanningDefer(
                prerequisite=RequestPlanningPrerequisite(
                    reader=reader_ref,
                    expected=admission,
                    deadline_at_ms=request.deadline_at_ms,
                )
            )
        }
    )
    request = request.model_copy(
        update={"policy_sha256": planning_policy_commitment(policy, redactor=REDACTOR)}
    )
    application = app(
        store,
        registered,
        collaboration_requests=RequestRegistration(
            mandates=resolver,
            max_ttl_ms=60000,
            planning_policies=(policy,),
            planning_readers=(
                RequestPlanningAdmissionReader(
                    reader_ref, receiving.collaboration_admission_reader()
                ),
            ),
        ),
    )
    await application.initialize_collaboration()
    first = await complete_plan(application, request, resolver.recipient.context)
    successor = request.model_copy(
        update={
            "operation": initial.operation("successor-plan"),
            "admission_operation": initial.operation("successor-admission"),
            "expected_revision": 2,
            "admission_generation": 2,
            "planning_generation": 2,
            "predecessor": RequestPlanningPredecessor(
                operation=request.operation, revision=first.revision
            ),
        }
    )
    with pytest.raises(CollaborationUnavailable):
        await application.plan_collaboration_request(successor, context=resolver.recipient.context)
    assert (
        await application.lookup_collaboration_plan(request, context=resolver.recipient.context)
    ).receipt == first
    await complete_plan(receiving, dependency_plan, resolver.recipient.context)
    async with asyncio.timeout(180):
        second = await complete_plan(application, successor, resolver.recipient.context)
    assert second.state == "deferred"
    previous = await application.lookup_collaboration_plan(
        request, context=resolver.recipient.context
    )
    assert isinstance(previous, ExactMatch) and previous.receipt.state == "superseded"
    assert previous.receipt.successor.prerequisite.command == admission
    assert provider.requests == []
