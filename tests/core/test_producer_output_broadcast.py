"""Finite broadcast repairs only unresolved receiving responsibilities."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from tests.core import test_prepared_admission_public as preparations
from tests.core import test_producer_output_contracts as scenarios
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import registration as participant_registration
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.collaboration._contracts import ExactMatch, ObjectRef, OwnerRef
from cayu.collaboration._producer_acceptance import ProducerOutputAcceptanceReader
from cayu.collaboration._producer_contracts import ProducerOutputRegistration
from cayu.collaboration._producer_delivery_store import read_delivery
from cayu.collaboration._session_export_store import digest
from cayu.collaboration.exports import SessionExportAccessContext
from cayu.collaboration.mandates import ResourceSelector
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import RunRequest
from cayu.sessions.context_views import RecipientSessionCreationRequest


@pytest.mark.anyio
@pytest.mark.parametrize("second_expires", [False, True, "unprepared"])
async def test_finite_broadcast_repairs_only_unresolved_recipient(
    native_stores, monkeypatch, second_expires
):
    exports = scenarios.output_exports
    setup = preparations.setup

    async def broadcast_setup(store):
        registration = participant_registration()
        registration = replace(
            registration,
            bootstrap=registration.bootstrap.model_copy(
                update={
                    "limits": registration.bootstrap.limits.model_copy(
                        update={"retained_bytes": 8 * 1024 * 1024}
                    )
                }
            ),
        )
        return await setup(store, reg=registration)

    monkeypatch.setattr(preparations, "setup", broadcast_setup)

    def two_readers(initialized, request, resolver, **kwargs):
        registration = exports(initialized, request, resolver, **kwargs)
        recipient = resolver.recipient.context.participant
        return replace(
            registration,
            readers=(
                *registration.readers,
                ProducerOutputAcceptanceReader(
                    **kwargs,
                    namespace=initialized.operation("producer-reader"),
                    audience=OwnerRef(
                        application_scope=initialized.owner.application_scope,
                        owner_id=recipient.participant_id,
                        incarnation=recipient.incarnation,
                    ),
                ),
            ),
        )

    monkeypatch.setattr(scenarios, "output_exports", two_readers)
    (
        app,
        resolver,
        admission,
        provider,
        session,
        initialized,
        proposal,
        execution,
    ) = await scenarios.output_scenario(native_stores, with_exports=True)
    second_recipient = admission.prepared.recipient
    second_session, _ = await app.create_recipient_session(
        RecipientSessionCreationRequest(
            request=RunRequest(agent_name="reviewer", messages=[]),
            creation_key="second-output-consumer:" + initialized.owner.application_scope,
            recipient=second_recipient,
        ),
        context=CONTEXT,
    )
    first = proposal.destinations[0]
    second = first.model_copy(
        update={
            "operation": initialized.operation("second-output-destination"),
            "recipient": second_recipient,
            "attempt": first.attempt.model_copy(
                update={
                    "deadline_at_ms": first.attempt.deadline_at_ms - 1,
                    "append_key": first.attempt.append_key.model_copy(
                        update={
                            "consumer_id": second_recipient.participant_id,
                            "consumer_participant_incarnation": second_recipient.incarnation,
                            "target_session_id": second_session.id,
                            "target_session_instance_id": second_session.instance_id,
                        }
                    ),
                    "target_run_epoch": second_session.run_epoch,
                    "target_transcript_cursor": len(
                        await native_stores[1].load_transcript(second_session.id)
                    ),
                }
            ),
        }
    )
    proposal = ProducerOutputRegistration.model_validate(
        proposal.model_copy(
            update={
                "destinations": (first, second),
                "limits": proposal.limits.model_copy(update={"destinations": 2}),
            }
        )
    )
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    monkeypatch.setattr(app._session_export_coordinator.owners, "observation_timeout", 60)
    await app.register_producer_output(proposal, execution, context=resolver.recipient.context)
    resolution = resolver.recipient.resolution
    actions = tuple(
        dict.fromkeys((*resolution.principal.actions, "execute", "source", "expose", "publish"))
    )
    audiences = (
        initialized.owner,
        *(
            OwnerRef(
                application_scope=initialized.owner.application_scope,
                owner_id=destination.recipient.participant_id,
                incarnation=destination.recipient.incarnation,
            )
            for destination in proposal.destinations
        ),
    )
    resource = ResourceSelector(
        resource=ObjectRef(
            owner=initialized.owner,
            kind="session_transcript_row",
            object_id=session.id,
            incarnation=session.instance_id,
            revision=3,
        )
    )
    resolver.recipient.resolution = resolution.model_copy(
        update={
            "principal": resolution.principal.model_copy(
                update={"actions": actions, "audiences": audiences}
            ),
            "chain": resolution.chain.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(
                            update={
                                "actions": actions,
                                "audiences": audiences,
                                "resources": (resource,),
                                "restrictions": entry.restrictions.model_copy(
                                    update={"channels": ("prompt", "source")}
                                ),
                            }
                        )
                        for entry in resolution.chain.entries
                    )
                }
            ),
        }
    )
    answer = "retained broadcast"
    provider._batches = ((ModelStreamEvent.text_delta(answer), ModelStreamEvent.completed()),)
    async for _ in app.execute_producer_output(
        proposal,
        execution,
        context=CONTEXT,
        producer_context=resolver.recipient.context,
    ):
        pass
    completion = await app.retain_producer_completion(proposal, context=CONTEXT)
    assert completion.output.disposition == "answer"
    context = SessionExportAccessContext(
        principal=resolver.recipient.context.principal, mandate=resolver.recipient.context
    )
    policy = app._session_export_coordinator.registration.policy
    for destination in proposal.destinations:
        exported = await app.export_producer_output(
            proposal, destination.operation, context=context
        )
        source = await app.lookup_session_export(exported.request, context=context)
        assert isinstance(source, ExactMatch)
        policy.register_export(
            source.receipt,
            payload_sha256=digest({"text": answer, "artifact_commitments": []}),
            consumer_id=destination.recipient.participant_id,
        )
    policy.allowed_receipts.add(proposal.operation.caller_key)
    outcome = await app.publish_producer_outcome(
        proposal, destination=first.operation, context=context
    )
    assert outcome.command.outcome == "answered"
    append = app.append_peer_content
    entered = []
    fail_second = True

    async def interrupt_second(request, **kwargs):
        entered.append(request.append_key.consumer_id)
        if fail_second and request.append_key.consumer_id == second_recipient.participant_id:
            raise RuntimeError("receiver unavailable before append")
        return await append(request, **kwargs)

    monkeypatch.setattr(app, "append_peer_content", interrupt_second)
    accepted_first = await app.deliver_producer_output(proposal, first.operation, context=context)
    assert accepted_first.receipt.status == "appended"
    if second_expires == "unprepared":
        from tests.core.producer_broadcast_cleanup import settle_broadcast

        from cayu import ProducerDeliveryRecovery

        pending = await app.pending_producer_outputs(
            proposal.admission.prepared.recipient, context=CONTEXT
        )
        token = next(
            item.recovery
            for item in pending.items
            if item.recovery.registration == proposal.operation
        )
        recovery = ProducerDeliveryRecovery(**token.model_dump(), destination=second.operation)
        excluded = await app.reconcile_producer_delivery(recovery, context=CONTEXT, exclude=True)
        assert excluded.state == "excluded"
        assert await app.reconcile_producer_delivery(recovery, context=CONTEXT) == excluded
        await settle_broadcast(
            app,
            proposal,
            resolver,
            context,
            completion,
            provider,
            deliveries=(accepted_first, None),
        )
        assert entered == [first.recipient.participant_id]
        return
    with pytest.raises(CollaborationUnavailable):
        await app.deliver_producer_output(proposal, second.operation, context=context)
    async with native_stores[2]()._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        pending = await read_delivery(tx, proposal, second, redactor=app._secret_redactor)
        retained_first = await read_delivery(tx, proposal, first, redactor=app._secret_redactor)
    assert pending.receipt is None and retained_first == accepted_first
    assert await native_stores[1].read_peer_content_attempt(pending.append) is None
    fail_second = False
    # A genuine current receiving-policy refusal is recipient-specific. The
    # already accepted sibling is not redelivered, and the refused attempt
    # retains its original durable identity for later authorized repair.
    policy.denied_consumers.add(second_recipient.participant_id)
    with pytest.raises(CollaborationUnavailable):
        await app.deliver_producer_output(proposal, second.operation, context=context)
    assert await native_stores[1].read_peer_content_attempt(pending.append) is None
    assert (
        await app.deliver_producer_output(proposal, first.operation, context=context)
        == accepted_first
    )
    async with native_stores[2]()._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        refused = await read_delivery(tx, proposal, second, redactor=app._secret_redactor)
    assert refused.receipt is None
    assert refused.operation == pending.operation and refused.append == pending.append
    policy.denied_consumers.clear()
    assert (
        await app.deliver_producer_output(proposal, first.operation, context=context)
        == accepted_first
    )
    with monkeypatch.context() as receiving_clock:
        if second_expires:
            # Expire only this destination, at the native queue transaction's
            # authoritative clock. The sibling's later deadline is still live.
            fixed_now = datetime.fromtimestamp(second.attempt.deadline_at_ms / 1000, UTC)
            if native_stores[3][0] == "postgres":

                async def store_time(_cur):
                    return fixed_now

                receiving_clock.setattr(native_stores[1], "_session_store_now", store_time)
            else:
                receiving_clock.setattr(native_stores[1], "_ownership_clock", lambda: fixed_now)
        accepted_second = await app.deliver_producer_output(
            proposal, second.operation, context=context
        )
    assert accepted_second.receipt.status == ("excluded" if second_expires else "appended")
    if second_expires:
        assert accepted_second.receipt.reason == "delivery_deadline_expired"
    assert (
        await app.deliver_producer_output(proposal, first.operation, context=context)
        == accepted_first
    )
    assert (
        await app.deliver_producer_output(proposal, second.operation, context=context)
        == accepted_second
    )
    assert (
        accepted_second.operation == pending.operation and accepted_second.append == pending.append
    )
    assert entered == [
        first.recipient.participant_id,
        second_recipient.participant_id,
        second_recipient.participant_id,
        second_recipient.participant_id,
    ]
    assert len(provider.requests) == 1
    from tests.core.producer_broadcast_cleanup import settle_broadcast

    await settle_broadcast(
        app,
        proposal,
        resolver,
        context,
        completion,
        provider,
        deliveries=(accepted_first, accepted_second),
    )
