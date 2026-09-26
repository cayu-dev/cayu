"""Genuine FRESH producer publication into an already servicing clarification wait."""

from dataclasses import replace

from tests.core._clarification_final_flow import deliver_original_final_wait
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import app as make_app
from tests.core.test_peer_content import QualifiedPeerProvider

from cayu import AgentSpec, ProducerOutputProposal
from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import ObjectRef, OwnerRef
from cayu.collaboration._producer_acceptance import ProducerOutputAcceptanceReader
from cayu.collaboration._producer_contracts import ProducerDeliveryDestination, ProducerOutputLimits
from cayu.collaboration._request_coordinator import _initiator
from cayu.collaboration.exports import SessionExportAccessContext
from cayu.collaboration.mandates import ResourceSelector
from cayu.collaboration.peer_content import PeerAppendKey, PeerDeliveryAttemptKey
from cayu.collaboration.request_access import PreparedAdmissionRegistration
from cayu.collaboration.requests import RequestAdmissionCommand
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import RunRequest
from cayu.sessions.context_views import (
    ParticipantSessionExecutionRequest,
    RecipientSessionCreationRequest,
)


async def finish_with_native_producer(
    original_app,
    *,
    accepted,
    initialized,
    actor_a,
    actor_b,
    wait,
    parked,
    payloads,
    context,
    consume=True,
    **_unused,
):
    source_store, _ = original_app._participant_coordinator._ready()
    sessions = original_app.session_store
    registration = original_app._request_coordinator._registration
    exports = original_app._session_export_coordinator.registration
    recipient = actor_b.context.participant
    consumer = actor_a.context.participant
    audience = OwnerRef(
        application_scope=consumer.owner.application_scope,
        owner_id=consumer.participant_id,
        incarnation=consumer.incarnation,
    )
    receiver = ObjectRef(
        owner=initialized.owner,
        kind="request_receiver",
        object_id="clarification-final-producer",
        incarnation="one",
        revision=1,
    )
    production = make_app(
        source_store,
        original_app._participant_coordinator._registration,
        session_store=sessions,
        budget_ledger=original_app._run_limit_controller._budget_ledger,
        budget_binding_receiver=original_app._run_limit_controller._budget_binding_receiver,
        enable_common_root_budget_binding=True,
        collaboration_requests=replace(
            registration, prepared_admission=PreparedAdmissionRegistration(receiver=receiver)
        ),
        session_exports=replace(
            exports,
            readers=(
                ProducerOutputAcceptanceReader(
                    collaboration_store=source_store,
                    session_store=sessions,
                    namespace=initialized.operation("final-producer-reader"),
                    audience=audience,
                ),
            ),
        ),
    )
    provider = QualifiedPeerProvider(
        [[ModelStreamEvent.text_delta("Original final answer"), ModelStreamEvent.completed()]],
        name="openai",
    )
    production.register_provider(provider, default=True)
    production.register_agent(AgentSpec(name="reviewer", model="gpt-test"))
    production._request_coordinator._owners.observation_timeout = 60
    production._session_export_coordinator.owners.observation_timeout = 60
    prior_authority = actor_b.resolution
    actions = tuple(
        dict.fromkeys(
            (
                *prior_authority.principal.actions,
                "prepare",
                "execute",
                "publish",
                "source",
                "expose",
            )
        )
    )
    audiences = (initialized.owner, audience)
    actor_b.resolution = prior_authority.model_copy(
        update={
            "principal": prior_authority.principal.model_copy(
                update={"actions": actions, "audiences": audiences}
            ),
            "chain": prior_authority.chain.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(update={"actions": actions, "audiences": audiences})
                        for entry in prior_authority.chain.entries
                    )
                }
            ),
        }
    )
    try:
        assert await production.initialize_collaboration() == initialized
        creation = RecipientSessionCreationRequest(
            creation_key="original-final-producer",
            recipient=recipient,
            request=RunRequest(
                agent_name="reviewer",
                messages=[Message.text("user", "Complete the original request")],
            ),
        )
        session, _ = await production.create_recipient_session(creation, context=CONTEXT)
        prepared = await production.prepare_recipient_admission(creation, context=CONTEXT)
        prior = await production.inspect_collaboration_request(
            accepted.expected, context=actor_a.context
        )
        admission = RequestAdmissionCommand(
            operation=initialized.operation("original-final-production-admission"),
            expected=accepted.expected,
            expected_revision=prior.revision,
            expected_input_revision=prior.clarification.input_revision,
            expected_input_sha256=prior.clarification.input_sha256
            or clarification_commitment(accepted.expected, production._secret_redactor),
            generation=prior.admission_generation + 1,
            decision="fresh",
            prepared=prepared,
            evidence=(),
            initiator=_initiator(actor_b.context),
        )
        await production.admit_collaboration_request(admission, context=actor_b.context)
        target = await sessions.load(parked.ticket.session_id)
        transcript = await sessions.load_transcript_snapshot(target.id)
        original = accepted.expected.intent.request
        destination = ProducerDeliveryDestination(
            operation=initialized.operation("original-final-destination"),
            recipient=consumer,
            attempt=PeerDeliveryAttemptKey(
                append_key=PeerAppendKey(
                    collaboration_namespace=admission.operation.namespace_incarnation,
                    collaboration_generation=admission.operation.generation,
                    occurrence_id="original-final-result",
                    consumer_id=consumer.participant_id,
                    consumer_participant_incarnation=consumer.incarnation,
                    projection_id="text",
                    projection_schema="visible-text.v1",
                    target_session_id=target.id,
                    target_session_instance_id=target.instance_id,
                ),
                interest_id="original-request",
                attempt_generation=1,
                target_run_epoch=target.run_epoch,
                target_transcript_cursor=transcript.cursor,
                withdrawal_generation=1,
                deadline_at_ms=accepted.expected.intent.selection.expires_at_ms,
            ),
            projector=original.output_contract,
            validator=original.output_contract,
            disclosure_policy=original.disclosure_policy,
            mandate=actor_b.context.mandate,
        )
        execution = ParticipantSessionExecutionRequest(
            request=creation.request.model_copy(update={"session_id": session.id}),
            session_instance_id=session.instance_id,
            execution_key="original-final-native-execution",
        )
        command = await production.prepare_producer_output(
            ProducerOutputProposal(
                operation=initialized.operation("original-final-output"),
                admission=admission,
                binding_incarnation="one",
                limits=ProducerOutputLimits(
                    output_bytes=1024,
                    progress_occurrences=4,
                    destinations=1,
                    deadline_at_ms=destination.attempt.deadline_at_ms,
                ),
                destinations=(destination,),
            ),
            execution,
            context=actor_b.context,
        )
        await production.register_producer_output(command, execution, context=actor_b.context)
        events = [
            event
            async for event in production.execute_producer_output(
                command, execution, context=CONTEXT, producer_context=actor_b.context
            )
        ]
        assert any(event.type == "session.completed" for event in events), events
        assert len(provider.requests) == 1
        completion = await production.retain_producer_completion(command, context=CONTEXT)
        assert completion.output.disposition == "answer"
        resolution = actor_b.resolution
        resources = tuple(
            ResourceSelector(
                resource=ObjectRef(
                    owner=initialized.owner,
                    kind="session_transcript_row",
                    object_id=session.id,
                    incarnation=session.instance_id,
                    revision=index + 1,
                )
            )
            for index in completion.output.source_indices
        )
        actor_b.resolution = resolution.model_copy(
            update={
                "chain": resolution.chain.model_copy(
                    update={
                        "entries": tuple(
                            entry.model_copy(update={"resources": resources})
                            for entry in resolution.chain.entries
                        )
                    }
                )
            }
        )
        disclosure = SessionExportAccessContext(
            principal=actor_b.context.principal, mandate=actor_b.context
        )
        await production.export_producer_output(command, destination.operation, context=disclosure)
        outcome = await production.publish_producer_outcome(
            command, destination=destination.operation, context=disclosure
        )
        assert outcome.command.outcome == "answered"
        assert (
            await production.publish_producer_outcome(
                command, destination=destination.operation, context=disclosure
            )
            == outcome
        )
        # Independent production must not invalidate the original participant.
        inspection = await original_app.inspect_participant(consumer, context=CONTEXT)
        assert inspection.participant.reference == consumer
        return await deliver_original_final_wait(
            original_app,
            initialized=initialized,
            actor_a=actor_a,
            wait=wait,
            parked=parked,
            payloads=payloads,
            context=context,
            consume=consume,
        )
    finally:
        actor_b.resolution = prior_authority
        # Request coordinators share their CollaborationStore's mutation owner.
        # The outer fixture drains it after the original service has returned;
        # draining here would close the store while that service still owns work.
