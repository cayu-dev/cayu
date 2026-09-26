"""Original final wait settlement through source-owned export and real latch delivery."""

import asyncio
from datetime import UTC, datetime

import pytest

from cayu.collaboration._capabilities import CapabilityDescriptor
from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import CollaborationConflict, ObjectRef, OperationRef
from cayu.collaboration._request_coordinator import _initiator
from cayu.collaboration.exports import SessionExportRef
from cayu.collaboration.mandates import ResourceSelector
from cayu.collaboration.requests import RequestAdmissionCommand, RequestOutcomeCommand
from cayu.messages import Message
from cayu.runtime._session_continuation import (
    ContinuationConflict,
    ContinuationService,
    ContinuationUnavailable,
)
from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
from cayu.sessions.base import ResumeRequest
from cayu.vaults.redaction import SecretRedactor


async def finish_original_wait(
    app,
    *,
    source,
    accepted,
    initialized,
    export_template,
    source_context,
    actor_a,
    actor_b,
    reader,
    wait,
    parked,
    payloads,
    context,
    consume=True,
):
    rows = await app.session_store.load_transcript(source.id)
    index = max(i for i, message in enumerate(rows) if message.role == "assistant")
    selected = ResourceSelector(
        resource=ObjectRef(
            owner=initialized.owner,
            kind="session_transcript_row",
            object_id=source.id,
            incarnation=source.instance_id,
            revision=index + 1,
        )
    )
    # A is explicitly authorized to publish B's actual completed row under the
    # original output contract. This does not change the producing session or
    # copy the text into a new record. B separately authorizes terminal outcome.
    for actor in (actor_a, actor_b):
        resolution = actor.resolution
        entry = resolution.chain.entries[-1]
        actions = tuple(dict.fromkeys((*resolution.principal.actions, "publish", "source")))
        actor.resolution = resolution.model_copy(
            update={
                "principal": resolution.principal.model_copy(update={"actions": actions}),
                "chain": resolution.chain.model_copy(
                    update={
                        "entries": (
                            entry.model_copy(
                                update={
                                    "actions": actions,
                                    "resources": entry.resources
                                    if selected in entry.resources
                                    else (*entry.resources, selected),
                                }
                            ),
                        )
                    }
                ),
            }
        )
    namespace = await app.initialize_session_exports(source.id, context=source_context)
    final_export = export_template.model_copy(
        update={
            "ref": SessionExportRef(
                session_id=source.id,
                session_instance_id=source.instance_id,
                operation=OperationRef(
                    application_scope=initialized.owner.application_scope,
                    namespace_incarnation=namespace.namespace_incarnation,
                    generation=namespace.generation,
                    caller_key="final-result",
                ),
            ),
            "source_indices": (index,),
            "audience": initialized.owner,
        }
    )
    # This export needs only the final row and collaboration-owner audience.
    # Retaining all earlier source grants in its immutable authorization would
    # exceed the export owner's fixed future-settlement publication allowance.
    # Narrow the real policy resolution for this operation, never the receipt.
    prior_resolution = actor_a.resolution
    actor_a.resolution = prior_resolution.model_copy(
        update={
            "principal": prior_resolution.principal.model_copy(
                update={"audiences": (initialized.owner,)}
            ),
            "chain": prior_resolution.chain.model_copy(
                update={
                    "entries": (
                        prior_resolution.chain.entries[-1].model_copy(
                            update={
                                "audiences": (initialized.owner,),
                                "resources": (selected,),
                            }
                        ),
                    )
                }
            ),
        }
    )
    try:
        exported = await app.export_session(final_export, context=source_context)
    finally:
        actor_a.resolution = prior_resolution
    prior = await app.inspect_collaboration_request(accepted.expected, context=actor_a.context)
    reader.permit = prior.permit
    admission = RequestAdmissionCommand(
        operation=initialized.operation("final-admission"),
        expected=accepted.expected,
        expected_revision=prior.revision,
        expected_input_revision=prior.clarification.input_revision,
        expected_input_sha256=prior.clarification.input_sha256
        or clarification_commitment(accepted.expected, SecretRedactor()),
        generation=prior.admission_generation + 1,
        decision="continue",
        source_export=final_export.ref,
        source_receipt=exported,
        evidence=(selected.resource,),
        initiator=_initiator(actor_b.context),
    )
    # Latch arbitration may complete the original request before any reply.
    # Only an accepted clarification makes the initial input tuple stale.
    if prior.clarification.input_revision > 0:
        stale = admission.model_copy(
            update={
                "expected_input_revision": 0,
                "expected_input_sha256": clarification_commitment(
                    accepted.expected, SecretRedactor()
                ),
            }
        )
        with pytest.raises(CollaborationConflict):
            await app.admit_collaboration_request(stale, context=actor_b.context)
        assert (
            await app.inspect_collaboration_request(accepted.expected, context=actor_a.context)
            == prior
        )
    admitted = await app.admit_collaboration_request(admission, context=actor_b.context)
    assert await app.admit_collaboration_request(admission, context=actor_b.context) == admitted
    terminal = RequestOutcomeCommand(
        operation=initialized.operation("final-outcome"),
        expected=accepted.expected,
        expected_revision=admitted.revision,
        outcome="answered",
        commitment=exported.expected.intent.output_commitment,
        source_receipt=exported,
        initiator=_initiator(actor_b.context),
    )
    outcome = await app.publish_collaboration_outcome(terminal, context=actor_b.context)
    assert await app.publish_collaboration_outcome(terminal, context=actor_b.context) == outcome
    return await deliver_original_final_wait(
        app,
        initialized=initialized,
        actor_a=actor_a,
        wait=wait,
        parked=parked,
        payloads=payloads,
        context=context,
        consume=consume,
    )


async def deliver_original_final_wait(
    app, *, initialized, actor_a, wait, parked, payloads, context, consume
):
    """Deliver an already elected final result through the existing latch owner."""
    bound_wait = wait.model_copy(update={"delivery_ticket": parked.preparation.intent})
    elected = await app.observe_collaboration_wait(bound_wait, context=actor_a.context)
    assert elected.state == "elected"
    owner = SessionContinuationOwner(
        store=app.session_store,
        owner=initialized.owner,
        receiver=app.collaboration_wait_latch_receiver(),
        receiver_capability=CapabilityDescriptor(
            owner=initialized.owner, mutations=(), readbacks=(LATCH_FAMILY,)
        ),
        redactor=SecretRedactor(),
    )
    try:
        await app.deliver_collaboration_wait(
            bound_wait,
            context=actor_a.context,
            continuation_owner=owner,
        )
        latched = await app.session_store.load_continuation_ticket(
            parked.ticket.session_id,
            session_instance_id=parked.ticket.session_instance_id,
            registration_key=parked.ticket.registration_key,
        )
        assert latched.latch is not None
        if consume:
            await consume_original_latch(
                app,
                initialized=initialized,
                latched=latched,
                payloads=payloads,
                context=context,
            )
        return latched
    finally:
        await owner.drain()


async def consume_original_latch(
    app, *, initialized, latched, payloads, context, blocked=False, waiting_ticket=None
):
    """A separate receiving owner competes with the temporary service owner."""
    owner = SessionContinuationOwner(
        store=app.session_store,
        owner=initialized.owner,
        receiver=app.collaboration_wait_latch_receiver(),
        receiver_capability=CapabilityDescriptor(
            owner=initialized.owner, mutations=(), readbacks=(LATCH_FAMILY,)
        ),
        redactor=SecretRedactor(),
    )
    try:
        service = ContinuationService(
            # A competing worker may retain the genuine earlier WAITING
            # observation. Admission must reject it after service owns the wait.
            ticket=latched.ticket if waiting_ticket is None else waiting_ticket,
            latch=latched.latch,
            continuation_id="original-final",
            mode="inline",
            accepted_at=datetime.now(UTC).isoformat(),
        )
        resume = ResumeRequest(
            session_id=latched.ticket.session_id,
            messages=[Message.text("user", "The original request has settled; continue.")],
        )
        count = len(payloads)
        with pytest.raises(PermissionError):
            await owner.service(app, resume, service)
        assert len(payloads) == count
        if blocked:
            with pytest.raises(ContinuationConflict):
                await owner.service(app, resume, service, participant_context=context)
            assert len(payloads) == count
            assert (
                await app.session_store.load_continuation_ticket(
                    latched.ticket.session_id,
                    session_instance_id=latched.ticket.session_instance_id,
                    registration_key=latched.ticket.registration_key,
                )
                == latched
            )
            return
        try:
            settled = await owner.service(app, resume, service, participant_context=context)
        except ContinuationUnavailable:
            pending = tuple(owner.owners.pending)
            if not pending:
                raise
            # The foreground deadline is not a failed continuation. Observe the
            # actual retained owner (including any real error), then reconcile
            # the identical service without admitting another continuation.
            await asyncio.wait_for(asyncio.gather(*pending), 60)
            settled = await owner.service(app, resume, service, participant_context=context)
        assert settled.ticket.state == "CONSUMED"
        assert len(payloads) == count + 1
        assert await owner.service(app, resume, service, participant_context=context) == settled
        assert len(payloads) == count + 1
    finally:
        await owner.drain()
