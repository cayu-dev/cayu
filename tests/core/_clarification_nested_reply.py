"""Accept a nested response from its actual completed provider turn."""

from cayu import ClarificationReplyRequest
from cayu.collaboration._contracts import ObjectRef, OwnerRef
from cayu.collaboration._request_coordinator import _initiator
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.exports import SessionExportRef
from cayu.collaboration.mandates import ResourceSelector
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._model_completion_publication import model_step_publication_from_checkpoint


async def accept_nested_reply(
    app,
    initialized,
    accepted,
    question,
    service,
    target,
    actor,
    context,
    export_template,
    payloads,
):
    checkpoint = await runtime_checkpoint_session_store(app.session_store).load_checkpoint(
        target.id
    )
    pointer = model_step_publication_from_checkpoint(checkpoint)
    assert pointer is not None
    resource = ResourceSelector(
        resource=ObjectRef(
            owner=initialized.owner,
            kind="session_transcript_row",
            object_id=target.id,
            incarnation=target.instance_id,
            revision=pointer.source_transcript_cursor + 1,
        )
    )
    audience = OwnerRef(
        application_scope=initialized.owner.application_scope,
        owner_id=accepted.expected.intent.selection.recipient.reference.participant_id,
        incarnation=accepted.expected.intent.selection.recipient.reference.incarnation,
    )
    resolution = actor.resolution
    entry = resolution.chain.entries[-1]
    actor.resolution = resolution.model_copy(
        update={
            "principal": resolution.principal.model_copy(
                update={"audiences": (initialized.owner, audience)}
            ),
            "chain": resolution.chain.model_copy(
                update={
                    "entries": (
                        entry.model_copy(
                            update={
                                "resources": (resource,),
                                "audiences": (initialized.owner, audience),
                            }
                        ),
                    )
                }
            ),
        }
    )
    count = len(payloads)
    try:
        namespace = await app.initialize_session_exports(target.id, context=context)
        export = export_template.model_copy(
            update={
                "ref": SessionExportRef(
                    session_id=target.id,
                    session_instance_id=target.instance_id,
                    operation=initialized.operation("nested-reply-export").model_copy(
                        update={
                            "namespace_incarnation": namespace.namespace_incarnation,
                            "generation": namespace.generation,
                        }
                    ),
                ),
                "source_indices": (pointer.source_transcript_cursor,),
                "audience": audience,
            }
        )
        await app.export_session(export, context=context)
        request = ClarificationReplyRequest(
            operation=initialized.operation("nested-reply"),
            initiator=_initiator(actor.context),
            expected=accepted.expected,
            service=service,
            source=export,
            production_stage_id=pointer.stage_id,
            expected_input_revision=0,
            expected_input_sha256=question.input_sha256,
        )
        result = await app.reply_to_clarification(request, context=context)
        assert result.input_revision == 1
        assert await app.reply_to_clarification(request, context=context) == result
    finally:
        actor.resolution = resolution
    assert len(payloads) == count
    collaboration, _ = app._participant_coordinator._ready()
    async with collaboration._transaction(initialized.owner.application_scope, write=False) as tx:
        lineage = await tx.get("clarification_lineages", operation_key(question.lineage))
        assert lineage["usage"]["questions"] == lineage["usage"]["service_turns"] == 2
        # Parent question and its still-admitted service remain owned. Accepting
        # the child reply cannot discharge either of these independent debts.
        assert lineage["usage"]["pending"] == 2
