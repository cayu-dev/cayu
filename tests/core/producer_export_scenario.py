"""Genuine admission/native production followed by authorized visible-text export."""

from tests.core.test_participant_identity import CONTEXT
from tests.core.test_producer_output_contracts import output_scenario

from cayu import ProducerOutputProposal
from cayu.collaboration._contracts import ObjectRef, OwnerRef
from cayu.collaboration.exports import SessionExportAccessContext
from cayu.collaboration.mandates import ResourceSelector
from cayu.providers.base import ModelStreamEvent


async def completed_export_scenario(
    native_stores,
    monkeypatch,
    *,
    private_text=None,
    visible_text="retained answer",
    output_bytes=1024,
    consumer_origin=None,
    operation_prefix="",
    budget_ledger=None,
    planned=False,
    requested_session_id=None,
    budget_binding_factory=None,
):
    values = await output_scenario(
        native_stores,
        with_exports=True,
        consumer_origin=consumer_origin,
        operation_prefix=operation_prefix,
        budget_ledger=budget_ledger,
        planned=planned,
        requested_session_id=requested_session_id,
        budget_binding_factory=budget_binding_factory,
    )
    app, resolver, _admission, provider, session, initialized, proposal, execution = values
    proposal = proposal.model_copy(
        update={"limits": proposal.limits.model_copy(update={"output_bytes": output_bytes})}
    )
    values = (*values[:6], proposal, execution)
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    monkeypatch.setattr(app._session_export_coordinator.owners, "observation_timeout", 60)
    prepared = await app.prepare_producer_output(
        ProducerOutputProposal(
            operation=proposal.operation,
            admission=proposal.admission,
            binding_incarnation=proposal.binding_incarnation,
            limits=proposal.limits,
            destinations=proposal.destinations,
        ),
        execution,
        context=resolver.recipient.context,
    )
    assert prepared == proposal
    proposal = prepared
    await app.register_producer_output(proposal, execution, context=resolver.recipient.context)
    original = resolver.recipient.resolution
    actions = tuple(
        dict.fromkeys((*original.principal.actions, "execute", "source", "expose", "publish"))
    )
    destination = proposal.destinations[0]
    audiences = (
        initialized.owner,
        OwnerRef(
            application_scope=initialized.owner.application_scope,
            owner_id=destination.recipient.participant_id,
            incarnation=destination.recipient.incarnation,
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
    resolver.recipient.resolution = original.model_copy(
        update={
            "principal": original.principal.model_copy(
                update={"actions": actions, "audiences": audiences}
            ),
            "chain": original.chain.model_copy(
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
                        for entry in original.chain.entries
                    )
                }
            ),
        }
    )
    provider._batches = (
        (
            *((ModelStreamEvent.thinking(private_text),) if private_text is not None else ()),
            ModelStreamEvent.text_delta(visible_text),
            ModelStreamEvent.completed(),
        ),
    )
    async for _ in app.execute_producer_output(
        proposal,
        execution,
        context=CONTEXT,
        producer_context=resolver.recipient.context,
    ):
        pass
    completion = await app.retain_producer_completion(proposal, context=CONTEXT)
    assert completion.output.disposition == "answer" and len(provider.requests) == 1
    context = SessionExportAccessContext(
        principal=resolver.recipient.context.principal, mandate=resolver.recipient.context
    )
    return values, context
