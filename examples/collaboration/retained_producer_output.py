"""Explicit host steps for one registered producer and one retained answer.

The host first obtains genuine request-only FRESH admission with a stop
cancellation disposition, builds ProducerOutputProposal,
and calls app.prepare_producer_output(proposal, execution, context=...). Retain
that exact returned command before registration. No function here creates a
policy, grants access, schedules recovery, or resolves a replacement operation.
"""

from contextlib import aclosing

from cayu import (
    CayuApp,
    CollaborationAccessContext,
    MandateAccessContext,
    ProducerCompletionRecord,
    ProducerDeliveryRecord,
    ProducerOutputRegistration,
    SessionExportAccessContext,
)
from cayu.collaboration._contracts import OperationRef
from cayu.sessions import ParticipantSessionExecutionRequest


async def produce_once(
    app: CayuApp,
    command: ProducerOutputRegistration,
    execution: ParticipantSessionExecutionRequest,
    *,
    access: CollaborationAccessContext,
    mandate: MandateAccessContext,
) -> ProducerCompletionRecord:
    """Start new work explicitly; recovery must not call this to rerun a producer.

    Closing or abandoning the event stream does not prove external quiescence.
    On interruption retain the original command, discover pending responsibility,
    and reconcile it through the registered owners.
    """
    await app.register_producer_output(command, execution, context=mandate)
    async with aclosing(
        app.execute_producer_output(command, execution, context=access, producer_context=mandate)
    ) as events:
        async for _event in events:
            pass
    return await app.retain_producer_completion(command, context=access)


async def deliver_answer(
    app: CayuApp,
    command: ProducerOutputRegistration,
    destination: OperationRef,
    *,
    disclosure: SessionExportAccessContext,
) -> ProducerDeliveryRecord:
    """Service a retained answer, not a failure, human pause, or replacement answer.

    Each step authenticates current disclosure independently. Exact retry after
    lost acknowledgement uses these same identities. An appended receipt does
    not imply provider exposure, execution settlement, or business acceptance.
    """
    await app.export_producer_output(command, destination, context=disclosure)
    await app.publish_producer_outcome(command, destination=destination, context=disclosure)
    return await app.deliver_producer_output(command, destination, context=disclosure)
