"""Host qualification helpers; native runtime and authorization remain unchanged."""

import asyncio

from tests.core.test_collaboration_waits import wait_for
from tests.core.test_participant_identity import CONTEXT

from cayu.events import EventType
from cayu.messages import Message
from cayu.runtime._session_continuation import continuation_digest
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.sessions.invocation import InvocationOriginClaim
from cayu.sessions.requests import RunRequest


async def create_target(app, accepted, initialized, participant, wait_context, *, park):
    creation = ParticipantSessionCreationRequest(
        creation_key="question-target-" + initialized.owner.application_scope,
        request=RunRequest(
            agent_name="reviewer",
            messages=[Message.text("user", "Wait for the question")],
            invocation_origin=InvocationOriginClaim(subject="operator"),
        ),
    )
    target, _ = await app.create_participant_session(
        creation,
        participant=participant,
        context=CONTEXT,
    )
    wait = None
    parked = None
    if park:
        wait = wait_for(accepted, initialized).model_copy(
            update={
                "service_policy": "clarification",
                "failure_policy": "return_and_report",
            }
        )
        events = [
            event
            async for event in app.execute_participant_session_to_wait(
                ParticipantSessionExecutionRequest(
                    request=creation.request.model_copy(update={"session_id": target.id}),
                    session_instance_id=target.instance_id,
                    execution_key="park-question-target",
                ),
                wait,
                participant=participant,
                context=CONTEXT,
                wait_context=wait_context,
            )
        ]
        assert any(event.type == EventType.SESSION_INTERRUPTED for event in events)
        parked = await app.session_store.load_continuation_ticket(
            target.id,
            session_instance_id=target.instance_id,
            registration_key="execution-wait:" + continuation_digest(wait.operation),
        )
        assert parked is not None and parked.ticket.state == "WAITING"
    return creation, target, wait, parked


class SingleSlotDriver:
    """Bound complete real execution streams, including their awaited cleanup."""

    def __init__(self):
        self.slot = asyncio.Semaphore(1)
        self.order = []
        self.active = 0

    def wrap(self, entrance):
        async def execute(request, *args, **kwargs):
            async with asyncio.timeout(180), self.slot:
                assert self.active == 0
                self.active += 1
                self.order.append(request.session_id)
                stream = entrance(request, *args, **kwargs)
                try:
                    async for event in stream:
                        yield event
                finally:
                    try:
                        await stream.aclose()
                    finally:
                        self.active -= 1

        return execute
