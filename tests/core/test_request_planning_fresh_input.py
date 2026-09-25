"""Real clarification reply races a prepared child, not a fabricated input update."""

import asyncio
from uuid import uuid4

import httpx
import pytest
from examples.collaboration.planning import fresh_policy
from tests.artifacts.test_resource_transfer_templates import registered_template
from tests.core.test_builtin_tools import TINY_PNG_BYTES
from tests.core.test_clarification_public import (
    test_public_question_uses_real_assistant_export as run_public_question,
)
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_request_planning_contracts import _policy
from tests.core.test_request_planning_public import complete_plan
from tests.core.test_request_planning_resource_creation import resource_environment

from cayu import AgentSpec, Message, RunRequest
from cayu.artifacts import ArtifactScope, LocalArtifactStore
from cayu.artifacts.attachments import file_attachment
from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import CollaborationConflict, ExactMatch
from cayu.collaboration.planning import (
    RequestPlanningControl,
    RequestPlanningPredecessor,
    RequestPlanningRequest,
    planning_policy_commitment,
)
from cayu.collaboration.request_access import PreparedAdmissionRegistration
from cayu.collaboration.requests import RequestAdmissionCommand
from cayu.collaboration.resource_preparation import RequestPlanningResource
from cayu.messages import FilePart
from cayu.providers.openai import HttpxOpenAITransport, OpenAIProvider
from cayu.sessions.context_views import RecipientSessionCreationRequest
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("with_resource", [False, True])
async def test_fresh_child_commit_cannot_admit_superseded_clarification_input(
    backend, tmp_path, request, monkeypatch, with_resource
):
    # PostgreSQL variants share the module database. Creation keys are global
    # there, but must remain stable within this scenario's retries and recovery.
    creation_identity = uuid4().hex
    planners = []
    release = asyncio.Event()
    pending = []
    clients = []
    resource_owners = ()

    async def journey(
        *,
        application_for,
        collaboration,
        sessions,
        initialized,
        command,
        source,
        context,
        payloads,
        reopen_collaboration,
        backend,
    ):
        nonlocal resource_owners
        receiver = PreparedAdmissionRegistration(receiver=command.question.receiver)
        recipes = ()
        input_parts = ()
        artifacts = None
        if with_resource:
            artifacts = LocalArtifactStore(tmp_path / "clarification-input-artifacts")
            artifact = await artifacts.put_bytes(
                TINY_PNG_BYTES,
                filename="input.png",
                content_type="image/png",
                scope=ArtifactScope.ENVIRONMENT,
                environment_name="resource",
            )
            _, _, permit, destination, template, reader, _, _, _ = await registered_template(
                tmp_path / "clarification-input-owner",
                artifacts,
                artifact,
                planning_deadline=command.expected.intent.selection.expires_at_ms,
                collaboration_store=collaboration,
                scope=initialized.owner.application_scope,
                participant_reference=command.expected.intent.selection.recipient.reference,
            )
            resource_owners = (destination,)
            recipes = (
                RequestPlanningResource(
                    acquisition_permit=permit,
                    transfer=template,
                    transfer_permit=reader._responsibilities[-1][1],
                ),
            )
            input_parts = (
                FilePart(
                    attachment=file_attachment(
                        artifact_id=artifact.id,
                        kind="image",
                        filename=artifact.filename,
                        content_type=artifact.content_type,
                        size_bytes=artifact.size_bytes,
                    )
                ),
            )

        def forbid_dispatch(_request):
            pytest.fail("Preparation/admission must not dispatch a recipient provider")

        transport = HttpxOpenAITransport()
        client = httpx.AsyncClient(transport=httpx.MockTransport(forbid_dispatch))
        clients.append(client)
        transport._client._client = client
        provider = OpenAIProvider(
            api_key="test-key",
            streaming=False,
            transport=transport,
        )

        def planner_for(policies=()):
            planner = application_for(
                collaboration,
                sessions,
                planning_policies=policies,
                prepared_admission=receiver,
                resource_owners=resource_owners,
            )

            planner.register_provider(provider, default=True)
            planner.register_agent(AgentSpec(name="reviewer", model="gpt-test"))
            if artifacts is not None:
                planner.register_environment(resource_environment(artifacts), default=True)
            planners.append(planner)
            return planner

        initial = planner_for()
        await initial.initialize_collaboration()
        # The shared journey leaves this ordinary clarify admission to its
        # planning callback. Its real question/export/service/reply remain owned
        # by the existing public clarification machinery.
        await initial.admit_collaboration_request(
            RequestAdmissionCommand(
                operation=command.question.admission,
                expected=command.expected,
                expected_revision=command.expected_revision - 1,
                expected_input_revision=command.question.input_revision,
                expected_input_sha256=command.question.input_sha256,
                generation=1,
                decision="clarify",
                evidence=(),
                initiator=command.question.initiator,
            ),
            context=context.mandate,
        )
        opened = await initial.open_clarification(command, source=source, context=context)
        snapshot = await initial.inspect_collaboration_request(
            command.expected, context=context.mandate
        )
        creation = RecipientSessionCreationRequest(
            request=RunRequest(
                agent_name="reviewer",
                messages=[
                    Message.text("user", "Review API"),
                    *([Message(role="user", content=input_parts)] if input_parts else []),
                ],
            ),
            creation_key="fresh-before-reply:" + creation_identity,
            recipient=command.expected.intent.selection.recipient.reference,
        )
        preparation = await initial.prepare_recipient_creation(creation, context=CONTEXT)
        policy = fresh_policy(
            _policy().reference.model_copy(update={"owner": initialized.owner}),
            _policy().limits,
            preparation,
            resources=recipes,
        )
        planner = planner_for((policy,))
        await planner.initialize_collaboration()
        planned = RequestPlanningRequest(
            operation=initialized.operation("fresh-before-reply-plan"),
            expected=command.expected,
            expected_revision=snapshot.revision,
            expected_input_revision=snapshot.clarification.input_revision,
            expected_input_sha256=snapshot.clarification.input_sha256
            or clarification_commitment(command.expected, SecretRedactor()),
            planning_generation=1,
            admission_operation=initialized.operation("fresh-before-reply-admission"),
            admission_generation=snapshot.admission_generation + 1,
            initiator=command.question.initiator,
            policy=policy.reference,
            policy_sha256=planning_policy_commitment(policy, redactor=SecretRedactor()),
            limits=policy.limits,
            deadline_at_ms=command.expected.intent.selection.expires_at_ms,
            predecessor=None,
        )
        entered = asyncio.Event()
        native_create = planner.create_recipient_session
        child = None
        committed_creation = None

        async def hold_created(*args, **kwargs):
            nonlocal child, committed_creation
            committed_creation = args[0]
            child = await native_create(*args, **kwargs)
            entered.set()
            await release.wait()
            return child

        monkeypatch.setattr(planner, "create_recipient_session", hold_created)
        planner._request_coordinator._owners.observation_timeout = 300
        task = asyncio.create_task(complete_plan(planner, planned, context.mandate))
        pending.append(task)
        barrier = asyncio.create_task(entered.wait())
        try:
            done, _ = await asyncio.wait(
                (barrier, task), timeout=120, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                await task
                pytest.fail("Planning ended before committing the recipient")
            assert barrier in done, "Recipient creation did not reach its commit barrier"
        finally:
            barrier.cancel()
            await asyncio.gather(barrier, return_exceptions=True)

        async def after_reply(*, current, reply, wait, parked, wait_context):
            before_calls = len(payloads)
            updated = await current.inspect_collaboration_request(
                command.expected, context=context.mandate
            )
            assert updated.clarification.input_revision == reply.input_revision == 1
            native_wait = await sessions.load_continuation_ticket(
                parked.ticket.session_id,
                session_instance_id=parked.ticket.session_instance_id,
                registration_key=parked.ticket.registration_key,
            )
            assert native_wait is not None and native_wait.ticket.state == "WAITING"
            release.set()
            with pytest.raises(CollaborationConflict):
                await task
            retained = await planner.lookup_collaboration_plan(planned, context=context.mandate)
            assert isinstance(retained, ExactMatch)
            assert retained.receipt.state == "preparing" and retained.receipt.pending_stages == 0
            assert retained.receipt.stage_count == (3 if with_resource else 1)
            assert committed_creation is not None
            if with_resource:
                # Native creation binds the material transfer into the exact
                # request; the pre-resolution request is not an exact replay.
                with pytest.raises(ValueError, match="conflicts"):
                    await planner.lookup_recipient_session(creation, context=CONTEXT)
            assert (
                await planner.lookup_recipient_session(committed_creation, context=CONTEXT) == child
            )
            cancelled = await complete_plan(
                planner,
                RequestPlanningControl(
                    expected=planned,
                    expected_revision=retained.receipt.revision,
                    kind="cancelled",
                    initiator=planned.initiator,
                ),
                context.mandate,
                control=True,
            )
            refreshed_creation = RecipientSessionCreationRequest(
                request=RunRequest(
                    agent_name="reviewer", messages=[Message.text("user", "Use API v2.")]
                ),
                creation_key="fresh-after-accepted-reply:" + creation_identity,
                recipient=creation.recipient,
            )
            refreshed = await initial.prepare_recipient_creation(
                refreshed_creation, context=CONTEXT
            )
            next_policy = fresh_policy(
                policy.reference.model_copy(update={"revision": 2}), policy.limits, refreshed
            )
            second = planner_for((next_policy,))
            await second.initialize_collaboration()
            successor = planned.model_copy(
                update={
                    "operation": initialized.operation("fresh-after-reply-plan"),
                    "admission_operation": initialized.operation("fresh-after-reply-admission"),
                    "planning_generation": 2,
                    "expected_revision": updated.revision,
                    "expected_input_revision": updated.clarification.input_revision,
                    "expected_input_sha256": updated.clarification.input_sha256,
                    "policy": next_policy.reference,
                    "policy_sha256": planning_policy_commitment(
                        next_policy, redactor=SecretRedactor()
                    ),
                    "predecessor": RequestPlanningPredecessor(
                        operation=planned.operation,
                        revision=cancelled.revision,
                    ),
                }
            )
            admitted = await complete_plan(second, successor, context.mandate)
            assert admitted.state == "admitted" and admitted.pending_stages == 0
            assert admitted.receipt.command.expected_input_revision == 1
            assert (
                admitted.receipt.command.expected_input_sha256 == updated.clarification.input_sha256
            )
            assert (
                await planner.lookup_recipient_session(committed_creation, context=CONTEXT) == child
            )
            assert (
                await sessions.load_continuation_ticket(
                    parked.ticket.session_id,
                    session_instance_id=parked.ticket.session_instance_id,
                    registration_key=parked.ticket.registration_key,
                )
                == native_wait
            )
            assert len(payloads) == before_calls

        return opened, after_reply

    try:
        await run_public_question(
            backend,
            tmp_path,
            request,
            monkeypatch,
            True,
            True,
            False,
            False,
            planning_journey=journey,
            # This is multi-owner integration, not a deadline-boundary test.
            journey_ttl_ms=900_000 if with_resource else 300_000,
        )
    finally:
        release.set()
        await asyncio.gather(*pending, return_exceptions=True)
        for planner in planners:
            await planner.drain_collaboration_requests()
            await planner.drain_session_exports()
        for client in clients:
            await client.aclose()
        for owner in resource_owners:
            await owner.drain()
