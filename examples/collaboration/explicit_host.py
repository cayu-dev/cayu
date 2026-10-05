"""Credential-free request -> producer -> delivery -> wait continuation.

Run from the repository: ``uv run python -m examples.collaboration.explicit_host``.
The real provider adapter serializes every request, but OfflineResponses performs
no network I/O. The application explicitly chooses policy and disclosure; the
host services the existing durable owners. No model is called to poll a wait.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from examples.collaboration.host_demo_authority import (
    PRINCIPAL,
    Administration,
    DemoMandates,
    ExactDisclosure,
    VisibleText,
    commitment,
    reference,
)
from examples.collaboration.host_demo_transport import OfflineProvider, OfflineResponses

from cayu import (
    AgentSpec,
    BudgetBinding,
    BudgetLimit,
    BudgetReservation,
    CayuApp,
    CollaborationAccessContext,
    CollaborationBootstrap,
    CollaborationHost,
    CollaborationLimits,
    CollaborationRegistration,
    CollaborationRequest,
    CollaborationWait,
    ConfiguredRequestPlanningPolicy,
    ContinuationService,
    HostContinuationRule,
    HostOwnershipLimits,
    HostPlannedProducer,
    HostPlannedProducerRule,
    HostPlanningRule,
    HostProducerDisclosure,
    HostProducerExecution,
    HostProducerExecutionRule,
    HostProducerMaintenance,
    HostProducerMaintenanceRule,
    HostProducerOutputRule,
    HostProducerSource,
    HostRegistration,
    HostWaitRule,
    InMemoryCollaborationStore,
    ParticipantConfiguration,
    ParticipantConfigurationRef,
    ParticipantCreate,
    PreparedAdmissionRegistration,
    ProducerDeliveryDestination,
    ProducerOutputAcceptanceReader,
    ProducerOutputLimits,
    RequestPlanningFresh,
    RequestPlanningLimits,
    RequestPlanningRequest,
    RequestRegistration,
    SessionExportAccessContext,
    SessionExportRegistration,
    planning_policy_commitment,
)
from cayu.budgets.base import BudgetLedger, InMemoryBudgetLedger
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.collaboration._contracts import ExactMatch, InitiatorBinding, ObjectRef, OwnerRef
from cayu.collaboration._host_planning import lookup_host_plan
from cayu.collaboration.base import CollaborationStore
from cayu.collaboration.exports import ExportLimits
from cayu.collaboration.mandates import ResourceSelector
from cayu.collaboration.participants import (
    CollaborationInitialization,
    CollaborationUnavailable,
    ParticipantRef,
)
from cayu.collaboration.peer_content import PeerAppendKey, PeerDeliveryAttemptKey
from cayu.messages import Message
from cayu.sessions import ResumeRequest, RunRequest
from cayu.sessions.base import InMemorySessionStore, SessionStatus, SessionStore
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
    RecipientSessionCreationRequest,
)
from cayu.sessions.invocation import InvocationOriginClaim
from cayu.vaults.redaction import SecretRedactor

ACCESS = CollaborationAccessContext(principal=PRINCIPAL)
SERVICE_TIMEOUT_S = 300
JOURNEY_TIMEOUT_S = 900
# Two finite questions retain their first round's evidence while admitting the
# next round (or reserve both rounds concurrently). Capacity includes native
# worst-case settlement envelopes, not merely the final JSON document sizes.
# This does not change the shared monetary ceiling or dispatch allowance.
TEAM_RETAINED_BYTES = 9 * 1024 * 1024
# Two parked waits reserve more than 256 native observation/settlement events
# before either journey finishes. Keep room for their production and delivery
# evidence as well as the eight control events; this is durable owner capacity,
# not a larger model-call allowance or a host execution-slot limit.
TEAM_EVENTS = 512


class ExampleHostFailure(RuntimeError):
    """Keep the failed host reachable for explicit inspection and draining.

    The message is content-free; retaining a host is not a settlement receipt.
    Callers must not close its shared stores while it still owns pending work.
    """

    def __init__(self, host):
        super().__init__("Example host requires owner reconciliation.")
        self.host = host


def demo_agent(*, request_key=None):
    return AgentSpec(
        name="demo-agent",
        model="gpt-5.6",
        system_prompt="" if request_key is None else "DEMO_REQUEST:" + request_key,
        provider_options={"openai": {"max_output_tokens": 256}},
    )


async def service_until(app, registration, inspect, ready, *, timeout_s=None):
    """Finite application wait; native readback, not host counters, proves completion."""
    timeout_s = SERVICE_TIMEOUT_S if timeout_s is None else timeout_s
    if type(timeout_s) not in (int, float) or not 0 < timeout_s <= JOURNEY_TIMEOUT_S:
        raise ValueError("Example observation requires a bounded journey interval.")
    host = CollaborationHost(app, registration)
    observation = _ExampleObservation(inspect)
    try:
        return await _service_until_owned(host, observation, ready, timeout_s=timeout_s)
    except BaseException as error:
        # Preserve the original exception/control signal, including ordinary
        # Task.cancel() handling, while making its retained owner reachable.
        # The handle is never formatted into diagnostics or treated as cleanup.
        error.__dict__["collaboration_host"] = host
        error.__dict__["collaboration_observation"] = observation
        raise


class _ExampleObservation:
    """One retained business read, independent of the servicing observer.

    A cancelled caller cannot prove a store read stopped. Keep its task and
    original outcome reachable alongside the host instead of cancelling it or
    launching overlapping readbacks. This does not confer mutation authority.
    """

    def __init__(self, inspect):
        self.inspect = inspect
        self.task = None

    def start(self):
        if self.task is not None:
            raise RuntimeError("Example observation is already retained.")

        async def observe():
            try:
                return await self.inspect(), None
            except BaseException as error:
                return None, error

        self.task = asyncio.create_task(observe(), name="cayu-example-observation")


async def _service_until_owned(host, observation, ready, *, timeout_s):
    async with host, asyncio.timeout(timeout_s):
        next_inspection = 0.0
        while True:
            state = await host.service_once()
            if state.failed or state.source_failures:
                raise ExampleHostFailure(host)
            # This is a per-pass count, not a monotonic progress generation.
            # Two consecutive passes can each complete one different operation.
            # Other hosts may commit progress without this host servicing any
            # operation. Periodic owner readback also covers that case.
            # Business completion need not coincide with a locally idle pass:
            # another source read or unrelated maintenance can always be due.
            # Retain one observation while continuing to pump the host, so a
            # slow read cannot prevent dispatch-window renewal or servicing.
            if observation.task is None and asyncio.get_running_loop().time() >= next_inspection:
                # Completion belongs to the exact question, not local host
                # idleness. A wait observation or unrelated maintenance may
                # occupy every pass even after this question has settled.
                # Keep just one read in flight while servicing continues. A
                # nonzero serviced count can remain visible for multiple polls
                # of an in-flight pass; it must not bypass the read interval.
                observation.start()
            if observation.task is not None and observation.task.done():
                evidence, error = observation.task.result()
                observation.task = None
                if error is not None:
                    if not isinstance(error, CollaborationUnavailable):
                        raise error
                else:
                    if ready(evidence):
                        break
                next_inspection = asyncio.get_running_loop().time() + 1.0
            await asyncio.sleep(0.01)
    if host.inspect().pending:
        raise ExampleHostFailure(host)
    return evidence


def host_registration(**rules):
    return HostRegistration(
        limits=HostOwnershipLimits(1, 1, 4, 524288),
        producer_sources=rules.pop("producer_sources", ()),
        producer_rules=rules.pop("producer_rules", ()),
        observation_timeout_s=30,
        shutdown_timeout_s=30,
        **rules,
    )


@dataclass(frozen=True, slots=True)
class DemoTeam:
    """One application-owned team sharing native stores and one budget binding."""

    scope: str
    initialized: CollaborationInitialization
    participants: tuple[ParticipantRef, ...]
    collaboration: CollaborationStore
    sessions: SessionStore
    ledger: BudgetLedger
    contract: ObjectRef
    mandates: DemoMandates
    disclosure: ExactDisclosure
    transport: OfflineResponses
    application: Callable[..., CayuApp]


async def create_demo_team(
    *, roles=("coordinator", "specialist"), collaboration=None, sessions=None, ledger=None
):
    """Register a finite team without executing a model or starting a host."""
    if (
        type(roles) is not tuple
        or not 2 <= len(roles) <= 8
        or any(
            type(role) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", role) is None
            for role in roles
        )
        or len(set(roles)) != len(roles)
    ):
        raise ValueError("Example roles must be two to eight distinct bounded names.")
    collaboration = collaboration if collaboration is not None else InMemoryCollaborationStore()
    sessions = sessions if sessions is not None else InMemorySessionStore()
    ledger = ledger if ledger is not None else InMemoryBudgetLedger()
    scope = "host-example-" + uuid4().hex
    configuration = ParticipantConfiguration(
        definition=ParticipantConfigurationRef(name="demo-agent", version=1),
        routing=ParticipantConfigurationRef(name="explicit", version=1),
        admission=ParticipantConfigurationRef(name="fresh", version=1),
    )
    registration = CollaborationRegistration(
        bootstrap=CollaborationBootstrap(
            application_scope=scope,
            provisioning_scope="example",
            owner_name="participants",
            limits=CollaborationLimits(
                participants=8,
                aliases=8,
                operations=128,
                events=TEAM_EVENTS,
                retained_bytes=TEAM_RETAINED_BYTES,
                control_operations=8,
                control_events=8,
                control_bytes=65536,
                namespaces=4,
                generations=8,
                obligations=64,
            ),
        ),
        access_policy=Administration(scope),
        configurations=(configuration,),
    )
    bootstrap = CayuApp(
        collaboration_store=collaboration, collaboration=registration, enable_logging=False
    )
    initialized = await bootstrap.initialize_collaboration()
    participants = []
    for alias_revision, role in enumerate(roles):
        created = await bootstrap.create_participant(
            ParticipantCreate(
                operation=initialized.operation("create-" + role),
                configuration=configuration,
                alias=role,
                expected_alias_revision=alias_revision,
            ),
            context=ACCESS,
        )
        participants.append(created.participants[0].reference)
    owner = initialized.owner
    contract = reference(owner, "contract", "visible-text")
    expires = time.time_ns() // 1_000_000 + 900_000
    mandates = DemoMandates(owner, participants, contract, expires_at_ms=expires)
    disclosure = ExactDisclosure(contract, expires_at_ms=expires, session_store=sessions)
    transport = OfflineResponses()
    provider = OfflineProvider(transport)
    budget = BudgetBinding(
        binding_id=scope + ":binding",
        application_scope=scope,
        initiator=PRINCIPAL,
        sponsor="local-example",
        purpose="finite-example",
        root_budget_id=scope + ":root",
        limits=(
            BudgetLimit(
                scope="causal",
                key=scope + ":root",
                max_estimated_cost=Decimal("1"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="offline",
                            model="gpt-5.6",
                            match="exact",
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                    )
                ),
                reservation=BudgetReservation(max_input_tokens=4096, max_output_tokens=256),
            ),
        ),
        ledger_owner="local-example",
        receiver_id="local-example",
        receiver_generation=1,
        allowance=16,
        retention_policy="conservative",
        settlement_policy="exact-or-conservative",
        provider_name="offline",
        model="gpt-5.6",
    )

    class BudgetReceiver:
        async def resolve_budget_binding(self, *, request):
            return budget

    budget_receiver = BudgetReceiver()
    receiver = reference(owner, "request_receiver", "native-fresh")
    exports = SessionExportRegistration(
        owner=owner,
        policy=disclosure,
        projectors=(VisibleText(contract),),
        mandates=mandates,
        readers=tuple(
            ProducerOutputAcceptanceReader(
                collaboration_store=collaboration,
                session_store=sessions,
                namespace=initialized.operation("producer-reader"),
                audience=OwnerRef(
                    application_scope=scope,
                    owner_id=participant.participant_id,
                    incarnation=participant.incarnation,
                ),
            )
            for participant in participants
        ),
        limits=ExportLimits(max_exports=8, max_pending=4, max_retained_bytes=262144),
    )

    def application(policies=(), *, request_key=None):
        app = CayuApp(
            collaboration_store=collaboration,
            collaboration=registration,
            session_store=sessions,
            budget_ledger=ledger,
            budget_binding_receiver=budget_receiver,
            enable_common_root_budget_binding=True,
            collaboration_requests=RequestRegistration(
                mandates=mandates,
                max_ttl_ms=900_000,
                planning_policies=policies,
                prepared_admission=PreparedAdmissionRegistration(receiver=receiver),
            ),
            session_exports=exports,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(demo_agent(request_key=request_key))
        return app

    return DemoTeam(
        scope,
        initialized,
        tuple(participants),
        collaboration,
        sessions,
        ledger,
        contract,
        mandates,
        disclosure,
        transport,
        application,
    )


async def run_demo(
    *, collaboration=None, sessions=None, ledger=None, report=None, creation_bound=False
):
    """One finite journey; optional caller-owned native stores enable qualification."""
    team = await create_demo_team(collaboration=collaboration, sessions=sessions, ledger=ledger)
    return await run_question(
        team,
        key="review-one",
        sender=team.participants[0],
        recipient=team.participants[1],
        task="revision one.",
        report=report,
        creation_bound=creation_bound,
    )


async def run_question(
    team, *, key, sender, recipient, task, report=None, creation_bound=False, input_handoff=None
):
    """One exact finite question inside an existing application-owned team.

    The caller selects roles and task input. Distinct keys isolate independent
    requests while the same team retains shared authority, capacity and budget.
    This is application composition, not a runtime workflow mode.
    """
    if (
        type(team) is not DemoTeam
        or type(key) is not str
        or re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", key) is None
        or type(sender) is not ParticipantRef
        or type(recipient) is not ParticipantRef
        or sender not in team.participants
        or recipient not in team.participants
        or sender == recipient
        or type(task) is not str
        or not task.strip()
        or len(task.encode("utf-8")) > 2048
    ):
        raise ValueError("Example question requires bounded input and distinct registered roles.")
    report = report if report is not None else lambda stage: None
    scope, initialized, sessions = team.scope, team.initialized, team.sessions
    owner, contract = initialized.owner, team.contract
    mandates, disclosure, transport = team.mandates, team.disclosure, team.transport
    application = team.application
    if input_handoff is not None:
        task += await disclosure.question_input(
            input_handoff, question_key=key, recipient=recipient
        )
        if len(task.encode("utf-8")) > 2048:
            raise ValueError("Derived question exceeds the example input bound.")

    def captured():
        return [
            item for item in transport.requests if item.get("instructions") == "DEMO_REQUEST:" + key
        ]

    if captured():
        raise ValueError(
            "Example question key already has transport activity; reconcile its owners."
        )

    def operation(suffix):
        return initialized.operation(key + ":" + suffix)

    app = application(request_key=key)
    await app.initialize_collaboration()
    request = CollaborationRequest(
        operation=operation("question"),
        kind="question",
        sender=sender,
        target=recipient,
        content=task,
        inputs=(),
        context_hint=None,
        delivery_contract=contract,
        output_contract=contract,
        independence_policy=contract,
        disclosure_policy=contract,
        ttl_ms=900_000,
        cancellation="stop",
    )
    accepted = await app.accept_collaboration_request(request, context=mandates.contexts[sender])
    assert not captured()  # acceptance is not execution
    report("accepted")
    wait = CollaborationWait(
        operation=operation("wait"),
        source_owner=owner,
        targets=(accepted.expected,),
        predicate="ALL_SUCCESS",
        predicate_version=1,
        threshold=None,
        deadline=(datetime.now(UTC) + timedelta(minutes=15)).isoformat(),
        failure_policy="settle",
        service_policy="external_observer",
        projection=None,
        wait_edge_revision=1,
        initiator=accepted.expected.initiator,
    )
    parked_input = RunRequest(
        agent_name="demo-agent",
        messages=[Message.text("user", "DEMO_PARK: await review.")],
        # Provider exposure needs the original invocation subject as well as
        # current participant and disclosure grants; none replaces the others.
        invocation_origin=InvocationOriginClaim(subject=PRINCIPAL),
    )
    parked_creation = None
    if creation_bound:
        parked_request = RecipientSessionCreationRequest(
            creation_key=scope + ":" + key + ":coordinator",
            request=parked_input,
            recipient=sender,
        )
        preparation = await app.prepare_recipient_creation(parked_request, context=ACCESS)
        parked_creation = preparation.creation
        parked, _ = await app.create_recipient_session(
            parked_request, preparation=preparation, context=ACCESS
        )
    else:
        parked, _ = await app.create_participant_session(
            ParticipantSessionCreationRequest(
                creation_key=scope + ":" + key + ":coordinator", request=parked_input
            ),
            participant=sender,
            context=ACCESS,
        )
    async for _event in app.execute_participant_session_to_wait(
        ParticipantSessionExecutionRequest(
            request=parked_input.model_copy(update={"session_id": parked.id}),
            session_instance_id=parked.instance_id,
            execution_key="park",
        ),
        wait,
        participant=sender,
        context=ACCESS,
        wait_context=mandates.contexts[sender],
    ):
        pass
    assert len(captured()) == 1
    report("coordinator parked")
    inventory, _cursor = await app.list_participant_sessions(sender, context=ACCESS)
    parked_ref = next(item for item in inventory if item.session_id == parked.id)
    continuation = (await app.list_session_continuations(parked_ref, context=ACCESS)).items[0]
    creation = RecipientSessionCreationRequest(
        creation_key=scope + ":" + key + ":specialist",
        recipient=recipient,
        request=RunRequest(
            agent_name="demo-agent",
            messages=[Message.text("user", "DEMO_TASK: " + task)],
            invocation_origin=InvocationOriginClaim(subject=PRINCIPAL),
        ),
    )
    preparation = await app.prepare_recipient_creation(creation, context=ACCESS)
    policy = ConfiguredRequestPlanningPolicy(
        reference=reference(owner, "request_planning_policy", key + ":fresh"),
        limits=RequestPlanningLimits(
            max_generations=1,
            max_stages=16,
            max_resources=0,
            max_record_bytes=65536,
            max_recovery_items=4,
        ),
        rules=(),
        default=RequestPlanningFresh(preparation=preparation, resources=()),
    )
    app = application((policy,), request_key=key)
    await app.initialize_collaboration()
    context = await mandates.question_context(recipient, key)
    planning = RequestPlanningRequest(
        operation=operation("plan"),
        expected=accepted.expected,
        expected_revision=1,
        expected_input_revision=0,
        expected_input_sha256=commitment(accepted.expected.model_dump(mode="json")),
        planning_generation=1,
        admission_operation=operation("admit"),
        admission_generation=1,
        initiator=InitiatorBinding(
            issuer=owner,
            principal=PRINCIPAL,
            participant=ObjectRef(
                owner=owner,
                kind="participant",
                object_id=recipient.participant_id,
                incarnation=recipient.incarnation,
            ),
            mandate=context.mandate,
            invocation_id=None,
            interaction_id=None,
        ),
        policy=policy.reference,
        policy_sha256=planning_policy_commitment(policy, redactor=SecretRedactor()),
        limits=policy.limits,
        deadline_at_ms=accepted.expected.intent.selection.expires_at_ms,
        predecessor=None,
    )
    await service_until(
        app,
        host_registration(planning_rules=(HostPlanningRule(planning, context),)),
        # This example retains the observation under its journey deadline. Use
        # the host's existing read owner so a slow exact read is not discarded
        # by the shorter public-client acknowledgement timeout.
        lambda: lookup_host_plan(app, planning, context=context),
        lambda found: (
            isinstance(found, ExactMatch)
            and found.receipt.state == "admitted"
            and not found.receipt.pending_stages
        ),
    )
    target = await sessions.load(parked.id)
    report("specialist admitted")
    destination = ProducerDeliveryDestination(
        operation=operation("destination"),
        recipient=sender,
        attempt=PeerDeliveryAttemptKey(
            append_key=PeerAppendKey(
                collaboration_namespace=planning.operation.namespace_incarnation,
                collaboration_generation=planning.operation.generation,
                occurrence_id="answer",
                consumer_id=sender.participant_id,
                consumer_participant_incarnation=sender.incarnation,
                projection_id="visible",
                projection_schema="visible-text.v1",
                target_session_id=None if creation_bound else parked.id,
                target_session_instance_id=None if creation_bound else parked.instance_id,
                creation_target=parked_creation,
            ),
            interest_id="question",
            attempt_generation=1,
            target_run_epoch=target.run_epoch,
            target_transcript_cursor=len(await sessions.load_transcript(parked.id)),
            withdrawal_generation=1,
            deadline_at_ms=planning.deadline_at_ms,
        ),
        projector=contract,
        validator=contract,
        disclosure_policy=contract,
        mandate=context.mandate,
    )
    selected = HostPlannedProducer(
        plan=planning,
        operation=operation("producer"),
        binding_incarnation="one",
        execution_key="produce",
        limits=ProducerOutputLimits(
            output_bytes=2048,
            progress_occurrences=0,
            destinations=1,
            deadline_at_ms=planning.deadline_at_ms,
        ),
        destinations=(destination,),
    )
    page = await service_until(
        app,
        host_registration(
            planned_producer_rules=(HostPlannedProducerRule(selected, ACCESS, context),)
        ),
        lambda: app.pending_producer_outputs(recipient, context=ACCESS),
        lambda found: any(item.recovery.registration == selected.operation for item in found.items),
    )
    token = next(
        item.recovery for item in page.items if item.recovery.registration == selected.operation
    )
    command = (await app.lookup_producer_registration(token, context=ACCESS)).receipt
    assert len(captured()) == 1
    report("producer attached")
    sources = (HostProducerSource(recipient, ACCESS),)
    await service_until(
        app,
        host_registration(
            producer_sources=sources,
            producer_execution_rules=(
                HostProducerExecutionRule(
                    HostProducerExecution(recovery=token, expected_plan=planning), ACCESS, context
                ),
            ),
        ),
        lambda: app.inspect_producer_output(token, context=ACCESS),
        lambda found: isinstance(found, ExactMatch) and found.receipt.completion is not None,
    )
    assert len(captured()) == 2
    report("native producer completed")
    completed = (await app.lookup_producer_completion(token, context=ACCESS)).receipt
    source = command.admission.prepared.target
    await disclosure.allow_source(source.session_id, source.session_instance_id)
    audience = OwnerRef(
        application_scope=scope, owner_id=sender.participant_id, incarnation=sender.incarnation
    )
    for index in completed.output.source_indices:
        await mandates.grant_source(
            context,
            ResourceSelector(
                resource=ObjectRef(
                    owner=owner,
                    kind="session_transcript_row",
                    object_id=source.session_id,
                    incarnation=source.session_instance_id,
                    revision=index + 1,
                )
            ),
            audience,
        )
    export_context = SessionExportAccessContext(principal=PRINCIPAL, mandate=context)
    await service_until(
        app,
        host_registration(
            producer_sources=sources,
            producer_rules=(
                HostProducerMaintenanceRule(
                    HostProducerMaintenance(
                        recovery=token,
                        action="export",
                        destination=destination.operation,
                    ),
                    ACCESS,
                    export_context,
                ),
            ),
        ),
        lambda: app.inspect_producer_output(token, context=ACCESS),
        lambda found: (
            isinstance(found, ExactMatch) and found.receipt.destinations[0].export == "published"
        ),
    )
    exported = await app.export_producer_output(
        command, destination.operation, context=export_context
    )
    retained = await app.lookup_session_export(exported.request, context=export_context)
    if not isinstance(retained, ExactMatch):
        raise RuntimeError("Exact export readback is required before a disclosure decision.")
    report("export published")
    payload = await app.read_session_export(exported.request, context=export_context)
    await disclosure.allow_delivery(
        receipt=retained.receipt,
        payload=payload,
        destination=destination,
        producer_key=command.operation.caller_key,
    )
    wait_page = await app.list_collaboration_waits(context=ACCESS)
    wait_token = next(
        item.recovery for item in wait_page.items if item.recovery.operation == wait.operation
    )
    await service_until(
        app,
        host_registration(
            producer_sources=sources,
            producer_output_rules=(
                HostProducerOutputRule(
                    token,
                    ACCESS,
                    export_context,
                    (HostProducerDisclosure(destination.operation, export_context),),
                ),
            ),
            wait_rules=(HostWaitRule(wait_token, mandates.contexts[sender]),),
        ),
        lambda: app.inspect_producer_output(token, context=ACCESS),
        lambda found: isinstance(found, ExactMatch) and found.receipt.cleanup_ack is not None,
        # This phase composes answer election, append, export release, wait
        # delivery and final settlement. The surrounding finite pattern retains
        # its original journey deadline; this does not extend native authority.
        timeout_s=JOURNEY_TIMEOUT_S,
    )
    report("delivery appended and producer settled")
    # Observe the retained final latch independently of producer cleanup timing.
    record = await service_until(
        app,
        host_registration(wait_rules=(HostWaitRule(wait_token, mandates.contexts[sender]),)),
        lambda: app.recover_session_continuation(continuation, context=ACCESS),
        lambda found: found.latch is not None,
    )
    await service_until(
        app,
        host_registration(
            continuation_rules=(
                HostContinuationRule(
                    continuation,
                    ResumeRequest(
                        session_id=parked.id,
                        messages=[Message.text("user", "DEMO_RESUME: integrate the result.")],
                    ),
                    ContinuationService(
                        ticket=record.ticket,
                        latch=record.latch,
                        continuation_id="integrate",
                        mode="inline",
                        accepted_at=datetime.now(UTC).isoformat(),
                    ),
                    ACCESS,
                ),
            )
        ),
        lambda: app.recover_session_continuation(continuation, context=ACCESS),
        lambda found: found.ticket.state == "CONSUMED",
    )
    inspection = await app.inspect_producer_output(token, context=ACCESS)
    assert isinstance(inspection, ExactMatch)
    assert inspection.receipt.cleanup_ack is not None
    assert inspection.receipt.destinations[0].delivery == "appended"
    finished = await sessions.load(parked.id)
    if finished is None or finished.status != SessionStatus.COMPLETED:
        raise RuntimeError(
            "The continuation was consumed but did not complete; inspect its native session events."
        )
    assert len(captured()) == 3
    # Append, exposure and continuation are distinct. Verify the selected
    # visible result actually reached the real adapter's serialized input.
    assert payload["text"] in json.dumps(captured()[-1]["input"])
    report("continuation consumed")
    # This question does not own the team's store lifetime. Request draining
    # closes the store-wide mutation owner, including other applications sharing
    # it; the team may still have another question to admit or inspect.
    return {
        "provider_calls": len(captured()),
        "producer": "settled",
        "delivery": "appended",
        "continuation": "consumed",
        "answer": payload["text"],
        "source_receipt": retained.receipt.event_id,
        "request": accepted.expected,
        "serialized_requests": json.loads(json.dumps(captured())),
    }


async def main():
    async with asyncio.timeout(JOURNEY_TIMEOUT_S):
        result = await run_demo(report=lambda stage: print(stage, file=sys.stderr, flush=True))
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"serialized_requests", "request"}
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
