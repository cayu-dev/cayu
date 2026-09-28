"""Frozen planner input reaches genuine producer registration and native execution."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_budget_refusal import authorize_execution
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration._host import (
    CollaborationHost,
    _HostRegistration,
    _ProducerExecutionRule,
    _ProducerRegistrationRule,
    _ProducerSource,
)
from cayu.collaboration._host_ownership import HostOwnershipLimits
from cayu.collaboration._host_producer_execution import HostProducerExecution
from cayu.collaboration._host_producer_recovery import (
    recover_planned_execution,
    recover_registered_execution,
)
from cayu.collaboration._producer_contracts import ProducerOutputProposal
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestAdmissionReceipt
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.context_views import ParticipantSessionExecutionRequest

pytestmark = pytest.mark.anyio


@asynccontextmanager
async def execution_deadline(host, provider, *, seconds=240):
    try:
        async with asyncio.timeout(seconds):
            yield
    except TimeoutError as error:
        error.add_note(
            f"Host execution observation: {host.inspect()}; provider calls: {len(provider.requests)}"
        )
        for index, failure in host._source_errors.items():
            error.add_note(f"Source {index} failure type: {type(failure).__name__}")
        for operation in host._owned._operations.values():
            coroutine = operation.task.get_coro()
            names = []
            while coroutine is not None:
                code = getattr(coroutine, "cr_code", None)
                if code is not None:
                    names.append(code.co_qualname)
                coroutine = getattr(coroutine, "cr_await", None)
            error.add_note("Retained await path: " + " -> ".join(names))
        raise


@pytest.mark.parametrize("cancel_observer", [False, True])
async def test_host_recovers_planner_input_before_first_producer_attachment(
    native_stores, monkeypatch, cancel_observer, prepare_in_host=False
):
    from tests.core.test_collaboration_host_planning import drive_host_plan

    plans = []

    async def planning_driver(application, command, context):
        plans.append(command)
        monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)
        return await drive_host_plan(application, command, context)

    application, resolver, admission, provider, session, _, original, _ = await output_scenario(
        native_stores,
        planned=True,
        planning_driver=planning_driver,
        request_ttl_ms=900_000,
        with_exports=True,
        provider_events=(
            (ModelStreamEvent.text_delta("Recovered planner input."), ModelStreamEvent.completed()),
        ),
    )
    # This journey qualifies owner composition; short observation deadlines are
    # covered separately with blocked native dispatch and real cancellation.
    monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)
    monkeypatch.setattr(application._session_export_coordinator.owners, "observation_timeout", 60)
    observation = await application.inspect_collaboration_request(
        admission.expected, context=resolver.recipient.context
    )
    recovered = await application.recover_collaboration_admission(
        observation, context=resolver.recipient.context
    )
    assert isinstance(recovered, ExactMatch)
    # No live preparation handle or private in-process provenance is required.
    receipt = RequestAdmissionReceipt.model_validate_json(recovered.receipt.model_dump_json())
    execution = await recover_planned_execution(
        application, receipt, context=resolver.recipient.context
    )
    assert not provider.requests
    assert execution.request.session_id == session.id
    assert execution.session_instance_id == session.instance_id
    assert observation.producer_operation is None
    execution = ParticipantSessionExecutionRequest(
        request=execution.request,
        session_instance_id=execution.session_instance_id,
        execution_key="retained-original-execution",
    )
    command = await application.prepare_producer_output(
        ProducerOutputProposal(
            operation=original.operation,
            admission=original.admission,
            binding_incarnation=original.binding_incarnation,
            limits=original.limits,
            destinations=original.destinations,
        ),
        execution,
        context=resolver.recipient.context,
    )
    planned_rules = ()
    if prepare_in_host:
        from cayu.collaboration import _host as host_module
        from cayu.collaboration._host_planned_producer import (
            HostPlannedProducer,
            _PlannedProducerRule,
        )

        start = host_module.start_planned_producer

        def require_available_slot(app, ownership, *args, **kwargs):
            # Public servicing must not repeatedly validate the complete plan
            # while a retained attachment already owns the maintenance slot.
            assert ownership.has_slot("maintenance")
            return start(app, ownership, *args, **kwargs)

        monkeypatch.setattr(host_module, "start_planned_producer", require_available_slot)

        planned_rules = (
            _PlannedProducerRule(
                HostPlannedProducer(
                    plan=plans[0],
                    operation=original.operation,
                    binding_incarnation=original.binding_incarnation,
                    execution_key=execution.execution_key,
                    limits=original.limits,
                    destinations=original.destinations,
                ),
                CONTEXT,
                resolver.recipient.context,
            ),
        )
    prepared = command.admission.prepared
    assert prepared is not None
    async with (
        CollaborationHost(
            application,
            _HostRegistration(
                limits=HostOwnershipLimits(1, 1, 2, 256 * 1024),
                producer_sources=(
                    () if prepare_in_host else (_ProducerSource(prepared.recipient, CONTEXT),)
                ),
                producer_rules=(),
                producer_registration_rules=()
                if prepare_in_host
                else (_ProducerRegistrationRule(command, CONTEXT, resolver.recipient.context),),
                planned_producer_rules=planned_rules,
                observation_timeout_s=60,
                shutdown_timeout_s=60,
            ),
        ) as registration_host,
        asyncio.timeout(180),
    ):
        while True:
            state = await registration_host.service_once()
            for outcome in registration_host._owned.inspect().completed:
                if outcome.error is not None:
                    raise outcome.error
            assert state.source_failures == 0, registration_host._source_errors
            if state.pending and not state.serviced:
                await asyncio.sleep(0.01)
                continue
            if isinstance(
                await application.lookup_producer_registration(command, context=CONTEXT), ExactMatch
            ):
                break
    assert registration_host.inspect().uncertain == 0
    assert not provider.requests
    inspection = await application.inspect_producer_output(command, context=CONTEXT)
    assert isinstance(inspection, ExactMatch)
    assert inspection.receipt.state == "registered"
    assert inspection.receipt.completion is None
    assert inspection.receipt.cleanup_ack is None
    assert inspection.receipt.deliveries == ()
    page = await application.pending_producer_outputs(prepared.recipient, context=CONTEXT)
    token = next(
        item.recovery for item in page.items if item.recovery.registration == command.operation
    )
    with pytest.raises(CollaborationUnavailable, match="registration"):
        await recover_registered_execution(
            application,
            token.model_copy(update={"registration_commitment": "sha256:" + "0" * 64}),
            context=CONTEXT,
            producer_context=resolver.recipient.context,
        )
    assert not provider.requests
    authorize_execution(resolver)
    if prepare_in_host:
        # Identical frozen input is insufficient: execution must retain the
        # complete first plan expectation, including the accepted revision.
        changed_plan = plans[0].model_copy(
            update={"expected_revision": plans[0].expected_revision + 1}
        )
        rejected_host = CollaborationHost(
            application,
            _HostRegistration(
                limits=HostOwnershipLimits(1, 1, 2, 256 * 1024),
                producer_sources=(_ProducerSource(prepared.recipient, CONTEXT),),
                producer_rules=(),
                producer_execution_rules=(
                    _ProducerExecutionRule(
                        HostProducerExecution(recovery=token, expected_plan=changed_plan),
                        CONTEXT,
                        resolver.recipient.context,
                    ),
                ),
                observation_timeout_s=60,
            ),
        )
        reported = []
        async with asyncio.timeout(90):
            while not rejected_host.inspect().failed and not reported:
                try:
                    await rejected_host.service_once()
                except CollaborationUnavailable as error:
                    reported.append(error)
        try:
            await rejected_host.aclose()
        except CollaborationUnavailable as error:
            reported.append(error)
        assert len(reported) == 1
        assert "another exact plan" in str(reported[0])
        assert not provider.requests
        # The failed read never entered a receiving mutation. Shutdown reports
        # its original rejection and releases only the local observation slot.
        assert not rejected_host._owned.inspect().completed
        assert not (await rejected_host.aclose()).pending
        still_registered = await application.inspect_producer_output(command, context=CONTEXT)
        assert isinstance(still_registered, ExactMatch)
        assert still_registered.receipt.state == "registered"
        assert still_registered.receipt.completion is None
    entered = asyncio.Event()
    release = asyncio.Event()
    if cancel_observer:
        original_stream = provider.stream

        async def blocked_stream(request):
            async for event in original_stream(request):
                entered.set()
                await release.wait()
                yield event

        monkeypatch.setattr(provider, "stream", blocked_stream)
    host = CollaborationHost(
        application,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 256 * 1024),
            producer_sources=(_ProducerSource(prepared.recipient, CONTEXT),),
            producer_rules=(),
            producer_execution_rules=()
            if prepare_in_host
            else (
                _ProducerExecutionRule(
                    HostProducerExecution(recovery=token),
                    CONTEXT,
                    resolver.recipient.context,
                ),
            ),
            planned_producer_rules=planned_rules,
            observation_timeout_s=60,
            shutdown_timeout_s=30,
        ),
    )
    assert not provider.requests
    async with host, execution_deadline(host, provider):
        try:
            if cancel_observer:
                for _ in range(2):
                    runner = asyncio.create_task(host.run())
                    await entered.wait()
                    await asyncio.sleep(0)
                    runner.cancel()
                    assert runner.cancelling() == 1
                    with pytest.raises(asyncio.CancelledError):
                        await runner
                    assert runner.cancelled()
                    observed = host.inspect()
                    assert observed.active + observed.uncertain == 1
                    assert len(provider.requests) == 1
                release.set()
            while host.inspect().serviced == 0:
                observed = await host.service_once()
                assert observed.failed == 0
                assert observed.source_failures == 0
                await asyncio.sleep(0.01)
        finally:
            release.set()
    assert host.inspect().uncertain == 0
    assert len(provider.requests) == 1
    retained = await application.lookup_producer_completion(token, context=CONTEXT)
    assert isinstance(retained, ExactMatch)
    completion = retained.receipt
    from cayu.collaboration._producer_contracts import ProducerCompletionRecord

    assert isinstance(completion, ProducerCompletionRecord)
    assert completion.output.disposition == "answer"
    inspection = await application.inspect_producer_output(token, context=CONTEXT)
    assert isinstance(inspection, ExactMatch)
    assert inspection.receipt.state == "launch_claimed"
    assert inspection.receipt.completion == completion.operation
    assert inspection.receipt.cleanup_ack is None
    assert inspection.receipt.deliveries == ()
    # Reconstructing launch intent again cannot repair output by executing it.
    _, replay = await recover_registered_execution(
        application, token, context=CONTEXT, producer_context=resolver.recipient.context
    )
    assert replay == execution
    assert len(provider.requests) == 1
    if not cancel_observer:
        await qualify_host_output(application, resolver, command, token, session, provider)


async def test_host_prepares_and_attaches_exact_planned_producer(native_stores, monkeypatch):
    await test_host_recovers_planner_input_before_first_producer_attachment(
        native_stores,
        monkeypatch,
        cancel_observer=False,
        prepare_in_host=True,
    )


async def qualify_host_output(application, resolver, command, token, session, provider):
    """Drive export, election, delivery and cleanup through separate host lifetimes."""
    from cayu.collaboration._contracts import ObjectRef, OwnerRef
    from cayu.collaboration._host import (
        _ProducerDisclosure,
        _ProducerMaintenanceRule,
        _ProducerOutputRule,
    )
    from cayu.collaboration._host_producer_maintenance import HostProducerMaintenance
    from cayu.collaboration._session_export_store import digest
    from cayu.collaboration.exports import SessionExportAccessContext
    from cayu.collaboration.mandates import ResourceSelector

    destination = command.destinations[0]
    owner = application._session_export_coordinator.owner
    audience = OwnerRef(
        application_scope=owner.application_scope,
        owner_id=destination.recipient.participant_id,
        incarnation=destination.recipient.incarnation,
    )
    resource = ResourceSelector(
        resource=ObjectRef(
            owner=owner,
            kind="session_transcript_row",
            object_id=session.id,
            incarnation=session.instance_id,
            revision=3,
        )
    )
    resolution = resolver.recipient.resolution
    actions = tuple(
        dict.fromkeys(
            (*resolution.principal.actions, "source", "expose", "publish", "release", "administer")
        )
    )
    resolver.recipient.resolution = resolution.model_copy(
        update={
            "principal": resolution.principal.model_copy(
                update={"actions": actions, "audiences": (owner, audience)}
            ),
            "chain": resolution.chain.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(
                            update={
                                "actions": actions,
                                "audiences": (owner, audience),
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
    access = SessionExportAccessContext(
        principal=resolver.recipient.context.principal, mandate=resolver.recipient.context
    )
    source = _ProducerSource(command.admission.prepared.recipient, CONTEXT)
    registration = dict(
        limits=HostOwnershipLimits(1, 1, 2, 256 * 1024),
        producer_sources=(source,),
        observation_timeout_s=60,
        shutdown_timeout_s=30,
    )
    # First host creates the authenticated export. The application policy then
    # explicitly grants its exact content/audience; the host never grants itself
    # disclosure permission or substitutes a permissive adapter.
    async with (
        CollaborationHost(
            application,
            _HostRegistration(
                **registration,
                producer_rules=(
                    _ProducerMaintenanceRule(
                        HostProducerMaintenance(
                            recovery=token, action="export", destination=destination.operation
                        ),
                        CONTEXT,
                        access,
                    ),
                ),
            ),
        ) as host,
        asyncio.timeout(120),
    ):
        next_inspection = 0.0
        while True:
            state = await host.service_once()
            for outcome in host._owned.inspect().completed:
                if outcome.error is not None:
                    raise outcome.error
            assert state.failed == state.source_failures == 0, host._source_errors
            if (state.pending and not state.serviced) or (
                asyncio.get_running_loop().time() < next_inspection
            ):
                await asyncio.sleep(0.01)
                continue
            # Do not compete with active native writes for durable inspection.
            # Per-pass counts are not cumulative generations. Equal consecutive
            # counts and progress by another worker must not suppress readback.
            inspection = await application.inspect_producer_output(token, context=CONTEXT)
            next_inspection = asyncio.get_running_loop().time() + 1
            assert isinstance(inspection, ExactMatch)
            if inspection.receipt.destinations[0].export == "published":
                break
    exported = await application.export_producer_output(
        command, destination.operation, context=access
    )
    retained = await application.lookup_session_export(exported.request, context=access)
    assert isinstance(retained, ExactMatch)
    policy = application._session_export_coordinator.registration.policy
    policy.register_export(
        retained.receipt,
        payload_sha256=digest({"text": "Recovered planner input.", "artifact_commitments": []}),
        consumer_id=destination.recipient.participant_id,
    )
    policy.allowed_receipts.add(command.operation.caller_key)
    # A replacement host reconstructs all protocol progress from owners. It
    # elects, appends, releases the export and settles producer responsibility.
    async with (
        CollaborationHost(
            application,
            _HostRegistration(
                **registration,
                producer_rules=(),
                producer_output_rules=(
                    _ProducerOutputRule(
                        token,
                        CONTEXT,
                        access,
                        (_ProducerDisclosure(destination.operation, access),),
                    ),
                ),
            ),
        ) as host,
        execution_deadline(host, provider, seconds=180),
    ):
        next_inspection = 0.0
        while True:
            state = await host.service_once()
            for outcome in host._owned.inspect().completed:
                if outcome.error is not None:
                    raise outcome.error
            assert state.failed == state.source_failures == 0, host._source_errors
            if (state.pending and not state.serviced) or (
                asyncio.get_running_loop().time() < next_inspection
            ):
                await asyncio.sleep(0.01)
                continue
            inspection = await application.inspect_producer_output(token, context=CONTEXT)
            next_inspection = asyncio.get_running_loop().time() + 1
            assert isinstance(inspection, ExactMatch)
            if inspection.receipt.cleanup_ack is not None:
                break
    assert host.inspect().uncertain == 0
    assert inspection.receipt.request_state == "answered"
    assert inspection.receipt.destinations[0].delivery == "appended"
    assert len(provider.requests) == 1
    assert await application.inspect_producer_output(token, context=CONTEXT) == inspection
    from cayu.collaboration._host_producer_registration import producer_registration_ready

    # Exact final cleanup is positive completion evidence, not new attachment
    # work. A later host must not recreate responsibility from the old command.
    assert await producer_registration_ready(application, command, context=CONTEXT)
