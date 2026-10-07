"""Registration proposals are bounded data, even with genuine admission evidence."""

import pytest
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_prepared_admission_public import prepared_scenario

from cayu.collaboration._contracts import ExactMatch, ObjectRef
from cayu.collaboration._producer_contracts import (
    ProducerDeliveryDestination,
    ProducerOutputLimits,
    ProducerOutputRegistration,
)
from cayu.collaboration.peer_content import PeerAppendKey, PeerDeliveryAttemptKey
from cayu.messages import Message
from cayu.runtime.execution_profiles import ExecutionProfileIdentity
from cayu.sessions import RunRequest
from cayu.sessions._participant_execution_identity import participant_execution_identity
from cayu.sessions.context_views import ParticipantSessionExecutionRequest
from cayu.vaults.redaction import SecretRedactor


def output_exports(initialized, request, resolver, *, collaboration_store, session_store):
    import asyncio
    import threading
    from contextlib import asynccontextmanager

    from tests.core.test_peer_content import QualificationPeerExposurePolicy

    from cayu.collaboration._contracts import OwnerRef
    from cayu.collaboration._producer_acceptance import ProducerOutputAcceptanceReader
    from cayu.collaboration.exports import (
        ExportLimits,
        SessionExportAuthorization,
        SessionExportDenied,
        SessionExportProjector,
        SessionExportRegistration,
    )

    class Policy(QualificationPeerExposurePolicy):
        def __init__(self):
            super().__init__()
            self.lock = asyncio.Lock()
            self.denied = set()
            self.denied_consumers = set()

        @property
        def ref(self):
            return request.disclosure_policy

        @asynccontextmanager
        async def acquire(self, context, **kwargs):
            async with self.lock:
                if self.denied.intersection(kwargs["actions"]):
                    raise SessionExportDenied()
                yield SessionExportAuthorization(
                    issuer=initialized.owner,
                    principal=context.principal,
                    policy=self.ref,
                    revision=1,
                    expires_at_ms=4102444800000,
                )

        @asynccontextmanager
        async def acquire_peer_append(self, context, **kwargs):
            async with self.lock:
                if (
                    "append" in self.denied
                    or kwargs["append_key"].consumer_id in self.denied_consumers
                ):
                    raise SessionExportDenied()
                async with super().acquire_peer_append(context, **kwargs) as authorization:
                    yield authorization

        @asynccontextmanager
        async def acquire_peer_exclusion(self, context, **kwargs):
            async with self.lock, super().acquire_peer_exclusion(context, **kwargs):
                yield

    class Projector(SessionExportProjector):
        def __init__(self):
            self.calls = 0
            self.entered = threading.Event()
            self.release = None

        @property
        def ref(self):
            return request.output_contract

        def project(self, source):
            self.calls += 1
            self.entered.set()
            if self.release is not None and not self.release.wait(60):
                raise AssertionError("Producer projector barrier did not settle")
            assert all(part.type == "text" for row in source for part in row.message.content)
            return {"text": "".join(part.text for row in source for part in row.message.content)}

        def validate(self, source, output, audience):
            return output == {
                "text": "".join(part.text for row in source for part in row.message.content)
            }

    return SessionExportRegistration(
        owner=initialized.owner,
        policy=Policy(),
        projectors=(Projector(),),
        mandates=resolver,
        readers=(
            ProducerOutputAcceptanceReader(
                collaboration_store=collaboration_store,
                session_store=session_store,
                namespace=initialized.operation("producer-reader"),
                audience=OwnerRef(
                    application_scope=initialized.owner.application_scope,
                    owner_id=request.sender.participant_id,
                    incarnation=request.sender.incarnation,
                ),
            ),
        ),
        limits=ExportLimits(max_exports=8, max_pending=4, max_retained_bytes=262144),
    )


async def output_scenario(
    native_stores,
    *,
    with_exports=False,
    provider_events=(),
    tools=(),
    tool_policy=None,
    loop_policies=(),
    config=None,
    budget_ledger=None,
    budget_binding_factory=None,
    consumer_origin=None,
    operation_prefix="",
    planned=False,
    planning_driver=None,
    request_ttl_ms=None,
    cancellation="stop",
    unchecked_registration=False,
    requested_session_id=None,
):
    from tests.core.test_peer_content import QualifiedPeerProvider

    application, resolver, admission, provider, session, initialized = await prepared_scenario(
        native_stores,
        use_example=True,
        provider_events=provider_events,
        session_exports_factory=(
            lambda initialized, request, resolver: output_exports(
                initialized,
                request,
                resolver,
                collaboration_store=native_stores[0],
                session_store=native_stores[1],
            )
        )
        if with_exports
        else None,
        provider_factory=QualifiedPeerProvider,
        tools=tools,
        tool_policy=tool_policy,
        loop_policies=loop_policies,
        config=config,
        budget_ledger=budget_ledger,
        budget_binding_factory=budget_binding_factory,
        operation_prefix=operation_prefix,
        planned=planned,
        planning_driver=planning_driver,
        request_cancellation=cancellation,
        requested_session_id=requested_session_id,
        request_ttl_ms=request_ttl_ms
        if request_ttl_ms is not None
        else 300_000
        if planned
        else None,
    )
    found = await application.collaboration_admission_reader().lookup(
        admission, context=resolver.recipient.context
    )
    assert isinstance(found, ExactMatch) and found.receipt.state == "admitted"
    recipient = admission.expected.intent.selection.sender.reference
    original = admission.expected.intent.request
    deadline = admission.expected.intent.selection.expires_at_ms
    target_id, target_instance, target_epoch, target_cursor = (
        "receiving-session",
        "receiving-instance",
        0,
        0,
    )
    if with_exports:
        from tests.core.test_participant_identity import CONTEXT

        from cayu.sessions.context_views import RecipientSessionCreationRequest

        target, _ = await application.create_recipient_session(
            RecipientSessionCreationRequest(
                request=RunRequest(
                    agent_name="reviewer", messages=[], invocation_origin=consumer_origin
                ),
                creation_key="output-consumer:"
                + operation_prefix
                + initialized.owner.application_scope,
                recipient=recipient,
            ),
            context=CONTEXT,
        )
        target_id, target_instance, target_epoch = target.id, target.instance_id, target.run_epoch
        target_cursor = len(await native_stores[1].load_transcript(target.id))
    destination = ProducerDeliveryDestination(
        operation=initialized.operation(operation_prefix + "output-destination"),
        recipient=recipient,
        attempt=PeerDeliveryAttemptKey(
            append_key=PeerAppendKey(
                collaboration_namespace=admission.operation.namespace_incarnation,
                collaboration_generation=admission.operation.generation,
                occurrence_id="final-output",
                consumer_id=recipient.participant_id,
                consumer_participant_incarnation=recipient.incarnation,
                projection_id="text",
                projection_schema="visible-text.v1",
                target_session_id=target_id,
                target_session_instance_id=target_instance,
            ),
            interest_id="original-request",
            attempt_generation=1,
            target_run_epoch=target_epoch,
            target_transcript_cursor=target_cursor,
            withdrawal_generation=1,
            deadline_at_ms=deadline,
        ),
        projector=original.output_contract,
        validator=original.output_contract,
        disclosure_policy=original.disclosure_policy,
        mandate=resolver.recipient.context.mandate,
    )
    execution = ParticipantSessionExecutionRequest(
        request=RunRequest(
            agent_name="reviewer", session_id=session.id, messages=[Message.text("user", "input")]
        ),
        session_instance_id=session.instance_id,
        execution_key="producer-one",
    )
    binding = await native_stores[1].load_participant_session_binding(session.id)
    identity = participant_execution_identity(
        execution,
        binding,
        execution_profile_fingerprint=ExecutionProfileIdentity.model_validate_json(
            admission.prepared.execution_profile_json
        ).fingerprint,
    )
    registration_type = (
        ProducerOutputRegistration.model_construct
        if unchecked_registration
        else ProducerOutputRegistration
    )
    proposal = registration_type(
        operation=initialized.operation(operation_prefix + "output-registration"),
        admission=admission,
        initiator=admission.initiator,
        receiver=admission.prepared.receiver,
        binding_incarnation="output-one",
        execution_key="producer-one",
        execution_commitment="sha256:" + identity.admission_commitment,
        publisher_generation=admission.generation,
        disposition=original.cancellation,
        limits=ProducerOutputLimits(
            output_bytes=1024, progress_occurrences=4, destinations=1, deadline_at_ms=deadline
        ),
        destinations=(destination,),
    )
    return application, resolver, admission, provider, session, initialized, proposal, execution


@pytest.mark.anyio
@pytest.mark.parametrize("launch_first", [False, True])
async def test_output_contract_from_real_admission_is_bounded_and_non_dispatching(
    native_stores, launch_first, monkeypatch
):
    (
        application,
        resolver,
        admission,
        provider,
        session,
        initialized,
        proposal,
        execution,
    ) = await output_scenario(native_stores, with_exports=True)
    # This matrix proves source/native exclusion ordering, not the separately
    # qualified short foreground observation deadline during backend readback.
    monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)
    found = await application.collaboration_admission_reader().lookup(
        admission, context=resolver.recipient.context
    )
    owner = initialized.owner
    deadline = admission.expected.intent.selection.expires_at_ms
    destination = proposal.destinations[0]
    recipient = destination.recipient
    assert ProducerOutputRegistration.model_validate_json(proposal.model_dump_json()) == proposal
    for update in (
        {"operation": admission.operation},
        {"publisher_generation": admission.generation + 1},
        {"destinations": (destination, destination)},
        {"limits": proposal.limits.model_copy(update={"deadline_at_ms": deadline + 1})},
        {"limits": proposal.limits.model_copy(update={"output_bytes": True})},
        {"limits": proposal.limits.model_copy(update={"output_bytes": 65537})},
        {"admission": admission.model_copy(update={"prepared": None})},
    ):
        with pytest.raises(ValueError):
            ProducerOutputRegistration.model_validate(proposal.model_copy(update=update))
    with pytest.raises(ValueError, match="Output destination identity conflicts"):
        ProducerDeliveryDestination.model_validate(
            destination.model_copy(
                update={"recipient": recipient.model_copy(update={"incarnation": "replacement"})}
            )
        )
    # Proposal construction/readback supplies no native launch or caller authority.
    assert provider.requests == []

    assert (await native_stores[1].load(session.id)).run_epoch == 0

    # The registered owner below authenticates real admission and native execution
    # identity. This remains registration coverage, not launch/delivery coverage.
    from cayu.collaboration._capacity import PERMIT_SETTLEMENT_BYTES
    from cayu.collaboration._permit_store import registered_receipt
    from cayu.collaboration._producer_registration import register_producer_output
    from cayu.collaboration._producer_store import (
        read_output_registration,
        read_request_output,
        register_output_in_transaction,
    )

    store = native_stores[0]
    redactor = SecretRedactor()

    # A failed transaction must not leave either the obligation or its capacity
    # reservation visible to another worker. This exercises the real backend
    # rollback, not a simulated failure before the first write.
    class PublicationInterrupted(Exception):
        pass

    async with store._transaction(owner.application_scope, write=False) as tx:
        initial = await store._anchor(tx, initialized, redactor)
        initial_permits = await store._permit_state(tx, admission.prepared.recipient, redactor)
    with pytest.raises(PublicationInterrupted):
        async with store._transaction(owner.application_scope, write=True) as tx:
            await register_output_in_transaction(
                store,
                tx,
                initialized,
                proposal,
                authority_expires_at_ms=deadline,
                redactor=redactor,
            )
            raise PublicationInterrupted
    async with native_stores[2]()._transaction(owner.application_scope, write=False) as tx:
        assert await read_output_registration(tx, proposal, redactor=redactor) is None
        assert await read_request_output(tx, admission.expected, redactor=redactor) is None
        assert await store._anchor(tx, initialized, redactor) == initial
        assert (
            await store._permit_state(tx, admission.prepared.recipient, redactor) == initial_permits
        )
    from cayu.collaboration.access import CollaborationAccessDenied

    with pytest.raises(CollaborationAccessDenied):
        await register_producer_output(
            application,
            proposal.model_copy(update={"execution_commitment": "sha256:" + "a" * 64}),
            execution,
            context=resolver.recipient.context,
        )
    record = await register_producer_output(
        application, proposal, execution, context=resolver.recipient.context
    )
    assert (
        await register_producer_output(
            application, proposal, execution, context=resolver.recipient.context
        )
        == record
    )
    from cayu.runtime._producer_output_store import (
        ROOT_KEY,
        NativeProducerAttachment,
        attachment_index,
    )
    from cayu.sessions._checkpoint_preservation import _invocation_lifecycle_authority_read_scope
    from cayu.sessions.base import SessionOperationPublication

    native = NativeProducerAttachment.from_registration(record)
    index = attachment_index(native)
    assert await native_stores[1].load_session_operation(
        session.id, index.operation_key
    ) == native.model_dump(mode="json")
    with _invocation_lifecycle_authority_read_scope():
        checkpoint = await native_stores[1].load_checkpoint(session.id)
    assert checkpoint[ROOT_KEY] == index.model_dump(mode="json")
    with pytest.raises((PermissionError, ValueError)):
        await native_stores[1].publish_session_operation(
            session.id,
            idempotency_key=index.operation_key,
            operation_transform=lambda _session, checkpoint, _record: SessionOperationPublication(
                checkpoint=checkpoint,
                operation_records={index.operation_key: native.model_dump(mode="json")},
            ),
            events=[],
        )
    with pytest.raises((PermissionError, ValueError)):
        await native_stores[1].checkpoint(
            session.id,
            {
                ROOT_KEY: index.model_copy(
                    update={"record_commitment": "sha256:" + "0" * 64}
                ).model_dump(mode="json")
            },
        )
    visible = []

    def generic_transform(_session, checkpoint):
        visible.append(checkpoint)
        return {}

    await native_stores[1].transform_checkpoint(session.id, generic_transform)
    assert ROOT_KEY not in (visible[0] or {})
    with _invocation_lifecycle_authority_read_scope():
        retained_checkpoint = await native_stores[1].load_checkpoint(session.id)
    assert retained_checkpoint[ROOT_KEY] == checkpoint[ROOT_KEY]
    with pytest.raises(ValueError, match="producer output responsibility"):
        await native_stores[1].delete_session(session.id)

    assert provider.requests == []
    assert (await native_stores[1].load(session.id)).run_epoch == 0
    async with store._transaction(owner.application_scope, write=False) as tx:
        before = initial
        after = await store._anchor(tx, initialized, redactor)
        assert after.permit_count == before.permit_count + 1
        assert after.reserved_bytes == (
            before.reserved_bytes + record.reserved_bytes + PERMIT_SETTLEMENT_BYTES
        )
        permits = await store._permit_state(tx, admission.prepared.recipient, redactor)
        assert permits.outstanding == initial_permits.outstanding + 1
        assert permits.issued_frontier == initial_permits.issued_frontier + 1
        assert await registered_receipt(tx, record.permit, redactor) is not None
    async with native_stores[2]()._transaction(owner.application_scope, write=True) as tx:
        assert await read_output_registration(tx, proposal, redactor=redactor) == record
        assert await read_request_output(tx, admission.expected, redactor=redactor) == record
        assert (
            await register_output_in_transaction(
                store,
                tx,
                initialized,
                proposal,
                authority_expires_at_ms=deadline,
                redactor=redactor,
            )
            == record
        )
        assert await store._anchor(tx, initialized, redactor) == after

        with pytest.raises(ValueError):
            await register_output_in_transaction(
                store,
                tx,
                initialized,
                proposal.model_copy(update={"execution_key": "different-execution"}),
                authority_expires_at_ms=deadline,
                redactor=redactor,
            )
        assert await store._anchor(tx, initialized, redactor) == after

    from cayu.collaboration._contracts import CollaborationConflict
    from cayu.collaboration._producer_registration import producer_launch_guard
    from cayu.collaboration._producer_store import claim_output_launch_in_transaction
    from cayu.collaboration.mandates import MandateDenied
    from cayu.collaboration.requests import RequestControl

    if launch_first:
        # Historical admission plus prepare/readback rights cannot launch.
        with pytest.raises(MandateDenied):
            async with producer_launch_guard(
                application, proposal, context=resolver.recipient.context
            ):
                pytest.fail("Preparation must not supply execution authority.")
        original_resolution = resolver.recipient.resolution
        actions = (*original_resolution.principal.actions, "execute")
        resolver.recipient.resolution = original_resolution.model_copy(
            update={
                "principal": original_resolution.principal.model_copy(update={"actions": actions}),
                "chain": original_resolution.chain.model_copy(
                    update={
                        "entries": tuple(
                            entry.model_copy(update={"actions": actions})
                            for entry in original_resolution.chain.entries
                        )
                    }
                ),
            }
        )
        async with producer_launch_guard(
            application, proposal, context=resolver.recipient.context
        ) as claimed:
            assert claimed.state == "launch_claimed"
        async with store._transaction(owner.application_scope, write=False) as tx:
            claimed_anchor = await store._anchor(tx, initialized, redactor)
            assert claimed.state == "launch_claimed"
            assert claimed.launch is not None
            assert claimed_anchor.permit_count == after.permit_count
            assert claimed_anchor.reserved_operations == after.reserved_operations - 1
        async with native_stores[2]()._transaction(owner.application_scope, write=True) as tx:
            assert (
                await claim_output_launch_in_transaction(
                    store,
                    tx,
                    initialized,
                    proposal,
                    authority_expires_at_ms=deadline,
                    redactor=redactor,
                )
                == claimed
            )
            assert await store._anchor(tx, initialized, redactor) == claimed_anchor
        # Source accounting changed, but native registration identity did not.
        assert NativeProducerAttachment.from_registration(claimed) == native
        resolver.recipient.resolution = original_resolution
        with pytest.raises(MandateDenied):
            async with producer_launch_guard(
                application, proposal, context=resolver.recipient.context
            ):
                pytest.fail("A retained election must not restore revoked execution authority.")

    control_request = RequestControl(
        operation=initialized.operation("cancel-before-native-launch"),
        expected=admission.expected,
        expected_revision=found.receipt.revision,
        kind="cancel",
    )
    from cayu.collaboration.participants import CollaborationUnavailable

    try:
        control = await application.control_collaboration_request(
            control_request, context=resolver.sender.context
        )
    except CollaborationUnavailable:
        import asyncio

        pending = tuple(application._request_coordinator._owners.pending)
        if not pending:
            raise
        # Observation expiry is not operation failure. Await the actual retained
        # owner, exposing any underlying defect, then reconcile the same key.
        await asyncio.wait_for(asyncio.gather(*pending), 30)
        control = await application.control_collaboration_request(
            control_request, context=resolver.sender.context
        )
    async with native_stores[2]()._transaction(owner.application_scope, write=True) as tx:
        closed_anchor = await store._anchor(tx, initialized, redactor)
        with pytest.raises(CollaborationConflict):
            await claim_output_launch_in_transaction(
                store,
                tx,
                initialized,
                proposal,
                authority_expires_at_ms=deadline,
                redactor=redactor,
            )
        assert await store._anchor(tx, initialized, redactor) == closed_anchor
        retained = await read_output_registration(tx, proposal, redactor=redactor)
        assert retained.command == record.command
        assert retained.state == "excluded"
        assert retained.cleanup is not None
        assert (
            retained.reserved_operations == retained.reserved_events == retained.reserved_bytes == 0
        )
    assert provider.requests == []
    assert (await native_stores[1].load(session.id)).run_epoch == 0
    # Genuine source closure is delivered to the native owner; absence alone
    # cannot establish exclusion, and subsequent attachment cannot reopen it.
    automatic_exclusion = await native_stores[1]._read_native_producer_exclusion(retained, control)
    assert automatic_exclusion.state == "excluded"
    from cayu.collaboration._producer_registration import exclude_prepared_producer

    # An exact-looking but unretained control cannot authorize native exclusion.
    with pytest.raises(CollaborationUnavailable):
        await exclude_prepared_producer(
            application,
            proposal,
            control.expected.model_copy(
                update={
                    "operation": initialized.operation("absent-control"),
                    "intent": control.expected.intent.model_copy(
                        update={"operation": initialized.operation("absent-control")}
                    ),
                }
            ),
            context=resolver.sender.context,
        )
    with _invocation_lifecycle_authority_read_scope():
        assert (await native_stores[1].load_checkpoint(session.id))[ROOT_KEY]["state"] == "excluded"
    exclusion = await exclude_prepared_producer(
        application, proposal, control.expected, context=resolver.sender.context
    )
    assert exclusion.state == "excluded"
    assert exclusion == automatic_exclusion
    assert await native_stores[1]._read_native_producer_exclusion(retained, control) == exclusion
    assert (
        await exclude_prepared_producer(
            application, proposal, control.expected, context=resolver.sender.context
        )
        == exclusion
    )
    await native_stores[1]._attach_native_producer(retained)
    with _invocation_lifecycle_authority_read_scope():
        excluded_checkpoint = await native_stores[1].load_checkpoint(session.id)
    assert excluded_checkpoint[ROOT_KEY]["state"] == "excluded"
    assert excluded_checkpoint[ROOT_KEY]["cleanup_commitment"] is not None
    assert provider.requests == []
    await native_stores[1].delete_session(session.id)
    assert await native_stores[1].load(session.id) is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "answer_bytes,export_failure",
    [(14, "ack"), (14, "cancel"), (14, "exclude"), (1024, "none"), (1025, "none")],
)
async def test_registered_producer_enters_native_participant_runtime(
    native_stores, answer_bytes, export_failure, monkeypatch
):
    from tests.core.test_participant_identity import CONTEXT

    from cayu.collaboration._producer_registration import register_producer_output
    from cayu.providers.base import ModelStreamEvent
    from cayu.runtime._producer_execution import _ProducerExecution

    (
        application,
        resolver,
        admission,
        provider,
        session,
        _,
        proposal,
        execution,
    ) = await output_scenario(native_stores, with_exports=True)
    oversized = answer_bytes > 1024
    # This matrix qualifies exact concurrent completion and output bounds, not
    # the default short foreground observation deadline under backend contention.
    # Dedicated ownership tests exercise timeout with a dispatched-work barrier.
    monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)
    answer = "a" * answer_bytes
    provider._batches = (
        (
            ModelStreamEvent.thinking("private-producer-canary"),
            ModelStreamEvent.text_delta(answer),
            ModelStreamEvent.completed(),
        ),
    )
    registration = await register_producer_output(
        application, proposal, execution, context=resolver.recipient.context
    )
    original = resolver.recipient.resolution
    actions = (*original.principal.actions, "execute")
    resolver.recipient.resolution = original.model_copy(
        update={
            "principal": original.principal.model_copy(update={"actions": actions}),
            "chain": original.chain.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(update={"actions": actions})
                        for entry in original.chain.entries
                    )
                }
            ),
        }
    )
    handoff = _ProducerExecution(application, proposal, resolver.recipient.context)
    from cayu.applications import _ParticipantExecutionSettlementReader
    from cayu.collaboration._contracts import ExactUnavailable
    from cayu.events import EventType
    from cayu.runtime._producer_output_store import ROOT_KEY, NativeProducerIndex
    from cayu.sessions._checkpoint_preservation import _invocation_lifecycle_authority_read_scope

    events = []
    execution_reader = None
    async for event in application._execute_participant_session(
        execution,
        participant=admission.prepared.recipient,
        context=CONTEXT,
        producer_output=handoff,
    ):
        events.append(event)
        if event.type is EventType.SESSION_COMPLETED:
            with _invocation_lifecycle_authority_read_scope():
                pending_checkpoint = await native_stores[1].load_checkpoint(session.id)
            pending_index = NativeProducerIndex.model_validate(pending_checkpoint[ROOT_KEY])
            invocation = pending_index.invocation
            store, initialized = application._participant_coordinator._ready()
            execution_permit = await store._lookup_registered_permit(
                initialized,
                initialized.operation(invocation.participant_permit_operation),
                redactor=application._secret_redactor,
            )
            execution_reader = _ParticipantExecutionSettlementReader(
                application,
                execution_permit.expected,
                invocation.participant_permit_commitment.removeprefix("sha256:"),
            )
            assert (await native_stores[1].load(session.id)).status == "completed"
            assert isinstance(
                await execution_reader.lookup(execution_permit.expected), ExactUnavailable
            )
    assert execution_reader is not None
    assert isinstance(await execution_reader.lookup(execution_reader.expected), ExactMatch)
    assert len(provider.requests) == 1
    assert events
    completed = await native_stores[1].load(session.id)
    assert completed.status == "completed"
    # Native release advances the epoch; producer identity remains bound to
    # the admitted epoch rather than the later terminal session snapshot.
    with _invocation_lifecycle_authority_read_scope():
        checkpoint = await native_stores[1].load_checkpoint(session.id)
    index = NativeProducerIndex.model_validate(checkpoint[ROOT_KEY])
    assert index.state == "admitted"
    assert index.invocation.run_epoch == 1
    assert index.invocation.launch.registration == proposal.operation
    assert completed.run_epoch > index.invocation.run_epoch
    release = await native_stores[1]._read_native_producer_release(proposal)
    assert release == await native_stores[1]._read_native_producer_release(proposal)
    assert release.run_epoch == index.invocation.run_epoch
    assert release.interaction_id == index.invocation.interaction_id
    assert release.registration == proposal
    from cayu.budgets.base import InMemoryBudgetLedger

    receiver = application._request_coordinator._registration.receiving_owner

    async def forbid_new_budget_authority(**kwargs):
        raise AssertionError("Cleanup must not resolve new budget authority")

    with monkeypatch.context() as patch:
        patch.setattr(receiver, "_resolve_budget", forbid_new_budget_authority)
        patch.setattr(application, "budget_ledger", InMemoryBudgetLedger())
        await receiver._read_producer_budget_registration(registration)
        accounting = await receiver._read_producer_budget_settlement(registration)
        assert accounting == await receiver._read_producer_budget_settlement(registration)
        assert accounting.reservation_count > 0
        assert accounting.registration == proposal.operation
    inventory = await application.budget_ledger._scan_reservation_records(session_id=session.id)
    assert inventory
    assert all(row.session_id == session.id for row in inventory)
    # Faults in retained accounting must not turn terminal native status into
    # cleanup permission. The ledger is the fixture's actual in-memory owner.
    from cayu.budgets.base import budget_settlement_id

    ledger = application.budget_ledger
    row = inventory[0]
    settlement_id = budget_settlement_id(row.reservation_id)
    saved_settlement = ledger._settlements[settlement_id]
    try:
        ledger._settlements[settlement_id] = saved_settlement.model_copy(
            update={"event_published": False}
        )
        with pytest.raises(ValueError, match="evidence is unavailable"):
            await receiver._read_producer_budget_settlement(registration)
    finally:
        ledger._settlements[settlement_id] = saved_settlement
    try:
        ledger._records[row.reservation_id] = row.model_copy(update={"status": "active"})
        with pytest.raises(ValueError, match="remains unsettled"):
            await receiver._read_producer_budget_settlement(registration)
        ledger._records[row.reservation_id] = row.model_copy(
            update={
                "settlement_event_payload": {
                    **row.settlement_event_payload,
                    "budget_binding_id": "foreign",
                },
            }
        )
        with pytest.raises(ValueError, match="original authority"):
            await receiver._read_producer_budget_settlement(registration)
    finally:
        ledger._records[row.reservation_id] = row
    assert await receiver._read_producer_budget_settlement(registration) == accounting
    with pytest.raises(ValueError):
        await native_stores[1]._read_native_producer_release(
            proposal.model_copy(update={"binding_incarnation": "replacement"})
        )
    from hashlib import sha256

    from cayu.collaboration._preparation import contract_bytes
    from cayu.runtime._producer_output_store import NativeProducerOutput
    from cayu.vaults.redaction import SecretRedactor

    output = NativeProducerOutput.model_validate(
        await native_stores[1].load_session_operation(session.id, index.operation_key + ":output")
    )
    assert output.registration == proposal
    assert output.disposition == ("oversized" if oversized else "answer")
    assert output.source_indices == (() if oversized else (2,))
    assert output.interaction_id == index.invocation.interaction_id
    assert output.run_epoch == index.invocation.run_epoch
    assert (output.source_commitment is None) == oversized
    assert "private-producer-canary" not in output.model_dump_json()
    assert (
        index.output_commitment
        == "sha256:" + sha256(contract_bytes(output, redactor=SecretRedactor())).hexdigest()
    )
    assert await native_stores[1]._read_retained_native_producer_output(proposal) == output
    backend, address = native_stores[3]
    reopened = None
    if backend == "sqlite":
        from pathlib import Path

        from cayu.storage.sqlite import SQLiteSessionStore

        reopened = SQLiteSessionStore(Path(address).with_name("sessions.sqlite"))
    elif backend == "postgres":
        from cayu.storage.postgres import PostgresSessionStore

        reopened = PostgresSessionStore(address)
    if reopened is not None:
        try:
            assert await reopened._read_retained_native_producer_output(proposal) == output
        finally:
            await reopened.close()
    with pytest.raises(ValueError):
        await native_stores[1]._read_retained_native_producer_output(
            proposal.model_copy(update={"binding_incarnation": "replacement"})
        )
    assert len(provider.requests) == 1
    transcript = await native_stores[1].load_transcript(session.id)
    assert [message.role for message in transcript] == ["system", "user", "assistant"]
    assert any(part.type == "thinking" for part in transcript[-1].content)
    if not oversized:
        from cayu.collaboration._session_export_store import source_digest
        from cayu.sessions.base import TranscriptRecord

        visible = TranscriptRecord(
            index=2,
            interaction_id=output.interaction_id,
            message=transcript[-1].model_copy(
                update={"content": tuple(p for p in transcript[-1].content if p.type == "text")}
            ),
        )
        assert output.source_commitment == "sha256:" + source_digest((visible,))
    with pytest.raises(ValueError, match="producer output responsibility"):
        await native_stores[1].delete_session(session.id)

    from cayu.collaboration._producer_completion import retain_producer_completion
    from cayu.collaboration._producer_store import read_output_registration
    from cayu.collaboration._request_store import retained_request

    collaboration = native_stores[0]
    _, initialized = application._participant_coordinator._ready()
    async with collaboration._transaction(initialized.owner.application_scope, write=False) as tx:
        before = await collaboration._anchor(tx, initialized, application._secret_redactor)
    if answer_bytes == 14:
        from contextlib import asynccontextmanager

        from cayu.collaboration._producer_store import completion_operation
        from cayu.collaboration._request_store import operation_key
        from cayu.collaboration.participants import CollaborationUnavailable

        transaction = collaboration._transaction
        lost = False
        completion_key = operation_key(completion_operation(proposal, application._secret_redactor))

        @asynccontextmanager
        async def lose_completion_ack(scope, *, write):
            nonlocal lost
            committed_completion = False
            async with transaction(scope, write=write) as tx:
                yield tx
                if write and not lost:
                    committed_completion = await tx.get("operations", completion_key) is not None
            if committed_completion:
                lost = True
                raise RuntimeError("injected source completion acknowledgement loss")

        monkeypatch.setattr(collaboration, "_transaction", lose_completion_ack)
        with pytest.raises(CollaborationUnavailable):
            await retain_producer_completion(application, proposal)
        assert lost
        monkeypatch.setattr(collaboration, "_transaction", transaction)
    elif answer_bytes == 1024:
        import asyncio

        receiver = application._request_coordinator._registration.receiving_owner
        read_output = receiver._read_producer_output
        entered, release = asyncio.Event(), asyncio.Event()
        monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)

        async def paused_read(record):
            entered.set()
            await release.wait()
            return await read_output(record)

        monkeypatch.setattr(receiver, "_read_producer_output", paused_read)
        task = asyncio.create_task(retain_producer_completion(application, proposal))
        try:
            await asyncio.wait_for(entered.wait(), 60)
            task.cancel()
            task.cancel()
            assert task.cancelling() == 2
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            assert task.cancelled()
            async with collaboration._transaction(
                initialized.owner.application_scope, write=False
            ) as tx:
                unresolved = await read_output_registration(
                    tx, proposal, redactor=application._secret_redactor
                )
                held = await collaboration._anchor(tx, initialized, application._secret_redactor)
            assert unresolved.completion is None
            assert held.reserved_operations == before.reserved_operations
            with pytest.raises(ValueError, match="producer output responsibility"):
                await native_stores[1].delete_session(session.id)
        finally:
            owners = tuple(application._request_coordinator._owners.pending)
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            if owners:
                await asyncio.wait_for(asyncio.gather(*owners), 60)
            monkeypatch.setattr(receiver, "_read_producer_output", read_output)
    else:
        import asyncio

        first, second = await asyncio.gather(
            retain_producer_completion(application, proposal),
            retain_producer_completion(application, proposal),
        )
        assert first == second
    completion = await retain_producer_completion(application, proposal)
    assert completion.output == output
    assert completion.native_commitment == index.output_commitment
    assert completion == await retain_producer_completion(application, proposal)
    # A completed source handoff owns replay. Native output may subsequently be
    # unavailable after authenticated cleanup; no reader or producer may rerun.
    receiver = application._request_coordinator._registration.receiving_owner

    async def unavailable_native_output(record):
        raise AssertionError("Accepted completion replay must not reread native output")

    with monkeypatch.context() as replay_patch:
        replay_patch.setattr(receiver, "_read_producer_output", unavailable_native_output)
        assert await retain_producer_completion(application, proposal) == completion
        from cayu.collaboration._contracts import CollaborationConflict

        changed = proposal.model_copy(
            update={"binding_incarnation": "different-binding-incarnation"}
        )
        with pytest.raises(CollaborationConflict):
            await retain_producer_completion(application, changed)
    async with collaboration._transaction(initialized.owner.application_scope, write=False) as tx:
        record = await read_output_registration(tx, proposal, redactor=application._secret_redactor)
        after = await collaboration._anchor(tx, initialized, application._secret_redactor)
        pending = await retained_request(
            collaboration,
            tx,
            initialized,
            proposal.admission.expected.intent.request,
            proposal.admission.expected.initiator,
            application._secret_redactor,
        )
    assert record.completion == completion.operation
    independent = native_stores[2]()
    async with independent._transaction(initialized.owner.application_scope, write=False) as tx:
        restored = await read_output_registration(
            tx, proposal, redactor=application._secret_redactor
        )
    assert restored == record
    assert after.operation_count == before.operation_count + 1
    assert after.event_count == before.event_count + 1
    assert after.reserved_operations == before.reserved_operations - 1
    assert after.reserved_events == before.reserved_events - 1
    assert pending.state == "open" and pending.delivery == "pending"
    assert record.cleanup is None
    assert len(provider.requests) == 1
    with pytest.raises(ValueError, match="producer output responsibility"):
        await native_stores[1].delete_session(session.id)
    if answer_bytes == 14:
        from cayu.collaboration._contracts import OwnerRef
        from cayu.collaboration._producer_export import export_producer_output
        from cayu.collaboration.exports import SessionExportAccessContext
        from cayu.collaboration.mandates import ResourceSelector

        destination = proposal.destinations[0]
        audience = OwnerRef(
            application_scope=initialized.owner.application_scope,
            owner_id=destination.recipient.participant_id,
            incarnation=destination.recipient.incarnation,
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
        resolution = resolver.recipient.resolution
        actions = tuple(
            dict.fromkeys(
                (*resolution.principal.actions, "source", "expose", "publish", "administer")
            )
        )
        resolver.recipient.resolution = resolution.model_copy(
            update={
                "principal": resolution.principal.model_copy(
                    update={"actions": actions, "audiences": (initialized.owner, audience)}
                ),
                "chain": resolution.chain.model_copy(
                    update={
                        "entries": tuple(
                            entry.model_copy(
                                update={
                                    "actions": actions,
                                    "audiences": (initialized.owner, audience),
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
        export_context = SessionExportAccessContext(
            principal=resolver.recipient.context.principal, mandate=resolver.recipient.context
        )
        exporter = application._session_export_coordinator
        # This matrix targets ACK loss/cancellation after actual export dispatch,
        # not a competing observation timeout while qualifying all owner layers.
        monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)
        monkeypatch.setattr(exporter.owners, "observation_timeout", 60)
        export_once = exporter.export
        lost_export_ack = False

        async def lose_export_ack(*args, **kwargs):
            nonlocal lost_export_ack
            receipt = await export_once(*args, **kwargs)
            if not lost_export_ack:
                lost_export_ack = True
                raise RuntimeError("injected committed export acknowledgement loss")
            return receipt

        if export_failure == "ack":
            monkeypatch.setattr(exporter, "export", lose_export_ack)
            with pytest.raises(CollaborationUnavailable) as lost_ack_error:
                await export_producer_output(
                    application, proposal, destination.operation, context=export_context
                )
            if not lost_export_ack:
                raise lost_ack_error.value
        elif export_failure == "cancel":
            import asyncio
            import threading

            from cayu.collaboration._producer_export_store import read_export
            from cayu.collaboration.exports import SessionExportConflict

            projector = exporter.projectors[destination.projector]
            projector.release = threading.Event()
            monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)
            monkeypatch.setattr(exporter.owners, "observation_timeout", 60)

            async def entered_projection():
                while not projector.entered.is_set():
                    await asyncio.sleep(0.01)

            task = asyncio.create_task(
                export_producer_output(
                    application, proposal, destination.operation, context=export_context
                )
            )
            try:
                await asyncio.wait_for(entered_projection(), 60)
                task.cancel()
                task.cancel()
                assert task.cancelling() == 2
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
                assert task.cancelled()
                async with collaboration._transaction(
                    initialized.owner.application_scope, write=False
                ) as tx:
                    retained = await read_export(
                        tx, proposal, destination, redactor=application._secret_redactor
                    )
                assert retained.state == "prepared" and retained.publication is None
                assert application._request_coordinator._owners.pending
                assert exporter.owners.pending
                with pytest.raises((ValueError, SessionExportConflict)) as fenced:
                    await native_stores[1].delete_session(session.id)
                assert isinstance(
                    fenced.value, SessionExportConflict
                ) or "producer output responsibility" in str(fenced.value)
            finally:
                owned = tuple(
                    application._request_coordinator._owners.pending | exporter.owners.pending
                )
                projector.release.set()
                await asyncio.gather(task, return_exceptions=True)
                if owned:
                    await asyncio.wait_for(asyncio.gather(*owned), 60)
        exported = await export_producer_output(
            application, proposal, destination.operation, context=export_context
        )
        assert exported.state == "published"
        assert (
            await export_producer_output(
                application, proposal, destination.operation, context=export_context
            )
            == exported
        )
        assert await application.read_session_export(exported.request, context=export_context) == {
            "text": answer
        }
        projector = application._session_export_coordinator.projectors[destination.projector]
        assert projector.calls == 1
        policy = exporter.registration.policy
        async with policy.lock:
            policy.denied = {"initialize", "source", "export"}
        assert (
            await export_producer_output(
                application, proposal, destination.operation, context=export_context
            )
            == exported
        )
        assert projector.calls == 1
        async with policy.lock:
            policy.denied = {"readback", "expose"}
        with pytest.raises(CollaborationUnavailable):
            await export_producer_output(
                application, proposal, destination.operation, context=export_context
            )
        assert projector.calls == 1
        assert len(provider.requests) == 1
        async with collaboration._transaction(
            initialized.owner.application_scope, write=False
        ) as tx:
            still_pending = await retained_request(
                collaboration,
                tx,
                initialized,
                proposal.admission.expected.intent.request,
                proposal.admission.expected.initiator,
                application._secret_redactor,
            )
        assert still_pending.state == "open" and still_pending.delivery == "pending"
        from cayu.collaboration._producer_export_store import read_export

        async with independent._transaction(initialized.owner.application_scope, write=False) as tx:
            assert (
                await read_export(tx, proposal, destination, redactor=application._secret_redactor)
                == exported
            )
        from cayu.collaboration._producer_outcome import publish_producer_outcome

        async with policy.lock:
            policy.denied = set()
        elected = await publish_producer_outcome(
            application,
            proposal,
            destination_operation=destination.operation,
            context=export_context,
        )
        assert elected.command.outcome == "answered"
        async with independent._transaction(initialized.owner.application_scope, write=False) as tx:
            answered = await retained_request(
                independent,
                tx,
                initialized,
                proposal.admission.expected.intent.request,
                proposal.admission.expected.initiator,
                application._secret_redactor,
            )
            owned = await read_output_registration(
                tx, proposal, redactor=application._secret_redactor
            )
        assert answered.state == "answered" and answered.delivery == "pending"
        assert answered.outcome == elected
        assert owned.cleanup is None
        # Identical durable content is not a caller's publication authority.
        from cayu.collaboration._contracts import CollaborationContractError

        with pytest.raises(CollaborationContractError):
            await application.publish_collaboration_outcome(
                elected.command, context=resolver.recipient.context
            )
        from cayu.collaboration._producer_delivery import deliver_producer_output
        from cayu.collaboration._session_export_store import digest

        source = await application.lookup_session_export(exported.request, context=export_context)
        assert isinstance(source, ExactMatch)
        policy.register_export(
            source.receipt,
            payload_sha256=digest({"text": answer, "artifact_commitments": []}),
            consumer_id=destination.recipient.participant_id,
        )
        policy.allowed_receipts.add(proposal.operation.caller_key)
        append_once = application.append_peer_content
        append_calls = 0
        import asyncio

        peer_entered, peer_release = asyncio.Event(), asyncio.Event()

        async def lose_peer_ack(*args, **kwargs):
            nonlocal append_calls
            append_calls += 1
            receipt = await append_once(*args, **kwargs)
            assert receipt.status == "appended"
            if export_failure == "cancel":
                peer_entered.set()
                await peer_release.wait()
                return receipt
            raise RuntimeError("injected committed peer acknowledgement loss")

        monkeypatch.setattr(application, "append_peer_content", lose_peer_ack)
        if export_failure == "cancel":
            from cayu.collaboration._producer_delivery_store import read_delivery

            task = asyncio.create_task(
                deliver_producer_output(
                    application, proposal, destination.operation, context=export_context
                )
            )
            try:
                await asyncio.wait_for(peer_entered.wait(), 60)
                task.cancel()
                task.cancel()
                assert task.cancelling() == 2
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
                assert task.cancelled()
                assert application._request_coordinator._owners.pending
                async with collaboration._transaction(
                    initialized.owner.application_scope, write=False
                ) as tx:
                    pending_delivery = await read_delivery(
                        tx, proposal, destination, redactor=application._secret_redactor
                    )
                assert pending_delivery.receipt is None
                accepted = await native_stores[1].read_peer_content_attempt(pending_delivery.append)
                assert accepted is not None and accepted.status == "appended"
            finally:
                owners = tuple(application._request_coordinator._owners.pending)
                peer_release.set()
                await asyncio.gather(task, return_exceptions=True)
                if owners:
                    await asyncio.wait_for(asyncio.gather(*owners), 60)
        elif export_failure == "exclude":
            pending_delivery = await deliver_producer_output(
                application,
                proposal,
                destination.operation,
                context=export_context,
                prepare_only=True,
            )
            assert pending_delivery.receipt is None and append_calls == 0
        else:
            with pytest.raises(CollaborationUnavailable):
                await deliver_producer_output(
                    application, proposal, destination.operation, context=export_context
                )
        # Recovery records only the receiving fact, even after source disclosure
        # is revoked. It does not return the retained peer payload or redispatch.
        from tests.core.test_participant_identity import app as make_recovery_app

        from cayu.collaboration._producer_delivery_recovery import (
            ProducerDeliveryRecovery,
            reconcile_producer_delivery,
        )
        from cayu.collaboration._producer_recovery import pending_producer_outputs

        recovery_app = make_recovery_app(
            independent,
            application._participant_coordinator._registration,
            session_store=native_stores[1],
        )
        await recovery_app.initialize_collaboration()
        monkeypatch.setattr(recovery_app._request_coordinator._owners, "observation_timeout", 60)
        from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied

        with pytest.raises(CollaborationAccessDenied):
            await pending_producer_outputs(
                recovery_app,
                admission.prepared.recipient,
                context=CollaborationAccessContext(principal="unregistered"),
            )
        with pytest.raises(CollaborationContractError):
            await pending_producer_outputs(
                recovery_app,
                admission.prepared.recipient,
                context=CONTEXT,
                after=True,
            )
        cursor, discoveries, pages = 0, [], []
        for _ in range(16):
            page = await pending_producer_outputs(
                recovery_app,
                admission.prepared.recipient,
                context=CONTEXT,
                after=cursor,
                limit=1,
            )
            pages.append(page)
            discoveries.extend(page.items)
            assert answer not in page.model_dump_json()
            if page.next_cursor is None:
                break
            assert page.next_cursor > cursor
            cursor = page.next_cursor
        else:
            pytest.fail("Bounded producer discovery did not finish")
        assert len(discoveries) == 1
        assert any(not page.items and page.next_cursor is not None for page in pages)
        assert discoveries[0].recovery.registration == proposal.operation
        assert discoveries[0].destinations == (destination.operation,)
        recovery = ProducerDeliveryRecovery(
            **discoveries[0].recovery.model_dump(), destination=discoveries[0].destinations[0]
        )
        async with policy.lock:
            policy.denied = {"source", "readback", "expose", "export", "append"}
        status = await reconcile_producer_delivery(
            application, recovery, context=CONTEXT, exclude=export_failure == "ack"
        )
        if export_failure == "exclude":
            from cayu.collaboration.access import CollaborationAccessDenied

            assert status.state == "pending" and status.receipt_commitment is None
            policy.allow_cleanup = False
            with pytest.raises(CollaborationAccessDenied):
                await reconcile_producer_delivery(
                    application, recovery, context=CONTEXT, exclude=True
                )
            assert await native_stores[1].read_peer_content_attempt(pending_delivery.append) is None
            policy.allow_cleanup = True
        status = await reconcile_producer_delivery(
            application, recovery, context=CONTEXT, exclude=True
        )
        expected_state = "excluded" if export_failure == "exclude" else "appended"
        assert status.state == expected_state and status.receipt_commitment is not None
        assert answer not in status.model_dump_json()
        assert status == await reconcile_producer_delivery(application, recovery, context=CONTEXT)
        assert status == await reconcile_producer_delivery(recovery_app, recovery, context=CONTEXT)
        still_owned = await pending_producer_outputs(
            recovery_app, admission.prepared.recipient, context=CONTEXT
        )
        assert tuple(item.recovery for item in still_owned.items) == (discoveries[0].recovery,)
        from cayu.collaboration._contracts import CollaborationConflict

        with pytest.raises(CollaborationConflict):
            await reconcile_producer_delivery(
                application,
                recovery.model_copy(update={"registration_commitment": "sha256:" + "0" * 64}),
                context=CONTEXT,
            )
        async with policy.lock:
            policy.denied = set()
        assert append_calls == (0 if export_failure == "exclude" else 1)
        delivered = await deliver_producer_output(
            application, proposal, destination.operation, context=export_context
        )
        assert delivered.receipt is not None and delivered.receipt.status == expected_state
        if expected_state == "appended":
            assert delivered.receipt.occurrence.payload.text == answer
        assert (
            await deliver_producer_output(
                application, proposal, destination.operation, context=export_context
            )
            == delivered
        )
        receiving = await native_stores[1].read_peer_content_attempt(delivered.append)
        assert receiving is not None and receiving.queue_id == delivered.receipt.queue_id
        assert append_calls == (0 if export_failure == "exclude" else 1)
        from cayu.collaboration._contracts import ExactConflict, ExactUnavailable

        acceptance_reader = exporter.readers[exported.request.audience]
        accepted_export = await acceptance_reader.lookup(source.receipt)
        if expected_state == "appended":
            assert isinstance(accepted_export, ExactMatch)
            assert accepted_export.receipt.export_receipt == source.receipt
            assert accepted_export.receipt.receiving_owner == exported.request.audience
        else:
            assert isinstance(accepted_export, ExactUnavailable)
        assert isinstance(
            await acceptance_reader.lookup(
                source.receipt.model_copy(update={"event_id": "wrong-export-receipt"})
            ),
            ExactConflict,
        )
        # Acceptance of copied peer text cannot prove producer execution quiescence.
        assert isinstance(
            await acceptance_reader.settlement(source.receipt, owned.permit), ExactUnavailable
        )
        async with independent._transaction(initialized.owner.application_scope, write=False) as tx:
            from cayu.collaboration._producer_delivery_store import read_delivery

            assert (
                await read_delivery(
                    tx, proposal, destination, redactor=application._secret_redactor
                )
                == delivered
            )
            retained = await read_output_registration(
                tx, proposal, redactor=application._secret_redactor
            )
        assert retained.cleanup is None
        async with policy.lock:
            policy.denied = {"source", "readback", "expose", "export"}
        assert (
            await publish_producer_outcome(
                application,
                proposal,
                destination_operation=destination.operation,
                context=export_context,
            )
            == elected
        )
        assert projector.calls == 1 and len(provider.requests) == 1
        if expected_state == "appended":
            from cayu.collaboration.exports import SessionExportSettlementRequest

            resolution = resolver.recipient.resolution
            actions = tuple(dict.fromkeys((*resolution.principal.actions, "release")))
            resolver.recipient.resolution = resolution.model_copy(
                update={
                    "principal": resolution.principal.model_copy(update={"actions": actions}),
                    "chain": resolution.chain.model_copy(
                        update={
                            "entries": tuple(
                                entry.model_copy(update={"actions": actions})
                                for entry in resolution.chain.entries
                            )
                        }
                    ),
                }
            )
            async with policy.lock:
                policy.denied = set()
            settlement = SessionExportSettlementRequest(
                request=exported.request,
                mode="release",
                operation=exported.request.ref.operation.model_copy(
                    update={"caller_key": "producer-output-released"}
                ),
            )
            from cayu.collaboration._producer_settlement import read_producer_settlement

            with pytest.raises(CollaborationUnavailable, match="export responsibility"):
                await read_producer_settlement(application, proposal)
            from cayu.collaboration._host_output_selection import select_producer_output

            inspected = await application.inspect_producer_output(proposal, context=CONTEXT)
            assert isinstance(inspected, ExactMatch)
            assert inspected.receipt.destinations[0].delivery == "appended"
            assert inspected.receipt.destinations[0].export_cleanup is None
            assert select_producer_output(inspected.receipt).intent.action == "release_export"
            released = await application.settle_session_export(settlement, context=export_context)
            assert released.acceptance == accepted_export.receipt
            assert (
                await application.settle_session_export(settlement, context=export_context)
                == released
            )
            inspected = await application.inspect_producer_output(proposal, context=CONTEXT)
            assert isinstance(inspected, ExactMatch)
            assert inspected.receipt.destinations[0].export_cleanup == "released"
            assert select_producer_output(inspected.receipt).intent.action == "settle"
            settled_output = await read_producer_settlement(application, proposal)
            assert settled_output.registration == proposal.operation
            assert settled_output.completion == completion.operation
            assert tuple(item.destination for item in settled_output.destinations) == (
                destination.operation,
            )
            assert await read_producer_settlement(application, proposal) == settled_output
            await assert_cleanup_stays_pending(application, proposal, settled_output)
            with pytest.raises(ValueError, match="producer output responsibility"):
                await native_stores[1].delete_session(session.id)
            await assert_final_cleanup(
                application, proposal, completion, monkeypatch, lose_ack=True
            )
    elif answer_bytes > 1024:
        from cayu.collaboration._producer_outcome import publish_producer_outcome
        from cayu.collaboration.access import CollaborationAccessDenied
        from cayu.collaboration.exports import SessionExportAccessContext

        # Qualify failure publication, not the independently tested short
        # foreground observation deadline during persistent-store validation.
        monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)
        failure_context = SessionExportAccessContext(
            principal=resolver.recipient.context.principal, mandate=resolver.recipient.context
        )
        with pytest.raises(CollaborationAccessDenied):
            await publish_producer_outcome(application, proposal, context=failure_context)
        async with collaboration._transaction(
            initialized.owner.application_scope, write=False
        ) as tx:
            denied = await retained_request(
                collaboration,
                tx,
                initialized,
                proposal.admission.expected.intent.request,
                proposal.admission.expected.initiator,
                application._secret_redactor,
            )
        assert denied.state == "open" and denied.outcome is None
        resolution = resolver.recipient.resolution
        actions = tuple(dict.fromkeys((*resolution.principal.actions, "publish")))
        resolver.recipient.resolution = resolution.model_copy(
            update={
                "principal": resolution.principal.model_copy(update={"actions": actions}),
                "chain": resolution.chain.model_copy(
                    update={
                        "entries": tuple(
                            entry.model_copy(update={"actions": actions})
                            for entry in resolution.chain.entries
                        )
                    }
                ),
            }
        )
        elected = await publish_producer_outcome(application, proposal, context=failure_context)
        assert elected.command.outcome == "failed"
        assert elected.command.export is None and elected.command.destination is None
        assert (
            await publish_producer_outcome(application, proposal, context=failure_context)
            == elected
        )
        async with independent._transaction(initialized.owner.application_scope, write=False) as tx:
            failed = await retained_request(
                independent,
                tx,
                initialized,
                proposal.admission.expected.intent.request,
                proposal.admission.expected.initiator,
                application._secret_redactor,
            )
            owned = await read_output_registration(
                tx, proposal, redactor=application._secret_redactor
            )
        assert failed.state == "failed" and failed.delivery == "pending"
        assert owned.cleanup is None and not owned.exports
        assert len(provider.requests) == 1
        from cayu.collaboration._producer_settlement import read_producer_settlement

        settled_output = await read_producer_settlement(application, proposal)
        assert settled_output.registration == proposal.operation
        assert settled_output.completion == completion.operation
        assert settled_output.destinations == ()
        assert await read_producer_settlement(application, proposal) == settled_output
        await assert_cleanup_stays_pending(application, proposal, settled_output)
        await assert_final_cleanup(application, proposal, completion, monkeypatch, lose_ack=False)


async def assert_cleanup_stays_pending(application, proposal, evidence):
    from tests.core.test_participant_identity import CONTEXT

    from cayu.collaboration._producer_recovery import pending_producer_outputs
    from cayu.collaboration._producer_settlement import prepare_producer_cleanup
    from cayu.collaboration._producer_store import read_output_registration
    from cayu.collaboration._request_store import retained_request

    store, initialized = application._participant_coordinator._ready()
    cleanup = await prepare_producer_cleanup(application, proposal)
    assert cleanup.evidence == evidence
    assert await prepare_producer_cleanup(application, proposal) == cleanup
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        record = await read_output_registration(tx, proposal, redactor=application._secret_redactor)
        request = await retained_request(
            store,
            tx,
            initialized,
            proposal.admission.expected.intent.request,
            proposal.admission.expected.initiator,
            application._secret_redactor,
        )
    assert record.cleanup == cleanup
    assert record.reserved_operations > 0 and record.reserved_bytes > 0
    assert request.producer_settlement is None
    cursor, discovered = 0, []
    for _ in range(16):
        page = await pending_producer_outputs(
            application,
            proposal.admission.prepared.recipient,
            context=CONTEXT,
            after=cursor,
            limit=4,
        )
        discovered.extend(item.recovery.registration for item in page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert discovered == [proposal.operation]
    with pytest.raises(ValueError, match="producer output responsibility"):
        await application.session_store.delete_session(
            proposal.admission.prepared.target.session_id
        )


async def assert_final_cleanup(application, proposal, completion, monkeypatch, *, lose_ack):
    from tests.core.test_participant_identity import CONTEXT

    from cayu.collaboration._producer_cleanup import settle_producer_output
    from cayu.collaboration._producer_completion import retain_producer_completion
    from cayu.collaboration._producer_recovery import pending_producer_outputs
    from cayu.collaboration._producer_store import read_output_registration
    from cayu.collaboration._request_store import retained_request
    from cayu.collaboration.participants import CollaborationUnavailable

    store, initialized = application._participant_coordinator._ready()
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        pending = await read_output_registration(
            tx, proposal, redactor=application._secret_redactor
        )
    sessions = application.session_store
    sid = proposal.admission.prepared.target.session_id
    with pytest.raises(PermissionError):
        await sessions._complete_native_producer_cleanup(pending, authority=None)
    assert await sessions._read_completed_native_producer_cleanup(pending) is None
    receiver = application._request_coordinator._registration.receiving_owner
    if lose_ack:
        complete = receiver._complete_producer_cleanup

        async def commit_then_lose_ack(record, *, authority):
            await complete(record, authority=authority)
            raise RuntimeError("Injected native cleanup acknowledgement loss")

        with monkeypatch.context() as patch:
            patch.setattr(receiver, "_complete_producer_cleanup", commit_then_lose_ack)
            with pytest.raises(CollaborationUnavailable):
                await settle_producer_output(application, proposal)
        native = await sessions._read_completed_native_producer_cleanup(pending)
        assert native is not None
        await sessions.delete_session(sid)
        assert await sessions.load(sid) is None
        assert await sessions._read_completed_native_producer_cleanup(pending) == native
    final = await settle_producer_output(application, proposal)
    assert final.registration == proposal.operation
    assert await settle_producer_output(application, proposal) == final
    if not lose_ack:
        await sessions.delete_session(sid)
    assert await sessions.load(sid) is None
    assert await retain_producer_completion(application, proposal) == completion
    assert await settle_producer_output(application, proposal) == final
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        settled = await read_output_registration(
            tx, proposal, redactor=application._secret_redactor
        )
        request = await retained_request(
            store,
            tx,
            initialized,
            proposal.admission.expected.intent.request,
            proposal.admission.expected.initiator,
            application._secret_redactor,
        )
    assert settled.cleanup_ack == final.operation == request.producer_settlement
    assert request.delivery == final.delivery == ("published" if lose_ack else "excluded")
    assert settled.reserved_operations == settled.reserved_events == settled.reserved_bytes == 0
    cursor = 0
    for _ in range(16):
        page = await pending_producer_outputs(
            application,
            proposal.admission.prepared.recipient,
            context=CONTEXT,
            after=cursor,
            limit=4,
        )
        assert all(item.recovery.registration != proposal.operation for item in page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor


@pytest.mark.anyio
@pytest.mark.parametrize("failure_mode", ["provider", "tool", "opaque"])
async def test_producer_failure_without_answer_uses_native_settlement(
    native_stores, monkeypatch, failure_mode
):
    from tests.core.test_participant_identity import CONTEXT

    from cayu.collaboration._producer_contracts import ProducerNativeFailure, ProducerNativeOutput
    from cayu.collaboration._producer_store import read_output_registration
    from cayu.collaboration._request_store import retained_request
    from cayu.collaboration.exports import SessionExportAccessContext
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.providers.base import ModelStreamEvent
    from cayu.tools.base import Tool, ToolEffect, ToolSpec

    calls = []

    class FailingTool(Tool):
        spec = ToolSpec(
            name="fail",
            effect=ToolEffect.EXTERNAL if failure_mode == "opaque" else ToolEffect.NONE,
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx, args):
            calls.append(ctx.session_id)
            raise ValueError("private-tool-failure-canary")

    (
        application,
        resolver,
        admission,
        provider,
        session,
        initialized,
        proposal,
        execution,
    ) = await output_scenario(
        native_stores,
        with_exports=True,
        tools=() if failure_mode == "provider" else (FailingTool(),),
    )
    monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)
    primary = ValueError("private-producer-failure-canary")
    provider._batches = (
        ((ModelStreamEvent.error(str(primary), cause=primary), ModelStreamEvent.completed()),)
        if failure_mode == "provider"
        else (
            (
                ModelStreamEvent.tool_call(name="fail", id="call", arguments={}),
                ModelStreamEvent.completed(),
            ),
            # An error tool result is not an assistant answer. Empty final
            # output must fail the output contract rather than inventing text.
            (ModelStreamEvent.completed(),),
        )
    )
    await application.register_producer_output(
        proposal, execution, context=resolver.recipient.context
    )
    with pytest.raises(CollaborationUnavailable):
        await application.retain_producer_completion(proposal, context=CONTEXT)
    original = resolver.recipient.resolution
    actions = (*original.principal.actions, "execute", "publish")
    resolver.recipient.resolution = original.model_copy(
        update={
            "principal": original.principal.model_copy(update={"actions": actions}),
            "chain": original.chain.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(update={"actions": actions})
                        for entry in original.chain.entries
                    )
                }
            ),
        }
    )
    events = []
    failure = None
    try:
        async for event in application.execute_producer_output(
            proposal,
            execution,
            context=CONTEXT,
            producer_context=resolver.recipient.context,
        ):
            events.append(event)
    except Exception as error:
        failure = error
    native_failure = failure_mode != "tool"
    expected_requests = 2 if failure_mode == "tool" else 1
    assert len(provider.requests) == expected_requests
    assert calls == ([session.id] if failure_mode == "tool" else [])
    assert (await native_stores[1].load(session.id)).status == (
        "failed" if native_failure else "completed"
    ), (events, failure)
    if failure_mode == "tool":
        from cayu.events import EventType

        assert any(event.type is EventType.TOOL_CALL_FAILED for event in events)

    # Session status/events are insufficient without the owner's immutable
    # invocation-bound failure or output receipt. Refuse without publication.
    async def unavailable_receipt(self, session_id, receipt_id):
        return None

    with monkeypatch.context() as fault:
        fault.setattr(
            type(native_stores[1]),
            "_load_historical_interaction_settlement_record"
            if native_failure
            else "load_runtime_publication_receipt",
            unavailable_receipt,
        )
        with pytest.raises(CollaborationUnavailable):
            await application.retain_producer_completion(proposal, context=CONTEXT)
    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        pending = await read_output_registration(
            tx, proposal, redactor=application._secret_redactor
        )
        assert pending.completion is None
    recovered = None
    backend, address = native_stores[3]
    if backend != "memory":
        import asyncio
        import json
        import sys

        from cayu.collaboration._producer_contracts import ProducerCompletionRecord

        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tests.recovery.producer_completion_reader_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                child.communicate(
                    json.dumps(
                        {
                            "backend": backend,
                            "address": address,
                            "expected": proposal.model_dump(mode="json"),
                        }
                    ).encode()
                ),
                60,
            )
            assert child.returncode == 0, stderr.decode()
            recovered = ProducerCompletionRecord.model_validate_json(stdout)
        finally:
            if child.returncode is None:
                child.kill()
                await child.wait()
    completion = await application.retain_producer_completion(proposal, context=CONTEXT)
    if recovered is not None:
        assert recovered == completion
    assert isinstance(
        completion.output,
        ProducerNativeFailure if native_failure else ProducerNativeOutput,
    )
    assert completion.output.disposition == ("failed" if native_failure else "empty")
    assert completion.output.source_indices == () and completion.output.source_commitment is None
    assert "private-producer-failure-canary" not in completion.model_dump_json()
    assert "private-tool-failure-canary" not in completion.model_dump_json()
    assert await application.retain_producer_completion(proposal, context=CONTEXT) == completion
    context = SessionExportAccessContext(
        principal=resolver.recipient.context.principal, mandate=resolver.recipient.context
    )
    if failure_mode == "provider":
        from contextlib import asynccontextmanager

        from cayu.collaboration._producer_outcome_store import outcome_operation
        from cayu.collaboration._request_store import operation_key

        transaction = native_stores[0]._transaction
        election_key = operation_key(outcome_operation(proposal, application._secret_redactor))
        rejected = []

        @asynccontextmanager
        async def rollback_election(scope, *, write):
            async with transaction(scope, write=write) as tx:
                yield tx
                if write and await tx.get("operations", election_key) is not None:
                    rejected.append(True)
                    raise OSError("failure-election-publication")

        with monkeypatch.context() as fault:
            fault.setattr(native_stores[0], "_transaction", rollback_election)
            with pytest.raises(CollaborationUnavailable):
                await application.publish_producer_outcome(proposal, context=context)
        assert rejected == [True]
        # A failed later publication cannot replace the exact native failure or
        # authorize repeating the provider. The same completion is still owned.
        assert await application.retain_producer_completion(proposal, context=CONTEXT) == completion
        assert len(provider.requests) == expected_requests
    elected = await application.publish_producer_outcome(proposal, context=context)
    assert elected.command.outcome == "failed"
    assert elected.command.export is None and elected.command.destination is None
    assert await application.publish_producer_outcome(proposal, context=context) == elected
    async with native_stores[2]()._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        request = await retained_request(
            native_stores[0],
            tx,
            initialized,
            admission.expected.intent.request,
            admission.expected.initiator,
            application._secret_redactor,
        )
        owned = await read_output_registration(tx, proposal, redactor=application._secret_redactor)
    assert request.state == "failed" and request.delivery == "pending"
    assert owned.cleanup is None and not owned.exports
    assert len(provider.requests) == expected_requests
    assert not any(
        message.role == "assistant" and any(part.type == "text" for part in message.content)
        for message in await native_stores[1].load_transcript(session.id)
    )
    with pytest.raises(ValueError, match="producer output responsibility"):
        await native_stores[1].delete_session(session.id)

    # Native failure has no retained assistant-output record. Cleanup must use
    # the authenticated invocation release and original accounting, not require
    # fabricated successful output merely to discharge responsibility.
    if failure_mode == "provider":
        receiver = application._request_coordinator._registration.receiving_owner

        async def fail_cleanup(*args, **kwargs):
            raise OSError("native-failure-cleanup")

        with monkeypatch.context() as fault:
            fault.setattr(receiver, "_complete_producer_cleanup", fail_cleanup)
            with pytest.raises(CollaborationUnavailable):
                await application.settle_producer_output(proposal, context=CONTEXT)
        assert await application.retain_producer_completion(proposal, context=CONTEXT) == completion
        assert await application.publish_producer_outcome(proposal, context=context) == elected
        with pytest.raises(ValueError, match="producer output responsibility"):
            await native_stores[1].delete_session(session.id)
        assert len(provider.requests) == expected_requests
    finalized = await application.settle_producer_output(proposal, context=CONTEXT)
    assert finalized.delivery == "excluded"
    assert len(provider.requests) == expected_requests
    async with native_stores[2]()._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        settled = await read_output_registration(
            tx, proposal, redactor=application._secret_redactor
        )
        assert settled.cleanup_ack == finalized.operation
        assert (settled.reserved_operations, settled.reserved_events, settled.reserved_bytes) == (
            0,
            0,
            0,
        )
    await native_stores[1].delete_session(session.id)
    assert await application.settle_producer_output(proposal, context=CONTEXT) == finalized
    assert await application.retain_producer_completion(proposal, context=CONTEXT) == completion
    assert len(provider.requests) == expected_requests


@pytest.mark.anyio
@pytest.mark.parametrize("gate", ["input", "approval"])
async def test_host_releases_human_paused_producer_slot(native_stores, gate, monkeypatch):
    await test_producer_human_pause_is_not_failure(
        native_stores, gate, monkeypatch, through_host=True
    )


@pytest.mark.anyio
@pytest.mark.parametrize("gate", ["input", "approval"])
async def test_producer_human_pause_is_not_failure(
    native_stores, gate, monkeypatch, through_host=False
):
    from tests.core.test_participant_identity import CONTEXT

    from cayu import ToolPolicy, ToolPolicyDecision, ToolPolicyResult
    from cayu.collaboration._producer_completion import retain_producer_completion
    from cayu.collaboration._producer_outcome import publish_producer_outcome
    from cayu.collaboration._producer_registration import register_producer_output
    from cayu.collaboration._producer_store import read_output_registration
    from cayu.collaboration.exports import SessionExportAccessContext
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.events import EventType
    from cayu.providers.base import ModelStreamEvent
    from cayu.runtime._producer_execution import _ProducerExecution
    from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
    from cayu.tools.user_input import UserInputTool

    class Approval(ToolPolicy):
        @property
        def execution_profile_identity(self):
            return ExecutionProfileBehaviorIdentity(
                name="tests:producer-human-approval",
                behavior_version="1",
                implementation_version="1",
            )

        async def authorize(self, request):
            return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL, reason="Review")

    from cayu import Tool, ToolEffect, ToolResult, ToolSpec

    class ReviewedTool(Tool):
        spec = ToolSpec(
            name="reviewed",
            effect=ToolEffect.NONE,
            input_schema={"type": "object", "properties": {}},
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:producer-reviewed-tool",
                behavior_version="1",
                implementation_version="1",
            ),
        )

        async def run(self, ctx, args):
            return ToolResult(content="reviewed")

    (
        application,
        resolver,
        admission,
        provider,
        session,
        initialized,
        proposal,
        execution,
    ) = await output_scenario(
        native_stores,
        with_exports=True,
        tools=(UserInputTool(),) if gate == "input" else (ReviewedTool(),),
        tool_policy=Approval() if gate == "approval" else None,
        planned=through_host,
    )
    monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)
    provider._batches = (
        (
            ModelStreamEvent.tool_call(
                name="ask_user" if gate == "input" else "reviewed",
                id="question",
                arguments={"question": "Continue?"} if gate == "input" else {},
            ),
            ModelStreamEvent.completed(),
        ),
    )
    await register_producer_output(
        application, proposal, execution, context=resolver.recipient.context
    )
    original = resolver.recipient.resolution
    actions = (*original.principal.actions, "execute", "publish")
    resolver.recipient.resolution = original.model_copy(
        update={
            "principal": original.principal.model_copy(update={"actions": actions}),
            "chain": original.chain.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(update={"actions": actions})
                        for entry in original.chain.entries
                    )
                }
            ),
        }
    )
    if through_host:
        import asyncio

        from cayu.collaboration._host import (
            CollaborationHost,
            _HostRegistration,
            _ProducerExecutionRule,
            _ProducerSource,
        )
        from cayu.collaboration._host_ownership import HostOwnershipLimits
        from cayu.collaboration._host_producer_execution import HostProducerExecution

        page = await application.pending_producer_outputs(
            admission.prepared.recipient, context=CONTEXT
        )
        token = next(
            item.recovery for item in page.items if item.recovery.registration == proposal.operation
        )
        events = []
        execute = application.execute_producer_output

        async def capture(*args, **kwargs):
            async for event in execute(*args, **kwargs):
                events.append(event)
                yield event

        monkeypatch.setattr(application, "execute_producer_output", capture)
        async with CollaborationHost(
            application,
            _HostRegistration(
                limits=HostOwnershipLimits(1, 1, 2, 262144),
                producer_sources=(_ProducerSource(admission.prepared.recipient, CONTEXT),),
                producer_rules=(),
                producer_execution_rules=(
                    _ProducerExecutionRule(
                        HostProducerExecution(recovery=token), CONTEXT, resolver.recipient.context
                    ),
                ),
                observation_timeout_s=60,
                shutdown_timeout_s=60,
            ),
        ) as host:
            async with asyncio.timeout(180):
                while not events or host.inspect().pending:
                    await host.service_once()
                    for outcome in host._owned.inspect().completed:
                        if outcome.error is not None:
                            raise outcome.error
            assert host.inspect().uncertain == 0
            # Another discovery pass must not redispatch the paused producer.
            await host.service_once()
            assert host.inspect().active == host.inspect().uncertain == 0
    else:
        events = [
            event
            async for event in application._execute_participant_session(
                execution,
                participant=admission.prepared.recipient,
                context=CONTEXT,
                producer_output=_ProducerExecution(
                    application, proposal, resolver.recipient.context
                ),
            )
        ]
    assert any(event.type is EventType.INTERACTION_PAUSED for event in events)
    assert (await native_stores[1].load(session.id)).status == "interrupted"
    with pytest.raises(CollaborationUnavailable):
        await retain_producer_completion(application, proposal)
    with pytest.raises(CollaborationUnavailable):
        await publish_producer_outcome(
            application,
            proposal,
            context=SessionExportAccessContext(
                principal=resolver.recipient.context.principal, mandate=resolver.recipient.context
            ),
        )
    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        owned = await read_output_registration(tx, proposal, redactor=application._secret_redactor)
    assert owned.completion is None and owned.cleanup is None
    assert len(provider.requests) == 1

    from cayu import ToolApprovalDecision, ToolApprovalRequest, UserInputResponse

    provider._batches = (
        *provider._batches,
        (
            ModelStreamEvent.text_delta("answer after human resolution"),
            ModelStreamEvent.completed(),
        ),
    )
    if gate == "input":
        question = next(
            event for event in events if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        stream = application.resolve_user_input(
            UserInputResponse(
                session_id=session.id, input_id=question.payload["input_id"], answer="yes"
            ),
            context=CONTEXT,
        )
    else:
        question = next(
            event for event in events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        stream = application.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=session.id,
                approval_id=question.payload["approval_id"],
                tool_round_id=question.payload["tool_round_id"],
                tool_call_id=question.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
            context=CONTEXT,
        )
    resumed = [event async for event in stream]
    assert any(event.type is EventType.SESSION_COMPLETED for event in resumed)
    from cayu.sessions import _invocation_lifecycle as lifecycle
    from cayu.sessions._invocation_lifecycle import (
        InvocationLifecycleCommandKind,
        _invocation_lifecycle_receipt_ledger_from_checkpoint,
        _InvocationLifecycleReceiptLedger,
    )
    from cayu.sessions.checkpoints import INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY

    require_lineage = lifecycle.require_invocation_rebind_lineage
    observed = []

    def missing_lineage(checkpoint, **kwargs):
        observed.append(True)
        ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
        assert any(
            receipt.kind is InvocationLifecycleCommandKind.REBIND for receipt in ledger.receipts
        )
        incomplete = _InvocationLifecycleReceiptLedger(
            receipts=tuple(
                receipt
                for receipt in ledger.receipts
                if receipt.kind is not InvocationLifecycleCommandKind.REBIND
            )
        )
        incomplete_checkpoint = {
            **checkpoint,
            INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY: incomplete.model_dump(mode="json"),
        }
        return require_lineage(incomplete_checkpoint, **kwargs)

    with monkeypatch.context() as missing:
        missing.setattr(lifecycle, "require_invocation_rebind_lineage", missing_lineage)
        with pytest.raises(CollaborationUnavailable):
            await application.retain_producer_completion(proposal, context=CONTEXT)
    assert observed
    completion = await application.retain_producer_completion(proposal, context=CONTEXT)
    assert completion.output.disposition == "answer"
    assert completion.output.run_epoch > 1
    assert await application.retain_producer_completion(proposal, context=CONTEXT) == completion
    assert len(provider.requests) == 2

    from cayu.runtime._producer_output_store import read_retained_native_output

    # Readback must prove the producing epoch from durable lineage, including
    # through an independent connection without the live invocation object.
    backend, address = native_stores[3]
    reopened = native_stores[1]
    if backend == "sqlite":
        from pathlib import Path

        from cayu.storage.sqlite import SQLiteSessionStore

        reopened = SQLiteSessionStore(Path(address).with_name("sessions.sqlite"))
    elif backend == "postgres":
        from cayu.storage.postgres import PostgresSessionStore

        reopened = PostgresSessionStore(address)
    try:
        assert await read_retained_native_output(reopened, proposal) == completion.output
    finally:
        if reopened is not native_stores[1]:
            await reopened.close()

    from cayu.collaboration._request_store import retained_request
    from cayu.collaboration.requests import RequestControl

    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        snapshot = await retained_request(
            native_stores[0],
            tx,
            initialized,
            admission.expected.intent.request,
            admission.expected.initiator,
            application._secret_redactor,
        )
    assert snapshot is not None
    await application.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("close-resolved-producer"),
            expected=admission.expected,
            expected_revision=snapshot.revision,
            kind="cancel",
        ),
        context=resolver.sender.context,
    )
    settled = await application.settle_producer_output(proposal, context=CONTEXT)
    assert await application.settle_producer_output(proposal, context=CONTEXT) == settled
    assert len(provider.requests) == 2


@pytest.mark.anyio
@pytest.mark.parametrize("missing", ["projector", "policy", "reader", "namespace"])
async def test_producer_launch_refuses_unserviceable_delivery(native_stores, monkeypatch, missing):
    from tests.core.test_participant_identity import CONTEXT

    from cayu.collaboration._producer_registration import register_producer_output
    from cayu.collaboration._producer_store import read_output_registration
    from cayu.collaboration.participants import CollaborationUnavailable
    from cayu.runtime._producer_execution import _ProducerExecution

    (
        application,
        resolver,
        admission,
        provider,
        session,
        initialized,
        proposal,
        execution,
    ) = await output_scenario(native_stores, with_exports=True)
    await register_producer_output(
        application, proposal, execution, context=resolver.recipient.context
    )
    original = resolver.recipient.resolution
    actions = (*original.principal.actions, "execute")
    resolver.recipient.resolution = original.model_copy(
        update={
            "principal": original.principal.model_copy(update={"actions": actions}),
            "chain": original.chain.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(update={"actions": actions})
                        for entry in original.chain.entries
                    )
                }
            ),
        }
    )
    exports = application._session_export_coordinator
    if missing == "projector":
        monkeypatch.setattr(exports, "projectors", {})
    elif missing == "policy":
        monkeypatch.setattr(exports, "policy_ref", None)
    elif missing == "reader":
        monkeypatch.setattr(exports, "readers", {})
    else:
        reader = next(iter(exports.readers.values()))
        monkeypatch.setattr(
            reader,
            "_namespace",
            reader._namespace.model_copy(update={"generation": reader._namespace.generation + 1}),
        )
    handoff = _ProducerExecution(application, proposal, resolver.recipient.context)
    with pytest.raises(CollaborationUnavailable):
        _ = [
            event
            async for event in application._execute_participant_session(
                execution,
                participant=admission.prepared.recipient,
                context=CONTEXT,
                producer_output=handoff,
            )
        ]
    assert provider.requests == []
    assert (await native_stores[1].load(session.id)).run_epoch == 0
    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        retained = await read_output_registration(tx, proposal, redactor=SecretRedactor())
        assert retained.state == "registered" and retained.launch is None
        assert retained.cleanup is None and retained.reserved_operations > 0


@pytest.mark.anyio
@pytest.mark.parametrize("phase", ["exclude", "release"])
async def test_cancelled_control_retains_native_exclusion_owner(native_stores, monkeypatch, phase):
    import asyncio

    from cayu.collaboration._producer_registration import register_producer_output
    from cayu.collaboration._producer_store import read_output_registration
    from cayu.collaboration._request_store import retained_request
    from cayu.collaboration.requests import RequestControl

    (
        application,
        resolver,
        admission,
        provider,
        session,
        initialized,
        proposal,
        execution,
    ) = await output_scenario(native_stores, planned=True)
    await register_producer_output(
        application, proposal, execution, context=resolver.recipient.context
    )
    found = await application.collaboration_admission_reader().lookup(
        admission, context=resolver.recipient.context
    )
    receiver = application._request_coordinator._registration.receiving_owner
    # This case exercises explicit cancellation after the selected dispatch,
    # not a competing observation timeout during backend setup.
    monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)
    method = (
        "_exclude_producer_after_control" if phase == "exclude" else "_acknowledge_producer_cleanup"
    )
    operation = getattr(receiver, method)
    entered, release = asyncio.Event(), asyncio.Event()

    async def paused(*args):
        entered.set()
        await release.wait()
        return await operation(*args)

    monkeypatch.setattr(receiver, method, paused)
    task = asyncio.create_task(
        application.control_collaboration_request(
            RequestControl(
                operation=initialized.operation("cancel-observer"),
                expected=admission.expected,
                expected_revision=found.receipt.revision,
                kind="cancel",
            ),
            context=resolver.sender.context,
        )
    )
    store, redactor = native_stores[0], SecretRedactor()
    try:
        await asyncio.wait_for(entered.wait(), 60)
        task.cancel()
        task.cancel()
        assert task.cancelling() == 2
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert task.cancelled()
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            pending = await retained_request(
                store,
                tx,
                initialized,
                admission.expected.intent.request,
                admission.expected.initiator,
                redactor,
            )
            assert pending.state == "cancelled"
            assert pending.delivery == "pending"
            assert pending.producer_settlement is None
            record = await read_output_registration(tx, proposal, redactor=redactor)
            assert (record.cleanup is None) == (phase == "exclude")
            assert record.reserved_operations > 0
            permits = await store._permit_state(tx, admission.prepared.recipient, redactor)
            assert permits.outstanding >= 2
        assert provider.requests == []
        assert (await native_stores[1].load(session.id)).run_epoch == 0
        with pytest.raises(ValueError, match="producer output responsibility"):
            await native_stores[1].delete_session(session.id)
    finally:
        pending = tuple(application._request_coordinator._owners.pending)
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending), 30)
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        settled = await retained_request(
            store,
            tx,
            initialized,
            admission.expected.intent.request,
            admission.expected.initiator,
            redactor,
        )
        assert settled.delivery == "excluded" and settled.producer_settlement is not None
        record = await read_output_registration(tx, proposal, redactor=redactor)
        assert record.state == "excluded" and record.reserved_bytes == 0
    await application.drain_collaboration_requests()
