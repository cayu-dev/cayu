"""Runtime-owned recipient creation responsibility and exact settlement."""

from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes
from cayu.collaboration._contracts import (
    ExactMatch,
    ExactUnavailable,
    InitiatorBinding,
    ObjectRef,
)
from cayu.collaboration._permit_store import prepare_permit
from cayu.collaboration._permits import (
    PermitCommand,
    PermitIntent,
    PermitRegistration,
    PermitSettlementReader,
    ReceivingSettlementReceipt,
)
from cayu.collaboration.participants import CollaborationUnavailable


def _digest(value):
    return sha256(canonical_durable_json_bytes(value, "recipient creation authority")).hexdigest()


async def prepare_continuation_admission(app, request, *, context):
    """Authenticate a selection proposal; no input append, permit or writer claim."""
    from cayu.collaboration._preparation import prepare_contract
    from cayu.collaboration.prepared_admission import (
        ContinueRecipientAdmissionTarget,
        PreparedRecipientAdmission,
        RecipientContinuationRequest,
        prepared_budget_request,
        prepared_budget_snapshot,
        require_secret_free_prepared,
    )

    request = prepare_contract(RecipientContinuationRequest, request, redactor=app._secret_redactor)
    receiver = app._request_coordinator.prepared_receiver_ref()
    inspection = await app._participant_coordinator.inspect(
        request.participant, context=context, action="administration"
    )
    participant = inspection.participant
    if participant.lifecycle != "active":
        raise PermissionError("Only active recipients can prepare admission.")
    from cayu.sessions._recipient_continuation import require_continuation_selection_store

    require_continuation_selection_store(app.session_store)
    selection = await app.session_store.capture_recipient_continuation(request.session_id)
    if (
        selection.session_id != request.session_id
        or selection.session_instance_id != request.session_instance_id
        or selection.participant != request.participant
    ):
        raise CollaborationUnavailable("Recipient continuation identity conflicts.")
    binding = await app._run_limit_controller.inspect_budget_binding(
        request=prepared_budget_request(
            session_id=selection.session_id,
            session_instance_id=selection.session_instance_id,
            profile=selection.execution_profile_json,
        )
    )
    evidence = PreparedRecipientAdmission(
        receiver=receiver,
        recipient=request.participant,
        lifecycle_revision=participant.lifecycle_revision,
        configuration_revision=participant.configuration_revision,
        admission_generation=participant.admission_generation,
        target=ContinueRecipientAdmissionTarget(selection=selection),
        execution_profile_json=selection.execution_profile_json,
        budget_binding_json=prepared_budget_snapshot(binding),
    )
    checked = prepare_contract(PreparedRecipientAdmission, evidence, redactor=app._secret_redactor)
    require_secret_free_prepared(checked, app._secret_redactor)
    return checked


async def prepare_request_admission(app, creation, *, context):
    """Read native child evidence without creating or admitting recipient work.

    This returns a proposal, not an authorization. The receiving owner rechecks
    native evidence and the permit owner arbitrates lifecycle at admission.
    """
    from dataclasses import replace

    from cayu.collaboration._preparation import prepare_contract
    from cayu.collaboration.prepared_admission import (
        PreparedRecipientAdmission,
        created_admission_target,
        prepared_budget_request,
        prepared_budget_snapshot,
        require_secret_free_prepared,
    )
    from cayu.sessions.context_views import RecipientSessionCreationRequest

    if type(creation) is not RecipientSessionCreationRequest:
        raise TypeError("Prepared admission requires typed recipient creation.")
    creation = replace(creation)
    receiver = app._request_coordinator.prepared_receiver_ref()
    found = await app.lookup_recipient_session(creation, context=context)
    if found is None:
        raise CollaborationUnavailable("Recipient creation is unavailable.")
    session, recipient_receipt = found
    receipt = recipient_receipt.participant_receipt
    inspected = await app.inspect_participant(creation.recipient, context=context)
    participant = inspected.participant
    if participant.lifecycle != "active":
        raise PermissionError("Only active recipients can prepare admission.")
    target = await admit_recipient_creation(
        app,
        creation.participant_request,
        creation.recipient,
        context,
        None,
        receipt.initial_input_commitment,
        receipt.binding.execution_profile_commitment,
        recovery=True,
    )
    binding = await app._run_limit_controller.inspect_budget_binding(
        request=prepared_budget_request(
            session_id=session.id,
            session_instance_id=session.instance_id,
            profile=receipt.execution_profile_json,
        )
    )
    evidence = PreparedRecipientAdmission(
        receiver=receiver,
        recipient=creation.recipient,
        lifecycle_revision=participant.lifecycle_revision,
        configuration_revision=participant.configuration_revision,
        admission_generation=participant.admission_generation,
        target=created_admission_target(target, receipt),
        execution_profile_json=receipt.execution_profile_json,
        budget_binding_json=prepared_budget_snapshot(binding),
    )
    checked = prepare_contract(PreparedRecipientAdmission, evidence, redactor=app._secret_redactor)
    require_secret_free_prepared(checked, app._secret_redactor)
    return checked


async def admit_recipient_creation(
    app,
    creation,
    participant,
    context,
    snapshot,
    input_commitment,
    profile_commitment,
    *,
    recovery=False,
    expected_target=None,
):
    from cayu.collaboration._preparation import prepare_contract, require_exact_contract
    from cayu.sessions.creation_fence import _SESSION_CREATION_AUTHORITY, SessionCreationTarget

    # A retained planner target is an expected operation, never permission to
    # register it. Detach it before awaiting and compare the complete resolved
    # native tuple before any receiving write or participant permit acquisition.
    if expected_target is not None:
        expected_target = prepare_contract(
            SessionCreationTarget, expected_target, redactor=app._secret_redactor
        )

    target, registered, store, initialized = await _prepare_recipient_creation_target(
        app,
        creation,
        participant,
        context,
        snapshot,
        input_commitment,
        profile_commitment,
        recovery=recovery,
    )
    if expected_target is not None:
        require_exact_contract(expected_target, target, redactor=app._secret_redactor)
    if not recovery:
        coordinator = app._participant_coordinator
        # Persist the complete expected receiving operation before the other
        # store can acquire responsibility. Preparation itself grants nothing:
        # the receiving store requires the later authenticated admission bit.
        await app.session_store._prepare_session_creation_target(
            target, authority=_SESSION_CREATION_AUTHORITY
        )
        if registered is None:
            await coordinator._store_result(
                store._register_permit(initialized, target.permit, redactor=app._secret_redactor)
            )
        await app.session_store._register_session_creation_target(
            target, authority=_SESSION_CREATION_AUTHORITY
        )
    return target


async def _prepare_recipient_creation_target(
    app,
    creation,
    participant,
    context,
    snapshot,
    input_commitment,
    profile_commitment,
    *,
    recovery=False,
):
    """Resolve the exact existing creation tuple without acquiring responsibility.

    This is a native preparation seam, not an admission or an exclusion receipt.
    The original receiving handoff still owns all mutations and authenticates
    lifecycle ordering through the durable participant permit.
    """
    from cayu.sessions.creation_fence import SessionCreationTarget

    coordinator = app._participant_coordinator
    store, initialized = coordinator._ready()
    key = "recipient-creation:" + _digest(creation.creation_key)
    operation = initialized.operation(key)
    admission = _digest(
        {
            "participant": participant.model_dump(mode="json"),
            "principal": context.principal,
            "creation_key": creation.creation_key,
            "requested_session_id": creation.request.session_id,
            "request_commitment": creation.request_commitment,
            "input_commitment": input_commitment,
            "profile_commitment": profile_commitment,
        }
    )
    registered = await store._lookup_registered_permit(
        initialized, operation, redactor=app._secret_redactor
    )
    if registered is not None:
        permit = prepare_permit(initialized, registered.expected, app._secret_redactor)
        if permit.intent.request.admission_commitment != admission:
            raise ValueError("Recipient creation conflicts with its admitted responsibility.")
        if (
            not recovery
            and snapshot.configuration_revision
            != permit.intent.request.expected_configuration_revision
        ):
            raise ValueError("Recipient configuration changed after creation admission.")
    else:
        if recovery:
            raise RuntimeError("Recipient creation responsibility is unavailable.")
        if snapshot.lifecycle != "active":
            raise PermissionError("Only active participants can admit recipient creation.")
        request = PermitRegistration(
            operation=operation,
            participant=participant,
            expected_lifecycle_revision=snapshot.lifecycle_revision,
            expected_configuration_revision=snapshot.configuration_revision,
            admission_generation=snapshot.admission_generation,
            admission_commitment=admission,
            source_operation=initialized.operation(key + ":source"),
            target=ObjectRef(
                owner=participant.owner,
                kind="recipient_creation",
                object_id=key,
                incarnation=operation.namespace_incarnation,
                revision=operation.generation,
            ),
            target_state="future",
            effect_scope="recipient_session_creation",
            required_settlement="exclusion",
            settlement_operation=initialized.operation(key + ":settled"),
        )
        permit = prepare_permit(
            initialized,
            PermitCommand(
                operation=operation,
                source=initialized.owner,
                destination=initialized.owner,
                initiator=InitiatorBinding(
                    issuer=initialized.owner,
                    principal=context.principal,
                    participant=ObjectRef(
                        owner=participant.owner,
                        kind="participant",
                        object_id=participant.participant_id,
                        incarnation=participant.incarnation,
                    ),
                    mandate=None,
                    invocation_id=None,
                    interaction_id=None,
                ),
                intent=PermitIntent(request=request, limits=initialized.binding.limits),
            ),
            app._secret_redactor,
        )
    target = SessionCreationTarget(
        permit=permit,
        receiving_owner=participant.owner,
        creation_key=creation.creation_key,
        requested_session_id=creation.request.session_id,
        request_commitment=creation.request_commitment,
        material_commitment=input_commitment,
        execution_identity_commitment=profile_commitment,
    )
    # Retain the same source owner/namespace across the read and subsequent
    # handoff. Re-resolving initialization after the await could switch epochs.
    return target, registered, store, initialized


def recipient_creation_settlement_id(target):
    return "recipient:" + _digest(target.model_dump(mode="json"))


class RecipientCreationSettlementReader(PermitSettlementReader):
    def __init__(self, store, target):
        self.store, self.target = store, target

    @property
    def owner(self):
        return self.target.receiving_owner

    async def lookup(self, expected):
        if expected != self.target.permit:
            raise ValueError("Recipient settlement authority conflicts.")
        found = await self.store.read_session_creation_decision(self.target)
        if not isinstance(found, ExactMatch):
            return ExactUnavailable()
        decision = found.receipt
        if decision.state == "pending":
            return ExactUnavailable()
        return ExactMatch[ReceivingSettlementReceipt](
            receipt=ReceivingSettlementReceipt(
                expected=expected,
                receiving_owner=self.owner,
                receipt_id=recipient_creation_settlement_id(self.target),
                outcome="quiescent" if decision.state == "created" else "excluded",
                admission_excluded=decision.state == "excluded",
            )
        )


async def settle_recipient_creation(app, target):
    from cayu.sessions.creation_fence import _SESSION_CREATION_AUTHORITY

    coordinator = app._participant_coordinator
    store, initialized = coordinator._ready()
    reader = RecipientCreationSettlementReader(app.session_store, target)
    found = await reader.lookup(target.permit)
    if not isinstance(found, ExactMatch):
        raise CollaborationUnavailable("Recipient creation settlement is unavailable.")
    # Permanent receiving exclusion also fences registration that never reached
    # the source store. This is positive exclusion, not inferred absence.
    settle = store._exclude_permit if found.receipt.proves_exclusion else store._settle_permit
    await coordinator._store_result(
        settle(
            initialized,
            target.permit,
            reader=reader,
            redactor=app._secret_redactor,
        )
    )
    # A lost acknowledgement on either side leaves the target discoverable.
    # Replaying source settlement is exact and idempotent before this final ACK.
    await app.session_store._acknowledge_session_creation_settlement(
        target, authority=_SESSION_CREATION_AUTHORITY
    )
