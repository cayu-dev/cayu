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
):
    from cayu.sessions.creation_fence import _SESSION_CREATION_AUTHORITY, SessionCreationTarget

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
    if not recovery:
        # Persist the complete expected receiving operation before the other
        # store can acquire responsibility. Preparation itself grants nothing:
        # the receiving store requires the later authenticated admission bit.
        await app.session_store._prepare_session_creation_target(
            target, authority=_SESSION_CREATION_AUTHORITY
        )
        if registered is None:
            await coordinator._store_result(
                store._register_permit(initialized, permit, redactor=app._secret_redactor)
            )
        await app.session_store._register_session_creation_target(
            target, authority=_SESSION_CREATION_AUTHORITY
        )
    return target


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
                receipt_id="recipient:" + _digest(self.target.model_dump(mode="json")),
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
