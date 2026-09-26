"""Private output registration transactions, with no foreign calls or launch rights.

These primitives require an authenticated registered owner. They are not exposed
until native attachment, closure, maintenance and writer fencing are integrated.
"""

from hashlib import sha256

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES, CollaborationConflict, ObjectRef
from cayu.collaboration._namespace_store import require_open_namespace
from cayu.collaboration._permit_store import (
    prepare_permit,
    register_permit_in_transaction,
    registered_receipt,
)
from cayu.collaboration._permits import (
    PermitCommand,
    PermitIntent,
    PermitRegistration,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._prepared_admission_store import require_prepared_admission_evidence
from cayu.collaboration._producer_contracts import (
    ProducerAdmittedCleanup,
    ProducerCleanupRecord,
    ProducerCompletionRecord,
    ProducerLaunchDecision,
    ProducerNativeExclusion,
    ProducerOutputRecord,
    ProducerOutputRegistration,
    ProducerRegistrationEvent,
    ProducerRequestIndex,
    prepare_native_production,
)
from cayu.collaboration._request_store import (
    operation_key,
    preflight_control,
    require_request_event,
    retained_request,
)
from cayu.collaboration.base import CollaborationStore, _Anchor, _Repository
from cayu.collaboration.participants import CollaborationInitialization, CollaborationUnavailable
from cayu.collaboration.requests import RequestAdmissionReceipt, RequestCommand, RequestSnapshot
from cayu.vaults.redaction import SecretRedactor


def request_output_index(command: ProducerOutputRegistration, redactor: SecretRedactor):
    return request_output_index_operation(command.admission.expected, redactor)


def request_output_index_operation(request: RequestCommand, redactor: SecretRedactor):
    original = request.operation
    key = "producer-request:" + sha256(contract_bytes(original, redactor=redactor)).hexdigest()
    return original.model_copy(update={"caller_key": key})


async def read_request_output(
    tx: _Repository, request: RequestCommand, *, redactor: SecretRedactor
) -> ProducerOutputRecord | None:
    """Read the request attachment under the caller's transaction fence.

    Absence means no registered attachment, not native exclusion or quiescence.
    An occupied but incomplete index is unavailable and may never become absence.
    """
    request = prepare_contract(RequestCommand, request, redactor=redactor)
    operation = request_output_index_operation(request, redactor)
    raw = await tx.get("operations", operation_key(operation))
    if raw is None:
        return None
    index = prepare_contract(ProducerRequestIndex, raw, redactor=redactor)
    require_exact_contract(operation, index.operation, redactor=redactor)
    record = prepare_contract(
        ProducerOutputRecord,
        await tx.get("operations", operation_key(index.registration)),
        redactor=redactor,
    )
    require_exact_contract(request, record.command.admission.expected, redactor=redactor)
    retained = await read_output_registration(tx, record.command, redactor=redactor)
    if retained is None:
        raise CollaborationUnavailable("Producer registration is unavailable.")
    return retained


def output_permit(command: ProducerOutputRegistration, redactor: SecretRedactor) -> PermitCommand:
    """Derive responsibility from the complete registered proposal, never caller permit data."""
    prepared = command.admission.prepared
    assert prepared is not None
    digest = sha256(contract_bytes(command.operation, redactor=redactor)).hexdigest()
    operation = command.operation.model_copy(update={"caller_key": "producer-permit:" + digest})
    settlement = command.operation.model_copy(update={"caller_key": "producer-settle:" + digest})
    return prepare_contract(
        PermitCommand,
        PermitCommand(
            operation=operation,
            source=command.receiver.owner,
            destination=command.receiver.owner,
            initiator=command.initiator,
            intent=PermitIntent(
                limits=command.admission.expected.intent.limits,
                request=PermitRegistration(
                    operation=operation,
                    participant=prepared.recipient,
                    expected_lifecycle_revision=prepared.lifecycle_revision,
                    expected_configuration_revision=prepared.configuration_revision,
                    admission_generation=prepared.admission_generation,
                    admission_commitment=sha256(
                        contract_bytes(command, redactor=redactor)
                    ).hexdigest(),
                    source_operation=command.operation,
                    target=ObjectRef(
                        owner=command.receiver.owner,
                        kind="producer_output",
                        object_id=command.execution_key,
                        incarnation=command.binding_incarnation,
                        revision=1,
                    ),
                    target_state="future",
                    effect_scope="producer_output",
                    required_settlement="quiescence",
                    settlement_operation=settlement,
                ),
            ),
        ),
        redactor=redactor,
    )


async def read_output_registration(tx, command, *, redactor):
    raw = await tx.get("operations", operation_key(command.operation))
    if raw is None:
        return None
    record = prepare_contract(ProducerOutputRecord, raw, redactor=redactor)
    require_exact_contract(command, record.command, redactor=redactor)
    require_exact_contract(output_permit(command, redactor), record.permit, redactor=redactor)
    if await registered_receipt(tx, record.permit, redactor) is None:
        raise CollaborationUnavailable("Producer registration responsibility is unavailable.")
    index_operation = request_output_index(command, redactor)
    index = prepare_contract(
        ProducerRequestIndex,
        await tx.get("operations", operation_key(index_operation)),
        redactor=redactor,
    )
    if (
        index.operation != index_operation
        or index.registration != command.operation
        or index.command_commitment
        != "sha256:" + sha256(contract_bytes(command, redactor=redactor)).hexdigest()
    ):
        raise CollaborationUnavailable("Producer registration index conflicts.")
    event = prepare_contract(
        ProducerRegistrationEvent,
        await tx.get("operations", operation_key(record.event.operation)),
        redactor=redactor,
    )
    require_exact_contract(record.event, event, redactor=redactor)
    if record.launch is not None:
        launch = prepare_contract(
            ProducerLaunchDecision,
            await tx.get("operations", operation_key(record.launch.operation)),
            redactor=redactor,
        )
        require_exact_contract(record.launch, launch, redactor=redactor)
        if launch.command_commitment != index.command_commitment:
            raise CollaborationUnavailable("Producer launch authority conflicts.")
    if record.cleanup is not None:
        cleanup = prepare_contract(
            type(record.cleanup),
            await tx.get("operations", operation_key(record.cleanup.operation)),
            redactor=redactor,
        )
        require_exact_contract(record.cleanup, cleanup, redactor=redactor)
        if isinstance(cleanup, ProducerAdmittedCleanup) and (
            cleanup.evidence.registration_commitment != index.command_commitment
            or cleanup.sequence <= record.event.sequence
        ):
            raise CollaborationUnavailable("Producer cleanup authority conflicts.")
    if record.completion is not None:
        completion = prepare_contract(
            ProducerCompletionRecord,
            await tx.get("operations", operation_key(record.completion)),
            redactor=redactor,
        )
        require_exact_contract(command, completion.output.registration, redactor=redactor)
        if (
            completion.operation != record.completion
            or completion.operation != completion_operation(command, redactor)
            or record.launch is None
            or completion.sequence <= record.launch.sequence
            or completion.native_commitment
            != "sha256:" + sha256(contract_bytes(completion.output, redactor=redactor)).hexdigest()
        ):
            raise CollaborationUnavailable("Producer completion representations conflict.")
        if isinstance(record.cleanup, ProducerAdmittedCleanup) and (
            record.cleanup.evidence.completion_commitment
            != "sha256:" + sha256(contract_bytes(completion, redactor=redactor)).hexdigest()
            or record.cleanup.sequence <= completion.sequence
        ):
            raise CollaborationUnavailable("Producer cleanup completion conflicts.")
    if record.exports:
        from cayu.collaboration._producer_export_store import export_operation, read_export

        destinations = {export_operation(item, redactor): item for item in command.destinations}
        for operation in record.exports:
            destination = destinations.get(operation)
            if destination is None:
                raise CollaborationUnavailable("Producer export index has an unknown destination.")
            exported = await read_export(tx, command, destination, redactor=redactor)
            if exported is None or exported.completion != record.completion:
                raise CollaborationUnavailable("Producer export index is incomplete.")
    if record.deliveries:
        from cayu.collaboration._producer_delivery_store import delivery_operation, read_delivery

        destinations = {delivery_operation(item, redactor): item for item in command.destinations}
        for operation in record.deliveries:
            destination = destinations.get(operation)
            if (
                destination is None
                or await read_delivery(tx, command, destination, redactor=redactor) is None
            ):
                raise CollaborationUnavailable("Producer delivery index conflicts.")
    if record.exclusions:
        from cayu.collaboration._producer_destination_exclusion import (
            exclusion_operation,
            read_destination_exclusion,
        )

        destinations = {exclusion_operation(item, redactor): item for item in command.destinations}
        for operation in record.exclusions:
            destination = destinations.get(operation)
            if (
                destination is None
                or await read_destination_exclusion(tx, command, destination, redactor=redactor)
                is None
            ):
                raise CollaborationUnavailable("Producer exclusion index conflicts.")
    if record.cleanup_ack is not None:
        from cayu.collaboration._producer_cleanup_finalization import read_finalization

        await read_finalization(tx, record, redactor=redactor)
    return record


def completion_operation(command, redactor):
    return command.operation.model_copy(
        update={
            "caller_key": "producer-completion:"
            + sha256(contract_bytes(command.operation, redactor=redactor)).hexdigest()
        }
    )


async def accept_native_completion(store, tx, initialized, command, output, *, redactor):
    """Accept fixed native-owner readback; no foreign calls or disclosure in this transaction."""
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    output = prepare_native_production(output, redactor=redactor)
    require_exact_contract(command, output.registration, redactor=redactor)
    record = await read_output_registration(tx, command, redactor=redactor)
    if record is None or record.launch is None or record.state != "launch_claimed":
        raise CollaborationUnavailable("Producer completion lacks its launch responsibility.")
    operation = completion_operation(command, redactor)
    raw = await tx.get("operations", operation_key(operation))
    if record.completion is not None:
        receipt = prepare_contract(ProducerCompletionRecord, raw, redactor=redactor)
        require_exact_contract(output, receipt.output, redactor=redactor)
        return receipt
    if raw is not None:
        raise CollaborationConflict("Producer completion identity is occupied.")
    # Retention is mandatory even after request closure or authority expiry.
    # It cannot reopen the request, elect an outcome or release any native pin.
    anchor = await store._anchor(tx, initialized, redactor)
    receipt = ProducerCompletionRecord(
        operation=operation,
        output=output,
        native_commitment="sha256:" + sha256(contract_bytes(output, redactor=redactor)).hexdigest(),
        sequence=anchor.event_sequence + 1,
        recorded_at_ms=await tx.now_ms(),
    )
    reservation = 2 * MAX_ENVELOPE_BYTES
    updated = prepare_contract(
        ProducerOutputRecord,
        record.model_copy(
            update={
                "completion": operation,
                "reserved_operations": record.reserved_operations - 1,
                "reserved_events": record.reserved_events - 1,
                "reserved_bytes": record.reserved_bytes - reservation,
            }
        ),
        redactor=redactor,
    )
    charge = sum(len(contract_bytes(item, redactor=redactor)) for item in (updated, receipt)) - len(
        contract_bytes(record, redactor=redactor)
    )
    if charge > reservation:
        raise CollaborationUnavailable("Producer completion exceeds reserved bookkeeping.")
    updated_anchor = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 1,
                "event_count": anchor.event_count + 1,
                "event_sequence": receipt.sequence,
                "reserved_operations": anchor.reserved_operations - 1,
                "reserved_events": anchor.reserved_events - 1,
                "reserved_bytes": anchor.reserved_bytes - reservation,
                "retained_bytes": anchor.retained_bytes + charge,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated_anchor, ordinary=False)
    await tx.put("operations", operation_key(operation), receipt, insert=True)
    await tx.put("operations", operation_key(command.operation), updated, insert=False)
    await tx.put("anchors", (), updated_anchor, insert=False)
    return receipt


async def acknowledge_native_exclusion(
    store, tx, initialized, command, control, exclusion, *, redactor
):
    """Source acknowledgement of registered native-owner readback, never raw public proof."""
    from cayu.runtime._producer_output_store import native_exclusion

    record = await read_output_registration(tx, command, redactor=redactor)
    if record is None:
        raise CollaborationUnavailable("Producer responsibility is unavailable.")
    exclusion = prepare_contract(ProducerNativeExclusion, exclusion, redactor=redactor)
    require_exact_contract(native_exclusion(record, control)[3], exclusion, redactor=redactor)
    expected = command.admission.expected
    request = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    if (
        request is None
        or request.terminal != control
        or request.producer_operation != command.operation
    ):
        raise CollaborationConflict("Producer cleanup requires its exact retained closure.")
    if record.cleanup is not None:
        if not isinstance(record.cleanup, ProducerCleanupRecord):
            raise CollaborationConflict("Admitted producer cannot be excluded retroactively.")
        require_exact_contract(record.cleanup.exclusion, exclusion, redactor=redactor)
        if request.producer_settlement != record.cleanup_ack:
            raise CollaborationUnavailable("Producer cleanup representations disagree.")
        return record.cleanup
    operation = command.operation.model_copy(
        update={
            "caller_key": "producer-cleanup:"
            + sha256(contract_bytes(command.operation, redactor=redactor)).hexdigest()
        }
    )
    if await tx.get("operations", operation_key(operation)) is not None:
        raise CollaborationConflict("Producer cleanup key is occupied.")
    anchor = await store._anchor(tx, initialized, redactor)
    reservation = 2 * MAX_ENVELOPE_BYTES
    cleanup = ProducerCleanupRecord(
        operation=operation,
        registration=command.operation,
        sequence=anchor.event_sequence + 1,
        exclusion=exclusion,
    )
    updated = prepare_contract(
        ProducerOutputRecord,
        record.model_copy(
            update={
                "state": "excluded",
                "cleanup": cleanup,
                "reserved_operations": record.reserved_operations - 1,
                "reserved_events": record.reserved_events - 1,
                "reserved_bytes": record.reserved_bytes - reservation,
            }
        ),
        redactor=redactor,
    )
    updated_request = prepare_contract(
        RequestSnapshot,
        request.model_copy(
            update={
                "producer_settlement": None,
                "delivery": "pending",
            }
        ),
        redactor=redactor,
    )
    charge = sum(
        len(contract_bytes(item, redactor=redactor)) for item in (cleanup, updated, updated_request)
    ) - sum(len(contract_bytes(item, redactor=redactor)) for item in (record, request))
    if charge > reservation:
        raise CollaborationUnavailable("Producer cleanup exceeds its reserved capacity.")
    accounting = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 1,
                "event_count": anchor.event_count + 1,
                "event_sequence": cleanup.sequence,
                "reserved_operations": anchor.reserved_operations - 1,
                "reserved_events": anchor.reserved_events - 1,
                "reserved_bytes": anchor.reserved_bytes - reservation,
                "retained_bytes": anchor.retained_bytes + charge,
            }
        ),
        redactor=redactor,
    )
    require_capacity(accounting, ordinary=False)
    await tx.put("operations", operation_key(operation), cleanup, insert=True)
    await tx.put("operations", operation_key(command.operation), updated, insert=False)
    await tx.put("requests", operation_key(expected.operation), updated_request, insert=False)
    await tx.put("anchors", (), accounting, insert=False)
    return cleanup


async def claim_output_launch_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    command: ProducerOutputRegistration,
    *,
    authority_expires_at_ms: int,
    redactor: SecretRedactor,
) -> ProducerOutputRecord:
    """Order launch election against request closure, without foreign calls.

    Exact historical readback remains separate: returning an old launch decision
    here requires the request and current authority to still permit launch.
    """
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    record = await read_output_registration(tx, command, redactor=redactor)
    if record is None:
        raise CollaborationUnavailable("Producer launch requires registered responsibility.")
    anchor = await store._anchor(tx, initialized, redactor)
    await require_open_namespace(tx, anchor, command.operation, redactor)
    now = await tx.now_ms()
    if (
        type(authority_expires_at_ms) is not int
        or not now < authority_expires_at_ms <= 2**53 - 1
        or now >= command.limits.deadline_at_ms
    ):
        raise CollaborationConflict("Producer launch authority or deadline expired.")
    expected = command.admission.expected
    prior = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    if (
        prior is None
        or prior.state != "open"
        or prior.admission != "admitted"
        or prior.admission_operation != command.admission.operation
        or prior.admission_generation != command.admission.generation
    ):
        raise CollaborationConflict("Request no longer admits this producer launch.")
    require_exact_contract(expected, prior.receipt.expected, redactor=redactor)
    input_commitment = (
        prior.clarification.input_sha256
        or sha256(contract_bytes(prior.receipt.expected, redactor=redactor)).hexdigest()
    )
    if (command.admission.expected_input_revision, command.admission.expected_input_sha256) != (
        prior.clarification.input_revision,
        input_commitment,
    ):
        raise CollaborationConflict("Producer launch effective input changed.")
    prepared = command.admission.prepared
    assert prepared is not None
    participant = await store._participant(tx, prepared.recipient, initialized.owner, redactor)
    if (
        participant.lifecycle != "active"
        or participant.lifecycle_revision != prepared.lifecycle_revision
        or participant.configuration_revision != prepared.configuration_revision
        or participant.admission_generation != prepared.admission_generation
    ):
        raise CollaborationConflict("Producer launch participant authority changed.")
    if record.launch is not None:
        if now >= record.launch.authority_expires_at_ms:
            raise CollaborationConflict("Retained launch decision expired; reconcile native state.")
        return record
    operation = command.operation.model_copy(
        update={
            "caller_key": "producer-launch:"
            + sha256(contract_bytes(command.operation, redactor=redactor)).hexdigest()
        }
    )
    if await tx.get("operations", operation_key(operation)) is not None:
        raise CollaborationConflict("Producer launch identity is already occupied.")
    launch = ProducerLaunchDecision(
        operation=operation,
        registration=command.operation,
        command_commitment="sha256:"
        + sha256(contract_bytes(command, redactor=redactor)).hexdigest(),
        sequence=anchor.event_sequence + 1,
        request_revision=prior.revision,
        elected_at_ms=now,
        authority_expires_at_ms=min(authority_expires_at_ms, command.limits.deadline_at_ms),
    )
    reservation = 2 * MAX_ENVELOPE_BYTES
    updated = prepare_contract(
        ProducerOutputRecord,
        record.model_copy(
            update={
                "state": "launch_claimed",
                "launch": launch,
                "reserved_operations": record.reserved_operations - 1,
                "reserved_events": record.reserved_events - 1,
                "reserved_bytes": record.reserved_bytes - reservation,
            }
        ),
        redactor=redactor,
    )
    charge = sum(len(contract_bytes(item, redactor=redactor)) for item in (updated, launch)) - len(
        contract_bytes(record, redactor=redactor)
    )
    if charge > reservation:
        raise CollaborationUnavailable("Producer launch exceeds its reserved bookkeeping.")
    updated_anchor = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 1,
                "event_count": anchor.event_count + 1,
                "event_sequence": launch.sequence,
                "reserved_operations": anchor.reserved_operations - 1,
                "reserved_events": anchor.reserved_events - 1,
                "reserved_bytes": anchor.reserved_bytes - reservation,
                "retained_bytes": anchor.retained_bytes + charge,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated_anchor, ordinary=False)
    await tx.put("operations", operation_key(operation), launch, insert=True)
    await tx.put("operations", operation_key(command.operation), updated, insert=False)
    await tx.put("anchors", (), updated_anchor, insert=False)
    return updated


async def register_output_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    command: ProducerOutputRegistration,
    *,
    authority_expires_at_ms: int,
    redactor: SecretRedactor,
) -> ProducerOutputRecord:
    """Register once under current authority, before the native launch handshake."""
    command = prepare_contract(ProducerOutputRegistration, command, redactor=redactor)
    if (
        command.operation.application_scope != initialized.owner.application_scope
        or command.operation.namespace_incarnation != initialized.namespace_incarnation
        or command.receiver.owner != initialized.owner
    ):
        raise CollaborationConflict("Producer registration belongs to another owner.")
    anchor = await store._anchor(tx, initialized, redactor)
    replay = await read_output_registration(tx, command, redactor=redactor)
    if replay is not None:
        return replay
    await require_open_namespace(tx, anchor, command.operation, redactor)
    now = await tx.now_ms()
    if (
        type(authority_expires_at_ms) is not int
        or not now < authority_expires_at_ms <= 2**53 - 1
        or now >= command.limits.deadline_at_ms
    ):
        raise CollaborationConflict("Producer registration authority or deadline expired.")
    expected = command.admission.expected
    request = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    if (
        request is None
        or request.state != "open"
        or request.admission != "admitted"
        or request.admission_operation != command.admission.operation
    ):
        raise CollaborationConflict("Request does not admit producer registration.")
    require_exact_contract(expected, request.receipt.expected, redactor=redactor)
    admission = prepare_contract(
        RequestAdmissionReceipt,
        await tx.get("operations", operation_key(command.admission.operation)),
        redactor=redactor,
    )
    require_exact_contract(command.admission, admission.command, redactor=redactor)
    await require_request_event(tx, admission.event, redactor)
    await require_prepared_admission_evidence(tx, admission, redactor=redactor)
    prepared = command.admission.prepared
    assert prepared is not None
    participant = await store._participant(tx, prepared.recipient, initialized.owner, redactor)
    if (
        participant.lifecycle != "active"
        or participant.lifecycle_revision != prepared.lifecycle_revision
        or participant.configuration_revision != prepared.configuration_revision
        or participant.admission_generation != prepared.admission_generation
    ):
        raise CollaborationConflict("Producer participant authority changed.")
    index_operation = request_output_index(command, redactor)
    if index_operation == command.operation:
        raise CollaborationConflict("Producer registration uses its reserved index identity.")
    if await tx.get("operations", operation_key(index_operation)) is not None:
        raise CollaborationConflict("Request already has producer responsibility.")
    if request.producer_operation is not None:
        raise CollaborationUnavailable("Request producer index is inconsistent.")
    updated_request = prepare_contract(
        RequestSnapshot,
        request.model_copy(update={"producer_operation": command.operation}),
        redactor=redactor,
    )
    preflight_control(updated_request, redactor)
    # Native cleanup can precede the final source acknowledgement. Prove the
    # source's enlarged snapshot fits before dispatch, not after releasing the
    # native retention. This probe is never persisted as settlement evidence.
    from cayu.collaboration._producer_cleanup_finalization import finalization_operation

    preflight_control(
        prepare_contract(
            RequestSnapshot,
            updated_request.model_copy(
                update={"producer_settlement": finalization_operation(command, redactor)}
            ),
            redactor=redactor,
        ),
        redactor,
    )
    event_operation = command.operation.model_copy(
        update={
            "caller_key": "producer-event:"
            + sha256(contract_bytes(command.operation, redactor=redactor)).hexdigest()
        }
    )
    if (
        event_operation in (command.operation, index_operation)
        or await tx.get("operations", operation_key(event_operation)) is not None
    ):
        raise CollaborationConflict("Producer event identity is already occupied.")
    permit = prepare_permit(initialized, output_permit(command, redactor), redactor)
    if any(
        operation in (command.operation, index_operation, event_operation)
        for operation in (permit.operation, permit.intent.request.settlement_operation)
    ):
        raise CollaborationConflict("Producer responsibility identities conflict.")
    await register_permit_in_transaction(store, tx, initialized, permit, redactor)
    anchor = await store._anchor(tx, initialized, redactor)
    event = ProducerRegistrationEvent(
        operation=event_operation,
        registration=command.operation,
        sequence=anchor.event_sequence + 1,
        registered_at_ms=now,
    )
    # Conservative envelope reservations, consumed by later owned transitions.
    # Include launch/close/failure/release plus per-destination prepare/settle.
    slots = 8 + command.limits.progress_occurrences + 6 * len(command.destinations)
    record = ProducerOutputRecord(
        command=command,
        event=event,
        permit=permit,
        reserved_operations=slots,
        reserved_events=slots,
        reserved_bytes=slots * 2 * MAX_ENVELOPE_BYTES + command.limits.output_bytes,
    )
    # Byte reservations do not enlarge ContractValue's per-record envelope.
    # Validate the largest immutable index growth now, before native dispatch.
    from cayu.collaboration._producer_cleanup_store import preflight_cleanup
    from cayu.collaboration._producer_delivery_store import delivery_operation
    from cayu.collaboration._producer_destination_exclusion import exclusion_operation
    from cayu.collaboration._producer_export_store import export_operation

    peak_record = prepare_contract(
        ProducerOutputRecord,
        record.model_copy(
            update={
                "state": "launch_claimed",
                "launch": ProducerLaunchDecision(
                    operation=command.operation.model_copy(
                        update={
                            "caller_key": "producer-launch:"
                            + sha256(
                                contract_bytes(command.operation, redactor=redactor)
                            ).hexdigest()
                        }
                    ),
                    registration=command.operation,
                    command_commitment="sha256:"
                    + sha256(contract_bytes(command, redactor=redactor)).hexdigest(),
                    sequence=2**53 - 1,
                    request_revision=2**53 - 1,
                    elected_at_ms=now,
                    authority_expires_at_ms=command.limits.deadline_at_ms,
                ),
                "completion": completion_operation(command, redactor),
                "exports": tuple(export_operation(item, redactor) for item in command.destinations),
                "deliveries": tuple(
                    delivery_operation(item, redactor) for item in command.destinations
                ),
                "exclusions": tuple(
                    exclusion_operation(item, redactor) for item in command.destinations
                ),
            }
        ),
        redactor=redactor,
    )
    preflight_cleanup(peak_record, redactor)
    index = ProducerRequestIndex(
        operation=index_operation,
        registration=command.operation,
        command_commitment="sha256:"
        + sha256(contract_bytes(command, redactor=redactor)).hexdigest(),
    )
    retained = sum(
        len(contract_bytes(value, redactor=redactor)) for value in (record, index, event)
    )
    retained += len(contract_bytes(updated_request, redactor=redactor)) - len(
        contract_bytes(request, redactor=redactor)
    )
    updated = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 3,
                "event_count": anchor.event_count + 1,
                "event_sequence": event.sequence,
                "retained_bytes": anchor.retained_bytes + retained,
                "reserved_operations": anchor.reserved_operations + record.reserved_operations,
                "reserved_events": anchor.reserved_events + record.reserved_events,
                "reserved_bytes": anchor.reserved_bytes + record.reserved_bytes,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated, ordinary=True)
    await tx.put("operations", operation_key(command.operation), record, insert=True)
    await tx.put("operations", operation_key(index_operation), index, insert=True)
    await tx.put("operations", operation_key(event_operation), event, insert=True)
    await tx.put("requests", operation_key(expected.operation), updated_request, insert=False)
    await tx.put("anchors", (), updated, insert=False)
    return record
