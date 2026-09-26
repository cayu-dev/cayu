"""Exact native absence exclusion, using reserved producer control capacity."""

from hashlib import sha256

from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES, MAX_ID_BYTES, OwnerRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._session_export_publication import publish_export_mutation
from cayu.collaboration._session_export_store import (
    ExportFutureExclusion,
    operation_key,
    read_scope,
)
from cayu.collaboration.exports import (
    SessionExportConflict,
    SessionExportRef,
    SessionExportRequest,
    SessionExportUnavailable,
)


def expected_exclusion(command, closure, intent, *, redactor):
    return ExportFutureExclusion(
        request=intent.request,
        registration=command.operation,
        registration_commitment=sha256(contract_bytes(command, redactor=redactor)).hexdigest(),
        closure_commitment=sha256(contract_bytes(closure, redactor=redactor)).hexdigest(),
        initiator=intent.initiator,
    )


def read_exclusion(raw, command, closure, intent, *, redactor):
    if type(raw) is not dict or raw.get("mode") != "producer_export_exclusion":
        return None
    retained = prepare_contract(ExportFutureExclusion, raw, redactor=redactor)
    require_exact_contract(
        expected_exclusion(command, closure, intent, redactor=redactor), retained, redactor=redactor
    )
    return retained


def preflight_exclusion_capacity(exports, command):
    """The source registration reserves one bounded negative-control slot per destination.

    These slots cannot be consumed by optional content exports. Their finite
    count is part of the native producer attachment, registered before dispatch.
    No new participant permit, content pin or disclosure is created by exclusion.
    """
    prepared = command.admission.prepared
    assert prepared is not None and exports.owner is not None
    for destination in command.destinations:
        request = SessionExportRequest(
            ref=SessionExportRef(
                session_id=prepared.target.session_id,
                session_instance_id=prepared.target.session_instance_id,
                operation=command.operation.model_copy(
                    update={
                        "application_scope": exports.owner.application_scope,
                        "namespace_incarnation": "\x01" * MAX_ID_BYTES,
                        "generation": 2**53 - 1,
                        "caller_key": "producer:" + "f" * 64,
                    }
                ),
            ),
            source_indices=tuple(range(2**53 - 16, 2**53)),
            source_selection="assistant_visible_text_v1",
            audience=OwnerRef(
                application_scope=destination.recipient.owner.application_scope,
                owner_id=destination.recipient.participant_id,
                incarnation=destination.recipient.incarnation,
            ),
            projector=destination.projector,
            policy=destination.disclosure_policy,
        )
        ExportFutureExclusion(
            request=request,
            registration=command.operation,
            registration_commitment="f" * 64,
            closure_commitment="f" * 64,
            initiator=command.initiator.model_copy(update={"mandate": destination.mandate}),
        )


async def exclude_absent_export(exports, session, command, closure, intent, *, authority):
    """CAS missing → excluded, competing with the first ordinary export preparation.

    Returns None only if a real native export/preparation won; its existing owner
    must then settle that record. Missing evidence alone never returns success.
    """
    request = intent.request
    key = operation_key(request.ref.operation)

    async def raw_record():
        with read_scope(session.id):
            return await exports.store.load_session_operation(session.id, key)

    for _ in range(4):
        root = await exports.root(session)
        if root is None:
            raise SessionExportUnavailable()
        exports.validate_namespace(root, request)
        raw = await raw_record()
        retained = read_exclusion(raw, command, closure, intent, redactor=exports.redactor)
        if retained is not None:
            return retained
        if raw is not None:
            return None
        if intent.state != "prepared":
            # Losing an acknowledged publication is not an absent future export.
            raise SessionExportUnavailable()
        if root.producer_exclusion_count >= len(command.destinations):
            raise SessionExportConflict()
        proposed = expected_exclusion(command, closure, intent, redactor=exports.redactor)
        desired = root.model_copy(
            update={
                "producer_exclusion_count": root.producer_exclusion_count + 1,
                "producer_exclusion_bytes": root.producer_exclusion_bytes + MAX_ENVELOPE_BYTES,
            }
        )
        try:
            await publish_export_mutation(
                exports.store,
                session,
                root,
                desired,
                key,
                proposed.model_dump(mode="json"),
                [],
                commit_guard=lambda _now: authority.require(exports, command, closure, intent),
            )
            return proposed
        except Exception as error:
            raw = await raw_record()
            retained = read_exclusion(raw, command, closure, intent, redactor=exports.redactor)
            if retained is not None:
                return retained
            if raw is not None:
                return None
            if not isinstance(error, SessionExportConflict):
                raise
    raise SessionExportConflict()
