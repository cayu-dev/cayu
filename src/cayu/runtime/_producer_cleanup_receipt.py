"""Native cleanup receipt retained independently of the producing session.

The source accepts cleanup responsibility first. The native owner then releases
its exact retention and commits this receipt in the same transaction. Keeping it
outside session-owned rows lets the source reconcile lost acknowledgement after
session deletion without treating absence as proof of cleanup.
"""

from dataclasses import dataclass
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import (
    ProducerAdmittedCleanup,
    ProducerCleanupRecord,
    ProducerOutputRecord,
)
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget
from cayu.runtime._producer_release import release_from_snapshot
from cayu.sessions._producer_checkpoint import (
    ROOT_KEY,
    NativeProducerAttachment,
    NativeProducerIndex,
    attachment_index,
    attachment_operation_key,
)
from cayu.sessions._producer_cleanup_contract import (
    NativeProducerCleanupReceipt as NativeProducerCleanupReceipt,
)
from cayu.vaults.redaction import SecretRedactor

_SEAL = object()


def _commitment(value):
    return "sha256:" + sha256(contract_bytes(value, redactor=SecretRedactor())).hexdigest()


@dataclass(frozen=True)
class _NativeCleanupHandoff:
    registration: bytes
    cleanup: bytes
    seal: object

    def require(self, registration):
        if (
            self.seal is not _SEAL
            or self.registration != contract_bytes(registration.command, redactor=SecretRedactor())
            or self.cleanup != contract_bytes(registration.cleanup, redactor=SecretRedactor())
        ):
            raise PermissionError("Native cleanup requires exact source-owner acceptance.")


def _accepted_source_cleanup(registration):
    """Only the registered coordinator mints this after reading committed source state."""
    registration = prepare_contract(ProducerOutputRecord, registration, redactor=SecretRedactor())
    if not isinstance(registration.cleanup, (ProducerAdmittedCleanup, ProducerCleanupRecord)):
        raise ValueError("Native cleanup requires exact source acceptance.")
    return _NativeCleanupHandoff(
        contract_bytes(registration.command, redactor=SecretRedactor()),
        contract_bytes(registration.cleanup, redactor=SecretRedactor()),
        _SEAL,
    )


def cleanup_target(registration):
    registration = prepare_contract(ProducerOutputRecord, registration, redactor=SecretRedactor())
    cleanup = registration.cleanup
    if not isinstance(cleanup, (ProducerAdmittedCleanup, ProducerCleanupRecord)):
        raise ValueError("Producer has no accepted cleanup responsibility.")
    command = registration.command
    prepared = command.admission.prepared
    assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
    if cleanup.registration != command.operation or (
        isinstance(cleanup, ProducerAdmittedCleanup)
        and (
            cleanup.evidence.registration_commitment != _commitment(command)
            or cleanup.evidence.completion != registration.completion
        )
    ):
        raise ValueError("Native cleanup source identity conflicts.")
    return registration, prepared.target.session_id, attachment_operation_key(command)


def _release_identity(command, cleanup, session_id):
    if isinstance(cleanup, ProducerCleanupRecord):
        return "exclusion", None, None, _commitment(cleanup.exclusion)
    assert isinstance(cleanup, ProducerAdmittedCleanup)
    return (
        "invocation",
        str(uuid5(NAMESPACE_URL, f"cayu-participant-session:{session_id}:{command.execution_key}")),
        cleanup.evidence.native_release_run_epoch,
        cleanup.evidence.native_release_commitment,
    )


def read_cleanup_receipt(registration, raw):
    """Complete expected comparison remains possible after native session erasure."""
    registration, session_id, _ = cleanup_target(registration)
    if raw is None:
        return None
    receipt = prepare_contract(NativeProducerCleanupReceipt, raw, redactor=SecretRedactor())
    command = registration.command
    prepared = command.admission.prepared
    assert prepared is not None
    if (
        receipt.registration != command.operation
        or receipt.receiver != command.receiver
        or receipt.registration_commitment != _commitment(command)
        or receipt.source_cleanup_commitment != _commitment(registration.cleanup)
        or receipt.session_id != session_id
        or receipt.session_instance_id != prepared.target.session_instance_id
        or (
            receipt.mode,
            receipt.interaction_id,
            receipt.run_epoch,
            receipt.native_release_commitment,
        )
        != _release_identity(command, registration.cleanup, session_id)
    ):
        raise ValueError("Native cleanup receipt conflicts with the expected operation.")
    return receipt


def prepare_cleanup_publication(
    registration, *, authority, session, checkpoint, attachment, children
):
    """Called only under a backend-owned mutation snapshot; no foreign calls.

    Descendant cleanup is not inferred from parent termination. Until their exact
    settlement is composed here, any child keeps this native release fenced.
    """
    registration, session_id, _ = cleanup_target(registration)
    if type(authority) is not _NativeCleanupHandoff:
        raise PermissionError("Native cleanup lacks a registered source handoff.")
    authority.require(registration)
    if children:
        raise ValueError("Producer descendant cleanup remains unresolved.")
    cleanup = registration.cleanup
    redactor = SecretRedactor()
    retained = prepare_contract(NativeProducerAttachment, attachment, redactor=redactor)
    require_exact_contract(
        NativeProducerAttachment.from_registration(registration), retained, redactor=redactor
    )
    index = prepare_contract(NativeProducerIndex, checkpoint[ROOT_KEY], redactor=redactor)
    if isinstance(cleanup, ProducerCleanupRecord):
        expected = attachment_index(retained).model_copy(
            update={"state": "excluded", "exclusion_commitment": _commitment(cleanup.exclusion)}
        )
        require_exact_contract(expected, index, redactor=redactor)
        if (
            session is None
            or (session.id, session.instance_id) != (index.session_id, index.session_instance_id)
            or cleanup.exclusion.attachment_commitment != index.record_commitment
        ):
            raise ValueError("Producer exclusion belongs to another native attachment.")
    else:
        assert isinstance(cleanup, ProducerAdmittedCleanup)
        release = release_from_snapshot(registration.command, session, checkpoint, attachment)
        if (
            release.release_commitment != cleanup.evidence.native_release_commitment
            or release.run_epoch != cleanup.evidence.native_release_run_epoch
        ):
            raise ValueError("Producer native release changed after source acceptance.")
    mode, interaction, epoch, commitment = _release_identity(
        registration.command, cleanup, session_id
    )
    receipt = NativeProducerCleanupReceipt(
        mode=mode,
        registration=registration.command.operation,
        receiver=registration.command.receiver,
        registration_commitment=_commitment(registration.command),
        source_cleanup_commitment=_commitment(cleanup),
        session_id=session_id,
        session_instance_id=index.session_instance_id,
        interaction_id=interaction,
        run_epoch=epoch,
        native_release_commitment=commitment,
    )
    released = prepare_contract(
        NativeProducerIndex,
        index.model_copy(
            update={"cleanup_commitment": _commitment(receipt), "cleanup_receipt": receipt}
        ),
        redactor=redactor,
    )
    return (
        receipt,
        {**checkpoint, ROOT_KEY: released.model_dump(mode="json")},
        {
            index.operation_key + ":cleanup": cleanup.model_dump(mode="json"),
            index.operation_key + ":cleanup-ack": receipt.model_dump(mode="json"),
        },
    )


def require_cleanup_for_erasure(*, session, checkpoint, records):
    """Verify the protected native release and exact source-acceptance handshake."""
    from cayu.collaboration._producer_contracts import ProducerNativeOutput

    redactor = SecretRedactor()
    index = prepare_contract(NativeProducerIndex, checkpoint[ROOT_KEY], redactor=redactor)
    expected_keys = {
        index.operation_key,
        index.operation_key + ":cleanup",
        index.operation_key + ":cleanup-ack",
    }
    if index.output_commitment is not None:
        expected_keys.add(index.operation_key + ":output")
    if (
        index.state != "admitted"
        or index.cleanup_commitment is None
        or set(records) != expected_keys
    ):
        raise ValueError("Producer cleanup evidence is incomplete.")
    attachment = prepare_contract(
        NativeProducerAttachment, records[index.operation_key], redactor=redactor
    )
    cleanup = prepare_contract(
        ProducerAdmittedCleanup, records[index.operation_key + ":cleanup"], redactor=redactor
    )
    receipt = prepare_contract(
        NativeProducerCleanupReceipt,
        records[index.operation_key + ":cleanup-ack"],
        redactor=redactor,
    )
    command = attachment.command
    native = release_from_snapshot(command, session, checkpoint, attachment)
    if (
        cleanup.registration != command.operation
        or cleanup.evidence.registration_commitment != _commitment(command)
        or receipt.registration != command.operation
        or receipt.receiver != command.receiver
        or receipt.registration_commitment != _commitment(command)
        or receipt.source_cleanup_commitment != _commitment(cleanup)
        or index.cleanup_commitment != _commitment(receipt)
        or receipt.session_id != native.session_id
        or receipt.session_instance_id != native.session_instance_id
        or receipt.interaction_id != native.interaction_id
        or receipt.run_epoch != native.run_epoch
        or receipt.native_release_commitment != native.release_commitment
        or cleanup.evidence.native_release_commitment != native.release_commitment
        or cleanup.evidence.native_release_run_epoch != native.run_epoch
    ):
        raise ValueError("Producer cleanup receipt identity conflicts.")
    if index.output_commitment is not None:
        output = prepare_contract(
            ProducerNativeOutput, records[index.operation_key + ":output"], redactor=redactor
        )
        require_exact_contract(command, output.registration, redactor=redactor)
        if index.output_commitment != _commitment(output):
            raise ValueError("Producer cleanup output identity conflicts.")
