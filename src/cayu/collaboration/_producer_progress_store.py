"""Producer progress proof at the existing request arbitration transaction."""

from dataclasses import dataclass
from hashlib import sha256

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._preparation import contract_bytes, require_exact_contract
from cayu.collaboration._producer_store import read_request_output
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
)

_SEAL = object()


@dataclass(frozen=True)
class _ProducerProgressAuthority:
    expected: bytes
    expires_at_ms: int
    seal: object

    def require(self, command, *, now_ms, redactor):
        if (
            self.seal is not _SEAL
            or self.expected != contract_bytes(command, redactor=redactor)
            or type(self.expires_at_ms) is not int
            or not now_ms < self.expires_at_ms
        ):
            raise CollaborationConflict("Producer progress authority is unavailable.")


def _progress_authority(command, *, expires_at_ms, redactor):
    return _ProducerProgressAuthority(
        contract_bytes(command, redactor=redactor), expires_at_ms, _SEAL
    )


async def require_producer_progress(tx, prior, command, *, redactor):
    record = await read_request_output(tx, command.expected, redactor=redactor)
    if record is None:
        raise CollaborationUnavailable("Producer progress registration is unavailable.")
    registration = record.command
    prepared = registration.admission.prepared
    assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
    evidence = command.evidence
    require_exact_contract(registration.initiator, command.publisher, redactor=redactor)
    input_commitment = (
        prior.clarification.input_sha256
        or sha256(contract_bytes(prior.receipt.expected, redactor=redactor)).hexdigest()
    )
    profile = ExecutionProfileIdentity.model_validate_json(prepared.execution_profile_json)
    if (
        command.producer != registration.operation
        or prior.producer_operation != registration.operation
        or command.admission_generation != registration.publisher_generation
        or prior.admission_generation != registration.publisher_generation
        or prior.admission_operation != registration.admission.operation
        or evidence.registration_commitment
        != "sha256:" + sha256(contract_bytes(registration, redactor=redactor)).hexdigest()
        or (evidence.session_id, evidence.session_instance_id)
        != (prepared.target.session_id, prepared.target.session_instance_id)
        or evidence.profile_commitment != "sha256:" + profile.fingerprint
        or command.sequence > registration.limits.progress_occurrences
        or (
            registration.admission.expected_input_revision,
            registration.admission.expected_input_sha256,
        )
        != (prior.clarification.input_revision, input_commitment)
        or (command.kind != "prepared" and record.launch is None)
    ):
        raise CollaborationConflict("Producer progress conflicts with its admitted identity.")
    return record
