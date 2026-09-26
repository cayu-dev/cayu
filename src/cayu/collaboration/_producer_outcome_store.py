"""Producer-specific proof checks at the existing request election boundary."""

from dataclasses import dataclass
from hashlib import sha256

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import ProducerCompletionRecord, ProducerOutputRecord
from cayu.collaboration._producer_export_store import read_export
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import ProducerOutcomeCommand

_SEAL = object()


@dataclass(frozen=True, slots=True)
class _ProducerOutcomeAuthority:
    """Ephemeral owner-created proof, never serialized or accepted by public mutation."""

    expected: bytes
    expires_at_ms: int
    seal: object

    def require(self, command, *, now_ms, redactor):
        if (
            self.seal is not _SEAL
            or self.expected != contract_bytes(command, redactor=redactor)
            or type(self.expires_at_ms) is not int
            or now_ms >= self.expires_at_ms
        ):
            raise CollaborationConflict("Producer publication authority is unavailable.")


def _publication_authority(command, *, expires_at_ms, redactor):
    return _ProducerOutcomeAuthority(
        contract_bytes(command, redactor=redactor), expires_at_ms, _SEAL
    )


def outcome_operation(registration, redactor):
    return registration.operation.model_copy(
        update={
            "caller_key": "producer-outcome:"
            + sha256(contract_bytes(registration.operation, redactor=redactor)).hexdigest()
        }
    )


async def require_producer_outcome(tx, prior, command, *, redactor):
    """Complete durable tuple validation; fresh publication additionally needs the seal."""
    command = prepare_contract(ProducerOutcomeCommand, command, redactor=redactor)
    raw = await tx.get("operations", operation_key(command.producer))
    registration = prepare_contract(ProducerOutputRecord, raw, redactor=redactor)
    retained = await read_output_registration(tx, registration.command, redactor=redactor)
    if retained is None:
        raise CollaborationUnavailable("Producer outcome registration is unavailable.")
    expected = registration.command
    require_exact_contract(expected.admission.expected, command.expected, redactor=redactor)
    if (
        prior.producer_operation != command.producer
        or outcome_operation(expected, redactor) != command.operation
        or registration.command.operation != command.producer
        or registration.state != "launch_claimed"
        or registration.completion != command.completion
        or command.publisher_generation != expected.publisher_generation
        or prior.admission_operation != expected.admission.operation
        or prior.admission_generation != command.publisher_generation
        or command.terminal_frontier != len(prior.progress)
    ):
        raise CollaborationConflict("Producer outcome does not match its admitted responsibility.")
    input_commitment = (
        prior.clarification.input_sha256
        or sha256(contract_bytes(prior.receipt.expected, redactor=redactor)).hexdigest()
    )
    if (expected.admission.expected_input_revision, expected.admission.expected_input_sha256) != (
        prior.clarification.input_revision,
        input_commitment,
    ):
        raise CollaborationConflict("Producer outcome effective input changed.")
    completion = prepare_contract(
        ProducerCompletionRecord,
        await tx.get("operations", operation_key(command.completion)),
        redactor=redactor,
    )
    require_exact_contract(expected, completion.output.registration, redactor=redactor)
    if command.native_commitment != completion.native_commitment:
        raise CollaborationConflict("Producer native completion conflicts.")
    if command.outcome == "answered":
        destination = next(
            (item for item in expected.destinations if item.operation == command.destination), None
        )
        if destination is None or completion.output.disposition != "answer":
            raise CollaborationConflict("Producer answer destination is not registered.")
        exported = await read_export(tx, expected, destination, redactor=redactor)
        if (
            exported is None
            or exported.state != "published"
            or exported.operation != command.export
            or exported.completion != command.completion
            or exported.output_commitment != command.commitment
            or exported.initiator != command.initiator
        ):
            raise CollaborationConflict("Producer answer lacks its exact export acceptance.")
    elif command.output_failure is not None:
        from cayu.collaboration._producer_output_failure import read_output_failure

        failure = None
        for destination in expected.destinations:
            exported = await read_export(tx, expected, destination, redactor=redactor)
            if exported is not None and exported.rejection == command.output_failure:
                failure = await read_output_failure(tx, exported, redactor=redactor)
                break
        if (
            failure is None
            or completion.output.disposition != "answer"
            or command.commitment != failure.native_commitment
            or command.initiator != expected.initiator
            or "sha256:" + failure.failure.source_commitment != completion.output.source_commitment
        ):
            raise CollaborationConflict("Producer contract failure lacks exact native evidence.")
    elif (
        completion.output.disposition == "answer"
        or command.commitment != completion.native_commitment
        or command.initiator != expected.initiator
    ):
        raise CollaborationConflict("Producer failure lacks native failure evidence.")
