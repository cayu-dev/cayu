"""Source-retirement authority and native cleanup's monotonic late-write fence."""

from dataclasses import dataclass
from hashlib import sha256

from pydantic import Field, StrictBool, StrictInt, model_validator

from cayu.collaboration._contracts import ContractValue, ObjectRef
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.lifecycle import NamespaceRef
from cayu.vaults.redaction import SecretRedactor

_SEAL = object()


class ProducerCleanupRetirement(ContractValue):
    receiver: ObjectRef
    namespace: NamespaceRef

    @model_validator(mode="after")
    def coherent(self):
        if self.receiver.owner != self.namespace.owner or self.receiver.revision is None:
            raise ValueError("Producer retirement belongs to another registered receiver.")
        return self


class ProducerCleanupReclamation(ContractValue):
    retirement: ProducerCleanupRetirement
    removed: StrictInt = Field(ge=0, le=32)
    remaining: StrictBool


@dataclass(frozen=True)
class _RetirementAuthority:
    expected: bytes
    seal: object

    def require(self, expected):
        if self.seal is not _SEAL or self.expected != contract_bytes(
            expected, redactor=SecretRedactor()
        ):
            raise PermissionError("Native reclamation requires authenticated source retirement.")


def _accepted_source_retirement(retirement):
    """Registered source owner calls only after reading its pruned-through frontier."""
    retirement = prepare_contract(ProducerCleanupRetirement, retirement, redactor=SecretRedactor())
    return _RetirementAuthority(contract_bytes(retirement, redactor=SecretRedactor()), _SEAL)


def retirement_for(command):
    return ProducerCleanupRetirement(
        receiver=command.receiver,
        namespace=NamespaceRef(
            owner=command.receiver.owner,
            namespace_incarnation=command.operation.namespace_incarnation,
            generation=command.operation.generation,
        ),
    )


def retirement_key(retirement):
    # Source retirement covers the complete owner namespace, including prior
    # receiving configurations on this physical native store. A new receiver
    # revision must neither reopen that namespace nor strand its old receipts.
    canonical = retirement.namespace.model_copy(update={"generation": 1})
    return sha256(contract_bytes(canonical, redactor=SecretRedactor())).hexdigest()


def require_unretired(retirement, through):
    if through is not None:
        if type(through) is not int or not 1 <= through < 2**53:
            raise ValueError("Native producer retirement evidence is invalid.")
        if retirement.namespace.generation <= through:
            raise ValueError("Native producer cleanup history has been retired.")


def prepare_retirement(retirement, authority, limit):
    retirement = prepare_contract(ProducerCleanupRetirement, retirement, redactor=SecretRedactor())
    if type(authority) is not _RetirementAuthority:
        raise PermissionError("Native reclamation requires its registered source owner.")
    authority.require(retirement)
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError("Native reclamation requires a bounded batch.")
    return retirement, retirement_key(retirement)


def validate_retiring_receipt(retirement, operation_key, raw):
    from cayu.runtime._producer_cleanup_receipt import NativeProducerCleanupReceipt
    from cayu.runtime._producer_output_store import OPERATION_PREFIX

    receipt = prepare_contract(NativeProducerCleanupReceipt, raw, redactor=SecretRedactor())
    if (
        operation_key
        != OPERATION_PREFIX
        + sha256(contract_bytes(receipt.registration, redactor=SecretRedactor())).hexdigest()
        or receipt.receiver.owner != retirement.namespace.owner
        or receipt.registration.application_scope != retirement.namespace.owner.application_scope
        or receipt.registration.namespace_incarnation != retirement.namespace.namespace_incarnation
        or receipt.registration.generation > retirement.namespace.generation
    ):
        raise ValueError("Native reclamation receipt conflicts with its namespace index.")
    return receipt
