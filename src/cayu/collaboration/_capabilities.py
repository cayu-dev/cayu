"""Explicit operation-family contracts, not a global adapter registry."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from cayu.collaboration._contracts import (
    MAX_ENTRIES,
    Code,
    ContractValue,
    Generation,
    OwnerRef,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.vaults.redaction import SecretRedactor


class FamilyVersion(ContractValue):
    family: Code
    version: Generation


class CapabilityDescriptor(ContractValue):
    """A family version includes its whole contract, not optional safety bits.

    Implementations/wrappers provide this through trusted registration, not
    request JSON. Qualification of their actual behavior remains mandatory.
    """

    owner: OwnerRef
    mutations: tuple[FamilyVersion, ...] = Field(max_length=MAX_ENTRIES)
    readbacks: tuple[FamilyVersion, ...] = Field(max_length=MAX_ENTRIES)

    @model_validator(mode="after")
    def unique_families(self) -> CapabilityDescriptor:
        for name in ("mutations", "readbacks"):
            families = getattr(self, name)
            keys = tuple((item.family, item.version) for item in families)
            if len(keys) != len(set(keys)):
                raise ValueError("Capability families must not repeat.")
            object.__setattr__(
                self, name, tuple(sorted(families, key=lambda x: (x.family, x.version)))
            )
        if not set(self.mutations) <= set(self.readbacks):
            raise ValueError("Mutations require their exact readback contract.")
        return self


class CollaborationCapabilityUnavailable(ValueError):
    """The registered owner does not advertise the required complete contract."""


def require_capability(
    descriptor: CapabilityDescriptor,
    *,
    expected_owner: OwnerRef,
    required: FamilyVersion,
    supported: tuple[FamilyVersion, ...],
    access: Literal["mutation", "readback"],
    redactor: SecretRedactor,
) -> None:
    """Validate a trusted registration against an explicit receiving contract.

    The receiving owner supplies supported families; an arbitrary v1 declaration
    cannot enable an unknown family. This is not a substitute for conformance.
    """
    descriptor = prepare_contract(CapabilityDescriptor, descriptor, redactor=redactor)
    expected_owner = prepare_contract(OwnerRef, expected_owner, redactor=redactor)
    required = prepare_contract(FamilyVersion, required, redactor=redactor)
    if type(supported) is not tuple or len(supported) > MAX_ENTRIES:
        raise CollaborationCapabilityUnavailable("Invalid supported family contract.")
    checked = tuple(prepare_contract(FamilyVersion, item, redactor=redactor) for item in supported)
    if type(access) is not str or access not in ("mutation", "readback"):
        raise CollaborationCapabilityUnavailable("Unknown capability access mode.")
    advertised = descriptor.mutations if access == "mutation" else descriptor.readbacks
    if descriptor.owner != expected_owner or required not in checked or required not in advertised:
        raise CollaborationCapabilityUnavailable("Required owner capability is unavailable.")
