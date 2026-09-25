"""Dependency-light identities; parsing never authenticates native material."""

from typing import Annotated

from pydantic import Field, StrictStr, model_validator

from cayu.collaboration._contracts import ContractValue, OperationRef, OwnerRef
from cayu.collaboration.participants import VersionOne

MaterialCommitment = Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class ResourceMaterialReference(ContractValue):
    """Complete expected content; authority comes from the registered native owner."""

    schema_version: VersionOne = 1
    owner: OwnerRef
    operation: OperationRef
    template_commitment: MaterialCommitment
    transfer_commitment: MaterialCommitment
    preparation_commitment: MaterialCommitment

    @model_validator(mode="after")
    def coherent(self):
        if self.operation.application_scope != self.owner.application_scope:
            raise ValueError("Resource material reference belongs to another scope.")
        return self
