"""Frozen resource handoff proposals; registered native owners retain authority."""

from pydantic import model_validator

from cayu.artifacts.resources import (
    ResourceAcquisitionCommand,
    ResourceTransferTemplate,
    _validate_preparation_permit,
)
from cayu.collaboration._contracts import ContractValue
from cayu.collaboration._permits import PermitCommand
from cayu.collaboration.participants import VersionOne


class RequestPlanningResource(ContractValue):
    """Exact acquisition plus destination intent before any pin is acquired.

    The transfer's actual source receipt is resolved by its registered owner,
    never guessed from this data. Cleanup uses the same native operation keys.
    """

    schema_version: VersionOne = 1
    acquisition_permit: PermitCommand
    transfer: ResourceTransferTemplate
    transfer_permit: PermitCommand

    @property
    def acquisition(self) -> ResourceAcquisitionCommand:
        """The transfer template owns the one complete frozen acquisition."""
        return self.transfer.acquisition

    @model_validator(mode="after")
    def coherent(self):
        if self.acquisition.intent.deadline_at_ms is None:
            raise ValueError("Planned resource preparation requires an exact finite deadline.")
        _validate_preparation_permit(
            self.acquisition.source, self.acquisition, self.acquisition_permit
        )
        _validate_preparation_permit(self.transfer.destination, self.transfer, self.transfer_permit)
        receiving = self.transfer_permit.intent.request
        if receiving.expected_configuration_revision is None:
            raise ValueError("Resource destination requires an exact participant configuration.")
        identities = (
            self.acquisition.operation,
            self.acquisition_permit.operation,
            self.acquisition_permit.intent.request.settlement_operation,
            self.transfer.operation,
            self.transfer_permit.operation,
            receiving.settlement_operation,
        )
        if len(set(identities)) != len(identities):
            raise ValueError("Resource handoff operation identities overlap.")
        return self
