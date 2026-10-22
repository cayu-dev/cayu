"""Child sandboxes brokered by Cayu Cloud, with no cloud-provider credential in the Agent.

The parts compose but each works alone:

- ``SandboxBrokerClient`` speaks the ``cayu.cloud.sandbox`` v1 wire contract.
- ``SandboxLeaseKeeper`` renews one sandbox's lease.
- ``BrokeredSandboxRunner`` is a ``Runner`` over one sandbox.
- ``BrokeredSandboxWorkspace`` is a ``RunnerWorkspace`` whose bulk copies use the broker's
  archives, and its staged copies for trees and files larger than one archive.
- ``BrokeredSandboxCodingEnvironmentFactory`` assembles them into per-session coding sandboxes.

See "Coding in a sandbox brokered by Cayu Cloud" in ``docs/environment-factories.md``.
"""

from cayu.sandbox_broker.client import (
    BROKER_API_URL_ENV,
    BROKER_APPLICATION_ID_ENV,
    BROKER_REQUEST_TIMEOUT_ENV,
    BROKER_TOKEN_ENV,
    SandboxAllocationFailedError,
    SandboxArchive,
    SandboxArchiveTooLargeError,
    SandboxBrokerClient,
    SandboxBrokerError,
    SandboxBrokerRefusedError,
    SandboxBrokerSettings,
    SandboxBrokerUnavailableError,
    SandboxBusyError,
    SandboxCommand,
    SandboxCopy,
    SandboxCopyFailedError,
    SandboxEgressMode,
    SandboxFailure,
    SandboxGuarantees,
    SandboxImage,
    SandboxLeaseLostError,
    SandboxNotFoundError,
    SandboxNotReadyError,
    SandboxQuotaExceededError,
    SandboxRecord,
    SandboxSignedTransfer,
    SandboxStagedCopies,
)
from cayu.sandbox_broker.coding import (
    BROKERED_SANDBOX_EGRESS_ENV,
    BROKERED_SANDBOX_IMAGE_ENV,
    BROKERED_SANDBOX_LEASE_ENV,
    BrokeredSandboxCodingBinding,
    BrokeredSandboxCodingEnvironmentFactory,
)
from cayu.sandbox_broker.lease import SandboxLeaseKeeper
from cayu.sandbox_broker.runner import (
    BrokeredSandboxRunner,
    BrokeredSandboxWorkspace,
    sandbox_capability_claims,
)

__all__ = [
    "BROKERED_SANDBOX_EGRESS_ENV",
    "BROKERED_SANDBOX_IMAGE_ENV",
    "BROKERED_SANDBOX_LEASE_ENV",
    "BROKER_API_URL_ENV",
    "BROKER_APPLICATION_ID_ENV",
    "BROKER_REQUEST_TIMEOUT_ENV",
    "BROKER_TOKEN_ENV",
    "BrokeredSandboxCodingBinding",
    "BrokeredSandboxCodingEnvironmentFactory",
    "BrokeredSandboxRunner",
    "BrokeredSandboxWorkspace",
    "SandboxAllocationFailedError",
    "SandboxArchive",
    "SandboxArchiveTooLargeError",
    "SandboxBrokerClient",
    "SandboxBrokerError",
    "SandboxBrokerRefusedError",
    "SandboxBrokerSettings",
    "SandboxBrokerUnavailableError",
    "SandboxBusyError",
    "SandboxCommand",
    "SandboxCopy",
    "SandboxCopyFailedError",
    "SandboxEgressMode",
    "SandboxFailure",
    "SandboxGuarantees",
    "SandboxImage",
    "SandboxLeaseKeeper",
    "SandboxLeaseLostError",
    "SandboxNotFoundError",
    "SandboxNotReadyError",
    "SandboxQuotaExceededError",
    "SandboxRecord",
    "SandboxSignedTransfer",
    "SandboxStagedCopies",
    "sandbox_capability_claims",
]
