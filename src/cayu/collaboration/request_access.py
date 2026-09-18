"""Trusted registration for request-owner operations, not agent execution."""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass

from cayu.collaboration._contracts import ContractValue, ObjectRef
from cayu.collaboration._permits import PermitCommand, ReceivingSettlementReceipt
from cayu.collaboration.mandates import MandateAccessContext, MandateResolver, ResourceSelectorOwner
from cayu.collaboration.requests import (
    Millis,
    RequestAdmissionCommand,
    RequestControlCommand,
    RequestOutcomeCommand,
    RequestProgressCommand,
)

RequestReceivingCommand = (
    RequestAdmissionCommand | RequestProgressCommand | RequestOutcomeCommand | RequestControlCommand
)


class RequestReceivingAuthorization(ContractValue):
    """Historical evidence; only a registered owner's live guard authenticates it."""

    receiver: ObjectRef
    command: RequestReceivingCommand
    expires_at_ms: Millis
    settlement: ReceivingSettlementReceipt | None = None


class RequestReceivingOwner(ABC):
    """Trusted, non-dispatching boundary for producer evidence authentication.

    Implementations must authenticate the complete command and caller, pinned
    policy, producer and required source evidence; a matching caller value or
    digest alone is insufficient. Acquire must serialize revocation with the
    yielded authorization until publication settles. It runs outside the store
    transaction and must never launch work. Missing adapters must refuse.
    """

    @property
    @abstractmethod
    def ref(self) -> ObjectRef: ...

    @abstractmethod
    def acquire(
        self, command: RequestReceivingCommand, *, context: MandateAccessContext
    ) -> AbstractAsyncContextManager[RequestReceivingAuthorization]: ...

    async def settlement(
        self,
        command: RequestReceivingCommand,
        expected: PermitCommand,
        *,
        context: MandateAccessContext,
    ) -> ReceivingSettlementReceipt | None:
        """Return receiver-authenticated terminal evidence, if qualified."""
        return None


@dataclass(frozen=True)
class RequestRegistration:
    mandates: MandateResolver
    max_ttl_ms: int
    resource_owners: tuple[ResourceSelectorOwner, ...] = ()
    receiving_owner: RequestReceivingOwner | None = None
