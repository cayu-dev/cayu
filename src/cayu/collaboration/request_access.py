"""Trusted registration for request-owner operations, not agent execution."""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import ClassVar

from cayu.collaboration._contracts import ContractValue, ExactLookup, ObjectRef
from cayu.collaboration._permits import PermitCommand, ReceivingSettlementReceipt
from cayu.collaboration.clarifications import ClarificationPolicy
from cayu.collaboration.mandates import MandateAccessContext, MandateResolver, ResourceSelectorOwner
from cayu.collaboration.planning import ConfiguredRequestPlanningPolicy
from cayu.collaboration.requests import (
    Millis,
    RequestAdmissionCommand,
    RequestAdmissionReceipt,
    RequestControlCommand,
    RequestOutcomeCommand,
    RequestProgressCommand,
)

RequestReceivingCommand = (
    RequestAdmissionCommand | RequestProgressCommand | RequestOutcomeCommand | RequestControlCommand
)


class RequestAdmissionReader(ABC):
    """Registered exact historical readback, not a recipient launch capability.

    Implementations must authenticate the current read context on each lookup
    and compare the complete expected command, including effective input and
    native target. Receipts must come from the durable receiving owner.
    """

    @abstractmethod
    async def lookup(
        self, expected: RequestAdmissionCommand, *, context: MandateAccessContext
    ) -> ExactLookup[RequestAdmissionReceipt]: ...


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

    # Explicit qualification, not inherited from the existing export contract.
    # 1: inert FRESH; 2: FRESH and exact released whole-turn CONTINUE selection.
    # 3: those families plus inert FORK with native historical-selection evidence.
    # 4: also authenticate exact adopted resource material through admission.
    prepared_admission_version: ClassVar[int] = 0

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
class PreparedAdmissionRegistration:
    """Opt in to the application-wired native FRESH receiving owner.

    Native sessions and the frozen runtime budget receiver are wired by CayuApp;
    an admission command cannot install callbacks or choose another store.
    """

    receiver: ObjectRef


@dataclass(frozen=True)
class RequestPlanningAdmissionReader:
    """Pin an existing authenticated admission reader for planning prerequisites."""

    reference: ObjectRef
    reader: RequestAdmissionReader


@dataclass(frozen=True)
class RequestRegistration:
    mandates: MandateResolver
    max_ttl_ms: int
    resource_owners: tuple[ResourceSelectorOwner, ...] = ()
    receiving_owner: RequestReceivingOwner | None = None
    clarification_policies: tuple[ClarificationPolicy, ...] = ()
    prepared_admission: PreparedAdmissionRegistration | None = None
    planning_policies: tuple[ConfiguredRequestPlanningPolicy, ...] = ()
    planning_readers: tuple[RequestPlanningAdmissionReader, ...] = ()
