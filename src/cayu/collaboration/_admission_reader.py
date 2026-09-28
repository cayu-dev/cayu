"""Production reader backed by the application's registered request owner."""

from typing import TYPE_CHECKING

from cayu.collaboration._contracts import ExactLookup
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.request_access import RequestAdmissionReader
from cayu.collaboration.requests import RequestAdmissionCommand, RequestAdmissionReceipt

if TYPE_CHECKING:
    from cayu.collaboration._request_coordinator import RequestCoordinator


class RegisteredRequestAdmissionReader(RequestAdmissionReader):
    def __init__(self, owner: "RequestCoordinator") -> None:
        self._owner = owner

    async def lookup(
        self, expected: RequestAdmissionCommand, *, context: MandateAccessContext
    ) -> ExactLookup[RequestAdmissionReceipt]:
        return await self._owner.lookup_admission(expected, context=context)

    async def _lookup_owned(
        self, expected: RequestAdmissionCommand, *, context: MandateAccessContext
    ) -> ExactLookup[RequestAdmissionReceipt]:
        """Native runtime observation under an enclosing retained owner."""
        return await self._owner.lookup_admission(
            expected, context=context, wait_for_settlement=True
        )
