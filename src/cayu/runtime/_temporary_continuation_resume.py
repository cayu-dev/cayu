"""Temporary clarification handoff through the existing runtime resume gates."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from cayu.runtime._invocation_lifecycle import (
    AdmitInvocationCommand,
    AdmittedInvocationBinding,
    InvocationContext,
    InvocationMutationResult,
)
from cayu.runtime._session_continuation import continuation_digest
from cayu.runtime._temporary_continuation import TemporaryServiceIntent

if TYPE_CHECKING:
    from cayu.runtime._session_continuation_owner import SessionContinuationOwner


@dataclass(frozen=True)
class _TemporaryContinuationResumeHandoff:
    """Runtime-constructed only, never accepted from an ordinary resume request.

    The coordinator authenticates the intent, including its durable preparation
    timestamp. All profile, context, budget and participant gates remain in the
    normal resume machinery before this handoff receives its prepared invocation.
    """

    owner: SessionContinuationOwner
    intent: TemporaryServiceIntent
    delivery: Callable[[InvocationContext], Awaitable[None]] | None = None

    async def after_admission(self, invocation: InvocationContext) -> None:
        invocation.require_runtime_authority()
        binding = invocation.binding
        if (
            type(binding) is not AdmittedInvocationBinding
            or binding.session_id != self.intent.target.object_id
            or binding.session_instance_id != self.intent.target.incarnation
            or binding.interaction_id != self.intent.invocation_id
        ):
            raise PermissionError("Temporary delivery requires its admitted invocation owner.")
        if self.delivery is not None:
            await self.delivery(invocation)

    @property
    def interaction_started_at(self) -> datetime:
        return self.intent.prepared_at

    def _identity(self, kind: str) -> str:
        return str(
            uuid5(
                NAMESPACE_URL,
                "cayu:temporary-continuation:"
                + kind
                + ":"
                + continuation_digest(self.intent.operation),
            )
        )

    @property
    def interaction_id(self) -> str:
        return self.intent.invocation_id

    @property
    def interaction_started_event_id(self) -> str:
        return self._identity("interaction-start")

    @property
    def run_operation_id(self) -> str:
        return self._identity("run")

    @property
    def model_transition_event_id(self) -> str:
        return self._identity("model-transition")

    async def admit(
        self, command: AdmitInvocationCommand, invocation: InvocationContext
    ) -> InvocationMutationResult:
        result, _ = await self.owner.prepare_temporary_admission(
            self.intent, command, invocation=invocation
        )
        return result
