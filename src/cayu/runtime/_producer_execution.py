"""Explicit registered producer handoff into ordinary participant execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import ProducerOutputRegistration
from cayu.collaboration._producer_registration import producer_launch_guard
from cayu.collaboration._request_coordinator import _safe_request_failure
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.prepared_admission import FreshRecipientAdmissionTarget
from cayu.sessions._producer_checkpoint import (
    ROOT_KEY,
    NativeProducerAttachment,
    NativeProducerIndex,
    attachment_index,
)
from cayu.sessions.base import _invocation_lifecycle_authority_read_scope
from cayu.sessions.checkpoints import decode_runtime_checkpoint

if TYPE_CHECKING:
    from cayu.applications import CayuApp


@dataclass(frozen=True, slots=True, repr=False)
class _ProducerExecution:
    """Private fixed owner, not a public receipt or a callback supplied in RunRequest."""

    app: CayuApp
    command: ProducerOutputRegistration
    context: MandateAccessContext

    def __post_init__(self):
        for field, schema in (
            ("command", ProducerOutputRegistration),
            ("context", MandateAccessContext),
        ):
            object.__setattr__(
                self,
                field,
                prepare_contract(schema, getattr(self, field), redactor=self.app._secret_redactor),
            )

    async def checkpoint(self, store, session_id):
        prepared = self.command.admission.prepared
        assert prepared is not None and isinstance(prepared.target, FreshRecipientAdmissionTarget)
        if store is not self.app._runtime_session_store or session_id != prepared.target.session_id:
            raise PermissionError("Producer handoff targets another native owner.")
        store = self.app.session_store
        with _invocation_lifecycle_authority_read_scope():
            checkpoint = await store.load_checkpoint(session_id)
        if checkpoint is None or set(checkpoint) != {ROOT_KEY}:
            raise PermissionError("Producer launch requires only its prepared checkpoint.")
        redactor = self.app._secret_redactor
        index = prepare_contract(NativeProducerIndex, checkpoint[ROOT_KEY], redactor=redactor)
        raw = await store.load_session_operation(session_id, index.operation_key)
        attachment = prepare_contract(NativeProducerAttachment, raw, redactor=redactor)
        require_exact_contract(self.command, attachment.command, redactor=redactor)
        require_exact_contract(attachment_index(attachment), index, redactor=redactor)
        return decode_runtime_checkpoint(checkpoint, session_id=session_id)

    async def admit(self, command):
        coordinator = self.app._request_coordinator
        redactor = self.app._secret_redactor

        async def mutate():
            async with producer_launch_guard(
                self.app, self.command, context=self.context
            ) as record:
                return await self.app._runtime_session_store._admit_native_producer(record, command)

        async def owned():
            return await coordinator._dependency(mutate)

        return await coordinator._observe(
            coordinator._owners.run(
                owned,
                key=("producer_native_admission", object()),
                expectation=contract_bytes(self.command, redactor=redactor),
                redactor=redactor,
                failure_snapshot=lambda error: _safe_request_failure(error, redactor),
            )
        )
