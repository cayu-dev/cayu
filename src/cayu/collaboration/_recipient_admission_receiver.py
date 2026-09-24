"""Registered, non-dispatching authentication of native FRESH recipient evidence."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from cayu.budgets.binding import BudgetBinding
from cayu.collaboration._contracts import ExactMatch, ObjectRef
from cayu.collaboration._mandate_validation import MandateUse, validate_mandate_resolution
from cayu.collaboration._permits import ReceivingSettlementReceipt
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.mandates import MandateResolution, MandateResolver
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.prepared_admission import (
    prepared_budget,
    prepared_budget_request,
    require_secret_free_prepared,
)
from cayu.collaboration.request_access import RequestReceivingAuthorization, RequestReceivingOwner
from cayu.collaboration.requests import (
    RequestAdmissionCommand,
    RequestAdmissionReceipt,
    RequestCommand,
    RequestControlCommand,
    RequestSnapshot,
)
from cayu.sessions.base import SessionStatus, SessionStore
from cayu.sessions.context_views import ParticipantSessionCreationReceipt, json_commitment
from cayu.sessions.creation_fence import SessionCreationDecision, validate_binding
from cayu.vaults.redaction import SecretRedactor


class RecipientAdmissionReceivingOwner(RequestReceivingOwner):
    """Application-wired native readers; constructor arguments are trusted registration.

    No caller-provided receipt or callback is installed through an admission
    command. The native creation decision and stored receipt authenticate data;
    the held mandate and atomic lifecycle permit authenticate new admission.
    """

    prepared_admission_version = 1

    def __init__(
        self,
        *,
        ref: ObjectRef,
        sessions: SessionStore,
        mandates: MandateResolver,
        resolve_budget: Callable[..., Awaitable[BudgetBinding]] | None,
        now_ms: Callable[[], Awaitable[int]],
        read_admission: Callable[
            [RequestCommand], Awaitable[tuple[RequestSnapshot, RequestAdmissionReceipt]]
        ],
        delegate: RequestReceivingOwner | None,
        redactor: SecretRedactor,
    ):
        self._ref = prepare_contract(ObjectRef, ref, redactor=redactor)
        if self._ref.revision is None:
            raise ValueError("Prepared admission receiver needs a pinned revision.")
        self._sessions = sessions
        self._mandates = mandates
        self._resolver_ref = prepare_contract(ObjectRef, mandates.ref, redactor=redactor)
        self._resolve_budget = resolve_budget
        self._now_ms = now_ms
        self._read_admission = read_admission
        self._delegate = delegate
        self._delegate_ref = (
            None
            if delegate is None
            else prepare_contract(ObjectRef, delegate.ref, redactor=redactor)
        )
        self._redactor = redactor

    @property
    def ref(self) -> ObjectRef:
        return self._ref.model_copy(deep=True)

    async def _native_evidence(self, command: RequestAdmissionCommand) -> None:
        prepared = command.prepared
        assert prepared is not None
        if self._resolve_budget is None:
            raise CollaborationUnavailable("New prepared admission requires its budget receiver.")
        target = prepared.target
        if (
            type(self._sessions.participant_session_binding_version) is not int
            or self._sessions.participant_session_binding_version != 1
        ):
            raise CollaborationUnavailable("Native recipient admission is not qualified.")
        found = await self._sessions.read_session_creation_decision(target.creation)
        if not isinstance(found, ExactMatch):
            raise CollaborationUnavailable("Exact recipient creation is unavailable.")
        decision = prepare_contract(SessionCreationDecision, found.receipt, redactor=self._redactor)
        require_exact_contract(target.creation, decision.target, redactor=self._redactor)
        if (
            decision.state != "created"
            or decision.session_id != target.session_id
            or decision.session_instance_id != target.session_instance_id
            or decision.creation_receipt_commitment != target.creation_receipt_commitment
        ):
            raise CollaborationAccessDenied("Recipient creation evidence conflicts.")
        raw = await self._sessions.load_participant_session_creation_receipt(target.session_id)
        receipt = prepare_contract(ParticipantSessionCreationReceipt, raw, redactor=self._redactor)
        validate_binding(
            target.creation, receipt.binding, requested_session_id=receipt.requested_session_id
        )
        if (
            receipt.receipt_commitment != decision.creation_receipt_commitment
            or receipt.binding.session_id != target.session_id
            or receipt.binding.session_instance_id != target.session_instance_id
            or receipt.initial_input_commitment != target.initial_input_commitment
            or receipt.execution_profile_json != prepared.execution_profile_json
            or receipt.recipient_metadata_json is None
        ):
            raise CollaborationAccessDenied("Stored recipient evidence conflicts.")
        metadata = json.loads(receipt.recipient_metadata_json)
        definition = json.loads(receipt.binding.historical_definition_json)
        if (
            set(metadata)
            != {
                "mode",
                "original_request_commitment",
                "recipient",
                "selected_view",
                "resource_transfers",
                "preparation_receipts",
            }
            or metadata.get("mode") != "fresh"
            or metadata.get("selected_view") is not None
            or metadata.get("recipient") != prepared.recipient.model_dump(mode="json")
            or metadata.get("resource_transfers") != []
            or metadata.get("preparation_receipts") != []
            or definition.get("agent_definition_commitment") != target.definition_commitment
        ):
            raise CollaborationUnavailable("Recipient preparation mode is not qualified.")
        if receipt.binding.execution_profile_commitment != json_commitment(
            prepared.execution_profile_json, "execution_profile"
        ):
            raise CollaborationAccessDenied("Recipient profile commitment conflicts.")
        session = await self._sessions.load(target.session_id)
        if session is None or session.instance_id != target.session_instance_id:
            raise CollaborationUnavailable("Recipient session incarnation is unavailable.")
        if session.status != SessionStatus.PENDING or session.run_epoch != 0:
            raise CollaborationUnavailable("Recipient is no longer an inert preparation.")
        binding = await self._resolve_budget(
            request=prepared_budget_request(
                session_id=session.id,
                session_instance_id=session.instance_id,
                profile=prepared.execution_profile_json,
            )
        )
        expected_binding = prepared_budget(prepared.budget_binding_json)
        if type(binding) is not BudgetBinding or binding != expected_binding:
            raise CollaborationAccessDenied("Prepared sponsor binding conflicts.")
        for expected, actual in (
            (binding.provider_name, session.provider_name),
            (binding.model, session.model),
            (binding.environment_name, session.environment_name),
        ):
            if expected is not None and expected != actual:
                raise CollaborationAccessDenied("Prepared budget target conflicts.")

    @asynccontextmanager
    async def acquire(self, command, *, context):
        if isinstance(command, RequestAdmissionCommand) and command.prepared is not None:
            require_exact_contract(self._ref, command.prepared.receiver, redactor=self._redactor)
            if context.participant != command.prepared.recipient:
                raise CollaborationAccessDenied("Prepared admission requires its recipient.")
            require_exact_contract(self._resolver_ref, self._mandates.ref, redactor=self._redactor)
            async with self._mandates.acquire(context) as raw:
                resolution = prepare_contract(MandateResolution, raw, redactor=self._redactor)
                validate_mandate_resolution(
                    resolution,
                    context=context,
                    resolver=self._resolver_ref,
                    use=MandateUse(
                        audience=self._ref.owner,
                        scope=self._ref.owner.application_scope,
                        actions=("readback", "prepare"),
                        resources=(),
                        inputs=(),
                    ),
                    now_ms=await self._now_ms(),
                    resource_owners={},
                    redactor=self._redactor,
                )
                # JSON strings must also be checked after decoding; escaped secret
                # values cannot bypass the outer contract's raw string check.
                require_secret_free_prepared(command.prepared, self._redactor)
                await self._native_evidence(command)
                yield RequestReceivingAuthorization(
                    receiver=self.ref,
                    command=command,
                    expires_at_ms=min(
                        resolution.principal.expires_at_ms,
                        *(entry.expires_at_ms for entry in resolution.chain.entries),
                    ),
                )
            return
        if isinstance(command, RequestControlCommand):
            prior, admitted = await self._read_admission(command.intent.expected)
            if admitted.command.prepared is not None:
                # The request coordinator already holds the administrative mandate
                # guard. Reacquiring it could deadlock a non-reentrant resolver.
                if command.intent.source_receipt is not None:
                    raise CollaborationAccessDenied("Inert admission has no producer export.")
                yield RequestReceivingAuthorization(
                    receiver=self.ref,
                    command=command,
                    expires_at_ms=2**53 - 1,
                    settlement=ReceivingSettlementReceipt(
                        expected=prior.permit,
                        receiving_owner=self._ref.owner,
                        receipt_id="inert-admission:" + admitted.event.id,
                        outcome="quiescent",
                    ),
                )
                return
        if self._delegate is None:
            raise CollaborationUnavailable("This receiving operation is not qualified.")
        assert self._delegate_ref is not None
        require_exact_contract(self._delegate_ref, self._delegate.ref, redactor=self._redactor)
        async with self._delegate.acquire(command, context=context) as raw:
            checked = prepare_contract(RequestReceivingAuthorization, raw, redactor=self._redactor)
            require_exact_contract(command, checked.command, redactor=self._redactor)
            require_exact_contract(self._delegate_ref, checked.receiver, redactor=self._redactor)
            yield checked.model_copy(update={"receiver": self.ref})

    async def settlement(self, command, expected, *, context):
        if self._delegate is None:
            return None
        return await self._delegate.settlement(command, expected, context=context)
