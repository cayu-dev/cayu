"""Registered, non-dispatching authentication of native recipient evidence."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager

from cayu.budgets.binding import BudgetBinding
from cayu.collaboration._contracts import ExactMatch, ObjectRef, OwnerRef
from cayu.collaboration._mandate_validation import MandateUse, validate_mandate_resolution
from cayu.collaboration._permits import ReceivingSettlementReceipt
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.mandates import MandateResolution, MandateResolver, ResourceSelectorOwner
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.prepared_admission import (
    ContinueRecipientAdmissionTarget,
    ForkRecipientAdmissionTarget,
    FreshRecipientAdmissionTarget,
    PreparedRecipientAdmission,
    created_admission_target,
    prepared_budget,
    prepared_budget_request,
    require_prepared_budget_target,
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
from cayu.sessions.base import Session, SessionStatus, SessionStore
from cayu.sessions.context_views import ParticipantSessionCreationReceipt, json_commitment
from cayu.sessions.creation_fence import SessionCreationDecision, validate_binding
from cayu.vaults.redaction import SecretRedactor


class RecipientAdmissionReceivingOwner(RequestReceivingOwner):
    """Application-wired native readers; constructor arguments are trusted registration.

    No caller-provided receipt or callback is installed through an admission
    command. The native creation decision and stored receipt authenticate data;
    the held mandate and atomic lifecycle permit authenticate new admission.
    """

    prepared_admission_version = 4

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
        resource_owners: Mapping[OwnerRef, ResourceSelectorOwner],
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
        self._resource_owners = dict(resource_owners)

    @property
    def ref(self) -> ObjectRef:
        return self._ref.model_copy(deep=True)

    async def _created_session(self, prepared: PreparedRecipientAdmission) -> Session:
        target = prepared.target
        assert isinstance(target, (FreshRecipientAdmissionTarget, ForkRecipientAdmissionTarget))
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
            or metadata.get("mode") != target.kind
            or metadata.get("recipient") != prepared.recipient.model_dump(mode="json")
            or definition.get("agent_definition_commitment") != target.definition_commitment
        ):
            raise CollaborationUnavailable("Recipient preparation mode is not qualified.")
        require_exact_contract(
            target, created_admission_target(target.creation, receipt), redactor=self._redactor
        )
        if receipt.binding.execution_profile_commitment != json_commitment(
            prepared.execution_profile_json, "execution_profile"
        ):
            raise CollaborationAccessDenied("Recipient profile commitment conflicts.")
        session = await self._sessions.load(target.session_id)
        if session is None or session.instance_id != target.session_instance_id:
            raise CollaborationUnavailable("Recipient session incarnation is unavailable.")
        if session.status != SessionStatus.PENDING or session.run_epoch != 0:
            raise CollaborationUnavailable("Recipient is no longer an inert preparation.")
        return session

    @asynccontextmanager
    async def _adopted_material(self, prepared):
        """No foreign effect: keep authenticated child's existing pins retained.

        _native_evidence has authenticated the durable child before this guard.
        The current receiving mandate is already held. Do not reacquire an old
        acquisition mandate or nest per-resource revocation locks here.
        """
        from cayu.artifacts.resources import LocalArtifactResourceOwner

        target = prepared.target
        if isinstance(target, ContinueRecipientAdmissionTarget) or not target.resources:
            yield
            return
        owner = self._resource_owners.get(prepared.recipient.owner)
        if not isinstance(owner, LocalArtifactResourceOwner):
            raise CollaborationUnavailable("Recipient material owner is not qualified.")
        async with owner._hold_adopted_recipient_material(
            target.resources, recipient=prepared.recipient
        ):
            yield

    async def _native_evidence(self, command: RequestAdmissionCommand) -> None:
        prepared = command.prepared
        assert prepared is not None
        if self._resolve_budget is None:
            raise CollaborationUnavailable("New prepared admission requires its budget receiver.")
        if isinstance(prepared.target, ContinueRecipientAdmissionTarget):
            from cayu.sessions._recipient_continuation import require_continuation_selection_store

            require_continuation_selection_store(self._sessions)
            expected = prepared.target.selection
            current = await self._sessions.capture_recipient_continuation(expected.session_id)
            require_exact_contract(expected, current, redactor=self._redactor)
            session = await self._sessions.load(expected.session_id)
            if (
                session is None
                or session.instance_id != expected.session_instance_id
                or session.run_epoch != expected.run_epoch
                or session.status is not SessionStatus.COMPLETED
            ):
                raise CollaborationUnavailable("Recipient continuation is no longer available.")
        else:
            session = await self._created_session(prepared)
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
        require_prepared_budget_target(
            binding,
            provider_name=session.provider_name,
            model=session.model,
            environment_name=session.environment_name,
        )

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
                async with self._adopted_material(command.prepared):
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
