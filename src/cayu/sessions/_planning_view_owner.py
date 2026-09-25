"""Native retained-view handoff using the existing participant permit owner.

Preparation freezes data. Registration orders against participant disablement
in CollaborationStore. The native selection transaction consumes only that
exact registered target. No transaction spans the two stores, and a timeout
never proves exclusion. Selection quiescence does not discharge pin ownership.
"""

from hashlib import sha256

from cayu.collaboration._contracts import (
    ExactConflict,
    ExactMatch,
    ExactUnavailable,
    InitiatorBinding,
    ObjectRef,
)
from cayu.collaboration._permit_store import prepare_permit
from cayu.collaboration._permits import (
    PermitCommand,
    PermitIntent,
    PermitRegistration,
    PermitSettlementReader,
    ReceivingSettlementReceipt,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.participants import CollaborationUnavailable, ParticipantRef
from cayu.sessions._context_selection_fence import (
    _CONTEXT_SELECTION_AUTHORITY,
    ContextViewSelectionConflict,
    ContextViewSelectionDecision,
    ContextViewSelectionTarget,
    require_selection_fence_store,
    selection_admission_commitment,
    selection_adoption_request,
    selection_receiving_target,
)
from cayu.sessions.context_views import ContextViewOwnershipRequest, ContextViewSelectionRequest


class NativePlanningViewOwner:
    """Application-wired adapter; not a caller-installed authorization reader."""

    def __init__(self, application):
        self._app = application

    def _target(self, value):
        require_selection_fence_store(self._app.session_store)
        return prepare_contract(
            ContextViewSelectionTarget, value, redactor=self._app._secret_redactor
        )

    def _access(self, target, context, *, cleanup=False):
        coordinator = self._app._participant_coordinator
        _, grant = coordinator._authorize(context, "administration")
        coordinator._require_refs(
            grant, tuple(p.intent.request.participant for p in target.permits)
        )
        if not cleanup and context.principal != target.permit.initiator.principal:
            raise PermissionError("Selection initiator conflicts with its retained operation.")
        return coordinator

    async def prepare(self, request, *, participant, context, deadline_at_ms, recipient=None):
        """Freeze one exact proposal without registering or acquiring a pin."""
        app = self._app
        require_selection_fence_store(app.session_store)
        request = prepare_contract(
            ContextViewSelectionRequest, request, redactor=app._secret_redactor
        )
        participant = prepare_contract(ParticipantRef, participant, redactor=app._secret_redactor)
        recipient = (
            participant
            if recipient is None
            else prepare_contract(ParticipantRef, recipient, redactor=app._secret_redactor)
        )
        coordinator = app._participant_coordinator
        _, grant = coordinator._authorize(context, "administration")
        coordinator._require_refs(grant, tuple(dict.fromkeys((participant, recipient))))
        found = await app.session_store.read_context_view_selection_decision(request)
        if isinstance(found, ExactConflict):
            raise ContextViewSelectionConflict(
                "Selection proposal conflicts with its retained request."
            )
        if isinstance(found, ExactUnavailable):
            raise CollaborationUnavailable("Selection proposal readback is unavailable.")
        if isinstance(found, ExactMatch):
            target = found.receipt.target
            if (
                target is None
                or target.deadline_at_ms != deadline_at_ms
                or target.permit.intent.request.participant != participant
                or target.recipient != recipient
            ):
                raise ContextViewSelectionConflict("Selection proposal authority conflicts.")
            self._access(target, context)
            return self._target(target)
        inspected = await coordinator.inspect(participant, context=context, action="administration")
        snapshot = inspected.participant
        if snapshot.lifecycle != "active" or participant.owner != request.source_owner:
            raise PermissionError("Active source participant authority is required.")
        binding = await app.session_store.load_participant_session_binding(
            request.source_session_id
        )
        if (
            binding is None
            or binding.participant != participant
            or binding.session_instance_id != request.source_session_instance_id
        ):
            raise PermissionError("Selection proposal does not own the exact source session.")
        _, initialized = coordinator._ready()
        recipient_permit = None
        if recipient != participant:
            destination = (
                await coordinator.inspect(recipient, context=context, action="administration")
            ).participant
            if destination.lifecycle != "active":
                raise PermissionError("Active recipient retention authority is required.")
            recipient_permit = self._permit(
                request, destination, initialized, context, deadline_at_ms, recipient=True
            )
        permit = self._permit(
            request,
            snapshot,
            initialized,
            context,
            deadline_at_ms,
            recipient_permit=recipient_permit,
        )
        return self._target(
            ContextViewSelectionTarget(
                request=request,
                permit=permit,
                deadline_at_ms=deadline_at_ms,
                recipient_permit=recipient_permit,
            )
        )

    def _permit(
        self,
        request,
        snapshot,
        initialized,
        context,
        deadline_at_ms,
        *,
        recipient=False,
        recipient_permit=None,
    ):
        participant = snapshot.reference
        key = ("plan-view-recipient:" if recipient else "plan-view:") + sha256(
            request.selection_key.encode()
        ).hexdigest()
        operation = initialized.operation(key)
        return prepare_permit(
            initialized,
            PermitCommand(
                operation=operation,
                source=initialized.owner,
                destination=initialized.owner,
                initiator=InitiatorBinding(
                    issuer=initialized.owner,
                    principal=context.principal,
                    participant=ObjectRef(
                        owner=participant.owner,
                        kind="participant",
                        object_id=participant.participant_id,
                        incarnation=participant.incarnation,
                    ),
                    mandate=None,
                    invocation_id=None,
                    interaction_id=None,
                ),
                intent=PermitIntent(
                    limits=initialized.binding.limits,
                    request=PermitRegistration(
                        operation=operation,
                        participant=participant,
                        expected_lifecycle_revision=snapshot.lifecycle_revision,
                        expected_configuration_revision=snapshot.configuration_revision,
                        admission_generation=snapshot.admission_generation,
                        admission_commitment=selection_admission_commitment(
                            request, deadline_at_ms, recipient_permit
                        ),
                        source_operation=initialized.operation(key + ":source"),
                        target=selection_receiving_target(request),
                        target_state="existing",
                        effect_scope="context_view_retention"
                        if recipient
                        else "context_view_selection",
                        required_settlement="exclusion",
                        settlement_operation=initialized.operation(key + ":settled"),
                    ),
                ),
            ),
            self._app._secret_redactor,
        )

    async def select(self, expected, *, context):
        target = self._target(expected)
        coordinator = self._access(target, context)
        store, initialized = coordinator._ready()
        native = self._app.session_store
        await native._prepare_context_view_selection_target(
            target, authority=_CONTEXT_SELECTION_AUTHORITY
        )
        # Exact registration is replayable after disablement only if it won
        # durably before that transition. A fresh stale proposal must fail.
        for permit in target.permits:
            await coordinator._store_result(
                store._register_permit(initialized, permit, redactor=self._app._secret_redactor)
            )
        await native._register_context_view_selection_target(
            target, authority=_CONTEXT_SELECTION_AUTHORITY
        )
        await native._select_context_view_target(target, authority=_CONTEXT_SELECTION_AUTHORITY)
        return await self.read(target)

    async def adopt(self, expected, *, context):
        target = self._target(expected)
        self._access(target, context)
        found = await self.read(target)
        if not isinstance(found, ExactMatch) or found.receipt.state != "selected":
            raise CollaborationUnavailable("Adoption requires exact acquired view evidence.")
        await self._app.session_store._adopt_context_view_target(
            target,
            selection_adoption_request(target, found.receipt),
            authority=_CONTEXT_SELECTION_AUTHORITY,
        )
        return await self._app.session_store._read_context_view_retention(target)

    async def read(self, expected):
        target = self._target(expected)
        found = await self._app.session_store.read_context_view_selection_decision(target.request)
        if not isinstance(found, ExactMatch):
            return found
        decision = prepare_contract(
            ContextViewSelectionDecision, found.receipt, redactor=self._app._secret_redactor
        )
        if decision.target != target:
            return ExactConflict()
        return ExactMatch[ContextViewSelectionDecision](receipt=decision)

    async def exclude(self, expected, *, context):
        target = self._target(expected)
        coordinator = self._access(target, context, cleanup=True)
        native = self._app.session_store
        decision = await native._exclude_context_view_selection_target(
            target, authority=_CONTEXT_SELECTION_AUTHORITY
        )
        if decision.state == "excluded":
            store, initialized = coordinator._ready()
            # A reservation can be created before the planner's permits.  Its
            # native exclusion must still be durable, but there is no permit
            # obligation to settle in that case.  For ordinary native callers
            # (and admitted planner stages), preserve the existing settlement.
            from cayu.collaboration._permit_store import registered_receipt
            from cayu.collaboration._planning_stages import read_stage
            from cayu.collaboration._planning_view_reservation import view_permit_stage

            permits_to_exclude = []
            async with store._transaction(initialized.binding.application_scope, write=False) as tx:
                for permit in target.permits:
                    stage = await view_permit_stage(tx, permit, redactor=self._app._secret_redactor)
                    if stage is not None and (
                        await registered_receipt(tx, permit, self._app._secret_redactor) is None
                    ):
                        stage = await read_stage(
                            tx, stage.intent, redactor=self._app._secret_redactor
                        )
                        if stage is None or stage.state != "excluded":
                            raise CollaborationUnavailable(
                                "Unregistered planning permits require a durable stage exclusion."
                            )
                        continue
                    permits_to_exclude.append(permit)
            # Ordinary native registration races exclusion in the permit writer
            # transaction. Fence absent permits too; a read-only absence check
            # cannot prevent a delayed registration from creating obligations.
            for permit in permits_to_exclude:
                await coordinator._store_result(
                    store._exclude_permit(
                        initialized,
                        permit,
                        reader=_ViewSettlementReader(native, target),
                        redactor=self._app._secret_redactor,
                    )
                )
        # A selected decision still owns a retention pin and must be settled by
        # the later exact release/recipient handoff, never by observer timeout.
        return decision

    async def release(self, expected, request, *, context):
        """Discharge one explicitly retained release intent, without new access."""
        target = self._target(expected)
        coordinator = self._access(target, context, cleanup=True)
        request = prepare_contract(
            ContextViewOwnershipRequest, request, redactor=self._app._secret_redactor
        )
        found = await self.read(target)
        if not isinstance(found, ExactMatch) or found.receipt.state != "selected":
            raise CollaborationUnavailable("Selection has no positive retained acquisition.")
        selection = found.receipt
        if (
            request.operation != "release"
            or request.current_participant is None
            or request.selection_key != target.request.selection_key
            or request.view_id != selection.view_id
            or request.pin_commitment != selection.pin_commitment
        ):
            raise ContextViewSelectionConflict("Selection release intent conflicts.")
        native = self._app.session_store
        reader = _ViewSettlementReader(native, target)
        retention = await native._read_context_view_retention(target)
        if not isinstance(retention, ExactMatch) or retention.receipt.state != "expired":
            # Even a prior release must replay its complete exact native command;
            # terminal state alone cannot authenticate a different operation key.
            await self._app.transition_context_view_ownership(
                request, participant=request.current_participant, context=context
            )
        await self._settle_retention(target, coordinator, reader)
        return await native._read_context_view_retention(target)

    async def _settle_retention(self, target, coordinator, reader):
        store, initialized = coordinator._ready()
        for permit in target.permits:
            await coordinator._store_result(
                store._settle_permit(
                    initialized, permit, reader=reader, redactor=self._app._secret_redactor
                )
            )

    async def discharge(self, expected, *, context):
        """Settle exact planner debt; only native evidence chooses the disposition.

        Exclusion arbitrates delayed selection first. A positive pin is released
        through its existing owner, not treated as an excluded acquisition. This
        operation is bound to the complete retained target and is reconstructable
        without an observer's in-process release handle.
        """
        target = self._target(expected)
        coordinator = self._access(target, context, cleanup=True)
        decision = await self.exclude(target, context=context)
        if decision.state == "excluded":
            return ExactMatch[ContextViewSelectionDecision](receipt=decision)
        native = self._app.session_store
        found = await native._read_context_view_retention(target)
        if not isinstance(found, ExactMatch):
            raise CollaborationUnavailable("Retention cleanup lacks native ownership evidence.")
        retained = found.receipt
        if retained.state in {"released", "expired"}:
            await self._settle_retention(target, coordinator, _ViewSettlementReader(native, target))
            return found
        if retained.state not in {"selected", "adopted"}:
            raise CollaborationUnavailable("Retention cleanup has an unsupported ownership state.")
        request = ContextViewOwnershipRequest(
            selection_key=target.request.selection_key,
            view_id=retained.selection.view_id,
            pin_commitment=retained.selection.pin_commitment,
            expected_state=retained.state,
            expected_revision=retained.revision,
            operation="release",
            current_owner=retained.owner,
            current_participant=retained.participant,
            operation_key="planning-release:"
            + sha256(target.model_dump_json().encode()).hexdigest(),
        )
        return await self.release(target, request, context=context)


class _ViewSettlementReader(PermitSettlementReader):
    def __init__(self, store, target):
        self.store, self.target = store, target

    @property
    def owner(self):
        return self.target.request.source_owner

    async def lookup(self, expected):
        if expected not in self.target.permits:
            raise ContextViewSelectionConflict("Selection exclusion authority conflicts.")
        found = await self.store.read_context_view_selection_decision(self.target.request)
        if not isinstance(found, ExactMatch) or found.receipt.target != self.target:
            return ExactUnavailable()
        excluded = found.receipt.state == "excluded"
        if not excluded:
            retention = await self.store._read_context_view_retention(self.target)
            if not isinstance(retention, ExactMatch) or retention.receipt.state not in {
                "released",
                "expired",
            }:
                return ExactUnavailable()
        return ExactMatch[ReceivingSettlementReceipt](
            receipt=ReceivingSettlementReceipt(
                expected=expected,
                receiving_owner=self.owner,
                receipt_id=("view-exclusion:" if excluded else "view-retention:")
                + sha256(self.target.model_dump_json().encode()).hexdigest(),
                outcome="excluded" if excluded else "quiescent",
                admission_excluded=excluded,
            )
        )
