"""Native creation/exclusion adapter for retained planning responsibilities.

This owner uses the existing application authorization, creation fence and permit
settlement. It never infers a terminal outcome from a missing receiving record.
"""

from cayu.collaboration._contracts import ExactMatch, ExactUnavailable
from cayu.collaboration._planning_creation_evidence import _CreationReadback
from cayu.collaboration._planning_creation_types import (
    RequestCreationStageCommand,
    RequestCreationStageReceipt,
)
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.collaboration.prepared_admission import (
    created_admission_target,
)
from cayu.collaboration.recipient_preparation import (
    MaterialRecipientCreationPreparation,
    require_secret_free_preparation,
)
from cayu.sessions._recipient_admission import settle_recipient_creation
from cayu.sessions._recipient_preparation import (
    require_created_preparation,
    resolved_material_creation_request,
)
from cayu.sessions.context_views import ParticipantSessionCreationReceipt
from cayu.sessions.creation_fence import (
    _SESSION_CREATION_AUTHORITY,
    SessionCreationDecision,
    validate_binding,
)


class NativePlanningCreationOwner:
    """Trusted application-wired adapter, not a caller-installed receiver."""

    def __init__(self, application):
        self._app = application

    def _command(self, value):
        command = prepare_contract(
            RequestCreationStageCommand, value, redactor=self._app._secret_redactor
        )
        proposal = command.preparation
        require_secret_free_preparation(
            proposal.base
            if isinstance(proposal, MaterialRecipientCreationPreparation)
            else proposal,
            self._app._secret_redactor,
        )
        return command

    async def create(self, expected, *, context):
        command = self._command(expected)
        proposal = command.preparation
        resource_owner = None
        if isinstance(proposal, MaterialRecipientCreationPreparation):
            participants = self._app._participant_coordinator
            _, grant = participants._authorize(context, "administration")
            participants._require_refs(
                grant, (proposal.creation.permit.intent.request.participant,)
            )
            prior = await self.read(command)
            if isinstance(prior, _CreationReadback):
                return prior
            selected = (
                None
                if proposal.selection is None
                else await self._app.session_store.lookup_context_view_selection(
                    proposal.selection.selection_key
                )
            )
            from cayu.artifacts.resources import LocalArtifactResourceOwner
            from cayu.collaboration.participants import CollaborationUnavailable

            materials = []
            for reference in proposal.resources:
                owner = self._app._request_coordinator._resource_owners.get(reference.owner)
                if not isinstance(owner, LocalArtifactResourceOwner):
                    raise CollaborationUnavailable("Recipient resource owner is not qualified.")
                if resource_owner is not None and owner is not resource_owner:
                    raise CollaborationUnavailable("Recipient resources have conflicting owners.")
                resource_owner = owner
                materials.append(await owner.read_material(reference))
            creation = resolved_material_creation_request(
                proposal,
                selected,
                tuple(item.transfer for item in materials),
                tuple(item.preparation for item in materials),
            )
        else:
            creation = proposal.creation_request
        await self._app.create_recipient_session(
            creation,
            context=context,
            preparation=command.preparation,
            resource_owner=resource_owner,
        )
        return await self.read(command)

    async def exclude(self, expected, *, context):
        command = self._command(expected)
        target = command.preparation.creation
        participants = self._app._participant_coordinator
        _, grant = participants._authorize(context, "administration")
        participants._require_refs(grant, (target.permit.intent.request.participant,))
        # The caller has already sealed the local stage owner. An earlier native
        # creation may still win; the returned native decision, not cancellation,
        # determines which exact terminal responsibility can be discharged.
        await self._app.session_store._exclude_session_creation_target(
            target, authority=_SESSION_CREATION_AUTHORITY
        )
        return await self.read(command)

    async def read(self, expected):
        command = self._command(expected)
        proposal = command.preparation
        target = proposal.creation
        store = self._app.session_store
        found = await store.read_session_creation_decision(target)
        if not isinstance(found, ExactMatch) or found.receipt.state == "pending":
            return ExactUnavailable()
        decision = prepare_contract(
            SessionCreationDecision, found.receipt, redactor=self._app._secret_redactor
        )
        require_exact_contract(target, decision.target, redactor=self._app._secret_redactor)
        # Reconcile the existing source permit before acknowledging receiving
        # settlement. Lost ACK on either side retains the same discoverable key.
        await settle_recipient_creation(self._app, target)
        found = await store.read_session_creation_decision(target)
        if not isinstance(found, ExactMatch):
            return ExactUnavailable()
        decision = prepare_contract(
            SessionCreationDecision, found.receipt, redactor=self._app._secret_redactor
        )
        definition_commitment = None
        if decision.state == "created":
            assert decision.session_id is not None
            assert decision.session_instance_id is not None
            assert decision.creation_receipt_commitment is not None
            receipt = prepare_contract(
                ParticipantSessionCreationReceipt,
                await store.load_participant_session_creation_receipt(decision.session_id),
                redactor=self._app._secret_redactor,
            )
            validate_binding(
                target, receipt.binding, requested_session_id=receipt.requested_session_id
            )
            require_created_preparation(proposal, receipt)
            if (
                receipt.binding.session_id != decision.session_id
                or receipt.binding.session_instance_id != decision.session_instance_id
                or receipt.receipt_commitment != decision.creation_receipt_commitment
            ):
                raise ValueError("Native creation readback has conflicting child evidence.")
            definition_commitment = created_admission_target(target, receipt).definition_commitment
        return _CreationReadback(
            RequestCreationStageReceipt(
                command=command,
                decision=decision,
                definition_commitment=definition_commitment,
            )
        )
