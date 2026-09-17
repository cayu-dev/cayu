"""Qualified participant permits for the source-owned export transaction.

The session retains the exact handoff before collaboration registration. Neither
absence nor an expired worker proves exclusion; only a terminal source record can
settle participant responsibility. No database transaction spans both owners.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.collaboration._contracts import (
    ExactLookup,
    ExactMatch,
    ExactUnavailable,
    InitiatorBinding,
    ObjectRef,
    OwnerRef,
)
from cayu.collaboration._permit_store import prepare_permit
from cayu.collaboration._permits import (
    PermitCommand,
    PermitIntent,
    PermitRegistration,
    PermitSettlementReader,
    ReceivingSettlementReceipt,
)
from cayu.collaboration._session_export_store import (
    ExportAdmission,
    ExportPreparation,
    ExportRecord,
    digest,
    operation_key,
    read_scope,
    source_digest,
)
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import LIFECYCLE_FAMILY
from cayu.collaboration.exports import (
    SessionExportAuthorization,
    SessionExportCapacityExceeded,
    SessionExportConflict,
    SessionExportDenied,
    SessionExportRequest,
    SessionExportUnavailable,
)
from cayu.collaboration.lifecycle import (
    NamespaceInspection,
    NamespaceRef,
    NamespaceRetirementEvidence,
)
from cayu.collaboration.participants import CollaborationUnavailable

if TYPE_CHECKING:
    from cayu.collaboration._coordinator import ParticipantCoordinator
    from cayu.collaboration._session_export_coordinator import SessionExportCoordinator


class _SourceSettlementReader(PermitSettlementReader):
    def __init__(self, exports: SessionExportCoordinator, admission: ExportAdmission):
        self.exports = exports
        self.admission = admission

    @property
    def owner(self) -> OwnerRef:
        assert self.exports.owner is not None
        return self.exports.owner

    async def lookup(self, expected: PermitCommand) -> ExactLookup[ReceivingSettlementReceipt]:
        if expected != self.admission.permit:
            raise SessionExportConflict()
        request = self.admission.request
        session = await self.exports.session(
            request.ref.session_id, request.ref.session_instance_id
        )
        root = await self.exports.root(session)
        if root is None:
            return ExactUnavailable()
        self.exports.validate_namespace(root, request)
        with read_scope(session.id):
            raw = await self.exports.store.load_session_operation(
                session.id, operation_key(request.ref.operation)
            )
        if raw is None:
            return ExactUnavailable()
        if "receipt" in raw:
            record = self.exports.prepare(ExportRecord, raw)
            admission = record.admission
            outcome = "quiescent"
        else:
            record = self.exports.prepare(ExportPreparation, raw)
            admission = record.admission
            if record.state != "excluded":
                return ExactUnavailable()
            outcome = "excluded"
        if admission is None or admission.model_copy(
            update={"settled": False}
        ) != self.admission.model_copy(update={"settled": False}):
            raise SessionExportConflict()
        return ExactMatch[ReceivingSettlementReceipt](
            receipt=ReceivingSettlementReceipt(
                expected=expected,
                receiving_owner=self.owner,
                receipt_id=digest(raw),
                outcome=outcome,
            )
        )


class ExportParticipantAdapter:
    def __init__(self, exports: SessionExportCoordinator, participants: ParticipantCoordinator):
        self.exports = exports
        self.participants = participants

    async def inspect(self, resolution) -> None:
        participant = resolution.chain.entries[-1].participant
        if participant is None:
            raise SessionExportDenied()
        await self.participants.inspect(
            participant,
            context=CollaborationAccessContext(principal=resolution.principal.principal),
        )

    async def admit(self, session, request, authorization, source, *, reserved_bytes):
        exports = self.exports
        commitment = source_digest(source)
        proposal = None
        for _ in range(4):
            root = await exports.root(session)
            if root is None:
                raise SessionExportUnavailable()
            current = await exports._record(session, request)
            if isinstance(current, ExportRecord):
                exports.require_replay_identity(
                    current.receipt.expected.intent.authorization, authorization
                )
                return current
            if isinstance(current, ExportPreparation):
                exports.require_replay_identity(current.admission.authorization, authorization)
                if current.state != "prepared" or current.admission.source_commitment != commitment:
                    raise SessionExportConflict()
                proposal = current
                break
            if proposal is None:
                proposal = exports.prepare(
                    ExportPreparation,
                    {
                        "admission": await self.prepare(request, authorization, commitment),
                        "reserved_bytes": reserved_bytes,
                    },
                )
            limits = root.namespace.limits
            exports.preflight_preparation(proposal)
            if (
                root.export_count >= limits.max_exports
                or root.pending_count >= limits.max_pending
                or root.admission_count >= limits.max_pending
                or root.retained_bytes + reserved_bytes > limits.max_retained_bytes
            ):
                raise SessionExportCapacityExceeded()
            desired = root.model_copy(
                update={
                    "export_count": root.export_count + 1,
                    "pending_count": root.pending_count + 1,
                    "admission_count": root.admission_count + 1,
                    "retained_bytes": root.retained_bytes + reserved_bytes,
                }
            )
            try:
                await exports.publish(
                    session,
                    root,
                    desired,
                    operation_key(request.ref.operation),
                    proposal.model_dump(mode="json"),
                    authorization,
                    [],
                    source,
                )
                break
            except Exception as error:
                current = await exports._record(session, request)
                if current is not None:
                    if isinstance(current, ExportRecord):
                        exports.require_replay_identity(
                            current.receipt.expected.intent.authorization, authorization
                        )
                        return current
                    exports.require_replay_identity(current.admission.authorization, authorization)
                    if current != proposal:
                        raise SessionExportConflict() from None
                    break
                if not isinstance(error, SessionExportConflict):
                    raise
        else:
            raise SessionExportConflict()
        assert proposal is not None
        await self.register(proposal.admission)
        # A reconciler may have excluded the source while registration was
        # pending or its acknowledgement was delayed. Never dispatch projection
        # merely because the participant owner acknowledged the older command.
        current = await exports._record(session, request)
        if isinstance(current, ExportRecord):
            exports.require_replay_identity(
                current.receipt.expected.intent.authorization, authorization
            )
            return current
        if current != proposal:
            raise SessionExportUnavailable()
        return proposal

    async def complete(self, session, record: ExportRecord, authorization) -> ExportRecord:
        if record.admission is None or record.admission.settled:
            return record
        await self.settle(record.admission)
        exports = self.exports
        request = record.receipt.expected.intent.request
        for _ in range(4):
            root = await exports.root(session)
            current = await exports._record(session, request)
            if (
                not isinstance(current, ExportRecord)
                or current.receipt != record.receipt
                or current.admission is None
            ):
                raise SessionExportConflict()
            if current.admission.settled:
                return current
            if root is None or root.admission_count == 0:
                raise SessionExportConflict()
            updated = exports.prepare(
                ExportRecord,
                current.model_copy(
                    update={
                        "admission": current.admission.model_copy(update={"settled": True}),
                    }
                ),
            )
            desired = root.model_copy(update={"admission_count": root.admission_count - 1})
            try:
                await exports.publish(
                    session,
                    root,
                    desired,
                    operation_key(request.ref.operation),
                    updated.model_dump(mode="json"),
                    authorization,
                    [],
                    expected_old=current.model_dump(mode="json"),
                )
                return updated
            except Exception as error:
                settled = await exports._record(session, request)
                if (
                    isinstance(settled, ExportRecord)
                    and settled.receipt == record.receipt
                    and settled.admission is not None
                    and settled.admission.settled
                ):
                    return settled
                if not isinstance(error, SessionExportConflict):
                    raise
        raise SessionExportConflict()

    async def exclude(self, session, request, authorization) -> ExportPreparation | ExportRecord:
        """Exclude publication before discharging, even if registration is still in flight."""
        exports = self.exports
        for _ in range(4):
            root = await exports.root(session)
            current = await exports._record(session, request)
            if isinstance(current, ExportRecord):
                return await self.complete(session, current, authorization)
            if root is None or not isinstance(current, ExportPreparation):
                raise SessionExportUnavailable()
            if current.state == "excluded":
                break
            if root.pending_count == 0:
                raise SessionExportConflict()
            excluded = exports.prepare(
                ExportPreparation,
                current.model_copy(
                    update={
                        "state": "excluded",
                        "excluded_by": exports.initiator(authorization),
                        "exclusion_mandate_commitment": exports.mandate_commitment(authorization),
                    }
                ),
            )
            desired = root.model_copy(update={"pending_count": root.pending_count - 1})
            try:
                await exports.publish(
                    session,
                    root,
                    desired,
                    operation_key(request.ref.operation),
                    excluded.model_dump(mode="json"),
                    authorization,
                    [],
                    expected_old=current.model_dump(mode="json"),
                )
                current = excluded
                break
            except Exception as error:
                reconciled = await exports._record(session, request)
                if isinstance(reconciled, ExportPreparation) and reconciled.state == "excluded":
                    current = reconciled
                    break
                if not isinstance(error, SessionExportConflict):
                    raise
        else:
            raise SessionExportConflict()
        if current.admission.settled:
            return current
        store, initialized = self.participants._ready()
        result = await self.participants._store_result(
            store._exclude_permit(
                initialized,
                current.admission.permit,
                reader=_SourceSettlementReader(exports, current.admission),
                redactor=exports.redactor,
            )
        )
        if result.expected != current.admission.permit:
            raise SessionExportUnavailable()
        for _ in range(4):
            root = await exports.root(session)
            current = await exports._record(session, request)
            if not isinstance(current, ExportPreparation) or current.state != "excluded":
                raise SessionExportConflict()
            if current.admission.settled:
                return current
            if root is None or root.admission_count == 0:
                raise SessionExportConflict()
            updated = exports.prepare(
                ExportPreparation,
                current.model_copy(
                    update={
                        "admission": current.admission.model_copy(update={"settled": True}),
                    }
                ),
            )
            desired = root.model_copy(update={"admission_count": root.admission_count - 1})
            try:
                await exports.publish(
                    session,
                    root,
                    desired,
                    operation_key(request.ref.operation),
                    updated.model_dump(mode="json"),
                    authorization,
                    [],
                    expected_old=current.model_dump(mode="json"),
                )
                return updated
            except Exception as error:
                reconciled = await exports._record(session, request)
                if isinstance(reconciled, ExportPreparation) and reconciled.admission.settled:
                    return reconciled
                if not isinstance(error, SessionExportConflict):
                    raise
        raise SessionExportConflict()

    async def prepare(
        self,
        request: SessionExportRequest,
        authorization: SessionExportAuthorization,
        source_commitment: str,
    ) -> ExportAdmission:
        resolution = authorization.mandate
        if resolution is None or resolution.chain.entries[-1].participant is None:
            raise SessionExportDenied()
        participant = resolution.chain.entries[-1].participant
        context = CollaborationAccessContext(principal=resolution.principal.principal)
        # Actual participant access boundary, not an identity-shaped assertion.
        inspection = await self.participants.inspect(participant, context=context)
        if inspection.participant.lifecycle != "active":
            raise SessionExportDenied()
        store, initialized = self.participants._ready()
        self.participants._capability(store, initialized, mutation=True, family=LIFECYCLE_FAMILY)
        namespace = self.exports.prepare(
            NamespaceInspection,
            await self.participants._store_result(
                store.inspect_namespace(initialized, redactor=self.exports.redactor)
            ),
        ).current.reference
        if (
            namespace.owner != initialized.owner
            or namespace.namespace_incarnation != initialized.namespace_incarnation
        ):
            raise SessionExportConflict()
        assert self.exports.owner is not None
        target = ObjectRef(
            owner=self.exports.owner,
            kind="session_export",
            object_id=operation_key(request.ref.operation),
            incarnation=request.ref.session_instance_id,
            revision=1,
        )
        identity = digest(
            {
                "source": request.ref.model_dump(mode="json"),
                "target": target.model_dump(mode="json"),
            }
        )
        permit = prepare_permit(
            initialized,
            PermitCommand(
                operation=namespace.operation("export-admission:" + identity),
                source=initialized.owner,
                destination=initialized.owner,
                initiator=InitiatorBinding(
                    issuer=initialized.owner,
                    principal=context.principal,
                    participant=authorization.initiating_identity().participant,
                    mandate=authorization.initiating_identity().mandate,
                    invocation_id=None,
                    interaction_id=None,
                ),
                intent=PermitIntent(
                    request=PermitRegistration(
                        operation=namespace.operation("export-admission:" + identity),
                        participant=participant,
                        expected_lifecycle_revision=inspection.participant.lifecycle_revision,
                        admission_generation=inspection.participant.admission_generation,
                        source_operation=request.ref.operation,
                        target=target,
                        target_state="future",
                        effect_scope="source_export",
                        required_settlement="exclusion",
                        settlement_operation=namespace.operation("export-settled:" + identity),
                    ),
                    limits=initialized.binding.limits,
                ),
            ),
            self.exports.redactor,
        )
        return self.exports.prepare(
            ExportAdmission,
            {
                "request": request,
                "authorization": authorization,
                "source_commitment": source_commitment,
                "permit": permit,
            },
        )

    async def register(self, admission: ExportAdmission) -> None:
        store, initialized = self.participants._ready()
        result = await self.participants._store_result(
            store._register_permit(initialized, admission.permit, redactor=self.exports.redactor)
        )
        if result.expected != admission.permit:
            raise SessionExportUnavailable()

    async def settle(self, admission: ExportAdmission) -> None:
        store, initialized = self.participants._ready()
        try:
            result = await self.participants._store_result(
                store._settle_permit(
                    initialized,
                    admission.permit,
                    reader=_SourceSettlementReader(self.exports, admission),
                    redactor=self.exports.redactor,
                )
            )
        except CollaborationUnavailable:
            # Publication at the source proves this exact permit was admitted.
            # Native settlement can commit before the source records its ACK, and
            # legitimate maintenance may then prune the native receipt. Positive
            # retirement of that exact namespace proves no obligations remain and
            # no future registration can reopen it. Absence alone proves nothing;
            # this discharges source responsibility, never fabricates a receipt.
            operation = admission.permit.operation
            namespace = NamespaceRef(
                owner=admission.permit.source,
                namespace_incarnation=operation.namespace_incarnation,
                generation=operation.generation,
            )
            raw = await self.participants._store_result(
                store.inspect_retirement(initialized, namespace, redactor=self.exports.redactor)
            )
            if raw is None:
                raise
            retirement = self.exports.prepare(NamespaceRetirementEvidence, raw)
            if retirement.namespace != namespace:
                raise SessionExportConflict() from None
            return
        if result.expected != admission.permit:
            raise SessionExportUnavailable()
