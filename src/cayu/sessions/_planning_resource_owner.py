"""Registered native resource preparation, without alternate pin/permit owners."""

from cayu.artifacts.resources import (
    LocalArtifactResourceOwner,
    ResourceAcquisitionReceipt,
)
from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration._planning_resource_types import (
    RequestResourceAcquisitionStageCommand,
    RequestResourceStageRelease,
    RequestResourceTransferStageCommand,
    _ResourceAcquisitionReadback,
    _ResourceAdoptionReadback,
    _ResourceCreationReadback,
    _ResourceReleaseReadback,
    _ResourceTransferReadback,
    resource_stage_adoption,
    resource_stage_command,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.resource_preparation import RequestPlanningResource


class NativePlanningResourceOwner:
    def __init__(self, application):
        self._app = application

    def _access(self, expected, context):
        expected = prepare_contract(
            RequestPlanningResource, expected, redactor=self._app._secret_redactor
        )
        participants = self._app._participant_coordinator
        _, grant = participants._authorize(context, "administration")
        participants._require_refs(
            grant,
            (
                expected.acquisition_permit.intent.request.participant,
                expected.transfer_permit.intent.request.participant,
            ),
        )
        registered = self._app._request_coordinator._resource_owners
        source = registered.get(expected.acquisition.source)
        destination = registered.get(expected.transfer.destination)
        if not isinstance(source, LocalArtifactResourceOwner) or not isinstance(
            destination, LocalArtifactResourceOwner
        ):
            raise CollaborationUnavailable("Planning resource owners are not qualified.")
        return expected, source, destination

    async def acquire(self, expected, *, context):
        expected, source, _ = self._access(expected, context)
        found = await source.readback(expected.acquisition)
        if isinstance(found, ExactMatch):
            preparation = await source.read_preparation(expected.acquisition)
            if preparation.permit != expected.acquisition_permit:
                raise CollaborationUnavailable("Native acquisition has another registered permit.")
            return _ResourceAcquisitionReadback(expected, found.receipt)
        preparation = await source.authorize(
            expected.acquisition, permit=expected.acquisition_permit
        )
        receipt = await source.acquire(expected.acquisition, preparation=preparation)
        return _ResourceAcquisitionReadback(expected, receipt)

    async def transfer(self, expected, acquisition, *, context):
        expected, source, destination = self._access(expected, context)
        acquisition = prepare_contract(
            ResourceAcquisitionReceipt, acquisition, redactor=self._app._secret_redactor
        )
        command = expected.transfer.bind(acquisition)
        # Template binding is data only. Native acceptance authenticates the
        # actual source under both mutation fences before any destination pin.
        receipt = await destination.accept_transfer(
            command, source_owner=source, expected_permit=expected.transfer_permit
        )
        preparation = await destination.read_preparation(command)
        if preparation.permit != expected.transfer_permit:
            raise CollaborationUnavailable("Native transfer used another registered permit.")
        return _ResourceTransferReadback(expected, receipt, preparation)

    async def exclude_acquisition(self, expected, *, context):
        expected, source, _ = self._access(expected, context)
        return await source.settle_preparation(
            expected.acquisition, permit=expected.acquisition_permit
        )

    async def read_transfer(self, command, *, context):
        """Read an already accepted exact transfer; never acquire on absence."""
        command = prepare_contract(
            RequestResourceTransferStageCommand, command, redactor=self._app._secret_redactor
        )
        expected, _, destination = self._access(command.resource, context)
        found = await destination.read_transfer(command.native_command)
        if not isinstance(found, ExactMatch) or found.receipt.stage != "accepted":
            raise CollaborationUnavailable("Retained resource transfer is not accepted.")
        preparation = await destination.read_preparation(command.native_command)
        if preparation.permit != expected.transfer_permit:
            raise CollaborationUnavailable("Resource transfer has another preparation permit.")
        return _ResourceTransferReadback(expected, found.receipt, preparation)

    async def prepare_creation(self, record, stages, *, context):
        """Resolve resources from native owners, not serialized caller receipts."""
        from cayu.collaboration._planning_creation_types import creation_stage_command
        from cayu.collaboration._planning_fork_types import view_stage_command
        from cayu.collaboration._planning_records import (
            RequestPlanningRecord,
            RequestPlanningStageRecord,
        )
        from cayu.collaboration._preparation import require_exact_contract
        from cayu.collaboration.planning import RequestPlanningFork, RequestPlanningFresh
        from cayu.sessions._planning_fork_owner import NativePlanningForkOwner
        from cayu.sessions._recipient_preparation import resolve_resource_recipient

        record = prepare_contract(
            RequestPlanningRecord, record, redactor=self._app._secret_redactor
        )
        decision = record.decision
        if not isinstance(decision, (RequestPlanningFresh, RequestPlanningFork)) or not (
            decision.resources
        ):
            raise CollaborationUnavailable("Planning has no frozen resource preparation.")
        offset = 1 if isinstance(decision, RequestPlanningFork) else 0
        if type(stages) is not tuple or len(stages) != offset + 2 * len(decision.resources):
            raise CollaborationUnavailable(
                "Resource preparation lacks its complete retained frontier."
            )
        stages = tuple(
            prepare_contract(RequestPlanningStageRecord, stage, redactor=self._app._secret_redactor)
            for stage in stages
        )
        references = []
        for index in range(len(decision.resources)):
            command = stages[offset + 2 * index + 1].intent.command
            if not isinstance(command, RequestResourceTransferStageCommand):
                raise CollaborationUnavailable("Resource transfer frontier has another command.")
            require_exact_contract(
                resource_stage_command(
                    record,
                    index,
                    redactor=self._app._secret_redactor,
                    acquisition=command.acquisition,
                ),
                command,
                redactor=self._app._secret_redactor,
            )
            retained = await self.read_transfer(command, context=context)
            references.append(retained.reference)
        if isinstance(decision, RequestPlanningFork):
            prepared = await NativePlanningForkOwner(self._app).prepare(
                view_stage_command(record), context=context
            )
            base = prepared.preparation
        else:
            base = decision.preparation
        resolved = await resolve_resource_recipient(
            self._app, base, tuple(references), context=context
        )
        return _ResourceCreationReadback(
            record.receipt.command, creation_stage_command(record, preparation=resolved)
        )

    async def exclude_transfer(self, expected, acquisition, *, context):
        expected, source, destination = self._access(expected, context)
        acquisition = prepare_contract(
            ResourceAcquisitionReceipt, acquisition, redactor=self._app._secret_redactor
        )
        return await destination.settle_preparation(
            expected.transfer.bind(acquisition),
            permit=expected.transfer_permit,
            source_owner=source,
        )

    async def discharge_stage(self, command, *, context):
        """Reconcile the exact retained stage, including before native dispatch.

        Typed caller data alone cannot mint this private proof. The native
        receiving owner fences late acquisition/transfer and settles its
        original registered permit before this method returns.
        """
        if type(command) is RequestResourceAcquisitionStageCommand:
            command = prepare_contract(
                RequestResourceAcquisitionStageCommand,
                command,
                redactor=self._app._secret_redactor,
            )
            receiving = await self.exclude_acquisition(command.resource, context=context)
        elif type(command) is RequestResourceTransferStageCommand:
            command = prepare_contract(
                RequestResourceTransferStageCommand,
                command,
                redactor=self._app._secret_redactor,
            )
            receiving = await self.exclude_transfer(
                command.resource, command.acquisition, context=context
            )
        else:
            raise CollaborationUnavailable("Resource stage has no qualified native command.")
        return _ResourceReleaseReadback(
            RequestResourceStageRelease(command=command, receiving=receiving)
        )

    async def read_adoption(self, command, creation, *, context):
        """Authenticate the receiving child, without releasing destination debt.

        Creation data supplied by the caller is only an expectation. The native
        creation owner must reconstruct the same exact durable decision first.
        Historical preparation grants are not renewed; this returns no bytes.
        """
        from cayu.artifacts._resource_material import retained_material
        from cayu.collaboration._planning_creation_evidence import _CreationReadback
        from cayu.sessions._planning_creation_owner import NativePlanningCreationOwner

        command = prepare_contract(
            RequestResourceTransferStageCommand, command, redactor=self._app._secret_redactor
        )
        _, _, destination = self._access(command.resource, context)
        found = await NativePlanningCreationOwner(self._app).read(creation)
        if type(found) is not _CreationReadback:
            raise CollaborationUnavailable("Resource adoption has no native child decision.")
        receipt = resource_stage_adoption(
            command, found.receipt, redactor=self._app._secret_redactor
        )
        async with destination._hold_adopted_recipient_material(
            (receipt.material,),
            recipient=command.resource.transfer_permit.intent.request.participant,
        ):
            native = await retained_material(destination, receipt.material)
            if (
                native.transfer.command != command.native_command
                or native.preparation.permit != command.permit
            ):
                raise CollaborationUnavailable("Resource adoption has conflicting native material.")
            return _ResourceAdoptionReadback(receipt)
