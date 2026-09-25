"""Corroborate resource adoption against the planner's exact creation stage."""

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._planning_creation_types import RequestCreationStageReceipt
from cayu.collaboration._planning_resource_types import resource_stage_adoption
from cayu.collaboration._preparation import prepare_contract, require_exact_contract


async def require_resource_adoption(tx, intent, receipt, *, redactor):
    from cayu.collaboration._planning_records import RequestPlanningStageRecord
    from cayu.collaboration._planning_stages import read_stage
    from cayu.collaboration.planning import MAX_REQUEST_PLANNING_STAGES

    stages = await tx.scan_request_plan_stages(intent.plan, limit=MAX_REQUEST_PLANNING_STAGES + 1)
    if len(stages) > MAX_REQUEST_PLANNING_STAGES:
        raise CollaborationConflict("Resource adoption exceeds its bounded planning history.")
    candidates = []
    for raw in stages:
        stage = prepare_contract(RequestPlanningStageRecord, raw, redactor=redactor)
        if stage.intent.command.operation == receipt.creation_operation:
            candidates.append(stage)
    if len(candidates) != 1:
        raise CollaborationConflict("Resource adoption lacks one exact retained creation stage.")
    stage = await read_stage(tx, candidates[0].intent, redactor=redactor)
    if (
        stage is None
        or stage.state != "settled"
        or not isinstance(stage.receipt, RequestCreationStageReceipt)
        or stage.intent.plan != intent.plan
        or stage.intent.plan_sha256 != intent.plan_sha256
        or stage.intent.ordinal <= intent.ordinal
    ):
        raise CollaborationConflict("Resource adoption lacks settled downstream creation evidence.")
    expected = resource_stage_adoption(intent.command, stage.receipt, redactor=redactor)
    require_exact_contract(expected, receipt, redactor=redactor)
