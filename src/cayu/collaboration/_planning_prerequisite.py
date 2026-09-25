"""An existing registered reader, never a caller-provided prerequisite receipt."""

from dataclasses import dataclass

from cayu.collaboration._contracts import (
    CollaborationConflict,
    ContractValue,
    ExactConflict,
    ExactLookup,
    ExactMatch,
)
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.planning import RequestPlanningPrerequisite
from cayu.collaboration.requests import RequestAdmissionReceipt


@dataclass(frozen=True)
class _PrerequisiteRead:
    expected: RequestPlanningPrerequisite


@dataclass(frozen=True)
class _PrerequisiteEvidence:
    expected: RequestPlanningPrerequisite
    receipt: RequestAdmissionReceipt


class _Result(ContractValue):
    lookup: ExactLookup[RequestAdmissionReceipt]


async def read_prerequisite(requests, pending, context):
    expected = prepare_contract(
        RequestPlanningPrerequisite, pending.expected, redactor=requests._redactor
    )
    reader = requests._planning_readers.get(expected.reader)
    if reader is None:
        raise CollaborationUnavailable("The exact prerequisite reader is not registered.")
    result = prepare_contract(
        _Result,
        {"lookup": await reader.lookup(expected.expected, context=context)},
        redactor=requests._redactor,
    ).lookup
    if isinstance(result, ExactConflict):
        raise CollaborationConflict("Planning prerequisite conflicts with receiving evidence.")
    if not isinstance(result, ExactMatch):
        raise CollaborationUnavailable("Planning prerequisite has no positive receiving evidence.")
    require_exact_contract(expected.expected, result.receipt.command, redactor=requests._redactor)
    return _PrerequisiteEvidence(expected, result.receipt)
