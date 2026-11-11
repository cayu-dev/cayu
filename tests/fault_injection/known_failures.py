"""Faults the Runtime does not survive yet, each mapped to the issue it reproduces.

The exhaustive tests accept exactly these violations and nothing else. Every entry
also has a strict ``xfail`` test that injects at its first point only, so it
reports XPASS once a fix lands; then delete the entry (and any sibling entries
of the same finding that pass too). ``README.md`` describes each finding.

Entries were recorded by running every scenario and fault at every point; points
are labels from :func:`tests.fault_injection.explore.injection_points`. Each entry
lists the invariants its points break and the stage each is found at, so a known
defect in the live process doesn't excuse a new one after recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests.fault_injection.harness import FaultKind
from tests.fault_injection.invariants import Invariant, Stage

if TYPE_CHECKING:
    from tests.fault_injection.explore import Violation


@dataclass(frozen=True)
class Finding:
    issue: str
    summary: str


# Issues are cayu-tech/cayu. Findings with no issue of their own are tracked in #3390.
FINDINGS: dict[str, Finding] = {
    "group-crash-conflict": Finding(
        "#3390 (related #3245)",
        "After a worker dies mid-run its unsettled group execution makes the "
        "continuation fail with TaskGroupConflict, so the task stays RUNNING",
    ),
    "group-crash-after-complete": Finding(
        "#3390",
        "A crash after the member's live terminalization, before its worker acknowledges "
        "the execution, leaves the group obligation unsettled: nothing durable proves the "
        "callback returned",
    ),
    # Found by test_continuation_failure_crash.py, whose scenario fails the
    # approval close on purpose, so it has no point list of its own here.
    "continuation-pause-interaction-open": Finding(
        "#3345",
        "An approval continuation that fails before its close restores the pause with a "
        "raw status write, so the resumed interaction is never settled",
    ),
    "continuation-pause-event-lost": Finding(
        "#3345",
        "A crash between that raw status write and SESSION_INTERRUPTED loses the event: "
        "its ID is random and nothing owes it",
    ),
}


@dataclass(frozen=True)
class KnownFailure:
    finding: str
    scenario: str
    fault: FaultKind
    # The (invariant, stage) pairs every one of the points produces.
    violations: frozenset[tuple[str, str]]
    points: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.finding}[{self.scenario}-{self.fault.value}@{self.points[0]}]"

    @property
    def reason(self) -> str:
        finding = FINDINGS[self.finding]
        return f"{finding.issue}: {finding.summary}"

    def matches(self, violation: Violation) -> bool:
        return (
            violation.scenario == self.scenario
            and violation.fault is self.fault
            and (violation.invariant, violation.stage) in self.violations
            and violation.point in self.points
        )


def _known(
    finding: str,
    scenario: str,
    fault: FaultKind,
    violations: tuple[tuple[str, str], ...],
    points: tuple[str, ...],
) -> KnownFailure:
    assert finding in FINDINGS, finding
    return KnownFailure(finding, scenario, fault, frozenset(violations), points)


KNOWN_FAILURES: tuple[KnownFailure, ...] = (
    _known(
        "group-crash-conflict",
        "task-group",
        FaultKind.CRASH,
        (
            (Invariant.TASK_UNSETTLED, Stage.TASK_RECOVERY),
            (Invariant.TASK_UNSETTLED, Stage.RECOVERED),
        ),
        (
            "store._task_group_retains_execution:enter#1",
            "store._observe_task_group_invocation:enter#0",
            "store.append_event_and_claim_side_effect:enter#1",
            "store.mark_persisted_event_side_effect_delivered:enter#2",
            "store.append_transcript_messages:enter#0",
            "store.deliver_queued_session_messages:enter#0",
            "store.append_events:enter#0",
            "store.append_event_and_claim_side_effect:enter#2",
            "store.mark_persisted_event_side_effect_delivered:enter#3",
            "store.transform_checkpoint:enter#1",
            "store.append_event_and_claim_side_effect:enter#3",
            "store.mark_persisted_event_side_effect_delivered:enter#4",
            "store.append_event_and_claim_side_effect:enter#4",
            "store.mark_persisted_event_side_effect_delivered:enter#5",
            "store.prepare_model_completion_stage:enter#0",
            "store.mark_model_completion_stage_dispatched:enter#0",
            "store.complete_model_completion_stage:enter#0",
            "store.promote_model_completion_stage:enter#0",
            "store.claim_persisted_event_side_effect:enter#1",
            "store.mark_persisted_event_side_effect_delivered:enter#6",
            "store.transform_checkpoint:enter#2",
            "store.append_event_and_claim_side_effect:enter#5",
            "store.mark_persisted_event_side_effect_delivered:enter#7",
            "store.append_event_and_claim_side_effect:enter#6",
            "store.mark_persisted_event_side_effect_delivered:enter#8",
            "store.transform_checkpoint:enter#3",
            "store.transform_checkpoint:enter#4",
            "store.append_event_and_claim_side_effect:enter#7",
            "store.mark_persisted_event_side_effect_delivered:enter#9",
            "store.append_event_and_claim_side_effect:enter#8",
            "store.mark_persisted_event_side_effect_delivered:enter#10",
            "store.publish_runtime_publication:enter#0",
            "store.append_events:enter#1",
            "store.append_event_and_claim_side_effect:enter#9",
            "store.mark_persisted_event_side_effect_delivered:enter#11",
            "store.transform_checkpoint:enter#5",
            "store.append_event_and_claim_side_effect:enter#10",
            "store.mark_persisted_event_side_effect_delivered:enter#12",
            "store.append_event_and_claim_side_effect:enter#11",
            "store.mark_persisted_event_side_effect_delivered:enter#13",
            "store.prepare_model_completion_stage:enter#1",
            "store.mark_model_completion_stage_dispatched:enter#1",
            "store.append_event_and_claim_side_effect:enter#12",
            "store.mark_persisted_event_side_effect_delivered:enter#14",
            "store.complete_model_completion_stage:enter#1",
            "store.promote_model_completion_stage:enter#1",
            "store.claim_persisted_event_side_effect:enter#2",
            "store.mark_persisted_event_side_effect_delivered:enter#15",
            "store._retain_native_producer_output:enter#0",
            "store.settle_session_invocation:enter#0",
        ),
    ),
    _known(
        "group-crash-after-complete",
        "task-group",
        FaultKind.CRASH,
        ((Invariant.TASK_GROUP_OBLIGATION, Stage.RECOVERED),),
        (
            "store.append_event_and_claim_side_effect:enter#13",
            "store.mark_persisted_event_side_effect_delivered:enter#17",
            "store.append_event_and_claim_side_effect:enter#14",
            "store.mark_persisted_event_side_effect_delivered:enter#18",
            "store.append_event_and_claim_side_effect:enter#15",
            "store.mark_persisted_event_side_effect_delivered:enter#19",
            "store.release_session_invocation:enter#0",
            "store._release_session_execution:enter#0",
            "store._observe_task_group_invocation:enter#1",
            "store._observe_task_group_invocation:enter#2",
            "store._settle_task_group_execution:enter#0",
        ),
    ),
)


def is_known(violation: Violation) -> bool:
    return any(known.matches(violation) for known in KNOWN_FAILURES)
