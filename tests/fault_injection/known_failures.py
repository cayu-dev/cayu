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
    "task-completion-cancel": Finding(
        "#3334 (same window, grouped cancellation)",
        "Grouped cancellation between interaction completion and task completion "
        "leaves the task RUNNING under a completed session",
    ),
    "approval-grouped-continuation": Finding(
        "#3390 (approval analogue of #3330)",
        "Grouped cancellation right after an approval continuation commits its status "
        "transition leaves the session RUNNING",
    ),
    "grouped-model-stage": Finding(
        "#3390",
        "Grouped cancellation right after a model completion stage commits leaves a "
        "session that cannot continue after recovery",
    ),
    "trailing-fence-cancel": Finding(
        "#3390 (same end state as #3306)",
        "Cancellation during the trailing run-fence release of a completed turn leaves "
        "the session fenced forever",
    ),
    "trailing-fence-crash": Finding(
        "#3390 (same end state as #3306)",
        "A crash between completing the interaction and releasing the invocation leaves "
        "the completed session fenced forever",
    ),
    "task-claim-cancel": Finding(
        "#3390",
        "Cancelling the worker while the execution-start mark is in flight leaves the "
        "claim with its pending entry settlement until the next worker recovers it",
    ),
    "task-crash-terminalize": Finding(
        "#3390 (crash analogue of #3334)",
        "A crash after the session completed, before the task is terminalized, leaves "
        "the task RUNNING forever",
    ),
    "group-crash-conflict": Finding(
        "#3390 (related #3245)",
        "After a worker dies mid-run its unsettled group execution makes the "
        "continuation fail with TaskGroupConflict, so the task stays RUNNING",
    ),
    "task-session-grouped-recovery": Finding(
        "#3390",
        "Grouped cancellation of a task-backed run during its model steps or tool round "
        "leaves the session RUNNING with a lost owner, and recovery fails with "
        "TaskClaimLost because the task was already settled",
    ),
    "group-crash-after-complete": Finding(
        "#3390",
        "A crash after the member completed, before its execution is settled, leaves "
        "the group obligation unsettled forever",
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
        "task-completion-cancel",
        "task",
        FaultKind.GROUPED_CANCEL,
        (
            (Invariant.TASK_SESSION_DISAGREE, Stage.LIVE),
            (Invariant.TASK_SESSION_DISAGREE, Stage.AFTER_SHUTDOWN),
            (Invariant.TASK_SESSION_DISAGREE, Stage.RECOVERED),
        ),
        (
            "store.claim_persisted_event_side_effect:enter#3",
            "store.claim_persisted_event_side_effect:exit#3",
            "store.mark_persisted_event_side_effect_delivered:enter#16",
            "store.mark_persisted_event_side_effect_delivered:exit#16",
            "store.query_events:enter#11",
            "store.load_state:enter#19",
            "store.terminalize_task:enter#0",
        ),
    ),
    _known(
        "task-completion-cancel",
        "task-group",
        FaultKind.GROUPED_CANCEL,
        (
            (Invariant.TASK_SESSION_DISAGREE, Stage.LIVE),
            (Invariant.TASK_SESSION_DISAGREE, Stage.AFTER_SHUTDOWN),
            (Invariant.TASK_SESSION_DISAGREE, Stage.RECOVERED),
        ),
        (
            "store.claim_persisted_event_side_effect:enter#3",
            "store.claim_persisted_event_side_effect:exit#3",
            "store.mark_persisted_event_side_effect_delivered:enter#16",
            "store.mark_persisted_event_side_effect_delivered:exit#16",
            "store.query_events:enter#11",
            "store.load_state:enter#19",
            "store.terminalize_task:enter#0",
        ),
    ),
    _known(
        "approval-grouped-continuation",
        "approval",
        FaultKind.GROUPED_CANCEL,
        (
            (Invariant.SESSION_UNSETTLED, Stage.LIVE),
            (Invariant.SESSION_UNSETTLED, Stage.AFTER_SHUTDOWN),
        ),
        ("store.transition_status_and_checkpoint:exit#0",),
    ),
    _known(
        "grouped-model-stage",
        "approval",
        FaultKind.GROUPED_CANCEL,
        ((Invariant.NOT_RESUMABLE, Stage.LIVENESS),),
        (
            "store.complete_model_completion_stage:exit#0",
            "store.promote_model_completion_stage:enter#0",
        ),
    ),
    _known(
        "grouped-model-stage",
        "resume",
        FaultKind.GROUPED_CANCEL,
        ((Invariant.NOT_RESUMABLE, Stage.LIVENESS),),
        (
            "store.complete_model_completion_stage:exit#0",
            "store.promote_model_completion_stage:enter#0",
            "store.complete_model_completion_stage:exit#1",
            "store.promote_model_completion_stage:enter#1",
        ),
    ),
    _known(
        "grouped-model-stage",
        "run",
        FaultKind.GROUPED_CANCEL,
        ((Invariant.NOT_RESUMABLE, Stage.LIVENESS),),
        (
            "store.complete_model_completion_stage:exit#0",
            "store.promote_model_completion_stage:enter#0",
        ),
    ),
    _known(
        "grouped-model-stage",
        "tool-round",
        FaultKind.GROUPED_CANCEL,
        ((Invariant.NOT_RESUMABLE, Stage.LIVENESS),),
        (
            "store.complete_model_completion_stage:exit#0",
            "store.promote_model_completion_stage:enter#0",
            "store.complete_model_completion_stage:exit#1",
            "store.promote_model_completion_stage:enter#1",
        ),
    ),
    _known(
        "trailing-fence-cancel",
        "approval",
        FaultKind.CANCEL,
        ((Invariant.NOT_RESUMABLE, Stage.LIVENESS),),
        (
            "store.load:enter#9",
            "store.load_checkpoint:enter#17",
        ),
    ),
    _known(
        "trailing-fence-cancel",
        "approval",
        FaultKind.GROUPED_CANCEL,
        ((Invariant.NOT_RESUMABLE, Stage.LIVENESS),),
        (
            "store.load:enter#9",
            "store.load_checkpoint:enter#17",
            "store.load:enter#10",
            "store.load_checkpoint:enter#18",
            "store.load_invocation_settlement_transition:enter#0",
            "store.load_interaction_transition_receipt:enter#0",
            "store.release_session_invocation:enter#0",
        ),
    ),
    _known(
        "trailing-fence-crash",
        "approval",
        FaultKind.CRASH,
        ((Invariant.NOT_RESUMABLE, Stage.LIVENESS),),
        (
            "store.claim_persisted_event_side_effect:enter#4",
            "store.mark_persisted_event_side_effect_delivered:enter#11",
            "store.append_event_and_claim_side_effect:enter#8",
            "store.mark_persisted_event_side_effect_delivered:enter#12",
            "store.append_event_and_claim_side_effect:enter#9",
            "store.mark_persisted_event_side_effect_delivered:enter#13",
            "store.transform_checkpoint:enter#2",
            "store.release_session_invocation:enter#0",
        ),
    ),
    _known(
        "trailing-fence-crash",
        "resume",
        FaultKind.CRASH,
        ((Invariant.NOT_RESUMABLE, Stage.LIVENESS),),
        (
            "store.claim_persisted_event_side_effect:enter#6",
            "store.mark_persisted_event_side_effect_delivered:enter#16",
            "store.append_event_and_claim_side_effect:enter#12",
            "store.mark_persisted_event_side_effect_delivered:enter#17",
            "store.append_event_and_claim_side_effect:enter#13",
            "store.mark_persisted_event_side_effect_delivered:enter#18",
            "store.transform_checkpoint:enter#5",
            "store.release_session_invocation:enter#0",
        ),
    ),
    _known(
        "trailing-fence-crash",
        "run",
        FaultKind.CRASH,
        ((Invariant.NOT_RESUMABLE, Stage.LIVENESS),),
        (
            "store.claim_persisted_event_side_effect:enter#2",
            "store.mark_persisted_event_side_effect_delivered:enter#7",
            "store.append_event_and_claim_side_effect:enter#5",
            "store.mark_persisted_event_side_effect_delivered:enter#8",
            "store.append_event_and_claim_side_effect:enter#6",
            "store.mark_persisted_event_side_effect_delivered:enter#9",
            "store.release_session_invocation:enter#0",
        ),
    ),
    _known(
        "trailing-fence-crash",
        "tool-round",
        FaultKind.CRASH,
        ((Invariant.NOT_RESUMABLE, Stage.LIVENESS),),
        (
            "store.claim_persisted_event_side_effect:enter#3",
            "store.mark_persisted_event_side_effect_delivered:enter#15",
            "store.append_event_and_claim_side_effect:enter#12",
            "store.mark_persisted_event_side_effect_delivered:enter#16",
            "store.append_event_and_claim_side_effect:enter#13",
            "store.mark_persisted_event_side_effect_delivered:enter#17",
            "store.release_session_invocation:enter#0",
        ),
    ),
    _known(
        "task-claim-cancel",
        "task",
        FaultKind.CANCEL,
        ((Invariant.TASK_UNSETTLED, Stage.LIVE), (Invariant.TASK_UNSETTLED, Stage.AFTER_SHUTDOWN)),
        (
            "store.mark_claimed_task_execution_started:enter#0",
            "store.mark_claimed_task_execution_started:exit#0",
        ),
    ),
    _known(
        "task-claim-cancel",
        "task",
        FaultKind.GROUPED_CANCEL,
        ((Invariant.TASK_UNSETTLED, Stage.LIVE), (Invariant.TASK_UNSETTLED, Stage.AFTER_SHUTDOWN)),
        (
            "store.mark_claimed_task_execution_started:enter#0",
            "store.mark_claimed_task_execution_started:exit#0",
        ),
    ),
    _known(
        "task-claim-cancel",
        "task-group",
        FaultKind.CANCEL,
        ((Invariant.TASK_UNSETTLED, Stage.LIVE), (Invariant.TASK_UNSETTLED, Stage.AFTER_SHUTDOWN)),
        (
            "store.mark_claimed_task_execution_started:enter#0",
            "store.mark_claimed_task_execution_started:exit#0",
        ),
    ),
    _known(
        "task-claim-cancel",
        "task-group",
        FaultKind.GROUPED_CANCEL,
        ((Invariant.TASK_UNSETTLED, Stage.LIVE), (Invariant.TASK_UNSETTLED, Stage.AFTER_SHUTDOWN)),
        (
            "store.mark_claimed_task_execution_started:enter#0",
            "store.mark_claimed_task_execution_started:exit#0",
        ),
    ),
    _known(
        "task-crash-terminalize",
        "task",
        FaultKind.CRASH,
        (
            (Invariant.TASK_SESSION_DISAGREE, Stage.RECOVERED),
            (Invariant.TASK_UNSETTLED, Stage.TASK_RECOVERY),
            (Invariant.TASK_UNSETTLED, Stage.RECOVERED),
        ),
        (
            "store.claim_persisted_event_side_effect:enter#3",
            "store.mark_persisted_event_side_effect_delivered:enter#16",
            "store.terminalize_task:enter#0",
        ),
    ),
    _known(
        "task-crash-terminalize",
        "task-group",
        FaultKind.CRASH,
        (
            (Invariant.TASK_SESSION_DISAGREE, Stage.RECOVERED),
            (Invariant.TASK_UNSETTLED, Stage.TASK_RECOVERY),
            (Invariant.TASK_UNSETTLED, Stage.RECOVERED),
        ),
        (
            "store.claim_persisted_event_side_effect:enter#3",
            "store.mark_persisted_event_side_effect_delivered:enter#16",
            "store.terminalize_task:enter#0",
        ),
    ),
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
    _known(
        "task-session-grouped-recovery",
        "task",
        FaultKind.GROUPED_CANCEL,
        (
            (Invariant.EXECUTION_LEASE_HELD, Stage.LIVE),
            (Invariant.EXECUTION_LEASE_HELD, Stage.AFTER_SHUTDOWN),
            (Invariant.EXECUTION_LEASE_HELD, Stage.RECOVERED),
            (Invariant.SESSION_UNSETTLED, Stage.LIVE),
            (Invariant.SESSION_UNSETTLED, Stage.AFTER_SHUTDOWN),
            (Invariant.SESSION_UNSETTLED, Stage.RECOVERED),
        ),
        (
            "store._task_group_retains_execution:enter#1",
            "store._task_group_retains_execution:exit#1",
            "provider.request#0",
            "provider.tool_call#0",
            "provider.tool_call#1",
            "provider.completed#0",
            "store.load_state:enter#8",
            "store.append_event_and_claim_side_effect:enter#5",
            "store.append_event_and_claim_side_effect:exit#5",
            "store.mark_persisted_event_side_effect_delivered:enter#7",
            "store.mark_persisted_event_side_effect_delivered:exit#7",
            "store.load_state:enter#9",
            "store.append_event_and_claim_side_effect:enter#6",
            "store.append_event_and_claim_side_effect:exit#6",
            "store.mark_persisted_event_side_effect_delivered:enter#8",
            "store.mark_persisted_event_side_effect_delivered:exit#8",
            "store.transform_checkpoint:enter#3",
            "store.transform_checkpoint:exit#3",
            "store.transform_checkpoint:enter#4",
            "store.transform_checkpoint:exit#4",
            "store.append_event_and_claim_side_effect:enter#7",
            "store.append_event_and_claim_side_effect:exit#7",
            "store.mark_persisted_event_side_effect_delivered:enter#9",
            "store.mark_persisted_event_side_effect_delivered:exit#9",
            "store.load_state:enter#10",
            "store.append_event_and_claim_side_effect:enter#8",
            "store.append_event_and_claim_side_effect:exit#8",
            "store.mark_persisted_event_side_effect_delivered:enter#10",
            "store.mark_persisted_event_side_effect_delivered:exit#10",
            "store.load_state:enter#11",
            "provider.request#1",
            "provider.text_delta#0",
            "provider.completed#1",
        ),
    ),
    _known(
        "task-session-grouped-recovery",
        "task-group",
        FaultKind.GROUPED_CANCEL,
        (
            (Invariant.EXECUTION_LEASE_HELD, Stage.LIVE),
            (Invariant.EXECUTION_LEASE_HELD, Stage.AFTER_SHUTDOWN),
            (Invariant.EXECUTION_LEASE_HELD, Stage.RECOVERED),
            (Invariant.SESSION_UNSETTLED, Stage.LIVE),
            (Invariant.SESSION_UNSETTLED, Stage.AFTER_SHUTDOWN),
            (Invariant.SESSION_UNSETTLED, Stage.RECOVERED),
        ),
        (
            "store._task_group_retains_execution:enter#1",
            "store._task_group_retains_execution:exit#1",
            "store._observe_task_group_invocation:enter#0",
            "store._observe_task_group_invocation:exit#0",
            "provider.request#0",
            "provider.tool_call#0",
            "provider.tool_call#1",
            "provider.completed#0",
            "store.load_state:enter#8",
            "store.append_event_and_claim_side_effect:enter#5",
            "store.append_event_and_claim_side_effect:exit#5",
            "store.mark_persisted_event_side_effect_delivered:enter#7",
            "store.mark_persisted_event_side_effect_delivered:exit#7",
            "store.load_state:enter#9",
            "store.append_event_and_claim_side_effect:enter#6",
            "store.append_event_and_claim_side_effect:exit#6",
            "store.mark_persisted_event_side_effect_delivered:enter#8",
            "store.mark_persisted_event_side_effect_delivered:exit#8",
            "store.transform_checkpoint:enter#3",
            "store.transform_checkpoint:exit#3",
            "store.transform_checkpoint:enter#4",
            "store.transform_checkpoint:exit#4",
            "store.append_event_and_claim_side_effect:enter#7",
            "store.append_event_and_claim_side_effect:exit#7",
            "store.mark_persisted_event_side_effect_delivered:enter#9",
            "store.mark_persisted_event_side_effect_delivered:exit#9",
            "store.load_state:enter#10",
            "store.append_event_and_claim_side_effect:enter#8",
            "store.append_event_and_claim_side_effect:exit#8",
            "store.mark_persisted_event_side_effect_delivered:enter#10",
            "store.mark_persisted_event_side_effect_delivered:exit#10",
            "store.load_state:enter#11",
            "provider.request#1",
            "provider.text_delta#0",
            "provider.completed#1",
        ),
    ),
)


def is_known(violation: Violation) -> bool:
    return any(known.matches(violation) for known in KNOWN_FAILURES)
