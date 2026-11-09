"""A grouped cancellation settles exactly like the plain cancellation it carries.

Cleanup that receives the run's cancellation and fails too raises the
``CancelledError`` grouped with its own error. Without an interrupt request that
is still the caller cancelling the run, so the session is abandoned, for
recovery to continue, exactly as a plain cancellation at the same point leaves
it, never interrupted as if an operator had asked. The points reach the exit
handlers of run setup, the run body, a tool round, a resume, an approval
continuation and a task-backed run with and without a task group.
"""

from __future__ import annotations

import pytest
from tests.fault_injection.explore import SCENARIOS, run_point
from tests.fault_injection.harness import FaultKind

pytestmark = pytest.mark.fault_injection

_POINTS = (
    ("run", "store.replace_initial_transcript_messages:exit#0"),
    ("run", "store.append_transcript_messages:exit#0"),
    ("tool-round", "store.deliver_queued_session_messages:exit#0"),
    ("resume", "store.prepare_model_completion_stage:exit#0"),
    ("approval", "store.prepare_model_completion_stage:exit#0"),
    ("task", "store.prepare_model_completion_stage:exit#0"),
    ("task-group", "store.deliver_queued_session_messages:exit#0"),
)
# Cancelling a task worker leaves its running handler to finish, so a plain
# cancellation there completes the session. The grouped fault also cancels the
# handler, which has no plain counterpart in these scenarios.
_HANDLER_SCENARIOS = frozenset({"task", "task-group"})
_ABANDONED = ("interrupted", "session.interrupted", "abandoned")


@pytest.mark.parametrize(("scenario", "point"), _POINTS, ids=[f"{s}@{p}" for s, p in _POINTS])
def test_grouped_cancellation_settles_like_plain_cancellation(scenario: str, point: str) -> None:
    grouped = run_point(SCENARIOS[scenario], FaultKind.GROUPED_CANCEL, point)

    assert grouped.reached
    assert grouped.settlement[:3] == _ABANDONED
    assert not grouped.violations, [violation.describe() for violation in grouped.violations]
    if scenario not in _HANDLER_SCENARIOS:
        plain = run_point(SCENARIOS[scenario], FaultKind.CANCEL, point)
        assert plain.reached and not plain.violations
        assert grouped.settlement == plain.settlement
