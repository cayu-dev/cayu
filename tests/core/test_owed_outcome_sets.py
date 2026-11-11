"""Sets of terminal outcomes come from the decision's outcomes, not from a hand list.

These tests enumerate ``InvocationTerminalOutcome``, so every outcome that
fences a lost worker's run is covered without maintaining a separate list.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from cayu.runtime._incomplete_session_recovery import IncompleteSessionRecovery
from cayu.sessions._invocation_terminal_decision import InvocationTerminalOutcome
from cayu.sessions._paired_writes import OUTCOME_SESSION_STATUS
from cayu.sessions.records import Session, SessionStatus
from cayu.tasks import worker
from cayu.tasks.records import Task, TaskStatus


def _session(status: SessionStatus) -> Session:
    return Session.model_construct(status=status)


def _task(*, cancellation_requested: bool) -> Task:
    # The task of a lost worker is still claimed or running: only there does the
    # cancellation text mean a fence.
    return Task.model_construct(
        status=TaskStatus.RUNNING,
        status_reason="cancellation_requested" if cancellation_requested else None,
    )


@pytest.mark.parametrize("outcome", list(InvocationTerminalOutcome))
def test_every_outcome_fences_a_lost_worker_with_a_pending_cancellation(
    outcome: InvocationTerminalOutcome,
) -> None:
    session = _session(OUTCOME_SESSION_STATUS[outcome])

    assert worker._outcome_fences_lost_worker_run(session, _task(cancellation_requested=True))
    assert worker._outcome_fences_lost_worker_run(session, _task(cancellation_requested=False)) is (
        outcome is InvocationTerminalOutcome.INTERRUPTED
    )


@pytest.mark.parametrize(
    "status", sorted(set(SessionStatus) - set(OUTCOME_SESSION_STATUS.values()), key=str)
)
def test_a_live_session_does_not_fence_a_lost_worker(status: SessionStatus) -> None:
    assert not worker._outcome_fences_lost_worker_run(
        _session(status), _task(cancellation_requested=True)
    )


def test_the_committed_outcome_sweep_lists_no_outcome_status_by_hand() -> None:
    source = textwrap.dedent(
        inspect.getsource(IncompleteSessionRecovery.recover_committed_outcomes)
    )
    outcome_names = {status.name for status in OUTCOME_SESSION_STATUS.values()}
    listed = [
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "SessionStatus"
        and node.attr in outcome_names
    ]
    assert not listed, f"Iterate OUTCOME_SESSION_STATUS instead of listing {listed}."
