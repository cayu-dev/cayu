"""Say when a task's event budget, not its worker, decides how long it may run.

An event's budget is fixed when its first task is admitted, usually in another
process than the worker that claims it. A worker can't change it, but it can say
when the budget is shorter than its own ``execution_timeout_s``, since such a
handler can never use its full limit, and a cancellation can say which budget
ran out and how to declare a longer one.

Cayu 0.16.0 gave tasks created without ``retry_budget`` or ``retry_policy`` an
elapsed limit of 1,800 seconds. Later versions default to no elapsed limit, but
an event keeps the budget it was first admitted with, so events queued under
0.16.0 still carry it. A store records only the budget's value, so such an event
is recognized by that value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cayu.tasks.retry_budgets import TaskRetryBudget

if TYPE_CHECKING:
    from cayu.tasks.records import Task
    from cayu.tasks.retry_budgets import TaskRetryKeyState

# Logged under the worker's name, where operators already watch task warnings.
logger = logging.getLogger("cayu.tasks.worker")

#: The budget Cayu 0.16.0 gave tasks created without retry_budget or retry_policy.
RELEASE_0_16_0_DEFAULT_TASK_RETRY_BUDGET = TaskRetryBudget(
    max_attempts=3, max_elapsed_seconds=1800.0
)

_HOW_TO_DECLARE = (
    "To give these tasks more time, create them with "
    "TaskCreate(retry_budget=TaskRetryBudget(max_elapsed_seconds=...)), or "
    "TaskRetryBudget(max_elapsed_seconds=None) to keep only the attempt limit; "
    "a task with a TaskRetryPolicy takes the limit from its max_elapsed_seconds."
)


def _seconds(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,g}"


def has_0_16_0_default_budget(task: Task, state: TaskRetryKeyState) -> bool:
    """Whether the event carries the budget 0.16.0 gave tasks that declared none."""
    return task.retry_series is None and state.budget == RELEASE_0_16_0_DEFAULT_TASK_RETRY_BUDGET


_0_16_0_DEFAULT_NOTE = (
    "This matches the budget Cayu 0.16.0 gave tasks created without retry_budget or "
    "retry_policy (3 attempts, 1,800 seconds from first admission). Later versions set "
    "no elapsed limit by default, but an event keeps the budget it was first admitted "
    "with, so an undeclared task first queued under 0.16.0 keeps it."
)


def with_event_budget_expiry(payload: dict[str, Any], task: Task, state: TaskRetryKeyState) -> None:
    """Say in a cancellation or failure payload, in place, which event budget ran out."""
    limit = state.budget.max_elapsed_seconds
    if limit is None:
        return
    note = (
        f"The task's event budget allows {_seconds(limit)} seconds from the event's first "
        "admission, queue time included, and it ran out."
    )
    if has_0_16_0_default_budget(task, state):
        note = f"{note} {_0_16_0_DEFAULT_NOTE}"
    note = f"{note} {_HOW_TO_DECLARE}"
    message = payload.get("message")
    payload["message"] = note if not isinstance(message, str) else f"{message} {note}"
    payload["event_max_elapsed_seconds"] = limit


@dataclass(slots=True)
class EventBudgetCheck:
    """One worker's check of claimed tasks' event budgets against its own time limit.

    It logs a warning once per task type and never changes what runs: a
    declared budget always applies, even when it is shorter than the worker's
    limit.
    """

    execution_timeout_s: float | None
    warned_types: set[str] = field(default_factory=set)

    def conflict(self, task: Task, state: TaskRetryKeyState) -> str | None:
        """Describe how the event budget cuts this worker's limit short, or ``None``."""
        limit = state.budget.max_elapsed_seconds
        if limit is None:
            return None
        legacy_default = has_0_16_0_default_budget(task, state)
        if self.execution_timeout_s is None:
            if not legacy_default:
                # A declared budget is how a worker without its own limit is bounded.
                return None
            worker_limit = "sets no execution_timeout_s"
            cut_off = (
                f"so these handlers are cut off after {_seconds(limit)} seconds even though "
                "the worker sets no limit of its own"
            )
        elif self.execution_timeout_s <= limit:
            return None
        else:
            worker_limit = (
                f"allows handlers {_seconds(self.execution_timeout_s)} seconds "
                f"(execution_timeout_s={self.execution_timeout_s:g})"
            )
            cut_off = "so these handlers can't use the worker's full time limit"
        message = (
            f"Tasks of this type have an event budget of {_seconds(limit)} seconds from the "
            f"event's first admission, queue time included, but this worker {worker_limit}. "
            f"The worker cancels a handler when its event budget runs out, {cut_off}."
        )
        if legacy_default:
            message = f"{message} {_0_16_0_DEFAULT_NOTE}"
        return (
            f"{message} {_HOW_TO_DECLARE} Or set execution_timeout_s to "
            f"{_seconds(limit)} or less if that is enough."
        )

    def check(self, app: Any, task: Task, state: TaskRetryKeyState) -> None:
        """Log the conflict the first time this worker claims a task of its type."""
        if task.type in self.warned_types:
            return
        message = self.conflict(task, state)
        if message is None:
            return
        self.warned_types.add(task.type)
        logger.warning(
            "%s task_type=%s task_id=%s",
            message,
            app.redact_json(task.type),
            app.redact_json(task.id),
        )
