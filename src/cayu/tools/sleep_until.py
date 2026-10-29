from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from cayu.approvals.user_input import session_wake_result_text
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.display import ToolDisplay

SLEEP_UNTIL_TOOL_NAME = "sleep_until"
DEFAULT_MAX_SLEEP = timedelta(days=30)


@dataclass(frozen=True)
class SleepUntilPlan:
    """One validated ``sleep_until`` request resolved against the runtime clock."""

    requested: dict[str, Any]
    deadline: datetime
    resolved_at: datetime

    @property
    def reached(self) -> bool:
        return self.deadline <= self.resolved_at


class SleepUntilTool(Tool):
    """Built-in tool that parks the session until a deadline, then continues it.

    Opt-in: register it explicitly on the agents that may schedule their own
    continuation; the application needs a task store that supports scheduling.
    When the model calls it with a future deadline, the runtime intercepts the
    call before execution, parks it under its own ``pending_wake`` pause, creates a
    durable wake task for the deadline, and interrupts the session. No process,
    worker slot or timer stays alive while it sleeps. ``run_task_worker`` runs
    the wake task when it is due; this call then receives a timer result and the
    same session continues.

    The model passes exactly one of ``duration_seconds`` or ``until`` (an ISO 8601
    timestamp with a UTC offset). The runtime clock turns it into one UTC deadline
    once. Deadlines further away than ``max_sleep`` are refused. A deadline that
    has already passed returns at once without pausing.

    In a session a task owns, the task's worker hands the task off parked on the
    wake: no other continuation claims it, and the wake continues the session under
    the same task. The tool pauses only top-level sessions that no task owns or
    that a task worker runs, in a round where no other tool waits for approval;
    anywhere else the call returns an error.
    """

    # Decoupled markers the runtime checks (via getattr) to recognize a pausing tool
    # without importing this class — cayu.tools depends on cayu.runtime, not the reverse.
    pauses_session = True
    pause_kind = "wake"

    spec = ToolSpec(
        display=ToolDisplay(verb="Slept", noun="timer"),
        name=SLEEP_UNTIL_TOOL_NAME,
        description=(
            "Pause this session and continue it later. Pass exactly one of "
            "duration_seconds (how long to wait) or until (an ISO 8601 timestamp with "
            "a timezone offset, for example 2027-01-15T12:00:00Z). The session stops "
            "using resources while it waits; when the time comes this call returns and "
            "you continue from here."
        ),
        # A pause must not run alongside other tools in a concurrent batch.
        parallel_safe=False,
        # ``effect`` classifies replaying ``run``, not authority. The runtime, not
        # ``run``, schedules the wake before execution; ``run`` only reports a
        # refusal or an already-passed deadline and touches nothing, so replaying
        # it after a crash is safe. Gate ``sleep_until`` by name in a ToolPolicy.
        effect=ToolEffect.NONE,
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "duration_seconds": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "How many seconds to wait from now.",
                },
                "until": {
                    "type": "string",
                    "minLength": 1,
                    "description": "When to continue, as an ISO 8601 timestamp with an offset.",
                },
            },
        },
    )

    def __init__(
        self,
        spec: ToolSpec | None = None,
        *,
        max_sleep: timedelta = DEFAULT_MAX_SLEEP,
    ) -> None:
        if type(max_sleep) is not timedelta or max_sleep <= timedelta(0):
            raise ValueError("max_sleep must be a positive timedelta.")
        super().__init__(type(self).spec if spec is None else spec)
        self.max_sleep = max_sleep

    def _execution_profile_material(self) -> dict[str, object]:
        return {"max_sleep_seconds": self.max_sleep.total_seconds()}

    def plan_wake(self, arguments: dict[str, Any], *, now: datetime) -> SleepUntilPlan:
        """Validate one call and resolve its deadline; raise ``ValueError`` if refused."""

        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware.")
        now = now.astimezone(UTC)
        forms = [name for name in ("duration_seconds", "until") if arguments.get(name) is not None]
        unexpected = sorted(set(arguments) - {"duration_seconds", "until"})
        if unexpected:
            raise ValueError(f"sleep_until does not accept {', '.join(unexpected)}.")
        if len(forms) != 1:
            raise ValueError("sleep_until needs exactly one of duration_seconds or until.")
        if forms[0] == "duration_seconds":
            duration = arguments["duration_seconds"]
            if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("duration_seconds must be a finite number greater than 0.")
            try:
                deadline = now + timedelta(seconds=duration)
            except OverflowError:
                raise ValueError("duration_seconds is too large.") from None
            requested: dict[str, Any] = {"duration_seconds": duration}
        else:
            until = arguments["until"]
            if type(until) is not str:
                raise ValueError("until must be an ISO 8601 timestamp string.")
            try:
                parsed = datetime.fromisoformat(until.strip())
            except ValueError:
                raise ValueError("until must be an ISO 8601 timestamp.") from None
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("until must include a timezone offset, for example Z.")
            try:
                deadline = parsed.astimezone(UTC)
            except OverflowError:
                raise ValueError("until is out of range.") from None
            requested = {"until": until}
        if deadline - now > self.max_sleep:
            raise ValueError(
                f"sleep_until can wait at most {_describe(self.max_sleep)}; choose an earlier time."
            )
        return SleepUntilPlan(requested=requested, deadline=deadline, resolved_at=now)

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        try:
            # The application's clock, the same one that decided this call does
            # not pause; the wall clock only when called outside the runtime.
            now = ctx._runtime_now() or datetime.now(UTC)
            plan = self.plan_wake(args, now=now)
        except ValueError as exc:
            return ToolResult(content=f"Invalid sleep_until arguments: {exc}", is_error=True)
        if plan.reached:
            return ToolResult(
                content=session_wake_result_text(
                    wake_id=None,
                    requested=plan.requested,
                    deadline=plan.deadline,
                    resolved_at=plan.resolved_at,
                    status="deadline_already_reached",
                )
            )
        # A valid future deadline reaches direct execution only when the runtime
        # could not pause this round for it.
        return ToolResult(
            content=(
                "sleep_until did not pause: it can pause only a top-level session that "
                "no task owns or that a task worker runs, in a round where no tool "
                "needed approval, including sleep_until itself. Continue without "
                "waiting, or call sleep_until again in a later round without other tools."
            ),
            is_error=True,
        )


def _describe(value: timedelta) -> str:
    seconds = value.total_seconds()
    if seconds % 86400 == 0:
        return f"{int(seconds // 86400)} days"
    return f"{seconds:g} seconds"
