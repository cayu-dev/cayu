"""Opt-in sharing of one factory allocation across related sessions.

By default every session gets its own allocation from an environment factory:
its own Docker container, its own brokered sandbox. An allocation scope lets
related sessions use one allocation instead:

- ``session`` (the default): one allocation per session, as before.
- ``parent``: a session and its direct children share one allocation, so the
  steps of one workflow run, or a parent agent and its subagents, work in the
  same sandbox. A subagent of a session that is itself attached, such as a
  workflow step's subagent, joins that session's allocation.
- ``causal_budget``: every session in one causal budget tree shares one
  allocation, however deep the tree.

The application sets the scope on the environment registration
(``CayuApp.register_environment_factory(..., allocation_scope=...)``), and a
workflow can set it per step with ``cayu.workflows.StepRunOptions.allocation_scope``.
Session and run metadata can't change it, so a model can't opt a session in or
out. The runtime resolves the scope once, before it calls the factory, so every
factory gets it without changing.

A workflow run records the allocations its steps share in its checkpoint.
``CayuApp.list_shared_allocations`` reads those records and
``CayuApp.release_shared_allocation`` releases one by hand, for an operator
cleaning up after a run that crashed and won't be resumed. Their results are
the types below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal


class AllocationSharingScope(StrEnum):
    """Which sessions share one environment factory allocation."""

    SESSION = "session"
    PARENT = "parent"
    CAUSAL_BUDGET = "causal_budget"


def allocation_sharing_scope(value: object) -> AllocationSharingScope:
    """Return ``value`` as an ``AllocationSharingScope``, refusing anything else."""

    if isinstance(value, AllocationSharingScope):
        return value
    if type(value) is not str:
        raise TypeError("allocation_scope must be an AllocationSharingScope or its string value.")
    try:
        return AllocationSharingScope(value)
    except ValueError:
        choices = ", ".join(scope.value for scope in AllocationSharingScope)
        raise ValueError(f"allocation_scope must be one of: {choices}.") from None


SharedAllocationState = Literal["creating", "live", "orphan", "reaping"]
"""Where a recorded allocation is.

``creating``: recorded, and the factory hasn't returned (or the process died
first). ``live``: created, and the run can reconnect to it if the factory
supports that. ``orphan``: left behind by an attempt that couldn't reconnect,
or by a failed creation; no run will use it again. ``reaping``: fenced for
disposal by recovery or by an earlier release; nothing reattaches to it.
"""

SharedAllocationRunState = Literal["running", "stopped", "ended"]
"""What the owning workflow run looks like from here.

``running``: this process runs it or holds the allocation, one of its step
sessions is running under a live owner, or it journaled something recently.
``stopped``: none of that, and its newest attempt hasn't completed; it may
have crashed, or it may be resumed later. ``ended``: its newest attempt
journaled ``workflow.completed``, and recovery reaps what's left.
"""


class SharedAllocationReleaseMode(StrEnum):
    """What a hand release does with the allocation's work."""

    #: Reconnect, copy the work back to the source workspace through the
    #: factory's binding (as a completed run would), then dispose of it.
    COPY_BACK = "copy_back"
    #: Dispose of it without copying anything back, as recovery does.
    DISCARD = "discard"


@dataclass(frozen=True)
class SharedAllocation:
    """One recorded shared allocation, without its reconnect or binding state."""

    allocation_id: str
    workflow_run_id: str
    workflow_name: str | None
    environment_name: str
    agent_name: str
    scope: str
    scope_id: str
    state: SharedAllocationState
    reconnectable: bool
    attempt_id: str
    claim_generation: int
    created_at: datetime | None
    run_state: SharedAllocationRunState
    run_state_reason: str | None = None
    #: The factory result's non-secret metadata, with the application's
    #: secrets redacted. Reconnect metadata and binding state are never included.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SharedAllocationPage:
    """One page of recorded shared allocations.

    A page covers a bounded number of workflow runs, so it can hold fewer
    allocations than that (or none) and still have a ``next_cursor``.
    """

    allocations: tuple[SharedAllocation, ...]
    next_cursor: str | None = None


@dataclass(frozen=True)
class SharedAllocationRelease:
    """What a hand release did."""

    allocation: SharedAllocation
    mode: SharedAllocationReleaseMode
    forced: bool
    message: str


class SharedAllocationReleaseRefused(RuntimeError):
    """Raised before a shared allocation is touched, when it can't be released as asked.

    ``reason`` is one of ``owner_running``, ``held_here``, ``not_copyable``,
    ``environment_not_registered``, ``changed`` or ``not_disposable``.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
