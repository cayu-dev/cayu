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
"""

from __future__ import annotations

from enum import StrEnum


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
