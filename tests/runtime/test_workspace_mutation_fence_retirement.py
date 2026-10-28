"""Retiring one environment's mutation fence scope once that environment is gone."""

from __future__ import annotations

import asyncio

import pytest

from cayu._workspace_mutation import (
    WorkspaceMutationProcessFence,
    WorkspaceMutationSettlementError,
    workspace_mutation_task_settlement_probe,
)


def test_retiring_a_scope_releases_a_waiter_blocked_on_its_stuck_mutation() -> None:
    async def scenario() -> None:
        registration = WorkspaceMutationProcessFence()
        gone = registration.child_fence()
        other = registration.child_fence()
        stuck = asyncio.create_task(asyncio.Event().wait())
        gone.fail_closed(workspace_mutation_task_settlement_probe(stuck))
        assert gone.has_uncertain_mutations() and not other.has_uncertain_mutations()
        with pytest.raises(WorkspaceMutationSettlementError):
            registration.require_available_nowait()

        waiter = asyncio.create_task(registration.wait_until_available())
        await asyncio.sleep(0.05)
        assert not waiter.done()

        assert other.retire_scope() == 0
        assert gone.retire_scope() == 1
        await asyncio.wait_for(waiter, 1)
        registration.require_available_nowait()
        assert not gone.has_uncertain_mutations()
        # The fenced task itself is left alone.
        assert not stuck.done()
        stuck.cancel()

    asyncio.run(scenario())


def test_retiring_a_scope_keeps_other_environments_fenced() -> None:
    async def scenario() -> None:
        registration = WorkspaceMutationProcessFence()
        gone = registration.child_fence()
        live = registration.child_fence()
        stuck = asyncio.create_task(asyncio.Event().wait())
        live_task = asyncio.create_task(asyncio.Event().wait())
        gone.fail_closed(workspace_mutation_task_settlement_probe(stuck))
        live.fail_closed(workspace_mutation_task_settlement_probe(live_task))

        assert gone.retire_scope() == 1
        with pytest.raises(WorkspaceMutationSettlementError):
            registration.require_available_nowait()
        assert live.has_uncertain_mutations()
        live_task.cancel()
        stuck.cancel()

    asyncio.run(scenario())


def test_only_a_child_scope_can_be_retired() -> None:
    with pytest.raises(RuntimeError, match="child"):
        WorkspaceMutationProcessFence().retire_scope()
