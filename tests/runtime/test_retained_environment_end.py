"""Watching a retained environment owner until its environment is gone."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from cayu._workspace_mutation import WorkspaceMutationProcessFence
from cayu.runtime._retained_environment_end import RetainedEnvironmentEndWatches


class _Binding:
    def __init__(self) -> None:
        self.asked = 0
        self.ends: list[asyncio.Future[str]] = []

    def _retained_release_end(self, bound: object) -> Any:
        self.asked += 1
        end: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self.ends.append(end)
        return end


@dataclass(eq=False)
class _Owner:
    registered_environment: Any
    cleanup_started: bool = True
    cleanup_finished: bool = True
    cleanup_settlement_started: bool = False
    cleanup_release_safe: bool = True
    pending_finalize_failure_event: object = None
    pending_completion_marker_clear: object = None
    cleanup_settlement_deferred: bool = True
    cleanup_requires_finalize_retry: bool = False
    environment_end_reason: str | None = None
    settled: list[str] = field(default_factory=list)


def _owner(binding: _Binding) -> _Owner:
    return _Owner(
        registered_environment=SimpleNamespace(
            factory_backed=True,
            environment=SimpleNamespace(binding=binding),
            bound_workspace=object(),
            spec=SimpleNamespace(name="sandbox"),
            workspace_mutation_fence=WorkspaceMutationProcessFence().child_fence(),
        )
    )


def test_a_watch_stopped_with_its_event_loop_starts_again_in_the_next() -> None:
    binding = _Binding()
    owner = _owner(binding)
    setups: dict[str, Any] = {"session": owner}

    async def settle(session_id: str, settled_owner: Any) -> None:
        settled_owner.settled.append(settled_owner.environment_end_reason)
        del setups[session_id]

    async def clear(session_id: str) -> None:
        del session_id

    watches = RetainedEnvironmentEndWatches(
        setups, owner_type=_Owner, settle=settle, clear_completion_marker=clear
    )

    async def first_loop() -> None:
        watches.observe("session")
        await asyncio.sleep(0)

    async def second_loop() -> None:
        watches.observe("session")
        binding.ends[-1].set_result("sandbox destroyed")
        for _ in range(5):
            await asyncio.sleep(0)

    asyncio.run(first_loop())
    assert binding.asked == 1 and "session" in setups
    asyncio.run(second_loop())
    assert binding.asked == 2
    assert owner.settled == ["sandbox destroyed"] and "session" not in setups


def test_a_running_cleanup_attempt_is_not_overtaken() -> None:
    binding = _Binding()
    owner = _owner(binding)
    setups: dict[str, Any] = {"session": owner}
    settled: list[str] = []

    async def settle(session_id: str, settled_owner: Any) -> None:
        settled.append(session_id)
        del setups[session_id]

    async def clear(session_id: str) -> None:
        del session_id

    watches = RetainedEnvironmentEndWatches(
        setups, owner_type=_Owner, settle=settle, clear_completion_marker=clear
    )

    async def scenario() -> None:
        watches.observe("session")
        owner.cleanup_settlement_started = True
        binding.ends[-1].set_result("sandbox destroyed")
        for _ in range(5):
            await asyncio.sleep(0)
        assert settled == []
        owner.cleanup_settlement_started = False
        watches.observe("session")
        for _ in range(5):
            await asyncio.sleep(0)
        assert settled == ["session"]

    asyncio.run(scenario())


def test_a_refused_retirement_backs_off_instead_of_settling_again_at_once() -> None:
    binding = _Binding()
    owner = _owner(binding)
    setups: dict[str, Any] = {"session": owner}
    attempts: list[str] = []
    watches: RetainedEnvironmentEndWatches

    async def settle(session_id: str, settled_owner: Any) -> None:
        # Like abort_environment_setup: claim, refuse, release, signal.
        settled_owner.cleanup_settlement_started = True
        try:
            attempts.append(session_id)
            await asyncio.sleep(0)
        finally:
            settled_owner.cleanup_settlement_started = False
            watches.observe(session_id)

    async def clear(session_id: str) -> None:
        del session_id

    watches = RetainedEnvironmentEndWatches(
        setups, owner_type=_Owner, settle=settle, clear_completion_marker=clear
    )

    async def scenario() -> None:
        watches.observe("session")
        binding.ends[-1].set_result("sandbox destroyed")
        await asyncio.sleep(0.3)

    asyncio.run(scenario())
    assert attempts == ["session"]
