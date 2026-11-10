"""Observe who settles a recovered invocation's exit, and its terminal hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.observability.hooks import RuntimeHook
from cayu.runtime._recovery_admission import RecoveryAdmission
from cayu.runtime._run_settlement import RunSettlement
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity

if TYPE_CHECKING:
    import pytest


class InterruptionHook(RuntimeHook):
    execution_profile_identity = ExecutionProfileBehaviorIdentity(
        name="tests:recovered-owner-hook", behavior_version="1", implementation_version="1"
    )

    def __init__(self) -> None:
        self.calls = 0

    async def after_session_interrupted(self, context):
        self.calls += 1


def observe_decisions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Which ledger decided each abandonment: a recovered owner's, or another."""

    recovered: set[int] = set()
    decided: list[str] = []
    ledger = RecoveryAdmission.ledger
    decide = RunSettlement._decide_abandoned

    def owner_ledger(self, context):
        created = ledger(self, context)
        recovered.add(id(created))
        return created

    async def recorded(self, run_ledger):
        decided.append("recovered" if id(run_ledger) in recovered else "run")
        await decide(self, run_ledger)

    monkeypatch.setattr(RecoveryAdmission, "ledger", owner_ledger)
    monkeypatch.setattr(RunSettlement, "_decide_abandoned", recorded)
    return decided
