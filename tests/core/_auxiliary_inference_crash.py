"""Real process-loss fixture for managed auxiliary inference recovery."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    ExecutionProfileBehaviorIdentity,
    Message,
    RunRequest,
    ScriptedModelProvider,
    SQLiteBudgetLedger,
    SQLiteSessionStore,
)
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.tools.base import Tool, ToolSpec
from cayu.tools.inference import AuxiliaryInferencePolicy, InferenceLimits

BOUNDS = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=30)
SESSION_ID = "auxiliary-crash-recovery"


class CrashProvider(ScriptedModelProvider):
    def __init__(self, *, crash: bool, phase: str = "dispatch"):
        super().__init__(
            [
                ModelStreamEvent.tool_call(name="summarize", arguments={}, id="parent"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                        },
                    }
                ),
            ],
            name="auxiliary-crash-provider",
        )
        self.crash = crash
        self.phase = phase

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:auxiliary-crash-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request):
        if not self.crash:
            raise AssertionError("Recovery must never dispatch a provider")
        print("PROVIDER_DISPATCH", flush=True)
        if self.requests:
            # Abrupt process loss after the runtime's real dispatch receipt,
            # without any Python cancellation, finally block, or terminal event.
            if self.phase == "dispatch":
                os._exit(73)
            yield ModelStreamEvent.text_delta("summary")
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 2}})
            return
        async for event in super().stream(request):
            yield event


class Summarize(Tool):
    spec = ToolSpec(
        name="summarize",
        description="Managed nested inference",
        input_schema={"type": "object"},
        auxiliary_inference=AuxiliaryInferencePolicy(limits=BOUNDS, purposes=("tool.summary",)),
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:auxiliary-crash-tool",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def run(self, ctx, args):
        self.calls += 1
        await ctx.inference.invoke(
            ModelRequest(model="test-model", messages=[Message.text("user", "nested")]),
            purpose="tool.summary",
            limits=BOUNDS,
        )
        raise AssertionError("Crash fixture must not return from auxiliary inference")


def build_app(directory: Path, *, crash: bool, phase: str = "dispatch"):
    store = SQLiteSessionStore(directory / "sessions.sqlite")
    ledger = SQLiteBudgetLedger(directory / "budgets.sqlite", reservation_ttl_seconds=None)
    app = CayuApp(
        session_store=store,
        budget_ledger=ledger,
        enable_logging=False,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=1,
                    pricing=PriceBook(
                        prices=(
                            ModelPrice.fixed(
                                provider_name="auxiliary-crash-provider",
                                model="test-model",
                                input_per_million=1,
                                output_per_million=1,
                            ),
                        )
                    ),
                    reservation=BudgetReservation(max_input_tokens=1000, max_output_tokens=1000),
                ),
            )
        ),
    )
    provider = CrashProvider(crash=crash, phase=phase)
    tool = Summarize()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="test-model"), tools=[tool])
    if crash and phase == "terminal":
        original = app._run_limit_controller.reconcile_model_completion_settlements

        async def lose_process_after_terminal(event, **kwargs):
            if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED:
                os._exit(74)
            return await original(event, **kwargs)

        app._run_limit_controller.reconcile_model_completion_settlements = (
            lose_process_after_terminal
        )
    return app, provider, tool, store, ledger


async def crash_run(directory: Path, *, phase: str):
    app, _, _, store, ledger = build_app(directory, crash=True, phase=phase)
    try:
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=SESSION_ID,
                messages=[Message.text("user", "go")],
            )
        ):
            pass
    finally:
        await store.close()
        await ledger.close()
    raise AssertionError("Expected process loss during auxiliary dispatch")


if __name__ == "__main__":
    import sys

    asyncio.run(crash_run(Path(sys.argv[1]), phase=sys.argv[2]))
