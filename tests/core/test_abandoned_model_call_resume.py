from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from decimal import Decimal
from uuid import uuid4

import pytest
from tests.core.test_session_execution_presence import _consume
from tests.runtime.runtime_provider_support import VersionedFakeProvider

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ModelStreamEvent,
    PostgresBudgetLedger,
    PostgresSessionStore,
    ResumeRequest,
    RunRequest,
    SessionExecutionConfig,
    SessionStatus,
    SQLiteBudgetLedger,
    SQLiteSessionStore,
    Tool,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.build_provenance import (
    RUNTIME_BUILD_PROVENANCE_ENV,
    RuntimeBuildArtifactKind,
    RuntimeBuildProvenance,
    RuntimeBuildProvenanceOrigin,
    current_runtime_build_provenance,
)
from cayu.runtime._run_limits import UNKNOWN_POST_DISPATCH_BUDGET_REASON
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.storage.migrations import SchemaMode


class _LostProvider(VersionedFakeProvider):
    """Same provider identity in both processes; the first process dies mid-call.

    With ``CAYU_MODEL_CRASH_TEST_AFTER_TOOL``, its first call asks for a lookup and
    the process dies in the next call, after a published tool round of the same turn.
    """

    calls = 0

    async def stream(self, request):
        if os.environ.get("CAYU_MODEL_CRASH_TEST_CRASH") == "1":
            _LostProvider.calls += 1
            if os.environ.get("CAYU_MODEL_CRASH_TEST_AFTER_TOOL") == "1" and self.calls == 1:
                yield ModelStreamEvent.tool_call(id="lookup", name="lookup", arguments={})
                yield ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                )
                return
            os._exit(137)
        if os.environ.get("CAYU_MODEL_CRASH_TEST_CRASH") == "block":
            print("dispatched", flush=True)
            await asyncio.Event().wait()
        async for event in super().stream(request):
            yield event


class _Lookup(Tool):
    def __init__(self):
        self.spec = ToolSpec(
            name="lookup",
            description="Look up the order.",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.NONE,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:model-crash-lookup", behavior_version="1", implementation_version="1"
            ),
        )

    async def run(self, context, arguments):
        return ToolResult(content="Order O1 is lost.")


def _app(store, ledger, provider):
    app = CayuApp(
        session_store=store,
        budget_ledger=ledger,
        enable_logging=False,
        session_execution=SessionExecutionConfig(heartbeat_interval_seconds=0.05, lease_seconds=60),
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=PriceBook(
                        prices=(
                            ModelPrice.fixed(
                                provider_name=provider.name,
                                model="fake-model",
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
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[_Lookup()])
    return app


def _open(backend, location):
    if backend == "sqlite":
        return (
            SQLiteSessionStore(location + ".sessions"),
            SQLiteBudgetLedger(location + ".budgets", reservation_ttl_seconds=None),
        )
    return (
        PostgresSessionStore(location, min_size=1, max_size=2, schema_mode=SchemaMode.CREATE),
        PostgresBudgetLedger(
            location,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            reservation_ttl_seconds=None,
        ),
    )


def _run_child():
    async def crash():
        store, ledger = _open(
            os.environ["CAYU_MODEL_CRASH_TEST_BACKEND"], os.environ["CAYU_MODEL_CRASH_TEST_STORE"]
        )
        app = _app(store, ledger, _LostProvider([]))
        await _consume(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=os.environ["CAYU_MODEL_CRASH_TEST_SESSION"],
                    messages=[Message.text("user", "My order never arrived.")],
                )
            )
        )
        raise AssertionError("The provider fixture did not stop the process.")

    asyncio.run(crash())


def _build(marker: str) -> str:
    return RuntimeBuildProvenance.from_artifact_digest(
        origin=RuntimeBuildProvenanceOrigin.EXPLICIT_MANIFEST,
        artifact_kind=RuntimeBuildArtifactKind.OTHER,
        artifact_digest=marker * 64,
        source_revision=f"revision-{marker}",
    ).model_dump_json()


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("upgraded", [False, True])
@pytest.mark.parametrize("after_tool", [False, True])
def test_resume_recovers_process_death_during_model_call(
    backend, upgraded, after_tool, request, tmp_path, monkeypatch
):
    # With ``upgraded``, the process that dies runs one Cayu build and the one that
    # takes over runs another, as when a deploy restarts the application.
    monkeypatch.setenv(RUNTIME_BUILD_PROVENANCE_ENV, _build("a"))
    current_runtime_build_provenance.cache_clear()

    async def scenario():
        location = (
            str(tmp_path / "model-crash")
            if backend == "sqlite"
            else request.getfixturevalue("postgres_dsn")
        )
        sid = "model-crash-" + uuid4().hex
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from tests.core.test_abandoned_model_call_resume import _run_child; _run_child()",
            ],
            env={
                **os.environ,
                "CAYU_MODEL_CRASH_TEST_BACKEND": backend,
                "CAYU_MODEL_CRASH_TEST_STORE": location,
                "CAYU_MODEL_CRASH_TEST_SESSION": sid,
                "CAYU_MODEL_CRASH_TEST_CRASH": "1",
                RUNTIME_BUILD_PROVENANCE_ENV: _build("a"),
                "CAYU_MODEL_CRASH_TEST_AFTER_TOOL": "1" if after_tool else "0",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.to_thread(child.communicate), 30)
            assert child.returncode == 137, (stdout, stderr)
        finally:
            if child.poll() is None:
                child.kill()
                await asyncio.to_thread(child.wait)

        if upgraded:
            monkeypatch.setenv(RUNTIME_BUILD_PROVENANCE_ENV, _build("b"))
            current_runtime_build_provenance.cache_clear()
        store, ledger = _open(backend, location)
        try:
            original = await store.load(sid)
            assert original.status is SessionStatus.RUNNING
            active = await store.load_active_model_completion_stage(sid)
            assert active is not None and active.stage.state == "in_flight"
            app = _app(
                store,
                ledger,
                _LostProvider(
                    [
                        ModelStreamEvent.text_delta("A replacement is on its way."),
                        ModelStreamEvent.completed(
                            {
                                "finish_reason": "stop",
                                "usage": {"input_tokens": 1, "output_tokens": 1},
                            }
                        ),
                    ]
                ),
            )
            assert (await app.inspect_session_execution(sid)).state == "owner_lost"

            events = await _consume(
                app.resume(
                    ResumeRequest(session_id=sid, messages=[Message.text("user", "Any update?")])
                )
            )

            current = await store.load(sid)
            assert current.status is SessionStatus.COMPLETED
            assert current.run_epoch > original.run_epoch
            stored = await store.load_events(sid)
            [interrupted] = [e for e in stored if e.type is EventType.SESSION_INTERRUPTED]
            assert interrupted.payload["reason"] == "continuation_recovered_abandoned_execution"
            if upgraded:
                # A lost call from another build is terminalized from its durable record.
                assert interrupted.payload["terminalization_only"] is True
            else:
                assert any(
                    event.type is EventType.SESSION_RUN_FENCED
                    and event.payload["reason"] == "continuation_recovered_abandoned_execution"
                    for event in events
                )
            settlement = await store.load_model_completion_stage_settlement(
                sid, active.stage.stage_id
            )
            assert settlement is not None
            assert settlement.disposition.value == "provider_effect_outcome_unknown"
            # The lost call may have incurred usage, so its reservation is charged in full.
            lost = [await ledger.load_reservation(rid) for rid in active.stage.reservation_ids]
            assert lost and all(record.status == "reconciled" for record in lost)
            assert all(record.reason == UNKNOWN_POST_DISPATCH_BUDGET_REASON for record in lost)
            assert all(record.actual_amount == record.reserved_amount for record in lost)
            adoptions = [
                event
                for event in stored
                if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
                and event.payload["policy_identity"] == "cayu:runtime-upgrade-adoption:v1"
            ]
            # The lost call is settled from its record; the next turn adopts the build.
            assert len(adoptions) == (1 if upgraded else 0)
        finally:
            await store.close()
            await ledger.close()

    try:
        asyncio.run(scenario())
    finally:
        current_runtime_build_provenance.cache_clear()
