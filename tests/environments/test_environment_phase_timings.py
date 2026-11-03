from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tests.core._workload_secret_support import FakeProvider, collect_events

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.environments._phase_timing import (
    MAX_PHASE_TIMINGS,
    EnvironmentPhaseTimer,
    environment_phase_step,
    observe_environment_phase,
)
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.bindings import BoundWorkspace, SyncBinding
from cayu.environments.lifecycle import (
    EnvironmentLifecyclePhase,
    EnvironmentLifecycleProgressStatus,
)
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runners.base import Runner
from cayu.sessions.requests import RunRequest
from cayu.workspaces.base import Workspace
from cayu.workspaces.local import LocalWorkspace

STARTED = EnvironmentLifecycleProgressStatus.STARTED
ADVANCED = EnvironmentLifecycleProgressStatus.ADVANCED
COMPLETED = EnvironmentLifecycleProgressStatus.COMPLETED
FAILED = EnvironmentLifecycleProgressStatus.FAILED


class _Clock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def now(self) -> datetime:
        return datetime(2026, 11, 2, 12, 0, tzinfo=UTC) + timedelta(seconds=self.seconds)

    def monotonic(self) -> float:
        return self.seconds


def _timer(clock: _Clock) -> EnvironmentPhaseTimer:
    return EnvironmentPhaseTimer(now=clock.now, monotonic=clock.monotonic)


def test_reports_become_nested_steps_with_start_times_and_durations() -> None:
    clock = _Clock()
    timer = _timer(clock)
    timer.observe(EnvironmentLifecyclePhase.TRANSFER, STARTED)
    clock.seconds = 1.0
    timer.observe(EnvironmentLifecyclePhase.ARCHIVE_PREPARATION, STARTED)
    clock.seconds = 3.5
    timer.observe(EnvironmentLifecyclePhase.ARCHIVE_PREPARATION, COMPLETED, bytes_=2048)
    timer.observe(EnvironmentLifecyclePhase.TRANSFER, ADVANCED, items=3)
    clock.seconds = 10.0
    timer.observe(EnvironmentLifecyclePhase.TRANSFER, COMPLETED, items=4, bytes_=100)
    # A completion nothing started records nothing.
    timer.observe(EnvironmentLifecyclePhase.COPY_BACK_PUBLICATION, COMPLETED)

    assert timer.payload() == [
        {
            "phase": "transfer",
            "started_at": "2026-11-02T12:00:00.000+00:00",
            "duration_ms": 10_000,
            "outcome": "completed",
            "depth": 0,
            "items": 4,
            "bytes": 100,
        },
        {
            "phase": "archive_preparation",
            "started_at": "2026-11-02T12:00:01.000+00:00",
            "duration_ms": 2_500,
            "outcome": "completed",
            "depth": 1,
            "bytes": 2048,
        },
    ]


def test_steps_record_failures_and_unfinished_work() -> None:
    clock = _Clock()
    timer = _timer(clock)
    with pytest.raises(RuntimeError), timer.step("git_baseline"):
        clock.seconds = 2.0
        raise RuntimeError("boom")
    timer.observe(EnvironmentLifecyclePhase.FINAL_SOURCE_OBSERVATION, STARTED)
    clock.seconds = 5.0
    timer.observe(EnvironmentLifecyclePhase.STAGING_ADMISSION, STARTED)
    timer.observe(EnvironmentLifecyclePhase.STAGING_ADMISSION, FAILED)

    assert [(row["phase"], row["outcome"], row["duration_ms"]) for row in timer.payload()] == [
        ("git_baseline", "failed", 2_000),
        ("final_source_observation", "unfinished", 3_000),
        ("staging_admission", "failed", 0),
    ]
    assert [row["phase"] for row in timer.payload(include_unfinished=False)] == [
        "git_baseline",
        "staging_admission",
    ]


def test_without_an_installed_timer_reports_and_steps_do_nothing() -> None:
    observe_environment_phase(EnvironmentLifecyclePhase.TRANSFER, STARTED, items_completed=1)
    with environment_phase_step("git_baseline"):
        pass

    timer = EnvironmentPhaseTimer()
    with timer.installed():
        observe_environment_phase(EnvironmentLifecyclePhase.TRANSFER, STARTED)
        observe_environment_phase(
            EnvironmentLifecyclePhase.TRANSFER, COMPLETED, items_completed=2, bytes_total=9
        )
        with environment_phase_step("workspace_setup_commands", when=False):
            pass
        with environment_phase_step("git_baseline"):
            pass
    observe_environment_phase(EnvironmentLifecyclePhase.SOURCE_OBSERVATION, STARTED)

    assert [(row["phase"], row.get("items")) for row in timer.payload()] == [
        ("transfer", 2),
        ("git_baseline", None),
    ]


def test_a_timer_keeps_the_first_steps_and_refuses_bad_names() -> None:
    timer = EnvironmentPhaseTimer()
    for _ in range(MAX_PHASE_TIMINGS + 5):
        with timer.step("setup_commands"):
            pass
    assert len(timer.payload()) == MAX_PHASE_TIMINGS
    with pytest.raises(ValueError), timer.step("Not A Phase"):
        pass


def _sync_app(source_root, binding: SyncBinding) -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]), default=True
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="sync"),
            workspace=LocalWorkspace(source_root, workspace_id="source"),
            binding=binding,
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return app


def _run(app: CayuApp, session_id: str) -> list[Event]:
    return asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
    )


def _timings(events: list[Event], event_type: EventType) -> list[dict[str, Any]]:
    (event,) = [item for item in events if item.type == event_type]
    return event.payload["phase_timings"]


def test_a_sync_binding_records_its_copy_in_and_copy_back_steps(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "input.txt").write_text("copied in and back")
    binding = SyncBinding(
        target_workspace=LocalWorkspace(target_root, workspace_id="target"),
        sync_back="always",
    )

    events = _run(_sync_app(source_root, binding), "phase-timing-sync")

    setup = _timings(events, EventType.ENVIRONMENT_BINDING_COMPLETED)
    phases = [row["phase"] for row in setup]
    assert {"target_materialization", "staging_admission", "source_observation", "transfer"} <= set(
        phases
    )
    transfer = next(row for row in setup if row["phase"] == "transfer")
    assert transfer["outcome"] == "completed"
    assert transfer["items"] == 1
    assert all(type(row["duration_ms"]) is int and row["duration_ms"] >= 0 for row in setup)
    started = [datetime.fromisoformat(row["started_at"]) for row in setup]
    assert started == sorted(started)

    teardown = _timings(events, EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED)
    assert {"final_target_observation", "copy_back_publication"} <= {
        row["phase"] for row in teardown
    }
    assert {row["outcome"] for row in teardown} == {"completed"}


class _FailingSetupBinding(SyncBinding):
    async def bind(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        bound = await super().bind(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )
        try:
            with environment_phase_step("workspace_setup_commands"):
                raise RuntimeError("setup command failed")
        except BaseException:
            self.abandon(bound)
            raise


def test_a_failed_binding_records_the_step_that_failed(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "input.txt").write_text("copied in")
    binding = _FailingSetupBinding(
        target_workspace=LocalWorkspace(target_root, workspace_id="target"),
        sync_back="never",
    )

    events = _run(_sync_app(source_root, binding), "phase-timing-failure")

    failed = _timings(events, EventType.ENVIRONMENT_BINDING_FAILED)
    assert failed[-1]["phase"] == "workspace_setup_commands"
    assert failed[-1]["outcome"] == "failed"
    assert "transfer" in {row["phase"] for row in failed}
