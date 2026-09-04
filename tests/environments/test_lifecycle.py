from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from cayu.core import Event, EventType
from cayu.environments.lifecycle import (
    DEFAULT_ENVIRONMENT_PHASE_TIMEOUT_SECONDS,
    MAX_ENVIRONMENT_PROGRESS_COUNTER,
    EnvironmentLifecycleDeadlineExceeded,
    EnvironmentLifecycleOperation,
    EnvironmentLifecyclePhase,
    EnvironmentLifecyclePolicy,
    EnvironmentLifecycleProgress,
    EnvironmentLifecycleProgressStatus,
    RuntimeEnvironmentLifecycleProgressReporter,
    copy_environment_lifecycle_policy,
    environment_lifecycle_progress_from_event,
)


class _Clock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 1, 2, tzinfo=UTC)
        self.elapsed = 0.0

    def now(self) -> datetime:
        return self.wall + timedelta(seconds=self.elapsed)

    def monotonic(self) -> float:
        return self.elapsed

    def advance(self, seconds: float) -> None:
        self.elapsed += seconds


def test_environment_lifecycle_policy_completes_partial_phase_overrides() -> None:
    overrides = {EnvironmentLifecyclePhase.TRANSFER: 3.0}
    policy = EnvironmentLifecyclePolicy(
        lifecycle_timeout_seconds=20.0,
        phase_timeout_seconds=overrides,
        progress_min_interval_seconds=0.0,
        max_progress_events=32,
    )
    overrides[EnvironmentLifecyclePhase.TRANSFER] = 9.0

    assert policy.timeout_for(EnvironmentLifecyclePhase.TRANSFER) == 3.0
    assert (
        policy.timeout_for(EnvironmentLifecyclePhase.SOURCE_OBSERVATION)
        == DEFAULT_ENVIRONMENT_PHASE_TIMEOUT_SECONDS
    )
    assert set(policy.phase_timeout_seconds) == set(EnvironmentLifecyclePhase)
    assert copy_environment_lifecycle_policy(policy) == policy
    assert copy_environment_lifecycle_policy(policy) is not policy
    assert deepcopy(policy) == policy
    assert policy.model_copy(deep=True) == policy
    with pytest.raises(TypeError):
        policy.phase_timeout_seconds[EnvironmentLifecyclePhase.TRANSFER] = 4.0  # type: ignore[index]


def test_environment_lifecycle_progress_rejects_content_and_invalid_aggregates() -> None:
    deadline = datetime(2026, 1, 2, tzinfo=UTC)
    base = {
        "operation_id": "envop_test",
        "operation": EnvironmentLifecycleOperation.BINDING,
        "phase": EnvironmentLifecyclePhase.TRANSFER,
        "status": EnvironmentLifecycleProgressStatus.ADVANCED,
        "event_index": 1,
        "elapsed_ms": 1,
        "phase_elapsed_ms": 1,
        "lifecycle_timeout_seconds": 10.0,
        "phase_timeout_seconds": 5.0,
        "deadline": deadline,
        "last_progress_at": deadline,
    }

    with pytest.raises(ValidationError):
        EnvironmentLifecycleProgress.model_validate({**base, "path": "private.txt"})
    with pytest.raises(ValidationError, match="items_completed cannot exceed"):
        EnvironmentLifecycleProgress.model_validate(
            {**base, "items_completed": 2, "items_total": 1}
        )
    with pytest.raises(ValidationError, match="less than or equal"):
        EnvironmentLifecycleProgress.model_validate(
            {**base, "bytes_completed": MAX_ENVIRONMENT_PROGRESS_COUNTER + 1}
        )
    with pytest.raises(ValidationError, match="binding_generation_id"):
        EnvironmentLifecycleProgress.model_validate(
            {**base, "binding_generation_id": "invalid\nidentity"}
        )

    largest = EnvironmentLifecycleProgress.model_validate(
        {
            **base,
            "operation_id": "o" * 128,
            "binding_generation_id": "b" * 128,
            "items_completed": MAX_ENVIRONMENT_PROGRESS_COUNTER,
            "items_total": MAX_ENVIRONMENT_PROGRESS_COUNTER,
            "bytes_completed": MAX_ENVIRONMENT_PROGRESS_COUNTER,
            "bytes_total": MAX_ENVIRONMENT_PROGRESS_COUNTER,
            "active_count": MAX_ENVIRONMENT_PROGRESS_COUNTER,
            "queued_count": MAX_ENVIRONMENT_PROGRESS_COUNTER,
        }
    )
    assert len(largest.model_dump_json().encode("utf-8")) < 2048


def test_environment_lifecycle_reporter_enforces_phase_deadline_at_progress_boundary() -> None:
    clock = _Clock()
    published: list[EnvironmentLifecycleProgress] = []

    async def publish(progress: EnvironmentLifecycleProgress) -> None:
        published.append(progress)

    reporter = RuntimeEnvironmentLifecycleProgressReporter(
        operation=EnvironmentLifecycleOperation.BINDING,
        policy=EnvironmentLifecyclePolicy(
            lifecycle_timeout_seconds=10.0,
            phase_timeout_seconds={EnvironmentLifecyclePhase.TRANSFER: 1.0},
            progress_min_interval_seconds=0.0,
            max_progress_events=32,
        ),
        publish=publish,
        now=clock.now,
        monotonic=clock.monotonic,
    )

    async def run() -> None:
        await reporter.report(
            EnvironmentLifecyclePhase.TRANSFER,
            EnvironmentLifecycleProgressStatus.STARTED,
        )
        clock.advance(1.1)
        with pytest.raises(EnvironmentLifecycleDeadlineExceeded) as exc_info:
            await reporter.report(
                EnvironmentLifecyclePhase.TRANSFER,
                EnvironmentLifecycleProgressStatus.ADVANCED,
                items_completed=1,
                items_total=2,
            )
        assert exc_info.value.scope == "phase"

    asyncio.run(run())

    assert [item.status for item in published] == [
        EnvironmentLifecycleProgressStatus.STARTED,
        EnvironmentLifecycleProgressStatus.DEADLINE_EXCEEDED,
    ]
    assert published[-1].phase is EnvironmentLifecyclePhase.TRANSFER
    assert published[-1].deadline_scope == "phase"
    assert published[-1].event_index == 2
    assert published[-1].operation_terminal is True
    assert published[-1].last_progress_at == published[0].last_progress_at
    assert published[-1].last_progress_at < published[-1].deadline


def test_environment_lifecycle_reporter_bounds_progress_volume_and_reserves_terminal() -> None:
    clock = _Clock()
    published: list[EnvironmentLifecycleProgress] = []

    async def publish(progress: EnvironmentLifecycleProgress) -> None:
        published.append(progress)

    reporter = RuntimeEnvironmentLifecycleProgressReporter(
        operation=EnvironmentLifecycleOperation.FINALIZATION,
        policy=EnvironmentLifecyclePolicy(
            progress_min_interval_seconds=0.0,
            max_progress_events=32,
        ),
        publish=publish,
        now=clock.now,
        monotonic=clock.monotonic,
    )

    async def run() -> None:
        for index in range(100):
            await reporter.report(
                EnvironmentLifecyclePhase.COPY_BACK_PUBLICATION,
                EnvironmentLifecycleProgressStatus.ADVANCED,
                items_completed=index,
            )
            clock.advance(0.001)
        await reporter.finish(status=EnvironmentLifecycleProgressStatus.COMPLETED)

    asyncio.run(run())

    assert len(published) == 32
    assert published[-1].status is EnvironmentLifecycleProgressStatus.COMPLETED
    assert published[-1].event_index == 32
    assert published[-1].operation_terminal is True


def test_100_concurrent_setup_and_finalization_projections_stay_bounded() -> None:
    published: list[EnvironmentLifecycleProgress] = []
    policy = EnvironmentLifecyclePolicy(
        progress_min_interval_seconds=0.0,
        max_progress_events=32,
    )

    async def publish(progress: EnvironmentLifecycleProgress) -> None:
        await asyncio.sleep(0)
        published.append(progress)

    async def worker(index: int) -> None:
        binding = RuntimeEnvironmentLifecycleProgressReporter(
            operation=EnvironmentLifecycleOperation.BINDING,
            policy=policy,
            publish=publish,
            binding_generation_id=f"envbind_{index}",
            operation_id=f"envop_binding_{index}",
        )
        await binding.report(
            EnvironmentLifecyclePhase.STAGING_ADMISSION,
            EnvironmentLifecycleProgressStatus.STARTED,
            active_count=index + 1,
            queued_count=99 - index,
        )
        await binding.report(
            EnvironmentLifecyclePhase.TRANSFER,
            EnvironmentLifecycleProgressStatus.STARTED,
            bytes_completed=0,
            bytes_total=100,
        )
        await binding.report(
            EnvironmentLifecyclePhase.TRANSFER,
            EnvironmentLifecycleProgressStatus.ADVANCED,
            bytes_completed=50,
            bytes_total=100,
        )
        await binding.report(
            EnvironmentLifecyclePhase.TRANSFER,
            EnvironmentLifecycleProgressStatus.COMPLETED,
            bytes_completed=100,
            bytes_total=100,
        )
        await binding.finish(status=EnvironmentLifecycleProgressStatus.COMPLETED)

        finalization = RuntimeEnvironmentLifecycleProgressReporter(
            operation=EnvironmentLifecycleOperation.FINALIZATION,
            policy=policy,
            publish=publish,
            binding_generation_id=f"envbind_{index}",
            operation_id=f"envop_finalization_{index}",
        )
        await finalization.report(
            EnvironmentLifecyclePhase.FINAL_TARGET_OBSERVATION,
            EnvironmentLifecycleProgressStatus.STARTED,
        )
        await finalization.report(
            EnvironmentLifecyclePhase.FINAL_TARGET_OBSERVATION,
            EnvironmentLifecycleProgressStatus.COMPLETED,
            items_completed=1,
            items_total=1,
        )
        await finalization.report(
            EnvironmentLifecyclePhase.COPY_BACK_PUBLICATION,
            EnvironmentLifecycleProgressStatus.COMPLETED,
            items_completed=0,
            items_total=0,
            bytes_completed=0,
            bytes_total=0,
        )
        await finalization.finish(status=EnvironmentLifecycleProgressStatus.COMPLETED)

    async def run() -> None:
        await asyncio.wait_for(
            asyncio.gather(*(worker(index) for index in range(100))),
            timeout=2.0,
        )

    asyncio.run(run())

    by_operation: dict[str, list[EnvironmentLifecycleProgress]] = {}
    for progress in published:
        by_operation.setdefault(progress.operation_id, []).append(progress)
    assert len(by_operation) == 200
    assert all(len(events) <= policy.max_progress_events for events in by_operation.values())
    assert all(events[-1].operation_terminal for events in by_operation.values())
    assert all(
        events[-1].status is EnvironmentLifecycleProgressStatus.COMPLETED
        for events in by_operation.values()
    )
    binding_phases = {
        event.phase
        for event in published
        if event.operation is EnvironmentLifecycleOperation.BINDING
    }
    assert binding_phases == {
        EnvironmentLifecyclePhase.STAGING_ADMISSION,
        EnvironmentLifecyclePhase.TRANSFER,
    }


def test_environment_lifecycle_reporter_rejects_late_terminal_completion() -> None:
    clock = _Clock()
    published: list[EnvironmentLifecycleProgress] = []

    async def publish(progress: EnvironmentLifecycleProgress) -> None:
        published.append(progress)

    reporter = RuntimeEnvironmentLifecycleProgressReporter(
        operation=EnvironmentLifecycleOperation.FACTORY,
        policy=EnvironmentLifecyclePolicy(lifecycle_timeout_seconds=1.0),
        publish=publish,
        now=clock.now,
        monotonic=clock.monotonic,
    )

    async def run() -> None:
        await reporter.report(
            EnvironmentLifecyclePhase.OWNERSHIP_ADMISSION,
            EnvironmentLifecycleProgressStatus.COMPLETED,
        )
        clock.advance(1.1)
        with pytest.raises(EnvironmentLifecycleDeadlineExceeded) as exc_info:
            await reporter.finish(status=EnvironmentLifecycleProgressStatus.COMPLETED)
        assert exc_info.value.scope == "lifecycle"

    asyncio.run(run())

    assert published[-1].status is EnvironmentLifecycleProgressStatus.DEADLINE_EXCEEDED
    assert published[-1].deadline_scope == "lifecycle"
    assert published[-1].operation_terminal is True


def test_environment_lifecycle_event_parser_uses_typed_public_projection() -> None:
    timestamp = datetime(2026, 1, 2, tzinfo=UTC)
    progress = EnvironmentLifecycleProgress(
        operation_id="envop_test",
        binding_generation_id="envbind_test",
        operation=EnvironmentLifecycleOperation.BINDING,
        phase=EnvironmentLifecyclePhase.EXECUTION_READY_PUBLICATION,
        status=EnvironmentLifecycleProgressStatus.COMPLETED,
        event_index=1,
        elapsed_ms=2,
        phase_elapsed_ms=1,
        lifecycle_timeout_seconds=10.0,
        phase_timeout_seconds=5.0,
        deadline=timestamp,
        last_progress_at=timestamp,
    )
    event = Event(
        type=EventType.ENVIRONMENT_LIFECYCLE_PROGRESS,
        session_id="session-1",
        payload=progress.to_payload(),
    )

    assert environment_lifecycle_progress_from_event(event) == progress
    with pytest.raises(ValueError, match="not environment lifecycle progress"):
        environment_lifecycle_progress_from_event(
            Event(type=EventType.SESSION_STARTED, session_id="session-1")
        )
