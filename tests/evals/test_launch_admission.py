from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from itertools import pairwise

import pytest

from cayu import EvalCase, EvalStatus, EvalSuite, RunRequest
from cayu.evals import _admission
from cayu.evals._admission import LaunchAdmission, admission_scope
from cayu.evals.models import EvalTrialResult
from cayu.evals.runner import _schedule_suite_trials
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1


class Clock:
    def __init__(self, monkeypatch):
        self.now = 100.0
        self.sleepers = []
        monkeypatch.setattr(_admission, "monotonic", lambda: self.now)
        monkeypatch.setattr(_admission, "sleep", self.sleep)

    async def sleep(self, delay):
        event = asyncio.Event()
        self.sleepers.append((self.now + delay, event))
        await event.wait()

    async def advance(self, seconds=0):
        self.now += seconds
        for deadline, event in self.sleepers:
            if deadline <= self.now:
                event.set()
        for _ in range(10):
            await asyncio.sleep(0)


async def schedule(execute, capacity=2):
    return await _schedule_suite_trials(
        EvalSuite(
            id="paced",
            cases=[
                EvalCase(id=str(i), request=RunRequest(agent_name="test", messages=[]))
                for i in range(4)
            ],
        ),
        trials=1,
        max_concurrency=capacity,
        trial_policy=EvalSuiteTrialPolicyV1.create(trial_count=1, max_concurrency=capacity),
        public_output_preview_bytes=None,
        execution_capacity=None,
        completed_trials={},
        trial_completed=None,
        execute_trial=execute,
    )


def result(number):
    now = datetime.now(UTC)
    return EvalTrialResult(
        trial_number=number,
        status=EvalStatus.ERROR,
        error="test result",
        started_at=now,
        completed_at=now,
    ), None


@pytest.mark.parametrize("interval", [0, 30])
def test_capacity_replacements_and_no_catch_up(monkeypatch, interval):
    async def exercise():
        clock = Clock(monkeypatch)
        gate = LaunchAdmission(interval)
        releases = [asyncio.Event() for _ in range(4)]
        starts = []

        async def execute(case, number):
            assert _admission.current_launch_admission() is None
            starts.append((case.id, clock.now))
            await releases[int(case.id)].wait()
            return result(number)

        with admission_scope(gate):
            task = asyncio.create_task(schedule(execute))
        await clock.advance()
        assert len(starts) == (2 if interval == 0 else 1)
        await clock.advance(30)
        assert len(starts) == 2
        # Capacity saturation skips time slots; freeing both slots cannot burst.
        await clock.advance(100)
        assert len(starts) == 2
        releases[0].set()
        releases[1].set()
        await clock.advance()
        assert len(starts) == (4 if interval == 0 else 3)
        await clock.advance(29)
        assert len(starts) == (4 if interval == 0 else 3)
        await clock.advance(1)
        assert len(starts) == 4
        for event in releases:
            event.set()
        await task
        times = [item.monotonic_seconds for item in gate.admissions]
        assert times == [time for _, time in starts]
        assert all(b - a >= interval for a, b in pairwise(times))

    asyncio.run(exercise())


def test_worker_gates_share_one_clock_without_catch_up(monkeypatch, tmp_path):
    async def exercise():
        clock = Clock(monkeypatch)
        first = LaunchAdmission(30, directory=tmp_path, worker=0)
        second = LaunchAdmission(30, directory=tmp_path, worker=1)
        await first.admit("a", 1)
        waiting = asyncio.create_task(second.admit("b", 1))
        await clock.advance()
        assert not waiting.done()
        await clock.advance(90)
        await waiting
        replacement = asyncio.create_task(first.admit("c", 1))
        await clock.advance()
        assert not replacement.done()
        await clock.advance(30)
        await replacement
        assert [item.monotonic_seconds for item in first.admissions] == [100, 220]
        assert [item.monotonic_seconds for item in second.admissions] == [190]

    asyncio.run(exercise())


@pytest.mark.parametrize("shared", [False, True])
def test_cancellation_prevents_admission(monkeypatch, tmp_path, shared):
    async def exercise():
        clock = Clock(monkeypatch)
        gate = LaunchAdmission(30, directory=tmp_path if shared else None)
        await gate.admit("a", 1)
        task = asyncio.create_task(gate.admit("b", 1))
        await clock.advance()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await clock.advance(100)
        assert [item.case_id for item in gate.admissions] == ["a"]

    asyncio.run(exercise())


def test_expired_overall_deadline_prevents_admission():
    from cayu.deadlines import (
        ExecutionDeadline,
        ExecutionDeadlineExceeded,
        execution_deadline_scope,
    )

    async def exercise():
        gate = LaunchAdmission(0)
        with pytest.raises(ExecutionDeadlineExceeded):
            async with execution_deadline_scope(ExecutionDeadline.after(0)):
                await gate.admit("a", 1)
        assert not gate.admissions

    asyncio.run(exercise())


@pytest.mark.parametrize("interval", [-1, float("inf"), float("-inf"), float("nan")])
def test_invalid_interval(interval):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        LaunchAdmission(interval)


def test_overall_timeout_cancels_wait_without_later_admissions(monkeypatch):
    from cayu.deadlines import ExecutionDeadline, execution_deadline_scope

    async def exercise():
        clock = Clock(monkeypatch)
        gate = LaunchAdmission(30)
        with pytest.raises(TimeoutError):
            async with execution_deadline_scope(ExecutionDeadline.after(0.01)):
                await gate.admit("a", 1)
                await gate.admit("b", 1)
        await clock.advance(100)
        assert [item.case_id for item in gate.admissions] == ["a"]

    asyncio.run(exercise())


def test_queue_wait_precedes_case_timeout(monkeypatch):

    async def exercise():
        clock = Clock(monkeypatch)
        gate = LaunchAdmission(30)
        entered = []

        async def execute(case, number):
            # This is the same boundary where the runner starts case execution.
            # Waiting 30 seconds must not create/consume this timeout scope.
            async with asyncio.timeout(0.01):
                entered.append(clock.now)
                await asyncio.sleep(0)
                return result(number)

        with admission_scope(gate):
            task = asyncio.create_task(schedule(execute, capacity=1))
        await clock.advance()
        assert entered == [100]
        for _ in range(3):
            await clock.advance(30)
        await task
        assert entered == [100, 130, 160, 190]

    asyncio.run(exercise())


def test_shared_file_lock_wait_is_cooperative_and_cancellable(monkeypatch, tmp_path):
    fcntl = pytest.importorskip("fcntl")

    async def exercise():
        clock = Clock(monkeypatch)
        gate = LaunchAdmission(30, directory=tmp_path)
        with (tmp_path / "admission.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            task = asyncio.create_task(gate.admit("case", 1))
            await clock.advance()
            assert not task.done()
            assert not gate.admissions
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            fcntl.flock(lock, fcntl.LOCK_UN)
        assert not gate.admissions

    asyncio.run(exercise())
