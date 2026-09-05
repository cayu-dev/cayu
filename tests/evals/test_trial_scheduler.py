from __future__ import annotations

import asyncio

import pytest

from cayu import EvalCase, EvalSuite, RunRequest
from cayu.evals.capacity import EvalExecutionCapacity
from cayu.evals.runner import _schedule_suite_trials
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1


def _schedule(execute, *, grouped, capacity):
    return _schedule_suite_trials(
        EvalSuite(
            id="scheduler",
            cases=[EvalCase(id="case", request=RunRequest(agent_name="test", messages=[]))],
        ),
        trials=1,
        max_concurrency=1,
        trial_policy=EvalSuiteTrialPolicyV1.create(trial_count=1),
        public_output_preview_bytes=None,
        execution_capacity=capacity,
        completed_trials={},
        trial_completed=None,
        execute_trial=execute,
        group_serial_trials=grouped,
    )


@pytest.mark.parametrize("grouped", [False, True])
def test_scheduler_preserves_serial_and_taskgroup_failure_identity(grouped):
    async def exercise():
        failure = RuntimeError("trial driver failed")
        capacity = EvalExecutionCapacity(1)

        async def execute(case, number):
            assert capacity.active_trials == 1
            raise failure

        if grouped:
            with pytest.raises(ExceptionGroup) as caught:
                await _schedule(execute, grouped=True, capacity=capacity)
            assert caught.value.exceptions == (failure,)
        else:
            with pytest.raises(RuntimeError) as caught:
                await _schedule(execute, grouped=False, capacity=capacity)
            assert caught.value is failure
        assert capacity.active_trials == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("grouped", [False, True])
def test_scheduler_parent_cancellation_settles_trial_and_capacity(grouped):
    async def exercise():
        entered = asyncio.Event()
        closed = asyncio.Event()
        capacity = EvalExecutionCapacity(1)

        async def execute(case, number):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        task = asyncio.create_task(_schedule(execute, grouped=grouped, capacity=capacity))
        await entered.wait()
        task.cancel("operator cancelled")
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.args == ("operator cancelled",)
        assert closed.is_set()
        assert capacity.active_trials == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("grouped", [False, True])
def test_scheduler_retains_capacity_until_checkpoint_publication_settles(grouped):
    from datetime import UTC, datetime

    from cayu import EvalStatus
    from cayu.evals.models import EvalTrialResult
    from cayu.evals.result_contract import (
        EvalTrialDiagnosticCode,
        EvalTrialOutputPreviewV1,
        _EvalTrialPublicData,
    )

    async def exercise():
        capacity = EvalExecutionCapacity(1)
        publishing = asyncio.Event()
        release = asyncio.Event()
        now = datetime.now(UTC)
        calls = []

        async def execute(case, number):
            calls.append(number)
            return (
                EvalTrialResult(
                    trial_number=number,
                    status=EvalStatus.ERROR,
                    error="trial execution failed",
                    started_at=now,
                    completed_at=now,
                ),
                _EvalTrialPublicData(
                    diagnostic_code=EvalTrialDiagnosticCode.EXECUTION_FAILED,
                    output=EvalTrialOutputPreviewV1.unavailable(),
                ),
            )

        async def publish(case_id, result, public_data):
            assert capacity.active_trials == 1
            publishing.set()
            await release.wait()
            assert capacity.active_trials == 1

        task = asyncio.create_task(
            _schedule_suite_trials(
                EvalSuite(
                    id="publish",
                    cases=[EvalCase(id="case", request=RunRequest(agent_name="test", messages=[]))],
                ),
                trials=1,
                max_concurrency=1,
                trial_policy=EvalSuiteTrialPolicyV1.create(trial_count=1),
                public_output_preview_bytes=1024,
                execution_capacity=capacity,
                completed_trials={},
                trial_completed=publish,
                execute_trial=execute,
                group_serial_trials=grouped,
            )
        )
        await asyncio.wait_for(publishing.wait(), 1)
        assert capacity.active_trials == 1 and not task.done()
        release.set()
        await task
        assert calls == [1]
        assert capacity.active_trials == 0

    asyncio.run(exercise())
