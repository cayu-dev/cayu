"""A claim with no handler left to drain leaves through one exit table."""

import asyncio

import pytest

from cayu.tasks._task_run_settlement import release_drained_claim
from cayu.tasks.records import TaskClaimLost
from cayu.tasks.terminalization import TaskTerminalizationConflict


class _Store:
    def __init__(self, *outcomes: BaseException | None) -> None:
        self.outcomes = list(outcomes)
        self.releases: list[tuple[str, str, int]] = []

    async def release_task(self, task_id, worker_id, *, claim_generation):
        self.releases.append((task_id, worker_id, claim_generation))
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if outcome is not None:
            raise outcome


async def _release(store: _Store, failure: Exception | None, settled: list[str]) -> bool:
    async def settle_cancellation() -> None:
        settled.append("cancelled")

    return await release_drained_claim(
        store,
        task_id="task",
        worker_id="worker",
        claim_generation=3,
        failure=failure,
        settle_cancellation=settle_cancellation,
        release_failure_message="failure and release both failed",
    )


@pytest.mark.parametrize("failure", [None, OSError("pre-admission")], ids=["stopped", "failed"])
@pytest.mark.parametrize(
    ("outcomes", "released", "cancelled"),
    [
        ((), True, False),
        ((ConnectionError("transient"),), True, False),
        ((TaskTerminalizationConflict("operator cancelled"),), False, True),
        ((TaskClaimLost("another owner"),), False, False),
    ],
    ids=["released", "retried", "operator-cancelled", "claim-lost"],
)
def test_every_release_exit_settles_the_claim(failure, outcomes, released, cancelled):
    store, settled = _Store(*outcomes), []
    assert asyncio.run(_release(store, failure, settled)) is released
    assert settled == (["cancelled"] if cancelled else [])
    assert set(store.releases) == {("task", "worker", 3)}


@pytest.mark.parametrize("failure", [None, OSError("pre-admission")], ids=["stopped", "failed"])
def test_a_release_that_keeps_failing_is_reported_with_its_cause(failure):
    store = _Store(ConnectionError("first"), ConnectionError("second"))
    with pytest.raises(BaseException) as raised:
        asyncio.run(_release(store, failure, []))
    assert len(store.releases) == 2
    if failure is None:
        assert type(raised.value) is ConnectionError
    else:
        assert type(raised.value) is ExceptionGroup
        first, second = raised.value.exceptions
        assert first is failure and str(second) == "second"


def test_caller_cancellation_is_chained_to_the_failure():
    failure = OSError("pre-admission")
    with pytest.raises(asyncio.CancelledError) as raised:
        asyncio.run(_release(_Store(asyncio.CancelledError()), failure, []))
    assert raised.value.__cause__ is failure
