"""Cancellation, stream-close and crash injection at every point of Runtime scenarios.

Pull requests run the sample of distinct call sites, the known-failure checks and
the point check; ``fault_injection_exhaustive`` adds every other point.

See ``tests/fault_injection/README.md`` for the design and the failures found.
"""

from __future__ import annotations

import pytest
from tests.fault_injection.explore import (
    SCENARIOS,
    Coverage,
    Exploration,
    baseline_points,
    explore,
    run_point,
)
from tests.fault_injection.harness import FaultKind
from tests.fault_injection.known_failures import KNOWN_FAILURES, KnownFailure, is_known

pytestmark = pytest.mark.fault_injection


def _assert_only_known_violations(exploration: Exploration) -> None:
    # Every point of the fault-free trace must be reached again; a scenario whose
    # trace changes from run to run fails here instead of exploring less.
    assert not exploration.unreached, exploration.unreached
    unexpected = [violation for violation in exploration.violations if not is_known(violation)]
    assert not unexpected, (
        f"{len(unexpected)} invariant violations over {len(exploration.points)} points:\n"
        + "\n".join(violation.describe() for violation in unexpected)
    )


@pytest.mark.parametrize("fault", list(FaultKind), ids=str)
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_sampled_injection_points_keep_runtime_invariants(scenario: str, fault: FaultKind) -> None:
    _assert_only_known_violations(explore(SCENARIOS[scenario], fault, Coverage.SAMPLE))


# Every stream-close point is in the sample, so the rest has none.
@pytest.mark.fault_injection_exhaustive
@pytest.mark.parametrize(
    "fault", [fault for fault in FaultKind if fault is not FaultKind.STREAM_CLOSE], ids=str
)
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_remaining_injection_points_keep_runtime_invariants(
    scenario: str, fault: FaultKind
) -> None:
    _assert_only_known_violations(explore(SCENARIOS[scenario], fault, Coverage.REST))


def test_known_failure_points_are_still_reached() -> None:
    baselines = {
        key: baseline_points(SCENARIOS[key[0]], key[1])
        for key in {(known.scenario, known.fault) for known in KNOWN_FAILURES}
    }

    stale = {
        known.key: sorted(set(known.points) - set(baselines[known.scenario, known.fault]))
        for known in KNOWN_FAILURES
        if not set(known.points) <= set(baselines[known.scenario, known.fault])
    }
    assert not stale, f"update these points in known_failures.py: {stale}"


@pytest.mark.parametrize(
    "known",
    [
        pytest.param(
            known,
            id=known.key,
            marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=known.reason),
        )
        for known in KNOWN_FAILURES
    ],
)
def test_known_failure_is_fixed(known: KnownFailure) -> None:
    # The first point stands for the entry; the exhaustive test covers the rest.
    reproduced = [
        violation
        for violation in run_point(
            SCENARIOS[known.scenario], known.fault, known.points[0]
        ).violations
        if known.matches(violation)
    ]

    assert not reproduced, "\n".join(violation.describe() for violation in reproduced)
