from __future__ import annotations

import pytest
from examples._live_checks import is_superseded_browser_read_abort


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/evals/scenarios/example"),
        ("POST", "/api/sessions/summary"),
        ("POST", "/api/usage/rollup"),
        ("POST", "/api/sessions/example/topology"),
    ],
)
def test_superseded_browser_read_abort_is_non_fatal(method: str, path: str) -> None:
    assert is_superseded_browser_read_abort(
        method=method,
        path=path,
        failure="net::ERR_ABORTED",
        mutation_id=None,
    )


@pytest.mark.parametrize(
    ("method", "path", "failure", "mutation_id"),
    [
        ("POST", "/api/run", "net::ERR_ABORTED", None),
        ("POST", "/api/sessions/summary", "net::ERR_FAILED", None),
        ("POST", "/api/sessions/summary", "net::ERR_ABORTED", "mutation-1"),
        ("GET", "/assets/dashboard.js", "net::ERR_ABORTED", None),
    ],
)
def test_browser_failures_that_can_signal_regressions_remain_fatal(
    method: str,
    path: str,
    failure: str,
    mutation_id: str | None,
) -> None:
    assert not is_superseded_browser_read_abort(
        method=method,
        path=path,
        failure=failure,
        mutation_id=mutation_id,
    )
