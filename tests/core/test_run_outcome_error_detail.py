"""Failed run outcomes keep exception-group leaves and cause chains (#2330)."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator

import pytest

from cayu import (
    Event,
    ExceptionCause,
    ExceptionDetail,
    ExceptionLeaf,
    Message,
    RunRequest,
    SecretRedactor,
    SessionStatus,
    exception_detail,
    run_to_completion,
)
from cayu.runtime._environment_lifecycle import exception_failure_payload
from cayu.runtime._exception_detail import (
    MAX_EXCEPTION_DETAIL_LEAVES,
    MAX_EXCEPTION_DETAIL_NODES,
)


def _request() -> RunRequest:
    return RunRequest(
        agent_name="assistant", session_id="s1", messages=[Message.text("user", "hi")]
    )


class _RaisingApp:
    def __init__(self, error: Exception, redactor: SecretRedactor | None = None) -> None:
        self._error = error
        if redactor is not None:
            self._secret_redactor = redactor

    async def run(self, request: RunRequest) -> AsyncIterator[Event]:
        raise self._error
        yield  # pragma: no cover


def _database_failure() -> sqlite3.OperationalError:
    try:
        try:
            raise OSError(24, "Too many open files")
        except OSError as error:
            raise sqlite3.OperationalError("unable to open database file") from error
    except sqlite3.OperationalError as error:
        return error


def _replay_failure() -> ExceptionGroup:
    return ExceptionGroup(
        "Interaction transition publication failed across replay attempts.",
        [_database_failure(), _database_failure(), _database_failure()],
    )


def test_run_to_completion_names_exception_group_leaves_and_their_causes() -> None:
    outcome = asyncio.run(run_to_completion(_RaisingApp(_replay_failure()), _request()))  # ty: ignore[invalid-argument-type]

    assert outcome.status is SessionStatus.FAILED
    assert outcome.error == (
        "ExceptionGroup: Interaction transition publication failed across replay "
        "attempts. (3 sub-exceptions); leaf errors: OperationalError: unable to open "
        "database file (3 times; caused by OSError: [Errno 24] Too many open files)"
    )
    assert outcome.error_detail == ExceptionDetail(
        error_type="ExceptionGroup",
        message=(
            "Interaction transition publication failed across replay attempts. (3 sub-exceptions)"
        ),
        leaves=(
            ExceptionLeaf(
                error_type="OperationalError",
                message="unable to open database file",
                count=3,
                causes=(
                    ExceptionCause(error_type="OSError", message="[Errno 24] Too many open files"),
                ),
            ),
        ),
    )


def test_run_to_completion_keeps_plain_exception_text_and_adds_cause_chain() -> None:
    try:
        try:
            raise KeyError("missing")
        except KeyError:
            raise RuntimeError("setup failed") from None
    except RuntimeError as suppressed:
        no_context = suppressed
    try:
        try:
            raise KeyError("missing")
        except KeyError:
            raise RuntimeError("setup failed")  # noqa: B904
    except RuntimeError as implicit:
        with_context = implicit

    plain = asyncio.run(run_to_completion(_RaisingApp(no_context), _request()))  # ty: ignore[invalid-argument-type]
    chained = asyncio.run(run_to_completion(_RaisingApp(with_context), _request()))  # ty: ignore[invalid-argument-type]

    assert plain.error == "RuntimeError: setup failed"
    assert plain.error_detail == ExceptionDetail(error_type="RuntimeError", message="setup failed")
    assert chained.error == "RuntimeError: setup failed; caused by KeyError: 'missing'"


def test_session_failed_event_outcome_has_no_exception_detail() -> None:
    class _FailedEventApp:
        async def run(self, request: RunRequest) -> AsyncIterator[Event]:
            yield Event(type="session.failed", session_id="s1", payload={"error": "boom"})

    outcome = asyncio.run(run_to_completion(_FailedEventApp(), _request()))  # ty: ignore[invalid-argument-type]

    assert outcome.error == "boom"
    assert outcome.error_detail is None


def test_exception_detail_redacts_registered_secrets_in_leaves_and_causes() -> None:
    secret = "sk-live-0123456789"
    try:
        try:
            raise ValueError(f"token {secret} rejected")
        except ValueError as error:
            raise ConnectionError(f"upstream refused {secret}") from error
    except ConnectionError as error:
        leaf = error
    group = ExceptionGroup(f"publish failed for {secret}", [leaf])
    redactor = SecretRedactor(secret)

    outcome = asyncio.run(run_to_completion(_RaisingApp(group, redactor), _request()))  # ty: ignore[invalid-argument-type]

    assert outcome.error == (
        "ExceptionGroup: publish failed for [REDACTED_SECRET] (1 sub-exception); "
        "leaf errors: ConnectionError: upstream refused [REDACTED_SECRET] "
        "(caused by ValueError: token [REDACTED_SECRET] rejected)"
    )
    assert secret not in repr(outcome.error_detail)


def test_exception_detail_bounds_distinct_leaves_and_counts_the_rest() -> None:
    group = ExceptionGroup(
        "many failures",
        [ValueError(f"failure {index}") for index in range(MAX_EXCEPTION_DETAIL_LEAVES + 5)],
    )

    detail = exception_detail(group)

    assert len(detail.leaves) == MAX_EXCEPTION_DETAIL_LEAVES
    assert detail.omitted_leaf_count == 5
    assert detail.summary().endswith(", and 5 more")


def test_exception_detail_walks_nested_groups_without_group_accessor_dispatch() -> None:
    class HostileGroup(ExceptionGroup):
        def __getattribute__(self, name: str):
            if name in {"exceptions", "__cause__", "__context__"}:
                raise RuntimeError(f"accessor {name} must not run")
            return super().__getattribute__(name)

        def __str__(self) -> str:
            raise RuntimeError("rendering must be guarded")

    nested = ExceptionGroup("outer", [HostileGroup("inner", [TimeoutError("slow")]), KeyError("k")])

    detail = exception_detail(nested)

    assert [(leaf.error_type, leaf.message) for leaf in detail.leaves] == [
        ("TimeoutError", "slow"),
        ("KeyError", "'k'"),
    ]
    assert exception_detail(HostileGroup("inner", [TimeoutError("slow")])).message == (
        "message could not be rendered"
    )


def test_exception_detail_counts_repeated_leaves_and_shared_subgroups() -> None:
    leaf = ValueError("retry failed")
    shared = ExceptionGroup("retry pair", [leaf, leaf])
    detail = exception_detail(ExceptionGroup("retries", [shared, shared, leaf]))

    assert detail.leaves == (
        ExceptionLeaf(error_type="ValueError", message="retry failed", count=5),
    )
    assert detail.omitted_leaf_count == 0
    assert not detail.traversal_truncated


@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("extra", [0, 1, 45])
def test_exception_detail_reports_node_budget_truncation(repeated: bool, extra: int) -> None:
    count = MAX_EXCEPTION_DETAIL_NODES - 1 + extra
    leaf = ValueError("failure")
    children = [leaf] * count if repeated else [ValueError("failure") for _ in range(count)]
    detail = exception_detail(ExceptionGroup("bulk", children))

    assert detail.leaves == (
        ExceptionLeaf(
            error_type="ValueError", message="failure", count=MAX_EXCEPTION_DETAIL_NODES - 1
        ),
    )
    assert detail.omitted_leaf_count == 0
    assert detail.traversal_truncated is (extra > 0)
    assert ("leaf counts are incomplete" in detail.summary()) is (extra > 0)


def test_exception_detail_reports_truncation_before_any_leaf_is_reached() -> None:
    error: Exception = ValueError("hidden failure")
    for _ in range(MAX_EXCEPTION_DETAIL_NODES):
        error = ExceptionGroup("nested", [error])

    outcome = asyncio.run(run_to_completion(_RaisingApp(error), _request()))  # ty: ignore[invalid-argument-type]

    assert outcome.error_detail is not None
    assert outcome.error_detail.leaves == ()
    assert outcome.error_detail.omitted_leaf_count == 0
    assert outcome.error_detail.traversal_truncated
    assert outcome.error is not None and "leaf counts are incomplete" in outcome.error
    assert "leaf counts are incomplete" in exception_failure_payload(error)["error"]


def test_exception_detail_separates_visited_omissions_from_unvisited_leaves() -> None:
    group = ExceptionGroup(
        "many failures", [ValueError(str(index)) for index in range(MAX_EXCEPTION_DETAIL_NODES)]
    )

    detail = exception_detail(group)

    assert len(detail.leaves) == MAX_EXCEPTION_DETAIL_LEAVES
    assert detail.omitted_leaf_count == MAX_EXCEPTION_DETAIL_NODES - 1 - MAX_EXCEPTION_DETAIL_LEAVES
    assert detail.traversal_truncated
    assert "leaf counts are incomplete" in detail.summary()


def test_session_failure_payload_for_a_group_lists_leaf_errors() -> None:
    payload = exception_failure_payload(_replay_failure())

    assert payload["error_type"] == "ExceptionGroup"
    assert payload["error"] == (
        "Interaction transition publication failed across replay attempts. "
        "(3 sub-exceptions); leaf errors: OperationalError: unable to open database "
        "file (3 times; caused by OSError: [Errno 24] Too many open files)"
    )
    assert exception_failure_payload(ValueError("plain"))["error"] == "plain"
