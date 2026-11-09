"""``close_on_failure``: a sync factory closes what it opened when building raises."""

from __future__ import annotations

import asyncio
import threading

import pytest

from cayu.runtime import close_on_failure


class _Resource:
    def __init__(self, name: str, log: list[str], *, fail: bool = False) -> None:
        self.name = name
        self._log = log
        self._fail = fail
        self.closed_on: threading.Thread | None = None

    async def close(self) -> None:
        self.closed_on = threading.current_thread()
        self._log.append(self.name)
        if self._fail:
            raise OSError(f"{self.name} did not close")


class _AcloseResource:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    async def aclose(self) -> None:
        self._log.append("aclose")


def test_success_leaves_resources_open() -> None:
    log: list[str] = []
    with close_on_failure(_Resource("stores", log)):
        pass
    assert log == []


def test_failure_closes_in_reverse_order_once_and_reraises() -> None:
    log: list[str] = []
    first = _Resource("first", log)
    second = _Resource("second", log)
    with (
        pytest.raises(ValueError, match="build failed"),
        close_on_failure(first, None, second, first, _AcloseResource(log)),
    ):
        raise ValueError("build failed")
    assert log == ["aclose", "second", "first"]


def test_a_failed_close_is_raised_with_the_build_error() -> None:
    log: list[str] = []
    with (
        pytest.raises(BaseExceptionGroup) as raised,
        close_on_failure(_Resource("first", log), _Resource("second", log, fail=True)),
    ):
        raise ValueError("build failed")
    errors = raised.value.exceptions
    assert [type(error) for error in errors] == [ValueError, OSError]
    assert str(errors[1]) == "second did not close"
    # A failed close does not stop the rest from closing.
    assert log == ["second", "first"]


class _CancelledResource:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    async def close(self) -> None:
        self._log.append("cancelled")
        raise asyncio.CancelledError


def test_a_cancelled_close_does_not_stop_the_rest_from_closing() -> None:
    log: list[str] = []
    with (
        pytest.raises(BaseExceptionGroup) as raised,
        close_on_failure(
            _Resource("first", log),
            _CancelledResource(log),
            _Resource("last", log, fail=True),
        ),
    ):
        raise ValueError("build failed")
    errors = raised.value.exceptions
    assert [type(error) for error in errors] == [
        ValueError,
        OSError,
        asyncio.CancelledError,
    ]
    assert log == ["last", "cancelled", "first"]


def test_inside_a_running_loop_the_closes_run_on_another_thread() -> None:
    log: list[str] = []
    resource = _Resource("stores", log)

    async def build() -> None:
        with close_on_failure(resource):
            raise ValueError("build failed")

    with pytest.raises(ValueError, match="build failed"):
        asyncio.run(build())
    assert log == ["stores"]
    assert resource.closed_on is not threading.current_thread()


def test_a_resource_that_cannot_close_is_rejected_before_building() -> None:
    built = False
    with (
        pytest.raises(TypeError, match="async close"),
        close_on_failure(object()),  # ty: ignore[invalid-argument-type]
    ):
        built = True
    assert not built
