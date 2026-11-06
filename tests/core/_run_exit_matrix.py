"""Drive one settling body to every exit point and record how it settled.

A scenario is a body with ``points`` steps. Step ``i`` awaits once and then
yields once. The matrix exits the body at every point, in every way a consumer
or caller can end it:

- ``close``: the consumer closes the stream after receiving item ``i``;
- ``cancel``: the caller cancels while the body awaits in step ``i``;
- ``grouped-cancel``: as ``cancel``, but body cleanup groups the delivered
  cancellation with an ``OSError``;
- ``fail``: the body raises an ordinary error in step ``i``;
- ``return``: the body runs to completion.

``run_exit_case`` returns what escaped and leaves the caller to assert on the
settlement it observed. Real Runtime scenarios reuse the same exits.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any

EXIT_KINDS = ("close", "cancel", "grouped-cancel", "fail", "return")


@dataclass(frozen=True, slots=True)
class ExitCase:
    kind: str
    point: int

    @property
    def id(self) -> str:
        return f"{self.kind}@{self.point}"


def exit_cases(points: int) -> list[ExitCase]:
    cases = [ExitCase(kind, point) for kind in EXIT_KINDS[:-1] for point in range(points)]
    cases.append(ExitCase("return", points))
    return cases


class BodyFailure(RuntimeError):
    pass


@dataclass(slots=True)
class ExitOutcome:
    escaped: BaseException | None
    received: list[Any]


async def run_exit_case(
    case: ExitCase,
    *,
    points: int,
    stream_for: Callable[[AsyncGenerator[int, None]], AsyncGenerator[Any, None]],
) -> ExitOutcome:
    """Run ``stream_for(body)`` and end it as ``case`` says."""

    reached = asyncio.Event()

    async def body() -> AsyncGenerator[int, None]:
        for index in range(points):
            if case.kind == "fail" and case.point == index:
                raise BodyFailure(f"failed at {index}")
            if case.kind in {"cancel", "grouped-cancel"} and case.point == index:
                reached.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as cancellation:
                    if case.kind == "grouped-cancel":
                        raise BaseExceptionGroup(
                            "delivery cancelled and cleanup failed",
                            [cancellation, OSError("sink close failed")],
                        ) from None
                    raise
            await asyncio.sleep(0)
            yield index

    received: list[Any] = []
    stream = stream_for(body())

    async def consume() -> None:
        async for item in stream:
            received.append(item)
            if case.kind == "close" and case.point == len(received) - 1:
                await stream.aclose()
                return

    task = asyncio.create_task(consume())
    if case.kind in {"cancel", "grouped-cancel"}:
        await reached.wait()
        task.cancel("caller left")
    escaped: BaseException | None = None
    try:
        await task
    except BaseException as error:
        escaped = error
    return ExitOutcome(escaped=escaped, received=received)
