"""Deterministic fault injection over real Runtime scenarios.

Every scenario runs a real ``CayuApp`` over an in-memory session (and task) store,
a scripted model provider and in-process tools. Each of those three boundaries
reports to one :class:`Injector`, which numbers the boundaries a scenario's phase
under test crosses. A first, fault-free run records that numbered trace; the
exploration then reruns the scenario once for every boundary ``k`` and injects
one fault there:

``cancel``
    Cancel the task driving the phase when it reaches boundary ``k``. Every store
    call has an entry boundary and every write also an exit boundary, so ``k``
    covers a cancellation that lands before a write and one that lands after the
    write committed but before its caller saw the result.
``grouped-cancel``
    Cancel the task driving the phase, and the task at boundary ``k`` if that is
    another one, as a task group cancels its children. The boundary then stands in
    for cleanup code that receives that cancellation and fails too: it raises the
    real ``CancelledError`` grouped with an ``OSError``, so the task's cancellation
    request stays pending, as it does in practice.
``stream-close``
    Close the phase's event stream (``aclose()``, so ``GeneratorExit`` at the
    suspended ``yield``) after its ``k``-th event.
``crash``
    Simulate process death immediately before the ``k``-th store write: that
    write and every later store, provider and tool call of the dead app raise
    :class:`SimulatedCrash`, so no ``finally`` block or background task can
    persist anything more. Its leftover tasks are cancelled and the store is then
    handed to a fresh app that runs recovery.

After each fault the exploration checks the invariants in :mod:`.invariants`,
first in the still-live process (not after a crash) and then again after
recovery. A point the fault never reaches is reported, so a scenario that stops
being deterministic fails instead of silently exploring other points.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import os
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from tests.runtime.runtime_provider_support import VersionedFakeProvider

import cayu
from cayu.messages import Message, TextPart, ToolResultPart
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.memory import InMemorySessionStore
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec


class FaultKind(StrEnum):
    CANCEL = "cancel"
    GROUPED_CANCEL = "grouped-cancel"
    STREAM_CLOSE = "stream-close"
    CRASH = "crash"


class SimulatedCrash(BaseException):
    """The simulated process died; nothing it does afterwards may reach the store."""


# Store methods whose names start with these only read. Everything else is
# treated as a write, which at worst adds a crash point before a read.
_READ_PREFIXES = (
    "load",
    "read",
    "query",
    "list",
    "inspect",
    "lookup",
    "get",
    "count",
    "resolve",
    "scan",
    "validate",
    "public_authority_private_value_exists",
)


def is_store_read(name: str) -> bool:
    return name.lstrip("_").startswith(_READ_PREFIXES)


@dataclass(frozen=True)
class Boundary:
    site: str
    write: bool = False
    # The innermost Runtime function that crossed it; recorded in fault-free
    # runs only, where it groups points into distinct call sites.
    caller: str = ""


def point_label(site: str, occurrence: int) -> str:
    """A point is a boundary and its occurrence among equal ones: ``store.load:enter#3``."""

    return f"{site}#{occurrence}"


@dataclass
class Injector:
    """Labels the boundaries of the armed phase and fires one fault at ``target``.

    Points are addressed by label rather than position, so a call that is added
    or removed shifts only the later occurrences of its own boundary.
    """

    kind: FaultKind | None = None
    target: str | None = None
    armed: bool = False
    dead: bool = False
    driver: asyncio.Task[Any] | None = None
    trace: list[Boundary] = field(default_factory=list)
    fired: str | None = None
    _occurrences: dict[str, int] = field(default_factory=dict)

    def _counts(self, boundary: Boundary) -> bool:
        if self.kind is FaultKind.CRASH:
            return boundary.write
        return self.kind is not FaultKind.STREAM_CLOSE

    def next_label(self, site: str) -> str:
        occurrence = self._occurrences.get(site, 0)
        self._occurrences[site] = occurrence + 1
        return point_label(site, occurrence)

    async def boundary(self, site: str, *, write: bool = False) -> None:
        if self.dead:
            raise SimulatedCrash(site)
        if not self.armed:
            return
        if self.kind is None:
            boundary = Boundary(site, write, runtime_caller())
        else:
            boundary = Boundary(site, write)
            if not self._counts(boundary):
                return
        self.trace.append(boundary)
        label = self.next_label(site)
        if self.kind is None or label != self.target or self.fired is not None:
            return
        self.fired = label
        if self.kind is FaultKind.CANCEL:
            assert self.driver is not None
            self.driver.cancel()
            if asyncio.current_task() is self.driver:
                await asyncio.sleep(0)
        elif self.kind is FaultKind.GROUPED_CANCEL:
            assert self.driver is not None
            current = asyncio.current_task()
            self.driver.cancel()
            if current is not None and current is not self.driver:
                current.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as cancellation:
                raise BaseExceptionGroup(
                    "injected cleanup failure during cancellation",
                    [cancellation, OSError("injected cleanup failure")],
                ) from None
            raise AssertionError(f"the cancellation did not reach {site}")
        elif self.kind is FaultKind.CRASH:
            self.dead = True
            raise SimulatedCrash(site)


# The installed ``cayu`` package directory, wherever the checkout lives.
_RUNTIME_ROOT = os.path.dirname(os.path.abspath(cayu.__file__)) + os.sep
_RUNTIME_PARENT = os.path.dirname(_RUNTIME_ROOT.rstrip(os.sep))


def _runtime_path(filename: str) -> str | None:
    """``cayu/...`` for a file of the Runtime package, else ``None``."""

    path = os.path.abspath(filename)
    if not path.startswith(_RUNTIME_ROOT):
        return None
    return os.path.relpath(path, _RUNTIME_PARENT).replace(os.sep, "/")


def runtime_frames(frames: traceback.StackSummary) -> tuple[str, ...]:
    """The Runtime's own frames of a stack, innermost last, for failure reports."""

    return tuple(
        f"{path}:{frame.lineno} {frame.name}"
        for frame in frames
        if (path := _runtime_path(frame.filename)) is not None
    )


def runtime_caller() -> str:
    """The innermost Runtime function on the current stack, as ``path qualname``."""

    for frame, _ in traceback.walk_stack(None):
        path = _runtime_path(frame.f_code.co_filename)
        if path is not None:
            return f"{path} {frame.f_code.co_qualname}"
    return ""


_STORE_CALL_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "fault_injection_store_call_depth", default=0
)


# Task lease renewals run on a wall-clock timer, so whether one lands inside a
# scenario depends on the host's speed. They stay unnumbered to keep every trace
# deterministic, but a dead app still cannot renew.
_TIMER_DRIVEN_STORE_METHODS = frozenset({"heartbeat"})


def _boundary_method(name: str, method: Callable[..., Awaitable[Any]]):
    write = not is_store_read(name)
    numbered = name not in _TIMER_DRIVEN_STORE_METHODS

    async def wrapped(self, *args: Any, **kwargs: Any) -> Any:
        depth = _STORE_CALL_DEPTH.get()
        token = _STORE_CALL_DEPTH.set(depth + 1)
        try:
            if self.fault_injector.dead:
                raise SimulatedCrash(f"store.{name}")
            if depth or not numbered:
                return await method(self, *args, **kwargs)
            await self.fault_injector.boundary(f"store.{name}:enter", write=write)
            result = await method(self, *args, **kwargs)
            if write:
                await self.fault_injector.boundary(f"store.{name}:exit")
            return result
        finally:
            _STORE_CALL_DEPTH.reset(token)

    wrapped.__name__ = name
    wrapped.__qualname__ = name
    return wrapped


def _instrumented_store_type(base: type) -> type:
    """Subclass a store so every top-level coroutine method crosses a boundary.

    The overrides are plain functions on a class that redeclares the store's
    capability versions, so the Runtime still recognises its native protocols.
    """

    namespace: dict[str, Any] = {}
    for name in dir(base):
        if name.startswith("__"):
            continue
        attribute = inspect.getattr_static(base, name)
        if inspect.iscoroutinefunction(attribute):
            namespace[name] = _boundary_method(name, attribute)
        elif type(attribute) in (bool, int):
            # Capability flags must be declared at least as derived as the
            # methods they vouch for, so redeclare each one beside the overrides.
            namespace[name] = attribute
    return type(f"FaultInjecting{base.__name__}", (base,), namespace)


_FaultInjectingSessionStore = _instrumented_store_type(InMemorySessionStore)
_FaultInjectingTaskStore = _instrumented_store_type(InMemoryTaskStore)


class ManualClock:
    """A store ownership clock the harness moves past every lease before recovery."""

    def __init__(self) -> None:
        self._now = datetime.now(UTC)

    def __call__(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


def session_store(injector: Injector, clock: ManualClock) -> InMemorySessionStore:
    store = _FaultInjectingSessionStore(ownership_clock=clock)
    store.fault_injector = injector
    return store


def task_store(injector: Injector, clock: ManualClock) -> InMemoryTaskStore:
    store = _FaultInjectingTaskStore(clock=clock, ownership_clock=clock)
    store.fault_injector = injector
    return store


TOOLS_PROMPT = "fault-injection: call the parallel tools"
APPROVAL_PROMPT = "fault-injection: call the reviewed tool"
FINAL_TEXT = "fault-injection: done"


class ScriptedProvider(VersionedFakeProvider):
    """Answers from the request itself, so replays after recovery stay scripted.

    A request whose last message carries tool results gets the final text. A user
    message asking for tools gets two parallel ``echo`` calls, one asking for
    review gets one ``reviewed`` call, and anything else gets the final text.
    """

    def __init__(self, injector: Injector) -> None:
        super().__init__([])
        self.injector = injector

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        await self.injector.boundary("provider.request")
        for event in _scripted_response(request.messages):
            await self.injector.boundary(f"provider.{event.type}")
            yield event


def _scripted_response(messages: list[Message]) -> list[ModelStreamEvent]:
    last = messages[-1] if messages else None
    if last is not None and any(type(part) is ToolResultPart for part in last.content):
        prompt = None
    else:
        prompt = "".join(
            part.text
            for part in (last.content if last is not None else ())
            if type(part) is TextPart
        )
    if prompt == TOOLS_PROMPT:
        return [
            ModelStreamEvent.tool_call(id="call_a", name="echo", arguments={"text": "a"}),
            ModelStreamEvent.tool_call(id="call_b", name="echo", arguments={"text": "b"}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    if prompt == APPROVAL_PROMPT:
        return [
            ModelStreamEvent.tool_call(id="call_reviewed", name="reviewed", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    return [
        ModelStreamEvent.text_delta(FINAL_TEXT),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


_TOOL_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="tests:fault-injection:boundary-tool",
    behavior_version="1",
    implementation_version="1",
)


class BoundaryTool(Tool):
    """A replay-safe tool whose entry and exit are injection boundaries."""

    def __init__(self, name: str, injector: Injector) -> None:
        super().__init__(
            ToolSpec(
                name=name,
                description=f"Fault-injection tool {name}.",
                input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                effect=ToolEffect.NONE,
                max_terminal_payload_bytes=4096,
                execution_profile_identity=_TOOL_IDENTITY,
            )
        )
        self.injector = injector

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        await self.injector.boundary(f"tool.{self.spec.name}:enter")
        result = ToolResult(content=str(args.get("text", "ok")))
        await self.injector.boundary(f"tool.{self.spec.name}:exit")
        return result
