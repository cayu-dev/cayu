"""An app's own post-deploy check, behind the operator login.

A deployed app often has checks of its own that only it can run: that its AI gateway answers,
that one live workflow completes. ``OperatorCheck`` collects them as named steps, runs them in
order, and serves the report on one route that only the operator login opens::

    from cayu.server.operator_check import CheckFailed, OperatorCheck

    check = OperatorCheck()

    @check.step("ai_gateway")
    async def ai_gateway() -> str:
        reply = await gateway.test_connection()
        if not reply.ok:
            raise CheckFailed(f"The gateway answered {reply.status}.")
        return f"Answered in {reply.elapsed_ms} ms"

    app.include_router(check.router())   # POST /operator-check

Declare the same route in the root ``cayu-cloud.toml`` and Cayu Cloud can run it after a deploy
(``cayu cloud service check --application NAME``), whatever the app's own sign-in::

    [operator_check]
    path = "/operator-check"

The route answers ``200`` with ``{"ok": true, "summary": ..., "checks": [...]}`` when every step
passes and ``503`` with ``"ok": false`` when any fails. Each step is a function without arguments,
plain or ``async``, that returns ``None`` or a short detail; raising ``CheckFailed`` fails it with
that message, and any other exception fails it with only the exception's type (its traceback goes
to the app's log, which is where secrets and stack details belong). Steps run one after another,
and a failed step doesn't stop the next.

By default the route accepts only the operator login Cayu Cloud injects
(``CAYU_OPERATOR_USERNAME`` and ``CAYU_OPERATOR_PASSWORD``, ``BasicAuth.from_environment``), so it
refuses an app's own users and the people Cloud signs in, and building it without those variables
raises ``AuthConfigurationError`` instead of serving an open check. Pass ``auth=`` any server auth
dependency to protect it another way. ``run()`` gives the same report without a route, for a
command or a test.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from fastapi import APIRouter

    from cayu.server.auth import AuthDependency

__all__ = [
    "DEFAULT_OPERATOR_CHECK_PATH",
    "CheckFailed",
    "OperatorCheck",
    "OperatorCheckReport",
    "OperatorCheckStepResult",
]

_LOG = logging.getLogger(__name__)

#: The route ``OperatorCheck.router()`` serves unless told otherwise.
DEFAULT_OPERATOR_CHECK_PATH = "/operator-check"
#: What one step's detail keeps; Cayu Cloud keeps no more than this of it either.
MAX_DETAIL_CHARS = 500
_MAX_NAME_CHARS = 100

Step = Callable[[], "str | None | Awaitable[str | None]"]


class CheckFailed(Exception):
    """Raised by a step to fail it; the message is shown to the operator."""


@dataclass(frozen=True)
class OperatorCheckStepResult:
    name: str
    ok: bool
    detail: str | None
    duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "detail": self.detail,
            "duration_ms": self.duration_ms,
            "name": self.name,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class OperatorCheckReport:
    checks: tuple[OperatorCheckStepResult, ...]

    @property
    def ok(self) -> bool:
        """Every step passed; a check without steps checks nothing, so it doesn't pass."""

        return bool(self.checks) and all(step.ok for step in self.checks)

    @property
    def summary(self) -> str:
        if not self.checks:
            return "No checks are registered."
        passed = sum(step.ok for step in self.checks)
        return f"{passed} of {len(self.checks)} checks passed."

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [step.to_dict() for step in self.checks],
            "ok": self.ok,
            "summary": self.summary,
        }


def _detail(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= MAX_DETAIL_CHARS else text[: MAX_DETAIL_CHARS - 3] + "..."


class OperatorCheck:
    """Named check steps, run in order, served on one operator-only route."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._steps: list[tuple[str, Step]] = []
        self._clock = clock

    def add(self, name: str, step: Step) -> None:
        """Add a step; ``name`` is what the report and ``cayu cloud service check`` show."""

        if type(name) is not str or not name or name != name.strip() or len(name) > _MAX_NAME_CHARS:
            raise ValueError(f"A check step name is 1 to {_MAX_NAME_CHARS} characters.")
        if any(existing == name for existing, _ in self._steps):
            raise ValueError(f"A check step named {name!r} is already added.")
        if not callable(step):
            raise TypeError("A check step must be a function without arguments.")
        self._steps.append((name, step))

    def step(self, name: str) -> Callable[[Step], Step]:
        """``add`` as a decorator; the function is returned unchanged."""

        def register(function: Step) -> Step:
            self.add(name, function)
            return function

        return register

    async def run(self) -> OperatorCheckReport:
        """Run every step once, in the order they were added."""

        from starlette.concurrency import run_in_threadpool

        results = []
        for name, step in self._steps:
            started = self._clock()
            try:
                if inspect.iscoroutinefunction(step):
                    value = await step()
                else:
                    value = await run_in_threadpool(step)
                    if inspect.isawaitable(value):
                        value = await value
                ok, detail = True, _detail(value)
            except CheckFailed as exc:
                ok, detail = False, _detail(str(exc) or "Failed.")
            except Exception as exc:
                _LOG.exception("Operator check step %r failed", name)
                ok, detail = False, f"{type(exc).__name__} (see the app's log)."
            results.append(
                OperatorCheckStepResult(
                    name=name,
                    ok=ok,
                    detail=detail,
                    duration_ms=max(0, int((self._clock() - started) * 1000)),
                )
            )
        return OperatorCheckReport(checks=tuple(results))

    def router(
        self,
        *,
        path: str = DEFAULT_OPERATOR_CHECK_PATH,
        method: Literal["POST", "GET"] = "POST",
        auth: AuthDependency | None = None,
    ) -> APIRouter:
        """A router serving the check at ``method path``, behind ``auth``.

        ``auth`` defaults to the operator login from ``CAYU_OPERATOR_USERNAME`` and
        ``CAYU_OPERATOR_PASSWORD``, read now. Match ``path`` and ``method`` to
        ``[operator_check]`` in ``cayu-cloud.toml``.
        """

        from fastapi import APIRouter, Depends
        from fastapi.responses import JSONResponse

        from cayu.server.auth import BasicAuth, server_auth_dependency

        if method not in {"POST", "GET"}:
            raise ValueError('The operator check method is "POST" or "GET".')
        if type(path) is not str or not path.startswith("/") or path.startswith("//"):
            raise ValueError('The operator check path is a path on this host, starting with "/".')
        dependency = server_auth_dependency(BasicAuth.from_environment() if auth is None else auth)
        router = APIRouter()

        async def operator_check() -> JSONResponse:
            report = await self.run()
            return JSONResponse(
                report.to_dict(),
                status_code=200 if report.ok else 503,
                headers={"Cache-Control": "no-store"},
            )

        router.add_api_route(
            path,
            operator_check,
            methods=[method],
            dependencies=[Depends(dependency)],
            include_in_schema=False,
        )
        return router
