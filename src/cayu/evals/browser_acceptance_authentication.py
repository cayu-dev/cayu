"""Read-only, trial-scoped authentication samples from ordinary runtime hooks."""

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.evals.browser_acceptance_authenticated import BrowserAcceptanceAuthenticatedConfigV1
from cayu.runtime.hooks import BeforeToolCallHookContext, RuntimeHook, ToolCallHookContext


@dataclass(frozen=True, slots=True, repr=False)
class _AuthenticationSample:
    session_id: str
    instance_id: str
    run_epoch: int
    execution_profile: str
    tool_call_id: str
    operation_id: str
    operation: str
    arguments_sha256: str
    browser_session_id: str
    before: int
    after: int


class BrowserAcceptanceAuthenticationCollector(RuntimeHook):
    """Register at application construction, then supply the same object to the plan.

    The counter must measure successful requests for the exclusive designated test
    account/browser journey, not shared site traffic. Samples contain no page or
    credential content. They become evidence only after private durable readback.
    One collector owns at most one active trial and six sequential browser calls.
    """

    def __init__(self, counter: Callable[[], int], *, observer_revision: str) -> None:
        if not callable(counter):
            raise TypeError("Authentication collection requires a site-owned counter.")
        self.counter = counter
        self.observer_revision = BrowserAcceptanceAuthenticatedConfigV1.validate_scope(
            observer_revision
        )
        self._active = False
        self._invalid = False
        self._pending: tuple[str, int] | None = None
        self._samples: list[_AuthenticationSample] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="browser-acceptance-authentication",
            behavior_version=self.observer_revision,
            implementation_version="phase-sampling-v1",
        )

    def _count(self) -> int:
        value = self.counter()
        if type(value) is not int or not 0 <= value <= 1 << 20:
            raise ValueError("Authentication counter is unavailable or outside its bound.")
        return value

    def _begin(self) -> None:
        if self._active:
            raise ValueError("Authentication collector already owns a trial.")
        self._samples.clear()
        self._pending = None
        self._invalid = False
        self._active = True

    def _finish(self) -> tuple[_AuthenticationSample, ...]:
        result = tuple(self._samples) if not self._invalid and self._pending is None else ()
        self._active = False
        self._pending = None
        self._samples.clear()
        return result

    async def before_tool_call(self, context: BeforeToolCallHookContext) -> None:
        if not self._active:
            return
        if (
            context.tool_name != "browser_session"
            or self._pending is not None
            or len(self._samples) >= 6
        ):
            self._invalid = True
            return
        try:
            self._pending = (context.tool_call_id, self._count())
        except Exception:
            self._invalid = True

    async def after_tool_call(self, context: ToolCallHookContext) -> None:
        if not self._active or self._invalid:
            return
        pending, self._pending = self._pending, None
        if pending is None or pending[0] != context.tool_call_id:
            self._invalid = True
            return
        try:
            arguments, result, session = context.arguments, context.result, context.session
            operation, operation_id = arguments.get("operation"), arguments.get("operation_id")
            browser_id = None if result.structured is None else result.structured.get("session_id")
            if (
                context.tool_name != "browser_session"
                or result.is_error
                or type(operation) is not str
                or type(operation_id) is not str
                or type(browser_id) is not str
                or context.execution_profile is None
            ):
                self._invalid = True
                return
            self._samples.append(
                _AuthenticationSample(
                    session.id,
                    session.instance_id,
                    session.run_epoch,
                    context.execution_profile.fingerprint,
                    context.tool_call_id,
                    operation_id,
                    operation,
                    sha256(
                        canonical_durable_json_bytes(arguments, "browser authentication arguments")
                    ).hexdigest(),
                    browser_id,
                    pending[1],
                    self._count(),
                )
            )
        except Exception:
            # Sampling cannot change tool execution or expose observer diagnostics.
            self._invalid = True
