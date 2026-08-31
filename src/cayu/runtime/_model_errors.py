from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from threading import Lock
from typing import Any, cast
from weakref import WeakKeyDictionary

from cayu._exception_groups import (
    exception_tree_contains,
    iter_exception_tree,
    rebuild_exception_group,
)
from cayu._exception_state import exception_state, set_exception_state
from cayu._validation import (
    DurableValueError,
    copy_durable_json_object,
    extract_durable_value_error,
    require_clean_nonblank,
    require_durable_text,
    safe_durable_value_error_details,
)
from cayu.core.billing import (
    BillingIdentity,
    completed_billing_identity,
    copy_billing_identity,
)
from cayu.providers.base import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
)
from cayu.runtime.workspace_observation_recovery import (
    copy_workspace_observation_pending_cancellation_requests,
)
from cayu.tools._operation_boundary import (
    BoundedInvocationOperationRegistry,
    InvocationOperationCapacityError,
)

_PROVIDER_ERROR_PAYLOAD_KEYS = frozenset(
    {
        "status_code",
        "provider_error_type",
        "provider_error_code",
        "request_id",
        "retryable",
        "retry_after_s",
        "context_overflow",
    }
)
_PROVIDER_ERROR_TYPE_MARKERS = frozenset({"ModelProviderError", "ModelContextOverflowError"})
_BILLING_IDENTITY_ERROR_MESSAGE = "Model provider billing identity resolution failed"
_BILLING_IDENTITY_ERROR_TYPE = "BillingIdentityResolutionError"
_BILLING_IDENTITY_ERROR_CODE = "billing_identity_resolution_failed"
_BILLING_CANCELLATION_FAILURES_STATE = "_cayu_billing_cancellation_failures"
_BILLING_CANCELLATION_FAILURES_TOKEN = object()
_MAX_RETAINED_BILLING_HOOKS = 1_024
_BILLING_HOOK_REGISTRIES: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    BoundedInvocationOperationRegistry,
] = WeakKeyDictionary()
_BILLING_HOOK_REGISTRIES_LOCK = Lock()


class _BillingIdentityResolutionCancelled(asyncio.CancelledError):
    """Internal marker carried only until a provider-free public boundary."""


class _FallbackBillingCancellationStateCheckFailed(RuntimeError):
    """Private signal raised after fallback provider state has been released."""


@dataclass(frozen=True, slots=True)
class _BillingCancellationFailures:
    message: str
    failures: tuple[dict[str, Any], ...]
    token: object


@dataclass(frozen=True, slots=True)
class _BillingHookOutcome:
    result: BillingIdentity | None = None
    error: BaseException | None = None


def _shared_billing_hook_registry() -> BoundedInvocationOperationRegistry:
    """Return one loop-owned bound for hooks retained after supervisory exit."""

    loop = asyncio.get_running_loop()
    with _BILLING_HOOK_REGISTRIES_LOCK:
        registry = _BILLING_HOOK_REGISTRIES.get(loop)
        if registry is None:
            registry = BoundedInvocationOperationRegistry(
                max_operations=_MAX_RETAINED_BILLING_HOOKS,
            )
            _BILLING_HOOK_REGISTRIES[loop] = registry
        return registry


def detach_billing_identity_cancellation(
    exc: asyncio.CancelledError,
) -> asyncio.CancelledError | None:
    """Return a fresh public cancellation for an internal billing-hook marker."""

    if not isinstance(exc, _BillingIdentityResolutionCancelled):
        return None
    handoff = exception_state(exc, _BILLING_CANCELLATION_FAILURES_STATE)
    if (
        type(handoff) is not _BillingCancellationFailures
        or handoff.token is not _BILLING_CANCELLATION_FAILURES_TOKEN
        or type(handoff.message) is not str
    ):
        return asyncio.CancelledError("Model provider billing identity resolution cancelled")
    return asyncio.CancelledError(handoff.message)


def billing_identity_cancellation_failures(
    exc: asyncio.CancelledError,
) -> tuple[dict[str, Any], ...]:
    """Return authenticated, content-free billing cleanup diagnostics."""

    if not isinstance(exc, _BillingIdentityResolutionCancelled):
        return ()
    handoff = exception_state(exc, _BILLING_CANCELLATION_FAILURES_STATE)
    if (
        type(handoff) is not _BillingCancellationFailures
        or handoff.token is not _BILLING_CANCELLATION_FAILURES_TOKEN
        or type(handoff.message) is not str
        or type(handoff.failures) is not tuple
        or len(handoff.failures) > 1
    ):
        return ()
    copied: list[dict[str, Any]] = []
    for failure in handoff.failures:
        if type(failure) is not dict or set(failure) != {"phase", "error", "error_type"}:
            return ()
        if failure.get("phase") != "billing_identity_for_request":
            return ()
        error = failure.get("error")
        error_type = failure.get("error_type")
        if type(error) is not str or type(error_type) is not str:
            return ()
        copied.append(dict(failure))
    return tuple(copied)


def detach_billing_identity_cancellation_group(
    exc: BaseExceptionGroup,
) -> BaseExceptionGroup | None:
    """Return a fresh, provider-free public group when a billing marker is present.

    A lifecycle finalizer can aggregate the private billing cancellation with a
    cleanup failure. Reusing that group, or any of its original leaves, would
    retain their traceback frames and provider-bearing locals. Preserve only
    the shape and cancellation/cleanup distinction in a fresh exception tree.
    """

    if not _contains_billing_identity_cancellation(exc):
        return None
    return _detach_billing_identity_cancellation_tree(exc)


def _contains_billing_identity_cancellation(exc: BaseException) -> bool:
    return exception_tree_contains(exc, _BillingIdentityResolutionCancelled)


def _detach_billing_identity_cancellation_tree(
    exc: BaseExceptionGroup,
) -> BaseExceptionGroup:
    detached = rebuild_exception_group(
        exc,
        group_message=(
            "Model provider billing identity resolution cancellation had concurrent failures"
        ),
        leaf_mapper=_detached_concurrent_billing_failure,
        invalid_leaf_factory=lambda: RuntimeError(
            "Concurrent failure during model provider billing identity resolution"
        ),
    )
    copy_workspace_observation_pending_cancellation_requests(exc, detached)
    return detached


def _detached_concurrent_billing_failure(exc: BaseException) -> BaseException:
    if isinstance(exc, asyncio.CancelledError):
        detached = detach_billing_identity_cancellation(exc)
        if detached is None:
            return RuntimeError(
                "Model provider billing identity hook cancelled itself during concurrent failure"
            )
        copy_workspace_observation_pending_cancellation_requests(exc, detached)
        return detached
    if isinstance(exc, KeyboardInterrupt):
        return KeyboardInterrupt(
            "Concurrent keyboard interrupt during model provider billing identity resolution"
        )
    if isinstance(exc, SystemExit):
        return _detached_system_exit(
            exc,
            fallback=("Concurrent system exit during model provider billing identity resolution"),
        )
    if isinstance(exc, GeneratorExit):
        return GeneratorExit(
            "Concurrent generator exit during model provider billing identity resolution"
        )
    del exc
    return RuntimeError("Concurrent failure during model provider billing identity resolution")


async def resolve_request_billing_identity(
    provider: ModelProvider,
    request: ModelRequest,
    *,
    provider_name: str,
) -> BillingIdentity | None:
    """Resolve and validate request billing identity across a secret-free boundary.

    Provider hooks may resolve or refresh credentials before returning.  Their
    exceptions, cancellation objects, and invalid return values are therefore
    untrusted credential-bearing state.  Convert every failure to a fresh,
    content-free exception and raise it only after the provider traceback has
    been released.
    """

    owner_task = asyncio.current_task()
    cancellation_baseline = 0 if owner_task is None else owner_task.cancelling()
    supplied_identity: BillingIdentity | None = None
    cancellation: asyncio.CancelledError | None = None
    failure: ModelProviderError | None = None
    fatal_failure: BaseException | None = None
    hook_outcome: _BillingHookOutcome | None = None
    caller_cancellation: asyncio.CancelledError | None = None
    supervisory_controls: tuple[BaseException, ...] = ()
    hook_returned = False
    try:
        # Deliver already-pending cancellation inside the secret-safe cleanup
        # boundary. A handled historical request may leave ``cancelling()``
        # non-zero, but must not authenticate a later provider-created signal.
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as exc:
            cancellation = _billing_identity_cancellation(exc)
        if cancellation is None:
            owner_task = asyncio.current_task()
            cancellation_baseline = 0 if owner_task is None else owner_task.cancelling()
            (
                hook_outcome,
                caller_cancellation,
                supervisory_controls,
            ) = await _run_request_billing_hook(
                provider,
                request,
                cancellation_baseline=cancellation_baseline,
            )
            hook_returned = hook_outcome.error is None
        if cancellation is not None:
            pass
        elif hook_outcome is None:  # pragma: no cover - assigned by the owned hook runner
            raise AssertionError("Billing hook outcome was not captured.")
        elif supervisory_controls:
            fatal_failure = _billing_supervisory_failure(
                supervisory_controls,
                hook_failure=hook_outcome.error,
                caller_cancellation=caller_cancellation,
            )
        elif caller_cancellation is not None:
            if _billing_hook_contains_process_control(hook_outcome.error):
                if hook_outcome.error is None:  # pragma: no cover - guarded above
                    raise AssertionError("Billing process-control evidence was lost.")
                fatal_failure = _billing_supervisory_failure(
                    (),
                    hook_failure=hook_outcome.error,
                    caller_cancellation=caller_cancellation,
                )
            else:
                cancellation = _billing_identity_cancellation(
                    caller_cancellation,
                    failures=_billing_cancellation_failure_diagnostics(hook_outcome.error),
                )
        elif hook_outcome.error is None:
            try:
                supplied_identity = copy_billing_identity(hook_outcome.result)
            except Exception as exc:
                failure = _billing_identity_error(
                    exc,
                    provider_name=provider_name,
                    direct_hook_failure=False,
                )
            except BaseException as exc:
                fatal_failure = _detached_billing_hook_failure(exc)
        elif isinstance(hook_outcome.error, BaseExceptionGroup):
            cancellation, failure, fatal_failure = _classify_billing_hook_group(
                hook_outcome.error,
                provider_name=provider_name,
            )
        elif isinstance(hook_outcome.error, asyncio.CancelledError):
            # A child/provider-created cancellation is not caller authority.
            failure = _billing_identity_error(
                RuntimeError("Provider billing identity hook cancelled itself."),
                provider_name=provider_name,
                direct_hook_failure=True,
            )
        elif isinstance(hook_outcome.error, Exception):
            failure = _billing_identity_error(
                hook_outcome.error,
                provider_name=provider_name,
                direct_hook_failure=not hook_returned,
            )
        else:
            fatal_failure = _detached_billing_hook_failure(hook_outcome.error)
    finally:
        hook_outcome = None
        caller_cancellation = None
        supervisory_controls = ()
        owner_task = None
        del provider
        del request
    if fatal_failure is not None:
        supplied_identity = None
        raise fatal_failure from None
    if cancellation is not None:
        supplied_identity = None
        raise cancellation from None
    if failure is not None:
        supplied_identity = None
        raise failure from None
    return supplied_identity


async def _run_request_billing_hook(
    provider: ModelProvider,
    request: ModelRequest,
    *,
    cancellation_baseline: int,
) -> tuple[
    _BillingHookOutcome,
    asyncio.CancelledError | None,
    tuple[BaseException, ...],
]:
    """Run one extension hook in an owned task and authenticate caller cancellation."""

    async def capture_hook_outcome() -> _BillingHookOutcome:
        try:
            return _BillingHookOutcome(result=await provider.billing_identity_for_request(request))
        except BaseException as exc:
            return _BillingHookOutcome(error=exc)
        finally:
            # This child owns the cancellation used to stop its hook. Python's
            # TaskGroup may retain that request after converting child cleanup
            # failure into an ExceptionGroup; consume it only on the child so
            # the caller task's cancellation authority remains untouched.
            child_task = asyncio.current_task()
            if child_task is not None:
                while child_task.cancelling() > 0:
                    child_task.uncancel()

    registry = _shared_billing_hook_registry()
    if not registry.reserve():
        return (
            _BillingHookOutcome(
                error=InvocationOperationCapacityError(
                    "Model provider billing identity hook capacity is exhausted."
                )
            ),
            None,
            (),
        )
    try:
        hook_task = asyncio.create_task(
            capture_hook_outcome(),
            name="cayu-provider-billing-identity-hook",
        )
    except BaseException:
        registry.release_reservation()
        raise
    registry.track(hook_task)
    caller_cancellation: asyncio.CancelledError | None = None
    try:
        while True:
            try:
                outcome = await asyncio.shield(hook_task)
                return outcome, caller_cancellation, ()
            except asyncio.CancelledError as exc:
                owner_task = asyncio.current_task()
                if owner_task is None or owner_task.cancelling() <= cancellation_baseline:
                    raise
                if caller_cancellation is None:
                    caller_cancellation = exc
                if hook_task.done() and not hook_task.cancelled():
                    return hook_task.result(), caller_cancellation, ()
                # The billing hook can own structured child cleanup (notably an
                # asyncio.TaskGroup). Deliver cancellation to that owner, then
                # keep waiting for its exact terminal outcome so concurrent
                # cleanup failures remain observable alongside caller authority.
                hook_task.cancel(*_safe_cancellation_args(exc))
            except BaseException as control:
                hook_task.cancel("Model provider billing identity resolution terminated")
                detached_control = _detached_billing_hook_failure(control)
                control = None
                raise detached_control from None
    finally:
        hook_task = None


def _billing_supervisory_failure(
    controls: tuple[BaseException, ...],
    *,
    hook_failure: BaseException | None,
    caller_cancellation: asyncio.CancelledError | None,
) -> BaseException:
    """Detach retained supervisory control after the provider hook settles."""

    failures = [_detached_billing_hook_failure(control) for control in controls]
    if hook_failure is not None and not isinstance(hook_failure, asyncio.CancelledError):
        failures.append(_detached_billing_hook_failure(hook_failure))
    if caller_cancellation is not None:
        failures.append(_billing_identity_cancellation(caller_cancellation))
    if len(failures) == 1:
        return failures[0]
    return BaseExceptionGroup(
        "Model provider billing identity resolution terminated with concurrent failures",
        failures,
    )


def _billing_cancellation_failure_diagnostics(
    failure: BaseException | None,
) -> tuple[dict[str, Any], ...]:
    if failure is None:
        return ()
    saw_ordinary_failure = any(
        isinstance(candidate, Exception) and not isinstance(candidate, asyncio.CancelledError)
        for candidate in iter_exception_tree(failure)
        if not isinstance(candidate, BaseExceptionGroup)
    )
    if not saw_ordinary_failure:
        return ()
    return (
        {
            "phase": "billing_identity_for_request",
            "error": "Provider billing identity cleanup failed during caller cancellation.",
            "error_type": "BillingIdentityCleanupError",
        },
    )


def _billing_hook_contains_process_control(failure: BaseException | None) -> bool:
    if failure is None:
        return False
    return any(
        isinstance(candidate, (GeneratorExit, KeyboardInterrupt, SystemExit))
        for candidate in iter_exception_tree(failure)
        if not isinstance(candidate, BaseExceptionGroup)
    )


def resolve_completion_billing_identity(
    provider: ModelProvider,
    request_identity: BillingIdentity | None,
    completed_payload: dict[str, Any],
    *,
    provider_name: str,
) -> BillingIdentity | None:
    """Resolve and validate completion billing identity without retaining hook state."""

    request_snapshot: BillingIdentity | None = None
    hook_identity: BillingIdentity | None = None
    hook_payload: dict[str, Any] | None = None
    completed_identity: BillingIdentity | None = None
    cancellation: asyncio.CancelledError | None = None
    failure: ModelProviderError | None = None
    fatal_failure: BaseException | None = None
    hook_invoked = False
    hook_returned = False
    try:
        # Frozen Pydantic models are an API guard, not a trust boundary. Keep
        # one pristine runtime-owned snapshot and expose only separate copies
        # to provider code so deliberate mutation cannot rewrite accounting
        # identity or completed model evidence.
        request_snapshot = copy_billing_identity(request_identity)
        hook_identity = copy_billing_identity(request_snapshot)
        hook_payload = copy_durable_json_object(
            completed_payload,
            "model_completed.payload",
        )
        hook_invoked = True
        completed_identity = provider.billing_identity_for_completion(
            hook_identity,
            hook_payload,
        )
        hook_returned = True
        completed_identity = completed_billing_identity(
            request_snapshot,
            completed_identity,
        )
    except BaseExceptionGroup as exc:
        cancellation, failure, fatal_failure = _classify_billing_hook_group(
            exc,
            provider_name=provider_name,
        )
    except asyncio.CancelledError:
        failure = ModelProviderError(
            _BILLING_IDENTITY_ERROR_MESSAGE,
            provider=provider_name,
            error_type=_BILLING_IDENTITY_ERROR_TYPE,
            error_code=_BILLING_IDENTITY_ERROR_CODE,
            retryable=False,
        )
    except Exception as exc:
        failure = _billing_identity_error(
            exc,
            provider_name=provider_name,
            direct_hook_failure=hook_invoked and not hook_returned,
        )
    except BaseException as exc:
        fatal_failure = _detached_billing_hook_failure(exc)
    finally:
        del provider
        del request_identity
        del completed_payload
        del request_snapshot
        del hook_identity
        del hook_payload
    if fatal_failure is not None:
        completed_identity = None
        raise fatal_failure from None
    if cancellation is not None:
        completed_identity = None
        raise cancellation from None
    if failure is not None:
        completed_identity = None
        raise failure from None
    return completed_identity


def _classify_billing_hook_group(
    exc: BaseExceptionGroup,
    *,
    provider_name: str,
) -> tuple[
    asyncio.CancelledError | None,
    ModelProviderError | None,
    BaseException | None,
]:
    """Classify one untrusted group into detached runtime-owned outcomes."""

    leaves = (
        candidate
        for candidate in iter_exception_tree(exc)
        if not isinstance(candidate, BaseExceptionGroup)
    )
    saw_cancellation = False
    saw_ordinary_failure = False
    saw_fatal_failure = False
    for candidate in leaves:
        if isinstance(candidate, asyncio.CancelledError):
            saw_cancellation = True
        elif isinstance(candidate, (GeneratorExit, KeyboardInterrupt, SystemExit)):
            saw_fatal_failure = True
        elif isinstance(candidate, Exception):
            saw_ordinary_failure = True
        else:
            saw_fatal_failure = True
    if saw_fatal_failure:
        return None, None, _detached_billing_hook_failure(exc)
    del saw_ordinary_failure, saw_cancellation
    return (
        None,
        ModelProviderError(
            _BILLING_IDENTITY_ERROR_MESSAGE,
            provider=provider_name,
            error_type=_BILLING_IDENTITY_ERROR_TYPE,
            error_code=_BILLING_IDENTITY_ERROR_CODE,
            retryable=False,
        ),
        None,
    )


def _detached_billing_hook_failure(exc: BaseException) -> BaseException:
    """Return fatal billing-hook control flow without provider-owned state."""

    if isinstance(exc, BaseExceptionGroup):
        return rebuild_exception_group(
            exc,
            group_message="Model provider billing identity resolution terminated",
            leaf_mapper=_detached_billing_hook_failure,
            invalid_leaf_factory=lambda: RuntimeError(
                "Model provider billing identity resolution failed"
            ),
        )
    if isinstance(exc, asyncio.CancelledError):
        detached = detach_billing_identity_cancellation(exc)
        if detached is not None:
            return detached
        return RuntimeError("Model provider billing identity hook cancelled itself")
    if isinstance(exc, KeyboardInterrupt):
        return KeyboardInterrupt("Model provider billing identity resolution interrupted")
    if isinstance(exc, SystemExit):
        return _detached_system_exit(
            exc,
            fallback="Model provider billing identity resolution exited",
        )
    if isinstance(exc, GeneratorExit):
        return GeneratorExit("Model provider billing identity resolution terminated")
    if isinstance(exc, Exception):
        return RuntimeError("Model provider billing identity resolution failed")
    return BaseException("Model provider billing identity resolution failed")


def _detached_system_exit(exc: SystemExit, *, fallback: str) -> SystemExit:
    """Preserve only a process-safe integer/None exit code."""

    try:
        args = BaseException.__dict__["args"].__get__(exc, BaseException)
    except BaseException:
        args = ()
    if type(args) is tuple and len(args) == 1:
        (code,) = cast("tuple[object]", args)
        if type(code) is int or code is None:
            return SystemExit(code)
    return SystemExit(fallback)


def _billing_identity_error(
    exc: Exception,
    *,
    provider_name: str,
    direct_hook_failure: bool,
) -> ModelProviderError:
    source: ModelProviderError | None = None
    portability_error = extract_durable_value_error(exc) if direct_hook_failure else None
    if portability_error is None:
        try:
            if isinstance(exc, ModelProviderError):
                source = copy_model_provider_error_control(exc)
            else:
                require_durable_text(str(exc), "provider_error.message")
        except DurableValueError as error:
            portability_error = error
        except Exception:
            portability_error = DurableValueError("invalid_json_type", "provider_error")
    if portability_error is not None:
        safe_error, _ = nonportable_model_provider_error(
            portability_error,
            fallback_provider=provider_name,
        )
        return safe_error
    return ModelProviderError(
        _BILLING_IDENTITY_ERROR_MESSAGE,
        provider=provider_name,
        status_code=source.status_code if source is not None else None,
        error_type=_BILLING_IDENTITY_ERROR_TYPE,
        error_code=_BILLING_IDENTITY_ERROR_CODE,
        retryable=source.retryable if source is not None else False,
        retry_after_s=source.retry_after_s if source is not None else None,
    )


def _safe_cancellation_args(exc: asyncio.CancelledError) -> tuple[str]:
    try:
        args = BaseException.__dict__["args"].__get__(exc, BaseException)
    except BaseException:
        args = ()
    if type(args) is tuple and len(args) == 1:
        (message,) = cast("tuple[object]", args)
        if type(message) is str:
            try:
                return (require_durable_text(message, "provider cancellation message"),)
            except (TypeError, ValueError):
                pass
    return ("Model provider billing identity resolution cancelled",)


def _billing_identity_cancellation(
    source: asyncio.CancelledError | None = None,
    *,
    failures: tuple[dict[str, Any], ...] = (),
) -> asyncio.CancelledError:
    cancellation = _BillingIdentityResolutionCancelled(
        *(
            _safe_cancellation_args(source)
            if source is not None
            else ("Model provider billing identity resolution cancelled",)
        )
    )
    set_exception_state(
        cancellation,
        _BILLING_CANCELLATION_FAILURES_STATE,
        _BillingCancellationFailures(
            message=_safe_cancellation_args(cancellation)[0],
            failures=tuple(dict(failure) for failure in failures[:1]),
            token=_BILLING_CANCELLATION_FAILURES_TOKEN,
        ),
    )
    return cancellation


_NON_PORTABLE_PROVIDER_ERROR_MESSAGE = "Model provider emitted a non-portable error value."


@dataclass(frozen=True)
class ProviderExceptionControl:
    """One detached, portable snapshot of a provider-owned exception."""

    message: str
    error_type: str
    cause: Exception


_DETACHABLE_BUILTIN_PROVIDER_ERRORS = frozenset(
    {
        Exception,
        RuntimeError,
        ValueError,
        TypeError,
        AssertionError,
        OSError,
        TimeoutError,
        ConnectionError,
        BrokenPipeError,
        ConnectionAbortedError,
        ConnectionRefusedError,
        ConnectionResetError,
    }
)


def copy_provider_exception_control(error: Exception) -> ProviderExceptionControl:
    """Capture portable provider failure control data exactly once.

    Generic exceptions remain supported for custom providers. Known built-in
    exception types are reconstructed from the captured message; custom types
    collapse to ``RuntimeError`` so provider-controlled ``__str__`` methods and
    mutable state cannot be reread by retry or durable publication code.
    """

    if not isinstance(error, Exception):
        raise TypeError("error must be an Exception.")
    if isinstance(error, ModelProviderError):
        copied = copy_model_provider_error_control(error)
        return ProviderExceptionControl(
            message=str(copied),
            error_type=type(copied).__name__,
            cause=copied,
        )
    try:
        message = require_durable_text(str(error), "provider_error.message")
        error_type = require_durable_text(type(error).__name__, "provider_error.type")
        if type(error) is ExceptionGroup:
            # Preserve ordinary-exception-group control flow without retaining
            # provider-owned children.  A single constant child is sufficient
            # for the built-in group invariant and cannot leak mutable provider
            # diagnostics through a later traceback or terminal event.
            group_message = require_durable_text(
                error.message,
                "provider_error.group_message",
            )
            detached = ExceptionGroup(
                group_message,
                [RuntimeError("Provider exception details were detached.")],
            )
        else:
            detached_type = (
                type(error) if type(error) in _DETACHABLE_BUILTIN_PROVIDER_ERRORS else RuntimeError
            )
            detached = detached_type(message)
    except DurableValueError:
        raise
    except Exception:
        raise DurableValueError("invalid_json_type", "provider_error") from None
    return ProviderExceptionControl(
        message=message,
        error_type=error_type,
        cause=detached,
    )


def copy_model_provider_error_control(error: ModelProviderError) -> ModelProviderError:
    """Snapshot the provider-error fields used by retry and durable events.

    ``ModelProviderError`` is an ephemeral provider-owned object. Its public
    attributes remain mutable after construction, and subclasses can override
    ``error_payload_fields``. Build the candidate payload with the base-class
    method so retry control data is validated without invoking provider code.
    ``response_body`` is deliberately excluded because the runtime never
    publishes it or uses it to classify retries. The returned base exception is
    runtime-owned, so later provider mutation cannot change retry or publication
    decisions. Context-overflow identity is the one subtype retained because it
    controls bounded context recovery.
    """

    if not isinstance(error, ModelProviderError):
        raise TypeError("error must be a ModelProviderError.")
    try:
        message = require_durable_text(str(error), "provider_error.message")
        require_durable_text(type(error).__name__, "provider_error.type")
        payload = copy_durable_json_object(
            ModelProviderError.error_payload_fields(error),
            "provider_error",
        )
        error_kwargs = {
            "provider": payload["provider"],
            "status_code": payload.get("status_code"),
            "error_type": payload.get("provider_error_type"),
            "error_code": payload.get("provider_error_code"),
            "request_id": payload.get("request_id"),
        }
        if isinstance(error, ModelContextOverflowError):
            if payload.get("retryable") is not False:
                raise ValueError("Context-overflow errors must not be retryable.")
            return ModelContextOverflowError(message, **error_kwargs)
        return ModelProviderError(
            message,
            **error_kwargs,
            retryable=payload.get("retryable"),
            retry_after_s=payload.get("retry_after_s"),
        )
    except DurableValueError:
        raise
    except Exception:
        # Provider-owned exception implementations and post-construction
        # mutation must not leak through diagnostics or drive retry decisions.
        raise DurableValueError("invalid_json_type", "provider_error") from None


def copy_provider_hook_error_control(
    error: Exception,
    *,
    fallback_provider: str,
    generic_error_code: str,
) -> tuple[ModelProviderError, dict[str, str]]:
    """Detach an exception raised by provider-owned request/completion hooks.

    Public ``DurableValueError`` instances are also caller-controlled: their
    diagnostic attributes and field label can be forged or mutated. Detect
    them before rendering, and collapse every non-portable hook failure to the
    same constant terminal error plus allowlisted code/path diagnostics.
    """

    if not isinstance(error, Exception):
        raise TypeError("error must be an Exception.")
    portability_error = extract_durable_value_error(error)
    if portability_error is not None:
        return nonportable_model_provider_error(
            portability_error,
            fallback_provider=fallback_provider,
        )
    try:
        if isinstance(error, ModelProviderError):
            copied = copy_model_provider_error_control(error)
        else:
            control = copy_provider_exception_control(error)
            copied = copy_model_provider_error_control(
                ModelProviderError(
                    control.message,
                    provider=fallback_provider,
                    error_type=control.error_type,
                    error_code=generic_error_code,
                    retryable=False,
                )
            )
    except DurableValueError as exc:
        return nonportable_model_provider_error(
            exc,
            fallback_provider=fallback_provider,
        )
    return copied, {}


def nonportable_model_provider_error(
    error: DurableValueError,
    *,
    fallback_provider: str,
) -> tuple[ModelProviderError, dict[str, str]]:
    """Return a runtime-owned terminal error and safe durable diagnostics."""

    code, path = safe_durable_value_error_details(error)
    try:
        provider = require_clean_nonblank(
            require_durable_text(fallback_provider, "provider"),
            "provider",
        )
    except (DurableValueError, ValueError):
        provider = "unknown"
    safe_error = ModelProviderError(
        _NON_PORTABLE_PROVIDER_ERROR_MESSAGE,
        provider=provider,
        error_type="DurableValueError",
        error_code="invalid_model_provider_error",
        retryable=False,
    )
    return safe_error, {
        "durable_value_error_code": code,
        "durable_value_path": path,
    }


def model_provider_error_from_payload(
    payload: dict[str, Any],
    *,
    fallback_provider: str,
    fallback_message: str = "Model provider error",
) -> ModelProviderError | None:
    """Rehydrate a typed provider failure from a model stream error payload.

    `ModelStreamEvent.error(..., cause=ModelProviderError(...))` preserves typed
    retry fields in `payload`. This helper rebuilds the exception after that
    event boundary so every runtime consumer classifies retries from the same
    typed surface, whether the provider raised the error or flattened it into a
    stream event.
    """

    if not _has_provider_error_payload_fields(payload):
        return None
    message = _clean_payload_string(payload.get("error")) or fallback_message
    provider = _clean_payload_string(payload.get("provider")) or fallback_provider
    if payload.get("context_overflow") is True:
        return ModelContextOverflowError(
            message,
            provider=provider,
            status_code=_payload_status_code(payload.get("status_code")),
            error_type=_clean_payload_string(payload.get("provider_error_type")),
            error_code=_clean_payload_string(payload.get("provider_error_code")),
            request_id=_clean_payload_string(payload.get("request_id")),
        )
    return ModelProviderError(
        message,
        provider=provider,
        status_code=_payload_status_code(payload.get("status_code")),
        error_type=_clean_payload_string(payload.get("provider_error_type")),
        error_code=_clean_payload_string(payload.get("provider_error_code")),
        request_id=_clean_payload_string(payload.get("request_id")),
        retryable=_payload_retryable(payload.get("retryable")),
        retry_after_s=_payload_retry_after_s(payload.get("retry_after_s")),
    )


def _has_provider_error_payload_fields(payload: dict[str, Any]) -> bool:
    if payload.get("error_type") in _PROVIDER_ERROR_TYPE_MARKERS:
        return True
    return any(key in payload for key in _PROVIDER_ERROR_PAYLOAD_KEYS)


def _clean_payload_string(value: Any) -> str | None:
    if type(value) is not str:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _payload_status_code(value: Any) -> int | None:
    if type(value) is not int or value < 100 or value > 599:
        return None
    return value


def _payload_retryable(value: Any) -> bool | None:
    return value if type(value) is bool else None


def _payload_retry_after_s(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    try:
        normalized = float(value)
    except OverflowError:
        return None
    return normalized if math.isfinite(normalized) and normalized >= 0 else None
