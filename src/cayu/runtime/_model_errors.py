from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

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


class _BillingIdentityResolutionCancelled(asyncio.CancelledError):
    """Internal marker carried only until a provider-free public boundary."""


def detach_billing_identity_cancellation(
    exc: asyncio.CancelledError,
) -> asyncio.CancelledError | None:
    """Return a fresh public cancellation for an internal billing-hook marker."""

    if not isinstance(exc, _BillingIdentityResolutionCancelled):
        return None
    return asyncio.CancelledError("Model provider billing identity resolution cancelled")


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
    if isinstance(exc, _BillingIdentityResolutionCancelled):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_contains_billing_identity_cancellation(child) for child in exc.exceptions)
    return False


def _detach_billing_identity_cancellation_tree(
    exc: BaseExceptionGroup,
) -> BaseExceptionGroup:
    detached_children: list[BaseException] = []
    for child in exc.exceptions:
        if isinstance(child, BaseExceptionGroup):
            detached_children.append(_detach_billing_identity_cancellation_tree(child))
        elif isinstance(child, asyncio.CancelledError):
            detached_children.append(
                detach_billing_identity_cancellation(child)
                or asyncio.CancelledError(
                    "Concurrent cancellation during model provider billing identity resolution"
                )
            )
        else:
            detached_children.append(_detached_concurrent_billing_failure(child))
    return BaseExceptionGroup(
        "Model provider billing identity resolution cancellation had concurrent failures",
        detached_children,
    )


def _detached_concurrent_billing_failure(exc: BaseException) -> BaseException:
    if isinstance(exc, KeyboardInterrupt):
        return KeyboardInterrupt(
            "Concurrent keyboard interrupt during model provider billing identity resolution"
        )
    if isinstance(exc, SystemExit):
        return SystemExit(
            "Concurrent system exit during model provider billing identity resolution"
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

    supplied_identity: BillingIdentity | None = None
    cancellation: asyncio.CancelledError | None = None
    failure: ModelProviderError | None = None
    hook_returned = False
    try:
        supplied_identity = await provider.billing_identity_for_request(request)
        hook_returned = True
        supplied_identity = copy_billing_identity(supplied_identity)
    except asyncio.CancelledError:
        cancellation = _billing_identity_cancellation()
    except Exception as exc:
        failure = _billing_identity_error(
            exc,
            provider_name=provider_name,
            direct_hook_failure=not hook_returned,
        )
    del provider
    del request
    if cancellation is not None:
        supplied_identity = None
        raise cancellation
    if failure is not None:
        supplied_identity = None
        raise failure
    return supplied_identity


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
    except asyncio.CancelledError:
        cancellation = _billing_identity_cancellation()
    except Exception as exc:
        failure = _billing_identity_error(
            exc,
            provider_name=provider_name,
            direct_hook_failure=hook_invoked and not hook_returned,
        )
    del provider
    del request_identity
    del completed_payload
    del request_snapshot
    del hook_identity
    del hook_payload
    if cancellation is not None:
        completed_identity = None
        raise cancellation
    if failure is not None:
        completed_identity = None
        raise failure
    return completed_identity


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


def _billing_identity_cancellation() -> asyncio.CancelledError:
    return _BillingIdentityResolutionCancelled(
        "Model provider billing identity resolution cancelled"
    )


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
    if type(value) is not int or value < 100:
        return None
    return value


def _payload_retryable(value: Any) -> bool | None:
    return value if type(value) is bool else None


def _payload_retry_after_s(value: Any) -> float | None:
    return float(value) if type(value) in {int, float} and value >= 0 else None
