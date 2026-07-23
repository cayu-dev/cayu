from __future__ import annotations

import asyncio
from typing import Any

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
    try:
        supplied_identity = await provider.billing_identity_for_request(request)
        supplied_identity = copy_billing_identity(supplied_identity)
    except asyncio.CancelledError:
        cancellation = _billing_identity_cancellation()
    except Exception as exc:
        failure = _billing_identity_error(exc, provider_name=provider_name)
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

    completed_identity: BillingIdentity | None = None
    cancellation: asyncio.CancelledError | None = None
    failure: ModelProviderError | None = None
    try:
        completed_identity = provider.billing_identity_for_completion(
            request_identity,
            completed_payload,
        )
        completed_identity = completed_billing_identity(request_identity, completed_identity)
    except asyncio.CancelledError:
        cancellation = _billing_identity_cancellation()
    except Exception as exc:
        failure = _billing_identity_error(exc, provider_name=provider_name)
    del provider
    del request_identity
    del completed_payload
    if cancellation is not None:
        completed_identity = None
        raise cancellation
    if failure is not None:
        completed_identity = None
        raise failure
    return completed_identity


def _billing_identity_error(exc: Exception, *, provider_name: str) -> ModelProviderError:
    source = exc if isinstance(exc, ModelProviderError) else None
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
