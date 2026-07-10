from __future__ import annotations

from typing import Any

from cayu.providers.base import ModelContextOverflowError, ModelProviderError

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
