"""One bounded provider request that shows whether a model answers and is priced.

The check sends a fixed prompt through a registered provider's governed stream
boundary, like a model attempt, with the provider's auxiliary output cap, no
tools, and one deadline over billing-identity resolution and dispatch. Nothing
is written to a session or store. The result keeps the outcome, latency, route,
usage and pricing. The prompt and any answer text are discarded, and a provider
failure keeps only a fixed category, an authored message, and the identifier
fields that pass the provider boundary's public identity check.
"""

from __future__ import annotations

import asyncio
import re
import time
from contextlib import aclosing
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from cayu.budgets.pricing import PriceBook, estimate_model_step_cost
from cayu.budgets.usage import UsageMetrics, normalize_usage_metrics
from cayu.deadlines import ExecutionDeadline, execution_deadline_scope
from cayu.messages import Message
from cayu.providers import ModelProviderError, ModelRequest, ModelStreamEvent, ModelStreamEventType
from cayu.providers._credential_boundary import (
    release_provider_stream_cleanup,
    reserve_provider_stream_cleanup,
)
from cayu.providers._http import _PUBLIC_PROVIDER_IDENTIFIER, _public_provider_identity
from cayu.providers._rejection_diagnostics import PREFIX as REJECTION_PREFIX
from cayu.providers._rejection_diagnostics import rejection_fields
from cayu.providers.base import (
    ModelFinishReason,
    ModelProvider,
    ModelStreamDeadlineError,
    _copy_auxiliary_request,
)
from cayu.providers.deadlines import ProviderStreamDeadlineAdmission
from cayu.runtime._model_errors import (
    model_provider_error_from_payload,
    resolve_completion_billing_identity,
    resolve_request_billing_identity,
)
from cayu.runtime._provider_stream import (
    _admitted_model_provider_events,
    _owned_model_provider_events,
    _provider_stream_self_cancellation_error,
    _ProviderStreamSelfCancellation,
)
from cayu.sessions._call_inspection import decimal_text

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.budgets.billing import BillingIdentity
    from cayu.runtime._runtime_records import RegisteredProvider
    from cayu.vaults.redaction import SecretRedactor

MODEL_CHECK_SCHEMA_VERSION: Literal["1"] = "1"
# OpenAI Responses rejects a max_output_tokens below 16; every other bundled
# adapter accepts it, so one cap serves all of them.
MODEL_CHECK_MAX_OUTPUT_TOKENS = 16
MODEL_CHECK_TIMEOUT_SECONDS = 30.0
_MODEL_CHECK_PROMPT = "Reply with the single word: ready"
_PREFLIGHT_DETAIL_MAX_BYTES = 512
# Provider model IDs also use '/' (OpenRouter) and '@' (Vertex) separators.
_PUBLIC_MODEL_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,255}\Z")
_TERMINAL_EVENT_TYPES = frozenset({ModelStreamEventType.COMPLETED, ModelStreamEventType.ERROR})

ModelCheckResolution = Literal["explicit", "model_pattern", "default"]
PriceBookSource = Literal["bundled", "project"]


class ModelCheckFailureCategory(StrEnum):
    PREFLIGHT_REJECTED = "preflight_rejected"
    UNSUPPORTED = "unsupported"
    BILLING_IDENTITY_FAILED = "billing_identity_failed"
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    REQUEST_REJECTED = "request_rejected"
    SERVER_ERROR = "server_error"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    STREAM_FAILED = "stream_failed"
    EVENT_AFTER_COMPLETION = "event_after_completion"
    GENERATION_FAILED = "generation_failed"


_FAILURE_MESSAGES = {
    ModelCheckFailureCategory.PREFLIGHT_REJECTED: (
        "The provider refused this model before sending a request."
    ),
    ModelCheckFailureCategory.UNSUPPORTED: (
        "The provider cannot prepare a bounded request without tools, so nothing was sent."
    ),
    ModelCheckFailureCategory.BILLING_IDENTITY_FAILED: (
        "The provider could not resolve the request's billing identity."
    ),
    ModelCheckFailureCategory.AUTHENTICATION_FAILED: (
        "The provider rejected the configured credentials."
    ),
    ModelCheckFailureCategory.PERMISSION_DENIED: (
        "The configured credentials are not allowed to use this model."
    ),
    ModelCheckFailureCategory.NOT_FOUND: (
        "The provider does not know this model, or the endpoint is wrong."
    ),
    ModelCheckFailureCategory.RATE_LIMITED: (
        "The provider is rate limiting these credentials, or their quota is used up."
    ),
    ModelCheckFailureCategory.REQUEST_REJECTED: (
        "The provider rejected the request for this model."
    ),
    ModelCheckFailureCategory.SERVER_ERROR: "The provider failed while serving the request.",
    ModelCheckFailureCategory.PROVIDER_ERROR: (
        "The provider failed without an HTTP status, for example on a connection error."
    ),
    ModelCheckFailureCategory.TIMEOUT: "The provider did not answer before the check's deadline.",
    ModelCheckFailureCategory.STREAM_FAILED: (
        "The provider stream ended without an answer or a typed error."
    ),
    ModelCheckFailureCategory.EVENT_AFTER_COMPLETION: (
        "The provider sent another stream event after completing the request."
    ),
    ModelCheckFailureCategory.GENERATION_FAILED: (
        "The provider completed the request, but the model's generation failed."
    ),
}
_STATUS_CATEGORIES = {
    401: ModelCheckFailureCategory.AUTHENTICATION_FAILED,
    403: ModelCheckFailureCategory.PERMISSION_DENIED,
    404: ModelCheckFailureCategory.NOT_FOUND,
    429: ModelCheckFailureCategory.RATE_LIMITED,
}


class ModelCheckRouteError(ValueError):
    """The model check cannot select exactly one registered provider."""


class _ModelCheckRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelCheckRoute(_ModelCheckRecord):
    provider: str
    model: str
    resolution: ModelCheckResolution
    provider_model: str | None = Field(default=None, exclude_if=lambda value: value is None)
    billing_provider: str | None = Field(default=None, exclude_if=lambda value: value is None)


class ModelCheckLimits(_ModelCheckRecord):
    max_output_tokens: int
    timeout_seconds: float


class ModelCheckAnswer(_ModelCheckRecord):
    finish_reason: str
    text_received: bool


class ModelCheckUsage(_ModelCheckRecord):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


class ModelCheckPriceBook(_ModelCheckRecord):
    source: PriceBookSource
    version: str
    generated_at: str


class ModelCheckPricing(_ModelCheckRecord):
    priced: bool
    price_book: ModelCheckPriceBook
    currency: str | None = Field(default=None, exclude_if=lambda value: value is None)
    estimated_cost: str | None = Field(default=None, exclude_if=lambda value: value is None)
    unpriced_reason: str | None = Field(default=None, exclude_if=lambda value: value is None)


class ModelCheckRejection(_ModelCheckRecord):
    reason: str
    parameter: str
    explanation: str


class ModelCheckFailure(_ModelCheckRecord):
    category: ModelCheckFailureCategory
    message: str
    detail: str | None = Field(default=None, exclude_if=lambda value: value is None)
    status_code: int | None = Field(default=None, exclude_if=lambda value: value is None)
    provider_error_type: str | None = Field(default=None, exclude_if=lambda value: value is None)
    provider_error_code: str | None = Field(default=None, exclude_if=lambda value: value is None)
    retryable: bool | None = Field(default=None, exclude_if=lambda value: value is None)
    rejection: ModelCheckRejection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ModelCheckResult(_ModelCheckRecord):
    schema_version: Literal["1"] = MODEL_CHECK_SCHEMA_VERSION
    ok: bool
    route: ModelCheckRoute
    limits: ModelCheckLimits
    latency_ms: int | None
    answer: ModelCheckAnswer | None
    usage: ModelCheckUsage | None
    pricing: ModelCheckPricing
    error: ModelCheckFailure | None


def resolve_model_check_route(
    app: CayuApp,
    *,
    model: str,
    provider_name: str | None,
) -> tuple[RegisteredProvider, ModelCheckResolution]:
    """Select the provider the way an agent without ``provider_name`` resolves it."""

    registry = app._provider_registry
    registrations = registry.registrations
    if provider_name is not None:
        registered = registrations.get(provider_name)
        if registered is None:
            names = ", ".join(registry.names()) or "none"
            raise ModelCheckRouteError(
                f"Provider not registered: {provider_name}. Registered providers: {names}."
            )
        return registered, "explicit"
    matches = registry.matching(model=model)
    if len(matches) > 1:
        names = ", ".join(sorted(item.name for item in matches))
        raise ModelCheckRouteError(
            f"Model {model} matches several registered providers ({names}); select one."
        )
    if matches:
        return matches[0], "model_pattern"
    if registry.default_name is None:
        raise ModelCheckRouteError("No model provider is registered.")
    return registrations[registry.default_name], "default"


async def check_model(
    registered: RegisteredProvider,
    *,
    model: str,
    resolution: ModelCheckResolution,
    pricing: PriceBook,
    price_book_source: PriceBookSource,
    redactor: SecretRedactor,
    timeout_seconds: float,
) -> ModelCheckResult:
    """Send one bounded request for ``model`` and report its outcome and pricing."""

    provider = registered.provider
    pricing_provider_name = provider.billing_provider_name or registered.name
    billing_identity: BillingIdentity | None = None

    def result(
        *,
        failure: ModelCheckFailure | None = None,
        latency_ms: int | None = None,
        answer: ModelCheckAnswer | None = None,
        metrics: UsageMetrics | None = None,
        provider_model: str | None = None,
    ) -> ModelCheckResult:
        billing_provider = (
            pricing_provider_name if billing_identity is None else billing_identity.provider_name
        )
        return ModelCheckResult(
            ok=failure is None,
            route=ModelCheckRoute(
                provider=registered.name,
                model=model,
                resolution=resolution,
                provider_model=None if provider_model == model else provider_model,
                billing_provider=None if billing_provider == registered.name else billing_provider,
            ),
            limits=ModelCheckLimits(
                max_output_tokens=MODEL_CHECK_MAX_OUTPUT_TOKENS,
                timeout_seconds=timeout_seconds,
            ),
            latency_ms=latency_ms,
            answer=answer,
            usage=None if metrics is None else _usage(metrics),
            pricing=_pricing(
                pricing,
                price_book_source,
                (
                    UsageMetrics(
                        provider_name=pricing_provider_name,
                        model=provider_model or model,
                        requested_model=model,
                        billing_identity=billing_identity,
                    )
                    if metrics is None
                    else metrics
                ),
                observed=metrics is not None,
            ),
            error=failure,
        )

    try:
        provider.preflight_model_target(model=model)
    except Exception as error:
        return result(
            failure=_failure(
                ModelCheckFailureCategory.PREFLIGHT_REJECTED,
                detail=_preflight_detail(error, redactor),
            )
        )
    request = _bounded_request(provider, model)
    if request is None:
        return result(failure=_failure(ModelCheckFailureCategory.UNSUPPORTED))

    # One deadline bounds billing-identity resolution and dispatch together.
    deadline = ExecutionDeadline.after(timeout_seconds, source="model_check", scope="model")
    try:
        async with execution_deadline_scope(deadline):
            billing_identity = await resolve_request_billing_identity(
                provider,
                request.model_copy(deep=True),
                provider_name=registered.name,
            )
    except TimeoutError:
        return result(failure=_failure(ModelCheckFailureCategory.TIMEOUT))
    except Exception:
        return result(failure=_failure(ModelCheckFailureCategory.BILLING_IDENTITY_FAILED))

    observed = _StreamObservation()
    dispatch_failure: ModelCheckFailure | None = None
    started = time.monotonic()
    try:
        async with execution_deadline_scope(deadline):
            await _dispatch(registered, request, redactor, observed)
    except Exception as error:
        dispatch_failure = _dispatch_failure(error, redactor)
    latency_ms = _elapsed_ms(started)
    terminal = observed.terminal
    if terminal is None:
        return result(
            failure=dispatch_failure or _failure(ModelCheckFailureCategory.STREAM_FAILED),
            latency_ms=latency_ms,
        )

    completion_billing_failed = False
    if terminal.type is ModelStreamEventType.COMPLETED:
        # As a model attempt does before accounting, let the completion narrow the
        # request's billing identity; on failure the request identity prices it.
        try:
            billing_identity = resolve_completion_billing_identity(
                provider, billing_identity, terminal.payload, provider_name=registered.name
            )
        except ModelProviderError:
            completion_billing_failed = True
    provider_model = _public_provider_identity(
        terminal.payload.get("model"), _PUBLIC_MODEL_IDENTIFIER, redactor
    )
    metrics = normalize_usage_metrics(
        provider_name=pricing_provider_name,
        model=provider_model or model,
        requested_model=model,
        raw_usage=terminal.payload.get("usage"),
        usage_dialect=registered.usage_dialect,
        billing_identity=billing_identity,
    )
    if terminal.type is ModelStreamEventType.ERROR:
        error = model_provider_error_from_payload(
            terminal.payload, fallback_provider=registered.name
        )
        failure = (
            _failure(ModelCheckFailureCategory.PROVIDER_ERROR)
            if error is None
            else _provider_failure(error, redactor)
        )
    elif completion_billing_failed:
        failure = _failure(ModelCheckFailureCategory.BILLING_IDENTITY_FAILED)
    elif observed.event_after_completion:
        failure = _failure(ModelCheckFailureCategory.EVENT_AFTER_COMPLETION)
    elif dispatch_failure is not None:
        # A failure after the completion, such as a failed drain or close, fails a
        # runtime model attempt too, while its completed usage is still accounted.
        failure = dispatch_failure
    else:
        completion = terminal.completion
        if completion is None:  # pragma: no cover - completed events always normalize
            raise AssertionError("Completed model stream event has no completion metadata.")
        if completion.finish_reason is not ModelFinishReason.ERROR:
            return result(
                latency_ms=latency_ms,
                answer=ModelCheckAnswer(
                    finish_reason=completion.finish_reason.value,
                    text_received=observed.text_received,
                ),
                metrics=metrics,
                provider_model=provider_model,
            )
        failure = _failure(ModelCheckFailureCategory.GENERATION_FAILED)
    return result(
        failure=failure,
        latency_ms=latency_ms,
        metrics=metrics,
        provider_model=provider_model,
    )


def _bounded_request(provider: ModelProvider, model: str) -> ModelRequest | None:
    """The provider's capped, tool-free form of the fixed prompt, or None."""

    request = ModelRequest(model=model, messages=[Message.text("user", _MODEL_CHECK_PROMPT)])
    try:
        prepared = _copy_auxiliary_request(
            provider.prepare_auxiliary_request(
                request, max_output_tokens=MODEL_CHECK_MAX_OUTPUT_TOKENS
            )
        )
    except Exception:
        return None
    if prepared.model != request.model or prepared.messages != request.messages:
        return None
    return prepared


@dataclass(slots=True)
class _StreamObservation:
    """What the stream showed before it ended, kept when finishing it then fails."""

    terminal: ModelStreamEvent | None = None
    text_received: bool = False
    event_after_completion: bool = False


async def _dispatch(
    registered: RegisteredProvider,
    request: ModelRequest,
    redactor: SecretRedactor,
    observed: _StreamObservation,
) -> None:
    """Run one governed provider stream, recording its first terminal event."""

    provider = registered.provider
    admission = ProviderStreamDeadlineAdmission(provider.stream_deadlines)
    cleanup = reserve_provider_stream_cleanup(admission.max_concurrent_streams)
    transferred = False
    task = asyncio.current_task()
    cancellation_baseline = 0 if task is None else task.cancelling()

    async def refresh() -> None:
        return None

    try:
        events = _owned_model_provider_events(
            lambda: _admitted_model_provider_events(
                provider, request, admission, refresh, error_redactor=redactor
            ),
            cancellation_baseline=cancellation_baseline,
            max_concurrent_streams=admission.max_concurrent_streams,
            cleanup_ownership=cleanup,
        )
        transferred = True
        async with aclosing(events):
            async for event in events:
                if observed.terminal is not None:
                    # A model attempt fails on any event after its completion.
                    observed.event_after_completion = True
                    break
                if event.type in _TERMINAL_EVENT_TYPES:
                    observed.terminal = event
                    if event.type is ModelStreamEventType.ERROR:
                        # As in a model attempt, an error event ends the stream.
                        break
                elif event.type is ModelStreamEventType.TEXT_DELTA and event.delta:
                    observed.text_received = True
    except _ProviderStreamSelfCancellation:
        if task is not None and task.cancelling() > cancellation_baseline:
            raise
        # Only the provider iterator boundary mints this marker. Report it as the
        # runtime does, a non-retryable provider failure without caller authority.
        raise _provider_stream_self_cancellation_error(registered.name) from None
    finally:
        admission.close()
        if not transferred:
            release_provider_stream_cleanup(cleanup)


def _failure(
    category: ModelCheckFailureCategory, *, detail: str | None = None
) -> ModelCheckFailure:
    return ModelCheckFailure(category=category, message=_FAILURE_MESSAGES[category], detail=detail)


def _dispatch_failure(error: Exception, redactor: SecretRedactor) -> ModelCheckFailure:
    if isinstance(error, ModelProviderError):
        return _provider_failure(error, redactor)
    if isinstance(error, TimeoutError):
        return _failure(ModelCheckFailureCategory.TIMEOUT)
    return _failure(ModelCheckFailureCategory.STREAM_FAILED)


def _provider_failure(error: ModelProviderError, redactor: SecretRedactor) -> ModelCheckFailure:
    """Classify a provider failure from its typed fields, never its message text."""

    if isinstance(error, ModelStreamDeadlineError):
        return _failure(ModelCheckFailureCategory.TIMEOUT)
    status = error.status_code
    if status is None or status < 400:
        category = ModelCheckFailureCategory.PROVIDER_ERROR
    elif status in _STATUS_CATEGORIES:
        category = _STATUS_CATEGORIES[status]
    elif status >= 500:
        category = ModelCheckFailureCategory.SERVER_ERROR
    else:
        category = ModelCheckFailureCategory.REQUEST_REJECTED
    rejection = rejection_fields(error.rejection_diagnostic)
    reason = rejection.get(REJECTION_PREFIX + "reason")
    return ModelCheckFailure(
        category=category,
        message=_FAILURE_MESSAGES[category],
        status_code=status,
        provider_error_type=_public_provider_identity(
            error.error_type, _PUBLIC_PROVIDER_IDENTIFIER, redactor
        ),
        provider_error_code=_public_provider_identity(
            error.error_code, _PUBLIC_PROVIDER_IDENTIFIER, redactor
        ),
        retryable=error.retryable,
        rejection=(
            None
            if reason is None
            else ModelCheckRejection(
                reason=reason,
                parameter=rejection[REJECTION_PREFIX + "parameter"],
                explanation=rejection[REJECTION_PREFIX + "explanation"],
            )
        ),
    )


def _preflight_detail(error: Exception, redactor: SecretRedactor) -> str | None:
    """Application-authored setup text, redacted first and bounded second."""

    try:
        text = str(error)
        text.encode("utf-8")
    except Exception:
        # Lossy repair after redaction could reconstruct a registered secret.
        return None
    bounded, _ = redactor.redact_text_bounded_with_marker(
        text,
        max_bytes=_PREFLIGHT_DETAIL_MAX_BYTES,
        truncation_marker="...[truncated]",
    )
    return bounded.strip() or None


def _usage(metrics: UsageMetrics) -> ModelCheckUsage:
    return ModelCheckUsage(
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        total_tokens=metrics.total_tokens,
        reasoning_tokens=metrics.reasoning_output_tokens,
        cache_read_tokens=metrics.cache.read_tokens,
        cache_write_tokens=metrics.cache.write_tokens,
    )


def _pricing(
    pricing: PriceBook,
    source: PriceBookSource,
    metrics: UsageMetrics,
    *,
    observed: bool,
) -> ModelCheckPricing:
    """Price observed usage, or only the route when the provider reported none."""

    estimate = estimate_model_step_cost(
        metrics=metrics,
        pricing=pricing,
        effective_on=datetime.now(UTC).date(),
    )
    return ModelCheckPricing(
        priced=estimate.priced,
        price_book=ModelCheckPriceBook(
            source=source,
            version=pricing.price_book_version,
            generated_at=pricing.generated_at,
        ),
        currency=estimate.currency,
        estimated_cost=decimal_text(estimate.total_cost) if estimate.priced and observed else None,
        unpriced_reason=estimate.missing_pricing_reason,
    )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
