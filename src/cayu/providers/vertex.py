from __future__ import annotations

import asyncio
import importlib
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from cayu._validation import require_clean_nonblank
from cayu.core.messages import Message
from cayu.providers._config import positive_finite_seconds
from cayu.providers._credential_boundary import (
    aclosing_provider_stream,
    detach_provider_call_traceback,
    detach_provider_stream_traceback,
)
from cayu.providers._http import (
    SharedAsyncClient,
    aclose_transport,
    credential_safe_error_event,
    credential_safe_post_completion_failure,
    credential_safe_provider_exception,
    json_error_text,
    optional_error_string,
    post_json,
    safe_error_response_text,
    sanitize_provider_cancellation,
    stream_sse_json_events,
    truncate_error_text,
    validate_base_url,
    validate_url,
)
from cayu.providers._reasoning_state import (
    ANTHROPIC_REASONING_PROTOCOL,
    ReasoningStateProvenance,
)
from cayu.providers.anthropic import (
    _anthropic_overflow_message,
    _anthropic_tool,
    _effective_anthropic_request_options,
    _validate_anthropic_tool_name,
    anthropic_response_events,
    anthropic_stream_events,
    build_anthropic_payload,
    build_anthropic_token_count_payload,
)
from cayu.providers.base import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    UsageDialect,
    _preflight_provider_portable_messages,
    _terminal_preserving_provider_stream,
    privacy_safe_provider_option_projection,
)
from cayu.providers.deadlines import (
    ProviderStreamDeadlines,
    _resolve_provider_stream_deadlines,
)

if TYPE_CHECKING:
    import httpx

DEFAULT_VERTEX_REGION = "global"
DEFAULT_VERTEX_ANTHROPIC_VERSION = "vertex-2023-10-16"
DEFAULT_VERTEX_MAX_TOKENS = 4096
DEFAULT_VERTEX_TIMEOUT_SECONDS = 60.0
VERTEX_OAUTH_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class VertexError(RuntimeError):
    """Base error for Vertex provider failures."""


class VertexAPIError(VertexError, ModelProviderError):
    """Raised when the Vertex AI HTTP API returns an error response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
        response_body: str | None = None,
    ) -> None:
        ModelProviderError.__init__(
            self,
            message,
            provider="vertex",
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            retryable=retryable,
            retry_after_s=retry_after_s,
            response_body=response_body,
        )


class VertexContextOverflowError(VertexAPIError, ModelContextOverflowError):
    """Raised when Vertex reports that the request exceeds context limits."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        response_body: str | None = None,
    ) -> None:
        ModelContextOverflowError.__init__(
            self,
            message,
            provider="vertex",
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            response_body=response_body,
        )


class VertexProtocolError(VertexError):
    """Raised when Vertex data does not match the expected Messages shape."""


class VertexTransport(Protocol):
    async def count_message_tokens(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """POST a count-tokens rawPredict payload and return decoded JSON."""

    async def create_message(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """POST a Vertex rawPredict payload and return decoded JSON."""

    def stream_message_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        transport_idle_timeout_s: float,
        protocol_idle_timeout_s: float,
        semantic_progress_timeout_s: float,
        absolute_stream_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """POST a streamRawPredict payload and yield decoded SSE data objects."""


class HttpxVertexTransport:
    """HTTP transport with explicit certifi-backed TLS verification.

    Owns one shared httpx.AsyncClient (created lazily) that is reused across
    requests so each model call does not pay for a fresh TLS handshake. Close it
    with :meth:`aclose` when the transport is no longer needed.
    """

    def __init__(self) -> None:
        self._client = SharedAsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def count_message_tokens(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._post_json(
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
        )

    async def create_message(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._post_json(
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
        )

    async def stream_message_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        transport_idle_timeout_s: float,
        protocol_idle_timeout_s: float,
        semantic_progress_timeout_s: float,
        absolute_stream_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        url = _validate_url(url, "url")
        events = stream_sse_json_events(
            client=self._client.get(),
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
            transport_idle_timeout_s=transport_idle_timeout_s,
            protocol_idle_timeout_s=protocol_idle_timeout_s,
            semantic_progress_timeout_s=semantic_progress_timeout_s,
            absolute_stream_timeout_s=absolute_stream_timeout_s,
            request_label="Vertex AI",
            response_label="Vertex",
            api_error=VertexAPIError,
            protocol_error=VertexProtocolError,
            error_response_text=_safe_error_response_text,
            raise_context_overflow=_raise_vertex_context_overflow_if_applicable,
            raise_context_overflow_from_status=_raise_vertex_context_overflow_from_status,
            api_error_from_response=_vertex_api_error_from_response,
        )
        async with aclosing_provider_stream(events):
            async for event in events:
                yield event

    async def _post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        url = _validate_url(url, "url")
        return await post_json(
            client=self._client.get(),
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
            request_label="Vertex AI",
            response_label="Vertex",
            api_error=VertexAPIError,
            protocol_error=VertexProtocolError,
            error_response_text=_safe_error_response_text,
            raise_context_overflow=_raise_vertex_context_overflow_if_applicable,
            api_error_from_response=_vertex_api_error_from_response,
        )


class VertexProvider(ModelProvider):
    """Anthropic Claude models hosted on Google Cloud Vertex AI.

    Sends the standard Anthropic Messages body to the regional ``:rawPredict``
    endpoint with an OAuth bearer token, reusing the Anthropic payload builder
    and response parser. The body omits ``model`` (it is in the URL) and carries
    ``anthropic_version`` instead.

    Notes:
        - Usage/cache accounting is Anthropic-shaped; this adapter declares
          ``usage_dialect = UsageDialect.ANTHROPIC`` so cache-token folding in
          ``normalize_usage_metrics`` survives a custom ``name``.
        - For budget enforcement, register pricing rows under provider ``"vertex"``
          (Vertex Claude rates differ from the direct Anthropic API).
    """

    name = "vertex"
    usage_dialect = UsageDialect.ANTHROPIC

    @property
    def stream_deadlines(self) -> ProviderStreamDeadlines:
        return self._stream_deadlines

    def preflight_portable_messages(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        _preflight_provider_portable_messages(
            model=model,
            messages=messages,
            tools=tools,
            supports_system_messages=True,
            supports_tool_history=True,
            supports_tool_definitions=True,
            supports_file_attachments=True,
            tool_name_validator=_validate_anthropic_tool_name,
            tool_definition_validator=_anthropic_tool,
        )

    def request_footprint_options(self, request: ModelRequest) -> dict[str, Any]:
        effective_options = _effective_anthropic_request_options(
            request.options,
            default_max_tokens=self.max_tokens,
        )
        projected = privacy_safe_provider_option_projection(effective_options)
        return {"anthropic": projected} if projected else {}

    def request_fingerprint_options(self, request: ModelRequest) -> dict[str, Any]:
        return {
            "anthropic": _effective_anthropic_request_options(
                request.options,
                default_max_tokens=self.max_tokens,
            )
        }

    @property
    def anthropic_version(self) -> str:
        """Return the immutable wire and reasoning-state protocol version."""

        return self._reasoning_state_provenance.protocol_version

    def __init__(
        self,
        *,
        project_id: str,
        region: str = DEFAULT_VERTEX_REGION,
        credentials: Any | None = None,
        service_account_info: Mapping[str, Any] | None = None,
        service_account_file: str | None = None,
        base_url: str | None = None,
        name: str = "vertex",
        anthropic_version: str = DEFAULT_VERTEX_ANTHROPIC_VERSION,
        max_tokens: int = DEFAULT_VERTEX_MAX_TOKENS,
        timeout_s: float = DEFAULT_VERTEX_TIMEOUT_SECONDS,
        stream_deadlines: ProviderStreamDeadlines | None = None,
        transport: VertexTransport | None = None,
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.project_id = require_clean_nonblank(project_id, "project_id")
        self.region = require_clean_nonblank(region, "region")
        validated_anthropic_version = require_clean_nonblank(
            anthropic_version,
            "anthropic_version",
        )
        self._reasoning_state_provenance = ReasoningStateProvenance(
            provider="vertex",
            protocol=ANTHROPIC_REASONING_PROTOCOL,
            protocol_version=validated_anthropic_version,
        )
        self.base_url = _validate_base_url(base_url) if base_url is not None else None
        if type(max_tokens) is not int:
            raise TypeError("max_tokens must be an integer.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")
        self.max_tokens = max_tokens
        self.timeout_s = positive_finite_seconds(timeout_s, "timeout_s")
        self._stream_deadlines = _resolve_provider_stream_deadlines(
            stream_deadlines=stream_deadlines,
        )
        self.credentials = _resolve_credentials(
            credentials=credentials,
            service_account_info=service_account_info,
            service_account_file=service_account_file,
        )
        self.transport = transport if transport is not None else HttpxVertexTransport()
        self._refresh_lock = asyncio.Lock()

    async def aclose(self) -> None:
        """Close the transport's shared HTTP client, if it owns one."""
        await aclose_transport(self.transport)

    @_terminal_preserving_provider_stream
    @detach_provider_stream_traceback
    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        token: str | None = None
        cancellation: asyncio.CancelledError | None = None
        overflow_failure: VertexContextOverflowError | None = None
        post_completion_failure: ModelProviderError | None = None
        error_event: ModelStreamEvent | None = None
        completion_emitted = False
        try:
            payload = build_anthropic_payload(
                request,
                default_max_tokens=self.max_tokens,
                reasoning_provenance=self._reasoning_state_provenance,
            )
            payload.pop("model", None)
            payload["anthropic_version"] = self.anthropic_version
            token = await self._access_token()
            stream_transport = getattr(self.transport, "stream_message_events", None)
            if stream_transport is None:
                # Back-compat: transports predating SSE support fall back to one
                # buffered POST and a synthetic event replay.
                response = await self.transport.create_message(
                    url=self._endpoint(request.model),
                    headers=self._request_headers(token),
                    payload=payload,
                    timeout_s=self.timeout_s,
                )
                for event in anthropic_response_events(
                    response,
                    reasoning_provenance=self._reasoning_state_provenance,
                ):
                    completion_emitted = event.type == ModelStreamEventType.COMPLETED
                    yield event
                    if completion_emitted:
                        break
            else:
                payload["stream"] = True
                raw_events = stream_transport(
                    url=self._endpoint(request.model, verb="streamRawPredict"),
                    headers=self._request_headers(token),
                    payload=payload,
                    timeout_s=self.timeout_s,
                    transport_idle_timeout_s=self.stream_deadlines.transport_idle_timeout_s,
                    protocol_idle_timeout_s=self.stream_deadlines.protocol_idle_timeout_s,
                    semantic_progress_timeout_s=(self.stream_deadlines.semantic_progress_timeout_s),
                    absolute_stream_timeout_s=self.stream_deadlines.absolute_stream_timeout_s,
                )
                events = anthropic_stream_events(
                    raw_events,
                    provider_label="Vertex",
                    api_error=VertexAPIError,
                    protocol_error=VertexProtocolError,
                    context_overflow_error=VertexContextOverflowError,
                    reasoning_provenance=self._reasoning_state_provenance,
                )
                async with aclosing_provider_stream(raw_events), aclosing_provider_stream(events):
                    # Completion is synthesized only after the raw Messages
                    # stream tail is validated. Exhaust the translator so a
                    # deferred transport failure remains observable after it.
                    async for event in events:
                        completion_emitted = event.type == ModelStreamEventType.COMPLETED
                        yield event
        except asyncio.CancelledError as exc:
            cancellation = sanitize_provider_cancellation(
                exc,
                provider_label="Vertex",
                credential_values=(token,) if token is not None else (),
            )
        except ModelContextOverflowError as exc:
            credential_values = (token,) if token is not None else ()
            if completion_emitted:
                post_completion_failure = credential_safe_post_completion_failure(
                    exc,
                    provider_label="Vertex",
                    provider_name="vertex",
                    credential_values=credential_values,
                )
            else:
                # Overflow must reach runtime recovery as a typed exception; an
                # error event would flatten it into unrecoverable message text.
                safe = credential_safe_provider_exception(
                    exc,
                    provider_label="Vertex",
                    provider_name="vertex",
                    credential_values=credential_values,
                )
                overflow_failure = VertexContextOverflowError(
                    str(safe),
                    status_code=safe.status_code,
                    error_type=safe.error_type,
                    error_code=safe.error_code,
                    request_id=safe.request_id,
                    response_body=None,
                )
        except Exception as exc:
            credential_values = (token,) if token is not None else ()
            if completion_emitted:
                post_completion_failure = credential_safe_post_completion_failure(
                    exc,
                    provider_label="Vertex",
                    provider_name="vertex",
                    credential_values=credential_values,
                )
            else:
                error_event = credential_safe_error_event(
                    exc,
                    provider_label="Vertex",
                    provider_name="vertex",
                    credential_values=credential_values,
                )
        token = None
        if cancellation is not None:
            raise cancellation from None
        if overflow_failure is not None:
            raise overflow_failure from None
        if post_completion_failure is not None:
            raise post_completion_failure from None
        if error_event is not None:
            yield error_event

    @detach_provider_call_traceback
    async def count_input_tokens(
        self,
        request: ModelRequest,
    ) -> InputTokenCountResult | None:
        """Count input tokens via the Anthropic count-tokens endpoint on Vertex.

        Vertex exposes the Anthropic Messages count-tokens API as a rawPredict
        call on the literal ``count-tokens`` model segment; unlike ``stream``,
        the real model stays in the request body.
        """
        count_transport = getattr(self.transport, "count_message_tokens", None)
        if count_transport is None:
            # Back-compat: transports predating token counting stay source-compatible.
            return None
        payload = build_anthropic_token_count_payload(
            request,
            default_max_tokens=self.max_tokens,
            reasoning_provenance=self._reasoning_state_provenance,
        )
        payload["anthropic_version"] = self.anthropic_version
        response = await self._safe_count_message_tokens(count_transport, payload)
        return InputTokenCountResult(
            input_tokens=_vertex_input_tokens_from_count_response(response),
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
            metadata={
                "endpoint": "count-tokens:rawPredict",
                "provider_billing_status": "not_documented",
            },
        )

    async def _safe_count_message_tokens(
        self,
        count_transport: Any,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        token: str | None = None
        cancellation: asyncio.CancelledError | None = None
        failure: ModelProviderError | None = None
        try:
            token = await self._access_token()
            return await count_transport(
                url=self._endpoint("count-tokens"),
                headers=self._request_headers(token),
                payload=payload,
                timeout_s=self.timeout_s,
            )
        except asyncio.CancelledError as exc:
            cancellation = sanitize_provider_cancellation(
                exc,
                provider_label="Vertex",
                credential_values=(token,) if token is not None else (),
            )
        except Exception as exc:
            safe = credential_safe_provider_exception(
                exc,
                provider_label="Vertex",
                provider_name="vertex",
                credential_values=(token,) if token is not None else (),
            )
            if isinstance(safe, ModelContextOverflowError):
                failure = VertexContextOverflowError(
                    str(safe),
                    status_code=safe.status_code,
                    error_type=safe.error_type,
                    error_code=safe.error_code,
                    request_id=safe.request_id,
                    response_body=None,
                )
            else:
                failure = VertexAPIError(
                    str(safe),
                    status_code=safe.status_code,
                    error_type=safe.error_type,
                    error_code=safe.error_code,
                    request_id=safe.request_id,
                    retryable=safe.retryable,
                    retry_after_s=safe.retry_after_s,
                    response_body=None,
                )
        token = None
        if cancellation is not None:
            raise cancellation
        if failure is None:  # pragma: no cover - the try branch returns
            raise AssertionError("Vertex count failure was not captured")
        raise failure

    def _endpoint(self, model: str, *, verb: str = "rawPredict") -> str:
        model = require_clean_nonblank(model, "model")
        host = self.base_url or _vertex_host(self.region)
        return (
            f"{host}/v1/projects/{self.project_id}/locations/{self.region}"
            f"/publishers/anthropic/models/{model}:{verb}"
        )

    def _request_headers(self, token: str) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    async def _access_token(self) -> str:
        credentials = self.credentials
        if not getattr(credentials, "valid", False):
            # Single-flight: concurrent stream() calls must not refresh the shared
            # google-auth credential in parallel. Re-check inside the lock in case a
            # peer already refreshed while we waited. The lazy import + Request() are
            # offloaded with the blocking refresh so nothing blocks the event loop.
            async with self._refresh_lock:
                if not getattr(credentials, "valid", False):
                    await asyncio.to_thread(lambda: credentials.refresh(_auth_request()))
        token = getattr(credentials, "token", None)
        if not token:
            raise VertexError("Vertex credentials did not produce an access token.")
        return token


def _resolve_credentials(
    *,
    credentials: Any | None,
    service_account_info: Mapping[str, Any] | None,
    service_account_file: str | None,
) -> Any:
    provided = [
        credentials is not None,
        service_account_info is not None,
        service_account_file is not None,
    ]
    if sum(provided) > 1:
        raise ValueError(
            "Provide at most one of credentials, service_account_info, service_account_file."
        )
    if credentials is not None:
        return credentials
    if service_account_info is not None:
        service_account = _import_google("google.oauth2.service_account")
        return service_account.Credentials.from_service_account_info(
            dict(service_account_info),
            scopes=[VERTEX_OAUTH_SCOPE],
        )
    if service_account_file is not None:
        service_account = _import_google("google.oauth2.service_account")
        return service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=[VERTEX_OAUTH_SCOPE],
        )
    auth = _import_google("google.auth")
    resolved, _ = auth.default(scopes=[VERTEX_OAUTH_SCOPE])
    return resolved


def _vertex_host(region: str) -> str:
    # Mirrors the official AnthropicVertex SDK's region->host mapping: `global` and
    # the `us`/`eu` multi-region endpoints are NOT the `{region}-aiplatform` template
    # (using it for `global` yields the non-resolving `global-aiplatform...` host).
    if region == "global":
        return "https://aiplatform.googleapis.com"
    if region in ("us", "eu"):
        return f"https://aiplatform.{region}.rep.googleapis.com"
    return f"https://{region}-aiplatform.googleapis.com"


def _auth_request() -> Any:
    requests_transport = _import_google("google.auth.transport.requests")
    return requests_transport.Request()


def _import_google(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        # Only remap to the optional-dependency hint when the requested google
        # module (or an ancestor of it, e.g. the `google` namespace) is what's
        # absent. A deeper/unrelated missing module (a corrupt install, a missing
        # transitive dep) must surface as its own error instead of being masked.
        if not (missing == module_name or module_name.startswith(f"{missing}.")):
            raise
        raise RuntimeError(
            "VertexProvider requires the optional google-auth package. "
            "Install it with `pip install cayu[vertex]`."
        ) from exc


def _vertex_input_tokens_from_count_response(response: Mapping[str, Any]) -> int:
    if not isinstance(response, Mapping):
        raise VertexProtocolError("Vertex token count response must be a JSON object.")
    input_tokens = response.get("input_tokens")
    if type(input_tokens) is not int or input_tokens < 0:
        raise VertexProtocolError("Vertex token count response requires input_tokens.")
    return input_tokens


def _decoded_gcp_error(response: httpx.Response) -> Mapping[str, Any] | None:
    """Extract the GCP error object; some endpoints array-wrap the envelope."""
    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        return None
    try:
        decoded = response.json()
    except ValueError:
        return None
    if isinstance(decoded, list) and decoded:
        decoded = decoded[0]
    if not isinstance(decoded, Mapping):
        return None
    error = decoded.get("error")
    return error if isinstance(error, Mapping) else None


def _raise_vertex_context_overflow_if_applicable(response: httpx.Response) -> None:
    error = _decoded_gcp_error(response)
    if error is None:
        _raise_vertex_context_overflow_from_status(response.status_code)
        return
    error_type = optional_error_string(error.get("status"))
    identity_missing = "status" not in error
    raw_message = error.get("message")
    message = raw_message if isinstance(raw_message, str) else None
    if not _is_vertex_context_overflow(
        status_code=response.status_code,
        error_type=error_type,
        identity_missing=identity_missing,
        message=message,
    ):
        return
    raise VertexContextOverflowError(
        "Vertex model context overflow",
        status_code=response.status_code,
        error_type=error_type,
        response_body=_safe_error_response_text(response),
    )


def _raise_vertex_context_overflow_from_status(status_code: int) -> None:
    """Classify overflow only when the unread HTTP status is authoritative."""
    if status_code != 413:
        return
    raise VertexContextOverflowError(
        "Vertex model context overflow",
        status_code=status_code,
    )


def _is_vertex_context_overflow(
    *,
    status_code: int,
    error_type: str | None,
    identity_missing: bool,
    message: str | None,
) -> bool:
    # Vertex proxies the Anthropic Messages API, so an oversized request comes
    # back as HTTP 400 INVALID_ARGUMENT carrying the Anthropic overflow message
    # (or as a request-entity-too-large 413 at the GCP front end).
    if status_code == 413:
        if identity_missing:
            return True
        return (
            error_type == "INVALID_ARGUMENT"
            and message is not None
            and _anthropic_overflow_message(message)
        )
    if status_code != 400 or message is None:
        return False
    return _anthropic_overflow_message(message)


_VERTEX_ERROR_STATUS = {
    "ABORTED": 409,
    "ALREADY_EXISTS": 409,
    "CANCELLED": 499,
    "DATA_LOSS": 500,
    "DEADLINE_EXCEEDED": 504,
    "FAILED_PRECONDITION": 400,
    "INTERNAL": 500,
    "INVALID_ARGUMENT": 400,
    "NOT_FOUND": 404,
    "OK": 200,
    "OUT_OF_RANGE": 400,
    "PERMISSION_DENIED": 403,
    "RESOURCE_EXHAUSTED": 429,
    "UNKNOWN": 500,
    "UNAUTHENTICATED": 401,
    "UNAVAILABLE": 503,
    "UNIMPLEMENTED": 501,
}
_VERTEX_RETRYABLE_STATUSES = frozenset({429, 500, 503, 504})


def _vertex_retry_metadata(
    *,
    transport_status_code: int,
    error_type: str | None,
) -> tuple[int, bool | None]:
    """Classify recognized GCP identities; status conflicts fail closed."""

    canonical_status = _VERTEX_ERROR_STATUS.get(error_type or "")
    if canonical_status is None:
        return transport_status_code, None
    if transport_status_code != canonical_status:
        return transport_status_code, False
    return canonical_status, canonical_status in _VERTEX_RETRYABLE_STATUSES


def _vertex_api_error_from_response(
    response: httpx.Response,
    message: str,
    retry_after_s: float | None,
) -> VertexAPIError:
    """Build a structured `VertexAPIError` from an HTTP error response.

    Keeps the GCP error body's typed identity (``status`` as error_type) and the
    HTTP status code so runtime retry classification keys off typed fields
    instead of reparsing the flattened message text.
    """
    error = _decoded_gcp_error(response) or {}
    error_type = optional_error_string(error.get("status"))
    status_code, retryable = _vertex_retry_metadata(
        transport_status_code=response.status_code,
        error_type=error_type,
    )
    return VertexAPIError(
        message,
        status_code=status_code,
        error_type=error_type,
        retryable=retryable,
        retry_after_s=retry_after_s,
        response_body=_safe_error_response_text(response),
    )


def _safe_error_response_text(response: httpx.Response) -> str:
    return safe_error_response_text(response, format_error_json=_safe_gcp_error)


def _safe_gcp_error(decoded: Any) -> str:
    # GCP returns {"error": {...}}; some endpoints array-wrap it as [{"error": {...}}].
    if isinstance(decoded, list) and decoded:
        decoded = decoded[0]
    if isinstance(decoded, Mapping):
        error = decoded.get("error")
        if isinstance(error, Mapping):
            safe: dict[str, Any] = {}
            code = error.get("code")
            status = error.get("status")
            message = error.get("message")
            if isinstance(code, int):
                safe["code"] = code
            if isinstance(status, str):
                safe["status"] = status
            if isinstance(message, str):
                safe["message"] = truncate_error_text(message)
            if safe:
                return json_error_text(safe)
    return truncate_error_text(json_error_text(decoded))


def _validate_base_url(base_url: str) -> str:
    return validate_base_url(base_url, provider_label="Vertex")


def _validate_url(url: str, field_name: str) -> str:
    return validate_url(url, field_name, provider_label="Vertex")
