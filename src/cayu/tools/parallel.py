"""Hosted Parallel adapters for Cayu's provider-neutral web tools."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from cayu.core.tools import ToolContext, ToolResult
from cayu.proxies import ProxyAuthorizationResult
from cayu.tools._redaction import active_secret_redactor
from cayu.tools.web import (
    MAX_WEB_FETCH_TITLE_BYTES,
    MAX_WEB_SEARCH_QUERY_BYTES,
    WebFetchAdapterRequest,
    WebSearchAdapterRequest,
    _BoundedNormalizedText,
    _canonicalize_url,
    _configuration_int,
    _error_result,
    _web_fetch_success_result,
    _web_search_success_result,
)
from cayu.vaults import ResolvedSecret, SecretRedactor, SecretRef, copy_secret_ref

DEFAULT_PARALLEL_ORIGIN = "https://api.parallel.ai"
DEFAULT_PARALLEL_SEARCH_MODE = "advanced"
DEFAULT_PARALLEL_MAX_PROVIDER_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_PARALLEL_PROVIDER_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_PARALLEL_OBJECTIVE_BYTES = 8 * 1024
MAX_PARALLEL_OBJECTIVE_CHARACTERS = 5_000
MAX_PARALLEL_SEARCH_QUERY_CHARACTERS = 200
MAX_PARALLEL_URL_CHARACTERS = 8_192
MAX_PARALLEL_REQUEST_ID_BYTES = 256
MAX_PARALLEL_SESSION_ID_BYTES = 1_000
MAX_PARALLEL_WARNING_TYPE_BYTES = 128
MAX_PARALLEL_WARNING_BYTES = 512
MAX_PARALLEL_WARNINGS = 16
MAX_PARALLEL_USAGE_ITEMS = 32
MAX_PARALLEL_USAGE_NAME_BYTES = 128
MAX_PARALLEL_EXCERPTS_PER_RESULT = 16
MAX_PARALLEL_FETCH_AGE_SECONDS = 365 * 24 * 60 * 60
_MAX_PARALLEL_RETRY_AFTER_SECONDS = 24 * 60 * 60
_PARALLEL_SEARCH_MODES = frozenset({"turbo", "fast", "basic", "advanced"})


@dataclass(frozen=True)
class _ParallelHttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes
    redactor: SecretRedactor


class _ParallelOversizedResponseError(Exception):
    pass


class _ParallelMalformedResponseError(Exception):
    pass


class ParallelAIWebAdapter:
    """Implement provider-neutral search and known-URL fetch through Parallel.

    Provider modes, objectives, cache policy, and credentials remain fixed
    application configuration. The model continues to see only Cayu's stable
    ``web_search`` and ``web_fetch`` contracts.
    """

    def __init__(
        self,
        *,
        api_key_ref: SecretRef,
        origin: str = DEFAULT_PARALLEL_ORIGIN,
        search_mode: Literal["turbo", "fast", "basic", "advanced"] = (DEFAULT_PARALLEL_SEARCH_MODE),
        search_location: str | None = None,
        search_objective: str | None = None,
        fetch_objective: str | None = None,
        search_fetch_max_age_seconds: int | None = None,
        fetch_max_age_seconds: int | None = None,
        search_disable_cache_fallback: bool = False,
        fetch_disable_cache_fallback: bool = False,
        max_provider_response_bytes: int = DEFAULT_PARALLEL_MAX_PROVIDER_RESPONSE_BYTES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if type(api_key_ref) is not SecretRef:
            raise TypeError("api_key_ref must be a SecretRef.")
        self.api_key_ref = copy_secret_ref(api_key_ref)
        self.origin = _parallel_origin(origin)
        if type(search_mode) is not str or search_mode not in _PARALLEL_SEARCH_MODES:
            raise ValueError("search_mode is not supported by this Parallel adapter.")
        self.search_mode = search_mode
        self.search_location = _parallel_search_location(search_location)
        self.search_objective = _parallel_objective(search_objective, "search_objective")
        self.fetch_objective = _parallel_objective(fetch_objective, "fetch_objective")
        self.search_fetch_max_age_seconds = _parallel_max_age_seconds(
            search_fetch_max_age_seconds,
            "search_fetch_max_age_seconds",
        )
        self.fetch_max_age_seconds = _parallel_max_age_seconds(
            fetch_max_age_seconds,
            "fetch_max_age_seconds",
        )
        if type(search_disable_cache_fallback) is not bool:
            raise TypeError("search_disable_cache_fallback must be a boolean.")
        if type(fetch_disable_cache_fallback) is not bool:
            raise TypeError("fetch_disable_cache_fallback must be a boolean.")
        self.search_disable_cache_fallback = search_disable_cache_fallback
        self.fetch_disable_cache_fallback = fetch_disable_cache_fallback
        if self.search_disable_cache_fallback and self.search_fetch_max_age_seconds is None:
            raise ValueError("search_disable_cache_fallback requires search_fetch_max_age_seconds.")
        if self.fetch_disable_cache_fallback and self.fetch_max_age_seconds is None:
            raise ValueError("fetch_disable_cache_fallback requires fetch_max_age_seconds.")
        self.max_provider_response_bytes = _configuration_int(
            max_provider_response_bytes,
            "max_provider_response_bytes",
            minimum=1,
            maximum=MAX_PARALLEL_PROVIDER_RESPONSE_BYTES,
        )
        if transport is not None and not isinstance(transport, httpx.AsyncBaseTransport):
            raise TypeError("transport must be an httpx.AsyncBaseTransport.")
        self._transport = transport

    async def search(
        self,
        ctx: ToolContext,
        request: WebSearchAdapterRequest,
    ) -> ToolResult:
        if len(request.query) > MAX_PARALLEL_SEARCH_QUERY_CHARACTERS:
            return _error_result(
                "unsupported_semantics",
                "The hosted search provider cannot accept a query of this length.",
            )
        unsupported = _parallel_unsupported_restriction(request)
        if unsupported is not None:
            return _error_result(
                "unsupported_semantics",
                f"Parallel cannot enforce the configured {unsupported} search restriction.",
            )
        advanced_settings: dict[str, Any] = {
            "excerpt_settings": {
                "max_chars_per_result": request.max_snippet_bytes,
            },
            "max_results": request.max_results,
        }
        source_policy: dict[str, Any] = {}
        restrictions = request.restrictions
        if restrictions.include_domains:
            source_policy["include_domains"] = list(restrictions.include_domains)
        if restrictions.exclude_domains:
            source_policy["exclude_domains"] = list(restrictions.exclude_domains)
        if restrictions.published_on_or_after is not None:
            source_policy["after_date"] = restrictions.published_on_or_after.isoformat()
        if source_policy:
            advanced_settings["source_policy"] = source_policy
        if self.search_location is not None:
            advanced_settings["location"] = self.search_location
        fetch_policy = _parallel_fetch_policy(
            self.search_fetch_max_age_seconds,
            self.search_disable_cache_fallback,
            timeout_seconds=request.timeout_seconds,
        )
        if fetch_policy is not None:
            advanced_settings["fetch_policy"] = fetch_policy
        objective = request.query
        if self.search_objective is not None:
            objective = f"{self.search_objective}\n\nSearch request: {request.query}"
        if not _parallel_objective_within_provider_limit(objective):
            return _error_result(
                "unsupported_semantics",
                "The configured Parallel search objective and query exceed the provider limit.",
            )
        payload = {
            "objective": objective,
            "search_queries": [request.query],
            "mode": self.search_mode,
            "max_chars_total": request.max_total_snippet_bytes,
            "advanced_settings": advanced_settings,
        }
        response = await self._post_json(
            ctx,
            path="/v1/search",
            action="parallel.search",
            payload=payload,
            timeout_seconds=request.timeout_seconds,
            max_response_bytes=self.max_provider_response_bytes,
        )
        if isinstance(response, ToolResult):
            return response
        if response.status_code < 200 or response.status_code >= 300:
            return _parallel_http_error_result(response)
        try:
            document = _parallel_json_object(response.body)
            return _parallel_search_result(
                request,
                document,
                redactor=response.redactor,
            )
        except _ParallelMalformedResponseError:
            return _error_result(
                "malformed_provider_response",
                "Parallel returned a malformed search response.",
            )

    async def fetch(
        self,
        ctx: ToolContext,
        request: WebFetchAdapterRequest,
    ) -> ToolResult:
        if len(request.requested_url) > MAX_PARALLEL_URL_CHARACTERS:
            return _error_result(
                "unsupported_semantics",
                "The hosted fetch provider cannot accept a URL of this length.",
            )
        advanced_settings: dict[str, Any] = {
            "excerpt_settings": {
                "max_chars_per_result": request.max_content_bytes,
            },
            "full_content": False,
        }
        fetch_policy = _parallel_fetch_policy(
            self.fetch_max_age_seconds,
            self.fetch_disable_cache_fallback,
            timeout_seconds=request.timeout_seconds,
        )
        if fetch_policy is not None:
            advanced_settings["fetch_policy"] = fetch_policy
        payload: dict[str, Any] = {
            "urls": [request.requested_url],
            "max_chars_total": request.max_content_bytes,
            "advanced_settings": advanced_settings,
        }
        if self.fetch_objective is not None:
            payload["objective"] = self.fetch_objective
        response = await self._post_json(
            ctx,
            path="/v1/extract",
            action="parallel.extract",
            payload=payload,
            timeout_seconds=request.timeout_seconds,
            max_response_bytes=min(
                self.max_provider_response_bytes,
                request.max_response_bytes,
            ),
        )
        if isinstance(response, ToolResult):
            return response
        if response.status_code < 200 or response.status_code >= 300:
            return _parallel_http_error_result(response)
        try:
            document = _parallel_json_object(response.body)
            return _parallel_fetch_result(
                request,
                document,
                redactor=response.redactor,
            )
        except _ParallelMalformedResponseError:
            return _error_result(
                "malformed_provider_response",
                "Parallel returned a malformed fetch response.",
            )

    async def _post_json(
        self,
        ctx: ToolContext,
        *,
        path: str,
        action: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> _ParallelHttpResponse | ToolResult:
        proxy = ctx.proxy
        if proxy is None:
            return _error_result(
                "credential_authority_unavailable",
                "Parallel requires an active credential proxy.",
            )
        try:
            authorization = await proxy.authorize_request(
                destination=self.origin,
                credential=self.api_key_ref,
                action=action,
                metadata={"method": "POST", "path": path},
            )
            if type(authorization) is not ProxyAuthorizationResult:
                return _error_result(
                    "credential_authority_invalid",
                    "The credential proxy returned an invalid authorization result.",
                )
            if not authorization.allowed:
                return _error_result(
                    "credential_denied",
                    "The credential proxy denied the Parallel request.",
                )
            resolved = await proxy.resolve(
                self.api_key_ref,
                scope={"destination": self.origin, "provider": "parallel"},
            )
            if type(resolved) is not ResolvedSecret:
                return _error_result(
                    "credential_authority_invalid",
                    "The credential proxy returned an invalid resolved secret.",
                )
        except httpx.TimeoutException:
            return _error_result("timeout", "The Parallel request timed out.")
        except Exception:
            return _error_result(
                "credential_unavailable",
                "The Parallel credential could not be authorized or resolved.",
            )

        redactor = active_secret_redactor(ctx).with_secret(resolved)
        prepared_payload = redactor.redact_json_values(dict(payload))
        if prepared_payload != payload:
            return _error_result(
                "secret_exposure_denied",
                "The Parallel request payload contains a protected secret.",
            )

        api_key = resolved.value.get_secret_value()
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Content-Type": "application/json",
            "User-Agent": "Cayu-Parallel-Web/0.1",
            "x-api-key": api_key,
        }
        try:
            timeout = httpx.Timeout(timeout_seconds)
            unsupported_response = False
            async with (
                httpx.AsyncClient(
                    base_url=self.origin,
                    timeout=timeout,
                    transport=self._transport,
                    trust_env=False,
                    follow_redirects=False,
                ) as client,
                client.stream(
                    "POST",
                    path,
                    headers=headers,
                    json=prepared_payload,
                ) as response,
            ):
                status_code = response.status_code
                response_headers = {key.lower(): value for key, value in response.headers.items()}
                body = bytearray()
                if 200 <= status_code < 300:
                    content_encoding = response.headers.get("content-encoding", "").strip().lower()
                    if content_encoding not in {"", "identity"}:
                        unsupported_response = True
                    else:
                        content_length = response.headers.get("content-length")
                        if content_length is not None:
                            try:
                                announced_bytes = int(content_length)
                            except ValueError:
                                announced_bytes = 0
                            if announced_bytes > max_response_bytes:
                                raise _ParallelOversizedResponseError
                        async for chunk in response.aiter_bytes():
                            if len(body) + len(chunk) > max_response_bytes:
                                raise _ParallelOversizedResponseError
                            body.extend(chunk)
            if unsupported_response:
                return _error_result(
                    "unsupported_provider_response",
                    "The Parallel response encoding is unsupported.",
                )
            return _ParallelHttpResponse(
                status_code=status_code,
                headers=response_headers,
                body=bytes(body),
                redactor=active_secret_redactor(ctx).with_secret(resolved),
            )
        except _ParallelOversizedResponseError:
            return _error_result(
                "oversized_provider_response",
                "The Parallel response exceeded the configured byte limit.",
            )
        except httpx.TimeoutException:
            return _error_result("timeout", "The Parallel request timed out.")
        except httpx.RequestError:
            return _error_result("provider_unavailable", "The Parallel request failed.")
        finally:
            api_key = ""
            headers.clear()


def _parallel_search_result(
    request: WebSearchAdapterRequest,
    document: Mapping[str, Any],
    *,
    redactor: SecretRedactor,
) -> ToolResult:
    raw_results = document.get("results")
    if type(raw_results) is not list:
        raise _ParallelMalformedResponseError
    provider_metadata, metadata_truncated = _parallel_provider_metadata(
        document,
        id_field="search_id",
        redactor=redactor,
    )
    results: list[dict[str, Any]] = []
    truncation_reasons: list[str] = []
    if metadata_truncated:
        truncation_reasons.append("provider_metadata")
    if len(raw_results) > request.max_results:
        truncation_reasons.append("result_count")
    remaining_snippet_bytes = request.max_total_snippet_bytes
    for index, raw_result in enumerate(raw_results[: request.max_results], start=1):
        if type(raw_result) is not dict:
            raise _ParallelMalformedResponseError
        result, consumed, result_reasons = _parallel_search_item(
            raw_result,
            rank=index,
            redactor=redactor,
            max_snippet_bytes=request.max_snippet_bytes,
            remaining_snippet_bytes=remaining_snippet_bytes,
        )
        remaining_snippet_bytes -= consumed
        results.append(result)
        for reason in result_reasons:
            if reason not in truncation_reasons:
                truncation_reasons.append(reason)
    return _web_search_success_result(
        query=redactor.redact_text_bounded(
            request.query,
            max_bytes=MAX_WEB_SEARCH_QUERY_BYTES,
        ),
        results=results,
        truncation_reasons=truncation_reasons,
        provider_metadata=provider_metadata,
        projection_redactor=redactor,
    )


def _parallel_search_item(
    value: Mapping[str, Any],
    *,
    rank: int,
    redactor: SecretRedactor,
    max_snippet_bytes: int,
    remaining_snippet_bytes: int,
) -> tuple[dict[str, Any], int, tuple[str, ...]]:
    url = _parallel_provider_url(value.get("url"), redactor=redactor)
    raw_title = value.get("title")
    title, title_truncated = _parallel_provider_text(
        url if raw_title is None else raw_title,
        redactor=redactor,
        max_bytes=MAX_WEB_FETCH_TITLE_BYTES,
        preserve_line_breaks=False,
        require_nonblank=True,
    )
    raw_excerpts = value.get("excerpts")
    if type(raw_excerpts) is not list:
        raise _ParallelMalformedResponseError
    count_truncated = len(raw_excerpts) > MAX_PARALLEL_EXCERPTS_PER_RESULT
    snippets: list[str] = []
    consumed = 0
    reasons: list[str] = []
    if title_truncated:
        reasons.append("title")
    if count_truncated:
        reasons.append("snippet_count")
    for index, raw_excerpt in enumerate(raw_excerpts[:MAX_PARALLEL_EXCERPTS_PER_RESULT]):
        if remaining_snippet_bytes <= 0:
            if "total_snippet_bytes" not in reasons:
                reasons.append("total_snippet_bytes")
            break
        snippet_limit = min(max_snippet_bytes, remaining_snippet_bytes)
        snippet, snippet_truncated = _parallel_provider_text(
            raw_excerpt,
            redactor=redactor,
            max_bytes=snippet_limit,
            preserve_line_breaks=True,
            require_nonblank=False,
        )
        if snippet:
            snippets.append(snippet)
            snippet_bytes = len(snippet.encode("utf-8"))
            consumed += snippet_bytes
            remaining_snippet_bytes -= snippet_bytes
        if snippet_truncated and "snippet" not in reasons:
            reasons.append("snippet")
        if remaining_snippet_bytes <= 0 and index + 1 < len(raw_excerpts):
            if "total_snippet_bytes" not in reasons:
                reasons.append("total_snippet_bytes")
            break
    result = {
        "rank": rank,
        "url": url,
        "title": title,
        "snippets": snippets,
        "published_at": _parallel_publication_date(
            value.get("publish_date"),
            redactor=redactor,
        ),
    }
    return result, consumed, tuple(reasons)


def _parallel_fetch_result(
    request: WebFetchAdapterRequest,
    document: Mapping[str, Any],
    *,
    redactor: SecretRedactor,
) -> ToolResult:
    raw_results = document.get("results")
    raw_errors = document.get("errors")
    if type(raw_results) is not list or type(raw_errors) is not list:
        raise _ParallelMalformedResponseError
    provider_metadata, metadata_truncated = _parallel_provider_metadata(
        document,
        id_field="extract_id",
        redactor=redactor,
    )
    if raw_errors:
        if raw_results or len(raw_errors) != 1 or type(raw_errors[0]) is not dict:
            raise _ParallelMalformedResponseError
        return _parallel_extract_error_result(
            request,
            raw_errors[0],
            provider_metadata=provider_metadata,
            redactor=redactor,
        )
    if len(raw_results) != 1 or type(raw_results[0]) is not dict:
        raise _ParallelMalformedResponseError
    raw_result = raw_results[0]
    final_url = _parallel_provider_url(raw_result.get("url"), redactor=redactor)
    if final_url != request.requested_url:
        return _error_result(
            "unsupported_semantics",
            "Parallel returned a different final URL without redirect provenance.",
        )
    title_value = raw_result.get("title")
    if title_value is None:
        title = None
        title_truncated = False
    else:
        title, title_truncated = _parallel_provider_text(
            title_value,
            redactor=redactor,
            max_bytes=MAX_WEB_FETCH_TITLE_BYTES,
            preserve_line_breaks=False,
            require_nonblank=False,
        )
        title = title or None
    raw_excerpts = raw_result.get("excerpts")
    if type(raw_excerpts) is not list:
        raise _ParallelMalformedResponseError
    content, content_truncated, excerpt_count_truncated = _parallel_fetch_content(
        raw_excerpts,
        redactor=redactor,
        max_bytes=request.max_content_bytes,
    )
    full_content = raw_result.get("full_content")
    if full_content is not None and type(full_content) is not str:
        raise _ParallelMalformedResponseError
    truncation_reasons: list[str] = []
    if title_truncated:
        truncation_reasons.append("title")
    if content_truncated:
        truncation_reasons.append("content")
    if excerpt_count_truncated:
        truncation_reasons.append("excerpt_count")
    if metadata_truncated:
        truncation_reasons.append("provider_metadata")
    return _web_fetch_success_result(
        requested_url=request.requested_url,
        final_url=final_url,
        title=title,
        representation="text",
        content=content,
        redirects=(),
        truncation_reasons=truncation_reasons,
        provider_metadata=provider_metadata,
        projection_redactor=redactor,
    )


def _parallel_fetch_content(
    raw_excerpts: Sequence[Any],
    *,
    redactor: SecretRedactor,
    max_bytes: int,
) -> tuple[str, bool, bool]:
    count_truncated = len(raw_excerpts) > MAX_PARALLEL_EXCERPTS_PER_RESULT
    normalizer = _BoundedNormalizedText(max_bytes, preserve_line_breaks=True)
    retained = 0
    for raw_excerpt in raw_excerpts[:MAX_PARALLEL_EXCERPTS_PER_RESULT]:
        if (
            type(raw_excerpt) is not str
            or "\x00" in raw_excerpt
            or any(0xD800 <= ord(character) <= 0xDFFF for character in raw_excerpt)
        ):
            raise _ParallelMalformedResponseError
        redacted = redactor.redact_text(raw_excerpt)
        if not redacted.strip():
            continue
        if retained:
            normalizer.feed("\n\n")
        normalizer.feed(redacted)
        retained += 1
    content, redaction_truncated = redactor.redact_text_head(
        normalizer.value,
        max_bytes=max_bytes,
    )
    return content, normalizer.truncated or redaction_truncated, count_truncated


def _parallel_extract_error_result(
    request: WebFetchAdapterRequest,
    value: Mapping[str, Any],
    *,
    provider_metadata: Mapping[str, Any],
    redactor: SecretRedactor,
) -> ToolResult:
    error_url = _parallel_provider_url(value.get("url"), redactor=redactor)
    if error_url != request.requested_url:
        raise _ParallelMalformedResponseError
    error_type, _ = _parallel_provider_text(
        value.get("error_type"),
        redactor=redactor,
        max_bytes=MAX_PARALLEL_WARNING_TYPE_BYTES,
        preserve_line_breaks=False,
        require_nonblank=True,
    )
    status_code = _parallel_optional_provider_int(
        value.get("http_status_code"),
        redactor=redactor,
        minimum=100,
        maximum=599,
    )
    raw_content = value.get("content")
    if raw_content is not None and type(raw_content) is not str:
        raise _ParallelMalformedResponseError
    parallel_metadata = dict(provider_metadata["parallel"])
    parallel_metadata["extract_error_type"] = error_type
    structured: dict[str, Any] = {
        "error": "fetch_failed",
        "provider_metadata": {"parallel": parallel_metadata},
    }
    if status_code is not None:
        structured["status_code"] = status_code
    return ToolResult(
        content="The hosted provider could not retrieve the requested URL.",
        structured=structured,
        is_error=True,
    )


def _parallel_provider_metadata(
    document: Mapping[str, Any],
    *,
    id_field: Literal["search_id", "extract_id"],
    redactor: SecretRedactor,
) -> tuple[dict[str, Any], bool]:
    metadata: dict[str, Any] = {}
    truncated = False
    request_id, was_truncated = _parallel_provider_text(
        document.get(id_field),
        redactor=redactor,
        max_bytes=MAX_PARALLEL_REQUEST_ID_BYTES,
        preserve_line_breaks=False,
        require_nonblank=True,
    )
    metadata["request_id"] = request_id
    truncated = truncated or was_truncated
    session_id, was_truncated = _parallel_provider_text(
        document.get("session_id"),
        redactor=redactor,
        max_bytes=MAX_PARALLEL_SESSION_ID_BYTES,
        preserve_line_breaks=False,
        require_nonblank=True,
    )
    metadata["session_id"] = session_id
    truncated = truncated or was_truncated
    raw_warnings = document.get("warnings")
    if raw_warnings is not None:
        if type(raw_warnings) is not list:
            raise _ParallelMalformedResponseError
        warnings: list[dict[str, str]] = []
        for warning in raw_warnings[:MAX_PARALLEL_WARNINGS]:
            if type(warning) is not dict:
                raise _ParallelMalformedResponseError
            warning_type, type_truncated = _parallel_provider_text(
                warning.get("type"),
                redactor=redactor,
                max_bytes=MAX_PARALLEL_WARNING_TYPE_BYTES,
                preserve_line_breaks=False,
                require_nonblank=True,
            )
            message, message_truncated = _parallel_provider_text(
                warning.get("message"),
                redactor=redactor,
                max_bytes=MAX_PARALLEL_WARNING_BYTES,
                preserve_line_breaks=False,
                require_nonblank=True,
            )
            warnings.append({"type": warning_type, "message": message})
            truncated = truncated or type_truncated or message_truncated
        if len(raw_warnings) > MAX_PARALLEL_WARNINGS:
            truncated = True
        metadata["warnings"] = warnings
    raw_usage = document.get("usage")
    if raw_usage is not None:
        if type(raw_usage) is not list:
            raise _ParallelMalformedResponseError
        usage: list[dict[str, Any]] = []
        for item in raw_usage[:MAX_PARALLEL_USAGE_ITEMS]:
            if type(item) is not dict:
                raise _ParallelMalformedResponseError
            name, name_truncated = _parallel_provider_text(
                item.get("name"),
                redactor=redactor,
                max_bytes=MAX_PARALLEL_USAGE_NAME_BYTES,
                preserve_line_breaks=False,
                require_nonblank=True,
            )
            count = _parallel_provider_int(
                item.get("count"),
                redactor=redactor,
                minimum=0,
                maximum=2**63 - 1,
            )
            if count is None:
                truncated = True
                continue
            usage.append({"name": name, "count": count})
            truncated = truncated or name_truncated
        if len(raw_usage) > MAX_PARALLEL_USAGE_ITEMS:
            truncated = True
        metadata["usage"] = usage
    return {"parallel": metadata}, truncated


def _parallel_http_error_result(response: _ParallelHttpResponse) -> ToolResult:
    if response.status_code in {401, 403}:
        code = "provider_authentication_failed"
        message = "Parallel rejected the configured credential."
    elif response.status_code == 402:
        code = "provider_quota_exhausted"
        message = "The Parallel account cannot accept this request."
    elif response.status_code == 429:
        code = "rate_limited"
        message = "Parallel rate-limited the request."
    elif response.status_code in {400, 404, 409, 422}:
        code = "provider_request_rejected"
        message = "Parallel rejected the bounded request."
    elif response.status_code >= 500:
        code = "provider_unavailable"
        message = "Parallel is temporarily unavailable."
    else:
        code = "provider_error"
        message = "Parallel returned an unsuccessful response."
    parallel_metadata: dict[str, Any] = {}
    request_id = _parallel_bounded_optional_text(
        response.headers.get("x-request-id"),
        redactor=response.redactor,
        max_bytes=MAX_PARALLEL_REQUEST_ID_BYTES,
    )
    if request_id is not None:
        parallel_metadata["request_id"] = request_id
    retry_after_header = response.headers.get("retry-after")
    retry_after = None
    if not _parallel_text_contains_secret(retry_after_header, redactor=response.redactor):
        retry_after, retry_after_unrepresentable = _parallel_retry_after_seconds(retry_after_header)
    else:
        retry_after_unrepresentable = False
    if retry_after is not None and not _parallel_scalar_contains_secret(
        retry_after,
        redactor=response.redactor,
    ):
        parallel_metadata["retry_after_seconds"] = retry_after
    if retry_after_unrepresentable:
        parallel_metadata["retry_after_unrepresentable"] = True
    structured: dict[str, Any] = {"error": code}
    if not _parallel_scalar_contains_secret(response.status_code, redactor=response.redactor):
        structured["status_code"] = response.status_code
    if parallel_metadata:
        structured["provider_metadata"] = {"parallel": parallel_metadata}
    return ToolResult(content=message, structured=structured, is_error=True)


def _parallel_unsupported_restriction(request: WebSearchAdapterRequest) -> str | None:
    restrictions = request.restrictions
    if restrictions.country is not None:
        return "country"
    if restrictions.locale is not None:
        return "locale"
    if restrictions.content_types:
        return "content-type"
    return None


def _parallel_fetch_policy(
    max_age_seconds: int | None,
    disable_cache_fallback: bool,
    *,
    timeout_seconds: float,
) -> dict[str, Any] | None:
    if max_age_seconds is None:
        if disable_cache_fallback:
            raise ValueError("cache fallback cannot be disabled without a maximum cache age.")
        return None
    return {
        "max_age_seconds": max_age_seconds,
        "timeout_seconds": timeout_seconds,
        "disable_cache_fallback": disable_cache_fallback,
    }


def _parallel_provider_url(value: Any, *, redactor: SecretRedactor) -> str:
    try:
        canonical = _canonicalize_url(value)
    except (TypeError, ValueError) as exc:
        raise _ParallelMalformedResponseError from exc
    if redactor.redact_text(canonical) != canonical:
        raise _ParallelMalformedResponseError
    return canonical


def _parallel_provider_text(
    value: Any,
    *,
    redactor: SecretRedactor,
    max_bytes: int,
    preserve_line_breaks: bool,
    require_nonblank: bool,
) -> tuple[str, bool]:
    if (
        type(value) is not str
        or "\x00" in value
        or any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    ):
        raise _ParallelMalformedResponseError
    normalizer = _BoundedNormalizedText(
        max_bytes,
        preserve_line_breaks=preserve_line_breaks,
    )
    normalizer.feed(redactor.redact_text(value))
    bounded, redaction_truncated = redactor.redact_text_head(
        normalizer.value,
        max_bytes=max_bytes,
    )
    if require_nonblank and not bounded:
        raise _ParallelMalformedResponseError
    return bounded, normalizer.truncated or redaction_truncated


def _parallel_bounded_optional_text(
    value: Any,
    *,
    redactor: SecretRedactor,
    max_bytes: int,
) -> str | None:
    if value is None:
        return None
    try:
        bounded, _ = _parallel_provider_text(
            value,
            redactor=redactor,
            max_bytes=max_bytes,
            preserve_line_breaks=False,
            require_nonblank=True,
        )
    except _ParallelMalformedResponseError:
        return None
    return bounded


def _parallel_provider_int(
    value: Any,
    *,
    redactor: SecretRedactor,
    minimum: int,
    maximum: int,
) -> int | None:
    if type(value) is not int or value < minimum or value > maximum:
        raise _ParallelMalformedResponseError
    if _parallel_scalar_contains_secret(value, redactor=redactor):
        return None
    return value


def _parallel_optional_provider_int(
    value: Any,
    *,
    redactor: SecretRedactor,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    return _parallel_provider_int(
        value,
        redactor=redactor,
        minimum=minimum,
        maximum=maximum,
    )


def _parallel_scalar_contains_secret(value: int | float, *, redactor: SecretRedactor) -> bool:
    rendered = str(value)
    return redactor.redact_text(rendered) != rendered


def _parallel_text_contains_secret(value: str | None, *, redactor: SecretRedactor) -> bool:
    return value is not None and redactor.redact_text(value) != value


def _parallel_publication_date(value: Any, *, redactor: SecretRedactor) -> str | None:
    if value is None:
        return None
    if type(value) is not str or redactor.redact_text(value) != value:
        raise _ParallelMalformedResponseError
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _ParallelMalformedResponseError from exc
    if parsed.isoformat() != value:
        raise _ParallelMalformedResponseError
    return value


def _parallel_json_object(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_parallel_unique_json_object,
            parse_constant=_parallel_reject_json_constant,
        )
    except (
        UnicodeDecodeError,
        ValueError,
        RecursionError,
        _ParallelMalformedResponseError,
    ) as exc:
        raise _ParallelMalformedResponseError from exc
    if type(value) is not dict:
        raise _ParallelMalformedResponseError
    return value


def _parallel_unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _ParallelMalformedResponseError
        value[key] = item
    return value


def _parallel_reject_json_constant(value: str) -> Any:
    del value
    raise _ParallelMalformedResponseError


def _parallel_retry_after_seconds(value: str | None) -> tuple[float | None, bool]:
    if value is None:
        return None, False
    value = value.strip()
    if not value:
        return None, False
    try:
        seconds = float(value)
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None, False
        if target is None:
            return None, False
        if target.tzinfo is None:
            return None, False
        seconds = max(0.0, (target - datetime.now(UTC)).total_seconds())
    if not math.isfinite(seconds):
        try:
            exact_seconds = Decimal(value)
        except InvalidOperation:
            return None, False
        return None, exact_seconds.is_finite() and exact_seconds >= 0
    if seconds < 0:
        return None, False
    if seconds > _MAX_PARALLEL_RETRY_AFTER_SECONDS:
        return None, True
    return seconds, False


def _parallel_origin(value: Any) -> str:
    if type(value) is not str:
        raise ValueError("origin must be a credentialless HTTPS origin without a path.")
    raw = urlsplit(value)
    if raw.path not in {"", "/"} or raw.query or raw.fragment:
        raise ValueError("origin must be a credentialless HTTPS origin without a path.")
    canonical = _canonicalize_url(value)
    split = urlsplit(canonical)
    if split.path != "/" or split.query:
        raise ValueError("origin must be a credentialless HTTPS origin without a path.")
    return canonical.rstrip("/")


def _parallel_objective(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or "\x00" in value
        or any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    ):
        raise ValueError(f"{name} must be a nonblank string or None.")
    normalized = " ".join(value.split())
    if not normalized or not _parallel_objective_within_provider_limit(normalized):
        raise ValueError(f"{name} must be a nonblank bounded string or None.")
    return normalized


def _parallel_objective_within_provider_limit(value: str) -> bool:
    return (
        len(value) <= MAX_PARALLEL_OBJECTIVE_CHARACTERS
        and len(value.encode("utf-8")) <= MAX_PARALLEL_OBJECTIVE_BYTES
    )


def _parallel_search_location(value: Any) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError("search_location must be a string or None.")
    normalized = value.strip().lower()
    if len(normalized) != 2 or not normalized.isascii() or not normalized.isalpha():
        raise ValueError("search_location must be an ISO 3166-1 alpha-2 country code or None.")
    return normalized


def _parallel_max_age_seconds(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 600 or value > MAX_PARALLEL_FETCH_AGE_SECONDS:
        raise ValueError(
            f"{name} must be None or an integer between 600 and {MAX_PARALLEL_FETCH_AGE_SECONDS}."
        )
    return value


__all__ = [
    "ParallelAIWebAdapter",
]
