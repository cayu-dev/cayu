"""Hosted Exa adapters for Cayu's provider-neutral web tools."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
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

DEFAULT_EXA_ORIGIN = "https://api.exa.ai"
DEFAULT_EXA_SEARCH_TYPE = "auto"
DEFAULT_EXA_MAX_PROVIDER_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_EXA_PROVIDER_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_EXA_CONTENT_CHARACTERS = 10_000
MAX_EXA_URL_CHARACTERS = 2_048
MAX_EXA_REQUEST_ID_BYTES = 256
MAX_EXA_WARNING_BYTES = 512
MAX_EXA_WARNINGS = 16
MAX_EXA_PUBLICATION_TIME_BYTES = 64
_MAX_EXA_SNIPPETS_PER_RESULT = 8
_MAX_EXA_RETRY_AFTER_SECONDS = 24 * 60 * 60

_EXA_SEARCH_TYPES = frozenset({"instant", "fast", "auto", "deep-lite", "deep", "deep-reasoning"})
_EXA_AUTH_HEADERS = frozenset({"x-api-key", "authorization"})


@dataclass(frozen=True)
class _ExaHttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes
    redactor: SecretRedactor


class _ExaOversizedResponseError(Exception):
    pass


class _ExaMalformedResponseError(Exception):
    pass


class ExaWebAdapter:
    """Implement provider-neutral search and known-URL fetch through Exa.

    The API key remains an application-owned :class:`SecretRef` and is resolved
    through the invocation's credential proxy immediately before each request.
    """

    def __init__(
        self,
        *,
        api_key_ref: SecretRef,
        origin: str = DEFAULT_EXA_ORIGIN,
        auth_header: Literal["x-api-key", "authorization"] = "x-api-key",
        search_type: Literal[
            "instant",
            "fast",
            "auto",
            "deep-lite",
            "deep",
            "deep-reasoning",
        ] = DEFAULT_EXA_SEARCH_TYPE,
        search_max_age_hours: int | None = None,
        fetch_max_age_hours: int | None = None,
        moderation: bool = False,
        max_provider_response_bytes: int = DEFAULT_EXA_MAX_PROVIDER_RESPONSE_BYTES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if type(api_key_ref) is not SecretRef:
            raise TypeError("api_key_ref must be a SecretRef.")
        self.api_key_ref = copy_secret_ref(api_key_ref)
        self.origin = _exa_origin(origin)
        if type(auth_header) is not str or auth_header not in _EXA_AUTH_HEADERS:
            raise ValueError("auth_header must be 'x-api-key' or 'authorization'.")
        self.auth_header = auth_header
        if type(search_type) is not str or search_type not in _EXA_SEARCH_TYPES:
            raise ValueError("search_type is not supported by this Exa adapter.")
        self.search_type = search_type
        self.search_max_age_hours = _max_age_hours(
            search_max_age_hours,
            "search_max_age_hours",
        )
        self.fetch_max_age_hours = _max_age_hours(
            fetch_max_age_hours,
            "fetch_max_age_hours",
        )
        if type(moderation) is not bool:
            raise TypeError("moderation must be a boolean.")
        self.moderation = moderation
        self.max_provider_response_bytes = _configuration_int(
            max_provider_response_bytes,
            "max_provider_response_bytes",
            minimum=1,
            maximum=MAX_EXA_PROVIDER_RESPONSE_BYTES,
        )
        if transport is not None and not isinstance(transport, httpx.AsyncBaseTransport):
            raise TypeError("transport must be an httpx.AsyncBaseTransport.")
        self._transport = transport

    async def search(
        self,
        ctx: ToolContext,
        request: WebSearchAdapterRequest,
    ) -> ToolResult:
        if (
            request.restrictions.include_domains
            or request.restrictions.exclude_domains
            or request.restrictions.published_on_or_after is not None
            or request.restrictions.country is not None
            or request.restrictions.locale is not None
            or request.restrictions.content_types
        ):
            return _error_result(
                "unsupported_semantics",
                "This Exa adapter cannot enforce the configured search restrictions.",
            )
        contents: dict[str, Any] = {
            "highlights": {
                "query": request.query,
                "maxCharacters": min(
                    request.max_snippet_bytes,
                    MAX_EXA_CONTENT_CHARACTERS,
                ),
            }
        }
        payload: dict[str, Any] = {
            "query": request.query,
            "type": self.search_type,
            "numResults": request.max_results,
            "moderation": self.moderation,
            "contents": contents,
        }
        if self.search_max_age_hours is not None:
            contents["maxAgeHours"] = self.search_max_age_hours
        response = await self._post_json(
            ctx,
            path="/search",
            action="exa.search",
            payload=payload,
            timeout_seconds=request.timeout_seconds,
            max_response_bytes=self.max_provider_response_bytes,
        )
        if isinstance(response, ToolResult):
            return response
        if response.status_code < 200 or response.status_code >= 300:
            return _exa_http_error_result(response)
        try:
            document = _json_object(response.body)
            return _exa_search_result(
                request,
                document,
                response.headers,
                redactor=response.redactor,
            )
        except _ExaMalformedResponseError:
            return _error_result(
                "malformed_provider_response",
                "Exa returned a malformed search response.",
            )

    async def fetch(
        self,
        ctx: ToolContext,
        request: WebFetchAdapterRequest,
    ) -> ToolResult:
        if len(request.requested_url) > MAX_EXA_URL_CHARACTERS:
            return _error_result(
                "unsupported_semantics",
                "The hosted fetch provider cannot accept a URL of this length.",
            )
        requested_characters = min(
            request.max_content_bytes,
            MAX_EXA_CONTENT_CHARACTERS,
        )
        payload: dict[str, Any] = {
            "urls": [request.requested_url],
            "text": {
                "maxCharacters": requested_characters,
                "includeHtmlTags": False,
                "verbosity": "compact",
            },
        }
        if self.fetch_max_age_hours is not None:
            payload["maxAgeHours"] = self.fetch_max_age_hours
        response = await self._post_json(
            ctx,
            path="/contents",
            action="exa.contents",
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
            return _exa_http_error_result(response)
        try:
            document = _json_object(response.body)
            return _exa_fetch_result(
                request,
                document,
                response.headers,
                redactor=response.redactor,
            )
        except _ExaMalformedResponseError:
            return _error_result(
                "malformed_provider_response",
                "Exa returned a malformed fetch response.",
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
    ) -> _ExaHttpResponse | ToolResult:
        proxy = ctx.proxy
        if proxy is None:
            return _error_result(
                "credential_authority_unavailable",
                "Exa requires an active credential proxy.",
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
                    "The credential proxy denied the Exa request.",
                )
            resolved = await proxy.resolve(
                self.api_key_ref,
                scope={"destination": self.origin, "provider": "exa"},
            )
            if type(resolved) is not ResolvedSecret:
                return _error_result(
                    "credential_authority_invalid",
                    "The credential proxy returned an invalid resolved secret.",
                )
        except httpx.TimeoutException:
            return _error_result("timeout", "The Exa request timed out.")
        except Exception:
            return _error_result(
                "credential_unavailable",
                "The Exa credential could not be authorized or resolved.",
            )

        redactor = active_secret_redactor(ctx).with_secret(resolved)
        prepared_payload = redactor.redact_json_values(dict(payload))
        if prepared_payload != payload:
            return _error_result(
                "secret_exposure_denied",
                "The Exa request payload contains a protected secret.",
            )

        api_key = resolved.value.get_secret_value()
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Content-Type": "application/json",
            "User-Agent": "Cayu-Exa-Web/0.1",
        }
        if self.auth_header == "authorization":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["x-api-key"] = api_key
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
                                raise _ExaOversizedResponseError
                        # Non-identity content encodings are rejected above, so
                        # the decoded iterator is byte-equivalent while remaining
                        # compatible with preloaded and streaming HTTPX transports.
                        async for chunk in response.aiter_bytes():
                            if len(body) + len(chunk) > max_response_bytes:
                                raise _ExaOversizedResponseError
                            body.extend(chunk)
            if unsupported_response:
                return _error_result(
                    "unsupported_provider_response",
                    "The Exa response encoding is unsupported.",
                )
            return _ExaHttpResponse(
                status_code=status_code,
                headers=response_headers,
                body=bytes(body),
                redactor=active_secret_redactor(ctx).with_secret(resolved),
            )
        except _ExaOversizedResponseError:
            return _error_result(
                "oversized_provider_response",
                "The Exa response exceeded the configured byte limit.",
            )
        except httpx.TimeoutException:
            return _error_result("timeout", "The Exa request timed out.")
        except httpx.RequestError:
            return _error_result("provider_unavailable", "The Exa request failed.")
        finally:
            api_key = ""
            headers.clear()


def _exa_search_result(
    request: WebSearchAdapterRequest,
    document: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    redactor: SecretRedactor,
) -> ToolResult:
    raw_results = document.get("results")
    if type(raw_results) is not list:
        raise _ExaMalformedResponseError
    provider_metadata, metadata_truncated = _exa_provider_metadata(
        document,
        headers,
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
            raise _ExaMalformedResponseError
        result, consumed, result_reasons = _exa_search_item(
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
    )


def _exa_search_item(
    value: Mapping[str, Any],
    *,
    rank: int,
    redactor: SecretRedactor,
    max_snippet_bytes: int,
    remaining_snippet_bytes: int,
) -> tuple[dict[str, Any], int, tuple[str, ...]]:
    url = _provider_url(value.get("url"), redactor=redactor)
    if "title" not in value:
        raise _ExaMalformedResponseError
    raw_title = value["title"]
    title, title_truncated = _provider_text(
        url if raw_title is None else raw_title,
        redactor=redactor,
        max_bytes=MAX_WEB_FETCH_TITLE_BYTES,
        preserve_line_breaks=False,
        require_nonblank=True,
    )
    raw_highlights = value.get("highlights")
    raw_text = value.get("text")
    snippet_count_truncated = False
    if raw_highlights is None:
        all_snippets = [] if raw_text is None else [raw_text]
    elif type(raw_highlights) is list:
        all_snippets = raw_highlights
        snippet_count_truncated = len(raw_highlights) > _MAX_EXA_SNIPPETS_PER_RESULT
    else:
        raise _ExaMalformedResponseError
    raw_scores = value.get("highlightScores")
    scores = _highlight_scores(raw_scores, count=len(all_snippets))
    snippets_source = all_snippets[:_MAX_EXA_SNIPPETS_PER_RESULT]
    snippets: list[str] = []
    retained_scores: list[float] = []
    consumed = 0
    reasons: list[str] = []
    if title_truncated:
        reasons.append("title")
    if snippet_count_truncated:
        reasons.append("snippet_count")
    for snippet_index, raw_snippet in enumerate(snippets_source):
        if remaining_snippet_bytes <= 0:
            if "total_snippet_bytes" not in reasons:
                reasons.append("total_snippet_bytes")
            break
        snippet_limit = min(max_snippet_bytes, remaining_snippet_bytes)
        snippet, snippet_truncated = _provider_text(
            raw_snippet,
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
            if scores is not None:
                retained_scores.append(scores[snippet_index])
        if snippet_truncated and "snippet" not in reasons:
            reasons.append("snippet")
        if remaining_snippet_bytes <= 0 and snippet_index + 1 < len(snippets_source):
            if "total_snippet_bytes" not in reasons:
                reasons.append("total_snippet_bytes")
            break
    result: dict[str, Any] = {
        "rank": rank,
        "url": url,
        "title": title,
        "snippets": snippets,
        "published_at": _publication_time(
            value.get("publishedDate"),
            redactor=redactor,
        ),
    }
    if retained_scores:
        result["provider_metadata"] = {"exa": {"highlight_scores": retained_scores}}
    return result, consumed, tuple(reasons)


def _exa_fetch_result(
    request: WebFetchAdapterRequest,
    document: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    redactor: SecretRedactor,
) -> ToolResult:
    raw_results = document.get("results")
    if type(raw_results) is not list:
        raise _ExaMalformedResponseError
    status_error = _exa_contents_status_error(document)
    if status_error is not None:
        return status_error
    if len(raw_results) != 1 or type(raw_results[0]) is not dict:
        raise _ExaMalformedResponseError
    raw_result = raw_results[0]
    final_url = _provider_url(raw_result.get("url"), redactor=redactor)
    if final_url != request.requested_url:
        return _error_result(
            "unsupported_semantics",
            "Exa returned a different final URL without redirect provenance.",
        )
    title_value = raw_result.get("title")
    if title_value is None:
        title = None
        title_truncated = False
    else:
        title, title_truncated = _provider_text(
            title_value,
            redactor=redactor,
            max_bytes=MAX_WEB_FETCH_TITLE_BYTES,
            preserve_line_breaks=False,
            require_nonblank=False,
        )
        title = title or None
    raw_content = raw_result.get("text")
    content, content_truncated = _provider_text(
        raw_content,
        redactor=redactor,
        max_bytes=request.max_content_bytes,
        preserve_line_breaks=True,
        require_nonblank=False,
    )
    truncation_reasons: list[str] = []
    if title_truncated:
        truncation_reasons.append("title")
    if content_truncated:
        truncation_reasons.append("content")
    requested_characters = min(
        request.max_content_bytes,
        MAX_EXA_CONTENT_CHARACTERS,
    )
    if len(raw_content) >= requested_characters:
        truncation_reasons.append("provider_content_limit")
    provider_metadata, metadata_truncated = _exa_provider_metadata(
        document,
        headers,
        redactor=redactor,
    )
    if metadata_truncated:
        truncation_reasons.append("provider_metadata")
    result = _web_fetch_success_result(
        requested_url=request.requested_url,
        final_url=final_url,
        title=title,
        representation="text",
        content=content,
        redirects=(),
        truncation_reasons=truncation_reasons,
        provider_metadata=provider_metadata,
    )
    return result


def _exa_contents_status_error(
    document: Mapping[str, Any],
) -> ToolResult | None:
    statuses = document.get("statuses")
    if statuses is None:
        return None
    if type(statuses) is not list:
        raise _ExaMalformedResponseError
    for status in statuses:
        if type(status) is not dict:
            raise _ExaMalformedResponseError
    if len(statuses) > 1:
        raise _ExaMalformedResponseError
    if not statuses:
        return None
    status = statuses[0]
    status_id = status.get("id")
    if type(status_id) is not str or not status_id:
        raise _ExaMalformedResponseError
    status_value = status.get("status")
    if status_value == "success":
        return None
    if status_value != "error":
        raise _ExaMalformedResponseError
    error = status.get("error")
    status_code: int | None = None
    if error is not None:
        if type(error) is not dict:
            raise _ExaMalformedResponseError
        candidate = error.get("httpStatusCode")
        if candidate is not None:
            if type(candidate) is not int or candidate < 100 or candidate > 599:
                raise _ExaMalformedResponseError
            status_code = candidate
    structured: dict[str, Any] = {"error": "fetch_failed"}
    if status_code is not None:
        structured["status_code"] = status_code
    return ToolResult(
        content="The hosted provider could not retrieve the requested URL.",
        structured=structured,
        is_error=True,
    )


def _exa_http_error_result(response: _ExaHttpResponse) -> ToolResult:
    if response.status_code in {401, 403}:
        code = "provider_authentication_failed"
        message = "Exa rejected the configured credential."
    elif response.status_code == 402:
        code = "provider_quota_exhausted"
        message = "The Exa account cannot accept this request."
    elif response.status_code == 429:
        code = "rate_limited"
        message = "Exa rate-limited the request."
    elif response.status_code in {400, 404, 409, 422}:
        code = "provider_request_rejected"
        message = "Exa rejected the bounded request."
    elif response.status_code >= 500:
        code = "provider_unavailable"
        message = "Exa is temporarily unavailable."
    else:
        code = "provider_error"
        message = "Exa returned an unsuccessful response."
    exa_metadata: dict[str, Any] = {}
    request_id = _bounded_optional_text(
        response.headers.get("x-request-id"),
        redactor=response.redactor,
        max_bytes=MAX_EXA_REQUEST_ID_BYTES,
    )
    if request_id is not None:
        exa_metadata["request_id"] = request_id
    retry_after = _retry_after_seconds(response.headers.get("retry-after"))
    if retry_after is not None:
        exa_metadata["retry_after_seconds"] = retry_after
    structured: dict[str, Any] = {
        "error": code,
        "status_code": response.status_code,
    }
    if exa_metadata:
        structured["provider_metadata"] = {"exa": exa_metadata}
    return ToolResult(content=message, structured=structured, is_error=True)


def _exa_provider_metadata(
    document: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    redactor: SecretRedactor,
) -> tuple[dict[str, Any], bool]:
    metadata: dict[str, Any] = {}
    truncated = False
    raw_request_id = document.get("requestId", headers.get("x-request-id"))
    if raw_request_id is not None:
        request_id, was_truncated = _provider_text(
            raw_request_id,
            redactor=redactor,
            max_bytes=MAX_EXA_REQUEST_ID_BYTES,
            preserve_line_breaks=False,
            require_nonblank=True,
        )
        metadata["request_id"] = request_id
        truncated = truncated or was_truncated
    raw_warnings = document.get("warnings")
    if raw_warnings is not None:
        if type(raw_warnings) is not list:
            raise _ExaMalformedResponseError
        warnings: list[str] = []
        for warning in raw_warnings[:MAX_EXA_WARNINGS]:
            bounded, was_truncated = _provider_text(
                warning,
                redactor=redactor,
                max_bytes=MAX_EXA_WARNING_BYTES,
                preserve_line_breaks=False,
                require_nonblank=True,
            )
            warnings.append(bounded)
            truncated = truncated or was_truncated
        if len(raw_warnings) > MAX_EXA_WARNINGS:
            truncated = True
        metadata["warnings"] = warnings
    cost = document.get("costDollars")
    if cost is not None:
        metadata["usage"] = _exa_usage(cost)
    return {"exa": metadata}, truncated


def _exa_usage(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise _ExaMalformedResponseError
    usage: dict[str, Any] = {}
    if "total" in value:
        usage["estimated_cost_usd"] = _nonnegative_finite_number(value["total"])
    search = value.get("search")
    if search is not None:
        if type(search) is not dict:
            raise _ExaMalformedResponseError
        if "neural" in search:
            usage["estimated_search_cost_usd"] = _nonnegative_finite_number(search["neural"])
    return usage


def _nonnegative_finite_number(value: Any) -> int | float:
    number = _portable_finite_number(value)
    if number < 0:
        raise _ExaMalformedResponseError
    return number


def _highlight_scores(value: Any, *, count: int) -> list[float] | None:
    if value is None:
        return None
    if type(value) is not list or len(value) != count:
        raise _ExaMalformedResponseError
    scores: list[float] = []
    for score in value:
        scores.append(float(_portable_finite_number(score)))
    return scores


def _portable_finite_number(value: Any) -> int | float:
    if type(value) is int:
        if value < -(2**63) or value > 2**63 - 1:
            raise _ExaMalformedResponseError
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise _ExaMalformedResponseError


def _provider_url(value: Any, *, redactor: SecretRedactor) -> str:
    try:
        canonical = _canonicalize_url(value)
    except (TypeError, ValueError) as exc:
        raise _ExaMalformedResponseError from exc
    if redactor.redact_text(canonical) != canonical:
        raise _ExaMalformedResponseError
    return canonical


def _provider_text(
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
        raise _ExaMalformedResponseError
    normalizer = _BoundedNormalizedText(
        max_bytes,
        preserve_line_breaks=preserve_line_breaks,
    )
    normalizer.feed(redactor.redact_text(value))
    bounded = normalizer.value
    if require_nonblank and not bounded:
        raise _ExaMalformedResponseError
    return bounded, normalizer.truncated


def _bounded_optional_text(
    value: Any,
    *,
    redactor: SecretRedactor,
    max_bytes: int,
) -> str | None:
    if value is None:
        return None
    try:
        bounded, _ = _provider_text(
            value,
            redactor=redactor,
            max_bytes=max_bytes,
            preserve_line_breaks=False,
            require_nonblank=True,
        )
    except _ExaMalformedResponseError:
        return None
    return bounded


def _publication_time(value: Any, *, redactor: SecretRedactor) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or "\x00" in value
        or any(0xD800 <= ord(character) <= 0xDFFF for character in value)
        or len(value.encode("utf-8")) > MAX_EXA_PUBLICATION_TIME_BYTES
    ):
        raise _ExaMalformedResponseError
    if redactor.redact_text(value) != value:
        raise _ExaMalformedResponseError
    if len(value) == 10:
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError as exc:
            raise _ExaMalformedResponseError from exc
        if parsed_date.isoformat() != value:
            raise _ExaMalformedResponseError
        return value
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _ExaMalformedResponseError from exc
    if parsed.tzinfo is None:
        raise _ExaMalformedResponseError
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _json_object(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (
        UnicodeDecodeError,
        ValueError,
        RecursionError,
        _ExaMalformedResponseError,
    ) as exc:
        raise _ExaMalformedResponseError from exc
    if type(value) is not dict:
        raise _ExaMalformedResponseError
    return value


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _ExaMalformedResponseError
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    del value
    raise _ExaMalformedResponseError


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if target is None:
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        return min(
            _MAX_EXA_RETRY_AFTER_SECONDS,
            max(0.0, (target - datetime.now(UTC)).total_seconds()),
        )
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, _MAX_EXA_RETRY_AFTER_SECONDS)


def _exa_origin(value: Any) -> str:
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


def _max_age_hours(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < -1 or value > 720:
        raise ValueError(f"{name} must be None or an integer between -1 and 720.")
    return value


__all__ = [
    "ExaWebAdapter",
]
