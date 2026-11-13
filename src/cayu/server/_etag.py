"""Weak entity tags for JSON read routes, so a poll of an unchanged body costs a 304."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from fastapi import Request, Response


def json_body_etag(body: Mapping[str, Any]) -> str:
    """A weak entity tag for one exact JSON response body."""

    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return f'W/"{hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]}"'


def if_none_match_matches(header: str | None, etag: str) -> bool:
    """Weakly compare an ``If-None-Match`` header with one entity tag."""

    if header is None:
        return False
    opaque = etag.removeprefix("W/")
    for candidate in header.split(","):
        candidate = candidate.strip()
        if candidate == "*" or candidate.removeprefix("W/") == opaque:
            return True
    return False


def tagged_json(
    request: Request,
    response: Response,
    body: Any,
    *,
    tagged: Mapping[str, Any],
    cache_control: str,
) -> Any:
    """``body`` with the weak entity tag of ``tagged``, or a bodiless 304 when the
    request's ``If-None-Match`` matches that tag."""

    headers = {"ETag": json_body_etag(tagged), "Cache-Control": cache_control}
    if if_none_match_matches(request.headers.get("if-none-match"), headers["ETag"]):
        return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    return body
