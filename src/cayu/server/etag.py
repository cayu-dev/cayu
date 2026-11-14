"""Entity tags for JSON read routes, so a read of an unchanged answer costs a ``304``.

A route that a page polls usually answers what it answered last time. Return its answer
through :func:`tagged_json` and a client that sends back the tag it last got
(``If-None-Match``) gets ``304 Not Modified`` with no body instead::

    from fastapi import APIRouter, Request, Response

    from cayu.server.etag import tagged_json

    router = APIRouter()


    @router.get("/app/overview")
    async def get_overview(request: Request, response: Response) -> OverviewView:
        view = await read_overview()
        # `as_of` changes on every read, so it stays out of the tag.
        return tagged_json(
            request, response, view, tagged=view.model_dump(mode="json", exclude={"as_of"})
        )

The default ``Cache-Control: private, no-cache`` lets a browser keep the body and revalidate
it on every read, so a page's plain ``fetch`` sends ``If-None-Match`` and gets the kept body
back from a ``304`` without code of its own. With ``private, no-store`` the browser keeps
nothing, and only a client that sends ``If-None-Match`` itself (like ``client.js``'s
watches) gets the ``304``.

Leave out of the tag anything that changes on every read (a timestamp of the read, a
freshly sealed cursor), or the tag never matches. A client that gets a ``304`` keeps the
body it has, with that field's old value.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import Request, Response
from pydantic import BaseModel

__all__ = [
    "if_none_match_matches",
    "json_body_etag",
    "tagged_json",
]

# Response headers a bodiless 304 doesn't repeat from the route's own response.
_BODY_HEADERS = frozenset({b"content-length", b"content-type"})


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return str(value)


def json_body_etag(body: Any) -> str:
    """A weak entity tag for one exact JSON answer.

    ``body`` is any JSON value; Pydantic models in it are tagged as they serialize to
    JSON. Object keys are sorted first, so their order doesn't change the tag.
    """

    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=_json_default)
    return f'W/"{hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]}"'


def if_none_match_matches(header: str | None, etag: str) -> bool:
    """Whether an ``If-None-Match`` header names ``etag``, compared weakly, or is ``*``."""

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
    tagged: Any = None,
    cache_control: str = "private, no-cache",
) -> Any:
    """``body`` with the weak entity tag of ``tagged`` (by default ``body`` itself), or a
    bodiless ``304`` when the request's ``If-None-Match`` names that tag.

    ``response`` is the route's own FastAPI ``Response``: the tag and ``cache_control`` are
    set on it, and the ``304`` keeps the other headers the route or its dependencies set
    there (a cookie, ``Vary``). Return the result from the route: FastAPI serializes
    ``body`` as usual and sends the ``304`` as it is.
    """

    headers = {
        "ETag": json_body_etag(body if tagged is None else tagged),
        "Cache-Control": cache_control,
    }
    if not if_none_match_matches(request.headers.get("if-none-match"), headers["ETag"]):
        response.headers.update(headers)
        return body
    not_modified = Response(status_code=304)
    for name, value in response.headers.raw:
        if name not in _BODY_HEADERS:
            not_modified.raw_headers.append((name, value))
    not_modified.headers.update(headers)
    return not_modified
