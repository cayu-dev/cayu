"""Read a Cayu server's own API, and find a Cayu Cloud Agent's with the CLI's Cloud login.

Read commands that inspect durable state (`cayu task`) open a local store, or, with
`--application NAME`, read the Agent on Cayu Cloud. For the Agent they read the same
authenticated API its Developer dashboard uses, at the Agent's own address: the CLI signs
in to Cayu Cloud as `cayu cloud` does (`cayu cloud login`, `--context` or
`CAYU_CLOUD_API_KEY`), asks Cloud for the Agent's Cayu address (`cayu cloud service
status`) and its Developer dashboard login (`cayu cloud service credentials`), and sends
that login to the Agent. Nothing is written locally.

`CayuApi` is the part that knows nothing about Cayu Cloud: a Cayu server's API under a
mount URL, with an optional `Authorization` value. `open_cloud_agent_api` is how Cayu
Cloud supplies both.
"""

from __future__ import annotations

import base64
import ipaddress
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit

import httpx

from cayu.cli._cloud_diagnostics import safe_text

if TYPE_CHECKING:
    from cayu.cli._cloud_api import CloudApiClient

# FastAPI's answer for a path no route serves, as opposed to a route's own "not found".
_ROUTE_NOT_FOUND = "Not Found"
# The most bytes one answer may carry; a full page of session events with payloads fits.
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_MAX_DETAIL_CHARS = 512
# Agent states in which its web process can't answer; Cloud wakes a sleeping one.
_NOT_SERVING_STATUSES = frozenset({"stopping", "stopped", "deleting", "archived", "failed"})


class CayuApiError(RuntimeError):
    """A Cayu API read that could not be answered, with a stable ``category``."""

    def __init__(self, category: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code


class CayuApiNotFound(CayuApiError):
    """The route answered that the requested resource does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__("not_found", message, status_code=404)


class CayuApi:
    """Read-only JSON client for one Cayu server's API.

    ``url`` is where the server mounts Cayu: an application that embeds it with
    ``mount_cayu(..., path="/cayu")`` serves the API at ``/cayu/api/``, and ``cayu serve``
    at ``/api/`` on the same origin. The first read tries the mount's ``api/`` and falls
    back to ``/api/`` only when no route answers there, as Cayu Cloud does.
    """

    def __init__(
        self,
        url: str,
        *,
        authorization: str | None = None,
        description: str = "The Cayu server",
        http: httpx.Client | None = None,
        timeout_seconds: float = 30.0,
        wake_timeout_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise CayuApiError("invalid_url", f"{description}'s Cayu URL is not an HTTP(S) URL.")
        if parts.scheme == "http" and authorization is not None and not _is_loopback(parts):
            raise CayuApiError(
                "invalid_url", f"{description}'s Cayu URL must use HTTPS to send credentials."
            )
        self._origin = f"{parts.scheme}://{parts.netloc}"
        self._candidates = tuple(dict.fromkeys((parts.path.rstrip("/") + "/api", "/api")))
        self._base: str | None = None
        self._headers = {"Accept": "application/json"}
        if authorization is not None:
            self._headers["Authorization"] = authorization
        self.description = description
        self._http = http
        self._owns_http = http is None
        self._timeout = httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds))
        self._wake_timeout_seconds = wake_timeout_seconds
        self._sleep = sleep
        self._monotonic = monotonic

    def get(
        self,
        path: str,
        *,
        query: Mapping[str, str | Sequence[str]] | None = None,
        resource: str,
    ) -> Any:
        """GET ``path`` under the API and return its JSON body.

        ``resource`` names what is read in messages, for example ``"tasks"``. A route the
        server doesn't serve raises ``CayuApiError("unsupported")``; a route's own 404
        raises ``CayuApiNotFound``.
        """

        return self._request("GET", path, query, None, resource)

    def post(
        self,
        path: str,
        body: Mapping[str, Any],
        *,
        query: Mapping[str, str | Sequence[str]] | None = None,
        resource: str,
    ) -> Any:
        """POST a JSON read request (such as a session rollup page) and return its answer.

        Errors are reported as for ``get``; a server that predates the route answers
        ``CayuApiError("unsupported")``.
        """

        return self._request("POST", path, query, body, resource)

    def _request(
        self,
        method: str,
        path: str,
        query: Mapping[str, str | Sequence[str]] | None,
        body: Mapping[str, Any] | None,
        resource: str,
    ) -> Any:
        if self._base is not None:
            return self._read(method, self._base, path, query, body, resource)
        missing: list[str] = []
        method_refused = False
        for base in self._candidates:
            try:
                answer = self._read(method, base, path, query, body, resource)
            except _RouteMissing as exc:
                missing.append(f"{base}/")
                method_refused = method_refused or exc.status_code == 405
                continue
            self._base = base
            return answer
        if method_refused:
            # A route matched the path but not the method: an older Cayu without this route.
            raise CayuApiError(
                "unsupported",
                f"{self.description} can't serve {resource}. Its Cayu may be older than this CLI.",
                status_code=405,
            )
        raise CayuApiError(
            "not_served",
            f"{self.description} does not serve Cayu's API at {' or '.join(missing)}.",
            status_code=404,
        )

    def close(self) -> None:
        if self._http is not None and self._owns_http:
            self._http.close()
            self._http = None

    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self._timeout, follow_redirects=False)
        return self._http

    def _read(
        self,
        method: str,
        base: str,
        path: str,
        query: Mapping[str, str | Sequence[str]] | None,
        body: Mapping[str, Any] | None,
        resource: str,
    ) -> Any:
        deadline = self._monotonic() + self._wake_timeout_seconds
        while True:
            try:
                response = self._bounded_request(
                    method, self._origin + base + path, query, body, resource
                )
            except httpx.TimeoutException as exc:
                raise CayuApiError(
                    "timeout", f"{self.description} did not answer in time."
                ) from exc
            except httpx.HTTPError as exc:
                raise CayuApiError(
                    "unreachable", f"Could not connect to {self.description.lower()}."
                ) from exc
            if response.status_code == 503 and "retry-after" in response.headers:
                # Cayu Cloud's router is waking a sleeping Agent; this request did the waking.
                if self._monotonic() >= deadline:
                    raise CayuApiError(
                        "starting",
                        f"{self.description} is starting and could not answer yet; try again.",
                        status_code=503,
                    )
                self._sleep(_retry_after_seconds(response.headers["retry-after"]))
                continue
            return self._body(response, resource)

    def _bounded_request(
        self,
        method: str,
        url: str,
        query: Mapping[str, str | Sequence[str]] | None,
        body: Mapping[str, Any] | None,
        resource: str,
    ) -> httpx.Response:
        """Send one request, reading at most ``MAX_RESPONSE_BYTES`` of its (decoded) answer."""

        with self._client().stream(
            method,
            url,
            params=None if query is None else dict(query),
            json=None if body is None else dict(body),
            headers=self._headers,
        ) as response:
            received = bytearray()
            for chunk in response.iter_bytes():
                received.extend(chunk)
                if len(received) > MAX_RESPONSE_BYTES:
                    raise CayuApiError(
                        "too_large",
                        f"{self.description}'s answer for {resource} is larger than the CLI "
                        "reads at once; narrow the request.",
                        status_code=response.status_code,
                    )
        # `iter_bytes` already decoded any Content-Encoding, so the rebuilt answer must not
        # claim to be encoded.
        return httpx.Response(
            response.status_code,
            headers=[
                (key, value)
                for key, value in response.headers.multi_items()
                if key.lower() not in {"content-encoding", "content-length"}
            ],
            content=bytes(received),
        )

    def _body(self, response: httpx.Response, resource: str) -> Any:
        established = self._base is not None
        status = response.status_code
        detail = _detail(response)
        if status in {401, 403}:
            raise CayuApiError(
                "credentials_rejected",
                f"{self.description} did not accept the credentials for reading {resource}.",
                status_code=status,
            )
        if status == 404:
            if detail is None or detail == _ROUTE_NOT_FOUND:
                if not established:
                    raise _RouteMissing
                raise CayuApiError(
                    "unsupported",
                    f"{self.description} runs a Cayu that doesn't serve {resource} yet; "
                    "deploy it with a newer Cayu.",
                    status_code=404,
                )
            raise CayuApiNotFound(detail)
        if status == 405 and not established:
            # Another app's route at this base (such as a page served for any GET), or an
            # older Cayu's; the next base, or the caller, tells which.
            raise _RouteMissing(405)
        if status in {405, 501}:
            raise CayuApiError(
                "unsupported",
                f"{self.description} can't serve {resource}"
                + (f": {detail}" if detail else ".")
                + ("" if detail else " Its Cayu may be older than this CLI."),
                status_code=status,
            )
        if status in {400, 409, 413, 422}:
            raise CayuApiError(
                "request_rejected",
                f"{self.description} rejected the {resource} request"
                + (f": {detail}" if detail else f" (HTTP {status})."),
                status_code=status,
            )
        if status != 200:
            raise CayuApiError(
                "server_error",
                f"{self.description} answered HTTP {status} for {resource}.",
                status_code=status,
            )
        try:
            return response.json()
        except ValueError:
            if not established:
                raise _RouteMissing from None
            raise CayuApiError(
                "not_cayu", f"{self.description} did not answer {resource} as Cayu."
            ) from None


class _RouteMissing(Exception):
    """No route served the path at this API base."""

    def __init__(self, status_code: int = 404) -> None:
        super().__init__(status_code)
        self.status_code = status_code


def basic_authorization(username: str, password: str) -> str:
    """The ``Authorization`` value of a Cayu operator login."""

    credentials = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    return f"Basic {credentials}"


def open_cloud_agent_api(
    application: str,
    *,
    context: Path | None = None,
    timeout_seconds: float = 30.0,
    http: httpx.Client | None = None,
) -> CayuApi:
    """The Cayu API of the Cayu Cloud Agent ``application`` (its slug or name).

    Signs in to Cayu Cloud the way `cayu cloud` does, with ``context`` standing for its
    ``--context``, then reads it as `cloud_agent_api` does.
    """

    from cayu.cli._cloud_api import CloudApiError
    from cayu.cli._cloud_auth import CloudAuthError
    from cayu.cli.cloud import CloudCommandError, cloud_client, resolve_application_id

    try:
        client = cloud_client(context=context, timeout_seconds=timeout_seconds)
        application_id = resolve_application_id(client, application)
    except (CloudApiError, CloudAuthError, CloudCommandError) as exc:
        raise CayuApiError(exc.category, str(exc)) from None
    return cloud_agent_api(client, application_id, timeout_seconds=timeout_seconds, http=http)


def cloud_agent_api(
    client: CloudApiClient,
    application_id: str,
    *,
    timeout_seconds: float = 30.0,
    wake_timeout_seconds: float = 60.0,
    http: httpx.Client | None = None,
) -> CayuApi:
    """The Cayu API of the Agent ``application_id``, found with a signed-in ``client``.

    Raises ``CayuApiError`` with Cloud's category (and HTTP status) when Cloud refuses,
    ``"not_serving"`` when the Agent's service is stopped or failing, and ``"no_address"``
    when it serves no Cayu address (an Agent without a web process).
    """

    from cayu.cli._cloud_api import CloudApiError
    from cayu.cli._cloud_auth import CloudAuthError
    from cayu.cli.cloud import CloudCommandError, operator_credentials

    try:
        service = client.request("GET", f"/v1/applications/{quote(application_id)}/service")
        status = service.get("status")
        cayu_url = service.get("cayu_url")
        if status in _NOT_SERVING_STATUSES:
            raise CayuApiError(
                "not_serving",
                f"The Agent {application_id}'s service is {status}, so it can't be read. "
                "`cayu cloud service status` shows why.",
            )
        if not isinstance(cayu_url, str) or not cayu_url:
            # For example a worker-only Agent, whose worker may still be running tasks.
            raise CayuApiError(
                "no_address",
                f"The Agent {application_id} serves no Cayu address, so its Cayu API can't be "
                "read; an Agent without a [web] process has none.",
            )
        login = operator_credentials(client, application_id=application_id)
        authorization = basic_authorization(login["username"], login["password"])
    except CloudApiError as exc:
        raise CayuApiError(exc.category, str(exc), status_code=exc.status_code) from None
    except (CloudAuthError, CloudCommandError) as exc:
        raise CayuApiError(exc.category, str(exc)) from None
    return CayuApi(
        cayu_url,
        authorization=authorization,
        description=f"The Agent {application_id}",
        http=http,
        timeout_seconds=timeout_seconds,
        wake_timeout_seconds=wake_timeout_seconds,
    )


def _detail(response: httpx.Response) -> str | None:
    """The server's own bounded ``detail``, never a request value it echoes."""

    try:
        body = response.json()
    except ValueError:
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list):
        # FastAPI request validation: name each rejected parameter and its rule.
        parts = []
        for item in detail[:10]:
            if not isinstance(item, dict):
                continue
            location = item.get("loc")
            name = (
                ".".join(str(part) for part in location[1:] or location)
                if isinstance(location, list) and location
                else None
            )
            message = item.get("msg")
            if isinstance(message, str):
                parts.append(f"{name}: {message}" if name else message)
        detail = "; ".join(parts) or None
    if not isinstance(detail, str) or not detail.strip():
        return None
    detail = " ".join(detail.split())
    # Never repeat a detail that looks like it carries a credential or control characters.
    if safe_text(detail, 64 * 1024) is None:
        return None
    return detail if len(detail) <= _MAX_DETAIL_CHARS else detail[: _MAX_DETAIL_CHARS - 1] + "…"


def _retry_after_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        return 3.0
    return min(max(seconds, 1.0), 5.0)


def _is_loopback(parts: Any) -> bool:
    hostname = parts.hostname
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname or "").is_loopback
    except ValueError:
        return False
