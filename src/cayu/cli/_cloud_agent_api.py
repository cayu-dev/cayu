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

import argparse
import base64
import ipaddress
import math
import re
import sys
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
# A refusal code worth repeating: a short identifier, like `role_forbidden`.
_REFUSAL_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
# The answers a Cayu route gives for success: 200 for a read, 201 for a created resource (an
# imported eval corpus) and 202 for accepted asynchronous work (an eval run admitted or
# cancelled).
_SUCCESS_STATUSES = frozenset({200, 201, 202})
# Agent states in which its web process can't answer; Cloud wakes a sleeping one.
_NOT_SERVING_STATUSES = frozenset({"stopping", "stopped", "deleting", "archived", "failed"})
#: How long a read waits for a sleeping Cloud Agent it woke (``--wake-timeout-seconds``).
DEFAULT_WAKE_TIMEOUT_SECONDS = 180.0
# Failures of one read that say nothing about the next: the Agent was waking, or the request
# timed out or could not connect.
_TRANSIENT_CATEGORIES = frozenset({"starting", "timeout", "unreachable"})


class CayuApiError(RuntimeError):
    """A Cayu API read that could not be answered, with a stable ``category``.

    ``detail`` is the route's own bounded reason for refusing the request, when it gave one,
    for callers that tell one refusal of a route from another, and ``code`` the machine code
    of a structured refusal (such as ``role_forbidden``).
    """

    def __init__(
        self,
        category: str,
        message: str,
        *,
        status_code: int | None = None,
        detail: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.detail = detail
        self.code = code

    @property
    def transient(self) -> bool:
        """Whether the same read may succeed later: the server was starting, slow, out of
        reach, or failed with a 5xx. A command that waits on the server reads again."""

        if self.category in _TRANSIENT_CATEGORIES:
            return True
        return self.category == "server_error" and (self.status_code or 0) >= 500


class CayuApiNotFound(CayuApiError):
    """The route answered that the requested resource does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__("not_found", message, status_code=404, detail=message)


class CayuApi:
    """JSON client for one Cayu server's reads and explicit one-shot mutations.

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
        wake_timeout_seconds: float = DEFAULT_WAKE_TIMEOUT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        notice: Callable[[str], None] | None = None,
        offers_wake_option: bool = False,
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
        self._credential_parts = _authorization_secrets(authorization)
        self.description = description
        self._http = http
        self._owns_http = http is None
        self._timeout = httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds))
        self._wake_timeout_seconds = wake_timeout_seconds
        # Whether the command reading through this API takes --wake-timeout-seconds, so an
        # Agent that is still starting can point at it.
        self._offers_wake_option = offers_wake_option
        self._notice = _stderr_notice if notice is None else notice
        self._told_waking = False
        self._sleep = sleep
        self._monotonic = monotonic

    @property
    def origin(self) -> str:
        """The server this API is on (scheme and host), as resolved for a Cloud Agent."""

        return self._origin

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
        """POST a JSON request that is safe to send again and return its answer.

        That is a read (such as a session rollup page) or a write the server deduplicates
        by its content (such as an eval corpus import). Errors are reported as for ``get``;
        a server that predates the route answers ``CayuApiError("unsupported")``.
        """

        return self._request("POST", path, query, body, resource)

    def mutate(
        self,
        path: str,
        body: Mapping[str, Any],
        *,
        resource: str,
        operation_id: str | None = None,
    ) -> Any:
        """Dispatch once to a discovered API; never retry a possibly committed write.

        Discovery is a read. A timeout, lost response or proxy error after POST is
        ambiguous; callers retain their original operation identity for explicit retry.
        """
        if self._base is None:
            self.get("/app", resource="agent")
        assert self._base is not None
        headers = dict(self._headers)
        if operation_id is not None:
            headers["Idempotency-Key"] = operation_id
        try:
            response = self._bounded_request(
                "POST",
                self._origin + self._base + path,
                None,
                body,
                resource,
                headers=headers,
            )
        except (httpx.HTTPError, CayuApiError) as exc:
            raise CayuApiError(
                "outcome_unknown",
                "The server did not confirm the mutation. It may have completed; inspect "
                "the result before retrying, using the same operation ID and request.",
            ) from exc
        if response.status_code >= 500:
            raise CayuApiError(
                "outcome_unknown",
                "The server did not confirm the mutation. It may have completed; inspect "
                "the result before retrying, using the same operation ID and request.",
                status_code=response.status_code,
            )
        try:
            return self._body(response, resource)
        except CayuApiError as exc:
            if response.status_code in _SUCCESS_STATUSES:
                raise CayuApiError(
                    "outcome_unknown",
                    "The mutation returned no readable receipt; inspect its result before retrying.",
                ) from exc
            raise

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
                    raise CayuApiError("starting", self._starting_message(), status_code=503)
                if not self._told_waking:
                    self._told_waking = True
                    self._notice(
                        f"{self.description} is waking up; waiting up to "
                        f"{_duration(self._wake_timeout_seconds)}."
                    )
                self._sleep(_retry_after_seconds(response.headers["retry-after"]))
                continue
            return self._body(response, resource)

    def _starting_message(self) -> str:
        if self._wake_timeout_seconds <= 0:
            message = f"{self.description} is starting and could not answer yet; try again"
            hint = ", or wait for it with --wake-timeout-seconds."
        else:
            message = (
                f"{self.description} is starting and didn't answer within "
                f"{self._wake_timeout_seconds:g} seconds; try again"
            )
            hint = ", or wait longer with --wake-timeout-seconds."
        return message + (hint if self._offers_wake_option else ".")

    def _bounded_request(
        self,
        method: str,
        url: str,
        query: Mapping[str, str | Sequence[str]] | None,
        body: Mapping[str, Any] | None,
        resource: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """Send one request, reading at most ``MAX_RESPONSE_BYTES`` of its (decoded) answer."""

        with self._client().stream(
            method,
            url,
            params=None if query is None else dict(query),
            json=None if body is None else dict(body),
            headers=self._headers if headers is None else headers,
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

    def ensure_safe_output(self, value: Any) -> None:
        """Refuse publication of the active credential, including JSON object keys."""
        if not self._credential_parts:
            return
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                if any(secret in item for secret in self._credential_parts):
                    raise CayuApiError(
                        "unsafe_response", "The server returned credential-bearing content."
                    )
            elif isinstance(item, dict):
                pending.extend(item.keys())
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)

    def _body(self, response: httpx.Response, resource: str) -> Any:
        established = self._base is not None
        status = response.status_code
        # Inspect decoded JSON before error details or successful bodies reach consumers.
        # Field-name heuristics cannot recognize opaque tokens or Basic credentials.
        try:
            body = response.json()
        except ValueError:
            body, valid_json = None, False
            self.ensure_safe_output(response.text)
        else:
            valid_json = True
            self.ensure_safe_output(body)
        detail = _detail(body)
        if status == 401:
            raise CayuApiError(
                "credentials_rejected",
                f"{self.description} did not accept the credentials for reading {resource}.",
                status_code=status,
            )
        if status == 403:
            # The credentials were accepted, but this caller may not do this: the role, an
            # access scope or another rule refused it. Keep the route's own reason.
            code = _code(body)
            raise CayuApiError(
                "forbidden",
                f"{self.description} refused the {resource} request"
                + (f": {detail}" if detail else ".")
                + (f" ({code})" if code else ""),
                status_code=status,
                detail=detail,
                code=code,
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
                detail=detail,
            )
        if status not in _SUCCESS_STATUSES:
            raise CayuApiError(
                "server_error",
                f"{self.description} answered HTTP {status} for {resource}.",
                status_code=status,
            )
        if not valid_json:
            if not established:
                raise _RouteMissing
            raise CayuApiError("not_cayu", f"{self.description} did not answer {resource} as Cayu.")
        return body


class _RouteMissing(Exception):
    """No route served the path at this API base."""

    def __init__(self, status_code: int = 404) -> None:
        super().__init__(status_code)
        self.status_code = status_code


def _authorization_secrets(authorization: str | None) -> tuple[str, ...]:
    if not authorization:
        return ()
    secrets = {authorization}
    parts = authorization.split(maxsplit=1)
    if len(parts) == 2:
        scheme, payload = parts[0], parts[1].strip()
        secrets.add(payload)
        if scheme.lower() == "basic":
            try:
                decoded = base64.b64decode(payload, validate=True).decode("utf-8")
            except (ValueError, UnicodeError):
                pass
            else:
                secrets.add(decoded)
                _, separator, password = decoded.partition(":")
                if separator and password:
                    secrets.add(password)
    return tuple(secret for secret in secrets if secret)


def basic_authorization(username: str, password: str) -> str:
    """The ``Authorization`` value of a Cayu operator login."""

    credentials = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    return f"Basic {credentials}"


def open_cloud_agent_api(
    application: str,
    *,
    context: Path | None = None,
    timeout_seconds: float = 30.0,
    wake_timeout_seconds: float = DEFAULT_WAKE_TIMEOUT_SECONDS,
    offers_wake_option: bool = False,
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
    return cloud_agent_api(
        client,
        application_id,
        timeout_seconds=timeout_seconds,
        wake_timeout_seconds=wake_timeout_seconds,
        offers_wake_option=offers_wake_option,
        http=http,
    )


def cloud_agent_api(
    client: CloudApiClient,
    application_id: str,
    *,
    timeout_seconds: float = 30.0,
    wake_timeout_seconds: float = DEFAULT_WAKE_TIMEOUT_SECONDS,
    offers_wake_option: bool = False,
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
        offers_wake_option=offers_wake_option,
    )


def _code(body: Any) -> str | None:
    """The machine code of a structured refusal (``{"detail": {"code": ...}}``), if plain."""

    detail = body.get("detail") if isinstance(body, dict) else None
    code = detail.get("code") if isinstance(detail, dict) else None
    if not isinstance(code, str) or not _REFUSAL_CODE.fullmatch(code):
        return None
    # Screened like the reason: an identifier can still look like a token.
    return safe_text(code, 64)


def _detail(body: Any) -> str | None:
    """The server's own bounded ``detail``, never a request value it echoes."""

    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        # A structured refusal, such as `{"code": "session_not_found", "message": ...}`.
        detail = detail.get("message")
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


def add_wake_options(parser: argparse.ArgumentParser) -> None:
    """``--wake-timeout-seconds`` and ``--no-wait``, for a command that reads a Cloud Agent."""

    wait = parser.add_mutually_exclusive_group()
    wait.add_argument(
        "--wake-timeout-seconds",
        type=_wake_seconds,
        default=DEFAULT_WAKE_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            "With --application: how long to wait for a sleeping Agent this command wakes "
            f"(default: {DEFAULT_WAKE_TIMEOUT_SECONDS:g})."
        ),
    )
    wait.add_argument(
        "--no-wait",
        dest="wake_timeout_seconds",
        action="store_const",
        const=0.0,
        help="With --application: fail at once if the Agent is asleep (its wake still starts).",
    )


def _wake_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number of seconds") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("must be zero or more seconds")
    return seconds


def _duration(seconds: float) -> str:
    if seconds >= 60 and seconds % 60 == 0:
        minutes = int(seconds // 60)
        return f"{minutes} minute" + ("" if minutes == 1 else "s")
    return f"{seconds:g} seconds"


def _stderr_notice(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


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
