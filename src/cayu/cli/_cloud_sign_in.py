"""Finishing a connection's sign-in from the command line.

`connections add` (for a connector that signs in) and `connections authorize` listen on a
loopback port before they start the sign-in, and name it to Cayu Cloud as `return_port`. Cloud
answers a state that ends in `.<port>`; the connector sends the person back to the portal, whose
server sends the browser on to `http://127.0.0.1:<port>/cayu-cloud/sign-in` with the state and
the code (or the connector's error). The command takes only the state it started, completes the
sign-in as the admin who ran it, and tells the browser and the terminal how it went, sending
the browser on to a page whose address holds neither.

The command opens (and prints) only the listener's own `/cayu-cloud/sign-in/start?k=<secret>`;
a visit without the secret is refused and changes nothing. Its first visit is sent on to the
approval page with a one-time cookie, and only a callback carrying that cookie is taken, so the
state leaves the command only to the browser that opened the link. A second visit means someone
else has the link: before a sign-in came back the command stops and connects nothing; after,
it fails loudly, as the connection may be theirs. The code
and the state are never printed or logged. A command that stops waiting (its time is up, or
Ctrl-C) leaves the connection `pending`; `connections authorize` starts again.
"""

from __future__ import annotations

import contextlib
import hmac
import secrets
import socket
import threading
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import monotonic
from typing import Any
from urllib.parse import parse_qs, urlsplit

from cayu.cli._cloud_api import CloudApiError

#: Where the portal sends the browser on to, on the command's loopback port.
SIGN_IN_RETURN_PATH = "/cayu-cloud/sign-in"
#: Where the command opens the browser: on to the connector's approval page, once.
SIGN_IN_START_PATH = f"{SIGN_IN_RETURN_PATH}/start"
#: Where the browser goes on to once the sign-in is done: the address without the code.
SIGN_IN_DONE_PATH = f"{SIGN_IN_RETURN_PATH}/done"
#: How long a command waits by default: as long as a sign-in's state lasts.
DEFAULT_SIGN_IN_WAIT_SECONDS = 600.0
#: The longest one connection may take to send its request.
REQUEST_SECONDS = 10.0
#: How long a command waits for the browser to come back for the done page.
DONE_PAGE_SECONDS = 5.0
#: What a second opener of the start page is told, and the command says.
LINK_REUSED = (
    "Another browser opened this sign-in's link, so the command stopped. Nothing was connected. "
    "Run it again."
)
#: What a second opener of the start page is told once a sign-in came back from the first.
LINK_SHARED_PAGE = (
    "Another browser already opened this sign-in's link and came back with it. See the "
    "terminal that started it."
)
#: The most connections served at once.
MAX_CONNECTIONS = 16
#: How often a waiting command checks its time and Ctrl-C.
_TICK = 0.1

#: Completes the sign-in with what came back: the connection, or raises Cloud's refusal.
Finish = Callable[[dict[str, str]], dict[str, Any]]


class SignInReturn(ThreadingHTTPServer):
    """The loopback listener one sign-in returns to. Bound before the sign-in starts, so its
    port can be named to Cloud (``port`` 0: any free one); it answers only the state it is
    told to wait for, once.

    Each connection is served on its own daemon thread, up to `MAX_CONNECTIONS` at once (any
    more are closed at once), and must send its request within `REQUEST_SECONDS` (and before
    the wait ends), so an idle or slow one never delays the approval or holds the command past
    its wait."""

    daemon_threads = True
    block_on_close = False

    def __init__(self, port: int = 0) -> None:
        super().__init__(("127.0.0.1", port), _ReturnHandler)
        self.port: int = self.server_address[1]
        self.state: str | None = None
        self.finish: Finish | None = None
        self.outcome: dict[str, Any] | None = None
        self.failure: Exception | None = None
        self.deadline = monotonic()
        #: Taken to claim the sign-in: only the first callback with its state completes it.
        self.lock = threading.Lock()
        self.claimed = False
        #: What the browser is shown once the sign-in is done, at `SIGN_IN_DONE_PATH`.
        self.done: str | None = None
        self.finished = threading.Event()
        self.shown = threading.Event()
        #: Only the browser that completed the sign-in is sent to the done page with this.
        self.done_token = secrets.token_urlsafe(16)
        #: Where `SIGN_IN_START_PATH` sends the browser on to, once.
        self.authorization_url: str | None = None
        self.started = False
        #: Set by the start page on the browser it sent on; only that browser's callback counts.
        self.cookie_name = f"cayu_sign_in_{self.port}"
        self.cookie = secrets.token_urlsafe(32)
        #: Another browser asked for the start page after the first: the command stops.
        self.reused = False
        #: The start page's secret, shown only in the command's own link.
        self.link_key = secrets.token_urlsafe(32)
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Quiet: a connection that failed (or whose thread couldn't start) is closed, and
        the wait goes on."""

    def opening(self, authorization_url: str) -> str:
        """The address a browser is opened at: it goes on to ``authorization_url`` the first
        time only. It carries no state, only a secret of its own, so finding the port isn't
        enough to start the sign-in."""

        self.authorization_url = authorization_url
        return f"http://127.0.0.1:{self.port}{SIGN_IN_START_PATH}?k={self.link_key}"

    def wait(self, *, state: str, finish: Finish, seconds: float) -> dict[str, Any]:
        """The connection once the sign-in with ``state`` came back and ``finish`` completed
        it; raises `SignInTimedOut` after ``seconds``, or what ``finish`` raised. A sign-in
        being completed when the time is up is waited for."""

        self.state, self.finish = state, finish
        self.deadline = monotonic() + seconds
        serving = threading.Thread(target=self.serve_forever, args=(0.05,), daemon=True)
        serving.start()
        try:
            while not self.finished.wait(_TICK):
                with self.lock:
                    reused = self.reused and not self.claimed
                    late = not self.claimed and monotonic() >= self.deadline
                if reused:
                    raise SignInLinkReused
                if late:
                    raise SignInTimedOut
            # The browser comes back for the page that replaced the address with the code in it.
            grace = monotonic() + DONE_PAGE_SECONDS
            while not self.shown.wait(_TICK) and not self.reused and monotonic() < grace:
                pass
        except KeyboardInterrupt:
            # Stopped while the browser was coming back for the done page: the sign-in is
            # already done, and its outcome stands.
            if not self.finished.is_set():
                if self.claimed:
                    raise SignInMayHaveCompleted from None
                raise
        finally:
            self.shutdown()
        if self.failure is not None:
            raise self.failure
        if self.outcome is None:
            raise SignInTimedOut
        if self.reused:
            # Completed, but another browser had the link too: it may have been theirs.
            raise SignInLinkShared(self.outcome)
        return self.outcome


class SignInTimedOut(Exception):
    """No sign-in came back in time."""


class SignInLinkReused(Exception):
    """A second browser opened the start page: the sign-in's state may be in other hands."""


class SignInLinkShared(Exception):
    """A sign-in completed, but a second browser also opened its start page: the completed one
    may have been someone else's."""

    def __init__(self, connection: dict[str, Any]) -> None:
        super().__init__("Another browser also opened this sign-in's link.")
        self.connection = connection


class SignInMayHaveCompleted(Exception):
    """Stopped (Ctrl-C) while a sign-in that came back was being completed."""


class _ReturnHandler(BaseHTTPRequestHandler):
    server: SignInReturn

    def setup(self) -> None:
        super().setup()
        # However the request comes (nothing, or a byte at a time), its connection is cut once
        # this passes, until the request has been read.
        seconds = max(0.1, min(REQUEST_SECONDS, self.server.deadline - monotonic()))
        self.connection.settimeout(seconds)
        self.cutoff = threading.Timer(seconds, self._cut)
        self.cutoff.daemon = True
        self.cutoff.start()

    def _cut(self) -> None:
        with contextlib.suppress(OSError):
            self.connection.shutdown(socket.SHUT_RDWR)

    def finish(self) -> None:
        self.cutoff.cancel()
        super().finish()

    def do_GET(self) -> None:
        self.cutoff.cancel()
        listener = self.server
        url = urlsplit(self.path)
        query = {name: values[0] for name, values in parse_qs(url.query).items()}
        if url.path == SIGN_IN_START_PATH:
            if hmac.compare_digest(query.get("k", "").encode(), listener.link_key.encode()):
                self._start()
            else:
                # A guess at the start page: it neither opens the sign-in nor stops it.
                self._answer(HTTPStatus.NOT_FOUND, "Not this sign-in's link.")
            return
        browser = self._browser_cookie()
        if (
            url.path == SIGN_IN_DONE_PATH
            and listener.done is not None
            and browser
            and hmac.compare_digest(query.get("t", "").encode(), listener.done_token.encode())
        ):
            self._answer(HTTPStatus.OK, listener.done)
            listener.shown.set()
            return
        state = query.get("state", "")
        with listener.lock:
            ours = not (
                url.path != SIGN_IN_RETURN_PATH
                or listener.state is None
                or listener.finish is None
                or listener.claimed
                or listener.reused
                or not browser
                or monotonic() >= listener.deadline
                or not hmac.compare_digest(state.encode(), listener.state.encode())
                or ("code" in query) == ("error" in query)
            )
            listener.claimed = listener.claimed or ours
        if not ours or listener.finish is None:
            self._answer(
                HTTPStatus.BAD_REQUEST,
                "This isn't the sign-in this command is waiting for. Nothing was connected.",
            )
            return
        returned = {name: query[name] for name in ("state", "code", "error") if name in query}
        done = "Couldn't connect. See your terminal for more."
        try:
            outcome = listener.finish(returned)
            done = f"{connected(outcome)} You can close this tab."
            listener.outcome = outcome
        except Exception as exc:
            listener.failure = exc
            if isinstance(exc, CloudApiError) and exc.code:
                done = f"Couldn't connect ({exc.code}). See your terminal for more."
        finally:
            # Whatever happened, the command stops waiting.
            listener.done = done
            listener.finished.set()
        # On to a page without the code and the state, so the address and the history keep
        # neither.
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", f"{SIGN_IN_DONE_PATH}?t={listener.done_token}")
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _browser_cookie(self) -> bool:
        """Whether this request comes from the browser the start page sent on."""

        listener = self.server
        jar = SimpleCookie()
        with contextlib.suppress(CookieError):
            jar.load(self.headers.get("Cookie", ""))
        given = jar.get(listener.cookie_name)
        return given is not None and hmac.compare_digest(
            given.value.encode(), listener.cookie.encode()
        )

    def _start(self) -> None:
        listener = self.server
        with listener.lock:
            first = not listener.started and listener.authorization_url is not None
            claimed = listener.claimed
            if listener.started:
                listener.reused = True
            listener.started = True
        if not first or listener.authorization_url is None:
            self._answer(HTTPStatus.GONE, LINK_SHARED_PAGE if claimed else LINK_REUSED)
            return
        # Only the browser that goes on from here may come back with the sign-in: it carries
        # this cookie back on the portal's redirect, a top-level navigation (`SameSite=Lax`).
        seconds = max(1, int(listener.deadline - monotonic()) + 1)
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header(
            "Set-Cookie",
            f"{listener.cookie_name}={listener.cookie}; Path={SIGN_IN_RETURN_PATH}; "
            f"Max-Age={seconds}; HttpOnly; SameSite=Lax",
        )
        self.send_header("Location", listener.authorization_url)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()

    def _answer(self, status: HTTPStatus, text: str) -> None:
        body = f"{text}\n".encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Silent: a request line holds the state and the code."""


def returning_state(authorization_url: str, port: int) -> str | None:
    """The state of a sign-in Cloud started to return to ``port``; None when Cloud didn't
    (one from before command-line sign-ins)."""

    states = parse_qs(urlsplit(authorization_url).query).get("state", [])
    if len(states) != 1 or not states[0].endswith(f".{port}"):
        return None
    return states[0]


def connected(connection: Mapping[str, Any]) -> str:
    """A completed sign-in in words: the connection and the account it connected."""

    account = connection.get("account")
    name = account.get("name") if isinstance(account, Mapping) else None
    to = f" to {name}" if isinstance(name, str) and name else ""
    return f"{connection.get('name')} is connected{to}."
