from __future__ import annotations

import argparse
import base64
import hashlib
import html
import math
import secrets
import sys
import time
import webbrowser
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from cayu.providers.openai_subscription import (
    HttpxOpenAISubscriptionOAuthTransport,
    OpenAISubscriptionAuthError,
    OpenAISubscriptionAuthStore,
    OpenAISubscriptionCredentials,
    build_openai_subscription_authorize_url,
    openai_subscription_credentials_from_token_response,
)

_CALLBACK_PORT = 1455
_CALLBACK_PATH = "/auth/callback"
_LOGIN_TIMEOUT_SECONDS = 300.0
_DEVICE_POLLING_SAFETY_SECONDS = 3.0


def add_auth_parser(subparsers: argparse._SubParsersAction) -> None:
    auth_parser = subparsers.add_parser(
        "auth",
        help="Manage local provider sign-ins.",
        description="Manage local provider sign-ins.",
    )
    providers = auth_parser.add_subparsers(dest="auth_provider")
    openai = providers.add_parser(
        "openai",
        help="Manage the experimental OpenAI subscription sign-in.",
        description=(
            "Manage the experimental OpenAI subscription sign-in. "
            "Use `cayu auth openai status` to inspect local state."
        ),
    )
    actions = openai.add_subparsers(dest="auth_action")
    login = actions.add_parser(
        "login",
        help="Sign in with a ChatGPT Plus/Pro subscription.",
        description=(
            "Sign in with a ChatGPT Plus/Pro subscription. "
            "Run `cayu auth openai status` afterward to verify local state."
        ),
    )
    login.add_argument(
        "--headless",
        action="store_true",
        help="Use the device-code flow for SSH, containers, and remote hosts.",
    )
    login.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the authorization URL without opening a browser.",
    )
    actions.add_parser(
        "status",
        help="Show subscription sign-in status.",
        description=(
            "Show subscription sign-in status. Use `cayu auth openai login` when no "
            "active credentials are available."
        ),
    )
    actions.add_parser(
        "logout",
        help="Delete Cayu's local subscription credentials.",
        description=(
            "Delete Cayu's local subscription credentials. "
            "Run `cayu auth openai status` to confirm removal."
        ),
    )


def run_auth(args: argparse.Namespace) -> int:
    if args.auth_provider != "openai" or args.auth_action is None:
        print("Usage: cayu auth openai {login,status,logout}", file=sys.stderr)
        return 2
    store = OpenAISubscriptionAuthStore()
    try:
        if args.auth_action == "status":
            credentials = store.load()
            if credentials is None:
                print("OpenAI subscription: not signed in")
                return 1
            account = credentials.account_id or "unknown account"
            expires = datetime.fromtimestamp(credentials.expires_at, tz=UTC).isoformat()
            state = "active" if credentials.expires_at > time.time() else "refresh required"
            print(f"OpenAI subscription: signed in ({account}; {state}; token expires {expires})")
            return 0
        if args.auth_action == "logout":
            removed = store.delete()
            print(
                "OpenAI subscription: signed out"
                if removed
                else "OpenAI subscription: already signed out"
            )
            return 0
        if args.auth_action == "login":
            _print_experimental_notice()
            transport = HttpxOpenAISubscriptionOAuthTransport()
            credentials = (
                _device_login(transport=transport)
                if args.headless
                else _browser_login(
                    transport=transport,
                    open_browser=not args.no_browser,
                )
            )
            store.save(credentials)
            account = credentials.account_id or "account id unavailable"
            print(f"OpenAI subscription: signed in ({account})")
            return 0
    except (OSError, ValueError, OpenAISubscriptionAuthError) as exc:
        print(f"OpenAI subscription login failed: {exc}", file=sys.stderr)
        return 1
    print(f"Unknown auth action: {args.auth_action}", file=sys.stderr)
    return 2


def _print_experimental_notice() -> None:
    print(
        "Experimental OpenAI subscription support: OpenAI documents ChatGPT sign-in for "
        "Codex clients, not Cayu's raw provider adapter. This integration may stop working."
    )
    print(
        "Cayu sends its own identity (`originator: cayu`) and will not impersonate Codex "
        "or bypass an upstream rejection."
    )


class _CallbackServer(HTTPServer):
    authorization_code: str | None = None
    callback_error: str | None = None
    expected_state: str = ""


class _OAuthCallbackIgnored(OpenAISubscriptionAuthError):
    """A callback that cannot complete the active OAuth attempt."""


class _CallbackHandler(BaseHTTPRequestHandler):
    server: _CallbackServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != _CALLBACK_PATH:
            self.send_error(404)
            return
        query = parse_qs(parsed.query)
        try:
            code = _authorization_code_from_callback(
                query,
                expected_state=self.server.expected_state,
            )
        except _OAuthCallbackIgnored as exc:
            self._respond("OpenAI sign-in failed", str(exc), status=400)
            return
        except OpenAISubscriptionAuthError as exc:
            self.server.callback_error = str(exc)
            self._respond("OpenAI sign-in failed", self.server.callback_error, status=400)
            return
        self.server.authorization_code = code
        self._respond("OpenAI sign-in complete", "You can close this window and return to Cayu.")

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _respond(self, title: str, message: str, *, status: int = 200) -> None:
        body = (
            "<!doctype html><meta charset=utf-8>"
            f"<title>{html.escape(title)}</title>"
            f"<h1>{html.escape(title)}</h1><p>{html.escape(message)}</p>"
        ).encode()
        self.send_response(status)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _browser_login(
    *,
    transport: HttpxOpenAISubscriptionOAuthTransport,
    open_browser: bool,
) -> OpenAISubscriptionCredentials:
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    state = secrets.token_urlsafe(32)
    redirect_uri = f"http://localhost:{_CALLBACK_PORT}{_CALLBACK_PATH}"
    authorize_url = build_openai_subscription_authorize_url(
        redirect_uri=redirect_uri,
        code_challenge=challenge,
        state=state,
        issuer=transport.issuer,
    )

    try:
        server = _CallbackServer(("127.0.0.1", _CALLBACK_PORT), _CallbackHandler)
    except OSError as exc:
        raise OpenAISubscriptionAuthError(
            f"Could not listen on localhost:{_CALLBACK_PORT}; try `--headless`."
        ) from exc
    server.expected_state = state
    server.timeout = 1.0
    print(f"Open this URL to sign in:\n{authorize_url}")
    if open_browser:
        webbrowser.open(authorize_url)
    deadline = time.monotonic() + _LOGIN_TIMEOUT_SECONDS
    try:
        while (
            server.authorization_code is None
            and server.callback_error is None
            and time.monotonic() < deadline
        ):
            server.handle_request()
    finally:
        server.server_close()
    if server.callback_error is not None:
        raise OpenAISubscriptionAuthError(server.callback_error)
    if server.authorization_code is None:
        raise OpenAISubscriptionAuthError("OpenAI subscription login timed out.")
    response = transport.exchange_authorization_code(
        code=server.authorization_code,
        redirect_uri=redirect_uri,
        code_verifier=verifier,
    )
    return openai_subscription_credentials_from_token_response(response, now=time.time())


def _device_login(
    *,
    transport: HttpxOpenAISubscriptionOAuthTransport,
) -> OpenAISubscriptionCredentials:
    initiation = transport.request_device_authorization()
    device_auth_id = _required_response_string(initiation, "device_auth_id")
    user_code = _required_response_string(initiation, "user_code")
    interval_raw = initiation.get("interval", 5)
    try:
        if isinstance(interval_raw, bool):
            raise TypeError
        interval = float(interval_raw)
    except (OverflowError, TypeError, ValueError):
        raise OpenAISubscriptionAuthError(
            "OpenAI device authorization contained invalid interval."
        ) from None
    if not math.isfinite(interval) or interval <= 0:
        raise OpenAISubscriptionAuthError("OpenAI device authorization contained invalid interval.")
    interval = max(interval, 1.0)
    print(f"Open {transport.issuer}/codex/device and enter code: {user_code}")
    deadline = time.monotonic() + 900.0
    while time.monotonic() < deadline:
        response = transport.poll_device_authorization(
            device_auth_id=device_auth_id,
            user_code=user_code,
        )
        if response is not None:
            authorization_code = _required_response_string(response, "authorization_code")
            verifier = _required_response_string(response, "code_verifier")
            tokens = transport.exchange_device_authorization(
                authorization_code=authorization_code,
                code_verifier=verifier,
            )
            return openai_subscription_credentials_from_token_response(tokens, now=time.time())
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval + _DEVICE_POLLING_SAFETY_SECONDS, remaining))
    raise OpenAISubscriptionAuthError("OpenAI device login timed out.")


def _required_response_string(response: Any, key: str) -> str:
    if not isinstance(response, dict):
        raise OpenAISubscriptionAuthError("OpenAI OAuth response must be a JSON object.")
    value = response.get(key)
    if not isinstance(value, str) or not value.strip():
        raise OpenAISubscriptionAuthError(f"OpenAI OAuth response omitted {key}.")
    return value.strip()


def _first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _authorization_code_from_callback(
    query: dict[str, list[str]],
    *,
    expected_state: str,
) -> str:
    returned_state = _first_query_value(query, "state")
    if returned_state is None or not secrets.compare_digest(returned_state, expected_state):
        raise _OAuthCallbackIgnored("OAuth callback state did not match.")
    error = _first_query_value(query, "error_description") or _first_query_value(query, "error")
    if error is not None:
        raise OpenAISubscriptionAuthError(error.strip()[:240] or "OpenAI sign-in failed.")
    code = _first_query_value(query, "code")
    if code is None:
        raise _OAuthCallbackIgnored("OAuth callback omitted the authorization code.")
    return code
