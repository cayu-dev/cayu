"""Private guest control transport; connection establishment grants no input authority.

This module is also shipped beside the standalone browser guest. Imports of the
optional WebSocket dependency are deferred until the explicitly enabled channel
is opened. The caller owns disconnect fencing and positive native quiescence;
closing a network connection is never evidence that browser input stopped.
"""

from __future__ import annotations

import logging
import os
import re
import ssl
import stat
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


CONTROL_MESSAGE_BYTES = 64 * 1024
CONTROL_FRAME_BYTES = 2 * 1024 * 1024
CONTROL_SUBPROTOCOL = "cayu.browser-control.v1"
# Public roots for the control plane, installed by a trusted setup step. The
# worker's sanitized environment points SSL_CERT_FILE at the session egress CA,
# which replaces a bundle-file system store (such as Amazon Linux's), so the
# control roots are loaded from this separate path instead.
CONTROL_CA_PATH = "/etc/cayu/control-ca.pem"
_CONTROL_CA_MAX_BYTES = 64 * 1024


class BrowserControlTransportUnavailable(RuntimeError):
    def __init__(self) -> None:
        super().__init__("The protected browser control channel is unavailable.")


def control_connection_closed_normally(error: BaseException) -> bool:
    """Recognize only the transport's positive normal-close outcome."""
    try:
        from websockets.exceptions import ConnectionClosedOK
    except ImportError:
        return False
    return isinstance(error, ConnectionClosedOK)


def validate_control_endpoint(endpoint: str) -> str:
    """Accept only an explicit TLS endpoint, without URL-carried credentials."""
    if (
        type(endpoint) is not str
        or not 1 <= len(endpoint) <= 2048
        or not endpoint.isascii()
        or any(ord(char) <= 32 or ord(char) == 127 for char in endpoint)
        or "\\" in endpoint
        or "?" in endpoint
        or "#" in endpoint
    ):
        raise BrowserControlTransportUnavailable()
    try:
        parsed = urlsplit(endpoint)
        valid = (
            parsed.scheme == "wss"
            and parsed.hostname is not None
            and parsed.username is None
            and parsed.password is None
            and parsed.port != 0
            and parsed.path.startswith("/")
        )
    except ValueError:
        valid = False
    if not valid:
        raise BrowserControlTransportUnavailable()
    return endpoint


def control_tls_context(ca_path: str = CONTROL_CA_PATH) -> ssl.SSLContext:
    """Verify the control plane with the default roots plus installed control roots.

    The control roots are trusted only from a root-owned regular file that no
    other user can write, in a root-owned directory, so an unprivileged guest
    process cannot substitute them. An absent file adds nothing, which keeps the
    default trust behavior. A present file that is unsafe or unreadable fails
    closed rather than silently falling back.
    """
    context = ssl.create_default_context()
    try:
        info = os.lstat(ca_path)
    except FileNotFoundError:
        return context
    except OSError:
        raise BrowserControlTransportUnavailable() from None
    try:
        parent = os.lstat(os.path.dirname(ca_path))
    except OSError:
        raise BrowserControlTransportUnavailable() from None
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_mode & 0o022
        or info.st_size > _CONTROL_CA_MAX_BYTES
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != 0
        or parent.st_mode & 0o022
    ):
        raise BrowserControlTransportUnavailable()
    try:
        context.load_verify_locations(cafile=ca_path)
    except (OSError, ssl.SSLError, ValueError):
        raise BrowserControlTransportUnavailable() from None
    return context


def _private_transport_logger() -> logging.Logger:
    # Never register this logger: application-wide DEBUG configuration must not
    # enable headers, credentials, screenshots, or input payloads in wire logs.
    logger = logging.Logger("cayu.private.browser-control", level=logging.CRITICAL + 1)
    logger.disabled = True
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    return logger


async def open_guest_control_channel(
    *,
    endpoint: str,
    credential: str,
    tls: ssl.SSLContext | None = None,
    subprotocol: str = CONTROL_SUBPROTOCOL,
) -> ClientConnection:
    """Open one bounded connection, without redirects, retry, or implicit proxy.

    ``credential`` is an opaque short-lived guest capability supplied through
    private bootstrap I/O, never a workload credential or ordinary tool argument.
    A test/deployment trust context may add roots but cannot disable verification.
    The channel owner must validate the server's exact protocol handshake before
    admitting any capture or input and supervise the returned connection's close.
    """
    endpoint = validate_control_endpoint(endpoint)
    if type(credential) is not str or re.fullmatch(r"[A-Za-z0-9._~-]{32,4096}", credential) is None:
        raise BrowserControlTransportUnavailable()
    if tls is None:
        tls = control_tls_context()
    if tls.verify_mode != ssl.CERT_REQUIRED or not tls.check_hostname:
        raise BrowserControlTransportUnavailable()

    from websockets.asyncio.client import connect
    from websockets.typing import Subprotocol

    class _ExactEndpointConnection(connect):
        def process_redirect(self, exc: Exception) -> Exception | str:
            # No hop is part of this capability's authority, even same-origin.
            return exc

    connection = await _ExactEndpointConnection(
        endpoint,
        ssl=tls,
        additional_headers={"Authorization": f"Bearer {credential}"},
        subprotocols=[Subprotocol(subprotocol)],
        compression=None,
        proxy=None,
        open_timeout=5,
        close_timeout=2,
        ping_interval=10,
        ping_timeout=10,
        max_size=CONTROL_MESSAGE_BYTES,
        max_queue=1,
        write_limit=64 * 1024,
        logger=_private_transport_logger(),
    )
    return connection
