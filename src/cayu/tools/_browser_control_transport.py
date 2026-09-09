"""Private guest control transport; connection establishment grants no input authority.

This module is also shipped beside the standalone browser guest. Imports of the
optional WebSocket dependency are deferred until the explicitly enabled channel
is opened. The caller owns disconnect fencing and positive native quiescence;
closing a network connection is never evidence that browser input stopped.
"""

from __future__ import annotations

import logging
import re
import ssl
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


CONTROL_MESSAGE_BYTES = 64 * 1024
CONTROL_FRAME_BYTES = 2 * 1024 * 1024
CONTROL_SUBPROTOCOL = "cayu.browser-control.v1"


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
        tls = ssl.create_default_context()
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
