from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit

from cayu.egress.errors import UnsupportedEgressError

MICROSANDBOX_HOST = "host.microsandbox.internal"


@dataclass(frozen=True)
class HttpProxyEndpoint:
    """Validated interpretation of one HTTP CONNECT proxy URL."""

    url: str
    host: str
    port: int

    @classmethod
    def parse(cls, url: str) -> HttpProxyEndpoint:
        split = urlsplit(url)
        if (
            split.scheme != "http"
            or split.hostname is None
            or split.username is not None
            or split.password is not None
            or split.path not in ("", "/")
            or split.query
            or split.fragment
        ):
            raise ValueError("HTTP proxy URL must be absolute and contain no credentials or path.")
        try:
            port = split.port or 80
        except ValueError as exc:
            raise ValueError("HTTP proxy URL contains an invalid port.") from exc
        if port <= 0:
            raise ValueError("HTTP proxy URL contains an invalid port.")
        return cls(url=url, host=split.hostname, port=port)


@dataclass
class ExposedProxy:
    """A runtime-reachable URL for one in-process Cayu proxy listener."""

    proxy_url: str
    teardown: Callable[[], Awaitable[None]] | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    async def close(self) -> None:
        if self._closed:
            return
        if self.teardown is not None:
            await self.teardown()
        self._closed = True


class ProxyExposure(Protocol):
    """Exposes a local Cayu proxy listener to a sandbox runtime."""

    async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy: ...


class MicrosandboxHostProxyExposure:
    """Advertises a host listener through Microsandbox's reserved host name."""

    async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
        normalized = local_host.strip().lower()
        if normalized in {"localhost", "127.0.0.1", "::1"}:
            raise UnsupportedEgressError(
                "Microsandbox cannot reach a loopback-only Cayu proxy listener."
            )
        if local_port <= 0:
            raise ValueError("local_port must be positive.")
        return ExposedProxy(proxy_url=f"http://{MICROSANDBOX_HOST}:{local_port}")
