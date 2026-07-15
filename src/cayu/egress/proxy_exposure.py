from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from ipaddress import IPv4Address, ip_address
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
    """A runtime-reachable URL for one in-process Cayu proxy listener.

    ``credentialless_isolated`` is an explicit transport-boundary assertion: the
    endpoint is reachable only by the intended sandbox and Cayu's trusted host,
    not by the public internet, a shared tenant network, or another sandbox.
    Credentialless routes fail closed unless an exposure makes this assertion.
    """

    proxy_url: str
    teardown: Callable[[], Awaitable[None]] | None = None
    credentialless_isolated: bool = False
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.credentialless_isolated) is not bool:
            raise TypeError("credentialless_isolated must be a boolean.")

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
        if normalized != "127.0.0.1":
            raise UnsupportedEgressError(
                "Microsandbox virtual egress requires Cayu's paired IPv4/IPv6 "
                "loopback proxy listener."
            )
        if local_port <= 0:
            raise ValueError("local_port must be positive.")
        return ExposedProxy(proxy_url=f"http://{MICROSANDBOX_HOST}:{local_port}")


class VpcTaskProxyExposure:
    """Advertise a proxy listener through the private IPv4 of its VPC task.

    This exposure is intended for a Cayu control plane running in ECS/Fargate.
    Lambda MicroVMs reach the in-process proxy through a VPC egress connector;
    the address is deliberately restricted to RFC 1918 space so an accidental
    public listener cannot become the credential-broker boundary.
    """

    def __init__(self, task_ipv4: str) -> None:
        try:
            address = ip_address(task_ipv4)
        except ValueError as exc:
            raise ValueError("task_ipv4 must be a private IPv4 address.") from exc
        if not isinstance(address, IPv4Address) or not _is_rfc1918(address):
            raise ValueError("task_ipv4 must be a private IPv4 address.")
        self.task_ipv4 = str(address)

    async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
        if local_host.strip() != "0.0.0.0":
            raise UnsupportedEgressError(
                "VPC task proxy exposure requires Cayu to listen on 0.0.0.0."
            )
        if local_port <= 0:
            raise ValueError("local_port must be positive.")
        return ExposedProxy(proxy_url=f"http://{self.task_ipv4}:{local_port}")


def _is_rfc1918(address: IPv4Address) -> bool:
    value = int(address)
    return (
        int(IPv4Address("10.0.0.0")) <= value <= int(IPv4Address("10.255.255.255"))
        or int(IPv4Address("172.16.0.0")) <= value <= int(IPv4Address("172.31.255.255"))
        or int(IPv4Address("192.168.0.0")) <= value <= int(IPv4Address("192.168.255.255"))
    )
