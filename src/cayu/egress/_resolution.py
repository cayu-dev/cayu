from __future__ import annotations

import asyncio
import ipaddress
import socket

# Keep public-only admission stable across supported Python patch versions. These
# classifications were corrected in Python 3.13 and backported to later 3.11/
# 3.12 patch releases, but Cayu also supports earlier patch releases.
_IPV4_NON_GLOBAL_NETWORKS = (ipaddress.IPv4Network("192.0.0.0/24"),)
_IPV4_GLOBAL_EXCEPTIONS = frozenset(
    {
        ipaddress.IPv4Address("192.0.0.9"),
        ipaddress.IPv4Address("192.0.0.10"),
    }
)
_IPV6_NON_GLOBAL_NETWORKS = (
    ipaddress.IPv6Network("64:ff9b:1::/48"),
    ipaddress.IPv6Network("2001::/23"),
    ipaddress.IPv6Network("2002::/16"),
)
_IPV6_GLOBAL_EXCEPTIONS = (
    ipaddress.IPv6Network("2001:1::1/128"),
    ipaddress.IPv6Network("2001:1::2/128"),
    ipaddress.IPv6Network("2001:3::/32"),
    ipaddress.IPv6Network("2001:4:112::/48"),
    ipaddress.IPv6Network("2001:20::/28"),
    ipaddress.IPv6Network("2001:30::/28"),
)


class InvalidResolvedAddressError(ValueError):
    """A resolver answer was not an IP address."""


class ProhibitedResolvedAddressError(ValueError):
    """A resolver answer is outside the admitted destination class."""


async def resolve_destination(host: str, port: int) -> tuple[str, ...]:
    """Resolve and de-duplicate stream addresses in resolver order."""

    records = await asyncio.get_running_loop().getaddrinfo(
        host,
        port,
        type=socket.SOCK_STREAM,
    )
    return tuple(dict.fromkeys(str(record[4][0]) for record in records))


def _is_globally_reachable(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Return a version-stable global-address classification."""

    if isinstance(address, ipaddress.IPv4Address):
        if address in _IPV4_GLOBAL_EXCEPTIONS:
            return True
        if any(address in network for network in _IPV4_NON_GLOBAL_NETWORKS):
            return False
        return address.is_global

    if address.ipv4_mapped is not None:
        return _is_globally_reachable(address.ipv4_mapped)
    if any(address in network for network in _IPV6_GLOBAL_EXCEPTIONS):
        return True
    if any(address in network for network in _IPV6_NON_GLOBAL_NETWORKS):
        return False
    return address.is_global


def validated_resolved_address(address: str, *, allow_private: bool) -> str:
    """Return one canonical admitted IP address or fail closed."""

    try:
        resolved = ipaddress.ip_address(address)
    except ValueError as exc:
        raise InvalidResolvedAddressError(
            "Upstream destination returned an invalid address."
        ) from exc
    if (
        resolved.is_loopback
        or resolved.is_link_local
        or resolved.is_multicast
        or resolved.is_reserved
        or resolved.is_unspecified
        or (not allow_private and not _is_globally_reachable(resolved))
    ):
        raise ProhibitedResolvedAddressError(
            "Upstream destination resolved to a prohibited address."
        )
    return resolved.compressed


__all__ = [
    "InvalidResolvedAddressError",
    "ProhibitedResolvedAddressError",
    "resolve_destination",
    "validated_resolved_address",
]
