from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import sys
from pathlib import Path
from typing import TypeVar

from cayu._task_wait import (
    ShieldedTaskOutcome,
    await_shielded_task_outcome,
    restore_task_cancellation_requests,
)

_MAX_RESOLVER_ADDRESSES = 64
_MAX_RESOLVER_OUTPUT_BYTES = 16 * 1024
_OWNED_RESOLVER_HELPER = Path(__file__).with_name("_resolution_helper.py")
_SettlementT = TypeVar("_SettlementT")

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


async def _terminate_and_wait_for_resolver_process(
    process: asyncio.subprocess.Process,
    settlement_task: asyncio.Task[_SettlementT],
    *,
    cancellation: asyncio.CancelledError | None = None,
) -> tuple[BaseException | None, ShieldedTaskOutcome[_SettlementT]]:
    """Request termination and retain ownership until the helper is quiescent."""

    termination_error: BaseException | None = None
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except BaseException as exc:
            # A failed termination request is not proof that the subprocess
            # stopped. Keep awaiting the exact process so upstream capacity
            # remains fenced until natural exit provides that proof.
            termination_error = exc
    outcome = await await_shielded_task_outcome(
        settlement_task,
        cancellation=cancellation,
    )
    return termination_error, outcome


def _resolver_cleanup_failure(
    *errors: BaseException | None,
) -> BaseException | None:
    retained = [error for error in errors if error is not None]
    if not retained:
        return None
    if len(retained) == 1:
        return retained[0]
    return BaseExceptionGroup("Destination resolver cleanup failed.", retained)


async def resolve_destination(host: str, port: int) -> tuple[str, ...]:
    """Resolve and de-duplicate stream addresses in resolver order."""

    records = await asyncio.get_running_loop().getaddrinfo(
        host,
        port,
        type=socket.SOCK_STREAM,
    )
    return tuple(dict.fromkeys(str(record[4][0]) for record in records))


async def resolve_destination_owned(host: str, port: int) -> tuple[str, ...]:
    """Resolve through a killable helper so cancellation proves quiescence."""

    helper = str(_OWNED_RESOLVER_HELPER)
    spawn_task = asyncio.create_task(
        asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-S",
            helper,
            host,
            str(port),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        ),
        name="cayu-egress-resolver-spawn",
    )
    process: asyncio.subprocess.Process | None = None
    try:
        try:
            process = await asyncio.shield(spawn_task)
        except asyncio.CancelledError as cancellation:
            outcome = await await_shielded_task_outcome(
                spawn_task,
                cancellation=cancellation,
            )
            process = outcome.result
            consumed = outcome.cancellation_requests_consumed
            cleanup_failure = outcome.error
            if process is not None and process.returncode is None:
                wait_task = asyncio.create_task(
                    process.wait(),
                    name="cayu-egress-resolver-spawn-cancellation-wait",
                )
                termination_error, wait_outcome = await _terminate_and_wait_for_resolver_process(
                    process,
                    wait_task,
                    cancellation=outcome.cancellation,
                )
                consumed += wait_outcome.cancellation_requests_consumed
                cleanup_failure = _resolver_cleanup_failure(
                    cleanup_failure,
                    termination_error,
                    wait_outcome.error,
                )
            restore_task_cancellation_requests(
                consumed,
                cancellation=outcome.cancellation or cancellation,
            )
            if cleanup_failure is not None:
                raise cancellation from cleanup_failure
            raise cancellation

        communicate_task = asyncio.create_task(
            process.communicate(),
            name="cayu-egress-resolver-communicate",
        )
        try:
            stdout, _stderr = await asyncio.shield(communicate_task)
        except asyncio.CancelledError as cancellation:
            termination_error, outcome = await _terminate_and_wait_for_resolver_process(
                process,
                communicate_task,
                cancellation=cancellation,
            )
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed,
                cancellation=outcome.cancellation or cancellation,
            )
            cleanup_failure = _resolver_cleanup_failure(
                termination_error,
                outcome.error,
            )
            if cleanup_failure is not None:
                raise cancellation from cleanup_failure
            raise cancellation
    except BaseException as primary_error:
        if process is not None and process.returncode is None:
            wait_task = asyncio.create_task(
                process.wait(),
                name="cayu-egress-resolver-failure-wait",
            )
            termination_error, wait_outcome = await _terminate_and_wait_for_resolver_process(
                process, wait_task
            )
            if wait_outcome.cancellation is not None:
                restore_task_cancellation_requests(
                    wait_outcome.cancellation_requests_consumed,
                    cancellation=wait_outcome.cancellation,
                )
                cleanup_failure = _resolver_cleanup_failure(
                    primary_error,
                    termination_error,
                    wait_outcome.error,
                )
                if cleanup_failure is not None:
                    raise wait_outcome.cancellation from cleanup_failure
                raise wait_outcome.cancellation from None
            cleanup_failure = _resolver_cleanup_failure(
                primary_error,
                termination_error,
                wait_outcome.error,
            )
            if cleanup_failure is not primary_error:
                assert cleanup_failure is not None
                raise cleanup_failure from None
        raise

    if process.returncode != 0 or len(stdout) > _MAX_RESOLVER_OUTPUT_BYTES:
        raise OSError("Destination resolution helper failed.")
    try:
        decoded = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OSError("Destination resolution helper returned invalid evidence.") from exc
    if (
        type(decoded) is not list
        or len(decoded) > _MAX_RESOLVER_ADDRESSES
        or any(type(address) is not str for address in decoded)
    ):
        raise OSError("Destination resolution helper returned invalid evidence.")
    return tuple(decoded)


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
    "resolve_destination_owned",
    "validated_resolved_address",
]
