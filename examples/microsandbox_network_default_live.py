from __future__ import annotations

import asyncio
import importlib.metadata
import os
import subprocess
import uuid
from collections.abc import Sequence

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from cayu import ExecCommand, ExecResult, MicrosandboxRunner

_IPV4_PROBE = """
import socket
import sys

try:
    connection = socket.create_connection(("1.1.1.1", 80), timeout=3)
except OSError as exc:
    print(f"blocked:{type(exc).__name__}")
    sys.exit(7)
else:
    connection.close()
    print("reachable")
"""
_DNS_PROBE = """
import socket
import sys

try:
    socket.getaddrinfo("example.com", 443)
except OSError as exc:
    print(f"blocked:{type(exc).__name__}")
    sys.exit(7)
else:
    print("reachable")
"""


async def main() -> None:
    if os.environ.get("CAYU_RUN_MICROSANDBOX_NETWORK_LIVE") != "1":
        raise RuntimeError(
            "Set CAYU_RUN_MICROSANDBOX_NETWORK_LIVE=1 to run the live network proof."
        )

    import microsandbox  # ty: ignore[unresolved-import]

    supported_version = declared_microsandbox_version(importlib.metadata.requires("cayu"))
    sdk_version = importlib.metadata.version("microsandbox")
    cli_version = _microsandbox_cli_version()
    if sdk_version != supported_version:
        raise RuntimeError(f"Expected microsandbox SDK {supported_version}, got {sdk_version}.")
    if cli_version != f"msb {supported_version}":
        raise RuntimeError(f"Expected msb CLI {supported_version}, got {cli_version!r}.")
    print(f"microsandbox_sdk_version={sdk_version}")
    print(f"microsandbox_cli_version={cli_version}")

    suffix = uuid.uuid4().hex[:10]
    image = os.environ.get("CAYU_MICROSANDBOX_IMAGE", "python:3.13")
    async with await MicrosandboxRunner.create(
        f"cayu-network-default-{suffix}",
        image=image,
        close_action="remove",
        replace=True,
    ) as denied_runner:
        ipv4_denied = await denied_runner.exec(
            ExecCommand.process("python3", "-c", _IPV4_PROBE),
            timeout_s=10,
        )
        _require_probe(ipv4_denied, label="default_ipv4", expected_exit=7)
        dns_denied = await denied_runner.exec(
            ExecCommand.process("python3", "-c", _DNS_PROBE),
            timeout_s=10,
        )
        _require_probe(dns_denied, label="default_dns", expected_exit=7)

    async with await MicrosandboxRunner.create(
        f"cayu-network-open-{suffix}",
        image=image,
        close_action="remove",
        replace=True,
        network=microsandbox.Network.allow_all(),
    ) as open_runner:
        ipv4_open = await open_runner.exec(
            ExecCommand.process("python3", "-c", _IPV4_PROBE),
            timeout_s=10,
        )
        _require_probe(ipv4_open, label="explicit_open_ipv4", expected_exit=0)


def declared_microsandbox_version(requirements: Sequence[str] | None) -> str:
    declarations: list[Requirement] = []
    for requirement in requirements or ():
        try:
            declaration = Requirement(requirement)
        except InvalidRequirement as exc:
            raise RuntimeError(
                "Cayu distribution metadata contains an invalid requirement."
            ) from exc
        if canonicalize_name(declaration.name) == "microsandbox":
            declarations.append(declaration)
    if len(declarations) == 1:
        declaration = declarations[0]
        specifiers = list(declaration.specifier)
        if (
            declaration.url is None
            and len(specifiers) == 1
            and specifiers[0].operator == "=="
            and not specifiers[0].version.endswith(".*")
        ):
            return specifiers[0].version
    raise RuntimeError(
        "Cayu distribution metadata must declare exactly one exact microsandbox requirement."
    )


def _microsandbox_cli_version() -> str:
    completed = subprocess.run(
        ["msb", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stdout or completed.stderr).strip()


def _require_probe(result: ExecResult, *, label: str, expected_exit: int) -> None:
    if result.timed_out or result.exit_code != expected_exit:
        raise AssertionError(
            f"{label} probe failed: exit_code={result.exit_code} timed_out={result.timed_out} "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    print(f"{label}={result.stdout.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
