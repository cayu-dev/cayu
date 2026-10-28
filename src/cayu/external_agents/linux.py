"""Run vendor CLIs with Linux filesystem, network and process confinement.

See ``cayu guide external-agents`` for launcher selection and examples.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from cayu.external_agents.base import ExternalAgentLaunch
from cayu.external_agents.environment import _guest_secret_values, _validated_environment
from cayu.external_agents.local import local_agent_launch

_BOOTSTRAP = Path(__file__).with_name("_linux_sandbox.py")
_SYSTEM_READ_PATHS = ("/usr", "/bin", "/lib", "/lib64", "/etc/ld.so.cache", "/etc/localtime")
_DEVICE_READ_PATHS = ("/dev/urandom", "/dev/random")


def _paths(values: Sequence[str | os.PathLike[str]]) -> list[str]:
    return list(dict.fromkeys(str(Path(value).resolve(strict=True)) for value in values))


@asynccontextmanager
async def linux_agent_launch(
    *,
    cli: str,
    cwd: str | os.PathLike[str],
    read_only_paths: Sequence[str | os.PathLike[str]] = (),
    writable_paths: Sequence[str | os.PathLike[str]] = (),
    allowed_tcp_ports: Sequence[int] = (),
    environment: Mapping[str, str] | None = None,
    close_timeout_s: float = 10.0,
) -> AsyncIterator[ExternalAgentLaunch]:
    """Confine a CLI without Docker, root privileges or user namespaces.

    Requires Linux x86_64/aarch64 and Landlock ABI 6. The checkout, a fresh
    temporary HOME/TMPDIR, and ``writable_paths`` are writable. System binaries
    and libraries, the CLI executable, and ``read_only_paths`` are readable.
    Add the CLI's installation directory if it loads adjacent files. The host
    home and credentials are not inherited. Only ``environment`` is forwarded,
    along with a system PATH and the private HOME/TMPDIR.

    TCP connections are denied unless their destination port appears in
    ``allowed_tcp_ports``. These are port grants on ANY address, not host/IP
    allowlists. Use a host firewall or a network-isolated runner when you need
    host-based egress. UDP, raw and external UNIX sockets are denied; ordinary
    stdin/stdout and child processes work. A gateway on an explicitly granted
    port can hold the real model credentials outside the guest.

    Descendants cannot leave their process group or signal the supervisor.
    ``close`` confirms that group is empty. No CPU, memory, attempt or execution
    deadline is imposed by the launcher; the run owns those policies. Missing
    confinement support fails before yielding a launcher, with no local fallback.
    """
    if sys.platform != "linux":
        raise RuntimeError("linux_agent_launch requires Linux with Landlock ABI 6.")
    executable = shutil.which(cli) if not os.path.isabs(cli) else cli
    if executable is None or not os.access(executable, os.X_OK):
        raise ValueError("The vendor CLI must name an executable file.")
    executable = str(Path(executable).resolve(strict=True))
    root = Path(cwd).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("The CLI working directory must be a directory.")
    ports = tuple(allowed_tcp_ports)
    if any(type(port) is not int or not 1 <= port <= 65535 for port in ports):
        raise ValueError("allowed_tcp_ports must contain TCP ports from 1 to 65535.")
    writable = _paths([root, *writable_paths])
    # A guest must not rewrite its trusted bootstrap, interpreter, or supervisor.
    protected = (
        _BOOTSTRAP.resolve(),
        Path(sys.executable).resolve(),
        Path(tempfile.gettempdir()).resolve(),
    )
    if any(path.is_relative_to(grant) for path in protected for grant in writable):
        raise ValueError(
            "Writable grants must not contain the launcher, interpreter or host temp root."
        )
    system_paths = [Path(name).resolve() for name in _SYSTEM_READ_PATHS if Path(name).exists()]
    if any(Path(grant).is_relative_to(path) for path in system_paths for grant in writable):
        raise ValueError(
            "Writable grants must not modify the launcher's system binaries or libraries."
        )
    readable = _paths(
        [
            *system_paths,
            *(name for name in _DEVICE_READ_PATHS if Path(name).exists()),
            executable,
            *read_only_paths,
        ]
    )
    explicit = _validated_environment(environment or {})
    with tempfile.TemporaryDirectory(prefix="cayu-linux-agent-") as directory:
        state = Path(directory)
        home = state / "home"
        home.mkdir()
        guest_env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            **explicit,
            "HOME": str(home),
            "TMPDIR": str(home),
        }
        config = {
            "cli": executable,
            "cwd": str(root),
            "read_only": readable,
            "writable": [*writable, str(home), "/dev/null"],
            "tcp_ports": ports,
            "environment": guest_env,
            "preflight": True,
        }
        config_path = state / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        config_path.chmod(0o600)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(_BOOTSTRAP),
            str(config_path),
            env={"PATH": os.defpath},
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await process.wait()
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode != 0:
            raise RuntimeError(
                "Linux sandbox preflight failed; Landlock ABI 6 and seccomp are required."
            )
        config["preflight"] = False
        config_path.write_text(json.dumps(config), encoding="utf-8")
        wrapper = state / "cli"
        wrapper.write_text(
            f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -I "
            f'{shlex.quote(str(_BOOTSTRAP))} {shlex.quote(str(config_path))} "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o700)
        # Guest loader variables (LD_PRELOAD, PYTHONPATH, etc.) belong only
        # after confinement. Never give them to the SDK-spawned supervisor.
        bootstrap_env = {"PATH": os.defpath, "LANG": "C.UTF-8", "HOME": str(home)}
        async with local_agent_launch(
            cli=str(wrapper),
            cwd=root,
            environment=bootstrap_env,
            close_timeout_s=close_timeout_s,
        ) as launch:
            yield replace(
                launch,
                environment=bootstrap_env,
                secret_values=tuple(value for value in _guest_secret_values(guest_env) if value),
            )
