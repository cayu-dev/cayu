"""Whole-process sandbox launchers; SDKs stay in the trusted service."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import secrets
import shutil
import struct
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar
from uuid import uuid4

from cayu._task_wait import await_shielded_task_outcome, restore_task_cancellation_requests
from cayu.external_agents.base import ExternalAgentLaunch
from cayu.runners.base import ExecCommand, Runner, RunnerInteractiveProcess, RunnerPortForward
from cayu.runners.docker import DockerRunner

_T = TypeVar("_T")


async def join_cleanup(operation: Awaitable[_T]) -> _T | None:
    """Finish owned cleanup through repeated cancellation and retain its outcome."""
    outcome = await await_shielded_task_outcome(asyncio.ensure_future(operation))
    restore_task_cancellation_requests(
        outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
    )
    if outcome.error is not None:
        raise outcome.error
    if outcome.cancellation is not None:
        raise outcome.cancellation
    return outcome.result


def scrub_external_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Override SDK inheritance without changing this process's environment."""
    return {**dict.fromkeys(os.environ, ""), **environment}


_MODEL_CREDENTIAL_KEYS = frozenset(
    {"ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "OPENAI_API_KEY"}
)


def _validated_guest_environment(environment: Mapping[str, str]) -> dict[str, str]:
    if _MODEL_CREDENTIAL_KEYS.intersection(environment):
        raise ValueError("Model credentials must stay in the service proxy.")
    return _validated_environment(environment)


def _validated_environment(environment: Mapping[str, str]) -> dict[str, str]:
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in environment):
        raise ValueError("External environment keys must be portable variable names.")
    if any(type(value) is not str for value in environment.values()):
        raise TypeError("External environment values must be strings.")
    return dict(environment)


def _guest_secret_values(environment: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        value
        for key, value in environment.items()
        if any(word in key.upper() for word in ("KEY", "TOKEN", "AUTHORIZATION"))
    )


async def _require_internal_network(docker: str, network: str) -> None:
    process = await asyncio.create_subprocess_exec(
        docker,
        "network",
        "inspect",
        "--format",
        "{{json .Internal}}",
        network,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(15):
            stdout, _ = await process.communicate()
        if process.returncode != 0 or stdout.strip() != b"true":
            raise ValueError("External agents require an existing Docker --internal network.")
    finally:
        if process.returncode is None:
            process.kill()
            await join_cleanup(process.wait())


@asynccontextmanager
async def docker_agent_launch(
    *,
    image: str,
    cli: str,
    workspace: Path,
    network: str,
    environment: Mapping[str, str],
    allocation_identity: str | None = None,
    workspace_read_only: bool = False,
) -> AsyncIterator[ExternalAgentLaunch]:
    """Allocate one CLI container and tunnel SDK stdio through ``docker exec``.

    ``network`` must be a deployment-owned internal Docker network whose only
    egress is the service proxy. CLI images contain their SDK-matching binary;
    no host home, Docker socket, or service credential is mounted. The checkout
    is the sole host mount. Container removal joins all guest descendants.
    """
    if not network or network in {"host", "bridge", "default"}:
        raise ValueError("An isolated, deployment-owned Docker network is required.")
    if not Path(cli).is_absolute():
        raise ValueError("The guest CLI path must be absolute.")
    root = workspace.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("External workspace must be a prepared directory.")
    environment = _validated_guest_environment(environment)
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker is required for the external agent launcher.")
    await _require_internal_network(docker, network)
    runner = await DockerRunner.create(
        f"cayu-external-{uuid4().hex}",
        image=image,
        mount_path=str(root),
        mount_read_only=workspace_read_only,
        network=network,
        replace=False,
        close_action="remove",
        allocation_identity=allocation_identity,
    )
    try:
        if runner.container_id is None:
            raise RuntimeError("Docker did not return an immutable container identity.")
        with TemporaryDirectory(prefix="cayu-external-") as directory:
            launcher = Path(directory) / "cli"
            command = [
                runner.docker_path,
                "exec",
                "-i",
                "--user",
                f"{root.stat().st_uid}:{root.stat().st_gid}",
                "--workdir",
                str(root),
                *[argument for key in environment for argument in ("-e", key)],
                runner.container_id,
                "sh",
                "-c",
                "mkdir -p /tmp/cayu-agent && exec env -i "
                "PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/cayu-agent "
                + " ".join(f'{key}="${{{key}}}"' for key in environment)
                + ' "$@"',
                "cayu-agent",
                cli,
            ]
            launcher.write_text(
                f"#!{sys.executable}\nimport os, sys\n"
                f"command = {command!r}\n"
                "os.execve(command[0], command + sys.argv[1:], "
                + repr(
                    {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp")}
                )
                + f" | {{key: os.environ[key] for key in {tuple(environment)!r}}})\n",
                encoding="utf-8",
            )
            launcher.chmod(0o700)
            yield ExternalAgentLaunch(
                str(launcher),
                str(root),
                runner.close,
                {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": directory, **environment},
                _guest_secret_values(environment),
            )
    finally:
        from cayu.external_agents._cleanup import close_external_guest

        await close_external_guest(runner.close)


_GUEST_HOME = "/tmp/cayu-agent"
# Set and create the guest home ($1), make the launch's working-directory path
# ($3, which also exists on the host for the SDK) name the guest working
# directory, then replace the shell with the CLI. "$0" is only a label. HOME
# travels as an argument, not in the environment, because a runner may own the
# environment's HOME (a brokered sandbox refuses it).
_GUEST_ENTRY = (
    'HOME=$1 && export HOME && mkdir -p -- "$HOME" "$2" '
    '&& { [ -e "$3" ] || ln -s -- "$PWD" "$3"; } && shift 3 && exec "$@"'
)
_MAX_SHIM_FRAME_BYTES = 4 * 1024 * 1024
_SHIM_HELLO_TIMEOUT_SECONDS = 30.0
# The SDK starts the CLI through this stand-in. It forwards its arguments and
# stdio over a private Unix socket to the launcher, which runs the real CLI in
# the guest, and exits with the guest's exit code. Frames are one kind byte and
# a big-endian length: h(ello), i(nput), e(nd of input) toward the launcher;
# o(utput) and x (exit code) back.
_SHIM_SOURCE = """\
import json, os, socket, struct, sys, threading
connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
connection.connect({socket_path!r})
lock = threading.Lock()
def send(kind, payload=b""):
    with lock:
        connection.sendall(kind + struct.pack(">I", len(payload)) + payload)
send(b"h", json.dumps({{"token": {token!r}, "argv": sys.argv[1:]}}).encode())
def forward_input():
    try:
        while True:
            chunk = os.read(0, 65536)
            if not chunk:
                break
            send(b"i", chunk)
        send(b"e")
    except OSError:
        pass
threading.Thread(target=forward_input, daemon=True).start()
def receive(size):
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)
status = 1
while True:
    header = receive(5)
    if header is None:
        break
    payload = receive(struct.unpack(">I", header[1:])[0])
    if payload is None:
        break
    if header[:1] == b"o":
        view = memoryview(payload)
        while view:
            view = view[os.write(1, view):]
    elif header[:1] == b"x":
        status = struct.unpack(">i", payload)[0]
        break
os._exit(status if 0 <= status < 256 else 1)
"""


async def _read_shim_frame(reader: asyncio.StreamReader) -> tuple[bytes, bytes]:
    header = await reader.readexactly(5)
    (size,) = struct.unpack(">I", header[1:])
    if size > _MAX_SHIM_FRAME_BYTES:
        raise ValueError("External CLI stand-in frame is too large.")
    return header[:1], await reader.readexactly(size)


def _shim_frame(kind: bytes, payload: bytes = b"") -> bytes:
    return kind + struct.pack(">I", len(payload)) + payload


@asynccontextmanager
async def runner_agent_launch(
    *,
    runner: Runner,
    cli: str,
    environment: Mapping[str, str],
    cwd: str | None = None,
    timeout_s: int | None = None,
    max_processes: int = 4,
    mcp_servers: Mapping[str, Any] | None = None,
    codex_config_overrides: tuple[str, ...] = (),
) -> AsyncIterator[ExternalAgentLaunch]:
    """Run the CLI inside any runner that can open interactive processes.

    The runner (for example a ``LambdaMicroVMRunner`` whose image has the CLI)
    is borrowed, not owned: the caller keeps its lifecycle and its network,
    which is where model credentials belong. The SDK starts a host-side stand-in
    that relays its arguments and stdio over a private Unix socket; each start
    opens one guest process through ``Runner.interactive_process_capability()``
    in ``cwd`` (default: the runner's working directory). ``close`` stops every
    guest process and returns only after the runner confirmed each whole
    process tree exited. ``timeout_s`` bounds each guest process even if this
    host disappears.

    The yielded ``cwd`` is a host directory, because the SDK starts the stand-in
    there; in the guest the same path is a link to the working directory, so a
    harness that reports or reuses its working directory sees one path.

    ``mcp_servers`` (Claude) and ``codex_config_overrides`` (Codex) carry
    per-run settings that name guest addresses, such as a gateway reached
    through ``runner_port_forward``.
    """
    if not Path(cli).is_absolute():
        raise ValueError("The guest CLI path must be absolute.")
    guest_environment = _validated_guest_environment(environment)
    guest_home = guest_environment.pop("HOME", _GUEST_HOME)
    if type(max_processes) is not int or max_processes < 1:
        raise ValueError("max_processes must be a positive integer.")
    capability = runner.interactive_process_capability()
    if capability is None:
        raise ValueError(
            "The runner cannot open interactive processes; use a runner that supports "
            "them, such as LambdaMicroVMRunner."
        )
    protect_secret: Callable[[str], None] | None = None
    declared_secrets = tuple(
        dict.fromkeys((*_guest_secret_values(guest_environment), *capability.secret_values))
    )

    def bind_secret_protector(protector: Callable[[str], None]) -> None:
        nonlocal protect_secret
        protect_secret = protector

    token = secrets.token_hex(32)
    processes: set[RunnerInteractiveProcess] = set()
    connections: set[asyncio.Task[None]] = set()
    closing = False

    async def relay(process: RunnerInteractiveProcess, writer: asyncio.StreamWriter) -> None:
        while data := await process.read():
            writer.write(_shim_frame(b"o", data))
            await writer.drain()
        writer.write(_shim_frame(b"x", struct.pack(">i", await process.wait())))
        await writer.drain()

    async def forward(process: RunnerInteractiveProcess, reader: asyncio.StreamReader) -> None:
        while True:
            kind, payload = await _read_shim_frame(reader)
            if kind not in {b"i", b"e"}:
                raise ValueError("Unexpected external CLI stand-in frame.")
            try:
                if kind == b"i":
                    await process.write(payload)
                else:
                    await process.close_stdin()
                    return
            except (BrokenPipeError, ConnectionResetError):
                # Closed guest stdin does not settle its independent output
                # channel. Relay that channel through its real exit status.
                return

    async def connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        connections.add(task)
        process = None
        try:
            if closing or len(connections) > max_processes:
                return
            async with asyncio.timeout(_SHIM_HELLO_TIMEOUT_SECONDS):
                kind, payload = await _read_shim_frame(reader)
            hello = json.loads(payload) if kind == b"h" else None
            if (
                not isinstance(hello, dict)
                or type(hello.get("token")) is not str
                or not hmac.compare_digest(hello["token"], token)
                or type(hello.get("argv")) is not list
                or any(type(argument) is not str for argument in hello["argv"])
            ):
                return
            process = await capability.open_interactive_process(
                ExecCommand.process(
                    "/bin/sh",
                    "-c",
                    _GUEST_ENTRY,
                    "cayu-agent",
                    guest_home,
                    directory,
                    workdir,
                    cli,
                    *hello["argv"],
                ),
                cwd=cwd,
                env=guest_environment,
                timeout_s=timeout_s,
            )
            processes.add(process)
            for secret in process.secret_values:
                if not secret:
                    continue
                if protect_secret is None:
                    # Static launch secrets are already declared; newly issued
                    # credentials require a live publication registry.
                    if secret not in declared_secrets:
                        raise RuntimeError("Interactive process secrets require a protector.")
                else:
                    protect_secret(secret)
            output = asyncio.create_task(relay(process, writer))
            source = asyncio.create_task(forward(process, reader))
            try:
                done, _ = await asyncio.wait((output, source), return_when=asyncio.FIRST_COMPLETED)
                if source in done and source.exception() is None:
                    # Guest stdin may close before its final output and exit
                    # status. Keep the independent output channel alive.
                    await output
            finally:
                for pending in (output, source):
                    pending.cancel()
                await asyncio.gather(output, source, return_exceptions=True)
        except BaseException as error:
            if process is None:
                process = capability.failed_start_process(error)
                if process is not None:
                    processes.add(process)
            # Keep the original cancellation while retaining any failed start's
            # cleanup handle. Other failures make the stand-in exit non-zero.
            if not isinstance(error, Exception):
                raise
        finally:
            connections.discard(task)
            writer.close()
            if process is not None and not closing:
                # An unconfirmed stop keeps the process for the launch's close.
                with suppress(Exception):
                    await join_cleanup(process.close())
                    processes.discard(process)

    async def close() -> None:
        nonlocal closing
        closing = True
        server.close()
        pending = tuple(connections)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await server.wait_closed()
        failures = []
        for process in tuple(processes):
            try:
                await process.close()
            except Exception as error:
                failures.append(error)
            else:
                processes.discard(process)
        if failures:
            raise ExceptionGroup("External guest processes were not confirmed stopped.", failures)

    # A short path under /tmp fits a Unix socket address and can be created in
    # the guest too.
    with TemporaryDirectory(
        prefix="cayu-agent-", dir="/tmp" if os.path.isdir("/tmp") else None
    ) as directory:
        workdir = os.path.join(directory, "workspace")
        os.mkdir(workdir)
        socket_path = os.path.join(directory, "s")
        server = await asyncio.start_unix_server(connect, path=socket_path)
        try:
            os.chmod(socket_path, 0o600)
            launcher = Path(directory) / "cli"
            launcher.write_text(
                f"#!{sys.executable} -I\n"
                + _SHIM_SOURCE.format(socket_path=socket_path, token=token),
                encoding="utf-8",
            )
            launcher.chmod(0o700)
            yield ExternalAgentLaunch(
                str(launcher),
                workdir,
                close,
                {"PATH": "/usr/bin:/bin", "HOME": directory},
                declared_secrets,
                mcp_servers=dict(mcp_servers or {}),
                codex_config_overrides=tuple(codex_config_overrides),
                bind_secret_protector=bind_secret_protector,
            )
        finally:
            from cayu.external_agents._cleanup import close_external_guest

            await close_external_guest(close)


@asynccontextmanager
async def runner_port_forward(
    runner: Runner, *, host: str, port: int, timeout_s: int | None = None
) -> AsyncIterator[RunnerPortForward]:
    """Give guest processes an address that reaches ``host``:``port`` on this host.

    It makes an ``ExternalAgentGateway`` listening only on this host's loopback
    reachable from a CLI in the runner: grant with
    ``public_base_url=forward.guest_url`` and hand the capability's URLs and
    token to the CLI. The runner is borrowed and needs a
    ``Runner.port_forward_capability()``. Leaving the block closes the guest
    listener and every relayed connection; ``timeout_s`` bounds the listener if
    this host disappears first.
    """
    capability = runner.port_forward_capability()
    if capability is None:
        raise ValueError(
            "The runner cannot forward guest ports; use a runner that supports them, "
            "such as LambdaMicroVMRunner."
        )
    from cayu.external_agents._cleanup import close_external_guest

    try:
        forward = await capability.open_port_forward(host=host, port=port, timeout_s=timeout_s)
    except BaseException as error:
        retained = capability.failed_open_forward(error)
        if retained is not None:
            await close_external_guest(retained.close)
        raise
    try:
        yield forward
    finally:
        await close_external_guest(forward.close)


# Applied to every Codex run, with or without a model proxy.
CODEX_RUN_OVERRIDES = ("project_doc_max_bytes=0", 'web_search="disabled"')


def codex_proxy_overrides(base_url: str) -> tuple[str, ...]:
    """A credential-free custom provider, backed by an authenticating service proxy."""
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("Codex requires a service model proxy URL.")
    return (
        'model_provider="cayu"',
        'model_providers.cayu.name="Cayu model proxy"',
        f"model_providers.cayu.base_url={json.dumps(base_url)}",
        'model_providers.cayu.wire_api="responses"',
        "model_providers.cayu.requires_openai_auth=false",
        'model_providers.cayu.env_key="CAYU_MODEL_PROXY_TOKEN"',
        *CODEX_RUN_OVERRIDES,
    )
