"""A throwaway PostgreSQL server in Docker for one project's tests.

`cayu dev postgres up|url|down` is the command line for these functions; other code can call
them directly. The server runs the same image, pinned to the same digest, as the CI workflow
`cayu new` writes, with the same password, so a test that passes here passes there.

Each project gets its own container, named after the project's folder and a hash of its path
and labeled with that path, so two projects never share a server and `stop` removes only the
container `start` made for the same project. Nothing is kept: the container has no named
volume and is removed with its data when it stops.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

# The variable the generated tests read for a server where they may create databases.
from cayu.testing.postgres import TEST_SERVER_VARIABLE

# Pinned to a digest, as the generated CI's actions are to commits. The generated CI's
# PostgreSQL service reads this constant too, so the two never drift apart.
POSTGRES_IMAGE = (
    "postgres:17-alpine@sha256:d3244e8ce3f165ee3c93d9c4a4b1d0fe48b3a89e2287b717fdcbbde92d405065"
)
POSTGRES_PASSWORD = "postgres"
PROJECT_LABEL = "dev.cayu.postgres.project"
_CONTAINER_PORT = "5432/tcp"
_HOST = "127.0.0.1"
_NAME_PREFIX = "cayu-postgres-"

DockerRunner = Callable[[Sequence[str], float], "subprocess.CompletedProcess[str]"]


class DevPostgresError(RuntimeError):
    """A throwaway PostgreSQL server could not be started, found or removed."""


class DockerUnavailableError(DevPostgresError):
    """Docker isn't installed, or its daemon can't be reached."""


class PortInUseError(DevPostgresError):
    """The host port asked for is already taken."""


class ContainerNameTakenError(DevPostgresError):
    """A container that `start` didn't make for this project has the project's name."""


@dataclass(frozen=True, slots=True)
class DevPostgres:
    """One project's running throwaway PostgreSQL server."""

    container: str
    project: Path
    port: int

    @property
    def url(self) -> str:
        """The connection URL for the server's `postgres` database."""

        return f"postgresql://postgres:{POSTGRES_PASSWORD}@{_HOST}:{self.port}/postgres"


def project_root(start: Path | None = None) -> Path:
    """The nearest folder at or above `start` (default: the working directory) with a
    `pyproject.toml`, or `start` itself when there is none."""

    directory = (Path.cwd() if start is None else start).expanduser().resolve()
    for candidate in (directory, *directory.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return directory


def container_name(project: Path) -> str:
    """The Docker container name for `project`'s server, stable for the project's path."""

    resolved = project.expanduser().resolve()
    slug = re.sub(r"[^a-z0-9_.-]+", "-", resolved.name.lower()).strip("-._")[:40] or "project"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:10]
    return f"{_NAME_PREFIX}{slug}-{digest}"


def run_docker(arguments: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run one `docker` command and return what it printed, without raising on failure."""

    try:
        return subprocess.run(
            ["docker", *arguments],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise DockerUnavailableError(
            "Docker isn't installed, or `docker` isn't on PATH. A throwaway PostgreSQL needs "
            f"Docker; without it, set {TEST_SERVER_VARIABLE} to a PostgreSQL server of your "
            "own where the tests may create databases."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DevPostgresError(
            f"`docker {arguments[0]}` didn't finish within {timeout:g} seconds."
        ) from exc


def find(project: Path, *, docker: DockerRunner | None = None) -> DevPostgres | None:
    """`project`'s running server, or None when there is none."""

    docker = run_docker if docker is None else docker
    _require_docker(docker)
    state = _inspect(container_name(project), project, docker)
    if state is None or not state.running:
        return None
    return _running(state, project)


def start(
    project: Path,
    *,
    port: int | None = None,
    timeout: float = 60.0,
    docker: DockerRunner | None = None,
    poll_interval: float = 0.5,
) -> tuple[DevPostgres, bool]:
    """Start `project`'s server unless it is running, and wait until it accepts connections.

    Returns the server and whether this call started it. `port` is the host port on
    127.0.0.1; without it Docker picks a free one. Raises `PortInUseError` when `port` is taken
    and `DockerUnavailableError` when Docker can't be reached.
    """

    docker = run_docker if docker is None else docker
    _require_docker(docker)
    name = container_name(project)
    state = _inspect(name, project, docker)
    if state is not None and state.running:
        # Another `up` may have started it moments ago and still be waiting for it.
        return _wait_until_ready(name, project, timeout, docker, poll_interval), False
    if state is not None:
        # Stopped but not removed: nothing in it is worth keeping.
        _remove(name, docker)
    if port is not None:
        _require_free_port(port)
    published = f"{_HOST}:{'' if port is None else port}:{_CONTAINER_PORT.split('/')[0]}"
    result = docker(
        [
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--label",
            f"{PROJECT_LABEL}={project.expanduser().resolve()}",
            "--env",
            f"POSTGRES_PASSWORD={POSTGRES_PASSWORD}",
            "--publish",
            published,
            POSTGRES_IMAGE,
        ],
        # The first run pulls the image.
        600.0,
    )
    if result.returncode != 0:
        if any(
            text in result.stderr
            for text in ("port is already allocated", "address already in use")
        ):
            raise PortInUseError(_port_in_use(port))
        # A first run's stderr starts with the image pull's progress; the error comes last.
        raise DevPostgresError(f"Docker couldn't start PostgreSQL: {_last_lines(result.stderr)}")
    try:
        server = _wait_until_ready(name, project, timeout, docker, poll_interval)
    except DevPostgresError as exc:
        _remove(name, docker)
        raise DevPostgresError(f"{exc}\n{name} was removed.") from exc
    except BaseException:
        _remove(name, docker)
        raise
    return server, True


def stop(project: Path, *, docker: DockerRunner | None = None) -> str | None:
    """Remove `project`'s server and its data. Returns the container removed, or None when
    there was none."""

    docker = run_docker if docker is None else docker
    _require_docker(docker)
    name = container_name(project)
    if _inspect(name, project, docker) is None:
        return None
    _remove(name, docker)
    return name


@dataclass(frozen=True, slots=True)
class _State:
    name: str
    running: bool
    port: int | None


def _require_docker(docker: DockerRunner) -> None:
    result = docker(["version", "--format", "{{.Server.Version}}"], 30.0)
    if result.returncode != 0:
        raise DockerUnavailableError(
            "Docker is installed but its daemon can't be reached; start Docker and try "
            f"again. Docker said: {_first_lines(result.stderr)}"
        )


def _inspect(name: str, project: Path, docker: DockerRunner) -> _State | None:
    result = docker(["container", "inspect", name], 30.0)
    if result.returncode != 0:
        if "no such" in result.stderr.lower():
            return None
        raise DevPostgresError(f"Docker couldn't inspect {name}: {_first_lines(result.stderr)}")
    try:
        (record,) = json.loads(result.stdout)
        labels = record["Config"].get("Labels") or {}
        running = bool(record["State"]["Running"])
        bindings = (record["NetworkSettings"].get("Ports") or {}).get(_CONTAINER_PORT) or []
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise DevPostgresError(f"Docker described {name} in a form Cayu can't read.") from exc
    expected = str(project.expanduser().resolve())
    if labels.get(PROJECT_LABEL) != expected:
        raise ContainerNameTakenError(
            f"A container named {name} exists, but cayu dev postgres didn't start it for "
            f"{expected}, so it is left alone. Remove or rename it with docker if it is yours."
        )
    port = next(
        (int(binding["HostPort"]) for binding in bindings if binding.get("HostPort")),
        None,
    )
    return _State(name=name, running=running, port=port)


def _running(state: _State, project: Path) -> DevPostgres:
    if state.port is None:
        raise DevPostgresError(f"{state.name} is running but publishes no port for PostgreSQL.")
    return DevPostgres(
        container=state.name, project=project.expanduser().resolve(), port=state.port
    )


def _wait_until_ready(
    name: str, project: Path, timeout: float, docker: DockerRunner, poll_interval: float
) -> DevPostgres:
    deadline = time.monotonic() + timeout
    while True:
        state = _inspect(name, project, docker)
        if state is None or not state.running:
            raise DevPostgresError(f"PostgreSQL stopped while starting in {name}.")
        # Over TCP, so the entrypoint's socket-only setup server doesn't count as ready.
        ready = docker(
            ["exec", name, "pg_isready", "--quiet", "--host", _HOST, "--username", "postgres"],
            30.0,
        )
        if ready.returncode == 0:
            return _running(state, project)
        if time.monotonic() >= deadline:
            logs = docker(["logs", "--tail", "20", name], 30.0)
            raise DevPostgresError(
                f"PostgreSQL in {name} didn't accept connections within {timeout:g} seconds. "
                f"Its last log lines:\n{(logs.stdout + logs.stderr).strip()}"
            )
        time.sleep(poll_interval)


def _remove(name: str, docker: DockerRunner) -> None:
    result = docker(["rm", "--force", "--volumes", name], 60.0)
    if result.returncode != 0 and "no such" not in result.stderr.lower():
        raise DevPostgresError(f"Docker couldn't remove {name}: {_first_lines(result.stderr)}")


def _require_free_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((_HOST, port))
        except OSError as exc:
            raise PortInUseError(_port_in_use(port)) from exc


def _port_in_use(port: int | None) -> str:
    where = "The port Docker picked" if port is None else f"Port {port} on {_HOST}"
    return (
        f"{where} is already in use, perhaps by another PostgreSQL. Leave --port out to let "
        "Docker pick a free one, or pass a different --port."
    )


def _last_lines(text: str, limit: int = 3) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return " ".join(lines[-limit:]) or "no output"


def _first_lines(text: str, limit: int = 3) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return " ".join(lines[:limit]) or "no output"
