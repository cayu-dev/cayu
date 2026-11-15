"""`cayu dev postgres up|url|down` against a fake Docker.

The fake keeps containers in a dict and answers the few `docker` commands the part runs, so
these tests need no Docker.
"""

from __future__ import annotations

import json
import socket
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from cayu.cli import dev, dev_postgres, main, scaffold_ci
from cayu.cli.dev_postgres import (
    POSTGRES_IMAGE,
    PROJECT_LABEL,
    ContainerNameTakenError,
    DevPostgresError,
    DockerUnavailableError,
    PortInUseError,
)


class FakeDocker:
    def __init__(self, *, daemon: bool = True, ready_after: int = 0) -> None:
        self.daemon = daemon
        self.ready_after = ready_after
        self.containers: dict[str, dict] = {}
        self.calls: list[list[str]] = []
        self.run_error = ""
        self.next_port = 49153
        self.image = ""

    def __call__(
        self, arguments: Sequence[str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        arguments = list(arguments)
        self.calls.append(arguments)
        command = arguments[0]
        if command == "version":
            if not self.daemon:
                return _result(1, stderr="Cannot connect to the Docker daemon. Is it running?")
            return _result(0, "27.0.0\n")
        if command == "container" and arguments[1] == "inspect":
            container = self.containers.get(arguments[2])
            if container is None:
                return _result(1, stderr=f"Error: No such container: {arguments[2]}")
            return _result(0, json.dumps([container]))
        if command == "run":
            if self.run_error:
                return _result(125, stderr=self.run_error)
            return self._run(arguments)
        if command == "exec":
            self.ready_after -= 1
            return _result(0 if self.ready_after < 0 else 2)
        if command == "logs":
            return _result(0, "database system is starting up\n")
        if command == "rm":
            if self.containers.pop(arguments[-1], None) is None:
                return _result(1, stderr=f"Error: No such container: {arguments[-1]}")
            return _result(0, arguments[-1])
        raise AssertionError(f"unexpected docker call: {arguments}")

    def _run(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        name = arguments[arguments.index("--name") + 1]
        label_key, label_value = arguments[arguments.index("--label") + 1].split("=", 1)
        published = arguments[arguments.index("--publish") + 1]
        host_port = published.split(":")[1] or str(self.next_port)
        self.containers[name] = _container(
            labels={label_key: label_value}, running=True, port=host_port
        )
        self.image = arguments[-1]
        return _result(0, "c0ffee\n")


def _container(*, labels: dict[str, str], running: bool, port: str | None) -> dict:
    ports = {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": port}] if port else None}
    return {
        "Config": {"Labels": labels},
        "State": {"Running": running},
        "NetworkSettings": {"Ports": ports},
    }


def _result(code: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["docker"], code, stdout, stderr)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "Acme Desk"
    (root / "tests").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'acme-desk'\n", encoding="utf-8")
    monkeypatch.chdir(root / "tests")
    return root.resolve()


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    fake = FakeDocker()
    monkeypatch.setattr(dev_postgres, "run_docker", fake)
    return fake


def test_the_generated_ci_and_the_local_server_use_one_pinned_image() -> None:
    assert scaffold_ci.POSTGRES_IMAGE is POSTGRES_IMAGE
    assert "@sha256:" in POSTGRES_IMAGE


def test_each_project_gets_its_own_stable_container_name(tmp_path: Path) -> None:
    first = dev_postgres.container_name(tmp_path / "Acme Desk")
    assert first == dev_postgres.container_name(tmp_path / "Acme Desk")
    assert first.startswith("cayu-postgres-acme-desk-")
    assert first != dev_postgres.container_name(tmp_path / "other" / "Acme Desk")


def test_up_url_down_cycle(
    project: Path, docker: FakeDocker, capsys: pytest.CaptureFixture[str]
) -> None:
    docker.ready_after = 1

    assert main(["dev", "postgres", "up", "--export"]) == 0
    captured = capsys.readouterr()
    url = "postgresql://postgres:postgres@127.0.0.1:49153/postgres"
    assert captured.out == f"export CAYU_TEST_POSTGRES_DSN={url}\n"
    assert "Started PostgreSQL" in captured.err and "cayu dev postgres down" in captured.err
    (name,) = docker.containers
    assert name == dev_postgres.container_name(project)
    assert docker.containers[name]["Config"]["Labels"] == {PROJECT_LABEL: str(project)}
    assert docker.image == POSTGRES_IMAGE
    run = next(call for call in docker.calls if call[0] == "run")
    assert "--rm" in run and "127.0.0.1::5432" in run

    assert main(["dev", "postgres", "url"]) == 0
    assert capsys.readouterr().out == f"{url}\n"
    assert main(["dev", "postgres", "url", "--export", "CAYU_DATABASE_URL"]) == 0
    assert capsys.readouterr().out == f"export CAYU_DATABASE_URL={url}\n"
    assert main(["dev", "postgres", "url", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "container": name,
        "project": str(project),
        "port": 49153,
        "url": url,
    }

    assert main(["dev", "postgres", "down"]) == 0
    assert f"Removed {name}" in capsys.readouterr().err
    assert docker.containers == {}
    assert ["rm", "--force", "--volumes", name] in docker.calls


def test_up_when_already_running_reuses_the_server(
    project: Path, docker: FakeDocker, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["dev", "postgres", "up"]) == 0
    capsys.readouterr()

    assert main(["dev", "postgres", "up", "--port", "5432", "--json"]) == 0
    captured = capsys.readouterr()
    assert "already running" in captured.err and "not on --port 5432" in captured.err
    assert json.loads(captured.out)["started"] is False
    assert [call[0] for call in docker.calls].count("run") == 1


def test_up_replaces_a_stopped_container_of_its_own(project: Path, docker: FakeDocker) -> None:
    name = dev_postgres.container_name(project)
    docker.containers[name] = _container(
        labels={PROJECT_LABEL: str(project)}, running=False, port=None
    )

    server, started = dev_postgres.start(project)

    assert started and server.port == 49153
    assert ["rm", "--force", "--volumes", name] in docker.calls


def test_url_and_down_without_a_server(
    project: Path, docker: FakeDocker, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["dev", "postgres", "url"]) == 3
    captured = capsys.readouterr()
    assert captured.out == "" and "cayu dev postgres up" in captured.err

    assert main(["dev", "postgres", "down"]) == 0
    assert "nothing to remove" in capsys.readouterr().err
    assert not any(call[0] == "rm" for call in docker.calls)


def test_down_leaves_other_projects_containers_alone(
    project: Path, docker: FakeDocker, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    dev_postgres.start(other.resolve())
    name = dev_postgres.container_name(project)
    docker.containers[name] = _container(labels={}, running=True, port="5432")

    assert main(["dev", "postgres", "down"]) == 1
    assert "didn't start it" in capsys.readouterr().err
    assert len(docker.containers) == 2
    assert not any(call[0] == "rm" for call in docker.calls)
    with pytest.raises(ContainerNameTakenError):
        dev_postgres.start(project)


def test_without_docker_it_says_so(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(subprocess, "run", missing)

    assert main(["dev", "postgres", "up"]) == 1
    error = capsys.readouterr().err
    assert "Docker isn't installed" in error and "CAYU_TEST_POSTGRES_DSN" in error


def test_with_the_daemon_down_it_says_so(
    project: Path, docker: FakeDocker, capsys: pytest.CaptureFixture[str]
) -> None:
    docker.daemon = False

    for command in ("up", "url", "down"):
        assert main(["dev", "postgres", command]) == 1
        error = capsys.readouterr().err
        assert "daemon can't be reached" in error and "Is it running?" in error
    assert [call[0] for call in docker.calls] == ["version"] * 3


def test_a_port_in_use_is_refused_before_docker_runs(
    project: Path, docker: FakeDocker, capsys: pytest.CaptureFixture[str]
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        assert main(["dev", "postgres", "up", "--port", str(port)]) == 1

    error = capsys.readouterr().err
    assert f"Port {port} on 127.0.0.1 is already in use" in error
    assert not any(call[0] == "run" for call in docker.calls)


def test_a_port_docker_finds_taken_is_reported_as_in_use(project: Path, docker: FakeDocker) -> None:
    docker.run_error = (
        "docker: Error response from daemon: driver failed programming external connectivity: "
        "Bind for 127.0.0.1:5432 failed: port is already allocated."
    )

    with pytest.raises(PortInUseError, match="already in use"):
        dev_postgres.start(project)


def test_a_failed_first_run_reports_the_error_after_the_pull_progress(
    project: Path, docker: FakeDocker
) -> None:
    pull = (
        f"Unable to find image '{POSTGRES_IMAGE}' locally\n"
        "17-alpine: Pulling from library/postgres\n"
        "abc: Pulling fs layer\n"
        "def: Pull complete\n"
    )
    docker.run_error = pull + "docker: Error response from daemon: port is already allocated."
    with pytest.raises(PortInUseError):
        dev_postgres.start(project)

    docker.run_error = pull + "docker: Error response from daemon: Conflict. The name is in use."
    with pytest.raises(DevPostgresError, match="Conflict. The name is in use."):
        dev_postgres.start(project)


def test_up_waits_for_a_running_server_another_up_is_still_starting(
    project: Path, docker: FakeDocker
) -> None:
    name = dev_postgres.container_name(project)
    docker.containers[name] = _container(
        labels={PROJECT_LABEL: str(project)}, running=True, port="49153"
    )
    docker.ready_after = 1

    server, started = dev_postgres.start(project, poll_interval=0.0)

    assert not started and server.port == 49153
    assert [call[0] for call in docker.calls].count("exec") == 2
    assert not any(call[0] in ("run", "rm") for call in docker.calls)


def test_a_server_that_never_gets_ready_is_removed(project: Path, docker: FakeDocker) -> None:
    docker.ready_after = 10**6

    with pytest.raises(DevPostgresError, match="didn't accept connections") as raised:
        dev_postgres.start(project, timeout=0.0, poll_interval=0.0)

    assert "database system is starting up" in str(raised.value)
    assert "was removed" in str(raised.value)
    assert docker.containers == {}


def test_docker_unavailable_is_a_dev_postgres_error() -> None:
    assert issubclass(DockerUnavailableError, DevPostgresError)


def test_the_project_is_the_nearest_folder_with_a_pyproject(project: Path) -> None:
    assert dev_postgres.project_root() == project
    assert dev_postgres.project_root(project / "tests") == project


def test_arguments_are_checked(capsys: pytest.CaptureFixture[str]) -> None:
    for arguments in (["up", "--port", "70000"], ["url", "--export", "not a name"]):
        with pytest.raises(SystemExit) as raised:
            main(["dev", "postgres", *arguments])
        assert raised.value.code == 2
    capsys.readouterr()
    assert dev._variable("CAYU_DATABASE_URL") == "CAYU_DATABASE_URL"


@pytest.mark.parametrize(
    ("preset", "capabilities"),
    [("b2b", ()), ("service", ("app-database",))],
)
def test_generated_guidance_starts_the_server_with_one_cayu_command(
    preset: str, capabilities: tuple[str, ...]
) -> None:
    from cayu.cli.scaffold import project_files

    files = project_files("acme", preset=preset, with_capabilities=capabilities)
    for guidance in ("README.md", "AGENTS.md"):
        text = files[guidance]
        assert 'eval "$(uv run --no-sync cayu dev postgres up --export)"' in text, guidance
        assert "uv run --no-sync cayu dev postgres down" in text, guidance
        assert "docker run" not in text, guidance
