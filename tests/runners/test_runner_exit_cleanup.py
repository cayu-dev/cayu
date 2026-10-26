"""Interpreter-exit backstop for sandbox removals that never settled."""

from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

import pytest
from tests.exit_cleanup_isolation import isolated_interpreter_exit_cleanup

from cayu.runners import _exit_cleanup
from cayu.runners.docker import DockerRunner

_CONTAINER = "c" * 64


@pytest.fixture
def unsettled(monkeypatch):
    registry: dict = {}
    monkeypatch.setattr(_exit_cleanup, "_unsettled", registry)
    return registry


@pytest.fixture
def docker_rm(monkeypatch):
    calls: list[list[str]] = []
    result = {"returncode": 0}

    def run(argv, **kwargs):
        calls.append(list(argv))
        assert kwargs["timeout"] > 0 and kwargs["check"] is False
        return SimpleNamespace(returncode=result["returncode"])

    monkeypatch.setattr("cayu.runners.docker.subprocess.run", run)
    return SimpleNamespace(calls=calls, result=result)


def _runner(close_action="remove") -> DockerRunner:
    return DockerRunner(
        "owned",
        docker_path="/usr/bin/docker",
        close_action=close_action,
        cancel_timeout_s=0.05,
        _container_id=_CONTAINER,
    )


def test_close_that_missed_its_deadline_is_finished_at_exit(monkeypatch, unsettled, docker_rm):
    release = asyncio.Event()

    async def slow_docker(path, args, **kwargs):
        await release.wait()
        raise AssertionError("the stalled removal never completes")

    monkeypatch.setattr("cayu.runners.docker._run_docker", slow_docker)
    runner = _runner()

    async def close_until_loop_stops():
        with pytest.raises(TimeoutError):
            await runner.close()
        # asyncio.run now cancels the still-running removal, as at process exit.

    asyncio.run(close_until_loop_stops())
    assert not runner.is_closed and not runner.container_removed
    assert list(unsettled) == [id(runner)]

    _exit_cleanup._finish_unsettled_at_exit()

    assert docker_rm.calls == [["/usr/bin/docker", "rm", "-f", _CONTAINER]]
    assert runner.container_removed
    assert unsettled == {}


def test_settled_close_is_forgotten(monkeypatch, unsettled, docker_rm):
    async def docker(path, args, **kwargs):
        from cayu.runners.base import ExecResult

        assert args == ["rm", "-f", _CONTAINER]
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker._run_docker", docker)
    runner = _runner()
    asyncio.run(runner.close())

    assert runner.is_closed and unsettled == {}
    _exit_cleanup._finish_unsettled_at_exit()
    assert docker_rm.calls == []


def test_forked_child_never_finishes_parent_removals(monkeypatch, unsettled, docker_rm):
    runner = _runner()
    _exit_cleanup.track_unsettled_terminal_lifecycle(runner, "remove")
    monkeypatch.setattr(_exit_cleanup, "_owner_pid", -1)

    _exit_cleanup._finish_unsettled_at_exit()

    assert docker_rm.calls == []


def test_dropped_runner_still_owes_its_removal(unsettled, docker_rm):
    runner = _runner()
    _exit_cleanup.track_unsettled_terminal_lifecycle(runner, "remove")
    del runner

    _exit_cleanup._finish_unsettled_at_exit()

    assert docker_rm.calls == [["/usr/bin/docker", "rm", "-f", _CONTAINER]]


def test_one_removal_per_daemon(unsettled, docker_rm):
    first = _runner()
    second = DockerRunner(
        "second", docker_path="/usr/bin/docker", close_action="remove", _container_id="d" * 64
    )
    other_daemon = DockerRunner(
        "other",
        docker_path="/opt/docker",
        close_action="remove",
        _container_id="e" * 64,
    )
    for runner in (first, second, other_daemon):
        _exit_cleanup.track_unsettled_terminal_lifecycle(runner, "remove")

    _exit_cleanup._finish_unsettled_at_exit()

    assert sorted(docker_rm.calls) == [
        ["/opt/docker", "rm", "-f", "e" * 64],
        ["/usr/bin/docker", "rm", "-f", _CONTAINER, "d" * 64],
    ]


@pytest.mark.parametrize(
    ("close_action", "action", "removes"),
    [
        ("remove", None, True),
        ("remove", "remove", True),
        ("none", "remove", True),  # an unsettled kill()
        ("stop", None, False),
        ("none", None, False),
        ("remove", "stop", False),
    ],
)
def test_docker_exit_close_only_removes(docker_rm, close_action, action, removes):
    runner = _runner(close_action)

    assert _exit_cleanup.close_runners_at_exit([(runner, action)]) == int(removes)
    assert docker_rm.calls == ([["/usr/bin/docker", "rm", "-f", _CONTAINER]] if removes else [])
    assert runner.container_removed is removes


def test_docker_exit_close_reports_failures_without_raising(monkeypatch, docker_rm):
    runner = _runner()
    docker_rm.result["returncode"] = 1
    assert _exit_cleanup.close_runners_at_exit([(runner, None)]) == 0
    assert not runner.container_removed

    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("cayu.runners.docker.subprocess.run", hang)
    assert _exit_cleanup.close_runners_at_exit([(runner, None)]) == 0


def test_docker_exit_close_never_targets_a_name(docker_rm):
    runner = DockerRunner("legacy-name", docker_path="/usr/bin/docker", close_action="remove")

    assert _exit_cleanup.close_runners_at_exit([(runner, None)]) == 0
    assert docker_rm.calls == []


def _launched(runner):
    from cayu.runners import LaunchedRunner

    return LaunchedRunner(runner, ["sudo", "-u", "sandbox"])


def test_launched_runner_close_that_missed_its_deadline_is_finished_at_exit(
    monkeypatch, unsettled, docker_rm
):
    release = asyncio.Event()

    async def slow_docker(path, args, **kwargs):
        await release.wait()
        raise AssertionError("the stalled removal never completes")

    monkeypatch.setattr("cayu.runners.docker._run_docker", slow_docker)
    inner = _runner()
    launched = _launched(inner)

    async def close_until_loop_stops():
        with pytest.raises(TimeoutError):
            await launched.close()

    asyncio.run(close_until_loop_stops())
    assert not launched.is_closed

    _exit_cleanup._finish_unsettled_at_exit()

    assert docker_rm.calls == [["/usr/bin/docker", "rm", "-f", _CONTAINER]]
    assert inner.container_removed


@pytest.mark.parametrize(("close_action", "removes"), [("remove", True), ("stop", False)])
def test_launched_runner_held_at_exit_delegates_to_the_wrapped_runner(
    docker_rm, close_action, removes
):
    inner = _runner(close_action)

    assert _exit_cleanup.close_runners_at_exit([(_launched(inner), None)]) == int(removes)
    assert docker_rm.calls == ([["/usr/bin/docker", "rm", "-f", _CONTAINER]] if removes else [])
    assert inner.container_removed is removes


def test_runners_a_test_leaves_registered_never_reach_the_real_exit_hook(request):
    # The root conftest wraps every test in this isolation, so interpreter exit
    # never runs `docker rm -f` against a fake container a test left behind.
    assert "_isolated_interpreter_exit_cleanup" in request.fixturenames
    runner = _runner()

    def held():
        return [runner]

    with isolated_interpreter_exit_cleanup():
        _exit_cleanup.track_unsettled_terminal_lifecycle(runner, "remove")
        _exit_cleanup._holders[id(held)] = held
        assert id(runner) in _exit_cleanup._unsettled

    assert id(runner) not in _exit_cleanup._unsettled
    assert id(held) not in _exit_cleanup._holders
