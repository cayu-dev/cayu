"""Timeout settlement of the real-browser regression harness, without Docker."""

from __future__ import annotations

import subprocess

import pytest
from tests.egress import _browser_visual_container as harness


@pytest.mark.parametrize("failure", [None, "timeout", "interrupt"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_visual_container_always_removes_its_exact_owner(monkeypatch, failure, cleanup_fails):
    calls = []
    primary = (
        subprocess.TimeoutExpired("docker", 45) if failure == "timeout" else KeyboardInterrupt()
    )
    cleanup = subprocess.CalledProcessError(1, "docker rm")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "run" and failure:
            raise primary
        if command[1] == "rm" and cleanup_fails:
            raise cleanup
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(harness.subprocess, "run", run)
    if failure or cleanup_fails:
        with pytest.raises(type(primary) if failure else type(cleanup)) as caught:
            harness.run_visual_container(["fixture-image", "test-command"])
        assert caught.value is (primary if failure else cleanup)
        if failure and cleanup_fails:
            assert caught.value.__cause__ is cleanup
    else:
        harness.run_visual_container(["fixture-image", "test-command"])
    assert len(calls) == 2
    name = calls[0][0][3]
    assert name.startswith("cayu-visual-guard-")
    assert calls[1][0] == ["docker", "rm", "--force", name]
    assert calls[1][1]["check"] is True and calls[1][1]["timeout"] == 15


def test_container_cleanup_interrupt_remains_authoritative(monkeypatch):
    primary = subprocess.TimeoutExpired("docker", 45)
    interruption = KeyboardInterrupt()

    def run(command, **kwargs):
        if command[1] == "run":
            raise primary
        raise interruption

    monkeypatch.setattr(harness.subprocess, "run", run)
    with pytest.raises(KeyboardInterrupt) as caught:
        harness.run_visual_container(["fixture-image"])
    assert caught.value is interruption and caught.value.__cause__ is primary
