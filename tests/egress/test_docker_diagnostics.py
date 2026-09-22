from __future__ import annotations

import asyncio
import json

import pytest

from cayu.egress._docker_diagnostics import docker_setup_failure
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.egress.errors import UnsupportedEgressError


@pytest.mark.parametrize(
    "stderr,expected",
    [
        ("", "[unavailable]"),
        (" \n\t", "[unavailable]"),
        ("opaque-secret-mount-contents", "[REDACTED]"),
        ("permission denied " * 500, "...[truncated]"),
        ("secret" * 20000, "...[truncated]"),
    ],
)
def test_stderr_is_bounded_and_explicit(stderr, expected):
    message = docker_setup_failure(["network", "create", "private-name"], 17, stderr)
    assert expected in message
    assert "exit_code=17" in message
    assert len(message.split("stderr: ", 1)[1].encode()) <= 1024
    assert "private-name" not in message
    assert "secret" not in message


def test_untrusted_stderr_and_argv_never_escape():
    secrets = [
        "password-canary",
        "token-canary",
        "environment-canary",
        "mount-canary",
        "username-canary",
        "path-canary",
        "opaque-canary",
        "argument-canary",
    ]
    stderr = (
        "Error response from daemon: permission denied\n"
        "https://username-canary:password-canary@example.com?token=token-canary\n"
        "Authorization: Bearer token-canary\n"
        "ENV=environment-canary /path-canary\n"
        "-----BEGIN PRIVATE KEY-----\nmount-canary\n-----END PRIVATE KEY-----\n"
        "opaque-canary argument-canary\x1b[31m\x00"
    )
    message = docker_setup_failure(["exec", "argument-canary", "cat", "/path-canary"], 1, stderr)
    assert "permission denied" in message
    assert "[REDACTED]" in message
    assert all(value not in message for value in secrets)
    assert "\x1b" not in message and "\x00" not in message and "\n" not in message
    assert "argument-canary" not in docker_setup_failure(["argument-canary"], 1, "")
    assert "argument-canary" not in docker_setup_failure(["network", "argument-canary"], 1, "")


@pytest.mark.parametrize(
    "operation,stderr",
    [
        (
            "create",
            "Error response from daemon: all predefined address pools have been fully subnetted",
        ),
        (
            "connect",
            "Error response from daemon: endpoint with name private-name already exists in network private-network",
        ),
    ],
)
def test_setup_failure_survives_sqlite_environment_and_session_evidence(
    tmp_path, monkeypatch, operation, stderr
):
    from cayu.agents import AgentSpec
    from cayu.applications import CayuApp
    from cayu.egress.policy import PublicWebEgressPolicy
    from cayu.egress.runtime import VirtualEgressEnvironmentFactory
    from cayu.environments import EnvironmentSpec
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.events import EventType
    from cayu.messages import Message
    from cayu.runtime import RunRequest
    from cayu.storage.sqlite import SQLiteSessionStore

    calls = []

    async def execute(argv):
        calls.append(list(argv))
        if list(argv[:2]) == ["network", operation]:
            return 1, stderr + " AUTH_TOKEN=secret-canary"
        return 0, ""

    async def start(_self):
        return 8123

    monkeypatch.setattr("cayu.egress.docker_adapter.TransparentEgressProxyServer.start", start)

    async def scenario():
        path = tmp_path / "sessions.db"
        store = SQLiteSessionStore(path)
        adapter = DockerEgressAdapter(docker_exec=execute, proxy_host="127.0.0.1")
        factory = VirtualEgressEnvironmentFactory(
            policies={"web": PublicWebEgressPolicy(name="web")},
            public_web_policy="web",
            adapter=adapter,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([]), default=True)
        app.register_environment_factory(EnvironmentSpec(name="docker"), factory, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake"))
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="docker-failure",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        await store.close()
        reopened = SQLiteSessionStore(path)
        persisted = await reopened.load_events("docker-failure")
        await reopened.close()
        for kind in (EventType.ENVIRONMENT_FACTORY_FAILED, EventType.SESSION_FAILED):
            live = next(event for event in events if event.type == kind)
            durable = next(event for event in persisted if event.type == kind)
            assert live.payload["error"] == durable.payload["error"]
            payload = json.dumps(durable.payload)
            assert f"docker network {operation}" in payload
            assert "exit_code=1" in payload
            assert (
                "fully subnetted" if operation == "create" else "already exists in network"
            ) in payload
            assert "secret-canary" not in payload
            assert "private-name" not in payload
        assert any(argv[:2] == ["network", "rm"] for argv in calls)
        assert any(argv[0] == "rm" for argv in calls)
        assert not adapter._preparation_cleanups

    asyncio.run(scenario())


def test_setup_exception_contains_safe_diagnostic():
    async def execute(_argv):
        return 9, "Pool overlaps with other one on this address space; TOKEN=secret"

    adapter = DockerEgressAdapter(docker_exec=execute)
    with pytest.raises(UnsupportedEgressError, match="network create.*exit_code=9") as raised:
        asyncio.run(adapter._run(["network", "create", "private-network"]))
    assert "Pool overlaps" in str(raised.value)
    assert "secret" not in str(raised.value)
