"""An unused deferred Docker reservation must not require an eager binding record."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.environments.test_docker_deferred import _factory

from cayu import AgentSpec, CayuApp, Message, RunRequest, SessionStatus
from cayu.environments import EnvironmentSpec
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runners.docker import DockerRunner
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import (
    IncompleteSessionRecoveryRequest,
    ModelCompletionManualRecoveryRequest,
)
from cayu.storage import SQLiteSessionStore


def _identity(name):
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


class _Provider(ModelProvider):
    name = "deferred-recovery"

    def __init__(self, *, crash=False):
        self.crash = crash
        self.calls = 0

    @property
    def execution_profile_identity(self):
        return _identity("tests:deferred-terminal-provider")

    async def stream(self, request):
        self.calls += 1
        if self.crash:
            # The native model stage was dispatched, but no runner was needed.
            os._exit(73)
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _app(root: Path, store, *, crash=False):
    factory = _factory(root / "source", deferred=True)
    provider = _Provider(crash=crash)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="builder", model="test-model"))
    app.register_environment_factory(
        EnvironmentSpec(
            name="sandbox", execution_profile_identity=_identity("tests:deferred-terminal")
        ),
        factory,
        default=True,
    )
    return app, provider


async def _crash(root: Path):
    store = SQLiteSessionStore(root / "sessions.sqlite")
    app, _ = _app(root, store, crash=True)
    async for _ in app.run(
        RunRequest(
            session_id="unused-deferred",
            agent_name="builder",
            messages=[Message.text("user", "answer without tools")],
        )
    ):
        pass


@pytest.mark.parametrize("process_loss", [False, True], ids=["clean-run", "crashed-model-call"])
def test_unused_deferred_allocation_recovers_after_manual_model_settlement(
    tmp_path, monkeypatch, process_loss
):
    (tmp_path / "source").mkdir()
    if process_loss:
        script = (
            "import asyncio,sys; from pathlib import Path; "
            "from tests.core.test_environment_lifecycle_deferred_terminal_recovery import _crash; "
            "asyncio.run(_crash(Path(sys.argv[1])))"
        )
        crashed = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert crashed.returncode == 73, crashed.stderr

    async def absent(name, *, docker_path=None):
        # Only Docker transport is replaced. No container ever materialized.
        return None

    monkeypatch.setattr(DockerRunner, "resolve_container_id", absent)

    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        app, provider = _app(tmp_path, store)
        try:
            if process_loss:
                session = await store.load("unused-deferred")
                assert session.status is SessionStatus.RUNNING
                active = await store.load_active_model_completion_stage(session.id)
                assert active is not None and active.stage.state == "in_flight"
                settled = await app.recover_model_completion_stage(
                    ModelCompletionManualRecoveryRequest(
                        session_id=session.id,
                        stage_id=active.stage.stage_id,
                        expected_run_epoch=session.run_epoch,
                        terminal_status=SessionStatus.INTERRUPTED,
                    )
                )
                assert settled.session.status is SessionStatus.INTERRUPTED
            else:
                events = [
                    event
                    async for event in app.run(
                        RunRequest(
                            session_id="unused-deferred",
                            agent_name="builder",
                            messages=[Message.text("user", "answer without tools")],
                        )
                    )
                ]
                session = await store.load("unused-deferred")
                assert session.status is SessionStatus.COMPLETED
                assert events[-1].type == "session.completed"
            result = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session.id)
            )
            assert result.status is (
                SessionStatus.INTERRUPTED if process_loss else SessionStatus.COMPLETED
            )
            assert provider.calls == (0 if process_loss else 1)
            assert await app.drain_environment_cleanups(timeout_s=2)
        finally:
            await store.close()

    asyncio.run(scenario())
