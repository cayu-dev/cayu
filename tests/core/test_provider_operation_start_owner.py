"""Startup composition retains stream and durable-identity ownership on every exit."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import aclosing
from pathlib import Path

import pytest
from tests.core.test_provider_operations import (
    _ReconnectableProvider,
    _TerminalBlockingStartAdapter,
)

import cayu
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.messages import Message
from cayu.sessions.requests import RunRequest


def test_start_owner_imports_without_execution_controllers() -> None:
    script = """
import importlib
import importlib.abc
import sys

blocked = {
    "cayu.applications",
    "cayu.runtime._model_step_executor",
    "cayu.runtime._session_engine",
    "cayu.runtime._recovery_coordinator",
}

class RejectControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Startup owner imported {fullname}")

sys.meta_path.insert(0, RejectControllers())
importlib.import_module("cayu.runtime._provider_operation_start_owner")
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "boundary",
    [EventType.PROVIDER_OPERATION_STARTING, EventType.PROVIDER_OPERATION_STARTED],
)
def test_closing_run_at_start_boundary_preserves_stream_ownership(boundary: EventType) -> None:
    async def scenario() -> None:
        provider = _ReconnectableProvider(background=True)
        adapter = _TerminalBlockingStartAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = f"close-at-{boundary.value}"

        async with aclosing(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                )
            )
        ) as events:
            async for event in events:
                if event.type is boundary:
                    break
            else:
                raise AssertionError("Startup did not reach the requested event boundary.")

        started = boundary is EventType.PROVIDER_OPERATION_STARTED
        assert adapter.start_calls == int(started)
        assert adapter.events.closed is started
        durable = await app.session_store.load_events(session_id)
        assert sum(e.type is EventType.PROVIDER_OPERATION_STARTED for e in durable) == int(started)
        assert all(e.type is not EventType.MODEL_RETRY for e in durable)

    asyncio.run(scenario())


def test_acknowledgement_failure_hands_durable_identity_and_stream_to_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        provider = _ReconnectableProvider(background=True)
        adapter = _TerminalBlockingStartAdapter()
        provider.adapter = adapter
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        owner = app._model_step_executor._provider_operation_start
        original_start = owner.start
        observed = []

        async def fail_acknowledgement(operation_id: str) -> None:
            assert operation_id == "terminal-blocking-start"
            raise RuntimeError("context acknowledgement unavailable")

        async def failing_start(**kwargs):
            observed.append(kwargs["progress"])
            kwargs["acknowledge_context_exposure"] = fail_acknowledgement
            async with aclosing(original_start(**kwargs)) as events:
                async for event in events:
                    yield event

        monkeypatch.setattr(owner, "start", failing_start)
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="start-acknowledgement-failure",
                    messages=[Message.text("user", "hello")],
                )
            )
        ]
        assert len(observed) == 1
        progress = observed[0]
        assert progress.dispatch_invoked and progress.identity_durable
        assert progress.operation_state.operation_id == "terminal-blocking-start"
        assert progress.interaction_id is not None
        assert progress.events is adapter.events
        assert adapter.events.closed
        assert adapter.start_calls == 1
        assert all(event.type is not EventType.MODEL_RETRY for event in events)
        durable = await app.session_store.load_events("start-acknowledgement-failure")
        assert sum(e.type is EventType.PROVIDER_OPERATION_STARTED for e in durable) == 1

    asyncio.run(scenario())
