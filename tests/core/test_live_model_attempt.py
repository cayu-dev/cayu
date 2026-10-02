"""Live-attempt composition preserves imports and closes the provider at caller exits."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import aclosing
from pathlib import Path

import pytest

import cayu
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.sessions.base import RunRequest


def test_live_attempt_imports_without_execution_controllers() -> None:
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
            raise AssertionError(f"Live attempt imported {fullname}")

sys.meta_path.insert(0, RejectControllers())
importlib.import_module("cayu.runtime._live_model_attempt")
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


def test_existing_attempt_imports_keep_exact_identity() -> None:
    from cayu.runtime import _live_model_attempt as owner
    from cayu.runtime import _model_step_executor as legacy

    for name in (
        "_provider_failure_proves_no_model_effect",
        "_assistant_step_result_with_published_targeted_authority",
        "_model_request_fingerprint",
        "_deadline_with_runtime_recovery_authority",
        "_model_context_overflow_error_event",
        "ModelCompletionPublicationRequest",
        "_model_stream_event_to_runtime_event",
        "_admitted_model_provider_events",
        "_owned_model_provider_events",
    ):
        assert getattr(legacy, name) is getattr(owner, name), name


class _ClosingProvider(ModelProvider):
    name = "closing-attempt"

    def __init__(self) -> None:
        self.calls = 0
        self.closes = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        try:
            yield ModelStreamEvent.text_delta("one answer")
            yield ModelStreamEvent.completed(
                {"model": request.model, "usage": {"input_tokens": 2, "output_tokens": 2}}
            )
        finally:
            self.closes += 1


@pytest.mark.parametrize("boundary", [EventType.MODEL_TEXT_DELTA, EventType.MODEL_COMPLETED])
def test_closing_live_attempt_at_event_boundary_closes_stream_once(boundary: EventType) -> None:
    async def scenario() -> None:
        provider = _ClosingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = f"close-live-attempt-{boundary.value}"
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
                raise AssertionError("The attempt did not reach the selected event boundary.")

        assert provider.calls == 1
        assert provider.closes == 1
        durable = await app.session_store.load_events(session_id)
        assert sum(e.type is EventType.MODEL_COMPLETED for e in durable) == int(
            boundary is EventType.MODEL_COMPLETED
        )
        assert all(e.type is not EventType.MODEL_RETRY for e in durable)

    asyncio.run(scenario())
