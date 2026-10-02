"""A real process: repeated app lifecycles leave no orphaned tasks or late writes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROGRAM = r"""
import asyncio
import gc
import sys
import tempfile
from pathlib import Path

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.configuration import CayuConfig, ToolExecutionConfig
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.providers.base import ModelProvider, ModelStreamEvent
from cayu.storage.application import open_application_stores
from cayu.storage.memory import InMemoryKnowledgeStore, KnowledgeAccessScope
from cayu.tools.knowledge import RememberKnowledgeTool

CLOSE_APP = sys.argv[1] == "aclose"


class SlowKnowledgeStore(InMemoryKnowledgeStore):
    def __init__(self):
        super().__init__(access_scope=KnowledgeAccessScope.privileged())
        self.dispatched = asyncio.Event()

    async def publish_entry_revision(self, entry, chunks, **kwargs):
        self.dispatched.set()
        await asyncio.sleep(2.0)
        return await super().publish_entry_revision(entry, chunks, **kwargs)


class RememberProvider(ModelProvider):
    name = "fake"

    async def stream(self, request):
        if len(request.messages) == 1:
            yield ModelStreamEvent.tool_call(
                id="call_remember", name="remember_knowledge", arguments={"text": "Kept."}
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def leftover_tasks():
    return [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]


async def lifecycle(root, index):
    stores = open_application_stores(None, sqlite_path=root / f"app-{index}.sqlite3")
    knowledge = SlowKnowledgeStore()
    app = CayuApp(
        session_store=stores.session_store,
        enable_logging=False,
        # The tool call times out, but the knowledge write it started keeps running.
        config=CayuConfig(tool_execution=ToolExecutionConfig(tool_timeout_seconds=0.05)),
        owned_resources=(stores,),
    )
    app.register_provider(RememberProvider(), default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="knowledge"), knowledge_store=knowledge), default=True
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[RememberKnowledgeTool()])

    async def consume():
        request = RunRequest(
            agent_name="assistant",
            session_id=f"remember-{index}",
            messages=[Message.text("user", "remember")],
        )
        return [event async for event in app.run(request)]

    await consume()
    assert knowledge.dispatched.is_set()
    if CLOSE_APP:
        outcome = await app.aclose(timeout_s=10)
        assert outcome.settled, outcome
        assert outcome.owned_resources == "released", outcome
    else:
        await stores.close()
    leftover = leftover_tasks()
    assert not leftover, f"{len(leftover)} task(s) still pending after shutdown"


async def main():
    with tempfile.TemporaryDirectory() as directory:
        for index in range(3):
            await lifecycle(Path(directory), index)
            gc.collect()
    print("ok")


asyncio.run(main())
gc.collect()
"""


def _run(mode: str) -> subprocess.CompletedProcess[str]:
    repository = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, "-c", _PROGRAM, mode],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository / "src"), "PYTHONASYNCIODEBUG": "1"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.process
def test_aclose_leaves_no_pending_work_behind_closed_stores() -> None:
    completed = _run("aclose")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
    assert "Task was destroyed but it is pending" not in completed.stderr
    assert "exception was never retrieved" not in completed.stderr


@pytest.mark.process
def test_closing_stores_without_aclose_leaves_pending_work() -> None:
    # Control: the scenario above really leaves work behind without aclose.
    completed = _run("stores-only")

    assert completed.returncode != 0
    assert "still pending after shutdown" in completed.stderr
