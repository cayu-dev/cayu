"""Real SDK transports retain exit evidence after a fake CLI returns its answer."""

from __future__ import annotations

import asyncio
import shlex
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from cayu import CayuApp, RunRequest
from cayu.deadlines import ExecutionDeadline
from cayu.external_agents import (
    ClaudeAgentAdapter,
    CodexAgentAdapter,
    ExternalAgentError,
    local_agent_launch,
    run_external_agent,
)
from cayu.sessions import SessionStatus
from cayu.tasks import InMemoryTaskStore, TaskQuery


@pytest.mark.parametrize("vendor", ["claude", "codex"])
@pytest.mark.parametrize("exit_status", ["0", "1", "hold"])
@pytest.mark.parametrize("strict", [False, True])
def test_cli_exit_is_recorded_after_result(tmp_path, vendor, exit_status, strict):
    pytest.importorskip("claude_agent_sdk" if vendor == "claude" else "openai_codex.client")
    cli = tmp_path / vendor
    fixture = Path(__file__).parents[1] / "fixtures/external_cli_exit.py"
    cli.write_text(
        f'#!/bin/sh\nexport CLI_EXIT={exit_status}\nexec {shlex.quote(sys.executable)} {shlex.quote(str(fixture))} "$@"\n'
    )
    cli.chmod(0o755)

    async def scenario():
        @asynccontextmanager
        async def launch(context):
            async with local_agent_launch(cli=str(cli), cwd=str(tmp_path)) as prepared:
                yield prepared

        adapter_class = ClaudeAgentAdapter if vendor == "claude" else CodexAgentAdapter
        adapter = adapter_class(model="test", launch=launch, fail_on_nonzero_exit=strict)
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        try:
            request = RunRequest(
                agent_name="external", messages=[], execution_deadline=ExecutionDeadline.after(30)
            )
            if strict and exit_status == "1":
                with pytest.raises(ExternalAgentError, match="exited nonzero"):
                    await run_external_agent(app, request, adapter, prompt="test")
            else:
                result = await run_external_agent(app, request, adapter, prompt="test")
                assert result.output == "answer"

            [task] = await app.task_store.list_tasks(TaskQuery())
            events = await app.session_store.load_events(task.session_id)
            results = [e for e in events if e.type == "custom.external_agent.result"]
            exits = [e for e in events if e.type == "custom.external_agent.process_exit"]
            assert len(results) == len(exits) == 1
            assert results[0].payload["output"] == "answer"
            assert exits[0].payload["disposition"] == (
                "closed_by_cayu" if exit_status == "hold" else "exited"
            )
            if exit_status != "hold":
                assert exits[0].payload["exit_code"] == int(exit_status)
            session = await app.session_store.load(task.session_id)
            assert session.status is (
                SessionStatus.FAILED if strict and exit_status == "1" else SessionStatus.COMPLETED
            )
        finally:
            await app.aclose()

    asyncio.run(scenario())
