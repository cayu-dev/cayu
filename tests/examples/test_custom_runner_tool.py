"""Guard for examples/custom_runner_tool.py — a Tool using ctx.runner + ctx.workspace."""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    LocalRunner,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    run_to_completion,
)

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "custom_runner_tool.py"


def _load():
    spec = importlib.util.spec_from_file_location("custom_runner_tool_example", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load()


def test_tool_uses_runner_and_workspace(tmp_path: Path) -> None:
    if shutil.which("wc") is None:
        pytest.skip("wc is required for this example")
    app = CayuApp()
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="c1",
                        name="line_count",
                        arguments={"filename": "f.txt", "text": "a\nb\n"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("ok"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=LocalWorkspace(tmp_path, workspace_id="ws"),
            runner=LocalRunner(tmp_path),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="m"), tools=[mod.LineCountTool()])

    outcome = asyncio.run(
        run_to_completion(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s",
                environment_name="local",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    assert outcome.ok
    completed = [e for e in outcome.events if e.type == EventType.TOOL_CALL_COMPLETED]
    assert completed
    assert completed[0].payload["result"]["structured"]["exit_code"] == 0
