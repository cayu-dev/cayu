"""The file-worker agent shared by both durable file workflows.

One agent writes ``transform.py``, runs it as a process, and reads ``result.txt``.
Each example composes this part with its own environment factory and completion
path; neither redefines the tools or their policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    ExecCommandTool,
    ParameterConstrainedToolPolicy,
    ProcessCommandPolicy,
    ReadFileTool,
    RequiredAllowlistRule,
    RequiredFieldRule,
    WriteFileTool,
)

AGENT_NAME = "file-worker"
PROGRAM = "transform.py"
ARTIFACT = "result.txt"


def register_file_worker(app: CayuApp, *, workspace_root: Path, system_prompt: str) -> None:
    """Register the agent with write, process, and read tools limited to its files."""

    app.register_agent(
        AgentSpec(
            name=AGENT_NAME,
            model="scripted",
            system_prompt=system_prompt,
            workflow_tool_names=("write_file", "exec_command", "read_file"),
        ),
        tools=[
            WriteFileTool(),
            ExecCommandTool(
                policy=ProcessCommandPolicy(
                    allowed_executables=(sys.executable,),
                    allowed_cwds=(str(workspace_root.resolve()),),
                    max_timeout_s=30,
                )
            ),
            ReadFileTool(),
        ],
        tool_policy=ParameterConstrainedToolPolicy(
            {
                "write_file": (
                    RequiredAllowlistRule("path", values=(PROGRAM,)),
                    RequiredFieldRule("content"),
                ),
                "exec_command": (
                    RequiredAllowlistRule("kind", values=("process",)),
                    RequiredFieldRule("argv"),
                ),
                "read_file": (RequiredAllowlistRule("path", values=(ARTIFACT,)),),
            }
        ),
    )
