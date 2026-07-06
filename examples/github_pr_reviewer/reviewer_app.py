from __future__ import annotations

import os
from pathlib import Path

from examples.github_pr_reviewer.github_tools import GetPRDiffTool, PostPRCommentTool
from examples.github_pr_reviewer.qa_policy import QaCommandPolicy
from examples.github_pr_reviewer.workspace import PRReviewWorkspaceFactory

from cayu import (
    AgentSpec,
    CayuApp,
    EnvironmentSpec,
    ExecCommandTool,
    ListFilesTool,
    ReadFileTool,
    SQLiteTaskStore,
    StaticToolPolicy,
    Tool,
)

REVIEWER_SYSTEM_PROMPT = """\
You are an autonomous pull-request review agent. For the pull request described in
the first user message:

1. Call get_pr_diff to see what changed.
2. Use list_files / read_file to inspect changed files and related code as needed.
3. QA the change by running the project's test suite and relevant checks with
   exec_command (only a fixed allowlist of test/build tools is permitted; raw shell
   strings are rejected).
4. Call post_pr_comment exactly once with a concise, specific review: what you
   checked, what passed/failed, and any concrete concerns. Do not comment twice.
"""


def build_app(
    task_db_path: Path,
    workspace_root: Path,
    *,
    provider,
    model: str,
    with_credentials: bool = True,
    pr_diff_tool: Tool | None = None,
) -> tuple[CayuApp, SQLiteTaskStore]:
    """Construct the full runtime: durable task store, per-PR workspace, agent + tools."""
    task_store = SQLiteTaskStore(task_db_path)
    app = CayuApp(task_store=task_store)
    app.register_provider(provider, default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="pr-workspace"),
        PRReviewWorkspaceFactory(workspace_root, with_credentials=with_credentials),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="pr-reviewer", model=model, system_prompt=REVIEWER_SYSTEM_PROMPT),
        tools=[
            pr_diff_tool if pr_diff_tool is not None else GetPRDiffTool(),
            PostPRCommentTool(),
            ReadFileTool(),
            ListFilesTool(),
            ExecCommandTool(policy=QaCommandPolicy()),
        ],
        tool_policy=StaticToolPolicy(
            allow=["get_pr_diff", "post_pr_comment", "read_file", "list_files", "exec_command"]
        ),
    )
    return app, task_store


def build_provider() -> tuple[object, str]:
    """Pick a real provider from whichever key is set (used outside the demo)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        from cayu import AnthropicProvider

        return AnthropicProvider(), os.environ.get("CAYU_MODEL", "claude-sonnet-4-6")
    if os.environ.get("OPENAI_API_KEY"):
        from cayu import OpenAIProvider

        return OpenAIProvider(), os.environ.get("CAYU_MODEL", "gpt-5.4-mini")
    raise RuntimeError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run a live review.")
