"""``cayu new`` — scaffold a runnable Cayu agent project.

Files-only: writes a small project (an agent, a custom tool, SQLite stores) that a
developer edits. No control-plane or deployment logic.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

_APP_PY = '''"""__PROJECT_NAME__: a Cayu agent.

Run it:
    export OPENAI_API_KEY=sk-...
    python app.py
"""

import asyncio
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    ExecCommandTool,
    # GitRepositoryBinding,  # uncomment with the checkout binding below
    LocalRunner,
    LocalWorkspace,
    Message,
    OpenAIProvider,
    RunRequest,
    SQLiteSessionStore,
    SQLiteTaskStore,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    run_to_completion,
)


class GreetTool(Tool):
    """An example custom tool. Replace it with your own."""

    spec = ToolSpec(
        name="greet",
        effect=ToolEffect.NONE,
        description="Return a friendly greeting for a name.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content=f"Hello, {args['name']}!")


def build_app() -> CayuApp:
    Path("data/workspace").mkdir(parents=True, exist_ok=True)
    app = CayuApp(
        session_store=SQLiteSessionStore("data/sessions.sqlite"),
        task_store=SQLiteTaskStore("data/tasks.sqlite"),
    )
    app.register_provider(OpenAIProvider(), default=True)  # reads OPENAI_API_KEY
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=LocalWorkspace("data/workspace", workspace_id="ws"),
            runner=LocalRunner("data/workspace"),
            # To review a repo, swap in a checkout:
            # binding=GitRepositoryBinding(repo_url="https://github.com/OWNER/REPO.git", ref="main"),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-5.4-mini"),
        tools=[GreetTool(), ExecCommandTool()],
    )
    return app


async def main() -> None:
    app = build_app()
    outcome = await run_to_completion(
        app,
        RunRequest(
            agent_name="assistant",
            session_id="demo",
            environment_name="local",
            messages=[Message.text("user", "Greet Ada, then say what you did.")],
        ),
    )
    print(outcome.final_text if outcome.ok else f"run failed: {outcome.error}")


if __name__ == "__main__":
    asyncio.run(main())
'''

_PYPROJECT = """[project]
name = "__PROJECT_NAME__"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["cayu"]
"""

_README = """# __PROJECT_NAME__

A Cayu agent, scaffolded with `cayu new`.

## Run

```bash
pip install cayu            # or: uv add cayu
export OPENAI_API_KEY=sk-...
python app.py
```

`app.py` registers one agent with a custom `GreetTool` and the built-in
`exec_command` tool, backed by SQLite session/task stores under `data/`. Edit
`build_app()` to add tools, swap the provider, or bind a git checkout.

## Layout

```
app.py            the agent + runtime wiring
pyproject.toml    project metadata
agents/           agent declarations as the project grows
tools/            custom tools
workflows/        orchestration code
prompts/          prompt/instruction files
memory/           curated project memory
evals/            eval cases and suites
config/           local configuration
tests/            project tests
data/             SQLite stores + workspace (gitignored)
```
"""

_GITIGNORE = "data/\n__pycache__/\n*.pyc\n"

_PROJECT_DIRS = (
    "agents",
    "tools",
    "workflows",
    "prompts",
    "memory",
    "evals",
    "config",
    "tests",
    "data",
)


def add_new_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("new", help="Scaffold a new Cayu agent project.")
    parser.add_argument("name", help="Project name (also the directory name).")
    parser.add_argument(
        "--dir",
        metavar="DIR",
        default=".",
        help="Parent directory to create the project in (default: current directory).",
    )


def _project_files(name: str) -> dict[str, str]:
    def render(template: str) -> str:
        return template.replace("__PROJECT_NAME__", name)

    return {
        "app.py": render(_APP_PY),
        "pyproject.toml": render(_PYPROJECT),
        "README.md": render(_README),
        ".gitignore": _GITIGNORE,
    }


def run_new(args: argparse.Namespace) -> int:
    name = args.name
    if not _NAME_RE.fullmatch(name):
        print(
            f"error: invalid project name {name!r} "
            "(use letters, digits, '-' or '_', starting with a letter).",
            file=sys.stderr,
        )
        return 1

    target = Path(args.dir) / name
    if target.exists() and not target.is_dir():
        print(f"error: {target} already exists and is not a directory.", file=sys.stderr)
        return 1
    if target.exists() and any(target.iterdir()):
        print(f"error: {target} already exists and is not empty.", file=sys.stderr)
        return 1

    for rel, content in _project_files(name).items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for rel in _PROJECT_DIRS:
        directory = target / rel
        directory.mkdir(parents=True, exist_ok=True)
        # git does not track empty directories; keep a .gitkeep so the structural
        # layout survives the first commit. data/ is gitignored, so skip it.
        if rel != "data":
            (directory / ".gitkeep").write_text("", encoding="utf-8")

    print(f"Scaffolded {target}/ — next steps:")
    print(f"  cd {target}")
    print("  pip install cayu")
    print("  export OPENAI_API_KEY=sk-...")
    print("  python app.py")
    return 0
