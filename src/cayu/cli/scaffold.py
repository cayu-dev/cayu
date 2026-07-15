"""``cayu new`` — scaffold a safe, verifiable Cayu agent project."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

GENERATED_IMPORTS_START = "# <cayu:generated-imports>"
GENERATED_IMPORTS_END = "# </cayu:generated-imports>"
GENERATED_REGISTRATIONS_START = "# <cayu:generated-registrations>"
GENERATED_REGISTRATIONS_END = "# </cayu:generated-registrations>"

_APP_PY = '''"""Application factory for __PROJECT_NAME__.

Every process calls ``build_app()`` and owns the returned CayuApp. Durable
stores, not this Python object, coordinate state between processes.
"""

import os

from cayu import (
    CayuApp,
    ModelProvider,
    OpenAIProvider,
    ScriptedModelProvider,
    SessionStore,
    SQLiteSessionStore,
    SQLiteTaskStore,
    TaskStore,
)

from agents.assistant import ASSISTANT_AGENT
from tools.greet import GreetTool

# Generated external-effect slices import AlwaysRequireApprovalToolPolicy and
# register an enforcing policy. The safe starter tool below has no side effect.
# <cayu:generated-imports>
# </cayu:generated-imports>


def configured_provider() -> ModelProvider:
    """Resolve the live provider only when its credential is available.

    The no-key placeholder is intentionally non-runnable; it lets public
    inspection and checking describe the project without claiming a live check.
    Tests and evals inject their own ScriptedModelProvider explicitly.
    """

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAIProvider(api_key=api_key)
    return ScriptedModelProvider([], name="openai-unconfigured")


def build_app(
    *,
    provider: ModelProvider | None = None,
    session_store: SessionStore | None = None,
    task_store: TaskStore | None = None,
) -> CayuApp:
    """Construct a fresh process-scoped application graph.

    Optional arguments are public test seams. Normal processes call build_app()
    with no arguments and receive the configured durable local stores.
    """

    app = CayuApp(
        session_store=session_store or SQLiteSessionStore("data/sessions.sqlite"),
        task_store=task_store or SQLiteTaskStore("data/tasks.sqlite"),
    )
    app.register_provider(provider or configured_provider(), default=True)
    app.register_agent(ASSISTANT_AGENT, tools=[GreetTool()])
    # <cayu:generated-registrations>
    # </cayu:generated-registrations>
    return app
'''

_AGENT_PY = """from cayu import AgentSpec

from tools.greet import GREET_TOOL_NAME


ASSISTANT_AGENT = AgentSpec(
    name="assistant",
    model="gpt-5.6-luna",
    system_prompt=f"Use {GREET_TOOL_NAME} when it directly helps the user.",
    workflow_tool_names=(GREET_TOOL_NAME,),
)
"""

_TOOL_PY = '''from cayu import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec


GREET_TOOL_NAME = "greet"


class GreetTool(Tool):
    """Return a greeting without changing external state."""

    spec = ToolSpec(
        name=GREET_TOOL_NAME,
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
'''

_TEST_PY = """from __future__ import annotations

import asyncio

from cayu import (
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    run_to_completion,
)

from app import build_app
from tools.greet import GREET_TOOL_NAME


def test_assistant_uses_greet_tool_through_the_runtime() -> None:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(name=GREET_TOOL_NAME, arguments={"name": "Ada"}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Greeted Ada."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = build_app(
        provider=provider,
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
    )

    outcome = asyncio.run(
        run_to_completion(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "Greet Ada")],
                max_steps=2,
            ),
        )
    )

    assert outcome.ok
    assert outcome.final_text == "Greeted Ada."
    assert len(provider.requests) == 2
"""

_EVAL_PY = """from cayu import (
    EvalCase,
    EvalPlan,
    EvalSuite,
    FinalOutputContains,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SessionCompleted,
    ToolCalled,
)

from app import build_app
from tools.greet import GREET_TOOL_NAME


def build_eval() -> EvalPlan:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(name=GREET_TOOL_NAME, arguments={"name": "Ada"}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Greeted Ada."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = build_app(
        provider=provider,
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
    )
    suite = EvalSuite(
        id="assistant-trajectory",
        cases=[
            EvalCase(
                id="greets-by-name",
                request=RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "Greet Ada")],
                    max_steps=2,
                ),
                assertions=[
                    SessionCompleted(),
                    ToolCalled(GREET_TOOL_NAME),
                    FinalOutputContains("Ada"),
                ],
            )
        ],
    )
    return EvalPlan(app=app, suite=suite)
"""

_PYPROJECT = """[project]
name = "__PROJECT_NAME__"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["cayu"]

[project.optional-dependencies]
console = ["cayu[console]"]
dev = ["pytest"]

[tool.cayu]
factory = "app:build_app"

[tool.pytest.ini_options]
pythonpath = ["."]
"""

_README = """# __PROJECT_NAME__

A safe, inspectable Cayu agent scaffold.

## Application structure

`build_app()` returns a fresh process-scoped `CayuApp`. Each script, console,
server, worker, or test calls the factory for itself. Configured durable stores
coordinate cross-process state; the Python application object is not a global
registry or singleton. Importing the project does not construct the app or
start workers, recovery, schedulers, sessions, models, or tools.

Run `cayu guide anatomy` for the canonical factory, process-role, durable-state,
and lifecycle contract.

## Setup and prove the project

```bash
pip install -e '.[console,dev]' # or: uv sync --extra console --extra dev
cayu guide anatomy
cayu inspect --json
cayu check --json
pytest
cayu eval run evals.assistant:build_eval
```

These commands require no model API key. They prove project construction,
static wiring, a deterministic runtime tool trajectory, and its eval. They do
not claim that a live provider or execution environment was verified.

## Run with a live provider

```bash
export OPENAI_API_KEY=sk-...
python run.py
```

The checked-in `AGENTS.md` is the canonical workflow for coding agents. Add a
reviewable agent/tool/test/eval slice with
`cayu generate slice NAME --tool TOOL --effect EFFECT`.
"""

_RUN_PY = """import asyncio
import os

from cayu import Message, RunRequest, run_to_completion

from app import build_app


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before live execution.")
    outcome = await run_to_completion(
        build_app(),
        RunRequest(
            agent_name="assistant",
            messages=[Message.text("user", "Greet Ada, then say what you did.")],
        ),
    )
    print(outcome.final_text if outcome.ok else f"run failed: {outcome.error}")


if __name__ == "__main__":
    asyncio.run(main())
"""

_AGENTS_MD = """# Coding-agent instructions

Build desired behavior first; introduce only the Cayu subsystems the product
needs. `app.py` is the application factory and explicit registration surface.

## Project commands

- Setup: `pip install -e '.[console,dev]'` or
  `uv sync --extra console --extra dev`.
- Application contract: `cayu guide anatomy`.
- Inspect/check: `cayu inspect --json` and `cayu check --json`.
- Safe generation: `cayu generate slice NAME --tool TOOL --effect EFFECT`.
- Hermetic proof: `pytest` and `cayu eval run MODULE:build_eval`.
- Interactive inspection: `cayu console` (constructs the app; starts no services).
- Live execution: `python run.py` only when `OPENAI_API_KEY` is explicitly available.

## Supported loop

1. Clarify users, jobs, triggers, inputs/outputs, autonomy, state, effects,
   approval boundaries, recovery, environments, artifacts, and eval cases.
   Read `cayu guide anatomy` for the application contract and
   `cayu guide authoring` when choosing Cayu concepts.
2. Run `cayu inspect --json` and `cayu check --json` before editing.
3. Plan generated edits with
   `cayu generate slice NAME --tool TOOL --effect EFFECT --dry-run --json`.
4. Apply with `cayu generate slice NAME --tool TOOL --effect EFFECT`; review every changed file.
5. Run `cayu check --json`, `pytest`, and the relevant
   `cayu eval run evals.assistant:build_eval` target.
6. Exercise optional live boundaries only when credentials and infrastructure
   are explicitly available, then report evidence by verification layer.

Use public `cayu` imports and public CLI JSON only. Do not depend on Cayu source,
private symbols, import-time application construction, or arbitrary Python
rewriting. Edit user-owned code normally; generated commands may update only
their delimited registration regions and independent generated files.

Every tool must declare `ToolEffect`. External-effect tools require an enforcing
policy such as `AlwaysRequireApprovalToolPolicy`; never treat a comment or UI
confirmation as authorization. Prefer closed JSON schemas.

Use one constant for each exact tool name across `ToolSpec`, generated workflow
instructions, `AgentSpec.workflow_tool_names`, tests, and evals. `cayu check`
compares the explicit workflow names with that agent's registered tool manifest;
it never infers names from arbitrary prose.

Report static inspection, hermetic runtime tests, process-boundary checks, and
credential-gated live checks separately. Do not claim live verification from
successful imports, construction, mocks, or scripted providers. A
`ScriptedModelProvider` proves handling of predetermined calls, not prompt
comprehension or model tool choice.
"""

_GITIGNORE = "data/\n__pycache__/\n*.pyc\n.pytest_cache/\n.venv/\n"


def add_new_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("new", help="Scaffold a new Cayu agent project.")
    parser.add_argument("name", help="Project name (also the directory name).")
    parser.add_argument(
        "--dir",
        metavar="DIR",
        default=".",
        help="Parent directory to create the project in (default: current directory).",
    )


def project_files(name: str) -> dict[str, str]:
    def render(template: str) -> str:
        return template.replace("__PROJECT_NAME__", name)

    return {
        "app.py": render(_APP_PY),
        "run.py": _RUN_PY,
        "agents/__init__.py": "",
        "agents/assistant.py": _AGENT_PY,
        "tools/__init__.py": "",
        "tools/greet.py": _TOOL_PY,
        "tests/test_assistant.py": _TEST_PY,
        "evals/__init__.py": "",
        "evals/assistant.py": _EVAL_PY,
        "pyproject.toml": render(_PYPROJECT),
        "README.md": render(_README),
        "AGENTS.md": _AGENTS_MD,
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

    for rel, content in project_files(name).items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    print(f"Scaffolded {target}/ — credential-free proof:")
    print(f"  cd {target}")
    print("  pip install -e '.[console,dev]'  # or: uv sync --extra console --extra dev")
    print("  cayu inspect --json")
    print("  cayu check --json")
    print("  pytest")
    print("  cayu eval run evals.assistant:build_eval")
    return 0
