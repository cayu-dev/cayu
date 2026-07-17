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

from agents.agent import AGENT

# Generated tool-backed slices add their imports and registrations here.
# <cayu:generated-imports>
# </cayu:generated-imports>

_UNCONFIGURED_PROVIDER_NAME = "openai-unconfigured"


def configured_provider() -> ModelProvider:
    """Resolve the live provider only when its credential is available.

    The no-key placeholder is intentionally non-runnable; it lets public
    inspection and checking describe the project without claiming a live check.
    Tests and evals inject their own ScriptedModelProvider explicitly.
    """

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAIProvider(api_key=api_key)
    return ScriptedModelProvider([], name=_UNCONFIGURED_PROVIDER_NAME)


def validate_run_configuration(app: CayuApp, agent_name: str) -> None:
    """Require live-provider setup after the command has selected an agent."""

    manifest_agent = next(
        agent for agent in app.describe().agents if agent.name == agent_name
    )
    if manifest_agent.resolved_provider is None:
        raise RuntimeError(
            f"agent {agent_name!r} does not resolve to exactly one model provider"
        )
    provider = app.get_provider(manifest_agent.resolved_provider)
    if provider.name == _UNCONFIGURED_PROVIDER_NAME:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; set it or update app.configured_provider()."
        )


def build_app(
    *,
    provider: ModelProvider | None = None,
    session_store: SessionStore | None = None,
    task_store: TaskStore | None = None,
) -> CayuApp:
    """Construct a fresh process-scoped application graph.

    Injected stores and providers are public test seams. Inspection can call
    ``build_app()`` without live-provider credentials.
    """

    app = CayuApp(
        session_store=session_store or SQLiteSessionStore("data/sessions.sqlite"),
        task_store=task_store or SQLiteTaskStore("data/tasks.sqlite"),
    )
    selected_provider = provider
    if selected_provider is None:
        selected_provider = configured_provider()
    app.register_provider(selected_provider, default=True)
    app.register_agent(AGENT)
    # <cayu:generated-registrations>
    # </cayu:generated-registrations>
    return app
'''

_AGENT_PY = """from cayu import AgentSpec


AGENT = AgentSpec(
    name="__PROJECT_NAME__",
    model="gpt-5.6-luna",
)
"""

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


def test_agent_runs_through_the_runtime() -> None:
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("Agent result."),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
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
                agent_name="__PROJECT_NAME__",
                messages=[Message.text("user", "Handle this request")],
            ),
        )
    )

    assert outcome.ok
    assert outcome.final_text == "Agent result."
    assert len(provider.requests) == 1
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
)

from app import build_app


def build_eval() -> EvalPlan:
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("Agent result."),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = build_app(
        provider=provider,
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
    )
    suite = EvalSuite(
        id="agent-output",
        cases=[
            EvalCase(
                id="returns-output",
                request=RunRequest(
                    agent_name="__PROJECT_NAME__",
                    messages=[Message.text("user", "Handle this request")],
                ),
                assertions=[
                    SessionCompleted(),
                    FinalOutputContains("Agent result"),
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
eval_target = "evals.agent:build_eval"

[tool.pytest.ini_options]
pythonpath = ["."]
"""

_README = """# __PROJECT_NAME__

A model-only Cayu agent scaffold. It starts with one agent, one deterministic
runtime test, and one output eval. Add capabilities only when the job needs them.

## Application structure

Describe the requested job in `agents/agent.py`; update `tests/test_agent.py`
and `evals/agent.py` to prove that behavior. The project factory is `build_app()`
in `app.py`. Run `cayu guide anatomy` for its lifecycle contract.

Use the [Cayu Map](https://github.com/cayu-dev/cayu/blob/main/src/cayu/guides/authoring.md#cayu-map)
to select another concept only when the requested behavior requires it. The
[examples index](https://github.com/cayu-dev/cayu/blob/main/examples/README.md)
links the smallest runnable references.

## Setup and prove the project

```bash
pip install -e '.[console,dev]' # or: uv sync --extra console --extra dev
cayu guide anatomy
cayu inspect --json
cayu check --json
pytest
cayu eval run
```

These commands require no model API key. They prove project construction,
static wiring, a deterministic model response, and its eval.

## Run with a live provider

```bash
export OPENAI_API_KEY=sk-...
python run.py --message "YOUR REQUEST"
```

`--agent` is optional while this is the only registered agent. The checked-in
`AGENTS.md` is the local instruction surface for coding agents.
"""

_RUN_PY = """from __future__ import annotations

from cayu import run_project_entrypoint

from app import build_app, validate_run_configuration


def main(argv: list[str] | None = None) -> int:
    return run_project_entrypoint(
        build_app,
        argv,
        validate_run=validate_run_configuration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
"""

_AGENTS_MD = """# Coding-agent instructions

Edit the existing agent, test, and eval to implement the user's first requested
job. Do not retain the starter and add a second agent. Tools are optional: add
one only for a real capability outside the model, such as reading a repository
or calling an API. Do not create echo, pass-through, or placeholder tools.

Use the Cayu Map to choose only the concepts the job needs:
https://github.com/cayu-dev/cayu/blob/main/src/cayu/guides/authoring.md#cayu-map

If another capability is required, use the smallest matching reference from:
https://github.com/cayu-dev/cayu/blob/main/examples/README.md

This scaffold is for local development. Deployment is a separate task.

## Project commands

- Setup: `pip install -e '.[console,dev]'` or
  `uv sync --extra console --extra dev`.
- Application contract: `cayu guide anatomy`.
- Authoring details: `cayu guide authoring`.
- Inspect/check: `cayu inspect --json` and `cayu check --json`.
- Hermetic proof: `pytest` and `cayu eval run`.
- Live execution: `python run.py --message "USER REQUEST"` after configuring a
  provider in `app.configured_provider()`.

Use public `cayu` imports and public CLI JSON only. Do not depend on Cayu source,
private symbols, or import-time application construction.

If the job truly needs a tool, read `cayu guide tool-effects`; every tool must
declare `ToolEffect`, and effect metadata does not authorize execution. A
`ScriptedModelProvider` proves handling of predetermined calls, not prompt
comprehension or live model behavior.
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
        "agents/agent.py": render(_AGENT_PY),
        "tests/test_agent.py": render(_TEST_PY),
        "evals/__init__.py": "",
        "evals/agent.py": render(_EVAL_PY),
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
    print("  cayu eval run")
    return 0
