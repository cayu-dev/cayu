"""``cayu new`` — scaffold a safe, verifiable Cayu agent project."""

from __future__ import annotations

import argparse
import json
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

GENERATED_IMPORTS_START = "# <cayu:generated-imports>"
GENERATED_IMPORTS_END = "# </cayu:generated-imports>"
GENERATED_STARTER_TOOLS_START = "# <cayu:generated-starter-tools>"
GENERATED_STARTER_TOOLS_END = "# </cayu:generated-starter-tools>"
GENERATED_REGISTRATIONS_START = "# <cayu:generated-registrations>"
GENERATED_REGISTRATIONS_END = "# </cayu:generated-registrations>"
GENERATED_AGENT_IMPORTS_START = "# <cayu:generated-agent-imports>"
GENERATED_AGENT_IMPORTS_END = "# </cayu:generated-agent-imports>"
GENERATED_AGENT_CONFIG_START = "# <cayu:generated-agent-config>"
GENERATED_AGENT_CONFIG_END = "# </cayu:generated-agent-config>"
PROVIDER_OVERRIDE_AGENT_HELPER = "_agent_for_provider_override"

_APP_PY = '''"""Application factory for __PROJECT_NAME__.

Every process calls ``build_app()`` and owns the returned CayuApp. Durable
stores, not this Python object, coordinate state between processes.
"""

import os

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    AnthropicProvider,
    CayuApp,
    ModelProvider,
    OpenAIProvider,
    OpenAISubscriptionProvider,
    ScriptedModelProvider,
    SessionStore,
    SQLiteSessionStore,
    SQLiteTaskStore,
    TaskStore,
)

from agents.agent import AGENT
from configuration import configured_provider_choice

# Generated tool-backed slices add their imports and registrations here.
# <cayu:generated-imports>
# </cayu:generated-imports>


class _ScaffoldPlaceholderProvider(ScriptedModelProvider):
    """Credential-free placeholder rejected only by live ``run.py`` validation."""


def configured_provider() -> ModelProvider:
    """Construct only the explicitly selected provider.

    Credential variables authenticate the selected provider; they never choose
    one. A same-name scripted placeholder keeps inspection and hermetic proof
    credential-free while ``run.py`` rejects it before live execution.
    """

    choice = configured_provider_choice()
    if choice is None:
        return _ScaffoldPlaceholderProvider([], name="unconfigured")
    if choice == "openai-subscription":
        return OpenAISubscriptionProvider()
    if choice == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        return (
            OpenAIProvider(api_key=api_key)
            if api_key
            else _ScaffoldPlaceholderProvider([], name="openai")
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    return (
        AnthropicProvider(api_key=api_key)
        if api_key
        else _ScaffoldPlaceholderProvider([], name="anthropic")
    )


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
    if not isinstance(provider, _ScaffoldPlaceholderProvider):
        return
    choice = configured_provider_choice()
    if choice is None:
        raise RuntimeError(
            "no provider is selected; set CAYU_PROVIDER to openai, anthropic, or "
            "openai-subscription (credentials do not select a provider)"
        )
    credential = "OPENAI_API_KEY" if choice == "openai" else "ANTHROPIC_API_KEY"
    raise RuntimeError(f"provider {choice!r} is selected but {credential} is not set")


def _agent_for_provider_override(
    agent: AgentSpec, provider: ModelProvider | None
) -> AgentSpec:
    """Route an agent through an explicitly injected test/eval provider."""

    if provider is None:
        return agent
    return agent.model_copy(update={"provider_name": provider.name})


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
        session_store=session_store or SQLiteSessionStore("data/cayu.db"),
        task_store=task_store or SQLiteTaskStore("data/cayu.db"),
    )
    selected_provider = provider
    if selected_provider is None:
        selected_provider = configured_provider()
    app.register_provider(selected_provider, default=True)
    starter_tools = []
    starter_external_tool_names = []
    # <cayu:generated-starter-tools>
    # </cayu:generated-starter-tools>
    app.register_agent(
        _agent_for_provider_override(AGENT, provider),
        tools=starter_tools,
        tool_policy=(
            AlwaysRequireApprovalToolPolicy(tools=starter_external_tool_names)
            if starter_external_tool_names
            else None
        ),
    )
    # <cayu:generated-registrations>
    # </cayu:generated-registrations>
    return app
'''

_CONFIGURATION_PY = '''"""Explicit provider and compatible model selection for this application."""

import os

_SCAFFOLDED_PROVIDER = __PROVIDER_LITERAL__
_SUPPORTED_PROVIDERS = {"openai", "anthropic", "openai-subscription"}
_PROVIDER_NAMES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "openai-subscription": "openai_subscription",
}
_DEFAULT_MODELS = {
    "openai": "gpt-5.6-luna",
    "anthropic": "claude-sonnet-4-6",
    "openai-subscription": "gpt-5.4",
}


def configured_provider_choice() -> str | None:
    """Return explicit project/env selection without inspecting credentials."""

    selected = os.environ.get("CAYU_PROVIDER", _SCAFFOLDED_PROVIDER)
    if selected is None:
        return None
    if selected not in _SUPPORTED_PROVIDERS:
        choices = ", ".join(sorted(_SUPPORTED_PROVIDERS))
        raise RuntimeError(f"CAYU_PROVIDER must be one of: {choices}")
    return selected


def configured_provider_name() -> str | None:
    selected = configured_provider_choice()
    return None if selected is None else _PROVIDER_NAMES[selected]


def configured_model() -> str:
    override = os.environ.get("CAYU_MODEL")
    if override:
        return override
    selected = configured_provider_choice()
    return (
        "provider-model-unconfigured" if selected is None else _DEFAULT_MODELS[selected]
    )
'''

_AGENT_PY = """from cayu import AgentSpec

from configuration import configured_model, configured_provider_name

# Generated first-tool imports and agent contract additions live in these regions.
# <cayu:generated-agent-imports>
# </cayu:generated-agent-imports>

_SYSTEM_PROMPT_PARTS: list[str] = []
_WORKFLOW_TOOL_NAMES: list[str] = []
_AUTHORING_STATE: str | None = None

# <cayu:generated-agent-config>
# </cayu:generated-agent-config>

AGENT = AgentSpec(
    name="__PROJECT_NAME__",
    model=configured_model(),
    provider_name=configured_provider_name(),
    system_prompt="\\n".join(_SYSTEM_PROMPT_PARTS) or None,
    workflow_tool_names=tuple(_WORKFLOW_TOOL_NAMES),
    authoring_state=_AUTHORING_STATE,
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
dependencies = ["cayu>=__CAYU_VERSION__"]

[project.optional-dependencies]
dev = ["pytest"]

[tool.cayu]
factory = "app:build_app"
eval_target = "evals.agent:build_eval"

[tool.cayu.session_store]
backend = "sqlite"
path = "data/cayu.db"

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

Run `uv run cayu guide authoring#cayu-map` to select another concept only when
the requested behavior requires it. `uv run cayu guide references` contains the
package-shipped offline references.

## Setup and prove the project

```bash
uv sync --extra dev
uv run cayu guide anatomy
uv run cayu inspect --json
uv run cayu check --json
uv run pytest
uv run cayu eval run
uv run cayu session list
```

These commands require no model API key. They prove project construction,
static wiring, a deterministic model response, and its eval.

## Run with a live provider

Provider intent is explicit. This scaffold defaults to `__PROVIDER_DISPLAY__`;
override it with `CAYU_PROVIDER=openai`, `anthropic`, or
`openai-subscription`. API-key variables authenticate that choice and never
select it automatically.

OpenAI Platform API:

```bash
export CAYU_PROVIDER=openai
export OPENAI_API_KEY=sk-...
uv run python run.py --message "YOUR REQUEST"
```

Anthropic API:

```bash
export CAYU_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
uv run python run.py --message "YOUR REQUEST"
```

Your own ChatGPT subscription for local testing:

```bash
uv run cayu auth openai login
CAYU_PROVIDER=openai-subscription uv run python run.py --message "YOUR REQUEST"
```

Subscription mode selects `gpt-5.4` by default. Set `CAYU_MODEL` if your plan
offers a different model.

This experimental path is intended for the subscription holder's own local
development and evaluation. It is not intended for production, customer-facing
or multi-user services, credential sharing, resale, or bypassing plan limits.
For production, use the OpenAI Platform API or another officially supported
provider. Run `uv run cayu guide providers#openai-subscription` for the local
support boundary.
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
`uv run cayu guide authoring#cayu-map`.

If another capability is required, use the smallest package-shipped reference
from `uv run cayu guide references`.

This scaffold is for local development. Deployment is a separate task.

## Project commands

- Setup: `uv sync --extra dev`.
- Application contract: `uv run cayu guide anatomy`.
- Authoring details: `uv run cayu guide authoring`.
- Inspect/check: `uv run cayu inspect --json` and `uv run cayu check --json`.
- Hermetic proof: `uv run pytest` and `uv run cayu eval run`.
- Live execution: `uv run python run.py --message "USER REQUEST"` after configuring a
  provider in `app.configured_provider()`.

Use public `cayu` imports and public CLI JSON only. Do not depend on Cayu source,
private symbols, or import-time application construction.

If the job truly needs a tool, read `cayu guide tool-effects`; every tool must
declare `ToolEffect`, and effect metadata does not authorize execution. A
`ScriptedModelProvider` proves handling of predetermined calls, not prompt
comprehension or live model behavior.

For the starter's first real tool, run
`uv run cayu generate tool TOOL_NAME --agent __PROJECT_NAME__ --effect EFFECT`.
Then replace the generated sample schema, implementation, test, and eval with
domain behavior; `cayu check` keeps the tracer-bullet warning active until the
explicit authoring marker is removed.
"""

_GITIGNORE = "data/\n__pycache__/\n*.pyc\n.pytest_cache/\n.venv/\n"


def add_new_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "new",
        help="Scaffold a new Cayu agent project.",
        description=(
            "Scaffold a new Cayu application project. Follow the printed `uv sync` "
            "and credential-free verification commands next."
        ),
    )
    parser.add_argument("name", help="Project name (also the directory name).")
    parser.add_argument(
        "--dir",
        metavar="DIR",
        default=".",
        help="Parent directory to create the project in (default: current directory).",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "openai-subscription"),
        help=(
            "Explicit live-provider default. Omit for a provider-neutral scaffold; "
            "CAYU_PROVIDER can select or override it later."
        ),
    )


def _installed_cayu_version() -> str:
    try:
        return version("cayu")
    except PackageNotFoundError:
        return "0.1.0rc2"


def project_files(name: str, *, provider: str | None = None) -> dict[str, str]:
    def render(template: str) -> str:
        provider_display = provider or "no live provider"
        provider_literal = "None" if provider is None else json.dumps(provider)
        return (
            template.replace("__PROJECT_NAME__", name)
            .replace("__CAYU_VERSION__", _installed_cayu_version())
            .replace("__PROVIDER_DISPLAY__", provider_display)
            .replace("__PROVIDER_LITERAL__", provider_literal)
        )

    return {
        "app.py": render(_APP_PY),
        "configuration.py": render(_CONFIGURATION_PY),
        "run.py": _RUN_PY,
        "agents/__init__.py": "",
        "agents/agent.py": render(_AGENT_PY),
        "tests/test_agent.py": render(_TEST_PY),
        "evals/__init__.py": "",
        "evals/agent.py": render(_EVAL_PY),
        "pyproject.toml": render(_PYPROJECT),
        "README.md": render(_README),
        "AGENTS.md": render(_AGENTS_MD),
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

    for rel, content in project_files(name, provider=args.provider).items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    print(f"Scaffolded {target}/ — credential-free proof:")
    print(f"  cd {target}")
    print("  uv sync --extra dev")
    print("  uv run cayu inspect --json")
    print("  uv run cayu check --json")
    print("  uv run pytest")
    print("  uv run cayu eval run")
    if args.provider is None:
        print("  Live provider: none selected; set CAYU_PROVIDER explicitly before `run.py`.")
    else:
        print(f"  Live provider: {args.provider} (credentials authenticate this choice).")
    return 0
