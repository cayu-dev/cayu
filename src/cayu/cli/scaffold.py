"""``cayu new`` — scaffold a safe, verifiable Cayu agent project."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

from cayu._version import package_version
from cayu.cli._bounded_command import (
    BoundedCommandOutputOverflowError,
    BoundedCommandReadError,
    BoundedCommandStartError,
    BoundedCommandTimeoutError,
    run_bounded_command,
)
from cayu.workspaces import LocalWorkspace

_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_TEMPLATE_TOKEN_RE = re.compile(
    r"__(?:PROJECT_NAME|AGENT_NAME|REVIEWER_NAME|CAYU_VERSION|PROVIDER_DISPLAY|PROVIDER_LITERAL|"
    r"PROVIDER_GUIDE_POINTER)__"
)

_SCAFFOLD_COMMAND_TIMEOUT_S = 10.0
_SCAFFOLD_COMMAND_OUTPUT_LIMIT_BYTES = 64 * 1024
_SAFE_SCAFFOLD_ENV_KEYS = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
)


class _ScaffoldCommandError(RuntimeError):
    """Bounded, content-safe command failure used by coding scaffolding."""


def _sanitized_scaffold_git_environment(*, cwd: Path) -> dict[str, str]:
    """Return Git authority confined to ``cwd`` and repository-local config."""

    environment = {key: os.environ[key] for key in _SAFE_SCAFFOLD_ENV_KEYS if key in os.environ}
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CEILING_DIRECTORIES": str(cwd.parent.resolve()),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_DISCOVERY_ACROSS_FILESYSTEM": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _run_scaffold_command(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    allowed_exit_codes: frozenset[int] = frozenset({0}),
) -> str:
    """Run a bounded scaffold command and return bounded combined output."""

    try:
        completed = run_bounded_command(
            argv,
            cwd=cwd,
            env=env,
            timeout_s=_SCAFFOLD_COMMAND_TIMEOUT_S,
            output_limit_bytes=_SCAFFOLD_COMMAND_OUTPUT_LIMIT_BYTES,
        )
    except BoundedCommandStartError:
        raise _ScaffoldCommandError(f"{Path(argv[0]).name} could not start") from None
    except BoundedCommandTimeoutError:
        raise _ScaffoldCommandError(f"{Path(argv[0]).name} timed out") from None
    except BoundedCommandOutputOverflowError:  # pragma: no cover - capture truncates here
        raise _ScaffoldCommandError(f"{Path(argv[0]).name} produced excessive output") from None
    except BoundedCommandReadError:
        raise _ScaffoldCommandError(f"{Path(argv[0]).name} output could not be read") from None
    rendered = completed.output.decode("utf-8", errors="replace")
    if completed.output_truncated:
        rendered += "\n[command output truncated]"
    if completed.returncode not in allowed_exit_codes:
        raise _ScaffoldCommandError(
            f"{Path(argv[0]).name} failed with exit code {completed.returncode}"
        )
    return rendered


def _safe_git_argv(git: str, *arguments: str, hooks_dir: Path) -> list[str]:
    return [
        git,
        "--no-pager",
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.excludesFile={os.devnull}",
        "-c",
        f"core.attributesFile={os.devnull}",
        "-c",
        f"core.hooksPath={hooks_dir}",
        "-c",
        "commit.gpgSign=false",
        "-c",
        "tag.gpgSign=false",
        *arguments,
    ]


def _scaffold_directory_identity(path: Path) -> tuple[int, int]:
    try:
        identity = path.stat(follow_symlinks=False)
    except OSError:
        raise _ScaffoldCommandError("scaffold target is no longer available") from None
    if not stat.S_ISDIR(identity.st_mode):
        raise _ScaffoldCommandError("scaffold target identity changed")
    return identity.st_dev, identity.st_ino


def _require_safe_scaffold_parent(parent: Path) -> None:
    """Reject shared parents where another OS user can replace temp trees."""

    try:
        parent_stat = parent.stat(follow_symlinks=False)
    except OSError:
        raise _ScaffoldCommandError("scaffold parent is unavailable") from None
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise _ScaffoldCommandError("scaffold parent is not a directory")
    if _unsafe_shared_scaffold_parent_mode(parent_stat.st_mode, platform=os.name):
        raise _ScaffoldCommandError(
            "scaffold parent must not be group/world-writable unless it is sticky"
        )


def _unsafe_shared_scaffold_parent_mode(mode: int, *, platform: str) -> bool:
    """Interpret shared-write and sticky bits only where POSIX modes are authoritative."""

    if platform == "nt":
        return False
    shared_write = mode & (stat.S_IWGRP | stat.S_IWOTH)
    return bool(shared_write and not mode & stat.S_ISVTX)


def _run_scaffold_git_command(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    expected_directory_identity: tuple[int, int],
) -> str:
    if _scaffold_directory_identity(cwd) != expected_directory_identity:
        raise _ScaffoldCommandError("scaffold target identity changed")
    output = _run_scaffold_command(argv, cwd=cwd, env=env)
    if _scaffold_directory_identity(cwd) != expected_directory_identity:
        raise _ScaffoldCommandError("scaffold target identity changed")
    return output


def _require_scaffold_probe_evidence(
    output: str,
    *,
    command: str,
    required_fragments: tuple[str, ...],
) -> None:
    """Require positive, content-bound evidence from one dependency probe."""

    if any(fragment not in output for fragment in required_fragments):
        raise _ScaffoldCommandError(f"{command} semantic probe failed")


def _preflight_coding_commands(*, parent: Path) -> tuple[str, str]:
    """Verify the Git and ripgrep dialects used by the generated project."""

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for command in ("git", "rg"):
        executable = shutil.which(command)
        if executable is None:
            missing.append(command)
        else:
            resolved[command] = executable
    if missing:
        raise _ScaffoldCommandError("requires these commands on PATH: " + ", ".join(missing))

    with tempfile.TemporaryDirectory(prefix="cayu-scaffold-probe-", dir=parent) as raw:
        probe = Path(raw)
        hooks = probe / "hooks"
        hooks.mkdir()
        git_env = _sanitized_scaffold_git_environment(cwd=probe)
        _run_scaffold_command(
            _safe_git_argv(
                resolved["git"],
                "init",
                "-b",
                "main",
                f"--template={hooks}",
                hooks_dir=hooks,
            ),
            cwd=probe,
            env=git_env,
        )
        (probe / "probe.txt").write_text("cayu scaffold probe\n", encoding="utf-8")
        _run_scaffold_command(
            _safe_git_argv(
                resolved["git"],
                "add",
                "--force",
                "--",
                "probe.txt",
                hooks_dir=hooks,
            ),
            cwd=probe,
            env=git_env,
        )
        _run_scaffold_command(
            _safe_git_argv(
                resolved["git"],
                "-c",
                "user.name=Cayu Scaffold",
                "-c",
                "user.email=scaffold@cayu.local",
                "commit",
                "-m",
                "probe",
                hooks_dir=hooks,
            ),
            cwd=probe,
            env=git_env,
        )
        (probe / "probe.txt").write_text(
            "cayu scaffold semantic probe\n",
            encoding="utf-8",
        )
        (probe / "staged.txt").write_text(
            "cayu staged semantic probe\n",
            encoding="utf-8",
        )
        protected = probe / ".CaYu"
        protected.mkdir()
        (protected / "ignored.txt").write_text(
            "cayu scaffold semantic probe\n",
            encoding="utf-8",
        )
        _run_scaffold_command(
            _safe_git_argv(
                resolved["git"],
                "add",
                "--force",
                "--",
                "staged.txt",
                hooks_dir=hooks,
            ),
            cwd=probe,
            env=git_env,
        )
        git_probes = (
            (
                ("ls-files", "--cached", "-z", "--"),
                ("probe.txt\0", "staged.txt\0"),
            ),
            (
                ("status", "--porcelain=v1", "-z", "--untracked-files=normal", "--"),
                (" M probe.txt\0", "A  staged.txt\0"),
            ),
            (
                (
                    "diff",
                    "--no-color",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--unified=3",
                    "--",
                ),
                ("-cayu scaffold probe", "+cayu scaffold semantic probe"),
            ),
            (
                (
                    "diff",
                    "--no-color",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--cached",
                    "--unified=3",
                    "--",
                ),
                ("staged.txt", "+cayu staged semantic probe"),
            ),
            (
                (
                    "diff",
                    "--no-color",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--numstat",
                    "-z",
                    "--",
                ),
                ("1\t1\tprobe.txt\0",),
            ),
            (
                (
                    "diff",
                    "--no-color",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--cached",
                    "--numstat",
                    "-z",
                    "--",
                ),
                ("1\t0\tstaged.txt\0",),
            ),
        )
        for arguments, required_fragments in git_probes:
            output = _run_scaffold_command(
                _safe_git_argv(
                    resolved["git"],
                    *arguments,
                    hooks_dir=hooks,
                ),
                cwd=probe,
                env=git_env,
            )
            _require_scaffold_probe_evidence(
                output,
                command="git",
                required_fragments=required_fragments,
            )
        command_env = {key: os.environ[key] for key in _SAFE_SCAFFOLD_ENV_KEYS if key in os.environ}
        files_output = _run_scaffold_command(
            [
                resolved["rg"],
                "--no-config",
                "--hidden",
                "--no-require-git",
                "--sort",
                "path",
                "--files",
                "--null",
                "--max-filesize",
                "1048576",
                "--iglob",
                "!.git",
                "--iglob",
                "!**/.git",
                "--iglob",
                "!.git/**",
                "--iglob",
                "!**/.git/**",
                "--iglob",
                "!.cayu",
                "--iglob",
                "!**/.cayu",
                "--iglob",
                "!.cayu/**",
                "--iglob",
                "!**/.cayu/**",
                "--",
                ".",
            ],
            cwd=probe,
            env=command_env,
        )
        _require_scaffold_probe_evidence(
            files_output,
            command="rg",
            required_fragments=("probe.txt\0", "staged.txt\0"),
        )
        if ".CaYu/ignored.txt" in files_output:
            raise _ScaffoldCommandError("rg semantic probe failed")
        rg_probes = (
            (("--files-with-matches", "--null"), ("probe.txt\0",)),
            (
                (
                    "--with-filename",
                    "--line-number",
                    "--null",
                    "--field-match-separator",
                    "|",
                    "--max-columns",
                    "1024",
                    "--max-columns-preview",
                ),
                ("probe.txt", "cayu scaffold semantic probe"),
            ),
            (("--with-filename", "--count-matches", "--null"), ("probe.txt", "1")),
        )
        for mode_arguments, required_fragments in rg_probes:
            output = _run_scaffold_command(
                [
                    resolved["rg"],
                    "--no-config",
                    "--hidden",
                    "--no-require-git",
                    "--sort",
                    "path",
                    "--color",
                    "never",
                    "--max-filesize",
                    "1048576",
                    *mode_arguments,
                    "--ignore-case",
                    "--glob",
                    "probe.txt",
                    "--iglob",
                    "!.git",
                    "--iglob",
                    "!**/.git",
                    "--iglob",
                    "!.git/**",
                    "--iglob",
                    "!**/.git/**",
                    "--iglob",
                    "!.cayu",
                    "--iglob",
                    "!**/.cayu",
                    "--iglob",
                    "!.cayu/**",
                    "--iglob",
                    "!**/.cayu/**",
                    "--",
                    "CAYU SCAFFOLD SEMANTIC PROBE",
                    ".",
                ],
                cwd=probe,
                env=command_env,
            )
            _require_scaffold_probe_evidence(
                output,
                command="rg",
                required_fragments=required_fragments,
            )
            if ".CaYu/ignored.txt" in output:
                raise _ScaffoldCommandError("rg semantic probe failed")
    return resolved["git"], resolved["rg"]


def _publish_staged_scaffold(
    *,
    staging: Path,
    target: Path,
    expected_target_identity: tuple[int, int],
    target_mode: int,
) -> None:
    """Publish a complete private staging tree over the reserved empty target."""

    if _scaffold_directory_identity(target) != expected_target_identity:
        raise _ScaffoldCommandError("scaffold target identity changed")
    staging.chmod(target_mode)
    try:
        target.rmdir()
    except OSError:
        raise _ScaffoldCommandError("scaffold target is no longer empty") from None
    try:
        staging.rename(target)
    except OSError:
        # Restore only the empty reservation we removed. Never remove or alter a
        # concurrently created replacement in order to complete publication.
        with suppress(FileExistsError):
            target.mkdir(mode=target_mode)
        raise _ScaffoldCommandError("could not publish completed scaffold") from None


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
    ChatCompletionsProvider,
    ModelProvider,
    OpenAIProvider,
    OpenAISubscriptionProvider,
    ScriptedModelProvider,
    SessionStore,
    SQLiteSessionStore,
    SQLiteTaskStore,
    public_authority_alias_codec_from_environment,
    TaskStore,
)

from agents.agent import AGENT
from configuration import configured_provider_choice

# Generated tool-backed slices add their imports and registrations here.
# <cayu:generated-imports>
# </cayu:generated-imports>


class _ScaffoldPlaceholderProvider(ScriptedModelProvider):
    """Credential-free placeholder rejected by every runtime entry point."""

    def __init__(self, *, name: str, setup_error: str) -> None:
        super().__init__([], name=name)
        self._setup_error = setup_error

    def preflight_model_target(self, *, model: str) -> None:
        del model
        raise RuntimeError(self._setup_error)


def configured_provider() -> ModelProvider:
    """Construct only the explicitly selected provider.

    Credential variables authenticate the selected provider; they never choose
    one. A same-name scripted placeholder keeps inspection and hermetic proof
    credential-free while ``run.py`` rejects it before live execution.
    """

    choice = configured_provider_choice()
    if choice is None:
        return _ScaffoldPlaceholderProvider(
            name="unconfigured",
            setup_error=(
                "no provider is selected; set CAYU_PROVIDER to openai, anthropic, "
                "openrouter, or openai-subscription (credentials do not select a provider)"
            ),
        )
    if choice == "openai-subscription":
        return OpenAISubscriptionProvider()
    if choice == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        return (
            OpenAIProvider(api_key=api_key)
            if api_key
            else _ScaffoldPlaceholderProvider(
                name="openai",
                setup_error="provider 'openai' is selected but OPENAI_API_KEY is not set",
            )
        )
    if choice == "openrouter":
        router_metadata_enabled = _openrouter_router_metadata_enabled()
        model = (os.environ.get("CAYU_MODEL") or "").strip()
        if not model:
            return _ScaffoldPlaceholderProvider(
                name="openrouter",
                setup_error="provider 'openrouter' requires an explicit CAYU_MODEL model slug",
            )
        api_key = os.environ.get("OPENROUTER_API_KEY")
        return (
            ChatCompletionsProvider(
                name="openrouter",
                api_key=api_key,
                api_key_env="OPENROUTER_API_KEY",
                base_url="https://openrouter.ai/api/v1",
                openrouter_http_referer=os.environ.get("OPENROUTER_HTTP_REFERER"),
                openrouter_app_title=os.environ.get("OPENROUTER_APP_TITLE"),
                openrouter_router_metadata=router_metadata_enabled,
            )
            if api_key
            else _ScaffoldPlaceholderProvider(
                name="openrouter",
                setup_error=(
                    "provider 'openrouter' is selected but OPENROUTER_API_KEY is not set"
                ),
            )
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    return (
        AnthropicProvider(api_key=api_key)
        if api_key
        else _ScaffoldPlaceholderProvider(
            name="anthropic",
            setup_error="provider 'anthropic' is selected but ANTHROPIC_API_KEY is not set",
        )
    )


def validate_run_configuration(app: CayuApp, agent_name: str) -> None:
    """Run the same target preflight used by every runtime entry point."""

    manifest_agent = next(
        agent for agent in app.describe().agents if agent.name == agent_name
    )
    if manifest_agent.resolved_provider is None:
        raise RuntimeError(
            f"agent {agent_name!r} does not resolve to exactly one model provider"
        )
    provider = app.get_provider(manifest_agent.resolved_provider)
    provider.preflight_model_target(model=manifest_agent.model)


def _openrouter_router_metadata_enabled() -> bool:
    value = os.environ.get("OPENROUTER_ROUTER_METADATA")
    if value is None or value.lower() == "disabled":
        return False
    if value.lower() == "enabled":
        return True
    raise RuntimeError(
        "OPENROUTER_ROUTER_METADATA must be 'enabled' or 'disabled' when set"
    )


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
        session_store=(
            session_store
            if session_store is not None
            else SQLiteSessionStore(
                "data/cayu.db",
                public_authority_alias_codec=public_authority_alias_codec_from_environment(),
            )
        ),
        task_store=(
            task_store if task_store is not None else SQLiteTaskStore("data/cayu.db")
        ),
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
_SUPPORTED_PROVIDERS = {"openai", "anthropic", "openrouter", "openai-subscription"}
_PROVIDER_NAMES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "openrouter": "openrouter",
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
    override = (os.environ.get("CAYU_MODEL") or "").strip()
    if override:
        return override
    selected = configured_provider_choice()
    if selected == "openrouter":
        # Inspection and hermetic tests remain credential-free. Live run.py
        # validation rejects this sentinel until CAYU_MODEL is explicit.
        return "openrouter-model-unconfigured"
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
    name="__AGENT_NAME__",
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
                agent_name="__AGENT_NAME__",
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
                    agent_name="__AGENT_NAME__",
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
dev = ["cayu[server]>=__CAYU_VERSION__", "pytest"]

[tool.cayu]
factory = "app:build_app"
eval_target = "evals.agent:build_eval"

[tool.cayu.session_store]
backend = "sqlite"
path = "data/cayu.db"

[tool.pytest.ini_options]
pythonpath = ["."]
"""

_PROVIDER_GUIDE_POINTER = """OpenRouter is a first-class scaffold choice. Fireworks, Baseten, OpenCode Go,
and other compatible endpoints work through Cayu's generic adapter. Run
`uv run cayu guide providers#compatible-chat-completions` for exact setup."""

_README = """# __PROJECT_NAME__

A model-only Cayu agent scaffold. It starts with one agent, one deterministic
runtime test, and one output eval. Its registered agent identity is
`__AGENT_NAME__`. Add capabilities only when the job needs them.

## Application structure

Describe the requested job in `agents/agent.py`; update `tests/test_agent.py`
and `evals/agent.py` to prove that behavior. The project factory is `build_app()`
in `app.py`. Run `cayu guide anatomy` for its lifecycle contract.

Run `uv run cayu guide authoring#cayu-map` to select another concept only when
the requested behavior requires it. For durable operational changes, start with
`uv run cayu guide durable-operations`; it covers propose, authorize, act once,
verify, inspect, and recover. `uv run cayu guide references` contains the
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

## Inspect with the local control plane

Cayu's packaged developer/operator control plane reads the same durable stores
as the project commands. Start it in a separate terminal:

```bash
uv run cayu serve --dev
```

Then open `http://127.0.0.1:8000/cayu/`. The explicit `--dev` flag enables
unauthenticated trusted-local access only. It does not make the control plane
the application's end-user UI or configure a production deployment.
Project serving also derives Evals project/release identity and uses this
project's durable `data/cayu.db` store automatically. With the normal live
provider configured, open a completed or failed simple session, choose
**Evaluate**, review its assertions, save or approve the captured result, and
start one bounded fresh trial. The generated target reuses the registered agent
and its ordinary runtime policy; no Evals-specific Python configuration is
required.
Never mount it with unauthenticated open access on a public listener;
client-IP checks are not authentication. Public or deployed control-plane
access requires an authenticated access policy.

## Run with a live provider

Provider intent is explicit. This scaffold defaults to `__PROVIDER_DISPLAY__`;
override it with `CAYU_PROVIDER=openai`, `anthropic`, `openrouter`, or
`openai-subscription`. API-key variables authenticate that choice and never
select it automatically.

__PROVIDER_GUIDE_POINTER__

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

OpenRouter (the model slug is always explicit):

```bash
export CAYU_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-...
export CAYU_MODEL=vendor/model
uv run python run.py --message "YOUR REQUEST"
```

Optional `OPENROUTER_HTTP_REFERER` and `OPENROUTER_APP_TITLE` values add
OpenRouter attribution headers. Set `OPENROUTER_ROUTER_METADATA=enabled` to
retain only bounded routing evidence; free-form pipeline and attempt data is
never persisted. Put OpenRouter routing controls such as `provider.order`,
`provider.allow_fallbacks`, `provider.require_parameters`, `provider.zdr`, and
`provider.data_collection` in `AgentSpec.provider_options["openrouter"]`.

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

The registered agent identity is `__AGENT_NAME__`.

Edit the existing agent, test, and eval to implement the user's first requested
job. Do not retain the starter and add a second agent. Tools are optional: add
one only for a real capability outside the model, such as reading a repository
or calling an API. Do not create echo, pass-through, or placeholder tools.

Use the Cayu Map to choose only the concepts the job needs:
`uv run cayu guide authoring#cayu-map`. If the job observes, proposes, authorizes,
executes, verifies, or recovers an operational change, read the runnable paved
path first: `uv run cayu guide durable-operations`.

If another capability is required, use the smallest package-shipped reference
from `uv run cayu guide references`.

__PROVIDER_GUIDE_POINTER__

This scaffold is for local development. Deployment is a separate task.
If the requested application is public or multi-user, regenerate with
`cayu new NAME --template service` or adopt Cayu's maintained service contract;
do not improvise product authorization around raw Cayu routes.

## Project commands

- Setup: `uv sync --extra dev`.
- Application contract: `uv run cayu guide anatomy`.
- Authoring details: `uv run cayu guide authoring`.
- Inspect/check: `uv run cayu inspect --json` and `uv run cayu check --json`.
- Hermetic proof: `uv run pytest` and `uv run cayu eval run`.
- Local developer/operator control plane: run `uv run cayu serve --dev` in a separate
  terminal and open `http://127.0.0.1:8000/cayu/`. This is not the application's
  end-user UI or a production server configuration.
- Never mount it with `OpenAccess()` on a public listener.
- Client-IP and forwarded-header checks are not authentication. Use
  `AuthenticatedAccess(...)` for any public or deployed control-plane surface.
- Live execution: `uv run python run.py --message "USER REQUEST"` after configuring a
  provider in `app.configured_provider()`.

Use public `cayu` imports and public CLI JSON only. Do not depend on Cayu source,
private symbols, or import-time application construction.

If the job truly needs a tool, read `cayu guide tool-effects`; every tool must
declare `ToolEffect`, and effect metadata does not authorize execution. A
`ScriptedModelProvider` proves handling of predetermined calls, not prompt
comprehension or live model behavior.

For the starter's first real tool, run
`uv run cayu generate tool TOOL_NAME --agent __AGENT_NAME__ --effect EFFECT`.
Then replace the generated sample schema, implementation, test, and eval with
domain behavior; `cayu check` keeps the tracer-bullet warning active until the
explicit authoring marker is removed.
"""

_SERVICE_PY = '''"""Maintained public-service factory for __PROJECT_NAME__."""

from __future__ import annotations

import hmac
import json
import os
import re

from fastapi import HTTPException, Request

from cayu import ModelProvider, SessionStore, TaskStore
from cayu.server import (
    AuthenticatedAccess,
    AuthenticatedProductAccess,
    DevelopmentProductAccess,
    OpenAccess,
    OperatorAccess,
    PlaceholderOperatorAccess,
    PlaceholderProductAccess,
    ProductOperationStore,
    ProductPrincipal,
    ProjectControlPlaneContext,
    ServerAccessConfig,
    ServiceMode,
    create_agent_service,
)

from app import build_app
from product_store import SQLiteProductOperationStore

_BEARER_TOKEN_RE = re.compile(r"[-A-Za-z0-9._~+/]+=*", flags=re.ASCII)
_MAX_BEARER_TOKEN_CHARS = 4096


def _validated_bearer_token(value: object) -> str | None:
    if (
        type(value) is not str
        or len(value) > _MAX_BEARER_TOKEN_CHARS
        or _BEARER_TOKEN_RE.fullmatch(value) is None
    ):
        return None
    return value


def _bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer":
        return None
    return _validated_bearer_token(token)


async def _development_product_auth(request: Request) -> ProductPrincipal:
    tenant = request.headers.get("x-cayu-dev-tenant")
    subject = request.headers.get("x-cayu-dev-subject")
    if not tenant or not subject:
        raise HTTPException(
            status_code=401, detail="Development identity headers required."
        )
    return ProductPrincipal(tenant_id=tenant, subject_id=subject)


def _configured_product_principals() -> dict[str, ProductPrincipal] | None:
    raw = os.environ.get("PRODUCT_AUTH_TOKENS_JSON")
    try:
        configured = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return None
    if type(configured) is not dict or not configured:
        return None
    principals: dict[str, ProductPrincipal] = {}
    try:
        for raw_token, raw_principal in configured.items():
            token = _validated_bearer_token(raw_token)
            if token is None or type(raw_principal) is not dict:
                return None
            principals[token] = ProductPrincipal.model_validate(raw_principal)
    except (TypeError, ValueError):
        return None
    return principals


def _production_product_access():
    configured = _configured_product_principals()
    if configured is None:
        return PlaceholderProductAccess()

    async def authenticate(request: Request) -> ProductPrincipal:
        token = _bearer_token(request)
        matched = next(
            (
                principal
                for candidate, principal in configured.items()
                if token is not None and hmac.compare_digest(token, candidate)
            ),
            None,
        )
        if matched is None:
            raise HTTPException(status_code=401, detail="Authentication required.")
        return matched

    return AuthenticatedProductAccess(dependency=authenticate)


def _production_operator_access() -> ServerAccessConfig | PlaceholderOperatorAccess:
    configured_token = _validated_bearer_token(
        os.environ.get("CAYU_OPERATOR_BEARER_TOKEN")
    )
    if configured_token is None:
        return PlaceholderOperatorAccess()
    product_principals = _configured_product_principals()
    if product_principals is not None and configured_token in product_principals:
        # Customer credentials must never grant operator-plane access.
        return PlaceholderOperatorAccess()

    async def authenticate(request: Request):
        token = _bearer_token(request)
        if token is None or not hmac.compare_digest(token, configured_token):
            raise HTTPException(
                status_code=401, detail="Operator authentication required."
            )
        return {"subject": "configured-operator"}

    return AuthenticatedAccess(dependency=authenticate)


def build_service(
    *,
    mode: ServiceMode,
    project_context: ProjectControlPlaneContext | None = None,
    provider: ModelProvider | None = None,
    session_store: SessionStore | None = None,
    task_store: TaskStore | None = None,
    product_store: ProductOperationStore | None = None,
    product_access=None,
    operator_access: OperatorAccess | None = None,
):
    """Build the one service used by serving, checks, docs, and security tests."""

    mode = ServiceMode(mode)
    app = build_app(
        provider=provider,
        session_store=session_store,
        task_store=task_store,
    )
    selected_product_access = (
        product_access
        if product_access is not None
        else (
            DevelopmentProductAccess(dependency=_development_product_auth)
            if mode is ServiceMode.DEVELOPMENT
            else _production_product_access()
        )
    )
    selected_operator_access = (
        operator_access
        if operator_access is not None
        else (
            OpenAccess()
            if mode is ServiceMode.DEVELOPMENT
            else _production_operator_access()
        )
    )
    selected_product_store = (
        product_store
        if product_store is not None
        else SQLiteProductOperationStore("data/product.db")
    )
    return create_agent_service(
        app,
        agent_name="__AGENT_NAME__",
        mode=mode,
        project_context=project_context,
        product_access=selected_product_access,
        operator_access=selected_operator_access,
        product_store=selected_product_store,
    )
'''

_PRODUCT_STORE_PY = '''"""Application-owned tenant/resource identity storage.

This store, not Cayu labels, metadata, or runtime identifiers, is the product
authorization boundary. SQLite is durable for the generated single-process
service; replace it with an equivalently atomic shared store before scaling to
multiple service processes. Execution claims use store-owned lease time, and
terminal updates are conditional on the current claim so only one valid owner
can settle an operation. Content-bound publication receipts are written before
Cayu session completion and are required for a completed product result. The
private session-id lookup exists only for trusted continuation hooks; product
HTTP reads remain tenant-qualified by public id.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from cayu.server import (
    ProductExecutionClaimLost,
    ProductIdempotencyConflict,
    ProductOperation,
    ProductOperationExecutionClaim,
    ProductOperationReservation,
    ProductOperationSettlementConflict,
    ProductRecoveryStatus,
    ProductResultReceipt,
    ProductResultReceiptConflict,
    ServiceIdentityStoreKind,
)


class SQLiteProductOperationStore:
    category = ServiceIdentityStoreKind.DURABLE

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._memory_connection: sqlite3.Connection | None = None
        self._memory_lock = RLock()
        self.category = (
            ServiceIdentityStoreKind.DEVELOPMENT
            if str(self.path) == ":memory:"
            else ServiceIdentityStoreKind.DURABLE
        )
        if self.category is ServiceIdentityStoreKind.DEVELOPMENT:
            self._memory_connection = sqlite3.connect(
                ":memory:", timeout=30, check_same_thread=False
            )
            self._memory_connection.row_factory = sqlite3.Row
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS product_operations (
                    work_id TEXT PRIMARY KEY,
                    public_id TEXT NOT NULL UNIQUE,
                    tenant_id TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_fingerprint TEXT NOT NULL,
                    session_id TEXT NOT NULL UNIQUE,
                    task_id TEXT NOT NULL UNIQUE,
                    request_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    result_receipt TEXT,
                    recovery_status TEXT,
                    execution_claim_id TEXT,
                    execution_claim_expires_at INTEGER
                )"""
            )
            columns = {
                row[1]: (str(row[2]).upper(), int(row[3]))
                for row in connection.execute("PRAGMA table_info(product_operations)")
            }
            if columns.get("subject_id") != ("TEXT", 1):
                raise RuntimeError(
                    "The product operation store predates required invocation "
                    "provenance. Recreate the prerelease product database."
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory_connection is not None:
            with self._memory_lock, self._memory_connection as connection:
                yield connection
            return
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    async def reserve(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        idempotency_key: str,
        request_fingerprint: str,
        public_id: str,
        work_id: str,
        session_id: str,
        task_id: str,
        request_text: str,
    ) -> ProductOperationReservation:
        return await asyncio.to_thread(
            self._reserve_sync,
            tenant_id=tenant_id,
            subject_id=subject_id,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            public_id=public_id,
            work_id=work_id,
            session_id=session_id,
            task_id=task_id,
            request_text=request_text,
        )

    def _reserve_sync(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        idempotency_key: str,
        request_fingerprint: str,
        public_id: str,
        work_id: str,
        session_id: str,
        task_id: str,
        request_text: str,
    ) -> ProductOperationReservation:
        requested = ProductOperation(
            tenant_id=tenant_id,
            subject_id=subject_id,
            public_id=public_id,
            work_id=work_id,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            session_id=session_id,
            task_id=task_id,
            request_text=request_text,
            status="pending",
            result=None,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM product_operations WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if row is not None:
                operation = self._operation(row)
                if (
                    operation.tenant_id != tenant_id
                    or operation.request_fingerprint != request_fingerprint
                ):
                    raise ProductIdempotencyConflict
                return ProductOperationReservation(operation=operation, created=False)
            connection.execute(
                """INSERT INTO product_operations (
                    work_id, public_id, tenant_id, subject_id, idempotency_key,
                    request_fingerprint, session_id, task_id, request_text, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (
                    requested.work_id,
                    requested.public_id,
                    requested.tenant_id,
                    requested.subject_id,
                    requested.idempotency_key,
                    requested.request_fingerprint,
                    requested.session_id,
                    requested.task_id,
                    requested.request_text,
                ),
            )
        return ProductOperationReservation(operation=requested, created=True)

    async def find(self, *, tenant_id: str, public_id: str) -> ProductOperation | None:
        return await asyncio.to_thread(
            self._find_sync,
            tenant_id=tenant_id,
            public_id=public_id,
        )

    def _find_sync(self, *, tenant_id: str, public_id: str) -> ProductOperation | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM product_operations
                   WHERE tenant_id = ? AND public_id = ?""",
                (tenant_id, public_id),
            ).fetchone()
        return None if row is None else self._operation(row)

    async def find_by_session_id(self, *, session_id: str) -> ProductOperation | None:
        return await asyncio.to_thread(
            self._find_by_session_id_sync,
            session_id=session_id,
        )

    def _find_by_session_id_sync(self, *, session_id: str) -> ProductOperation | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM product_operations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return None if row is None else self._operation(row)

    async def claim_execution(
        self,
        *,
        work_id: str,
        claim_id: str,
        lease_seconds: int,
    ) -> ProductOperationExecutionClaim | None:
        return await asyncio.to_thread(
            self._claim_execution_sync,
            work_id=work_id,
            claim_id=claim_id,
            lease_seconds=lease_seconds,
        )

    def _claim_execution_sync(
        self,
        *,
        work_id: str,
        claim_id: str,
        lease_seconds: int,
    ) -> ProductOperationExecutionClaim | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM product_operations WHERE work_id = ?", (work_id,)
            ).fetchone()
            if row is None:
                return None
            operation = self._operation(row)
            if operation.status != "pending":
                return ProductOperationExecutionClaim(
                    operation=operation, acquired=False
                )
            changed = connection.execute(
                """UPDATE product_operations
                   SET execution_claim_id = ?,
                       execution_claim_expires_at = MAX(
                           COALESCE(execution_claim_expires_at, 0),
                           CAST(strftime('%s', 'now') AS INTEGER) + ?
                       )
                   WHERE work_id = ?
                     AND status = 'pending'
                     AND (
                         execution_claim_id = ?
                         OR execution_claim_id IS NULL
                         OR execution_claim_expires_at IS NULL
                         OR execution_claim_expires_at <=
                             CAST(strftime('%s', 'now') AS INTEGER)
                     )""",
                (claim_id, lease_seconds, work_id, claim_id),
            ).rowcount
            row = connection.execute(
                "SELECT * FROM product_operations WHERE work_id = ?", (work_id,)
            ).fetchone()
        if row is None:
            raise RuntimeError("Product work disappeared during execution claim.")
        return ProductOperationExecutionClaim(
            operation=self._operation(row),
            acquired=changed == 1,
        )

    async def heartbeat_execution(
        self,
        *,
        work_id: str,
        claim_id: str,
        lease_seconds: int,
    ) -> bool:
        return await asyncio.to_thread(
            self._heartbeat_execution_sync,
            work_id=work_id,
            claim_id=claim_id,
            lease_seconds=lease_seconds,
        )

    def _heartbeat_execution_sync(
        self,
        *,
        work_id: str,
        claim_id: str,
        lease_seconds: int,
    ) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """UPDATE product_operations
                   SET execution_claim_expires_at = MAX(
                       COALESCE(execution_claim_expires_at, 0),
                       CAST(strftime('%s', 'now') AS INTEGER) + ?
                   )
                   WHERE work_id = ?
                     AND status = 'pending'
                     AND execution_claim_id = ?""",
                (lease_seconds, work_id, claim_id),
            ).rowcount
            if changed == 1:
                return True
            row = connection.execute(
                "SELECT status, execution_claim_id FROM product_operations "
                "WHERE work_id = ?",
                (work_id,),
            ).fetchone()
        return (
            row is not None
            and row["status"] != "pending"
            and row["execution_claim_id"] == claim_id
        )

    async def release_execution(
        self,
        *,
        work_id: str,
        claim_id: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._release_execution_sync,
            work_id=work_id,
            claim_id=claim_id,
        )

    def _release_execution_sync(
        self,
        *,
        work_id: str,
        claim_id: str,
    ) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, execution_claim_id FROM product_operations "
                "WHERE work_id = ?",
                (work_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    "Product work disappeared during execution-claim release."
                )
            if row["status"] != "pending" or row["execution_claim_id"] != claim_id:
                return row["status"] == "pending" and row["execution_claim_id"] is None
            changed = connection.execute(
                """UPDATE product_operations
                   SET execution_claim_id = NULL,
                       execution_claim_expires_at = NULL
                   WHERE work_id = ?
                     AND status = 'pending'
                     AND execution_claim_id = ?""",
                (work_id, claim_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError(
                    "Product execution claim changed during atomic release."
                )
            return True

    async def record_result_receipt(
        self,
        *,
        work_id: str,
        claim_id: str,
        receipt: ProductResultReceipt,
    ) -> ProductResultReceipt:
        return await asyncio.to_thread(
            self._record_result_receipt_sync,
            work_id=work_id,
            claim_id=claim_id,
            receipt=receipt,
        )

    def _record_result_receipt_sync(
        self,
        *,
        work_id: str,
        claim_id: str,
        receipt: ProductResultReceipt,
    ) -> ProductResultReceipt:
        receipt = ProductResultReceipt.model_validate(receipt.model_dump(mode="python"))
        encoded = json.dumps(
            receipt.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM product_operations WHERE work_id = ?", (work_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    "Product work disappeared during result publication."
                )
            operation = self._operation(row)
            if operation.status != "pending" or row["execution_claim_id"] != claim_id:
                if (
                    operation.status != "pending"
                    and row["execution_claim_id"] == claim_id
                    and operation.result_receipt == receipt
                ):
                    return receipt
                raise ProductExecutionClaimLost(
                    "Product execution ownership was lost before result publication."
                )
            if (
                receipt.work_id != operation.work_id
                or receipt.public_id != operation.public_id
                or receipt.request_fingerprint != operation.request_fingerprint
                or receipt.session_id != operation.session_id
                or receipt.task_id != operation.task_id
            ):
                raise ProductResultReceiptConflict(
                    "Result receipt does not belong to this product operation."
                )
            if operation.result_receipt is not None:
                if operation.result_receipt == receipt:
                    return operation.result_receipt
                if receipt.source_event_sequence <= (
                    operation.result_receipt.source_event_sequence
                ):
                    raise ProductResultReceiptConflict(
                        "Product work already has newer result evidence."
                    )
            changed = connection.execute(
                """UPDATE product_operations
                   SET result_receipt = ?, recovery_status = NULL
                   WHERE work_id = ?
                     AND status = 'pending'
                     AND result_receipt IS ?
                     AND execution_claim_id = ?""",
                (encoded, work_id, row["result_receipt"], claim_id),
            ).rowcount
            if changed != 1:
                raise ProductExecutionClaimLost(
                    "Product execution ownership was lost before result publication."
                )
        return receipt

    async def record_recovery_status(
        self,
        *,
        work_id: str,
        claim_id: str,
        recovery_status: ProductRecoveryStatus,
    ) -> ProductOperation:
        return await asyncio.to_thread(
            self._record_recovery_status_sync,
            work_id=work_id,
            claim_id=claim_id,
            recovery_status=recovery_status,
        )

    def _record_recovery_status_sync(
        self,
        *,
        work_id: str,
        claim_id: str,
        recovery_status: ProductRecoveryStatus,
    ) -> ProductOperation:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM product_operations WHERE work_id = ?", (work_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    "Product work disappeared during recovery reporting."
                )
            operation = self._operation(row)
            if operation.status != "pending" or row["execution_claim_id"] != claim_id:
                raise ProductExecutionClaimLost(
                    "Product execution ownership was lost before recovery reporting."
                )
            changed = connection.execute(
                """UPDATE product_operations
                   SET recovery_status = ?
                   WHERE work_id = ?
                     AND status = 'pending'
                     AND execution_claim_id = ?""",
                (recovery_status, work_id, claim_id),
            ).rowcount
            if changed != 1:
                raise ProductExecutionClaimLost(
                    "Product execution ownership was lost before recovery reporting."
                )
            row = connection.execute(
                "SELECT * FROM product_operations WHERE work_id = ?", (work_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    "Product work disappeared during recovery reporting."
                )
            return self._operation(row)

    async def finish(
        self,
        *,
        work_id: str,
        claim_id: str,
        status: str,
        result: str | None,
    ) -> ProductOperation:
        return await asyncio.to_thread(
            self._finish_sync,
            work_id=work_id,
            claim_id=claim_id,
            status=status,
            result=result,
        )

    def _finish_sync(
        self,
        *,
        work_id: str,
        claim_id: str,
        status: str,
        result: str | None,
    ) -> ProductOperation:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM product_operations WHERE work_id = ?", (work_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Product work disappeared during completion.")
            operation = self._operation(row)
            receipt = operation.result_receipt
            if status == "completed" and (
                receipt is None
                or receipt.publication_status != "completed"
                or receipt.result != result
            ):
                raise ProductOperationSettlementConflict(
                    "Completed product work does not match its result receipt."
                )
            if status == "failed" and result is not None:
                raise ProductOperationSettlementConflict(
                    "Failed product work cannot persist a public result."
                )
            if operation.status != "pending":
                if (
                    row["execution_claim_id"] == claim_id
                    and operation.status == status
                    and operation.result == result
                ):
                    return operation
                if row["execution_claim_id"] != claim_id:
                    raise ProductExecutionClaimLost(
                        "Product execution ownership was lost before completion."
                    )
                raise ProductOperationSettlementConflict(
                    "Product work already has a different terminal result."
                )
            if row["execution_claim_id"] != claim_id:
                raise ProductExecutionClaimLost(
                    "Product execution ownership was lost before completion."
                )
            changed = connection.execute(
                """UPDATE product_operations
                   SET status = ?, result = ?, recovery_status = NULL,
                       execution_claim_expires_at = NULL
                   WHERE work_id = ?
                     AND status = 'pending'
                     AND execution_claim_id = ?""",
                (status, result, work_id, claim_id),
            ).rowcount
            if changed != 1:
                raise ProductExecutionClaimLost(
                    "Product execution ownership was lost before completion."
                )
            row = connection.execute(
                "SELECT * FROM product_operations WHERE work_id = ?", (work_id,)
            ).fetchone()
        if row is None:
            raise RuntimeError("Product work disappeared during completion.")
        return self._operation(row)

    @staticmethod
    def _operation(row: sqlite3.Row) -> ProductOperation:
        fields = dict(row)
        fields.pop("execution_claim_id", None)
        fields.pop("execution_claim_expires_at", None)
        raw_receipt = fields.get("result_receipt")
        fields["result_receipt"] = (
            None if raw_receipt is None else json.loads(raw_receipt)
        )
        return ProductOperation(**fields)
'''

_SERVICE_SECURITY_TEST_PY = """from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from cayu import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    AgentSpec,
    CayuApp,
    Event,
    EventType,
    InMemorySessionStore,
    InMemoryTaskStore,
    InteractionStatus,
    InteractionSummaryEvidence,
    InvocationOrigin,
    InvocationOriginTrust,
    LoopPolicy,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SecretRedactor,
    SessionIdentity,
    SessionInvocationBinding,
    SessionStatus,
    SQLiteSessionStore,
    SQLiteTaskStore,
    TaskCreate,
    TaskExecutionSource,
    TaskInvocationSnapshot,
    TaskStatus,
    ToolCapabilityCeiling,
    current_runtime_build_provenance,
)
from cayu.runtime import ActiveInvocationExecutionProfile, AdmitInvocationCommand
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime import _execution_profile_admission as execution_profile_admission
from cayu.runtime._invocation_lifecycle import invocation_checkpoint_state_sha256
from cayu.runtime.sessions import run_request_with_task_invocation
from cayu.runtime.tasks import task_create_with_runtime_invocation
from cayu.server import (
    AuthenticatedAccess,
    AuthenticatedProductAccess,
    BasicAuth,
    ProductExecutionClaimLost,
    ProductOperation,
    ProductOperationSettlementConflict,
    ProductPrincipal,
    ProductResultReceipt,
    ProductResultReceiptConflict,
    ServiceMode,
    create_agent_service,
)

from product_store import SQLiteProductOperationStore
from service import build_service


def product_request_fingerprint(request_text: str, *, agent_name: str) -> str:
    encoded = json.dumps(
        {
            "agent_name": agent_name,
            "request": request_text,
            "schema_version": 1,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def product_task_create(operation: ProductOperation, *, agent_name: str) -> TaskCreate:
    return task_create_with_runtime_invocation(
        TaskCreate(
            task_id=operation.task_id,
            type="public_agent_operation",
            session_id=operation.session_id,
            assigned_agent_name=agent_name,
        ),
        source=TaskExecutionSource.PRODUCT_OPERATION,
        verified_origin=InvocationOrigin(
            trust=InvocationOriginTrust.SERVER_VERIFIED,
            subject=operation.subject_id,
            tenant=operation.tenant_id,
        ),
    )


def profiled_session_identity(
    app: CayuApp,
    *,
    agent_name: str,
    provider_name: str,
    model: str,
    invocation_loop_policies: tuple[LoopPolicy, ...] = (),
) -> SessionIdentity:
    # Mirror the runtime-owned identity for this manually seeded resume fixture.
    runtime_version = version("cayu")
    runtime_build_provenance = current_runtime_build_provenance()
    registered_agent = app._agents[agent_name]
    engine = app._session_engine
    invocation_loop_policy_identities = tuple(
        policy.execution_profile_identity for policy in invocation_loop_policies
    )
    return SessionIdentity(
        provider_name=provider_name,
        model=model,
        runtime_name="cayu",
        runtime_version=runtime_version,
        runtime_build_provenance=runtime_build_provenance,
        execution_profile=execution_profile_admission.resolve_execution_profile_identity(
            registered_agent=registered_agent,
            runtime_name="cayu",
            runtime_version=runtime_version,
            runtime_build_provenance=runtime_build_provenance,
            provider_name=provider_name,
            model=model,
            durable_system_prompt=registered_agent.spec.system_prompt,
            redactor=app._secret_redactor,
            process_identity=app._execution_profile_process_identity,
            runtime_hooks=engine._runtime_hooks,
            loop_policies=engine._loop_policies,
            loop_policy_identities=engine._loop_policy_execution_profile_identities,
            invocation_loop_policies=invocation_loop_policies,
            invocation_loop_policy_identities=invocation_loop_policy_identities,
            invocation_loop_policy_instance_identities=(
                engine._request_loop_policy_instance_identities(
                    invocation_loop_policies
                )
            ),
            registered_provider=app._providers.get(provider_name),
        ),
    )


async def customer_auth(request: Request) -> ProductPrincipal:
    principals = {
        "Bearer customer-a": ProductPrincipal(tenant_id="tenant-a", subject_id="alice"),
        "Bearer customer-b": ProductPrincipal(tenant_id="tenant-b", subject_id="bob"),
    }
    principal = principals.get(request.headers.get("authorization", ""))
    if principal is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return principal


def assembled_service(tmp_path, provider=None):
    provider = provider or ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("allow-listed answer"),
            ModelStreamEvent.completed(
                {"finish_reason": "stop", "provider_body": "provider-body-sentinel"}
            ),
        ]
    )
    store = SQLiteProductOperationStore(str(tmp_path / "product.db"))
    service = build_service(
        mode=ServiceMode.PRODUCTION,
        provider=provider,
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
        product_store=store,
        product_access=AuthenticatedProductAccess(dependency=customer_auth),
        operator_access=AuthenticatedAccess(
            dependency=BasicAuth(username="operator", password="operator-secret")
        ),
    )
    return service, store, provider


def test_anonymous_denial_and_authorized_happy_path(tmp_path) -> None:
    service, _store, _provider = assembled_service(tmp_path)
    client = TestClient(service.asgi_app)
    assert client.post("/api/operations", json={"request": "work"}).status_code == 401
    response = client.post(
        "/api/operations",
        headers={"Authorization": "Bearer customer-a", "Idempotency-Key": "one"},
        json={"request": "work"},
    )
    assert response.status_code == 201
    assert response.json()["status"] == "completed"
    assert response.headers["cache-control"] == "private, no-store"
    assert client.get(f"/api/operations/{response.json()['id']}").status_code == 401


def test_invalid_request_is_rejected_without_a_durable_reservation(tmp_path) -> None:
    service, store, provider = assembled_service(tmp_path)
    client = TestClient(service.asgi_app, raise_server_exceptions=False)
    response = client.post(
        "/api/operations",
        headers={"Authorization": "Bearer customer-a", "Idempotency-Key": "invalid"},
        json={"request": "   "},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid product request."}
    assert response.headers["cache-control"] == "private, no-store"

    duplicate = client.post(
        "/api/operations",
        headers={
            "Authorization": "Bearer customer-a",
            "Content-Type": "application/json",
            "Idempotency-Key": "duplicate",
        },
        content=b'{"request":"first","request":"second"}',
    )
    assert duplicate.status_code == 400
    assert duplicate.json() == {"detail": "Invalid product request."}

    oversized = client.post(
        "/api/operations",
        headers={
            "Authorization": "Bearer customer-a",
            "Content-Type": "application/json",
            "Idempotency-Key": "oversized",
        },
        content=b'{"request":"' + (b"x" * (1024 * 1024)) + b'"}',
    )
    assert oversized.status_code == 413
    assert oversized.json() == {
        "detail": "Product request exceeds the server byte limit."
    }
    assert oversized.headers["cache-control"] == "private, no-store"
    assert provider.requests == []
    with store._connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM product_operations").fetchone()[0]
            == 0
        )

    with pytest.raises(ValueError):
        asyncio.run(
            store.reserve(
                tenant_id="tenant-a",
                subject_id="test-subject",
                idempotency_key="invalid-direct",
                request_fingerprint="invalid-fingerprint",
                public_id="op_invalid",
                work_id="work_invalid",
                session_id="session_invalid",
                task_id="task_invalid",
                request_text="   ",
            )
        )
    with store._connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM product_operations").fetchone()[0]
            == 0
        )


@pytest.mark.parametrize(
    "configured_product_tokens",
    [
        '{"customer-token":null}',
        '{"tøk":{"tenant_id":"tenant-a","subject_id":"alice"}}',
    ],
)
def test_malformed_product_auth_configuration_fails_closed(
    tmp_path,
    monkeypatch,
    configured_product_tokens,
) -> None:
    monkeypatch.setenv("PRODUCT_AUTH_TOKENS_JSON", configured_product_tokens)
    monkeypatch.setenv("CAYU_OPERATOR_BEARER_TOKEN", "operator-token")
    service = build_service(
        mode=ServiceMode.PRODUCTION,
        provider=ScriptedModelProvider([]),
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
        product_store=SQLiteProductOperationStore(str(tmp_path / "invalid-product.db")),
    )

    assert service.manifest.product_access == "placeholder"
    assert service.manifest.operator_access == "authenticated"
    client = TestClient(service.asgi_app, raise_server_exceptions=False)
    response = client.post(
        "/api/operations",
        headers={
            "Authorization": "Bearer customer-token",
            "Idempotency-Key": "invalid-auth",
        },
        json={"request": "work"},
    )
    assert response.status_code == 503
    assert (
        client.get(
            "/cayu/", headers={"Authorization": "Bearer operator-token"}
        ).status_code
        == 200
    )


def test_malformed_operator_auth_configuration_fails_closed(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "PRODUCT_AUTH_TOKENS_JSON",
        '{"customer-token":{"tenant_id":"tenant-a","subject_id":"alice"}}',
    )
    monkeypatch.setenv("CAYU_OPERATOR_BEARER_TOKEN", "øperator-token")
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("allow-listed answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    service = build_service(
        mode=ServiceMode.PRODUCTION,
        provider=provider,
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
        product_store=SQLiteProductOperationStore(
            str(tmp_path / "invalid-operator.db")
        ),
    )

    assert service.manifest.product_access == "authenticated"
    assert service.manifest.operator_access == "placeholder"
    client = TestClient(service.asgi_app, raise_server_exceptions=False)
    response = client.post(
        "/api/operations",
        headers={
            "Authorization": "Bearer customer-token",
            "Idempotency-Key": "valid-product",
        },
        json={"request": "work"},
    )
    assert response.status_code == 201
    assert client.get("/cayu/").status_code == 503


def test_explicit_falsy_dependencies_are_not_replaced(tmp_path) -> None:
    class FalsyProductAccess(AuthenticatedProductAccess):
        def __bool__(self) -> bool:
            return False

    class FalsyOperatorAccess(AuthenticatedAccess):
        def __bool__(self) -> bool:
            return False

    class FalsyProductStore(SQLiteProductOperationStore):
        def __bool__(self) -> bool:
            return False

    class FalsySessionStore(InMemorySessionStore):
        def __bool__(self) -> bool:
            return False

    class FalsyTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __bool__(self) -> bool:
            return False

    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("allow-listed answer"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    product_store = FalsyProductStore(str(tmp_path / "falsy-product.db"))
    session_store = FalsySessionStore()
    task_store = FalsyTaskStore()
    service = build_service(
        mode=ServiceMode.PRODUCTION,
        provider=provider,
        session_store=session_store,
        task_store=task_store,
        product_store=product_store,
        product_access=FalsyProductAccess(dependency=customer_auth),
        operator_access=FalsyOperatorAccess(
            dependency=BasicAuth(username="operator", password="operator-secret")
        ),
    )

    assert service.product_store is product_store
    assert service.cayu_app.session_store is session_store
    assert service.cayu_app.task_store is task_store
    client = TestClient(service.asgi_app)
    assert (
        client.post(
            "/api/operations",
            headers={"Authorization": "Bearer customer-a", "Idempotency-Key": "falsy"},
            json={"request": "work"},
        ).status_code
        == 201
    )
    assert client.get("/cayu/", auth=("operator", "operator-secret")).status_code == 200


def test_cross_tenant_enumeration_and_mutation_are_denied(tmp_path) -> None:
    service, store, _provider = assembled_service(tmp_path)
    client = TestClient(service.asgi_app)
    created = client.post(
        "/api/operations",
        headers={"Authorization": "Bearer customer-a", "Idempotency-Key": "shared"},
        json={"request": "work"},
    ).json()
    with store._connect() as connection:
        private = next(iter(connection.execute("SELECT * FROM product_operations")))
    assert (
        client.get(
            f"/api/operations/{created['id']}",
            headers={"Authorization": "Bearer customer-b"},
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/api/operations/{private['session_id']}",
            headers={"Authorization": "Bearer customer-a"},
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/operations",
            headers={"Authorization": "Bearer customer-b", "Idempotency-Key": "shared"},
            json={"request": "work"},
        ).status_code
        == 409
    )


def test_idempotency_and_public_response_redaction(tmp_path) -> None:
    service, store, provider = assembled_service(tmp_path)
    client = TestClient(service.asgi_app)
    headers = {"Authorization": "Bearer customer-a", "Idempotency-Key": "same"}
    first = client.post("/api/operations", headers=headers, json={"request": "work"})
    second = client.post("/api/operations", headers=headers, json={"request": "work"})
    assert second.status_code == 200
    assert first.json() == second.json()
    assert len(provider.requests) == 1
    with store._connect() as connection:
        private = next(iter(connection.execute("SELECT * FROM product_operations")))
    receipt = json.loads(private["result_receipt"])
    assert receipt["publication_status"] == "completed"
    assert receipt["result"] == first.json()["result"]
    projected = repr(first.json())
    for field in ("tenant_id", "work_id", "session_id", "task_id", "idempotency_key"):
        assert private[field] not in projected
    conflict = client.post(
        "/api/operations", headers=headers, json={"request": "different"}
    )
    assert conflict.status_code == 409


def test_control_plane_separation_and_background_ownership_reload(tmp_path) -> None:
    producer_service, store, provider = assembled_service(tmp_path)

    async def reserve_work():
        return await store.reserve(
            tenant_id="tenant-a",
            subject_id="test-subject",
            idempotency_key="background",
            request_fingerprint=product_request_fingerprint(
                "work", agent_name=producer_service.agent_name
            ),
            public_id="op_background",
            work_id="work_background",
            session_id="session_background",
            task_id="task_background",
            request_text="work",
        )

    reservation = asyncio.run(reserve_work())
    assert reservation.created

    # Simulate a separately constructed worker. Its queue input contains only
    # the opaque work id; tenant ownership and private Cayu ids come from the
    # reopened application-owned store.
    service, reloaded_store, _provider = assembled_service(tmp_path, provider=provider)
    completed = asyncio.run(service.execute_work("work_background"))

    assert completed is not None
    assert completed.status == "completed"
    assert completed.result == "allow-listed answer"
    assert len(provider.requests) == 1
    with reloaded_store._connect() as connection:
        private = next(iter(connection.execute("SELECT * FROM product_operations")))
    assert private["tenant_id"] == "tenant-a"
    assert private["public_id"] == "op_background"

    client = TestClient(service.asgi_app)
    assert client.get("/cayu/").status_code == 401
    assert (
        client.get("/cayu/", headers={"Authorization": "Bearer customer-a"}).status_code
        == 401
    )
    assert client.get("/cayu/", auth=("operator", "operator-secret")).status_code == 200
    assert client.get("/cayu/assets/missing.js").status_code == 401
    assert client.delete("/cayu/api/sessions/guessed-private-id").status_code == 401
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


@pytest.mark.parametrize("precreate_task", [False, True])
def test_replacement_worker_recovers_reservation_and_precreated_task(
    tmp_path,
    precreate_task,
) -> None:
    async def scenario() -> None:
        runtime_path = str(tmp_path / f"initial-runtime-{precreate_task}.db")
        product_path = str(tmp_path / f"initial-product-{precreate_task}.db")
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("allow-listed answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        first_store = SQLiteProductOperationStore(product_path)
        first_service = build_service(
            mode=ServiceMode.PRODUCTION,
            provider=provider,
            session_store=SQLiteSessionStore(runtime_path),
            task_store=SQLiteTaskStore(runtime_path),
            product_store=first_store,
            product_access=AuthenticatedProductAccess(dependency=customer_auth),
            operator_access=AuthenticatedAccess(
                dependency=BasicAuth(username="operator", password="operator-secret")
            ),
        )
        reservation = await first_store.reserve(
            tenant_id="tenant-a",
            subject_id="test-subject",
            idempotency_key=f"initial-redelivery-{precreate_task}",
            request_fingerprint=product_request_fingerprint(
                "recover work", agent_name=first_service.agent_name
            ),
            public_id=f"op_initial_{precreate_task}",
            work_id=f"work_initial_{precreate_task}",
            session_id=f"session_initial_{precreate_task}",
            task_id=f"task_initial_{precreate_task}",
            request_text="recover work",
        )
        if precreate_task:
            await first_service.cayu_app.create_task(
                product_task_create(
                    reservation.operation,
                    agent_name=first_service.agent_name,
                )
            )

        replacement_service = build_service(
            mode=ServiceMode.PRODUCTION,
            provider=provider,
            session_store=SQLiteSessionStore(runtime_path),
            task_store=SQLiteTaskStore(runtime_path),
            product_store=SQLiteProductOperationStore(product_path),
            product_access=AuthenticatedProductAccess(dependency=customer_auth),
            operator_access=AuthenticatedAccess(
                dependency=BasicAuth(username="operator", password="operator-secret")
            ),
        )
        completed = await replacement_service.execute_work(
            reservation.operation.work_id
        )

        assert completed is not None
        assert completed.status == "completed"
        assert completed.result == "allow-listed answer"
        assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_replacement_worker_settles_terminal_receipt_without_redispatch(
    tmp_path,
) -> None:
    class FailingSettlementStore(SQLiteProductOperationStore):
        async def finish(self, **_kwargs):
            raise RuntimeError("product settlement unavailable")

    async def scenario() -> None:
        runtime_path = str(tmp_path / "replacement-runtime.db")
        product_path = str(tmp_path / "replacement-product.db")
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("allow-listed answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        first_store = FailingSettlementStore(product_path)
        first_service = build_service(
            mode=ServiceMode.PRODUCTION,
            provider=provider,
            session_store=SQLiteSessionStore(runtime_path),
            task_store=SQLiteTaskStore(runtime_path),
            product_store=first_store,
            product_access=AuthenticatedProductAccess(dependency=customer_auth),
            operator_access=AuthenticatedAccess(
                dependency=BasicAuth(username="operator", password="operator-secret")
            ),
        )
        reservation = await first_store.reserve(
            tenant_id="tenant-a",
            subject_id="test-subject",
            idempotency_key="replacement-terminal",
            request_fingerprint=product_request_fingerprint(
                "work", agent_name=first_service.agent_name
            ),
            public_id="op_replacement_terminal",
            work_id="work_replacement_terminal",
            session_id="session_replacement_terminal",
            task_id="task_replacement_terminal",
            request_text="work",
        )

        with pytest.raises(RuntimeError, match="product settlement unavailable"):
            await first_service.execute_work(reservation.operation.work_id)
        with first_store._connect() as connection:
            row = connection.execute(
                "SELECT status, result_receipt FROM product_operations WHERE work_id = ?",
                (reservation.operation.work_id,),
            ).fetchone()
            assert row["status"] == "pending"
            assert json.loads(row["result_receipt"])["result"] == "allow-listed answer"
            connection.execute(
                "UPDATE product_operations SET execution_claim_expires_at = 0 WHERE work_id = ?",
                (reservation.operation.work_id,),
            )

        replacement_store = SQLiteProductOperationStore(product_path)
        replacement_service = build_service(
            mode=ServiceMode.PRODUCTION,
            provider=provider,
            session_store=SQLiteSessionStore(runtime_path),
            task_store=SQLiteTaskStore(runtime_path),
            product_store=replacement_store,
            product_access=AuthenticatedProductAccess(dependency=customer_auth),
            operator_access=AuthenticatedAccess(
                dependency=BasicAuth(username="operator", password="operator-secret")
            ),
        )
        completed = await replacement_service.execute_work(
            reservation.operation.work_id
        )

        assert completed is not None
        assert completed.status == "completed"
        assert completed.result == "allow-listed answer"
        assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_durable_receipt_and_settlement_acknowledgements_are_reconstructed(
    tmp_path,
) -> None:
    class CommitThenRaiseStore(SQLiteProductOperationStore):
        receipt_calls = 0
        finish_calls = 0

        async def record_result_receipt(self, **kwargs):
            receipt = await super().record_result_receipt(**kwargs)
            self.receipt_calls += 1
            if self.receipt_calls == 1:
                raise RuntimeError("receipt acknowledgement lost")
            return receipt

        async def finish(self, **kwargs):
            operation = await super().finish(**kwargs)
            self.finish_calls += 1
            if self.finish_calls == 1:
                raise RuntimeError("settlement acknowledgement lost")
            return operation

    async def scenario() -> None:
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("allow-listed answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        store = CommitThenRaiseStore(str(tmp_path / "acknowledgements-product.db"))
        service = build_service(
            mode=ServiceMode.PRODUCTION,
            provider=provider,
            session_store=SQLiteSessionStore(
                str(tmp_path / "acknowledgements-runtime.db")
            ),
            task_store=SQLiteTaskStore(str(tmp_path / "acknowledgements-runtime.db")),
            product_store=store,
            product_access=AuthenticatedProductAccess(dependency=customer_auth),
            operator_access=AuthenticatedAccess(
                dependency=BasicAuth(username="operator", password="operator-secret")
            ),
        )
        reservation = await store.reserve(
            tenant_id="tenant-a",
            subject_id="test-subject",
            idempotency_key="acknowledgement-reconstruction",
            request_fingerprint=product_request_fingerprint(
                "work", agent_name=service.agent_name
            ),
            public_id="op_acknowledgement_reconstruction",
            work_id="work_acknowledgement_reconstruction",
            session_id="session_acknowledgement_reconstruction",
            task_id="task_acknowledgement_reconstruction",
            request_text="work",
        )

        completed = await service.execute_work(reservation.operation.work_id)

        assert completed is not None
        assert completed.status == "completed"
        assert completed.result == "allow-listed answer"
        assert store.receipt_calls == 2
        assert store.finish_calls == 2
        assert len(provider.requests) == 1
        with store._connect() as connection:
            row = connection.execute(
                "SELECT status, result, result_receipt FROM product_operations WHERE work_id = ?",
                (reservation.operation.work_id,),
            ).fetchone()
        assert row["status"] == "completed"
        assert row["result"] == "allow-listed answer"
        assert json.loads(row["result_receipt"])["result"] == "allow-listed answer"

    asyncio.run(scenario())


def test_concurrent_durable_replacement_workers_dispatch_once(tmp_path) -> None:
    class BlockingTaskStore(SQLiteTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self, path):
            super().__init__(path)
            self.create_started = asyncio.Event()
            self.allow_create = asyncio.Event()

        async def create_task(self, request):
            self.create_started.set()
            await self.allow_create.wait()
            return await super().create_task(request)

    async def scenario() -> None:
        runtime_path = str(tmp_path / "concurrent-runtime.db")
        product_path = str(tmp_path / "concurrent-product.db")
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("single execution"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        first_task_store = BlockingTaskStore(runtime_path)
        first_store = SQLiteProductOperationStore(product_path)
        first_service = build_service(
            mode=ServiceMode.PRODUCTION,
            provider=provider,
            session_store=SQLiteSessionStore(runtime_path),
            task_store=first_task_store,
            product_store=first_store,
            product_access=AuthenticatedProductAccess(dependency=customer_auth),
            operator_access=AuthenticatedAccess(
                dependency=BasicAuth(username="operator", password="operator-secret")
            ),
        )
        replacement_service = build_service(
            mode=ServiceMode.PRODUCTION,
            provider=provider,
            session_store=SQLiteSessionStore(runtime_path),
            task_store=SQLiteTaskStore(runtime_path),
            product_store=SQLiteProductOperationStore(product_path),
            product_access=AuthenticatedProductAccess(dependency=customer_auth),
            operator_access=AuthenticatedAccess(
                dependency=BasicAuth(username="operator", password="operator-secret")
            ),
        )
        reservation = await first_store.reserve(
            tenant_id="tenant-a",
            subject_id="test-subject",
            idempotency_key="concurrent-replacement",
            request_fingerprint=product_request_fingerprint(
                "work", agent_name=first_service.agent_name
            ),
            public_id="op_concurrent_replacement",
            work_id="work_concurrent_replacement",
            session_id="session_concurrent_replacement",
            task_id="task_concurrent_replacement",
            request_text="work",
        )

        first_execution = asyncio.create_task(
            first_service.execute_work(reservation.operation.work_id)
        )
        await first_task_store.create_started.wait()
        duplicate = await replacement_service.execute_work(
            reservation.operation.work_id
        )

        assert duplicate is not None and duplicate.status == "pending"
        assert provider.requests == []
        first_task_store.allow_create.set()
        completed = await first_execution
        assert completed is not None and completed.status == "completed"
        assert completed.result == "single execution"
        assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_replacement_worker_continues_same_durable_session(tmp_path) -> None:
    async def scenario() -> None:
        runtime_path = str(tmp_path / "continuation-runtime.db")
        product_path = str(tmp_path / "continuation-product.db")
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("continued answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        first_store = SQLiteProductOperationStore(product_path)
        first_service = build_service(
            mode=ServiceMode.PRODUCTION,
            provider=provider,
            session_store=SQLiteSessionStore(runtime_path),
            task_store=SQLiteTaskStore(runtime_path),
            product_store=first_store,
            product_access=AuthenticatedProductAccess(dependency=customer_auth),
            operator_access=AuthenticatedAccess(
                dependency=BasicAuth(username="operator", password="operator-secret")
            ),
        )
        reservation = await first_store.reserve(
            tenant_id="tenant-a",
            subject_id="test-subject",
            idempotency_key="replacement-continuation",
            request_fingerprint=product_request_fingerprint(
                "recover work", agent_name=first_service.agent_name
            ),
            public_id="op_replacement_continuation",
            work_id="work_replacement_continuation",
            session_id="session_replacement_continuation",
            task_id="task_replacement_continuation",
            request_text="recover work",
        )
        original_message = Message.text("user", reservation.operation.request_text)
        product_task = await first_service.cayu_app.task_store.create_task(
            product_task_create(
                reservation.operation,
                agent_name=first_service.agent_name,
            )
        )
        invocation_loop_policies = await first_service._continuation_loop_policies(
            reservation.operation.session_id
        )
        session_identity = profiled_session_identity(
            first_service.cayu_app,
            agent_name=first_service.agent_name,
            provider_name=provider.name,
            model="scripted-model",
            invocation_loop_policies=invocation_loop_policies,
        )
        tool_capability_ceiling = ToolCapabilityCeiling(
            tool_names=tuple(
                first_service.cayu_app._agents[first_service.agent_name].tools
            )
        )
        created_session = await first_service.cayu_app.session_store.create(
            run_request_with_task_invocation(
                RunRequest(
                    agent_name=first_service.agent_name,
                    session_id=reservation.operation.session_id,
                    task_id=reservation.operation.task_id,
                    messages=[original_message],
                    tool_capability_ceiling=tool_capability_ceiling,
                ),
                TaskInvocationSnapshot(
                    id=product_task.id,
                    session_id=product_task.session_id,
                    invocation=product_task.invocation,
                ),
            ),
            identity=session_identity,
            checkpoint_transform=lambda _session, checkpoint: (
                {
                    CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
                }
                if checkpoint is None
                else checkpoint
            ),
        )
        await first_service.cayu_app.task_store.start_task(
            reservation.operation.task_id,
            session_id=created_session.id,
            session_invocation=SessionInvocationBinding(
                id=created_session.id,
                session_instance_id=created_session.instance_id,
                invocation=created_session.invocation,
            ),
        )
        execution_profile = session_identity.execution_profile
        assert execution_profile is not None
        runtime_session_store = runtime_checkpoint_session_store(
            first_service.cayu_app.session_store
        )
        created_checkpoint = await runtime_session_store.load_checkpoint(created_session.id)
        interaction_id = "interaction_replacement_continuation"
        interaction_started_at = datetime.now(UTC)
        interaction_started_event_id = "interaction_start_replacement_continuation"
        interaction_started_event = Event(
            id=interaction_started_event_id,
            type=EventType.INTERACTION_STARTED,
            session_id=reservation.operation.session_id,
            interaction_id=interaction_id,
            timestamp=interaction_started_at,
            agent_name=first_service.agent_name,
            payload=InteractionSummaryEvidence(
                status=InteractionStatus.ACTIVE,
                start_event_id=interaction_started_event_id,
                started_at=interaction_started_at,
            ).model_dump(mode="json"),
        )
        await runtime_session_store.apply_invocation_lifecycle_command(
            AdmitInvocationCommand(
                session_id=reservation.operation.session_id,
                expected_session_instance_id=created_session.instance_id,
                expected_statuses=(SessionStatus.PENDING,),
                expected_run_epoch=created_session.run_epoch,
                expected_checkpoint_sha256=invocation_checkpoint_state_sha256(
                    created_checkpoint
                ),
                target_active_profile=ActiveInvocationExecutionProfile(
                    session_id=reservation.operation.session_id,
                    interaction_id=interaction_id,
                    run_epoch=created_session.run_epoch + 1,
                    profile=execution_profile,
                ),
                tool_capability_ceiling=tool_capability_ceiling,
                interaction_started_event=interaction_started_event,
                interaction_source_messages=(original_message,),
                defer_interaction_source=True,
                allow_pending_initial_interaction=True,
            ),
        )

        replacement_store = SQLiteProductOperationStore(product_path)
        replacement_service = build_service(
            mode=ServiceMode.PRODUCTION,
            provider=provider,
            session_store=SQLiteSessionStore(runtime_path),
            task_store=SQLiteTaskStore(runtime_path),
            product_store=replacement_store,
            product_access=AuthenticatedProductAccess(dependency=customer_auth),
            operator_access=AuthenticatedAccess(
                dependency=BasicAuth(username="operator", password="operator-secret")
            ),
        )
        completed = await replacement_service.execute_work(
            reservation.operation.work_id
        )

        assert completed is not None
        assert completed.status == "completed"
        assert completed.result == "continued answer"
        assert len(provider.requests) == 1
        assert [
            part.text
            for message in provider.requests[0].messages
            for part in message.content
        ] == [
            "recover work",
            "Continue this interrupted operation from its durable session state. "
            "Do not repeat work whose outcome is already recorded.",
        ]
        task = await replacement_service.cayu_app.task_store.load_task(
            reservation.operation.task_id
        )
        assert task is not None and task.status is TaskStatus.COMPLETED
        state = await replacement_service.cayu_app.session_store.load_state(
            reservation.operation.session_id
        )
        assert state is not None and state.status is SessionStatus.COMPLETED

    asyncio.run(scenario())


def test_workload_secret_is_rejected_before_generated_store_write(tmp_path) -> None:
    secret = "workload-secret-value"
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("unused"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="__AGENT_NAME__", model="scripted-model"))
    store = SQLiteProductOperationStore(str(tmp_path / "secret-boundary.db"))
    service = create_agent_service(
        app,
        agent_name="__AGENT_NAME__",
        mode=ServiceMode.PRODUCTION,
        product_access=AuthenticatedProductAccess(dependency=customer_auth),
        operator_access=AuthenticatedAccess(
            dependency=BasicAuth(username="operator", password="operator-secret")
        ),
        product_store=store,
    )

    response = TestClient(service.asgi_app).post(
        "/api/operations",
        headers={
            "Authorization": "Bearer customer-a",
            "Idempotency-Key": "secret-boundary",
        },
        json={"request": f"use {secret}"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid product request."}
    assert provider.requests == []
    with store._connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM product_operations").fetchone()[0]
            == 0
        )


def test_split_model_secret_is_redacted_before_generated_store_write(tmp_path) -> None:
    secret = "workload-secret-value"
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("workload-"),
            ModelStreamEvent.text_delta("secret-value"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="__AGENT_NAME__", model="scripted-model"))
    store = SQLiteProductOperationStore(str(tmp_path / "result-secret-boundary.db"))
    service = create_agent_service(
        app,
        agent_name="__AGENT_NAME__",
        mode=ServiceMode.PRODUCTION,
        product_access=AuthenticatedProductAccess(dependency=customer_auth),
        operator_access=AuthenticatedAccess(
            dependency=BasicAuth(username="operator", password="operator-secret")
        ),
        product_store=store,
    )

    response = TestClient(service.asgi_app).post(
        "/api/operations",
        headers={
            "Authorization": "Bearer customer-a",
            "Idempotency-Key": "result-secret-boundary",
        },
        json={"request": "safe work"},
    )

    assert response.status_code == 201
    assert response.json()["result"] == "[REDACTED_SECRET]"
    with store._connect() as connection:
        row = connection.execute(
            "SELECT result, result_receipt FROM product_operations"
        ).fetchone()
    result = row["result"]
    receipt = row["result_receipt"]
    assert result == "[REDACTED_SECRET]"
    assert secret not in result
    assert secret not in receipt
    assert json.loads(receipt)["result"] == "[REDACTED_SECRET]"


def test_product_execution_claim_and_terminal_settlement_are_authoritative(
    tmp_path,
) -> None:
    store = SQLiteProductOperationStore(str(tmp_path / "claims.db"))

    async def scenario() -> None:
        reservation = await store.reserve(
            tenant_id="tenant-a",
            subject_id="test-subject",
            idempotency_key="claim",
            request_fingerprint="claim-fingerprint",
            public_id="op_claim",
            work_id="work_claim",
            session_id="session_claim",
            task_id="task_claim",
            request_text="work",
        )
        first = await store.claim_execution(
            work_id="work_claim", claim_id="claim-one", lease_seconds=3600
        )
        assert first is not None and first.acquired
        with store._connect() as connection:
            initial_expiry = connection.execute(
                "SELECT execution_claim_expires_at FROM product_operations "
                "WHERE work_id = 'work_claim'"
            ).fetchone()[0]
        assert await store.heartbeat_execution(
            work_id="work_claim", claim_id="claim-one", lease_seconds=120
        )
        with store._connect() as connection:
            heartbeat_expiry = connection.execute(
                "SELECT execution_claim_expires_at FROM product_operations "
                "WHERE work_id = 'work_claim'"
            ).fetchone()[0]
        assert heartbeat_expiry >= initial_expiry
        reconstructed = await store.claim_execution(
            work_id="work_claim", claim_id="claim-one", lease_seconds=60
        )
        assert reconstructed is not None and reconstructed.acquired
        with store._connect() as connection:
            reconstructed_expiry = connection.execute(
                "SELECT execution_claim_expires_at FROM product_operations "
                "WHERE work_id = 'work_claim'"
            ).fetchone()[0]
        assert reconstructed_expiry >= heartbeat_expiry
        duplicate = await store.claim_execution(
            work_id="work_claim", claim_id="claim-two", lease_seconds=120
        )
        assert duplicate is not None and not duplicate.acquired
        assert not await store.heartbeat_execution(
            work_id="work_claim", claim_id="claim-two", lease_seconds=120
        )
        with pytest.raises(ProductExecutionClaimLost):
            await store.finish(
                work_id="work_claim",
                claim_id="claim-two",
                status="failed",
                result=None,
            )

        receipt = ProductResultReceipt.create(
            work_id=reservation.operation.work_id,
            public_id=reservation.operation.public_id,
            request_fingerprint=reservation.operation.request_fingerprint,
            session_id=reservation.operation.session_id,
            task_id=reservation.operation.task_id,
            source_event_id="model-completed-one",
            source_event_sequence=10,
            model_step_id="model-step-one",
            model_attempt_id="model-attempt-one",
            interaction_id="interaction-one",
            publication_status="completed",
            result="answer",
        )
        assert (
            await store.record_result_receipt(
                work_id="work_claim",
                claim_id="claim-one",
                receipt=receipt,
            )
            == receipt
        )
        assert (
            await store.record_result_receipt(
                work_id="work_claim",
                claim_id="claim-one",
                receipt=receipt,
            )
            == receipt
        )
        conflicting_receipt = ProductResultReceipt.create(
            work_id=reservation.operation.work_id,
            public_id=reservation.operation.public_id,
            request_fingerprint=reservation.operation.request_fingerprint,
            session_id=reservation.operation.session_id,
            task_id=reservation.operation.task_id,
            source_event_id="model-completed-conflict",
            source_event_sequence=10,
            model_step_id="model-step-one",
            model_attempt_id="model-attempt-conflict",
            interaction_id="interaction-one",
            publication_status="completed",
            result="different answer",
        )
        with pytest.raises(ProductResultReceiptConflict):
            await store.record_result_receipt(
                work_id="work_claim",
                claim_id="claim-one",
                receipt=conflicting_receipt,
            )
        foreign_receipt = ProductResultReceipt.create(
            work_id="other-work",
            public_id=reservation.operation.public_id,
            request_fingerprint=reservation.operation.request_fingerprint,
            session_id=reservation.operation.session_id,
            task_id=reservation.operation.task_id,
            source_event_id="model-completed-foreign",
            source_event_sequence=20,
            model_step_id="model-step-foreign",
            model_attempt_id="model-attempt-foreign",
            interaction_id="interaction-foreign",
            publication_status="completed",
            result="foreign answer",
        )
        with pytest.raises(ProductResultReceiptConflict):
            await store.record_result_receipt(
                work_id="work_claim",
                claim_id="claim-one",
                receipt=foreign_receipt,
            )
        with store._connect() as connection:
            connection.execute(
                "UPDATE product_operations SET execution_claim_expires_at = 0 "
                "WHERE work_id = 'work_claim'"
            )
        replacement = await store.claim_execution(
            work_id="work_claim", claim_id="claim-two", lease_seconds=120
        )
        assert replacement is not None and replacement.acquired
        with pytest.raises(ProductExecutionClaimLost):
            await store.record_result_receipt(
                work_id="work_claim",
                claim_id="claim-one",
                receipt=receipt,
            )

        completed = await store.finish(
            work_id="work_claim",
            claim_id="claim-two",
            status="completed",
            result="answer",
        )
        assert completed.status == "completed"
        assert await store.heartbeat_execution(
            work_id="work_claim", claim_id="claim-two", lease_seconds=120
        )
        assert not await store.heartbeat_execution(
            work_id="work_claim", claim_id="claim-one", lease_seconds=120
        )
        assert (
            await store.finish(
                work_id="work_claim",
                claim_id="claim-two",
                status="completed",
                result="answer",
            )
            == completed
        )
        with pytest.raises(ProductExecutionClaimLost):
            await store.finish(
                work_id="work_claim",
                claim_id="claim-one",
                status="completed",
                result="answer",
            )
        with pytest.raises(ProductOperationSettlementConflict):
            await store.finish(
                work_id="work_claim",
                claim_id="claim-two",
                status="failed",
                result=None,
            )

        await store.reserve(
            tenant_id="tenant-a",
            subject_id="test-subject",
            idempotency_key="expired-claim",
            request_fingerprint="expired-fingerprint",
            public_id="op_expired",
            work_id="work_expired",
            session_id="session_expired",
            task_id="task_expired",
            request_text="work",
        )
        abandoned = await store.claim_execution(
            work_id="work_expired", claim_id="abandoned", lease_seconds=120
        )
        assert abandoned is not None and abandoned.acquired
        with store._connect() as connection:
            connection.execute(
                "UPDATE product_operations SET execution_claim_expires_at = 0 "
                "WHERE work_id = 'work_expired'"
            )
        recovered = await store.claim_execution(
            work_id="work_expired", claim_id="recovered", lease_seconds=120
        )
        assert recovered is not None and recovered.acquired
        assert not await store.release_execution(
            work_id="work_expired", claim_id="abandoned"
        )
        assert await store.heartbeat_execution(
            work_id="work_expired", claim_id="recovered", lease_seconds=120
        )
        assert await store.release_execution(
            work_id="work_expired", claim_id="recovered"
        )
        assert await store.release_execution(
            work_id="work_expired", claim_id="recovered"
        )
        takeover = await store.claim_execution(
            work_id="work_expired", claim_id="takeover", lease_seconds=120
        )
        assert takeover is not None and takeover.acquired
        assert not await store.release_execution(
            work_id="work_expired", claim_id="recovered"
        )
        assert await store.heartbeat_execution(
            work_id="work_expired", claim_id="takeover", lease_seconds=120
        )

    asyncio.run(scenario())


def test_in_memory_product_store_keeps_one_shared_database() -> None:
    store = SQLiteProductOperationStore(":memory:")

    async def scenario() -> None:
        reservation = await store.reserve(
            tenant_id="tenant-a",
            subject_id="test-subject",
            idempotency_key="memory-operation",
            request_fingerprint="memory-fingerprint",
            public_id="op_memory",
            work_id="work_memory",
            session_id="session_memory",
            task_id="task_memory",
            request_text="work",
        )
        assert reservation.created
        assert (
            await store.find(tenant_id="tenant-a", public_id="op_memory")
            == reservation.operation
        )
        claim = await store.claim_execution(
            work_id="work_memory",
            claim_id="memory-claim",
            lease_seconds=120,
        )
        assert claim is not None and claim.acquired
        await store.record_result_receipt(
            work_id="work_memory",
            claim_id="memory-claim",
            receipt=ProductResultReceipt.create(
                work_id=reservation.operation.work_id,
                public_id=reservation.operation.public_id,
                request_fingerprint=reservation.operation.request_fingerprint,
                session_id=reservation.operation.session_id,
                task_id=reservation.operation.task_id,
                source_event_id="model-completed-memory",
                source_event_sequence=10,
                model_step_id="model-step-memory",
                model_attempt_id="model-attempt-memory",
                interaction_id="interaction-memory",
                publication_status="completed",
                result="answer",
            ),
        )
        completed = await store.finish(
            work_id="work_memory",
            claim_id="memory-claim",
            status="completed",
            result="answer",
        )
        assert completed.status == "completed"
        assert (
            await store.find(tenant_id="tenant-a", public_id="op_memory") == completed
        )

    asyncio.run(scenario())


def test_provider_error_and_prompt_sentinels_are_redacted(tmp_path) -> None:
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.error("provider-error-sentinel"),
            ModelStreamEvent.completed({"finish_reason": "error"}),
        ]
    )
    service, _store, _provider = assembled_service(tmp_path, provider=provider)
    response = TestClient(service.asgi_app).post(
        "/api/operations",
        headers={"Authorization": "Bearer customer-a", "Idempotency-Key": "failure"},
        json={"request": "private-prompt-sentinel"},
    )
    assert response.status_code == 201
    assert response.json()["status"] == "failed"
    assert response.json()["result"] is None
    assert "provider-error-sentinel" not in response.text
    assert "private-prompt-sentinel" not in response.text
"""

_SERVICE_GUIDANCE = """

## Public service security contract

This project uses Cayu's maintained public-service factory. Product customers
authenticate at `/api/operations`; `/cayu/` is a separate operator-only control
plane. Every product read is tenant-qualified through the application-owned
mapping in `product_store.py`. Never authorize from request tenant fields, Cayu
IDs, labels, metadata, model output, or tool input, and never return raw runtime
records to customers.

The product store's execution claim, heartbeat, and terminal write are one
durability contract. Preserve their atomic, same-claim replay behavior when
replacing SQLite. The worker must match the reservation's canonical agent/request
fingerprint before it creates Cayu work. Recheck the claim immediately before
provider execution, and retain the settling claim identity on terminal rows so
late heartbeats and acknowledgement reconstruction cannot confuse a successful
settlement with ownership loss. A queue or worker redelivery must not execute
live owned work while another claim remains valid or overwrite an existing
terminal result.
Before Cayu commits session completion, the maintained executor records a
content-bound publication receipt for the exact final conversational model
event and bounded result (or an explicit unsafe-result publication failure).
The same receipt reconstructs acknowledgement loss; only the current execution
claim may advance it to a later durable event sequence. A replacement worker can
therefore settle terminal completed or failed Cayu work from bounded evidence
without redispatching provider work or scraping a transcript. Live, interrupted,
contradictory, unsupported, or evidence-bounded work remains pending and releases
its exact execution claim. Recoverable abandoned work is fenced through Cayu's
durable incomplete-session recovery and resumed on the same session and task;
the allow-listed `recovery_status` reports active, approval, input, interrupted,
or manual-reconciliation states without exposing raw runtime records.
The maintained control-plane mount attaches this publication contract to
operator resume, approval, user-input, and tool-recovery continuations. It uses
the private application-owned session index, acquires and heartbeats the product
claim only at final publication, and interrupts instead of completing without a
receipt when another product worker owns that claim. These process-local loop
policies never cross the HTTP body or durable runtime record boundary.
Release is idempotent after acknowledgement loss and cannot clear a successor's
claim.
This lease does not provide exactly-once provider effects if a process stalls
beyond its lease or dies after external work begins; applications that require
that guarantee need an idempotent external-effect and worker-recovery design.

The create endpoint accepts at most 1 MiB of encoded JSON, rejects duplicate
object keys, rejects caller-controlled tenant, idempotency, or request values
that collide with the application's workload-secret registry before reservation,
and marks product responses `private, no-store`. It never persists a redacted
request because doing so would change delayed execution semantics. The maintained
executor redacts across model-delta boundaries before retaining only the bounded
final model turn, not the complete runtime event stream. Every public projection
redacts stored results again with the current workload-secret registry. A secret
collision in a public identifier fails closed. Keep equivalent bounds,
redaction, and cache controls when extending the product API.

Local development is explicit and loopback-only:

```bash
uv run cayu serve --dev
```

In another terminal, exercise the customer route with explicit development
identity headers (these headers are rejected as an identity source in the
production profile):

```bash
curl -X POST http://127.0.0.1:8000/api/operations \\
  -H 'Content-Type: application/json' \\
  -H 'Idempotency-Key: local-request-1' \\
  -H 'X-Cayu-Dev-Tenant: local-tenant' \\
  -H 'X-Cayu-Dev-Subject: local-user' \\
  -d '{"request":"YOUR REQUEST"}'
```

For production, configure `PRODUCT_AUTH_TOKENS_JSON` and
`CAYU_OPERATOR_BEARER_TOKEN`, run without `--dev`, and require both:

```bash
export PRODUCT_AUTH_TOKENS_JSON='{"replace-customer-token":{"tenant_id":"tenant-a","subject_id":"user-a"}}'
export CAYU_OPERATOR_BEARER_TOKEN='replace-operator-token'
uv run cayu serve --host 0.0.0.0
```

The `cayu serve` listener uses plain HTTP. Put it behind a trusted
TLS-terminating ingress or reverse proxy, restrict the backend listener to that
trusted network, and expose only the HTTPS endpoint to customers and operators.
Never send either bearer token over a directly exposed HTTP connection.

The generated token map is a bounded self-hosted example, not an identity
provider. Replace its authentication dependency with your trusted application
authority while continuing to return server-derived `ProductPrincipal` values.

```bash
uv run cayu check --deploy --fail-on warning --json
uv run pytest -q tests/test_public_service_security.py
```

The check verifies the maintained factory, configured exposure posture, and
authenticated control plane. The assembled-ASGI tests prove the generated
customer authorization behavior. Routes added outside this factory, including
an arbitrary ASGI or Uvicorn target, remain unverified by Cayu.

The unauthenticated `/health` exception returns only `{"ok": true}`. Do not add
product, tenant, session, task, provider, or runtime evidence to health or
readiness responses.
"""

_SERVICE_AGENTS_GUIDANCE = """

## Public-service invariant

This is the supported multi-user service template. Preserve `build_service()`
as the one serving/check/test factory. Product customer identity comes only
from `AuthenticatedProductAccess`; authorize every resource through a
tenant-qualified lookup in the application-owned product store. Keep the
operator policy separate and never expose `/cayu/` or raw Cayu evidence to
customers. An arbitrary ASGI route is outside Cayu's verification boundary.

Before declaring production work complete, run:

- `uv run cayu check --deploy --fail-on warning --json`
- `uv run pytest -q tests/test_public_service_security.py`
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
        "--agent-name",
        help="Registered first-agent name (default: project name).",
    )
    parser.add_argument(
        "--dir",
        metavar="DIR",
        default=".",
        help="Parent directory to create the project in (default: current directory).",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "openrouter", "openai-subscription"),
        help=(
            "Explicit live-provider default. Omit for a provider-neutral scaffold; "
            "CAYU_PROVIDER can select or override it later."
        ),
    )
    parser.add_argument(
        "--template",
        choices=("agent", "service"),
        default="agent",
        help=("Project shape: minimal agent (default) or maintained public-agent service."),
    )
    parser.add_argument(
        "--composition",
        choices=("coding",),
        help=(
            "Opt in to an explicit maintained starter composition. "
            "The coding composition requires git and rg."
        ),
    )
    parser.add_argument(
        "--coding-execution",
        choices=("docker",),
        help=(
            "Explicit execution backend for --composition coding. Omit to keep "
            "the bounded no-command composition; docker adds trusted-repository "
            "named checks in an admitted Docker environment."
        ),
    )
    parser.add_argument(
        "--coding-toolchain",
        choices=("python",),
        help=(
            "Explicit admitted profile for Docker coding execution. The generated "
            "Python profile is the first built-in; applications may register custom "
            "DockerCodingToolchainProfile values in composition.py."
        ),
    )


def _installed_cayu_version() -> str:
    return package_version()


def project_files(
    name: str,
    *,
    agent_name: str | None = None,
    provider: str | None = None,
    template: str = "agent",
    composition: str | None = None,
    coding_execution: str | None = None,
    coding_toolchain: str | None = None,
) -> dict[str, str]:
    resolved_agent_name = name if agent_name is None else agent_name
    reviewer_name = f"{resolved_agent_name}-reviewer"

    def render(template: str) -> str:
        provider_display = provider or "no live provider"
        provider_literal = "None" if provider is None else json.dumps(provider)
        replacements = {
            "__PROJECT_NAME__": name,
            "__AGENT_NAME__": resolved_agent_name,
            "__REVIEWER_NAME__": reviewer_name,
            "__CAYU_VERSION__": _installed_cayu_version(),
            "__PROVIDER_DISPLAY__": provider_display,
            "__PROVIDER_LITERAL__": provider_literal,
            "__PROVIDER_GUIDE_POINTER__": _PROVIDER_GUIDE_POINTER,
        }
        return _TEMPLATE_TOKEN_RE.sub(
            lambda match: replacements[match.group(0)],
            template,
        )

    files = {
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
    if coding_execution is not None and composition != "coding":
        raise ValueError("coding_execution requires composition='coding'.")
    if coding_toolchain is not None and coding_execution != "docker":
        raise ValueError("coding_toolchain requires coding_execution='docker'.")
    if composition is not None:
        if composition != "coding":
            raise ValueError("composition must be 'coding'.")
        if template != "agent":
            raise ValueError("the coding composition cannot be combined with a service template.")
        from cayu.cli.coding_composition import coding_project_files

        files.update(
            coding_project_files(
                files=files,
                render=render,
                execution=coding_execution,
                toolchain=coding_toolchain,
            )
        )
        return files
    if template == "agent":
        return files
    if template != "service":
        raise ValueError("template must be 'agent' or 'service'.")
    service_pyproject = (
        files["pyproject.toml"]
        .replace(
            f'dependencies = ["cayu>={_installed_cayu_version()}"]',
            f'dependencies = ["cayu[server]>={_installed_cayu_version()}"]',
        )
        .replace(
            f'dev = ["cayu[server]>={_installed_cayu_version()}", "pytest"]',
            'dev = ["pytest", "ruff>=0.15.15,<0.16"]',
        )
    )
    service_pyproject = service_pyproject.replace(
        'factory = "app:build_app"\n',
        'factory = "app:build_app"\nservice_factory = "service:build_service"\n',
    )
    files.update(
        {
            "service.py": render(_SERVICE_PY),
            "product_store.py": _PRODUCT_STORE_PY,
            "tests/test_public_service_security.py": _SERVICE_SECURITY_TEST_PY,
            "pyproject.toml": service_pyproject,
            "README.md": files["README.md"] + _SERVICE_GUIDANCE,
            "AGENTS.md": files["AGENTS.md"] + _SERVICE_AGENTS_GUIDANCE,
        }
    )
    return files


def run_new(args: argparse.Namespace) -> int:
    name = args.name
    if not _NAME_RE.fullmatch(name):
        print(
            f"error: invalid project name {name!r} "
            "(use letters, digits, '-' or '_', starting with a letter).",
            file=sys.stderr,
        )
        return 1
    agent_name = name if args.agent_name is None else args.agent_name
    if not _NAME_RE.fullmatch(agent_name):
        print(
            f"error: invalid agent name {agent_name!r} "
            "(use letters, digits, '-' or '_', starting with a letter).",
            file=sys.stderr,
        )
        return 1
    composition = getattr(args, "composition", None)
    coding_execution = getattr(args, "coding_execution", None)
    coding_toolchain = getattr(args, "coding_toolchain", None)
    if coding_execution is not None and composition != "coding":
        print(
            "error: --coding-execution requires --composition coding.",
            file=sys.stderr,
        )
        return 1
    if coding_toolchain is not None and coding_execution != "docker":
        print(
            "error: --coding-toolchain requires --coding-execution docker.",
            file=sys.stderr,
        )
        return 1
    if composition is not None and args.template != "agent":
        print(
            "error: --composition coding cannot be combined with --template service.",
            file=sys.stderr,
        )
        return 1
    target = Path(args.dir) / name
    if target.is_symlink():
        print(f"error: {target} cannot be a symbolic link.", file=sys.stderr)
        return 1
    if target.exists() and not target.is_dir():
        print(f"error: {target} already exists and is not a directory.", file=sys.stderr)
        return 1
    if target.exists() and any(target.iterdir()):
        print(f"error: {target} already exists and is not empty.", file=sys.stderr)
        return 1
    coding_git: str | None = None
    coding_target_identity: tuple[int, int] | None = None
    coding_target_mode: int | None = None
    if composition == "coding":
        try:
            LocalWorkspace.require_path_operations_supported()
        except RuntimeError as exc:
            print(f"error: --composition coding {exc}", file=sys.stderr)
            return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            _require_safe_scaffold_parent(target.parent)
            coding_git, _ = _preflight_coding_commands(parent=target.parent)
            target.mkdir(parents=True, exist_ok=True)
            target_stat = target.stat(follow_symlinks=False)
            coding_target_identity = (target_stat.st_dev, target_stat.st_ino)
            coding_target_mode = stat.S_IMODE(target_stat.st_mode)
        except (_ScaffoldCommandError, OSError) as exc:
            print(f"error: --composition coding {exc}", file=sys.stderr)
            return 1

    files = project_files(
        name,
        agent_name=agent_name,
        provider=args.provider,
        template=args.template,
        composition=composition,
        coding_execution=coding_execution,
        coding_toolchain=coding_toolchain,
    )
    if composition == "coding":
        assert coding_git is not None
        assert coding_target_identity is not None
        assert coding_target_mode is not None
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{name}.cayu-scaffold-",
                dir=target.parent,
            ) as raw_staging:
                staging = Path(raw_staging)
                staging_identity = _scaffold_directory_identity(staging)
                for rel, content in files.items():
                    path = staging / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                with tempfile.TemporaryDirectory(
                    prefix="cayu-scaffold-hooks-", dir=target.parent
                ) as raw_hooks:
                    hooks = Path(raw_hooks)
                    git_env = _sanitized_scaffold_git_environment(cwd=staging)
                    _run_scaffold_git_command(
                        _safe_git_argv(
                            coding_git,
                            "init",
                            "-b",
                            "main",
                            f"--template={hooks}",
                            hooks_dir=hooks,
                        ),
                        cwd=staging,
                        env=git_env,
                        expected_directory_identity=staging_identity,
                    )
                    repository_root = _run_scaffold_git_command(
                        _safe_git_argv(
                            coding_git,
                            "rev-parse",
                            "--show-toplevel",
                            hooks_dir=hooks,
                        ),
                        cwd=staging,
                        env=git_env,
                        expected_directory_identity=staging_identity,
                    ).strip()
                    try:
                        resolved_repository_root = Path(repository_root).resolve(strict=True)
                    except (OSError, ValueError):
                        raise _ScaffoldCommandError(
                            "git repository root verification failed"
                        ) from None
                    if resolved_repository_root != staging.resolve(strict=True):
                        raise _ScaffoldCommandError(
                            "git repository authority escaped the scaffold staging directory"
                        )
                    _run_scaffold_git_command(
                        _safe_git_argv(
                            coding_git,
                            "add",
                            "--force",
                            "--",
                            ".",
                            hooks_dir=hooks,
                        ),
                        cwd=staging,
                        env=git_env,
                        expected_directory_identity=staging_identity,
                    )
                    tracked_output = _run_scaffold_git_command(
                        _safe_git_argv(
                            coding_git,
                            "ls-files",
                            "--cached",
                            "-z",
                            "--",
                            hooks_dir=hooks,
                        ),
                        cwd=staging,
                        env=git_env,
                        expected_directory_identity=staging_identity,
                    )
                    tracked_files = frozenset(path for path in tracked_output.split("\0") if path)
                    if tracked_files != frozenset(files):
                        raise _ScaffoldCommandError(
                            "git index does not contain the complete generated project"
                        )
                    _run_scaffold_git_command(
                        _safe_git_argv(
                            coding_git,
                            "-c",
                            "user.name=Cayu Scaffold",
                            "-c",
                            "user.email=scaffold@cayu.local",
                            "commit",
                            "-m",
                            "Initial Cayu coding composition",
                            hooks_dir=hooks,
                        ),
                        cwd=staging,
                        env=git_env,
                        expected_directory_identity=staging_identity,
                    )
                _publish_staged_scaffold(
                    staging=staging,
                    target=target,
                    expected_target_identity=coding_target_identity,
                    target_mode=coding_target_mode,
                )
        except (_ScaffoldCommandError, OSError, ValueError) as exc:
            print(
                f"error: could not initialize coding workspace: {exc}",
                file=sys.stderr,
            )
            return 1
    else:
        for rel, content in files.items():
            path = target / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    print(f"Scaffolded {target}/ — credential-free proof:")
    print(f"  cd {target}")
    print("  uv sync --extra dev")
    print("  uv run cayu inspect --json")
    if args.template == "service":
        print("  uv run cayu check --deploy --fail-on warning --json")
        print("  uv run pytest -q tests/test_public_service_security.py")
    elif composition == "coding":
        print("  uv run cayu check --json")
        print("  uv run pytest -q tests/test_coding_composition.py")
        if coding_execution == "docker":
            print("  Configure pinned inputs: docker-coding-build.json")
            print("  Build and record image: uv run python build_coding_image.py")
            print(f"  Toolchain profile: {coding_toolchain or 'python'}")
    else:
        print("  uv run cayu check --json")
        print("  uv run pytest")
    print("  uv run cayu eval run")
    if args.template == "service":
        print("  Local public service: uv run cayu serve --dev")
        print("  Product API: http://127.0.0.1:8000/api/operations")
        print("  Operator control plane: http://127.0.0.1:8000/cayu/")
    elif composition == "coding":
        if coding_execution == "docker":
            print(
                "  Execution: admitted trusted-repository Docker checks and commands "
                "(network disabled)"
            )
        print(f'  Live run: uv run python run.py --agent {agent_name} --message "YOUR REQUEST"')
        print("  Local control plane: uv run cayu serve --dev")
        print("  Open: http://127.0.0.1:8000/cayu/")
    else:
        print("  Local control plane: uv run cayu serve --dev")
        print("  Open: http://127.0.0.1:8000/cayu/")
    if args.provider is None:
        print("  Live provider: none selected; set CAYU_PROVIDER explicitly before `run.py`.")
    else:
        print(f"  Live provider: {args.provider} (credentials authenticate this choice).")
    return 0
