"""``cayu new`` — scaffold a safe, verifiable Cayu agent project."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cayu._version import package_version
from cayu.cli._bounded_command import (
    BoundedCommandOutputOverflowError,
    BoundedCommandReadError,
    BoundedCommandStartError,
    BoundedCommandTimeoutError,
    run_bounded_command,
)
from cayu.cli._guarded_tree_publication import (
    DestinationPolicy,
    GuardedTreePublicationError,
    GuardedTreeStage,
    publish_guarded_tree,
)
from cayu.cli.scaffold_convention import (
    application_guidance,
    convention_files,
    local_database_path,
    scaffold_contract,
)
from cayu.cli.scaffold_plan import (
    ADAPTERS,
    CAPABILITIES,
    PRESETS,
    ApplicationPlan,
    ScaffoldPlanError,
    capability_spec,
    normalize_application_plan,
)
from cayu.workspaces.local import LocalWorkspace

_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
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


class _ScaffoldTargetNotEmpty(_ScaffoldCommandError):
    """The guarded publisher positively observed a non-empty target."""


class _ScaffoldCodingPreflightError(_ScaffoldCommandError):
    """Coding dependencies failed before stage population began."""


def _translated_scaffold_publication_error(
    error: GuardedTreePublicationError,
) -> _ScaffoldCommandError:
    """Retain bounded shared-owner diagnostics at the scaffold boundary."""

    message = str(error)
    if error.paths:
        message += "; affected paths: " + ", ".join(repr(path) for path in error.paths)
    translated = _ScaffoldCommandError(message)
    for note in getattr(error, "__notes__", ()):
        translated.add_note(note)
    return translated


def _scaffold_error_message(error: BaseException) -> str:
    """Render bounded exception notes through the CLI's existing error schema."""

    message = str(error)
    for note in getattr(error, "__notes__", ()):
        message += f"; note: {note}"
    return message


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
        # A detached post-commit repack can mutate .git while publication seals it.
        "-c",
        "maintenance.auto=false",
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
    assert_directory_unchanged: Callable[[], object] | None = None,
) -> str:
    if assert_directory_unchanged is not None:
        assert_directory_unchanged()
    if _scaffold_directory_identity(cwd) != expected_directory_identity:
        raise _ScaffoldCommandError("scaffold target identity changed")
    output = _run_scaffold_command(argv, cwd=cwd, env=env)
    if assert_directory_unchanged is not None:
        assert_directory_unchanged()
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


def _populate_scaffold_stage(
    *,
    staging: GuardedTreeStage,
    files: dict[str, str],
    plan: ApplicationPlan,
    hooks_parent: Path,
) -> None:
    """Populate one publisher-owned stage with a complete scaffold."""

    coding_git: str | None = None
    if plan.preset == "coding":
        try:
            LocalWorkspace.require_path_operations_supported()
            coding_git, _ = _preflight_coding_commands(parent=hooks_parent)
        except (RuntimeError, _ScaffoldCommandError, OSError) as exc:
            raise _ScaffoldCodingPreflightError(f"coding preset {exc}") from None
    contents = {relative: content.encode("utf-8") for relative, content in files.items()}
    directories = {"data"}
    if "artifacts" in plan.capabilities:
        directories.add("data/artifacts")
    file_modes: dict[str, int] = {}
    if "memory" in plan.capabilities:
        private_path = "data/memory-evidence.key"
        contents[private_path] = (secrets.token_urlsafe(48) + "\n").encode("utf-8")
        if os.name != "nt":
            file_modes[private_path] = 0o600
    staging.write_tree(
        contents,
        directories=directories,
        file_modes=file_modes,
        file_mode=None,
        directory_mode=None,
        root_mode=staging.publication_root_mode(),
    )
    if plan.preset != "coding":
        return
    assert coding_git is not None
    stage_path = staging._specialized_path()
    with tempfile.TemporaryDirectory(
        prefix="cayu-scaffold-hooks-",
        dir=hooks_parent,
    ) as raw_hooks:
        _initialize_coding_git(
            staging=stage_path,
            files=files,
            git=coding_git,
            hooks=Path(raw_hooks),
            assert_staging_unchanged=staging.capture_owned_identity,
        )
    staging.capture_owned_identity()


def _scaffold_publication_request_digest(
    *,
    files: dict[str, str],
    plan: ApplicationPlan,
) -> str:
    """Bind retry identity to every deterministic scaffold-publication input."""

    directories = {"data"}
    if "artifacts" in plan.capabilities:
        directories.add("data/artifacts")
    directories.update(parent for path in files if (parent := str(Path(path).parent)) != ".")
    payload = {
        "schema_version": 1,
        "plan": plan.as_dict(
            files=tuple(sorted(files)),
            directories=tuple(sorted(directories)),
            private_files=(("data/memory-evidence.key",) if "memory" in plan.capabilities else ()),
        ),
        "files": [[relative, files[relative]] for relative in sorted(files)],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


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
dependencies = __RUNTIME_DEPENDENCIES__

[project.optional-dependencies]
dev = __DEV_DEPENDENCIES__

[tool.cayu]
factory = "app:build_app"
__SERVICE_FACTORY____EVAL_TARGET__
[tool.cayu.session_store]
__SESSION_STORE__

[tool.pytest.ini_options]
pythonpath = ["."]

[tool.uv]
cache-dir = ".cayu/uv-cache"
__UV_SOURCES__"""

_PROVIDER_GUIDE_POINTER = """OpenRouter is a first-class scaffold choice. Fireworks, Baseten, OpenCode Go,
and other compatible endpoints work through Cayu's generic adapter. Run
`uv run --no-sync cayu guide providers#compatible-chat-completions` for exact setup."""

_README = """# __PROJECT_NAME__

__PRESET_OVERVIEW__

## Application structure

Describe the requested job in `agents/agent.py`; update `tests/test_agent.py`
and `evals/agent.py` to prove that behavior. The project factory is `build_app()`
in `app.py`. Run `cayu guide anatomy` for its lifecycle contract.

Run `uv run --no-sync cayu guide authoring#cayu-map` to select another concept only when
the requested behavior requires it. For durable operational changes, start with
`uv run --no-sync cayu guide durable-operations`; it covers propose, authorize, act once,
verify, inspect, and recover. `uv run --no-sync cayu guide references` contains the
package-shipped offline references.

## Setup and prove the project

```bash
uv sync --extra dev
uv run --no-sync cayu guide anatomy
uv run --no-sync cayu inspect --json
uv run --no-sync cayu check --fail-on warning --json
uv run --no-sync pytest
uv run --no-sync cayu eval run
uv run --no-sync cayu session list
```

__DATABASE_README_PROOF_GUIDANCE__

These commands require no model API key. They prove project construction,
static wiring, a deterministic model response, and its eval.

## Inspect with the local control plane

Cayu's packaged developer/operator control plane reads the same durable stores
as the project commands. Start it in a separate terminal:

```bash
uv run --no-sync cayu serve --dev
```

Then open `http://127.0.0.1:8000/cayu/`. The explicit `--dev` flag enables
unauthenticated trusted-local access only. It does not make the control plane
the application's end-user UI or configure a production deployment.
Project serving also derives Evals project/release identity and uses
__DATABASE_EVALS_STORAGE_GUIDANCE__ automatically. With the normal live
provider configured, open a completed or failed simple session, choose
**Evaluate**, review its assertions, save or approve the captured result, and
start one bounded fresh trial. The generated target reuses the registered agent
and its ordinary runtime policy; no Evals-specific Python configuration is
required.
Run `uv run --no-sync cayu guide evals-first` for the shortest suite, baseline,
and comparison workflow. Add `cayu guide evals-ai-quality` only after explicitly
choosing judge provider, model, privacy, and same-model policy.
Never mount it with unauthenticated open access on a public listener;
client-IP checks are not authentication. Public or deployed control-plane
access requires an authenticated access policy.

## Run with a live provider

Provider intent is explicit. This scaffold defaults to `__PROVIDER_DISPLAY__`;
override it with `CAYU_PROVIDER=openai`, `anthropic`, `openrouter`, `cayu-gateway`, or
`openai-subscription`. API-key variables authenticate that choice and never
select it automatically.

__PROVIDER_GUIDE_POINTER__

OpenAI Platform API:

```bash
export CAYU_PROVIDER=openai
export OPENAI_API_KEY=sk-...
uv run --no-sync python run.py --message "YOUR REQUEST"
```

Anthropic API:

```bash
export CAYU_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
uv run --no-sync python run.py --message "YOUR REQUEST"
```

OpenRouter (the model slug is always explicit):

```bash
export CAYU_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-...
export CAYU_MODEL=vendor/model
uv run --no-sync python run.py --message "YOUR REQUEST"
```

Optional `OPENROUTER_HTTP_REFERER` and `OPENROUTER_APP_TITLE` values add
OpenRouter attribution headers. Set `OPENROUTER_ROUTER_METADATA=enabled` to
retain only bounded routing evidence; free-form pipeline and attempt data is
never persisted. Put OpenRouter routing controls such as `provider.order`,
`provider.allow_fallbacks`, `provider.require_parameters`, `provider.zdr`, and
`provider.data_collection` in `AgentSpec.provider_options["openrouter"]`.

Cayu Gateway (explicit endpoint and model):

```bash
export CAYU_PROVIDER=cayu-gateway
export CAYU_GATEWAY_BASE_URL=https://YOUR_GATEWAY/v1
export CAYU_GATEWAY_API_KEY=YOUR_KEY
export CAYU_MODEL=YOUR_MODEL
uv run --no-sync python run.py --message "YOUR REQUEST"
```

Gateway owns balances and spending caps. Runtime's local cost estimates remain
execution safeguards. Run `cayu guide providers#cayu-gateway` for reported usage
and authenticated generation lookup.

Your own ChatGPT subscription for local testing:

```bash
uv run --no-sync cayu auth openai login
CAYU_PROVIDER=openai-subscription uv run --no-sync python run.py --message "YOUR REQUEST"
```

Subscription mode selects `gpt-6-luna` by default. Set `CAYU_MODEL` if your plan
offers a different model. If a run fails at the provider, add
`--show-provider-errors` to print the provider's explanation.

This experimental path is intended for the subscription holder's own local
development and evaluation. It is not intended for production, customer-facing
or multi-user services, credential sharing, resale, or bypassing plan limits.
For production, use the OpenAI Platform API or another officially supported
provider. Run `uv run --no-sync cayu guide providers#openai-subscription` for the local
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

__AGENT_OWNERSHIP__

## When the user asks for...

Users describe product behavior, not Cayu concepts. Before building one of these
yourself, use the Cayu feature that already does it:

| The user asks for | Use | Read |
| --- | --- | --- |
| A chat, follow-up questions, "remember what I said" | Keep `outcome.session_id`, then `run_to_completion(app, ResumeRequest(session_id=..., messages=[...]))`. A new `RunRequest` always starts an empty conversation. | `cayu guide references#sessions` |
| Approval before the agent acts, "let me confirm first" | A tool generated with `cayu generate tool NAME --agent __AGENT_NAME__ --effect external`. Listed external tools pause for approval under `policies/tools.py`; resolve pauses on the control plane's Pending page. Do not build a separate approve button or approval table. | `cayu guide durable-operations` |
| The agent asking the user a question mid-task | The `ask_user` tool, a durable user-input pause | `cayu guide references#approvals` |
| Structured data back (JSON, fields, a list) | `RunRequest(structured_output=StructuredOutputSpec(...))`, then `outcome.structured_output.output`. Do not parse JSON out of `final_text`. | `cayu guide structured-output` |
| A new capability: call an API, read data, compute | A typed tool with a declared `ToolEffect` | `cayu guide references#domain-tool` |
| A web UI that shows progress or live results | `mount_cayu(...)` and `connect()` from the served `client.js`. Do not poll on a fixed timer. | `cayu guide app-ui` |
| Work that runs in the background or later | `TaskStore`, dispatcher and worker | `cayu guide references#background-work` |
| Remembering facts across conversations | The knowledge tools (remember, search) | `cayu guide references#knowledge` |
| Running a fixed command or parsing files in isolation | `cayu new NAME --execution docker` on the agent preset: a hardened per-session sandbox your tools reach with `require_sandbox_runner(ctx)`. Don't switch to the coding preset for this. | `cayu guide references#environments` |
| Another agent for a sub-task | Subagent tools | `cayu guide references#subagents` |
| Spending or usage limits | Budgets and run limits | `cayu guide references#cost-control` |
| A public or multi-user product | `cayu new NAME --preset service` | `cayu guide references#server` |

Run every `cayu guide` command as `uv run --no-sync cayu guide ...`.

Use the Cayu Map to choose only the concepts the job needs:
`uv run --no-sync cayu guide authoring#cayu-map`. If the job observes, proposes, authorizes,
executes, verifies, or recovers an operational change, read the runnable paved
path first: `uv run --no-sync cayu guide durable-operations`.

If another capability is required, use the smallest package-shipped reference
from `uv run --no-sync cayu guide references`.

__PROVIDER_GUIDE_POINTER__

This scaffold is for local development. Deployment is a separate task.
If the requested application is public or multi-user, regenerate with
`cayu new NAME --preset service` or adopt Cayu's maintained service contract;
do not improvise product authorization around raw Cayu routes.

## Project commands

- Setup: `uv sync --extra dev`.
__DATABASE_AGENTS_PROOF_GUIDANCE__
- Application contract: `uv run --no-sync cayu guide anatomy`.
- Authoring details: `uv run --no-sync cayu guide authoring`.
- Inspect/check: `uv run --no-sync cayu inspect --json` and
  `uv run --no-sync cayu check --fail-on warning --json`.
- Hermetic proof: `uv run --no-sync pytest` and `uv run --no-sync cayu eval run`.
- Local developer/operator control plane: run `uv run --no-sync cayu serve --dev` in a separate
  terminal and open `http://127.0.0.1:8000/cayu/`. This is not the application's
  end-user UI or a production server configuration.
- First Control Plane evaluation: `uv run --no-sync cayu guide evals-first`.
- Never mount it with `OpenAccess()` on a public listener.
- Client-IP and forwarded-header checks are not authentication. Use
  `AuthenticatedAccess(...)` for any public or deployed control-plane surface.
- Live execution: `uv run --no-sync python run.py --message "USER REQUEST"` after configuring a
  provider in `app.configured_provider()`.

Use public `cayu` imports and public CLI JSON only. Do not depend on Cayu source,
private symbols, or import-time application construction.

If the job truly needs a tool, read `cayu guide tool-effects`; every tool must
declare `ToolEffect`, and effect metadata does not authorize execution. A
`ScriptedModelProvider` proves handling of predetermined calls, not prompt
comprehension or live model behavior.

For the starter's first real tool, run
`uv run --no-sync cayu generate tool TOOL_NAME --agent __AGENT_NAME__ --effect EFFECT`.
Then replace the generated sample schema, implementation, test, and eval with
domain behavior; `cayu check` keeps the tracer-bullet warning active until the
explicit authoring marker is removed.
"""

_SERVICE_SETTINGS_APPEND = '''


def configured_product_auth_tokens_json() -> str | None:
    """Return the serialized product-token mapping without interpreting secrets."""

    return os.environ.get("PRODUCT_AUTH_TOKENS_JSON")


def configured_operator_bearer_token() -> str | None:
    """Return the configured operator credential for service authentication."""

    return os.environ.get("CAYU_OPERATOR_BEARER_TOKEN")
'''


_SERVICE_PY = '''"""Maintained public-service factory for __PROJECT_NAME__."""

from __future__ import annotations

import hmac
import json
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
from configuration.settings import (
    configured_operator_bearer_token,
    configured_product_auth_tokens_json,
)
from configuration.storage import build_stores
from knowledge.retrieval import build_knowledge_scope

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
    raw = configured_product_auth_tokens_json()
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
    configured_token = _validated_bearer_token(configured_operator_bearer_token())
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
    """Build the one service used by serving, checks, docs, and security tests.

    Cayu's product operation store is the product authorization boundary. It
    lives in the configured database beside the session and task stores:
    CAYU_DATABASE_URL selects PostgreSQL, which several service processes can
    share, and local development uses SQLite at data/cayu.db.
    """

    mode = ServiceMode(mode)
    stores = build_stores(
        session_store=session_store,
        task_store=task_store,
        knowledge_scope=build_knowledge_scope(),
        product_store=product_store,
        product_operations=True,
    )
    app = build_app(
        provider=provider,
        session_store=stores.session_store,
        task_store=stores.task_store,
        knowledge_store=stores.knowledge_store,
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
    return create_agent_service(
        app,
        agent_name="__AGENT_NAME__",
        mode=mode,
        project_context=project_context,
        product_access=selected_product_access,
        operator_access=selected_operator_access,
        product_store=stores.product_store,
    )
'''

_SERVICE_SECURITY_TEST_PY = """from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import contextmanager
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
    SQLiteProductOperationStore,
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
from cayu.sessions.base import run_request_with_task_invocation
from cayu.tasks.base import task_create_with_runtime_invocation
from cayu.server import (
    AuthenticatedAccess,
    AuthenticatedProductAccess,
    BasicAuth,
    ProductOperation,
    ProductPrincipal,
    ServiceMode,
    create_agent_service,
)

from service import build_service


@contextmanager
def product_database(store):
    # Inspect or fault the runtime product store's SQLite table directly.
    connection = sqlite3.connect(store.path)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


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
            finalization=execution_profile_admission.model_finalization_material(
                max_steps=app.config.run.max_steps,
                limits=app.config.run.copy_limits(),
                retry_policy=app.config.run.retry_policy,
            ),
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
    with product_database(store) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM cayu_product_operations"
            ).fetchone()[0]
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
    with product_database(store) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM cayu_product_operations"
            ).fetchone()[0]
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
    with product_database(store) as connection:
        private = next(
            iter(connection.execute("SELECT * FROM cayu_product_operations"))
        )
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
    with product_database(store) as connection:
        private = next(
            iter(connection.execute("SELECT * FROM cayu_product_operations"))
        )
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
    with product_database(reloaded_store) as connection:
        private = next(
            iter(connection.execute("SELECT * FROM cayu_product_operations"))
        )
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
        with product_database(first_store) as connection:
            row = connection.execute(
                "SELECT status, result_receipt FROM cayu_product_operations WHERE work_id = ?",
                (reservation.operation.work_id,),
            ).fetchone()
            assert row["status"] == "pending"
            assert json.loads(row["result_receipt"])["result"] == "allow-listed answer"
            connection.execute(
                "UPDATE cayu_product_operations SET execution_claim_expires_at = 0 WHERE work_id = ?",
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
        with product_database(store) as connection:
            row = connection.execute(
                "SELECT status, result, result_receipt FROM cayu_product_operations WHERE work_id = ?",
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
        created_checkpoint = await runtime_session_store.load_checkpoint(
            created_session.id
        )
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
    with product_database(store) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM cayu_product_operations"
            ).fetchone()[0]
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
    with product_database(store) as connection:
        row = connection.execute(
            "SELECT result, result_receipt FROM cayu_product_operations"
        ).fetchone()
    result = row["result"]
    receipt = row["result_receipt"]
    assert result == "[REDACTED_SECRET]"
    assert secret not in result
    assert secret not in receipt
    assert json.loads(receipt)["result"] == "[REDACTED_SECRET]"


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
plane. Every product read is tenant-qualified through the product operation
store, which `build_service()` opens with the other stores in
`configuration/storage.py`. Never authorize from request tenant fields, Cayu
IDs, labels, metadata, model output, or tool input, and never return raw runtime
records to customers.

The product operation store is Cayu's `SQLiteProductOperationStore` locally and
`PostgresProductOperationStore` when `CAYU_DATABASE_URL` selects PostgreSQL. Its
records live in the `cayu_product_operations` table of the configured database,
so `cayu storage migrate` creates them, and several service processes can share
one PostgreSQL database. Its execution claim, heartbeat, and terminal write are
one durability contract; a replacement store must keep their atomic, same-claim
replay behavior. The worker must match the reservation's canonical agent/request
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
uv run --no-sync cayu serve --dev
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
uv run --no-sync cayu serve --host 0.0.0.0
```

The `cayu serve` listener uses plain HTTP. Put it behind a trusted
TLS-terminating ingress or reverse proxy, restrict the backend listener to that
trusted network, and expose only the HTTPS endpoint to customers and operators.
Never send either bearer token over a directly exposed HTTP connection.

The generated token map is a bounded self-hosted example, not an identity
provider. Replace its authentication dependency with your trusted application
authority while continuing to return server-derived `ProductPrincipal` values.

```bash
uv run --no-sync cayu check --deploy --fail-on warning --json
uv run --no-sync pytest -q tests/test_public_service_security.py
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

- `uv run --no-sync cayu check --deploy --fail-on warning --json`
- `uv run --no-sync pytest -q tests/test_public_service_security.py`
"""

_GITIGNORE = ".cayu/\ndata/\n__pycache__/\n*.pyc\n.pytest_cache/\n.venv/\n"


def add_new_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "new",
        help="Scaffold a new Cayu agent project.",
        description=(
            "Scaffold a new Cayu application project. Follow the printed `uv sync` "
            "and credential-free verification commands next."
        ),
    )
    parser.add_argument(
        "name",
        nargs="?",
        help="Project name (also the directory name); required unless using discovery.",
    )
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
        choices=(
            "neutral",
            "openai",
            "anthropic",
            "openrouter",
            "cayu-gateway",
            "openai-subscription",
        ),
        help=(
            "Provider adapter (default: neutral). Omit for a provider-neutral scaffold; "
            "CAYU_PROVIDER can select or override it later."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=tuple(spec.name for spec in PRESETS),
        help="Coherent application shape: agent (default), service, or coding.",
    )
    # Deprecated: every project now selects its database at runtime through
    # CAYU_DATABASE_URL. Existing scripts and generated AGENTS.md commands may
    # still pass it, so both old values are accepted and ignored with a notice.
    parser.add_argument(
        "--database",
        choices=("sqlite", "postgres"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--execution",
        choices=tuple(spec.name for spec in ADAPTERS if spec.kind == "execution"),
        help="Execution adapter (default: none; docker is admitted only for coding).",
    )
    parser.add_argument(
        "--with",
        dest="with_capabilities",
        action="append",
        default=[],
        metavar="CAPABILITY",
        help="Enable a maintained optional capability; repeat or use comma-separated names.",
    )
    parser.add_argument(
        "--without",
        dest="without_capabilities",
        action="append",
        default=[],
        metavar="CAPABILITY",
        help="Disable an optional preset capability while retaining its ownership home.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for omitted choices when attached to a terminal.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate the exact generation plan without writing files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit stable structured discovery, plan, error, or creation output.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List package-shipped maintained presets without creating a project.",
    )
    parser.add_argument(
        "--list-capabilities",
        action="store_true",
        help="List selectable, preset-owned, and extension-only capability seams.",
    )
    parser.add_argument(
        "--explain",
        metavar="CAPABILITY",
        help="Explain one package-shipped capability without creating a project.",
    )
    parser.add_argument(
        "--coding-toolchain",
        choices=("python",),
        help=(
            "Explicit admitted profile for Docker coding execution. The generated "
            "Python profile is the first built-in; applications may register custom "
            "DockerCodingToolchainProfile values in environments/coding.py."
        ),
    )
    parser.add_argument(
        "--coding-command-authority",
        choices=("structured",),
        help=(
            "Model-facing command authority for Docker coding. The maintained "
            "surface accepts only application-owned structured selectors."
        ),
    )


def _installed_cayu_version() -> str:
    return package_version()


@dataclass(frozen=True)
class _CayuSourceCheckout:
    root: Path
    version: str


def _cayu_source_checkout(package_dir: Path | None = None) -> _CayuSourceCheckout | None:
    """Return the source checkout this Cayu runs from, or None for a release install.

    An editable or ``PYTHONPATH`` install runs a checkout's code, and its installed
    metadata can be stale. A ``cayu==<version>`` pin would then install a PyPI
    release with different code, so projects point at the checkout instead. A
    release install lives under site-packages and keeps the PyPI pin.
    """

    package = (
        Path(__file__).resolve().parents[1] if package_dir is None else package_dir
    ).resolve()
    if any(part in {"site-packages", "dist-packages"} for part in package.parts):
        return None
    # Flat (``cayu/``) or src (``src/cayu/``) layout.
    for root in (package.parent, package.parent.parent):
        try:
            document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            continue
        project = document.get("project")
        if (
            isinstance(project, dict)
            and project.get("name") == "cayu"
            and isinstance(project.get("version"), str)
        ):
            return _CayuSourceCheckout(root=root, version=project["version"])
    return None


def _uses_cayu_source_checkout(plan: ApplicationPlan) -> bool:
    # The coding Docker image runs `uv sync --frozen` without the host checkout;
    # it installs unreleased Cayu through the reviewed `cayu_wheel` instead.
    return not (plan.preset == "coding" and plan.execution == "docker")


def project_files(
    name: str,
    *,
    agent_name: str | None = None,
    provider: str | None = None,
    coding_toolchain: str | None = None,
    coding_command_authority: str | None = None,
    preset: str | None = None,
    execution: str | None = None,
    with_capabilities: tuple[str, ...] = (),
    without_capabilities: tuple[str, ...] = (),
    application_plan: ApplicationPlan | None = None,
) -> dict[str, str]:
    resolved_agent_name = name if agent_name is None else agent_name
    if application_plan is None:
        application_plan = normalize_application_plan(
            name=name,
            agent_name=resolved_agent_name,
            preset=preset or "agent",
            provider=provider or "neutral",
            execution=execution or "none",
            coding_toolchain=coding_toolchain,
            coding_command_authority=coding_command_authority,
            with_capabilities=with_capabilities,
            without_capabilities=without_capabilities,
        )
    plan = application_plan
    resolved_agent_name = plan.agent_name
    provider = plan.provider_alias
    if coding_toolchain is not None and coding_toolchain != plan.coding_toolchain:
        raise ValueError("coding_toolchain conflicts with the normalized plan.")
    if (
        coding_command_authority is not None
        and coding_command_authority != plan.coding_command_authority
    ):
        raise ValueError("coding_command_authority conflicts with the normalized plan.")
    reviewer_name = f"{resolved_agent_name}-reviewer"
    source_checkout = _cayu_source_checkout()
    # A checkout's own version, not its possibly stale installed metadata.
    version = _installed_cayu_version() if source_checkout is None else source_checkout.version
    source_path = (
        json.dumps(str(source_checkout.root), ensure_ascii=False).replace("\x7f", r"\u007f")
        if source_checkout is not None
        else ""
    )
    uv_sources = (
        f"\n[tool.uv.sources]\ncayu = {{ path = {source_path}, editable = true }}\n"
        if source_checkout is not None and _uses_cayu_source_checkout(plan)
        else ""
    )
    # Every project can switch to PostgreSQL through CAYU_DATABASE_URL.
    runtime_extra = "[postgres,server]" if plan.preset == "service" else "[postgres]"
    dev_dependencies = ["pytest"]
    if plan.preset != "service":
        dev_dependencies.insert(0, f"cayu[postgres,server]=={version}")
    if plan.preset == "service" or (plan.preset == "coding" and plan.execution == "docker"):
        dev_dependencies.append("ruff>=0.15.15,<0.16")

    def render(
        template: str,
        additional_replacements: dict[str, str] | None = None,
    ) -> str:
        provider_display = provider or "no live provider"
        provider_literal = "None" if provider is None else json.dumps(provider)
        knowledge_selected = "knowledge" in plan.capabilities
        database_readme_proof_guidance = (
            "Run setup and proof commands in the listed order. Do not parallelize "
            "commands that construct the application against the same database; "
            "first use of local SQLite may initialize or migrate its schema."
        )
        database_agents_proof_guidance = (
            "- Run setup and proof commands sequentially. Do not parallelize application-\n"
            "  constructing commands against the same database."
        )
        database_evals_storage_guidance = (
            "the configured durable Evals store (`CAYU_DATABASE_URL`, else "
            f"`{local_database_path(plan)}`)"
        )
        coding_database_summary = (
            "durable knowledge" if knowledge_selected else "no configured knowledge store or tools"
        )
        coding_state_storage = (
            "Artifact state is stored below that protected `.cayu` boundary; "
            + (
                "session, task, and knowledge state lives in the configured database "
                "(`CAYU_DATABASE_URL`, else `.cayu/runtime/cayu.db`). Use the registered "
                "Git, artifact, and knowledge tools at their authenticated boundaries instead."
                if knowledge_selected
                else "session and task state lives in the configured database "
                "(`CAYU_DATABASE_URL`, else `.cayu/runtime/cayu.db`). "
                "No knowledge store or knowledge tools are configured."
            )
        )
        if plan.preset == "coding" and not {"tasks", "artifacts"} <= set(plan.capabilities):
            states = ["session"] + [
                item for item in ("tasks", "knowledge") if item in plan.capabilities
            ]
            coding_state_storage = (
                ", ".join(states).capitalize()
                + " state uses the configured database (`CAYU_DATABASE_URL`, else the "
                + "protected `.cayu/runtime/cayu.db`). "
                + (
                    "Artifacts use protected `.cayu` storage. "
                    if "artifacts" in plan.capabilities
                    else "Artifacts are not configured. "
                )
                + (
                    "No knowledge store or knowledge tools are configured."
                    if not knowledge_selected
                    else "Use knowledge tools through their scoped boundary."
                )
            )
        replacements = {
            "__PROJECT_NAME__": name,
            "__PRESET_OVERVIEW__": (
                "A maintained two-agent coding composition for a trusted Git repository. Its primary\nagent and bounded reviewer use generated repository tools, policy, knowledge,\ndelegation, and human-input seams that are part of this preset rather than optional\nadditions to a model-only starter."
                if plan.preset == "coding"
                and {"delegation", "knowledge", "human-input"} <= set(plan.capabilities)
                else "A maintained coding composition for a trusted Git repository. Its explicit\nconstructors expose only the capabilities selected in the scaffold profile."
                if plan.preset == "coding"
                else "A Cayu application with the standard capability layout. Its registered agent\nidentity is `__AGENT_NAME__`. The scaffold profile records its selected capabilities.".replace(
                    "__AGENT_NAME__", resolved_agent_name
                )
            ),
            "__AGENT_OWNERSHIP__": (
                "This preset registers a primary coding agent and a bounded reviewer. Extend the\nprimary through the canonical generated regions in `agents/agent.py` and\n`agents/registration.py`. Keep the reviewer tool-free unless a reviewed composition\nchange intentionally expands its role.\nDo not create echo, pass-through, or placeholder tools."
                if plan.preset == "coding" and "delegation" in plan.capabilities
                else "The registered agent identity is `__AGENT_NAME__`.\n\nEdit the existing agent, test, and eval to implement the user's first requested\njob. Do not retain the starter and add a second agent. Tools are registered in\n`agents/registration.py`; change their policies deliberately. Do not create echo,\npass-through, or placeholder tools.".replace(
                    "__AGENT_NAME__", resolved_agent_name
                )
            ),
            "__AGENT_NAME__": resolved_agent_name,
            "__REVIEWER_NAME__": reviewer_name,
            "__CAYU_VERSION__": version,
            "__RUNTIME_DEPENDENCIES__": json.dumps([f"cayu{runtime_extra}=={version}"]),
            "__DEV_DEPENDENCIES__": json.dumps(dev_dependencies),
            "__UV_SOURCES__": uv_sources,
            "__SERVICE_FACTORY__": (
                'service_factory = "service:build_service"\n' if plan.preset == "service" else ""
            ),
            "__EVAL_TARGET__": (
                'eval_target = "evals.agent:build_eval"\n' if "evals" in plan.capabilities else ""
            ),
            # Local tooling default; CAYU_DATABASE_URL overrides it for the CLI and app.
            "__SESSION_STORE__": f'backend = "sqlite"\npath = "{local_database_path(plan)}"',
            "__PROVIDER_DISPLAY__": provider_display,
            "__PROVIDER_LITERAL__": provider_literal,
            "__PROVIDER_GUIDE_POINTER__": _PROVIDER_GUIDE_POINTER,
            "__DATABASE_README_PROOF_GUIDANCE__": database_readme_proof_guidance,
            "__DATABASE_AGENTS_PROOF_GUIDANCE__": database_agents_proof_guidance,
            "__DATABASE_EVALS_STORAGE_GUIDANCE__": database_evals_storage_guidance,
            "__CODING_DATABASE_SUMMARY__": coding_database_summary,
            "__CODING_STATE_STORAGE__": coding_state_storage,
        }
        if additional_replacements is not None:
            replacements.update(additional_replacements)
        token_re = re.compile(
            "|".join(re.escape(token) for token in sorted(replacements, key=len, reverse=True))
        )
        return token_re.sub(
            lambda match: replacements[match.group(0)],
            template,
        )

    files = {
        "run.py": _RUN_PY,
        "agents/__init__.py": "",
        "tests/test_agent.py": render(_TEST_PY),
        "evals/__init__.py": "",
        "evals/agent.py": render(_EVAL_PY),
        "pyproject.toml": render(_PYPROJECT),
        "README.md": render(_README),
        "AGENTS.md": render(_AGENTS_MD),
        ".gitignore": _GITIGNORE,
    }
    files.update(convention_files(plan, render=render))
    if "knowledge" in plan.capabilities:
        for relative in ("tests/test_agent.py", "evals/agent.py"):
            files[relative] = (
                files[relative]
                .replace(
                    "    InMemorySessionStore,\n",
                    "    InMemoryKnowledgeStore,\n    InMemorySessionStore,\n",
                )
                .replace(
                    "        task_store=InMemoryTaskStore(),\n",
                    "        task_store=InMemoryTaskStore(),\n"
                    "        knowledge_store=InMemoryKnowledgeStore(),\n",
                )
            )
    files["pyproject.toml"] += scaffold_contract(plan)
    if "tasks" not in plan.capabilities:
        for relative in ("tests/test_agent.py", "evals/agent.py"):
            files[relative] = files[relative].replace(
                "task_store=InMemoryTaskStore(),", "task_store=None,"
            )
    files["README.md"] += application_guidance(plan)
    files["AGENTS.md"] += application_guidance(plan)
    if "evals" not in plan.capabilities:
        files.pop("evals/agent.py", None)
        for guidance_name in ("README.md", "AGENTS.md"):
            files[guidance_name] = files[guidance_name].replace(
                "uv run --no-sync cayu eval run",
                "evals are not configured in this profile",
            )
    if plan.preset == "coding":
        from cayu.cli.coding_composition import coding_project_files

        files.update(
            coding_project_files(
                files=files,
                render=render,
                plan=plan,
            )
        )
        return files
    if plan.preset == "agent":
        return files
    files.update(
        {
            "configuration/settings.py": (
                files["configuration/settings.py"].rstrip() + _SERVICE_SETTINGS_APPEND
            ),
            "service.py": render(_SERVICE_PY),
            "tests/test_public_service_security.py": _SERVICE_SECURITY_TEST_PY,
            "README.md": files["README.md"] + _SERVICE_GUIDANCE,
            "AGENTS.md": files["AGENTS.md"] + _SERVICE_AGENTS_GUIDANCE,
        }
    )
    return files


def _new_error(*, code: str, message: str, as_json: bool) -> int:
    if as_json:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "error",
                    "error": {"code": code, "message": message},
                },
                sort_keys=True,
            )
        )
    else:
        print(f"error: {message}", file=sys.stderr)
    return 1


def _run_new_discovery(args: argparse.Namespace) -> int | None:
    requests = sum(
        (
            bool(args.list_presets),
            bool(args.list_capabilities),
            args.explain is not None,
        )
    )
    if requests == 0:
        return None
    if requests > 1:
        return _new_error(
            code="DISCOVERY_SELECTION_CONFLICT",
            message="choose only one of --list-presets, --list-capabilities, or --explain",
            as_json=args.json,
        )
    if args.list_presets:
        payload: object = {
            "schema_version": 1,
            "presets": [spec.as_dict() for spec in PRESETS],
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("Cayu application presets:")
            for spec in PRESETS:
                print(f"  {spec.name:<10} {spec.summary}")
        return 0
    if args.list_capabilities:
        payload = {
            "schema_version": 1,
            "capabilities": [spec.as_dict() for spec in CAPABILITIES],
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("Cayu application capabilities:")
            for spec in CAPABILITIES:
                print(f"  {spec.name:<14} {spec.status:<14} {spec.summary}")
        return 0
    try:
        spec = capability_spec(args.explain)
    except ScaffoldPlanError as exc:
        return _new_error(code=exc.code.upper(), message=str(exc), as_json=args.json)
    if args.json:
        print(
            json.dumps(
                {"schema_version": 1, "capability": spec.as_dict()}, indent=2, sort_keys=True
            )
        )
    else:
        print(f"{spec.name} ({spec.status})")
        print(spec.summary)
        print("Supported presets: " + ", ".join(spec.supported_presets))
        if spec.extension_presets:
            print("Explicit extension presets: " + ", ".join(spec.extension_presets))
            print("Declaration: [tool.cayu.scaffold].extensions (owning-module wiring required)")
            print("Guide: cayu guide applications#explicit-service-extensions")
        if spec.implied:
            print("Implies: " + ", ".join(spec.implied))
        if spec.files:
            print("Canonical homes: " + ", ".join(spec.files))
    return 0


def _resolve_new_plan(args: argparse.Namespace, *, name: str) -> ApplicationPlan:
    preset = args.preset or "agent"
    execution = args.execution or "none"
    if args.coding_toolchain is not None and execution != "docker":
        raise ScaffoldPlanError(
            "coding_toolchain_requires_docker",
            "--coding-toolchain requires --execution docker",
        )
    if args.coding_command_authority is not None and execution != "docker":
        raise ScaffoldPlanError(
            "coding_command_authority_requires_docker",
            "--coding-command-authority requires --execution docker",
        )
    agent_name = name if args.agent_name is None else args.agent_name
    return normalize_application_plan(
        name=name,
        agent_name=agent_name,
        preset=preset,
        provider=args.provider or "neutral",
        execution=execution,
        coding_toolchain=args.coding_toolchain,
        coding_command_authority=args.coding_command_authority,
        with_capabilities=tuple(args.with_capabilities),
        without_capabilities=tuple(args.without_capabilities),
    )


def _agent_context(plan: ApplicationPlan) -> dict[str, object]:
    return {
        "instructions": "AGENTS.md",
        "cross_agent_bridge": "CLAUDE.md",
        "scaffold_contract": "pyproject.toml:[tool.cayu.scaffold]",
        "selected_plan": {
            "preset": plan.preset,
            "provider": plan.provider,
            "execution": plan.execution,
            "coding_toolchain": plan.coding_toolchain,
            "coding_command_authority": plan.coding_command_authority,
            "capabilities": list(plan.capabilities),
        },
        "authoring_map": "uv run --no-sync cayu guide authoring#cayu-map --json",
        "inspection": "uv run --no-sync cayu inspect --json",
        "verification_commands": list(plan.verification_commands()),
    }


def _initialize_coding_git(
    *,
    staging: Path,
    files: dict[str, str],
    git: str,
    hooks: Path,
    assert_staging_unchanged: Callable[[], object] | None = None,
) -> None:
    staging_identity = _scaffold_directory_identity(staging)
    git_env = _sanitized_scaffold_git_environment(cwd=staging)
    _run_scaffold_git_command(
        _safe_git_argv(
            git,
            "init",
            "-b",
            "main",
            f"--template={hooks}",
            hooks_dir=hooks,
        ),
        cwd=staging,
        env=git_env,
        expected_directory_identity=staging_identity,
        assert_directory_unchanged=assert_staging_unchanged,
    )
    repository_root = _run_scaffold_git_command(
        _safe_git_argv(git, "rev-parse", "--show-toplevel", hooks_dir=hooks),
        cwd=staging,
        env=git_env,
        expected_directory_identity=staging_identity,
        assert_directory_unchanged=assert_staging_unchanged,
    ).strip()
    try:
        resolved_repository_root = Path(repository_root).resolve(strict=True)
    except (OSError, ValueError):
        raise _ScaffoldCommandError("git repository root verification failed") from None
    if resolved_repository_root != staging.resolve(strict=True):
        raise _ScaffoldCommandError(
            "git repository authority escaped the scaffold staging directory"
        )
    _run_scaffold_git_command(
        _safe_git_argv(git, "add", "--force", "--", ".", hooks_dir=hooks),
        cwd=staging,
        env=git_env,
        expected_directory_identity=staging_identity,
        assert_directory_unchanged=assert_staging_unchanged,
    )
    tracked_output = _run_scaffold_git_command(
        _safe_git_argv(git, "ls-files", "--cached", "-z", "--", hooks_dir=hooks),
        cwd=staging,
        env=git_env,
        expected_directory_identity=staging_identity,
        assert_directory_unchanged=assert_staging_unchanged,
    )
    tracked_files = frozenset(path for path in tracked_output.split("\0") if path)
    if tracked_files != frozenset(files):
        raise _ScaffoldCommandError("git index does not contain the complete generated project")
    _run_scaffold_git_command(
        _safe_git_argv(
            git,
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
        assert_directory_unchanged=assert_staging_unchanged,
    )


def _publish_new_project(
    *,
    target: Path,
    files: dict[str, str],
    plan: ApplicationPlan,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _require_safe_scaffold_parent(target.parent)
    try:
        request_digest = _scaffold_publication_request_digest(
            files=files,
            plan=plan,
        )
        publish_guarded_tree(
            target,
            consumer="scaffold_new",
            request_digest=request_digest,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            settle_active_operation=True,
            populate=lambda staging: _populate_scaffold_stage(
                staging=staging,
                files=files,
                plan=plan,
                hooks_parent=target.parent,
            ),
        )
    except BaseException as error:
        if isinstance(error, GuardedTreePublicationError):
            if error.code == "destination_not_empty":
                raise _ScaffoldTargetNotEmpty from None
            raise _translated_scaffold_publication_error(error) from error
        raise


def _render_new_receipt(
    *,
    target: Path,
    plan: ApplicationPlan,
    coding_toolchain: str | None,
) -> None:
    print(f"Scaffolded {target}/ — Cayu application convention {plan.convention}")
    print(f"  Plan: preset={plan.preset} provider={plan.provider} execution={plan.execution}")
    capabilities = ", ".join(plan.capabilities) if plan.capabilities else "none"
    print(f"  Capabilities: {capabilities}")
    if "memory" in plan.capabilities:
        print("  Private runtime state: data/memory-evidence.key (ignored; mode 0600 on POSIX)")
    print("  Agent instructions: AGENTS.md (CLAUDE.md imports the same contract)")
    print("  Scaffold contract: pyproject.toml [tool.cayu.scaffold]")
    source_checkout = _cayu_source_checkout()
    if source_checkout is not None and _uses_cayu_source_checkout(plan):
        print(f"  Cayu source: {source_checkout.root} (local checkout, editable)")
    print(f"  cd {target}")
    if plan.preset == "coding" and plan.execution == "docker":
        print(
            "  Before proof: uv run --no-sync python build_coding_image.py --resolve-pins, "
            "then review docker-coding-build.json"
        )
    if plan.preset == "agent" and plan.execution == "docker":
        print(
            "  Execution: hardened Docker sandbox for the agent's own tools "
            "(network disabled; data/sandbox is synced to /workspace)"
        )
    for command in plan.verification_commands():
        label = (
            "Build and record image: "
            if command == "uv run --no-sync python build_coding_image.py"
            else ""
        )
        print(f"  {label}{command}")
    if plan.preset == "coding" and plan.execution == "docker":
        print(f"  Toolchain profile: {coding_toolchain or 'python'}")
        print(
            "  Execution: admitted trusted-repository Docker checks and commands (network disabled)"
        )
    if plan.preset == "coding":
        print(
            f"  Live run: uv run --no-sync python run.py --agent {plan.agent_name} "
            '--message "YOUR REQUEST"'
        )
    if plan.preset == "service":
        print("  Local public service: uv run --no-sync cayu serve --dev")
        print("  Product API: http://127.0.0.1:8000/api/operations")
        print("  Operator control plane: http://127.0.0.1:8000/cayu/")
    else:
        print("  Local control plane: uv run --no-sync cayu serve --dev")
        print("  Open: http://127.0.0.1:8000/cayu/")
    if plan.provider == "neutral":
        print("  Live provider: none selected; set CAYU_PROVIDER explicitly before `run.py`.")
    else:
        print(f"  Live provider: {plan.provider} (credentials authenticate this choice).")


def run_new(args: argparse.Namespace) -> int:
    """Plan, render, validate, and atomically publish one Cayu application."""

    discovered = _run_new_discovery(args)
    if discovered is not None:
        return discovered
    if args.interactive and args.json:
        return _new_error(
            code="INTERACTIVE_JSON_CONFLICT",
            message="--interactive cannot be combined with --json",
            as_json=True,
        )
    if args.database is not None:
        print(
            "cayu new: --database is deprecated and ignored; every project selects "
            "PostgreSQL through CAYU_DATABASE_URL and uses local SQLite otherwise.",
            file=sys.stderr,
        )
    name = args.name
    if args.interactive:
        if not sys.stdin.isatty():
            return _new_error(
                code="INTERACTIVE_TTY_REQUIRED",
                message="--interactive requires a terminal",
                as_json=False,
            )
        if name is None:
            name = input("Project name: ").strip()
    if name is None:
        return _new_error(
            code="PROJECT_NAME_REQUIRED",
            message="project name is required (or use a discovery option)",
            as_json=args.json,
        )
    if not _NAME_RE.fullmatch(name):
        return _new_error(
            code="INVALID_PROJECT_NAME",
            message=(
                f"invalid project name {name!r} "
                "(use letters, digits, '-' or '_', starting with a letter)"
            ),
            as_json=args.json,
        )
    try:
        plan = _resolve_new_plan(args, name=name)
    except ScaffoldPlanError as exc:
        return _new_error(code=exc.code.upper(), message=str(exc), as_json=args.json)
    if not _NAME_RE.fullmatch(plan.agent_name):
        return _new_error(
            code="INVALID_AGENT_NAME",
            message=(
                f"invalid agent name {plan.agent_name!r} "
                "(use letters, digits, '-' or '_', starting with a letter)"
            ),
            as_json=args.json,
        )

    target = Path(args.dir) / name
    if target.is_symlink():
        return _new_error(
            code="TARGET_IS_SYMLINK",
            message=f"{target} cannot be a symbolic link",
            as_json=args.json,
        )
    if target.exists() and not target.is_dir():
        return _new_error(
            code="TARGET_NOT_DIRECTORY",
            message=f"{target} already exists and is not a directory",
            as_json=args.json,
        )
    if args.dry_run and target.exists() and any(target.iterdir()):
        return _new_error(
            code="TARGET_NOT_EMPTY",
            message=f"{target} already exists and is not empty",
            as_json=args.json,
        )
    try:
        files = project_files(
            name,
            coding_toolchain=args.coding_toolchain,
            coding_command_authority=args.coding_command_authority,
            application_plan=plan,
        )
    except (ScaffoldPlanError, ValueError) as exc:
        code = exc.code.upper() if isinstance(exc, ScaffoldPlanError) else "PLAN_RENDER_FAILED"
        return _new_error(code=code, message=str(exc), as_json=args.json)
    directories = {"data"}
    if "artifacts" in plan.capabilities:
        directories.add("data/artifacts")
    directories.update(parent for path in files if (parent := str(Path(path).parent)) != ".")
    payload = {
        "status": "planned" if args.dry_run else "created",
        "target": str(target),
        "plan": plan.as_dict(
            files=tuple(sorted(files)),
            directories=tuple(sorted(directories)),
            private_files=(("data/memory-evidence.key",) if "memory" in plan.capabilities else ()),
        ),
        "agent_context": _agent_context(plan),
    }
    if args.dry_run:
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Plan for {target}/ ({len(files)} files, no writes):")
            print(f"  preset={plan.preset} provider={plan.provider} execution={plan.execution}")
            for relative in sorted(files):
                print(f"  create {relative}")
            if "memory" in plan.capabilities:
                print("  create private data/memory-evidence.key (content generated on apply)")
        return 0

    try:
        _publish_new_project(
            target=target,
            files=files,
            plan=plan,
        )
    except _ScaffoldCodingPreflightError as exc:
        return _new_error(
            code="CODING_PREFLIGHT_FAILED",
            message=_scaffold_error_message(exc),
            as_json=args.json,
        )
    except _ScaffoldTargetNotEmpty:
        return _new_error(
            code="TARGET_NOT_EMPTY",
            message=f"{target} already exists and is not empty",
            as_json=args.json,
        )
    except (_ScaffoldCommandError, OSError, ValueError) as exc:
        return _new_error(
            code="SCAFFOLD_PUBLICATION_FAILED",
            message=f"could not publish scaffold: {_scaffold_error_message(exc)}",
            as_json=args.json,
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_new_receipt(
            target=target,
            plan=plan,
            coding_toolchain=args.coding_toolchain,
        )
    return 0
