"""Package-shipped templates for the Cayu application convention."""

from __future__ import annotations

from collections.abc import Callable

from cayu.cli.scaffold_plan import CAPABILITIES, ApplicationPlan, preset_spec

_APP_PY = '''"""Composition root for __PROJECT_NAME__.

This module constructs and wires the application. Implement prompts, tools,
policies, environments, workflows, operations, domain rules, and integrations
in their owning packages, then connect them through the explicit registration
seams imported here.
"""

from cayu import (
    CayuApp,
    ModelProvider,
    SessionStore,
    TaskStore,
)

from agents.registration import register_agents
from configuration.providers import (
    configured_provider,
    validate_run_configuration as validate_run_configuration,
)
from configuration.runtime import build_runtime_options
from configuration.storage import build_stores


def build_app(
    *,
    provider: ModelProvider | None = None,
    session_store: SessionStore | None = None,
    task_store: TaskStore | None = None,
) -> CayuApp:
    """Construct a fresh process-scoped application graph.

    Injected stores and providers are public hermetic-test seams. Importing this
    module never constructs the application or connects to an external service.
    """

    stores = build_stores(session_store=session_store, task_store=task_store)
    runtime = build_runtime_options()
    app = CayuApp(
        session_store=stores.session_store,
        task_store=stores.task_store,
        enable_logging=runtime.enable_logging,
    )
    selected_provider = provider if provider is not None else configured_provider()
    app.register_provider(selected_provider, default=True)
    register_agents(app, provider_override=provider)
    return app
'''

_CONFIGURATION_INIT_PY = '''"""Validated application configuration and adapter construction."""

from configuration.providers import (
    configured_provider as configured_provider,
    validate_run_configuration as validate_run_configuration,
)
from configuration.settings import (
    configured_model as configured_model,
    configured_provider_choice as configured_provider_choice,
    configured_provider_name as configured_provider_name,
)

__all__ = [
    "configured_model",
    "configured_provider",
    "configured_provider_choice",
    "configured_provider_name",
    "validate_run_configuration",
]
'''

_PROVIDERS_PY = '''"""Explicit provider construction for this application."""

import os

from cayu import (
    AnthropicProvider,
    CayuApp,
    ChatCompletionsProvider,
    ModelProvider,
    OpenAIProvider,
    OpenAISubscriptionProvider,
    ScriptedModelProvider,
)

from configuration.settings import (
    SCAFFOLDED_DATABASE,
    configured_provider_choice,
)


class _ScaffoldPlaceholderProvider(ScriptedModelProvider):
    """Credential-free placeholder rejected by every runtime entry point."""

    def __init__(self, *, name: str, setup_error: str) -> None:
        super().__init__([], name=name)
        self._setup_error = setup_error

    def preflight_model_target(self, *, model: str) -> None:
        del model
        raise RuntimeError(self._setup_error)


def configured_provider() -> ModelProvider:
    """Construct only the explicitly selected provider without dispatching it."""

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
    """Run the target and adapter preflight used by live entry points."""

    if SCAFFOLDED_DATABASE == "postgres" and not os.environ.get("CAYU_DATABASE_URL"):
        raise RuntimeError(
            "database 'postgres' is selected but CAYU_DATABASE_URL is not set"
        )
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
'''

_SETTINGS_PY = '''"""Validated source-controlled defaults and environment overrides."""

import os

SCAFFOLDED_DATABASE = "__DATABASE__"
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
    """Return explicit project or environment selection, never credential inference."""

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
        return "openrouter-model-unconfigured"
    return (
        "provider-model-unconfigured" if selected is None else _DEFAULT_MODELS[selected]
    )
'''

_SQLITE_STORAGE_PY = '''"""Construct the selected durable application stores."""

from dataclasses import dataclass

from cayu import (
    SessionStore,
    SQLiteSessionStore,
    SQLiteTaskStore,
    TaskStore,
    public_authority_alias_codec_from_environment,
)


@dataclass(frozen=True, slots=True)
class ApplicationStores:
    session_store: SessionStore
    task_store: TaskStore


def build_stores(
    *,
    session_store: SessionStore | None = None,
    task_store: TaskStore | None = None,
) -> ApplicationStores:
    """Build SQLite stores unless a caller injects hermetic test stores."""

    return ApplicationStores(
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
'''

_POSTGRES_STORAGE_PY = '''"""Construct the selected durable Postgres application stores."""

import os
from dataclasses import dataclass

from cayu import (
    PostgresSessionStore,
    PostgresTaskStore,
    SessionStore,
    TaskStore,
    public_authority_alias_codec_from_environment,
)

_INSPECTION_DSN = "postgresql://cayu-unconfigured@127.0.0.1/cayu"


@dataclass(frozen=True, slots=True)
class ApplicationStores:
    session_store: SessionStore
    task_store: TaskStore


def build_stores(
    *,
    session_store: SessionStore | None = None,
    task_store: TaskStore | None = None,
) -> ApplicationStores:
    """Build lazy Postgres stores without connecting during import or inspection."""

    conninfo = os.environ.get("CAYU_DATABASE_URL") or _INSPECTION_DSN
    return ApplicationStores(
        session_store=(
            session_store
            if session_store is not None
            else PostgresSessionStore(
                conninfo,
                public_authority_alias_codec=public_authority_alias_codec_from_environment(),
            )
        ),
        task_store=(
            task_store if task_store is not None else PostgresTaskStore(conninfo)
        ),
    )
'''

_AGENT_PY = '''"""Narrow AgentSpec declaration for the starter agent."""

from cayu import AgentSpec

from configuration import configured_model, configured_provider_name
from prompts.agent import SYSTEM_PROMPT_PARTS

# Generated first-tool imports and agent contract additions live in these regions.
# <cayu:generated-agent-imports>
# </cayu:generated-agent-imports>

_SYSTEM_PROMPT_PARTS: list[str] = list(SYSTEM_PROMPT_PARTS)
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
'''

_PROMPT_PY = '''"""System and developer prompt material for the starter agent."""

SYSTEM_PROMPT_PARTS: tuple[str, ...] = ()
'''

_TOOLS_REGISTRATION_PY = '''"""Native model-callable tools selected for the starter agent."""

from cayu import Tool


def build_agent_tools() -> tuple[Tool, ...]:
    """Return explicitly constructed tools; the model-only starter has none."""

    return ()


def external_effect_tool_names() -> tuple[str, ...]:
    """Return tools that require the generated approval policy."""

    return ()
'''

_TOOL_POLICY_PY = '''"""Tool exposure and authorization policy for registered agents."""

from cayu import AlwaysRequireApprovalToolPolicy, ToolPolicy


def build_tool_policy(external_tool_names: tuple[str, ...]) -> ToolPolicy | None:
    """Require approval for every generated externally effectful tool."""

    if not external_tool_names:
        return None
    return AlwaysRequireApprovalToolPolicy(tools=external_tool_names)
'''

_RUNTIME_PY = '''"""Application-wide runtime policy and collaborator construction seam."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    enable_logging: bool


def build_runtime_options() -> RuntimeOptions:
    """Construct the selected event-sink profile without starting lifecycle work."""

    return RuntimeOptions(enable_logging=__ENABLE_LOGGING__)
'''

_AGENT_REGISTRATION_PY = '''"""Explicit agent, tool, policy, and runtime registration seam."""

from cayu import AgentSpec, CayuApp, ModelProvider

from agents.agent import AGENT
from policies.tools import build_tool_policy
from tools.registration import build_agent_tools, external_effect_tool_names

# Generated tool-backed slices add imports only inside this owned region.
# <cayu:generated-imports>
# </cayu:generated-imports>


def _agent_for_provider_override(
    agent: AgentSpec, provider: ModelProvider | None
) -> AgentSpec:
    if provider is None:
        return agent
    return agent.model_copy(update={"provider_name": provider.name})


def register_agents(
    app: CayuApp, *, provider_override: ModelProvider | None = None
) -> None:
    """Register every agent and its explicitly constructed capabilities."""

    starter_tools = list(build_agent_tools())
    starter_external_tool_names = list(external_effect_tool_names())
    # <cayu:generated-starter-tools>
    # </cayu:generated-starter-tools>
    app.register_agent(
        _agent_for_provider_override(AGENT, provider_override),
        tools=starter_tools,
        tool_policy=build_tool_policy(tuple(starter_external_tool_names)),
    )
    # <cayu:generated-registrations>
    # </cayu:generated-registrations>
'''

_TEST_APPLICATION_PY = '''"""Composition-root contract tests."""

from cayu import InMemorySessionStore, InMemoryTaskStore, ScriptedModelProvider

from app import build_app


def test_factory_returns_fresh_apps_and_preserves_injected_stores() -> None:
    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    first = build_app(
        provider=ScriptedModelProvider([]),
        session_store=sessions,
        task_store=tasks,
    )
    second = build_app(
        provider=ScriptedModelProvider([]),
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
    )

    assert first is not second
    assert first.session_store is sessions
    assert first.task_store is tasks
'''

_TEST_ARCHITECTURE_PY = '''"""Declared Cayu application convention tests."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
REQUIRED_HOMES = (
    "configuration",
    "agents",
    "prompts",
    "tools",
    "policies",
    "environments",
    "workflows",
    "operations",
    "knowledge",
    "memory",
    "domain",
    "integrations",
    "evals",
    "observability",
)


def test_declared_convention_homes_exist() -> None:
    for relative in REQUIRED_HOMES:
        package = ROOT / relative
        assert package.is_dir(), relative
        assert (package / "__init__.py").is_file(), relative


def test_app_is_a_composition_root() -> None:
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]

    assert classes == []
    assert functions == __PUBLIC_APP_FACTORIES__
'''

_CLAUDE_MD = "@AGENTS.md\n"

_OWNERSHIP_FILES: dict[str, str] = {
    "prompts/__init__.py": '"""System and developer prompt material."""\n',
    "tools/__init__.py": '"""Native model-callable capabilities and ToolEffect declarations."""\n',
    "policies/__init__.py": '"""Exposure, authorization, approval, execution, egress, and budget policy."""\n',
    "policies/context.py": '"""Context selection, overflow, and compaction policy extension seam."""\n',
    "policies/execution.py": '"""Execution admission and isolation policy extension seam."""\n',
    "policies/egress.py": '"""Network egress and credential-release policy extension seam."""\n',
    "policies/budgets.py": '"""Token, cost, time, and operation budget policy extension seam."""\n',
    "policies/retries.py": '"""Provider and operation retry policy extension seam."""\n',
    "environments/__init__.py": '"""Per-session workspaces, runners, artifacts, vaults, and resources."""\n',
    "environments/registration.py": '"""Explicit environment factory and environment registration seam."""\n',
    "environments/local.py": '"""Trusted-local environment construction owned by the application."""\n',
    "workflows/__init__.py": '"""Deterministic multi-step orchestration owned by the application."""\n',
    "operations/__init__.py": '"""Durable task, worker, approval, completion, and recovery behavior."""\n',
    "operations/tasks.py": '"""Task declaration, dispatch, and durable task-state boundaries."""\n',
    "operations/workers.py": '"""Explicit worker construction and lifecycle entry points."""\n',
    "operations/watchers.py": '"""Durable event watcher construction and delivery policy."""\n',
    "operations/approvals.py": '"""Human input, approval requests, and settlement boundaries."""\n',
    "operations/completion.py": '"""Completion verification and result resolution boundaries."""\n',
    "operations/recovery.py": '"""Application-owned recovery and reconciliation entry points."""\n',
    "knowledge/__init__.py": '"""Reviewed durable knowledge and retrieval behavior."""\n',
    "knowledge/retrieval.py": '"""Knowledge retrieval sources and ranking configuration."""\n',
    "knowledge/curation.py": '"""Knowledge review, curation, and publication behavior."""\n',
    "knowledge/maintenance.py": '"""Knowledge maintenance planning and execution behavior."""\n',
    "knowledge/seeds/.gitkeep": "",
    "memory/__init__.py": '"""Model-facing context, recall, compaction, and memory attribution."""\n',
    "memory/context.py": '"""Context assembly and memory-attribution wiring."""\n',
    "memory/recall.py": '"""Recall policy, sources, and intervention wiring."""\n',
    "domain/__init__.py": '"""Business rules and data transformations independent of Cayu wiring."""\n',
    "integrations/__init__.py": '"""External clients, protocols, and adapter construction."""\n',
    "integrations/mcp.py": '"""MCP client, transport, and hosted-tool integration seam."""\n',
    "observability/__init__.py": '"""Application event sinks, logging, tracing, and metrics adapters."""\n',
    "observability/events.py": '"""Runtime event-sink construction and redaction boundaries."""\n',
    "observability/tracing.py": '"""Tracing adapter construction and export policy."""\n',
}

_APPLICATION_GUIDANCE = """

## Cayu application convention

`[tool.cayu.scaffold]` records the exact generated plan. It is a source
generation and checking contract, not runtime authority. Implement a concern
in its owning package first, then wire it through `agents/registration.py` or
the explicit application composition seam. Keep `app.py` composition-only.

| Requested concern | Canonical home |
| --- | --- |
| Agent identity, model defaults, thinking, and metadata | `agents/` |
| System/developer prompt material | `prompts/` |
| Native model-callable capability and `ToolEffect` | `tools/` |
| Exposure, grants, authorization, approval, context, execution, egress, budgets, retries | `policies/` |
| Workspace, runner, artifacts, vaults, MCP servers, knowledge bindings, lifecycle | `environments/` |
| Deterministic multi-step orchestration | `workflows/` |
| Tasks, workers, watchers, interruptions, completion, and recovery | `operations/` |
| Reviewed knowledge, retrieval, curation, embeddings, and maintenance | `knowledge/` |
| Model-facing context, recall, compaction, and memory attribution | `memory/` |
| Business rules and data transformations | `domain/` |
| External clients, protocols, and MCP adapters | `integrations/` |
| Behavioral acceptance evidence | `evals/` |
| Runtime correctness and architectural contracts | `tests/` |
| Event sinks, logging, tracing, and metrics adapters | `observability/` |
| Final explicit construction and registration only | `app.py` |

Runtime sessions, transcripts, events, checkpoints, tasks, leases, approvals,
receipts, knowledge entries, indexes, usage, artifacts, eval results, and
snapshots belong in configured stores, never in source packages.

Use `understand -> inspect -> plan -> change -> test -> eval -> exercise ->
report evidence`. Reproduce this exact selected architecture safely in a
disposable reference, reviewing the write-free preview before applying it:

```bash
reference_parent="$(mktemp -d)"
__DRY_RUN_COMMAND__
__APPLY_COMMAND__
```

Run the command through `uv --no-sync` so the disposable reference uses this
project's installed, exactly pinned Cayu version even though its target directory
is elsewhere. The project-local uv cache under ignored `.cayu/` keeps these
commands usable in workspace-restricted coding-agent sandboxes. `cayu new`
creates projects; it does not migrate an existing repository. For a reviewed
plan change, adjust the reference flags, compare only the owning files, then
update this project and `[tool.cayu.scaffold]` explicitly.

Do not grant authority through prompts, add filesystem auto-discovery, rewrite
arbitrary Python, start lifecycle work during import, or delete the scaffold
contract to silence diagnostics. A custom layout is an explicit migration.
"""

_POSTGRES_GUIDANCE = """

### Postgres database profile

`CAYU_DATABASE_URL` is the only DSN source; never commit it. The generated
session, task, and every other active database-backed store use Postgres
coherently and construct lazily. Live runs require a real migrated database.
Credential-free inspect/tests use injected stores or inert construction. The
printed check command supplies a non-secret unreachable placeholder DSN only so
read-only construction can prove the manifest; it does not prove connectivity
or schema readiness. Exercise production against a disposable real Postgres
database before deployment.
"""


def convention_files(
    plan: ApplicationPlan,
    *,
    render: Callable[[str], str],
) -> dict[str, str]:
    """Return the complete architecture overlay for one normalized plan."""

    if plan.minimal:
        return {
            "CLAUDE.md": _CLAUDE_MD,
        }
    settings = render(_SETTINGS_PY).replace("__DATABASE__", plan.database)
    public_app_factories = (
        '["build_app", "build_coding_product_application"]'
        if plan.preset == "coding" and plan.execution == "docker"
        else '["build_app"]'
    )
    files = dict(_OWNERSHIP_FILES)
    files.update(
        {
            "app.py": render(_APP_PY),
            "configuration/__init__.py": _CONFIGURATION_INIT_PY,
            "configuration/settings.py": settings,
            "configuration/providers.py": _PROVIDERS_PY,
            "configuration/storage.py": (
                _POSTGRES_STORAGE_PY if plan.database == "postgres" else _SQLITE_STORAGE_PY
            ),
            "configuration/runtime.py": _RUNTIME_PY.replace(
                "__ENABLE_LOGGING__",
                "True" if "observability" in plan.capabilities else "False",
            ),
            "agents/agent.py": render(_AGENT_PY),
            "agents/registration.py": _AGENT_REGISTRATION_PY,
            "prompts/agent.py": _PROMPT_PY,
            "tools/registration.py": _TOOLS_REGISTRATION_PY,
            "policies/tools.py": _TOOL_POLICY_PY,
            "tests/test_application.py": _TEST_APPLICATION_PY,
            "tests/test_architecture.py": _TEST_ARCHITECTURE_PY.replace(
                "__PUBLIC_APP_FACTORIES__",
                public_app_factories,
            ),
            "CLAUDE.md": _CLAUDE_MD,
        }
    )
    return files


def scaffold_contract(plan: ApplicationPlan) -> str:
    """Render the normalized source-controlled scaffold contract."""

    capabilities = ", ".join(f'"{name}"' for name in plan.capabilities)
    minimal = "true" if plan.minimal else "false"
    return (
        "\n[tool.cayu.scaffold]\n"
        f"convention = {plan.convention}\n"
        f'preset = "{plan.preset}"\n'
        f'database = "{plan.database}"\n'
        f'provider = "{plan.provider}"\n'
        f'execution = "{plan.execution}"\n'
        + (
            ""
            if plan.coding_toolchain is None
            else f'coding_toolchain = "{plan.coding_toolchain}"\n'
        )
        + (
            ""
            if plan.coding_command_authority is None
            else (f'coding_command_authority = "{plan.coding_command_authority}"\n')
        )
        + f"capabilities = [{capabilities}]\n"
        + f"minimal = {minimal}\n"
    )


def application_guidance(plan: ApplicationPlan) -> str:
    """Return shared concise ownership and working-contract guidance."""

    reference = _creation_command(plan, name=f"{plan.name}_reference")
    reference = f'{reference} --agent-name "{plan.agent_name}" --dir "$reference_parent"'
    guidance = _APPLICATION_GUIDANCE.replace(
        "__DRY_RUN_COMMAND__",
        f"{reference} --dry-run",
    ).replace(
        "__APPLY_COMMAND__",
        reference,
    )
    return guidance + (_POSTGRES_GUIDANCE if plan.database == "postgres" else "")


def _creation_command(plan: ApplicationPlan, *, name: str) -> str:
    arguments = [
        "uv run --no-sync cayu new",
        name,
        "--preset",
        plan.preset,
        "--database",
        plan.database,
        "--provider",
        plan.provider,
        "--execution",
        plan.execution,
    ]
    if plan.coding_toolchain is not None:
        arguments.extend(("--coding-toolchain", plan.coding_toolchain))
    if plan.coding_command_authority is not None:
        arguments.extend(("--coding-command-authority", plan.coding_command_authority))
    defaults = set(preset_spec(plan.preset).default_capabilities)
    selected = set(plan.capabilities)
    for capability in CAPABILITIES:
        if capability.status != "selectable":
            continue
        if capability.name in selected - defaults:
            arguments.extend(("--with", capability.name))
        elif capability.name in defaults - selected:
            arguments.extend(("--without", capability.name))
    if plan.minimal:
        arguments.append("--minimal")
    arguments.append("--json")
    return " ".join(arguments)
