"""Declarative catalog and normalized plans for ``cayu new``.

The catalog is the single source of truth for human CLI discovery, structured
JSON discovery, validation, and generated scaffold metadata.  It deliberately
describes source generation only; a scaffold plan is never runtime authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from cayu._operator_credentials import OPERATOR_PASSWORD_VARIABLE, OPERATOR_USERNAME_VARIABLE

SCAFFOLD_CONVENTION_VERSION = 1

PresetName = Literal["agent", "service", "coding", "long-running"]
ProviderName = Literal[
    "neutral",
    "openai",
    "anthropic",
    "openrouter",
    "cayu-gateway",
    "openai-subscription",
]
ExecutionName = Literal["none", "docker"]
CodingToolchainName = Literal["python"]
CodingCommandAuthorityName = Literal["structured"]
CapabilityStatus = Literal["selectable", "extension-only", "preset-owned"]


class ScaffoldPlanError(ValueError):
    """Stable, non-secret plan validation failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class PresetSpec:
    """One maintained coherent application shape."""

    name: PresetName
    summary: str
    default_capabilities: tuple[str, ...] = ()
    supported_executions: tuple[ExecutionName, ...] = ("none",)
    environment: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "summary": self.summary,
            "default_capabilities": list(self.default_capabilities),
            "supported_executions": list(self.supported_executions),
            "environment": list(self.environment),
        }


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """One maintained implementation choice for an orthogonal adapter axis."""

    name: str
    kind: Literal["provider", "execution"]
    summary: str
    supported_presets: tuple[PresetName, ...]
    dependencies: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "summary": self.summary,
            "supported_presets": list(self.supported_presets),
            "dependencies": list(self.dependencies),
            "environment": list(self.environment),
        }


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """One Cayu concern exposed by the application convention."""

    name: str
    summary: str
    status: CapabilityStatus
    supported_presets: tuple[PresetName, ...]
    extension_presets: tuple[PresetName, ...] = ()
    implied: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    supported_executions: tuple[ExecutionName, ...] = ("none", "docker")
    files: tuple[str, ...] = ()
    verification: tuple[str, ...] = ()
    # Generated files that exist only for this capability. `--without` removes
    # them when the preset turns the capability on by default. Shared seams that
    # every project imports, such as configuration/storage.py, stay in `files`.
    owned_files: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "summary": self.summary,
            "status": self.status,
            "supported_presets": list(self.supported_presets),
            "extension_presets": list(self.extension_presets),
            "extension_declaration": "tool.cayu.scaffold.extensions",
            "extension_docs": "cayu guide applications#explicit-service-extensions",
            "implied": list(self.implied),
            "requires": list(self.implied),
            "conflicts": list(self.conflicts),
            "dependencies": list(self.dependencies),
            "environment": list(self.environment),
            "supported_executions": list(self.supported_executions),
            "files": list(self.files),
            "owned_files": list(self.owned_files),
            "verification": list(self.verification),
        }


PRESETS: tuple[PresetSpec, ...] = (
    PresetSpec(
        name="agent",
        summary="Batteries-included local Cayu agent with explicit safe capability wiring.",
        default_capabilities=(
            "approvals",
            "artifacts",
            "evals",
            "human-input",
            "knowledge",
            "memory",
            "observability",
            "recovery",
            "tasks",
        ),
        supported_executions=("none", "docker"),
        environment=(OPERATOR_PASSWORD_VARIABLE, OPERATOR_USERNAME_VARIABLE),
    ),
    PresetSpec(
        name="service",
        summary="Maintained authenticated multi-user product-service shape.",
        default_capabilities=("tasks", "approvals", "observability", "evals"),
        environment=("PRODUCT_AUTH_TOKENS_JSON", "CAYU_OPERATOR_BEARER_TOKEN"),
    ),
    PresetSpec(
        name="coding",
        summary="Maintained trusted-repository coding composition.",
        default_capabilities=(
            "artifacts",
            "knowledge",
            "tasks",
            "delegation",
            "human-input",
            "evals",
        ),
        supported_executions=("none", "docker"),
        environment=(OPERATOR_PASSWORD_VARIABLE, OPERATOR_USERNAME_VARIABLE),
    ),
    PresetSpec(
        name="long-running",
        summary=(
            "Agent example with approval, a waited-for reply, reminder and deadline "
            "timers, a spend cap, and restart tests."
        ),
        default_capabilities=(
            "approvals",
            "evals",
            "human-input",
            "observability",
            "recovery",
            "tasks",
        ),
        environment=(OPERATOR_PASSWORD_VARIABLE, OPERATOR_USERNAME_VARIABLE),
    ),
)

ADAPTERS: tuple[AdapterSpec, ...] = (
    AdapterSpec(
        name="neutral",
        kind="provider",
        summary="Credential-free provider-neutral construction.",
        supported_presets=("agent", "service", "coding", "long-running"),
    ),
    AdapterSpec(
        name="openai",
        kind="provider",
        summary="OpenAI Platform API provider.",
        supported_presets=("agent", "service", "coding", "long-running"),
        environment=("OPENAI_API_KEY",),
    ),
    AdapterSpec(
        name="anthropic",
        kind="provider",
        summary="Anthropic API provider.",
        supported_presets=("agent", "service", "coding", "long-running"),
        environment=("ANTHROPIC_API_KEY",),
    ),
    AdapterSpec(
        name="cayu-gateway",
        kind="provider",
        summary="Cayu Gateway Chat Completions provider.",
        supported_presets=("agent", "service", "coding", "long-running"),
        environment=("CAYU_GATEWAY_API_KEY", "CAYU_GATEWAY_BASE_URL", "CAYU_MODEL"),
    ),
    AdapterSpec(
        name="openrouter",
        kind="provider",
        summary="OpenRouter compatible chat-completions provider.",
        supported_presets=("agent", "service", "coding", "long-running"),
        environment=("OPENROUTER_API_KEY", "CAYU_MODEL"),
    ),
    AdapterSpec(
        name="openai-subscription",
        kind="provider",
        summary="Local OpenAI subscription development provider.",
        supported_presets=("agent", "service", "coding", "long-running"),
    ),
    AdapterSpec(
        name="none",
        kind="execution",
        summary="No generated command-execution backend.",
        supported_presets=("agent", "service", "coding", "long-running"),
    ),
    AdapterSpec(
        name="docker",
        kind="execution",
        summary=(
            "No-network Docker execution: admitted repository commands for the coding "
            "preset, a hardened sandbox for the agent preset's own tools."
        ),
        supported_presets=("agent", "coding"),
        dependencies=("docker",),
    ),
)

CAPABILITIES: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        name="knowledge",
        summary="Reviewed durable knowledge, retrieval, curation, and maintenance.",
        status="preset-owned",
        supported_presets=("agent", "coding"),
        extension_presets=("service",),
        files=("knowledge/", "configuration/storage.py"),
        verification=("uv run --no-sync cayu check --json",),
        owned_files=(
            "knowledge/__init__.py",
            "knowledge/retrieval.py",
            "knowledge/curation.py",
            "knowledge/maintenance.py",
            "knowledge/seeds/.gitkeep",
            "knowledge/CAPABILITY.md",
        ),
    ),
    CapabilitySpec(
        name="memory",
        summary="Context selection, recall, compaction, and memory attribution.",
        status="preset-owned",
        supported_presets=("agent", "service", "coding"),
        implied=("knowledge",),
        files=("memory/", "policies/context.py"),
        verification=("uv run --no-sync pytest -q tests/test_memory.py",),
        owned_files=(
            "memory/__init__.py",
            "memory/context.py",
            "memory/recall.py",
            "memory/CAPABILITY.md",
            "tests/test_memory.py",
        ),
    ),
    CapabilitySpec(
        name="mcp",
        summary="External MCP clients, protocols, and hosted-tool adapters.",
        status="extension-only",
        supported_presets=("agent", "service", "coding", "long-running"),
        files=("integrations/mcp.py",),
    ),
    CapabilitySpec(
        name="tasks",
        summary="Durable tasks and application-owned operation lifecycle wiring.",
        status="preset-owned",
        supported_presets=("agent", "service", "coding", "long-running"),
        files=("operations/tasks.py",),
        owned_files=("operations/tasks.py", "operations/TASKS.md"),
    ),
    CapabilitySpec(
        name="workers",
        summary="Explicit background worker construction and startup commands.",
        status="extension-only",
        supported_presets=("service", "coding"),
        implied=("tasks",),
        files=("operations/workers.py",),
    ),
    CapabilitySpec(
        name="delegation",
        summary="Child-agent registration, delegation, and result recovery.",
        status="preset-owned",
        supported_presets=("coding",),
        extension_presets=("service",),
        files=("operations/completion.py", "agents/registration.py"),
        owned_files=("operations/DELEGATION.md",),
    ),
    CapabilitySpec(
        name="human-input",
        summary="Durable pause and application-owned human input resolution.",
        status="preset-owned",
        supported_presets=("agent", "coding", "long-running"),
        files=("operations/approvals.py",),
        owned_files=("operations/approvals.py", "operations/HUMAN_INPUT.md"),
    ),
    CapabilitySpec(
        name="approvals",
        summary="Explicit approval policy and settlement boundaries.",
        status="preset-owned",
        supported_presets=("agent", "service", "coding", "long-running"),
        files=("operations/approvals.py", "policies/tools.py"),
        owned_files=("operations/approvals.py", "operations/APPROVALS.md"),
    ),
    CapabilitySpec(
        name="artifacts",
        summary="Durable local artifacts with bounded model-facing discovery.",
        status="preset-owned",
        supported_presets=("agent", "coding"),
        extension_presets=("service",),
        files=("environments/local.py", "data/artifacts/"),
        verification=("uv run --no-sync pytest -q tests/test_application.py",),
        owned_files=("environments/ARTIFACTS.md",),
    ),
    CapabilitySpec(
        name="recovery",
        summary="Durable interruption inspection and explicit recovery entry points.",
        status="preset-owned",
        supported_presets=("agent", "service", "coding", "long-running"),
        files=("operations/recovery.py", "operations/workers.py"),
        verification=("uv run --no-sync cayu inspect --json",),
        owned_files=("operations/recovery.py", "operations/RECOVERY.md"),
    ),
    CapabilitySpec(
        name="evals",
        summary="Credential-free behavioral acceptance and durable eval evidence.",
        status="preset-owned",
        supported_presets=("agent", "service", "coding", "long-running"),
        files=("evals/",),
        verification=("uv run --no-sync cayu eval run",),
        owned_files=("evals/__init__.py", "evals/agent.py", "evals/CAPABILITY.md"),
    ),
    CapabilitySpec(
        name="observability",
        summary="Event sinks, logging, tracing, and metrics adapters.",
        status="selectable",
        supported_presets=("agent", "service", "long-running"),
        files=("configuration/runtime.py", "observability/"),
        verification=("uv run --no-sync cayu inspect --json", "uv run --no-sync pytest"),
        owned_files=(
            "observability/__init__.py",
            "observability/events.py",
            "observability/tracing.py",
            "observability/CAPABILITY.md",
        ),
    ),
    CapabilitySpec(
        name="app-database",
        summary=(
            "Application-owned tables in the `app` PostgreSQL schema beside Cayu's, "
            "with SQLAlchemy models and Alembic migrations."
        ),
        status="selectable",
        supported_presets=("agent", "service", "coding", "long-running"),
        dependencies=("alembic>=1.13,<2", "sqlalchemy>=2,<3"),
        environment=("CAYU_DATABASE_URL", "CAYU_DATABASE_DIRECT_URL"),
        files=("database/", "migrations/", "alembic.ini"),
        verification=("uv run --no-sync pytest -q tests/test_app_database.py",),
        owned_files=(
            "alembic.ini",
            "database/CAPABILITY.md",
            "database/__init__.py",
            "database/models.py",
            "database/session.py",
            "migrations/env.py",
            "migrations/script.py.mako",
            "migrations/versions/0001_app_tables.py",
            "tests/test_app_database.py",
        ),
    ),
    CapabilitySpec(
        name="remote-git-delivery",
        summary="Approved host-side commit and push of an exact patch-ready result.",
        status="selectable",
        supported_presets=("coding",),
        supported_executions=("docker",),
        files=("integrations/remote_git.py",),
        verification=("uv run --no-sync pytest -q tests/test_coding_composition.py",),
    ),
    CapabilitySpec(
        name="github-delivery",
        summary="Approved exact-commit pull request, checks, reviews, and follow-up evidence.",
        status="selectable",
        supported_presets=("coding",),
        supported_executions=("docker",),
        implied=("remote-git-delivery",),
        files=("integrations/github.py", "integrations/remote_git.py", "README.md", "AGENTS.md"),
        verification=("uv run --no-sync pytest -q tests/test_coding_composition.py",),
    ),
)

_PRESET_BY_NAME = {spec.name: spec for spec in PRESETS}
_CAPABILITY_BY_NAME = {spec.name: spec for spec in CAPABILITIES}
_ADAPTER_BY_KEY = {(spec.kind, spec.name): spec for spec in ADAPTERS}


@dataclass(frozen=True, slots=True)
class ApplicationPlan:
    """Fully normalized, deterministic source-generation intent."""

    name: str
    agent_name: str
    preset: PresetName
    provider: ProviderName
    execution: ExecutionName
    capabilities: tuple[str, ...]
    coding_toolchain: CodingToolchainName | None = None
    coding_command_authority: CodingCommandAuthorityName | None = None
    convention: int = SCAFFOLD_CONVENTION_VERSION

    @property
    def provider_alias(self) -> str | None:
        return None if self.provider == "neutral" else self.provider

    def verification_commands(self) -> tuple[str, ...]:
        check = (
            "uv run --no-sync cayu check --deploy --fail-on warning --json"
            if self.preset == "service"
            else "uv run --no-sync cayu check --fail-on warning --json"
        )
        check_environment: list[str] = []
        if self.preset == "service":
            check_environment.extend(
                (
                    "PRODUCT_AUTH_TOKENS_JSON="
                    '\'{"local-customer-token":{"tenant_id":"local-tenant",'
                    '"subject_id":"local-user"}}\'',
                    "CAYU_OPERATOR_BEARER_TOKEN=local-operator-token",
                )
            )
        if check_environment:
            check = " ".join((*check_environment, check))
        focused_test = {
            "agent": "uv run --no-sync pytest",
            "service": "uv run --no-sync pytest -q tests/test_public_service_security.py",
            "coding": "uv run --no-sync pytest -q tests/test_coding_composition.py",
            "long-running": "uv run --no-sync pytest",
        }[self.preset]
        if self.execution == "docker" and self.preset == "agent":
            setup: tuple[str, ...] = (
                "uv sync --extra dev",
                "uv run --no-sync python sandbox_image.py --resolve",
            )
        elif self.execution == "docker":
            setup = (
                "uv lock",
                "uv sync --extra dev",
                "uv run --no-sync python build_coding_image.py",
            )
        else:
            setup = ("uv sync --extra dev",)
        commands = (
            *setup,
            "uv run --no-sync cayu inspect --json",
            check,
            focused_test,
        )
        if "evals" in self.capabilities:
            commands = (*commands, "uv run --no-sync cayu eval run")
        return commands

    def as_dict(
        self,
        *,
        files: tuple[str, ...] = (),
        directories: tuple[str, ...] = (),
        private_files: tuple[str, ...] = (),
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "convention": self.convention,
            "name": self.name,
            "agent_name": self.agent_name,
            "preset": self.preset,
            "adapters": {
                "provider": self.provider,
                "execution": self.execution,
            },
            "coding": {
                "toolchain": self.coding_toolchain,
                "command_authority": self.coding_command_authority,
            },
            "capabilities": list(self.capabilities),
            "extensions": [],
            "files": list(files),
            "directories": list(directories),
            "private_files": list(private_files),
            "environment": list(_plan_environment(self)),
            "dependencies": list(_plan_dependencies(self)),
            "verification_commands": list(self.verification_commands()),
        }


def catalog_dict() -> dict[str, object]:
    """Return the stable package-shipped discovery projection."""

    return {
        "schema_version": 1,
        "convention": SCAFFOLD_CONVENTION_VERSION,
        "presets": [spec.as_dict() for spec in PRESETS],
        "adapters": [spec.as_dict() for spec in ADAPTERS],
        "capabilities": [spec.as_dict() for spec in CAPABILITIES],
    }


def preset_spec(name: str) -> PresetSpec:
    try:
        return _PRESET_BY_NAME[cast("PresetName", name)]
    except KeyError:
        choices = ", ".join(_PRESET_BY_NAME)
        raise ScaffoldPlanError("unknown_preset", f"preset must be one of: {choices}") from None


def capability_spec(name: str) -> CapabilitySpec:
    try:
        return _CAPABILITY_BY_NAME[name]
    except KeyError:
        choices = ", ".join(_CAPABILITY_BY_NAME)
        raise ScaffoldPlanError(
            "unknown_capability",
            f"unknown capability {name!r}; choose one of: {choices}",
        ) from None


def normalize_application_plan(
    *,
    name: str,
    agent_name: str,
    preset: str = "agent",
    provider: str = "neutral",
    execution: str = "none",
    coding_toolchain: str | None = None,
    coding_command_authority: str | None = None,
    with_capabilities: tuple[str, ...] = (),
    without_capabilities: tuple[str, ...] = (),
) -> ApplicationPlan:
    """Resolve and validate every generator choice before rendering or writing."""

    selected_preset = preset_spec(preset)
    provider_spec = _adapter("provider", provider)
    execution_spec = _adapter("execution", execution)
    for adapter in (provider_spec, execution_spec):
        if selected_preset.name not in adapter.supported_presets:
            raise ScaffoldPlanError(
                "unsupported_adapter",
                f"{adapter.kind} {adapter.name!r} is not supported by preset {preset!r}",
            )
    if preset != "coding" and (
        coding_toolchain is not None or coding_command_authority is not None
    ):
        raise ScaffoldPlanError(
            "coding_options_require_coding",
            "coding toolchain and command authority require the coding preset",
        )
    if execution != "docker" and (
        coding_toolchain is not None or coding_command_authority is not None
    ):
        raise ScaffoldPlanError(
            "coding_options_require_docker",
            "coding toolchain and command authority require Docker execution",
        )
    if coding_toolchain not in {None, "python"}:
        raise ScaffoldPlanError(
            "unknown_coding_toolchain",
            "coding toolchain must be 'python'",
        )
    if coding_command_authority not in {None, "structured"}:
        raise ScaffoldPlanError(
            "unknown_coding_command_authority",
            "coding command authority must be 'structured'",
        )
    resolved_coding_toolchain = "python" if preset == "coding" and execution == "docker" else None
    resolved_coding_command_authority = (
        "structured" if preset == "coding" and execution == "docker" else None
    )

    requested = _expand_capability_arguments(with_capabilities)
    excluded = _expand_capability_arguments(without_capabilities)
    overlap = sorted(set(requested) & set(excluded))
    if overlap:
        raise ScaffoldPlanError(
            "capability_conflict",
            "capabilities cannot be both included and excluded: " + ", ".join(overlap),
        )

    capabilities = set(selected_preset.default_capabilities)
    for name_value in requested:
        spec = capability_spec(name_value)
        if name_value not in selected_preset.default_capabilities and spec.status != "selectable":
            if selected_preset.name in spec.extension_presets:
                raise ScaffoldPlanError(
                    "capability_not_selectable",
                    f"capability {name_value!r} requires explicit owning-module wiring for "
                    f"preset {preset!r}; declare it in [tool.cayu.scaffold].extensions "
                    "after implementation, keeping capabilities unchanged. "
                    "See `cayu guide applications#explicit-service-extensions`",
                )
            raise ScaffoldPlanError(
                "capability_not_selectable",
                f"capability {name_value!r} is {spec.status}; use its documented "
                "extension seam or a preset that maintains it",
            )
        if selected_preset.name not in spec.supported_presets:
            raise ScaffoldPlanError(
                "unsupported_capability",
                f"capability {name_value!r} is not supported by preset {preset!r}",
            )
        if execution_spec.name not in spec.supported_executions:
            raise ScaffoldPlanError(
                "unsupported_capability_execution",
                f"capability {name_value!r} does not support execution {execution!r}",
            )
        capabilities.add(name_value)
        capabilities.update(spec.implied)

    for name_value in excluded:
        capability_spec(name_value)
        capabilities.discard(name_value)

    for name_value in excluded:
        for selected in capabilities:
            if name_value in capability_spec(selected).implied:
                raise ScaffoldPlanError(
                    "capability_dependency_conflict",
                    f"capability {selected!r} requires excluded capability {name_value!r}",
                )

    required = {"tasks"} if preset == "service" else set()
    if preset == "long-running":
        # The example's approval, reply and timer patterns are built on these.
        required.update(("approvals", "human-input", "tasks"))
    if preset == "coding" and execution == "docker":
        required.add("artifacts")
    missing = sorted(required - capabilities)
    if missing:
        raise ScaffoldPlanError(
            "capability_required",
            f"preset {preset!r} with execution {execution!r} requires: " + ", ".join(missing),
        )

    for name_value in sorted(capabilities):
        spec = capability_spec(name_value)
        conflicts = sorted(set(spec.conflicts) & capabilities)
        if conflicts:
            raise ScaffoldPlanError(
                "capability_conflict",
                f"capability {name_value!r} conflicts with: " + ", ".join(conflicts),
            )

    return ApplicationPlan(
        name=name,
        agent_name=agent_name,
        preset=selected_preset.name,
        provider=cast("ProviderName", provider_spec.name),
        execution=cast("ExecutionName", execution_spec.name),
        capabilities=tuple(sorted(capabilities)),
        coding_toolchain=resolved_coding_toolchain,
        coding_command_authority=resolved_coding_command_authority,
    )


def _adapter(kind: Literal["provider", "execution"], name: str) -> AdapterSpec:
    try:
        return _ADAPTER_BY_KEY[(kind, name)]
    except KeyError:
        choices = ", ".join(spec.name for spec in ADAPTERS if spec.kind == kind)
        raise ScaffoldPlanError(
            f"unknown_{kind}",
            f"{kind} must be one of: {choices}",
        ) from None


def _expand_capability_arguments(values: tuple[str, ...]) -> tuple[str, ...]:
    expanded: list[str] = []
    for value in values:
        for candidate in value.split(","):
            name = candidate.strip()
            if not name:
                raise ScaffoldPlanError(
                    "invalid_capability",
                    "capability names must be non-empty",
                )
            if name not in expanded:
                expanded.append(name)
    return tuple(expanded)


def _plan_environment(plan: ApplicationPlan) -> tuple[str, ...]:
    # CAYU_DATABASE_URL selects the durable store for every project.
    environment = {"CAYU_DATABASE_URL", *_PRESET_BY_NAME[plan.preset].environment}
    for kind, name in (
        ("provider", plan.provider),
        ("execution", plan.execution),
    ):
        environment.update(_ADAPTER_BY_KEY[(kind, name)].environment)
    for name in plan.capabilities:
        environment.update(capability_spec(name).environment)
    return tuple(sorted(environment))


def _plan_dependencies(plan: ApplicationPlan) -> tuple[str, ...]:
    # Every project can switch to PostgreSQL through CAYU_DATABASE_URL.
    # Every preset is served with `cayu serve`, which needs the server extra.
    dependencies = {"cayu[postgres,server]"}
    for kind, name in (
        ("provider", plan.provider),
        ("execution", plan.execution),
    ):
        dependencies.update(_ADAPTER_BY_KEY[(kind, name)].dependencies)
    dependencies.update(capability_dependencies(plan))
    return tuple(sorted(dependencies))


def removed_capability_files(plan: ApplicationPlan) -> frozenset[str]:
    """Return the owned files a plan's `--without` choices leave out.

    A file is removed when a capability that owns it is a preset default the
    plan turned off and no capability that owns it is still selected. Presets
    keep the homes of capabilities they don't turn on by default, so those can
    be wired in later.
    """

    selected = set(plan.capabilities)
    excluded = set(_PRESET_BY_NAME[plan.preset].default_capabilities) - selected
    removed: set[str] = set()
    for name in excluded:
        for path in _CAPABILITY_BY_NAME[name].owned_files:
            owners = {spec.name for spec in CAPABILITIES if path in spec.owned_files}
            if not owners & selected:
                removed.add(path)
    return frozenset(removed)


def capability_dependents(plan: ApplicationPlan, name: str) -> tuple[str, ...]:
    """Return ``name`` and every selected capability that requires it, in catalog order."""

    selected = set(plan.capabilities)
    dependents = {name}
    changed = True
    while changed:
        changed = False
        for candidate in selected - dependents:
            if dependents & set(_CAPABILITY_BY_NAME[candidate].implied):
                dependents.add(candidate)
                changed = True
    return tuple(spec.name for spec in CAPABILITIES if spec.name in dependents)


def capability_dependencies(plan: ApplicationPlan) -> tuple[str, ...]:
    """The requirements the plan's capabilities add to the generated project."""

    return tuple(
        sorted(
            {
                dependency
                for name in plan.capabilities
                for dependency in capability_spec(name).dependencies
            }
        )
    )


def normalize_extension_declarations(
    plan: ApplicationPlan, extensions: tuple[str, ...]
) -> tuple[str, ...]:
    """Validate application-owned declarations without making them generator selections."""

    for name in extensions:
        spec = capability_spec(name)
        if plan.preset not in spec.extension_presets:
            raise ScaffoldPlanError(
                "unsupported_extension",
                f"capability {name!r} has no declared extension contract for {plan.preset!r}",
            )
        if name in plan.capabilities:
            raise ScaffoldPlanError("extension_overlap", "extensions must not repeat capabilities")
        if plan.execution not in spec.supported_executions:
            raise ScaffoldPlanError(
                "unsupported_extension_adapter", "extension adapters are unsupported"
            )
    normalized = tuple(sorted(set(extensions)))
    if extensions != normalized:
        raise ScaffoldPlanError("extensions_not_normalized", "extensions must be sorted and unique")
    return normalized
