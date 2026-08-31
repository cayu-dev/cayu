"""Declarative catalog and normalized plans for ``cayu new``.

The catalog is the single source of truth for human CLI discovery, structured
JSON discovery, validation, and generated scaffold metadata.  It deliberately
describes source generation only; a scaffold plan is never runtime authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

SCAFFOLD_CONVENTION_VERSION = 1

PresetName = Literal["agent", "service", "coding"]
DatabaseName = Literal["sqlite", "postgres"]
ProviderName = Literal[
    "neutral",
    "openai",
    "anthropic",
    "openrouter",
    "openai-subscription",
]
ExecutionName = Literal["none", "docker"]
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
    supported_databases: tuple[DatabaseName, ...] = ("sqlite", "postgres")
    supported_executions: tuple[ExecutionName, ...] = ("none",)
    environment: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "summary": self.summary,
            "default_capabilities": list(self.default_capabilities),
            "supported_databases": list(self.supported_databases),
            "supported_executions": list(self.supported_executions),
            "environment": list(self.environment),
        }


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """One maintained implementation choice for an orthogonal adapter axis."""

    name: str
    kind: Literal["database", "provider", "execution"]
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
    implied: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    supported_databases: tuple[DatabaseName, ...] = ("sqlite", "postgres")
    supported_executions: tuple[ExecutionName, ...] = ("none", "docker")
    files: tuple[str, ...] = ()
    verification: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "summary": self.summary,
            "status": self.status,
            "supported_presets": list(self.supported_presets),
            "implied": list(self.implied),
            "requires": list(self.implied),
            "conflicts": list(self.conflicts),
            "dependencies": list(self.dependencies),
            "environment": list(self.environment),
            "supported_databases": list(self.supported_databases),
            "supported_executions": list(self.supported_executions),
            "files": list(self.files),
            "verification": list(self.verification),
        }


PRESETS: tuple[PresetSpec, ...] = (
    PresetSpec(
        name="agent",
        summary="Complete Cayu application convention with one model-only agent.",
    ),
    PresetSpec(
        name="service",
        summary="Maintained authenticated multi-user product-service shape.",
        default_capabilities=("tasks", "approvals", "observability"),
        supported_databases=("sqlite",),
        environment=("PRODUCT_AUTH_TOKENS_JSON", "CAYU_OPERATOR_BEARER_TOKEN"),
    ),
    PresetSpec(
        name="coding",
        summary="Maintained trusted-repository coding composition.",
        default_capabilities=(
            "knowledge",
            "tasks",
            "delegation",
            "human-input",
        ),
        supported_executions=("none", "docker"),
    ),
)

ADAPTERS: tuple[AdapterSpec, ...] = (
    AdapterSpec(
        name="sqlite",
        kind="database",
        summary="Local durable SQLite stores under data/cayu.db.",
        supported_presets=("agent", "service", "coding"),
    ),
    AdapterSpec(
        name="postgres",
        kind="database",
        summary="Durable Postgres stores selected by CAYU_DATABASE_URL.",
        supported_presets=("agent", "coding"),
        dependencies=("cayu[postgres]",),
        environment=("CAYU_DATABASE_URL",),
    ),
    AdapterSpec(
        name="neutral",
        kind="provider",
        summary="Credential-free provider-neutral construction.",
        supported_presets=("agent", "service", "coding"),
    ),
    AdapterSpec(
        name="openai",
        kind="provider",
        summary="OpenAI Platform API provider.",
        supported_presets=("agent", "service", "coding"),
        environment=("OPENAI_API_KEY",),
    ),
    AdapterSpec(
        name="anthropic",
        kind="provider",
        summary="Anthropic API provider.",
        supported_presets=("agent", "service", "coding"),
        environment=("ANTHROPIC_API_KEY",),
    ),
    AdapterSpec(
        name="openrouter",
        kind="provider",
        summary="OpenRouter compatible chat-completions provider.",
        supported_presets=("agent", "service", "coding"),
        environment=("OPENROUTER_API_KEY", "CAYU_MODEL"),
    ),
    AdapterSpec(
        name="openai-subscription",
        kind="provider",
        summary="Local OpenAI subscription development provider.",
        supported_presets=("agent", "service", "coding"),
    ),
    AdapterSpec(
        name="none",
        kind="execution",
        summary="No generated command-execution backend.",
        supported_presets=("agent", "service", "coding"),
    ),
    AdapterSpec(
        name="docker",
        kind="execution",
        summary="Admitted no-network Docker execution for trusted repositories.",
        supported_presets=("coding",),
        dependencies=("docker",),
    ),
)

CAPABILITIES: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        name="knowledge",
        summary="Reviewed durable knowledge, retrieval, curation, and maintenance.",
        status="preset-owned",
        supported_presets=("coding",),
        files=("knowledge/", "configuration/storage.py"),
        verification=("uv run --no-sync cayu check --json",),
    ),
    CapabilitySpec(
        name="memory",
        summary="Context selection, recall, compaction, and memory attribution.",
        status="extension-only",
        supported_presets=("agent", "service", "coding"),
        files=("memory/", "policies/context.py"),
    ),
    CapabilitySpec(
        name="mcp",
        summary="External MCP clients, protocols, and hosted-tool adapters.",
        status="extension-only",
        supported_presets=("agent", "service", "coding"),
        files=("integrations/mcp.py",),
    ),
    CapabilitySpec(
        name="tasks",
        summary="Durable tasks and application-owned operation lifecycle wiring.",
        status="preset-owned",
        supported_presets=("service", "coding"),
        files=("operations/tasks.py",),
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
        files=("operations/completion.py", "agents/registration.py"),
    ),
    CapabilitySpec(
        name="human-input",
        summary="Durable pause and application-owned human input resolution.",
        status="preset-owned",
        supported_presets=("coding",),
        files=("operations/approvals.py",),
    ),
    CapabilitySpec(
        name="approvals",
        summary="Explicit approval policy and settlement boundaries.",
        status="preset-owned",
        supported_presets=("service",),
        files=("operations/approvals.py", "policies/tools.py"),
    ),
    CapabilitySpec(
        name="observability",
        summary="Event sinks, logging, tracing, and metrics adapters.",
        status="selectable",
        supported_presets=("agent", "service"),
        files=("configuration/runtime.py", "observability/"),
        verification=("uv run --no-sync cayu inspect --json", "uv run --no-sync pytest"),
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
    database: DatabaseName
    provider: ProviderName
    execution: ExecutionName
    capabilities: tuple[str, ...]
    minimal: bool = False
    convention: int = SCAFFOLD_CONVENTION_VERSION

    @property
    def template_alias(self) -> str:
        return "service" if self.preset == "service" else "agent"

    @property
    def composition_alias(self) -> str | None:
        return "coding" if self.preset == "coding" else None

    @property
    def provider_alias(self) -> str | None:
        return None if self.provider == "neutral" else self.provider

    @property
    def execution_alias(self) -> str | None:
        return None if self.execution == "none" else self.execution

    def verification_commands(self) -> tuple[str, ...]:
        check = (
            "uv run --no-sync cayu check --deploy --fail-on warning --json"
            if self.preset == "service"
            else "uv run --no-sync cayu check --fail-on warning --json"
        )
        check_environment: list[str] = []
        if self.database == "postgres":
            check_environment.append(
                "CAYU_DATABASE_URL=postgresql://cayu-unconfigured@127.0.0.1/cayu"
            )
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
        }[self.preset]
        setup = (
            (
                "uv lock",
                "uv sync --extra dev",
                "uv run --no-sync python build_coding_image.py",
            )
            if self.execution == "docker"
            else ("uv sync --extra dev",)
        )
        return (
            *setup,
            "uv run --no-sync cayu inspect --json",
            check,
            focused_test,
            "uv run --no-sync cayu eval run",
        )

    def as_dict(
        self,
        *,
        files: tuple[str, ...] = (),
        directories: tuple[str, ...] = (),
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "convention": self.convention,
            "name": self.name,
            "agent_name": self.agent_name,
            "preset": self.preset,
            "adapters": {
                "database": self.database,
                "provider": self.provider,
                "execution": self.execution,
            },
            "capabilities": list(self.capabilities),
            "minimal": self.minimal,
            "files": list(files),
            "directories": list(directories),
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
    database: str = "sqlite",
    provider: str = "neutral",
    execution: str = "none",
    with_capabilities: tuple[str, ...] = (),
    without_capabilities: tuple[str, ...] = (),
    minimal: bool = False,
) -> ApplicationPlan:
    """Resolve and validate every generator choice before rendering or writing."""

    selected_preset = preset_spec(preset)
    database_spec = _adapter("database", database)
    provider_spec = _adapter("provider", provider)
    execution_spec = _adapter("execution", execution)
    for adapter in (database_spec, provider_spec, execution_spec):
        if selected_preset.name not in adapter.supported_presets:
            raise ScaffoldPlanError(
                "unsupported_adapter",
                f"{adapter.kind} {adapter.name!r} is not supported by preset {preset!r}",
            )
    if minimal and preset != "agent":
        raise ScaffoldPlanError(
            "minimal_requires_agent",
            "--minimal is supported only by the agent preset",
        )
    if minimal and database != "sqlite":
        raise ScaffoldPlanError(
            "minimal_database_unsupported",
            "--minimal supports only --database sqlite; use the complete convention "
            "for maintained Postgres composition",
        )

    requested = _expand_capability_arguments(with_capabilities)
    excluded = _expand_capability_arguments(without_capabilities)
    if minimal and requested:
        raise ScaffoldPlanError(
            "minimal_capability_unsupported",
            "--minimal cannot activate capability slices; use the complete convention",
        )
    overlap = sorted(set(requested) & set(excluded))
    if overlap:
        raise ScaffoldPlanError(
            "capability_conflict",
            "capabilities cannot be both included and excluded: " + ", ".join(overlap),
        )

    capabilities = set(() if minimal else selected_preset.default_capabilities)
    for name_value in requested:
        spec = capability_spec(name_value)
        if spec.status != "selectable":
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
        if database_spec.name not in spec.supported_databases:
            raise ScaffoldPlanError(
                "unsupported_capability_database",
                f"capability {name_value!r} does not support database {database!r}",
            )
        if execution_spec.name not in spec.supported_executions:
            raise ScaffoldPlanError(
                "unsupported_capability_execution",
                f"capability {name_value!r} does not support execution {execution!r}",
            )
        capabilities.add(name_value)
        capabilities.update(spec.implied)

    for name_value in excluded:
        spec = capability_spec(name_value)
        if name_value in selected_preset.default_capabilities and spec.status != "selectable":
            raise ScaffoldPlanError(
                "preset_capability_required",
                f"preset {preset!r} requires capability {name_value!r}; choose another preset",
            )
        capabilities.discard(name_value)
        for selected in capabilities:
            if name_value in capability_spec(selected).implied:
                raise ScaffoldPlanError(
                    "capability_dependency_conflict",
                    f"capability {selected!r} requires excluded capability {name_value!r}",
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
        database=cast("DatabaseName", database_spec.name),
        provider=cast("ProviderName", provider_spec.name),
        execution=cast("ExecutionName", execution_spec.name),
        capabilities=tuple(sorted(capabilities)),
        minimal=minimal,
    )


def _adapter(kind: Literal["database", "provider", "execution"], name: str) -> AdapterSpec:
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
    environment = set(_PRESET_BY_NAME[plan.preset].environment)
    for kind, name in (
        ("database", plan.database),
        ("provider", plan.provider),
        ("execution", plan.execution),
    ):
        environment.update(_ADAPTER_BY_KEY[(kind, name)].environment)
    for name in plan.capabilities:
        environment.update(capability_spec(name).environment)
    return tuple(sorted(environment))


def _plan_dependencies(plan: ApplicationPlan) -> tuple[str, ...]:
    dependencies = {"cayu"}
    for kind, name in (
        ("database", plan.database),
        ("provider", plan.provider),
        ("execution", plan.execution),
    ):
        dependencies.update(_ADAPTER_BY_KEY[(kind, name)].dependencies)
    for name in plan.capabilities:
        dependencies.update(capability_spec(name).dependencies)
    return tuple(sorted(dependencies))
