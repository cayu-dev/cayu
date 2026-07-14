from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Mapping
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PlainValidator

from cayu._validation import collision_safe_json_object, copy_json_value
from cayu.runtime.tool_policy import (
    AllowAllToolPolicy,
    AlwaysRequireApprovalToolPolicy,
    ParameterConstrainedToolPolicy,
    StaticToolPolicy,
    TaintAwareToolPolicy,
    ToolPolicy,
)

if TYPE_CHECKING:
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime.app import CayuApp

APP_MANIFEST_SCHEMA_VERSION = "1"
_ABSOLUTE_PATH_PLACEHOLDER = "[ABSOLUTE_PATH]"
_MEMORY_ADDRESS_PLACEHOLDER = "[MEMORY_ADDRESS]"
_OBJECT_REPRESENTATION_PLACEHOLDER = "[OBJECT_REPRESENTATION]"
_TIMESTAMP_PLACEHOLDER = "[TIMESTAMP]"
_OBJECT_REPRESENTATION_PATTERN = re.compile(
    r"<[^<>\r\n]*\bobject at 0x[0-9a-f]+>",
    re.IGNORECASE,
)
_MEMORY_ADDRESS_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])0x[0-9a-f]{6,}(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_TIMESTAMP_PATTERN = re.compile(
    r"(?<!\d)\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})?(?!\d)",
)
_URI_OR_ABSOLUTE_PATH_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+"
    r"|(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|\\\\)[^\s\"'<>]+"
    r"|(?<![A-Za-z0-9_./-])/[^\s\"'<>]+",
    re.IGNORECASE,
)


def _freeze_json_object(value: object) -> Mapping[str, object]:
    copied = copy_json_value(value, "JSON object")
    if type(copied) is not dict:
        raise ValueError("JSON object must be an object.")
    return cast("Mapping[str, object]", _freeze_json_value(copied))


def _freeze_json_value(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({str(key): _freeze_json_value(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_object(value: Mapping[str, object]) -> dict[str, object]:
    return {key: _thaw_json_value(item) for key, item in value.items()}


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _normalize_manifest_unstable_values(value: object) -> object:
    if type(value) is str:
        return _normalize_manifest_string(value)
    if type(value) is list:
        return [_normalize_manifest_unstable_values(item) for item in value]
    if type(value) is dict:
        items: list[tuple[str, object]] = []
        for key, item in value.items():
            if type(key) is not str:
                raise AssertionError("Manifest JSON object keys must be strings.")
            normalized_key = _normalize_manifest_string(key)
            normalized_item = _normalize_manifest_unstable_values(item)
            items.append((normalized_key, normalized_item))
        return collision_safe_json_object(items, preserve_input_order=False)
    return value


def _normalize_manifest_string(value: str) -> str:
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return _ABSOLUTE_PATH_PLACEHOLDER
    normalized = _OBJECT_REPRESENTATION_PATTERN.sub(
        _OBJECT_REPRESENTATION_PLACEHOLDER,
        value,
    )
    normalized = _TIMESTAMP_PATTERN.sub(_TIMESTAMP_PLACEHOLDER, normalized)
    normalized = _MEMORY_ADDRESS_PATTERN.sub(_MEMORY_ADDRESS_PLACEHOLDER, normalized)
    return _URI_OR_ABSOLUTE_PATH_PATTERN.sub(_normalize_manifest_token, normalized)


def _normalize_manifest_token(match: re.Match[str]) -> str:
    token = match.group(0)
    if "://" in token and not token.lower().startswith("file://"):
        return token
    return _ABSOLUTE_PATH_PLACEHOLDER


FrozenJsonObject = Annotated[
    Mapping[str, object],
    PlainValidator(_freeze_json_object, json_schema_input_type=dict[str, object]),
    PlainSerializer(_thaw_json_object, return_type=dict[str, object]),
]


class _ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegistrationProvenance(_ManifestModel):
    kind: Literal["project", "built_in", "external", "dynamic", "unavailable"]
    symbol: str
    location: str | None = None


class CapabilityManifest(_ManifestModel):
    name: str
    declared: bool
    resolved: bool
    process_available: Literal["not_checked"] = "not_checked"
    live_verified: Literal["not_run"] = "not_run"


class ToolManifest(_ManifestModel):
    name: str
    description: str
    effect: str
    parallel_safe: bool
    input_schema: FrozenJsonObject = Field(default_factory=lambda: MappingProxyType({}))
    policy_coverage: Literal["allowed", "denied", "approval_required", "conditional", "unknown"]
    registration_provenance: RegistrationProvenance
    implementation_provenance: RegistrationProvenance


class AgentManifest(_ManifestModel):
    name: str
    model: str
    configured_provider: str | None
    resolved_provider: str | None
    provider_resolution: Literal["explicit", "model_pattern", "default", "missing", "ambiguous"]
    provider_candidates: tuple[str, ...] = ()
    tools: tuple[ToolManifest, ...] = ()
    tool_policy: str
    context_policy: str
    context_overflow_policy: str | None
    runtime_hooks: tuple[str, ...] = ()
    loop_policies: tuple[str, ...] = ()
    registration_provenance: RegistrationProvenance
    implementation_provenance: RegistrationProvenance


class ProviderManifest(_ManifestModel):
    name: str
    model_patterns: tuple[str, ...]
    is_default: bool
    implementation: str
    usage_dialect: str
    supports_native_structured_output: bool
    registration_provenance: RegistrationProvenance
    implementation_provenance: RegistrationProvenance


class EnvironmentManifest(_ManifestModel):
    name: str
    is_default: bool
    factory_backed: bool
    workspace: str | None
    runner: str | None
    artifact_store: str | None
    vault: str | None
    credential_proxy: str | None
    binding: str | None
    knowledge_store: str | None
    mcp_servers: tuple[str, ...]
    registration_provenance: RegistrationProvenance
    implementation_provenance: RegistrationProvenance


class StoreManifest(_ManifestModel):
    session: str
    task: str | None
    knowledge: str | None
    budget: str
    budget_ledger: str
    event_watcher: str


class ApplicationDefaultsManifest(_ManifestModel):
    provider: str | None
    environment: str | None


class RuntimeManifest(_ManifestModel):
    dispatcher: str
    retry_policy: str | None
    budget_policy: str | None
    runtime_hooks: tuple[str, ...]
    loop_policies: tuple[str, ...]
    event_sinks: tuple[str, ...]
    mcp_manifest_policy: str | None
    context_counting: str
    max_file_attachment_bytes: int
    max_total_file_attachment_bytes: int
    max_file_attachments_per_request: int
    tool_timeout_seconds: float | None
    max_parallel_tool_calls: int


class AppManifest(_ManifestModel):
    schema_version: Literal["1"] = APP_MANIFEST_SCHEMA_VERSION
    fingerprint: str
    agents: tuple[AgentManifest, ...]
    providers: tuple[ProviderManifest, ...]
    environments: tuple[EnvironmentManifest, ...]
    stores: StoreManifest
    defaults: ApplicationDefaultsManifest
    runtime: RuntimeManifest
    capabilities: tuple[CapabilityManifest, ...]


def describe_app(app: CayuApp, *, project_root: str | Path | None = None) -> AppManifest:
    """Build a deterministic, side-effect-free description of a booted application."""

    root = Path(project_root).resolve() if project_root is not None else None
    providers = tuple(
        ProviderManifest(
            name=name,
            model_patterns=tuple(sorted(registration.model_patterns)),
            is_default=name == app._default_provider_name,
            implementation=_type_name(registration.provider),
            usage_dialect=str(registration.provider.usage_dialect),
            supports_native_structured_output=bool(
                registration.provider.supports_native_structured_output
            ),
            registration_provenance=_registration_provenance(
                registration.registration_source,
                registration.registration_symbol,
                root,
                fallback=registration.provider,
            ),
            implementation_provenance=_provenance(registration.provider, root),
        )
        for name, registration in sorted(app._providers.items())
    )
    agents = tuple(
        _describe_agent(app, name=name, registration=registration, project_root=root)
        for name, registration in sorted(app._agents.items())
    )
    environments = tuple(
        _describe_environment(app, name=name, registration=registration, project_root=root)
        for name, registration in sorted(app._environments.items())
    )
    stores = StoreManifest(
        session=_type_name(app.session_store),
        task=_optional_type_name(app.task_store),
        knowledge=_optional_type_name(app.knowledge_store),
        budget=_type_name(app.budget_store),
        budget_ledger=_type_name(app.budget_ledger),
        event_watcher=_type_name(app.event_watcher_store),
    )
    defaults = ApplicationDefaultsManifest(
        provider=app._default_provider_name,
        environment=app._default_environment_name,
    )
    runtime = RuntimeManifest(
        dispatcher=_type_name(app.dispatcher),
        retry_policy=_optional_type_name(app._default_retry_policy),
        budget_policy=_optional_type_name(app.budget_policy),
        runtime_hooks=tuple(_type_name(item) for item in app._runtime_hooks),
        loop_policies=tuple(_type_name(item) for item in app._loop_policies),
        event_sinks=tuple(_type_name(item) for item in app._event_sinks),
        mcp_manifest_policy=_optional_type_name(app._mcp_manifest_policy),
        context_counting=_type_name(app._context_counting),
        max_file_attachment_bytes=app._max_file_attachment_bytes,
        max_total_file_attachment_bytes=app._max_total_file_attachment_bytes,
        max_file_attachments_per_request=app._max_file_attachments_per_request,
        tool_timeout_seconds=app._tool_timeout_seconds,
        max_parallel_tool_calls=app._max_parallel_tool_calls,
    )
    capabilities = _capabilities(
        agents=agents,
        providers=providers,
        environments=environments,
    )
    payload = {
        "schema_version": APP_MANIFEST_SCHEMA_VERSION,
        "agents": [agent.model_dump(mode="json") for agent in agents],
        "providers": [provider.model_dump(mode="json") for provider in providers],
        "environments": [environment.model_dump(mode="json") for environment in environments],
        "stores": stores.model_dump(mode="json"),
        "defaults": defaults.model_dump(mode="json"),
        "runtime": runtime.model_dump(mode="json"),
        "capabilities": [item.model_dump(mode="json") for item in capabilities],
    }
    redacted_payload = app.redact_json(payload)
    normalized_payload = _normalize_manifest_unstable_values(redacted_payload)
    if type(normalized_payload) is not dict:
        raise TypeError("Application manifest redaction must preserve the JSON object shape.")
    canonical = json.dumps(
        normalized_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return AppManifest.model_validate(
        {
            **normalized_payload,
            "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }
    )


def _describe_agent(
    app: CayuApp,
    *,
    name: str,
    registration: runtime_records.RegisteredAgentState,
    project_root: Path | None,
) -> AgentManifest:
    spec = registration.spec
    registration_provenance = _registration_provenance(
        registration.registration_source,
        registration.registration_symbol,
        project_root,
        fallback=spec,
    )
    candidates = tuple(
        provider_name
        for provider_name, provider in sorted(app._providers.items())
        if any(fnmatchcase(spec.model, pattern) for pattern in provider.model_patterns)
    )
    if spec.provider_name is not None:
        resolved = spec.provider_name if spec.provider_name in app._providers else None
        resolution = "explicit" if resolved is not None else "missing"
        candidates = (spec.provider_name,)
    elif len(candidates) == 1:
        resolved = candidates[0]
        resolution = "model_pattern"
    elif len(candidates) > 1:
        resolved = None
        resolution = "ambiguous"
    elif app._default_provider_name is not None:
        resolved = app._default_provider_name
        resolution = "default"
    else:
        resolved = None
        resolution = "missing"
    return AgentManifest(
        name=name,
        model=spec.model,
        configured_provider=spec.provider_name,
        resolved_provider=resolved,
        provider_resolution=resolution,
        provider_candidates=candidates,
        tools=tuple(
            ToolManifest(
                name=tool_name,
                description=tool.description,
                effect=tool.effect.value,
                parallel_safe=tool.parallel_safe,
                input_schema=app.redact_json(tool.schema),
                policy_coverage=_tool_policy_coverage(registration.tool_policy, tool_name),
                registration_provenance=registration_provenance,
                implementation_provenance=_provenance(tool.tool, project_root),
            )
            for tool_name, tool in sorted(registration.tools.items())
        ),
        tool_policy=_type_name(registration.tool_policy),
        context_policy=_type_name(registration.context_policy),
        context_overflow_policy=_optional_type_name(registration.context_overflow_policy),
        runtime_hooks=tuple(_type_name(item) for item in registration.runtime_hooks),
        loop_policies=tuple(_type_name(item) for item in registration.loop_policies),
        registration_provenance=registration_provenance,
        implementation_provenance=_provenance(spec, project_root),
    )


def _describe_environment(
    app: CayuApp,
    *,
    name: str,
    registration: runtime_records.RegisteredEnvironment,
    project_root: Path | None,
) -> EnvironmentManifest:
    environment = registration.environment
    mcp_servers = tuple(
        sorted(getattr(server, "name", _type_name(server)) for server in environment.mcp_servers)
    )
    described_object = registration.factory or environment
    return EnvironmentManifest(
        name=name,
        is_default=name == app._default_environment_name,
        factory_backed=registration.factory is not None,
        workspace=_optional_type_name(environment.workspace),
        runner=_optional_type_name(environment.runner),
        artifact_store=_optional_type_name(environment.artifact_store),
        vault=_optional_type_name(environment.vault),
        credential_proxy=_optional_type_name(environment.proxy),
        binding=_optional_type_name(environment.binding),
        knowledge_store=_optional_type_name(environment.knowledge_store),
        mcp_servers=mcp_servers,
        registration_provenance=_registration_provenance(
            registration.registration_source,
            registration.registration_symbol,
            project_root,
            fallback=described_object,
        ),
        implementation_provenance=_provenance(described_object, project_root),
    )


def _capabilities(
    *,
    agents: tuple[AgentManifest, ...],
    providers: tuple[ProviderManifest, ...],
    environments: tuple[EnvironmentManifest, ...],
) -> tuple[CapabilityManifest, ...]:
    present = {
        "agents": bool(agents),
        "providers": bool(providers),
        "environments": bool(environments),
        "workspaces": any(item.workspace is not None for item in environments),
        "runners": any(item.runner is not None for item in environments),
        "artifacts": any(item.artifact_store is not None for item in environments),
    }
    return tuple(
        CapabilityManifest(name=name, declared=value, resolved=value)
        for name, value in sorted(present.items())
    )


def _type_name(value: object) -> str:
    return type(value).__name__


def _optional_type_name(value: object | None) -> str | None:
    return None if value is None else _type_name(value)


def _tool_policy_coverage(
    policy: ToolPolicy,
    tool_name: str,
) -> Literal["allowed", "denied", "approval_required", "conditional", "unknown"]:
    # These descriptions are static facts about Cayu's concrete built-ins. A
    # subclass can override authorize(), so treating it as its parent would
    # turn an unknown custom policy into trusted coverage.
    if type(policy) is AllowAllToolPolicy:
        return "allowed"
    if type(policy) is AlwaysRequireApprovalToolPolicy:
        return (
            "approval_required" if policy.tools is None or tool_name in policy.tools else "allowed"
        )
    if type(policy) is StaticToolPolicy:
        if tool_name in policy.deny or (policy.allow is not None and tool_name not in policy.allow):
            return "denied"
        return "allowed"
    if type(policy) is ParameterConstrainedToolPolicy:
        return "conditional" if tool_name in policy.rules else "allowed"
    if type(policy) is TaintAwareToolPolicy:
        return "conditional" if policy.protected_labels_for_tool(tool_name) else "allowed"
    return "unknown"


def _provenance(value: object, project_root: Path | None) -> RegistrationProvenance:
    target = value if inspect.isclass(value) or inspect.isfunction(value) else type(value)
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", None))
    symbol = ":".join(part for part in (module, qualname) if part) or "unavailable"
    if module == "__main__" or module is None:
        kind: Literal["project", "built_in", "external", "dynamic", "unavailable"] = "dynamic"
    elif module == "cayu" or module.startswith("cayu."):
        kind = "built_in"
    else:
        kind = "external"
    location: str | None = None
    try:
        source = inspect.getsourcefile(target)
    except (TypeError, OSError):
        source = None
    if source is not None and project_root is not None:
        try:
            relative = Path(source).resolve().relative_to(project_root)
        except (OSError, ValueError):
            pass
        else:
            kind = "project"
            location = relative.as_posix()
    if symbol == "unavailable":
        kind = "unavailable"
    return RegistrationProvenance(kind=kind, symbol=symbol, location=location)


def _registration_provenance(
    source: str | None,
    symbol: str | None,
    project_root: Path | None,
    *,
    fallback: object,
) -> RegistrationProvenance:
    if source is None or symbol is None:
        return _provenance(fallback, project_root)
    location: str | None = None
    kind: Literal["project", "built_in", "external", "dynamic", "unavailable"] = "external"
    if project_root is not None:
        try:
            location = Path(source).resolve().relative_to(project_root).as_posix()
        except (OSError, ValueError):
            pass
        else:
            kind = "project"
    if symbol.startswith("cayu."):
        kind = "built_in"
    return RegistrationProvenance(kind=kind, symbol=symbol, location=location)
