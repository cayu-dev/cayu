from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import (
    copy_json_value,
    require_clean_nonblank,
    require_nonblank,
    require_unicode_scalar_text,
)
from cayu.artifacts import ArtifactStore
from cayu.environments.bindings import WorkspaceBinding
from cayu.mcp import McpServerSpec
from cayu.proxies import CredentialProxy
from cayu.runners import Runner
from cayu.vaults import ResolvedSecret, SecretRef, Vault, VaultError
from cayu.workspaces import Workspace

if TYPE_CHECKING:
    from cayu.storage.memory import KnowledgeStore
else:
    KnowledgeStore = Any

DEFAULT_WORKSPACE_INSTRUCTION_PATHS = ("AGENTS.md", ".cayu/AGENTS.md")
DEFAULT_WORKSPACE_INSTRUCTIONS_MAX_BYTES = 32 * 1024
_KNOWLEDGE_STORE_METHODS = (
    "put_entry",
    "get_entry",
    "update_entry_status",
    "transition_entry_status",
    "delete_entry",
    "replace_chunks",
    "put_entry_with_chunks",
    "read_chunks",
    "search",
    "list_entries",
)


class EnvironmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _reject_live_object_kwargs(cls, data: Any) -> Any:
        # workspace/runner/binding/vault/proxy are live objects passed to
        # Environment(spec, workspace=..., ...), not fields of EnvironmentSpec.
        if isinstance(data, dict):
            misplaced = sorted({"workspace", "runner", "binding", "vault", "proxy"} & set(data))
            if misplaced:
                names = ", ".join(misplaced)
                raise ValueError(
                    f"EnvironmentSpec does not accept {names}; pass it to "
                    f"Environment(spec, {misplaced[0]}=...), not EnvironmentSpec(...)."
                )
        return data

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("name")
    @classmethod
    def validate_nonblank_name(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


class WorkspaceInstructions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    sources: tuple[str, ...] = ("explicit",)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return require_nonblank(value, "content")

    @field_validator("sources", mode="before")
    @classmethod
    def validate_sources(cls, value) -> tuple[str, ...]:
        if isinstance(value, str | bytes):
            sources = (value,)
        else:
            try:
                sources = tuple(value)
            except TypeError as exc:
                raise TypeError("sources must be an iterable of strings.") from exc
        if not sources:
            raise ValueError("sources must contain at least one entry.")
        return tuple(require_nonblank(source, "source") for source in sources)


class WorkspaceInstructionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: tuple[str, ...] = DEFAULT_WORKSPACE_INSTRUCTION_PATHS
    mode: Literal["first_found", "merge"] = "first_found"
    max_bytes: int = DEFAULT_WORKSPACE_INSTRUCTIONS_MAX_BYTES

    @field_validator("paths", mode="before")
    @classmethod
    def validate_paths(cls, value) -> tuple[str, ...]:
        if isinstance(value, str | bytes):
            paths = (value,)
        else:
            try:
                paths = tuple(value)
            except TypeError as exc:
                raise TypeError("paths must be an iterable of strings.") from exc
        if not paths:
            raise ValueError("paths must contain at least one entry.")
        return tuple(_validate_workspace_instruction_path(path) for path in paths)

    @field_validator("max_bytes")
    @classmethod
    def validate_max_bytes(cls, value: int) -> int:
        if type(value) is not int:
            raise TypeError("max_bytes must be an integer.")
        if value <= 0:
            raise ValueError("max_bytes must be greater than zero.")
        return value


WorkspaceInstructionsInput = str | WorkspaceInstructions | WorkspaceInstructionsConfig


class Environment:
    """Execution context an agent session can use.

    The environment is intentionally thin for now. Concrete local, Docker,
    hosted, or customer-hosted environments can bind workspace, runner, vault,
    and MCP services without making those details part of AgentSpec.
    """

    def __init__(
        self,
        spec: EnvironmentSpec,
        *,
        workspace: Workspace | None = None,
        artifact_store: ArtifactStore | None = None,
        runner: Runner | None = None,
        vault: Vault | None = None,
        proxy: CredentialProxy | None = None,
        knowledge_store: KnowledgeStore | None = None,
        binding: WorkspaceBinding | None = None,
        mcp_servers: Iterable[McpServerSpec] | None = None,
        workspace_instructions: WorkspaceInstructionsInput | None = None,
    ) -> None:
        if not isinstance(spec, EnvironmentSpec):
            raise TypeError("Environment requires an EnvironmentSpec.")
        self.spec = copy_environment_spec(spec)

        if workspace is not None and not isinstance(workspace, Workspace):
            raise TypeError("workspace must be a Workspace.")
        if artifact_store is not None and not isinstance(artifact_store, ArtifactStore):
            raise TypeError("artifact_store must be an ArtifactStore.")
        if artifact_store is not None:
            artifact_store_id = getattr(artifact_store, "id", None)
            if type(artifact_store_id) is not str:
                raise ValueError("`artifact_store.id` must be a string.")
            artifact_store_id = require_clean_nonblank(artifact_store_id, "artifact_store.id")
            require_unicode_scalar_text(artifact_store_id, "artifact_store.id")
        if runner is not None and not isinstance(runner, Runner):
            raise TypeError("runner must be a Runner.")
        if vault is not None and not isinstance(vault, Vault):
            raise TypeError("vault must be a Vault.")
        if proxy is not None and not isinstance(proxy, CredentialProxy):
            raise TypeError("proxy must be a CredentialProxy.")
        if knowledge_store is not None:
            _validate_knowledge_store(knowledge_store)
        if binding is not None and not isinstance(binding, WorkspaceBinding):
            raise TypeError("binding must be a WorkspaceBinding.")

        if mcp_servers is None:
            servers = []
        else:
            if isinstance(mcp_servers, str | bytes):
                raise TypeError("mcp_servers must be an iterable of McpServerSpec.")
            try:
                servers = list(mcp_servers)
            except TypeError as exc:
                raise TypeError("mcp_servers must be an iterable of McpServerSpec.") from exc

        self.workspace = workspace
        self.artifact_store = artifact_store
        self.runner = runner
        self.vault = vault
        self.proxy = proxy
        self.knowledge_store = knowledge_store
        self.binding = binding
        self.mcp_servers = tuple(copy_mcp_server_spec(server) for server in servers)
        self.workspace_instructions = copy_workspace_instructions_input(
            workspace_instructions,
        )

    async def resolve_secret(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        """Resolve a secret ref through the environment vault."""

        if not isinstance(ref, SecretRef):
            raise TypeError("Environment secret refs must be SecretRef instances.")
        if self.vault is None:
            raise VaultError(f"Environment has no vault configured: {self.spec.name}")
        return await self.vault.resolve(ref, scope=scope)


def copy_environment(environment: Environment) -> Environment:
    if not isinstance(environment, Environment):
        raise TypeError("Environment copies require an Environment.")
    return type(environment)(
        copy_environment_spec(environment.spec),
        workspace=environment.workspace,
        artifact_store=environment.artifact_store,
        runner=environment.runner,
        vault=environment.vault,
        proxy=environment.proxy,
        knowledge_store=environment.knowledge_store,
        binding=environment.binding,
        mcp_servers=environment.mcp_servers,
        workspace_instructions=environment.workspace_instructions,
    )


def _validate_knowledge_store(value: Any) -> None:
    for method_name in _KNOWLEDGE_STORE_METHODS:
        if not callable(getattr(value, method_name, None)):
            raise TypeError("knowledge_store must implement KnowledgeStore.")


def copy_environment_spec(spec: EnvironmentSpec) -> EnvironmentSpec:
    if not isinstance(spec, EnvironmentSpec):
        raise TypeError("Environment specs must be EnvironmentSpec instances.")
    if type(spec.name) is not str:
        # Exact-type on purpose: str subclasses can override strip()/__eq__
        # and defeat validation (see require_nonblank).
        raise ValueError("`name` must be a string.")
    return type(spec)(
        name=spec.name,
        metadata=copy_json_value(spec.metadata, "metadata"),
    )


def copy_mcp_server_spec(spec: McpServerSpec) -> McpServerSpec:
    if not isinstance(spec, McpServerSpec):
        raise TypeError("MCP server entries must be McpServerSpec instances.")
    return type(spec).model_validate(spec.model_dump())


def copy_workspace_instructions_input(
    instructions: WorkspaceInstructionsInput | None,
) -> WorkspaceInstructionsInput | None:
    if instructions is None:
        return None
    if isinstance(instructions, WorkspaceInstructions):
        return type(instructions).model_validate(instructions.model_dump())
    if isinstance(instructions, WorkspaceInstructionsConfig):
        return type(instructions).model_validate(instructions.model_dump())
    if isinstance(instructions, str):
        return WorkspaceInstructions(content=str(instructions))
    raise TypeError(
        "workspace_instructions must be a string, WorkspaceInstructions, "
        "WorkspaceInstructionsConfig, or None."
    )


async def load_workspace_instructions(environment: Environment) -> WorkspaceInstructions | None:
    if not isinstance(environment, Environment):
        raise TypeError("Environment instructions require an Environment.")
    instructions = environment.workspace_instructions
    if instructions is None:
        return None
    if isinstance(instructions, WorkspaceInstructions):
        return type(instructions).model_validate(instructions.model_dump())
    if isinstance(instructions, str):
        return WorkspaceInstructions(content=str(instructions))
    if not isinstance(instructions, WorkspaceInstructionsConfig):
        raise TypeError("Invalid workspace_instructions configuration.")
    if environment.workspace is None:
        return None

    loaded: list[WorkspaceInstructions] = []
    for path in instructions.paths:
        try:
            result = await environment.workspace.read_bytes(
                path,
                max_bytes=instructions.max_bytes,
            )
        except FileNotFoundError:
            continue
        if result.truncated:
            raise ValueError(
                f"Workspace instructions file exceeds {instructions.max_bytes} bytes: {path}"
            )
        try:
            content = result.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Workspace instructions file is not valid UTF-8: {path}") from exc
        if not content.strip():
            continue
        loaded.append(WorkspaceInstructions(content=content, sources=(path,)))
        if instructions.mode == "first_found":
            break

    if not loaded:
        return None
    if instructions.mode == "first_found":
        return loaded[0]

    merged_sections = [
        f"Source: {source.sources[0]}\n{source.content.strip()}" for source in loaded
    ]
    return WorkspaceInstructions(
        content="\n\n---\n\n".join(merged_sections),
        sources=tuple(source.sources[0] for source in loaded),
    )


def _validate_workspace_instruction_path(path: str) -> str:
    path = require_nonblank(path, "path")
    candidate = PurePosixPath(path)
    if candidate.is_absolute():
        raise ValueError("workspace instruction paths must be relative.")
    if ".." in candidate.parts:
        raise ValueError("workspace instruction paths must stay inside the workspace.")
    return candidate.as_posix()
