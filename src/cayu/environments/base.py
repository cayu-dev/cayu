from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.artifacts import ArtifactStore
from cayu.mcp import McpServerSpec
from cayu.runners import Runner
from cayu.vaults import ResolvedSecret, SecretRef, Vault, VaultError
from cayu.workspaces import Workspace

DEFAULT_WORKSPACE_INSTRUCTION_PATHS = ("AGENTS.md", ".cayu/AGENTS.md")
DEFAULT_WORKSPACE_INSTRUCTIONS_MAX_BYTES = 32 * 1024


class EnvironmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)

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
        mcp_servers: Iterable[McpServerSpec] | None = None,
        workspace_instructions: WorkspaceInstructionsInput | None = None,
    ) -> None:
        if type(spec) is not EnvironmentSpec:
            raise TypeError("Environment requires an EnvironmentSpec.")
        self.spec = copy_environment_spec(spec)

        if workspace is not None and not isinstance(workspace, Workspace):
            raise TypeError("workspace must be a Workspace.")
        if artifact_store is not None and not isinstance(artifact_store, ArtifactStore):
            raise TypeError("artifact_store must be an ArtifactStore.")
        if runner is not None and not isinstance(runner, Runner):
            raise TypeError("runner must be a Runner.")
        if vault is not None and not isinstance(vault, Vault):
            raise TypeError("vault must be a Vault.")

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

        if type(ref) is not SecretRef:
            raise TypeError("Environment secret refs must be SecretRef instances.")
        if self.vault is None:
            raise VaultError(f"Environment has no vault configured: {self.spec.name}")
        return await self.vault.resolve(ref, scope=scope)


def copy_environment(environment: Environment) -> Environment:
    if type(environment) is not Environment:
        raise TypeError("Environment copies require an Environment.")
    return Environment(
        copy_environment_spec(environment.spec),
        workspace=environment.workspace,
        artifact_store=environment.artifact_store,
        runner=environment.runner,
        vault=environment.vault,
        mcp_servers=environment.mcp_servers,
        workspace_instructions=environment.workspace_instructions,
    )


def copy_environment_spec(spec: EnvironmentSpec) -> EnvironmentSpec:
    if type(spec) is not EnvironmentSpec:
        raise TypeError("Environment specs must be EnvironmentSpec instances.")
    if type(spec.name) is not str:
        raise ValueError("`name` must be a string.")
    return EnvironmentSpec(
        name=spec.name,
        metadata=copy_json_value(spec.metadata, "metadata"),
    )


def copy_mcp_server_spec(spec: McpServerSpec) -> McpServerSpec:
    if type(spec) is not McpServerSpec:
        raise TypeError("MCP server entries must be McpServerSpec instances.")
    return McpServerSpec.model_validate(spec.model_dump())


def copy_workspace_instructions_input(
    instructions: WorkspaceInstructionsInput | None,
) -> WorkspaceInstructionsInput | None:
    if instructions is None:
        return None
    if type(instructions) is str:
        return WorkspaceInstructions(content=instructions)
    if type(instructions) is WorkspaceInstructions:
        return WorkspaceInstructions.model_validate(instructions.model_dump())
    if type(instructions) is WorkspaceInstructionsConfig:
        return WorkspaceInstructionsConfig.model_validate(instructions.model_dump())
    raise TypeError(
        "workspace_instructions must be a string, WorkspaceInstructions, "
        "WorkspaceInstructionsConfig, or None."
    )


async def load_workspace_instructions(environment: Environment) -> WorkspaceInstructions | None:
    if type(environment) is not Environment:
        raise TypeError("Environment instructions require an Environment.")
    instructions = environment.workspace_instructions
    if instructions is None:
        return None
    if type(instructions) is WorkspaceInstructions:
        return WorkspaceInstructions.model_validate(instructions.model_dump())
    if type(instructions) is str:
        return WorkspaceInstructions(content=instructions)
    if type(instructions) is not WorkspaceInstructionsConfig:
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
