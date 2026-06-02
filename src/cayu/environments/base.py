from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_json_value, require_nonblank
from cayu.mcp import McpServerSpec
from cayu.runners import Runner
from cayu.vaults import Vault
from cayu.workspaces import Workspace


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
        return require_nonblank(value, info.field_name)


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
        runner: Runner | None = None,
        vault: Vault | None = None,
        mcp_servers: Iterable[McpServerSpec] | None = None,
    ) -> None:
        if type(spec) is not EnvironmentSpec:
            raise TypeError("Environment requires an EnvironmentSpec.")
        self.spec = copy_environment_spec(spec)

        if workspace is not None and not isinstance(workspace, Workspace):
            raise TypeError("workspace must be a Workspace.")
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
        self.runner = runner
        self.vault = vault
        self.mcp_servers = tuple(copy_mcp_server_spec(server) for server in servers)


def copy_environment(environment: Environment) -> Environment:
    if type(environment) is not Environment:
        raise TypeError("Environment copies require an Environment.")
    return Environment(
        copy_environment_spec(environment.spec),
        workspace=environment.workspace,
        runner=environment.runner,
        vault=environment.vault,
        mcp_servers=environment.mcp_servers,
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
