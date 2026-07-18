from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cayu import (
    AgentAuthoringState,
    AgentSpec,
    AppManifest,
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    ExecutionRequirements,
    ScriptedModelProvider,
    SecretRedactor,
    Tool,
    ToolEffect,
    ToolResult,
    ToolSpec,
)


class _SchemaTool(Tool):
    spec = ToolSpec(
        name="schema_tool",
        effect=ToolEffect.NONE,
        input_schema={
            "type": "object",
            "properties": {"token": {"const": "manifest-secret"}},
            "additionalProperties": False,
        },
    )

    async def run(self, ctx, args):
        return ToolResult(content="ok")


class _PathSchemaTool(Tool):
    def __init__(
        self,
        *,
        posix_path: str,
        windows_path: str,
        reverse_path_keys: bool = False,
    ) -> None:
        path_properties = [
            (f"{posix_path}/alpha", {"const": "alpha"}),
            (f"{posix_path}/beta", {"const": "beta"}),
        ]
        if reverse_path_keys:
            path_properties.reverse()
        super().__init__(
            ToolSpec(
                name="path_schema_tool",
                effect=ToolEffect.NONE,
                input_schema={
                    "type": "object",
                    "description": f"Write the result to {posix_path} before returning.",
                    "comment": f"output:{posix_path}",
                    "properties": {
                        **dict(path_properties),
                        "output": {
                            "type": "string",
                            "default": posix_path,
                            "examples": [
                                windows_path,
                                f"file://{posix_path}",
                                "relative/output.txt",
                                "https://example.com/output",
                            ],
                        },
                    },
                    "additionalProperties": False,
                },
            )
        )

    async def run(self, ctx, args):
        return ToolResult(content="ok")


class _UnstableSchemaTool(Tool):
    def __init__(self, *, memory_address: str, timestamp: str) -> None:
        super().__init__(
            ToolSpec(
                name="unstable_schema_tool",
                effect=ToolEffect.NONE,
                input_schema={
                    "type": "object",
                    "properties": {
                        "manifest-secret": {
                            "description": f"Captured <object object at {memory_address}>",
                            "default": timestamp,
                        },
                        "debug": {
                            "description": f"Raw address {memory_address}",
                        },
                    },
                    "additionalProperties": False,
                },
            )
        )

    async def run(self, ctx, args):
        return ToolResult(content="ok")


class _UncalledEnvironmentFactory(EnvironmentFactory):
    def __init__(self) -> None:
        self.called = False

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.called = True
        raise AssertionError("describe() must not invoke environment factories")


def _described_app(*, reverse: bool = False) -> CayuApp:
    app = CayuApp(enable_logging=False)
    providers = (
        ScriptedModelProvider([], name="secondary"),
        ScriptedModelProvider([], name="primary"),
    )
    if reverse:
        providers = tuple(reversed(providers))
    for provider in providers:
        app.register_provider(
            provider,
            default=provider.name == "primary",
            model_patterns=("primary-*",) if provider.name == "primary" else (),
        )
    agents = (
        AgentSpec(name="writer", model="primary-small"),
        AgentSpec(name="reviewer", model="other", provider_name="secondary"),
    )
    if reverse:
        agents = tuple(reversed(agents))
    for agent in agents:
        app.register_agent(agent)
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
    return app


def test_describe_returns_a_deterministic_public_application_manifest() -> None:
    manifest = _described_app().describe()
    reversed_manifest = _described_app(reverse=True).describe()

    assert manifest.schema_version == "3"
    assert manifest.defaults.provider == "primary"
    assert manifest.defaults.environment == "local"
    assert [agent.name for agent in manifest.agents] == ["reviewer", "writer"]
    assert [agent.resolved_provider for agent in manifest.agents] == ["secondary", "primary"]
    assert [provider.name for provider in manifest.providers] == ["primary", "secondary"]
    assert manifest.model_dump(mode="json") == reversed_manifest.model_dump(mode="json")
    assert manifest.fingerprint == reversed_manifest.fingerprint
    assert len(manifest.fingerprint) == 64


def test_agent_authoring_state_is_typed_copied_and_fingerprinted() -> None:
    spec = AgentSpec(
        name="generated",
        model="model",
        authoring_state=AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET,
    )
    app = CayuApp(enable_logging=False)
    app.register_agent(spec)

    marked = app.describe()
    spec.authoring_state = None

    assert app.get_agent("generated").spec.authoring_state is (
        AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET
    )
    assert marked.agents[0].authoring_state is (
        AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET
    )

    complete = CayuApp(enable_logging=False)
    complete.register_agent(AgentSpec(name="generated", model="model"))
    completed_manifest = complete.describe()

    assert completed_manifest.agents[0].authoring_state is None
    assert marked.fingerprint != completed_manifest.fingerprint

    with pytest.raises(ValidationError, match="authoring_state"):
        AgentSpec(name="bad", model="model", authoring_state="complete")


def test_agent_execution_requirements_are_typed_and_fingerprinted() -> None:
    trusted = CayuApp(enable_logging=False)
    trusted.register_agent(AgentSpec(name="worker", model="model"))

    isolated = CayuApp(enable_logging=False)
    isolated.register_agent(
        AgentSpec(name="worker", model="model"),
        execution_requirements=ExecutionRequirements.untrusted(),
    )

    trusted_manifest = trusted.describe()
    isolated_manifest = isolated.describe()

    assert trusted_manifest.agents[0].execution_requirements == ExecutionRequirements.trusted()
    assert isolated_manifest.agents[0].execution_requirements == (ExecutionRequirements.untrusted())
    assert trusted_manifest.fingerprint != isolated_manifest.fingerprint


def test_describe_reports_no_default_for_a_non_default_static_environment() -> None:
    app = CayuApp(enable_logging=False)
    app.register_environment(Environment(EnvironmentSpec(name="optional")), default=False)

    manifest = app.describe()

    assert manifest.defaults.environment is None
    assert [(item.name, item.is_default) for item in manifest.environments] == [("optional", False)]


def test_describe_reports_no_default_for_a_non_default_environment_factory() -> None:
    app = CayuApp(enable_logging=False)
    factory = _UncalledEnvironmentFactory()
    app.register_environment_factory(
        EnvironmentSpec(name="optional"),
        factory,
        default=False,
    )

    manifest = app.describe()

    assert manifest.defaults.environment is None
    assert [(item.name, item.is_default) for item in manifest.environments] == [("optional", False)]
    assert factory.called is False


def test_explicit_default_replaces_the_previous_default_and_non_default_does_not() -> None:
    app = CayuApp(enable_logging=False)
    optional_factory = _UncalledEnvironmentFactory()
    replacement_factory = _UncalledEnvironmentFactory()
    app.register_environment(
        Environment(EnvironmentSpec(name="primary")),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="optional"),
        optional_factory,
        default=False,
    )

    before_replacement = app.describe()

    assert before_replacement.defaults.environment == "primary"
    assert {item.name: item.is_default for item in before_replacement.environments} == {
        "optional": False,
        "primary": True,
    }

    app.register_environment_factory(
        EnvironmentSpec(name="replacement"),
        replacement_factory,
        default=True,
    )
    after_replacement = app.describe()

    assert after_replacement.defaults.environment == "replacement"
    assert {item.name: item.is_default for item in after_replacement.environments} == {
        "optional": False,
        "primary": False,
        "replacement": True,
    }
    assert optional_factory.called is False
    assert replacement_factory.called is False


def test_manifest_is_public_versioned_redacted_and_deeply_read_only(tmp_path: Path) -> None:
    app = CayuApp(
        enable_logging=False,
        secret_redactor=SecretRedactor("manifest-secret"),
    )
    app.register_agent(
        AgentSpec(name="reader", model="manifest-secret"),
        tools=[_SchemaTool()],
    )
    factory = _UncalledEnvironmentFactory()
    app.register_environment_factory(EnvironmentSpec(name="dynamic"), factory)

    manifest = app.describe(project_root=tmp_path)
    payload = manifest.model_dump_json()
    schema = AppManifest.model_json_schema(mode="serialization")

    assert schema["properties"]["schema_version"]["const"] == "3"
    assert "manifest-secret" not in payload
    assert str(tmp_path) not in payload
    assert factory.called is False
    capabilities = {item.name: item for item in manifest.capabilities}
    assert capabilities["environments"].declared is True
    assert capabilities["workspaces"].declared is False
    assert capabilities["runners"].declared is False
    assert capabilities["artifacts"].declared is False
    tool_schema = manifest.agents[0].tools[0].input_schema
    with pytest.raises(TypeError):
        tool_schema["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        tool_schema["properties"]["token"] = {}  # type: ignore[index]


def test_manifest_normalizes_nested_absolute_schema_paths_before_fingerprinting() -> None:
    def describe(
        *,
        posix_path: str,
        windows_path: str,
        reverse_path_keys: bool = False,
    ) -> AppManifest:
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="reader", model="model"),
            tools=[
                _PathSchemaTool(
                    posix_path=posix_path,
                    windows_path=windows_path,
                    reverse_path_keys=reverse_path_keys,
                )
            ],
        )
        return app.describe()

    first = describe(
        posix_path="/private/tmp/host-a/output.txt",
        windows_path=r"C:\Users\alice\output.txt",
    )
    second = describe(
        posix_path="/var/tmp/host-b/output.txt",
        windows_path=r"D:\Users\bob\output.txt",
        reverse_path_keys=True,
    )

    schema = first.agents[0].tools[0].input_schema
    assert schema["description"] == "Write the result to [ABSOLUTE_PATH] before returning."
    assert schema["comment"] == "output:[ABSOLUTE_PATH]"
    assert "[ABSOLUTE_PATH]" in schema["properties"]
    assert schema["properties"]["[ABSOLUTE_PATH]"]["const"] == "alpha"
    assert schema["properties"]["[ABSOLUTE_PATH]_2"]["const"] == "beta"
    output = schema["properties"]["output"]
    assert output["default"] == "[ABSOLUTE_PATH]"
    assert output["examples"] == (
        "[ABSOLUTE_PATH]",
        "[ABSOLUTE_PATH]",
        "relative/output.txt",
        "https://example.com/output",
    )
    assert "/private/tmp/host-a" not in first.model_dump_json()
    assert r"C:\\Users\\alice" not in first.model_dump_json()
    assert first.fingerprint == second.fingerprint


def test_manifest_normalizes_secrets_and_unstable_schema_strings_before_fingerprinting() -> None:
    def describe(*, memory_address: str, timestamp: str) -> AppManifest:
        app = CayuApp(
            enable_logging=False,
            secret_redactor=SecretRedactor("manifest-secret"),
        )
        app.register_agent(
            AgentSpec(name="reader", model="model"),
            tools=[
                _UnstableSchemaTool(
                    memory_address=memory_address,
                    timestamp=timestamp,
                )
            ],
        )
        return app.describe()

    first = describe(
        memory_address="0xDEADBEEF",
        timestamp="2026-07-13T23:59:59+00:00",
    )
    second = describe(
        memory_address="0xCAFEBABE",
        timestamp="2027-08-14T12:34:56.123456Z",
    )

    payload = first.model_dump_json()
    schema = first.agents[0].tools[0].input_schema
    assert "manifest-secret" not in payload
    assert "0xDEADBEEF" not in payload
    assert "2026-07-13T23:59:59+00:00" not in payload
    assert "[REDACTED_SECRET]" in schema["properties"]
    assert schema["properties"]["[REDACTED_SECRET]"]["description"] == (
        "Captured [OBJECT_REPRESENTATION]"
    )
    assert schema["properties"]["[REDACTED_SECRET]"]["default"] == "[TIMESTAMP]"
    assert schema["properties"]["debug"]["description"] == ("Raw address [MEMORY_ADDRESS]")
    assert first.fingerprint == second.fingerprint


def test_manifest_rejects_non_json_schema_payloads() -> None:
    with pytest.raises(ValidationError, match="JSON-compatible"):
        AppManifest.model_validate(
            {
                "schema_version": "3",
                "fingerprint": "0" * 64,
                "agents": [
                    {
                        "name": "agent",
                        "model": "model",
                        "configured_provider": None,
                        "resolved_provider": None,
                        "provider_resolution": "missing",
                        "execution_requirements": ExecutionRequirements.trusted().model_dump(
                            mode="json"
                        ),
                        "tools": [
                            {
                                "name": "bad",
                                "description": "",
                                "effect": "none",
                                "parallel_safe": True,
                                "input_schema": {"bad": object()},
                                "policy_coverage": "allowed",
                                "registration_provenance": {
                                    "kind": "unavailable",
                                    "symbol": "unavailable",
                                },
                                "implementation_provenance": {
                                    "kind": "unavailable",
                                    "symbol": "unavailable",
                                },
                            }
                        ],
                        "tool_policy": "AllowAllToolPolicy",
                        "context_policy": "FullContextPolicy",
                        "registration_provenance": {
                            "kind": "unavailable",
                            "symbol": "unavailable",
                        },
                        "implementation_provenance": {
                            "kind": "unavailable",
                            "symbol": "unavailable",
                        },
                    }
                ],
                "providers": [],
                "environments": [],
                "stores": {
                    "session": "InMemorySessionStore",
                    "task": None,
                    "knowledge": None,
                    "budget": "InMemoryBudgetStore",
                    "budget_ledger": "InMemoryBudgetLedger",
                    "event_watcher": "InMemoryEventWatcherStore",
                },
                "defaults": {"provider": None, "environment": None},
                "runtime": {
                    "dispatcher": "InlineDispatcher",
                    "retry_policy": None,
                    "budget_policy": None,
                    "runtime_hooks": [],
                    "loop_policies": [],
                    "event_sinks": [],
                    "mcp_manifest_policy": None,
                    "context_counting": "ContextCountingConfig",
                    "max_file_attachment_bytes": 1,
                    "max_total_file_attachment_bytes": 1,
                    "max_file_attachments_per_request": 1,
                    "tool_timeout_seconds": None,
                    "max_parallel_tool_calls": 1,
                },
                "capabilities": [],
            }
        )
