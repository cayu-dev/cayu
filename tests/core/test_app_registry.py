"""Tests for CayuApp registration, lookup, and registry introspection."""

from __future__ import annotations

import hashlib

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    DefaultContextPolicy,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    LocalWorkspace,
    ScriptedModelProvider,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.artifacts import LocalArtifactStore


class _UncalledEnvironmentFactory(EnvironmentFactory):
    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        del request
        raise AssertionError("registry introspection must not materialize environment factories")


class _RegistryTool(Tool):
    spec = ToolSpec(
        name="registry_tool",
        description="Exercise tool registration.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        return ToolResult(content=args["text"])


def test_registry_introspection_lists_sorted_names() -> None:
    app = CayuApp()
    provider = ScriptedModelProvider([])
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="beta", model="m"))
    app.register_agent(AgentSpec(name="alpha", model="m"))
    app.register_environment(Environment(EnvironmentSpec(name="env-1")), default=True)

    assert app.list_agents() == ("alpha", "beta")
    assert app.list_providers() == (provider.name,)
    assert app.list_environments() == ("env-1",)


def test_empty_registries_are_empty_tuples() -> None:
    app = CayuApp()
    assert app.list_agents() == ()
    assert app.list_providers() == ()
    assert app.list_environments() == ()


def test_cayu_app_rejects_duplicate_agents_and_missing_registrations() -> None:
    app = CayuApp()
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(ValueError, match="Agent already registered"):
        app.register_agent(AgentSpec(name="assistant", model="other-model"))

    with pytest.raises(RuntimeError, match="No model provider"):
        app.get_provider()

    with pytest.raises(KeyError, match="Agent not registered"):
        app.get_agent("missing")


def test_cayu_app_rejects_invalid_agent_registration_inputs() -> None:
    class ToolLike:
        name = "tool_like"
        description = "Not actually a Tool."
        schema = {}

    class BadString(str):
        def strip(self):
            raise RuntimeError("strip should not run")

    class BadMetadata(dict):
        def items(self):
            raise RuntimeError("agent metadata traversal should not run")

    class BadSchema(dict):
        def items(self):
            raise RuntimeError("tool schema traversal should not run")

    app = CayuApp()

    with pytest.raises(TypeError, match="AgentSpec"):
        app.register_agent({"name": "assistant"})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="`name` cannot be blank"):
        app.register_agent(
            AgentSpec.model_construct(
                name=" ",
                model="fake-model",
                metadata={},
            )
        )

    with pytest.raises(ValueError, match="JSON-compatible"):
        app.register_agent(
            AgentSpec.model_construct(
                name="bad_metadata",
                model="fake-model",
                metadata=BadMetadata({"bad": "value"}),
            )
        )

    with pytest.raises(TypeError, match="Tool"):
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[ToolLike()],  # type: ignore[list-item]
        )

    blank_tool = _RegistryTool()
    blank_tool.spec = ToolSpec.model_construct(name=" ", description="")
    with pytest.raises(ValueError, match="`name` cannot be blank"):
        app.register_agent(
            AgentSpec(name="bad_tool_name", model="fake-model"),
            tools=[blank_tool],
        )

    bad_string_tool = _RegistryTool()
    bad_string_tool.spec = ToolSpec.model_construct(
        name=BadString("bad_string_tool"),
        description="",
    )
    with pytest.raises(ValueError, match="must be a string"):
        app.register_agent(
            AgentSpec(name="bad_tool_string_name", model="fake-model"),
            tools=[bad_string_tool],
        )

    bad_schema_tool = _RegistryTool()
    bad_schema = ToolSpec.model_construct(name="bad_schema", description="")
    object.__setattr__(bad_schema, "_input_schema", BadSchema({"type": "object"}))
    bad_schema_tool.spec = bad_schema
    with pytest.raises(ValueError, match="JSON-compatible"):
        app.register_agent(
            AgentSpec(name="bad_tool_schema", model="fake-model"),
            tools=[bad_schema_tool],
        )

    class BadScalarString(str):
        def __deepcopy__(self, memo):
            raise RuntimeError("tool schema scalar deepcopy should not run")

    bad_scalar_schema_tool = _RegistryTool()
    bad_scalar_schema = ToolSpec.model_construct(
        name="bad_scalar_schema",
        description="",
    )
    object.__setattr__(
        bad_scalar_schema,
        "_input_schema",
        {"bad": BadScalarString("value")},
    )
    bad_scalar_schema_tool.spec = bad_scalar_schema
    with pytest.raises(ValueError, match="JSON-compatible"):
        app.register_agent(
            AgentSpec(name="bad_tool_scalar_schema", model="fake-model"),
            tools=[bad_scalar_schema_tool],
        )

    with pytest.raises(TypeError, match="Agent tools"):
        app.register_agent(
            AgentSpec(name="tools_false", model="fake-model"),
            tools=False,  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="Agent tools"):
        app.register_agent(
            AgentSpec(name="tools_zero", model="fake-model"),
            tools=0,  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="Agent tools"):
        app.register_agent(
            AgentSpec(name="tools_empty_string", model="fake-model"),
            tools="",  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="context_policy"):
        app.register_agent(
            AgentSpec(name="bad_context_policy", model="fake-model"),
            context_policy=object(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="tool_policy"):
        app.register_agent(
            AgentSpec(name="bad_tool_policy", model="fake-model"),
            tool_policy=object(),  # type: ignore[arg-type]
        )


def test_cayu_app_rejects_blank_agent_lookup_name() -> None:
    app = CayuApp()
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(ValueError, match="agent.name"):
        app.get_agent("")

    with pytest.raises(ValueError, match="agent.name"):
        app.get_agent(" ")


def test_cayu_app_rejects_blank_provider_name() -> None:
    app = CayuApp()

    with pytest.raises(ValueError, match="provider.name"):
        app.register_provider(ScriptedModelProvider([], name=" "))


def test_cayu_app_rejects_invalid_provider_registration_inputs() -> None:
    class ProviderLike:
        name = "fake_like"

    class BadString(str):
        def strip(self):
            raise RuntimeError("strip should not run")

    app = CayuApp()

    with pytest.raises(TypeError, match="ModelProvider"):
        app.register_provider(ProviderLike())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="bool"):
        app.register_provider(
            ScriptedModelProvider([]),
            default="false",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="must be a string"):
        app.register_provider(ScriptedModelProvider([], name=BadString("bad_provider")))


def test_cayu_app_rejects_blank_provider_lookup_name() -> None:
    app = CayuApp()
    app.register_provider(ScriptedModelProvider([]), default=True)

    with pytest.raises(ValueError, match="provider.name"):
        app.get_provider("")

    with pytest.raises(ValueError, match="provider.name"):
        app.get_provider(" ")


@pytest.mark.parametrize("first_kind", ["concrete", "factory"])
@pytest.mark.parametrize("conflicting_kind", ["concrete", "factory"])
def test_cayu_app_rejects_conflicting_artifact_store_ids_atomically(
    first_kind: str,
    conflicting_kind: str,
    tmp_path,
) -> None:
    app = CayuApp()
    first_store = LocalArtifactStore(tmp_path / "first", store_id="shared-id")
    conflicting_store = LocalArtifactStore(tmp_path / "second", store_id="shared-id")

    def register(
        kind: str,
        name: str,
        store: LocalArtifactStore,
        *,
        default: bool,
    ) -> _UncalledEnvironmentFactory | None:
        if kind == "concrete":
            app.register_environment(
                Environment(EnvironmentSpec(name=name), artifact_store=store),
                default=default,
            )
            return None
        factory = _UncalledEnvironmentFactory()
        app.register_environment_factory(
            EnvironmentSpec(name=name),
            factory,
            artifact_store=store,
            default=default,
        )
        return factory

    first_factory = register(first_kind, "first", first_store, default=True)
    expected_fingerprint = f"sha256:{hashlib.sha256(b'shared-id').hexdigest()}"
    assert app.artifact_store_registration_fingerprints(limit=64) == (
        (expected_fingerprint,),
        1,
    )

    with pytest.raises(ValueError, match="different registered store: shared-id"):
        register(conflicting_kind, "conflicting", conflicting_store, default=True)

    assert app.list_environments() == ("first",)
    assert app.has_registered_artifact_store() is True
    assert app.artifact_store_registration_fingerprints(limit=64) == (
        (expected_fingerprint,),
        1,
    )
    if first_kind == "concrete":
        assert app.get_environment().spec.name == "first"
    else:
        assert app.get_environment_factory() is first_factory

    register("concrete", "shared", first_store, default=False)
    assert app.list_environments() == ("first", "shared")
    assert app.artifact_store_registration_fingerprints(limit=64) == (
        (expected_fingerprint,),
        1,
    )


def test_cayu_app_allows_one_artifact_store_instance_in_multiple_environments(tmp_path) -> None:
    app = CayuApp()
    shared_store = LocalArtifactStore(tmp_path / "shared", store_id="shared-id")
    app.register_environment(
        Environment(EnvironmentSpec(name="concrete"), artifact_store=shared_store),
        default=True,
    )
    factory = _UncalledEnvironmentFactory()

    app.register_environment_factory(
        EnvironmentSpec(name="factory"),
        factory,
        artifact_store=shared_store,
    )

    assert app.list_environments() == ("concrete", "factory")
    assert app.has_registered_artifact_store() is True
    assert app.artifact_store_registration_fingerprints(limit=1) == (
        (f"sha256:{hashlib.sha256(b'shared-id').hexdigest()}",),
        1,
    )


def test_cayu_app_bounds_artifact_store_registration_fingerprint_snapshots(tmp_path) -> None:
    app = CayuApp()
    for index in range(3):
        app.register_environment(
            Environment(
                EnvironmentSpec(name=f"environment-{index}"),
                artifact_store=LocalArtifactStore(
                    tmp_path / f"store-{index}",
                    store_id=f"store-{index}",
                ),
            )
        )

    fingerprints, total_count = app.artifact_store_registration_fingerprints(limit=2)

    assert fingerprints == tuple(
        f"sha256:{hashlib.sha256(f'store-{index}'.encode()).hexdigest()}" for index in range(2)
    )
    assert total_count == 3
    with pytest.raises(TypeError, match="must be an integer"):
        app.artifact_store_registration_fingerprints(limit=True)
    with pytest.raises(ValueError, match="must be positive"):
        app.artifact_store_registration_fingerprints(limit=0)


def test_cayu_app_rejects_invalid_environment_lookup_name() -> None:
    app = CayuApp()
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)

    with pytest.raises(ValueError, match="environment.name"):
        app.get_environment("")

    with pytest.raises(ValueError, match="environment.name"):
        app.get_environment(" ")


def test_cayu_app_registers_and_selects_default_environment_factory() -> None:
    factory = _UncalledEnvironmentFactory()
    app = CayuApp()
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic", metadata={"kind": "registration"}),
        factory,
        default=True,
    )

    assert app.get_environment_factory() is factory
    assert app.get_environment_factory("dynamic") is factory
    with pytest.raises(RuntimeError, match="factory-backed"):
        app.get_environment()


def test_cayu_app_rejects_invalid_environment_factory_registration_inputs() -> None:
    class FactoryLike:
        pass

    class EnvironmentSpecSubclass(EnvironmentSpec):
        pass

    factory = _UncalledEnvironmentFactory()
    app = CayuApp()

    with pytest.raises(TypeError, match="EnvironmentSpec"):
        app.register_environment_factory(
            EnvironmentSpecSubclass(name="dynamic"),
            factory,
        )

    with pytest.raises(TypeError, match="EnvironmentFactory"):
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            FactoryLike(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="bool"):
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default="true",  # type: ignore[arg-type]
        )


def test_cayu_app_rejects_duplicate_environment_and_factory_names() -> None:
    factory = _UncalledEnvironmentFactory()
    app = CayuApp()
    app.register_environment(Environment(EnvironmentSpec(name="local")))

    with pytest.raises(ValueError, match="Environment already registered"):
        app.register_environment_factory(EnvironmentSpec(name="local"), factory)

    app.register_environment_factory(EnvironmentSpec(name="dynamic"), factory)
    with pytest.raises(ValueError, match="Environment already registered"):
        app.register_environment(Environment(EnvironmentSpec(name="dynamic")))


def test_cayu_app_rejects_environment_factory_lookup_for_static_environment() -> None:
    app = CayuApp()
    app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)

    with pytest.raises(RuntimeError, match="not factory-backed"):
        app.get_environment_factory()


def test_cayu_app_rejects_invalid_environment_factory_lookup_name() -> None:
    app = CayuApp()

    with pytest.raises(ValueError, match="environment.name"):
        app.get_environment_factory("")

    with pytest.raises(ValueError, match="environment.name"):
        app.get_environment_factory(" ")


def test_cayu_app_isolates_registered_environment_shell(tmp_path) -> None:
    original_root = tmp_path / "original"
    mutated_root = tmp_path / "mutated"
    returned_root = tmp_path / "returned"
    original_root.mkdir()
    mutated_root.mkdir()
    returned_root.mkdir()
    original_workspace = LocalWorkspace(original_root, workspace_id="workspace_original")
    environment = Environment(
        EnvironmentSpec(name="local", metadata={"kind": "dev"}),
        workspace=original_workspace,
    )
    app = CayuApp()

    app.register_environment(environment, default=True)

    environment.spec = EnvironmentSpec(name="mutated", metadata={"kind": "mutated"})
    environment.workspace = LocalWorkspace(mutated_root, workspace_id="workspace_mutated")

    registered = app.get_environment()
    registered.spec.metadata["kind"] = "returned"
    registered.environment.workspace = LocalWorkspace(
        returned_root,
        workspace_id="workspace_returned",
    )

    registered_again = app.get_environment()

    assert registered_again.spec.name == "local"
    assert registered_again.spec.metadata == {"kind": "dev"}
    assert registered_again.environment.workspace is original_workspace


def test_cayu_app_rejects_invalid_environment_registration_inputs() -> None:
    class EnvironmentLike:
        spec = EnvironmentSpec(name="fake")

    class EnvironmentSubclass(Environment):
        pass

    class BadString(str):
        def strip(self):
            raise RuntimeError("strip should not run")

    class BadMetadata(dict):
        def items(self):
            raise RuntimeError("environment metadata traversal should not run")

    app = CayuApp()

    with pytest.raises(TypeError, match="Environment"):
        app.register_environment(EnvironmentLike())  # type: ignore[arg-type]

    # Environment subclasses are advertised extension points and are accepted.
    app.register_environment(EnvironmentSubclass(EnvironmentSpec(name="subclass")))

    with pytest.raises(TypeError, match="bool"):
        app.register_environment(
            Environment(EnvironmentSpec(name="local")),
            default="false",  # type: ignore[arg-type]
        )

    bad_name_environment = Environment(EnvironmentSpec(name="bad_name"))
    bad_name_environment.spec = EnvironmentSpec.model_construct(
        name=BadString("bad"),
        metadata={},
    )
    with pytest.raises(ValueError, match="must be a string"):
        app.register_environment(bad_name_environment)

    bad_metadata_environment = Environment(EnvironmentSpec(name="bad_metadata"))
    bad_metadata_environment.spec = EnvironmentSpec.model_construct(
        name="bad_metadata",
        metadata=BadMetadata({"bad": "value"}),
    )
    with pytest.raises(ValueError, match="JSON-compatible"):
        app.register_environment(bad_metadata_environment)

    with pytest.raises(ValueError, match="Environment already registered"):
        app.register_environment(Environment(EnvironmentSpec(name="local")))
        app.register_environment(Environment(EnvironmentSpec(name="local")))


def test_cayu_app_isolates_registered_agent_state() -> None:
    app = CayuApp()
    spec = AgentSpec(name="assistant", model="fake-model")
    tool = _RegistryTool()

    app.register_agent(
        spec,
        tools=[tool],
        context_policy=DefaultContextPolicy(),
    )
    spec.model = "mutated"

    registered = app.get_agent("assistant")
    registered.tools["other"] = tool

    assert registered.spec.model == "fake-model"
    assert not hasattr(registered, "context_policy")
    assert app.get_agent("assistant").tools.keys() == {"registry_tool"}


def test_cayu_app_isolates_returned_registered_tool_declarations() -> None:
    app = CayuApp()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_RegistryTool()],
    )

    registered = app.get_agent("assistant")
    registered.tools["registry_tool"].schema["properties"]["text"]["type"] = "integer"

    assert (
        app.get_agent("assistant").tools["registry_tool"].schema == _RegistryTool.spec.input_schema
    )
