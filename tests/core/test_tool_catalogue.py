from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    Tool,
    ToolCatalogSnapshot,
    ToolDescriptor,
    ToolDescriptorProvenance,
    ToolEffect,
    ToolSpec,
    build_tool_catalog_snapshot,
    build_tool_descriptor,
    copy_tool_catalog_snapshot,
    tool_catalogue_descriptors_within_ceiling,
    validate_application_tool_name,
)
from cayu.runtime import _execution_profile_admission as execution_profile_admission
from cayu.runtime.execution_profiles import ExecutionProfileComponentClass
from cayu.runtime.tool_catalogue import mcp_source_tool_fingerprint


def _descriptor(
    name: str,
    *,
    description: str = "One tool.",
    input_schema: dict[str, Any] | None = None,
    effect: ToolEffect = ToolEffect.NONE,
    provenance: ToolDescriptorProvenance | None = None,
) -> ToolDescriptor:
    return build_tool_descriptor(
        name=name,
        description=description,
        input_schema=(
            {"type": "object", "properties": {"value": {"type": "string"}}}
            if input_schema is None
            else input_schema
        ),
        parallel_safe=True,
        effect=effect,
        publishes_arguments=True,
        workspace_mutation=False,
        provenance=provenance,
    )


class _DeclaredTool(Tool):
    def __init__(
        self,
        name: str,
        *,
        implementation_version: str = "1",
    ) -> None:
        self.spec = ToolSpec(
            name=name,
            description="One tool.",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.NONE,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name=f"tests:{name}",
                behavior_version="1",
                implementation_version=implementation_version,
            ),
        )
        super().__init__()

    async def run(self, ctx, args):
        del ctx, args
        raise AssertionError("Catalogue tests must not execute tools.")


def _profile(app: CayuApp):
    return execution_profile_admission.resolve_execution_profile_identity(
        registered_agent=app._agents["assistant"],
        provider_name="fake",
        model="fake-model",
        durable_system_prompt=None,
        runtime_name="cayu",
        runtime_version="test",
        redactor=app._secret_redactor,
        process_identity=app._execution_profile_process_identity,
    )


def test_descriptor_owns_schema_and_revalidates_derived_identity() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string", "enum": ["original"]}},
    }
    descriptor = _descriptor("remember knowledge", input_schema=schema)
    version = descriptor.version

    schema["properties"]["value"]["enum"].append("mutated")

    assert descriptor.tool_id == "cayu:remember%20knowledge"
    assert descriptor.input_schema_copy()["properties"]["value"]["enum"] == ["original"]
    assert descriptor.version == version
    assert descriptor.execution_profile_material() == {
        "tool_id": descriptor.tool_id,
        "descriptor_version": descriptor.version,
    }
    with pytest.raises(TypeError, match="cannot be mutated"):
        cast("dict[str, Any]", descriptor.input_schema)["new"] = True

    dumped = descriptor.model_dump(mode="json")
    dumped["version"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="version does not match"):
        ToolDescriptor.model_validate(dumped)


def test_descriptor_version_covers_callable_contract_but_not_implementation_identity() -> None:
    base = _descriptor("remember")
    changed_description = _descriptor("remember", description="Another tool.")
    changed_schema = _descriptor(
        "remember",
        input_schema={"type": "object", "required": ["fact"]},
    )
    changed_effect = _descriptor("remember", effect=ToolEffect.EXTERNAL)

    assert (
        len(
            {
                base.version,
                changed_description.version,
                changed_schema.version,
                changed_effect.version,
            }
        )
        == 4
    )

    first = CayuApp(enable_logging=False)
    first.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=(_DeclaredTool("remember", implementation_version="1"),),
    )
    second = CayuApp(enable_logging=False)
    second.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=(_DeclaredTool("remember", implementation_version="2"),),
    )

    assert first._agents["assistant"].tool_catalogue == second._agents["assistant"].tool_catalogue
    assert _profile(first).component(ExecutionProfileComponentClass.DIRECT_TOOLS) == _profile(
        second
    ).component(ExecutionProfileComponentClass.DIRECT_TOOLS)
    assert _profile(first).component(
        ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS
    ) != _profile(second).component(ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS)


def test_mcp_descriptor_identity_is_bounded_and_does_not_retain_source_name() -> None:
    source_name = "private-tenant-sentinel-" + "x" * 10_000
    provenance = ToolDescriptorProvenance(
        kind="mcp",
        source_id="sha256:" + "1" * 64,
        source_tool_fingerprint=mcp_source_tool_fingerprint(source_name),
        source_contract_fingerprint="sha256:" + "2" * 64,
    )
    descriptor = _descriptor("mcp__server__tool", provenance=provenance)

    assert descriptor.tool_id == f"mcp:{'1' * 64}:{provenance.source_tool_fingerprint[7:]}"
    assert source_name not in descriptor.model_dump_json()
    assert "private-tenant-sentinel" not in descriptor.model_dump_json()

    malformed = provenance.model_dump(mode="json")
    malformed["source_id"] = "tenant/raw"
    with pytest.raises(ValidationError, match="authoritative SHA-256"):
        ToolDescriptorProvenance.model_validate(malformed)


def test_targeted_delivery_mode_is_bound_without_changing_catalogued_tools() -> None:
    disabled = CayuApp(enable_logging=False)
    disabled.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=(_DeclaredTool("remember"),),
    )
    enabled = CayuApp(enable_logging=False)
    enabled.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=(_DeclaredTool("remember"),),
        targeted_tool_mode="call_tool",
    )

    disabled_profile = _profile(disabled)
    enabled_profile = _profile(enabled)
    assert disabled_profile.component(
        ExecutionProfileComponentClass.DIRECT_TOOLS
    ) == enabled_profile.component(ExecutionProfileComponentClass.DIRECT_TOOLS)
    assert disabled_profile.component(
        ExecutionProfileComponentClass.EXECUTION_POLICIES
    ) != enabled_profile.component(ExecutionProfileComponentClass.EXECUTION_POLICIES)
    assert disabled_profile.fingerprint != enabled_profile.fingerprint


def test_catalogue_revision_is_order_and_json_object_order_independent() -> None:
    alpha = _descriptor(
        "alpha",
        input_schema={
            "properties": {"b": {"type": "number"}, "a": {"type": "string"}},
            "type": "object",
        },
    )
    equivalent_alpha = _descriptor(
        "alpha",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "number"}},
        },
    )
    beta = _descriptor("beta")

    first = build_tool_catalog_snapshot((beta, alpha))
    second = build_tool_catalog_snapshot((equivalent_alpha, beta))

    assert first == second
    assert tuple(item.name for item in first.descriptors) == ("alpha", "beta")
    assert first.descriptor_count == 2
    assert first.revision.startswith("sha256:")
    copied = copy_tool_catalog_snapshot(first)
    assert copied == first
    assert copied.descriptors[0] is not first.descriptors[0]


def test_catalogue_rejects_duplicate_names_noncanonical_order_and_tampering() -> None:
    cayu_descriptor = _descriptor("same")
    mcp_descriptor = _descriptor(
        "same",
        provenance=ToolDescriptorProvenance(
            kind="mcp",
            source_id="sha256:" + "1" * 64,
            source_tool_fingerprint="sha256:" + "2" * 64,
            source_contract_fingerprint="sha256:" + "3" * 64,
        ),
    )
    with pytest.raises(ValidationError, match="unique model-visible names"):
        ToolCatalogSnapshot(
            descriptors=tuple(
                sorted((cayu_descriptor, mcp_descriptor), key=lambda item: item.tool_id)
            ),
            descriptor_count=2,
        )

    alpha = _descriptor("alpha")
    beta = _descriptor("beta")
    with pytest.raises(ValidationError, match="canonical tool_id order"):
        ToolCatalogSnapshot(descriptors=(beta, alpha), descriptor_count=2)

    dumped = build_tool_catalog_snapshot((alpha, beta)).model_dump(mode="json")
    dumped["descriptor_count"] = 1
    with pytest.raises(ValidationError, match="descriptor_count"):
        ToolCatalogSnapshot.model_validate(dumped)
    dumped = build_tool_catalog_snapshot((alpha, beta)).model_dump(mode="json")
    dumped["revision"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="revision does not match"):
        ToolCatalogSnapshot.model_validate(dumped)


def test_ceiling_view_is_canonical_and_fails_closed() -> None:
    snapshot = build_tool_catalog_snapshot((_descriptor("beta"), _descriptor("alpha")))

    selected = tool_catalogue_descriptors_within_ceiling(snapshot, ("beta", "alpha"))

    assert tuple(item.name for item in selected) == ("alpha", "beta")
    by_name = snapshot.descriptor_for_name("alpha")
    assert by_name == snapshot.descriptor_for_id(by_name.tool_id)
    assert by_name is not snapshot.descriptors[0]
    with pytest.raises(KeyError, match="missing"):
        snapshot.descriptor_for_id("cayu:missing")
    with pytest.raises(ValueError, match="unregistered"):
        tool_catalogue_descriptors_within_ceiling(snapshot, ("missing",))
    with pytest.raises(ValueError, match="unique"):
        tool_catalogue_descriptors_within_ceiling(snapshot, ("alpha", "alpha"))


@pytest.mark.parametrize("name", ["call_tool", "search_tools", "__cayu_submit_structured_output"])
def test_framework_tool_names_are_reserved_at_registration(name: str) -> None:
    with pytest.raises(ValueError, match="reserved by the Cayu framework"):
        validate_application_tool_name(name)

    app = CayuApp(enable_logging=False)
    with pytest.raises(ValueError, match="reserved by the Cayu framework"):
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(_DeclaredTool(name),),
        )
    assert "assistant" not in app._agents


def test_registration_uses_one_catalogue_source_without_changing_exposure_order() -> None:
    first = CayuApp(enable_logging=False)
    first.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=(_DeclaredTool("beta"), _DeclaredTool("alpha")),
    )
    second = CayuApp(enable_logging=False)
    second.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=(_DeclaredTool("alpha"), _DeclaredTool("beta")),
    )

    first_agent = first._agents["assistant"]
    second_agent = second._agents["assistant"]
    assert first_agent.tool_catalogue.revision == second_agent.tool_catalogue.revision
    assert tuple(item.name for item in first_agent.tool_catalogue.descriptors) == (
        "alpha",
        "beta",
    )
    assert tuple(item.name for item in first_agent.tool_capabilities) == ("beta", "alpha")
    assert first_agent.all_registered_tool_exposure.tool_names == ("beta", "alpha")

    first_direct = _profile(first).component(ExecutionProfileComponentClass.DIRECT_TOOLS)
    second_direct = _profile(second).component(ExecutionProfileComponentClass.DIRECT_TOOLS)
    assert first_direct != second_direct
