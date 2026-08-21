from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from typing import Any, cast

import pytest
from pydantic import ValidationError

import cayu
import cayu.runtime.tool_exposure as exposure_contracts
from cayu import (
    ALL_REGISTERED_TOOLS_PROFILE_ID,
    TOOL_EXPOSURE_METADATA_MAX_BYTES,
    AgentSpec,
    AllRegisteredToolsExposurePolicy,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    RegisteredToolCapability,
    SearchTextTool,
    StaticToolExposurePolicy,
    ToolCapabilityCeiling,
    ToolEffect,
    ToolExposureDecision,
    ToolExposurePolicy,
    ToolExposurePolicyRequest,
    resolve_tool_capability_ceiling,
    resolve_tool_exposure,
)
from cayu.runtime import _execution_profile_admission as execution_profile_admission


def _capability(
    name: str,
    *,
    description: str = "",
    schema: dict[str, Any] | None = None,
    effect: ToolEffect = ToolEffect.NONE,
) -> RegisteredToolCapability:
    return RegisteredToolCapability(
        name=name,
        description=description,
        input_schema=({"type": "object", "properties": {}} if schema is None else schema),
        effect=effect,
    )


def _request(
    *tools: RegisteredToolCapability,
    ceiling: tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolExposurePolicyRequest:
    return ToolExposurePolicyRequest(
        session_id="session-1",
        agent_name="assistant",
        provider_name="provider",
        model="model",
        step=1,
        transcript_cursor=3,
        registered_tools=tools,
        capability_ceiling=(tuple(tool.name for tool in tools) if ceiling is None else ceiling),
        metadata={} if metadata is None else metadata,
    )


def test_registered_tool_capability_owns_deeply_immutable_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string", "enum": ["original"]}},
    }
    capability = _capability("remember", schema=schema)
    original_fingerprint = capability.definition_fingerprint

    schema["properties"]["value"]["enum"].append("mutated")

    assert capability.input_schema_copy() == {
        "type": "object",
        "properties": {"value": {"type": "string", "enum": ["original"]}},
    }
    assert capability.definition_fingerprint == original_fingerprint
    with pytest.raises(TypeError, match="cannot be mutated"):
        cast("dict[str, Any]", capability.input_schema)["new"] = True

    dumped = capability.model_dump(mode="json")
    assert dumped["schema_fingerprint"] == capability.schema_fingerprint
    assert dumped["definition_fingerprint"] == capability.definition_fingerprint
    dumped["schema_fingerprint"] = "0" * 64
    dumped["definition_fingerprint"] = "0" * 64
    with pytest.raises(ValidationError, match="schema_fingerprint does not match"):
        RegisteredToolCapability.model_validate(dumped)
    dumped = capability.model_dump(mode="json")
    restored = RegisteredToolCapability.model_validate(dumped)
    assert restored == capability
    assert restored.schema_fingerprint == capability.schema_fingerprint
    assert restored.definition_fingerprint == capability.definition_fingerprint


def test_capability_fingerprints_exact_contract_changes() -> None:
    original = _capability("remember", description="Save one fact.")
    changed_description = _capability("remember", description="Save a fact.")
    changed_schema = _capability(
        "remember",
        description="Save one fact.",
        schema={"type": "object", "required": ["fact"]},
    )
    changed_effect = _capability(
        "remember",
        description="Save one fact.",
        effect=ToolEffect.EXTERNAL,
    )

    assert original.schema_fingerprint != changed_schema.schema_fingerprint
    assert (
        len(
            {
                original.definition_fingerprint,
                changed_description.definition_fingerprint,
                changed_schema.definition_fingerprint,
                changed_effect.definition_fingerprint,
            }
        )
        == 4
    )


def test_policy_request_is_bounded_owned_and_ceiling_ordered() -> None:
    first = _capability("first")
    second = _capability("second")
    metadata = {"phase": {"name": "review"}}
    request = _request(first, second, ceiling=("second",), metadata=metadata)

    metadata["phase"]["name"] = "mutated"

    assert request.eligible_tools == (second,)
    assert request.metadata == {"phase": {"name": "review"}}
    assert ToolExposurePolicyRequest.model_validate(request.model_dump(mode="json")) == request
    with pytest.raises(TypeError, match="cannot be mutated"):
        cast("dict[str, Any]", request.metadata)["phase"] = {}

    with pytest.raises(ValidationError, match="preserve registered tool order"):
        _request(first, second, ceiling=("second", "first"))
    with pytest.raises(ValidationError, match="unregistered tool name"):
        _request(first, ceiling=("missing",))
    with pytest.raises(ValidationError, match="unique tool names"):
        _request(first, first)


def test_tool_capability_ceiling_is_versioned_owned_and_fingerprint_bound() -> None:
    names = ["alpha", "beta"]
    ceiling = ToolCapabilityCeiling(tool_names=names)
    names.append("gamma")

    assert ceiling.schema_version == 1
    assert ceiling.tool_names == ("alpha", "beta")
    assert len(ceiling.fingerprint) == 64
    assert cayu.ToolCapabilityCeiling is ToolCapabilityCeiling
    assert ToolCapabilityCeiling.model_validate(ceiling.model_dump(mode="json")) == ceiling

    dumped = ceiling.model_dump(mode="json")
    dumped["tool_names"] = ["alpha"]
    with pytest.raises(ValidationError, match="fingerprint does not match"):
        ToolCapabilityCeiling.model_validate(dumped)
    with pytest.raises(ValidationError, match="unique tool names"):
        ToolCapabilityCeiling(tool_names=("alpha", "alpha"))


def test_tool_capability_ceiling_copy_suppresses_hostile_serializer_diagnostics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "tool-ceiling-hostile-repr-canary"

    class SecretRenderingValue:
        def __repr__(self) -> str:
            return secret

    ceiling = ToolCapabilityCeiling(tool_names=("alpha",))
    object.__setattr__(ceiling, "tool_names", SecretRenderingValue())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises((TypeError, ValidationError)) as exc_info:
            exposure_contracts.copy_tool_capability_ceiling(ceiling)

    captured = capsys.readouterr()
    assert caught == []
    assert secret not in repr((exc_info.value, captured.out, captured.err))


def test_tool_capability_ceiling_resolution_canonicalizes_and_never_widens() -> None:
    alpha = _capability("alpha")
    beta = _capability("beta")
    gamma = _capability("gamma")

    initial = resolve_tool_capability_ceiling(None, (alpha, beta))
    selected = resolve_tool_capability_ceiling(
        ToolCapabilityCeiling(tool_names=("beta", "alpha")),
        (alpha, beta),
    )
    reconstructed = resolve_tool_capability_ceiling(
        None,
        (alpha, beta, gamma),
        maximum=initial,
    )
    removed = resolve_tool_capability_ceiling(
        None,
        (beta, gamma),
        maximum=initial,
    )

    assert initial.tool_names == ("alpha", "beta")
    assert selected.tool_names == ("alpha", "beta")
    assert reconstructed.tool_names == ("alpha", "beta")
    assert removed.tool_names == ("beta",)

    with pytest.raises(ValueError, match="never widened"):
        resolve_tool_capability_ceiling(
            ToolCapabilityCeiling(tool_names=("alpha", "gamma")),
            (alpha, beta, gamma),
            maximum=initial,
        )
    with pytest.raises(ValueError, match="unregistered tool"):
        resolve_tool_capability_ceiling(
            ToolCapabilityCeiling(tool_names=("missing",)),
            (alpha, beta),
        )


def test_initial_ceiling_defers_oversized_catalog_failure_to_the_model_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _capability("alpha")
    monkeypatch.setattr(exposure_contracts, "TOOL_EXPOSURE_MAX_CATALOG_BYTES", 1)

    assert (
        exposure_contracts._resolve_initial_tool_capability_ceiling(
            None,
            (capability,),
        )
        is None
    )
    with pytest.raises(ValueError, match="canonical JSON bytes"):
        exposure_contracts._resolve_initial_tool_capability_ceiling(
            ToolCapabilityCeiling(tool_names=()),
            (capability,),
        )


def test_policy_request_bounds_the_complete_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exposure_contracts, "TOOL_EXPOSURE_MAX_CATALOG_BYTES", 128)

    with pytest.raises(ValidationError, match="canonical JSON bytes in total"):
        _request(_capability("large", description="x" * 128))


def test_policy_request_catalog_byte_limit_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _capability("first")
    second = _capability("second")
    exact_size = 2 + first._canonical_size_bytes + 1 + second._canonical_size_bytes
    monkeypatch.setattr(
        exposure_contracts,
        "TOOL_EXPOSURE_MAX_CATALOG_BYTES",
        exact_size,
    )

    assert _request(first, second).registered_tools == (first, second)

    monkeypatch.setattr(
        exposure_contracts,
        "TOOL_EXPOSURE_MAX_CATALOG_BYTES",
        exact_size - 1,
    )
    with pytest.raises(ValidationError, match="canonical JSON bytes in total"):
        _request(first, second)


def test_policy_request_stops_copying_after_catalog_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exposure_contracts, "TOOL_EXPOSURE_MAX_CATALOG_BYTES", 1)
    consumed: list[str] = []

    def capabilities() -> Iterator[RegisteredToolCapability]:
        consumed.append("first")
        yield _capability("first")
        consumed.append("past-limit")
        raise AssertionError("catalog iteration continued after exceeding its byte limit")

    with pytest.raises(ValidationError, match="canonical JSON bytes in total"):
        ToolExposurePolicyRequest.model_validate(
            {
                "session_id": "session-1",
                "agent_name": "assistant",
                "provider_name": "provider",
                "model": "model",
                "step": 1,
                "registered_tools": capabilities(),
                "capability_ceiling": (),
            }
        )

    assert consumed == ["first"]


def test_policy_request_revalidates_constructed_capability_instances() -> None:
    malformed = RegisteredToolCapability.model_construct(
        name=" ",
        description="",
        input_schema={},
        parallel_safe=True,
        effect=ToolEffect.NONE,
        publishes_arguments=True,
        workspace_mutation=False,
    )
    cast("Any", malformed)._validated = True

    with pytest.raises(ValidationError, match="name"):
        _request(malformed)

    wrong_strict_type = RegisteredToolCapability.model_construct(
        name="wrong-strict-type",
        description="",
        input_schema={},
        parallel_safe="yes",
        effect=ToolEffect.NONE,
        publishes_arguments=True,
        workspace_mutation=False,
    )
    cast("Any", wrong_strict_type)._validated = True

    with pytest.raises(ValidationError, match="parallel_safe"):
        _request(wrong_strict_type)


def test_policy_request_strips_capability_subclass_state() -> None:
    class CapabilityWithLiveHandle(RegisteredToolCapability):
        live_handle: Any

    capability = CapabilityWithLiveHandle(name="safe", live_handle=lambda: None)

    request = _request(capability)

    assert type(request.registered_tools[0]) is RegisteredToolCapability
    assert not hasattr(request.registered_tools[0], "live_handle")


def test_policy_request_detaches_exact_capabilities_and_recomputes_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _capability("safe")
    cast("Any", capability)._live_handle = lambda: None
    request = _request(capability)

    assert request.registered_tools[0] is not capability
    assert not hasattr(request.registered_tools[0], "_live_handle")

    large = _capability("large", description="x" * 128)
    large._canonical_size_bytes = 0
    monkeypatch.setattr(exposure_contracts, "TOOL_EXPOSURE_MAX_CATALOG_BYTES", 128)

    with pytest.raises(ValidationError, match="canonical JSON bytes in total"):
        _request(large)


def test_policy_metadata_is_json_safe_and_bounded() -> None:
    with pytest.raises(ValidationError, match="canonical JSON bytes"):
        ToolExposureDecision(
            profile_id="review",
            metadata={"value": "x" * TOOL_EXPOSURE_METADATA_MAX_BYTES},
        )
    with pytest.raises(ValidationError, match="JSON-compatible"):
        ToolExposureDecision(profile_id="review", metadata={"value": object()})


def test_policy_decision_does_not_claim_the_exposure_evidence_name() -> None:
    assert cayu.ToolExposureDecision is ToolExposureDecision
    assert not hasattr(cayu, "ToolExposure")


def test_all_registered_policy_exposes_only_the_effective_ceiling() -> None:
    first = _capability("first")
    second = _capability("second")
    request = _request(first, second, ceiling=("first",))

    exposure = resolve_tool_exposure(AllRegisteredToolsExposurePolicy(), request)

    assert exposure.profile_id == ALL_REGISTERED_TOOLS_PROFILE_ID
    assert exposure.tool_names == ("first",)
    assert exposure.registered_count == 2
    assert exposure.ceiling_count == 1


def test_static_policy_supports_empty_and_canonical_ordered_profiles() -> None:
    first = _capability("first")
    second = _capability("second")
    request = _request(first, second)

    empty = resolve_tool_exposure(
        StaticToolExposurePolicy(profile_id="tool-free", tools=()),
        request,
    )
    reversed_policy = StaticToolExposurePolicy(
        profile_id="review",
        tools=("second", "first"),
    )
    canonical_policy = StaticToolExposurePolicy(
        profile_id="review",
        tools=("first", "second"),
    )
    reversed_decision = resolve_tool_exposure(reversed_policy, request)

    assert empty.tool_names == ()
    assert reversed_policy.tools == canonical_policy.tools == ("first", "second")
    assert reversed_decision.tool_names == ("first", "second")
    assert (
        reversed_decision.fingerprint
        == resolve_tool_exposure(canonical_policy, request).fingerprint
    )


def test_resolved_exposure_fingerprint_binds_profile_and_definitions() -> None:
    original_request = _request(_capability("remember", description="Save a fact."))
    changed_request = _request(_capability("remember", description="Save one fact."))

    original = resolve_tool_exposure(
        StaticToolExposurePolicy(profile_id="review", tools=("remember",)),
        original_request,
    )
    changed_definition = resolve_tool_exposure(
        StaticToolExposurePolicy(profile_id="review", tools=("remember",)),
        changed_request,
    )
    changed_profile = resolve_tool_exposure(
        StaticToolExposurePolicy(profile_id="review-v2", tools=("remember",)),
        original_request,
    )

    assert original.fingerprint != changed_definition.fingerprint
    assert original.fingerprint != changed_profile.fingerprint
    dumped = original.model_dump(mode="json")
    assert dumped["tools"][0]["name"] == "remember"
    assert dumped["fingerprint"] == original.fingerprint
    dumped["fingerprint"] = "0" * 64
    with pytest.raises(ValidationError, match="fingerprint does not match"):
        type(original).model_validate(dumped)
    restored = type(original).model_validate(original.model_dump(mode="json"))
    assert restored == original
    assert restored.fingerprint == original.fingerprint


class _ReturningPolicy(ToolExposurePolicy):
    def __init__(self, exposure: object) -> None:
        self.exposure = exposure

    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        return cast("ToolExposureDecision", self.exposure)


def test_resolution_rejects_unknown_and_out_of_ceiling_tools() -> None:
    first = _capability("first")
    second = _capability("second")
    request = _request(first, second, ceiling=("first",))

    with pytest.raises(ValueError, match="unregistered"):
        resolve_tool_exposure(
            _ReturningPolicy(ToolExposureDecision(profile_id="bad", tool_names=("missing",))),
            request,
        )
    with pytest.raises(ValueError, match="outside the capability ceiling"):
        resolve_tool_exposure(
            _ReturningPolicy(ToolExposureDecision(profile_id="bad", tool_names=("second",))),
            request,
        )


def test_resolution_defensively_revalidates_policy_results() -> None:
    request = _request(_capability("first"))
    malformed = ToolExposureDecision.model_construct(
        profile_id=" ",
        tool_names=("first",),
        metadata={},
    )

    with pytest.raises(ValidationError, match="profile_id"):
        resolve_tool_exposure(_ReturningPolicy(malformed), request)
    with pytest.raises(TypeError, match="must return ToolExposureDecision"):
        resolve_tool_exposure(_ReturningPolicy(object()), request)


class _IdentityChangingPolicy(ToolExposurePolicy):
    def __init__(self) -> None:
        self.version = "1"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="phase-selector",
            behavior_version=self.version,
            implementation_version=self.version,
        )

    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        self.version = "2"
        return ToolExposureDecision(profile_id="review", tool_names=())


def test_resolution_rejects_policy_identity_mutation_during_selection() -> None:
    with pytest.raises(RuntimeError, match="identity changed during selection"):
        resolve_tool_exposure(
            _IdentityChangingPolicy(),
            _request(_capability("first")),
        )


class _RequestMutatingPolicy(ToolExposurePolicy):
    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        object.__setattr__(request, "capability_ceiling", ("first", "second"))
        return ToolExposureDecision(profile_id="widened", tool_names=("second",))


def test_resolution_rejects_policy_request_mutation_without_widening_ceiling() -> None:
    first = _capability("first")
    second = _capability("second")
    request = _request(first, second, ceiling=("first",))

    with pytest.raises(RuntimeError, match="mutated its request"):
        resolve_tool_exposure(_RequestMutatingPolicy(), request)

    assert request.capability_ceiling == ("first",)


class _CapabilityMutatingPolicy(ToolExposurePolicy):
    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        object.__setattr__(request.registered_tools[0], "description", "changed")
        return ToolExposureDecision(profile_id="changed", tool_names=("first",))


def test_resolution_rejects_selected_capability_mutation() -> None:
    with pytest.raises(RuntimeError, match="mutated a capability"):
        resolve_tool_exposure(
            _CapabilityMutatingPolicy(),
            _request(_capability("first")),
        )


class _PrivateStateInspectingPolicy(ToolExposurePolicy):
    saw_private_state = False

    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        self.saw_private_state = hasattr(request, "_live_handle")
        return ToolExposureDecision(profile_id="detached", tool_names=())


def test_resolution_detaches_request_private_state() -> None:
    request = _request(_capability("first"))
    cast("Any", request)._live_handle = lambda: None
    policy = _PrivateStateInspectingPolicy()

    resolve_tool_exposure(policy, request)

    assert not policy.saw_private_state


def test_execution_profile_reuses_registration_time_capability_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=(SearchTextTool(),),
    )
    registered_agent = app._agents["assistant"]
    assert tuple(tool.name for tool in registered_agent.tool_capabilities) == ("search_text",)

    def unexpected_capability_hash(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise AssertionError("execution-profile resolution rebuilt a capability")

    monkeypatch.setattr(
        exposure_contracts,
        "_sha256_durable_json",
        unexpected_capability_hash,
    )

    execution_profile_admission.resolve_execution_profile_identity(
        registered_agent=registered_agent,
        provider_name="fake",
        model="fake-model",
        durable_system_prompt=None,
        runtime_name="cayu",
        runtime_version="test",
        redactor=app._secret_redactor,
        process_identity=app._execution_profile_process_identity,
    )


def test_execution_profile_preserves_default_identity_and_binds_static_exposure() -> None:
    def profile(app: CayuApp):
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

    default_app = CayuApp(enable_logging=False)
    default_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=(SearchTextTool(),),
    )
    explicit_default_app = CayuApp(enable_logging=False)
    explicit_default_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=(SearchTextTool(),),
        tool_exposure_policy=AllRegisteredToolsExposurePolicy(),
    )
    static_app = CayuApp(enable_logging=False)
    static_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=(SearchTextTool(),),
        tool_exposure_policy=StaticToolExposurePolicy(
            profile_id="tool-free",
            tools=(),
        ),
    )

    assert profile(default_app).fingerprint == profile(explicit_default_app).fingerprint
    assert profile(default_app).fingerprint != profile(static_app).fingerprint


def test_policy_input_contains_no_live_execution_objects() -> None:
    request = _request(_capability("first"))
    capability_fields = set(RegisteredToolCapability.model_fields)

    assert capability_fields == {
        "name",
        "description",
        "input_schema",
        "parallel_safe",
        "effect",
        "publishes_arguments",
        "workspace_mutation",
        "schema_fingerprint",
        "definition_fingerprint",
    }
    assert isinstance(request.metadata, Mapping)
    assert not hasattr(request.registered_tools[0], "tool")
    assert not hasattr(request.registered_tools[0], "run")
