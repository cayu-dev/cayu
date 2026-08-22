from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from cayu import (
    AnthropicProvider,
    CacheBreakpoint,
    CachePolicy,
    FileAttachmentKind,
    Message,
    ModelContextPressureProfile,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    OpenAIProvider,
    OpenAIWebSearch,
    PromptContributionAvailability,
    PromptContributionKind,
    RequestFingerprintAvailability,
    RequestFootprint,
    RequestFootprintConfig,
    RequestVariant,
    StructuredOutputSpec,
    ToolExposure,
    build_prompt_contribution_manifest,
    build_request_footprint,
)
from cayu.artifacts.attachments import RESOLVED_FILE_ATTACHMENTS_OPTION, file_attachment
from cayu.core.messages import FilePart, ProviderStatePart, TextPart, ThinkingPart, ToolCallPart
from cayu.providers.anthropic import build_anthropic_payload
from cayu.providers.openai import build_openai_payload
from cayu.runtime.request_footprints import (
    analyze_request_context_pressure,
    analyze_request_footprint,
)
from cayu.runtime.structured_output import (
    structured_output_spec_payload,
    structured_output_tool_instruction,
    structured_output_tool_spec,
)

_CATALOGUE_REVISION = f"sha256:{'c' * 64}"


def _request(*, tool_description: str = "Inspect the repository") -> ModelRequest:
    return ModelRequest(
        model="model-a",
        messages=[
            Message.text("system", "System café"),
            Message(
                role="user",
                content=(
                    TextPart(text="Read the report"),
                    FilePart(
                        attachment=file_attachment(
                            artifact_id="artifact-secret-id",
                            kind="document",
                            filename="quarterly-secret-name.pdf",
                            content_type="application/pdf",
                            size_bytes=4,
                        )
                    ),
                ),
            ),
            Message(
                role="assistant",
                content=(
                    TextPart(text="I will inspect it."),
                    ToolCallPart(
                        tool_call_id="call-1",
                        tool_name="inspect",
                        arguments={"path": "reports/q1.md", "limit": 10},
                    ),
                ),
            ),
        ],
        tools=[
            {
                "name": "inspect",
                "description": tool_description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer"},
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
            }
        ],
        options={
            "temperature": 0,
            "structured_output": {
                "strategy": "native",
                "schema": {"type": "object"},
            },
            "agent_metadata": {"private_note": "do-not-persist-this"},
            "vendor_private_option": "unknown-option-value",
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "artifact-secret-id": {
                    "artifact_id": "artifact-secret-id",
                    "kind": "document",
                    "filename": "quarterly-secret-name.pdf",
                    "content_type": "application/pdf",
                    "data_base64": "YWJjZA==",
                    "metadata": {"private": "attachment-secret-metadata"},
                }
            },
        },
    )


def _build(
    request: ModelRequest,
    *,
    config: RequestFootprintConfig | None = None,
    measured_provider_options: dict[str, Any] | None = None,
    fingerprint_provider_options: dict[str, Any] | None = None,
) -> RequestFootprint:
    return build_request_footprint(
        request,
        provider_name="provider-a",
        step=2,
        attempt=3,
        max_attempts=4,
        request_variant=RequestVariant.STRUCTURED_OUTPUT_REPAIR,
        observation_id="observation-1",
        model_step_id="mstep_00000000000000000000000000000001",
        model_attempt_id="matt_00000000000000000000000000000002",
        config=config,
        measured_provider_options=measured_provider_options,
        fingerprint_provider_options=fingerprint_provider_options,
        context_pressure_profile=ModelContextPressureProfile(),
        cache_policy=CachePolicy(
            breakpoints=(
                CacheBreakpoint.SYSTEM_PROMPT,
                CacheBreakpoint.TOOL_DEFINITIONS,
                CacheBreakpoint.CONVERSATION_PREFIX,
            ),
            conversation_prefix_strategy="all_but_last",
            ttl="extended",
        ),
    )


@pytest.mark.parametrize("invalid_version", [True, 1.0, "1", 4])
def test_request_footprint_versions_require_supported_exact_integers(
    invalid_version: object,
) -> None:
    footprint_payload = _build(_request()).model_dump(mode="python")
    footprint_payload["schema_version"] = invalid_version
    with pytest.raises(ValidationError, match="schema_version must be integer 1, 2, or 3"):
        RequestFootprint.model_validate(footprint_payload)

    fingerprint_payload = _build(
        _request(),
        config=RequestFootprintConfig(
            fingerprint_key_id="version-key",
            fingerprint_key=SecretStr("v" * 32),
        ),
    ).model_dump(mode="python")
    fingerprint_payload["fingerprints"]["provider_neutral_request"]["canonicalization_version"] = (
        invalid_version
    )
    with pytest.raises(ValidationError, match="canonicalization_version must be the integer 1"):
        RequestFootprint.model_validate(fingerprint_payload)

    manifest = build_prompt_contribution_manifest(
        rendered_system_prompt="system prompt",
        contributions={PromptContributionKind.AGENT_INSTRUCTIONS: ("system prompt",)},
        config=RequestFootprintConfig(),
    )
    assert manifest is not None
    manifest_payload = manifest.model_dump(mode="python")
    manifest_payload["schema_version"] = invalid_version
    with pytest.raises(ValidationError, match="schema_version must be the integer 1"):
        type(manifest).model_validate(manifest_payload)


def test_request_footprint_v2_requires_and_retains_its_governing_profile() -> None:
    fingerprint = "a" * 64
    footprint = build_request_footprint(
        _request(),
        provider_name="fake",
        step=1,
        attempt=1,
        max_attempts=1,
        request_variant=RequestVariant.INITIAL,
        observation_id="profile-linked-footprint",
        model_step_id="mstep_00000000000000000000000000000001",
        model_attempt_id="matt_00000000000000000000000000000001",
        execution_profile_fingerprint=fingerprint,
    )

    assert footprint.schema_version == 2
    assert footprint.execution_profile_fingerprint == fingerprint
    missing_profile = footprint.model_dump(mode="python")
    missing_profile["execution_profile_fingerprint"] = None
    with pytest.raises(ValidationError, match=r"schema v2\+ requires an execution profile"):
        RequestFootprint.model_validate(missing_profile)

    legacy_with_profile = footprint.model_dump(mode="python")
    legacy_with_profile["schema_version"] = 1
    with pytest.raises(ValidationError, match="schema v1 cannot carry an execution profile"):
        RequestFootprint.model_validate(legacy_with_profile)


def test_request_footprint_v3_binds_the_prepared_tool_exposure() -> None:
    execution_profile_fingerprint = "a" * 64
    exposure = ToolExposure(
        execution_profile_fingerprint=execution_profile_fingerprint,
        profile_id="review",
        catalogue_revision=_CATALOGUE_REVISION,
        exposure_fingerprint="b" * 64,
        registered_count=2,
        ceiling_count=1,
        exposed_count=1,
        profile_changed=False,
        step=1,
        provider_name="provider-a",
        model="model-a",
        model_step_id="mstep_00000000000000000000000000000001",
    )
    footprint = build_request_footprint(
        _request(),
        provider_name="provider-a",
        step=1,
        attempt=1,
        max_attempts=1,
        request_variant=RequestVariant.INITIAL,
        observation_id="exposure-linked-footprint",
        model_step_id=exposure.model_step_id,
        model_attempt_id="matt_00000000000000000000000000000001",
        execution_profile_fingerprint=execution_profile_fingerprint,
        tool_exposure=exposure,
    )

    assert footprint.schema_version == 3
    assert footprint.tool_exposure is not None
    assert footprint.tool_exposure.profile_id == "review"
    assert footprint.tool_exposure.exposure_fingerprint == exposure.exposure_fingerprint
    assert footprint.fingerprints.tool_manifest.availability == "unavailable"

    mismatched = exposure.model_copy(update={"exposed_count": 0})
    with pytest.raises(ValueError, match="exposed_count must match"):
        build_request_footprint(
            _request(),
            provider_name="provider-a",
            step=1,
            attempt=1,
            max_attempts=1,
            request_variant=RequestVariant.INITIAL,
            observation_id="mismatched-exposure",
            model_step_id=exposure.model_step_id,
            model_attempt_id="matt_00000000000000000000000000000002",
            execution_profile_fingerprint=execution_profile_fingerprint,
            tool_exposure=mismatched,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"execution_profile_fingerprint": "c" * 64},
            "execution profile must match",
        ),
        ({"provider_name": "provider-b"}, "provider_name must match"),
        ({"model": "model-b"}, "model must match"),
        ({"step": 2}, "step must match"),
        (
            {"model_step_id": "mstep_00000000000000000000000000000002"},
            "model_step_id must match",
        ),
    ],
)
def test_request_footprint_v3_rejects_mismatched_exposure_authority(
    changes: dict[str, object],
    message: str,
) -> None:
    execution_profile_fingerprint = "a" * 64
    exposure = ToolExposure(
        execution_profile_fingerprint=execution_profile_fingerprint,
        profile_id="review",
        catalogue_revision=_CATALOGUE_REVISION,
        exposure_fingerprint="b" * 64,
        registered_count=1,
        ceiling_count=1,
        exposed_count=1,
        profile_changed=False,
        step=1,
        provider_name="provider-a",
        model="model-a",
        model_step_id="mstep_00000000000000000000000000000001",
    ).model_copy(update=changes)

    with pytest.raises(ValueError, match=message):
        build_request_footprint(
            _request(),
            provider_name="provider-a",
            step=1,
            attempt=1,
            max_attempts=1,
            request_variant=RequestVariant.INITIAL,
            observation_id="mismatched-exposure-authority",
            model_step_id="mstep_00000000000000000000000000000001",
            model_attempt_id="matt_00000000000000000000000000000001",
            execution_profile_fingerprint=execution_profile_fingerprint,
            tool_exposure=exposure,
        )


def test_request_footprint_versions_are_required() -> None:
    footprint_payload = _build(_request()).model_dump(mode="python")
    del footprint_payload["schema_version"]
    with pytest.raises(ValidationError, match="Field required"):
        RequestFootprint.model_validate(footprint_payload)

    fingerprint_payload = _build(
        _request(),
        config=RequestFootprintConfig(
            fingerprint_key_id="version-key",
            fingerprint_key=SecretStr("v" * 32),
        ),
    ).model_dump(mode="python")
    del fingerprint_payload["fingerprints"]["provider_neutral_request"]["canonicalization_version"]
    with pytest.raises(ValidationError, match="Field required"):
        RequestFootprint.model_validate(fingerprint_payload)

    manifest = build_prompt_contribution_manifest(
        rendered_system_prompt="system prompt",
        contributions={PromptContributionKind.AGENT_INSTRUCTIONS: ("system prompt",)},
        config=RequestFootprintConfig(),
    )
    assert manifest is not None
    manifest_payload = manifest.model_dump(mode="python")
    del manifest_payload["schema_version"]
    with pytest.raises(ValidationError, match="Field required"):
        type(manifest).model_validate(manifest_payload)


def test_request_footprint_records_exact_content_free_shape() -> None:
    footprint = _build(_request())

    assert footprint.schema_version == 1
    assert footprint.observation_id == "observation-1"
    assert footprint.provider_name == "provider-a"
    assert footprint.model == "model-a"
    assert footprint.step == 2
    assert footprint.attempt == 3
    assert footprint.max_attempts == 4
    assert footprint.request_variant == RequestVariant.STRUCTURED_OUTPUT_REPAIR
    assert footprint.messages.count == 3
    assert footprint.messages.system.count == 1
    assert [(group.role, group.part_type, group.count) for group in footprint.messages.groups] == [
        ("assistant", "text", 1),
        ("assistant", "tool_call", 1),
        ("user", "file", 1),
        ("user", "text", 1),
    ]
    assert footprint.tools.count == 1
    assert footprint.attachments.count == 1
    assert footprint.attachments.source_bytes == 4
    assert [
        (group.kind, group.count, group.source_bytes) for group in footprint.attachments.groups
    ] == [(FileAttachmentKind.DOCUMENT, 1, 4)]
    assert footprint.options.known_categories == (
        "cache_policy",
        "structured_output",
        "temperature",
    )
    assert footprint.options.unknown_count == 1
    assert footprint.structured_output.count == 1
    assert footprint.structured_output.size.canonical_json_bytes > 0
    assert footprint.context_pressure.method == "local_full_request_estimate"
    assert footprint.context_pressure.estimated_context_input_tokens > 0
    assert footprint.component_tokens.method == "local_full_request_estimate"
    assert footprint.component_tokens.confidence == "estimated"
    assert footprint.component_tokens.total_input_tokens == (
        footprint.context_pressure.estimated_context_input_tokens
    )
    assert footprint.component_tokens.system_message_input_tokens > 0
    assert footprint.component_tokens.non_system_message_input_tokens > 0
    assert footprint.component_tokens.tool_schema_input_tokens > 0
    assert footprint.component_tokens.structured_output_input_tokens > 0
    assert footprint.component_tokens.attachment_input_tokens >= 0
    assert footprint.component_tokens.request_options_input_tokens > 0
    assert footprint.total.size.canonical_json_bytes > footprint.messages.size.canonical_json_bytes

    serialized = json.dumps(footprint.model_dump(mode="json"), sort_keys=True)
    for private_value in (
        "System café",
        "Read the report",
        "Inspect the repository",
        "reports/q1.md",
        "artifact-secret-id",
        "quarterly-secret-name.pdf",
        "do-not-persist-this",
        "unknown-option-value",
        "attachment-secret-metadata",
    ):
        assert private_value not in serialized


def test_request_footprint_binds_hosted_web_search_authority() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="hosted-tool-key",
        fingerprint_key=SecretStr("h" * 32),
    )
    baseline_request = _request()
    hosted_request = baseline_request.model_copy(
        update={
            "hosted_tools": (
                OpenAIWebSearch(
                    search_context_size="high",
                    external_web_access=False,
                    blocked_domains=("example.com",),
                ),
            )
        }
    )

    baseline = _build(baseline_request, config=config)
    hosted = _build(hosted_request, config=config)

    assert hosted.tools.count == baseline.tools.count + 1
    assert (
        hosted.fingerprints.provider_neutral_request.value
        != baseline.fingerprints.provider_neutral_request.value
    )
    assert hosted.fingerprints.tool_manifest.value != baseline.fingerprints.tool_manifest.value


def test_unknown_provider_option_values_do_not_affect_persisted_measurements() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="options-key",
        fingerprint_key=SecretStr("o" * 32),
    )
    short = _request()
    short.options["vendor_private_option"] = "x"
    long = _request()
    long.options["vendor_private_option"] = "private-" * 1_000

    short_footprint = _build(short, config=config)
    long_footprint = _build(long, config=config)

    assert short_footprint.options.unknown_count == 1
    assert short_footprint.options.size == long_footprint.options.size
    assert short_footprint.total.size == long_footprint.total.size
    assert short_footprint.context_pressure == long_footprint.context_pressure
    assert short_footprint.component_tokens == long_footprint.component_tokens
    assert short_footprint.fingerprints.provider_neutral_request.value != (
        long_footprint.fingerprints.provider_neutral_request.value
    )


def test_tool_structured_output_runtime_controls_do_not_change_request_evidence() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="structured-key",
        fingerprint_key=SecretStr("s" * 32),
    )

    def request_for(spec: StructuredOutputSpec) -> ModelRequest:
        return ModelRequest(
            model="model-a",
            messages=[
                Message.text("system", structured_output_tool_instruction(spec)),
                Message.text("user", "Return the answer"),
            ],
            tools=[structured_output_tool_spec(spec)],
            options={"structured_output": structured_output_spec_payload(spec)},
        )

    baseline_spec = StructuredOutputSpec(
        json_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        max_retries=1,
        repair_prompt="Try once more",
    )
    changed_controls_spec = baseline_spec.model_copy(
        update={"max_retries": 7, "repair_prompt": "Use a different repair prompt"},
        deep=True,
    )
    changed_schema_spec = StructuredOutputSpec(
        json_schema={"type": "object", "properties": {"result": {"type": "integer"}}},
        max_retries=1,
        repair_prompt="Try once more",
    )

    baseline = build_request_footprint(
        request_for(baseline_spec),
        provider_name="provider-a",
        step=1,
        attempt=1,
        max_attempts=1,
        request_variant=RequestVariant.INITIAL,
        observation_id="structured-baseline",
        model_step_id="mstep_00000000000000000000000000000001",
        model_attempt_id="matt_00000000000000000000000000000002",
        config=config,
        structured_output_instruction=structured_output_tool_instruction(baseline_spec),
    )
    changed_controls = build_request_footprint(
        request_for(changed_controls_spec),
        provider_name="provider-a",
        step=1,
        attempt=1,
        max_attempts=1,
        request_variant=RequestVariant.INITIAL,
        observation_id="structured-controls",
        model_step_id="mstep_00000000000000000000000000000001",
        model_attempt_id="matt_00000000000000000000000000000002",
        config=config,
        structured_output_instruction=structured_output_tool_instruction(changed_controls_spec),
    )
    changed_schema = build_request_footprint(
        request_for(changed_schema_spec),
        provider_name="provider-a",
        step=1,
        attempt=1,
        max_attempts=1,
        request_variant=RequestVariant.INITIAL,
        observation_id="structured-schema",
        model_step_id="mstep_00000000000000000000000000000001",
        model_attempt_id="matt_00000000000000000000000000000002",
        config=config,
        structured_output_instruction=structured_output_tool_instruction(changed_schema_spec),
    )

    assert baseline.total == changed_controls.total
    assert baseline.options == changed_controls.options
    assert baseline.context_pressure == changed_controls.context_pressure
    assert baseline.component_tokens == changed_controls.component_tokens
    assert baseline.fingerprints.provider_neutral_request.value == (
        changed_controls.fingerprints.provider_neutral_request.value
    )
    assert baseline.fingerprints.provider_neutral_request.value != (
        changed_schema.fingerprints.provider_neutral_request.value
    )


def test_native_structured_output_runtime_controls_do_not_change_request_evidence() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="native-structured-key",
        fingerprint_key=SecretStr("n" * 32),
    )

    def observe(
        *,
        schema: dict[str, Any],
        max_retries: int,
        repair_prompt: str,
    ) -> RequestFootprint:
        return _build(
            ModelRequest(
                model="model-a",
                messages=[Message.text("user", "Return JSON")],
                options={
                    "temperature": 0.25,
                    "structured_output": {
                        "strategy": "native",
                        "name": "answer",
                        "schema": schema,
                        "max_retries": max_retries,
                        "repair_prompt": repair_prompt,
                    },
                },
            ),
            config=config,
        )

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    baseline = observe(schema=schema, max_retries=1, repair_prompt="Try again")
    changed_controls = observe(
        schema=schema,
        max_retries=8,
        repair_prompt="Use another repair instruction",
    )
    changed_schema = observe(
        schema={
            "type": "object",
            "properties": {
                "result": {
                    "type": "integer",
                    "description": "Detailed result field. " * 100,
                }
            },
        },
        max_retries=1,
        repair_prompt="Try again",
    )

    assert baseline.total == changed_controls.total
    assert baseline.options == changed_controls.options
    assert baseline.context_pressure == changed_controls.context_pressure
    assert baseline.component_tokens == changed_controls.component_tokens
    assert baseline.fingerprints.provider_neutral_request.value == (
        changed_controls.fingerprints.provider_neutral_request.value
    )
    assert baseline.fingerprints.provider_neutral_request.value != (
        changed_schema.fingerprints.provider_neutral_request.value
    )
    assert baseline.context_pressure.estimated_request_options_input_tokens > 0
    assert (
        changed_schema.context_pressure.estimated_request_options_input_tokens
        == baseline.context_pressure.estimated_request_options_input_tokens
    )
    assert (
        changed_schema.context_pressure.estimated_structured_output_input_tokens
        > baseline.context_pressure.estimated_structured_output_input_tokens
    )
    assert (
        changed_schema.context_pressure.estimated_context_input_tokens
        > baseline.context_pressure.estimated_context_input_tokens
    )
    for footprint in (baseline, changed_schema):
        pressure = footprint.context_pressure
        assert pressure.estimated_request_overhead_input_tokens == (
            pressure.estimated_tool_schema_input_tokens
            + pressure.estimated_structured_output_input_tokens
            + pressure.estimated_request_options_input_tokens
        )
        assert pressure.estimated_delta_input_tokens == (
            pressure.estimated_message_input_tokens
            + pressure.estimated_attachment_input_tokens
            + pressure.estimated_request_overhead_input_tokens
        )
        assert pressure.estimated_context_input_tokens == pressure.estimated_delta_input_tokens
        assert pressure.estimated_context_window_tokens == (
            pressure.estimated_context_input_tokens + pressure.reserved_output_tokens
        )
        assert footprint.component_tokens.total_input_tokens == (
            pressure.estimated_context_input_tokens
        )


def test_effective_provider_defaults_change_options_and_request_identity() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="provider-default-key",
        fingerprint_key=SecretStr("d" * 32),
    )
    request = ModelRequest(
        model="model-a",
        messages=[Message.text("user", "Hello")],
    )

    def observe(max_tokens: int) -> RequestFootprint:
        return analyze_request_footprint(
            request,
            provider=AnthropicProvider(api_key="test-key", max_tokens=max_tokens),
            provider_name="anthropic",
            step=1,
            attempt=1,
            max_attempts=1,
            request_variant=RequestVariant.INITIAL,
            observation_id=f"provider-default-{max_tokens}",
            model_step_id="mstep_00000000000000000000000000000001",
            model_attempt_id="matt_00000000000000000000000000000002",
            config=config,
        )

    smaller = observe(512)
    larger = observe(4_096_000_000)

    assert smaller.options != larger.options
    assert smaller.context_pressure != larger.context_pressure
    assert smaller.fingerprints.provider_neutral_request.value != (
        larger.fingerprints.provider_neutral_request.value
    )


def test_provider_neutral_thinking_uses_effective_adapter_projection() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="thinking-key",
        fingerprint_key=SecretStr("t" * 32),
    )
    provider = OpenAIProvider(api_key="test-key")

    def observe(*, effort: str, max_tokens: int) -> RequestFootprint:
        return analyze_request_footprint(
            ModelRequest(
                model="model-a",
                messages=[Message.text("user", "Think")],
                options={
                    "thinking": {
                        "enabled": True,
                        "effort": effort,
                        "max_tokens": max_tokens,
                    }
                },
            ),
            provider=provider,
            provider_name="openai",
            step=1,
            attempt=1,
            max_attempts=1,
            request_variant=RequestVariant.INITIAL,
            observation_id=f"thinking-{effort}-{max_tokens}",
            model_step_id="mstep_00000000000000000000000000000001",
            model_attempt_id="matt_00000000000000000000000000000002",
            config=config,
        )

    baseline = observe(effort="high", max_tokens=100)
    ignored_control_change = observe(effort="high", max_tokens=1_000)
    provider_visible_change = observe(effort="low", max_tokens=100)

    assert baseline.options.known_categories == ("openai.reasoning",)
    assert baseline.options.unknown_count == 0
    assert baseline.total == ignored_control_change.total
    assert baseline.context_pressure == ignored_control_change.context_pressure
    assert baseline.fingerprints.provider_neutral_request.value == (
        ignored_control_change.fingerprints.provider_neutral_request.value
    )
    assert baseline.fingerprints.provider_neutral_request.value != (
        provider_visible_change.fingerprints.provider_neutral_request.value
    )


def test_openai_fingerprint_uses_only_effective_selected_provider_options() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="selected-provider-key",
        fingerprint_key=SecretStr("p" * 32),
    )
    provider = OpenAIProvider(api_key="test-key")

    def request(*, openai_route: str, anthropic_route: str) -> ModelRequest:
        return ModelRequest(
            model="model-a",
            messages=[Message.text("user", "Hello")],
            options={
                "openai": {
                    "temperature": 0.25,
                    "metadata": {"route": openai_route},
                },
                "anthropic": {"metadata": {"route": anthropic_route}},
            },
        )

    def observe(model_request: ModelRequest) -> RequestFootprint:
        return analyze_request_footprint(
            model_request,
            provider=provider,
            provider_name="openai",
            step=1,
            attempt=1,
            max_attempts=1,
            request_variant=RequestVariant.INITIAL,
            observation_id="selected-provider-options",
            model_step_id="mstep_00000000000000000000000000000001",
            model_attempt_id="matt_00000000000000000000000000000002",
            config=config,
        )

    baseline_request = request(openai_route="primary", anthropic_route="first")
    changed_inactive_request = request(openai_route="primary", anthropic_route="second")
    changed_active_request = request(openai_route="secondary", anthropic_route="first")
    baseline = observe(baseline_request)
    changed_inactive = observe(changed_inactive_request)
    changed_active = observe(changed_active_request)

    assert build_openai_payload(baseline_request) == build_openai_payload(changed_inactive_request)
    assert baseline.fingerprints.provider_neutral_request.value == (
        changed_inactive.fingerprints.provider_neutral_request.value
    )
    assert build_openai_payload(baseline_request) != build_openai_payload(changed_active_request)
    assert baseline.fingerprints.provider_neutral_request.value != (
        changed_active.fingerprints.provider_neutral_request.value
    )
    assert baseline.options == changed_active.options
    assert baseline.context_pressure == changed_active.context_pressure


def test_inactive_provider_namespace_does_not_change_request_evidence() -> None:
    provider = OpenAIProvider(api_key="test-key")

    def observe(options: dict[str, Any], observation_id: str) -> RequestFootprint:
        return analyze_request_footprint(
            ModelRequest(
                model="model-a",
                messages=[Message.text("user", "Hello")],
                options=options,
            ),
            provider=provider,
            provider_name="openai",
            step=1,
            attempt=1,
            max_attempts=1,
            request_variant=RequestVariant.INITIAL,
            observation_id=observation_id,
            model_step_id="mstep_00000000000000000000000000000001",
            model_attempt_id="matt_00000000000000000000000000000002",
        )

    baseline = observe({}, "inactive-provider-baseline")
    inactive = observe(
        {"anthropic": {"temperature": 0.75, "metadata": {"private": "ignored"}}},
        "inactive-provider-added",
    )

    assert baseline.options.unknown_count == 0
    assert inactive.options.unknown_count == 0
    assert baseline.options == inactive.options
    assert baseline.total == inactive.total
    assert baseline.context_pressure == inactive.context_pressure
    assert baseline.component_tokens == inactive.component_tokens
    assert baseline.fingerprints == inactive.fingerprints
    assert (
        baseline.fingerprints.provider_neutral_request.availability
        == RequestFingerprintAvailability.UNAVAILABLE
    )


def test_anthropic_fingerprint_uses_mode_exclusive_effective_thinking() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="anthropic-thinking-key",
        fingerprint_key=SecretStr("a" * 32),
    )
    provider = AnthropicProvider(api_key="test-key")

    def request(*, raw_budget: int, raw_marker: str, effort: str) -> ModelRequest:
        return ModelRequest(
            model="claude-test",
            messages=[Message.text("user", "Think")],
            options={
                "anthropic": {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": raw_budget,
                        "marker": raw_marker,
                    }
                },
                "thinking": {"enabled": True, "effort": effort},
            },
        )

    def observe(model_request: ModelRequest) -> RequestFootprint:
        return analyze_request_footprint(
            model_request,
            provider=provider,
            provider_name="anthropic",
            step=1,
            attempt=1,
            max_attempts=1,
            request_variant=RequestVariant.INITIAL,
            observation_id="anthropic-effective-thinking",
            model_step_id="mstep_00000000000000000000000000000001",
            model_attempt_id="matt_00000000000000000000000000000002",
            config=config,
        )

    baseline_request = request(raw_budget=100, raw_marker="first", effort="high")
    changed_shadowed_request = request(raw_budget=900, raw_marker="second", effort="high")
    changed_effective_request = request(raw_budget=100, raw_marker="first", effort="low")
    baseline = observe(baseline_request)
    changed_shadowed = observe(changed_shadowed_request)
    changed_effective = observe(changed_effective_request)

    assert build_anthropic_payload(
        baseline_request,
        default_max_tokens=provider.max_tokens,
    ) == build_anthropic_payload(
        changed_shadowed_request,
        default_max_tokens=provider.max_tokens,
    )
    assert baseline.fingerprints.provider_neutral_request.value == (
        changed_shadowed.fingerprints.provider_neutral_request.value
    )
    assert baseline.options == changed_shadowed.options
    assert baseline.context_pressure == changed_shadowed.context_pressure
    assert build_anthropic_payload(
        baseline_request,
        default_max_tokens=provider.max_tokens,
    ) != build_anthropic_payload(
        changed_effective_request,
        default_max_tokens=provider.max_tokens,
    )
    assert baseline.fingerprints.provider_neutral_request.value != (
        changed_effective.fingerprints.provider_neutral_request.value
    )


def test_attachment_footprint_counts_each_prompt_and_tool_result_occurrence() -> None:
    reference = file_attachment(
        artifact_id="artifact-one",
        kind="document",
        filename="report.pdf",
        content_type="application/pdf",
        size_bytes=4,
    )
    resolved = {
        "artifact_id": "artifact-one",
        "kind": "document",
        "filename": "report.pdf",
        "content_type": "application/pdf",
        "data_base64": "YWJjZA==",
        "metadata": {},
    }
    request = ModelRequest(
        model="model-a",
        messages=[
            Message(
                role="user",
                content=(
                    FilePart(attachment=reference),
                    FilePart(attachment=reference),
                ),
            ),
            Message.tool_result(
                tool_call_id="call-one",
                tool_name="inspect",
                artifacts=[reference],
            ),
        ],
        options={RESOLVED_FILE_ATTACHMENTS_OPTION: {"artifact-one": resolved}},
    )

    footprint = _build(request)

    assert footprint.attachments.count == 3
    assert footprint.attachments.source_bytes == 12
    assert [
        (group.kind, group.count, group.source_bytes) for group in footprint.attachments.groups
    ] == [(FileAttachmentKind.DOCUMENT, 3, 12)]


def test_attachment_lookup_identity_does_not_change_provider_neutral_evidence() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="attachment-key",
        fingerprint_key=SecretStr("a" * 32),
    )

    def request_for(artifact_id: str) -> ModelRequest:
        reference = file_attachment(
            artifact_id=artifact_id,
            kind="document",
            filename="report.pdf",
            content_type="application/pdf",
            size_bytes=4,
        )
        return ModelRequest(
            model="model-a",
            messages=[
                Message(
                    role="user",
                    content=(FilePart(attachment=reference),),
                ),
                Message.text("assistant", "tail"),
            ],
            options={
                RESOLVED_FILE_ATTACHMENTS_OPTION: {
                    artifact_id: {
                        "artifact_id": artifact_id,
                        "kind": "document",
                        "filename": "report.pdf",
                        "content_type": "application/pdf",
                        "data_base64": "YWJjZA==",
                        "metadata": {"private": artifact_id},
                    }
                }
            },
        )

    first_request = request_for("a")
    second_request = request_for("artifact-private-identity-" * 20)
    first = _build(first_request, config=config)
    second = _build(second_request, config=config)

    assert first.messages == second.messages
    assert first.total == second.total
    assert first.attachments == second.attachments
    assert first.context_pressure == second.context_pressure
    assert first.component_tokens == second.component_tokens
    provider = OpenAIProvider(api_key="test-key")
    assert analyze_request_context_pressure(
        first_request,
        provider=provider,
    ) == analyze_request_context_pressure(
        second_request,
        provider=provider,
    )
    assert first.fingerprints.provider_neutral_request.value == (
        second.fingerprints.provider_neutral_request.value
    )
    assert first.fingerprints.conversation_prefix.availability == (
        RequestFingerprintAvailability.AVAILABLE
    )
    assert first.fingerprints.conversation_prefix.value == (
        second.fingerprints.conversation_prefix.value
    )
    assert [item.fingerprint.value for item in first.cache_breakpoints] == [
        item.fingerprint.value for item in second.cache_breakpoints
    ]


def test_runtime_only_tool_result_fields_do_not_change_provider_neutral_evidence() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="tool-result-key",
        fingerprint_key=SecretStr("r" * 32),
    )

    def request_for(*, identity_digit: str, private_value: str) -> ModelRequest:
        tool_round_id = f"tround_{identity_digit * 32}"
        model_step_id = f"mstep_{identity_digit * 32}"
        model_attempt_id = f"matt_{identity_digit * 32}"
        return ModelRequest(
            model="model-a",
            messages=[
                Message.text("user", "Inspect the repository"),
                Message.tool_call(
                    tool_call_id="call-one",
                    tool_name="inspect",
                    arguments={"path": "README.md"},
                    tool_round_id=tool_round_id,
                    model_step_id=model_step_id,
                    model_attempt_id=model_attempt_id,
                ),
                Message.tool_result(
                    tool_call_id="call-one",
                    tool_name="inspect",
                    content="Inspection complete",
                    structured={"private": private_value},
                    artifacts=[{"type": "internal-record", "value": private_value}],
                    tool_round_id=tool_round_id,
                    model_step_id=model_step_id,
                    model_attempt_id=model_attempt_id,
                ),
            ],
        )

    baseline = _build(
        request_for(identity_digit="1", private_value="short"),
        config=config,
    )
    changed = _build(
        request_for(identity_digit="2", private_value="private-value-" * 100),
        config=config,
    )

    assert baseline.messages == changed.messages
    assert baseline.total == changed.total
    assert baseline.context_pressure == changed.context_pressure
    assert baseline.component_tokens == changed.component_tokens
    assert baseline.fingerprints.provider_neutral_request.value == (
        changed.fingerprints.provider_neutral_request.value
    )
    assert baseline.fingerprints.conversation_prefix.value == (
        changed.fingerprints.conversation_prefix.value
    )
    assert [item.fingerprint.value for item in baseline.cache_breakpoints] == [
        item.fingerprint.value for item in changed.cache_breakpoints
    ]


def test_attachment_metadata_does_not_change_payload_or_request_pressure() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="attachment-metadata-key",
        fingerprint_key=SecretStr("m" * 32),
    )
    provider = OpenAIProvider(api_key="test-key")

    def request_for(metadata: dict[str, Any]) -> ModelRequest:
        reference = file_attachment(
            artifact_id="artifact-one",
            kind="document",
            filename="report.pdf",
            content_type="application/pdf",
            size_bytes=4,
            metadata=metadata,
        )
        return ModelRequest(
            model="model-a",
            messages=[Message(role="user", content=(FilePart(attachment=reference),))],
            options={
                RESOLVED_FILE_ATTACHMENTS_OPTION: {
                    "artifact-one": {
                        "artifact_id": "artifact-one",
                        "kind": "document",
                        "filename": "report.pdf",
                        "content_type": "application/pdf",
                        "data_base64": "YWJjZA==",
                        "metadata": metadata,
                    }
                }
            },
        )

    def observe(request: ModelRequest, observation_id: str) -> RequestFootprint:
        return analyze_request_footprint(
            request,
            provider=provider,
            provider_name="openai",
            step=1,
            attempt=1,
            max_attempts=1,
            request_variant=RequestVariant.INITIAL,
            observation_id=observation_id,
            model_step_id="mstep_00000000000000000000000000000001",
            model_attempt_id="matt_00000000000000000000000000000002",
            config=config,
        )

    baseline_request = request_for({"private": "short"})
    changed_request = request_for({"private": "private-value-" * 1_000})
    baseline = observe(baseline_request, "attachment-metadata-baseline")
    changed = observe(changed_request, "attachment-metadata-changed")

    assert build_openai_payload(baseline_request) == build_openai_payload(changed_request)
    assert baseline.messages == changed.messages
    assert baseline.total == changed.total
    assert baseline.context_pressure == changed.context_pressure
    assert baseline.component_tokens == changed.component_tokens
    assert baseline.fingerprints.provider_neutral_request.value == (
        changed.fingerprints.provider_neutral_request.value
    )


def test_effective_cache_policy_changes_complete_request_identity() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="effective-cache-key",
        fingerprint_key=SecretStr("c" * 32),
    )
    request = ModelRequest(
        model="claude-test",
        messages=[
            Message.text("system", "Stable instructions"),
            Message.text("user", "Hello"),
        ],
    )
    standard_provider = AnthropicProvider(
        api_key="test-key",
        cache_policy=CachePolicy(breakpoints=(CacheBreakpoint.SYSTEM_PROMPT,)),
    )
    extended_provider = AnthropicProvider(
        api_key="test-key",
        cache_policy=CachePolicy(
            breakpoints=(CacheBreakpoint.SYSTEM_PROMPT,),
            ttl="extended",
        ),
    )

    def observe(
        provider: AnthropicProvider,
        observation_id: str,
    ) -> tuple[RequestFootprint, dict[str, Any]]:
        effective_policy = provider.request_cache_policy(request)
        assert effective_policy is not None
        footprint = analyze_request_footprint(
            request,
            provider=provider,
            provider_name="anthropic",
            step=1,
            attempt=1,
            max_attempts=1,
            request_variant=RequestVariant.INITIAL,
            observation_id=observation_id,
            model_step_id="mstep_00000000000000000000000000000001",
            model_attempt_id="matt_00000000000000000000000000000002",
            config=config,
        )
        payload = build_anthropic_payload(
            request,
            default_max_tokens=provider.max_tokens,
            cache_policy=effective_policy,
        )
        return footprint, payload

    standard, standard_payload = observe(standard_provider, "standard-cache")
    extended, extended_payload = observe(extended_provider, "extended-cache")

    assert standard_payload != extended_payload
    assert standard.options != extended.options
    assert "cache_policy" in standard.options.known_categories
    assert "cache_policy" in extended.options.known_categories
    assert standard.total != extended.total
    assert standard.context_pressure != extended.context_pressure
    assert standard.context_pressure == analyze_request_context_pressure(
        request,
        provider=standard_provider,
    )
    assert extended.context_pressure == analyze_request_context_pressure(
        request,
        provider=extended_provider,
    )
    assert standard.component_tokens != extended.component_tokens
    assert standard.fingerprints.provider_neutral_request.value != (
        extended.fingerprints.provider_neutral_request.value
    )
    assert standard.cache_breakpoints[0].ttl == "standard"
    assert extended.cache_breakpoints[0].ttl == "extended"
    assert standard.cache_breakpoints[0].fingerprint.value != (
        extended.cache_breakpoints[0].fingerprint.value
    )


@pytest.mark.parametrize("drop_entire_message", [True, False], ids=["entire", "partial"])
def test_anthropic_cache_prefix_uses_only_transmitted_message_projection(
    drop_entire_message: bool,
) -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="anthropic-projected-prefix-key",
        fingerprint_key=SecretStr("v" * 32),
    )
    provider = AnthropicProvider(
        api_key="test-key",
        cache_policy=CachePolicy(breakpoints=(CacheBreakpoint.CONVERSATION_PREFIX,)),
    )
    first = Message.text("user", "first")
    last = Message.text("user", "last")
    dropped_parts = (
        ProviderStatePart(provider="other", state={"opaque": "state"}),
        ThinkingPart(text="not echoable"),
    )
    if drop_entire_message:
        projected_messages = [first, last]
        original_middle = Message(role="assistant", content=dropped_parts)
    else:
        visible = TextPart(text="visible answer")
        projected_messages = [first, Message(role="assistant", content=(visible,)), last]
        original_middle = Message(role="assistant", content=(*dropped_parts, visible))
    original_request = ModelRequest(
        model="claude-test",
        messages=[first, original_middle, last],
    )
    projected_request = ModelRequest(model="claude-test", messages=projected_messages)

    def observe(request: ModelRequest, observation_id: str) -> RequestFootprint:
        return analyze_request_footprint(
            request,
            provider=provider,
            provider_name="anthropic",
            step=1,
            attempt=1,
            max_attempts=1,
            request_variant=RequestVariant.INITIAL,
            observation_id=observation_id,
            model_step_id="mstep_00000000000000000000000000000001",
            model_attempt_id="matt_00000000000000000000000000000002",
            config=config,
        )

    original_projection = provider.request_cache_projection(original_request)
    projected_projection = provider.request_cache_projection(projected_request)
    assert original_projection is not None
    assert projected_projection is not None
    assert provider.cache_policy is not None
    original_payload = build_anthropic_payload(
        original_request,
        default_max_tokens=provider.max_tokens,
        cache_policy=provider.cache_policy,
    )
    projected_payload = build_anthropic_payload(
        projected_request,
        default_max_tokens=provider.max_tokens,
        cache_policy=provider.cache_policy,
    )
    assert original_payload == projected_payload
    actual_messages = original_payload["messages"]
    assert isinstance(actual_messages, list)
    expected_prefix = json.loads(json.dumps(actual_messages[:-1]))
    expected_content = expected_prefix[-1]["content"]
    assert isinstance(expected_content, list)
    expected_content[-1].pop("cache_control")
    assert original_projection.conversation_prefix == tuple(expected_prefix)
    assert original_projection.conversation_prefix == projected_projection.conversation_prefix

    original = observe(original_request, "anthropic-original-prefix")
    projected = observe(projected_request, "anthropic-projected-prefix")
    assert original.fingerprints.conversation_prefix.value == (
        projected.fingerprints.conversation_prefix.value
    )
    original_breakpoint = next(
        item
        for item in original.cache_breakpoints
        if item.kind == CacheBreakpoint.CONVERSATION_PREFIX
    )
    projected_breakpoint = next(
        item
        for item in projected.cache_breakpoints
        if item.kind == CacheBreakpoint.CONVERSATION_PREFIX
    )
    assert original_breakpoint.fingerprint.value == projected_breakpoint.fingerprint.value
    assert original.fingerprints.provider_neutral_request.value != (
        projected.fingerprints.provider_neutral_request.value
    )


def test_cache_breakpoint_records_are_unique_and_canonically_ordered() -> None:
    request = _request()
    config = RequestFootprintConfig(
        fingerprint_key_id="canonical-cache-breakpoint-key",
        fingerprint_key=SecretStr("b" * 32),
    )

    def observe(breakpoints: tuple[CacheBreakpoint, ...], observation_id: str) -> RequestFootprint:
        return build_request_footprint(
            request,
            provider_name="provider-a",
            step=1,
            attempt=1,
            max_attempts=1,
            request_variant=RequestVariant.INITIAL,
            observation_id=observation_id,
            model_step_id="mstep_00000000000000000000000000000001",
            model_attempt_id="matt_00000000000000000000000000000002",
            config=config,
            cache_policy=CachePolicy(breakpoints=breakpoints),
        )

    permuted = observe(
        (
            CacheBreakpoint.TOOL_DEFINITIONS,
            CacheBreakpoint.CONVERSATION_PREFIX,
            CacheBreakpoint.SYSTEM_PROMPT,
            CacheBreakpoint.CONVERSATION_PREFIX,
        ),
        "permuted-cache-breakpoints",
    )
    canonical = observe(
        (
            CacheBreakpoint.CONVERSATION_PREFIX,
            CacheBreakpoint.SYSTEM_PROMPT,
            CacheBreakpoint.TOOL_DEFINITIONS,
        ),
        "canonical-cache-breakpoints",
    )

    assert permuted.cache_breakpoints == canonical.cache_breakpoints
    assert [item.kind for item in permuted.cache_breakpoints] == [
        CacheBreakpoint.CONVERSATION_PREFIX,
        CacheBreakpoint.SYSTEM_PROMPT,
        CacheBreakpoint.TOOL_DEFINITIONS,
    ]
    noncanonical_payload = canonical.model_dump(mode="python")
    noncanonical_payload["cache_breakpoints"] = list(
        reversed(noncanonical_payload["cache_breakpoints"])
    )
    with pytest.raises(ValidationError, match="unique and canonically ordered"):
        RequestFootprint.model_validate(noncanonical_payload)

    duplicate_payload = canonical.model_dump(mode="python")
    duplicate_payload["cache_breakpoints"] = [
        *duplicate_payload["cache_breakpoints"],
        duplicate_payload["cache_breakpoints"][0],
    ]
    with pytest.raises(ValidationError, match="unique and canonically ordered"):
        RequestFootprint.model_validate(duplicate_payload)


def test_unused_raw_cache_controls_do_not_change_effective_request_evidence() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="effective-cache-control-key",
        fingerprint_key=SecretStr("u" * 32),
    )
    provider = AnthropicProvider(
        api_key="test-key",
        cache_policy=CachePolicy(breakpoints=(CacheBreakpoint.SYSTEM_PROMPT,)),
    )

    def request_for(conversation_prefix_n: int) -> ModelRequest:
        return ModelRequest(
            model="claude-test",
            messages=[
                Message.text("system", "Stable instructions"),
                Message.text("user", "Hello"),
            ],
            options={"cache_policy": {"conversation_prefix_n": conversation_prefix_n}},
        )

    def observe(model_request: ModelRequest, observation_id: str) -> RequestFootprint:
        return analyze_request_footprint(
            model_request,
            provider=provider,
            provider_name="anthropic",
            step=1,
            attempt=1,
            max_attempts=1,
            request_variant=RequestVariant.INITIAL,
            observation_id=observation_id,
            model_step_id="mstep_00000000000000000000000000000001",
            model_attempt_id="matt_00000000000000000000000000000002",
            config=config,
        )

    baseline_request = request_for(1)
    changed_request = request_for(8)
    baseline_policy = provider.request_cache_policy(baseline_request)
    changed_policy = provider.request_cache_policy(changed_request)
    assert baseline_policy is not None
    assert changed_policy is not None
    baseline = observe(baseline_request, "effective-cache-control-baseline")
    changed = observe(changed_request, "effective-cache-control-changed")

    assert build_anthropic_payload(
        baseline_request,
        default_max_tokens=provider.max_tokens,
        cache_policy=baseline_policy,
    ) == build_anthropic_payload(
        changed_request,
        default_max_tokens=provider.max_tokens,
        cache_policy=changed_policy,
    )
    assert baseline.options == changed.options
    assert baseline.options.unknown_count == 0
    assert baseline.total == changed.total
    assert baseline.context_pressure == changed.context_pressure
    assert baseline.fingerprints.provider_neutral_request.value == (
        changed.fingerprints.provider_neutral_request.value
    )


def test_provider_owned_reserved_option_names_remain_authoritative() -> None:
    class ReservedNameProjectionProvider(ModelProvider):
        supports_native_structured_output = True

        def __init__(self, *, effort: str, provider_mode: str) -> None:
            self.effort = effort
            self.provider_mode = provider_mode

        def request_footprint_options(self, _request: ModelRequest) -> dict[str, Any]:
            return {
                "thinking": {"effort": self.effort},
                "structured_output": {"provider_mode": self.provider_mode},
            }

        def request_fingerprint_options(self, _request: ModelRequest) -> dict[str, Any]:
            return self.request_footprint_options(_request)

        async def stream(
            self,
            _request: ModelRequest,
        ) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    config = RequestFootprintConfig(
        fingerprint_key_id="reserved-provider-option-key",
        fingerprint_key=SecretStr("v" * 32),
    )
    request = ModelRequest(
        model="custom-model",
        messages=[Message.text("user", "Return JSON")],
        options={
            "structured_output": {
                "strategy": "native",
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            }
        },
    )

    def observe(provider: ModelProvider, observation_id: str) -> RequestFootprint:
        return analyze_request_footprint(
            request,
            provider=provider,
            provider_name="custom",
            step=1,
            attempt=1,
            max_attempts=1,
            request_variant=RequestVariant.INITIAL,
            observation_id=observation_id,
            model_step_id="mstep_00000000000000000000000000000001",
            model_attempt_id="matt_00000000000000000000000000000002",
            config=config,
        )

    baseline = observe(
        ReservedNameProjectionProvider(effort="low", provider_mode="first"),
        "reserved-option-baseline",
    )
    changed_thinking = observe(
        ReservedNameProjectionProvider(effort="high", provider_mode="first"),
        "reserved-option-thinking",
    )
    changed_provider_structured_output = observe(
        ReservedNameProjectionProvider(effort="low", provider_mode="second"),
        "reserved-option-structured-output",
    )

    assert baseline.options.known_categories == (
        "structured_output",
        "structured_output.provider_mode",
        "thinking.effort",
    )
    assert baseline.options != changed_thinking.options
    assert baseline.options != changed_provider_structured_output.options
    assert baseline.fingerprints.provider_neutral_request.value != (
        changed_thinking.fingerprints.provider_neutral_request.value
    )
    assert baseline.fingerprints.provider_neutral_request.value != (
        changed_provider_structured_output.fingerprints.provider_neutral_request.value
    )


def test_builtin_provider_namespace_projects_safe_visible_options() -> None:
    request = _request()
    request.options = {
        RESOLVED_FILE_ATTACHMENTS_OPTION: request.options[RESOLVED_FILE_ATTACHMENTS_OPTION],
        "openai": {
            "temperature": 0.5,
            "tool_choice": {"provider-option-secret-key": "value"},
            "metadata": {"private": "provider-option-secret"},
        },
    }

    footprint = analyze_request_footprint(
        request,
        provider=OpenAIProvider(api_key="test-key"),
        provider_name="openai",
        step=1,
        attempt=1,
        max_attempts=1,
        request_variant=RequestVariant.INITIAL,
        observation_id="observation-openai-options",
        model_step_id="mstep_00000000000000000000000000000001",
        model_attempt_id="matt_00000000000000000000000000000002",
    )

    assert footprint.options.known_categories == (
        "openai.temperature",
        "openai.tool_choice",
    )
    assert footprint.options.unknown_count == 1
    assert footprint.options.size.canonical_json_bytes > 2
    assert footprint.component_tokens.request_options_input_tokens > 0
    assert "provider-option-secret" not in json.dumps(
        footprint.model_dump(mode="json"), sort_keys=True
    )
    assert "provider-option-secret-key" not in json.dumps(
        footprint.model_dump(mode="json"), sort_keys=True
    )


def test_request_footprint_without_key_has_typed_unavailable_fingerprints() -> None:
    footprint = _build(_request())

    fingerprints = (
        footprint.fingerprints.provider_neutral_request,
        footprint.fingerprints.provider_wire_request,
        footprint.fingerprints.system,
        footprint.fingerprints.tool_manifest,
        footprint.fingerprints.conversation_prefix,
    )
    assert all(
        item.availability == RequestFingerprintAvailability.UNAVAILABLE for item in fingerprints
    )
    assert all(item.value is None for item in fingerprints)
    assert footprint.cache_breakpoints
    assert all(
        item.fingerprint.availability == RequestFingerprintAvailability.UNAVAILABLE
        for item in footprint.cache_breakpoints
    )


def test_request_footprint_hmac_is_stable_for_mapping_order_and_sensitive_to_semantics() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="analysis-key",
        fingerprint_key=SecretStr("a" * 32),
    )
    first_request = _request()
    reordered = _request()
    reordered.tools[0]["input_schema"]["properties"] = {
        "path": {"type": "string"},
        "limit": {"type": "integer"},
    }

    first = _build(first_request, config=config)
    second = _build(reordered, config=config)
    changed = _build(_request(tool_description="Inspect and edit the repository"), config=config)

    first_identity = first.fingerprints.provider_neutral_request
    assert first_identity.availability == RequestFingerprintAvailability.AVAILABLE
    assert first_identity.algorithm == "hmac-sha256"
    assert first_identity.key_id == "analysis-key"
    assert first_identity.canonicalization_version == 1
    assert first_identity.value == second.fingerprints.provider_neutral_request.value
    assert first_identity.value != changed.fingerprints.provider_neutral_request.value
    assert first.fingerprints.tool_manifest.value != changed.fingerprints.tool_manifest.value
    assert first.fingerprints.system.value == changed.fingerprints.system.value
    assert first.fingerprints.provider_wire_request.availability == (
        RequestFingerprintAvailability.UNAVAILABLE
    )
    assert first.fingerprints.provider_wire_request.unavailable_reason == (
        "provider_wire_not_observed"
    )


def test_request_footprint_hmac_normalizes_backend_numeric_reconstruction() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="numeric-key",
        fingerprint_key=SecretStr("n" * 32),
    )

    exponent_form = _build(
        _request(),
        config=config,
        fingerprint_provider_options={"provider.output_limit": 1e18},
    )
    reconstructed_integer = _build(
        _request(),
        config=config,
        fingerprint_provider_options={
            "provider.output_limit": 1_000_000_000_000_000_000,
        },
    )

    assert exponent_form.fingerprints.provider_neutral_request.value == (
        reconstructed_integer.fingerprints.provider_neutral_request.value
    )


def test_request_footprint_hmac_tracks_order_attachments_and_provider_options() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="shape-key",
        fingerprint_key=SecretStr("h" * 32),
    )
    baseline = _build(_request(), config=config)
    baseline_identity = baseline.fingerprints.provider_neutral_request.value

    reordered_messages = _request()
    reordered_messages.messages[1:] = reversed(reordered_messages.messages[1:])

    reordered_tools = _request()
    reordered_tools.tools.append(
        {
            "name": "summarize",
            "description": "Summarize one file",
            "input_schema": {"type": "object"},
        }
    )
    added_tool = _build(reordered_tools, config=config)
    reordered_tools.tools.reverse()

    changed_attachment = _request()
    changed_attachment.options[RESOLVED_FILE_ATTACHMENTS_OPTION]["artifact-secret-id"][
        "data_base64"
    ] = "YWJjZGU="

    changed_option = _request()
    changed_option.options["temperature"] = 0.5

    assert _build(
        reordered_messages, config=config
    ).fingerprints.provider_neutral_request.value != (baseline_identity)
    assert _build(reordered_tools, config=config).fingerprints.provider_neutral_request.value != (
        added_tool.fingerprints.provider_neutral_request.value
    )
    assert _build(
        changed_attachment, config=config
    ).fingerprints.provider_neutral_request.value != (baseline_identity)
    assert _build(changed_option, config=config).fingerprints.provider_neutral_request.value != (
        baseline_identity
    )


def test_cache_breakpoint_fingerprints_cover_cumulative_prefixes() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="cache-key",
        fingerprint_key=SecretStr("c" * 32),
    )
    baseline = _build(_request(), config=config)
    changed_system_request = _request()
    changed_system_request.messages[0] = Message.text("system", "Changed system")
    changed_system = _build(changed_system_request, config=config)

    baseline_breakpoints = {
        item.kind: item.fingerprint.value for item in baseline.cache_breakpoints
    }
    changed_breakpoints = {
        item.kind: item.fingerprint.value for item in changed_system.cache_breakpoints
    }

    assert baseline_breakpoints.keys() == changed_breakpoints.keys()
    assert all(
        baseline_breakpoints[kind] != changed_breakpoints[kind] for kind in baseline_breakpoints
    )


def test_request_footprint_key_rotation_changes_identity_domain() -> None:
    first_config = RequestFootprintConfig(
        fingerprint_key_id="shape-key-2026-08",
        fingerprint_key=SecretStr("a" * 32),
    )
    rotated_config = RequestFootprintConfig(
        fingerprint_key_id="shape-key-2026-09",
        fingerprint_key=SecretStr("b" * 32),
    )

    first = _build(_request(), config=first_config).fingerprints.provider_neutral_request
    rotated = _build(_request(), config=rotated_config).fingerprints.provider_neutral_request

    assert first.key_id == "shape-key-2026-08"
    assert rotated.key_id == "shape-key-2026-09"
    assert first.value != rotated.value


def test_request_footprint_config_defaults_on_and_requires_complete_strong_key() -> None:
    assert RequestFootprintConfig().enabled is True

    with pytest.raises(ValidationError, match="fingerprint_key_id and fingerprint_key"):
        RequestFootprintConfig(fingerprint_key_id="key-only")
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        RequestFootprintConfig(
            fingerprint_key_id="weak",
            fingerprint_key=SecretStr("too-short"),
        )

    config = RequestFootprintConfig(
        fingerprint_key_id="safe-key",
        fingerprint_key=SecretStr("s" * 32),
    )
    assert "s" * 32 not in repr(config)


def test_prompt_contribution_attribution_requires_matching_keyed_final_system() -> None:
    config = RequestFootprintConfig(
        fingerprint_key_id="prompt-key",
        fingerprint_key=SecretStr("p" * 32),
    )
    rendered = (
        "[Agent instructions]\nReview code\n\n[Workspace instructions]\nFollow repository policy"
    )
    manifest = build_prompt_contribution_manifest(
        rendered_system_prompt=rendered,
        contributions={
            PromptContributionKind.AGENT_INSTRUCTIONS: ("Review code",),
            PromptContributionKind.CAYU_FRAMING: (
                "[Agent instructions]\n",
                "\n\n[Workspace instructions]\n",
            ),
            PromptContributionKind.WORKSPACE_INSTRUCTIONS: ("Follow repository policy",),
        },
        config=config,
    )
    assert manifest is not None

    request = _request()
    request.messages[0] = Message.text("system", rendered)
    matched = build_request_footprint(
        request,
        provider_name="provider-a",
        step=1,
        attempt=1,
        max_attempts=1,
        request_variant=RequestVariant.INITIAL,
        observation_id="observation-prompt-match",
        model_step_id="mstep_00000000000000000000000000000001",
        model_attempt_id="matt_00000000000000000000000000000002",
        config=config,
        prompt_contribution_manifest=manifest,
    )
    assert matched.prompt_contributions.availability == PromptContributionAvailability.AVAILABLE
    assert [contribution.kind for contribution in matched.prompt_contributions.contributions] == [
        PromptContributionKind.AGENT_INSTRUCTIONS,
        PromptContributionKind.CAYU_FRAMING,
        PromptContributionKind.WORKSPACE_INSTRUCTIONS,
    ]

    request.messages[0] = Message.text("system", f"{rendered}\nChanged")
    changed = build_request_footprint(
        request,
        provider_name="provider-a",
        step=1,
        attempt=1,
        max_attempts=1,
        request_variant=RequestVariant.INITIAL,
        observation_id="observation-prompt-changed",
        model_step_id="mstep_00000000000000000000000000000001",
        model_attempt_id="matt_00000000000000000000000000000002",
        config=config,
        prompt_contribution_manifest=manifest,
    )
    assert changed.prompt_contributions.availability == PromptContributionAvailability.UNAVAILABLE
    assert changed.prompt_contributions.unavailable_reason == "final_system_changed"

    serialized = json.dumps(manifest.model_dump(mode="json"), sort_keys=True)
    assert "Review code" not in serialized
    assert "Follow repository policy" not in serialized
    assert "p" * 32 not in json.dumps(config.model_dump(mode="json"), default=str)
