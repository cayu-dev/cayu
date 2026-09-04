from __future__ import annotations

import asyncio
import math

import pytest
from pydantic import ValidationError

from cayu.core import AgentSpec, Message, ThinkingConfig
from cayu.evals.testing import ScriptedModelProvider
from cayu.runtime import (
    DEFAULT_MAX_ENVIRONMENT_LIFECYCLE_OWNERS,
    DEFAULT_MAX_PARALLEL_TOOL_CALLS,
    DEFAULT_MAX_STEPS,
    MAX_STEPS,
    CayuApp,
    CayuConfig,
    EffectiveRunConfiguration,
    EvalConfig,
    RetryPolicy,
    RunDefaults,
    RunLimits,
    RunRequest,
    ToolExecutionConfig,
    copy_cayu_config,
)
from cayu.runtime.sessions import copy_run_request


def test_zero_configuration_uses_canonical_runtime_defaults() -> None:
    config = CayuConfig()

    assert config.run.max_steps == DEFAULT_MAX_STEPS == 64
    assert MAX_STEPS == 256
    assert config.run.limits.model_dump(mode="python") == RunLimits().model_dump(mode="python")
    assert config.run.thinking is None
    assert config.tool_execution.max_parallel_tool_calls == DEFAULT_MAX_PARALLEL_TOOL_CALLS == 4
    assert (
        config.operations.max_environment_lifecycle_owners
        == DEFAULT_MAX_ENVIRONMENT_LIFECYCLE_OWNERS
        == 256
    )


@pytest.mark.parametrize("value", [0, 257, True, 1.5, "64"])
def test_run_defaults_reject_invalid_max_steps(value: object) -> None:
    with pytest.raises(ValidationError):
        RunDefaults(max_steps=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, True, "30"])
def test_tool_execution_rejects_invalid_timeout(value: object) -> None:
    with pytest.raises(ValidationError):
        ToolExecutionConfig(tool_timeout_seconds=value)  # type: ignore[arg-type]


def test_configuration_models_are_frozen_and_inputs_are_detached() -> None:
    source_limits = RunLimits(max_total_tokens=100)
    run_defaults = RunDefaults(limits=source_limits)
    config = CayuConfig(run=run_defaults)

    source_limits.max_total_tokens = 200

    assert config.run.limits.max_total_tokens == 100
    with pytest.raises(ValidationError, match="frozen"):
        config.run = RunDefaults()  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        config.run.limits.max_total_tokens = 300


def test_default_factories_do_not_alias_mutable_run_limits() -> None:
    first = CayuConfig()
    second = CayuConfig()

    assert first.run.limits is not second.run.limits
    assert second.run.limits.max_tool_calls is None


def test_copy_cayu_config_is_detached_and_preserves_values() -> None:
    source = CayuConfig(run=RunDefaults(max_steps=128))
    copied = copy_cayu_config(source)

    assert copied is not source
    assert copied.run is not source.run
    assert copied.run.limits is not source.run.limits
    assert copied.run.limits.max_total_tokens is None


def test_run_defaults_copy_limits_returns_mutable_request_value() -> None:
    defaults = RunDefaults(limits=RunLimits(max_total_tokens=100))

    request_limits = defaults.copy_limits()
    request_limits.max_total_tokens = 200

    assert type(request_limits) is RunLimits
    assert defaults.limits.max_total_tokens == 100


def test_copy_cayu_config_rejects_untyped_input() -> None:
    with pytest.raises(TypeError, match="config must be a CayuConfig"):
        copy_cayu_config({})  # type: ignore[arg-type]


def test_cayu_app_resolves_only_omitted_request_defaults() -> None:
    configured_limits = RunLimits(max_total_tokens=100)
    configured_thinking = ThinkingConfig(enabled=False)
    app = CayuApp(
        config=CayuConfig(
            run=RunDefaults(
                max_steps=96,
                limits=configured_limits,
                thinking=configured_thinking,
            )
        ),
        enable_logging=False,
    )
    omitted = RunRequest(
        agent_name="assistant",
        messages=[Message.text("user", "hello")],
    )
    explicit = RunRequest(
        agent_name="assistant",
        messages=[Message.text("user", "hello")],
        max_steps=16,
        limits=RunLimits(),
        thinking=None,
    )

    resolved_omitted = app._with_application_run_defaults(omitted)
    resolved_explicit = app._with_application_run_defaults(explicit)

    assert resolved_omitted.max_steps == 96
    assert resolved_omitted.limits.max_total_tokens == 100
    assert resolved_omitted.thinking is None
    assert "max_steps" not in resolved_omitted.model_fields_set
    assert "limits" not in resolved_omitted.model_fields_set
    copied_omitted = copy_run_request(resolved_omitted)
    assert copied_omitted.max_steps == 96
    assert copied_omitted.limits.max_total_tokens == 100
    assert "max_steps" not in copied_omitted.model_fields_set
    assert "limits" not in copied_omitted.model_fields_set
    assert resolved_explicit.max_steps == 16
    assert resolved_explicit.limits == RunLimits()
    assert resolved_explicit.thinking is None


def test_agent_thinking_overrides_application_default() -> None:
    application_thinking = ThinkingConfig(enabled=False)
    role_thinking = ThinkingConfig(effort="high")
    app = CayuApp(
        config=CayuConfig(run=RunDefaults(thinking=application_thinking)),
        enable_logging=False,
    )

    app.register_agent(AgentSpec(name="inherited", model="model"))
    app.register_agent(AgentSpec(name="role-owned", model="model", thinking=role_thinking))

    assert app.get_agent("inherited").spec.thinking == application_thinking
    assert app.get_agent("role-owned").spec.thinking == role_thinking


def test_app_manifest_exposes_effective_config_ownership_and_source() -> None:
    app = CayuApp(
        config=CayuConfig(run=RunDefaults(max_steps=96)),
        enable_logging=False,
    )

    configuration = app.describe().runtime.configuration
    provenance = {item.path: item for item in configuration.provenance}

    assert configuration.values["run"]["max_steps"] == 96
    assert provenance["run.max_steps"].source == "application"
    assert provenance["run.max_steps"].owner == "cayu.runtime.config.RunDefaults"
    assert provenance["run.limits"].source == "framework"


def test_effective_run_configuration_inspection_is_typed_read_only_and_compatible() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
    request = RunRequest(
        agent_name="assistant",
        session_id="effective-config-inspection",
        messages=[Message.text("user", "inspect")],
    )

    inspection = asyncio.run(app.inspect_effective_run_configuration(request))
    legacy_fingerprint = asyncio.run(app.inspect_run_execution_profile(request))
    sessions = asyncio.run(app.session_store.list_sessions())

    assert type(inspection) is EffectiveRunConfiguration
    assert inspection.max_steps.value == DEFAULT_MAX_STEPS
    assert inspection.max_steps.source == "framework"
    assert inspection.limits.source == "framework"
    assert inspection.retry_policy.source == "framework"
    assert inspection.thinking.value is None
    assert inspection.thinking.source == "framework"
    assert inspection.execution_profile.fingerprint == legacy_fingerprint
    assert sessions.sessions == []
    with pytest.raises(ValidationError, match="frozen"):
        inspection.max_steps.value = 16
    with pytest.raises(ValidationError, match="frozen"):
        inspection.limits.value.max_total_tokens = 1


def test_effective_run_configuration_reports_application_role_and_request_sources() -> None:
    app = CayuApp(config=CayuConfig(run=RunDefaults(max_steps=128)), enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="scripted-model",
            thinking=ThinkingConfig(effort="high"),
        )
    )

    inherited = asyncio.run(
        app.inspect_effective_run_configuration(
            RunRequest(
                agent_name="assistant",
                session_id="profile-config-inspection",
                messages=[Message.text("user", "inspect")],
            )
        )
    )
    explicit = asyncio.run(
        app.inspect_effective_run_configuration(
            RunRequest(
                agent_name="assistant",
                session_id="explicit-config-inspection",
                messages=[Message.text("user", "inspect")],
                max_steps=16,
                limits=RunLimits(max_total_tokens=100),
                retry_policy=RetryPolicy(max_attempts=3),
                thinking=ThinkingConfig(enabled=False),
            )
        )
    )

    assert inherited.max_steps.value == 128
    assert inherited.max_steps.source == "application"
    assert inherited.thinking.value == ThinkingConfig(effort="high")
    assert inherited.thinking.source == "explicit"
    assert explicit.max_steps.source == "explicit"
    assert explicit.limits.source == "explicit"
    assert explicit.retry_policy.source == "explicit"
    assert explicit.thinking.source == "explicit"


@pytest.mark.parametrize("value", [0, 101, True, 1.5, "2"])
def test_eval_config_rejects_invalid_concurrency(value) -> None:
    with pytest.raises(ValidationError):
        EvalConfig(max_concurrency=value)


def test_eval_configuration_is_detached_and_inspectable() -> None:
    config = CayuConfig(evals=EvalConfig(max_concurrency=100))
    app = CayuApp(config=config, enable_logging=False)
    assert app.config.evals is not config.evals
    assert app.config.evals.max_concurrency == 100
    assert CayuConfig().evals.max_concurrency == 1
    assert CayuConfig(run=RunDefaults(max_steps=128)).evals.max_concurrency == 1
    with pytest.raises(ValidationError, match="frozen"):
        app.config.evals.max_concurrency = 2
    manifest = app.describe().runtime.configuration
    assert manifest.values["evals"]["max_concurrency"] == 100
    provenance = {item.path: item for item in manifest.provenance}
    assert provenance["evals.max_concurrency"].source == "application"
    assert provenance["evals.max_concurrency"].owner == "cayu.runtime.config.EvalConfig"
