"""Parts declare managed settings where they are registered; the app assembles the schema."""

from __future__ import annotations

import asyncio
import hashlib
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cayu import AgentSpec, CayuApp, CayuConfig
from cayu.budgets import BudgetLimit, BudgetPolicy, BudgetWindow
from cayu.configuration import RunDefaults
from cayu.context.thinking import ThinkingConfig
from cayu.environments import EnvironmentFactory, EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.managed_settings import (
    MAX_MANAGED_SETTINGS,
    ManagedSetting,
    ManagedSettingsLimitExceeded,
    managed_settings_schema_document,
    model_setting,
    provider_option_setting,
    settings_digest,
    thinking_effort_setting,
)
from cayu.runtime import AppManifest
from cayu.vaults import SecretRedactor


def _provider(name: str = "scripted") -> ScriptedModelProvider:
    return ScriptedModelProvider([], name=name)


def _app(**kwargs) -> CayuApp:
    app = CayuApp(enable_logging=False, **kwargs)
    app.register_provider(_provider(), default=True)
    return app


def _by_key(app: CayuApp) -> dict[str, dict]:
    return {item["key"]: item for item in app.managed_settings_schema()["settings"]}


class _Factory(EnvironmentFactory):
    async def create(self, request):  # pragma: no cover - never materialized here
        raise AssertionError("not used")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"key": "Bad.Key", "type": "bool"}, "key must be"),
        ({"key": "name", "type": "string"}, "max_length is required"),
        ({"key": "name", "type": "int", "max_length": 4}, "max_length is required"),
        ({"key": "tone", "type": "enum"}, "choices are required"),
        ({"key": "tone", "type": "enum", "choices": []}, "1 to 100 choices"),
        ({"key": "steps", "type": "int", "minimum": 5, "maximum": 1}, "minimum must not"),
        ({"key": "steps", "type": "int", "maximum": 3, "default": 4}, "greater than the maximum"),
        ({"key": "steps", "type": "int", "default": True}, "expected an integer"),
        ({"key": "ratio", "type": "number", "minimum": 0.5, "default": 0.1}, "less than"),
        ({"key": "flag", "type": "bool", "minimum": 1}, "only to int and number"),
        ({"key": "api_key", "type": "secret_ref", "default": "sk-live-1"}, "environment variable"),
        ({"key": "x", "type": "bool", "label": "x" * 81}, "at most 80"),
    ],
)
def test_descriptor_follows_the_contract(fields, message):
    with pytest.raises(ValidationError, match=message):
        ManagedSetting.model_validate({"label": "Setting", **fields})


def test_descriptor_json_matches_the_contract_shape():
    setting = ManagedSetting(
        key="agent.reviewer.max_steps",
        owner="agent:reviewer",
        type="int",
        default=40,
        minimum=1,
        maximum=500,
        applies="next_session",
        label="Maximum steps",
        description="Steps a session may take before it stops.",
    )
    # The contract's own example, unchanged.
    assert setting.descriptor() == {
        "key": "agent.reviewer.max_steps",
        "owner": "agent:reviewer",
        "type": "int",
        "default": 40,
        "minimum": 1,
        "maximum": 500,
        "applies": "next_session",
        "label": "Maximum steps",
        "description": "Steps a session may take before it stops.",
    }
    unset = ManagedSetting(key="api_key", type="secret_ref", label="API key")
    assert unset.descriptor()["default"] is None
    assert unset.validate_value("OPENAI_API_KEY") == "OPENAI_API_KEY"


def test_an_agent_declares_its_settings_where_it_is_registered():
    app = _app(config=CayuConfig(run=RunDefaults(thinking=ThinkingConfig(effort="low"))))
    app.register_agent(
        AgentSpec(name="reviewer", model="model-a", provider_options={"temperature": 0.2}),
        settings=[
            model_setting(),
            thinking_effort_setting(),
            provider_option_setting("temperature", type="number", label="Temperature", maximum=2),
            ManagedSetting(key="max_files", type="int", label="Files", default=4, minimum=1),
        ],
    )
    settings = _by_key(app)
    assert settings["agent.reviewer.model"]["default"] == "model-a"
    assert settings["agent.reviewer.model"]["applies"] == "next_session"
    # The agent's own thinking is unset, so the application default is its code default.
    assert settings["agent.reviewer.thinking.effort"]["default"] == "low"
    assert settings["agent.reviewer.thinking.effort"]["applies"] == "restart"
    assert settings["agent.reviewer.provider_options.temperature"]["default"] == 0.2
    assert settings["agent.reviewer.max_files"]["default"] == 4
    assert {item["owner"] for key, item in settings.items() if key.startswith("agent.")} == {
        "agent:reviewer"
    }


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        (ManagedSetting(key="model", type="bool", label="Model"), "string or enum"),
        (
            ManagedSetting(key="model", type="string", max_length=9, label="M", applies="live"),
            "next_session or restart",
        ),
        (
            ManagedSetting(
                key="thinking.effort",
                type="enum",
                choices=[{"value": "high", "label": "High"}],
                label="Effort",
            ),
            "applies only at restart",
        ),
        (ManagedSetting(key="thinking.budget", type="int", label="Budget"), "thinking.effort"),
        (
            ManagedSetting(key="provider_options.temperature", type="number", label="T"),
            "applies only at restart",
        ),
        (model_setting(choices=["model-b"]), "not one of the allowed choices"),
    ],
)
def test_settings_cayu_applies_are_checked_before_the_agent_registers(setting, message):
    app = _app()
    with pytest.raises(ValueError, match=message):
        app.register_agent(AgentSpec(name="reviewer", model="model-a"), settings=[setting])
    # Nothing was registered or declared.
    assert "reviewer" not in app._agent_registry.registrations
    assert not any(key.startswith("agent.") for key in _by_key(app))


def test_a_part_whose_name_cannot_be_a_key_segment_cannot_declare_settings():
    app = _app()
    with pytest.raises(ValueError, match="lowercase"):
        app.register_agent(AgentSpec(name="Reviewer", model="m"), settings=[model_setting()])
    # Without settings, the same agent still registers.
    app.register_agent(AgentSpec(name="Reviewer", model="m"))


def test_a_setting_that_contains_a_secret_is_refused():
    app = _app(secret_redactor=SecretRedactor("secret-value-canary"))
    with pytest.raises(ValueError, match="secret value"):
        app.declare_settings(
            [
                ManagedSetting(
                    key="token",
                    type="string",
                    max_length=64,
                    label="T",
                    default="secret-value-canary",
                )
            ]
        )


def test_providers_environment_factories_and_the_app_declare_their_own_namespaces():
    app = CayuApp(enable_logging=False)
    app.register_provider(
        _provider("primary"),
        default=True,
        settings=[ManagedSetting(key="timeout_s", type="int", label="Timeout", default=30)],
    )
    app.register_environment_factory(
        EnvironmentSpec(name="sandbox"),
        _Factory(),
        settings=[ManagedSetting(key="cpu", type="int", label="CPU", applies="restart")],
    )
    app.declare_settings([ManagedSetting(key="api_key", type="secret_ref", label="Search API key")])
    settings = _by_key(app)
    assert settings["provider.primary.timeout_s"]["owner"] == "provider:primary"
    assert settings["environment.sandbox.cpu"]["owner"] == "environment:sandbox"
    assert settings["app.api_key"]["owner"] == "app"
    with pytest.raises(ValueError, match="more than once"):
        app.declare_settings([ManagedSetting(key="api_key", type="bool", label="Again")])


def test_framework_declares_chosen_config_fields_and_keyed_app_budget_limits():
    policy = BudgetPolicy(
        limits=[
            BudgetLimit.usd(
                Decimal("250.50"), scope="app", window=BudgetWindow.calendar(period="month")
            ),
            BudgetLimit.usd(10, scope="agent", key="reviewer"),
            # An agent name that can't be part of a key is left out.
            BudgetLimit.usd(10, scope="agent", key="Writer"),
        ]
    )
    app = _app(config=CayuConfig(run=RunDefaults(max_steps=40)), budget_policy=policy)
    settings = _by_key(app)
    assert settings["config.run.max_steps"]["default"] == 40
    assert settings["config.run.max_steps"]["owner"] == "config"
    assert settings["config.run.limits.max_total_tokens"]["default"] is None
    assert settings["config.run.retry_policy.max_attempts"]["default"] == 5
    assert settings["config.run.thinking.effort"]["applies"] == "restart"
    budget = settings["budget.app-month.max_estimated_cost"]
    assert budget["default"] == 250.5
    assert budget["owner"] == "budget:app-month"
    assert budget["applies"] == "restart"
    assert [key for key in settings if key.startswith("budget.")] == [
        "budget.agent-reviewer.max_estimated_cost",
        "budget.app-month.max_estimated_cost",
    ]
    assert settings["budget.agent-reviewer.max_estimated_cost"]["default"] == 10


def test_a_hand_assembled_app_with_no_declarations_still_has_a_schema():
    schema = CayuApp(enable_logging=False).managed_settings_schema()
    assert schema["contract"] == "cayu.managed-settings-schema/v1"
    assert {item["owner"] for item in schema["settings"]} == {"config"}


def _declare(app: CayuApp, order: list[str]) -> None:
    for name in order:
        app.register_agent(AgentSpec(name=name, model=f"{name}-model"), settings=[model_setting()])


def test_assembly_is_sorted_and_deterministic_whatever_the_registration_order():
    first, second = _app(), _app()
    _declare(first, ["writer", "reviewer", "triage"])
    _declare(second, ["triage", "writer", "reviewer"])
    schema = first.managed_settings_schema()
    assert schema == second.managed_settings_schema()
    keys = [item["key"] for item in schema["settings"]]
    assert keys == sorted(keys)


def test_digest_is_sha256_of_the_canonical_settings_json():
    app = _app()
    app.declare_settings(
        [
            ManagedSetting(
                key="greeting", type="string", max_length=40, label="Grüße", default="héllo"
            )
        ]
    )
    schema = app.managed_settings_schema()
    canonical = json.dumps(
        schema["settings"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert schema["digest"] == "sha256:" + hashlib.sha256(canonical).hexdigest()
    assert settings_digest(schema["settings"]) == schema["digest"]
    # Non-ASCII stays literal in the canonical form.
    assert "Grüße".encode() in canonical


def test_assembly_refuses_a_schema_over_the_limits_instead_of_truncating():
    app = _app()
    room = MAX_MANAGED_SETTINGS - len(app.managed_settings())
    app.declare_settings(
        [ManagedSetting(key=f"flag_{index}", type="bool", label="Flag") for index in range(room)]
    )
    with pytest.raises(ManagedSettingsLimitExceeded, match="limit is 500"):
        app.declare_settings([ManagedSetting(key="one_more", type="bool", label="Flag")])
    assert len(app.managed_settings()) == MAX_MANAGED_SETTINGS

    choices = [{"value": f"{index:03d}" + "v" * 200, "label": "l" * 80} for index in range(100)]
    big = [
        ManagedSetting(key=f"big_{index}", type="enum", choices=choices, label="Big")
        for index in range(20)
    ]
    with pytest.raises(ManagedSettingsLimitExceeded, match="bytes"):
        managed_settings_schema_document(
            [item.model_copy(update={"key": f"app.{item.key}", "owner": "app"}) for item in big]
        )
    with pytest.raises(ManagedSettingsLimitExceeded, match="bytes"):
        _app().declare_settings(big)


def test_an_agent_whose_model_the_gateway_policy_manages_leaves_model_out():
    app = _app()
    app.register_agent(AgentSpec(name="reviewer", model="m"), settings=[model_setting()])
    app.register_agent(AgentSpec(name="writer", model="m"), settings=[model_setting()])
    app.model_policy = SimpleNamespace(controllers=(SimpleNamespace(agent_name="reviewer"),))
    keys = set(_by_key(app))
    assert "agent.reviewer.model" not in keys
    assert "agent.writer.model" in keys


def test_manifest_carries_the_schema_and_fingerprints_it():
    app = _app()
    before = app.describe()
    assert before.settings is not None
    assert before.settings.model_dump(mode="json") == app.managed_settings_schema()
    app.register_agent(AgentSpec(name="reviewer", model="m"), settings=[model_setting()])
    after = app.describe()
    assert after.settings is not None
    assert after.settings.digest != before.settings.digest
    assert after.fingerprint != app_without_settings_fingerprint(app)
    # The manifest's own JSON round-trips, and an older manifest without settings still loads.
    assert AppManifest.model_validate_json(after.model_dump_json()) == after
    legacy = after.model_dump(mode="json")
    del legacy["settings"]
    assert AppManifest.model_validate(legacy).settings is None


def app_without_settings_fingerprint(app: CayuApp) -> str:
    other = _app()
    other.register_agent(AgentSpec(name="reviewer", model="m"))
    return other.describe().fingerprint


def _documented_schemas() -> dict[str, dict]:
    import re
    from pathlib import Path

    text = (Path(__file__).parents[2] / "docs" / "managed-settings.md").read_text()
    blocks = [json.loads(block) for block in re.findall(r"```json\n(.*?)```", text, re.S)]
    return {block["$id"]: block for block in blocks if "$id" in block}


def test_documented_json_schemas_validate_what_the_runtime_emits():
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    schemas = _documented_schemas()
    assert set(schemas) == {
        "urn:cayu:managed-setting/v1",
        "urn:cayu:managed-settings-schema/v1",
        "urn:cayu:managed-settings-values/v1",
        "urn:cayu:managed-settings-report/v1",
    }
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
    registry = Registry().with_resources(
        (key, Resource.from_contents(value)) for key, value in schemas.items()
    )
    app = _app(budget_policy=BudgetPolicy(limits=[BudgetLimit.usd(5, scope="app")]))
    app.register_agent(
        AgentSpec(name="reviewer", model="m"),
        settings=[model_setting(), thinking_effort_setting()],
    )
    app.declare_settings([ManagedSetting(key="search_key", type="secret_ref", label="Key")])
    validator = Draft202012Validator(
        schemas["urn:cayu:managed-settings-schema/v1"], registry=registry
    )
    validator.validate(app.managed_settings_schema())
    with pytest.raises(Exception, match="does not match"):
        bad = app.managed_settings_schema()
        bad["settings"][0] = {**bad["settings"][0], "key": "Not A Key"}
        validator.validate(bad)
    # Only platform settings may be one segment; a Release's Agent settings may not.
    descriptor = Draft202012Validator(schemas["urn:cayu:managed-setting/v1"], registry=registry)
    platform = {"key": "sign_in", "owner": "platform", "type": "bool", "default": False}
    descriptor.validate({**platform, "applies": "restart", "label": "Sign-in"})
    with pytest.raises(Exception, match="does not match"):
        single = app.managed_settings_schema()
        single["settings"][0] = {**single["settings"][0], "key": "sign_in"}
        validator.validate(single)


def test_budget_names_too_long_or_secret_are_left_out():
    secret = "tenant-secret-canary"
    policy = BudgetPolicy(
        limits=[
            BudgetLimit.usd(10, scope="agent", key="a" * 120),
            BudgetLimit.usd(10, scope="agent", key=secret),
            BudgetLimit.usd(10, scope="agent", key="reviewer"),
            BudgetLimit.usd("0.0000005", scope="app"),
        ]
    )
    app = _app(budget_policy=policy, secret_redactor=SecretRedactor(secret))
    keys = [key for key in _by_key(app) if key.startswith("budget.")]
    assert keys == ["budget.agent-reviewer.max_estimated_cost", "budget.app.max_estimated_cost"]
    # An amount below the published minimum lowers the minimum instead of failing.
    assert _by_key(app)["budget.app.max_estimated_cost"]["minimum"] == 5e-07
    assert secret not in app.describe().model_dump_json()
    assert _by_key(app)[keys[0]]["minimum"] > 0


def test_unregistering_an_environment_drops_its_settings():
    async def scenario():
        app = CayuApp(enable_logging=False)
        settings = [ManagedSetting(key="cpu", type="int", label="CPU")]
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox"), _Factory(), settings=settings
        )
        await app.unregister_environment("sandbox")
        assert "environment.sandbox.cpu" not in _by_key(app)
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox"), _Factory(), settings=settings
        )
        assert "environment.sandbox.cpu" in _by_key(app)
        await app.aclose()

    asyncio.run(scenario())
