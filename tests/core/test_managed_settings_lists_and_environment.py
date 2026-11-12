"""List and map settings, the environment fallback, and reading a value with its source."""

from __future__ import annotations

import asyncio
import logging

import pytest
from pydantic import ValidationError
from tests.core.test_managed_settings_controller import FakeCloud, _controller, _outcomes

from cayu import AgentSpec, CayuApp
from cayu.managed_settings import (
    ManagedSetting,
    ManagedSettingReading,
    list_setting,
    map_setting,
    thinking_effort_setting,
)

REPOSITORY = r"[A-Za-z0-9-]+/[A-Za-z0-9._-]+"


def _repositories(**fields) -> ManagedSetting:
    return list_setting("repositories", label="Repositories", item_pattern=REPOSITORY, **fields)


def _merge_modes(**fields) -> ManagedSetting:
    return map_setting(
        "merge_modes",
        label="Merge modes",
        item_pattern=REPOSITORY,
        value_choices=("report", "approve", "auto"),
        **fields,
    )


def test_a_list_setting_reads_its_items_and_names_an_item_that_does_not_fit():
    setting = _repositories(max_items=3)

    assert setting.validate_value(" o/a,\no/b , o/a,") == "o/a,o/b"
    assert setting.validate_value(["o/a", "o/b"]) == "o/a,o/b"
    assert setting.read_value("o/a,o/b") == ("o/a", "o/b")
    assert setting.read_value(None) == ()
    for value, message in [
        ("o/a,not a repo", "item 'not a repo' does not match"),
        ("x,y", "items 'x' and 'y' do not match"),
        (["o/a", "o/b,o/c"], "'o/b,o/c' isn't an item"),
        ("o/a,o/b,o/c,o/d", "more than 3 items"),
        (["o/a", 1], "expected a list of strings"),
        (7, "expected a list"),
    ]:
        with pytest.raises(ValueError, match=message):
            setting.validate_value(value)

    labels = list_setting("labels", label="Labels", item_choices=["hold", "wip"])
    with pytest.raises(ValueError, match="item 'merge' is not one of hold, wip"):
        labels.validate_value("hold,merge")


def test_a_map_setting_reads_a_dict_and_names_a_bad_key_or_value():
    setting = _merge_modes()

    assert setting.validate_value("o/a = auto\no/b=report") == "o/a=auto,o/b=report"
    assert setting.validate_value({"o/a": "approve"}) == "o/a=approve"
    assert setting.read_value("o/a=auto,o/b=report") == {"o/a": "auto", "o/b": "report"}
    assert setting.read_value(None) == {}
    for value, message in [
        ("o/a=merge", r"the value in 'o/a=merge' is not one of report, approve, auto"),
        ("bad key=auto", "key 'bad key' does not match"),
        ("o/a", "'o/a' isn't KEY=VALUE"),
        ("o/a=", "'o/a' has no value"),
        ("o/a=auto,o/a=report", "'o/a' listed more than once"),
        (["o/a=auto"], "expected a map"),
    ]:
        with pytest.raises(ValueError, match=message):
            setting.validate_value(value)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"item_pattern": "x"}, "only to list and map"),
        ({"format": "list", "item_pattern": "x", "item_choices": ["x"]}, "not both"),
        ({"format": "list", "value_choices": ["x"]}, "only to map settings"),
        ({"format": "list", "item_pattern": "("}, "doesn't compile"),
        ({"format": "list", "item_choices": ["a,b"]}, "can't contain a comma"),
        ({"format": "map", "item_choices": ["a=b"]}, "can't contain '='"),
        ({"format": "list", "item_pattern": "[a-z]+", "default": ["a", "B"]}, "'B' does not"),
        ({"format": "list", "default": ["o/a,o/b"]}, "list item 'o/a,o/b' can't contain"),
        ({"format": "map", "default": {"a=b": "c"}}, "map key 'a=b' can't contain"),
        ({"format": "map", "default": {"a": "b\nc"}}, "map value 'b.*can't contain"),
        ({"format": "list", "type": "bool", "max_length": None}, "only to string settings"),
        ({"format": "set"}, "Input should be 'list' or 'map'"),
        ({"env": "lower_case"}, "env must name an environment variable"),
    ],
)
def test_list_and_map_declarations_are_checked(fields, message):
    with pytest.raises(ValidationError, match=message):
        ManagedSetting.model_validate(
            {"key": "setting", "label": "Setting", "type": "string", "max_length": 99, **fields}
        )


def test_list_and_map_settings_publish_as_v1_strings_and_read_back():
    app = CayuApp(enable_logging=False)
    app.declare_settings(
        [_repositories(default=["o/a", "o/b"], env="APP_REPOSITORIES"), _merge_modes()]
    )
    published = {
        item["key"]: item
        for item in app.managed_settings_schema()["settings"]
        if item["key"].startswith("app.")
    }

    assert published["app.repositories"] == {
        "key": "app.repositories",
        "owner": "app",
        "type": "string",
        "format": "list",
        "item_pattern": REPOSITORY,
        "max_length": 4096,
        "default": "o/a,o/b",
        "applies": "next_session",
        "label": "Repositories",
        "env": "APP_REPOSITORIES",
    }
    assert published["app.merge_modes"]["format"] == "map"
    assert published["app.merge_modes"]["value_choices"][0] == {
        "value": "report",
        "label": "report",
    }
    # A descriptor as Cloud returns it, with fields this version doesn't know.
    read = ManagedSetting.from_descriptor({**published["app.merge_modes"], "admin_only": False})
    assert read.format == "map"
    assert read.descriptor() == published["app.merge_modes"]
    with pytest.raises(ValidationError, match="only to string settings"):
        ManagedSetting.from_descriptor(
            {**published["app.merge_modes"], "type": "int", "max_length": None}
        )


@pytest.mark.parametrize(
    ("setting", "text", "value"),
    [
        (ManagedSetting(key="flag", type="bool", label="Flag"), " Yes ", True),
        (ManagedSetting(key="flag", type="bool", label="Flag"), "OFF", False),
        (ManagedSetting(key="flag", type="bool", label="Flag"), "0", False),
        (ManagedSetting(key="flag", type="bool", label="Flag"), "  ", None),
        (ManagedSetting(key="count", type="int", label="Count", minimum=0), " 42 ", 42),
        (ManagedSetting(key="ratio", type="number", label="Ratio"), "0.5", 0.5),
        (ManagedSetting(key="ratio", type="number", label="Ratio"), "2", 2),
        (ManagedSetting(key="name", type="string", label="Name", max_length=9), " x ", " x "),
        (_repositories(), "", ""),
        (_repositories(), "o/a, o/b", "o/a,o/b"),
        (_merge_modes(), "o/a=auto", "o/a=auto"),
    ],
)
def test_text_is_read_by_the_settings_own_type(setting, text, value):
    assert setting.parse_text(text) == value


@pytest.mark.parametrize(
    ("setting", "text", "message"),
    [
        (ManagedSetting(key="flag", type="bool", label="Flag"), "maybe", "true or false"),
        (ManagedSetting(key="count", type="int", label="Count"), "4.5", "expected an integer"),
        (ManagedSetting(key="count", type="int", label="Count", maximum=3), "4", "greater"),
        (ManagedSetting(key="ratio", type="number", label="Ratio"), "nan", "finite"),
    ],
)
def test_text_that_does_not_fit_the_type_is_refused(setting, text, message):
    with pytest.raises(ValueError, match=message):
        setting.parse_text(text)


def test_the_environment_applies_when_cloud_has_no_value(monkeypatch, caplog):
    app = CayuApp(enable_logging=False)
    app.declare_settings(
        [
            _repositories(env="APP_REPOSITORIES"),
            ManagedSetting(
                key="introduce", type="bool", label="Introduce", default=True, env="APP_INTRODUCE"
            ),
            ManagedSetting(key="rounds", type="int", label="Rounds", default=5, env="APP_ROUNDS"),
        ]
    )
    monkeypatch.delenv("APP_REPOSITORIES", raising=False)
    monkeypatch.setenv("APP_INTRODUCE", "off")
    monkeypatch.setenv("APP_ROUNDS", "many")

    with caplog.at_level(logging.WARNING, logger="cayu.managed_settings"):
        readings = {item.key: item for item in app.read_managed_settings()}
        app.read_managed_settings()

    assert readings["app.repositories"] == ManagedSettingReading(
        key="app.repositories", value=(), source="default", env="APP_REPOSITORIES"
    )
    assert readings["app.introduce"].value is False
    assert readings["app.introduce"].source == "environment"
    assert readings["app.rounds"].value == 5
    assert readings["app.rounds"].source == "default"
    assert readings["app.rounds"].refused == "APP_ROUNDS: expected an integer"
    # Warned once, however often it is read.
    assert [record.getMessage() for record in caplog.records].count(
        "Ignoring the environment's value for app.rounds: APP_ROUNDS: expected an integer"
    ) == 1
    monkeypatch.setenv("APP_REPOSITORIES", "o/a,o/b")
    assert app.managed_setting_value("app.repositories") == ("o/a", "o/b")
    assert app.read_managed_setting("config.run.max_steps").source == "default"
    with pytest.raises(KeyError, match="app.nope"):
        app.read_managed_setting("app.nope")


def test_cloud_wins_over_the_environment_and_a_bad_cloud_item_is_refused(monkeypatch):
    cloud = FakeCloud()
    cloud.values = {"app.repositories": "o/c", "app.merge_modes": "o/a=merge"}
    app = CayuApp(enable_logging=False, managed_settings=_controller(cloud))
    app.declare_settings(
        [
            _repositories(env="APP_REPOSITORIES", applies="live"),
            _merge_modes(env="APP_MERGE_MODES", applies="live", default={"o/a": "report"}),
        ]
    )
    monkeypatch.setenv("APP_REPOSITORIES", "o/a")
    monkeypatch.delenv("APP_MERGE_MODES", raising=False)

    async def scenario():
        await app.start_managed_settings(role="worker")
        repositories = app.read_managed_setting("app.repositories")
        assert (repositories.value, repositories.source) == (("o/c",), "cloud")
        modes = app.read_managed_setting("app.merge_modes")
        assert (modes.value, modes.source) == ({"o/a": "report"}, "default")
        assert modes.refused == (
            "Cayu Cloud's value: the value in 'o/a=merge' is not one of report, approve, auto"
        )
        outcomes = _outcomes(cloud.reports[-1])
        assert outcomes["app.repositories"] == ("adopted", None)
        assert outcomes["app.merge_modes"] == (
            "refused",
            "the value in 'o/a=merge' is not one of report, approve, auto",
        )
        await app.aclose()

    asyncio.run(scenario())


def test_an_environment_secret_ref_to_an_unset_variable_falls_back(monkeypatch):
    app = CayuApp(enable_logging=False)
    app.declare_settings(
        [
            ManagedSetting(
                key="search_key",
                type="secret_ref",
                label="Key",
                default="FALLBACK_TOKEN",
                env="APP_SEARCH_KEY",
            )
        ]
    )
    monkeypatch.setenv("APP_SEARCH_KEY", "NOT_SET_ANYWHERE")
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)

    reading = app.read_managed_setting("app.search_key")

    assert (reading.value, reading.source) == ("FALLBACK_TOKEN", "default")
    assert reading.refused == "APP_SEARCH_KEY: the environment variable is not set on the Agent"
    monkeypatch.setenv("NOT_SET_ANYWHERE", "x")
    assert app.read_managed_setting("app.search_key").source == "environment"


def test_settings_cayu_applies_to_an_agent_cannot_take_env():
    app = CayuApp(enable_logging=False)
    effort = thinking_effort_setting().model_copy(update={"env": "EFFORT"})
    with pytest.raises(ValueError, match="can't take env"):
        app.register_agent(AgentSpec(name="reviewer", model="m"), settings=[effort])
