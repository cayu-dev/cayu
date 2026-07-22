from __future__ import annotations

import json
from inspect import signature

import pytest
from pydantic import SecretStr, ValidationError

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
pytest.importorskip("pydantic_settings")

from pydantic_settings import DotEnvSettingsSource, EnvSettingsSource, SettingsError

from cayu.server import AuthenticatedAccess, OpenAccess
from cayu.server.auth import BasicAuth
from cayu.server.settings import DashboardSettings, ServerAccessSettings, ServerSettings


def _external_auth(_request):
    return {"subject": "external"}


def test_settings_fail_closed_without_an_access_selection(monkeypatch) -> None:
    monkeypatch.delenv("CAYU_SERVER_ACCESS__MODE", raising=False)
    settings = ServerSettings(_env_file=None)

    with pytest.raises(ValueError, match="do not select an access policy"):
        settings.to_config()


def test_settings_load_nested_environment_configuration(monkeypatch) -> None:
    monkeypatch.setenv("CAYU_SERVER_DEPLOYMENT_NAME", "preprod-eu")
    monkeypatch.setenv("CAYU_SERVER_ACCESS__MODE", "open")
    monkeypatch.setenv("CAYU_SERVER_DOCS__ENABLED", "true")
    monkeypatch.setenv(
        "CAYU_SERVER_CORS__ALLOWED_ORIGINS",
        '["https://control.example.com"]',
    )
    monkeypatch.setenv("CAYU_SERVER_LIFECYCLE__REPLAY_IDLE_TIMEOUT_S", "45")

    config = ServerSettings(_env_file=None).to_config()

    assert config.deployment_name == "preprod-eu"
    assert isinstance(config.access, OpenAccess)
    assert config.docs.enabled is True
    assert config.cors.allowed_origins == ("https://control.example.com",)
    assert config.lifecycle.replay_idle_timeout_s == 45.0


def test_settings_allow_nested_runtime_configuration_keys(monkeypatch) -> None:
    monkeypatch.setenv("CAYU_SERVER_ACCESS__MODE", "open")
    monkeypatch.setenv(
        "CAYU_SERVER_DASHBOARD__RUNTIME_CONFIG__FEATURES__REVIEW",
        "enabled",
    )

    settings = ServerSettings(_env_file=None)

    assert settings.dashboard.runtime_config == {
        "features": {"review": "enabled"},
    }


def test_explicit_constructor_values_override_environment_and_dotenv(monkeypatch, tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "CAYU_SERVER_DEPLOYMENT_NAME=dotenv\n"
        "CAYU_SERVER_ACCESS__MODE=open\n"
        "CAYU_SERVER_DOCS__ENABLED=false\n"
    )
    monkeypatch.setenv("CAYU_SERVER_DEPLOYMENT_NAME", "environment")

    settings = ServerSettings(
        deployment_name="constructor",
        docs={"enabled": True},
        _env_file=dotenv,
    )
    config = settings.to_config()

    assert config.deployment_name == "constructor"
    assert config.docs.enabled is True
    assert isinstance(config.access, OpenAccess)


def test_environment_values_override_dotenv(monkeypatch, tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("CAYU_SERVER_DEPLOYMENT_NAME=dotenv\nCAYU_SERVER_ACCESS__MODE=open\n")
    monkeypatch.setenv("CAYU_SERVER_DEPLOYMENT_NAME", "environment")

    config = ServerSettings(_env_file=dotenv).to_config()

    assert config.deployment_name == "environment"


def test_valid_higher_priority_value_overrides_malformed_lower_value(
    monkeypatch,
    tmp_path,
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        'CAYU_SERVER_ACCESS={"mode":"basic","username":"operator",'
        '"password":{"token":"lower-secret"}}\n'
    )
    monkeypatch.setenv(
        "CAYU_SERVER_ACCESS",
        json.dumps(
            {
                "mode": "basic",
                "username": "operator",
                "password": "higher-secret",
            }
        ),
    )

    config = ServerSettings(_env_file=dotenv).to_config()

    assert isinstance(config.access, AuthenticatedAccess)
    assert isinstance(config.access.dependency, BasicAuth)
    assert config.access.dependency.password == "higher-secret"


def test_file_secrets_use_prefixed_nested_names_and_merge_with_environment(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("CAYU_SERVER_ACCESS__MODE", "basic")
    monkeypatch.setenv("CAYU_SERVER_ACCESS__USERNAME", "operator")
    (tmp_path / "CAYU_SERVER_ACCESS__PASSWORD").write_text("file-secret")
    (tmp_path / "UNRELATED_APPLICATION_SECRET").write_text("ignored")

    settings = ServerSettings(_env_file=None, _secrets_dir=tmp_path)
    config = settings.to_config()

    assert isinstance(config.access, AuthenticatedAccess)
    assert isinstance(config.access.dependency, BasicAuth)
    assert config.access.dependency.username == "operator"
    assert config.access.dependency.password == "file-secret"

    monkeypatch.setenv("CAYU_SERVER_ACCESS__PASSWORD", "environment-secret")
    overridden = ServerSettings(_env_file=None, _secrets_dir=tmp_path).to_config()
    assert overridden.access.dependency.password == "environment-secret"


def test_file_secrets_reject_unknown_prefixed_nested_names(tmp_path) -> None:
    (tmp_path / "CAYU_SERVER_DASHBOARD__ENABELD").write_text("must-not-appear")

    with pytest.raises(ValidationError, match="enabeld") as exc_info:
        ServerSettings(
            access={"mode": "open"},
            dashboard=DashboardSettings(path="/ops"),
            _env_file=None,
            _secrets_dir=tmp_path,
        )
    assert "must-not-appear" not in str(exc_info.value)
    assert "must-not-appear" not in repr(exc_info.value.errors())


def test_file_secrets_redact_unknown_prefixed_root_values(tmp_path) -> None:
    (tmp_path / "CAYU_SERVER_ACCES__PASSWORD").write_text("top-secret")

    with pytest.raises(ValidationError, match="acces") as exc_info:
        ServerSettings(
            access={"mode": "open"},
            _env_file=None,
            _secrets_dir=tmp_path,
        )

    assert "top-secret" not in str(exc_info.value)
    assert "top-secret" not in repr(exc_info.value.errors())
    assert exc_info.value.errors()[0]["input"] is None


@pytest.mark.parametrize("source_kind", ["environment", "dotenv", "file_secret"])
def test_settings_redact_nested_suffixes_beneath_secret_fields(
    monkeypatch,
    tmp_path,
    source_kind: str,
) -> None:
    variable = "CAYU_SERVER_ACCESS__PASSWORD__TYPO"
    secret = "top-secret"
    settings_options = {"_env_file": None}

    if source_kind == "environment":
        monkeypatch.setenv(variable, secret)
    elif source_kind == "dotenv":
        dotenv = tmp_path / ".env"
        dotenv.write_text(f"{variable}={secret}\n")
        settings_options["_env_file"] = dotenv
    else:
        secrets_directory = tmp_path / "secrets"
        secrets_directory.mkdir()
        (secrets_directory / variable).write_text(secret)
        settings_options["_secrets_dir"] = secrets_directory

    with pytest.raises(ValidationError, match="typo") as exc_info:
        ServerSettings(access={"mode": "open"}, **settings_options)

    errors = exc_info.value.errors()
    assert errors[0]["loc"] == ("access", "password", "typo")
    assert errors[0]["input"] is None
    assert secret not in str(exc_info.value)
    assert secret not in repr(errors)


def test_settings_redact_invalid_values_supplied_as_whole_field_json(monkeypatch) -> None:
    secret = "top-secret"
    monkeypatch.setenv(
        "CAYU_SERVER_ACCESS",
        json.dumps({"mode": "basic", "password": {"token": secret}}),
    )

    with pytest.raises(ValidationError) as exc_info:
        ServerSettings(_env_file=None)

    errors = exc_info.value.errors()
    assert errors[0]["loc"] == ("access", "password")
    assert errors[0]["input"] is None
    assert secret not in str(exc_info.value)
    assert secret not in repr(errors)


@pytest.mark.parametrize(
    "contents",
    [
        b'{"mode":"basic","password":"top-secret"',
        b"\xfftop-secret",
    ],
)
def test_file_secrets_redact_parse_failures(tmp_path, contents: bytes) -> None:
    secret_file = tmp_path / "CAYU_SERVER_ACCESS"
    secret_file.write_bytes(contents)

    with pytest.raises(SettingsError, match="file secrets") as exc_info:
        ServerSettings(_env_file=None, _secrets_dir=tmp_path)

    assert "top-secret" not in str(exc_info.value)
    assert "top-secret" not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_dotenv_redacts_decoding_failures(tmp_path) -> None:
    secret = "DOTENV-SECRET"
    dotenv = tmp_path / ".env"
    dotenv.write_bytes(f"CAYU_SERVER_ACCESS__PASSWORD={secret}".encode() + b"\xff")

    with pytest.raises(SettingsError, match="dotenv file") as exc_info:
        ServerSettings(_env_file=dotenv)

    assert secret not in str(exc_info.value)
    assert secret not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_file_secrets_use_effective_prefix_and_nested_delimiter(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ALT_ACCESS___MODE", "basic")
    monkeypatch.setenv("ALT_ACCESS___USERNAME", "operator")
    (tmp_path / "ALT_ACCESS___PASSWORD").write_text("file-secret")

    config = ServerSettings(
        _env_file=None,
        _env_prefix="ALT_",
        _env_nested_delimiter="___",
        _secrets_dir=tmp_path,
    ).to_config()

    assert isinstance(config.access, AuthenticatedAccess)
    assert isinstance(config.access.dependency, BasicAuth)
    assert config.access.dependency.password == "file-secret"


@pytest.mark.skipif(
    "env_nested_max_split" not in signature(EnvSettingsSource).parameters,
    reason="This pydantic-settings version predates env_nested_max_split.",
)
def test_file_secrets_use_effective_nested_max_split(tmp_path) -> None:
    (tmp_path / "CAYU_SERVER_DASHBOARD__ACCESS__MODE").write_text("basic")
    (tmp_path / "CAYU_SERVER_DASHBOARD__ACCESS__USERNAME").write_text("operator")
    (tmp_path / "CAYU_SERVER_DASHBOARD__ACCESS__PASSWORD").write_text("file-secret")

    with pytest.raises(ValidationError, match="access__mode"):
        ServerSettings(
            access={"mode": "open"},
            _env_file=None,
            _env_nested_max_split=1,
            _secrets_dir=tmp_path,
        )


def test_settings_reject_unknown_explicit_top_level_fields() -> None:
    with pytest.raises(ValidationError, match="apii"):
        ServerSettings(access={"mode": "open"}, apii={"enabled": False}, _env_file=None)


def test_settings_reject_unknown_cayu_environment_fields(monkeypatch) -> None:
    monkeypatch.setenv("CAYU_SERVER_ACCESS__MODE", "open")
    monkeypatch.setenv("CAYU_SERVER_APII__ENABLED", "false")

    with pytest.raises(ValidationError, match="apii"):
        ServerSettings(_env_file=None)


def test_settings_reject_unknown_nested_cayu_environment_fields(monkeypatch) -> None:
    monkeypatch.setenv("CAYU_SERVER_DASHBOARD__ENABELD", "false")

    with pytest.raises(ValidationError, match="enabeld"):
        ServerSettings(
            access={"mode": "open"},
            dashboard=DashboardSettings(path="/ops"),
            _env_file=None,
        )


@pytest.mark.parametrize("field_name", ["TITLE", "DEPLOYMENT_NAME"])
def test_settings_reject_nested_suffixes_on_scalar_environment_fields(
    monkeypatch,
    field_name: str,
) -> None:
    monkeypatch.setenv("CAYU_SERVER_ACCESS__MODE", "open")
    monkeypatch.setenv(f"CAYU_SERVER_{field_name}__TYPO", "ignored")

    with pytest.raises(ValidationError, match="typo"):
        ServerSettings(_env_file=None)


def test_settings_reject_unknown_cayu_dotenv_fields_but_ignore_unrelated_keys(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "CAYU_SERVER_ACCESS__MODE=open\n"
        "CAYU_SERVER_DASHBORD__ENABLED=false\n"
        "UNRELATED_APPLICATION_SETTING=value\n"
    )

    with pytest.raises(ValidationError, match="dashbord"):
        ServerSettings(_env_file=dotenv)


def test_settings_reject_nested_suffixes_on_scalar_dotenv_fields(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("CAYU_SERVER_ACCESS__MODE=open\nCAYU_SERVER_TITLE__TYPO=ignored\n")

    with pytest.raises(ValidationError, match="typo"):
        ServerSettings(_env_file=dotenv)


def test_settings_reject_blank_dashboard_directory_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("CAYU_SERVER_ACCESS__MODE", "open")
    monkeypatch.setenv("CAYU_SERVER_DASHBOARD__DIRECTORY", "")

    with pytest.raises(ValidationError, match="dashboard.directory"):
        ServerSettings(_env_file=None)


def test_settings_allow_unrelated_keys_in_a_shared_dotenv(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("CAYU_SERVER_ACCESS__MODE=open\nUNRELATED_APPLICATION_SETTING=value\n")

    config = ServerSettings(_env_file=dotenv).to_config()

    assert isinstance(config.access, OpenAccess)


def test_settings_ignore_unprefixed_known_fields_from_dotenv(
    monkeypatch,
    tmp_path,
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("CAYU_SERVER_ACCESS__MODE=open\nTITLE=unrelated-application\n")
    original_call = DotEnvSettingsSource.__call__

    def emulate_older_dotenv_source(source):
        data = original_call(source)
        data["title"] = "unrelated-application"
        return data

    monkeypatch.setattr(DotEnvSettingsSource, "__call__", emulate_older_dotenv_source)

    config = ServerSettings(_env_file=dotenv).to_config()

    assert config.title == "Cayu"


def test_basic_access_secret_is_redacted_and_resolves_to_existing_auth_contract() -> None:
    settings = ServerSettings(
        access=ServerAccessSettings(
            mode="basic",
            username="operator",
            password=SecretStr("real-password"),
        ),
        _env_file=None,
    )

    config = settings.to_config()

    assert isinstance(config.access, AuthenticatedAccess)
    assert isinstance(config.access.dependency, BasicAuth)
    assert "real-password" not in repr(settings)
    assert "real-password" not in settings.model_dump_json()
    assert "real-password" not in json.dumps(config.safe_summary())


def test_direct_settings_models_redact_malformed_access_secrets() -> None:
    secret = "top-secret"
    invalid_settings = [
        lambda: ServerAccessSettings(
            mode="basic",
            username="operator",
            password={"token": secret},
        ),
        lambda: ServerAccessSettings(
            mode="basic",
            username=f" {secret} ",
        ),
        lambda: DashboardSettings(
            access={
                "mode": "basic",
                "username": "operator",
                "password": {"token": secret},
            }
        ),
        lambda: DashboardSettings(access=secret),
    ]

    for construct in invalid_settings:
        with pytest.raises(ValidationError) as exc_info:
            construct()
        errors = exc_info.value.errors()
        assert all(error["input"] is None for error in errors)
        assert all("ctx" not in error for error in errors)
        assert secret not in str(exc_info.value)
        assert secret not in repr(errors)


def test_settings_redaction_preserves_safe_validation_diagnostics() -> None:
    invalid_settings = [
        (
            lambda: ServerSettings(
                access={"mode": "open"},
                api={"enabled": "not-a-boolean"},
                _env_file=None,
            ),
            "bool_parsing",
            "valid boolean",
        ),
        (
            lambda: DashboardSettings(directory=" "),
            "value_error",
            "cannot be blank",
        ),
        (
            lambda: ServerAccessSettings(
                mode="basic",
                username="operator",
                password={"token": "top-secret"},
            ),
            "string_type",
            "valid string",
        ),
    ]

    for construct, error_type, message in invalid_settings:
        with pytest.raises(ValidationError) as exc_info:
            construct()
        error = exc_info.value.errors()[0]
        assert error["type"] == error_type
        assert message in error["msg"]
        assert error["input"] is None


@pytest.mark.parametrize(
    "settings_model", [ServerAccessSettings, DashboardSettings, ServerSettings]
)
@pytest.mark.parametrize(
    "payload",
    [
        '{"password":"top-secret"',
        b'{"password":"top-secret"',
        bytearray(b'{"password":"top-secret"'),
    ],
    ids=["text", "bytes", "bytearray"],
)
def test_settings_json_parse_errors_redact_the_source_payload(
    settings_model,
    payload,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        settings_model.model_validate_json(payload)

    errors = exc_info.value.errors()
    assert errors == [
        {
            "type": "json_invalid",
            "loc": (),
            "msg": "Invalid JSON for server settings.",
            "input": None,
        }
    ]
    assert "top-secret" not in str(exc_info.value)
    assert "top-secret" not in repr(errors)
    assert "top-secret" not in exc_info.value.json()
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_settings_json_validation_preserves_parsed_model_diagnostics() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ServerAccessSettings.model_validate_json(
            '{"mode":"basic","username":"operator","password":{"token":"top-secret"}}'
        )

    error = exc_info.value.errors()[0]
    assert error["type"] == "string_type"
    assert "valid string" in error["msg"]
    assert error["input"] is None
    assert "top-secret" not in repr(exc_info.value.errors())

    settings = ServerSettings.model_validate_json('{"access":{"mode":"open"}}')
    assert settings.access == ServerAccessSettings(mode="open")


def test_invalid_access_policy_errors_do_not_reveal_passwords() -> None:
    settings = ServerSettings(
        access={
            "mode": "open",
            "username": "operator",
            "password": "must-not-appear",
        },
        _env_file=None,
    )

    with pytest.raises(ValueError, match="cannot include Basic authentication fields") as exc_info:
        settings.to_config()

    assert "must-not-appear" not in str(exc_info.value)


def test_external_auth_can_be_resolved_before_constructing_runtime_config() -> None:
    settings = ServerSettings(
        deployment_name="production-eu",
        access={"mode": "external"},
        _env_file=None,
    )
    external = AuthenticatedAccess(dependency=_external_auth)

    config = settings.to_config(access=external)

    assert config.access is external
    assert config.deployment_name == "production-eu"


def test_explicit_access_cannot_silently_override_open_or_basic_settings() -> None:
    settings = ServerSettings(access={"mode": "open"}, _env_file=None)

    with pytest.raises(ValueError, match="conflicts"):
        settings.to_config(access=AuthenticatedAccess(dependency=_external_auth))


def test_dashboard_access_can_inherit_or_resolve_independently() -> None:
    inherited = ServerSettings(access={"mode": "open"}, _env_file=None).to_config()
    assert inherited.dashboard.access is None

    settings = ServerSettings(
        access={"mode": "open"},
        dashboard={"access": {"mode": "external"}},
        _env_file=None,
    )
    dashboard_access = AuthenticatedAccess(dependency=_external_auth)
    config = settings.to_config(dashboard_access=dashboard_access)
    assert config.dashboard.access is dashboard_access


def test_access_settings_reject_partial_or_unused_credentials() -> None:
    invalid_settings = [
        (
            ServerAccessSettings(mode="basic", username="operator"),
            "requires username and password",
        ),
        (
            ServerAccessSettings(
                mode="open",
                username="operator",
                password=SecretStr("password"),
            ),
            "cannot include Basic authentication fields",
        ),
        (
            ServerAccessSettings(mode="basic", username="operator", password=SecretStr(" ")),
            "cannot be blank",
        ),
    ]
    for access, match in invalid_settings:
        with pytest.raises(ValueError, match=match):
            ServerSettings(access=access, _env_file=None).to_config()

    external = ServerSettings(
        access={"mode": "open"},
        dashboard={"access": {"mode": "external", "realm": "unused"}},
        _env_file=None,
    )
    with pytest.raises(ValueError, match="cannot include Basic authentication fields"):
        external.to_config(dashboard_access=AuthenticatedAccess(dependency=_external_auth))
