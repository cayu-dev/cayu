"""`cayu cloud login --api-url`: a login and context of its own for each other install."""

from __future__ import annotations

import base64
import json
import stat
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_api import CloudApiClient
from cayu.cli._cloud_auth import (
    CloudAuthCredentials,
    CloudAuthStore,
    DeviceAuthorization,
    WorkOSDeviceAuthClient,
)

_PRODUCTION = "https://cloud.example.test"
_INSTALL = "https://staging-portal.example.test"


def _jwt(marker: str) -> str:
    def encode(value: dict[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    claims = {"exp": int(time.time() + 3600), "marker": marker}
    return f"{encode({'alg': 'RS256'})}.{encode(claims)}.signature"


def _marker(token: str) -> str:
    encoded = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))["marker"]


def _credentials(api_url: str, marker: str) -> CloudAuthCredentials:
    return CloudAuthCredentials(
        api_url=api_url,
        workos_api_hostname="api.workos.example.test",
        workos_client_id=f"client_{marker}",
        access_token=_jwt(marker),
        refresh_token=f"refresh-{marker}",
        expires_at=time.time() + 3600,
        organization_id="org_test",
        user_id="user_test",
    )


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A private config directory, with production signed in as the marker `production`."""

    for name in (
        "CAYU_CLOUD_API_KEY",
        "CAYU_CLOUD_API_KEY_FILE",
        "CAYU_CLOUD_API_URL",
        "CAYU_CLOUD_AUTH",
        "CAYU_CLOUD_CONTEXT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cayu-cloud" / "config.json"))
    monkeypatch.setattr(cloud_cli, "_PRODUCTION_API_URL", _PRODUCTION)
    CloudAuthStore().save(_credentials(_PRODUCTION, "production"))
    return tmp_path / "cayu-cloud"


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Each Cloud call as (API URL, the marker of the login it was signed with)."""

    calls: list[tuple[str, str]] = []

    def request(client: CloudApiClient, method: str, path: str, **_options: Any) -> dict:
        calls.append((client.api_url, _marker(client.api_key)))
        assert (method, path) == ("GET", "/v1/me")
        return {"authentication": "workos", "organization_id": "org_test", "user_id": "user_test"}

    monkeypatch.setattr(CloudApiClient, "request", request)
    return calls


def _browser_signs_in(monkeypatch: pytest.MonkeyPatch, marker: str) -> list[str]:
    """The device sign-in completes as the marker ``marker``; the API URLs it signed in to."""

    signed_in: list[str] = []

    def portal_config(client: WorkOSDeviceAuthClient) -> tuple[str, str]:
        signed_in.append(client.api_url)
        return f"client_{marker}", "api.workos.example.test"

    def poll(client: WorkOSDeviceAuthClient, *_args: object, **_kwargs: object):
        return _credentials(client.api_url, marker)

    monkeypatch.setattr(WorkOSDeviceAuthClient, "portal_config", portal_config)
    monkeypatch.setattr(
        WorkOSDeviceAuthClient,
        "authorize_device",
        lambda _client, **_kwargs: DeviceAuthorization(
            device_code="private-device-code",
            user_code="CAYU-TEST",
            verification_uri="https://auth.example.test/device",
            verification_uri_complete="https://auth.example.test/device?user_code=CAYU-TEST",
            expires_in=300,
            interval=1,
        ),
    )
    monkeypatch.setattr(WorkOSDeviceAuthClient, "poll_for_credentials", poll)
    return signed_in


def _login_install(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> dict:
    signed_in = _browser_signs_in(monkeypatch, "staging")
    assert main(["cloud", "login", "--no-browser", "--api-url", f"{_INSTALL}/"]) == 0
    assert signed_in == [_INSTALL]
    return json.loads(capsys.readouterr().out)["result"]


def test_login_to_an_install_saves_its_own_login_and_selects_its_context(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    cloud: list[tuple[str, str]],
) -> None:
    result = _login_install(monkeypatch, capsys)

    context = Path(result["context"])
    assert context.parent.parent == config / "installs"
    assert context.parent.name.startswith("staging-portal-example-test-")
    assert json.loads(context.read_text()) == {
        "api_url": _INSTALL,
        "login_file": "auth.json",
        "schema_version": 1,
        "status": "ready",
    }
    assert (result["api_url"], result["status"]) == (_INSTALL, "signed_in")
    install_login = CloudAuthStore(context.parent / "auth.json").load()
    assert install_login is not None
    assert (install_login.api_url, install_login.workos_client_id) == (_INSTALL, "client_staging")
    # Production's login is untouched.
    production = CloudAuthStore().load()
    assert production is not None
    assert production.api_url == _PRODUCTION

    assert main(["cloud", "whoami"]) == 0
    capsys.readouterr()
    assert main(["cloud", "context", "show"]) == 0
    shown = json.loads(capsys.readouterr().out)["result"]
    assert (shown["api_url"], shown["authentication"]) == (_INSTALL, "workos")
    assert main(["cloud", "context", "clear"]) == 0
    assert main(["cloud", "whoami"]) == 0
    assert main(["cloud", "context", "use", str(context)]) == 0
    assert main(["cloud", "whoami"]) == 0
    # The install's context signs in with its own login, production with the default one.
    assert cloud == [
        (_INSTALL, "staging"),  # login confirms the identity
        (_INSTALL, "staging"),
        (_PRODUCTION, "production"),
        (_INSTALL, "staging"),
    ]


def test_an_install_context_never_signs_in_with_another_cloud_s_login(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    cloud: list[tuple[str, str]],
) -> None:
    context = Path(_login_install(monkeypatch, capsys)["context"])
    # The install's login file now holds a login made for production.
    CloudAuthStore(context.parent / "auth.json").save(_credentials(_PRODUCTION, "production"))

    assert main(["cloud", "whoami"]) == 2

    error = json.loads(capsys.readouterr().out)["error"]
    assert error == {
        "category": "login_api_mismatch",
        "message": (
            "The saved login belongs to another Cayu Cloud; run "
            f"`cayu cloud login --api-url {_INSTALL}`."
        ),
    }
    assert cloud == [(_INSTALL, "staging")]


def test_logout_and_organization_switch_act_on_the_selected_install_login(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    cloud: list[tuple[str, str]],
) -> None:
    context = Path(_login_install(monkeypatch, capsys)["context"])

    assert main(["cloud", "logout"]) == 0

    result = json.loads(capsys.readouterr().out)["result"]
    assert (result["api_url"], result["signed_out"]) == (_INSTALL, True)
    assert CloudAuthStore(context.parent / "auth.json").load() is None
    assert CloudAuthStore().load() is not None
    assert main(["cloud", "organizations", "use", "org_demo"]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == {
        "category": "login_required",
        "message": f"Sign in first with `cayu cloud login --api-url {_INSTALL}`.",
    }
    assert main(["cloud", "whoami"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["category"] == "login_required"


def test_api_url_must_agree_with_an_explicit_context(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    tmp_path: Path,
) -> None:
    signed_in = _browser_signs_in(monkeypatch, "staging")
    key = tmp_path / "key"
    key.write_text("private-key\n")
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            {
                "api_key_file": str(key),
                "api_url": "https://other.example.test",
                "schema_version": 1,
                "status": "ready",
            }
        )
    )

    command = ["cloud", "--context", str(context), "login", "--api-url", _INSTALL]
    assert main(command) == 2

    assert json.loads(capsys.readouterr().out)["error"]["category"] == "invalid_input"
    assert signed_in == []


@pytest.mark.parametrize(
    "api_url",
    [
        "http://staging.example.test",
        "https://user:pw@x.test",
        "https://x.test/?a=1",
        # The API client refuses the name `localhost`, and a saved login refuses other loopback
        # addresses, so sign-in is refused before the browser step for either.
        "http://localhost:8000",
        "http://127.0.0.2:8000",
    ],
)
def test_api_url_must_be_a_plain_https_address(
    capsys: pytest.CaptureFixture[str], config: Path, api_url: str
) -> None:
    assert main(["cloud", "login", "--api-url", api_url]) == 2

    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "invalid_input"
    assert "must be an https:// URL" in error["message"]


@pytest.mark.parametrize("context", [{}, {"api_key_file": "key", "login_file": "auth.json"}])
def test_a_context_signs_in_with_exactly_one_credential(
    capsys: pytest.CaptureFixture[str], config: Path, tmp_path: Path, context: dict
) -> None:
    path = tmp_path / "context.json"
    path.write_text(
        json.dumps({**context, "api_url": _INSTALL, "schema_version": 1, "status": "ready"})
    )

    assert main(["cloud", "context", "use", str(path)]) == 2

    assert json.loads(capsys.readouterr().out)["error"]["category"] == "context_invalid"


def _cloud_answers(monkeypatch: pytest.MonkeyPatch, responses: dict[str, httpx.Response]) -> None:
    """WorkOS's and Cloud's answers to the sign-in's own requests, by URL."""

    def request(_client: WorkOSDeviceAuthClient, _method: str, url: str, **_kwargs: object):
        return responses[url]

    monkeypatch.setattr(WorkOSDeviceAuthClient, "_request", request)


def test_an_address_without_sign_in_settings_is_named(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Path
) -> None:
    api = "https://staging-api.example.test"
    _cloud_answers(
        monkeypatch,
        {f"{api}/v1/portal-config": httpx.Response(404, json={"detail": "Not Found"})},
    )

    assert main(["cloud", "login", "--no-browser", "--api-url", api]) == 2

    assert json.loads(capsys.readouterr().out)["error"] == {
        "category": "portal_config_unavailable",
        "message": (
            f"{api} didn't give its sign-in settings: GET /v1/portal-config answered HTTP 404. "
            "Use the install's Cayu Cloud address (where its portal opens), not its API host."
        ),
    }
    assert not (config / "installs").exists()


def test_workos_refusing_device_authorization_says_what_to_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Path
) -> None:
    _cloud_answers(
        monkeypatch,
        {
            f"{_INSTALL}/v1/portal-config": httpx.Response(
                200,
                json={
                    "authentication": "workos",
                    "workos_api_hostname": "api.workos.example.test",
                    "workos_client_id": "client_staging",
                },
            ),
            "https://api.workos.example.test/user_management/authorize/device": httpx.Response(
                400,
                json={
                    "error": "unauthorized_client",
                    "error_description": "Device authorization is not enabled <script>",
                },
            ),
        },
    )

    assert main(["cloud", "login", "--no-browser", "--api-url", _INSTALL]) == 2

    streams = capsys.readouterr()
    assert json.loads(streams.out)["error"] == {
        "category": "device_authorization_refused",
        "message": (
            f"WorkOS refused CLI sign-in to {_INSTALL} (HTTP 400, error `unauthorized_client`). "
            "Check that the install's WorkOS environment allows device authorization (CLI "
            "sign-in) for its client client_staging."
        ),
    }
    assert "<script>" not in streams.out + streams.err
    assert not (config / "installs").exists()


@pytest.mark.parametrize(
    "spelling",
    ["HTTPS://cloud.example.test", "https://Cloud.Example.TEST/", "https://cloud.example.test:443"],
)
def test_another_spelling_of_production_signs_in_to_production(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    cloud: list[tuple[str, str]],
    spelling: str,
) -> None:
    signed_in = _browser_signs_in(monkeypatch, "production-again")

    assert main(["cloud", "login", "--no-browser", "--api-url", spelling]) == 0

    result = json.loads(capsys.readouterr().out)["result"]
    assert (signed_in, result["api_url"], result["context"]) == ([_PRODUCTION], _PRODUCTION, None)
    assert not (config / "installs").exists()


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [
        (
            "https://BÜCHER.example.test:8443/cloud/",
            "https://xn--bcher-kva.example.test:8443/cloud",
        ),
        ("HTTP://127.0.0.1:80/", "http://127.0.0.1"),
        ("http://[::1]:8080", "http://[::1]:8080"),
    ],
)
def test_api_url_has_one_spelling(spelling: str, canonical: str) -> None:
    assert cloud_cli._install_api_url(spelling) == canonical


def test_a_context_url_follows_the_api_client_s_loopback_rule() -> None:
    assert cloud_cli._normalized_api_url("http://127.0.0.2:8000") == "http://127.0.0.2:8000"
    with pytest.raises(ValueError, match="HTTPS outside loopback"):
        cloud_cli._normalized_api_url("http://localhost:8000")


def test_an_install_s_directories_are_private(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    cloud: list[tuple[str, str]],
) -> None:
    context = Path(_login_install(monkeypatch, capsys)["context"])

    for directory in (config, config / "installs", context.parent):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((context.parent / "auth.json").stat().st_mode) == 0o600


def _select_missing_context(config: Path, tmp_path: Path) -> None:
    (config / "config.json").write_text(
        json.dumps({"active_context": str(tmp_path / "deleted.json"), "schema_version": 1})
    )


def test_logout_with_an_unreadable_selection_signs_out_of_production(
    capsys: pytest.CaptureFixture[str], config: Path, tmp_path: Path
) -> None:
    _select_missing_context(config, tmp_path)

    assert main(["cloud", "logout"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["result"]["api_url"] == _PRODUCTION
    assert output["result"]["signed_out"] is True
    assert "can't be read" in output["warnings"][0]
    assert CloudAuthStore().load() is None
    # An explicit context must still be readable.
    assert main(["cloud", "--context", str(tmp_path / "deleted.json"), "logout"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["category"] == "context_invalid"


def test_switching_organization_with_an_unreadable_selection_uses_production(
    capsys: pytest.CaptureFixture[str],
    config: Path,
    tmp_path: Path,
    cloud: list[tuple[str, str]],
) -> None:
    _select_missing_context(config, tmp_path)

    assert main(["cloud", "organizations", "use", "org_test"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["result"]["api_url"] == _PRODUCTION
    assert "can't be read" in output["warnings"][0]
    assert cloud == [(_PRODUCTION, "production")]
    # The production login clears the unreadable selection, as it clears any other.
    assert not (config / "config.json").exists()


@pytest.mark.parametrize("content", ["not a login\n", None])
def test_logout_keeps_a_login_file_that_is_not_this_install_s_login(
    capsys: pytest.CaptureFixture[str], config: Path, tmp_path: Path, content: str | None
) -> None:
    named = tmp_path / "named-file"
    if content is None:
        CloudAuthStore(named).save(_credentials(_PRODUCTION, "production"))
    else:
        named.write_text(content)
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            {"api_url": _INSTALL, "login_file": str(named), "schema_version": 1, "status": "ready"}
        )
    )

    assert main(["cloud", "--context", str(context), "logout"]) == 2

    assert json.loads(capsys.readouterr().out)["error"]["message"].endswith("it was kept.")
    assert named.exists()


def test_an_api_key_given_with_an_install_context_signs_in_to_that_install(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    tmp_path: Path,
    cloud: list[tuple[str, str]],
) -> None:
    _login_install(monkeypatch, capsys)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        CloudApiClient,
        "request",
        lambda client, *_args, **_kwargs: calls.append((client.api_url, client.api_key)) or {},
    )
    key = tmp_path / "key"
    key.write_text("key-from-file\n")

    monkeypatch.setenv("CAYU_CLOUD_API_KEY", "key-from-environment")
    assert main(["cloud", "whoami"]) == 0
    monkeypatch.delenv("CAYU_CLOUD_API_KEY")
    monkeypatch.setenv("CAYU_CLOUD_API_KEY_FILE", str(key))
    assert main(["cloud", "whoami"]) == 0
    monkeypatch.delenv("CAYU_CLOUD_API_KEY_FILE")
    assert main(["cloud", "--api-key-file", str(key), "whoami"]) == 0
    capsys.readouterr()

    assert calls == [
        (_INSTALL, "key-from-environment"),
        (_INSTALL, "key-from-file"),
        (_INSTALL, "key-from-file"),
    ]
    monkeypatch.setenv("CAYU_CLOUD_API_URL", _PRODUCTION)
    assert main(["cloud", "whoami"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["category"] == "context_api_mismatch"


def test_eval_commands_protect_an_install_context_s_login(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    cloud: list[tuple[str, str]],
) -> None:
    from cayu.cli._eval_curation import _cloud_files

    context = Path(_login_install(monkeypatch, capsys)["context"])

    files = _cloud_files(SimpleNamespace(application="my-agent", context=None))

    assert ("the Cloud login", context.parent / "auth.json") in files
    assert ("the Cloud context", context) in files


def test_login_to_an_install_replaces_a_selected_context_for_another_cloud(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    tmp_path: Path,
    cloud: list[tuple[str, str]],
) -> None:
    key = tmp_path / "key"
    key.write_text("private-key\n")
    other = tmp_path / "other.json"
    other.write_text(
        json.dumps(
            {
                "api_key_file": str(key),
                "api_url": "https://other.example.test",
                "schema_version": 1,
                "status": "ready",
            }
        )
    )
    assert main(["cloud", "context", "use", str(other)]) == 0
    capsys.readouterr()

    context = Path(_login_install(monkeypatch, capsys)["context"])

    selection = json.loads((config / "config.json").read_text())
    assert selection["active_context"] == str(context.resolve())


def test_a_refusal_shows_only_a_plain_client_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], config: Path
) -> None:
    _cloud_answers(
        monkeypatch,
        {
            f"{_INSTALL}/v1/portal-config": httpx.Response(
                200,
                json={
                    "authentication": "workos",
                    "workos_api_hostname": "api.workos.example.test",
                    "workos_client_id": "client_<script>" + "x" * 200,
                },
            ),
            "https://api.workos.example.test/user_management/authorize/device": httpx.Response(
                403, json={"message": "Forbidden"}
            ),
        },
    )

    assert main(["cloud", "login", "--no-browser", "--api-url", _INSTALL]) == 2

    message = json.loads(capsys.readouterr().out)["error"]["message"]
    assert message.endswith("(CLI sign-in) for its client.")
    assert "(HTTP 403)." in message
    assert "<script>" not in message


def test_login_with_a_context_whose_api_url_is_invalid_is_refused(
    capsys: pytest.CaptureFixture[str], config: Path, tmp_path: Path
) -> None:
    key = tmp_path / "key"
    key.write_text("private-key\n")
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            {
                "api_key_file": str(key),
                "api_url": "http://staging.internal",
                "schema_version": 1,
                "status": "ready",
            }
        )
    )

    assert main(["cloud", "--context", str(context), "login", "--no-browser"]) == 2

    assert json.loads(capsys.readouterr().out)["error"]["category"] == "context_invalid"


def test_logout_refuses_an_unreadable_environment_context(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CAYU_CLOUD_CONTEXT", str(tmp_path / "deleted.json"))

    assert main(["cloud", "logout"]) == 2

    assert json.loads(capsys.readouterr().out)["error"]["category"] == "context_invalid"
    assert CloudAuthStore().load() is not None


def test_logout_names_an_install_login_its_deleted_context_leaves(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    cloud: list[tuple[str, str]],
) -> None:
    context = Path(_login_install(monkeypatch, capsys)["context"])
    context.unlink()

    assert main(["cloud", "logout"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["result"]["api_url"] == _PRODUCTION
    assert str(context.resolve()) in output["warnings"][0]
    assert (
        f"That install's login is still at {context.resolve().parent / 'auth.json'}"
        in (output["warnings"][0])
    )
    assert (context.parent / "auth.json").exists()


def test_logout_deletes_a_corrupt_install_login(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    cloud: list[tuple[str, str]],
) -> None:
    context = Path(_login_install(monkeypatch, capsys)["context"])
    (context.parent / "auth.json").write_text("corrupt\n")

    assert main(["cloud", "logout"]) == 0

    assert json.loads(capsys.readouterr().out)["result"]["signed_out"] is True
    assert not (context.parent / "auth.json").exists()


def test_logout_keeps_another_install_s_login_named_by_a_hand_written_context(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    tmp_path: Path,
    cloud: list[tuple[str, str]],
) -> None:
    other = Path(_login_install(monkeypatch, capsys)["context"])
    login = other.parent / "auth.json"
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            {
                "api_url": "https://third.example.test",
                "login_file": str(login),
                "schema_version": 1,
                "status": "ready",
            }
        )
    )

    assert main(["cloud", "--context", str(context), "logout"]) == 2

    assert json.loads(capsys.readouterr().out)["error"]["category"] == "login_api_mismatch"
    assert CloudAuthStore(login).load() is not None


@pytest.mark.parametrize(
    "spelling",
    [
        "HTTPS://STAGING-PORTAL.example.test",
        "https://staging-portal.example.test/",
        "https://staging-portal.example.test:443",
    ],
)
def test_a_context_spelling_its_install_differently_still_uses_its_login(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: Path,
    tmp_path: Path,
    cloud: list[tuple[str, str]],
    spelling: str,
) -> None:
    login = Path(_login_install(monkeypatch, capsys)["context"]).parent / "auth.json"
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            {"api_url": spelling, "login_file": str(login), "schema_version": 1, "status": "ready"}
        )
    )
    # The environment's URL is compared in the same spelling.
    monkeypatch.setenv("CAYU_CLOUD_API_URL", f"{_INSTALL.upper()}/")

    assert main(["cloud", "context", "use", str(context)]) == 0
    assert main(["cloud", "whoami"]) == 0

    capsys.readouterr()
    assert cloud[-1] == (_INSTALL, "staging")


def test_logout_keeps_the_production_login_a_context_names_for_another_cloud(
    capsys: pytest.CaptureFixture[str], config: Path, tmp_path: Path
) -> None:
    production = CloudAuthStore()
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            {
                "api_url": _INSTALL,
                "login_file": str(production.path),
                "schema_version": 1,
                "status": "ready",
            }
        )
    )

    assert main(["cloud", "--context", str(context), "logout"]) == 2

    assert json.loads(capsys.readouterr().out)["error"]["category"] == "login_api_mismatch"
    assert production.load() is not None
