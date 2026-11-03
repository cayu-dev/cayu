"""An Agent remembers the Cayu Cloud and Organization it deploys to.

`tests/cli/conftest.py` answers the Organization lookup and moves the records aside for every
other CLI test; these tests put the real ones back and give the CLI a fake Cloud instead.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_agent_homes import (
    Organization,
    acting_organization,
    find_agent_home,
    remember_agent_home,
    require_agent_home,
)
from cayu.cli._cloud_api import CloudApiError

# The real lookup and records directory, before the conftest fixture replaces them.
_ACTING_ORGANIZATION = cloud_cli._acting_organization
_AGENT_HOMES_DIRECTORY = cloud_cli._agent_homes_directory

_PRODUCTION = "https://cloud.cayu.dev"
_STAGING = "https://staging-api.example.test"
_AGENT = "support-agent"
_ORGANIZATIONS = {
    "items": [
        {"current": True, "id": "org_cayu", "name": "Cayu", "role": "admin"},
        {"current": False, "id": "org_demo", "name": "Cayu Demo", "role": "admin"},
    ]
}


class _Cloud:
    """A Cayu Cloud whose login acts in ``organization``; it records every request."""

    def __init__(
        self,
        organization: str,
        *,
        api_url: str = _PRODUCTION,
        authentication: str = "workos",
        organizations: dict[str, Any] | CloudApiError = _ORGANIZATIONS,
        agents: list[dict[str, str]] | None = None,
    ) -> None:
        self.api_url = api_url
        self.agents = [{"id": _AGENT, "name": "Support Agent"}] if agents is None else agents
        self.organization = organization
        self.authentication = authentication
        self.organizations = organizations
        self.requests: list[tuple[str, str]] = []

    def request(self, method: str, path: str, **_kwargs: object) -> dict[str, Any]:
        self.requests.append((method, path))
        if path == "/v1/me":
            return {"authentication": self.authentication, "organization_id": self.organization}
        if path == "/v1/me/organizations":
            if isinstance(self.organizations, CloudApiError):
                raise self.organizations
            return self.organizations
        if path.startswith("/v1/applications?") or path == "/v1/applications":
            return {"items": self.agents}
        return {"status": "changed"}

    def changes(self) -> list[tuple[str, str]]:
        return [request for request in self.requests if request[0] != "GET"]


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The real Organization lookup and records, kept under this test's config directory."""

    monkeypatch.setattr(cloud_cli, "_acting_organization", _ACTING_ORGANIZATION)
    monkeypatch.setattr(cloud_cli, "_agent_homes_directory", _AGENT_HOMES_DIRECTORY)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cayu-cloud" / "config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    return tmp_path / "cayu-cloud"


@pytest.fixture
def deploys(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """`deploy` of a project whose manifest names `support-agent`, without the upload."""

    deployed: list[str] = []

    def deploy(_arguments: object, *, client: _Cloud, **_kwargs: object) -> dict[str, object]:
        deployed.append(client.organization)
        return {"operation": "deploy", "result": {"status": "succeeded"}}

    project = SimpleNamespace(
        root=None, manifest=SimpleNamespace(application=_AGENT, name="Support Agent")
    )
    monkeypatch.setattr(cloud_cli, "resolve_project", lambda *_args, **_kwargs: project)
    monkeypatch.setattr(cloud_cli, "_deploy", deploy)
    return deployed


def _use(monkeypatch: pytest.MonkeyPatch, cloud: _Cloud) -> _Cloud:
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: cloud)
    return cloud


def _cloud(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, *arguments: str
) -> tuple[int, dict[str, Any], str]:
    code = main(["cloud", "--evidence-dir", str(tmp_path / "evidence"), *arguments])
    streams = capsys.readouterr()
    return code, json.loads(streams.out), streams.err


def _record(config: Path, api_url: str, organization_id: str, name: str | None) -> None:
    warning = remember_agent_home(
        config, _AGENT, Organization(api_url=api_url, id=organization_id, name=name)
    )
    assert warning is not None


def test_the_first_deploy_records_where_the_agent_deploys_and_says_so_once(
    config: Path,
    deploys: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _use(monkeypatch, _Cloud("org_cayu"))

    code, output, err = _cloud(capsys, tmp_path, "deploy", "--no-wait")

    assert code == 0
    assert deploys == ["org_cayu"]
    assert err == "cayu cloud: acting in Organization Cayu (org_cayu) on https://cloud.cayu.dev\n"
    assert output["organization"] == {"api_url": _PRODUCTION, "id": "org_cayu", "name": "Cayu"}
    assert output["warnings"] == [
        "Recorded that Agent support-agent deploys to Organization Cayu (org_cayu) on "
        "https://cloud.cayu.dev. From now on a command that changes support-agent from another "
        "Organization or Cayu Cloud is refused unless it passes `--organization`."
    ]
    record = config / "agents" / f"{_AGENT}.json"
    payload = json.loads(record.read_text())
    assert {key: value for key, value in payload.items() if key != "recorded_at"} == {
        "api_url": _PRODUCTION,
        "application": _AGENT,
        "name": "Support Agent",
        "organization_id": "org_cayu",
        "organization_name": "Cayu",
        "schema_version": 1,
    }
    assert stat.S_IMODE(record.stat().st_mode) == 0o600
    assert stat.S_IMODE(record.parent.stat().st_mode) == 0o700

    code, output, _err = _cloud(capsys, tmp_path, "deploy", "--no-wait")

    assert code == 0
    assert deploys == ["org_cayu", "org_cayu"]
    assert "warnings" not in output


def test_a_deploy_from_another_organization_is_refused_before_anything_changes(
    config: Path,
    deploys: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record(config, _PRODUCTION, "org_cayu", "Cayu")
    cloud = _use(monkeypatch, _Cloud("org_demo"))

    code, output, err = _cloud(capsys, tmp_path, "deploy", "--no-wait")

    assert code == 2
    assert deploys == []
    assert cloud.changes() == []
    assert err == (
        "cayu cloud: acting in Organization Cayu Demo (org_demo) on https://cloud.cayu.dev\n"
    )
    assert output["error"] == {
        "category": "agent_organization_mismatch",
        "message": (
            "Agent support-agent deploys to Organization Cayu (org_cayu) on "
            "https://cloud.cayu.dev, but this command would act in Organization Cayu Demo "
            "(org_demo) on https://cloud.cayu.dev. Switch with `cayu cloud organizations use "
            "org_cayu` (or `cayu cloud login --organization org_cayu`), or pass `--organization "
            "org_demo` to act on support-agent in Cayu Demo anyway."
        ),
        "acting_organization": {"api_url": _PRODUCTION, "id": "org_demo", "name": "Cayu Demo"},
        "agent_organization": {"api_url": _PRODUCTION, "id": "org_cayu", "name": "Cayu"},
        "application": _AGENT,
    }


def test_a_deploy_to_another_cloud_names_the_sign_in_to_the_agents_own(
    config: Path,
    deploys: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The staging test Agent of the 27 October 2026 incident, deployed without its context.
    _record(config, _STAGING, "org_staging", "Cayu Staging")
    _use(monkeypatch, _Cloud("org_cayu"))

    code, output, _err = _cloud(capsys, tmp_path, "deploy", "--no-wait")

    assert code == 2
    assert deploys == []
    assert output["error"]["category"] == "agent_organization_mismatch"
    assert output["error"]["message"] == (
        "Agent support-agent deploys to Organization Cayu Staging (org_staging) on "
        "https://staging-api.example.test, but this command would act in Organization Cayu "
        "(org_cayu) on https://cloud.cayu.dev. Sign in there with `cayu cloud login --api-url "
        "https://staging-api.example.test --organization org_staging`, or pass `--organization "
        "org_cayu` to act on support-agent in Cayu anyway."
    )


def test_naming_the_organization_moves_the_agent_there(
    config: Path,
    deploys: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record(config, _PRODUCTION, "org_cayu", "Cayu")
    _use(monkeypatch, _Cloud("org_demo"))

    code, output, _err = _cloud(
        capsys, tmp_path, "deploy", "--no-wait", "--organization", "org_demo"
    )

    assert code == 0
    assert deploys == ["org_demo"]
    assert output["warnings"] == [
        "Agent support-agent now deploys to Organization Cayu Demo (org_demo) on "
        "https://cloud.cayu.dev; it deployed to Organization Cayu (org_cayu) on "
        "https://cloud.cayu.dev before."
    ]
    home = find_agent_home(config, _AGENT)
    assert home is not None
    assert home.organization == Organization(api_url=_PRODUCTION, id="org_demo", name="Cayu Demo")

    # Now a deploy from the first Organization is the one refused.
    _use(monkeypatch, _Cloud("org_cayu"))
    code, output, _err = _cloud(capsys, tmp_path, "deploy", "--no-wait")
    assert code == 2
    assert output["error"]["category"] == "agent_organization_mismatch"


def test_organization_names_where_the_command_must_act(
    config: Path,
    deploys: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del config
    _use(monkeypatch, _Cloud("org_demo"))

    code, output, _err = _cloud(
        capsys, tmp_path, "deploy", "--no-wait", "--organization", "org_cayu"
    )

    assert code == 2
    assert deploys == []
    assert output["error"]["category"] == "organization_mismatch"
    assert output["error"]["message"] == (
        "This command would act in Organization Cayu Demo (org_demo) on https://cloud.cayu.dev, "
        "not in Organization org_cayu. Switch with `cayu cloud organizations use org_cayu` (or "
        "`cayu cloud login --organization org_cayu`), then run it again."
    )
    # A refused deploy records nothing.
    assert not (tmp_path / "cayu-cloud" / "agents").exists()


@pytest.mark.parametrize(
    "command",
    [
        ["env", "set", "LOG_LEVEL=debug", "--application", _AGENT],
        ["env", "unset", "LOG_LEVEL", "--application", _AGENT],
        ["rollback", "dep_earlier", "--application", _AGENT],
        ["schedule", "run", "nightly", "--application", _AGENT],
        ["service", "restart", "--application", _AGENT],
        ["applications", "archive", _AGENT, "--expected-revision", "3"],
        # By the name it was deployed with, too.
        ["service", "pause", "--application", "Support Agent"],
    ],
)
def test_a_change_to_the_agent_from_another_organization_is_refused(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: list[str],
) -> None:
    remember_agent_home(
        config,
        _AGENT,
        Organization(api_url=_PRODUCTION, id="org_cayu", name="Cayu"),
        name="Support Agent",
    )
    cloud = _use(monkeypatch, _Cloud("org_demo"))

    code, output, err = _cloud(capsys, tmp_path, *command)

    assert code == 2
    assert output["error"]["category"] == "agent_organization_mismatch"
    assert cloud.changes() == []
    assert err.startswith("cayu cloud: acting in Organization Cayu Demo (org_demo)")


def test_a_change_says_where_it_acts_and_records_nothing(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cloud = _use(monkeypatch, _Cloud("org_demo"))

    code, output, err = _cloud(
        capsys, tmp_path, "env", "set", "LOG_LEVEL=debug", "--application", _AGENT
    )

    assert code == 0
    assert err == (
        "cayu cloud: acting in Organization Cayu Demo (org_demo) on https://cloud.cayu.dev\n"
    )
    assert output["organization"] == {
        "api_url": _PRODUCTION,
        "id": "org_demo",
        "name": "Cayu Demo",
    }
    assert cloud.changes()[-1] == ("PUT", f"/v1/applications/{_AGENT}/environment/LOG_LEVEL")
    # Only a deploy records where an Agent deploys.
    assert find_agent_home(config, _AGENT) is None


def test_a_change_with_the_organization_named_acts_on_the_agent_there(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record(config, _PRODUCTION, "org_cayu", "Cayu")
    cloud = _use(monkeypatch, _Cloud("org_demo"))

    code, _output, _err = _cloud(
        capsys,
        tmp_path,
        "service",
        "restart",
        "--application",
        _AGENT,
        "--organization",
        "org_demo",
    )

    assert code == 0
    assert ("POST", f"/v1/applications/{_AGENT}/service/restart") in cloud.changes()
    # Not a deploy, so the Agent still deploys where it did.
    home = find_agent_home(config, _AGENT)
    assert home is not None
    assert home.organization.id == "org_cayu"


def test_an_organization_change_says_where_it_acts(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del config
    _use(monkeypatch, _Cloud("org_cayu"))

    code, output, err = _cloud(capsys, tmp_path, "api-keys", "revoke", "key_1")

    assert code == 0
    assert err == "cayu cloud: acting in Organization Cayu (org_cayu) on https://cloud.cayu.dev\n"
    assert output["organization"]["id"] == "org_cayu"


def test_a_read_neither_asks_nor_says_where_it_acts(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record(config, _PRODUCTION, "org_cayu", "Cayu")
    cloud = _use(monkeypatch, _Cloud("org_demo"))

    code, output, err = _cloud(capsys, tmp_path, "env", "list", "--application", _AGENT)

    assert code == 0
    assert err == ""
    assert "organization" not in output
    assert not any(path.startswith("/v1/me") for _method, path in cloud.requests)


def test_an_unreadable_record_refuses_the_change(
    config: Path,
    deploys: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = config / "agents" / f"{_AGENT}.json"
    record.parent.mkdir(parents=True)
    record.write_text("{not json")
    _use(monkeypatch, _Cloud("org_cayu"))

    code, output, _err = _cloud(capsys, tmp_path, "deploy", "--no-wait")

    assert code == 2
    assert deploys == []
    assert output["error"]["category"] == "agent_record_invalid"
    assert output["error"]["path"] == str(record)


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs directory permissions"
)
def test_a_record_that_cant_be_written_leaves_the_deploy_succeeded(
    config: Path,
    deploys: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Records can be read there, and none written.
    agents = config / "agents"
    agents.mkdir(parents=True)
    agents.chmod(0o500)
    _use(monkeypatch, _Cloud("org_cayu"))

    try:
        code, output, _err = _cloud(capsys, tmp_path, "deploy", "--no-wait")
    finally:
        agents.chmod(0o700)

    assert code == 0
    assert deploys == ["org_cayu"]
    assert output["warnings"][0].startswith(
        "Agent support-agent deployed to Organization Cayu (org_cayu) on https://cloud.cayu.dev, "
        "but this machine couldn't record that"
    )


def test_the_organization_comes_from_cloud_with_its_name_when_a_person_can_list_it() -> None:
    cloud = _Cloud("org_demo")

    assert acting_organization(cloud, api_url=_PRODUCTION) == Organization(
        api_url=_PRODUCTION, id="org_demo", name="Cayu Demo"
    )


def test_an_api_key_names_its_organization_by_id_only() -> None:
    cloud = _Cloud("org_cayu", authentication="api_key")

    organization = acting_organization(cloud, api_url=_PRODUCTION)

    assert organization == Organization(api_url=_PRODUCTION, id="org_cayu", name=None)
    assert organization.describe() == "Organization org_cayu on https://cloud.cayu.dev"
    # Only a signed-in person can list their Organizations.
    assert cloud.requests == [("GET", "/v1/me")]


def test_an_organization_list_that_fails_leaves_only_the_id() -> None:
    cloud = _Cloud(
        "org_cayu", organizations=CloudApiError("api_unavailable", "down", status_code=503)
    )

    assert acting_organization(cloud, api_url=_PRODUCTION).name is None


def test_cloud_must_say_which_organization() -> None:
    cloud = _Cloud("")

    with pytest.raises(CloudApiError) as raised:
        acting_organization(cloud, api_url=_PRODUCTION)

    assert raised.value.category == "api_response_invalid"


def test_every_changing_command_takes_organization() -> None:
    import argparse

    parser = cloud_cli._build_parser()
    for path, change in cloud_cli._CHANGING_COMMANDS.items():
        command = parser
        for name in path:
            (choices,) = (
                action.choices
                for action in command._actions
                if isinstance(action, argparse._SubParsersAction)
            )
            command = choices[name]
        assert "--organization" in command._option_string_actions, path
        assert command.get_default("_cloud_change") == change, path


def test_any_name_cloud_resolves_for_the_agent_is_checked(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Here the Agent's display name is "Helper", which the record never saw.
    _record(config, _STAGING, "org_staging", "Cayu Staging")
    cloud = _use(monkeypatch, _Cloud("org_cayu", agents=[{"id": _AGENT, "name": "Helper"}]))

    for reference in ("Helper", "helper"):
        code, output, _err = _cloud(
            capsys, tmp_path, "env", "set", "LOG_LEVEL=debug", "--application", reference
        )

        assert code == 2
        assert output["error"]["category"] == "agent_organization_mismatch"
    assert cloud.changes() == []


def test_an_agent_cloud_doesnt_know_here_is_explained_by_its_record(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    remember_agent_home(
        config,
        _AGENT,
        Organization(api_url=_PRODUCTION, id="org_cayu", name="Cayu"),
        name="Support Agent",
    )
    _use(monkeypatch, _Cloud("org_demo", agents=[]))

    code, output, _err = _cloud(
        capsys, tmp_path, "service", "restart", "--application", "Support Agent"
    )
    assert code == 2
    assert output["error"]["category"] == "agent_organization_mismatch"

    # An Agent this machine knows nothing of fails as before.
    code, output, _err = _cloud(capsys, tmp_path, "service", "restart", "--application", "other")
    assert code == 2
    assert output["error"]["category"] == "application_not_found"


def test_an_unreadable_record_refuses_a_change_by_name_too(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = config / "agents" / f"{_AGENT}.json"
    record.parent.mkdir(parents=True)
    record.write_text("{not json")
    cloud = _use(monkeypatch, _Cloud("org_demo"))

    code, output, _err = _cloud(
        capsys, tmp_path, "env", "unset", "API_TOKEN", "--application", "Support Agent"
    )

    assert code == 2
    assert output["error"]["category"] == "agent_record_invalid"
    assert cloud.changes() == []


def test_an_organization_name_with_control_characters_is_never_printed(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record(config, _PRODUCTION, "org_cayu", "Cayu")
    hostile = {"items": [{"id": "org_evil", "name": "Evil\x1b]52;c;ZWNobw==\x07\x1b[2K"}]}
    _use(monkeypatch, _Cloud("org_evil", organizations=hostile))

    code = main(["cloud", "settings", "set", "x=1", "--application", _AGENT])

    streams = capsys.readouterr()
    assert code == 2
    assert "\x1b" not in streams.err
    assert "\x07" not in streams.err
    assert "Organization org_evil on https://cloud.cayu.dev" in streams.err


def test_an_organization_id_that_isnt_one_is_refused() -> None:
    with pytest.raises(CloudApiError) as raised:
        acting_organization(_Cloud("org\x1b[2K"), api_url=_PRODUCTION)

    assert raised.value.category == "api_response_invalid"


def test_a_record_with_control_characters_is_unreadable(config: Path) -> None:
    record = config / "agents" / f"{_AGENT}.json"
    record.parent.mkdir(parents=True)
    _record(config, _PRODUCTION, "org_cayu", "Cayu")
    payload = json.loads(record.read_text())
    record.write_text(json.dumps({**payload, "organization_name": "Cayu\x1b[2K"}))

    with pytest.raises(CloudApiError) as raised:
        require_agent_home(
            config,
            _AGENT,
            Organization(api_url=_PRODUCTION, id="org_demo", name=None),
            intended=None,
            login_command=str,
        )

    assert raised.value.category == "agent_record_invalid"


def test_organization_goes_before_or_after_the_command_and_into_its_follow_ups() -> None:
    parser = cloud_cli._build_parser()

    before = parser.parse_args(
        ["--organization", "org_demo", "rollback", "dep_1", "--application", _AGENT]
    )
    after = parser.parse_args(
        ["rollback", "dep_1", "--application", _AGENT, "--organization", "org_demo"]
    )

    assert before.intended_organization == after.intended_organization == "org_demo"
    # Follow-up commands put it right after `cayu cloud`, where every command takes it.
    assert cloud_cli._cloud_recovery_arguments(after) == ("--organization", "org_demo")
    assert parser.parse_args(["--organization", "org_demo", "deployment", "status", "dep_1"])


def test_a_read_with_organization_says_where_it_acts_and_checks_it(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del config
    _use(monkeypatch, _Cloud("org_demo"))

    code, output, err = _cloud(
        capsys, tmp_path, "--organization", "org_demo", "env", "list", "--application", _AGENT
    )
    assert code == 0
    assert output["organization"]["id"] == "org_demo"
    assert err.startswith("cayu cloud: acting in Organization Cayu Demo (org_demo)")

    code, output, _err = _cloud(
        capsys, tmp_path, "--organization", "org_cayu", "env", "list", "--application", _AGENT
    )
    assert code == 2
    assert output["error"]["category"] == "organization_mismatch"


def test_organization_is_refused_where_nothing_acts_in_cloud(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["cloud", "--organization", "org_demo", "login"]) == 2

    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "invalid_input"
    assert "cayu cloud login --organization ORG_ID" in error["message"]


def test_doctor_checks_organization_too(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del config
    _use(monkeypatch, _Cloud("org_cayu"))

    code, output, _err = _cloud(capsys, tmp_path, "--organization", "org_staging", "doctor")
    assert code == 2
    assert output["error"]["category"] == "organization_mismatch"

    code, output, _err = _cloud(capsys, tmp_path, "--organization", "org_cayu", "doctor")
    assert code == 0
    assert output["organization"]["id"] == "org_cayu"


def test_the_command_changes_the_agent_that_was_checked(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del config

    class Racing(_Cloud):
        """After the first listing, an Agent with the exact slug `support` appears."""

        listings = 0

        def request(self, method: str, path: str, **kwargs: object) -> dict[str, Any]:
            if path == "/v1/applications":
                self.listings += 1
                if self.listings > 1:
                    self.agents = [*self.agents, {"id": "support", "name": "Support"}]
            return super().request(method, path, **kwargs)

    cloud = _use(monkeypatch, Racing("org_cayu", agents=[{"id": "helper", "name": "Support"}]))

    code, _output, _err = _cloud(
        capsys, tmp_path, "env", "unset", "SECRET", "--application", "support"
    )

    assert code == 0
    assert cloud.changes() == [("DELETE", "/v1/applications/helper/environment/SECRET")]
