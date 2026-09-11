"""Native host-only secret references; no remote transport is exercised here."""

import importlib
import json
import shutil
import warnings

import pytest

from cayu import LocalArtifactStore, RemoteGitHttpCredentials
from cayu.cli.project import project_context
from cayu.vaults import LocalEnvVault, SecretRef
from tests.qualification.test_repository_maintenance_application import project as project
from tests.qualification.test_repository_maintenance_delivery_configuration import authority


def arguments(tmp_path):
    return {
        "artifact_store": LocalArtifactStore(tmp_path / "artifacts", store_id="host-artifacts"),
        "broker_root": tmp_path / "broker",
        "git_executable": shutil.which("git") or "/usr/bin/git",
        "repository_id": "target",
        "broker_repository_id": "delivery",
        "remote_url": "https://github.com/example/disposable.git",
        "remote_identity": "target-github",
        "default_branch_ref": "refs/heads/main",
    }


def credentials():
    return RemoteGitHttpCredentials(
        credential_profile_id="maintenance-git-https",
        username=SecretRef(name="git-user"),
        password=SecretRef(name="git-token"),
        resolver=LocalEnvVault(
            {
                "git-user": "CAYU_MAINTENANCE_GIT_USER",
                "git-token": "CAYU_MAINTENANCE_GIT_TOKEN",
            }
        ),
    )


@pytest.mark.parametrize("profile", ["none", "maintenance-git-https"])
def test_named_host_configuration_constructs_without_resolving_secrets(
    project, tmp_path, monkeypatch, profile
):
    options = arguments(tmp_path)
    config = authority()
    config["security"]["credential_profile_id"] = profile
    monkeypatch.setenv("CAYU_MAINTENANCE_GIT_JSON", json.dumps(config))
    monkeypatch.setenv(
        "CAYU_MAINTENANCE_GIT_HOST_JSON",
        json.dumps(
            {key: str(options[key]) for key in ("broker_root", "git_executable", "remote_url")}
        ),
    )

    async def forbidden(*args, **kwargs):
        pytest.fail("Host configuration resolved a secret")

    monkeypatch.setattr(LocalEnvVault, "resolve", forbidden)
    with project_context(project):
        host = importlib.import_module("integrations.maintenance_git_host")
        broker = host.configured_git_broker(options["artifact_store"])
        remote = broker.profile.remotes["origin"]
        assert remote.credential_profile_id == profile
        assert remote.remote_identity == config["repository"]["remote_identity"]
        assert remote.egress_profile_id == config["security"]["egress_profile_id"]
        assert (remote.credentials is None) is (profile == "none")
        if remote.credentials is not None:
            assert remote.credentials.username.name == "git-user"
            assert remote.credentials.password.name == "git-token"
        assert list(broker.profile.root.iterdir()) == []


@pytest.mark.parametrize(
    "bad", ["missing", "extra", "relative", "type", "profile", "alias", "embedded"]
)
def test_named_host_rejects_before_directory_creation(
    project, tmp_path, monkeypatch, caplog, capsys, bad
):
    options = arguments(tmp_path)
    config = authority()
    config["security"]["credential_profile_id"] = "maintenance-git-https"
    value: dict[str, object] = {
        key: str(options[key]) for key in ("broker_root", "git_executable", "remote_url")
    }
    canary = "private-host-config-canary"
    if bad == "extra":
        value["token"] = canary
    elif bad == "relative":
        value["broker_root"] = "relative"
    elif bad == "type":
        value["git_executable"] = True
    elif bad == "profile":
        config["security"]["credential_profile_id"] = canary
    elif bad == "alias":
        config["repository"]["remote_alias"] = "other"
    elif bad == "embedded":
        value["remote_url"] = f"https://user:{canary}@github.com/example/disposable.git"
    monkeypatch.setenv("CAYU_MAINTENANCE_GIT_JSON", json.dumps(config))
    if bad == "missing":
        monkeypatch.delenv("CAYU_MAINTENANCE_GIT_HOST_JSON", raising=False)
    else:
        monkeypatch.setenv("CAYU_MAINTENANCE_GIT_HOST_JSON", json.dumps(value))
    with project_context(project), warnings.catch_warnings(record=True) as recorded:
        host = importlib.import_module("integrations.maintenance_git_host")
        with pytest.raises(ValueError, match="maintenance Git host configuration") as caught:
            host.configured_git_broker(options["artifact_store"])
        assert canary not in str(caught.value)
        assert not options["broker_root"].exists()
    output = capsys.readouterr()
    assert not recorded and canary not in caplog.text + output.out + output.err


def test_native_credentials_are_not_resolved_during_construction(project, tmp_path, monkeypatch):
    secret = "private-maintenance-git-token"
    monkeypatch.setenv("CAYU_MAINTENANCE_GIT_USER", "fixture-user")
    monkeypatch.setenv("CAYU_MAINTENANCE_GIT_TOKEN", secret)
    supplied = credentials()

    async def forbidden(*args, **kwargs):
        pytest.fail("Factory resolved a delivery credential")

    monkeypatch.setattr(supplied.resolver, "resolve", forbidden)
    with project_context(project):
        integration = importlib.import_module("integrations.remote_git")
        broker = integration.build_remote_git_delivery_broker(
            **arguments(tmp_path), credentials=supplied, egress_profile_id="github-git-only"
        )
        remote = broker.profile.remotes["origin"]
        assert remote.credentials is supplied
        assert remote.credential_profile_id == supplied.credential_profile_id
        assert remote.egress_profile_id == "github-git-only"
        assert remote.credentials.username.name == "git-user"
        assert remote.credentials.password.name == "git-token"
        assert secret not in repr(broker.profile)
        assert list(broker.profile.root.iterdir()) == []


def test_local_fixture_keeps_no_credential_default(project, tmp_path):
    with project_context(project):
        integration = importlib.import_module("integrations.remote_git")
        options = arguments(tmp_path)
        options["remote_url"] = str(tmp_path / "local-bare-repository")
        broker = integration.build_remote_git_delivery_broker(**options)
        remote = broker.profile.remotes["origin"]
        assert remote.credentials is None
        assert remote.credential_profile_id == "none"
        assert remote.egress_profile_id == "application-local"


@pytest.mark.parametrize("case", ["wrong-type", "local", "http", "embedded"])
def test_invalid_auth_configuration_precedes_broker_directory_creation(
    project, tmp_path, case, caplog, capsys
):
    secret = "private-maintenance-git-canary"

    class Hostile:
        def __repr__(self):
            return secret

    with project_context(project), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        integration = importlib.import_module("integrations.remote_git")
        options = arguments(tmp_path)
        options["credentials"] = credentials()
        if case == "wrong-type":
            options["credentials"] = Hostile()
        elif case == "local":
            options["remote_url"] = str(tmp_path / "local")
        elif case == "http":
            options["remote_url"] = "http://github.com/example/disposable.git"
        else:
            options["remote_url"] = f"https://user:{secret}@github.com/example/disposable.git"
        with pytest.raises(ValueError) as caught:
            integration.build_remote_git_delivery_broker(**options)
        assert secret not in str(caught.value) + repr(caught.value)
        assert not (tmp_path / "broker").exists()
    assert all(secret not in str(item.message) for item in captured)
    output = capsys.readouterr()
    assert secret not in caplog.text + output.out + output.err
