"""Native host configuration; no network or secret resolution required."""

import asyncio
import importlib
import json
import traceback
import warnings

import pytest

from cayu import LocalArtifactStore, github_connector_behavior_fingerprint
from cayu.cli.project import project_context
from cayu.vaults import LocalEnvVault
from tests.qualification.test_repository_maintenance_application import project as project
from tests.qualification.test_repository_maintenance_delivery_configuration import authority
from tests.qualification.test_repository_maintenance_github_intake import configuration


@pytest.fixture
def host_config(project, monkeypatch, tmp_path):
    git, github = authority(), configuration()
    github["security"]["connector_behavior_fingerprint"] = github_connector_behavior_fingerprint()
    host = {
        "owner": "fixture",
        "repository_name": "disposable",
        "api_base_url": "https://api.github.example",
    }
    monkeypatch.setenv("CAYU_MAINTENANCE_GIT_JSON", json.dumps(git))
    monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_JSON", json.dumps(github))
    monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_HOST_JSON", json.dumps(host))
    with project_context(project):
        module = importlib.import_module("integrations.maintenance_github_host")
        store = LocalArtifactStore(tmp_path / "artifacts", store_id="host-artifacts")
        yield module, store, github, host


def test_native_factory_is_lazy_fresh_and_secret_free(host_config, monkeypatch):
    module, store, github, host = host_config
    constructed = []
    original = module.build_github_connector

    async def forbidden(*args):
        pytest.fail("Setup resolved a credential")

    def observed(**kwargs):
        value = original(**kwargs)
        constructed.append(value)
        return value

    monkeypatch.setattr(LocalEnvVault, "resolve", forbidden)
    monkeypatch.setattr(module, "build_github_connector", observed)
    factory = module.configured_github_connector_factory(store)
    assert constructed == []
    # Captured validated authority is not the mutable environment on later calls.
    monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_HOST_JSON", "invalid")

    async def scenario():
        first, second = factory(), factory()
        try:
            assert first is not second and len(constructed) == 2
            for connector in (first, second):
                configured = connector.profile.repositories["github"]
                assert configured.owner == host["owner"]
                assert configured.name == host["repository_name"]
                assert configured.installation_id == github["installation_id"]
                assert configured.credentials.token.name == "github-token"
                assert configured.credentials.resolver is not None
        finally:
            assert await first.aclose(timeout_s=1) is True
            assert await second.aclose(timeout_s=1) is True

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "bad", ["missing", "extra", "owner", "embedded", "http", "profile", "behavior", "alias"]
)
def test_invalid_host_configuration_precedes_resource_creation(
    host_config, monkeypatch, caplog, capsys, bad
):
    module, store, github, host = host_config
    canary = "private-github-host-canary"
    if bad == "extra":
        host["token"] = canary
    elif bad == "owner":
        host["owner"] = "owner/" + canary
    elif bad == "embedded":
        host["api_base_url"] = f"https://user:{canary}@api.github.example"
    elif bad == "http":
        host["api_base_url"] = "http://api.github.example"
    elif bad == "profile":
        github["security"]["credential_profile_id"] = canary
    elif bad == "behavior":
        github["security"]["connector_behavior_fingerprint"] = "sha256:" + "0" * 64
    elif bad == "alias":
        github["repository_alias"] = "other"
    if bad == "missing":
        monkeypatch.delenv("CAYU_MAINTENANCE_GITHUB_HOST_JSON")
    else:
        monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_HOST_JSON", json.dumps(host))
    monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_JSON", json.dumps(github))

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid configuration constructed connector")

    monkeypatch.setattr(module, "build_github_connector", forbidden)
    with warnings.catch_warnings(record=True) as captured:
        with pytest.raises(ValueError, match="maintenance GitHub host configuration") as caught:
            module.configured_github_connector_factory(store)
        assert canary not in "".join(traceback.format_exception(caught.value))
    output = capsys.readouterr()
    assert not captured and canary not in caplog.text + output.out + output.err
