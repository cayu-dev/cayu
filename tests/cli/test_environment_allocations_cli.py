"""`cayu environment allocations` lists and releases a crashed workflow run's sandbox."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from tests.cli.test_environment_cli import _http_bridge
from tests.environments.test_durable_shared_allocations import _World
from tests.server.test_shared_allocation_routes import crashed_world

from cayu.cli import main
from cayu.cli._cloud_agent_api import CayuApi
from cayu.server import ServerConfig, create_server

_ROOT_ENV = "CAYU_TEST_SHARED_ALLOCATION_ROOT"


def build_app():
    """The project factory `--target` names: an app over the test's world."""

    return _World.reopen(Path(os.environ[_ROOT_ENV])).app()


def _sandbox_names(world: _World) -> list[str]:
    return sorted(path.name for path in world.sandboxes.root.iterdir())


def test_list_and_discard_through_a_running_server(tmp_path: Path, monkeypatch, capsys) -> None:
    world = crashed_world(tmp_path, "run-cli")
    assert len(_sandbox_names(world)) == 1
    server = create_server(world.app(), config=ServerConfig.local_development())
    with TestClient(server) as client, _http_bridge(client) as http:
        monkeypatch.setattr(
            "cayu.cli._cloud_agent_api.CayuApi",
            lambda url, **kwargs: CayuApi(url, http=http, **kwargs),
        )
        target = ["--server-url", "http://127.0.0.1"]
        assert main(["environment", "allocations", "list", *target, "--json"]) == 0
        listing = json.loads(capsys.readouterr().out)
        assert listing["where"] == "server"
        (allocation,) = listing["allocations"]
        assert (allocation["workflow_run_id"], allocation["state"]) == ("run-cli", "live")
        allocation_id = allocation["allocation_id"]

        assert main(["environment", "allocations", "list", *target, "--table"]) == 0
        table = capsys.readouterr().out
        assert allocation_id in table and "running" in table

        def cloud(application, *, context, timeout_seconds):
            assert application == "demo"
            return CayuApi("http://127.0.0.1", http=http, timeout_seconds=timeout_seconds)

        monkeypatch.setattr("cayu.cli._cloud_agent_api.open_cloud_agent_api", cloud)
        on_cloud = ["environment", "allocations", "list", "--application", "demo", "--json"]
        assert main([*on_cloud, "--workflow-run", "run-cli", "--inactive-for", "0"]) == 0
        cloud_listing = json.loads(capsys.readouterr().out)
        assert cloud_listing["where"] == "cloud"
        assert [row["run_state"] for row in cloud_listing["allocations"]] == ["stopped"]

        release = ["environment", "allocations", "release", allocation_id, *target]
        # Refused while the run journaled something moments ago.
        assert main([*release, "--discard", "--json"]) == 1
        refused = json.loads(capsys.readouterr().out)
        assert refused["error"]["code"] == "SHARED_ALLOCATION_RELEASE_FAILED"
        assert "force" in refused["error"]["message"]
        assert len(_sandbox_names(world)) == 1

        assert main([*release, "--discard", "--inactive-for", "0", "--table"]) == 0
        assert "nothing was copied back" in capsys.readouterr().out
        assert _sandbox_names(world) == []

        assert main([*release, "--discard", "--inactive-for", "0", "--json"]) == 1
        assert "not found" in json.loads(capsys.readouterr().out)["error"]["message"]
        assert main(["environment", "allocations", "list", *target, "--table"]) == 0
        assert "No shared allocations are recorded." in capsys.readouterr().out


def test_copy_back_in_the_project_application(tmp_path: Path, monkeypatch, capsys) -> None:
    world = crashed_world(tmp_path, "run-local")
    monkeypatch.setenv(_ROOT_ENV, str(tmp_path))
    target = ["--target", "tests.cli.test_environment_allocations_cli:build_app"]
    assert main(["environment", "allocations", "list", *target, "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["where"] == "local"
    (allocation,) = listing["allocations"]
    assert allocation["created_at"] is not None
    assert "reconnect_metadata" not in allocation and "binding_state" not in allocation

    release = ["environment", "allocations", "release", allocation["allocation_id"], *target]
    assert main([*release, "--copy-back", "--force", "--json"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert (answer["mode"], answer["forced"]) == ("copy_back", True)
    assert (world.source / "plan.md").read_bytes() == b"replace a - b with a + b\n"
    assert _sandbox_names(world) == []


def test_release_needs_a_mode_and_rejects_mixed_targets(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["environment", "allocations", "release", "x.shared-0123456789abcdef"])
    assert error.value.code == 2
    capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        main(
            [
                "environment",
                "allocations",
                "release",
                "x.shared-0123456789abcdef",
                "--discard",
                "--copy-back",
            ]
        )
    assert error.value.code == 2
    capsys.readouterr()
    argv = ["environment", "allocations", "list", "--context", "c.json", "--json"]
    assert main(argv) == 1
    assert "--application" in json.loads(capsys.readouterr().out)["error"]["message"]
    with pytest.raises(SystemExit) as error:
        main(["environment", "allocations", "list", "--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--server-url" in help_text and "--application" in help_text
    assert "--target" in help_text
