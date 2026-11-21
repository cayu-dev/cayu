"""`cayu cloud connectors tools` and `cayu cloud connections run-tool` over Cloud's member
routes for connector package tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.cli.test_cloud_api_keys import cloud as _cloud
from tests.unit.test_connector_package_tools import catalog, digest

from cayu.cli import _release_check, main, serve
from cayu.connections import _tool_catalog

cloud = _cloud

_CATALOG = ("GET", "/v1/connector-tools/sample")
_AGENTS = {"items": [{"id": "app-1", "name": "Sample Agent"}]}


@pytest.fixture
def project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'sample'\n")
    monkeypatch.chdir(root)
    monkeypatch.setattr(_release_check, "user_cache_directory", lambda: tmp_path / "cache")
    return root


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = main(["cloud", *arguments])
    return code, capsys.readouterr().out


def test_tools_print_a_toolsets_catalog_and_refresh_the_cache(
    cloud, project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({_CATALOG: httpx.Response(200, json=catalog())})

    code, out = _run(["connectors", "tools", "sample", "--json"], capsys)

    assert code == 0
    envelope = json.loads(out)
    assert envelope["operation"] == "connectors.tools"
    assert envelope["result"] == catalog()
    assert "saved" not in envelope
    cached = tmp_path / "cache" / "connector-tools" / "sample.json"
    assert json.loads(cached.read_text()) == catalog()
    assert not (project / ".cayu").exists()


def test_tools_print_a_table_and_save_the_snapshot_on_purpose(
    cloud, project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({_CATALOG: httpx.Response(200, json=catalog())})

    code, out = _run(["connectors", "tools", "sample", "--save"], capsys)

    assert code == 0
    snapshot = project / ".cayu" / "connector-tools" / "sample.json"
    assert json.loads(snapshot.read_text()) == catalog()
    lines = out.splitlines()
    assert lines[0] == (
        "sample 0.1.0 (example-connector-sample): connectors sample, sample-sandbox; "
        "default slot sample"
    )
    assert lines[2].split() == ["TOOL", "EFFECT", "SCOPES", "APPROVAL", "DESCRIPTION"]
    assert lines[3].split()[:4] == ["sample_list_items", "none", "items:read", "-"]
    assert "Post a message to {channel}." in lines[4]
    assert lines[-1] == f"Saved to {snapshot}."


def test_a_toolset_cloud_doesnt_offer_is_refused_in_its_words(
    cloud, project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            _CATALOG: httpx.Response(
                404,
                json={"detail": {"code": "connector_unavailable", "message": "No toolset sample."}},
            )
        }
    )

    code, out = _run(["connectors", "tools", "sample"], capsys)

    assert code != 0
    assert "connector_unavailable" in out
    assert "No toolset sample." in out


def test_run_tool_runs_on_the_agents_binding_with_the_tools_digest(
    cloud, project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = "/v1/applications/app-1/connections/sample-eu/tools/sample_post_message"
    answer: dict[str, Any] = {"result": {"content": "Posted.", "structured": None}}
    fake = cloud(
        {
            ("GET", "/v1/applications?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            _CATALOG: httpx.Response(200, json=catalog()),
            ("POST", run): httpx.Response(200, json=answer),
        }
    )
    arguments = tmp_path / "arguments.json"
    arguments.write_text(json.dumps({"channel": "C1", "text": "hello"}))

    code, out = _run(
        [
            "connections",
            "run-tool",
            "--application",
            "sample-agent",
            "sample-eu",
            "sample_post_message",
            "--arguments-file",
            str(arguments),
        ],
        capsys,
    )

    assert code == 0
    envelope = json.loads(out)
    assert envelope["operation"] == "connections.run-tool"
    assert envelope["result"] == answer
    (sent,) = (json.loads(request.content) for request in fake.requests if request.method == "POST")
    assert sent["arguments"] == {"channel": "C1", "text": "hello"}
    assert sent["digest"] == digest("sample_post_message")
    assert isinstance(sent["call_id"], str) and sent["call_id"]


def test_run_tool_reports_a_changed_tool(
    cloud, project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", "/v1/applications?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            _CATALOG: httpx.Response(200, json=catalog()),
            ("POST", "/v1/applications/app-1/connections/sample/tools/sample_list_items"): (
                httpx.Response(
                    409, json={"detail": {"code": "tool_changed", "message": "It changed."}}
                )
            ),
        }
    )
    arguments = tmp_path / "arguments.json"
    arguments.write_text("{}")

    code, out = _run(
        [
            "connections",
            "run-tool",
            "--application",
            "app-1",
            "sample",
            "sample_list_items",
            "--arguments-file",
            str(arguments),
        ],
        capsys,
    )

    assert code != 0
    assert "tool_changed" in out


def test_serve_says_once_which_cloud_and_agent_the_tools_come_from(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    origin = _tool_catalog.Origin(kind="member", cloud_url="https://cloud.example", agent="sample")
    read = {
        toolset: _tool_catalog.Loaded(
            toolset, _tool_catalog.catalog_from(catalog(toolset), toolset), origin
        )
        for toolset in ("sample", "other")
    }
    monkeypatch.setattr(_tool_catalog, "_read", read)

    serve._print_connector_tools()

    assert capsys.readouterr().out == (
        "Connector tools: sample 0.1.0, other 0.1.0 from https://cloud.example, Agent sample\n"
    )
    monkeypatch.setattr(_tool_catalog, "_read", {})
    serve._print_connector_tools()
    assert capsys.readouterr().out == ""


_SNAPSHOT_FACTORY = """import cayu.connections as connections
from cayu import AgentSpec, CayuApp
from cayu.storage import SQLiteSessionStore


def build_app():
    app = CayuApp(session_store=SQLiteSessionStore("runtime.db"), enable_logging=False)
    app.register_agent(
        AgentSpec(name="assistant", model="unset"), tools=connections.tools("sample")
    )
    return app
"""


def test_a_command_that_builds_the_app_in_its_own_loop_loads_snapshot_tools(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("fastapi")
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "snapshot_project:build_app"\n', encoding="utf-8"
    )
    (project / "snapshot_project.py").write_text(_SNAPSHOT_FACTORY, encoding="utf-8")
    snapshot = project / ".cayu" / "connector-tools" / "sample.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps(catalog()))
    monkeypatch.setenv("CAYU_CONNECTOR_TOOLS", "snapshot")
    monkeypatch.setattr(_tool_catalog, "_read", {})
    monkeypatch.syspath_prepend(str(project))
    sys.modules.pop("snapshot_project", None)

    code = main(
        [
            "session",
            "start",
            "--agent",
            "assistant",
            "--message",
            "Hello.",
            "--scripted-reply",
            "Hello back.",
            "--json",
        ]
    )

    started = json.loads(capsys.readouterr().out)
    assert code == 0, started
    assert started["session"]["status"] == "completed"
