"""`cayu cloud applications show`, `cayu cloud deployment list`, and the release reads that find
a release's Agent from its id alone."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient

_LIST = ("GET", "/v1/applications?lifecycle=all")
_AGENT = {"id": "my-agent", "name": "My Agent", "current_deployment_id": "dep_live"}
_AGENTS = httpx.Response(200, json={"items": [_AGENT]})
_RUNTIME = {
    "schema_version": 1,
    "version": "0.16.1",
    "read_from": "uv.lock",
    "lock_digest": "sha256:" + "a" * 64,
    "source": {"kind": "git", "url": "https://github.com/cayu-tech/cayu", "commit": "3f2a" * 10},
}


def _release(release_id: str, status: str, **extra: object) -> dict[str, Any]:
    return {
        "id": release_id,
        "application_id": "my-agent",
        "version": f"v-{release_id}",
        "status": status,
        "created_at": "2026-10-28T20:51:00+00:00",
        "updated_at": "2026-10-28T20:59:00+00:00",
        "manifest": {
            "source": {"repository": "https://github.com/acme/my-agent", "revision": "b" * 40}
        },
        "manifest_digest": "sha256:" + "c" * 64,
        "provenance": {"source_revision": "b" * 40},
        "runtime_artifact_id": "rta_one",
        "stage_only": False,
        "candidate": None,
        **extra,
    }


_LIVE = _release(
    "dep_live", "promoted", selected_at="2026-10-28T21:03:00+00:00", cayu_runtime=_RUNTIME
)
_BUILDING = _release("dep_next", "image_built", cayu_runtime=None)
_STAGED = _release("dep_staged", "smoke_tested", stage_only=True)
_OLD = _release("dep_old", "promoted", selected_at="2026-10-27T10:00:00+00:00")


class _Cloud:
    def __init__(self, routes: dict[tuple[str, str], httpx.Response]) -> None:
        self.routes = {_LIST: _AGENTS, **routes}
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.raw_path.decode())
        self.requests.append(" ".join(key))
        return self.routes.get(key) or httpx.Response(404, json={"detail": "Not Found"})


@pytest.fixture
def cloud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[tuple[str, str], httpx.Response]], _Cloud]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    def install(routes: dict[tuple[str, str], httpx.Response]) -> _Cloud:
        fake = _Cloud(routes)
        real_client = httpx.Client
        monkeypatch.setattr(
            _cloud_api.httpx,
            "Client",
            lambda **kwargs: real_client(transport=httpx.MockTransport(fake), **kwargs),
        )
        monkeypatch.setattr(
            cloud_cli,
            "_cloud_client",
            lambda *_args, **_kwargs: CloudApiClient(
                api_url="https://cloud.example", api_key="login-token"
            ),
        )
        return fake

    return install


def _page(*items: dict[str, Any], next_cursor: str | None = None) -> httpx.Response:
    return httpx.Response(200, json={"items": list(items), "next_cursor": next_cursor})


def _show_routes(*recent: dict[str, Any]) -> dict[tuple[str, str], httpx.Response]:
    return {
        ("GET", "/v1/applications/my-agent"): httpx.Response(
            200, json={**_AGENT, "lifecycle": "active", "revision": 4, "home": "/"}
        ),
        ("GET", "/v1/applications/my-agent/deployments/dep_live"): httpx.Response(200, json=_LIVE),
        ("GET", "/v1/applications/my-agent/deployments?limit=20"): _page(*recent),
    }


def test_show_prints_the_selected_release_and_what_is_pending(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(_show_routes(_BUILDING, _STAGED, _LIVE, _OLD))

    assert main(["cloud", "applications", "show", "My Agent"]) == 0
    out = capsys.readouterr().out
    assert "Selected release: dep_live (version v-dep_live, promoted)" in out
    assert "Selected at: 2026-10-28 21:03Z" in out
    assert f"Source: https://github.com/acme/my-agent @ {'b' * 40}" in out
    assert "Cayu runtime: cayu 0.16.1" in out
    assert "Cayu installed from: git https://github.com/cayu-tech/cayu @ 3f2a3f2a3f2a" in out
    pending = out.split("Candidate or in-flight releases:")[1]
    assert "dep_next" in pending and "in-flight" in pending
    assert "dep_staged" in pending and "candidate" in pending
    # Neither the selected release nor an earlier promoted one is pending.
    assert "dep_live" not in pending and "dep_old" not in pending
    assert "not recorded" in pending


def test_show_leaves_out_a_release_another_selection_superseded(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    superseded = _release("dep_waited", "smoke_tested", session_preflight={"state": "superseded"})
    cloud(_show_routes(superseded, _LIVE))

    assert main(["cloud", "applications", "show", "my-agent"]) == 0
    out = capsys.readouterr().out
    assert "dep_waited" not in out
    assert "Candidate or in-flight releases: none" in out


def test_show_json_carries_the_schema_version_and_release_summaries(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(_show_routes(_BUILDING, _LIVE))

    assert main(["cloud", "applications", "show", "my-agent", "--json"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["ok"] is True
    assert answer["operation"] == "applications.show"
    result = answer["result"]
    assert result["schema_version"] == 1
    assert result["application"]["id"] == "my-agent"
    assert result["selected_release"]["cayu_runtime"] == _RUNTIME
    assert result["selected_release"]["selected_at"] == "2026-10-28T21:03:00+00:00"
    assert result["selected_release"]["source_revision"] == "b" * 40
    assert [item["id"] for item in result["pending_releases"]] == ["dep_next"]
    assert result["pending_releases"][0]["pending"] == "in_flight"


def test_show_on_an_older_cloud_says_what_it_did_not_record(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    old_live = {
        key: value for key, value in _LIVE.items() if key not in {"selected_at", "cayu_runtime"}
    }
    routes = _show_routes(old_live)
    routes[("GET", "/v1/applications/my-agent/deployments/dep_live")] = httpx.Response(
        200, json=old_live
    )
    cloud(routes)

    assert main(["cloud", "applications", "show", "my-agent"]) == 0
    out = capsys.readouterr().out
    assert "Selected at: not recorded" in out
    assert "Cayu runtime: not recorded" in out
    assert "Candidate or in-flight releases: none" in out


def test_show_an_agent_without_a_selected_release(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    routes = _show_routes(_BUILDING)
    routes[("GET", "/v1/applications/my-agent")] = httpx.Response(
        200, json={**_AGENT, "current_deployment_id": None, "lifecycle": "active"}
    )
    cloud(routes)

    assert main(["cloud", "applications", "show", "my-agent"]) == 0
    out = capsys.readouterr().out
    assert "Selected release: none" in out
    assert "dep_next" in out


def test_list_prints_the_newest_releases_first(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", "/v1/applications/my-agent/deployments?limit=3"): _page(
                _BUILDING, _LIVE, _OLD, next_cursor="c1"
            )
        }
    )

    assert main(["cloud", "deployment", "list", "--application", "My Agent", "--limit", "3"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == [
        "RELEASE",
        "VERSION",
        "STATUS",
        "CREATED",
        "SELECTED",
        "SOURCE",
        "CAYU",
    ]
    assert [line.split()[0] for line in lines[1:4]] == ["dep_next", "dep_live", "dep_old"]
    assert "2026-10-28 21:03Z" in lines[2] and "cayu 0.16.1" in lines[2]
    assert "pass a larger --limit" in lines[-1]
    assert "GET /v1/applications/my-agent/deployments?limit=3" in fake.requests


# Cloud refused to publish the selected release; it stays promoted and selected.
_REFUSAL = {
    "acknowledge_sessions": ["s-input"],
    "code": "unfinished_sessions",
    "detail": "s-input is paused on a pending user_input. Nothing was changed.",
    "hint": (
        "Answer or finish them and retry, or accept that they may not resume: `cayu cloud "
        "deployment retry dep_live --application my-agent --acknowledge-session s-input`."
    ),
    "message": "1 unfinished session of the serving release may not resume on this Release.",
    "retryable": True,
}
_REFUSED = {**_LIVE, "publication_error": _REFUSAL}


def test_list_marks_a_release_cloud_refused_to_publish_with_its_reason_and_hint(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("GET", "/v1/applications/my-agent/deployments?limit=20"): _page(_REFUSED, _OLD)})

    assert main(["cloud", "deployment", "list", "--application", "my-agent"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "promoted, not published" in lines[1]
    assert "not published" not in lines[2]
    assert lines[3:] == [
        "",
        "dep_live:",
        f"  Not published: {_REFUSAL['message']}",
        f"  {_REFUSAL['hint']}",
    ]


def test_list_json_keeps_the_publication_error(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("GET", "/v1/applications/my-agent/deployments?limit=20"): _page(_REFUSED)})

    assert main(["cloud", "deployment", "list", "--application", "my-agent", "--json"]) == 0
    (item,) = json.loads(capsys.readouterr().out)["result"]["items"]
    assert item["status"] == "promoted"
    assert item["publication_error"] == _REFUSAL


def test_show_says_the_selected_release_was_not_published(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    routes = _show_routes(_REFUSED, _OLD)
    routes[("GET", "/v1/applications/my-agent/deployments/dep_live")] = httpx.Response(
        200, json=_REFUSED
    )
    cloud(routes)

    assert main(["cloud", "applications", "show", "my-agent"]) == 0
    out = capsys.readouterr().out
    assert "Selected release: dep_live (version v-dep_live, promoted, not published)" in out
    assert f"\n  Not published: {_REFUSAL['message']}\n  {_REFUSAL['hint']}\n" in out


def test_list_follows_pages_up_to_the_limit(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    first = [_release(f"dep_a{index}", "failed") for index in range(100)]
    second = [_release(f"dep_b{index}", "failed") for index in range(20)]
    cloud(
        {
            ("GET", "/v1/applications/my-agent/deployments?limit=100"): _page(
                *first, next_cursor="next"
            ),
            ("GET", "/v1/applications/my-agent/deployments?limit=20&cursor=next"): _page(
                *second, next_cursor="more"
            ),
        }
    )

    assert (
        main(
            ["cloud", "deployment", "list", "--application", "my-agent", "--limit", "120", "--json"]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["schema_version"] == 1
    assert len(result["items"]) == 120
    assert result["items"][-1]["id"] == "dep_b19"
    assert result["next_cursor"] == "more"


def test_list_rejects_a_limit_out_of_range(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["cloud", "deployment", "list", "--application", "a", "--limit", "0"]) == 2
    assert "--limit" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_status_finds_the_agent_from_the_release_id_alone(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", "/v1/deployments/dep_next"): httpx.Response(200, json=_BUILDING),
            ("GET", "/v1/applications/my-agent/deployments/dep_next"): httpx.Response(
                200, json=_BUILDING
            ),
        }
    )

    assert main(["cloud", "deployment", "status", "dep_next"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["result"]["id"] == "dep_next"
    # The Agent came from the release, not from listing every Agent.
    assert " ".join(_LIST) not in fake.requests


def test_timeline_and_logs_take_the_release_id_alone(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    base = "/v1/applications/my-agent/deployments/dep_next"
    cloud(
        {
            ("GET", "/v1/deployments/dep_next"): httpx.Response(200, json=_BUILDING),
            ("GET", f"{base}/timeline"): httpx.Response(200, json={"items": []}),
            ("GET", f"{base}/logs"): httpx.Response(200, json={"items": []}),
        }
    )

    assert main(["cloud", "deployment", "timeline", "dep_next"]) == 0
    assert json.loads(capsys.readouterr().out)["operation"] == "deployment.timeline"
    assert main(["cloud", "deployment", "logs", "dep_next"]) == 0
    assert json.loads(capsys.readouterr().out)["operation"] == "deployment.logs"


def test_wait_takes_the_release_id_alone(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", "/v1/deployments/dep_live"): httpx.Response(200, json=_LIVE),
            ("GET", "/v1/applications/my-agent/deployments/dep_live"): httpx.Response(
                200, json=_LIVE
            ),
        }
    )

    assert main(["cloud", "deployment", "wait", "dep_live", "--wait-seconds", "5"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["status"] == "promoted"


def test_a_missing_release_is_cloud_s_own_answer(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", "/v1/deployments/dep_gone"): httpx.Response(
                404,
                json={
                    "detail": {"code": "release_not_found", "message": "No Release has this id."}
                },
            )
        }
    )

    assert main(["cloud", "deployment", "status", "dep_gone"]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "release_not_found"


def test_an_older_cloud_finds_a_selected_release_and_asks_for_the_agent_otherwise(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {("GET", "/v1/applications/my-agent/deployments/dep_live"): httpx.Response(200, json=_LIVE)}
    )

    # No /v1/deployments route: the selected release is found from the Agent list.
    assert main(["cloud", "deployment", "status", "dep_live"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["id"] == "dep_live"

    assert main(["cloud", "deployment", "status", "dep_other"]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "release_lookup_unsupported"
    assert "--application" in error["message"]


def test_a_name_that_is_not_a_release_id_still_needs_the_agent(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({})

    assert main(["cloud", "deployment", "status", "latest"]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "invalid_input"
    assert "--application" in error["message"]


def test_promote_still_requires_the_agent(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["cloud", "deployment", "promote", "dep_live"]) == 2
    assert "--application" in json.loads(capsys.readouterr().out)["error"]["message"]
