"""`cayu cloud settings list|set|unset` over Cloud's settings API."""

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

_SETTINGS = "/v1/applications/cayu-scout/settings"
_LIST = ("GET", "/v1/applications?lifecycle=all")
_AGENTS = httpx.Response(200, json={"items": [{"id": "cayu-scout", "name": "Cayu Scout"}]})


def _setting(
    namespace: str, key: str, effective: object, *, source: str = "release", **extra: object
) -> dict[str, Any]:
    return {
        "namespace": namespace,
        "descriptor": {"key": key, "owner": "app", "type": "string", "label": key},
        "release_value": effective,
        "override": None if source == "release" else effective,
        "effective": effective,
        "source": source,
        "status": "applied",
        "updated_by": None,
        "updated_at": None,
        **extra,
    }


def _document(revision: int = 17, **changes: object) -> dict[str, Any]:
    return {
        "application_id": "cayu-scout",
        "release_id": "rel_one",
        "revision": revision,
        "agent_settings": "published",
        "settings": [
            _setting("platform", "web.sleep", 900),
            _setting("platform", "web.resources", {"cpu_millis": 512, "memory_mb": 1024}),
            _setting(
                "agent",
                "agent.reviewer.max_steps",
                60,
                source="cloud",
                reports={"adopted": 1, "refused": 1, "processes": 2, "refused_reasons": ["busy"]},
            ),
            # The same key in both namespaces needs a prefix.
            _setting("platform", "shared.flag", True),
            _setting("agent", "shared.flag", False),
        ],
        "conflicts": [{"namespace": "agent", "key": "agent.old.flag", "reason": "unknown_key"}],
        **changes,
    }


class _Cloud:
    def __init__(self, routes: dict[tuple[str, str], httpx.Response]) -> None:
        self.routes = {_LIST: _AGENTS, **routes}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.routes[(request.method, request.url.raw_path.decode())]

    def sent(self, method: str) -> list[object]:
        return [
            json.loads(request.content) if request.content else None
            for request in self.requests
            if request.method == method
        ]


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


def _json(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", "settings", *arguments, "--application", "Cayu Scout", "--json"])
    return code, json.loads(capsys.readouterr().out)


def _text(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(["cloud", "settings", *arguments, "--application", "cayu-scout"])
    streams = capsys.readouterr()
    return code, streams.out, streams.err


def test_cloud_help_lists_settings(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cloud", "settings", "--help"])

    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    for command in ("list", "set", "unset"):
        assert command in help_text


def test_list_prints_each_settings_value_source_and_status(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("GET", _SETTINGS): httpx.Response(200, json=_document())})

    code, out, err = _text(["list"], capsys)

    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[0] == "Agent cayu-scout, Release rel_one, revision 17"
    header = next(line for line in lines if line.startswith("SETTING"))
    assert header.split() == ["SETTING", "NAMESPACE", "VALUE", "SOURCE", "STATUS"]
    rows = {line.split()[0] + "/" + line.split()[1]: line.split() for line in lines[3:8]}
    assert rows["web.sleep/platform"] == ["web.sleep", "platform", "900", "release", "applied"]
    assert rows["web.resources/platform"][2] == '{"cpu_millis":512,"memory_mb":1024}'
    assert " ".join(rows["agent.reviewer.max_steps/agent"]) == (
        "agent.reviewer.max_steps agent 60 cloud applied (1 of 2 adopted, 1 refused)"
    )
    assert "agent.reviewer.max_steps refused: busy" in lines
    assert "  agent:agent.old.flag  unknown_key" in lines


def test_list_says_when_the_release_hasnt_published_its_settings(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _document(agent_settings="waiting_for_release", settings=[], conflicts=[])
    cloud({("GET", _SETTINGS): httpx.Response(200, json=document)})

    code, out, _ = _text(["list"], capsys)

    assert code == 0
    assert "Agent settings: waiting for the Release to start." in out.splitlines()
    assert "No settings." in out.splitlines()


def test_list_json_prints_the_cloud_envelope_with_the_document(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({("GET", _SETTINGS): httpx.Response(200, json=_document())})

    code, output = _json(["list"], capsys)

    assert code == 0
    assert output == {"ok": True, "operation": "settings.list", "result": _document()}
    assert fake.requests[-1].headers["Authorization"] == "Bearer login-token"


def test_set_sends_one_change_under_the_read_revision_with_each_keys_namespace(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", _SETTINGS): httpx.Response(200, json=_document()),
            ("PUT", _SETTINGS): httpx.Response(200, json=_document(18)),
        }
    )

    code, output = _json(
        [
            "set",
            "web.sleep=always_on",
            'web.resources={"cpu_millis": 1024, "memory_mb": 2048}',
            "agent.reviewer.max_steps=80",
            "platform:shared.flag=false",
            'agent:shared.flag="true"',
            "agent:agent.reviewer.label=NaN",
            "agent:agent.reviewer.limit=1e999",
        ],
        capsys,
    )

    assert code == 0
    (body,) = fake.sent("PUT")
    assert isinstance(body, dict)
    operation_id = body.pop("operation_id")
    assert operation_id.startswith("op_")
    assert body == {
        "revision": 17,
        "set": {
            "platform": {
                "web.sleep": "always_on",
                "web.resources": {"cpu_millis": 1024, "memory_mb": 2048},
                "shared.flag": False,
            },
            "agent": {
                "agent.reviewer.max_steps": 80,
                "shared.flag": "true",
                # Not JSON, so a string; an agent: prefix reaches a key the list hasn't got.
                "agent.reviewer.label": "NaN",
                "agent.reviewer.limit": "1e999",
            },
        },
        "clear": {"platform": [], "agent": []},
    }
    assert output["operation"] == "settings.set"
    assert output["operation_id"] == operation_id
    assert output["result"]["revision"] == 18
    assert {"key": "web.sleep", "namespace": "platform"} in output["set"]


def test_set_repeats_a_given_operation_id_and_prints_a_summary(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", _SETTINGS): httpx.Response(200, json=_document()),
            ("PUT", _SETTINGS): httpx.Response(200, json=_document(18)),
        }
    )

    code, out, _ = _text(["set", "web.sleep=3600", "--operation-id", "op_again"], capsys)

    assert code == 0
    assert fake.sent("PUT")[0]["operation_id"] == "op_again"  # type: ignore[index]
    assert out.splitlines()[:2] == ["Set platform:web.sleep.", "Operation op_again."]
    assert "Agent cayu-scout, Release rel_one, revision 18" in out


@pytest.mark.parametrize(
    ("assignment", "category", "message"),
    [
        ("shared.flag=1", "setting_ambiguous", "write platform:shared.flag or agent:shared.flag"),
        ("web.nope=1", "setting_unknown", "no setting named web.nope"),
        ("agent.reviewer.max_steps=1", "setting_repeated", "assigned more than once"),
    ],
)
def test_set_refuses_a_key_it_cant_place_before_writing(
    cloud, capsys: pytest.CaptureFixture[str], assignment: str, category: str, message: str
) -> None:
    fake = cloud({("GET", _SETTINGS): httpx.Response(200, json=_document())})
    arguments = ["set", assignment]
    if category == "setting_repeated":
        arguments.append("agent.reviewer.max_steps=2")

    code, output = _json(arguments, capsys)

    assert code == 2
    assert output["error"]["category"] == category
    assert message in output["error"]["message"]
    assert fake.sent("PUT") == []


def test_set_names_the_agent_prefix_while_the_release_hasnt_published_settings(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _document(agent_settings="waiting_for_release")
    cloud({("GET", _SETTINGS): httpx.Response(200, json=document)})

    code, output = _json(["set", "agent.scout.model=gpt"], capsys)

    assert code == 2
    assert "write agent:agent.scout.model" in output["error"]["message"]


@pytest.mark.parametrize("assignment", ["web.sleep", "Web.Sleep=1", "team:web.sleep=1"])
def test_set_refuses_a_malformed_assignment_without_a_request(
    cloud, capsys: pytest.CaptureFixture[str], assignment: str
) -> None:
    fake = cloud({})

    code, output = _json(["set", assignment], capsys)

    assert code == 2
    assert output["error"]["category"] in {"setting_assignment_invalid", "setting_key_invalid"}
    assert [request.url.path for request in fake.requests] == ["/v1/applications"]


def test_a_revision_conflict_says_nothing_changed_and_names_the_operation(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    conflict = {"detail": {"code": "settings_revision_conflict", "revision": 19}}
    cloud(
        {
            ("GET", _SETTINGS): httpx.Response(200, json=_document()),
            ("PUT", _SETTINGS): httpx.Response(409, json=conflict),
        }
    )

    code, output = _json(["set", "web.sleep=600", "--operation-id", "op_one"], capsys)

    assert code == 2
    error = output["error"]
    assert error["code"] == "settings_revision_conflict"
    assert error["revision"] == 19
    assert error["operation_id"] == "op_one"
    assert "changed while this command ran, so nothing was changed" in error["message"]


def test_a_validation_refusal_lists_each_settings_reason(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    refusal = {
        "detail": {
            "code": "invalid_setting",
            "errors": [
                {
                    "namespace": "platform",
                    "key": "web.resources",
                    "reason": "this install can't run that size",
                    "allowed": [{"cpu_millis": 512, "memory_mb": 1024}],
                },
                # Text that looks private is dropped, not shown.
                {"namespace": "agent", "key": "agent.x.y", "reason": "token=abc"},
                {"key": "Not A Key", "reason": "ignored"},
            ],
        }
    }
    cloud(
        {
            ("GET", _SETTINGS): httpx.Response(200, json=_document()),
            ("PUT", _SETTINGS): httpx.Response(422, json=refusal),
        }
    )

    code, output = _json(["set", 'web.resources={"cpu_millis": 99}'], capsys)

    assert code == 2
    assert output["error"]["code"] == "invalid_setting"
    assert output["error"]["message"] == "Cayu Cloud refused 2 settings, so nothing was changed."
    assert output["error"]["errors"] == [
        {
            "namespace": "platform",
            "key": "web.resources",
            "reason": "this install can't run that size",
            "allowed": [{"cpu_millis": 512, "memory_mb": 1024}],
        },
        {"namespace": "agent", "key": "agent.x.y", "reason": "refused"},
    ]

    code, out, err = _text(["set", 'web.resources={"cpu_millis": 99}'], capsys)

    assert (code, out) == (2, "")
    assert err.splitlines()[:3] == [
        "Error: Cayu Cloud refused 2 settings, so nothing was changed.",
        "  platform:web.resources: this install can't run that size "
        '(allowed: [{"cpu_millis":512,"memory_mb":1024}])',
        "  agent:agent.x.y: refused",
    ]
    assert err.splitlines()[3].startswith("Operation op_")


def test_an_admin_only_setting_names_the_key(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    refusal = {"detail": {"code": "organization_admin_required", "key": "web.sleep"}}
    cloud(
        {
            ("GET", _SETTINGS): httpx.Response(200, json=_document()),
            ("PUT", _SETTINGS): httpx.Response(403, json=refusal),
        }
    )

    code, output = _json(["set", "web.sleep=always_on"], capsys)

    assert code == 2
    assert output["error"]["code"] == "organization_admin_required"
    assert output["error"]["key"] == "web.sleep"
    assert output["error"]["message"].startswith("Only an organization admin can change web.sleep")


def test_unset_deletes_the_override_in_the_keys_namespace(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", _SETTINGS): httpx.Response(200, json=_document()),
            ("DELETE", f"{_SETTINGS}/agent/agent.reviewer.max_steps"): httpx.Response(
                200, json=_document(18)
            ),
            # A stored value that conflicts with the Release can still be cleared.
            ("DELETE", f"{_SETTINGS}/agent/agent.old.flag"): httpx.Response(
                200, json=_document(19, conflicts=[])
            ),
            ("DELETE", f"{_SETTINGS}/platform/shared.flag"): httpx.Response(
                200, json=_document(20)
            ),
        }
    )

    code, output = _json(["unset", "agent.reviewer.max_steps"], capsys)
    assert code == 0
    assert output["operation"] == "settings.unset"
    assert output["unset"] == [{"key": "agent.reviewer.max_steps", "namespace": "agent"}]
    assert output["result"]["revision"] == 18

    code, out, _ = _text(["unset", "agent.old.flag"], capsys)
    assert code == 0
    assert out.splitlines()[0] == "Unset agent:agent.old.flag."

    code, output = _json(["unset", "platform:shared.flag"], capsys)
    assert code == 0
    assert fake.sent("DELETE") == [None, None, None]


def test_unset_refuses_an_ambiguous_key(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud({("GET", _SETTINGS): httpx.Response(200, json=_document())})

    code, out, err = _text(["unset", "shared.flag"], capsys)

    assert (code, out) == (2, "")
    assert err.startswith("Error: shared.flag is both a platform and an agent setting")
    assert fake.sent("DELETE") == []


def test_an_invalid_settings_document_is_refused(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud({("GET", _SETTINGS): httpx.Response(200, json={"settings": []})})

    code, output = _json(["list"], capsys)

    assert code == 2
    assert output["error"]["category"] == "api_response_invalid"


def test_a_cloud_without_the_settings_api_is_named(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("GET", _SETTINGS): httpx.Response(404, json={"detail": "Not Found"})})

    code, output = _json(["list"], capsys)

    assert code == 2
    assert output["error"] == {
        "category": "settings_unsupported",
        "message": "This Cayu Cloud can't manage Agent settings yet.",
    }


def test_a_change_whose_answer_isnt_the_document_still_names_the_operation(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", _SETTINGS): httpx.Response(200, json=_document()),
            ("PUT", _SETTINGS): httpx.Response(200, json={"ok": True}),
        }
    )

    code, output = _json(["set", "web.sleep=600", "--operation-id", "op_lost"], capsys)

    assert code == 2
    assert output["error"]["category"] == "api_response_invalid"
    assert output["error"]["operation_id"] == "op_lost"
