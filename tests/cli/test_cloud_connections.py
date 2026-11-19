"""`cayu cloud connectors` and `cayu cloud connections` over Cloud's connection routes."""

from __future__ import annotations

import _thread
import contextlib
import json
import re
import socket
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import httpx
import pytest
from tests.cli.test_cloud_api_keys import cloud as _cloud

from cayu.cli import _cloud_connections, _cloud_sign_in, main

cloud = _cloud

_APPS = "/v1/applications"
_CONNECTION = {"name": "team-anthropic", "connector": "anthropic", "status": "ready"}
_AGENTS = {"items": [{"id": "support-desk", "name": "Support desk"}]}


def _field(
    name: str, *, secret: bool = True, optional: bool = False, multiline: bool = False
) -> dict[str, Any]:
    return {
        "name": name,
        "title": name.replace("_", " ").capitalize(),
        "secret": secret,
        "optional": optional,
        "multiline": multiline,
        "pem": multiline,
        "hint": None,
    }


_SAP_SETTINGS = [
    _field("base_url", secret=False),
    _field("token_url", secret=False),
    _field("client_id", secret=False),
    _field("client_secret"),
    _field("oauth_scope", secret=False, optional=True),
    _field("sap_client", secret=False, optional=True),
    _field("ca_bundle", secret=False, optional=True, multiline=True),
]
#: Cloud's `GET /v1/connectors`, with each connector's credential fields.
_CONNECTORS = {
    "items": [
        {"id": "anthropic", "fields": [_field("api_key")]},
        {
            "id": "snowflake",
            "fields": [
                _field("account", secret=False),
                _field("user", secret=False),
                _field("private_key", multiline=True),
            ],
        },
        {
            "id": "github-app",
            "fields": [_field("app_id", secret=False), _field("private_key", multiline=True)],
        },
        {"id": "sap", "fields": _SAP_SETTINGS},
        {
            "id": "sap-certificate",
            "fields": [
                _field("base_url", secret=False),
                _field("client_certificate", secret=False, multiline=True),
                _field("client_key", multiline=True),
                _field("sap_client", secret=False, optional=True),
                _field("ca_bundle", secret=False, optional=True, multiline=True),
            ],
        },
    ]
}
_LIST_CONNECTORS = ("GET", "/v1/connectors")


def _sent(fake: Any) -> dict[str, Any]:
    """What the command sent Cloud: its one write."""

    (write,) = (request for request in fake.requests if request.method in {"POST", "PUT"})
    return json.loads(write.content)


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", *arguments])
    return code, json.loads(capsys.readouterr().out)


def test_add_reads_the_key_from_a_file_and_never_prints_it(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud(
        {
            ("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            ("POST", "/v1/connections"): httpx.Response(201, json=_CONNECTION),
        }
    )
    key_file = tmp_path / "anthropic-key"
    key_file.write_text("sk-ant-secret\n")

    code, output = _run(
        [
            "connections",
            "add",
            "anthropic",
            "--name",
            "team-anthropic",
            "--key-file",
            str(key_file),
            "--agent",
            "Support desk",
        ],
        capsys,
    )

    assert code == 0
    (post,) = (request for request in fake.requests if request.method == "POST")
    assert json.loads(post.content) == {
        "connector": "anthropic",
        "name": "team-anthropic",
        "credential": {"api_key": "sk-ant-secret"},
        "agents": ["support-desk"],
    }
    assert output["operation"] == "connections.add"
    assert "sk-ant-secret" not in json.dumps(output)


def test_a_key_pair_takes_its_secret_field_from_a_file(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud(
        {
            _LIST_CONNECTORS: httpx.Response(200, json=_CONNECTORS),
            ("POST", "/v1/connections"): httpx.Response(201, json=_CONNECTION),
        }
    )
    private_key = tmp_path / "snowflake.p8"
    private_key.write_text("-----BEGIN PRIVATE KEY-----\n")

    code, _ = _run(
        [
            "connections",
            "add",
            "snowflake",
            "--name",
            "warehouse",
            "--credential",
            "account=acme",
            "--credential",
            "user=cayu",
            "--credential",
            f"private_key=@{private_key}",
        ],
        capsys,
    )

    assert code == 0
    assert _sent(fake)["credential"] == {
        "account": "acme",
        "user": "cayu",
        "private_key": "-----BEGIN PRIVATE KEY-----",
    }


def test_an_unreadable_credential_file_asks_cloud_nothing(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud({})

    code, output = _run(
        [
            "connections",
            "add",
            "anthropic",
            "--name",
            "team-anthropic",
            "--key-file",
            str(tmp_path / "missing"),
        ],
        capsys,
    )

    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert "missing" not in output["error"]["message"]
    assert fake.requests == []


def test_a_secret_field_given_on_the_command_line_is_refused_unrepeated(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({_LIST_CONNECTORS: httpx.Response(200, json=_CONNECTORS)})

    for given, said in (
        ("private_key=sk-live-secret", "private_key is secret: give it from a file"),
        ("sk-live-secret", "FIELD=@PATH or FIELD=VALUE"),
        (
            "sk_live_secret=x",
            "isn't one the snowflake connector takes: account, user, private_key.",
        ),
    ):
        code, output = _run(
            ["connections", "add", "snowflake", "--name", "warehouse", "--credential", given],
            capsys,
        )
        assert code == 2
        assert said in output["error"]["message"]
        assert "sk-live-secret" not in json.dumps(output)
        assert "sk_live_secret" not in json.dumps(output)
    # Cloud was only asked which fields the connector takes.
    assert {(request.method, request.url.path) for request in fake.requests} == {_LIST_CONNECTORS}


def test_a_plain_field_cloud_describes_is_taken_plain(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A connector this CLI has no list for takes the plain fields Cloud says it has."""

    connectors = {
        "items": [
            {
                "id": "acme",
                "fields": [_field("region", secret=False), _field("client_secret")],
            }
        ]
    }
    fake = cloud(
        {
            _LIST_CONNECTORS: httpx.Response(200, json=connectors),
            ("POST", "/v1/connections"): httpx.Response(201, json=_CONNECTION),
        }
    )
    secret = tmp_path / "acme-secret"
    secret.write_text("acme-secret-value\n")

    code, _ = _run(
        [
            "connections",
            "add",
            "acme",
            "--name",
            "acme",
            "--credential",
            "region=eu",
            "--credential",
            f"client_secret=@{secret}",
        ],
        capsys,
    )

    assert code == 0
    assert _sent(fake)["credential"] == {"region": "eu", "client_secret": "acme-secret-value"}


def test_an_older_cloud_without_field_descriptions_uses_the_known_fields(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    older = {"items": [{"id": "snowflake", "credential_fields": ["account", "user"]}]}
    fake = cloud(
        {
            _LIST_CONNECTORS: httpx.Response(200, json=older),
            ("POST", "/v1/connections"): httpx.Response(201, json=_CONNECTION),
        }
    )

    for given, said in (
        ("private_key=sk-live-secret", "The credential field private_key is secret"),
        ("sk_live_secret=x", "The credential field FIELD is secret"),
    ):
        code, output = _run(
            ["connections", "add", "snowflake", "--name", "warehouse", "--credential", given],
            capsys,
        )
        assert code == 2
        assert said in output["error"]["message"]
        assert "sk_live_secret" not in json.dumps(output)

    code, _ = _run(
        [
            "connections",
            "add",
            "snowflake",
            "--name",
            "warehouse",
            "--credential",
            "account=acme",
            "--credential",
            "user=cayu",
        ],
        capsys,
    )
    assert code == 0
    assert _sent(fake)["credential"] == {"account": "acme", "user": "cayu"}


def test_rotate_reads_the_fields_of_the_connections_connector(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    warehouse = {"name": "warehouse", "connector": "snowflake", "status": "ready"}
    fake = cloud(
        {
            ("GET", "/v1/connections/warehouse"): httpx.Response(200, json=warehouse),
            _LIST_CONNECTORS: httpx.Response(200, json=_CONNECTORS),
            ("PUT", "/v1/connections/warehouse/credential"): httpx.Response(200, json=warehouse),
        }
    )
    private_key = tmp_path / "snowflake.p8"
    private_key.write_text("key-pair-material\n")

    code, _ = _run(
        [
            "connections",
            "rotate",
            "warehouse",
            "--credential",
            "account=acme",
            "--credential",
            "user=cayu",
            "--credential",
            f"private_key=@{private_key}",
        ],
        capsys,
    )

    assert code == 0
    assert _sent(fake)["credential"] == {
        "account": "acme",
        "user": "cayu",
        "private_key": "key-pair-material",
    }


def test_many_agents_are_resolved_from_one_list(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    key_file = tmp_path / "anthropic-key"
    key_file.write_text("sk-ant-secret\n")
    agents = {
        "items": [
            {"id": "support-desk", "name": "Support desk"},
            {"id": "triage", "name": "Triage"},
        ]
    }
    fake = cloud(
        {
            ("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=agents),
            ("POST", "/v1/connections"): httpx.Response(201, json=_CONNECTION),
        }
    )

    code, _ = _run(
        [
            "connections",
            "add",
            "anthropic",
            "--name",
            "team-anthropic",
            "--key-file",
            str(key_file),
            "--agent",
            "Support desk",
            "--agent",
            "Triage",
        ],
        capsys,
    )

    assert code == 0
    lists = [request for request in fake.requests if request.url.path == _APPS]
    assert len(lists) == 1
    (post,) = (request for request in fake.requests if request.method == "POST")
    assert json.loads(post.content)["agents"] == ["support-desk", "triage"]


def test_agents_all_makes_it_available_to_every_agent(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {("PUT", "/v1/connections/team-anthropic/agents"): httpx.Response(200, json=_CONNECTION)}
    )

    code, _ = _run(["connections", "agents", "team-anthropic", "--all"], capsys)

    assert code == 0
    assert json.loads(fake.requests[0].content) == {"agents": None}


def test_bind_and_unbind_an_agents_slot(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    slot = f"{_APPS}/support-desk/connections/primary"
    fake = cloud(
        {
            ("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            ("PUT", slot): httpx.Response(200, json={"slot": "primary"}),
            ("DELETE", slot): httpx.Response(204),
        }
    )

    code, _ = _run(
        [
            "connections",
            "bind",
            "primary",
            "team-anthropic",
            "--application",
            "Support desk",
            "--scope",
            "read",
        ],
        capsys,
    )
    assert code == 0
    (put,) = (request for request in fake.requests if request.method == "PUT")
    assert json.loads(put.content) == {
        "connection": "team-anthropic",
        "scopes": ["read"],
    }

    code, output = _run(
        ["connections", "unbind", "primary", "--application", "support-desk"], capsys
    )
    assert code == 0
    assert output["result"] == {
        "application_id": "support-desk",
        "slot": "primary",
        "unbound": True,
    }


def test_a_github_app_binding_names_its_repositories(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    slot = f"{_APPS}/support-desk/connections/github"
    fake = cloud(
        {
            ("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            ("PUT", slot): httpx.Response(200, json={"slot": "github"}),
        }
    )

    code, _ = _run(
        [
            "connections",
            "bind",
            "github",
            "triage-app",
            "--application",
            "support-desk",
            "--scope",
            "issues:write",
            "--repository",
            "acme/widgets",
            "--repository",
            "acme-labs/*",
        ],
        capsys,
    )

    assert code == 0
    (put,) = (request for request in fake.requests if request.method == "PUT")
    assert json.loads(put.content) == {
        "connection": "triage-app",
        "scopes": ["issues:write"],
        "repositories": ["acme/widgets", "acme-labs/*"],
    }


def test_a_model_slot_takes_no_repositories(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud({("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=_AGENTS)})

    code, output = _run(
        [
            "connections",
            "bind",
            "primary",
            "--model",
            "claude-opus",
            "--repository",
            "acme/widgets",
            "--application",
            "support-desk",
        ],
        capsys,
    )

    assert code == 2
    assert "--repository" in output["error"]["message"]
    assert all(request.method == "GET" for request in fake.requests)


def test_a_github_app_takes_its_id_plain_and_its_key_from_a_file(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud(
        {
            _LIST_CONNECTORS: httpx.Response(200, json=_CONNECTORS),
            ("POST", "/v1/connections"): httpx.Response(201, json=_CONNECTION),
        }
    )
    pem = tmp_path / "app.pem"
    pem.write_text("-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----\n")

    code, output = _run(
        [
            "connections",
            "add",
            "github-app",
            "--name",
            "triage-app",
            "--credential",
            "app_id=123456",
            "--credential",
            f"private_key=@{pem}",
        ],
        capsys,
    )

    assert code == 0
    assert _sent(fake)["credential"] == {
        "app_id": "123456",
        "private_key": "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
    }
    assert "fake" not in json.dumps(output)


def test_a_refused_removal_reports_clouds_reason(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud(
        {
            ("DELETE", "/v1/connections/team-anthropic"): httpx.Response(
                409,
                json={
                    "detail": {
                        "code": "connection_bound",
                        "message": "Agents support-desk are bound to this connection.",
                    }
                },
            )
        }
    )

    code, output = _run(["connections", "remove", "team-anthropic"], capsys)

    assert code != 0
    assert output["error"]["code"] == "connection_bound"
    assert "support-desk" in output["error"]["message"]


_SIGN_IN = {
    "name": "example-slack",
    "connector": "slack",
    "status": "pending",
    "account": None,
    "error": None,
    "authorization_url": "https://slack.com/oauth/v2/authorize?state=abc",
}


def test_add_without_a_credential_prints_where_to_approve_it(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({("POST", "/v1/connections"): httpx.Response(201, json=_SIGN_IN)})

    code = main(
        [
            "cloud",
            "connections",
            "add",
            "slack",
            "--name",
            "example-slack",
            "--scope",
            "chat:write",
            "--no-wait",
        ]
    )

    assert code == 0
    assert json.loads(fake.requests[-1].content) == {
        "connector": "slack",
        "name": "example-slack",
        "scopes": ["chat:write"],
    }
    printed = capsys.readouterr().out
    assert "  https://slack.com/oauth/v2/authorize?state=abc\n" in printed
    assert "signed in to the Cayu Cloud portal as the same person" in " ".join(printed.split())
    assert "cayu cloud connections authorize example-slack" in printed


def test_add_without_a_credential_with_json_prints_the_link_in_the_envelope(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("POST", "/v1/connections"): httpx.Response(201, json=_SIGN_IN)})

    code, output = _run(
        ["connections", "add", "slack", "--name", "example-slack", "--json", "--no-wait"], capsys
    )

    assert code == 0
    assert output["operation"] == "connections.add"
    assert output["result"] == _SIGN_IN


def test_authorize_prints_a_new_link(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): httpx.Response(
                200, json={**_SIGN_IN, "authorization_url": "https://slack.com/new"}
            )
        }
    )

    assert main(["cloud", "connections", "authorize", "example-slack", "--no-wait"]) == 0
    assert fake.requests[-1].content == b""
    assert "  https://slack.com/new\n" in capsys.readouterr().out

    code, output = _run(
        ["connections", "authorize", "example-slack", "--json", "--no-wait"], capsys
    )
    assert code == 0
    assert output["operation"] == "connections.authorize"
    assert output["result"]["authorization_url"] == "https://slack.com/new"


def test_a_refused_sign_in_reports_clouds_reason(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): httpx.Response(
                403,
                json={
                    "detail": {
                        "code": "admin_required",
                        "message": "Only an organization admin can do this.",
                    }
                },
            )
        }
    )

    code, output = _run(["connections", "authorize", "example-slack", "--no-wait"], capsys)

    assert code != 0
    assert output["error"]["code"] == "admin_required"


_STATE = "s" * 43
_READY = {
    **{key: value for key, value in _SIGN_IN.items() if key != "authorization_url"},
    "status": "ready",
    "account": {"id": "T0123", "name": "Example Workspace"},
}
_COMPLETE = ("POST", "/v1/connections/sign-in/complete")


def _returning(request: httpx.Request, *, status: int = 201) -> httpx.Response:
    """Cloud's answer to a sign-in that returns to the command: a state ending in its port."""

    port = json.loads(request.content)["return_port"]
    url = f"https://slack.com/oauth/v2/authorize?state={_STATE}.{port}"
    return httpx.Response(status, json={**_SIGN_IN, "authorization_url": url})


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _status(opener: Any, url: str, cookie: str = "") -> tuple[int, str]:
    try:
        request = Request(url, headers={"Cookie": cookie} if cookie else {})
        with opener.open(request, timeout=10) as answer:
            return answer.status, answer.headers.get("Location", "")
    except HTTPError as refused:
        return refused.code, refused.headers.get("Location", "")
    except URLError:
        # Nothing listens any more: the command already stopped.
        return 0, ""


def _begin(start: str) -> tuple[str, str]:
    """What a browser opening the command's start page gets: the sign-in's state, from the
    approval page it is sent on to, and the cookie it then carries back (``name=value``)."""

    try:
        with _NOT_FOLLOWING.open(start, timeout=10) as answer:
            headers = answer.headers
    except HTTPError as redirected:
        headers = redirected.headers
    approval = headers["Location"]
    cookie = headers["Set-Cookie"].split(";")[0]
    return parse_qs(urlsplit(approval).query)["state"][0], cookie


class _Browser:
    """The browser and the portal: given the link, it opens the command's start page, then comes
    back to the command's loopback port once for each visit, ``visit(state)`` naming the path
    and the query (and, as a third item, False to come back without the start page's cookie)."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *visits: Callable[[str], tuple[Any, ...]],
    ) -> None:
        self.visits = visits
        self.answers: list[tuple[int, str]] = []
        #: Where each visit ended, after any redirect: what the address bar shows.
        self.addresses: list[str] = []
        self.opened: list[str] = []
        self.thread: threading.Thread | None = None
        monkeypatch.setattr(_cloud_connections.webbrowser, "open", self.open)

    def open(self, url: str) -> bool:
        self.opened.append(url)
        self.thread = threading.Thread(target=self._return, args=(url,), daemon=True)
        self.thread.start()
        return True

    def _return(self, url: str) -> None:
        # The command's own start page sends the browser on to the approval page.
        try:
            state, cookie = _begin(url)
        except URLError:
            return
        port = _port_of(url)
        for visit in self.visits:
            path, query, *carried = visit(state)
            headers = {"Cookie": cookie} if carried != [False] else {}
            address = f"http://127.0.0.1:{port}{path}?{urlencode(query)}"
            try:
                with _DIRECT.open(Request(address, headers=headers)) as answer:
                    self.answers.append((answer.status, answer.read().decode()))
                    self.addresses.append(answer.geturl())
            except HTTPError as refused:
                self.answers.append((refused.code, refused.read().decode()))
                self.addresses.append(refused.geturl())
            except URLError:
                # The command stopped listening (Ctrl-C) before the browser came back.
                self.answers.append((0, ""))

    def joined(self) -> list[tuple[int, str]]:
        assert self.thread is not None
        self.thread.join(timeout=10)
        return self.answers


_RETURN = "/cayu-cloud/sign-in"
_DIRECT = build_opener(ProxyHandler({}))
#: A browser that shows where a redirect goes instead of following it.
_NOT_FOLLOWING = build_opener(ProxyHandler({}), _NoRedirect)


def test_add_waits_for_the_sign_in_and_completes_only_its_own_state(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = cloud(
        {
            ("POST", "/v1/connections"): _returning,
            _COMPLETE: httpx.Response(200, json=_READY),
        }
    )
    browser = _Browser(
        monkeypatch,
        lambda state: ("/elsewhere", {"state": state, "code": "the-code"}),
        lambda state: (_RETURN, {"state": f"{_STATE}.1", "code": "the-code"}),
        lambda state: (_RETURN, {"state": state}),
        lambda state: (_RETURN, {"state": state, "code": "the-code"}),
    )

    code = main(["cloud", "connections", "add", "slack", "--name", "example-slack"])

    assert code == 0
    started = json.loads(fake.requests[0].content)
    assert started == {
        "connector": "slack",
        "name": "example-slack",
        "return_port": started["return_port"],
    }
    state = f"{_STATE}.{started['return_port']}"
    # Only the state it started, once, with what came back and nothing else.
    completions = [request for request in fake.requests if request.url.path == _COMPLETE[1]]
    assert [json.loads(request.content) for request in completions] == [
        {"state": state, "code": "the-code"}
    ]
    answers = browser.joined()
    assert [status for status, _text in answers] == [400, 400, 400, 200]
    assert (
        "example-slack is connected to Example Workspace. You can close this tab." in answers[-1][1]
    )
    # The browser ends on a page whose address holds neither the code nor the state.
    port = started["return_port"]
    ended = urlsplit(browser.addresses[-1])
    assert (ended.netloc, ended.path) == (f"127.0.0.1:{port}", "/cayu-cloud/sign-in/done")
    assert set(parse_qs(ended.query)) == {"t"}
    printed = capsys.readouterr()
    assert printed.out == "example-slack is connected to Example Workspace.\n"
    assert f"Waiting for the sign-in on 127.0.0.1:{port}." in printed.err
    # The browser is opened, and the link printed, at the command's start page with its own
    # secret: no state.
    (opened,) = browser.opened
    start = urlsplit(opened)
    assert (start.netloc, start.path) == (f"127.0.0.1:{port}", "/cayu-cloud/sign-in/start")
    (key,) = parse_qs(start.query)["k"]
    assert len(key) >= 43
    assert f"  {opened}\n" in printed.err
    for secret in (state, _STATE, "the-code"):
        assert secret not in printed.out + printed.err + opened
    # The start page's secret never goes to Cloud.
    for request in fake.requests:
        assert key not in str(request.url) + request.content.decode()


def test_authorize_waits_and_json_prints_the_connected_connection(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200),
            _COMPLETE: httpx.Response(200, json=_READY),
        }
    )
    _Browser(monkeypatch, lambda state: (_RETURN, {"state": state, "code": "the-code"}))

    code, output = _run(["connections", "authorize", "example-slack", "--json"], capsys)

    assert code == 0
    assert set(json.loads(fake.requests[0].content)) == {"return_port"}
    assert (output["operation"], output["result"], output["sign_in"]) == (
        "connections.authorize",
        _READY,
        "completed",
    )


def test_a_refused_approval_reports_clouds_reason_to_the_browser_and_the_terminal(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200),
            _COMPLETE: httpx.Response(
                409,
                json={"detail": {"code": "access_denied", "message": "The person refused."}},
            ),
        }
    )
    browser = _Browser(
        monkeypatch, lambda state: (_RETURN, {"state": state, "error": "access_denied"})
    )

    code, output = _run(["connections", "authorize", "example-slack"], capsys)

    assert code != 0
    assert output["error"]["code"] == "access_denied"
    assert json.loads(fake.requests[-1].content)["error"] == "access_denied"
    ((status, text),) = browser.joined()
    assert status == 200, text
    assert text == "Couldn't connect (access_denied). See your terminal for more.\n"


def test_waiting_stops_after_its_time_and_leaves_the_connection_as_it_was(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = cloud(
        {("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200)}
    )
    browser = _Browser(monkeypatch)

    code = main(
        [
            "cloud",
            "connections",
            "authorize",
            "example-slack",
            "--no-browser",
            "--wait-seconds",
            "0.2",
        ]
    )
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert code != 0
    assert output["error"]["category"] == "sign_in_timed_out"
    assert "cayu cloud connections authorize example-slack" in output["error"]["message"]
    assert browser.opened == []
    # Without a browser the link to open is still the start page, never the state.
    printed = captured.err
    assert re.search(r"  http://127\.0\.0\.1:\d+/cayu-cloud/sign-in/start\?k=[\w-]{43}\n", printed)
    assert _STATE not in printed
    assert [request.url.path for request in fake.requests] == [
        "/v1/connections/example-slack/authorize"
    ]


def test_ctrl_c_stops_waiting_and_says_how_to_go_on(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cloud({("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200)})
    _Browser(monkeypatch)

    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(_cloud_sign_in.SignInReturn, "wait", interrupted)

    code, output = _run(["connections", "authorize", "example-slack"], capsys)

    assert code != 0
    assert output["error"]["category"] == "sign_in_interrupted"
    assert "cayu cloud connections authorize example-slack" in output["error"]["message"]


def test_authorize_on_a_cloud_that_ignores_the_port_prints_its_link_for_the_portal(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cloud({("POST", "/v1/connections/example-slack/authorize"): httpx.Response(200, json=_SIGN_IN)})
    browser = _Browser(monkeypatch)

    assert main(["cloud", "connections", "authorize", "example-slack"]) == 0

    printed = capsys.readouterr()
    assert "can't finish a sign-in from the terminal" in printed.err
    assert "  https://slack.com/oauth/v2/authorize?state=abc\n" in printed.out
    assert "signed in to the Cayu Cloud portal as the same person" in " ".join(printed.out.split())
    assert browser.opened == []


def test_add_on_a_cloud_that_refuses_the_port_adds_it_again_for_the_portal(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def older(request: httpx.Request) -> httpx.Response:
        if "return_port" in json.loads(request.content):
            refusal = {
                "code": "invalid_request",
                "message": "The request doesn't have the shape this route takes.",
                "errors": [{"loc": ["return_port"], "msg": "Extra inputs are not permitted"}],
            }
            return httpx.Response(422, json={"detail": refusal})
        return httpx.Response(201, json=_SIGN_IN)

    fake = cloud({("POST", "/v1/connections"): older})
    _Browser(monkeypatch)

    code, output = _run(
        ["connections", "add", "slack", "--name", "example-slack", "--json"], capsys
    )

    assert code == 0
    assert output["result"] == _SIGN_IN
    assert [set(json.loads(request.content)) for request in fake.requests] == [
        {"connector", "name", "return_port"},
        {"connector", "name"},
    ]


def test_another_invalid_request_is_not_taken_for_an_older_cloud(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    refusal = {
        "code": "invalid_request",
        "message": "The request doesn't have the shape this route takes.",
        "errors": [{"loc": ["scopes", 0], "msg": "String should have at most 128 characters"}],
    }
    fake = cloud({("POST", "/v1/connections"): httpx.Response(422, json={"detail": refusal})})
    _Browser(monkeypatch)

    code, output = _run(["connections", "add", "slack", "--name", "example-slack"], capsys)

    assert code != 0
    assert output["error"]["code"] == "invalid_request"
    assert len(fake.requests) == 1


def test_an_idle_connection_doesnt_hold_the_command_past_its_wait(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cloud({("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200)})
    idle: list[socket.socket] = []

    def preconnect(url: str) -> bool:
        port = _port_of(url)
        idle.append(socket.create_connection(("127.0.0.1", port)))
        return True

    monkeypatch.setattr(_cloud_connections.webbrowser, "open", preconnect)
    began = time.monotonic()

    code, output = _run(
        ["connections", "authorize", "example-slack", "--wait-seconds", "0.5"], capsys
    )

    assert time.monotonic() - began < 5
    assert code != 0
    assert output["error"]["category"] == "sign_in_timed_out"
    assert len(idle) == 1
    idle[0].close()


def _port_of(url: str) -> int:
    """The command's loopback port, from the start page the browser is opened at."""

    port = urlsplit(url).port
    assert port is not None
    return port


def test_an_idle_connection_doesnt_delay_the_approval(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200),
            _COMPLETE: httpx.Response(200, json=_READY),
        }
    )
    idle: list[socket.socket] = []
    took: list[float] = []

    def approve(url: str) -> bool:
        port = _port_of(url)
        # A browser's preconnect: open, and nothing sent.
        idle.append(socket.create_connection(("127.0.0.1", port)))

        def come_back() -> None:
            state, cookie = _begin(url)
            time.sleep(0.2)
            began = time.monotonic()
            query = urlencode({"state": state, "code": "the-code"})
            address = f"http://127.0.0.1:{port}{_RETURN}?{query}"
            with _DIRECT.open(Request(address, headers={"Cookie": cookie})) as answer:
                answer.read()
            took.append(time.monotonic() - began)

        threading.Thread(target=come_back, daemon=True).start()
        return True

    monkeypatch.setattr(_cloud_connections.webbrowser, "open", approve)

    assert main(["cloud", "connections", "authorize", "example-slack"]) == 0
    assert len(took) == 1
    assert took[0] < 1
    idle[0].close()


def test_a_trickling_connection_cant_hold_the_command_past_its_wait(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cloud({("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200)})
    cut: list[float] = []

    def trickle(url: str) -> bool:
        port = _port_of(url)

        def send() -> None:
            began = time.monotonic()
            with socket.create_connection(("127.0.0.1", port)) as connection:
                try:
                    for byte in b"GET /cayu-cloud/sign-in?state=" + b"x" * 100:
                        connection.sendall(bytes([byte]))
                        time.sleep(0.05)
                except OSError:
                    pass
            cut.append(time.monotonic() - began)

        threading.Thread(target=send, daemon=True).start()
        return True

    monkeypatch.setattr(_cloud_connections.webbrowser, "open", trickle)
    began = time.monotonic()

    code, output = _run(
        ["connections", "authorize", "example-slack", "--wait-seconds", "1"], capsys
    )

    assert time.monotonic() - began < 2
    assert output["error"]["category"] == "sign_in_timed_out"
    assert code != 0
    # Its connection is cut once its time is up, not kept while it trickles on.
    deadline = time.monotonic() + 3
    while not cut and time.monotonic() < deadline:
        time.sleep(0.05)
    assert cut and cut[0] < 2.5


def test_two_callbacks_at_once_complete_the_sign_in_once(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def slowly(_request: httpx.Request) -> httpx.Response:
        time.sleep(0.3)
        return httpx.Response(200, json=_READY)

    fake = cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200),
            _COMPLETE: slowly,
        }
    )
    answers: list[int] = []
    lock = threading.Lock()

    def twice(url: str) -> bool:
        port = _port_of(url)
        start = threading.Barrier(2)

        def come_back(state: str, cookie: str) -> None:
            query = urlencode({"state": state, "code": "the-code"})
            address = f"http://127.0.0.1:{port}{_RETURN}?{query}"
            start.wait()
            try:
                with _DIRECT.open(Request(address, headers={"Cookie": cookie})) as answer:
                    status = answer.status
            except HTTPError as refused:
                status = refused.code
            with lock:
                answers.append(status)

        def open_then_come_back_twice() -> None:
            state, cookie = _begin(url)
            for _ in range(2):
                threading.Thread(target=come_back, args=(state, cookie), daemon=True).start()

        threading.Thread(target=open_then_come_back_twice, daemon=True).start()
        return True

    monkeypatch.setattr(_cloud_connections.webbrowser, "open", twice)

    assert main(["cloud", "connections", "authorize", "example-slack"]) == 0
    completions = [request for request in fake.requests if request.url.path == _COMPLETE[1]]
    assert len(completions) == 1
    deadline = time.monotonic() + 5
    while len(answers) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert sorted(answers) == [200, 400]


_APPROVAL = "https://slack.com/oauth/v2/authorize?state=the-state"


def test_only_the_browser_that_completed_ends_the_wait_on_the_done_page() -> None:
    finishing = _NOT_FOLLOWING
    results: list[dict[str, Any]] = []
    with _cloud_sign_in.SignInReturn() as listener:
        base = f"http://127.0.0.1:{listener.port}"
        opened = listener.opening(_APPROVAL)
        waiting = threading.Thread(
            target=lambda: results.append(
                listener.wait(state="the-state", finish=lambda _r: _READY, seconds=10)
            ),
            daemon=True,
        )
        waiting.start()
        _state, cookie = _begin(opened)
        callback = f"{base}{_RETURN}?state=the-state&code=c"
        status, location = _status(finishing, callback, cookie)
        assert (status, location.startswith("/cayu-cloud/sign-in/done?t=")) == (303, True)
        for stray, carried in (
            ("/cayu-cloud/sign-in/done", cookie),
            ("/cayu-cloud/sign-in/done?t=guess", cookie),
            (location, ""),
        ):
            assert _status(finishing, f"{base}{stray}", carried)[0] == 400
        assert not listener.shown.is_set()
        assert _status(finishing, f"{base}{location}", cookie)[0] == 200
        waiting.join(timeout=5)
    assert results == [_READY]


def test_the_start_page_sends_one_browser_on_and_only_its_callback_counts() -> None:
    completed: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    with _cloud_sign_in.SignInReturn() as listener:
        base = f"http://127.0.0.1:{listener.port}"
        opened = listener.opening(_APPROVAL)
        assert "state" not in opened
        waiting = threading.Thread(
            target=lambda: results.append(
                listener.wait(
                    state="the-state",
                    finish=lambda returned: completed.append(returned) or _READY,
                    seconds=10,
                )
            ),
            daemon=True,
        )
        waiting.start()
        try:
            with _NOT_FOLLOWING.open(opened) as answer:
                headers = answer.headers
        except HTTPError as redirected:
            headers = redirected.headers
        assert headers["Location"] == _APPROVAL
        cookie = headers["Set-Cookie"]
        assert cookie.startswith(f"cayu_sign_in_{listener.port}=")
        assert {"Path=/cayu-cloud/sign-in", "HttpOnly", "SameSite=Lax"} <= {
            part.strip() for part in cookie.split(";")
        }
        callback = f"{base}{_RETURN}?state=the-state&code=c"
        # Without the cookie, or with another one, it isn't this browser: the wait goes on.
        for carried in ("", f"cayu_sign_in_{listener.port}=guess"):
            assert _status(_NOT_FOLLOWING, callback, carried)[0] == 400
        assert not listener.finished.is_set()
        status, done = _status(_NOT_FOLLOWING, callback, cookie.split(";")[0])
        assert status == 303
        assert _status(_NOT_FOLLOWING, f"{base}{done}", cookie.split(";")[0])[0] == 200
        waiting.join(timeout=5)
    assert results == [_READY]
    assert completed == [{"state": "the-state", "code": "c"}]


def test_the_start_page_without_its_secret_is_not_found_and_changes_nothing() -> None:
    with _cloud_sign_in.SignInReturn() as listener:
        opened = listener.opening(_APPROVAL)
        base = opened.split("?")[0]

        def wait() -> None:
            with contextlib.suppress(_cloud_sign_in.SignInTimedOut):
                listener.wait(state="the-state", finish=lambda _r: _READY, seconds=1)

        waiting = threading.Thread(target=wait, daemon=True)
        waiting.start()
        for guess in (base, f"{base}?k=guess", f"{base}?k="):
            try:
                _DIRECT.open(guess)
                raise AssertionError("A guess was sent on.")
            except HTTPError as missing:
                assert (missing.code, missing.read().decode()) == (
                    404,
                    "Not this sign-in's link.\n",
                )
        assert not (listener.started or listener.reused)
        first = _status(_NOT_FOLLOWING, opened)
        waiting.join(timeout=5)
    assert first == (303, _APPROVAL)


def test_a_second_opener_of_the_start_page_stops_the_command() -> None:
    completed: list[dict[str, str]] = []
    failures: list[Exception] = []
    with _cloud_sign_in.SignInReturn() as listener:
        base = f"http://127.0.0.1:{listener.port}"
        opened = listener.opening(_APPROVAL)

        def wait() -> None:
            try:
                listener.wait(
                    state="the-state",
                    finish=lambda returned: completed.append(returned) or _READY,
                    seconds=10,
                )
            except Exception as exc:
                failures.append(exc)

        waiting = threading.Thread(target=wait, daemon=True)
        waiting.start()
        _state, cookie = _begin(opened)
        try:
            _DIRECT.open(opened)
            raise AssertionError("A second opener was sent on.")
        except HTTPError as gone:
            assert (gone.code, gone.read().decode()) == (410, f"{_cloud_sign_in.LINK_REUSED}\n")
        # The command stops; a callback even with the first browser's cookie connects nothing.
        status = _status(_NOT_FOLLOWING, f"{base}{_RETURN}?state=the-state&code=c", cookie)
        waiting.join(timeout=5)
    assert status[0] in {0, 400}
    assert completed == []
    assert [type(failure) for failure in failures] == [_cloud_sign_in.SignInLinkReused]


def test_ctrl_c_on_the_way_to_the_done_page_keeps_the_completed_sign_in() -> None:
    with _cloud_sign_in.SignInReturn() as listener:
        base = f"http://127.0.0.1:{listener.port}"
        opened = listener.opening(_APPROVAL)

        def come_back() -> None:
            # Completes, but never comes back for the done page; then Ctrl-C.
            _state, cookie = _begin(opened)
            _status(_NOT_FOLLOWING, f"{base}{_RETURN}?state=the-state&code=c", cookie)
            listener.finished.wait(5)
            _thread.interrupt_main()

        threading.Thread(target=come_back, daemon=True).start()
        result = listener.wait(state="the-state", finish=lambda _r: _READY, seconds=10)
    assert result == _READY


def test_a_callback_from_another_browser_is_refused_and_the_wait_goes_on(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200),
            _COMPLETE: httpx.Response(200, json=_READY),
        }
    )
    browser = _Browser(
        monkeypatch,
        lambda state: (_RETURN, {"state": state, "code": "theirs"}, False),
        lambda state: (_RETURN, {"state": state, "code": "the-code"}),
    )

    assert main(["cloud", "connections", "authorize", "example-slack"]) == 0

    assert [status for status, _text in browser.joined()] == [400, 200]
    completions = [request for request in fake.requests if request.url.path == _COMPLETE[1]]
    assert [json.loads(request.content)["code"] for request in completions] == ["the-code"]


def test_a_sign_in_another_opener_came_back_with_first_fails_loudly(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200),
            _COMPLETE: httpx.Response(200, json=_READY),
        }
    )
    late: list[int] = []

    def raced(url: str) -> bool:
        def attacker_then_victim() -> None:
            # Someone who read the link opens it first and comes straight back with a code of
            # their own; the person's browser opens it after.
            state, cookie = _begin(url)
            query = urlencode({"state": state, "code": "theirs"})
            _status(_NOT_FOLLOWING, f"{url.split('?')[0].rsplit('/', 1)[0]}?{query}", cookie)
            late.append(_status(_NOT_FOLLOWING, url)[0])

        threading.Thread(target=attacker_then_victim, daemon=True).start()
        return True

    monkeypatch.setattr(_cloud_connections.webbrowser, "open", raced)

    code, output = _run(["connections", "authorize", "example-slack"], capsys)

    assert code != 0
    assert output["error"]["category"] == "sign_in_link_shared"
    message = output["error"]["message"]
    assert "may be connected to someone else's workspace (Example Workspace)" in message
    assert "cayu cloud connections show example-slack" in message
    assert "Nothing was connected" not in message
    assert late == [410]
    completions = [request for request in fake.requests if request.url.path == _COMPLETE[1]]
    assert [json.loads(request.content)["code"] for request in completions] == ["theirs"]


def test_another_opener_of_the_link_stops_the_command_with_nothing_connected(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200),
            _COMPLETE: httpx.Response(200, json=_READY),
        }
    )
    second: list[int] = []

    def raced(url: str) -> bool:
        def both() -> None:
            _begin(url)
            second.append(_status(_NOT_FOLLOWING, url)[0])

        threading.Thread(target=both, daemon=True).start()
        return True

    monkeypatch.setattr(_cloud_connections.webbrowser, "open", raced)

    code, output = _run(["connections", "authorize", "example-slack"], capsys)

    assert code != 0
    assert output["error"]["category"] == "sign_in_link_reused"
    assert output["error"]["message"] == _cloud_sign_in.LINK_REUSED
    assert second == [410]
    assert all(request.url.path != _COMPLETE[1] for request in fake.requests)


def test_a_failure_after_cloud_answers_ends_the_wait_with_an_error(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200),
            _COMPLETE: httpx.Response(200, json=_READY),
        }
    )
    _Browser(monkeypatch, lambda state: (_RETURN, {"state": state, "code": "the-code"}))

    def broken(_connection: object) -> str:
        raise ValueError("An answer this command can't read.")

    monkeypatch.setattr(_cloud_sign_in, "connected", broken)
    began = time.monotonic()

    code, output = _run(
        ["connections", "authorize", "example-slack", "--wait-seconds", "30"], capsys
    )

    assert time.monotonic() - began < 10
    assert code != 0
    assert output["error"]["message"] == "An answer this command can't read."


def test_a_flood_of_connections_is_closed_quietly_and_the_approval_still_completes(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200),
            _COMPLETE: httpx.Response(200, json=_READY),
        }
    )
    flood: list[socket.socket] = []

    def flooded(url: str) -> bool:
        port = _port_of(url)

        def come_back() -> None:
            state, cookie = _begin(url)
            # More idle connections than are served at once, then let them go.
            for _ in range(_cloud_sign_in.MAX_CONNECTIONS * 3):
                flood.append(socket.create_connection(("127.0.0.1", port)))
            time.sleep(0.2)
            for connection in flood:
                connection.close()
            time.sleep(0.3)
            query = urlencode({"state": state, "code": "the-code"})
            address = f"http://127.0.0.1:{port}{_RETURN}?{query}"
            with _DIRECT.open(Request(address, headers={"Cookie": cookie})) as answer:
                answer.read()

        threading.Thread(target=come_back, daemon=True).start()
        return True

    monkeypatch.setattr(_cloud_connections.webbrowser, "open", flooded)

    assert main(["cloud", "connections", "authorize", "example-slack"]) == 0
    printed = capsys.readouterr()
    assert "connected" in printed.out
    assert "Traceback" not in printed.err


def test_ctrl_c_while_the_sign_in_is_being_completed_says_it_may_have_connected(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(_request: httpx.Request) -> httpx.Response:
        _thread.interrupt_main()
        time.sleep(0.5)
        return httpx.Response(200, json=_READY)

    cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200),
            _COMPLETE: interrupted,
        }
    )
    _Browser(monkeypatch, lambda state: (_RETURN, {"state": state, "code": "the-code"}))

    code, output = _run(["connections", "authorize", "example-slack"], capsys)

    assert code != 0
    assert output["error"]["category"] == "sign_in_interrupted"
    assert "cayu cloud connections show example-slack" in output["error"]["message"]


def test_port_waits_on_the_port_given_and_says_when_it_is_taken(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): partial(_returning, status=200),
            _COMPLETE: httpx.Response(200, json=_READY),
        }
    )
    _Browser(monkeypatch, lambda state: (_RETURN, {"state": state, "code": "the-code"}))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    assert main(["cloud", "connections", "authorize", "example-slack", "--port", str(port)]) == 0
    assert json.loads(fake.requests[0].content) == {"return_port": port}
    assert f"127.0.0.1:{port}" in capsys.readouterr().err

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        busy = taken.getsockname()[1]
        code, output = _run(
            ["connections", "authorize", "example-slack", "--port", str(busy)], capsys
        )
    assert code != 0
    assert output["error"]["category"] == "sign_in_port_unavailable"
    assert len(fake.requests) == 2


def test_show_carries_the_account_status_and_why_it_failed(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    failed = {**_SIGN_IN, "status": "error", "error": "access_denied"}
    failed.pop("authorization_url")
    ready = {**failed, "name": "team-slack", "status": "ready", "error": None}
    ready["account"] = {"id": "T0123", "name": "Halden"}
    cloud(
        {
            ("GET", "/v1/connections/example-slack"): httpx.Response(200, json=failed),
            ("GET", "/v1/connections"): httpx.Response(200, json={"items": [failed, ready]}),
        }
    )

    code, output = _run(["connections", "show", "example-slack"], capsys)
    assert code == 0
    assert (output["result"]["status"], output["result"]["error"]) == ("error", "access_denied")

    code, output = _run(["connections", "list"], capsys)
    assert code == 0
    assert [item["account"] for item in output["result"]["items"]] == [
        None,
        {"id": "T0123", "name": "Halden"},
    ]


def test_reconnecting_a_ready_connection_says_it_keeps_working(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("POST", "/v1/connections/example-slack/authorize"): httpx.Response(
                200, json={**_SIGN_IN, "status": "ready"}
            )
        }
    )

    assert main(["cloud", "connections", "authorize", "example-slack", "--no-wait"]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith(
        "example-slack keeps working until you sign in again. To reconnect it,"
    )
    assert "waiting for its sign-in" not in printed


def test_an_sap_connection_takes_its_settings_plain_and_its_secrets_from_files(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud(
        {
            _LIST_CONNECTORS: httpx.Response(200, json=_CONNECTORS),
            ("POST", "/v1/connections"): httpx.Response(201, json=_CONNECTION),
        }
    )
    secret = tmp_path / "sap-secret"
    secret.write_text("fake-client-value\n")
    plain = {
        "base_url": "https://s4.example.test",
        "token_url": "https://s4.example.test/oauth/token",
        "client_id": "CAYU_READER",
        "oauth_scope": "API_BUSINESS_PARTNER_0001",
        "sap_client": "100",
    }

    code, output = _run(
        [
            "connections",
            "add",
            "sap",
            "--name",
            "erp",
            *(
                item
                for name, value in plain.items()
                for item in ("--credential", f"{name}={value}")
            ),
            "--credential",
            f"client_secret=@{secret}",
            "--scope",
            "API_BUSINESS_PARTNER.read",
        ],
        capsys,
    )

    assert code == 0
    sent = _sent(fake)
    assert sent["credential"] == {**plain, "client_secret": "fake-client-value"}
    assert sent["scopes"] == ["API_BUSINESS_PARTNER.read"]
    assert "fake-client-value" not in json.dumps(output)


def test_sap_secret_and_multi_line_fields_given_plain_are_refused_unrepeated(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({_LIST_CONNECTORS: httpx.Response(200, json=_CONNECTORS)})

    for connector, field in (
        ("sap", "client_secret"),
        ("sap", "ca_bundle"),
        ("sap-certificate", "client_certificate"),
        ("sap-certificate", "client_key"),
    ):
        code, output = _run(
            [
                "connections",
                "add",
                connector,
                "--name",
                "erp",
                "--credential",
                f"{field}=not-a-file-value",
            ],
            capsys,
        )
        assert code == 2
        why = "secret" if field in {"client_secret", "client_key"} else "multi-line"
        assert output["error"]["message"] == (
            f"The credential field {field} is {why}: give it from a file, {field}=@PATH."
        )
        assert "not-a-file-value" not in json.dumps(output)
    assert all(request.method == "GET" for request in fake.requests)


def test_a_certificate_connection_reads_its_pem_fields_from_files(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud(
        {
            _LIST_CONNECTORS: httpx.Response(200, json=_CONNECTORS),
            ("POST", "/v1/connections"): httpx.Response(201, json=_CONNECTION),
        }
    )
    files = {}
    for field in ("client_certificate", "client_key", "ca_bundle"):
        files[field] = tmp_path / f"{field}.pem"
        files[field].write_text(f"-----BEGIN TEST {field}-----\n")

    code, _ = _run(
        [
            "connections",
            "add",
            "sap-certificate",
            "--name",
            "erp",
            "--credential",
            "base_url=https://s4.example.test",
            *(
                item
                for field, path in files.items()
                for item in ("--credential", f"{field}=@{path}")
            ),
        ],
        capsys,
    )

    assert code == 0
    assert _sent(fake)["credential"] == {
        "base_url": "https://s4.example.test",
        **{field: f"-----BEGIN TEST {field}-----" for field in files},
    }


_CHECK = {
    "name": "erp",
    "connector": "sap",
    "status": "ready",
    "error": None,
    "check_error": None,
    "check_message": None,
    "services": [
        {
            "service": "API_BUSINESS_PARTNER",
            "title": "Business partners",
            "scope": "API_BUSINESS_PARTNER.read",
            "result": "ok",
            "status": 200,
            "message": None,
        },
        {
            "service": "API_SUPPLIERINVOICE_PROCESS_SRV",
            "title": "Supplier invoices",
            "scope": "API_SUPPLIERINVOICE_PROCESS_SRV.read",
            "result": "not_authorised",
            "status": 403,
            "message": "The communication arrangement isn't set up.",
        },
    ],
}


def test_check_prints_each_services_result(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud({("POST", "/v1/connections/erp/check"): httpx.Response(200, json=_CHECK)})

    assert main(["cloud", "connections", "check", "erp"]) == 0

    assert [(request.method, request.url.path) for request in fake.requests] == [
        ("POST", "/v1/connections/erp/check")
    ]
    assert capsys.readouterr().out.splitlines() == [
        "erp: ready",
        "",
        "SERVICE                          RESULT          DETAIL",
        "API_BUSINESS_PARTNER             ok              200",
        "API_SUPPLIERINVOICE_PROCESS_SRV  not_authorised  The communication arrangement isn't set up.",
    ]


def test_check_with_json_prints_the_envelope(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    failed = {**_CHECK, "status": "error", "error": "auth_failed", "services": []}
    cloud({("POST", "/v1/connections/erp/check"): httpx.Response(200, json=failed)})

    code, output = _run(["connections", "check", "erp", "--json"], capsys)
    assert code == 0
    assert output["operation"] == "connections.check"
    assert output["result"] == failed

    assert main(["cloud", "connections", "check", "erp"]) == 0
    assert capsys.readouterr().out == "erp: error (auth_failed)\n"


@pytest.mark.parametrize(
    ("status", "check_error", "check_message", "said"),
    [
        (
            "ready",
            "host_unreachable",
            None,
            "The check couldn't reach the SAP host. (host_unreachable)",
        ),
        (
            "ready",
            "token_url_unreachable",
            "The token URL didn't answer.",
            "The token URL didn't answer. (token_url_unreachable)",
        ),
        (
            "error",
            "auth_failed",
            None,
            "SAP rejected the connection's credential; an admin should check or rotate it. "
            "(auth_failed)",
        ),
        (
            "error",
            "auth_failed",
            "SAP's sandbox rejected the API key.",
            "SAP's sandbox rejected the API key. (auth_failed)",
        ),
        (
            "error",
            "certificate_rejected",
            None,
            "SAP rejected the connection's client certificate; an admin should check or rotate "
            "it. (certificate_rejected)",
        ),
        (
            "error",
            "host_not_allowed",
            None,
            "This Cayu Cloud doesn't allow the connection's host; an admin should fix base_url "
            "or ask the install's operator to allow it. (host_not_allowed)",
        ),
        ("error", "something_new", None, "The check failed. (something_new)"),
    ],
)
def test_check_says_why_the_check_itself_failed_and_what_to_do(
    cloud,
    capsys: pytest.CaptureFixture[str],
    status: str,
    check_error: str,
    check_message: str | None,
    said: str,
) -> None:
    failed = {
        **_CHECK,
        "status": status,
        "check_error": check_error,
        "check_message": check_message,
        "services": [],
    }
    cloud({("POST", "/v1/connections/erp/check"): httpx.Response(200, json=failed)})

    assert main(["cloud", "connections", "check", "erp"]) == 0
    heading, explained = capsys.readouterr().out.splitlines()
    assert heading == f"erp: {status}"
    unchanged = " The connection's status is unchanged; try again later."
    assert explained == said + (unchanged if check_error.endswith("_unreachable") else "")


_MISSING = httpx.Response(404, json={"detail": "Not Found"})
_STATUS = ("GET", "/v1/connections-status")
_OFF = httpx.Response(
    200,
    json={
        "enabled": False,
        "message": "Connections are turned off in this Cayu Cloud installation. "
        "Its operator turns them on.",
    },
)


@pytest.mark.parametrize(
    ("arguments", "route"),
    [
        (["connectors"], ("GET", "/v1/connectors")),
        (["connections", "list"], ("GET", "/v1/connections")),
        (["connections", "show", "team-anthropic"], ("GET", "/v1/connections/team-anthropic")),
    ],
)
def test_connections_turned_off_are_named_in_words(
    cloud, capsys: pytest.CaptureFixture[str], arguments: list[str], route: tuple[str, str]
) -> None:
    cloud({route: _MISSING, _STATUS: _OFF})

    code, output = _run(arguments, capsys)

    assert code != 0
    assert output["error"]["category"] == "connections_disabled"
    assert output["error"]["message"] == (
        "Connections are turned off in this Cayu Cloud installation."
    )


@pytest.mark.parametrize(
    ("arguments", "route"),
    [
        (
            ["connections", "bound", "--application", "support-desk"],
            ("GET", f"{_APPS}/support-desk/connections"),
        ),
        (
            ["models", "list", "--application", "support-desk"],
            ("GET", f"{_APPS}/support-desk/models"),
        ),
    ],
)
def test_an_agents_slots_on_a_cloud_with_connections_off(
    cloud, capsys: pytest.CaptureFixture[str], arguments: list[str], route: tuple[str, str]
) -> None:
    cloud(
        {
            ("GET", f"{_APPS}?lifecycle=all"): httpx.Response(200, json=_AGENTS),
            route: _MISSING,
            _STATUS: _OFF,
        }
    )

    code, output = _run([*arguments, "--json"] if arguments[0] == "models" else arguments, capsys)

    assert code != 0
    assert output["error"]["category"] == "connections_disabled"


def test_an_older_cloud_without_the_status_route_says_what_that_can_mean(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("GET", "/v1/connections"): _MISSING, _STATUS: _MISSING})

    code, output = _run(["connections", "list"], capsys)

    assert code != 0
    assert output["error"]["category"] == "connections_unavailable"
    assert "turned off in this installation, or it predates them" in output["error"]["message"]


def test_a_bare_404_with_connections_on_is_passed_on(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", "/v1/connections"): _MISSING,
            _STATUS: httpx.Response(200, json={"enabled": True, "message": ""}),
        }
    )

    code, output = _run(["connections", "list"], capsys)

    assert code != 0
    assert output["error"]["category"] == "api_request_rejected"
    assert "404" in output["error"]["message"]


def test_a_missing_connection_is_still_reported_as_missing(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", "/v1/connections/absent"): httpx.Response(
                404,
                json={
                    "detail": {"code": "connection_not_found", "message": "No connection absent."}
                },
            )
        }
    )

    code, output = _run(["connections", "show", "absent"], capsys)

    assert code != 0
    assert output["error"]["code"] == "connection_not_found"
    assert not any(request.url.path == "/v1/connections-status" for request in fake.requests)


class _AnyQuery(dict[tuple[str, str], Any]):
    """Routes answered whatever query the request carries."""

    def __missing__(self, key: tuple[str, str]) -> Any:
        method, target = key
        return self[(method, target.split("?", 1)[0])]


_ACTIVITY = {
    "items": [
        {
            "id": "call_" + "1" * 24,
            "at": "2026-11-19T09:05:00+00:00",
            "application_id": "sample-agent",
            "application_name": "Sample Agent",
            "slot": "warehouse",
            "method": "POST",
            "path": "/api/v2/statements",
            "status": 200,
            "error": None,
            "request_bytes": 120,
            "response_bytes": 2048,
            "duration_ms": 231,
        },
        {
            "id": "call_" + "2" * 24,
            "at": "2026-11-19T09:00:00+00:00",
            "application_id": "gone-agent",
            "application_name": None,
            "slot": "warehouse",
            "method": "GET",
            "path": "/api/v2/other",
            "status": None,
            "error": "path_not_allowed",
            "request_bytes": 0,
            "response_bytes": 0,
            "duration_ms": 1,
        },
    ],
    "next_cursor": "MjAyNi0xMS0xOQ==",
    "since": "2026-11-18T09:10:00+00:00",
    "retention_days": 30,
}


def test_activity_prints_the_calls_newest_first_and_how_to_read_on(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {("GET", "/v1/connections/warehouse/activity"): httpx.Response(200, json=_ACTIVITY)}
    )

    assert main(["cloud", "connections", "activity", "warehouse"]) == 0

    [read] = fake.requests
    assert (read.method, read.url.path, dict(read.url.params)) == (
        "GET",
        "/v1/connections/warehouse/activity",
        {},
    )
    assert capsys.readouterr().out.splitlines() == [
        "TIME                       AGENT         SLOT       REQUEST                  RESULT"
        "            DURATION",
        # The Agent's name, or its id for one Cloud no longer has.
        "2026-11-19T09:05:00+00:00  Sample Agent  warehouse  POST /api/v2/statements  200"
        "               231 ms",
        "2026-11-19T09:00:00+00:00  gone-agent    warehouse  GET /api/v2/other        path_not_allowed"
        "  1 ms",
        "",
        "2 calls since 2026-11-18T09:10:00+00:00.",
        # The window as an instant: a relative --since would read a later one next time.
        "More calls follow: pass --cursor MjAyNi0xMS0xOQ== --since 2026-11-18T09:10:00+00:00.",
    ]


def test_activity_sends_since_limit_and_cursor_and_prints_json(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        _AnyQuery(
            {("GET", "/v1/connections/warehouse/activity"): httpx.Response(200, json=_ACTIVITY)}
        )
    )

    code, output = _run(
        [
            "connections",
            "activity",
            "warehouse",
            "--since",
            "7d",
            "--limit",
            "2",
            "--cursor",
            "c1",
            "--json",
        ],
        capsys,
    )

    assert code == 0
    assert output["operation"] == "connections.activity"
    assert output["result"] == _ACTIVITY
    [read] = fake.requests
    sent = dict(read.url.params)
    since = datetime.fromisoformat(sent.pop("since"))
    assert abs(datetime.now(UTC) - timedelta(days=7) - since) < timedelta(minutes=1)
    assert sent == {"limit": "2", "cursor": "c1"}

    assert (
        main(["cloud", "connections", "activity", "warehouse", "--since", "2026-11-19T09:00:00Z"])
        == 0
    )
    assert dict(fake.requests[-1].url.params) == {"since": "2026-11-19T09:00:00+00:00"}


def test_activity_with_no_calls_says_so(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    empty = {"items": [], "next_cursor": None, "since": "2026-11-18T09:10:00+00:00"}
    cloud({("GET", "/v1/connections/warehouse/activity"): httpx.Response(200, json=empty)})

    assert main(["cloud", "connections", "activity", "warehouse"]) == 0
    assert capsys.readouterr().out == "No calls since 2026-11-18T09:10:00+00:00.\n"


def test_a_connection_checks_calls_are_labelled_as_such(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    checked = {
        **_ACTIVITY["items"][0],
        "application_id": "",
        "application_name": None,
        "slot": "check",
    }
    answer = {"items": [checked], "next_cursor": None, "since": _ACTIVITY["since"]}
    cloud({("GET", "/v1/connections/warehouse/activity"): httpx.Response(200, json=answer)})

    assert main(["cloud", "connections", "activity", "warehouse"]) == 0
    row = capsys.readouterr().out.splitlines()[1]
    assert row.split("  ")[1:3] == ["connection check", "check"]


def test_activity_names_a_refused_time_range_in_words(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    refused = {"detail": {"code": "invalid_time_range", "message": "until must be after since."}}
    cloud(
        _AnyQuery(
            {("GET", "/v1/connections/warehouse/activity"): httpx.Response(422, json=refused)}
        )
    )

    code, output = _run(["connections", "activity", "warehouse", "--since", "1h"], capsys)

    assert code != 0
    assert "Cayu Cloud refused the time range: give --since" in output["error"]["message"]


def test_activity_refuses_a_cursor_a_cloud_before_paging_would_ignore(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    # Such a Cloud answers only `items`, the first page again.
    older = {"items": _ACTIVITY["items"]}
    cloud(
        _AnyQuery({("GET", "/v1/connections/warehouse/activity"): httpx.Response(200, json=older)})
    )

    code, output = _run(["connections", "activity", "warehouse", "--cursor", "c1"], capsys)

    assert code != 0
    assert output["error"]["category"] == "activity_paging_unavailable"
    assert "doesn't page a connection's activity" in output["error"]["message"]

    # Without --cursor its answer still prints.
    assert main(["cloud", "connections", "activity", "warehouse"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "2 calls."


@pytest.mark.parametrize(
    "arguments",
    [
        ["--since", "yesterday"],
        ["--since", "2026-11-19T09:00:00"],
        ["--since", "0d"],
        ["--limit", "0"],
        ["--limit", "501"],
    ],
)
def test_activity_refuses_a_malformed_since_or_limit_before_asking_cloud(
    cloud, capsys: pytest.CaptureFixture[str], arguments: list[str]
) -> None:
    fake = cloud({})

    code, output = _run(["connections", "activity", "warehouse", *arguments], capsys)

    assert (code, output["error"]["category"]) == (2, "invalid_input")
    assert fake.requests == []
