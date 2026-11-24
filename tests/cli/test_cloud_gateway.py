"""`cayu cloud gateway keys|models|limits|usage` over Cayu Cloud's Gateway management API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.cli.test_cloud_api_keys import cloud as _cloud

from cayu.cli import main

cloud = _cloud

_KEYS = "/v1/gateway/keys"
_ME = "/v1/me"
_APPS = "/v1/applications?lifecycle=all"
_AGENTS = {"items": [{"id": "support-desk", "name": "Support desk"}]}
_READ = "/v1/gateway/policies/read"
_EDIT = "/v1/gateway/policies/edit"
_ALERTS_READ = "/v1/gateway/alerts/configurations/read"
_ALERTS = "/v1/gateway/alerts/configurations"
_TOKEN = "cayu_gw_key1.secret-token-value"
_ALLOWANCE = "/v1/gateway/delegations/mine"
_PERSONAL = "gateway_personal_keys_available"
_IDENTITY = {
    "organization_id": "org_1",
    "gateway_keys_available": True,
    "gateway_policy_available": True,
    "gateway_usage_available": True,
    "gateway_catalog_available": True,
}


def _key(**changes: object) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "key_id": "key1",
        "name": "ci-evals",
        "organization_id": "org_1",
        "agent_id": None,
        "owner_kind": "service",
        "owner_id": "org_1",
        "scopes": ["inference"],
        "allowed_models": ["claude-sonnet", "claude-haiku"],
        "expires_at": "2027-01-31T00:00:00Z",
        "revision": 3,
        "revoked": False,
        **changes,
    }


def _refusal(status: int, code: str, message: str, **fields: object) -> httpx.Response:
    return httpx.Response(
        status, json={"error": {"code": code, "message": message, "retryable": False, **fields}}
    )


def _policy(
    scope: str,
    scope_id: str,
    policy_id: str,
    *,
    models: list[str],
    default: str | None = None,
    caps: list[tuple[str, int]] = (),  # type: ignore[assignment]
    lock: bool = False,
) -> dict[str, Any]:
    rules: dict[str, Any] = {
        "schema_version": 1,
        "allowed_models": models,
        "default_model": default,
        "hard_limits": [
            {"schema_version": 1, "period": period, "ceiling_nano_usd": amount}
            for period, amount in caps
        ],
    }
    if lock:
        rules["model_lock"] = True
    return {
        "schema_version": 1,
        "binding": {
            "schema_version": 1,
            "policy_id": policy_id,
            "revision": 1,
            "scope": scope,
            "scope_id": scope_id,
            "window_id": None,
        },
        "rules": rules,
    }


def _row(scope: str, scope_id: str, *policies: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "principal": {
            "schema_version": 1,
            "scope": scope,
            "scope_id": scope_id,
            "principal_id": f"{scope}:{scope_id}",
        },
        "policies": list(policies),
    }


_SCHEDULE = {"schema_version": 1, "expected": {"organization_id": "org_1"}, "windows": []}
_ORG_POLICY = _policy(
    "organization",
    "org_1",
    "org-default",
    models=["claude-sonnet", "claude-haiku"],
    default="claude-sonnet",
    caps=[("monthly", 500_000_000_000)],
)


def _configuration(*rows: dict[str, Any] | None, schedule: object = _SCHEDULE) -> dict[str, Any]:
    keys = [
        [row["principal"]["scope"], row["principal"]["scope_id"]]
        if row is not None
        else ["agent", "support-desk"]
        for row in rows
    ]
    return {
        "schema_version": 1,
        "organization_id": "org_1",
        "keys": keys,
        "schedule": schedule,
        "rows": list(rows),
    }


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", "gateway", *arguments])
    return code, json.loads(capsys.readouterr().out)


def _body(request: httpx.Request) -> Any:
    return json.loads(request.content)


# Keys


def test_create_prints_the_token_once_and_sends_the_key_shape(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", _KEYS): httpx.Response(
                201, json={"key": _key(agent_id="support-desk"), "secret": _TOKEN}
            ),
        }
    )

    code, output = _run(
        [
            "keys",
            "create",
            "--name",
            "ci-evals",
            "--model",
            "claude-sonnet",
            "--model",
            "claude-haiku",
            "--model",
            "claude-sonnet",
            "--expires",
            "2027-01-31T00:00:00Z",
            "--application",
            "Support desk",
            "--scope",
            "inference",
            "--scope",
            "receipts",
        ],
        capsys,
    )

    assert code == 0
    (create,) = [request for request in fake.requests if request.method == "POST"]
    assert _body(create) == {
        "name": "ci-evals",
        "owner_kind": "service",
        "agent_id": "support-desk",
        "expires_at": "2027-01-31T00:00:00+00:00",
        "allowed_models": ["claude-sonnet", "claude-haiku"],
        "scopes": ["inference", "receipts"],
    }
    assert output["operation"] == "gateway.keys.create"
    assert output["result"]["token"] == _TOKEN
    assert output["result"]["key"]["key_id"] == "key1"
    assert output["organization"]["id"] == "org_test"


def test_create_writes_the_token_to_a_new_private_file(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cloud(
        {
            ("GET", _ME): httpx.Response(200, json={**_IDENTITY, _PERSONAL: True}),
            ("POST", _KEYS): httpx.Response(201, json={"key": _key(), "secret": _TOKEN}),
        }
    )
    key_file = tmp_path / "gateway-token"

    code, output = _run(
        [
            "keys",
            "create",
            "--name",
            "ci-evals",
            "--model",
            "claude-sonnet",
            "--expires",
            "30d",
            "--personal",
            "--key-file",
            str(key_file),
        ],
        capsys,
    )

    assert code == 0
    assert key_file.read_text() == _TOKEN + "\n"
    assert key_file.stat().st_mode & 0o777 == 0o600
    assert output["result"]["key_file"] == str(key_file)
    assert _TOKEN not in json.dumps(output)


def test_create_refuses_an_existing_key_file_before_asking_cloud(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = cloud({})
    key_file = tmp_path / "gateway-token"
    key_file.write_text("keep me\n")

    code, output = _run(
        [
            "keys",
            "create",
            "--name",
            "n",
            "--model",
            "m",
            "--expires",
            "1d",
            "--key-file",
            str(key_file),
        ],
        capsys,
    )

    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert "already exists" in output["error"]["message"]
    assert key_file.read_text() == "keep me\n"
    assert fake.requests == []


def test_a_refused_personal_key_removes_the_key_file_and_says_personal_keys_are_off(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # A Cloud whose `/v1/me` doesn't say (an older one) is asked, and its refusal explained.
    cloud(
        {
            ("GET", _ME): httpx.Response(200, json=_IDENTITY),
            ("POST", _KEYS): _refusal(
                403,
                "personal_keys_unavailable",
                "Personal keys aren't available on this install yet.",
            ),
        }
    )
    key_file = tmp_path / "gateway-token"

    code, output = _run(
        [
            "keys",
            "create",
            "--name",
            "mine",
            "--model",
            "m",
            "--expires",
            "1d",
            "--personal",
            "--key-file",
            str(key_file),
        ],
        capsys,
    )

    assert code == 2
    assert output["error"]["code"] == "personal_keys_unavailable"
    assert "aren't available on this install yet" in output["error"]["message"]
    assert "admin can create a key for an Agent" in output["error"]["message"]
    assert "Only an Organization admin" not in output["error"]["message"]
    assert not key_file.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["keys", "create", "--name", "n", "--model", "m", "--expires", "1d", "--personal"],
        ["keys", "list", "--mine", "--json"],
        ["keys", "allowance"],
    ],
)
def test_personal_key_commands_are_refused_where_the_install_offers_none(
    cloud, capsys: pytest.CaptureFixture[str], arguments: list[str]
) -> None:
    fake = cloud({("GET", _ME): httpx.Response(200, json={**_IDENTITY, _PERSONAL: False})})

    code, output = _run(arguments, capsys)

    assert code == 2
    assert output["error"]["category"] == "personal_keys_unavailable"
    assert [request.url.path for request in fake.requests] == [_ME]


def test_allowance_on_an_install_without_delegations_says_personal_keys_are_not_offered(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", _ME): httpx.Response(200, json={**_IDENTITY, _PERSONAL: True}),
            ("GET", _ALLOWANCE): httpx.Response(404, json={"detail": "Not Found"}),
        }
    )

    code, output = _run(["keys", "allowance"], capsys)

    assert code == 2
    assert output["error"]["category"] == "personal_keys_unavailable"
    assert "HTTP 404" not in output["error"]["message"]


def test_a_refusal_without_a_gateway_code_says_who_manages_the_gateway(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", _KEYS): _refusal(
                403,
                "forbidden",
                "The requested inference operation is not permitted.",
                request_id="key-management",
            )
        }
    )

    code, output = _run(["keys", "list", "--json"], capsys)

    assert code == 2
    assert output["error"]["code"] == "forbidden"
    assert "Only an Organization admin manages the Gateway" in output["error"]["message"]


def test_a_refused_key_by_id_is_reported_as_not_found_or_not_allowed(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", f"{_KEYS}/key9"): _refusal(
                403,
                "forbidden",
                "The requested inference operation is not permitted.",
                request_id="key-management",
            )
        }
    )

    code, output = _run(["keys", "show", "key9"], capsys)

    assert code == 2
    assert "Key key9 wasn't found, or you aren't allowed to manage it" in output["error"]["message"]
    assert "Only an Organization admin manages" not in output["error"]["message"]


def test_keys_show_prints_the_key(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud({("GET", f"{_KEYS}/key1"): httpx.Response(200, json=_key())})

    code, output = _run(["keys", "show", "key1"], capsys)

    assert (code, output["operation"], output["result"]) == (0, "gateway.keys.show", _key())


@pytest.mark.parametrize(
    ("answer", "code", "said"),
    [
        (
            httpx.Response(
                503,
                headers={"Retry-After": "5"},
                json={
                    "error": {
                        "code": "not_enrolled",
                        "message": "This Agent isn't enrolled in the Gateway yet. Retry in a "
                        "few seconds.",
                        "scope": "agent",
                        "retryable": True,
                    }
                },
            ),
            "not_enrolled",
            "Retry in a few seconds. Nothing was changed.",
        ),
        (
            _refusal(
                409, "no_models", "This Organization allows no models. Choose its models first."
            ),
            "no_models",
            "cayu cloud gateway models set --model MODEL",
        ),
    ],
)
def test_create_explains_a_scope_the_gateway_has_not_enrolled(
    cloud, capsys: pytest.CaptureFixture[str], answer: httpx.Response, code: str, said: str
) -> None:
    cloud({("GET", _APPS): httpx.Response(200, json=_AGENTS), ("POST", _KEYS): answer})

    exit_code, output = _run(
        [
            "keys",
            "create",
            "--name",
            "n",
            "--model",
            "m",
            "--expires",
            "1d",
            "--application",
            "support-desk",
        ],
        capsys,
    )

    assert exit_code == 2
    assert output["error"]["code"] == code
    assert said in output["error"]["message"]
    assert "may have created" not in output["error"]["message"]


def test_a_lost_create_answer_says_the_key_may_exist(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("POST", _KEYS): httpx.Response(500, json={"detail": "boom"})})

    code, output = _run(
        ["keys", "create", "--name", "n", "--model", "m", "--expires", "1d"], capsys
    )

    assert code == 2
    assert "may have created the key 'n'" in output["error"]["message"]
    assert "keys list" in output["error"]["message"]


def test_create_with_caps_needs_a_key_file(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud({})

    code, output = _run(
        [
            "keys",
            "create",
            "--name",
            "n",
            "--model",
            "m",
            "--expires",
            "1d",
            "--cap",
            "monthly=25",
        ],
        capsys,
    )

    assert code == 2
    assert "--key-file" in output["error"]["message"]
    assert fake.requests == []


def test_create_with_caps_sets_the_new_keys_own_policy(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    configuration = _configuration(_row("key", "key1"), _row("organization", "org_1", _ORG_POLICY))
    fake = cloud(
        {
            ("POST", _KEYS): httpx.Response(201, json={"key": _key(), "secret": _TOKEN}),
            ("GET", _ME): httpx.Response(200, json=_IDENTITY),
            ("POST", _READ): httpx.Response(200, json=configuration),
            ("POST", _EDIT): httpx.Response(200, json={"schema_version": 1, "change": {}}),
        }
    )

    code, output = _run(
        [
            "keys",
            "create",
            "--name",
            "ci-evals",
            "--model",
            "claude-sonnet",
            "--expires",
            "1d",
            "--cap",
            "monthly=25",
            "--cap",
            "daily=0.5",
            "--key-file",
            str(tmp_path / "token"),
        ],
        capsys,
    )

    assert code == 0
    read, edit = (
        request for request in fake.requests if request.url.path.startswith("/v1/gateway/p")
    )
    assert _body(read) == {"keys": [["key", "key1"], ["organization", "org_1"]]}
    body = _body(edit)
    assert body["before"] == {
        "schema_version": 1,
        "schedule": _SCHEDULE,
        "scopes": configuration["rows"],
    }
    (change,) = body["changes"]
    assert change["scope"] == "key"
    assert change["scope_id"] == "key1"
    assert change["before"] is None
    assert change["policy_id"] == f"cli-{body['operation_id']}"
    assert change["rules"]["allowed_models"] == ["claude-sonnet", "claude-haiku"]
    assert change["rules"]["hard_limits"] == [
        {"schema_version": 1, "period": "daily", "ceiling_nano_usd": 500_000_000},
        {"schema_version": 1, "period": "monthly", "ceiling_nano_usd": 25_000_000_000},
    ]
    assert output["result"]["limits"] == {"schema_version": 1, "change": {}}


def test_list_shows_a_table_or_the_page_and_reads_an_agents_keys(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    page = {"items": [_key(agent_id="support-desk")], "next_cursor": "key1"}
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", f"{_KEYS}?limit=1&after=key0&agent=support-desk"): httpx.Response(
                200, json=page
            ),
            ("GET", _ME): httpx.Response(200, json={**_IDENTITY, _PERSONAL: True}),
            ("GET", f"{_KEYS}/mine"): httpx.Response(200, json={"items": [], "next_cursor": None}),
        }
    )

    arguments = ["keys", "list", "--application", "support-desk", "--limit", "1"]
    assert main(["cloud", "gateway", *arguments, "--cursor", "key0"]) == 0
    table = capsys.readouterr().out.splitlines()
    assert table[0].split() == ["KEY", "NAME", "OWNER", "AGENT", "MODELS", "EXPIRES", "REVOKED"]
    assert table[1].split()[:4] == ["key1", "ci-evals", "service", "support-desk"]
    assert table[-1] == "More: --cursor key1"

    code, output = _run(["keys", "list", "--mine", "--json"], capsys)
    assert code == 0
    assert output == {
        "ok": True,
        "operation": "gateway.keys.list",
        "result": {"items": [], "next_cursor": None},
    }
    assert fake.requests[-1].url.path == f"{_KEYS}/mine"


def test_rotate_starts_from_the_current_revision_and_prints_the_new_token(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", f"{_KEYS}/key1"): httpx.Response(200, json=_key()),
            ("POST", f"{_KEYS}/key1/rotate"): httpx.Response(
                200, json={"key": _key(key_id="key2", revision=1), "secret": _TOKEN}
            ),
        }
    )

    code, output = _run(["keys", "rotate", "key1"], capsys)

    assert code == 0
    assert _body(fake.requests[-1]) == {"expected_revision": 3}
    assert output["result"] == {
        "key": _key(key_id="key2", revision=1),
        "token": _TOKEN,
        "replaced_key_id": "key1",
    }


def test_restrict_keeps_what_is_not_given_and_revoke_sends_the_revision(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", f"{_KEYS}/key1"): httpx.Response(200, json=_key()),
            ("PATCH", f"{_KEYS}/key1"): httpx.Response(200, json=_key(revision=4)),
            ("POST", f"{_KEYS}/key1/revoke"): httpx.Response(200, json=_key(revoked=True)),
        }
    )

    code, output = _run(["keys", "restrict", "key1", "--model", "claude-haiku"], capsys)
    assert code == 0
    assert _body(fake.requests[-1]) == {
        "expected_revision": 3,
        "allowed_models": ["claude-haiku"],
        "scopes": ["inference"],
        "expires_at": "2027-01-31T00:00:00Z",
    }
    assert output["operation"] == "gateway.keys.restrict"

    code, output = _run(["keys", "revoke", "key1"], capsys)
    assert code == 0
    assert _body(fake.requests[-1]) == {"expected_revision": 3}
    assert output["result"]["revoked"] is True


def test_restrict_needs_something_to_narrow(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud({})

    code, output = _run(["keys", "restrict", "key1"], capsys)

    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert fake.requests == []


def test_a_key_conflict_says_to_run_the_command_again(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", f"{_KEYS}/key1"): httpx.Response(200, json=_key()),
            ("POST", f"{_KEYS}/key1/revoke"): _refusal(
                409, "key_conflict", "Reload key metadata and retry."
            ),
        }
    )

    code, output = _run(["keys", "revoke", "key1"], capsys)

    assert code == 2
    assert output["error"]["code"] == "key_conflict"
    assert "Run the command again" in output["error"]["message"]


def test_audit_pages_by_revision_and_allowance_reads_your_own(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", f"{_KEYS}/key1/audit?after=4"): httpx.Response(
                200, json={"items": [], "next_cursor": None}
            ),
            ("GET", _ME): httpx.Response(200, json={**_IDENTITY, _PERSONAL: True}),
            ("GET", _ALLOWANCE): httpx.Response(200, json={"delegation": None}),
        }
    )

    assert _run(["keys", "audit", "key1", "--cursor", "4"], capsys)[0] == 0
    code, output = _run(["keys", "allowance"], capsys)
    assert code == 0
    assert output["result"] == {"delegation": None}
    assert len(fake.requests) == 3


@pytest.mark.parametrize(
    ("identity", "category", "said"),
    [
        ({**_IDENTITY, "gateway_keys_available": False}, "gateway_management_off", "turns it on"),
        (
            {**_IDENTITY, "gateway_catalog_available": False},
            "gateway_management_off",
            "not its model catalog",
        ),
        ({"organization_id": "org_1"}, "gateway_management_unavailable", "older"),
    ],
)
def test_a_cloud_without_gateway_management_says_so(
    cloud, capsys: pytest.CaptureFixture[str], identity: dict[str, Any], category: str, said: str
) -> None:
    cloud(
        {
            ("GET", "/v1/gateway/catalog"): httpx.Response(404, json={"detail": "Not Found"}),
            ("GET", _ME): httpx.Response(200, json=identity),
        }
    )

    code, output = _run(["models", "catalog"], capsys)

    assert code == 2
    assert output["error"]["category"] == category
    assert said in output["error"]["message"]


def test_restrict_refuses_what_would_widen_the_key_before_asking_cloud(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({("GET", f"{_KEYS}/key1"): httpx.Response(200, json=_key())})

    code, output = _run(["keys", "restrict", "key1", "--model", "claude-opus"], capsys)
    assert code == 2
    assert output["error"]["category"] == "gateway_key_widening"
    assert "not claude-opus" in output["error"]["message"]

    code, output = _run(["keys", "restrict", "key1", "--expires", "2027-06-01T00:00:00Z"], capsys)
    assert code == 2
    assert output["error"]["category"] == "gateway_key_widening"
    assert "earlier --expires" in output["error"]["message"]

    code, output = _run(["keys", "restrict", "key1", "--scope", "receipts"], capsys)
    assert code == 2
    assert "not receipts" in output["error"]["message"]
    assert {request.method for request in fake.requests} == {"GET"}


def test_a_revoked_key_is_revoked_once_and_not_changed(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({("GET", f"{_KEYS}/key1"): httpx.Response(200, json=_key(revoked=True))})

    code, output = _run(["keys", "revoke", "key1"], capsys)
    assert code == 0
    assert output["result"]["already_revoked"] is True

    for arguments in (["restrict", "key1", "--model", "claude-haiku"], ["rotate", "key1"]):
        code, output = _run(["keys", *arguments], capsys)
        assert code == 2
        assert output["error"]["category"] == "gateway_key_revoked"
    assert {request.method for request in fake.requests} == {"GET"}


def test_an_unknown_rotation_outcome_names_the_key_and_the_old_token(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cloud(
        {
            ("GET", f"{_KEYS}/key1"): httpx.Response(200, json=_key()),
            ("POST", f"{_KEYS}/key1/rotate"): _refusal(
                409,
                "key_publication_unknown",
                "Inspect and revoke this key before issuing a replacement.",
                key_id="key2",
            ),
        }
    )
    key_file = tmp_path / "token"

    code, output = _run(["keys", "rotate", "key1", "--key-file", str(key_file)], capsys)

    assert code == 2
    message = output["error"]["message"]
    assert output["error"]["code"] == "key_publication_unknown"
    assert "keys audit key2" in message
    assert "keys revoke key2" in message
    assert "Key key1's old token may already have stopped working" in message
    assert not key_file.exists()


def test_a_key_file_that_cannot_be_written_says_to_rotate(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cayu.cli import _cloud_api_keys

    cloud({("POST", _KEYS): httpx.Response(201, json={"key": _key(), "secret": _TOKEN})})

    def full(descriptor: int, mode: str) -> Any:
        import os

        os.close(descriptor)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_cloud_api_keys.os, "fdopen", full)
    key_file = tmp_path / "token"

    code, output = _run(
        [
            "keys",
            "create",
            "--name",
            "n",
            "--model",
            "m",
            "--expires",
            "1d",
            "--key-file",
            str(key_file),
        ],
        capsys,
    )

    assert code == 2
    assert output["error"]["category"] == "local_state_unavailable"
    assert "rotate the key" in output["error"]["message"]
    assert _TOKEN not in json.dumps(output)
    assert not key_file.exists()


def test_an_interrupted_create_removes_the_key_file_and_says_the_key_may_exist(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def interrupted(_request: httpx.Request) -> httpx.Response:
        raise KeyboardInterrupt

    cloud({("POST", _KEYS): interrupted})
    key_file = tmp_path / "token"

    with pytest.raises(KeyboardInterrupt):
        main(
            [
                "cloud",
                "gateway",
                "keys",
                "create",
                "--name",
                "n",
                "--model",
                "m",
                "--expires",
                "1d",
                "--key-file",
                str(key_file),
            ]
        )

    assert "may have created the key 'n'" in capsys.readouterr().err
    assert not key_file.exists()


def test_a_change_says_where_it_acts_and_refuses_another_organization(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", f"{_KEYS}/key1"): httpx.Response(200, json=_key()),
            ("POST", f"{_KEYS}/key1/revoke"): httpx.Response(200, json=_key(revoked=True)),
        }
    )

    assert main(["cloud", "gateway", "keys", "revoke", "key1"]) == 0
    captured = capsys.readouterr()
    assert "acting in Organization Test Organization (org_test)" in captured.err
    assert json.loads(captured.out)["organization"]["id"] == "org_test"

    code, output = _run(["keys", "revoke", "key1", "--organization", "org_other"], capsys)
    assert code == 2
    assert "org_other" in output["error"]["message"]
    assert len(fake.requests) == 2


def test_a_refusal_with_its_own_code_keeps_the_gateways_message(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("POST", _KEYS): _refusal(
                403,
                "model_not_allowed",
                "The requested model is not allowed by a model policy; choose an allowed model "
                "or ask an administrator to allow it.",
                request_id="key-management",
                scope="key",
            )
        }
    )

    code, output = _run(
        ["keys", "create", "--name", "n", "--model", "claude-opus", "--expires", "1d"], capsys
    )

    assert code == 2
    message = output["error"]["message"]
    assert output["error"]["code"] == "model_not_allowed"
    assert "not allowed by a model policy" in message
    assert "cayu cloud gateway models show" in message
    assert "Only an Organization admin" not in message
    assert "wasn't found" not in message


# Model policy


def test_models_show_gives_the_models_an_agent_may_use(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _policy("agent", "support-desk", "a1", models=["claude-haiku"], default="claude-haiku")
    cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", _ME): httpx.Response(200, json=_IDENTITY),
            ("POST", _READ): httpx.Response(
                200,
                json=_configuration(
                    _row("agent", "support-desk", agent),
                    _row("organization", "org_1", _ORG_POLICY),
                ),
            ),
        }
    )

    code, output = _run(["models", "show", "--application", "support-desk", "--json"], capsys)
    assert code == 0
    assert output["result"]["allowed_models"] == ["claude-haiku"]
    assert output["result"]["default_model"] == "claude-haiku"
    assert output["result"]["locked_to"] is None

    assert main(["cloud", "gateway", "models", "show", "--application", "support-desk"]) == 0
    text = capsys.readouterr().out
    assert "Models it may use: claude-haiku" in text
    assert "Default model: claude-haiku" in text


def test_models_set_replaces_the_organizations_models_and_default(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", _ME): httpx.Response(200, json=_IDENTITY),
            ("POST", _READ): httpx.Response(
                200, json=_configuration(_row("organization", "org_1", _ORG_POLICY))
            ),
            ("POST", _EDIT): httpx.Response(200, json={"schema_version": 1}),
        }
    )

    code, output = _run(
        [
            "models",
            "set",
            "--model",
            "claude-haiku",
            "--model",
            "claude-opus",
            "--default",
            "claude-opus",
            "--lock",
        ],
        capsys,
    )

    assert code == 0
    assert _body(fake.requests[1]) == {"keys": [["organization", "org_1"]]}
    (change,) = _body(fake.requests[-1])["changes"]
    assert change["policy_id"] == "org-default"
    assert change["before"] == _ORG_POLICY
    assert change["rules"] == {
        **_ORG_POLICY["rules"],
        "allowed_models": ["claude-haiku", "claude-opus"],
        "default_model": "claude-opus",
        "model_lock": True,
    }
    assert output["operation"] == "gateway.models.set"


def test_models_set_checks_the_default_and_the_lock_before_asking_cloud(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", _ME): httpx.Response(200, json=_IDENTITY),
            ("POST", _READ): httpx.Response(
                200, json=_configuration(_row("organization", "org_1", _ORG_POLICY))
            ),
        }
    )

    code, output = _run(["models", "set", "--default", "claude-opus"], capsys)
    assert code == 2
    assert "isn't among its models" in output["error"]["message"]

    code, output = _run(["models", "set", "--key", "key1", "--lock"], capsys)
    assert code == 2
    assert all(request.url.path != _EDIT for request in fake.requests)


def test_models_set_on_an_agent_the_gateway_has_not_enrolled_says_why(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", _ME): httpx.Response(200, json=_IDENTITY),
            ("POST", _READ): httpx.Response(
                200, json=_configuration(None, _row("organization", "org_1"))
            ),
        }
    )

    code, output = _run(
        ["models", "set", "--application", "support-desk", "--default", "claude-haiku"], capsys
    )

    assert code == 2
    assert output["error"]["category"] == "gateway_no_models"
    assert "models set --model" in output["error"]["message"]
    assert all(request.url.path != _EDIT for request in fake.requests)


def test_models_catalog_lists_the_gateways_models(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = {"schema_version": 1, "object": "list", "data": [{"id": "claude-haiku"}]}
    cloud({("GET", "/v1/gateway/catalog"): httpx.Response(200, json=catalog)})

    code, output = _run(["models", "catalog"], capsys)

    assert (code, output["operation"], output["result"]) == (0, "gateway.models.catalog", catalog)


def test_a_missing_route_whose_check_fails_says_so(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", "/v1/gateway/catalog"): httpx.Response(404, json={"detail": "Not Found"}),
            ("GET", _ME): httpx.Response(502, json={"detail": "bad gateway"}),
        }
    )

    code, output = _run(["models", "catalog"], capsys)

    assert code == 2
    assert output["error"]["category"] == "gateway_management_unknown"
    assert "older" not in output["error"]["message"]


def test_models_show_for_a_key_keeps_to_the_keys_own_models(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    # The key has no policy of its own: the Organization's allows both models, the key one.
    cloud(
        {
            ("GET", _ME): httpx.Response(200, json=_IDENTITY),
            ("POST", _READ): httpx.Response(
                200,
                json=_configuration(
                    _row("key", "key1"), _row("organization", "org_1", _ORG_POLICY)
                ),
            ),
            ("GET", f"{_KEYS}/key1"): httpx.Response(
                200, json=_key(allowed_models=["claude-haiku"])
            ),
        }
    )

    code, output = _run(["models", "show", "--key", "key1", "--json"], capsys)

    assert code == 0
    assert output["result"]["allowed_models"] == ["claude-haiku"]


# Limits


def test_limits_set_keeps_other_periods_and_creates_an_agent_policy(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", _ME): httpx.Response(200, json=_IDENTITY),
            ("POST", _READ): httpx.Response(
                200,
                json=_configuration(
                    _row("agent", "support-desk"), _row("organization", "org_1", _ORG_POLICY)
                ),
            ),
            ("POST", _EDIT): httpx.Response(200, json={"schema_version": 1}),
        }
    )

    code, _ = _run(["limits", "set", "--application", "support-desk", "--cap", "weekly=10"], capsys)

    assert code == 0
    (change,) = _body(fake.requests[-1])["changes"]
    assert change["scope"] == "agent"
    assert change["rules"]["allowed_models"] == ["claude-sonnet", "claude-haiku"]
    assert change["rules"]["hard_limits"] == [
        {"schema_version": 1, "period": "weekly", "ceiling_nano_usd": 10_000_000_000}
    ]


def test_limits_set_removes_a_cap_and_refuses_a_lifetime_policy(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    lifetime = _policy("key", "key1", "eval", models=["m"], caps=[("lifetime", 5)])
    fake = cloud(
        {
            ("GET", _ME): httpx.Response(200, json=_IDENTITY),
            ("POST", _READ): httpx.Response(
                200, json=_configuration(_row("organization", "org_1", _ORG_POLICY))
            ),
            ("POST", _EDIT): httpx.Response(200, json={"schema_version": 1}),
        }
    )
    code, _ = _run(["limits", "set", "--remove-cap", "monthly"], capsys)
    assert code == 0
    (change,) = _body(fake.requests[-1])["changes"]
    assert change["rules"]["hard_limits"] == []

    fake.routes[("POST", _READ)] = httpx.Response(
        200,
        json=_configuration(_row("key", "key1", lifetime), _row("organization", "org_1")),
    )
    code, output = _run(["limits", "set", "--key", "key1", "--cap", "daily=1"], capsys)
    assert code == 2
    assert "lifetime cap" in output["error"]["message"]


def test_an_unknown_policy_outcome_names_the_operation(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", _ME): httpx.Response(200, json=_IDENTITY),
            ("POST", _READ): httpx.Response(
                200, json=_configuration(_row("organization", "org_1", _ORG_POLICY))
            ),
            ("POST", _EDIT): _refusal(
                409,
                "outcome_unknown",
                "Request outcome requires reconciliation; do not redispatch.",
            ),
        }
    )

    code, output = _run(["limits", "set", "--cap", "daily=1"], capsys)

    assert code == 2
    operation = _body(fake.requests[-1])["operation_id"]
    assert f"operation {operation}" in output["error"]["message"]
    assert "may have been applied" in output["error"]["message"]


def test_limits_alerts_sets_a_keys_budgets_under_its_owner(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    before = {
        "schema_version": 1,
        "organization_id": "org_1",
        "scope": "key",
        "scope_id": "key:key1",
        "revision": 2,
        "rules": {
            "schema_version": 1,
            "budgets": [
                {
                    "schema_version": 1,
                    "period": "daily",
                    "amount_nano_usd": 1,
                    "thresholds_basis_points": [10000],
                },
                {
                    "schema_version": 1,
                    "period": "weekly",
                    "amount_nano_usd": 2,
                    "thresholds_basis_points": [9000],
                },
            ],
        },
    }
    fake = cloud(
        {
            ("GET", _ME): httpx.Response(200, json=_IDENTITY),
            ("POST", _ALERTS_READ): httpx.Response(
                200, json={"configuration": before, "key_id": "key1", "scope_id": "key:key1"}
            ),
            ("POST", _ALERTS): httpx.Response(200, json={"schema_version": 1}),
        }
    )

    code, _ = _run(
        [
            "limits",
            "alerts",
            "--key",
            "key1",
            "--budget",
            "monthly=100",
            "--remove-budget",
            "daily",
            "--thresholds",
            "80,50",
        ],
        capsys,
    )

    assert code == 0
    assert _body(fake.requests[1]) == {"scope": "key", "scope_id": "key1"}
    body = _body(fake.requests[-1])
    assert body["key_id"] == "key1"
    assert body["before"] == before
    assert body["after"]["scope_id"] == "key:key1"
    assert body["after"]["revision"] == 3
    assert body["after"]["rules"]["budgets"] == [
        before["rules"]["budgets"][1],
        {
            "schema_version": 1,
            "period": "monthly",
            "amount_nano_usd": 100_000_000_000,
            "thresholds_basis_points": [5000, 8000],
        },
    ]


def test_limits_show_lists_caps_and_alert_budgets(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", _ME): httpx.Response(200, json=_IDENTITY),
            ("POST", _READ): httpx.Response(
                200, json=_configuration(_row("organization", "org_1", _ORG_POLICY))
            ),
            ("POST", _ALERTS_READ): httpx.Response(200, json={"configuration": None}),
        }
    )

    assert main(["cloud", "gateway", "limits", "show"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[2].split() == ["SCOPE", "POLICY", "PERIOD", "CAP"]
    assert lines[3].split() == ["organization", "org-default", "monthly", "$500.00"]
    assert lines[-1] == "Alert budgets: none"


def test_remove_budget_keeps_the_other_budgets(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    budget = {
        "schema_version": 1,
        "period": "daily",
        "amount_nano_usd": 1,
        "thresholds_basis_points": [10000],
    }
    before = {
        "schema_version": 1,
        "organization_id": "org_1",
        "scope": "organization",
        "scope_id": "org_1",
        "revision": 1,
        "rules": {
            "schema_version": 1,
            "budgets": [budget, {**budget, "period": "monthly", "amount_nano_usd": 9}],
        },
    }
    fake = cloud(
        {
            ("GET", _ME): httpx.Response(200, json=_IDENTITY),
            ("POST", _ALERTS_READ): httpx.Response(200, json={"configuration": before}),
            ("POST", _ALERTS): httpx.Response(200, json={"schema_version": 1}),
        }
    )

    code, _ = _run(["limits", "alerts", "--remove-budget", "daily"], capsys)

    assert code == 0
    body = _body(fake.requests[-1])
    assert "key_id" not in body
    assert body["after"]["rules"]["budgets"] == [
        {**budget, "period": "monthly", "amount_nano_usd": 9}
    ]


# Usage


def _usage_page(items: list[dict[str, Any]], cursor: str | None) -> dict[str, Any]:
    return {
        "organization_id": "org_1",
        "agent_id": "support-desk",
        "items": items,
        "next_cursor": cursor,
        "complete": cursor is None,
        "by_model": [
            {
                "key": "claude-haiku",
                "requests": len(items),
                "customer_cost_nano_usd": 3 * len(items),
            }
        ],
        "by_level": [],
        "by_slot": [],
        "by_evaluation": [{"key": "eval-1", "requests": 1, "customer_cost_nano_usd": 5}],
    }


def _usage(request_id: str, at: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "model_id": "claude-haiku",
        "requested_at": at,
        "execution_status": "succeeded",
        "customer_cost_nano_usd": 3,
        "currency": "USD",
        "measurements": None,
        "model_level": "agent",
        "model_slot": None,
    }


def test_usage_agent_reads_every_page_and_totals_a_window(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    path = "/v1/gateway/agents/support-desk/usage"
    cursor = "a" * 64
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", f"{path}?limit=100"): httpx.Response(
                200, json=_usage_page([_usage("r1", "2026-11-01T00:00:00Z")], cursor)
            ),
            ("GET", f"{path}?limit=100&cursor={cursor}"): httpx.Response(
                200, json=_usage_page([_usage("r2", "2026-11-20T00:00:00Z")], None)
            ),
        }
    )

    code, output = _run(
        ["usage", "agent", "--application", "support-desk", "--all", "--json"], capsys
    )
    assert code == 0
    result = output["result"]
    assert [item["request_id"] for item in result["items"]] == ["r1", "r2"]
    assert result["by_model"] == [
        {"key": "claude-haiku", "requests": 2, "customer_cost_nano_usd": 6}
    ]
    assert result["complete"] is True
    assert result["pages"] == 2

    code, output = _run(
        [
            "usage",
            "agent",
            "--application",
            "support-desk",
            "--since",
            "2026-11-10T00:00:00Z",
            "--json",
        ],
        capsys,
    )
    assert code == 0
    result = output["result"]
    assert [item["request_id"] for item in result["items"]] == ["r2"]
    assert result["customer_cost_nano_usd"] == 3
    assert result["by_level"] == [{"key": "agent", "requests": 1, "customer_cost_nano_usd": 3}]
    assert result["by_evaluation"] == [
        {"key": "eval-1", "requests": 2, "customer_cost_nano_usd": 10}
    ]
    assert result["evaluations_windowed"] is False
    assert len(fake.requests) == 6

    code, output = _run(
        [
            "usage",
            "agent",
            "--application",
            "support-desk",
            "--until",
            "2026-11-10T00:00:00Z",
            "--json",
        ],
        capsys,
    )
    assert code == 0
    assert [item["request_id"] for item in output["result"]["items"]] == ["r1"]


def test_usage_agent_prints_one_page_as_text(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", "/v1/gateway/agents/support-desk/usage"): httpx.Response(
                200, json=_usage_page([_usage("r1", "2026-11-01T00:00:00Z")], "b" * 64)
            ),
        }
    )

    assert main(["cloud", "gateway", "usage", "agent", "--application", "support-desk"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["REQUESTED", "REQUEST", "MODEL", "STATUS", "COST"]
    assert "Total so far: 1 settled requests, $0.000000003" in lines
    assert lines[-1] == f"More: --cursor {'b' * 64}"


def test_usage_request_and_model_status_read_their_routes(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", "/v1/gateway/requests/req_1/evidence"): httpx.Response(200, json={"a": 1}),
            ("GET", "/v1/gateway/agents/support-desk/replicas?limit=5"): httpx.Response(
                200, json={"items": []}
            ),
        }
    )

    code, output = _run(["usage", "request", "req_1", "--evidence"], capsys)
    assert (code, output["result"]) == (0, {"a": 1})
    code, output = _run(
        ["models", "status", "--application", "support-desk", "--limit", "5"], capsys
    )
    assert (code, output["operation"]) == (0, "gateway.models.status")
    assert len(fake.requests) == 3


def test_usage_totals_say_they_are_partial_when_pages_remain(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from cayu.cli import _cloud_gateway

    monkeypatch.setattr(_cloud_gateway, "MAX_USAGE_PAGES", 1)
    path = "/v1/gateway/agents/support-desk/usage"
    cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            # An empty page can still have more after it.
            ("GET", f"{path}?limit=100"): httpx.Response(200, json=_usage_page([], "c" * 64)),
        }
    )

    arguments = ["cloud", "gateway", "usage", "agent", "--application", "support-desk"]
    assert main([*arguments, "--since", "2026-11-01T00:00:00Z"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "Total so far: 0 settled requests, $0.00" in lines
    assert any("the window may hold more" in line for line in lines)
    assert lines[-1] == f"More: --cursor {'c' * 64}"

    code, output = _run([*arguments[2:], "--all", "--json"], capsys)
    assert code == 0
    assert output["result"]["complete"] is False
    assert output["result"]["next_cursor"] == "c" * 64
