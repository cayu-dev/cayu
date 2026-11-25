"""`cayu cloud console` calls Cloud's Console routes and moves a project's source."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.cli.test_cloud_api_keys import cloud as _cloud

from cayu.cli import _cloud_console, _cloud_console_source, main
from cayu.cli._cloud_console_source import (
    console_source_problem,
    source_archive,
    source_digest,
    source_files,
)

cloud = _cloud

_ME = "/v1/me"
_ORGANIZATIONS = "/v1/me/organizations"
_APPS = "/v1/applications?lifecycle=all"
_AGENTS = {"items": [{"id": "customer-health", "name": "Customer health"}]}
_PROJECT = "/v1/console/projects/stp_1"
_IDENTITY = httpx.Response(200, json={"organization_id": "org_1"})
_NAMED = httpx.Response(200, json={"items": [{"id": "org_1", "name": "Acme"}]})
_UPLOAD_URL = "https://uploads.example/bundle"
PULLED = {
    "app/main.py": b"print('hello')\n",
    ".env.example": b"TOKEN=\n",
    ".env.local": b"TOKEN=local\n",
    "data/seed.json": b"{}\n",
    ".gitignore": b"*.log\n",
    "cayu-cloud.toml": b'application = "customer-health"\n',
}
ARCHIVE = source_archive(PULLED)
DIGEST = source_digest(ARCHIVE)


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", "console", *arguments])
    return code, json.loads(capsys.readouterr().out)


def _sent(fake: Any) -> list[tuple[str, str, object, str | None]]:
    return [
        (
            request.method,
            request.url.raw_path.decode(),
            json.loads(request.content)
            if request.content and request.headers.get("content-type") == "application/json"
            else None,
            request.headers.get("Idempotency-Key"),
        )
        for request in fake.requests
    ]


def _project(*revisions: tuple[int, str], **fields: object) -> dict[str, Any]:
    return {
        "id": "stp_1",
        "turn_active": False,
        "revisions": [{"number": number, "content_digest": digest} for number, digest in revisions],
        **fields,
    }


def _pull_routes(archive: bytes = ARCHIVE, digest: str = DIGEST) -> dict[Any, Any]:
    return {
        ("GET", f"{_PROJECT}/source"): httpx.Response(
            200, content=archive, headers={"content-type": "application/gzip"}
        ),
        ("GET", _PROJECT): httpx.Response(200, json=_project((2, "sha256:old"), (3, digest))),
    }


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_help_lists_every_console_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cloud", "console", "--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    for command in (
        "list",
        "insights",
        "activity",
        "rename",
        "delete",
        "create",
        "show",
        "stop",
        "retry",
        "wake",
        "preview",
        "instruct",
        "answer",
        "source",
        "pull",
        "push",
        "events",
        "diff",
        "restore",
        "deploy",
        "pull-request",
        "deploy-merged",
    ):
        assert command in help_text


@pytest.mark.parametrize(
    ("arguments", "method", "path", "body"),
    [
        (["list"], "GET", "/v1/console/projects", None),
        (["insights", "--tz", "America/Los_Angeles"], "GET",
         "/v1/console/insights?tz=America%2FLos_Angeles", None),
        (["activity", "--limit", "5", "--cursor", "c1"], "GET",
         "/v1/console/activity?limit=5&cursor=c1", None),
        (["show", "stp_1"], "GET", _PROJECT, None),
        (["stop", "stp_1"], "POST", f"{_PROJECT}/stop", None),
        (["retry", "stp_1"], "POST", f"{_PROJECT}/retry", None),
        (["wake", "stp_1"], "POST", f"{_PROJECT}/workspace", None),
        (["preview", "stp_1"], "POST", f"{_PROJECT}/preview-session", None),
        (["rename", "stp_1", "--name", "Account health"], "PATCH", _PROJECT,
         {"name": "Account health"}),
        (["create", "--name", "Customer health"], "POST", "/v1/console/projects",
         {"name": "Customer health"}),
        (["answer", "stp_1", "--message", "Billing"], "POST", f"{_PROJECT}/answer",
         {"content": "Billing"}),
        (
            ["instruct", "stp_1", "--message", "Fix", "--comment", "2:app/main.py:12:Why: x"],
            "POST",
            f"{_PROJECT}/messages",
            {
                "content": "Fix",
                "comments": [{"revision": 2, "path": "app/main.py", "line": 12, "body": "Why: x"}],
            },
        ),
        (["events", "stp_1", "--after", "7", "--limit", "2"], "GET",
         f"{_PROJECT}/events?after=7&limit=2", None),
        (["diff", "stp_1", "2"], "GET", f"{_PROJECT}/revisions/2/diff", None),
        (["restore", "stp_1", "2"], "POST", f"{_PROJECT}/revisions/2/restore", None),
        (["pull-request", "stp_1"], "POST", f"{_PROJECT}/pull-request", {"new_branch": False}),
        (["pull-request", "stp_1", "--revision", "3", "--new-branch"], "POST",
         f"{_PROJECT}/pull-request", {"new_branch": True, "revision": 3}),
        (["pull-request", "stp_1", "--status"], "GET", f"{_PROJECT}/pull-request", None),
        (["deploy-merged", "stp_1"], "POST", f"{_PROJECT}/deploy-merged", None),
    ],
)  # fmt: skip
def test_each_command_sends_its_documented_request(
    cloud,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    method: str,
    path: str,
    body: object,
) -> None:
    fake = cloud({(method, path): httpx.Response(200, json={"id": "stp_1"})})

    code, output = _run(arguments, capsys)

    assert code == 0, output
    assert output["operation"] == f"console.{arguments[0]}"
    assert output["result"] == {"id": "stp_1"}
    assert _sent(fake)[-1] == (method, path, body, None)


def test_a_project_change_says_which_organization_it_acts_in(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("DELETE", _PROJECT): httpx.Response(204)})

    code, output = _run(["delete", "stp_1"], capsys)

    assert code == 0, output
    assert output["result"] == {"deleted": "stp_1"}
    assert output["organization"]["id"] == "org_test"


def test_create_from_an_agent_resolves_it_and_names_the_release(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", _ME): _IDENTITY,
            ("GET", _ORGANIZATIONS): _NAMED,
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", "/v1/console/projects"): httpx.Response(201, json={"id": "stp_1"}),
        }
    )

    code, output = _run(["create", "--from-agent", "Customer health", "--release", "dep_1"], capsys)

    assert code == 0, output
    assert _sent(fake)[-1] == (
        "POST",
        "/v1/console/projects",
        {"application_id": "customer-health", "release_id": "dep_1"},
        None,
    )


def test_deploy_names_a_new_agent_by_its_slug_without_looking_it_up(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    # No Agent has the slug yet, and Cloud is never asked for its Agents: a first deploy
    # creates the one it names.
    fake = cloud(
        {
            ("POST", f"{_PROJECT}/deployments"): httpx.Response(
                201, json={"deployment": {"id": "dep_2", "stage_only": True}}
            ),
        }
    )

    code, output = _run(
        [
            "deploy",
            "stp_1",
            "--application",
            "customer-health",
            "--no-promote",
            "--idempotency-key",
            "console-v1",
        ],
        capsys,
    )

    assert code == 0, output
    assert output["idempotency_key"] == "console-v1"
    assert _sent(fake)[-1] == (
        "POST",
        f"{_PROJECT}/deployments",
        {"application_id": "customer-health", "promote": False},
        "console-v1",
    )


def test_deploy_without_a_key_chooses_one_and_reports_a_cloud_that_would_promote(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("POST", f"{_PROJECT}/deployments"): httpx.Response(
                201, json={"deployment": {"id": "dep_2", "application_id": "customer-health"}}
            ),
        }
    )

    code, output = _run(["deploy", "stp_1", "--no-promote"], capsys)

    assert code == 2
    error = output["error"]
    assert error["category"] == "staging_unsupported"
    assert (error["deployment_id"], error["application_id"]) == ("dep_2", "customer-health")
    key = _sent(fake)[-1][3]
    assert key is not None and key.startswith("console-deploy-")
    assert error["idempotency_key"] == key


def test_deploy_merged_sends_a_given_key(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud({("POST", f"{_PROJECT}/deploy-merged"): httpx.Response(201, json={"id": "d"})})

    code, output = _run(["deploy-merged", "stp_1", "--idempotency-key", "merged-1"], capsys)

    assert code == 0, output
    assert output["idempotency_key"] == "merged-1"
    assert _sent(fake)[-1][3] == "merged-1"


def test_a_coded_refusal_reports_its_code_and_a_plain_one_its_reason(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("POST", f"{_PROJECT}/deploy-merged"): httpx.Response(
                409,
                json={
                    "detail": {
                        "code": "pull_request_not_merged",
                        "message": "The project's pull request isn't merged; merge it first.",
                    }
                },
            ),
            ("POST", f"{_PROJECT}/answer"): httpx.Response(
                422, json={"detail": "An answer can't be empty."}
            ),
            ("GET", _PROJECT): httpx.Response(
                404, json={"detail": _cloud_console.CONSOLE_UNAVAILABLE}
            ),
        }
    )

    code, merged = _run(["deploy-merged", "stp_1"], capsys)
    assert (code, merged["error"]["code"]) == (2, "pull_request_not_merged")
    assert "isn't merged" in merged["error"]["message"]
    code, answered = _run(["answer", "stp_1", "--message", " "], capsys)
    assert code == 2
    assert answered["error"]["message"].endswith("An answer can't be empty.")
    code, shown = _run(["show", "stp_1"], capsys)
    assert (code, shown["error"]["category"]) == (2, "console_unavailable")


def test_source_writes_the_archive_to_the_output_file(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cloud(_pull_routes())
    output_file = tmp_path / "out" / "source.tar.gz"

    code, output = _run(["source", "stp_1", "--output", str(output_file)], capsys)

    assert code == 0, output
    assert output_file.read_bytes() == ARCHIVE
    assert output["result"] == {
        "output": str(output_file.resolve()),
        "size_bytes": len(ARCHIVE),
        "source_digest": DIGEST,
    }


def test_pull_writes_the_latest_revision_and_records_it(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cloud(_pull_routes())
    root = tmp_path / "checkout"

    code, output = _run(["pull", "stp_1", "--dir", str(root)], capsys)

    assert code == 0, output
    written = {path: content for path, content in PULLED.items() if path != "data/seed.json"}
    record = {"project_id": "stp_1", "revision": 3, "source_digest": DIGEST}
    assert _tree(root) == {
        **written,
        ".cayu/console-pull.json": (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
    }
    assert output["result"] == {
        "directory": str(root.resolve()),
        "file_count": len(PULLED),
        "kept": [],
        "not_pushed": [".env.local"],
        "project_id": "stp_1",
        "revision": 3,
        "skipped": ["data/seed.json"],
        "source_digest": DIGEST,
    }


def test_pull_refuses_a_directory_holding_anything_but_its_own_earlier_pull(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cloud(_pull_routes())
    (tmp_path / "notes.txt").write_text("mine\n")

    for force in ([], ["--force"]):
        code, output = _run(["pull", "stp_1", "--dir", str(tmp_path), *force], capsys)
        assert (code, output["error"]["category"]) == (2, "console_directory_not_empty")
    assert _tree(tmp_path) == {"notes.txt": b"mine\n"}

    root = tmp_path / "checkout"
    assert _run(["pull", "stp_1", "--dir", str(root)], capsys)[0] == 0
    (root / "stale.py").write_text("old\n")
    (root / ".env").write_text("TOKEN=kept\n")
    code, output = _run(["pull", "stp_1", "--dir", str(root), "--force"], capsys)
    assert code == 0, output
    assert not (root / "stale.py").exists()
    assert (root / ".env").read_text() == "TOKEN=kept\n"


def test_pull_refuses_a_revision_the_previous_builder_stored(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    routes = _pull_routes()
    routes[("GET", _PROJECT)] = httpx.Response(200, json=_project((1, "x")))
    cloud(routes)

    code, output = _run(["pull", "stp_1", "--dir", str(tmp_path / "x")], capsys)

    assert (code, output["error"]["category"]) == (2, "console_pull_unavailable")
    assert not (tmp_path / "x").exists()


def test_a_pull_never_writes_a_repository_or_a_path_another_system_reads_differently(
    tmp_path: Path,
) -> None:
    hostile = {**PULLED, ".git/config": b"x", "vendor/.GIT/hooks/x": b"x", ".cayu/x": b"x"}

    skipped, kept = _cloud_console_source.write_tree(
        tmp_path / "a", hostile, project_id="stp_1", force=False
    )

    assert kept == []
    assert skipped == [".cayu/x", ".git/config", "data/seed.json", "vendor/.GIT/hooks/x"]
    assert not (tmp_path / "a" / ".git").exists()
    with pytest.raises(_cloud_console_source.ConsoleError) as refused:
        _cloud_console_source.write_tree(
            tmp_path / "b", {"GIT~1/hooks/x": b"x"}, project_id="stp_1", force=False
        )
    assert refused.value.category == "console_directory_invalid"


@pytest.mark.parametrize(
    "path",
    [
        ".g\u200cit/config",
        "\u200d.git/config",
        ".git\ufeff/hooks/pre-commit",
        ".gi\u00adt/config",
        ".g\u2060it/config",
        ".git\ufe0f/config",
        "vendor/.G\u200bIT/config",
        "vendor/.c\u200bayu/console-pull.json",
        ".v\u200cenv/bin/activate",
    ],
)
def test_a_pull_never_writes_a_name_a_filesystem_reads_as_a_protected_one(
    tmp_path: Path, path: str
) -> None:
    root = tmp_path / "checkout"

    skipped, _ = _cloud_console_source.write_tree(
        root, {**PULLED, path: b"[core]\n"}, project_id="stp_1", force=False
    )

    assert not _cloud_console_source.pulled(path)
    assert path in skipped
    assert not (root / path).exists()


def test_a_pull_writes_names_with_ordinary_joiners_and_variation_selectors(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    emoji = "docs/\u2764\ufe0f.md"
    persian = "docs/\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645.md"
    family = "docs/\U0001f468\u200d\U0001f469\u200d\U0001f467.md"
    files = {**PULLED, emoji: b"love\n", persian: b"want\n", family: b"family\n"}
    archive = source_archive(files)
    cloud(_pull_routes(archive, source_digest(archive)))
    root = tmp_path / "checkout"

    code, output = _run(["pull", "stp_1", "--dir", str(root)], capsys)

    assert code == 0, output
    assert output["result"]["skipped"] == ["data/seed.json"]
    assert (root / emoji).read_bytes() == b"love\n"
    assert (root / persian).read_bytes() == b"want\n"
    assert (root / family).read_bytes() == b"family\n"
    assert console_source_problem(files) is None


def test_a_forced_pull_keeps_local_files_a_push_leaves_out_and_replaces_stale_files(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    root = tmp_path / "checkout"
    fake = _pulled(cloud, capsys, root)
    (root / ".env.local").write_text("TOKEN=mine\n")
    (root / "config").write_text("a file the next revision makes a directory\n")
    newer = {**PULLED, "config/base.toml": b"x = 1\n"}
    archive = source_archive(newer)
    _answer(fake, _pull_routes(archive, source_digest(archive)))

    code, output = _run(["pull", "stp_1", "--dir", str(root), "--force"], capsys)

    assert code == 0, output
    assert output["result"]["kept"] == [".env.local"]
    assert ".env.local" not in output["result"]["not_pushed"]
    assert (root / ".env.local").read_text() == "TOKEN=mine\n"
    assert (root / "config" / "base.toml").read_text() == "x = 1\n"


def test_pull_records_the_latest_revision_a_restore_gave_the_same_source(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    routes = _pull_routes()
    restored = _project((1, "sha256:one"), (2, DIGEST), (3, "sha256:three"), (4, DIGEST))
    routes[("GET", _PROJECT)] = httpx.Response(200, json=restored)
    cloud(routes)
    root = tmp_path / "checkout"

    code, output = _run(["pull", "stp_1", "--dir", str(root)], capsys)

    assert code == 0, output
    assert output["result"]["revision"] == 4
    assert json.loads((root / ".cayu" / "console-pull.json").read_text())["revision"] == 4


def test_deploy_refuses_an_application_that_is_not_a_slug(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({})

    code, output = _run(["deploy", "stp_1", "--application", "Customer health"], capsys)

    assert (code, output["error"]["category"]) == (2, "invalid_input")
    assert fake.requests == []


def test_an_empty_from_agent_is_refused_as_json(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud({("GET", _APPS): httpx.Response(200, json=_AGENTS)})

    code, output = _run(["create", "--from-agent", ""], capsys)

    assert code == 2
    assert output["ok"] is False


def _pulled(cloud: Any, capsys: pytest.CaptureFixture[str], root: Path) -> Any:
    """Pull the project into ``root``; the fake Cloud then answers what ``_answer`` sets."""

    fake = cloud(_pull_routes())
    code, output = _run(["pull", "stp_1", "--dir", str(root)], capsys)
    assert code == 0, output
    return fake


def _answer(fake: Any, routes: dict[Any, Any]) -> Any:
    fake.routes = routes
    fake.requests.clear()
    return fake


def _push_routes(*answers: httpx.Response, latest: str = DIGEST) -> dict[Any, Any]:
    def upload(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            201, json={"content_digest": body["content_digest"], "upload_url": _UPLOAD_URL}
        )

    pending = list(answers)
    routes: dict[Any, Any] = {
        ("GET", _PROJECT): httpx.Response(200, json=_project((3, latest))),
        ("POST", "/v1/source-bundles/uploads"): upload,
        ("PUT", "/bundle"): httpx.Response(200),
        ("POST", f"{_PROJECT}/revisions"): lambda _request: pending.pop(0),
        ("GET", f"{_PROJECT}/revision-jobs/stj_1"): lambda _request: pending.pop(0),
    }
    return routes


def test_push_records_the_directory_source_on_the_pulled_revision(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    root = tmp_path / "checkout"
    pulled = _pulled(cloud, capsys, root)
    (root / "app" / "main.py").write_text("print('changed')\n")
    for local in (".env", ".venv/pyvenv.cfg", "app/x.pyc", "node_modules/x/i.js", "server.log"):
        (root / local).parent.mkdir(parents=True, exist_ok=True)
        (root / local).write_text("x")
    expected = {
        path: content
        for path, content in PULLED.items()
        if path not in {"data/seed.json", ".env.local"}
    }
    expected["app/main.py"] = b"print('changed')\n"
    digest = source_digest(source_archive(expected))
    saved = {"number": 4, "content_digest": digest, "recorded": True, "status": "saved"}
    fake = _answer(pulled, _push_routes(httpx.Response(201, json=saved)))

    code, output = _run(["push", "stp_1", "--dir", str(root), "--summary", "Edit main."], capsys)

    assert code == 0, output
    assert output["result"] == {
        "directory": str(root.resolve()),
        "file_count": len(expected),
        "project_id": "stp_1",
        "pulled_revision": 3,
        "revision": 4,
        "source_digest": digest,
        "status": "saved",
    }
    sent = _sent(fake)
    assert sent[1][:3] == (
        "POST",
        "/v1/source-bundles/uploads",
        {"content_digest": digest, "size_bytes": len(source_archive(expected))},
    )
    assert sent[2][:2] == ("PUT", "/bundle")
    assert fake.requests[2].content == source_archive(expected)
    assert sent[3][:3] == (
        "POST",
        f"{_PROJECT}/revisions",
        {"expected_latest_digest": DIGEST, "source_digest": digest, "summary": "Edit main."},
    )
    record = json.loads((root / ".cayu" / "console-pull.json").read_text())
    assert record == {"project_id": "stp_1", "revision": 4, "source_digest": digest}


def test_push_waits_for_the_workspace_job_and_says_so(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkout"
    pulled = _pulled(cloud, capsys, root)
    (root / "app" / "main.py").write_text("print('changed')\n")
    files = _cloud_console_source.source_tree(root)
    digest = source_digest(source_archive(files))
    monkeypatch.setattr(_cloud_console, "_sleep", lambda _seconds: None)
    _answer(
        pulled,
        _push_routes(
            httpx.Response(202, json={"job_id": "stj_1", "status": "queued"}),
            httpx.Response(200, json={"job_id": "stj_1", "status": "running"}),
            httpx.Response(
                200,
                json={"job_id": "stj_1", "number": 4, "content_digest": digest, "recorded": True},
            ),
        ),
    )

    code = main(["cloud", "console", "push", "stp_1", "--dir", str(root), "--summary", "x"])
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert code == 0, output
    assert (output["result"]["revision"], output["result"]["job_id"]) == (4, "stj_1")
    assert "the job is running" in captured.err


def test_push_refusals_keep_the_pull_record_and_name_the_pulled_revision(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    root = tmp_path / "checkout"
    fake = _pulled(cloud, capsys, root)
    record = (root / ".cayu" / "console-pull.json").read_text()
    push = ["push", "stp_1", "--dir", str(root), "--summary", "x"]

    _answer(fake, _push_routes(latest="sha256:" + "5" * 64))
    code, output = _run(push, capsys)
    assert (code, output["error"]["category"]) == (2, "console_push_conflict")
    assert "changed since you pulled revision 3" in output["error"]["message"]

    conflict = httpx.Response(409, json={"detail": "The project changed since you pulled."})
    _answer(fake, _push_routes(conflict))
    code, output = _run(push, capsys)
    assert output["error"] == {
        "category": "console_push_conflict",
        "message": "The project changed since you pulled.",
        "pulled_revision": 3,
    }

    refused = {"job_id": "stj_1", "status": "refused", "error": "The preview didn't start."}
    _answer(fake, _push_routes(httpx.Response(202, json=refused)))
    code, output = _run(push, capsys)
    assert (code, output["error"]["category"], output["error"]["job_id"]) == (
        2,
        "console_push_refused",
        "stj_1",
    )
    assert (root / ".cayu" / "console-pull.json").read_text() == record

    code, output = _run(["push", "stp_2", "--dir", str(root), "--summary", "x"], capsys)
    assert output["error"]["category"] == "console_checkout_mismatch"
    code, output = _run(["push", "stp_1", "--dir", str(root / "app"), "--summary", "x"], capsys)
    assert output["error"]["category"] == "console_checkout_missing"
    (root / "link.py").symlink_to(root / "app" / "main.py")
    code, output = _run(push, capsys)
    assert output["error"]["category"] == "source_invalid"
    assert "link.py is a symlink" in output["error"]["message"]


def test_a_source_past_what_a_workspace_saves_is_refused() -> None:
    many = {f"{index}.txt": b"x" for index in range(4097)}
    assert "at most 4096" in (console_source_problem(many) or "")
    assert "16 MiB" in (console_source_problem({"big.bin": b"x" * (17 * 2**20)}) or "")
    assert ".git" in (console_source_problem({"a/.Git/config": b"x"}) or "")
    assert console_source_problem(PULLED) is None


def test_a_source_archive_is_deterministic_and_reads_back() -> None:
    assert source_archive(dict(reversed(PULLED.items()))) == ARCHIVE
    assert source_files(ARCHIVE) == PULLED


def test_events_follow_prints_each_new_step_once_until_the_turn_ends(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(_cloud_console, "_sleep", sleeps.append)
    running = {"id": "stp_1", "turn_active": True}
    finished = {"id": "stp_1", "turn_active": False}
    step = [{"sequence": number, "kind": "tool"} for number in range(1, 6)]
    projects = [running, running, finished]
    pages = {
        0: {"events": step[0:2], "next_after": 2},
        2: [{"events": step[2:3], "next_after": 3}],
        3: [{"events": [], "next_after": 3}, {"events": step[3:5], "next_after": 5}],
        5: {"events": [], "next_after": 5},
    }

    def events(request: httpx.Request) -> httpx.Response:
        after = int(request.url.params["after"])
        page = pages[after]
        return httpx.Response(200, json=page.pop(0) if isinstance(page, list) else page)

    fake = cloud(
        {
            ("GET", _PROJECT): lambda _request: httpx.Response(200, json=projects.pop(0)),
            **{
                ("GET", f"{_PROJECT}/events?after={after}&limit=2"): events
                for after in (0, 2, 3, 5)
            },
        }
    )

    code = main(
        ["cloud", "console", "events", "stp_1", "--follow", "--limit", "2", "--poll-seconds", "0.5"]
    )

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0
    assert lines == step
    assert sleeps == [0.5, 0.5]
    assert [request.url.raw_path.decode() for request in fake.requests][:3] == [
        _PROJECT,
        f"{_PROJECT}/events?after=0&limit=2",
        f"{_PROJECT}/events?after=2&limit=2",
    ]


def test_events_follow_reports_an_unknown_project(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("GET", _PROJECT): httpx.Response(404, json={"detail": "Not Found"})})

    code, output = _run(["events", "stp_1", "--follow"], capsys)

    assert (code, output["error"]["category"]) == (2, "console_project_not_found")


def test_poll_seconds_without_follow_is_refused(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud({})

    code, output = _run(["events", "stp_1", "--poll-seconds", "1"], capsys)

    assert (code, output["error"]["category"]) == (2, "invalid_input")
    assert fake.requests == []


def test_push_reports_a_job_still_pending_at_its_deadline(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkout"
    pulled = _pulled(cloud, capsys, root)
    (root / "app" / "main.py").write_text("print('changed')\n")
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(_cloud_console, "_monotonic", lambda: now[0])
    monkeypatch.setattr(_cloud_console, "_sleep", sleep)
    running = httpx.Response(200, json={"job_id": "stj_1", "status": "running"})
    _answer(
        pulled,
        _push_routes(
            httpx.Response(202, json={"job_id": "stj_1", "status": "queued"}), *[running] * 3
        ),
    )

    code, output = _run(
        ["push", "stp_1", "--dir", str(root), "--summary", "x", "--wait-seconds", "3"], capsys
    )

    assert (code, output["error"]["category"]) == (2, "console_push_pending")
    assert (output["error"]["job_id"], output["error"]["pulled_revision"]) == ("stj_1", 3)
    assert sleeps == [2.0, 1.0]


def test_a_forced_pull_keeps_a_local_file_gitignore_leaves_out(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    root = tmp_path / "checkout"
    fake = _pulled(cloud, capsys, root)
    (root / ".gitignore").write_text("*.log\nconfig.local.json\n")
    (root / "config.local.json").write_text('{"mine": true}\n')
    newer = {
        **PULLED,
        ".gitignore": b"*.log\nconfig.local.json\n",
        "config.local.json": b'{"theirs": true}\n',
    }
    archive = source_archive(newer)
    _answer(fake, _pull_routes(archive, source_digest(archive)))

    code, output = _run(["pull", "stp_1", "--dir", str(root), "--force"], capsys)

    assert code == 0, output
    assert output["result"]["kept"] == [".env.local", "config.local.json"]
    assert (root / "config.local.json").read_text() == '{"mine": true}\n'


_BARE = {"HEAD": b"ref: refs/heads/main\n", "config": b"[core]\n\tpager = touch x\n"}


@pytest.mark.parametrize(
    ("files", "where"),
    [
        ({**_BARE, "objects/info/packs": b"", "refs/heads/main": b""}, "root"),
        (
            {f"tools/x/{path}": content for path, content in _BARE.items()}
            | {"tools/x/objects/info/packs": b"", "tools/x/refs/heads/main": b""},
            "'tools/x'",
        ),
        ({"vendor/head": b"x", "vendor/Objects/a": b"", "vendor/REFS/b": b""}, "'vendor'"),
        (
            {
                "vendor/lib/HEAD": b"ref: refs/heads/main\n",
                "vendor/lib/CommonDir": b"../store\n",
                "vendor/store/config": b"[core]\n\tpager = touch x\n",
                "vendor/store/objects/info/packs": b"",
                "vendor/store/refs/heads/main": b"",
            },
            "'vendor/lib'",
        ),
        (
            {
                "HEAD": b"ref: refs/heads/main\n",
                "commondir": b"vendor\n",
                "vendor/config": b"[core]\n\tpager = touch x\n",
                "vendor/objects/info/packs": b"",
                "vendor/refs/heads/main": b"",
            },
            "root",
        ),
        ({"lib/HE​AD": b"x", "lib/objects/a": b"", "lib/refs/b": b""}, "lib"),
    ],
)
def test_a_pull_refuses_a_revision_that_makes_a_bare_repository(
    tmp_path: Path, files: dict[str, bytes], where: str
) -> None:
    root = tmp_path / "checkout"

    with pytest.raises(_cloud_console_source.ConsoleError) as refused:
        _cloud_console_source.write_tree(root, {**PULLED, **files}, project_id="stp_1", force=False)

    assert refused.value.category == "console_source_unsafe"
    assert where in str(refused.value)
    assert not root.exists()


def test_a_lone_head_file_or_one_beside_only_refs_is_pulled(tmp_path: Path) -> None:
    files = {**PULLED, "HEAD": b"notes\n", "docs/HEAD": b"x", "docs/refs/a.md": b"x"}

    skipped, _ = _cloud_console_source.write_tree(
        tmp_path / "checkout", files, project_id="stp_1", force=False
    )

    assert skipped == ["data/seed.json"]
    assert (tmp_path / "checkout" / "HEAD").read_text() == "notes\n"
    assert (
        _cloud_console_source.bare_repository(
            tmp_path, {"lib/HE​AD": b"", "lib/objects/a": b"", "lib/refs/b": b""}
        )
        == "lib"
    )


def test_a_pull_counts_what_the_directory_already_holds_toward_a_bare_repository(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    (root / "lib" / "objects").mkdir(parents=True)
    (root / "lib" / "Refs").mkdir()
    (root / "tools").mkdir()
    (root / "tools" / "COMMONDIR").write_text("../store\n")

    for planted in ({"lib/HEAD": b"x"}, {"tools/head": b"x"}):
        assert _cloud_console_source.bare_repository(root, planted) is not None
    assert _cloud_console_source.bare_repository(root, {"docs/HEAD": b"x"}) is None


def test_a_forced_pull_of_a_case_only_rename_keeps_the_file(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkout"
    fake = _pulled(cloud, capsys, root)
    (root / "Readme.md").write_text("old\n")
    real_lexists = os.path.lexists

    def lexists(path: object) -> bool:
        # As on a case-insensitive filesystem (macOS), where README.md finds Readme.md.
        target = Path(str(path))
        if target.parent.is_dir():
            return any(
                entry.name.casefold() == target.name.casefold() for entry in target.parent.iterdir()
            )
        return real_lexists(str(path))

    monkeypatch.setattr(_cloud_console_source.os.path, "lexists", lexists)
    newer = {**PULLED, "README.md": b"new\n"}
    archive = source_archive(newer)
    _answer(fake, _pull_routes(archive, source_digest(archive)))

    code, output = _run(["pull", "stp_1", "--dir", str(root), "--force"], capsys)

    assert code == 0, output
    assert "README.md" not in output["result"]["kept"]
    assert "README.md" not in output["result"]["skipped"]
    assert (root / "README.md").read_text() == "new\n"


def test_a_pull_refuses_a_bare_repository_split_across_spellings_of_one_directory(
    tmp_path: Path,
) -> None:
    # One directory on HFS+, three spellings in the revision.
    files = {
        "vendor/HEAD": b"ref: refs/heads/main\n",
        "vend\u200cor/objects/info/packs": b"",
        "VENDOR/refs/heads/main": b"",
    }
    root = tmp_path / "checkout"

    with pytest.raises(_cloud_console_source.ConsoleError) as refused:
        _cloud_console_source.write_tree(root, {**PULLED, **files}, project_id="stp_1", force=False)

    assert refused.value.category == "console_source_unsafe"
    assert not root.exists()
    # What is already on disk under another spelling counts too.
    (tmp_path / "Lib" / "objects").mkdir(parents=True)
    (tmp_path / "LIB" / "refs").mkdir(parents=True)
    assert _cloud_console_source.bare_repository(tmp_path, {"lib/HEAD": b"x"}) == "lib"
