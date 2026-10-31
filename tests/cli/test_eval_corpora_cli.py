"""`cayu eval corpora list|import|download` move portable corpora through the server API the
dashboard's Settings → Import / export calls, against a running server (`--server-url`) and a
Cloud Agent (`--application`) reached through a fake Cayu Cloud.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from tests.cli.test_cloud_evals import _AGENT, _PASSWORD, _agent, _corpus_file
from tests.cli.test_eval_curation_cli import _LOCAL, _command, _ok, _refused
from tests.cli.test_session_cli_cloud import _FakeCloud
from tests.evals.test_corpus_execution import _corpus

from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_agent_api import basic_authorization
from cayu.cli._eval_curation import _rereadable

_BASIC = basic_authorization("operator", _PASSWORD)


def test_a_corpus_is_imported_listed_and_downloaded_as_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(trials=1)
    path = _corpus_file(tmp_path, corpus)
    downloaded = tmp_path / "downloaded.eval.json"
    on_cloud = ("--application", _AGENT)
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", _BASIC)

    with _agent(tmp_path, monkeypatch) as client:
        assert _ok(capsys, "corpora", "list", *_LOCAL)["items"] == []

        imported = _ok(capsys, "corpora", "import", str(path), *_LOCAL)
        assert imported["entry"]["revision"] == corpus.revision
        [suite] = imported["suites"]
        assert suite["suite_id"] == corpus.suites[0].id
        assert [item["id"] for item in _ok(capsys, "suites", "list", *_LOCAL)["items"]] == [
            suite["id"]
        ]
        assert _ok(capsys, *_command(suite["show"]))["suite"]["id"] == suite["id"]
        # A revision is its content: importing it again, here on the Agent, changes nothing.
        again = _ok(capsys, "corpora", "import", str(path), *on_cloud)
        assert (again["application_id"], again["entry"]) == (_AGENT, imported["entry"])
        listed = _ok(capsys, "corpora", "list", *on_cloud)
        assert [item["revision"] for item in listed["items"]] == [corpus.revision]

        # Download writes the revision as the server's Download serves it.
        written = _ok(
            capsys,
            *("corpora", "download", corpus.revision, "--json-output", str(downloaded)),
            *_LOCAL,
        )
        assert (written["file"], written["suites"]) == (str(downloaded), [corpus.suites[0].id])
        served = client.get(
            f"/api/evals/corpora/{corpus.revision}/download", headers={"Authorization": _BASIC}
        )
        assert downloaded.read_text(encoding="utf-8") == served.text
        assert main(["eval", "validate", str(downloaded), "--format", "json"]) == 0
        assert json.loads(capsys.readouterr().out)["revision"] == corpus.revision

        error = _refused(
            capsys,
            "corpora",
            "download",
            "sha256:" + "0" * 64,
            "--json-output",
            str(tmp_path / "missing.json"),
            *_LOCAL,
        )
        assert error["category"] == "not_found"
        assert main(["eval", "corpora", "list", *_LOCAL, "--table"]) == 0
        assert main(["eval", "corpora", "import", str(path), *_LOCAL, "--table"]) == 0
        text = capsys.readouterr().out
        assert corpus.revision in text
        assert f"  {corpus.suites[0].id}: cayu eval suites show {suite['id']}" in text


def test_corpus_files_are_checked_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "corpus.json"
    path.write_text("{}", encoding="utf-8")
    error = _refused(capsys, "corpora", "import", str(path))
    assert error["category"] == "invalid_input"
    assert error["message"].startswith(f"{path} is not a valid eval corpus:")
    error = _refused(capsys, "corpora", "import", str(tmp_path / "missing.json"))
    assert (
        error["message"]
        == f"Could not read {tmp_path / 'missing.json'}: No such file or directory."
    )
    # The answer and the corpus can't be one file, also through a link to it.
    out, link = tmp_path / "out.json", tmp_path / "link.json"
    out.write_text("kept", encoding="utf-8")
    link.hardlink_to(out)
    revision = "sha256:" + "a" * 64
    for written in (out, link):
        error = _refused(
            capsys, "corpora", "download", revision, "--json-output", str(written), "-o", str(out)
        )
        assert (error["category"], error["message"]) == (
            "invalid_input",
            "--json-output and --output name the same file.",
        )
    assert out.read_text(encoding="utf-8") == "kept"
    # Nor can the corpus be a file the command reads, such as its Cloud context.
    context = tmp_path / "context.json"
    context.write_text("{}", encoding="utf-8")
    for written in (context, link):
        link.unlink()
        link.hardlink_to(context)
        error = _refused(
            *(capsys, "corpora", "download", revision, "--json-output", str(written)),
            *("--application", "my-agent", "--context", str(context)),
        )
        assert (error["category"], error["message"]) == (
            "invalid_input",
            "--json-output and --context name the same file.",
        )
    # Also the context the command signs in with when CAYU_CLOUD_CONTEXT selects it.
    monkeypatch.setenv("CAYU_CLOUD_CONTEXT", str(context))
    error = _refused(
        *(capsys, "corpora", "download", revision, "--json-output", str(context)),
        *("--application", "my-agent"),
    )
    assert (error["category"], error["message"]) == (
        "invalid_input",
        "--json-output and the Cloud context name the same file.",
    )
    # And the API key file a context names, beside it. (The key is a placeholder.)
    signed_in = tmp_path / "signed-in.json"
    signed_in.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ready",
                "api_url": "https://cloud.invalid",
                "api_key_file": "cloud.key",
            }
        ),
        encoding="utf-8",
    )
    key = tmp_path / "cloud.key"
    key.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("CAYU_CLOUD_CONTEXT", str(signed_in))
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    error = _refused(
        *(capsys, "corpora", "download", revision, "--json-output", str(key)),
        *("--application", "my-agent"),
    )
    assert (error["category"], error["message"]) == (
        "invalid_input",
        "--json-output and the Cloud API key file name the same file.",
    )
    assert key.read_text(encoding="utf-8") == "placeholder"
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT")
    assert context.read_text(encoding="utf-8") == "{}"
    # A file the check can't look at, such as one in a directory the user can't search, is no
    # collision; the command answers as usual.
    answer = tmp_path / "answer.json"
    answer.write_text("", encoding="utf-8")

    def unreadable(self: Path, other: Path) -> bool:
        raise PermissionError(13, "Permission denied", str(self))

    with monkeypatch.context() as patched:
        patched.setattr(Path, "samefile", unreadable)
        code = main(
            [
                *("eval", "corpora", "download", revision, "--json-output", str(out)),
                *("--output", str(answer), "--server-url", "http://127.0.0.1:1"),
            ]
        )
    assert code == 2
    assert json.loads(answer.read_text(encoding="utf-8"))["error"]["category"] == "unreachable"


def test_the_output_check_reads_no_sign_in_file_signing_in_reads_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    key = tmp_path / "cloud.key"
    key.write_text("placeholder", encoding="utf-8")
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ready",
                "api_url": "https://cloud.invalid",
                "api_key_file": str(key),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *_args, **_kwargs: _FakeCloud())
    reads: list[Path] = []
    read_context = cloud_cli._read_context

    def counted(path: Path) -> dict:
        reads.append(path)
        return read_context(path)

    monkeypatch.setattr(cloud_cli, "_read_context", counted)
    # A command that writes no file leaves the context to signing in, which reads it once,
    # so a context piped through /dev/stdin still reaches it.
    _refused(capsys, "corpora", "list", "--application", "unknown", "--context", str(context))
    assert reads == [context]

    # A pipe, or a descriptor such as /dev/stdin, is never read to find its key file.
    fifo = tmp_path / "context.fifo"
    os.mkfifo(fifo)
    assert (_rereadable(fifo), _rereadable(Path("/dev/stdin")), _rereadable(context)) == (
        False,
        False,
        True,
    )
