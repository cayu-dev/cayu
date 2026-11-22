"""`cayu knowledge list --group` and `cayu knowledge drop-duplicates` on a real server."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.cli import main
from cayu.knowledge.records import KnowledgeEntry, KnowledgeStatus
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.server import ServerConfig, create_server
from cayu.storage import SQLiteKnowledgeStore
from cayu.storage.memory import InMemoryKnowledgeStore

_TEXT = "Queued delivery retries keep the same delivery identity after acknowledgement loss."
_SCOPE = KnowledgeAccessScope.privileged()


def _pending(entry_id: str, text: str = _TEXT, **fields) -> KnowledgeEntry:
    return KnowledgeEntry(id=entry_id, text=text, status=KnowledgeStatus.PENDING, **fields)


def _serve(monkeypatch, store):
    from cayu.cli import _control_plane_actions, knowledge
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.cli._knowledge_readers import ApiKnowledgeReader

    app = CayuApp(knowledge_store=store, enable_logging=False)
    client = TestClient(create_server(app, config=ServerConfig.local_development()))
    monkeypatch.setattr(
        _control_plane_actions,
        "open_action_api",
        lambda args: CayuApi("http://testserver", http=client),
    )
    monkeypatch.setattr(
        knowledge,
        "open_knowledge_reader",
        lambda target: ApiKnowledgeReader(
            CayuApi("http://testserver", http=client), source={"kind": "server"}
        ),
    )
    return client


def _run(capsys, *args, code: int = 0) -> dict:
    assert main(["knowledge", *args, "--json"]) == code
    return json.loads(capsys.readouterr().out)


def test_keep_one_drop_the_rest_and_replay_after_a_failure(monkeypatch, capsys):
    store = InMemoryKnowledgeStore(
        [_pending("first"), _pending("second"), _pending("third")], access_scope=_SCOPE
    )
    with _serve(monkeypatch, store) as client:
        grouped = _run(
            capsys, "list", "--status", "pending", "--group", "--server-url", "http://localhost"
        )
        assert [[row["entry_id"] for row in c["members"]] for c in grouped["clusters"]] == [
            ["first", "second", "third"]
        ]
        command = (
            "drop-duplicates",
            "second",
            "third",
            "late",
            "--keep",
            "first",
            "--reason",
            "Same fact.",
            "--operation-id",
            "cluster-1",
        )
        # `late` isn't proposed yet: the command stops there, after deciding the others.
        failed = _run(capsys, *command, code=1)
        assert "3 decision(s) were made first" in failed["error"]["message"]
        assert [d["entry_id"] for d in failed["decisions"]] == ["first", "second", "third"]
        asyncio.run(store.create_entry(_pending("late"), access_scope=_SCOPE))

        done = _run(capsys, *command)
        decisions = {d["entry_id"]: d for d in done["decisions"]}
        assert decisions["first"]["decision"] == "kept"
        assert {d["decision"] for e, d in decisions.items() if e != "first"} == {"dropped"}
        # The decisions made before replay at the same revisions instead of conflicting.
        for earlier in failed["decisions"]:
            assert decisions[earlier["entry_id"]]["revision"] == earlier["revision"]
        assert decisions["second"]["reason"] == "Duplicate of first. Same fact."
        assert done["duplicate_of"] == "first"
        assert _run(capsys, *command) == done

        statuses = {
            entry_id: asyncio.run(store.get_entry(entry_id, access_scope=_SCOPE)).status
            for entry_id in ("first", "second", "third", "late")
        }
        assert statuses == {
            "first": KnowledgeStatus.ACTIVE,
            "second": KnowledgeStatus.ARCHIVED,
            "third": KnowledgeStatus.ARCHIVED,
            "late": KnowledgeStatus.ARCHIVED,
        }
        assert client.get("/api/knowledge/pending").json()["entries"] == []
        # Each dropped entry records its own reviewed decision and reason.
        archived = client.get("/api/knowledge/archived").json()["entries"]
        assert {row["entry_id"]: (row["decision"], row["reason"]) for row in archived} == {
            entry_id: ("drop", "Duplicate of first. Same fact.")
            for entry_id in ("second", "third", "late")
        }


def test_drop_duplicates_of_a_kept_entry(monkeypatch, capsys):
    store = InMemoryKnowledgeStore(
        [KnowledgeEntry(id="kept", text=_TEXT), _pending("copy"), _pending("other-copy")],
        access_scope=_SCOPE,
    )
    with _serve(monkeypatch, store):
        grouped = _run(
            capsys, "list", "--status", "pending", "--group", "--server-url", "http://localhost"
        )
        assert [row["entry_id"] for row in grouped["clusters"][0]["similar_kept"]] == ["kept"]
        # A target that isn't kept is refused before anything is decided.
        refused = _run(
            capsys, "drop-duplicates", "copy", "--of", "other-copy", "--operation-id", "x", code=1
        )
        assert refused["decisions"] == []
        done = _run(
            capsys, "drop-duplicates", "copy", "other-copy", "--of", "kept", "--operation-id", "y"
        )
        assert [(d["entry_id"], d["decision"]) for d in done["decisions"]] == [
            ("copy", "dropped"),
            ("other-copy", "dropped"),
        ]
        assert done["decisions"][0]["reason"] == "Duplicate of kept."
        kept = asyncio.run(store.get_entry("kept", access_scope=_SCOPE))
        assert kept.status is KnowledgeStatus.ACTIVE and kept.revision == 1


@pytest.mark.parametrize(
    "arguments",
    [
        ("a", "--operation-id", "x"),
        ("a", "--keep", "b", "--of", "c", "--operation-id", "x"),
        ("a", "--keep", "b"),
    ],
)
def test_drop_duplicates_requires_one_target_and_an_operation_id(arguments):
    with pytest.raises(SystemExit):
        main(["knowledge", "drop-duplicates", *arguments, "--json"])


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("a", "a", "--keep", "b"), "listed twice"),
        (("a", "--keep", "a"), "duplicate of itself"),
        (("a", "--of", "b ", "--reason", " "), "--of"),
    ],
)
def test_drop_duplicates_refuses_unclear_requests(arguments, message, capsys):
    body = _run(capsys, "drop-duplicates", *arguments, "--operation-id", "x", code=1)
    assert message in body["error"]["message"] and body["decisions"] == []


def test_group_lists_a_local_store(tmp_path, capsys):
    path = tmp_path / "knowledge.sqlite"

    async def seed():
        store = SQLiteKnowledgeStore(path, access_scope=_SCOPE)
        for entry in (
            _pending("a"),
            _pending("b", metadata={"cayu:corroboration": {"source_count": 4}}),
            _pending("c", "Branch snapshots stay immutable."),
        ):
            await store.create_entry(entry, access_scope=_SCOPE)
        await store.close()

    asyncio.run(seed())
    body = _run(capsys, "list", "--status", "pending", "--group", "--sqlite", str(path))
    assert [[row["entry_id"] for row in c["members"]] for c in body["clusters"]] == [
        ["b", "a"],
        ["c"],
    ]
    assert [c["sources"] for c in body["clusters"]] == [5, 1]
    assert main(["knowledge", "list", "--group", "--sqlite", str(path), "--json"]) == 1
    assert main(["knowledge", "list", "--status", "pending", "--group", "--sqlite", str(path)]) == 0
    table = capsys.readouterr().out
    assert "group" in table and "sources" in table
