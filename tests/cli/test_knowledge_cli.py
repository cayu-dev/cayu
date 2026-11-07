"""`cayu knowledge list/show`: one reader seam over a local store and an Agent's API."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from cayu.cli import main
from cayu.knowledge.records import KnowledgeEntry, KnowledgeStatus
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.storage import SQLiteKnowledgeStore
from cayu.storage import migrations as schema

SCOPE = KnowledgeAccessScope.privileged()


def _seed(path: Path) -> None:
    async def seed() -> None:
        store = SQLiteKnowledgeStore(path, access_scope=SCOPE)
        try:
            for entry in (
                KnowledgeEntry(
                    id="denial",
                    namespace="reviews",
                    kind="denial",
                    labels={"agent": "hunter"},
                    title="Denied rm -rf",
                    text="The reviewer denied deleting the workspace.",
                    status=KnowledgeStatus.ACTIVE,
                ),
                KnowledgeEntry(
                    id="fact",
                    namespace="reviews",
                    kind="fact",
                    labels={"agent": "hunter"},
                    text="The repository uses uv.",
                    status=KnowledgeStatus.ACTIVE,
                ),
                KnowledgeEntry(
                    id="other",
                    namespace="ops",
                    kind="denial",
                    labels={"agent": "scout"},
                    text="Another namespace.",
                    status=KnowledgeStatus.ACTIVE,
                ),
                KnowledgeEntry(
                    id="proposal",
                    namespace="reviews",
                    kind="denial",
                    text="Not kept yet.",
                    status=KnowledgeStatus.PENDING,
                ),
            ):
                await store.create_entry(entry, access_scope=SCOPE)
        finally:
            await store.close()

    asyncio.run(seed())


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "cayu.db"
    _seed(path)
    return path


def _run(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, Any]:
    code = main(["knowledge", *arguments])
    out = capsys.readouterr().out
    return code, json.loads(out) if out.startswith("{") else out


def _ids(listed: dict[str, Any]) -> list[str]:
    return sorted(entry["entry_id"] for entry in listed["entries"])


def test_list_filters_by_status_namespace_kind_and_label(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = ("--sqlite", str(database), "--json")

    code, listed = _run(capsys, "list", *target)
    assert code == 0
    assert listed["source"] == {"kind": "sqlite", "path": str(database)}
    assert listed["status"] == "active"
    assert _ids(listed) == ["denial", "fact", "other"]
    assert _run(capsys, "list", "--namespace", "reviews", *target)[1]["entries"][0][
        "namespace"
    ] == ("reviews")
    assert _ids(_run(capsys, "list", "--kind", "denial", *target)[1]) == ["denial", "other"]
    assert _ids(
        _run(capsys, "list", "--kind", "denial", "--label", "agent=hunter", *target)[1]
    ) == ["denial"]
    assert _ids(_run(capsys, "list", "--status", "pending", *target)[1]) == ["proposal"]

    code, page = _run(capsys, "list", "--limit", "1", *target)
    assert code == 0 and len(page["entries"]) == 1 and page["has_more"] is True
    code, rest = _run(capsys, "list", "--cursor", page["next_cursor"], *target)
    assert code == 0
    assert _ids(rest) == sorted({"denial", "fact", "other"} - set(_ids(page)))


def test_show_prints_text_for_any_status(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, shown = _run(capsys, "show", "proposal", "--sqlite", str(database), "--json")
    assert code == 0
    assert shown["entry"]["status"] == "pending"
    assert shown["entry"]["text"] == "Not kept yet."
    assert shown["entry"]["chunks"][0]["text"] == "Not kept yet."

    code, out = _run(capsys, "show", "denial", "--sqlite", str(database))
    assert code == 0
    assert "The reviewer denied deleting the workspace." in out
    assert "agent=hunter" in out

    code, out = _run(capsys, "show", "absent", "--sqlite", str(database), "--json")
    assert code == 1
    assert out["error"] == {
        "code": "KNOWLEDGE_INSPECTION_FAILED",
        "message": "Knowledge entry not found: absent",
    }


def test_table_output(database: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(capsys, "list", "--sqlite", str(database), "--kind", "denial")

    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == ["entry_id", "kind", "namespace", "labels", "updated_at", "preview"]
    assert sorted(line.split()[0] for line in lines[1:]) == ["denial", "other"]
    assert "Denied rm -rf" in out


def test_read_only_sqlite_knowledge_store_cannot_write(database: Path) -> None:
    with pytest.raises(ValueError, match="schema_mode=validate"):
        SQLiteKnowledgeStore(database, read_only=True)

    async def write() -> None:
        store = SQLiteKnowledgeStore(
            database, schema_mode=schema.SchemaMode.VALIDATE, access_scope=SCOPE, read_only=True
        )
        try:
            assert await store.get_entry("denial") is not None
            await store.create_entry(KnowledgeEntry(id="new", text="New."))
        finally:
            await store.close()

    with pytest.raises(Exception, match="readonly|read-only|read only"):
        asyncio.run(write())


def test_store_and_server_readers_return_the_same_entries(database: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from cayu.applications import CayuApp
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.cli._knowledge_readers import (
        ApiKnowledgeReader,
        KnowledgeListRequest,
        StoreKnowledgeReader,
    )
    from cayu.server import ServerConfig, create_server

    async def read() -> tuple[Any, ...]:
        local = StoreKnowledgeReader(
            SQLiteKnowledgeStore(
                database,
                schema_mode=schema.SchemaMode.VALIDATE,
                access_scope=SCOPE,
                read_only=True,
            ),
            access_scope=SCOPE,
            source={},
        )
        store = SQLiteKnowledgeStore(
            database, schema_mode=schema.SchemaMode.VALIDATE, access_scope=SCOPE
        )
        app = CayuApp(knowledge_store=store, knowledge_access_scope=SCOPE, enable_logging=False)
        try:
            with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
                served = ApiKnowledgeReader(CayuApi("http://testserver/", http=client), source={})
                request = KnowledgeListRequest(kinds=("denial",))
                return (
                    await local.list_entries(request),
                    await served.list_entries(request),
                    await local.load_entry("denial"),
                    await served.load_entry("denial"),
                    await served.load_entry("proposal"),
                    await served.load_entry("absent"),
                )
        finally:
            await local.close()
            await store.close()

    local_page, served_page, local_entry, served_entry, pending, absent = asyncio.run(read())

    def common(row: dict[str, Any], keys: Any) -> dict[str, Any]:
        return {key: row[key] for key in keys}

    keys = local_page.entries[0].keys()
    assert [common(row, keys) for row in served_page.entries] == local_page.entries
    assert common(served_entry, local_entry.keys()) == local_entry
    assert pending["status"] == "pending"
    assert absent is None


_PAGED_STATUSES = (KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING, KnowledgeStatus.ARCHIVED)


def _paged_entries() -> list[KnowledgeEntry]:
    """Five entries per status, sharing a timestamp so pages split on importance and id."""

    updated = datetime(2026, 11, 1, tzinfo=UTC)
    return [
        KnowledgeEntry(
            id=f"{status.value}-{index}",
            namespace="reviews",
            kind="fact",
            text=f"Entry {index}.",
            status=status,
            importance=0.5 if index == 3 else None,
            created_at=updated,
            updated_at=updated,
        )
        for status in _PAGED_STATUSES
        for index in range(5)
    ]


def _page_through(
    capsys: pytest.CaptureFixture[str], status: KnowledgeStatus, *target: str
) -> list[str]:
    ids: list[str] = []
    cursor: tuple[str, ...] = ()
    for _ in range(10):
        code, page = _run(
            capsys, "list", "--status", status.value, "--limit", "2", *cursor, *target, "--json"
        )
        assert code == 0, page
        ids.extend(entry["entry_id"] for entry in page["entries"])
        assert page["has_more"] is (page["next_cursor"] is not None)
        if page["next_cursor"] is None:
            return ids
        cursor = ("--cursor", page["next_cursor"])
    raise AssertionError("The listing didn't end.")


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_store_listing_pages_through_every_status(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if backend == "postgres":
        pytest.importorskip("psycopg_pool")
        from cayu.storage import PostgresKnowledgeStore

        dsn = request.getfixturevalue("postgres_dsn")
        store: Any = PostgresKnowledgeStore(
            dsn, schema_mode=schema.SchemaMode.CREATE, access_scope=SCOPE
        )
        target: tuple[str, ...] = ("--postgres", request.getfixturevalue("postgres_url"))
    else:
        path = tmp_path / "cayu.db"
        store = SQLiteKnowledgeStore(path, access_scope=SCOPE)
        target = ("--sqlite", str(path))

    async def seed() -> None:
        try:
            for entry in _paged_entries():
                await store.create_entry(entry, access_scope=SCOPE)
        finally:
            await store.close()

    asyncio.run(seed())
    for status in _PAGED_STATUSES:
        ids = _page_through(capsys, status, *target)
        assert len(ids) == len(set(ids))
        assert sorted(ids) == [f"{status.value}-{index}" for index in range(5)]


def test_server_listing_pages_through_kept_and_pending_knowledge() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from cayu.applications import CayuApp
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.cli._knowledge_readers import ApiKnowledgeReader, KnowledgeListRequest
    from cayu.server import ServerConfig, create_server
    from cayu.storage.memory import InMemoryKnowledgeStore

    store = InMemoryKnowledgeStore(_paged_entries(), access_scope=SCOPE)
    app = CayuApp(knowledge_store=store, knowledge_access_scope=SCOPE, enable_logging=False)

    async def page_through(reader: ApiKnowledgeReader, status: KnowledgeStatus) -> list[str]:
        ids: list[str] = []
        cursor = None
        for _ in range(10):
            page = await reader.list_entries(
                KnowledgeListRequest(status=status, limit=2, cursor=cursor)
            )
            assert page.truncated is False
            ids.extend(entry["entry_id"] for entry in page.entries)
            if page.next_cursor is None:
                return ids
            cursor = page.next_cursor
        raise AssertionError("The listing didn't end.")

    async def read() -> dict[KnowledgeStatus, list[str]]:
        with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
            reader = ApiKnowledgeReader(CayuApi("http://testserver/", http=client), source={})
            return {
                status: await page_through(reader, status)
                for status in (KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING)
            }

    for status, ids in asyncio.run(read()).items():
        assert len(ids) == len(set(ids))
        assert sorted(ids) == [f"{status.value}-{index}" for index in range(5)]


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, Any]]:
    """A fake Cayu Cloud and Agent; ``routes`` maps Agent API paths to (status, body)."""

    state: dict[str, Any] = {"routes": {}, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parts = urlsplit(self.path)
            state["requests"].append((parts.path, parse_qs(parts.query)))
            origin = f"http://127.0.0.1:{self.server.server_port}"
            fixed: dict[str, tuple[int, Any]] = {
                "/v1/applications": (200, {"items": [{"id": "bug-hunter", "name": "Bug Hunter"}]}),
                "/v1/applications/bug-hunter/service": (
                    200,
                    {"status": "running", "cayu_url": f"{origin}/cayu/"},
                ),
                "/v1/applications/bug-hunter/operator-credentials": (
                    200,
                    {
                        "username": "operator",
                        "password": "pw",
                        "env": {"username": "U", "password": "P"},
                    },
                ),
            }
            answer: Callable[[], tuple[int, Any]] | tuple[int, Any] | None = fixed.get(
                parts.path
            ) or state["routes"].get(parts.path)
            status, body = answer if answer is not None else (404, {"detail": "Not Found"})
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("CAYU_CLOUD_API_KEY", "customer-secret-material")
    monkeypatch.setenv("CAYU_CLOUD_API_URL", origin)
    try:
        yield state
    finally:
        server.shutdown()
        thread.join()


def _api_entry(entry_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "namespace": "reviews",
        "kind": "denial",
        "labels": {"agent": "hunter"},
        "status": "active",
        "updated_at": "2026-10-28T16:19:00+00:00",
        "text_preview": "Denied.",
        **extra,
    }


def test_application_sends_filters_and_still_filters_an_older_servers_rows(
    agent: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    # An older Cayu ignores the filters and returns every kept entry.
    agent["routes"]["/cayu/api/knowledge"] = (
        200,
        {
            "entries": [_api_entry("denial"), _api_entry("fact", kind="fact")],
            "next_cursor": None,
        },
    )

    code, listed = _run(capsys, "list", "--application", "bug-hunter", "--kind", "denial", "--json")

    assert code == 0
    assert _ids(listed) == ["denial"]
    path, query = agent["requests"][-1]
    assert path == "/cayu/api/knowledge" and query["kind"] == ["denial"]


def test_application_pending_listing_forwards_the_cursor(
    agent: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    agent["routes"]["/cayu/api/knowledge/pending"] = (
        200,
        {
            "entries": [_api_entry("proposal", status="pending")],
            "next_cursor": "sealed-next",
            "truncated": True,
        },
    )
    target = ("--application", "bug-hunter", "--status", "pending", "--json")

    # A sealed cursor is base64url text, so it can start with a dash.
    for cursor in ("sealed-first", "-Jx4_sealed"):
        code, listed = _run(capsys, "list", "--cursor", cursor, *target)

        assert code == 0
        assert listed["next_cursor"] == "sealed-next" and listed["has_more"] is True
        path, query = agent["requests"][-1]
        assert path == "/cayu/api/knowledge/pending" and query["cursor"] == [cursor]


def test_application_pending_listing_on_an_older_server_without_a_cursor(
    agent: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    # An older Cayu's pending list has no `next_cursor` and ignores a `cursor`.
    agent["routes"]["/cayu/api/knowledge/pending"] = (
        200,
        {"entries": [_api_entry("proposal", status="pending")], "truncated": True},
    )
    target = ("--application", "bug-hunter", "--status", "pending", "--json")

    code, listed = _run(capsys, "list", *target)
    assert code == 0
    assert listed["next_cursor"] is None and listed["has_more"] is True

    code, out = _run(capsys, "list", "--cursor", "from-a-newer-cayu", *target)
    assert code == 1
    assert out["error"]["category"] == "unsupported"
    assert "deploy it with a newer Cayu" in out["error"]["message"]


def test_application_show_falls_back_to_pending_on_an_older_server(
    agent: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    agent["routes"]["/cayu/api/knowledge/pending/proposal"] = (
        200,
        _api_entry("proposal", status="pending", text="Not kept yet.", chunks=[]),
    )
    agent["routes"]["/cayu/api/knowledge/pending/denial"] = (
        404,
        {"detail": "Knowledge entry not found."},
    )

    code, shown = _run(capsys, "show", "proposal", "--application", "bug-hunter", "--json")
    assert code == 0
    assert shown["entry"]["text"] == "Not kept yet."

    code, out = _run(capsys, "show", "denial", "--application", "bug-hunter", "--json")
    assert code == 1
    assert out["error"]["category"] == "unsupported"
    assert "shows only pending knowledge entries" in out["error"]["message"]


def test_application_lists_and_shows_archived_entries(
    agent: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    decision = {"decision": "archive", "decided_by": "operator", "reason": "Stale."}
    agent["routes"]["/cayu/api/knowledge/archived"] = (
        200,
        {"entries": [_api_entry("old", status="archived", **decision)], "next_cursor": None},
    )
    recorded = {"reviewer_identity": "operator", "reviewer_version": "1", "reason": "Stale."}
    agent["routes"]["/cayu/api/knowledge/archived/old"] = (
        200,
        _api_entry(
            "old",
            status="archived",
            text="No longer true.",
            metadata={"cayu:review_archive": recorded},
            chunks=[],
        ),
    )
    agent["routes"]["/cayu/api/knowledge/kept/old"] = (
        404,
        {"detail": "Knowledge entry not found."},
    )
    agent["routes"]["/cayu/api/knowledge/pending/old"] = (
        409,
        {"detail": "Knowledge entry 'old' is 'archived', not 'pending'."},
    )
    target = ("--application", "bug-hunter")

    code, listed = _run(capsys, "list", "--status", "archived", *target, "--json")
    assert code == 0
    assert _ids(listed) == ["old"]
    assert listed["entries"][0]["reason"] == "Stale."
    path, _ = agent["requests"][-1]
    assert path == "/cayu/api/knowledge/archived"

    # Neither kept nor pending, so `show` finds it archived.
    code, shown = _run(capsys, "show", "old", *target, "--json")
    assert code == 0
    assert shown["entry"]["text"] == "No longer true."
    assert [path for path, _ in agent["requests"][-3:]] == [
        "/cayu/api/knowledge/kept/old",
        "/cayu/api/knowledge/pending/old",
        "/cayu/api/knowledge/archived/old",
    ]
    assert main(["knowledge", "show", "old", *target]) == 0
    table = capsys.readouterr().out
    assert "archived_reason" in table and "Stale." in table


def test_application_on_an_older_cayu_says_it_lists_no_archived_entries(
    agent: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    # A Cayu older than the archived routes serves the kept list only.
    agent["routes"]["/cayu/api/knowledge"] = (200, {"entries": [], "next_cursor": None})
    for status in ("kept", "pending"):
        agent["routes"][f"/cayu/api/knowledge/{status}/old"] = (
            404,
            {"detail": "Knowledge entry not found."},
        )

    code, out = _run(
        capsys, "list", "--application", "bug-hunter", "--status", "archived", "--json"
    )
    assert code == 1
    assert out["error"]["category"] == "unsupported"
    assert "deploy it with a newer Cayu" in out["error"]["message"]

    code, out = _run(capsys, "show", "old", "--application", "bug-hunter", "--json")
    assert code == 1
    assert "not found" in out["error"]["message"]


def test_show_escapes_terminal_control_sequences(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cayu.db"

    async def seed() -> None:
        store = SQLiteKnowledgeStore(path, access_scope=SCOPE)
        try:
            await store.create_entry(
                KnowledgeEntry(
                    id="forged",
                    title="Kept \x1b[2J",
                    text="line one\nline two \x1b]52;c;Zm9v\x07",
                    status=KnowledgeStatus.ACTIVE,
                ),
                access_scope=SCOPE,
            )
        finally:
            await store.close()

    asyncio.run(seed())

    code, out = _run(capsys, "show", "forged", "--sqlite", str(path))

    assert code == 0
    assert "\x1b" not in out and "\x07" not in out
    assert "line one\nline two \\x1b]52;c;Zm9v\\x07" in out


def test_show_reads_out_of_scope_entries_as_absent_and_shows_archived_ones(
    tmp_path: Path,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from cayu.applications import CayuApp
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.cli._knowledge_readers import ApiKnowledgeReader
    from cayu.server import ServerConfig, create_server
    from cayu.storage.memory import InMemoryKnowledgeStore

    store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(
                id="other", namespace="ops", text="Elsewhere.", status=KnowledgeStatus.ACTIVE
            ),
            KnowledgeEntry(
                id="arch", namespace="reviews", text="Old.", status=KnowledgeStatus.ARCHIVED
            ),
        ],
        access_scope=SCOPE,
    )
    app = CayuApp(
        knowledge_store=store,
        knowledge_access_scope=SCOPE,
        knowledge_review_namespace="reviews",
        enable_logging=False,
    )

    async def read() -> tuple[Any, Any]:
        with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
            reader = ApiKnowledgeReader(CayuApi("http://testserver/", http=client), source={})
            return await reader.load_entry("other"), await reader.load_entry("arch")

    other, archived = asyncio.run(read())
    assert other is None
    assert (archived["entry_id"], archived["status"]) == ("arch", "archived")


def test_postgres_knowledge_store_can_be_opened_read_only_for_inspection() -> None:
    pytest.importorskip("psycopg_pool")
    from cayu.storage.knowledge_postgres import PostgresKnowledgeStore

    # Construction doesn't connect; it used to refuse read_only outright.
    store = PostgresKnowledgeStore("postgresql://inspector@localhost/cayu", read_only=True)
    assert store._read_only is True


def test_application_shows_an_entry_named_pending(
    agent: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    # `/knowledge/pending` is the pending list, so the kept entry route has its own prefix.
    agent["routes"]["/cayu/api/knowledge/kept/pending"] = (
        404,
        {"detail": "Knowledge entry not found."},
    )
    agent["routes"]["/cayu/api/knowledge/pending/pending"] = (
        200,
        _api_entry("pending", status="pending", text="Named like the list.", chunks=[]),
    )

    code, shown = _run(capsys, "show", "pending", "--application", "bug-hunter", "--json")

    assert code == 0
    assert shown["entry"]["text"] == "Named like the list."


def test_show_redacts_a_private_key_that_crosses_the_chunk_byte_bound(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cayu.db"
    key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        + "MIIEsecretkeymaterialXYZ" * 20
        + "\n-----END RSA PRIVATE KEY-----"
    )

    async def seed() -> None:
        store = SQLiteKnowledgeStore(path, access_scope=SCOPE)
        try:
            await store.create_entry(
                KnowledgeEntry(id="k", text="a" * 127_900 + key, status=KnowledgeStatus.ACTIVE),
                access_scope=SCOPE,
            )
        finally:
            await store.close()

    asyncio.run(seed())

    code, shown = _run(capsys, "show", "k", "--sqlite", str(path), "--json")

    assert code == 0
    chunks = shown["entry"]["chunks"]
    assert chunks
    assert all("BEGIN RSA PRIVATE KEY" not in chunk["text"] for chunk in chunks)
    assert all("secretkeymaterial" not in chunk["text"] for chunk in chunks)
    assert sum(len(chunk["text"].encode()) for chunk in chunks) <= 128_000


def _seed_entries(path: Path, *entries: KnowledgeEntry) -> None:
    async def seed() -> None:
        store = SQLiteKnowledgeStore(path, access_scope=SCOPE)
        try:
            for entry in entries:
                await store.create_entry(entry, access_scope=SCOPE)
        finally:
            await store.close()

    asyncio.run(seed())


def test_list_previews_each_entrys_text_beside_its_title(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cayu.db"
    invoice = (
        "Retry failed invoice exports three times with a five minute backoff, then page the "
        "billing on-call. Exports that fail on a closed accounting period must not be "
        "retried at all: they need the controller to reopen the period first."
    )
    _seed_entries(
        path,
        KnowledgeEntry(id="invoice", title="Retry policy", text=invoice, importance=0.9),
        KnowledgeEntry(
            id="payroll", title="Retry policy", text="Retry payroll exports once.", importance=0.5
        ),
    )

    code, listed = _run(capsys, "list", "--sqlite", str(path), "--json")

    assert code == 0
    rows = {row["entry_id"]: row for row in listed["entries"]}
    assert rows["invoice"]["title"] == rows["payroll"]["title"] == "Retry policy"
    assert rows["invoice"]["text_preview"].startswith("Retry failed invoice exports")
    assert rows["invoice"]["text_preview"].endswith("…")
    assert len(rows["invoice"]["text_preview"]) <= 200
    assert invoice.startswith(rows["invoice"]["text_preview"].removesuffix("…"))
    assert rows["payroll"]["text_preview"] == "Retry payroll exports once."

    # The table has one preview column: the title, or the text when there is none.
    code, out = _run(capsys, "list", "--sqlite", str(path))
    assert code == 0
    assert "Retry policy" in out
    assert "Retry failed invoice" not in out


def test_list_preview_redacts_a_private_key_before_cutting_the_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cayu.db"
    key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        + "MIIEsecretkeymaterialXYZ" * 20
        + "\n-----END RSA PRIVATE KEY-----"
    )
    # Cut first, the excerpt would keep the key's start without the END marker the
    # redactor needs to recognize it.
    _seed_entries(path, KnowledgeEntry(id="k", title="Deploy key", text="Rotate: " + key))

    code, listed = _run(capsys, "list", "--sqlite", str(path), "--json")
    assert code == 0
    [row] = listed["entries"]
    assert "BEGIN RSA PRIVATE KEY" not in row["text_preview"]
    assert "secretkeymaterial" not in row["text_preview"]

    code, shown = _run(capsys, "show", "k", "--sqlite", str(path), "--json")
    assert code == 0
    assert row["text_preview"] == shown["entry"]["text"]
