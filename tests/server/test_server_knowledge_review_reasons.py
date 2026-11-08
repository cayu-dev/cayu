"""Reasons and retry keys on keep, drop and kept-entry retirement over HTTP."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.knowledge.search import KnowledgeQuery
from cayu.server import ServerConfig, create_server
from cayu.storage.memory import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeStatus,
)

_SCOPE = KnowledgeAccessScope.privileged()


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    from cayu.storage import SQLiteKnowledgeStore

    opened = (
        InMemoryKnowledgeStore(access_scope=_SCOPE)
        if request.param == "memory"
        else SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_SCOPE)
    )
    yield opened
    close = getattr(opened, "close", None)
    if close is not None:
        asyncio.run(close())


def _propose(store, entry_id: str, **fields) -> None:
    asyncio.run(
        store.create_entry(
            KnowledgeEntry(
                id=entry_id,
                text=f"Made-up knowledge {entry_id}",
                status=KnowledgeStatus.PENDING,
                **fields,
            ),
            access_scope=_SCOPE,
        )
    )


def _client(store, config=None, **app_options) -> TestClient:
    app = CayuApp(
        knowledge_store=store, knowledge_access_scope=_SCOPE, enable_logging=False, **app_options
    )
    return TestClient(create_server(app, config=config or ServerConfig.local_development()))


def _current(store, entry_id: str) -> KnowledgeEntry:
    return asyncio.run(store.get_entry(entry_id, access_scope=_SCOPE))


def test_keep_records_its_reason_and_replays_under_its_key(store):
    _propose(store, "one")
    with _client(store) as client:
        kept = client.post(
            "/api/knowledge/one/approve",
            headers={"Idempotency-Key": "keep-one"},
            json={"reason": "Checked against the runbook."},
        )
        assert kept.status_code == 200, kept.text
        assert kept.json()["status"] == "active"
        assert kept.json()["reason"] == "Checked against the runbook."
        replayed = client.post(
            "/api/knowledge/one/approve",
            headers={"Idempotency-Key": "keep-one"},
            json={"reason": "Checked against the runbook."},
        )
        assert replayed.status_code == 200
        assert replayed.json() == kept.json()
        receipt = asyncio.run(store.load_activation_receipt("keep-one", access_scope=_SCOPE))
        assert receipt.authority.decision.annotations["reason"] == "Checked against the runbook."
        # A keep without a body still works, and records no reason.
        _propose(store, "two")
        plain = client.post("/api/knowledge/two/approve", headers={"Idempotency-Key": "keep-two"})
        assert plain.status_code == 200
        assert plain.json()["reason"] is None
        assert client.post("/api/knowledge/two/approve", json={"reason": "  "}).status_code == 422


def test_drop_with_a_reason_or_key_records_it_and_replays(store):
    for entry_id in ("dup", "plain", "keyed"):
        _propose(store, entry_id)
    with _client(store) as client:
        dropped = client.post(
            "/api/knowledge/dup/reject",
            headers={"Idempotency-Key": "drop-dup"},
            json={"reason": "Duplicates another entry."},
        )
        assert dropped.status_code == 200, dropped.text
        assert dropped.json()["status"] == "archived"
        assert dropped.json()["reason"] == "Duplicates another entry."
        entry = _current(store, "dup")
        assert entry.status is KnowledgeStatus.ARCHIVED
        decision = entry.metadata["cayu:review_rejection"]
        assert decision["reason"] == "Duplicates another entry."
        assert decision["reviewer_identity"] == "cayu:trusted-local-development"
        # A retry under the same key replays the drop instead of reporting a conflict.
        again = client.post(
            "/api/knowledge/dup/reject",
            headers={"Idempotency-Key": "drop-dup"},
            json={"reason": "Duplicates another entry."},
        )
        assert again.status_code == 200, again.text
        assert again.json() == dropped.json()
        # Another reason under that key is a different decision.
        changed = client.post(
            "/api/knowledge/dup/reject",
            headers={"Idempotency-Key": "drop-dup"},
            json={"reason": "Something else."},
        )
        assert changed.status_code == 409
        # Without a reason or key, drop archives as it always has.
        plain = client.post("/api/knowledge/plain/reject")
        assert plain.status_code == 200
        assert plain.json()["reason"] is None
        assert "cayu:review_rejection" not in _current(store, "plain").metadata
        keyed = client.post("/api/knowledge/keyed/reject", headers={"Idempotency-Key": "k"})
        assert keyed.status_code == 200
        assert keyed.json()["reason"] is None
        assert _current(store, "keyed").metadata["cayu:review_rejection"]["reason"] is None
        # A kept entry isn't a proposal to drop.
        _propose(store, "kept")
        assert (
            client.post(
                "/api/knowledge/kept/approve", headers={"Idempotency-Key": "keep-kept"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/knowledge/kept/reject",
                headers={"Idempotency-Key": "drop-kept"},
                json={"reason": "Too late."},
            ).status_code
            == 409
        )


def test_retire_with_a_reason_archives_out_of_search_and_replays(store):
    _propose(store, "stale")
    with _client(store) as client:
        assert (
            client.post(
                "/api/knowledge/stale/approve", headers={"Idempotency-Key": "keep-stale"}
            ).status_code
            == 200
        )
        assert _current(store, "stale").revision == 2
        # Without a revision there must be a reason or key, so the retirement is recorded.
        assert client.post("/api/knowledge/stale/retire", json={}).status_code == 422
        stale = client.post(
            "/api/knowledge/stale/retire",
            json={"reason": "Outdated.", "expected_revision": 1},
        )
        assert stale.status_code == 409
        retired = client.post(
            "/api/knowledge/stale/retire",
            headers={"Idempotency-Key": "archive-stale"},
            json={"reason": "The worker now runs three processes."},
        )
        assert retired.status_code == 200, retired.text
        assert retired.json() == {
            "entry_id": "stale",
            "revision": 3,
            "status": "archived",
            "reason": "The worker now runs three processes.",
        }
        entry = _current(store, "stale")
        assert entry.status is KnowledgeStatus.ARCHIVED
        assert entry.metadata["cayu:review_archive"]["reason"] == (
            "The worker now runs three processes."
        )
        found = asyncio.run(
            store.search(KnowledgeQuery(any_terms=["Made-up"]), access_scope=_SCOPE)
        )
        assert found.hits == []
        assert client.get("/api/knowledge").json()["entries"] == []
        replayed = client.post(
            "/api/knowledge/stale/retire",
            headers={"Idempotency-Key": "archive-stale"},
            json={"reason": "The worker now runs three processes.", "expected_revision": 2},
        )
        assert replayed.status_code == 200, replayed.text
        assert replayed.json() == retired.json()
        assert (
            client.post(
                "/api/knowledge/stale/retire",
                headers={"Idempotency-Key": "archive-stale"},
                json={"reason": "Another reason."},
            ).status_code
            == 409
        )
        # A pending proposal is dropped, not retired.
        _propose(store, "proposal")
        assert (
            client.post("/api/knowledge/proposal/retire", json={"reason": "Not kept."}).status_code
            == 409
        )


def test_retire_with_a_reason_stays_full_access_and_inside_the_review_scope(store):
    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    key = ec.generate_private_key(ec.SECP256R1())
    _propose(store, "mine", labels={"team": "mine"})
    _propose(store, "theirs", labels={"team": "other"})
    with _client(
        store,
        config=ServerConfig.protected(_auth(FakeJwks(key))),
        knowledge_review_labels={"team": "mine"},
    ) as client:
        member, admin = _as(_token(key, role="member")), _as(_token(key))
        # Members review proposals, with reasons, but can't retire kept entries.
        kept = client.post(
            "/api/knowledge/mine/approve",
            headers={**member, "Idempotency-Key": "keep-mine"},
            json={"reason": "Correct."},
        )
        assert kept.status_code == 200, kept.text
        refused = client.post(
            "/api/knowledge/mine/retire",
            headers={**member, "Idempotency-Key": "archive-mine"},
            json={"reason": "Stale."},
        )
        assert refused.status_code == 403
        assert _current(store, "mine").status is KnowledgeStatus.ACTIVE
        # Outside the review scope the entry can't be dropped or retired.
        assert (
            client.post(
                "/api/knowledge/theirs/reject",
                headers={**admin, "Idempotency-Key": "drop-theirs"},
                json={"reason": "Not ours."},
            ).status_code
            == 403
        )
        asyncio.run(
            store.transition_entry_status(
                "theirs",
                expected_revision=1,
                access_scope=_SCOPE,
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ACTIVE,
            )
        )
        assert client.post(
            "/api/knowledge/theirs/retire", headers=admin, json={"reason": "Not ours."}
        ).status_code in {403, 404}
        assert _current(store, "theirs").status is KnowledgeStatus.ACTIVE
        retired = client.post(
            "/api/knowledge/mine/retire",
            headers={**admin, "Idempotency-Key": "archive-mine"},
            json={"reason": "Stale."},
        )
        assert retired.status_code == 200, retired.text
        recorded = _current(store, "mine").metadata["cayu:review_archive"]
        assert recorded["reason"] == "Stale."
        assert recorded["annotations"]["identity_source"] == "http_auth"


def test_reasons_are_redacted_before_they_are_recorded(store):
    from cayu.vaults.redaction import SecretRedactor

    secret = "review-reason-secret-canary"
    _propose(store, "one")
    with _client(store, secret_redactor=SecretRedactor(secret)) as client:
        assert (
            client.post(
                "/api/knowledge/one/approve", headers={"Idempotency-Key": "keep-one"}
            ).status_code
            == 200
        )
        retired = client.post(
            "/api/knowledge/one/retire",
            headers={"Idempotency-Key": "archive-one"},
            json={"reason": f"Leaked {secret} here."},
        )
        assert retired.status_code == 200, retired.text
        assert secret not in retired.text
        assert secret not in str(_current(store, "one").metadata)


def test_inbox_shows_a_proposal_dropped_with_a_reason_as_dropped(store):
    _propose(store, "dup")
    with _client(store) as client:
        listed = client.get("/api/inbox")
        assert listed.status_code == 200, listed.text
        (item,) = [row for row in listed.json()["items"] if row["kind"] == "knowledge"]
        dropped = client.post(
            "/api/knowledge/dup/reject",
            headers={"Idempotency-Key": "drop-dup"},
            json={"reason": "Duplicates another entry."},
        )
        assert dropped.status_code == 200, dropped.text
        state = client.get(f"/api/inbox/{item['id']}")
        assert state.status_code == 200, state.text
        assert state.json()["status"] == "settled"
        settlement = state.json()["settlement"]
        assert settlement["decision"] == "drop"
        assert settlement["reason"] == "Duplicates another entry."
        assert client.get("/api/inbox").json()["items"] == []
        successor = _current(store, "dup")
        proposal = asyncio.run(store.get_entry("dup", revision=1, access_scope=_SCOPE))

    # The same archived revision without the store's receipt for the operation it names
    # isn't a drop.
    forged_store = InMemoryKnowledgeStore([proposal], access_scope=_SCOPE)
    with _client(forged_store) as client:
        asyncio.run(
            forged_store.append_entry_revision(successor, expected_revision=1, access_scope=_SCOPE)
        )
        state = client.get(f"/api/inbox/{item['id']}")
        assert state.status_code == 200, state.text
        assert state.json()["status"] == "superseded"
        assert state.json()["settlement"] is None


def test_retire_past_the_archive_bound_says_so(store, monkeypatch):
    import dataclasses

    from cayu.storage import KnowledgeChunk, _knowledge_rejection

    monkeypatch.setattr(
        _knowledge_rejection,
        "_ARCHIVE",
        dataclasses.replace(_knowledge_rejection._ARCHIVE, max_bytes=100),
    )
    asyncio.run(
        store.create_entry(
            KnowledgeEntry(id="big", text="Made-up big entry"),
            [KnowledgeChunk(id="big-0", entry_id="big", chunk_index=0, text="y" * 200)],
            access_scope=_SCOPE,
        )
    )
    with _client(store) as client:
        refused = client.post("/api/knowledge/big/retire", json={"reason": "Too big."})
        assert refused.status_code == 413
        assert "expected_revision alone" in refused.json()["detail"]
        assert _current(store, "big").status is KnowledgeStatus.ACTIVE
        plain = client.post("/api/knowledge/big/retire", json={"expected_revision": 1})
        assert plain.status_code == 200


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_operator_records_decisions_on_a_recall_scope_as_on_cayu_cloud(store_kind, tmp_path):
    """The scaffold's knowledge scope admits active and pending entries only. On Cayu Cloud
    the CLI sends the operator login, which proxy identity's fallback signs in as an admin;
    archiving and dropping with a reason must work there, as they do by revision alone."""

    import base64

    import httpx

    from cayu.server import BasicAuth, ProxyIdentityAuth
    from cayu.storage import SQLiteKnowledgeStore

    recall = KnowledgeAccessScope(
        allowed_namespaces=["default"],
        allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    )
    bound = (
        InMemoryKnowledgeStore(access_scope=recall)
        if store_kind == "memory"
        else SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=recall)
    )
    for entry_id, status in (
        ("kept", KnowledgeStatus.ACTIVE),
        ("proposal", KnowledgeStatus.PENDING),
        ("inbox-proposal", KnowledgeStatus.PENDING),
    ):
        asyncio.run(
            bound.create_entry(
                KnowledgeEntry(
                    id=entry_id, namespace="default", text=f"Made-up {entry_id}", status=status
                )
            )
        )
    auth = ProxyIdentityAuth(
        issuer="https://sign-in.example.com",
        audience="agent:app-1",
        jwks_url="https://sign-in.example.com/jwks",
        # No request here carries an identity, so the keys are never fetched.
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(404))),
        fallback=BasicAuth(username="operator", password="made-up-password"),
    )
    operator = {"Authorization": "Basic " + base64.b64encode(b"operator:made-up-password").decode()}
    app = CayuApp(knowledge_store=bound, knowledge_access_scope=recall, enable_logging=False)
    try:
        with TestClient(create_server(app, config=ServerConfig.protected(auth))) as client:
            archived = client.post(
                "/api/knowledge/kept/retire",
                headers={**operator, "Idempotency-Key": "archive-kept"},
                json={"reason": "Stale."},
            )
            assert archived.status_code == 200, archived.text
            assert archived.json() == {
                "entry_id": "kept",
                "revision": 2,
                "status": "archived",
                "reason": "Stale.",
            }
            assert client.get("/api/knowledge", headers=operator).json()["entries"] == []
            dropped = client.post(
                "/api/knowledge/proposal/reject",
                headers={**operator, "Idempotency-Key": "drop-proposal"},
                json={"reason": "Duplicate."},
            )
            assert dropped.status_code == 200, dropped.text
            assert dropped.json()["status"] == "archived"
            assert dropped.json()["reason"] == "Duplicate."
            (item,) = client.get("/api/inbox", headers=operator).json()["items"]
            settled = client.post(
                f"/api/inbox/{item['id']}/settle",
                headers=operator,
                json={"decision": "drop", "reason": "Duplicate too."},
            )
            assert settled.status_code == 200, settled.text
            assert settled.json()["replayed"] is False
            answered = settled.json()["item"]
            assert answered["id"] == item["id"]
            assert answered["status"] == "settled"
            assert answered["settlement"]["decision"] == "drop"
            assert answered["settlement"]["reason"] == "Duplicate too."
            assert client.get("/api/inbox", headers=operator).json()["items"] == []
        for entry_id in ("kept", "proposal", "inbox-proposal"):
            # Out of the recall scope, which still reads no archived entry.
            assert asyncio.run(bound.get_entry(entry_id)) is None
    finally:
        close = getattr(bound, "close", None)
        if close is not None:
            asyncio.run(close())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_operator_reads_back_what_a_recall_scope_archived_as_on_cayu_cloud(store_kind, tmp_path):
    """On Cayu Cloud the CLI sends the operator login, which proxy identity's fallback signs
    in as an admin, to an Agent whose store is bound to the scaffold's recall scope (active
    and pending only). The operator reads back what it archived or dropped, with the
    reason, and a retry replays; search still finds none of it, and a member sees none."""

    import base64

    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    from cayu.server import BasicAuth
    from cayu.storage import KnowledgeChunk, SQLiteKnowledgeStore

    recall = KnowledgeAccessScope(
        allowed_namespaces=["default"],
        allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    )
    bound = (
        InMemoryKnowledgeStore(access_scope=recall)
        if store_kind == "memory"
        else SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=recall)
    )
    for entry_id, status in (
        ("kept", KnowledgeStatus.ACTIVE),
        ("proposal", KnowledgeStatus.PENDING),
        ("plain", KnowledgeStatus.ACTIVE),
    ):
        text = f"Made-up retiredquasar {entry_id}"
        asyncio.run(
            bound.create_entry(
                KnowledgeEntry(id=entry_id, namespace="default", text=text, status=status),
                [KnowledgeChunk(id=f"{entry_id}:0", entry_id=entry_id, chunk_index=0, text=text)],
            )
        )
    key = ec.generate_private_key(ec.SECP256R1())
    auth = _auth(
        FakeJwks(key), fallback=BasicAuth(username="operator", password="made-up-password")
    )
    operator = {"Authorization": "Basic " + base64.b64encode(b"operator:made-up-password").decode()}
    member = _as(_token(key, role="member"))
    app = CayuApp(knowledge_store=bound, knowledge_access_scope=recall, enable_logging=False)
    try:
        with TestClient(create_server(app, config=ServerConfig.protected(auth))) as client:
            archive = {"headers": {**operator, "Idempotency-Key": "archive-kept"}}
            archived = client.post(
                "/api/knowledge/kept/retire", **archive, json={"reason": "Stale."}
            )
            assert archived.status_code == 200, archived.text
            drop = {"headers": {**operator, "Idempotency-Key": "drop-proposal"}}
            dropped = client.post(
                "/api/knowledge/proposal/reject", **drop, json={"reason": "Duplicate."}
            )
            assert dropped.status_code == 200, dropped.text
            plain = client.post(
                "/api/knowledge/plain/retire", headers=operator, json={"expected_revision": 1}
            )
            assert plain.status_code == 200, plain.text

            # Show and list return the archived entries, with the recorded reason.
            shown = client.get("/api/knowledge/archived/kept", headers=operator)
            assert shown.status_code == 200, shown.text
            body = shown.json()
            assert (body["status"], body["revision"]) == ("archived", 2)
            assert body["metadata"]["cayu:review_archive"]["reason"] == "Stale."
            assert body["metadata"]["cayu:review_archive"]["reviewer_identity"] == "operator"
            assert [chunk["text"] for chunk in body["chunks"]] == ["Made-up retiredquasar kept"]
            listed = client.get("/api/knowledge/archived", headers=operator)
            assert listed.status_code == 200, listed.text
            rows = {row["entry_id"]: row for row in listed.json()["entries"]}
            assert {
                entry_id: (row["status"], row["decision"], row["decided_by"], row["reason"])
                for entry_id, row in rows.items()
            } == {
                "kept": ("archived", "archive", "operator", "Stale."),
                "proposal": ("archived", "drop", "operator", "Duplicate."),
                "plain": ("archived", None, None, None),
            }
            assert listed.json()["next_cursor"] is None
            paged, cursor = [], None
            while True:
                page = client.get(
                    "/api/knowledge/archived",
                    headers=operator,
                    params={"limit": 1} if cursor is None else {"limit": 1, "cursor": cursor},
                )
                assert page.status_code == 200, page.text
                paged += [row["entry_id"] for row in page.json()["entries"]]
                cursor = page.json()["next_cursor"]
                if cursor is None:
                    break
            assert sorted(paged) == sorted(rows)
            # A cursor continues only the listing that returned it.
            first = client.get("/api/knowledge/archived", headers=operator, params={"limit": 1})
            assert (
                client.get(
                    "/api/knowledge/archived",
                    headers=operator,
                    params={"limit": 1, "kind": "denial", "cursor": first.json()["next_cursor"]},
                ).status_code
                == 422
            )
            # An entry that was never archived answers 404.
            assert client.get("/api/knowledge/archived/missing", headers=operator).status_code == (
                404
            )

            # Retrying with the same operation ID replays the recorded decision.
            replayed = client.post(
                "/api/knowledge/kept/retire", **archive, json={"reason": "Stale."}
            )
            assert replayed.status_code == 200, replayed.text
            assert replayed.json() == archived.json()
            redropped = client.post(
                "/api/knowledge/proposal/reject", **drop, json={"reason": "Duplicate."}
            )
            assert redropped.status_code == 200, redropped.text
            assert redropped.json() == dropped.json()
            changed = client.post(
                "/api/knowledge/kept/retire", **archive, json={"reason": "Another reason."}
            )
            assert changed.status_code == 409
            # An archived entry no decision of this operation wrote stays as hidden as it was.
            fresh = client.post(
                "/api/knowledge/kept/retire",
                headers={**operator, "Idempotency-Key": "archive-again"},
                json={"reason": "Stale."},
            )
            assert fresh.status_code == 404

            # Search, recall and the kept list (the agent's recall scope) still see none of it.
            assert client.get("/api/knowledge", headers=operator).json()["entries"] == []
            for entry_id in ("kept", "proposal", "plain"):
                assert asyncio.run(bound.get_entry(entry_id)) is None
                assert (
                    client.get(f"/api/knowledge/kept/{entry_id}", headers=operator).status_code
                    == 404
                )
            found = asyncio.run(
                bound.search(
                    KnowledgeQuery(
                        text="retiredquasar",
                        statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.ARCHIVED],
                    )
                )
            )
            assert found.hits == []

            # A member still sees no archived entry, as before.
            for path in ("/api/knowledge/archived", "/api/knowledge/archived/kept"):
                refused = client.get(path, headers=member)
                assert refused.status_code == 403, refused.text
                assert "Knowledge" not in refused.text
            assert client.get("/api/knowledge/kept/kept", headers=member).status_code == 404
            assert client.get("/api/knowledge", headers=member).json()["entries"] == []
            # Replaying the operator's drop as the member records a different reviewer, so
            # it conflicts without returning the entry.
            mine = client.post(
                "/api/knowledge/proposal/reject",
                headers={**member, "Idempotency-Key": "drop-proposal"},
                json={"reason": "Duplicate."},
            )
            assert mine.status_code == 409, mine.text
            assert "retiredquasar" not in mine.text
            probe = client.post(
                "/api/knowledge/proposal/reject",
                headers={**member, "Idempotency-Key": "drop-probe"},
                json={"reason": "Duplicate."},
            )
            assert probe.status_code == 404, probe.text
    finally:
        close = getattr(bound, "close", None)
        if close is not None:
            asyncio.run(close())


def test_archived_list_and_detail_withhold_a_tool_entrys_source_session(store):
    """As the kept list: a tool entry's source session is shown only when verified, and
    never to a caller the inbox access rule limits."""

    from cayu.server.auth import AuthContext

    async def everything(person, item):
        return True

    asyncio.run(
        store.create_entry(
            KnowledgeEntry(
                id="tool-note",
                text="Made-up tool note",
                status=KnowledgeStatus.ARCHIVED,
                source_type="tool",
                source_id="private-session",
            ),
            access_scope=_SCOPE,
        )
    )
    app = CayuApp(knowledge_store=store, knowledge_access_scope=_SCOPE, enable_logging=False)
    server = create_server(
        app,
        config=ServerConfig.protected(lambda _: AuthContext(subject="reader")),
        inbox_access=everything,
    )
    with TestClient(server) as client:
        listed = client.get("/api/knowledge/archived")
        assert listed.status_code == 200, listed.text
        (row,) = listed.json()["entries"]
        assert row["entry_id"] == "tool-note" and row["source_id"] is None
        shown = client.get("/api/knowledge/archived/tool-note")
        assert shown.status_code == 200, shown.text
        assert shown.json()["source_id"] is None
        assert "private-session" not in listed.text + shown.text


def test_archived_list_survives_an_entry_relabeled_into_the_review_labels(store):
    """An archived entry whose earlier revision is outside the review labels still lists;
    its earlier revision just can't be read."""

    entry = KnowledgeEntry(
        id="moved",
        text="Made-up moved note",
        status=KnowledgeStatus.ARCHIVED,
        labels={"team": "other"},
    )
    asyncio.run(store.create_entry(entry, access_scope=_SCOPE))
    asyncio.run(
        store.append_entry_revision(
            entry.model_copy(update={"revision": 2, "labels": {"team": "mine"}}),
            expected_revision=1,
            access_scope=_SCOPE,
        )
    )
    _propose(store, "mine", labels={"team": "mine"})
    with _client(store, knowledge_review_labels={"team": "mine"}) as client:
        listed = client.get("/api/knowledge/archived")
        assert listed.status_code == 200, listed.text
        (row,) = listed.json()["entries"]
        assert (row["entry_id"], row["decision"]) == ("moved", None)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_a_repeated_inbox_drop_on_a_recall_scope_replays_as_on_cayu_cloud(store_kind, tmp_path):
    """The recall scope can't read a proposal once the inbox drops it. Repeating that Drop
    replays the settlement to anyone the inbox access rule lets see the proposal; who
    dropped it and why go only to that person and to full-access callers."""

    import base64

    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    from cayu.server import BasicAuth
    from cayu.storage import SQLiteKnowledgeStore

    recall = KnowledgeAccessScope(
        allowed_namespaces=["default"],
        allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    )
    bound = (
        InMemoryKnowledgeStore(access_scope=recall)
        if store_kind == "memory"
        else SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=recall)
    )
    asyncio.run(
        bound.create_entry(
            KnowledgeEntry(
                id="proposal",
                namespace="default",
                text="Made-up droppedquasar proposal",
                status=KnowledgeStatus.PENDING,
            )
        )
    )

    async def inbox_access(person, item):
        # The third member may not see knowledge proposals.
        return person.subject != "person-3"

    key = ec.generate_private_key(ec.SECP256R1())
    auth = _auth(
        FakeJwks(key), fallback=BasicAuth(username="operator", password="made-up-password")
    )
    operator = {"Authorization": "Basic " + base64.b64encode(b"operator:made-up-password").decode()}
    dropper, other, outsider = (
        _as(_token(key, role="member", sub=subject))
        for subject in ("person-1", "person-2", "person-3")
    )
    app = CayuApp(knowledge_store=bound, knowledge_access_scope=recall, enable_logging=False)
    server = create_server(app, config=ServerConfig.protected(auth), inbox_access=inbox_access)
    try:
        with TestClient(server) as client:
            (item,) = client.get("/api/inbox", headers=dropper).json()["items"]
            settle = f"/api/inbox/{item['id']}/settle"
            drop = {"decision": "drop", "reason": "Duplicate."}
            first = client.post(settle, headers=dropper, json=drop)
            assert first.status_code == 200, first.text
            assert first.json()["replayed"] is False
            recorded = first.json()["item"]["settlement"]

            # The member who dropped it gets the settlement replayed, with who and why.
            again = client.post(settle, headers=dropper, json=drop)
            assert again.status_code == 200, again.text
            assert again.json() == {"replayed": True, "item": first.json()["item"]}
            assert recorded["decided_by"]["subject"] == "person-1"
            assert recorded["reason"] == "Duplicate."
            changed = client.post(
                settle, headers=dropper, json={"decision": "drop", "reason": "Another."}
            )
            assert changed.status_code == 409, changed.text

            # Another member who may see it gets the settlement without who and why.
            seen = client.post(settle, headers=other, json={"decision": "drop"})
            assert seen.status_code == 200, seen.text
            settlement = seen.json()["item"]["settlement"]
            assert seen.json()["replayed"] is True
            assert seen.json()["item"]["status"] == "settled"
            assert settlement == {
                "decision": "drop",
                "decided_at": recorded["decided_at"],
                "decided_by": None,
                "reason": None,
            }
            kept = client.post(settle, headers=other, json={"decision": "keep"})
            assert kept.status_code == 409, kept.text
            assert "Duplicate." not in kept.text and "person-1" not in kept.text

            # A member the inbox access rule doesn't let see it gets 404, as before.
            assert client.post(settle, headers=outsider, json=drop).status_code == 404

            # The operator has full access, so gets who and why.
            full = client.post(settle, headers=operator, json={"decision": "drop"})
            assert full.status_code == 200, full.text
            assert full.json()["item"]["settlement"] == recorded

            # A fresh operation key on the dropped entry still finds nothing.
            fresh = client.post(
                "/api/knowledge/proposal/reject",
                headers={**operator, "Idempotency-Key": "fresh-key"},
                json={"reason": "Duplicate."},
            )
            assert fresh.status_code == 404, fresh.text
            # Members still can't use the archived routes.
            assert client.get("/api/knowledge/archived", headers=other).status_code == 403
        # Search, with the agent's recall scope, still finds nothing archived.
        found = asyncio.run(
            bound.search(
                KnowledgeQuery(
                    text="droppedquasar",
                    statuses=[KnowledgeStatus.PENDING, KnowledgeStatus.ARCHIVED],
                )
            )
        )
        assert found.hits == []
    finally:
        close = getattr(bound, "close", None)
        if close is not None:
            asyncio.run(close())
