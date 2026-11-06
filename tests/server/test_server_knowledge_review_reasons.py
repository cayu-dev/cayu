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
