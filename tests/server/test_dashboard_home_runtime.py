"""Home prerequisite contracts through authenticated HTTP entrances."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.server.test_server_inbox import _as, _authenticate, _durable_app, _pause

from cayu.server import ServerConfig, create_server


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_inbox_search_scopes_pages_counts_and_cursor(store_kind, tmp_path):
    runtime = _durable_app(store_kind, tmp_path)
    with TestClient(create_server(runtime, config=ServerConfig.protected(_authenticate))) as client:
        for session in ["invoice-first", "other", "invoice-last"]:
            _pause(client, runtime, session, "approver")
        query = {"q": "INVOICE", "limit": 1}
        first = client.get("/api/inbox", params=query, headers=_as("priya"))
        assert first.status_code == 200, first.text
        body = first.json()
        cursor = body["next_cursor"]
        assert cursor is not None
        ids = {item["session_id"] for item in body["items"]}
        position = cursor
        while position is not None:
            second = client.get(
                "/api/inbox", params={**query, "cursor": position}, headers=_as("priya")
            )
            assert second.status_code == 200, second.text
            ids.update(item["session_id"] for item in second.json()["items"])
            position = second.json()["next_cursor"]
        assert ids == {"invoice-first", "invoice-last"}
        counts = client.get("/api/inbox/counts", params={"q": "INVOICE"}, headers=_as("priya"))
        assert counts.status_code == 200, counts.text
        assert counts.json()["total_count"] == 2
        changed = client.get(
            "/api/inbox", params={**query, "q": "other", "cursor": cursor}, headers=_as("priya")
        )
        assert changed.status_code == 422


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_inbox_search_never_matches_text_hidden_by_current_redaction(store_kind, tmp_path):
    from tests.server.test_server_inbox import _app

    from cayu.vaults.redaction import SecretRedactor

    if store_kind == "sqlite":
        import base64

        from pydantic import SecretStr

        from cayu.runtime.public_authority import (
            PublicAuthorityAliasCodec,
            PublicAuthorityAliasKeyring,
        )
        from cayu.storage import SQLiteSessionStore

        codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(
                active_key_id="test",
                keys={
                    "test": SecretStr(
                        base64.urlsafe_b64encode(bytes(range(32))).decode().rstrip("=")
                    )
                },
            )
        )
        writer = _app(
            session_store=SQLiteSessionStore(
                tmp_path / "redaction.db", public_authority_alias_codec=codec
            )
        )
    else:
        writer = _durable_app(store_kind, tmp_path)
    reader = _app(session_store=writer.session_store, secret_redactor=SecretRedactor("September"))
    with TestClient(create_server(reader, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, writer, "safe-session", "approver")
        all_items = client.get("/api/inbox", headers=_as("priya")).json()["items"]
        assert len(all_items) == 1
        assert "September" not in str(all_items)
        for needle, total in [("September", 0), ("correction", 1)]:
            response = client.get("/api/inbox", params={"q": needle}, headers=_as("priya"))
            assert response.status_code == 200, response.text
            assert len(response.json()["items"]) == total
            counts = client.get("/api/inbox/counts", params={"q": needle}, headers=_as("priya"))
            assert counts.json()["total_count"] == total


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_answer_reason_is_durable_and_part_of_exact_replay(store_kind, tmp_path):
    from tests.server.test_server_inbox import _items, _settle, _supervision, _wait_for_status

    runtime = _durable_app(store_kind, tmp_path)
    with TestClient(create_server(runtime, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, runtime, "answer-reason", "asker")
        [item] = _items(client, "priya")
        request = {"decision": "answer", "answer": "staging", "reason": "Change reviewed."}
        settled = _settle(client, item["id"], "priya", **request)
        assert settled.status_code == 200, settled.text
        _wait_for_status(client, runtime, "answer-reason")
        state = client.get(f"/api/inbox/{item['id']}", headers=_as("tom")).json()
        assert state["settlement"]["reason"] == "Change reviewed."
        assert _supervision(client)[0]["reason"] == "Change reviewed."
        replay = _settle(client, item["id"], "priya", **request)
        assert replay.status_code == 200 and replay.json()["replayed"] is True
        conflict = _settle(
            client, item["id"], "priya", **{**request, "reason": "Different review."}
        )
        assert conflict.status_code == 409


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_daily_overview_uses_exact_accounting_window(store_kind, tmp_path):
    from datetime import UTC, datetime, timedelta
    from decimal import Decimal

    from tests.core.test_cloud_reporting import _app as usage_app
    from tests.core.test_cloud_reporting import _start

    from cayu.budgets.pricing import default_price_book
    from cayu.server.config import DashboardConfig
    from cayu.sessions.event_queries import EventQuery
    from cayu.storage import SQLiteSessionStore

    runtime = usage_app(
        None if store_kind == "memory" else SQLiteSessionStore(tmp_path / "usage.db")
    )
    pricing = default_price_book()
    config = ServerConfig.local_development(
        dashboard=DashboardConfig(runtime_config={"priceBook": pricing})
    )
    with TestClient(create_server(runtime, config=config)) as client:
        since = datetime.now(UTC) - timedelta(seconds=1)
        client.portal.call(_start, runtime, "accounted", "writer")
        until = datetime.now(UTC) + timedelta(seconds=1)
        query = EventQuery(session_ids=("accounted",), since=since, until=until)
        expected_usage = client.portal.call(runtime.session_store.read_usage_accounting, query)
        expected_cost = client.portal.call(
            runtime.session_store.read_cost_accounting, query, pricing
        )
        response = client.get(
            "/api/sessions/overview/counts",
            params={"since": since.isoformat(), "until": until.isoformat()},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["tokens"] == {
            "input": str(expected_usage.summary.usage.input_tokens),
            "output": str(expected_usage.summary.usage.output_tokens),
            "cached": str(expected_usage.summary.usage.cache.read_tokens),
        }
        assert Decimal(body["cost"]["amount"]) == expected_cost.totals.total_cost > 0
        assert body["cost"]["complete"] is True
        empty_window = client.get(
            "/api/sessions/overview/counts",
            params={
                "since": until.isoformat(),
                "until": (until + timedelta(seconds=1)).isoformat(),
            },
        ).json()
        assert empty_window["tokens"] == {"input": "0", "output": "0", "cached": "0"}
        assert Decimal(empty_window["cost"]["amount"]) == 0


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("decision", ["keep", "drop"])
def test_knowledge_reason_has_exact_atomic_audit_and_replay(store_kind, decision, tmp_path):
    from tests.server.test_server_inbox import _app, _items, _settle

    from cayu.storage import (
        InMemoryKnowledgeStore,
        KnowledgeAccessScope,
        KnowledgeEntry,
        KnowledgeStatus,
        SQLiteKnowledgeStore,
    )

    scope = KnowledgeAccessScope.privileged()
    store = (
        InMemoryKnowledgeStore(access_scope=scope)
        if store_kind == "memory"
        else SQLiteKnowledgeStore(tmp_path / "knowledge.db", access_scope=scope)
    )
    runtime = _app(knowledge_store=store, knowledge_access_scope=scope)
    with TestClient(create_server(runtime, config=ServerConfig.protected(_authenticate))) as client:
        client.portal.call(
            store.create_entry,
            KnowledgeEntry(
                id="proposal",
                text="Preserved evidence",
                title="A proposal",
                status=KnowledgeStatus.PENDING,
            ),
        )
        [item] = _items(client, "priya")
        request = {"decision": decision, "reason": "Reviewed source evidence."}
        response = _settle(client, item["id"], "priya", **request)
        assert response.status_code == 200, response.text
        state = client.get(f"/api/inbox/{item['id']}", headers=_as("tom")).json()
        assert state["settlement"]["reason"] == request["reason"]
        assert state["settlement"]["decided_by"]["subject"] == "priya"
        replay = _settle(client, item["id"], "priya", **request)
        assert replay.status_code == 200 and replay.json()["replayed"] is True
        for person, reason in [("tom", request["reason"]), ("priya", "Changed evidence.")]:
            changed = _settle(client, item["id"], person, decision=decision, reason=reason)
            assert changed.status_code == 409
        entry = client.portal.call(store.get_entry, "proposal")
        assert entry.revision == 2 and entry.text == "Preserved evidence"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_drop_acknowledgement_loss_replays_one_committed_revision(
    store_kind, tmp_path, monkeypatch
):
    from tests.server.test_server_inbox import _app, _items, _settle

    from cayu.storage import (
        InMemoryKnowledgeStore,
        KnowledgeAccessScope,
        KnowledgeEntry,
        KnowledgeStatus,
        SQLiteKnowledgeStore,
    )

    scope = KnowledgeAccessScope.privileged()
    store = (
        InMemoryKnowledgeStore(access_scope=scope)
        if store_kind == "memory"
        else SQLiteKnowledgeStore(tmp_path / "drop-loss.db", access_scope=scope)
    )
    runtime = _app(knowledge_store=store, knowledge_access_scope=scope)
    original = store.publish_entry_revision
    lost = False

    async def lose_ack(*args, **kwargs):
        nonlocal lost
        result = await original(*args, **kwargs)
        if not lost:
            lost = True
            raise OSError("lost publication acknowledgement")
        return result

    monkeypatch.setattr(store, "publish_entry_revision", lose_ack)
    with TestClient(
        create_server(runtime, config=ServerConfig.protected(_authenticate)),
        raise_server_exceptions=False,
    ) as client:
        client.portal.call(
            store.create_entry,
            KnowledgeEntry(id="proposal", text="Must survive", status=KnowledgeStatus.PENDING),
        )
        [item] = _items(client, "priya")
        request = {"decision": "drop", "reason": "Duplicated source."}
        assert _settle(client, item["id"], "priya", **request).status_code == 500
        assert client.portal.call(store.get_entry, "proposal").revision == 2
        retry = _settle(client, item["id"], "priya", **request)
        assert retry.status_code == 200, retry.text
        assert retry.json()["replayed"] is True
        assert retry.json()["item"]["settlement"]["reason"] == request["reason"]
        assert client.portal.call(store.get_entry, "proposal").revision == 2


def test_drop_receipt_failure_rolls_back_sqlite_revision(tmp_path, monkeypatch):
    from tests.server.test_server_inbox import _app, _items, _settle

    from cayu.storage import (
        KnowledgeAccessScope,
        KnowledgeEntry,
        KnowledgeStatus,
        SQLiteKnowledgeStore,
    )

    scope = KnowledgeAccessScope.privileged()
    store = SQLiteKnowledgeStore(tmp_path / "drop-rollback.db", access_scope=scope)
    runtime = _app(knowledge_store=store, knowledge_access_scope=scope)
    insert = store._insert_publication_receipt_unlocked

    def fail_receipt(*_args, **_kwargs):
        raise OSError("receipt write failed after staged revision")

    with TestClient(
        create_server(runtime, config=ServerConfig.protected(_authenticate)),
        raise_server_exceptions=False,
    ) as client:
        client.portal.call(
            store.create_entry,
            KnowledgeEntry(id="proposal", text="Must survive", status=KnowledgeStatus.PENDING),
        )
        [item] = _items(client, "priya")
        monkeypatch.setattr(store, "_insert_publication_receipt_unlocked", fail_receipt)
        request = {"decision": "drop", "reason": "Duplicated source."}
        assert _settle(client, item["id"], "priya", **request).status_code == 500
        unchanged = client.portal.call(store.get_entry, "proposal")
        assert unchanged.revision == 1 and unchanged.status is KnowledgeStatus.PENDING
        monkeypatch.setattr(store, "_insert_publication_receipt_unlocked", insert)
        retry = _settle(client, item["id"], "priya", **request)
        assert retry.status_code == 200, retry.text
        assert retry.json()["replayed"] is False
        assert client.portal.call(store.get_entry, "proposal").revision == 2


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_daily_accounting_excludes_eval_and_fork_lineage(store_kind, tmp_path):
    from datetime import UTC, datetime, timedelta

    from tests.core.test_cloud_reporting import _app as usage_app
    from tests.core.test_cloud_reporting import _start

    from cayu.sessions import ForkSessionRequest
    from cayu.sessions._run_identity import eval_trial_scope
    from cayu.sessions.event_queries import EventQuery
    from cayu.storage import SQLiteSessionStore

    runtime = usage_app(
        None if store_kind == "memory" else SQLiteSessionStore(tmp_path / "excluded.db")
    )
    since = datetime.now(UTC) - timedelta(seconds=1)

    async def populate():
        await _start(runtime, "normal", "writer")
        async with eval_trial_scope():
            await _start(runtime, "eval", "writer")
        async for _event in runtime.fork_session(
            ForkSessionRequest(source_session_id="normal", session_id="fork")
        ):
            pass

    with TestClient(create_server(runtime, config=ServerConfig.local_development())) as client:
        client.portal.call(populate)
        until = datetime.now(UTC) + timedelta(seconds=1)
        params = {"since": since.isoformat(), "until": until.isoformat()}
        body = client.get("/api/sessions/overview/counts", params=params).json()
        expected = client.portal.call(
            runtime.session_store.read_usage_accounting,
            EventQuery(session_ids=("normal",), since=since, until=until),
        )
        assert body["tokens"]["input"] == str(expected.summary.usage.input_tokens)
        assert body["total_count"] == "1" and body["cost"] is None
        included = client.get(
            "/api/sessions/overview/counts",
            params={**params, "include_eval": True, "include_forks": True},
        ).json()
        expected_all = client.portal.call(
            runtime.session_store.read_usage_accounting,
            EventQuery(session_ids=("normal", "eval", "fork"), since=since, until=until),
        )
        assert included["total_count"] == "3"
        assert included["tokens"]["input"] == str(expected_all.summary.usage.input_tokens)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_daily_accounting_refuses_changed_incarnation(store_kind, tmp_path, monkeypatch):
    from tests.core.test_cloud_reporting import _app as usage_app
    from tests.core.test_cloud_reporting import _start

    from cayu.storage import SQLiteSessionStore

    runtime = usage_app(
        None if store_kind == "memory" else SQLiteSessionStore(tmp_path / "replaced.db")
    )
    original = runtime.session_store.read_usage_accounting
    changed = False

    async def replace_then_read(query, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            await runtime.session_store.delete_session("reused")
            await _start(runtime, "reused", "writer")
        return await original(query, **kwargs)

    with TestClient(create_server(runtime, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, runtime, "reused", "writer")
        monkeypatch.setattr(runtime.session_store, "read_usage_accounting", replace_then_read)
        response = client.get("/api/sessions/overview/counts")
        assert response.status_code == 503, response.text
        assert "tokens" not in response.json()


def test_reviewed_drop_metadata_alone_cannot_forge_audit():
    from tests.server.test_server_inbox import _app, _items, _settle

    from cayu.storage import (
        InMemoryKnowledgeStore,
        KnowledgeAccessScope,
        KnowledgeEntry,
        KnowledgeStatus,
    )

    scope = KnowledgeAccessScope.privileged()
    proposal = KnowledgeEntry(
        id="proposal", text="Identical material", status=KnowledgeStatus.PENDING
    )
    trusted_store = InMemoryKnowledgeStore([proposal], access_scope=scope)
    trusted = _app(knowledge_store=trusted_store, knowledge_access_scope=scope)
    with TestClient(create_server(trusted, config=ServerConfig.protected(_authenticate))) as client:
        [item] = _items(client, "priya")
        settled = _settle(client, item["id"], "priya", decision="drop", reason="Reviewed.")
        assert settled.status_code == 200, settled.text
        successor = client.portal.call(trusted_store.get_entry, "proposal")
        assert settled.json()["item"]["settlement"]["decided_by"]["subject"] == "priya"

    untrusted_store = InMemoryKnowledgeStore([proposal], access_scope=scope)
    untrusted = _app(knowledge_store=untrusted_store, knowledge_access_scope=scope)
    with TestClient(
        create_server(untrusted, config=ServerConfig.protected(_authenticate))
    ) as client:
        # Identical public material and actor strings, but no store-owned receipt.
        async def forge():
            await untrusted_store.append_entry_revision(successor, expected_revision=1)

        client.portal.call(forge)
        state = client.get(f"/api/inbox/{item['id']}", headers=_as("priya")).json()
        assert state["status"] == "superseded"
        assert state["settlement"] is None


def test_reviewed_drop_and_answer_reasons_survive_sqlite_reopen(tmp_path):
    import asyncio

    from tests.server.test_server_inbox import _app, _settle, _wait_for_status

    from cayu.storage import (
        KnowledgeAccessScope,
        KnowledgeEntry,
        KnowledgeStatus,
        SQLiteKnowledgeStore,
        SQLiteSessionStore,
    )

    scope = KnowledgeAccessScope.privileged()
    sessions_path = tmp_path / "sessions-reopen.db"
    knowledge_path = tmp_path / "knowledge-reopen.db"
    store = SQLiteKnowledgeStore(knowledge_path, access_scope=scope)
    runtime = _app(
        session_store=SQLiteSessionStore(sessions_path),
        knowledge_store=store,
        knowledge_access_scope=scope,
    )
    with TestClient(create_server(runtime, config=ServerConfig.protected(_authenticate))) as client:
        client.portal.call(
            store.create_entry,
            KnowledgeEntry(id="proposal", text="Preserve source", status=KnowledgeStatus.PENDING),
        )
        _pause(client, runtime, "answer", "asker")
        items = {}
        cursor = None
        while True:
            page = client.get(
                "/api/inbox",
                params={} if cursor is None else {"cursor": cursor},
                headers=_as("priya"),
            ).json()
            items.update({item["kind"]: item for item in page["items"]})
            cursor = page["next_cursor"]
            if cursor is None:
                break
        answer = {"decision": "answer", "answer": "staging", "reason": "Approved environment."}
        drop = {"decision": "drop", "reason": "Source duplicated."}
        for kind, request in [("question", answer), ("knowledge", drop)]:
            response = _settle(client, items[kind]["id"], "priya", **request)
            assert response.status_code == 200, response.text
        _wait_for_status(client, runtime, "answer")
    asyncio.run(runtime.aclose())
    reopened = _app(
        session_store=SQLiteSessionStore(sessions_path),
        knowledge_store=SQLiteKnowledgeStore(knowledge_path, access_scope=scope),
        knowledge_access_scope=scope,
    )
    with TestClient(
        create_server(reopened, config=ServerConfig.protected(_authenticate))
    ) as client:
        for kind, request in [("question", answer), ("knowledge", drop)]:
            item_id = items[kind]["id"]
            state = client.get(f"/api/inbox/{item_id}", headers=_as("tom")).json()
            assert state["settlement"]["reason"] == request["reason"]
            replay = _settle(client, item_id, "priya", **request)
            assert replay.status_code == 200 and replay.json()["replayed"] is True
    asyncio.run(reopened.aclose())


@pytest.mark.parametrize("bad", [True, "secret-repr"])
def test_daily_accounting_rejects_mutated_summary_without_diagnostics(
    bad, monkeypatch, capsys, caplog
):
    import warnings

    from tests.core.test_cloud_reporting import _app as usage_app
    from tests.core.test_cloud_reporting import _start

    canary = "private-accounting-object-canary"

    class SecretValue:
        def __repr__(self):
            return canary

        def __str__(self):
            return canary

    runtime = usage_app()
    original = runtime.session_store.read_usage_accounting

    async def malformed(query, **kwargs):
        result = await original(query, **kwargs)
        object.__setattr__(
            result.summary.usage, "input_tokens", bad if bad is True else SecretValue()
        )
        return result

    with TestClient(create_server(runtime, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, runtime, "measured", "writer")
        monkeypatch.setattr(runtime.session_store, "read_usage_accounting", malformed)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            response = client.get("/api/sessions/overview/counts")
        assert response.status_code == 503, response.text
        assert "tokens" not in response.json()
        captured = capsys.readouterr()
        assert canary not in captured.out + captured.err + caplog.text + response.text + str(
            [warning.message for warning in recorded]
        )


def test_home_prerequisites_postgres_http(postgres_dsn):
    from tests.server.test_server_inbox import _app, _settle, _wait_for_status

    from cayu.storage import (
        KnowledgeAccessScope,
        KnowledgeEntry,
        KnowledgeStatus,
        PostgresKnowledgeStore,
        PostgresSessionStore,
    )
    from cayu.storage.migrations import SchemaMode

    scope = KnowledgeAccessScope.privileged()
    sessions = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
    knowledge = PostgresKnowledgeStore(
        postgres_dsn, schema_mode=SchemaMode.CREATE, access_scope=scope
    )
    runtime = _app(session_store=sessions, knowledge_store=knowledge, knowledge_access_scope=scope)
    with TestClient(create_server(runtime, config=ServerConfig.protected(_authenticate))) as client:
        try:
            client.portal.call(
                knowledge.create_entry,
                KnowledgeEntry(id="pg-proposal", text="Evidence", status=KnowledgeStatus.PENDING),
            )
            _pause(client, runtime, "pg-question", "asker")
            absent = client.get(
                "/api/inbox/counts", params={"q": "unmatched needle"}, headers=_as("priya")
            )
            assert absent.status_code == 200 and absent.json()["total_count"] == 0
            for kind, request in [
                (
                    "question",
                    {"decision": "answer", "answer": "staging", "reason": "Reviewed environment."},
                ),
                ("knowledge", {"decision": "drop", "reason": "Reviewed source."}),
            ]:
                items = client.get(
                    "/api/inbox", params={"kind": kind}, headers=_as("priya")
                ).json()["items"]
                assert len(items) == 1
                response = _settle(client, items[0]["id"], "priya", **request)
                assert response.status_code == 200, response.text
                assert response.json()["item"]["settlement"]["reason"] == request["reason"]
            _wait_for_status(client, runtime, "pg-question")
            daily = client.get("/api/sessions/overview/counts", headers=_as("priya"))
            assert daily.status_code == 200, daily.text
            assert set(daily.json()["tokens"]) == {"input", "output", "cached"}
        finally:
            client.portal.call(runtime.aclose)


@pytest.mark.parametrize(
    "agent,decision",
    [
        ("approver", {"decision": "approve", "reason": "Approved after review."}),
        ("asker", {"decision": "answer", "answer": "staging", "reason": "Answered after review."}),
    ],
)
@pytest.mark.parametrize(
    "boundary",
    [
        "ordinary",
        "lost_ack",
        "active_peer",
        "replaced_handoff",
        "protected_review",
        "unsupported",
        "human_backoff",
        "dynamic_scope",
        "unknown_scope",
    ],
)
@pytest.mark.parametrize("task_backend", ["memory", "sqlite"])
def test_inbox_explicitly_claims_its_own_interrupted_task_continuation(
    agent, decision, boundary, task_backend, tmp_path, monkeypatch
):
    from tests.server.test_server_inbox import _app, _items, _portal, _settle

    from cayu import Message, RunRequest
    from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
    from cayu.tasks.creation import TaskCreate
    from cayu.tasks.memory import InMemoryTaskStore
    from cayu.tasks.records import TaskStatus
    from cayu.tasks.worker import TaskHandlerOutcome, run_task_worker

    store_base = InMemoryTaskStore if task_backend == "memory" else SQLiteTaskStore

    class ClaimStore(store_base):
        supports_interrupted_task_handoffs = True
        supports_bound_interrupted_task_handoffs = boundary != "unsupported"
        claim_called = False
        verified_work_mutations_are_cancellation_quiescent = True
        lost_ack = False

        async def claim_interrupted_task_continuation(self, *args, **kwargs):
            self.claim_called = True
            if boundary == "replaced_handoff" and kwargs.get("expected") is not None:
                from uuid import uuid4

                from cayu.tasks.handoff import interrupted_task_handoff_request

                peer = await super().claim_interrupted_task_continuation(
                    "racing-peer", handoff_id=str(uuid4()), task_id="controlled-task"
                )
                session = await runtime.session_store.load("controlled-inbox")
                await self.release_interrupted_task_worker(
                    interrupted_task_handoff_request(peer.task, session_run_epoch=session.run_epoch)
                )
                self.raced_task = await self.load_task("controlled-task")
            result = await super().claim_interrupted_task_continuation(*args, **kwargs)
            if boundary == "lost_ack" and kwargs.get("expected") is not None and not self.lost_ack:
                self.lost_ack = True
                raise OSError("claim committed before acknowledgement disappeared")
            return result

    from cayu.approvals import HumanReviewDisclosure, HumanReviewField, HumanReviewPolicy

    class Review(HumanReviewPolicy):
        version = "controlled-inbox-review-v1"

        @property
        def binding_key(self):
            return b"k" * 32

        def authorize(self, context, *, session_id, session_metadata, action):
            return context.recipient == "priya" and context.purpose == "inbox"

        def project(self, context, source):
            return HumanReviewDisclosure(
                status="permitted",
                sensitive_content="application_attested",
                fields=(HumanReviewField(label="Decision", text="Review this request."),),
            )

    review_options = {"human_review_policy": Review()} if boundary == "protected_review" else {}
    if task_backend == "sqlite":
        tasks = ClaimStore(tmp_path / "controlled-tasks.db")
        runtime = _app(
            task_store=tasks,
            session_store=SQLiteSessionStore(tmp_path / "controlled-sessions.db"),
            **review_options,
        )
    else:
        tasks = ClaimStore()
        runtime = _app(task_store=tasks, **review_options)

    if boundary == "dynamic_scope":
        from tests.server.test_inbox_reason_privacy import configure_scope

        configure_scope(runtime, "dynamic")

    async def pause(app, task, worker_id):
        async for _event in app.run(
            RunRequest(
                agent_name=agent,
                session_id="controlled-inbox",
                task_id=task.id,
                task_worker_id=worker_id,
                task_lease_expires_at=task.lease_expires_at,
                messages=[Message.text("user", "go")],
            )
        ):
            pass
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def seed():
        await tasks.create_task(TaskCreate(task_id="controlled-task", type="job"))
        await run_task_worker(
            runtime, tasks, pause, worker_id="original", max_tasks=1, reclaim=False
        )
        return await tasks.load_task("controlled-task")

    from cayu.server import ControlPlaneRoles

    class MemberAuth:
        control_plane_roles = ControlPlaneRoles()

        def __call__(self, request):
            context = _authenticate(request).model_copy(update={"claims": {"role": "member"}})
            self.control_plane_roles.require(request, context)
            return context

    with TestClient(create_server(runtime, config=ServerConfig.protected(MemberAuth()))) as client:
        assert client.post("/api/run", json={}, headers=_as("priya")).status_code == 403
        before = _portal(client).call(seed)
        if boundary == "unknown_scope":
            from tests.server.test_inbox_reason_privacy import retain_unknown_scope

            _portal(client).call(retain_unknown_scope, runtime, "controlled-inbox")
        delays = []
        if boundary == "human_backoff":
            import cayu.tasks.worker as worker_module

            async def waiting_human(app, task, worker_id):
                return TaskHandlerOutcome.SESSION_INTERRUPTED

            async def automatic_handoff():
                await worker_module._run_recovered_interrupted_task_handler(
                    runtime,
                    tasks,
                    waiting_human,
                    worker_id="automated-probe",
                    lease_seconds=300,
                    query=None,
                    after=None,
                )

            _portal(client).call(automatic_handoff)
            before = _portal(client).call(tasks.load_task, "controlled-task")
            original_backoff = worker_module._unproductive_continuation_backoff

            def record_backoff(count):
                delay = original_backoff(count)
                if delay > 0:
                    delays.append(delay)
                return delay

            monkeypatch.setattr(worker_module, "_unproductive_continuation_backoff", record_backoff)
            _portal(client).call(automatic_handoff)
            assert delays == [1.0], "Automated continuation must retain its retry backoff."
            delays.clear()
            before = _portal(client).call(tasks.load_task, "controlled-task")
        assert before.worker_id is None
        [item] = _items(client, "priya")
        if agent == "approver":
            override = _settle(
                client,
                item["id"],
                "priya",
                **decision,
                task_continuation="claim",
                arguments={"note": "member override"},
            )
            assert override.status_code == 403, override.text
            assert _portal(client).call(tasks.load_task, "controlled-task") == before
        if boundary == "protected_review":
            denied = _settle(client, item["id"], "priya", **decision, task_continuation="claim")
            assert denied.status_code == 403, denied.text
            assert _portal(client).call(tasks.load_task, "controlled-task") == before
            review = client.get(
                "/api/sessions/controlled-inbox/human-review",
                params={"purpose": "inbox"},
                headers=_as("priya"),
            )
            assert review.status_code == 200, review.text
            decision = {**decision, "review_reference": review.json()["reference"]}
            denied = _settle(client, item["id"], "tom", **decision, task_continuation="claim")
            assert denied.status_code == 403
            assert _portal(client).call(tasks.load_task, "controlled-task") == before
        if boundary == "active_peer":

            async def elect_peer():
                from uuid import uuid4

                return await tasks.claim_interrupted_task_continuation(
                    "peer", handoff_id=str(uuid4()), task_id="controlled-task"
                )

            elected = _portal(client).call(elect_peer)
        response = _settle(client, item["id"], "priya", **decision, task_continuation="claim")
        if boundary == "human_backoff":
            assert delays == [], "Explicit human decision entered automated retry backoff."
        if boundary == "unsupported":
            assert response.status_code == 501, response.text
            assert not tasks.claim_called
            assert _portal(client).call(tasks.load_task, "controlled-task") == before
            assert len(_items(client, "priya")) == 1
            return
        if boundary in {"active_peer", "replaced_handoff"}:
            assert response.status_code == 409, response.text
            expected_task = elected.task if boundary == "active_peer" else tasks.raced_task
            assert _portal(client).call(tasks.load_task, "controlled-task") == expected_task
            assert len(_items(client, "priya")) == 1
            return
        assert response.status_code == 200, response.text
        assert response.json()["replayed"] is False
        expected_reason = (
            None if boundary in {"dynamic_scope", "unknown_scope"} else decision["reason"]
        )
        assert response.json()["item"]["settlement"]["reason"] == expected_reason
        retry = _settle(client, item["id"], "priya", **decision, task_continuation="claim")
        assert retry.status_code == 200, retry.text
        assert retry.json()["replayed"] is True

        async def completed():
            import asyncio

            async with asyncio.timeout(10):
                while True:
                    current = await tasks.load_task("controlled-task")
                    if current.status == TaskStatus.COMPLETED:
                        return current
                    await asyncio.sleep(0.01)

        after = _portal(client).call(completed)
        assert after.status == TaskStatus.COMPLETED
        assert "task_worker_id" not in response.text
        assert "task_handoff_id" not in response.text


def test_inbox_controlled_worker_retains_lease_until_dispatched_tool_quiesces(monkeypatch):
    import asyncio

    from tests.server.test_server_inbox import _app, _items, _NoteTool, _portal, _settle

    from cayu import Message, RunRequest
    from cayu.server.routes import _detached_event_pumps
    from cayu.tasks.creation import TaskCreate
    from cayu.tasks.memory import InMemoryTaskStore
    from cayu.tasks.worker import TaskHandlerOutcome, run_task_worker

    entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    tasks = InMemoryTaskStore()
    runtime = _app(task_store=tasks)

    async def blocked_tool(self, ctx, args):
        entered.set()
        from cayu.tools import ToolResult

        await release_cleanup.wait()
        return ToolResult(content="Completed after the owner stopped waiting.")

    monkeypatch.setattr(_NoteTool, "run", blocked_tool)

    async def pause(app, task, worker_id):
        async for _event in app.run(
            RunRequest(
                agent_name="approver",
                session_id="cancelled-inbox",
                task_id=task.id,
                task_worker_id=worker_id,
                task_lease_expires_at=task.lease_expires_at,
                messages=[Message.text("user", "go")],
            )
        ):
            pass
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def seed():
        await tasks.create_task(TaskCreate(task_id="cancelled-task", type="job"))
        await run_task_worker(runtime, tasks, pause, worker_id="seed", max_tasks=1, reclaim=False)
        return set(_detached_event_pumps)

    with TestClient(create_server(runtime, config=ServerConfig.protected(_authenticate))) as client:
        prior_pumps = _portal(client).call(seed)
        [item] = _items(client, "priya")
        response = _settle(
            client, item["id"], "priya", decision="approve", task_continuation="claim"
        )
        assert response.status_code == 200, response.text

        async def cancel_owner():
            from uuid import uuid4

            async with asyncio.timeout(15):
                await entered.wait()
                [pump] = set(_detached_event_pumps) - prior_pumps
                before = await tasks.load_task("cancelled-task")
                assert before.worker_id is not None
                pump.cancel("cancel controlled continuation")
                from cayu.tasks.cancellation import _task_cancellation_requested

                while not _task_cancellation_requested(await tasks.load_task("cancelled-task")):
                    await asyncio.sleep(0.01)
                assert pump.cancelling() > 0
                assert not pump.done()
                during = await tasks.load_task("cancelled-task")
                assert during.worker_id == before.worker_id
                competitor = await tasks.claim_interrupted_task_continuation(
                    "competitor", task_id="cancelled-task", handoff_id=str(uuid4())
                )
                assert competitor.task is None
                release_cleanup.set()
                try:
                    await pump
                except asyncio.CancelledError:
                    pass
                else:
                    pytest.fail("The owner cancellation was swallowed.")
                assert pump.cancelled()
                assert pump.cancelling() > 0
                after = await tasks.load_task("cancelled-task")
                from cayu.tasks.records import TaskStatus

                assert after.status is TaskStatus.CANCELLED
                assert after.worker_id is None
                assert after.lease_expires_at is None

        try:
            _portal(client).call(cancel_owner)
        finally:
            _portal(client).call(release_cleanup.set)


def test_audited_drop_without_atomic_publication_is_unavailable_without_mutation():
    from tests.server.test_server_inbox import _app, _items, _settle

    from cayu.storage import (
        InMemoryKnowledgeStore,
        KnowledgeAccessScope,
        KnowledgeEntry,
        KnowledgeStatus,
    )

    class ReadOnlyReviewStore(InMemoryKnowledgeStore):
        publish_entry_revision = None

    scope = KnowledgeAccessScope.privileged()
    store = ReadOnlyReviewStore(access_scope=scope)
    runtime = _app(knowledge_store=store, knowledge_access_scope=scope)
    with TestClient(create_server(runtime, config=ServerConfig.protected(_authenticate))) as client:
        proposal = KnowledgeEntry(
            id="unpublishable", text="Evidence", status=KnowledgeStatus.PENDING
        )
        client.portal.call(store.create_entry, proposal)
        [item] = _items(client, "priya")
        response = _settle(client, item["id"], "priya", decision="drop", reason="Reviewed")
        assert response.status_code == 501, response.text
        assert client.portal.call(store.get_entry, proposal.id) == proposal


@pytest.mark.parametrize("role,expected_status", [("reader", 403), ("owner", 200)])
def test_custom_role_policy_guards_inbox_argument_override(role, expected_status):
    from tests.server.test_server_inbox import _app, _items, _settle

    from cayu.server import ControlPlaneRoles

    class Auth:
        control_plane_roles = ControlPlaneRoles(
            role_claim="access", full_access=("owner",), read_and_decide=("reader",)
        )

        def __call__(self, request):
            context = _authenticate(request).model_copy(
                update={"claims": {"role": "admin", "access": role}}
            )
            self.control_plane_roles.require(request, context)
            return context

    runtime = _app()
    with TestClient(create_server(runtime, config=ServerConfig.protected(Auth()))) as client:
        _pause(client, runtime, "override-role", "approver")
        identity = client.get("/api/app", headers=_as("priya")).json()
        assert identity["full_access"] is (role == "owner")
        [item] = _items(client, "priya")
        response = _settle(
            client, item["id"], "priya", decision="approve", arguments={"note": "changed"}
        )
        assert response.status_code == expected_status, response.text
        if expected_status == 403:
            assert response.json()["detail"]["code"] == "role_forbidden"
            assert len(_items(client, "priya")) == 1


def _review_store(backend, tmp_path, request):
    from cayu.storage import (
        InMemoryKnowledgeStore,
        KnowledgeAccessScope,
        PostgresKnowledgeStore,
        SQLiteKnowledgeStore,
    )
    from cayu.storage.migrations import SchemaMode

    scope = KnowledgeAccessScope.privileged()
    if backend == "memory":
        return InMemoryKnowledgeStore(access_scope=scope), scope
    if backend == "sqlite":
        return SQLiteKnowledgeStore(tmp_path / "review-bounds.db", access_scope=scope), scope
    return PostgresKnowledgeStore(
        request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE, access_scope=scope
    ), scope


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("bound", ["count", "bytes", "utf8", "exact_bytes", "exact_count"])
def test_reviewed_drop_preserves_complete_material_or_refuses(backend, bound, tmp_path, request):
    from uuid import uuid4

    from tests.server.test_server_inbox import _app, _items, _settle

    from cayu.knowledge.activation_contracts import (
        MAX_KNOWLEDGE_ACTIVATION_CHUNKS as count_limit,
    )
    from cayu.knowledge.activation_contracts import (
        MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES as byte_limit,
    )
    from cayu.knowledge.records import KnowledgeChunk
    from cayu.storage import KnowledgeEntry, KnowledgeStatus

    entry_id = str(uuid4())
    store, scope = _review_store(backend, tmp_path, request)
    runtime = _app(knowledge_store=store, knowledge_access_scope=scope)
    count = count_limit + (bound == "count") if "count" in bound else 1
    text = "x" if "count" in bound else "x" * (byte_limit + (bound == "bytes"))
    if bound == "utf8":
        text = "x" * (byte_limit - 1) + "😀"
    chunks = [
        KnowledgeChunk(
            id=f"{entry_id}-chunk-{i}",
            entry_id=entry_id,
            entry_revision=1,
            chunk_index=i,
            text=text,
        )
        for i in range(count)
    ]
    with TestClient(create_server(runtime, config=ServerConfig.protected(_authenticate))) as client:

        async def seed():
            return await store.create_entry(
                KnowledgeEntry(id=entry_id, text="Source", status=KnowledgeStatus.PENDING), chunks
            )

        original = client.portal.call(seed)
        [item] = [i for i in _items(client, "priya") if i["knowledge"]["entry_id"] == entry_id]
        response = _settle(client, item["id"], "priya", decision="drop", reason="reviewed")
        if not bound.startswith("exact"):
            assert response.status_code == 409, response.text
            assert client.portal.call(store.get_entry, entry_id) == original
            from cayu.server._inbox_routes import _knowledge_operation_id, _KnowledgeLocator

            async def uncommitted():
                receipt = await store.load_entry_publication_receipt(
                    _knowledge_operation_id(_KnowledgeLocator(entry_id, 1), "drop")
                )
                successor = await store.get_entry(entry_id, revision=2)
                evidence = await store.read_evidence(entry_id, revision=1)
                return receipt, successor, evidence

            receipt, successor, evidence = client.portal.call(uncommitted)
            assert receipt is None and successor is None
            assert evidence is not None and not evidence.truncated and not evidence.evidence

        else:
            assert response.status_code == 200, response.text
            assert client.portal.call(store.get_entry, entry_id).revision == 2

        async def material():
            revision = 2 if bound.startswith("exact") else 1
            return await store.read_chunks(
                entry_id, revision=revision, max_chunks=count + 1, max_bytes=byte_limit * 2
            )

        result = client.portal.call(material)
        assert [(c.chunk_index, c.text) for c in result] == [
            (c.chunk_index, c.text) for c in chunks
        ]
        if not bound.startswith("exact"):
            assert (
                client.get(f"/api/inbox/{item['id']}", headers=_as("priya")).json()["settlement"]
                is None
            )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("changed", [False, True])
def test_reviewed_drop_typed_annotation_identity(
    backend, concurrent, changed, tmp_path, request, monkeypatch
):
    import asyncio
    from uuid import uuid4

    from cayu.knowledge.publication_contracts import KnowledgePublicationConflict
    from cayu.storage import KnowledgeEntry, KnowledgeStatus
    from cayu.storage.knowledge_review import KnowledgeReviewWorkflow

    entry_id = str(uuid4())
    store, scope = _review_store(backend, tmp_path, request)

    async def run():
        await store.create_entry(
            KnowledgeEntry(id=entry_id, text="source", status=KnowledgeStatus.PENDING)
        )
        workflow = KnowledgeReviewWorkflow(store, access_scope=scope)

        async def decide(value):
            return await workflow.reject_reviewed(
                entry_id,
                expected_revision=1,
                operation_id=entry_id,
                reviewer_identity="reviewer",
                reviewer_version="1",
                reason="reviewed",
                annotations={"nested": [value]},
            )

        value = 1 if changed else True
        if concurrent:
            original = store.publish_entry_revision
            both = asyncio.Event()
            arrivals = 0

            async def publish(*args, **kwargs):
                nonlocal arrivals
                arrivals += 1
                if arrivals == 2:
                    both.set()
                await both.wait()
                return await original(*args, **kwargs)

            monkeypatch.setattr(store, "publish_entry_revision", publish)
            results = await asyncio.gather(decide(True), decide(value), return_exceptions=True)
            assert sum(isinstance(r, KnowledgePublicationConflict) for r in results) == int(changed)
            assert sum(not isinstance(r, BaseException) for r in results) == 2 - int(changed)
        else:
            first = await decide(True)
            if backend == "sqlite":
                from cayu.storage import SQLiteKnowledgeStore

                await store.close()
                reopened = SQLiteKnowledgeStore(tmp_path / "review-bounds.db", access_scope=scope)
                workflow = KnowledgeReviewWorkflow(reopened, access_scope=scope)
            if changed:
                with pytest.raises(KnowledgePublicationConflict):
                    await decide(value)
            else:
                assert (await decide(value)).receipt.replayed
            assert (await workflow.get_entry(entry_id)).revision == first.entry.revision == 2
        close = getattr(workflow.store, "close", None)
        if close is not None:
            await close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("decision", ["approve", "deny"])
def test_approval_reason_survives_http_projection_and_replay(backend, decision, tmp_path, request):
    from tests.server.test_server_inbox import _app, _items, _settle, _wait_for_status

    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode
    from cayu.vaults.redaction import SecretRedactor

    runtime = (
        _app(
            session_store=PostgresSessionStore(
                request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
            )
        )
        if backend == "postgres"
        else _durable_app(backend, tmp_path)
    )
    runtime._secret_redactor = SecretRedactor("reason-canary")
    from uuid import uuid4

    session_id = str(uuid4())
    with TestClient(create_server(runtime, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, runtime, session_id, "approver")
        [item] = [i for i in _items(client, "priya") if i["session_id"] == session_id]
        body = {"decision": decision, "reason": "Reviewed reason-canary"}
        response = _settle(client, item["id"], "priya", **body)
        assert response.status_code == 200, response.text
        reason = response.json()["item"]["settlement"]["reason"]
        assert reason is not None and "Reviewed" in reason and "reason-canary" not in reason
        _wait_for_status(client, runtime, session_id)
        state = client.get(f"/api/inbox/{item['id']}", headers=_as("priya")).json()
        assert state["settlement"]["reason"] == reason
        replay = _settle(client, item["id"], "priya", **body)
        assert replay.status_code == 200 and replay.json()["replayed"] is True
        assert replay.json()["item"]["settlement"]["reason"] == reason
        changed = _settle(client, item["id"], "priya", decision=decision, reason="Changed")
        assert changed.status_code == 409
