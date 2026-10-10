"""Revision 115 preserves pre-existing continuation and lifecycle commitments."""

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.core._execution_profile_fixtures import (
    create_admitted_session,
    interrupt_and_release_test_invocation,
)
from tests.core.test_session_continuation import _context, _QualifiedReceiver, _ticket

from cayu import CayuApp
from cayu.agents import AgentSpec
from cayu.collaboration._capabilities import CapabilityDescriptor
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
from cayu.sessions._invocation_lifecycle import _invocation_lifecycle_receipt_ledger_from_checkpoint
from cayu.sessions._session_continuation import (
    ContinuationLatch,
    ContinuationRetirement,
    ContinuationService,
    ContinuationWait,
)
from cayu.sessions._session_continuation_store import ROOT_KEY, digest
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("state", ["waiting", "retired", "corrupt"])
def test_revision_115_upgrades_indexes_without_changing_receipt_commitments(
    backend, state, tmp_path, request
):
    async def scenario():
        database = (
            tmp_path / "upgrade.sqlite"
            if backend == "sqlite"
            else request.getfixturevalue("postgres_dsn")
        )

        def open_store(mode):
            cls = SQLiteSessionStore if backend == "sqlite" else PostgresSessionStore
            return cls(database, schema_mode=mode)

        def runtime(store):
            app = CayuApp(session_store=store, enable_logging=False)
            provider = ScriptedModelProvider([ModelStreamEvent.completed({})], name="provider")
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            return app, provider

        def receiving_owner(store, ticket):
            return SessionContinuationOwner(
                store=store,
                owner=ticket.owner,
                receiver=_QualifiedReceiver(),
                receiver_capability=CapabilityDescriptor(
                    owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                ),
                redactor=SecretRedactor(),
            )

        store = open_store(SchemaMode.CREATE)
        app, provider = runtime(store)
        session_id = "upgrade-" + uuid4().hex
        admitted = await create_admitted_session(
            store,
            app=app,
            request=RunRequest(
                agent_name="root",
                session_id=session_id,
                messages=[Message.text("user", "wait")],
            ),
            provider_name=provider.name,
            model="model",
        )
        ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
        owner = receiving_owner(store, ticket)
        invocation = await _context(store, session_id)
        await owner.prepare(
            ContinuationWait.model_validate(
                ticket.model_dump(include=set(ContinuationWait.model_fields))
            ),
            invocation=invocation,
        )
        native = await owner.park(ticket, invocation=invocation)
        if state == "retired":
            native = await owner.retire(
                ContinuationRetirement(
                    ticket=native.ticket,
                    control_id="retire",
                    reason="cancelled",
                    retired_at=datetime.now(UTC).isoformat(),
                ),
                invocation=invocation,
            )
        await interrupt_and_release_test_invocation(store, session_id)
        checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(session_id)
        ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
        assert all("external_execution_origin" not in item.model_dump() for item in ledger.receipts)
        material = native.model_dump(mode="json")
        assert "recovery_writer" not in material
        # Reconstruct exactly the version-2 shape, including its original hashes.
        # New optional fields must not silently alter the commitments below.
        assert native.events[-1].record_sha256 == digest(
            {key: value for key, value in material.items() if key != "events"}
        )
        checkpoint[ROOT_KEY]["schema_version"] = 2
        for entry in checkpoint[ROOT_KEY]["entries"]:
            del entry["purpose"]
        if state == "corrupt":
            checkpoint[ROOT_KEY]["entries"][0]["record_sha256"] = "f" * 64
        await owner.drain()
        await app.aclose()
        await store.close()

        def install_legacy():
            if backend == "sqlite":
                with sqlite3.connect(database) as conn:
                    conn.execute(
                        "UPDATE cayu_checkpoints SET state_json=? WHERE session_id=?",
                        (json.dumps(checkpoint), session_id),
                    )
                    conn.execute("DELETE FROM cayu_schema_migrations WHERE revision>=115")
                    conn.execute("PRAGMA user_version=114")
            else:
                import psycopg
                from psycopg.types.json import Jsonb

                with psycopg.connect(database) as conn:
                    conn.execute(
                        "UPDATE cayu_checkpoints SET state=%s WHERE session_id=%s",
                        (Jsonb(checkpoint), session_id),
                    )
                    conn.execute("DELETE FROM cayu_schema_migrations WHERE revision>=115")

        install_legacy()
        if state == "corrupt":
            broken = None
            try:
                with pytest.raises(ValueError, match="indexed receiving evidence") as caught:
                    broken = open_store(SchemaMode.MIGRATE)
                    await broken.load(session_id)
                assert f"session {session_id!r}" in str(caught.value)
            finally:
                if broken is not None:
                    await broken.close()
            if backend == "sqlite":
                with sqlite3.connect(database) as conn:
                    assert conn.execute("PRAGMA user_version").fetchone()[0] == 114
                    saved = json.loads(
                        conn.execute(
                            "SELECT state_json FROM cayu_checkpoints WHERE session_id=?",
                            (session_id,),
                        ).fetchone()[0]
                    )
            else:
                import psycopg

                with psycopg.connect(database) as conn:
                    assert (
                        conn.execute("SELECT MAX(revision) FROM cayu_schema_migrations").fetchone()[
                            0
                        ]
                        == 114
                    )
                    saved = conn.execute(
                        "SELECT state FROM cayu_checkpoints WHERE session_id=%s", (session_id,)
                    ).fetchone()[0]
            assert saved == checkpoint
            return

        restored = open_store(SchemaMode.MIGRATE)
        app, provider = runtime(restored)
        owner = receiving_owner(restored, ticket)
        try:
            upgraded = await runtime_checkpoint_session_store(restored).load_checkpoint(session_id)
            assert upgraded[ROOT_KEY]["schema_version"] == 3
            assert upgraded[ROOT_KEY]["entries"][0]["purpose"] == ticket.purpose
            assert _invocation_lifecycle_receipt_ledger_from_checkpoint(upgraded) == ledger
            retained = await restored.load_continuation_ticket(
                session_id,
                registration_key=ticket.registration_key,
                session_instance_id=ticket.session_instance_id,
            )
            assert retained == native
            if state == "waiting":
                latch = ContinuationLatch(
                    ticket=ticket,
                    wait_receipt_digest="wait-receipt",
                    outcome_kind="success",
                    selected_manifest=(),
                    disclosure_digest="disclosure",
                    latch_key="latch",
                    outcome_digest="outcome",
                    accepted_at=datetime.now(UTC).isoformat(),
                )
                latched = await owner.latch(latch)
                service = ContinuationService(
                    ticket=latched.ticket,
                    latch=latched.latch,
                    continuation_id="resume",
                    mode="inline",
                    accepted_at=latch.accepted_at,
                )
                resume = ResumeRequest(session_id=session_id, messages=[Message.text("user", "go")])
                result = await owner.service(app, resume, service)
                assert result.ticket.state == "CONSUMED"
                assert await owner.service(app, resume, service) == result
                assert len(provider.requests) == 1
            await restored.delete_session(session_id)
            assert await restored.load(session_id) is None
        finally:
            await owner.drain()
            await app.aclose()
            await restored.close()

    asyncio.run(scenario())
