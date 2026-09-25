"""Native durable planning indices must agree with authenticated public readback."""

import asyncio
import json
import sys

import pytest
from tests.core.test_request_planning_public import complete_plan, scenario

from cayu.collaboration._contracts import ExactMatch, ExactUnavailable
from cayu.collaboration._planning_records import RequestPlanningRecord
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_fresh_process_loss_after_native_creation_recovers_original_child(
    backend, tmp_path, request
):
    from tests.core.test_participant_identity import CONTEXT
    from tests.core.test_request_planning_fresh import scenario as fresh_scenario

    from cayu.storage.postgres import PostgresSessionStore
    from cayu.storage.sqlite import SQLiteSessionStore

    address = (
        str(tmp_path / "fresh-planning.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    session_address = str(tmp_path / "fresh-sessions.sqlite") if backend == "sqlite" else address
    store = (
        SQLiteCollaborationStore(address)
        if backend == "sqlite"
        else PostgresCollaborationStore(address, schema_mode=SchemaMode.CREATE)
    )
    sessions = (
        SQLiteSessionStore(session_address)
        if backend == "sqlite"
        else PostgresSessionStore(address, schema_mode=SchemaMode.CREATE)
    )
    try:
        application, resolver, command, provider, _, creation, preparation = await fresh_scenario(
            (store, sessions, lambda: store, (backend, address))
        )
        policy = application._request_coordinator._registration.planning_policies[0]

        async def run(mode):
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "tests.recovery.request_planning_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    child.communicate(
                        json.dumps(
                            {
                                "backend": backend,
                                "address": address,
                                "session_address": session_address,
                                "mode": mode,
                                "expected": command.model_dump(mode="json"),
                                "policy": policy.model_dump(mode="json"),
                            }
                        ).encode()
                    ),
                    240,
                )
                assert child.returncode == (19 if mode == "crash_after_creation_commit" else 0), (
                    stderr.decode()
                )
                return stdout
            finally:
                if child.returncode is None:
                    child.kill()
                    await child.wait()

        await run("crash_after_creation_commit")
        receiving = await sessions.read_session_creation_decision(preparation.creation)
        assert isinstance(receiving, ExactMatch)
        assert receiving.receipt.state == "created"
        assert not receiving.receipt.settlement_acknowledged
        retained = await application.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(retained, ExactMatch)
        assert retained.receipt.state == "preparing" and retained.receipt.pending_stages == 1
        restored = RequestPlanningRecord.model_validate_json(await run("recover_fresh"))
        assert restored.state == "admitted" and restored.pending_stages == 0
        child = await application.lookup_recipient_session(creation, context=CONTEXT)
        assert child is not None and child[0].instance_id == receiving.receipt.session_instance_id
        final = await sessions.read_session_creation_decision(preparation.creation)
        assert isinstance(final, ExactMatch) and final.receipt.settlement_acknowledged
        assert final.receipt.session_instance_id == receiving.receipt.session_instance_id
        found = await application.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(found, ExactMatch) and found.receipt == restored
        assert provider.requests == []
    finally:
        await store.close()
        await sessions.close()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_process_loss_after_intent_commit_recovers_exact_frozen_policy(
    backend, tmp_path, request
):
    address = (
        str(tmp_path / "planning-process.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    store = (
        SQLiteCollaborationStore(address)
        if backend == "sqlite"
        else PostgresCollaborationStore(address, schema_mode=SchemaMode.CREATE)
    )
    try:
        application, resolver, command, policy, provider = await scenario(store, "decline")

        async def run(mode):
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "tests.recovery.request_planning_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    child.communicate(
                        json.dumps(
                            {
                                "backend": backend,
                                "address": address,
                                "mode": mode,
                                "expected": command.model_dump(mode="json"),
                                "policy": policy.model_dump(mode="json")
                                if mode == "crash_after_intent"
                                else None,
                            }
                        ).encode()
                    ),
                    120,
                )
                assert child.returncode == (19 if mode == "crash_after_intent" else 0), (
                    stderr.decode()
                )
                return stdout
            finally:
                if child.returncode is None:
                    child.kill()
                    await child.wait()

        await run("crash_after_intent")
        retained = await application.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(retained, ExactMatch) and retained.receipt.state == "evaluating"
        reconstructed = RequestPlanningRecord.model_validate_json(await run("recover"))
        assert reconstructed.state == "declined"
        assert reconstructed.receipt == retained.receipt.receipt
        found = await application.lookup_collaboration_plan(
            command, context=resolver.recipient.context
        )
        assert isinstance(found, ExactMatch) and found.receipt == reconstructed
        assert provider.requests == []
    finally:
        await store.close()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_public_plan_readback_rejects_every_divergent_native_index(
    backend, tmp_path, request
):
    store = (
        SQLiteCollaborationStore(tmp_path / "planning-indices.sqlite")
        if backend == "sqlite"
        else PostgresCollaborationStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    )
    try:
        application, resolver, command, _, _ = await scenario(store, "defer")
        record = await complete_plan(application, command, resolver.recipient.context)
        operation = command.operation
        selection = command.expected.intent.selection
        for field, original, corrupt in (
            ("request_id", selection.reference.request_id, "wrong-request"),
            ("request_incarnation", selection.reference.incarnation, "wrong-incarnation"),
            ("participant_id", selection.recipient.reference.participant_id, "wrong-participant"),
            ("planning_generation", command.planning_generation, 2),
            ("state", record.state, "cancelled"),
            ("pending_stages", record.pending_stages, 1),
            ("next_due_at_ms", record.next_due_at_ms, record.next_due_at_ms + 1),
        ):

            async def write_index(value, *, column=field):
                async with store._transaction(operation.application_scope, write=True) as tx:
                    # Test-owned identifiers from the fixed tuple above, never input.
                    await tx._execute(
                        f"UPDATE cayu_collaboration_request_plans SET {column}=? "
                        "WHERE scope=? AND namespace=? AND generation=? AND caller_key=?",
                        (
                            value,
                            operation.application_scope,
                            operation.namespace_incarnation,
                            operation.generation,
                            operation.caller_key,
                        ),
                    )

            await write_index(corrupt)
            try:
                found = await application.lookup_collaboration_plan(
                    command, context=resolver.recipient.context
                )
                assert isinstance(found, ExactUnavailable), field
            finally:
                await write_index(original)
            restored = await application.lookup_collaboration_plan(
                command, context=resolver.recipient.context
            )
            assert isinstance(restored, ExactMatch) and restored.receipt == record
    finally:
        await store.close()
