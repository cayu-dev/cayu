"""Real process loss after native output, before source acknowledgement."""

import asyncio
import json
import signal
import sys

import pytest

from cayu.collaboration._producer_contracts import ProducerCompletionRecord, ProducerNativeOutput


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_completed_producer_accounting_reopens_without_new_authority(
    backend, tmp_path, request
):
    from cayu.applications import CayuApp
    from cayu.collaboration._producer_contracts import ProducerOutputRegistration
    from cayu.runtime._producer_budget import ProducerBudgetSettlement
    from cayu.storage.budget_ledger import SQLiteBudgetLedger
    from cayu.storage.postgres import PostgresBudgetLedger, PostgresSessionStore
    from cayu.storage.sqlite import SQLiteSessionStore

    address = (
        str(tmp_path / "collaboration.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.recovery.producer_output_process_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        output, error = await asyncio.wait_for(
            child.communicate(
                json.dumps(
                    {
                        "backend": backend,
                        "address": address,
                        "crash": False,
                    }
                ).encode()
            ),
            90,
        )
        assert child.returncode == 0, error.decode()
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
    result = json.loads(output)
    assert result["provider_calls"] == 1
    command = ProducerOutputRegistration.model_validate(result["expected"])
    expected = ProducerBudgetSettlement.model_validate(result["accounting"])
    assert expected.reservation_count > 0
    assert "private-process-loss-canary" not in output.decode()
    sessions = (
        SQLiteSessionStore(tmp_path / "sessions.sqlite")
        if backend == "sqlite"
        else PostgresSessionStore(address)
    )
    ledger = (
        SQLiteBudgetLedger(tmp_path / "ledger.sqlite")
        if backend == "sqlite"
        else PostgresBudgetLedger(address)
    )
    try:
        # No provider or binding resolver is registered in the replacement app.
        app = CayuApp(session_store=sessions, budget_ledger=ledger, enable_logging=False)
        assert await app._run_limit_controller._read_producer_budget_settlement(command) == expected
        assert await app._run_limit_controller._read_producer_budget_settlement(command) == expected
    finally:
        await sessions.close()
        await ledger.close()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("planned", [False, True])
async def test_sigkill_after_native_production_recovers_exact_output(
    backend, tmp_path, request, planned
):
    address = (
        str(tmp_path / "collaboration.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    material = {"backend": backend, "address": address, "planned": planned}
    producer = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.recovery.producer_output_process_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=256 * 1024,
    )
    try:
        producer.stdin.write(json.dumps(material).encode())
        await producer.stdin.drain()
        producer.stdin.close()
        # Planned setup includes creation/preparation and exact plan replay before
        # native dispatch. This observes a durable crash barrier, not a runtime
        # deadline; retain a finite bound for the longer composed journey.
        committed = await asyncio.wait_for(producer.stdout.readline(), 240 if planned else 90)
        if not committed:
            _, error = await asyncio.wait_for(producer.communicate(), 10)
            pytest.fail(error.decode())
        snapshot = json.loads(committed)
        assert snapshot["provider_calls"] == 1
        expected = ProducerNativeOutput.model_validate(snapshot["output"])
        assert expected.disposition == "answer"
        assert "private-process-loss-canary" not in committed.decode()
        producer.kill()
        await asyncio.wait_for(producer.wait(), 10)
        assert producer.returncode == -signal.SIGKILL
    finally:
        if producer.returncode is None:
            producer.kill()
            await producer.wait()

    # No provider is registered in this independent process. Recovery can only
    # read the pinned native receipt and repair collaboration completion.
    material["expected"] = snapshot["expected"]
    first = None
    for _ in range(2):
        reader = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tests.recovery.producer_completion_reader_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            output, error = await asyncio.wait_for(
                reader.communicate(json.dumps(material).encode()), 60
            )
            assert reader.returncode == 0, error.decode()
            recovered = ProducerCompletionRecord.model_validate_json(output)
            assert recovered.output == expected
            if first is not None:
                assert recovered == first
            first = recovered
        finally:
            if reader.returncode is None:
                reader.kill()
                await reader.wait()

    from cayu.budgets.base import BudgetReservationRecord
    from cayu.collaboration.prepared_admission import prepared_budget
    from cayu.runtime._producer_output_store import ROOT_KEY, NativeProducerIndex
    from cayu.sessions.base import SessionRunFenced, _invocation_lifecycle_authority_read_scope
    from cayu.storage.budget_ledger import SQLiteBudgetLedger
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresBudgetLedger, PostgresSessionStore
    from cayu.storage.sqlite import SQLiteSessionStore

    if backend == "sqlite":
        ledger = SQLiteBudgetLedger(tmp_path / "ledger.sqlite")
        sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    else:
        ledger = PostgresBudgetLedger(address, schema_mode=SchemaMode.CREATE)
        sessions = PostgresSessionStore(address, schema_mode=SchemaMode.CREATE)
    try:
        original_binding = prepared_budget(
            expected.registration.admission.prepared.budget_binding_json
        )
        await ledger._require_registered_budget_binding(
            binding_id=original_binding.binding_id,
            authority_digest=original_binding.authority_digest,
            allowance=original_binding.allowance,
        )
        assert snapshot["reservations"]
        for raw in snapshot["reservations"]:
            original = BudgetReservationRecord.model_validate(raw)
            assert await ledger.load_reservation(original.reservation_id) == original
        # Reconstruct the inventory from the ledger, not the worker's in-memory
        # handles or a successful reservation-event publication.
        after = None
        inventory = {}
        expected_ids = {raw["reservation_id"] for raw in snapshot["reservations"]}
        session_id = expected.registration.admission.prepared.target.session_id
        while page := await ledger._scan_reservation_records(session_id=session_id, after=after):
            inventory.update(
                (row.reservation_id, row) for row in page if row.reservation_id in expected_ids
            )
            after = page[-1].reservation_id
        assert set(inventory) == expected_ids
        for raw in snapshot["reservations"]:
            assert inventory[raw["reservation_id"]] == BudgetReservationRecord.model_validate(raw)
        # Output acceptance is not native cleanup. Recovery must neither erase
        # the producer pin nor change its original durable budget accounting.
        session_id = expected.registration.admission.prepared.target.session_id
        with _invocation_lifecycle_authority_read_scope():
            checkpoint = await sessions.load_checkpoint(session_id)
        index = NativeProducerIndex.model_validate(checkpoint[ROOT_KEY])
        assert index.state == "admitted" and index.cleanup_commitment is None
        assert index.output_commitment is not None
        assert (await sessions.load(session_id)).status == "running"
        with pytest.raises(SessionRunFenced):
            await sessions._read_native_producer_release(expected.registration)
        with pytest.raises(ValueError, match="Cannot delete a session while it is running"):
            await sessions.delete_session(session_id)
    finally:
        await sessions.close()
        await ledger.close()
