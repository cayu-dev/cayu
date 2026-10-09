"""Real pre-ticket process loss and bounded native recovery-proof retention."""

import asyncio
import json
import signal
import sys
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import AgentSpec, CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.external_waits import ExternalEventWaits
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions import _invocation_lifecycle
from cayu.sessions.external_waits import ExternalEventDelivery, ExternalWaitUnavailable
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("resume_existing", [False, True])
def test_pre_ticket_process_loss_preserves_bounded_cleanup_proof(
    backend, resume_existing, tmp_path, request, monkeypatch
):
    monkeypatch.setattr(_invocation_lifecycle, "INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS", 4)

    async def scenario():
        clock = [datetime.now(UTC)]
        async with stores(backend, tmp_path, request, clock) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            session_id = "pre-ticket-retention-" + uuid4().hex
            setup = {
                "backend": backend,
                "database": str(tmp_path / "waits.sqlite")
                if backend == "sqlite"
                else request.getfixturevalue("postgres_dsn"),
                "session_id": session_id,
                "registration": registered.model_dump_json(),
                "boundary": "before_ticket",
                "resume_existing": resume_existing,
                "receipt_limit": 4,
            }

            async def kill_at_committed_boundary(expected_requests):
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "tests.recovery.external_wait_worker",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    process.stdin.write((json.dumps(setup) + "\n").encode())
                    await process.stdin.drain()
                    ready = await asyncio.wait_for(process.stdout.readline(), 40)
                    if not ready:
                        _, diagnostic = await asyncio.wait_for(process.communicate(), 10)
                        pytest.fail("Worker missed its boundary:\n" + diagnostic.decode())
                    assert json.loads(ready) == {"ready": True, "requests": expected_requests}
                    process.send_signal(signal.SIGKILL)
                    assert await asyncio.wait_for(process.wait(), 10) == -signal.SIGKILL
                finally:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                # Let the real execution and recovery leases expire; never
                # rewrite a durable claim to force the next takeover.
                await asyncio.sleep(2.1)
                clock[0] = datetime.now(UTC)

            async def ledger(native):
                checkpoint = await runtime_checkpoint_session_store(native).load_checkpoint(
                    session_id
                )
                assert not checkpoint.get("session_continuations")
                return _invocation_lifecycle._invocation_lifecycle_receipt_ledger_from_checkpoint(
                    checkpoint
                )

            await kill_at_committed_boundary(int(resume_existing))
            initial = await ledger(store)
            admitted = next(item for item in initial.receipts if item.external_execution_origin)
            origin = admitted.external_execution_origin
            assert origin.admission_kind == ("admit" if resume_existing else "create")
            assert not any(
                item.kind.value == "release" and item.external_execution_origin
                for item in initial.receipts
            )
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="done", payload_json='{"done":true}'
                ),
                context=CONTEXT,
            )
            elected = await waits.inspect(correlation, context=CONTEXT)
            setup["boundary"] = "recovery_admission"
            for _ in range(5):
                before = (await store.load(session_id)).run_epoch
                await kill_at_committed_boundary(0)
                assert (await store.load(session_id)).run_epoch > before
                current = await ledger(store)
                assert len(current.receipts) <= 4
                assert all(item.external_execution_origin == origin for item in current.receipts)
            assert admitted.command_identity not in {
                item.command_identity for item in current.receipts
            }, "The original admission must actually be compacted."
            await waits.aclose()

            native = reopen()
            restored_waits = ExternalEventWaits(store=native, access_policy=Policy())
            app = CayuApp(session_store=native, enable_logging=False)
            provider = ScriptedModelProvider([])
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            adapter = SessionExternalWaitAdapter(app, restored_waits)
            with pytest.raises(ExternalWaitUnavailable, match="native writer release"):
                await adapter.exclude_prepared_execution(registered, context=CONTEXT)
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id, inactive_for_seconds=0)
            )
            assert not provider.requests
            # Lost acknowledgement: commit exclusion, discard its result, and
            # reconstruct both owners before asking for the exact same operation.
            await adapter.exclude_prepared_execution(registered, context=CONTEXT)
            await app.aclose()
            await restored_waits.aclose()
            native = reopen()
            restored_waits = ExternalEventWaits(store=native, access_policy=Policy())
            app = CayuApp(session_store=native, enable_logging=False)
            adapter = SessionExternalWaitAdapter(app, restored_waits)
            terminal = await adapter.exclude_prepared_execution(registered, context=CONTEXT)
            assert terminal.execution_excluded and not terminal.pending_handoff
            assert terminal.outcome == elected.outcome
            await native.delete_session(session_id)
            assert await adapter.exclude_prepared_execution(registered, context=CONTEXT) == terminal
            await app.aclose()
            await restored_waits.aclose()

    asyncio.run(scenario())
