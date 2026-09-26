"""Process-loss cleanup does not require restoring producer execution access."""

import asyncio
import json
import signal
import sys
from pathlib import Path

import pytest
from tests.core.test_collaboration_request_foundation import RequestResolver
from tests.core.test_participant_identity import CONTEXT, Policy, app, registration
from tests.recovery.producer_recovery_fixture import (
    ProducerRecoveryProvider,
    producer_recovery_config,
)

from cayu import (
    AgentSpec,
    IncompleteSessionRecoveryRequest,
    InterruptSessionRequest,
    ProducerOutputRegistration,
)
from cayu.collaboration.prepared_admission import prepared_budget
from cayu.collaboration.request_access import PreparedAdmissionRegistration, RequestRegistration
from cayu.collaboration.requests import RequestControl
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.postgres import PostgresBudgetLedger, PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_native_process_loss_can_be_interrupted_after_execution_revocation(
    backend, tmp_path, request
):
    address = (
        str(tmp_path / "collaboration.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
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
        producer.stdin.write(
            json.dumps(
                {
                    "backend": backend,
                    "address": address,
                    "native_recovery": True,
                    "before_output": True,
                }
            ).encode()
        )
        await producer.stdin.drain()
        producer.stdin.close()
        committed = await asyncio.wait_for(producer.stdout.readline(), 90)
        if not committed:
            _, error = await asyncio.wait_for(producer.communicate(), 10)
            pytest.fail(error.decode())
        snapshot = json.loads(committed)
        assert snapshot["provider_calls"] == 1
        producer.kill()
        await asyncio.wait_for(producer.wait(), 10)
        assert producer.returncode == -signal.SIGKILL
    finally:
        if producer.returncode is None:
            producer.kill()
            await producer.wait()

    command = ProducerOutputRegistration.model_validate(snapshot["expected"])
    if backend == "sqlite":
        collaboration = SQLiteCollaborationStore(address)
        sessions = SQLiteSessionStore(Path(address).with_name("sessions.sqlite"))
        ledger = SQLiteBudgetLedger(Path(address).with_name("ledger.sqlite"))
    else:
        collaboration = PostgresCollaborationStore(address)
        sessions = PostgresSessionStore(address)
        ledger = PostgresBudgetLedger(address)
    binding = prepared_budget(command.admission.prepared.budget_binding_json)

    class BudgetReceiver:
        async def resolve_budget_binding(self, *, request):
            return binding

    resolver = RequestResolver(command.admission.expected.intent.request)
    policy = Policy()
    application = app(
        collaboration,
        registration(
            scope=command.operation.application_scope,
            limits=command.admission.expected.intent.limits,
            policy=policy,
        ),
        session_store=sessions,
        config=producer_recovery_config(),
        budget_ledger=ledger,
        budget_binding_receiver=BudgetReceiver(),
        enable_common_root_budget_binding=True,
        collaboration_requests=RequestRegistration(
            mandates=resolver,
            prepared_admission=PreparedAdmissionRegistration(receiver=command.receiver),
            max_ttl_ms=300_000,
        ),
    )
    provider = ProducerRecoveryProvider([], name="provider")
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    application._request_coordinator._owners.observation_timeout = 60
    try:
        initialized = await application.initialize_collaboration()
        session_id = command.admission.prepared.target.session_id
        # Native continuation uses the participant's administration grant.
        policy.denied.add("administration")
        before = await sessions.load(session_id)
        checkpoint = await sessions.load_checkpoint(session_id)
        with pytest.raises(PermissionError):
            await application.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id), context=CONTEXT
            )
        assert await sessions.load(session_id) == before
        assert await sessions.load_checkpoint(session_id) == checkpoint
        assert not provider.requests

        # Explicit operator cleanup is not renewed answer-production authority.
        try:
            async for _ in application.interrupt_session(
                InterruptSessionRequest(session_id=session_id)
            ):
                pass
        except TimeoutError:
            # An offline RUNNING owner first receives a durable interruption;
            # its missing process cannot acknowledge the bounded live-owner wait.
            assert (await sessions.load(session_id)).status.value == "interrupting"
            assert not provider.requests
            await application.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
        assert await application.drain_recovery_cleanups(timeout_s=30)
        native = await sessions._read_native_producer_release(command)
        assert native.interaction_id == snapshot["interaction_id"]
        completion = await application.retain_producer_completion(command, context=CONTEXT)
        assert completion.output.disposition == "stopped"
        prior = await application.inspect_collaboration_request(
            command.admission.expected, context=resolver.context
        )
        await application.control_collaboration_request(
            RequestControl(
                operation=initialized.operation("close-revoked-producer"),
                expected=command.admission.expected,
                expected_revision=prior.revision,
                kind="cancel",
            ),
            context=resolver.context,
        )
        final = await application.settle_producer_output(command, context=CONTEXT)
        assert final.delivery == "excluded"
        await sessions.delete_session(session_id)
        assert await application.settle_producer_output(command, context=CONTEXT) == final
        assert not provider.requests
        assert "administration" in policy.denied
    finally:
        await application.drain_recovery_cleanups(timeout_s=30)
        await application.drain_collaboration_requests()
        await collaboration.close()
        await sessions.close()
        await ledger.close()
