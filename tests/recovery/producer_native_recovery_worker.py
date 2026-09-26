"""Public native cleanup after process loss, with no new provider work allowed."""

import asyncio
import json
import sys
from pathlib import Path

from tests.core.test_collaboration_request_foundation import RequestResolver
from tests.core.test_participant_identity import CONTEXT, app, registration
from tests.recovery.producer_recovery_fixture import (
    ProducerRecoveryPolicy,
    ProducerRecoveryProvider,
    producer_recovery_config,
)
from tests.recovery.producer_recovery_lineage_checks import require_missing_rebind_refusal

from cayu import (
    AgentSpec,
    CollaborationAccessContext,
    EnqueueSessionMessageRequest,
    IncompleteSessionRecoveryRequest,
    ProducerOutputRegistration,
)
from cayu.collaboration.prepared_admission import prepared_budget
from cayu.collaboration.request_access import PreparedAdmissionRegistration, RequestRegistration
from cayu.collaboration.requests import RequestControl
from cayu.runtime.authority import SessionRunFenced
from cayu.runtime.session_message_lifecycle import SessionMessageQuery
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.postgres import PostgresBudgetLedger, PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.user_input import UserInputTool


async def main():
    material = json.loads(sys.stdin.read())
    command = ProducerOutputRegistration.model_validate(material["expected"])
    address = material["address"]
    if material["backend"] == "sqlite":
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
    application = app(
        collaboration,
        registration(
            scope=command.operation.application_scope,
            limits=command.admission.expected.intent.limits,
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
    continuing = material.get("recovery_mode") == "continue"
    cancelling = material.get("recovery_mode") == "cancel"
    queued = material.get("recovery_mode") == "queue"
    policy = (
        ProducerRecoveryPolicy(continue_requested=continuing, block=cancelling)
        if continuing or cancelling
        else None
    )
    application.register_agent(
        AgentSpec(name="reviewer", model="model", system_prompt="system"),
        loop_policies=() if policy is None else (policy,),
        tools=(UserInputTool(),) if material.get("human_gate", False) else (),
    )
    application._request_coordinator._owners.observation_timeout = 60
    try:
        initialized = await application.initialize_collaboration()
        session_id = command.admission.prepared.target.session_id
        before = await sessions.load(session_id)
        checkpoint = await sessions.load_checkpoint(session_id)
        try:
            await application.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id),
                context=CollaborationAccessContext(principal="foreign-operator"),
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("Foreign recovery authority was accepted")
        assert await sessions.load(session_id) == before
        assert await sessions.load_checkpoint(session_id) == checkpoint
        assert not provider.requests
        if material.get("before_output", False):
            try:
                await application.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                )
            except PermissionError:
                pass
            else:
                raise AssertionError("Completion replay accepted missing execution authority")
            assert await sessions.load(session_id) == before
            assert await sessions.load_checkpoint(session_id) == checkpoint
            assert not provider.requests
        if queued:
            await application.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key="not-part-of-producer-recovery",
                    content="This successor must retain its own execution admission.",
                    delivery_mode="on_idle",
                )
            )
            pending_queue = await sessions.inspect_session_messages(
                SessionMessageQuery(session_id=session_id)
            )
            assert len(pending_queue.records) == 1
            assert pending_queue.records[0].status.value == "queued"
        recovery = asyncio.create_task(
            application.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id), context=CONTEXT
            )
        )
        if cancelling:
            entered = asyncio.create_task(policy.entered.wait())
            done, _ = await asyncio.wait(
                (entered, recovery), timeout=60, return_when=asyncio.FIRST_COMPLETED
            )
            if recovery in done:
                await recovery
                raise AssertionError("Recovery bypassed the blocking policy")
            assert entered in done
            recovery.cancel("real caller cancellation during native completion replay")
            assert recovery.cancelling() == 1
            try:
                await asyncio.wait_for(asyncio.shield(recovery), 15)
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("Recovery swallowed caller cancellation")
            assert recovery.cancelled()
            assert recovery.cancelling() == 1
            assert (await sessions.load(session_id)).status.value == "running"
            try:
                await sessions._read_native_producer_release(command)
            except SessionRunFenced:
                pass
            else:
                raise AssertionError("Blocked recovery lost its invocation fence")
            competing = await application.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id), context=CONTEXT
            )
            assert "skipped_active" in competing.actions
            assert not provider.requests and policy.calls == 1
            policy.release.set()
            assert await application.drain_recovery_cleanups(timeout_s=30)
            await sessions._read_native_producer_release(command)
            outcome = None
        else:
            outcome = await recovery
        assert not provider.requests
        if queued:
            after_queue = await sessions.inspect_session_messages(
                SessionMessageQuery(session_id=session_id)
            )
            assert after_queue.records == pending_queue.records
        native = await sessions._read_native_producer_release(command)
        accounting = await application._run_limit_controller._read_producer_budget_settlement(
            command
        )
        assert accounting.native_release_commitment == native.release_commitment
        assert await sessions._read_native_producer_release(command) == native
        completion = await application.retain_producer_completion(command, context=CONTEXT)
        if continuing:
            assert policy.calls == 1
            assert completion.output.disposition == "stopped", completion.output.disposition
            assert (await sessions.load(session_id)).status.value == "interrupted"
            if outcome is not None:
                assert "interrupted_abandoned" in outcome.actions
        else:
            assert completion.output.disposition == "answer", completion.output.disposition
            assert (
                completion.output.source_indices and completion.output.source_commitment is not None
            )
            if cancelling:
                assert policy.calls == 1
        prior = await application.inspect_collaboration_request(
            command.admission.expected, context=resolver.context
        )
        await application.control_collaboration_request(
            RequestControl(
                operation=initialized.operation("close-recovered-producer"),
                expected=command.admission.expected,
                expected_revision=prior.revision,
                kind="cancel",
            ),
            context=resolver.context,
        )
        await require_missing_rebind_refusal(application, command)
        finalized = await application.settle_producer_output(command, context=CONTEXT)
        assert finalized.native_receipt.run_epoch == native.run_epoch
        assert await application.settle_producer_output(command, context=CONTEXT) == finalized
        await sessions.delete_session(session_id)
        # Native deletion is not the source's cleanup authority. Its exact ACK
        # remains replayable independently of session-owned rows.
        assert await application.settle_producer_output(command, context=CONTEXT) == finalized
        assert await application.retain_producer_completion(command, context=CONTEXT) == completion
        assert not provider.requests
        print(
            json.dumps(
                {
                    "recovery": None if outcome is None else outcome.model_dump(mode="json"),
                    "release": native.model_dump(mode="json"),
                    "accounting": accounting.model_dump(mode="json"),
                    "finalized": finalized.model_dump(mode="json"),
                    "provider_calls": len(provider.requests),
                }
            )
        )
    finally:
        if policy is not None:
            policy.release.set()
        await application.drain_collaboration_requests()
        await collaboration.close()
        await sessions.close()
        await ledger.close()


if __name__ == "__main__":
    asyncio.run(main())
