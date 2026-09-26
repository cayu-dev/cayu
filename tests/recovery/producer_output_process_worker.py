"""Run genuine production and stop on either side of native output retention."""

import asyncio
import json
import sys
from pathlib import Path

from tests.core.test_budget_binding import _binding, _limit
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_producer_output_contracts import output_scenario

from cayu.providers.base import ModelStreamEvent
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresBudgetLedger, PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


class _PauseAfterOutput:
    _producer_attachment_version = 1

    async def _retain_native_producer_output(self, session_id, *, invocation, stage_id):
        if self.before_output:
            observer = self.observer()
            try:
                stage = await observer.load_model_completion_stage(session_id, stage_id)
                assert stage.state == "completed" and stage.publication is not None
                publication = await observer.load_runtime_publication_receipt(
                    session_id, stage.publication.publication_id
                )
                assert publication is not None
                assert await observer._read_retained_native_producer_output(self.proposal) is None
                reservations = []
                after = None
                while page := await self.ledger._scan_reservation_records(
                    session_id=session_id, after=after
                ):
                    reservations.extend(record.model_dump(mode="json") for record in page)
                    assert len(reservations) <= 8
                    after = page[-1].reservation_id
            finally:
                await observer.close()
            print(
                json.dumps(
                    {
                        "expected": self.proposal.model_dump(mode="json"),
                        "source_run_epoch": publication.source_run_epoch,
                        "interaction_id": publication.interaction_id,
                        "provider_calls": len(self.provider.requests),
                        "reservations": reservations,
                    }
                ),
                flush=True,
            )
            await asyncio.Event().wait()
            raise AssertionError("The pre-retention process-loss barrier must not resume")
        output = await super()._retain_native_producer_output(
            session_id, invocation=invocation, stage_id=stage_id
        )
        if output is None:
            return output
        # A second connection must observe the record before notifying the
        # controller. No terminal release or collaboration completion runs yet.
        observer = self.observer()
        try:
            retained = await observer._read_retained_native_producer_output(self.proposal)
            assert retained == output
            stage = await observer.load_model_completion_stage(session_id, stage_id)
            assert stage.reservation_ids
            reservations = [
                (await self.ledger.load_reservation(identifier)).model_dump(mode="json")
                for identifier in stage.reservation_ids
            ]
        finally:
            await observer.close()
        print(
            json.dumps(
                {
                    "expected": self.proposal.model_dump(mode="json"),
                    "output": retained.model_dump(mode="json"),
                    "provider_calls": len(self.provider.requests),
                    "reservations": reservations,
                }
            ),
            flush=True,
        )
        await asyncio.Event().wait()
        raise AssertionError("The process-loss barrier must not resume")


class CrashSQLite(_PauseAfterOutput, SQLiteSessionStore):
    invocation_lifecycle_command_version = 1


class CrashPostgres(_PauseAfterOutput, PostgresSessionStore):
    invocation_lifecycle_command_version = 1


def binding(scope):
    root = "producer-root:" + scope
    return _binding(
        binding_id="producer-binding:" + scope,
        application_scope=scope,
        root_budget_id=root,
        limits=(_limit().model_copy(update={"key": root}),),
    )


async def main():
    material = json.loads(sys.stdin.read())
    from cayu import UserInputResponse
    from cayu.events import EventType
    from cayu.tools.user_input import UserInputTool

    human_gate = material.get("human_gate", False)
    policies = ()
    config = None
    if material.get("native_recovery", False):
        from tests.core import test_peer_content
        from tests.recovery.producer_recovery_fixture import (
            ProducerRecoveryPolicy,
            ProducerRecoveryProvider,
            producer_recovery_config,
        )

        test_peer_content.QualifiedPeerProvider = ProducerRecoveryProvider
        config = producer_recovery_config()
        if material.get("recovery_mode") in {"continue", "cancel"}:
            policies = (ProducerRecoveryPolicy(),)
    address = material["address"]
    crash = material.get("crash", True)
    cleanup_crash = material.get("cleanup_crash", False)
    if material["backend"] == "sqlite":
        path = Path(address)
        collaboration = SQLiteCollaborationStore(path)
        sessions = (CrashSQLite if crash else SQLiteSessionStore)(path.with_name("sessions.sqlite"))
        sessions.observer = lambda: SQLiteSessionStore(path.with_name("sessions.sqlite"))
        ledger = SQLiteBudgetLedger(path.with_name("ledger.sqlite"))
    else:
        collaboration = PostgresCollaborationStore(address, schema_mode=SchemaMode.CREATE)
        sessions = (CrashPostgres if crash else PostgresSessionStore)(
            address, schema_mode=SchemaMode.CREATE
        )
        sessions.observer = lambda: PostgresSessionStore(address, schema_mode=SchemaMode.CREATE)
        ledger = PostgresBudgetLedger(address, schema_mode=SchemaMode.CREATE)
    native_stores = collaboration, sessions, lambda: collaboration, (material["backend"], address)
    try:
        (
            application,
            resolver,
            _admission,
            provider,
            _,
            _,
            proposal,
            execution,
        ) = await output_scenario(
            native_stores,
            with_exports=True,
            budget_ledger=ledger,
            budget_binding_factory=binding,
            loop_policies=policies,
            config=config,
            tools=(UserInputTool(),) if human_gate else (),
            planned=material.get("planned", False),
        )
        # Process-loss qualification uses an explicit durable barrier, not
        # the default short foreground observation bound during fixture setup.
        application._request_coordinator._owners.observation_timeout = 60
        sessions.proposal, sessions.provider, sessions.ledger = proposal, provider, ledger
        sessions.before_output = material.get("before_output", False)
        provider._batches = (
            *(
                (
                    (
                        ModelStreamEvent.tool_call(
                            name="ask_user", id="human-gate", arguments={"question": "Continue?"}
                        ),
                        ModelStreamEvent.completed(),
                    ),
                )
                if human_gate
                else ()
            ),
            (
                ModelStreamEvent.thinking("private-process-loss-canary"),
                ModelStreamEvent.text_delta(
                    "a" * 1025 if cleanup_crash else "retained after process loss"
                ),
                ModelStreamEvent.completed(),
            ),
        )
        registration = await application.register_producer_output(
            proposal, execution, context=resolver.recipient.context
        )
        original = resolver.recipient.resolution
        actions = (*original.principal.actions, "execute", "publish")
        resolver.recipient.resolution = original.model_copy(
            update={
                "principal": original.principal.model_copy(update={"actions": actions}),
                "chain": original.chain.model_copy(
                    update={
                        "entries": tuple(
                            entry.model_copy(update={"actions": actions})
                            for entry in original.chain.entries
                        )
                    }
                ),
            }
        )
        events = [
            event
            async for event in application.execute_producer_output(
                proposal,
                execution,
                context=CONTEXT,
                producer_context=resolver.recipient.context,
            )
        ]
        if human_gate:
            question = next(
                event for event in events if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            async for _ in application.resolve_user_input(
                UserInputResponse(
                    session_id=proposal.admission.prepared.target.session_id,
                    input_id=question.payload["input_id"],
                    answer="yes",
                ),
                context=CONTEXT,
            ):
                pass
        if crash:
            raise AssertionError("Production did not reach its durable output barrier")
        receiver = application._request_coordinator._registration.receiving_owner
        if cleanup_crash:
            from cayu.collaboration.exports import SessionExportAccessContext

            await application.retain_producer_completion(proposal, context=CONTEXT)
            outcome = await application.publish_producer_outcome(
                proposal,
                context=SessionExportAccessContext(
                    principal=resolver.recipient.context.principal,
                    mandate=resolver.recipient.context,
                ),
            )
            assert outcome.command.outcome == "failed"
            complete = receiver._complete_producer_cleanup

            async def lose_native_ack(record, *, authority):
                native = await complete(record, authority=authority)
                observer = sessions.observer()
                try:
                    assert await observer._read_completed_native_producer_cleanup(record) == native
                    await observer.delete_session(native.session_id)
                    assert await observer._read_completed_native_producer_cleanup(record) == native
                finally:
                    await observer.close()
                print(
                    json.dumps(
                        {
                            "expected": proposal.model_dump(mode="json"),
                            "native": native.model_dump(mode="json"),
                            "provider_calls": len(provider.requests),
                        }
                    ),
                    flush=True,
                )
                await asyncio.Event().wait()
                raise AssertionError("Native acknowledgement loss barrier must not resume")

            receiver._complete_producer_cleanup = lose_native_ack
            await application.settle_producer_output(proposal, context=CONTEXT)
            raise AssertionError("Cleanup must remain at the native acknowledgement barrier")
        accounting = await receiver._read_producer_budget_settlement(registration)
        print(
            json.dumps(
                {
                    "expected": proposal.model_dump(mode="json"),
                    "accounting": accounting.model_dump(mode="json"),
                    "provider_calls": len(provider.requests),
                }
            ),
            flush=True,
        )
    finally:
        await collaboration.close()
        await sessions.close()
        await ledger.close()


if __name__ == "__main__":
    asyncio.run(main())
