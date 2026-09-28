"""Fresh-process host composition with explicitly restored fixture authority.

The controller retains this credential-free administrator configuration. It is
not a public receipt-to-authority mechanism or a production configuration format.
"""

import asyncio
import json
import sys
import time
import traceback
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4

from examples.collaboration import explicit_host as example

from cayu import (
    ContinuationService,
    HostContinuationRule,
    HostProducerDisclosure,
    HostProducerOutputRule,
    HostProducerSource,
    HostWaitRule,
    MandateAccessContext,
    MandateResolution,
    SessionExportAccessContext,
)
from cayu.collaboration import _host_producer_maintenance
from cayu.collaboration._contracts import ExactMatch, OwnerRef
from cayu.collaboration._producer_recovery import ProducerOutputRecovery
from cayu.collaboration._wait_discovery import WaitRecovery
from cayu.collaboration.exports import SessionExportReceipt
from cayu.collaboration.peer_content import PeerAppendKey
from cayu.messages import Message
from cayu.runtime._host_continuation_discovery import ContinuationRecovery
from cayu.runtime._session_continuation_owner import SessionContinuationOwner
from cayu.sessions import ResumeRequest, SessionStatus
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresBudgetLedger, PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


def capture_authority(team):
    return {
        "mandates": [
            [context.model_dump(mode="json"), resolution.model_dump(mode="json")]
            for context, resolution in team.mandates._resolutions.items()
        ],
        "sources": list(team.disclosure._sources),
        "grants": [
            [
                receipt.model_dump(mode="json"),
                digest,
                key.model_dump(mode="json"),
                producer,
                owner.model_dump(mode="json"),
            ]
            for receipt, digest, key, producer, owner in team.disclosure._grants.values()
        ],
    }


def restore_authority(team, authority):
    # This is the controller's original administrator catalogue, not a grant
    # derived from source receipts discovered in the stores after restart.
    team.mandates._resolutions = {
        MandateAccessContext.model_validate(context): MandateResolution.model_validate(resolution)
        for context, resolution in authority["mandates"]
    }
    team.disclosure._sources = {tuple(source) for source in authority["sources"]}
    for raw, digest, key, producer, owner in authority["grants"]:
        receipt = SessionExportReceipt.model_validate(raw)
        team.disclosure._grants[receipt.event_id] = (
            receipt,
            digest,
            PeerAppendKey.model_validate(key),
            producer,
            OwnerRef.model_validate(owner),
        )


async def main(material):
    address = material["address"]
    if material["backend"] == "sqlite":
        path = Path(address)
        collaboration = SQLiteCollaborationStore(path)
        sessions = SQLiteSessionStore(path.with_name("sessions.sqlite"))
        ledger = SQLiteBudgetLedger(path.with_name("ledger.sqlite"))
    else:
        collaboration = PostgresCollaborationStore(address, schema_mode=SchemaMode.CREATE)
        sessions = PostgresSessionStore(address, schema_mode=SchemaMode.CREATE)
        ledger = PostgresBudgetLedger(address, schema_mode=SchemaMode.CREATE)
    scope_uuid = material.setdefault("scope_uuid", uuid4().hex)
    now = material.setdefault("bootstrap_time_ns", time.time_ns())
    # Replay only the example bootstrap configuration. Runtime clocks remain
    # live: this replaces the example module's two configuration dependencies,
    # not time.time_ns or UUID generation in any production module.
    patches = ExitStack()
    patches.enter_context(patch.object(example, "uuid4", lambda: UUID(hex=scope_uuid)))
    patches.enter_context(patch.object(example, "time", SimpleNamespace(time_ns=lambda: now)))
    team = await example.create_demo_team(
        collaboration=collaboration, sessions=sessions, ledger=ledger
    )
    try:
        if "authority" not in material:
            original = example.service_until
            delivery = _host_producer_maintenance.deliver_producer_output
            admission = SessionContinuationOwner._admit_owned

            async def barrier():
                material["authority"] = capture_authority(team)
                material["provider_calls"] = len(team.transport.requests)
                print(json.dumps(material), flush=True)
                await asyncio.Event().wait()
                raise AssertionError("The process-loss barrier must not resume")

            async def observed_service(app, registration, inspect, ready, **options):
                if registration.continuation_rules:
                    rule = registration.continuation_rules[0]
                    material["continuation_rule"] = {
                        "expected": rule.expected.model_dump(mode="json"),
                        "request": rule.request.model_dump(mode="json", exclude_unset=True),
                        "service": rule.service.model_dump(mode="json"),
                    }
                if registration.producer_output_rules:
                    rule = registration.producer_output_rules[0]
                    wait = registration.wait_rules[0]
                    material["recovery"] = rule.recovery.model_dump(mode="json")
                    material["disclosure"] = rule.failure_context.model_dump(mode="json")
                    material["wait"] = wait.expected.model_dump(mode="json")
                    material["wait_context"] = wait.context.model_dump(mode="json")
                    if material["boundary"] == "before-delivery":
                        await barrier()
                return await original(app, registration, inspect, ready, **options)

            async def delivered_then_lost(*args, **kwargs):
                result = await delivery(*args, **kwargs)
                assert result.receipt is not None
                await barrier()
                return result

            async def admitted_then_lost(*args, **kwargs):
                result = await admission(*args, **kwargs)
                consumption = result[1].consumption
                assert consumption is not None and consumption.receipt_stage == "admitted"
                await barrier()
                return result

            patches.enter_context(patch.object(example, "service_until", observed_service))
            if material["boundary"] == "after-delivery":
                patches.enter_context(
                    patch.object(
                        _host_producer_maintenance, "deliver_producer_output", delivered_then_lost
                    )
                )
            if material["boundary"] == "after-continuation-admission":
                patches.enter_context(
                    patch.object(SessionContinuationOwner, "_admit_owned", admitted_then_lost)
                )
            await example.run_question(
                team,
                key="restart-one",
                sender=team.participants[0],
                recipient=team.participants[1],
                task="Recover the original finite question.",
                report=lambda stage: print(stage, file=sys.stderr, flush=True),
            )
            raise AssertionError("Producer process failed to reach its durable barrier")

        restore_authority(team, material["authority"])
        print("Restored administrator fixture configuration", file=sys.stderr, flush=True)
        app = team.application(request_key="restart-one")
        await app.initialize_collaboration()
        recovery = ProducerOutputRecovery.model_validate(material["recovery"])
        disclosure = SessionExportAccessContext.model_validate(material["disclosure"])
        wait = HostWaitRule(
            WaitRecovery.model_validate(material["wait"]),
            MandateAccessContext.model_validate(material["wait_context"]),
        )
        registered = await app.lookup_producer_registration(recovery, context=example.ACCESS)
        assert isinstance(registered, ExactMatch)
        command = registered.receipt
        destination = command.destinations[0]
        initial = await app.inspect_producer_output(recovery, context=example.ACCESS)
        assert isinstance(initial, ExactMatch) and initial.receipt.completion is not None
        admitted = material["boundary"] == "after-continuation-admission"
        assert (initial.receipt.cleanup_ack is not None) == admitted
        assert initial.receipt.destinations[0].delivery == (
            None if material["boundary"] == "before-delivery" else "appended"
        )
        original_completion = initial.receipt.completion

        async def continuation_status():
            record = await app.recover_session_continuation(continuation, context=example.ACCESS)
            session = await sessions.load(target.target_session_id)
            assert session is not None
            return record, session.status

        recovering_admission = material["boundary"] == "after-continuation-admission"
        expected_status = (
            SessionStatus.INTERRUPTED if recovering_admission else SessionStatus.COMPLETED
        )
        await example.service_until(
            app,
            example.host_registration(
                producer_sources=(
                    HostProducerSource(command.admission.prepared.recipient, example.ACCESS),
                ),
                producer_output_rules=(
                    HostProducerOutputRule(
                        recovery,
                        example.ACCESS,
                        disclosure,
                        (HostProducerDisclosure(destination.operation, disclosure),),
                    ),
                ),
                wait_rules=(wait,),
            ),
            lambda: app.inspect_producer_output(recovery, context=example.ACCESS),
            lambda found: isinstance(found, ExactMatch) and found.receipt.cleanup_ack is not None,
            timeout_s=example.JOURNEY_TIMEOUT_S,
        )
        assert not team.transport.requests  # No producer redispatch or model polling.
        print("Original producer delivered and settled", file=sys.stderr, flush=True)
        references, _ = await app.list_participant_sessions(
            destination.recipient, context=example.ACCESS
        )
        target = destination.attempt.append_key
        reference = next(item for item in references if item.session_id == target.target_session_id)
        assert reference.session_instance_id == target.target_session_instance_id
        continuation = (
            await app.list_session_continuations(reference, context=example.ACCESS)
        ).items[0]
        retained = await example.service_until(
            app,
            example.host_registration(wait_rules=(wait,)),
            lambda: app.recover_session_continuation(continuation, context=example.ACCESS),
            lambda found: found.latch is not None,
        )
        if admitted:
            selected = material["continuation_rule"]
            assert ContinuationRecovery.model_validate(selected["expected"]) == continuation
            resume = ResumeRequest.model_validate(selected["request"])
            service = ContinuationService.model_validate(selected["service"])
            # Surface reconstruction failure before a bounded host repeatedly
            # observes the legitimately blocked native recovery plan. This is
            # a read-only preflight, not a replacement claim or profile bypass.
            selected_session = await sessions.load(target.target_session_id)
            await app._session_engine._recovery_coordinator.preflight_incomplete_session(
                session=selected_session,
                inactive_for_seconds=1,
                participant_context=example.ACCESS,
            )
        else:
            resume = ResumeRequest(
                session_id=target.target_session_id,
                messages=[Message.text("user", "DEMO_RESUME: integrate the result.")],
            )
            service = ContinuationService(
                ticket=retained.ticket,
                latch=retained.latch,
                continuation_id="integrate",
                mode="inline",
                accepted_at=datetime.now(UTC).isoformat(),
            )
        await example.service_until(
            app,
            example.host_registration(
                continuation_rules=(
                    HostContinuationRule(
                        continuation,
                        resume,
                        service,
                        example.ACCESS,
                        recovery_inactive_for_seconds=1 if recovering_admission else None,
                    ),
                )
            ),
            continuation_status,
            lambda found: found[0].ticket.state == "CONSUMED" and found[1] is expected_status,
        )
        final = await app.inspect_producer_output(recovery, context=example.ACCESS)
        assert isinstance(final, ExactMatch)
        assert final.receipt.completion == original_completion
        assert final.receipt.cleanup_ack is not None
        assert final.receipt.destinations[0].delivery == "appended"
        finished = await sessions.load(target.target_session_id)
        assert finished is not None and finished.status is expected_status
        # Native abandoned-work recovery settles uncertainty without pretending
        # that the lost invocation completed or replaying a possible dispatch.
        expected_calls = 0 if recovering_admission else 1
        assert len(team.transport.requests) == expected_calls
        print(
            json.dumps(
                {
                    "provider_calls": expected_calls,
                    "continuation_status": expected_status.value,
                    "producer_redispatched": False,
                    "continuation": "consumed",
                    "delivery": "appended",
                    "cleanup": "settled",
                }
            ),
            flush=True,
        )
    except BaseException as error:
        host = getattr(error, "host", None) or getattr(error, "collaboration_host", None)
        if host is not None:
            # Credential-free test-fixture diagnostics: retain the real owned
            # failure instead of reporting only the application's summary.
            for failure in host._source_errors.values():
                traceback.print_exception(failure, file=sys.stderr)
            for outcome in host._owned.inspect().completed:
                if outcome.error is not None:
                    traceback.print_exception(outcome.error, file=sys.stderr)
        raise
    finally:
        await collaboration.close()
        await sessions.close()
        await ledger.close()
        patches.close()


if __name__ == "__main__":
    asyncio.run(main(json.loads(sys.stdin.read())))
