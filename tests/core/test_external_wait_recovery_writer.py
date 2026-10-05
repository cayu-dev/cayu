"""Native succession characterization; public recovery qualification is separate."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core._execution_profile_fixtures import (
    create_admitted_session,
    interrupt_and_release_test_invocation,
    runtime_checkpoint_session_store,
)
from tests.core.test_session_continuation import _context, _QualifiedReceiver, _ticket
from tests.external_wait_support import stores

from cayu.collaboration._capabilities import CapabilityDescriptor
from cayu.messages import Message
from cayu.runtime._invocation_lifecycle import prepare_rebind_invocation_command
from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
from cayu.sessions._execution_profile_checkpoint import (
    checkpoint_with_active_invocation_execution_profile,
)
from cayu.sessions._session_continuation import (
    ContinuationConflict,
    ContinuationReleasedRetirement,
    ContinuationRetirement,
    ContinuationWait,
    continuation_writer_frontier,
)
from cayu.sessions.base import _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY, RunRequest
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_native_recovery_writer_keeps_origin_and_reconstructs(backend, tmp_path, request):
    async def scenario():
        now = datetime.now(UTC)
        async with stores(backend, tmp_path, request, [now]) as (store, reopen):
            admitted = await create_admitted_session(
                store,
                request=RunRequest(
                    agent_name="assistant",
                    session_id="recover-writer",
                    messages=[Message.text("user", "wait")],
                ),
                provider_name="continuation-provider",
                model="continuation-model",
            )
            session = admitted.session
            ticket = _ticket(session, admitted.active_invocation_profile.interaction_id).model_copy(
                update={"purpose": "external-event-v1"}
            )

            def owner(native):
                return SessionContinuationOwner(
                    store=native,
                    owner=ticket.owner,
                    receiver=_QualifiedReceiver(),
                    receiver_capability=CapabilityDescriptor(
                        owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                    ),
                    redactor=SecretRedactor(),
                )

            original = await _context(store, session.id)
            prepared = await owner(store).prepare(
                ContinuationWait.model_validate(
                    ticket.model_dump(include=set(ContinuationWait.model_fields))
                ),
                invocation=original,
            )
            with pytest.raises(PermissionError):
                await owner(store).recover_writer(prepared.ticket, invocation=original)
            wrapped = runtime_checkpoint_session_store(store)
            source = await wrapped.load(session.id)
            checkpoint = await wrapped.load_checkpoint(session.id)
            claim = "actual-native-rebind"

            def rebind(current, current_checkpoint):
                updated = checkpoint_with_active_invocation_execution_profile(
                    current_checkpoint,
                    session_id=current.id,
                    interaction_id=original.binding.interaction_id,
                    run_epoch=current.run_epoch + 1,
                    profile=original.profile,
                    expected=original.active_profile,
                )
                updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
                    "version": 1,
                    "claim_id": claim,
                    "claimed_at": now.isoformat(),
                    "claim_expires_at": (now + timedelta(minutes=5)).isoformat(),
                }
                return updated

            command = prepare_rebind_invocation_command(
                source, checkpoint, expected_statuses={source.status}, checkpoint_transform=rebind
            )
            await wrapped.apply_invocation_lifecycle_command(command)
            recovered = await _context(store, session.id, recovery_claim_id=claim)
            with pytest.raises(PermissionError):
                await store._recover_continuation_writer(prepared.ticket)
            with pytest.raises(ContinuationConflict):
                await owner(store).park_recovered(prepared.ticket, invocation=recovered)
            result = await owner(store).recover_writer(prepared.ticket, invocation=recovered)
            assert result.ticket == prepared.ticket and result.preparation == prepared.preparation
            assert result.recovery_writer.run_epoch == recovered.binding.run_epoch
            assert continuation_writer_frontier(result) == (recovered.binding.run_epoch, False)
            assert len(result.events) == len(prepared.events)
            other = reopen()
            assert (
                await owner(other).recover_writer(prepared.ticket, invocation=recovered) == result
            )
            wrong = await _context(store, session.id, recovery_claim_id="different-claim")
            with pytest.raises(ContinuationConflict):
                await owner(other).recover_writer(prepared.ticket, invocation=wrong)
            assert (
                await other.load_continuation_ticket(
                    session.id,
                    registration_key=ticket.registration_key,
                    session_instance_id=session.instance_id,
                )
                == result
            )
            waiting = await owner(other).park_recovered(prepared.ticket, invocation=recovered)
            assert waiting.ticket.state == "WAITING"
            assert waiting.ticket.writer_generation == prepared.ticket.writer_generation
            assert waiting.recovery_writer == result.recovery_writer
            assert (
                await owner(other).park_recovered(prepared.ticket, invocation=recovered) == waiting
            )
            with pytest.raises(ContinuationConflict):
                await owner(other).park(prepared.ticket, invocation=original)
            assert continuation_writer_frontier(waiting) == (recovered.binding.run_epoch, False)

            def expire(_session, checkpoint):
                updated = dict(checkpoint)
                updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
                    "version": 1,
                    "claim_id": claim,
                    "claimed_at": (now - timedelta(minutes=5)).isoformat(),
                    "claim_expires_at": (now - timedelta(seconds=1)).isoformat(),
                }
                return updated

            await wrapped.transform_checkpoint(session.id, expire)
            with pytest.raises(ContinuationConflict):
                await owner(other).park_recovered(prepared.ticket, invocation=recovered)
            assert (
                await other.load_continuation_ticket(
                    session.id,
                    registration_key=ticket.registration_key,
                    session_instance_id=session.instance_id,
                )
                == waiting
            )
            await interrupt_and_release_test_invocation(wrapped, session.id)
            retirement = ContinuationReleasedRetirement(
                retirement=ContinuationRetirement(
                    ticket=waiting.ticket,
                    control_id="cancel-recovered",
                    reason="cancelled",
                    retired_at=now.isoformat(),
                ),
                permit_operation=None,
                permit_commitment=None,
            )
            retired = await owner(other).retire_released(retirement)
            assert retired.ticket.state == "RETIRED"
            assert retired.released_retirement is not None
            assert await owner(other).retire_released(retirement) == retired

    asyncio.run(scenario())
