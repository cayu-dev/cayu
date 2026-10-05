"""Native binding qualification; public runtime qualification is separate."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.core._execution_profile_fixtures import (
    create_admitted_session,
    interrupt_and_release_test_invocation,
)
from tests.core.test_session_continuation import _context
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu.collaboration._capabilities import CapabilityDescriptor
from cayu.collaboration._contracts import OwnerRef
from cayu.external_waits import ExternalEventWaits
from cayu.messages import Message
from cayu.runtime._external_wait_binding import binding_scope
from cayu.runtime._external_wait_receiver import ExternalWaitLatchReceiver
from cayu.runtime._external_wait_settlement import settlement_scope
from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
from cayu.sessions._external_wait_records import (
    elected_external_latch,
    external_continuation_intent,
    external_wait_owner,
)
from cayu.sessions._session_continuation import (
    ContinuationConflict,
    ContinuationReleasedRetirement,
    ContinuationRetirement,
    continuation_digest,
)
from cayu.sessions.base import RunRequest, SessionRunFenced
from cayu.sessions.external_waits import (
    ExternalEventDelivery,
    ExternalWaitConflict,
    ExternalWaitUnavailable,
    external_wait_digest,
)
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("release_first", [False, True])
@pytest.mark.parametrize("retire_bound", ["none", "writer", "released"])
def test_native_binding_requires_sealed_current_writer_and_replays_after_reopen(
    backend, release_first, retire_bound, tmp_path, request
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            policy = Policy()
            waits = ExternalEventWaits(store=store, access_policy=policy)
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            admitted = await create_admitted_session(
                store,
                request=RunRequest(
                    agent_name="assistant",
                    session_id=uuid4().hex,
                    messages=[Message.text("user", "wait")],
                ),
                provider_name="continuation-provider",
                model="continuation-model",
            )
            invocation = await _context(store, admitted.session.id)
            owner = SessionContinuationOwner(
                store=store,
                owner=OwnerRef(
                    application_scope=correlation.request.scope.application_scope,
                    owner_id="runtime",
                    incarnation="test-app",
                ),
                receiver=ExternalWaitLatchReceiver(waits, registered, CONTEXT),
                receiver_capability=CapabilityDescriptor(
                    owner=external_wait_owner(registered), mutations=(), readbacks=(LATCH_FAMILY,)
                ),
                redactor=SecretRedactor(),
            )
            prepared = await owner.prepare(
                external_continuation_intent(registered), invocation=invocation
            )
            command = waits._command(
                "bind", correlation, registration=registered, continuation=prepared.preparation
            )
            before = await waits.inspect(correlation, context=CONTEXT)
            with pytest.raises(PermissionError):
                await store._mutate_external_wait(command)
            assert await waits.inspect(correlation, context=CONTEXT) == before
            if release_first:
                await interrupt_and_release_test_invocation(store, admitted.session.id)
                with binding_scope(command, invocation), pytest.raises(SessionRunFenced):
                    await store._mutate_external_wait(command)
                assert await waits.inspect(correlation, context=CONTEXT) == before
            else:
                with binding_scope(command, invocation):
                    bound = await store._mutate_external_wait(command)
                assert bound.continuation == prepared.preparation
                assert bound.handoff == "pending"
                for disposition in ("settled", "excluded"):
                    forged = waits._command(
                        "settle",
                        correlation,
                        registration=registered,
                        continuation=prepared.preparation,
                        handoff_disposition=disposition,
                        handoff_receipt_sha256=continuation_digest(prepared),
                    )
                    with pytest.raises(PermissionError):
                        await store._mutate_external_wait(forged)
                    with settlement_scope(forged), pytest.raises(ExternalWaitConflict):
                        await store._mutate_external_wait(forged)
                assert (
                    await store._read_external_wait(
                        correlation.request.scope, correlation.request.correlation_key
                    )
                    == bound
                )
                receiver = ExternalWaitLatchReceiver(waits, registered, CONTEXT)
                with pytest.raises(ExternalWaitUnavailable, match="unresolved"):
                    await receiver.reconcile_handoff()
                if retire_bound != "none":
                    cancelled = await waits.cancel(
                        correlation, operation_key="cancel-wait", context=CONTEXT
                    )
                    retirement = ContinuationRetirement(
                        ticket=prepared.ticket,
                        control_id="external-retire:" + external_wait_digest(registered),
                        reason="cancelled",
                        retired_at=datetime.fromtimestamp(
                            cancelled.outcome.selected_at_ms / 1000, tz=UTC
                        ).isoformat(),
                    )
                    if retire_bound == "released":
                        released = ContinuationReleasedRetirement(
                            retirement=retirement,
                            permit_operation=None,
                            permit_commitment=None,
                        )
                        with pytest.raises(ContinuationConflict):
                            await owner.retire_released(released)
                        await interrupt_and_release_test_invocation(store, admitted.session.id)
                        retired = await owner.retire_released(released)
                        assert retired.released_retirement is not None
                        assert await owner.retire_released(released) == retired
                    else:
                        await owner.retire(retirement, invocation=invocation)
                        await interrupt_and_release_test_invocation(store, admitted.session.id)
                    with pytest.raises(ValueError, match="pending external-wait handoff"):
                        await store.delete_session(admitted.session.id)
                    settled = await receiver.reconcile_handoff()
                    assert settled.handoff == "excluded"
                    assert settled.handoff_receipt_sha256 is not None
                    conflict = waits._command(
                        "settle",
                        correlation,
                        registration=registered,
                        continuation=prepared.preparation,
                        handoff_disposition="excluded",
                        handoff_receipt_sha256="0" * 64,
                    )
                    with settlement_scope(conflict), pytest.raises(ExternalWaitConflict):
                        await store._mutate_external_wait(conflict)
                    restored = reopen()
                    restored_waits = ExternalEventWaits(store=restored, access_policy=policy)
                    restored_receiver = ExternalWaitLatchReceiver(
                        restored_waits, registered, CONTEXT
                    )
                    assert await restored_receiver.reconcile_handoff() == settled
                    if retire_bound == "released":
                        # External exclusion alone does not erase the native
                        # acknowledgement obligation.
                        assert settled.pending_handoff
                        assert not settled.retirement_complete
                        incomplete = conflict.model_copy(
                            update={
                                "kind": "complete_retirement",
                                "handoff_receipt_sha256": settled.handoff_receipt_sha256,
                            }
                        )
                        with pytest.raises(PermissionError):
                            await restored._mutate_external_wait(incomplete)
                        with settlement_scope(incomplete), pytest.raises(ExternalWaitConflict):
                            await restored._mutate_external_wait(incomplete)
                        with pytest.raises(ValueError, match="pending external-wait handoff"):
                            await restored.delete_session(admitted.session.id)
                        restored_owner = SessionContinuationOwner(
                            store=restored,
                            owner=owner.owner,
                            receiver=restored_receiver,
                            receiver_capability=owner.receiver_capability,
                            redactor=SecretRedactor(),
                        )
                        settled = await restored_receiver.retire_released(restored_owner)
                        assert settled.retirement_complete
                        assert not settled.pending_handoff
                        assert await restored_receiver.retire_released(restored_owner) == settled
                        await restored_owner.drain()
                    await restored.delete_session(admitted.session.id)
                    assert await restored.load(admitted.session.id) is None
                    assert await restored_receiver.reconcile_handoff() == settled
                    if retire_bound == "released":
                        assert await restored_receiver.retire_released(restored_owner) == settled
                    await restored_waits.aclose()
                    await owner.drain()
                    return
                await interrupt_and_release_test_invocation(store, admitted.session.id)
                restored = reopen()
                with binding_scope(command, invocation):
                    assert await restored._mutate_external_wait(command) == bound
                with pytest.raises(PermissionError):
                    await restored._mutate_external_wait(command)
                # The external handoff has its own durable obligation, rather
                # than depending on the temporary native ticket pin alone.
                with pytest.raises(ValueError, match="pending external-wait handoff"):
                    await restored.validate_session_closure_admission(admitted.session.id)
                with pytest.raises(ValueError, match="pending external-wait handoff"):
                    await restored.delete_session(admitted.session.id)
                assert await restored.load(admitted.session.id) is not None
                assert (
                    await restored._read_external_wait(
                        correlation.request.scope, correlation.request.correlation_key
                    )
                    == bound
                )
                await waits.deliver(
                    ExternalEventDelivery(
                        correlation=correlation,
                        delivery_id="completed",
                        payload_json='{"result":42}',
                    ),
                    context=CONTEXT,
                )
                await waits.project(registered, context=CONTEXT)
                ready = await restored._read_external_wait(
                    correlation.request.scope, correlation.request.correlation_key
                )
                candidate = elected_external_latch(ready)
                reopened_waits = ExternalEventWaits(store=restored, access_policy=policy)
                receiver = ExternalWaitLatchReceiver(reopened_waits, registered, CONTEXT)
                assert await receiver.authenticate_continuation_latch(candidate) == candidate
                latched = await owner.latch(candidate)
                assert latched.latch == candidate
                assert (
                    await restored.load_continuation_ticket(
                        admitted.session.id,
                        registration_key=prepared.ticket.registration_key,
                        session_instance_id=admitted.session.instance_id,
                    )
                    == latched
                )
                with pytest.raises(ValueError):
                    await receiver.authenticate_continuation_latch(
                        candidate.model_copy(update={"outcome_digest": "0" * 64})
                    )
                policy.revoked = True
                with pytest.raises(PermissionError):
                    await receiver.authenticate_continuation_latch(candidate)
                await reopened_waits.aclose()
            await owner.drain()

    asyncio.run(scenario())
