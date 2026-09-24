"""Request-owner transaction qualification, not public service qualification."""

import asyncio
import time

import pytest
from tests.core.test_clarification_commands import question_for
from tests.core.test_collaboration_request_foundation import accept, setup
from tests.core.test_participant_identity import stores

from cayu.collaboration._clarification_commands import (
    ClarificationCloseCommand,
    ClarificationOpenCommand,
    ClarificationReplyCommand,
)
from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._clarification_store import (
    close_in_transaction,
    discover_due_in_transaction,
    open_in_transaction,
    reply_in_transaction,
)
from cayu.collaboration._contracts import CollaborationConflict, snapshot_input
from cayu.collaboration._permits import ReceivingSettlementReceipt
from cayu.collaboration._request_arbitration import admit_in_transaction
from cayu.collaboration._request_store import control_in_transaction, retained_request
from cayu.collaboration.clarifications import ClarificationReply
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import (
    RequestAdmissionCommand,
    RequestControl,
    RequestControlCommand,
)
from cayu.vaults.redaction import SecretRedactor

__all__ = ["stores"]
pytestmark = pytest.mark.anyio
REDACTOR = SecretRedactor()


async def test_native_root_lineage_cannot_use_unowned_namespace(stores):
    store = stores()
    initial, command = await preparation(store)
    question = command.question.model_copy(
        update={
            "lineage": command.question.lineage.model_copy(
                update={"namespace_incarnation": "foreign"}
            ),
        }
    )
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        before = await store._anchor(tx, initial, REDACTOR)
    with pytest.raises(CollaborationConflict):
        await open_question(store, initial, command.model_copy(update={"question": question}))
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        assert await store._anchor(tx, initial, REDACTOR) == before
        assert (
            await tx.get(
                "clarification_lineages",
                ("foreign", question.lineage.generation, question.lineage.caller_key),
            )
            is None
        )
    assert (await open_question(store, initial, command)).decision.state == "open"


async def test_native_due_discovery_and_expiry_after_real_deadline(stores):
    store = stores()
    initial, command = await preparation(store)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        deadline = await tx.now_ms() + 2500
    question = command.question.model_copy(update={"deadline_at_ms": deadline})
    command = command.model_copy(update={"question": question})
    opening = await open_question(store, initial, command)
    await asyncio.sleep(max(0, (deadline - time.time_ns() // 1_000_000) / 1000) + 0.02)
    reopened = stores()
    async with reopened._transaction(initial.binding.application_scope, write=False) as tx:
        due = await discover_due_in_transaction(
            reopened, tx, initial, after=None, limit=1, redactor=REDACTOR
        )
    assert due == (opening,)
    closure = ClarificationCloseCommand(
        operation=initial.operation("expire-question"),
        expected=command.expected,
        question=question.operation,
        question_sha256=clarification_commitment(question, REDACTOR),
        kind="expired",
        initiator=question.initiator,
    )
    async with reopened._transaction(initial.binding.application_scope, write=True) as tx:
        receipt = await close_in_transaction(reopened, tx, initial, closure, redactor=REDACTOR)
    assert receipt.decision.state == "expired"
    async with reopened._transaction(initial.binding.application_scope, write=False) as tx:
        assert (
            await discover_due_in_transaction(
                reopened, tx, initial, after=None, limit=1, redactor=REDACTOR
            )
            == ()
        )


async def test_native_question_close_replays_and_releases_only_decision_reserve(stores):
    store = stores()
    initial, command = await preparation(store)
    await open_question(store, initial, command)
    close = ClarificationCloseCommand(
        operation=initial.operation("close-question"),
        expected=command.expected,
        question=command.question.operation,
        question_sha256=clarification_commitment(command.question, REDACTOR),
        kind="cancelled",
        initiator=command.question.initiator,
    )

    async def settle(instance, intent):
        async with instance._transaction(initial.binding.application_scope, write=True) as tx:
            return await close_in_transaction(instance, tx, initial, intent, redactor=REDACTOR)

    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        before = await store._anchor(tx, initial, REDACTOR)
    for kind in ("expired", "superseded", "request_terminal"):
        with pytest.raises(CollaborationConflict):
            await settle(store, close.model_copy(update={"kind": kind}))
    first, second = await asyncio.gather(settle(store, close), settle(stores(), close))
    assert first == second
    assert first.decision.state == "cancelled"
    assert first.decision.input is None
    assert await settle(stores(), close) == first
    with pytest.raises(CollaborationConflict):
        await settle(store, close.model_copy(update={"kind": "expired"}))
    with pytest.raises(CollaborationConflict):
        await answer_question(store, initial, reply_for(command))
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        after = await store._anchor(tx, initial, REDACTOR)
        assert after.reserved_operations == before.reserved_operations - 1
        assert after.reserved_events == before.reserved_events - 1
        parent = await retained_request(
            store,
            tx,
            initial,
            command.expected.intent.request,
            command.expected.initiator,
            REDACTOR,
        )
        assert parent is not None and parent.state == "open"
        assert parent.clarification.input_revision == 0


async def test_native_missing_question_tail_cannot_reset_frontier(stores):
    store = stores()
    initial, command = await preparation(store)
    await open_question(store, initial, command)
    async with store._transaction(initial.binding.application_scope, write=True) as tx:
        before = await store._anchor(tx, initial, REDACTOR)
        await tx.delete(
            "clarification_questions",
            (
                command.operation.namespace_incarnation,
                command.operation.generation,
                command.operation.caller_key,
            ),
        )
    with pytest.raises(CollaborationUnavailable, match="frontier"):
        await open_question(stores(), initial, command)
    different = command.operation.model_copy(update={"caller_key": "replacement"})
    with pytest.raises(CollaborationUnavailable, match="frontier"):
        await open_question(
            stores(),
            initial,
            command.model_copy(
                update={
                    "operation": different,
                    "question": command.question.model_copy(update={"operation": different}),
                }
            ),
        )
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        assert await store._anchor(tx, initial, REDACTOR) == before


async def preparation(store):
    values = await setup(store)
    initial = values[1]
    accepted = await accept(store, values)
    admission = RequestAdmissionCommand(
        operation=initial.operation("admission"),
        expected=accepted.receipt.expected,
        expected_revision=accepted.revision,
        expected_input_revision=0,
        expected_input_sha256=clarification_commitment(accepted.receipt.expected, REDACTOR),
        generation=1,
        decision="clarify",
        evidence=(),
        initiator=accepted.receipt.expected.initiator,
    )
    async with store._transaction(initial.binding.application_scope, write=True) as tx:
        receipt = await admit_in_transaction(store, tx, initial, admission, redactor=REDACTOR)
    question = question_for(accepted.receipt.expected)
    command = ClarificationOpenCommand(
        operation=question.operation,
        expected=accepted.receipt.expected,
        expected_revision=receipt.revision,
        question=question,
    )
    return initial, command


async def open_question(store, initial, command, *, fail=False):
    async with store._transaction(initial.binding.application_scope, write=True) as tx:
        result = await open_in_transaction(store, tx, initial, command, redactor=REDACTOR)
        if fail:
            raise ConnectionError("after all clarification writes")
        return result


def reply_for(command):
    question = command.question
    reply = ClarificationReply(
        operation=question.operation.model_copy(update={"caller_key": "reply"}),
        question=question.operation,
        question_sha256=clarification_commitment(question, REDACTOR),
        question_generation=question.generation,
        question_input_revision=question.input_revision,
        expected_input_revision=question.input_revision,
        expected_input_sha256=question.input_sha256,
        initiator=question.initiator,
        responder=question.responder,
        service=question.operation.model_copy(update={"caller_key": "service"}),
        service_generation=1,
        service_session=question.receiver.model_copy(
            update={"kind": "session", "object_id": "service"}
        ),
        invocation=question.receiver.model_copy(
            update={"kind": "invocation", "object_id": "invocation"}
        ),
        admission_sha256="f" * 64,
        production_stage_id="reply-stage",
        production_sha256="a" * 64,
        source=question.source,
    )
    return ClarificationReplyCommand(
        operation=reply.operation, expected=command.expected, reply=reply
    )


async def answer_question(store, initial, command, *, fail=False):
    async with store._transaction(initial.binding.application_scope, write=True) as tx:
        result = await reply_in_transaction(store, tx, initial, command, redactor=REDACTOR)
        if fail:
            raise ConnectionError("after all reply writes")
        return result


async def test_native_question_open_replay_conflict_and_reserve(stores):
    store = stores()
    initial, command = await preparation(store)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        before = await store._anchor(tx, initial, REDACTOR)
    receipt = await open_question(store, initial, command)
    assert await open_question(stores(), initial, command) == receipt
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        after = await store._anchor(tx, initial, REDACTOR)
        assert after.operation_count == before.operation_count + 1
        assert after.reserved_operations == before.reserved_operations + 1
        assert after.reserved_events == before.reserved_events + 1
        assert await tx.scan_clarification_questions(command.question.request, limit=2) == [
            snapshot_input(receipt.decision)
        ]
    changed = command.model_copy(update={"expected_revision": command.expected_revision + 1})
    with pytest.raises(CollaborationConflict):
        await open_question(store, initial, changed)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        assert await store._anchor(tx, initial, REDACTOR) == after


async def test_native_question_late_failure_rolls_back_reservation(stores):
    store = stores()
    initial, command = await preparation(store)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        before = await store._anchor(tx, initial, REDACTOR)
    with pytest.raises(ConnectionError):
        await open_question(store, initial, command, fail=True)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        assert await tx.scan_clarification_questions(command.question.request, limit=1) == []
        assert await store._anchor(tx, initial, REDACTOR) == before
    assert (await open_question(stores(), initial, command)).command == command


async def test_native_competing_question_workers_replay_once(stores):
    store = stores()
    initial, command = await preparation(store)
    first, second = await asyncio.gather(
        open_question(store, initial, command), open_question(stores(), initial, command)
    )
    assert first == second
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        lineage = await tx.get(
            "clarification_lineages",
            (
                command.question.lineage.namespace_incarnation,
                command.question.lineage.generation,
                command.question.lineage.caller_key,
            ),
        )
        assert lineage["usage"]["questions"] == 1
        assert lineage["usage"]["pending"] == 1
        assert lineage["usage"]["content_bytes"] == (
            command.question.source.content_bytes + command.question.policy.max_reply_bytes
        )


async def test_native_reply_and_cancellation_workers_elect_once(stores):
    store = stores()
    initial, command = await preparation(store)
    await open_question(store, initial, command)
    reply = reply_for(command)
    close = ClarificationCloseCommand(
        operation=initial.operation("racing-question-cancellation"),
        expected=command.expected,
        question=command.question.operation,
        question_sha256=clarification_commitment(command.question, REDACTOR),
        kind="cancelled",
        initiator=command.question.initiator,
    )
    start = asyncio.Event()

    async def accept():
        await start.wait()
        return await answer_question(store, initial, reply)

    async def cancel():
        await start.wait()
        other = stores()
        async with other._transaction(initial.binding.application_scope, write=True) as tx:
            return await close_in_transaction(other, tx, initial, close, redactor=REDACTOR)

    tasks = (asyncio.create_task(accept()), asyncio.create_task(cancel()))
    start.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    winners = [item for item in results if not isinstance(item, BaseException)]
    failures = [item for item in results if isinstance(item, BaseException)]
    assert len(winners) == len(failures) == 1
    assert isinstance(failures[0], CollaborationConflict)
    winner = winners[0]
    accepted = winner.decision.state == "answered"
    assert winner.decision.state in {"answered", "cancelled"}
    assert await (accept() if accepted else cancel()) == winner
    with pytest.raises(CollaborationConflict):
        await (cancel() if accepted else accept())
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        parent = await retained_request(
            store,
            tx,
            initial,
            command.expected.intent.request,
            command.expected.initiator,
            REDACTOR,
        )
        assert parent is not None and parent.state == "open"
        assert parent.clarification.input_revision == int(accepted)


async def test_native_reply_replay_input_ancestry_and_next_question(stores):
    store = stores()
    initial, command = await preparation(store)
    await open_question(store, initial, command)
    reply = reply_for(command)
    first, second = await asyncio.gather(
        answer_question(store, initial, reply), answer_question(stores(), initial, reply)
    )
    assert first == second
    assert first.decision.input.revision == 1
    next_question = command.question.model_copy(
        update={
            "operation": command.operation.model_copy(update={"caller_key": "question-2"}),
            "generation": 2,
            "input_revision": 1,
            "input_sha256": clarification_commitment(first.decision.input, REDACTOR),
        }
    )
    next_command = command.model_copy(
        update={"operation": next_question.operation, "question": next_question}
    )
    assert (await open_question(store, initial, next_command)).command == next_command
    assert await answer_question(stores(), initial, reply) == first
    changed_reply = reply.reply.model_copy(update={"admission_sha256": "a" * 64})
    with pytest.raises(CollaborationConflict):
        await answer_question(store, initial, reply.model_copy(update={"reply": changed_reply}))


async def test_admission_binds_effective_input_and_preserves_committed_replay(stores):
    store = stores()
    initial, command = await preparation(store)
    await open_question(store, initial, command)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        prior = await retained_request(
            store,
            tx,
            initial,
            command.expected.intent.request,
            command.expected.initiator,
            REDACTOR,
        )
        assert prior is not None
        original = await tx.get(
            "operations", (initial.namespace_incarnation, initial.generation, "admission")
        )
    from cayu.collaboration._preparation import prepare_contract
    from cayu.collaboration.requests import RequestAdmissionReceipt

    original = prepare_contract(RequestAdmissionReceipt, original, redactor=REDACTOR)
    stale = original.command.model_copy(
        update={
            "operation": initial.operation("later-admission"),
            "expected_revision": prior.revision,
            "generation": prior.admission_generation + 1,
        }
    )
    reply = await answer_question(store, initial, reply_for(command))
    reopened = stores()
    async with reopened._transaction(initial.binding.application_scope, write=False) as tx:
        before = await reopened._anchor(tx, initial, REDACTOR)
    for candidate in (
        stale,
        stale.model_copy(update={"expected_input_revision": 1}),
        stale.model_copy(
            update={
                "expected_input_sha256": clarification_commitment(reply.decision.input, REDACTOR)
            }
        ),
    ):
        with pytest.raises(CollaborationConflict, match="effective input"):
            async with reopened._transaction(initial.binding.application_scope, write=True) as tx:
                await admit_in_transaction(reopened, tx, initial, candidate, redactor=REDACTOR)
    async with reopened._transaction(initial.binding.application_scope, write=True) as tx:
        assert await reopened._anchor(tx, initial, REDACTOR) == before
        assert (
            await admit_in_transaction(reopened, tx, initial, original.command, redactor=REDACTOR)
            == original
        )
        refreshed = stale.model_copy(
            update={
                "expected_input_revision": 1,
                "expected_input_sha256": clarification_commitment(reply.decision.input, REDACTOR),
            }
        )
        receipt = await admit_in_transaction(reopened, tx, initial, refreshed, redactor=REDACTOR)
        assert receipt.command == refreshed
    async with reopened._transaction(initial.binding.application_scope, write=True) as tx:
        assert (
            await admit_in_transaction(reopened, tx, initial, refreshed, redactor=REDACTOR)
            == receipt
        )
    with pytest.raises(CollaborationConflict):
        async with reopened._transaction(initial.binding.application_scope, write=True) as tx:
            await admit_in_transaction(reopened, tx, initial, stale, redactor=REDACTOR)


async def test_native_reply_late_failure_preserves_open_question_and_reserve(stores):
    store = stores()
    initial, command = await preparation(store)
    opening = await open_question(store, initial, command)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        before = await store._anchor(tx, initial, REDACTOR)
    reply = reply_for(command)
    with pytest.raises(ConnectionError):
        await answer_question(store, initial, reply, fail=True)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        assert await store._anchor(tx, initial, REDACTOR) == before
        assert await tx.scan_clarification_questions(command.question.request, limit=1) == [
            snapshot_input(opening.decision)
        ]
    result = await answer_question(stores(), initial, reply)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        after = await store._anchor(tx, initial, REDACTOR)
        assert after.reserved_operations == before.reserved_operations - 1
        assert after.reserved_events == before.reserved_events - 1
    assert result.decision.state == "answered"


async def test_native_cancelled_parent_rejects_reply_without_mutation(stores):
    store = stores()
    initial, command = await preparation(store)
    opening = await open_question(store, initial, command)
    operation = initial.operation("cancel-parent")
    cancel = RequestControlCommand(
        operation=operation,
        kind="cancel",
        source=initial.owner,
        destination=initial.owner,
        initiator=command.expected.initiator,
        intent=RequestControl(
            operation=operation,
            expected=command.expected,
            expected_revision=command.expected_revision,
            kind="cancel",
        ),
    )
    async with store._transaction(initial.binding.application_scope, write=True) as tx:
        prior = await retained_request(
            store,
            tx,
            initial,
            command.expected.intent.request,
            command.expected.initiator,
            REDACTOR,
        )
        assert prior is not None
        # This transaction test has dispatched no producer. Supply the exact
        # receiving evidence required by the private control composition API;
        # public receiver provenance is a separate acceptance qualification.
        settlement = ReceivingSettlementReceipt(
            expected=prior.permit,
            receiving_owner=initial.owner,
            receipt_id="test-no-producer",
            outcome="quiescent",
        )
        await control_in_transaction(
            store,
            tx,
            initial,
            cancel,
            authority_expires_at_ms=time.time_ns() // 1_000_000 + 300_000,
            redactor=REDACTOR,
            settlement=settlement,
        )
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        before = await store._anchor(tx, initial, REDACTOR)
    with pytest.raises(CollaborationConflict):
        await answer_question(stores(), initial, reply_for(command))
    assert await open_question(store, initial, command) == opening
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        assert await store._anchor(tx, initial, REDACTOR) == before
