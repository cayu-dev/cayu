"""Real question deadline expiry through content-free administrative recovery."""

import asyncio
import time

import pytest
from tests.core.test_participant_identity import CONTEXT, app

from cayu import ClarificationExpiryRequest, ClarificationQuestionRecovery
from cayu.collaboration import _clarification_question_recovery as recovery_owner
from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import CollaborationConflict, CollaborationContractError
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.vaults.redaction import SecretRedactor


async def expire_due_question(
    application,
    factory,
    initialized,
    command,
    registration,
    monkeypatch,
    *,
    cancellation,
    restart=None,
):
    redactor = SecretRedactor()
    selected = ClarificationQuestionRecovery(
        operation=command.question.operation,
        question_sha256=clarification_commitment(command.question, redactor),
        request_sha256=clarification_commitment(command.expected, redactor),
    )
    expiry = ClarificationExpiryRequest(
        operation=initialized.operation("expire-question"), recovery=selected
    )
    outsider = CONTEXT.model_copy(update={"principal": "outsider"})
    with pytest.raises(CollaborationAccessDenied):
        await application.list_due_clarification_questions(context=outsider)
    with pytest.raises(CollaborationAccessDenied):
        await application.expire_clarification_question(expiry, context=outsider)
    for invalid in (0, 65, True):
        with pytest.raises(CollaborationContractError):
            await application.list_due_clarification_questions(context=CONTEXT, limit=invalid)
    assert not (await application.list_due_clarification_questions(context=CONTEXT)).items
    with pytest.raises(CollaborationConflict):
        await application.expire_clarification_question(expiry, context=CONTEXT)

    await asyncio.sleep(
        max(0, (command.question.deadline_at_ms - time.time_ns() // 1_000_000) / 1000) + 0.02
    )
    # Recovery knows no source grant, provider, dispatch request or executable
    # question context. It reconstructs only the original owner initialization.
    store = factory()
    reopened = app(store, registration)
    try:
        assert await reopened.initialize_collaboration() == initialized
        page = await reopened.list_due_clarification_questions(context=CONTEXT, limit=1)
        assert len(page.items) == 1 and page.items[0].recovery == selected
        assert page.next_cursor is not None
        assert not (
            await reopened.list_due_clarification_questions(
                context=CONTEXT, cursor=page.next_cursor, limit=1
            )
        ).items
        for private in ("Which API version?", "private-state", "mandate", "payload", "permit"):
            assert private not in page.model_dump_json()
        expiry = ClarificationExpiryRequest.model_validate_json(expiry.model_dump_json())
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            before = await store._anchor(tx, initialized, redactor)
            deliveries = await tx.scan_clarification_request_handoffs(
                command.question.request,
                family="clarification_deliveries",
                pending_only=True,
                limit=64,
            )
            assert len(deliveries) == 1
        for field in ("question_sha256", "request_sha256"):
            with pytest.raises(CollaborationConflict):
                await reopened.expire_clarification_question(
                    expiry.model_copy(
                        update={"recovery": selected.model_copy(update={field: "f" * 64})}
                    ),
                    context=CONTEXT,
                )
        with pytest.raises(CollaborationUnavailable):
            await reopened.expire_clarification_question(
                expiry.model_copy(
                    update={
                        "recovery": selected.model_copy(
                            update={
                                "operation": selected.operation.model_copy(
                                    update={"caller_key": "unknown-question"}
                                ),
                            }
                        )
                    }
                ),
                context=CONTEXT,
            )
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            assert await store._anchor(tx, initialized, redactor) == before

        original = recovery_owner.expire_question
        receipts = []
        settled = asyncio.Event()
        entered, release = asyncio.Event(), asyncio.Event()

        async def interrupted(*args, **kwargs):
            if cancellation:
                entered.set()
                await release.wait()
            result = await original(*args, **kwargs)
            receipts.append(result)
            settled.set()
            if not cancellation:
                raise ConnectionError("expiry acknowledgement lost after native commit")
            return result

        with monkeypatch.context() as fault:
            fault.setattr(recovery_owner, "expire_question", interrupted)
            if cancellation:
                observer = asyncio.create_task(
                    reopened.expire_clarification_question(expiry, context=CONTEXT)
                )
                try:
                    await asyncio.wait_for(entered.wait(), 10)
                    observer.cancel()
                    observer.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await observer
                    assert observer.cancelled() and observer.cancelling() == 2
                    assert (
                        await reopened.list_due_clarification_questions(context=CONTEXT, limit=1)
                        == page
                    )
                finally:
                    release.set()
                    await asyncio.gather(observer, return_exceptions=True)
            else:
                with pytest.raises(CollaborationUnavailable):
                    await reopened.expire_clarification_question(expiry, context=CONTEXT)
        if restart is not None:
            await restart(expiry)
        result = await reopened.expire_clarification_question(expiry, context=CONTEXT)
        assert (
            result.status == "expired" and result.expired_at_ms >= command.question.deadline_at_ms
        )
        assert await reopened.expire_clarification_question(expiry, context=CONTEXT) == result
        with pytest.raises(CollaborationConflict):
            await reopened.expire_clarification_question(
                expiry.model_copy(
                    update={
                        "operation": expiry.operation.model_copy(
                            update={"caller_key": "replacement-expiry"}
                        )
                    }
                ),
                context=CONTEXT,
            )
        assert await reopened.expire_clarification_question(expiry, context=CONTEXT) == result
        await asyncio.wait_for(settled.wait(), 10)
        assert receipts and all(receipt == result for receipt in receipts)
        assert not (await reopened.list_due_clarification_questions(context=CONTEXT)).items
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            after = await store._anchor(tx, initialized, redactor)
            assert after.reserved_operations == before.reserved_operations - 1
            assert after.reserved_events == before.reserved_events - 1
            assert (
                await tx.scan_clarification_request_handoffs(
                    command.question.request,
                    family="clarification_deliveries",
                    pending_only=True,
                    limit=64,
                )
                == deliveries
            )
            lineage = await tx.get(
                "clarification_lineages", operation_key(command.question.lineage)
            )
            assert lineage["usage"]["pending"] == 1
    finally:
        if not isinstance(store, InMemoryCollaborationStore):
            await store.close()
