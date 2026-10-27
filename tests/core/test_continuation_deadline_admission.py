"""A settle's new execution deadline is stored only once its decision is admitted."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_continuation_deadlines import skip_ahead as skip_ahead
from tests.core.test_human_review import CONTEXT, ReviewPolicy, decision, make_app, pause, resolve

from cayu.approvals.review import HumanReviewConflict
from cayu.deadlines import ExecutionDeadline
from cayu.events import EventType
from cayu.sessions import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval", [False, True])
def test_a_stale_human_review_decision_does_not_replace_the_deadline(
    skip_ahead, tmp_path, backend, approval
) -> None:
    async def scenario() -> None:
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        policy = ReviewPolicy()
        app, _provider, tool = make_app(store, policy, approval=approval)
        try:
            await pause(app)
            before = (await store.load("review-session")).execution_deadline
            stale = await app.inspect_human_review("review-session", context=CONTEXT)
            # The policy changes; the reviewer is still allowed to decide.
            policy.version = "delivery-v2"
            refused = decision(stale, approval=approval).model_copy(
                update={"execution_deadline": ExecutionDeadline.after(60)}
            )
            with pytest.raises(HumanReviewConflict):
                await resolve(app, refused)
            after = (await store.load("review-session")).execution_deadline
            assert after.expires_at == before.expires_at
            assert after.origin == before.origin
            if approval:
                assert not tool.calls

            # Well past the refused decision's 60 seconds, a current decision still runs.
            skip_ahead(120)
            current = await app.inspect_human_review("review-session", context=CONTEXT)
            events = await resolve(app, decision(current, approval=approval))
            assert events[-1].type is EventType.SESSION_COMPLETED
            if approval:
                assert tool.calls == [{"value": "morning"}]
        finally:
            await app.aclose()
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_an_admitted_decision_stores_its_new_deadline(skip_ahead, tmp_path, backend) -> None:
    async def scenario() -> None:
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        app, _, tool = make_app(store, ReviewPolicy(), approval=True)
        try:
            await pause(app)
            view = await app.inspect_human_review("review-session", context=CONTEXT)
            accepted = decision(view, approval=True).model_copy(
                update={"execution_deadline": ExecutionDeadline.after(600)}
            )
            events = await resolve(app, accepted)
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert tool.calls == [{"value": "morning"}]
            stored = (await store.load("review-session")).execution_deadline
            assert stored.origin == "explicit"
            assert stored.expires_at == accepted.execution_deadline.expires_at
        finally:
            await app.aclose()
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())
