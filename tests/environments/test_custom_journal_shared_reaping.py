"""Recovery reaps a custom-journal run's shared sandbox only when the journal says the run ended.

The journal keeps the run's events outside the session store (in a list that
outlives every app, like an external database), and creates the run's anchor
session so the steps can share a sandbox. Each "process" is a separate
CayuApp on one SQLite file.
"""

from __future__ import annotations

import asyncio
from hashlib import sha256

import pytest
from tests.core.test_workflows_hardening import MemoryJournal
from tests.environments.test_durable_shared_allocations import (
    _SHARED,
    _crash,
    _records,
    _World,
)
from tests.sandbox_broker.test_brokered_shared_allocation import _call, _done

from cayu import EventType
from cayu.environments import EnvironmentFactoryOperation
from cayu.events import Event
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import SessionStatus
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionsRecoveryRequest,
)
from cayu.workflows import (
    WORKFLOW_ATTEMPT_EVENT_TYPE,
    EventStoreJournal,
    WorkflowBase,
    WorkflowJournalContext,
    WorkflowJournalRunEnd,
    WorkflowSpec,
    step,
)


class _ExternalJournal(MemoryJournal):
    """Keeps events in an external list; reports nothing about the run's end."""

    def __init__(self, context: WorkflowJournalContext, events: list[Event]) -> None:
        super().__init__()
        self.events = events
        self._anchor = EventStoreJournal(
            context.session_store, context.session_id, context.workflow_name
        )

    async def append(self, event: Event) -> None:
        # The anchor session lets the run's steps share an allocation.
        await self._anchor._ensure_session()
        await super().append(event)


class _EndReportingJournal(_ExternalJournal):
    """Also reports when the run's newest attempt journaled its completion."""

    async def run_ended(self) -> bool:
        latest = self._latest_attempt_id()
        return latest is not None and any(
            event.type == EventType.WORKFLOW_COMPLETED and event.payload.get("attempt_id") == latest
            for event in self.events
        )


class _Crash(Exception):
    """The process died here."""


class _PlanThenStop(WorkflowBase):
    spec = WorkflowSpec(name="plan-then-solve")

    def __init__(self, app, journal_factory, *, crash: bool = False) -> None:
        super().__init__(app, journal_factory=journal_factory)
        self._crash = crash

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        await step(ctx, agent="planner", step_id="plan", prompt="plan", run_options=_SHARED)
        if self._crash:
            # The newest attempt journals its completion, and the process dies
            # before it released the sandbox.
            await ctx._append_current_attempt_event(
                ctx.event(EventType.WORKFLOW_COMPLETED, payload={"workflow": ctx.workflow_name})
            )
            raise _Crash()


@pytest.mark.parametrize("reports_end", [True, False])
def test_recovery_asks_a_custom_journal_whether_its_run_ended(tmp_path, reports_end):
    world = _World(tmp_path)
    events: list[Event] = []
    journal_type = _EndReportingJournal if reports_end else _ExternalJournal

    def journal_factory(context: WorkflowJournalContext):
        return journal_type(context, events)

    async def scenario():
        try:
            first = world.app(_call("write_plan"), _done())
            with pytest.raises(_Crash):
                async for _ in _PlanThenStop(first, journal_factory, crash=True).run("run-j"):
                    pass
            _crash(first)
            # The attempt markers live only in the external journal.
            assert any(event.type == WORKFLOW_ATTEMPT_EVENT_TYPE for event in events)
            assert not await first.session_store.query_events(
                EventQuery(session_id="run-j", event_type=WORKFLOW_ATTEMPT_EVENT_TYPE)
            )
            assert len((await _records(first, "run-j"))["records"]) == 1
            recovering = world.app()
            # The recovering process runs this workflow too, with its journal.
            _PlanThenStop(recovering, journal_factory)
            page = await recovering.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(statuses={SessionStatus.COMPLETED})
            )
            return page, await _records(recovering, "run-j")
        finally:
            await world.close()

    page, records = asyncio.run(scenario())
    actions = {result.session_id: result.actions for result in page.results}
    (created, *rest) = world.sandboxes.requests
    sandbox = sha256(created.session_id.encode()).hexdigest()[:16]
    assert isinstance(_EndReportingJournal.__new__(_EndReportingJournal), WorkflowJournalRunEnd)
    assert not isinstance(_ExternalJournal.__new__(_ExternalJournal), WorkflowJournalRunEnd)
    if reports_end:
        assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION in actions["run-j"]
        # Reconnected and discarded, without a copy-back.
        assert [request.operation for request in rest] == [EnvironmentFactoryOperation.RECONNECT]
        assert world.sandboxes.finalizations == []
        assert world.sandboxes.disposed == [sandbox]
        assert records == {}
    else:
        # A journal that can't say the run ended leaves it resumable: the
        # sandbox and its record stay for the run's resume.
        assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION not in actions.get("run-j", ())
        assert rest == []
        assert world.sandboxes.path(sandbox).is_dir()
        assert len(records["records"]) == 1


def test_an_unordered_attempt_never_takes_over_a_claim_made_after_its_fence_check(tmp_path):
    """Custom-journal attempts have no journal sequence; the claim generation orders them."""

    from cayu.runtime._shared_allocation_records import (
        SharedAllocationClaimLost,
        SharedAllocationRecords,
    )

    world = _World(tmp_path)
    events: list[Event] = []

    def journal_factory(context: WorkflowJournalContext):
        return _ExternalJournal(context, events)

    class Start(WorkflowBase):
        spec = WorkflowSpec(name="plan-then-solve")

        async def run(self, session_id):
            yield await self.context(session_id).start()

    async def scenario():
        try:
            app = world.app()
            async for _ in Start(app, journal_factory=journal_factory).run("run-order"):
                pass
            records = SharedAllocationRecords(app._runtime_session_store)
            fields = {
                "record_id": "r",
                "scope": "parent",
                "scope_id": "run-order",
                "owner_session_id": "run-order",
                "environment_name": "sandbox",
                "agent_name": "planner",
                "attempt_sequence": 0,
            }
            # Attempt A reads the record (none yet) and passes its fence check;
            # then newer attempt B, which also has no sequence, claims it.
            seen_by_a = 0
            b = await records.claim(**fields, attempt_id="attempt-b", expected_generation=0)
            assert b is not None
            # A's claim is refused rather than taking over B's.
            with pytest.raises(SharedAllocationClaimLost):
                await records.claim(**fields, attempt_id="attempt-a", expected_generation=seen_by_a)
            assert await records.holds("run-order", b.record.allocation_session_id, {"attempt-b"})
            # An attempt that saw B's claim before its own fence check may take
            # it over (B is then the older one).
            c = await records.claim(
                **fields, attempt_id="attempt-c", expected_generation=b.record.claim_generation
            )
            assert c is not None and c.record.attempt_id == "attempt-c"
        finally:
            await world.close()

    asyncio.run(scenario())
