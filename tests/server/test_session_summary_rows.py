"""Overviews served from session summary rows equal the reference reader's.

Random session histories (runs, approvals, questions, resumes, forks, children,
interruption, deletion and reuse of IDs) are applied to each native store, with
overview reads between the writes so stored documents are reused, extended and
invalidated. After every step the stored overviews must equal what
``SessionOverviewReader`` computes from the event log, and the CLI's
recompute check must agree.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from datetime import timedelta

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_cloud_reporting import (
    ANSWER,
    REVIEWER,
    _app,
    _delegating_app,
    _drain,
    _pending,
    _start,
)

from cayu import Message, ResumeRequest, RunRequest
from cayu.approvals import ToolApprovalDecision, ToolApprovalRequest, UserInputResponse
from cayu.budgets.pricing import default_price_book
from cayu.server._session_overview import SessionOverviewService
from cayu.server._session_summaries import check_session_summary
from cayu.sessions import ForkSessionRequest, InterruptSessionRequest
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import PendingActionKind
from cayu.sessions.summary_rows import SummarySettlement
from cayu.storage.retention import RetentionMode, SessionRetentionPolicy

APPLIED: list[str] = []

__all__ = ["conformance_postgres_dsn", "session_store_case"]


def _service(app) -> SessionOverviewService:
    return SessionOverviewService(
        store=app.session_store,
        pricing=default_price_book(),
        project_session=lambda session_id: session_id,
        redact=lambda value: value,
        artifact_stores=dict,
    )


async def _all_sessions(store):
    page = await store.list_sessions(SessionQuery(limit=1000, order_by=SessionOrder.CREATED_AT_ASC))
    return page.sessions


async def _assert_matches_reference(app) -> None:
    store = app.session_store
    sessions = await _all_sessions(store)
    served_sessions, served = await _service(app).read(list(sessions))
    reference_sessions, reference = await _service(app).read_reference(list(sessions))
    assert [s.id for s in served_sessions] == [s.id for s in reference_sessions]
    assert [row.model_dump(mode="json") for row in served] == [
        row.model_dump(mode="json") for row in reference
    ]
    summaries = _service(app).summaries
    if await summaries.documents(()) is None:
        return  # The store keeps no summary rows; read() was the reference read.
    for session in sessions:
        check = await check_session_summary(summaries, session.id)
        assert check.matches, (check.stored, check.recomputed)
    # A second read serves every row from storage without deriving it again.
    for session in sessions:
        check = await check_session_summary(summaries, session.id)
        assert check.current, session.id


async def _pending_kinds(store, session_id: str) -> set[PendingActionKind]:
    page = await store.query_pending_actions(PendingActionQuery(session_id=session_id))
    return {action.kind for action in page.actions}


async def _step(app, rng: random.Random, names: list[str], counter: list[int]) -> None:
    store = app.session_store
    existing = [session.id for session in await _all_sessions(store)]
    choice = rng.choice(
        [
            "start",
            "start",
            "child",
            "resume",
            "approve",
            "answer",
            "fork",
            "interrupt",
            "delete",
            "retention",
        ]
    )
    APPLIED.append(choice)
    if choice in ("start", "child") or not existing:
        counter[0] += 1
        name = rng.choice(names) if names and rng.random() < 0.2 else f"s{counter[0]}"
        if name in existing:
            await store.delete_session(name)
            existing.remove(name)
        parent = rng.choice(existing) if choice == "child" and existing else None
        await _drain(
            app.run(
                RunRequest(
                    agent_name=rng.choice(["writer", "asker", "approver"]),
                    session_id=name,
                    parent_session_id=parent,
                    messages=[Message.text("user", "go")],
                )
            )
        )
        names.append(name)
        return
    target = rng.choice(existing)
    kinds = await _pending_kinds(store, target)
    if choice == "approve" and PendingActionKind.TOOL_APPROVAL in kinds:
        action = await _pending(app, target)
        await _drain(
            app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=target,
                    approval_id=action.approval_id,
                    tool_round_id=action.round_id,
                    tool_call_id=action.tool_call_id,
                    decision=rng.choice(list(ToolApprovalDecision)),
                    resolved_by=REVIEWER,
                )
            )
        )
    elif choice == "answer" and PendingActionKind.USER_INPUT in kinds:
        action = await _pending(app, target)
        await _drain(
            app.resolve_user_input(
                UserInputResponse(session_id=target, input_id=action.input_id, answer=ANSWER)
            )
        )
    elif choice == "resume" and not kinds:
        session = await store.load(target)
        if session is not None and session.status.value == "completed":
            await _drain(
                app.resume(
                    ResumeRequest(session_id=target, messages=[Message.text("user", "again")])
                )
            )
    elif choice == "fork":
        counter[0] += 1
        await _drain(
            app.fork_session(
                ForkSessionRequest(source_session_id=target, session_id=f"f{counter[0]}")
            )
        )
    elif choice == "interrupt":
        session = await store.load(target)
        if session is not None and session.status.value in ("running", "interrupted"):
            await _drain(
                app.interrupt_session(
                    InterruptSessionRequest(
                        session_id=target, reason="Stopped.", requested_by=REVIEWER
                    )
                )
            )
    elif choice == "delete":
        await store.delete_session(target)
    elif choice == "retention":
        # Compaction deletes and rewrites events of every finished session.
        mode = rng.choice([RetentionMode.COMPACT, RetentionMode.DELETE])
        with contextlib.suppress(NotImplementedError):
            await store.apply_retention_policy(
                SessionRetentionPolicy(
                    older_than=timedelta(microseconds=1), mode=mode, dry_run=False
                )
            )


@pytest.mark.store_semantics
@pytest.mark.parametrize("seed", range(4))
def test_summary_rows_equal_the_reference_across_random_histories(session_store_case, seed):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            rng = random.Random(seed)
            names: list[str] = []
            counter = [0]
            for _ in range(16):
                # Some random operations are invalid for the session's current
                # state; the store must still summarize it.
                with contextlib.suppress(KeyError, ValueError, RuntimeError):
                    await _step(app, rng, names, counter)
                if rng.random() < 0.6:
                    await _assert_matches_reference(app)
            await _assert_matches_reference(app)
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_a_change_two_levels_down_invalidates_every_waiting_summary(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _delegating_app(store)
            await _start(app, "root", "root")
            await _assert_matches_reference(app)
            sessions = await _all_sessions(store)
            leaf = next(session.id for session in sessions if session.agent_name == "leaf")
            approval = await _pending(app, leaf)
            await _drain(
                app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=leaf,
                        approval_id=approval.approval_id,
                        tool_round_id=approval.round_id,
                        tool_call_id=approval.tool_call_id,
                        decision=ToolApprovalDecision.APPROVE,
                        resolved_by=REVIEWER,
                    )
                )
            )
            # The leaf now asks a question; root and middle report it from stored rows.
            await _assert_matches_reference(app)
            _, overviews = await _service(app).read(await _all_sessions(store))
            assert {row.state for row in overviews} == {"waiting_input"}
            question = await _pending(app, leaf)
            await _drain(
                app.resolve_user_input(
                    UserInputResponse(session_id=leaf, input_id=question.input_id, answer=ANSWER)
                )
            )
            await _assert_matches_reference(app)
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_usage_is_extended_from_the_stored_summary(session_store_case):
    if session_store_case[0] == "memory":
        pytest.skip("The in-memory store keeps no summary rows.")

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "talk", "writer")
            service = _service(app)
            _, (first,) = await service.read(await _all_sessions(store))
            (row,) = (await store.load_session_summaries(("talk",))).values()
            through = row.derived["usage_through"]
            assert through > 0
            await _drain(
                app.resume(
                    ResumeRequest(session_id="talk", messages=[Message.text("user", "more")])
                )
            )
            reads = []
            original = store.read_session_usage_deltas

            async def recording(afters):
                reads.append(dict(afters))
                return await original(afters)

            store.read_session_usage_deltas = recording
            try:
                _, (second,) = await service.read(await _all_sessions(store))
            finally:
                del store.read_session_usage_deltas
            assert reads == [{"talk": through}]
            assert int(second.tokens.input) == 2 * int(first.tokens.input)
            assert second.model_steps == 2 * first.model_steps
            await _assert_matches_reference(app)
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_a_stored_overview_is_current_only_while_its_row_and_reads_are(session_store_case):
    if session_store_case[0] == "memory":
        pytest.skip("The in-memory store keeps no summary rows.")

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "parent", "writer")
            await _drain(
                app.run(
                    RunRequest(
                        agent_name="writer",
                        session_id="child",
                        parent_session_id="parent",
                        messages=[Message.text("user", "go")],
                    )
                )
            )
            rows = await store.load_session_summaries(("parent", "child"))
            parent, child = rows["parent"], rows["child"]

            def settlement(row, generation=None, **dependencies):
                return SummarySettlement(
                    session_id=row.session_id,
                    instance_id=row.instance_id,
                    generation=row.source_generation if generation is None else generation,
                    state="completed",
                    forked=False,
                    derived={"version": 0},
                    dependencies=dependencies,
                )

            # A generation the row has moved past refuses the settlement.
            assert (
                await store.settle_session_summaries(
                    [settlement(parent, generation=parent.source_generation - 1)]
                )
                == frozenset()
            )
            # A dependency whose stamp moved since it was read is stored, but the
            # stored overview is not current.
            stale = (parent.instance_id, parent.source_generation - 1)
            assert await store.settle_session_summaries(
                [settlement(child, parent=stale), settlement(parent)]
            ) == {"child", "parent"}
            rows = await store.load_session_summaries(("parent", "child"))
            assert rows["parent"].settled and not rows["child"].settled
            assert await store.settle_session_summaries(
                [settlement(child, parent=parent.stamp, gone=None)]
            ) == {"child"}
            assert (await store.load_session_summaries(("child",)))["child"].settled
            # A write to a dependency makes its dependents' overviews not current,
            # without the writer touching their rows.
            child = (await store.load_session_summaries(("child",)))["child"]
            await _drain(
                app.resume(
                    ResumeRequest(session_id="parent", messages=[Message.text("user", "more")])
                )
            )
            again = (await store.load_session_summaries(("child",)))["child"]
            assert again.source_generation == child.source_generation
            assert not again.settled
            assert await store.settle_session_summaries([settlement(again, gone=None)]) == {"child"}
            # A session appearing where none was read makes them not current too.
            await _start(app, "gone", "writer")
            assert not (await store.load_session_summaries(("child",)))["child"].settled
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_a_page_of_stale_rows_depends_only_on_each_tree(session_store_case):
    if session_store_case[0] == "memory":
        pytest.skip("The in-memory store keeps no summary rows.")

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "alone", "writer")
            await _start(app, "parent", "writer")
            await _drain(
                app.run(
                    RunRequest(
                        agent_name="writer",
                        session_id="child",
                        parent_session_id="parent",
                        messages=[Message.text("user", "go")],
                    )
                )
            )
            settles = []
            original = store.settle_session_summaries

            async def recording(settlements):
                settles.append({item.session_id: set(item.dependencies) for item in settlements})
                return await original(settlements)

            store.settle_session_summaries = recording
            try:
                await _service(app).read(await _all_sessions(store))
            finally:
                del store.settle_session_summaries
            # Every stale row of the page settles in one call, each depending
            # only on the sessions of its own tree.
            assert settles == [{"alone": set(), "parent": {"child"}, "child": {"parent"}}]
            # A write to one tree leaves the other's row settled.
            await _drain(
                app.resume(
                    ResumeRequest(session_id="parent", messages=[Message.text("user", "more")])
                )
            )
            rows = await store.load_session_summaries(("alone", "parent", "child"))
            assert rows["alone"].settled
            assert not rows["parent"].settled and not rows["child"].settled
            await _assert_matches_reference(app)
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_postgres_advances_a_row_once_per_transaction(session_store_case):
    if session_store_case[0] != "postgres":
        pytest.skip("SQLite updates a row in place; Postgres keeps each version.")
    import psycopg

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "talk", "writer")
            await _start(app, "reader", "writer")
            await store.settle_session_summaries(
                [
                    SummarySettlement(
                        session_id="reader",
                        instance_id=(
                            row := (await store.load_session_summaries(("reader",)))["reader"]
                        ).instance_id,
                        generation=row.source_generation,
                        state="completed",
                        forked=False,
                        derived={"version": 0},
                        dependencies={
                            "talk": (await store.load_session_summary_stamps(("talk",)))["talk"]
                        },
                    )
                ]
            )
            before = await store.load_session_summaries(("talk", "reader"))
            async with (
                await psycopg.AsyncConnection.connect(session_store_case[2]) as conn,
                conn.transaction(),
                conn.cursor() as cur,
            ):
                await cur.execute(
                    "SELECT sequence FROM cayu_events WHERE session_id = %s ORDER BY sequence",
                    ("talk",),
                )
                sequences = [row[0] for row in await cur.fetchall()]
                assert len(sequences) > 2
                # A rewrite one event at a time, as retention compaction does.
                for sequence in sequences:
                    await cur.execute(
                        "UPDATE cayu_events SET payload = payload || '{\"probe\": 1}'::jsonb "
                        "WHERE sequence = %s",
                        (sequence,),
                    )
            after = await store.load_session_summaries(("talk", "reader"))
            assert after["talk"].source_generation == before["talk"].source_generation + 1
            assert after["talk"].usage_epoch == before["talk"].usage_epoch + 1
            # The reader's row is untouched; its overview is no longer current.
            assert before["reader"].settled
            assert after["reader"].source_generation == before["reader"].source_generation
            assert not after["reader"].settled
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_a_check_retries_a_session_that_changes_and_reports_one_that_keeps_changing(
    session_store_case,
):
    if session_store_case[0] == "memory":
        pytest.skip("The in-memory store keeps no summary rows.")

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "talk", "writer")
            summaries = _service(app).summaries
            original = summaries.recompute
            changes = []

            async def changing(session_id, *, times):
                # A write lands between the stored document and the recompute.
                if len(changes) < times:
                    changes.append(session_id)
                    await store.update_metadata(session_id, {"probe": len(changes)})
                return await original(session_id)

            summaries.recompute = lambda session_id: changing(session_id, times=1)
            check = await check_session_summary(summaries, "talk")
            assert check.stable and check.matches and changes == ["talk"]
            changes.clear()
            summaries.recompute = lambda session_id: changing(session_id, times=99)
            check = await check_session_summary(summaries, "talk")
            assert not check.stable and check.matches
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_postgres_writers_to_related_sessions_never_wait_on_each_other(session_store_case):
    if session_store_case[0] != "postgres":
        pytest.skip("Only Postgres writers can wait on each other's row locks.")
    import psycopg

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "parent", "writer")
            for child in ("a", "b"):
                await _drain(
                    app.run(
                        RunRequest(
                            agent_name="writer",
                            session_id=child,
                            parent_session_id="parent",
                            messages=[Message.text("user", "go")],
                        )
                    )
                )
            # The page settles every row of the tree, each depending on the others.
            await _service(app).read(await _all_sessions(store))
            assert all(
                row.settled for row in (await store.load_session_summaries(("a", "b"))).values()
            )
            dsn = session_store_case[2]
            rewrite = (
                "UPDATE cayu_events SET payload = payload || '{\"probe\": 1}'::jsonb "
                "WHERE sequence = (SELECT MAX(sequence) FROM cayu_events WHERE session_id = %s)"
            )
            async with (
                await psycopg.AsyncConnection.connect(dsn) as first,
                await psycopg.AsyncConnection.connect(dsn) as second,
            ):
                # One writer holds a's summary row; the other writes b and the parent.
                await first.execute(rewrite, ("a",))
                await second.execute("SET lock_timeout = '2s'")
                await second.execute(rewrite, ("b",))
                await second.execute(rewrite, ("parent",))
                await second.commit()
                await first.commit()
            rows = await store.load_session_summaries(("parent", "a", "b"))
            assert not any(row.settled for row in rows.values())
            await _assert_matches_reference(app)
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_postgres_usage_is_bounded_per_session_not_by_other_open_writers(session_store_case):
    if session_store_case[0] != "postgres":
        pytest.skip("Only Postgres has open writers below the visible sequence.")
    import psycopg

    from cayu.sessions.event_queries import EventQuery

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "early", "writer")
            dsn = session_store_case[2]
            async with await psycopg.AsyncConnection.connect(dsn) as writer:
                # An open event writer holds the reservation its statement trigger
                # takes at the current high-water, as cayu_reserve_event_sequence does.
                await writer.execute(
                    "SELECT pg_advisory_xact_lock_shared("
                    "((((('cayu_events'::regclass)::oid::bigint % 32768) << 16) "
                    "| (high_water >> 32))::integer), "
                    "(((high_water & 4294967295) - CASE WHEN (high_water & 4294967295) "
                    ">= 2147483648 THEN 4294967296 ELSE 0 END)::integer)) "
                    "FROM (SELECT cayu_event_sequence_high_water('cayu_events'::regclass) "
                    "AS high_water) AS reserved"
                )
                await _start(app, "late", "writer")
                async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as probe:
                    visible = (
                        await (
                            await probe.execute(
                                "SELECT cayu_event_visible_sequence('cayu_events'::regclass)"
                            )
                        ).fetchone()
                    )[0]
                    head = (
                        await (
                            await probe.execute(
                                "SELECT MAX(sequence) FROM cayu_events WHERE session_id = 'late'"
                            )
                        ).fetchone()
                    )[0]
                assert visible < head
                deltas = await store.read_session_usage_deltas({"late": 0})
                await writer.rollback()
            reference = await store.read_usage_accounting(
                EventQuery(session_ids=("late",)), by_session=True
            )
            assert deltas.through_sequences == {"late": head}
            assert deltas.summaries["late"] == reference.session_summaries[0]
            await _assert_matches_reference(app)
        finally:
            await _close_store(store)

    asyncio.run(scenario())
