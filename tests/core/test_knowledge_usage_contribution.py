"""Cloud and local readers share namespace-scoped, content-free usage evidence."""

from cayu.events import Event, EventType


def test_knowledge_usage_contribution_preserves_identity_and_deduplicates():
    from cayu.knowledge.usage import knowledge_usage_contribution

    event = Event(
        type=EventType.KNOWLEDGE_LOOKUP,
        session_id="s",
        payload={"namespace": "ns", "entry_ids": ["one", "one", 1, "two"]},
    )
    assert knowledge_usage_contribution(event) == ("ns", ("one", "two"))
    assert (
        knowledge_usage_contribution(
            event.model_copy(update={"payload": {"namespace": 1, "entry_ids": ["one"]}})
        )
        is None
    )
    assert (
        knowledge_usage_contribution(
            event.model_copy(update={"payload": {"namespace": "ns", "entry_ids": "one"}})
        )
        is None
    )
    assert (
        knowledge_usage_contribution(event.model_copy(update={"type": EventType.SESSION_STARTED}))
        is None
    )


def test_retained_usage_and_cloud_share_deduplication_without_namespace_collision():
    import asyncio

    from tests.core.test_cloud_inventory import _usage_app

    from cayu.cloud_inventory import _cloud_usage_report, _Registered
    from cayu.sessions.knowledge_usage import retained_knowledge_usage

    async def scenario():
        app = await _usage_app()
        # The existing runtime fixture contains one default/seed lookup with
        # duplicate IDs. A same-ID other-namespace lookup must remain distinct.
        await app.session_store.append_event(
            "session",
            Event(
                type=EventType.KNOWLEDGE_LOOKUP,
                session_id="session",
                payload={"namespace": "other", "entry_ids": ["seed", "seed"]},
            ),
        )
        totals, through = await retained_knowledge_usage(
            app.session_store, {("default", "seed"), ("other", "seed")}
        )
        assert through > 0
        assert totals[("default", "seed")][0] == 1
        assert totals[("other", "seed")][0] == 1
        assert totals[("default", "seed")][1] is not None
        report, _ = await _cloud_usage_report(
            app, _Registered(knowledge=frozenset({("default", "seed")})), after=None
        )
        assert report is not None
        assert [(row.id, row.lookups) for row in report.knowledge] == [("seed", 1)]
        await app.aclose()

    asyncio.run(scenario())


def test_retained_lookup_totals_cross_multiple_bounded_pages():
    import asyncio

    from tests.core.test_cloud_inventory import _usage_app

    from cayu.sessions.knowledge_usage import retained_knowledge_usage

    async def scenario():
        app = await _usage_app()
        for _ in range(205):
            await app.session_store.append_event(
                "session",
                Event(
                    type=EventType.KNOWLEDGE_LOOKUP,
                    session_id="session",
                    payload={"namespace": "default", "entry_ids": ["seed", "seed"]},
                ),
            )
        totals, _ = await retained_knowledge_usage(app.session_store, {("default", "seed")})
        assert totals[("default", "seed")][0] == 206
        await app.aclose()

    asyncio.run(scenario())


def test_retained_usage_cancellation_propagates_without_partial_result():
    import asyncio

    from tests.core.test_cloud_inventory import _usage_app

    from cayu.sessions.knowledge_usage import retained_knowledge_usage

    async def scenario():
        app = await _usage_app()
        original = app.session_store.fold_event_snapshot
        entered = asyncio.Event()
        released = asyncio.Event()

        async def blocked(query, consume, *, max_bytes):
            through = await original(query, consume, max_bytes=max_bytes)
            entered.set()
            await released.wait()
            return through

        app.session_store.fold_event_snapshot = blocked
        owner = asyncio.create_task(
            retained_knowledge_usage(app.session_store, {("default", "seed")})
        )
        await entered.wait()
        owner.cancel()
        assert owner.cancelling() == 1
        try:
            await owner
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancelled read published totals")
        assert owner.cancelled()
        assert owner.cancelling() == 1
        app.session_store.fold_event_snapshot = original
        totals, _ = await retained_knowledge_usage(app.session_store, {("default", "seed")})
        assert totals[("default", "seed")][0] == 1
        await app.aclose()

    asyncio.run(scenario())


def test_retained_usage_excludes_append_after_captured_frontier():
    import asyncio

    from tests.core.test_cloud_inventory import _usage_app

    from cayu.sessions.knowledge_usage import retained_knowledge_usage

    async def scenario():
        app = await _usage_app()
        original = app.session_store.fold_event_snapshot

        async def append_after_capture(query, consume, *, max_bytes):
            through = await original(query, consume, max_bytes=max_bytes)
            await app.session_store.append_event(
                "session",
                Event(
                    type=EventType.KNOWLEDGE_LOOKUP,
                    session_id="session",
                    payload={"namespace": "default", "entry_ids": ["seed"]},
                ),
            )
            return through

        app.session_store.fold_event_snapshot = append_after_capture
        first, frontier = await retained_knowledge_usage(app.session_store, {("default", "seed")})
        assert first[("default", "seed")][0] == 1
        app.session_store.fold_event_snapshot = original
        later, newer = await retained_knowledge_usage(app.session_store, {("default", "seed")})
        assert later[("default", "seed")][0] == 2
        assert newer > frontier
        await app.aclose()

    asyncio.run(scenario())
