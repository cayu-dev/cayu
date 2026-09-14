"""Fork admission composes with durable closure lineage ownership."""

import asyncio

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    ForkSessionRequest,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
)
from cayu.runtime.session_closure import (
    SessionClosureChildPolicy,
    SessionClosureDisposition,
    SessionClosurePolicy,
    SessionClosureRecord,
)
from cayu.sessions.base import InMemorySessionStore, fork_session_invocation
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("target", ["root", "child"])
def test_closure_rejects_public_and_direct_forks_before_mutation(
    tmp_path, request, backend, target
):
    if backend == "postgres":
        dsn = request.getfixturevalue("postgres_dsn")

    async def run():
        if backend == "memory":
            store = competitor = InMemorySessionStore()
        elif backend == "sqlite":
            path = tmp_path / "closure-forks.sqlite"
            store, competitor = SQLiteSessionStore(path), SQLiteSessionStore(path)
        else:
            from cayu.storage.migrations import SchemaMode
            from cayu.storage.postgres import PostgresSessionStore

            store = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
            competitor = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)

        entered, release = asyncio.Event(), asyncio.Event()

        class Dependent:
            store_id = "fork-barrier"

            async def inspect_session_closure(self, session_id, *, policy):
                return SessionClosureRecord(
                    store_id=self.store_id,
                    record_class="records",
                    disposition=SessionClosureDisposition.OWNED_ELIGIBLE,
                )

            async def erase_session_closure(self, session_id, *, policy, plan_id):
                entered.set()
                await release.wait()
                return SessionClosureRecord(
                    store_id=self.store_id,
                    record_class="records",
                    disposition=SessionClosureDisposition.ERASED,
                )

            async def export_session_closure(self, session_id, *, policy):
                return {"records": []}

        provider = ScriptedModelProvider(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]]
        )
        app = CayuApp(
            enable_logging=False, session_store=store, session_closure_stores=(Dependent(),)
        )
        other = CayuApp(enable_logging=False, session_store=competitor)
        for runtime in (app, other):
            runtime.register_provider(provider, default=True)
            runtime.register_agent(AgentSpec(name="worker", model="scripted-model"))
        closing = None
        root_id, child_id = f"fork-closure-root-{target}", f"fork-closure-child-{target}"
        try:
            async for _ in app.run(
                RunRequest(
                    session_id=root_id, agent_name="worker", messages=[Message.text("user", "go")]
                )
            ):
                pass
            # Normal fork creation must still work before closure admission.
            prior = ForkSessionRequest(source_session_id=root_id, session_id=child_id)
            prior_events = [event async for event in other.fork_session(prior)]
            assert await competitor.load(child_id) is not None
            closing = asyncio.create_task(
                app.erase_session_closure(
                    root_id,
                    policy=SessionClosurePolicy(child_policy=SessionClosureChildPolicy.RECURSIVE),
                )
            )
            await asyncio.wait_for(entered.wait(), 10)
            source_id = root_id if target == "root" else child_id
            source = await competitor.load(source_id)
            assert source is not None
            before = await competitor.load_session_closure_records(
                source_id, max_records=1000, max_bytes=1_000_000
            )
            # Existing exact fork acknowledgement is read-only and remains replayable.
            replayed = [event async for event in other.fork_session(prior)]
            assert [event.id for event in replayed] == [event.id for event in prior_events]
            with pytest.raises(ValueError, match="owned by"):
                async for _ in other.fork_session(
                    ForkSessionRequest(source_session_id=source_id, session_id="late-public-fork")
                ):
                    pass

            def forbidden(*args):
                raise AssertionError("Rejected forks must not invoke transformation callbacks.")

            for method in ("create_fork", "create_fork_with_transcript_validation"):
                fork_id = f"late-{method}"
                fork = source.model_copy(
                    update={
                        "id": fork_id,
                        "parent_session_id": source_id,
                        "invocation": fork_session_invocation(source),
                    }
                )
                with pytest.raises(ValueError, match="owned by"):
                    await getattr(competitor, method)(
                        source_session_id=source_id,
                        fork=fork,
                        source_statuses={source.status},
                        expected_source_run_epoch=source.run_epoch,
                        transcript_cursor=None,
                        checkpoint_transform=forbidden,
                        operation_initializer=forbidden,
                        **(
                            {"transcript_validator": forbidden}
                            if method.endswith("validation")
                            else {}
                        ),
                    )
                assert await competitor.load(fork_id) is None
            assert await competitor.load("late-public-fork") is None
            assert (
                await competitor.load_session_closure_records(
                    source_id, max_records=1000, max_bytes=1_000_000
                )
                == before
            )
            release.set()
            assert (await closing).complete
            assert await store.load(root_id) is None
            assert await store.load(child_id) is None
        finally:
            release.set()
            if closing is not None:
                if not closing.done():
                    closing.cancel()
                await asyncio.gather(closing, return_exceptions=True)
            if not isinstance(competitor, InMemorySessionStore):
                await competitor.close()
            if not isinstance(store, InMemorySessionStore):
                await store.close()

    asyncio.run(run())
