from __future__ import annotations

import asyncio

import pytest
from tests.core.test_workspace_mutation_receipts import (
    _ExclusiveWriterBinding,
    _portable_environment_spec,
    _PublicWorkspaceWriteTool,
    _SingleToolProvider,
    collect_events,
)

from cayu.artifacts import LocalArtifactStore
from cayu.core import AgentSpec, EventType, Message
from cayu.environments import Environment
from cayu.runtime import CayuApp, EventQuery, InMemorySessionStore, RunRequest
from cayu.runtime.workspace_checkpoints import WORKSPACE_CHECKPOINTS_KEY
from cayu.workspaces import LocalWorkspace
from cayu.workspaces.checkpoints import WorkspaceCheckpointPolicy


@pytest.mark.parametrize("fail_write", [False, True])
def test_tool_success_is_published_only_after_checkpoint(tmp_path, fail_write):
    async def run():
        root = tmp_path / "workspace"
        root.mkdir()

        class FailingStore(LocalArtifactStore):
            async def put_bytes(self, content, **kwargs):
                if fail_write and (root / "created.txt").exists():
                    raise OSError("checkpoint unavailable")
                return await super().put_bytes(content, **kwargs)

        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _SingleToolProvider(
                tool_name="public_workspace_write", arguments={"path": "created.txt"}
            ),
            default=True,
        )
        spec = _portable_environment_spec("local")
        spec.workspace_checkpoint_policy = WorkspaceCheckpointPolicy()
        app.register_environment(
            Environment(
                spec,
                workspace=LocalWorkspace(root, workspace_id="owned"),
                binding=_ExclusiveWriterBinding(),
                artifact_store=FailingStore(tmp_path / "artifacts"),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"), tools=[_PublicWorkspaceWriteTool()]
        )
        try:
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="session-checkpoints",
                    messages=[Message.text("user", "create")],
                ),
            )
        except Exception:
            if not fail_write:
                raise
        records = await store.query_events(EventQuery(session_id="session-checkpoints"))
        checkpoint = await store.load_checkpoint("session-checkpoints")
        return [record.event for record in records], checkpoint

    events, checkpoint = asyncio.run(run())
    receipt = checkpoint[WORKSPACE_CHECKPOINTS_KEY]["local"]
    if fail_write:
        assert receipt["phase"] == "checkpointing"
        assert not any(event.type == EventType.TOOL_CALL_COMPLETED for event in events)
    else:
        assert receipt["phase"] == "durable"
        assert receipt["tool_call_id"] is not None
        complete = next(i for i, e in enumerate(events) if e.type == EventType.TOOL_CALL_COMPLETED)
        durable = next(
            i
            for i, e in enumerate(events)
            if e.type == EventType.WORKSPACE_CHECKPOINT_UPDATED
            and e.payload["phase"] == "durable"
            and e.payload["tool_call_id"] is not None
        )
        assert durable < complete


def _registered_checkpoint_environment(tmp_path, generation="first"):
    from cayu.runtime._runtime_records import RegisteredEnvironment

    root = tmp_path / generation
    root.mkdir(exist_ok=True)
    spec = _portable_environment_spec("local")
    spec.workspace_checkpoint_policy = WorkspaceCheckpointPolicy()
    return RegisteredEnvironment(
        spec=spec,
        environment=Environment(
            spec,
            workspace=LocalWorkspace(root),
            binding=_ExclusiveWriterBinding(),
            artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        ),
        binding_generation_id=generation,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_unknown_mutation_blocks_fresh_environment_and_cas_rejects_stale_epoch(tmp_path, backend):
    from cayu.runtime.sessions import SessionIdentity
    from cayu.runtime.workspace_checkpoints import (
        begin_workspace_checkpoint_mutation,
        ensure_workspace_checkpoint,
    )
    from cayu.storage.sqlite import SQLiteSessionStore
    from cayu.workspaces.checkpoints import WorkspaceCheckpointError

    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.db")
        )
        session = await store.create(
            RunRequest(agent_name="a", session_id="unknown", messages=[]),
            identity=SessionIdentity(provider_name="test", model="test"),
        )
        registered = _registered_checkpoint_environment(tmp_path)
        await ensure_workspace_checkpoint(store, session, registered)
        await begin_workspace_checkpoint_mutation(
            store, session, registered, window_id="w1", tool_call_id="t1", interaction_id=None
        )
        fresh = _registered_checkpoint_environment(tmp_path, "fresh")
        with pytest.raises(WorkspaceCheckpointError, match="unknown"):
            await ensure_workspace_checkpoint(store, session, fresh)
        # A stale epoch cannot publish a fresh baseline over a current session.
        other = await store.create(
            RunRequest(agent_name="a", session_id="stale", messages=[]),
            identity=SessionIdentity(provider_name="test", model="test"),
        )
        with pytest.raises(Exception):
            await ensure_workspace_checkpoint(
                store, other.model_copy(update={"run_epoch": 100}), fresh
            )
        assert WORKSPACE_CHECKPOINTS_KEY not in (await store.load_checkpoint("stale") or {})

    asyncio.run(run())


def test_live_drift_is_not_silently_rolled_back_and_fresh_restore_is_verified(tmp_path):
    from cayu.runtime.sessions import SessionIdentity
    from cayu.runtime.workspace_checkpoints import ensure_workspace_checkpoint
    from cayu.workspaces.checkpoints import WorkspaceCheckpointError

    async def run():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(agent_name="a", session_id="drift", messages=[]),
            identity=SessionIdentity(provider_name="test", model="test"),
        )
        registered = _registered_checkpoint_environment(tmp_path)
        await registered.environment.workspace.write_bytes("a", b"durable")
        await ensure_workspace_checkpoint(store, session, registered)
        await registered.environment.workspace.write_bytes("a", b"external")
        with pytest.raises(WorkspaceCheckpointError, match="outside"):
            await ensure_workspace_checkpoint(store, session, registered)
        assert (await registered.environment.workspace.read_bytes("a")).content == b"external"
        fresh = _registered_checkpoint_environment(tmp_path, "fresh")
        await ensure_workspace_checkpoint(store, session, fresh)
        assert (await fresh.environment.workspace.read_bytes("a")).content == b"durable"
        receipt = (await store.load_checkpoint(session.id))[WORKSPACE_CHECKPOINTS_KEY]["local"]
        assert receipt["binding_generation_id"] == "fresh"

    asyncio.run(run())


def test_checkpoint_timeout_retains_environment_until_write_settles(tmp_path):
    from cayu.runtime.sessions import SessionIdentity
    from cayu.runtime.workspace_checkpoints import ensure_workspace_checkpoint
    from cayu.workspaces.checkpoints import WorkspaceCheckpointError

    async def run():
        release = asyncio.Event()

        class DelayedStore(LocalArtifactStore):
            async def put_bytes(self, content, **kwargs):
                await release.wait()
                return await super().put_bytes(content, **kwargs)

        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(agent_name="a", session_id="deadline", messages=[]),
            identity=SessionIdentity(provider_name="test", model="test"),
        )
        registered = _registered_checkpoint_environment(tmp_path)
        registered.spec.workspace_checkpoint_policy = WorkspaceCheckpointPolicy(timeout_seconds=1)
        registered.environment.artifact_store = DelayedStore(tmp_path / "delayed")
        with pytest.raises(WorkspaceCheckpointError, match="deadline"):
            await ensure_workspace_checkpoint(store, session, registered)
        with pytest.raises(Exception):
            registered.workspace_mutation_fence.require_available_nowait()
        release.set()
        await registered.workspace_mutation_fence.wait_until_available()
        registered.workspace_mutation_fence.require_available_nowait()

    asyncio.run(run())


def test_checkpoint_authority_is_not_imported_from_old_schema_or_generic_writes():
    from cayu.runtime._session_engine import _replace_checkpoint_preserving_runtime_state
    from cayu.runtime.checkpoints import decode_runtime_checkpoint, runtime_checkpoint_writer_view

    old = {"schema_version": 7, "workspace_checkpoints": {"forged": {}}}
    # Use the actual root schema key; caller-looking data from an older writer is dropped.
    from cayu.runtime.checkpoints import (
        CHECKPOINT_SCHEMA_VERSION_KEY,
        CURRENT_CHECKPOINT_SCHEMA_VERSION,
    )

    old = {CHECKPOINT_SCHEMA_VERSION_KEY: 7, "workspace_checkpoints": {"forged": {}}}
    assert "workspace_checkpoints" not in decode_runtime_checkpoint(old, session_id="s")
    current = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "workspace_checkpoints": {"real": {}},
    }
    with pytest.raises(ValueError, match="older"):
        runtime_checkpoint_writer_view(current, writer_version=7, session_id="s")
    replaced = _replace_checkpoint_preserving_runtime_state(
        {"workspace_checkpoints": {"forged": {}}}
    )(None, current)
    assert replaced["workspace_checkpoints"] == {"real": {}}


def test_postgres_checkpoint_publication_recovers_and_rejects_stale_owner(tmp_path, postgres_dsn):
    from uuid import uuid4

    from cayu.runtime.sessions import SessionIdentity
    from cayu.runtime.workspace_checkpoints import (
        begin_workspace_checkpoint_mutation,
        complete_workspace_checkpoint_mutation,
        ensure_workspace_checkpoint,
    )
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            session = await store.create(
                RunRequest(agent_name="a", session_id="checkpoint-" + uuid4().hex, messages=[]),
                identity=SessionIdentity(provider_name="test", model="test"),
            )
            registered = _registered_checkpoint_environment(tmp_path)
            await ensure_workspace_checkpoint(store, session, registered)
            await begin_workspace_checkpoint_mutation(
                store, session, registered, window_id="w", tool_call_id="t", interaction_id=None
            )
            await registered.environment.workspace.write_bytes("a", b"committed")
            await complete_workspace_checkpoint_mutation(
                store, session, registered, window_id="w", successful=True
            )
            fresh = _registered_checkpoint_environment(tmp_path, "fresh")
            await ensure_workspace_checkpoint(store, session, fresh)
            assert (await fresh.environment.workspace.read_bytes("a")).content == b"committed"
            with pytest.raises(Exception):
                await begin_workspace_checkpoint_mutation(
                    store,
                    session.model_copy(update={"run_epoch": 100}),
                    fresh,
                    window_id="stale",
                    tool_call_id="stale",
                    interaction_id=None,
                )
            receipt = (await store.load_checkpoint(session.id))[WORKSPACE_CHECKPOINTS_KEY]["local"]
            assert receipt["window_id"] == "w" and receipt["phase"] == "durable"
        finally:
            await store.close()

    asyncio.run(run())


def test_reacquired_exclusive_lease_can_mutate_the_verified_revision(tmp_path):
    from cayu.runtime.sessions import SessionIdentity
    from cayu.runtime.workspace_checkpoints import (
        begin_workspace_checkpoint_mutation,
        complete_workspace_checkpoint_mutation,
        ensure_workspace_checkpoint,
    )
    from cayu.workspaces.revisions import WorkspaceWriterIsolationEvidence

    class NextLease(_ExclusiveWriterBinding):
        def observe_writer_isolation(self, bound):
            return WorkspaceWriterIsolationEvidence(
                status="exclusive",
                mechanism="test-held-writer-lease",
                generation="next-lease",
                detail_code=None,
            )

    async def run():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(agent_name="a", session_id="lease", messages=[]),
            identity=SessionIdentity(provider_name="test", model="test"),
        )
        registered = _registered_checkpoint_environment(tmp_path)
        await ensure_workspace_checkpoint(store, session, registered)
        registered.environment.binding = NextLease()
        await begin_workspace_checkpoint_mutation(
            store, session, registered, window_id="new-lease", tool_call_id="t", interaction_id=None
        )
        await registered.environment.workspace.write_bytes("a", b"new")
        await complete_workspace_checkpoint_mutation(
            store, session, registered, window_id="new-lease", successful=True
        )
        receipt = (await store.load_checkpoint(session.id))[WORKSPACE_CHECKPOINTS_KEY]["local"]
        assert receipt["isolation_generation"] == "next-lease" and receipt["phase"] == "durable"

    asyncio.run(run())


def test_durable_directory_replacement_recovers_into_seeded_binding(tmp_path):
    from cayu.runtime.sessions import SessionIdentity
    from cayu.runtime.workspace_checkpoints import (
        begin_workspace_checkpoint_mutation,
        complete_workspace_checkpoint_mutation,
        ensure_workspace_checkpoint,
    )

    async def run():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(agent_name="a", session_id="directory-replacement", messages=[]),
            identity=SessionIdentity(provider_name="test", model="test"),
        )
        registered = _registered_checkpoint_environment(tmp_path)
        workspace = registered.environment.workspace
        await workspace.write_bytes("package/old.py", b"seed")
        await ensure_workspace_checkpoint(store, session, registered)
        await begin_workspace_checkpoint_mutation(
            store, session, registered, window_id="w", tool_call_id="t", interaction_id=None
        )
        old = await workspace.read_bytes("package/old.py")
        await workspace.delete_if_revision("package/old.py", expected_revision=old.revision)
        (tmp_path / "first/package").rmdir()
        await workspace.write_bytes("package", b"checkpointed file")
        await complete_workspace_checkpoint_mutation(
            store, session, registered, window_id="w", successful=True
        )
        receipt = (await store.load_checkpoint(session.id))[WORKSPACE_CHECKPOINTS_KEY]["local"]
        assert receipt["phase"] == "durable"
        fresh = _registered_checkpoint_environment(tmp_path, "fresh")
        await fresh.environment.workspace.write_bytes("package/old.py", b"seed")
        for _ in range(2):
            await ensure_workspace_checkpoint(store, session, fresh)
            assert (
                await fresh.environment.workspace.read_bytes("package")
            ).content == b"checkpointed file"
        restored = (await store.load_checkpoint(session.id))[WORKSPACE_CHECKPOINTS_KEY]["local"]
        assert restored["revision"] == receipt["revision"]
        assert restored["binding_generation_id"] == "fresh"

    asyncio.run(run())
