from __future__ import annotations

import asyncio
import io
import os
import tarfile
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from cayu.artifacts import LocalArtifactStore
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.dmtcp_snapshots import validate_snapshot_archive
from cayu.environments.snapshot_lifecycle import (
    EXECUTION_SNAPSHOTS_KEY,
    ExecutionSnapshots,
    ensure_execution_snapshot_binding,
)
from cayu.environments.snapshots import (
    CapturedExecutionSnapshot,
    ExecutionSnapshotAdapter,
    ExecutionSnapshotCapability,
    ExecutionSnapshotConflict,
    ExecutionSnapshotError,
    ExecutionSnapshotFidelity,
    ExecutionSnapshotOutcomeUnknown,
    ExecutionSnapshotPolicy,
)
from cayu.runtime._runtime_records import RegisteredEnvironment
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionIdentity, SessionStatus
from cayu.sessions.checkpoints import (
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    decode_runtime_checkpoint,
    runtime_checkpoint_writer_view,
)


class FakeAdapter(ExecutionSnapshotAdapter):
    def __init__(self, allocation="a", fail=None):
        self.allocation = allocation
        self.fail = fail
        self.calls = []
        self.active = False
        self.held = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def capability(self):
        return ExecutionSnapshotCapability(
            fidelity=ExecutionSnapshotFidelity.SELECTED_PROCESSES,
            adapter="fake",
            compatibility_sha256="f" * 64,
            snapshot_format="test-1",
        )

    @property
    def allocation_sha256(self):
        return self.allocation * 64

    async def capture(self, operation_id, policy, fence):
        await fence()
        self.calls.append("capture")
        self.held = True
        self.entered.set()
        if self.fail == "wait":
            await self.release.wait()
        if self.fail == "ack":
            raise OSError("lost acknowledgement")
        await fence()
        return CapturedExecutionSnapshot(process=b"memory", workspace=b"files")

    async def restore(self, operation_id, snapshot, policy, fence):
        await fence()
        self.calls.append("restore")
        assert snapshot == CapturedExecutionSnapshot(process=b"memory", workspace=b"files")
        self.held = True
        if self.fail == "ack":
            raise OSError("lost acknowledgement")
        await fence()

    async def activate(self, operation_id, fence):
        await fence()
        self.calls.append("activate")
        if self.fail == "activate":
            self.fail = None
            raise OSError("lost activation acknowledgement")
        self.active = True
        self.held = False

    async def resume_source(self, operation_id, fence):
        await fence()
        self.calls.append("resume")
        if self.fail == "resume":
            self.fail = None
            raise OSError("lost source-release acknowledgement")
        self.held = False

    async def recover_capture(self, operation_id, policy, fence):
        await fence()
        assert self.held
        self.calls.append("recover_capture")
        return CapturedExecutionSnapshot(process=b"memory", workspace=b"files")

    async def verify_restore(self, operation_id, policy, fence):
        await fence()
        assert self.held and not self.active
        self.calls.append("verify_restore")


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def backend(request):
    return request.param, (
        request.getfixturevalue("postgres_dsn") if request.param == "postgres" else None
    )


@asynccontextmanager
async def setup(tmp_path, backend, *, policy=None, clock=None):
    kind, dsn = backend
    if kind == "memory":
        store = InMemorySessionStore()
    elif kind == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        store = SQLiteSessionStore(tmp_path / "sessions.db")
    else:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        store = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE, min_size=1, max_size=4)
        await store.ensure_schema()
    try:
        session = await store.create(
            RunRequest(agent_name="agent", session_id=str(uuid4()), messages=[]),
            identity=SessionIdentity(provider_name="test", model="test"),
        )
        artifacts = LocalArtifactStore(tmp_path / "private-snapshots")
        controller = ExecutionSnapshots(store, artifacts, policy=policy, clock=clock)
        yield store, session, artifacts, controller
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            await close()


async def capture(controller, session, adapter=None, key="capture"):
    return await controller.capture(
        session.id,
        "sandbox",
        adapter or FakeAdapter(),
        expected_run_epoch=session.run_epoch,
        expected_generation="source",
        idempotency_key=key,
    )


async def restore(controller, session, record, adapter=None, key="restore", generation="target"):
    return await controller.restore(
        session.id,
        "sandbox",
        record.id,
        adapter or FakeAdapter("b"),
        expected_run_epoch=session.run_epoch,
        expected_generation="source",
        target_generation=generation,
        idempotency_key=key,
    )


def test_capture_restore_delete_and_repeated_identity(tmp_path, backend):
    async def run():
        async with setup(tmp_path, backend) as (store, session, artifacts, controller):
            source = FakeAdapter()
            record = await capture(controller, session, source)
            assert await capture(controller, session, source) == record
            assert source.calls == ["capture", "resume"]
            fresh = ExecutionSnapshots(store, artifacts)
            target = FakeAdapter("b")
            result = await restore(fresh, session, record, target)
            assert result.state == "succeeded" and target.active
            assert await restore(fresh, session, record, target) == result
            assert target.calls == ["restore", "activate"]
            await fresh.delete(
                session.id,
                "sandbox",
                record.id,
                expected_run_epoch=0,
                expected_generation="target",
                idempotency_key="delete",
            )
            await fresh.delete(
                session.id,
                "sandbox",
                record.id,
                expected_run_epoch=0,
                expected_generation="target",
                idempotency_key="delete",
            )
            inspected = await fresh.inspect(session.id, "sandbox")
            assert inspected["snapshots"][0]["retention"] == "deleted"
            with pytest.raises(ExecutionSnapshotError):
                await restore(fresh, session, record, FakeAdapter("c"), key="other")

    asyncio.run(run())


@pytest.mark.parametrize(
    "change",
    ["expired", "corrupt", "missing", "compatibility", "generation", "controller", "transcript"],
)
def test_restore_rejects_before_target_mutation(tmp_path, backend, change):
    async def run():
        now = datetime.now(UTC)
        async with setup(tmp_path, backend, clock=lambda: now) as (
            store,
            session,
            artifacts,
            controller,
        ):
            record = await capture(controller, session)
            target = FakeAdapter("b")
            if change == "expired":
                now += timedelta(days=2)
            elif change in {"corrupt", "missing"}:
                part = record.artifacts[0]
                await artifacts.release_pin(part.artifact_id, owner=record.pin_owner)
                await artifacts.delete(part.artifact_id)
                if change == "corrupt":
                    await artifacts.put_bytes(
                        b"bad", artifact_id=part.artifact_id, filename="bad", session_id=session.id
                    )
            elif change == "compatibility":

                class Incompatible(FakeAdapter):
                    @property
                    def capability(self):
                        return super().capability.model_copy(
                            update={"compatibility_sha256": "c" * 64}
                        )

                target = Incompatible("b")
            elif change == "generation":

                def mutate(_, checkpoint):
                    checkpoint[EXECUTION_SNAPSHOTS_KEY]["environments"]["sandbox"]["binding"][
                        "generation"
                    ] = "newer"
                    return checkpoint

                from cayu.sessions._checkpoint_preservation import (
                    _execution_snapshot_authority_mutation_scope,
                )

                with _execution_snapshot_authority_mutation_scope():
                    await store.transform_checkpoint(session.id, mutate)
            elif change == "controller":

                def advance(_, checkpoint):
                    checkpoint["effect_result"] = {"state": "completed", "count": 2}
                    return checkpoint

                await store.transform_checkpoint(session.id, advance)
            else:
                from cayu.messages import Message

                await store.append_transcript_messages(
                    session.id, [Message.text("user", "advance")]
                )
            with pytest.raises(ExecutionSnapshotError):
                await restore(controller, session, record, target)
            assert not target.calls and not target.active
            # A rejected restore leaves nothing that blocks the session.
            state = await controller.inspect(session.id, "sandbox")
            assert [op["state"] for op in state["operations"]] == ["succeeded"]

    asyncio.run(run())


def _advance_controller(_, checkpoint):
    checkpoint["effect_result"] = {"state": "completed", "count": 2}
    return checkpoint


def _registration(adapter=None, generation="source"):
    return RegisteredEnvironment(
        spec=EnvironmentSpec(name="sandbox"),
        environment=Environment(
            EnvironmentSpec(name="sandbox"), execution_snapshot_adapter=adapter
        ),
        binding_generation_id=generation,
    )


@pytest.mark.parametrize("failure", ["preflight", "deadline", "cancel"])
def test_unsubmitted_capture_is_dropped_and_retryable(tmp_path, backend, failure):
    async def run():
        class Gated(FakeAdapter):
            async def preflight_capture(self, policy):
                self.calls.append("preflight")
                if self.fail == "preflight":
                    raise ExecutionSnapshotError("Workspace is not eligible.")
                if self.fail == "wait":
                    self.entered.set()
                    await self.release.wait()

        policy = ExecutionSnapshotPolicy(timeout_seconds=1)
        async with setup(tmp_path, backend, policy=policy) as (store, session, _, controller):
            adapter = Gated(fail="preflight" if failure == "preflight" else "wait")
            if failure == "cancel":
                work = asyncio.create_task(capture(controller, session, adapter))
                await adapter.entered.wait()
                work.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await work
            else:
                with pytest.raises(ExecutionSnapshotError):
                    await capture(controller, session, adapter)
            state = await controller.inspect(session.id, "sandbox")
            assert state["operations"] == [] and state["binding"] is None
            assert "capture" not in adapter.calls
            await ensure_execution_snapshot_binding(store, session, _registration())
            adapter.fail = None
            adapter.release.set()
            await capture(controller, session, adapter)
            assert adapter.calls[-2:] == ["capture", "resume"]

    asyncio.run(run())


def test_lost_binding_is_released_explicitly(tmp_path, backend):
    async def run():
        async with setup(tmp_path, backend) as (store, session, _, controller):
            await capture(controller, session)
            # The session ran on past its only snapshot, then lost its allocation.
            await store.transform_checkpoint(session.id, _advance_controller)
            fresh = FakeAdapter("c")
            registered = _registration(fresh, "fresh")
            with pytest.raises(ExecutionSnapshotConflict, match="binding release"):
                await ensure_execution_snapshot_binding(store, session, registered)

            async def release(adapter, generation):
                await controller.release_binding(
                    session.id,
                    "sandbox",
                    adapter,
                    expected_run_epoch=0,
                    expected_generation="source",
                    target_generation=generation,
                )

            with pytest.raises(ExecutionSnapshotConflict, match="Delete retained"):
                await release(None, None)
            await release(fresh, "fresh")
            await release(fresh, "fresh")
            await ensure_execution_snapshot_binding(store, session, registered)
            record = await controller.capture(
                session.id,
                "sandbox",
                fresh,
                expected_run_epoch=0,
                expected_generation="fresh",
                idempotency_key="after-release",
            )
            assert record.allocation_sha256 == fresh.allocation_sha256

    asyncio.run(run())


def test_tombstones_compact_and_deletion_stays_admissible(tmp_path, backend):
    async def run():
        policy = ExecutionSnapshotPolicy(max_records=2)
        async with setup(tmp_path, backend, policy=policy) as (_, session, _, controller):

            async def delete(record, key):
                await controller.delete(
                    session.id,
                    "sandbox",
                    record.id,
                    expected_run_epoch=0,
                    expected_generation="source",
                    idempotency_key=key,
                )

            first = await capture(controller, session, key="c1")
            second = await capture(controller, session, key="c2")
            with pytest.raises(ExecutionSnapshotError, match="retention limit"):
                await capture(controller, session, key="c3")
            # Operation records are full; the oldest succeeded one makes room.
            await delete(first, "d1")
            await delete(first, "d1-again")
            third = await capture(controller, session, key="c3")
            state = await controller.inspect(session.id, "sandbox")
            assert len(state["operations"]) == 2
            assert all(op["state"] == "succeeded" for op in state["operations"])
            assert {record["id"] for record in state["snapshots"]} == {second.id, third.id}

    asyncio.run(run())


def test_fences_and_exposure_gate_avoid_full_checkpoint_io(tmp_path, backend):
    async def run():
        class Chatty(FakeAdapter):
            async def capture(self, operation_id, policy, fence):
                for _ in range(20):
                    await fence()
                return await super().capture(operation_id, policy, fence)

        async with setup(tmp_path, backend) as (store, session, _, controller):
            writes = []
            publish = store.publish_checkpoint_and_events

            async def counted(*args, **kwargs):
                writes.append(kwargs.get("events"))
                return await publish(*args, **kwargs)

            store.publish_checkpoint_and_events = counted
            await capture(controller, session, Chatty())
            # Reserve, submitted, verified and succeeded; fences never write.
            assert len(writes) == 4

            async def heavy_read(*args, **kwargs):
                raise AssertionError("The exposure gate must not load the full checkpoint.")

            store.load_checkpoint = heavy_read
            await ensure_execution_snapshot_binding(store, session, _registration(FakeAdapter()))

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["capture", "restore"])
def test_ack_loss_is_durable_unknown_and_never_replayed(tmp_path, backend, phase):
    async def run():
        async with setup(tmp_path, backend) as (store, session, artifacts, controller):
            adapter = FakeAdapter("a" if phase == "capture" else "b", fail="ack")
            record = None if phase == "capture" else await capture(controller, session)

            def invoke(service):
                return (
                    capture(service, session, adapter)
                    if phase == "capture"
                    else restore(service, session, record, adapter)
                )

            with pytest.raises(OSError):
                await invoke(controller)
            with pytest.raises(ExecutionSnapshotOutcomeUnknown):
                await invoke(ExecutionSnapshots(store, artifacts))
            assert adapter.calls == [phase] and not adapter.active
            state = await controller.inspect(session.id, "sandbox")
            assert state["operations"][-1]["state"] == "unknown"
            registered = RegisteredEnvironment(
                spec=EnvironmentSpec(name="sandbox"),
                environment=Environment(EnvironmentSpec(name="sandbox")),
                binding_generation_id="source",
            )
            with pytest.raises(ExecutionSnapshotOutcomeUnknown):
                await ensure_execution_snapshot_binding(store, session, registered)

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["resume", "activate"])
def test_committed_release_can_retry_without_repeating_capture_or_restore(tmp_path, backend, phase):
    async def run():
        async with setup(tmp_path, backend) as (store, session, artifacts, controller):
            adapter = FakeAdapter("a" if phase == "resume" else "b", fail=phase)
            record = None if phase == "resume" else await capture(controller, session)

            def invoke(service):
                return (
                    capture(service, session, adapter)
                    if phase == "resume"
                    else restore(service, session, record, adapter)
                )

            with pytest.raises(OSError):
                await invoke(controller)
            await invoke(ExecutionSnapshots(store, artifacts))
            assert adapter.calls == (
                ["capture", "resume", "resume"]
                if phase == "resume"
                else ["restore", "activate", "activate"]
            )

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["cancel", "fence"])
def test_pending_capture_blocks_competitors_and_stale_owner(tmp_path, backend, failure):
    async def run():
        async with setup(tmp_path, backend) as (store, session, artifacts, controller):
            adapter = FakeAdapter(fail="wait")
            work = asyncio.create_task(capture(controller, session, adapter))
            await adapter.entered.wait()
            with pytest.raises(ExecutionSnapshotOutcomeUnknown):
                await capture(
                    ExecutionSnapshots(store, artifacts), session, FakeAdapter(), key="competing"
                )
            if failure == "cancel":
                work.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await work
            else:
                await store.fence_run_and_transform_checkpoint(
                    session.id,
                    statuses={SessionStatus.PENDING},
                    checkpoint_transform=lambda _, checkpoint: checkpoint,
                )
                adapter.release.set()
                with pytest.raises(Exception):
                    await work
            state = await controller.inspect(session.id, "sandbox")
            assert not state["snapshots"] and state["operations"][0]["state"] in {
                "unknown",
                "submitted",
            }
            assert adapter.held

    asyncio.run(run())


def test_execution_position_preserves_completed_and_unknown_effects(tmp_path, backend):
    async def run():
        async with setup(tmp_path, backend) as (store, session, _, controller):
            await store.checkpoint(
                session.id,
                {
                    "completed_results": {"write": {"count": 1}},
                    "external_effect": {"state": "outcome_unknown"},
                },
            )
            record = await capture(controller, session)
            await restore(controller, session, record)
            checkpoint = await store.load_checkpoint(session.id)
            assert checkpoint["completed_results"]["write"]["count"] == 1
            assert checkpoint["external_effect"]["state"] == "outcome_unknown"
            # The controller is retained exactly; restoration never invokes tools.
            assert record.position.checkpoint_sha256

    asyncio.run(run())


def test_schema_does_not_promote_legacy_application_json():
    decoded = decode_runtime_checkpoint(
        {"checkpoint_schema_version": 10, EXECUTION_SNAPSHOTS_KEY: {"forged": "authority"}},
        session_id="s",
    )
    assert EXECUTION_SNAPSHOTS_KEY not in decoded
    assert decoded["checkpoint_schema_version"] == CURRENT_CHECKPOINT_SCHEMA_VERSION
    with pytest.raises(ValueError, match="snapshot authority"):
        runtime_checkpoint_writer_view(
            {
                "checkpoint_schema_version": CURRENT_CHECKPOINT_SCHEMA_VERSION,
                EXECUTION_SNAPSHOTS_KEY: {},
            },
            writer_version=10,
            session_id="s",
        )


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", "file"),
        ("/absolute", "file"),
        ("a", "symlink"),
        ("a", "hardlink"),
        ("image.dmtcp.temp", "file"),
    ],
)
def test_unsafe_archive_rejected(name, kind):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        entry = tarfile.TarInfo(name)
        if kind != "file":
            entry.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            entry.linkname = "target"
        archive.addfile(entry)
    with pytest.raises(ExecutionSnapshotError):
        validate_snapshot_archive(output.getvalue(), ExecutionSnapshotPolicy(), process=True)


@pytest.mark.parametrize("phase", ["capture", "restore"])
def test_reconciliation_fences_old_owner_without_resubmitting(tmp_path, backend, phase):
    async def run():
        async with setup(tmp_path, backend) as (store, session, artifacts, controller):
            adapter = FakeAdapter("a" if phase == "capture" else "b", fail="ack")
            record = None if phase == "capture" else await capture(controller, session)
            with pytest.raises(OSError):
                if phase == "capture":
                    await capture(controller, session, adapter)
                else:
                    await restore(controller, session, record, adapter)
            state = await controller.inspect(session.id, "sandbox")
            op = state["operations"][-1]
            assert op["position"] is not None
            await ExecutionSnapshots(store, artifacts).reconcile(
                session.id,
                "sandbox",
                op["id"],
                adapter,
                expected_run_epoch=0,
                expected_generation="source",
                idempotency_key=phase,
                target_generation="recovered" if phase == "restore" else "source",
            )
            assert (await store.load(session.id)).run_epoch == 1
            assert adapter.calls == (
                ["capture", "recover_capture", "resume"]
                if phase == "capture"
                else ["restore", "verify_restore", "activate"]
            )
            with pytest.raises(ExecutionSnapshotConflict):
                await capture(controller, session, FakeAdapter())

    asyncio.run(run())


@pytest.mark.parametrize("source_state", ["expired", "missing"])
@pytest.mark.parametrize("recovery", ["submitted", "verified", "retry", "terminal"])
def test_restore_settlement_outlives_source_snapshot(tmp_path, backend, source_state, recovery):
    async def run():
        now = datetime.now(UTC)
        async with setup(tmp_path, backend, clock=lambda: now) as (
            store,
            session,
            artifacts,
            controller,
        ):
            record = await capture(controller, session)
            failure = "ack" if recovery == "submitted" else "activate"
            target = FakeAdapter("b", fail=None if recovery == "terminal" else failure)
            if recovery == "terminal":
                await restore(controller, session, record, target)
            else:
                with pytest.raises(OSError):
                    await restore(controller, session, record, target)
            if source_state == "expired":
                now = record.expires_at + timedelta(seconds=1)
            else:
                part = record.artifacts[0]
                await artifacts.release_pin(part.artifact_id, owner=record.pin_owner)
                await artifacts.delete(part.artifact_id)

            fresh = ExecutionSnapshots(store, artifacts, clock=lambda: now)
            # Retained operation identity still authenticates every retry.
            with pytest.raises(ExecutionSnapshotConflict):
                await restore(fresh, session, record, FakeAdapter("c"))
            if recovery in {"retry", "terminal"}:
                result = await restore(fresh, session, record, target)
            else:
                state = await fresh.inspect(session.id, "sandbox")
                op = next(item for item in state["operations"] if item["kind"] == "restore")
                result = await fresh.reconcile(
                    session.id,
                    "sandbox",
                    op["id"],
                    target,
                    expected_run_epoch=session.run_epoch,
                    expected_generation="source" if recovery == "submitted" else "target",
                    idempotency_key="restore",
                )
            assert result.state == "succeeded" and target.active
            assert (
                target.calls
                == {
                    "submitted": ["restore", "verify_restore", "activate"],
                    "verified": ["restore", "activate", "activate"],
                    "retry": ["restore", "activate", "activate"],
                    "terminal": ["restore", "activate"],
                }[recovery]
            )
            current = await store.load(session.id)
            assert current is not None
            replacement = FakeAdapter("c")
            with pytest.raises(ExecutionSnapshotError):
                await fresh.restore(
                    session.id,
                    "sandbox",
                    record.id,
                    replacement,
                    expected_run_epoch=current.run_epoch,
                    expected_generation="target",
                    target_generation="replacement",
                    idempotency_key="new-restore",
                )
            assert replacement.calls == []
            await fresh.delete(
                session.id,
                "sandbox",
                record.id,
                expected_run_epoch=current.run_epoch,
                expected_generation="target",
                idempotency_key="delete",
            )

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["capture", "restore"])
@pytest.mark.parametrize("submitted", [False, True])
@pytest.mark.parametrize("retry", ["reconcile", "direct"])
def test_recovery_claim_ack_loss_preserves_submission(
    tmp_path, backend, monkeypatch, phase, submitted, retry
):
    async def run():
        async with setup(tmp_path, backend) as (store, session, artifacts, controller):
            adapter = FakeAdapter("a" if phase == "capture" else "b", fail="ack")
            record = None if phase == "capture" else await capture(controller, session)
            if submitted:
                with pytest.raises(OSError):
                    if phase == "capture":
                        await capture(controller, session, adapter)
                    else:
                        await restore(controller, session, record, adapter)
            else:
                # Simulate process loss after reservation, before submission.
                await controller._reserve(
                    session,
                    "sandbox",
                    phase,
                    "source",
                    adapter,
                    phase,
                    None if record is None else record.id,
                    None if phase == "capture" else "target",
                )
            op = (await controller.inspect(session.id, "sandbox"))["operations"][-1]
            claim = store.fence_run_and_transform_checkpoint

            async def lose_claim_ack(*args, **kwargs):
                await claim(*args, **kwargs)
                raise OSError("lost recovery claim acknowledgement")

            monkeypatch.setattr(store, "fence_run_and_transform_checkpoint", lose_claim_ack)
            with pytest.raises(OSError, match="lost recovery claim"):
                await ExecutionSnapshots(store, artifacts).reconcile(
                    session.id,
                    "sandbox",
                    op["id"],
                    adapter,
                    expected_run_epoch=0,
                    expected_generation="source",
                    idempotency_key=phase,
                    target_generation="target" if phase == "restore" else None,
                )
            monkeypatch.setattr(store, "fence_run_and_transform_checkpoint", claim)
            current = await store.load(session.id)
            assert current.run_epoch == 1
            claimed = (await controller.inspect(session.id, "sandbox"))["operations"][-1]
            assert (claimed["position"] is not None) == submitted
            adapter.fail = None
            fresh = ExecutionSnapshots(store, artifacts)
            if retry == "reconcile":
                await fresh.reconcile(
                    session.id,
                    "sandbox",
                    op["id"],
                    adapter,
                    expected_run_epoch=current.run_epoch,
                    expected_generation="source",
                    idempotency_key=phase,
                    target_generation="target" if phase == "restore" else None,
                )
            elif phase == "capture":
                await capture(fresh, current, adapter)
            else:
                await restore(fresh, current, record, adapter)
            recovery_call = "recover_capture" if phase == "capture" else "verify_restore"
            final_call = "resume" if phase == "capture" else "activate"
            assert adapter.calls == [phase, *([recovery_call] if submitted else []), final_call]
            assert (await fresh.inspect(session.id, "sandbox"))["operations"][-1][
                "state"
            ] == "succeeded"
            with pytest.raises(ExecutionSnapshotConflict):
                await capture(controller, session, FakeAdapter())

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["capture", "restore", "delete"])
@pytest.mark.parametrize("failed", [False, True])
def test_snapshot_recovery_after_blocked_resume(tmp_path, backend, monkeypatch, phase, failed):
    from cayu import (
        AgentSpec,
        CayuApp,
        Message,
        ModelStreamEvent,
        ResumeRequest,
        ScriptedModelProvider,
    )

    async def run():
        async with setup(tmp_path, backend) as (store, _, artifacts, controller):
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                ScriptedModelProvider(
                    response_factory=lambda request: [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed(
                            {"usage": {"input_tokens": 1, "output_tokens": 1}}
                        ),
                    ]
                ),
                default=True,
            )
            source = FakeAdapter()
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="sandbox"),
                    execution_snapshot_adapter=source,
                ),
                default=True,
            )
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            try:
                session_id = str(uuid4())
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "first")],
                    )
                ):
                    pass
                session = await store.load(session_id)
                generation = app.get_environment("sandbox").binding_generation_id
                source.fail = "ack" if phase == "capture" else None
                capture_args = dict(
                    snapshot_store=artifacts,
                    expected_run_epoch=session.run_epoch,
                    expected_generation=generation,
                    idempotency_key="capture",
                )
                if phase == "capture":
                    with pytest.raises(OSError):
                        await app.capture_execution_snapshot(session_id, "sandbox", **capture_args)
                    adapter = source
                else:
                    record = await app.capture_execution_snapshot(
                        session_id, "sandbox", **capture_args
                    )
                    if phase == "restore":
                        adapter = FakeAdapter("b", fail="ack")
                        with pytest.raises(OSError):
                            await controller.restore(
                                session_id,
                                "sandbox",
                                record.id,
                                adapter,
                                expected_run_epoch=session.run_epoch,
                                expected_generation=generation,
                                target_generation="target",
                                idempotency_key="restore",
                            )
                    else:
                        delete = artifacts.delete

                        async def lose_delete_ack(*args, **kwargs):
                            await delete(*args, **kwargs)
                            raise OSError("lost delete acknowledgement")

                        monkeypatch.setattr(artifacts, "delete", lose_delete_ack)
                        with pytest.raises(OSError):
                            await app.delete_execution_snapshot(
                                session_id,
                                "sandbox",
                                record.id,
                                **(capture_args | {"idempotency_key": "delete"}),
                            )
                        monkeypatch.setattr(artifacts, "delete", delete)
                if failed:
                    # A prior failed entrance may already have advanced the run
                    # fence, while no model/tool exposure changed its position.
                    await store.update_status(session_id, SessionStatus.FAILED)
                    await store.fence_run_and_transform_checkpoint(
                        session_id,
                        statuses={SessionStatus.FAILED},
                        checkpoint_transform=lambda current, checkpoint: checkpoint,
                    )
                before_resume = await store.load(session_id)
                checkpoint = await store.load_checkpoint(session_id)
                transcript = await store.load_transcript(session_id)
                with pytest.raises(ExecutionSnapshotOutcomeUnknown):
                    async for _ in app.resume(
                        ResumeRequest(
                            session_id=session_id,
                            messages=[Message.text("user", "next")],
                        )
                    ):
                        pass
                current = await store.load(session_id)
                assert current.status == before_resume.status
                assert current.run_epoch == before_resume.run_epoch
                assert await store.load_checkpoint(session_id) == checkpoint
                assert await store.load_transcript(session_id) == transcript
                if phase == "delete":
                    if failed:
                        with pytest.raises(ExecutionSnapshotConflict):
                            await app.delete_execution_snapshot(
                                session_id,
                                "sandbox",
                                record.id,
                                **(capture_args | {"idempotency_key": "delete"}),
                            )
                    await app.delete_execution_snapshot(
                        session_id,
                        "sandbox",
                        record.id,
                        **(
                            capture_args
                            | {"idempotency_key": "delete", "expected_run_epoch": current.run_epoch}
                        ),
                    )
                else:
                    op = (await controller.inspect(session_id, "sandbox"))["operations"][-1]
                    await ExecutionSnapshots(store, artifacts).reconcile(
                        session_id,
                        "sandbox",
                        op["id"],
                        adapter,
                        expected_run_epoch=current.run_epoch,
                        expected_generation=generation,
                        idempotency_key=phase,
                        target_generation="target" if phase == "restore" else None,
                    )
                    assert adapter.calls == (
                        ["capture", "recover_capture", "resume"]
                        if phase == "capture"
                        else ["restore", "verify_restore", "activate"]
                    )
                inspected = await controller.inspect(session_id, "sandbox")
                assert all(op["state"] == "succeeded" for op in inspected["operations"])
                if phase == "delete":
                    assert inspected["snapshots"][0]["retention"] == "deleted"
            finally:
                await app.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("boundary", ["after_preflight", "before_admission"])
def test_snapshot_resume_rejects_racing_reservation(tmp_path, backend, monkeypatch, boundary):
    from cayu import (
        AgentSpec,
        CayuApp,
        Message,
        ModelStreamEvent,
        ResumeRequest,
        ScriptedModelProvider,
    )
    from cayu.environments import snapshot_lifecycle
    from cayu.sessions.base import SessionRunFenced

    async def run():
        async with setup(tmp_path, backend) as (store, _, _artifacts, controller):
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                ScriptedModelProvider(
                    response_factory=lambda request: [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed(),
                    ]
                ),
                default=True,
            )
            adapter = FakeAdapter()
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="sandbox"),
                    execution_snapshot_adapter=adapter,
                ),
                default=True,
            )
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            try:
                session_id = str(uuid4())
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "first")],
                    )
                ):
                    pass
                session = await store.load(session_id)
                checkpoint = await store.load_checkpoint(session_id)
                transcript = await store.load_transcript(session_id)
                generation = app.get_environment("sandbox").binding_generation_id
                entered = False

                async def reserve():
                    nonlocal entered
                    assert not entered
                    entered = True
                    await controller._reserve(
                        session, "sandbox", "racing", generation, adapter, "capture"
                    )

                if boundary == "after_preflight":
                    ensure = snapshot_lifecycle.ensure_execution_snapshot_binding

                    async def race_after_preflight(*args, **kwargs):
                        await ensure(*args, **kwargs)
                        await reserve()

                    monkeypatch.setattr(
                        snapshot_lifecycle,
                        "ensure_execution_snapshot_binding",
                        race_after_preflight,
                    )
                    error = ExecutionSnapshotOutcomeUnknown
                else:
                    from cayu.runtime import _invocation_lifecycle

                    admit = _invocation_lifecycle.apply_invocation_lifecycle_command

                    async def race_before_admission(runtime_store, command):
                        await reserve()
                        return await admit(runtime_store, command)

                    monkeypatch.setattr(
                        _invocation_lifecycle,
                        "apply_invocation_lifecycle_command",
                        race_before_admission,
                    )
                    error = SessionRunFenced
                with pytest.raises(error):
                    async for _ in app.resume(
                        ResumeRequest(
                            session_id=session_id,
                            messages=[Message.text("user", "next")],
                        )
                    ):
                        pass
                assert entered
                current = await store.load(session_id)
                assert current.status == session.status
                assert current.run_epoch == session.run_epoch
                assert await store.load_transcript(session_id) == transcript
                updated = await store.load_checkpoint(session_id)
                assert {
                    k: v for k, v in updated.items() if k != EXECUTION_SNAPSHOTS_KEY
                } == checkpoint
                assert adapter.calls == []
            finally:
                await app.aclose()

    asyncio.run(run())


def test_deadline_revokes_cancellation_opaque_capture(tmp_path, backend):
    async def run():
        class OpaqueAdapter(FakeAdapter):
            async def capture(self, operation_id, policy, fence):
                await fence()
                self.held = True
                self.entered.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    await self.release.wait()
                await fence()
                return CapturedExecutionSnapshot(process=b"memory", workspace=b"files")

        async with setup(tmp_path, backend, policy=ExecutionSnapshotPolicy(timeout_seconds=1)) as (
            _store,
            session,
            _,
            controller,
        ):
            adapter = OpaqueAdapter()
            started = asyncio.get_running_loop().time()
            try:
                with pytest.raises((ExecutionSnapshotOutcomeUnknown, TimeoutError)):
                    await capture(controller, session, adapter)
                assert asyncio.get_running_loop().time() - started < 2
                state = await controller.inspect(session.id, "sandbox")
                assert state["operations"][0]["state"] == "unknown"
                assert not state["snapshots"]
            finally:
                adapter.release.set()
                await asyncio.sleep(0.05)
            assert not (await controller.inspect(session.id, "sandbox"))["snapshots"]

    asyncio.run(run())


def test_inspection_excludes_private_artifact_material(tmp_path, backend):
    from cayu.environments.snapshot_lifecycle import inspect_execution_snapshots

    async def run():
        async with setup(tmp_path, backend) as (store, session, _, controller):
            await store.checkpoint(session.id, {"unrelated": "controller payload"})
            record = await capture(controller, session)
            projection = await store.load_execution_snapshot_checkpoint(session.id)
            assert "unrelated" not in projection

            async def heavy_read(*args, **kwargs):
                raise AssertionError("Snapshot inspection must not load full state")

            store.load = heavy_read
            store.load_checkpoint = heavy_read
            inspection = await inspect_execution_snapshots(store, session.id, limit=1)
            encoded = inspection[0].model_dump_json()
            assert (
                "memory" not in encoded
                and "pin_owner" not in encoded
                and "artifact_id" not in encoded
            )
            assert record.artifacts[0].artifact_id not in encoded
            assert inspection[0].snapshots[0].total_bytes == record.total_bytes

    asyncio.run(run())


@pytest.mark.parametrize("mode", [0o640, 0o770])
def test_guest_archive_preserves_empty_directories_and_modes(tmp_path, mode):
    from cayu.environments._snapshot_guest import pack, unpack

    source = tmp_path / "source"
    source.mkdir()
    (source / "empty").mkdir(mode=0o750)
    (source / "nested").mkdir(mode=0o755)
    (source / "nested" / "file").write_bytes(b"application bytes")
    (source / "nested" / "file").chmod(mode)
    archive = tmp_path / "workspace.tar"
    policy = ExecutionSnapshotPolicy()
    pack(source, archive, policy.model_dump())
    validate_snapshot_archive(archive.read_bytes(), policy, process=False)
    target = tmp_path / "target"
    target.mkdir()
    previous_umask = os.umask(0o022)
    try:
        unpack(archive, target, policy.model_dump())
    finally:
        os.umask(previous_umask)
    assert (target / "empty").is_dir()
    assert (target / "empty").stat().st_mode & 0o777 == 0o750
    assert (target / "nested" / "file").read_bytes() == b"application bytes"
    assert (target / "nested" / "file").stat().st_mode & 0o777 == mode


def test_guest_workspace_named_images_is_packed_whole(tmp_path):
    from cayu.environments._snapshot_guest import pack

    workspace = tmp_path / "images"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "app.py").write_bytes(b"print()")
    (workspace / "cache.temp").write_bytes(b"partial")
    archive = tmp_path / "workspace.tar"
    policy = ExecutionSnapshotPolicy()
    pack(workspace, archive, policy.model_dump())
    validate_snapshot_archive(archive.read_bytes(), policy, process=False)
    with tarfile.open(archive) as packed:
        assert sorted(packed.getnames()) == ["cache.temp", "src", "src/app.py"]


@pytest.mark.parametrize(
    "problem", ["eligible", "symlink", "hardlink", "files", "bytes", "unreadable"]
)
def test_guest_inventory_rejects_ineligible_workspace_before_freezing(tmp_path, problem):
    from cayu.environments._snapshot_guest import inventory

    if problem == "unreadable" and os.geteuid() == 0:
        pytest.skip("root can read any file")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file").write_bytes(b"x" * 4096)
    request = ExecutionSnapshotPolicy().model_dump()
    if problem == "symlink":
        (workspace / "link").symlink_to("file")
    elif problem == "hardlink":
        os.link(workspace / "file", workspace / "copy")
    elif problem == "files":
        request["max_files"] = 1
        (workspace / "other").write_bytes(b"y")
    elif problem == "bytes":
        request["max_artifact_bytes"] = 4096
    elif problem == "unreadable":
        (workspace / "file").chmod(0)
    if problem == "eligible":
        inventory(workspace, request)
    else:
        with pytest.raises(ValueError):
            inventory(workspace, request)


def test_cli_and_server_inspect_safe_snapshot_metadata(tmp_path, capsys):
    import json

    from cayu.applications import CayuApp
    from cayu.cli import main
    from cayu.storage.sqlite import SQLiteSessionStore

    database = tmp_path / "sessions.db"
    store = SQLiteSessionStore(database)

    async def seed():
        session = await store.create(
            RunRequest(agent_name="agent", session_id="inspect-snapshot", messages=[]),
            identity=SessionIdentity(provider_name="test", model="test"),
        )
        artifacts = LocalArtifactStore(tmp_path / "private")
        controller = ExecutionSnapshots(store, artifacts)
        record = await capture(controller, session)
        await store.close()
        return record

    record = asyncio.run(seed())
    assert main(["session", "snapshots", "inspect-snapshot", "--sqlite", str(database)]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli["execution_snapshots"][0]["snapshots"][0]["id"] == record.id
    assert record.artifacts[0].artifact_id not in json.dumps(cli)
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from tests.server.test_server import _LOCAL_SERVER_CONFIG

    from cayu.server import create_server

    app = CayuApp(session_store=SQLiteSessionStore(database), enable_logging=False)
    with TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)) as client:
        response = client.get("/api/sessions/inspect-snapshot/state")
        assert response.status_code == 200
        assert response.json()["execution_snapshots"] == cli["execution_snapshots"]
        assert record.artifacts[0].artifact_id not in response.text
        assert (
            client.get(
                "/api/sessions/inspect-snapshot/state",
                headers={"If-None-Match": response.headers["ETag"]},
            ).status_code
            == 304
        )
        assert (
            client.get(
                f"/api/artifacts/{record.artifacts[0].artifact_id}/content",
                params={"artifact_store_id": LocalArtifactStore(tmp_path / "private").id},
            ).status_code
            == 404
        )


def test_snapshot_restore_reuses_real_runtime_tool_result(tmp_path):
    from tests.core.test_workspace_mutation_receipts import (
        _ExclusiveWriterBinding,
        _portable_environment_spec,
        _PublicWorkspaceWriteTool,
        _SingleToolProvider,
        collect_events,
    )

    from cayu.agents import AgentSpec
    from cayu.applications import CayuApp
    from cayu.events import EventType
    from cayu.messages import Message
    from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
    from cayu.sessions.base import EventQuery, ResumeRequest
    from cayu.storage.sqlite import SQLiteSessionStore
    from cayu.workspaces import LocalWorkspace

    class StableWriteTool(_PublicWorkspaceWriteTool):
        spec = _PublicWorkspaceWriteTool.spec.model_copy(
            update={
                "execution_profile_identity": ExecutionProfileBehaviorIdentity(
                    name="snapshot-write-test", behavior_version="1", implementation_version="1"
                )
            }
        )

    class WorkspaceAdapter(FakeAdapter):
        def __init__(self, allocation, root):
            super().__init__(allocation)
            self.root = root

        async def capture(self, operation_id, policy, fence):
            await fence()
            self.calls.append("capture")
            self.held = True
            return CapturedExecutionSnapshot(
                process=b"memory", workspace=(self.root / "created.txt").read_bytes()
            )

        async def restore(self, operation_id, snapshot, policy, fence):
            await fence()
            self.calls.append("restore")
            self.held = True
            (self.root / "created.txt").write_bytes(snapshot.workspace)

    async def run():
        database = tmp_path / "sessions.db"
        private = LocalArtifactStore(tmp_path / "private")
        roots = [tmp_path / "source", tmp_path / "target"]
        for root in roots:
            root.mkdir()

        def build(store, index):
            app = CayuApp(session_store=store, enable_logging=False)
            provider = _SingleToolProvider(
                tool_name="public_workspace_write", arguments={"path": "created.txt"}
            )
            if index:
                provider.requests = 1  # Next turn consumes the existing tool transcript.
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(
                    _portable_environment_spec("sandbox"),
                    workspace=LocalWorkspace(roots[index], workspace_id="owned-workspace"),
                    binding=_ExclusiveWriterBinding(),
                    execution_snapshot_adapter=WorkspaceAdapter(
                        "a" if index == 0 else "b", roots[index]
                    ),
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="scripted-model"),
                tools=[StableWriteTool()],
            )
            return app, provider

        store = SQLiteSessionStore(database)
        app, _ = build(store, 0)
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="real-tool",
                messages=[Message.text("user", "create")],
            ),
        )
        session = await store.load("real-tool")
        assert session is not None and session.status == SessionStatus.COMPLETED
        source_generation = app.get_environment("sandbox").binding_generation_id
        record = await app.capture_execution_snapshot(
            session.id,
            "sandbox",
            snapshot_store=private,
            expected_run_epoch=session.run_epoch,
            expected_generation=source_generation,
            idempotency_key="capture",
        )
        checkpoint = await store.load_checkpoint(session.id)
        transcript = await store.load_transcript(session.id)
        await app.aclose()
        await store.close()
        (roots[0] / "created.txt").unlink()

        fresh = SQLiteSessionStore(database)
        restored, provider = build(fresh, 1)
        await restored.restore_execution_snapshot(
            session.id,
            "sandbox",
            record.id,
            snapshot_store=private,
            expected_run_epoch=session.run_epoch,
            expected_generation=source_generation,
            idempotency_key="restore",
        )
        assert (roots[1] / "created.txt").read_bytes() == b"public"
        current = await fresh.load_checkpoint(session.id)
        assert {k: v for k, v in current.items() if k != EXECUTION_SNAPSHOTS_KEY} == {
            k: v for k, v in checkpoint.items() if k != EXECUTION_SNAPSHOTS_KEY
        }
        assert await fresh.load_transcript(session.id) == transcript
        async for _ in restored.resume(
            ResumeRequest(session_id=session.id, messages=[Message.text("user", "continue")])
        ):
            pass
        assert provider.seen_requests
        assert "written" in str(provider.seen_requests[-1].messages)
        records = await fresh.query_events(EventQuery(session_id=session.id))
        assert sum(item.event.type == EventType.TOOL_CALL_COMPLETED for item in records) == 1
        await restored.aclose()
        await fresh.close()

    asyncio.run(run())


def test_generic_checkpoint_writes_cannot_forge_or_erase_snapshot_authority(tmp_path, backend):
    async def run():
        async with setup(tmp_path, backend) as (store, session, _, controller):
            await store.checkpoint(
                session.id,
                {
                    "checkpoint_schema_version": CURRENT_CHECKPOINT_SCHEMA_VERSION,
                    EXECUTION_SNAPSHOTS_KEY: {"forged": "authority"},
                },
            )
            assert EXECUTION_SNAPSHOTS_KEY not in (await store.load_checkpoint(session.id) or {})
            record = await capture(controller, session)
            root = (await store.load_checkpoint(session.id))[EXECUTION_SNAPSHOTS_KEY]
            seen = []

            def overwrite(_, checkpoint):
                seen.append(checkpoint)
                return {
                    "ordinary": True,
                    "checkpoint_schema_version": CURRENT_CHECKPOINT_SCHEMA_VERSION,
                    EXECUTION_SNAPSHOTS_KEY: {"forged": "authority"},
                }

            await store.transform_checkpoint(session.id, overwrite)
            assert EXECUTION_SNAPSHOTS_KEY not in seen[0]
            assert (await store.load_checkpoint(session.id))[EXECUTION_SNAPSHOTS_KEY] == root
            assert (await controller.inspect(session.id, "sandbox"))["snapshots"][0][
                "id"
            ] == record.id

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure", ["unsupported", "component_bytes", "total_bytes", "record_limit"]
)
def test_capability_and_resource_limits_fail_closed(tmp_path, backend, failure):
    async def run():
        policy = (
            ExecutionSnapshotPolicy(max_artifact_bytes=5)
            if failure == "component_bytes"
            else ExecutionSnapshotPolicy(max_total_bytes=5)
            if failure == "total_bytes"
            else ExecutionSnapshotPolicy(max_records=1)
            if failure == "record_limit"
            else None
        )
        async with setup(tmp_path, backend, policy=policy) as (store, session, _, controller):

            class Unsupported(FakeAdapter):
                @property
                def capability(self):
                    return ExecutionSnapshotCapability()

            adapter = Unsupported() if failure == "unsupported" else FakeAdapter()
            if failure == "record_limit":
                await capture(controller, session, adapter)
            with pytest.raises(ExecutionSnapshotError):
                await capture(controller, session, adapter, key="bounded")
            assert adapter.calls == (
                []
                if failure == "unsupported"
                else ["capture", "resume"]
                if failure == "record_limit"
                else ["capture"]
            )
            if failure in {"component_bytes", "total_bytes"}:
                assert not (await controller.inspect(session.id, "sandbox"))["snapshots"]
            if failure == "unsupported":
                assert EXECUTION_SNAPSHOTS_KEY not in (
                    await store.load_checkpoint(session.id) or {}
                )

    asyncio.run(run())


def test_app_rejects_model_visible_snapshot_store(tmp_path):
    from cayu.applications import CayuApp

    app = CayuApp(enable_logging=False)
    public = LocalArtifactStore(tmp_path / "public")
    app.register_environment(Environment(EnvironmentSpec(name="sandbox"), artifact_store=public))
    with pytest.raises(ExecutionSnapshotError, match="private"):
        app._snapshot_controller(public, None)
    with pytest.raises(ExecutionSnapshotError, match="private"):
        app._snapshot_controller(
            LocalArtifactStore(tmp_path / "different", store_id=public.id), None
        )
    asyncio.run(app.aclose())
