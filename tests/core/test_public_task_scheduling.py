from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tests.core.task_invocation_fixtures import task_backed_session_invocation

from cayu import (
    CayuApp,
    InMemoryTaskStore,
    InvocationOrigin,
    InvocationOriginClaim,
    InvocationOriginTrust,
    TaskCreate,
    TaskQuery,
    TaskRescheduleRequest,
    TaskRetryPolicy,
    TaskScheduleCancelRequest,
    TaskScheduleConflict,
    TaskScheduleEventType,
    TaskSchedulePolicy,
    TaskStatus,
)
from cayu._validation import canonical_durable_json_bytes
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import copy_task
from cayu.tasks.contracts import (
    WORK_CONTRACT_TASK_CREATION_MAX_BYTES,
    WORK_CONTRACT_TASK_MAX_BYTES,
    CompletionResultResolverRef,
    CompletionVerifierRef,
    WorkContractDraft,
    WorkContractRef,
    WorkCriterion,
)
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize(
    "field",
    [
        "type",
        "title",
        "description",
        "parent_task_id",
        "assigned_agent_name",
        "input",
        "metadata",
        "input_scalar",
        "metadata_scalar",
        "work_contract",
        "retry_series",
        "origin",
        "source",
        "root_invocation_id",
        "root_session_id",
    ],
)
def test_public_schedule_creation_rejects_valid_content_substitution(
    field, capsys, caplog, recwarn
):
    canary = "private-schedule-substitution"

    class SubstitutingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def create_task(self, request):
            task = await super().create_task(request)
            original_digest = task.schedule.creation_sha256
            if field in {"input_scalar", "metadata_scalar"}:
                task = task.model_copy(update={field.removesuffix("_scalar"): {"value": 1}})
            elif field in {"input", "metadata"}:
                task = task.model_copy(update={field: {"changed": canary}})
            elif field == "work_contract":
                task = task.model_copy(
                    update={
                        field: WorkContractRef(
                            contract_id="different", version=1, fingerprint="a" * 64
                        )
                    }
                )
            elif field == "retry_series":
                task = task.model_copy(update={field: None})
            elif field in {"origin", "source", "root_invocation_id", "root_session_id"}:
                value = {
                    "origin": InvocationOrigin(
                        trust=InvocationOriginTrust.HOST_ASSERTED, subject=canary
                    ),
                    "source": "scheduled",
                    "root_invocation_id": str(uuid4()),
                    "root_session_id": canary,
                }[field]
                task = task.model_copy(
                    update={"invocation": task.invocation.model_copy(update={field: value})}
                )
            else:
                task = task.model_copy(update={field: canary})
            # Prove the conflict is semantic, not a malformed result/digest.
            task = copy_task(task)
            assert task.schedule.creation_sha256 == original_digest
            return task

    async def run():
        app = CayuApp(task_store=SubstitutingStore(), enable_logging=False)
        with pytest.raises(TaskScheduleConflict) as raised:
            await app.create_task(
                TaskCreate(
                    task_id="followup",
                    type="followup",
                    available_at=datetime.now(UTC),
                    schedule_policy=TaskSchedulePolicy(),
                    retry_policy=TaskRetryPolicy(max_attempts=2)
                    if field == "retry_series"
                    else None,
                    input={"value": True},
                    metadata={"value": True},
                )
            )
        assert canary not in str(raised.value)

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert all(canary not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize(
    "fault", ["session_id", "session_instance_id", "missing_instance", "fabricated"]
)
def test_public_schedule_creation_rejects_false_attachment(fault):
    class SubstitutingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        substitute = False

        async def create_task(self, request):
            task = await super().create_task(request)
            if not self.substitute:
                return task
            updates = {
                "session_id": {"session_id": "other-session"},
                "session_instance_id": {"session_instance_id": str(uuid4())},
                "missing_instance": {"session_instance_id": None},
                "fabricated": {"session_id": "other-session", "session_instance_id": str(uuid4())},
            }[fault]
            return copy_task(task.model_copy(update=updates))

    async def run():
        store = SubstitutingStore()
        app = CayuApp(task_store=store, enable_logging=False)
        request = TaskCreate(
            task_id="followup",
            type="followup",
            available_at=datetime.now(UTC),
            schedule_policy=TaskSchedulePolicy(),
        )
        await app.create_task(request)
        if fault != "fabricated":
            claimed = await store.claim_task("worker")
            await store.attach_task(
                claimed.id,
                session_id="session",
                worker_id="worker",
                lease_expires_at=claimed.lease_expires_at,
                session_invocation=await task_backed_session_invocation(
                    store, claimed.id, "session"
                ),
            )
        before = await store.load_task("followup")
        store.substitute = True
        with pytest.raises(TaskScheduleConflict, match="authority"):
            await app.create_task(request)
        assert await store.load_task("followup") == before
        store.substitute = False
        assert await app.create_task(request) == before

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_public_schedule_creation_allows_attachment_during_readback(backend, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def run():
        reading = asyncio.Event()
        attached = asyncio.Event()
        base = {
            "memory": InMemoryTaskStore,
            "sqlite": SQLiteTaskStore,
            "postgres": PostgresTaskStore,
        }[backend]

        class BarrierStore(base):
            verified_work_mutations_are_cancellation_quiescent = True
            hold_readback = True

            async def load_invocation_snapshot(self, task_id):
                if self.hold_readback:
                    self.hold_readback = False
                    reading.set()
                    await attached.wait()
                return await super().load_invocation_snapshot(task_id)

        store = (
            BarrierStore()
            if backend == "memory"
            else (
                BarrierStore(tmp_path / "concurrent-attachment.sqlite")
                if backend == "sqlite"
                else BarrierStore(dsn, schema_mode=SchemaMode.CREATE)
            )
        )
        creation = None
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            request = TaskCreate(
                task_id="concurrent-followup",
                type="followup",
                available_at=datetime.now(UTC),
                schedule_policy=TaskSchedulePolicy(),
            )
            creation = asyncio.create_task(app.create_task(request))
            await asyncio.wait_for(reading.wait(), timeout=10)
            claimed = await store.claim_task("worker")
            current = await store.attach_task(
                claimed.id,
                session_id="session",
                worker_id="worker",
                lease_expires_at=claimed.lease_expires_at,
                session_invocation=await task_backed_session_invocation(
                    store, claimed.id, "session"
                ),
            )
            attached.set()
            earlier = await asyncio.wait_for(creation, timeout=10)
            assert earlier.session_id is None and earlier.session_instance_id is None
            assert earlier.status is TaskStatus.PENDING
            assert await app.create_task(request) == current
            assert current.session_id == "session" and current.session_instance_id is not None
            await store.complete_task(
                current.id,
                {"done": True},
                worker_id="worker",
                lease_expires_at=current.lease_expires_at,
            )
        finally:
            attached.set()
            if creation is not None:
                await asyncio.gather(creation, return_exceptions=True)
            if backend != "memory":
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("field", ["origin", "source", "root_invocation_id"])
def test_public_schedule_creation_rejects_consistent_but_wrong_origin(field):
    class SubstitutingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def create_task(self, request):
            task = await super().create_task(request)
            if request.task_id == "parent":
                return task
            value = {
                "origin": InvocationOrigin(
                    trust=InvocationOriginTrust.HOST_ASSERTED, subject="other"
                ),
                "source": "scheduled",
                "root_invocation_id": str(uuid4()),
            }[field]
            task = task.model_copy(
                update={"invocation": task.invocation.model_copy(update={field: value})}
            )
            self._tasks[task.id] = task
            return copy_task(task)

    async def run():
        store = SubstitutingStore()
        await store.create_task(
            TaskCreate(
                task_id="parent",
                type="parent",
                invocation_origin=InvocationOriginClaim(subject="expected"),
            )
        )
        app = CayuApp(task_store=store, enable_logging=False)
        with pytest.raises(TaskScheduleConflict):
            await app.create_task(
                TaskCreate(
                    task_id="followup",
                    type="followup",
                    available_at=datetime.now(UTC),
                    schedule_policy=TaskSchedulePolicy(),
                    parent_task_id="parent",
                )
            )

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("parented", [False, True])
def test_public_schedule_creation_replays_current_lifecycle(backend, parented, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def run():
        def open_store():
            if backend == "memory":
                return InMemoryTaskStore()
            if backend == "sqlite":
                return SQLiteTaskStore(tmp_path / "replay.sqlite")
            return PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)

        store = open_store()
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            if parented:
                await store.create_task(
                    TaskCreate(
                        task_id="parent",
                        type="parent",
                        invocation_origin=InvocationOriginClaim(subject="owner", tenant="tenant"),
                    )
                )
                await store.pause_task("parent")
            creation = TaskCreate(
                task_id=f"followup-{parented}",
                type="followup",
                parent_task_id="parent" if parented else None,
                invocation_origin=None if parented else InvocationOriginClaim(subject="owner"),
                input={"task": "unchanged"},
                metadata={"key": "value"},
                assigned_agent_name="agent",
                available_at=datetime.now(UTC) + timedelta(hours=1),
                schedule_policy=TaskSchedulePolicy(),
            )
            original = await app.create_task(creation)
            assert await app.create_task(creation) == original
            await app.reschedule_task(
                TaskRescheduleRequest(
                    task_id=original.id,
                    operation_id="due",
                    expected_revision=1,
                    available_at=datetime.now(UTC),
                )
            )
            assert (await app.create_task(creation)).schedule.revision == 2
            claimed = await store.claim_task("worker", TaskQuery(type="followup"))
            assert await app.create_task(creation) == claimed
            attached = await store.attach_task(
                claimed.id,
                session_id="session",
                worker_id="worker",
                lease_expires_at=claimed.lease_expires_at,
                session_invocation=await task_backed_session_invocation(
                    store, claimed.id, "session"
                ),
            )
            assert await app.create_task(creation) == attached
            terminal = await store.complete_task(
                claimed.id,
                {"done": True},
                worker_id="worker",
                lease_expires_at=attached.lease_expires_at,
            )
            if backend != "memory":
                await store.close()
                store = open_store()
                app = CayuApp(task_store=store, enable_logging=False)
            assert await app.create_task(creation) == terminal
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())


def test_public_schedule_creation_replays_after_unavailable_provenance_readback():
    class MissingReadbackStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        missing = True

        async def load_invocation_snapshot(self, task_id):
            if self.missing:
                self.missing = False
                return None
            return await super().load_invocation_snapshot(task_id)

    async def run():
        store = MissingReadbackStore()
        app = CayuApp(task_store=store, enable_logging=False)
        request = TaskCreate(
            task_id="followup",
            type="followup",
            available_at=datetime.now(UTC),
            schedule_policy=TaskSchedulePolicy(),
        )
        with pytest.raises(TaskScheduleConflict, match="authority is unavailable"):
            await app.create_task(request)
        committed = await store.load_task("followup")
        assert committed is not None
        assert await app.create_task(request) == committed
        assert [event.type for event in await app.list_task_schedule_events("followup")] == [
            TaskScheduleEventType.SCHEDULED
        ]

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_contracted_schedule_creation_replays_after_edit_and_cancellation(
    backend, tmp_path, request
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    def open_store():
        if backend == "postgres":
            return PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
        if backend == "sqlite":
            return SQLiteTaskStore(tmp_path / "contract-schedule.sqlite")
        return InMemoryTaskStore()

    async def run():
        store = open_store()
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            contract = await app.create_work_contract(
                WorkContractDraft(
                    contract_id="scheduled-work",
                    version=1,
                    objective="Evaluate a follow-up.",
                    criteria=(WorkCriterion(criterion_id="done", ordinal=1, description="Done"),),
                    verifier=CompletionVerifierRef(
                        verifier_id="verifier", version="1", configuration_fingerprint="a" * 64
                    ),
                    result_resolver=CompletionResultResolverRef(
                        resolver_id="resolver", version="1", configuration_fingerprint="b" * 64
                    ),
                )
            )
            due = datetime.now(UTC) + timedelta(hours=1)
            original = TaskCreate(
                task_id="contracted-followup",
                type="followup",
                available_at=due,
                schedule_policy=TaskSchedulePolicy(),
                work_contract=contract.reference(),
            )
            created = await app.create_task(original)
            assert await app.create_task(original) == created
            await app.reschedule_task(
                TaskRescheduleRequest(
                    task_id=created.id,
                    operation_id="move",
                    expected_revision=1,
                    available_at=due + timedelta(hours=1),
                )
            )
            if backend != "memory":
                await store.close()
                store = open_store()
            app = CayuApp(task_store=store, enable_logging=False)
            moved = await store.load_task(created.id)
            assert await app.create_task(original) == moved
            await app.cancel_scheduled_task(
                TaskScheduleCancelRequest(
                    task_id=created.id, operation_id="cancel", expected_revision=2
                )
            )
            cancelled = await app.create_task(original)
            assert cancelled.status is TaskStatus.CANCELLED
            assert cancelled.work_contract == contract.reference()
            assert cancelled.schedule.revision == 3
            with pytest.raises(TaskScheduleConflict):
                await app.create_task(original.model_copy(update={"input": {"changed": True}}))
            assert len(await app.list_task_schedule_events(created.id)) == 3
            # Replay must use the ordinary task bound once claim ownership has
            # consumed the space deliberately reserved at initial creation.
            initial_size = len(
                canonical_durable_json_bytes(created.model_dump(mode="json"), "task")
            )
            boundary_request = original.model_copy(
                update={
                    "task_id": "contracted-boundary",
                    "input": {
                        "payload": "x"
                        * (WORK_CONTRACT_TASK_CREATION_MAX_BYTES - initial_size - 256)
                    },
                }
            )
            boundary = await app.create_task(boundary_request)
            await app.reschedule_task(
                TaskRescheduleRequest(
                    task_id=boundary.id,
                    operation_id="due",
                    expected_revision=1,
                    available_at=datetime.now(UTC),
                )
            )
            claimed = await store.claim_task("w" * 512, TaskQuery(type="followup"))
            assert claimed is not None and claimed.id == boundary.id
            claimed_size = len(
                canonical_durable_json_bytes(claimed.model_dump(mode="json"), "task")
            )
            assert (
                WORK_CONTRACT_TASK_CREATION_MAX_BYTES < claimed_size <= WORK_CONTRACT_TASK_MAX_BYTES
            )
            assert await app.create_task(boundary_request) == claimed
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_public_schedule_mutations_replay_after_reconstruction(backend, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    def open_store():
        if backend == "postgres":
            return PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
        if backend == "sqlite":
            return SQLiteTaskStore(tmp_path / "schedule.sqlite")
        return InMemoryTaskStore()

    async def run():
        store = open_store()
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            due = datetime.now(UTC) + timedelta(hours=1)
            created = await app.create_task(
                TaskCreate(
                    task_id="public-scheduled-followup",
                    type="followup",
                    available_at=due,
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            assert created.schedule is not None
            change = TaskRescheduleRequest(
                task_id=created.id,
                operation_id="move",
                expected_revision=created.schedule.revision,
                available_at=due + timedelta(hours=1),
            )
            moved = await app.reschedule_task(change)
            with pytest.raises(TaskScheduleConflict):
                await app.cancel_scheduled_task(
                    TaskScheduleCancelRequest(
                        task_id=created.id, operation_id="stale", expected_revision=1
                    )
                )
            cancel = TaskScheduleCancelRequest(
                task_id=created.id,
                operation_id="cancel",
                expected_revision=moved.schedule.revision,
            )
            cancelled = await app.cancel_scheduled_task(cancel)
            if isinstance(store, (SQLiteTaskStore, PostgresTaskStore)):
                await store.close()
                store = open_store()
            app = CayuApp(task_store=store, enable_logging=False)
            assert await app.reschedule_task(change) == moved
            assert await app.cancel_scheduled_task(cancel) == cancelled
            current = await store.load_task(created.id)
            assert current is not None and current.status is TaskStatus.CANCELLED
            assert current.schedule == cancelled.schedule
            assert await store.claim_task("worker") is None
            assert [event.type for event in await store.list_task_schedule_events(created.id)] == [
                TaskScheduleEventType.SCHEDULED,
                TaskScheduleEventType.RESCHEDULED,
                TaskScheduleEventType.CANCELLED,
            ]
            first_page = await app.list_task_schedule_events(created.id, limit=2)
            assert [event.sequence for event in first_page] == [1, 2]
            last_page = await app.list_task_schedule_events(
                created.id, after_sequence=first_page[-1].sequence, limit=2
            )
            assert [event.sequence for event in last_page] == [3]
            assert last_page[0].type is TaskScheduleEventType.CANCELLED
            assert await app.list_task_schedule_events(created.id, after_sequence=3) == []
        finally:
            if isinstance(store, (SQLiteTaskStore, PostgresTaskStore)):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("field", ["task_id", "operation_id", "policy"])
def test_public_schedule_rejects_mutated_input_without_diagnostics(field, capsys, caplog, recwarn):
    canary = "private-scheduling-input-canary"

    class PrivateValue:
        def __repr__(self):
            return canary

    async def run():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        mutation = TaskRescheduleRequest(
            task_id="followup",
            operation_id="move",
            expected_revision=1,
            available_at=datetime.now(UTC) + timedelta(hours=1),
        )
        if field == "policy":
            object.__setattr__(mutation.policy, "misfire_grace_seconds", PrivateValue())
        else:
            object.__setattr__(mutation, field, PrivateValue())
        with pytest.raises(ValueError, match="request is invalid") as raised:
            await app.reschedule_task(mutation)
        assert canary not in str(raised.value)
        assert await store.list_tasks() == []

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert not recwarn


def test_public_schedule_rejects_secret_identity_before_store_access(capsys, caplog, recwarn):
    canary = "private-scheduling-identity-canary"

    async def run():
        store = InMemoryTaskStore()
        app = CayuApp(
            task_store=store, secret_redactor=SecretRedactor(canary), enable_logging=False
        )
        with pytest.raises(ValueError, match="workload secret") as raised:
            await app.cancel_scheduled_task(
                TaskScheduleCancelRequest(
                    task_id=canary, operation_id="cancel", expected_revision=1
                )
            )
        assert canary not in str(raised.value)
        assert await store.list_tasks() == []

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert not recwarn


@pytest.mark.parametrize("malformed", [False, True])
def test_public_schedule_rejects_wrong_store_receipt(malformed, capsys, caplog, recwarn):
    canary = "private-scheduling-receipt-canary"

    class PrivateValue:
        def __repr__(self):
            return canary

    class IncorrectStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def reschedule_task(self, request):
            receipt = await super().reschedule_task(request)
            object.__setattr__(receipt, "operation_id", PrivateValue() if malformed else "other")
            return receipt

    async def run():
        store = IncorrectStore()
        app = CayuApp(task_store=store, enable_logging=False)
        due = datetime.now(UTC) + timedelta(hours=1)
        await app.create_task(
            TaskCreate(
                task_id="followup",
                type="followup",
                available_at=due,
                schedule_policy=TaskSchedulePolicy(),
            )
        )
        with pytest.raises(TaskScheduleConflict) as raised:
            await app.reschedule_task(
                TaskRescheduleRequest(
                    task_id="followup",
                    operation_id="move",
                    expected_revision=1,
                    available_at=due + timedelta(hours=1),
                )
            )
        assert canary not in str(raised.value)

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert not recwarn


@pytest.mark.parametrize("cancel_schedule", [False, True])
def test_public_schedule_cancellation_after_commit_preserves_exact_replay(cancel_schedule):
    async def run():
        committed = asyncio.Event()
        delay_acknowledgement = True

        class DelayedAcknowledgementStore(InMemoryTaskStore):
            verified_work_mutations_are_cancellation_quiescent = True

            async def reschedule_task(self, request):
                receipt = await super().reschedule_task(request)
                if delay_acknowledgement:
                    committed.set()
                    await asyncio.Event().wait()
                return receipt

            async def cancel_scheduled_task(self, request):
                receipt = await super().cancel_scheduled_task(request)
                if delay_acknowledgement:
                    committed.set()
                    await asyncio.Event().wait()
                return receipt

        store = DelayedAcknowledgementStore()
        app = CayuApp(task_store=store, enable_logging=False)
        due = datetime.now(UTC) + timedelta(hours=1)
        await app.create_task(
            TaskCreate(
                task_id="followup",
                type="followup",
                available_at=due,
                schedule_policy=TaskSchedulePolicy(),
            )
        )
        if cancel_schedule:
            cancellation = TaskScheduleCancelRequest(
                task_id="followup", operation_id="cancel", expected_revision=1
            )

            async def publish():
                return await app.cancel_scheduled_task(cancellation)
        else:
            reschedule = TaskRescheduleRequest(
                task_id="followup",
                operation_id="move",
                expected_revision=1,
                available_at=due + timedelta(hours=1),
            )

            async def publish():
                return await app.reschedule_task(reschedule)

        owner = asyncio.create_task(publish())
        try:
            await asyncio.wait_for(committed.wait(), timeout=5)
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner
            assert owner.cancelled() and owner.cancelling() == 1
            durable = await store.load_task("followup")
            assert durable is not None and durable.schedule is not None
            assert durable.schedule.revision == 2
            delay_acknowledgement = False
            replay = await publish()
            assert replay.schedule == durable.schedule
            assert len(await store.list_task_schedule_events("followup")) == 2
        finally:
            if not owner.done():
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(run())


def test_public_schedule_keeps_expected_authority_separate_from_store_argument():
    class MutatingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def reschedule_task(self, request):
            receipt = await super().reschedule_task(request)
            changed_time = request.available_at + timedelta(days=1)
            object.__setattr__(request, "available_at", changed_time)
            object.__setattr__(receipt, "available_at", changed_time)
            return receipt

    async def run():
        store = MutatingStore()
        app = CayuApp(task_store=store, enable_logging=False)
        due = datetime.now(UTC) + timedelta(hours=1)
        await app.create_task(
            TaskCreate(
                task_id="followup",
                type="followup",
                available_at=due,
                schedule_policy=TaskSchedulePolicy(),
            )
        )
        request = TaskRescheduleRequest(
            task_id="followup",
            operation_id="move",
            expected_revision=1,
            available_at=due + timedelta(hours=1),
        )
        with pytest.raises(TaskScheduleConflict, match="changed the requested schedule"):
            await app.reschedule_task(request)
        assert request.available_at == due + timedelta(hours=1)

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["other_task", "duplicate", "over_limit", "malformed"])
def test_public_schedule_history_rejects_untrusted_page(fault, capsys, caplog, recwarn):
    canary = "private-schedule-history-canary"

    class PrivateValue:
        def __repr__(self):
            return canary

    class IncorrectHistoryStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def list_task_schedule_events(self, task_id, *, after_sequence=0, limit=100):
            result = await super().list_task_schedule_events(
                task_id, after_sequence=after_sequence, limit=limit
            )
            if fault == "other_task":
                object.__setattr__(result[0], "task_id", "other")
            elif fault == "malformed":
                object.__setattr__(result[0], "invocation_id", PrivateValue())
            elif fault == "duplicate":
                result.append(result[0])
            else:
                result *= limit + 1
            return result

    async def run():
        store = IncorrectHistoryStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await app.create_task(
            TaskCreate(
                task_id="followup",
                type="followup",
                available_at=datetime.now(UTC) + timedelta(hours=1),
                schedule_policy=TaskSchedulePolicy(),
            )
        )
        with pytest.raises(TaskScheduleConflict) as raised:
            await app.list_task_schedule_events("followup", limit=2)
        assert canary not in str(raised.value)

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert not recwarn


@pytest.mark.parametrize("fault", ["identity", "missing", "digest", "malformed"])
def test_public_schedule_creation_rejects_invalid_evidence(fault, capsys, caplog, recwarn):
    canary = "private-schedule-creation-canary"

    class PrivateValue:
        def __repr__(self):
            return canary

    class IncorrectCreationStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def create_task(self, request):
            task = await super().create_task(request)
            if fault == "identity":
                object.__setattr__(task, "id", "other")
            elif fault == "missing":
                object.__setattr__(task, "schedule", None)
            elif fault == "digest":
                object.__setattr__(task.schedule, "creation_sha256", "0" * 64)
            else:
                object.__setattr__(task, "title", PrivateValue())
            return task

    async def run():
        app = CayuApp(task_store=IncorrectCreationStore(), enable_logging=False)
        with pytest.raises(TaskScheduleConflict) as raised:
            await app.create_task(
                TaskCreate(
                    task_id="followup",
                    type="followup",
                    available_at=datetime.now(UTC) + timedelta(hours=1),
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
        assert canary not in str(raised.value)

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert not recwarn
