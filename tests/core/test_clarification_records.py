"""Native record mapping tests; receiving authorization is qualified separately."""

import asyncio
import sqlite3
from uuid import uuid4

import pytest
from tests.core.test_clarification_contracts import accept, fixture
from tests.core.test_clarification_schema import _install
from tests.core.test_participant_identity import stores

from cayu.collaboration._clarification_records import (
    ClarificationLineageRecord,
    clarification_record_projection,
)
from cayu.collaboration._contracts import CollaborationContractError, snapshot_input
from cayu.collaboration.clarifications import ClarificationDueCursor, ClarificationLineageUsage
from cayu.collaboration.memory import _MemoryRepository
from cayu.storage._collaboration_repository import _SQLRepository

__all__ = ["stores"]


@pytest.mark.anyio
async def test_native_due_question_scan_restarts_with_exact_cursor(stores):
    store = stores()
    state = fixture()[0]
    # Repository qualification only: these are stored decision records, not
    # authenticated question creation or a service execution permit.
    keys = []
    try:
        async with store._transaction("app", write=True) as tx:
            for index, deadline in enumerate((200, 200, 300)):
                operation = state.question.operation.model_copy(
                    update={"caller_key": f"due-{index}"}
                )
                key = (operation.namespace_incarnation, operation.generation, operation.caller_key)
                keys.append(key)
                candidate = state.model_copy(
                    update={
                        "question": state.question.model_copy(
                            update={"operation": operation, "deadline_at_ms": deadline}
                        )
                    }
                )
                await tx.put("clarification_questions", key, candidate, insert=True)
        reopened = stores()
        async with reopened._transaction("app", write=False) as tx:
            related = await tx.scan_clarification_lineage_questions(state.question.lineage, limit=2)
            assert [row["question"]["operation"]["caller_key"] for row in related] == [
                "due-0",
                "due-1",
            ]
            assert (
                await tx.scan_clarification_lineage_questions(
                    state.question.lineage.model_copy(update={"caller_key": "other-lineage"}),
                    limit=1,
                )
                == []
            )
            for invalid in (True, 0, 34):
                with pytest.raises(CollaborationContractError):
                    await tx.scan_clarification_lineage_questions(
                        state.question.lineage, limit=invalid
                    )
            assert await tx.scan_due_clarifications(after=None, now_ms=199, limit=1) == []
            first = await tx.scan_due_clarifications(after=None, now_ms=200, limit=1)
            assert first[0]["question"]["operation"]["caller_key"] == "due-0"
            cursor = ClarificationDueCursor(
                deadline_at_ms=200,
                operation=state.question.operation.model_copy(update={"caller_key": "due-0"}),
            )
            second = await tx.scan_due_clarifications(after=cursor, now_ms=200, limit=64)
            assert [row["question"]["operation"]["caller_key"] for row in second] == ["due-1"]
            for limit in (True, 0, -1, 65):
                with pytest.raises(CollaborationContractError):
                    await tx.scan_due_clarifications(after=None, now_ms=200, limit=limit)
            with pytest.raises(CollaborationContractError):
                await tx.scan_due_clarifications(after=None, now_ms=True, limit=1)
    finally:
        async with store._transaction("app", write=True) as tx:
            for key in keys:
                await tx.delete("clarification_questions", key)


def records():
    question, reply = fixture()
    answered = accept(question, reply)
    assert answered.input is not None
    lineage = ClarificationLineageRecord(
        operation=question.question.lineage,
        root_request=question.question.request,
        policy=question.question.policy,
        budget_binding=question.question.budget_binding,
        budget_authority_sha256=question.question.budget_authority_sha256,
        usage=ClarificationLineageUsage(questions=1, content_bytes=32, pending=1),
    )
    return [
        ("clarification_questions", ("namespace", 1, "question"), answered),
        ("clarification_inputs", ("request", "one", 1), answered.input),
        ("clarification_lineages", ("namespace", 1, "lineage"), lineage),
    ]


@pytest.fixture(params=["memory", "sqlite"])
def repository(request):
    if request.param == "memory":
        yield _MemoryRepository({}, scope="app")
    else:
        connection = sqlite3.connect(":memory:")
        try:
            _install(connection)
            yield _SQLRepository(connection, "app", postgres=False)
        finally:
            connection.close()


@pytest.mark.anyio
async def test_clarification_records_roundtrip(repository):
    for family, key, record in records():
        await repository.put(family, key, record, insert=True)
        assert await repository.get(family, key) == snapshot_input(record)


@pytest.mark.anyio
async def test_clarification_question_scan_is_bounded_and_exact(repository):
    state = fixture()[0]
    for ordinal in range(3):
        operation = state.question.operation.model_copy(
            update={"caller_key": f"question-{ordinal}"}
        )
        candidate = state.model_copy(
            update={"question": state.question.model_copy(update={"operation": operation})}
        )
        await repository.put(
            "clarification_questions",
            ("namespace", 1, operation.caller_key),
            candidate,
            insert=True,
        )
    selected = await repository.scan_clarification_questions(state.question.request, limit=2)
    assert len(selected) == 2
    assert [row["question"]["operation"]["caller_key"] for row in selected] == [
        "question-0",
        "question-1",
    ]
    for invalid in (0, -1, True, 34, "2"):
        with pytest.raises(CollaborationContractError):
            await repository.scan_clarification_questions(state.question.request, limit=invalid)
    changed = state.question.request.model_copy(
        update={
            "owner": state.question.request.owner.model_copy(update={"incarnation": "replaced"})
        }
    )
    with pytest.raises(CollaborationContractError):
        await repository.scan_clarification_questions(changed, limit=2)


@pytest.mark.anyio
async def test_clarification_records_reject_wrong_key_without_write(repository):
    for family, key, record in records():
        wrong = (*key[:-1], "different")
        with pytest.raises(CollaborationContractError):
            await repository.put(family, wrong, record, insert=True)
        assert await repository.get(family, wrong) is None


@pytest.mark.anyio
async def test_clarification_records_sqlite_restart(tmp_path):
    path = tmp_path / "records.sqlite"
    connection = sqlite3.connect(path)
    try:
        _install(connection)
        repository = _SQLRepository(connection, "app", postgres=False)
        for family, key, record in records():
            await repository.put(family, key, record, insert=True)
        connection.commit()
    finally:
        connection.close()
    connection = sqlite3.connect(path)
    try:
        repository = _SQLRepository(connection, "app", postgres=False)
        for family, key, record in records():
            assert await repository.get(family, key) == snapshot_input(record)
    finally:
        connection.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "column,value",
    [
        ("request_id", "wrong"),
        ("request_incarnation", "wrong"),
        ("participant_id", "wrong"),
        ("state", "cancelled"),
        ("next_due_at_ms", 999),
        ("lineage_namespace", "wrong"),
        ("lineage_generation", 999),
        ("lineage_key", "wrong"),
    ],
)
async def test_sqlite_clarification_rejects_index_corruption(column, value):
    connection = sqlite3.connect(":memory:")
    try:
        _install(connection)
        repository = _SQLRepository(connection, "app", postgres=False)
        family, key, record = records()[0]
        await repository.put(family, key, record, insert=True)
        connection.execute(
            f"UPDATE cayu_collaboration_clarification_questions SET {column}=?", (value,)
        )
        with pytest.raises(CollaborationContractError):
            await repository.get(family, key)
    finally:
        connection.close()


@pytest.mark.parametrize("index", [0, 1, 2])
def test_clarification_projection_rejects_wrong_scope(index):
    family, key, record = records()[index]
    with pytest.raises(CollaborationContractError):
        clarification_record_projection(family, record, scope="other", key=key)


def test_clarification_projection_rejects_boolean_revision():
    family, key, record = records()[1]
    with pytest.raises(CollaborationContractError):
        clarification_record_projection(family, record, scope="app", key=(*key[:-1], True))


@pytest.mark.anyio
async def test_native_clarification_transaction_rollback(stores):
    store = stores()
    with pytest.raises(ConnectionError, match="after writes"):
        async with store._transaction("app", write=True) as tx:
            for family, key, record in records():
                await tx.put(family, key, record, insert=True)
            raise ConnectionError("after writes")
    async with store._transaction("app", write=False) as tx:
        for family, key, _ in records():
            assert await tx.get(family, key) is None
    async with store._transaction("app", write=True) as tx:
        for family, key, record in records():
            await tx.put(family, key, record, insert=True)
    reopened = stores()
    async with reopened._transaction("app", write=False) as tx:
        for family, key, record in records():
            assert await tx.get(family, key) == snapshot_input(record)
        assert await tx.scan_clarification_questions(records()[0][2].question.request, limit=1) == [
            snapshot_input(records()[0][2])
        ]


@pytest.mark.anyio
async def test_native_clarification_cancellation_rolls_back(stores):
    store = stores()
    # Each PostgreSQL case shares a database, but not an application namespace.
    scope = uuid4().hex
    ready = asyncio.Event()
    parked = asyncio.Event()
    values = []
    for family, key, record in records():
        # Rewrite the fixture's complete scope before reconstructing the typed
        # record; partial authority tuples must not pass repository validation.
        def rescope(value):
            if isinstance(value, dict):
                return {
                    k: scope if k == "application_scope" else rescope(v) for k, v in value.items()
                }
            if isinstance(value, list):
                return [rescope(v) for v in value]
            return value

        # Use the open question here: an answered record also commits its exact
        # original question/reply, so changing its scope would invalidate hashes.
        if family == "clarification_questions":
            record = fixture()[0]
        elif family == "clarification_inputs":
            continue
        values.append((family, key, type(record).model_validate(rescope(snapshot_input(record)))))

    async def writer():
        try:
            async with store._transaction(scope, write=True) as tx:
                for family, key, record in values:
                    await tx.put(family, key, record, insert=True)
                ready.set()
                await parked.wait()
        except asyncio.CancelledError:
            assert asyncio.current_task().cancelling() == 1
            raise

    task = asyncio.create_task(writer())
    try:
        await asyncio.wait_for(ready.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    async with store._transaction(scope, write=False) as tx:
        for family, key, _ in values:
            assert await tx.get(family, key) is None
