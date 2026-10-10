"""Task creation composes directly with stores and preserves public boundaries."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.core.test_verified_work_contracts import _assert_secret_absent_from_cayu_error

import cayu
from cayu import _application_task_creation as task_creation
from cayu.applications import CayuApp
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.invocation import (
    InvocationOrigin,
    InvocationOriginTrust,
    SessionExecutionSource,
)
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest, run_request_with_runtime_invocation
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.storage.tasks_postgres import PostgresTaskStore
from cayu.tasks.contracts import (
    CompletionResultResolverRef,
    CompletionVerifierRef,
    WorkContractDraft,
    WorkCriterion,
    work_contract_from_draft,
)
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.vaults.redaction import SecretRedactor


def contract_draft() -> WorkContractDraft:
    return WorkContractDraft(
        contract_id="report",
        version=1,
        objective="Publish a verified report.",
        criteria=(WorkCriterion(criterion_id="ready", ordinal=1, description="Report is ready."),),
        verifier=CompletionVerifierRef(
            verifier_id="report", version="1", configuration_fingerprint="a" * 64
        ),
        result_resolver=CompletionResultResolverRef(
            resolver_id="report", version="1", configuration_fingerprint="b" * 64
        ),
    )


def test_task_creation_composes_without_application_controllers() -> None:
    script = """
import asyncio
import importlib.abc
import sys
blocked = {
    "cayu.applications", "cayu.runtime._session_engine",
    "cayu.runtime._model_step_executor", "cayu.runtime._recovery_coordinator",
    "cayu.runtime._tool_round_executor",
}
class RejectControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(f"Task creation imported {fullname}")
sys.meta_path.insert(0, RejectControllers())
from cayu._application_task_creation import create_task
from cayu.sessions.base import InMemorySessionStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.vaults.redaction import SecretRedactor
async def scenario():
    tasks = InMemoryTaskStore()
    task = await create_task(
        TaskCreate(task_id="standalone", type="report"),
        task_store=tasks, session_store=InMemorySessionStore(),
        redactor=SecretRedactor(),
    )
    assert task.id == "standalone" and task.work_contract is None
    assert await tasks.load_task(task.id) == task
asyncio.run(scenario())
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_task_creation_composes_contracts_and_session_provenance(
    backend, tmp_path, request
) -> None:
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario() -> None:
        if backend == "postgres":
            tasks = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
        elif backend == "sqlite":
            tasks = SQLiteTaskStore(tmp_path / "tasks.sqlite")
        else:
            tasks = InMemoryTaskStore()
        sessions = InMemorySessionStore()
        redactor = SecretRedactor()
        try:
            draft = contract_draft()
            contract = await task_creation.create_work_contract(
                draft, task_store=tasks, redactor=redactor
            )
            assert contract == work_contract_from_draft(draft)
            loaded = await task_creation.load_work_contract(
                contract.reference(), task_store=tasks, redactor=redactor
            )
            assert loaded == contract and loaded is not contract
            await sessions.create(
                run_request_with_runtime_invocation(
                    RunRequest(agent_name="reporter", session_id="source", messages=[]),
                    source=SessionExecutionSource.HTTP_RUN,
                    verified_origin=InvocationOrigin(
                        trust=InvocationOriginTrust.SERVER_VERIFIED, subject="report-reader"
                    ),
                ),
                identity=SessionIdentity(provider_name="fixture", model="fixture"),
            )
            snapshot = await sessions.load_invocation_snapshot("source")
            assert snapshot is not None
            for contracted in (False, True):
                task = await task_creation.create_task(
                    TaskCreate(
                        task_id="verified" if contracted else "ordinary",
                        type="report",
                        session_id="source",
                        work_contract=contract.reference() if contracted else None,
                    ),
                    task_store=tasks,
                    session_store=sessions,
                    redactor=redactor,
                )
                assert task.work_contract == (contract.reference() if contracted else None)
                assert task.session_instance_id == snapshot.session_instance_id
                assert task.invocation.origin == snapshot.invocation.origin
                assert task.invocation.root_invocation_id == snapshot.invocation.root_invocation_id
                assert await tasks.load_task(task.id) == task
        finally:
            if isinstance(tasks, (SQLiteTaskStore, PostgresTaskStore)):
                await tasks.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("entrance", ["component", "application"])
@pytest.mark.parametrize("operation", ["create_task", "create_work_contract", "load_work_contract"])
def test_task_creation_does_not_retain_sensitive_dependency_representations(
    entrance, operation
) -> None:
    secret = "task-creation-dependency-secret"

    class SensitiveStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __repr__(self) -> str:
            return secret

        async def publish_work_contract(self, contract):
            raise ValueError("publication failed")

        async def load_work_contract(self, reference):
            raise ValueError("lookup failed")

        async def create_task(self, request):
            raise ValueError("creation failed")

    class SensitiveSessions(InMemorySessionStore):
        def __repr__(self) -> str:
            return secret

    class SensitiveRedactor(SecretRedactor):
        def __repr__(self) -> str:
            return secret

    async def scenario() -> None:
        tasks = SensitiveStore()
        sessions = SensitiveSessions()
        redactor = SensitiveRedactor(secret)
        app = CayuApp(
            task_store=tasks,
            session_store=sessions,
            secret_redactor=redactor,
            enable_logging=False,
        )
        draft = contract_draft()
        if operation == "create_work_contract":
            argument = draft
        elif operation == "load_work_contract":
            argument = work_contract_from_draft(draft).reference()
        else:
            argument = TaskCreate(
                task_id="rejected",
                type="report",
                work_contract=work_contract_from_draft(draft).reference(),
            )
        with pytest.raises(ValueError, match="failed") as raised:
            if entrance == "application":
                await getattr(app, operation)(argument)
            else:
                dependencies = {"task_store": tasks, "redactor": redactor}
                if operation == "create_task":
                    dependencies["session_store"] = sessions
                await getattr(task_creation, operation)(argument, **dependencies)
        _assert_secret_absent_from_cayu_error(raised.value, secret)

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["create_task", "create_work_contract", "load_work_contract"])
def test_task_creation_keeps_missing_task_store_optional(operation) -> None:
    async def scenario() -> None:
        redactor = SecretRedactor()
        draft = contract_draft()
        with pytest.raises(RuntimeError, match="task_store is required"):
            if operation == "create_task":
                await task_creation.create_task(
                    TaskCreate(type="report"),
                    task_store=None,
                    session_store=InMemorySessionStore(),
                    redactor=redactor,
                )
            elif operation == "create_work_contract":
                await task_creation.create_work_contract(draft, task_store=None, redactor=redactor)
            else:
                await task_creation.load_work_contract(
                    work_contract_from_draft(draft).reference(), task_store=None, redactor=redactor
                )

    asyncio.run(scenario())
