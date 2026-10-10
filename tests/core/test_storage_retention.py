"""Application-level retention: workspaces, cross-store references and the worker."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.core.session_retention_conformance import RetentionHarness, create_retention_session

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.factory import (
    EnvironmentAllocationScope,
    EnvironmentAllocationState,
    EnvironmentFactory,
    EnvironmentFactoryResult,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.storage_retention import (
    apply_workspace_retention,
    run_storage_retention_worker,
)
from cayu.sessions.base import RunRequest
from cayu.sessions.records import SessionStatus
from cayu.storage.retention import (
    RetentionAuditState,
    RetentionMode,
    RetentionProtection,
    StorageRetentionPolicy,
    StorageRetentionTarget,
    WorkspaceRetentionPolicy,
)
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.workspaces.branches import (
    WorkspaceBranchCapabilities,
    WorkspaceBranchRetentionStrength,
)


class _Clock:
    def __init__(self) -> None:
        self.offset = timedelta(0)

    def __call__(self) -> datetime:
        return datetime.now(UTC) + self.offset


class _FailingFactory(EnvironmentFactory):
    """Allocates a durable resource, then fails, leaving it to recovery to reap."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.reaped: list[str] = []

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:retention-factory", behavior_version="1", implementation_version="1"
        )

    def allocation_scope(self, request):
        return EnvironmentAllocationScope(provider="fixture", adapter_generation="fixture.v1")

    async def create_recoverable(self, request, allocation):
        if allocation.state is EnvironmentAllocationState.UNPREPARED:
            await allocation.prepare({"resource": allocation.intent.allocation_id})
        if allocation.state is EnvironmentAllocationState.PREPARED:
            await allocation.mark_dispatched()
        path = self.root / (allocation.intent.allocation_id + ".resource")
        path.write_text(json.dumps(allocation.intent.to_payload()))
        await allocation.acknowledge({"resource": allocation.intent.allocation_id})
        raise RuntimeError("fixture failed after acknowledgement")

    async def create(self, request):
        raise AssertionError("retention never reconnects")

    def result(self, request, path, reconnect):  # pragma: no cover - not reached
        return EnvironmentFactoryResult(
            environment=Environment(EnvironmentSpec(name=request.environment_name)),
            reconnect_metadata=reconnect,
        )

    async def reap_allocation(self, request, allocation):
        if not await allocation.mark_reaping():
            return
        self.reaped.append(allocation.intent.allocation_id)
        (self.root / (allocation.intent.allocation_id + ".resource")).unlink(missing_ok=True)
        await allocation.mark_reaped()


class _DurableBranchWorkspace:
    def branch_capabilities(self) -> WorkspaceBranchCapabilities:
        return WorkspaceBranchCapabilities.model_construct(
            retention=WorkspaceBranchRetentionStrength.DURABLE
        )


async def _app_with_leftover(root: Path, *, task_store=None) -> tuple[CayuApp, _Clock, Path]:
    clock = _Clock()
    store = SQLiteSessionStore(root / "sessions.db", ownership_clock=clock)
    app = CayuApp(session_store=store, task_store=task_store, enable_logging=False, clock=clock)
    app.register_provider(
        ScriptedModelProvider(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="probe", model="scripted-model"))
    factory = _FailingFactory(root)
    app.register_environment_factory(
        EnvironmentSpec(
            name="remote",
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:retention-environment", behavior_version="1", implementation_version="1"
            ),
        ),
        factory,
        default=True,
    )
    async for _ in app.run(
        RunRequest(
            agent_name="probe",
            session_id="abandoned",
            messages=[Message.text("user", "finish")],
            max_steps=1,
        )
    ):
        pass
    assert (await store.load("abandoned")).status is SessionStatus.FAILED
    assert list(root.glob("*.resource"))
    return app, clock, root


def _policy(**overrides) -> WorkspaceRetentionPolicy:
    return WorkspaceRetentionPolicy(older_than=timedelta(days=1), **overrides)


def test_workspace_retention_reaps_leftovers_of_old_terminal_sessions(tmp_path) -> None:
    async def run() -> None:
        app, clock, root = await _app_with_leftover(tmp_path)
        store = app.session_store
        try:
            assert (await apply_workspace_retention(app, _policy())).items == ()
            clock.offset = timedelta(days=3)
            planned = await apply_workspace_retention(app, _policy())
            assert [item.item_id for item in planned.items] == ["abandoned"]
            assert planned.items[0].counts["pending_allocations"] == 1
            assert list(root.glob("*.resource")), "a dry run disposes nothing"
            applied = await apply_workspace_retention(app, _policy(dry_run=False), audit=store)
            [item] = applied.items
            assert item.counts["allocations_reaped"] == 1
            assert not list(root.glob("*.resource"))
            record = await store.load_retention_audit(applied.audit_id)
            assert record.store_kind == "workspaces"
            assert record.state is RetentionAuditState.COMPLETED
            assert [entry.item_id for entry in record.entries] == ["abandoned"]
            again = await apply_workspace_retention(app, _policy())
            assert again.items == (), "nothing is left to clean"
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("case", ["caller", "snapshot", "checkpoint", "branch", "task"])
def test_workspace_protections_keep_the_leftover(tmp_path, case) -> None:
    async def run() -> None:
        tasks = SQLiteTaskStore(tmp_path / "tasks.db") if case == "task" else None
        app, clock, root = await _app_with_leftover(tmp_path, task_store=tasks)
        store = app.session_store
        try:
            clock.offset = timedelta(days=3)
            options: dict = {}
            expected = RetentionProtection.CALLER_PROTECTED
            if case == "caller":
                options["protected_session_ids"] = ["abandoned"]
            elif case == "snapshot":
                options["references"] = {RetentionProtection.SNAPSHOT_PIN: ["abandoned"]}
                expected = RetentionProtection.SNAPSHOT_PIN
            elif case == "checkpoint":
                with contextlib.closing(sqlite3.connect(tmp_path / "sessions.db")) as connection:
                    connection.execute(
                        "UPDATE cayu_checkpoints SET state_json = json_set(state_json, "
                        '\'$.workspace_checkpoints\', json(\'{"remote": {"phase": "mutating"}}\')) '
                        "WHERE session_id = 'abandoned'"
                    )
                    connection.commit()
                expected = RetentionProtection.WORKSPACE_CHECKPOINT
            elif case == "branch":
                app.get_environment_factory("remote").source_workspace = _DurableBranchWorkspace()
                expected = RetentionProtection.BRANCH_RETENTION
            else:
                await tasks.create_task(
                    TaskCreate(type="test", task_id="live", session_id="abandoned")
                )
                report = await app.apply_storage_retention(
                    StorageRetentionPolicy(workspaces=_policy(), dry_run=False)
                )
                [workspace_report] = report.reports
                assert {i.item_id: set(i.protections) for i in workspace_report.protected} == {
                    "abandoned": {RetentionProtection.LIVE_TASK}
                }
                assert list(root.glob("*.resource"))
                return
            report = await apply_workspace_retention(
                app, _policy(dry_run=False), audit=store, **options
            )
            assert report.items == ()
            assert {item.item_id: set(item.protections) for item in report.protected} == {
                "abandoned": {expected}
            }
            assert list(root.glob("*.resource"))
        finally:
            await store.close()
            if tasks is not None:
                await tasks.close()

    asyncio.run(run())


def test_workspace_policy_contract() -> None:
    with pytest.raises(ValueError, match="terminal"):
        WorkspaceRetentionPolicy(older_than=timedelta(days=1), statuses={"interrupted"})
    with pytest.raises(ValueError, match="only mode='delete'"):
        WorkspaceRetentionPolicy(older_than=timedelta(days=1), mode=RetentionMode.COMPACT)
    with pytest.raises(ValueError, match="at least one target"):
        StorageRetentionPolicy()


def test_application_policy_collects_references_from_other_databases(tmp_path) -> None:
    """Tasks, evals and snapshots in other databases protect what they name."""

    async def run() -> None:
        from tests.evals.test_corpus_execution import _corpus

        from cayu.evals.store import EvalRunInvocation, EvalRunRequest
        from cayu.snapshots.base import SQLiteAgentSnapshotStore
        from cayu.storage.evals_sqlite import SQLiteEvalStore
        from cayu.vaults.redaction import SecretRedactor

        clock = _Clock()
        sessions = SQLiteSessionStore(tmp_path / "sessions.db", ownership_clock=clock)
        tasks = SQLiteTaskStore(tmp_path / "tasks.db")
        evals = SQLiteEvalStore(tmp_path / "evals.db")
        snapshots = SQLiteAgentSnapshotStore(tmp_path / "snapshots.db")
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
        try:
            harness = RetentionHarness(store=sessions)
            for session_id in ("tasked", "evaluated", "pinned", "free"):
                await create_retention_session(harness, session_id)
            await tasks.create_task(TaskCreate(type="test", task_id="t", session_id="tasked"))
            corpus = _corpus(trials=1)
            redact = SecretRedactor().redact_json
            await evals.save_corpus(corpus, redact_json=redact)
            suite = corpus.suites[0]
            await evals.admit_run(
                EvalRunRequest(
                    run_id="run",
                    idempotency_key="sha256:" + "1" * 64,
                    corpus_revision=corpus.revision,
                    target_key=corpus.target_key,
                    suite_id=suite.id,
                    suite_revision=suite.revision,
                    max_concurrency=1,
                    invocation=EvalRunInvocation(),
                ),
                redact_json=redact,
            )
            with contextlib.closing(sqlite3.connect(tmp_path / "evals.db")) as connection:
                connection.execute(
                    "UPDATE cayu_eval_runs SET scenario_progress_json = ?",
                    (json.dumps({"trials": [{"session_id": "evaluated"}]}),),
                )
                connection.commit()
            with contextlib.closing(sqlite3.connect(tmp_path / "snapshots.db")) as connection:
                connection.execute(
                    "INSERT INTO cayu_agent_snapshot_bindings (binding_id, snapshot_root, "
                    "authority_scope_fingerprint, binding_document, snapshot_document, "
                    "put_receipt_document) VALUES ('b', 'r', 'f', '{}', ?, '{}')",
                    (json.dumps({"capture": {"session_id": "pinned"}}),),
                )
                connection.execute(
                    "INSERT INTO cayu_agent_snapshot_pins (pin_id, snapshot_root, binding_id, "
                    "document, released) VALUES ('p', 'r', 'b', '{}', 0)"
                )
                connection.commit()
            clock.offset = timedelta(days=3)
            policy = StorageRetentionPolicy.for_targets(
                [StorageRetentionTarget.SESSIONS],
                older_than=timedelta(days=1),
                session_mode=RetentionMode.DELETE,
                dry_run=False,
            )
            report = await app.apply_storage_retention(
                policy, eval_store=evals, snapshot_stores=[snapshots]
            )
            [session_report] = report.reports
            assert [item.item_id for item in session_report.items] == ["free"]
            assert {i.item_id: set(i.protections) for i in session_report.protected} == {
                "tasked": {RetentionProtection.LIVE_TASK},
                "evaluated": {RetentionProtection.EVAL_REFERENCE},
                "pinned": {RetentionProtection.SNAPSHOT_PIN},
            }
            assert report.errors == {}
        finally:
            await evals.close()
            await tasks.close()
            await sessions.close()

    asyncio.run(run())


def test_targets_without_stores_are_skipped_not_guessed(tmp_path) -> None:
    async def run() -> None:
        sessions = SQLiteSessionStore(tmp_path / "sessions.db")
        app = CayuApp(session_store=sessions, enable_logging=False)
        try:
            report = await app.apply_storage_retention(
                StorageRetentionPolicy.for_targets(
                    list(StorageRetentionTarget), older_than=timedelta(days=1)
                )
            )
            assert set(report.skipped) == {"evals", "artifacts"}
            assert {r.store_kind for r in report.reports} == {"sessions", "workspaces"}
        finally:
            await sessions.close()

    asyncio.run(run())


def test_worker_applies_on_an_interval_until_stopped(tmp_path) -> None:
    async def run() -> None:
        sessions = SQLiteSessionStore(tmp_path / "sessions.db")
        app = CayuApp(session_store=sessions, enable_logging=False)
        stop = asyncio.Event()
        reports = []

        async def on_report(report) -> None:
            reports.append(report)
            if len(reports) == 2:
                stop.set()

        policy = StorageRetentionPolicy.for_targets(
            [StorageRetentionTarget.SESSIONS], older_than=timedelta(days=1), dry_run=False
        )
        try:
            await asyncio.wait_for(
                run_storage_retention_worker(
                    app, stop, policy=policy, interval_seconds=1, on_report=on_report
                ),
                timeout=20,
            )
            assert len(reports) == 2
            audits = await sessions.list_retention_audits(store_kind="sessions")
            assert len(audits) == 2
            with pytest.raises(ValueError, match="interval_seconds"):
                await run_storage_retention_worker(app, stop, policy=policy, interval_seconds=0)
        finally:
            await sessions.close()

    asyncio.run(run())
