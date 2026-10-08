"""Task value contracts remain usable independently of store implementations."""

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


@pytest.mark.parametrize(
    ("record_module", "topology_module"),
    (
        ("cayu", "cayu"),
        ("cayu.tasks", "cayu.tasks"),
        ("cayu.runtime", "cayu.runtime"),
        ("cayu.tasks.records", "cayu.tasks.topology"),
    ),
)
def test_task_contracts_do_not_load_store_implementations(record_module, topology_module):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys
from typing import get_type_hints

from cayu.sessions.invocation import TaskInvocation

records = importlib.import_module(sys.argv[1])
topology = importlib.import_module(sys.argv[2])
task = records.Task(
    id="task",
    type="step",
    graph_id="graph",
    session_id="session",
    invocation=TaskInvocation(
        origin={"trust": "unattributed"},
        root_invocation_id="c092a88f-bc0a-4ec0-a92c-d3b8cc21d8cb",
        source="sdk_task",
    ),
)
node = topology.TaskTopologyNode.from_task(task)
query = topology.TaskTopologyQuery(linked_session_ids=("session",))
result = topology.TaskTopologyStoreResult(
    observed_at=task.created_at,
    session_branches=(topology.TaskTopologySessionBranch(session_id="session", tasks=(node,)),),
)
assert result.validate_for_query(query) is result
assert get_type_hints(records.Task)["status"] is records.TaskStatus
assert get_type_hints(topology.TaskTopologyNode.from_task)["task"] is records.Task
assert not {
    "cayu.tasks.base",
    "cayu.storage.sqlite",
    "cayu.storage.postgres",
}.intersection(sys.modules)
""",
            record_module,
            topology_module,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_task_contracts_preserve_public_and_legacy_identity():
    from cayu.tasks import (
        base,
        cancellation,
        creation,
        handoff,
        queries,
        records,
        retry,
        terminalization,
        topology,
        work_receipts,
    )

    public = [importlib.import_module(name) for name in ("cayu", "cayu.tasks", "cayu.runtime")]
    for owner, names in (
        (
            records,
            (
                "Task",
                "TaskStatus",
                "TaskRetryPolicy",
                "TaskRetrySeriesDisposition",
                "TaskRetrySeriesSnapshot",
                "TaskClaimLost",
                "TaskSessionClosureClaim",
            ),
        ),
        (
            terminalization,
            (
                "TaskTerminalKind",
                "TaskTerminalizationConflict",
                "TaskTerminalizationRequest",
                "TaskTerminalizationReceipt",
                "TaskTerminalizationRetryPolicy",
                "TaskTerminalizationRetryResult",
                "TaskTerminalizationUncertain",
            ),
        ),
        (
            handoff,
            (
                "TaskInterruptedHandoffConflict",
                "TaskInterruptedHandoffRequest",
                "TaskInterruptedHandoffReceipt",
                "InterruptedTaskContinuationClaimPage",
            ),
        ),
        (
            cancellation,
            (
                "TaskCancellationReconciliationRequest",
                "TaskCancellationReconciliationResult",
                "TaskCancellationReconciliation",
                "TaskCancellationReconciliationEvidence",
                "TaskRetryCancellationReconciliationRequest",
                "TaskRetryCancellationReconciliation",
                "TaskRetryCancellationReconciliationEvidence",
            ),
        ),
        (
            retry,
            (
                "TaskRetryAttemptDisposition",
                "TaskRetryAttemptReport",
                "TaskRetryEvent",
                "TaskRetryEventType",
                "TaskRetrySettlementRequest",
                "TaskRetrySettlementResult",
            ),
        ),
        (creation, ("TaskCreate", "TaskInvocationSnapshot")),
        (
            queries,
            (
                "TaskQuery",
                "TaskAggregateFilter",
                "TaskOrder",
                "TaskStatusCounts",
                "TaskOperationalSnapshot",
            ),
        ),
        (work_receipts, ("CompletionDecisionApplicationReceipt",)),
        (
            topology,
            (
                "TaskTopologyNode",
                "TaskTopologyQuery",
                "TaskTopologySessionBranch",
                "TaskTopologyChildBranch",
                "TaskTopologyStoreResult",
                "TaskTopologyCycle",
                "TaskTopologyInconsistent",
                "TaskTopologyTraversalLimitExceeded",
            ),
        ),
    ):
        for name in names:
            canonical = getattr(owner, name)
            assert getattr(base, name) is canonical
            assert all(getattr(module, name) is canonical for module in public)
            assert canonical.__module__ == owner.__name__
            # Protocol 0 GLOBAL records represent the original persisted class path.
            assert pickle.loads(f"ccayu.tasks.base\n{name}\n.".encode()) is canonical


@pytest.mark.parametrize("public_module", ("cayu", "cayu.tasks", "cayu.runtime"))
def test_task_terminal_contracts_work_without_loading_stores(public_module):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import pickle
import sys
from datetime import UTC, datetime
from typing import get_type_hints

public = importlib.import_module(sys.argv[1])
from cayu.tasks.handoff import prepare_interrupted_task_handoff
from cayu.tasks.terminalization import prepare_task_terminalization

terminal = public.TaskTerminalizationRequest(
    task_id="task", worker_id="worker", kind="failed", error={"reason": "test"},
    idempotency_key="terminal",
)
detached, digest = prepare_task_terminalization(terminal)
assert detached == terminal and detached is not terminal and len(digest) == 64
detached.error["reason"] = "changed"
assert terminal.error == {"reason": "test"}
handoff = public.TaskInterruptedHandoffRequest(
    task_id="task", worker_id="worker", lease_expires_at=datetime(2030, 1, 1, tzinfo=UTC),
    session_id="session", session_instance_id="12345678-1234-4234-8234-123456789abc",
    session_run_epoch=1, handoff_id="handoff",
)
detached_handoff, digest = prepare_interrupted_task_handoff(handoff)
assert detached_handoff == handoff and detached_handoff is not handoff and len(digest) == 64
page = public.InterruptedTaskContinuationClaimPage(
    scanned_candidates=0, rejected_candidates=0, exhausted=True,
)
for value in (terminal, handoff, page):
    assert type(pickle.loads(pickle.dumps(value))) is type(value)
    assert get_type_hints(type(value))
for contract in (
    public.TaskCancellationReconciliationRequest,
    public.TaskCancellationReconciliationResult,
    public.TaskRetryCancellationReconciliationRequest,
    public.TaskRetryCancellationReconciliation,
    public.TaskRetrySettlementRequest,
    public.TaskRetrySettlementResult,
    public.TaskCreate,
    public.TaskQuery,
    public.TaskOperationalSnapshot,
    public.TaskSessionClosureClaim,
    public.CompletionDecisionApplicationReceipt,
):
    assert contract.model_json_schema()
    assert get_type_hints(contract)
assert not {
    "cayu.tasks.base", "cayu.tasks.memory", "cayu.tasks.store",
    "cayu.storage.tasks_sqlite", "cayu.storage.sqlite", "cayu.storage.postgres",
}.intersection(sys.modules)
""",
            public_module,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_task_records_and_topology_support_detached_pickle_round_trips():
    from cayu.sessions.invocation import TaskInvocation
    from cayu.tasks.records import Task, copy_task
    from cayu.tasks.topology import TaskTopologyNode, TaskTopologyQuery, TaskTopologyStoreResult

    task = Task(
        id="task",
        type="step",
        input={"nested": ["original"]},
        invocation=TaskInvocation(
            origin={"trust": "unattributed"},
            root_invocation_id="c092a88f-bc0a-4ec0-a92c-d3b8cc21d8cb",
            source="sdk_task",
        ),
    )
    copied = copy_task(task)
    copied.input["nested"].append("changed")
    assert task.input == {"nested": ["original"]}
    for value in (
        task,
        TaskTopologyNode.from_task(task),
        TaskTopologyQuery(expanded_parent_ids=(task.id,)),
        TaskTopologyStoreResult(observed_at=task.created_at),
    ):
        encoded = pickle.dumps(value)
        assert type(value).__module__.encode() in encoded
        restored = pickle.loads(encoded)
        assert type(restored) is type(value)
        assert restored.model_dump(mode="json") == value.model_dump(mode="json")
