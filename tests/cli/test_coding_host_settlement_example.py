"""Application transaction characterization; full worker-loss proof is separate."""

import asyncio
import subprocess
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from tests.core.test_coding_products import _request

from cayu import (
    Event,
    EventType,
    ResolutionActor,
    ResolutionActorSource,
    SQLiteTaskStore,
    TaskCancellationReconciliationEvidence,
    TaskCancellationReconciliationOutcome,
    TaskCancellationReconciliationRequest,
    TaskCreate,
    TaskStatus,
)
from cayu.guides.coding_host import BusinessStore, Reservation, SettlementConflict


def _events():
    """Minimal normalized transcript; no qualification-host dependency."""
    return (
        Event(
            type=EventType.MODEL_STARTED,
            session_id="fixture",
            payload={"step": 1, "attempt": 1, "model": "fixture"},
        ),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="fixture",
            payload={
                "step": 1,
                "attempt": 1,
                "model": "fixture",
                "status": "completed",
                "completion": {"finish_reason": "tool_calls", "status": "completed"},
            },
        ),
        Event(type=EventType.TOOL_CALL_STARTED, session_id="fixture", tool_name="run_check"),
        Event(
            type=EventType.TOOL_CALL_FAILED,
            session_id="fixture",
            tool_name="run_check",
            payload={
                "result": {
                    "structured": {
                        "status": "timed_out",
                        "workspace_mutation_settlement": "complete",
                        "cleanup_uncertain": False,
                    }
                }
            },
        ),
    )


def reservation():
    from cayu import ModelPrice, PriceBook

    return Reservation(
        tenant="example-tenant",
        public_id="example-request",
        request=_request(baseline="sha256:" + "a" * 64),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fixture",
                    model="fixture",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        ),
        cost_basis="synthetic",
    )


def test_installed_guide_section_contains_complete_executable_example(capsys):
    import ast
    import json
    import re

    from cayu.cli import main

    assert main(["guide", "authoring#coding-host-settlement-example", "--json"]) == 0
    content = json.loads(capsys.readouterr().out)["content"]
    snippets = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
    assert len(snippets) == 1
    ast.parse(snippets[0])
    assert "cost_basis=" in snippets[0]
    assert "relabelling" in content or "relabeling" in content
    assert "Historical reads" in content


@pytest.mark.parametrize(
    "completion",
    [
        None,
        {},
        {"finish_reason": "unknown"},
        {"finish_reason": "stop", "status": "failed"},
        {"finish_reason": "tool_calls", "status": "outcome_unknown"},
    ],
)
def test_model_completion_requires_positive_normalized_evidence(completion):
    from cayu.guides.coding_host_evidence import (
        MaintenanceReconciliationUnavailable,
        _require_serial_check_quiescence,
    )

    events = _events()
    events[1].payload["completion"] = completion
    with pytest.raises(MaintenanceReconciliationUnavailable):
        _require_serial_check_quiescence(events)


@pytest.mark.parametrize("raw_status", [None, "completed", "failed", "outcome_unknown"])
def test_normalized_completion_does_not_hide_conflicting_raw_status(raw_status):
    from cayu.guides.coding_host_evidence import (
        MaintenanceReconciliationUnavailable,
        _require_serial_check_quiescence,
    )

    events = _events()
    events[1].payload["status"] = raw_status
    if raw_status in {None, "completed"}:
        _require_serial_check_quiescence(events)
    else:
        with pytest.raises(MaintenanceReconciliationUnavailable):
            _require_serial_check_quiescence(events)


def test_explicit_completed_response_does_not_require_chat_finish_reason():
    from cayu.guides.coding_host_evidence import _require_serial_check_quiescence

    events = _events()
    events[1].payload["completion"] = {"finish_reason": "unknown", "status": "completed"}
    _require_serial_check_quiescence(events)


def test_business_exact_replay_never_releases_a_new_source_owner(tmp_path):
    path = tmp_path / "business.sqlite"
    store = BusinessStore(path)
    original = reservation()
    store.reserve(original)
    assert store.read(original) is None
    newer = original.model_copy(update={"public_id": "another-request"})
    with pytest.raises(SettlementConflict, match="unsettled owner"):
        store.reserve(newer)
    result = {"classification": "cancelled", "receipt": "exact-native-receipt"}
    store.settle(original, result)
    # Model a lost reply by reopening without using the returned value.
    restarted = BusinessStore(path)
    assert restarted.settle(original, result) == result
    restarted.reserve(newer)
    assert restarted.settle(original, result) == result
    assert restarted.read(newer) is None
    with pytest.raises(SettlementConflict, match="settlement changed"):
        restarted.settle(original, {**result, "receipt": "different"})
    changed = original.model_copy(update={"tenant": "other-tenant"})
    with pytest.raises(SettlementConflict, match="absent or conflicting"):
        restarted.read(changed)


def test_malformed_copied_reservation_is_rejected_without_diagnostic_leak(tmp_path, capsys, caplog):
    import warnings

    from pydantic import ValidationError

    class Rejected:
        def __repr__(self):
            return "PRIVATE-RESERVATION-CANARY"

    store = BusinessStore(tmp_path / "business.sqlite")
    original = reservation()
    invalid = original.model_copy(
        update={"request": original.request.model_copy(update={"task": Rejected()})}
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValidationError) as error:
            store.reserve(invalid)
    output = capsys.readouterr()
    assert "PRIVATE-RESERVATION-CANARY" not in (
        str(error.value)
        + repr(error.value)
        + caplog.text
        + output.out
        + output.err
        + str([str(item.message) for item in captured])
    )
    store.reserve(original)
    assert store.read(original) is None


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "product_run_id", "other-product"),
        (None, "session_id", "other-session"),
        (None, "agent_name", "other-agent"),
        (None, "parent_session_id", "other-parent"),
        (None, "causal_budget_id", "other-budget"),
        ("task", "task_id", "other-task"),
        ("task", "instruction_sha256", "sha256:" + "b" * 64),
        ("source", "workspace_id", "other-workspace"),
        ("source", "origin_id", "other-origin"),
        ("source", "destination_id", "other-destination"),
        ("source", "baseline_revision", "sha256:" + "b" * 64),
        ("runtime", "execution_profile_fingerprint", "sha256:" + "b" * 64),
        ("runtime", "tool_policy_fingerprint", "sha256:" + "b" * 64),
        ("runtime", "approval_policy_fingerprint", "sha256:" + "b" * 64),
        ("settlement", "reviewer_required", True),
    ],
)
def test_business_binding_rejects_changed_native_authority(tmp_path, section, field, value):
    store = BusinessStore(tmp_path / "business.sqlite")
    original = reservation()
    store.reserve(original)
    request = original.request.model_copy(
        update={field: value}
        if section is None
        else {section: getattr(original.request, section).model_copy(update={field: value})}
    )
    with pytest.raises(SettlementConflict, match="absent or conflicting"):
        store.read(original.model_copy(update={"request": request}))
    assert store.read(original) is None


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_native_receipt_ack_loss_then_application_ack_loss(tmp_path, backend):
    """Real native task entrance; evidence validation is intentionally not claimed here."""
    from cayu.tasks.base import InMemoryTaskStore as Memory

    clock = [datetime.now(UTC)]

    async def scenario():
        native = (
            Memory(clock=lambda: clock[0], ownership_clock=lambda: clock[0])
            if backend == "memory"
            else SQLiteTaskStore(
                tmp_path / "tasks.sqlite", clock=lambda: clock[0], ownership_clock=lambda: clock[0]
            )
        )
        business_path = tmp_path / "business.sqlite"
        business = BusinessStore(business_path)
        expected = reservation()
        business.reserve(expected)
        try:
            await native.create_task(
                TaskCreate(task_id=expected.request.task.task_id, type="example")
            )
            claimed = await native.claim_task("stopped-worker", lease_seconds=1)
            assert claimed is not None
            assert claimed.worker_id is not None and claimed.lease_expires_at is not None
            await native.mark_claimed_task_execution_started(
                claimed.id,
                claimed.worker_id,
                claimed.lease_expires_at,
            )
            await native.cancel_task(claimed.id, {"code": "operator"})
            task = await native.load_task(claimed.id)
            assert task is not None and task.status_payload is not None
            assert task.worker_id is not None and task.lease_expires_at is not None
            clock[0] += timedelta(seconds=2)
            request = TaskCancellationReconciliationRequest(
                task_id=task.id,
                original_worker_id=task.worker_id,
                original_lease_expires_at=task.lease_expires_at,
                cancellation_requested_at=task.status_payload["event"]["occurred_at"],
                cancellation_idempotency_key=task.status_payload["terminalization_idempotency_key"],
                reconciliation_idempotency_key="one-exact-operation",
                reconciliation_requested_at=clock[0],
                reconciled_by=ResolutionActor(
                    subject="operator",
                    tenant=expected.tenant,
                    source=ResolutionActorSource.HTTP_AUTH,
                ),
                evidence=TaskCancellationReconciliationEvidence(
                    outcome=TaskCancellationReconciliationOutcome.QUIESCENT,
                    validator_id="controlled-owner",
                    validator_version="1",
                    evidence_id="controlled-evidence",
                    evidence_sha256="a" * 64,
                    validated_at=clock[0],
                ),
            )
            business.prepare(expected, request)
            committed = await native.reconcile_task_cancellation(request)
            if backend == "sqlite":
                assert isinstance(native, SQLiteTaskStore)
                await native.close()
                native = SQLiteTaskStore(
                    tmp_path / "tasks.sqlite",
                    clock=lambda: clock[0],
                    ownership_clock=lambda: clock[0],
                )
            business = BusinessStore(business_path)
            pending = business.pending(expected)
            assert pending is not None
            replay = await native.reconcile_task_cancellation(pending)
            assert replay == committed
            assert replay.task.status is TaskStatus.CANCELLED
            projection = {"receipt": replay.reconciliation.events[-1].id, "cost": None}
            business.settle(expected, projection)
            assert BusinessStore(business_path).settle(expected, projection) == projection
            assert BusinessStore(business_path).pending(expected) == request
        finally:
            if isinstance(native, SQLiteTaskStore):
                await native.close()

    asyncio.run(scenario())


def test_historical_readback_is_distinct_from_current_source_bytes(tmp_path):
    """Real artifact and workspace APIs; candidate evidence is a controlled fixture."""
    from tests.core.test_coding_products import _check_event, _terminal_events

    from cayu import (
        CayuApp,
        CodingProductArtifactRepository,
        CodingProductRunner,
        InMemoryTaskStore,
        TaskQuery,
        complete_managed_task,
        run_task_worker,
    )
    from cayu.artifacts import LocalArtifactStore
    from cayu.budgets.pricing import ModelPrice, PriceBook
    from cayu.coding_products import compile_coding_product_candidate
    from cayu.guides.coding_host import project_result, require_current_source, settle_completed
    from cayu.workspaces import LocalWorkspace
    from cayu.workspaces.revisions import (
        WorkspaceRevisionObservationLimits,
        observe_deterministic_workspace,
    )

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    target = source / "example.py"
    target.write_text("before\n")
    subprocess.run(["git", "-C", str(source), "add", "example.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture base",
        ],
        check=True,
    )
    workspace = LocalWorkspace(
        source, workspace_id="source-workspace", excluded_directory_names=(".git",)
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fixture",
                model="fixture",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
        )
    )

    async def scenario():
        limits = WorkspaceRevisionObservationLimits()
        initial = await observe_deterministic_workspace(
            workspace, observer="cayu-coding-product-source", limits=limits
        )
        assert initial.revision is not None
        request = _request(baseline=initial.revision)
        expected = Reservation(
            tenant="example",
            public_id="original",
            request=request,
            pricing=pricing,
            cost_basis="synthetic",
        )
        target.write_text("after\n")
        final = await observe_deterministic_workspace(
            workspace, observer="cayu-coding-product-source", limits=limits
        )
        assert final.revision is not None
        repository = CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts"))
        events = (
            *(
                _check_event(name, workspace_revision=final.revision)
                for name in request.settlement.required_checks
            ),
            *_terminal_events(request=request, workspace_revision=final.revision, changed=True),
        )
        candidate = await compile_coding_product_candidate(
            request,
            events,
            initial_observation=initial,
            final_observation=final,
            repository=repository,
        )
        publication = await repository.publish_candidate(candidate)

        async def no_delivery(_expected):
            raise AssertionError("readback must not request Git publication")

        task_store = InMemoryTaskStore()
        app = CayuApp(task_store=task_store)
        runner = CodingProductRunner(
            app,
            source_workspace=workspace,
            repository=repository,
            source_git_authority_validator=no_delivery,
        )
        projected = await project_result(
            runner,
            expected,
            pricing=pricing,
            cost_basis="synthetic",
            result_digest=publication.result_reference.digest,
        )
        assert projected["cost"]["basis"] == "synthetic"
        assert projected["cost"]["estimated_total"] is None
        await require_current_source(runner, expected, projected)
        business = BusinessStore(tmp_path / "business.sqlite")
        business.reserve(expected)
        await app.create_task(TaskCreate(task_id=request.task.task_id, type="example"))

        async def handler(_app, claimed, worker):
            await complete_managed_task(
                task_store,
                claimed,
                worker,
                {
                    "product_run_id": request.product_run_id,
                    "result_digest": publication.result_reference.digest,
                },
            )

        await run_task_worker(
            app,
            task_store,
            handler,
            worker_id="normal-worker",
            query=TaskQuery(type="example"),
            max_tasks=1,
        )
        assert (
            await settle_completed(
                runner,
                business,
                expected,
                result_digest=publication.result_reference.digest,
                pricing=pricing,
                cost_basis="synthetic",
            )
            == projected
        )
        status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"])
        target.write_text("unrelated later content\n")
        assert (
            subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]) == status
        )
        historical = BusinessStore(tmp_path / "business.sqlite").read(expected)
        assert historical is not None
        assert historical == projected
        assert (
            await settle_completed(
                runner,
                BusinessStore(tmp_path / "business.sqlite"),
                expected,
                result_digest=publication.result_reference.digest,
                pricing=pricing,
                cost_basis="synthetic",
            )
            == projected
        )
        assert (
            await project_result(
                runner,
                expected,
                pricing=pricing,
                cost_basis="synthetic",
                result_digest=publication.result_reference.digest,
            )
            == projected
        )
        with pytest.raises(SettlementConflict, match="Current source bytes"):
            await require_current_source(runner, expected, historical)

    asyncio.run(scenario())
