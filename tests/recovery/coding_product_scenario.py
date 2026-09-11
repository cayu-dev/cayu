"""Coding-product terminal recovery using the existing process-loss harness.

The ordinary environment deliberately supplies no Docker publication receipt;
this qualifies exact non-success reconstruction, not a verified coding change.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Never
from unittest.mock import patch

from worker_harness import (
    BackendConfig,
    _append_json_line,
    _RecoveryProvider,
    _session_store,
    _task_store,
    _wait_for_task_lease_expiry,
    _write_json_atomic,
)

from cayu import (
    AgentSpec,
    CayuApp,
    CodingGitBaselineAuthority,
    CodingLifecycleReceipt,
    CodingProductArtifactRepository,
    CodingProductCandidate,
    CodingProductRunner,
    CodingProductState,
    CodingRuntimeAuthority,
    ExecutionProfileBehaviorIdentity,
    LocalArtifactStore,
    LocalWorkspace,
    Message,
    ResolutionActor,
    ResolutionActorSource,
    RunRequest,
    TaskCancellationReconciliationEvent,
    TaskCancellationReconciliationEvidence,
    TaskCancellationReconciliationOutcome,
    TaskCancellationReconciliationRequest,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    admit_coding_product_request,
    run_task_worker,
)
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.runtime import (
    RecoveryDecision,
    RecoveryExecutionRequest,
    RecoveryItemExecutionStatus,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
)


def _digest(value: str) -> str:
    return "sha256:" + sha256(value.encode()).hexdigest()


class _CodingProvider(_RecoveryProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(
            "model_start" if config["publication_phase"] == "model" else "complete",
            Path(config["phase_path"]),
        )
        self.config = config

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        assert self.config["action"] == "start", "recovery dispatched a provider"
        _append_json_line(
            Path(self.config["marker_path"]),
            {"session_id": self.config["session_id"], "operation": "provider_dispatch"},
        )
        async for event in super().stream(request):
            yield event


async def run_coding_product(config: dict[str, Any]) -> dict[str, Any]:
    if not config.get("managed_task", False):
        return await _run_coding_product(config)
    task_store = _task_store(BackendConfig.from_json(config["backend"]))
    task_id = "coding-process-task"
    task_type = "coding-product-recovery"
    try:
        if config["action"] == "start":
            await task_store.create_task(TaskCreate(task_id=task_id, type=task_type))
        else:
            await _wait_for_task_lease_expiry(task_store, task_id)
            query = TaskQuery(type=task_type)
            # Handler dispatch was durable: expiry fences, it never grants a
            # replacement permission to repeat opaque work.
            assert await task_store.reclaim_expired(query=query) == []
            assert await task_store.claim_task("replacement", query, lease_seconds=10) is None
            held = await task_store.load_task(task_id)
            assert held is not None and held.status_reason == "cancellation_requested"
            assert held.worker_id == "coding-start" and held.lease_expires_at is not None
            assert held.status_payload is not None
            event = TaskCancellationReconciliationEvent.model_validate(held.status_payload["event"])
            result = await _run_coding_product(config, task_store=task_store)
            assert result["state"] in {"cancelled", "reconstruction_required"}
            # The harness already proved the sole local provider process died.
            # This scenario has no external environment allocation. Registered
            # recovery and exact product readback above prove Runtime settlement;
            # these facts would NOT validate a live Docker/remote-provider owner.
            validated_at = datetime.now(UTC)
            reconciliation = TaskCancellationReconciliationRequest(
                task_id=held.id,
                original_worker_id=held.worker_id,
                original_lease_expires_at=held.lease_expires_at,
                cancellation_requested_at=event.occurred_at,
                cancellation_idempotency_key=held.status_payload["terminalization_idempotency_key"],
                reconciliation_idempotency_key="coding-process-settlement",
                reconciliation_requested_at=validated_at,
                reconciled_by=ResolutionActor(
                    subject="coding-process-harness", source=ResolutionActorSource.SYSTEM
                ),
                evidence=TaskCancellationReconciliationEvidence(
                    outcome=TaskCancellationReconciliationOutcome.QUIESCENT,
                    validator_id="coding-process-harness",
                    validator_version="1",
                    evidence_id=config["product_run_id"],
                    evidence_sha256=result["digest"],
                    validated_at=validated_at,
                    execution_profile_fingerprint=held.metadata.get(
                        "execution_profile_fingerprint"
                    ),
                    effect_fingerprint=held.metadata.get("effect_fingerprint"),
                ),
                expected_execution_profile_fingerprint=held.metadata.get(
                    "execution_profile_fingerprint"
                ),
                expected_effect_fingerprint=held.metadata.get("effect_fingerprint"),
            )
            terminal = await task_store.reconcile_task_cancellation(reconciliation)
            assert await task_store.reconcile_task_cancellation(reconciliation) == terminal
            current = await task_store.load_task(task_id)
            assert current is not None and current.status is TaskStatus.CANCELLED
            assert current.worker_id is None and current.lease_expires_at is None
            return {**result, "task": current.model_dump(mode="json")}

        async def handler(_app, claimed, worker_id):
            assert claimed.id == task_id
            assert worker_id == "coding-start"
            await _run_coding_product(config, task_store=task_store)
            raise AssertionError("Original coding owner should remain at the SIGKILL barrier.")

        assert (
            await run_task_worker(
                CayuApp(task_store=task_store, enable_logging=False),
                task_store,
                handler,
                worker_id="coding-" + config["action"],
                query=TaskQuery(type=task_type),
                lease_seconds=10,
                poll_interval_s=0.01,
                max_tasks=1,
            )
            == 1
        )
        raise AssertionError("Original coding worker should not return before SIGKILL.")
    finally:
        await task_store.close()


async def _run_coding_product(config: dict[str, Any], *, task_store=None) -> dict[str, Any]:
    store = _session_store(BackendConfig.from_json(config["backend"]))
    try:
        app = CayuApp(session_store=store, task_store=task_store, enable_logging=False)
        app.register_provider(_CodingProvider(config))
        app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="coding",
                    execution_profile_identity=ExecutionProfileBehaviorIdentity(
                        name="tests:coding-product-process-environment",
                        behavior_version="1",
                        implementation_version="1",
                    ),
                )
            )
        )
        app.register_agent(AgentSpec(name="coder", model="fake-model"))
        workspace = LocalWorkspace(Path(config["source_path"]), workspace_id="source-workspace")
        repository = CodingProductArtifactRepository(
            LocalArtifactStore(
                Path(config["artifact_path"]),
                store_id="coding-process-artifacts",
            )
        )
        run_request = RunRequest(
            agent_name="coder",
            session_id=config["session_id"],
            environment_name="coding",
            messages=[Message.text("user", "repair the bounded fixture")],
        )
        profile = await app.inspect_run_execution_profile(run_request)
        request = await admit_coding_product_request(
            product_run_id=config["product_run_id"],
            session_id=config["session_id"],
            agent_name="coder",
            task_id="coding-process-task",
            messages=run_request.messages,
            source_workspace=workspace,
            source_origin_id="fixture-origin",
            source_destination_id="fixture-destination",
            source_git_baseline=CodingGitBaselineAuthority(
                head_revision="a" * 40,
                staged_entries_sha256=_digest("index"),
                tracked_flags_sha256=_digest("flags"),
                status_sha256=_digest("status"),
                diff_sha256=_digest("diff"),
            ),
            runtime=CodingRuntimeAuthority(
                toolchain_profile_id="fixture",
                toolchain_profile_revision="1",
                toolchain_profile_fingerprint=_digest("toolchain"),
                image_fingerprint=_digest("image"),
                dependency_identity=_digest("dependencies"),
                execution_profile_fingerprint=profile,
                tool_manifest_fingerprint=_digest("tools"),
                tool_policy_fingerprint=_digest("policy"),
                approval_policy_fingerprint=_digest("approval"),
                redaction_profile_fingerprint=_digest("redaction"),
            ),
        )

        async def validate_fixture_git(expected: CodingGitBaselineAuthority) -> None:
            assert expected == request.source.git_baseline

        runner = CodingProductRunner(
            app,
            source_workspace=workspace,
            repository=repository,
            source_git_authority_validator=validate_fixture_git,
        )
        if config["action"] == "start":

            async def barrier(candidate_digest: str | None = None) -> Never:
                _write_json_atomic(
                    Path(config["phase_path"]),
                    {"phase": "coding_session_settled", "candidate_digest": candidate_digest},
                )
                await asyncio.Event().wait()
                raise AssertionError("unreachable process-loss barrier")

            async def before_compile(*args: Any, **kwargs: Any) -> Never:
                await barrier()

            async def before_publish(candidate: CodingProductCandidate) -> Never:
                await barrier(candidate.digest)

            append = repository.append_lifecycle

            async def after_ready(receipt: CodingLifecycleReceipt):
                result = await append(receipt)
                if receipt.state is CodingProductState.READY_TO_PUBLISH:
                    assert receipt.evidence_sha256 is not None
                    await barrier(receipt.evidence_sha256.removeprefix("sha256:"))
                return result

            phase = config["publication_phase"]
            target, method, replacement = {
                "compile": (runner, "_compile_and_publish", before_compile),
                "model": (runner, "_compile_and_publish", before_compile),
                "ready": (repository, "append_lifecycle", after_ready),
                "publish": (repository, "publish_candidate", before_publish),
            }[phase]
            with patch.object(target, method, replacement):
                await runner.run(request, run_request)
            raise AssertionError("expected SIGKILL at the settled-session boundary")

        recovery_receipt = None
        if config["publication_phase"] == "model":
            plan = await app.plan_recovery(
                RecoveryPlanRequest(
                    selection=RecoveryPlanSelection(session_ids=(request.session_id,))
                )
            )
            assert len(plan.items) == 1
            item = plan.items[0]
            assert RecoveryPlanAction.MODEL_MARK_INTERRUPTED in item.allowed_actions
            execution = RecoveryExecutionRequest(
                plan=plan,
                execution_id="coding-product-model-stop",
                decisions=(
                    RecoveryDecision(
                        item_id=item.item_id,
                        action=RecoveryPlanAction.MODEL_MARK_INTERRUPTED,
                    ),
                ),
            )
            recovery_receipt = await app.execute_recovery(execution)
            assert recovery_receipt.items[0].status is RecoveryItemExecutionStatus.EXECUTED
            replay_receipt = await app.execute_recovery(execution)
            assert replay_receipt.items[0].replayed
            assert (
                replay_receipt.items[0].receipt_event_id
                == recovery_receipt.items[0].receipt_event_id
            )
            checkpoint = await store.load_checkpoint(request.session_id)
            assert checkpoint is not None
            _write_json_atomic(
                Path(config["phase_path"]),
                {
                    "phase": "registered_model_stop",
                    "planned_epoch": item.run_epoch,
                    "final_epoch": recovery_receipt.items[0].final_run_epoch,
                    "active_epoch": checkpoint["active_invocation_execution_profile"]["run_epoch"],
                },
            )

        publication = await runner.recover_settled_execution(request)
        replay = await runner.recover_settled_execution(request)
        assert publication == replay
        lifecycle, _ = await repository.load_lifecycle(
            request.product_run_id,
            session_id=request.session_id,
            request_fingerprint=request.fingerprint,
        )
        return {
            "state": publication.candidate.state.value,
            "digest": publication.candidate.digest,
            "replay_digest": replay.candidate.digest,
            "session_id": publication.candidate.session_id,
            "initial_revision": publication.candidate.initial_revision,
            "final_revision": publication.candidate.final_revision,
            "terminal_lifecycle_state": lifecycle[-1].state.value,
            "recovery_receipt": None
            if recovery_receipt is None
            else recovery_receipt.model_dump(mode="json"),
        }
    finally:
        await store.close()
