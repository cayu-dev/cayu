"""API-key-free stateful evaluation from one portable AgentSnapshot.

The demo captures one logical baseline, materializes two isolated candidates,
runs the same case through ordinary Cayu sessions, records exact result lineage,
then starts a fresh Python process to recover the materializations without
repeating candidate, runtime, or evaluator effects.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import tempfile
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

from cayu import (
    AgentSnapshot,
    AgentSnapshotAccess,
    AgentSnapshotAuthorityRef,
    AgentSnapshotCaptureRequest,
    AgentSnapshotCompleteness,
    AgentSnapshotComponentCapture,
    AgentSnapshotComponentKind,
    AgentSnapshotComponentProvider,
    AgentSnapshotComponentRef,
    AgentSnapshotComponentSelector,
    AgentSnapshotConsistency,
    AgentSnapshotCoordinator,
    AgentSnapshotLearningDisposition,
    AgentSnapshotLogicalRef,
    AgentSnapshotMaterialization,
    AgentSnapshotMaterializationCapability,
    AgentSnapshotMaterializationOperation,
    AgentSnapshotMaterializationRequest,
    AgentSnapshotMaterializedComponent,
    AgentSnapshotOverlayKind,
    AgentSnapshotOverlayRef,
    AgentSnapshotRedaction,
    AgentSnapshotResultBinding,
    AgentSnapshotSubject,
    AgentSnapshotTerminalDisposition,
    AgentSnapshotTrialStateMode,
    AgentSpec,
    CayuApp,
    MemoryStateRef,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SQLiteAgentSnapshotStore,
    Trajectory,
    WorkspaceIdentity,
    WorkspacePathRevision,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationStatus,
    app_body_snapshot_ref,
    build_execution_profile_identity,
    execution_profile_snapshot_ref,
    run_to_completion,
    trajectory_snapshot_ref,
    workspace_snapshot_ref,
)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


def _snapshot_access(snapshot: AgentSnapshot) -> AgentSnapshotAccess:
    return AgentSnapshotAccess(
        snapshot=snapshot.ref,
        binding_id=snapshot.identity_binding.binding_id,
        authority_scope_fingerprint=snapshot.authority_scope_fingerprint,
    )


def _logical_ref(
    name: str,
    *,
    scope_fingerprint: str | None = None,
    revision: str | None = None,
    frontier: str | None = None,
) -> AgentSnapshotLogicalRef:
    return AgentSnapshotLogicalRef(
        fingerprint=_fingerprint({"logical_component": name}),
        revision=revision or f"demo-revision:{name}",
        frontier=frontier,
        scope_fingerprint=scope_fingerprint,
        source_ref=f"cayu-ref:demo:{name}",
    )


class _DemoJournal:
    """Durable proof that demo effects happen once and stay candidate-scoped."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_snapshot_demo_effects (
                    effect_kind TEXT NOT NULL,
                    effect_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (effect_kind, effect_id)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def record_once(self, kind: str, identity: str, payload: dict[str, Any]) -> bool:
        document = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO agent_snapshot_demo_effects "
                "(effect_kind, effect_id, payload) VALUES (?, ?, ?)",
                (kind, identity, document),
            )
        return cursor.rowcount == 1

    def require(self, kind: str, identity: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM agent_snapshot_demo_effects "
                "WHERE effect_kind = ? AND effect_id = ?",
                (kind, identity),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Missing durable demo effect {kind}:{identity}.")
        value = json.loads(row["payload"])
        if not isinstance(value, dict):
            raise RuntimeError("Demo journal payload is not an object.")
        return value

    def count(self, kind: str, identity: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM agent_snapshot_demo_effects "
                "WHERE effect_kind = ? AND effect_id = ?",
                (kind, identity),
            ).fetchone()
        return 0 if row is None else int(row["count"])

    def materialize_operation_once(
        self,
        operation_id: str,
        component: AgentSnapshotMaterializedComponent,
    ) -> AgentSnapshotMaterializedComponent:
        component_document = json.dumps(
            component.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload FROM agent_snapshot_demo_effects "
                "WHERE effect_kind = 'materialization-operation' AND effect_id = ?",
                (operation_id,),
            ).fetchone()
            if existing is not None:
                return AgentSnapshotMaterializedComponent.model_validate_json(existing["payload"])
            if component.overlay is not None:
                overlay_document = json.dumps(
                    {
                        "baseline_fingerprint": component.overlay.baseline_fingerprint,
                        "candidate_id": component.overlay.candidate_id,
                        "kind": component.overlay.kind.value,
                        "writes": [],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                connection.execute(
                    "INSERT INTO agent_snapshot_demo_effects "
                    "(effect_kind, effect_id, payload) VALUES ('overlay', ?, ?)",
                    (component.overlay.fingerprint, overlay_document),
                )
            connection.execute(
                "INSERT INTO agent_snapshot_demo_effects "
                "(effect_kind, effect_id, payload) "
                "VALUES ('materialization-operation', ?, ?)",
                (operation_id, component_document),
            )
        return component


class _DemoComponentProvider(AgentSnapshotComponentProvider):
    def __init__(
        self,
        capture: AgentSnapshotComponentCapture,
        journal: _DemoJournal,
    ) -> None:
        self.kind = capture.component.kind
        self.provider_id = capture.component.provider_id
        self._capture = capture
        self._journal = journal

    async def capture(
        self,
        request: AgentSnapshotCaptureRequest,
        selector: AgentSnapshotComponentSelector,
    ) -> AgentSnapshotComponentCapture:
        return self._capture

    async def verify(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
    ) -> bool:
        return snapshot.component(self.kind) == component

    async def materialize(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        return self._materialize_once(component, request, operation)

    async def recover_materialization_operation(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        return self._materialize_once(component, request, operation)

    def _materialize_once(
        self,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        overlay = None
        if self.kind in {
            AgentSnapshotComponentKind.MEMORY,
            AgentSnapshotComponentKind.WORKSPACE,
        }:
            overlay_kind = (
                AgentSnapshotOverlayKind.MEMORY
                if self.kind is AgentSnapshotComponentKind.MEMORY
                else AgentSnapshotOverlayKind.WORKSPACE
            )
            overlay = AgentSnapshotOverlayRef.create(
                kind=overlay_kind,
                overlay_id=f"{self.kind.value}-{request.state_scope_id[:16]}",
                baseline_fingerprint=component.logical.fingerprint,
                candidate_id=request.candidate_id,
                state_scope_id=request.state_scope_id,
                source_ref=f"cayu-ref:demo-overlay:{self.kind.value}:{request.candidate_id}",
            )
        result = AgentSnapshotMaterializedComponent(
            kind=self.kind,
            baseline_fingerprint=component.logical.fingerprint,
            capability=component.materialization,
            materialization_ref=f"cayu-ref:demo-materialization:{self.kind.value}",
            overlay=overlay,
        )
        return self._journal.materialize_operation_once(operation.operation_id, result)

    async def recover(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        materialized: AgentSnapshotMaterializedComponent,
        materialization: AgentSnapshotMaterialization,
    ) -> AgentSnapshotMaterializedComponent:
        if materialized.overlay is not None:
            self._journal.require("overlay", materialized.overlay.fingerprint)
        return materialized


def _component(
    kind: AgentSnapshotComponentKind,
    logical: AgentSnapshotLogicalRef,
    capability: AgentSnapshotMaterializationCapability,
) -> AgentSnapshotComponentRef:
    return AgentSnapshotComponentRef(
        kind=kind,
        provider_id=f"cayu.demo.{kind.value}.v1",
        logical=logical,
        consistency=AgentSnapshotConsistency.FRONTIER_CONSISTENT,
        completeness=AgentSnapshotCompleteness.COMPLETE,
        redaction=AgentSnapshotRedaction.BOUNDED_PROJECTION,
        materialization=capability,
        required=True,
        limitations=(),
    )


def _capture_contract() -> tuple[
    AgentSnapshotCaptureRequest, tuple[AgentSnapshotComponentCapture, ...]
]:
    scope = _fingerprint({"application": "snapshot-demo", "project": "reference"})
    baseline_app = CayuApp(enable_logging=False)
    baseline_app.register_agent(
        AgentSpec(
            name="assistant",
            model="scripted-model",
            system_prompt="Baseline policy; candidate changes are not active.",
        )
    )
    body = app_body_snapshot_ref(
        baseline_app.describe(),
        application_release_id="snapshot-demo-v1",
        agent_id="assistant",
        source_ref="cayu-ref:demo:body-release-v1",
    )
    profile = execution_profile_snapshot_ref(
        build_execution_profile_identity(
            runtime_name="cayu",
            runtime_version="reference-demo-v1",
            provider_name="scripted",
            model="scripted-model",
            durable_system_prompt="Baseline policy; candidate changes are not active.",
            direct_tools=(),
            tool_catalogue_revision=f"sha256:{_fingerprint('empty-tool-catalogue')}",
        )
    )
    profile_logical = AgentSnapshotLogicalRef(
        fingerprint=profile.fingerprint,
        revision=f"execution-profile:{profile.schema_version}:{profile.fingerprint}",
        scope_fingerprint=scope,
        source_ref="cayu-ref:demo:execution-profile-v1",
    )
    memory = MemoryStateRef.create(
        knowledge=_logical_ref(
            "knowledge",
            scope_fingerprint=scope,
            frontier="knowledge-change:0",
        ),
        transcript_evidence=_logical_ref(
            "transcript-evidence",
            scope_fingerprint=scope,
            frontier="session-event:0",
        ),
        artifact_evidence=_logical_ref(
            "artifact-evidence",
            scope_fingerprint=scope,
            frontier="artifact:0",
        ),
        work_context=_logical_ref(
            "work-context",
            scope_fingerprint=scope,
            revision="checkpoint:0",
        ),
        recall_policy=_logical_ref("recall-policy", scope_fingerprint=scope),
        admission_policy=_logical_ref("admission-policy", scope_fingerprint=scope),
        context_projection_policy=_logical_ref(
            "context-projection-policy", scope_fingerprint=scope
        ),
        recall_receipts=_logical_ref(
            "recall-receipts",
            scope_fingerprint=scope,
            frontier="recall-receipt:0",
        ),
        context_exposures=_logical_ref(
            "context-exposures",
            scope_fingerprint=scope,
            frontier="context-exposure:0",
        ),
        index_readiness=_logical_ref(
            "index-readiness",
            scope_fingerprint=scope,
            frontier="index-generation:1",
        ),
        learning_disposition=AgentSnapshotLearningDisposition.ISOLATED,
        limitations=("interaction_focus_not_active",),
    )
    memory_logical = AgentSnapshotLogicalRef(
        fingerprint=memory.fingerprint,
        frontier="memory-frontier:0",
        scope_fingerprint=scope,
        source_ref="cayu-ref:demo:memory-view-v1",
    )
    trajectory = trajectory_snapshot_ref(
        Trajectory(final_output="baseline has no candidate result"),
        scope_fingerprint=scope,
        source_ref="cayu-ref:demo:trajectory-v1",
    ).model_copy(update={"frontier": "session-event:0"})
    workspace = workspace_snapshot_ref(
        WorkspaceRevisionObservation(
            identity=WorkspaceIdentity(workspace_id="logical-demo-workspace", observer="demo"),
            status=WorkspaceRevisionObservationStatus.SUPPORTED,
            revision="workspace-baseline-v1",
            paths=(
                WorkspacePathRevision(
                    path="policy.json",
                    working_tree="present",
                    present=True,
                    kind="file",
                    content_sha256=_fingerprint({"policy": "baseline"}),
                ),
            ),
            total_paths=1,
        ),
        scope_fingerprint=scope,
        source_ref="cayu-ref:demo:workspace-baseline-v1",
    )
    environment = _logical_ref(
        "environment-fixture",
        scope_fingerprint=scope,
        revision="fixture:local-scripted-v1",
    )

    subject = AgentSnapshotSubject(
        agent_id="assistant",
        application_id="snapshot-demo",
        project_id="reference-experiment",
        body_release=body,
    )
    component_captures = (
        AgentSnapshotComponentCapture(
            component=_component(
                AgentSnapshotComponentKind.BODY,
                body,
                AgentSnapshotMaterializationCapability.RESTORABLE,
            )
        ),
        AgentSnapshotComponentCapture(
            component=_component(
                AgentSnapshotComponentKind.ENVIRONMENT,
                environment,
                AgentSnapshotMaterializationCapability.REFERENCE_ONLY,
            )
        ),
        AgentSnapshotComponentCapture(
            component=_component(
                AgentSnapshotComponentKind.EXECUTION_PROFILE,
                profile_logical,
                AgentSnapshotMaterializationCapability.REPLAYABLE,
            ),
            execution_profile=profile,
        ),
        AgentSnapshotComponentCapture(
            component=_component(
                AgentSnapshotComponentKind.MEMORY,
                memory_logical,
                AgentSnapshotMaterializationCapability.RESTORABLE,
            ),
            memory_state=memory,
        ),
        AgentSnapshotComponentCapture(
            component=_component(
                AgentSnapshotComponentKind.SESSION,
                trajectory,
                AgentSnapshotMaterializationCapability.REPLAYABLE,
            )
        ),
        AgentSnapshotComponentCapture(
            component=_component(
                AgentSnapshotComponentKind.WORKSPACE,
                workspace,
                AgentSnapshotMaterializationCapability.RESTORABLE,
            )
        ),
    )
    evaluator = AgentSnapshotAuthorityRef(
        identity=_logical_ref("hidden-evaluator", scope_fingerprint=scope)
    )
    promotion = AgentSnapshotAuthorityRef(
        identity=_logical_ref("promotion-authority", scope_fingerprint=scope)
    )
    request = AgentSnapshotCaptureRequest(
        capture_request_id="reference-capture-v1",
        subject=subject,
        authority_scope_fingerprint=scope,
        components=tuple(
            AgentSnapshotComponentSelector(kind=capture.component.kind)
            for capture in component_captures
        ),
        required_consistency=AgentSnapshotConsistency.FRONTIER_CONSISTENT,
        session_id="baseline-session",
        environment_name="local-scripted-v1",
        evaluator=evaluator,
        promotion_authority=promotion,
    )
    return request, component_captures


def _providers(
    captures: Iterable[AgentSnapshotComponentCapture],
    journal: _DemoJournal,
) -> tuple[_DemoComponentProvider, ...]:
    return tuple(_DemoComponentProvider(capture, journal) for capture in captures)


def _captures_from_snapshot(
    snapshot: AgentSnapshot,
) -> tuple[AgentSnapshotComponentCapture, ...]:
    return tuple(
        AgentSnapshotComponentCapture(
            component=component,
            execution_profile=(
                snapshot.execution_profile
                if component.kind is AgentSnapshotComponentKind.EXECUTION_PROFILE
                else None
            ),
            memory_state=(
                snapshot.memory_state
                if component.kind is AgentSnapshotComponentKind.MEMORY
                else None
            ),
        )
        for component in snapshot.components
    )


def _overlays(
    materialization: AgentSnapshotMaterialization,
) -> dict[AgentSnapshotOverlayKind, AgentSnapshotOverlayRef]:
    return {
        component.overlay.kind: component.overlay
        for component in materialization.components
        if component.overlay is not None
    }


async def _run_candidate(
    *,
    candidate_id: str,
    answer: str,
    body_policy: str,
    recall_policy: str,
    materialization: AgentSnapshotMaterialization,
    coordinator: AgentSnapshotCoordinator,
    journal: _DemoJournal,
    evaluator_fingerprint: str,
) -> tuple[AgentSnapshotResultBinding, dict[str, Any]]:
    trial_id = f"trial-{candidate_id}"
    trial = await coordinator.begin_trial(
        materialization,
        case_id="hidden-case-1",
        trial_id=trial_id,
        evaluator_fingerprint=evaluator_fingerprint,
    )
    overlays = _overlays(materialization)
    memory_overlay = overlays[AgentSnapshotOverlayKind.MEMORY]
    workspace_overlay = overlays[AgentSnapshotOverlayKind.WORKSPACE]
    if not journal.record_once(
        "candidate-memory-write",
        memory_overlay.fingerprint,
        {"candidate_id": candidate_id, "recall_policy_mutation": recall_policy},
    ):
        raise RuntimeError("Candidate memory mutation would be applied twice.")
    if not journal.record_once(
        "candidate-workspace-write",
        workspace_overlay.fingerprint,
        {"body_policy_mutation": body_policy, "candidate_id": candidate_id},
    ):
        raise RuntimeError("Candidate body mutation would be applied twice.")

    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            (
                ModelStreamEvent.text_delta(answer),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            )
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="scripted-model",
            system_prompt=body_policy,
        )
    )
    outcome = await run_to_completion(
        app,
        RunRequest(
            agent_name="assistant",
            session_id=f"snapshot-demo-{candidate_id}",
            messages=[Message.text("user", "Return the value selected by your private policy.")],
            metadata=trial.session_metadata(),
        ),
    )
    if not outcome.ok:
        raise RuntimeError(f"Candidate runtime failed: {outcome.error}")
    if not journal.record_once(
        "runtime",
        trial.fingerprint,
        {
            "final_text_fingerprint": _fingerprint(outcome.final_text),
            "session_id": outcome.session_id,
            "status": outcome.status.value,
        },
    ):
        raise RuntimeError("Candidate runtime effect would be recorded twice.")

    # The expected value exists only inside the evaluator. It is not placed in
    # the snapshot, candidate session metadata, overlays, or machine result.
    score = int(outcome.final_text == "BLUE")
    if not journal.record_once(
        "evaluator",
        trial.fingerprint,
        {"evaluator_fingerprint": evaluator_fingerprint, "score": score},
    ):
        raise RuntimeError("Evaluator effect would be recorded twice.")
    runtime_evidence = _fingerprint(
        {
            "events": [event.model_dump(mode="json") for event in outcome.events],
            "session_id": outcome.session_id,
            "status": outcome.status.value,
        }
    )
    memory_evidence = _fingerprint(
        {
            "memory_overlay": memory_overlay.fingerprint,
            "mutation": recall_policy,
        }
    )
    result = await coordinator.record_result(
        AgentSnapshotResultBinding.create(
            trial=trial,
            session_id=outcome.session_id,
            terminal_disposition=AgentSnapshotTerminalDisposition.COMPLETED,
            runtime_evidence_fingerprint=runtime_evidence,
            eval_result_revision=_fingerprint({"evaluator": evaluator_fingerprint, "score": score}),
            memory_evidence_fingerprint=memory_evidence,
            usage_fingerprint=_fingerprint({"scripted_model_requests": 1}),
            cost_fingerprint=_fingerprint({"currency": "USD", "value": "0"}),
            recorded_at=materialization.created_at,
        )
    )
    return result, {
        "candidate_id": candidate_id,
        "materialization_fingerprint": materialization.fingerprint,
        "memory_overlay_fingerprint": memory_overlay.fingerprint,
        "result_fingerprint": result.fingerprint,
        "score": score,
        "trial_fingerprint": trial.fingerprint,
        "workspace_overlay_fingerprint": workspace_overlay.fingerprint,
    }


async def recover_reference_experiment(database_path: Path) -> dict[str, Any]:
    """Recover prior materializations in this fresh interpreter process."""

    journal = _DemoJournal(database_path)
    state = journal.require("phase-state", "reference-experiment")
    store = SQLiteAgentSnapshotStore(database_path)
    snapshot = await store.load_snapshot(state["snapshot_fingerprint"])
    if snapshot is None:
        raise RuntimeError("The starting snapshot did not survive restart.")
    coordinator = AgentSnapshotCoordinator(
        _providers(_captures_from_snapshot(snapshot), journal),
        store=store,
    )
    recovered_materializations = []
    recovered_results = []
    for item in state["candidates"]:
        materialization = await coordinator.recover_materialization(
            item["materialization_fingerprint"],
            access=_snapshot_access(snapshot),
        )
        result = await store.load_result(item["result_fingerprint"])
        if result is None:
            raise RuntimeError("A result binding did not survive restart.")
        if journal.count("runtime", item["trial_fingerprint"]) != 1:
            raise RuntimeError("Runtime effect count changed during recovery.")
        if journal.count("evaluator", item["trial_fingerprint"]) != 1:
            raise RuntimeError("Evaluator effect count changed during recovery.")
        if journal.count("candidate-memory-write", item["memory_overlay_fingerprint"]) != 1:
            raise RuntimeError("Candidate memory effect count changed during recovery.")
        if journal.count("candidate-workspace-write", item["workspace_overlay_fingerprint"]) != 1:
            raise RuntimeError("Candidate workspace effect count changed during recovery.")
        for overlay_fingerprint in (
            item["memory_overlay_fingerprint"],
            item["workspace_overlay_fingerprint"],
        ):
            if journal.count("overlay-disposition", overlay_fingerprint) != 1:
                raise RuntimeError("Overlay disposition changed during recovery.")
        recovered_materializations.append(materialization.fingerprint)
        recovered_results.append(result.fingerprint)
    return {
        "candidate_effects_replayed": 0,
        "fresh_process": True,
        "recovered_materializations": recovered_materializations,
        "recovered_results": recovered_results,
        "runtime_effects_replayed": 0,
        "evaluator_effects_replayed": 0,
    }


async def run_reference_experiment(database_path: Path) -> dict[str, Any]:
    """Run the bounded experiment and verify recovery in a new process."""

    journal = _DemoJournal(database_path)
    request, captures = _capture_contract()
    store = SQLiteAgentSnapshotStore(database_path)
    coordinator = AgentSnapshotCoordinator(_providers(captures, journal), store=store)
    snapshot = await coordinator.capture(request)
    baseline_fingerprint = snapshot.fingerprint
    evaluator_fingerprint = snapshot.evaluator.identity.fingerprint if snapshot.evaluator else ""

    candidates = (
        {
            "candidate_id": "candidate-a",
            "answer": "RED",
            "body_policy": "Candidate A policy: answer RED.",
            "recall_policy": "prefer-baseline-recall",
        },
        {
            "candidate_id": "candidate-b",
            "answer": "BLUE",
            "body_policy": "Candidate B policy: answer BLUE.",
            "recall_policy": "prefer-evaluation-recall",
        },
    )
    result_rows = []
    for candidate in candidates:
        materialization = await coordinator.materialize(
            AgentSnapshotMaterializationRequest(
                access=_snapshot_access(snapshot),
                candidate_id=candidate["candidate_id"],
                trial_id=f"trial-{candidate['candidate_id']}",
                state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
            )
        )
        _, result_row = await _run_candidate(
            candidate_id=candidate["candidate_id"],
            answer=candidate["answer"],
            body_policy=candidate["body_policy"],
            recall_policy=candidate["recall_policy"],
            materialization=materialization,
            coordinator=coordinator,
            journal=journal,
            evaluator_fingerprint=evaluator_fingerprint,
        )
        result_rows.append(result_row)

    persisted_snapshot = await store.load_snapshot(snapshot.fingerprint)
    if persisted_snapshot is None or persisted_snapshot.fingerprint != baseline_fingerprint:
        raise RuntimeError("The immutable baseline changed during candidate execution.")
    memory_overlays = {row["memory_overlay_fingerprint"] for row in result_rows}
    workspace_overlays = {row["workspace_overlay_fingerprint"] for row in result_rows}
    if len(memory_overlays) != len(candidates) or len(workspace_overlays) != len(candidates):
        raise RuntimeError("Candidate overlays were not isolated.")
    winner = max(result_rows, key=lambda item: (item["score"], item["candidate_id"]))
    for row in result_rows:
        disposition = (
            "retained_for_review"
            if row["candidate_id"] == winner["candidate_id"]
            else "quarantined"
        )
        for overlay_kind in ("memory", "workspace"):
            if not journal.record_once(
                "overlay-disposition",
                row[f"{overlay_kind}_overlay_fingerprint"],
                {
                    "candidate_id": row["candidate_id"],
                    "disposition": disposition,
                },
            ):
                raise RuntimeError("Overlay disposition would be recorded twice.")
        row["overlay_disposition"] = disposition
    phase_state = {
        "candidates": result_rows,
        "snapshot_fingerprint": snapshot.fingerprint,
    }
    if not journal.record_once("phase-state", "reference-experiment", phase_state):
        raise RuntimeError("Reference experiment database must start empty.")

    child_environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    inherited_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = (
        source_root if not inherited_pythonpath else source_root + os.pathsep + inherited_pythonpath
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).resolve()),
        "--database",
        str(database_path),
        "--recover-only",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=child_environment,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            "Fresh-process recovery failed: " + stderr.decode(errors="replace").strip()
        )
    recovery = json.loads(stdout)
    return {
        "schema_version": 1,
        "snapshot_fingerprint": snapshot.fingerprint,
        "snapshot_consistency": snapshot.consistency.value,
        "baseline_unchanged": True,
        "candidate_results": result_rows,
        "cross_candidate_contamination": False,
        "hidden_evaluator_truth_exported": False,
        "recovery": recovery,
        "recommendations": [
            {
                "candidate_id": winner["candidate_id"],
                "result_fingerprint": winner["result_fingerprint"],
            }
        ],
        "production_activation": "not_requested",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path)
    parser.add_argument("--recover-only", action="store_true")
    return parser.parse_args()


def _print_result(result: dict[str, Any]) -> None:
    print("MACHINE_RESULT=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
    recommendation = result["recommendations"][0]
    print(
        "AgentSnapshot reference experiment: "
        f"{len(result['candidate_results'])} isolated candidates, baseline unchanged."
    )
    print(
        "Fresh-process recovery: no runtime or evaluator effects replayed; "
        f"recommend {recommendation['candidate_id']} for review only."
    )


def main() -> None:
    args = _parse_args()
    if args.recover_only:
        if args.database is None:
            raise SystemExit("--recover-only requires --database")
        print(
            json.dumps(
                asyncio.run(recover_reference_experiment(args.database)),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return
    if args.database is not None:
        _print_result(asyncio.run(run_reference_experiment(args.database)))
        return
    with tempfile.TemporaryDirectory(prefix="cayu-agent-snapshot-") as directory:
        _print_result(
            asyncio.run(run_reference_experiment(Path(directory) / "reference-experiment.db"))
        )


if __name__ == "__main__":
    main()
