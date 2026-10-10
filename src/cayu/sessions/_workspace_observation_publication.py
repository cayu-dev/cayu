"""Exact runtime-publication validation for durable workspace observations."""

from __future__ import annotations

from typing import Any, cast

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import EventType
from cayu.sessions._checkpoint_mutations import runtime_publication_checkpoint_value_digest
from cayu.sessions._runtime_publication import RuntimePublicationRequest
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)


def _validate_workspace_observation_publication(
    request: RuntimePublicationRequest,
    *,
    session_id: str,
) -> None:
    """Validate the closed exact-publication contract for workspace evidence."""

    if request.kind != "workspace-observation":
        return
    from cayu.workspaces.observation_recovery import (
        WORKSPACE_OBSERVATION_TERMINAL_CONTROLS,
        WorkspaceObservationArtifact,
        WorkspaceObservationArtifactState,
        WorkspaceObservationPhase,
        WorkspaceObservationTerminalStatus,
        validate_workspace_observation_transition,
        workspace_observation_event_digest,
        workspace_observations_from_checkpoint,
    )

    if request.transcript_messages or request.referenced_events:
        raise ValueError(
            "A workspace-observation publication cannot append transcript or referenced events."
        )
    intent = request.intent
    required = {
        "schema_version",
        "window_id",
        "phase",
        "source_run_epoch",
        "binding_generation_id",
        "workspace_id",
        "observer",
        "observer_authority",
        "artifact_store_id",
        "agent_name",
        "environment_name",
        "tool_name",
        "tool_call_id",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "model_step",
        "before_observation_id",
        "tool_outcome_event_id",
        "tool_outcome_event_digest",
        "after_observation_id",
        "mutation_event_id",
        "mutation_event_digest",
        "artifacts",
        "terminal_status",
        "terminal_detail_code",
        "source_observations",
        "source_root_digest",
        "target_root_digest",
    }
    if set(intent) != required or intent.get("schema_version") != 1:
        raise ValueError("Workspace-observation publication intent is malformed.")
    for field_name in (
        "window_id",
        "phase",
        "binding_generation_id",
        "workspace_id",
        "observer",
        "agent_name",
        "tool_name",
        "tool_call_id",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
    ):
        require_clean_nonblank(
            cast("str", intent.get(field_name)),
            f"intent.{field_name}",
        )
    artifact_store_id = intent.get("artifact_store_id")
    if artifact_store_id is not None:
        require_clean_nonblank(artifact_store_id, "intent.artifact_store_id")
    environment_name = intent.get("environment_name")
    if environment_name is not None:
        require_clean_nonblank(environment_name, "intent.environment_name")
    if intent.get("observer_authority") not in {"runtime_builtin", "configured"}:
        raise ValueError("Workspace-observation observer authority is invalid.")
    if type(intent.get("source_run_epoch")) is not int or intent["source_run_epoch"] < 0:
        raise ValueError("Workspace-observation source_run_epoch is invalid.")
    model_step = intent.get("model_step")
    if model_step is not None and (type(model_step) is not int or model_step < 1):
        raise ValueError("Workspace-observation model_step is invalid.")
    for field_name in (
        "before_observation_id",
        "tool_outcome_event_id",
        "after_observation_id",
        "mutation_event_id",
    ):
        value = intent.get(field_name)
        if value is not None:
            require_clean_nonblank(value, f"intent.{field_name}")
    for field_name in ("tool_outcome_event_digest", "mutation_event_digest"):
        value = intent.get(field_name)
        if value is not None and (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Workspace-observation {field_name} is invalid.")
    for field_name in ("source_root_digest", "target_root_digest"):
        value = intent.get(field_name)
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Workspace-observation {field_name} is invalid.")
    raw_artifacts = intent.get("artifacts")
    if type(raw_artifacts) is not list or len(raw_artifacts) > 3:
        raise ValueError("Workspace-observation artifact intent is malformed.")
    try:
        intent_artifacts = tuple(
            WorkspaceObservationArtifact.model_validate(artifact) for artifact in raw_artifacts
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Workspace-observation artifact intent is malformed.") from exc
    if len({artifact.evidence_kind for artifact in intent_artifacts}) != len(intent_artifacts):
        raise ValueError("Workspace-observation artifact intent is duplicated.")
    phase = intent["phase"]
    base_phases = {
        "intent",
        "before-capture",
        "tool-outcome",
        "before-evidence",
        "after-capture",
        "delta-publication",
        "terminal",
        "recovered-tool-outcome",
    }
    artifact_phases = {
        f"artifact-{evidence_kind}-{artifact_state}"
        for evidence_kind in ("revision-before", "revision-after", "revision-delta")
        for artifact_state in ("intent", "published", "failed")
    }
    if phase not in base_phases | artifact_phases:
        raise ValueError("Workspace-observation publication phase is invalid.")
    if request.publication_id != (f"workspace-observation:{intent['window_id']}:{phase}"):
        raise ValueError(
            "Workspace-observation publication id conflicts with its window and phase."
        )
    terminal_status = intent.get("terminal_status")
    terminal_detail = intent.get("terminal_detail_code")
    if (terminal_status, terminal_detail) not in (
        {(None, None)} | WORKSPACE_OBSERVATION_TERMINAL_CONTROLS
    ):
        raise ValueError("Workspace-observation terminal classification is invalid.")
    if (phase == "terminal") != (terminal_status is not None):
        raise ValueError(
            "Workspace-observation terminal status conflicts with its publication phase."
        )
    if phase != "terminal" and terminal_detail is not None:
        raise ValueError(
            "Workspace-observation terminal detail conflicts with its publication phase."
        )
    operation_keys = {operation.key for operation in request.mutation.operations}
    if operation_keys not in (
        {"workspace_observations"},
        {"workspace_observations", CHECKPOINT_SCHEMA_VERSION_KEY},
    ):
        raise ValueError(
            "Workspace-observation publication must mutate exactly its lifecycle root."
        )
    schema_operation = next(
        (
            operation
            for operation in request.mutation.operations
            if operation.key == CHECKPOINT_SCHEMA_VERSION_KEY
        ),
        None,
    )
    if schema_operation is not None:
        supported_schema_digests = {
            runtime_publication_checkpoint_value_digest(version)
            for version in range(1, CURRENT_CHECKPOINT_SCHEMA_VERSION + 1)
        }
        if (
            schema_operation.action != "set"
            or schema_operation.value != CURRENT_CHECKPOINT_SCHEMA_VERSION
            or schema_operation.expected_value_digest not in supported_schema_digests | {None}
        ):
            raise ValueError("Workspace-observation root checkpoint schema stamp is invalid.")
    workspace_operations = tuple(
        operation
        for operation in request.mutation.operations
        if operation.key == "workspace_observations"
    )
    if len(workspace_operations) != 1:
        raise ValueError("Workspace-observation publication must contain one lifecycle mutation.")
    workspace_operation = workspace_operations[0]
    raw_source_observations = intent.get("source_observations")
    if type(raw_source_observations) is not dict:
        raise ValueError("Workspace-observation source lifecycle root is malformed.")
    try:
        source_records = workspace_observations_from_checkpoint(
            {"workspace_observations": raw_source_observations}
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Workspace-observation source lifecycle root is malformed.") from exc
    source_value = raw_source_observations or None
    source_root_digest = runtime_publication_checkpoint_value_digest(source_value)
    if intent["source_root_digest"] != source_root_digest:
        raise ValueError("Workspace-observation source lifecycle digest is invalid.")
    expected_source_digest = None if source_value is None else source_root_digest
    if workspace_operation.expected_value_digest != expected_source_digest:
        raise ValueError("Workspace-observation mutation is not fenced to its source root.")
    if workspace_operation.action == "set":
        if type(workspace_operation.value) is not dict:
            raise ValueError("Workspace-observation lifecycle root must be an object.")
        try:
            records = workspace_observations_from_checkpoint(
                {"workspace_observations": workspace_operation.value}
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Workspace-observation lifecycle mutation is malformed.") from exc
        current_record = records.get(intent["window_id"])
        target_value = workspace_operation.value
    else:
        records = {}
        current_record = None
        target_value = None
    if intent["target_root_digest"] != runtime_publication_checkpoint_value_digest(target_value):
        raise ValueError("Workspace-observation target lifecycle digest is invalid.")
    previous_record = source_records.get(intent["window_id"])
    expected_records = dict(source_records)
    if current_record is None:
        expected_records.pop(intent["window_id"], None)
    else:
        expected_records[intent["window_id"]] = current_record
    if records != expected_records:
        raise ValueError("Workspace-observation publication changed a sibling lifecycle.")
    try:
        validate_workspace_observation_transition(
            previous=previous_record,
            current=current_record,
            phase=phase,
            terminal_status=(
                None
                if terminal_status is None
                else WorkspaceObservationTerminalStatus(terminal_status)
            ),
            terminal_artifacts=(intent_artifacts if phase == "terminal" else None),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Workspace-observation lifecycle transition is invalid.") from exc
    if phase == "terminal":
        if current_record is not None or previous_record is None:
            raise ValueError(
                "Workspace-observation terminal publication retained its active lifecycle."
            )
        expected_authority = (
            session_id,
            request.interaction_id,
            intent["window_id"],
            intent["source_run_epoch"],
            intent["binding_generation_id"],
            intent["workspace_id"],
            intent["observer"],
            intent["observer_authority"],
            intent["artifact_store_id"],
            intent["agent_name"],
            intent["environment_name"],
            intent["tool_name"],
            intent["tool_call_id"],
            intent["model_step_id"],
            intent["model_attempt_id"],
            intent["tool_round_id"],
            intent["model_step"],
        )
        if previous_record.authority_tuple() != expected_authority:
            raise ValueError(
                "Workspace-observation terminal lifecycle conflicts with its authority."
            )
        for field_name in (
            "before_observation_id",
            "tool_outcome_event_id",
            "tool_outcome_event_digest",
            "after_observation_id",
            "mutation_event_id",
            "mutation_event_digest",
        ):
            if getattr(previous_record, field_name) != intent[field_name]:
                raise ValueError(
                    "Workspace-observation terminal lifecycle conflicts with its "
                    "content-bound intent."
                )
    else:
        if workspace_operation.action != "set" or current_record is None:
            raise ValueError("Workspace-observation publication omitted its active lifecycle.")
        expected_authority = (
            session_id,
            request.interaction_id,
            intent["window_id"],
            intent["source_run_epoch"],
            intent["binding_generation_id"],
            intent["workspace_id"],
            intent["observer"],
            intent["observer_authority"],
            intent["artifact_store_id"],
            intent["agent_name"],
            intent["environment_name"],
            intent["tool_name"],
            intent["tool_call_id"],
            intent["model_step_id"],
            intent["model_attempt_id"],
            intent["tool_round_id"],
            intent["model_step"],
        )
        if current_record.authority_tuple() != expected_authority:
            raise ValueError(
                "Workspace-observation lifecycle conflicts with its publication authority."
            )
        for field_name in (
            "before_observation_id",
            "tool_outcome_event_id",
            "tool_outcome_event_digest",
            "after_observation_id",
            "mutation_event_id",
            "mutation_event_digest",
        ):
            if getattr(current_record, field_name) != intent[field_name]:
                raise ValueError(
                    "Workspace-observation lifecycle conflicts with its content-bound intent."
                )
        if current_record.artifacts != intent_artifacts:
            raise ValueError("Workspace-observation lifecycle conflicts with its artifact intent.")
        expected_record_phase = {
            "intent": WorkspaceObservationPhase.INTENT,
            "before-capture": WorkspaceObservationPhase.BEFORE_CAPTURED,
            "tool-outcome": WorkspaceObservationPhase.TOOL_OUTCOME_STAGED,
            "recovered-tool-outcome": WorkspaceObservationPhase.TOOL_OUTCOME_STAGED,
            "after-capture": WorkspaceObservationPhase.AFTER_CAPTURED,
            "delta-publication": WorkspaceObservationPhase.DELTA_PUBLISHED,
        }.get(phase)
        if expected_record_phase is not None and current_record.phase is not expected_record_phase:
            raise ValueError(
                "Workspace-observation lifecycle phase conflicts with its publication phase."
            )
        if phase.startswith("artifact-"):
            evidence_kind, artifact_state = phase.removeprefix("artifact-").rsplit("-", 1)
            matches = tuple(
                artifact
                for artifact in current_record.artifacts
                if artifact.evidence_kind == evidence_kind
            )
            if len(matches) != 1 or matches[0].state is not WorkspaceObservationArtifactState(
                artifact_state
            ):
                raise ValueError(
                    "Workspace-observation artifact lifecycle conflicts with its publication."
                )
    allowed_events = {
        EventType.WORKSPACE_REVISION_OBSERVED,
        EventType.WORKSPACE_MUTATION_RECORDED,
        EventType.WORKSPACE_OBSERVATION_FINALIZED,
    }
    expected_event_type = {
        "before-evidence": EventType.WORKSPACE_REVISION_OBSERVED,
        "after-capture": EventType.WORKSPACE_REVISION_OBSERVED,
        "delta-publication": EventType.WORKSPACE_MUTATION_RECORDED,
        "terminal": EventType.WORKSPACE_OBSERVATION_FINALIZED,
    }.get(phase)
    if expected_event_type is None and request.events:
        raise ValueError("Workspace-observation publication phase cannot append lifecycle events.")
    if expected_event_type is not None and (
        len(request.events) != 1 or request.events[0].type is not expected_event_type
    ):
        raise ValueError("Workspace-observation publication event conflicts with its phase.")
    expected_observation_phase = {
        "before-evidence": "before",
        "after-capture": "after",
    }.get(phase)
    for event in request.events:
        if event.session_id != session_id or event.type not in allowed_events:
            raise ValueError("Workspace-observation publication contains an invalid event.")
        if (
            event.interaction_id != request.interaction_id
            or event.agent_name != intent["agent_name"]
            or event.environment_name != intent["environment_name"]
            or event.tool_name != intent["tool_name"]
        ):
            raise ValueError("Workspace-observation event conflicts with its invocation authority.")
        if (
            expected_observation_phase is not None
            and event.payload.get("phase") != expected_observation_phase
        ):
            raise ValueError(
                "Workspace-observation revision event conflicts with its capture phase."
            )
        if event.payload.get("window_id") != intent["window_id"]:
            raise ValueError("Workspace-observation event conflicts with its window.")
        if (
            type(event.payload.get("session_run_epoch")) is not int
            or event.payload.get("session_run_epoch") != intent["source_run_epoch"]
        ):
            raise ValueError("Workspace-observation event conflicts with its source run epoch.")
        event_model_step = event.payload.get("model_step")
        if (
            event_model_step is not None and type(event_model_step) is not int
        ) or event_model_step != intent["model_step"]:
            raise ValueError("Workspace-observation event conflicts with its model step.")
        for field_name in (
            "tool_call_id",
            "model_step_id",
            "model_attempt_id",
            "tool_round_id",
            "binding_generation_id",
            "workspace_id",
            "observer",
        ):
            if event.payload.get(field_name) != intent[field_name]:
                raise ValueError(f"Workspace-observation event conflicts with intent.{field_name}.")
        if event.payload.get("artifact_store_id") != intent["artifact_store_id"]:
            raise ValueError("Workspace-observation event conflicts with intent.artifact_store_id.")
        expected_event_id = {
            "before-evidence": intent["before_observation_id"],
            "after-capture": intent["after_observation_id"],
            "delta-publication": intent["mutation_event_id"],
        }.get(phase)
        if expected_event_id is not None and event.id != expected_event_id:
            raise ValueError(
                "Workspace-observation event conflicts with its lifecycle event identity."
            )
        if phase == "delta-publication" and (
            workspace_observation_event_digest(event) != intent["mutation_event_digest"]
        ):
            raise ValueError(
                "Workspace-observation delta event conflicts with its lifecycle digest."
            )
        evidence_kind = {
            "before-evidence": "revision-before",
            "after-capture": "revision-after",
            "delta-publication": "revision-delta",
        }.get(phase)
        if evidence_kind is not None:
            referenced_artifacts = tuple(
                artifact
                for artifact in intent_artifacts
                if artifact.evidence_kind == evidence_kind
                and artifact.state is WorkspaceObservationArtifactState.REFERENCED
            )
            manifest_fields = (
                event.payload.get("manifest_artifact_id"),
                event.payload.get("manifest_artifact_sha256"),
                event.payload.get("manifest_artifact_size_bytes"),
            )
            if not referenced_artifacts:
                if any(value is not None for value in manifest_fields):
                    raise ValueError(
                        "Workspace-observation event contains an unbound artifact reference."
                    )
            elif len(referenced_artifacts) != 1 or manifest_fields != (
                referenced_artifacts[0].artifact_id,
                referenced_artifacts[0].sha256,
                referenced_artifacts[0].size_bytes,
            ):
                raise ValueError(
                    "Workspace-observation event artifact conflicts with its lifecycle."
                )
        linked_fields = (
            (
                "before_observation_id",
                "after_observation_id",
                "tool_outcome_event_id",
                "tool_outcome_event_digest",
            )
            if phase == "delta-publication"
            else (
                "before_observation_id",
                "after_observation_id",
                "tool_outcome_event_id",
                "tool_outcome_event_digest",
                "mutation_event_id",
                "mutation_event_digest",
            )
            if phase == "terminal"
            else ()
        )
        for field_name in linked_fields:
            if event.payload.get(field_name) != intent[field_name]:
                raise ValueError(f"Workspace-observation event conflicts with intent.{field_name}.")
        if phase == "terminal" and (
            event.payload.get("status") != terminal_status
            or event.payload.get("detail_code") != terminal_detail
        ):
            raise ValueError(
                "Workspace-observation terminal event conflicts with its final status."
            )
        if phase == "terminal":
            expected_artifact_fields: dict[str, Any] = {}
            for artifact in intent_artifacts:
                prefix = artifact.evidence_kind.replace("-", "_")
                expected_artifact_fields.update(
                    {
                        f"{prefix}_artifact_id": artifact.artifact_id,
                        f"{prefix}_artifact_sha256": artifact.sha256,
                        f"{prefix}_artifact_size_bytes": artifact.size_bytes,
                        f"{prefix}_artifact_state": artifact.state.value,
                    }
                )
            known_artifact_fields = {
                f"revision_{observation_phase}_artifact_{suffix}"
                for observation_phase in ("before", "after", "delta")
                for suffix in ("id", "sha256", "size_bytes", "state")
            }
            if any(
                field_name in event.payload and field_name not in expected_artifact_fields
                for field_name in known_artifact_fields
            ):
                raise ValueError(
                    "Workspace-observation terminal event contains an unbound artifact."
                )
            for field_name, expected_value in expected_artifact_fields.items():
                actual_value = event.payload.get(field_name)
                if (
                    field_name.endswith("_size_bytes") and type(actual_value) is not int
                ) or actual_value != expected_value:
                    raise ValueError(
                        "Workspace-observation terminal artifact conflicts with its intent."
                    )
            referenced_artifact_count = event.payload.get("referenced_artifact_count")
            failed_artifact_count = event.payload.get("failed_artifact_count")
            if (
                type(referenced_artifact_count) is not int
                or type(failed_artifact_count) is not int
                or referenced_artifact_count
                != sum(
                    artifact.state is WorkspaceObservationArtifactState.REFERENCED
                    for artifact in intent_artifacts
                )
                or failed_artifact_count
                != sum(
                    artifact.state is WorkspaceObservationArtifactState.FAILED
                    for artifact in intent_artifacts
                )
            ):
                raise ValueError(
                    "Workspace-observation terminal artifact counts conflict with its intent."
                )
