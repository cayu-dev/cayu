from __future__ import annotations

import asyncio
import copy
from typing import Any, cast

import pytest
from tests.core.checkpoint_schema_conformance import (
    assert_assistant_publication_checkpoint_conformance,
    assert_current_checkpoint_publication_upgrade_conformance,
    assert_future_checkpoint_rejection_conformance,
    assert_reserved_checkpoint_key_migration_conformance,
    assert_runtime_publication_rejects_invocation_authority_mutation,
    assert_versionless_noop_transform_stamps_conformance,
    assert_versionless_pending_continuation_fails_closed_conformance,
)

from cayu import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    SQLiteSessionStore,
)
from cayu.runtime import InMemorySessionStore
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _model_completion_publication as model_completion_publication
from cayu.runtime import _session_engine as session_engine
from cayu.runtime._invocation_lifecycle import (
    invocation_lifecycle_receipt_history_present,
)
from cayu.runtime._tool_round_recovery import pending_tool_round_from_checkpoint
from cayu.runtime.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    CheckpointMigration,
    CheckpointMigrationDefinitionError,
    CheckpointMigrator,
    decode_runtime_checkpoint,
    runtime_checkpoint_writer_view,
)
from cayu.runtime.context import _compaction_checkpoint
from cayu.runtime.user_input import pending_user_input_from_checkpoint
from cayu.runtime.workspace_observation_recovery import (
    WorkspaceObservationArtifact,
    WorkspaceObservationArtifactState,
    WorkspaceObservationEvidenceState,
    WorkspaceObservationLifecycle,
    WorkspaceObservationPhase,
    workspace_observations_from_checkpoint,
)

_FROZEN_VERSIONLESS_ROOT_CHECKPOINTS = {
    "approval": {
        "pending_tool_approval": {
            "approval_id": "approval-frozen",
            "tool_round_id": f"tround_{'3' * 32}",
            "model_step_id": f"mstep_{'1' * 32}",
            "model_attempt_id": f"matt_{'2' * 32}",
            "tool_call_id": "call-frozen",
            "tool_name": "charge",
            "arguments": {},
            "agent_name": "assistant",
            "publish_arguments": True,
            "tool_calls": [
                {
                    "tool_call_id": "call-frozen",
                    "tool_name": "charge",
                    "arguments": {},
                    "policy_evidence": "authoritative",
                    "policy_decision": "require_approval",
                }
            ],
        },
        "unknown_additive": {"preserved": True},
    },
    "user-input": {
        "pending_user_input": {
            "input_id": "input-frozen",
            "tool_round_id": f"tround_{'3' * 32}",
            "model_step_id": f"mstep_{'1' * 32}",
            "model_attempt_id": f"matt_{'2' * 32}",
            "tool_call_id": "call-frozen",
            "tool_name": "ask_user",
            "question": "Continue?",
            "options": ["yes", "no"],
            "arguments": {},
            "agent_name": "assistant",
            "tool_calls": [
                {
                    "tool_call_id": "call-frozen",
                    "tool_name": "ask_user",
                    "arguments": {},
                    "policy_evidence": "unplanned",
                }
            ],
        },
        "unknown_additive": {"preserved": True},
    },
    "pending-tool-round": {
        "pending_tool_round": {
            "tool_round_id": f"tround_{'3' * 32}",
            "model_step_id": f"mstep_{'1' * 32}",
            "model_attempt_id": f"matt_{'2' * 32}",
            "agent_name": "assistant",
            "tool_calls": [
                {
                    "tool_call_id": "call-frozen",
                    "tool_name": "charge",
                    "arguments": {},
                    "policy_evidence": "unplanned",
                }
            ],
        },
        "unknown_additive": {"preserved": True},
    },
    "session-operation": {
        "session_operations": {
            "version": 1,
            "active_operation_id": None,
            "records": {
                "operation-frozen": {
                    "status": "completed",
                    "request_digest": "frozen-request-digest",
                    "event_ids": ["event-frozen"],
                }
            },
        },
        "unknown_additive": {"preserved": True},
    },
    "compaction": {
        "context_compaction": {
            "version": 2,
            "summary": "Frozen compacted context.",
            "compacted_transcript_cursor": 2,
            "metadata": {"compactor": "frozen", "mode": "deterministic"},
        },
        "unknown_additive": {"preserved": True},
    },
    "empty": {},
}


@pytest.mark.parametrize(
    ("fixture_name", "fixture"),
    _FROZEN_VERSIONLESS_ROOT_CHECKPOINTS.items(),
    ids=_FROZEN_VERSIONLESS_ROOT_CHECKPOINTS,
)
def test_frozen_versionless_root_payloads_decode_without_data_loss_and_remain_consumable(
    fixture_name: str,
    fixture: dict[str, object],
) -> None:
    decoded = decode_runtime_checkpoint(fixture, session_id="sess-frozen-shape")

    assert decoded is not None
    assert decoded[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
    without_version = dict(decoded)
    without_version.pop(CHECKPOINT_SCHEMA_VERSION_KEY)
    expected = copy.deepcopy(fixture)
    if fixture_name in {"pending-tool-round", "user-input"}:
        checkpoint_key = (
            "pending_tool_round" if fixture_name == "pending-tool-round" else "pending_user_input"
        )
        pending_state = cast("dict[str, object]", expected[checkpoint_key])
        pending_state.update(
            assistant_message_state="published",
            quarantined_assistant_message=None,
        )
    assert without_version == expected
    if fixture_name == "approval":
        assert approval_support.pending_approval_from_checkpoint(decoded) is not None
    elif fixture_name == "user-input":
        assert pending_user_input_from_checkpoint(decoded) is not None
    elif fixture_name == "pending-tool-round":
        assert pending_tool_round_from_checkpoint(decoded) is not None
    elif fixture_name == "session-operation":
        assert session_engine._session_operation_state(decoded)["version"] == 1
    elif fixture_name == "compaction":
        assert _compaction_checkpoint(decoded) is not None


def test_versionless_root_checkpoint_is_migrated_to_current_version() -> None:
    source = {
        "pending_user_input": {"input_id": "input-1"},
        "future_additive_field": {"kept": True},
    }

    decoded = decode_runtime_checkpoint(source, session_id="sess-versionless")

    assert decoded == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "pending_user_input": {
            "input_id": "input-1",
            "assistant_message_state": "published",
            "quarantined_assistant_message": None,
        },
        "future_additive_field": {"kept": True},
    }
    assert source == {
        "pending_user_input": {"input_id": "input-1"},
        "future_additive_field": {"kept": True},
    }


def test_versionless_root_checkpoint_has_a_fixed_legacy_version() -> None:
    migrator = CheckpointMigrator(
        current_version=2,
        min_supported_version=2,
    )

    with pytest.raises(CheckpointCompatibilityError) as caught:
        migrator.decode({}, session_id="sess-versionless-too-old")

    assert caught.value.reason == "checkpoint_schema_version_too_old"
    assert caught.value.observed_version == 1


def test_v1_model_publication_pointer_is_upcast_to_v2() -> None:
    source = {
        CHECKPOINT_SCHEMA_VERSION_KEY: 1,
        model_completion_publication.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY: {
            "record_type": "cayu.model-step-publication",
            "schema_version": 1,
            "logical_step_id": "logical-step",
            "stage_id": "stage-1",
            "source_transcript_cursor": 0,
            "transcript_end_cursor": 1,
            "completion_event_id": "completion-event",
            "classification": {"type": "final"},
            "assistant_message_published": True,
            "tool_round_id": None,
        },
    }

    decoded = decode_runtime_checkpoint(source, session_id="sess-v1-publication")

    assert decoded is not None
    pointer = model_completion_publication.ModelStepPublicationCheckpoint.model_validate(
        decoded[model_completion_publication.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY]
    )
    assert pointer.schema_version == 2
    assert pointer.assistant_message_deferred is False
    original_pointer = cast(
        "dict[str, object]",
        source[model_completion_publication.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY],
    )
    assert original_pointer["schema_version"] == 1


def test_v2_root_checkpoint_is_upcast_to_v3_without_inventing_invocation_authority() -> None:
    source = {
        CHECKPOINT_SCHEMA_VERSION_KEY: 2,
        "future_additive_field": {"kept": True},
    }

    decoded = decode_runtime_checkpoint(source, session_id="sess-v2-root")

    assert decoded is not None
    assert decoded == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "future_additive_field": {"kept": True},
    }
    assert ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY not in decoded
    assert source[CHECKPOINT_SCHEMA_VERSION_KEY] == 2


def test_v2_root_checkpoint_discards_reserved_invocation_authority_collision() -> None:
    source = {
        CHECKPOINT_SCHEMA_VERSION_KEY: 2,
        ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY: {
            "record_type": "cayu.active-invocation-execution-profile",
            "session_id": "forged-session",
        },
        "future_additive_field": {"kept": True},
    }

    decoded = decode_runtime_checkpoint(source, session_id="sess-v2-authority-collision")

    assert decoded == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "future_additive_field": {"kept": True},
    }
    assert ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY in source
    assert source[CHECKPOINT_SCHEMA_VERSION_KEY] == 2


def test_v3_root_checkpoint_discards_reserved_workspace_observation_collision() -> None:
    source = {
        CHECKPOINT_SCHEMA_VERSION_KEY: 3,
        "workspace_observations": {
            "caller-controlled-window": {
                "record_type": "cayu.workspace-observation",
                "schema_version": 1,
            }
        },
        "future_additive_field": {"kept": True},
    }

    decoded = decode_runtime_checkpoint(source, session_id="sess-v3-workspace-collision")

    assert decoded is not None
    assert invocation_lifecycle_receipt_history_present(decoded)
    decoded_without_tombstone = dict(decoded)
    decoded_without_tombstone.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY)
    assert decoded_without_tombstone == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "future_additive_field": {"kept": True},
    }
    assert "workspace_observations" in source
    assert source[CHECKPOINT_SCHEMA_VERSION_KEY] == 3


def test_v4_root_checkpoint_discards_invocation_lifecycle_receipt_collision() -> None:
    source = {
        CHECKPOINT_SCHEMA_VERSION_KEY: 4,
        "invocation_lifecycle_receipt": {
            "record_type": "cayu.invocation-lifecycle-command-receipt-ledger",
            "schema_version": 1,
            "receipts": [
                {
                    "record_type": "cayu.invocation-lifecycle-command-receipt",
                    "schema_version": 1,
                    "command_identity": "caller-owned-collision",
                }
            ],
        },
        "future_additive_field": {"kept": True},
    }

    decoded = decode_runtime_checkpoint(source, session_id="sess-v4-receipt-collision")

    assert decoded is not None
    assert invocation_lifecycle_receipt_history_present(decoded)
    decoded_without_tombstone = dict(decoded)
    decoded_without_tombstone.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY)
    assert decoded_without_tombstone == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "future_additive_field": {"kept": True},
    }
    assert "invocation_lifecycle_receipt" in source
    assert source[CHECKPOINT_SCHEMA_VERSION_KEY] == 4


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("source_run_epoch", True),
        ("source_run_epoch", 1.0),
        ("model_step", True),
        ("model_step", 1.0),
    ],
)
def test_workspace_observation_authority_rejects_non_exact_integer_fields(
    field_name: str,
    invalid_value: object,
) -> None:
    record = {
        "record_type": "cayu.workspace-observation",
        "schema_version": 1,
        "session_id": "sess-workspace-exact-integers",
        "interaction_id": None,
        "window_id": "wmut-workspace-exact-integers",
        "source_run_epoch": 1,
        "binding_generation_id": "wbind-workspace-exact-integers",
        "workspace_id": "workspace-exact-integers",
        "observer": "ExactIntegerBinding",
        "observer_authority": "configured",
        "artifact_store_id": None,
        "agent_name": "assistant",
        "environment_name": "local",
        "tool_name": "mutate",
        "tool_call_id": "call-workspace-exact-integers",
        "model_step_id": "mstep-workspace-exact-integers",
        "model_attempt_id": "matt-workspace-exact-integers",
        "tool_round_id": "tround-workspace-exact-integers",
        "model_step": 1,
        "phase": "intent",
        "before_state": "pending",
        "before_observation_id": None,
        "tool_outcome_event_id": None,
        "tool_outcome_event_digest": None,
        "after_state": "pending",
        "after_observation_id": None,
        "delta_state": "pending",
        "mutation_event_id": None,
        "mutation_event_digest": None,
        "artifacts": [],
    }
    record[field_name] = invalid_value

    with pytest.raises(ValueError):
        workspace_observations_from_checkpoint(
            {"workspace_observations": {record["window_id"]: record}}
        )


@pytest.mark.parametrize("invalid_authority", [None, True, "future"])
def test_workspace_observation_checkpoint_rejects_invalid_observer_authority(
    invalid_authority: object,
) -> None:
    lifecycle = WorkspaceObservationLifecycle(
        session_id="sess-invalid-observer-authority",
        window_id="wmut-invalid-observer-authority",
        source_run_epoch=1,
        binding_generation_id="wbind-invalid-observer-authority",
        workspace_id="workspace-invalid-observer-authority",
        observer="ConfiguredWorkspaceBinding",
        observer_authority="configured",
        agent_name="assistant",
        tool_name="mutate",
        tool_call_id="call-invalid-observer-authority",
        model_step_id="mstep-invalid-observer-authority",
        model_attempt_id="matt-invalid-observer-authority",
        tool_round_id="tround-invalid-observer-authority",
    )
    record: dict[str, Any] = lifecycle.model_dump(mode="json")
    record["observer_authority"] = invalid_authority

    with pytest.raises(ValueError):
        workspace_observations_from_checkpoint(
            {"workspace_observations": {lifecycle.window_id: record}}
        )


def test_workspace_observation_checkpoint_requires_observer_authority() -> None:
    lifecycle = WorkspaceObservationLifecycle(
        session_id="sess-missing-observer-authority",
        window_id="wmut-missing-observer-authority",
        source_run_epoch=1,
        binding_generation_id="wbind-missing-observer-authority",
        workspace_id="workspace-missing-observer-authority",
        observer="ConfiguredWorkspaceBinding",
        observer_authority="configured",
        agent_name="assistant",
        tool_name="mutate",
        tool_call_id="call-missing-observer-authority",
        model_step_id="mstep-missing-observer-authority",
        model_attempt_id="matt-missing-observer-authority",
        tool_round_id="tround-missing-observer-authority",
    )
    record = lifecycle.model_dump(mode="json")
    del record["observer_authority"]

    with pytest.raises(ValueError):
        workspace_observations_from_checkpoint(
            {"workspace_observations": {lifecycle.window_id: record}}
        )


@pytest.mark.parametrize("invalid_value", [True, 1.0])
def test_workspace_observation_artifact_size_requires_exact_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError):
        WorkspaceObservationArtifact(
            evidence_kind="revision-before",
            artifact_id="artifact-workspace-exact-integers",
            sha256="a" * 64,
            size_bytes=invalid_value,
            state="intent",
        )


def test_workspace_observation_checkpoint_revalidates_forged_nested_artifact() -> None:
    forged = WorkspaceObservationArtifact.model_construct(
        evidence_kind="revision-before",
        artifact_id="artifact-forged-workspace-observation",
        sha256="a" * 64,
        size_bytes=True,
        state="intent",
    )
    record = WorkspaceObservationLifecycle(
        session_id="sess-forged-workspace-artifact",
        window_id="wmut-forged-workspace-artifact",
        source_run_epoch=1,
        binding_generation_id="wbind-forged-workspace-artifact",
        workspace_id="workspace-forged-workspace-artifact",
        observer="ForgedArtifactBinding",
        observer_authority="configured",
        artifact_store_id="artifact-forged-workspace-artifact",
        agent_name="assistant",
        environment_name="local",
        tool_name="mutate",
        tool_call_id="call-forged-workspace-artifact",
        model_step_id="mstep-forged-workspace-artifact",
        model_attempt_id="matt-forged-workspace-artifact",
        tool_round_id="tround-forged-workspace-artifact",
        model_step=1,
        phase=WorkspaceObservationPhase.TOOL_OUTCOME_STAGED,
        before_state=WorkspaceObservationEvidenceState.CAPTURED_PRIVATE,
        tool_outcome_event_id="evt-forged-workspace-artifact",
        tool_outcome_event_digest="b" * 64,
    ).model_dump(mode="json")
    record["artifacts"] = [forged]

    with pytest.raises(ValueError):
        workspace_observations_from_checkpoint(
            {"workspace_observations": {record["window_id"]: record}}
        )


@pytest.mark.parametrize("terminal_state", ["orphaned", "missing"])
def test_active_workspace_observation_rejects_terminal_only_artifact_state(
    terminal_state: str,
) -> None:
    lifecycle = WorkspaceObservationLifecycle(
        session_id="sess-active-artifact-state",
        window_id="wmut-active-artifact-state",
        source_run_epoch=1,
        binding_generation_id="wbind-active-artifact-state",
        workspace_id="workspace-active-artifact-state",
        observer="ActiveArtifactStateBinding",
        observer_authority="configured",
        artifact_store_id="artifact-active-artifact-state",
        agent_name="assistant",
        environment_name="local",
        tool_name="mutate",
        tool_call_id="call-active-artifact-state",
        model_step_id="mstep-active-artifact-state",
        model_attempt_id="matt-active-artifact-state",
        tool_round_id="tround-active-artifact-state",
        model_step=1,
        phase=WorkspaceObservationPhase.TOOL_OUTCOME_STAGED,
        before_state=WorkspaceObservationEvidenceState.CAPTURED_PRIVATE,
        tool_outcome_event_id="evt-active-artifact-state",
        tool_outcome_event_digest="a" * 64,
        artifacts=(
            WorkspaceObservationArtifact(
                evidence_kind="revision-before",
                artifact_id="artifact-active-artifact-state",
                sha256="b" * 64,
                size_bytes=1,
                state=WorkspaceObservationArtifactState(terminal_state),
            ),
        ),
    )

    with pytest.raises(ValueError, match="terminal-only artifact state"):
        workspace_observations_from_checkpoint(
            {"workspace_observations": {lifecycle.window_id: lifecycle.model_dump(mode="json")}}
        )


@pytest.mark.parametrize("writer_version", [1, 2, 3])
def test_older_writer_view_rejects_active_workspace_observation_authority(
    writer_version: int,
) -> None:
    source = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "workspace_observations": {
            "wmut-writer-view": {
                "record_type": "cayu.workspace-observation",
            }
        },
    }

    with pytest.raises(ValueError, match="cannot be represented"):
        runtime_checkpoint_writer_view(
            source,
            writer_version=writer_version,
            session_id="sess-active-workspace-observation-writer-view",
        )


@pytest.mark.parametrize("writer_version", [1, 2])
def test_older_writer_view_rejects_active_invocation_profile_authority(
    writer_version: int,
) -> None:
    source = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY: {
            "record_type": "cayu.active-invocation-execution-profile",
        },
    }

    with pytest.raises(ValueError, match="cannot be represented"):
        runtime_checkpoint_writer_view(
            source,
            writer_version=writer_version,
            session_id="sess-active-profile-writer-view",
        )


def test_v2_writer_view_projects_v3_state_without_active_invocation_authority() -> None:
    source = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "future_additive_field": {"kept": True},
    }

    projected = runtime_checkpoint_writer_view(
        source,
        writer_version=2,
        session_id="sess-v2-writer-view",
    )

    assert projected == {
        CHECKPOINT_SCHEMA_VERSION_KEY: 2,
        "future_additive_field": {"kept": True},
    }
    assert source[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION


def test_v1_writer_view_rejects_unrecognized_current_model_publication_pointer() -> None:
    source = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        model_completion_publication.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY: {
            "record_type": "cayu.model-step-publication",
            "schema_version": CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
            "logical_step_id": "logical-step",
            "stage_id": "stage-1",
            "source_transcript_cursor": 0,
            "transcript_end_cursor": 1,
            "completion_event_id": "completion-event",
            "classification": {"type": "final"},
            "assistant_message_published": True,
            "assistant_message_deferred": False,
            "tool_round_id": None,
        },
    }

    with pytest.raises(ValueError, match="not a recognized v2 pointer"):
        runtime_checkpoint_writer_view(
            source,
            writer_version=1,
            session_id="sess-future-nested-publication",
        )


def test_current_root_checkpoint_is_defensively_copied() -> None:
    source = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "nested": {"value": 1},
    }

    decoded = decode_runtime_checkpoint(source, session_id="sess-current")

    assert decoded == source
    assert decoded is not source
    assert decoded is not None
    assert decoded["nested"] is not source["nested"]


def test_in_memory_runtime_publication_rejects_invocation_authority_mutation() -> None:
    asyncio.run(
        assert_runtime_publication_rejects_invocation_authority_mutation(
            InMemorySessionStore(),
            session_id_prefix="sess-memory-reserved-invocation-publication",
        )
    )


def test_non_object_root_checkpoint_fails_with_typed_evidence() -> None:
    with pytest.raises(CheckpointCompatibilityError) as caught:
        decode_runtime_checkpoint(
            cast("Any", ["private", "checkpoint"]),
            session_id="sess-invalid-root",
        )

    assert caught.value.reason == "invalid_root_checkpoint"
    assert caught.value.observed_version is None
    assert "private" not in str(caught.value)


@pytest.mark.parametrize(
    "invalid_version",
    [True, "1", 1.0, None, -1, 0],
)
def test_malformed_root_checkpoint_version_fails_with_bounded_typed_evidence(
    invalid_version: object,
) -> None:
    source = {
        CHECKPOINT_SCHEMA_VERSION_KEY: invalid_version,
        "secret_checkpoint_content": "must-not-appear",
    }

    with pytest.raises(CheckpointCompatibilityError) as caught:
        decode_runtime_checkpoint(source, session_id="sess-invalid-version")

    error = caught.value
    assert error.reason == "invalid_checkpoint_schema_version"
    assert error.checkpoint_kind == "root"
    assert error.observed_version is None
    assert error.supported_min_version == 1
    assert error.supported_max_version == CURRENT_CHECKPOINT_SCHEMA_VERSION
    assert error.session_id == "sess-invalid-version"
    assert error.recovery_disposition == "cannot_migrate"
    assert error.resumable_in_place is False
    assert "must-not-appear" not in str(error)
    assert repr(invalid_version) not in str(error)


def test_future_root_checkpoint_version_fails_with_bounded_typed_evidence() -> None:
    source = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
        "secret_checkpoint_content": "must-not-appear",
    }

    with pytest.raises(CheckpointCompatibilityError) as caught:
        decode_runtime_checkpoint(source, session_id="sess-future-version")

    error = caught.value
    assert error.reason == "checkpoint_schema_version_too_new"
    assert error.observed_version == CURRENT_CHECKPOINT_SCHEMA_VERSION + 1
    assert error.safe_evidence() == {
        "checkpoint_kind": "root",
        "observed_version": CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
        "reason": "checkpoint_schema_version_too_new",
        "recovery_disposition": "cannot_migrate",
        "resumable_in_place": False,
        "session_id": "sess-future-version",
        "supported_max_version": CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "supported_min_version": 1,
    }
    assert "must-not-appear" not in str(error)


def test_compatibility_evidence_hashes_an_oversized_session_identity() -> None:
    oversized_session_id = "session-" + ("x" * 10_000)

    with pytest.raises(CheckpointCompatibilityError) as caught:
        decode_runtime_checkpoint(
            {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION + 1},
            session_id=oversized_session_id,
        )

    assert caught.value.session_id.startswith("sha256:")
    assert len(caught.value.session_id) == len("sha256:") + 64
    assert oversized_session_id not in str(caught.value)
    assert oversized_session_id not in str(caught.value.safe_evidence())


def test_ordered_checkpoint_upcaster_preserves_unknown_additive_fields() -> None:
    def migrate_v1_to_v2(checkpoint: dict[str, object]) -> dict[str, object]:
        updated = dict(checkpoint)
        updated["renamed_field"] = updated.pop("owned_v1_field")
        updated[CHECKPOINT_SCHEMA_VERSION_KEY] = 2
        return updated

    migrator = CheckpointMigrator(
        current_version=2,
        migrations=(
            CheckpointMigration(
                source_version=1,
                target_version=2,
                migrate=migrate_v1_to_v2,
            ),
        ),
    )
    source = {
        CHECKPOINT_SCHEMA_VERSION_KEY: 1,
        "owned_v1_field": "value",
        "future_additive_field": {"kept": True},
    }

    decoded = migrator.decode(source, session_id="sess-upcast")

    assert decoded == {
        CHECKPOINT_SCHEMA_VERSION_KEY: 2,
        "renamed_field": "value",
        "future_additive_field": {"kept": True},
    }
    assert source == {
        CHECKPOINT_SCHEMA_VERSION_KEY: 1,
        "owned_v1_field": "value",
        "future_additive_field": {"kept": True},
    }


def test_checkpoint_upcaster_chain_rejects_missing_intermediate_step() -> None:
    with pytest.raises(
        CheckpointMigrationDefinitionError,
        match="missing migration from version 2",
    ):
        CheckpointMigrator(
            current_version=3,
            migrations=(
                CheckpointMigration(
                    source_version=1,
                    target_version=2,
                    migrate=lambda checkpoint: checkpoint,
                ),
            ),
        )


def test_checkpoint_upcaster_chain_rejects_duplicate_source_version() -> None:
    migration = CheckpointMigration(
        source_version=1,
        target_version=2,
        migrate=lambda checkpoint: checkpoint,
    )

    with pytest.raises(
        CheckpointMigrationDefinitionError,
        match="duplicate migration from version 1",
    ):
        CheckpointMigrator(
            current_version=2,
            migrations=(migration, migration),
        )


def test_checkpoint_upcaster_chain_rejects_non_unit_jump() -> None:
    with pytest.raises(
        CheckpointMigrationDefinitionError,
        match="advance exactly one version",
    ):
        CheckpointMigration(
            source_version=1,
            target_version=3,
            migrate=lambda checkpoint: checkpoint,
        )


def test_checkpoint_upcaster_verifies_reported_target_version() -> None:
    migrator = CheckpointMigrator(
        current_version=2,
        migrations=(
            CheckpointMigration(
                source_version=1,
                target_version=2,
                migrate=lambda checkpoint: checkpoint,
            ),
        ),
    )

    with pytest.raises(
        CheckpointMigrationDefinitionError,
        match="must return checkpoint schema version 2",
    ):
        migrator.decode(
            {CHECKPOINT_SCHEMA_VERSION_KEY: 1},
            session_id="sess-wrong-target",
        )


def test_in_memory_checkpoint_schema_runtime_conformance() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        await assert_versionless_pending_continuation_fails_closed_conformance(
            store,
            session_id="sess-memory-versionless-checkpoint",
        )
        await assert_versionless_noop_transform_stamps_conformance(
            store,
            session_id="sess-memory-versionless-noop-transform",
        )
        await assert_future_checkpoint_rejection_conformance(
            store,
            session_id="sess-memory-future-checkpoint",
        )
        await assert_reserved_checkpoint_key_migration_conformance(
            store,
            session_id="sess-memory-reserved-key-migration",
        )
        await assert_current_checkpoint_publication_upgrade_conformance(
            store,
            session_id_prefix="sess-memory-current-publication",
        )
        await assert_runtime_publication_rejects_invocation_authority_mutation(
            store,
            session_id_prefix="sess-memory-invocation-authority-publication",
        )
        await assert_assistant_publication_checkpoint_conformance(
            store,
            session_id="sess-memory-assistant-publication",
        )

    asyncio.run(run())


def test_sqlite_checkpoint_schema_runtime_conformance(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(tmp_path / "checkpoint-schema.sqlite")
        try:
            await assert_versionless_pending_continuation_fails_closed_conformance(
                store,
                session_id="sess-sqlite-versionless-checkpoint",
            )
            await assert_versionless_noop_transform_stamps_conformance(
                store,
                session_id="sess-sqlite-versionless-noop-transform",
            )
            await assert_future_checkpoint_rejection_conformance(
                store,
                session_id="sess-sqlite-future-checkpoint",
            )
            await assert_reserved_checkpoint_key_migration_conformance(
                store,
                session_id="sess-sqlite-reserved-key-migration",
            )
            await assert_current_checkpoint_publication_upgrade_conformance(
                store,
                session_id_prefix="sess-sqlite-current-publication",
            )
            await assert_runtime_publication_rejects_invocation_authority_mutation(
                store,
                session_id_prefix="sess-sqlite-invocation-authority-publication",
            )
            await assert_assistant_publication_checkpoint_conformance(
                store,
                session_id="sess-sqlite-assistant-publication",
            )
        finally:
            await store.close()

    asyncio.run(run())
