"""Record characterization; store/public acceptance lives at the dispatch boundary."""

from __future__ import annotations

import asyncio
import warnings

import pytest

from cayu.sessions._model_failover import (
    MODEL_FAILOVER_CHECKPOINT_KEY,
    ModelFailoverCandidate,
    ModelFailoverPlan,
    ModelFailoverProgress,
    ModelFailoverSelection,
    copy_model_failover_state,
    validate_model_failover_successor,
)
from cayu.sessions.base import (
    InMemorySessionStore,
    RunRequest,
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
    SessionIdentity,
    apply_runtime_publication_checkpoint_mutation,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    decode_runtime_checkpoint,
    runtime_checkpoint_writer_view,
)
from cayu.storage.sqlite import SQLiteSessionStore


def _progress(**changes) -> ModelFailoverProgress:
    values = {
        "session_id": "session",
        "session_instance_id": "instance",
        "interaction_id": "interaction",
        "execution_profile_fingerprint": "a" * 64,
        "plan": {
            "candidates": [
                {
                    "provider_name": "primary",
                    "model": "small",
                    "execution_profile_fingerprint": "b" * 64,
                    "execution_mode": "synchronous",
                },
                {
                    "provider_name": "backup",
                    "model": "large",
                    "execution_profile_fingerprint": "c" * 64,
                    "execution_mode": "synchronous",
                },
            ],
            "max_total_attempts": 5,
        },
        "generation": 1,
        "candidate_index": 0,
        "logical_step_id": "step",
        "stage_id": "stage-0",
        "request_fingerprint": "d" * 64,
        "source_run_epoch": 1,
        "source_transcript_cursor": 5,
        "projection_cursor": 0,
        "dispatch_ordinal": 0,
        "attempts_used": 1,
        "candidate_attempt": 1,
        "provider_effect_observed": False,
    }
    values.update(changes)
    return ModelFailoverProgress.model_validate(values)


def _selection(**changes) -> ModelFailoverSelection:
    progress = _progress()
    payload = {
        name: getattr(progress, name)
        for name in (
            "session_id",
            "session_instance_id",
            "execution_profile_fingerprint",
            "plan",
            "candidate_index",
            "source_run_epoch",
            "source_transcript_cursor",
            "projection_cursor",
        )
    }
    payload.update(origin_id="e" * 64, source_run_epoch=0, candidate_index=1)
    payload.update(changes)
    return ModelFailoverSelection.model_validate(payload)


def test_selected_origin_has_no_dispatch_identity_and_reconstructs_exactly():
    selected = _selection()
    assert copy_model_failover_state(selected) == selected
    assert copy_model_failover_state(selected.payload()) == selected
    assert ModelFailoverSelection.model_validate_json(selected.model_dump_json()) == selected
    assert selected.candidate_index == 1 and selected.source_run_epoch == 0
    assert not {"stage_id", "attempts_used", "generation", "logical_step_id"}.intersection(
        selected.payload()
    )
    assert copy_model_failover_state(_progress().payload()) == _progress()
    missing_tag = selected.payload()
    missing_tag.pop("state")
    with pytest.raises(ValueError):
        copy_model_failover_state(missing_tag)
    with pytest.raises(ValueError):
        copy_model_failover_state({**selected.payload(), "state": "future"})


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_index": True},
        {"candidate_index": 2},
        {"source_run_epoch": True},
        {"source_run_epoch": -1},
        {"projection_cursor": 6},
        {"schema_version": True},
        {"origin_id": "not-an-identity"},
        {"attempts_used": 1},
        {"stage_id": "forged"},
    ],
)
def test_selected_origin_rejects_ambiguous_or_dispatch_shaped_values(changes):
    with pytest.raises(ValueError):
        _selection(**changes)


def _successor(previous: ModelFailoverProgress, **changes) -> ModelFailoverProgress:
    values = previous.payload()
    values.update(
        generation=previous.generation + 1,
        stage_id="stage-1",
        dispatch_ordinal=previous.dispatch_ordinal + 1,
        attempts_used=previous.attempts_used + 1,
        candidate_attempt=previous.candidate_attempt + 1,
    )
    values.update(changes)
    return ModelFailoverProgress.model_validate(values)


def test_route_round_trip_detaches_and_revalidates_every_nested_value():
    progress = _progress()
    payload = progress.payload()
    restored = ModelFailoverProgress.model_validate_json(progress.model_dump_json())
    assert restored == progress
    assert restored.route_id == progress.route_id
    assert restored.plan.fingerprint == progress.plan.fingerprint
    payload["plan"]["candidates"][0]["model"] = "changed"
    assert progress.plan.candidates[0].model == "small"


def test_route_retry_fallback_and_sticky_next_step_progression():
    initial = _progress()
    retry = _successor(initial)
    validate_model_failover_successor(initial, retry, transition="retry")
    fallback = _successor(
        retry, stage_id="stage-2", candidate_index=1, candidate_attempt=1, projection_cursor=5
    )
    validate_model_failover_successor(retry, fallback, transition="fallback")
    next_step = _successor(
        fallback,
        logical_step_id="step-2",
        stage_id="stage-3",
        dispatch_ordinal=0,
        source_transcript_cursor=8,
        attempts_used=1,
        candidate_attempt=1,
    )
    validate_model_failover_successor(fallback, next_step, transition="next_step")
    assert initial.route_id == retry.route_id == fallback.route_id == next_step.route_id


@pytest.mark.parametrize("epoch_advance", [0, 1], ids=["queue", "resume"])
def test_new_interaction_can_only_inherit_selection_in_a_nonstale_next_step(epoch_advance):
    previous = _successor(
        _progress(source_run_epoch=2), candidate_index=1, candidate_attempt=1, projection_cursor=5
    )
    successor = _successor(
        previous,
        interaction_id="new-interaction",
        source_run_epoch=previous.source_run_epoch + epoch_advance,
        logical_step_id="new-step",
        stage_id="new-stage",
        dispatch_ordinal=0,
        attempts_used=1,
        candidate_attempt=1,
    )
    validate_model_failover_successor(previous, successor, transition="next_step")
    assert successor.candidate_index == previous.candidate_index == 1
    assert successor.route_id != previous.route_id
    for transition in ("retry", "fallback", "reprepare"):
        with pytest.raises(ValueError, match="authority changed"):
            validate_model_failover_successor(previous, successor, transition=transition)
    for field, value in (
        ("source_run_epoch", previous.source_run_epoch - 1),
        ("session_id", "different-session"),
        ("session_instance_id", "different-instance"),
        ("execution_profile_fingerprint", "f" * 64),
        ("plan", previous.plan.model_copy(update={"max_total_attempts": 4})),
    ):
        with pytest.raises(ValueError, match="authority changed"):
            validate_model_failover_successor(
                previous, successor.model_copy(update={field: value}), transition="next_step"
            )
    with pytest.raises(ValueError, match="conflicts"):
        validate_model_failover_successor(
            previous, successor.model_copy(update={"candidate_index": 0}), transition="next_step"
        )


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("session_id", "different"),
        ("session_instance_id", "different"),
        ("interaction_id", "different"),
        ("execution_profile_fingerprint", "e" * 64),
        ("generation", 3),
        ("stage_id", "stage-0"),
        ("logical_step_id", "different"),
        ("candidate_index", 1),
        ("dispatch_ordinal", 0),
        ("attempts_used", 3),
        ("candidate_attempt", 1),
        ("source_transcript_cursor", 6),
        ("projection_cursor", 1),
    ],
)
def test_retry_rejects_conflicting_transition_fields(field, changed):
    previous = _progress()
    successor = _successor(previous, **{field: changed})
    with pytest.raises(ValueError):
        validate_model_failover_successor(previous, successor, transition="retry")


@pytest.mark.parametrize("field", ["provider_name", "model", "execution_profile_fingerprint"])
def test_same_target_slot_cannot_change_candidate_authority(field):
    previous = _progress()
    plan = previous.plan.payload()
    plan["candidates"][1][field] = "e" * 64 if field.endswith("fingerprint") else "changed"
    successor = _successor(previous, plan=plan)
    assert successor.plan.fingerprint != previous.plan.fingerprint
    with pytest.raises(ValueError, match="authority"):
        validate_model_failover_successor(previous, successor, transition="retry")


def test_candidate_order_and_total_ceiling_are_route_authority():
    previous = _progress()
    for reverse in (False, True):
        plan = previous.plan.payload()
        if reverse:
            plan["candidates"].reverse()
        else:
            plan["max_total_attempts"] = 6
        with pytest.raises(ValueError, match="authority"):
            validate_model_failover_successor(
                previous, _successor(previous, plan=plan), transition="retry"
            )


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "generation",
        "candidate_index",
        "source_run_epoch",
        "source_transcript_cursor",
        "projection_cursor",
        "dispatch_ordinal",
        "attempts_used",
        "candidate_attempt",
    ],
)
def test_boolean_is_not_durable_route_version_or_counter(field):
    with pytest.raises(ValueError):
        _progress(**{field: True})


def test_fallback_cannot_overspend_attempt_cap_or_return_to_primary():
    previous = _progress(attempts_used=5, candidate_attempt=5)
    with pytest.raises(ValueError, match="attempts"):
        _successor(previous, candidate_index=1, candidate_attempt=1, projection_cursor=5)
    selected = _progress(candidate_index=1, generation=2, projection_cursor=5)
    with pytest.raises(ValueError):
        validate_model_failover_successor(
            selected, _successor(selected, candidate_index=0), transition="fallback"
        )


def test_retry_cannot_erase_prior_output_and_fallback_cannot_ignore_it():
    previous = _progress(provider_effect_observed=True)
    with pytest.raises(ValueError):
        validate_model_failover_successor(
            previous, _successor(previous, provider_effect_observed=False), transition="retry"
        )
    for observed in (False, True):
        with pytest.raises(ValueError):
            validate_model_failover_successor(
                previous,
                _successor(
                    previous,
                    candidate_index=1,
                    candidate_attempt=1,
                    projection_cursor=5,
                    provider_effect_observed=observed,
                ),
                transition="fallback",
            )
    next_step = _successor(
        previous,
        logical_step_id="next",
        attempts_used=1,
        candidate_attempt=1,
        provider_effect_observed=False,
    )
    validate_model_failover_successor(previous, next_step, transition="next_step")


@pytest.mark.parametrize("field", ["model", "execution_mode"])
def test_mutated_record_is_rejected_before_serialization(capsys, caplog, field):
    canary = "FAILOVER-RECORD-SECRET"

    class Hostile:
        def __repr__(self):
            return canary

        def __str__(self):
            return canary

    progress = _progress()
    object.__setattr__(progress.plan.candidates[0], field, Hostile())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as failure:
            progress.payload()
    assert canary not in str(failure.value)
    assert not captured
    assert canary not in caplog.text
    streams = capsys.readouterr()
    assert canary not in streams.out + streams.err


def test_plan_rejects_duplicate_and_unknown_schema_candidates():
    candidate = ModelFailoverCandidate(
        provider_name="p",
        model="m",
        execution_profile_fingerprint="a" * 64,
        execution_mode="synchronous",
    )
    with pytest.raises(ValueError, match="distinct"):
        ModelFailoverPlan(candidates=(candidate, candidate), max_total_attempts=2)
    for version in (True, "1", 2):
        payload = _progress().plan.payload()
        payload["schema_version"] = version
        with pytest.raises(ValueError, match="schema"):
            ModelFailoverPlan.model_validate(payload)


@pytest.mark.parametrize("mode", [None, True, 1, "future", {}, "BACKGROUND"])
def test_plan_requires_positive_known_execution_mode(mode):
    payload = _progress().plan.payload()
    payload["candidates"][1]["execution_mode"] = mode
    with pytest.raises(ValueError, match="mode"):
        ModelFailoverPlan.model_validate(payload)
    del payload["candidates"][1]["execution_mode"]
    with pytest.raises(ValueError, match="execution_mode"):
        ModelFailoverPlan.model_validate(payload)


def test_plan_mode_is_bound_and_cannot_change_between_candidates():
    previous = _progress().plan
    payload = previous.payload()
    payload["candidates"][1]["execution_mode"] = "background"
    with pytest.raises(ValueError, match="preserve.*mode"):
        ModelFailoverPlan.model_validate(payload)
    payload["candidates"][0]["execution_mode"] = "background"
    current = ModelFailoverPlan.model_validate(payload)
    assert current.fingerprint != previous.fingerprint
    assert ModelFailoverPlan.model_validate_json(current.model_dump_json()) == current
    progress = _progress(plan=current)
    with pytest.raises(ValueError, match="transition"):
        validate_model_failover_successor(
            progress,
            _successor(
                progress,
                candidate_index=1,
                candidate_attempt=1,
                projection_cursor=progress.source_transcript_cursor,
            ),
            transition="fallback",
        )


def test_checkpoint_upgrade_never_authenticates_old_route_lookalike():
    payload = _progress().payload()
    for version in range(1, 10):
        checkpoint = {
            CHECKPOINT_SCHEMA_VERSION_KEY: version,
            MODEL_FAILOVER_CHECKPOINT_KEY: payload,
            "application": {"preserved": True},
        }
        decoded = decode_runtime_checkpoint(checkpoint, session_id="session")
        assert decoded is not None
        assert decoded[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
        assert MODEL_FAILOVER_CHECKPOINT_KEY not in decoded
        assert decoded["application"] == {"preserved": True}
        assert checkpoint[MODEL_FAILOVER_CHECKPOINT_KEY] == payload


@pytest.mark.parametrize("writer_version", range(1, 10))
def test_older_writer_cannot_discard_route_authority(writer_version):
    checkpoint = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        MODEL_FAILOVER_CHECKPOINT_KEY: _progress().payload(),
    }
    with pytest.raises(ValueError, match="failover authority"):
        runtime_checkpoint_writer_view(
            checkpoint, writer_version=writer_version, session_id="session"
        )


def test_v9_writer_preserves_other_v9_authority_without_a_route():
    checkpoint = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "browser_controls": {"opaque": "preserved"},
    }
    projected = runtime_checkpoint_writer_view(checkpoint, writer_version=9, session_id="session")
    assert projected == {**checkpoint, CHECKPOINT_SCHEMA_VERSION_KEY: 9}


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_generic_store_checkpoint_cannot_inject_model_route(tmp_path, backend):
    asyncio.run(_assert_generic_store_checkpoint_cannot_inject_model_route(tmp_path, backend))


async def _assert_generic_store_checkpoint_cannot_inject_model_route(tmp_path, backend):
    store = (
        InMemorySessionStore()
        if backend == "memory"
        else SQLiteSessionStore(tmp_path / "route-injection.sqlite")
    )
    try:
        await store.create(
            RunRequest(agent_name="agent", session_id="session", messages=[]),
            identity=SessionIdentity(provider_name="primary", model="small"),
        )
        await store.checkpoint(
            "session",
            {
                CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
                MODEL_FAILOVER_CHECKPOINT_KEY: _progress().payload(),
                "application": True,
            },
        )
        checkpoint = await store.load_checkpoint("session")
        assert checkpoint is not None
        assert MODEL_FAILOVER_CHECKPOINT_KEY not in checkpoint
        assert checkpoint["application"] is True
    finally:
        if isinstance(store, SQLiteSessionStore):
            await store.close()


def test_public_checkpoint_operation_cannot_introduce_or_delete_model_route():
    for action in ("set", "delete"):
        with pytest.raises(ValueError, match="model-stage"):
            RuntimePublicationCheckpointOperation.model_validate(
                {
                    "key": MODEL_FAILOVER_CHECKPOINT_KEY,
                    "expected_value_digest": None if action == "set" else "a" * 64,
                    "action": action,
                    "value": _progress().payload() if action == "set" else None,
                }
            )
    # Construction bypass does not turn this generic application helper into
    # the stage transaction's private route mutation entrance.
    operation = RuntimePublicationCheckpointOperation.model_construct(
        key=MODEL_FAILOVER_CHECKPOINT_KEY,
        expected_value_digest=None,
        action="set",
        value=_progress().payload(),
    )
    mutation = RuntimePublicationMutation.model_construct(operations=(operation,))
    with pytest.raises(ValueError, match="model-stage"):
        apply_runtime_publication_checkpoint_mutation(mutation, None)
