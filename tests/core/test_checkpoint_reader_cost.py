"""Operation-count guards for call-local checkpoint reader ownership."""

from __future__ import annotations

from copy import deepcopy

import pytest

import cayu._validation as validation
from cayu.approvals.tools import PendingToolCallApproval
from cayu.approvals.user_input import (
    pending_user_input_from_checkpoint,
    user_input_lifecycle_authority_from_checkpoint,
    user_input_resolution_intent_from_checkpoint,
)
from cayu.runtime._approval_support import pending_approval_from_checkpoint
from cayu.runtime._tool_round_recovery import PendingToolRound, pending_tool_round_from_checkpoint
from cayu.sessions.pending_actions import (
    _pending_action_checkpoint_index_state,
    pending_action_evidence_round_from_checkpoint,
)
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize(
    "reader",
    [
        user_input_lifecycle_authority_from_checkpoint,
        pending_action_evidence_round_from_checkpoint,
        _pending_action_checkpoint_index_state,
    ],
)
@pytest.mark.parametrize("size", [0, 1000])
def test_compound_reader_admits_full_checkpoint_once(monkeypatch, reader, size):
    checkpoint = {"retained": [{"text": "history"}] * size}
    visits = []
    walk = validation._walk_bounded_durable_json

    def count(value, field_name, **kwargs):
        if field_name == "checkpoint":
            visits.append(value)
        return walk(value, field_name, **kwargs)

    monkeypatch.setattr(validation, "_walk_bounded_durable_json", count)
    reader(checkpoint)
    assert len(visits) == 1
    assert visits[0] is checkpoint


@pytest.mark.parametrize(
    "reader",
    [
        pending_user_input_from_checkpoint,
        user_input_resolution_intent_from_checkpoint,
        user_input_lifecycle_authority_from_checkpoint,
        pending_approval_from_checkpoint,
        pending_tool_round_from_checkpoint,
        pending_action_evidence_round_from_checkpoint,
    ],
)
def test_readers_still_reject_malformed_unrelated_retained_data(reader):
    with pytest.raises(validation.DurableValueError):
        reader({"retained": {"invalid": object()}})


@pytest.mark.parametrize(
    "reader",
    [
        pending_action_evidence_round_from_checkpoint,
        lambda checkpoint: _pending_action_checkpoint_index_state(checkpoint)[1],
    ],
)
def test_shared_snapshot_result_is_detached_and_never_cached(reader):
    pending = PendingToolRound(
        model_step_id="mstep_" + "1" * 32,
        model_attempt_id="matt_" + "2" * 32,
        tool_round_id="tround_" + "3" * 32,
        agent_name="worker",
        tool_calls=[
            PendingToolCallApproval(
                tool_call_id="call-1", tool_name="echo", arguments={"nested": ["before"]}
            )
        ],
    )
    checkpoint = {"pending_tool_round": pending.model_dump(mode="json")}
    original = deepcopy(checkpoint)
    result = reader(checkpoint)
    assert result is not None
    result.tool_calls[0].arguments["nested"].append("changed result")
    assert checkpoint == original
    checkpoint["pending_tool_round"]["tool_calls"][0]["arguments"]["nested"].append(
        "changed source"
    )
    fresh = reader(checkpoint)
    assert fresh is not None
    assert fresh.tool_calls[0].arguments["nested"] == ["before", "changed source"]
    assert result.tool_calls[0].arguments["nested"] == ["before", "changed result"]


@pytest.mark.parametrize("consume", [False, True])
def test_shared_user_input_snapshot_preserves_secret_rejection_ownership(consume):
    secret = "private-workload-token-1770"
    checkpoint = {"pending_user_input": {"arguments": {"token": secret}}}
    original = deepcopy(checkpoint)
    with pytest.raises(ValueError, match="workload secret"):
        user_input_lifecycle_authority_from_checkpoint(
            checkpoint, redactor=SecretRedactor().with_secret(secret), consume_on_rejection=consume
        )
    assert checkpoint == ({} if consume else original)


def test_internal_diagnostic_paths_need_no_sanitization(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("successful JSON walking must not sanitize generated paths")

    monkeypatch.setattr(validation, "_bounded_ascii_label", unexpected)
    assert validation.copy_durable_json_value({"private/key": [1, {"nested": "ok"}]}, "record") == {
        "private/key": [1, {"nested": "ok"}]
    }


def test_diagnostic_path_retains_bounded_index_only_representation():
    path = "$"
    for _ in range(100):
        path = validation._durable_child_path(path, 123456789, object_value=True)
    assert len(path) == 512 and path.endswith("...")
    with pytest.raises(validation.DurableValueError) as caught:
        validation.copy_durable_json_value({"secret-key": [{"other-secret": object()}]}, "record")
    assert caught.value.path == "$/#0/0/#0"
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize(
    ("reader", "key"),
    [
        (pending_user_input_from_checkpoint, "pending_user_input"),
        (pending_approval_from_checkpoint, "pending_tool_approval"),
        (pending_tool_round_from_checkpoint, "pending_tool_round"),
        (user_input_lifecycle_authority_from_checkpoint, "pending_user_input"),
    ],
)
@pytest.mark.parametrize("consume", [False, True])
def test_reader_wrappers_do_not_retain_rejected_secret_in_tracebacks(reader, key, consume):
    secret = "private-traceback-token-1770"
    checkpoint = {key: {"arguments": {"token": secret}}}
    with pytest.raises(ValueError, match="workload secret") as caught:
        reader(
            checkpoint, redactor=SecretRedactor().with_secret(secret), consume_on_rejection=consume
        )
    traceback = caught.value.__traceback__
    while traceback is not None:
        if "/src/cayu/" in traceback.tb_frame.f_code.co_filename:
            assert not [
                name for name, value in traceback.tb_frame.f_locals.items() if secret in repr(value)
            ]
        traceback = traceback.tb_next
    assert (not checkpoint) == consume


@pytest.mark.parametrize("change", ["none", "invalid_round", "invalid_retained", "conflict"])
def test_index_state_preserves_independent_approval_lookup(change):
    from tests.core.test_approval_from_event import _pending
    from tests.core.test_tool_round_publication import _pending_round

    approval = _pending()
    checkpoint = {"pending_tool_approval": approval.model_dump(mode="json")}
    if change == "invalid_round":
        checkpoint["pending_tool_round"] = {"invalid": True}
    elif change == "invalid_retained":
        checkpoint["retained"] = object()
    elif change == "conflict":
        checkpoint["pending_tool_round"] = _pending_round().model_dump(mode="json")
    ids, evidence = _pending_action_checkpoint_index_state(checkpoint)
    if change == "none":
        assert evidence is not None
        assert ids == frozenset(
            {
                approval.approval_id,
                evidence.tool_round_id,
                *(call.tool_call_id for call in evidence.tool_calls),
            }
        )
    else:
        assert evidence is None
        assert ids == (
            frozenset() if change == "invalid_retained" else frozenset({approval.approval_id})
        )
