"""Schema-aware callback behavior stays independent of runtime dispatch."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._session_provenance import fixture_session_invocation

import cayu
from cayu.sessions import _checkpoint_transforms as transforms
from cayu.sessions.base import ModelCompletionStageRelease, Session, SessionOperationPublication
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY as VERSION,
)
from cayu.sessions.checkpoints import (
    CURRENT_CHECKPOINT_SCHEMA_VERSION as CURRENT,
)
from cayu.sessions.checkpoints import (
    CheckpointCompatibilityError,
)

_KINDS = ("checkpoint", "store_time_checkpoint", "operation", "store_time_operation")
_FACTORIES = {
    "checkpoint": transforms._versioned_checkpoint_transform,
    "store_time_checkpoint": transforms._versioned_store_time_checkpoint_transform,
    "operation": transforms._versioned_operation_transform,
    "store_time_operation": transforms._versioned_store_time_operation_transform,
}


@pytest.fixture
def session():
    return Session(
        id="session",
        agent_name="agent",
        provider_name="provider",
        model="model",
        invocation=fixture_session_invocation("session"),
    )


def _invoke(kind, callback, session, checkpoint, *, record=None, now=None, **options):
    wrapped = _FACTORIES[kind](session.id, callback, **options)
    args = [session, checkpoint]
    if "operation" in kind:
        args.append(record)
    if kind.startswith("store_time"):
        args.append(now)
    return wrapped(*args)


def _result(kind, checkpoint):
    if "operation" in kind:
        return SessionOperationPublication(checkpoint=checkpoint)
    return checkpoint


def test_checkpoint_transforms_import_and_operate_without_runtime_dispatch():
    code = """
import importlib.abc
import sys

class BlockDispatch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {
            'cayu.runtime._checkpoint_store', 'cayu.runtime._session_engine',
            'cayu.runtime._invocation_lifecycle', 'cayu.runtime._producer_output_store',
            'cayu.storage.sqlite', 'cayu.storage.postgres',
        }:
            raise AssertionError(fullname)

sys.meta_path.insert(0, BlockDispatch())
from cayu.sessions import _checkpoint_transforms as transforms
from cayu.sessions.base import Session, SessionOperationPublication
from cayu.sessions.checkpoints import CHECKPOINT_SCHEMA_VERSION_KEY, CURRENT_CHECKPOINT_SCHEMA_VERSION
from tests._session_provenance import fixture_session_invocation
session = Session(
    id='session', agent_name='agent', provider_name='provider', model='model',
    invocation=fixture_session_invocation('session'),
)
checkpoint = {'ordinary': [1]}
result = transforms._versioned_checkpoint_transform(
    session.id, lambda session, state: state
)(session, checkpoint)
assert result == {**checkpoint, CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
publication = transforms._versioned_operation_transform(
    session.id, lambda session, state, record: SessionOperationPublication(checkpoint=state)
)(session, checkpoint, None)
assert type(publication) is SessionOperationPublication
assert publication.checkpoint == result
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(cayu.__file__).resolve().parent.parent), env.get("PYTHONPATH", "")]
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=30)


@pytest.mark.parametrize("kind", _KINDS)
def test_callback_receives_detached_migrated_state_and_exact_context(kind, session):
    checkpoint = {"ordinary": {"nested": ["before"]}}
    original = deepcopy(checkpoint)
    record = {"retained": [1]}
    now = datetime(2030, 1, 1, tzinfo=UTC)
    received = []

    def callback(actual_session, state, *extra):
        assert actual_session is session
        assert state == {**original, VERSION: CURRENT}
        assert state is not checkpoint
        if "operation" in kind:
            assert extra[0] is record
        if kind.startswith("store_time"):
            assert extra[-1] is now
        received.append(state)
        state["ordinary"]["nested"].append("callback")
        return _result(kind, state)

    result = _invoke(kind, callback, session, checkpoint, record=record, now=now)
    state = result.checkpoint if "operation" in kind else result
    assert state == {VERSION: CURRENT, "ordinary": {"nested": ["before", "callback"]}}
    assert checkpoint == original
    received[0]["ordinary"]["nested"].append("later callback mutation")
    assert state["ordinary"]["nested"] == ["before", "callback"]
    state["ordinary"]["nested"].append("later result mutation")
    assert checkpoint == original


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("invalid", ["future_schema", "non_json"])
def test_invalid_input_is_rejected_before_callback(kind, invalid, session):
    checkpoint = {VERSION: CURRENT + 1} if invalid == "future_schema" else {"bad": object()}
    calls = []

    def callback(*args):
        calls.append(args)
        return _result(kind, {})

    error_type = CheckpointCompatibilityError if invalid == "future_schema" else ValueError
    with pytest.raises(error_type):
        _invoke(kind, callback, session, checkpoint)
    assert calls == []
    assert checkpoint


@pytest.mark.parametrize("kind", _KINDS)
def test_mutated_callback_output_is_validated_again(kind, session):
    checkpoint = {"ordinary": ["before"]}
    received = []

    def callback(_session, state, *extra):
        received.append(state)
        state[VERSION] = CURRENT + 1
        return _result(kind, state)

    with pytest.raises(CheckpointCompatibilityError):
        _invoke(kind, callback, session, checkpoint)
    assert received == [{}]
    assert checkpoint == {"ordinary": ["before"]}


@pytest.mark.parametrize("kind", _KINDS)
def test_cancellation_clears_callback_snapshot_without_mutating_store_input(kind, session):
    checkpoint = {"ordinary": ["private value"]}
    received = []
    error = asyncio.CancelledError()

    def callback(_session, state, *extra):
        received.append(state)
        raise error

    with pytest.raises(asyncio.CancelledError) as caught:
        _invoke(kind, callback, session, checkpoint)
    assert caught.value is error
    assert received == [{}]
    assert checkpoint == {"ordinary": ["private value"]}


@pytest.mark.parametrize(
    "checkpoint,options,expected",
    [
        (None, {}, None),
        ({}, {}, None),
        (None, {"stamp_noop": True}, None),
        ({}, {"stamp_noop": True}, {VERSION: CURRENT}),
        ({"ordinary": [1]}, {"stamp_noop": True}, {VERSION: CURRENT, "ordinary": [1]}),
        (None, {"stamp_empty": True}, {VERSION: CURRENT}),
        ({"ordinary": [1]}, {"stamp_empty": True}, {VERSION: CURRENT}),
        (
            {"ordinary": [1]},
            {"stamp_empty": True, "stamp_noop": True},
            {VERSION: CURRENT},
        ),
    ],
)
def test_noop_stamping_distinguishes_absent_empty_and_retained_state(
    checkpoint, options, expected, session
):
    before = deepcopy(checkpoint)

    def discard(_session, state):
        if state is not None:
            state["discarded"] = True
        return None

    assert _invoke("checkpoint", discard, session, checkpoint, **options) == expected
    assert checkpoint == before


@pytest.mark.parametrize("kind", ["operation", "store_time_operation"])
def test_operation_transform_preserves_records_and_exact_release_contract(kind, session):
    release = ModelCompletionStageRelease(stage_id="stage", preparation_digest="a" * 64)
    publication = SessionOperationPublication(
        checkpoint={"ordinary": [1]},
        operation_records={"custom-operation": {"value": [1]}},
        model_completion_stage_release=release,
    )
    result = _invoke(kind, lambda *_: publication, session, None)
    assert type(result) is SessionOperationPublication
    assert result.model_completion_stage_release is release
    assert result.operation_records == publication.operation_records
    publication.operation_records["custom-operation"]["value"].append(2)
    publication.checkpoint["ordinary"].append(2)
    assert result.operation_records == {"custom-operation": {"value": [1]}}
    assert result.checkpoint == {VERSION: CURRENT, "ordinary": [1]}


@pytest.mark.parametrize("kind", ["operation", "store_time_operation"])
def test_operation_transform_rejects_subclass_publication(kind, session):
    class DerivedPublication(SessionOperationPublication):
        pass

    with pytest.raises(TypeError, match="must return a SessionOperationPublication"):
        _invoke(kind, lambda *_: DerivedPublication(checkpoint={}), session, None)


def test_optional_transform_and_preserve_callback_keep_absence_and_identity(session):
    assert transforms._optional_versioned_checkpoint_transform(session.id, None) is None
    checkpoint = {"ordinary": [1]}
    assert transforms._preserve_checkpoint(session, checkpoint) is checkpoint
    assert transforms._preserve_checkpoint(session, None) is None
