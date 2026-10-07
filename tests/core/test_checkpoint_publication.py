"""Publication schema rules compose without a runtime adapter or concrete store."""

from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from tests._session_provenance import session_fixture

import cayu
from cayu.messages import Message
from cayu.sessions import _checkpoint_publication as publication
from cayu.sessions.base import (
    RuntimePublicationCheckpointOperation as Operation,
)
from cayu.sessions.base import (
    RuntimePublicationMutation as Mutation,
)
from cayu.sessions.base import (
    RuntimePublicationRequest as Request,
)
from cayu.sessions.base import (
    SessionRuntimePublicationConflict,
)
from cayu.sessions.base import (
    runtime_publication_checkpoint_value_digest as digest,
)
from cayu.sessions.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    CheckpointCompatibilityError,
    decode_runtime_checkpoint,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY as VERSION,
)
from cayu.sessions.checkpoints import (
    CURRENT_CHECKPOINT_SCHEMA_VERSION as CURRENT,
)


@pytest.fixture
def session():
    return session_fixture(
        id="session", agent_name="agent", provider_name="provider", model="model"
    )


def _request(*operations, kind="model-step"):
    return Request(
        publication_id="publication",
        kind=kind,
        interaction_id="interaction",
        intent={"retained": ["intent"]},
        mutation=Mutation(operations=operations),
        transcript_messages=(Message.text("assistant", "answer"),),
        events=(),
    )


def _set(key, value, expected=None):
    return Operation(key=key, action="set", value=value, expected_value_digest=expected)


def test_publication_codec_operates_without_runtime_dispatch_or_native_stores():
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
from cayu.sessions import _checkpoint_publication as codec
from cayu.sessions.base import RuntimePublicationRequest, RuntimePublicationMutation
from cayu.sessions.checkpoints import CHECKPOINT_SCHEMA_VERSION_KEY, CURRENT_CHECKPOINT_SCHEMA_VERSION
from tests._session_provenance import session_fixture
session = session_fixture(id='session', agent_name='agent', provider_name='provider', model='model')
request = RuntimePublicationRequest(
    publication_id='publication', kind='model-step', intent={},
    mutation=RuntimePublicationMutation(), transcript_messages=(), events=(),
)
stamped = codec._versioned_publication_request(request)
decoded = codec._decode_publication_checkpoint(session, None)
assert decoded == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
applied = codec._apply_publication_checkpoint_mutation(session, decoded, stamped.mutation)
assert codec._encode_publication_checkpoint(session, applied) == decoded
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(cayu.__file__).resolve().parent.parent), env.get("PYTHONPATH", "")]
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=30)


def test_request_stamping_preserves_bundle_and_detaches_mutable_data():
    request = _request(_set("ordinary", {"values": [1]}))
    before = request.model_dump(mode="json")
    stamped = publication._versioned_publication_request(request)
    assert type(stamped) is Request
    assert stamped.model_dump(mode="json", exclude={"mutation"}) == request.model_dump(
        mode="json", exclude={"mutation"}
    )
    operations = {operation.key: operation for operation in stamped.mutation.operations}
    assert operations[VERSION] == _set(VERSION, CURRENT, digest(CURRENT))
    assert operations["ordinary"] == request.mutation.operations[0]
    operations["ordinary"].value["values"].append(2)
    stamped.intent["retained"].append("changed")
    assert request.model_dump(mode="json") == before


def test_auxiliary_request_keeps_exact_no_mutation_identity():
    request = _request(kind="auxiliary-inference")
    assert publication._versioned_publication_request(request) is request
    with pytest.raises(ValueError, match="cannot mutate the parent checkpoint"):
        publication._versioned_publication_request(
            _request(_set("ordinary", 1), kind="auxiliary-inference")
        )


@pytest.mark.parametrize("expected_version", [None, *range(1, CURRENT + 1)])
def test_workspace_schema_stamp_preserves_original_fence(expected_version):
    expected = None if expected_version is None else digest(expected_version)
    request = _request(_set(VERSION, CURRENT, expected), kind="workspace-observation")
    assert publication._versioned_publication_request(request) is request


@pytest.mark.parametrize("invalid", ["kind", "version", "digest", "action", "duplicate"])
def test_request_rejects_invalid_explicit_schema_stamp(invalid):
    operation = _set(VERSION, CURRENT, digest(CURRENT))
    kind = "model-step" if invalid == "kind" else "workspace-observation"
    if invalid == "version":
        operation = _set(VERSION, CURRENT + 1, digest(CURRENT))
    elif invalid == "digest":
        operation = _set(VERSION, CURRENT, digest(CURRENT + 1))
    elif invalid == "action":
        operation = Operation(key=VERSION, action="delete", expected_value_digest=digest(CURRENT))
    request = _request(operation, kind=kind)
    if invalid == "duplicate":
        # A forged nested record must not bypass the codec's own schema check.
        request = request.model_copy(
            update={"mutation": Mutation.model_construct(operations=(operation, operation))}
        )
    with pytest.raises(ValueError, match="schema"):
        publication._versioned_publication_request(request)


@pytest.mark.parametrize(
    "key",
    [
        ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
        SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
        COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    ],
)
def test_request_rejects_protected_authority_even_in_forged_records(key):
    operation = Operation.model_construct(
        key=key, action="set", value={}, expected_value_digest=None
    )
    request = _request().model_copy(
        update={"mutation": Mutation.model_construct(operations=(operation,))}
    )
    with pytest.raises(ValueError, match="authority"):
        publication._versioned_publication_request(request)


def test_absent_root_decode_and_encode_have_distinct_semantics(session):
    assert publication._decode_publication_checkpoint(session, None) == {VERSION: CURRENT}
    assert publication._encode_publication_checkpoint(session, None) is None


@pytest.mark.parametrize("version", [None, 1, CURRENT])
@pytest.mark.parametrize(
    "method", ["_decode_publication_checkpoint", "_encode_publication_checkpoint"]
)
def test_codec_migrates_and_detaches_retained_data(session, version, method):
    checkpoint = {"ordinary": {"values": [1]}}
    if version is not None:
        checkpoint[VERSION] = version
    before = deepcopy(checkpoint)
    result = getattr(publication, method)(session, checkpoint)
    assert result == {VERSION: CURRENT, "ordinary": {"values": [1]}}
    result["ordinary"]["values"].append(2)
    assert checkpoint == before


@pytest.mark.parametrize("method", ["decode", "encode", "apply"])
def test_codec_rejects_future_input_with_session_evidence(session, method):
    checkpoint = {VERSION: CURRENT + 1, "ordinary": ["retained"]}
    before = deepcopy(checkpoint)
    with pytest.raises(CheckpointCompatibilityError) as caught:
        if method == "apply":
            publication._apply_publication_checkpoint_mutation(session, checkpoint, Mutation())
        else:
            getattr(publication, f"_{method}_publication_checkpoint")(session, checkpoint)
    assert caught.value.session_id == session.id
    assert checkpoint == before


@pytest.mark.parametrize("writer_version", range(1, CURRENT + 1))
def test_mutation_evaluates_writer_schema_fence_then_upcasts(session, writer_version):
    checkpoint = {VERSION: CURRENT, "ordinary": ["before"]}
    before = deepcopy(checkpoint)
    mutation = Mutation(
        operations=(
            _set(VERSION, writer_version, digest(writer_version)),
            _set("ordinary", ["after"], digest(["before"])),
        )
    )
    result = publication._apply_publication_checkpoint_mutation(session, checkpoint, mutation)
    assert result == decode_runtime_checkpoint(
        {VERSION: writer_version, "ordinary": ["after"]}, session_id=session.id
    )
    result["ordinary"].append("detached")
    assert checkpoint == before
    assert next(op for op in mutation.operations if op.key == "ordinary").value == ["after"]


def test_current_writer_initializes_absent_root_without_changing_original_fence(session):
    mutation = Mutation(operations=(_set(VERSION, CURRENT), _set("ordinary", [1])))
    before = mutation.model_dump(mode="json")
    assert publication._apply_publication_checkpoint_mutation(session, None, mutation) == {
        VERSION: CURRENT,
        "ordinary": [1],
    }
    assert mutation.model_dump(mode="json") == before


def test_conflicting_mutation_leaves_input_unchanged(session):
    checkpoint = {VERSION: CURRENT, "a": [1], "z": [2]}
    before = deepcopy(checkpoint)
    mutation = Mutation(operations=(_set("a", [3], digest([1])), _set("z", [4], digest([0]))))
    with pytest.raises(SessionRuntimePublicationConflict, match="'z'"):
        publication._apply_publication_checkpoint_mutation(session, checkpoint, mutation)
    assert checkpoint == before


def test_mutation_output_is_validated_before_return(session):
    checkpoint = {VERSION: CURRENT, "ordinary": [1]}
    # JSON-valid content can still be invalid under the checkpoint schema.
    mutation = Mutation(operations=(_set(VERSION, True, digest(CURRENT)),))
    with pytest.raises(CheckpointCompatibilityError):
        publication._apply_publication_checkpoint_mutation(session, checkpoint, mutation)
    assert checkpoint == {VERSION: CURRENT, "ordinary": [1]}
