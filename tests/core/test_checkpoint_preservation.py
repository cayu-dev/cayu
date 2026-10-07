"""Shared authority and snapshot boundaries used by native checkpoint stores."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import cayu
from cayu.sessions import _checkpoint_preservation as preservation
from cayu.sessions.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
)

_SCOPES = [
    (
        preservation._invocation_lifecycle_authority_mutation_scope,
        preservation._INVOCATION_LIFECYCLE_AUTHORITY_MUTATION_ALLOWED,
    ),
    (
        preservation._invocation_lifecycle_authority_read_scope,
        preservation._INVOCATION_LIFECYCLE_AUTHORITY_READ_ALLOWED,
    ),
    (
        preservation._workspace_observation_authority_mutation_scope,
        preservation._WORKSPACE_OBSERVATION_AUTHORITY_MUTATION_ALLOWED,
    ),
]


def test_checkpoint_preservation_operates_without_runtime_or_native_stores():
    code = """
import importlib.abc
import sys

class BlockExecutionOwners(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'cayu.sessions.base', 'cayu.runtime'} or fullname.startswith(
            ('cayu.runtime.', 'cayu.storage.sqlite', 'cayu.storage.postgres')
        ):
            raise AssertionError(fullname)

sys.meta_path.insert(0, BlockExecutionOwners())
from cayu.sessions import _checkpoint_preservation as preservation
checkpoint = {'ordinary': {'nested': [1]}}
assert preservation._copy_checkpoint_for_transform(
    checkpoint, session_id='session'
) == checkpoint
assert preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
    None, checkpoint, session_id='session'
) == checkpoint
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(cayu.__file__).resolve().parent.parent), env.get("PYTHONPATH", "")]
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=30)


@pytest.mark.parametrize("scope,context", _SCOPES)
def test_checkpoint_authority_scope_restores_nested_and_failed_calls(scope, context):
    assert context.get() is False
    with scope():
        assert context.get() is True
        with pytest.raises(RuntimeError, match="failed publication"), scope():
            assert context.get() is True
            raise RuntimeError("failed publication")
        assert context.get() is True
    assert context.get() is False


@pytest.mark.parametrize("scope,context", _SCOPES)
def test_checkpoint_authority_scope_isolated_from_concurrent_task(scope, context):
    async def scenario():
        entered = asyncio.Event()
        checked = asyncio.Event()

        async def authorized():
            with scope():
                entered.set()
                await checked.wait()
                assert context.get() is True

        async def ordinary():
            await entered.wait()
            try:
                assert context.get() is False
            finally:
                checked.set()

        async with asyncio.timeout(10):
            await asyncio.gather(authorized(), ordinary())
        assert context.get() is False

    asyncio.run(scenario())


@pytest.mark.parametrize("decoded", [False, True])
def test_callback_checkpoint_is_detached_in_both_directions(decoded):
    current = {"ordinary": {"nested": ["before"]}}
    visible = preservation._copy_checkpoint_for_transform(
        current, session_id="session", decoded=decoded
    )
    assert visible == current
    assert visible is not None
    visible["ordinary"]["nested"].append("callback")
    assert current == {"ordinary": {"nested": ["before"]}}
    current["ordinary"]["nested"].append("store")
    assert visible == {"ordinary": {"nested": ["before", "callback"]}}


@pytest.mark.parametrize(
    "scope",
    [
        preservation._invocation_lifecycle_authority_read_scope,
        preservation._invocation_lifecycle_authority_mutation_scope,
    ],
)
def test_decoded_lifecycle_visibility_uses_canonical_scope_and_detaches(scope):
    roots = (
        ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
        SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    )
    # This boundary receives an already-decoded snapshot; domain/schema readers
    # validate the real root records before passing ownership to this path.
    current = {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
    current.update({key: {"nested": ["owned"]} for key in roots})
    original = deepcopy(current)
    ordinary = preservation._copy_checkpoint_for_transform(
        current, session_id="session", decoded=True
    )
    assert ordinary == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
    with scope():
        visible = preservation._copy_checkpoint_for_transform(
            current, session_id="session", decoded=True
        )
        assert visible == current
        for key in roots:
            visible[key]["nested"].append("callback")
    assert current == original
    assert (
        preservation._copy_checkpoint_for_transform(current, session_id="session", decoded=True)
        == ordinary
    )


def test_transform_result_keeps_validated_completion_reservation_detached():
    digest = "a" * 64
    publication_id = "completion-result-publication:v1:" + digest
    owner_id = "completion-result-owner:v1:" + "b" * 64
    reservation = {
        "schema_version": 2,
        "reservations": {
            publication_id: {
                "schema_version": 2,
                "publication_id": publication_id,
                "authority_sha256": digest,
                "owners": {
                    owner_id: {
                        "schema_version": 2,
                        "owner_id": owner_id,
                        "expires_at": "2030-01-01T00:00:00+00:00",
                    }
                },
            }
        },
    }
    current = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY: reservation,
    }
    original = deepcopy(current)
    transformed = {"ordinary": ["callback"]}
    result = (
        preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
            current, transformed, session_id="session"
        )
    )
    assert result[COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY] == reservation
    result[COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY]["reservations"].clear()
    result["ordinary"].append("changed")
    assert current == original
    assert transformed == {"ordinary": ["callback"]}


def test_transform_result_rejects_malformed_retained_completion_reservation():
    current = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY: {
            "schema_version": 2,
            "reservations": {"invalid": {}},
        },
    }
    original = deepcopy(current)
    with pytest.raises(ValueError, match="publication reservation is malformed"):
        preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
            current, {}, session_id="session"
        )
    assert current == original
