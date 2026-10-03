"""Runtime producers and native validation share exact continuation authority."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import pickle
import subprocess
import sys
from contextvars import ContextVar
from pathlib import Path

import pytest
from tests.core.test_session_continuation import (
    _admitted_store,
    _consumption,
    _context,
    _preparation,
    _ready_continuation,
    _ticket,
)
from tests.core.test_temporary_continuation_admission import native_command
from tests.core.test_temporary_continuation_contracts import admission

import cayu
from cayu.runtime import _session_continuation_scope as runtime
from cayu.runtime import _temporary_continuation_scope as temporary_runtime
from cayu.runtime._temporary_service_target import target_service_key
from cayu.sessions import _session_continuation_scope as shared
from cayu.sessions import _temporary_continuation_scope as temporary_shared
from cayu.sessions._session_continuation import (
    CONTINUATION_NAMESPACE_KEY,
    ContinuationConflict,
    ContinuationRetirement,
    continuation_operation_key,
)
from cayu.sessions._temporary_continuation import temporary_service_key
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.base import InMemorySessionStore


@pytest.mark.parametrize(
    "module_name", ("_session_continuation_scope", "_temporary_continuation_scope")
)
def test_legacy_scopes_share_contexts_callables_and_historical_pickle_paths(module_name):
    canonical = importlib.import_module(f"cayu.sessions.{module_name}")
    legacy = importlib.import_module(f"cayu.runtime.{module_name}")
    contexts = []
    for name, value in vars(canonical).items():
        if isinstance(value, ContextVar):
            assert getattr(legacy, name) is value
            contexts.append(value)
        elif inspect.isfunction(value) and value.__module__ == canonical.__name__:
            assert getattr(legacy, name) is value
            assert pickle.loads(f"c{legacy.__name__}\n{name}\n.".encode()) is value
    assert len(contexts) == (8 if module_name == "_session_continuation_scope" else 1)


@pytest.mark.parametrize("producer", (runtime, shared), ids=("runtime", "sessions"))
def test_publication_scope_binds_exact_parent_child_and_restores_nested_authority(producer):
    with pytest.raises(PermissionError):
        shared.require_publication("parent")
    with producer.service_publication_scope("parent", "child"):
        for receiver in (runtime, shared):
            receiver.require_publication("parent")
            receiver.require_publication("child")
            assert receiver.current_publication_key() == "parent"
            with pytest.raises(PermissionError):
                receiver.require_publication("sibling")
        with (
            pytest.raises(RuntimeError, match="unwind"),
            shared.publication_scope("nested"),
        ):
            runtime.require_publication("nested")
            with pytest.raises(PermissionError):
                runtime.require_publication("child")
            raise RuntimeError("unwind")
        shared.require_publication("child")
    assert runtime.current_publication_key() is None
    assert not shared.continuation_authority_visible()
    with pytest.raises(PermissionError):
        runtime.require_publication("child")


def test_publication_authority_is_task_local_and_resets_after_cancellation():
    async def run():
        entered = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()

        async def worker(index):
            try:
                with shared.service_publication_scope(f"parent-{index}", f"child-{index}"):
                    entered[index].set()
                    await release.wait()
                    runtime.require_publication(f"child-{index}")
                    with pytest.raises(PermissionError):
                        runtime.require_publication(f"child-{1 - index}")
            finally:
                assert shared.current_publication_key() is None
                assert not runtime.continuation_authority_visible()

        tasks = [asyncio.create_task(worker(index)) for index in range(2)]
        try:
            async with asyncio.timeout(10):
                await asyncio.gather(*(event.wait() for event in entered))
                assert shared.current_publication_key() is None
                tasks[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await tasks[0]
                release.set()
                await tasks[1]
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


def test_runtime_invocation_scopes_keep_authentication_and_native_writer_revalidation():
    async def run():
        store, session, interaction = await _admitted_store()
        ticket = _ticket(session, interaction)
        invocation = await _context(store, session.id)
        preparation = await _preparation(store, ticket)
        checkpoint = await store.load_checkpoint(session.id)
        retirement = ContinuationRetirement(
            ticket=ticket,
            control_id="scope-retirement",
            reason="cancelled",
            retired_at="2030-01-01T00:00:00+00:00",
        )
        for make_scope in (
            lambda value: runtime.preparation_scope(preparation, value),
            lambda value: runtime.park_scope(ticket, value),
            lambda value: runtime.retirement_scope(retirement, value),
        ):
            with pytest.raises(PermissionError), make_scope(object()):
                pytest.fail("A caller-shaped value cannot establish invocation authority")
            assert not shared.continuation_authority_visible()

        with runtime.preparation_scope(preparation, invocation):
            assert shared._PREPARATION.get()[2] is invocation
            shared.require_preparation(preparation)
            shared.require_namespace_preparation(ticket.namespace)
            shared.require_preparation_writer(session, checkpoint)
            with pytest.raises(SessionRunFenced):
                shared.require_preparation_writer(
                    session.model_copy(update={"run_epoch": session.run_epoch + 1}), checkpoint
                )
            token = invocation._authority_token
            try:
                object.__setattr__(invocation, "_authority_token", object())
                with pytest.raises(TypeError):
                    shared.require_preparation_writer(session, checkpoint)
            finally:
                object.__setattr__(invocation, "_authority_token", token)
        with runtime.park_scope(ticket, invocation):
            shared.require_park(ticket)
        with runtime.retirement_scope(retirement, invocation):
            assert shared.require_retirement(retirement) is False
        for validate in (
            lambda: shared.require_preparation(preparation),
            lambda: shared.require_namespace_preparation(ticket.namespace),
            lambda: shared.require_preparation_writer(session, checkpoint),
            lambda: shared.require_park(ticket),
            lambda: shared.require_retirement(retirement),
        ):
            with pytest.raises(PermissionError):
                validate()
        assert not shared.continuation_authority_visible()

    asyncio.run(run())


def test_latch_consumption_and_admission_claims_share_exact_evidence():
    async def run():
        _, ticket, latch = await _ready_continuation()
        consumption = _consumption(ticket, latch, "scope-consumption")
        claimed = consumption.model_copy(
            update={"admission_claimed": True, "admission_claim_id": "claim"}
        )
        with runtime.authenticated_latch_scope(latch):
            shared.require_authenticated_latch(latch)
            with pytest.raises(PermissionError):
                shared.require_authenticated_latch(latch.model_copy(update={"latch_key": "other"}))
        with runtime.consumption_scope(consumption):
            shared.require_consumption(claimed)
            with pytest.raises(PermissionError):
                shared.require_consumption(
                    consumption.model_copy(update={"continuation_id": "other"})
                )
            with (
                pytest.raises(PermissionError),
                shared.admission_claim_scope(consumption),
            ):
                pytest.fail("An unclaimed consumption cannot establish admission authority")
            with shared.admission_claim_scope(claimed):
                assert runtime.current_admission_claim() is claimed
            assert runtime.current_admission_claim() is None
        with pytest.raises(PermissionError):
            shared.require_authenticated_latch(latch)
        with pytest.raises(PermissionError):
            shared.require_consumption(consumption)
        assert not shared.continuation_authority_visible()

    asyncio.run(run())


@pytest.mark.parametrize("side_session", (False, True), ids=("same-session", "side-session"))
def test_temporary_scope_shares_exact_admission_and_resets_on_exception(side_session):
    async def run():
        admitted, command = await native_command(InMemorySessionStore(), side_session=side_session)
        with pytest.raises(PermissionError):
            temporary_shared.require_temporary_admission(command)
        with (
            pytest.raises(ContinuationConflict),
            temporary_runtime.temporary_admission_scope(
                admitted, command.model_copy(update={"participant_permit_commitment": "0" * 64})
            ),
        ):
            pytest.fail("A mismatched command cannot establish temporary authority")
        assert temporary_shared._ADMISSION.get() is None
        with (
            pytest.raises(RuntimeError, match="unwind"),
            temporary_runtime.temporary_admission_scope(admitted, command),
        ):
            assert temporary_shared.require_temporary_admission(command) == admitted
            assert temporary_shared.prepare_temporary_transition(admitted) == admitted
            expected_key = (
                target_service_key(admitted.dispatch.intent.operation)
                if side_session
                else continuation_operation_key(admitted.dispatch.intent.ticket)
            )
            assert shared.current_publication_key() == expected_key
            shared.require_publication(
                CONTINUATION_NAMESPACE_KEY
                if side_session
                else temporary_service_key(admitted.dispatch.intent.operation)
            )
            with pytest.raises(PermissionError):
                temporary_shared.prepare_temporary_transition(None)
            with pytest.raises(PermissionError):
                temporary_shared.require_temporary_admission(
                    command.model_copy(update={"temporary_service_operation_key": None})
                )
            with pytest.raises(PermissionError):
                temporary_shared.require_temporary_transition(
                    admitted.model_copy(update={"permit_receipt_sha256": "0" * 64})
                )
            raise RuntimeError("unwind")
        assert temporary_shared._ADMISSION.get() is None
        assert shared.current_publication_key() is None
        assert temporary_shared.prepare_temporary_transition(None) is None
        with pytest.raises(PermissionError):
            temporary_runtime.require_temporary_transition(admitted)

    asyncio.run(run())


def test_shared_scopes_work_without_loading_runtime_or_store_implementations():
    script = """
import importlib.abc
import json
import sys
class RejectRuntimeAndStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "cayu.runtime" or fullname.startswith("cayu.runtime.")
                or fullname == "cayu.storage" or fullname.startswith("cayu.storage.")
                or fullname == "cayu.sessions.base"):
            raise AssertionError(f"Shared continuation scopes imported {fullname}")
sys.meta_path.insert(0, RejectRuntimeAndStores())
from cayu.sessions import _session_continuation_scope as scope
from cayu.sessions import _temporary_continuation_scope as temporary
from cayu.sessions._temporary_continuation import TemporaryServiceAdmission
admission = TemporaryServiceAdmission.model_validate(json.load(sys.stdin))
assert not scope.continuation_authority_visible()
with scope.service_publication_scope("parent", "child"):
    scope.require_publication("parent")
    scope.require_publication("child")
assert scope.current_publication_key() is None
assert temporary.prepare_temporary_transition(None) is None
try:
    temporary.require_temporary_transition(admission)
except PermissionError:
    pass
else:
    raise AssertionError("A typed record cannot grant temporary authority")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=admission().model_dump_json(),
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
