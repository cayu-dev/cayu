from __future__ import annotations

import asyncio

import pytest
from tests.core.test_tool_effect_reconciliation_registration import _register, _spec
from tests.core.test_tool_effect_state import _intent, _receipt

from cayu.runtime import InMemorySessionStore
from cayu.runtime._tool_effect_reconciliation import (
    ToolEffectReconciliationOwner,
    ToolEffectReconciliationTimeout,
    project_accepted_reconciliation,
)
from cayu.runtime._tool_effect_state import ToolEffectConflict, ToolEffectStateOwner
from cayu.runtime.tool_effects import (
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)
from cayu.tools._operation_boundary import InvocationOperationCapacityError
from cayu.vaults.redaction import SecretRedactor


async def _setup(reconciler, **spec_changes):
    registered = _register(
        ToolEffectReconciliationRegistration(reconciler=reconciler, spec=_spec(**spec_changes))
    )
    store = InMemorySessionStore()
    intent = (await _intent(store)).model_copy(
        update={"reconciler_fingerprint": registered.fingerprint}
    )
    state_owner = ToolEffectStateOwner(store)
    prepared = await state_owner.prepare(intent, run_epoch=0)
    executing = await state_owner.transition(prepared, state="executing", run_epoch=0)
    record = await state_owner.transition(executing, state="outcome_unknown", run_epoch=0)
    request = ToolEffectReconciliationRequest(
        **{
            name: getattr(intent, name)
            for name in (
                "session_id",
                "session_instance_id",
                "tool_round_id",
                "tool_call_id",
                "tool_name",
                "idempotency_key",
            )
        },
        expected_run_epoch=0,
        expected_revision=record.revision,
        receipt=_receipt(intent),
    )
    return registered, record, request, state_owner


class _Echo:
    def __init__(self):
        self.calls = 0
        self.result = None

    async def reconcile(self, *, context, receipt):
        self.calls += 1
        self.result = ToolEffectReconciliationResult(
            outcome="completed",
            observation="sent",
            receipt=receipt,
        )
        return self.result


def test_owned_reconciliation_validates_and_detaches_without_settling_call():
    async def scenario():
        callback = _Echo()
        registered, record, request, state_owner = await _setup(callback)
        owner = ToolEffectReconciliationOwner()
        accepted = await owner.reconcile(
            request=request,
            record=record,
            run_epoch=0,
            registered=registered,
        )
        assert callback.calls == 1
        assert accepted.result.receipt == request.receipt
        callback.result.receipt.resource_versions["changed"] = "later"
        assert accepted.result.receipt.resource_versions == {}
        assert accepted.context.record_revision == record.revision
        assert len(accepted.request_digest) == 64
        assert await state_owner.load(record.intent) == record
        assert owner.pending_operations == 0
        assert await owner.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("source", ["adapter", "reconciler", "operator"])
def test_accepted_receipt_preserves_validated_source_control(source):
    async def scenario():
        callback = _Echo()
        registered, record, request, _ = await _setup(callback)
        request = request.model_copy(
            update={"receipt": request.receipt.model_copy(update={"source": source})}
        )
        owner = ToolEffectReconciliationOwner()
        accepted = await owner.reconcile(
            request=request, record=record, run_epoch=0, registered=registered
        )
        projected = project_accepted_reconciliation(
            accepted, registered=registered, redactor=SecretRedactor([source, "completed"])
        )
        assert projected.receipt.source == source
        assert projected.receipt.outcome == "completed"
        assert owner.pending_operations == 0
        assert await owner.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "other"),
        ("session_instance_id", "other"),
        ("tool_round_id", "other"),
        ("expected_revision", 99),
        ("expected_run_epoch", 99),
    ],
)
def test_identity_conflict_precedes_application_callback(field, value):
    async def scenario():
        callback = _Echo()
        registered, record, request, _ = await _setup(callback)
        request = request.model_copy(update={field: value})
        owner = ToolEffectReconciliationOwner()
        with pytest.raises(ToolEffectConflict):
            await owner.reconcile(
                request=request, record=record, run_epoch=0, registered=registered
            )
        assert callback.calls == 0
        assert owner.pending_operations == 0
        assert await owner.aclose()

    asyncio.run(scenario())


class _Barrier:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.settled = asyncio.Event()
        self.calls = 0

    async def reconcile(self, *, context, receipt):
        self.calls += 1
        self.entered.set()
        try:
            await self.release.wait()
            return ToolEffectReconciliationResult(outcome="not_found", observation="not_sent")
        finally:
            self.settled.set()


@pytest.mark.parametrize("signal", ["deadline", "cancel", "repeated_cancel"])
def test_abandoned_lookup_retains_capacity_until_actual_settlement(signal):
    async def scenario():
        callback = _Barrier()
        registered, record, request, state_owner = await _setup(
            callback,
            timeout_seconds=0.03 if signal == "deadline" else 30,
        )
        owner = ToolEffectReconciliationOwner(max_operations=1)

        async def invoke():
            return await owner.reconcile(
                request=request,
                record=record,
                run_epoch=0,
                registered=registered,
            )

        task = asyncio.create_task(invoke())
        try:
            await asyncio.wait_for(callback.entered.wait(), 1)
            if signal == "deadline":
                with pytest.raises(ToolEffectReconciliationTimeout):
                    await task
                assert not task.cancelled()
                assert task.cancelling() == 0
            else:
                count = 2 if signal == "repeated_cancel" else 1
                for _ in range(count):
                    task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled()
                assert task.cancelling() == count
            assert owner.pending_operations == 1
            assert not callback.settled.is_set()
            with pytest.raises(InvocationOperationCapacityError):
                await invoke()
            assert callback.calls == 1
            assert await state_owner.load(record.intent) == record
            callback.release.set()
            await asyncio.wait_for(callback.settled.wait(), 1)
            # Completion callbacks release capacity; no polling delay or wall-clock assumption.
            for _ in range(4):
                await asyncio.sleep(0)
            assert owner.pending_operations == 0
            assert (await invoke()).result.outcome == "not_found"
            assert callback.calls == 2
            assert await state_owner.load(record.intent) == record
        finally:
            callback.release.set()
            await owner.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("child_cancel", [False, True])
def test_callback_error_is_not_misclassified_as_owner_deadline_or_cancellation(child_cancel):
    async def scenario():
        original = asyncio.CancelledError("child") if child_cancel else TimeoutError("callback")

        class Failing:
            async def reconcile(self, *, context, receipt):
                raise original

        registered, record, request, _ = await _setup(Failing())
        owner = ToolEffectReconciliationOwner()
        with pytest.raises(RuntimeError if child_cancel else TimeoutError) as caught:
            await owner.reconcile(
                request=request, record=record, run_epoch=0, registered=registered
            )
        if child_cancel:
            assert caught.value.__cause__ is original
        else:
            assert caught.value is original
        assert not isinstance(caught.value, ToolEffectReconciliationTimeout)
        assert asyncio.current_task().cancelling() == 0
        assert owner.pending_operations == 0
        assert await owner.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change",
    [
        {"lookup": False, "receipt": None},
        {"lookup": True},
        {"expected_run_epoch": True},
        {"expected_revision": True},
        {"max_steps": True},
        {"max_steps": 257},
        {"max_steps": 0},
        {"task_handoff_id": "handoff"},
        {"tool_name": "other"},
        {"tool_call_id": "other"},
        {"idempotency_key": "other"},
    ],
)
def test_public_request_rejects_ambiguous_authority(change):
    async def scenario():
        _, _, request, _ = await _setup(_Echo())
        with pytest.raises(ValueError):
            ToolEffectReconciliationRequest(**(request.model_dump() | change))

    asyncio.run(scenario())


def test_unsupported_lookup_never_invokes_callback_or_changes_uncertainty():
    async def scenario():
        callback = _Echo()
        registered, record, request, state_owner = await _setup(callback, supports_lookup=False)
        request = request.model_copy(update={"receipt": None, "lookup": True})
        owner = ToolEffectReconciliationOwner()
        accepted = await owner.reconcile(
            request=request,
            record=record,
            run_epoch=0,
            registered=registered,
        )
        assert accepted.result.outcome == "unsupported"
        assert accepted.result.observation == "outcome_unknown"
        assert not accepted.result.retryable
        assert callback.calls == 0
        assert await state_owner.load(record.intent) == record
        assert await owner.aclose()

    asyncio.run(scenario())


def test_reconciler_contract_drift_rejected_before_callback():
    async def scenario():
        callback = _Echo()
        _, record, request, _ = await _setup(callback)
        changed = _register(
            ToolEffectReconciliationRegistration(
                reconciler=callback,
                spec=_spec(receipt_schema_version=2),
            )
        )
        owner = ToolEffectReconciliationOwner()
        with pytest.raises(ToolEffectConflict):
            await owner.reconcile(request=request, record=record, run_epoch=0, registered=changed)
        with pytest.raises(ToolEffectConflict):
            await owner.reconcile(request=request, record=record, run_epoch=0, registered=None)
        assert callback.calls == 0
        assert await owner.aclose()

    asyncio.run(scenario())


def test_precancelled_owner_never_enters_callback():
    async def scenario():
        callback = _Echo()
        registered, record, request, _ = await _setup(callback)
        owner = ToolEffectReconciliationOwner()

        async def invoke():
            asyncio.current_task().cancel()
            return await owner.reconcile(
                request=request,
                record=record,
                run_epoch=0,
                registered=registered,
            )

        task = asyncio.create_task(invoke())
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert task.cancelling() == 1
        assert callback.calls == 0
        assert owner.pending_operations == 0
        assert await owner.aclose()

    asyncio.run(scenario())
