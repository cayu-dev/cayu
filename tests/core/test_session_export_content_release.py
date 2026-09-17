"""Reviewed prose traverses the public source owner and native durable stores."""

from __future__ import annotations

import asyncio
import warnings
from contextlib import asynccontextmanager
from functools import wraps
from hashlib import sha256

import pytest
from tests.core.test_session_exports import (
    AUDIENCE,
    CONTEXT,
    OWNER,
    SECRET,
    _ref,
    harness,
    published,
)
from tests.core.test_session_exports import backend as backend

from cayu.collaboration._contracts import InitiatorBinding
from cayu.collaboration._preparation import contract_bytes
from cayu.collaboration._session_export_store import source_digest
from cayu.collaboration.exports import (
    SessionExportConflict,
    SessionExportDenied,
    SessionExportRequest,
    SessionExportSettlementRequest,
    SessionExportUnavailable,
)
from cayu.collaboration.releases import (
    ContentExposure,
    ContentReleaseExpectation,
    ContentReleaseReader,
    ContentReleaseReceipt,
    ContentReleaseRequest,
    ReleasedContent,
)
from cayu.vaults.redaction import SecretRedactor


def run_async(test):
    @wraps(test)
    def run(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return run


class ReviewOwner(ContentReleaseReader):
    def __init__(self):
        self.approved = {}
        self.lock = asyncio.Lock()
        self.calls = []
        self.revoked = False
        self.replacement = None
        self.entered = asyncio.Event()
        self.resume = None

    @property
    def ref(self):
        return _ref("review-reader")

    @asynccontextmanager
    async def acquire(self, expected):
        async with self.lock:
            self.calls.append(expected)
            key = contract_bytes(expected, redactor=SecretRedactor())
            if self.revoked or key not in self.approved:
                raise SessionExportDenied()
            self.entered.set()
            if self.resume is not None:
                await self.resume.wait()
            yield self.approved[key] if self.replacement is None else self.replacement

    def approve(self, request, text):
        expected = ContentReleaseExpectation(
            request=request.release,
            source_owner=OWNER,
            session_id=request.ref.session_id,
            session_instance_id=request.ref.session_instance_id,
            source_indices=request.source_indices,
            audience=request.audience,
            validator=request.projector,
            policy=request.policy,
        )
        approved = ReleasedContent(
            receipt=ContentReleaseReceipt(
                expected=expected,
                reviewer=InitiatorBinding(
                    issuer=OWNER,
                    principal="human-reviewer",
                    participant=None,
                    mandate=None,
                    invocation_id=None,
                    interaction_id=None,
                ),
                authorization_revision=1,
                expires_at_ms=4102444800000,
            ),
            text=text,
        )
        self.approved[contract_bytes(expected, redactor=SecretRedactor())] = approved
        return approved


async def reviewed_request(case, app, store, owner, *, text="Reviewed advice."):
    base = await case.request(app)
    rows = await store.load_transcript_window(case.session_id, start_index=0, limit=1)
    request = SessionExportRequest.model_validate(
        base.model_copy(
            update={
                "mode": "reviewed_prose",
                "projector": owner.ref,
                "release": ContentReleaseRequest(
                    decision=_ref("review-decision"),
                    source_commitment=source_digest(rows.records),
                    text_commitment=sha256(text.encode()).hexdigest(),
                    exposure=(),
                ),
            }
        )
    )
    approval = owner.approve(request, text)
    return request, approval


@run_async
async def test_reviewed_export_replay_read_retire_without_projection(backend):
    async with harness(backend) as case:
        owner = ReviewOwner()
        app, store, _, projector = case.app(release_readers=(owner,))
        await case.create(store)
        request, approval = await reviewed_request(case, app, store, owner)
        receipt = await app.export_session(request, context=CONTEXT)
        assert receipt.expected.intent.release_receipt == approval.receipt
        assert approval.text not in receipt.model_dump_json()
        assert projector.calls == 0
        assert await app.read_session_export(request, context=CONTEXT) == {"text": approval.text}
        assert await app.export_session(request, context=CONTEXT) == receipt
        assert len(await published(store, case.session_id)) == 1
        # Historical replay does not need the former review implementation.
        other, _, _, _ = case.app(projectors=())
        assert await other.export_session(request, context=CONTEXT) == receipt
        with pytest.raises(SessionExportUnavailable):
            await other.read_session_export(request, context=CONTEXT)
        settlement = SessionExportSettlementRequest(
            request=request,
            operation=request.ref.operation.model_copy(update={"caller_key": "retire"}),
            mode="retire",
        )
        await app.settle_session_export(settlement, context=CONTEXT)
        with pytest.raises(SessionExportUnavailable):
            await app.read_session_export(request, context=CONTEXT)
        assert (await app.lookup_session_export(request, context=CONTEXT)).receipt == receipt


@pytest.mark.parametrize("change", ["byte", "audience", "source", "exposure", "decision"])
@run_async
async def test_changed_release_proposal_is_not_an_approval(backend, change):
    async with harness(backend) as case:
        owner = ReviewOwner()
        app, store, _, projector = case.app(release_readers=(owner,))
        await case.create(store)
        request, _ = await reviewed_request(case, app, store, owner)
        release = request.release
        if change == "byte":
            release = release.model_copy(
                update={"text_commitment": sha256(b"Reviewed advice!").hexdigest()}
            )
        elif change == "source":
            release = release.model_copy(update={"source_commitment": "f" * 64})
        elif change == "exposure":
            release = release.model_copy(
                update={
                    "exposure": (
                        ContentExposure(source=_ref("other"), channel="tool", commitment="a" * 64),
                    )
                }
            )
        elif change == "decision":
            release = release.model_copy(update={"decision": _ref("different-decision")})
        else:
            request = request.model_copy(
                update={"audience": AUDIENCE.model_copy(update={"incarnation": "two"})}
            )
        request = request.model_copy(update={"release": release})
        with pytest.raises((SessionExportDenied, SessionExportConflict)):
            await app.export_session(request, context=CONTEXT)
        assert not await published(store, case.session_id)
        assert projector.calls == 0


@run_async
async def test_release_owner_cannot_return_other_bytes_and_revocation_gates_reads(backend):
    async with harness(backend) as case:
        owner = ReviewOwner()
        app, store, _, _ = case.app(release_readers=(owner,))
        await case.create(store)
        request, approval = await reviewed_request(case, app, store, owner)
        owner.replacement = approval.model_copy(update={"text": "Unreviewed advice."})
        with pytest.raises(SessionExportDenied):
            await app.export_session(request, context=CONTEXT)
        assert not await published(store, case.session_id)
        owner.replacement = None
        receipt = await app.export_session(request, context=CONTEXT)
        owner.revoked = True
        with pytest.raises(SessionExportDenied):
            await app.read_session_export(request, context=CONTEXT)
        assert (await app.lookup_session_export(request, context=CONTEXT)).receipt == receipt


@run_async
async def test_release_expiry_rechecked_at_store_boundary(backend):
    async with harness(backend) as case:
        owner = ReviewOwner()
        app, store, _, _ = case.app(release_readers=(owner,))
        await case.create(store)
        request, approval = await reviewed_request(case, app, store, owner)
        owner.replacement = approval.model_copy(
            update={"receipt": approval.receipt.model_copy(update={"expires_at_ms": 1})}
        )
        with pytest.raises(SessionExportDenied):
            await app.export_session(request, context=CONTEXT)
        assert not await published(store, case.session_id)


@pytest.mark.parametrize("signal", ["cancel", "timeout"])
@run_async
async def test_actual_cancellation_retains_release_guard_and_publication(backend, signal):
    async with harness(backend) as case:
        owner = ReviewOwner()
        owner.resume = asyncio.Event()
        app, store, _, _ = case.app(release_readers=(owner,))
        await case.create(store)
        request, _ = await reviewed_request(case, app, store, owner)
        deadline = asyncio.timeout(None)

        async def observe():
            async with deadline:
                return await app.export_session(request, context=CONTEXT)

        task = asyncio.create_task(observe())
        await asyncio.wait_for(owner.entered.wait(), 5)
        if signal == "cancel":
            task.cancel(SECRET)
            with pytest.raises(asyncio.CancelledError) as error:
                await task
            assert task.cancelled() and task.cancelling() == 1
        else:
            deadline.reschedule(asyncio.get_running_loop().time())
            with pytest.raises(TimeoutError) as error:
                await task
            assert not task.cancelled() and task.cancelling() == 0
        assert SECRET not in str(error.value)
        assert owner.lock.locked()
        owner.resume.set()
        receipt = await app.export_session(request, context=CONTEXT)
        assert receipt.expected.intent.request == request
        assert not owner.lock.locked()
        assert len(await published(store, case.session_id)) == 1


@pytest.mark.parametrize("field", ["text", "reviewer"])
@pytest.mark.parametrize("malformed", [False, True])
@run_async
async def test_review_owner_rejection_has_no_diagnostic_secret_channel(
    backend, field, malformed, caplog, capsys
):
    class Hostile:
        def __repr__(self):
            return SECRET

        def __str__(self):
            return SECRET

    async with harness(backend) as case:
        owner = ReviewOwner()
        app, store, _, _ = case.app(release_readers=(owner,), redactor=SecretRedactor(SECRET))
        await case.create(store)
        request, approval = await reviewed_request(case, app, store, owner)
        damaged = approval.model_copy(deep=True)
        value = Hostile() if malformed else SECRET
        if field == "text":
            object.__setattr__(damaged, "text", value)
        else:
            object.__setattr__(damaged.receipt.reviewer, "principal", value)
        owner.replacement = damaged
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            with pytest.raises(SessionExportUnavailable) as caught:
                await app.export_session(request, context=CONTEXT)
        captured = capsys.readouterr()
        signals = [caught.value]
        seen = set()
        while signals:
            signal = signals.pop()
            if id(signal) in seen:
                continue
            seen.add(id(signal))
            assert SECRET not in str(signal) + repr(signal)
            signals.extend(
                item for item in (signal.__cause__, signal.__context__) if item is not None
            )
            if isinstance(signal, BaseExceptionGroup):
                signals.extend(signal.exceptions)
        assert SECRET not in str(observed) + caplog.text + captured.out + captured.err
        assert not await published(store, case.session_id)
        assert (await app.lookup_session_export(request, context=CONTEXT)).status == "not_found"
