"""A deadline between reads must retain the already-reserved dispatch slot."""

import asyncio

import pytest

from cayu import Message, ModelStreamEvent, ScriptedModelProvider
from cayu.providers import ModelRequest, ModelStreamDeadlineError
from cayu.providers import _credential_boundary as boundary
from cayu.providers import deadlines as ds
from cayu.providers._stream_cleanup import _LocalHttpCleanupObserver


def _evidence():
    return ds.ProviderStreamDeadlineEvidence(
        deadline_kind=ds.ProviderDeadlineKind.SEMANTIC_IDLE,
        configured_timeout_s=1,
        elapsed_s=1,
        last_progress_kind=None,
        last_progress_elapsed_s=None,
        last_progress_at=None,
    )


@pytest.mark.parametrize("layer", ["raw", "model"])
@pytest.mark.parametrize("blocked", ["close", "receipt"])
@pytest.mark.parametrize("pending_read_callback", [False, True])
def test_between_read_deadline_retains_dispatch_until_cleanup_settles(
    monkeypatch, layer, blocked, pending_read_callback
):
    async def run():
        initial_tasks = set(asyncio.all_tasks())
        owners_before = set(ds._PROVIDER_DEADLINE_AWAIT_OWNERS)
        policy = ds.ProviderStreamDeadlines(max_concurrent_streams=len(owners_before) + 1)
        controller = ds.ProviderStreamDeadlineController(policy)
        owner = controller._await_ownership
        release = asyncio.Event()
        started = asyncio.Event()
        completed = asyncio.Event()
        physical_close = asyncio.Event()
        receipts = []
        evidence = _evidence()

        async def publish(received, succeeded):
            started.set()
            await release.wait()
            receipts.append((received, succeeded))

        observer = _LocalHttpCleanupObserver(publish)
        observer.expired(evidence)
        assert observer.confirm_expiry() is None

        class Stream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise AssertionError("Expired dispatch must not start another read")

            async def aclose(self):
                if blocked == "close":
                    started.set()
                    await release.wait()
                physical_close.set()
                if blocked == "receipt":
                    await observer.closed(succeeded=True)
                completed.set()

        # Cleanup must use the slot this dispatch already owns even when the
        # legacy fallback registry cannot accept any more cleanup work.
        monkeypatch.setattr(boundary, "_MAX_OWNED_PROVIDER_STREAM_DEADLINE_CLEANUPS", 0)
        failure = (
            ds.ProviderStreamDeadlineExceeded(evidence)
            if layer == "raw"
            else ModelStreamDeadlineError(provider="synthetic", evidence=evidence)
        )
        try:
            with pytest.raises(type(failure)) as raised:
                async with boundary.aclosing_provider_stream(
                    Stream(),
                    pending_read=(lambda: None) if pending_read_callback else None,
                    retain_cleanup=controller.retain_dispatched_operation,
                ):
                    raise failure
            controller.close()
            received_evidence = (
                raised.value.evidence if layer == "raw" else raised.value.deadline_evidence
            )
            assert received_evidence == evidence
            assert raised.value.stream_cleanup_failed is True
            assert started.is_set()
            assert not completed.is_set()
            assert physical_close.is_set() is (blocked == "receipt")
            assert owner in ds._PROVIDER_DEADLINE_AWAIT_OWNERS
            with pytest.raises(RuntimeError, match="capacity is exhausted"):
                ds.ProviderStreamDeadlineController(policy)
            release.set()
            await asyncio.wait_for(completed.wait(), 2)
            async with asyncio.timeout(2):
                while owner in ds._PROVIDER_DEADLINE_AWAIT_OWNERS:
                    await asyncio.sleep(0)
            assert receipts == ([(evidence, True)] if blocked == "receipt" else [])
            replacement = ds.ProviderStreamDeadlineController(policy)
            replacement.close()
        finally:
            release.set()
            controller.close()
            async with asyncio.timeout(2):
                while owner in ds._PROVIDER_DEADLINE_AWAIT_OWNERS:
                    await asyncio.sleep(0)
            assert set(asyncio.all_tasks()) <= initial_tasks

    asyncio.run(run())


@pytest.mark.parametrize("cancel_caller", [False, True])
@pytest.mark.parametrize("close_fails", [False, True])
def test_between_read_cleanup_survives_caller_exit(monkeypatch, cancel_caller, close_fails):
    async def run():
        owners_before = set(ds._PROVIDER_DEADLINE_AWAIT_OWNERS)
        controller = ds.ProviderStreamDeadlineController(
            ds.ProviderStreamDeadlines(max_concurrent_streams=len(owners_before) + 1)
        )
        owner = controller._await_ownership
        release = asyncio.Event()
        started = asyncio.Event()
        settled = asyncio.Event()
        close_cancelled = False

        class Stream:
            async def aclose(self):
                nonlocal close_cancelled
                started.set()
                if cancel_caller:
                    asyncio.get_running_loop().call_soon(consumer.cancel)
                try:
                    await release.wait()
                    if close_fails:
                        raise RuntimeError("synthetic late close failure")
                except asyncio.CancelledError:
                    close_cancelled = True
                    raise
                finally:
                    settled.set()

        async def consume():
            try:
                async with boundary.aclosing_provider_stream(
                    Stream(), retain_cleanup=controller.retain_dispatched_operation
                ):
                    raise ds.ProviderStreamDeadlineExceeded(_evidence())
            finally:
                controller.close()

        monkeypatch.setattr(boundary, "_MAX_OWNED_PROVIDER_STREAM_DEADLINE_CLEANUPS", 0)
        consumer = asyncio.create_task(consume())
        try:
            expected = (
                asyncio.CancelledError if cancel_caller else ds.ProviderStreamDeadlineExceeded
            )
            with pytest.raises(expected):
                await consumer
            assert started.is_set()
            assert not settled.is_set()
            assert owner in ds._PROVIDER_DEADLINE_AWAIT_OWNERS
        finally:
            release.set()
            controller.close()
            async with asyncio.timeout(2):
                while owner in ds._PROVIDER_DEADLINE_AWAIT_OWNERS:
                    await asyncio.sleep(0)
        assert settled.is_set()
        assert not close_cancelled

    asyncio.run(run())


def test_runtime_deadline_between_normalized_reads_keeps_close_owned(monkeypatch):
    async def run():
        owners_before = set(ds._PROVIDER_DEADLINE_AWAIT_OWNERS)
        started = asyncio.Event()
        release = asyncio.Event()
        closed = asyncio.Event()
        reads = 0
        expire = False
        deadline_at = ds.ProviderStreamDeadlineController._deadline_at

        def expired_between_reads(self, kind):
            if expire and kind is ds.ProviderDeadlineKind.SEMANTIC_IDLE:
                return self._loop.time() - 1
            return deadline_at(self, kind)

        monkeypatch.setattr(
            ds.ProviderStreamDeadlineController, "_deadline_at", expired_between_reads
        )

        class Stream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                nonlocal reads
                reads += 1
                return ModelStreamEvent.text_delta("synthetic progress")

            async def aclose(self):
                started.set()
                await release.wait()
                closed.set()

        class Provider(ScriptedModelProvider):
            @property
            def stream_deadlines(self):
                return ds.ProviderStreamDeadlines(max_concurrent_streams=len(owners_before) + 1)

            def stream(self, request):
                return Stream()

        provider = Provider([])
        iterator = provider.runtime_stream(
            ModelRequest(model="synthetic", messages=[Message.text("user", "test")])
        )
        try:
            assert (await anext(iterator)).delta == "synthetic progress"
            expire = True
            with pytest.raises(ModelStreamDeadlineError):
                await anext(iterator)
            assert reads == 1
            assert started.is_set()
            assert not closed.is_set()
            assert ds._PROVIDER_DEADLINE_AWAIT_OWNERS - owners_before
        finally:
            release.set()
            await iterator.aclose()
            await asyncio.wait_for(closed.wait(), 2)
            async with asyncio.timeout(2):
                while ds._PROVIDER_DEADLINE_AWAIT_OWNERS - owners_before:
                    await asyncio.sleep(0)

    asyncio.run(run())
