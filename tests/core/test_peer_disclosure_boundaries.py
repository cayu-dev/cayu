"""Public HTTP disclosure and whole-attempt revocation guard qualification."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_peer_content import _exposure_id
from tests.core.test_peer_content_lifecycle import LifecyclePolicy, journey, peer_count

from cayu.collaboration.exports import SessionExportDenied, SessionExportPolicy
from cayu.events import EventType
from cayu.messages import Message
from cayu.sessions.base import ResumeRequest, RunRequest


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_inputs_reject_identical_delivered_peer_without_mutation(
    backend, tmp_path, request
):
    async with journey(backend, tmp_path, request, "ordinary") as state:
        current, store, provider, _, _, target, delivery, stream = state
        accepted = await current.append_peer_content(await delivery(), context=CONTEXT)
        assert accepted.status == "appended"
        events = [event async for event in stream]
        assert any(event.type == EventType.SESSION_COMPLETED for event in events)
        transcript = await store.load_transcript(target.id)
        assert peer_count(transcript) == 1
        original = next(m for m in transcript if any(p.type == "peer_content" for p in m.content))
        # Durable reconstruction remains valid; it is not caller insertion authority.
        reconstructed = Message.model_validate_json(original.model_dump_json())
        assert reconstructed == original
        before_session = await store.load(target.id)
        before_events = await store.load_events(target.id)
        before_dispatches = len(provider.requests)
        for request_type in (RunRequest, ResumeRequest):
            arguments = {"session_id": target.id}
            if request_type is RunRequest:
                arguments["agent_name"] = "reviewer"
            with pytest.raises(ValueError, match="authenticated peer delivery"):
                request_type(**arguments, messages=[reconstructed])
            mutated = request_type(**arguments, messages=[Message.text("user", "valid")])
            mutated.messages = [reconstructed]
            with pytest.raises(ValueError, match="authenticated peer delivery"):
                if request_type is RunRequest:
                    async for _ in current.run(mutated):
                        pass
                else:
                    async for _ in current.resume(mutated, context=CONTEXT):
                        pass
            assert await store.load(target.id) == before_session
            assert await store.load_transcript(target.id) == transcript
            assert await store.load_events(target.id) == before_events
            assert len(provider.requests) == before_dispatches


@pytest.mark.anyio
@pytest.mark.postgres
async def test_postgres_exclusion_does_not_use_retired_connection(tmp_path, request, monkeypatch):
    async with journey("postgres", tmp_path, request, "ordinary") as state:
        _, store, _, _, _, _, delivery, stream = state
        expected = await delivery()
        connection = store._connection

        @asynccontextmanager
        async def retire_after_transaction():
            async with connection() as owned:
                yield owned
                # Successful commit followed by retirement before pool return.
                # An additional out-of-context commit must never touch this connection.
                await owned.commit()
                await owned.close()

        with monkeypatch.context() as patch:
            patch.setattr(store, "_connection", retire_after_transaction)
            receipt = await store.exclude_peer_content(expected, reason="withdrawn")
        assert receipt.status == "excluded"
        assert await store.read_peer_content(expected.append_key) == receipt
        replay = await store.exclude_peer_content(expected, reason="withdrawn")
        assert replay.replayed and replay.status == "excluded"
        await stream.aclose()


def test_generic_peer_projection_never_serializes_private_authority():
    pytest.importorskip("fastapi")
    pytest.importorskip("sse_starlette")
    from cayu.server.routes import _serialize_message_part

    class PrivatePeer:
        type = "peer_content"

        def model_dump(self, **kwargs):
            raise AssertionError("Private append/creation permit must not be serialized")

    assert _serialize_message_part(None, PrivatePeer()) == {
        "type": "peer_content",
        "disclosure": "withheld",
    }


class LockedPolicy(LifecyclePolicy):
    def __init__(self):
        super().__init__()
        self.lock = asyncio.Lock()
        self.waiting = asyncio.Event()
        self.acquired = asyncio.Event()
        self.single_calls = 0
        self.batches = []
        self.invalid_projection = False

    @asynccontextmanager
    async def acquire_peer_exposure(self, context, **kwargs):
        # This valid single-item implementation cannot be entered recursively.
        self.single_calls += 1
        async with self.lock, super().acquire_peer_exposure(context, **kwargs) as projection:
            yield projection

    @asynccontextmanager
    async def acquire_peer_exposures(self, context, *, items):
        self.items = items
        self.waiting.set()
        async with self.lock:
            self.active += 1
            try:
                self.acquired.set()
                if self.block:
                    await self.release.wait()
                if self.revoked:
                    raise SessionExportDenied()
                assert len(items) == 2
                assert len({item.origin.model_attempt_id for item in items}) == 1
                assert len({item.occurrence.occurrence_id for item in items}) == 2
                assert all(
                    item.origin.append_key.occurrence_id == item.occurrence.occurrence_id
                    for item in items
                )
                assert all(
                    item.occurrence.producer_receipt_id in self.allowed_receipts for item in items
                )
                self.calls.extend(item.single_arguments() for item in items)
                self.batches.append(items)
                yield (
                    ()
                    if self.invalid_projection
                    else tuple(item.occurrence.payload for item in items)
                )
            finally:
                self.active -= 1


class SingleOnlyPolicy(LockedPolicy):
    @asynccontextmanager
    async def acquire_peer_exposures(self, context, *, items):
        self.items = items
        async with SessionExportPolicy.acquire_peer_exposures(
            self, context, items=items
        ) as projections:
            yield projections


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "case",
    [
        "success",
        "revoke_waits",
        "revoked",
        "cancel_waiting",
        "cancel_acquired",
        "cancel_serialized",
        "unsupported",
        "invalid_projection",
    ],
)
async def test_public_peer_batch_holds_one_nonreentrant_guard(backend, case, tmp_path, request):
    policy_factory = SingleOnlyPolicy if case == "unsupported" else LockedPolicy
    async with journey(backend, tmp_path, request, "batch", policy_factory=policy_factory) as state:
        current, store, provider, policy, _, target, delivery, stream = state
        requests = [await delivery(), await delivery()]
        for value in requests:
            assert (await current.append_peer_content(value, context=CONTEXT)).status == "appended"
        events = []
        policy.invalid_projection = case == "invalid_projection"
        serialized = asyncio.Event()
        if case in {"revoked", "cancel_waiting"}:
            await policy.lock.acquire()
        if case in {"cancel_acquired", "revoke_waits"}:
            policy.block = True
        revoker = None

        async def consume():
            try:
                async for event in stream:
                    events.append(event)
                    if case == "cancel_serialized" and event.type == EventType.MODEL_TEXT_DELTA:
                        serialized.set()
                        await asyncio.Future()
            finally:
                await stream.aclose()

        task = asyncio.create_task(consume())
        try:
            async with asyncio.timeout(30):
                if case in {"revoked", "cancel_waiting"}:
                    await policy.waiting.wait()
                elif case in {"cancel_acquired", "revoke_waits"}:
                    await policy.acquired.wait()
                elif case == "cancel_serialized":
                    await serialized.wait()
                if case == "revoked":
                    policy.revoked = True
                    policy.lock.release()
                if case == "revoke_waits":
                    revocation_waiting = asyncio.Event()

                    async def revoke():
                        revocation_waiting.set()
                        async with policy.lock:
                            assert len(provider.requests) == 1
                            policy.revoked = True

                    revoker = asyncio.create_task(revoke())
                    await revocation_waiting.wait()
                    assert not revoker.done()
                    policy.release.set()
                if case.startswith("cancel"):
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert task.cancelled() and task.cancelling() == 1
                    if case == "cancel_waiting":
                        policy.lock.release()
                else:
                    await task
                if revoker is not None:
                    await revoker
            assert not policy.lock.locked() and policy.active == 0
            assert policy.single_calls == 0
            if case in {"success", "revoke_waits"}:
                assert any(event.type == EventType.SESSION_COMPLETED for event in events)
                assert (
                    len(provider.requests) == 1 and peer_count(provider.requests[0].messages) == 2
                )
                assert len(policy.batches) == 1
            elif case != "cancel_serialized":
                assert not provider.requests
            attempt = policy.items[0].origin.model_attempt_id
            expected_outcome = (
                "not_exposed"
                if case in {"revoked", "unsupported", "invalid_projection"}
                else "pending"
                if case in {"cancel_waiting", "cancel_acquired"}
                else "exposed"
            )
            for value in requests:
                evidence = await store.read_peer_content_exposure(
                    value.append_key, _exposure_id(value, attempt)
                )
                assert evidence.outcome == expected_outcome
            assert peer_count(await store.load_transcript(target.id)) == 2
        finally:
            policy.release.set()
            if case in {"revoked", "cancel_waiting"} and policy.lock.locked() and not policy.active:
                policy.lock.release()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if revoker is not None:
                if not revoker.done():
                    revoker.cancel()
                await asyncio.gather(revoker, return_exceptions=True)
            await stream.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_http_transcript_withholds_delivered_peer_content(backend, tmp_path, request):
    pytest.importorskip("fastapi")
    pytest.importorskip("sse_starlette")
    from cayu.server import AuthContext, ServerConfig, create_server

    async with journey(backend, tmp_path, request, "http") as state:
        current, store, provider, policy, _, target, delivery, stream = state
        expected = await delivery()
        assert (await current.append_peer_content(expected, context=CONTEXT)).status == "appended"
        events = [event async for event in stream]
        assert any(event.type == EventType.SESSION_COMPLETED for event in events)
        assert peer_count(await store.load_transcript(target.id)) == 1
        positive = await current.read_peer_content(expected.append_key, context=CONTEXT)
        assert positive.occurrence == expected.occurrence

        async def authenticate(request):
            return AuthContext(subject="generic-transcript-reader")

        server = create_server(current, config=ServerConfig.protected(authenticate))
        async with AsyncClient(
            transport=ASGITransport(app=server), base_url="http://local"
        ) as client:
            for revoked in (False, True):
                policy.revoked = revoked
                response = await client.get(
                    f"/api/sessions/{current.project_session_id_for_exposure(target.id)}/transcript"
                )
                assert response.status_code == 200, response.text
                parts = [
                    part for message in response.json()["messages"] for part in message["content"]
                ]
                peers = [part for part in parts if part["type"] == "peer_content"]
                assert peers == [{"type": "peer_content", "disclosure": "withheld"}]
                for private in (
                    expected.occurrence.payload.text,
                    expected.operation_key,
                    "append_key_json",
                    "creation_target",
                    "provenance_sha256",
                ):
                    assert private not in response.text
            denied = await current.read_peer_content(expected.append_key, context=CONTEXT)
            assert (
                denied.status == "appended"
                and denied.occurrence is None
                and denied.disclosure == "withheld"
            )
        assert len(provider.requests) == 1
