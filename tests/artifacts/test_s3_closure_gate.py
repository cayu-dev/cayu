import asyncio
import io
import json
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from threading import Barrier, Event, Lock

import pytest
from tests.artifacts.test_aws_s3 import _S3Client

from cayu.artifacts._closure import ArtifactClosureClaim
from cayu.artifacts._s3_closure import S3ArtifactClosureGate


def test_publication_gate_preserves_long_artifact_session_identity():
    from cayu import S3ArtifactStore

    async def run():
        client = _S3Client()
        store = S3ArtifactStore("bucket", client=client)
        session_id = "session-" + "s" * 100_000
        artifact = await store.put_bytes(b"owned", filename="item", session_id=session_id)
        reopened = S3ArtifactStore("bucket", client=client)
        assert (await reopened.read_bytes(artifact.id)).content == b"owned"
        assert (await reopened.list(session_id=session_id)).artifacts == (artifact,)
        with pytest.raises(ValueError, match="closure identity"):
            await reopened.load_session_closure_claim(session_id)
        with pytest.raises(ValueError, match="closure identity"):
            await reopened.claim_session_closure(
                session_id, "f" * 64, max_records=10, max_bytes=1_000_000
            )
        assert (await reopened.read_bytes(artifact.id)).content == b"owned"

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["partial_delete", "lost_acknowledgement"])
def test_public_s3_closure_resumes_claimed_deletion_after_restart(tmp_path, failure):
    from tests.core.session_closure_conformance import create_closure_session

    from cayu import CayuApp, S3ArtifactStore
    from cayu.runtime.session_closure import ArtifactSessionClosureStore, SessionClosurePolicy
    from cayu.storage.sqlite import SQLiteSessionStore

    class Client(_S3Client):
        lose_ack = False

        def delete_objects(self, **kwargs):
            result = super().delete_objects(**kwargs)
            if self.lose_ack:
                self.lose_ack = False
                raise OSError("lost deletion acknowledgement")
            return result

    async def run():
        client = Client()
        artifacts = S3ArtifactStore("bucket", client=client)
        artifact = await artifacts.put_bytes(b"owned", filename="item", session_id="session")
        path = tmp_path / "sessions.sqlite"
        sessions = SQLiteSessionStore(path)
        try:
            await create_closure_session(sessions, "session")
            app = CayuApp(
                session_store=sessions,
                session_closure_stores=(ArtifactSessionClosureStore(artifacts),),
            )
            if failure == "partial_delete":
                client.delete_errors_by_suffix["/content"] = "AccessDenied"
            else:
                client.lose_ack = True
            first = await app.erase_session_closure("session")
            assert not first.complete
            assert await sessions.load("session") is not None
            claim = await artifacts.load_session_closure_claim("session")
            assert claim is not None and claim.plan_id == first.plan_id
            assert claim.artifacts[0].artifact_id == artifact.id
        finally:
            await sessions.close()

        client.delete_errors_by_suffix.clear()
        sessions = SQLiteSessionStore(path)
        try:
            artifacts = S3ArtifactStore("bucket", client=client)
            adapter = ArtifactSessionClosureStore(artifacts)
            before = len(client.delete_calls)
            with pytest.raises(ValueError, match="authority conflicts"):
                await adapter.claim_session_closure("session", SessionClosurePolicy(), "f" * 64)
            assert len(client.delete_calls) == before
            app = CayuApp(session_store=sessions, session_closure_stores=(adapter,))
            manifest = await app.inspect_session_closure("session")
            cleanup = next(r for r in manifest.records if r.store_id == adapter.store_id)
            assert cleanup.record_class == "artifact_cleanup_claim"
            assert cleanup.count == 1
            assert "not live artifacts" in cleanup.detail
            second = await app.erase_session_closure("session", expected_plan_id=first.plan_id)
            assert second.complete
            assert await sessions.load("session") is None
            assert not any(
                key.endswith(("/content", "/metadata.json")) for _, key in client.objects
            )
            assert await artifacts.load_session_closure_claim("session") == claim
        finally:
            await sessions.close()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["retirement", "settlement_acknowledgement"])
def test_reopened_public_closure_recovers_only_durable_write_settlement(failure):
    from tests.core.session_closure_conformance import create_closure_session

    from cayu import CayuApp, S3ArtifactStore
    from cayu.runtime.session_closure import ArtifactSessionClosureStore

    class Client(_S3Client):
        fail_readback = False
        injected = False

        def get_object(self, **kwargs):
            if self.fail_readback:
                self.fail_readback = False
                raise OSError("settlement readback unavailable")
            return super().get_object(**kwargs)

        def put_object(self, **kwargs):
            if "/_closure/" in kwargs["Key"] and "/artifact-owners/" not in kwargs["Key"]:
                state = json.loads(kwargs["Body"])
                previous = self.objects.get((kwargs["Bucket"], kwargs["Key"]))
                previous = {} if previous is None else json.loads(previous)
                if (
                    not self.injected
                    and failure == "retirement"
                    and previous.get("settled")
                    and not state["settled"]
                ):
                    self.injected = True
                    raise OSError("settlement retirement unavailable")
                result = super().put_object(**kwargs)
                if (
                    not self.injected
                    and failure == "settlement_acknowledgement"
                    and state["settled"]
                ):
                    self.injected = True
                    self.fail_readback = True
                    raise OSError("settlement acknowledgement lost")
                return result
            return super().put_object(**kwargs)

    async def run():
        client = Client()
        store = S3ArtifactStore("bucket", client=client)
        app = CayuApp(session_closure_stores=(ArtifactSessionClosureStore(store),))
        await create_closure_session(app.session_store, "session")
        with pytest.raises(Exception):
            await store.put_bytes(b"settled", filename="owned.txt", session_id="session")
        assert client.injected
        state, _ = store._closure_gate(client, "session").read()
        assert not state["active"] and len(state["settled"]) == 1
        writes = len(
            [
                call
                for call in client.put_calls
                if call["Key"].endswith(("/content", "/metadata.json"))
            ]
        )
        reopened = S3ArtifactStore("bucket", client=client)
        recovered = CayuApp(
            session_store=app.session_store,
            session_closure_stores=(ArtifactSessionClosureStore(reopened),),
        )
        result = await recovered.erase_session_closure("session")
        assert result.complete
        assert (
            len(
                [
                    call
                    for call in client.put_calls
                    if call["Key"].endswith(("/content", "/metadata.json"))
                ]
            )
            == writes
        )
        final, _ = reopened._closure_gate(client, "session").read()
        assert final["active"] == final["settled"] == {}
        assert final["claim"] is not None

    asyncio.run(run())


def test_complete_s3_objects_do_not_prove_publication_owner_has_settled():
    from cayu import S3ArtifactStore

    committed, release = Event(), Event()

    class Client(_S3Client):
        def put_object(self, **kwargs):
            result = super().put_object(**kwargs)
            if kwargs["Key"].endswith("/metadata.json"):
                committed.set()
                assert release.wait(5)
            return result

    async def run():
        client = Client()
        first = S3ArtifactStore("bucket", client=client)
        writer = asyncio.create_task(
            first.put_bytes(b"committed", filename="owned.txt", session_id="session")
        )
        try:
            assert await asyncio.to_thread(committed.wait, 5)
            reopened = S3ArtifactStore("bucket", client=client)
            listing = await reopened.list(session_id="session")
            assert listing.total_count == 1
            assert (await reopened.read_bytes(listing.artifacts[0].id)).content == b"committed"
            writer.cancel("cancel after remote commit")
            assert writer.cancelling() == 1
            with pytest.raises(ValueError, match="quiesced"):
                await reopened.claim_session_closure(
                    "session", "a" * 64, max_records=10, max_bytes=10000
                )
            assert client.delete_calls == []
            assert not writer.done()
            release.set()
            with pytest.raises(asyncio.CancelledError, match="cancel after remote commit"):
                await writer
            assert writer.cancelled() and writer.cancelling() == 1
            claim = await reopened.claim_session_closure(
                "session", "a" * 64, max_records=10, max_bytes=10000
            )
            assert len(claim.artifacts) == 1
        finally:
            release.set()
            await asyncio.gather(writer, return_exceptions=True)

    asyncio.run(run())


def test_s3_cancellation_retains_reservation_until_dispatched_upload_settles():
    from cayu import S3ArtifactStore

    dispatched, release = Event(), Event()

    class Client(_S3Client):
        def put_object(self, **kwargs):
            if kwargs["Key"].endswith("/content"):
                dispatched.set()
                assert release.wait(5)
            return super().put_object(**kwargs)

    async def run():
        client = Client()
        first = S3ArtifactStore("bucket", client=client)
        other = S3ArtifactStore("bucket", client=client)
        writer = asyncio.create_task(
            first.put_bytes(b"in-flight", filename="owned.txt", session_id="session")
        )
        try:
            assert await asyncio.to_thread(dispatched.wait, 5)
            writer.cancel("cancel upload owner")
            assert writer.cancelling() == 1
            with pytest.raises(ValueError, match="quiesced"):
                await other.claim_session_closure(
                    "session", "a" * 64, max_records=10, max_bytes=10000
                )
            assert client.delete_calls == []
            release.set()
            with pytest.raises(asyncio.CancelledError, match="cancel upload owner"):
                await writer
            assert writer.cancelled() and writer.cancelling() == 1
            claim = await other.claim_session_closure(
                "session", "a" * 64, max_records=10, max_bytes=10000
            )
            assert len(claim.artifacts) == 1
            await other.delete_session_closure_artifact(claim, claim.artifacts[0].artifact_id)
            assert not any(
                key.endswith(("/content", "/metadata.json")) for _, key in client.objects
            )
        finally:
            release.set()
            await asyncio.gather(writer, return_exceptions=True)

    asyncio.run(run())


def test_public_s3_closure_seals_before_delete_and_replays():
    from tests.core.session_closure_conformance import create_closure_session

    from cayu import CayuApp, S3ArtifactStore
    from cayu.runtime.session_closure import ArtifactSessionClosureStore

    async def run():
        client = _S3Client()
        store = S3ArtifactStore("bucket", client=client)
        app = CayuApp(session_closure_stores=(ArtifactSessionClosureStore(store),))
        await create_closure_session(app.session_store, "session")
        artifact = await store.put_bytes(b"owned", filename="owned.txt", session_id="session")
        report = await app.erase_session_closure("session")
        assert report.complete
        reopened = S3ArtifactStore("bucket", client=client)
        claim = await reopened.load_session_closure_claim("session")
        assert claim is not None
        assert [item.artifact_id for item in claim.artifacts] == [artifact.id]
        assert (
            await reopened.claim_session_closure(
                "session", report.plan_id, max_records=10, max_bytes=10000
            )
            == claim
        )
        await reopened.delete_session_closure_artifact(claim, artifact.id)
        with pytest.raises(ValueError, match="fenced"):
            await reopened.put_bytes(b"late", filename="late.txt", session_id="session")
        with pytest.raises(ValueError, match="different content or metadata"):
            await reopened.put_bytes(
                b"replacement", artifact_id=artifact.id, filename="new.txt", session_id="other"
            )
        assert not any(key.endswith(("/content", "/metadata.json")) for _, key in client.objects)
        other_claim = await reopened.claim_session_closure(
            "other", "e" * 64, max_records=10, max_bytes=10000
        )
        assert other_claim.artifacts == ()

    asyncio.run(run())


def test_s3_uncertain_publication_reservation_blocks_closure_after_reopen():
    from cayu import S3ArtifactStore

    async def run():
        client = _S3Client()
        client.fail_put_suffix = "metadata.json"
        store = S3ArtifactStore("bucket", client=client)
        with pytest.raises(Exception):
            await store.put_bytes(b"partial", filename="partial.txt", session_id="session")
        assert any(key.endswith("/content") for _, key in client.objects)
        reopened = S3ArtifactStore("bucket", client=client)
        with pytest.raises(ValueError, match="quiesced"):
            await reopened.claim_session_closure(
                "session", "a" * 64, max_records=10, max_bytes=10000
            )
        assert client.delete_calls == []
        assert await reopened.load_session_closure_claim("session") is None

    asyncio.run(run())


class _Error(RuntimeError):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


class _ConditionalClient:
    def __init__(self):
        self.lock = Lock()
        self.data = None
        self.barrier = None
        self.put_calls = 0
        self.lose_ack = False

    def get_object(self, **kwargs):
        with self.lock:
            if self.data is None:
                raise _Error("NoSuchKey")
            return {
                "Body": io.BytesIO(self.data),
                "ContentLength": len(self.data),
                "ETag": sha256(self.data).hexdigest(),
            }

    def put_object(self, **kwargs):
        with self.lock:
            self.put_calls += 1
            wait = self.barrier if self.put_calls <= 2 else None
        if wait is not None:
            wait.wait(timeout=5)
        with self.lock:
            if kwargs.get("IfNoneMatch") == "*":
                if self.data is not None:
                    raise _Error("PreconditionFailed")
            elif self.data is None or kwargs.get("IfMatch") != sha256(self.data).hexdigest():
                raise _Error("PreconditionFailed")
            self.data = kwargs["Body"]
            if self.lose_ack:
                self.lose_ack = False
                raise OSError("acknowledgement lost")
            return {"ETag": sha256(self.data).hexdigest()}


def _gate(client):
    return S3ArtifactClosureGate(
        client,
        bucket="bucket",
        prefix="artifacts",
        store_id="store",
        session_id="session",
        encryption={},
    )


def test_s3_gate_concurrent_reservations_cannot_be_lost_or_expired():
    client = _ConditionalClient()
    client.barrier = Barrier(2)
    first, second = _gate(client), _gate(client)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(gate.reserve, token * 32, "art_" + token * 32, "c" * 64)
            for gate, token in ((first, "a"), (second, "b"))
        ]
        for future in futures:
            future.result(timeout=10)
    state, _ = first.read()
    assert len(state["active"]) == 2 and state["revision"] == 2
    claim = ArtifactClosureClaim("store", "session", "d" * 64, ())
    with pytest.raises(ValueError, match="quiesced"):
        first.seal(claim, expected_revision=2)
    first.release("a" * 32, "art_" + "a" * 32, "c" * 64)
    with pytest.raises(ValueError, match="quiesced"):
        second.seal(claim, expected_revision=3)
    second.release("b" * 32, "art_" + "b" * 32, "c" * 64)
    _gate(client).seal(claim, expected_revision=6)
    with pytest.raises(ValueError, match="fenced"):
        first.reserve("e" * 32, "art_" + "e" * 32, "c" * 64)


def test_s3_gate_reconciles_lost_ack_and_rejects_stale_inventory():
    client = _ConditionalClient()
    client.lose_ack = True
    gate = _gate(client)
    gate.reserve("a" * 32, "art_" + "a" * 32, "b" * 64)
    assert gate.read()[0]["revision"] == 1
    gate.reserve("a" * 32, "art_" + "a" * 32, "b" * 64)
    assert client.put_calls == 1
    gate.release("a" * 32, "art_" + "a" * 32, "b" * 64)
    claim = ArtifactClosureClaim("store", "session", "d" * 64, ())
    with pytest.raises(ValueError, match="inventory changed"):
        gate.seal(claim, expected_revision=0)
    gate.seal(claim, expected_revision=3)
    assert gate.read()[0]["revision"] == 4
    _gate(client).seal(claim, expected_revision=2)
    assert gate.read()[0]["revision"] == 4


def test_s3_gate_rejects_conflicting_intent_and_malformed_durable_state():
    client = _ConditionalClient()
    gate = _gate(client)
    gate.reserve("a" * 32, "art_" + "a" * 32, "b" * 64)
    with pytest.raises(ValueError, match="conflicts"):
        gate.release("a" * 32, "art_" + "a" * 32, "c" * 64)
    assert client.data is not None
    state = json.loads(client.data)
    state["revision"] = True
    client.data = json.dumps(state).encode()
    calls = client.put_calls
    with pytest.raises(ValueError, match="Invalid"):
        gate.release("a" * 32, "art_" + "a" * 32, "b" * 64)
    assert client.put_calls == calls
