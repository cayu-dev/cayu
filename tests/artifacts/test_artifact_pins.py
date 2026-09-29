"""Durable artifact pin contract for built-in stores, plus S3 CAS interleavings."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from threading import Barrier, RLock
from typing import Any

import pytest
from tests.artifacts.test_aws_s3 import _ClientError, _S3Client

from cayu import ArtifactScope, ArtifactStoreUnavailableError, InvalidArtifactIdError
from cayu.artifacts import LocalArtifactStore, S3ArtifactStore, aws_s3
from cayu.artifacts._s3_pins import S3ArtifactPinState

_PIN_KEY = "/_pins/"


class _ConditionalS3Client(_S3Client):
    """Serialized in-memory S3 with conditional-write faults and interleaving hooks.

    Conditional semantics follow S3: ``If-None-Match: *`` on an existing key and
    a stale ``If-Match`` fail with 412 ``PreconditionFailed``; ``If-Match`` on a
    missing key fails with 404 ``NoSuchKey``. ``conflict_409`` models the
    ``ConditionalRequestConflict`` S3 returns for racing conditional writes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lock = RLock()
        self.before_put: list[tuple[str, Callable[[], None]]] = []
        self.before_delete: list[Callable[[], None]] = []
        self.put_faults: list[tuple[str, str]] = []
        self.pin_puts = 0

    @staticmethod
    def _take(entries: list[Any], key: str) -> Any:
        for index, entry in enumerate(entries):
            fragment = entry[0] if isinstance(entry, tuple) else entry
            if fragment in key:
                del entries[index]
                return entry
        return None

    # Each put_object call below sends exactly one request.
    cayu_single_attempt_put_object = True

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        hook = self._take(self.before_put, key)
        if hook is not None:
            hook[1]()
        fault = self._take(self.put_faults, key)
        kind = None if fault is None else fault[1]
        with self.lock:
            if _PIN_KEY in key:
                self.pin_puts += 1
            if kind == "conflict_409":
                raise _ClientError("ConditionalRequestConflict")
            if kind == "precondition_412":
                raise _ClientError("PreconditionFailed")
            if kind == "unavailable":
                raise _ClientError("ServiceUnavailable")
            if "IfMatch" in kwargs and (kwargs["Bucket"], key) not in self.objects:
                raise _ClientError("NoSuchKey")
            response = super().put_object(**kwargs)
        if kind == "lost_ack":
            raise OSError("connection reset after S3 applied the write")
        if kind == "applied_412":
            # A transport retry of an already-applied conditional request.
            raise _ClientError("PreconditionFailed")
        return response

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        with self.lock:
            return super().get_object(**kwargs)

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        with self.lock:
            return super().head_object(**kwargs)

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        with self.lock:
            return super().list_objects_v2(**kwargs)

    def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
        if self.before_delete:
            self.before_delete.pop(0)()
        with self.lock:
            return super().delete_objects(**kwargs)


def _owner(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _state(client: _S3Client, artifact_id: str) -> dict[str, Any] | None:
    raw = client.objects.get(("bucket", f"cayu/artifacts/_pins/{artifact_id}.json"))
    return None if raw is None else json.loads(raw)


def _generation(client: _S3Client, artifact_id: str) -> str:
    """The publication generation named by an artifact's committed metadata."""

    raw = client.objects[("bucket", f"cayu/artifacts/{artifact_id}/metadata.json")]
    return json.loads(raw)["generation"]


def _data_keys(client: _S3Client, artifact_id: str) -> set[str]:
    return {
        key
        for _, key in client.objects
        if key.startswith(f"cayu/artifacts/{artifact_id}/")
        and key.endswith(("/content", "/metadata.json"))
    }


@pytest.fixture(params=["local", "s3"])
def open_store(request, tmp_path) -> Callable[[], Any]:
    """Each call models an independent process attached to the same storage."""

    if request.param == "local":
        if not LocalArtifactStore(tmp_path / "artifacts").supports_pins:
            pytest.skip("Local durable pins need durable publication support.")
        return lambda: LocalArtifactStore(tmp_path / "artifacts")
    client = _ConditionalS3Client()
    return lambda: S3ArtifactStore("bucket", client=client)


def test_pin_contract_rejects_absent_invalid_and_malformed_owners(open_store) -> None:
    async def run():
        store = open_store()
        assert store.supports_pins is True
        missing = "art_" + "1" * 32
        with pytest.raises(FileNotFoundError):
            await store.pin(missing, owner="session:a")
        with pytest.raises(FileNotFoundError):
            await store.release_pin(missing, owner="session:a")
        with pytest.raises(InvalidArtifactIdError):
            await store.pin("not-an-artifact", owner="session:a")
        for owner in ("", " padded", "padded ", 3, "x" * 1025, "\ud800"):
            with pytest.raises(ValueError):
                await store.pin("not-an-artifact", owner=owner)
            with pytest.raises(ValueError):
                await store.release_pin(missing, owner=owner)
        await store.pin(
            (await store.put_bytes(b"x", filename="x", session_id="s")).id, owner="é" * 512
        )

    asyncio.run(run())


def test_pin_contract_is_owner_exact_idempotent_and_durable_across_instances(open_store) -> None:
    async def run():
        artifact = await open_store().put_bytes(b"keep", filename="keep.txt", session_id="s")
        await open_store().pin(artifact.id, owner="session:a")
        await open_store().pin(artifact.id, owner="session:a")
        await open_store().pin(artifact.id, owner="snapshot:b")
        await open_store().release_pin(artifact.id, owner="foreign")
        with pytest.raises(ValueError, match="durable pin"):
            await open_store().delete(artifact.id)
        await open_store().release_pin(artifact.id, owner="session:a")
        await open_store().release_pin(artifact.id, owner="session:a")
        with pytest.raises(ValueError, match="durable pin"):
            await open_store().delete(artifact.id)
        assert (await open_store().read_bytes(artifact.id)).content == b"keep"
        await open_store().release_pin(artifact.id, owner="snapshot:b")
        await open_store().delete(artifact.id)
        with pytest.raises(FileNotFoundError):
            await open_store().read_bytes(artifact.id)
        with pytest.raises(FileNotFoundError):
            await open_store().pin(artifact.id, owner="too-late")
        with pytest.raises(FileNotFoundError):
            await open_store().release_pin(artifact.id, owner="session:a")
        await open_store().delete(artifact.id)

    asyncio.run(run())


def test_pin_contract_retains_claimed_closure_items(open_store) -> None:
    async def run():
        store = open_store()
        artifact = await store.put_bytes(b"owned", filename="item", session_id="session")
        await store.pin(artifact.id, owner="session:retained")
        claim = await store.claim_session_closure(
            "session", "a" * 64, max_records=10, max_bytes=1_000_000
        )
        with pytest.raises(ValueError, match="durable pin"):
            await open_store().delete_session_closure_artifact(claim, artifact.id)
        assert (await store.read_bytes(artifact.id)).content == b"owned"
        await store.release_pin(artifact.id, owner="session:retained")
        await open_store().delete_session_closure_artifact(claim, artifact.id)
        await open_store().delete_session_closure_artifact(claim, artifact.id)
        with pytest.raises(FileNotFoundError):
            await store.read_bytes(artifact.id)

    asyncio.run(run())


def _s3_pair() -> tuple[_ConditionalS3Client, S3ArtifactStore, S3ArtifactStore]:
    client = _ConditionalS3Client()
    return (
        client,
        S3ArtifactStore("bucket", client=client),
        S3ArtifactStore("bucket", client=client),
    )


@pytest.mark.parametrize("existing_state", [False, True])
def test_s3_pin_that_loses_the_race_to_a_completed_delete_fails_as_absent(existing_state) -> None:
    client, pinner, deleter = _s3_pair()

    async def run():
        artifact = await pinner.put_bytes(b"gone", filename="gone.txt", session_id="s")
        if existing_state:
            await pinner.pin(artifact.id, owner="earlier")
            await pinner.release_pin(artifact.id, owner="earlier")
        # Pin has read its state and proved presence; the whole deletion,
        # including returning the state to idle, lands before its CAS.
        client.before_put.append((_PIN_KEY, lambda: asyncio.run(deleter.delete(artifact.id))))
        with pytest.raises(FileNotFoundError):
            await pinner.pin(artifact.id, owner="session:late")
        return artifact

    artifact = asyncio.run(run())
    assert _data_keys(client, artifact.id) == set()
    state = _state(client, artifact.id)
    assert state is not None
    assert (state["owners"], state["deleters"]) == ([], [])


@pytest.mark.parametrize("existing_state", [False, True])
def test_s3_delete_that_loses_the_race_to_a_pin_refuses_and_preserves(existing_state) -> None:
    client, pinner, deleter = _s3_pair()

    async def run():
        artifact = await pinner.put_bytes(b"kept", filename="kept.txt", session_id="s")
        if existing_state:
            await pinner.pin(artifact.id, owner="earlier")
            await pinner.release_pin(artifact.id, owner="earlier")
        # Delete has read an owner-free state; the pin lands before its fence.
        client.before_put.append(
            (_PIN_KEY, lambda: asyncio.run(pinner.pin(artifact.id, owner="session:a")))
        )
        with pytest.raises(ValueError, match="durable pin"):
            await deleter.delete(artifact.id)
        assert (await deleter.read_bytes(artifact.id)).content == b"kept"
        return artifact

    artifact = asyncio.run(run())
    state = _state(client, artifact.id)
    assert state is not None
    assert state["owners"] == [_owner("session:a")]
    assert state["deleters"] == []
    assert client.delete_calls == []


def test_s3_pin_during_an_in_flight_deletion_fails_as_absent() -> None:
    client, pinner, deleter = _s3_pair()
    outcomes: list[BaseException] = []

    def pin_inside_deletion() -> None:
        try:
            asyncio.run(pinner.pin(artifact.id, owner="session:a"))
        except BaseException as error:
            outcomes.append(error)

    artifact = asyncio.run(pinner.put_bytes(b"x", filename="x", session_id="s"))
    client.before_delete.append(pin_inside_deletion)
    asyncio.run(deleter.delete(artifact.id))

    assert len(outcomes) == 1 and type(outcomes[0]) is FileNotFoundError
    assert _data_keys(client, artifact.id) == set()
    assert _state(client, artifact.id)["owners"] == []


def test_s3_lost_pin_acknowledgement_is_accepted_only_by_readback() -> None:
    client, store, other = _s3_pair()

    async def run():
        artifact = await store.put_bytes(b"x", filename="x", session_id="s")
        client.put_faults.append((_PIN_KEY, "lost_ack"))
        await store.pin(artifact.id, owner="session:a")
        with pytest.raises(ValueError, match="durable pin"):
            await other.delete(artifact.id)

        second = await store.put_bytes(b"y", filename="y", session_id="s")
        client.put_faults.append((_PIN_KEY, "lost_ack"))
        original_get = client.get_object
        pin_reads = 0

        def get_object(**kwargs):
            nonlocal pin_reads
            if _PIN_KEY in kwargs["Key"]:
                pin_reads += 1
                if pin_reads == 2:
                    raise OSError("readback unavailable")
            return original_get(**kwargs)

        client.get_object = get_object  # type: ignore[method-assign]
        with pytest.raises(ArtifactStoreUnavailableError, match="reconciled") as raised:
            await store.pin(second.id, owner="session:b")
        assert isinstance(raised.value.__cause__, ExceptionGroup)
        client.get_object = original_get  # type: ignore[method-assign]
        # The uncertain pin landed; it retains rather than exposes the artifact.
        with pytest.raises(ValueError, match="durable pin"):
            await other.delete(second.id)
        await other.release_pin(second.id, owner="session:b")
        await other.delete(second.id)

    asyncio.run(run())


def test_s3_unapplied_pin_failure_is_reported_without_retention() -> None:
    client, store, _ = _s3_pair()

    async def run():
        artifact = await store.put_bytes(b"x", filename="x", session_id="s")
        client.put_faults.append((_PIN_KEY, "unavailable"))
        with pytest.raises(ArtifactStoreUnavailableError, match="could not publish"):
            await store.pin(artifact.id, owner="session:a")
        assert _state(client, artifact.id) is None
        await store.delete(artifact.id)

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["lost_ack", "applied_412"])
def test_s3_ambiguous_deletion_fence_publication_still_completes(fault) -> None:
    client, store, _ = _s3_pair()

    async def run():
        artifact = await store.put_bytes(b"x", filename="x", session_id="s")
        client.put_faults.append((_PIN_KEY, fault))
        await store.delete(artifact.id)
        return artifact

    artifact = asyncio.run(run())
    assert _data_keys(client, artifact.id) == set()
    state = _state(client, artifact.id)
    assert (state["deleters"], state["revision"]) == ([], 2)


def test_s3_already_applied_pin_retry_is_idempotent() -> None:
    client, store, _ = _s3_pair()

    async def run():
        artifact = await store.put_bytes(b"x", filename="x", session_id="s")
        client.put_faults.append((_PIN_KEY, "applied_412"))
        await store.pin(artifact.id, owner="session:a")
        return artifact

    artifact = asyncio.run(run())
    assert _state(client, artifact.id)["owners"] == [_owner("session:a")]
    assert client.pin_puts == 1


def test_s3_conditional_conflicts_retry_within_a_fixed_bound() -> None:
    client, store, _ = _s3_pair()

    async def run():
        artifact = await store.put_bytes(b"x", filename="x", session_id="s")
        client.put_faults.extend([(_PIN_KEY, "conflict_409"), (_PIN_KEY, "precondition_412")])
        await store.pin(artifact.id, owner="session:a")
        assert client.pin_puts == 3
        client.pin_puts = 0
        client.put_faults.extend([(_PIN_KEY, "precondition_412")] * 20)
        with pytest.raises(ArtifactStoreUnavailableError, match="retry bound"):
            await store.pin(artifact.id, owner="session:b")
        assert client.pin_puts == 16
        return artifact

    artifact = asyncio.run(run())
    assert _state(client, artifact.id)["owners"] == [_owner("session:a")]


def test_s3_slow_deletion_cannot_remove_a_republication_pinned_in_the_meantime() -> None:
    client, first, second = _s3_pair()
    observed: list[BaseException | None] = []

    async def republish_and_pin(artifact) -> None:
        await second.delete(artifact.id)
        # The first deletion's request has not been applied yet. The identity
        # is published again with identical bytes, as a new generation.
        await second.put_bytes(
            b"same", artifact_id=artifact.id, filename="same.txt", session_id="s"
        )
        try:
            await second.pin(artifact.id, owner="session:republished")
        except BaseException as error:
            observed.append(error)
        else:
            observed.append(None)

    async def run():
        artifact = await first.put_bytes(b"same", filename="same.txt", session_id="s")
        client.before_delete.append(lambda: asyncio.run(republish_and_pin(artifact)))
        # The slow request targets only the first publication: its content key
        # is gone, and its metadata condition fails on the republication.
        await first.delete(artifact.id)
        assert observed == [None]
        assert (await first.read_bytes(artifact.id)).content == b"same"
        assert _state(client, artifact.id)["deleters"] == []
        with pytest.raises(ValueError, match="durable pin"):
            await first.delete(artifact.id)

    asyncio.run(run())


def test_s3_abandoned_deletion_fence_is_settled_by_a_completed_deletion() -> None:
    client, store, other = _s3_pair()

    async def run():
        artifact = await store.put_bytes(
            b"blob", artifact_id="art_" + "a" * 32, filename="b", session_id="s"
        )
        # Model process loss after the deletion fence committed.
        S3ArtifactPinState(
            client, bucket="bucket", prefix="cayu/artifacts", encryption={}
        ).begin_deletion(artifact.id, "f" * 32, [_generation(client, artifact.id)])
        with pytest.raises(FileNotFoundError):
            await other.pin(artifact.id, owner="session:a")
        # A completed deletion of the same generation settles the lost fence
        # at once: whatever that process sent can only touch that generation.
        await other.delete(artifact.id)
        assert _data_keys(client, artifact.id) == set()
        assert _state(client, artifact.id)["deleters"] == []
        await other.put_bytes(b"blob", artifact_id=artifact.id, filename="b", session_id="s")
        await other.pin(artifact.id, owner="session:a")
        with pytest.raises(ValueError, match="durable pin"):
            await other.delete(artifact.id)

    asyncio.run(run())


def test_s3_failed_object_removal_leaves_nothing_pinnable_until_a_retry_completes() -> None:
    client, store, other = _s3_pair()

    async def run():
        artifact = await store.put_bytes(b"x", filename="x", session_id="s")
        client.delete_errors_by_suffix["/content"] = "AccessDenied"
        with pytest.raises(ArtifactStoreUnavailableError, match="AccessDenied"):
            await store.delete(artifact.id)
        # S3 answered the one request, so its fence is retired; the metadata
        # is gone, so the surviving content cannot be pinned.
        assert _state(client, artifact.id)["deleters"] == []
        with pytest.raises(FileNotFoundError):
            await other.pin(artifact.id, owner="session:a")
        client.delete_errors_by_suffix.clear()
        # The retry sweeps the orphaned generation content.
        await other.delete(artifact.id)
        assert _data_keys(client, artifact.id) == set()
        assert _state(client, artifact.id)["deleters"] == []

    asyncio.run(run())


def test_s3_corrupt_pin_state_fails_closed() -> None:
    client, store, _ = _s3_pair()

    async def run():
        artifact = await store.put_bytes(b"x", filename="x", session_id="s")
        client.objects[("bucket", f"cayu/artifacts/_pins/{artifact.id}.json")] = b'{"owners": []}'
        with pytest.raises(ValueError, match="pin state"):
            await store.pin(artifact.id, owner="session:a")
        with pytest.raises(ValueError, match="pin state"):
            await store.delete(artifact.id)
        assert (await store.read_bytes(artifact.id)).content == b"x"

    asyncio.run(run())


def test_s3_pin_owner_bound_and_encryption(monkeypatch) -> None:
    import cayu.artifacts._s3_pins as pins

    monkeypatch.setattr(pins, "MAX_PIN_OWNERS", 2)
    client = _ConditionalS3Client()
    store = S3ArtifactStore("bucket", client=client, kms_key_id="arn:aws:kms:us-east-1:1:key/k")

    async def run():
        artifact = await store.put_bytes(
            b"x", filename="x", scope=ArtifactScope.ENVIRONMENT, environment_name="e"
        )
        await store.pin(artifact.id, owner="a")
        await store.pin(artifact.id, owner="b")
        with pytest.raises(ValueError, match="bound"):
            await store.pin(artifact.id, owner="c")
        await store.pin(artifact.id, owner="a")
        return artifact

    artifact = asyncio.run(run())
    pin_puts = [call for call in client.put_calls if _PIN_KEY in call["Key"]]
    assert pin_puts and all(call["SSEKMSKeyId"].endswith("key/k") for call in pin_puts)
    assert _state(client, artifact.id)["owners"] == sorted([_owner("a"), _owner("b")])


def test_s3_listing_ignores_pin_state_objects() -> None:
    _, store, _ = _s3_pair()

    async def run():
        artifact = await store.put_bytes(b"x", filename="x", session_id="s")
        await store.pin(artifact.id, owner="session:a")
        assert (await store.list()).artifacts == (artifact,)
        await store.release_pin(artifact.id, owner="session:a")
        await store.delete(artifact.id)
        assert (await store.list()).artifacts == ()

    asyncio.run(run())


@pytest.mark.parametrize("round_", range(10))
def test_s3_concurrent_pins_and_delete_never_retain_a_deleted_artifact(round_) -> None:
    client = _ConditionalS3Client()
    artifact = asyncio.run(
        S3ArtifactStore("bucket", client=client).put_bytes(b"x", filename="x", session_id="s")
    )
    owners = [f"session:{index}" for index in range(6)]
    barrier = Barrier(len(owners) + 1)

    def attempt(operation: Callable[[S3ArtifactStore], Any]) -> BaseException | None:
        barrier.wait(timeout=5)
        try:
            asyncio.run(operation(S3ArtifactStore("bucket", client=client)))
        except (FileNotFoundError, ValueError) as error:
            return error
        return None

    with ThreadPoolExecutor(len(owners) + 1) as pool:
        futures = [
            pool.submit(attempt, lambda store, owner=owner: store.pin(artifact.id, owner=owner))
            for owner in owners
        ]
        deletion = pool.submit(attempt, lambda store: store.delete(artifact.id))
        pin_results = [future.result() for future in futures]
        delete_result = deletion.result()

    pinned = {owner for owner, result in zip(owners, pin_results, strict=True) if result is None}
    state = _state(client, artifact.id)
    assert state is not None
    assert set(state["owners"]) == {_owner(owner) for owner in pinned}
    if delete_result is None:
        assert pinned == set()
        assert _data_keys(client, artifact.id) == set()
        assert all(type(result) is FileNotFoundError for result in pin_results)
    else:
        assert type(delete_result) is ValueError and "durable pin" in str(delete_result)
        assert pinned
        assert len(_data_keys(client, artifact.id)) == 2


def test_live_s3_pin_contract_probes_run_hermetically(monkeypatch) -> None:
    pytest.importorskip("boto3")
    from examples.aws import s3_artifact_pins_live as live

    class _SingleAttemptProbe(live._ProbeClient):
        # Live, the probe forwards meta.events to the botocore client it wraps;
        # here it wraps a fake whose put_object sends one request per call.
        cayu_single_attempt_put_object = True

    monkeypatch.setattr(live, "_ProbeClient", _SingleAttemptProbe)
    client = _ConditionalS3Client()

    class Session:
        def client(self, name: str) -> _ConditionalS3Client:
            assert name == "s3"
            return client

    live._prove_serial_contract(S3ArtifactStore("bucket", client=client))
    assert live._prove_interleavings("bucket", Session()) >= 2
    live._prove_ambiguous_acknowledgements("bucket", Session())
    live._prove_late_deletion_cannot_remove_a_republication("bucket", Session())
    assert not any(key.endswith(("/content", "/metadata.json")) for _, key in client.objects)


class _PendingDeleteClient(_ConditionalS3Client):
    """The first DeleteObjects times out at the client while S3 still holds it."""

    def __init__(self) -> None:
        super().__init__()
        self.pending: dict[str, Any] | None = None

    def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
        if self.pending is None:
            self.pending = kwargs
            raise TimeoutError("client timed out; the request is still pending at S3")
        return super().delete_objects(**kwargs)

    def apply_pending(self) -> None:
        assert self.pending is not None
        super().delete_objects(**self.pending)


def test_s3_late_timed_out_delete_cannot_remove_a_republished_pinned_artifact() -> None:
    client = _PendingDeleteClient()
    store = S3ArtifactStore("bucket", client=client)

    async def run():
        artifact = await store.put_bytes(b"checkpoint", filename="x", session_id="s")
        first = _generation(client, artifact.id)
        # The first DeleteObjects times out at the client while S3 still holds it.
        with pytest.raises(ArtifactStoreUnavailableError):
            await store.delete(artifact.id)
        [unresolved] = _state(client, artifact.id)["deleters"]
        assert (unresolved["targets"], unresolved["unresolved"]) == ([first], True)
        # While S3 may still apply it, the targeted publication is not pinnable.
        with pytest.raises(ArtifactStoreUnavailableError, match="unknown outcome"):
            await store.pin(artifact.id, owner="session:checkpoint")

        # A later deletion completes, and identical bytes are published again.
        await store.delete(artifact.id)
        assert _state(client, artifact.id)["deleters"] == []
        await store.put_bytes(b"checkpoint", artifact_id=artifact.id, filename="x", session_id="s")
        assert _generation(client, artifact.id) != first
        await store.pin(artifact.id, owner="session:checkpoint")

        # The first request is applied arbitrarily late. It names only the
        # first generation, so the pinned republication survives it.
        client.apply_pending()
        assert (await store.read_bytes(artifact.id)).content == b"checkpoint"
        assert len(_data_keys(client, artifact.id)) == 2
        with pytest.raises(ValueError, match="durable pin"):
            await store.delete(artifact.id)

    asyncio.run(run())


def test_s3_sdk_retry_cannot_resurrect_a_generation_a_deletion_retired() -> None:
    """A lost content acknowledgement is never retried into a swept generation."""

    boto3 = pytest.importorskip("boto3")
    from botocore.awsrequest import AWSResponse
    from botocore.config import Config
    from botocore.exceptions import ReadTimeoutError

    class _Raw:
        def stream(self, **_: Any):
            return iter([b""])

    class _Client(_PendingDeleteClient):
        """Content PUTs go through real botocore signing and retries."""

        def __init__(self) -> None:
            super().__init__()
            self.sent_keys: list[str] = []
            self.sdk = boto3.client(
                "s3",
                region_name="us-east-1",
                aws_access_key_id="AKIDEXAMPLE",
                aws_secret_access_key="secret",
                endpoint_url="https://s3.invalid",
                config=Config(retries={"mode": "standard", "total_max_attempts": 3}),
            )
            self.sdk._endpoint.http_session = self
            # The store registers its send guard on the client's own events.
            self.meta = self.sdk.meta

        def put_object(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["Key"].endswith("/content"):
                self.content_request = kwargs
                return self.sdk.put_object(**kwargs)
            return super().put_object(**kwargs)

        def send(self, request: Any) -> Any:
            key = self.content_request["Key"]
            self.sent_keys.append(key)
            response = _ConditionalS3Client.put_object(self, **self.content_request)
            if len(self.sent_keys) == 1:
                artifact_id = key.split("/")[-3]

                async def sweep() -> None:
                    # One deletion times out while still pending at S3; a second
                    # completes, removes the generation and settles both fences.
                    collector = S3ArtifactStore("bucket", client=self)
                    with pytest.raises(ArtifactStoreUnavailableError):
                        await collector.delete(artifact_id)
                    await collector.delete(artifact_id)
                    assert _state(self, artifact_id)["deleters"] == []

                asyncio.run(sweep())
                raise ReadTimeoutError(endpoint_url=request.url)
            return AWSResponse(request.url, 200, {"etag": response["ETag"]}, _Raw())

    client = _Client()
    store = S3ArtifactStore("bucket", client=client)
    artifact_id = f"art_{'b' * 32}"

    async def run() -> None:
        artifact = await store.put_bytes(
            b"checkpoint", artifact_id=artifact_id, filename="x", session_id="s"
        )
        # The lost attempt was never re-sent; a fresh generation was written.
        assert len(client.sent_keys) == 2
        assert len(set(client.sent_keys)) == 2
        committed = _generation(client, artifact.id)
        assert client.sent_keys[1].endswith(f"/{committed}/content")
        await store.pin(artifact.id, owner="session:checkpoint")

        # The pending deletion named only the abandoned generation.
        client.apply_pending()
        assert (await store.read_bytes(artifact.id)).content == b"checkpoint"
        assert _state(client, artifact.id)["owners"] == [_owner("session:checkpoint")]

    asyncio.run(run())


def test_s3_wrapper_that_hides_a_retrying_sdk_is_refused_before_any_write() -> None:
    """The review's wrapper: an inherited declaration does not cover a new put_object."""

    boto3 = pytest.importorskip("boto3")
    from botocore.config import Config

    class _Wrapper(_PendingDeleteClient):
        # Routes content through a retrying SDK but exposes no meta.events.
        def __init__(self) -> None:
            super().__init__()
            self.sdk = boto3.client(
                "s3",
                region_name="us-east-1",
                aws_access_key_id="AKIDEXAMPLE",
                aws_secret_access_key="secret",
                endpoint_url="https://s3.invalid",
                config=Config(retries={"mode": "standard", "total_max_attempts": 2}),
            )

        def put_object(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["Key"].endswith("/content"):
                return self.sdk.put_object(**kwargs)
            return super().put_object(**kwargs)

    client = _Wrapper()
    store = S3ArtifactStore("bucket", client=client)
    with pytest.raises(aws_s3.S3ArtifactClientConfigurationError, match="single-attempt"):
        asyncio.run(
            store.put_bytes(
                b"checkpoint", artifact_id=f"art_{'b' * 32}", filename="x", session_id="s"
            )
        )
    # Refused before dispatch: nothing was written and nothing can be pinned.
    assert not any(key.endswith(("/content", "/metadata.json")) for _, key in client.objects)


def test_s3_plain_wrapper_without_guard_or_declaration_is_refused() -> None:
    class _Plain:
        """Forwards to a fake but neither exposes meta.events nor declares."""

        def __init__(self) -> None:
            self._inner = _S3Client()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def put_object(self, **kwargs: Any) -> dict[str, Any]:
            return self._inner.put_object(**kwargs)

    plain = _Plain()
    store = S3ArtifactStore("bucket", client=plain)
    with pytest.raises(aws_s3.S3ArtifactClientConfigurationError):
        asyncio.run(store.put_bytes(b"x", filename="x", session_id="s"))
    # Reads, listing and deletion do not depend on single-attempt writes.
    assert asyncio.run(store.list(session_id="s")).artifacts == ()

    class _Declared(_Plain):
        cayu_single_attempt_put_object = True

    class _InstanceOnly(_Plain):
        def __init__(self) -> None:
            super().__init__()
            self.cayu_single_attempt_put_object = True

    assert aws_s3._declares_single_attempt_put_object(_Declared())
    assert not aws_s3._declares_single_attempt_put_object(_InstanceOnly())
    declared = S3ArtifactStore("bucket", client=_Declared())
    artifact = asyncio.run(declared.put_bytes(b"x", filename="x", session_id="s"))
    assert asyncio.run(declared.read_bytes(artifact.id)).content == b"x"


def test_s3_generation_swept_before_its_metadata_commit_is_never_pinnable() -> None:
    class _SweepBeforeCommit(_ConditionalS3Client):
        swept = False

        # Each put_object call below sends exactly one request.
        cayu_single_attempt_put_object = True

        def put_object(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["Key"].endswith("/metadata.json") and not self.swept:
                self.swept = True
                artifact_id = kwargs["Key"].split("/")[-2]
                # The acknowledged content has no metadata yet, so a concurrent
                # deletion removes it as an orphan before the commit lands.
                asyncio.run(S3ArtifactStore("bucket", client=self).delete(artifact_id))
            return super().put_object(**kwargs)

    client = _SweepBeforeCommit()
    store = S3ArtifactStore("bucket", client=client)

    async def run() -> None:
        artifact = await store.put_bytes(b"checkpoint", filename="x", session_id="s")
        assert client.swept
        with pytest.raises(FileNotFoundError):
            await store.pin(artifact.id, owner="session:checkpoint")
        assert _state(client, artifact.id)["owners"] == []

    asyncio.run(run())


@pytest.mark.parametrize(
    ("status", "unapplied"),
    [(200, None), (403, True), (500, False)],
)
def test_s3_content_put_is_sent_once_and_classified(status: int, unapplied: bool | None) -> None:
    botocore_session = pytest.importorskip("botocore.session")
    from botocore.awsrequest import AWSResponse
    from botocore.config import Config

    client = botocore_session.get_session().create_client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="AKIDEXAMPLE",
        aws_secret_access_key="secret",
        config=Config(retries={"mode": "standard", "max_attempts": 3}),
    )
    answered: list[int] = []

    class _Raw:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def stream(self, **_: Any):
            yield self.body

    def answer(request: Any, **_: Any) -> Any:
        answered.append(status)
        body = (
            b""
            if status == 200
            else b"<Error><Code>AccessDenied</Code><Message>no</Message></Error>"
            if status == 403
            else b"<Error><Code>InternalError</Code><Message>retry</Message></Error>"
        )
        return AWSResponse(request.url, status, {"etag": '"e"'}, _Raw(body))

    # Installed first, so the guard runs before this local answer.
    aws_s3._install_content_put_guard(client)
    client.meta.events.register("before-send.s3.PutObject", answer)
    submission = aws_s3._ContentSubmission()
    error: BaseException | None = None
    try:
        submission.send(
            client, Bucket="bucket", Key="cayu/artifacts/x/g/content", Body=b"c", IfNoneMatch="*"
        )
    except Exception as caught:
        error = caught
    # One attempt was signed and sent; botocore never retried it.
    assert submission.sends == len(answered) == 1
    if unapplied is None:
        assert error is None
    else:
        assert error is not None
        assert submission.unapplied(error) is unapplied


def test_s3_timed_out_closure_delete_is_settled_by_a_completed_retry() -> None:
    client = _PendingDeleteClient()
    store = S3ArtifactStore("bucket", client=client)

    async def run():
        artifact = await store.put_bytes(b"owned", filename="x", session_id="s")
        claim = await store.claim_session_closure(
            "s", "a" * 64, max_records=10, max_bytes=1_000_000
        )
        with pytest.raises(TimeoutError):
            await store.delete_session_closure_artifact(claim, artifact.id)
        [unresolved] = _state(client, artifact.id)["deleters"]
        assert unresolved["unresolved"] is True
        await store.delete_session_closure_artifact(claim, artifact.id)
        assert _state(client, artifact.id)["deleters"] == []
        client.apply_pending()
        assert _data_keys(client, artifact.id) == set()
        with pytest.raises(FileNotFoundError):
            await store.pin(artifact.id, owner="session:late")

    asyncio.run(run())


def test_s3_pin_refuses_an_identity_missing_either_committed_object() -> None:
    client, store, other = _s3_pair()

    async def run():
        artifact = await store.put_bytes(b"x", filename="x", session_id="s")
        client.delete_errors_by_suffix["/metadata.json"] = "AccessDenied"
        with pytest.raises(ArtifactStoreUnavailableError, match="AccessDenied"):
            await store.delete(artifact.id)
        # S3 answered the one request, so its fence is retired; the metadata
        # survived but the content did not, so nothing may pin it.
        assert _state(client, artifact.id)["deleters"] == []
        with pytest.raises(FileNotFoundError):
            await other.pin(artifact.id, owner="session:a")

    asyncio.run(run())


def test_s3_cancelled_delete_settles_its_fence_before_returning() -> None:
    client, store, _ = _s3_pair()
    entered = threading.Event()
    release = threading.Event()

    def hold() -> None:
        entered.set()
        assert release.wait(5)

    async def run():
        artifact = await store.put_bytes(b"x", filename="x", session_id="s")
        client.before_delete.append(hold)
        task = asyncio.create_task(store.delete(artifact.id))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel("caller stopped")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The worker thread finished the removal and retired its fence.
        for _ in range(100):
            state = _state(client, artifact.id)
            if state is not None and state["deleters"] == []:
                break
            await asyncio.sleep(0.01)
        return artifact

    artifact = asyncio.run(run())
    assert _data_keys(client, artifact.id) == set()
    assert _state(client, artifact.id)["deleters"] == []


def _botocore_s3(
    responses: list[int], answered: list[int], bodies: list[bytes] | None = None
) -> Any:
    """A real botocore S3 client whose attempts are answered locally, after signing."""

    botocore_session = pytest.importorskip("botocore.session")
    from botocore.awsrequest import AWSResponse
    from botocore.config import Config

    client = botocore_session.get_session().create_client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="AKIDEXAMPLE",
        aws_secret_access_key="secret",
        config=Config(retries={"mode": "standard", "max_attempts": 3}),
    )

    class _Raw:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def stream(self, **_: Any):
            yield self.body

    def answer(request: Any, **_: Any) -> Any:
        status = responses.pop(0) if responses else 500
        answered.append(status)
        if bodies is not None:
            bodies.append(request.body)
        body = (
            b'<?xml version="1.0" encoding="UTF-8"?><DeleteResult '
            b'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"></DeleteResult>'
            if status == 200
            else b"<Error><Code>AccessDenied</Code><Message>no</Message></Error>"
            if status == 403
            else b"<Error><Code>InternalError</Code><Message>retry</Message></Error>"
        )
        return AWSResponse(request.url, status, {}, _Raw(body))

    # Installed first, so the attempt counter runs before this local answer.
    aws_s3._install_deletion_attempt_counter(client)
    client.meta.events.register("before-send.s3.DeleteObjects", answer)
    return client


@pytest.mark.parametrize(
    ("responses", "succeeds", "resolved"),
    [
        ([200], True, True),
        ([403], False, True),
        ([500, 200], True, False),
        ([], False, False),
    ],
)
def test_s3_deletion_counts_every_signed_attempt(responses, succeeds, resolved) -> None:
    answered: list[int] = []
    client = _botocore_s3(list(responses), answered)
    submission = aws_s3._DeletionSubmission()
    error: BaseException | None = None
    try:
        submission.send(client, "bucket", [{"Key": "cayu/artifacts/x/g/content"}])
    except Exception as caught:
        error = caught
    # Every attempt botocore signed and sent, including retries, was counted.
    assert submission.sends == len(answered) >= 1
    assert (error is None) is succeeds
    # Only one attempt that S3 answered with a success or a client error is
    # settled; a server error or a retried request may still be applied late.
    assert submission.resolved(error) is resolved


def test_s3_deletion_conditions_the_metadata_object_on_its_etag_on_the_wire() -> None:
    answered: list[int] = []
    bodies: list[bytes] = []
    client = _botocore_s3([200], answered, bodies)
    aws_s3._DeletionSubmission().send(
        client,
        "bucket",
        [
            {"Key": "cayu/artifacts/x/g/content"},
            {"Key": "cayu/artifacts/x/metadata.json", "ETag": '"etag-of-publication"'},
        ],
    )
    [body] = bodies
    assert b"<Key>cayu/artifacts/x/metadata.json</Key>" in body
    assert b"<ETag>&quot;etag-of-publication&quot;</ETag>" in body or (
        b'<ETag>"etag-of-publication"</ETag>' in body
    )


def test_s3_republishing_identical_bytes_is_a_new_publication() -> None:
    client, store, _ = _s3_pair()

    async def run():
        artifact = await store.put_bytes(b"same", filename="same.txt", session_id="s")
        first = _generation(client, artifact.id)
        first_metadata = client.objects[("bucket", f"cayu/artifacts/{artifact.id}/metadata.json")]
        await store.delete(artifact.id)
        await store.put_bytes(b"same", artifact_id=artifact.id, filename="same.txt", session_id="s")
        second = _generation(client, artifact.id)
        second_metadata = client.objects[("bucket", f"cayu/artifacts/{artifact.id}/metadata.json")]
        # Different content keys and different metadata bytes, hence ETags,
        # although the artifact and its content are identical.
        assert first != second
        assert first_metadata != second_metadata
        assert _data_keys(client, artifact.id) == {
            f"cayu/artifacts/{artifact.id}/{second}/content",
            f"cayu/artifacts/{artifact.id}/metadata.json",
        }
        assert (await store.read_bytes(artifact.id)).content == b"same"

    asyncio.run(run())


def test_s3_legacy_layout_artifact_is_read_pinned_and_deleted_by_its_etag() -> None:
    client = _PendingDeleteClient()
    store = S3ArtifactStore("bucket", client=client)

    async def run():
        # Model an artifact written before generation-keyed content.
        published = await store.put_bytes(b"old", filename="old.txt", session_id="s")
        generation = _generation(client, published.id)
        legacy_content = f"cayu/artifacts/{published.id}/content"
        metadata_key = f"cayu/artifacts/{published.id}/metadata.json"
        client.objects[("bucket", legacy_content)] = client.objects.pop(
            ("bucket", f"cayu/artifacts/{published.id}/{generation}/content")
        )
        client.objects[("bucket", metadata_key)] = published.model_dump_json().encode()

        assert (await store.read_bytes(published.id)).content == b"old"
        assert [item.id for item in (await store.list()).artifacts] == [published.id]
        await store.pin(published.id, owner="session:a")
        await store.release_pin(published.id, owner="session:a")

        # The legacy deletion's request stays pending at S3; a retry completes.
        with pytest.raises(ArtifactStoreUnavailableError):
            await store.delete(published.id)
        [unresolved] = _state(client, published.id)["deleters"]
        assert unresolved["targets"] == ["legacy"]
        await store.delete(published.id)
        assert _data_keys(client, published.id) == set()

        # A republication is generation-keyed; the late legacy request cannot
        # remove it, because its metadata condition names the legacy ETag.
        await store.put_bytes(b"old", artifact_id=published.id, filename="old.txt", session_id="s")
        await store.pin(published.id, owner="session:b")
        client.apply_pending()
        assert (await store.read_bytes(published.id)).content == b"old"

    asyncio.run(run())


def test_s3_applied_deletion_with_a_lost_acknowledgement_is_settled_by_the_next_delete() -> None:
    client, store, _ = _s3_pair()
    original = client.delete_objects
    lost: list[bool] = []

    def delete_objects(**kwargs: Any) -> dict[str, Any]:
        response = original(**kwargs)
        if not lost:
            lost.append(True)
            raise ConnectionResetError("S3 applied the deletion; the acknowledgement was lost")
        return response

    client.delete_objects = delete_objects  # type: ignore[method-assign]

    async def run():
        artifact = await store.put_bytes(b"x", filename="x", session_id="s")
        with pytest.raises(ArtifactStoreUnavailableError):
            await store.delete(artifact.id)
        assert _data_keys(client, artifact.id) == set()
        [unresolved] = _state(client, artifact.id)["deleters"]
        assert unresolved["unresolved"] is True
        # Nothing is left to remove, yet the retry still targets the fenced
        # generation, so its completion settles the unresolved fence.
        await store.delete(artifact.id)
        assert _state(client, artifact.id)["deleters"] == []

    asyncio.run(run())
