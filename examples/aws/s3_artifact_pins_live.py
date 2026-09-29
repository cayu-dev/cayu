"""Live contract for S3ArtifactStore durable pins on real S3 conditional writes.

Creates one temporary bucket, proves that pins and deletions serialize through
S3 compare-and-swap (including from two processes), proves that a deletion
request S3 applies again late cannot remove a later publication of the same
identity, then deletes every object and the bucket and verifies that the bucket
no longer exists.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing
import os
import random
import sys
import time
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

import boto3  # ty: ignore[unresolved-import]

from cayu import ArtifactScope, ArtifactStoreUnavailableError, S3ArtifactStore

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
_CONDITIONAL_CODES = frozenset({"PreconditionFailed", "ConditionalRequestConflict"})
_RACE_ROUNDS = 16
_PINNED = "Artifact is retained by a durable pin."


def _error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None
    code = response.get("Error", {}).get("Code")
    return code if isinstance(code, str) else None


class _ProbeClient:
    """Real S3 client that counts conditional rejections and can inject faults.

    Faults apply only to pin-state writes and to ``DeleteObjects``. ``lost_ack``
    performs the real request and then drops its acknowledgement; ``resend``
    performs the real write and then re-sends the identical conditional request,
    as a transport retry would.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self.before_pin_write: Callable[[], None] | None = None
        self.after_pin_write: str | None = None
        self.after_delete: str | None = None
        self.last_delete: dict[str, Any] | None = None
        self.conditional_rejections = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def put_object(self, **kwargs: Any) -> Any:
        is_pin_state = "/_pins/" in kwargs["Key"]
        if is_pin_state and self.before_pin_write is not None:
            hook, self.before_pin_write = self.before_pin_write, None
            hook()
        fault = None
        if is_pin_state:
            fault, self.after_pin_write = self.after_pin_write, None
        response = self._counted_put(**kwargs)
        if fault == "lost_ack":
            raise ConnectionResetError("simulated lost S3 acknowledgement")
        if fault == "resend":
            self._counted_put(**kwargs)
        return response

    def delete_objects(self, **kwargs: Any) -> Any:
        fault, self.after_delete = self.after_delete, None
        self.last_delete = kwargs
        response = self._client.delete_objects(**kwargs)
        if fault == "lost_ack":
            raise ConnectionResetError("simulated lost S3 DeleteObjects acknowledgement")
        return response

    def _counted_put(self, **kwargs: Any) -> Any:
        try:
            return self._client.put_object(**kwargs)
        except Exception as error:
            if _error_code(error) in _CONDITIONAL_CODES:
                self.conditional_rejections += 1
            raise


def _outcome(operation: Coroutine[Any, Any, Any]) -> str:
    try:
        asyncio.run(operation)
    except FileNotFoundError:
        return "absent"
    except ValueError as error:
        if str(error) == _PINNED:
            return "pinned"
        raise
    return "ok"


def _race_worker(
    role: str, bucket: str, region: str, artifact_ids: list[str], barrier: Any, results: Any
) -> None:
    # A separate process with its own client and store shares only S3 state.
    client = _ProbeClient(boto3.Session(region_name=region).client("s3"))
    store = S3ArtifactStore(bucket, client=client)
    outcomes = []
    for index, artifact_id in enumerate(artifact_ids):
        barrier.wait(timeout=120)
        # Seeded jitter spreads the two request sequences across each other's
        # conditional writes, so both winners and real CAS rejections occur.
        time.sleep(random.Random(f"{role}:{index}").uniform(0.0, 0.08))
        if role == "pin":
            outcomes.append(_outcome(store.pin(artifact_id, owner="live:pinner")))
        else:
            outcomes.append(_outcome(store.delete(artifact_id)))
    results.put((role, outcomes, client.conditional_rejections))


def _prove_serial_contract(store: S3ArtifactStore) -> None:
    artifact = asyncio.run(
        store.put_bytes(
            b"pinned",
            filename="pinned.txt",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="live",
        )
    )
    asyncio.run(store.pin(artifact.id, owner="live:a"))
    asyncio.run(store.pin(artifact.id, owner="live:a"))
    asyncio.run(store.release_pin(artifact.id, owner="live:foreign"))
    if _outcome(store.delete(artifact.id)) != "pinned":
        raise RuntimeError("S3 deletion removed a pinned artifact.")
    asyncio.run(store.release_pin(artifact.id, owner="live:a"))
    asyncio.run(store.delete(artifact.id))
    if _outcome(store.pin(artifact.id, owner="live:late")) != "absent":
        raise RuntimeError("S3 pin accepted an absent artifact.")


def _prove_interleavings(bucket: str, session: Any) -> int:
    """Force each race order on real S3 and require the stale CAS to be rejected."""

    probe = _ProbeClient(session.client("s3"))
    store = S3ArtifactStore(bucket, client=probe)
    other = S3ArtifactStore(bucket, client=session.client("s3"))

    first = asyncio.run(store.put_bytes(b"one", filename="one.txt", session_id="live"))
    probe.before_pin_write = lambda: asyncio.run(other.delete(first.id))
    if _outcome(store.pin(first.id, owner="live:late")) != "absent":
        raise RuntimeError("A pin racing a completed S3 deletion was accepted.")
    if probe.conditional_rejections < 1:
        raise RuntimeError("S3 did not reject the stale pin compare-and-swap.")

    second = asyncio.run(store.put_bytes(b"two", filename="two.txt", session_id="live"))
    before = probe.conditional_rejections
    probe.before_pin_write = lambda: asyncio.run(other.pin(second.id, owner="live:winner"))
    if _outcome(store.delete(second.id)) != "pinned":
        raise RuntimeError("An S3 deletion racing a pin removed the artifact.")
    if probe.conditional_rejections <= before:
        raise RuntimeError("S3 did not reject the stale deletion compare-and-swap.")
    if asyncio.run(other.read_bytes(second.id)).content != b"two":
        raise RuntimeError("A pinned S3 artifact was not preserved.")
    asyncio.run(other.release_pin(second.id, owner="live:winner"))
    asyncio.run(other.delete(second.id))
    return probe.conditional_rejections


def _prove_ambiguous_acknowledgements(bucket: str, session: Any) -> None:
    probe = _ProbeClient(session.client("s3"))
    store = S3ArtifactStore(bucket, client=probe)
    other = S3ArtifactStore(bucket, client=session.client("s3"))
    for fault in ("lost_ack", "resend"):
        artifact = asyncio.run(store.put_bytes(b"ack", filename="ack.txt", session_id="live"))
        probe.after_pin_write = fault
        asyncio.run(store.pin(artifact.id, owner="live:ack"))
        if _outcome(other.delete(artifact.id)) != "pinned":
            raise RuntimeError(f"S3 pin after {fault} was not durable.")
        asyncio.run(other.release_pin(artifact.id, owner="live:ack"))
        probe.after_pin_write = fault
        asyncio.run(store.delete(artifact.id))
        if _outcome(other.read_bytes(artifact.id)) != "absent":
            raise RuntimeError(f"S3 deletion after {fault} did not complete.")
        if _outcome(other.pin(artifact.id, owner="live:late")) != "absent":
            raise RuntimeError(f"S3 identity was pinnable after deletion with {fault}.")


def _pin_state(s3: Any, bucket: str, artifact_id: str) -> dict[str, Any]:
    response = s3.get_object(Bucket=bucket, Key=f"cayu/artifacts/_pins/{artifact_id}.json")
    return json.loads(response["Body"].read())


def _prove_late_deletion_cannot_remove_a_republication(bucket: str, session: Any) -> dict[str, Any]:
    """Replay an applied deletion after the identity was republished and pinned."""

    s3 = session.client("s3")
    probe = _ProbeClient(s3)
    store = S3ArtifactStore(bucket, client=probe)
    artifact = asyncio.run(store.put_bytes(b"late", filename="late.txt", session_id="live"))
    probe.after_delete = "lost_ack"
    try:
        asyncio.run(store.delete(artifact.id))
    except ArtifactStoreUnavailableError:
        pass
    else:
        raise RuntimeError("An S3 deletion with a lost acknowledgement reported success.")
    late_request = probe.last_delete
    deleters = _pin_state(s3, bucket, artifact.id)["deleters"]
    if not (len(deleters) == 1 and deleters[0]["unresolved"]):
        raise RuntimeError("An S3 deletion with an unknown outcome did not keep its fence.")
    if late_request is None or not any(
        "ETag" in item for item in late_request["Delete"]["Objects"]
    ):
        raise RuntimeError("The S3 deletion did not condition the metadata on its ETag.")

    # The identity is published again with identical bytes and pinned. The
    # unresolved fence names only the first publication, so it does not block.
    asyncio.run(
        store.put_bytes(b"late", artifact_id=artifact.id, filename="late.txt", session_id="live")
    )
    asyncio.run(store.pin(artifact.id, owner="live:late"))

    # S3 applies the first deletion request again, as a late duplicate would.
    replay = s3.delete_objects(**late_request)
    codes = sorted({error.get("Code") for error in replay.get("Errors", [])})
    if "PreconditionFailed" not in codes:
        raise RuntimeError(f"S3 did not reject the stale metadata deletion: {codes}.")
    if asyncio.run(store.read_bytes(artifact.id)).content != b"late":
        raise RuntimeError("A late S3 deletion removed a pinned republication.")
    if _outcome(store.delete(artifact.id)) != "pinned":
        raise RuntimeError("A pinned S3 republication was deletable.")

    # Once released, the next deletion settles the earlier fence as well.
    asyncio.run(store.release_pin(artifact.id, owner="live:late"))
    asyncio.run(store.delete(artifact.id))
    if _pin_state(s3, bucket, artifact.id)["deleters"]:
        raise RuntimeError("A completed S3 deletion left an earlier fence unsettled.")
    return {"late_deletion_replay_codes": codes}


def _prove_cross_process_race(bucket: str, region: str, session: Any) -> dict[str, int]:
    store = S3ArtifactStore(bucket, client=session.client("s3"))
    artifact_ids = [
        asyncio.run(
            store.put_bytes(f"race {index}".encode(), filename="race.txt", session_id="live")
        ).id
        for index in range(_RACE_ROUNDS)
    ]
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    workers = [
        context.Process(
            target=_race_worker, args=(role, bucket, region, artifact_ids, barrier, results)
        )
        for role in ("pin", "delete")
    ]
    for worker in workers:
        worker.start()
    reports = {}
    for _ in workers:
        role, outcomes, rejections = results.get(timeout=300)
        reports[role] = (outcomes, rejections)
    for worker in workers:
        worker.join(timeout=60)
        if worker.exitcode != 0:
            raise RuntimeError("An S3 pin race worker failed.")
    pin_wins = delete_wins = 0
    for artifact_id, pinned, deleted in zip(
        artifact_ids, reports["pin"][0], reports["delete"][0], strict=True
    ):
        if (pinned, deleted) == ("ok", "pinned"):
            pin_wins += 1
            if asyncio.run(store.read_bytes(artifact_id)).metadata.id != artifact_id:
                raise RuntimeError("A pinned S3 artifact was not preserved.")
            asyncio.run(store.release_pin(artifact_id, owner="live:pinner"))
            asyncio.run(store.delete(artifact_id))
        elif (pinned, deleted) == ("absent", "ok"):
            delete_wins += 1
            if _outcome(store.read_bytes(artifact_id)) != "absent":
                raise RuntimeError("An S3 deletion reported success but left the artifact.")
        else:
            raise RuntimeError(f"S3 pin/delete race violated exclusion: {pinned}/{deleted}.")
    return {
        "race_rounds": _RACE_ROUNDS,
        "race_pin_wins": pin_wins,
        "race_delete_wins": delete_wins,
        "race_conditional_rejections": reports["pin"][1] + reports["delete"][1],
    }


def _delete_bucket(s3: Any, bucket: str) -> None:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if keys:
            response = s3.delete_objects(Bucket=bucket, Delete={"Objects": keys, "Quiet": True})
            if response.get("Errors"):
                raise RuntimeError("S3 cleanup could not delete every object.")
    s3.delete_bucket(Bucket=bucket)
    s3.get_waiter("bucket_not_exists").wait(Bucket=bucket)
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception as error:
        if _error_code(error) in {"404", "NoSuchBucket", "NotFound"}:
            return
        raise
    raise RuntimeError("S3 cleanup bucket still exists.")


def main() -> None:
    if os.environ.get("CAYU_AWS_S3_PINS_LIVE") != "1":
        raise SystemExit("Set CAYU_AWS_S3_PINS_LIVE=1 to run this contract.")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise SystemExit("Set AWS_REGION or AWS_DEFAULT_REGION.")

    session = boto3.Session(region_name=region)
    s3 = session.client("s3")
    bucket: str | None = f"cayu-dev-pins-{uuid.uuid4().hex[:12]}"
    created = False
    try:
        options: dict[str, Any] = {"Bucket": bucket}
        if region != "us-east-1":
            options["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**options)
        created = True
        print(f"created temporary bucket {bucket}", file=sys.stderr, flush=True)
        s3.get_waiter("bucket_exists").wait(Bucket=bucket)
        s3.put_bucket_tagging(
            Bucket=bucket,
            Tagging={"TagSet": [{"Key": "cayu:purpose", "Value": "dev-lane"}]},
        )

        _prove_serial_contract(S3ArtifactStore(bucket, client=s3))
        stale_rejections = _prove_interleavings(bucket, session)
        _prove_ambiguous_acknowledgements(bucket, session)
        late = _prove_late_deletion_cannot_remove_a_republication(bucket, session)
        race = _prove_cross_process_race(bucket, region, session)

        _delete_bucket(s3, bucket)
        bucket = None
        print(
            EVIDENCE_PREFIX
            + json.dumps(
                {
                    "adapter": "aws-s3-artifact-pins",
                    "region": region,
                    "pin_contract": "verified",
                    "conditional_cas": "verified",
                    "forced_interleaving_rejections": stale_rejections,
                    "lost_ack_readback": "verified",
                    "applied_retry": "verified",
                    "late_deletion_after_republication": "verified",
                    **late,
                    "cross_process_race": "verified",
                    **race,
                    "cleanup": "verified",
                },
                sort_keys=True,
            )
        )
    finally:
        if created and bucket is not None:
            with contextlib.suppress(Exception):
                _delete_bucket(s3, bucket)


if __name__ == "__main__":
    main()
