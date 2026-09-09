"""Recording consent, publication atomicity and private playback boundaries."""

from __future__ import annotations

import asyncio
import io
import sqlite3

import pytest
from PIL import Image
from pydantic import ValidationError

from cayu._browser_recording_store import BrowserRecordingStore, BrowserRecordingUnavailable
from cayu.browser_recording import BrowserRecordingPolicy


def policy(**overrides):
    return BrowserRecordingPolicy(
        scope="run", allowed_origins=("https://public.test",), retention_seconds=60, **overrides
    )


def png():
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "green").save(output, format="PNG")
    return output.getvalue()


def identity(**overrides):
    return {
        "session_id": "session",
        "session_instance_id": "instance",
        "allocation_fingerprint": "a" * 64,
        "browser_id": "browser",
        "worker_instance": "worker",
        **overrides,
    }


async def prepared(tmp_path, **overrides):
    store = BrowserRecordingStore(tmp_path / "private" / "recordings.sqlite")
    selected = policy(**overrides)
    config = await store.authorize_capture(
        session_id="session",
        policy=selected,
        guest_endpoint="wss://recording.test/browser-recordings/guest",
    )
    recording = await store.begin(selected, identity())
    owner = await store.claim(recording)
    return store, selected, config, recording, owner


@pytest.mark.parametrize(
    "values",
    [
        {"allowed_origins": ("http://public.test",)},
        {"allow_authenticated_pages": True},
        {"allow_authenticated_pages": 0},
        {"capture_during_sensitive_entry": True},
        {"allowed_profile_contexts": ()},
        {"frames_per_second": 6},
        {"max_duration_seconds": True},
        {"scope": "../run"},
        {"record": True},
    ],
)
def test_recording_policy_refuses_widening(values):
    args = {
        "scope": "run",
        "allowed_origins": ("https://public.test",),
        "retention_seconds": 60,
        **values,
    }
    with pytest.raises(ValidationError):
        BrowserRecordingPolicy(**args)


def test_video_is_settled_private_and_idempotent(tmp_path):
    async def check():
        store, selected, config, recording, owner = await prepared(tmp_path)
        assert config.credential.get_secret_value() not in config.model_dump_json()
        with pytest.raises(BrowserRecordingUnavailable):
            await store.authenticate_capture("0" * 64)
        assert (await store.authenticate_capture(config.credential.get_secret_value()))[
            0
        ] == selected
        await store.append(
            recording,
            owner=owner,
            policy=selected,
            sequence=0,
            page_id="page",
            elapsed_ms=0,
            png=png(),
        )
        with pytest.raises(BrowserRecordingUnavailable):
            await store.media(recording, 0)
        # Lost acknowledgement: retry the exact frame after reopening the DB.
        reopened = BrowserRecordingStore(store.path)
        await reopened.append(
            recording,
            owner=owner,
            policy=selected,
            sequence=0,
            page_id="page",
            elapsed_ms=0,
            png=png(),
        )
        with pytest.raises(BrowserRecordingUnavailable):
            await reopened.append(
                recording,
                owner=owner,
                policy=selected,
                sequence=0,
                page_id="different",
                elapsed_ms=0,
                png=png(),
            )
        await reopened.finish(recording, owner=owner)
        await reopened.finish(recording, owner=owner, reason="worker_lost")
        manifest = await reopened.manifest(recording)
        assert manifest.status == "complete"
        assert len(manifest.segments) == 1
        assert manifest.video_sha256
        assert (await reopened.video(recording)).startswith(b"\x1aE\xdf\xa3")
        assert (await reopened.media(recording, 0)).startswith(b"\x1aE\xdf\xa3")

    asyncio.run(check())


def test_retained_worker_new_allocation_and_owner_fencing(tmp_path):
    async def check():
        store, selected, _, recording, old_owner = await prepared(tmp_path)
        assert await store.begin(selected, identity()) == recording
        owner = await store.claim(recording)
        with pytest.raises(BrowserRecordingUnavailable):
            await store.append(
                recording,
                owner=old_owner,
                policy=selected,
                sequence=0,
                page_id="page",
                elapsed_ms=0,
                png=png(),
            )
        await store.append(
            recording,
            owner=owner,
            policy=selected,
            sequence=2,
            page_id="page",
            elapsed_ms=1000,
            png=png(),
        )
        next_recording = await store.begin(selected, identity(allocation_fingerprint="b" * 64))
        assert next_recording != recording
        await store.finish(recording, owner=owner, elapsed_ms=2000)
        manifest = await store.manifest(recording)
        assert manifest.status == "partial"
        assert [(gap.start_ms, gap.end_ms) for gap in manifest.gaps] == [(0, 1000), (1500, 2000)]
        assert (await store.manifest(next_recording)).status == "recording"

    asyncio.run(check())


def test_duration_and_storage_limits_do_not_publish_extra_media(tmp_path):
    async def check():
        store, selected, _, recording, owner = await prepared(tmp_path, max_duration_seconds=1)
        with pytest.raises(BrowserRecordingUnavailable):
            await store.append(
                recording,
                owner=owner,
                policy=selected,
                sequence=2,
                page_id="page",
                elapsed_ms=1000,
                png=png(),
            )
        await store.finish(recording, reason="limit_exhausted", owner=owner)
        assert (await store.manifest(recording)).status == "unavailable"
        assert not (await store.manifest(recording)).segments

    asyncio.run(check())


def test_retention_deletion_and_wrong_scope(tmp_path):
    async def check():
        store, selected, _, recording, owner = await prepared(tmp_path)
        with pytest.raises(BrowserRecordingUnavailable):
            await store.begin(selected, identity(session_id="other"))
        await store.append(
            recording,
            owner=owner,
            policy=selected,
            sequence=0,
            page_id="page",
            elapsed_ms=0,
            png=png(),
        )
        await store.finish(recording, owner=owner)
        with sqlite3.connect(store.path) as db:
            db.execute("UPDATE recordings SET expires=0")
        with pytest.raises(BrowserRecordingUnavailable):
            await store.media(recording, 0)
        await store.purge_expired()
        with sqlite3.connect(store.path) as db:
            assert db.execute("SELECT COUNT(*) FROM segments").fetchone()[0] == 0

    asyncio.run(check())


@pytest.mark.parametrize(
    "boundary", ["before_append", "after_append", "during_finalize", "after_finalize"]
)
def test_recording_process_loss_never_replays_capture(tmp_path, boundary):
    import os
    import subprocess
    import sys
    from pathlib import Path

    async def scenario():
        store, selected, _, recording, _ = await prepared(tmp_path)
        program = r"""
import asyncio, os, sys
from cayu import BrowserRecordingStore, BrowserRecordingPolicy
from tests.core.test_browser_recording import png
async def main():
    store = BrowserRecordingStore(sys.argv[1])
    recording, boundary = sys.argv[2:4]
    policy = BrowserRecordingPolicy.model_validate_json(sys.argv[4])
    owner = await store.claim(recording)
    if boundary == 'before_append': os._exit(73)
    await store.append(recording, owner=owner, policy=policy, sequence=0, page_id='page', elapsed_ms=0, png=png())
    if boundary == 'after_append': os._exit(73)
    async def lose_before_media(*args, **kwargs): os._exit(73)
    if boundary == 'during_finalize': store._combine = lose_before_media
    await store.finish(recording, owner=owner)
    os._exit(73)
asyncio.run(main())
"""
        root = Path(__file__).resolve().parents[2]
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            program,
            str(store.path),
            recording,
            boundary,
            selected.model_dump_json(),
            env={**os.environ, "PYTHONPATH": f"{root / 'src'}{os.pathsep}{root}"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        assert process.returncode == 73, stderr.decode()
        reopened = BrowserRecordingStore(store.path)
        await reopened.finish(recording, reason="worker_lost")
        receipt = await reopened.manifest(recording)
        assert receipt.status == (
            "unavailable"
            if boundary == "before_append"
            else "complete"
            if boundary == "after_finalize"
            else "partial"
        )
        assert len(receipt.segments) == (0 if boundary == "before_append" else 1)
        await reopened.finish(recording, reason="worker_lost")
        assert await reopened.manifest(recording) == receipt

    asyncio.run(scenario())


def test_missing_tail_is_partial_and_delete_cannot_resurrect(tmp_path):
    async def scenario():
        store, selected, _, recording, owner = await prepared(tmp_path)
        await store.append(
            recording,
            owner=owner,
            policy=selected,
            sequence=0,
            page_id="page",
            elapsed_ms=0,
            png=png(),
        )
        await store.finish(recording, owner=owner, elapsed_ms=2000)
        receipt = await store.manifest(recording)
        assert receipt.status == "partial"
        assert [(gap.start_ms, gap.end_ms) for gap in receipt.gaps] == [(500, 2000)]
        await store.delete(recording)
        assert await store.begin(selected, identity()) == recording
        with pytest.raises(BrowserRecordingUnavailable):
            await store.claim(recording)
        with pytest.raises(BrowserRecordingUnavailable):
            await store.manifest(recording)
        assert await store.recordings_for_session("session") == []

    asyncio.run(scenario())


def test_full_database_finalization_preserves_retrievable_segments(tmp_path):
    import random

    async def scenario():
        store = BrowserRecordingStore(tmp_path / "recordings.sqlite", max_storage_bytes=1024**2)
        selected = policy()
        await store.authorize_capture(
            session_id="session",
            policy=selected,
            guest_endpoint="wss://recording.test/browser-recordings/guest",
        )
        recording = await store.begin(selected, identity())
        owner = await store.claim(recording)
        output = io.BytesIO()
        Image.frombytes("L", (640, 480), random.Random(1627).randbytes(640 * 480)).convert(
            "RGB"
        ).save(output, format="PNG")
        for sequence in range(20):
            try:
                await store.append(
                    recording,
                    policy=selected,
                    owner=owner,
                    sequence=sequence,
                    page_id="page",
                    elapsed_ms=sequence * 500,
                    png=output.getvalue(),
                )
            except BrowserRecordingUnavailable:
                break
        else:
            pytest.fail("The bounded database did not fill.")
        committed = (await store.manifest(recording)).segments
        assert committed
        await store.finish(recording, owner=owner, reason="storage_failure")
        receipt = await store.manifest(recording)
        assert receipt.status == "partial"
        assert receipt.reason == "storage_failure"
        assert receipt.video_sha256 is None
        assert receipt.segments == committed
        for segment in committed:
            assert (await store.media(recording, segment.sequence)).startswith(b"\x1aE\xdf\xa3")
        reopened = BrowserRecordingStore(store.path, max_storage_bytes=1024**2)
        await reopened.finish(recording, reason="worker_lost")
        assert await reopened.manifest(recording) == receipt

    asyncio.run(scenario())
