import asyncio
import os
import threading

import pytest

from cayu.artifacts import ArtifactScope, LocalArtifactStore


def test_invocation_handle_does_not_forward_administrative_closure(tmp_path):
    from cayu.tools._resources import InvocationArtifactStoreHandle

    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        handle = InvocationArtifactStoreHandle(
            store,
            redactor_snapshot_provider=lambda: None,
            capture_observer=lambda revision: None,
        )
        assert store.supports_session_closure_claims
        assert not handle.supports_session_closure_claims
        with pytest.raises(NotImplementedError):
            await handle.claim_session_closure("session", "a" * 64, max_records=10, max_bytes=10000)
        assert await store.load_session_closure_claim("session") is None
        artifact = await store.put_bytes(b"still open", filename="safe.txt", session_id="session")
        assert (await store.read_bytes(artifact.id)).content == b"still open"

    asyncio.run(run())


@pytest.mark.parametrize("max_bytes", [16, 64, 256])
def test_public_closure_uses_local_claim_and_replays_after_reopen(tmp_path, max_bytes):
    from tests.core.session_closure_conformance import create_closure_session

    from cayu import CayuApp
    from cayu.runtime.session_closure import ArtifactSessionClosureStore, SessionClosurePolicy
    from cayu.storage.sqlite import SQLiteSessionStore

    async def run():
        policy = SessionClosurePolicy(max_bytes=max_bytes * 1024 * 1024)
        path = tmp_path / "sessions.sqlite"
        sessions = SQLiteSessionStore(path)
        store = LocalArtifactStore(tmp_path / "artifacts")
        try:
            await create_closure_session(sessions, "session")
            artifact = await store.put_bytes(b"owned", filename="owned.txt", session_id="session")
            app = CayuApp(
                session_store=sessions, session_closure_stores=(ArtifactSessionClosureStore(store),)
            )
            report = await app.erase_session_closure("session", policy=policy)
            assert report.complete
            claim = await store.load_session_closure_claim("session")
            assert [item.artifact_id for item in claim.artifacts] == [artifact.id]
        finally:
            await sessions.close()
        reopened = SQLiteSessionStore(path)
        try:
            app = CayuApp(
                session_store=reopened,
                session_closure_stores=(
                    ArtifactSessionClosureStore(LocalArtifactStore(store.root)),
                ),
            )
            replay = await app.erase_session_closure("session", policy=policy)
            assert replay.complete
            with pytest.raises(ValueError, match="fenced"):
                await store.put_bytes(b"late", filename="late.txt", session_id="session")
        finally:
            await reopened.close()

    asyncio.run(run())


def test_local_claim_retry_retires_owned_staging_after_cleanup_failure(tmp_path, monkeypatch):
    from cayu.artifacts import local

    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        artifact = await store.put_bytes(b"owned", filename="owned.txt", session_id="session")
        publication_error = OSError("rename unavailable")
        cleanup_error = OSError("unlink unavailable")
        original_unlink = os.unlink
        with monkeypatch.context() as patch:

            def fail_rename(*args, **kwargs):
                raise publication_error

            def fail_unlink(path, *args, **kwargs):
                if ".cayu-closure-" in str(path) and ".staging-" in str(path):
                    raise cleanup_error
                return original_unlink(path, *args, **kwargs)

            patch.setattr(local, "_rename_directory_no_replace", fail_rename)
            patch.setattr(os, "unlink", fail_unlink)
            with pytest.raises(BaseExceptionGroup) as error:
                await store.claim_session_closure(
                    "session", "a" * 64, max_records=10, max_bytes=10000
                )
            assert error.value.exceptions == (publication_error, cleanup_error)
        assert len(list(store.root.glob(".cayu-closure-*.staging-*"))) == 1
        reopened = LocalArtifactStore(store.root)
        claim = await reopened.claim_session_closure(
            "session", "a" * 64, max_records=10, max_bytes=10000
        )
        assert [item.artifact_id for item in claim.artifacts] == [artifact.id]
        assert not list(store.root.glob(".cayu-closure-*.staging-*"))

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["write_sync", "rename", "after_rename"])
def test_local_claim_publication_failure_reconciles_or_removes_staging(
    tmp_path, monkeypatch, phase
):
    from cayu.artifacts import local

    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        artifact = await store.put_bytes(b"owned", filename="owned.txt", session_id="session")
        original_sync = local._sync_descriptor
        original_rename = local._rename_directory_no_replace
        failure = OSError("claim publication fault")
        cleanup_error = OSError("claim cleanup sync fault")
        sync_calls = 0
        with monkeypatch.context() as patch:
            if phase == "write_sync":

                def fail_sync(fd):
                    nonlocal sync_calls
                    sync_calls += 1
                    raise failure if sync_calls == 1 else cleanup_error

                patch.setattr(local, "_sync_descriptor", fail_sync)
            else:

                def fail_rename(*args, **kwargs):
                    if phase == "after_rename":
                        original_rename(*args, **kwargs)
                    raise failure

                patch.setattr(local, "_rename_directory_no_replace", fail_rename)
            with pytest.raises(BaseException) as caught:
                await store.claim_session_closure(
                    "session", "a" * 64, max_records=10, max_bytes=10000
                )
            if phase == "write_sync":
                # The same injected sync fault also affects directory cleanup;
                # both phases must remain observable, without losing publication.
                assert isinstance(caught.value, BaseExceptionGroup)
                assert caught.value.exceptions[0] is failure
                assert caught.value.exceptions == (failure, cleanup_error)
            else:
                assert caught.value is failure
        assert local._sync_descriptor is original_sync
        assert not list(store.root.glob(".cayu-closure-*.staging-*"))
        reopened = LocalArtifactStore(store.root)
        retained = await reopened.load_session_closure_claim("session")
        assert (retained is not None) == (phase == "after_rename")
        claim = await reopened.claim_session_closure(
            "session", "a" * 64, max_records=10, max_bytes=10000
        )
        assert [item.artifact_id for item in claim.artifacts] == [artifact.id]
        assert (await reopened.read_bytes(artifact.id)).content == b"owned"

    asyncio.run(run())


def test_local_claim_waits_for_dispatched_write_after_owner_cancellation(tmp_path, monkeypatch):
    from cayu.artifacts import local

    dispatched = threading.Event()
    release = threading.Event()
    original = local._write_artifact

    def held_write(*args, **kwargs):
        dispatched.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(local, "_write_artifact", held_write)

    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        artifact_id = "art_" + "a" * 32
        writer = asyncio.create_task(
            store.put_bytes(
                b"in-flight", artifact_id=artifact_id, filename="write.txt", session_id="session"
            )
        )
        claimer = None
        try:
            assert await asyncio.to_thread(dispatched.wait, 5)
            claimer = asyncio.create_task(
                store.claim_session_closure("session", "b" * 64, max_records=10, max_bytes=10000)
            )
            writer.cancel("owner stopped waiting")
            assert writer.cancelling() == 1
            await asyncio.sleep(0)
            assert not claimer.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await writer
            assert writer.cancelled() and writer.cancelling() == 1
            claim = await claimer
            assert [item.artifact_id for item in claim.artifacts] == [artifact_id]
            await store.delete_session_closure_artifact(claim, artifact_id)
        finally:
            release.set()
            await asyncio.gather(
                writer, *(() if claimer is None else (claimer,)), return_exceptions=True
            )

    asyncio.run(run())


def test_local_closure_reopens_exact_claim_and_fences_new_publication(tmp_path):
    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts", store_id="local")
        artifact = await store.put_bytes(b"owned", filename="owned.txt", session_id="session")
        claim = await store.claim_session_closure(
            "session", "a" * 64, max_records=10, max_bytes=10000
        )
        assert [item.artifact_id for item in claim.artifacts] == [artifact.id]
        reopened = LocalArtifactStore(store.root, store_id="local")
        assert await reopened.load_session_closure_claim("session") == claim
        with pytest.raises(ValueError, match="fenced"):
            await reopened.put_bytes(b"late", filename="late.txt", session_id="session")
        with pytest.raises(ValueError, match="plan conflicts"):
            await reopened.claim_session_closure(
                "session", "b" * 64, max_records=10, max_bytes=10000
            )
        await reopened.delete_session_closure_artifact(claim, artifact.id)
        await reopened.delete_session_closure_artifact(claim, artifact.id)
        assert (
            await reopened.claim_session_closure(
                "session", "a" * 64, max_records=10, max_bytes=10000
            )
            == claim
        )
        assert (
            await reopened.list(scope=ArtifactScope.SESSION, session_id="session")
        ).total_count == 0
        assert await reopened.load_session_closure_claim("session") == claim

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["partial_removal", "rename_acknowledgement"])
def test_public_closure_retries_partial_artifact_removal_after_reopen(
    tmp_path, monkeypatch, failure
):
    from tests.core.session_closure_conformance import create_closure_session

    from cayu import CayuApp
    from cayu.artifacts import local
    from cayu.runtime.session_closure import ArtifactSessionClosureStore
    from cayu.storage.sqlite import SQLiteSessionStore

    async def run():
        path = tmp_path / "sessions.sqlite"
        sessions = SQLiteSessionStore(path)
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        try:
            await create_closure_session(sessions, "partial-artifact")
            artifact = await artifacts.put_bytes(
                b"must not survive closure", filename="owned.txt", session_id="partial-artifact"
            )
            target = artifacts.root / artifact.id
            app = CayuApp(
                session_store=sessions,
                session_closure_stores=(ArtifactSessionClosureStore(artifacts),),
            )
            original = local._remove_artifact_directory_if_unchanged
            original_rename = local._rename_directory_no_replace
            pending = []

            def partial_remove(path, expected_identity, **kwargs):
                if path == target or path.name.startswith(".cayu-closure-delete-"):
                    pending.append(path)
                    (path / "metadata.json").unlink()
                    raise OSError("injected partial artifact removal")
                return original(path, expected_identity, **kwargs)

            def lost_rename(source, destination, **kwargs):
                original_rename(source, destination, **kwargs)
                if destination.name.startswith(".cayu-closure-delete-"):
                    pending.append(destination)
                    raise OSError("injected rename acknowledgement loss")

            with monkeypatch.context() as patch:
                if failure == "partial_removal":
                    patch.setattr(local, "_remove_artifact_directory_if_unchanged", partial_remove)
                else:
                    patch.setattr(local, "_rename_directory_no_replace", lost_rename)
                first = await app.erase_session_closure("partial-artifact")
            assert not first.complete
            assert await sessions.load("partial-artifact") is not None
            assert len(pending) == 1
            assert (pending[0] / "content").read_bytes() == b"must not survive closure"
        finally:
            await sessions.close()

        reopened = SQLiteSessionStore(path)
        try:
            app = CayuApp(
                session_store=reopened,
                session_closure_stores=(
                    ArtifactSessionClosureStore(LocalArtifactStore(artifacts.root)),
                ),
            )
            second = await app.erase_session_closure("partial-artifact")
            assert second.complete
            assert not target.exists()
            assert not pending[0].exists()
            assert await reopened.load("partial-artifact") is None
        finally:
            await reopened.close()

    asyncio.run(run())


def test_local_claimed_delete_rejects_replacement_under_same_artifact_id(tmp_path):
    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        original = await store.put_bytes(b"old", filename="old.txt", session_id="old-session")
        claim = await store.claim_session_closure(
            "old-session", "a" * 64, max_records=10, max_bytes=10000
        )
        await store.delete_session_closure_artifact(claim, original.id)
        replacement = await store.put_bytes(
            b"new", artifact_id=original.id, filename="new.txt", session_id="new-session"
        )
        with pytest.raises(ValueError, match="replaced"):
            await store.delete_session_closure_artifact(claim, original.id)
        assert (await store.read_bytes(replacement.id)).content == b"new"

    asyncio.run(run())


def test_local_claimed_delete_rejects_missing_metadata_without_staging(tmp_path):
    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        artifact = await store.put_bytes(b"owned", filename="owned.txt", session_id="session")
        claim = await store.claim_session_closure(
            "session", "a" * 64, max_records=10, max_bytes=10000
        )
        target = store.root / artifact.id
        (target / "metadata.json").unlink()
        with pytest.raises(ValueError, match="incomplete unowned deletion"):
            await store.delete_session_closure_artifact(claim, artifact.id)
        assert (target / "content").read_bytes() == b"owned"

    asyncio.run(run())


def test_local_claimed_delete_rejects_symlinked_staging(tmp_path):
    import hashlib

    from cayu.artifacts._closure import encode_artifact_closure_claim

    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        artifact = await store.put_bytes(b"owned", filename="owned.txt", session_id="session")
        claim = await store.claim_session_closure(
            "session", "a" * 64, max_records=10, max_bytes=10000
        )
        digest = hashlib.sha256(
            encode_artifact_closure_claim(claim) + b"\0" + artifact.id.encode()
        ).hexdigest()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep").write_bytes(b"not owned")
        try:
            (store.root / (".cayu-closure-delete-" + digest)).symlink_to(
                outside, target_is_directory=True
            )
        except OSError:
            pytest.skip("Directory symlinks unavailable")
        with pytest.raises((ValueError, OSError)):
            await store.delete_session_closure_artifact(claim, artifact.id)
        assert (outside / "keep").read_bytes() == b"not owned"
        assert (await store.read_bytes(artifact.id)).content == b"owned"

    asyncio.run(run())


def test_local_closure_rejects_oversized_set_before_claiming(tmp_path):
    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        for i in range(2):
            await store.put_bytes(b"data", filename=f"{i}.txt", session_id="session")
        with pytest.raises(ValueError, match="truncated"):
            await store.claim_session_closure("session", "a" * 64, max_records=1, max_bytes=10000)
        assert await store.load_session_closure_claim("session") is None
        await store.put_bytes(b"still-open", filename="third.txt", session_id="session")

    asyncio.run(run())
