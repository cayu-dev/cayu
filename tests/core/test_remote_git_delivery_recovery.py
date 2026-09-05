from __future__ import annotations

import asyncio
import copy
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from tests.core.test_remote_git_delivery import (
    _broker,
    _coding_publication,
    _git,
    _remote_fixture,
    _request,
)

from cayu.remote_git_delivery import (
    RemoteGitDeliveryAdmissionError,
    RemoteGitDeliveryError,
    RemoteGitDeliveryReconstructionRequiredError,
    RemoteGitDeliveryRequest,
    RemoteGitDeliveryState,
    RemoteGitLifecycleReceipt,
    approve_remote_git_delivery,
)
from cayu.runners import ExecCommand


def _case(tmp_path: Path, *, baseline_commit: str | None = None):
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, repository, store = asyncio.run(
        _coding_publication(tmp_path, source, baseline_commit=baseline_commit)
    )
    return (
        remote,
        workspace,
        product,
        _broker(tmp_path, remote, repository, store),
        _request(product, base),
    )


@pytest.mark.parametrize("failure_phase", ["before_cleanup", "after_cleanup", "terminal_ack"])
def test_terminal_publication_loss_recovers_without_local_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_phase: str
) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    original = broker.repository.publish_result
    injected = False

    async def fail_publication(request, result):
        nonlocal injected
        target = (
            RemoteGitDeliveryState.PARTIAL
            if failure_phase == "before_cleanup"
            else RemoteGitDeliveryState.PUSHED
        )
        if result.state is target and not injected:
            injected = True
            if failure_phase == "terminal_ack":
                await original(request, result)
            raise OSError("terminal publication interrupted")
        return await original(request, result)

    monkeypatch.setattr(broker.repository, "publish_result", fail_publication)
    with pytest.raises(OSError, match="publication interrupted"):
        asyncio.run(broker.run(request, product, source_workspace=workspace, approval=approval))
    committed = _git(remote, "rev-parse", request.repository.destination_ref)
    if failure_phase != "before_cleanup":
        assert not broker._delivery_root(request).exists()
    monkeypatch.setattr(broker.repository, "publish_result", original)
    recovered = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    assert recovered.result.state is RemoteGitDeliveryState.PUSHED
    assert recovered.result.local_commit == committed
    assert recovered.result.cleanup_settled
    assert not broker._delivery_root(request).exists()


def test_concurrent_prepare_cannot_delete_owned_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    peer = _broker(tmp_path, remote, broker.coding_repository, broker.repository.store)
    original = broker._observe_remote

    async def exercise():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def pause(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(broker, "_observe_remote", pause)
        task = asyncio.create_task(broker.prepare(request, product, source_workspace=workspace))
        await asyncio.wait_for(entered.wait(), 5)
        root = broker._delivery_root(request)
        identity = root.stat().st_ino
        try:
            with pytest.raises(RemoteGitDeliveryReconstructionRequiredError, match="already owned"):
                await peer.prepare(request, product, source_workspace=workspace)
            assert root.stat().st_ino == identity
        finally:
            release.set()
            await task

    asyncio.run(exercise())


def test_cleanup_deadline_keeps_partial_receipt_and_allows_owned_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _remote, workspace, product, broker, request = _case(tmp_path)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    original = broker._local_runner.exec
    cleanup_pid = None

    async def stall_cleanup(command, **kwargs):
        nonlocal cleanup_pid
        if any("_remote_git_cleanup.py" in part for part in command.argv):
            command = ExecCommand.process(
                sys.executable,
                "-I",
                "-c",
                "import os,time; print(os.getpid(), flush=True); time.sleep(60)",
            )
            kwargs["timeout_s"] = 1
            result = await original(command, **kwargs)
            cleanup_pid = int(result.stdout.strip())
            return result
        return await original(command, **kwargs)

    monkeypatch.setattr(broker._local_runner, "exec", stall_cleanup)
    partial = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    assert partial.result.state is RemoteGitDeliveryState.PARTIAL
    assert cleanup_pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(cleanup_pid, 0)
    assert broker._delivery_root(request).is_dir()
    monkeypatch.setattr(broker._local_runner, "exec", original)
    complete = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    assert complete.result.state is RemoteGitDeliveryState.PUSHED
    assert complete.result.local_commit == partial.result.local_commit


def test_delivery_preserves_executable_source_mode(tmp_path: Path) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    script = source / "app.py"
    script.write_text("VALUE = 'before'\n", encoding="utf-8")
    script.chmod(0o755)
    workspace, product, repository, store = asyncio.run(_coding_publication(tmp_path, source))
    broker = _broker(tmp_path, remote, repository, store)
    request = _request(product, base)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    result = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    assert result.result.state is RemoteGitDeliveryState.PUSHED
    assert _git(remote, "ls-tree", result.result.local_commit, "app.py").startswith("100755 ")


def test_large_remote_pack_is_bounded_during_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, _base = _remote_fixture(tmp_path)
    seed = tmp_path / "seed"
    (seed / "large.bin").write_bytes(os.urandom(256 * 1024))
    _git(seed, "add", "large.bin")
    _git(
        seed,
        "-c",
        "user.name=Cayu",
        "-c",
        "user.email=cayu@example.invalid",
        "commit",
        "-m",
        "large",
    )
    _git(seed, "push", str(remote), "HEAD:refs/heads/main")
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, repository, store = asyncio.run(_coding_publication(tmp_path, source))
    broker = _broker(tmp_path, remote, repository, store)
    request = _request(product, _git(seed, "rev-parse", "HEAD"))
    request = request.model_copy(
        update={
            "repository": request.repository.model_copy(
                update={
                    "expected_base_commit": _git(remote, "rev-parse", "refs/heads/main"),
                }
            ),
            "limits": request.limits.model_copy(update={"max_git_storage_bytes": 16 * 1024}),
        }
    )
    original = broker._git
    operations = []

    async def record(request, operation, *args, **kwargs):
        operations.append(operation)
        return await original(request, operation, *args, **kwargs)

    monkeypatch.setattr(broker, "_git", record)
    with pytest.raises(RemoteGitDeliveryError, match="fetch"):
        asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    assert "fetch_base" in operations
    assert "stage_exact_source" not in operations
    for path in broker._delivery_root(request).rglob("*"):
        if path.is_file():
            assert path.stat().st_size <= request.limits.max_git_storage_bytes
    assert not _git(
        remote, "for-each-ref", "--format=%(refname)", request.repository.destination_ref
    )


def test_recovery_rejects_a_different_commit_under_the_same_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    receipts = asyncio.run(broker.repository.load_lifecycle(request))
    forged = RemoteGitLifecycleReceipt(
        delivery_id=request.delivery_id,
        request_fingerprint=request.fingerprint,
        ordinal=len(receipts) + 1,
        prior_state=receipts[-1].state,
        state=RemoteGitDeliveryState.COMMITTED_LOCALLY,
        tree=prepared.tree,
        commit=request.repository.expected_base_commit,
    )

    async def substituted(request):
        return (*receipts, forged)

    monkeypatch.setattr(broker.repository, "load_lifecycle", substituted)
    with pytest.raises(RemoteGitDeliveryReconstructionRequiredError, match="Retained commit"):
        asyncio.run(broker.run(request, product, source_workspace=workspace, approval=approval))
    assert not _git(
        remote, "for-each-ref", "--format=%(refname)", request.repository.destination_ref
    )


def test_exact_terminal_replay_rejects_changed_source(tmp_path: Path) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    result = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    (tmp_path / "source" / "app.py").write_text("unreviewed change\n", encoding="utf-8")
    with pytest.raises(RemoteGitDeliveryAdmissionError, match="no longer matches"):
        asyncio.run(broker.run(request, product, source_workspace=workspace, approval=approval))
    assert (
        _git(remote, "rev-parse", request.repository.destination_ref) == result.result.local_commit
    )


def test_prepared_replay_revalidates_source_without_mutation(tmp_path: Path) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    root = broker._delivery_root(request)
    identity = root.stat().st_ino
    (tmp_path / "source" / "app.py").write_text("unreviewed change\n", encoding="utf-8")
    with pytest.raises(RemoteGitDeliveryAdmissionError, match="no longer matches"):
        asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    assert root.stat().st_ino == identity
    assert asyncio.run(broker.repository.load_prepared(request)) == prepared
    assert not _git(
        remote, "for-each-ref", "--format=%(refname)", request.repository.destination_ref
    )


def test_every_request_authority_field_rejects_stable_identity_drift(tmp_path: Path) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    root_identity = broker._delivery_root(request).stat().st_ino
    original = request.model_dump(mode="json")
    paths = []
    for key, value in original.items():
        if key == "delivery_id":
            continue  # Stable operation identity is deliberately held constant.
        if isinstance(value, dict):
            paths.extend((key, child) for child in value)
        else:
            paths.append((key,))

    async def exercise():
        for path in paths:
            changed = copy.deepcopy(original)
            owner = changed if len(path) == 1 else changed[path[0]]
            key = path[-1]
            value = owner[key]
            if type(value) is bool:
                owner[key] = not value
            elif type(value) is int:
                owner[key] = 0 if key == "retry_limit" else value + 1
            elif key == "authored_at":
                owner[key] = "2026-08-30T13:00:00-07:00"
            elif value.startswith("sha256:") or (
                len(value) in {40, 64} and all(char in "0123456789abcdef" for char in value)
            ):
                owner[key] = value[:-1] + ("0" if value[-1] != "0" else "1")
            else:
                owner[key] = value + "-other"
            if key in {"schema_version", "force_update", "delete_ref"}:
                with pytest.raises(ValueError):
                    RemoteGitDeliveryRequest.model_validate(changed)
                continue
            # Email, refs, fingerprints and all other bounded values remain
            # valid; it is conflicting authority, not malformed input, that
            # must be rejected at both public entrances.
            conflicting = RemoteGitDeliveryRequest.model_validate(changed)
            assert conflicting.fingerprint != request.fingerprint, path
            with pytest.raises(
                (RemoteGitDeliveryAdmissionError, RemoteGitDeliveryReconstructionRequiredError)
            ):
                await broker.prepare(conflicting, product, source_workspace=workspace)
            with pytest.raises(
                (RemoteGitDeliveryAdmissionError, RemoteGitDeliveryReconstructionRequiredError)
            ):
                await broker.run(conflicting, product, source_workspace=workspace, approval=None)
            assert broker._delivery_root(request).stat().st_ino == root_identity, path
        assert await broker.repository.load_prepared(request) == prepared

    asyncio.run(exercise())
    assert not _git(
        remote, "for-each-ref", "--format=%(refname)", request.repository.destination_ref
    )


def test_terminal_replay_requires_fresh_exact_remote_state(tmp_path: Path) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    completed = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    _git(
        remote,
        "update-ref",
        request.repository.destination_ref,
        request.repository.expected_base_commit,
    )
    with pytest.raises(
        RemoteGitDeliveryAdmissionError, match="terminal_remote_destination_changed"
    ):
        asyncio.run(broker.run(request, product, source_workspace=workspace, approval=approval))
    assert (
        _git(remote, "rev-parse", request.repository.destination_ref)
        == request.repository.expected_base_commit
    )
    assert asyncio.run(broker.repository.load_result(request, completed.result.digest)) == completed


def test_same_alias_cannot_redirect_an_approved_delivery(tmp_path: Path) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    other = tmp_path / "other.git"
    _git(tmp_path, "clone", "--bare", str(remote), str(other))
    changed = replace(broker.profile.remotes["origin"], url=str(other))
    profile = replace(broker.profile, remotes={"origin": changed})
    peer = type(broker)(
        profile, repository=broker.repository, coding_repository=broker.coding_repository
    )
    with pytest.raises(RemoteGitDeliveryAdmissionError, match="different authority"):
        asyncio.run(peer.run(request, product, source_workspace=workspace, approval=approval))
    for destination in (remote, other):
        assert not _git(
            destination, "for-each-ref", "--format=%(refname)", request.repository.destination_ref
        )


def test_repository_config_cannot_redirect_the_push(tmp_path: Path) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    other = tmp_path / "other.git"
    _git(tmp_path, "clone", "--bare", str(remote), str(other))
    _git(broker._delivery_root(request), "config", f"url.{other}.insteadOf", str(remote))
    result = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    assert result.result.state is RemoteGitDeliveryState.FAILED
    for destination in (remote, other):
        assert not _git(
            destination, "for-each-ref", "--format=%(refname)", request.repository.destination_ref
        )


@pytest.mark.parametrize("marker_failure", [False, True])
def test_cancelled_artifact_writer_fences_peer_until_real_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker_failure: bool
) -> None:
    import cayu._remote_git_ownership as ownership
    import cayu.artifacts.local as local

    remote, workspace, product, broker, request = _case(tmp_path)
    peer = _broker(tmp_path, remote, broker.coding_repository, broker.repository.store)
    release = threading.Event()
    original = local._put_deterministic_artifact
    settle = local._settle_artifact_write
    original_write = ownership.os.write

    def fail_marker(descriptor, content):
        if marker_failure and content == b"U":
            raise OSError("injected ownership marker failure")
        return original_write(descriptor, content)

    async def bounded_settle(**kwargs):
        return await settle(**kwargs, settlement_timeout_s=0.02)

    async def exercise():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()

        def paused(*args, **kwargs):
            if args[2].filename == "remote-git-delivery-request.json":
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5)
            return original(*args, **kwargs)

        monkeypatch.setattr(local, "_put_deterministic_artifact", paused)
        monkeypatch.setattr(local, "_settle_artifact_write", bounded_settle)
        monkeypatch.setattr(ownership.os, "write", fail_marker)
        task = asyncio.create_task(broker.prepare(request, product, source_workspace=workspace))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled() and task.cancelling() == 1
            with pytest.raises(RemoteGitDeliveryReconstructionRequiredError, match="settlement"):
                await peer.prepare(request, product, source_workspace=workspace)
        finally:
            release.set()
        async with asyncio.timeout(5):
            while True:
                try:
                    await peer.prepare(request, product, source_workspace=workspace)
                    break
                except RemoteGitDeliveryReconstructionRequiredError:
                    await asyncio.sleep(0.01)
        assert not _git(
            remote, "for-each-ref", "--format=%(refname)", request.repository.destination_ref
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("retry_limit", [0, 1])
def test_interrupted_pushing_receipt_consumes_retry_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_limit: int
) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    request = request.model_copy(
        update={"limits": request.limits.model_copy(update={"retry_limit": retry_limit})}
    )
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    original_push = broker._push
    original_append = broker.repository.append_lifecycle
    pushes = 0

    async def exercise():
        entered = asyncio.Event()

        async def pause_push(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        async def fail_cancel_record(request, receipt):
            if receipt.state is RemoteGitDeliveryState.AMBIGUOUS:
                raise OSError("cancellation receipt unavailable")
            return await original_append(request, receipt)

        monkeypatch.setattr(broker, "_push", pause_push)
        monkeypatch.setattr(broker.repository, "append_lifecycle", fail_cancel_record)
        task = asyncio.create_task(
            broker.run(request, product, source_workspace=workspace, approval=approval)
        )
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert task.cancelled() and task.cancelling() == 1
        assert isinstance(caught.value.__cause__, BaseExceptionGroup)
        assert (await broker.repository.load_lifecycle(request))[
            -1
        ].state is RemoteGitDeliveryState.PUSHING

        async def count_push(*args, **kwargs):
            nonlocal pushes
            pushes += 1
            return await original_push(*args, **kwargs)

        monkeypatch.setattr(broker, "_push", count_push)
        monkeypatch.setattr(broker.repository, "append_lifecycle", original_append)
        return await broker.run(request, product, source_workspace=workspace, approval=approval)

    result = asyncio.run(exercise())
    assert pushes == retry_limit
    assert result.result.state is (
        RemoteGitDeliveryState.PUSHED
        if retry_limit
        else RemoteGitDeliveryState.RECONSTRUCTION_REQUIRED
    )
    assert bool(
        _git(remote, "for-each-ref", "--format=%(refname)", request.repository.destination_ref)
    ) is bool(retry_limit)


@pytest.mark.parametrize("base_removed", [False, True])
def test_partial_success_recovers_after_base_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_removed: bool
) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    original = broker._cleanup_delivery_root

    async def no_cleanup(*args):
        return False

    monkeypatch.setattr(broker, "_cleanup_delivery_root", no_cleanup)
    partial = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    assert partial.result.state is RemoteGitDeliveryState.PARTIAL
    if base_removed:
        _git(remote, "update-ref", "-d", request.repository.base_ref)
    else:
        _git(remote, "update-ref", request.repository.base_ref, partial.result.local_commit)
    monkeypatch.setattr(broker, "_cleanup_delivery_root", original)
    recovered = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    assert recovered.result.state is RemoteGitDeliveryState.PUSHED
    assert recovered.result.local_commit == partial.result.local_commit
    assert recovered.result.observed_base_commit == (
        None if base_removed else partial.result.local_commit
    )


@pytest.mark.parametrize("phase", ["prepare", "commit_receipt", "push"])
def test_public_cancellation_survives_diagnostic_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    original_append = broker.repository.append_lifecycle
    original_prepare = broker._prepare_owned
    original_push = broker._push

    async def exercise():
        entered = asyncio.Event()
        cancelling = False

        async def append(request, receipt):
            if cancelling:
                raise OSError("diagnostic write failed")
            if (
                phase == "commit_receipt"
                and receipt.state is RemoteGitDeliveryState.COMMITTED_LOCALLY
            ):
                entered.set()
                await asyncio.Event().wait()
            return await original_append(request, receipt)

        async def prepare(*args, **kwargs):
            if phase == "prepare":
                entered.set()
                await asyncio.Event().wait()
            return await original_prepare(*args, **kwargs)

        async def push(*args, **kwargs):
            result = await original_push(*args, **kwargs)
            if phase == "push":
                entered.set()
                await asyncio.Event().wait()
            return result

        monkeypatch.setattr(broker.repository, "append_lifecycle", append)
        monkeypatch.setattr(broker, "_prepare_owned", prepare)
        monkeypatch.setattr(broker, "_push", push)
        task = asyncio.create_task(
            broker.run(request, product, source_workspace=workspace, approval=approval)
        )
        await asyncio.wait_for(entered.wait(), 5)
        cancelling = True
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert task.cancelled() and task.cancelling() == 1
        cause = caught.value.__cause__
        assert isinstance(cause, BaseExceptionGroup)
        assert len(cause.exceptions) == 1
        assert str(cause.exceptions[0]) == "diagnostic write failed"

    asyncio.run(exercise())
    exists = bool(
        _git(remote, "for-each-ref", "--format=%(refname)", request.repository.destination_ref)
    )
    assert exists is (phase == "push")
    monkeypatch.setattr(broker.repository, "append_lifecycle", original_append)
    monkeypatch.setattr(broker, "_prepare_owned", original_prepare)
    monkeypatch.setattr(broker, "_push", original_push)
    if phase == "push":
        recovered = asyncio.run(
            broker.run(request, product, source_workspace=workspace, approval=approval)
        )
        assert recovered.result.state is RemoteGitDeliveryState.PUSHED
