"""Public delivery regressions for repository, patch and cleanup boundaries."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.core import test_remote_git_delivery as fixtures
from tests.core.test_remote_git_delivery import (
    _broker,
    _coding_publication,
    _git,
    _remote_fixture,
    _request,
)
from tests.core.test_remote_git_delivery_recovery import _case

from cayu.core.tools import ToolContext
from cayu.remote_git_delivery import (
    RemoteGitDeliveryAdmissionError,
    RemoteGitDeliveryState,
    approve_remote_git_delivery,
)
from cayu.runners import LocalRunner
from cayu.tools import GitChangesTool


@pytest.mark.parametrize("changed_source", [False, True])
def test_production_no_change_evidence_requires_the_exact_parent_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_source: bool,
) -> None:
    remote, base = _remote_fixture(tmp_path)
    ctx = ToolContext(session_id="coding-session", runner=LocalRunner(tmp_path / "seed"))
    results = {
        mode: asyncio.run(GitChangesTool().run(ctx, {"mode": mode, "scope": "all"}))
        for mode in ("status", "summary", "diff")
    }
    assert all(not result.is_error for result in results.values())
    assert results["diff"].content == "No textual diff for the selected changes."

    def no_change_events(diff_content=None):
        return tuple(
            fixtures._tool_event(
                "git_changes",
                result.structured,
                event_id=f"git-{mode}",
                content=result.content,
            )
            for mode, result in results.items()
        )

    monkeypatch.setattr(fixtures, "_git_events", no_change_events)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'after'\n" if changed_source else "VALUE = 'before'\n")
    workspace, product, repository, store = asyncio.run(
        _coding_publication(
            tmp_path,
            source,
            final_content=b"VALUE = 'after'\n" if changed_source else b"VALUE = 'before'\n",
        )
    )
    broker = _broker(tmp_path, remote, repository, store)
    request = _request(product, base)
    if changed_source:
        with pytest.raises(RemoteGitDeliveryAdmissionError, match="reviewed delta"):
            asyncio.run(broker.prepare(request, product, source_workspace=workspace))
        assert asyncio.run(broker.repository.load_prepared(request)) is None
        assert not _git(
            remote, "for-each-ref", "--format=%(refname)", request.repository.destination_ref
        )
        return
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    assert prepared.tree == _git(remote, "rev-parse", f"{base}^{{tree}}")
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    delivered = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    assert delivered.result.state is RemoteGitDeliveryState.PUSHED
    assert _git(remote, "rev-parse", f"{delivered.result.local_commit}^") == base
    assert not _git(remote, "diff", base, delivered.result.local_commit)
    assert (
        asyncio.run(broker.run(request, product, source_workspace=workspace, approval=approval))
        == delivered
    )


def test_ancestor_url_rewrite_cannot_redirect_public_delivery(tmp_path: Path) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    _git(tmp_path, "init", "--quiet")
    alternate = tmp_path / "alternate.git"
    _git(tmp_path, "clone", "--bare", str(remote), str(alternate))
    _git(tmp_path, "config", f"url.{alternate}.insteadOf", str(remote))
    # A redirected observation would fail; a redirected push would leave a ref.
    _git(alternate, "update-ref", "-d", "refs/heads/main")
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    result = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    replay = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    assert result == replay
    assert result.result.state is RemoteGitDeliveryState.PUSHED
    assert (
        _git(remote, "rev-parse", request.repository.destination_ref) == result.result.local_commit
    )
    assert not _git(alternate, "for-each-ref", "--format=%(refname)")


def test_same_changed_path_does_not_authorize_a_different_parent(tmp_path: Path) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    seed = tmp_path / "seed"
    (seed / "app.py").write_text("VALUE = 'before'\nOTHER = 'peer change'\n")
    _git(seed, "add", "app.py")
    _git(
        seed,
        "-c",
        "user.name=Peer",
        "-c",
        "user.email=peer@example.invalid",
        "commit",
        "-m",
        "peer",
    )
    _git(seed, "push", str(remote), "HEAD:refs/heads/main")
    changed = request.model_copy(
        update={
            "repository": request.repository.model_copy(
                update={
                    "expected_base_commit": _git(seed, "rev-parse", "HEAD"),
                }
            ),
        }
    )
    with pytest.raises(RemoteGitDeliveryAdmissionError, match="coding baseline"):
        asyncio.run(broker.prepare(changed, product, source_workspace=workspace))
    assert not broker._delivery_root(changed).exists()
    assert asyncio.run(broker.repository.load_prepared(changed)) is None
    assert not _git(
        remote, "for-each-ref", "--format=%(refname)", request.repository.destination_ref
    )


def test_tracked_ignored_manifest_file_is_delivered(tmp_path: Path) -> None:
    remote, _base = _remote_fixture(tmp_path)
    seed = tmp_path / "seed"
    (seed / ".gitignore").write_text("fixtures/\n")
    (seed / "fixtures").mkdir()
    (seed / "fixtures" / "kept.txt").write_text("tracked fixture\n")
    _git(seed, "add", "--force", ".gitignore", "fixtures/kept.txt")
    _git(
        seed,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    _git(seed, "push", str(remote), "HEAD:refs/heads/main")
    source = tmp_path / "source"
    (source / "fixtures").mkdir(parents=True)
    for name in ("app.py", ".gitignore", "fixtures/kept.txt"):
        (source / name).write_bytes((seed / name).read_bytes())
    workspace, product, repository, store = asyncio.run(_coding_publication(tmp_path, source))
    broker = _broker(tmp_path, remote, repository, store)
    request = _request(product, _git(seed, "rev-parse", "HEAD"))
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    result = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=approve_remote_git_delivery(request, prepared, approval_id="approved"),
        )
    )
    assert result.result.state is RemoteGitDeliveryState.PUSHED
    assert (
        _git(remote, "show", f"{result.result.local_commit}:fixtures/kept.txt") == "tracked fixture"
    )


def test_same_parent_and_path_require_the_exact_reviewed_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = fixtures._git_events

    def different_reviewed_delta(diff_content=None):
        assert diff_content is not None
        return original(diff_content.replace("+VALUE = 'after'", "+VALUE = 'different'"))

    monkeypatch.setattr(fixtures, "_git_events", different_reviewed_delta)
    remote, workspace, product, broker, request = _case(tmp_path)
    with pytest.raises(RemoteGitDeliveryAdmissionError, match="reviewed delta"):
        asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    assert asyncio.run(broker.repository.load_prepared(request)) is None
    assert not _git(
        remote, "for-each-ref", "--format=%(refname)", request.repository.destination_ref
    )


@pytest.mark.parametrize("after_write", [False, True])
def test_terminal_cleanup_publication_loss_recovers_after_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_write: bool,
) -> None:
    _remote, workspace, product, broker, request = _case(tmp_path)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved").model_copy(
        update={"push_approved": False}
    )
    original = broker.repository.publish_result

    async def lose_cleanup_ack(request, result):
        if result.cleanup_settled:
            if after_write:
                await original(request, result)
            raise OSError("cleanup receipt acknowledgement lost")
        return await original(request, result)

    monkeypatch.setattr(broker.repository, "publish_result", lose_cleanup_ack)
    with pytest.raises(OSError, match="acknowledgement lost"):
        asyncio.run(broker.run(request, product, source_workspace=workspace, approval=approval))
    assert not broker._delivery_root(request).exists()
    monkeypatch.setattr(broker.repository, "publish_result", original)
    recovered = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    assert recovered.result.state is RemoteGitDeliveryState.DENIED
    assert recovered.result.cleanup_settled


@pytest.mark.parametrize("outcome", ["denied", "conflict", "failed"])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_definitive_outcomes_reclaim_private_repository_on_exact_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    cleanup_failure: bool,
) -> None:
    remote, workspace, product, broker, request = _case(tmp_path)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approved")
    if outcome == "denied":
        approval = approval.model_copy(update={"push_approved": False})
    elif outcome == "conflict":
        _git(
            remote,
            "update-ref",
            request.repository.destination_ref,
            request.repository.expected_base_commit,
        )
    else:
        hook = remote / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o700)
    original_cleanup = broker._cleanup_delivery_root

    async def fail_cleanup(*args):
        raise OSError("cleanup failed")

    if cleanup_failure:
        monkeypatch.setattr(broker, "_cleanup_delivery_root", fail_cleanup)
    result = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    assert result.result.state.value == outcome
    assert result.result.cleanup_settled is not cleanup_failure
    assert broker._delivery_root(request).exists() is cleanup_failure
    monkeypatch.setattr(broker, "_cleanup_delivery_root", original_cleanup)

    async def no_git(*args, **kwargs):
        raise AssertionError("terminal cleanup must not dispatch Git")

    monkeypatch.setattr(broker, "_git", no_git)
    recovered = asyncio.run(
        broker.run(request, product, source_workspace=workspace, approval=approval)
    )
    assert recovered.result.state.value == outcome
    assert recovered.result.cleanup_settled
    assert not broker._delivery_root(request).exists()
    assert (
        asyncio.run(broker.run(request, product, source_workspace=workspace, approval=approval))
        == recovered
    )
