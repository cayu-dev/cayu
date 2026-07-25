from __future__ import annotations

import asyncio
import errno
import json
import multiprocessing
import os
import stat
import threading
import time
import traceback
from base64 import urlsafe_b64encode
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from tests.provider_traceback_assertions import (
    assert_cayu_traceback_does_not_retain,
    is_cayu_source_filename,
)

import cayu.providers.openai_subscription as subscription_auth
from cayu.providers.openai_subscription import (
    OPENAI_SUBSCRIPTION_OAUTH_CLIENT_ID,
    HttpxOpenAISubscriptionOAuthTransport,
    OpenAISubscriptionAuth,
    OpenAISubscriptionAuthError,
    OpenAISubscriptionAuthStore,
    OpenAISubscriptionCredentials,
    build_openai_subscription_authorize_url,
    openai_subscription_credentials_from_token_response,
)


def _assert_auth_store_traceback_does_not_retain_credentials(
    exc: BaseException,
    credentials: OpenAISubscriptionCredentials,
) -> None:
    secrets = tuple(
        value
        for value in (
            credentials.access_token,
            credentials.refresh_token,
            credentials.account_id,
        )
        if value is not None
    )

    def retains_credentials(value: object, seen: set[int]) -> bool:
        if value is credentials:
            return True
        if isinstance(value, str):
            return any(secret in value for secret in secrets)
        identity = id(value)
        if identity in seen:
            return False
        seen.add(identity)
        if isinstance(value, dict):
            return any(retains_credentials(item, seen) for pair in value.items() for item in pair)
        if isinstance(value, (list, tuple, set, frozenset)):
            return any(retains_credentials(item, seen) for item in value)
        return False

    retained_frames = [
        frame.f_code.co_name
        for frame, _line_number in traceback.walk_tb(exc.__traceback__)
        if is_cayu_source_filename(frame.f_code.co_filename)
        and any(retains_credentials(value, set()) for value in frame.f_locals.values())
    ]
    assert retained_frames == []


def _serialized_auth_store_save(
    auth_path: str,
    access_token: str,
    refresh_token: str,
    expires_at: float,
    started,
    release,
    lock_attempted=None,
    completed=None,
) -> None:
    store = OpenAISubscriptionAuthStore(auth_path)
    credentials = OpenAISubscriptionCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
    )
    if release is None:
        if lock_attempted is not None:
            import fcntl

            real_flock = fcntl.flock

            def recording_flock(descriptor: int, operation: int) -> None:
                if operation & fcntl.LOCK_EX:
                    lock_attempted.set()
                real_flock(descriptor, operation)

            fcntl.flock = recording_flock
        started.set()
        try:
            store.save(credentials)
        finally:
            if completed is not None:
                completed.set()
        return
    with store._exclusive_lock() as directory_fd:
        started.set()
        if not release.wait(timeout=5):
            raise AssertionError("auth-store concurrency test did not release first writer")
        store._save_credentials_unlocked(credentials, directory_fd)


def _load_auth_store_and_report(auth_path: str, connection) -> None:
    try:
        OpenAISubscriptionAuthStore(auth_path).load()
    except Exception as exc:
        connection.send((type(exc).__name__, str(exc)))
    else:
        connection.send(("success", ""))
    finally:
        connection.close()


def _crash_auth_store_before_replace(
    auth_path: str,
    credentials: OpenAISubscriptionCredentials,
) -> None:
    def exit_before_replace(*_args, **_kwargs) -> None:
        os._exit(73)

    subscription_auth.os.replace = exit_before_replace
    OpenAISubscriptionAuthStore(auth_path).save(credentials)


def _save_auth_store_then_exit(
    auth_path: str,
    credentials: OpenAISubscriptionCredentials,
) -> None:
    OpenAISubscriptionAuthStore(auth_path).save(credentials)
    os._exit(0)


def _crash_auth_store_after_parent_creation(auth_path: str) -> None:
    subscription_auth._sync_auth_store_directory_chain = lambda *_args, **_kwargs: os._exit(74)
    OpenAISubscriptionAuthStore(auth_path).save(
        OpenAISubscriptionCredentials(
            access_token="crashed-access",
            refresh_token="crashed-refresh",
            expires_at=2_000_000_000,
        )
    )


def _save_auth_store_and_report_directory_syncs(auth_path: str, connection) -> None:
    synchronized: list[str] = []
    real_sync = subscription_auth._sync_auth_store_directory_path

    def recording_sync(path: Path) -> None:
        synchronized.append(str(path))
        real_sync(path)

    subscription_auth._sync_auth_store_directory_path = recording_sync
    try:
        OpenAISubscriptionAuthStore(auth_path).save(
            OpenAISubscriptionCredentials(
                access_token="recovered-access",
                refresh_token="recovered-refresh",
                expires_at=2_100_000_000,
            )
        )
        connection.send(("success", synchronized))
    except BaseException as exc:
        connection.send((type(exc).__name__, str(exc)))
    finally:
        connection.close()


def _save_auth_store_with_umask(auth_path: str, mask: int, connection) -> None:
    old_umask = os.umask(mask)
    try:
        path = Path(auth_path)
        OpenAISubscriptionAuthStore(path).save(
            OpenAISubscriptionCredentials(
                access_token="access",
                refresh_token="refresh",
                expires_at=2_000_000_000,
            )
        )
        connection.send(
            (
                "success",
                [
                    (stat.S_IMODE(candidate.stat().st_mode), candidate.stat().st_uid)
                    for candidate in (path.parent.parent, path.parent, path)
                ],
            )
        )
    except BaseException as exc:
        connection.send((type(exc).__name__, str(exc)))
    finally:
        os.umask(old_umask)
        connection.close()


def _delete_auth_store_and_report(auth_path: str, lock_attempted, connection) -> None:
    import fcntl

    real_flock = fcntl.flock

    def recording_flock(descriptor: int, operation: int) -> None:
        if operation & fcntl.LOCK_EX:
            lock_attempted.set()
        real_flock(descriptor, operation)

    fcntl.flock = recording_flock
    try:
        connection.send(("success", OpenAISubscriptionAuthStore(auth_path).delete()))
    except BaseException as exc:
        connection.send((type(exc).__name__, str(exc)))
    finally:
        connection.close()


def _run_auth_store_operation_after_lock_attempt(
    auth_path: str,
    operation: str,
    lock_attempted,
    connection,
) -> None:
    import fcntl

    real_flock = fcntl.flock

    def recording_flock(descriptor: int, lock_operation: int) -> None:
        if lock_operation & fcntl.LOCK_EX:
            lock_attempted.set()
        real_flock(descriptor, lock_operation)

    fcntl.flock = recording_flock
    store = OpenAISubscriptionAuthStore(auth_path)
    try:
        if operation == "load":
            result = store.load()
            connection.send(("success", result is not None))
        elif operation == "delete":
            connection.send(("success", store.delete()))
        else:
            raise AssertionError(f"unsupported auth-store operation: {operation}")
    except BaseException as exc:
        connection.send((type(exc).__name__, str(exc)))
    finally:
        connection.close()


def _join_auth_store_process(process) -> None:
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        raise AssertionError("auth-store subprocess did not terminate")


def test_subscription_credentials_round_trip_through_private_auth_store(tmp_path: Path) -> None:
    auth_path = tmp_path / "cayu" / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    credentials = OpenAISubscriptionCredentials(
        access_token="access-token",
        refresh_token="refresh-token",
        expires_at=2_000_000_000.0,
        account_id="acct-cayu",
    )

    store.save(credentials)

    assert store.load() == credentials
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600
    persisted = json.loads(auth_path.read_text(encoding="utf-8"))
    assert persisted["version"] == 1
    assert persisted["providers"]["openai_subscription"]["account_id"] == "acct-cayu"


def test_auth_store_preserves_permissions_of_existing_parent(tmp_path: Path) -> None:
    auth_home = tmp_path / "shared-cayu-home"
    auth_home.mkdir(mode=0o755)
    auth_home.chmod(0o755)

    assert OpenAISubscriptionAuthStore(auth_home / "auth.json").load() is None

    assert stat.S_IMODE(auth_home.stat().st_mode) == 0o755
    assert list(auth_home.iterdir()) == []


def test_auth_store_rejects_symlinked_existing_parent_without_side_effects(
    tmp_path: Path,
) -> None:
    target = tmp_path / "auth-target"
    target.mkdir()
    auth_home = tmp_path / "cayu-home"
    auth_home.symlink_to(target, target_is_directory=True)

    try:
        OpenAISubscriptionAuthStore(auth_home / "auth.json").load()
    except ValueError as exc:
        assert str(exc) == "Refusing to use a symlinked Cayu auth-store directory."
        assert str(auth_home) not in repr(exc)
    else:
        raise AssertionError("symlinked auth store directory must fail validation")

    assert list(target.iterdir()) == []


def test_subscription_credentials_repr_redacts_tokens() -> None:
    credentials = OpenAISubscriptionCredentials(
        access_token="secret-access-token",
        refresh_token="secret-refresh-token",
        expires_at=2_000_000_000.0,
        account_id="acct-cayu",
    )

    rendered = repr(credentials)

    assert "secret-access-token" not in rendered
    assert "secret-refresh-token" not in rendered
    assert "acct-cayu" not in rendered


def test_oauth_error_projection_omits_untrusted_credential_shaped_detail(monkeypatch) -> None:
    canaries = {
        "access": "provider-access-canary-0123456789",
        "refresh": "provider-refresh-canary-0123456789",
        "account": "provider-account-canary-0123456789",
        "header": "Bearer provider-header-canary-0123456789",
    }

    def fake_post(_url: str, **_kwargs: Any) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": "invalid_grant",
                "error_description": " ".join(canaries.values()),
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(OpenAISubscriptionAuthError) as exc_info:
        HttpxOpenAISubscriptionOAuthTransport().refresh(canaries["refresh"])

    rendered = str(exc_info.value) + repr(exc_info.value)
    assert rendered == (
        "OpenAI OAuth request failed with HTTP 401."
        "OpenAISubscriptionAuthError('OpenAI OAuth request failed with HTTP 401.')"
    )
    assert all(value not in rendered for value in canaries.values())


def test_auth_store_parse_failure_omits_path_content_and_exception_graph(tmp_path: Path) -> None:
    canary = "provider-auth-store-canary-0123456789"
    auth_path = tmp_path / canary / "auth.json"
    auth_path.parent.mkdir()
    auth_path.write_text(f'{{"access_token":"{canary}"', encoding="utf-8")
    auth_path.chmod(0o600)

    with pytest.raises(ValueError) as exc_info:
        OpenAISubscriptionAuthStore(auth_path).load()

    retained = repr(exc_info.value) + repr(vars(exc_info.value))
    assert str(exc_info.value) == "Could not read Cayu auth store."
    assert canary not in retained
    assert str(auth_path) not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_auth_store_syncs_file_then_directory_for_creation_and_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "cayu-home" / "auth.json"
    events: list[tuple[str, str | None]] = []
    real_sync = subscription_auth._sync_auth_store_descriptor
    real_replace = os.replace

    def recording_sync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        kind = "directory" if stat.S_ISDIR(mode) else "regular"
        events.append(("fsync", kind))
        real_sync(descriptor)

    def recording_replace(src, dst, **kwargs) -> None:
        staged = os.stat(
            src,
            dir_fd=kwargs["src_dir_fd"],
            follow_symlinks=False,
        )
        assert stat.S_ISREG(staged.st_mode)
        assert stat.S_IMODE(staged.st_mode) == 0o600
        assert staged.st_uid == os.geteuid()
        events.append(("replace", None))
        real_replace(src, dst, **kwargs)

    monkeypatch.setattr(subscription_auth, "_sync_auth_store_descriptor", recording_sync)
    monkeypatch.setattr(subscription_auth.os, "replace", recording_replace)
    store = OpenAISubscriptionAuthStore(auth_path)

    first = OpenAISubscriptionCredentials(
        access_token="first-access",
        refresh_token="first-refresh",
        expires_at=2_000_000_000,
    )
    store.save(first)

    first_replace = events.index(("replace", None))
    assert ("fsync", "regular") in events[:first_replace]
    assert ("fsync", "directory") in events[first_replace + 1 :]
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600

    events.clear()
    second = OpenAISubscriptionCredentials(
        access_token="second-access",
        refresh_token="second-refresh",
        expires_at=2_100_000_000,
    )
    store.save(second)

    second_replace = events.index(("replace", None))
    assert events[:second_replace].count(("fsync", "regular")) == 1
    assert ("fsync", "directory") in events[second_replace + 1 :]
    assert store.load() == second

    events.clear()
    assert store.delete()

    delete_replace = events.index(("replace", None))
    assert events[:delete_replace].count(("fsync", "regular")) == 1
    assert ("fsync", "directory") in events[delete_replace + 1 :]
    assert store.load() is None


@pytest.mark.skipif(
    not subscription_auth._SUPPORTS_DURABLE_AUTH_STORE,
    reason="requires POSIX durable-store primitives",
)
def test_auth_store_uses_full_sync_for_files_and_directories_on_darwin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fcntl = pytest.importorskip("fcntl")
    full_sync_command = getattr(fcntl, "F_FULLFSYNC", 51)
    calls: list[tuple[str, int]] = []

    def recording_fcntl(descriptor: int, command: int) -> int:
        mode = os.fstat(descriptor).st_mode
        kind = "directory" if stat.S_ISDIR(mode) else "regular"
        calls.append((kind, command))
        return 0

    monkeypatch.setattr(subscription_auth.sys, "platform", "darwin")
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", full_sync_command, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", recording_fcntl)
    monkeypatch.setattr(
        subscription_auth.os,
        "fsync",
        lambda _descriptor: pytest.fail("Darwin durability must not use plain fsync"),
    )

    OpenAISubscriptionAuthStore(tmp_path / "cayu-home" / "auth.json").save(
        OpenAISubscriptionCredentials(
            access_token="access",
            refresh_token="refresh",
            expires_at=2_000_000_000,
        )
    )

    assert ("regular", full_sync_command) in calls
    regular_sync = calls.index(("regular", full_sync_command))
    assert ("directory", full_sync_command) in calls[:regular_sync]
    assert ("directory", full_sync_command) in calls[regular_sync + 1 :]
    assert all(command == full_sync_command for _, command in calls)


@pytest.mark.skipif(
    not subscription_auth._SUPPORTS_DURABLE_AUTH_STORE,
    reason="requires POSIX durable-store primitives",
)
def test_auth_store_marks_darwin_unsupported_without_full_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fcntl = pytest.importorskip("fcntl")
    monkeypatch.setattr(subscription_auth.sys, "platform", "darwin")
    monkeypatch.delattr(fcntl, "F_FULLFSYNC", raising=False)

    assert not subscription_auth._supports_durable_auth_store()


@pytest.mark.skipif(
    not subscription_auth._SUPPORTS_DURABLE_AUTH_STORE,
    reason="requires POSIX durable-store primitives",
)
@pytest.mark.parametrize("failure_point", ["preflight", "post_publish"])
def test_auth_store_darwin_full_sync_failure_preserves_complete_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    fcntl = pytest.importorskip("fcntl")
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    original = OpenAISubscriptionCredentials(
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at=2_000_000_000,
    )
    replacement = OpenAISubscriptionCredentials(
        access_token="new-access",
        refresh_token="new-refresh",
        expires_at=2_100_000_000,
    )
    store.save(original)
    full_sync_command = getattr(fcntl, "F_FULLFSYNC", 51)
    real_replace = os.replace
    published = False

    def recording_replace(src, dst, **kwargs) -> None:
        nonlocal published
        real_replace(src, dst, **kwargs)
        published = True

    def fail_selected_full_sync(descriptor: int, command: int) -> int:
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        should_fail = (failure_point == "preflight" and not published and is_directory) or (
            failure_point == "post_publish" and published and is_directory
        )
        if should_fail:
            raise OSError("injected full-sync failure with new-refresh")
        assert command == full_sync_command
        return 0

    monkeypatch.setattr(subscription_auth.sys, "platform", "darwin")
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", full_sync_command, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", fail_selected_full_sync)
    monkeypatch.setattr(subscription_auth.os, "replace", recording_replace)

    with pytest.raises(
        ValueError,
        match="Cayu auth store could not be made durable",
    ) as exc_info:
        store.save(replacement)

    expected = original if failure_point == "preflight" else replacement
    assert OpenAISubscriptionAuthStore(auth_path).load() == expected
    retained = repr(exc_info.value) + repr(vars(exc_info.value))
    assert "new-refresh" not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_auth_store_parent_creation_sync_failure_does_not_publish_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "nested" / "cayu-home" / "auth.json"
    monkeypatch.setattr(
        subscription_auth,
        "_sync_auth_store_directory_path",
        lambda _path: (_ for _ in ()).throw(
            ValueError("Cayu auth store could not be made durable.")
        ),
    )

    with pytest.raises(
        ValueError,
        match="Cayu auth store could not be made durable",
    ):
        OpenAISubscriptionAuthStore(auth_path).save(
            OpenAISubscriptionCredentials(
                access_token="access",
                refresh_token="refresh",
                expires_at=2_000_000_000,
            )
        )

    assert not auth_path.exists()
    assert list(auth_path.parent.glob(".auth.json.tmp-*")) == []


def test_auth_store_concurrent_parent_creation_is_verified_and_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "nested" / "cayu-home" / "auth.json"
    real_mkdir = os.mkdir
    raced = False

    def race_parent_creation(path, mode=0o777, *, dir_fd=None) -> None:
        nonlocal raced
        if not raced and path == "nested" and dir_fd is not None:
            raced = True
            real_mkdir(path, mode=0o700, dir_fd=dir_fd)
            raise FileExistsError
        real_mkdir(path, mode=mode, dir_fd=dir_fd)

    monkeypatch.setattr(subscription_auth.os, "mkdir", race_parent_creation)

    OpenAISubscriptionAuthStore(auth_path)._prepare_parent()

    assert raced
    assert stat.S_IMODE((tmp_path / "nested").stat().st_mode) == 0o700
    assert stat.S_IMODE(auth_path.parent.stat().st_mode) == 0o700
    assert auth_path.parent.is_dir()


def test_auth_store_rejects_unsafe_concurrently_created_parent_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "nested" / "cayu-home" / "auth.json"
    real_mkdir = os.mkdir
    raced = False

    def race_unsafe_parent_creation(path, mode=0o777, *, dir_fd=None) -> None:
        nonlocal raced
        if not raced and path == "nested" and dir_fd is not None:
            raced = True
            real_mkdir(path, mode=0o700, dir_fd=dir_fd)
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=dir_fd,
            )
            try:
                os.fchmod(descriptor, 0o777)
            finally:
                os.close(descriptor)
            raise FileExistsError
        real_mkdir(path, mode=mode, dir_fd=dir_fd)

    monkeypatch.setattr(subscription_auth.os, "mkdir", race_unsafe_parent_creation)

    with pytest.raises(
        ValueError,
        match="Cayu auth-store directory has unsafe permissions or ownership",
    ):
        OpenAISubscriptionAuthStore(auth_path).save(
            OpenAISubscriptionCredentials(
                access_token="access",
                refresh_token="refresh",
                expires_at=2_000_000_000,
            )
        )

    assert raced
    assert stat.S_IMODE((tmp_path / "nested").stat().st_mode) == 0o777
    assert not auth_path.exists()


def test_auth_store_preserves_existing_ancestor_symlink_compatibility(tmp_path: Path) -> None:
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    alias_home = tmp_path / "alias-home"
    alias_home.symlink_to(real_home, target_is_directory=True)
    auth_path = alias_home / "cayu" / "auth.json"
    expected = OpenAISubscriptionCredentials(
        access_token="access",
        refresh_token="refresh",
        expires_at=2_000_000_000,
    )

    OpenAISubscriptionAuthStore(auth_path).save(expected)

    assert alias_home.is_symlink()
    assert OpenAISubscriptionAuthStore(auth_path).load() == expected
    assert stat.S_IMODE((real_home / "cayu" / "auth.json").stat().st_mode) == 0o600


def test_auth_store_file_sync_failure_preserves_old_credentials_and_removes_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    original = OpenAISubscriptionCredentials(
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at=2_000_000_000,
    )
    replacement = OpenAISubscriptionCredentials(
        access_token="new-access",
        refresh_token="new-refresh",
        expires_at=2_100_000_000,
    )
    store.save(original)
    real_sync = subscription_auth._sync_auth_store_descriptor

    def fail_regular_file_sync(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("injected file sync failure with new-refresh")
        real_sync(descriptor)

    monkeypatch.setattr(
        subscription_auth,
        "_sync_auth_store_descriptor",
        fail_regular_file_sync,
    )

    with pytest.raises(ValueError, match="Could not access Cayu auth store") as exc_info:
        store.save(replacement)

    assert store.load() == original
    assert list(tmp_path.glob(".auth.json.tmp-*")) == []
    retained = repr(exc_info.value) + repr(vars(exc_info.value))
    assert "new-refresh" not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_auth_store_initial_staging_stat_failure_closes_descriptor_and_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    original = OpenAISubscriptionCredentials(
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at=2_000_000_000,
    )
    replacement = OpenAISubscriptionCredentials(
        access_token="new-access",
        refresh_token="new-refresh",
        expires_at=2_100_000_000,
    )
    store.save(original)
    real_open_temporary = subscription_auth._open_auth_store_temporary
    real_fstat = os.fstat
    staging_descriptor: int | None = None

    def record_open_temporary(leaf_name: str, *, directory_fd: int) -> tuple[int, str]:
        nonlocal staging_descriptor
        result = real_open_temporary(leaf_name, directory_fd=directory_fd)
        staging_descriptor = result[0]
        return result

    def fail_initial_staging_stat(descriptor: int) -> os.stat_result:
        if descriptor == staging_descriptor:
            raise OSError("injected initial staging stat failure with new-refresh")
        return real_fstat(descriptor)

    monkeypatch.setattr(
        subscription_auth,
        "_open_auth_store_temporary",
        record_open_temporary,
    )
    monkeypatch.setattr(subscription_auth.os, "fstat", fail_initial_staging_stat)

    with pytest.raises(ValueError, match="Could not access Cayu auth store") as exc_info:
        store.save(replacement)

    assert staging_descriptor is not None
    with pytest.raises(OSError) as closed_error:
        real_fstat(staging_descriptor)
    assert closed_error.value.errno == errno.EBADF
    monkeypatch.setattr(subscription_auth.os, "fstat", real_fstat)
    assert store.load() == original
    retained = repr(exc_info.value) + repr(vars(exc_info.value))
    assert "new-refresh" not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None

    store.save(replacement)

    assert store.load() == replacement


@pytest.mark.parametrize("failure_point", ["write", "replace"])
def test_auth_store_prepublication_failure_preserves_old_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    original = OpenAISubscriptionCredentials(
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at=2_000_000_000,
    )
    replacement = OpenAISubscriptionCredentials(
        access_token="new-access",
        refresh_token="new-refresh-canary",
        expires_at=2_100_000_000,
    )
    store.save(original)

    if failure_point == "write":
        monkeypatch.setattr(
            subscription_auth,
            "_write_auth_store_payload",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected write failure with new-refresh-canary")
            ),
        )
    else:
        monkeypatch.setattr(
            subscription_auth.os,
            "replace",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected replace failure with new-refresh-canary")
            ),
        )

    with pytest.raises(ValueError, match="Could not access Cayu auth store") as exc_info:
        store.save(replacement)

    assert OpenAISubscriptionAuthStore(auth_path).load() == original
    assert list(tmp_path.glob(".auth.json.tmp-*")) == []
    retained = repr(exc_info.value) + repr(vars(exc_info.value))
    assert "new-refresh-canary" not in retained


def test_auth_store_directory_sync_failure_reports_failure_but_keeps_complete_new_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    original = OpenAISubscriptionCredentials(
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at=2_000_000_000,
    )
    replacement = OpenAISubscriptionCredentials(
        access_token="new-access-canary",
        refresh_token="new-refresh-canary",
        expires_at=2_100_000_000,
    )
    store.save(original)
    published = False
    real_sync = subscription_auth._sync_auth_store_descriptor
    real_replace = os.replace

    def recording_replace(src, dst, **kwargs) -> None:
        nonlocal published
        real_replace(src, dst, **kwargs)
        published = True

    def fail_post_publish_directory_sync(descriptor: int) -> None:
        if published and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("Cayu auth store could not be made durable.")
        real_sync(descriptor)

    monkeypatch.setattr(subscription_auth.os, "replace", recording_replace)
    monkeypatch.setattr(
        subscription_auth,
        "_sync_auth_store_descriptor",
        fail_post_publish_directory_sync,
    )

    with pytest.raises(
        ValueError,
        match="Cayu auth store could not be made durable",
    ) as exc_info:
        store.save(replacement)

    assert OpenAISubscriptionAuthStore(auth_path).load() == replacement
    retained = repr(exc_info.value) + repr(vars(exc_info.value))
    assert str(auth_path) not in retained
    assert "new-refresh-canary" not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_auth_store_detects_destination_substitution_before_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    replacement = OpenAISubscriptionCredentials(
        access_token="new-access",
        refresh_token="new-refresh",
        expires_at=2_100_000_000,
    )
    real_sync = subscription_auth._sync_auth_store_directory
    real_replace = os.replace
    published = False

    def substitute_before_post_publish_sync(directory_fd: int) -> None:
        if published:
            attacker = tmp_path / "attacker"
            attacker.write_text("operator-content", encoding="utf-8")
            attacker.chmod(0o600)
            os.replace(attacker, auth_path)
        real_sync(directory_fd)

    def recording_replace(src, dst, **kwargs) -> None:
        nonlocal published
        real_replace(src, dst, **kwargs)
        published = True

    monkeypatch.setattr(subscription_auth.os, "replace", recording_replace)
    monkeypatch.setattr(
        subscription_auth,
        "_sync_auth_store_directory",
        substitute_before_post_publish_sync,
    )

    with pytest.raises(
        ValueError,
        match="Cayu auth store could not be made durable",
    ):
        store.save(replacement)

    assert auth_path.read_text(encoding="utf-8") == "operator-content"


def test_auth_store_ignores_orphaned_staging_files(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    expected = OpenAISubscriptionCredentials(
        access_token="current-access",
        refresh_token="current-refresh",
        expires_at=2_000_000_000,
    )
    store = OpenAISubscriptionAuthStore(auth_path)
    store.save(expected)
    orphan = tmp_path / ".auth.json.tmp-crashed"
    orphan.write_text('{"refresh_token":"orphan-canary"', encoding="utf-8")
    orphan.chmod(0o600)
    symlink_orphan = tmp_path / ".auth.json.tmp-symlink"
    symlink_orphan.symlink_to(orphan)

    assert OpenAISubscriptionAuthStore(auth_path).load() == expected
    assert orphan.exists()
    assert symlink_orphan.is_symlink()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
def test_auth_store_rejects_fifo_without_blocking_and_releases_lock(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    os.mkfifo(auth_path, 0o600)
    auth_path.chmod(0o600)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_load_auth_store_and_report,
        args=(str(auth_path), sender),
    )

    process.start()
    sender.close()
    _join_auth_store_process(process)

    assert process.exitcode == 0
    assert receiver.poll(timeout=1)
    assert receiver.recv() == (
        "ValueError",
        "Cayu auth store has unsafe permissions or ownership.",
    )
    receiver.close()

    auth_path.unlink()
    expected = OpenAISubscriptionCredentials(
        access_token="access",
        refresh_token="refresh",
        expires_at=2_000_000_000,
    )
    OpenAISubscriptionAuthStore(auth_path).save(expected)

    assert OpenAISubscriptionAuthStore(auth_path).load() == expected


def test_auth_store_rejects_symlinked_file_without_touching_target(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    target = tmp_path / "operator-file"
    target.write_text("untouched", encoding="utf-8")
    target.chmod(0o600)
    auth_path.symlink_to(target)

    with pytest.raises(
        ValueError,
        match="Refusing to read a symlinked Cayu auth store",
    ):
        OpenAISubscriptionAuthStore(auth_path).load()

    assert target.read_text(encoding="utf-8") == "untouched"


def test_auth_store_non_symlink_open_failure_reports_bounded_read_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = "auth-open-failure-canary-0123456789"
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    store.save(
        OpenAISubscriptionCredentials(
            access_token="access",
            refresh_token="refresh",
            expires_at=2_000_000_000,
        )
    )
    real_open = os.open

    def fail_auth_file_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is not None and path == auth_path.name and not flags & os.O_CREAT:
            raise OSError(errno.EIO, f"read failed near {canary}")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(subscription_auth.os, "open", fail_auth_file_open)

    with pytest.raises(ValueError, match="Could not read Cayu auth store") as exc_info:
        store.load()

    retained = repr(exc_info.value) + repr(vars(exc_info.value))
    assert canary not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_auth_store_rejects_symlinked_lock_without_touching_target(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    target = tmp_path / "operator-file"
    target.write_text("untouched", encoding="utf-8")
    (tmp_path / ".auth.json.lock").symlink_to(target)

    with pytest.raises(ValueError, match="Could not access Cayu auth store"):
        OpenAISubscriptionAuthStore(auth_path).save(
            OpenAISubscriptionCredentials(
                access_token="access",
                refresh_token="refresh",
                expires_at=2_000_000_000,
            )
        )

    assert target.read_text(encoding="utf-8") == "untouched"
    assert not auth_path.exists()


def test_auth_store_rejects_hardlinked_lock_without_changing_target_mode(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    target = tmp_path / "operator-file"
    target.write_text("untouched", encoding="utf-8")
    target.chmod(0o640)
    os.link(target, tmp_path / ".auth.json.lock")

    with pytest.raises(
        ValueError,
        match="Cayu auth store has unsafe permissions or ownership",
    ):
        OpenAISubscriptionAuthStore(auth_path).save(
            OpenAISubscriptionCredentials(
                access_token="access",
                refresh_token="refresh",
                expires_at=2_000_000_000,
            )
        )

    assert target.read_text(encoding="utf-8") == "untouched"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert not auth_path.exists()


def test_auth_store_rejects_writable_parent_directory(tmp_path: Path) -> None:
    auth_home = tmp_path / "auth-home"
    auth_home.mkdir()
    auth_home.chmod(0o777)

    with pytest.raises(
        ValueError,
        match="Cayu auth-store directory has unsafe permissions or ownership",
    ):
        OpenAISubscriptionAuthStore(auth_home / "auth.json").save(
            OpenAISubscriptionCredentials(
                access_token="access",
                refresh_token="refresh",
                expires_at=2_000_000_000,
            )
        )

    assert list(auth_home.iterdir()) == []


def test_auth_store_rejects_existing_file_with_unsafe_mode(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"version":1,"providers":{}}\n', encoding="utf-8")
    auth_path.chmod(0o644)

    with pytest.raises(
        ValueError,
        match="Cayu auth store has unsafe permissions or ownership",
    ):
        OpenAISubscriptionAuthStore(auth_path).load()


def test_auth_store_parent_substitution_cannot_redirect_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_home = tmp_path / "auth-home"
    moved_home = tmp_path / "auth-home-original"
    alternate = tmp_path / "alternate"
    auth_home.mkdir()
    alternate.mkdir()
    sentinel = alternate / "operator-file"
    sentinel.write_text("untouched", encoding="utf-8")
    real_open = os.open
    swapped = False

    def swap_after_parent_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if not swapped and dir_fd is None and Path(path) == auth_home:
            swapped = True
            auth_home.rename(moved_home)
            auth_home.symlink_to(alternate, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(subscription_auth.os, "open", swap_after_parent_open)

    with pytest.raises(
        ValueError,
        match="Refusing to use a symlinked Cayu auth-store directory",
    ):
        OpenAISubscriptionAuthStore(auth_home / "auth.json").save(
            OpenAISubscriptionCredentials(
                access_token="access",
                refresh_token="refresh",
                expires_at=2_000_000_000,
            )
        )

    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert list(alternate.iterdir()) == [sentinel]
    assert not (moved_home / "auth.json").exists()


def test_auth_store_parent_substitution_after_replace_cannot_be_acknowledged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_home = tmp_path / "auth-home"
    moved_home = tmp_path / "auth-home-original"
    alternate = tmp_path / "alternate"
    auth_home.mkdir()
    alternate.mkdir()
    sentinel = alternate / "operator-file"
    sentinel.write_text("untouched", encoding="utf-8")
    real_sync = subscription_auth._sync_auth_store_directory
    real_replace = os.replace
    published = False

    def swap_before_commit_sync(directory_fd: int) -> None:
        if published:
            auth_home.rename(moved_home)
            auth_home.symlink_to(alternate, target_is_directory=True)
        real_sync(directory_fd)

    def recording_replace(src, dst, **kwargs) -> None:
        nonlocal published
        real_replace(src, dst, **kwargs)
        published = True

    monkeypatch.setattr(subscription_auth.os, "replace", recording_replace)
    monkeypatch.setattr(
        subscription_auth,
        "_sync_auth_store_directory",
        swap_before_commit_sync,
    )

    with pytest.raises(
        ValueError,
        match="Refusing to use a symlinked Cayu auth-store directory",
    ):
        OpenAISubscriptionAuthStore(auth_home / "auth.json").save(
            OpenAISubscriptionCredentials(
                access_token="access",
                refresh_token="refresh",
                expires_at=2_000_000_000,
            )
        )

    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert list(alternate.iterdir()) == [sentinel]
    assert (moved_home / "auth.json").is_file()


def test_auth_store_concurrent_processes_are_complete_and_last_waiter_wins(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    store.save(
        OpenAISubscriptionCredentials(
            access_token="seed-access",
            refresh_token="seed-refresh",
            expires_at=2_000_000_000,
        )
    )
    document = json.loads(auth_path.read_text(encoding="utf-8"))
    document["providers"]["independent"] = {"opaque": "preserved"}
    auth_path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    auth_path.chmod(0o600)

    context = multiprocessing.get_context("spawn")
    first_started = context.Event()
    release_first = context.Event()
    second_started = context.Event()
    second_lock_attempted = context.Event()
    second_completed = context.Event()
    first = context.Process(
        target=_serialized_auth_store_save,
        args=(
            str(auth_path),
            "first-access",
            "first-refresh",
            2_100_000_000,
            first_started,
            release_first,
        ),
    )
    second = context.Process(
        target=_serialized_auth_store_save,
        args=(
            str(auth_path),
            "second-access",
            "second-refresh",
            2_200_000_000,
            second_started,
            None,
            second_lock_attempted,
            second_completed,
        ),
    )
    first.start()
    assert first_started.wait(timeout=5)
    second.start()
    assert second_started.wait(timeout=5)
    assert second_lock_attempted.wait(timeout=5)
    assert not second_completed.wait(timeout=0.2)
    release_first.set()
    _join_auth_store_process(first)
    _join_auth_store_process(second)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert second_completed.is_set()
    assert store.load() == OpenAISubscriptionCredentials(
        access_token="second-access",
        refresh_token="second-refresh",
        expires_at=2_200_000_000,
    )
    persisted = json.loads(auth_path.read_text(encoding="utf-8"))
    assert persisted["providers"]["independent"] == {"opaque": "preserved"}


@pytest.mark.skipif(
    not subscription_auth._SUPPORTS_DURABLE_AUTH_STORE,
    reason="requires POSIX durable-store primitives",
)
@pytest.mark.parametrize("operation", ["load", "delete"])
def test_auth_store_revalidates_directory_after_process_lock_wait(
    tmp_path: Path,
    operation: str,
) -> None:
    import fcntl

    auth_home = tmp_path / "auth-home"
    moved_home = tmp_path / "auth-home-detached"
    auth_path = auth_home / "auth.json"
    old_store = OpenAISubscriptionAuthStore(auth_path)
    if operation == "load":
        old_store.save(
            OpenAISubscriptionCredentials(
                access_token="old-access",
                refresh_token="old-refresh",
                expires_at=2_000_000_000,
            )
        )
    else:
        auth_home.mkdir(mode=0o700)
        auth_path.write_text(
            '{"providers":{"independent":{"opaque":"preserved"}},"version":1}\n',
            encoding="utf-8",
        )
        auth_path.chmod(0o600)
        with old_store._exclusive_lock():
            pass

    lock_path = auth_home / ".auth.json.lock"
    lock_descriptor = os.open(lock_path, os.O_RDWR)
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
    context = multiprocessing.get_context("spawn")
    lock_attempted = context.Event()
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_auth_store_operation_after_lock_attempt,
        args=(str(auth_path), operation, lock_attempted, sender),
    )
    process.start()
    sender.close()
    try:
        assert lock_attempted.wait(timeout=5)
        auth_home.rename(moved_home)
        auth_home.mkdir(mode=0o700)
        OpenAISubscriptionAuthStore(auth_path).save(
            OpenAISubscriptionCredentials(
                access_token="current-access",
                refresh_token="current-refresh",
                expires_at=2_100_000_000,
            )
        )
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)

    _join_auth_store_process(process)

    assert process.exitcode == 0
    assert receiver.poll(timeout=1)
    assert receiver.recv() == (
        "ValueError",
        "Cayu auth-store directory changed while in use.",
    )
    receiver.close()
    assert OpenAISubscriptionAuthStore(auth_path).load() == OpenAISubscriptionCredentials(
        access_token="current-access",
        refresh_token="current-refresh",
        expires_at=2_100_000_000,
    )


@pytest.mark.skipif(
    not subscription_auth._SUPPORTS_DURABLE_AUTH_STORE,
    reason="requires POSIX durable-store primitives",
)
@pytest.mark.parametrize("operation", ["load", "delete"])
def test_auth_store_rejects_directory_replacement_after_successful_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    auth_home = tmp_path / "auth-home"
    moved_home = tmp_path / "auth-home-detached"
    auth_path = auth_home / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    old_credentials = OpenAISubscriptionCredentials(
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at=2_000_000_000,
    )
    current_credentials = OpenAISubscriptionCredentials(
        access_token="current-access",
        refresh_token="current-refresh",
        expires_at=2_100_000_000,
    )
    store.save(old_credentials)

    def replace_directory() -> None:
        auth_home.rename(moved_home)
        auth_home.mkdir(mode=0o700)
        OpenAISubscriptionAuthStore(auth_path).save(current_credentials)

    if operation == "load":
        original_load = store._load_credentials_unlocked

        def load_then_replace(directory_fd: int | None):
            result = original_load(directory_fd)
            replace_directory()
            return result

        monkeypatch.setattr(store, "_load_credentials_unlocked", load_then_replace)
        operation_call = store.load
    else:

        def no_op_delete_then_replace(_directory_fd: int | None) -> bool:
            replace_directory()
            return False

        monkeypatch.setattr(store, "_delete_unlocked", no_op_delete_then_replace)
        operation_call = store.delete

    with pytest.raises(
        ValueError,
        match="Cayu auth-store directory changed while in use",
    ):
        operation_call()

    assert OpenAISubscriptionAuthStore(auth_path).load() == current_credentials
    assert OpenAISubscriptionAuthStore(moved_home / "auth.json").load() == old_credentials


def test_auth_store_missing_delete_waits_for_concurrent_save(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    context = multiprocessing.get_context("spawn")
    save_started = context.Event()
    release_save = context.Event()
    delete_lock_attempted = context.Event()
    read_result, write_result = context.Pipe(duplex=False)
    save = context.Process(
        target=_serialized_auth_store_save,
        args=(
            str(auth_path),
            "access",
            "refresh",
            2_000_000_000,
            save_started,
            release_save,
        ),
    )
    delete = context.Process(
        target=_delete_auth_store_and_report,
        args=(str(auth_path), delete_lock_attempted, write_result),
    )

    save.start()
    assert save_started.wait(timeout=5)
    assert not auth_path.exists()
    delete.start()
    attempted = delete_lock_attempted.wait(timeout=5)
    blocked = not read_result.poll(0.2)
    release_save.set()
    _join_auth_store_process(save)
    _join_auth_store_process(delete)

    assert attempted
    assert blocked
    assert save.exitcode == 0
    assert delete.exitcode == 0
    assert read_result.recv() == ("success", True)
    assert OpenAISubscriptionAuthStore(auth_path).load() is None


def test_auth_store_process_crash_before_replace_preserves_old_complete_record(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    original = OpenAISubscriptionCredentials(
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at=2_000_000_000,
    )
    replacement = OpenAISubscriptionCredentials(
        access_token="new-access",
        refresh_token="new-refresh",
        expires_at=2_100_000_000,
    )
    OpenAISubscriptionAuthStore(auth_path).save(original)
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_crash_auth_store_before_replace,
        args=(str(auth_path), replacement),
    )

    process.start()
    _join_auth_store_process(process)

    assert process.exitcode == 73
    assert OpenAISubscriptionAuthStore(auth_path).load() == original
    assert len(list(tmp_path.glob(".auth.json.tmp-*"))) == 1


def test_auth_store_restart_resyncs_parent_chain_after_creation_crash(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "nested" / "cayu-home" / "auth.json"
    context = multiprocessing.get_context("spawn")
    crashed = context.Process(
        target=_crash_auth_store_after_parent_creation,
        args=(str(auth_path),),
    )

    crashed.start()
    _join_auth_store_process(crashed)

    assert crashed.exitcode == 74
    assert auth_path.parent.is_dir()
    assert not auth_path.exists()

    receiver, sender = context.Pipe(duplex=False)
    recovered = context.Process(
        target=_save_auth_store_and_report_directory_syncs,
        args=(str(auth_path), sender),
    )
    recovered.start()
    sender.close()
    _join_auth_store_process(recovered)

    assert recovered.exitcode == 0
    assert receiver.poll(timeout=1)
    status, synchronized = receiver.recv()
    receiver.close()
    assert status == "success"
    assert str(tmp_path.resolve()) in synchronized
    assert str((tmp_path / "nested").resolve()) in synchronized
    assert str(auth_path.parent.resolve()) in synchronized
    assert OpenAISubscriptionAuthStore(auth_path).load() == OpenAISubscriptionCredentials(
        access_token="recovered-access",
        refresh_token="recovered-refresh",
        expires_at=2_100_000_000,
    )


@pytest.mark.parametrize("mask", [0o000, 0o002])
def test_auth_store_nested_creation_is_private_under_permissive_umask(
    tmp_path: Path,
    mask: int,
) -> None:
    auth_path = tmp_path / f"nested-{mask:o}" / "cayu-home" / "auth.json"
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_save_auth_store_with_umask,
        args=(str(auth_path), mask, sender),
    )

    process.start()
    sender.close()
    _join_auth_store_process(process)

    assert process.exitcode == 0
    assert receiver.poll(timeout=1)
    status, modes_and_owners = receiver.recv()
    receiver.close()
    assert status == "success"
    assert modes_and_owners == [
        (0o700, os.geteuid()),
        (0o700, os.geteuid()),
        (0o600, os.geteuid()),
    ]


def test_auth_store_process_exit_after_acknowledgement_keeps_new_record(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    replacement = OpenAISubscriptionCredentials(
        access_token="new-access",
        refresh_token="new-refresh",
        expires_at=2_100_000_000,
    )
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_save_auth_store_then_exit,
        args=(str(auth_path), replacement),
    )

    process.start()
    _join_auth_store_process(process)

    assert process.exitcode == 0
    assert OpenAISubscriptionAuthStore(auth_path).load() == replacement


def test_subscription_refresh_fails_before_provider_call_when_durability_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    store.save(
        OpenAISubscriptionCredentials(
            access_token="expired-access",
            refresh_token="old-refresh",
            expires_at=1,
        )
    )

    class RecordingTransport:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def refresh(self, refresh_token: str) -> dict[str, Any]:
            self.calls.append(refresh_token)
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

    transport = RecordingTransport()
    auth = OpenAISubscriptionAuth(
        store=store,
        oauth_transport=transport,
        now=lambda: 1_000,
    )
    monkeypatch.setattr(subscription_auth, "_supports_durable_auth_store", lambda: False)

    with pytest.raises(
        OpenAISubscriptionAuthError,
        match="OpenAI subscription login could not refresh",
    ):
        asyncio.run(auth.credentials())

    assert transport.calls == []
    assert store.load() == OpenAISubscriptionCredentials(
        access_token="expired-access",
        refresh_token="old-refresh",
        expires_at=1,
    )


@pytest.mark.skipif(
    not subscription_auth._SUPPORTS_DURABLE_AUTH_STORE or os.geteuid() == 0,
    reason="requires non-root POSIX permission enforcement",
)
def test_subscription_refresh_reserves_writable_staging_before_provider_call(
    tmp_path: Path,
) -> None:
    auth_home = tmp_path / "auth-home"
    auth_path = auth_home / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    original = OpenAISubscriptionCredentials(
        access_token="expired-access",
        refresh_token="old-refresh",
        expires_at=1,
    )
    store.save(original)

    class RecordingTransport:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def refresh(self, refresh_token: str) -> dict[str, Any]:
            self.calls.append(refresh_token)
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

    transport = RecordingTransport()
    auth = OpenAISubscriptionAuth(
        store=store,
        oauth_transport=transport,
        now=lambda: 1_000,
    )
    auth_home.chmod(0o500)
    try:
        with pytest.raises(
            OpenAISubscriptionAuthError,
            match="OpenAI subscription login could not refresh",
        ):
            asyncio.run(auth.credentials())
    finally:
        auth_home.chmod(0o700)

    assert transport.calls == []
    assert list(auth_home.glob(".auth.json.tmp-*")) == []
    assert store.load() == original


def test_subscription_refresh_syncs_reserved_staging_before_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    original = OpenAISubscriptionCredentials(
        access_token="expired-access",
        refresh_token="old-refresh",
        expires_at=1,
    )
    store.save(original)

    class RecordingTransport:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def refresh(self, refresh_token: str) -> dict[str, Any]:
            self.calls.append(refresh_token)
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

    transport = RecordingTransport()
    auth = OpenAISubscriptionAuth(
        store=store,
        oauth_transport=transport,
        now=lambda: 1_000,
    )
    real_sync = subscription_auth._sync_auth_store_descriptor

    def fail_regular_file_sync(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("injected staging sync failure")
        real_sync(descriptor)

    monkeypatch.setattr(
        subscription_auth,
        "_sync_auth_store_descriptor",
        fail_regular_file_sync,
    )

    with pytest.raises(
        OpenAISubscriptionAuthError,
        match="OpenAI subscription login could not refresh",
    ):
        asyncio.run(auth.credentials())

    assert transport.calls == []
    assert list(tmp_path.glob(".auth.json.tmp-*")) == []
    monkeypatch.undo()
    assert store.load() == original


def test_subscription_refresh_reserves_data_capacity_before_provider_call(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    store.save(
        OpenAISubscriptionCredentials(
            access_token="expired-access",
            refresh_token="old-refresh",
            expires_at=1,
        )
    )
    staged_sizes: list[int] = []

    class InspectingTransport:
        def refresh(self, _refresh_token: str) -> dict[str, Any]:
            staging = list(tmp_path.glob(".auth.json.tmp-*"))
            assert len(staging) == 1
            staged_sizes.append(staging[0].stat().st_size)
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

    auth = OpenAISubscriptionAuth(
        store=store,
        oauth_transport=InspectingTransport(),
        now=lambda: 1_000,
    )

    refreshed = asyncio.run(auth.credentials())

    assert staged_sizes
    assert staged_sizes[0] >= (
        subscription_auth._AUTH_ACCESS_TOKEN_MAX_BYTES
        + subscription_auth._AUTH_REFRESH_TOKEN_MAX_BYTES
        + subscription_auth._AUTH_ACCOUNT_ID_MAX_BYTES
    )
    assert refreshed.refresh_token == "new-refresh"
    assert auth_path.stat().st_size < staged_sizes[0]
    assert list(tmp_path.glob(".auth.json.tmp-*")) == []


def test_subscription_refresh_keeps_reservation_through_final_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    store.save(
        OpenAISubscriptionCredentials(
            access_token="expired-access",
            refresh_token="old-refresh",
            expires_at=1,
        )
    )
    write_events: list[tuple[str, int]] = []
    real_ftruncate = os.ftruncate
    real_write = os.write

    def recording_ftruncate(descriptor: int, length: int) -> None:
        write_events.append(("truncate", length))
        real_ftruncate(descriptor, length)

    def recording_write(descriptor: int, payload) -> int:
        write_events.append(("write", len(payload)))
        return real_write(descriptor, payload)

    class InspectingTransport:
        def refresh(self, _refresh_token: str) -> dict[str, Any]:
            assert any(event == "truncate" and length > 0 for event, length in write_events)
            assert all(event != "write" for event, _ in write_events)
            write_events.clear()
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

    monkeypatch.setattr(subscription_auth.os, "ftruncate", recording_ftruncate)
    monkeypatch.setattr(subscription_auth.os, "write", recording_write)
    auth = OpenAISubscriptionAuth(
        store=store,
        oauth_transport=InspectingTransport(),
        now=lambda: 1_000,
    )

    refreshed = asyncio.run(auth.credentials())

    assert write_events[0][0] == "write"
    assert ("truncate", 0) not in write_events
    assert write_events[-1] == ("truncate", auth_path.stat().st_size)
    assert OpenAISubscriptionAuthStore(auth_path).load() == refreshed
    assert list(tmp_path.glob(".auth.json.tmp-*")) == []


def test_subscription_refresh_disk_pressure_after_provider_uses_live_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    store.save(
        OpenAISubscriptionCredentials(
            access_token="expired-access",
            refresh_token="old-refresh",
            expires_at=1,
        )
    )
    state = {"provider_returned": False, "reservation_live": False}
    calls: list[str] = []
    real_reserve = subscription_auth._reserve_auth_store_capacity
    real_ftruncate = os.ftruncate
    real_write = os.write

    def recording_reserve(descriptor: int, reserved_bytes: int) -> None:
        real_reserve(descriptor, reserved_bytes)
        state["reservation_live"] = True

    def disk_pressure_ftruncate(descriptor: int, length: int) -> None:
        if state["provider_returned"] and length == 0:
            state["reservation_live"] = False
        real_ftruncate(descriptor, length)

    def disk_pressure_write(descriptor: int, payload) -> int:
        if state["provider_returned"] and not state["reservation_live"]:
            raise OSError(errno.ENOSPC, "injected post-refresh capacity failure")
        return real_write(descriptor, payload)

    class RecordingTransport:
        def refresh(self, refresh_token: str) -> dict[str, Any]:
            calls.append(refresh_token)
            state["provider_returned"] = True
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

    monkeypatch.setattr(
        subscription_auth,
        "_reserve_auth_store_capacity",
        recording_reserve,
    )
    monkeypatch.setattr(subscription_auth.os, "ftruncate", disk_pressure_ftruncate)
    monkeypatch.setattr(subscription_auth.os, "write", disk_pressure_write)
    auth = OpenAISubscriptionAuth(
        store=store,
        oauth_transport=RecordingTransport(),
        now=lambda: 1_000,
    )

    refreshed = asyncio.run(auth.credentials())
    reloaded = OpenAISubscriptionAuthStore(auth_path).load()
    assert reloaded == refreshed
    assert reloaded is not None
    assert reloaded.refresh_token == "new-refresh"
    assert (
        asyncio.run(
            OpenAISubscriptionAuth(
                store=OpenAISubscriptionAuthStore(auth_path),
                oauth_transport=RecordingTransport(),
                now=lambda: 1_000,
            ).credentials()
        )
        == refreshed
    )
    assert calls == ["old-refresh"]
    assert list(tmp_path.glob(".auth.json.tmp-*")) == []


def test_subscription_refresh_shrink_failure_publishes_valid_padded_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    store.save(
        OpenAISubscriptionCredentials(
            access_token="expired-access",
            refresh_token="old-refresh",
            expires_at=1,
        )
    )
    state = {"provider_returned": False}
    staged_sizes: list[int] = []
    real_ftruncate = os.ftruncate

    def fail_final_shrink(descriptor: int, length: int) -> None:
        if state["provider_returned"] and length < os.fstat(descriptor).st_size:
            raise OSError("injected non-critical shrink failure")
        real_ftruncate(descriptor, length)

    class RecordingTransport:
        def refresh(self, _refresh_token: str) -> dict[str, Any]:
            staging = list(tmp_path.glob(".auth.json.tmp-*"))
            assert len(staging) == 1
            staged_sizes.append(staging[0].stat().st_size)
            state["provider_returned"] = True
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

    monkeypatch.setattr(subscription_auth.os, "ftruncate", fail_final_shrink)
    refreshed = asyncio.run(
        OpenAISubscriptionAuth(
            store=store,
            oauth_transport=RecordingTransport(),
            now=lambda: 1_000,
        ).credentials()
    )

    assert staged_sizes
    assert auth_path.stat().st_size == staged_sizes[0]
    assert OpenAISubscriptionAuthStore(auth_path).load() == refreshed
    assert list(tmp_path.glob(".auth.json.tmp-*")) == []


def test_subscription_refresh_capacity_failure_precedes_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    original = OpenAISubscriptionCredentials(
        access_token="expired-access",
        refresh_token="old-refresh",
        expires_at=1,
    )
    store.save(original)

    class RecordingTransport:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def refresh(self, refresh_token: str) -> dict[str, Any]:
            self.calls.append(refresh_token)
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

    transport = RecordingTransport()
    auth = OpenAISubscriptionAuth(
        store=store,
        oauth_transport=transport,
        now=lambda: 1_000,
    )
    monkeypatch.setattr(
        subscription_auth,
        "_reserve_auth_store_capacity",
        lambda _descriptor, _payload: (_ for _ in ()).throw(
            OSError(errno.ENOSPC, "injected capacity failure")
        ),
    )

    with pytest.raises(
        OpenAISubscriptionAuthError,
        match="OpenAI subscription login could not refresh",
    ):
        asyncio.run(auth.credentials())

    assert transport.calls == []
    assert list(tmp_path.glob(".auth.json.tmp-*")) == []
    monkeypatch.undo()
    assert store.load() == original


def test_auth_store_capacity_reservation_uses_posix_fallocate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "reserved"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    calls: list[tuple[int, int, int]] = []

    def recording_fallocate(current: int, offset: int, length: int) -> None:
        calls.append((current, offset, length))

    monkeypatch.setattr(subscription_auth.sys, "platform", "linux")
    monkeypatch.setattr(
        subscription_auth.os,
        "posix_fallocate",
        recording_fallocate,
        raising=False,
    )
    try:
        subscription_auth._reserve_auth_store_capacity(descriptor, 4096)
        assert calls == [(descriptor, 0, 4096)]
        assert os.fstat(descriptor).st_size == 4096
    finally:
        os.close(descriptor)


def test_auth_store_capacity_reservation_uses_full_darwin_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fcntl = pytest.importorskip("fcntl")
    path = tmp_path / "reserved"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    calls: list[tuple[int, int, bytes]] = []

    def recording_fcntl(current: int, command: int, request: bytes) -> bytes:
        calls.append((current, command, request))
        flags, position_mode, offset, length, _ = subscription_auth._DARWIN_FSTORE.unpack(request)
        return subscription_auth._DARWIN_FSTORE.pack(
            flags,
            position_mode,
            offset,
            length,
            length,
        )

    monkeypatch.setattr(subscription_auth.sys, "platform", "darwin")
    monkeypatch.setattr(fcntl, "fcntl", recording_fcntl)
    try:
        subscription_auth._reserve_auth_store_capacity(descriptor, 4096)
        assert len(calls) == 1
        current, command, request = calls[0]
        assert current == descriptor
        assert command == subscription_auth._DARWIN_F_PREALLOCATE
        assert subscription_auth._DARWIN_FSTORE.unpack(request) == (
            subscription_auth._DARWIN_F_ALLOCATEALL,
            subscription_auth._DARWIN_F_PEOFPOSMODE,
            0,
            4096,
            0,
        )
        assert os.fstat(descriptor).st_size == 4096
    finally:
        os.close(descriptor)


def test_auth_store_capacity_reservation_rejects_partial_darwin_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fcntl = pytest.importorskip("fcntl")
    path = tmp_path / "reserved"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

    def partial_fcntl(_current: int, _command: int, request: bytes) -> bytes:
        flags, position_mode, offset, length, _ = subscription_auth._DARWIN_FSTORE.unpack(request)
        return subscription_auth._DARWIN_FSTORE.pack(
            flags,
            position_mode,
            offset,
            length,
            length - 1,
        )

    monkeypatch.setattr(subscription_auth.sys, "platform", "darwin")
    monkeypatch.setattr(fcntl, "fcntl", partial_fcntl)
    try:
        with pytest.raises(
            ValueError,
            match="Cayu auth store could not be made durable",
        ):
            subscription_auth._reserve_auth_store_capacity(descriptor, 4096)
        assert os.fstat(descriptor).st_size == 0
    finally:
        os.close(descriptor)


def test_auth_store_capacity_reservation_fails_closed_when_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "reserved"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    monkeypatch.setattr(subscription_auth.sys, "platform", "freebsd")
    monkeypatch.delattr(subscription_auth.os, "posix_fallocate", raising=False)
    try:
        with pytest.raises(
            ValueError,
            match="Cayu auth store durability is unsupported on this platform",
        ):
            subscription_auth._reserve_auth_store_capacity(descriptor, 4096)
        assert os.fstat(descriptor).st_size == 0
    finally:
        os.close(descriptor)


@pytest.mark.skipif(
    not subscription_auth._SUPPORTS_DURABLE_AUTH_STORE,
    reason="requires POSIX durable-store primitives",
)
@pytest.mark.parametrize("mutation", ["replace", "unsafe_mode"])
def test_subscription_refresh_revalidates_directory_before_publishing(
    tmp_path: Path,
    mutation: str,
) -> None:
    auth_home = tmp_path / "auth-home"
    moved_home = tmp_path / "auth-home-detached"
    auth_path = auth_home / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    original = OpenAISubscriptionCredentials(
        access_token="expired-access",
        refresh_token="old-refresh",
        expires_at=1,
    )
    store.save(original)

    class MutatingTransport:
        def refresh(self, _refresh_token: str) -> dict[str, Any]:
            if mutation == "replace":
                auth_home.rename(moved_home)
                auth_home.mkdir(mode=0o700)
                OpenAISubscriptionAuthStore(auth_path).save(original)
            else:
                auth_home.chmod(0o777)
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

    auth = OpenAISubscriptionAuth(
        store=store,
        oauth_transport=MutatingTransport(),
        now=lambda: 1_000,
    )
    try:
        with pytest.raises(
            OpenAISubscriptionAuthError,
            match="OpenAI subscription login could not refresh",
        ):
            asyncio.run(auth.credentials())
    finally:
        if mutation == "unsafe_mode":
            auth_home.chmod(0o700)

    assert OpenAISubscriptionAuthStore(auth_path).load() == original
    if mutation == "replace":
        assert OpenAISubscriptionAuthStore(moved_home / "auth.json").load() == original
        assert list(moved_home.glob(".auth.json.tmp-*")) == []
    else:
        assert list(auth_home.glob(".auth.json.tmp-*")) == []


def test_subscription_refresh_fails_before_provider_call_when_parent_chain_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    original = OpenAISubscriptionCredentials(
        access_token="expired-access",
        refresh_token="old-refresh",
        expires_at=1,
    )
    store.save(original)

    class RecordingTransport:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def refresh(self, refresh_token: str) -> dict[str, Any]:
            self.calls.append(refresh_token)
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

    transport = RecordingTransport()
    auth = OpenAISubscriptionAuth(
        store=store,
        oauth_transport=transport,
        now=lambda: 1_000,
    )
    monkeypatch.setattr(
        subscription_auth,
        "_sync_auth_store_directory_path",
        lambda _path: (_ for _ in ()).throw(
            ValueError("Cayu auth store could not be made durable.")
        ),
    )

    with pytest.raises(
        OpenAISubscriptionAuthError,
        match="OpenAI subscription login could not refresh",
    ):
        asyncio.run(auth.credentials())

    assert transport.calls == []
    monkeypatch.undo()
    assert store.load() == original


def test_auth_store_write_fails_explicitly_when_durability_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    credentials = OpenAISubscriptionCredentials(
        access_token="access-canary",
        refresh_token="refresh-canary",
        expires_at=2_000_000_000,
    )
    monkeypatch.setattr(subscription_auth, "_supports_durable_auth_store", lambda: False)

    with pytest.raises(
        ValueError,
        match="Cayu auth store durability is unsupported on this platform",
    ) as exc_info:
        OpenAISubscriptionAuthStore(auth_path).save(credentials)

    assert not auth_path.exists()
    _assert_auth_store_traceback_does_not_retain_credentials(
        exc_info.value,
        credentials,
    )
    retained = repr(exc_info.value) + repr(vars(exc_info.value))
    assert "access-canary" not in retained
    assert "refresh-canary" not in retained


def test_auth_store_load_exit_failure_drops_decoded_credentials_from_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    store.save(
        OpenAISubscriptionCredentials(
            access_token="loaded-access-canary",
            refresh_token="loaded-refresh-canary",
            expires_at=2_000_000_000,
            account_id="loaded-account-canary",
        )
    )
    loaded: list[OpenAISubscriptionCredentials] = []
    real_load = store._load_credentials_unlocked
    real_lock = store._exclusive_lock

    def recording_load(directory_fd: int | None) -> OpenAISubscriptionCredentials | None:
        credentials = real_load(directory_fd)
        assert credentials is not None
        loaded.append(credentials)
        return credentials

    @contextmanager
    def fail_after_read():
        with real_lock() as directory_fd:
            yield directory_fd
        raise ValueError("Cayu auth-store directory changed while in use.")

    monkeypatch.setattr(store, "_load_credentials_unlocked", recording_load)
    monkeypatch.setattr(store, "_exclusive_lock", fail_after_read)

    with pytest.raises(
        ValueError,
        match="Cayu auth-store directory changed while in use",
    ) as exc_info:
        store.load()

    assert len(loaded) == 1
    _assert_auth_store_traceback_does_not_retain_credentials(
        exc_info.value,
        loaded[0],
    )
    retained = repr(exc_info.value) + repr(vars(exc_info.value))
    assert "loaded-access-canary" not in retained
    assert "loaded-refresh-canary" not in retained
    assert "loaded-account-canary" not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_auth_store_missing_delete_is_noop_without_durability_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "missing-cayu-home" / "auth.json"
    monkeypatch.setattr(subscription_auth, "_supports_durable_auth_store", lambda: False)

    assert not OpenAISubscriptionAuthStore(auth_path).delete()
    assert not auth_path.parent.exists()


def test_auth_store_existing_delete_fails_closed_without_durability_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    credentials = OpenAISubscriptionCredentials(
        access_token="access",
        refresh_token="refresh",
        expires_at=2_000_000_000,
    )
    store = OpenAISubscriptionAuthStore(auth_path)
    store.save(credentials)
    monkeypatch.setattr(subscription_auth, "_supports_durable_auth_store", lambda: False)

    with pytest.raises(
        ValueError,
        match="Cayu auth store durability is unsupported on this platform",
    ):
        store.delete()

    assert OpenAISubscriptionAuthStore(auth_path).load() == credentials


def test_auth_store_missing_delete_does_not_bypass_dangling_symlink_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    target = tmp_path / "missing-target"
    auth_path.symlink_to(target)
    monkeypatch.setattr(subscription_auth, "_supports_durable_auth_store", lambda: False)

    with pytest.raises(
        ValueError,
        match="Refusing to read a symlinked Cayu auth store",
    ):
        OpenAISubscriptionAuthStore(auth_path).delete()

    assert auth_path.is_symlink()
    assert not target.exists()


def test_auth_store_filesystem_failure_omits_private_path_and_exception_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = "provider-auth-path-canary-0123456789"
    auth_path = tmp_path / canary / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)

    def fail_validation() -> None:
        raise OSError(f"permission denied for {auth_path}")

    monkeypatch.setattr(store, "_validate_existing_parent", fail_validation)

    with pytest.raises(ValueError) as exc_info:
        store.load()

    retained = repr(exc_info.value) + repr(vars(exc_info.value))
    assert str(exc_info.value) == "Could not access Cayu auth store."
    assert canary not in retained
    assert str(auth_path) not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_subscription_refresh_failure_has_no_credential_bearing_cause(tmp_path: Path) -> None:
    canary = "provider-refresh-canary-0123456789"
    store = OpenAISubscriptionAuthStore(tmp_path / "auth.json")
    store.save(
        OpenAISubscriptionCredentials(
            access_token="provider-access-canary-0123456789",
            refresh_token=canary,
            expires_at=1,
        )
    )

    class LeakingRefreshTransport:
        def refresh(self, refresh_token: str) -> dict[str, Any]:
            raise RuntimeError(f"refresh failed for {refresh_token}")

    auth = OpenAISubscriptionAuth(
        store=store,
        oauth_transport=LeakingRefreshTransport(),
        now=lambda: 1_000,
    )

    with pytest.raises(OpenAISubscriptionAuthError) as exc_info:
        asyncio.run(auth.credentials())

    assert_cayu_traceback_does_not_retain(exc_info.value, auth)
    retained = repr(exc_info.value) + repr(vars(exc_info.value))
    assert canary not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert list(tmp_path.glob(".auth.json.tmp-*")) == []


def test_subscription_refresh_lock_cancellation_drops_message_notes_and_metadata(
    tmp_path: Path,
) -> None:
    canary = "provider-refresh-cancel-canary-0123456789"
    store = OpenAISubscriptionAuthStore(tmp_path / "auth.json")
    store.save(
        OpenAISubscriptionCredentials(
            access_token="provider-access-canary-0123456789",
            refresh_token=canary,
            expires_at=1,
        )
    )
    auth = OpenAISubscriptionAuth(store=store, now=lambda: 1_000)

    async def cancel_while_waiting_for_refresh_lock() -> asyncio.CancelledError:
        await auth._refresh_lock.acquire()
        task = asyncio.create_task(auth.credentials())
        try:
            await asyncio.sleep(0.01)
            task.cancel(f"cancelled near {canary}")
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task
            return exc_info.value
        finally:
            auth._refresh_lock.release()

    error = asyncio.run(cancel_while_waiting_for_refresh_lock())

    assert_cayu_traceback_does_not_retain(error, auth)
    retained = repr(error) + repr(vars(error))
    assert canary not in retained
    assert error.args == ("OpenAI subscription request cancelled.",)
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("failure_kind", ["request", "invalid_json"])
def test_oauth_transport_failures_drop_raw_exception_graph(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    canary = "provider-oauth-transport-canary-0123456789"

    def fake_post(_url: str, **_kwargs: Any) -> httpx.Response:
        if failure_kind == "request":
            raise httpx.ConnectError(f"connect failed near {canary}")
        return httpx.Response(200, text=f'{{"credential":"{canary}"')

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(OpenAISubscriptionAuthError) as exc_info:
        HttpxOpenAISubscriptionOAuthTransport().refresh(canary)

    retained = repr(exc_info.value) + repr(vars(exc_info.value))
    assert canary not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_subscription_credentials_reject_non_finite_expiry() -> None:
    for expires_at in (float("inf"), float("-inf"), float("nan"), 10**400):
        try:
            OpenAISubscriptionCredentials(
                access_token="access-token",
                refresh_token="refresh-token",
                expires_at=expires_at,
            )
        except ValueError as exc:
            assert str(exc) == "expires_at must be finite and greater than zero."
        else:
            raise AssertionError(f"expires_at={expires_at!r} must fail")


@pytest.mark.parametrize(
    ("field_name", "limit"),
    [
        ("access_token", subscription_auth._AUTH_ACCESS_TOKEN_MAX_BYTES),
        ("refresh_token", subscription_auth._AUTH_REFRESH_TOKEN_MAX_BYTES),
        ("account_id", subscription_auth._AUTH_ACCOUNT_ID_MAX_BYTES),
    ],
)
def test_subscription_credentials_bound_storage_fields(
    field_name: str,
    limit: int,
) -> None:
    values: dict[str, Any] = {
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_at": 2_000_000_000,
        "account_id": "account",
    }
    values[field_name] = "x" * (limit + 1)

    with pytest.raises(ValueError, match=rf"`{field_name}` must not exceed {limit} bytes"):
        OpenAISubscriptionCredentials(**values)

    values[field_name] = "\ud800"
    with pytest.raises(ValueError, match=rf"`{field_name}` must contain valid Unicode"):
        OpenAISubscriptionCredentials(**values)


def test_oauth_token_response_rejects_non_finite_expiry_duration() -> None:
    for expires_in in (float("inf"), float("-inf"), float("nan"), 10**400):
        try:
            openai_subscription_credentials_from_token_response(
                {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_in": expires_in,
                },
                now=1_000.0,
            )
        except OpenAISubscriptionAuthError as exc:
            assert str(exc) == "OpenAI OAuth response contained invalid expires_in."
        else:
            raise AssertionError(f"expires_in={expires_in!r} must fail")


def test_subscription_authorize_url_uses_pkce_and_honest_cayu_originator() -> None:
    url = build_openai_subscription_authorize_url(
        redirect_uri="http://localhost:1455/auth/callback",
        code_challenge="challenge-value",
        state="csrf-state",
    )

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://auth.openai.com/oauth/authorize"
    )
    assert query["client_id"] == [OPENAI_SUBSCRIPTION_OAUTH_CLIENT_ID]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == ["challenge-value"]
    assert query["state"] == ["csrf-state"]
    assert query["originator"] == ["cayu"]


def test_http_oauth_refresh_uses_codex_public_client_without_printing_tokens(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        calls.append({"url": url, **kwargs})
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    response = HttpxOpenAISubscriptionOAuthTransport().refresh("old-refresh")

    assert response["access_token"] == "new-access"
    assert calls[0]["url"] == "https://auth.openai.com/oauth/token"
    assert calls[0]["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
        "client_id": OPENAI_SUBSCRIPTION_OAUTH_CLIENT_ID,
    }
    assert calls[0]["headers"]["user-agent"].startswith("cayu/")


class RecordingOAuthTransport:
    def __init__(self, response: dict[str, Any], *, delay_seconds: float = 0) -> None:
        self.response = response
        self.delay_seconds = delay_seconds
        self.refresh_tokens: list[str] = []

    def refresh(self, refresh_token: str) -> dict[str, Any]:
        self.refresh_tokens.append(refresh_token)
        time.sleep(self.delay_seconds)
        return self.response


def _jwt(claims: dict[str, Any]) -> str:
    def encode(value: dict[str, Any]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{encode({'alg': 'none'})}.{encode(claims)}.signature"


def test_subscription_auth_refreshes_expiring_token_and_persists_rotation(
    tmp_path: Path,
) -> None:
    store = OpenAISubscriptionAuthStore(tmp_path / "auth.json")
    store.save(
        OpenAISubscriptionCredentials(
            access_token="expired-access",
            refresh_token="old-refresh",
            expires_at=time.time() - 1,
            account_id="old-account",
        )
    )
    transport = RecordingOAuthTransport(
        {
            "access_token": _jwt(
                {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-refreshed"}}
            ),
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        }
    )
    auth = OpenAISubscriptionAuth(store=store, oauth_transport=transport)

    credentials = asyncio.run(auth.credentials())

    assert transport.refresh_tokens == ["old-refresh"]
    assert credentials.refresh_token == "new-refresh"
    assert credentials.account_id == "acct-refreshed"
    assert credentials.expires_at > time.time() + 3500
    assert store.load() == credentials
    assert list(tmp_path.glob(".auth.json.tmp-*")) == []


def test_subscription_auth_serializes_refresh_across_auth_instances(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    OpenAISubscriptionAuthStore(auth_path).save(
        OpenAISubscriptionCredentials(
            access_token="expired-access",
            refresh_token="old-refresh",
            expires_at=time.time() - 1,
        )
    )
    transport = RecordingOAuthTransport(
        {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        },
        delay_seconds=0.05,
    )
    first = OpenAISubscriptionAuth(
        store=OpenAISubscriptionAuthStore(auth_path),
        oauth_transport=transport,
    )
    second = OpenAISubscriptionAuth(
        store=OpenAISubscriptionAuthStore(auth_path),
        oauth_transport=transport,
    )

    async def load_both() -> tuple[OpenAISubscriptionCredentials, ...]:
        return tuple(await asyncio.gather(first.credentials(), second.credentials()))

    credentials = asyncio.run(load_both())

    assert transport.refresh_tokens == ["old-refresh"]
    assert credentials[0] == credentials[1]
    assert credentials[0].refresh_token == "new-refresh"


def test_subscription_auth_does_not_block_event_loop_behind_refresh_lock(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    OpenAISubscriptionAuthStore(auth_path).save(
        OpenAISubscriptionCredentials(
            access_token="expired-access",
            refresh_token="old-refresh",
            expires_at=time.time() - 1,
        )
    )

    class BlockingOAuthTransport(RecordingOAuthTransport):
        def __init__(self) -> None:
            super().__init__(
                {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "expires_in": 3600,
                }
            )
            self.started = threading.Event()
            self.release = threading.Event()

        def refresh(self, refresh_token: str) -> dict[str, Any]:
            self.started.set()
            if not self.release.wait(timeout=2):
                raise AssertionError("test did not release the OAuth refresh")
            return super().refresh(refresh_token)

    transport = BlockingOAuthTransport()
    first = OpenAISubscriptionAuth(
        store=OpenAISubscriptionAuthStore(auth_path),
        oauth_transport=transport,
    )
    second = OpenAISubscriptionAuth(
        store=OpenAISubscriptionAuthStore(auth_path),
        oauth_transport=transport,
    )

    async def exercise() -> None:
        first_task = asyncio.create_task(first.credentials())
        assert await asyncio.to_thread(transport.started.wait, 1)
        release_timer = threading.Timer(0.25, transport.release.set)
        release_timer.start()
        try:
            second_task = asyncio.create_task(second.credentials())
            started_at = asyncio.get_running_loop().time()
            await asyncio.sleep(0.03)
            assert asyncio.get_running_loop().time() - started_at < 0.15
            assert not transport.release.is_set()
            transport.release.set()
            await asyncio.gather(first_task, second_task)
        finally:
            transport.release.set()
            release_timer.cancel()
            release_timer.join()

    asyncio.run(exercise())
