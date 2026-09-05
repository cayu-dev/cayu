from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import pytest

from cayu.cli._generator_transaction import (
    GeneratorTransactionEdit,
    GeneratorTransactionError,
    GeneratorTransactionPrecondition,
    GeneratorTransactionRequest,
    apply_generator_transaction,
    generator_planning_guard,
    recover_generator_transaction,
)


def _digest(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


def _request(
    *,
    edits: tuple[
        tuple[str, Literal["create", "update_region"], bytes, bytes | None],
        ...,
    ],
    preconditions: tuple[tuple[str, bytes], ...] = (),
    slice_name: str = "transaction_test",
) -> GeneratorTransactionRequest:
    return GeneratorTransactionRequest(
        schema_version="17",
        slice_name=slice_name,
        tool_name="transaction_test",
        effect="none",
        authoring_state="unfinished_generated_tracer_bullet",
        edits=tuple(
            GeneratorTransactionEdit(
                path=path,
                operation=operation,
                content=content,
                content_sha256=_digest(content),
                preimage_sha256=None if before is None else _digest(before),
            )
            for path, operation, content, before in edits
        ),
        preconditions=tuple(
            GeneratorTransactionPrecondition(path=path, content_sha256=_digest(content))
            for path, content in preconditions
        ),
        verification_commands=("pytest -q",),
    )


def _run_crashing_apply(
    root: Path,
    request: GeneratorTransactionRequest,
    *,
    phase: str,
    signal_interrupt: bool = False,
    rollback_crash: bool = False,
    created_rollback_crash: bool = False,
) -> subprocess.CompletedProcess[str]:
    script = """
import json
import os
import sys
from pathlib import Path

import cayu.cli._generator_transaction as owner

root = Path(sys.argv[1])
phase = sys.argv[2]
payload = json.loads(sys.argv[3])
mode = sys.argv[4]
request = owner.GeneratorTransactionRequest(
    schema_version=payload["schema_version"],
    slice_name=payload["slice_name"],
    tool_name=payload["tool_name"],
    effect=payload["effect"],
    authoring_state=payload["authoring_state"],
    edits=tuple(
        owner.GeneratorTransactionEdit(
            path=item["path"],
            operation=item["operation"],
            content=bytes.fromhex(item["content"]),
            content_sha256=item["content_sha256"],
            preimage_sha256=item["preimage_sha256"],
        )
        for item in payload["edits"]
    ),
    preconditions=tuple(
        owner.GeneratorTransactionPrecondition(**item)
        for item in payload["preconditions"]
    ),
    verification_commands=tuple(payload["verification_commands"]),
)

def crash(observed):
    if mode == "rollback" and observed == "new_published:1":
        raise RuntimeError("begin rollback")
    if mode == "rollback-created" and observed == "created_root_published:0":
        raise RuntimeError("begin created-root rollback")
    if observed == phase:
        if mode == "signal":
            import signal
            os.kill(os.getpid(), signal.SIGINT)
        else:
            os._exit(73)

owner._fault = crash
owner.apply_generator_transaction(root, request)
"""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root),
            phase,
            _serialized_request(request),
            (
                "rollback-created"
                if created_rollback_crash
                else "rollback"
                if rollback_crash
                else "signal"
                if signal_interrupt
                else "exit"
            ),
        ],
        check=False,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")},
        text=True,
    )


def _serialized_request(request: GeneratorTransactionRequest) -> str:
    import json

    payload = {
        "schema_version": request.schema_version,
        "slice_name": request.slice_name,
        "tool_name": request.tool_name,
        "effect": request.effect,
        "authoring_state": request.authoring_state,
        "edits": [
            {
                "path": item.path,
                "operation": item.operation,
                "content": item.content.hex(),
                "content_sha256": item.content_sha256,
                "preimage_sha256": item.preimage_sha256,
            }
            for item in request.edits
        ],
        "preconditions": [item.__dict__ for item in request.preconditions],
        "verification_commands": list(request.verification_commands),
    }
    return json.dumps(payload)


def test_transaction_applies_mixed_edits_and_exact_retry(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    before = b"before\n"
    after = b"after\n"
    (root / "package" / "existing.py").write_bytes(before)
    request = _request(
        edits=(
            ("package/existing.py", "update_region", after, before),
            ("package/new.py", "create", b"new\n", None),
        ),
        preconditions=(("package/existing.py", before),),
    )

    apply_generator_transaction(root, request)
    apply_generator_transaction(root, request)

    assert (root / "package" / "existing.py").read_bytes() == after
    assert (root / "package" / "new.py").read_bytes() == b"new\n"
    assert not (root / ".cayu" / "generator-transactions" / "active").exists()


def test_exact_retry_rejects_a_self_consistent_receipt_for_other_edits(
    tmp_path: Path,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    package = root / "package"
    package.mkdir()
    before = b"before\n"
    after = b"after\n"
    target = package / "existing.py"
    target.write_bytes(before)
    unrelated = package / "unrelated.py"
    unrelated.write_bytes(after)
    request = _request(edits=(("package/existing.py", "update_region", after, before),))
    state = root / ".cayu" / "generator-transactions"
    state.mkdir(parents=True)
    receipt_payload: dict[str, object] = {
        "schema_version": owner._SCHEMA_VERSION,
        "request_digest": request.digest,
        "root_identity": owner._capture_directory_identity(
            root,
            label="project root",
        ).as_json(),
        "edits": [
            {
                "path": "package/unrelated.py",
                "snapshot": owner._snapshot_regular(
                    unrelated,
                    label="unrelated file",
                ).payload(),
                "parent_identity": owner._capture_directory_identity(
                    package,
                    label="package directory",
                ).as_json(),
            }
        ],
        "preconditions": [],
        "created_roots": [],
    }
    receipt_payload["receipt_sha256"] = owner._sha256(owner._canonical_json(receipt_payload))
    (state / "receipt.json").write_bytes(owner._canonical_json(receipt_payload) + b"\n")

    with pytest.raises(
        GeneratorTransactionError,
        match="receipt does not match the exact request",
    ):
        apply_generator_transaction(root, request)

    assert target.read_bytes() == before
    assert unrelated.read_bytes() == after
    assert not (state / "active").exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows cleanup contract")
def test_windows_transaction_cleanup_uses_dacl_authority_not_posix_modes(
    tmp_path: Path,
) -> None:
    import cayu.cli._guarded_tree_publication as publication

    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    request = _request(
        edits=(
            ("package/one.py", "create", b"one\n", None),
            ("feature/two.py", "create", b"two\n", None),
        )
    )

    apply_generator_transaction(root, request)

    assert (root / "package" / "one.py").read_bytes() == b"one\n"
    assert (root / "feature" / "two.py").read_bytes() == b"two\n"
    for published in (root / "package" / "one.py", root / "feature"):
        dacl_present, dacl_protected = publication._windows_directory_dacl_state(published)
        assert dacl_present
        assert not dacl_protected
    state = root / ".cayu" / "generator-transactions"
    assert not tuple(state.glob("prepare-*"))
    assert not tuple(state.glob("cleanup-*"))


def test_transaction_publishes_and_replays_a_created_subtree(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    request = _request(
        edits=(
            ("feature/agents/one.py", "create", b"one\n", None),
            ("feature/tools/two.py", "create", b"two\n", None),
        )
    )

    apply_generator_transaction(root, request)
    apply_generator_transaction(root, request)

    assert (root / "feature" / "agents" / "one.py").read_bytes() == b"one\n"
    assert (root / "feature" / "tools" / "two.py").read_bytes() == b"two\n"


def test_exact_retry_rejects_an_edited_created_subtree(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    request = _request(edits=(("feature/agents/one.py", "create", b"one\n", None),))
    apply_generator_transaction(root, request)
    (root / "feature" / "operator.txt").write_text("preserve me", encoding="utf-8")

    with pytest.raises(GeneratorTransactionError, match="changed"):
        apply_generator_transaction(root, request)

    assert (root / "feature" / "operator.txt").read_text(encoding="utf-8") == "preserve me"


def test_successor_request_replaces_the_exact_terminal_receipt(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    first = _request(edits=(("package/one.py", "create", b"one\n", None),))
    second = _request(
        edits=(("package/two.py", "create", b"two\n", None),),
        slice_name="second",
    )

    apply_generator_transaction(root, first)
    apply_generator_transaction(root, second)
    apply_generator_transaction(root, second)

    assert (root / "package" / "one.py").read_bytes() == b"one\n"
    assert (root / "package" / "two.py").read_bytes() == b"two\n"


@pytest.mark.parametrize(
    "phase",
    ["previous_receipt_moved_before_sync", "previous_receipt_moved"],
)
def test_successor_receipt_acknowledgement_loss_recovers_exactly(
    tmp_path: Path,
    phase: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    first = _request(edits=(("package/one.py", "create", b"one\n", None),))
    second = _request(
        edits=(("package/two.py", "create", b"two\n", None),),
        slice_name="second",
    )
    apply_generator_transaction(root, first)

    crashed = _run_crashing_apply(
        root,
        second,
        phase=phase,
    )
    assert crashed.returncode == 73, crashed.stderr

    assert recover_generator_transaction(root)
    apply_generator_transaction(root, second)
    assert (root / "package" / "one.py").read_bytes() == b"one\n"
    assert (root / "package" / "two.py").read_bytes() == b"two\n"


def test_exact_retry_rejects_same_bytes_with_an_unowned_identity(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    request = _request(edits=(("package/one.py", "create", b"one\n", None),))
    apply_generator_transaction(root, request)
    target = root / "package" / "one.py"
    target.unlink()
    target.write_bytes(b"one\n")

    with pytest.raises(GeneratorTransactionError, match="exact retry conflicts"):
        apply_generator_transaction(root, request)

    assert target.read_bytes() == b"one\n"


def test_parent_swap_at_the_namespace_boundary_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    package = root / "package"
    package.mkdir()
    before = b"before\n"
    (package / "existing.py").write_bytes(before)
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))
    real_rename = owner._rename_no_replace
    swapped = False

    def swap_parent_before_publish(
        source: Path,
        destination: Path,
        *,
        expected,
        expected_source_parent,
        expected_destination_parent,
        label: str,
        unexpected_policy=owner._UnexpectedRenamePolicy.RESTORE_SOURCE,
    ) -> None:
        nonlocal swapped
        if label == "generator stage" and not swapped:
            swapped = True
            package.rename(root / "displaced-package")
            package.mkdir()
            (package / "operator.txt").write_text("preserve", encoding="utf-8")
        real_rename(
            source,
            destination,
            expected=expected,
            expected_source_parent=expected_source_parent,
            expected_destination_parent=expected_destination_parent,
            label=label,
            unexpected_policy=unexpected_policy,
        )

    monkeypatch.setattr(owner, "_rename_no_replace", swap_parent_before_publish)
    with pytest.raises(GeneratorTransactionError, match="parent changed"):
        apply_generator_transaction(root, request)

    assert (package / "operator.txt").read_text(encoding="utf-8") == "preserve"
    assert not (package / "existing.py").exists()
    assert (root / ".cayu" / "generator-transactions" / "active").is_dir()


def test_source_swap_after_pinning_is_restored_before_conflict_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    package = root / "package"
    package.mkdir()
    before = b"before\n"
    target = package / "existing.py"
    target.write_bytes(before)
    replacement = package / ".editor-save"
    replacement.write_bytes(b"operator edit\n")
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))
    real_rename = owner._rename_names_no_replace
    swapped = False

    def swap_after_pin(**kwargs) -> None:
        nonlocal swapped
        if kwargs["source_name"] == target.name and not swapped:
            swapped = True
            os.replace(replacement, target)
        real_rename(**kwargs)

    monkeypatch.setattr(owner, "_rename_names_no_replace", swap_after_pin)
    with pytest.raises(GeneratorTransactionError, match="namespace transition"):
        apply_generator_transaction(root, request)

    assert swapped
    assert target.read_bytes() == b"operator edit\n"
    assert not replacement.exists()
    with pytest.raises(GeneratorTransactionError):
        recover_generator_transaction(root)
    assert target.read_bytes() == b"operator edit\n"


def test_created_subtree_replacement_after_native_rename_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    operator_tree = tmp_path / "operator-feature"
    operator_tree.mkdir()
    (operator_tree / "operator.txt").write_text("preserve me", encoding="utf-8")
    displaced_tree = tmp_path / "displaced-generated-feature"
    request = _request(
        edits=(
            ("feature/agents/one.py", "create", b"one\n", None),
            ("feature/tools/two.py", "create", b"two\n", None),
        )
    )
    real_rename = owner._rename_names_no_replace
    swapped = False

    def swap_public_destination_after_rename(**kwargs) -> None:
        nonlocal swapped
        real_rename(**kwargs)
        if (
            Path(kwargs["destination_parent_path"]) == root
            and kwargs["destination_name"] == "feature"
            and not swapped
        ):
            swapped = True
            (root / "feature").rename(displaced_tree)
            operator_tree.rename(root / "feature")

    monkeypatch.setattr(owner, "_rename_names_no_replace", swap_public_destination_after_rename)
    with pytest.raises(GeneratorTransactionError, match="namespace transition"):
        apply_generator_transaction(root, request)

    assert swapped
    assert (root / "feature" / "operator.txt").read_text(encoding="utf-8") == "preserve me"
    assert (displaced_tree / "agents" / "one.py").read_bytes() == b"one\n"
    assert (root / ".cayu" / "generator-transactions" / "active").is_dir()
    with pytest.raises(GeneratorTransactionError):
        recover_generator_transaction(root)
    assert (root / "feature" / "operator.txt").read_text(encoding="utf-8") == "preserve me"


def test_rollback_preserves_a_destination_replaced_after_native_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    package = root / "package"
    package.mkdir()
    before = b"before\n"
    target = package / "existing.py"
    target.write_bytes(before)
    replacement = package / ".editor-save"
    replacement.write_bytes(b"operator edit\n")
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))
    real_rename = owner._rename_names_no_replace
    swapped = False

    def swap_rollback_destination_after_rename(**kwargs) -> None:
        nonlocal swapped
        real_rename(**kwargs)
        if (
            Path(kwargs["source_parent_path"]).name == "backup"
            and Path(kwargs["destination_parent_path"]) == package
            and kwargs["destination_name"] == target.name
            and not swapped
        ):
            swapped = True
            os.replace(replacement, target)

    def begin_rollback(phase: str) -> None:
        if phase == "new_published:0":
            raise RuntimeError("begin rollback")

    monkeypatch.setattr(owner, "_rename_names_no_replace", swap_rollback_destination_after_rename)
    monkeypatch.setattr(owner, "_fault", begin_rollback)
    with pytest.raises(RuntimeError, match="begin rollback") as raised:
        apply_generator_transaction(root, request)

    assert swapped
    assert isinstance(raised.value.__cause__, GeneratorTransactionError)
    assert target.read_bytes() == b"operator edit\n"
    assert (root / ".cayu" / "generator-transactions" / "active").is_dir()
    with pytest.raises(GeneratorTransactionError):
        recover_generator_transaction(root)
    assert target.read_bytes() == b"operator edit\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX source-descriptor ownership")
def test_source_swap_during_pin_validation_closes_the_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    package = root / "package"
    package.mkdir()
    before = b"before\n"
    target = package / "existing.py"
    target.write_bytes(before)
    replacement = package / ".editor-save"
    replacement.write_bytes(b"operator edit\n")
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))
    real_open = owner.os.open
    pinned_descriptors: list[int] = []

    def swap_before_open(path, flags, *args, **kwargs):
        if (
            path == target.name
            and kwargs.get("dir_fd") is not None
            and not flags & getattr(os, "O_NONBLOCK", 0)
            and not pinned_descriptors
        ):
            os.replace(replacement, target)
            descriptor = real_open(path, flags, *args, **kwargs)
            pinned_descriptors.append(descriptor)
            return descriptor
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(owner.os, "open", swap_before_open)
    with pytest.raises(GeneratorTransactionError, match="while its source was pinned"):
        apply_generator_transaction(root, request)

    assert len(pinned_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(pinned_descriptors[0])
    assert target.read_bytes() == b"operator edit\n"


def test_state_directory_creation_does_not_follow_a_replaced_cayu_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    cayu = root / ".cayu"
    cayu.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    request = _request(edits=(("package/one.py", "create", b"one\n", None),))
    real_create = owner._create_owned_directory
    swapped = False

    def replace_state_parent(path: Path, **kwargs):
        nonlocal swapped
        if path.name == "generator-transactions" and not swapped:
            swapped = True
            cayu.rename(root / ".cayu-displaced")
            cayu.symlink_to(outside, target_is_directory=True)
        return real_create(path, **kwargs)

    monkeypatch.setattr(owner, "_create_owned_directory", replace_state_parent)

    with pytest.raises(
        GeneratorTransactionError,
        match="changed|ordinary directory|link or reparse point",
    ):
        apply_generator_transaction(root, request)

    assert not (outside / "generator-transactions").exists()
    assert not (root / "package" / "one.py").exists()


def test_terminal_cleanup_failure_retains_one_recoverable_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    request = _request(edits=(("package/one.py", "create", b"one\n", None),))
    real_remove = owner._remove_owned_directory
    failed = False

    def fail_once(path: Path, *, expected, expected_parent, authority) -> None:
        nonlocal failed
        if path.name.startswith("cleanup-") and not failed:
            failed = True
            raise OSError("simulated cleanup failure")
        real_remove(
            path,
            expected=expected,
            expected_parent=expected_parent,
            authority=authority,
        )

    monkeypatch.setattr(owner, "_remove_owned_directory", fail_once)
    with pytest.raises(OSError, match="simulated cleanup failure"):
        apply_generator_transaction(root, request)

    cleanup = tuple((root / ".cayu" / "generator-transactions").glob("cleanup-*"))
    cleanup_directories = tuple(path for path in cleanup if path.is_dir())
    cleanup_claims = tuple(path for path in cleanup if path.name.endswith(".claim.json"))
    cleanup_owners = tuple(path for path in cleanup if path.name.endswith(".owner.json"))
    assert len(cleanup_directories) == 1
    assert len(cleanup_claims) == 1
    assert len(cleanup_owners) == 1
    assert (cleanup_directories[0] / "owner.json").is_file()
    assert (root / "package" / "one.py").read_bytes() == b"one\n"

    assert recover_generator_transaction(root)
    assert not cleanup_directories[0].exists()
    assert not cleanup_claims[0].exists()
    assert not cleanup_owners[0].exists()
    apply_generator_transaction(root, request)


def test_cleanup_claim_rejects_unrecorded_private_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    request = _request(edits=(("package/one.py", "create", b"one\n", None),))
    real_remove = owner._remove_owned_directory
    failed = False

    def fail_once(path: Path, *, expected, expected_parent, authority) -> None:
        nonlocal failed
        if path.name.startswith("cleanup-") and not failed:
            failed = True
            raise OSError("retain cleanup authority")
        real_remove(
            path,
            expected=expected,
            expected_parent=expected_parent,
            authority=authority,
        )

    monkeypatch.setattr(owner, "_remove_owned_directory", fail_once)
    with pytest.raises(OSError, match="retain cleanup authority"):
        apply_generator_transaction(root, request)
    state = root / ".cayu" / "generator-transactions"
    cleanup = next(path for path in state.glob("cleanup-*") if path.is_dir())
    unexpected = cleanup / "operator-content.txt"
    unexpected.write_text("preserve", encoding="utf-8")

    with pytest.raises(GeneratorTransactionError, match="absent from its durable authority"):
        recover_generator_transaction(root)

    assert unexpected.read_text(encoding="utf-8") == "preserve"
    assert (root / "package" / "one.py").read_bytes() == b"one\n"


def test_private_stage_population_rejects_a_replaced_preparation_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    request = _request(edits=(("package/one.py", "create", b"one\n", None),))
    real_write = owner._write_stage_files
    displaced: Path | None = None

    def replace_before_population(path: Path, **kwargs) -> None:
        nonlocal displaced
        displaced = path.with_name(f"{path.name}-displaced")
        path.rename(displaced)
        path.mkdir(mode=0o700)
        real_write(path, **kwargs)

    monkeypatch.setattr(owner, "_write_stage_files", replace_before_population)
    with pytest.raises(GeneratorTransactionError, match="staging directory changed"):
        apply_generator_transaction(root, request)

    assert displaced is not None
    assert not tuple(displaced.iterdir())
    assert not tuple(displaced.with_name(displaced.name.removesuffix("-displaced")).iterdir())
    assert not (root / "package" / "one.py").exists()


def test_parent_sync_failure_after_original_move_rolls_back_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    package = root / "package"
    package.mkdir()
    before = b"before\n"
    target = package / "existing.py"
    target.write_bytes(before)
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))
    real_sync = owner._sync_directory
    failed = False

    def fail_once(path: Path) -> None:
        nonlocal failed
        if path == package and not target.exists() and not failed:
            failed = True
            raise OSError("simulated parent sync failure")
        real_sync(path)

    monkeypatch.setattr(owner, "_sync_directory", fail_once)
    with pytest.raises(OSError, match="simulated parent sync failure"):
        apply_generator_transaction(root, request)

    assert target.read_bytes() == before
    assert not (root / ".cayu" / "generator-transactions" / "active").exists()


def test_staged_regular_file_is_synced_before_commit_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    package = root / "package"
    package.mkdir()
    before = b"before\n"
    target = package / "existing.py"
    target.write_bytes(before)
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))
    real_sync = owner._sync_file

    def fail_stage_sync(path: Path, *, expected) -> None:
        if path.parent.name == "new" and path.parent.parent.name.startswith("prepare-"):
            raise OSError("simulated staged-file sync failure")
        real_sync(path, expected=expected)

    monkeypatch.setattr(owner, "_sync_file", fail_stage_sync)
    with pytest.raises(OSError, match="staged-file sync failure"):
        apply_generator_transaction(root, request)

    assert target.read_bytes() == before
    assert tuple((root / ".cayu" / "generator-transactions").glob("prepare-*"))
    monkeypatch.setattr(owner, "_sync_file", real_sync)
    assert recover_generator_transaction(root)
    assert target.read_bytes() == before


def test_request_bounds_reject_before_allocating_transaction_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    monkeypatch.setattr(owner, "_MAX_EDITS", 2)
    request = _request(
        edits=(
            ("package/one.py", "create", b"one\n", None),
            ("package/two.py", "create", b"two\n", None),
            ("package/three.py", "create", b"three\n", None),
        )
    )

    with pytest.raises(GeneratorTransactionError, match="between 1 and 2 edits"):
        apply_generator_transaction(root, request)

    assert not (root / ".cayu").exists()


@pytest.mark.parametrize(
    ("limit_name", "limit", "message"),
    [
        ("_MAX_RECEIPT_BYTES", 512, "receipt exceeds"),
        ("_MAX_CLEANUP_CLAIM_BYTES", 1_024, "cleanup authority exceeds"),
    ],
)
def test_complete_durable_records_are_bounded_before_state_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit: int,
    message: str,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    package = root / "package"
    package.mkdir()
    before = b"before\n"
    (package / "existing.py").write_bytes(before)
    request = _request(
        edits=(("package/existing.py", "update_region", b"after\n", before),),
        preconditions=(("package/existing.py", before),),
    )
    monkeypatch.setattr(owner, limit_name, limit)

    with pytest.raises(GeneratorTransactionError, match=message):
        apply_generator_transaction(root, request)

    assert not (root / ".cayu").exists()


@pytest.mark.parametrize(
    "platform_incarnation",
    [
        (0x1234_5678 << 128) | 1_700_000_000_000_000_000,
        (((1 << 64) - 1) << 128) | ((1 << 127) - 1),
    ],
    ids=("linux", "windows"),
)
def test_transaction_accepts_platform_width_identities_and_replays_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform_incarnation: int,
) -> None:
    import cayu.cli._generator_transaction as owner
    import cayu.cli._guarded_tree_publication as publication

    real_capture = publication._capture_stable_identity

    def capture_platform_width_identity(
        value: os.stat_result,
        *,
        path: Path | None = None,
        descriptor: int | None = None,
        dir_fd: int | None = None,
        name: str | None = None,
    ) -> publication._Identity:
        captured = real_capture(
            value,
            path=path,
            descriptor=descriptor,
            dir_fd=dir_fd,
            name=name,
        )
        return owner._Identity(
            device=(1 << 96) | captured.device,
            inode=(1 << 96) | captured.inode,
            kind=captured.kind,
            incarnation=platform_incarnation | captured.inode,
        )

    monkeypatch.setattr(owner, "_capture_stable_identity", capture_platform_width_identity)
    monkeypatch.setattr(
        publication,
        "_capture_stable_identity",
        capture_platform_width_identity,
    )
    root = tmp_path / "project"
    root.mkdir()
    package = root / "package"
    package.mkdir()
    request = _request(edits=(("package/value.py", "create", b"value\n", None),))

    apply_generator_transaction(root, request)
    apply_generator_transaction(root, request)

    assert (package / "value.py").read_bytes() == b"value\n"


def test_identity_record_bounds_match_supported_platform_encodings() -> None:
    import cayu.cli._generator_transaction as owner

    maximum = owner._Identity(
        device=owner._MAX_IDENTITY_DEVICE,
        inode=owner._MAX_IDENTITY_INODE,
        kind=owner._MAX_IDENTITY_KIND,
        incarnation=owner._MAX_IDENTITY_INCARNATION,
    )
    assert owner._identity_from_json(maximum.as_json(), field="identity") == maximum

    limits = (
        owner._MAX_IDENTITY_DEVICE,
        owner._MAX_IDENTITY_INODE,
        owner._MAX_IDENTITY_KIND,
        owner._MAX_IDENTITY_INCARNATION,
    )
    for index, limit in enumerate(limits):
        invalid = maximum.as_json()
        invalid[index] = limit + 1
        with pytest.raises(GeneratorTransactionError, match="invalid identity"):
            owner._identity_from_json(invalid, field="identity")


def test_identity_overflow_rejects_before_allocating_transaction_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    real_capture = owner._capture_stable_identity

    def capture_oversized_identity(value: os.stat_result, **kwargs: object):
        captured = real_capture(value, **kwargs)
        return owner._Identity(
            device=captured.device,
            inode=captured.inode,
            kind=captured.kind,
            incarnation=owner._MAX_IDENTITY_INCARNATION + 1,
        )

    monkeypatch.setattr(owner, "_capture_stable_identity", capture_oversized_identity)
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    request = _request(edits=(("package/value.py", "create", b"value\n", None),))

    with pytest.raises(GeneratorTransactionError, match="identity exceeds"):
        apply_generator_transaction(root, request)

    assert not (root / ".cayu").exists()


def test_cleanup_entry_aggregate_is_bounded_before_state_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    request = _request(
        edits=(
            (
                "/".join([f"level{index:03d}" for index in range(12)] + ["value.py"]),
                "create",
                b"value\n",
                None,
            ),
        )
    )
    monkeypatch.setattr(owner, "_MAX_PRIVATE_TREE_ENTRIES", 8)

    with pytest.raises(GeneratorTransactionError, match="entry-count limit"):
        apply_generator_transaction(root, request)

    assert not (root / ".cayu").exists()


def test_request_limits_accept_the_exact_supported_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    (root / "p").mkdir()
    before = b"old\n"
    (root / "p" / "a").write_bytes(before)
    monkeypatch.setattr(owner, "_MAX_EDITS", 2)
    monkeypatch.setattr(owner, "_MAX_PRECONDITIONS", 1)
    monkeypatch.setattr(owner, "_MAX_PATH_DEPTH", 2)
    monkeypatch.setattr(owner, "_MAX_PATH_BYTES", 3)
    monkeypatch.setattr(owner, "_MAX_STAGED_BYTES", 8 * 1024)
    first_content = b"a" * (4 * 1024)
    second_content = b"b" * (4 * 1024)
    request = _request(
        edits=(
            ("p/a", "update_region", first_content, before),
            ("p/b", "create", second_content, None),
        ),
        preconditions=(("p/a", before),),
    )

    apply_generator_transaction(root, request)

    assert (root / "p" / "a").read_bytes() == first_content
    assert (root / "p" / "b").read_bytes() == second_content


@pytest.mark.parametrize(
    ("limit_name", "limit", "transaction_request"),
    [
        (
            "_MAX_PRECONDITIONS",
            1,
            _request(
                edits=(("p/a", "create", b"a", None),),
                preconditions=(("p/b", b"b"), ("p/c", b"c")),
            ),
        ),
        (
            "_MAX_STAGED_BYTES",
            4,
            _request(edits=(("p/a", "create", b"abcde", None),)),
        ),
        (
            "_MAX_PATH_DEPTH",
            2,
            _request(edits=(("a/b/c", "create", b"a", None),)),
        ),
        (
            "_MAX_PATH_BYTES",
            3,
            _request(edits=(("a/bb", "create", b"a", None),)),
        ),
    ],
)
def test_request_limits_reject_over_boundary_before_state_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit: int,
    transaction_request: GeneratorTransactionRequest,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    (root / "p").mkdir()
    monkeypatch.setattr(owner, limit_name, limit)

    with pytest.raises(GeneratorTransactionError):
        apply_generator_transaction(root, transaction_request)

    assert not (root / ".cayu").exists()


def test_created_tree_census_stops_at_the_global_entry_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    tree = tmp_path / "tree"
    tree.mkdir()
    for name in ("one", "two", "three"):
        (tree / name).write_text(name, encoding="utf-8")
    real_scandir = owner.os.scandir
    retained = list(real_scandir(tree))

    class GuardedIterator:
        def __init__(self) -> None:
            self._index = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def __iter__(self):
            return self

        def __next__(self):
            if self._index >= 2:
                raise AssertionError("tree census consumed past its bound")
            value = retained[self._index]
            self._index += 1
            return value

    def guarded_scandir(path):
        return GuardedIterator() if Path(path) == tree else real_scandir(path)

    monkeypatch.setattr(owner, "_MAX_CREATED_TREE_ENTRIES", 2)
    monkeypatch.setattr(owner.os, "scandir", guarded_scandir)

    with pytest.raises(GeneratorTransactionError, match="entry limit"):
        owner._snapshot_tree(tree, label="test tree")


def test_created_tree_verification_enforces_one_aggregate_byte_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "one").write_bytes(b"one")
    (tree / "two").write_bytes(b"two")
    monkeypatch.setattr(owner, "_MAX_STAGED_BYTES", 4)

    with pytest.raises(GeneratorTransactionError, match="byte limit"):
        owner._snapshot_tree(tree, label="test tree")


def test_split_preimages_share_one_aggregate_inspection_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    package = root / "p"
    package.mkdir()
    first_before = b"one"
    second_before = b"two"
    (package / "a").write_bytes(first_before)
    (package / "b").write_bytes(second_before)
    monkeypatch.setattr(owner, "_MAX_STAGED_BYTES", 4)
    request = _request(
        edits=(
            ("p/a", "update_region", b"aa", first_before),
            ("p/b", "update_region", b"bb", second_before),
        )
    )

    with pytest.raises(GeneratorTransactionError, match="preimages exceed"):
        apply_generator_transaction(root, request)

    assert not (root / ".cayu").exists()


@pytest.mark.parametrize(
    "paths",
    [
        ("package/Result.py", "package/result.py"),
        ("package/one.py", "package/one.py/child.py"),
        ("package/one.py", "package//one.py"),
    ],
)
def test_request_rejects_alias_and_topology_conflicts_before_writes(
    tmp_path: Path,
    paths: tuple[str, str],
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    request = _request(edits=tuple((path, "create", b"content\n", None) for path in paths))

    with pytest.raises(GeneratorTransactionError):
        apply_generator_transaction(root, request)

    assert not (root / ".cayu").exists()


def test_request_cannot_target_the_transaction_owner_namespace(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    request = _request(
        edits=((".cayu/generator-transactions/receipt.json", "create", b"forged", None),)
    )

    with pytest.raises(GeneratorTransactionError, match="outside the supported"):
        apply_generator_transaction(root, request)

    assert not (root / ".cayu").exists()


@pytest.mark.parametrize(
    "phase",
    [
        "phase:committing",
        "original_moved_before_sync:0",
        "original_moved_before_record:0",
        "original_moved:0",
        "new_published_before_sync:0",
        "new_published_before_record:0",
        "new_published:0",
        "new_published_before_sync:1",
        "new_published_before_record:1",
        "new_published:1",
        "phase:committed",
        "receipt_published_before_sync",
        "receipt_published",
        "cleanup_claimed_before_sync",
        "cleanup_claimed",
        "cleanup_claim_published_before_sync",
        "cleanup_claim_published",
        "cleanup_owner_removed",
        "cleanup_directory_removed_before_sync",
        "cleanup_directory_removed",
    ],
)
def test_process_death_during_commit_rolls_forward_exactly(
    tmp_path: Path,
    phase: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    before = b"before\n"
    after = b"after\n"
    (root / "package" / "existing.py").write_bytes(before)
    request = _request(
        edits=(
            ("package/existing.py", "update_region", after, before),
            ("package/new.py", "create", b"new\n", None),
        ),
        preconditions=(("package/existing.py", before),),
    )

    crashed = _run_crashing_apply(root, request, phase=phase)
    assert crashed.returncode == 73, crashed.stderr

    assert recover_generator_transaction(root)
    assert (root / "package" / "existing.py").read_bytes() == after
    assert (root / "package" / "new.py").read_bytes() == b"new\n"
    apply_generator_transaction(root, request)


@pytest.mark.parametrize(
    "phase",
    [
        "preparation_synced",
        "preparation_promoted_before_sync",
        "preparation_promoted",
    ],
)
def test_process_death_before_commit_recovers_the_complete_old_state(
    tmp_path: Path,
    phase: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    before = b"before\n"
    (root / "package" / "existing.py").write_bytes(before)
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))

    crashed = _run_crashing_apply(root, request, phase=phase)
    assert crashed.returncode == 73, crashed.stderr

    assert recover_generator_transaction(root)
    assert (root / "package" / "existing.py").read_bytes() == before
    assert not (root / ".cayu" / "generator-transactions" / "active").exists()


@pytest.mark.parametrize(
    "phase",
    [
        "phase:rolling_back",
        "new_removed_before_sync:1",
        "new_removed_before_record:1",
        "new_removed:1",
        "new_removed_before_sync:0",
        "new_removed_before_record:0",
        "new_removed:0",
        "original_restored_before_sync:0",
        "original_restored_before_record:0",
        "original_restored:0",
        "rollback_stage_restored_before_sync:0",
        "phase:rolled_back",
    ],
)
def test_process_death_during_rollback_recovers_the_complete_old_state(
    tmp_path: Path,
    phase: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    before = b"before\n"
    (root / "package" / "existing.py").write_bytes(before)
    request = _request(
        edits=(
            ("package/existing.py", "update_region", b"after\n", before),
            ("package/new.py", "create", b"new\n", None),
        )
    )

    crashed = _run_crashing_apply(
        root,
        request,
        phase=phase,
        rollback_crash=True,
    )
    assert crashed.returncode == 73, crashed.stderr

    assert recover_generator_transaction(root)
    assert (root / "package" / "existing.py").read_bytes() == before
    assert not (root / "package" / "new.py").exists()
    assert not (root / ".cayu" / "generator-transactions" / "active").exists()


@pytest.mark.parametrize(
    "phase",
    [
        "created_root_published_before_sync:0",
        "created_root_published_before_record:0",
        "created_root_published:0",
    ],
)
def test_process_death_after_created_subtree_publication_recovers_new_state(
    tmp_path: Path,
    phase: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    request = _request(
        edits=(
            ("feature/agents/one.py", "create", b"one\n", None),
            ("feature/tools/two.py", "create", b"two\n", None),
        )
    )
    crashed = _run_crashing_apply(
        root,
        request,
        phase=phase,
    )
    assert crashed.returncode == 73, crashed.stderr

    assert recover_generator_transaction(root)
    assert (root / "feature" / "agents" / "one.py").read_bytes() == b"one\n"
    assert (root / "feature" / "tools" / "two.py").read_bytes() == b"two\n"


def test_process_death_after_terminal_cleanup_replays_the_exact_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    request = _request(edits=(("package/one.py", "create", b"one\n", None),))

    crashed = _run_crashing_apply(root, request, phase="transaction_cleaned")
    assert crashed.returncode == 73, crashed.stderr

    assert not recover_generator_transaction(root)
    apply_generator_transaction(root, request)
    assert (root / "package" / "one.py").read_bytes() == b"one\n"


def test_process_death_before_manifest_publication_recovers_owned_preparation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    before = b"before\n"
    target = root / "package" / "existing.py"
    target.write_bytes(before)
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))

    crashed = _run_crashing_apply(root, request, phase="preparation_directory_created")
    assert crashed.returncode == 73, crashed.stderr

    assert recover_generator_transaction(root)
    assert target.read_bytes() == before
    state = root / ".cayu" / "generator-transactions"
    assert not tuple(state.glob("prepare-*"))
    assert not tuple(state.glob("cleanup-*"))


def test_process_death_before_preparation_owner_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    before = b"before\n"
    target = root / "package" / "existing.py"
    target.write_bytes(before)
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))

    crashed = _run_crashing_apply(
        root,
        request,
        phase="preparation_directory_allocated",
    )
    assert crashed.returncode == 73, crashed.stderr

    state = root / ".cayu" / "generator-transactions"
    preparations = tuple(state.glob("prepare-" + "[0-9a-f]" * 64))
    assert len(preparations) == 1
    assert not tuple(state.glob("prepare-*.owner.json"))
    with pytest.raises(
        GeneratorTransactionError,
        match="unauthenticated generator preparation was preserved",
    ):
        recover_generator_transaction(root)

    assert target.read_bytes() == before
    assert preparations[0].is_dir()
    assert not (state / "active").exists()


def test_cleanup_claim_without_its_published_owner_cannot_delete_a_lookalike(
    tmp_path: Path,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    state = root / ".cayu" / "generator-transactions"
    state.mkdir(parents=True)
    token = "a" * 64
    cleanup = state / f"cleanup-{token}"
    cleanup.mkdir(mode=0o700)
    user_file = cleanup / "operator.txt"
    user_file.write_text("preserve", encoding="utf-8")
    cleanup_identity = owner._capture_directory_identity(cleanup, label="lookalike")
    _digest_value, entries = owner._capture_tree_authority(
        cleanup,
        expected=cleanup_identity,
        require_cleanup_access=True,
        entry_limit=owner._MAX_PRIVATE_TREE_ENTRIES,
    )
    false_owner = owner._snapshot_regular(user_file, label="lookalike owner")
    claim = state / f"cleanup-{token}.claim.json"
    claim.write_bytes(
        owner._cleanup_claim_content(
            owner_kind="preparation",
            token=token,
            request_digest="b" * 64,
            root_identity=owner._capture_directory_identity(root, label="project root"),
            transaction_identity=cleanup_identity,
            phase="preparing",
            preparation_owner=false_owner,
            journal=None,
            transaction_owner=None,
            entries=entries,
        )
    )

    with pytest.raises(GeneratorTransactionError, match="durable owner authority"):
        recover_generator_transaction(root)

    assert user_file.read_text(encoding="utf-8") == "preserve"
    assert cleanup.is_dir()
    assert claim.is_file()


@pytest.mark.parametrize(
    "phase",
    [
        "created_root_restored_before_sync:0",
        "created_root_restored_before_record:0",
        "created_root_restored:0",
    ],
)
def test_process_death_while_restoring_created_subtree_recovers_old_state(
    tmp_path: Path,
    phase: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    request = _request(
        edits=(
            ("feature/agents/one.py", "create", b"one\n", None),
            ("feature/tools/two.py", "create", b"two\n", None),
        )
    )
    crashed = _run_crashing_apply(
        root,
        request,
        phase=phase,
        created_rollback_crash=True,
    )
    assert crashed.returncode == 73, crashed.stderr

    assert recover_generator_transaction(root)
    assert not (root / "feature").exists()
    assert not (root / ".cayu" / "generator-transactions" / "active").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode mutation contract")
def test_recovery_rejects_an_edited_created_directory_mode(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    request = _request(
        edits=(
            ("feature/agents/one.py", "create", b"one\n", None),
            ("feature/tools/two.py", "create", b"two\n", None),
        )
    )
    crashed = _run_crashing_apply(
        root,
        request,
        phase="phase:rolling_back",
        created_rollback_crash=True,
    )
    assert crashed.returncode == 73, crashed.stderr
    feature = root / "feature"
    original_mode = stat.S_IMODE(feature.stat().st_mode)
    changed_mode = 0o700 if original_mode != 0o700 else 0o755
    feature.chmod(changed_mode)

    with pytest.raises(GeneratorTransactionError, match="changed before rollback"):
        recover_generator_transaction(root)

    assert stat.S_IMODE(feature.stat().st_mode) == changed_mode
    assert (feature / "agents" / "one.py").read_bytes() == b"one\n"


@pytest.mark.parametrize(
    ("signal_type", "message"),
    [
        (KeyboardInterrupt, "interrupt"),
        (SystemExit, "exit"),
        (GeneratorExit, "generator exit"),
    ],
)
def test_process_control_signals_remain_authoritative_after_exact_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_type: type[BaseException],
    message: str,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    before = b"before\n"
    (root / "package" / "existing.py").write_bytes(before)
    request = _request(
        edits=(
            ("package/existing.py", "update_region", b"after\n", before),
            ("package/new.py", "create", b"new\n", None),
        )
    )

    def interrupt(phase: str) -> None:
        if phase == "new_published:0":
            raise signal_type(message)

    monkeypatch.setattr(owner, "_fault", interrupt)
    with pytest.raises(signal_type, match=message):
        apply_generator_transaction(root, request)

    assert (root / "package" / "existing.py").read_bytes() == before
    assert not (root / "package" / "new.py").exists()
    assert not (root / ".cayu" / "generator-transactions" / "active").exists()


@pytest.mark.parametrize(
    ("durable_phase", "expected"),
    [
        ("committing", b"before\n"),
        ("committed", b"after\n"),
    ],
)
def test_real_sigint_after_phase_fsync_reconciles_the_durable_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    durable_phase: str,
    expected: bytes,
) -> None:
    import signal

    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    package = root / "package"
    package.mkdir()
    before = b"before\n"
    target = package / "existing.py"
    target.write_bytes(before)
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))
    delivered = False

    def interrupt_after_fsync(phase: str) -> None:
        nonlocal delivered
        if phase == f"journal_appended:phase:{durable_phase}" and not delivered:
            delivered = True
            os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(owner, "_fault", interrupt_after_fsync)
    with pytest.raises(KeyboardInterrupt):
        apply_generator_transaction(root, request)

    assert delivered
    assert target.read_bytes() == expected
    assert not (root / ".cayu" / "generator-transactions" / "active").exists()
    monkeypatch.setattr(owner, "_fault", lambda _phase: None)
    assert not recover_generator_transaction(root)


@pytest.mark.parametrize("failure_kind", ["ordinary", "sigint"])
def test_descriptor_cleanup_cannot_replace_an_authoritative_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    import signal

    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    package = root / "package"
    package.mkdir()
    before = b"before\n"
    target = package / "existing.py"
    target.write_bytes(before)
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))
    real_write_all = owner._write_all
    real_close_descriptor = owner._close_descriptor
    delivered = False

    def fail_journal_write(descriptor: int, content: bytes) -> None:
        nonlocal delivered
        if b'"event":"phase"' in content and b'"phase":"committing"' in content:
            delivered = True
            if failure_kind == "sigint":
                os.kill(os.getpid(), signal.SIGINT)
            raise RuntimeError("journal write failed")
        real_write_all(descriptor, content)

    def close_with_secondary_failure(descriptor: int) -> None:
        if sys.exception() is None:
            real_close_descriptor(descriptor)
            return
        os.close(descriptor)
        real_close_descriptor(descriptor)

    monkeypatch.setattr(owner, "_write_all", fail_journal_write)
    monkeypatch.setattr(owner, "_close_descriptor", close_with_secondary_failure)
    expected = KeyboardInterrupt if failure_kind == "sigint" else RuntimeError
    with pytest.raises(expected) as raised:
        apply_generator_transaction(root, request)

    assert delivered
    assert isinstance(raised.value.__cause__, OSError)
    assert target.read_bytes() == before
    assert not (root / ".cayu" / "generator-transactions" / "active").exists()


@pytest.mark.parametrize(
    "signal_type",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
)
def test_process_control_during_settlement_overrides_the_ordinary_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_type: type[BaseException],
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    package = root / "package"
    package.mkdir()
    before = b"before\n"
    (package / "existing.py").write_bytes(before)
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))

    def interrupt_settlement(phase: str) -> None:
        if phase == "new_published:0":
            raise RuntimeError("ordinary primary")
        if phase == "phase:rolling_back":
            raise signal_type("settlement signal")

    monkeypatch.setattr(owner, "_fault", interrupt_settlement)
    with pytest.raises(signal_type, match="settlement signal") as raised:
        apply_generator_transaction(root, request)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "ordinary primary"
    monkeypatch.setattr(owner, "_fault", lambda _phase: None)
    assert recover_generator_transaction(root)
    assert (package / "existing.py").read_bytes() == before


def test_grouped_process_control_during_settlement_remains_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.cli._generator_transaction as owner

    root = tmp_path / "project"
    root.mkdir()
    package = root / "package"
    package.mkdir()
    before = b"before\n"
    (package / "existing.py").write_bytes(before)
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))

    def fail(phase: str) -> None:
        if phase == "new_published:0":
            raise RuntimeError("primary failure")
        if phase == "phase:rolling_back":
            raise BaseExceptionGroup(
                "settlement group",
                [RuntimeError("cleanup failure"), KeyboardInterrupt("shutdown")],
            )

    monkeypatch.setattr(owner, "_fault", fail)
    with pytest.raises(KeyboardInterrupt, match="shutdown") as raised:
        apply_generator_transaction(root, request)

    assert isinstance(raised.value.__cause__, BaseExceptionGroup)
    assert "primary failure" in repr(raised.value.__cause__)
    assert "cleanup failure" in repr(raised.value.__cause__)
    monkeypatch.setattr(owner, "_fault", lambda _phase: None)
    assert recover_generator_transaction(root)
    assert (package / "existing.py").read_bytes() == before


def test_real_sigint_rolls_back_before_the_process_exits(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    before = b"before\n"
    (root / "package" / "existing.py").write_bytes(before)
    request = _request(
        edits=(
            ("package/existing.py", "update_region", b"after\n", before),
            ("package/new.py", "create", b"new\n", None),
        )
    )

    interrupted = _run_crashing_apply(
        root,
        request,
        phase="new_published:0",
        signal_interrupt=True,
    )

    assert interrupted.returncode != 0
    assert (root / "package" / "existing.py").read_bytes() == before
    assert not (root / "package" / "new.py").exists()
    assert not (root / ".cayu" / "generator-transactions" / "active").exists()


def test_recovery_preserves_a_later_edit_and_keeps_the_transaction_explicit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    before = b"before\n"
    (root / "package" / "existing.py").write_bytes(before)
    request = _request(
        edits=(
            ("package/existing.py", "update_region", b"after\n", before),
            ("package/new.py", "create", b"new\n", None),
        )
    )
    crashed = _run_crashing_apply(root, request, phase="new_published:0")
    assert crashed.returncode == 73, crashed.stderr
    (root / "package" / "existing.py").write_bytes(b"operator edit\n")

    with pytest.raises(GeneratorTransactionError, match="preserved all state"):
        recover_generator_transaction(root)

    assert (root / "package" / "existing.py").read_bytes() == b"operator edit\n"
    assert not (root / "package" / "new.py").exists()
    assert (root / ".cayu" / "generator-transactions" / "active").is_dir()


def test_dry_recovery_and_planning_are_write_free_while_work_is_pending(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    before = b"before\n"
    (root / "package" / "existing.py").write_bytes(before)
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))
    crashed = _run_crashing_apply(root, request, phase="phase:committing")
    assert crashed.returncode == 73, crashed.stderr
    state = root / ".cayu" / "generator-transactions"
    before_state = {
        path.relative_to(state).as_posix(): path.read_bytes()
        for path in state.rglob("*")
        if path.is_file()
    }

    with pytest.raises(GeneratorTransactionError, match="rerun without --dry-run"):
        recover_generator_transaction(root, dry_run=True)
    with (
        pytest.raises(GeneratorTransactionError, match="must be recovered"),
        generator_planning_guard(root),
    ):
        pass

    assert before_state == {
        path.relative_to(state).as_posix(): path.read_bytes()
        for path in state.rglob("*")
        if path.is_file()
    }


def test_two_processes_cannot_interleave_one_project_transaction(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    before = b"before\n"
    (root / "package" / "existing.py").write_bytes(before)
    request = _request(
        edits=(
            ("package/existing.py", "update_region", b"after\n", before),
            ("package/new.py", "create", b"new\n", None),
        )
    )
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    script = """
import json
import os
import sys
import time
from pathlib import Path

import cayu.cli._generator_transaction as owner

root = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
ready = Path(sys.argv[3])
release = Path(sys.argv[4])
blocking = sys.argv[5] == "blocking"
request = owner.GeneratorTransactionRequest(
    schema_version=payload["schema_version"],
    slice_name=payload["slice_name"],
    tool_name=payload["tool_name"],
    effect=payload["effect"],
    authoring_state=payload["authoring_state"],
    edits=tuple(
        owner.GeneratorTransactionEdit(
            path=item["path"],
            operation=item["operation"],
            content=bytes.fromhex(item["content"]),
            content_sha256=item["content_sha256"],
            preimage_sha256=item["preimage_sha256"],
        )
        for item in payload["edits"]
    ),
    preconditions=tuple(
        owner.GeneratorTransactionPrecondition(**item)
        for item in payload["preconditions"]
    ),
    verification_commands=tuple(payload["verification_commands"]),
)

def pause(phase):
    if blocking and phase == "phase:committing":
        ready.write_text("ready", encoding="utf-8")
        deadline = time.monotonic() + 10
        while not release.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("test release was not published")
            time.sleep(0.01)

owner._fault = pause
owner.apply_generator_transaction(root, request)
"""
    command = [
        sys.executable,
        "-c",
        script,
        str(root),
        _serialized_request(request),
        str(ready),
        str(release),
    ]
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")}
    first = subprocess.Popen(
        [*command, "blocking"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
    )
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    second = subprocess.Popen(
        [*command, "ordinary"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
    )
    time.sleep(0.2)
    assert second.poll() is None
    assert (root / "package" / "existing.py").read_bytes() == before
    assert not (root / "package" / "new.py").exists()

    release.write_text("release", encoding="utf-8")
    first_output = first.communicate(timeout=15)
    second_output = second.communicate(timeout=15)

    assert first.returncode == 0, first_output
    assert second.returncode == 0, second_output
    assert (root / "package" / "existing.py").read_bytes() == b"after\n"
    assert (root / "package" / "new.py").read_bytes() == b"new\n"


def test_malformed_active_record_fails_closed_and_preserves_state(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package").mkdir()
    before = b"before\n"
    (root / "package" / "existing.py").write_bytes(before)
    request = _request(edits=(("package/existing.py", "update_region", b"after\n", before),))
    crashed = _run_crashing_apply(root, request, phase="phase:committing")
    assert crashed.returncode == 73, crashed.stderr
    active = root / ".cayu" / "generator-transactions" / "active"
    journal = active / "journal.jsonl"
    journal.write_bytes(
        journal.read_bytes().replace(b'"schema_version":1', b'"schema_version":2', 1)
    )

    with pytest.raises(GeneratorTransactionError, match="digest does not match"):
        recover_generator_transaction(root)

    assert active.is_dir()
    assert (root / "package" / "existing.py").read_bytes() == before
