from __future__ import annotations

import runpy
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

_ROOT = Path(__file__).parents[2]
_RUNNER = runpy.run_path(str(_ROOT / "scripts" / "run_ci.py"))
CommandEvidence = _RUNNER["CommandEvidence"]
LocalCiRunner = _RUNNER["LocalCiRunner"]
VerificationScope = _RUNNER["VerificationScope"]
PackageStep = _RUNNER["PackageStep"]
_ensure_exact_head = _RUNNER["_ensure_exact_head"]
_isolated_execution_worktree = _RUNNER["_isolated_execution_worktree"]
_load_package_steps = _RUNNER["_load_package_steps"]
_package_temporary_parent = _RUNNER["_package_temporary_parent"]
_run_general_shard = _RUNNER["_run_general_shard"]
_run_general_shards = _RUNNER["_run_general_shards"]
_run_release_artifacts = _RUNNER["_run_release_artifacts"]
_run_specialist_lane = _RUNNER["_run_specialist_lane"]
_run_sqlite_cancellation_version = _RUNNER["_run_sqlite_cancellation_version"]
_validate_node_environment = _RUNNER["_validate_node_environment"]
_write_proof = _RUNNER["_write_proof"]


def _dry_runner() -> object:
    return LocalCiRunner(root=_ROOT, dry_run=True, keep_going=False)


def test_workflow_routes_canonical_lanes_and_keeps_release_state_tag_only() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "ci.yml").read_text()

    for command in (
        "python3 scripts/run_ci.py --lane static",
        'python3 scripts/run_ci.py --lane general --shard "${{ matrix.shard }}"',
        "python3 scripts/run_ci.py --lane specialist",
        "python3 scripts/run_ci.py --lane sqlite-cancellation",
        "python3 scripts/run_ci.py --lane package",
        "python3 scripts/run_ci.py --lane dashboard",
    ):
        assert command in workflow

    assert "docker/setup-qemu-action@" in workflow
    package_steps = _load_package_steps(_ROOT / "scripts" / "package_ci_steps.yml")
    immutable = next(
        step for step in package_steps if step.name == "Verify immutable release state"
    )
    assert immutable.publishing_only is True
    assert all("uses:" not in step.command for step in package_steps)


def test_general_and_specialist_lane_plans_own_the_pytest_topology() -> None:
    runner = _dry_runner()
    _run_general_shard(runner, 4)
    assert len(runner.evidence) == 1
    assert "--splits 6 --group 4" in runner.evidence[0].command
    assert "not (stress or process or postgres)" in runner.evidence[0].command

    specialist = _dry_runner()
    _run_specialist_lane(specialist, "postgres-conformance-2")
    assert len(specialist.evidence) == 1
    assert "--splits 2 --group 2" in specialist.evidence[0].command
    assert "postgres and not (stress or process)" in specialist.evidence[0].command

    with pytest.raises(ValueError, match="between 1 and 6"):
        _run_general_shard(_dry_runner(), 7)
    with pytest.raises(ValueError, match="unknown specialist lane"):
        _run_specialist_lane(_dry_runner(), "unknown")


def test_local_general_shards_use_bounded_parallel_workers(monkeypatch) -> None:
    from threading import Barrier

    rendezvous = Barrier(2)

    def fake_shard(runner, shard: int) -> None:
        rendezvous.wait(timeout=2)
        runner.evidence.append(
            CommandEvidence(
                label=f"shard {shard}",
                command=f"pytest shard {shard}",
                cwd=".",
                status="PASS",
                returncode=0,
                duration_seconds=0.0,
                output_tail=(),
            )
        )

    monkeypatch.setitem(_run_general_shards.__globals__, "_run_general_shard", fake_shard)
    runner = LocalCiRunner(root=_ROOT, dry_run=False, keep_going=False)
    _run_general_shards(runner, jobs=2)

    assert [item.label for item in runner.evidence] == [
        "shard 1",
        "shard 2",
        "shard 3",
        "shard 4",
        "shard 5",
        "shard 6",
    ]
    with pytest.raises(ValueError, match="between 1 and 6"):
        _run_general_shards(_dry_runner(), jobs=7)


def test_sqlite_lane_repeats_the_canonical_nodes_three_times() -> None:
    runner = _dry_runner()
    _run_sqlite_cancellation_version(runner, "3.12")

    assert [item.status for item in runner.evidence] == ["PLANNED"] * 5
    assert runner.evidence[0].command == "uv python install 3.12"
    assert "--python 3.12" in runner.evidence[1].command
    attempts = runner.evidence[2:]
    assert len(attempts) == 3
    assert len({item.command for item in attempts}) == 1
    assert "<10 canonical SQLite cancellation nodes>" in attempts[0].command


def test_package_plan_skips_publication_policy_unless_explicit(
    tmp_path: Path,
) -> None:
    ordinary = _dry_runner()
    _run_release_artifacts(ordinary, tmp_path / "ordinary", publishing=False)
    skipped = [item.label for item in ordinary.evidence if item.status == "SKIP"]
    assert skipped == [
        "Publishing only: Verify immutable release state",
        "Publishing only: Enforce freshness for release artifacts",
        "Publishing only: Check tag matches project version",
    ]

    publishing = _dry_runner()
    _run_release_artifacts(publishing, tmp_path / "publishing", publishing=True)
    assert all(item.status == "PLANNED" for item in publishing.evidence)
    labels = {item.label for item in publishing.evidence}
    assert "Release artifacts: Verify immutable release state" in labels
    assert "Release artifacts: Enforce freshness for release artifacts" in labels
    assert "Release artifacts: Check tag matches project version" in labels

    lane_choices = _RUNNER["_parse_args"](["--lane", "package", "--dry-run"])
    assert lane_choices.lane == "package"


def test_package_environment_resolves_macos_temporary_aliases(tmp_path: Path, monkeypatch) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(canonical, target_is_directory=True)
    observed: dict[str, str] = {}

    class RecordingRunner:
        root = tmp_path
        dry_run = False

        def run(self, label, command, **kwargs) -> bool:
            del label, command
            if env := kwargs.get("env"):
                observed.update(env)
            return True

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    system_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda command: "/usr/bin/rg" if command == "rg" else system_which(command),
    )
    monkeypatch.setitem(
        _run_release_artifacts.__globals__,
        "_load_package_steps",
        lambda _path: [PackageStep(name="Safe step", command="true", publishing_only=False)],
    )
    monkeypatch.setitem(
        _run_release_artifacts.__globals__, "_prepare_package_environment", lambda _runner: None
    )

    _run_release_artifacts(RecordingRunner(), alias, publishing=False)

    assert Path(observed["RUNNER_TEMP"]) == canonical.resolve()
    assert Path(observed["TMPDIR"]) == canonical.resolve()


def test_package_shell_can_allocate_below_the_canonical_temporary_root(
    tmp_path: Path, monkeypatch
) -> None:
    runner = LocalCiRunner(root=tmp_path, dry_run=False, keep_going=False)

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    system_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda command: "/usr/bin/rg" if command == "rg" else system_which(command),
    )
    monkeypatch.setitem(
        _run_release_artifacts.__globals__,
        "_load_package_steps",
        lambda _path: [
            PackageStep(
                name="Inspect temporary root",
                command=('printf "%s\\n" "$RUNNER_TEMP"; mktemp -d "$RUNNER_TEMP/proof.XXXXXX"'),
                publishing_only=False,
            )
        ],
    )
    monkeypatch.setitem(
        _run_release_artifacts.__globals__, "_prepare_package_environment", lambda _runner: None
    )

    with TemporaryDirectory(
        prefix="cayu-package-test-",
        dir=_package_temporary_parent(),
    ) as temporary:
        canonical = Path(temporary).resolve()
        _run_release_artifacts(runner, canonical, publishing=False)

    observed_root, created = runner.evidence[-1].output_tail
    assert Path(observed_root) == canonical
    assert Path(created).parent == canonical
    assert Path(created) == Path(created).resolve()


def test_exact_head_allows_untracked_evidence_but_rejects_tracked_drift(
    tmp_path: Path,
) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("committed\n")
    subprocess.run(("git", "add", "tracked.txt"), cwd=tmp_path, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Cayu CI",
            "-c",
            "user.email=ci@example.invalid",
            "commit",
            "-qm",
            "initial",
        ),
        cwd=tmp_path,
        check=True,
    )

    (tmp_path / "proof.txt").write_text("untracked\n")
    head, untracked = _ensure_exact_head(tmp_path, "HEAD")
    assert len(head) == 40
    assert untracked == ("?? proof.txt",)

    tracked.write_text("changed\n")
    with pytest.raises(RuntimeError, match="tracked worktree or index changes"):
        _ensure_exact_head(tmp_path, "HEAD")


def test_isolated_execution_excludes_source_untracked_files_and_pins_head(
    tmp_path: Path,
) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("committed\n")
    subprocess.run(("git", "add", "tracked.txt"), cwd=tmp_path, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Cayu CI",
            "-c",
            "user.email=ci@example.invalid",
            "commit",
            "-qm",
            "initial",
        ),
        cwd=tmp_path,
        check=True,
    )
    original = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=tmp_path, text=True
    ).strip()
    (tmp_path / "conftest.py").write_text("raise RuntimeError('must not load')\n")

    with _isolated_execution_worktree(tmp_path, original) as checkout:
        assert not (checkout / "conftest.py").exists()
        assert _ensure_exact_head(checkout, original) == (original, ())


def test_release_preflight_preserves_a_dangling_owned_symlink(tmp_path: Path) -> None:
    dangling = tmp_path / ".release-venv"
    dangling.symlink_to(tmp_path / "missing", target_is_directory=True)
    runner = LocalCiRunner(root=tmp_path, dry_run=False, keep_going=False)

    with pytest.raises(RuntimeError, match=".release-venv"):
        _run_release_artifacts(runner, tmp_path / "temporary", publishing=False)

    assert dangling.is_symlink()


def test_system_node_is_accepted_without_mise() -> None:
    runner = _dry_runner()
    observed: list[tuple[str, ...]] = []

    def record(_label, command, **_kwargs) -> bool:
        observed.append(tuple(command))
        return True

    runner.run = record
    _validate_node_environment(runner)

    assert observed == [("node", "--version"), ("npm", "--version")]


def test_proof_distinguishes_publication_policy_and_skipped_commands(
    tmp_path: Path,
) -> None:
    proof = tmp_path / "proof.md"
    now = datetime(2026, 8, 28, tzinfo=UTC)
    _write_proof(
        proof,
        status="PASS",
        started_at=now,
        finished_at=now,
        root=_ROOT,
        base_sha="a" * 40,
        head_sha="b" * 40,
        merge_base="a" * 40,
        changed_paths=("src/cayu/runtime/tasks.py",),
        scope=VerificationScope(True, True, True),
        source_untracked=("?? local-proof.md",),
        execution_untracked=("?? generated-output.txt",),
        open_file_limits=(256, 8192),
        jobs=2,
        publishing=False,
        evidence=(
            CommandEvidence(
                label="Publishing only: Verify immutable release state",
                command="python scripts/verify_release_state.py",
                cwd=".",
                status="SKIP",
                returncode=None,
                duration_seconds=0.0,
                output_tail=("pass --publishing only when preparing a publication",),
            ),
        ),
        error=None,
    )

    rendered = proof.read_text()
    assert "Status: **PASS**" in rendered
    assert "`0 passed`, `0 failed`, `1 skipped`, `0 planned`" in rendered
    assert "`publishing=false`" in rendered
    assert "### SKIP: Publishing only: Verify immutable release state" in rendered
    assert "Commit tested: `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`" in rendered
    assert "`local-proof.md`" in rendered
    assert "`generated-output.txt`" in rendered
