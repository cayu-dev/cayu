"""Base-bound generated entrance, persistent replay, and local-only delivery."""

import importlib
import os
import subprocess
import sys

import pytest

from cayu.cli.project import project_context
from tests.qualification.repository_maintenance_application import maintenance_project_files
from tests.qualification.repository_maintenance_case import (
    SEED_BASE_REVISION,
    materialize_seed_repository,
)
from tests.qualification.repository_maintenance_fixture import capture_repository_fixture
from tests.qualification.test_repository_maintenance_application import (
    project as project,
)
from tests.qualification.test_repository_maintenance_application import (
    test_public_workflow_seals_verifies_and_replays as exercise_journey,
)


@pytest.mark.parametrize(
    "repository_fixture", [True, "without-project", "stale-base", "stale-manifest", "stale-size"]
)
def test_bound_repository_public_journey(project, tmp_path, monkeypatch, repository_fixture):
    exercise_journey(
        project,
        tmp_path,
        monkeypatch,
        incorrect_probe=False,
        backend="sqlite",
        lose_push_ack=False,
        lose_github_ack=False,
        managed_worker=repository_fixture in {True, "without-project"},
        workflow_eval=False,
        restart_approval=False,
        repository_fixture=repository_fixture,
    )


@pytest.mark.parametrize("change", ["base", "untracked", "ignored", "seed"])
def test_capture_refuses_before_generation_and_without_git_mutation(tmp_path, change):
    source = tmp_path / "source"
    materialize_seed_repository(source)
    expected = SEED_BASE_REVISION
    if change == "base":
        expected = "0" * 40
    elif change == "untracked":
        (source / "extra").write_bytes(b"untracked")
    elif change == "ignored":
        (source / ".git/info/exclude").write_text("hidden\n")
        (source / "hidden").write_bytes(b"ignored")
    else:
        (source / "range_ops.py").write_bytes(b"different seed")
    before = {
        str(path.relative_to(source)): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError):
        capture_repository_fixture(source, expected_base=expected)
    after = {
        str(path.relative_to(source)): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("invalid", ["boolean-size", "duplicate", "traversal", "seed-digest"])
def test_generator_revalidates_mutated_capture(tmp_path, invalid):
    source = tmp_path / "source"
    materialize_seed_repository(source)
    fixture = capture_repository_fixture(source, expected_base=SEED_BASE_REVISION)
    entries = list(fixture.files)
    path, digest, size, mode = entries[0]
    if invalid == "boolean-size":
        entries[0] = (path, digest, True, mode)
    elif invalid == "duplicate":
        entries.append(entries[0])
    elif invalid == "traversal":
        entries[0] = ("../outside", digest, size, mode)
    else:
        entries = [(p, "0" * 64 if p == "range_ops.py" else d, s, m) for p, d, s, m in entries]
    object.__setattr__(fixture, "files", tuple(entries))
    with pytest.raises(ValueError):
        maintenance_project_files(fixture=fixture)


def test_capture_is_read_only_and_generated_binding_is_complete(project, tmp_path, monkeypatch):
    source = tmp_path / "source"
    materialize_seed_repository(source)
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    captured = capture_repository_fixture(source, expected_base=SEED_BASE_REVISION)
    assert {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
    } == before
    generated = maintenance_project_files(fixture=captured)
    for name, content in generated.items():
        destination = project / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    with project_context(project):
        case = importlib.import_module("domain.maintenance_case")
        checks = importlib.import_module("environments.maintenance_toolchain").maintenance_checks()
        assert tuple(checks[0].command.argv[-3:]) == ("--isolated", *case.ALLOWED_CHANGE_PATHS)
        assert "pythonpath=." in checks[-1].command.argv
        initial = case.corpus_fingerprint()
        assert captured.base_revision == case.SEED_BASE_REVISION
        assert dict(case.BASE_FILES) == {p: (d, s, m) for p, d, s, m in captured.files}
        seed = importlib.import_module("evals.maintenance_seed")
        with pytest.raises(ValueError, match="clean checkout"):
            seed.materialize_seed_repository(tmp_path / "must-not-exist")
        assert not (tmp_path / "must-not-exist").exists()
        monkeypatch.setattr(case, "SEED_BASE_REVISION", "0" * 40)
        assert case.corpus_fingerprint() != initial
        monkeypatch.setattr(case, "SEED_BASE_REVISION", captured.base_revision)
        path, digest, size, mode = captured.files[0]
        for changed in (("0" * 64, size, mode), (digest, size + 1, mode), (digest, size, "100755")):
            monkeypatch.setattr(case, "BASE_FILES", {p: (d, s, m) for p, d, s, m in captured.files})
            case.BASE_FILES[path] = changed
            assert case.corpus_fingerprint() != initial


@pytest.mark.parametrize("kind", ["symlink", "executable-seed", "missing-seed"])
def test_capture_rejects_unsupported_committed_tree(tmp_path, kind):
    from tests.qualification.repository_maintenance_delivery_case import local_git

    source = tmp_path / "source"
    materialize_seed_repository(source)
    if kind == "symlink":
        (source / "alias").symlink_to("range_ops.py")
    elif kind == "executable-seed":
        (source / "range_ops.py").chmod(0o755)
    else:
        (source / "range_ops.py").unlink()
    local_git(source, "add", "--all")
    local_git(
        source,
        "-c",
        "user.name=Q",
        "-c",
        "user.email=q@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "Unsupported fixture",
    )
    base = local_git(source, "rev-parse", "HEAD")
    with pytest.raises(ValueError):
        capture_repository_fixture(source, expected_base=base)
    assert local_git(source, "rev-parse", "HEAD") == base
    assert local_git(source, "status", "--porcelain") == ""


def test_bound_check_commands_use_fixed_case_without_target_configuration(project, tmp_path):
    from tests.qualification.repository_maintenance_case import SEED_FILES

    # Literal test-owned case only; never run arbitrary target-repository code.
    target = tmp_path / "check-source"
    target.mkdir()
    (target / "tests").mkdir()
    for name in ("range_ops.py", "tests/test_range_ops.py"):
        (target / name).write_text(SEED_FILES[name])
    (target / "unrelated.py").write_text("invalid python must not enter scoped checks !!\n")
    seed = tmp_path / "captured"
    materialize_seed_repository(seed)
    fixture = capture_repository_fixture(seed, expected_base=SEED_BASE_REVISION)
    for name, content in maintenance_project_files(fixture=fixture).items():
        destination = project / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    with project_context(project):
        checks = importlib.import_module("environments.maintenance_toolchain").maintenance_checks()
        for check in (checks[0], checks[2], checks[3]):
            argv = check.command.argv
            result = subprocess.run(
                [sys.executable, "-m", "pytest" if check.name == "test" else "ruff", *argv[1:]],
                cwd=target,
                capture_output=True,
                timeout=30,
                env={
                    "PATH": os.defpath,
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            assert result.returncode == (1 if check.name == "test" else 0), (
                result.stdout + result.stderr
            )
        (target / "range_ops.py").write_text(
            SEED_FILES["range_ops.py"].replace("value < upper", "value <= upper")
        )
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *checks[3].command.argv[1:]],
            cwd=target,
            capture_output=True,
            timeout=30,
            env={
                "PATH": os.defpath,
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert (
        target / "unrelated.py"
    ).read_text() == "invalid python must not enter scoped checks !!\n"
