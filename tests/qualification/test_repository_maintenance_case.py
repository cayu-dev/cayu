"""Qualification of the fixed oracle, not a production application trial."""

import os
import subprocess
import sys

import pytest

from tests.qualification.repository_maintenance_case import (
    ALLOWED_CHANGE_PATHS,
    EXPECTED_RESPONSES,
    PROBES,
    SEED_FILES,
    BehavioralOutcome,
    check_behavioral_responses,
    corpus_fingerprint,
    materialize_seed_repository,
)


def _responses_from_authored_fixture(source):
    # Only literal fixture code authored in this test is executed on the host.
    # Production candidate evaluation must use admitted Docker execution.
    namespace = {}
    exec(compile(source, "authored-range-fixture", "exec"), namespace)
    return [namespace["in_closed_range"](*probe) for probe in PROBES]


def test_seed_fails_and_corrected_fixture_passes_independent_oracle():
    seed = SEED_FILES["range_ops.py"]
    assert len(PROBES) == 105
    assert (
        check_behavioral_responses(_responses_from_authored_fixture(seed))
        is BehavioralOutcome.INCORRECT
    )
    fixed = seed.replace("lower <= value < upper", "lower <= value <= upper")
    assert (
        check_behavioral_responses(_responses_from_authored_fixture(fixed))
        is BehavioralOutcome.PASSED
    )


def test_weakening_project_tests_does_not_change_oracle():
    candidate = dict(SEED_FILES)
    candidate["tests/test_range_ops.py"] = "def test_everything():\n    assert True\n"
    assert (
        check_behavioral_responses(_responses_from_authored_fixture(candidate["range_ops.py"]))
        is BehavioralOutcome.INCORRECT
    )
    assert "tests/test_range_ops.py" in ALLOWED_CHANGE_PATHS
    assert all("qualification" not in path for path in SEED_FILES)


@pytest.mark.parametrize("responses", [None, [], [True], [True] * 106, [1] * 105, ["true"] * 105])
def test_incomplete_or_untyped_responses_do_not_pass(responses):
    assert check_behavioral_responses(responses) is BehavioralOutcome.INVALID_RESPONSE


def test_each_answer_is_required_and_digest_is_deterministic():
    for index, expected in enumerate(EXPECTED_RESPONSES):
        changed = list(EXPECTED_RESPONSES)
        changed[index] = not expected
        assert check_behavioral_responses(changed) is BehavioralOutcome.INCORRECT
    assert (
        corpus_fingerprint()
        == "sha256:95933d96773bcd2cd28f080cf8e8d97b5b42278ed400ba72535ea58342e0a242"
    )


def test_seed_git_identity_is_reproducible_and_existing_directory_is_preserved(tmp_path):
    first = materialize_seed_repository(tmp_path / "first")
    second = materialize_seed_repository(tmp_path / "second")
    assert first == second
    assert first == "219ebbfb37f09b0a08e9ca98ad4474bf4640c06f"
    with pytest.raises(FileExistsError):
        materialize_seed_repository(tmp_path / "first")
    assert (tmp_path / "first" / "range_ops.py").read_text() == SEED_FILES["range_ops.py"]


def test_seed_project_failure_is_reproducible_and_fix_passes(tmp_path):
    repository = tmp_path / "seed"
    materialize_seed_repository(repository)

    def run_project_tests():
        return subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider"],
            cwd=repository,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )

    seeded = run_project_tests()
    assert seeded.returncode == 1
    assert "1 failed, 1 passed" in seeded.stdout
    fixed = SEED_FILES["range_ops.py"].replace("lower <= value < upper", "lower <= value <= upper")
    (repository / "range_ops.py").write_text(fixed, encoding="utf-8")
    corrected = run_project_tests()
    assert corrected.returncode == 0, corrected.stdout + corrected.stderr
    assert "2 passed" in corrected.stdout
    assert (
        check_behavioral_responses(_responses_from_authored_fixture(fixed))
        is BehavioralOutcome.PASSED
    )
