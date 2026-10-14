from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from cayu.support_bundles import _OPTIONAL_DISTRIBUTIONS


def _normalized(requirements: list[str]) -> set[str]:
    return {str(Requirement(requirement)) for requirement in requirements}


def _project() -> dict:
    root = Path(__file__).parents[2]
    with (root / "pyproject.toml").open("rb") as source:
        return tomllib.load(source)


def test_optional_dependency_policy_classifies_every_extra_once() -> None:
    project = _project()
    extras = project["project"]["optional-dependencies"]
    policy = project["tool"]["cayu"]["optional-dependency-policy"]
    classified = [
        *policy["all-runtime-extras"],
        *policy["excluded-from-all"],
        *policy["non-runtime-extras"],
    ]
    assert len(classified) == len(set(classified)), "extras must have exactly one policy role"
    assert set(classified) == extras.keys() - {"all"}, "every extra must have a declared role"
    for field in ("excluded-from-all", "non-runtime-extras"):
        for extra, reason in policy[field].items():
            assert isinstance(reason, str) and reason.strip(), f"{extra} needs an exclusion reason"


def test_all_extra_contains_every_declared_runtime_extra_exactly() -> None:
    project = _project()
    extras = project["project"]["optional-dependencies"]
    policy = project["tool"]["cayu"]["optional-dependency-policy"]
    combined = set().union(*(_normalized(extras[name]) for name in policy["all-runtime-extras"]))
    assert _normalized(extras["all"]) == combined


def test_support_bundle_optional_packages_match_every_supported_runtime_extra() -> None:
    project = _project()
    extras = project["project"]["optional-dependencies"]
    policy = project["tool"]["cayu"]["optional-dependency-policy"]
    supported_extras = (
        *policy["all-runtime-extras"],
        *policy["excluded-from-all"],
    )
    expected = {
        canonicalize_name(Requirement(requirement).name)
        for extra in supported_extras
        for requirement in extras[extra]
    }

    assert tuple(sorted(expected)) == _OPTIONAL_DISTRIBUTIONS
