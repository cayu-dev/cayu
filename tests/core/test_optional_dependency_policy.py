from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from cayu.support_bundles import _OPTIONAL_DISTRIBUTIONS


def _normalized(requirements: list[str]) -> set[str]:
    return {str(Requirement(requirement)) for requirement in requirements}


def test_all_extra_contains_every_declared_runtime_extra_exactly() -> None:
    root = Path(__file__).parents[2]
    with (root / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)

    extras = project["project"]["optional-dependencies"]
    policy = project["tool"]["cayu"]["optional-dependency-policy"]
    combined = set().union(*(_normalized(extras[name]) for name in policy["all-runtime-extras"]))
    assert _normalized(extras["all"]) == combined
    assert set(policy["excluded-from-all"]) == {"browser"}


def test_support_bundle_optional_packages_match_every_supported_runtime_extra() -> None:
    root = Path(__file__).parents[2]
    with (root / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)

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
