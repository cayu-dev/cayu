from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


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
