from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_EXAMPLE = Path(__file__).parents[2] / "examples" / "microsandbox_network_default_live.py"


def _load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("microsandbox_network_default_live", _EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "requirement",
    [
        "microsandbox==0.6.6; extra == 'microsandbox'",
        'Microsandbox == 0.6.6 ; extra == "microsandbox"',
    ],
    ids=("normalized", "valid-whitespace"),
)
def test_live_check_reads_supported_version_from_cayu_distribution_requirement(
    requirement: str,
) -> None:
    example = _load_example()

    assert example.declared_microsandbox_version([requirement]) == "0.6.6"


@pytest.mark.parametrize(
    "requirements",
    [
        None,
        [],
        ["microsandbox>=0.6.6,<0.7; extra == 'microsandbox'"],
        ["microsandbox==0.6.*; extra == 'microsandbox'"],
        [
            "microsandbox==0.6.6; extra == 'microsandbox'",
            "microsandbox==0.6.7; extra == 'microsandbox'",
        ],
    ],
    ids=("missing-metadata", "missing-requirement", "range", "wildcard", "duplicate"),
)
def test_live_check_rejects_non_exact_microsandbox_requirements(
    requirements: list[str] | None,
) -> None:
    example = _load_example()

    with pytest.raises(RuntimeError, match="exactly one exact microsandbox requirement"):
        example.declared_microsandbox_version(requirements)
