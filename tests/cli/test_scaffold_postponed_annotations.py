"""Variable annotations Python leaves unevaluated are not scanned as import-time code."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from cayu.cli.scaffold_check import _check_import_inertness, _without_postponed_annotations

_POSTPONED = "from __future__ import annotations\n"

# The module from issue #3952. Lines 6 and 10 were refused under the future import.
_ISSUE_MODULE = """from __future__ import annotations

import threading
from contextvars import ContextVar

_GIT_CANCELLED: ContextVar[threading.Event | None] = ContextVar("git_cancelled", default=None)


class Holder:
    current: ContextVar[threading.Event | None]


def run(event: ContextVar[threading.Event | None]) -> None:
    return None
"""

_IMPORTS = "import threading\nfrom contextvars import ContextVar\n"
_ANNOTATION = "ContextVar[threading.Event | None]"


def _findings(root: Path, source: str) -> list[tuple[str, object]]:
    return [(d.path, d.parameters["reason"]) for d in _check_import_inertness(_write(root, source))]


def _line(source: str, text: str) -> str:
    return f"operations/workspaces.py:{source.splitlines().index(text) + 1}"


def test_issue_module_is_accepted_with_unquoted_annotations(tmp_path):
    assert _findings(tmp_path, _ISSUE_MODULE) == []


def test_module_without_the_future_import_is_scanned_as_before(tmp_path):
    source = _ISSUE_MODULE.removeprefix(_POSTPONED)
    findings = _findings(tmp_path, source)
    assert (
        _line(
            source,
            f'_GIT_CANCELLED: {_ANNOTATION} = ContextVar("git_cancelled", default=None)',
        ),
        "unsupported_expression",
    ) in findings
    assert (_line(source, f"    current: {_ANNOTATION}"), "unsupported_expression") in findings


@pytest.mark.parametrize(
    "body",
    [
        f"VALUE: {_ANNOTATION}\n",
        f"if True:\n    VALUE: {_ANNOTATION} = None\n",
        f"try:\n    VALUE: {_ANNOTATION} = None\nexcept ImportError:\n    VALUE = None\n",
        f"class Holder(object):\n    current: {_ANNOTATION} = None\n",
    ],
    ids=["bare", "if", "try", "object-base"],
)
def test_postponed_variable_annotations_are_accepted(tmp_path, body):
    assert _findings(tmp_path, _POSTPONED + _IMPORTS + body) == []


def test_postponed_annotation_keeps_its_value_scanned(tmp_path):
    source = _POSTPONED + _IMPORTS + f"VALUE: {_ANNOTATION} = open('data.txt').read()\n"
    assert _findings(tmp_path, source) == [("operations/workspaces.py:4", "import_time_effect")]


_EVALUATING_CLASSES = {
    "pydantic": "from pydantic import BaseModel\nclass Holder(BaseModel):\n",
    "named-tuple": "from typing import NamedTuple\nclass Holder(NamedTuple):\n",
    "typed-dict": "from typing import TypedDict\nclass Holder(TypedDict):\n",
    "dataclass": "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Holder:\n",
    "decorated": "from typing import final\n@final\nclass Holder:\n",
    "local-base": "class Base:\n    pass\nclass Holder(Base):\n",
}


@pytest.mark.parametrize(
    "header", list(_EVALUATING_CLASSES.values()), ids=list(_EVALUATING_CLASSES)
)
def test_annotations_of_classes_that_may_read_them_are_still_scanned(tmp_path, header):
    field = f"    current: {_ANNOTATION}"
    source = _POSTPONED + _IMPORTS + header + field + "\n"
    findings = _check_import_inertness(_write(tmp_path, source))
    assert [(d.path, d.parameters["reason"]) for d in findings] == [
        (_line(source, field), "unsupported_expression")
    ]
    assert "from __future__ import annotations" in (findings[0].hint or "")


# Classes the scan refuses for another reason first, so only the transform shows
# whether their annotations are kept.
_HEADERS = {
    "keyword": ("from abc import ABCMeta\nclass Holder(metaclass=ABCMeta):\n", False),
    "shadowed-object": ("object = type('Base', (), {})\nclass Holder(object):\n", False),
    "nested-in-model": (
        "from pydantic import BaseModel\nclass Model(BaseModel):\n    class Holder:\n",
        False,
    ),
    "nested-in-plain": ("class Outer:\n    class Holder:\n", True),
}
if sys.version_info >= (3, 12):
    _HEADERS["type-params"] = ("class Holder[T]:\n", False)


@pytest.mark.parametrize("header, postponed", list(_HEADERS.values()), ids=list(_HEADERS))
def test_class_annotations_are_postponed_only_in_plain_classes(header, postponed):
    indent = header.splitlines()[-1].index("class") + 4
    source = _POSTPONED + _IMPORTS + header + " " * indent + f"current: {_ANNOTATION}\n"
    tree = _without_postponed_annotations(ast.parse(source))
    (annotated,) = [node for node in ast.walk(tree) if isinstance(node, ast.AnnAssign)]
    if postponed:
        assert isinstance(annotated.annotation, ast.Constant)
        assert annotated.annotation.value == _ANNOTATION
    else:
        assert isinstance(annotated.annotation, ast.Subscript)


def test_annotations_are_kept_without_the_future_import():
    source = _IMPORTS + f"VALUE: {_ANNOTATION}\nclass Holder:\n    current: {_ANNOTATION}\n"
    tree = _without_postponed_annotations(ast.parse(source))
    assert all(
        isinstance(node.annotation, ast.Subscript)
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
    )


def _write(root: Path, source: str) -> Path:
    path = root / "operations" / "workspaces.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return root
