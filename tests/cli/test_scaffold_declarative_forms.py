"""Ordinary model, dataclass and alias declarations prove inert; causes report once."""

from __future__ import annotations

from pathlib import Path

import pytest

from cayu.cli.scaffold_check import _check_import_inertness

_MODELS = """from typing import Any, Literal, Self
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    field_serializer,
    model_serializer,
    model_validator,
)

LIMIT = 6 * 60 * 60.0
MEMORY = 8 * 1024**3


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class Entry(_Config):
    name: str


class Packet(_Config):
    entry: Entry | None = Field(default=None, exclude_if=lambda value: value is None)
    entries: tuple[Entry, ...] = Field(default=(), max_length=16)
    forward: "Entry | None" = None
    nested: list["Entry"] = Field(default_factory=lambda: [])
    phase: Literal["one", "two"] = Field(default="one", exclude_if=lambda value: value == "one")
    timeout: float = Field(default=60.0, le=LIMIT)
    memory: int = Field(default=MEMORY, ge=16 * 1024**2, le=1024**4)
    _cache: dict[str, Any] = PrivateAttr(default_factory=dict)
    _count: int = PrivateAttr(default=0)

    @model_validator(mode="wrap")
    @classmethod
    def read(cls, value: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        return handler(value)

    @model_serializer(mode="wrap")
    def write(self, handler: Any) -> dict[str, Any]:
        return handler(self)


class Labelled(_Config):
    label: str

    @field_serializer("label", when_used="json")
    def write_label(self, value: str) -> str:
        return value
"""


def _write(root: Path, modules: dict[str, str]) -> None:
    for name, source in modules.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _findings(root: Path, modules: dict[str, str]) -> list[tuple[str, str]]:
    _write(root, modules)
    return [(d.path, d.parameters["reason"]) for d in _check_import_inertness(root)]


@pytest.mark.parametrize("future", ["", "from __future__ import annotations\n"])
def test_pydantic_annotations_defaults_and_hooks_are_declarative(tmp_path, future):
    modules = {
        "domain/contracts.py": future + _MODELS,
        "domain/uses.py": "from domain.contracts import Packet\n"
        "class Envelope(Packet):\n    sequence: int = 0\n",
    }
    assert _findings(tmp_path, modules) == []


def test_stdlib_declarations_are_declarative(tmp_path):
    modules = {
        "domain/records.py": """import os
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from typing import NamedTuple, Protocol

from cayu import AgentSpec
from pydantic import BaseModel, Field

NEVER = datetime(1970, 1, 1, tzinfo=UTC)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
Factory = Callable[..., AbstractContextManager[str]]


class Row(tuple):
    __slots__ = ()


class Point(NamedTuple):
    x: int
    y: int = 0


class Reader(Protocol):
    name: str

    def read(self) -> str: ...


@dataclass(frozen=True)
class Facts:
    probes: Mapping[str, int] = field(default_factory=dict)
    held: frozenset[str] = frozenset()
    environment: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))


class MoreFacts(Facts):
    pass


class Launch(BaseModel):
    agent: AgentSpec | None = Field(default_factory=AgentSpec)
""",
    }
    assert _findings(tmp_path, modules) == []


@pytest.mark.parametrize(
    "declaration",
    [
        "    value: int = Field(default=2 ** 100)",
        "    value: int = Field(default=2147483649 * 1)",
        "    value: int = Field(default=0, exclude_if=lambda value, cached=load(): value)",
        "    value: int = Field(default=0, field_title_generator=lambda name, info: name)",
        "    value: list[int] = Field(default_factory=load)",
        "    value: 'Unknown'",
        "    value: 'not a type('",
        "    _value: int = PrivateAttr(default=load())",
        "    _value: int = PrivateAttr(default_factory=load)",
        "    _value: int = PrivateAttr(load())",
    ],
)
def test_computed_model_declarations_stay_rejected(tmp_path, declaration):
    source = (
        "from pydantic import BaseModel, Field, PrivateAttr\n"
        "from vendor import load\n"
        "class Unknown:\n    pass\n"
        f"class Model(BaseModel):\n{declaration}\n"
    )
    assert _findings(tmp_path, {"domain/model.py": source}) == [
        ("domain/model.py:6", "unsupported_expression")
    ]


@pytest.mark.parametrize(
    "serializer",
    [
        "@model_serializer(mode=load())\n    def write(self) -> dict:\n        return {}",
        "@model_serializer(return_type=dict)\n    def write(self) -> dict:\n        return {}",
        "@field_serializer(load())\n    def write(self, value: int) -> int:\n        return value",
        "@model_serializer\n    def write(self) -> Unknown:\n        return Unknown()",
    ],
)
def test_unproven_serializers_stay_rejected(tmp_path, serializer):
    source = (
        "from pydantic import BaseModel, field_serializer, model_serializer\n"
        "from vendor import load\n"
        "class Unknown:\n    pass\n"
        f"class Model(BaseModel):\n    value: int = 0\n    {serializer}\n"
    )
    findings = _findings(tmp_path, {"domain/model.py": source})
    # The decorator, or the return annotation Pydantic builds a schema from.
    assert findings in (
        [("domain/model.py:7", "unsupported_expression")],
        [("domain/model.py:8", "unsupported_expression")],
    )


@pytest.mark.parametrize(
    "statement",
    [
        "STARTED = datetime.now()",
        "STARTED = datetime(1970, 1, 1, tzinfo=load())",
        "STARTED = datetime(year=load())",
        "Factory = AbstractContextManager[load()]",
        "@dataclass\nclass Facts:\n    value: list = field(default_factory=lambda cached=load(): cached)",
        "@dataclass\nclass Facts:\n    value: frozenset = frozenset(load())",
    ],
)
def test_executing_module_declarations_stay_rejected(tmp_path, statement):
    source = (
        "from contextlib import AbstractContextManager\n"
        "from dataclasses import dataclass, field\n"
        "from datetime import datetime\n"
        "from vendor import load\n" + statement + "\n"
    )
    line = 4 + len(statement.splitlines())
    assert _findings(tmp_path, {"domain/values.py": source}) == [
        (f"domain/values.py:{line}", "unsupported_expression")
    ]


def test_module_level_effect_is_still_reported_beside_declarations(tmp_path):
    modules = {
        "domain/contracts.py": _MODELS,
        "domain/client.py": "from domain.contracts import Packet\n"
        "from vendor import connect\n"
        "CLIENT = connect()\n"
        "class Request(Packet):\n    pass\n",
    }
    assert _findings(tmp_path, modules) == [("domain/client.py:3", "unsupported_expression")]


def test_base_cause_is_reported_once_at_its_definition(tmp_path):
    modules = {
        "domain/contracts.py": """from pydantic import BaseModel, Field
from vendor import load

class _Contract(BaseModel):
    value: list[str] = Field(default_factory=load)

class First(_Contract):
    pass

class Second(_Contract):
    pass
""",
        "domain/more.py": "from domain.contracts import First, _Contract\n"
        "class Third(_Contract):\n    pass\n"
        "class Fourth(First):\n    item: First | None = None\n",
    }
    assert _findings(tmp_path, modules) == [("domain/contracts.py:5", "unsupported_expression")]


def test_dependency_effect_is_reported_once_for_every_importer(tmp_path):
    modules = {
        "domain/settings.py": "from vendor import load\nSETTINGS = load()\n",
        "domain/contracts.py": "from domain.settings import SETTINGS\n" + _MODELS,
        "domain/uses.py": "from domain.contracts import Entry, Packet\n"
        "class Envelope(Packet):\n    entry_copy: Entry | None = None\n",
    }
    assert _findings(tmp_path, modules) == [("domain/settings.py:2", "unsupported_expression")]


def test_uninheritable_base_is_reported_at_its_hook(tmp_path):
    modules = {
        "tools/bounds.py": """from cayu import Tool

class BoundedTool(Tool):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
""",
        "tools/product.py": """from pydantic import Field
from tools.bounds import BoundedTool
from vendor import load

class ProductTool(BoundedTool):
    pass

class MemoryTool(BoundedTool):
    limit = load()

class SearchTool(ProductTool):
    pass
""",
    }
    _write(tmp_path, modules)
    findings = _check_import_inertness(tmp_path)
    assert [(d.path, d.parameters["reason"]) for d in findings] == [
        ("tools/bounds.py:4", "unproven_base_definition"),
        ("tools/product.py:9", "unsupported_expression"),
    ]
    base = findings[0]
    assert base.parameters["symbol"] == "BoundedTool"
    assert base.parameters["construct"] == "__init_subclass__"
    assert base.parameters["dependent_classes"] == 2


def test_uninheritable_base_alone_is_one_finding(tmp_path):
    modules = {
        "tools/bounds.py": "from cayu import Tool\n"
        "class BoundedTool(Tool):\n"
        "    def __init_subclass__(cls, **kwargs):\n"
        "        super().__init_subclass__(**kwargs)\n",
        "tools/product.py": "from tools.bounds import BoundedTool\n"
        "class ProductTool(BoundedTool):\n    pass\n",
        "tools/search.py": "from tools.product import ProductTool\n"
        "class SearchTool(ProductTool):\n    pass\n",
    }
    assert _findings(tmp_path, modules) == [("tools/bounds.py:3", "unproven_base_definition")]


def test_unexplained_cyclic_failures_are_still_reported(tmp_path):
    modules = {
        "domain/first.py": "from domain.second import Second\nclass First(Second):\n    pass\n",
        "domain/second.py": "from domain.first import First\nclass Second(First):\n    pass\n",
    }
    findings = _findings(tmp_path, modules)
    assert findings
    assert {reason for _path, reason in findings} == {"unproven_class_base"}


@pytest.mark.parametrize(
    "source, expected",
    [
        ("LIMIT = 65536.0 ** 64\n", []),
        (
            "from pydantic import BaseModel, Field\n"
            "class Model(BaseModel):\n    value: float = Field(default=65536.0 ** 64)\n",
            [("domain/values.py:3", "unsupported_expression")],
        ),
    ],
)
def test_float_power_overflow_is_unproven_not_a_crash(tmp_path, source, expected):
    assert _findings(tmp_path, {"domain/values.py": source}) == expected


_POSTPONED = "from __future__ import annotations\n"


@pytest.mark.parametrize(
    "future, decorator, returns",
    [
        ("", "@model_serializer", '"Annotated[int, Field(default=0)]"'),
        (_POSTPONED, "@model_serializer", '"Annotated[int, Field(default=0)]"'),
        (_POSTPONED, "@model_serializer(mode='plain')", "Annotated[int, Field(default=0)]"),
        ("", "@computed_field", '"Annotated[int, Field(default=0)]"'),
        (_POSTPONED, "@computed_field", "Annotated[int, Field(default=0)]"),
    ],
)
def test_deferred_return_annotation_sees_names_bound_later_in_the_class(
    tmp_path, future, decorator, returns
):
    # Pydantic resolves the annotation in the finished class namespace, where
    # the later method shadows the imported Field.
    source = (
        future + "from typing import Annotated\n"
        "from pydantic import BaseModel, Field, computed_field, model_serializer\n"
        "class Model(BaseModel):\n"
        "    value: int = 0\n"
        f"    {decorator}\n"
        f"    def dump(self) -> {returns}:\n"
        "        return 0\n"
        "    def Field(self, default):\n"
        "        open('executed', 'w').write('payload')\n"
    )
    assert _findings(tmp_path, {"domain/model.py": source})
    assert not (tmp_path / "executed").exists()


@pytest.mark.parametrize(
    "local_module, statement",
    [
        (
            "contextlib.py",
            "from contextlib import AbstractContextManager\nAlias = AbstractContextManager[int]\n",
        ),
        (
            "pydantic.py",
            "from pydantic import ModelWrapValidatorHandler\n"
            "Alias = ModelWrapValidatorHandler[int]\n",
        ),
    ],
)
def test_project_local_subscription_substitutes_stay_rejected(tmp_path, local_module, statement):
    modules = {
        local_module: "class AbstractContextManager:\n"
        "    def __class_getitem__(cls, item):\n"
        "        open('executed', 'w').write('payload')\n"
        "ModelWrapValidatorHandler = AbstractContextManager\n",
        "domain/values.py": statement,
    }
    assert ("domain/values.py:2", "unsupported_expression") in _findings(tmp_path, modules)


@pytest.mark.parametrize("future", ["", _POSTPONED])
@pytest.mark.parametrize(
    "decorators, returns",
    [
        ("@computed_field\n    @property", "Annotated[int, Field(default=0)]"),
        ("@model_serializer", "Annotated[int, Field(default=0)]"),
        ("@model_serializer", '"Model"'),
    ],
)
def test_return_annotation_resolves_in_the_completed_class(tmp_path, future, decorators, returns):
    # Python 3.14 defers every annotation, and Pydantic resolves a deferred one
    # in the finished class namespace, where later methods shadow imports and
    # module classes. So even an unquoted annotation cannot rely on the names
    # bound when the method is defined.
    source = (
        future + "from typing import Annotated\n"
        "from pydantic import BaseModel, Field, computed_field, model_serializer\n"
        "class Model(BaseModel):\n"
        "    pass\n"
        "class Output(BaseModel):\n"
        "    value: int = 0\n"
        f"    {decorators}\n"
        f"    def dump(self) -> {returns}:\n"
        "        return 0\n"
        "    def Field(self, default):\n"
        "        open('executed', 'w').write('payload')\n"
        "    def Model(self, info: \"print('effect') or int\"):\n"
        "        return {}\n"
    )
    assert _findings(tmp_path, {"domain/model.py": source})
    assert not (tmp_path / "executed").exists()


def test_class_member_shadowing_a_module_model_is_not_that_model(tmp_path):
    # Pydantic resolves "Output" to the serializer method, not the module model,
    # and then evaluates the method's quoted parameter annotation.
    source = (
        "from pydantic import BaseModel, model_serializer\n"
        "class Output(BaseModel):\n    pass\n"
        "class Model(BaseModel):\n"
        "    @model_serializer\n"
        '    def Output(self, info: "print(\'effect\') or int") -> "Output":\n'
        "        return {}\n"
    )
    assert _findings(tmp_path, {"domain/model.py": source})


@pytest.mark.parametrize("alias", ["Optional['Thing']", "list['Thing']"])
def test_imported_forward_alias_cannot_trust_exporter_namespace(tmp_path: Path, alias: str) -> None:
    findings = _findings(
        tmp_path,
        {
            "domain/__init__.py": "",
            "domain/a.py": f"""from typing import Optional
from pydantic import BaseModel
class Thing(BaseModel):
    value: str
Alias = {alias}
""",
            "domain/b.py": """from pydantic import BaseModel
from domain.a import Alias
class Thing:
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        raise RuntimeError("import-time payload")
class Model(BaseModel):
    value: Alias = None
""",
        },
    )
    assert any(path.startswith("domain/b.py:") for path, _reason in findings)


@pytest.mark.parametrize("imported", [False, True])
@pytest.mark.parametrize("decorator", ["model_serializer", 'field_serializer("value")'])
@pytest.mark.parametrize("safe", [False, True])
def test_inherited_serializer_override_checks_return_annotation(
    tmp_path, imported, decorator, safe
):
    parent = f"""from pydantic import BaseModel, model_serializer, field_serializer
class Parent(BaseModel):
    value: int = 0
    @{decorator}
    def dump(self) -> int:
        return 0
class Middle(Parent):
    pass
"""
    annotation = "int" if safe else "(open('executed', 'w').write('payload'), int)[1]"
    child = f"""class Child(Middle):
    def dump(self) -> {annotation!r}:
        return 1
"""
    modules = (
        {"parent.py": parent, "child.py": "from parent import Middle\n" + child}
        if imported
        else {"child.py": parent + child}
    )
    findings = _findings(tmp_path, modules)
    if safe:
        assert findings == []
    else:
        assert any(path.startswith("child.py:") for path, _reason in findings)


@pytest.mark.parametrize("imported", [False, True])
@pytest.mark.parametrize("decorator", ["model_serializer", 'field_serializer("value")'])
@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("middle_binding", ["pass", "dump: int = 0", "dump: int"])
def test_serializer_rebinding_checks_other_mro_bases(
    tmp_path, imported, decorator, safe, middle_binding
):
    annotation = "dict" if safe else "(open('executed', 'w').write('payload'), dict)[1]"
    earlier = f"""from pydantic import BaseModel
class Earlier(BaseModel):
    value: int = 0
    def dump(self) -> {annotation!r}:
        return {{}}
class Middle(Earlier):
    {middle_binding}
"""
    later = f"""from pydantic import BaseModel, model_serializer, field_serializer
class Later(BaseModel):
    value: int = 0
    @{decorator}
    def dump(self) -> dict:
        return {{}}
"""
    child = "class Child(Middle, Later):\n    pass\n"
    modules = (
        {
            "earlier.py": earlier,
            "later.py": later,
            "child.py": "from earlier import Middle\nfrom later import Later\n" + child,
        }
        if imported
        else {"child.py": earlier + later.replace("BaseModel, ", "") + child}
    )
    findings = _findings(tmp_path, modules)
    if safe:
        assert findings == []
    else:
        assert any(path.startswith("child.py:") for path, _reason in findings)
