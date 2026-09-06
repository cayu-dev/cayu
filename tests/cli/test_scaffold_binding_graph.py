"""Exact local declaration identities and fail-closed dependency propagation."""

import pytest

from cayu.cli.scaffold_check import _check_import_inertness

MODULES = {
    "domain/types.py": """from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

LIMIT = 64 * 1024 * 1024
LABELS = ("one", "two")
class Mode(StrEnum):
    READY = "ready"
    OTHER = "other"
class Configured(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
class Entry(Configured):
    mode: Literal[Mode.READY] = Mode.READY
    labels: tuple[str, ...] = LABELS
class Other(Configured):
    mode: Literal[Mode.OTHER] = Mode.OTHER

def entry_factory() -> Entry:
    raise RuntimeError("factory must not execute at declaration")
""",
    "domain/__init__.py": "from .types import Entry, Other, Configured, Mode, LIMIT, entry_factory\n",
    "domain/typed.py": """from typing import Annotated, Any, TypeVar
from pydantic import Field, computed_field
from domain import Entry, Other, Configured, Mode, LIMIT, entry_factory
from datetime import timedelta
from asyncio import Future
from cayu import RetryPolicy

Choice = Annotated[Entry | Other, Field(discriminator="mode")]
ModelT = TypeVar("ModelT", bound=Configured)
TRANSITIONS = {Mode.READY: (Mode.OTHER,)}
VALUES = {Mode.READY.value: Mode.OTHER.value}
POLICY = RetryPolicy(max_attempts=3)
class Packet(Configured):
    entry: Entry = Field(default_factory=Entry)
    deferred: Entry = Field(default_factory=entry_factory)
    choice: Choice
    labels: list[str] = Field(default_factory=list, max_length=len(Mode), exclude_if=lambda value: not value)
    size: int = Field(default=LIMIT, le=64 * 1024 * 1024)
    description: str = f"limit={LIMIT}"
    @computed_field
    @property
    def summary(self) -> str:
        raise RuntimeError("property must not execute at declaration")

def wait(future: Future[Any], timeout=timedelta(seconds=30), state=Mode.READY):
    pass
""",
}


def write(root, modules):
    for name, source in modules.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


def test_local_schema_graph(tmp_path):
    write(tmp_path, MODULES)
    assert _check_import_inertness(tmp_path) == ()


@pytest.mark.parametrize("module", ["domain/types.py", "domain/__init__.py", "vendor/helper.py"])
def test_dependency_effect_invalidates_schema_proofs(tmp_path, module):
    modules = dict(MODULES)
    modules["domain/types.py"] += "\nimport vendor.helper\n"
    modules["vendor/helper.py"] = "VALUE = 1\n"
    modules[module] += '\nopen("executed", "w").write("effect")\n'
    write(tmp_path, modules)
    findings = _check_import_inertness(tmp_path)
    assert findings
    assert any(d.parameters.get("reason") == "unproven_class_base" for d in findings)
    assert not (tmp_path / "executed").exists()


@pytest.mark.parametrize(
    "change",
    [
        "Entry = unknown\n",
        "del Entry\n",
        "Entry.__get_pydantic_core_schema__ = unknown\n",
        "Mode.__getattr__ = unknown\n",
        "entry_factory.__signature__ = unknown\n",
        "from vendor import *\n",
        "def f(arg: (Entry := unknown)):\n    pass\n",
        "if condition:\n    from vendor import Entry\n",
    ],
)
def test_rebound_or_mutated_identity_stays_closed(tmp_path, change):
    modules = dict(MODULES)
    modules["domain/types.py"] += change
    write(tmp_path, modules)
    assert _check_import_inertness(tmp_path)


@pytest.mark.parametrize(
    "hook",
    ["__init__", "__new__", "__getattr__", "__getattribute__", "__get_pydantic_core_schema__"],
)
def test_enum_hooks_are_not_schema_or_member_proofs(tmp_path, hook):
    modules = dict(MODULES)
    modules["domain/types.py"] = modules["domain/types.py"].replace(
        '    READY = "ready"',
        f'    def {hook}(self, *args):\n        raise RuntimeError("effect")\n    READY = "ready"',
    )
    write(tmp_path, modules)
    assert _check_import_inertness(tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        '    def __get_pydantic_core_schema__(cls, *args):\n        raise RuntimeError("effect")\n',
        '    def __init_subclass__(cls):\n        raise RuntimeError("effect")\n',
        "    __signature__ = unknown_descriptor\n",
        "    member = unknown_descriptor\n",
    ],
)
def test_model_hook_or_descriptor_stays_closed(tmp_path, body):
    modules = dict(MODULES)
    modules["domain/types.py"] = modules["domain/types.py"].replace(
        "class Entry(Configured):", "class Entry(Configured):\n" + body
    )
    write(tmp_path, modules)
    assert _check_import_inertness(tmp_path)


@pytest.mark.parametrize("shadow", ["http.py", "http/__init__.py", "asyncio.py", "pydantic.py"])
def test_external_package_identity_shadowing(tmp_path, shadow):
    write(
        tmp_path,
        MODULES
        | {
            "operations/server.py": "from http.server import ThreadingHTTPServer\nclass Server(ThreadingHTTPServer):\n    pass\n",
            shadow: 'raise RuntimeError("effect")\n',
        },
    )
    assert _check_import_inertness(tmp_path)


@pytest.mark.parametrize(
    "source",
    [
        "from pydantic import BaseModel\nclass Model(BaseModel):\n    pass\ninstance = Model()\n",
        "from cayu import RetryPolicy\npolicy = RetryPolicy(**unknown)\n",
        "from cayu import RetryPolicy\npolicy = RetryPolicy(max_attempts=construct())\n",
        "from datetime import timedelta\nvalue = timedelta(seconds=unknown)\n",
        "from asyncio import Future\nFuture = unknown\nvalue: Future[int]\n",
        "from pydantic import BaseModel, Field\nclass Model(BaseModel):\n    value: list[str] = Field(default_factory=lambda arg=construct(): [])\n",
        "from pydantic import BaseModel, Field\nclass Model(BaseModel):\n    value: list[str] = Field(default_factory=unknown_callable)\n",
        "from pydantic import BaseModel, Field\nclass Model(BaseModel):\n    value: list[str] = Field(default_factory=construct())\n",
    ],
)
def test_opaque_construction_and_signature_inputs_stay_closed(tmp_path, source):
    write(tmp_path, {"domain/model.py": source})
    assert _check_import_inertness(tmp_path)


def test_factory_body_is_deferred_but_default_expression_is_not(tmp_path):
    source = """from pydantic import BaseModel, Field
class Model(BaseModel):
    value: list[str] = Field(default_factory=lambda: open("executed", "w"))
"""
    write(tmp_path, {"domain/model.py": source})
    assert _check_import_inertness(tmp_path) == ()
    assert not (tmp_path / "executed").exists()
    write(
        tmp_path,
        {"domain/model.py": source.replace("lambda:", 'lambda arg=open("executed", "w"):')},
    )
    assert _check_import_inertness(tmp_path)
    assert not (tmp_path / "executed").exists()


def test_graph_is_bounded_without_local_subclasses(tmp_path, monkeypatch):
    monkeypatch.setattr("cayu.cli.scaffold_check._MAX_DECLARATION_MODULES", 2)
    write(
        tmp_path,
        {"domain/a.py": "VALUE = 1\n", "domain/b.py": "VALUE = 1\n", "domain/c.py": "VALUE = 1\n"},
    )
    findings = _check_import_inertness(tmp_path)
    assert findings[0].parameters["reason"] == "declaration_graph_limit"


def test_next_scan_does_not_reuse_previous_proof(tmp_path):
    write(tmp_path, MODULES)
    assert _check_import_inertness(tmp_path) == ()
    with (tmp_path / "domain/types.py").open("a") as stream:
        stream.write("\nEntry.__get_pydantic_core_schema__ = unknown\n")
    assert _check_import_inertness(tmp_path)


def test_generated_service_uses_local_schema_graph(tmp_path, monkeypatch):
    from tests.cli import test_scaffold_declarations as generated

    modules = generated.COMPOSED | MODULES
    modules["domain/contracts.py"] = (
        "from domain.typed import Packet\n" + modules["domain/contracts.py"]
    )
    # Retain the generated helper's base-module injection negative control.
    modules["domain/__init__.py"] += "from .base import Configured as Contract\n"
    monkeypatch.setattr(generated, "COMPOSED", modules)
    generated.test_generated_composed_service(tmp_path)


def test_primary_expression_survives_dependent_base_failures(tmp_path):
    modules = dict(MODULES)
    modules["domain/typed.py"] += """
class Broken(Configured):
    values: list[str] = Field(default_factory=unknown_callable)
class Dependent(Broken):
    pass
"""
    write(tmp_path, modules)
    findings = _check_import_inertness(tmp_path)
    assert any(
        d.parameters.get("reason") == "unsupported_expression"
        and d.parameters.get("symbol") == "Field"
        for d in findings
    )
    assert any(
        d.parameters.get("reason") == "unproven_class_base"
        and "domain/typed.py" in d.parameters["blocking_modules"]
        for d in findings
    )


def test_explicit_helper_primary_is_reported_without_inheritance(tmp_path):
    write(
        tmp_path,
        {
            "domain/example.py": "import vendor.helper\n",
            "vendor/helper.py": 'open("executed", "w").write("effect")\n',
        },
    )
    findings = _check_import_inertness(tmp_path)
    assert any(d.path == "vendor/helper.py:1" for d in findings)
    assert not (tmp_path / "executed").exists()


@pytest.mark.parametrize("base", ["", "from http.server import ThreadingHTTPServer\n"])
def test_leaf_class_descriptor_is_rejected(tmp_path, base):
    parent = "(ThreadingHTTPServer)" if base else ""
    write(
        tmp_path,
        {"domain/example.py": base + f"class Leaf{parent}:\n    value = unknown_descriptor\n"},
    )
    assert _check_import_inertness(tmp_path)


def test_literal_alias_and_function_are_not_class_bases(tmp_path):
    write(
        tmp_path,
        {
            "domain/example.py": """
VALUE = 1
def factory():
    return []
class First(VALUE):
    pass
class Second(factory):
    pass
"""
        },
    )
    findings = _check_import_inertness(tmp_path)
    assert len(findings) == 2
    assert all(d.parameters["reason"] == "unproven_class_base" for d in findings)


def test_rebound_local_enum_member_lookup_stays_closed(tmp_path):
    write(
        tmp_path,
        {
            "domain/example.py": """from enum import StrEnum
class Mode(StrEnum):
    READY = "ready"
Mode = unknown
TABLE = {Mode.READY: "value"}
"""
        },
    )
    findings = _check_import_inertness(tmp_path)
    assert any(d.parameters.get("expression_kind") == "Attribute" for d in findings)


@pytest.mark.parametrize(
    "source",
    [
        "import vendor",
        "from vendor import helper",
        "import vendor.helper",
        "from vendor.nested import helper",
        "import vendor.nested",
    ],
)
def test_namespace_package_imports_are_declarative(tmp_path, source):
    write(
        tmp_path,
        {
            "domain/example.py": source + "\n",
            "vendor/helper.py": "VALUE = 1\n",
            "vendor/nested/helper.py": "VALUE = 1\n",
        },
    )
    assert _check_import_inertness(tmp_path) == ()


def test_relative_namespace_import_keeps_local_model_proof(tmp_path):
    write(
        tmp_path,
        {
            "domain/example.py": "from . import helpers\nfrom .helpers import model\n"
            "from .helpers.model import Model\nclass Child(Model):\n    pass\n",
            "domain/helpers/model.py": "from pydantic import BaseModel\n"
            "class Model(BaseModel):\n    value: int = 1\n",
        },
    )
    assert _check_import_inertness(tmp_path) == ()


@pytest.mark.parametrize("child", ["missing", "effectful", "unparseable", "symlink"])
def test_namespace_package_child_must_have_safe_source(tmp_path, child):
    write(tmp_path, {"domain/example.py": "from vendor import helper\n"})
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    helper = vendor / "helper.py"
    if child == "effectful":
        helper.write_text('open("executed", "w").write("effect")\n')
    elif child == "unparseable":
        helper.write_text("def\n")
    elif child == "symlink":
        target = tmp_path / "safe.txt"
        target.write_text("VALUE = 1\n")
        helper.symlink_to(target)
    findings = _check_import_inertness(tmp_path)
    assert findings
    if child == "effectful":
        assert any(d.path == "vendor/helper.py:1" for d in findings)
    assert not (tmp_path / "executed").exists()


def test_namespace_import_checks_regular_parent_initializer(tmp_path):
    write(
        tmp_path,
        {
            "domain/example.py": "import vendor.nested\n",
            "vendor/__init__.py": 'open("executed", "w").write("effect")\n',
            "vendor/nested/helper.py": "VALUE = 1\n",
        },
    )
    assert any(d.path == "vendor/__init__.py:1" for d in _check_import_inertness(tmp_path))
    assert not (tmp_path / "executed").exists()


def test_module_file_takes_precedence_over_namespace_directory(tmp_path):
    write(
        tmp_path,
        {
            "domain/example.py": "import vendor\n",
            "vendor.py": 'open("executed", "w").write("effect")\n',
            "vendor/helper.py": "VALUE = 1\n",
        },
    )
    assert any(d.path == "vendor.py:1" for d in _check_import_inertness(tmp_path))
    assert not (tmp_path / "executed").exists()


def test_module_file_blocks_nested_namespace_resolution(tmp_path):
    write(
        tmp_path,
        {
            "domain/example.py": "import vendor.parent.child\n",
            "vendor/parent.py": 'open("executed", "w").write("effect")\n',
            "vendor/parent/child/helper.py": "VALUE = 1\n",
        },
    )
    assert _check_import_inertness(tmp_path)
    assert not (tmp_path / "executed").exists()


def test_regular_package_takes_precedence_over_module_file(tmp_path):
    write(
        tmp_path,
        {
            "domain/example.py": "import vendor.parent.child\n",
            "vendor/parent.py": 'open("executed", "w").write("effect")\n',
            "vendor/parent/__init__.py": "VALUE = 1\n",
            "vendor/parent/child/helper.py": "VALUE = 1\n",
        },
    )
    assert _check_import_inertness(tmp_path) == ()
    assert not (tmp_path / "executed").exists()
