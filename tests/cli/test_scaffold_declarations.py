import pytest

from cayu.cli.scaffold_check import _check_import_inertness

COMPOSED = {
    "domain/base.py": """from pydantic import BaseModel as Model, ConfigDict
class Configured(Model):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)
""",
    "domain/__init__.py": "from .base import Configured as Contract\n",
    "domain/contracts.py": """from domain import Contract as Parent
from pydantic import Field, field_validator
class Grant(Parent):
    value: str = Field(default="ready")
    @field_validator("value")
    @classmethod
    def validate(cls, value):
        return value
class FinalGrant(Grant):
    count: int = 1
""",
    "domain/declarations.py": """from contextvars import ContextVar as Variable
import typing as types
from contextlib import asynccontextmanager as lease
import contextlib as contexts
from enum import StrEnum
Result = types.TypeVar("Result")
Arguments = types.ParamSpec("Arguments")
Shape = types.TypeVarTuple("Shape")
validation: Variable[bool] = Variable("validation", default=False)
Labels = dict[str, tuple[str, ...]]
class State(StrEnum):
    READY = "ready"
TABLE = {State.READY: ("ready",)}
class Lease:
    @lease
    async def acquire(self):
        raise RuntimeError("body must not execute")
        yield self
    @contexts.contextmanager
    def sync(self):
        raise RuntimeError("body must not execute")
        yield self
""",
    "tools/example.py": """from cayu import Tool, ToolSpec
from domain.contracts import FinalGrant
from domain.declarations import Lease
class Concern:
    def describe(self):
        return "example"
class BaseTool(Tool):
    spec = ToolSpec(name="example", description="Example", input_schema={})
class Example(BaseTool, Concern):
    async def run(self, args, ctx):
        return FinalGrant().value
""",
}


def write(root, modules):
    for name, source in modules.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


def test_composed_declarations(tmp_path):
    write(tmp_path, COMPOSED)
    assert _check_import_inertness(tmp_path) == ()


@pytest.mark.parametrize(
    "binding, expression",
    [
        ("from typing import TypeVar as T", 'T("Result")'),
        ("import typing as t", 't.TypeVar("Result", covariant=True)'),
        ("from contextvars import ContextVar as C", 'C("validation", default=False)'),
        ("import contextvars as c", 'c.ContextVar("validation", default={"ready": [1]})'),
    ],
)
def test_declaration_aliases(tmp_path, binding, expression):
    write(tmp_path, {"domain/example.py": f"{binding}\nvalue = {expression}\n"})
    assert _check_import_inertness(tmp_path) == ()


@pytest.mark.parametrize(
    "source",
    [
        "from typing import TypeVar\nvalue = TypeVar(load())",
        'from typing import TypeVar\nvalue = TypeVar("T", bound=unknown)',
        'from typing import TypeVar\nvalue = TypeVar("T", **options)',
        'from typing import TypeVar\nTypeVar = unknown\nvalue = TypeVar("T")',
        'from contextvars import ContextVar\nvalue = ContextVar("v", default=construct())',
        'from contextvars import ContextVar\nvalue = ContextVar("v", default=unknown)',
        "from contextvars import ContextVar\nvalue = ContextVar(*args)",
        'from vendor import ContextVar\nvalue = ContextVar("v", default=False)',
        "import contextlib as c\nc = unknown\nclass Lease:\n    @c.asynccontextmanager\n    async def acquire(self):\n        yield self",
        "from contextlib import asynccontextmanager as cm\nclass Lease:\n    cm = unknown\n    @cm\n    async def acquire(self):\n        yield self",
        "from contextlib import asynccontextmanager\n@asynccontextmanager\nclass Lease:\n    pass",
        "from vendor import contextmanager\nclass Lease:\n    @contextmanager\n    def acquire(self):\n        yield self",
    ],
)
def test_unproven_declarations(tmp_path, source):
    write(tmp_path, {"domain/example.py": source + "\n"})
    findings = _check_import_inertness(tmp_path)
    assert any(
        item.code == "SCAFFOLD_IMPORT_SIDE_EFFECT" and item.path.startswith("domain/example.py:")
        for item in findings
    )


@pytest.mark.parametrize("module", ["typing", "contextvars", "contextlib"])
def test_local_stdlib_shadow(tmp_path, module):
    write(tmp_path, COMPOSED | {f"{module}.py": 'raise RuntimeError("do not import")\n'})
    assert _check_import_inertness(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        '    __annotations__ = {"x": "effect()"}\n',
        '    def __get_pydantic_core_schema__(cls, *args):\n        raise RuntimeError("effect")\n',
        '    def __init_subclass__(cls):\n        raise RuntimeError("effect")\n',
        "    descriptor = unknown\n",
        "    value: CustomType\n",
    ],
)
@pytest.mark.parametrize("target", ["domain/base.py", "domain/contracts.py"])
def test_inherited_models_keep_safeguards(tmp_path, change, target):
    modules = dict(COMPOSED)
    modules[target] += change
    if target == "domain/contracts.py":
        modules[target] += "class Descendant(FinalGrant):\n    pass\n"
    write(tmp_path, modules)
    assert _check_import_inertness(tmp_path)


@pytest.mark.parametrize("effect", [False, True])
def test_model_dependency_chain(tmp_path, effect):
    modules = dict(COMPOSED)
    modules["domain/base.py"] += "\nimport vendor.helper\n"
    modules["vendor/helper.py"] = "import vendor.effects\n"
    modules["vendor/effects.py"] = "VALUE = 1\n" + (
        'open("executed", "w").write("effect")\n' if effect else ""
    )
    write(tmp_path, modules)
    assert bool(_check_import_inertness(tmp_path)) is effect
    assert not (tmp_path / "executed").exists()


def test_generated_composed_service(tmp_path):
    import json
    import os
    import subprocess
    import sys

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PRODUCT_AUTH_TOKENS_JSON"] = json.dumps(
        {"test-token": {"tenant_id": "test", "subject_id": "test"}}
    )
    environment["CAYU_OPERATOR_BEARER_TOKEN"] = "test-operator-token"

    def cli(*args, cwd=tmp_path, expected=0):
        result = subprocess.run(
            [sys.executable, "-m", "cayu", *args],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        return result.stdout

    command = (
        "new",
        "composed",
        "--preset",
        "service",
        "--database",
        "sqlite",
        "--provider",
        "openai-subscription",
        "--execution",
        "none",
        "--agent-name",
        "compound-root",
        "--dir",
        str(tmp_path),
    )
    cli(*command, "--dry-run")
    cli(*command)
    project = tmp_path / "composed"
    write(project, COMPOSED)
    registration = project / "agents/registration.py"
    registration.write_text("from tools.example import Example\n" + registration.read_text())
    check = ("check", "--deploy", "--fail-on", "warning", "--json")
    assert json.loads(cli(*check, cwd=project))["diagnostics"] == []
    marker = project / "executed"
    with (project / "domain/base.py").open("a") as stream:
        stream.write(f"    value: \"(open({str(marker)!r}, 'w').write('effect'), str)[1]\"\n")
    report = json.loads(cli(*check, cwd=project, expected=1))
    assert report["manifest_fingerprint"] == "unavailable"
    assert not marker.exists()


@pytest.mark.parametrize("location", ["same-module", "imported"])
def test_model_metaclass_cannot_activate_mixin_hooks(tmp_path, location):
    mixin = 'class Mixin:\n    def __get_pydantic_core_schema__(cls, *args):\n        raise RuntimeError("effect")\n'
    child = "from pydantic import BaseModel\nclass Grant(Mixin, BaseModel):\n    value: str\n"
    modules = (
        {"domain/model.py": mixin + child}
        if location == "same-module"
        else {"domain/mixin.py": mixin, "domain/model.py": "from .mixin import Mixin\n" + child}
    )
    write(tmp_path, modules)
    assert _check_import_inertness(tmp_path)
