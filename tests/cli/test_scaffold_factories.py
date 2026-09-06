"""Deferred factory registration must not admit import-time construction."""

import pytest
from tests.cli.test_scaffold_declarations import COMPOSED, write

from cayu.cli.scaffold_check import _check_import_inertness


@pytest.mark.parametrize(
    "factory",
    [
        "list",
        "dict",
        "tuple",
        "set",
        "frozenset",
        "lambda: construct()",
        "lambda: []",
        'lambda: {"ready": [1]}',
    ],
)
def test_inherited_deferred_field_factory(tmp_path, factory):
    modules = dict(COMPOSED)
    modules["domain/contracts.py"] = (
        modules["domain/contracts.py"]
        .replace('Field(default="ready")', f"Field(default_factory={factory})")
        .replace("value: str", "value: list[str]")
    )
    write(tmp_path, modules)
    assert _check_import_inertness(tmp_path) == ()


@pytest.mark.parametrize(
    "prefix, factory",
    [
        ("", "list()"),
        ("", "construct()"),
        ("", "unknown"),
        ("", "lambda value=construct(): []"),
        ("", "object.factory"),
        ("list = unknown\n", "list"),
        ("from vendor import list\n", "list"),
        ("", "Configured()"),
        ("", "Configured"),
    ],
)
def test_unproven_factory_is_rejected(tmp_path, prefix, factory):
    write(
        tmp_path,
        {
            "domain/model.py": prefix
            + f"""from pydantic import BaseModel, Field
class Configured(BaseModel):
    value: list[str] = Field(default_factory={factory})
class Child(Configured):
    pass
"""
        },
    )
    findings = _check_import_inertness(tmp_path)
    assert findings
    assert any(d.parameters["reason"] == "unsupported_expression" for d in findings)
    assert any(d.parameters["reason"] == "unproven_class_base" for d in findings)


@pytest.mark.parametrize("shadow", ["module", "class", "package"])
def test_factory_identity_shadowing(tmp_path, shadow):
    source = """from pydantic import BaseModel, Field
class Model(BaseModel):
    value: list[str] = Field(default_factory=list)
"""
    if shadow == "module":
        source += "list = unknown\n"
    elif shadow == "class":
        source = source.replace("    value:", "    list = unknown\n    value:")
    modules = {"domain/model.py": source}
    if shadow == "package":
        modules["pydantic.py"] = "raise RuntimeError('must not execute')\n"
    write(tmp_path, modules)
    assert _check_import_inertness(tmp_path)


def test_diagnostics_distinguish_primary_from_dependent_without_source_literals(tmp_path):
    write(
        tmp_path,
        {
            "domain/model.py": """from pydantic import BaseModel, Field
class Parent(BaseModel):
    value: list[str] = Field(default_factory=construct("private-value"))
class Child(Parent):
    pass
"""
        },
    )
    primary, dependent = _check_import_inertness(tmp_path)
    assert primary.parameters["reason"] == "unsupported_expression"
    assert primary.parameters["symbol"] == "Field"
    assert dependent.parameters["reason"] == "unproven_class_base"
    assert dependent.parameters["symbol"] == "Parent"
    assert "private-value" not in primary.model_dump_json()
