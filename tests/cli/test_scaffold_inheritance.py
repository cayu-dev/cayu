from __future__ import annotations

import json
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.scaffold_check import _check_import_inertness


def _write_modules(root: Path, modules: dict[str, str]) -> None:
    for name, source in modules.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


@pytest.mark.parametrize(
    "modules",
    [
        {
            "domain/errors.py": "class ApplicationError(RuntimeError):\n    pass\n\n"
            "class RecoveryBlocked(ApplicationError):\n    pass\n"
        },
        {
            "domain/base.py": "class ApplicationError(RuntimeError):\n    pass\n",
            "domain/errors.py": "from .base import ApplicationError as Base\n"
            "class RecoveryBlocked(Base):\n    pass\n",
        },
        {
            "domain/base.py": "class ApplicationError(RuntimeError):\n    pass\n",
            "domain/__init__.py": "from .base import ApplicationError\n",
            "domain/errors.py": "from domain import ApplicationError\n"
            "class RecoveryBlocked(ApplicationError):\n    pass\n",
        },
        {
            "tools/base.py": "from cayu import Tool, ToolSpec\n"
            "class ProductTool(Tool):\n"
            "    spec = ToolSpec(name='example', description='Example', parameters={})\n"
            "    def __init__(self, service):\n        self.service = service\n"
            "    @property\n    def service_name(self):\n        return self.service.name\n",
            "tools/learning.py": "from tools.base import ProductTool\n"
            "class RecallLessons(ProductTool):\n"
            "    async def run(self, args, ctx):\n        return await self.service.recall(args)\n",
        },
        {
            "domain/errors.py": "class Left:\n    pass\nclass Right:\n    pass\n"
            "class Both(Left, Right):\n    label: str = 'both'\n"
        },
    ],
    ids=["same-module", "relative-alias", "package-reexport", "tool-base", "multiple-bases"],
)
def test_static_local_inheritance_is_inert(tmp_path: Path, modules: dict[str, str]) -> None:
    _write_modules(tmp_path, modules)
    assert _check_import_inertness(tmp_path) == ()


@pytest.mark.parametrize(
    "source",
    [
        "class Child(UnknownBase):\n    pass\n",
        "class Child(Base):\n    pass\nclass Base:\n    pass\n",
        "class Base:\n    pass\nBase = unknown\nclass Child(Base):\n    pass\n",
        "class Base:\n    pass\ndel Base\nclass Child(Base):\n    pass\n",
        "class Base:\n    pass\nfrom vendor import other as Base\nclass Child(Base):\n    pass\n",
        "from vendor import *\nclass Base:\n    pass\nclass Child(Base):\n    pass\n",
        "class Base:\n    pass\nclass Child(Base()):\n    pass\n",
        "class Base:\n    pass\nclass Child(Base, metaclass=Unknown):\n    pass\n",
        "def decorate(cls):\n    return cls\n@decorate\nclass Base:\n    pass\n"
        "class Child(Base):\n    pass\n",
        "class Base:\n    def __init_subclass__(cls):\n        pass\n"
        "class Child(Base):\n    pass\n",
        "class Base:\n    __init_subclass__ = unknown\nclass Child(Base):\n    pass\n",
        "class Base:\n    pass\nclass Child(Base):\n    value = unknown_descriptor\n",
        "class Base:\n    value = unknown_descriptor\nclass Child(Base):\n    pass\n",
        "class Base:\n    pass\nclass Child(Base):\n    value = construct()\n",
        "class Base:\n    pass\n@decorate\nclass Child(Base):\n    pass\n",
        "class Base:\n    pass\nclass Child(Base[object]):\n    pass\n",
        "class Base:\n    pass\nclass Outer:\n    Base = unknown\n"
        "    class Child(Base):\n        pass\n",
        "if True:\n    class Base:\n        pass\nclass Child(Base):\n    pass\n",
    ],
    ids=[
        "unresolved",
        "forward-reference",
        "rebound",
        "deleted",
        "import-shadow",
        "wildcard",
        "base-call",
        "metaclass",
        "base-decorator",
        "subclass-hook",
        "aliased-hook",
        "child-descriptor",
        "base-descriptor",
        "body-call",
        "child-decorator",
        "subscription",
        "class-shadow",
        "conditional-base",
    ],
)
def test_unproven_local_inheritance_stays_closed(tmp_path: Path, source: str) -> None:
    _write_modules(tmp_path, {"domain/errors.py": source})
    findings = _check_import_inertness(tmp_path)
    assert any(item.code == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in findings)


@pytest.mark.parametrize(
    "base_source",
    [
        "class Base:\n    def __init_subclass__(cls):\n"
        "        open('executed', 'w').write('hook')\n",
        "class Parent:\n    def __init_subclass__(cls):\n"
        "        open('executed', 'w').write('hook')\nclass Base(Parent):\n    pass\n",
        "class Base:\n    pass\nBase.__init_subclass__ = unknown\n",
        "class Base:\n    pass\nopen('executed', 'w').write('module')\n",
        "def __getattr__(name):\n    open('executed', 'w').write('getattr')\n"
        "class Base:\n    pass\n",
        "from domain.errors import Child as Base\n",
        "class Base:\n    pass\nBase = unknown\n",
    ],
    ids=["hook", "inherited-hook", "mutation", "module-effect", "getattr", "cycle", "rebind"],
)
def test_explicit_external_local_base_is_checked_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_source: str,
) -> None:
    # vendor is outside the declared application directories; only the explicit
    # base import gives the checker a reason to inspect it.
    _write_modules(
        tmp_path,
        {
            "vendor/base.py": base_source,
            "domain/errors.py": "from vendor.base import Base\nclass Child(Base):\n    pass\n",
        },
    )
    monkeypatch.chdir(tmp_path)
    findings = _check_import_inertness(tmp_path)
    assert any(item.code == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in findings)
    assert not (tmp_path / "executed").exists()


def test_generated_service_accepts_exception_and_tool_hierarchies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    command = [
        "new",
        "inheritance-probe",
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
    ]
    assert main([*command, "--dry-run"]) == 0
    capsys.readouterr()
    assert not (tmp_path / "inheritance-probe").exists()
    assert main(command) == 0
    capsys.readouterr()
    project = tmp_path / "inheritance-probe"
    monkeypatch.chdir(project)
    monkeypatch.setenv(
        "PRODUCT_AUTH_TOKENS_JSON",
        json.dumps(
            {
                "probe-customer-token": {"tenant_id": "probe-tenant", "subject_id": "probe-user"},
            }
        ),
    )
    monkeypatch.setenv("CAYU_OPERATOR_BEARER_TOKEN", "probe-operator-token")
    check = ["check", "--deploy", "--fail-on", "warning", "--json"]
    assert main(check) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []
    _write_modules(
        project,
        {
            "domain/errors.py": '"""Inert application exception hierarchy."""\n\n'
            "class ApplicationError(RuntimeError):\n    pass\n\n"
            "class RecoveryBlocked(ApplicationError):\n    pass\n"
        },
    )
    assert main(check) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []
    _write_modules(
        project,
        {
            "tools/base.py": "from cayu import Tool\nclass ProductTool(Tool):\n    pass\n",
            "tools/learning.py": "from tools.base import ProductTool\n"
            "class ProductRecallLessonsTool(ProductTool):\n"
            "    async def run(self, args, ctx):\n        return 'lessons'\n",
        },
    )
    assert main(check) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []


@pytest.mark.parametrize(
    "package_source",
    [
        "open('executed', 'w').write('package')\n",
        "def __getattr__(name):\n    open('executed', 'w').write('getattr')\n",
    ],
)
def test_local_base_package_initialization_is_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package_source: str,
) -> None:
    _write_modules(
        tmp_path,
        {
            "vendor/__init__.py": package_source,
            "vendor/base.py": "class Base:\n    pass\n",
            "domain/errors.py": "from vendor.base import Base\nclass Child(Base):\n    pass\n",
        },
    )
    monkeypatch.chdir(tmp_path)
    assert _check_import_inertness(tmp_path)
    assert not (tmp_path / "executed").exists()


def test_duplicate_import_alias_does_not_prove_a_base(tmp_path: Path) -> None:
    _write_modules(
        tmp_path,
        {
            "domain/base.py": "class Base:\n    pass\n"
            "class Hook:\n    def __init_subclass__(cls):\n        pass\n",
            "domain/errors.py": "from domain.base import Base, Hook as Base\n"
            "class Child(Base):\n    pass\n",
        },
    )
    assert _check_import_inertness(tmp_path)


def test_exception_handler_rebinding_does_not_prove_a_base(tmp_path: Path) -> None:
    _write_modules(
        tmp_path,
        {
            "domain/errors.py": "class Base:\n    pass\ntry:\n    pass\nexcept Exception as Base:\n    pass\n"
            "class Child(Base):\n    pass\n"
        },
    )
    assert _check_import_inertness(tmp_path)


def test_symlinked_imported_base_is_unproven(tmp_path: Path) -> None:
    _write_modules(
        tmp_path,
        {
            "actual/base.py": "class Base:\n    pass\n",
            "domain/errors.py": "from vendor.base import Base\nclass Child(Base):\n    pass\n",
        },
    )
    (tmp_path / "vendor").symlink_to(tmp_path / "actual", target_is_directory=True)
    assert _check_import_inertness(tmp_path)


@pytest.mark.parametrize(
    "source",
    [
        "class Base:\n    pass\ndef function(arg: (Base := unknown)):\n    pass\n"
        "class Child(Base):\n    pass\n",
        "if True:\n    from unknown_vendor import *\nclass Base:\n    pass\n"
        "class Child(Base):\n    pass\n",
        "class Base:\n    pass\nclass Child(Base, **unknown_mapping):\n    pass\n",
    ],
)
def test_opaque_bindings_and_class_keywords_stay_closed(tmp_path: Path, source: str) -> None:
    _write_modules(tmp_path, {"domain/errors.py": source})
    assert _check_import_inertness(tmp_path)


def test_descriptor_hook_is_not_executed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_modules(
        tmp_path,
        {
            "vendor/descriptor.py": "class Descriptor:\n"
            "    def __set_name__(self, owner, name):\n"
            "        open('executed', 'w').write('descriptor')\ninstance = Descriptor()\n",
            "domain/errors.py": "from vendor.descriptor import instance\n"
            "class Base:\n    pass\nclass Child(Base):\n    value = instance\n",
        },
    )
    monkeypatch.chdir(tmp_path)
    findings = _check_import_inertness(tmp_path)
    # Report the opaque namespace value and its helper's constructor, rather
    # than pointing only at the dependent base declaration.
    assert any(item.path == "domain/errors.py:5" for item in findings)
    assert any(item.path == "vendor/descriptor.py:4" for item in findings)
    assert not (tmp_path / "executed").exists()


def test_unrelated_modules_are_not_discovered(tmp_path: Path) -> None:
    _write_modules(
        tmp_path,
        {
            "vendor/unrelated.py": "open('executed', 'w').write('unrelated')\n",
            "domain/errors.py": "class Base:\n    pass\nclass Child(Base):\n    pass\n",
        },
    )
    assert _check_import_inertness(tmp_path) == ()


def test_builtin_exception_subtypes_remain_supported(tmp_path: Path) -> None:
    _write_modules(
        tmp_path,
        {
            "domain/errors.py": "class Missing(FileNotFoundError):\n    pass\nclass RecoveryMissing(Missing):\n    pass\n"
        },
    )
    assert _check_import_inertness(tmp_path) == ()


@pytest.mark.parametrize("enum_base", ["Enum", "IntEnum", "StrEnum"])
@pytest.mark.parametrize("hook", ["__init__", "__new__"])
@pytest.mark.parametrize("layout", ["same-module", "imported", "reexported"])
def test_inherited_enum_member_hooks_are_unproven(
    tmp_path: Path,
    enum_base: str,
    hook: str,
    layout: str,
) -> None:
    marker = tmp_path / "executed"
    base = (
        f"from enum import {enum_base}\nclass Base({enum_base}):\n"
        f"    def {hook}(self, value):\n"
        f'        with open({str(marker)!r}, "w") as marker:\n'
        '            marker.write("inherited enum hook")\n'
    )
    if hook == "__new__":
        constructor = {"Enum": "object", "IntEnum": "int", "StrEnum": "str"}[enum_base]
        arguments = "self" if enum_base == "Enum" else "self, value"
        base += f"        return {constructor}.__new__({arguments})\n"
    value = repr("value") if enum_base == "StrEnum" else "1"
    child = f"class Child(Base):\n    VALUE = {value}\n"
    if layout == "same-module":
        modules = {"domain/errors.py": base + child}
    else:
        modules = {"domain/base.py": base}
        if layout == "imported":
            modules["domain/errors.py"] = "from .base import Base\n" + child
        else:
            modules["domain/__init__.py"] = "from .base import Base\n"
            modules["domain/errors.py"] = "from domain import Base\n" + child
    _write_modules(tmp_path, modules)
    findings = _check_import_inertness(tmp_path)
    assert any(item.code == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in findings)
    assert not marker.exists()


@pytest.mark.parametrize("enum_base", ["Enum", "IntEnum", "StrEnum"])
@pytest.mark.parametrize("imported", [False, True])
def test_enum_metaclass_cannot_activate_a_proven_mixin(
    tmp_path: Path,
    enum_base: str,
    imported: bool,
) -> None:
    mixin = 'class Mixin:\n    def __init__(self, value):\n        raise RuntimeError("effect")\n'
    value = repr("value") if enum_base == "StrEnum" else "1"
    child = f"from enum import {enum_base}\nclass Child(Mixin, {enum_base}):\n    VALUE = {value}\n"
    modules = (
        {"domain/base.py": mixin, "domain/errors.py": "from .base import Mixin\n" + child}
        if imported
        else {"domain/errors.py": mixin + child}
    )
    _write_modules(tmp_path, modules)
    assert _check_import_inertness(tmp_path)


def test_direct_enum_declarations_remain_supported(tmp_path: Path) -> None:
    _write_modules(
        tmp_path,
        {"domain/status.py": 'from enum import Enum\nclass Status(Enum):\n    READY = "ready"\n'},
    )
    assert _check_import_inertness(tmp_path) == ()


def test_cli_blocks_inherited_enum_execution_before_app_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    marker = project / "executed"
    _write_modules(
        project,
        {
            "domain/errors.py": "from enum import Enum\nclass Base(Enum):\n"
            "    def __init__(self, value):\n"
            f'        with open({str(marker)!r}, "w") as marker:\n'
            '            marker.write("import-time effect")\n'
            "class Child(Base):\n    VALUE = 1\n"
        },
    )
    registration = project / "agents/registration.py"
    registration.write_text("from domain.errors import Child\n" + registration.read_text())
    monkeypatch.chdir(project)
    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["manifest_fingerprint"] == "unavailable"
    assert any(item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in report["diagnostics"])
    assert not marker.exists()


@pytest.mark.parametrize(
    "binding",
    [
        "from abc import ABC as Root",
        "from cayu import Tool as Root",
        "from cayu.core.tools import Tool as Root",
    ],
)
def test_reviewed_external_roots_propagate_inheritance_safety(tmp_path: Path, binding: str) -> None:
    _write_modules(
        tmp_path,
        {
            "tools/base.py": binding + "\nclass Base(Root):\n    pass\n",
            "tools/child.py": "from tools.base import Base\nclass Child(Base):\n    pass\n",
        },
    )
    assert _check_import_inertness(tmp_path) == ()


@pytest.mark.parametrize(
    "binding", ["from cayu import AgentSpec as Root", "from typing import Protocol as Root"]
)
def test_other_external_metaclasses_do_not_propagate_a_local_proof(
    tmp_path: Path,
    binding: str,
) -> None:
    _write_modules(
        tmp_path,
        {
            "domain/errors.py": binding + "\nclass Base(Root):\n    pass\n"
            "class Child(Base):\n    pass\n"
        },
    )
    assert _check_import_inertness(tmp_path)


@pytest.mark.parametrize(
    "helper_import",
    [
        "import vendor.hooks",
        "import vendor.hooks as hooks",
        "from vendor import hooks",
        "from vendor.hooks import hook",
        "from . import hooks",
        "from .hooks import hook",
        "if True:\n    import vendor.hooks",
        "class Namespace:\n    import vendor.hooks",
    ],
)
def test_imported_base_helper_cannot_install_a_subclass_hook(
    tmp_path: Path, helper_import: str
) -> None:
    _write_modules(
        tmp_path,
        {
            "vendor/__init__.py": "",
            "vendor/base.py": "class Base:\n    pass\n" + helper_import + "\n",
            "vendor/hooks.py": "from vendor.base import Base\n"
            "def hook(cls):\n    raise RuntimeError('import-time hook')\n"
            "Base.__init_subclass__ = classmethod(hook)\n",
            "domain/errors.py": "from vendor.base import Base\nclass Child(Base):\n    pass\n",
        },
    )
    assert _check_import_inertness(tmp_path)


@pytest.mark.parametrize("effectful", [False, True])
@pytest.mark.parametrize("location", ["base", "package", "same-module"])
def test_local_base_checks_transitive_helpers(
    tmp_path: Path, effectful: bool, location: str
) -> None:
    base = "class Base:\n    pass\n"
    helper_import = "import vendor.helper\n"
    modules = {
        "vendor/__init__.py": helper_import if location == "package" else "",
        "vendor/helper.py": "from vendor.effects import VALUE\n",
        "vendor/effects.py": (
            "VALUE = 1\nopen('executed', 'w').write('effect')\n" if effectful else "VALUE = 1\n"
        ),
        "vendor/base.py": base + (helper_import if location == "base" else ""),
        "domain/errors.py": (
            base + helper_import if location == "same-module" else "from vendor.base import Base\n"
        )
        + "class Child(Base):\n    pass\n",
    }
    _write_modules(tmp_path, modules)
    assert bool(_check_import_inertness(tmp_path)) is effectful


def test_cli_blocks_helper_installed_hook_before_app_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    marker = project / "executed"
    _write_modules(
        project,
        {
            "vendor/__init__.py": "",
            "vendor/base.py": "class Base:\n    pass\nimport vendor.hooks\n",
            "vendor/hooks.py": "from vendor.base import Base\n"
            "def hook(cls):\n"
            f"    with open({str(marker)!r}, 'w') as marker:\n"
            "        marker.write('subclass hook executed')\n"
            "Base.__init_subclass__ = classmethod(hook)\n",
            "domain/errors.py": "from vendor.base import Base\nclass Child(Base):\n    pass\n",
        },
    )
    registration = project / "agents/registration.py"
    registration.write_text("from domain.errors import Child\n" + registration.read_text())
    monkeypatch.chdir(project)
    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["manifest_fingerprint"] == "unavailable"
    assert any(item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in report["diagnostics"])
    assert not marker.exists()


@pytest.mark.parametrize("imported", ["base", "helper"])
def test_local_base_proof_prefers_package_over_same_named_module(
    tmp_path: Path, imported: str
) -> None:
    _write_modules(
        tmp_path,
        {
            "vendor/__init__.py": "",
            "vendor/base.py": "class Base:\n    pass\nimport vendor.helper\n",
            "vendor/helper.py": "VALUE = 1\n",
            f"vendor/{imported}/__init__.py": "class Base:\n    pass\n"
            "open('executed', 'w').write('package effect')\n",
            "domain/errors.py": "from vendor.base import Base\nclass Child(Base):\n    pass\n",
        },
    )
    assert _check_import_inertness(tmp_path)


def test_local_base_dependency_proof_is_bounded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("cayu.cli.scaffold_check._MAX_DECLARATION_MODULES", 64)
    modules = {
        "domain/errors.py": "class Base:\n    pass\nimport vendor.helper0\n"
        "class Child(Base):\n    pass\n",
    }
    for index in range(65):
        modules[f"vendor/helper{index}.py"] = (
            f"import vendor.helper{index + 1}\n" if index < 64 else "VALUE = 1\n"
        )
    _write_modules(tmp_path, modules)
    assert _check_import_inertness(tmp_path)


def test_symlinked_local_base_helper_is_unproven(tmp_path: Path) -> None:
    _write_modules(
        tmp_path,
        {
            "vendor/base.py": "class Base:\n    pass\nimport vendor.helper\n",
            "actual/helper.py": "VALUE = 1\n",
            "domain/errors.py": "from vendor.base import Base\nclass Child(Base):\n    pass\n",
        },
    )
    (tmp_path / "vendor/helper.py").symlink_to(tmp_path / "actual/helper.py")
    assert _check_import_inertness(tmp_path)
