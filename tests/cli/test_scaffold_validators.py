from __future__ import annotations

from pathlib import Path

import pytest

from cayu.cli.scaffold_check import _check_import_inertness

GRANT = """from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class Grant(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fingerprint: str = Field(default="a", min_length=1)
    components: tuple[str, ...]

    @field_validator("components")
    @classmethod
    def validate_components(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("components must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_grant(self):
        return self
"""


def _check(root: Path, source: str) -> bool:
    domain = root / "domain"
    domain.mkdir(exist_ok=True)
    (domain / "model.py").write_text(source)
    return bool(_check_import_inertness(root))


@pytest.mark.parametrize(
    "binding,decorator",
    [
        ("from pydantic import field_validator", 'field_validator("components")'),
        (
            "from pydantic import field_validator as validate",
            'validate("components", "other", mode="before", check_fields=False)',
        ),
        ("import pydantic as pd", 'pd.field_validator("*", mode="plain")'),
        (
            "import pydantic",
            'pydantic.field_validator("components", mode="wrap", check_fields=None)',
        ),
        ("from pydantic import model_validator as validate", 'validate(mode="before")'),
        ("from pydantic import model_validator", 'model_validator(mode="after")'),
        ("import pydantic as pd", 'pd.model_validator(mode="wrap")'),
    ],
)
def test_validator_registration_is_inert(tmp_path: Path, binding: str, decorator: str) -> None:
    source = f"""from pydantic import BaseModel
{binding}
class Grant(BaseModel):
    components: tuple[str, ...]
    @{decorator}
    @classmethod
    def validate_value(cls, value):
        raise RuntimeError("body must not execute during registration")
"""
    assert not _check(tmp_path, source)


def test_exact_grant_keeps_validation_semantics(tmp_path: Path) -> None:
    assert not _check(tmp_path, GRANT)
    namespace = {}
    exec(GRANT, namespace)
    grant = namespace["Grant"]
    assert grant(components=("a", "b")).components == ("a", "b")
    for components in [("b", "a"), ("a", "a")]:
        with pytest.raises(ValueError, match="sorted and unique"):
            grant(components=components)


@pytest.mark.parametrize(
    "change",
    [
        ('field_validator("components")', "field_validator(load())"),
        ('field_validator("components")', "field_validator(*fields)"),
        ('field_validator("components")', 'field_validator("components", **options)'),
        ('field_validator("components")', 'field_validator("components", check_fields=flag)'),
        ('field_validator("components")', 'field_validator("components", mode=settings.mode)'),
        ('field_validator("components")', 'field_validator("components", mode="a" + suffix)'),
        ('field_validator("components")', 'field_validator("components", mode=f"{mode}")'),
        ('field_validator("components")', 'field_validator("components", unknown=True)'),
        ('field_validator("components")', 'field_validator("components", mode="unknown")'),
        ('field_validator("components")', "field_validator"),
        ('model_validator(mode="after")', "model_validator(mode=load())"),
        ('model_validator(mode="after")', 'model_validator(mode="plain")'),
        ('model_validator(mode="after")', "model_validator()"),
        ("class Grant", "field_validator = unknown\n\nclass Grant"),
        ("class Grant", "def field_validator(*args):\n    return unknown\n\nclass Grant"),
        ("class Grant", "from vendor import field_validator\n\nclass Grant"),
        (
            "class Grant",
            "from vendor import field_validator\nfrom pydantic import field_validator\n\nclass Grant",
        ),
        ("class Grant", "from vendor import *\n\nclass Grant"),
        ("class Grant", "if True:\n    from vendor import field_validator\n\nclass Grant"),
        ("    components:", "    field_validator = unknown\n    components:"),
        ("    components:", "    from vendor import field_validator\n    components:"),
        ("    components:", "    effect = load()\n    components:"),
        ("    components:", "    descriptor = unknown\n    components:"),
        ("    @classmethod", "    @unknown\n    @classmethod"),
        ("    @classmethod", "    @unknown()\n    @classmethod"),
        ("class Grant", "del field_validator\n\nclass Grant"),
    ],
)
def test_validator_uncertainty_stays_closed(tmp_path: Path, change: tuple[str, str]) -> None:
    assert _check(tmp_path, GRANT.replace(*change))


@pytest.mark.parametrize("binding", ["import pydantic as pd", "from vendor import pd"])
def test_module_validator_shadowing(tmp_path: Path, binding: str) -> None:
    source = GRANT.replace("@field_validator(", "@pd.field_validator(")
    source = source.replace("class Grant", binding + "\npd = unknown\n\nclass Grant")
    assert _check(tmp_path, source)


def test_project_local_pydantic_is_not_trusted(tmp_path: Path) -> None:
    (tmp_path / "pydantic.py").write_text('raise RuntimeError("must not import")\n')
    assert _check(tmp_path, GRANT)


def test_validator_factory_is_not_a_general_call_allowance(tmp_path: Path) -> None:
    assert _check(tmp_path, GRANT + '\nregistration = field_validator("components")\n')


@pytest.mark.parametrize(
    "hook",
    [
        "__get_pydantic_core_schema__",
        "__get_pydantic_json_schema__",
        "__pydantic_init_subclass__",
        "__pydantic_on_complete__",
    ],
)
def test_model_construction_hooks_remain_unproven(tmp_path: Path, hook: str) -> None:
    assert _check(
        tmp_path, GRANT + f'\n    def {hook}(cls, *args):\n        raise RuntimeError("effect")\n'
    )


@pytest.mark.parametrize(
    "replacement",
    [
        "ConfigDict(extra=load())",
        "ConfigDict(**settings)",
        "Field(default_factory=load)",
        "Field(default=load())",
    ],
)
def test_model_declaration_arguments_stay_closed(tmp_path: Path, replacement: str) -> None:
    source = GRANT.replace('ConfigDict(extra="forbid", frozen=True)', replacement)
    assert _check(tmp_path, source)


@pytest.mark.parametrize("annotation", ['"load()"', "CustomType", "Annotated[str, metadata]"])
def test_model_schema_annotations_stay_closed(tmp_path: Path, annotation: str) -> None:
    assert _check(
        tmp_path, GRANT.replace("components: tuple[str, ...]", f"components: {annotation}")
    )


@pytest.mark.parametrize(
    "imports",
    [
        "from vendor import classmethod\nfrom vendor import classmethod\n",
        "from vendor import *\n",
    ],
)
def test_ambiguous_imports_do_not_restore_builtin_trust(tmp_path: Path, imports: str) -> None:
    assert _check(
        tmp_path, imports + "class Example:\n    @classmethod\n    def method(cls):\n        pass\n"
    )


@pytest.mark.parametrize(
    "binding",
    [
        "__annotations__ = {payload}",
        "__annotations__: dict[str, str] = {payload}",
        "alias = __annotations__ = {payload}",
        "(__annotations__,) = ({payload},)",
        "__annotations__ = {{}}",
    ],
)
def test_explicit_model_annotations_are_unproven(tmp_path: Path, binding: str) -> None:
    marker = tmp_path / "executed"
    payload = repr({"x": f"(open({str(marker)!r}, 'w').write('effect'), int)[1]"})
    source = (
        "from pydantic import BaseModel\nclass Grant(BaseModel):\n    "
        + binding.format(payload=payload)
        + "\n"
    )
    assert _check(tmp_path, source)
    assert not marker.exists()


@pytest.mark.parametrize("hook", ["__annotate__", "__annotate_func__"])
@pytest.mark.parametrize("binding", ["method", "assignment", "chained"])
def test_explicit_annotation_hooks_are_unproven(tmp_path: Path, hook: str, binding: str) -> None:
    marker = tmp_path / "executed"
    method = (
        f"    def {hook if binding == 'method' else 'annotation_hook'}(format):\n"
        f"        open({str(marker)!r}, 'w').write('effect')\n"
        "        return {'x': int}\n"
    )
    if binding != "method":
        method += f"    {hook} = {'alias = ' if binding == 'chained' else ''}annotation_hook\n"
    assert _check(tmp_path, "from pydantic import BaseModel\nclass Grant(BaseModel):\n" + method)
    assert not marker.exists()


@pytest.mark.parametrize("name", ["BaseModel", "Model"])
@pytest.mark.parametrize("expression", ["{name}.__mro__[0]", "{name}.__base__", "{name}[int]"])
@pytest.mark.parametrize("body", ["pass", "__annotations__ = {'x': 'effect()'}"])
def test_indirect_model_bases_are_unproven(
    tmp_path: Path, name: str, expression: str, body: str
) -> None:
    source = (
        f"from pydantic import BaseModel as {name}\n"
        f"class Grant({expression.format(name=name)}):\n    {body}\n"
    )
    assert _check(tmp_path, source)


def test_direct_model_import_alias_remains_supported(tmp_path: Path) -> None:
    source = GRANT.replace("import BaseModel,", "import BaseModel as Model,").replace(
        "Grant(BaseModel)", "Grant(Model)"
    )
    assert not _check(tmp_path, source)


def test_generated_service_validator_cli(tmp_path: Path) -> None:
    import json
    import os
    import subprocess
    import sys

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PRODUCT_AUTH_TOKENS_JSON"] = json.dumps(
        {"probe-customer-token": {"tenant_id": "probe-tenant", "subject_id": "probe-user"}}
    )
    environment["CAYU_OPERATOR_BEARER_TOKEN"] = "probe-operator-token"

    def cli(*args: str, cwd: Path = tmp_path, expected: int = 0) -> str:
        result = subprocess.run(
            [sys.executable, "-m", "cayu", *args],
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        return result.stdout

    command = (
        "new",
        "validator-probe",
        "--preset",
        "service",
        "--database",
        "sqlite",
        "--provider",
        "openai-subscription",
        "--execution",
        "none",
        "--dir",
        str(tmp_path),
    )
    cli(*command, "--dry-run")
    project = tmp_path / "validator-probe"
    assert not project.exists()
    cli(*command)
    check = ("check", "--deploy", "--fail-on", "warning", "--json")
    assert json.loads(cli(*check, cwd=project))["diagnostics"] == []
    model = project / "domain/model.py"
    model.write_text(GRANT)
    # Import the contract during actual app preparation as well as scanning it.
    registration = project / "agents/registration.py"
    registration.write_text("from domain.model import Grant\n" + registration.read_text())
    report = json.loads(cli(*check, cwd=project))
    assert report["diagnostics"] == []
    assert report["manifest_fingerprint"] != "unavailable"
    # Moving the unchanged contract to its canonical policy owner also passes.
    model.unlink()
    (project / "policies/execution.py").write_text(GRANT)
    registration.write_text(registration.read_text().replace("domain.model", "policies.execution"))
    assert json.loads(cli(*check, cwd=project))["diagnostics"] == []
    marker = project / "executed"
    (project / "policies/execution.py").write_text(
        GRANT.replace(
            'field_validator("components")',
            f'field_validator(open({str(marker)!r}, "w").write("effect"))',
        )
    )
    report = json.loads(cli(*check, cwd=project, expected=1))
    assert report["manifest_fingerprint"] == "unavailable"
    assert any(item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in report["diagnostics"])
    assert not marker.exists()

    # Explicit annotation dictionaries must be rejected before Pydantic can
    # evaluate their string values during actual application preparation.
    payload = repr({"x": f"(open({str(marker)!r}, 'w').write('effect'), int)[1]"})
    (project / "policies/execution.py").write_text(
        "from pydantic import BaseModel\nclass Grant(BaseModel):\n    __annotations__ = "
        + payload
        + "\n"
    )
    report = json.loads(cli(*check, cwd=project, expected=1))
    assert report["manifest_fingerprint"] == "unavailable"
    assert any(item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in report["diagnostics"])
    assert not marker.exists()

    for hook in ("__annotate__", "__annotate_func__"):
        (project / "policies/execution.py").write_text(
            "from pydantic import BaseModel\nclass Grant(BaseModel):\n"
            f"    def {hook}(format):\n"
            f"        open({str(marker)!r}, 'w').write('effect')\n"
            "        return {'x': int}\n"
        )
        report = json.loads(cli(*check, cwd=project, expected=1))
        assert report["manifest_fingerprint"] == "unavailable"
        assert any(item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in report["diagnostics"])
        assert not marker.exists()

    # A derived expression must not inherit BaseModel's admission while evading
    # the direct-model annotation and construction-hook safeguards.
    (project / "policies/execution.py").write_text(
        "from pydantic import BaseModel\nclass Grant(BaseModel.__mro__[0]):\n"
        "    __annotations__ = " + payload + "\n"
    )
    report = json.loads(cli(*check, cwd=project, expected=1))
    assert report["manifest_fingerprint"] == "unavailable"
    assert any(item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in report["diagnostics"])
    assert not marker.exists()
