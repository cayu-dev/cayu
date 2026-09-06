"""ToolSpec metadata is statically proven without running application code."""

import ast
from pathlib import Path

import pytest

from cayu.cli.scaffold_check import _check_import_inertness


def check(root, source):
    path = root / "tools" / "example.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(source)
    return _check_import_inertness(root)


@pytest.mark.parametrize(
    "effect", ['"none"', "ToolEffect.NONE", "ToolEffect.IDEMPOTENT", "ToolEffect.EXTERNAL"]
)
@pytest.mark.parametrize("placement", ["module", "class"])
def test_literal_and_enum_effects(tmp_path, effect, placement):
    declaration = f'spec = ToolSpec(name="example", effect={effect})\n'
    if placement == "class":
        declaration = "class Example(Tool):\n    " + declaration
    assert check(tmp_path, "from cayu import Tool, ToolSpec, ToolEffect\n" + declaration) == ()


@pytest.mark.parametrize("value", ["-1", "+1", "-0.5", "+0.5", "64 * 1024", "64 * 1024 * 1024"])
@pytest.mark.parametrize("placement", ["module", "class"])
def test_numeric_metadata(tmp_path, value, placement):
    declaration = (
        'spec = ToolSpec(name="example", '
        f'input_schema={{"type": "number", "minimum": {value}}}, '
        "max_terminal_payload_bytes=64 * 1024)\n"
    )
    if placement == "class":
        declaration = "class Example(Tool):\n    " + declaration
    assert check(tmp_path, "from cayu import Tool, ToolSpec\n" + declaration) == ()


@pytest.mark.parametrize(
    "value",
    [
        "-opaque",
        "+opaque",
        "opaque * 1024",
        "'x' * 1024",
        "2 ** 64",
        "2147483649 * 1",
        "2147483648 * 2147483648 * 2",
    ],
)
@pytest.mark.parametrize("placement", ["module", "class"])
def test_unsafe_numeric_metadata_fails_closed(tmp_path, value, placement):
    declaration = f'spec = ToolSpec(name="example", input_schema={{"minimum": {value}}})\n'
    if placement == "class":
        declaration = "class Example(Tool):\n    " + declaration
    diagnostics = check(tmp_path, "from cayu import Tool, ToolSpec\n" + declaration)
    assert any(diagnostic.code == "SCAFFOLD_IMPORT_SIDE_EFFECT" for diagnostic in diagnostics)


@pytest.mark.parametrize(
    "imports, spec, effect",
    [
        ("from cayu import ToolSpec as Spec, ToolEffect as Effect", "Spec", "Effect.NONE"),
        ("import cayu as runtime", "runtime.ToolSpec", "runtime.ToolEffect.NONE"),
    ],
)
def test_trusted_aliases(tmp_path, imports, spec, effect):
    assert (
        check(
            tmp_path,
            f'{imports}\nclass Example:\n    spec = {spec}(name="example", effect={effect})\n',
        )
        == ()
    )


def test_realistic_description_and_schema(tmp_path):
    assert (
        check(
            tmp_path,
            """from cayu import Tool, ToolSpec, ToolEffect
_GUIDANCE = "Use the latest snapshot."
_OPERATIONS = ("open", "snapshot", "click")
class Browser(Tool):
    spec = ToolSpec(
        name="browser", effect=ToolEffect.EXTERNAL,
        parallel_safe=False, workspace_mutation=True,
        description="Browse. " + _GUIDANCE + " Verify the result.",
        input_schema={"type": "object", "properties": {
            "operation": {"type": "string", "enum": list(_OPERATIONS)},
            "key": {"description": _GUIDANCE},
        }, "required": ["operation"], "additionalProperties": False},
    )
""",
        )
        == ()
    )


@pytest.mark.parametrize(
    "prefix, metadata",
    [
        ("", "ToolEffect.UNKNOWN"),
        ("", "ToolEffect.NONE.value"),
        ("", "ToolEffect.__class__"),
        ("ToolEffect = object()", "ToolEffect.NONE"),
        ("from other import ToolEffect", "ToolEffect.NONE"),
        (
            "class Custom:\n    def __getattr__(self, name):\n        raise RuntimeError('executed')\nvalue = Custom()",
            "value.effect",
        ),
        ("OPS = ('open',)\nOPS = object()", "{'enum': list(OPS)}"),
        ("OPS = ['open']", "{'enum': list(OPS)}"),
        ("OPS = ([],)", "{'enum': list(OPS)}"),
        ("OPS = ('open',)\nlist = tuple", "{'enum': list(OPS)}"),
        ("TEXT = object()", "'prefix' + TEXT"),
        ("", "dict(type='object')"),
        ("", "{'value': object()}"),
        ("", "ToolEffect('none')"),
        ("", "unknown"),
    ],
)
@pytest.mark.parametrize("placement", ["module", "class"])
def test_unknown_metadata_fails_closed(tmp_path, prefix, metadata, placement):
    declaration = f'spec = ToolSpec(name="example", input_schema={metadata})\n'
    if placement == "class":
        declaration = "class Example(Tool):\n    " + declaration
    assert check(
        tmp_path, "from cayu import Tool, ToolSpec, ToolEffect\n" + prefix + "\n" + declaration
    )


@pytest.mark.parametrize(
    "shadow", ["ToolEffect = object", "ToolSpec = object", "list = tuple", "OPS = object"]
)
def test_class_shadowing(tmp_path, shadow):
    assert check(
        tmp_path,
        f"""from cayu import Tool, ToolSpec, ToolEffect
OPS = ("open",)
class Example(Tool):
    {shadow}
    spec = ToolSpec(name="example", effect=ToolEffect.NONE, input_schema={{"enum": list(OPS)}})
""",
    )


def test_project_shadowed_runtime(tmp_path):
    (tmp_path / "cayu.py").write_text("class ToolSpec: pass\nclass ToolEffect: NONE = 'none'\n")
    assert check(
        tmp_path,
        'from cayu import ToolSpec, ToolEffect\nspec = ToolSpec(name="example", effect=ToolEffect.NONE)\n',
    )


def test_shipped_domain_tool_declaration(tmp_path):
    guide = Path(__file__).parents[2] / "src/cayu/guides/references.md"
    section = guide.read_text().split("## domain-tool\n", 1)[1]
    source = section.split("```python\n", 1)[1].split("```", 1)[0]
    tree = ast.parse(source)
    # The guide follows the declaration with an explicit invocation demo; only
    # imports and the actual, unmodified tool class belong in an app module.
    tree.body = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
    ]
    assert check(tmp_path, ast.unparse(tree)) == ()
