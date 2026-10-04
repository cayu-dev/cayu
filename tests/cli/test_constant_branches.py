"""Generated projects keep only their selected capability branches."""

from __future__ import annotations

import ast
import re

import pytest

from cayu.cli._constant_branches import (
    ConstantBranchError,
    has_constant_branches,
    simplify_constant_branches,
)
from cayu.cli.scaffold import project_files
from cayu.cli.scaffold_plan import CAPABILITIES, PRESETS, ScaffoldPlanError

_LITERAL_BRANCH = re.compile(
    r"\bif (not )?(True|False)\b|\b(True|False) (and|or)\b|\b(and|or) (not )?(True|False):"
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "def f(store):\n"
            "    if not True and store is not None:\n"
            '        raise ValueError("x")\n'
            "    return store\n",
            "def f(store):\n    return store\n",
        ),
        (
            "def f(tools):\n"
            "    if True:\n"
            "        # kept\n"
            "        tools.append(1)\n"
            "    else:\n"
            "        tools.append(2)\n",
            "def f(tools):\n    # kept\n    tools.append(1)\n",
        ),
        (
            "def f():\n    if not False:\n        return None\n    return compute()\n",
            "def f():\n    return None\n",
        ),
        (
            "X = (\n    Build(\n        a=1,\n    )\n    if True\n    else None\n)\nY = (\n"
            "    Build(a=1)\n    if False\n    else None\n)\n",
            "X = Build(\n    a=1,\n)\nY = None\n",
        ),
        (
            "build = True and store is None\nlate = False and store is None\n"
            "kept = store and True\n",
            "build = store is None\nlate = False\nkept = store and True\n",
        ),
        (
            "if store is None and True:\n    store = 1\n",
            "if store is None:\n    store = 1\n",
        ),
        (
            "def f():\n    if False:\n        work()\n",
            "def f():\n    pass\n",
        ),
        (
            "if a:\n    x = 1\nelif False:\n    x = 2\nelse:\n    x = 3\n",
            "if a:\n    x = 1\nelse:\n    x = 3\n",
        ),
    ],
)
def test_simplify_constant_branches(source: str, expected: str) -> None:
    assert simplify_constant_branches(source) == expected


def test_simplification_keeps_strings_and_prunes_only_newly_unused_imports() -> None:
    source = (
        "from cayu import Kept, Used, Unused as Unused\n"
        "\n"
        "\n"
        "def f():\n"
        "    if False:\n"
        "        return Used()\n"
        '    text = """first\n'
        "  indented line\n"
        '"""\n'
        "    return Kept(text)\n"
    )

    result = simplify_constant_branches(source)

    assert result.startswith("from cayu import Kept, Unused as Unused\n")
    assert '"""first\n  indented line\n"""' in result
    assert not has_constant_branches(result)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "y = x * (\n    a + b\n    if True\n    else c\n)\n",
            "y = x * (\n    a + b\n)\n",
        ),
        (
            "y = (\n    a + b if True else c\n).bit_length()\n",
            "y = (\n    a + b\n).bit_length()\n",
        ),
        (
            "y = (\n    a + b\n    if True\n    else c\n)\n",
            "y = a + b\n",
        ),
        (
            "y = f(\n    1,\n    (\n        a + b\n        if True\n        else c\n    ),\n)\n",
            "y = f(\n    1,\n    a + b,\n)\n",
        ),
        (
            "y = x * (\n    value\n    if True\n    else c\n)\n",
            "y = x * value\n",
        ),
        # A kept branch that spans lines keeps its parentheses, or the line
        # break would end the statement.
        (
            "y = (\n    a\n    + b\n    if True\n    else c\n)\n",
            "y = (\n    a\n    + b\n)\n",
        ),
        (
            "def f():\n    return (\n        a\n        + b\n        if True\n"
            "        else c\n    )\n",
            "def f():\n    return (\n        a\n        + b\n    )\n",
        ),
        (
            "x = 1, (\n    a\n    + b if True else c\n)\n",
            "x = 1, (\n    a\n    + b\n)\n",
        ),
        (
            'y = (\n    "abc"\n    "def"\n    if True\n    else c\n)\n',
            'y = (\n    "abc"\n    "def"\n)\n',
        ),
        (
            "y = (\n    a +\n    b\n    if True\n    else c\n)\n",
            "y = (\n    a +\n    b\n)\n",
        ),
        # Brackets of the kept value already enclose its line breaks.
        (
            "y = (\n    build(\n        a,\n    )\n    if True\n    else c\n)\n",
            "y = build(\n    a,\n)\n",
        ),
    ],
)
def test_grouping_parentheses_are_dropped_only_where_precedence_allows(
    source: str, expected: str
) -> None:
    result = simplify_constant_branches(source)

    assert result == expected
    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(expected))


def test_plain_read_cleanup_edits_each_nested_scope_once() -> None:
    source = (
        "def outer(cfg):\n"
        "    def inner():\n"
        "        y = cfg\n"
        "        if False:\n"
        "            return y\n"
        "        return None\n"
        "    after = 2\n"
        "    return inner, after\n"
    )

    result = simplify_constant_branches(source)

    assert result == (
        "def outer(cfg):\n"
        "    def inner():\n"
        "        return None\n"
        "    after = 2\n"
        "    return inner, after\n"
    )


def test_plain_read_cleanup_keeps_closure_reads_and_global_writes() -> None:
    source = (
        "import os\n"
        "\n"
        "V = None\n"
        "\n"
        "\n"
        "def outer(cfg):\n"
        "    global V\n"
        "    V = os.sep\n"
        "    y = cfg\n"
        "    if False:\n"
        "        return V, y\n"
        "\n"
        "    def inner():\n"
        "        return y\n"
        "\n"
        "    return inner\n"
    )

    result = simplify_constant_branches(source)

    assert "    V = os.sep\n" in result
    assert "    y = cfg\n" in result
    assert result.startswith("import os\n")


def test_multi_line_constant_if_header_keeps_only_the_body() -> None:
    source = "def f():\n    if (\n        True\n        and True\n    ):\n        body()\n"

    assert simplify_constant_branches(source) == "def f():\n    body()\n"


@pytest.mark.parametrize(
    "source",
    [
        "def f():\n    if load() and False:\n        work()\n    return 1\n",
        "def f():\n    if load() or True:\n        work()\n    return 1\n",
        "y = load() and False\n",
        "y = (\n    a\n    if load() and False\n    else b\n)\n",
        # A property or __getattr__ runs code.
        "def f(load):\n    if load.ready and False:\n        work()\n    return 1\n",
        # An unbound name raises NameError.
        "def f():\n    if load and False:\n        work()\n    return 1\n",
        # `and`, `or` and `not` call the value's __bool__ (numpy arrays raise).
        "def f(load):\n    if load and False:\n        work()\n    return 1\n",
        "def f(load):\n    if not load or True:\n        work()\n    return 1\n",
    ],
)
def test_operands_that_may_have_side_effects_are_never_dropped(source: str) -> None:
    result = simplify_constant_branches(source)

    assert "load" in result
    ast.parse(result)


def test_side_effect_free_operands_still_fold_in_conditions() -> None:
    source = (
        "def f(store):\n"
        "    if store is None and False:\n"
        "        store = 1\n"
        "    if store is not None or True:\n"
        "        return store\n"
    )

    assert simplify_constant_branches(source) == "def f(store):\n    return store\n"


@pytest.mark.parametrize(
    "source",
    [
        # A local assigned later in the function.
        "def f():\n    if load is None and False:\n        work()\n    load = 1\n    return load\n",
        # A parameter the function deletes.
        "def f(load):\n    del load\n    if load is None and False:\n        work()\n    return 1\n",
        # A conditional import.
        "try:\n    import load\nexcept ImportError:\n    pass\n\n"
        "if load is None and False:\n    work()\n",
        # A module name bound only after the use.
        "if load is None and False:\n    work()\nload = 1\n",
        # A class-scope name is not visible inside a comprehension.
        "class C:\n    load = 1\n    items = [(1 if load is None and False else 2) for _ in ()]\n",
        # A comprehension variable read after the comprehension.
        "def f():\n    items = [load for load in ()]\n    if load is None and False:\n"
        "        work()\n    return items\n",
        # Defaults, decorators and annotations are evaluated outside the function.
        "def f(load, d=(1 if load is None and False else 2)):\n    return d\n",
        "g = lambda load, d=(1 if load is None and False else 2): d\n",
        "@deco(1 if load is None and False else 2)\ndef g(load):\n    pass\n",
        # `except ... as name` deletes the name when the handler ends.
        "def f(load):\n    try:\n        pass\n    except ValueError as load:\n        pass\n"
        "    if load is None and False:\n        work()\n",
        "load = None\ntry:\n    pass\nexcept ValueError as load:\n    pass\n\n"
        "if load is None and False:\n    work()\n",
        # A nested function may delete a parameter through nonlocal.
        "def f(load):\n    def g():\n        nonlocal load\n        del load\n\n"
        "    if load is None and False:\n        work()\n    return g\n",
    ],
)
def test_names_that_may_be_unbound_are_never_folded_away(source: str) -> None:
    result = simplify_constant_branches(source)

    assert "load is None" in result
    ast.parse(result)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "def f(load):\n    if load is None and False:\n        work()\n    return 1\n",
            "def f(load):\n    return 1\n",
        ),
        (
            "load = None\n\nif load is None and False:\n    work()\n",
            "load = None\n\n",
        ),
    ],
)
def test_parameters_and_earlier_module_names_fold_in_identity_comparisons(
    source: str, expected: str
) -> None:
    assert simplify_constant_branches(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x = not(\n    a\n    if True\n    else b\n)\n", "x = not a\n"),
        ("def f():\n    return(\n        a if True else b\n    )\n", "def f():\n    return a\n"),
        (
            "async def f():\n    return await(\n        a if True else b\n    )\n",
            "async def f():\n    return await a\n",
        ),
    ],
)
def test_removed_parentheses_never_fuse_a_keyword_with_its_value(
    source: str, expected: str
) -> None:
    assert simplify_constant_branches(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "def f(cfg):\n    y = cfg\n    if False:\n        use(y)\n    y += 1\n",
        "def f(cfg):\n    y = cfg\n    if False:\n        use(y)\n\n    def g():\n"
        "        nonlocal y\n        y = 2\n\n    return g\n",
    ],
)
def test_plain_read_cleanup_keeps_assignments_other_code_depends_on(source: str) -> None:
    assert "    y = cfg\n" in simplify_constant_branches(source)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "def f(p):\n    if True and (yield p):\n        a()\n",
            "def f(p):\n    if (yield p):\n        a()\n",
        ),
        ("x = 1 if True and (y := g()) else 2\n", "x = 1 if (y := g()) else 2\n"),
        ("x = 1 if True and (lambda: 0) else 2\n", "x = 1 if (lambda: 0) else 2\n"),
        ("x = 1 if True and (p if p else 0) else 2\n", "x = 1 if (p if p else 0) else 2\n"),
    ],
)
def test_kept_condition_operands_keep_the_parentheses_they_need(source: str, expected: str) -> None:
    result = simplify_constant_branches(source)

    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(expected))


def test_plain_read_cleanup_keeps_an_emptied_body_valid() -> None:
    source = "def f(cfg):\n    store = cfg\n    if False:\n        use(store)\n"

    assert simplify_constant_branches(source) == "def f(cfg):\n    pass\n"


def test_kept_yield_stays_parenthesized() -> None:
    result = simplify_constant_branches("def g():\n    f((yield) if True else 1)\n")

    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse("def g():\n    f((yield))\n"))


def test_simplification_refuses_to_return_unparseable_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import _constant_branches

    monkeypatch.setattr(
        _constant_branches,
        "_without_newly_unused_imports",
        lambda original, source, original_locals: source + "def broken(:\n",
    )

    with pytest.raises(ConstantBranchError, match="scaffold bug"):
        simplify_constant_branches("if True:\n    x = 1\n")


def test_intermediate_parse_failures_are_reported_as_scaffold_bugs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import _constant_branches

    edits = iter([_constant_branches._Edit(0, 0, "(\n")])
    monkeypatch.setattr(
        _constant_branches, "_next_edit", lambda tree, source, **_: next(edits, None)
    )

    with pytest.raises(ConstantBranchError, match="scaffold bug"):
        simplify_constant_branches("x = 1\n")


def test_project_files_name_the_file_that_failed_to_simplify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.cli import scaffold

    monkeypatch.setattr(
        scaffold, "_rendered_project_files", lambda *args, **kwargs: {"app.py": "if True: x = 1\n"}
    )

    with pytest.raises(ConstantBranchError, match=r"^app\.py: .*scaffold bug"):
        project_files("probe")


@pytest.mark.parametrize(
    "source",
    [
        # The only yield makes the function a generator, even when unreachable.
        "def f():\n    if False:\n        yield 1\n    return None\n",
        "def f():\n    x = (yield 1) if False else 2\n    return x\n",
        # A global declaration applies to the whole function.
        "def f():\n    if False:\n        global V\n    V = 1\n",
        # A dead assignment still makes the name local: the read raises today.
        "load = 1\n\n\ndef f():\n    if False:\n        load = 2\n    return load\n",
        "load = 1\n\n\ndef f():\n    if load is None and False:\n        work()\n"
        "    if False:\n        load = 2\n    return 1\n",
        # More than one statement on a line.
        "x = 1; y = 2 if True else 3\n",
        "def f():\n    if False:\n        a(); b()\n    return 1\n",
        # A branch body on its header line.
        "if False: a()\n",
        "if True:\n    a()\nelse: b()\n",
        "if x:\n    pass\nelif False: c()\n",
    ],
)
def test_dead_branches_that_change_their_scope_are_refused(source: str) -> None:
    with pytest.raises(ConstantBranchError, match="scaffold bug"):
        simplify_constant_branches(source)


def test_dead_yield_is_removed_when_the_function_stays_a_generator() -> None:
    source = "def f():\n    if False:\n        yield 1\n    yield 2\n"

    assert simplify_constant_branches(source) == "def f():\n    yield 2\n"


def test_unreachable_cleanup_keeps_the_empty_generator_idiom() -> None:
    source = (
        "def f():\n    if True:\n        return\n    yield\n\n\n"
        "def g():\n    if True:\n        return 1\n    return 2\n"
    )

    assert simplify_constant_branches(source) == (
        "def f():\n    return\n    yield\n\n\ndef g():\n    return 1\n"
    )


def _project_variants() -> list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    variants = []
    for preset in PRESETS:
        defaults = set(preset.default_capabilities)
        selectable = [
            capability.name for capability in CAPABILITIES if capability.status == "selectable"
        ]
        for execution in preset.supported_executions:
            toggles: list[tuple[tuple[str, ...], tuple[str, ...]]] = [((), ())]
            toggles += [((), (name,)) for name in sorted(defaults)]
            toggles += [((name,), ()) for name in selectable if name not in defaults]
            toggles.append(((), tuple(sorted(defaults - {"observability"}))))
            variants += [(preset.name, execution, added, removed) for added, removed in toggles]
    return variants


def test_generated_projects_have_no_literal_capability_branches() -> None:
    generated = 0
    for preset, execution, added, removed in _project_variants():
        try:
            files = project_files(
                "probe",
                preset=preset,
                execution=execution,
                with_capabilities=added,
                without_capabilities=removed,
            )
        except (ScaffoldPlanError, ValueError):
            continue  # Not a selectable combination for this preset.
        generated += 1
        for relative, content in files.items():
            if not relative.endswith(".py"):
                continue
            ast.parse(content, filename=relative)
            assert not has_constant_branches(content), (preset, execution, relative)
            assert not _LITERAL_BRANCH.search(content), (preset, execution, relative)
            assert "_ENABLED__" not in content, (preset, execution, relative)
    assert generated >= len(PRESETS) * 3
