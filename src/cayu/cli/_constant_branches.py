"""Remove constant capability branches from rendered scaffold Python.

Scaffold templates select capabilities with placeholders that render as the
literals ``True`` or ``False``. Generated code should read as the project it
is, so this pass keeps only the selected branch of every ``if``/``else``,
conditional expression and boolean operand whose value is a literal. It edits
the rendered source text in place (never ``ast.unparse``), so comments,
strings and formatting outside the removed branches are kept byte for byte.

It only has to handle Cayu's own templates, so it supports a deliberately
small input shape and refuses everything else with ``ConstantBranchError``
rather than guessing:

- one statement per line (no ``;``), and every compound statement with a
  literal branch keeps its bodies on their own lines (no ``if False: x()``);
- a literal is folded away only past operands whose evaluation cannot fail or
  run code: other literals and ``is``/``is not`` comparisons of literals,
  undeleted parameters, and module names bound unconditionally earlier;
- a dead branch must not hold a ``global``/``nonlocal`` declaration, a
  function's only ``yield``, or the only assignment that makes a name read
  elsewhere local to its function.

Every refusal is a scaffold bug: change the template, not the generated code.
"""

from __future__ import annotations

import ast
import io
import keyword
import re
import tokenize
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import pairwise

_MAX_EDITS = 10_000
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


class ConstantBranchError(RuntimeError):
    """Simplification could not keep the rendered template's meaning.

    This is always a Cayu scaffold bug: the template, or this pass, must change.
    It is deliberately not a ``ValueError``, so callers that treat invalid
    scaffold options as ``ValueError`` never mistake it for one.
    """


@dataclass(frozen=True, slots=True)
class _Edit:
    start: int
    end: int
    text: str


def simplify_constant_branches(source: str) -> str:
    """Return *source* with literal ``True``/``False`` branches resolved.

    Code left unreachable by a resolved guard (``if not False: return None``)
    and imports used only by removed branches are removed too.
    """

    original = source
    _parse(source, "the rendered template is not valid Python")
    # Judge which names are local by the template as written: removing a dead
    # assignment must not turn an unbound local read into a global one.
    original_locals = _scope_locals(ast.parse(original))
    source = _apply(source, partial(_next_edit, original_locals=original_locals))
    if source == original:
        return source
    source = _apply(source, _unreachable_edit)
    source = _without_newly_unused_imports(original, source, original_locals)
    # A simplifier bug must never write a broken project file.
    _parse(source, "simplification produced invalid Python")
    return source


def _parse(source: str, problem: str) -> ast.Module:
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise ConstantBranchError(
            f"Constant branch simplification failed: {problem} (line {exc.lineno}: "
            f"{exc.msg}). This is a Cayu scaffold bug."
        ) from exc


def _apply(source: str, find_edit) -> str:
    for _ in range(_MAX_EDITS):
        tree = _parse(source, "simplification produced invalid Python")
        edit = find_edit(tree, source)
        if edit is None:
            return source
        _refuse_shared_lines(tree, source, (edit,))
        source = source[: edit.start] + edit.text + source[edit.end :]
    raise ConstantBranchError("Constant branch simplification did not converge.")


def _refuse_shared_lines(tree: ast.Module, source: str, edits: Sequence[_Edit]) -> None:
    """Refuse an edit on a line that holds more than one statement (``a(); b()``)."""

    shared: set[int] = set()
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for first, second in pairwise(block):
                if first.end_lineno == second.lineno:
                    shared.add(second.lineno)
    if not shared:
        return
    for edit in edits:
        first_line = source.count("\n", 0, edit.start) + 1
        last_line = source.count("\n", 0, max(edit.start, edit.end - 1)) + 1
        if any(first_line <= line <= last_line for line in shared):
            raise ConstantBranchError(
                f"Constant branch simplification failed: line {first_line} holds more than "
                "one statement, which this pass does not edit. This is a Cayu scaffold bug; "
                "put one statement per line."
            )


def _unreachable_edit(tree: ast.Module, source: str) -> _Edit | None:
    text = _Source(source)
    parents = _parents(tree)
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(node, field, None)
            if not isinstance(statements, list):
                continue
            for index, statement in enumerate(statements[:-1]):
                if isinstance(statement, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
                    dead = statements[index + 1 :]
                    scope = node if isinstance(node, _SCOPES) else _scope_of(node, parents, tree)
                    if _removal_problem(dead, scope) is not None:
                        # `return` then `yield` is the empty-generator idiom; keep it.
                        break
                    start = text.line_start(dead[0].lineno)
                    end = text.line_end(statements[-1].end_lineno)
                    return _Edit(start, end, "")
    return None


def _parents(tree: ast.AST) -> dict[ast.AST, tuple[ast.AST, str]]:
    parents: dict[ast.AST, tuple[ast.AST, str]] = {}
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if isinstance(value, list):
                for child in value:
                    if isinstance(child, ast.AST):
                        parents[child] = (parent, field)
            elif isinstance(value, ast.AST):
                parents[value] = (parent, field)
    return parents


def _scope_of(
    node: ast.AST, parents: dict[ast.AST, tuple[ast.AST, str]], tree: ast.Module
) -> ast.AST:
    """Return the scope *node* is evaluated in.

    A function's decorators, defaults and annotations, and a class's decorators
    and bases, are evaluated in the enclosing scope, not the function's or class's.
    """

    child = node
    while child in parents:
        parent, field = parents[child]
        if isinstance(parent, _SCOPES) and field == "body":
            return parent
        child = parent
    return tree


def _outer_evaluated(scope: ast.AST) -> list[ast.AST]:
    """The parts of a function or class definition evaluated where it is defined."""

    parts: list[ast.AST] = list(getattr(scope, "decorator_list", ()))
    if isinstance(scope, _FUNCTIONS):
        arguments = scope.args
        parts += arguments.defaults
        parts += [default for default in arguments.kw_defaults if default is not None]
        if not isinstance(scope, ast.Lambda):
            parts += [
                argument.annotation
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                    arguments.vararg,
                    arguments.kwarg,
                )
                if argument is not None and argument.annotation is not None
            ]
            if scope.returns is not None:
                parts.append(scope.returns)
    elif isinstance(scope, ast.ClassDef):
        parts += scope.bases
        parts += [keyword.value for keyword in scope.keywords]
    return parts


def _same_scope_nodes(roots) -> list[ast.AST]:
    """Return *roots* and their descendants evaluated in the same scope.

    A nested function or class is included, with its decorators, defaults,
    annotations and bases, but not its body.
    """

    found: list[ast.AST] = []
    pending = list(roots)
    while pending:
        node = pending.pop()
        found.append(node)
        if isinstance(node, _SCOPES):
            pending.extend(_outer_evaluated(node))
            continue
        pending.extend(ast.iter_child_nodes(node))
    return found


def _scope_nodes(scope: ast.AST) -> list[ast.AST]:
    """Return every node evaluated in *scope* itself."""

    body = scope.body
    return _same_scope_nodes(body if isinstance(body, list) else [body])


def _removal_problem(removed: Sequence[ast.AST], scope: ast.AST) -> str | None:
    """Name what removing *removed* would change beyond its own code, if anything.

    A ``global``/``nonlocal`` declaration applies to its whole scope, and a
    ``yield`` makes its function a generator, even when never reached.
    (``await`` cannot change a function's kind: only ``async def`` allows it.)
    """

    nodes = _same_scope_nodes(removed)
    if any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in nodes):
        return "a global or nonlocal declaration"
    if not isinstance(scope, _FUNCTIONS):
        # Module and class names are looked up at run time; nothing turns local.
        return None
    scope_nodes = _scope_nodes(scope)
    removed_yields = sum(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in nodes)
    if removed_yields:
        scope_yields = sum(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in scope_nodes)
        if scope_yields == removed_yields:
            return "the only yield that makes its function a generator"
    removed_bound = _bound_names(nodes)
    if removed_bound:
        removed_ids = {id(node) for root in removed for node in ast.walk(root)}
        kept = [node for node in scope_nodes if id(node) not in removed_ids]
        still_local = _bound_names(kept) | _parameters(scope)
        used = {
            node.id
            for node in ast.walk(scope)
            if isinstance(node, ast.Name)
            and not isinstance(node.ctx, ast.Store)
            and id(node) not in removed_ids
        }
        made_local = sorted((removed_bound - still_local) & used)
        if made_local:
            return f"the only assignment that makes {', '.join(made_local)} local to its function"
    return None


def _require_removable(
    removed: Sequence[ast.AST], node: ast.AST, parents, tree: ast.Module
) -> None:
    problem = _removal_problem(removed, _scope_of(node, parents, tree))
    if problem is not None:
        raise ConstantBranchError(
            f"Constant branch simplification failed: the dead branch at line {node.lineno} "
            f"contains {problem}, so removing it would change the code around it. "
            "This is a Cayu scaffold bug; keep such statements out of capability branches."
        )


def _scope_bindings(scope: ast.AST) -> set[str]:
    names = _bound_names(_scope_nodes(scope))
    if isinstance(scope, _FUNCTIONS):
        names |= _parameters(scope)
    return names


def _bound_names(nodes: Sequence[ast.AST]) -> set[str]:
    names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names


class _SafeReads:
    """Answer whether reading a bare name at *node* can neither fail nor run code.

    Only two kinds of name qualify: a parameter of an enclosing function that
    the function never deletes, and a module name bound by an unconditional
    top-level statement before the one holding the read, and never deleted.
    Any other name (a local that may still be unassigned, a comprehension or
    class name, a conditional import, a builtin that may be shadowed) does not.
    """

    def __init__(
        self,
        node: ast.AST,
        parents: dict[ast.AST, tuple[ast.AST, str]],
        tree: ast.Module,
        original_locals: Mapping[tuple[str, ...], frozenset[str]],
    ) -> None:
        self._node = node
        self._parents = parents
        self._tree = tree
        self._original_locals = original_locals
        self._answers: dict[str, bool] = {}

    def _is_local(self, scope: ast.AST, name: str) -> bool:
        key = _scope_key(scope, self._parents)
        return name in _scope_bindings(scope) or name in self._original_locals.get(key, ())

    def __call__(self, name: str) -> bool:
        if name not in self._answers:
            self._answers[name] = self._decide(name)
        return self._answers[name]

    def _decide(self, name: str) -> bool:
        in_function = False
        child = self._node
        while child in self._parents:
            current, field = self._parents[child]
            child = current
            if isinstance(current, _SCOPES) and field != "body":
                continue  # Decorators, defaults, annotations and bases run outside.
            if isinstance(current, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                if any(
                    isinstance(target, ast.Name) and target.id == name
                    for generator in current.generators
                    for target in ast.walk(generator.target)
                ):
                    return False
            elif isinstance(current, _FUNCTIONS):
                own = _scope_nodes(current)
                if any(isinstance(item, ast.Global) and name in item.names for item in own):
                    return self._module_safe(name)
                if any(isinstance(item, ast.Nonlocal) and name in item.names for item in own):
                    return False
                if name in _parameters(current):
                    # Nested functions may delete it too, through nonlocal.
                    return not _deletes(ast.walk(current), name)
                if self._is_local(current, name):
                    return False  # A local: it may not be assigned yet.
                in_function = True
            elif (
                isinstance(current, ast.ClassDef)
                and not in_function
                and self._is_local(current, name)
            ):
                return False
        return self._module_safe(name)

    def _module_safe(self, name: str) -> bool:
        top = self._node
        while top in self._parents and self._parents[top][0] is not self._tree:
            top = self._parents[top][0]
        if _deletes(ast.walk(self._tree), name):
            return False
        return any(
            statement.lineno < top.lineno and name in _unconditional_bindings(statement)
            for statement in self._tree.body
        )


def _scope_key(scope: ast.AST, parents: dict[ast.AST, tuple[ast.AST, str]]) -> tuple[str, ...]:
    """Name a function or class scope by its nesting, stable across edits."""

    names: list[str] = []
    current: ast.AST | None = scope
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(current.name)
        elif isinstance(current, ast.Lambda):
            names.append("<lambda>")
        current = parents[current][0] if current in parents else None
    return tuple(reversed(names))


def _scope_locals(tree: ast.Module) -> dict[tuple[str, ...], frozenset[str]]:
    """Map every function and class scope to the names it binds."""

    parents = _parents(tree)
    found: dict[tuple[str, ...], set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, _SCOPES):
            found.setdefault(_scope_key(node, parents), set()).update(_scope_bindings(node))
    return {key: frozenset(names) for key, names in found.items()}


def _parameters(function: ast.AST) -> frozenset[str]:
    arguments = function.args
    return frozenset(
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            arguments.vararg,
            arguments.kwarg,
        )
        if argument is not None
    )


def _deletes(nodes, name: str) -> bool:
    """Whether any of *nodes* may unbind *name*: ``del``, or ``except ... as name``."""

    return any(
        (isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Del))
        or (isinstance(node, ast.ExceptHandler) and node.name == name)
        for node in nodes
    )


def _unconditional_bindings(statement: ast.stmt) -> frozenset[str]:
    """Names a top-level statement always binds once it completes."""

    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return frozenset({statement.name})
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return frozenset((alias.asname or alias.name).split(".")[0] for alias in statement.names)
    targets: list[ast.expr] = []
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        targets = [statement.target]
    return frozenset(
        item.id
        for target in targets
        for item in ast.walk(target)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
    )


def _loaded_names(tree: ast.AST) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Store):
            counts[node.id] = counts.get(node.id, 0) + 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # __all__ entries and string annotations still reference a name.
            counts[node.value] = counts.get(node.value, 0) + 1
    return counts


def _functions(tree: ast.AST, prefix: str = "") -> list[tuple[str, ast.AST]]:
    found: list[tuple[str, ast.AST]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = f"{prefix}{node.name}"
            if not isinstance(node, ast.ClassDef):
                found.append((name, node))
            found.extend(_functions(node, f"{name}."))
    return found


def _function_loads(tree: ast.AST) -> dict[str, dict[str, int]]:
    # Loads include nested scopes, which may read this scope's locals as closures.
    return {name: _loaded_names(function) for name, function in _functions(tree)}


def _own_scope_statements(function: ast.AST) -> list[ast.AST]:
    """Return the nodes of *function*'s own scope, not of nested functions or classes."""

    return _scope_nodes(function)


def _declared_outer_names(function: ast.AST) -> frozenset[str]:
    return frozenset(
        name
        for node in _own_scope_statements(function)
        if isinstance(node, (ast.Global, ast.Nonlocal))
        for name in node.names
    )


def _is_plain_read(node: ast.expr, is_safe: Callable[[str], bool]) -> bool:
    """A literal, or a name whose read cannot fail (see ``_SafeReads``).

    Attribute reads are excluded, because a property or ``__getattr__`` runs code.
    """

    if isinstance(node, ast.Constant):
        return True
    return isinstance(node, ast.Name) and is_safe(node.id)


def _without_newly_unused_imports(
    original: str, source: str, original_locals: Mapping[tuple[str, ...], frozenset[str]]
) -> str:
    """Drop imports, and plain local reads, that only removed branches used."""

    before = _loaded_names(ast.parse(original))
    tree = ast.parse(source)
    after = _loaded_names(tree)
    text = _Source(source)
    edits: list[_Edit] = []
    for statement in tree.body:
        if not isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        kept = []
        for alias in statement.names:
            bound = (alias.asname or alias.name).split(".")[0]
            if before.get(bound, 0) > 0 and after.get(bound, 0) == 0 and alias.asname is None:
                continue
            kept.append(alias)
        if len(kept) == len(statement.names):
            continue
        start = text.line_start(statement.lineno)
        end = text.line_end(statement.end_lineno)
        if not kept:
            edits.append(_Edit(start, end, ""))
            continue
        names = [
            alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
            for alias in kept
        ]
        indent = text.indent(statement.lineno)
        if isinstance(statement, ast.ImportFrom):
            module = "." * statement.level + (statement.module or "")
            head = f"{indent}from {module} import "
        else:
            head = f"{indent}import "
        one_line = head + ", ".join(names) + "\n"
        if len(one_line) <= 89:
            replacement = one_line
        elif isinstance(statement, ast.ImportFrom):
            replacement = head + "(\n" + "".join(f"{indent}    {name},\n" for name in names)
            replacement += f"{indent})\n"
        else:
            replacement = one_line
        edits.append(_Edit(start, end, replacement))
    before_functions = _function_loads(ast.parse(original))
    parents = _parents(tree)
    dead_reads: dict[ast.stmt, None] = {}
    for qualified_name, function in _functions(tree):
        loads_before = before_functions.get(qualified_name, {})
        loads_after = _loaded_names(function)
        outer_names = _declared_outer_names(function)
        # Keep an assignment that a later `y += 1`, or a nested `nonlocal y` or
        # `global y`, still depends on; keeping it is always safe.
        for node in ast.walk(function):
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                outer_names |= {node.target.id}
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                outer_names |= set(node.names)
        for statement in _own_scope_statements(function):
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id not in outer_names
                and _is_plain_read(
                    statement.value, _SafeReads(statement, parents, tree, original_locals)
                )
                and loads_before.get(statement.targets[0].id, 0) > 0
                and loads_after.get(statement.targets[0].id, 0) == 0
            ):
                dead_reads[statement] = None
    for statement in dead_reads:
        parent, field = parents[statement]
        block = getattr(parent, field)
        # A block whose every statement goes still needs one to stay valid.
        emptied = all(sibling in dead_reads for sibling in block) and statement is block[0]
        edits.append(
            _Edit(
                text.line_start(statement.lineno),
                text.line_end(statement.end_lineno),
                f"{text.indent(statement.lineno)}pass\n" if emptied else "",
            )
        )
    unique = {(edit.start, edit.end): edit for edit in edits}
    _refuse_shared_lines(tree, source, tuple(unique.values()))
    for edit in sorted(unique.values(), key=lambda item: item.start, reverse=True):
        source = source[: edit.start] + edit.text + source[edit.end :]
    return _collapse_blank_lines(source)


def _collapse_blank_lines(source: str) -> str:
    """Keep at most two consecutive blank lines where removals left more."""

    strings = _string_continuation_lines(source)
    result: list[str] = []
    blank_run = 0
    for lineno, line in enumerate(source.splitlines(keepends=True), start=1):
        if not line.strip() and lineno not in strings:
            blank_run += 1
            if blank_run > 2:
                continue
        else:
            blank_run = 0
        result.append(line)
    return "".join(result)


def has_constant_branches(source: str) -> bool:
    """Report whether *source* still branches on a literal ``True``/``False``."""

    return _next_edit(ast.parse(source), source) is not None


def _truth(node: ast.expr) -> bool | None:
    """Return the literal value of *node*, evaluated left to right like Python.

    A boolean operation is decided only by literal operands reached before any
    other operand, so an operand that could have side effects is never dropped.
    """

    return _resolved_truth(node, is_safe=None)


def _condition_truth(node: ast.expr, is_safe: Callable[[str], bool]) -> bool | None:
    """Return the truthiness of a condition, if literals decide it.

    In a condition only truthiness matters, so an operand whose truth test runs
    no code does not prevent folding: ``x is None and False`` is falsy whatever
    ``x`` is. Any other operand stops folding, because skipping it would skip
    its evaluation, including the ``__bool__``/``__len__`` call that ``and``,
    ``or`` and ``not`` make on it.
    """

    return _resolved_truth(node, is_safe=is_safe)


def _has_free_truthiness(node: ast.expr, is_safe: Callable[[str], bool]) -> bool:
    """Whether testing *node*'s truth can neither fail nor run code.

    Only literals and ``is``/``is not`` comparisons of safe reads qualify: an
    identity comparison never calls user code and yields a ``bool``. A bare
    name does not, because its truth test calls the value's ``__bool__``.
    """

    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _has_free_truthiness(node.operand, is_safe)
    if isinstance(node, ast.Compare):
        return all(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops) and all(
            _is_plain_read(operand, is_safe) for operand in (node.left, *node.comparators)
        )
    return False


def _resolved_truth(node: ast.expr, *, is_safe: Callable[[str], bool] | None) -> bool | None:
    if isinstance(node, ast.Constant) and type(node.value) is bool:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        operand = _resolved_truth(node.operand, is_safe=is_safe)
        return None if operand is None else not operand
    if isinstance(node, ast.BoolOp):
        deciding = not isinstance(node.op, ast.And)
        undecided = False
        for value in node.values:
            truth = _resolved_truth(value, is_safe=is_safe)
            if truth is None:
                if is_safe is not None and _has_free_truthiness(value, is_safe):
                    undecided = True
                    continue
                return None
            if truth is deciding:
                return deciding
        return None if undecided else not deciding
    return None


class _Source:
    def __init__(self, source: str) -> None:
        self.source = source
        self.lines = source.splitlines(keepends=True)
        self.line_starts: list[int] = []
        position = 0
        for line in self.lines:
            self.line_starts.append(position)
            position += len(line)
        self.string_continuations = _string_continuation_lines(source)

    def index(self, lineno: int, col_offset: int) -> int:
        line = self.lines[lineno - 1]
        # ast column offsets are UTF-8 byte offsets.
        prefix = line.encode("utf-8")[:col_offset].decode("utf-8")
        return self.line_starts[lineno - 1] + len(prefix)

    def span(self, node: ast.AST) -> tuple[int, int]:
        return (
            self.index(node.lineno, node.col_offset),
            self.index(node.end_lineno, node.end_col_offset),
        )

    def segment(self, node: ast.AST) -> str:
        start, end = self.span(node)
        return self.source[start:end]

    def line_start(self, lineno: int) -> int:
        return self.line_starts[lineno - 1]

    def line_end(self, lineno: int) -> int:
        return self.line_starts[lineno - 1] + len(self.lines[lineno - 1])

    def header_end_line(self, test: ast.expr) -> int:
        """Return the line holding the ``:`` that ends a compound statement header."""

        index = self.index(test.end_lineno, test.end_col_offset)
        while index < len(self.source):
            character = self.source[index]
            if character == ":":
                return self.source.count("\n", 0, index) + 1
            if character == "#":
                newline = self.source.find("\n", index)
                index = len(self.source) if newline < 0 else newline
                continue
            index += 1
        raise ValueError("compound statement header has no ':'")

    def indent(self, lineno: int) -> str:
        line = self.lines[lineno - 1]
        return line[: len(line) - len(line.lstrip(" \t"))]

    def dedented(self, first: int, last: int, amount: int) -> str:
        """Return whole lines *first*..*last*, shifted left by *amount* columns."""

        result = []
        for lineno in range(first, last + 1):
            line = self.lines[lineno - 1]
            if lineno in self.string_continuations or not line.strip():
                result.append(line if line.strip() else line.lstrip(" \t"))
            elif line[:amount].strip() == "":
                result.append(line[amount:])
            else:
                result.append(line.lstrip(" \t"))
        return "".join(result)


def _string_continuation_lines(source: str) -> frozenset[int]:
    lines: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.STRING and token.end[0] > token.start[0]:
            lines.update(range(token.start[0] + 1, token.end[0] + 1))
    return frozenset(lines)


def _next_edit(
    tree: ast.Module,
    source: str,
    original_locals: Mapping[tuple[str, ...], frozenset[str]] | None = None,
) -> _Edit | None:
    text = _Source(source)
    parents = _parents(tree)
    if original_locals is None:
        original_locals = _scope_locals(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            truth = _condition_truth(node.test, _SafeReads(node, parents, tree, original_locals))
            if truth is not None:
                _require_removable(
                    [node.test, *(node.orelse if truth else node.body)], node, parents, tree
                )
                return _if_edit(node, truth, text, parents)
            edit = _condition_operand_edit(node.test, text)
            if edit is not None:
                return edit
        elif isinstance(node, ast.IfExp):
            truth = _condition_truth(node.test, _SafeReads(node, parents, tree, original_locals))
            if truth is not None:
                _require_removable(
                    [node.test, node.orelse if truth else node.body], node, parents, tree
                )
                return _if_expression_edit(node, truth, text, parents)
            edit = _condition_operand_edit(node.test, text)
            if edit is not None:
                return edit
        elif isinstance(node, ast.BoolOp):
            edit = _bool_operand_edit(node, text, parents, tree)
            if edit is not None:
                return edit
    return None


def _if_edit(
    node: ast.If,
    truth: bool,
    text: _Source,
    parents: dict[ast.AST, tuple[ast.AST, str]],
) -> _Edit:
    header_end = text.header_end_line(node.test)
    # `else: x()` puts the first else statement on the `else` line itself.
    alternative_on_header = bool(node.orelse) and bool(
        re.match(r"\s*else\s*:", text.lines[node.orelse[0].lineno - 1])
    )
    if node.body[0].lineno == header_end or alternative_on_header:
        raise ConstantBranchError(
            f"Constant branch simplification failed: the constant if statement at line "
            f"{node.lineno} keeps a body on its header line. This is a Cayu scaffold bug; "
            "put every branch body on its own lines."
        )
    indent = text.indent(node.lineno)
    start = text.line_start(node.lineno)
    end = text.line_end(node.end_lineno)
    is_elif = text.lines[node.lineno - 1].lstrip().startswith("elif")
    if truth:
        first = header_end + 1
        last = node.body[-1].end_lineno
        amount = len(text.indent(node.body[0].lineno)) - len(indent)
        replacement = text.dedented(first, last, amount)
        if is_elif:
            replacement = f"{indent}else:\n" + text.dedented(first, last, 0)
        return _Edit(start, end, replacement)
    if not node.orelse:
        if is_elif:
            return _Edit(start, end, "")
        parent, field = parents[node]
        siblings = getattr(parent, field)
        replacement = f"{indent}pass\n" if len(siblings) == 1 else ""
        return _Edit(start, end, replacement)
    alternative = node.orelse[0]
    alternative_line = text.lines[alternative.lineno - 1].lstrip()
    if isinstance(alternative, ast.If) and alternative_line.startswith("elif"):
        # `if False: ... elif x:` becomes `if x:` (or `elif x:` inside an elif chain).
        keyword_start = text.line_start(alternative.lineno) + len(indent)
        remainder = text.source[keyword_start + len("elif") : end]
        prefix = "elif" if is_elif else "if"
        return _Edit(start, end, f"{indent}{prefix}{remainder}")
    else_line = next(
        lineno
        for lineno in range(node.body[-1].end_lineno + 1, alternative.lineno + 1)
        if text.lines[lineno - 1].startswith(f"{indent}else")
    )
    amount = len(text.indent(alternative.lineno)) - len(indent)
    if is_elif:
        return _Edit(
            start,
            end,
            f"{indent}else:\n" + text.dedented(else_line + 1, node.orelse[-1].end_lineno, 0),
        )
    return _Edit(
        start,
        end,
        text.dedented(else_line + 1, node.orelse[-1].end_lineno, amount),
    )


def _needs_parentheses(node: ast.expr) -> bool:
    return isinstance(
        node,
        (ast.IfExp, ast.Lambda, ast.NamedExpr, ast.BoolOp, ast.Yield, ast.YieldFrom, ast.Await),
    )


def _is_atom(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        # `1 .bit_length()` needs its space or parentheses; keep numbers grouped.
        return not isinstance(node.value, (int, float, complex)) or isinstance(node.value, bool)
    return isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Subscript))


def _is_whole_value(node: ast.expr, parents: dict[ast.AST, tuple[ast.AST, str]]) -> bool:
    """Whether *node* is a complete value that no operator binds to."""

    parent, field = parents[node]
    if isinstance(parent, (ast.stmt, ast.keyword)):
        return True
    if isinstance(parent, ast.Call):
        return field == "args"
    if isinstance(parent, (ast.List, ast.Tuple, ast.Set)):
        return field == "elts"
    return isinstance(parent, ast.Dict)


def _if_expression_edit(
    node: ast.IfExp,
    truth: bool,
    text: _Source,
    parents: dict[ast.AST, tuple[ast.AST, str]],
) -> _Edit:
    chosen = node.body if truth else node.orelse
    start, end = text.span(node)
    replacement = text.segment(chosen)
    if _needs_parentheses(chosen):
        replacement = f"({replacement})"
    grouping = _grouping_parentheses(text, start, end)
    if (
        grouping is not None
        and not _needs_parentheses(chosen)
        and (_is_atom(chosen) or _is_whole_value(node, parents))
        and _self_delimited(replacement)
    ):
        open_index, close_index = grouping
        open_line = text.source.count("\n", 0, open_index) + 1
        shift = node.col_offset - len(text.indent(open_line).encode("utf-8"))
        lines = replacement.split("\n")
        replacement = "\n".join(
            [lines[0]]
            + [line[shift:] if line[:shift].strip() == "" else line for line in lines[1:]]
        )
        preceding = text.source[open_index - 1 : open_index]
        if preceding.isalnum() or preceding == "_":
            # `return(` or `not(`: without the parentheses the keyword needs a space.
            replacement = f" {replacement}"
        return _Edit(open_index, close_index + 1, replacement)
    return _Edit(start, end, replacement)


def _self_delimited(segment: str) -> bool:
    """Whether every line break in *segment* sits inside its own brackets or strings.

    Without the enclosing parentheses, any other line break would end the statement.
    """

    if "\n" not in segment:
        return True
    depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(f"({segment})").readline):
            if token.type == tokenize.OP and token.string in {"(", "[", "{"}:
                depth += 1
            elif token.type == tokenize.OP and token.string in {")", "]", "}"}:
                depth -= 1
            elif token.type in {tokenize.NL, tokenize.NEWLINE} and depth == 1:
                return False
    except (tokenize.TokenError, SyntaxError):
        return False
    return True


def _grouping_parentheses(text: _Source, start: int, end: int) -> tuple[int, int] | None:
    """Find parentheses that only group this expression across lines."""

    before = text.source[:start].rstrip(" \t\n")
    after_index = end + (len(text.source[end:]) - len(text.source[end:].lstrip(" \t\n")))
    if not before.endswith("(") or text.source[after_index : after_index + 1] != ")":
        return None
    if "\n" not in text.source[len(before) : start] and "\n" not in text.source[end:after_index]:
        return None
    preceding = before[:-1].rstrip(" \t")
    if preceding and (preceding[-1].isalnum() or preceding[-1] in "_)]}\"'"):
        word = preceding.split()[-1] if preceding.split() else ""
        if word not in keyword.kwlist:
            return None
    return len(before) - 1, after_index


def _condition_operand_edit(test: ast.expr, text: _Source) -> _Edit | None:
    """In a condition only truthiness matters, so drop neutral literals anywhere."""

    if not isinstance(test, ast.BoolOp):
        return None
    neutral = isinstance(test.op, ast.And)
    remaining = [value for value in test.values if _truth(value) is not neutral]
    if len(remaining) == len(test.values) or not remaining:
        return None
    operator = " and " if neutral else " or "

    def rendered(value: ast.expr) -> str:
        # A lone `a or b` or `await x` is a valid condition as it is; anything
        # else that binds looser than `and`/`or` keeps its parentheses.
        if len(remaining) == 1 and isinstance(value, (ast.BoolOp, ast.Await)):
            return text.segment(value)
        if _needs_parentheses(value):
            return f"({text.segment(value)})"
        return text.segment(value)

    replacement = operator.join(rendered(value) for value in remaining)
    start, end = text.span(test)
    return _Edit(start, end, replacement)


def _bool_operand_edit(
    node: ast.BoolOp,
    text: _Source,
    parents: dict[ast.AST, tuple[ast.AST, str]],
    tree: ast.Module,
) -> _Edit | None:
    """Resolve literal operands that are evaluated before anything else.

    Only leading operands are folded, so evaluation order and the value of a
    non-boolean operand (``x and True`` is ``x`` when ``x`` is falsy) are kept.
    """

    neutral = isinstance(node.op, ast.And)
    remaining = list(node.values)
    while remaining and _truth(remaining[0]) is not None:
        if _truth(remaining[0]) is not neutral:
            _require_removable(remaining[1:], node, parents, tree)
            start, end = text.span(node)
            return _Edit(start, end, repr(not neutral))
        remaining.pop(0)
    if len(remaining) == len(node.values):
        return None
    if not remaining:
        start, end = text.span(node)
        return _Edit(start, end, repr(neutral))
    operator = " and " if neutral else " or "
    rendered = [
        f"({text.segment(value)})" if _needs_parentheses(value) else text.segment(value)
        for value in remaining
    ]
    replacement = operator.join(rendered)
    start, end = text.span(node)
    return _Edit(start, end, replacement)
