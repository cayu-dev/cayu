"""Prove that a module-level call to an application function is inert.

``cayu check`` refuses to import an application until every statement that
runs at import is proven side-effect free. A module-level call such as
``provider_options=scout_provider_options()`` runs the called function's body
at import, so this module reads that body, without executing it, and accepts
the call only when everything the body can do is proven inert.

The proof decides by what the function does, never by its name. It follows a
call only to a function the application defines itself: a single undecorated
module-level ``def`` in project source, reached directly or through explicit
``from module import name`` re-exports. Functions from installed packages are
never followed. Recursion is refused, and the call chain is bounded.

Values are tracked as either *data* or *opaque*. Data is built from literals,
environment reads and the operations below on other data, so it is always a
builtin ``str``, number, ``bool``, ``None`` or a builtin collection of data.
Operations on data run no application code. A module-level constant is data
when its own statement built it from data and nothing can have changed it
since: a value immutable all the way down always, and a list, dict or set only
when the caller says the scanned source changes no value in place at import. Anything else (an imported object,
a class, a value the proof cannot classify) is opaque: it may be stored,
returned, passed to another proven helper or compared with ``is``, but any
operation that could run its code (truthiness, comparison, formatting,
iteration, hashing, attribute access, calls) is refused.

Inside a proven helper body:

- statements: docstrings and other expressions, ``pass``, assignments and
  annotated assignments to local names, ``if``/``else``, ``return``,
  ``assert`` and ``raise`` of a builtin exception built from data. Augmented
  assignment is refused: ``items += [...]`` extends a shared list in place;
- expressions: literals, f-strings of data with literal format specs,
  ``and``/``or``/``not``, comparisons, ``+ - | & ^`` and unary operators,
  conditional expressions, subscripts and slices, collection displays and
  comprehensions over data;
- calls: other proven helpers, the side-effect-free builtins in
  ``_DATA_BUILTINS``, the read-only ``str``/``dict``/``tuple``/``set``
  methods in ``_DATA_METHODS``, settings read with ``os.environ.get(...)``,
  ``os.environ[...]`` or ``os.getenv(...)``, and the calls the import check
  already reviews (for example ``AgentSpec(...)``).

Settings reads are accepted here, unlike at module level, because a helper is
the call-time form the guide recommends: the generated ``configured_*``
settings functions read the environment the same way and are called at import
by the generated agent modules.

Repetition and exponent operators, ``range`` and padding methods are left out
so that a proven body cannot build an unbounded value from a small input.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# How many nested application helper calls a proof follows from the call at
# module level. Deeper chains are refused rather than assumed inert.
MAX_HELPER_DEPTH = 4
# Expression and statement nodes one proof may evaluate, across every helper
# body and module constant it reads.
MAX_PROOF_STEPS = 20_000
# How many ``from module import name`` re-exports a name may pass through.
_MAX_REEXPORTS = 8

_DATA = "data"
_OPAQUE = "opaque"
_Kind = Literal["data", "opaque"]

# Builtins that only compute a value from data arguments.
_DATA_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "frozenset",
        "int",
        "len",
        "list",
        "max",
        "min",
        "repr",
        "reversed",
        "set",
        "sorted",
        "str",
        "tuple",
        "zip",
    }
)
# Methods of builtin data types that neither mutate their receiver nor build a
# value larger than their inputs. No builtin data type has a mutating method
# with any of these names.
_DATA_METHODS = frozenset(
    {
        # str
        "capitalize",
        "casefold",
        "endswith",
        "find",
        "isalnum",
        "isalpha",
        "isascii",
        "isdecimal",
        "isdigit",
        "isidentifier",
        "islower",
        "isnumeric",
        "isprintable",
        "isspace",
        "istitle",
        "isupper",
        "join",
        "lower",
        "lstrip",
        "partition",
        "removeprefix",
        "removesuffix",
        "replace",
        "rfind",
        "rindex",
        "rpartition",
        "rsplit",
        "rstrip",
        "split",
        "splitlines",
        "startswith",
        "strip",
        "swapcase",
        "title",
        "upper",
        # str, tuple and list
        "count",
        "index",
        # dict
        "get",
        "items",
        "keys",
        "values",
        # dict, list and set
        "copy",
        # set and frozenset
        "difference",
        "intersection",
        "isdisjoint",
        "issubset",
        "issuperset",
        "symmetric_difference",
        "union",
    }
)
# Builtins and methods that iterate a generator argument before they return,
# so a generator expression passed straight to one is evaluated where it stands.
_CONSUMING_BUILTINS = frozenset(
    {"all", "any", "dict", "frozenset", "list", "max", "min", "set", "sorted", "tuple"}
)
_CONSUMING_METHODS = frozenset({"join"})
# Attributes the import system sets on every module: strings or None.
_MODULE_ATTRIBUTES = frozenset({"__doc__", "__file__", "__name__", "__package__"})
# Builtins whose result is a str, number or bool.
_SCALAR_BUILTINS = frozenset({"abs", "bool", "float", "int", "len", "repr", "str"})
_DATA_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.BitOr, ast.BitAnd, ast.BitXor)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
# ``os`` functions and attributes that read a setting.
_ENVIRONMENT_FUNCTIONS = frozenset({"getenv"})


def _builtin_exception(name: str) -> bool:
    value = getattr(builtins, name, None)
    # SystemExit, KeyboardInterrupt and GeneratorExit would stop the importing
    # process instead of failing the import.
    return isinstance(value, type) and issubclass(value, Exception)


ObstacleReason = Literal["expression", "recursion", "depth_limit", "proof_limit"]


@dataclass(frozen=True)
class HelperObstacle:
    """Why a module-level call to an application function was not proven inert."""

    helper: str
    helper_path: Path
    helper_line: int
    # The construct that could not be proven, and where it is.
    node: ast.AST
    path: Path
    reason: ObstacleReason


@dataclass(frozen=True)
class HelperCallVerdict:
    """``helper`` is None when the callee is not an application function."""

    helper: str | None
    obstacle: HelperObstacle | None = None

    @property
    def proven(self) -> bool:
        return self.helper is not None and self.obstacle is None


# Resolves ``(importing_path, module, level)`` to ``(is_project_local, path)``.
ImportResolver = Callable[[Path, str | None, int], tuple[bool, Path | None]]
# Whether a call is one of the forms the import check already reviews, read in
# the namespace of the module at the given path.
ReviewedCall = Callable[[Path, ast.Call], bool]


class _Unproven(Exception):
    def __init__(self, node: ast.AST, path: Path, reason: ObstacleReason = "expression") -> None:
        super().__init__(reason)
        self.node = node
        self.path = path
        self.reason = reason


class _ProofLimit(Exception):
    pass


@dataclass
class _Module:
    path: Path
    binding_counts: dict[str, int]
    star_import: bool
    functions: dict[str, ast.FunctionDef]
    constants: dict[str, ast.expr | None]
    imports: dict[str, ast.ImportFrom | ast.Import]


@dataclass(frozen=True)
class _Function:
    module: _Module
    definition: ast.FunctionDef


@dataclass(frozen=True)
class _Constant:
    module: _Module
    name: str


@dataclass(frozen=True)
class _Builtin:
    name: str


@dataclass(frozen=True)
class _OsModule:
    pass


@dataclass(frozen=True)
class _OsSymbol:
    name: str


@dataclass(frozen=True)
class _Opaque:
    pass


_Resolution = _Function | _Constant | _Builtin | _OsModule | _OsSymbol | _Opaque


@dataclass
class _Frame:
    module: _Module
    # Names the function binds anywhere in its body; Python treats each as local.
    local_names: frozenset[str]
    env: dict[str, _Kind] = field(default_factory=dict)
    returns: list[_Kind] = field(default_factory=list)


class HelperCallProver:
    """Proves application helper calls for one scan. Never imports or runs code."""

    def __init__(
        self,
        *,
        root: Path,
        resolve_import: ImportResolver,
        reviewed_call: ReviewedCall,
        mutable_constants_are_data: bool = False,
        appendable_collections: frozenset[str] = frozenset(),
    ) -> None:
        self._root = root
        self._resolve_import = resolve_import
        self._reviewed_call = reviewed_call
        # Whether a module-level list, dict or set still holds the data it was
        # written with when a helper reads it. The import check refuses every
        # in-place change at import except augmented assignment (`CONFIG |=
        # {...}`, from any module that imports or aliases it) and `.append(...)`
        # on the generated collections, so the caller passes True only when the
        # scanned source has no import-time augmented assignment.
        self._mutable_constants_are_data = mutable_constants_are_data
        self._appendable_collections = appendable_collections
        self._modules: dict[Path, _Module | None] = {}
        self._constants: dict[tuple[Path, str], _Kind] = {}
        self._constants_in_progress: set[tuple[Path, str]] = set()
        self._frozen_constants: dict[tuple[Path, str], bool] = {}
        self._defaults: dict[tuple[Path, int, int], _Kind] = {}
        self._steps = 0

    def check(
        self,
        path: Path,
        call: ast.Call,
        enclosing: ast.expr,
        shadowed: frozenset[str] = frozenset(),
    ) -> HelperCallVerdict:
        """Prove *call*, which appears in *enclosing* at module level of *path*.

        *shadowed* names the class-body bindings visible at the call, which
        hide module globals of the same name.
        """
        module = self._module(path)
        if module is None or not isinstance(call.func, ast.Name) or call.func.id in shadowed:
            return HelperCallVerdict(None)
        try:
            comprehension_names = _enclosing_comprehension_names(enclosing, call)
        except RecursionError:
            comprehension_names = None
        if comprehension_names is None or call.func.id in comprehension_names:
            return HelperCallVerdict(None)
        callee = self._resolve_global(module, call.func.id)
        if not isinstance(callee, _Function):
            return HelperCallVerdict(None)
        self._steps = 0
        frame = _Frame(module, local_names=frozenset())
        try:
            env = self._comprehension_env(frame, enclosing, call, shadowed)
            arguments = self._call_site_arguments(frame, call, env, shadowed)
            self._call_function(callee, call, arguments, path, stack=(), depth=1)
        except _Unproven as exc:
            return HelperCallVerdict(
                call.func.id,
                self._obstacle(callee, exc.node, exc.path, exc.reason),
            )
        except (_ProofLimit, RecursionError):
            # A long operator chain can nest deeper than the interpreter's
            # stack before the step budget runs out.
            return HelperCallVerdict(
                call.func.id, self._obstacle(callee, call, path, "proof_limit")
            )
        return HelperCallVerdict(call.func.id)

    @staticmethod
    def _obstacle(
        callee: _Function, node: ast.AST, path: Path, reason: ObstacleReason
    ) -> HelperObstacle:
        return HelperObstacle(
            helper=callee.definition.name,
            helper_path=callee.module.path,
            helper_line=callee.definition.lineno,
            node=node,
            path=path,
            reason=reason,
        )

    # Module source -----------------------------------------------------------

    def _module(self, path: Path) -> _Module | None:
        if path in self._modules:
            return self._modules[path]
        module = None
        if (
            path.suffix == ".py"
            and path.is_relative_to(self._root)
            and path.is_file()
            and not any(
                part.is_symlink()
                for part in (path, *path.parents)
                if part != self._root and part.is_relative_to(self._root)
            )
        ):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
            except (OSError, UnicodeError, SyntaxError, ValueError, RecursionError, MemoryError):
                tree = None
            if tree is not None:
                try:
                    module = _read_module(path, tree)
                except RecursionError:
                    module = None
        self._modules[path] = module
        return module

    def _resolve_global(self, module: _Module, name: str, reexports: int = 0) -> _Resolution:
        """What a module-level name holds whenever it is bound.

        Only a name bound exactly once in module scope, never declared
        ``global`` in a function and never reachable through ``import *``, has
        a single value the proof can read.
        """
        if module.star_import:
            return _Opaque()
        # Importing a submodule sets an attribute of the same name on its package.
        if module.path.name == "__init__.py" and (
            (module.path.parent / f"{name}.py").exists() or (module.path.parent / name).is_dir()
        ):
            return _Opaque()
        count = module.binding_counts.get(name, 0)
        if count == 0:
            # A name a re-exporting module does not bind is a submodule or an
            # ImportError, never the builtin. Binding __builtins__ replaces them.
            if (
                reexports == 0
                and hasattr(builtins, name)
                and "__builtins__" not in module.binding_counts
            ):
                return _Builtin(name)
            return _Opaque()
        if count != 1:
            return _Opaque()
        if name in module.functions:
            return _Function(module, module.functions[name])
        if name in module.constants:
            return _Constant(module, name)
        imported = module.imports.get(name)
        if isinstance(imported, ast.Import):
            alias = next(
                item
                for item in imported.names
                if (item.asname or item.name.partition(".")[0]) == name
            )
            # ``import os`` and ``import os.path`` both bind the os module.
            binds_os = alias.name == "os" if alias.asname else alias.name.partition(".")[0] == "os"
            if binds_os and not self._is_project_local(module.path, "os", 0):
                return _OsModule()
            return _Opaque()
        if isinstance(imported, ast.ImportFrom):
            alias = next(item for item in imported.names if (item.asname or item.name) == name)
            local, target = self._resolve_import(module.path, imported.module, imported.level)
            if not local:
                if (
                    imported.module == "os"
                    and imported.level == 0
                    and (alias.name == "environ" or alias.name in _ENVIRONMENT_FUNCTIONS)
                ):
                    return _OsSymbol(alias.name)
                return _Opaque()
            if target is None or reexports >= _MAX_REEXPORTS:
                return _Opaque()
            source = self._module(target)
            if source is None:
                return _Opaque()
            return self._resolve_global(source, alias.name, reexports + 1)
        return _Opaque()

    def _is_project_local(self, path: Path, module: str, level: int) -> bool:
        local, _target = self._resolve_import(path, module, level)
        return local

    def _constant_kind(self, constant: _Constant) -> _Kind:
        key = (constant.module.path, constant.name)
        if key in self._constants:
            return self._constants[key]
        if key in self._constants_in_progress:
            return _OPAQUE
        value = constant.module.constants[constant.name]
        if value is None:
            return _OPAQUE
        self._constants_in_progress.add(key)
        try:
            # A constant holds whatever its own statement produced at import, so
            # it is evaluated in module scope with a fresh call stack. Whether
            # that statement was safe to run is the import check's own concern.
            kind = self._module_value_kind(constant.module, value)
            if (
                kind == _DATA
                and not (
                    self._mutable_constants_are_data
                    and constant.name not in self._appendable_collections
                )
                and not self._frozen(constant.module, value, {})
            ):
                kind = _OPAQUE
        finally:
            self._constants_in_progress.discard(key)
        self._constants[key] = kind
        return kind

    def _frozen_constant(self, constant: _Constant) -> bool:
        key = (constant.module.path, constant.name)
        if key not in self._frozen_constants:
            value = constant.module.constants[constant.name]
            self._frozen_constants[key] = False
            self._frozen_constants[key] = value is not None and self._frozen(
                constant.module, value, {}
            )
        return self._frozen_constants[key]

    def _frozen(
        self,
        module: _Module,
        expression: ast.expr,
        scope: dict[str, ast.expr],
        *,
        elements: bool = False,
    ) -> bool:
        """Whether a module-level data value (or, with *elements*, each item it
        yields) is immutable all the way down.

        *scope* maps the comprehension variables in view to their iterables.
        """
        self._step()
        if isinstance(expression, ast.Name):
            if expression.id in scope:
                # The iterable is read where the variable is not yet bound.
                outer = {name: item for name, item in scope.items() if name != expression.id}
                return self._frozen(module, scope[expression.id], outer, elements=True)
            resolved = self._resolve_global(module, expression.id)
            # A frozen value's items are frozen too.
            return (
                isinstance(resolved, _Constant)
                and self._constant_kind(resolved) == _DATA
                and self._frozen_constant(resolved)
            )
        if isinstance(expression, ast.Constant):
            return True
        if isinstance(expression, ast.JoinedStr):
            return True
        if isinstance(expression, (ast.Tuple, ast.List, ast.Set)) and (
            elements or isinstance(expression, ast.Tuple)
        ):
            return all(
                self._frozen(module, item.value, scope, elements=True)
                if isinstance(item, ast.Starred)
                else self._frozen(module, item, scope)
                for item in expression.elts
            )
        if isinstance(expression, (ast.GeneratorExp, ast.ListComp, ast.SetComp)) and elements:
            inner = dict(scope)
            for generator in expression.generators:
                for name in _target_names(generator.target):
                    inner[name] = generator.iter
                if not isinstance(generator.target, ast.Name):
                    # Unpacked items are only frozen when every item is.
                    return False
            return self._frozen(module, expression.elt, inner)
        if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
            resolved = self._resolve_global(module, expression.func.id)
            if isinstance(resolved, _Builtin) and len(expression.args) == 1:
                if resolved.name in {"tuple", "frozenset"}:
                    return self._frozen(module, expression.args[0], scope, elements=True)
                return resolved.name in _SCALAR_BUILTINS
            return False
        if isinstance(expression, (ast.Compare, ast.UnaryOp)):
            return True
        if isinstance(expression, ast.BinOp):
            return self._frozen(module, expression.left, scope) and self._frozen(
                module, expression.right, scope
            )
        if isinstance(expression, ast.BoolOp):
            return all(self._frozen(module, item, scope) for item in expression.values)
        if isinstance(expression, ast.IfExp):
            return self._frozen(module, expression.body, scope) and self._frozen(
                module, expression.orelse, scope
            )
        if isinstance(expression, ast.Subscript):
            return self._frozen(module, expression.value, scope)
        return False

    def _default_kind(self, module: _Module, default: ast.expr) -> _Kind:
        """A parameter default, computed once when its ``def`` ran at import."""
        key = (module.path, default.lineno, default.col_offset)
        if key not in self._defaults:
            # A default that reaches its own function again is not followed.
            self._defaults[key] = _OPAQUE
            self._defaults[key] = self._module_value_kind(module, default)
        return self._defaults[key]

    def _module_value_kind(
        self,
        module: _Module,
        value: ast.expr,
        env: dict[str, _Kind] | None = None,
    ) -> _Kind:
        frame = _Frame(module, local_names=frozenset(env or ()), env=dict(env or {}))
        try:
            return self._expression(frame, value, stack=(), depth=0)
        except _Unproven:
            return _OPAQUE

    # Call sites ----------------------------------------------------------------

    def _comprehension_env(
        self,
        frame: _Frame,
        enclosing: ast.expr,
        call: ast.Call,
        shadowed: frozenset[str],
    ) -> dict[str, _Kind]:
        """Kinds of the comprehension variables visible at a module-level call."""
        env: dict[str, _Kind] = {}
        for _comprehension, generators in _comprehensions_around(enclosing, call):
            for generator in generators:
                kind = (
                    _OPAQUE
                    if _names(generator.iter) & shadowed
                    else self._module_value_kind(frame.module, generator.iter, env)
                )
                for name in _target_names(generator.target):
                    env[name] = kind
        return env

    def _call_site_arguments(
        self,
        frame: _Frame,
        call: ast.Call,
        env: dict[str, _Kind],
        shadowed: frozenset[str],
    ) -> tuple[list[_Kind], dict[str, _Kind]]:
        if any(isinstance(argument, ast.Starred) for argument in call.args) or any(
            keyword.arg is None for keyword in call.keywords
        ):
            raise _Unproven(call, frame.module.path)

        def kind(argument: ast.expr) -> _Kind:
            if _names(argument) & shadowed:
                return _OPAQUE
            return self._module_value_kind(frame.module, argument, env)

        return (
            [kind(argument) for argument in call.args],
            {keyword.arg: kind(keyword.value) for keyword in call.keywords if keyword.arg},
        )

    # Helper bodies -------------------------------------------------------------

    def _call_function(
        self,
        function: _Function,
        call: ast.Call,
        arguments: tuple[list[_Kind], dict[str, _Kind]],
        caller_path: Path,
        *,
        stack: tuple[tuple[Path, str], ...],
        depth: int,
    ) -> _Kind:
        """Prove *function*'s body for one *call*, which is in *caller_path*."""
        definition = function.definition
        path = function.module.path
        key = (path, definition.name)
        if key in stack:
            raise _Unproven(call, caller_path, "recursion")
        if depth > MAX_HELPER_DEPTH:
            raise _Unproven(call, caller_path, "depth_limit")
        # A decorator replaces the function; generators and coroutines run
        # their body later, so neither is the body read here.
        if definition.decorator_list or _is_generator(definition):
            raise _Unproven(definition, path)
        parameters = definition.args
        if parameters.vararg is not None or parameters.kwarg is not None:
            raise _Unproven(definition, path)
        env = self._bind_parameters(function, call, arguments, caller_path)
        frame = _Frame(
            function.module,
            local_names=_function_local_names(definition),
            env=env,
        )
        inner_stack = (*stack, key)
        if self._block(frame, definition.body, stack=inner_stack, depth=depth) is not None:
            frame.returns.append(_DATA)  # falls off the end: None
        return _OPAQUE if _OPAQUE in frame.returns else _DATA

    def _bind_parameters(
        self,
        function: _Function,
        call: ast.Call,
        arguments: tuple[list[_Kind], dict[str, _Kind]],
        caller_path: Path,
    ) -> dict[str, _Kind]:
        definition = function.definition
        # A call that does not match the signature is reported at the call.
        path = caller_path
        parameters = definition.args
        positional = [*parameters.posonlyargs, *parameters.args]
        positional_defaults = [None] * (len(positional) - len(parameters.defaults)) + list(
            parameters.defaults
        )
        values, keywords = arguments
        if len(values) > len(positional):
            raise _Unproven(call, path)
        env: dict[str, _Kind] = {}
        keyword_names = {argument.arg for argument in (*parameters.args, *parameters.kwonlyargs)}
        if set(keywords) - keyword_names:
            raise _Unproven(call, path)
        for index, (parameter, default) in enumerate(
            zip(positional, positional_defaults, strict=True)
        ):
            if index < len(values):
                if parameter.arg in keywords:
                    raise _Unproven(call, path)
                env[parameter.arg] = values[index]
            elif parameter.arg in keywords and parameter not in parameters.posonlyargs:
                env[parameter.arg] = keywords[parameter.arg]
            elif default is not None:
                env[parameter.arg] = self._default_kind(function.module, default)
            else:
                raise _Unproven(call, path)
        for parameter, default in zip(parameters.kwonlyargs, parameters.kw_defaults, strict=True):
            if parameter.arg in keywords:
                env[parameter.arg] = keywords[parameter.arg]
            elif default is not None:
                env[parameter.arg] = self._default_kind(function.module, default)
            else:
                raise _Unproven(call, path)
        return env

    def _step(self) -> None:
        self._steps += 1
        if self._steps > MAX_PROOF_STEPS:
            raise _ProofLimit

    def _block(
        self,
        frame: _Frame,
        statements: list[ast.stmt],
        *,
        stack: tuple[tuple[Path, str], ...],
        depth: int,
    ) -> dict[str, _Kind] | None:
        """Run *statements*; return the environment after, or None if every path left."""
        for statement in statements:
            self._step()
            path = frame.module.path
            if isinstance(statement, ast.Pass):
                continue
            if isinstance(statement, ast.Expr):
                self._expression(frame, statement.value, stack=stack, depth=depth)
                continue
            if isinstance(statement, ast.Return):
                frame.returns.append(
                    _DATA
                    if statement.value is None
                    else self._expression(frame, statement.value, stack=stack, depth=depth)
                )
                return None
            if isinstance(statement, ast.Raise):
                self._raise(frame, statement, stack=stack, depth=depth)
                return None
            if isinstance(statement, ast.Assert):
                self._data(frame, statement.test, stack=stack, depth=depth)
                if statement.msg is not None:
                    self._data(frame, statement.msg, stack=stack, depth=depth)
                continue
            if isinstance(statement, ast.Assign):
                kind = self._expression(frame, statement.value, stack=stack, depth=depth)
                for target in statement.targets:
                    self._assign(frame, target, kind)
                continue
            if isinstance(statement, ast.AnnAssign):
                # A local annotation is never evaluated.
                if not isinstance(statement.target, ast.Name):
                    raise _Unproven(statement.target, path)
                if statement.value is not None:
                    kind = self._expression(frame, statement.value, stack=stack, depth=depth)
                    self._assign(frame, statement.target, kind)
                continue
            if isinstance(statement, ast.If):
                self._data(frame, statement.test, stack=stack, depth=depth)
                branches = []
                for body in (statement.body, statement.orelse):
                    branch = _Frame(frame.module, frame.local_names, dict(frame.env), frame.returns)
                    branches.append(self._block(branch, body, stack=stack, depth=depth))
                remaining = [env for env in branches if env is not None]
                if not remaining:
                    return None
                frame.env = _merge(remaining)
                continue
            raise _Unproven(statement, path)
        return frame.env

    def _assign(self, frame: _Frame, target: ast.expr, kind: _Kind) -> None:
        if isinstance(target, ast.Name):
            frame.env[target.id] = kind
            return
        if isinstance(target, (ast.Tuple, ast.List)) and all(
            isinstance(item, ast.Name) for item in target.elts
        ):
            # Unpacking iterates the value.
            if kind != _DATA:
                raise _Unproven(target, frame.module.path)
            for item in target.elts:
                assert isinstance(item, ast.Name)
                frame.env[item.id] = _DATA
            return
        raise _Unproven(target, frame.module.path)

    def _raise(
        self,
        frame: _Frame,
        statement: ast.Raise,
        *,
        stack: tuple[tuple[Path, str], ...],
        depth: int,
    ) -> None:
        path = frame.module.path
        if statement.cause is not None and not (
            isinstance(statement.cause, ast.Constant) and statement.cause.value is None
        ):
            raise _Unproven(statement.cause, path)
        exception = statement.exc
        call = exception if isinstance(exception, ast.Call) else None
        name = call.func if call is not None else exception
        if not (
            isinstance(name, ast.Name)
            and name.id not in frame.local_names
            and isinstance(resolved := self._resolve_global(frame.module, name.id), _Builtin)
            and _builtin_exception(resolved.name)
        ):
            raise _Unproven(exception or statement, path)
        if call is not None:
            # The message is formatted when the failed import is reported.
            if call.keywords or any(isinstance(item, ast.Starred) for item in call.args):
                raise _Unproven(call, path)
            for argument in call.args:
                self._data(frame, argument, stack=stack, depth=depth)

    # Expressions ---------------------------------------------------------------

    def _data(
        self,
        frame: _Frame,
        expression: ast.expr,
        *,
        stack: tuple[tuple[Path, str], ...],
        depth: int,
        at: ast.AST | None = None,
    ) -> None:
        """Require data; an opaque value is reported at *at*, the operation using it."""
        if self._expression(frame, expression, stack=stack, depth=depth) != _DATA:
            raise _Unproven(at or expression, frame.module.path)

    def _expression(
        self,
        frame: _Frame,
        expression: ast.expr,
        *,
        stack: tuple[tuple[Path, str], ...],
        depth: int,
    ) -> _Kind:
        self._step()
        path = frame.module.path

        def data(item: ast.expr) -> None:
            self._data(frame, item, stack=stack, depth=depth, at=expression)

        def value(item: ast.expr) -> _Kind:
            return self._expression(frame, item, stack=stack, depth=depth)

        def consumed(item: ast.expr) -> None:
            self._consumed(frame, item, stack=stack, depth=depth, at=expression)

        if isinstance(expression, ast.Constant):
            return _DATA
        if isinstance(expression, ast.JoinedStr):
            for part in expression.values:
                if isinstance(part, ast.Constant):
                    continue
                if not isinstance(part, ast.FormattedValue):
                    raise _Unproven(part, path)
                data(part.value)
                # A computed width could build an unbounded string.
                if part.format_spec is not None and not (
                    isinstance(part.format_spec, ast.JoinedStr)
                    and all(isinstance(item, ast.Constant) for item in part.format_spec.values)
                ):
                    raise _Unproven(part.format_spec, path)
            return _DATA
        if isinstance(expression, ast.Name) and isinstance(expression.ctx, ast.Load):
            return self._name(frame, expression)
        if isinstance(expression, ast.BoolOp):
            kinds = [value(item) for item in expression.values]
            # Every operand but the last has its truth value tested.
            if _OPAQUE in kinds[:-1]:
                raise _Unproven(expression, path)
            return kinds[-1]
        if isinstance(expression, ast.UnaryOp):
            data(expression.operand)
            return _DATA
        if isinstance(expression, ast.BinOp):
            if not isinstance(expression.op, _DATA_BINARY_OPERATORS):
                raise _Unproven(expression, path)
            data(expression.left)
            data(expression.right)
            return _DATA
        if isinstance(expression, ast.Compare):
            operands = [expression.left, *expression.comparators]
            kinds = [value(item) for item in operands]
            for index, operator in enumerate(expression.ops):
                # Identity never runs code; every other comparison can.
                if not isinstance(operator, (ast.Is, ast.IsNot)) and (
                    kinds[index] != _DATA or kinds[index + 1] != _DATA
                ):
                    raise _Unproven(expression, path)
            return _DATA
        if isinstance(expression, ast.IfExp):
            data(expression.test)
            return (
                _OPAQUE if _OPAQUE in (value(expression.body), value(expression.orelse)) else _DATA
            )
        if isinstance(expression, (ast.Tuple, ast.List)) and isinstance(expression.ctx, ast.Load):
            kinds = []
            for item in expression.elts:
                if isinstance(item, ast.Starred):
                    consumed(item.value)
                    kinds.append(_DATA)
                else:
                    kinds.append(value(item))
            return _OPAQUE if _OPAQUE in kinds else _DATA
        if isinstance(expression, ast.Set):
            for item in expression.elts:
                if isinstance(item, ast.Starred):
                    consumed(item.value)
                else:
                    data(item)
            return _DATA
        if isinstance(expression, ast.Dict):
            kinds = []
            for key, item in zip(expression.keys, expression.values, strict=True):
                if key is None:
                    data(item)
                    continue
                data(key)
                kinds.append(value(item))
            return _OPAQUE if _OPAQUE in kinds else _DATA
        if isinstance(expression, ast.GeneratorExp):
            # A generator runs its body when something iterates it, with the
            # values its variables hold then; see _consumed.
            raise _Unproven(expression, path)
        if isinstance(expression, _COMPREHENSIONS):
            return self._comprehension(frame, expression, stack=stack, depth=depth)
        if isinstance(expression, ast.Subscript) and isinstance(expression.ctx, ast.Load):
            if not self._is_environ(frame, expression.value):
                data(expression.value)
            data(expression.slice)
            return _DATA
        if isinstance(expression, ast.Slice):
            for item in (expression.lower, expression.upper, expression.step):
                if item is not None:
                    data(item)
            return _DATA
        if isinstance(expression, ast.Call):
            return self._call(frame, expression, stack=stack, depth=depth)
        raise _Unproven(expression, path)

    def _consumed(
        self,
        frame: _Frame,
        expression: ast.expr,
        *,
        stack: tuple[tuple[Path, str], ...],
        depth: int,
        at: ast.AST,
    ) -> None:
        """Require data that is iterated right away, which may be a generator.

        No statement runs between creating a generator passed straight to an
        eager consumer and iterating it, so its variables hold the values seen here.
        """
        if isinstance(expression, ast.GeneratorExp):
            kind = self._comprehension(frame, expression, stack=stack, depth=depth)
            if kind != _DATA:
                raise _Unproven(at, frame.module.path)
            return
        self._data(frame, expression, stack=stack, depth=depth, at=at)

    def _name(self, frame: _Frame, name: ast.Name) -> _Kind:
        if name.id in frame.env:
            return frame.env[name.id]
        if name.id in frame.local_names:
            # Read before any assignment on this path.
            raise _Unproven(name, frame.module.path)
        resolved = self._resolve_global(frame.module, name.id)
        if isinstance(resolved, _Constant):
            return self._constant_kind(resolved)
        if isinstance(resolved, _Builtin) and resolved.name in _MODULE_ATTRIBUTES:
            return _DATA
        # Reading a global runs no code; what it holds is unknown.
        return _OPAQUE

    def _comprehension(
        self,
        frame: _Frame,
        expression: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        *,
        stack: tuple[tuple[Path, str], ...],
        depth: int,
    ) -> _Kind:
        path = frame.module.path
        scope = _Frame(
            frame.module,
            frame.local_names | _comprehension_names(expression),
            dict(frame.env),
            frame.returns,
        )
        for generator in expression.generators:
            if generator.is_async:
                raise _Unproven(expression, path)
            self._consumed(scope, generator.iter, stack=stack, depth=depth, at=expression)
            for target in _target_nodes(generator.target):
                if not isinstance(target, ast.Name):
                    raise _Unproven(target, path)
                scope.env[target.id] = _DATA
            for condition in generator.ifs:
                self._data(scope, condition, stack=stack, depth=depth, at=expression)
        if isinstance(expression, ast.DictComp):
            self._data(scope, expression.key, stack=stack, depth=depth, at=expression)
            return self._expression(scope, expression.value, stack=stack, depth=depth)
        if isinstance(expression, ast.SetComp):
            self._data(scope, expression.elt, stack=stack, depth=depth, at=expression)
            return _DATA
        return self._expression(scope, expression.elt, stack=stack, depth=depth)

    def _is_environ(self, frame: _Frame, expression: ast.expr) -> bool:
        if isinstance(expression, ast.Name):
            return expression.id not in frame.local_names and self._resolve_global(
                frame.module, expression.id
            ) == _OsSymbol("environ")
        return (
            isinstance(expression, ast.Attribute)
            and expression.attr == "environ"
            and isinstance(expression.value, ast.Name)
            and expression.value.id not in frame.local_names
            and isinstance(self._resolve_global(frame.module, expression.value.id), _OsModule)
        )

    def _is_getenv(self, frame: _Frame, function: ast.expr) -> bool:
        if isinstance(function, ast.Name):
            return function.id not in frame.local_names and self._resolve_global(
                frame.module, function.id
            ) == _OsSymbol("getenv")
        return (
            isinstance(function, ast.Attribute)
            and function.attr in _ENVIRONMENT_FUNCTIONS
            and isinstance(function.value, ast.Name)
            and function.value.id not in frame.local_names
            and isinstance(self._resolve_global(frame.module, function.value.id), _OsModule)
        ) or (
            isinstance(function, ast.Attribute)
            and function.attr == "get"
            and self._is_environ(frame, function.value)
        )

    def _call(
        self,
        frame: _Frame,
        call: ast.Call,
        *,
        stack: tuple[tuple[Path, str], ...],
        depth: int,
    ) -> _Kind:
        path = frame.module.path
        if any(isinstance(argument, ast.Starred) for argument in call.args) or any(
            keyword.arg is None for keyword in call.keywords
        ):
            raise _Unproven(call, path)
        function = call.func
        if self._is_getenv(frame, function):
            if call.keywords or not 1 <= len(call.args) <= 2:
                raise _Unproven(call, path)
            for argument in call.args:
                self._data(frame, argument, stack=stack, depth=depth, at=call)
            return _DATA
        if isinstance(function, ast.Attribute):
            if function.attr in _DATA_METHODS:
                self._data(frame, function.value, stack=stack, depth=depth, at=call)
                for argument in (*call.args, *(keyword.value for keyword in call.keywords)):
                    if function.attr in _CONSUMING_METHODS:
                        self._consumed(frame, argument, stack=stack, depth=depth, at=call)
                    else:
                        self._data(frame, argument, stack=stack, depth=depth, at=call)
                return _DATA
            receiver = _expression_root(function)
            if (
                receiver is None
                or receiver.id in frame.local_names
                or not self._reviewed_call(path, call)
            ):
                raise _Unproven(call, path)
            # A reviewed method form such as logging.getLogger(...) or
            # Path(...).resolve(): its receiver and data arguments must be proven.
            self._expression(frame, function.value, stack=stack, depth=depth)
            for argument in (*call.args, *(keyword.value for keyword in call.keywords)):
                self._data(frame, argument, stack=stack, depth=depth, at=call)
            return _OPAQUE
        if not isinstance(function, ast.Name) or function.id in frame.local_names:
            raise _Unproven(call, path)
        resolved = self._resolve_global(frame.module, function.id)
        if isinstance(resolved, _Builtin) and resolved.name in _DATA_BUILTINS:
            if call.keywords:
                raise _Unproven(call, path)
            for argument in call.args:
                if resolved.name in _CONSUMING_BUILTINS:
                    self._consumed(frame, argument, stack=stack, depth=depth, at=call)
                else:
                    self._data(frame, argument, stack=stack, depth=depth, at=call)
            return _DATA
        if isinstance(resolved, _Function):
            arguments = (
                [
                    self._expression(frame, argument, stack=stack, depth=depth)
                    for argument in call.args
                ],
                {
                    keyword.arg: self._expression(frame, keyword.value, stack=stack, depth=depth)
                    for keyword in call.keywords
                    if keyword.arg
                },
            )
            try:
                return self._call_function(
                    resolved, call, arguments, path, stack=stack, depth=depth + 1
                )
            except _Unproven:
                # The configured_* settings functions the generated agents
                # call at import stay accepted here as they are at module level.
                if not self._reviewed_call(path, call):
                    raise
                return _OPAQUE
        if not isinstance(resolved, _Builtin) and self._reviewed_call(path, call):
            # A reviewed declaration such as AgentSpec(...): the module-level
            # check accepts it with any module-level value as an argument, and
            # every value here is one of those or built from them.
            for argument in (*call.args, *(keyword.value for keyword in call.keywords)):
                self._expression(frame, argument, stack=stack, depth=depth)
            return _OPAQUE
        raise _Unproven(call, path)


def _expression_root(expression: ast.expr) -> ast.Name | None:
    while isinstance(expression, (ast.Attribute, ast.Call, ast.Subscript)):
        expression = expression.func if isinstance(expression, ast.Call) else expression.value
    return expression if isinstance(expression, ast.Name) else None


def _read_module(path: Path, tree: ast.Module) -> _Module:
    counts: dict[str, int] = {}
    star_import = False
    functions: dict[str, ast.FunctionDef] = {}
    constants: dict[str, ast.expr | None] = {}
    imports: dict[str, ast.ImportFrom | ast.Import] = {}

    def bind(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    for name in _module_scope_bindings(tree):
        bind(name)
    # A function that declares a name global can rebind it whenever it runs.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                counts[name] = counts.get(name, 0) + 2
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            star_import = True
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef):
            functions[statement.name] = statement
        elif isinstance(statement, ast.Assign):
            if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                constants[statement.targets[0].id] = statement.value
        elif isinstance(statement, ast.AnnAssign):
            if isinstance(statement.target, ast.Name):
                constants[statement.target.id] = statement.value
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                imports[alias.asname or alias.name.partition(".")[0]] = statement
        elif isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                imports[alias.asname or alias.name] = statement
    return _Module(path, counts, star_import, functions, constants, imports)


def _module_scope_bindings(tree: ast.Module) -> Iterator[str]:
    """Every binding of a module-scope name, counting each occurrence.

    Comprehension variables are counted too: that only makes a name look
    rebound, which the proof treats as unknown.
    """

    def visit(node: ast.AST) -> Iterator[str]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
            for decorator in node.decorator_list:
                yield from visit(decorator)
            if isinstance(node, ast.ClassDef):
                for base in (*node.bases, *(keyword.value for keyword in node.keywords)):
                    yield from visit(base)
            else:
                for default in (*node.args.defaults, *node.args.kw_defaults):
                    if default is not None:
                        yield from visit(default)
            return
        if isinstance(node, ast.Lambda):
            for default in (*node.args.defaults, *node.args.kw_defaults):
                if default is not None:
                    yield from visit(default)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    yield alias.asname or alias.name.partition(".")[0]
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            yield node.id
        elif (isinstance(node, ast.ExceptHandler) and node.name) or (
            isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name
        ):
            yield node.name
        elif isinstance(node, ast.MatchMapping) and node.rest:
            yield node.rest
        elif type(node).__name__ == "TypeAlias":
            name = getattr(node, "name", None)
            if isinstance(name, ast.Name):
                yield name.id
        for child in ast.iter_child_nodes(node):
            yield from visit(child)

    for statement in tree.body:
        yield from visit(statement)


def _function_local_names(definition: ast.FunctionDef) -> frozenset[str]:
    names = {
        argument.arg
        for argument in (
            *definition.args.posonlyargs,
            *definition.args.args,
            *definition.args.kwonlyargs,
        )
    }

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            return
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in definition.body:
        visit(statement)
    return frozenset(names)


def _is_generator(definition: ast.FunctionDef) -> bool:
    def visit(node: ast.AST) -> bool:
        if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return False
        return any(visit(child) for child in ast.iter_child_nodes(node))

    return any(visit(statement) for statement in definition.body)


def _merge(environments: list[dict[str, _Kind]]) -> dict[str, _Kind]:
    merged: dict[str, _Kind] = {}
    for env in environments:
        for name, kind in env.items():
            merged[name] = _OPAQUE if _OPAQUE in (kind, merged.get(name, kind)) else _DATA
    return merged


def _target_nodes(target: ast.expr) -> tuple[ast.expr, ...]:
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(item for element in target.elts for item in _target_nodes(element))
    return (target,)


def _target_names(target: ast.expr) -> set[str]:
    return {item.id for item in _target_nodes(target) if isinstance(item, ast.Name)}


def _names(expression: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}


def _comprehension_names(
    expression: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
) -> frozenset[str]:
    return frozenset(
        name for generator in expression.generators for name in _target_names(generator.target)
    )


def _comprehensions_around(
    enclosing: ast.expr, call: ast.Call
) -> list[tuple[ast.expr, list[ast.comprehension]]]:
    """The comprehensions whose variables are visible at *call*, outermost first.

    For each, the generators bound before the call is evaluated: a call in a
    generator's iterable sees only the generators before it.
    """

    found: list[tuple[ast.expr, list[ast.comprehension]]] = []

    def contains(node: ast.AST) -> bool:
        return any(item is call for item in ast.walk(node))

    def visit(node: ast.AST) -> bool:
        if node is call:
            return True
        if isinstance(node, _COMPREHENSIONS):
            for index, generator in enumerate(node.generators):
                if contains(generator.iter):
                    found.append((node, node.generators[:index]))
                    return visit(generator.iter)
                for condition in generator.ifs:
                    if contains(condition):
                        found.append((node, node.generators[: index + 1]))
                        return visit(condition)
            parts = (node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)
            for part in parts:
                if contains(part):
                    found.append((node, list(node.generators)))
                    return visit(part)
            return False
        return any(visit(child) for child in ast.iter_child_nodes(node) if contains(child))

    visit(enclosing)
    return found


def _enclosing_comprehension_names(enclosing: ast.expr, call: ast.Call) -> frozenset[str] | None:
    """Names comprehensions bind around *call*, or None if *call* is not in *enclosing*."""
    if not any(item is call for item in ast.walk(enclosing)):
        return None
    return frozenset(
        name
        for _comprehension, generators in _comprehensions_around(enclosing, call)
        for generator in generators
        for name in _target_names(generator.target)
    )
