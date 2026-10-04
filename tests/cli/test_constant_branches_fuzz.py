"""Differential fuzz: simplification never changes what a module does.

Random modules mix literal branches with calls, properties, ``__bool__``,
walrus, lambdas, comprehensions, generators and coroutines. Each one runs before
and after simplification on the same inputs; the traced effects, results and
exceptions must match. A shape the pass refuses must raise ConstantBranchError.
"""

from __future__ import annotations

import inspect
import random
from typing import Any

import pytest

from cayu.cli._constant_branches import ConstantBranchError, simplify_constant_branches

_SEEDS = range(300)
_INPUTS = ("None", "0", "1", "Obj(True)", "Obj(False)")

_PRELUDE = """
T = []
def c(tag, v=None):
    T.append(tag)
    return v
class Obj:
    def __init__(self, b): self.b = b
    def __bool__(self):
        T.append("bool"); return self.b
    @property
    def attr(self):
        T.append("attr"); return self.b
    def __repr__(self): return f"Obj({self.b})"
class Aw:
    def __init__(self, v): self.v = v
    def __await__(self):
        T.append("aw")
        return self.v
        yield
G = None
H = Obj(True)
"""


class _Generator:
    def __init__(self, rnd: random.Random, kind: str) -> None:
        self.rnd = rnd
        self.kind = kind
        self.count = 0

    def tag(self) -> str:
        self.count += 1
        return f"t{self.count}"

    def literal(self) -> str:
        return self.rnd.choice(["True", "False", "not True", "not False"])

    def leaf(self) -> str:
        rnd = self.rnd
        options = [
            self.literal,
            self.literal,
            self.literal,
            lambda: rnd.choice(["p", "q", "G", "H", "None", "0", "1", "'s'"]),
            lambda: f"c('{self.tag()}', {rnd.choice(['0', '1', 'None', 'p', 'True', 'False'])})",
            lambda: f"{rnd.choice(['p', 'q', 'G'])} is {rnd.choice(['None', 'G', 'p', 'True'])}",
            lambda: f"{rnd.choice(['p', 'q', 'G'])} is not None",
            lambda: "H.attr",
            lambda: f"(w := c('{self.tag()}', {rnd.choice(['0', '1'])}))",
            lambda: f"len([{self.literal()} for _i in range(2)])",
            lambda: f'f"{{{self.literal()}}}"',
            lambda: f"(lambda z=p: {self.literal()} or z)()",
        ]
        if self.kind == "gen":
            options.append(lambda: f"(yield c('{self.tag()}', 1))")
        if self.kind == "async":
            options.append(lambda: f"(await Aw(c('{self.tag()}', 1)))")
        return rnd.choice(options)()

    def expression(self, depth: int = 0) -> str:
        rnd = self.rnd
        if depth > 2 or rnd.random() < 0.3:
            return self.leaf()
        shape = rnd.randrange(6)
        if shape == 0:
            inner = self.expression(depth + 1)
            return f"not {inner}" if rnd.random() < 0.5 else f"not ({inner})"
        if shape in (1, 2):
            operator = rnd.choice([" and ", " or "])
            operands = [self.expression(depth + 1) for _ in range(rnd.randint(2, 3))]
            return "(" + operator.join(operands) + ")"
        if shape == 3:
            body, test, orelse = (self.expression(depth + 1) for _ in range(3))
            return f"({body} if {test} else {orelse})"
        if shape == 4:
            return f"({self.expression(depth + 1)}) == ({self.expression(depth + 1)})"
        return f"c('{self.tag()}', {self.expression(depth + 1)})"

    def condition(self) -> str:
        rnd = self.rnd
        shape = rnd.randrange(5)
        if shape == 0:
            return self.literal()
        if shape == 1:
            operator = rnd.choice([" and ", " or "])
            choices = [
                rnd.choice([self.literal(), self.leaf(), self.expression(1)])
                for _ in range(rnd.randint(2, 3))
            ]
            return operator.join(choices)
        if shape == 2:
            return f"{self.leaf()} {rnd.choice(['and', 'or'])} {self.literal()}"
        return self.expression()

    def multiline(self, indent: str) -> str:
        value = (
            f"{self.expression(1)}\n{indent}    if {self.condition()}\n"
            f"{indent}    else {self.expression(1)}"
        )
        opener = self.rnd.choice(["x = ", "return", "return ", "x = not", "c('ml', ", "x = -"])
        if opener == "c('ml', ":
            return f"{indent}c('ml', (\n{indent}    {value}\n{indent}))"
        return f"{indent}{opener}(\n{indent}    {value}\n{indent})"

    def block(self, indent: str, depth: int) -> list[str]:
        lines: list[str] = []
        for _ in range(self.rnd.randint(1, 3)):
            lines.extend(self.statement(indent, depth))
        return lines

    def statement(self, indent: str, depth: int) -> list[str]:
        shape = self.rnd.randrange(16 if depth < 2 else 6)
        if shape < 6:
            return self.simple_statement(shape, indent)
        return self.compound_statement(shape, indent, depth)

    def simple_statement(self, shape: int, indent: str) -> list[str]:
        rnd = self.rnd
        if shape == 0:
            return [f"{indent}c('{self.tag()}', {self.expression()})"]
        if shape == 1:
            return [f"{indent}x = {self.expression()}"]
        if shape == 2:
            if rnd.random() < 0.3:
                return [f"{indent}return {self.expression()}"]
            return [f"{indent}x = {self.condition()}"]
        if shape == 3:
            return [self.multiline(indent)]
        if shape == 4:
            return [
                f"{indent}y = {rnd.choice(['p', 'q', 'G', '1'])}",
                f"{indent}if {self.literal()}:",
                f"{indent}    c('use', y)",
            ]
        return [f"{indent}c('{self.tag()}', {self.condition()})"]

    def compound_statement(self, shape: int, indent: str, depth: int) -> list[str]:
        rnd = self.rnd
        inner = indent + "    "
        if shape in (6, 7, 8):
            lines = [f"{indent}if {self.condition()}:"]
            if rnd.random() < 0.3:
                lines.append(f"{inner}# comment")
            lines += self.block(inner, depth + 1)
            if rnd.random() < 0.3:
                lines.append("")
            for _ in range(rnd.randint(0, 2)):
                lines += [f"{indent}elif {self.condition()}:", *self.block(inner, depth + 1)]
            if rnd.random() < 0.5:
                lines += [f"{indent}else:", *self.block(inner, depth + 1)]
            return lines
        if shape in (9, 10):
            if shape == 9:
                lines = [f"{indent}for _j in range(2):", *self.block(inner, depth + 1)]
                lines += [f"{inner}if {self.condition()}:", f"{inner}    break"]
            else:
                # Always break, so no generated loop can run forever.
                lines = [f"{indent}while {self.condition()}:", *self.block(inner, depth + 1)]
                lines.append(f"{inner}break")
            if rnd.random() < 0.5:
                lines += [f"{indent}else:", *self.block(inner, depth + 1)]
            return lines
        if shape == 11:
            lines = [f"{indent}try:", *self.block(inner, depth + 1)]
            lines += [f"{indent}except ZeroDivisionError:", *self.block(inner, depth + 1)]
            if rnd.random() < 0.5:
                lines += [f"{indent}finally:", *self.block(inner, depth + 1)]
            return lines
        if shape == 12:
            case = f"{inner}case True if {self.condition()}:"
            lines = [f"{indent}match {self.expression()}:", case]
            lines += self.block(inner + "    ", depth + 1)
            return [*lines, f"{inner}case _:", *self.block(inner + "    ", depth + 1)]
        if shape == 13 and self.kind == "def":
            lines = [f"{indent}def inner(a=p):", *self.block(inner, depth + 1)]
            return [*lines, f"{inner}return a", f"{indent}c('inner', inner())"]
        if shape == 14:
            return [f"{indent}x = [{self.expression()} for _k in range(2) if {self.condition()}]"]
        test = self.condition()
        return [f"{indent}x = {self.expression()} if {test} else {self.expression()}"]

    def module(self) -> str:
        head = "async def f(p, q):" if self.kind == "async" else "def f(p, q):"
        body = self.block("    ", 0)
        if self.kind == "gen":
            body.append("    yield 'end'")
        body.append("    return 'done'")
        top: list[str] = []
        if self.rnd.random() < 0.4:
            top = [f"if {self.condition()}:", "    M = c('mod', 1)", "else:", "    M = c('mod', 2)"]
        return "\n".join([*top, head, *body]) + "\n"


def _shown(value: Any) -> str:
    if callable(value) and not isinstance(value, type):
        return "<callable>"
    return repr(value)


def _call(function: Any, namespace: dict[str, Any], arguments: tuple[Any, Any]) -> Any:
    if inspect.isgeneratorfunction(function):
        items: list[Any] = []
        generator = function(*arguments)
        try:
            item = next(generator)
            for _ in range(20):
                items.append(_shown(item))
                item = generator.send(namespace["Obj"](False))
        except StopIteration as stop:
            items.append(("returned", _shown(stop.value)))
        return items
    if inspect.iscoroutinefunction(function):
        coroutine = function(*arguments)
        try:
            coroutine.send(None)
        except StopIteration as stop:
            return _shown(stop.value)
        else:
            return "suspended"
        finally:
            coroutine.close()
    return _shown(function(*arguments))


def _behaviour(source: str) -> list[Any]:
    namespace: dict[str, Any] = {}
    exec(compile(_PRELUDE, "<prelude>", "exec"), namespace)
    try:
        exec(compile(source, "<module>", "exec"), namespace)
    except Exception as exc:
        return [("module", type(exc).__name__, list(namespace["T"]))]
    function = namespace["f"]
    observed: list[Any] = [("module", inspect.isgeneratorfunction(function), list(namespace["T"]))]
    for first in _INPUTS:
        for second in _INPUTS[:3]:
            namespace["T"].clear()
            arguments = (eval(first, namespace), eval(second, namespace))
            try:
                outcome = (_call(function, namespace, arguments), None)
            except Exception as exc:
                outcome = (None, type(exc).__name__)
            observed.append((first, second, outcome, list(namespace["T"])))
    return observed


# Random modules may `return` inside `finally`; that is valid, if unusual.
@pytest.mark.filterwarnings("ignore::SyntaxWarning")
def test_simplification_preserves_behaviour_of_random_modules() -> None:
    simplified = refused = 0
    for seed in _SEEDS:
        rnd = random.Random(seed)
        source = _Generator(rnd, rnd.choice(["def", "def", "gen", "async"])).module()
        try:
            compile(source, "<module>", "exec")
        except SyntaxError:
            continue
        try:
            result = simplify_constant_branches(source)
        except ConstantBranchError:
            refused += 1
            continue
        simplified += 1
        assert _behaviour(result) == _behaviour(source), (seed, source, result)
    assert simplified >= len(_SEEDS) * 0.8
    assert refused <= len(_SEEDS) * 0.05
