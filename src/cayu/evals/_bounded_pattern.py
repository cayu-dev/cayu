"""Linear-time search for the regular-expression subset portable specs accept.

Corpus patterns are authority-free input evaluated inside eval workers. Python's
backtracking ``re`` can take exponential time on patterns such as ``(a+)+$``,
so a portable pattern compiles to a Thompson NFA that is simulated breadth-first:
one search costs at most the weighted program size times the text length.
Character-class members count toward that budget, including inside repeats.

Every accepted pattern means what it means to Python ``re.search``. The subset
is literals, ``.``, bracket classes with ranges and negation, ``\\d \\D \\w \\W
\\s \\S``, the control escapes ``\\n \\r \\t \\f \\v``, escaped ASCII
punctuation, ``^`` and ``$``, ``(...)`` and ``(?:...)`` groups, ``|``, and the
greedy or lazy quantifiers ``* + ? {m} {m,} {m,n}``. Backreferences, lookaround,
inline flags, word boundaries, and possessive quantifiers are rejected.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TypeAlias

BOUNDED_PATTERN_MAX_CHARS = 256
BOUNDED_PATTERN_MAX_INSTRUCTIONS = 512
_MAX_GROUP_DEPTH = 32
_MAX_REPEAT_DIGITS = 3

_CharTest: TypeAlias = Callable[[str], bool]
# Parse nodes: ("char", test), ("start",), ("end",), ("seq", nodes), ("alt", nodes),
# and ("repeat", node, minimum, maximum or None).
_Node: TypeAlias = tuple
_Instruction: TypeAlias = tuple

_CHAR, _SPLIT, _JUMP, _START, _END, _MATCH = range(6)
_CONTROL_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "f": "\f", "v": "\v"}
# Python's Unicode str semantics: \d is category Nd, \w is alphanumeric or "_".
_SHORTHAND_ESCAPES: dict[str, _CharTest] = {
    "d": str.isdecimal,
    "D": lambda character: not character.isdecimal(),
    "w": lambda character: character.isalnum() or character == "_",
    "W": lambda character: not (character.isalnum() or character == "_"),
    "s": str.isspace,
    "S": lambda character: not character.isspace(),
}


def _literal(expected: str) -> _CharTest:
    return expected.__eq__


def _not_newline(character: str) -> bool:
    return character != "\n"


class _Parser:
    __slots__ = ("_depth", "_index", "_pattern")

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern
        self._index = 0
        self._depth = 0

    def parse(self) -> _Node:
        node = self._alternation()
        if self._index != len(self._pattern):
            raise ValueError("Pattern has an unmatched ')'.")
        return node

    def _peek(self, offset: int = 0) -> str | None:
        index = self._index + offset
        return self._pattern[index] if index < len(self._pattern) else None

    def _take(self) -> str:
        character = self._peek()
        if character is None:
            raise ValueError("Pattern ends unexpectedly.")
        self._index += 1
        return character

    def _alternation(self) -> _Node:
        branches = [self._sequence()]
        while self._peek() == "|":
            self._index += 1
            branches.append(self._sequence())
        return branches[0] if len(branches) == 1 else ("alt", tuple(branches))

    def _sequence(self) -> _Node:
        items: list[_Node] = []
        while (character := self._peek()) is not None and character not in "|)":
            items.append(self._repeat())
        return ("seq", tuple(items))

    def _repeat(self) -> _Node:
        atom = self._atom()
        character = self._peek()
        if character is None or character not in "*+?{":
            return atom
        if atom[0] in {"start", "end"}:
            raise ValueError("Pattern anchors cannot be repeated.")
        self._index += 1
        if character == "*":
            bounds: tuple[int, int | None] = (0, None)
        elif character == "+":
            bounds = (1, None)
        elif character == "?":
            bounds = (0, 1)
        else:
            bounds = self._counted_bounds()
        # Laziness changes which match is reported, never whether one exists.
        if self._peek() == "?":
            self._index += 1
        if (following := self._peek()) is not None and following in "*+?{":
            raise ValueError("Pattern quantifiers cannot be repeated or possessive.")
        return ("repeat", atom, *bounds)

    def _counted_bounds(self) -> tuple[int, int | None]:
        minimum = self._count()
        maximum: int | None = minimum
        if self._peek() == ",":
            self._index += 1
            maximum = self._count()
        if minimum is None or self._peek() != "}":
            raise ValueError("Pattern repeat counts must use {m}, {m,}, or {m,n}.")
        self._index += 1
        if maximum is not None and maximum < minimum:
            raise ValueError("Pattern repeat maximum is below its minimum.")
        return minimum, maximum

    def _count(self) -> int | None:
        start = self._index
        while (character := self._peek()) is not None and character in "0123456789":
            self._index += 1
        digits = self._pattern[start : self._index]
        if len(digits) > _MAX_REPEAT_DIGITS:
            raise ValueError(f"Pattern repeat counts must be at most {_MAX_REPEAT_DIGITS} digits.")
        return int(digits) if digits else None

    def _atom(self) -> _Node:
        character = self._take()
        if character == "(":
            if self._peek() == "?":
                if self._peek(1) != ":":
                    raise ValueError("Pattern groups support only (...) and (?:...).")
                self._index += 2
            self._depth += 1
            if self._depth > _MAX_GROUP_DEPTH:
                raise ValueError(f"Pattern groups nest deeper than {_MAX_GROUP_DEPTH} levels.")
            node = self._alternation()
            if self._peek() != ")":
                raise ValueError("Pattern has an unterminated group.")
            self._index += 1
            self._depth -= 1
            return node
        if character == "[":
            test, cost = self._bracket_class()
            return ("char", test, cost)
        if character == ".":
            return ("char", _not_newline)
        if character == "^":
            return ("start",)
        if character == "$":
            return ("end",)
        if character == "\\":
            shorthand = self._shorthand()
            return ("char", shorthand or _literal(self._escaped_literal()))
        if character in "*+?{":
            raise ValueError("Pattern quantifier has nothing to repeat.")
        return ("char", _literal(character))

    def _shorthand(self) -> _CharTest | None:
        """Consume a shorthand class escape after a backslash, if one follows."""

        test = _SHORTHAND_ESCAPES.get(self._peek() or "")
        if test is not None:
            self._index += 1
        return test

    def _escaped_literal(self) -> str:
        """Consume the character a backslash escapes and return its literal value."""

        character = self._peek()
        if character is None:
            raise ValueError("Pattern ends with a dangling backslash.")
        self._index += 1
        if character in _CONTROL_ESCAPES:
            return _CONTROL_ESCAPES[character]
        if character.isascii() and not character.isalnum():
            return character
        raise ValueError(f"Pattern escape \\{character} is not supported.")

    def _class_member(self) -> str | _CharTest:
        character = self._take()
        if character != "\\":
            return character
        return self._shorthand() or self._escaped_literal()

    def _bracket_class(self) -> tuple[_CharTest, int]:
        negated = self._peek() == "^"
        if negated:
            self._index += 1
        members: set[str] = set()
        ranges: list[tuple[str, str]] = []
        tests: list[_CharTest] = []
        first = True
        while True:
            if self._peek() is None:
                raise ValueError("Pattern has an unterminated character class.")
            # A "]" immediately after "[" or "[^" is a member, which can start a range, as
            # in Python.
            if self._peek() == "]" and not first:
                self._index += 1
                break
            first = False
            low = self._class_member()
            if self._peek() != "-" or self._peek(1) in {"]", None}:
                if isinstance(low, str):
                    members.add(low)
                else:
                    tests.append(low)
                continue
            self._index += 1
            high = self._class_member()
            if not isinstance(low, str) or not isinstance(high, str):
                raise ValueError("Pattern class ranges cannot use shorthand escapes.")
            if high < low:
                raise ValueError("Pattern class range is out of order.")
            ranges.append((low, high))
        frozen_members = frozenset(members)
        frozen_ranges = tuple(ranges)
        frozen_tests = tuple(tests)

        def test(character: str) -> bool:
            found = (
                character in frozen_members
                or any(low <= character <= high for low, high in frozen_ranges)
                or any(member(character) for member in frozen_tests)
            )
            return found is not negated

        return test, 1 + len(frozen_ranges) + len(frozen_tests)


def _program_size(node: _Node) -> int:
    kind = node[0]
    if kind == "char":
        return node[2] if len(node) > 2 else 1
    if kind in {"start", "end"}:
        return 1
    if kind == "seq":
        return sum(_program_size(item) for item in node[1])
    if kind == "alt":
        return sum(_program_size(item) for item in node[1]) + 2 * (len(node[1]) - 1)
    body = _program_size(node[1])
    if body == 0:
        # `_emit` emits nothing for a repeated body that matches only the empty string.
        return 0
    minimum, maximum = node[2], node[3]
    optional = body + 2 if maximum is None else (body + 1) * (maximum - minimum)
    return body * minimum + optional


def _emit(node: _Node, program: list[_Instruction]) -> None:
    kind = node[0]
    if kind == "char":
        program.append((_CHAR, node[1]))
    elif kind == "start":
        program.append((_START,))
    elif kind == "end":
        program.append((_END,))
    elif kind == "seq":
        for item in node[1]:
            _emit(item, program)
    elif kind == "alt":
        jumps: list[int] = []
        *leading, last = node[1]
        for branch in leading:
            split = len(program)
            program.append((_SPLIT, 0, 0))
            _emit(branch, program)
            jumps.append(len(program))
            program.append((_JUMP, 0))
            program[split] = (_SPLIT, split + 1, len(program))
        _emit(last, program)
        for jump in jumps:
            program[jump] = (_JUMP, len(program))
    else:
        body, minimum, maximum = node[1], node[2], node[3]
        if _program_size(body) == 0:
            # A body with no instructions (an empty group, or x{0}) matches only the empty
            # string, and so does any repetition of it. Expanding it would cost its repeat
            # counts multiplied across nesting levels while emitting nothing.
            return
        for _ in range(minimum):
            _emit(body, program)
        if maximum is None:
            loop = len(program)
            program.append((_SPLIT, 0, 0))
            _emit(body, program)
            program.append((_JUMP, loop))
            program[loop] = (_SPLIT, loop + 1, len(program))
        else:
            for _ in range(maximum - minimum):
                split = len(program)
                program.append((_SPLIT, 0, 0))
                _emit(body, program)
                program[split] = (_SPLIT, split + 1, len(program))


class BoundedPattern:
    """One validated pattern whose search time is linear in the searched text."""

    __slots__ = ("_program", "pattern")

    def __init__(self, pattern: str) -> None:
        if type(pattern) is not str:
            raise TypeError("pattern must be a string.")
        if not pattern:
            raise ValueError("Pattern must not be empty.")
        if len(pattern) > BOUNDED_PATTERN_MAX_CHARS:
            raise ValueError(f"Pattern must be at most {BOUNDED_PATTERN_MAX_CHARS} characters.")
        node = _Parser(pattern).parse()
        if _program_size(node) + 1 > BOUNDED_PATTERN_MAX_INSTRUCTIONS:
            raise ValueError(
                "Pattern expands beyond the "
                f"{BOUNDED_PATTERN_MAX_INSTRUCTIONS}-instruction matching limit."
            )
        program: list[_Instruction] = []
        _emit(node, program)
        program.append((_MATCH,))
        self.pattern = pattern
        self._program = tuple(program)

    def search(self, text: str) -> bool:
        """Whether the pattern matches anywhere in ``text``, as ``re.search`` would."""

        program = self._program
        length = len(text)
        advanced: list[int] = []
        for position in range(length + 1):
            waiting: list[int] = []
            seen: set[int] = set()
            # A fresh thread at every position gives search, not match, semantics.
            pending = [0, *reversed(advanced)]
            while pending:
                index = pending.pop()
                if index in seen:
                    continue
                seen.add(index)
                instruction = program[index]
                code = instruction[0]
                if code == _MATCH:
                    return True
                if code == _CHAR:
                    waiting.append(index)
                elif code == _SPLIT:
                    pending.append(instruction[2])
                    pending.append(instruction[1])
                elif code == _JUMP:
                    pending.append(instruction[1])
                elif code == _START:
                    if position == 0:
                        pending.append(index + 1)
                elif position == length or (position == length - 1 and text[position] == "\n"):
                    # Python's "$" also matches before one final newline.
                    pending.append(index + 1)
            if position == length:
                break
            character = text[position]
            advanced = [index + 1 for index in waiting if program[index][1](character)]
        return False


@lru_cache(maxsize=256)
def compile_bounded_pattern(pattern: str) -> BoundedPattern:
    """Compile one portable pattern, reusing recent compilations."""

    return BoundedPattern(pattern)
