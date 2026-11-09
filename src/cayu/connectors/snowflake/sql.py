"""What the Snowflake toolkit reads from a statement's text before sending it: whether it is one
statement, and whether it is a read.

The check is lexical: comments, strings, quoted identifiers and ``$$`` bodies are skipped, and
only the statement's words and parentheses are looked at. The Snowflake role a connection uses
stays the real limit on what a statement may do; this check keeps writes behind
``snowflake_execute`` and its approval.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The keywords a read statement may start with. ``WITH`` is a read only when its common table
#: expressions are followed by ``SELECT`` and it defines no procedure or function.
READ_KEYWORDS = frozenset({"SELECT", "SHOW", "DESCRIBE", "DESC", "WITH"})
_PLAIN_READS = READ_KEYWORDS - {"WITH"}
_ROUTINES = frozenset({"PROCEDURE", "FUNCTION"})
#: Snowflake's system functions, several of which change state (``SYSTEM$CANCEL_ALL_QUERIES``).
_SYSTEM_FUNCTION = "SYSTEM$"
#: The table functions a read may call in ``TABLE(...)``, besides ``INFORMATION_SCHEMA``'s:
#: ``TABLE(my_procedure())`` could run a procedure that writes.
READ_TABLE_FUNCTIONS = frozenset(
    {"RESULT_SCAN", "FLATTEN", "GENERATOR", "SPLIT_TO_TABLE", "INFER_SCHEMA"}
)


class StatementShapeError(ValueError):
    """A statement this toolkit doesn't send: empty, or several."""


@dataclass(frozen=True)
class _Token:
    #: ``word``, ``quoted`` (a quoted identifier), ``string``, ``flow`` (the ``->>`` operator),
    #: ``(``, ``)``, ``,``, ``.`` or ``other``.
    kind: str
    #: A word's or quoted identifier's text, uppercased; empty for the other kinds.
    text: str = ""


def is_read_statement(statement: str) -> bool:
    """Whether one statement is a read.

    It is when it starts with ``SELECT``, ``SHOW``, ``DESCRIBE`` or ``DESC``, or with ``WITH``
    whose common table expressions are followed by ``SELECT`` and which defines no procedure or
    function; and it chains no statement with the flow operator ``->>``, names no ``SYSTEM$``
    function, uses no ``IDENTIFIER(...)``, and calls in ``TABLE(...)`` only
    ``READ_TABLE_FUNCTIONS`` or an ``INFORMATION_SCHEMA`` function. Raises
    ``StatementShapeError`` for an empty statement or several.
    """

    tokens = statement_tokens(statement)
    words = {token.text for token in tokens if token.kind == "word"}
    names = {token.text for token in tokens if token.kind in ("word", "quoted")}
    if any(_SYSTEM_FUNCTION in name for name in names) or not _calls_read_only(tokens):
        return False
    # The flow operator chains statements into one: what follows it can write.
    if any(token.kind == "flow" for token in tokens):
        return False
    keyword = first_keyword(tokens)
    if keyword in _PLAIN_READS:
        return True
    if keyword != "WITH" or words & _ROUTINES:
        return False
    items = _top_level(tokens)
    # A WITH inside opening parentheses is refused: only a top-level one is read.
    if not items or not _is_word(items[0], "WITH"):
        return False
    return _selects_after_ctes(items[1:])


def statement_tokens(statement: str) -> list[_Token]:
    """The statement's tokens. Raises ``StatementShapeError`` for an empty statement or
    several."""

    tokens, several = _tokens(statement)
    if not tokens:
        raise StatementShapeError("The statement is empty.")
    if several:
        raise StatementShapeError("Send one statement per call.")
    return tokens


def first_keyword(tokens: list[_Token]) -> str | None:
    """The first word, after opening parentheses; None when something else comes first."""

    for token in tokens:
        if token.kind == "word":
            return token.text
        if token.kind != "(":
            return None
    return None


def _is_word(token: _Token, text: str) -> bool:
    return token.kind == "word" and token.text == text


def _calls_read_only(tokens: list[_Token]) -> bool:
    """Whether every ``TABLE(...)`` calls a read-only table function, and no ``IDENTIFIER(...)``
    names something the check can't read."""

    for index, token in enumerate(tokens[:-1]):
        if tokens[index + 1].kind != "(":
            continue
        if _is_word(token, "IDENTIFIER"):
            return False
        if _is_word(token, "TABLE") and not _read_table_function(tokens[index + 2 :]):
            return False
    return True


def _read_table_function(tokens: list[_Token]) -> bool:
    """Whether ``tokens`` start with a call to a read-only table function: one of
    ``READ_TABLE_FUNCTIONS``, or ``[database.]INFORMATION_SCHEMA.<function>``."""

    parts: list[str] = []
    index = 0
    while index < len(tokens) and tokens[index].kind in ("word", "quoted"):
        parts.append(tokens[index].text)
        index += 1
        if index < len(tokens) and tokens[index].kind == ".":
            index += 1
        else:
            break
    if not parts or index >= len(tokens) or tokens[index].kind != "(":
        return False
    if len(parts) == 1:
        return parts[0] in READ_TABLE_FUNCTIONS
    return parts[-2] == "INFORMATION_SCHEMA"


def _identifier_character(character: str) -> bool:
    return character.isalnum() or character in "_$"


def _tokens(sql: str) -> tuple[list[_Token], bool]:
    """The first statement's tokens, and whether anything but comments follows its ``;``."""

    tokens: list[_Token] = []
    ended = False
    index, length = 0, len(sql)
    while index < length:
        character = sql[index]
        if character.isspace():
            index += 1
            continue
        if sql.startswith(("--", "//"), index):
            end = sql.find("\n", index)
            index = length if end < 0 else end + 1
            continue
        if sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            index = length if end < 0 else end + 2
            continue
        if ended:
            return tokens, True
        if character == ";":
            ended = True
            index += 1
        elif character == "'":
            tokens.append(_Token("string"))
            index = _after_quoted(sql, index)
        elif character == '"':
            end = _after_quoted(sql, index)
            name = sql[index + 1 : end - 1].replace('""', '"')
            tokens.append(_Token("quoted", name.upper()))
            index = end
        elif sql.startswith("$$", index) and (
            index == 0 or not _identifier_character(sql[index - 1])
        ):
            tokens.append(_Token("string"))
            end = sql.find("$$", index + 2)
            index = length if end < 0 else end + 2
        elif sql.startswith("->>", index):
            tokens.append(_Token("flow"))
            index += 3
        elif character.isalpha() or character == "_":
            end = index
            while end < length and _identifier_character(sql[end]):
                end += 1
            tokens.append(_Token("word", sql[index:end].upper()))
            index = end
        else:
            tokens.append(_Token(character if character in "(),." else "other"))
            index += 1
    return tokens, False


def _after_quoted(sql: str, start: int) -> int:
    """The index after the string or quoted identifier opening at ``start``."""

    quote = sql[start]
    index = start + 1
    while index < len(sql):
        if quote == "'" and sql[index] == "\\":
            index += 2
        elif sql[index] == quote:
            if sql.startswith(quote * 2, index):
                index += 2
            else:
                return index + 1
        else:
            index += 1
    return len(sql)


def _top_level(tokens: list[_Token]) -> list[_Token] | None:
    """The tokens outside parentheses, each parenthesized group as one ``group`` token; None
    when the parentheses don't balance."""

    items: list[_Token] = []
    depth = 0
    for token in tokens:
        if token.kind == "(":
            if depth == 0:
                items.append(_Token("group"))
            depth += 1
        elif token.kind == ")":
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0:
            items.append(token)
    return items if depth == 0 else None


def _selects_after_ctes(items: list[_Token]) -> bool:
    """Whether ``[RECURSIVE] name [(columns)] AS (query) [, ...]`` is followed by ``SELECT``."""

    index = 1 if items and _is_word(items[0], "RECURSIVE") else 0

    def at(position: int) -> _Token:
        return items[position] if position < len(items) else _Token("end")

    while True:
        if at(index).kind not in ("word", "quoted") or _is_word(at(index), "AS"):
            return False
        index += 1
        if at(index).kind == "group":
            index += 1
        if not _is_word(at(index), "AS") or at(index + 1).kind != "group":
            return False
        index += 2
        if at(index).kind != ",":
            return _is_word(at(index), "SELECT")
        index += 1
