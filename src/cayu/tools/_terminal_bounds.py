"""Declared terminal payload bounds for Cayu's built-in tools.

A deferred tool round reserves the result size each of its tools declares in
``ToolSpec.max_terminal_payload_bytes`` from capacity shared by the whole
process. A tool without a declaration makes its round exclusive, so every
built-in declares one. The declaration is honest only if the tool keeps its
result inside it: ``bounded_terminal_result`` fits whatever ``run`` returns to
the declared bound, keeping as much of the result as fits and saying in the
result that it was cut.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from cayu._validation import DURABLE_EVENT_LIMITS, JsonUtf8SizeCounter, _json_string_body_size
from cayu.tools._redaction import active_secret_redactor_snapshot, record_ambiguous_secret_output
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.vaults import SecretRedactor

KIB = 1024
MIB = 1024 * KIB

# ToolSpec refuses a smaller declaration.
MIN_TOOL_TERMINAL_PAYLOAD_BYTES = 64 * KIB
# The terminal event carries the whole result beside up to 64 KiB of runtime
# envelope, and Cayu stores no event larger than the durable event limit.
MAX_TOOL_TERMINAL_PAYLOAD_BYTES = DURABLE_EVENT_LIMITS.max_bytes - 64 * KIB

# Defaults. Tools that return file text, command output, diffs or child results
# use the larger one; tools that return short records use the smaller one.
DEFAULT_TEXT_TOOL_TERMINAL_PAYLOAD_BYTES = 1 * MIB
DEFAULT_RECORD_TOOL_TERMINAL_PAYLOAD_BYTES = 256 * KIB

TERMINAL_PAYLOAD_TRUNCATION_FIELD = "terminal_payload_truncated"

# Room for the runtime's redaction placeholders beside the measured result.
TERMINAL_PAYLOAD_FIT_MARGIN_BYTES = 4 * KIB
# String fitting never cuts a value to fewer bytes than this, so identifiers,
# digests, statuses and paths survive intact.
_MIN_KEPT_STRING_BYTES = 512
# A reduced structured result keeps scalar fields up to this size, and names
# at most this many omitted fields with keys up to this size.
_KEPT_SCALAR_MAX_BYTES = 512
_KEPT_KEY_MAX_BYTES = 128
_MAX_LISTED_OMITTED_FIELDS = 64
_UNBOUNDED = 2**63 - 1

_ToolT = TypeVar("_ToolT", bound=Tool)
_RunMethod = Callable[[_ToolT, ToolContext, dict[str, Any]], Awaitable[ToolResult]]


def tool_terminal_payload_bound(value: object, *, default: int) -> int:
    """Validate an application's bound for a built-in tool, or return ``default``."""

    if value is None:
        return default
    if type(value) is not int:
        raise TypeError("max_terminal_payload_bytes must be an integer number of bytes.")
    if not MIN_TOOL_TERMINAL_PAYLOAD_BYTES <= value <= MAX_TOOL_TERMINAL_PAYLOAD_BYTES:
        raise ValueError(
            "max_terminal_payload_bytes must be between "
            f"{MIN_TOOL_TERMINAL_PAYLOAD_BYTES} and {MAX_TOOL_TERMINAL_PAYLOAD_BYTES} bytes."
        )
    return value


def spec_with_terminal_payload_bound(spec: ToolSpec, value: object, *, default: int) -> ToolSpec:
    """Return ``spec`` declaring the application's bound, its own, or ``default``.

    An explicit ``value`` always wins. Otherwise a bound the spec already
    declares is kept, and a spec without one gets ``default``.
    """

    if value is None and spec.max_terminal_payload_bytes is not None:
        return spec
    bound = tool_terminal_payload_bound(value, default=default)
    if spec.max_terminal_payload_bytes == bound:
        return spec
    return spec.model_copy(update={"max_terminal_payload_bytes": bound})


def json_payload_bytes(value: Any) -> int:
    """Return the compact UTF-8 JSON size Cayu measures for durable payloads."""

    counter = JsonUtf8SizeCounter(_UNBOUNDED, canonical_durable_numbers=True)
    if not counter.value(value) or counter.encountered_unsupported_value:
        raise ValueError("Tool result has no bounded durable JSON size.")
    return _UNBOUNDED - counter.remaining


def tool_result_payload_bytes(result: ToolResult) -> int:
    """Return the size the runtime compares with ``max_terminal_payload_bytes``."""

    return json_payload_bytes(result.model_dump(mode="json"))


def tool_result_fits(result: ToolResult, max_bytes: int | None) -> bool:
    """Whether ``fit_tool_result`` would return ``result`` unchanged."""

    if max_bytes is None:
        return True
    return tool_result_payload_bytes(result) <= max_bytes - TERMINAL_PAYLOAD_FIT_MARGIN_BYTES


def fit_tool_result(result: ToolResult, max_bytes: int | None) -> ToolResult:
    """Return ``result``, or the most of it that fits in ``max_bytes``.

    The longest strings are cut first, in ``content`` and inside ``structured``
    alike, each keeping its start and end around a visible marker, so every
    field and identifier stays in place. Only when that cannot fit, for example
    a listing with too many entries, ``structured`` is reduced to its short
    scalar fields. Either way ``is_error`` is kept, ``content`` ends with a
    notice, and ``structured`` names the original and maximum sizes under
    ``terminal_payload_truncated``.
    """

    if max_bytes is None:
        return result
    budget = max_bytes - TERMINAL_PAYLOAD_FIT_MARGIN_BYTES
    observed = tool_result_payload_bytes(result)
    if observed <= budget:
        return result
    note: dict[str, Any] = {"original_bytes": observed, "max_bytes": max_bytes}
    notice = (
        f"\n[Result truncated: the complete result was {observed} bytes and this tool "
        f"returns at most {max_bytes} bytes.]"
    )
    dumped = result.model_dump(mode="json")
    structured = dumped["structured"]
    artifacts = dumped["artifacts"]
    if artifacts and json_payload_bytes(artifacts) > budget // 4:
        note["artifacts_omitted"] = len(artifacts)
        artifacts = []
    fitted = _fit_strings(
        content=dumped["content"],
        structured=structured,
        artifacts=artifacts,
        is_error=result.is_error,
        note=note,
        notice=notice,
        budget=budget,
    )
    if fitted is not None:
        return fitted
    return _fit_reduced(
        content=dumped["content"],
        structured=structured,
        artifacts=artifacts,
        is_error=result.is_error,
        note=note,
        notice=notice,
        budget=budget,
    )


def fit_redacted_tool_result(
    ctx: ToolContext, result: ToolResult, max_bytes: int | None
) -> ToolResult:
    """Fit ``result`` after redacting the invocation's registered secrets.

    The size is judged on the redacted result, because the runtime redacts
    before it enforces the bound and a placeholder can be longer than the
    secret it replaces. A cut could otherwise split a secret so that the
    runtime's later redaction no longer recognizes, and so publishes, the part
    that remains. A cut result is bound to the registry revision it was
    redacted with.
    """

    if max_bytes is None:
        return result
    snapshot = active_secret_redactor_snapshot(ctx)
    redactor = snapshot.redactor
    redacted = redact_tool_result_with(redactor, result) if redactor.has_values else result
    if tool_result_fits(redacted, max_bytes):
        # Fits as the runtime will publish it; the runtime still redacts it,
        # so a tool's own framing and attested fields are left as they are.
        return result
    fitted = fit_tool_result(redacted, max_bytes)
    if fitted is not redacted:
        record_ambiguous_secret_output(ctx, snapshot)
    return fitted


def redact_tool_result_with(redactor: SecretRedactor, result: ToolResult) -> ToolResult:
    """Return ``result`` with ``redactor``'s secrets replaced everywhere."""

    dumped = result.model_dump(mode="json")
    return ToolResult(
        content=redactor.redact_text(dumped["content"]),
        structured=redactor.redact_json(dumped["structured"]),
        artifacts=redactor.redact_json(dumped["artifacts"]),
        is_error=result.is_error,
    )


def bounded_terminal_result(run: _RunMethod[_ToolT]) -> _RunMethod[_ToolT]:
    """Fit every result of a built-in tool's ``run`` to its declared bound."""

    @wraps(run)
    async def wrapper(self: _ToolT, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        result = await run(self, ctx, args)
        if type(result) is not ToolResult:
            return result
        return fit_redacted_tool_result(ctx, result, self.spec.max_terminal_payload_bytes)

    return wrapper


def json_string_body_bytes(text: str) -> int:
    """Escaped JSON size of ``text`` without its quotes."""

    return _json_string_body_size(text)


def cut_text_to_json_bytes(text: str, max_bytes: int) -> str:
    """Keep the start and end of ``text`` around a marker within about ``max_bytes``."""

    return _cut_string(text, _json_string_body_size(text), max_bytes)


def prefix_within_json_bytes(text: str, max_bytes: int) -> str:
    """Longest prefix of ``text`` whose escaped JSON string body fits in ``max_bytes``."""

    if max_bytes <= 0:
        return ""
    if _json_string_body_size(text) <= max_bytes:
        return text
    low, high = 0, min(len(text), max_bytes)
    while low < high:
        middle = (low + high + 1) // 2
        if _json_string_body_size(text[:middle]) <= max_bytes:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def _string_sizes(value: Any, sizes: list[int]) -> None:
    if type(value) is str:
        sizes.append(_json_string_body_size(value))
    elif type(value) is dict:
        for item in value.values():
            _string_sizes(item, sizes)
    elif type(value) is list:
        for item in value:
            _string_sizes(item, sizes)


def _cut_marker(omitted: int) -> str:
    return f"\n[... {omitted} bytes truncated ...]\n"


def _cut_string(value: str, size: int, cap: int) -> str:
    if size <= cap:
        return value
    head = prefix_within_json_bytes(value, cap // 2)
    rest = value[len(head) :]
    tail = prefix_within_json_bytes(rest[::-1], cap - _json_string_body_size(head))[::-1]
    omitted = size - _json_string_body_size(head) - _json_string_body_size(tail)
    return head + _cut_marker(omitted) + tail


def _cut_strings(value: Any, cap: int) -> Any:
    if type(value) is str:
        return _cut_string(value, _json_string_body_size(value), cap)
    if type(value) is dict:
        return {key: _cut_strings(item, cap) for key, item in value.items()}
    if type(value) is list:
        return [_cut_strings(item, cap) for item in value]
    return value


def _fit_strings(
    *,
    content: str,
    structured: dict[str, Any] | None,
    artifacts: list[Any],
    is_error: bool,
    note: dict[str, Any],
    notice: str,
    budget: int,
) -> ToolResult | None:
    """Cut the longest strings to one common length until the result fits."""

    noted = {**({} if structured is None else structured)}
    noted[TERMINAL_PAYLOAD_TRUNCATION_FIELD] = note
    sizes: list[int] = []
    _string_sizes(content, sizes)
    _string_sizes(structured, sizes)
    marker_bytes = _json_string_body_size(_cut_marker(10**12))
    whole = ToolResult(
        content=content + notice, structured=noted, artifacts=artifacts, is_error=is_error
    )
    excess = tool_result_payload_bytes(whole) - budget

    def removed(cap: int) -> int:
        return sum(size - cap - marker_bytes for size in sizes if size > cap)

    if not sizes or removed(_MIN_KEPT_STRING_BYTES) < excess:
        return None
    # The largest common cap that removes enough, found by bisection.
    low, high = _MIN_KEPT_STRING_BYTES, max(sizes)
    while low < high:
        middle = (low + high + 1) // 2
        if removed(middle) >= excess:
            low = middle
        else:
            high = middle - 1
    cap = low
    for _ in range(8):
        candidate = ToolResult(
            content=_cut_strings(content, cap) + notice,
            structured={
                **_cut_strings({} if structured is None else structured, cap),
                TERMINAL_PAYLOAD_TRUNCATION_FIELD: note,
            },
            artifacts=artifacts,
            is_error=is_error,
        )
        if tool_result_payload_bytes(candidate) <= budget:
            return candidate
        if cap == _MIN_KEPT_STRING_BYTES:
            return None
        cap = max(_MIN_KEPT_STRING_BYTES, cap * 3 // 4)
    return None


def _fit_reduced(
    *,
    content: str,
    structured: Any,
    artifacts: list[Any],
    is_error: bool,
    note: dict[str, Any],
    notice: str,
    budget: int,
) -> ToolResult:
    """Keep short scalar structured fields and as much content as fits."""

    # Kept fields, their keys and the omission list share a quarter of the
    # budget, so the rest is always left for content.
    kept_budget = budget // 4
    reduced: dict[str, Any] = {}
    omitted: list[str] = []
    for key, value in (structured or {}).items():
        cost = json_payload_bytes(key) + json_payload_bytes(value) + 2
        if (
            (value is None or type(value) in {bool, int, float, str})
            and cost <= _KEPT_SCALAR_MAX_BYTES + _KEPT_KEY_MAX_BYTES
            and cost <= kept_budget
        ):
            reduced[key] = value
            kept_budget -= cost
        elif len(omitted) < _MAX_LISTED_OMITTED_FIELDS and len(key) <= _KEPT_KEY_MAX_BYTES:
            omitted.append(key)
    if omitted:
        note = {**note, "structured_fields_omitted": omitted}
    reduced[TERMINAL_PAYLOAD_TRUNCATION_FIELD] = note
    shell = ToolResult(content=notice, structured=reduced, artifacts=artifacts, is_error=is_error)
    remaining = budget - tool_result_payload_bytes(shell)
    if remaining < 0:  # pragma: no cover - kept fields use at most a quarter of the budget.
        raise RuntimeError("A fitted tool result cannot hold its truncation notice.")
    size = _json_string_body_size(content)
    if size > remaining:
        content = _cut_string(content, size, max(0, remaining - 64))
    return ToolResult(
        content=content + notice, structured=reduced, artifacts=artifacts, is_error=is_error
    )
