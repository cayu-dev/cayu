"""Bounded offset structure; never retain annotation or associated text."""

from collections.abc import Mapping
from dataclasses import dataclass

_KINDS: dict[type, str] = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}
_CONDITIONS = frozenset(
    {
        "missing_endpoint",
        "non_integer_endpoint",
        "negative_start",
        "empty_range",
        "reversed_range",
        "end_out_of_bounds",
    }
)
_LIMIT = 2**31 - 1


@dataclass(frozen=True)
class CitationOffsetDiagnostic:
    condition: str
    start_kind: str
    end_kind: str
    start_index: int | None
    end_index: int | None
    text_length: int
    text_offset: int

    @classmethod
    def invalid_offsets(
        cls, annotation: Mapping[str, object], *, text_length: int, text_offset: int
    ) -> "CitationOffsetDiagnostic | None":
        start, end = annotation.get("start_index"), annotation.get("end_index")
        # Preserve the adapter's existing optional-pair contract.
        if start is None and end is None:
            return None
        if start is None or end is None:
            condition = "missing_endpoint"
        elif type(start) is not int or type(end) is not int:
            condition = "non_integer_endpoint"
        elif start < 0:
            condition = "negative_start"
        elif end == start:
            condition = "empty_range"
        elif end < start:
            condition = "reversed_range"
        elif end > text_length:
            condition = "end_out_of_bounds"
        else:
            return None
        return cls(
            condition,
            _KINDS.get(type(start), "non_json") if "start_index" in annotation else "absent",
            _KINDS.get(type(end), "non_json") if "end_index" in annotation else "absent",
            start if type(start) is int and -_LIMIT <= start <= _LIMIT else None,
            end if type(end) is int and -_LIMIT <= end <= _LIMIT else None,
            min(text_length, _LIMIT),
            min(text_offset, _LIMIT),
        )


def citation_offset_fields(diagnostic: object) -> dict[str, str | int]:
    """Revalidate attributes at every public projection boundary."""
    if type(diagnostic) is not CitationOffsetDiagnostic:
        return {}
    if type(diagnostic.condition) is not str or diagnostic.condition not in _CONDITIONS:
        return {}
    kinds = (*_KINDS.values(), "absent", "non_json")
    if any(
        type(k) is not str or k not in kinds for k in (diagnostic.start_kind, diagnostic.end_kind)
    ):
        return {}
    if any(
        type(n) is not int or not 0 <= n <= _LIMIT
        for n in (diagnostic.text_length, diagnostic.text_offset)
    ):
        return {}
    if any(
        n is not None and (type(n) is not int or not -_LIMIT <= n <= _LIMIT)
        for n in (diagnostic.start_index, diagnostic.end_index)
    ):
        return {}
    fields: dict[str, str | int] = {
        "condition": diagnostic.condition,
        "start_kind": diagnostic.start_kind,
        "end_kind": diagnostic.end_kind,
        # Lengths are capped; _LIMIT means at least that many code points.
        "text_length": diagnostic.text_length,
        "text_offset": diagnostic.text_offset,
    }
    for endpoint in ("start", "end"):
        value = getattr(diagnostic, f"{endpoint}_index")
        if type(value) is int and -_LIMIT <= value <= _LIMIT:
            fields[f"{endpoint}_index"] = value
        elif getattr(diagnostic, f"{endpoint}_kind") == "integer":
            fields[f"{endpoint}_index_status"] = "outside_diagnostic_bounds"
    return {f"provider_protocol_citation_{key}": value for key, value in fields.items()}
