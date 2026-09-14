"""Bounded native-record snapshots shared by closure inventory and export.

Store implementations feed records while holding their read snapshot. Conversion
walks the owned fields without first calling a model serializer or allocating an
unbounded intermediate representation.
"""

from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel

from cayu._validation import FrozenJsonDict, FrozenJsonList, canonical_durable_json_bytes
from cayu.messages import _ValidatedContent

SESSION_CLOSURE_NATIVE_CLASSES = (
    "session",
    "labels",
    "metadata",
    "events",
    "transcript",
    "checkpoint",
    "queued_messages",
    "session_operations",
    "event_side_effect_deliveries",
    "queue_deliveries",
    "deferred_interaction_inputs",
    "targeted_tool_grants",
    "targeted_tool_grant_uses",
    "recall_receipts",
    "context_exposures",
    "recall_item_exposures",
)


def require_terminal_protected_effect(
    session_id: str, instance_id: str, key: str, raw: object
) -> None:
    """Require positive settled evidence, not merely a terminal session status."""
    from cayu._validation import copy_durable_record
    from cayu.runtime._tool_effect_state import ToolEffectRecord, effect_storage_key

    try:
        value = copy_durable_record(raw, "closure protected effect")
        record = ToolEffectRecord.model_validate(value)
        if (
            record.intent.session_id != session_id
            or record.intent.session_instance_id != instance_id
            or effect_storage_key(record.intent) != key
            # ToolEffectRecord requires terminal material exactly for settled states.
            or record.terminal is None
        ):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("Session closure requires settled protected tool effects.") from None


def validate_closure_records(raw: object, *, max_records: int, max_bytes: int) -> dict[str, Any]:
    """Authenticate bounded native enumeration, including every declared class."""
    if type(raw) is not dict:
        raise ValueError("Invalid session closure record inventory.")
    value = cast("dict[str, Any]", raw)
    if (
        set(value) != {"schema_version", "records", "counts", "record_bytes"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or any(type(value[key]) is not dict for key in ("records", "counts", "record_bytes"))
        or any(
            set(value[key]) != set(SESSION_CLOSURE_NATIVE_CLASSES)
            for key in ("records", "counts", "record_bytes")
        )
    ):
        raise ValueError("Invalid session closure record inventory.")
    builder = ClosureRecordsBuilder(max_records=max_records, max_bytes=max_bytes)
    for name in SESSION_CLOSURE_NATIVE_CLASSES:
        rows = value["records"][name]
        if type(rows) is not list:
            raise ValueError("Session closure records must be a list.")
        builder.add_class(name, rows)
        if (
            type(value["counts"][name]) is not int
            or value["counts"][name] != len(rows)
            or type(value["record_bytes"][name]) is not int
            or value["record_bytes"][name] != builder.class_bytes[name]
        ):
            raise ValueError("Session closure record counts or bytes conflict.")
    return builder.finish()


class ClosureRecordsTooLarge(ValueError):
    def __init__(self) -> None:
        super().__init__("Session closure record enumeration exceeds its bounds.")


class ClosureRecordsBuilder:
    """Own one bounded, detached collection; never return a partial source row."""

    def __init__(self, *, max_records: int, max_bytes: int) -> None:
        if type(max_records) is not int or not 0 < max_records <= 100_000:
            raise ValueError("Invalid closure record limit.")
        if type(max_bytes) is not int or not 0 < max_bytes <= 256 * 1024 * 1024:
            raise ValueError("Invalid closure byte limit.")
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.count = 0
        self.size = 0
        self.records: dict[str, list[Any]] = {}
        self.class_bytes: dict[str, int] = {}
        self._active_ids: set[int] = set()
        self._nodes = 0
        self._failed = False

    def _charge(self, size: int) -> None:
        if self.size + size > self.max_bytes:
            raise ClosureRecordsTooLarge()
        self.size += size

    def _copy(self, value: object, depth: int = 0) -> Any:
        self._nodes += 1
        if self._nodes > 1_000_000:
            raise ClosureRecordsTooLarge()
        if depth > 64:
            raise ValueError("Session closure record nesting exceeds its bound.")
        if isinstance(value, Enum):
            value = value.value
        elif type(value) is datetime:
            value = value.isoformat()
        if value is None or type(value) in (bool, int, float, str):
            encoded = canonical_durable_json_bytes(
                value,
                "session_closure.record",
                max_bytes=self.max_bytes - self.size,
                max_nodes=1_000_000,
            )
            self._charge(len(encoded))
            return value
        if id(value) in self._active_ids:
            raise ValueError("Session closure record contains a cycle.")
        self._active_ids.add(id(value))
        try:
            if isinstance(value, BaseModel):
                entries: Iterable[tuple[str, object]] = (
                    (field.serialization_alias or field.alias or name, getattr(value, name))
                    for name, field in type(value).model_fields.items()
                    if field.exclude is not True
                )
            elif is_dataclass(value) and not isinstance(value, type):
                entries = ((field.name, getattr(value, field.name)) for field in fields(value))
            elif type(value) in (dict, FrozenJsonDict):
                entries = cast("Mapping[str, object]", value).items()
            elif type(value) in (list, tuple, FrozenJsonList, _ValidatedContent):
                self._charge(2)
                result = []
                for index, item in enumerate(cast("Iterable[object]", value)):
                    if index:
                        self._charge(1)
                    result.append(self._copy(item, depth + 1))
                return result
            else:
                raise TypeError("Session closure record contains an unsupported value.")
            self._charge(2)
            copied: dict[str, Any] = {}
            for index, (key, item) in enumerate(entries):
                if type(key) is not str:
                    raise TypeError("Session closure record keys must be strings.")
                if key in copied:
                    raise ValueError("Session closure record has conflicting field aliases.")
                if index:
                    self._charge(1)
                self._copy(key, depth + 1)
                self._charge(1)
                copied[key] = self._copy(item, depth + 1)
            return copied
        finally:
            self._active_ids.remove(id(value))

    def add_class(self, record_class: str, values: Iterable[object]) -> None:
        if self._failed:
            raise ValueError("Session closure record collection has already failed.")
        if record_class in self.records:
            raise ValueError("Duplicate closure record class.")
        rows: list[Any] = []
        self.records[record_class] = rows
        self.class_bytes[record_class] = 0
        try:
            for value in values:
                self.add_record(record_class, value)
        except BaseException:
            self._failed = True
            raise

    def add_record(self, record_class: str, value: object) -> None:
        if self._failed:
            raise ValueError("Session closure record collection has already failed.")
        try:
            if self.count == self.max_records:
                raise ClosureRecordsTooLarge()
            previous_size = self.size
            row = self._copy(value)
            self.records[record_class].append(row)
            self.count += 1
            self.class_bytes[record_class] += self.size - previous_size
        except BaseException:
            self._failed = True
            raise

    def finish(self) -> dict[str, Any]:
        if self._failed:
            raise ValueError("Session closure record collection has already failed.")
        document = {
            "schema_version": 1,
            "records": self.records,
            "counts": {name: len(rows) for name, rows in self.records.items()},
            "record_bytes": self.class_bytes,
        }
        canonical_durable_json_bytes(
            document,
            "session_closure.records",
            max_bytes=self.max_bytes,
            max_nodes=1_000_000,
        )
        return document
