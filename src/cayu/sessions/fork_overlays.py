"""Controller-attached fork overlays: per-fork instructions, tools and memory.

An overlay is part of one fork's request identity. The runtime stores the
admitted overlay as runtime-owned child metadata and renders its instructions
and memory as a separate model-visible layer after the copied source history.
Neither the source session, its registered body, nor the host's knowledge or
memory stores are changed. Overlay tools are only names; their schemas and
executables come from the host's enrollment of the controller.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
    require_durable_nonblank,
)
from cayu.messages import Message, MessageRole

if TYPE_CHECKING:
    from cayu.sessions.records import Session

FORK_OVERLAY_METADATA_KEY = "cayu:fork_overlay"
FORK_OVERLAY_SCHEMA_VERSION = 1
FORK_OVERLAY_CONTROLLER_MAX_CHARS = 128
FORK_OVERLAY_INSTRUCTIONS_MAX_CHARS = 32_000
FORK_OVERLAY_MAX_TOOL_GRANTS = 32
FORK_OVERLAY_TOOL_NAME_MAX_CHARS = 128
FORK_OVERLAY_MAX_MEMORY_ITEMS = 32
FORK_OVERLAY_MEMORY_ID_MAX_CHARS = 128
FORK_OVERLAY_MEMORY_CONTENT_MAX_CHARS = 8_000
FORK_OVERLAY_MEMORY_SOURCE_MAX_CHARS = 256
# The overlay layer is appended after context selection, so its total text is
# bounded tightly instead of being fitted with the selected history.
FORK_OVERLAY_CONTEXT_MAX_CHARS = 32_000
FORK_CONTROLLER_AGENT_SEPARATOR = "@fork-controller:"
_CONTEXT_OPEN_MARKER = '<cayu_fork_overlay version="1">'
_CONTEXT_CLOSE_MARKER = "</cayu_fork_overlay>"


def _bounded(value: str, field_name: str, max_chars: int) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > max_chars:
        raise ValueError(f"{field_name} cannot exceed {max_chars} characters.")
    return value


def _bounded_text(value: str, field_name: str, max_chars: int) -> str:
    value = require_durable_nonblank(value, field_name)
    if len(value) > max_chars:
        raise ValueError(f"{field_name} cannot exceed {max_chars} characters.")
    return value


def validate_fork_controller_name(value: str, field_name: str = "controller") -> str:
    """Validate one enrolled controller name as it appears in overlays and agent names."""

    value = _bounded(value, field_name, FORK_OVERLAY_CONTROLLER_MAX_CHARS)
    if FORK_CONTROLLER_AGENT_SEPARATOR in value:
        raise ValueError(f"{field_name} cannot contain {FORK_CONTROLLER_AGENT_SEPARATOR!r}.")
    return value


def fork_controller_agent_name(agent_name: str, controller: str) -> str:
    """Return the host body that carries one enrolled controller's served tools."""

    agent_name = require_durable_clean_nonblank(agent_name, "agent_name")
    if FORK_CONTROLLER_AGENT_SEPARATOR in agent_name:
        raise ValueError("A fork-controller body cannot be enrolled for another controller.")
    return (
        f"{agent_name}{FORK_CONTROLLER_AGENT_SEPARATOR}{validate_fork_controller_name(controller)}"
    )


class ForkOverlayToolGrant(BaseModel):
    """Typed request for one tool the enrolled controller serves.

    The grant names a tool. It carries no schema, executable, credential or
    policy; the runtime resolves all of those from the host's enrollment.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    tool_name: str

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        return _bounded(value, "tool_name", FORK_OVERLAY_TOOL_NAME_MAX_CHARS)


class ForkOverlayMemoryItem(BaseModel):
    """One controller-supplied context item visible to a single fork only."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    content: str
    source: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _bounded(value, "id", FORK_OVERLAY_MEMORY_ID_MAX_CHARS)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return _bounded_text(value, "content", FORK_OVERLAY_MEMORY_CONTENT_MAX_CHARS)

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded(value, "source", FORK_OVERLAY_MEMORY_SOURCE_MAX_CHARS)


class ForkOverlay(BaseModel):
    """Optional controller layer for one fork; every part is independently optional.

    ``instructions`` and ``memory`` reach the model as one layer after the
    copied source history. ``tools`` admits controller-served tools next to the
    narrowed host ceiling. Budget, stop conditions and result shape use the
    fork's ``initial_invocation`` or a later ``ResumeRequest``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    controller: str
    instructions: str | None = None
    tools: tuple[ForkOverlayToolGrant, ...] = Field(
        default=(), max_length=FORK_OVERLAY_MAX_TOOL_GRANTS
    )
    memory: tuple[ForkOverlayMemoryItem, ...] = Field(
        default=(), max_length=FORK_OVERLAY_MAX_MEMORY_ITEMS
    )

    @field_validator("controller")
    @classmethod
    def validate_controller(cls, value: str) -> str:
        return validate_fork_controller_name(value)

    @field_validator("instructions")
    @classmethod
    def validate_instructions(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, "instructions", FORK_OVERLAY_INSTRUCTIONS_MAX_CHARS)

    @model_validator(mode="after")
    def validate_overlay(self) -> ForkOverlay:
        if self.instructions is None and not self.tools and not self.memory:
            raise ValueError("A fork overlay requires instructions, tools, or memory.")
        tool_names = [grant.tool_name for grant in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Fork overlay tools must name each tool once.")
        memory_ids = [item.id for item in self.memory]
        if len(memory_ids) != len(set(memory_ids)):
            raise ValueError("Fork overlay memory items must have unique ids.")
        context_chars = len(self.instructions or "") + sum(
            len(item.content) for item in self.memory
        )
        if context_chars > FORK_OVERLAY_CONTEXT_MAX_CHARS:
            raise ValueError(
                "Fork overlay instructions and memory together exceed "
                f"{FORK_OVERLAY_CONTEXT_MAX_CHARS} characters."
            )
        return self

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(grant.tool_name for grant in self.tools)

    @property
    def fingerprint(self) -> str:
        """Content identity bound into the fork request and child lineage."""

        return fork_overlay_fingerprint(self)


def copy_fork_overlay(value: object) -> ForkOverlay | None:
    if value is None:
        return None
    if isinstance(value, ForkOverlay):
        value = value.model_dump(mode="json")
    return ForkOverlay.model_validate(value)


def fork_overlay_fingerprint(overlay: ForkOverlay) -> str:
    if type(overlay) is not ForkOverlay:
        raise TypeError("overlay must be a ForkOverlay.")
    material = {
        "record_type": "cayu.fork-overlay",
        "schema_version": FORK_OVERLAY_SCHEMA_VERSION,
        "overlay": overlay.model_dump(mode="json"),
    }
    return "sha256:" + sha256(canonical_durable_json_bytes(material, "fork_overlay")).hexdigest()


class ForkOverlayRecord(BaseModel):
    """Runtime-owned child metadata recording the exact admitted overlay."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = FORK_OVERLAY_SCHEMA_VERSION
    overlay: ForkOverlay
    fingerprint: str
    controller_subject: str

    @field_validator("controller_subject")
    @classmethod
    def validate_subject(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "controller_subject")

    @model_validator(mode="after")
    def validate_fingerprint(self) -> ForkOverlayRecord:
        if self.fingerprint != fork_overlay_fingerprint(self.overlay):
            raise ValueError("Fork overlay fingerprint does not match its content.")
        return self


def fork_overlay_record_from_metadata(metadata: Mapping[str, Any]) -> ForkOverlayRecord | None:
    """Load the exact durable overlay or fail closed when the record is malformed."""

    if not isinstance(metadata, Mapping):
        raise TypeError("session metadata must be a mapping.")
    raw = metadata.get(FORK_OVERLAY_METADATA_KEY)
    if raw is None:
        return None
    try:
        return ForkOverlayRecord.model_validate(raw)
    except ValueError:
        raise ValueError("Fork overlay metadata is malformed; the fork fails closed.") from None


def fork_overlay_granted_tool_names(session: Session) -> frozenset[str]:
    """Return controller tools a fork's durable overlay adds beyond its source ceiling.

    Only a fork-controller body named for the overlay's controller may carry
    them; any other combination admits nothing.
    """

    record = fork_overlay_record_from_metadata(session.metadata)
    if record is None or not record.overlay.tools:
        return frozenset()
    agent_name = session.agent_name
    suffix = f"{FORK_CONTROLLER_AGENT_SEPARATOR}{record.overlay.controller}"
    if not agent_name.endswith(suffix) or agent_name == suffix:
        return frozenset()
    return frozenset(record.overlay.tool_names)


def fork_overlay_context_message(record: ForkOverlayRecord | None) -> Message | None:
    """Render the overlay's instructions and memory as one model-visible layer.

    The layer is rebuilt from durable metadata on every model request and is
    never appended to the transcript or written to any knowledge store.
    """

    if record is None:
        return None
    if type(record) is not ForkOverlayRecord:
        raise TypeError("record must be a ForkOverlayRecord.")
    overlay = record.overlay
    if overlay.instructions is None and not overlay.memory:
        return None
    material: dict[str, Any] = {
        "controller": overlay.controller,
        "scope": "this_fork_only",
    }
    if overlay.instructions is not None:
        material["instructions"] = overlay.instructions
    if overlay.memory:
        material["memory"] = [
            item.model_dump(mode="json", exclude_none=True) for item in overlay.memory
        ]
    payload = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return Message.text(
        MessageRole.USER,
        f"{_CONTEXT_OPEN_MARKER}\n{payload}\n{_CONTEXT_CLOSE_MARKER}",
    )


__all__ = [
    "FORK_CONTROLLER_AGENT_SEPARATOR",
    "FORK_OVERLAY_CONTEXT_MAX_CHARS",
    "FORK_OVERLAY_INSTRUCTIONS_MAX_CHARS",
    "FORK_OVERLAY_MAX_MEMORY_ITEMS",
    "FORK_OVERLAY_MAX_TOOL_GRANTS",
    "FORK_OVERLAY_MEMORY_CONTENT_MAX_CHARS",
    "FORK_OVERLAY_METADATA_KEY",
    "ForkOverlay",
    "ForkOverlayMemoryItem",
    "ForkOverlayRecord",
    "ForkOverlayToolGrant",
    "fork_controller_agent_name",
    "fork_overlay_context_message",
    "fork_overlay_fingerprint",
    "fork_overlay_record_from_metadata",
]
