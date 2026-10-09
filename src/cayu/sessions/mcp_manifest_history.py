"""Durable MCP manifest history, hashed evidence and shared validation rules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import copy_durable_json_value
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import EVENT_ID_MAX_CHARS


class McpManifestHistoryConflict(RuntimeError):
    """An MCP manifest check could not establish or fence authoritative history."""


class _McpManifestBaselineEvidenceInvalid(ValueError):
    """Stored MCP baseline evidence could not be decoded or validated safely."""


_MCP_MANIFEST_BASELINE_MAX_TOOLS = 10_000


def _require_sha256_identifier(value: str, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must be a SHA-256 identifier.")
    return value


def _mcp_manifest_session_ref(session_id: str) -> str:
    session_id = require_clean_nonblank(session_id, "session_id")
    encoded = json.dumps(
        {
            "schema": "cayu.mcp.accepted_session_ref.v1",
            "session_id": session_id,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _mcp_authoritative_manifest_hash(
    *,
    source_manifest_hash: str,
    server_hash: str,
    tools: Iterable[Mapping[str, str]],
    exposed_tools: Iterable[Mapping[str, str]],
) -> str:
    """Bind every durable manifest-evidence dimension into one authority hash."""

    source_manifest_hash = _require_sha256_identifier(
        source_manifest_hash,
        "source_manifest_hash",
    )
    server_hash = _require_sha256_identifier(server_hash, "server_hash")

    def canonical_evidence(
        value: Iterable[Mapping[str, str]],
        field_name: str,
    ) -> list[dict[str, str]]:
        evidence: list[dict[str, str]] = []
        tool_ids: set[str] = set()
        for item in value:
            if not isinstance(item, Mapping) or set(item) != {"tool_id", "contract_hash"}:
                raise ValueError(
                    f"{field_name} entries must contain only tool_id and contract_hash."
                )
            tool_id = _require_sha256_identifier(item["tool_id"], f"{field_name} tool_id")
            contract_hash = _require_sha256_identifier(
                item["contract_hash"],
                f"{field_name} contract_hash",
            )
            if tool_id in tool_ids:
                raise ValueError(f"{field_name} must not contain duplicate tool_id values.")
            tool_ids.add(tool_id)
            evidence.append(
                {
                    "tool_id": tool_id,
                    "contract_hash": contract_hash,
                }
            )
        evidence.sort(key=lambda entry: entry["tool_id"])
        return evidence

    encoded = json.dumps(
        {
            "schema": "cayu.mcp.authorized_manifest.v2",
            "source_manifest_hash": source_manifest_hash,
            "server_hash": server_hash,
            "tools": canonical_evidence(tools, "tools"),
            "exposed_tools": canonical_evidence(exposed_tools, "exposed_tools"),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class McpManifestBaseline(BaseModel):
    """Authoritative accepted MCP manifest for one durable history identity.

    ``tools`` and ``exposed_tools`` contain only fixed-size ``tool_id`` and
    ``contract_hash`` SHA-256 evidence. Raw MCP or Cayu tool names do not cross
    this durable boundary.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    history_key: str
    generation: StrictInt = Field(ge=1)
    manifest_identity: str
    manifest_hash: str
    source_manifest_hash: str
    server_hash: str
    tools: tuple[dict[str, Any], ...] = Field(
        default_factory=tuple,
        max_length=_MCP_MANIFEST_BASELINE_MAX_TOOLS,
    )
    exposed_tools: tuple[dict[str, Any], ...] = Field(
        default_factory=tuple,
        max_length=_MCP_MANIFEST_BASELINE_MAX_TOOLS,
    )
    accepted_session_ref: str
    accepted_event_id: str = Field(max_length=EVENT_ID_MAX_CHARS)
    accepted_at: datetime

    @field_validator(
        "history_key",
        "manifest_identity",
        "manifest_hash",
        "source_manifest_hash",
        "server_hash",
        "accepted_session_ref",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _require_sha256_identifier(value, info.field_name)

    @field_validator("accepted_event_id")
    @classmethod
    def validate_reference_ids(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("tools", "exposed_tools", mode="before")
    @classmethod
    def copy_tools(cls, value, info) -> tuple[dict[str, Any], ...]:
        field_name = info.field_name
        if not isinstance(value, list | tuple):
            raise TypeError(f"{field_name} must be a list or tuple.")
        if len(value) > _MCP_MANIFEST_BASELINE_MAX_TOOLS:
            raise ValueError(
                f"{field_name} must not contain more than "
                f"{_MCP_MANIFEST_BASELINE_MAX_TOOLS} entries."
            )
        copied = copy_durable_json_value(list(value), field_name)
        if type(copied) is not list or any(type(item) is not dict for item in copied):
            raise TypeError(f"{field_name} must contain JSON objects.")
        tool_ids: set[str] = set()
        for item in copied:
            if set(item) != {"tool_id", "contract_hash"}:
                raise ValueError(
                    f"{field_name} entries must contain only tool_id and contract_hash."
                )
            for item_field_name in ("tool_id", "contract_hash"):
                item[item_field_name] = _require_sha256_identifier(
                    item[item_field_name],
                    f"{info.field_name} {item_field_name}",
                )
            tool_id = item["tool_id"]
            if tool_id in tool_ids:
                raise ValueError(f"{info.field_name} must not contain duplicate tool_id values.")
            tool_ids.add(tool_id)
        return tuple(copied)

    @field_validator("accepted_at")
    @classmethod
    def normalize_accepted_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("accepted_at must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_manifest_evidence(self) -> McpManifestBaseline:
        authoritative_hash = _mcp_authoritative_manifest_hash(
            source_manifest_hash=self.source_manifest_hash,
            server_hash=self.server_hash,
            tools=self.tools,
            exposed_tools=self.exposed_tools,
        )
        if self.manifest_hash != authoritative_hash:
            raise ValueError(
                "manifest_hash does not match the authoritative MCP manifest evidence."
            )
        return self


class McpManifestBaselineLoadResult(BaseModel):
    """Authoritative accepted baselines loaded from one durable store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baselines: dict[str, McpManifestBaseline] = Field(default_factory=dict)

    @field_validator("baselines", mode="before")
    @classmethod
    def copy_baselines(cls, value) -> dict[str, McpManifestBaseline]:
        if type(value) is not dict:
            raise TypeError("baselines must be a dict.")
        return {
            _require_sha256_identifier(key, "baselines key"): McpManifestBaseline.model_validate(
                baseline.model_dump(mode="python")
                if isinstance(baseline, McpManifestBaseline)
                else baseline
            )
            for key, baseline in value.items()
        }

    @model_validator(mode="after")
    def validate_baseline_keys(self) -> McpManifestBaselineLoadResult:
        for key, baseline in self.baselines.items():
            if baseline.history_key != key:
                raise ValueError("MCP baseline history_key does not match its load-result key.")
        return self


class McpManifestPublicationResult(BaseModel):
    """Result of one atomic manifest-baseline compare-and-publication attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    published: StrictBool
    baselines: dict[str, McpManifestBaseline] = Field(default_factory=dict)

    @field_validator("baselines", mode="before")
    @classmethod
    def copy_baselines(cls, value) -> dict[str, McpManifestBaseline]:
        if type(value) is not dict:
            raise TypeError("baselines must be a dict.")
        return {
            _require_sha256_identifier(
                key,
                "baselines key",
            ): McpManifestBaseline.model_validate(
                baseline.model_dump(mode="python")
                if isinstance(baseline, McpManifestBaseline)
                else baseline
            )
            for key, baseline in value.items()
        }

    @model_validator(mode="after")
    def validate_baseline_keys(self) -> McpManifestPublicationResult:
        for key, baseline in self.baselines.items():
            if baseline.history_key != key:
                raise ValueError(
                    "MCP baseline history_key does not match its publication-result key."
                )
        return self
