"""Principal-derived knowledge access constraints and detached scope copies."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import canonical_durable_json_bytes, copy_json_value, copy_label_map
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.knowledge.records import KnowledgeStatus, KnowledgeVisibility, _dedupe_strings


class KnowledgeAccessDenied(PermissionError):
    """Raised when a knowledge mutation falls outside its explicit access scope."""

    def __init__(self, operation: str) -> None:
        self.operation = require_clean_nonblank(operation, "operation")
        super().__init__(f"Knowledge access denied for {self.operation}.")


class KnowledgeAccessScope(BaseModel):
    """Principal-derived constraints enforced inside every knowledge operation.

    Cayu deliberately does not model tenants, organizations, users, or RBAC. The
    hosting application maps those concepts into namespaces, labels, visibility,
    source identity, lifecycle state, and expiration eligibility. Namespace-wide
    access must be explicit; constructing a scope with no namespace is invalid.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    resource_constraints: tuple[str, ...] = Field(default=(), exclude_if=lambda v: not v)

    @field_validator("resource_constraints")
    @classmethod
    def validate_resource_constraints(cls, values):
        from cayu.knowledge.access import validate_constraints

        return validate_constraints(values)

    allowed_namespaces: list[str] = Field(default_factory=list)
    allow_all_namespaces: bool = False
    required_labels: dict[str, str] = Field(default_factory=dict)
    allowed_visibilities: list[KnowledgeVisibility] = Field(
        default_factory=lambda: [KnowledgeVisibility.GLOBAL]
    )
    allowed_source_types: list[str] | None = None
    allowed_source_ids: list[str] | None = None
    allowed_statuses: list[KnowledgeStatus] = Field(
        default_factory=lambda: [KnowledgeStatus.ACTIVE]
    )
    include_expired: bool = False

    @field_validator(
        "allowed_namespaces", "allowed_source_types", "allowed_source_ids", mode="before"
    )
    @classmethod
    def copy_string_lists(cls, value, info) -> list[str] | None:
        if value is None and info.field_name in {"allowed_source_types", "allowed_source_ids"}:
            return None
        if value is None:
            return []
        copied = copy_json_value(value, info.field_name)
        if type(copied) is not list:
            raise ValueError(f"`{info.field_name}` must be a list.")
        result: list[str] = []
        for index, item in enumerate(copied):
            if type(item) is not str:
                raise ValueError(f"`{info.field_name}[{index}]` must be a string.")
            result.append(require_clean_nonblank(item, f"{info.field_name}[{index}]"))
        return sorted(_dedupe_strings(result))

    @field_validator("required_labels", mode="before")
    @classmethod
    def copy_required_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "required_labels")

    @field_validator("allowed_visibilities", "allowed_statuses")
    @classmethod
    def validate_nonempty_enum_lists(cls, value: list[Any], info) -> list[Any]:
        if not value:
            raise ValueError(f"`{info.field_name}` cannot be empty.")
        return sorted(dict.fromkeys(value), key=str)

    @field_validator("allow_all_namespaces", "include_expired", mode="before")
    @classmethod
    def validate_boolean_fields(cls, value, info) -> bool:
        if type(value) is not bool:
            raise ValueError(f"`{info.field_name}` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_namespace_boundary(self) -> KnowledgeAccessScope:
        if self.allow_all_namespaces and self.allowed_namespaces:
            raise ValueError(
                "`allowed_namespaces` must be empty when `allow_all_namespaces` is true."
            )
        if not self.allow_all_namespaces and not self.allowed_namespaces:
            raise ValueError(
                "Knowledge access requires `allowed_namespaces` or explicit "
                "`allow_all_namespaces=True`."
            )
        return self

    @classmethod
    def for_namespace(
        cls,
        namespace: str,
        *,
        required_labels: dict[str, str] | None = None,
        allowed_visibilities: list[KnowledgeVisibility] | None = None,
        allowed_source_types: list[str] | None = None,
        allowed_source_ids: list[str] | None = None,
        allowed_statuses: list[KnowledgeStatus] | None = None,
        include_expired: bool = False,
    ) -> KnowledgeAccessScope:
        """Create an explicit single-namespace application scope."""

        values: dict[str, Any] = {
            "allowed_namespaces": [namespace],
            "required_labels": required_labels or {},
            "allowed_source_types": allowed_source_types,
            "allowed_source_ids": allowed_source_ids,
            "include_expired": include_expired,
        }
        if allowed_visibilities is not None:
            values["allowed_visibilities"] = allowed_visibilities
        if allowed_statuses is not None:
            values["allowed_statuses"] = allowed_statuses
        return cls(**values)

    @classmethod
    def privileged(cls) -> KnowledgeAccessScope:
        """Create an explicit all-knowledge scope for trusted host maintenance."""

        return cls(
            allow_all_namespaces=True,
            allowed_visibilities=list(KnowledgeVisibility),
            allowed_statuses=list(KnowledgeStatus),
            include_expired=True,
        )


def copy_knowledge_access_scope(scope: KnowledgeAccessScope) -> KnowledgeAccessScope:
    if type(scope) is not KnowledgeAccessScope:
        raise TypeError("KnowledgeAccessScope instances must not be subclasses.")
    return KnowledgeAccessScope(
        resource_constraints=scope.resource_constraints,
        allowed_namespaces=list(scope.allowed_namespaces),
        allow_all_namespaces=scope.allow_all_namespaces,
        required_labels=copy_label_map(scope.required_labels, "required_labels"),
        allowed_visibilities=list(scope.allowed_visibilities),
        allowed_source_types=(
            None if scope.allowed_source_types is None else list(scope.allowed_source_types)
        ),
        allowed_source_ids=(
            None if scope.allowed_source_ids is None else list(scope.allowed_source_ids)
        ),
        allowed_statuses=list(scope.allowed_statuses),
        include_expired=scope.include_expired,
    )


def _knowledge_access_scope_sha256(scope: KnowledgeAccessScope) -> str:
    scope = copy_knowledge_access_scope(scope)
    return sha256(
        canonical_durable_json_bytes(
            scope.model_dump(mode="json"),
            "knowledge change access scope",
        )
    ).hexdigest()


def knowledge_access_scope_sha256(scope: KnowledgeAccessScope) -> str:
    """Return the canonical public identity of one enforced knowledge access scope."""

    return _knowledge_access_scope_sha256(scope)
