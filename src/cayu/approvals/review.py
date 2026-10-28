"""Explicit disclosure to humans, separate from public execution projections.

Policies run inside the trusted application boundary. They must be deterministic,
side-effect free, and independently authorize the recipient's session/tenant scope.
Neither authentication nor a known-secret scan proves that private text is safe.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cayu.vaults import SecretRedactor

# Per-field limits apply to the HTML-escaped display text; the aggregate applies
# to the UTF-8 JSON of the selected fields before escaping.
_FIELD_LABEL_MAX_CHARS = 128
_FIELD_TEXT_MAX_CHARS = 4096
_FIELDS_MAX_BYTES = 16 * 1024


class _ReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class HumanReviewContext(_ReviewModel):
    """Verified recipient supplied by trusted SDK code or server authentication."""

    recipient: str = Field(min_length=1, max_length=512)
    tenant: str | None = Field(default=None, min_length=1, max_length=512)
    purpose: str = Field(min_length=1, max_length=128)


class HumanReviewReference(_ReviewModel):
    """Opaque content binding; conveys no authority to execute or resolve."""

    context: HumanReviewContext
    policy_version: str = Field(min_length=1, max_length=128)
    content_tag: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)


class HumanReviewField(_ReviewModel):
    """Untrusted plain text selected by the application policy."""

    label: str = Field(min_length=1, max_length=_FIELD_LABEL_MAX_CHARS)
    text: str = Field(min_length=1, max_length=_FIELD_TEXT_MAX_CHARS, repr=False)


class HumanReviewDisclosure(_ReviewModel):
    status: Literal["permitted", "redacted", "unavailable"]
    fields: tuple[HumanReviewField, ...] = Field(default=(), max_length=32, repr=False)
    sensitive_content: Literal["reject_unknown_scope", "application_attested"] = (
        "reject_unknown_scope"
    )

    @model_validator(mode="after")
    def bound_escaped_fields(self) -> HumanReviewDisclosure:
        # The view returns escaped text under the same field bounds, so admit only
        # fields that still fit after escaping instead of failing the view later.
        if any(
            len(html.escape(item.label)) > _FIELD_LABEL_MAX_CHARS
            or len(html.escape(item.text)) > _FIELD_TEXT_MAX_CHARS
            for item in self.fields
        ):
            raise ValueError("Human-review field exceeds its size limit after HTML escaping.")
        return self


class HumanReviewCall(_ReviewModel):
    tool_call_id: str = Field(max_length=512)
    tool_name: str = Field(max_length=512)
    on_grant: Literal["eligible", "denied", "withheld"]


@dataclass(frozen=True)
class HumanReviewSource:
    """Private policy input. Never log, serialize, or expose this object to a client."""

    kind: Literal["user_input", "tool_approval"]
    interaction_id: str
    tool_round_id: str
    tool_call_id: str
    secret_resolution_scope: Literal["static", "dynamic", "unknown"]
    calls: tuple[HumanReviewCall, ...]
    arguments_by_call: Mapping[str, Mapping[str, Any]] = field(repr=False)
    question: str | None = field(default=None, repr=False)
    options: tuple[str, ...] = field(default=(), repr=False)
    expires_at: datetime | None = None
    executable: bool = True


class HumanReviewView(_ReviewModel):
    """Protected response. All display fields are HTML-escaped untrusted text.

    This response has no generic event, trace, transcript or export contract.
    """

    status: Literal["permitted", "redacted", "unavailable"]
    guidance: Literal[
        "Review all calls before deciding.",
        "Review content is withheld; contact the application owner or deny the action.",
        "No current review is available; refresh pending interactions.",
        "This recovery gate cannot authorize execution; use explicit blocked recovery.",
    ]
    session_id: str
    interaction_id: str | None = None
    tool_round_id: str | None = None
    tool_call_id: str | None = None
    kind: Literal["user_input", "tool_approval"] | None = None
    calls: tuple[HumanReviewCall, ...] = ()
    fields: tuple[HumanReviewField, ...] = Field(default=(), repr=False)
    reference: HumanReviewReference | None = Field(default=None, repr=False)
    display_format: Literal["html_escaped_text"] = "html_escaped_text"


class HumanReviewPolicy(ABC):
    """Application-owned authorization and narrow display projection.

    ``version`` must change whenever authorization, projection or sensitivity rules
    change. ``binding_key`` must contain at least 32 private random bytes, be stable
    across workers/restarts, and never enter session storage. Losing it invalidates
    outstanding views. Projection must select only explicitly permitted fields.
    For dynamic/unknown secret scopes it must independently attest the selected
    output (e.g. validate against an application-owned vocabulary); redaction of
    currently known secrets is insufficient. Never attest arbitrary argument dumps.
    """

    @property
    @abstractmethod
    def version(self) -> str: ...

    @property
    @abstractmethod
    def binding_key(self) -> bytes: ...

    @abstractmethod
    def authorize(
        self,
        context: HumanReviewContext,
        *,
        session_id: str,
        session_metadata: Mapping[str, Any],
        action: Literal["inspect", "decide"],
    ) -> bool:
        """Deny by default; enforce session, tenant, recipient and purpose together."""
        ...

    @abstractmethod
    def project(
        self, context: HumanReviewContext, source: HumanReviewSource
    ) -> HumanReviewDisclosure: ...

    def _project_with_redactor(
        self, context: HumanReviewContext, source: HumanReviewSource, redactor: SecretRedactor
    ) -> HumanReviewDisclosure:
        """Projection boundary for policies that transform text before disclosure."""
        return self.project(context, source)

    def audit(self, *, action: Literal["inspect", "decide"], status: str) -> None:
        """Optional bounded access audit. No model text, arguments or private identities."""
        return None

    @property
    def dashboard_purpose(self) -> str | None:
        """The purpose the dashboard requests reviews with, or None to leave it unset.

        When the dashboard's runtime config sets no ``humanReviewPurpose``, the
        server uses this one, so approvals and questions go through protected review.
        """
        return None


class HumanReviewDenied(PermissionError):
    def __init__(self) -> None:
        super().__init__("Human-review access is not authorized.")


class HumanReviewConflict(ValueError):
    def __init__(self) -> None:
        super().__init__("Human-review content is no longer current; inspect again.")


def require_review_authority(
    policy: HumanReviewPolicy | None,
    context: HumanReviewContext,
    *,
    session_id: str,
    session_metadata: Mapping[str, Any],
    action: Literal["inspect", "decide"],
) -> HumanReviewPolicy:
    try:
        if (
            policy is None
            or policy.authorize(
                context, session_id=session_id, session_metadata=session_metadata, action=action
            )
            is not True
        ):
            raise HumanReviewDenied()
        if type(policy.binding_key) is not bytes or len(policy.binding_key) < 32:
            raise HumanReviewDenied()
        if not 1 <= len(policy.version) <= 128:
            raise HumanReviewDenied()
    except Exception:
        raise HumanReviewDenied() from None
    return policy


def build_review(
    *,
    policy: HumanReviewPolicy,
    context: HumanReviewContext,
    source: HumanReviewSource,
    session_id: str,
    session_instance_id: str,
    authoritative_content: object,
    redactor: SecretRedactor,
    now: datetime,
) -> HumanReviewView:
    """Project a private detached snapshot; bind both display and execution content."""
    try:
        disclosure = policy._project_with_redactor(context, source, redactor)
        if type(disclosure) is not HumanReviewDisclosure:
            raise ValueError
        disclosure = HumanReviewDisclosure.model_validate(disclosure.model_dump())
        status = disclosure.status
        fields = disclosure.fields
        if (source.expires_at is not None and source.expires_at <= now) or not source.executable:
            status, fields = "unavailable", ()
        elif (
            source.secret_resolution_scope != "static"
            and disclosure.sensitive_content != "application_attested"
        ):
            status, fields = "redacted", ()
        if status != "permitted":
            fields = ()
        elif not fields:
            status = "unavailable"
        raw = json.dumps([item.model_dump() for item in fields], ensure_ascii=True)
        if redactor.redact_text(raw) != raw:
            status, fields = "redacted", ()
        # Check original scalars too: JSON escapes must not hide registered secrets.
        if any(
            redactor.redact_text(value) != value
            for item in fields
            for value in (item.label, item.text)
        ):
            status, fields = "redacted", ()
        size = json.dumps([item.model_dump() for item in fields], ensure_ascii=False)
        if len(size.encode("utf-8")) > _FIELDS_MAX_BYTES:
            status, fields = "unavailable", ()
        escaped = tuple(
            HumanReviewField(label=html.escape(item.label), text=html.escape(item.text))
            for item in fields
        )
        display = [item.model_dump() for item in escaped]
        material = json.dumps(
            [
                "cayu.human-review.v1",
                policy.version,
                context.model_dump(),
                session_id,
                session_instance_id,
                authoritative_content,
                status,
                display,
                [call.model_dump() for call in source.calls],
            ],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        tag = hmac.new(policy.binding_key, material, hashlib.sha256).hexdigest()
        return HumanReviewView(
            status=status,
            guidance=(
                "This recovery gate cannot authorize execution; use explicit blocked recovery."
                if not source.executable
                else "Review all calls before deciding."
                if status == "permitted"
                else "Review content is withheld; contact the application owner or deny the action."
            ),
            session_id=session_id,
            interaction_id=source.interaction_id,
            tool_round_id=source.tool_round_id,
            tool_call_id=source.tool_call_id,
            kind=source.kind,
            calls=source.calls,
            fields=escaped,
            reference=HumanReviewReference(
                context=context, policy_version=policy.version, content_tag=tag
            ),
        )
    except Exception:
        # A policy exception may itself contain private arguments. Never chain it.
        return HumanReviewView(
            status="unavailable",
            session_id=session_id,
            guidance="No current review is available; refresh pending interactions.",
        )


def require_current_review(
    reference: HumanReviewReference, current: HumanReviewView, *, denying: bool
) -> None:
    if (
        current.reference is None
        or reference.policy_version != current.reference.policy_version
        or reference.context != current.reference.context
        or not hmac.compare_digest(reference.content_tag, current.reference.content_tag)
        or (not denying and current.status != "permitted")
    ):
        raise HumanReviewConflict()
