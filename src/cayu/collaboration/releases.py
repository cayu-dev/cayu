"""Exact reviewed-content contracts and trusted release-owner readback."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Annotated

from pydantic import Field, StrictInt, StrictStr, model_validator

from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.collaboration._contracts import (
    ContractValue,
    Generation,
    Identifier,
    InitiatorBinding,
    ObjectRef,
    OwnerRef,
)
from cayu.collaboration.mandates import Expiry, InputChannel

ContentDigest = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
ReleaseSourceIndex = Annotated[StrictInt, Field(ge=0, le=MAX_PORTABLE_JSON_INTEGER)]


class ContentExposure(ContractValue):
    """A declared input occurrence; its release owner must authenticate provenance."""

    source: ObjectRef
    channel: InputChannel
    commitment: ContentDigest

    @model_validator(mode="after")
    def pinned_source(self) -> ContentExposure:
        if self.source.revision is None:
            raise ValueError("Content exposure requires an exact source revision.")
        return self


class ContentReleaseRequest(ContractValue):
    """Public reference to an approval, not an approval or bearer capability."""

    unordered_fields = frozenset({"exposure"})

    decision: ObjectRef
    source_commitment: ContentDigest
    text_commitment: ContentDigest
    exposure: tuple[ContentExposure, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def pinned_decision(self) -> ContentReleaseRequest:
        if self.decision.revision is None:
            raise ValueError("Content release requires an exact decision revision.")
        if len({(item.source, item.channel) for item in self.exposure}) != len(self.exposure):
            raise ValueError("Content exposure repeats a source/channel authority.")
        return self


class ContentReleaseExpectation(ContractValue):
    """Complete release input resolved by the source owner, without draft bytes."""

    request: ContentReleaseRequest
    source_owner: OwnerRef
    session_id: Identifier
    session_instance_id: Identifier
    source_indices: tuple[ReleaseSourceIndex, ...] = Field(max_length=16)
    audience: OwnerRef
    validator: ObjectRef
    policy: ObjectRef

    @model_validator(mode="after")
    def exact_selection(self) -> ContentReleaseExpectation:
        if (
            tuple(sorted(set(self.source_indices))) != self.source_indices
            or self.validator.revision is None
            or self.validator.owner != self.request.decision.owner
        ):
            raise ValueError("Content release selection or validator conflicts.")
        return self


class ContentReleaseReceipt(ContractValue):
    """Historical authenticated review evidence; no current disclosure grant."""

    expected: ContentReleaseExpectation
    reviewer: InitiatorBinding
    authorization_revision: Generation
    expires_at_ms: Expiry

    @model_validator(mode="after")
    def exact_issuer(self) -> ContentReleaseReceipt:
        if self.reviewer.issuer != self.expected.request.decision.owner:
            raise ValueError("Content review issuer conflicts with its decision owner.")
        return self


class ReleasedContent(ContractValue):
    receipt: ContentReleaseReceipt
    text: Annotated[StrictStr, Field(max_length=8192)]


class ContentReleaseReader(ABC):
    @property
    @abstractmethod
    def ref(self) -> ObjectRef:
        """Exact versioned registered decision-reader/validator configuration."""

    @abstractmethod
    def acquire(
        self, expected: ContentReleaseExpectation
    ) -> AbstractAsyncContextManager[ReleasedContent]:
        """Authenticate exact reviewed bytes and all exposure/source authority.

        This must load positive review evidence from its application-owned source,
        not echo a caller proposal as approval. Hold current revocation through
        publication/exposure. Failure or missing evidence must refuse. No model
        calls, implicit review, semantic DLP promise or mutable regeneration.
        """
