"""Immutable session-addition plans and receipts; never a portable authority token."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictStr, model_validator

from cayu.evals.corpus import AssertionSpec, EvalCorpusDocument
from cayu.evals.promotion import CapturedEvaluationCandidateV1
from cayu.evals.read_views import EvalSuiteViewIdentity
from cayu.evals.results import CapturedEvaluationResultV1
from cayu.evals.suite_authoring import EvalSuiteDocument


class _AdditionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class EvalExistingDestination(_AdditionModel):
    kind: Literal["existing"] = "existing"
    selector: StrictStr = Field(min_length=1, max_length=4096)


class EvalNewDestination(_AdditionModel):
    kind: Literal["new"] = "new"
    suite_id: StrictStr = Field(pattern=r"^[a-z][a-z0-9._-]{0,127}$")
    name: StrictStr = Field(min_length=1, max_length=256)


class EvalAdditionRequest(_AdditionModel):
    mode: Literal["case", "score"]
    destination: Annotated[
        EvalExistingDestination | EvalNewDestination, Field(discriminator="kind")
    ]
    case_id: StrictStr = Field(pattern=r"^[a-z][a-z0-9._-]{0,127}$")
    case_name: StrictStr = Field(min_length=1, max_length=256)
    assertions: tuple[AssertionSpec, ...] | None = Field(
        default=None, min_length=1, max_length=1000
    )


class EvalAdditionCommitRequest(EvalAdditionRequest):
    expected_plan_revision: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    expected_source_revision: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class EvalAdditionPlan(_AdditionModel):
    revision: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mode: Literal["case", "score"]
    source_revision: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_binding_revision: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    parent: EvalSuiteViewIdentity | None
    destination: EvalSuiteViewIdentity
    case_id: StrictStr
    case_revision: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    case_name: StrictStr
    case_definition: dict[str, JsonValue]
    destination_case_count: int = Field(ge=1)
    membership_changed: bool
    execution_case_count: Literal[1] | None
    captured_result_revision: StrictStr | None

    @model_validator(mode="after")
    def validate_contract(self):
        if self.revision != addition_revision(self.model_dump(mode="json", exclude={"revision"})):
            raise ValueError("Addition plan revision does not match its content.")
        if (
            self.case_definition.get("id") != self.case_id
            or self.case_definition.get("revision") != self.case_revision
        ):
            raise ValueError("Addition preview does not match its canonical case.")
        if self.parent is not None and self.parent.target_key != self.destination.target_key:
            raise ValueError("An addition cannot change target authority.")
        if self.membership_changed != (self.parent != self.destination):
            raise ValueError("Addition membership change does not match its immutable destination.")
        if self.mode == "score":
            if self.execution_case_count != 1 or self.captured_result_revision is None:
                raise ValueError("A scored addition must identify one captured execution.")
            if self.parent is not None and self.parent != self.destination:
                raise ValueError("A scored attachment cannot change membership.")
        elif self.execution_case_count is not None or self.captured_result_revision is not None:
            raise ValueError("Case membership alone is not an execution.")
        return self


class EvalAdditionReceipt(_AdditionModel):
    operation_id: StrictStr
    request_revision: StrictStr
    plan: EvalAdditionPlan
    created_at: datetime


class EvalAdditionRecord(_AdditionModel):
    """Private nonportable record, written only by the capture publication owner."""

    receipt: EvalAdditionReceipt
    subject: StrictStr | None
    tenant: StrictStr | None
    public_session_id: StrictStr
    private_session_id: StrictStr
    request_material: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_request_commitment(self):
        if self.receipt.request_revision != addition_revision(self.request_material):
            raise ValueError("Addition receipt does not match its full request material.")
        if self.request_material.get("session_id") != self.public_session_id:
            raise ValueError("Addition source does not match its authenticated request.")
        if self.receipt.plan.source_binding_revision != addition_source_binding(
            self.public_session_id, self.private_session_id, self.receipt.plan.source_revision
        ):
            raise ValueError("Addition preview does not bind its server-resolved source.")
        return self


@dataclass(frozen=True)
class PreparedEvalAddition:
    """Internal handoff, distinct from all HTTP request and portable document models."""

    plan: EvalAdditionPlan
    source: CapturedEvaluationCandidateV1
    private_session_id: str
    authored: EvalSuiteDocument | None = None
    corpus: EvalCorpusDocument | None = None
    captured: CapturedEvaluationResultV1 | None = None


class EvalAdditionConflict(ValueError):
    """The same durable operation identity was supplied with different material."""


def addition_revision(material: object) -> str:
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def addition_source_binding(
    public_session_id: str, private_session_id: str, source_revision: str
) -> str:
    """Opaque commitment to the resolved source; never a portable authority token."""
    return addition_revision(
        ["eval-addition-source", public_session_id, private_session_id, source_revision]
    )


def addition_operation_id(subject: str | None, tenant: str | None, key: str) -> str:
    return addition_revision(["eval-session-addition", subject, tenant, key])


class EvalAdditionSourceLink(_AdditionModel):
    session_id: StrictStr | None
    state: Literal["available", "unavailable"]
