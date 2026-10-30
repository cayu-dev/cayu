"""Immutable session-addition plans and receipts; never a portable authority token."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictStr,
    ValidationError,
    model_validator,
)

from cayu.evals.corpus import AssertionSpec, EvalCorpusDocument
from cayu.evals.execution import CorpusTarget, WorkflowEvalTarget
from cayu.evals.models import Trajectory
from cayu.evals.promotion import CapturedEvaluationCandidateV1
from cayu.evals.read_views import EvalSuiteViewIdentity
from cayu.evals.results import CapturedEvaluationResultV1
from cayu.evals.suite_authoring import EvalSuiteDocument
from cayu.evals.workspace_fixtures import WorkspaceFixtureSpec
from cayu.events import EventType
from cayu.workspaces._workspace_origin import WORKSPACE_ORIGIN_METADATA_KEY, WorkspaceOrigin


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


class EvalAdditionWorkspace(_AdditionModel):
    """The workspace fixture a runnable case gets from its session, or why it gets none.

    ``captured``: the case starts from ``repository`` at ``revision``, the commit
    the session's source workspace was at when it bound; ``clean`` says whether
    Git confirmed the checkout had no uncommitted change, which the environment
    doesn't record. ``unregistered``: the session ran on that commit of a source
    the eval target's application hasn't registered. ``unsupported``: the eval
    target runs a workflow, whose trials can't start from a fixture.
    ``unrecorded``: the session's environment recorded no revision. ``message``
    says which, plainly.
    """

    state: Literal["captured", "unregistered", "unsupported", "unrecorded"]
    repository: StrictStr | None = None
    revision: StrictStr | None = None
    clean: StrictBool | None = None
    message: StrictStr = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def validate_state(self):
        recorded = self.state != "unrecorded"
        if (self.repository is not None, self.revision is not None) != (recorded, recorded):
            raise ValueError("Only a recorded workspace names its repository and revision.")
        if (self.clean is not None) != (self.state == "captured"):
            raise ValueError("Only a captured workspace says whether its checkout was clean.")
        return self

    def fixture(self) -> WorkspaceFixtureSpec | None:
        """The case's fixture: the captured commit with no files written over it."""

        if self.state != "captured":
            return None
        assert self.repository is not None and self.revision is not None
        return WorkspaceFixtureSpec(repository=self.repository, revision=self.revision)


def session_addition_workspace(
    trajectory: Trajectory, *, target: CorpusTarget
) -> EvalAdditionWorkspace:
    """The workspace fixture the session's first environment binding recorded.

    Only the first binding counts: a later turn binds a checkout the session
    may already have written back into. The fixture must name a source that
    ``target``'s application, which runs the case, registered.
    """

    session = trajectory.session
    binding = next(
        (
            event
            for event in trajectory.events
            if event.type == EventType.ENVIRONMENT_BINDING_COMPLETED
            and session is not None
            and event.session_id == session.id
        ),
        None,
    )
    metadata = None if binding is None else binding.payload.get("bound_metadata")
    raw = metadata.get(WORKSPACE_ORIGIN_METADATA_KEY) if isinstance(metadata, dict) else None
    try:
        origin = None if raw is None else WorkspaceOrigin.model_validate(raw)
    except ValidationError:
        origin = None
    if origin is None:
        return EvalAdditionWorkspace(
            state="unrecorded",
            message=(
                "The session's environment recorded no workspace revision, so the case has no "
                "workspace fixture."
            ),
        )
    if type(target) is WorkflowEvalTarget:
        return EvalAdditionWorkspace(
            state="unsupported",
            repository=origin.repository,
            revision=origin.revision,
            message=(
                f"The session ran on commit {origin.revision} of workspace source "
                f"{origin.repository!r}, but its eval target runs a workflow, which builds its "
                "own runtime for each trial, so the case has no workspace fixture."
            ),
        )
    try:
        target.app.get_workspace_source(origin.repository)
    except KeyError:
        return EvalAdditionWorkspace(
            state="unregistered",
            repository=origin.repository,
            revision=origin.revision,
            message=(
                f"The session ran on commit {origin.revision} of workspace source "
                f"{origin.repository!r}, which the eval target's application hasn't registered, "
                "so the case has no workspace fixture. Register the source with "
                "CayuApp.register_workspace_source and preview again to start the case at "
                "that commit."
            ),
        )
    message = (
        f"The case starts from workspace source {origin.repository!r} at commit "
        f"{origin.revision}, where the session's checkout was."
    )
    if not origin.clean:
        message += (
            " The checkout also had uncommitted changes (or Git couldn't confirm it had none), "
            "which the environment doesn't record, so the case starts from the commit alone."
        )
    return EvalAdditionWorkspace(
        state="captured",
        repository=origin.repository,
        revision=origin.revision,
        clean=origin.clean,
        message=message,
    )


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
    # What a runnable case's workspace starts from; absent for a scored capture, which
    # doesn't run, and from plans made before sessions recorded their workspace.
    workspace: EvalAdditionWorkspace | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

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
        if self.workspace is not None:
            if self.mode == "score":
                raise ValueError("A scored capture doesn't run, so it has no workspace.")
            fixture = self.workspace.fixture()
            expected = None if fixture is None else fixture.model_dump(mode="json")
            if self.case_definition.get("workspace") != expected:
                raise ValueError("Addition preview does not match its case's workspace fixture.")
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
