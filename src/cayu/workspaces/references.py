"""Identity bindings for application-owned workspace data."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import require_durable_clean_nonblank
from cayu.workspaces.revisions import WorkspaceIdentity


class WorkspaceReferenceBindingError(RuntimeError):
    """An application reference cannot be used for the selected workspace."""

    def __init__(
        self, code: Literal["workspace_binding_unavailable", "workspace_binding_mismatch"]
    ):
        self.code = code
        super().__init__(code)


class WorkspaceReferenceBinding(BaseModel):
    """Serializable ownership assertion, not a file or permission certificate.

    Create with Workspace.reference_binding() or ToolContext.workspace_reference_binding().
    Consume with ToolContext.require_workspace_binding() before using application claims.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    identity: WorkspaceIdentity
    generation: str

    @field_validator("generation")
    @classmethod
    def validate_generation(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "generation")

    def require_match(self, selected: WorkspaceReferenceBinding) -> None:
        """Check against an observed binding; never against reconstructed metadata."""
        expected = type(self).model_validate(self.model_dump())
        actual = WorkspaceReferenceBinding.model_validate(selected.model_dump())
        if expected != actual:
            raise WorkspaceReferenceBindingError("workspace_binding_mismatch")
