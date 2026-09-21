"""Bounded immutable input descriptions; these values confer no retention authority.

Receiving owners must authenticate selectors and pin every member before issuing
an acquisition receipt. A valid manifest, including its digest, is not that receipt.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from cayu.collaboration._contracts import ContractValue, ObjectRef

InputDigest = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
InputSize = Annotated[StrictInt, Field(ge=0, le=4 * 1024**3)]
MAX_INPUT_MANIFEST_MEMBERS = 32


class ArtifactInputMember(ContractValue):
    """Exact immutable member identity, including metadata and original bytes."""

    resource: ObjectRef
    content_sha256: InputDigest
    metadata_sha256: InputDigest
    size_bytes: InputSize

    @model_validator(mode="after")
    def pinned_artifact(self) -> Self:
        if self.resource.kind != "artifact" or self.resource.revision is None:
            raise ValueError("Input members require exact artifact revisions.")
        return self


class FolderInputEntry(ContractValue):
    """Canonical relative member path; no ambient filesystem lookup is implied."""

    path: Annotated[StrictStr, Field(min_length=1, max_length=512)]
    git_mode: Literal["100644", "100755"]
    member: ArtifactInputMember

    @field_validator("path")
    @classmethod
    def canonical_relative_path(cls, value: str) -> str:
        if (
            len(value.encode("utf-8")) > 512
            or "\\" in value
            or ":" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError("Input manifest path is not canonical and relative.")
        return value


class FolderInputManifest(ContractValue):
    """Small artifact-backed folder; member bytes remain at their resource owner.

    Paths are ordered, not normalized or silently deduplicated. Multiple paths
    may reference the same immutable artifact, but contradictory descriptions of
    the same resource are invalid. Logical bytes count paths; retention counts
    and physical bytes count distinct resource identities.
    """

    schema_version: Literal[1] = 1
    entries: tuple[FolderInputEntry, ...] = Field(max_length=MAX_INPUT_MANIFEST_MEMBERS)

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_schema_version(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported input manifest schema.")
        return value

    @model_validator(mode="after")
    def canonical_members(self) -> Self:
        paths = tuple(entry.path for entry in self.entries)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("Input manifest paths must be unique and sorted.")
        members: dict[ObjectRef, ArtifactInputMember] = {}
        path_set = set(paths)
        owners = set()
        for entry in self.entries:
            parts = entry.path.split("/")
            if any("/".join(parts[:index]) in path_set for index in range(1, len(parts))):
                raise ValueError("Input manifest file and directory paths conflict.")
            previous = members.setdefault(entry.member.resource, entry.member)
            if previous != entry.member:
                raise ValueError("Input manifest has conflicting member commitments.")
            owners.add(entry.member.resource.owner)
        if len(owners) > 1:
            raise ValueError("Initial folder inputs require one material owner.")
        return self

    @property
    def logical_bytes(self) -> int:
        return sum(entry.member.size_bytes for entry in self.entries)

    @property
    def retained_members(self) -> tuple[ArtifactInputMember, ...]:
        return tuple(dict.fromkeys(entry.member for entry in self.entries))
