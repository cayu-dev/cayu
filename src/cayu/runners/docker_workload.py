"""Typed, reconstructable restrictions for Docker workloads."""

from __future__ import annotations

import posixpath
import re
from hashlib import sha256
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_PINNED_REFERENCE_PATTERN = re.compile(r"@sha256:([0-9a-f]{64})$")
_CAPABILITY_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class DockerImageIdentity(BaseModel):
    """Trusted immutable image selection admitted before a coding session.

    Registry images use a digest-pinned ``reference``. A trusted locally built
    image may use a tag only when ``content_digest`` names the exact Docker image
    ID expected after allocation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    reference: str = Field(max_length=512)
    content_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("reference")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        reference = require_durable_clean_nonblank(value, "reference")
        if any(character in reference for character in ("\x00", "\n", "\r")):
            raise ValueError("Docker image reference contains unsupported control text.")
        return reference

    @model_validator(mode="after")
    def validate_immutable_identity(self) -> Self:
        pinned = _PINNED_REFERENCE_PATTERN.search(self.reference)
        if pinned is None and self.content_digest is None:
            raise ValueError(
                "Docker image identity requires a digest-pinned reference or content_digest."
            )
        if (
            self.content_digest is not None
            and _SHA256_PATTERN.fullmatch(self.content_digest) is None
        ):
            raise ValueError("Docker image content_digest must be a lowercase SHA-256 ID.")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the stable non-secret identity used by admission evidence."""

        return (
            "sha256:"
            + sha256(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "docker_image_identity",
                )
            ).hexdigest()
        )


class DockerTmpfsMount(BaseModel):
    """One bounded guest-writable tmpfs mount."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    target: str = Field(max_length=4096)
    size_bytes: StrictInt = Field(ge=1024 * 1024, le=16 * 1024 * 1024 * 1024)
    mode: StrictInt = Field(default=0o700, ge=0, le=0o7777)
    noexec: StrictBool = True

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        target = require_durable_clean_nonblank(value, "target")
        if not posixpath.isabs(target):
            raise ValueError("Docker tmpfs target must be an absolute guest path.")
        normalized = posixpath.normpath(target)
        if normalized in {"/", "/proc", "/sys", "/dev"}:
            raise ValueError("Docker tmpfs target must be a bounded workload path.")
        if "," in normalized:
            raise ValueError("Docker tmpfs target must not contain commas.")
        return normalized


class DockerWorkloadRestrictions(BaseModel):
    """Normalized Docker controls for one explicitly trusted workload.

    The value intentionally has no raw ``docker run`` argument escape hatch.
    Capability drop is unconditional; ``capability_add`` is the explicit,
    empty-by-default add-back set.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    home_directory: str = Field(default="/tmp/cayu-home", max_length=4000)

    uid: StrictInt = Field(default=1000, ge=1, le=2**31 - 1)
    gid: StrictInt = Field(default=1000, ge=1, le=2**31 - 1)
    read_only_root: StrictBool = True
    no_new_privileges: StrictBool = True
    capability_add: tuple[str, ...] = ()
    pids_limit: StrictInt = Field(default=128, ge=1, le=32_768)
    memory_bytes: StrictInt = Field(
        default=512 * 1024 * 1024,
        ge=16 * 1024 * 1024,
        le=1024 * 1024 * 1024 * 1024,
    )
    memory_swap_bytes: StrictInt = Field(
        default=512 * 1024 * 1024,
        ge=16 * 1024 * 1024,
        le=1024 * 1024 * 1024 * 1024,
    )
    cpu_period_us: StrictInt = Field(default=100_000, ge=1_000, le=1_000_000)
    cpu_quota_us: StrictInt = Field(default=100_000, ge=1_000, le=2**53 - 1)
    shm_size_bytes: StrictInt = Field(
        default=64 * 1024 * 1024,
        ge=1024 * 1024,
        le=16 * 1024 * 1024 * 1024,
    )
    tmpfs: tuple[DockerTmpfsMount, ...] = (
        DockerTmpfsMount(
            target="/tmp",
            size_bytes=64 * 1024 * 1024,
            mode=0o1777,
            noexec=False,
        ),
        DockerTmpfsMount(
            target="/workspace",
            size_bytes=256 * 1024 * 1024,
            mode=0o750,
            noexec=False,
        ),
    )

    @field_validator("capability_add")
    @classmethod
    def validate_capability_add(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        owned = tuple(require_durable_clean_nonblank(item, "capability_add item") for item in value)
        if len(owned) > 32:
            raise ValueError("Docker capability_add must contain at most 32 entries.")
        if any(_CAPABILITY_PATTERN.fullmatch(item) is None for item in owned):
            raise ValueError("Docker capability_add entries must use canonical capability names.")
        if tuple(sorted(set(owned))) != owned:
            raise ValueError("Docker capability_add entries must be unique and sorted.")
        return owned

    @field_validator("tmpfs")
    @classmethod
    def validate_tmpfs(
        cls,
        value: tuple[DockerTmpfsMount, ...],
    ) -> tuple[DockerTmpfsMount, ...]:
        if not value:
            raise ValueError("Docker restrictions require at least one bounded tmpfs mount.")
        if len(value) > 16:
            raise ValueError("Docker restrictions allow at most 16 tmpfs mounts.")
        targets = [mount.target for mount in value]
        if len(targets) != len(set(targets)):
            raise ValueError("Docker tmpfs targets must be unique.")
        if tuple(sorted(targets)) != tuple(targets):
            raise ValueError("Docker tmpfs mounts must be sorted by target.")
        return value

    @model_validator(mode="after")
    def validate_resource_relationships(self) -> Self:
        if self.memory_swap_bytes < self.memory_bytes:
            raise ValueError("Docker memory_swap_bytes must be at least memory_bytes.")
        if not any(mount.target == "/tmp" for mount in self.tmpfs):
            raise ValueError("Docker writable home requires a bounded /tmp tmpfs allocation.")
        if any(mount.target.startswith("/tmp/") for mount in self.tmpfs):
            raise ValueError("Docker writable home forbids nested /tmp mounts.")
        return self

    @field_validator("home_directory")
    @classmethod
    def validate_home_directory(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "home_directory")
        if not value.startswith("/tmp/") or posixpath.normpath(value) != value:
            raise ValueError("Docker home_directory must be a normalized directory below /tmp.")
        return value

    @property
    def home_environment(self) -> dict[str, str]:
        """Non-secret, versioned by the restrictions identity, disposable tool state."""

        return {
            "HOME": self.home_directory,
            "XDG_CACHE_HOME": f"{self.home_directory}/.cache",
            "XDG_CONFIG_HOME": f"{self.home_directory}/.config",
            "XDG_DATA_HOME": f"{self.home_directory}/.local/share",
            "XDG_STATE_HOME": f"{self.home_directory}/.local/state",
            "UV_CACHE_DIR": f"{self.home_directory}/.cache/uv",
        }

    @property
    def user(self) -> str:
        """Return Docker's canonical numeric user/group selector."""

        return f"{self.uid}:{self.gid}"

    @property
    def supports_strict_privilege_evidence(self) -> bool:
        """Return whether the controls support Cayu's strict privilege claims."""

        return self.read_only_root and self.no_new_privileges and not self.capability_add

    def run_args(self) -> tuple[str, ...]:
        """Project the typed value into deterministic ``docker run`` arguments."""

        args = ["--user", self.user]
        for name, value in self.home_environment.items():
            args += ["--env", f"{name}={value}"]
        if self.read_only_root:
            args.append("--read-only")
        if self.no_new_privileges:
            args += ["--security-opt", "no-new-privileges=true"]
        args += ["--cap-drop", "ALL"]
        for capability in self.capability_add:
            args += ["--cap-add", capability]
        args += [
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            str(self.memory_bytes),
            "--memory-swap",
            str(self.memory_swap_bytes),
            "--cpu-period",
            str(self.cpu_period_us),
            "--cpu-quota",
            str(self.cpu_quota_us),
            "--shm-size",
            str(self.shm_size_bytes),
        ]
        for mount in self.tmpfs:
            options = [
                "rw",
                "nosuid",
                "nodev",
                f"size={mount.size_bytes}",
                f"uid={self.uid}",
                f"gid={self.gid}",
                f"mode={mount.mode:o}",
            ]
            if mount.noexec:
                options.append("noexec")
            args += ["--tmpfs", f"{mount.target}:{','.join(options)}"]
        return tuple(args)

    @property
    def fingerprint(self) -> str:
        """Return the stable non-secret identity of all normalized controls."""

        return (
            "sha256:"
            + sha256(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "docker_workload_restrictions",
                )
            ).hexdigest()
        )


__all__ = [
    "DockerImageIdentity",
    "DockerTmpfsMount",
    "DockerWorkloadRestrictions",
]
