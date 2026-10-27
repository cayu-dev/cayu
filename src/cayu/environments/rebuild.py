"""Recreate a per-job environment factory in another process from durable session state.

An application that registers environments per job (a clone, a checkout, a
mission sandbox) in one process can let another process of the same
application recreate them, for example the serving process that forks those
sessions. Three parts compose, and each is usable on its own:

- A factory opts in by returning an :class:`EnvironmentRebuildRecord` from
  ``EnvironmentFactory.rebuild_record``: a kind and a small, secret-free JSON
  payload, such as a clone source and revision.
- Cayu records that record in the session's durable
  factory events whenever the factory creates or reconnects the session's
  environment. Forks also retain the selected record in their own
  ``session.forked`` event, so they remain rebuildable after source deletion.
- A process that should recreate such environments registers an
  :data:`EnvironmentRebuilder` for the kind with
  ``CayuApp.register_environment_rebuilder``. When it forks or resumes a
  session whose environment it does not register, it reads the record from the
  session (or the session it was forked or spawned from), asks the rebuilder
  for an :class:`EnvironmentRebuild`, and registers the rebuilt factory under
  the session's environment name.

The payload is data the factory wrote, not authority: a rebuilder must
validate it as it would any other input (for example, allow only known clone
sources).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
)
from cayu.artifacts.base import ArtifactStore
from cayu.environments.base import EnvironmentSpec
from cayu.environments.factory import EnvironmentFactory

ENVIRONMENT_REBUILD_KIND_MAX_BYTES = 128
"""Longest rebuild kind, in UTF-8 bytes."""

ENVIRONMENT_REBUILD_PAYLOAD_MAX_BYTES = 16 * 1024
"""Largest rebuild payload, in canonical JSON bytes."""


@dataclass(frozen=True, slots=True)
class EnvironmentRebuildRecord:
    """What another process needs to recreate one environment factory.

    ``kind`` names the rebuilder that understands ``payload``, for example
    ``"scout.clone"``. ``payload`` is durable JSON, at most 16 KiB, and must not
    hold secrets: it is stored in session events that operators and fork
    controllers can read. Registration refuses a payload that contains a
    secret the application's redactor knows.
    """

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = require_durable_clean_nonblank(self.kind, "rebuild_record.kind")
        if len(kind.encode("utf-8")) > ENVIRONMENT_REBUILD_KIND_MAX_BYTES:
            raise ValueError(
                f"rebuild_record.kind exceeds {ENVIRONMENT_REBUILD_KIND_MAX_BYTES} bytes."
            )
        payload = copy_durable_json_object(self.payload, "rebuild_record.payload")
        canonical_durable_json_bytes(
            payload,
            "rebuild_record.payload",
            max_bytes=ENVIRONMENT_REBUILD_PAYLOAD_MAX_BYTES,
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "payload", payload)

    def to_payload(self) -> dict[str, Any]:
        """Return the durable JSON form recorded in session events."""

        return {"kind": self.kind, "payload": copy_durable_json_object(self.payload, "payload")}

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> EnvironmentRebuildRecord:
        """Read a record back from its durable JSON form."""

        if not isinstance(value, Mapping) or set(value) != {"kind", "payload"}:
            raise ValueError("Environment rebuild record must have exactly kind and payload.")
        return cls(kind=value["kind"], payload=value["payload"])


@dataclass(frozen=True, slots=True)
class EnvironmentRebuildRequest:
    """One environment to recreate, with the durable record that describes it.

    ``session_id`` is the session whose durable events carried ``record``: the
    session being forked or resumed, or the session it was forked or spawned
    from.
    """

    environment_name: str
    record: EnvironmentRebuildRecord
    session_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment_name",
            require_durable_clean_nonblank(self.environment_name, "environment_name"),
        )
        if type(self.record) is not EnvironmentRebuildRecord:
            raise TypeError("record must be an EnvironmentRebuildRecord.")
        object.__setattr__(
            self, "session_id", require_durable_clean_nonblank(self.session_id, "session_id")
        )


@dataclass(frozen=True, slots=True)
class EnvironmentRebuild:
    """A recreated registration: the spec, factory and optional artifact store.

    ``spec.name`` must equal the requested environment name, and the factory
    must declare the same ``rebuild_record`` it was rebuilt from, so a fork of
    the fork can be rebuilt the same way.

    If ``artifact_store.id`` is already in the application's inventory, Cayu
    uses that registered store. A rebuilder may return a fresh store object
    with the same stable id on each call.
    """

    spec: EnvironmentSpec
    factory: EnvironmentFactory
    artifact_store: ArtifactStore | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, EnvironmentSpec):
            raise TypeError("EnvironmentRebuild.spec must be an EnvironmentSpec.")
        if not isinstance(self.factory, EnvironmentFactory):
            raise TypeError("EnvironmentRebuild.factory must be an EnvironmentFactory.")
        if self.artifact_store is not None and not isinstance(self.artifact_store, ArtifactStore):
            raise TypeError("EnvironmentRebuild.artifact_store must be an ArtifactStore or None.")


class EnvironmentRebuildMismatch(RuntimeError):
    """A session would run in an environment built from a different rebuild record.

    Raised before any work when the factory registered under the session's
    environment name declares a rebuild record that differs from the one the
    session's own environment was built from, for example a slot name that a
    worker reused for another job's clone. Nothing changed; run the session
    where the name matches, or unregister the name once nothing uses it so it
    can be rebuilt from this session's record.
    """

    def __init__(self, session_id: str, environment_name: str, *, verified: bool = True) -> None:
        self.session_id = session_id
        self.environment_name = environment_name
        #: False when the session's own record could not be read, so whether it
        #: matches the registration is unknown; retry once the store answers.
        self.verified = verified
        detail = (
            "built from a different rebuild record than"
            if verified
            else "whose rebuild record could not be read to compare with"
        )
        super().__init__(
            f"Session {session_id} ran in environment {environment_name!r} {detail} the "
            "factory registered under that name here."
        )


EnvironmentRebuilder = Callable[[EnvironmentRebuildRequest], Awaitable[EnvironmentRebuild]]
"""Recreates the registration a rebuild record describes; called at most once per name."""


def declared_rebuild_record(factory: EnvironmentFactory) -> EnvironmentRebuildRecord | None:
    """Read and type-check a factory's opt-in rebuild declaration."""

    record = factory.rebuild_record
    if record is not None and type(record) is not EnvironmentRebuildRecord:
        raise TypeError("EnvironmentFactory.rebuild_record must be an EnvironmentRebuildRecord.")
    return record


__all__ = [
    "ENVIRONMENT_REBUILD_KIND_MAX_BYTES",
    "ENVIRONMENT_REBUILD_PAYLOAD_MAX_BYTES",
    "EnvironmentRebuild",
    "EnvironmentRebuildMismatch",
    "EnvironmentRebuildRecord",
    "EnvironmentRebuildRequest",
    "EnvironmentRebuilder",
]
