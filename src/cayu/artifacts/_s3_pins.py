"""Conditional S3 pin state that serializes durable pins with artifact deletion.

Each artifact that has ever been pinned or deleted owns one JSON state object at
``<prefix>/_pins/<artifact_id>.json``. Every change is a compare-and-swap:
creation uses ``If-None-Match: *`` and replacement uses ``If-Match`` on the
ETag that was read. The object is never removed by the store, and its revision
increases on every write, so an ETag can never reappear for a different state.

Each publication of an artifact identity is an immutable *generation*: its
content lives under a key unique to that generation, and its metadata object
names the generation in its body, so the metadata ETag is unique to that
publication even when the bytes are republished unchanged. A deletion targets
exact generations: it removes their generation-keyed content objects and removes
the metadata object only on a matching ETag. A deletion request that S3 applies
late can therefore never remove a later publication of the same identity; no
settlement depends on time.

The state holds pin owners and deletion fences. A fence names the generations
its deletion targets and is committed, with no owner present, before any
artifact object is removed. A pin is refused while any fence targets the
generation being pinned: an in-flight fence reads as absence, and an unresolved
one (a request whose outcome is unknown, which S3 may still apply) as
unavailable. A fence is dropped once its own call settled, or once any deletion
completes for every generation it targets, since generations are never reused.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from itertools import pairwise
from typing import Any

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    durable_json_object_from_pairs,
)
from cayu.artifacts.base import ArtifactStoreUnavailableError

MAX_PIN_OWNERS = 10_000
MAX_DELETION_FENCES = 64
MAX_FENCE_TARGETS = 64
# The generation of an artifact published before generation-keyed content.
LEGACY_GENERATION = "legacy"
_SCHEMA_VERSION = 3
_STATE_MAX_BYTES = 1024 * 1024
_MAX_ATTEMPTS = 16
_CONDITIONAL_CODES = frozenset({"PreconditionFailed", "ConditionalRequestConflict", "409", "412"})
_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
_STATE_KEYS = frozenset({"schema_version", "artifact_id", "revision", "owners", "deleters"})
_DELETER_KEYS = frozenset({"token", "targets", "unresolved"})
_HEX = frozenset("0123456789abcdef")

PINNED_MESSAGE = "Artifact is retained by a durable pin."


def _is_hex(value: Any, length: int) -> bool:
    return type(value) is str and len(value) == length and set(value) <= _HEX


def is_generation(value: Any) -> bool:
    return value == LEGACY_GENERATION or _is_hex(value, 32)


def _sorted_unique(values: Any, length: int, bound: int) -> bool:
    if type(values) is not list or len(values) > bound:
        return False
    items: list[str] = []
    for value in values:
        if not _is_hex(value, length):
            return False
        items.append(value)
    return all(left < right for left, right in pairwise(items))


def _valid_targets(values: Any) -> bool:
    if type(values) is not list or not 0 < len(values) <= MAX_FENCE_TARGETS:
        return False
    targets: list[str] = []
    for value in values:
        if not is_generation(value):
            return False
        targets.append(value)
    return all(left < right for left, right in pairwise(targets))


def _valid_deleters(values: Any) -> bool:
    if type(values) is not list or len(values) > MAX_DELETION_FENCES:
        return False
    tokens: list[str] = []
    for value in values:
        if (
            type(value) is not dict
            or set(value) != _DELETER_KEYS
            or not _is_hex(value["token"], 32)
            or not _valid_targets(value["targets"])
            or type(value["unresolved"]) is not bool
        ):
            return False
        tokens.append(value["token"])
    return all(left < right for left, right in pairwise(tokens))


def _tokens(state: dict[str, Any]) -> list[str]:
    return [deleter["token"] for deleter in state["deleters"]]


class S3ArtifactPinState:
    """Synchronous pin-state transitions; callers run them off the event loop."""

    def __init__(
        self, client: Any, *, bucket: str, prefix: str, encryption: dict[str, str]
    ) -> None:
        self.client = client
        self.bucket = bucket
        self.prefix = prefix
        self.encryption = dict(encryption)

    def key(self, artifact_id: str) -> str:
        return (self.prefix + "/" if self.prefix else "") + "_pins/" + artifact_id + ".json"

    @staticmethod
    def _empty(artifact_id: str) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "revision": 0,
            "owners": [],
            "deleters": [],
        }

    @staticmethod
    def _validate(artifact_id: str, state: Any) -> dict[str, Any]:
        if (
            type(state) is not dict
            or set(state) != _STATE_KEYS
            or type(state["schema_version"]) is not int
            or state["schema_version"] != _SCHEMA_VERSION
            or state["artifact_id"] != artifact_id
            or type(state["revision"]) is not int
            or not 1 <= state["revision"] <= MAX_DURABLE_JSON_INTEGER
            or not _sorted_unique(state["owners"], 64, MAX_PIN_OWNERS)
            or not _valid_deleters(state["deleters"])
        ):
            raise ValueError("Invalid S3 artifact pin state.")
        return state

    def read(self, artifact_id: str) -> tuple[dict[str, Any], str | None]:
        """Read the state and the ETag its next write is conditional on."""

        from cayu.artifacts.aws_s3 import _aws_error_code

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key(artifact_id))
        except Exception as error:
            if _aws_error_code(error) in _NOT_FOUND_CODES:
                return self._empty(artifact_id), None
            raise ArtifactStoreUnavailableError(
                "S3 artifact store could not read artifact pin state."
            ) from error
        body = response["Body"]
        failure: BaseException | None = None
        state: Any = None
        etag: Any = None
        try:
            length = response.get("ContentLength")
            if type(length) is not int or not 0 <= length <= _STATE_MAX_BYTES:
                raise ValueError("S3 artifact pin state exceeds its byte bound.")
            data = body.read(_STATE_MAX_BYTES + 1)
            if type(data) is not bytes or len(data) != length:
                raise ValueError("S3 artifact pin state read is incomplete.")
            etag = response.get("ETag")
            if (
                type(etag) is not str
                or not 0 < len(etag) <= 256
                or any(not 32 <= ord(c) < 127 for c in etag)
            ):
                raise ValueError("S3 artifact pin state has no conditional-write authority.")
            state = json.loads(
                data,
                object_pairs_hook=lambda pairs: durable_json_object_from_pairs(
                    pairs, "S3 artifact pin state"
                ),
            )
            self._validate(artifact_id, state)
        except BaseException as error:
            failure = error
        try:
            body.close()
        except BaseException as cleanup_error:
            if failure is not None and cleanup_error is not failure:
                raise BaseExceptionGroup(
                    "S3 artifact pin state read and cleanup failed.", [failure, cleanup_error]
                ) from None
            raise
        if failure is not None:
            if isinstance(failure, json.JSONDecodeError):
                raise ValueError("Invalid S3 artifact pin state.") from failure
            raise failure
        return state, etag

    def _transition(
        self,
        artifact_id: str,
        *,
        satisfied: Callable[[dict[str, Any]], bool],
        plan: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare-and-swap until ``satisfied`` holds; ``plan`` may refuse by raising."""

        from cayu.artifacts.aws_s3 import _aws_error_code

        for _ in range(_MAX_ATTEMPTS):
            state, etag = self.read(artifact_id)
            if satisfied(state):
                return state
            updated = plan(state)
            if state["revision"] >= MAX_DURABLE_JSON_INTEGER:
                raise ValueError("S3 artifact pin state revision is exhausted.")
            updated["revision"] = state["revision"] + 1
            self._validate(artifact_id, updated)
            encoded = canonical_durable_json_bytes(
                updated, "S3 artifact pin state", max_bytes=_STATE_MAX_BYTES
            )
            try:
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=self.key(artifact_id),
                    Body=encoded,
                    ContentType="application/json",
                    **({"IfNoneMatch": "*"} if etag is None else {"IfMatch": etag}),
                    **self.encryption,
                )
            except Exception as error:
                code = _aws_error_code(error)
                if code in _CONDITIONAL_CODES or (etag is not None and code in _NOT_FOUND_CODES):
                    # Another writer won, or a retried request already applied
                    # this change. Re-read and re-evaluate from the new state.
                    continue
                # A lost acknowledgement is accepted only after readback proves
                # the requested state holds.
                try:
                    observed, _ = self.read(artifact_id)
                except Exception as read_error:
                    raise ArtifactStoreUnavailableError(
                        "S3 artifact pin state publication could not be reconciled."
                    ) from ExceptionGroup(
                        "S3 artifact pin state publication and readback failed.",
                        [error, read_error],
                    )
                if satisfied(observed):
                    return observed
                raise ArtifactStoreUnavailableError(
                    "S3 artifact store could not publish artifact pin state."
                ) from error
            return updated
        raise ArtifactStoreUnavailableError("S3 artifact pin contention exceeded its retry bound.")

    def pin(self, artifact_id: str, owner_digest: str, require_present: Callable[[], str]) -> None:
        """Add an owner to the publication ``require_present`` proves is committed.

        ``require_present`` returns that publication's generation. It runs after
        the read this write is conditional on, so a deletion that begins or
        removes the artifact in between changes the state and forces a retry.
        """

        def satisfied(state: dict[str, Any]) -> bool:
            return owner_digest in state["owners"]

        def plan(state: dict[str, Any]) -> dict[str, Any]:
            generation = require_present()
            fences = [item for item in state["deleters"] if generation in item["targets"]]
            if any(not item["unresolved"] for item in fences):
                raise FileNotFoundError(f"Artifact not found: {artifact_id}")
            if fences:
                raise ArtifactStoreUnavailableError(
                    "An S3 artifact deletion of this publication has an unknown outcome and "
                    "may still be applied; delete the artifact again to settle it."
                )
            if len(state["owners"]) >= MAX_PIN_OWNERS:
                raise ValueError("Artifact pin owners exceed their bound.")
            return {**state, "owners": sorted([*state["owners"], owner_digest])}

        self._transition(artifact_id, satisfied=satisfied, plan=plan)

    def release(self, artifact_id: str, owner_digest: str) -> None:
        def satisfied(state: dict[str, Any]) -> bool:
            return owner_digest not in state["owners"]

        def plan(state: dict[str, Any]) -> dict[str, Any]:
            return {**state, "owners": [item for item in state["owners"] if item != owner_digest]}

        self._transition(artifact_id, satisfied=satisfied, plan=plan)

    def begin_deletion(self, artifact_id: str, token: str, targets: list[str]) -> None:
        """Commit a fence over ``targets`` before any artifact object is removed."""

        targets = sorted(set(targets))
        if not _is_hex(token, 32) or not _valid_targets(targets):
            raise ValueError("Invalid S3 artifact deletion fence.")

        def satisfied(state: dict[str, Any]) -> bool:
            return token in _tokens(state)

        def plan(state: dict[str, Any]) -> dict[str, Any]:
            if state["owners"]:
                raise ValueError(PINNED_MESSAGE)
            if len(state["deleters"]) >= MAX_DELETION_FENCES:
                raise ArtifactStoreUnavailableError(
                    "S3 artifact deletion fences exceed their bound; delete the artifact "
                    "again to settle unresolved deletions."
                )
            deleter = {"token": token, "targets": targets, "unresolved": False}
            deleters = sorted([*state["deleters"], deleter], key=lambda item: item["token"])
            return {**state, "deleters": deleters}

        self._transition(artifact_id, satisfied=satisfied, plan=plan)

    def end_deletion(
        self,
        artifact_id: str,
        token: str,
        targets: list[str],
        *,
        complete: bool,
        resolved: bool,
    ) -> None:
        """Settle this deletion's fence after its object removal returned.

        A resolved call (nothing sent, or exactly one attempt that S3 answered)
        retires its fence. An unresolved one stays, marked unresolved, because
        S3 may still apply it. A completed removal proves every targeted
        generation is gone for good, so it also drops every fence, including
        unresolved or abandoned ones, whose targets it covers.
        """

        targets = sorted(set(targets))
        if not _is_hex(token, 32) or not _valid_targets(targets):
            raise ValueError("Invalid S3 artifact deletion fence.")
        gone = set(targets) if complete else set()

        def keep(item: dict[str, Any]) -> bool:
            if gone and set(item["targets"]) <= gone:
                return False
            return not (resolved and item["token"] == token)

        def settled(state: dict[str, Any]) -> list[dict[str, Any]]:
            return [
                {**item, "unresolved": True} if item["token"] == token else item
                for item in state["deleters"]
                if keep(item)
            ]

        def satisfied(state: dict[str, Any]) -> bool:
            return settled(state) == state["deleters"]

        def plan(state: dict[str, Any]) -> dict[str, Any]:
            return {**state, "deleters": settled(state)}

        self._transition(artifact_id, satisfied=satisfied, plan=plan)
