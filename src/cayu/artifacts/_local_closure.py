"""Local artifact closure uses the same process-shared lock as publication."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import contextmanager
from uuid import uuid4

from cayu._filesystem_lock import cooperative_path_lock
from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.artifacts._closure import (
    ARTIFACT_CLOSURE_MAX_BYTES,
    ARTIFACT_CLOSURE_MAX_POLICY_BYTES,
    ARTIFACT_CLOSURE_MAX_RECORDS,
    ArtifactClosureClaim,
    ArtifactClosureItem,
    decode_artifact_closure_claim,
    encode_artifact_closure_claim,
)


@contextmanager
def closure_lock(root):
    with cooperative_path_lock(
        root, "publication-and-closure", lock_directory_name="cayu-artifact-closure-locks"
    ):
        yield


def _name(session_id):
    # Publication checks also serve artifact owners outside the narrower closure
    # API. Hash the exact owner without imposing the closure claim's ID limit.
    require_durable_clean_nonblank(session_id, "artifact session identity")
    return ".cayu-closure-" + hashlib.sha256(session_id.encode()).hexdigest() + ".json"


def load_claim(root, root_identity, store_id, session_id):
    from cayu.artifacts import local

    ArtifactClosureClaim(store_id, session_id, "0" * 64, ())
    name = _name(session_id)
    with local._open_store_root(root, root_identity) as root_fd:
        try:
            fd = local._open_artifact_file(
                root_fd, root, root_identity, name, missing_message="No closure claim"
            )
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as file:
            if os.fstat(file.fileno()).st_size > ARTIFACT_CLOSURE_MAX_BYTES:
                raise ValueError("Artifact closure record exceeds its byte bound.")
            claim = decode_artifact_closure_claim(file.read(ARTIFACT_CLOSURE_MAX_BYTES + 1))
        if claim.store_id != store_id or claim.session_id != session_id:
            raise ValueError("Artifact closure authority conflicts.")
        return claim


def metadata_digest(metadata):
    # Native callers supply a freshly parsed metadata record, never an extension model.
    return hashlib.sha256(
        canonical_durable_json_bytes(metadata.model_dump(mode="json"), "artifact closure metadata")
    ).hexdigest()


def _retire_pending_claims(root, root_identity, store_id, session_id):
    from cayu.artifacts import local

    prefix = _name(session_id) + ".staging-"
    count = 0
    with (
        local._open_store_root(root, root_identity) as root_fd,
        os.scandir(root_fd if root_fd is not None else root) as entries,
    ):
        for entry in entries:
            name = entry.name
            suffix = name.removeprefix(prefix)
            if (
                not name.startswith(prefix)
                or len(suffix) != 32
                or any(c not in "0123456789abcdef" for c in suffix)
            ):
                continue
            count += 1
            if count > 1000:
                raise ValueError("Artifact closure staging inventory exceeds its bound.")
            fd = local._open_artifact_file(
                root_fd,
                root,
                root_identity,
                name,
                missing_message="Closure staging disappeared",
            )
            with os.fdopen(fd, "rb") as file:
                info = os.fstat(file.fileno())
                if (
                    os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077
                ) or info.st_size > ARTIFACT_CLOSURE_MAX_BYTES:
                    raise ValueError("Artifact closure staging is not a private bounded record.")
                pending = decode_artifact_closure_claim(file.read(ARTIFACT_CLOSURE_MAX_BYTES + 1))
            if pending.store_id != store_id or pending.session_id != session_id:
                raise ValueError("Artifact closure staging authority conflicts.")
            current = local._stat_directory_entry(root / name, parent_fd=root_fd)
            if local._stat_identity(current) != local._stat_identity(info):
                raise ValueError("Artifact closure staging identity changed.")
            if root_fd is None:
                os.unlink(root / name)
            else:
                os.unlink(name, dir_fd=root_fd)
            local._sync_open_directory(root_fd, root, root_identity)


def claim_session(root, root_identity, store_id, session_id, plan_id, max_records, max_bytes):
    from cayu.artifacts import local
    from cayu.artifacts.base import ArtifactScope

    ArtifactClosureClaim(store_id, session_id, plan_id, ())
    if type(max_records) is not int or not 0 < max_records <= ARTIFACT_CLOSURE_MAX_RECORDS:
        raise ValueError("Invalid artifact closure record bound.")
    if type(max_bytes) is not int or not 0 < max_bytes <= ARTIFACT_CLOSURE_MAX_POLICY_BYTES:
        raise ValueError("Invalid artifact closure byte bound.")
    local._require_durable_publication_support()
    with closure_lock(root):
        _retire_pending_claims(root, root_identity, store_id, session_id)
        existing = load_claim(root, root_identity, store_id, session_id)
        if existing is not None:
            if existing.plan_id != plan_id:
                raise ValueError("Artifact closure plan conflicts.")
            if (
                len(existing.artifacts) > max_records
                or len(encode_artifact_closure_claim(existing)) > max_bytes
            ):
                raise ValueError("Artifact closure replay exceeds its requested bounds.")
            with local._open_store_root(root, root_identity) as root_fd:
                local._sync_open_directory(root_fd, root, root_identity)
            return existing
        listing = local._list_artifacts(
            root, root_identity, ArtifactScope.SESSION, session_id, None, None, max_records
        )
        if listing.truncated:
            raise ValueError("Artifact closure inventory is truncated.")
        claim = ArtifactClosureClaim(
            store_id,
            session_id,
            plan_id,
            tuple(
                sorted(
                    (
                        ArtifactClosureItem(item.id, item.size_bytes, metadata_digest(item))
                        for item in listing.artifacts
                    ),
                    key=lambda item: item.artifact_id,
                )
            ),
        )
        encoded = encode_artifact_closure_claim(claim)
        if len(encoded) > max_bytes:
            raise ValueError("Artifact closure claim exceeds its requested bound.")
        name = _name(session_id)
        staging = name + ".staging-" + uuid4().hex
        with local._open_store_root(root, root_identity) as root_fd:
            staged_identity = None

            def created(identity):
                nonlocal staged_identity
                staged_identity = identity

            failure = None
            try:
                local._write_artifact_file(
                    root_fd, root, root_identity, staging, encoded, on_created=created
                )
                local._rename_directory_no_replace(root / staging, root / name, parent_fd=root_fd)
                local._sync_open_directory(root_fd, root, root_identity)
            except BaseException as error:
                failure = error
            cleanup_failure = None
            if staged_identity is not None:
                try:
                    try:
                        remaining = local._stat_directory_entry(root / staging, parent_fd=root_fd)
                    except FileNotFoundError:
                        remaining = None
                    if remaining is not None:
                        if local._stat_identity(remaining) != staged_identity:
                            raise ValueError("Artifact closure staging identity changed.")
                        if root_fd is None:
                            os.unlink(root / staging)
                        else:
                            os.unlink(staging, dir_fd=root_fd)
                        local._sync_open_directory(root_fd, root, root_identity)
                except BaseException as error:
                    cleanup_failure = error
            if failure is not None and cleanup_failure is not None:
                if failure is cleanup_failure:
                    raise failure
                raise BaseExceptionGroup(
                    "Artifact closure publication and staging cleanup failed.",
                    [failure, cleanup_failure],
                )
            if failure is not None:
                raise failure
            if cleanup_failure is not None:
                raise cleanup_failure
        return claim


def require_publication_open(root, root_identity, artifact):
    from cayu.artifacts import local
    from cayu.artifacts.base import ArtifactScope

    if artifact.scope is ArtifactScope.SESSION:
        # Existence is sufficient to reject, even when the record is malformed:
        # a corrupt claim must never reopen publication.
        with local._open_store_root(root, root_identity) as root_fd:
            try:
                local._stat_directory_entry(root / _name(artifact.session_id), parent_fd=root_fd)
            except FileNotFoundError:
                return
        raise local._AbsentLocalArtifactError(ValueError("Artifact session is fenced for closure."))


def delete_claimed_artifact(root, root_identity, root_fd, store_id, claim, artifact_id):
    """Transfer an authenticated directory to durable cleanup ownership.

    The caller holds the ordinary artifact lock. Only this exact claim can name
    the staging directory; retries do not need metadata inside a partially
    removed directory and never follow a replacement at the original path.
    """
    from cayu.artifacts import local
    from cayu.artifacts.base import ArtifactScope

    if load_claim(root, root_identity, store_id, claim.session_id) != claim:
        raise ValueError("Artifact closure claim conflicts.")
    expected = next((item for item in claim.artifacts if item.artifact_id == artifact_id), None)
    if expected is None:
        raise ValueError("Artifact is not owned by the closure claim.")
    identity = hashlib.sha256(
        encode_artifact_closure_claim(claim) + b"\0" + artifact_id.encode()
    ).hexdigest()
    staged = root / (".cayu-closure-delete-" + identity)
    target = local._artifact_dir(root, artifact_id)
    try:
        with local._open_artifact_directory(staged, parent_fd=root_fd) as (_, directory_identity):
            pass
    except FileNotFoundError:
        try:
            with local._open_artifact_directory(target, parent_fd=root_fd) as (
                fd,
                original_identity,
            ):
                try:
                    actual = local._load_metadata_from_directory(target, fd, original_identity)
                except FileNotFoundError:
                    raise ValueError(
                        "Artifact closure found an incomplete unowned deletion."
                    ) from None
                if (
                    actual.scope is not ArtifactScope.SESSION
                    or actual.session_id != claim.session_id
                    or metadata_digest(actual) != expected.metadata_sha256
                ):
                    raise ValueError("Artifact closure item was replaced.")
                if any(
                    name.startswith("pin_") for name in os.listdir(fd if fd is not None else target)
                ):
                    raise ValueError("Artifact is retained by a durable pin.")
        except FileNotFoundError:
            # Synchronize even an absent retry: a prior removal may have lost
            # its directory-sync acknowledgement.
            local._sync_open_directory(root_fd, root, root_identity)
            return
        local._rename_directory_no_replace(target, staged, parent_fd=root_fd)
        local._sync_open_directory(root_fd, root, root_identity)
        with local._open_artifact_directory(staged, parent_fd=root_fd) as (_, directory_identity):
            if directory_identity != original_identity:
                raise ValueError("Artifact deletion staging identity changed.") from None
    # A previous rename may have committed before its synchronization failed.
    # Establish its durability before destroying any payload or metadata.
    local._sync_open_directory(root_fd, root, root_identity)
    local._remove_artifact_directory_if_unchanged(
        staged, directory_identity, parent_fd=root_fd, ignore_errors=False
    )
    try:
        local._stat_directory_entry(staged, parent_fd=root_fd)
    except FileNotFoundError:
        local._sync_open_directory(root_fd, root, root_identity)
        return
    raise ValueError("Artifact deletion staging remains unresolved.")
