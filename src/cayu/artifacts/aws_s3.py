from __future__ import annotations

import asyncio
import importlib
import json
import mimetypes
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from uuid import uuid4

from cayu._exception_groups import exception_cause, exception_context, set_exception_context
from cayu._resource_store_surface import model_store_surface
from cayu._task_wait import run_thread_to_completion
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_metadata,
    require_clean_nonblank,
    require_nonblank,
    require_unicode_scalar_text,
)
from cayu.artifacts._closure import (
    ARTIFACT_CLOSURE_MAX_POLICY_BYTES,
    ArtifactClosureClaim,
    ArtifactClosureItem,
    copy_artifact_closure_claim,
    decode_artifact_closure_claim,
    encode_artifact_closure_claim,
)
from cayu.artifacts._listing import BoundedArtifactListing
from cayu.artifacts._pin_owner import pin_owner_digest
from cayu.artifacts._s3_closure import S3ArtifactClosureGate
from cayu.artifacts._s3_pins import (
    LEGACY_GENERATION,
    MAX_FENCE_TARGETS,
    PINNED_MESSAGE,
    S3ArtifactPinState,
)
from cayu.artifacts._settlement import (
    _absent_artifact_write,
    _ArtifactWritePhaseReporter,
    _ArtifactWriteRegistry,
    _await_owned_sync_call,
    _committed_artifact_write,
    _settle_artifact_write,
    _unsettled_artifact_write,
)
from cayu.artifacts.access import runtime_artifact_operation
from cayu.artifacts.base import (
    ArtifactIdentityConflictError,
    ArtifactListResult,
    ArtifactMetadata,
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
    ArtifactStoreUnavailableError,
    InvalidArtifactIdError,
    _require_matching_artifact,
)
from cayu.artifacts.settlement import (
    ArtifactWriteSettlementFailureCode,
    ArtifactWriteSettlementPhase,
    ArtifactWriteSettlementStatus,
)

_ARTIFACT_ID_PATTERN = re.compile(r"\Aart_[0-9a-f]{32}\Z")
_GENERATION_CONTENT_PATTERN = re.compile(r"\A([0-9a-f]{32})/content\Z")
_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
# A metadata object written since generation-keyed content wraps the artifact
# metadata with its publication generation, so the metadata ETag is unique to
# one publication even when identical bytes are published again.
_PUBLICATION_FORMAT_KEY = "cayu_s3_publication"
_PUBLICATION_FORMAT = 1
_PUBLICATION_KEYS = frozenset({"artifact", _PUBLICATION_FORMAT_KEY, "generation"})


@model_store_surface("artifacts")
class S3ArtifactStore(ArtifactStore):
    """Direct S3 object implementation of ArtifactStore.

    Content is written first and metadata.json last. The metadata object is
    the commit marker, so interrupted writes are never listed as artifacts.
    Each publication writes its content under a key unique to that publication
    (``<artifact_id>/<generation>/content``) and names the generation in its
    metadata. Artifacts published before generations keep reading from
    ``<artifact_id>/content``.

    ``list`` scans every committed metadata object under ``prefix`` and fetches
    each metadata document before filtering, sorting, and applying ``limit``.
    This keeps totals exact but targets modest artifact volumes. High-volume
    deployments should maintain a separate query index.

    Durable pins live in one conditional-write state object per artifact under
    ``_pins/``. Pin, release, and deletion change it only by compare-and-swap,
    so pins serialize with deletion across processes without a lock service.
    Deletion removes exact generations and removes the metadata only on a
    matching ETag, so a deletion S3 applies late cannot remove a later
    publication of the same identity.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "cayu/artifacts",
        store_id: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        endpoint_url: str | None = None,
        kms_key_id: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.bucket = require_clean_nonblank(bucket, "bucket")
        if type(prefix) is not str:
            raise TypeError("S3ArtifactStore prefix must be a string.")
        self.prefix = prefix.strip("/")
        self._region_name = _optional_clean_string(region_name, "region_name")
        self._profile_name = _optional_clean_string(profile_name, "profile_name")
        self._endpoint_url = _optional_clean_string(endpoint_url, "endpoint_url")
        self._kms_key_id = _optional_clean_string(kms_key_id, "kms_key_id")
        if client is not None and (
            self._profile_name is not None or self._endpoint_url is not None
        ):
            raise ValueError(
                "An injected client cannot be combined with profile_name or endpoint_url."
            )
        self._client = client
        self._client_lock = asyncio.Lock()
        self._write_registry = _ArtifactWriteRegistry()
        default_id = f"s3://{self.bucket}/{self.prefix}" if self.prefix else f"s3://{self.bucket}"
        value = default_id if store_id is None else require_clean_nonblank(store_id, "store_id")
        self.id = require_unicode_scalar_text(value, "store_id")

    artifact_access_version = 1

    @runtime_artifact_operation("create")
    async def put_bytes(
        self,
        content: bytes,
        *,
        artifact_id: str | None = None,
        filename: str,
        content_type: str | None = None,
        scope: ArtifactScope = ArtifactScope.SESSION,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactMetadata:
        if type(content) is not bytes:
            raise TypeError("Artifact content must be bytes.")
        filename = require_nonblank(filename, "filename")
        resolved_content_type = content_type or (
            mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )
        resolved_content_type = require_clean_nonblank(resolved_content_type, "content_type")
        validated_scope = _validate_scope(scope)
        session_id = _optional_identifier(session_id, "session_id")
        agent_name = _optional_identifier(agent_name, "agent_name")
        environment_name = _optional_identifier(environment_name, "environment_name")
        _validate_scope_owner(
            validated_scope,
            session_id=session_id,
            environment_name=environment_name,
        )
        resolved_artifact_id = (
            f"art_{uuid4().hex}" if artifact_id is None else _validate_artifact_id(artifact_id)
        )
        from cayu.artifacts.access import creation_labels

        artifact = ArtifactMetadata(
            labels=creation_labels(),
            id=resolved_artifact_id,
            filename=filename,
            content_type=resolved_content_type,
            size_bytes=len(content),
            scope=validated_scope,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=copy_durable_metadata({} if metadata is None else metadata, "metadata"),
        )
        return await _settle_artifact_write(
            registry=self._write_registry,
            store_id=self.id,
            artifact_id=artifact.id,
            operation_name="S3 artifact publication",
            operation=lambda reporter: self._run_artifact_write(
                reporter,
                artifact=artifact,
                content=content,
                supplied_identity=artifact_id is not None,
            ),
        )

    async def _run_artifact_write(self, reporter, *, artifact, content, supplied_identity):
        gate = None
        reserved = False
        token = uuid4().hex
        request_digest = sha256(
            canonical_durable_json_bytes(
                {
                    "artifact": artifact.model_dump(mode="json"),
                    "content_sha256": sha256(content).hexdigest(),
                },
                "S3 artifact write intent",
            )
        ).hexdigest()
        try:
            client = await self._get_client(reporter=reporter)
            if artifact.scope is ArtifactScope.SESSION:
                gate = S3ArtifactClosureGate(
                    client,
                    bucket=self.bucket,
                    prefix=self.prefix,
                    store_id=self.id,
                    session_id=artifact.session_id,
                    encryption=self._encryption_options(),
                )
                await _run_s3_sync_call(reporter, gate.reserve, token, artifact.id, request_digest)
                reserved = True
            await _run_s3_sync_call(reporter, self._bind_artifact_owner, client, artifact)
        except BaseException as error:
            if reserved and gate is not None:
                # The owner check precedes every content/metadata upload. Its
                # failure cannot leave a business mutation running, so this
                # exact reservation can be retired once the guard settles.
                try:
                    await _run_s3_sync_call(
                        reporter, gate.release, token, artifact.id, request_digest
                    )
                except BaseException as cleanup_error:
                    error = _combined_s3_write_failure(
                        "S3 artifact admission and reservation cleanup failed.",
                        primary=error,
                        reconciliation=cleanup_error,
                    )
                else:
                    return _absent_artifact_write(
                        error,
                        phase=ArtifactWriteSettlementPhase.PRE_DISPATCH,
                        failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
                    )
            elif gate is None:
                return _absent_artifact_write(
                    error,
                    phase=ArtifactWriteSettlementPhase.PRE_DISPATCH,
                    failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
                )
            # A reserve acknowledgement may have been lost. No artifact upload
            # starts without positive admission; retained gate state stays fenced.
            return _unsettled_artifact_write(
                error,
                phase=ArtifactWriteSettlementPhase.PRE_DISPATCH,
                failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
            )
        try:
            outcome = await self._run_reserved_artifact_write(
                reporter, artifact=artifact, content=content, supplied_identity=supplied_identity
            )
        except BaseException as error:
            return _unsettled_artifact_write(
                error,
                phase=reporter.phase,
                failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
            )
        if gate is not None and outcome.status in {
            ArtifactWriteSettlementStatus.COMMITTED,
            ArtifactWriteSettlementStatus.ABSENT,
        }:
            try:
                await _run_s3_sync_call(reporter, gate.release, token, artifact.id, request_digest)
            except BaseException as error:
                if outcome.error is not None and outcome.error is not error:
                    error = _combined_s3_write_failure(
                        "S3 artifact write and reservation settlement failed.",
                        primary=outcome.error,
                        reconciliation=error,
                    )
                return _unsettled_artifact_write(
                    error,
                    phase=ArtifactWriteSettlementPhase.RECONCILIATION,
                    failure_codes=(ArtifactWriteSettlementFailureCode.RECONCILIATION_FAILED,),
                )
        return outcome

    def _bind_artifact_owner(self, client, artifact):
        key = (
            (self.prefix + "/" if self.prefix else "")
            + "_closure/artifact-owners/"
            + artifact.id
            + ".json"
        )
        encoded = canonical_durable_json_bytes(
            {
                "artifact_id": artifact.id,
                "scope": artifact.scope.value,
                "session_id": artifact.session_id,
                "environment_name": artifact.environment_name
                if artifact.scope is ArtifactScope.ENVIRONMENT
                else None,
            },
            "S3 artifact ownership",
        )
        try:
            client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=encoded,
                ContentType="application/json",
                IfNoneMatch="*",
                **self._encryption_options(),
            )
        except Exception:
            response = client.get_object(Bucket=self.bucket, Key=key)
            observed = _response_body_bytes(response, len(encoded) + 1)
            if observed != encoded:
                raise ArtifactIdentityConflictError(
                    "Artifact identity already exists with different content or metadata."
                ) from None

    def _closure_gate(self, client, session_id):
        return S3ArtifactClosureGate(
            client,
            bucket=self.bucket,
            prefix=self.prefix,
            store_id=self.id,
            session_id=session_id,
            encryption=self._encryption_options(),
        )

    @property
    def supports_session_closure_claims(self) -> bool:
        return True

    async def load_session_closure_claim(self, session_id: str) -> ArtifactClosureClaim | None:
        ArtifactClosureClaim(self.id, session_id, "0" * 64, ())
        client = await self._get_client()
        state, _ = await _run_s3_sync_call(None, self._closure_gate(client, session_id).read)
        if state["claim"] is None:
            return None
        return decode_artifact_closure_claim(
            canonical_durable_json_bytes(
                state["claim"], "S3 artifact closure claim", max_bytes=16 * 1024 * 1024
            )
        )

    async def _require_closure_owner(self, client, artifact_id, session_id):
        key = (
            (self.prefix + "/" if self.prefix else "")
            + "_closure/artifact-owners/"
            + artifact_id
            + ".json"
        )
        expected = canonical_durable_json_bytes(
            {
                "artifact_id": artifact_id,
                "scope": "session",
                "session_id": session_id,
                "environment_name": None,
            },
            "S3 artifact owner",
        )
        _, value = await _run_s3_sync_call(
            None, _get_object_content, client, len(expected) + 1, Bucket=self.bucket, Key=key
        )
        if value != expected:
            raise ValueError("S3 artifact closure ownership conflicts.")

    async def claim_session_closure(
        self, session_id: str, plan_id: str, *, max_records: int, max_bytes: int
    ) -> ArtifactClosureClaim:
        ArtifactClosureClaim(self.id, session_id, plan_id, ())
        if type(max_records) is not int or not 0 < max_records <= 100_000:
            raise ValueError("Invalid artifact closure record bound.")
        if type(max_bytes) is not int or not 0 < max_bytes <= ARTIFACT_CLOSURE_MAX_POLICY_BYTES:
            raise ValueError("Invalid artifact closure byte bound.")
        client = await self._get_client()
        gate = self._closure_gate(client, session_id)
        await _run_s3_sync_call(None, gate.retire_settled)
        state, _ = await _run_s3_sync_call(None, gate.read)
        if state["claim"] is not None:
            claim = decode_artifact_closure_claim(
                canonical_durable_json_bytes(
                    state["claim"], "S3 artifact closure claim", max_bytes=16 * 1024 * 1024
                )
            )
            if claim.plan_id != plan_id:
                raise ValueError("S3 artifact closure plan conflicts.")
        else:
            if state["active"]:
                raise ValueError("S3 artifact publication has not quiesced.")
            listing = await self.list(
                scope=ArtifactScope.SESSION, session_id=session_id, limit=max_records
            )
            if listing.truncated:
                raise ValueError("S3 artifact closure inventory is truncated.")
            items = []
            for artifact in listing.artifacts:
                if artifact.scope is not ArtifactScope.SESSION or artifact.session_id != session_id:
                    raise ValueError("S3 artifact closure inventory ownership conflicts.")
                await self._require_closure_owner(client, artifact.id, session_id)
                items.append(
                    ArtifactClosureItem(
                        artifact.id,
                        artifact.size_bytes,
                        sha256(
                            canonical_durable_json_bytes(
                                artifact.model_dump(mode="json"), "artifact closure metadata"
                            )
                        ).hexdigest(),
                    )
                )
            claim = ArtifactClosureClaim(
                self.id,
                session_id,
                plan_id,
                tuple(sorted(items, key=lambda item: item.artifact_id)),
            )
        if (
            len(claim.artifacts) > max_records
            or len(encode_artifact_closure_claim(claim)) > max_bytes
        ):
            raise ValueError("S3 artifact closure claim exceeds its requested bounds.")
        await _run_s3_sync_call(None, gate.seal, claim, expected_revision=state["revision"])
        return claim

    async def delete_session_closure_artifact(
        self, claim: ArtifactClosureClaim, artifact_id: str
    ) -> None:
        claim = copy_artifact_closure_claim(claim)
        if claim.store_id != self.id:
            raise ValueError("S3 artifact closure store conflicts.")
        expected = next((item for item in claim.artifacts if item.artifact_id == artifact_id), None)
        if expected is None:
            raise ValueError("Artifact is not owned by the closure claim.")
        if await self.load_session_closure_claim(claim.session_id) != claim:
            raise ValueError("S3 artifact closure claim conflicts.")
        client = await self._get_client()
        await self._require_closure_owner(client, artifact_id, claim.session_id)
        try:
            metadata = await self._read_metadata(client, artifact_id)
        except FileNotFoundError:
            pass
        else:
            actual = sha256(
                canonical_durable_json_bytes(
                    metadata.model_dump(mode="json"), "artifact closure metadata"
                )
            ).hexdigest()
            if actual != expected.metadata_sha256:
                raise ValueError("S3 artifact closure item was replaced.")
        # The permanent owner binding and sealed session exclude future writers,
        # including attempts to reuse this ID from another scope or session.
        # Durable pins still retain the item, exactly as for ordinary deletion.
        await _run_s3_sync_call(None, self._delete_fenced, client, artifact_id)

    async def _run_reserved_artifact_write(
        self,
        reporter: _ArtifactWritePhaseReporter,
        *,
        artifact: ArtifactMetadata,
        content: bytes,
        supplied_identity: bool,
    ):
        try:
            client = await self._get_client(reporter=reporter)
        except BaseException as error:
            return _absent_artifact_write(
                error,
                phase=ArtifactWriteSettlementPhase.PRE_DISPATCH,
                failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
            )

        if supplied_identity:
            try:
                existing = await self._read_complete_artifact(
                    client,
                    artifact.id,
                    reporter=reporter,
                )
            except FileNotFoundError:
                pass
            except BaseException as error:
                return _absent_artifact_write(
                    error,
                    phase=ArtifactWriteSettlementPhase.PRE_DISPATCH,
                    failure_codes=(ArtifactWriteSettlementFailureCode.RECONCILIATION_FAILED,),
                )
            else:
                try:
                    _require_matching_artifact(existing, expected=artifact, content=content)
                except BaseException as error:
                    return _absent_artifact_write(
                        error,
                        phase=ArtifactWriteSettlementPhase.PRE_DISPATCH,
                        failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
                    )
                return _committed_artifact_write(existing.metadata)

        try:
            _require_single_attempt_content_writes(client)
        except S3ArtifactClientConfigurationError as error:
            return _absent_artifact_write(
                error,
                phase=ArtifactWriteSettlementPhase.PRE_DISPATCH,
                failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
            )
        metadata_key = self._artifact_key(artifact.id, "metadata.json")
        # A content PUT is sent exactly once per generation. An attempt whose
        # outcome is unknown may still be applied later, possibly after a
        # deletion has swept its generation, so its generation is abandoned and
        # never referenced; the next attempt writes a fresh generation. Only a
        # generation whose single attempt S3 acknowledged is ever committed.
        reconciled_failure_codes: tuple[ArtifactWriteSettlementFailureCode, ...] = ()
        generation: str | None = None
        content_error: BaseException | None = None
        for _attempt in range(_CONTENT_PUBLICATION_ATTEMPTS):
            candidate = uuid4().hex
            submission = _ContentSubmission()
            reporter.set(ArtifactWriteSettlementPhase.CONTENT)
            try:
                await _run_s3_sync_call(
                    reporter,
                    submission.send,
                    client,
                    Bucket=self.bucket,
                    Key=self._content_key(artifact.id, candidate),
                    Body=content,
                    ContentType=artifact.content_type,
                    IfNoneMatch="*",
                    **self._encryption_options(),
                )
            except BaseException as error:
                if not issubclass(type(error), Exception):
                    return _unsettled_artifact_write(
                        error,
                        phase=ArtifactWriteSettlementPhase.CONTENT,
                        failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
                    )
                if submission.unapplied(error):
                    # Nothing was sent, or S3 refused the only attempt.
                    return _absent_artifact_write(
                        error,
                        phase=ArtifactWriteSettlementPhase.CONTENT,
                        failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
                    )
                content_error = error
                reconciled_failure_codes = (ArtifactWriteSettlementFailureCode.MUTATION_FAILED,)
                continue
            generation = candidate
            break
        if generation is None:
            assert content_error is not None
            failure = ArtifactStoreUnavailableError(
                "S3 artifact store could not write artifact content."
            )
            failure.__cause__ = content_error
            return _unsettled_artifact_write(
                failure,
                phase=ArtifactWriteSettlementPhase.CONTENT,
                failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
            )

        reporter.set(ArtifactWriteSettlementPhase.COMMIT)
        try:
            await _run_s3_sync_call(
                reporter,
                client.put_object,
                Bucket=self.bucket,
                Key=metadata_key,
                Body=_publication_body(artifact, generation),
                ContentType="application/json",
                IfNoneMatch="*",
                **self._encryption_options(),
            )
        except BaseException as metadata_error:
            if not issubclass(type(metadata_error), Exception):
                return _unsettled_artifact_write(
                    metadata_error,
                    phase=ArtifactWriteSettlementPhase.COMMIT,
                    failure_codes=(ArtifactWriteSettlementFailureCode.COMMIT_FAILED,),
                )
            reporter.set(ArtifactWriteSettlementPhase.RECONCILIATION)
            try:
                existing = await self._read_complete_artifact(
                    client,
                    artifact.id,
                    reporter=reporter,
                )
            except BaseException as reconciliation_error:
                failure = _combined_s3_write_failure(
                    "S3 artifact store could not commit artifact metadata.",
                    primary=metadata_error,
                    reconciliation=reconciliation_error,
                )
                return _unsettled_artifact_write(
                    failure,
                    phase=ArtifactWriteSettlementPhase.RECONCILIATION,
                    failure_codes=(
                        ArtifactWriteSettlementFailureCode.COMMIT_FAILED,
                        ArtifactWriteSettlementFailureCode.RECONCILIATION_FAILED,
                    ),
                )
            try:
                _require_matching_artifact(existing, expected=artifact, content=content)
            except ValueError as conflict:
                conflict.__cause__ = metadata_error
                return _unsettled_artifact_write(
                    conflict,
                    phase=ArtifactWriteSettlementPhase.RECONCILIATION,
                    failure_codes=tuple(
                        dict.fromkeys(
                            (
                                *reconciled_failure_codes,
                                ArtifactWriteSettlementFailureCode.COMMIT_FAILED,
                            )
                        )
                    ),
                )
            return _committed_artifact_write(
                existing.metadata,
                failure_codes=tuple(
                    dict.fromkeys(
                        (
                            *reconciled_failure_codes,
                            ArtifactWriteSettlementFailureCode.COMMIT_FAILED,
                        )
                    )
                ),
            )
        return _committed_artifact_write(
            artifact,
            failure_codes=reconciled_failure_codes,
        )

    @runtime_artifact_operation("read")
    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        artifact_id = _validate_artifact_id(artifact_id)
        limit = _validate_limit(max_bytes, "max_bytes")
        client = await self._get_client()
        metadata, generation = await self._read_publication(client, artifact_id)
        get_options: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self._content_key(artifact_id, generation),
        }
        if limit is not None and limit < metadata.size_bytes:
            get_options["Range"] = f"bytes=0-{limit - 1}"
        try:
            _, content = await run_thread_to_completion(
                _get_object_content, client, None, **get_options
            )
        except Exception as exc:
            if _aws_error_code(exc) in _NOT_FOUND_CODES:
                raise FileNotFoundError(f"Artifact not found: {artifact_id}") from exc
            raise ArtifactStoreUnavailableError(
                "S3 artifact store could not read artifact content."
            ) from exc
        if limit is not None and len(content) > limit:
            raise ArtifactStoreUnavailableError(
                "S3 artifact store returned content beyond the requested byte limit."
            )
        if limit is None and len(content) != metadata.size_bytes:
            raise ArtifactStoreUnavailableError(
                "S3 artifact content size did not match committed metadata."
            )
        return ArtifactReadResult(
            metadata=metadata,
            content=content,
            total_bytes=metadata.size_bytes,
            truncated=len(content) < metadata.size_bytes,
        )

    @runtime_artifact_operation("read")
    async def read_range(
        self, artifact_id: str, *, offset: int, max_bytes: int
    ) -> ArtifactReadResult:
        artifact_id = _validate_artifact_id(artifact_id)
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer.")
        limit = _validate_limit(max_bytes, "max_bytes")
        if limit is None:
            raise ValueError("max_bytes is required for range reads.")
        client = await self._get_client()
        metadata, generation = await self._read_publication(client, artifact_id)
        content_key = self._content_key(artifact_id, generation)
        if offset > metadata.size_bytes:
            raise ValueError("offset exceeds artifact size.")
        length = min(limit, metadata.size_bytes - offset)
        content = b""
        try:
            if length:
                response, content = await run_thread_to_completion(
                    _get_object_content,
                    client,
                    length,
                    Bucket=self.bucket,
                    Key=content_key,
                    Range=f"bytes={offset}-{offset + length - 1}",
                )
                if (
                    response.get("ContentLength") != length
                    or response.get("ContentRange")
                    != f"bytes {offset}-{offset + length - 1}/{metadata.size_bytes}"
                    or len(content) != length
                ):
                    raise ValueError("S3 range length did not match committed metadata.")
            else:
                response = await run_thread_to_completion(
                    client.head_object,
                    Bucket=self.bucket,
                    Key=content_key,
                )
                if response.get("ContentLength") != metadata.size_bytes:
                    raise ValueError("S3 artifact size did not match committed metadata.")
        except Exception as exc:
            if _aws_error_code(exc) in _NOT_FOUND_CODES:
                raise FileNotFoundError(f"Artifact not found: {artifact_id}") from exc
            raise ArtifactStoreUnavailableError("S3 artifact range read failed.") from exc
        return ArtifactReadResult(
            metadata=metadata,
            content=content,
            total_bytes=metadata.size_bytes,
            truncated=offset + len(content) < metadata.size_bytes,
            offset=offset,
        )

    @runtime_artifact_operation("list")
    async def list(
        self,
        *,
        scope: ArtifactScope | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        limit: int | None = None,
    ) -> ArtifactListResult:
        validated_scope = _validate_scope(scope) if scope is not None else None
        session_id = _optional_identifier(session_id, "session_id")
        agent_name = _optional_identifier(agent_name, "agent_name")
        environment_name = _optional_identifier(environment_name, "environment_name")
        validated_limit = _validate_limit(limit, "limit")
        client = await self._get_client()
        try:
            inventory = BoundedArtifactListing(validated_limit)
            async for artifact_id in self._iter_metadata_ids(client):
                try:
                    artifact = await self._read_metadata(client, artifact_id)
                except FileNotFoundError:
                    await self._require_no_surviving_content(client, artifact_id)
                    continue
                if validated_scope is not None and artifact.scope != validated_scope:
                    continue
                if session_id is not None and artifact.session_id != session_id:
                    continue
                if agent_name is not None and artifact.agent_name != agent_name:
                    continue
                if environment_name is not None and artifact.environment_name != environment_name:
                    continue
                from cayu.artifacts.access import visible

                if visible(artifact):
                    inventory.add(artifact)
        except ArtifactStoreUnavailableError:
            raise
        except Exception as exc:
            raise ArtifactStoreUnavailableError(
                "S3 artifact store could not list artifacts."
            ) from exc
        return inventory.result()

    @runtime_artifact_operation("delete")
    async def delete(self, artifact_id: str) -> None:
        artifact_id = _validate_artifact_id(artifact_id)
        client = await self._get_client()
        from cayu.artifacts.access import _active

        if _active.get() is not None:
            await self._read_metadata(client, artifact_id)
        try:
            await asyncio.to_thread(self._delete_fenced, client, artifact_id)
        except (ArtifactStoreUnavailableError, ValueError):
            raise
        except Exception as exc:
            raise ArtifactStoreUnavailableError(
                "S3 artifact store could not delete artifact content."
            ) from exc

    @property
    def supports_pins(self) -> bool:
        return True

    async def pin(self, artifact_id: str, *, owner: str) -> None:
        owner_digest = pin_owner_digest(owner)
        artifact_id = _validate_artifact_id(artifact_id)
        client = await self._get_client()
        state = self._pin_state(client)
        await asyncio.to_thread(
            state.pin,
            artifact_id,
            owner_digest,
            lambda: self._require_committed_publication(client, artifact_id),
        )

    async def release_pin(self, artifact_id: str, *, owner: str) -> None:
        owner_digest = pin_owner_digest(owner)
        artifact_id = _validate_artifact_id(artifact_id)
        client = await self._get_client()
        await asyncio.to_thread(self._release_pin, client, artifact_id, owner_digest)

    def _pin_state(self, client: Any) -> S3ArtifactPinState:
        return S3ArtifactPinState(
            client, bucket=self.bucket, prefix=self.prefix, encryption=self._encryption_options()
        )

    def _release_pin(self, client: Any, artifact_id: str, owner_digest: str) -> None:
        if self._publication(client, artifact_id) is None:
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        self._pin_state(client).release(artifact_id, owner_digest)

    def _require_committed_publication(self, client: Any, artifact_id: str) -> str:
        """Prove the committed metadata and its content exist; return their generation."""

        publication = self._publication(client, artifact_id)
        if publication is None:
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        try:
            present = self._object_exists(
                client, self._content_key(artifact_id, publication.generation)
            )
        except Exception as exc:
            raise ArtifactStoreUnavailableError(
                "S3 artifact store could not read artifact content."
            ) from exc
        if not present:
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        return publication.generation

    def _publication(self, client: Any, artifact_id: str) -> _Publication | None:
        """Read the committed metadata with the ETag that identifies its publication."""

        try:
            response = client.get_object(
                Bucket=self.bucket, Key=self._artifact_key(artifact_id, "metadata.json")
            )
            etag = response.get("ETag") if isinstance(response, Mapping) else None
            payload = _response_body_bytes(response)
        except Exception as exc:
            if _aws_error_code(exc) in _NOT_FOUND_CODES:
                return None
            raise ArtifactStoreUnavailableError(
                "S3 artifact store could not read artifact metadata."
            ) from exc
        if type(etag) is not str or not etag:
            raise ArtifactStoreUnavailableError(
                "S3 artifact metadata has no ETag to condition its deletion on."
            )
        _, generation = _parse_publication(artifact_id, payload)
        return _Publication(generation=generation, etag=etag)

    def _content_generations(self, client: Any, artifact_id: str) -> dict[str, str]:
        """Map each content object stored for this identity, by generation, to its key."""

        prefix = self._artifact_key(artifact_id, "")
        found: dict[str, str] = {}
        continuation: str | None = None
        while True:
            options: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix, "MaxKeys": 1000}
            if continuation is not None:
                options["ContinuationToken"] = continuation
            response = client.list_objects_v2(**options)
            if not isinstance(response, Mapping) or type(response.get("IsTruncated")) is not bool:
                raise ArtifactStoreUnavailableError(
                    "S3 artifact store received an invalid list response."
                )
            for entry in response.get("Contents", []) or []:
                key = entry.get("Key") if isinstance(entry, Mapping) else None
                if type(key) is not str or not key.startswith(prefix):
                    continue
                relative = key[len(prefix) :]
                if relative == "content":
                    found[LEGACY_GENERATION] = key
                    continue
                match = _GENERATION_CONTENT_PATTERN.fullmatch(relative)
                if match is not None:
                    found[match.group(1)] = key
            if not response["IsTruncated"]:
                return found
            continuation_value = response.get("NextContinuationToken")
            if (
                type(continuation_value) is not str
                or not continuation_value
                or continuation_value == continuation
            ):
                raise ArtifactStoreUnavailableError(
                    "S3 artifact store list response omitted continuation token."
                )
            continuation = continuation_value

    def _delete_fenced(self, client: Any, artifact_id: str) -> None:
        """Delete exact publication generations under a committed pin-state fence.

        The current metadata is removed only on its ETag, and content objects
        are removed by generation-unique keys. Content left by an interrupted
        publication or an earlier partial deletion is swept with them. Runs to
        completion in one worker thread, so caller cancellation cannot abandon a
        fence between object removal and its settlement.
        """

        pins = self._pin_state(client)
        state, _ = pins.read(artifact_id)
        if state["owners"]:
            raise ValueError(PINNED_MESSAGE)
        publication = self._publication(client, artifact_id)
        content = self._content_generations(client, artifact_id)
        if publication is not None:
            content[publication.generation] = self._content_key(artifact_id, publication.generation)
        # Target the generations of earlier fences too. Removing an absent
        # generation-keyed object is harmless, and completing it settles those
        # fences, including one whose request S3 already applied.
        for deleter in state["deleters"]:
            for generation in deleter["targets"]:
                content.setdefault(generation, self._content_key(artifact_id, generation))
        if not content:
            # Nothing to remove: linearize as a no-op without writing state.
            return
        # Bound one fence; content beyond it is swept by a later deletion.
        swept = sorted(content)
        if publication is not None:
            swept.remove(publication.generation)
            swept = [publication.generation, *swept[: MAX_FENCE_TARGETS - 1]]
        else:
            swept = swept[:MAX_FENCE_TARGETS]
        targets = sorted(swept)
        metadata_key = self._artifact_key(artifact_id, "metadata.json")
        objects: list[dict[str, str]] = [{"Key": content[generation]} for generation in targets]
        if publication is not None:
            objects.append({"Key": metadata_key, "ETag": publication.etag})
        token = uuid4().hex
        pins.begin_deletion(artifact_id, token, targets)
        submission = _DeletionSubmission()
        try:
            response = submission.send(client, self.bucket, objects)
            _require_complete_deletion(response, conditional_key=metadata_key)
        except Exception as error:
            # A transport failure does not prove S3 will not still apply the
            # request, so such a fence stays until a completed deletion of the
            # same generations settles it.
            try:
                pins.end_deletion(
                    artifact_id,
                    token,
                    targets,
                    complete=False,
                    resolved=submission.resolved(error),
                )
            except Exception as settlement_error:
                failure = _combined_s3_write_failure(
                    "S3 artifact deletion and fence settlement failed.",
                    primary=error,
                    reconciliation=settlement_error,
                )
                raise failure from failure.__cause__
            raise
        pins.end_deletion(
            artifact_id, token, targets, complete=True, resolved=submission.resolved(None)
        )

    def _object_exists(self, client: Any, key: str) -> bool:
        try:
            client.head_object(Bucket=self.bucket, Key=key)
        except Exception as error:
            if _aws_error_code(error) in _NOT_FOUND_CODES:
                return False
            raise
        return True

    async def _read_metadata(
        self,
        client: Any,
        artifact_id: str,
        *,
        reporter: _ArtifactWritePhaseReporter | None = None,
    ) -> ArtifactMetadata:
        metadata, _ = await self._read_publication(client, artifact_id, reporter=reporter)
        return metadata

    async def _read_publication(
        self,
        client: Any,
        artifact_id: str,
        *,
        reporter: _ArtifactWritePhaseReporter | None = None,
    ) -> tuple[ArtifactMetadata, str]:
        """Read the committed metadata and the generation that holds its content."""

        try:
            _, payload = await _run_s3_sync_call(
                reporter,
                _get_object_content,
                client,
                None,
                Bucket=self.bucket,
                Key=self._artifact_key(artifact_id, "metadata.json"),
            )
        except Exception as exc:
            if _aws_error_code(exc) in _NOT_FOUND_CODES:
                raise FileNotFoundError(f"Artifact not found: {artifact_id}") from exc
            raise ArtifactStoreUnavailableError(
                "S3 artifact store could not read artifact metadata."
            ) from exc
        metadata, generation = _parse_publication(artifact_id, payload)
        from cayu.artifacts.access import require_artifact

        require_artifact(metadata)
        return metadata, generation

    async def _read_object_bytes(
        self,
        client: Any,
        key: str,
        *,
        reporter: _ArtifactWritePhaseReporter | None = None,
    ) -> bytes:
        _, content = await _run_s3_sync_call(
            reporter, _get_object_content, client, None, Bucket=self.bucket, Key=key
        )
        return content

    async def _read_complete_artifact(
        self,
        client: Any,
        artifact_id: str,
        *,
        reporter: _ArtifactWritePhaseReporter | None = None,
    ) -> ArtifactReadResult:
        metadata, generation = await self._read_publication(client, artifact_id, reporter=reporter)
        try:
            content = await self._read_object_bytes(
                client,
                self._content_key(artifact_id, generation),
                reporter=reporter,
            )
        except Exception as exc:
            if _aws_error_code(exc) in _NOT_FOUND_CODES:
                raise FileNotFoundError(f"Artifact not found: {artifact_id}") from exc
            raise ArtifactStoreUnavailableError(
                "S3 artifact store could not read artifact content."
            ) from exc
        if len(content) != metadata.size_bytes:
            raise ArtifactStoreUnavailableError(
                "S3 artifact content size did not match committed metadata."
            )
        return ArtifactReadResult(
            metadata=metadata,
            content=content,
            total_bytes=metadata.size_bytes,
            truncated=False,
        )

    async def _require_no_surviving_content(self, client: Any, artifact_id: str) -> None:
        """Fail when content of any generation survives without committed metadata."""

        content = await _run_s3_sync_call(None, self._content_generations, client, artifact_id)
        for key in content.values():
            try:
                await _run_s3_sync_call(None, client.head_object, Bucket=self.bucket, Key=key)
            except Exception as error:
                if _aws_error_code(error) in _NOT_FOUND_CODES:
                    continue
                raise
            raise ArtifactStoreUnavailableError(
                "S3 artifact inventory contains content without ownership metadata."
            )

    async def _iter_metadata_ids(self, client: Any) -> AsyncIterator[str]:
        prefix = f"{self.prefix}/" if self.prefix else ""
        continuation: str | None = None
        while True:
            options: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix, "MaxKeys": 1000}
            if continuation is not None:
                options["ContinuationToken"] = continuation
            response = await asyncio.to_thread(client.list_objects_v2, **options)
            if not isinstance(response, Mapping):
                raise ArtifactStoreUnavailableError(
                    "S3 artifact store received an invalid list response."
                )
            entries = response.get("Contents", [])
            if (
                type(entries) is not list
                or len(entries) > 1000
                or type(response.get("IsTruncated")) is not bool
            ):
                raise ArtifactStoreUnavailableError(
                    "S3 artifact store received an invalid list page."
                )
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                key = entry.get("Key")
                if type(key) is not str or not key.startswith(prefix):
                    continue
                relative = key[len(prefix) :]
                artifact_id, separator, filename = relative.partition("/")
                if (
                    separator
                    and (
                        filename in {"metadata.json", "content"}
                        or _GENERATION_CONTENT_PATTERN.fullmatch(filename)
                    )
                    and _ARTIFACT_ID_PATTERN.fullmatch(artifact_id)
                ):
                    if filename == "metadata.json":
                        yield artifact_id
                    else:
                        # Inspect content keys too, without collecting an unbounded
                        # set of IDs or counting the two objects twice. A partial
                        # DeleteObjects failure can leave content without metadata.
                        try:
                            await _run_s3_sync_call(
                                None,
                                client.head_object,
                                Bucket=self.bucket,
                                Key=self._artifact_key(artifact_id, "metadata.json"),
                            )
                        except Exception as error:
                            if _aws_error_code(error) not in _NOT_FOUND_CODES:
                                raise
                            await self._require_no_surviving_content(client, artifact_id)
            if not response["IsTruncated"]:
                return
            continuation_value = response.get("NextContinuationToken")
            if (
                type(continuation_value) is not str
                or not continuation_value
                or continuation_value == continuation
            ):
                raise ArtifactStoreUnavailableError(
                    "S3 artifact store list response omitted continuation token."
                )
            continuation = continuation_value

    def _artifact_key(self, artifact_id: str, filename: str) -> str:
        suffix = f"{artifact_id}/{filename}"
        return f"{self.prefix}/{suffix}" if self.prefix else suffix

    def _content_key(self, artifact_id: str, generation: str) -> str:
        if generation == LEGACY_GENERATION:
            return self._artifact_key(artifact_id, "content")
        return self._artifact_key(artifact_id, f"{generation}/content")

    def _encryption_options(self) -> dict[str, str]:
        if self._kms_key_id is None:
            return {}
        return {
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": self._kms_key_id,
        }

    async def _get_client(
        self,
        *,
        reporter: _ArtifactWritePhaseReporter | None = None,
    ) -> Any:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = await _run_s3_sync_call(
                    reporter,
                    self._create_client,
                )
        return self._client

    def _create_client(self) -> Any:
        boto3 = _boto3_module()
        session_options: dict[str, Any] = {}
        if self._profile_name is not None:
            session_options["profile_name"] = self._profile_name
        session = boto3.Session(**session_options)
        client_options: dict[str, Any] = {}
        if self._region_name is not None:
            client_options["region_name"] = self._region_name
        if self._endpoint_url is not None:
            client_options["endpoint_url"] = self._endpoint_url
        return session.client("s3", **client_options)


def _validate_artifact_id(value: str) -> str:
    if type(value) is not str or _ARTIFACT_ID_PATTERN.fullmatch(value) is None:
        raise InvalidArtifactIdError("Invalid S3 artifact id.")
    return value


def _validate_scope(value: ArtifactScope | str) -> ArtifactScope:
    if isinstance(value, ArtifactScope):
        return value
    if type(value) is str:
        try:
            return ArtifactScope(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported artifact scope: {value!r}") from exc
    raise TypeError("Artifact scope must be an ArtifactScope.")


def _validate_scope_owner(
    scope: ArtifactScope,
    *,
    session_id: str | None,
    environment_name: str | None,
) -> None:
    if scope == ArtifactScope.SESSION and session_id is None:
        raise ValueError("Session-scoped artifacts require session_id.")
    if scope == ArtifactScope.ENVIRONMENT and environment_name is None:
        raise ValueError("Environment-scoped artifacts require environment_name.")


def _validate_limit(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"Artifact {field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"Artifact {field_name} must be greater than zero.")
    return value


def _optional_identifier(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return require_clean_nonblank(value, field_name)


def _optional_clean_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return require_clean_nonblank(value, field_name)


async def _run_s3_sync_call(
    reporter: _ArtifactWritePhaseReporter | None,
    callback: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if reporter is None:
        # An unreported sync call runs to completion even if its caller is
        # cancelled, so whoever waits for the caller also waits for the call.
        return await run_thread_to_completion(callback, *args, **kwargs)
    return await _await_owned_sync_call(reporter, callback, *args, **kwargs)


def _get_object_content(client: Any, max_bytes: int | None, /, **options: Any) -> tuple[Any, bytes]:
    """Fetch an object and read its body in one call, closing the body either way.

    One call so a caller cancelled in between cannot leave an open body behind.
    """

    response = client.get_object(**options)
    return response, _response_body_bytes(response, max_bytes)


def _response_body_bytes(response: Any, max_bytes: int | None = None) -> bytes:
    if not isinstance(response, Mapping):
        raise TypeError("S3 object response must be a mapping.")
    body = response.get("Body")
    read = getattr(body, "read", None)
    if read is None or not callable(read):
        raise TypeError("S3 object response omitted a readable body.")
    try:
        value = read() if max_bytes is None else read(max_bytes)
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if type(value) is not bytes:
        raise TypeError("S3 object body must return bytes.")
    return value


def _combined_s3_write_failure(
    message: str,
    *,
    primary: BaseException,
    reconciliation: BaseException,
) -> ArtifactStoreUnavailableError:
    if exception_cause(reconciliation) is None and exception_context(reconciliation) is primary:
        set_exception_context(reconciliation, None)
    failure = ArtifactStoreUnavailableError(message)
    failure.__cause__ = BaseExceptionGroup(
        "S3 artifact mutation and reconciliation failures.",
        [primary, reconciliation],
    )
    return failure


@dataclass(frozen=True)
class _Publication:
    """The committed generation of an artifact and its metadata object's ETag."""

    generation: str
    etag: str


def _publication_body(artifact: ArtifactMetadata, generation: str) -> bytes:
    return (
        b'{"artifact":'
        + artifact.model_dump_json().encode("utf-8")
        + f',"{_PUBLICATION_FORMAT_KEY}":{_PUBLICATION_FORMAT},"generation":"{generation}"}}'.encode(
            "ascii"
        )
    )


def _parse_publication(artifact_id: str, payload: bytes) -> tuple[ArtifactMetadata, str]:
    """Decode committed metadata; unwrapped metadata is the legacy generation."""

    invalid = f"S3 artifact metadata is invalid: {artifact_id}"
    try:
        document = json.loads(payload)
    except (ValueError, TypeError) as exc:
        raise ValueError(invalid) from exc
    body: bytes = payload
    generation = LEGACY_GENERATION
    if isinstance(document, dict) and _PUBLICATION_FORMAT_KEY in document:
        generation_value = document.get("generation")
        if (
            set(document) != _PUBLICATION_KEYS
            or type(document[_PUBLICATION_FORMAT_KEY]) is not int
            or document[_PUBLICATION_FORMAT_KEY] != _PUBLICATION_FORMAT
            or type(generation_value) is not str
            or re.fullmatch(r"[0-9a-f]{32}", generation_value) is None
            or not isinstance(document["artifact"], dict)
        ):
            raise ValueError(invalid)
        generation = generation_value
        body = json.dumps(document["artifact"]).encode("utf-8")
    try:
        metadata = ArtifactMetadata.model_validate_json(body)
    except (ValueError, TypeError) as exc:
        raise ValueError(invalid) from exc
    if metadata.id != artifact_id:
        raise ValueError("S3 artifact metadata id did not match its object key.")
    return metadata, generation


_DELETION_ATTEMPT_COUNTER_ID = "cayu-s3-artifact-deletion-attempts"
_ACTIVE_DELETION: ContextVar[_DeletionSubmission | None] = ContextVar(
    "cayu_s3_artifact_deletion", default=None
)


@dataclass
class _DeletionSubmission:
    """One fenced DeleteObjects call and the attempts it actually sent."""

    sends: int = 0
    answered: bool = False

    def send(self, client: Any, bucket: str, objects: Sequence[Mapping[str, str]]) -> Any:
        if not _install_deletion_attempt_counter(client):
            # Without botocore's per-attempt hook, the call counts as one attempt.
            self.sends = 1
        bound = _ACTIVE_DELETION.set(self)
        try:
            response = client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [dict(item) for item in objects], "Quiet": True},
            )
        finally:
            _ACTIVE_DELETION.reset(bound)
        self.answered = True
        return response

    def resolved(self, error: BaseException | None) -> bool:
        """Whether S3 can no longer apply anything this call sent.

        True when nothing was sent, or when exactly one attempt was sent and S3
        answered it with a success or a client error. A transport failure, a
        server error, or a retried call may still be applied later.
        """

        if self.sends == 0:
            return True
        if self.sends != 1:
            return False
        if self.answered:
            # S3 returned a response, possibly carrying per-object errors.
            return True
        status = _http_status(error)
        return status is not None and status < 500


def _install_deletion_attempt_counter(client: Any) -> bool:
    events = getattr(getattr(client, "meta", None), "events", None)
    register = getattr(events, "register", None)
    if not callable(register):
        return False
    register(
        "before-send.s3.DeleteObjects",
        _count_deletion_attempt,
        unique_id=_DELETION_ATTEMPT_COUNTER_ID,
    )
    return True


def _count_deletion_attempt(request: Any, **_: Any) -> None:
    """Count every attempt botocore puts on the wire, including SDK retries."""

    del request
    submission = _ACTIVE_DELETION.get()
    if submission is not None:
        submission.sends += 1
    return None


_CONTENT_PUBLICATION_ATTEMPTS = 3
_CONTENT_PUT_GUARD_ID = "cayu-s3-artifact-content-put"
_ACTIVE_CONTENT_PUT: ContextVar[_ContentSubmission | None] = ContextVar(
    "cayu_s3_artifact_content_put", default=None
)


SINGLE_ATTEMPT_PUT_OBJECT_ATTRIBUTE = "cayu_single_attempt_put_object"


class S3ArtifactClientConfigurationError(TypeError):
    """The S3 client cannot guarantee single-attempt content writes."""


class _ContentResendRefused(ArtifactStoreUnavailableError):
    """The send guard stopped a second content PUT before it left the host."""


def _declares_single_attempt_put_object(client: Any) -> bool:
    """Whether the class implementing ``put_object`` declares one request per call.

    The declaration is a class attribute set to ``True`` on the class that
    defines ``put_object`` or a subclass of it. A subclass that overrides
    ``put_object`` (for example to route through a retrying SDK) must declare
    it again for its own implementation; an inherited declaration does not
    cover it. Instance attributes are ignored.
    """

    mro = type(client).__mro__
    declaring = next((cls for cls in mro if SINGLE_ATTEMPT_PUT_OBJECT_ATTRIBUTE in vars(cls)), None)
    if declaring is None or vars(declaring)[SINGLE_ATTEMPT_PUT_OBJECT_ATTRIBUTE] is not True:
        return False
    implementing = next((cls for cls in mro if "put_object" in vars(cls)), None)
    return implementing is not None and issubclass(declaring, implementing)


def _require_single_attempt_content_writes(client: Any) -> None:
    if _install_content_put_guard(client) or _declares_single_attempt_put_object(client):
        return
    raise S3ArtifactClientConfigurationError(
        "S3ArtifactStore needs single-attempt content writes: pass a botocore S3 client "
        "(it exposes meta.events, so the store vetoes retries of content PutObject), or "
        f"a client whose put_object class sets {SINGLE_ATTEMPT_PUT_OBJECT_ATTRIBUTE} = True "
        "because each put_object call sends at most one request. A retried content write "
        "could otherwise recreate a generation a deletion already removed."
    )


@dataclass
class _ContentSubmission:
    """One content PUT for a fresh generation, sent at most once.

    A retried PUT could recreate its generation after a deletion swept it and
    settled that deletion's fences, so botocore retries are vetoed and a second
    send is refused. Only an acknowledged attempt makes its generation usable.
    """

    sends: int = 0

    def send(self, client: Any, **options: Any) -> Any:
        if not _install_content_put_guard(client):
            # Without botocore's per-attempt hooks the call counts as one attempt.
            self.sends = 1
        bound = _ACTIVE_CONTENT_PUT.set(self)
        try:
            return client.put_object(**options)
        finally:
            _ACTIVE_CONTENT_PUT.reset(bound)

    def unapplied(self, error: BaseException) -> bool:
        """Whether this call can never store its generation's content.

        True when nothing was sent, or when S3 answered the only attempt with a
        client error. A transport failure or server error may still be applied.
        """

        if self.sends == 0:
            return True
        status = _http_status(error)
        return self.sends == 1 and status is not None and 400 <= status < 500


def _install_content_put_guard(client: Any) -> bool:
    events = getattr(getattr(client, "meta", None), "events", None)
    register = getattr(events, "register", None)
    if not callable(register):
        return False
    register(
        "before-send.s3.PutObject",
        _guard_content_put,
        unique_id=f"{_CONTENT_PUT_GUARD_ID}-send",
    )
    register(
        "needs-retry.s3.PutObject",
        _veto_content_put_retry,
        unique_id=f"{_CONTENT_PUT_GUARD_ID}-retry",
    )
    return True


def _guard_content_put(**_: Any) -> None:
    submission = _ACTIVE_CONTENT_PUT.get()
    if submission is None:
        return None
    if submission.sends:
        raise _ContentResendRefused(
            "S3 artifact content PUT retry was refused: a generation is written by one "
            "request, and the earlier one may still be applied."
        )
    submission.sends += 1
    return None


def _veto_content_put_retry(**_: Any) -> bool | None:
    # Registered on the exact operation, so it answers before the service-wide
    # retry handler; False tells botocore not to retry.
    return False if _ACTIVE_CONTENT_PUT.get() is not None else None


def _http_status(error: BaseException | None) -> int | None:
    response = getattr(error, "response", None)
    metadata = response.get("ResponseMetadata") if isinstance(response, Mapping) else None
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    return status if type(status) is int else None


# Per-object outcomes on the ETag-conditioned metadata key that prove the
# targeted publication's metadata no longer exists.
_METADATA_GONE_CODES = frozenset({"PreconditionFailed", "NoSuchKey"})


def _require_complete_deletion(response: Any, *, conditional_key: str | None = None) -> None:
    if not isinstance(response, Mapping):
        raise ArtifactStoreUnavailableError("S3 DeleteObjects returned an invalid response.")
    errors = response.get("Errors")
    if errors is None or errors == []:
        return
    if not isinstance(errors, Sequence) or isinstance(errors, (str, bytes, bytearray)):
        raise ArtifactStoreUnavailableError(
            "S3 DeleteObjects returned an invalid per-object error collection."
        )
    errors = [
        error
        for error in errors
        if not (
            isinstance(error, Mapping)
            and conditional_key is not None
            and error.get("Key") == conditional_key
            and error.get("Code") in _METADATA_GONE_CODES
        )
    ]
    if not errors:
        return
    safe_codes: list[str] = []
    for error in errors[:3]:
        code = error.get("Code") if isinstance(error, Mapping) else None
        safe_code = (
            code if type(code) is str and re.fullmatch(r"[A-Za-z0-9._-]{1,64}", code) else "Unknown"
        )
        if safe_code not in safe_codes:
            safe_codes.append(safe_code)
    error_label = "error" if len(errors) == 1 else "errors"
    raise ArtifactStoreUnavailableError(
        f"S3 DeleteObjects returned {len(errors)} per-object {error_label} "
        f"(codes: {', '.join(safe_codes)})."
    )


def _aws_error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return None
    error = response.get("Error")
    if not isinstance(error, Mapping):
        return None
    code = error.get("Code")
    return code if type(code) is str else None


def _boto3_module() -> Any:
    try:
        return importlib.import_module("boto3")
    except ModuleNotFoundError as exc:
        if exc.name != "boto3":
            raise
        raise RuntimeError(
            "S3ArtifactStore requires the optional AWS dependencies; install cayu[aws]."
        ) from exc
