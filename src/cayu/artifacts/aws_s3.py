from __future__ import annotations

import asyncio
import importlib
import json
import mimetypes
import re
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

from cayu._exception_groups import exception_cause, exception_context, set_exception_context
from cayu._validation import (
    copy_durable_json_object,
    require_clean_nonblank,
    require_nonblank,
    require_unicode_scalar_text,
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
from cayu.artifacts.base import (
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
)

_ARTIFACT_ID_PATTERN = re.compile(r"\Aart_[0-9a-f]{32}\Z")
_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class S3ArtifactStore(ArtifactStore):
    """Direct S3 object implementation of ArtifactStore.

    Content is written first and metadata.json last. The metadata object is
    the commit marker, so interrupted writes are never listed as artifacts.

    ``list`` scans every committed metadata object under ``prefix`` and fetches
    each metadata document before filtering, sorting, and applying ``limit``.
    This keeps totals exact but targets modest artifact volumes. High-volume
    deployments should maintain a separate query index.
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
        artifact = ArtifactMetadata(
            id=resolved_artifact_id,
            filename=filename,
            content_type=resolved_content_type,
            size_bytes=len(content),
            scope=validated_scope,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=copy_durable_json_object(metadata or {}, "metadata"),
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

    async def _run_artifact_write(
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

        content_key = self._artifact_key(artifact.id, "content")
        metadata_key = self._artifact_key(artifact.id, "metadata.json")
        reconciled_failure_codes: tuple[ArtifactWriteSettlementFailureCode, ...] = ()
        reporter.set(ArtifactWriteSettlementPhase.CONTENT)
        try:
            await _run_s3_sync_call(
                reporter,
                client.put_object,
                Bucket=self.bucket,
                Key=content_key,
                Body=content,
                ContentType=artifact.content_type,
                IfNoneMatch="*",
                **self._encryption_options(),
            )
        except BaseException as content_error:
            if not issubclass(type(content_error), Exception):
                return _unsettled_artifact_write(
                    content_error,
                    phase=ArtifactWriteSettlementPhase.CONTENT,
                    failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
                )
            reporter.set(ArtifactWriteSettlementPhase.RECONCILIATION)
            try:
                existing_content = await self._read_object_bytes(
                    client,
                    content_key,
                    reporter=reporter,
                )
            except BaseException as reconciliation_error:
                failure = _combined_s3_write_failure(
                    "S3 artifact store could not write artifact content.",
                    primary=content_error,
                    reconciliation=reconciliation_error,
                )
                return _unsettled_artifact_write(
                    failure,
                    phase=ArtifactWriteSettlementPhase.RECONCILIATION,
                    failure_codes=(
                        ArtifactWriteSettlementFailureCode.MUTATION_FAILED,
                        ArtifactWriteSettlementFailureCode.RECONCILIATION_FAILED,
                    ),
                )
            if existing_content != content:
                conflict = ValueError(
                    "Artifact identity already exists with different content or metadata."
                )
                conflict.__cause__ = content_error
                return _unsettled_artifact_write(
                    conflict,
                    phase=ArtifactWriteSettlementPhase.RECONCILIATION,
                    failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
                )
            reconciled_failure_codes = (ArtifactWriteSettlementFailureCode.MUTATION_FAILED,)

        reporter.set(ArtifactWriteSettlementPhase.COMMIT)
        try:
            await _run_s3_sync_call(
                reporter,
                client.put_object,
                Bucket=self.bucket,
                Key=metadata_key,
                Body=artifact.model_dump_json().encode("utf-8"),
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

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        artifact_id = _validate_artifact_id(artifact_id)
        limit = _validate_limit(max_bytes, "max_bytes")
        client = await self._get_client()
        metadata = await self._read_metadata(client, artifact_id)
        get_options: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self._artifact_key(artifact_id, "content"),
        }
        if limit is not None and limit < metadata.size_bytes:
            get_options["Range"] = f"bytes=0-{limit - 1}"
        try:
            response = await asyncio.to_thread(client.get_object, **get_options)
            content = await asyncio.to_thread(_response_body_bytes, response)
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
            metadata_ids = await asyncio.to_thread(self._list_metadata_ids, client)
            artifacts: list[ArtifactMetadata] = []
            for artifact_id in metadata_ids:
                try:
                    artifact = await self._read_metadata(client, artifact_id)
                except (FileNotFoundError, ValueError):
                    continue
                if validated_scope is not None and artifact.scope != validated_scope:
                    continue
                if session_id is not None and artifact.session_id != session_id:
                    continue
                if agent_name is not None and artifact.agent_name != agent_name:
                    continue
                if environment_name is not None and artifact.environment_name != environment_name:
                    continue
                artifacts.append(artifact)
        except ArtifactStoreUnavailableError:
            raise
        except Exception as exc:
            raise ArtifactStoreUnavailableError(
                "S3 artifact store could not list artifacts."
            ) from exc
        artifacts.sort(key=lambda artifact: artifact.created_at, reverse=True)
        total_count = len(artifacts)
        selected = artifacts if validated_limit is None else artifacts[:validated_limit]
        return ArtifactListResult(
            artifacts=tuple(selected),
            total_count=total_count,
            truncated=len(selected) < total_count,
        )

    async def delete(self, artifact_id: str) -> None:
        artifact_id = _validate_artifact_id(artifact_id)
        client = await self._get_client()
        keys = (
            self._artifact_key(artifact_id, "content"),
            self._artifact_key(artifact_id, "metadata.json"),
        )
        try:
            await asyncio.to_thread(self._delete_keys, client, keys)
        except ArtifactStoreUnavailableError:
            raise
        except Exception as exc:
            raise ArtifactStoreUnavailableError(
                "S3 artifact store could not delete artifact content."
            ) from exc

    async def _read_metadata(
        self,
        client: Any,
        artifact_id: str,
        *,
        reporter: _ArtifactWritePhaseReporter | None = None,
    ) -> ArtifactMetadata:
        try:
            response = await _run_s3_sync_call(
                reporter,
                client.get_object,
                Bucket=self.bucket,
                Key=self._artifact_key(artifact_id, "metadata.json"),
            )
            payload = await _run_s3_sync_call(reporter, _response_body_bytes, response)
        except Exception as exc:
            if _aws_error_code(exc) in _NOT_FOUND_CODES:
                raise FileNotFoundError(f"Artifact not found: {artifact_id}") from exc
            raise ArtifactStoreUnavailableError(
                "S3 artifact store could not read artifact metadata."
            ) from exc
        try:
            metadata = ArtifactMetadata.model_validate_json(payload)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"S3 artifact metadata is invalid: {artifact_id}") from exc
        if metadata.id != artifact_id:
            raise ValueError("S3 artifact metadata id did not match its object key.")
        return metadata

    async def _read_object_bytes(
        self,
        client: Any,
        key: str,
        *,
        reporter: _ArtifactWritePhaseReporter | None = None,
    ) -> bytes:
        response = await _run_s3_sync_call(
            reporter,
            client.get_object,
            Bucket=self.bucket,
            Key=key,
        )
        return await _run_s3_sync_call(reporter, _response_body_bytes, response)

    async def _read_complete_artifact(
        self,
        client: Any,
        artifact_id: str,
        *,
        reporter: _ArtifactWritePhaseReporter | None = None,
    ) -> ArtifactReadResult:
        metadata = await self._read_metadata(client, artifact_id, reporter=reporter)
        try:
            content = await self._read_object_bytes(
                client,
                self._artifact_key(artifact_id, "content"),
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

    def _list_metadata_ids(self, client: Any) -> Sequence[str]:
        prefix = f"{self.prefix}/" if self.prefix else ""
        continuation: str | None = None
        artifact_ids: list[str] = []
        while True:
            options: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if continuation is not None:
                options["ContinuationToken"] = continuation
            response = client.list_objects_v2(**options)
            if not isinstance(response, Mapping):
                raise ArtifactStoreUnavailableError(
                    "S3 artifact store received an invalid list response."
                )
            for entry in response.get("Contents", []):
                if not isinstance(entry, Mapping):
                    continue
                key = entry.get("Key")
                if type(key) is not str or not key.endswith("/metadata.json"):
                    continue
                relative = key[len(prefix) :]
                artifact_id, separator, filename = relative.partition("/")
                if (
                    separator
                    and filename == "metadata.json"
                    and _ARTIFACT_ID_PATTERN.fullmatch(artifact_id)
                ):
                    artifact_ids.append(artifact_id)
            if not response.get("IsTruncated"):
                return artifact_ids
            continuation_value = response.get("NextContinuationToken")
            if type(continuation_value) is not str or not continuation_value:
                raise ArtifactStoreUnavailableError(
                    "S3 artifact store list response omitted continuation token."
                )
            continuation = continuation_value

    def _artifact_key(self, artifact_id: str, filename: str) -> str:
        suffix = f"{artifact_id}/{filename}"
        return f"{self.prefix}/{suffix}" if self.prefix else suffix

    def _encryption_options(self) -> dict[str, str]:
        if self._kms_key_id is None:
            return {}
        return {
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": self._kms_key_id,
        }

    def _delete_keys(self, client: Any, keys: tuple[str, ...]) -> None:
        response = client.delete_objects(
            Bucket=self.bucket,
            Delete={"Objects": [{"Key": key} for key in keys], "Quiet": True},
        )
        if not isinstance(response, Mapping):
            raise ArtifactStoreUnavailableError("S3 DeleteObjects returned an invalid response.")
        errors = response.get("Errors")
        if errors is None or errors == []:
            return
        if not isinstance(errors, Sequence) or isinstance(errors, (str, bytes, bytearray)):
            raise ArtifactStoreUnavailableError(
                "S3 DeleteObjects returned an invalid per-object error collection."
            )
        safe_codes: list[str] = []
        for error in errors[:3]:
            code = error.get("Code") if isinstance(error, Mapping) else None
            safe_code = (
                code
                if type(code) is str and re.fullmatch(r"[A-Za-z0-9._-]{1,64}", code)
                else "Unknown"
            )
            if safe_code not in safe_codes:
                safe_codes.append(safe_code)
        error_label = "error" if len(errors) == 1 else "errors"
        raise ArtifactStoreUnavailableError(
            f"S3 DeleteObjects returned {len(errors)} per-object {error_label} "
            f"(codes: {', '.join(safe_codes)})."
        )

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
        return await asyncio.to_thread(callback, *args, **kwargs)
    return await _await_owned_sync_call(reporter, callback, *args, **kwargs)


def _response_body_bytes(response: Any) -> bytes:
    if not isinstance(response, Mapping):
        raise TypeError("S3 object response must be a mapping.")
    body = response.get("Body")
    read = getattr(body, "read", None)
    if read is None or not callable(read):
        raise TypeError("S3 object response omitted a readable body.")
    try:
        value = read()
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
