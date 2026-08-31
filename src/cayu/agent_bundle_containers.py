"""Deterministic single-file transport for canonical :class:`AgentBundle` directories.

The ``.cayu`` container is a representation only.  Its digest is useful for
download verification but is never an agent-state root, bundle identity, or
authorization token.
"""

from __future__ import annotations

import ctypes
import errno
import io
import os
import stat
import struct
import sys
import tempfile
import unicodedata
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from cayu._filesystem_lock import cooperative_path_lock
from cayu._validation import canonical_durable_json_bytes
from cayu.agent_bundles import (
    AGENT_BUNDLE_INDEX_FILENAME,
    AGENT_BUNDLE_MAX_INDEX_BYTES,
    AGENT_BUNDLE_MAX_OBJECT_BYTES,
    AGENT_BUNDLE_MAX_OBJECTS,
    AGENT_BUNDLE_MAX_TOTAL_BYTES,
    AGENT_BUNDLE_OBJECT_DIRECTORY,
    AgentBundle,
    AgentBundleCoordinator,
    AgentBundleError,
    AgentBundleMode,
    AgentBundleObjectKind,
    AgentBundleObjectRef,
)
from cayu.agent_snapshots import AgentSnapshotNode

AGENT_BUNDLE_CONTAINER_EXTENSION = ".cayu"
AGENT_BUNDLE_CONTAINER_MEDIA_TYPE = "application/vnd.cayu.agent-bundle"
AGENT_BUNDLE_CONTAINER_SCHEMA_VERSION = 1
AGENT_BUNDLE_CONTAINER_MIMETYPE_ENTRY = "mimetype"
AGENT_BUNDLE_CONTAINER_VERSION_EXTRA_ID = 0xCA7A
AGENT_BUNDLE_CONTAINER_MAX_METADATA_BYTES = 1024 * 1024 * 1024
AGENT_BUNDLE_CONTAINER_MAX_BYTES = (
    AGENT_BUNDLE_MAX_TOTAL_BYTES
    + AGENT_BUNDLE_MAX_INDEX_BYTES
    + AGENT_BUNDLE_CONTAINER_MAX_METADATA_BYTES
)
AGENT_BUNDLE_CONTAINER_MAX_ENTRIES = AGENT_BUNDLE_MAX_OBJECTS + 2

_CONTAINER_RECORD_TYPE = "cayu.agent-bundle-container"
_CANONICAL_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_CANONICAL_DOS_DATE = 33
_CANONICAL_DOS_TIME = 0
_CANONICAL_EXTERNAL_ATTR = (stat.S_IFREG | 0o600) << 16
_COPY_CHUNK_BYTES = 1024 * 1024
_ZIP_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_ZIP_CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")
_ZIP_END_RECORD = struct.Struct("<4s4H2LH")
_ZIP64_END_RECORD = struct.Struct("<4sQ2H2L4Q")
_ZIP64_LOCATOR = struct.Struct("<4sLQL")
_ZIP_LOCAL_SIGNATURE = b"PK\x03\x04"
_ZIP_CENTRAL_SIGNATURE = b"PK\x01\x02"
_ZIP_END_SIGNATURE = b"PK\x05\x06"
_ZIP64_END_SIGNATURE = b"PK\x06\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP64_EXTRA_ID = 0x0001
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 4
_VERSION_EXTRA = struct.pack(
    "<HHH",
    AGENT_BUNDLE_CONTAINER_VERSION_EXTRA_ID,
    2,
    AGENT_BUNDLE_CONTAINER_SCHEMA_VERSION,
)
_MIMETYPE_BYTES = AGENT_BUNDLE_CONTAINER_MEDIA_TYPE.encode("ascii")
_SHA256_CHARACTERS = frozenset("0123456789abcdef")

BinarySource = str | os.PathLike[str] | BinaryIO
BinaryDestination = str | os.PathLike[str] | BinaryIO


class _ContainerModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )


class AgentBundleContainerInspection(_ContainerModel):
    """Bounded safe metadata reported after complete container validation."""

    record_type: Literal["cayu.agent-bundle-container"] = _CONTAINER_RECORD_TYPE
    schema_version: Literal[1] = AGENT_BUNDLE_CONTAINER_SCHEMA_VERSION
    media_type: Literal["application/vnd.cayu.agent-bundle"] = AGENT_BUNDLE_CONTAINER_MEDIA_TYPE
    transport_sha256: StrictStr
    container_bytes: StrictInt = Field(ge=1, le=AGENT_BUNDLE_CONTAINER_MAX_BYTES)
    bundle_id: StrictStr
    snapshot_root: StrictStr
    profile: StrictStr
    mode: AgentBundleMode
    object_count: StrictInt = Field(ge=1, le=AGENT_BUNDLE_MAX_OBJECTS)
    transferred_object_count: StrictInt = Field(ge=1, le=AGENT_BUNDLE_MAX_OBJECTS)
    logical_closure_bytes: StrictInt = Field(ge=1, le=AGENT_BUNDLE_MAX_TOTAL_BYTES)
    transferred_bytes: StrictInt = Field(ge=0, le=AGENT_BUNDLE_MAX_TOTAL_BYTES)
    destination_inventory_fingerprint: StrictStr
    requires_preexisting_objects: StrictBool
    unresolved_external_bindings: tuple[StrictStr, ...]


class AgentBundleContainerReceipt(_ContainerModel):
    """Receipt for one verified representation conversion."""

    inspection: AgentBundleContainerInspection
    bundle: AgentBundle


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = sha256()
    byte_count = 0
    with path.open("rb") as source:
        while chunk := source.read(_COPY_CHUNK_BYTES):
            byte_count += len(chunk)
            if byte_count > AGENT_BUNDLE_CONTAINER_MAX_BYTES:
                raise AgentBundleError("container_size_limit_exceeded")
            digest.update(chunk)
    return digest.hexdigest(), byte_count


def _canonical_object_entry(digest: str) -> str:
    if len(digest) != 64 or any(character not in _SHA256_CHARACTERS for character in digest):
        raise AgentBundleError("container_object_digest_invalid")
    return f"{AGENT_BUNDLE_OBJECT_DIRECTORY}/{digest[:2]}/{digest[2:]}"


def _safe_entry_name(name: str) -> str:
    if not name or "\x00" in name or "\\" in name:
        raise AgentBundleError("container_entry_name_invalid")
    try:
        encoded = name.encode("ascii")
    except UnicodeEncodeError as error:
        raise AgentBundleError("container_entry_name_invalid") from error
    if encoded.decode("ascii") != name or unicodedata.normalize("NFC", name) != name:
        raise AgentBundleError("container_entry_name_invalid")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or name.startswith(("/", "//"))
        or (len(name) >= 2 and name[1] == ":")
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != name
    ):
        raise AgentBundleError("container_entry_name_invalid")
    return name


def _parse_extra(extra: bytes) -> tuple[tuple[int, bytes], ...]:
    fields: list[tuple[int, bytes]] = []
    position = 0
    while position < len(extra):
        if len(extra) - position < 4:
            raise AgentBundleError("container_extra_field_malformed")
        identifier, size = struct.unpack_from("<HH", extra, position)
        position += 4
        end = position + size
        if end > len(extra):
            raise AgentBundleError("container_extra_field_malformed")
        fields.append((identifier, extra[position:end]))
        position = end
    identifiers = tuple(identifier for identifier, _ in fields)
    if len(identifiers) != len(set(identifiers)):
        raise AgentBundleError("container_extra_field_duplicated")
    return tuple(fields)


def _validate_extra(extra: bytes, *, mimetype: bool) -> None:
    fields = _parse_extra(extra)
    allowed = {_ZIP64_EXTRA_ID}
    if mimetype:
        allowed.add(AGENT_BUNDLE_CONTAINER_VERSION_EXTRA_ID)
    if any(identifier not in allowed for identifier, _ in fields):
        raise AgentBundleError("container_extra_field_unsupported")
    versions = [
        value
        for identifier, value in fields
        if identifier == AGENT_BUNDLE_CONTAINER_VERSION_EXTRA_ID
    ]
    if mimetype:
        if versions != [struct.pack("<H", AGENT_BUNDLE_CONTAINER_SCHEMA_VERSION)]:
            raise AgentBundleError("container_schema_unsupported")
    elif versions:
        raise AgentBundleError("container_version_metadata_reordered")


def _zip64_values(extra: bytes) -> tuple[int, ...]:
    for identifier, value in _parse_extra(extra):
        if identifier == _ZIP64_EXTRA_ID:
            if len(value) % 8:
                raise AgentBundleError("container_zip64_malformed")
            return tuple(
                struct.unpack_from("<Q", value, position)[0] for position in range(0, len(value), 8)
            )
    return ()


def _resolved_local_sizes(
    compressed: int,
    uncompressed: int,
    extra: bytes,
) -> tuple[int, int]:
    values = iter(_zip64_values(extra))
    try:
        actual_uncompressed = next(values) if uncompressed == 0xFFFFFFFF else uncompressed
        actual_compressed = next(values) if compressed == 0xFFFFFFFF else compressed
    except StopIteration as error:
        raise AgentBundleError("container_zip64_malformed") from error
    if tuple(values):
        raise AgentBundleError("container_zip64_malformed")
    return actual_compressed, actual_uncompressed


def _validate_central_zip64(
    *,
    extra: bytes,
    file_size: int,
    compressed_size: int,
    local_offset: int,
    disk_start: int,
    info: zipfile.ZipInfo,
) -> None:
    payload = next(
        (value for identifier, value in _parse_extra(extra) if identifier == _ZIP64_EXTRA_ID),
        b"",
    )
    position = 0

    def read_value(size: int) -> int:
        nonlocal position
        if position + size > len(payload):
            raise AgentBundleError("container_zip64_malformed")
        value = int.from_bytes(payload[position : position + size], "little")
        position += size
        return value

    if file_size == 0xFFFFFFFF and read_value(8) != info.file_size:
        raise AgentBundleError("container_zip64_contradiction")
    if compressed_size == 0xFFFFFFFF and read_value(8) != info.compress_size:
        raise AgentBundleError("container_zip64_contradiction")
    if local_offset == 0xFFFFFFFF and read_value(8) != info.header_offset:
        raise AgentBundleError("container_zip64_contradiction")
    if disk_start == 0xFFFF and read_value(4) != info.volume:
        raise AgentBundleError("container_zip64_contradiction")
    if position != len(payload):
        raise AgentBundleError("container_zip64_malformed")


def _validate_end_records(stream: BinaryIO, archive: zipfile.ZipFile, file_size: int) -> int:
    if file_size < _ZIP_END_RECORD.size:
        raise AgentBundleError("container_truncated")
    stream.seek(file_size - _ZIP_END_RECORD.size)
    raw = stream.read(_ZIP_END_RECORD.size)
    if len(raw) != _ZIP_END_RECORD.size:
        raise AgentBundleError("container_truncated")
    (
        signature,
        disk_number,
        central_disk,
        entries_on_disk,
        entry_count,
        central_size,
        central_offset,
        comment_size,
    ) = _ZIP_END_RECORD.unpack(raw)
    if signature != _ZIP_END_SIGNATURE or comment_size != 0:
        raise AgentBundleError("container_trailing_data")
    if disk_number != 0 or central_disk != 0:
        raise AgentBundleError("container_multidisk_unsupported")
    end_offset = file_size - _ZIP_END_RECORD.size
    archive_entry_count = len(archive.infolist())
    extended_entry_count = archive_entry_count > zipfile.ZIP_FILECOUNT_LIMIT
    uses_zip64_end = (
        (entries_on_disk == 0xFFFF and extended_entry_count)
        or (entry_count == 0xFFFF and extended_entry_count)
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    )
    locator_offset = end_offset - _ZIP64_LOCATOR.size
    has_zip64_locator = False
    if locator_offset >= 0:
        stream.seek(locator_offset)
        has_zip64_locator = stream.read(4) == _ZIP64_LOCATOR_SIGNATURE
    requires_zip64_end = (
        extended_entry_count
        or central_size > zipfile.ZIP64_LIMIT
        or central_offset > zipfile.ZIP64_LIMIT
    )
    if has_zip64_locator and not uses_zip64_end and not requires_zip64_end:
        raise AgentBundleError("container_zip64_not_canonical")
    if uses_zip64_end or requires_zip64_end:
        if locator_offset < 0:
            raise AgentBundleError("container_zip64_malformed")
        stream.seek(locator_offset)
        locator = stream.read(_ZIP64_LOCATOR.size)
        if len(locator) != _ZIP64_LOCATOR.size:
            raise AgentBundleError("container_zip64_malformed")
        locator_signature, locator_disk, zip64_offset, disk_count = _ZIP64_LOCATOR.unpack(locator)
        if (
            locator_signature != _ZIP64_LOCATOR_SIGNATURE
            or locator_disk != 0
            or disk_count != 1
            or zip64_offset >= locator_offset
        ):
            raise AgentBundleError("container_zip64_malformed")
        stream.seek(zip64_offset)
        header = stream.read(_ZIP64_END_RECORD.size)
        if len(header) != _ZIP64_END_RECORD.size:
            raise AgentBundleError("container_zip64_malformed")
        (
            zip64_signature,
            zip64_size,
            _created,
            _required,
            zip64_disk,
            zip64_central_disk,
            zip64_entries_on_disk,
            zip64_entry_count,
            zip64_central_size,
            zip64_central_offset,
        ) = _ZIP64_END_RECORD.unpack(header)
        if (
            zip64_signature != _ZIP64_END_SIGNATURE
            or zip64_size != 44
            or zip64_disk != 0
            or zip64_central_disk != 0
            or zip64_entries_on_disk != zip64_entry_count
            or zip64_offset + 12 + zip64_size != locator_offset
        ):
            raise AgentBundleError("container_zip64_malformed")
        for ordinary, sentinel, extended in (
            (entries_on_disk, 0xFFFF, zip64_entries_on_disk),
            (entry_count, 0xFFFF, zip64_entry_count),
            (central_size, 0xFFFFFFFF, zip64_central_size),
            (central_offset, 0xFFFFFFFF, zip64_central_offset),
        ):
            if ordinary != sentinel and ordinary != extended:
                raise AgentBundleError("container_zip64_contradiction")
        entry_count = zip64_entry_count
        entries_on_disk = zip64_entries_on_disk
        central_size = zip64_central_size
        central_offset = zip64_central_offset
        end_offset = zip64_offset
    if entries_on_disk != entry_count or entry_count != len(archive.infolist()):
        raise AgentBundleError("container_entry_count_mismatch")
    if central_offset != archive.start_dir or central_offset + central_size != end_offset:
        raise AgentBundleError("container_central_directory_mismatch")
    return int(central_size)


def _validate_central_directory(
    stream: BinaryIO,
    archive: zipfile.ZipFile,
    central_size: int,
) -> None:
    position = archive.start_dir
    end = position + central_size
    for info in archive.infolist():
        stream.seek(position)
        raw = stream.read(_ZIP_CENTRAL_HEADER.size)
        if len(raw) != _ZIP_CENTRAL_HEADER.size:
            raise AgentBundleError("container_central_directory_truncated")
        (
            signature,
            version_created,
            version_required,
            flags,
            method,
            dos_time,
            dos_date,
            crc,
            compressed_size,
            file_size,
            name_size,
            extra_size,
            comment_size,
            disk_start,
            internal_attr,
            external_attr,
            local_offset,
        ) = _ZIP_CENTRAL_HEADER.unpack(raw)
        name = stream.read(name_size)
        extra = stream.read(extra_size)
        comment = stream.read(comment_size)
        position += _ZIP_CENTRAL_HEADER.size + name_size + extra_size + comment_size
        stream.seek(info.header_offset)
        local_header = stream.read(_ZIP_LOCAL_HEADER.size)
        if len(local_header) != _ZIP_LOCAL_HEADER.size:
            raise AgentBundleError("container_local_header_truncated")
        local_values = _ZIP_LOCAL_HEADER.unpack(local_header)
        local_uses_zip64 = local_values[7] == 0xFFFFFFFF or local_values[8] == 0xFFFFFFFF
        central_uses_zip64 = any(
            (
                compressed_size == 0xFFFFFFFF,
                file_size == 0xFFFFFFFF,
                local_offset == 0xFFFFFFFF,
                disk_start == 0xFFFF,
            )
        )
        expected_version_required = 45 if local_uses_zip64 or central_uses_zip64 else 20
        if signature != _ZIP_CENTRAL_SIGNATURE:
            raise AgentBundleError("container_central_directory_malformed")
        try:
            decoded_name = name.decode("ascii")
        except UnicodeDecodeError as error:
            raise AgentBundleError("container_entry_name_invalid") from error
        if decoded_name != info.filename or comment or info.comment:
            raise AgentBundleError("container_central_directory_mismatch")
        if (
            flags != info.flag_bits
            or method != info.compress_type
            or crc != info.CRC
            or dos_time != _CANONICAL_DOS_TIME
            or dos_date != _CANONICAL_DOS_DATE
            or disk_start != 0
            or internal_attr != 0
            or external_attr != info.external_attr
            or (info.header_offset != local_offset and local_offset != 0xFFFFFFFF)
            or compressed_size not in {info.compress_size, 0xFFFFFFFF}
            or file_size not in {info.file_size, 0xFFFFFFFF}
            or version_created != (3 << 8) | expected_version_required
            or version_created >> 8 != 3
            or version_required != expected_version_required
        ):
            raise AgentBundleError("container_central_directory_mismatch")
        _validate_extra(extra, mimetype=info.filename == AGENT_BUNDLE_CONTAINER_MIMETYPE_ENTRY)
        _validate_central_zip64(
            extra=extra,
            file_size=file_size,
            compressed_size=compressed_size,
            local_offset=local_offset,
            disk_start=disk_start,
            info=info,
        )
    if position != end:
        raise AgentBundleError("container_central_directory_mismatch")


def _validate_local_entries(stream: BinaryIO, archive: zipfile.ZipFile) -> None:
    expected_offset = 0
    seen_casefolded: set[str] = set()
    infos = archive.infolist()
    for index, info in enumerate(infos):
        name = _safe_entry_name(info.filename)
        folded = name.casefold()
        if folded in seen_casefolded:
            raise AgentBundleError("container_entry_name_collision")
        seen_casefolded.add(folded)
        if info.header_offset != expected_offset:
            raise AgentBundleError("container_entry_overlap_or_gap")
        if info.flag_bits != 0:
            if info.flag_bits & 0x1:
                raise AgentBundleError("container_encryption_unsupported")
            raise AgentBundleError("container_entry_flags_unsupported")
        if info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size:
            raise AgentBundleError("container_compression_unsupported")
        if info.volume != 0:
            raise AgentBundleError("container_multidisk_unsupported")
        if info.is_dir() or stat.S_IFMT(info.external_attr >> 16) != stat.S_IFREG:
            raise AgentBundleError("container_entry_not_regular")
        if info.date_time != _CANONICAL_TIMESTAMP or info.external_attr != _CANONICAL_EXTERNAL_ATTR:
            raise AgentBundleError("container_entry_metadata_not_canonical")
        stream.seek(info.header_offset)
        raw = stream.read(_ZIP_LOCAL_HEADER.size)
        if len(raw) != _ZIP_LOCAL_HEADER.size:
            raise AgentBundleError("container_local_header_truncated")
        (
            signature,
            version_required,
            flags,
            method,
            dos_time,
            dos_date,
            crc,
            compressed_size,
            file_size,
            name_size,
            extra_size,
        ) = _ZIP_LOCAL_HEADER.unpack(raw)
        raw_name = stream.read(name_size)
        extra = stream.read(extra_size)
        try:
            local_name = raw_name.decode("ascii")
        except UnicodeDecodeError as error:
            raise AgentBundleError("container_entry_name_invalid") from error
        resolved_compressed, resolved_file = _resolved_local_sizes(
            compressed_size,
            file_size,
            extra,
        )
        compressed_uses_zip64 = compressed_size == 0xFFFFFFFF
        file_uses_zip64 = file_size == 0xFFFFFFFF
        expected_version_required = 45 if compressed_uses_zip64 else 20
        if (
            signature != _ZIP_LOCAL_SIGNATURE
            or local_name != name
            or compressed_uses_zip64 != file_uses_zip64
            or version_required != expected_version_required
            or flags != info.flag_bits
            or method != info.compress_type
            or dos_time != _CANONICAL_DOS_TIME
            or dos_date != _CANONICAL_DOS_DATE
            or crc != info.CRC
            or resolved_compressed != info.compress_size
            or resolved_file != info.file_size
        ):
            raise AgentBundleError("container_local_header_mismatch")
        _validate_extra(extra, mimetype=index == 0)
        data_offset = info.header_offset + _ZIP_LOCAL_HEADER.size + name_size + extra_size
        expected_offset = data_offset + info.compress_size
    if expected_offset != archive.start_dir:
        raise AgentBundleError("container_entry_overlap_or_gap")


def _validate_archive_structure(stream: BinaryIO, archive: zipfile.ZipFile) -> int:
    stream.seek(0, io.SEEK_END)
    file_size = stream.tell()
    if file_size <= 0 or file_size > AGENT_BUNDLE_CONTAINER_MAX_BYTES:
        raise AgentBundleError("container_size_limit_exceeded")
    infos = archive.infolist()
    if len(infos) < 2 or len(infos) > AGENT_BUNDLE_CONTAINER_MAX_ENTRIES:
        raise AgentBundleError("container_entry_count_limit_exceeded")
    if archive.comment:
        raise AgentBundleError("container_comment_unsupported")
    central_size = _validate_end_records(stream, archive, file_size)
    _validate_central_directory(stream, archive, central_size)
    _validate_local_entries(stream, archive)
    return file_size


def _read_entry_bytes(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    max_bytes: int,
) -> bytes:
    if info.file_size > max_bytes:
        raise AgentBundleError("container_entry_size_limit_exceeded")
    with archive.open(info, "r") as source:
        value = source.read(max_bytes + 1)
        if len(value) > max_bytes or source.read(1):
            raise AgentBundleError("container_entry_size_limit_exceeded")
    if len(value) != info.file_size:
        raise AgentBundleError("container_entry_size_mismatch")
    return value


def _copy_verified_entry(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    reference: AgentBundleObjectRef,
    destination: Path | None,
) -> bytes | None:
    if info.file_size != reference.byte_count or info.file_size > AGENT_BUNDLE_MAX_OBJECT_BYTES:
        raise AgentBundleError("container_object_size_mismatch", reference.digest)
    structured = reference.kind is not AgentBundleObjectKind.COMPONENT_BLOB
    captured = bytearray() if structured else None
    digest = sha256()
    byte_count = 0
    output: BinaryIO | None = None
    try:
        if destination is not None:
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            output = os.fdopen(descriptor, "wb")
        with archive.open(info, "r") as source:
            while chunk := source.read(_COPY_CHUNK_BYTES):
                byte_count += len(chunk)
                if byte_count > reference.byte_count:
                    raise AgentBundleError("container_object_size_mismatch", reference.digest)
                digest.update(chunk)
                if captured is not None:
                    captured.extend(chunk)
                if output is not None:
                    output.write(chunk)
            if byte_count != reference.byte_count:
                raise AgentBundleError("container_object_size_mismatch", reference.digest)
        if reference.kind is AgentBundleObjectKind.SNAPSHOT_NODE:
            try:
                node = AgentSnapshotNode.model_validate_json(bytes(captured or b""))
            except Exception as error:
                raise AgentBundleError(
                    "container_snapshot_node_invalid", reference.digest
                ) from error
            if node.digest != reference.digest or canonical_durable_json_bytes(
                node.model_dump(mode="json"),
                "agent_snapshot_node",
            ) != bytes(captured or b""):
                raise AgentBundleError("container_object_integrity_mismatch", reference.digest)
        elif digest.hexdigest() != reference.digest:
            raise AgentBundleError("container_object_integrity_mismatch", reference.digest)
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
    finally:
        if output is not None:
            output.close()
    return None if captured is None else bytes(captured)


def _container_inspection(
    bundle: AgentBundle,
    *,
    transport_sha256: str,
    container_bytes: int,
) -> AgentBundleContainerInspection:
    transferred_digests = set(bundle.transferred_digests)
    transferred_bytes = sum(
        reference.byte_count
        for reference in bundle.closure
        if reference.digest in transferred_digests
    )
    return AgentBundleContainerInspection(
        transport_sha256=transport_sha256,
        container_bytes=container_bytes,
        bundle_id=bundle.bundle_id,
        snapshot_root=bundle.snapshot_ref.snapshot_root,
        profile=bundle.profile.value,
        mode=bundle.mode,
        object_count=len(bundle.closure),
        transferred_object_count=len(bundle.transferred_digests),
        logical_closure_bytes=bundle.size_report.logical_closure_bytes,
        transferred_bytes=transferred_bytes,
        destination_inventory_fingerprint=bundle.destination_inventory_fingerprint,
        requires_preexisting_objects=bundle.mode is AgentBundleMode.THIN,
        unresolved_external_bindings=bundle.size_report.unresolved_external_bindings,
    )


@contextmanager
def _seekable_source(source: BinarySource) -> Iterator[BinaryIO]:
    source_path: Path | None = None
    source_stream: BinaryIO
    source_identity: tuple[int, int, int, int] | None = None
    captured_content_digest: bytes | None = None
    if isinstance(source, str | os.PathLike):
        source_path = Path(cast("str | os.PathLike[str]", source))
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(source_path, flags)
        except OSError as error:
            raise AgentBundleError("container_source_not_regular") from error
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            os.close(descriptor)
            raise AgentBundleError("container_source_not_regular")
        source_identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        source_stream = os.fdopen(descriptor, "rb")
    else:
        if not hasattr(source, "read"):
            raise TypeError("source must be a path or binary stream.")
        source_stream = source

    def current_path_identities() -> tuple[
        tuple[int, int, int, int],
        tuple[int, int, int, int],
        bool,
    ]:
        assert source_path is not None
        try:
            descriptor_info = os.fstat(source_stream.fileno())
            path_info = source_path.stat(follow_symlinks=False)
        except (OSError, ValueError) as error:
            raise AgentBundleError("container_source_changed") from error
        return (
            (
                descriptor_info.st_dev,
                descriptor_info.st_ino,
                descriptor_info.st_size,
                descriptor_info.st_mtime_ns,
            ),
            (
                path_info.st_dev,
                path_info.st_ino,
                path_info.st_size,
                path_info.st_mtime_ns,
            ),
            stat.S_ISREG(path_info.st_mode),
        )

    def require_unchanged_path_source(*, verify_content: bool = False) -> None:
        if source_path is None or source_identity is None:
            return
        descriptor_identity, path_identity, path_is_regular = current_path_identities()
        if (
            not path_is_regular
            or descriptor_identity != source_identity
            or path_identity != source_identity
        ):
            raise AgentBundleError("container_source_changed")
        if not verify_content:
            return
        assert captured_content_digest is not None
        digest = sha256()
        byte_count = 0
        try:
            source_stream.seek(0)
            while chunk := source_stream.read(_COPY_CHUNK_BYTES):
                if not isinstance(chunk, bytes):
                    raise TypeError("path source must return bytes.")
                byte_count += len(chunk)
                if byte_count > source_identity[2]:
                    raise AgentBundleError("container_source_changed")
                digest.update(chunk)
        except (OSError, ValueError) as error:
            raise AgentBundleError("container_source_changed") from error
        descriptor_identity, path_identity, path_is_regular = current_path_identities()
        if (
            not path_is_regular
            or descriptor_identity != source_identity
            or path_identity != source_identity
            or byte_count != source_identity[2]
            or digest.digest() != captured_content_digest
        ):
            raise AgentBundleError("container_source_changed")

    try:
        byte_count = 0
        captured_digest = sha256() if source_path is not None else None
        with tempfile.TemporaryFile(mode="w+b") as temporary:
            while chunk := source_stream.read(_COPY_CHUNK_BYTES):
                if not isinstance(chunk, bytes):
                    raise TypeError("source stream must return bytes.")
                byte_count += len(chunk)
                if byte_count > AGENT_BUNDLE_CONTAINER_MAX_BYTES:
                    raise AgentBundleError("container_size_limit_exceeded")
                temporary.write(chunk)
                if captured_digest is not None:
                    captured_digest.update(chunk)
            if captured_digest is not None:
                captured_content_digest = captured_digest.digest()
            require_unchanged_path_source()
            temporary.flush()
            temporary.seek(0)
            yield temporary
            require_unchanged_path_source(verify_content=True)
    finally:
        if source_path is not None:
            source_stream.close()


def _read_container(
    source: BinarySource,
    *,
    destination: Path | None = None,
) -> AgentBundleContainerReceipt:
    with _seekable_source(source) as stream:
        digest = sha256()
        stream.seek(0)
        while chunk := stream.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
        stream.seek(0)
        try:
            with zipfile.ZipFile(stream, "r", allowZip64=True) as archive:
                container_bytes = _validate_archive_structure(stream, archive)
                infos = archive.infolist()
                if infos[0].filename != AGENT_BUNDLE_CONTAINER_MIMETYPE_ENTRY:
                    raise AgentBundleError("container_mimetype_not_first")
                if infos[1].filename != AGENT_BUNDLE_INDEX_FILENAME:
                    raise AgentBundleError("container_index_not_second")
                if _read_entry_bytes(archive, infos[0], max_bytes=len(_MIMETYPE_BYTES)) != (
                    _MIMETYPE_BYTES
                ):
                    raise AgentBundleError("container_mimetype_invalid")
                index_bytes = _read_entry_bytes(
                    archive,
                    infos[1],
                    max_bytes=AGENT_BUNDLE_MAX_INDEX_BYTES,
                )
                try:
                    bundle = AgentBundle.model_validate_json(index_bytes)
                except Exception as error:
                    raise AgentBundleError("bundle_index_invalid") from error
                canonical_index = canonical_durable_json_bytes(
                    bundle.model_dump(mode="json"),
                    "agent_bundle",
                )
                if canonical_index != index_bytes:
                    raise AgentBundleError("bundle_index_not_canonical")
                expected_names = (
                    AGENT_BUNDLE_CONTAINER_MIMETYPE_ENTRY,
                    AGENT_BUNDLE_INDEX_FILENAME,
                    *(
                        _canonical_object_entry(digest_value)
                        for digest_value in bundle.transferred_digests
                    ),
                )
                actual_names = tuple(info.filename for info in infos)
                if actual_names != expected_names:
                    raise AgentBundleError("container_entry_set_or_order_mismatch")
                by_digest = {reference.digest: reference for reference in bundle.closure}
                total_object_bytes = 0
                if destination is not None:
                    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
                    index_path = destination / AGENT_BUNDLE_INDEX_FILENAME
                    descriptor = os.open(index_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, "wb") as index_stream:
                        index_stream.write(index_bytes)
                        index_stream.flush()
                        os.fsync(index_stream.fileno())
                for info, digest_value in zip(infos[2:], bundle.transferred_digests, strict=True):
                    reference = by_digest[digest_value]
                    total_object_bytes += reference.byte_count
                    if total_object_bytes > AGENT_BUNDLE_MAX_TOTAL_BYTES:
                        raise AgentBundleError("bundle_total_size_limit_exceeded")
                    target = (
                        None
                        if destination is None
                        else destination
                        / AGENT_BUNDLE_OBJECT_DIRECTORY
                        / digest_value[:2]
                        / digest_value[2:]
                    )
                    _copy_verified_entry(archive, info, reference, target)
        except zipfile.BadZipFile as error:
            raise AgentBundleError("container_zip_invalid") from error
    return AgentBundleContainerReceipt(
        inspection=_container_inspection(
            bundle,
            transport_sha256=digest.hexdigest(),
            container_bytes=container_bytes,
        ),
        bundle=bundle,
    )


def _zip_info(name: str, *, mimetype: bool, byte_count: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_CANONICAL_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.reserved = 0
    info.flag_bits = 0
    info.volume = 0
    info.internal_attr = 0
    info.external_attr = _CANONICAL_EXTERNAL_ATTR
    info.extra = _VERSION_EXTRA if mimetype else b""
    info.comment = b""
    info.file_size = byte_count
    return info


def _write_bytes_entry(
    archive: zipfile.ZipFile, name: str, value: bytes, *, mimetype: bool
) -> None:
    info = _zip_info(name, mimetype=mimetype, byte_count=len(value))
    with archive.open(info, "w", force_zip64=len(value) >= zipfile.ZIP64_LIMIT) as output:
        output.write(value)


def _write_path_entry(
    archive: zipfile.ZipFile,
    name: str,
    source: Path,
    reference: AgentBundleObjectRef,
) -> None:
    if source.is_symlink() or not source.is_file() or source.stat().st_size != reference.byte_count:
        raise AgentBundleError("bundle_object_size_mismatch", reference.digest)
    info = _zip_info(name, mimetype=False, byte_count=reference.byte_count)
    digest = sha256()
    byte_count = 0
    captured = bytearray() if reference.kind is AgentBundleObjectKind.SNAPSHOT_NODE else None
    with (
        source.open("rb") as input_stream,
        archive.open(
            info,
            "w",
            force_zip64=reference.byte_count >= zipfile.ZIP64_LIMIT,
        ) as output,
    ):
        while chunk := input_stream.read(_COPY_CHUNK_BYTES):
            byte_count += len(chunk)
            if byte_count > reference.byte_count:
                raise AgentBundleError("bundle_object_size_mismatch", reference.digest)
            digest.update(chunk)
            if captured is not None:
                captured.extend(chunk)
            output.write(chunk)
    if byte_count != reference.byte_count:
        raise AgentBundleError("bundle_object_integrity_mismatch", reference.digest)
    if reference.kind is AgentBundleObjectKind.SNAPSHOT_NODE:
        try:
            node = AgentSnapshotNode.model_validate_json(bytes(captured or b""))
        except Exception as error:
            raise AgentBundleError("bundle_snapshot_node_invalid", reference.digest) from error
        if node.digest != reference.digest or canonical_durable_json_bytes(
            node.model_dump(mode="json"),
            "agent_snapshot_node",
        ) != bytes(captured or b""):
            raise AgentBundleError("bundle_object_integrity_mismatch", reference.digest)
    elif digest.hexdigest() != reference.digest:
        raise AgentBundleError("bundle_object_integrity_mismatch", reference.digest)


def _write_container_from_directory(source: Path, destination: Path) -> AgentBundle:
    bundle, _ = AgentBundleCoordinator._read_and_verify_bundle_directory(source)
    index_bytes = (source / AGENT_BUNDLE_INDEX_FILENAME).read_bytes()
    by_digest: Mapping[str, AgentBundleObjectRef] = {
        reference.digest: reference for reference in bundle.closure
    }
    with destination.open("xb") as raw:
        with zipfile.ZipFile(
            raw,
            "w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
            strict_timestamps=True,
        ) as archive:
            archive.comment = b""
            _write_bytes_entry(
                archive,
                AGENT_BUNDLE_CONTAINER_MIMETYPE_ENTRY,
                _MIMETYPE_BYTES,
                mimetype=True,
            )
            _write_bytes_entry(
                archive,
                AGENT_BUNDLE_INDEX_FILENAME,
                index_bytes,
                mimetype=False,
            )
            for digest_value in bundle.transferred_digests:
                reference = by_digest[digest_value]
                object_path = (
                    source / AGENT_BUNDLE_OBJECT_DIRECTORY / digest_value[:2] / digest_value[2:]
                )
                _write_path_entry(
                    archive,
                    _canonical_object_entry(digest_value),
                    object_path,
                    reference,
                )
        raw.flush()
        os.fsync(raw.fileno())
    return bundle


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    if os.name == "nt":
        return
    directories = [Path(path) for path, _names, _files in os.walk(root, topdown=False)]
    for directory in directories:
        _fsync_directory(directory)


def _regular_file_identity(path: Path) -> tuple[int, int, int, int]:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as error:
        raise AgentBundleError("container_staging_changed") from error
    if not stat.S_ISREG(info.st_mode):
        raise AgentBundleError("container_staging_changed")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _unlink_owned_file(path: Path, expected_identity: tuple[int, int, int, int]) -> None:
    try:
        if _regular_file_identity(path) == expected_identity:
            path.unlink()
            _fsync_directory(path.parent)
    except (AgentBundleError, FileNotFoundError, OSError):
        return


def _publish_file(
    source: Path,
    destination: Path,
    *,
    expected_identity: tuple[int, int, int, int],
) -> None:
    if _regular_file_identity(source) != expected_identity:
        raise AgentBundleError("container_staging_changed")
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError:
        raise AgentBundleError("container_destination_conflict") from None
    except OSError as error:
        raise AgentBundleError("atomic_container_publication_unsupported") from error
    if _regular_file_identity(destination) != expected_identity:
        raise AgentBundleError("container_staging_changed")
    _fsync_directory(destination.parent)
    source.unlink()


def _native_rename_no_replace(function_name: str, arguments: tuple[object, ...]) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = getattr(libc, function_name)
    except AttributeError as error:
        raise AgentBundleError("atomic_container_publication_unsupported") from error
    function.restype = ctypes.c_int
    if function(*arguments) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise AgentBundleError("bundle_destination_conflict")
    raise AgentBundleError("atomic_container_publication_unsupported") from OSError(
        error_number, os.strerror(error_number)
    )


def _publish_directory(source: Path, destination: Path) -> None:
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError:
            raise AgentBundleError("bundle_destination_conflict") from None
        return
    if sys.platform == "darwin":
        _native_rename_no_replace(
            "renamex_np",
            (os.fsencode(source), os.fsencode(destination), _RENAME_EXCL),
        )
        return
    if sys.platform.startswith("linux"):
        _native_rename_no_replace(
            "renameat2",
            (
                _AT_FDCWD,
                os.fsencode(source),
                _AT_FDCWD,
                os.fsencode(destination),
                _RENAME_NOREPLACE,
            ),
        )
        return
    raise AgentBundleError("atomic_container_publication_unsupported")


def _prepare_empty_transactional_stream(destination: BinaryIO) -> None:
    """Require a destination that can be rolled back and validated before ack."""

    try:
        initial_position = destination.tell()
        destination.seek(0, io.SEEK_END)
        initial_size = destination.tell()
        destination.seek(initial_position)
        if initial_position != 0 or initial_size != 0:
            raise AgentBundleError("container_stream_destination_not_empty")
        probe = destination.read(0)
        if not isinstance(probe, bytes):
            raise TypeError("binary stream read did not return bytes")
        destination.seek(0)
        destination.truncate(0)
        destination.flush()
        destination.seek(0)
    except (AttributeError, OSError, TypeError, ValueError, io.UnsupportedOperation) as error:
        raise AgentBundleError("container_stream_transaction_unsupported") from error


def _reset_transactional_stream(destination: BinaryIO) -> None:
    destination.seek(0)
    destination.truncate(0)
    destination.flush()
    destination.seek(0)


def _copy_to_stream(
    source: Path,
    destination: BinaryIO,
    *,
    expected_receipt: AgentBundleContainerReceipt,
) -> AgentBundleContainerReceipt:
    _prepare_empty_transactional_stream(destination)
    try:
        with source.open("rb") as input_stream:
            while chunk := input_stream.read(_COPY_CHUNK_BYTES):
                offset = 0
                while offset < len(chunk):
                    accepted = destination.write(chunk[offset:])
                    if type(accepted) is not int or accepted <= 0 or accepted > len(chunk) - offset:
                        raise AgentBundleError("container_stream_write_invalid")
                    offset += accepted
        destination.flush()
        destination.seek(0)
        published = _read_container(destination)
        if published != expected_receipt:
            raise AgentBundleError("container_stream_publication_changed")
        destination.seek(0, io.SEEK_END)
        return published
    except BaseException:
        try:
            _reset_transactional_stream(destination)
        except BaseException as rollback_error:
            raise AgentBundleError("container_stream_rollback_failed") from rollback_error
        raise


def pack_agent_bundle(
    source: str | os.PathLike[str],
    destination: BinaryDestination,
) -> AgentBundleContainerReceipt:
    """Pack one canonical directory bundle into deterministic ``.cayu`` bytes."""

    source_path = Path(source)
    if not source_path.is_absolute():
        raise ValueError("Bundle source must be absolute.")
    if isinstance(destination, str | os.PathLike):
        destination_path = Path(cast("str | os.PathLike[str]", destination))
        if not destination_path.is_absolute():
            raise ValueError("Container destination must be absolute.")
        destination_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with cooperative_path_lock(
            destination_path.parent,
            destination_path.name,
            lock_directory_name="cayu-agent-bundle-container-locks",
        ):
            if destination_path.exists():
                with tempfile.TemporaryDirectory(
                    prefix=".cayu-agent-bundle-pack-",
                    dir=destination_path.parent,
                ) as temporary_root:
                    canonical_path = Path(temporary_root) / "bundle.cayu"
                    _write_container_from_directory(source_path, canonical_path)
                    canonical = _read_container(canonical_path)
                    existing = _read_container(destination_path)
                if existing != canonical:
                    raise AgentBundleError("container_destination_conflict")
                return existing
            with tempfile.TemporaryDirectory(
                prefix=".cayu-agent-bundle-pack-",
                dir=destination_path.parent,
            ) as temporary_root:
                temporary = Path(temporary_root) / "bundle.cayu"
                _write_container_from_directory(source_path, temporary)
                receipt = _read_container(temporary)
                validated_identity = _regular_file_identity(temporary)
                _publish_file(
                    temporary,
                    destination_path,
                    expected_identity=validated_identity,
                )
                try:
                    published = _read_container(destination_path)
                except BaseException:
                    _unlink_owned_file(destination_path, validated_identity)
                    raise
                if published != receipt:
                    _unlink_owned_file(destination_path, validated_identity)
                    raise AgentBundleError("container_staging_changed")
                return published
    if not hasattr(destination, "write"):
        raise TypeError("destination must be a path or binary stream.")
    with tempfile.TemporaryDirectory(prefix="cayu-agent-bundle-pack-") as root:
        temporary = Path(root) / "bundle.cayu"
        _write_container_from_directory(source_path, temporary)
        receipt = _read_container(temporary)
        return _copy_to_stream(
            temporary,
            destination,
            expected_receipt=receipt,
        )


def inspect_agent_bundle_container(source: BinarySource) -> AgentBundleContainerInspection:
    """Validate every transferred byte and return bounded safe metadata."""

    return _read_container(source).inspection


def unpack_agent_bundle_container(
    source: BinarySource,
    destination: str | os.PathLike[str],
) -> AgentBundleContainerReceipt:
    """Unpack exact declared entries into a canonical directory atomically."""

    destination_path = Path(destination)
    if not destination_path.is_absolute():
        raise ValueError("Bundle destination must be absolute.")
    destination_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with cooperative_path_lock(
        destination_path.parent,
        destination_path.name,
        lock_directory_name="cayu-agent-bundle-container-unpack-locks",
    ):
        if destination_path.exists():
            existing_bundle, _ = AgentBundleCoordinator._read_and_verify_bundle_directory(
                destination_path
            )
            incoming = _read_container(source)
            if incoming.bundle != existing_bundle:
                raise AgentBundleError("bundle_destination_conflict")
            return incoming
        with tempfile.TemporaryDirectory(
            prefix=".cayu-agent-bundle-unpack-",
            dir=destination_path.parent,
        ) as temporary_root:
            temporary = Path(temporary_root) / "bundle"
            receipt = _read_container(source, destination=temporary)
            verified, _ = AgentBundleCoordinator._read_and_verify_bundle_directory(temporary)
            if verified != receipt.bundle:
                raise AgentBundleError("container_bundle_identity_mismatch")
            _fsync_tree_directories(temporary)
            _publish_directory(temporary, destination_path)
            _fsync_directory(destination_path.parent)
            return receipt


__all__ = [
    "AGENT_BUNDLE_CONTAINER_EXTENSION",
    "AGENT_BUNDLE_CONTAINER_MAX_BYTES",
    "AGENT_BUNDLE_CONTAINER_MAX_ENTRIES",
    "AGENT_BUNDLE_CONTAINER_MEDIA_TYPE",
    "AGENT_BUNDLE_CONTAINER_MIMETYPE_ENTRY",
    "AGENT_BUNDLE_CONTAINER_SCHEMA_VERSION",
    "AgentBundleContainerInspection",
    "AgentBundleContainerReceipt",
    "inspect_agent_bundle_container",
    "pack_agent_bundle",
    "unpack_agent_bundle_container",
]
