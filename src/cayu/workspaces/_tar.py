from __future__ import annotations

from collections.abc import Sequence

TAR_MEMBER_OVERHEAD_BYTES = 3072
TAR_ARCHIVE_OVERHEAD_BYTES = 10 * 1024
TAR_BLOCK_BYTES = 512
TAR_PAX_RECORD_OVERHEAD_BYTES = 128


def tar_archive_size_bound(logical_bytes: int, paths: Sequence[str]) -> int:
    """Conservative upper bound for the uncompressed tar produced by Cayu.

    The bound covers regular member headers, content padding, PAX path metadata,
    and Python tarfile's final 10 KiB record padding.
    """

    if type(logical_bytes) is not int:
        raise TypeError("Tar logical_bytes must be an integer.")
    if logical_bytes < 0:
        raise ValueError("Tar logical_bytes must not be negative.")
    return (
        logical_bytes
        + (TAR_MEMBER_OVERHEAD_BYTES * len(paths))
        + sum(tar_path_metadata_bound(path) for path in paths)
        + TAR_ARCHIVE_OVERHEAD_BYTES
    )


def tar_path_metadata_bound(path: str) -> int:
    encoded_path_bytes = len(path.encode("utf-8", errors="surrogateescape"))
    pax_record_bytes = encoded_path_bytes + TAR_PAX_RECORD_OVERHEAD_BYTES
    padded_record_bytes = (
        (pax_record_bytes + TAR_BLOCK_BYTES - 1) // TAR_BLOCK_BYTES * TAR_BLOCK_BYTES
    )
    # One extended-header block plus its padded ``path=...`` record. Include
    # this for every member even when the name fits in a ustar header.
    return TAR_BLOCK_BYTES + padded_record_bytes
