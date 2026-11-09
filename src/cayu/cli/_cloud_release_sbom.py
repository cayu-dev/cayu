"""`cayu cloud deployment sbom RELEASE [--output FILE]`: a Release's SBOM, saved as a file.

Cloud keeps the CycloneDX SBOM each Release's image build wrote, the file SBOM admission
scanned, and serves it at `GET /v1/applications/{id}/deployments/{release}/sbom`, byte for
byte, with its SHA-256 in `X-Cayu-Content-Digest`. A refusal keeps at most 50 findings; the
SBOM has every package. The CLI streams it to a temporary file beside FILE, checks the digest
and that it is a CycloneDX JSON document, and only then puts it at FILE, which must not exist
yet (default `RELEASE.cdx.json` in the current directory). Nothing is written otherwise.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError, DownloadedFile
from cayu.cli._cloud_data import _publish_new
from cayu.cli._cloud_settings import _cell

#: The largest SBOM Cloud serves (its admission limit too).
SBOM_MAX_BYTES = 10 * 1024 * 1024
_MEDIA_TYPE = "application/vnd.cyclonedx+json"
_DIGEST = re.compile(r"sha256:([0-9a-f]{64})")
_SPEC_VERSION = re.compile(r"[0-9]{1,3}(?:\.[0-9]{1,3}){0,2}")
_TEMPORARY_PREFIX = ".cayu-sbom-"
# A Release status as Cloud names it.
_RELEASE_STATUS = re.compile(r"[a-z][a-z_]{0,31}")


class SbomDownloadError(CloudApiError):
    """An SBOM download Cloud refused, with the fields its refusal adds: the Release's
    ``status`` when it has no SBOM because its image was never built."""

    def __init__(self, exc: CloudApiError, *, details: dict[str, str]) -> None:
        super().__init__(
            exc.category,
            str(exc),
            status_code=exc.status_code,
            detail=exc.detail,
            code=exc.code,
            body=exc.body,
        )
        self.details = details


def default_output(deployment_id: str) -> Path:
    """Where the SBOM goes without --output: the name Cloud's download gives it."""

    return Path(f"{deployment_id}.cdx.json")


def download_release_sbom(
    client: CloudApiClient,
    *,
    application_id: str,
    deployment_id: str,
    output: Path | None,
) -> dict[str, Any]:
    """Download the Release's SBOM, verify it, and write it to a new file."""

    chosen = default_output(deployment_id) if output is None else output
    # Absolute, so the temporary file lands beside the file it becomes.
    target = Path(os.path.abspath(chosen.expanduser()))
    _refuse_existing(target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=_TEMPORARY_PREFIX)
    except OSError as exc:
        raise _unwritable(target, exc) from None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            downloaded = _download(client, application_id, deployment_id, stream)
        expected = _expected_digest(downloaded.headers)
        if downloaded.sha256 != expected:
            raise CloudApiError(
                "sbom_digest_mismatch",
                f"The downloaded SBOM of {deployment_id} doesn't match the digest Cayu Cloud "
                f"sent (sha256:{expected}); nothing was written. Run the command again.",
            )
        spec_version, components = _cyclonedx(Path(temporary), deployment_id)
        try:
            _publish_new(temporary, target)
        except FileExistsError:
            raise _exists(target) from None
        except OSError as exc:
            raise _unwritable(target, exc) from None
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {
        "application_id": application_id,
        "deployment_id": deployment_id,
        "output": str(target),
        "format": "CycloneDX",
        "spec_version": spec_version,
        "components": components,
        "digest": f"sha256:{downloaded.sha256}",
        "bytes": downloaded.size,
    }


def render_sbom(output: Mapping[str, Any]) -> str:
    result = output["result"]
    components = result["components"]
    described = f"{components} component{'' if components == 1 else 's'}"
    if result["spec_version"] is not None:
        described += f", CycloneDX {result['spec_version']}"
    return (
        f"Wrote the SBOM of release {_cell(result['deployment_id'])} to "
        f"{_cell(result['output'], limit=4096)} ({result['bytes']} bytes, {described}).\n"
        f"Verified digest {result['digest']}."
    )


def _download(
    client: CloudApiClient, application_id: str, deployment_id: str, stream: BinaryIO
) -> DownloadedFile:
    path = (
        f"/v1/applications/{quote(application_id, safe='')}/deployments/"
        f"{quote(deployment_id, safe='')}/sbom"
    )
    try:
        return client.download(path, stream, max_bytes=SBOM_MAX_BYTES, accept=_MEDIA_TYPE)
    except CloudApiError as exc:
        raise _refusal(exc, deployment_id) from None


def _refusal(exc: CloudApiError, deployment_id: str) -> CloudApiError:
    """Cloud's refusal, said in terms of what the caller can do next."""

    if exc.status_code == 405 or (exc.status_code == 404 and exc.code is None):
        return CloudApiError(
            "sbom_download_unsupported",
            "This Cayu Cloud can't download a release's SBOM yet.",
            status_code=exc.status_code,
        )
    if exc.category == "api_response_too_large":
        return CloudApiError(
            "sbom_invalid",
            f"Cayu Cloud sent more than {SBOM_MAX_BYTES // (1024 * 1024)} MiB for the SBOM of "
            f"{deployment_id}, more than Cloud keeps; nothing was written.",
        )
    if exc.code == "release_sbom_not_found":
        status = (exc.body or {}).get("status")
        if isinstance(status, str) and _RELEASE_STATUS.fullmatch(status):
            return SbomDownloadError(exc, details={"status": status})
    return exc


def _expected_digest(headers: Mapping[str, str]) -> str:
    match = _DIGEST.fullmatch(headers.get("x-cayu-content-digest", "").strip())
    if match is None:
        raise CloudApiError(
            "api_response_invalid",
            "Cayu Cloud sent the SBOM without a valid X-Cayu-Content-Digest, so it can't be "
            "verified; nothing was written.",
        )
    return match.group(1)


def _cyclonedx(path: Path, deployment_id: str) -> tuple[str | None, int]:
    """The SBOM's CycloneDX spec version (None when it names none it can show) and how many
    components it lists; anything that isn't a CycloneDX JSON document is refused."""

    try:
        document = json.loads(path.read_bytes())
    # A body nested past the interpreter's limit raises RecursionError, not ValueError.
    except (OSError, RecursionError, ValueError):
        document = None
    if not isinstance(document, dict) or document.get("bomFormat") != "CycloneDX":
        raise CloudApiError(
            "sbom_invalid",
            f"Cayu Cloud's SBOM for {deployment_id} isn't a CycloneDX JSON document; nothing "
            "was written.",
        )
    components = document.get("components", [])
    spec_version = document.get("specVersion")
    return (
        spec_version
        if isinstance(spec_version, str) and _SPEC_VERSION.fullmatch(spec_version)
        else None,
        len(components) if isinstance(components, list) else 0,
    )


def _refuse_existing(output: Path) -> None:
    if output.is_dir():
        raise CloudApiError(
            "output_exists", f"{output} is a directory; name a new file with --output."
        )
    if output.exists() or output.is_symlink():
        raise _exists(output)


def _exists(output: Path) -> CloudApiError:
    return CloudApiError("output_exists", f"{output} exists; name a new file with --output.")


def _unwritable(output: Path, error: OSError) -> CloudApiError:
    return CloudApiError(
        "output_unwritable",
        f"Could not write {output}: {error.strerror or type(error).__name__}.",
    )
