"""Build the Lambda MicroVM image code artifact expected by infra.yaml."""

from __future__ import annotations

import argparse
import tomllib
import zipfile
from pathlib import Path

from cayu.cli.lambda_microvm import _load_validated_artifact

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
SIDECAR = HERE.parent / "lambda_microvm_sidecar"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def package(output: Path) -> Path:
    archive, _provenance = _package_with_sidecar_provenance(output)
    return archive


def _package_with_sidecar_provenance(
    output: Path,
) -> tuple[Path, dict[str, str | int]]:
    """Package one validated sidecar snapshot and return its provenance."""
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sidecar = _load_validated_artifact(
        SIDECAR,
        expected_cayu_version=project["project"]["version"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(sidecar.contents.items()):
            _write_file(archive, name, content, executable=name == "entrypoint.sh")
    manifest = sidecar.manifest
    return (
        output,
        {
            "sidecar_artifact_version": manifest.artifact_version,
            "sidecar_cayu_version": manifest.cayu_version,
            "sidecar_content_digest": manifest.content_digest,
            "sidecar_protocol_version": manifest.protocol_version,
        },
    )


def _write_file(
    archive: zipfile.ZipFile,
    name: str,
    content: bytes,
    *,
    executable: bool,
) -> None:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o100755 if executable else 0o100644) << 16
    archive.writestr(info, content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(package(args.output))


if __name__ == "__main__":
    main()
