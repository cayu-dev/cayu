"""Generate or verify the canonical Lambda MicroVM sidecar manifest."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "examples" / "aws" / "lambda_microvm_sidecar"
MANIFEST_NAME = "cayu-lambda-microvm-sidecar-manifest.json"


def _expected_manifest() -> bytes:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from cayu.cli.lambda_microvm import _collect_resource_files, _render_manifest
    from cayu.runners.aws_lambda_microvm import LAMBDA_MICROVM_PROTOCOL_VERSION

    with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
        cayu_version = tomllib.load(project_file)["project"]["version"]
    contents = _collect_resource_files(SOURCE_ROOT)
    contents.pop(MANIFEST_NAME, None)
    return _render_manifest(
        contents,
        cayu_version=cayu_version,
        protocol_version=LAMBDA_MICROVM_PROTOCOL_VERSION,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of rewriting when the checked-in manifest is stale.",
    )
    args = parser.parse_args(argv)

    manifest_path = SOURCE_ROOT / MANIFEST_NAME
    expected = _expected_manifest()
    if args.check:
        try:
            actual = manifest_path.read_bytes()
        except OSError as exc:
            print(f"could not read {manifest_path}: {exc}", file=sys.stderr)
            return 1
        if actual != expected:
            print(
                "Lambda MicroVM sidecar manifest is stale; run "
                "`uv run python scripts/generate_sidecar_manifest.py`.",
                file=sys.stderr,
            )
            return 1
        print(f"validated {manifest_path}")
        return 0

    manifest_path.write_bytes(expected)
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
