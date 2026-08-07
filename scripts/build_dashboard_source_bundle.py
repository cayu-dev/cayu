"""Build or verify Cayu's deterministic editable-dashboard source bundle."""

from __future__ import annotations

import argparse
import io
import json
import stat
import tomllib
import zipfile
from pathlib import Path

from cayu._server_contract_version import SERVER_CONTRACT_VERSION
from cayu.cli.dashboard import (
    contents_digest,
    render_dashboard_source_manifest,
    validate_dashboard_source_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = REPO_ROOT / "dashboard"
COMPILED_DASHBOARD_ROOT = REPO_ROOT / "src" / "cayu" / "server" / "dashboard"
BUNDLE_ROOT = REPO_ROOT / "src" / "cayu" / "data" / "dashboard_source"
MANIFEST_NAME = "cayu-dashboard-source.json"
_IGNORED_PARTS = {"__pycache__", "dist", "node_modules"}
_IGNORED_NAMES = {".DS_Store"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of rewriting when the committed bundle is stale.",
    )
    args = parser.parse_args(argv)

    cayu_version = _project_version()
    bundle_name = f"cayu-dashboard-source-{cayu_version}.zip"
    expected_path = BUNDLE_ROOT / bundle_name
    expected = _build_bundle(cayu_version=cayu_version)
    existing = sorted(BUNDLE_ROOT.glob("cayu-dashboard-source-*.zip"))

    if args.check:
        if existing != [expected_path] or expected_path.read_bytes() != expected:
            print(
                "Dashboard source bundle is stale; run "
                "`uv run python scripts/build_dashboard_source_bundle.py`."
            )
            return 1
        print(f"validated {expected_path}")
        return 0

    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    for path in existing:
        if path != expected_path:
            path.unlink()
    expected_path.write_bytes(expected)
    print(f"wrote {expected_path}")
    return 0


def _build_bundle(*, cayu_version: str) -> bytes:
    contents = _dashboard_contents()
    expected_release_metadata = (
        f'export const DASHBOARD_SOURCE_CAYU_VERSION = "{cayu_version}"\n'
        f'export const SUPPORTED_SERVER_CONTRACT_VERSION = "{SERVER_CONTRACT_VERSION}"\n'
    ).encode()
    release_metadata_path = "src/lib/release-metadata.ts"
    if contents.get(release_metadata_path) != expected_release_metadata:
        raise ValueError(
            f"{release_metadata_path} must match Cayu {cayu_version} and server contract "
            f"v{SERVER_CONTRACT_VERSION}"
        )
    compiled_digest = contents_digest(_tree_contents(COMPILED_DASHBOARD_ROOT))
    manifest = render_dashboard_source_manifest(
        contents,
        cayu_version=cayu_version,
        server_contract_version=SERVER_CONTRACT_VERSION,
        compiled_dashboard_digest=compiled_digest,
    )
    archive_contents = {**contents, MANIFEST_NAME: manifest}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in sorted(archive_contents.items()):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(member, content)
    bundle = output.getvalue()
    validate_dashboard_source_bundle(
        bundle,
        expected_cayu_version=cayu_version,
        expected_server_contract_version=SERVER_CONTRACT_VERSION,
    )
    return bundle


def _dashboard_contents() -> dict[str, bytes]:
    contents = _tree_contents(DASHBOARD_ROOT)
    package = json.loads(contents["package.json"])
    del package["scripts"]["build:package"]
    contents["package.json"] = (json.dumps(package, indent=2) + "\n").encode()
    contents["LICENSE"] = (REPO_ROOT / "LICENSE").read_bytes()
    contents["NOTICE"] = (REPO_ROOT / "NOTICE").read_bytes()
    return contents


def _tree_contents(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        raise ValueError(f"source directory is missing: {root}")
    contents: dict[str, bytes] = {}
    casefolded: set[str] = set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if _IGNORED_PARTS.intersection(relative.parts) or path.name in _IGNORED_NAMES:
            continue
        if path.is_symlink():
            raise ValueError(f"source tree must not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"source tree contains an unsupported entry: {path}")
        name = relative.as_posix()
        if name.casefold() in casefolded:
            raise ValueError(f"source tree contains case-colliding paths: {name}")
        casefolded.add(name.casefold())
        contents[name] = path.read_bytes()
    return contents


def _project_version() -> str:
    with (REPO_ROOT / "pyproject.toml").open("rb") as project_file:
        value = tomllib.load(project_file)["project"]["version"]
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("pyproject.toml contains an invalid project version")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
