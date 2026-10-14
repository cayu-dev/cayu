"""Build or verify Cayu's deterministic editable-dashboard source bundle."""

from __future__ import annotations

import argparse
import io
import json
import stat
import tomllib
import zipfile
from pathlib import Path

from cayu._dashboard_artifacts import (
    BUILD_IGNORED_NAMES,
    BUILD_IGNORED_PARTS,
    BUILD_INPUT_FILES,
    BUNDLE_GLOB,
    BUNDLE_MANIFEST_NAME,
    BUNDLE_PACKAGE_ROOT,
    COMPILED_PACKAGE_ROOT,
    RELEASE_METADATA_FILE,
    SOURCE_ROOT,
    render_release_metadata,
    source_bundle_name,
)
from cayu._server_contract_version import SERVER_CONTRACT_VERSION
from cayu.cli.dashboard import (
    contents_digest,
    render_dashboard_source_manifest,
    validate_dashboard_source_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "src" / "cayu" / BUNDLE_PACKAGE_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of rewriting when the committed bundle is stale.",
    )
    args = parser.parse_args(argv)

    cayu_version = project_version(REPO_ROOT)
    bundle_name = source_bundle_name(cayu_version)
    expected_path = BUNDLE_ROOT / bundle_name
    expected = build_bundle(cayu_version=cayu_version, repo_root=REPO_ROOT)
    existing = sorted(BUNDLE_ROOT.glob(BUNDLE_GLOB))

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


def build_bundle(*, cayu_version: str, repo_root: Path) -> bytes:
    """Render and validate a source bundle from one staged repository tree."""
    contents = _dashboard_contents(repo_root)
    expected_release_metadata = render_release_metadata(
        cayu_version=cayu_version, server_contract_version=SERVER_CONTRACT_VERSION
    )
    release_metadata_path = RELEASE_METADATA_FILE
    if contents.get(release_metadata_path) != expected_release_metadata:
        raise ValueError(
            f"{release_metadata_path} must match Cayu {cayu_version} and server contract "
            f"v{SERVER_CONTRACT_VERSION}"
        )
    compiled_digest = contents_digest(
        tree_contents(repo_root / "src" / "cayu" / COMPILED_PACKAGE_ROOT)
    )
    manifest = render_dashboard_source_manifest(
        contents,
        cayu_version=cayu_version,
        server_contract_version=SERVER_CONTRACT_VERSION,
        compiled_dashboard_digest=compiled_digest,
    )
    archive_contents = {**contents, BUNDLE_MANIFEST_NAME: manifest}
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


def _dashboard_contents(repo_root: Path) -> dict[str, bytes]:
    contents = tree_contents(repo_root / SOURCE_ROOT, ignore_build_products=True)
    package = json.loads(contents["package.json"])
    del package["scripts"]["build:package"]
    contents["package.json"] = (json.dumps(package, indent=2) + "\n").encode()
    for name in BUILD_INPUT_FILES:
        contents[name] = (repo_root / name).read_bytes()
    return contents


def tree_contents(root: Path, *, ignore_build_products: bool = False) -> dict[str, bytes]:
    """Read ordinary files, optionally excluding declared local build products."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"source directory is missing: {root}")
    contents: dict[str, bytes] = {}
    casefolded: set[str] = set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ignore_build_products and (
            BUILD_IGNORED_PARTS.intersection(relative.parts) or path.name in BUILD_IGNORED_NAMES
        ):
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


def project_version(repo_root: Path) -> str:
    """Read the repository-owned version used in the editable bundle name."""
    with (repo_root / "pyproject.toml").open("rb") as project_file:
        value = tomllib.load(project_file)["project"]["version"]
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("pyproject.toml contains an invalid project version")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
