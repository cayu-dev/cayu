"""Shared output declarations for dashboard generation and package validation.

Source paths are relative to the editable dashboard. Package paths are relative
to ``cayu`` in an installed distribution. The bundle manifest still records the
complete source inventory and compiled digest; these declarations define which
generated outputs must be present in that inventory.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath

SOURCE_ROOT = "dashboard"
# The canonical command builds this input tree plus the root redistribution
# documents. Python's bundle writer uses stored ZIP entries with fixed metadata;
# only the frontend compiler needs the pinned execution environment.
BUILD_INPUT_FILES = ("LICENSE", "NOTICE")
BUILD_IGNORED_PARTS = frozenset({"__pycache__", "dist", "node_modules"})
BUILD_IGNORED_NAMES = frozenset({".DS_Store"})
BUILD_PLATFORM = "linux/amd64"
BUILD_IMAGE = (
    "node:22.18.0-bookworm-slim@"
    "sha256:0d130e2ee18e88e1561375276daced6bff032539200173f2daf48c2e33f38ff5"
)
BUILD_NODE_VERSION = "22.18.0"
BUILD_NPM_VERSION = "10.9.3"
BUILD_ENVIRONMENT = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC"}
COMPILED_PACKAGE_ROOT = "server/dashboard"
COMPILED_REQUIRED_FILES = frozenset(
    {"LICENSE", "NOTICE", "REDISTRIBUTION.md", "THIRD_PARTY_LICENSES.md", "index.html"}
)
BUNDLE_PACKAGE_ROOT = "data/dashboard_source"
BUNDLE_MANIFEST_NAME = "cayu-dashboard-source.json"
_BUNDLE_PREFIX = "cayu-dashboard-source-"
BUNDLE_GLOB = f"{_BUNDLE_PREFIX}*.zip"
BUNDLE_NAME_PATTERN = re.compile(rf"{re.escape(_BUNDLE_PREFIX)}(?P<version>[^/]+)\.zip\Z")
# @cayu-tech/dashboard-parts, packed by `npm run build:parts` at the Cayu version. Apps
# install it from `cayu dashboard parts` as a `file:` dependency.
PARTS_PACKAGE_ROOT = "data/dashboard_parts"
PARTS_PACKAGE_NAME = "@cayu-tech/dashboard-parts"
_PARTS_PREFIX = "cayu-dashboard-parts-"
PARTS_NAME_PATTERN = re.compile(rf"{re.escape(_PARTS_PREFIX)}(?P<version>[^/]+)\.tgz\Z")
# Files every packed parts tarball holds, under npm's `package/` folder.
PARTS_REQUIRED_FILES = frozenset(
    {
        "package.json",
        "styles.css",
        "LICENSE",
        "NOTICE",
        "dist/parts/index.js",
        "dist/parts/index.d.ts",
        "dist/parts/inbox/index.js",
        "dist/parts/inbox/index.d.ts",
    }
)

OPENAPI_FILE = "server-openapi.json"
GENERATED_ROOT = "src/lib/generated"
API_TYPES_ROOT = f"{GENERATED_ROOT}/server-api"
API_TYPE_FILES = frozenset(f"{API_TYPES_ROOT}/{name}" for name in ("index.ts", "types.gen.ts"))
RELEASE_METADATA_FILE = "src/lib/release-metadata.ts"
EVAL_TEXT_LIMITS_FILE = f"{GENERATED_ROOT}/eval-text-limits.ts"
LABEL_LIMITS_FILE = f"{GENERATED_ROOT}/label-limits.ts"
GENERATED_SOURCE_FILES = frozenset(
    {OPENAPI_FILE, RELEASE_METADATA_FILE, EVAL_TEXT_LIMITS_FILE, LABEL_LIMITS_FILE, *API_TYPE_FILES}
)
REQUIRED_SOURCE_FILES = GENERATED_SOURCE_FILES | frozenset(
    {
        "LICENSE",
        "NOTICE",
        "README.md",
        "REDISTRIBUTION.md",
        "THIRD_PARTY_LICENSES.md",
        "biome.json",
        "package-lock.json",
        "package.json",
        "scripts/check-third-party-licenses.mjs",
        "scripts/finalize-third-party-licenses.mjs",
        "scripts/generate-api-types.py",
        "scripts/run-python.mjs",
        "parts/package.json",
        "scripts/pack-parts.mjs",
        "scripts/parts-declarations.mjs",
        "src/kit/ui.ts",
        "src/main.tsx",
        "src/parts/index.ts",
        "tests/dashboard-capabilities.test.mjs",
        "tests/run-python.test.mjs",
        "third_party/shadcn-ui.LICENSE",
        "tsconfig.json",
        "tsconfig.parts.json",
        "vite.config.ts",
        "vite.parts.config.ts",
    }
)


def source_bundle_name(cayu_version: str) -> str:
    return f"{_BUNDLE_PREFIX}{cayu_version}.zip"


def parts_tarball_name(cayu_version: str) -> str:
    return f"{_PARTS_PREFIX}{cayu_version}.tgz"


def render_release_metadata(*, cayu_version: str, server_contract_version: str) -> bytes:
    """Render dashboard versions from the package and server contract owners."""
    return (
        "// Generated from Cayu package metadata and cayu._server_contract_version.\n"
        "// Regenerate with npm run generate:api in the intended Cayu environment.\n"
        f'export const DASHBOARD_SOURCE_CAYU_VERSION = "{cayu_version}"\n'
        f'export const SUPPORTED_SERVER_CONTRACT_VERSION = "{server_contract_version}"\n'
    ).encode()


def validate_generated_source_inventory(paths: Iterable[str]) -> None:
    """Reject incomplete outputs and undeclared files in the generated directory."""
    actual = set(paths)
    missing = sorted(GENERATED_SOURCE_FILES - actual)
    unexpected = sorted(
        path
        for path in actual - GENERATED_SOURCE_FILES
        if path == GENERATED_ROOT or path.startswith(f"{GENERATED_ROOT}/")
    )
    details = []
    if missing:
        details.append(f"missing generated files: {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected generated files: {', '.join(unexpected)}")
    if details:
        raise ValueError("dashboard generated source inventory is invalid; " + "; ".join(details))


def generated_api_contents(contents: Mapping[str, bytes]) -> dict[str, bytes]:
    """Preserve the version-1 bundle manifest's API-only digest population."""
    return {
        path: content
        for path, content in contents.items()
        if path == OPENAPI_FILE or path.startswith(f"{API_TYPES_ROOT}/")
    }


_PARTS_MAX_BYTES = 4 * 1024 * 1024


def validate_parts_tarball(content: bytes, *, cayu_version: str) -> None:
    """Check a packed parts tarball: npm's layout, the required files, and its version.

    Raises ``ValueError`` naming what is wrong. Reads the archive without extracting it.
    """

    if len(content) > _PARTS_MAX_BYTES:
        raise ValueError("dashboard parts tarball is larger than 4 MiB")
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("package",):
                    raise ValueError(f"dashboard parts tarball has an unsafe path: {member.name}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"dashboard parts tarball has a link: {member.name}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"dashboard parts tarball can't read {member.name}")
                files[str(PurePosixPath(*path.parts[1:]))] = extracted.read()
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ValueError(f"dashboard parts tarball is not a gzipped tar archive: {exc}") from exc
    missing = sorted(PARTS_REQUIRED_FILES - files.keys())
    if missing:
        raise ValueError(f"dashboard parts tarball lacks {', '.join(missing)}")
    try:
        manifest = json.loads(files["package.json"])
    except ValueError as exc:
        raise ValueError("dashboard parts tarball has an invalid package.json") from exc
    if not isinstance(manifest, dict) or manifest.get("name") != PARTS_PACKAGE_NAME:
        raise ValueError(f"dashboard parts tarball is not {PARTS_PACKAGE_NAME}")
    if manifest.get("version") != cayu_version:
        raise ValueError(
            f"dashboard parts tarball is version {manifest.get('version')!r}, "
            f"not Cayu's {cayu_version}"
        )
