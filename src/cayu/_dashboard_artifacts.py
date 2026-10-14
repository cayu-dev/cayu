"""Shared output declarations for dashboard generation and package validation.

Source paths are relative to the editable dashboard. Package paths are relative
to ``cayu`` in an installed distribution. The bundle manifest still records the
complete source inventory and compiled digest; these declarations define which
generated outputs must be present in that inventory.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

SOURCE_ROOT = "dashboard"
COMPILED_PACKAGE_ROOT = "server/dashboard"
COMPILED_REQUIRED_FILES = frozenset(
    {"LICENSE", "NOTICE", "REDISTRIBUTION.md", "THIRD_PARTY_LICENSES.md", "index.html"}
)
BUNDLE_PACKAGE_ROOT = "data/dashboard_source"
BUNDLE_MANIFEST_NAME = "cayu-dashboard-source.json"
_BUNDLE_PREFIX = "cayu-dashboard-source-"
BUNDLE_GLOB = f"{_BUNDLE_PREFIX}*.zip"
BUNDLE_NAME_PATTERN = re.compile(rf"{re.escape(_BUNDLE_PREFIX)}(?P<version>[^/]+)\.zip\Z")

OPENAPI_FILE = "server-openapi.json"
GENERATED_ROOT = "src/lib/generated"
API_TYPES_ROOT = f"{GENERATED_ROOT}/server-api"
API_TYPE_FILES = frozenset(f"{API_TYPES_ROOT}/{name}" for name in ("index.ts", "types.gen.ts"))
RELEASE_METADATA_FILE = "src/lib/release-metadata.ts"
EVAL_TEXT_LIMITS_FILE = f"{GENERATED_ROOT}/eval-text-limits.ts"
GENERATED_SOURCE_FILES = frozenset(
    {OPENAPI_FILE, RELEASE_METADATA_FILE, EVAL_TEXT_LIMITS_FILE, *API_TYPE_FILES}
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
        "src/main.tsx",
        "tests/dashboard-capabilities.test.mjs",
        "tests/run-python.test.mjs",
        "third_party/shadcn-ui.LICENSE",
        "tsconfig.json",
        "vite.config.ts",
    }
)


def source_bundle_name(cayu_version: str) -> str:
    return f"{_BUNDLE_PREFIX}{cayu_version}.zip"


def render_release_metadata(*, cayu_version: str, server_contract_version: str) -> bytes:
    """Render the existing TypeScript version contract without changing its bytes."""
    return (
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
