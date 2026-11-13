"""The version-matched ``@cayu-tech/dashboard-parts`` package for an Agent's web app.

The wheel carries the package as an npm tarball at its own version (built by
``scripts/build_dashboard.py``). ``cayu dashboard parts`` writes it where the app installs
it from as a ``file:`` dependency, and ``cayu check`` warns when the app's lockfile
resolves another version, since the parts follow the server contract of the Cayu they
shipped with.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from cayu._dashboard_artifacts import (
    PARTS_NAME_PATTERN,
    PARTS_PACKAGE_NAME,
    PARTS_PACKAGE_ROOT,
    parts_tarball_name,
    validate_parts_tarball,
)
from cayu.runtime.checks import DiagnosticSeverity, ProjectDiagnostic

DEFAULT_DESTINATION = Path("web/vendor")
_LOCKFILE = "web/package-lock.json"
# What a git checkout gives the file, so whoever builds the web app can read it (mkstemp
# makes it 0600).
_TARBALL_MODE = 0o644


class DashboardPartsError(RuntimeError):
    """The packaged parts are missing or invalid, or the destination can't take them."""


@dataclass(frozen=True)
class DashboardPartsResult:
    tarball: Path
    version: str
    removed: tuple[Path, ...]
    package_json: Path | None
    spec: str | None
    """The ``file:`` spec now in package.json, or None when it lists no ``file:`` spec."""
    changed: bool
    """Whether package.json changed."""


def _installed_version() -> str:
    try:
        return importlib.metadata.version("cayu")
    except importlib.metadata.PackageNotFoundError as exc:
        raise DashboardPartsError("the installed Cayu has no distribution metadata") from exc


def packaged_parts_tarball() -> tuple[str, str, bytes]:
    """The installed Cayu's parts tarball: its version, file name and bytes, checked."""

    version = _installed_version()
    name = parts_tarball_name(version)
    resource = files("cayu").joinpath(*PARTS_PACKAGE_ROOT.split("/"), name)
    if not resource.is_file():
        raise DashboardPartsError(
            f"the installed Cayu {version} does not include {PARTS_PACKAGE_NAME} ({name})"
        )
    content = resource.read_bytes()
    try:
        validate_parts_tarball(content, cayu_version=version)
    except ValueError as exc:
        raise DashboardPartsError(str(exc)) from exc
    return version, name, content


def write_dashboard_parts(destination: Path) -> DashboardPartsResult:
    """Write the parts tarball into ``destination`` and point package.json's spec at it.

    Other versions' tarballs in ``destination`` are removed. The package.json next to
    ``destination`` (``web/package.json`` for ``web/vendor``) is updated only when it
    already depends on the package through a ``file:`` spec.
    """

    version, name, content = packaged_parts_tarball()
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise DashboardPartsError(f"{destination} is not a directory")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name
    descriptor, temporary = tempfile.mkstemp(prefix=".cayu-parts-", dir=destination)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.chmod(temporary, _TARBALL_MODE)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    removed = tuple(
        path
        for path in sorted(destination.iterdir())
        if path.name != name and PARTS_NAME_PATTERN.fullmatch(path.name) and path.is_file()
    )
    for path in removed:
        path.unlink()
    package_json = destination.resolve().parent / "package.json"
    if not package_json.is_file():
        return DashboardPartsResult(target, version, removed, None, None, False)
    spec = "file:" + os.path.relpath(target.resolve(), package_json.parent).replace(os.sep, "/")
    changed, listed = _update_spec(package_json, spec)
    return DashboardPartsResult(
        target, version, removed, package_json, spec if listed else None, changed
    )


def _update_spec(package_json: Path, spec: str) -> tuple[bool, bool]:
    """Point an existing ``file:`` dependency at ``spec``; return (changed, listed)."""

    text = package_json.read_text(encoding="utf-8")
    try:
        manifest = json.loads(text)
    except ValueError as exc:
        raise DashboardPartsError(f"{package_json} is not valid JSON: {exc}") from exc
    listed = changed = False
    for section in ("dependencies", "devDependencies"):
        dependencies = _object(_object(manifest).get(section))
        current = dependencies.get(PARTS_PACKAGE_NAME)
        if isinstance(current, str) and current.startswith("file:"):
            listed = True
            if current != spec:
                dependencies[PARTS_PACKAGE_NAME] = spec
                changed = True
    if changed:
        indent = re.search(r"\n([ \t]+)\S", text)
        package_json.write_text(
            json.dumps(manifest, indent=indent.group(1) if indent else 2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    return changed, listed


def add_parts_parser(dashboard_commands: Any) -> None:
    parts = dashboard_commands.add_parser(
        "parts",
        help=f"Write the version-matched {PARTS_PACKAGE_NAME} package for a web app.",
        description=(
            f"Write {PARTS_PACKAGE_NAME}, the Developer dashboard's pages as parts an app "
            "imports, at the installed Cayu's version into DESTINATION (default web/vendor). "
            "When the package.json next to DESTINATION already depends on it through a "
            "`file:` spec, that spec is pointed at the new tarball. Run `npm install` there "
            "afterwards."
        ),
    )
    parts.add_argument(
        "destination",
        type=Path,
        nargs="?",
        default=None,
        metavar="DESTINATION",
        help="Folder for the tarball (default: web/vendor, run from the project root).",
    )


def run_dashboard_parts(args: argparse.Namespace) -> int:
    destination = args.destination
    if destination is None:
        if not DEFAULT_DESTINATION.parent.is_dir():
            print(
                "error: there is no web/ folder here; run this from the project root "
                "or pass DESTINATION",
                file=sys.stderr,
            )
            return 1
        destination = DEFAULT_DESTINATION
    try:
        result = write_dashboard_parts(destination)
    except (DashboardPartsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {result.tarball} ({PARTS_PACKAGE_NAME} {result.version})")
    for path in result.removed:
        print(f"removed {path}")
    if result.package_json is None:
        print(f"Install it in your web app: npm install {result.tarball}")
        return 0
    app = result.package_json.parent
    if result.spec is None:
        relative = os.path.relpath(result.tarball.resolve(), app)
        print(f"{result.package_json} doesn't depend on it yet. In {app}, run:")
        print(f"  npm install ./{relative}")
        return 0
    if result.changed:
        print(f'updated {result.package_json}: "{PARTS_PACKAGE_NAME}": "{result.spec}"')
    print(f"In {app}, run `npm install` to update package-lock.json, then rebuild.")
    return 0


def dashboard_parts_findings(root: Path) -> tuple[ProjectDiagnostic, ...]:
    """Warn when the web app's lockfile resolves the parts to another Cayu version."""

    lockfile = root / _LOCKFILE
    if not lockfile.is_file():
        return ()
    try:
        lock = json.loads(lockfile.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    version = _locked_version(lock)
    if version is None:
        return ()
    try:
        installed = _installed_version()
    except DashboardPartsError:
        return ()
    if version == installed:
        return ()
    return (
        ProjectDiagnostic(
            code="DASHBOARD_PARTS_VERSION_MISMATCH",
            severity=DiagnosticSeverity.WARNING,
            subject="web app",
            path=_LOCKFILE,
            message=(
                f"{_LOCKFILE} resolves {PARTS_PACKAGE_NAME} to {version}, but the installed "
                f"Cayu is {installed}. The parts' types and calls follow the server contract "
                "of the Cayu they shipped with."
            ),
            hint="Run `cayu dashboard parts`, then `npm install` and rebuild the web app.",
            tags=("configuration", "deploy"),
            parameters={"locked_version": version, "cayu_version": installed},
            verification_command="cayu check",
        ),
    )


def _object(value: object) -> dict[str, Any]:
    """A JSON object as a dict, or an empty one for anything else."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _locked_version(lock: object) -> str | None:
    entry = _object(_object(lock).get("packages")).get(f"node_modules/{PARTS_PACKAGE_NAME}")
    if entry is None:
        # Lockfile version 1 lists dependencies instead of packages.
        entry = _object(_object(lock).get("dependencies")).get(PARTS_PACKAGE_NAME)
    version = _object(entry).get("version")
    if not isinstance(version, str):
        return None
    if version.startswith("file:"):
        # Version 1 names a local tarball by its spec; the version is in the file name
        # `cayu dashboard parts` gave it, and unknown otherwise.
        match = PARTS_NAME_PATTERN.fullmatch(version.replace("\\", "/").rpartition("/")[2])
        return match.group("version") if match else None
    return version
