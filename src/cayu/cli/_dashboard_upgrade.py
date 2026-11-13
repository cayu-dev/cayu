"""Bring a dashboard made with ``cayu dashboard eject`` up to the installed Cayu release.

Eject writes ``cayu-dashboard-source.json``, which records the sha256 of every file as it
was ejected. Comparing the project's files with it tells which ones the project edited, and
comparing it with the installed release's source bundle tells which ones the release
changed. :func:`plan_dashboard_upgrade` decides what to do from those three alone, and
:func:`apply_dashboard_upgrade` carries a plan out, so another per-release refresh can reuse
either part.

There is no copy of the old release's content, only its hashes, so a file both sides
changed is not merged: the project's file stays and the release's is written next to it as
``FILE.cayu-VERSION``. Generated files (the release metadata, the OpenAPI baseline and the
generated client) and the manifest always take the release's content, because
``npm run check:api`` requires exactly what the installed Cayu generates.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from cayu._dashboard_artifacts import (
    BUNDLE_MANIFEST_NAME,
    GENERATED_ROOT,
    GENERATED_SOURCE_FILES,
)
from cayu.cli.dashboard import (
    DashboardSourceError,
    ValidatedDashboardSource,
    _is_windows_reparse_point,
    _parse_manifest,
    _sha256,
    load_dashboard_source,
)
from cayu.runtime.checks import DiagnosticSeverity, ProjectDiagnostic

_VERSION_PATTERN = re.compile(
    r"(?P<release>\d+(?:\.\d+)*)(?:(?P<pre>a|b|rc)(?P<pre_number>\d+))?"
    r"(?:\.post(?P<post>\d+))?(?:\.dev(?P<dev>\d+))?(?:\+[0-9A-Za-z.]+)?\Z"
)
_SIDECAR_VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+!_-]*")
_PRE_RELEASE_ORDER = {"a": 0, "b": 1, "rc": 2}
_TEMPORARY_PREFIX = ".cayu-upgrade-"
_NEW_FILE_MODE = 0o644
_NEW_DIRECTORY_MODE = 0o755
# Folders `cayu check` doesn't look in for an ejected dashboard.
_SKIPPED_DIRECTORIES = frozenset(
    {"node_modules", "venv", "__pycache__", "dist", "build", "data", "tests"}
)
_MAX_SCANNED_DIRECTORIES = 2000


class UpgradeAction(StrEnum):
    ADD = "add"
    """The release adds the file and the project has none there."""
    REPLACE = "replace"
    """The project didn't edit the file and the release changed it."""
    REMOVE = "remove"
    """The project didn't edit the file and the release removed it."""
    KEEP = "keep"
    """The project edited (or deleted) the file and the release didn't change it."""
    MERGE = "merge"
    """Both changed the file: the release's version goes next to it, to merge by hand.

    Also an edited file with a sidecar from an earlier upgrade still next to it, since that
    merge is still pending.
    """
    KEEP_REMOVED = "keep_removed"
    """The project edited the file and the release removed it: kept, for the project to decide."""


_DESCRIPTIONS = {
    UpgradeAction.ADD: "added by the release",
    UpgradeAction.REPLACE: "changed by the release",
    UpgradeAction.REMOVE: "removed by the release",
    UpgradeAction.KEEP: "edited here; the release didn't change it",
    UpgradeAction.MERGE: "edited here, with a release's changes to merge by hand",
    UpgradeAction.KEEP_REMOVED: (
        "edited here and removed by the release; kept, and no longer tracked by the manifest"
    ),
}
MANUAL_ACTIONS = frozenset({UpgradeAction.MERGE, UpgradeAction.KEEP_REMOVED})


@dataclass(frozen=True)
class DashboardUpgradeStep:
    path: str
    action: UpgradeAction
    sidecar: str | None = None
    """Where this release's version goes, for ``merge`` when the release changed the file."""
    pending_sidecars: tuple[str, ...] = ()
    """Sidecars earlier upgrades left next to the file, whose merges are still pending."""
    replaces_edit: bool = False
    """The project edited this generated file, which takes the release's content anyway."""

    def as_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "action": self.action.value,
            "sidecar": self.sidecar,
            "pending_sidecars": list(self.pending_sidecars),
            "replaces_edit": self.replaces_edit,
        }


@dataclass(frozen=True)
class DashboardUpgradePlan:
    from_version: str
    to_version: str
    steps: tuple[DashboardUpgradeStep, ...]
    """Every file that needs something done or that the project should know about."""
    unchanged: tuple[str, ...]
    """Files that already match the release."""

    @property
    def manual(self) -> tuple[DashboardUpgradeStep, ...]:
        """The files to merge or decide on by hand."""
        return tuple(step for step in self.steps if step.action in MANUAL_ACTIONS)


@dataclass(frozen=True)
class DashboardUpgradeResult:
    directory: Path
    plan: DashboardUpgradePlan
    applied: bool


def plan_dashboard_upgrade(
    ejected: Mapping[str, str],
    release: Mapping[str, bytes],
    project: Mapping[str, str | None],
    *,
    from_version: str,
    to_version: str,
    sidecars: Mapping[str, Iterable[str]] | None = None,
) -> DashboardUpgradePlan:
    """Decide what an upgrade does with each file, without touching the filesystem.

    ``ejected`` maps each path to its sha256 as ejected (the old manifest's ``files``),
    ``release`` maps each path of the new release to its content (the manifest included),
    and ``project`` maps a path to the sha256 of the project's file, or None when it has
    none. A path missing from ``project`` counts as absent. ``sidecars`` maps a path to the
    sidecars of it the project still has (see :func:`find_sidecars`), from any release: an
    edited file stays a pending merge while one is there.
    """

    existing_sidecars = {} if sidecars is None else sidecars

    steps: list[DashboardUpgradeStep] = []
    unchanged: list[str] = []
    for path in sorted(set(ejected) | set(release)):
        old = ejected.get(path)
        new = _sha256(release[path]) if path in release else None
        current = project.get(path)
        if current == new:
            unchanged.append(path)
            continue
        generated = _is_generated(path)
        sidecar = None
        pending: tuple[str, ...] = ()
        if generated or current == old:
            if new is None:
                action = UpgradeAction.REMOVE
            elif current is None:
                action = UpgradeAction.ADD
            else:
                action = UpgradeAction.REPLACE
        elif new is None:
            action = UpgradeAction.KEEP_REMOVED
        else:
            if new != old:
                sidecar = sidecar_path(path, to_version)
            pending = tuple(
                sorted(item for item in existing_sidecars.get(path, ()) if item != sidecar)
            )
            action = UpgradeAction.MERGE if sidecar or pending else UpgradeAction.KEEP
        steps.append(
            DashboardUpgradeStep(
                path=path,
                action=action,
                sidecar=sidecar,
                pending_sidecars=pending,
                replaces_edit=generated and current != old and old is not None,
            )
        )
    return DashboardUpgradePlan(
        from_version=from_version,
        to_version=to_version,
        steps=tuple(steps),
        unchanged=tuple(unchanged),
    )


def sidecar_path(path: str, version: str) -> str:
    """Where the release's version of a file both sides changed goes."""
    _require_sidecar_version(version)
    return f"{path}.cayu-{version}"


def find_sidecars(root: Path, paths: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """The regular files named like a sidecar of each path (``FILE.cayu-VERSION``)."""

    listings: dict[str, tuple[str, ...]] = {}
    found: dict[str, tuple[str, ...]] = {}
    for path in paths:
        relative = PurePosixPath(path)
        parent = relative.parent.as_posix()
        if parent not in listings:
            listings[parent] = _regular_file_names(root, relative.parent.parts)
        prefix = f"{relative.name}.cayu-"
        names = tuple(
            name
            for name in listings[parent]
            if name.startswith(prefix) and _SIDECAR_VERSION.fullmatch(name[len(prefix) :])
        )
        if names:
            found[path] = tuple((relative.parent / name).as_posix() for name in names)
    return found


def read_project_digests(root: Path, paths: Iterable[str]) -> dict[str, str | None]:
    """The sha256 of each project file at ``paths`` under ``root``, or None when absent.

    Refuses a path that is not a regular file or that passes through a link, so neither
    planning nor applying ever follows one out of the project.
    """

    digests: dict[str, str | None] = {}
    for path in paths:
        target = _inspect_project_path(root, path)
        digests[path] = None if target is None else _sha256(_read_regular_file(target, path))
    return digests


def apply_dashboard_upgrade(
    root: Path, plan: DashboardUpgradePlan, release: Mapping[str, bytes]
) -> None:
    """Carry out ``plan`` in the project at ``root`` with the release's ``release`` content.

    Writes go through a temporary file and a rename. The manifest is written last, so an
    interrupted upgrade can be run again: files already done then match the release.
    """

    # Refuse a link or a non-file at any path the plan touches before changing anything.
    for step in plan.steps:
        _inspect_project_path(root, step.path)
        if step.sidecar is not None:
            _inspect_project_path(root, step.sidecar)
    write_manifest = False
    removals: list[str] = []
    for step in plan.steps:
        if step.path == BUNDLE_MANIFEST_NAME:
            write_manifest = True
        elif step.action in (UpgradeAction.ADD, UpgradeAction.REPLACE):
            _write_project_file(root, step.path, release[step.path])
        elif step.action is UpgradeAction.MERGE and step.sidecar is not None:
            # An existing sidecar is this release's version, perhaps being merged: keep it.
            if _inspect_project_path(root, step.sidecar) is None:
                _write_project_file(root, step.sidecar, release[step.path])
        elif step.action is UpgradeAction.REMOVE:
            removals.append(step.path)
    for path in removals:
        _remove_project_file(root, path)
    if write_manifest:
        _write_project_file(root, BUNDLE_MANIFEST_NAME, release[BUNDLE_MANIFEST_NAME])


def upgrade_dashboard_source(
    directory: Path,
    *,
    dry_run: bool = False,
    release: ValidatedDashboardSource | None = None,
) -> DashboardUpgradeResult:
    """Plan and, unless ``dry_run``, apply an upgrade to the installed release.

    ``release`` defaults to the installed Cayu's validated dashboard source bundle.
    """

    root = _project_root(directory)
    manifest_path = _inspect_project_path(root, BUNDLE_MANIFEST_NAME)
    if manifest_path is None:
        raise DashboardSourceError(
            f"{directory} has no {BUNDLE_MANIFEST_NAME}, so it wasn't made by "
            "`cayu dashboard eject` (or the file was removed); there is nothing to compare "
            "its files with"
        )
    try:
        ejected = _parse_manifest(_read_regular_file(manifest_path, BUNDLE_MANIFEST_NAME))
    except DashboardSourceError as exc:
        raise DashboardSourceError(f"{manifest_path} is not a valid eject manifest: {exc}") from exc
    if release is None:
        release = load_dashboard_source()
    from_version = ejected.cayu_version
    to_version = release.manifest.cayu_version
    order = compare_versions(from_version, to_version)
    if order is None:
        raise DashboardSourceError(
            f"can't tell whether Cayu {from_version}, which {directory} was ejected from, is "
            f"older than the installed Cayu {to_version}"
        )
    if order > 0:
        raise DashboardSourceError(
            f"{directory} was ejected from Cayu {from_version}, newer than the installed Cayu "
            f"{to_version}; install Cayu {from_version} or later instead"
        )
    ejected_files = {item.path: item.sha256 for item in ejected.files}
    paths = sorted(set(ejected_files) | set(release.contents))
    project = read_project_digests(root, paths)
    plan = plan_dashboard_upgrade(
        ejected_files,
        release.contents,
        project,
        from_version=from_version,
        to_version=to_version,
        sidecars=find_sidecars(root, paths),
    )
    if not dry_run:
        apply_dashboard_upgrade(root, plan, release.contents)
    return DashboardUpgradeResult(directory=root, plan=plan, applied=not dry_run)


def compare_versions(left: str, right: str) -> int | None:
    """-1, 0 or 1 as ``left`` is older than, equal to or newer than ``right``.

    None when either isn't a plain release version (``1.2.3``, with optional ``rcN``,
    ``.postN``, ``.devN`` and a local part).
    """

    left_key, right_key = _version_key(left), _version_key(right)
    if left_key is None or right_key is None:
        return 0 if left == right else None
    return (left_key > right_key) - (left_key < right_key)


def run_dashboard_upgrade(args: argparse.Namespace) -> int:
    try:
        result = upgrade_dashboard_source(args.directory, dry_run=args.dry_run)
    except (DashboardSourceError, OSError) as exc:
        if args.json:
            error = {"code": "DASHBOARD_UPGRADE_REFUSED", "message": str(exc)}
            print(json.dumps({"error": error}, indent=2))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(_result_json(result), indent=2))
    else:
        print(_render_result(result))
    return 1 if result.plan.manual else 0


def dashboard_source_findings(root: Path) -> tuple[ProjectDiagnostic, ...]:
    """Warn about each ejected dashboard whose manifest names another Cayu version."""

    try:
        installed = importlib.metadata.version("cayu")
    except importlib.metadata.PackageNotFoundError:
        return ()
    findings: list[ProjectDiagnostic] = []
    for manifest in _ejected_manifests(root):
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        version = value.get("cayu_version") if isinstance(value, dict) else None
        if not isinstance(version, str) or version == installed:
            continue
        directory = manifest.parent.relative_to(root).as_posix() or "."
        newer = (compare_versions(version, installed) or 0) > 0
        hint = (
            f"It came from a newer Cayu: install Cayu {version}, or eject the installed "
            "release's source again."
            if newer
            else f"Run `cayu dashboard upgrade {directory}`, merge any files it lists, then "
            f"run `npm ci` and `npm run check:api` in {directory}."
        )
        findings.append(
            ProjectDiagnostic(
                code="DASHBOARD_SOURCE_VERSION_MISMATCH",
                severity=DiagnosticSeverity.WARNING,
                subject="dashboard",
                path=manifest.relative_to(root).as_posix(),
                message=(
                    f"The Developer dashboard in {directory} was ejected from Cayu {version}, "
                    f"but the installed Cayu is {installed}. Its `npm run check:api` fails "
                    "until it matches, and its build follows the server contract of the "
                    "release it came from."
                ),
                hint=hint,
                tags=("configuration", "deploy"),
                parameters={
                    "directory": directory,
                    "ejected_version": version,
                    "cayu_version": installed,
                },
                documentation_anchor="cayu guide diagnostics#dashboard-source-version-mismatch",
                verification_command="cayu check",
            )
        )
    return tuple(findings)


def _ejected_manifests(root: Path) -> list[Path]:
    """Eject manifests in ``root`` and in folders up to two levels below it."""

    found: list[Path] = []
    scanned = 0
    level = [root]
    for depth in range(3):
        next_level: list[Path] = []
        for directory in level:
            manifest = directory / BUNDLE_MANIFEST_NAME
            try:
                if manifest.is_file() and not manifest.is_symlink():
                    found.append(manifest)
            except OSError:
                # An unreadable folder (say, a database bind mount) holds no dashboard.
                continue
            if depth == 2:
                continue
            try:
                children = sorted(directory.iterdir())
            except OSError:
                continue
            for child in children:
                if child.name.startswith(".") or child.name in _SKIPPED_DIRECTORIES:
                    continue
                try:
                    if child.is_symlink() or not child.is_dir():
                        continue
                except OSError:
                    continue
                scanned += 1
                if scanned > _MAX_SCANNED_DIRECTORIES:
                    return found
                next_level.append(child)
        level = next_level
    return found


def _result_json(result: DashboardUpgradeResult) -> dict[str, Any]:
    plan = result.plan
    return {
        "directory": str(result.directory),
        "from_version": plan.from_version,
        "to_version": plan.to_version,
        "applied": result.applied,
        "files": [step.as_json() for step in plan.steps],
        "unchanged": len(plan.unchanged),
        "manual": [step.path for step in plan.manual],
    }


def _render_result(result: DashboardUpgradeResult) -> str:
    plan = result.plan
    directory = result.directory
    header = f"Dashboard source in {directory}: Cayu {plan.from_version} -> {plan.to_version}"
    lines = [header + ("" if result.applied else " (dry run; nothing written)")]
    for step in plan.steps:
        note = (
            f"the eject manifest, rewritten for Cayu {plan.to_version}"
            if step.path == BUNDLE_MANIFEST_NAME
            else _DESCRIPTIONS[step.action]
        )
        if step.replaces_edit:
            note += "; edited here, but generated, so the release's version replaces it"
        if step.sidecar is not None:
            note += f"; this release's version is in {step.sidecar}"
        if step.pending_sidecars:
            note += f"; still to merge: {', '.join(step.pending_sidecars)}"
        lines.append(f"  {step.action.value:<12} {step.path} ({note})")
    lines.append(f"{len(plan.unchanged)} other files already match the release.")
    merges = [step for step in plan.manual if step.action is UpgradeAction.MERGE]
    if merges:
        lines += [
            "",
            "Merge each release's version into these files by hand, then delete its "
            ".cayu-VERSION file:",
        ]
        lines += [
            f"  {step.path} <- {sidecar}"
            for step in merges
            for sidecar in (*step.pending_sidecars, *([step.sidecar] if step.sidecar else []))
        ]
    removed = [step for step in plan.manual if step.action is UpgradeAction.KEEP_REMOVED]
    if removed:
        lines += ["", "The release removed these files, which the project edited. Keep or delete:"]
        lines += [f"  {step.path}" for step in removed]
    if plan.steps:
        written = {step.path for step in plan.steps if step.action is not UpgradeAction.KEEP}
        install = "`npm ci`, then " if written & {"package.json", "package-lock.json"} else ""
        lines += ["", f"Then, in {directory}, run {install}`npm run check:api`."]
    return "\n".join(lines)


def _is_generated(path: str) -> bool:
    return (
        path == BUNDLE_MANIFEST_NAME
        or path in GENERATED_SOURCE_FILES
        or path.startswith(f"{GENERATED_ROOT}/")
    )


def _require_sidecar_version(version: str) -> None:
    if _SIDECAR_VERSION.fullmatch(version) is None:
        raise DashboardSourceError(f"Cayu version {version!r} can't name a file")


def _version_key(version: str) -> tuple[Any, ...] | None:
    match = _VERSION_PATTERN.fullmatch(version)
    if match is None:
        return None
    release = tuple(int(part) for part in match["release"].split("."))
    while len(release) > 1 and release[-1] == 0:
        release = release[:-1]
    if match["pre"] is not None:
        pre = (_PRE_RELEASE_ORDER[match["pre"]], int(match["pre_number"]))
    elif match["dev"] is not None and match["post"] is None:
        pre = (-1, 0)
    else:
        pre = (3, 0)
    post = -1 if match["post"] is None else int(match["post"])
    dev = float("inf") if match["dev"] is None else int(match["dev"])
    return (release, pre, post, dev)


def _project_root(directory: Path) -> Path:
    try:
        root = directory.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DashboardSourceError(f"could not open {directory}: {exc}") from exc
    if not root.is_dir():
        raise DashboardSourceError(f"{directory} is not a directory")
    return root


def _is_link(identity: os.stat_result) -> bool:
    return stat.S_ISLNK(identity.st_mode) or _is_windows_reparse_point(identity)


def _inspect_project_path(root: Path, path: str) -> Path | None:
    """The regular file at ``path`` under ``root``, or None when nothing is there.

    Refuses links, and anything other than a folder or a regular file on the way.
    """

    parts = PurePosixPath(path).parts
    current = root
    for index, part in enumerate(parts):
        current = current / part
        try:
            identity = current.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise DashboardSourceError(f"could not inspect {current}: {exc}") from exc
        relative = "/".join(parts[: index + 1])
        if _is_link(identity):
            raise DashboardSourceError(
                f"{relative} in {root} is a link; the upgrade doesn't follow links"
            )
        last = index == len(parts) - 1
        if last and not stat.S_ISREG(identity.st_mode):
            raise DashboardSourceError(f"{relative} in {root} is not a regular file")
        if not last and not stat.S_ISDIR(identity.st_mode):
            raise DashboardSourceError(f"{relative} in {root} is not a folder")
    return current


def _regular_file_names(root: Path, parts: tuple[str, ...]) -> tuple[str, ...]:
    current = root
    for part in parts:
        current = current / part
        try:
            identity = current.lstat()
        except OSError:
            return ()
        if _is_link(identity) or not stat.S_ISDIR(identity.st_mode):
            return ()
    try:
        with os.scandir(current) as entries:
            return tuple(
                sorted(entry.name for entry in entries if entry.is_file(follow_symlinks=False))
            )
    except OSError:
        return ()


def _read_regular_file(path: Path, relative: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DashboardSourceError(f"could not read {relative}: {exc}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise DashboardSourceError(f"{relative} is not a regular file")
        return handle.read()


def _project_directory(root: Path, parts: tuple[str, ...]) -> Path:
    """The folder ``parts`` under ``root``, made when missing; refuses links on the way."""

    current = root
    for part in parts:
        current = current / part
        try:
            identity = current.lstat()
        except FileNotFoundError:
            current.mkdir(mode=_NEW_DIRECTORY_MODE)
            identity = current.lstat()
        if _is_link(identity) or not stat.S_ISDIR(identity.st_mode):
            raise DashboardSourceError(f"{current} is not a folder; the upgrade stopped there")
    return current


def _write_project_file(root: Path, path: str, content: bytes) -> None:
    parts = PurePosixPath(path).parts
    parent = _project_directory(root, parts[:-1])
    existing = _inspect_project_path(root, path)
    mode = _NEW_FILE_MODE if existing is None else stat.S_IMODE(existing.lstat().st_mode)
    descriptor, temporary = tempfile.mkstemp(prefix=_TEMPORARY_PREFIX, dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.chmod(temporary, mode)
        os.replace(temporary, parent / parts[-1])
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _remove_project_file(root: Path, path: str) -> None:
    target = _inspect_project_path(root, path)
    if target is None:
        return
    target.unlink()
    # Drop the folders the removal left empty, up to the project root.
    for parent in target.parents:
        if parent == root or root not in parent.parents:
            break
        try:
            parent.rmdir()
        except OSError:
            break
