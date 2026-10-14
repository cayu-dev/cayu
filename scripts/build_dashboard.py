"""Rebuild or check packaged dashboard outputs with the canonical Linux toolchain."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from cayu._dashboard_artifacts import (
    BUILD_ENVIRONMENT,
    BUILD_IMAGE,
    BUILD_INPUT_FILES,
    BUILD_NODE_VERSION,
    BUILD_NPM_VERSION,
    BUILD_PLATFORM,
    BUNDLE_PACKAGE_ROOT,
    COMPILED_PACKAGE_ROOT,
    SOURCE_ROOT,
    source_bundle_name,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_dashboard_source_bundle import (  # noqa: E402
    build_bundle,
    project_version,
    tree_contents,
)
from scripts.check_release_artifacts import validate_dashboard_build  # noqa: E402


def _inputs(root: Path) -> dict[str, bytes]:
    contents = {
        f"{SOURCE_ROOT}/{name}": content
        for name, content in tree_contents(root / SOURCE_ROOT, ignore_build_products=True).items()
    }
    for name in BUILD_INPUT_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"build input must be an ordinary file: {name}")
        contents[name] = path.read_bytes()
    # Respect effective attributes, including nested binary overrides, before
    # normalizing editor-produced CRLF in the declared UTF-8 text inputs.
    result = subprocess.run(
        ["git", "check-attr", "-z", "--stdin", "text"],
        cwd=root,
        input="\0".join(contents) + "\0",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=15,
    )
    records = result.stdout.split("\0")[:-1]
    for name, _, attribute in zip(records[::3], records[1::3], records[2::3], strict=True):
        content = contents[name]
        if attribute not in {"set", "auto"} or b"\0" in content:
            continue
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        contents[name] = content.replace(b"\r\n", b"\n")
    return contents


def _write_files(root: Path, contents: dict[str, bytes]) -> None:
    for name, content in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _outputs(root: Path) -> dict[str, bytes]:
    result = {}
    for package_path in (COMPILED_PACKAGE_ROOT, BUNDLE_PACKAGE_ROOT):
        prefix = f"src/cayu/{package_path}"
        directory = root / prefix
        if directory.is_symlink() or directory.exists():
            result.update(
                {f"{prefix}/{name}": content for name, content in tree_contents(directory).items()}
            )
    return result


def _build_frontend(stage: Path) -> None:
    """Mount only disposable inputs and outputs; keep npm dependencies in the container."""
    name = f"cayu-dashboard-{uuid.uuid4().hex}"
    output = stage / "src" / "cayu" / COMPILED_PACKAGE_ROOT
    output.mkdir(parents=True)
    command = ["docker", "run", "--rm", "--init", "--name", name, "--platform", BUILD_PLATFORM]
    if hasattr(os, "getuid"):
        command += ["--user", f"{os.getuid()}:{os.getgid()}"]
    for key, value in {**BUILD_ENVIRONMENT, "HOME": "/tmp"}.items():
        command += ["--env", f"{key}={value}"]
    command += [
        "--volume",
        f"{stage}:/input:ro",
        "--volume",
        f"{output}:/output",
        BUILD_IMAGE,
        "sh",
        "-ec",
        "\n".join(
            (
                "umask 022",
                f'test "$(node --version)" = {shlex.quote("v" + BUILD_NODE_VERSION)}',
                f'test "$(npm --version)" = {shlex.quote(BUILD_NPM_VERSION)}',
                "mkdir /tmp/build",
                "cp -R /input/. /tmp/build/",
                f"cd /tmp/build/{SOURCE_ROOT}",
                "npm ci --include=dev --no-audit --no-fund",
                "npm run build:package",
                f"cp -R /tmp/build/src/cayu/{COMPILED_PACKAGE_ROOT}/. /output/",
            )
        ),
    ]
    try:
        subprocess.run(command, check=True, timeout=1200)
    except BaseException:
        # A timed-out or interrupted Docker client can leave its container alive.
        # Remove only this invocation's container; never prune shared Docker state.
        try:
            subprocess.run(
                ["docker", "rm", "--force", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            print(f"Could not remove temporary build container {name}.", file=sys.stderr)
        raise


def _changes(actual: dict[str, bytes], expected: dict[str, bytes]) -> list[str]:
    return [
        *[f"added   {name}" for name in sorted(expected.keys() - actual.keys())],
        *[f"removed {name}" for name in sorted(actual.keys() - expected.keys())],
        *[
            f"changed {name}"
            for name in sorted(actual.keys() & expected.keys())
            if actual[name] != expected[name]
        ],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Report stale outputs without writing."
    )
    args = parser.parse_args(argv)
    inputs = _inputs(ROOT)
    original = _outputs(ROOT)
    version = project_version(ROOT)
    subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], check=True, timeout=15)
    print(
        f"Dashboard toolchain: {BUILD_PLATFORM}, Node {BUILD_NODE_VERSION}, npm {BUILD_NPM_VERSION}"
    )
    with tempfile.TemporaryDirectory(prefix="cayu-dashboard-build-") as directory:
        stage = Path(directory)
        _write_files(stage, inputs)
        _build_frontend(stage)
        compiled = tree_contents(stage / "src" / "cayu" / COMPILED_PACKAGE_ROOT)
        bundle = build_bundle(cayu_version=version, repo_root=stage)
        validate_dashboard_build(compiled, bundle, package_version=version)
        _write_files(
            stage, {f"src/cayu/{BUNDLE_PACKAGE_ROOT}/{source_bundle_name(version)}": bundle}
        )
        expected = _outputs(stage)

    if _inputs(ROOT) != inputs or project_version(ROOT) != version or _outputs(ROOT) != original:
        raise ValueError("Dashboard inputs or outputs changed during the build; rerun the command.")
    changes = _changes(original, expected)
    if not changes:
        print("Packaged dashboard assets and source bundle match the canonical build.")
        return 0
    print("\n".join(changes))
    if args.check:
        print("Dashboard outputs are stale; run `uv run python scripts/build_dashboard.py`.")
        return 1
    for name in original.keys() - expected.keys():
        (ROOT / name).unlink()
    # Remove empty output directories from the leaves up. This also lets new
    # paths replace their parents' capitalization on case-insensitive filesystems.
    # Keep the output roots and any directories that still contain retained files.
    for package_path in (COMPILED_PACKAGE_ROOT, BUNDLE_PACKAGE_ROOT):
        directory = ROOT / "src" / "cayu" / package_path
        for path in sorted(directory.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    _write_files(
        ROOT, {name: value for name, value in expected.items() if original.get(name) != value}
    )
    print("Updated packaged dashboard assets and source bundle; commit both outputs.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Dashboard build failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
