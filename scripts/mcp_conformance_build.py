"""Build and verify the locked referee; never attest an arbitrary existing bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

UPSTREAM_REVISION = "7169291ec0b68eb370fddcd9947313ab0d5e4156"
RECEIPT = "dist/cayu-build-receipt.json"


def verify_source(upstream: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("Official conformance checkout does not match the pinned revision.")
    subprocess.run(
        ["git", "-C", str(upstream), "diff", "--exit-code", "HEAD"], check=True, timeout=10
    )
    untracked = subprocess.check_output(
        ["git", "-C", str(upstream), "ls-files", "--others", "--exclude-standard"],
        text=True,
        timeout=10,
    )
    if untracked.strip():
        raise ValueError("Remove untracked source files before building the official referee.")


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_identity(upstream: Path) -> dict:
    files = []
    for tree in ("dist", "node_modules"):
        directory = upstream / tree
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"Missing regular build directory: {tree}")
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(upstream).as_posix()
            if relative == RECEIPT:
                continue
            if path.is_symlink():
                path.resolve(strict=True).relative_to(upstream.resolve())
                files.append((relative, "symlink", str(path.readlink())))
            elif path.is_file():
                files.append((relative, "file", file_digest(path)))
            elif not path.is_dir():
                raise ValueError(f"Unsupported build entry: {relative}")
    if not (upstream / "dist/index.js").is_file():
        raise ValueError("Missing official runner bundle.")
    return {
        "revision": UPSTREAM_REVISION,
        "lock_sha256": file_digest(upstream / "package-lock.json"),
        "artifacts_sha256": hashlib.sha256(json.dumps(files).encode()).hexdigest(),
    }


def prepare(upstream: Path) -> None:
    verify_source(upstream)
    # Invalidate prior evidence before any install/build can fail. There is no
    # standalone 'stamp' operation for a preexisting, possibly stale bundle.
    (upstream / RECEIPT).unlink(missing_ok=True)
    for command in (
        ["npm", "ci", "--ignore-scripts"],
        ["npm", "run", "build"],
        ["npm", "prune", "--omit=dev", "--ignore-scripts"],
    ):
        subprocess.run(command, cwd=upstream, check=True, timeout=300)
    verify_source(upstream)
    (upstream / RECEIPT).write_text(json.dumps(build_identity(upstream), sort_keys=True) + "\n")


def verify_build(upstream: Path) -> dict:
    verify_source(upstream)
    receipt = json.loads((upstream / RECEIPT).read_text())
    if receipt != build_identity(upstream):
        raise ValueError("Referee build differs from its locked build receipt; rebuild it.")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    prepare(parser.parse_args().upstream.resolve())
