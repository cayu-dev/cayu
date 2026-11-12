"""Record, or check, how `cayu new --preset b2b` matches the b2b reference project.

The reference project (cayu-tech/cayu-b2b-reference, private) is a `cayu new --preset b2b`
project changed by hand until it is what the generator should write. CI can't read it, so
`tests/fixtures/b2b_reference/` keeps what the generator writes for the files that work
covers, and `manifest.json` says, for each, whether it is byte for byte the reference's file
(with that file's SHA-256, so anyone with the reference can verify the claim) or why it
differs. `tests/cli/test_scaffold_b2b_reference.py` checks the generator against them.

    python scripts/b2b_reference_fixtures.py --reference PATH          # check
    python scripts/b2b_reference_fixtures.py --reference PATH --write  # record

`--write` rewrites the fixture files from the generator and each entry's hashes from the
manifest's commit in the reference repository at PATH (read from git, whatever is checked
out). A file that differs from
the reference needs a `deviation` written by hand first: the script refuses to record a
difference nobody explained, or a reference file the generator leaves out without a reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "b2b_reference"
MANIFEST = FIXTURES / "manifest.json"
# Fixture files carry this suffix, so pytest, ruff and type checkers leave them alone.
SUFFIX = ".fixture"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def in_scope(path: str, manifest: dict) -> bool:
    if path.startswith(tuple(manifest["excluded_prefixes"])):
        return False
    return path in manifest["paths"] or path.startswith(tuple(manifest["prefixes"]))


def generated(manifest: dict) -> dict[str, str]:
    """What the generator writes for the reference project, in the manifest's scope."""

    sys.path.insert(0, str(ROOT / "src"))
    from cayu.cli.scaffold import project_files

    files = project_files(manifest["project"], preset="b2b")
    return {path: text for path, text in sorted(files.items()) if in_scope(path, manifest)}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture_path(path: str) -> Path:
    return FIXTURES / "files" / f"{path}{SUFFIX}"


def _git(reference: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(reference), *args], check=True, capture_output=True
    ).stdout


def _reference_files(reference: Path, manifest: dict) -> dict[str, bytes]:
    """The reference's files at the manifest's commit, read from git, not from the working
    tree, so uncommitted changes in the checkout can't be recorded as the reference's."""

    commit = manifest["reference"]["commit"]
    try:
        _git(reference, "cat-file", "-e", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError:
        raise SystemExit(f"{reference} has no commit {commit}; fetch it first.") from None
    listed = _git(reference, "ls-tree", "-r", "-z", "--name-only", commit).decode().split("\0")
    return {
        path: _git(reference, "show", f"{commit}:{path}")
        for path in listed
        if path and in_scope(path, manifest)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="the reference checkout")
    parser.add_argument("--write", action="store_true", help="record the fixtures and hashes")
    args = parser.parse_args(argv)
    manifest = load_manifest()
    reference = _reference_files(args.reference, manifest)
    files = generated(manifest)
    entries: dict[str, dict] = manifest["files"]
    recorded: dict[str, dict] = {}
    problems: list[str] = []

    for path, text in files.items():
        entry = entries.get(path, {})
        theirs = reference.get(path)
        record: dict = {"reference_sha256": None if theirs is None else sha256(theirs)}
        if theirs == text.encode("utf-8"):
            record["same"] = True
        elif entry.get("deviation"):
            record["deviation"] = entry["deviation"]
        else:
            problems.append(f"{path}: differs from the reference; say why in its `deviation`")
            continue
        recorded[path] = record
        if not args.write and entry != record:
            problems.append(f"{path}: the manifest says {entry}, the reference {record}")
    for path in sorted(set(reference) - set(files)):
        if not manifest["not_generated"].get(path):
            problems.append(f"{path}: in the reference, not generated; say why in `not_generated`")
    for path in sorted(set(manifest["not_generated"]) - set(reference)):
        problems.append(f"{path}: listed in `not_generated`, but not in the reference")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    if args.write:
        manifest["files"] = recorded
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for stale in (FIXTURES / "files").rglob(f"*{SUFFIX}"):
            stale.unlink()
        for path, text in files.items():
            target = fixture_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
    print(f"{len(files)} generated files checked against the reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
