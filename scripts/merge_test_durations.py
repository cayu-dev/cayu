#!/usr/bin/env python3
"""Merge CI shard timings into ``.test_durations``.

Each CI test job uploads a ``test-durations-<lane>`` artifact holding the
seconds it measured per node ID. This script merges those files into the
snapshot: measured IDs are updated or added, and, when given the full list of
collected IDs (the ``collected-test-ids`` artifact from a run with
``CAYU_TEST_STORES=all``), IDs that no longer exist are dropped.

The collection job also uploads ``test-lanes-<mode>``: the lane family of every
test it collected. For each mode found, ``.test_lanes`` is rewritten with the
timed tests whose family the planner would otherwise estimate wrongly.

    python3 scripts/merge_test_durations.py --run 123456789
    python3 scripts/merge_test_durations.py --input downloaded/ --collected ids.txt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))
import ci_shard_plan

_ROOT = Path(__file__).resolve().parents[1]
DURATION_ARTIFACT_PREFIX = "test-durations-"
COLLECTED_ARTIFACT = "collected-test-ids"
LANES_ARTIFACT_PREFIX = "test-lanes-"
LANES_NAME = "test-lanes.json"


def merge_durations(
    current: Mapping[str, float],
    measured: Iterable[Mapping[str, float]],
    collected: Iterable[str] | None = None,
) -> dict[str, float]:
    """Return the snapshot with measured timings applied.

    A node measured by more than one shard (a rerun) keeps its largest time.
    ``collected``, when given, is the complete set of current node IDs.
    """

    merged = {nodeid: float(seconds) for nodeid, seconds in current.items()}
    latest: dict[str, float] = {}
    for timings in measured:
        for nodeid, seconds in timings.items():
            latest[nodeid] = max(float(seconds), latest.get(nodeid, 0.0))
    merged.update(latest)
    if collected is not None:
        keep = set(collected)
        merged = {nodeid: seconds for nodeid, seconds in merged.items() if nodeid in keep}
    return dict(sorted(merged.items()))


def merge_lanes(
    root: Path,
    current: Mapping[str, Mapping[str, str | None]],
    durations: Iterable[str],
    observed: Iterable[tuple[str, Mapping[str, str]]],
) -> dict[str, dict[str, str | None]]:
    """Replace each observed mode's corrections in the ``.test_lanes`` data."""

    members = ci_shard_plan.family_members(root)
    nodeids = list(durations)
    merged = {mode: dict(corrections) for mode, corrections in current.items()}
    for mode, lanes in observed:
        corrections = ci_shard_plan.lane_corrections(nodeids, lanes, mode, members)
        merged[mode] = dict(sorted(corrections.items()))
    return dict(sorted(merged.items()))


def _read_lanes(path: Path) -> tuple[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mode = data.get("mode") if isinstance(data, dict) else None
    lanes = data.get("lanes") if isinstance(data, dict) else None
    if mode not in ci_shard_plan.STORE_MODES or not isinstance(lanes, dict):
        raise ValueError(f"{path} must hold a store mode and node lane families")
    return mode, lanes


def _read_json(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, (int, float)) for key, value in data.items()
    ):
        raise ValueError(f"{path} must map node IDs to seconds")
    return data


def _read_ids(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if "::" in line]


def _download(run: str, directory: Path) -> None:
    subprocess.run(
        (
            "gh",
            "run",
            "download",
            run,
            "--dir",
            str(directory),
            "--pattern",
            f"{DURATION_ARTIFACT_PREFIX}*",
            "--pattern",
            COLLECTED_ARTIFACT,
            "--pattern",
            f"{LANES_ARTIFACT_PREFIX}*",
        ),
        check=True,
    )


def _inputs(directory: Path) -> tuple[list[Path], list[Path], Path | None]:
    timings = sorted(
        path
        for path in directory.rglob("*.json")
        if path.parent.name != COLLECTED_ARTIFACT and path.name != LANES_NAME
    )
    lanes = sorted(directory.rglob(LANES_NAME))
    collected = sorted(directory.rglob("collected-test-ids.txt"))
    return timings, lanes, (collected[0] if collected else None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="GitHub Actions run ID to download artifacts from")
    source.add_argument("--input", type=Path, help="directory of already downloaded artifacts")
    parser.add_argument(
        "--collected",
        type=Path,
        help="file of all collected node IDs; overrides one found among the artifacts",
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="keep IDs absent from the collected list (when the run collected a subset)",
    )
    parser.add_argument("--durations", type=Path, default=_ROOT / ".test_durations")
    args = parser.parse_args(argv)
    root = args.durations.resolve().parent
    lanes_path = root / ci_shard_plan.LANES_FILE

    with tempfile.TemporaryDirectory(prefix="cayu-test-durations-") as temporary:
        directory = args.input
        if args.run:
            directory = Path(temporary)
            _download(args.run, directory)
        timings, lane_files, found_collected = _inputs(directory)
        if not timings and not lane_files:
            print(f"no shard timing or lane files under {directory}", file=sys.stderr)
            return 1
        collected_path = args.collected or found_collected
        collected = (
            None if args.keep_missing or collected_path is None else _read_ids(collected_path)
        )
        current = _read_json(args.durations) if args.durations.exists() else {}
        merged = merge_durations(current, (_read_json(path) for path in timings), collected)
        observed = [_read_lanes(path) for path in lane_files]

    if timings:
        args.durations.write_text(
            json.dumps(merged, indent=4, sort_keys=True) + "\n", encoding="utf-8"
        )
        added = len(merged.keys() - current.keys())
        dropped = len(current.keys() - merged.keys())
        print(
            f"merged {len(timings)} shard files: {len(merged)} timings "
            f"({added} added, {dropped} dropped)"
            + ("" if collected is not None else "; no collected list, nothing dropped")
        )
    if observed:
        recorded = json.loads(lanes_path.read_text(encoding="utf-8")) if lanes_path.exists() else {}
        lanes = merge_lanes(root, recorded, merged, observed)
        lanes_path.write_text(json.dumps(lanes, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        counts = ", ".join(f"{mode} {len(lanes[mode])}" for mode, _ in observed)
        print(f"refreshed {ci_shard_plan.LANES_FILE} corrections: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
