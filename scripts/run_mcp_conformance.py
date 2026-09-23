"""Run the explicitly covered official client scenarios, never a full-suite claim.

Requires a POSIX host with Node and a built, pinned upstream checkout. Docker is
the supported reproducible entry point. Raw evidence is retained on failure.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_conformance_build import UPSTREAM_REVISION, verify_build

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Scenario:
    version: str
    name: str
    required_checks: frozenset[str]


SCENARIOS = (
    Scenario("2025-06-18", "initialize", frozenset({"mcp-client-initialization"})),
    *(
        Scenario(version, "tools_call", frozenset({"tool-add-numbers", "wire-schema-valid"}))
        for version in ("2025-06-18", "2026-07-28")
    ),
    Scenario(
        "2026-07-28", "json-schema-ref-no-deref", frozenset({"sep-2106-no-network-ref-deref"})
    ),
    Scenario(
        "2026-07-28",
        "json-schema-2020-12-preservation",
        frozenset(
            {
                "json-schema-2020-12-client-tool-found",
                "json-schema-2020-12-client-echo-completed",
                "json-schema-2020-12-client-$schema-preserved",
                "json-schema-2020-12-client-$defs-preserved",
                "json-schema-2020-12-client-additionalProperties-preserved",
                "sep-2106-client-composition-keywords-preserved",
                "sep-2106-client-conditional-keywords-preserved",
                "sep-2106-client-anchor-keyword-preserved",
                "wire-schema-valid",
            }
        ),
    ),
)
BLOCKED_SCENARIOS = ("http-standard-headers", "http-custom-headers", "http-invalid-tool-headers")
SDK_TESTS = (
    "tests/core/test_mcp_stdio_modern.py::test_modern_stdio_interoperates_with_official_sdk",
    "tests/core/test_mcp_http_subscriptions.py::test_modern_http_subscription_official_sdk_interoperability",
    "tests/core/test_mcp_stdio_subscriptions.py::test_modern_stdio_subscription_official_sdk_interoperability[False]",
    "tests/core/test_mcp_stdio_subscriptions.py::test_modern_stdio_subscription_official_sdk_interoperability[True]",
)


def run_bounded(
    command: list[str], directory: Path, *, timeout: float, token: str | None = None
) -> int:
    """Keep logs off memory and reap the runner's shell/client process group."""
    with (
        (directory / "runner.stdout.txt").open("w") as stdout,
        (directory / "runner.stderr.txt").open("w") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env={**os.environ, "CAYU_MCP_CONFORMANCE_TOKEN": token or ""},
        )
        try:
            return process.wait(timeout=timeout)
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def validate_checks(checks: Any, required: frozenset[str]) -> None:
    if not isinstance(checks, list) or not checks:
        raise ValueError("Official runner produced no checks.")
    successful = set()
    for check in checks:
        if (
            not isinstance(check, dict)
            or not isinstance(check.get("id"), str)
            or not isinstance(check.get("name"), str)
            or check.get("status") not in {"SUCCESS", "INFO"}
        ):
            raise ValueError(
                "Official evidence contains a failed, skipped, warning, or malformed check."
            )
        if check["status"] == "SUCCESS":
            successful.add(check["id"])
    missing = required - successful
    if missing:
        raise ValueError(f"Required successful checks missing: {sorted(missing)}")


def report_sdk_failure(directory: Path) -> None:
    """Expose bounded local test diagnostics even when artifact uploads are unavailable."""
    for name in ("runner.stdout.txt", "runner.stderr.txt"):
        path = directory / name
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 65536))
            tail = stream.read(65536).decode("utf-8", errors="replace")
        print(f"SDK diagnostic {name} (last 65536 bytes):\n{tail}", flush=True)


def validate_sdk_report(path: Path) -> None:
    cases = ET.parse(path).getroot().findall(".//testcase")
    expected = {node.split("::")[1] for node in SDK_TESTS}
    if len(cases) != len(expected) or {case.get("name") for case in cases} != expected:
        raise ValueError("SDK evidence must contain exactly the selected interoperability tests.")
    if any(case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")):
        raise ValueError("SDK interoperability tests failed or skipped.")


def validate_completion(path: Path, scenario: Scenario, token: str) -> None:
    expected = {"cayu_completion": token, "scenario": scenario.name, "version": scenario.version}
    receipts = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and "cayu_completion" in value:
            receipts.append(value)
    if receipts != [expected]:
        raise ValueError("Missing or mismatched adapter completion receipt after session cleanup.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New evidence directory (must not exist)."
    )
    parser.add_argument(
        "--include-blocked",
        action="store_true",
        help="Also reproduce upstream-blocked header fixtures; they remain outside the passing subset.",
    )
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Use the Docker entry point on non-POSIX hosts.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    upstream = args.upstream.resolve()
    records = []
    report: dict[str, Any] = {
        "upstream_revision": UPSTREAM_REVISION,
        "full_conformance": False,
        "coverage_document": "docs/mcp-conformance.md",
        "upstream_blocked": list(BLOCKED_SCENARIOS),
        "results": records,
    }
    try:
        report["referee_build"] = verify_build(upstream)
        scenarios = list(SCENARIOS)
        if args.include_blocked:
            scenarios.extend(
                Scenario("2026-07-28", name, frozenset()) for name in BLOCKED_SCENARIOS
            )
        for scenario in scenarios:
            directory = output / f"{scenario.version}-{scenario.name}"
            directory.mkdir()
            record: dict[str, Any] = {
                "kind": "official",
                "version": scenario.version,
                "scenario": scenario.name,
                "status": "failed",
            }
            records.append(record)
            try:
                token = secrets.token_hex(32)
                code = run_bounded(
                    [
                        "node",
                        str(upstream / "dist/index.js"),
                        "client",
                        "--command",
                        shlex.join(
                            [sys.executable, str(ROOT / "scripts/mcp_conformance_client.py")]
                        ),
                        "--scenario",
                        scenario.name,
                        "--spec-version",
                        scenario.version,
                        "--timeout",
                        "30000",
                        "--output-dir",
                        str(directory),
                    ],
                    directory,
                    timeout=45,
                    token=token,
                )
                record["exit_code"] = code
                artifacts = list(directory.glob("*/checks.json"))
                if len(artifacts) != 1:
                    raise ValueError("Expected exactly one fresh official checks artifact.")
                checks = json.loads(artifacts[0].read_text())
                record["checks"] = checks
                if code != 0:
                    raise ValueError(f"Official runner exited {code}; see retained logs.")
                validate_checks(checks, scenario.required_checks)
                validate_completion(artifacts[0].with_name("stdout.txt"), scenario, token)
                if scenario.name in BLOCKED_SCENARIOS:
                    raise ValueError(
                        "Previously blocked scenario changed; review the coverage matrix before admitting it."
                    )
                record["status"] = "passed"
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                record["error"] = str(exc)
            print(f"{scenario.version} {scenario.name}: {record['status']}", flush=True)
        directory = output / "sdk-interoperability"
        directory.mkdir()
        record = {"kind": "sdk", "status": "failed", "tests": list(SDK_TESTS)}
        records.append(record)
        try:
            code = run_bounded(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    *SDK_TESTS,
                    "-q",
                    "-o",
                    "addopts=",
                    f"--junitxml={directory / 'junit.xml'}",
                ],
                directory,
                timeout=180,
            )
            record["exit_code"] = code
            if code != 0:
                raise ValueError(f"SDK tests exited {code}; see retained logs.")
            validate_sdk_report(directory / "junit.xml")
            record["status"] = "passed"
        except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as exc:
            record["error"] = str(exc)
        print(f"SDK interoperability: {record['status']}", flush=True)
        if record["status"] != "passed":
            print(f"SDK failure: {record.get('error', 'unknown')}", flush=True)
            report_sdk_failure(directory)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["error"] = str(exc)
    finally:
        report["covered_subset_passed"] = (
            "error" not in report
            and len(records) == len(SCENARIOS) + 1
            and all(record["status"] == "passed" for record in records)
        )
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["covered_subset_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
