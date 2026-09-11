#!/usr/bin/env python3
"""Run the registered qualification suite against an already installed Cayu wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_RECIPE = "cayu.qualification-fixtures.v2"
sys.path.insert(0, str(ROOT))
from tests.qualification.postgres_cleanup import DATABASE_ENV, postgres_operation  # noqa: E402
from tests.qualification.process_cleanup import (  # noqa: E402
    GROUP_REGISTRY_ENV,
    drain_process_groups,
    process_group_exists,
    registered_process_groups,
)
from tests.qualification.registry import (  # noqa: E402
    DOCKER_SCENARIOS,
    POSTGRES_SCENARIOS,
    SCENARIOS,
)

PROBE = """
import importlib.metadata, json
from pathlib import Path
from uuid import uuid4
import cayu
from cayu.build_provenance import current_runtime_build_provenance
package = Path(cayu.__file__).resolve()
dist = importlib.metadata.distribution('cayu')
expected = Path(dist.locate_file('cayu/__init__.py')).resolve()
assert package == expected and expected.is_file()
p = current_runtime_build_provenance()
assert p.origin.value != 'development_source_tree'
print(json.dumps({'package': str(package), 'build': {
    'availability': p.availability.value, 'origin': p.origin.value, 'fingerprint': p.fingerprint}}))
"""


def fixture_fingerprint(stage: Path) -> str:
    """Bind all staged inputs, including non-Python assets consumed by emitters."""
    files = [file for file in (stage / "tests").rglob("*") if file.is_file()]
    files.extend((stage / "pyproject.toml", stage / "scripts" / Path(__file__).name))
    digest = hashlib.sha256(FIXTURES_RECIPE.encode() + b"\0")
    for file in sorted(files, key=lambda file: file.relative_to(stage).as_posix()):
        relative = file.relative_to(stage).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def run_bounded(command, *, cwd, env, timeout, cleanup=None):
    cleanup = {} if cleanup is None else cleanup
    cleanup["postgres_databases_retained"] = 0
    child_env = dict(env)
    child_env.pop(DATABASE_ENV, None)
    database = (
        "cayu_qualification_" + uuid4().hex
        if env.get("CAYU_QUALIFICATION_POSTGRES") == "1"
        else None
    )
    code = 2
    try:
        if database is not None:
            # Retain ownership before dispatching CREATE, including failed acknowledgements.
            cleanup["postgres_databases_retained"] = 1
            created = postgres_operation(command[0], database, env, "create")
            child_env[DATABASE_ENV] = database
        else:
            created = True
        if created:
            code = _run_bounded_process(command, cwd=cwd, env=child_env, timeout=timeout)
    finally:
        if database is not None:
            cleanup["postgres_databases_retained"] = int(
                not postgres_operation(command[0], database, env, "drop")
            )
    return code or cleanup["postgres_databases_retained"]


def _run_bounded_process(command, *, cwd, env, timeout):
    # The parent owns this journal, independently of pytest's shutdown hooks.
    with tempfile.TemporaryDirectory(prefix="cayu-qualification-groups-") as directory:
        registry = Path(directory) / "groups"
        registry.touch(mode=0o600)
        child_env = dict(env)
        child_env[GROUP_REGISTRY_ENV] = str(registry)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=child_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        code = 1
        try:
            code = process.wait(timeout=timeout)
            if process_group_exists(process.pid):
                code = code or 1
        except subprocess.TimeoutExpired:
            code = 124
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=10)
            # Stop pytest before reading the journal so it cannot launch more work.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            groups = {process.pid}
            try:
                groups.update(registered_process_groups(registry))
                if any(process_group_exists(group_id) for group_id in groups):
                    code = code or 1
            finally:
                groups_remaining = drain_process_groups(groups)
        return code or int(groups_remaining != 0)


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(report, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python", default=sys.executable, help="Python with installed cayu and dev dependencies"
    )
    parser.add_argument("--profile", choices=("default", "stress"), default="default")
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Require disposable PostgreSQL via CAYU_TEST_POSTGRES_DSN",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Require local Docker and an existing CAYU_DOCKER_CODING_IMAGE",
    )
    parser.add_argument("--repeat", type=int, default=2, choices=range(1, 6))
    parser.add_argument("--report", type=Path, default=Path("runtime-qualification.json"))
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[s.name for s in SCENARIOS + POSTGRES_SCENARIOS + DOCKER_SCENARIOS],
        help="Focused diagnosis; does not qualify the full profile",
    )
    args = parser.parse_args()
    if not args.postgres and any(
        name in {s.name for s in POSTGRES_SCENARIOS} for name in args.scenario or ()
    ):
        parser.error("PostgreSQL scenarios require --postgres")
    if not args.docker and any(
        name in {s.name for s in DOCKER_SCENARIOS} for name in args.scenario or ()
    ):
        parser.error("Docker scenarios require --docker")
    report = {
        "schema_version": 1,
        "suite": "cayu-runtime-qualification-v1",
        "profile": args.profile,
        "backend": "sqlite+postgres" if args.postgres else "sqlite",
        "repeat": args.repeat,
        "scope": "focused" if args.scenario else "full",
        "build": {"availability": "unavailable", "origin": "unavailable", "fingerprint": None},
        "status": "prerequisite-unavailable",
        "scenarios": [],
        "prerequisites": [
            "POSIX process groups and SIGKILL",
            "installed Cayu wheel and dev dependencies",
            "disposable PostgreSQL DSN" if args.postgres else "SQLite writable temporary directory",
        ],
    }
    if args.docker:
        report["prerequisites"].append("local Docker and an existing CAYU_DOCKER_CODING_IMAGE")
    if args.profile == "stress":
        report["prerequisites"] += [
            "100 concurrent sessions and environment bindings; 200 idle workers",
            "at least 512 open file descriptors; recommended 4 CPUs and 4 GiB available RAM",
        ]
    try:
        if os.name != "posix":
            return 2
        if args.postgres and not os.environ.get("CAYU_TEST_POSTGRES_DSN"):
            return 2
        if args.docker and not os.environ.get("CAYU_DOCKER_CODING_IMAGE"):
            return 2
        if args.profile == "stress":
            import resource

            if resource.getrlimit(resource.RLIMIT_NOFILE)[0] < 512:
                return 2
        with tempfile.TemporaryDirectory(prefix="cayu-qualification-") as directory:
            stage = Path(directory)
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.pop("PYTEST_ADDOPTS", None)
            env.pop("CAYU_QUALIFICATION_FAULT", None)
            env.pop("CAYU_QUALIFICATION_POSTGRES", None)
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            env["PYTHONNOUSERSITE"] = "1"
            probe = subprocess.run(
                [args.python, "-c", PROBE],
                cwd=stage,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            if probe.returncode:
                return 2
            identity = json.loads(probe.stdout)
            report["build"] = identity["build"]
            shutil.copytree(
                ROOT / "tests",
                stage / "tests",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            shutil.copy(ROOT / "pyproject.toml", stage / "pyproject.toml")
            (stage / "scripts").mkdir()
            shutil.copy(Path(__file__), stage / "scripts" / Path(__file__).name)
            # This staged tree intentionally has no src directory. Fresh-process helpers
            # may add that nonexistent path; their imports still resolve to the wheel.
            env["PYTHONPATH"] = str(stage)
            env["CAYU_QUALIFICATION_PACKAGE"] = identity["package"]
            env["CAYU_QUALIFICATION_PROFILE"] = args.profile
            if args.docker:
                env["CAYU_REQUIRE_DOCKER_CODING"] = "1"
            if args.postgres:
                env["CAYU_REQUIRE_POSTGRES"] = "1"
                env["CAYU_QUALIFICATION_POSTGRES"] = "1"
            else:
                env.pop("CAYU_TEST_POSTGRES_DSN", None)
                env.pop("CAYU_REQUIRE_POSTGRES", None)
            manifest = json.dumps(
                [s.__dict__ for s in SCENARIOS + POSTGRES_SCENARIOS + DOCKER_SCENARIOS],
                sort_keys=True,
            ).encode()
            report["registry_sha256"] = hashlib.sha256(manifest).hexdigest()
            report["fixtures_recipe"] = FIXTURES_RECIPE
            report["fixtures_sha256"] = fixture_fingerprint(stage)
            report["status"] = "running"
            write_report(args.report, report)
            scenarios = (
                SCENARIOS
                + (POSTGRES_SCENARIOS if args.postgres else ())
                + (DOCKER_SCENARIOS if args.docker else ())
            )
            for repetition in range(args.repeat):
                for scenario in scenarios:
                    if args.scenario and scenario.name not in args.scenario:
                        continue
                    result = stage / "result.json"
                    result.unlink(missing_ok=True)
                    env["CAYU_QUALIFICATION_RESULT"] = str(result)
                    marks = []
                    if args.profile == "default":
                        marks.append("not stress")
                    if not args.postgres:
                        marks += ["not postgres", "not postgres_recovery"]
                    command = [
                        args.python,
                        "-m",
                        "pytest",
                        "-p",
                        "tests.qualification.report_plugin",
                        "-p",
                        "anyio.pytest_plugin",
                        "-q",
                        "--tb=no",
                        *scenario.selectors,
                    ]
                    if marks:
                        command += ["-m", " and ".join(marks)]
                    start = time.monotonic()
                    cleanup = {}
                    code = run_bounded(
                        command,
                        cwd=stage,
                        env=env,
                        timeout=600 if args.profile == "stress" else 300,
                        cleanup=cleanup,
                    )
                    evidence = json.loads(result.read_text()) if result.exists() else {}
                    evidence.setdefault("resources", {}).update(cleanup)
                    cases = evidence.get("cases", {})
                    skipped = sum("skipped" in c["phases"].values() for c in cases.values())
                    passed = (
                        code == 0
                        and bool(cases)
                        and skipped == 0
                        and all(
                            c["phases"].get("call") == "passed"
                            and all(p == "passed" for p in c["phases"].values())
                            for c in cases.values()
                        )
                        and evidence.get("resources", {}).get("subprocesses_retained") == 0
                        and evidence.get("resources", {}).get("postgres_databases_retained") == 0
                        and evidence.get("resources", {}).get("subprocess_groups_remaining") == 0
                        and evidence.get("resources", {}).get("subprocesses_remaining") == 0
                        and evidence.get("build") == report["build"]
                    )
                    entry = {
                        "name": scenario.name,
                        "repetition": repetition + 1,
                        "invariant": scenario.invariant,
                        "first_durable_boundary": scenario.boundary,
                        "status": "passed" if passed else "failed",
                        "exit_code": code,
                        "elapsed_seconds": round(time.monotonic() - start, 3),
                        "recovery_availability": "scenario-specific; see invariant",
                        "evidence": evidence,
                    }
                    report["scenarios"].append(entry)
                    write_report(args.report, report)
                    print(
                        f"{scenario.name} [{repetition + 1}/{args.repeat}]: {entry['status']}",
                        flush=True,
                    )
            report["status"] = (
                "passed" if all(s["status"] == "passed" for s in report["scenarios"]) else "failed"
            )
            return 0 if report["status"] == "passed" else 1
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        return 130
    except (OSError, ValueError, subprocess.SubprocessError):
        report["status"] = "infrastructure-error"
        return 2
    finally:
        write_report(args.report, report)


if __name__ == "__main__":
    raise SystemExit(main())
