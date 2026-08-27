"""Real process-tree fixture for general local execution containment tests."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _identity(pid: int | None = None) -> dict[str, int | str]:
    effective_pid = os.getpid() if pid is None else pid
    stat_text = Path(f"/proc/{effective_pid}/stat").read_text(encoding="ascii")
    close = stat_text.rfind(")")
    fields = stat_text[close + 2 :].split()
    return {
        "pid": effective_pid,
        "process_group": int(fields[2]),
        "role": "",
        "start_tick": int(fields[19]),
        "proc_inode": Path(f"/proc/{effective_pid}").stat().st_ino,
    }


def _descendant_pids(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                stat_text = Path(entry.path, "stat").read_text(encoding="ascii")
            except (OSError, UnicodeError):
                continue
            close = stat_text.rfind(")")
            if close < 0:
                continue
            fields = stat_text[close + 2 :].split()
            if len(fields) <= 1:
                continue
            try:
                parents[int(entry.name)] = int(fields[1])
            except ValueError:
                continue
    descendants: set[int] = set()
    changed = True
    while changed:
        changed = False
        for pid, parent_pid in parents.items():
            if pid not in descendants and (parent_pid == root_pid or parent_pid in descendants):
                descendants.add(pid)
                changed = True
    return descendants


def _publish_playwright_processes(root: Path, *, browser_executable: Path) -> None:
    browser_executable = browser_executable.resolve()
    try:
        chromium_bundle = next(
            parent for parent in browser_executable.parents if parent.name.startswith("chromium-")
        )
    except StopIteration:
        raise RuntimeError("Playwright browser process authority is unavailable.") from None
    browser_revision = chromium_bundle.name.removeprefix("chromium-")
    if not browser_revision:
        raise RuntimeError("Playwright browser process authority is unavailable.")
    browsers_root = chromium_bundle.parent
    allowed_bundles = {
        chromium_bundle.name,
        f"chromium_headless_shell-{browser_revision}",
    }
    deadline = time.monotonic() + 5
    identities: list[dict[str, int | str]] = []
    while time.monotonic() < deadline:
        identities = []
        for pid in _descendant_pids(os.getpid()):
            try:
                executable = Path(f"/proc/{pid}/exe").resolve()
                relative = executable.relative_to(browsers_root)
                if not relative.parts or relative.parts[0] not in allowed_bundles:
                    continue
                identity = _identity(pid)
            except (OSError, UnicodeError, ValueError):
                continue
            identity["role"] = "playwright_chromium"
            identities.append(identity)
        if identities:
            break
        time.sleep(0.02)
    if not identities:
        raise RuntimeError("Playwright started without observable Chromium process authority.")
    path = root / "playwright-processes.json"
    staging = path.with_suffix(".staging")
    staging.write_text(json.dumps(identities, sort_keys=True), encoding="utf-8")
    os.replace(staging, path)


def _publish(root: Path, role: str) -> None:
    payload = _identity()
    payload["role"] = role
    path = root / f"{role}.json"
    staging = path.with_suffix(".staging")
    staging.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(staging, path)


def _ignore_term(_signum, _frame) -> None:
    return None


def _wait_forever() -> None:
    signal.signal(signal.SIGTERM, _ignore_term)
    while True:
        time.sleep(1)


def _grandchild(root: Path) -> None:
    _publish(root, "grandchild")
    _wait_forever()


def _background_server(root: Path) -> None:
    signal.signal(signal.SIGTERM, _ignore_term)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(root / "background.sock"))
    server.listen(4)
    _publish(root, "background_server")
    while True:
        server.settimeout(1)
        try:
            connection, _ = server.accept()
        except TimeoutError:
            continue
        connection.close()


def _child(
    root: Path,
    fixture: Path,
    *,
    complete: bool,
    with_playwright: bool,
) -> None:
    signal.signal(signal.SIGTERM, _ignore_term)
    _publish(root, "child")
    children = [
        subprocess.Popen(
            [sys.executable, str(fixture), "grandchild", str(root)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        ),
        subprocess.Popen(
            [sys.executable, str(fixture), "server", str(root)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        ),
    ]
    if with_playwright:
        playwright = None
        try:
            from playwright.sync_api import sync_playwright

            playwright = sync_playwright().start()
            chromium = playwright.chromium
            browser_executable = Path(chromium.executable_path)
            browser = chromium.launch(headless=True)
        except Exception:
            if playwright is not None:
                playwright.stop()
            (root / "playwright-unavailable").write_text("unavailable", encoding="ascii")
            (root / "tree-ready").write_text("ready", encoding="ascii")
            return
        _publish_playwright_processes(root, browser_executable=browser_executable)
        (root / "playwright-ready").write_text(str(browser.version), encoding="utf-8")
    (root / "tree-ready").write_text("ready", encoding="ascii")
    if complete:
        return
    while True:
        for child in children:
            if child.poll() is not None:
                raise RuntimeError("A fixture descendant exited before containment.")
        time.sleep(0.05)


async def _root(
    root: Path,
    fixture: Path,
    *,
    complete: bool,
    signal_process_group: bool,
    with_isolated_tool: bool,
    with_playwright: bool,
) -> None:
    from cayu import ExecCommand, LocalRunner

    _publish(root, "root")
    runner = LocalRunner(root)
    child_environment: dict[str, str] = {}
    if with_playwright and (browsers_path := os.environ.get("PLAYWRIGHT_BROWSERS_PATH")):
        child_environment["PLAYWRIGHT_BROWSERS_PATH"] = browsers_path
    runner_call = runner.exec(
        ExecCommand.process(
            sys.executable,
            str(fixture),
            "child",
            str(root),
            *(("--complete",) if complete else ()),
            *(("--with-playwright",) if with_playwright else ()),
        ),
        env=child_environment,
        output_limit_bytes=4096,
    )
    group_signal_task: asyncio.Task[None] | None = None
    if signal_process_group:
        group_signal_task = asyncio.create_task(
            _signal_root_process_group(root),
            name="local-execution-fixture-process-group-signal",
        )
    if with_isolated_tool:
        result, _ = await asyncio.gather(runner_call, _run_process_isolated_tool(root))
    else:
        result = await runner_call
    if result.exit_code != 0:
        raise RuntimeError(
            f"local execution child failed with exit code {result.exit_code}: {result.stderr}"
        )
    if group_signal_task is not None:
        await group_signal_task


async def _signal_root_process_group(root: Path) -> None:
    required = (
        root / "tree-ready",
        root / "child.json",
        root / "grandchild.json",
        root / "background_server.json",
    )
    while not all(path.is_file() for path in required):
        await asyncio.sleep(0.01)
    os.kill(0, signal.SIGTERM)


async def _run_process_isolated_tool(root: Path) -> None:
    from cayu.core import ExecutionProfileBehaviorIdentity
    from cayu.core.isolated_tools import (
        ProcessIsolatedTool,
        ProcessIsolatedToolContextProjection,
        ProcessIsolatedToolFactoryRef,
        ProcessIsolatedToolLimits,
    )
    from cayu.core.tools import (
        ToolContext,
        ToolEffect,
        ToolSpec,
        _bind_runtime_tool_invocation_authority,
    )
    from cayu.runtime._isolated_tool_process import execute_process_isolated_tool
    from cayu.vaults import SecretRedactor

    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "additionalProperties": False,
    }
    identity = ExecutionProfileBehaviorIdentity(
        name="cayu:testing:deterministic-isolated-tool",
        behavior_version="1",
        implementation_version="1",
    )
    tool = ProcessIsolatedTool(
        ToolSpec(
            name="isolated_fixture",
            description="Exercise the nested hard process boundary.",
            input_schema=schema,
            effect=ToolEffect.NONE,
            execution_profile_identity=identity,
        ),
        factory=ProcessIsolatedToolFactoryRef(
            module="cayu.testing_isolated_tools",
            qualname="build_deterministic_isolated_tool",
            identity=identity,
        ),
        limits=ProcessIsolatedToolLimits(
            deadline_seconds=30,
            term_grace_seconds=0.1,
            kill_grace_seconds=0.5,
        ),
        factory_config={
            "mode": "grandchild",
            "pid_path": str(root / "isolated-grandchild.pid"),
            "started_path": str(root / "isolated-started"),
            "seconds": 30,
        },
        context_projection=ProcessIsolatedToolContextProjection(fields=("session_id",)),
    )
    arguments = {"text": "nested"}
    context = ToolContext(
        session_id="local-execution-composition",
        idempotency_key="local-execution-isolated-tool",
    )
    operations: dict[str, dict[str, Any]] = {}

    async def load_operation(storage_key: str) -> dict[str, Any] | None:
        record = operations.get(storage_key)
        return None if record is None else dict(record)

    async def compare_and_set_operation(
        storage_key: str,
        expected: dict[str, Any] | None,
        desired: dict[str, Any],
        secondary: Mapping[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if operations.get(storage_key) != expected:
            return dict(operations[storage_key])
        operations[storage_key] = dict(desired)
        operations.update({key: dict(value) for key, value in secondary.items()})
        return dict(desired)

    _bind_runtime_tool_invocation_authority(
        context,
        parent_task_id="local-execution-task",
        parent_run_epoch=1,
        model_step_id="local-execution-model-step",
        model_attempt_id="local-execution-model-attempt",
        tool_round_id="local-execution-tool-round",
        tool_call_id="local-execution-tool-call",
        tool_name=tool.spec.name,
        idempotency_key="local-execution-isolated-tool",
        effective_arguments=arguments,
        execution_profile_fingerprint="e" * 64,
        environment_allocation_fingerprint=None,
        load_durable_operation=load_operation,
        compare_and_set_durable_operation=compare_and_set_operation,
        seal_durable_output=lambda value: dict(value),
        secret_publication_sealer=lambda: None,
    )
    await execute_process_isolated_tool(
        tool=tool,
        context=context,
        arguments=arguments,
        registered_schema=schema,
        redactor=SecretRedactor(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("root", "child", "grandchild", "server"))
    parser.add_argument("state_dir")
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--signal-process-group", action="store_true")
    parser.add_argument("--with-isolated-tool", action="store_true")
    parser.add_argument("--with-playwright", action="store_true")
    args = parser.parse_args()
    root = Path(args.state_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fixture = Path(__file__).resolve()
    if args.role == "root":
        asyncio.run(
            _root(
                root,
                fixture,
                complete=args.complete,
                signal_process_group=args.signal_process_group,
                with_isolated_tool=args.with_isolated_tool,
                with_playwright=args.with_playwright,
            )
        )
    elif args.role == "child":
        _child(
            root,
            fixture,
            complete=args.complete,
            with_playwright=args.with_playwright,
        )
    elif args.role == "grandchild":
        _grandchild(root)
    else:
        _background_server(root)


if __name__ == "__main__":
    main()
