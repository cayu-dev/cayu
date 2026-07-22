from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pexpect
from nightly_verification import _LIVE_CREDENTIAL_ENV, _STRUCTURED_EVIDENCE_PREFIX

_PROMPT_PATTERN = r"In \[\d+\]: "


def main() -> None:
    executable = shutil.which("cayu")
    if executable is None:
        raise RuntimeError("cayu executable is unavailable")

    with tempfile.TemporaryDirectory(prefix="cayu-console-pty-") as temporary_directory:
        temporary_root = Path(temporary_directory).resolve()
        project = temporary_root / "project"
        ipython_directory = temporary_root / "ipython"

        environment = dict(os.environ)
        for name in _LIVE_CREDENTIAL_ENV:
            environment.pop(name, None)
        environment.update(
            {
                "IPYTHONDIR": str(ipython_directory),
                "NO_COLOR": "1",
                "OPENAI_API_KEY": "cayu-nightly-no-network",
                "PYTHONNOUSERSITE": "1",
                "TERM": "dumb",
            }
        )

        scaffold = subprocess.run(
            [executable, "new", "project", "--dir", str(temporary_root)],
            check=False,
            cwd=temporary_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if scaffold.returncode != 0:
            raise RuntimeError(
                f"cayu new failed: stdout={scaffold.stdout!r} stderr={scaffold.stderr!r}"
            )

        nested = project / "agents" / "reviewer"
        nested.mkdir(parents=True)
        _write_ipython_config(ipython_directory)

        transcript = io.StringIO()
        child = pexpect.spawn(
            executable,
            ["console"],
            cwd=str(nested),
            env=environment,
            encoding="utf-8",
            echo=False,
            timeout=20,
        )
        child.logfile_read = transcript
        try:
            child.expect_exact(f"Project: {project}")
            child.expect_exact("Factory: app:build_app")
            child.expect_exact("Agents: assistant")
            child.expect_exact("Providers: openai")
            child.expect_exact("Environments: none")
            child.expect_exact("Session store: SQLiteSessionStore")
            child.expect_exact("Task store: SQLiteTaskStore")
            _expect_prompt(child)

            _run_cell(child, "import asyncio")
            _run_cell(
                child,
                "async def _loop_id(): return id(asyncio.get_running_loop())",
            )
            _run_cell(child, "_first_loop = await _loop_id()")
            _run_cell(child, "_second_loop = await _loop_id()")
            _run_cell(child, "_sessions = await sessions.list_sessions()")
            _run_cell(child, "_tasks = await tasks.list_tasks()")
            _run_cell(
                child,
                "print('CAYU_CONSOLE_NAMESPACE=' + repr((isinstance(app, "
                "cayu.CayuApp), sessions is app.session_store, tasks is app.task_store, "
                "knowledge is None, app.list_agents())))",
                expected=("CAYU_CONSOLE_NAMESPACE=(True, True, True, True, ('assistant',))"),
            )
            _run_cell(
                child,
                "print('CAYU_CONSOLE_ASYNC=' + "
                "repr((_first_loop == _second_loop, len(_sessions.sessions), len(_tasks))))",
                expected="CAYU_CONSOLE_ASYNC=(True, 0, 0)",
            )

            child.sendcontrol("d")
            child.expect(pexpect.EOF)
            child.close()
        except Exception as exc:
            child.close(force=True)
            tail = transcript.getvalue()[-4000:]
            raise RuntimeError(f"console PTY verification failed\n{tail}") from exc

        if child.exitstatus != 0:
            raise RuntimeError(
                "console exited unsuccessfully: "
                f"exitstatus={child.exitstatus} signalstatus={child.signalstatus}"
            )

        if not (project / "data" / "cayu.db").is_file():
            raise RuntimeError("scaffold factory did not create its project-root Cayu store")
        if (nested / "data").exists():
            raise RuntimeError("scaffold factory resolved relative state from the nested directory")

    print(
        _STRUCTURED_EVIDENCE_PREFIX
        + json.dumps(
            {
                "async_store_operation": True,
                "clean_eof_exit": True,
                "ipython": True,
                "loop_reused_across_cells": True,
                "namespace_aliases": True,
                "nested_project_discovery": True,
                "project_relative_state": True,
                "scaffold_generated": True,
            },
            sort_keys=True,
        )
    )


def _expect_prompt(child: pexpect.spawn) -> None:
    child.expect(_PROMPT_PATTERN)


def _run_cell(child: pexpect.spawn, source: str, *, expected: str | None = None) -> None:
    child.sendline(source)
    if expected is not None:
        child.expect_exact(expected)
    _expect_prompt(child)


def _write_ipython_config(ipython_directory: Path) -> None:
    profile = ipython_directory / "profile_default"
    profile.mkdir(parents=True)
    (profile / "ipython_config.py").write_text(
        """c = get_config()
c.TerminalInteractiveShell.colors = "NoColor"
c.TerminalInteractiveShell.confirm_exit = False
c.TerminalInteractiveShell.simple_prompt = True
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
