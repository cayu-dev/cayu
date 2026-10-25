"""local_agent_launch: the vendor CLI as an unsandboxed local process."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from cayu import CayuApp, Message, RunRequest
from cayu.deadlines import ExecutionDeadline
from cayu.external_agents import (
    ClaudeAgentAdapter,
    CodexAgentAdapter,
    ExternalCleanupIncomplete,
    local_agent_launch,
    run_external_agent,
)
from cayu.external_agents.environment import scrub_external_environment
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import TaskStatus

pytestmark = pytest.mark.skipif(os.name != "posix", reason="local_agent_launch is POSIX only")

# Package CI installs the built wheel's claude-agent-sdk extra, which resolves to
# the newest SDK the extra allows, and sets this so a missing SDK fails.
_REQUIRE_SDK_ENV = "CAYU_REQUIRE_CLAUDE_AGENT_SDK"

# Records what it was started with, leaves a descendant in its process group,
# echoes stdin and exits 3.
_RECORDING_CLI = """\
import json, os, subprocess, sys
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(300)"],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "env": dict(os.environ),
                  "descendant": child.pid}), flush=True)
for line in sys.stdin:
    sys.stdout.write(line.upper())
    sys.stdout.flush()
sys.exit(3)
"""


def _cli(tmp_path: Path, source: str, name: str = "vendor-cli") -> Path:
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(0o700)
    return path


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed orphan can stay a zombie for a moment until init reaps it.
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat:
            return stat.read().split(") ", 1)[1][:1] != "Z"
    except OSError:
        return True


async def _wait_gone(pid: int) -> bool:
    for _ in range(250):
        if not _alive(pid):
            return True
        await asyncio.sleep(0.02)
    return False


def test_local_launch_runs_the_cli_with_only_safe_and_explicit_environment(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    cli = _cli(tmp_path, _RECORDING_CLI)
    monkeypatch.setenv("HOST_ONLY_SECRET", "host-secret")

    async def scenario():
        async with local_agent_launch(
            cli=str(cli),
            cwd=checkout,
            environment={"MODE": "review", "VENDOR_API_TOKEN": "scoped-token"},
        ) as launch:
            assert launch.cwd == str(checkout.resolve())
            assert launch.environment["MODE"] == "review"
            assert launch.environment["PATH"] == os.environ["PATH"]
            assert "HOST_ONLY_SECRET" not in launch.environment
            assert launch.secret_values == ("scoped-token",)
            # The adapters blank every host variable the way the SDKs receive them.
            process = await asyncio.create_subprocess_exec(
                launch.executable,
                "--print",
                cwd=launch.cwd,
                env={**scrub_external_environment(launch.environment), "SDK_SETTING": "1"},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
            assert process.stdout is not None and process.stdin is not None
            record = json.loads(await process.stdout.readline())
            process.stdin.write(b"hello\n")
            await process.stdin.drain()
            assert await process.stdout.readline() == b"HELLO\n"
            process.stdin.close()
            assert await process.wait() == 3
            assert _alive(record["descendant"])
            # The descendant keeps the CLI's group, so close still owns it.
            assert len(list((Path(launch.executable).parent / "groups").iterdir())) == 1
        assert await _wait_gone(record["descendant"])
        assert record["argv"] == ["--print"]
        assert record["cwd"] == str(checkout.resolve())
        assert record["env"]["MODE"] == "review"
        assert record["env"]["SDK_SETTING"] == "1"
        assert "HOST_ONLY_SECRET" not in record["env"]
        assert all(value or key in launch.environment for key, value in record["env"].items())

    asyncio.run(scenario())


def test_local_launch_close_stops_a_running_cli_and_refuses_later_starts(tmp_path):
    cli = _cli(tmp_path, _RECORDING_CLI)

    async def scenario():
        async with local_agent_launch(cli=str(cli), cwd=tmp_path) as launch:
            process = await asyncio.create_subprocess_exec(
                launch.executable,
                env=scrub_external_environment(launch.environment),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
            assert process.stdout is not None
            record = json.loads(await process.stdout.readline())
            await launch.close()
            await launch.close()
            assert await process.wait() == 128 + 9
            assert await _wait_gone(record["descendant"])
            late = await asyncio.create_subprocess_exec(
                launch.executable,
                env=scrub_external_environment(launch.environment),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
            )
            assert await late.stdout.read() == b""
            assert await late.wait() == 127

    asyncio.run(scenario())


def test_local_launch_keeps_group_owner_until_cleanup_after_cli_exits(tmp_path):
    cli = _cli(tmp_path, "print('2.1.300 (Claude Code)')\n")

    async def scenario():
        async with local_agent_launch(cli=str(cli), cwd=tmp_path) as launch:
            process = await asyncio.create_subprocess_exec(
                launch.executable,
                "-v",
                env=scrub_external_environment(launch.environment),
                stdout=asyncio.subprocess.PIPE,
            )
            output, _ = await process.communicate()
            assert output == b"2.1.300 (Claude Code)\n" and process.returncode == 0
            owners = list((Path(launch.executable).parent / "groups").iterdir())
            assert len(owners) == 1
            owner = int(owners[0].name)
            assert _alive(owner)
        assert await _wait_gone(owner)

    asyncio.run(scenario())


def test_local_launch_reports_unconfirmed_cleanup_for_retry(tmp_path, monkeypatch):
    import cayu.external_agents.local as local

    cli = _cli(tmp_path, "import time\ntime.sleep(300)\n")

    async def delay_stop(writer):
        pass

    monkeypatch.setattr(local, "_request_group_stop", delay_stop)

    async def scenario():
        settings = {"cli": str(cli), "cwd": tmp_path, "close_timeout_s": 0.2}
        with pytest.raises(ExternalCleanupIncomplete) as failure:
            async with local_agent_launch(**settings) as launch:
                process = await asyncio.create_subprocess_exec(
                    launch.executable, env=scrub_external_environment(launch.environment)
                )
                groups = Path(launch.executable).parent / "groups"
                for _ in range(250):
                    if any(groups.iterdir()):
                        break
                    await asyncio.sleep(0.02)
        # The state directory is gone; the retry still knows the CLI's group.
        assert not groups.exists() and process.returncode is None
        monkeypatch.undo()
        await failure.value.retry()
        assert await process.wait() == 128 + 9

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"cli": "cayu-no-such-vendor-cli"}, "not found"),
        ({"environment": {"BAD-NAME": "x"}}, "portable"),
        ({"cwd": "missing"}, "No such file"),
        ({"close_timeout_s": 0}, "positive"),
    ],
)
def test_local_launch_refuses_invalid_configuration(tmp_path, arguments, error):
    cli = _cli(tmp_path, "")
    settings = {"cli": str(cli), "cwd": tmp_path, **arguments}
    if settings["cwd"] == "missing":
        settings["cwd"] = tmp_path / "missing"

    async def scenario():
        async with local_agent_launch(**settings):
            pass

    with pytest.raises((ValueError, OSError), match=error):
        asyncio.run(scenario())


def test_codex_without_a_gateway_keeps_its_own_provider():
    @asynccontextmanager
    async def launch(context):
        yield None

    local = CodexAgentAdapter(model="codex", launch=launch)
    assert not any(item.startswith("model_provider") for item in local.overrides)
    assert {"project_doc_max_bytes=0", 'web_search="disabled"'} <= set(local.overrides)
    proxied = CodexAgentAdapter(model="codex", base_url="http://gateway/models", launch=launch)
    assert 'model_provider="cayu"' in proxied.overrides


# Speaks the Claude Code stream-json control protocol the SDK drives: answers
# initialize, asks the PreToolUse hook and can_use_tool about one Read, streams
# one message with usage and returns a result with the reported cost.
_FAKE_CLAUDE = """\
import json, os, subprocess, sys
if sys.argv[1:] == ["-v"]:
    print("2.1.300 (Claude Code)")
    sys.exit(0)
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(300)"],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
record = {"argv": sys.argv[1:], "cwd": os.getcwd(), "env": dict(os.environ),
          "descendant": child.pid}
counter = 0

def send(message):
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()

def receive():
    while True:
        line = sys.stdin.readline()
        if not line:
            sys.exit(0)
        message = json.loads(line)
        if message.get("type") != "control_request" or message["request"]["subtype"] == "initialize":
            return message
        send({"type": "control_response", "response": {
            "subtype": "success", "request_id": message["request_id"], "response": {}}})

def ask(request):
    global counter
    counter += 1
    identity = f"cli_{counter}"
    send({"type": "control_request", "request_id": identity, "request": request})
    while True:
        message = receive()
        if message.get("type") == "control_response" and \\
                message["response"]["request_id"] == identity:
            return message["response"]["response"]

initialize = receive()
hook = initialize["request"]["hooks"]["PreToolUse"][0]["hookCallbackIds"][0]
send({"type": "control_response", "response": {
    "subtype": "success", "request_id": initialize["request_id"], "response": {}}})
while receive().get("type") != "user":
    pass
arguments = {"file_path": "README.md"}
record["hook"] = ask({"subtype": "hook_callback", "callback_id": hook, "tool_use_id": "toolu_1",
                      "input": {"hook_event_name": "PreToolUse", "tool_name": "Read",
                                "tool_input": arguments}})
record["permission"] = ask({"subtype": "can_use_tool", "tool_name": "Read", "input": arguments,
                            "tool_use_id": "toolu_1", "permission_suggestions": []})
with open(os.path.join(os.getcwd(), "..", "fake-claude.json"), "w") as output:
    json.dump(record, output)
session = "fake-session"
message = {"id": "msg_1", "model": "claude-test", "role": "assistant", "type": "message",
           "usage": {"input_tokens": 120, "output_tokens": 1}}
send({"type": "stream_event", "uuid": "e1", "session_id": session,
      "event": {"type": "message_start", "message": {**message, "content": []}}})
send({"type": "assistant", "uuid": "a1", "session_id": session,
      "message": {**message, "content": [{"type": "text", "text": "No problems found."}]}})
send({"type": "stream_event", "uuid": "e2", "session_id": session,
      "event": {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 30}}})
send({"type": "stream_event", "uuid": "e3", "session_id": session,
      "event": {"type": "message_stop"}})
send({"type": "result", "subtype": "success", "duration_ms": 5, "duration_api_ms": 4,
      "is_error": False, "num_turns": 1, "session_id": session,
      "result": "No problems found.", "total_cost_usd": 0.0012,
      "usage": {"input_tokens": 120, "output_tokens": 30},
      "modelUsage": {"claude-test": {"inputTokens": 120, "outputTokens": 30}}})
while sys.stdin.readline():
    pass
"""


def test_claude_adapter_runs_the_installed_sdk_through_a_local_launch(tmp_path, monkeypatch):
    """The installed claude-agent-sdk drives the adapter end to end, with no gateway."""
    if os.environ.get(_REQUIRE_SDK_ENV) == "1":
        import claude_agent_sdk as sdk
    else:
        sdk = pytest.importorskip("claude_agent_sdk")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    cli = _cli(tmp_path, _FAKE_CLAUDE, "claude")
    monkeypatch.setenv("HOST_ONLY_SECRET", "host-secret")
    decisions = []

    async def allow_reads(name, arguments, call_id):
        decisions.append((name, arguments, call_id))
        return name == "Read"

    @asynccontextmanager
    async def launch(context):
        async with local_agent_launch(cli=str(cli), cwd=checkout) as prepared:
            yield prepared

    async def scenario():
        app = CayuApp(
            session_store=SQLiteSessionStore(tmp_path / "sessions.db"),
            task_store=SQLiteTaskStore(tmp_path / "tasks.db"),
            enable_logging=False,
        )
        try:
            prompt = "Review the checkout."
            result = await run_external_agent(
                app,
                RunRequest(
                    agent_name="claude-local",
                    messages=[Message.text("user", prompt)],
                    execution_deadline=ExecutionDeadline.after(60),
                ),
                ClaudeAgentAdapter(model="claude-test", launch=launch),
                prompt=prompt,
                authorize=allow_reads,
            )
            assert result.output == "No problems found."
            task = await app.task_store.load_task(result.task_id)
            assert task.status is TaskStatus.COMPLETED
            usage = await app.get_session_usage(result.session_id)
            assert (usage.usage.input_tokens, usage.usage.output_tokens) == (120, 30)
        finally:
            await app.aclose()

    asyncio.run(scenario())
    record = json.loads((tmp_path / "fake-claude.json").read_text())
    assert decisions == [("Read", {"file_path": "README.md"}, "toolu_1")]
    assert record["hook"]["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert record["permission"]["behavior"] == "allow"
    assert record["cwd"] == str(checkout.resolve())
    assert record["env"]["CLAUDE_AGENT_SDK_VERSION"] == sdk.__version__
    assert "HOST_ONLY_SECRET" not in record["env"]
    assert "--model" in record["argv"] and "claude-test" in record["argv"]
    assert asyncio.run(_wait_gone(record["descendant"]))


@pytest.mark.parametrize("relative_path", [True, False])
def test_local_launch_resolves_executable_before_changing_directory(
    tmp_path, monkeypatch, relative_path
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _cli(tmp_path, "print('started')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "." + os.pathsep + os.environ["PATH"])

    async def scenario():
        async with local_agent_launch(
            cli="./vendor-cli" if relative_path else "vendor-cli",
            cwd=checkout,
        ) as launch:
            process = await asyncio.create_subprocess_exec(
                launch.executable,
                cwd=launch.cwd,
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            assert process.returncode == 0
            assert stdout == b"started\n"

    asyncio.run(scenario())


def test_local_launch_supports_interpreter_paths_with_spaces(tmp_path, monkeypatch):
    cli = _cli(tmp_path, "print('started')\n")
    interpreter = tmp_path / "space directory" / "python"
    interpreter.parent.mkdir()
    interpreter.symlink_to(sys.executable)
    monkeypatch.setattr(sys, "executable", str(interpreter))

    async def scenario():
        async with local_agent_launch(cli=str(cli), cwd=tmp_path) as launch:
            process = await asyncio.create_subprocess_exec(
                launch.executable,
                cwd=launch.cwd,
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            assert process.returncode == 0
            assert stdout == b"started\n"

    asyncio.run(scenario())


def test_local_group_probe_reaps_only_its_adopted_children(monkeypatch):
    from cayu.external_agents.local import _group_alive

    reaped = iter([(22, 9), (23, 9)])
    calls = []

    def waitpid(pid, options):
        calls.append((pid, options))
        try:
            return next(reaped)
        except StopIteration:
            raise ChildProcessError from None

    def killpg(group, signal):
        assert group == 17 and signal == 0
        raise ProcessLookupError

    monkeypatch.setattr(os, "waitpid", waitpid)
    monkeypatch.setattr(os, "killpg", killpg)
    assert not _group_alive(17)
    assert calls == [(-17, os.WNOHANG)] * 3


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper adoption")
def test_local_launch_reaps_orphans_when_host_is_subreaper(tmp_path):
    cli = _cli(tmp_path, _RECORDING_CLI)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio, ctypes, os, sys
from cayu.external_agents import local_agent_launch
libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(36, 1, 0, 0, 0) == 0
async def scenario():
    async with local_agent_launch(cli=sys.argv[1], cwd=sys.argv[2], close_timeout_s=2) as launch:
        process = await asyncio.create_subprocess_exec(
            launch.executable, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        assert process.returncode == 3
    try:
        os.waitpid(-1, os.WNOHANG)
    except ChildProcessError:
        pass
    else:
        raise AssertionError("Adopted descendants were not reaped")
asyncio.run(scenario())
""",
            str(cli),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_local_launch_delayed_start_cannot_outlive_close_and_state_removal(tmp_path):
    marker = tmp_path / "cli-started"
    waiting = tmp_path / "waiting"
    release = tmp_path / "release"
    cli = _cli(tmp_path, f"open({str(marker)!r}, 'w').close()\n")

    async def scenario():
        async with local_agent_launch(cli=str(cli), cwd=tmp_path) as launch:
            script = Path(launch.executable).with_name("cli.py")
            source = script.read_text()
            source = source.replace(
                "        os.setsid()\n",
                "        os.setsid()\n"
                "        import time\n"
                f"        open({str(waiting)!r}, 'w').close()\n"
                f"        while not os.path.exists({str(release)!r}):\n"
                "            time.sleep(0.01)\n",
            )
            script.write_text(source)
            process = await asyncio.create_subprocess_exec(launch.executable)
            for _ in range(250):
                if waiting.exists():
                    break
                await asyncio.sleep(0.02)
            assert waiting.exists()
        assert not script.parent.exists()
        release.touch()
        assert await asyncio.wait_for(process.wait(), 5) == 127
        assert not marker.exists()

    asyncio.run(scenario())


def test_local_launch_keeps_owner_after_last_descendant_exits(tmp_path, monkeypatch):
    cli = _cli(
        tmp_path,
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.1)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
    )
    original_probe = os.killpg

    def probe_only(group, number):
        assert number == 0, "Host must never signal a saved process group"
        return original_probe(group, number)

    monkeypatch.setattr(os, "killpg", probe_only)

    async def scenario():
        async with local_agent_launch(cli=str(cli), cwd=tmp_path) as launch:
            process = await asyncio.create_subprocess_exec(launch.executable)
            assert await process.wait() == 0
            await asyncio.sleep(0.2)
            owners = list((Path(launch.executable).parent / "groups").iterdir())
            assert len(owners) == 1
            owner = int(owners[0].name)
            assert _alive(owner)
        assert await _wait_gone(owner)

    asyncio.run(scenario())


@pytest.mark.parametrize("acknowledge", [True, False])
def test_local_launch_cleanup_after_delayed_or_lost_supervisor(tmp_path, acknowledge, monkeypatch):
    import cayu.external_agents.local as local

    requests = []
    request_stop = local._request_group_stop

    async def count_request(writer):
        requests.append(writer)
        return await request_stop(writer)

    monkeypatch.setattr(local, "_request_group_stop", count_request)
    marker = tmp_path / "started"
    cli = _cli(tmp_path, f"open({str(marker)!r}, 'w').close()\nimport time; time.sleep(300)\n")

    async def scenario():
        async with local_agent_launch(cli=str(cli), cwd=tmp_path, close_timeout_s=2) as launch:
            script = Path(launch.executable).with_name("cli.py")
            source = script.read_text().replace(
                "                    command = connection.recv(1)",
                "                    import time; time.sleep(0.15)\n"
                "                    command = connection.recv(1)",
            )
            if not acknowledge:
                source = source.replace('connection.sendall(b"S")', "pass")
            script.write_text(source)
            process = await asyncio.create_subprocess_exec(launch.executable)
            for _ in range(250):
                if marker.exists():
                    break
                await asyncio.sleep(0.02)
            assert marker.exists()
            await launch.close()
            await launch.close()
            assert await process.wait() == 137
            assert len(requests) == 1

    asyncio.run(scenario())


def test_local_launch_lost_owner_remains_unconfirmed_until_group_is_empty(tmp_path):
    import signal

    marker = tmp_path / "cli-pid"
    cli = _cli(
        tmp_path,
        f"import os, time\nopen({str(marker)!r}, 'w').write(str(os.getpid()))\ntime.sleep(300)\n",
    )

    async def scenario():
        cli_pid = None
        try:
            with pytest.raises(ExternalCleanupIncomplete) as failure:
                async with local_agent_launch(
                    cli=str(cli), cwd=tmp_path, close_timeout_s=0.2
                ) as launch:
                    process = await asyncio.create_subprocess_exec(launch.executable)
                    for _ in range(250):
                        if marker.exists() and marker.read_text():
                            break
                        await asyncio.sleep(0.02)
                    cli_pid = int(marker.read_text())
                    groups = list((Path(launch.executable).parent / "groups").iterdir())
                    os.kill(int(groups[0].name), signal.SIGKILL)
            assert _alive(cli_pid)
            os.kill(cli_pid, signal.SIGKILL)
            assert await _wait_gone(cli_pid)
            await failure.value.retry()
            assert await process.wait() == 137
        finally:
            if cli_pid is not None and _alive(cli_pid):
                os.kill(cli_pid, signal.SIGKILL)

    asyncio.run(scenario())
