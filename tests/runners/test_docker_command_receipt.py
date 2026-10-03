from __future__ import annotations

import base64
import copy
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

from cayu.runners._docker_command_receipt import (
    IDENTITY_FIELDS,
    SCHEMA,
    guest_program,
    seal,
    verify,
)

KEY = bytes(range(32))
IDENTITY = {
    "schema": "cayu.durable_runner_operation.v1",
    **{field: "sha256:" + "a" * 64 for field in IDENTITY_FIELDS},
}


def _payload():
    return {
        "schema": SCHEMA,
        "request_sha256": "b" * 64,
        "identity": copy.deepcopy(IDENTITY),
        "timeout_seconds": 5,
        "output_limit": 1024,
        "exit_code": 0,
        "timed_out": False,
        "stdout": base64.b64encode(b"hello").decode(),
        "stderr": "",
        "stdout_bytes": 5,
        "stderr_bytes": 0,
    }


def _verify(document, **kwargs):
    expected = {
        "identity": IDENTITY,
        "key": KEY,
        "output_limit": 1024,
        "timeout_seconds": 5,
        "request_sha256": "b" * 64,
    }
    expected.update(kwargs)
    return verify(document, **expected)


def test_receipt_authentication_and_private_output():
    payload = _payload()
    assert _verify(seal(payload, KEY)) == payload
    with pytest.raises(ValueError, match="unauthenticated"):
        _verify(seal(payload, bytes(reversed(KEY))))
    envelope = json.loads(seal(payload, KEY))
    envelope["payload"]["stdout"] = base64.b64encode(b"private-canary").decode()
    with pytest.raises(ValueError) as error:
        _verify(json.dumps(envelope).encode())
    assert "private-canary" not in str(error.value)


@pytest.mark.parametrize("field", sorted(IDENTITY_FIELDS))
def test_every_expected_identity_field_must_match(field):
    identity = copy.deepcopy(IDENTITY)
    identity[field] = "sha256:" + "b" * 64
    with pytest.raises(ValueError, match="unauthenticated"):
        _verify(seal(_payload(), KEY), identity=identity)


@pytest.mark.parametrize(
    "field,value",
    [
        ("exit_code", True),
        ("timed_out", 1),
        ("stdout_bytes", True),
        ("stdout_bytes", 6),
        ("output_limit", True),
        ("timeout_seconds", True),
        ("schema", "future"),
        ("stdout", "not-base64"),
    ],
)
def test_even_authenticated_malformed_receipts_are_rejected(field, value):
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError):
        _verify(seal(payload, KEY))


@pytest.mark.parametrize("bounds", [{"output_limit": 512}, {"timeout_seconds": 4}])
def test_receipt_cannot_reconcile_different_execution_bounds(bounds):
    with pytest.raises(ValueError):
        _verify(seal(_payload(), KEY), **bounds)


def _launch(tmp_path, code, *, timeout=5, limit=1024):
    return {
        "identity": IDENTITY,
        "key": KEY.hex(),
        "request_sha256": "b" * 64,
        "directory": str(tmp_path / "operation"),
        "argv": [sys.executable, "-c", code],
        "timeout_seconds": timeout,
        "output_limit": limit,
    }


def _run(launch, *, program=None):
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", guest_program() if program is None else program],
        input=json.dumps(launch) + "\n",
        text=True,
        capture_output=True,
        timeout=12,
    )


def _wait_file(path: Path, timeout=8):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            pytest.fail(f"Timed out waiting for {path.name}")
        time.sleep(0.01)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux guest supervision requires prctl")
def test_supervisor_output_is_bounded_and_duplicate_launch_never_executes(tmp_path):
    launch = _launch(
        tmp_path, "import sys; print('x'*10000); print('err', file=sys.stderr); sys.exit(7)"
    )
    result = _run(launch)
    assert result.returncode == 0, result.stderr
    receipt_path = tmp_path / "operation/receipt.json"
    saved = receipt_path.read_bytes()
    result = _verify(saved)
    assert result["exit_code"] == 7
    assert result["stdout_bytes"] == 10001
    assert len(base64.b64decode(result["stdout"])) == 1024
    marker = tmp_path / "unexpected"
    launch["argv"][-1] = f"open({str(marker)!r}, 'w').close()"
    assert _run(launch).returncode == 125
    assert not marker.exists()
    assert receipt_path.read_bytes() == saved


@pytest.mark.skipif(sys.platform != "linux", reason="Linux guest supervision requires prctl")
def test_command_cannot_read_supervisors_private_stdin(tmp_path):
    code = """
import os
try:
    fd = os.open('/proc/' + str(os.getppid()) + '/fd/0', os.O_RDONLY)
except PermissionError:
    print('denied')
else:
    os.close(fd)
    raise SystemExit(17)
"""
    result = _run(_launch(tmp_path, code))
    assert result.returncode == 0, result.stderr
    receipt = _verify((tmp_path / "operation/receipt.json").read_bytes())
    assert receipt["exit_code"] == 0
    assert base64.b64decode(receipt["stdout"]) == b"denied\n"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux guest supervision requires prctl")
def test_supervisor_sigterm_contains_work_without_publishing_success(tmp_path):
    started = tmp_path / "started"
    launch = _launch(
        tmp_path,
        f"import os,time; open({str(started)!r},'w').write(str(os.getpid())); time.sleep(30)",
    )
    owner = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", guest_program()], stdin=subprocess.PIPE
    )
    assert owner.stdin is not None
    try:
        owner.stdin.write((json.dumps(launch) + "\n").encode())
        owner.stdin.close()
        _wait_file(started)
        owner.send_signal(signal.SIGTERM)
        assert owner.wait(timeout=5) == 125
        with pytest.raises(ProcessLookupError):
            os.kill(int(started.read_text()), 0)
        assert not (tmp_path / "operation/receipt.json").exists()
    finally:
        if owner.poll() is None:
            owner.send_signal(signal.SIGTERM)
        owner.wait(timeout=8)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux guest supervision requires prctl")
def test_rejected_launch_does_not_disclose_key_or_dispatch(tmp_path):
    launch = _launch(tmp_path, "print('must-not-run')")
    launch["timeout_seconds"] = True
    result = _run(launch)
    assert result.returncode == 125
    assert result.stdout == result.stderr == ""
    assert not (tmp_path / "operation").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux guest supervision requires prctl")
def test_incomplete_private_transfer_cannot_dispatch(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", guest_program()],
        input=json.dumps(_launch(tmp_path, "print('must-not-run')")),
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 125
    assert result.stdout == result.stderr == ""
    assert not (tmp_path / "operation").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux guest supervision requires prctl")
@pytest.mark.parametrize("after_publication", [False, True])
def test_publication_failure_after_command_does_not_authorize_redispatch(
    tmp_path, after_publication
):
    marker = tmp_path / "dispatches"
    launch = _launch(tmp_path, f"open({str(marker)!r}, 'a').write('once\\n')")
    program = guest_program()
    target = "    os.fsync(directory)" if after_publication else "    os.link("
    assert program.count(target) == 1
    program = program.replace(
        target, "    raise OSError('injected-private-publication-failure')\n" + target
    )
    result = _run(launch, program=program)
    assert result.returncode == 125
    assert result.stdout == result.stderr == ""
    receipt = tmp_path / "operation/receipt.json"
    assert receipt.exists() is after_publication
    if after_publication:
        assert _verify(receipt.read_bytes())["exit_code"] == 0
    assert _run(launch).returncode == 125
    assert marker.read_text() == "once\n"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux guest supervision requires prctl")
def test_supervisor_deadline_reaps_escaped_descendants_before_receipt(tmp_path):
    child_path = tmp_path / "child"
    code = (
        "import os,time; pid=os.fork(); "
        f"open({str(child_path)!r},'w').write(str(pid)) if pid else os.setsid(); "
        "time.sleep(30)"
    )
    start = time.monotonic()
    result = _run(_launch(tmp_path, code, timeout=1))
    assert result.returncode == 0, result.stderr
    assert time.monotonic() - start < 6
    receipt = _verify((tmp_path / "operation/receipt.json").read_bytes(), timeout_seconds=1)
    assert receipt["timed_out"] is True
    assert receipt["exit_code"] == -signal.SIGKILL
    child = int(child_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux guest supervision requires prctl")
@pytest.mark.parametrize("orphan_descendant", [False, True])
def test_terminal_receipt_survives_real_mid_command_worker_sigkill(tmp_path, orphan_descendant):
    started, release = tmp_path / "started", tmp_path / "release"
    code = (
        "import pathlib,time\n"
        f"pathlib.Path({str(started)!r}).touch()\n"
        f"while not pathlib.Path({str(release)!r}).exists(): time.sleep(0.01)\n"
        "print('command-finished')"
    )
    if orphan_descendant:
        code = "import os\nif os.fork(): os._exit(0)\nos.setsid()\n" + code
    launch = _launch(tmp_path, code)
    worker_pid = tmp_path / "worker"
    # This helper owns the attached client/worker lifetime, not the receipt.
    # The actual supervisor and command are separate real processes.
    # A separate subreaper acts as Docker init and reaps the orphaned supervisor.
    # Do not change pytest's process-wide subreaper state or leave zombies to PID 1.
    witness_code = f"""
import ctypes,os,pathlib,subprocess,sys
assert ctypes.CDLL(None).prctl(36,1,0,0,0) == 0
launch = sys.stdin.buffer.read()
worker = os.fork()
if worker == 0:
    p = subprocess.Popen([sys.executable,'-I','-S','-c',{guest_program()!r}],
                         stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    p.communicate(launch)
    os._exit(p.returncode)
pathlib.Path({str(worker_pid)!r}).write_text(str(worker))
pid,status = os.waitpid(worker,0)
assert os.waitstatus_to_exitcode(status) == -9
while True:
    try:
        pid,status = os.waitpid(-1,0)
        assert os.waitstatus_to_exitcode(status) == 0
    except ChildProcessError:
        break
"""
    witness = subprocess.Popen([sys.executable, "-c", witness_code], stdin=subprocess.PIPE)
    assert witness.stdin is not None
    try:
        witness.stdin.write((json.dumps(launch) + "\n").encode())
        witness.stdin.close()
        _wait_file(started)
        assert not (tmp_path / "operation/receipt.json").exists()
        os.kill(int(worker_pid.read_text()), signal.SIGKILL)
        release.touch()
        path = tmp_path / "operation/receipt.json"
        _wait_file(path)
        receipt = _verify(path.read_bytes())
        assert receipt["timed_out"] is False
        assert base64.b64decode(receipt["stdout"]) == b"command-finished\n"
        assert witness.wait(timeout=3) == 0
    finally:
        release.touch()
        # The bounded guest deadline also settles the command if an assertion
        # fails before release. Retain the witness until it has reaped children.
        if witness.poll() is None and worker_pid.exists():
            with suppress(ProcessLookupError):
                os.kill(int(worker_pid.read_text()), signal.SIGKILL)
        witness.wait(timeout=10)
