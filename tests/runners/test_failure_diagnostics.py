from __future__ import annotations

import asyncio
import errno
import json

import pytest

from cayu.runners._diagnostics import tag_runner_failure_phase
from cayu.runners._subprocess import SubprocessCommand, run_subprocess
from cayu.runners.base import RunnerExecutionError, runner_execution_error


@pytest.mark.parametrize("number", [errno.E2BIG, errno.EMFILE, errno.ENOSPC, errno.EIO])
@pytest.mark.parametrize("phase", ["launch", "transport", "stream_handling", "cleanup"])
def test_safe_errno_survives_wrapping(number, phase):
    raw = OSError(number, "secret command /private/path TOKEN=secret")
    tag_runner_failure_phase(raw, phase)
    first = runner_execution_error(raw, adapter="docker")
    wrapped = runner_execution_error(first, adapter="docker")
    assert wrapped.diagnostic == first.diagnostic
    assert wrapped.diagnostic["errno"] == number
    assert wrapped.diagnostic["errno_code"] == errno.errorcode[number]
    assert wrapped.diagnostic["execution_phase"] == phase
    assert wrapped.artifacts == [wrapped.diagnostic]
    assert "secret" not in json.dumps(wrapped.artifacts)
    assert wrapped.__cause__ is None
    assert wrapped.__context__ is None


@pytest.mark.parametrize("number", [None, True, -1, 2**80, "28", object()])
def test_invalid_errno_is_unavailable(number):
    raw = OSError("secret")
    raw.errno = number
    safe = runner_execution_error(raw, adapter="docker").diagnostic
    assert safe["errno"] is None
    assert safe["errno_code"] is None
    assert safe["execution_phase"] == "unknown"


def test_unknown_numeric_errno_has_no_guessed_symbol():
    safe = runner_execution_error(OSError(123456, "E2BIG secret"), adapter="docker").diagnostic
    assert safe["errno"] == 123456
    assert safe["errno_code"] is None


def test_hostile_exception_and_embedded_diagnostics_do_not_call_hooks():
    class Hostile(OSError):
        @property
        def errno(self):
            pytest.fail("errno property called")

        @property
        def diagnostic(self):
            pytest.fail("diagnostic property called")

        def __str__(self):
            pytest.fail("formatting called")

    class HostileValue:
        def __hash__(self):
            pytest.fail("hash called")

        def __eq__(self, other):
            pytest.fail("comparison called")

    raw = Hostile("secret")
    safe = runner_execution_error(raw, adapter="docker").diagnostic
    assert safe["errno"] is None
    assert safe["execution_phase"] == "unknown"
    malformed = RunnerExecutionError(
        diagnostic={
            "errno": HostileValue(),
            "errno_code": "SECRET",
            "execution_phase": HostileValue(),
            "error_type": HostileValue(),
        }
    )
    assert malformed.diagnostic["errno"] is None
    assert malformed.diagnostic["errno_code"] is None
    assert malformed.diagnostic["execution_phase"] == "unknown"
    spoofed = RuntimeError("secret")
    spoofed.diagnostic = {"errno": errno.E2BIG, "execution_phase": "launch"}
    assert runner_execution_error(spoofed, adapter="docker").diagnostic["errno"] is None


@pytest.mark.parametrize("number", [errno.E2BIG, errno.EMFILE])
def test_subprocess_launch_boundary_preserves_errno(monkeypatch, number):
    async def fail(*args, **kwargs):
        raise OSError(number, "secret argv")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail)
    with pytest.raises(OSError) as caught:
        asyncio.run(run_subprocess(SubprocessCommand(argv=["unused"])))
    safe = runner_execution_error(caught.value, adapter="local").diagnostic
    assert safe["errno"] == number
    assert safe["execution_phase"] == "launch"


def test_environment_file_creation_phase(monkeypatch):
    from cayu.runners._secrets import runner_env_file

    def fail(**kwargs):
        raise OSError(errno.ENOSPC, "secret path")

    monkeypatch.setattr("cayu.runners._secrets.tempfile.mkstemp", fail)
    with pytest.raises(OSError) as caught, runner_env_file({"TOKEN": "secret"}):
        pytest.fail("creation should fail")
    safe = runner_execution_error(caught.value, adapter="docker").diagnostic
    assert safe["errno"] == errno.ENOSPC
    assert safe["execution_phase"] == "filesystem"


def test_hostile_dictionary_keys_are_not_compared_during_diagnostics():
    class Key:
        def __hash__(self):
            return hash("diagnostic")

        def __eq__(self, other):
            pytest.fail("extension key compared")

    raw = OSError(errno.EIO, "secret")
    raw.__dict__[Key()] = "secret"
    safe = runner_execution_error(raw, adapter="docker")
    assert safe.diagnostic["errno"] == errno.EIO
    malformed = RunnerExecutionError(diagnostic={Key(): "secret"})
    assert malformed.diagnostic["errno"] is None


def test_environment_file_cleanup_phase(tmp_path, monkeypatch):
    from cayu.runners._secrets import runner_env_file

    paths = []

    def fail(path):
        paths.append(path)
        raise OSError(errno.EIO, "secret path")

    with monkeypatch.context() as patch:
        patch.setattr("cayu.runners._secrets.os.unlink", fail)
        with pytest.raises(OSError) as caught, runner_env_file({"TOKEN": "secret"}):
            pass
    for path in paths:
        from pathlib import Path

        Path(path).unlink()
    safe = runner_execution_error(caught.value, adapter="docker").diagnostic
    assert safe["errno"] == errno.EIO
    assert safe["execution_phase"] == "cleanup"
