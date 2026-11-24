"""run_isolated moves files in and out of any runner by the transport it offers."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest

from cayu.runners import (
    ExecCommand,
    IsolatedLimits,
    LocalRunner,
    isolated_run_key,
    run_isolated,
)

pytestmark = pytest.mark.anyio

UPPER = ExecCommand.process(
    sys.executable,
    "-c",
    "import pathlib\n"
    "source = pathlib.Path('in/a.txt').read_bytes()\n"
    "pathlib.Path('out').mkdir(exist_ok=True)\n"
    "pathlib.Path('out/a.txt').write_bytes(source.upper())\n"
    "pathlib.Path('result.json').write_text('{\"ok\": true}')\n",
)


class _TextOnlyRunner(LocalRunner):
    """A runner without binary streams, like a transport that only returns text."""

    def binary_stream_capability(self):
        return None


class _StagedRunner(LocalRunner):
    """A runner with staged copies, the shape a brokered sandbox has."""

    staged_copies = object()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.copies: list[str] = []

    async def copy_in(self, source, *, size, sha256, path=None):
        self.copies.append("in")
        data = source.read()
        assert len(data) == size
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(self.root, filter="data")

    async def copy_out(self, sink, *, max_bytes, paths=(), excludes=()):
        self.copies.append("out")
        with tarfile.open(fileobj=sink, mode="w") as archive:
            for path in paths:
                if (self.root / path).is_file():
                    archive.add(self.root / path, arcname=path)


@pytest.mark.parametrize("runner_type", [LocalRunner, _TextOnlyRunner, _StagedRunner])
async def test_inputs_go_in_and_declared_outputs_come_back(tmp_path: Path, runner_type) -> None:
    runner = runner_type(root=tmp_path)
    run = await run_isolated(
        runner,
        UPPER,
        inputs={"in/a.txt": b"hello"},
        outputs=["out/a.txt", "result.json", "never-written.txt"],
    )
    assert run.result.exit_code == 0
    assert run.outputs == {"out/a.txt": b"HELLO", "result.json": b'{"ok": true}'}
    assert run.missing == ("never-written.txt",)
    assert not run.succeeded
    assert run.key == isolated_run_key(UPPER, {"in/a.txt": b"hello"})
    if runner_type is _StagedRunner:
        assert runner.copies == ["in", "out"]


async def test_failed_command_returns_no_outputs(tmp_path: Path) -> None:
    fail = ExecCommand.process(
        sys.executable, "-c", "open('out.txt', 'w').write('x'); raise SystemExit(3)"
    )
    run = await run_isolated(LocalRunner(root=tmp_path), fail, inputs={}, outputs=["out.txt"])
    assert run.result.exit_code == 3
    assert run.outputs == {} and run.missing == () and not run.succeeded


async def test_limits_bound_inputs_and_outputs(tmp_path: Path) -> None:
    runner = LocalRunner(root=tmp_path)
    with pytest.raises(ValueError, match="max_input_bytes"):
        await run_isolated(
            runner,
            UPPER,
            inputs={"in/a.txt": b"x" * 11},
            outputs=[],
            limits=IsolatedLimits(max_input_bytes=10),
        )
    with pytest.raises((ValueError, RuntimeError)):
        await run_isolated(
            runner,
            UPPER,
            inputs={"in/a.txt": b"x" * 100},
            outputs=["out/a.txt"],
            limits=IsolatedLimits(max_output_file_bytes=10),
        )


@pytest.mark.parametrize("path", ["/etc/passwd", "../up", "a/../b", "./a", "a//b", " a", ""])
async def test_paths_must_stay_inside_the_working_directory(tmp_path: Path, path: str) -> None:
    runner = LocalRunner(root=tmp_path)
    with pytest.raises(ValueError):
        await run_isolated(runner, UPPER, inputs={path: b"x"}, outputs=[])
    with pytest.raises(ValueError):
        await run_isolated(runner, UPPER, inputs={}, outputs=[path])


async def test_arguments_are_checked(tmp_path: Path) -> None:
    runner = LocalRunner(root=tmp_path)
    with pytest.raises(TypeError):
        await run_isolated(runner, UPPER, inputs={"a": "text"}, outputs=[])  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        await run_isolated(runner, UPPER, inputs={}, outputs="a")
    with pytest.raises(ValueError):
        await run_isolated(runner, UPPER, inputs={}, outputs=["a", "a"])
    with pytest.raises(TypeError):
        await run_isolated(object(), UPPER, inputs={}, outputs=[])  # type: ignore[arg-type]


def test_run_key_is_stable_and_covers_command_and_contents() -> None:
    key = isolated_run_key(UPPER, {"b": b"2", "a": b"1"})
    assert key == isolated_run_key(UPPER, {"a": b"1", "b": b"2"})
    assert len(key) == 64
    assert key != isolated_run_key(UPPER, {"a": b"1", "b": b"3"})
    assert key != isolated_run_key(ExecCommand.process("true"), {"a": b"1", "b": b"2"})


@pytest.mark.parametrize("runner_type", [LocalRunner, _StagedRunner])
async def test_a_reused_runner_never_returns_an_earlier_runs_outputs(
    tmp_path: Path, runner_type
) -> None:
    runner = runner_type(root=tmp_path)
    write = ExecCommand.process(sys.executable, "-c", "open('out.csv', 'w').write('first')")
    first = await run_isolated(runner, write, inputs={}, outputs=["out.csv"])
    assert first.outputs == {"out.csv": b"first"}

    nothing = ExecCommand.process(sys.executable, "-c", "pass")
    second = await run_isolated(runner, nothing, inputs={}, outputs=["out.csv"])
    assert second.outputs == {} and second.missing == ("out.csv",) and not second.succeeded
