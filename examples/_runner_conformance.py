from __future__ import annotations

import asyncio
from typing import Any

from cayu import ExecCommand, ExecResult, Runner

PIPE_PRESSURE_BYTES = 128 * 1024
CAPTURE_LIMIT_BYTES = 1024


async def verify_bounded_output_drain(
    runner: Runner,
    *,
    adapter: str,
) -> ExecResult:
    """Exercise one portable conformance scenario against fake or live runners."""

    script = (
        f"(yes o | head -c {PIPE_PRESSURE_BYTES}) & "
        f"(yes e | head -c {PIPE_PRESSURE_BYTES} >&2) & wait"
    )
    result = await asyncio.wait_for(
        runner.exec(
            ExecCommand.process("sh", "-c", script),
            timeout_s=30,
            output_limit_bytes=CAPTURE_LIMIT_BYTES,
        ),
        timeout=35,
    )
    observed = _bounded_output_observation(result)
    _require(
        not result.timed_out,
        adapter=adapter,
        observed=observed,
        artifacts=result.artifacts,
        detail="command timed out instead of draining output",
    )
    _require(
        result.exit_code == 0,
        adapter=adapter,
        observed=observed,
        artifacts=result.artifacts,
        detail="command did not exit successfully",
    )
    _require(
        len(result.stdout.encode("utf-8")) == CAPTURE_LIMIT_BYTES
        and len(result.stderr.encode("utf-8")) == CAPTURE_LIMIT_BYTES,
        adapter=adapter,
        observed=observed,
        artifacts=result.artifacts,
        detail="captured output did not honor the per-stream bound",
    )
    _require(
        result.stdout_truncated and result.stderr_truncated,
        adapter=adapter,
        observed=observed,
        artifacts=result.artifacts,
        detail="truncation flags did not report discarded output",
    )
    _require(
        result.stdout_bytes == PIPE_PRESSURE_BYTES and result.stderr_bytes == PIPE_PRESSURE_BYTES,
        adapter=adapter,
        observed=observed,
        artifacts=result.artifacts,
        detail="total byte counts did not include drained output beyond the capture bound",
    )
    return result


def _bounded_output_observation(result: ExecResult) -> dict[str, Any]:
    return {
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "stdout_captured_bytes": len(result.stdout.encode("utf-8")),
        "stderr_captured_bytes": len(result.stderr.encode("utf-8")),
        "stdout_bytes": result.stdout_bytes,
        "stderr_bytes": result.stderr_bytes,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
    }


def _require(
    condition: bool,
    *,
    adapter: str,
    observed: object,
    artifacts: object,
    detail: str,
) -> None:
    if condition:
        return
    raise AssertionError(
        "scenario=bounded-output-drain "
        f"adapter={adapter} capability=required observed={observed!r} "
        f"cleanup_artifact={artifacts!r}: {detail}"
    )
