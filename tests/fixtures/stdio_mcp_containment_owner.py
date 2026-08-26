from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import suppress
from pathlib import Path

from cayu.mcp import McpServerSpec, StdioMcpClient, StdioMcpProcessLifetime
from cayu.mcp._stdio_process import ContainedStdioMcpProcess
from cayu.mcp.stdio import _stdio_containment_rendezvous_identity


async def _run(args: argparse.Namespace) -> None:
    fixture = Path(__file__).with_name("fake_mcp_process_tree.py")
    if args.close_standard_fds:
        for fd in (0, 1, 2):
            with suppress(OSError):
                os.close(fd)
    held_fds: list[int] = []
    if args.high_descriptors:
        while not held_fds or held_fds[-1] < 1100:
            held_fds.append(os.open(os.devnull, os.O_RDWR))
    server = McpServerSpec(
        name="contained-process-tree",
        connection_id=args.connection_id,
        command=[
            sys.executable,
            str(fixture),
            "--role",
            args.server_role,
            "--lock-path",
            str(args.lock_path),
            "--state-path",
            str(args.server_state_path),
            *(
                ["--eof-marker-path", str(args.eof_marker_path)]
                if args.eof_marker_path is not None
                else []
            ),
            *(
                ["--eof-release-path", str(args.eof_release_path)]
                if args.eof_release_path is not None
                else []
            ),
        ],
    )
    if args.attack_rendezvous:
        server = server.model_copy(
            update={
                "env": {
                    "CAYU_TEST_CONTAINMENT_RENDEZVOUS_IDENTITY": (
                        _stdio_containment_rendezvous_identity(server)
                    )
                }
            }
        )
    try:
        session = await StdioMcpClient(
            containment_startup_timeout_s=args.startup_timeout_s,
            containment_term_timeout_s=args.term_timeout_s,
            containment_kill_timeout_s=args.kill_timeout_s,
            process_lifetime=StdioMcpProcessLifetime(args.process_lifetime),
        ).connect(server)
    finally:
        for fd in held_fds:
            os.close(fd)
    process = session.process
    inherited_writer_pid = os.fork()
    if inherited_writer_pid == 0:
        # Deliberately retain the fork-inherited owner-liveness descriptor.
        # Strong containment must also verify the direct parent relationship.
        if args.close_inherited_protocol_stdin and process.stdin is not None:
            pipe = process.stdin.transport.get_extra_info("pipe")
            if pipe is None:
                raise AssertionError("stdio process stdin has no pipe transport")
            os.close(pipe.fileno())
        time.sleep(30)
        os._exit(0)
    process_evidence: dict[str, object] = {
        "owner_pid": os.getpid(),
        "inherited_writer_pid": inherited_writer_pid,
        "parent_death_containment": session.process_capability_evidence.state_for(
            "parent_death_containment"
        ),
        "persistent_detached": session.process_capability_evidence.state_for("persistent_detached"),
    }
    if isinstance(process, ContainedStdioMcpProcess):
        process_evidence.update(
            {
                "supervisor_pid": process.pid,
                "anchor_pid": process.anchor_pid,
                "anchor_pgid": process.anchor_pgid,
                "launcher_pid": process.server_pid,
                "owner_control_fd": process._control.fileno(),
                "owner_liveness_fd": process._owner_write_fd,
            }
        )
    else:
        process_evidence["direct_process_pid"] = process.pid
    args.owner_state_path.write_text(
        json.dumps(process_evidence, sort_keys=True),
        encoding="utf-8",
    )
    if args.graceful_close_trigger_path is not None:
        while not args.graceful_close_trigger_path.exists():
            await asyncio.sleep(0.01)
        await session.close()
        return
    await asyncio.Event().wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-state-path", type=Path, required=True)
    parser.add_argument("--server-state-path", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--server-role", choices=("server", "launcher"), required=True)
    parser.add_argument("--close-standard-fds", action="store_true")
    parser.add_argument("--high-descriptors", action="store_true")
    parser.add_argument("--graceful-close-trigger-path", type=Path)
    parser.add_argument("--eof-marker-path", type=Path)
    parser.add_argument("--eof-release-path", type=Path)
    parser.add_argument("--close-inherited-protocol-stdin", action="store_true")
    parser.add_argument("--attack-rendezvous", action="store_true")
    parser.add_argument("--connection-id")
    parser.add_argument("--startup-timeout-s", type=float, default=3.0)
    parser.add_argument("--term-timeout-s", type=float, default=0.2)
    parser.add_argument("--kill-timeout-s", type=float, default=1.0)
    parser.add_argument(
        "--process-lifetime",
        choices=("parent_death_containment", "persistent_detached"),
        default="parent_death_containment",
    )
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
