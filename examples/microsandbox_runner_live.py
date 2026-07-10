from __future__ import annotations

import asyncio
import os
from typing import cast

from _live_checks import require_cleanup_artifact, require_equal, require_exec_success
from cayu import ExecCommand, MicrosandboxRunner, RunnerCleanupPolicy


async def main() -> None:
    sandbox_name = os.environ.get("CAYU_MICROSANDBOX_NAME", "cayu-live-runner")
    image = os.environ.get("CAYU_MICROSANDBOX_IMAGE", "alpine")
    cancel_delay_s = float(os.environ.get("CAYU_CANCEL_DELAY_S", "0.2"))
    cancellation_cleanup = cast(
        "RunnerCleanupPolicy",
        os.environ.get("CAYU_RUNNER_CANCELLATION_CLEANUP", "command"),
    )
    timeout_cleanup = cast(
        "RunnerCleanupPolicy",
        os.environ.get("CAYU_RUNNER_TIMEOUT_CLEANUP", "command"),
    )

    os.environ["CAYU_HOST_SECRET_SHOULD_NOT_LEAK"] = "hidden"

    print(f"sandbox_name {sandbox_name}")
    print(f"image {image}")
    print(f"cancellation_cleanup {cancellation_cleanup}")
    print(f"timeout_cleanup {timeout_cleanup}")
    print(f"cancel_delay_s {cancel_delay_s}")
    print("creating sandbox")

    async with await MicrosandboxRunner.create(
        sandbox_name,
        image=image,
        replace=True,
        close_action="remove",
        cancellation_cleanup=cancellation_cleanup,
        timeout_cleanup=timeout_cleanup,
    ) as runner:
        print("sandbox ready")

        pwd = await runner.exec(ExecCommand.process("pwd"))
        require_exec_success(pwd, label="pwd")
        print(f"pwd {pwd.stdout.strip()}")

        os_release = await runner.exec(
            ExecCommand.process("sh", "-c", "grep '^ID=' /etc/os-release || true"),
            timeout_s=30,
        )
        require_exec_success(os_release, label="os_release")
        print(f"os_release {os_release.stdout.strip()}")

        env_check = await runner.exec(
            ExecCommand.process(
                "sh",
                "-c",
                (
                    'if [ -n "$CAYU_HOST_SECRET_SHOULD_NOT_LEAK" ]; '
                    "then echo visible; else echo hidden; fi"
                ),
            )
        )
        require_exec_success(env_check, stdout="hidden\n", label="host_secret")
        print(f"host_secret {env_check.stdout.strip()}")

        explicit_env = await runner.exec(
            ExecCommand.process(
                "sh",
                "-c",
                "printf '%s\\n' \"$CAYU_EXPLICIT_ENV\"",
            ),
            env={"CAYU_EXPLICIT_ENV": "visible"},
        )
        require_exec_success(explicit_env, stdout="visible\n", label="explicit_env")
        print(f"explicit_env {explicit_env.stdout.strip()}")

        bounded = await runner.exec(
            ExecCommand.process(
                "sh",
                "-c",
                "printf abcdef; printf uvwxyz >&2",
            ),
            output_limit_bytes=3,
        )
        print(
            "bounded "
            f"stdout={bounded.stdout!r} "
            f"stdout_truncated={bounded.stdout_truncated} "
            f"stderr={bounded.stderr!r} "
            f"stderr_truncated={bounded.stderr_truncated}"
        )
        require_equal(bounded.stdout, "abc", "bounded stdout")
        require_equal(bounded.stderr, "uvw", "bounded stderr")
        require_equal(bounded.stdout_truncated, True, "bounded stdout_truncated")
        require_equal(bounded.stderr_truncated, True, "bounded stderr_truncated")

        cwd = await runner.exec(
            ExecCommand.process("pwd"),
            cwd=".",
        )
        require_exec_success(cwd, label="relative_cwd")
        print(f"relative_cwd {cwd.stdout.strip()}")

        cancelled = asyncio.create_task(runner.exec(ExecCommand.process("sh", "-c", "sleep 30")))
        await asyncio.sleep(cancel_delay_s)
        cancelled.cancel()
        try:
            await cancelled
            raise RuntimeError("Microsandbox command cancellation did not cancel the task")
        except asyncio.CancelledError as exc:
            artifacts = getattr(exc, "artifacts", [])
            require_cleanup_artifact(
                artifacts,
                adapter="microsandbox",
                action=_cleanup_action(cancellation_cleanup),
            )
            print(f"cancelled true artifacts={artifacts}")

        try:
            after_cancel = await runner.exec(ExecCommand.process("printf", "after-cancel"))
            require_exec_success(after_cancel, stdout="after-cancel", label="after_cancel")
            print(f"after_cancel reusable stdout={after_cancel.stdout!r}")
        except RuntimeError as exc:
            if cancellation_cleanup != "sandbox":
                raise
            print(f"after_cancel closed: {exc}")

        print("closing sandbox")

    print("completed")


def _cleanup_action(policy: RunnerCleanupPolicy) -> str:
    if policy == "sandbox":
        return "kill_sandbox"
    return "kill_command"


if __name__ == "__main__":
    asyncio.run(main())
