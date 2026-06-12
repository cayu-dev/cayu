from __future__ import annotations

import asyncio
import os
from typing import cast

from cayu import E2BRunner, ExecCommand, RunnerCancelledError, RunnerCleanupPolicy


async def main() -> None:
    if not os.environ.get("E2B_API_KEY"):
        print("Set E2B_API_KEY to run this live E2B example.")
        return

    template = os.environ.get("CAYU_E2B_TEMPLATE")
    sandbox_timeout_s = int(os.environ.get("CAYU_E2B_SANDBOX_TIMEOUT_S", "300"))
    cancel_delay_s = float(os.environ.get("CAYU_CANCEL_DELAY_S", "2.0"))
    cancellation_cleanup = cast(
        "RunnerCleanupPolicy",
        os.environ.get("CAYU_RUNNER_CANCELLATION_CLEANUP", "command"),
    )
    timeout_cleanup = cast(
        "RunnerCleanupPolicy",
        os.environ.get("CAYU_RUNNER_TIMEOUT_CLEANUP", "command"),
    )
    print(f"template {template or '<e2b-default>'}")
    print(f"cancellation_cleanup {cancellation_cleanup}")
    print(f"timeout_cleanup {timeout_cleanup}")
    print(f"cancel_delay_s {cancel_delay_s}")
    print("creating sandbox")
    async with await E2BRunner.create(
        template=template,
        sandbox_timeout_s=sandbox_timeout_s,
        close_action="kill",
        cancellation_cleanup=cancellation_cleanup,
        timeout_cleanup=timeout_cleanup,
    ) as runner:
        print(f"sandbox_id {runner.sandbox_id}")
        print("sandbox ready")

        pwd = await runner.exec(ExecCommand.process("pwd"))
        print(f"pwd {pwd.stdout.strip()}")

        host_secret_name = "CAYU_HOST_SECRET_SHOULD_NOT_LEAK"
        os.environ[host_secret_name] = "hidden"
        env_check = await runner.exec(
            ExecCommand.bash(
                f'if [ -n "${host_secret_name}" ]; then echo visible; else echo hidden; fi'
            )
        )
        print(f"host_secret {env_check.stdout.strip()}")

        explicit_env = await runner.exec(
            ExecCommand.bash('printf "%s" "$CAYU_EXPLICIT_ENV"'),
            env={"CAYU_EXPLICIT_ENV": "visible"},
        )
        print(f"explicit_env {explicit_env.stdout.strip()}")

        bounded = await runner.exec(
            ExecCommand.bash("printf abcdef; printf uvwxyz >&2"),
            output_limit_bytes=3,
        )
        print(
            "bounded "
            f"stdout={bounded.stdout!r} "
            f"stdout_truncated={bounded.stdout_truncated} "
            f"stderr={bounded.stderr!r} "
            f"stderr_truncated={bounded.stderr_truncated}"
        )

        cancelled = asyncio.create_task(runner.exec(ExecCommand.bash("sleep 30")))
        await asyncio.sleep(cancel_delay_s)
        cancelled.cancel()
        try:
            await cancelled
            print("cancelled missing")
        except RunnerCancelledError as exc:
            print(f"cancelled true artifacts={exc.artifacts}")

        try:
            after_cancel = await runner.exec(ExecCommand.bash("printf after-cancel"))
            print(f"after_cancel reusable stdout={after_cancel.stdout!r}")
        except RuntimeError as exc:
            print(f"after_cancel closed: {exc}")

        print("closing sandbox")
    print("completed")


if __name__ == "__main__":
    asyncio.run(main())
