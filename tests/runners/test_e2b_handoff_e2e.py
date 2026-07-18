"""Opt-in live proof for the irreversible E2B guest handoff contract."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from cayu import E2BGuestProvisioner, E2BRunner, ExecCommand

e2b = pytest.importorskip("e2b")

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_RUN_E2B_HANDOFF_E2E") != "1" or not os.environ.get("E2B_API_KEY"),
    reason="Set CAYU_RUN_E2B_HANDOFF_E2E=1 and E2B_API_KEY.",
)

_PROTECTED_PATH = "/opt/cayu-verification/pristine.txt"
_ROOT_ONLY_NESTED_PATH = "/opt/cayu-private/nested.txt"


async def _listed_sandbox_ids(metadata: dict[str, str]) -> list[str]:
    paginator = e2b.AsyncSandbox.list(
        query=e2b.SandboxQuery(metadata=metadata),
        limit=100,
    )
    sandbox_ids: list[str] = []
    while paginator.has_next:
        sandbox_ids.extend(item.sandbox_id for item in await paginator.next_items())
    return sandbox_ids


async def _wait_for_no_sandboxes(metadata: dict[str, str], *, timeout_s: float) -> list[str]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        sandbox_ids = await _listed_sandbox_ids(metadata)
        if not sandbox_ids:
            return []
        if asyncio.get_running_loop().time() >= deadline:
            return sandbox_ids
        await asyncio.sleep(0.5)


def test_e2b_live_irreversible_guest_handoff_security_contract() -> None:
    asyncio.run(_drive_live_handoff())


async def _drive_live_handoff() -> None:
    run_metadata = {"cayu_e2b_handoff_e2e": uuid4().hex}
    retained: list[E2BGuestProvisioner] = []
    runner: E2BRunner | None = None
    primary_error: BaseException | None = None

    async def bootstrap(provisioner: E2BGuestProvisioner) -> None:
        retained.append(provisioner)
        await provisioner.install_directory("/opt/cayu-verification", mode=0o555)
        await provisioner.install_file(_PROTECTED_PATH, b"trusted-pristine\n", mode=0o444)
        await provisioner.install_directory("/opt/cayu-private", mode=0o700)
        await provisioner.install_file(_ROOT_ONLY_NESTED_PATH, b"root-only\n", mode=0o400)

    try:
        runner = await E2BRunner.create_hardened(
            template=os.environ.get("CAYU_E2B_TEMPLATE"),
            sandbox_timeout_s=int(os.environ.get("CAYU_E2B_SANDBOX_TIMEOUT_S", "300")),
            close_action="kill",
            metadata=run_metadata,
            bootstrap=bootstrap,
        )
        assert retained and retained[0].is_sealed is True
        with pytest.raises(RuntimeError, match="sealed"):
            await retained[0].install_file("/opt/cayu-verification/late.txt", b"late")

        sudo = await runner.exec(ExecCommand.bash("sudo -n true"), timeout_s=10)
        su = await runner.exec(ExecCommand.bash("su -c true root"), timeout_s=10)
        firewall = await runner.exec(
            ExecCommand.bash("/usr/sbin/iptables -D OUTPUT -d 169.254.169.254/32 -j REJECT"),
            timeout_s=10,
        )
        overwrite = await runner.exec(
            ExecCommand.bash(f"printf x >> {_PROTECTED_PATH}"),
            timeout_s=10,
        )
        unlink = await runner.exec(
            ExecCommand.process("rm", "-f", _PROTECTED_PATH),
            timeout_s=10,
        )
        rename = await runner.exec(
            ExecCommand.process("mv", _PROTECTED_PATH, f"{_PROTECTED_PATH}.moved"),
            timeout_s=10,
        )
        atomic_replace = await runner.exec(
            ExecCommand.bash(
                "printf replacement >/tmp/cayu-replacement"
                " && python3 -c "
                '"import os; os.replace('
                f"'/tmp/cayu-replacement', '{_PROTECTED_PATH}'"
                ')"'
            ),
            timeout_s=10,
        )
        public_network = await runner.exec(
            ExecCommand.bash(
                "python3 - <<'PY'\n"
                "import socket\n"
                "import ssl\n"
                "def tls_reachable(host, server_hostname):\n"
                "    raw = None\n"
                "    try:\n"
                "        raw = socket.create_connection((host, 443), timeout=2)\n"
                "        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)\n"
                "        context.check_hostname = False\n"
                "        context.verify_mode = ssl.CERT_NONE\n"
                "        connection = context.wrap_socket(raw, server_hostname=server_hostname)\n"
                "    except OSError:\n"
                "        if raw is not None:\n"
                "            raw.close()\n"
                "        return False\n"
                "    connection.close()\n"
                "    return True\n"
                'for host, name in (("1.1.1.1", "one.one.one.one"), '
                '("8.8.8.8", "dns.google")):\n'
                "    if tls_reachable(host, name):\n"
                "        raise SystemExit(91)\n"
                "raise SystemExit(0)\n"
                "PY"
            ),
            timeout_s=10,
        )
        pristine = await runner.exec(
            ExecCommand.process("cat", _PROTECTED_PATH),
            timeout_s=10,
        )

        assert sudo.exit_code != 0
        assert su.exit_code != 0
        assert firewall.exit_code != 0
        assert overwrite.exit_code != 0
        assert unlink.exit_code != 0
        assert rename.exit_code != 0
        assert atomic_replace.exit_code != 0
        assert public_network.exit_code == 0
        assert pristine.exit_code == 0
        assert pristine.stdout == "trusted-pristine\n"
    except BaseException as exc:
        primary_error = exc

    cleanup_errors: list[BaseException] = []
    if runner is not None:
        try:
            await runner.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
    leaked: list[str] = []
    try:
        leaked = await _wait_for_no_sandboxes(run_metadata, timeout_s=10)
    except BaseException as exc:
        cleanup_errors.append(exc)
    for sandbox_id in leaked:
        try:
            await e2b.AsyncSandbox.kill(sandbox_id)
        except BaseException as exc:
            cleanup_errors.append(exc)
    if leaked:
        cleanup_errors.append(
            AssertionError(f"E2B handoff test leaked running sandboxes: {leaked}")
        )

    failures = ([primary_error] if primary_error is not None else []) + cleanup_errors
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("E2B handoff test and cleanup failed.", failures)
