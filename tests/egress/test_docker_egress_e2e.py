"""Adversarial end-to-end tests for Docker virtual egress.

These prove *non-possession*: the sandbox never receives the real secret, cannot
reach the provider directly, and the credential dies with the session. They spin
real containers, so they are gated on a responsive Docker daemon.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile

import pytest

from cayu.egress import (
    CapturedRequest,
    CapturedResponse,
    HttpEgressPolicy,
    TransparentEgressBroker,
    VirtualCredentialError,
    VirtualCredentialRegistry,
)
from cayu.runners.base import ExecCommand
from cayu.runners.docker import DockerRunner
from cayu.vaults import SecretRef, StaticVault

pytest.importorskip("cryptography")

from cayu.egress.docker_adapter import DockerEgressAdapter

REAL_SECRET = "sk_test_51E2ERealSecretNeverInSandbox"
SANDBOX_IMAGE = "python:3.12-slim"


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not available for virtual-egress E2E.",
)


class _FakeStripe:
    """Stands in for api.stripe.com so the E2E needs no real Stripe key."""

    def __init__(self) -> None:
        self.upstream_authorization: str | None = None

    async def send(self, request: CapturedRequest) -> CapturedResponse:
        self.upstream_authorization = request.headers.get("Authorization")
        return CapturedResponse(
            status_code=200,
            headers={"Request-Id": "req_e2e"},
            body=b'{"id":"cus_fake123","object":"customer"}',
        )


def _stripe_example_policy() -> HttpEgressPolicy:
    return HttpEgressPolicy(
        name="stripe-example",
        allowed_hosts=["api.stripe.com"],
        allowed_endpoints=[("POST", "/v1/customers")],
    )


def _search_needles_script(*needles: str) -> str:
    roots = ["/workspace", "/tmp", "/etc", "/root", "/proc/self/environ"]
    needle_hex = [needle.encode().hex() for needle in needles]
    return (
        "import json, os\n"
        f"needles = [bytes.fromhex(value) for value in {needle_hex!r}]\n"
        f"roots = {roots!r}\n"
        "found = {needle.hex(): [] for needle in needles}\n"
        "def check(path):\n"
        "    try:\n"
        "        with open(path, 'rb') as handle:\n"
        "            data = handle.read()\n"
        "    except Exception:\n"
        "        return\n"
        "    for needle in needles:\n"
        "        if needle in data:\n"
        "            found[needle.hex()].append(path)\n"
        "for root in roots:\n"
        "    if os.path.isfile(root):\n"
        "        check(root)\n"
        "    elif os.path.isdir(root):\n"
        "        for dirpath, _dirnames, filenames in os.walk(root):\n"
        "            for filename in filenames:\n"
        "                check(os.path.join(dirpath, filename))\n"
        "print(json.dumps({'found': found}))\n"
    )


def _connect_probe_script(host: str, port: int) -> str:
    return (
        "import json, socket\n"
        "socket.setdefaulttimeout(6)\n"
        "try:\n"
        f"    sock = socket.create_connection(({host!r}, {port!r}))\n"
        "    sock.close()\n"
        "    result = {'connected': True, 'error': None}\n"
        "except Exception as exc:\n"
        "    result = {'connected': False, 'error': repr(exc)}\n"
        "print(json.dumps(result))\n"
    )


def _json_stdout(stdout: str) -> dict[str, object]:
    return json.loads(stdout.strip().splitlines()[-1])


async def _drive() -> dict[str, object]:
    loop = asyncio.get_running_loop()
    upstream = _FakeStripe()
    registry = VirtualCredentialRegistry()
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"stripe_test_key": REAL_SECRET}),
        policies={"stripe-example": _stripe_example_policy()},
        upstream=upstream,
    )
    grant = registry.mint(
        session_id="e2e",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
        policy_name="stripe-example",
    )

    adapter = DockerEgressAdapter(loop=loop)
    binding = await adapter.prepare(session_id="e2e", grants=[grant], broker=broker)

    ca_dir = tempfile.mkdtemp(prefix="cayu-egress-e2e-ca-")
    ca_host = os.path.join(ca_dir, "ca.pem")
    with open(ca_host, "wb") as handle:
        handle.write(binding.ca_cert_pem or b"")

    sandbox_env = {"STRIPE_SECRET_KEY": grant.presented_value}
    results: dict[str, object] = {"virtual": grant.presented_value}
    runner = None
    try:
        runner = await DockerRunner.create(
            "cayu-egress-e2e-sandbox",
            image=SANDBOX_IMAGE,
            close_action="remove",
            network=str(binding.network),
            env_overlay=binding.env,
            ca_mount=(ca_host, str(binding.guest_ca_path)),
        )

        env_result = await runner.exec(ExecCommand.process("env"), env=sandbox_env)
        results["env_stdout"] = env_result.stdout

        proc_result = await runner.exec(
            ExecCommand.bash("cat /proc/self/environ | tr '\\0' '\\n'"),
            env=sandbox_env,
        )
        results["proc_stdout"] = proc_result.stdout

        search_result = await runner.exec(
            ExecCommand.process("python3", "-c", _search_needles_script(REAL_SECRET)),
            env=sandbox_env,
        )
        results["secret_search"] = _json_stdout(search_result.stdout)

        call_script = (
            "import os, urllib.request as u\n"
            "req = u.Request('https://api.stripe.com/v1/customers', data=b'email=a@b.co',\n"
            "    headers={'Authorization': 'Bearer ' + os.environ['STRIPE_SECRET_KEY']})\n"
            "print(u.urlopen(req, timeout=25).read().decode())\n"
        )
        call_result = await runner.exec(
            ExecCommand.process("python3", "-c", call_script),
            env=sandbox_env,
            timeout_s=60,
        )
        results["call_stdout"] = call_result.stdout
        results["call_stderr"] = call_result.stderr
        results["call_exit_code"] = call_result.exit_code
        results["upstream_authorization"] = upstream.upstream_authorization

        direct_result = await runner.exec(
            ExecCommand.process("python3", "-c", _connect_probe_script("1.1.1.1", 443)),
            env=sandbox_env,
            timeout_s=30,
        )
        results["direct_probe"] = _json_stdout(direct_result.stdout)

        # Cloud metadata endpoint must be unreachable (no route on the internal net).
        metadata_result = await runner.exec(
            ExecCommand.process("python3", "-c", _connect_probe_script("169.254.169.254", 80)),
            env=sandbox_env,
            timeout_s=30,
        )
        results["metadata_probe"] = _json_stdout(metadata_result.stdout)

        # Exfil-by-encoding is impossible because the real secret is absent: its
        # base64/hex encodings cannot be found anywhere the sandbox can read.
        import base64 as _b64

        b64 = _b64.b64encode(REAL_SECRET.encode()).decode()
        hexed = REAL_SECRET.encode().hex()
        encoded_result = await runner.exec(
            ExecCommand.process("python3", "-c", _search_needles_script(b64, hexed)),
            env=sandbox_env,
            timeout_s=30,
        )
        results["encoded_search"] = _json_stdout(encoded_result.stdout)
        results["real_b64"] = b64
        results["real_hex"] = hexed
    finally:
        if runner is not None:
            await runner.close()
        await binding.close()
        shutil.rmtree(ca_dir, ignore_errors=True)

    # Session is closed: the virtual credential must now be rejected.
    try:
        registry.lookup(grant.presented_value)
        results["revoked"] = False
    except VirtualCredentialError:
        results["revoked"] = True

    return results


@pytest.fixture(scope="module")
def e2e_results() -> dict[str, object]:
    return asyncio.run(_drive())


def test_env_shows_only_virtual_credential(e2e_results: dict[str, object]) -> None:
    env_stdout = str(e2e_results["env_stdout"])
    assert str(e2e_results["virtual"]) in env_stdout
    assert REAL_SECRET not in env_stdout
    assert "HTTPS_PROXY=" in env_stdout


def test_proc_environ_shows_only_virtual(e2e_results: dict[str, object]) -> None:
    proc_stdout = str(e2e_results["proc_stdout"])
    assert str(e2e_results["virtual"]) in proc_stdout
    assert REAL_SECRET not in proc_stdout


def test_recursive_search_finds_no_real_secret(e2e_results: dict[str, object]) -> None:
    search = e2e_results["secret_search"]
    assert isinstance(search, dict)
    assert search["found"] == {REAL_SECRET.encode().hex(): []}


def test_allowed_call_succeeds_with_swapped_credential(e2e_results: dict[str, object]) -> None:
    call_stdout = str(e2e_results["call_stdout"])
    assert e2e_results["call_exit_code"] == 0
    assert "cus_fake123" in call_stdout  # provider (fake) accepted the request
    assert REAL_SECRET not in call_stdout
    # The broker injected the REAL secret only on the upstream leg.
    assert e2e_results["upstream_authorization"] == f"Bearer {REAL_SECRET}"


def test_direct_egress_is_blocked(e2e_results: dict[str, object]) -> None:
    probe = e2e_results["direct_probe"]
    assert isinstance(probe, dict)
    assert probe["connected"] is False


def test_cloud_metadata_endpoint_is_blocked(e2e_results: dict[str, object]) -> None:
    probe = e2e_results["metadata_probe"]
    assert isinstance(probe, dict)
    assert probe["connected"] is False


def test_encoded_secret_cannot_be_exfiltrated(e2e_results: dict[str, object]) -> None:
    # The real secret is absent, so its base64/hex forms are absent too.
    search = e2e_results["encoded_search"]
    assert isinstance(search, dict)
    assert search["found"] == {
        str(e2e_results["real_b64"]).encode().hex(): [],
        str(e2e_results["real_hex"]).encode().hex(): [],
    }


def test_credential_rejected_after_session_close(e2e_results: dict[str, object]) -> None:
    assert e2e_results["revoked"] is True


# --- Full runtime wiring: the VirtualEgressEnvironmentFactory lifecycle. ---


async def _drive_factory() -> dict[str, object]:
    from cayu.core.events import Event, EventType
    from cayu.environments import EnvironmentFactoryRequest
    from cayu.runtime.egress import VirtualCredentialSpec, VirtualEgressEnvironmentFactory

    events: list[Event] = []
    authorized_seen = asyncio.Event()

    async def emitter(event: Event) -> Event:
        events.append(event)
        if event.type == EventType.EGRESS_REQUEST_AUTHORIZED:
            authorized_seen.set()
        return event

    upstream = _FakeStripe()
    factory = VirtualEgressEnvironmentFactory(
        resolver=StaticVault({"stripe_test_key": REAL_SECRET}),
        policies={"stripe-example": _stripe_example_policy()},
        credentials=[
            VirtualCredentialSpec(
                env_name="STRIPE_SECRET_KEY",
                secret=SecretRef(name="stripe_test_key"),
                destination="api.stripe.com",
                policy_name="stripe-example",
            )
        ],
        image=SANDBOX_IMAGE,
        event_emitter=emitter,
        upstream=upstream,
    )
    request = EnvironmentFactoryRequest(
        session_id="e2e-factory", agent_name="agent", environment_name="egress-env"
    )
    result = await factory.create(request)
    runner = result.environment.runner
    assert runner is not None

    call_script = (
        "import os, urllib.request as u\n"
        "req = u.Request('https://api.stripe.com/v1/customers', data=b'email=a@b.co',\n"
        "    headers={'Authorization': 'Bearer ' + os.environ['STRIPE_SECRET_KEY']})\n"
        "print(u.urlopen(req, timeout=25).read().decode())\n"
    )
    try:
        call = await runner.exec(ExecCommand.process("python3", "-c", call_script), timeout_s=60)
        await asyncio.wait_for(authorized_seen.wait(), timeout=2)
    finally:
        bound = await result.environment.binding.bind(None, runner, session_id="e2e-factory")
        await result.environment.binding.finalize(bound, outcome="completed")

    return {
        "call_stdout": call.stdout,
        "upstream_authorization": upstream.upstream_authorization,
        "event_types": [str(e.type) for e in events],
        "event_payloads": " ".join(str(e.payload) for e in events),
        "authorized": EventType.EGRESS_REQUEST_AUTHORIZED,
        "minted": EventType.EGRESS_GRANT_MINTED,
        "revoked": EventType.EGRESS_GRANT_REVOKED,
    }


@pytest.fixture(scope="module")
def factory_results() -> dict[str, object]:
    return asyncio.run(_drive_factory())


def test_factory_allowed_call_succeeds_with_swap(factory_results: dict[str, object]) -> None:
    assert "cus_fake123" in str(factory_results["call_stdout"])
    assert REAL_SECRET not in str(factory_results["call_stdout"])
    assert factory_results["upstream_authorization"] == f"Bearer {REAL_SECRET}"


def test_factory_emits_audit_events_without_secret(factory_results: dict[str, object]) -> None:
    types = factory_results["event_types"]
    assert str(factory_results["minted"]) in types
    assert str(factory_results["authorized"]) in types
    assert str(factory_results["revoked"]) in types
    assert REAL_SECRET not in str(factory_results["event_payloads"])
