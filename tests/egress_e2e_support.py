from __future__ import annotations

import base64
import json
import os
from collections.abc import Sequence
from time import monotonic
from typing import Any, Literal
from uuid import uuid4

from cayu.egress import (
    CapturedRequest,
    CapturedResponse,
    EgressBinding,
    HttpEgressPolicy,
    SandboxEgressAdapter,
    TransparentEgressBroker,
    VirtualCredentialError,
    VirtualCredentialGrant,
    VirtualEgressRunnerRequest,
)
from cayu.environments import EnvironmentFactoryRequest
from cayu.runners.base import ExecCommand, Runner
from cayu.runtime.egress import VirtualCredentialSpec, VirtualEgressEnvironmentFactory
from cayu.vaults import SecretRef, StaticVault
from tests.egress_conformance import (
    EgressConformanceRegistration,
    EgressScenarioEvidence,
)

_MASK = (1 << 64) - 1
_BASE = 257
_BLOOM_BITS = 1 << 20
_MISSING = object()


class RecordingProviderUpstream:
    def __init__(self, response_id: str) -> None:
        self._response_id = response_id
        self.authorization: str | None = None

    async def send(self, request: CapturedRequest) -> CapturedResponse:
        self.authorization = request.headers.get("Authorization")
        return CapturedResponse(
            status_code=200,
            body=json.dumps({"id": self._response_id}).encode(),
        )


class CapturingEgressAdapter(SandboxEgressAdapter):
    """Typed test harness that observes lifecycle objects without changing enforcement."""

    def __init__(self, inner: SandboxEgressAdapter) -> None:
        self._inner = inner
        self.runner_kind = inner.runner_kind
        self._broker: TransparentEgressBroker | None = None
        self._grants: tuple[VirtualCredentialGrant, ...] = ()

    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> EgressBinding:
        binding = await self._inner.prepare(
            session_id=session_id,
            grants=grants,
            broker=broker,
        )
        self._broker = broker
        self._grants = tuple(grants)
        return binding

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        return await self._inner.create_runner(request)

    def captured_single_grant(
        self,
    ) -> tuple[TransparentEgressBroker, VirtualCredentialGrant]:
        if self._broker is None or len(self._grants) != 1:
            raise AssertionError("Egress conformance did not capture exactly one prepared grant.")
        return self._broker, self._grants[0]


async def drive_adversarial_egress_contract(
    *,
    registration: EgressConformanceRegistration,
    adapter: CapturingEgressAdapter,
    real_secret: str,
    image: str,
    search_roots: tuple[str, ...],
    response_id: str,
) -> tuple[EgressScenarioEvidence, ...]:
    """Exercise the shared non-possession and network-denial runtime contract."""

    started_at = monotonic()
    upstream = RecordingProviderUpstream(response_id)
    factory = VirtualEgressEnvironmentFactory(
        resolver=StaticVault({"stripe": real_secret}),
        policies={
            "stripe": HttpEgressPolicy(
                name="stripe",
                allowed_hosts=["api.stripe.com"],
                allowed_endpoints=[("POST", "/v1/customers")],
            )
        },
        credentials=[
            VirtualCredentialSpec(
                env_name="STRIPE_SECRET_KEY",
                secret=SecretRef(name="stripe"),
                destination="api.stripe.com",
                policy_name="stripe",
            )
        ],
        image=image,
        adapter=adapter,
        upstream=upstream,
    )
    session_id = f"{registration.name}-egress-{uuid4().hex[:12]}"
    host_env_name = "CAYU_EGRESS_HOST_ONLY_SENTINEL"
    host_env_value = f"host-only-{uuid4().hex}"
    previous_host_value = os.environ.get(host_env_name, _MISSING)
    os.environ[host_env_name] = host_env_value
    try:
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id=session_id,
                agent_name="e2e",
                environment_name=f"{registration.name}-egress",
            )
        )
    finally:
        if previous_host_value is _MISSING:
            os.environ.pop(host_env_name, None)
        else:
            os.environ[host_env_name] = str(previous_host_value)
    runner = result.environment.runner
    binding = result.environment.binding
    assert runner is not None
    assert binding is not None
    observed: dict[str, Any] = {}
    try:
        env_result = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                "import json,os; print(json.dumps({k:os.environ.get(k) for k in "
                "['STRIPE_SECRET_KEY','HTTPS_PROXY','SSL_CERT_FILE']}))",
            )
        )
        assert env_result.exit_code == 0, env_result.stderr
        observed["env"] = json.loads(env_result.stdout)

        proc_result = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                "from pathlib import Path; print(Path('/proc/self/environ').read_bytes().hex())",
            )
        )
        assert proc_result.exit_code == 0, proc_result.stderr
        observed["proc"] = bytes.fromhex(proc_result.stdout.strip()).decode(errors="replace")

        search_result = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                recursive_window_bloom_script(
                    roots=search_roots,
                    window_size=len(real_secret.encode()),
                ),
            ),
            timeout_s=60,
        )
        assert search_result.exit_code == 0, search_result.stderr
        observed["search"] = json.loads(search_result.stdout)

        host_search_result = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                recursive_window_bloom_script(
                    roots=search_roots,
                    window_size=len(host_env_value.encode()),
                ),
            ),
            timeout_s=60,
        )
        assert host_search_result.exit_code == 0, host_search_result.stderr
        observed["host_search"] = json.loads(host_search_result.stdout)
        observed["host_env_name"] = host_env_name
        observed["host_env_absent"] = not bloom_maybe_contains(
            observed["host_search"]["bloom"],
            host_env_value.encode(),
        )

        call_result = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                "import os,urllib.request as u\n"
                "r=u.Request('https://api.stripe.com/v1/customers',data=b'x=1',"
                "headers={'Authorization':'Bearer '+os.environ['STRIPE_SECRET_KEY']})\n"
                "print(u.urlopen(r,timeout=20).read().decode())\n",
            ),
            timeout_s=30,
        )
        assert call_result.exit_code == 0, call_result.stderr
        observed["call"] = call_result.stdout

        direct = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                connect_probe_script("1.1.1.1", 443, probe_kind="tls"),
            ),
            timeout_s=15,
        )
        metadata = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                connect_probe_script(
                    "169.254.169.254",
                    80,
                    probe_kind="metadata",
                ),
            ),
            timeout_s=15,
        )
        observed["direct"] = json.loads(direct.stdout)
        observed["metadata"] = json.loads(metadata.stdout)

        bound = await binding.bind(None, runner, session_id=session_id)
        await binding.finalize(bound, outcome="completed")
    except BaseException:
        await runner.close()
        raise

    broker, grant = adapter.captured_single_grant()
    observed["upstream_authorization"] = upstream.authorization
    observed["virtual"] = grant.presented_value
    try:
        broker.registry.lookup(grant.presented_value)
    except VirtualCredentialError:
        pass
    else:
        raise AssertionError("Virtual credential remained valid after teardown.")
    observed["revoked"] = True
    assert_guest_non_possession(observed, real_secret)
    assert_brokered_provider_call(
        observed,
        real_secret=real_secret,
        response_id=response_id,
    )
    assert_direct_egress_blocked(observed)
    duration_ms = min(round((monotonic() - started_at) * 1000), 600_000)
    return (
        EgressScenarioEvidence(
            adapter=registration.name,
            scenario="brokered-provider-and-session-ca",
            status="verified",
            proof_source="live",
            observations=("brokered-call-succeeded", "session-ca-trusted"),
            cleanup_outcome="complete",
            duration_ms=duration_ms,
            reason="contract-satisfied",
        ),
        EgressScenarioEvidence(
            adapter=registration.name,
            scenario="guest-network-bypass-denial",
            status="verified",
            proof_source="live",
            observations=("public-ip-denied", "metadata-service-denied"),
            cleanup_outcome="complete",
            duration_ms=duration_ms,
            reason="contract-satisfied",
        ),
        EgressScenarioEvidence(
            adapter=registration.name,
            scenario="guest-secret-non-possession",
            status="verified",
            proof_source="live",
            observations=(
                "virtual-credential-only",
                "raw-secret-absent",
                "broker-secret-absent",
                "enforced-proxy-only",
                "trusted-host-env-absent",
            ),
            cleanup_outcome="complete-and-grant-revoked",
            duration_ms=duration_ms,
            reason="contract-satisfied",
        ),
    )


def assert_guest_non_possession(observed: dict[str, Any], real_secret: str) -> None:
    assert observed["env"]["STRIPE_SECRET_KEY"] == observed["virtual"]
    assert real_secret not in str(observed["env"])
    assert real_secret not in observed["proc"]
    assert observed["search"]["files_scanned"] > 0
    assert not bloom_maybe_contains(observed["search"]["bloom"], real_secret.encode())
    assert observed["host_search"]["files_scanned"] > 0
    assert observed["host_env_absent"] is True
    assert observed["host_env_name"] not in str(observed["env"])
    proxy_url = str(observed["env"]["HTTPS_PROXY"])
    assert "0.0.0.0" not in proxy_url
    assert "127.0.0.1" not in proxy_url


def assert_brokered_provider_call(
    observed: dict[str, Any],
    *,
    real_secret: str,
    response_id: str,
) -> None:
    assert response_id in observed["call"]
    assert observed["upstream_authorization"] == f"Bearer {real_secret}"
    assert real_secret not in observed["call"]


def assert_direct_egress_blocked(observed: dict[str, Any]) -> None:
    assert observed["direct"]["tcp_connected"] is False
    assert observed["metadata"]["tcp_connected"] is False


def connect_probe_script(
    host: str,
    port: int,
    *,
    probe_kind: Literal["tls", "metadata"] | None = None,
) -> str:
    kind = ("tls" if port == 443 else "metadata") if probe_kind is None else probe_kind
    if kind not in {"tls", "metadata"}:  # pragma: no cover - guarded by the type contract
        raise ValueError(f"Unsupported connection probe kind: {kind}")
    if kind == "tls":
        return (
            "import json,socket,ssl\n"
            "socket.setdefaulttimeout(3)\n"
            "result={'tcp_connected':False,'tls_completed':False}\n"
            "sock=None\n"
            "try:\n"
            f" sock=socket.create_connection(({host!r},{port}))\n"
            " result['tcp_connected']=True\n"
            " context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)\n"
            " context.check_hostname=False\n"
            " context.verify_mode=ssl.CERT_NONE\n"
            " tls_socket=context.wrap_socket(sock)\n"
            " result['tls_completed']=True\n"
            " tls_socket.close()\n"
            "except Exception as exc:\n"
            " result['error']=repr(exc)\n"
            "finally:\n"
            " if sock is not None:\n"
            "  try: sock.close()\n"
            "  except Exception: pass\n"
            "print(json.dumps(result))\n"
        )
    return (
        "import json,socket\n"
        "socket.setdefaulttimeout(3)\n"
        "def probe(request):\n"
        " result={'tcp_connected':False,'http_status':None}\n"
        " sock=None\n"
        " try:\n"
        f"  sock=socket.create_connection(({host!r},{port}))\n"
        "  result['tcp_connected']=True\n"
        "  sock.sendall(request)\n"
        "  head=b''\n"
        "  while b'\\r\\n\\r\\n' not in head and len(head)<8192:\n"
        "   chunk=sock.recv(1024)\n"
        "   if not chunk: break\n"
        "   head+=chunk\n"
        "  result['http_status']=int(head.split(b' ',2)[1])\n"
        " except Exception as exc:\n"
        "  result['error']=repr(exc)\n"
        " finally:\n"
        "  if sock is not None:\n"
        "   try: sock.close()\n"
        "   except Exception: pass\n"
        " return result\n"
        "get=b'GET /latest/meta-data/ HTTP/1.1\\r\\nHost: 169.254.169.254\\r\\nConnection: close\\r\\n\\r\\n'\n"
        "token=b'PUT /latest/api/token HTTP/1.1\\r\\nHost: 169.254.169.254\\r\\nX-aws-ec2-metadata-token-ttl-seconds: 60\\r\\nConnection: close\\r\\n\\r\\n'\n"
        "attempts=[probe(get)]\n"
        "if not (isinstance(attempts[0]['http_status'],int) and 200<=attempts[0]['http_status']<300):\n"
        " attempts.append(probe(token))\n"
        "statuses=[item['http_status'] for item in attempts if isinstance(item['http_status'],int)]\n"
        "http_succeeded=any(200<=status<300 for status in statuses)\n"
        "result={'tcp_connected':any(item['tcp_connected'] for item in attempts),'http_statuses':statuses,'http_succeeded':http_succeeded}\n"
        "if not http_succeeded: result['error']='metadata access denied'\n"
        "print(json.dumps(result))\n"
    )


def recursive_window_bloom_script(*, roots: tuple[str, ...], window_size: int) -> str:
    """Build a guest script that summarizes every file window without a search needle.

    The guest receives only the real secret's byte length. It never receives the
    value or an encoding of it. The host compares the returned Bloom filter with
    the real value after the sandbox has closed. Bloom collisions can cause a
    conservative test failure, but an actual matching window cannot be missed.
    """

    return f"""
import base64
import collections
import json
import os

roots = {roots!r}
window_size = {window_size}
mask = (1 << 64) - 1
base = 257
high = pow(base, window_size - 1, 1 << 64)
bloom_bits = 1 << 20
bloom = bytearray(bloom_bits // 8)
files_scanned = 0

def add(value):
    for shift in (0, 16, 32, 48):
        index = (value >> shift) & (bloom_bits - 1)
        bloom[index >> 3] |= 1 << (index & 7)

def scan(path):
    global files_scanned
    window = collections.deque()
    value = 0
    try:
        handle = open(path, 'rb')
    except Exception:
        return
    files_scanned += 1
    try:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            for byte in chunk:
                if len(window) < window_size:
                    window.append(byte)
                    value = ((value * base) + byte) & mask
                    if len(window) == window_size:
                        add(value)
                    continue
                oldest = window.popleft()
                window.append(byte)
                value = (((value - oldest * high) * base) + byte) & mask
                add(value)
    except Exception:
        pass
    finally:
        handle.close()

for root in roots:
    if os.path.isfile(root):
        scan(root)
    elif os.path.isdir(root):
        for base_path, _directories, files in os.walk(root):
            for name in files:
                scan(os.path.join(base_path, name))

print(json.dumps({{
    'bloom': base64.b64encode(bloom).decode(),
    'files_scanned': files_scanned,
}}))
""".strip()


def bloom_maybe_contains(encoded_bloom: str, value: bytes) -> bool:
    bloom = base64.b64decode(encoded_bloom)
    rolling = 0
    for byte in value:
        rolling = ((rolling * _BASE) + byte) & _MASK
    for shift in (0, 16, 32, 48):
        index = (rolling >> shift) & (_BLOOM_BITS - 1)
        if not bloom[index >> 3] & (1 << (index & 7)):
            return False
    return True
