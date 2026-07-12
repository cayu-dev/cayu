from __future__ import annotations

import base64
import json
from typing import Any
from uuid import uuid4

from cayu.egress import (
    CapturedRequest,
    CapturedResponse,
    HttpEgressPolicy,
    VirtualCredentialError,
)
from cayu.environments import EnvironmentFactoryRequest
from cayu.runners.base import ExecCommand
from cayu.runtime.egress import VirtualCredentialSpec, VirtualEgressEnvironmentFactory
from cayu.vaults import SecretRef, StaticVault

_MASK = (1 << 64) - 1
_BASE = 257
_BLOOM_BITS = 1 << 20


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


async def drive_adversarial_egress_contract(
    *,
    adapter: Any,
    real_secret: str,
    image: str,
    session_prefix: str,
    search_roots: tuple[str, ...],
    response_id: str,
) -> dict[str, Any]:
    """Exercise the shared non-possession and network-denial runtime contract."""

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
    session_id = f"{session_prefix}-{uuid4().hex[:12]}"
    result = await factory.create(
        EnvironmentFactoryRequest(
            session_id=session_id,
            agent_name="e2e",
            environment_name=session_prefix,
        )
    )
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
            ExecCommand.process("python3", "-c", connect_probe_script("1.1.1.1", 443)),
            timeout_s=15,
        )
        metadata = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                connect_probe_script("169.254.169.254", 80),
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

    observed["upstream_authorization"] = upstream.authorization
    observed["virtual"] = adapter.grant.presented_value
    try:
        adapter.broker.registry.lookup(adapter.grant.presented_value)
    except VirtualCredentialError:
        pass
    else:
        raise AssertionError("Virtual credential remained valid after teardown.")
    return observed


def assert_guest_non_possession(observed: dict[str, Any], real_secret: str) -> None:
    assert observed["env"]["STRIPE_SECRET_KEY"] == observed["virtual"]
    assert real_secret not in str(observed["env"])
    assert real_secret not in observed["proc"]
    assert observed["search"]["files_scanned"] > 0
    assert not bloom_maybe_contains(observed["search"]["bloom"], real_secret.encode())


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
    assert observed["direct"]["connected"] is False
    assert observed["metadata"]["connected"] is False


def connect_probe_script(host: str, port: int) -> str:
    if port == 443:
        probe = (
            f"s=socket.create_connection(({host!r},{port}))\n"
            "c=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)\n"
            "c.check_hostname=False\n"
            "c.verify_mode=ssl.CERT_NONE\n"
            "t=c.wrap_socket(s)\n"
            "t.close()\n"
        )
    else:
        probe = (
            "def succeeds(request):\n"
            " try:\n"
            f"  s=socket.create_connection(({host!r},{port}))\n"
            "  s.sendall(request)\n"
            "  head=b''\n"
            "  while b'\\r\\n\\r\\n' not in head and len(head)<8192:\n"
            "   chunk=s.recv(1024)\n"
            "   if not chunk: break\n"
            "   head+=chunk\n"
            "  s.close()\n"
            "  status=int(head.split(b' ',2)[1])\n"
            "  return 200<=status<300\n"
            " except Exception:\n"
            "  return False\n"
            "get=b'GET /latest/meta-data/ HTTP/1.1\\r\\nHost: 169.254.169.254\\r\\nConnection: close\\r\\n\\r\\n'\n"
            "token=b'PUT /latest/api/token HTTP/1.1\\r\\nHost: 169.254.169.254\\r\\nX-aws-ec2-metadata-token-ttl-seconds: 60\\r\\nConnection: close\\r\\n\\r\\n'\n"
            "if not (succeeds(get) or succeeds(token)): raise OSError('metadata access denied')\n"
        )
    return (
        "import json,socket,ssl\n"
        "socket.setdefaulttimeout(3)\n"
        "try:\n"
        + "\n".join(f" {line}" for line in probe.splitlines())
        + "\n result={'connected':True}\n"
        "except Exception as exc:\n"
        " result={'connected':False,'error':repr(exc)}\n"
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
