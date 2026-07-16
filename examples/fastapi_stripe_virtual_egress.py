"""FastAPI app flow under virtual egress.

Runs a small FastAPI app inside an explicitly selected Docker container created
by ``VirtualEgressEnvironmentFactory``. The app receives a virtual
``STRIPE_SECRET_KEY``; the broker resolves and injects the real vault secret
only for authorized outbound requests.

    python examples/fastapi_stripe_virtual_egress.py  # needs Docker + cayu[egress]

The upstream is fake, so no Stripe account or real Stripe key is required.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import subprocess
from typing import Any

from cayu import (
    EnvironmentFactoryRequest,
    Event,
    EventType,
    HttpEgressPolicy,
    SecretRef,
    StaticVault,
    VirtualCredentialSpec,
    VirtualEgressEnvironmentFactory,
)
from cayu.runners.base import ExecCommand, ExecResult, Runner

DEMO_REAL_SECRET = "sk_test_51DemoRealFastApiKeyHeldOnlyByBroker"
IMAGE = "cayu-egress-fastapi-stripe:demo"
APP_PATH = "/tmp/cayu_fastapi_app.py"
POLICY_NAME = "stripe-example"

_DOCKERFILE = b"""FROM python:3.12-slim
RUN pip install --no-cache-dir fastapi uvicorn
"""

FASTAPI_APP = r"""
import os
import urllib.parse
import urllib.request

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class CustomerRequest(BaseModel):
    email: str


@app.post("/customers")
def create_customer(request: CustomerRequest):
    body = urllib.parse.urlencode({"email": request.email}).encode()
    stripe_request = urllib.request.Request(
        "https://api.stripe.com/v1/customers",
        data=body,
        headers={
            "Authorization": "Bearer " + os.environ["STRIPE_SECRET_KEY"],
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(stripe_request, timeout=25) as response:
        payload = response.read().decode()
    return {"stripe_response": payload}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
"""

_DIRECT_EGRESS = (
    'python3 -c "import socket; socket.setdefaulttimeout(6); '
    "socket.create_connection(('1.1.1.1', 443))\" 2>&1; echo EXIT=$?"
)


class _FakeStripe:
    def __init__(self) -> None:
        self.saw_authorization: str | None = None

    async def send(self, request: Any) -> Any:
        from cayu.egress import CapturedResponse

        self.saw_authorization = request.headers.get("Authorization")
        return CapturedResponse(
            status_code=200,
            headers={"Request-Id": "req_demo", "Content-Type": "application/json"},
            body=b'{"id":"cus_demo123","object":"customer","email":"user@example.com"}',
        )


def _docker_running() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _ensure_fastapi_image() -> None:
    exists = (
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    )
    if exists:
        return
    print("building FastAPI image (one-time, needs internet)...")
    built = subprocess.run(
        ["docker", "build", "-t", IMAGE, "-"], input=_DOCKERFILE, capture_output=True
    )
    if built.returncode != 0:
        raise RuntimeError(f"docker build failed: {built.stderr.decode('utf-8', 'replace')[:400]}")


async def main() -> None:
    if not _docker_running():
        print("Docker daemon is not available. Start Docker and retry.")
        return
    _ensure_fastapi_image()

    upstream = _FakeStripe()
    events: list[Event] = []

    async def emit(event: Event) -> Event:
        events.append(event)
        return event

    factory = VirtualEgressEnvironmentFactory(
        resolver=StaticVault({"stripe_test_key": DEMO_REAL_SECRET}),
        policies={
            POLICY_NAME: HttpEgressPolicy(
                name=POLICY_NAME,
                allowed_hosts=["api.stripe.com"],
                allowed_endpoints=[("POST", "/v1/customers")],
            )
        },
        credentials=[
            VirtualCredentialSpec(
                env_name="STRIPE_SECRET_KEY",
                secret=SecretRef(name="stripe_test_key"),
                destination="api.stripe.com",
                policy_name=POLICY_NAME,
            )
        ],
        runner_kind="docker",
        image=IMAGE,
        event_emitter=emit,
        upstream=upstream,
    )
    request = EnvironmentFactoryRequest(
        session_id="fastapi-demo",
        agent_name="demo-agent",
        environment_name="billing",
    )
    result = await factory.create(request)
    runner = result.environment.runner
    binding = result.environment.binding
    if runner is None or binding is None:
        raise RuntimeError("virtual egress factory did not return a runner and binding")

    bound = await binding.bind(
        None,
        runner,
        session_id=request.session_id,
        agent_name=request.agent_name,
        environment_name=request.environment_name,
    )
    outcome = "failed"
    try:
        await _install_app(runner)

        env_result = await runner.exec(ExecCommand.bash("env | grep -E 'STRIPE|PROXY' | sort"))
        print("\ncontainer env (STRIPE/PROXY):")
        print(_indent(env_result.stdout))
        print("real secret present in container env:", DEMO_REAL_SECRET in env_result.stdout)

        server_result = await runner.exec(
            ExecCommand.bash(
                f"python3 {APP_PATH} >/tmp/cayu-fastapi.log 2>&1 & echo $! >/tmp/cayu-fastapi.pid"
            ),
            timeout_s=10,
        )
        if server_result.exit_code != 0:
            raise RuntimeError(server_result.stderr or server_result.stdout)
        await _wait_for_app(runner)

        app_result = await _call_app(runner)
        print("\nFastAPI /customers response:")
        print(_indent(app_result.stdout or app_result.stderr))
        print("broker injected upstream Authorization:", upstream.saw_authorization)
        print("real secret leaked to container output:", DEMO_REAL_SECRET in app_result.stdout)

        direct = await runner.exec(ExecCommand.bash(_DIRECT_EGRESS), timeout_s=30)
        print("\ndirect egress attempt (should be BLOCKED):")
        print(_indent(direct.stdout))
        outcome = "completed"
    finally:
        with contextlib.suppress(Exception):
            await _stop_app(runner)
        await binding.finalize(bound, outcome=outcome)

    revoked = any(event.type == EventType.EGRESS_GRANT_REVOKED for event in events)
    print("\nSession finalized; grant revoked event emitted:", revoked)


async def _install_app(runner: Runner) -> None:
    script = f"open({APP_PATH!r}, 'w').write({FASTAPI_APP!r})"
    result = await runner.exec(
        ExecCommand.process("python3", "-c", script),
        timeout_s=10,
    )
    if result.exit_code != 0:
        raise RuntimeError(result.stderr or result.stdout)


async def _wait_for_app(runner: Runner) -> None:
    probe = (
        "import time, urllib.request\n"
        "last = None\n"
        "for _ in range(50):\n"
        "    try:\n"
        "        urllib.request.urlopen('http://127.0.0.1:8000/docs', timeout=1).read()\n"
        "        raise SystemExit(0)\n"
        "    except Exception as exc:\n"
        "        last = exc\n"
        "        time.sleep(0.2)\n"
        "raise SystemExit('FastAPI app did not start: ' + repr(last))\n"
    )
    result = await runner.exec(
        ExecCommand.process("python3", "-c", probe),
        env={"NO_PROXY": "127.0.0.1,localhost"},
        timeout_s=15,
    )
    if result.exit_code != 0:
        logs = await runner.exec(ExecCommand.bash("cat /tmp/cayu-fastapi.log 2>/dev/null || true"))
        raise RuntimeError((result.stderr or result.stdout) + "\n" + logs.stdout)


async def _call_app(runner: Runner) -> ExecResult:
    payload = json.dumps({"email": "user@example.com"}).encode()
    script = (
        "import json, urllib.request\n"
        f"payload = {payload!r}\n"
        "req = urllib.request.Request(\n"
        "    'http://127.0.0.1:8000/customers',\n"
        "    data=payload,\n"
        "    headers={'Content-Type': 'application/json'},\n"
        "    method='POST',\n"
        ")\n"
        "print(urllib.request.urlopen(req, timeout=35).read().decode())\n"
    )
    return await runner.exec(
        ExecCommand.process("python3", "-c", script),
        env={"NO_PROXY": "127.0.0.1,localhost"},
        timeout_s=60,
    )


async def _stop_app(runner: Runner) -> None:
    await runner.exec(
        ExecCommand.bash("test -f /tmp/cayu-fastapi.pid && kill $(cat /tmp/cayu-fastapi.pid)"),
        timeout_s=10,
    )


def _indent(text: str) -> str:
    return "\n".join(f"    {line}" for line in text.strip().splitlines()) or "    (empty)"


if __name__ == "__main__":
    asyncio.run(main())
