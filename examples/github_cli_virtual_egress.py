"""Run the unmodified GitHub CLI through Cayu virtual egress.

Builds a checksum-pinned Linux ``gh`` image, then runs ``gh api user`` inside
the explicitly selected Docker egress topology. The runner receives only a
virtual ``GH_TOKEN``. A fake upstream keeps the example credential-free while
proving that the broker injects the real token only after authorization.

    python examples/github_cli_virtual_egress.py  # needs Docker + cayu[egress]

The one-time image build needs internet access. The resulting runner does not.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess

from cayu import (
    EnvironmentFactoryRequest,
    HttpEgressPolicy,
    SecretRef,
    StaticVault,
    VirtualCredentialSpec,
    VirtualEgressEnvironmentFactory,
)
from cayu.egress import CapturedRequest, CapturedResponse
from cayu.runners.base import ExecCommand

GH_VERSION = "2.86.0"
IMAGE = f"cayu-egress-github-cli:{GH_VERSION}"
POLICY_NAME = "github-cli-read"
DEMO_REAL_TOKEN = "github_pat_11CayuDemoRealTokenHeldOnlyByBroker"
_DOCKER_COMMAND_TIMEOUT_SECONDS = 15
_DOCKER_BUILD_TIMEOUT_SECONDS = 300

_DOCKERFILE = f"""FROM alpine:3.20
ARG TARGETARCH
RUN set -eu; \\
    case "$TARGETARCH" in \\
      amd64) GH_SHA256=f3b08bd6a28420cc2229b0a1a687fa25f2b838d3f04b297414c1041ca68103c7 ;; \\
      arm64) GH_SHA256=83cf7a7962ea9dfcc2c123666695792916a87af32cba5f1f6e585db08fa57547 ;; \\
      *) echo "unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \\
    esac; \\
    apk add --no-cache ca-certificates curl; \\
    GH_ARCHIVE=gh_{GH_VERSION}_linux_${{TARGETARCH}}.tar.gz; \\
    curl -fsSLo /tmp/gh.tar.gz \\
      https://github.com/cli/cli/releases/download/v{GH_VERSION}/${{GH_ARCHIVE}}; \\
    echo "$GH_SHA256  /tmp/gh.tar.gz" | sha256sum -c -; \\
    tar -xzf /tmp/gh.tar.gz -C /tmp; \\
    cp /tmp/gh_{GH_VERSION}_linux_${{TARGETARCH}}/bin/gh /usr/local/bin/gh; \\
    chmod +x /usr/local/bin/gh; \\
    rm -rf /tmp/gh.tar.gz /tmp/gh_{GH_VERSION}_linux_${{TARGETARCH}}
""".encode()


class _FakeGitHub:
    def __init__(self) -> None:
        self.request: CapturedRequest | None = None

    async def send(self, request: CapturedRequest) -> CapturedResponse:
        self.request = request
        return CapturedResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b'{"login":"cayu-probe"}',
        )


def _docker_running() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=_DOCKER_COMMAND_TIMEOUT_SECONDS,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _ensure_image() -> None:
    exists = (
        subprocess.run(
            ["docker", "image", "inspect", IMAGE],
            capture_output=True,
            timeout=_DOCKER_COMMAND_TIMEOUT_SECONDS,
        ).returncode
        == 0
    )
    if exists:
        return
    print(f"building GitHub CLI {GH_VERSION} image (one-time, needs internet)...")
    built = subprocess.run(
        ["docker", "build", "-t", IMAGE, "-"],
        input=_DOCKERFILE,
        capture_output=True,
        timeout=_DOCKER_BUILD_TIMEOUT_SECONDS,
    )
    if built.returncode != 0:
        detail = built.stderr.decode("utf-8", "replace")[-800:]
        raise RuntimeError(f"docker build failed:\n{detail}")


async def main() -> None:
    if not _docker_running():
        print("Docker daemon is not available. Start Docker and retry.")
        return
    _ensure_image()

    upstream = _FakeGitHub()
    factory = VirtualEgressEnvironmentFactory(
        resolver=StaticVault({"github_cli_token": DEMO_REAL_TOKEN}),
        policies={
            POLICY_NAME: HttpEgressPolicy(
                name=POLICY_NAME,
                allowed_hosts=["api.github.com"],
                allowed_endpoints=[("GET", "/user")],
            )
        },
        credentials=[
            VirtualCredentialSpec(
                env_name="GH_TOKEN",
                secret=SecretRef(name="github_cli_token"),
                destination="api.github.com",
                policy_name=POLICY_NAME,
                credential_kind="opaque_token",
            )
        ],
        runner_kind="docker",
        image=IMAGE,
        upstream=upstream,
    )
    request = EnvironmentFactoryRequest(
        session_id="github-cli-demo",
        agent_name="demo-agent",
        environment_name="github-read",
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
        virtual_check = await runner.exec(
            ExecCommand.process(
                "/bin/sh",
                "-c",
                'case "$GH_TOKEN" in cayu_vc_*) printf "virtual\\n" ;; *) exit 1 ;; esac',
            )
        )
        call = await runner.exec(
            ExecCommand.process(
                "gh",
                "api",
                "user",
                "--hostname",
                "github.com",
                "--jq",
                ".login",
            ),
            env={"GH_NO_UPDATE_NOTIFIER": "1", "GH_PROMPT_DISABLED": "1"},
            timeout_s=30,
        )
        if virtual_check.exit_code != 0 or call.exit_code != 0:
            raise RuntimeError(call.stderr or call.stdout or "GitHub CLI invocation failed")

        sent = upstream.request
        if sent is None:
            raise RuntimeError("GitHub CLI request did not reach the fake upstream")
        output = call.stdout.strip()
        upstream_received_real = sent.headers.get("Authorization") == f"token {DEMO_REAL_TOKEN}"
        upstream_received_virtual = "cayu_vc_" in str(sent.headers)
        real_leaked_to_output = DEMO_REAL_TOKEN in call.stdout or DEMO_REAL_TOKEN in call.stderr
        if not upstream_received_real:
            raise RuntimeError("broker did not inject the real GitHub token upstream")
        if upstream_received_virtual:
            raise RuntimeError("virtual GitHub token was forwarded upstream")
        if real_leaked_to_output:
            raise RuntimeError("real GitHub token leaked to CLI output")

        print("runner GH_TOKEN shape:", virtual_check.stdout.strip())
        print("gh api user:", output)
        print("captured request:", sent.method, sent.host, sent.path)
        print("broker injected real token upstream:", upstream_received_real)
        print("virtual token forwarded upstream:", upstream_received_virtual)
        print("real token leaked to CLI output:", real_leaked_to_output)
        outcome = "completed"
    finally:
        await binding.finalize(bound, outcome=outcome)


if __name__ == "__main__":
    asyncio.run(main())
