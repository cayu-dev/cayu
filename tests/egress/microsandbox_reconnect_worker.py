"""Two-process worker for the opt-in Microsandbox reconnect integration test."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from tests.egress_e2e_support import (
    CapturingEgressAdapter,
    RecordingProviderUpstream,
    connect_probe_script,
)

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.core.events import Event
from cayu.egress import CapturedRequest, HttpEgressPolicy
from cayu.egress.microsandbox_adapter import MicrosandboxEgressAdapter
from cayu.environments import (
    EnvironmentFactoryOperation,
    EnvironmentFactoryRequest,
    EnvironmentSpec,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runners import ExecCommand
from cayu.runtime.egress import VirtualCredentialSpec, VirtualEgressEnvironmentFactory
from cayu.runtime.event_sinks import EventSink
from cayu.storage import SQLiteSessionStore
from cayu.vaults import SecretRef, StaticVault

REAL_SECRET = "sk_test_51MicrosandboxRealSecretNeverInGuest"
_SENTINEL_PATH = "/workspace/reconnect-sentinel"


class _UnusedProvider(ModelProvider):
    name = "unused"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        raise AssertionError("Provider must not run before the producer process exits.")
        yield ModelStreamEvent.text_delta("unreachable")


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise


def _factory(
    *,
    adapter: CapturingEgressAdapter,
    upstream: RecordingProviderUpstream,
    image: str,
    setup_commands: tuple[str, ...] = (),
) -> VirtualEgressEnvironmentFactory:
    return VirtualEgressEnvironmentFactory(
        resolver=StaticVault({"stripe": REAL_SECRET}),
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
        setup_commands=setup_commands,
        adapter=adapter,
        upstream=upstream,
    )


class _ExitAfterCheckpoint(EventSink):
    def __init__(
        self,
        *,
        adapter: CapturingEgressAdapter,
        sidecar_path: Path,
        sentinel: str,
    ) -> None:
        self._adapter = adapter
        self._sidecar_path = sidecar_path
        self._sentinel = sentinel

    async def emit(self, event: Event) -> None:
        if event.type != EventType.ENVIRONMENT_FACTORY_COMPLETED:
            return
        _, grant = self._adapter.captured_single_grant()
        certificate = self._adapter.captured_binding().ca_cert_pem
        metadata = event.payload.get("reconnect_metadata")
        if not isinstance(metadata, dict):
            raise AssertionError("Factory completion event omitted reconnect metadata.")
        _write_private_json(
            self._sidecar_path,
            {
                "old_presented_value": grant.presented_value,
                "first_ca_sha256": hashlib.sha256(certificate or b"").hexdigest(),
                "first_ca_pem_base64": base64.b64encode(certificate or b"").decode("ascii"),
                "sentinel": self._sentinel,
                "event_metadata": metadata,
            },
        )
        # Deliberately skip every Python finalizer. This models an orchestrator
        # process disappearing after the checkpoint commit while the microVM survives.
        os._exit(0)


async def _produce(args: argparse.Namespace) -> None:
    sentinel_command = (
        'python3 -c "from pathlib import Path; '
        f"Path('{_SENTINEL_PATH}').write_text('{args.sentinel}')\""
    )
    adapter = CapturingEgressAdapter(MicrosandboxEgressAdapter(reconnect_state_dir=args.state_dir))
    sink = _ExitAfterCheckpoint(
        adapter=adapter,
        sidecar_path=args.sidecar,
        sentinel=args.sentinel,
    )
    store = SQLiteSessionStore(args.database)
    app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
    app.register_provider(_UnusedProvider(), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name=args.environment_name),
        _factory(
            adapter=adapter,
            upstream=RecordingProviderUpstream("unused-before-restart"),
            image=args.image,
            setup_commands=(sentinel_command,),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="e2e", model="unused-model"))
    async for _ in app.run(
        RunRequest(
            agent_name="e2e",
            session_id=args.session_id,
            environment_name=args.environment_name,
            messages=[Message.text("user", "establish the reconnect checkpoint")],
        )
    ):
        pass
    raise AssertionError("Producer reached model execution instead of exiting after checkpoint.")


async def _consume(args: argparse.Namespace) -> None:
    producer_evidence = json.loads(args.sidecar.read_text(encoding="utf-8"))
    store = SQLiteSessionStore(args.database)
    try:
        checkpoint = await store.load_checkpoint(args.session_id)
    finally:
        await store.close()
    if not isinstance(checkpoint, dict):
        raise AssertionError("Producer process did not persist a Cayu checkpoint.")
    reconnect_state = checkpoint.get("environment_factory_reconnect")
    if not isinstance(reconnect_state, dict):
        raise AssertionError("Checkpoint omitted environment factory reconnect state.")
    durable_metadata = reconnect_state.get(args.environment_name)
    if not isinstance(durable_metadata, dict):
        raise AssertionError("Checkpoint omitted this environment's reconnect envelope.")
    if durable_metadata != producer_evidence["event_metadata"]:
        raise AssertionError("Event and durable checkpoint reconnect metadata diverged.")
    serialized_checkpoint = json.dumps(checkpoint, sort_keys=True)
    if REAL_SECRET in serialized_checkpoint:
        raise AssertionError("Checkpoint persisted the real credential.")
    if producer_evidence["old_presented_value"] in serialized_checkpoint:
        raise AssertionError("Checkpoint persisted the old virtual credential.")

    adapter = CapturingEgressAdapter(MicrosandboxEgressAdapter(reconnect_state_dir=args.state_dir))
    upstream = RecordingProviderUpstream("cus_after_process_restart")
    result = await _factory(adapter=adapter, upstream=upstream, image=args.image).create(
        EnvironmentFactoryRequest(
            session_id=args.session_id,
            agent_name="e2e",
            environment_name=args.environment_name,
            operation=EnvironmentFactoryOperation.RECONNECT,
            reconnect_metadata=durable_metadata,
        )
    )
    runner = result.environment.runner
    binding = result.environment.binding
    if runner is None or binding is None:
        raise AssertionError("Reconnected environment omitted its runner or binding.")
    finalized = False
    try:
        if result.reconnect_metadata != durable_metadata:
            raise AssertionError("Reconnect changed the durable sandbox identity.")
        identity = durable_metadata.get("identity")
        if not isinstance(identity, dict) or runner.name != identity.get("sandbox_name"):
            raise AssertionError("Reconnect attached a different Microsandbox sandbox.")
        second_ca_sha256 = hashlib.sha256(adapter.captured_binding().ca_cert_pem or b"").hexdigest()
        if second_ca_sha256 == producer_evidence["first_ca_sha256"]:
            raise AssertionError("Reconnect reused the prior process session CA.")

        read = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                f"from pathlib import Path; print(Path('{_SENTINEL_PATH}').read_text())",
            )
        )
        if read.exit_code != 0 or read.stdout.strip() != producer_evidence["sentinel"]:
            raise AssertionError(f"Workspace sentinel did not survive: {read.stderr}")

        provider_call = await runner.exec(
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
        if provider_call.exit_code != 0 or "cus_after_process_restart" not in provider_call.stdout:
            raise AssertionError(f"Fresh brokered provider call failed: {provider_call.stderr}")
        if upstream.authorization != f"Bearer {REAL_SECRET}":
            raise AssertionError("Fresh broker did not inject the configured real credential.")

        stale_ca = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                "import base64,pathlib,ssl,sys,urllib.error as e,urllib.request as u\n"
                "p=pathlib.Path('/tmp/cayu-prior-process-ca.pem')\n"
                "p.write_bytes(base64.b64decode(sys.argv[1]))\n"
                "context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)\n"
                "context.load_verify_locations(cafile=str(p))\n"
                "r=u.Request('https://api.stripe.com/v1/customers',data=b'x=1')\n"
                "try:\n"
                " u.urlopen(r,timeout=20,context=context).read()\n"
                "except e.URLError as exc:\n"
                " reason=exc.reason\n"
                " if not isinstance(reason,ssl.SSLCertVerificationError) and "
                "'CERTIFICATE_VERIFY_FAILED' not in str(reason):\n"
                "  raise\n"
                " print('stale-ca-rejected')\n"
                "else:\n"
                " raise SystemExit('prior process CA unexpectedly authenticated the new proxy')\n"
                "finally:\n"
                " p.unlink(missing_ok=True)\n",
                producer_evidence["first_ca_pem_base64"],
            ),
            timeout_s=30,
        )
        if stale_ca.exit_code != 0 or "stale-ca-rejected" not in stale_ca.stdout:
            raise AssertionError(f"Prior process CA was not rejected: {stale_ca.stderr}")

        direct = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                connect_probe_script("1.1.1.1", 443, probe_kind="tls"),
            ),
            timeout_s=15,
        )
        if direct.exit_code != 0 or json.loads(direct.stdout)["tcp_connected"] is not False:
            raise AssertionError("Reconnected guest could bypass the fresh broker.")

        broker, fresh_grant = adapter.captured_single_grant()
        old_presented_value = producer_evidence["old_presented_value"]
        if fresh_grant.presented_value == old_presented_value:
            raise AssertionError("Reconnect reused the old process virtual credential.")
        stale = await broker.handle_request(
            CapturedRequest(
                method="POST",
                host="api.stripe.com",
                path="/v1/customers",
                headers={"Authorization": f"Bearer {old_presented_value}"},
            )
        )
        if stale.status_code != 403:
            raise AssertionError("Old process virtual credential authorized after reconnect.")

        bound = await binding.bind(None, runner, session_id=args.session_id)
        await binding.finalize(bound, outcome="completed")
        finalized = True
        _write_private_json(
            args.result,
            {
                "checkpoint_round_trip": True,
                "same_identity": True,
                "sentinel_preserved": True,
                "fresh_ca": True,
                "fresh_brokered_call": True,
                "old_ca_denied": True,
                "old_grant_denied": True,
                "direct_egress_denied": True,
                "sandbox_removed": True,
            },
        )
    finally:
        if not finalized:
            await runner.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("produce", "consume"))
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--environment-name", required=True)
    parser.add_argument("--sentinel", required=True)
    parser.add_argument("--image", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.phase == "produce":
        asyncio.run(_produce(args))
    else:
        asyncio.run(_consume(args))


if __name__ == "__main__":
    main()
