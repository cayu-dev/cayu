"""Explicit colocated server setup for the canonical deterministic browser corpus.

No infrastructure or dependencies are installed here. The caller supplies an
already running application container and its private TLS files.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import ssl
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from pydantic import SecretBytes, SecretStr

from cayu.evals.internal.browser_acceptance_operator import (
    OperatorFixtureBinding,
    OperatorFixtureSetup,
)


def _fatal_cleanup_signal(failure):
    if isinstance(failure, BaseExceptionGroup):
        return any(_fatal_cleanup_signal(item) for item in failure.exceptions)
    return not isinstance(failure, (Exception, asyncio.CancelledError))


async def settle_operator_fixture(applications, server, server_task, primary_failure=None):
    """Settle owned fixture resources without releasing an undrained store."""
    failures = []
    drained = []
    task = asyncio.current_task()
    cancellation_count = task.cancelling() if task is not None else 0
    owner_cancellation = None

    def retain(failure):
        nonlocal cancellation_count, owner_cancellation
        current = task.cancelling() if task is not None else 0
        if isinstance(failure, asyncio.CancelledError) and current > cancellation_count:
            owner_cancellation = failure
        cancellation_count = current
        if failure is not primary_failure and all(failure is not item for item in failures):
            failures.append(failure)

    for application in applications:
        try:
            if not await application.drain_environment_cleanups(timeout_s=30):
                raise RuntimeError("Operator acceptance environment cleanup is unsettled.")
        except BaseException as failure:
            retain(failure)
        else:
            drained.append(application)
    if server is not None:
        server.should_exit = True
    if server_task is not None:
        try:
            await server_task
        except BaseException as failure:
            retain(failure)
    for application in drained:
        try:
            await application.session_store.close()
        except BaseException as failure:
            retain(failure)
    if failures:
        if primary_failure is not None:
            failures.insert(0, primary_failure)
        cancellation = owner_cancellation or (
            primary_failure if isinstance(primary_failure, asyncio.CancelledError) else None
        )
        if cancellation is not None and not any(_fatal_cleanup_signal(item) for item in failures):
            causes = [item for item in failures if item is not cancellation]
            if cancellation.__cause__ is not None:
                causes.insert(0, cancellation.__cause__)
            if causes:
                raise cancellation from BaseExceptionGroup(
                    "Operator acceptance cleanup failed", causes
                )
            raise cancellation
        if len(failures) == 1:
            raise failures[0]
        raise BaseExceptionGroup("Operator acceptance and cleanup failed", failures)


@asynccontextmanager
async def operator_fixture_setup(
    *,
    server_container_id: str,
    ca_certificate: Path,
    server_private_key: Path,
    private_text: SecretStr | None = None,
):
    """Own a private local fixture client and serve the exact canonical plan.

    TLS must cover both 127.0.0.1 and cayu-control. Port 8443 must be free inside
    the specified application container. These are fixture identities, not live
    account credentials; this setup cannot authorize an authenticated campaign.
    """
    import httpx
    import uvicorn

    from cayu.runtime.browser_control import (
        BrowserControlPolicy,
        BrowserControlPolicyResult,
        BrowserOperatorPurpose,
    )
    from cayu.runtime.browser_control_config import BrowserControlConfig
    from cayu.server import BasicAuth, BrowserControlServerConfig, ServerConfig, create_server

    if not server_private_key.is_absolute() or not server_private_key.is_file():
        raise ValueError("Operator fixture requires an existing absolute private TLS key path.")
    purpose = BrowserOperatorPurpose(
        code="acceptance", expected_origins=("https://docs.browser.test",)
    )

    class Policy(BrowserControlPolicy):
        identity = "canonical-fixture-operator:v1"
        profiles = frozenset()

        async def decide(self, request):
            return BrowserControlPolicyResult(
                allowed=(
                    request.principal.subject == "operator"
                    and request.identity.operator_purpose == purpose
                    and request.identity.execution_profile_fingerprint in self.profiles
                )
            )

    policy = Policy()
    tls = ssl.create_default_context(cafile=str(ca_certificate))
    password = secrets.token_urlsafe(32)
    async with httpx.AsyncClient(
        base_url="https://127.0.0.1:8443",
        verify=tls,
        auth=httpx.BasicAuth("operator", password),
        trust_env=False,
    ) as client:
        binding = OperatorFixtureBinding(
            control=BrowserControlConfig(
                policy=policy,
                purpose=purpose,
                guest_endpoint="wss://cayu-control:8443/api/browser-control/guest",
            ),
            server_container_id=server_container_id,
            ca_certificate=ca_certificate,
            client=client,
            tls=tls,
            operator_origin="https://operator.test",
            private_text=private_text or SecretStr(secrets.token_urlsafe(24)),
        )

        @asynccontextmanager
        async def serve(plan):
            app, suite = plan.eval_plan.app, plan.eval_plan.suite
            if app is None or suite is None:
                raise ValueError("Operator fixture requires the canonical application plan.")
            applications = {id(app): app, **{id(value): value for _, value in plan.case_apps}}
            server = server_task = None
            try:
                profiles = []
                for case in suite.cases:
                    if case.id == "operator-private-handoff":
                        profiles.append(await app.inspect_run_execution_profile(case.request))
                policy.profiles = frozenset(profiles)
                server = uvicorn.Server(
                    uvicorn.Config(
                        create_server(
                            app,
                            config=ServerConfig.protected(
                                BasicAuth(username="operator", password=password),
                                browser_control=BrowserControlServerConfig(
                                    operator_origin=binding.operator_origin,
                                    signing_key=SecretBytes(secrets.token_bytes(32)),
                                ),
                            ),
                        ),
                        host="0.0.0.0",
                        port=8443,
                        ssl_certfile=str(ca_certificate),
                        ssl_keyfile=str(server_private_key),
                        ws="websockets-sansio",
                        ws_per_message_deflate=False,
                        access_log=False,
                        log_level="error",
                        timeout_graceful_shutdown=5,
                    )
                )
                server_task = asyncio.create_task(server.serve())
                async with asyncio.timeout(10):
                    while not server.started:
                        if server_task.done():
                            await server_task
                            raise RuntimeError("Operator fixture server exited before readiness.")
                        await asyncio.sleep(0.025)
                yield
            finally:
                await settle_operator_fixture(
                    applications.values(), server, server_task, sys.exception()
                )

        yield OperatorFixtureSetup(binding=binding, serve=serve)


@asynccontextmanager
async def configured_fixture():
    """CLI target configured with explicit local container and TLS file authority."""
    names = (
        "CAYU_BROWSER_ACCEPTANCE_CONTROL_CONTAINER",
        "CAYU_BROWSER_ACCEPTANCE_CONTROL_CERTIFICATE",
        "CAYU_BROWSER_ACCEPTANCE_CONTROL_PRIVATE_KEY",
    )
    values = tuple(os.environ.get(name) for name in names)
    if any(value is None or not value for value in values):
        raise ValueError("Operator fixture container and TLS paths must be configured explicitly.")
    container, certificate, key = values
    assert container is not None and certificate is not None and key is not None
    async with operator_fixture_setup(
        server_container_id=container,
        ca_certificate=Path(certificate),
        server_private_key=Path(key),
    ) as setup:
        yield setup
