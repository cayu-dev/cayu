"""Opt-in canonical local OpenAI/Docker MFA acceptance.

Run only with the documented existing Docker/dependency prerequisites and an
explicit spending allowance. The key arrives on stdin, never in Docker config.
The trusted setup yields a public EvalPlan/WebBridge application to the standard
acceptance command; it does not inspect or modify browser-worker state.
"""

import asyncio
import ipaddress
import json
import os
import secrets
import shlex
import ssl
import subprocess
import sys
import tempfile
import time
from contextlib import aclosing, asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = "gpt-5.4-mini"
PAID_LIMIT = Decimal(os.environ.get("CAYU_ACCEPTANCE_MAX_USD", "0"))


class OperatorReadPending(RuntimeError):
    pass


def _raise_cleanup_failures(primary, failures, owner_cancellation=None):
    from cayu.evals.internal.browser_acceptance_operator_server import _fatal_cleanup_signal

    failures = [failure for failure in failures if failure is not primary]
    if not failures:
        return
    ordered = failures if primary is None else [primary, *failures]
    cancellation = owner_cancellation or (
        primary if isinstance(primary, asyncio.CancelledError) else None
    )
    if cancellation is not None and not any(_fatal_cleanup_signal(item) for item in ordered):
        causes = [item for item in ordered if item is not cancellation]
        if cancellation.__cause__ is not None:
            causes.insert(0, cancellation.__cause__)
        if causes:
            raise cancellation from BaseExceptionGroup(
                "Authenticated acceptance cleanup failed", causes
            )
        raise cancellation
    if len(ordered) == 1:
        raise ordered[0]
    raise BaseExceptionGroup("Authenticated acceptance and cleanup failed", ordered) from None


async def _retire_fixture(app, server, task, client, profiles, provider, primary):
    from cayu.evals.internal.browser_acceptance_operator_server import settle_operator_fixture

    failures = []
    owner = asyncio.current_task()
    cancellation_baseline = owner.cancelling() if owner is not None else 0
    quiescent = False
    try:
        await settle_operator_fixture([app], server, task)
        quiescent = True
    except BaseException as failure:
        failures.append(failure)
    # The operator client is independent. Profile/provider ownership must survive
    # an uncertain environment; the host launcher then owns allocation teardown.
    closers = [client.aclose]
    if quiescent:
        closers.extend((profiles.close, provider.aclose))
    for close in closers:
        try:
            await close()
        except BaseException as failure:
            failures.append(failure)
    owner_cancellation = (
        next((failure for failure in failures if isinstance(failure, asyncio.CancelledError)), None)
        if owner is not None and owner.cancelling() > cancellation_baseline
        else None
    )
    _raise_cleanup_failures(primary, failures, owner_cancellation)


class Operator:
    def __init__(self, client, tls, fixture):
        self.client, self.tls, self.fixture = client, tls, fixture
        self.headers = {}
        self.session_id = None
        self.request_id = "bt_" + secrets.token_hex(16)

    async def call(self, method, path, data=None):
        response = await self.client.request(
            method, "/api/browser-control" + path, headers=self.headers, json=data
        )
        print(
            "PHASE operator-"
            + method
            + "-"
            + path.split("/")[-1]
            + "-"
            + str(response.status_code),
            flush=True,
        )
        if method == "GET" and response.status_code == 403:
            raise OperatorReadPending("Read crossed a control transition")
        if response.status_code not in (200, 202):
            raise RuntimeError("Operator request rejected: " + str(response.status_code))
        if len(response.content) > 65536:
            raise RuntimeError("Operator response exceeds bound")
        return response.json()

    async def current(self):
        value = await self.call("GET", "/sessions/" + self.session_id)
        rows = value["browsers"]
        if len(rows) != 1:
            raise RuntimeError("Expected one owned browser")
        return rows[0]

    async def wait(self, predicate):
        async with asyncio.timeout(20):
            while True:
                try:
                    row = await self.current()
                except OperatorReadPending:
                    await asyncio.sleep(0.05)
                    continue
                if predicate(row):
                    return row
                if row["state"] in ("closed", "control_uncertain", "allocation_lost"):
                    raise RuntimeError("Operator ownership did not settle")
                await asyncio.sleep(0.05)

    def intent(self, row):
        return dict(
            identity=row["identity"],
            request_id=self.request_id,
            expected_record_revision=row["revision"],
            expected_control_epoch=row["control_epoch"],
        )

    async def pages(self, row):
        result = await self.call(
            "POST",
            "/pages",
            dict(identity=row["identity"], expected_record_revision=row["revision"]),
        )
        if len(result["pages"]) != 1:
            raise RuntimeError("Expected one admitted page")
        return result["pages"]

    async def send(self, kind, value):
        from websockets.asyncio.client import connect

        from cayu.server._browser_input_routes import OPERATOR_INPUT_SUBPROTOCOL
        from cayu.tools._browser_control_transport import _private_transport_logger

        row = await self.wait(
            lambda r: (
                r["state"] == "operator_controlled"
                and r["sensitive_entry"]
                and not r["sensitive_entry_pending"]
            )
        )
        seq = row["owned_request"]["settled_input_sequence"] + 1
        ticket = await self.call(
            "POST",
            "/input-ticket",
            {
                **self.intent(row),
                "page": (await self.pages(row))[0],
                "input_sequence": seq,
                "input_kind": kind,
            },
        )
        async with connect(
            "wss://127.0.0.1:8443/api/browser-control/input",
            ssl=self.tls,
            origin="https://operator.test",
            proxy=None,
            compression=None,
            max_size=65536,
            subprotocols=[OPERATOR_INPUT_SUBPROTOCOL],
            logger=_private_transport_logger(),
        ) as channel:
            await channel.send(ticket["ticket"])
            if await channel.recv() != "ready":
                raise RuntimeError("Private input not ready")
            await channel.send(value.encode())
            receipt = json.loads(await channel.recv())
            if receipt.get("state") != "settled" or receipt.get("settled_input_sequence") != seq:
                raise RuntimeError("Private input not settled")

    async def login(self):
        from examples.browser_acceptance.local_mfa import totp

        async with asyncio.timeout(90):
            token = await self.call("POST", "/operator-session")
            self.headers = {"X-Cayu-Browser-Operator": token["operator_session_token"]}
            row = await self.current()
            now = time.time_ns() // 1000000
            await self.call(
                "POST",
                "/takeover",
                {
                    **self.intent(row),
                    "pages": await self.pages(row),
                    "purpose_code": "login",
                    "requested_at_ms": now,
                    "expires_at_ms": now + 60000,
                    "maximum_until_ms": now + 90000,
                    "checkpoint_consent": "allow",
                },
            )
            row = await self.wait(lambda r: r["state"] == "operator_controlled")
            await self.call("POST", "/sensitive-entry", self.intent(row))
            await self.send("text", "disposable")
            await self.send("tab", "tab")
            await self.send("text", self.fixture.password)
            await self.send("enter", "enter")
            while not self.fixture.mfa_seen:
                await asyncio.sleep(0.05)
            code = totp(self.fixture.key, int(time.time()) // 30)
            self.fixture.private_values.append(code)
            await self.send("text", code)
            await self.send("enter", "enter")
            while not self.fixture.member_reads:
                await asyncio.sleep(0.05)
            row = await self.wait(
                lambda r: (
                    r["state"] == "operator_controlled"
                    and r["owned_request"]["pending_input_sequence"] is None
                )
            )
            await self.call("POST", "/handback", self.intent(row))
            await self.wait(
                lambda r: r["state"] == "agent_controlled" and r["fresh_observation_required"]
            )


@asynccontextmanager
async def setup():
    startup_cleanups = []
    entered = False
    try:
        async with _setup(startup_cleanups) as plan:
            entered = True
            yield plan
    except BaseException as primary:
        if not entered:
            failures = []
            for close in reversed(startup_cleanups):
                try:
                    await close()
                except BaseException as failure:
                    failures.append(failure)
            _raise_cleanup_failures(primary, failures)
        raise


@asynccontextmanager
async def _setup(startup_cleanups):
    if not Decimal("0") < PAID_LIMIT <= Decimal("1"):
        raise ValueError(
            "Set CAYU_ACCEPTANCE_MAX_USD to an explicitly authorized amount up to one dollar"
        )
    proof = Path(os.environ["CAYU_ACCEPTANCE_PROOF_DIR"])
    api_key = sys.stdin.readline().strip()
    if not api_key:
        raise ValueError("Provide an explicitly authorized API key on stdin")
    import socket
    from hashlib import sha256

    import httpx
    import uvicorn
    from examples.browser_acceptance.local_mfa import LocalMfa
    from fastapi import Request
    from pydantic import SecretBytes, SecretStr

    from cayu import (
        AESGCMBrowserProfileKeyAuthority,
        AgentSpec,
        ApprovedEgressDestination,
        BrowserProfileBinding,
        BrowserProfileDestinationPolicy,
        BrowserProfileScope,
        BudgetLimit,
        BudgetReservation,
        CayuApp,
        ChatCompletionsProvider,
        EnvironmentSpec,
        EvalCase,
        EvalPlan,
        EvalSuite,
        HttpEgressPolicy,
        LocalArtifactStore,
        Message,
        ModelPrice,
        PriceBook,
        RunLimits,
        RunRequest,
        RuntimeHook,
        SecretRedactor,
        SessionCompleted,
        SQLiteBrowserProfileStore,
        SQLiteSessionStore,
        VirtualEgressEnvironmentFactory,
        WebBridge,
    )
    from cayu.egress import HttpxUpstream
    from cayu.evals import (
        BrowserAcceptanceAuthenticatedConfigV1,
        BrowserAcceptanceAuthenticationCollector,
        BrowserAcceptancePlanV1,
        live_authenticated_browser_acceptance_manifest,
    )
    from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD
    from cayu.runtime.browser_control import (
        BrowserControlPolicy,
        BrowserControlPolicyResult,
        BrowserOperatorPurpose,
    )
    from cayu.runtime.browser_control_config import BrowserControlConfig
    from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
    from cayu.runtime.retry_policy import RetryPolicy
    from cayu.server import BasicAuth, BrowserControlServerConfig, ServerConfig, create_server
    from cayu.tools.browser_session import (
        BROWSER_SESSION_PROTOCOL_VERSION,
        BROWSER_SESSION_WORKER_VERSION,
    )

    class Site(LocalMfa):
        login_seen = mfa_seen = member_reads = 0

        async def login(self):
            self.login_seen += 1
            return await super().login()

        async def mfa(self, request: Request):
            self.mfa_seen += 1
            return await super().mfa(request)

        async def member(self, request: Request):
            if request.cookies.get("fixture_auth_cookie") in self.sessions:
                self.member_reads += 1
            return await super().member(request)

    site = Site()
    # Use the existing disposable fixture, reached only through admitted broker traffic.
    transport = httpx.ASGITransport(app=site.api)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://mfa.example.test"
    ) as negative:
        assert (await negative.get("/fixture/member")).status_code == 307
        assert (
            await negative.post(
                "/fixture/password", json={"username": "disposable", "password": "wrong"}
            )
        ).status_code == 401
    site.login_seen = site.mfa_seen = site.member_reads = 0
    operator_password = secrets.token_urlsafe(32)
    site.private_values.extend([operator_password, api_key])

    class Policy(BrowserControlPolicy):
        identity = "local-openai-mfa:v1"

        async def decide(self, request):
            return BrowserControlPolicyResult(
                allowed=(
                    request.principal.subject == "operator"
                    and request.identity.session_id == operator.session_id
                    and request.identity.operator_purpose.code == "login"
                )
            )

    control = BrowserControlConfig(
        policy=Policy(),
        purpose=BrowserOperatorPurpose(
            code="login", expected_origins=("https://mfa.example.test",)
        ),
        guest_endpoint="wss://cayu-control:8443/api/browser-control/guest",
    )

    class SessionBindingHook(RuntimeHook):
        async def after_tool_call(self, context):
            if context.tool_name == "browser_session" and not context.result.is_error:
                if operator.session_id is not None and operator.session_id != context.session.id:
                    raise RuntimeError("The local MFA setup cannot serve a second trial")
                operator.session_id = context.session.id

    store = SQLiteSessionStore(
        proof / "sessions.sqlite",
        public_authority_alias_codec=PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(
                active_key_id="proof", keys={"proof": SecretStr(secrets.token_urlsafe(32))}
            )
        ),
    )
    startup_cleanups.append(store.close)
    observer_revision = (
        "sha256:" + sha256(b"exact-password-totp-once-and-member-cookie:v1").hexdigest()
    )

    def authenticated_count():
        return site.member_reads if site.password_checks == site.mfa_checks == 1 else 0

    authentication_collector = BrowserAcceptanceAuthenticationCollector(
        authenticated_count, observer_revision=observer_revision
    )
    app = CayuApp(
        session_store=store,
        browser_control=control,
        enable_logging=False,
        runtime_hooks=[SessionBindingHook(), authentication_collector],
        secret_redactor=SecretRedactor([site.password, operator_password, api_key]),
    )
    artifacts = LocalArtifactStore(proof / "artifacts", store_id="local-mfa-artifacts")
    factory = VirtualEgressEnvironmentFactory(
        policies={
            "mfa": HttpEgressPolicy(
                name="mfa",
                allowed_hosts=("mfa.example.test",),
                allowed_endpoints=(
                    ("GET", "/fixture/login"),
                    ("GET", "/fixture/mfa"),
                    ("GET", "/fixture/member"),
                    ("POST", "/fixture/password"),
                    ("POST", "/fixture/verify"),
                ),
            )
        },
        approved_destinations=(
            ApprovedEgressDestination(destination="mfa.example.test", policy_name="mfa"),
        ),
        adapter=_campaign_docker_adapter(
            proof,
            docker_exec=_recording_docker_exec(proof),
            seccomp_profile=str(REPO / "examples/browser_fetch/seccomp_profile.json"),
            control_server_container_id=(proof / "controller-id").read_text().strip(),
        ),
        upstream=HttpxUpstream(
            transport=transport,
            routes={
                "mfa.example.test": "http://" + socket.gethostbyname(socket.gethostname()) + ":8080"
            },
        ),
        image=PINNED_BROWSER_SESSION_WORKLOAD.image,
        artifact_store=artifacts,
        host_workspace_path=str(proof / "trust"),
        setup_commands=(
            "cp "
            + shlex.quote(str(proof / "trust/control.crt"))
            + " /usr/local/share/ca-certificates/cayu-control.crt && update-ca-certificates",
        ),
    )
    profiles = SQLiteBrowserProfileStore(proof / "profiles.sqlite", store_id="local-mfa-profiles")
    startup_cleanups.append(profiles.close)
    binding = BrowserProfileBinding.build(
        scope=BrowserProfileScope.build(
            application_id="mfa-proof", tenant_id="local", sharing_scope="one-test"
        ),
        destination_policy=BrowserProfileDestinationPolicy.build(("https://mfa.example.test",)),
        browser_protocol=BROWSER_SESSION_PROTOCOL_VERSION,
        browser_worker_version=BROWSER_SESSION_WORKER_VERSION,
        store=profiles,
        key_authority=AESGCMBrowserProfileKeyAuthority(
            authority_id="disposable", key=os.urandom(32)
        ),
    )
    await binding.initialize()
    bridge = WebBridge.sandboxed_browser(
        environment=factory,
        browser_image=PINNED_BROWSER_SESSION_WORKLOAD.image,
        interactive=True,
        browser_profile=binding,
        interactive_options={
            "max_operations": 10,
            "max_sessions": 1,
            "idle_timeout_seconds": 180,
            "max_artifact_bytes": 1024 * 1024,
        },
    )
    tls = ssl.create_default_context(cafile=str(proof / "trust/control.crt"))
    client = httpx.AsyncClient(
        base_url="https://127.0.0.1:8443",
        verify=tls,
        auth=httpx.BasicAuth("operator", operator_password),
        trust_env=False,
    )
    startup_cleanups.append(client.aclose)
    operator = Operator(client, tls, site)
    handed_back = False

    provider = ChatCompletionsProvider(api_key=api_key, name="openai", timeout_s=45)
    startup_cleanups.append(provider.aclose)
    delegate = provider.transport

    class Guard:
        count = 0
        reserved = Decimal(0)

        async def stream_chat_completions(self, **kwargs):
            nonlocal handed_back
            if site.login_seen and not handed_back:
                print("PHASE private-login", flush=True)
                await operator.login()
                handed_back = True
                print("PHASE handback-complete", flush=True)
            payload = kwargs["payload"]
            raw = json.dumps(payload, ensure_ascii=True)
            print("PHASE payload-bytes-" + str(len(raw.encode())), flush=True)
            site.assert_safe(raw)
            # Bound every actual HTTP dispatch, including any runtime retry.
            # Text only, no hosted tools: byte count * 2 plus framing is a
            # deliberately conservative input reservation, never an actual bill.
            if payload.get("model") != MODEL or len(raw.encode()) > 120000:
                raise RuntimeError("Paid request outside approved model/input bound")
            if payload.get("max_completion_tokens", payload.get("max_tokens")) != 1024:
                raise RuntimeError("Paid output bound missing")
            if any(
                p.get("type") != "text"
                for m in payload["messages"]
                if isinstance(m.get("content"), list)
                for p in m["content"]
            ):
                raise RuntimeError("Paid proof requires text-only model input")
            reserve = Decimal(2 * len(raw.encode()) + 2048) * Decimal(".00000075") + Decimal(
                ".004608"
            )
            if self.count >= 12 or self.reserved + reserve > PAID_LIMIT:
                raise RuntimeError("Approved paid budget exhausted before dispatch")
            self.count += 1
            self.reserved += reserve
            print("PHASE provider-request-" + str(self.count), flush=True)
            async with aclosing(delegate.stream_chat_completions(**kwargs)) as events:
                async for event in events:
                    yield event

        async def aclose(self):
            await delegate.aclose()

    guard = Guard()
    provider.transport = guard
    app.register_provider(provider, default=True)
    app.register_environment_factory(EnvironmentSpec(name="browser"), factory, default=True)
    bridge.register_agent(
        app,
        AgentSpec(
            name="mfa-agent",
            model=MODEL,
            provider_options={
                "openai": {
                    "max_completion_tokens": 1024,
                    "reasoning_effort": "none",
                    "parallel_tool_calls": False,
                }
            },
            system_prompt=(
                "Use only browser_session, one call at a time. Follow this ordered local test: "
                "navigate https://mfa.example.test/fixture/login. An application-owned operator then performs private "
                "password and TOTP entry before your next turn; never request or type credentials. "
                "After your first navigation call observe with operation_id acceptance-post-handback. "
                "Verify the member page says Authenticated localhost member and Password and TOTP verified. "
                "Close that browser, then navigate https://mfa.example.test/fixture/member in a new browser "
                "without reusing the closed session id. The encrypted profile should restore authentication. "
                "Call observe with operation_id acceptance-restored, verify the same member state, close, and summarize. "
                "Do not capture pixels, download, use other origins, or change the account. "
                "Observe accepts operation, session_id, page_id, operation_id only. "
                "Use exact identities returned by successful tools. Never invent revisions."
            ),
        ),
    )
    price = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="openai",
                model=MODEL,
                input_per_million=Decimal(".75"),
                output_per_million=Decimal("4.5"),
                currency="USD",
            ),
        )
    )
    request = RunRequest(
        agent_name="mfa-agent",
        messages=[
            Message.text(
                "user",
                "Perform the designated local login, fresh observation, close and restored-profile verification.",
            )
        ],
        max_steps=12,
        retry_policy=RetryPolicy(max_attempts=1, max_unknown_attempts=1),
        limits=RunLimits(
            max_tool_calls=10,
            max_elapsed_seconds=330,
            max_input_tokens=96000,
            max_output_tokens=12288,
            max_total_tokens=108288,
        ),
        budget_limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=PAID_LIMIT,
                pricing=price,
                reservation=BudgetReservation(max_input_tokens=242048, max_output_tokens=1024),
            ),
        ),
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_server(
                app,
                config=ServerConfig.protected(
                    BasicAuth(username="operator", password=operator_password),
                    browser_control=BrowserControlServerConfig(
                        operator_origin="https://operator.test",
                        signing_key=SecretBytes(os.urandom(32)),
                    ),
                ),
            ),
            host="0.0.0.0",
            port=8443,
            ssl_certfile=str(proof / "trust/control.crt"),
            ssl_keyfile=str(proof / "server.key"),
            ws="websockets-sansio",
            access_log=False,
            log_level="error",
            timeout_graceful_shutdown=5,
        )
    )
    task = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.05)
        config = BrowserAcceptanceAuthenticatedConfigV1(
            authorized=True,
            account_scope_revision="sha256:" + sha256(b"disposable-local-mfa-v1").hexdigest(),
            site_observer_revision=observer_revision,
            profile_authority_fingerprint=binding.authority.fingerprint,
            operator_policy_fingerprint=sha256(Policy.identity.encode()).hexdigest(),
            origin="https://mfa.example.test",
            login_path="/fixture/login",
            protected_path="/fixture/member",
            allowed_endpoints=(
                ("GET", "/fixture/login"),
                ("GET", "/fixture/member"),
                ("GET", "/fixture/mfa"),
                ("POST", "/fixture/password"),
                ("POST", "/fixture/verify"),
            ),
            max_estimated_cost=str(PAID_LIMIT) + " USD",
        )
        manifest = live_authenticated_browser_acceptance_manifest(config)
        case = manifest.cases[0]
        plan = BrowserAcceptancePlanV1(
            manifest=manifest,
            bridge=bridge,
            authenticated=config,
            authenticated_request_count=authenticated_count,
            authentication_collector=authentication_collector,
            pricing=price,
            cost_currencies=("USD",),
            eval_plan=EvalPlan(
                app=app,
                suite=EvalSuite(
                    id=manifest.suite_id,
                    cases=[
                        EvalCase(
                            id=case.case_id,
                            request=request,
                            assertions=[SessionCompleted()],
                            metadata={"browser_acceptance_case_revision": case.revision},
                        )
                    ],
                ),
            ),
        )
        yield plan
        if operator.session_id is not None:
            events = await store.load_events(operator.session_id)
            site.assert_safe(json.dumps([e.model_dump(mode="json") for e in events]))
        assert not (await artifacts.list()).artifacts
    finally:
        await _retire_fixture(app, server, task, client, profiles, provider, sys.exception())
        for name in (
            "sessions.sqlite",
            "sessions.sqlite-wal",
            "profiles.sqlite",
            "profiles.sqlite-wal",
        ):
            path = proof / name
            if path.exists():
                site.assert_safe(path.read_bytes().decode("utf-8", errors="ignore"))
        for report in proof.glob("reports/*.json"):
            site.assert_safe(report.read_text())
        print("PHASE reserved-upper-usd-" + str(guard.reserved), flush=True)


def _append_resource_record(proof, record):
    with (proof / "resource-intents.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _campaign_docker_adapter(proof, **options):
    from cayu.egress.docker_adapter import DockerEgressAdapter

    class CampaignDockerAdapter(DockerEgressAdapter):
        async def create_runner(self, request):
            if not request.session_id or not request.binding.network:
                raise RuntimeError("Campaign runner lacks session/network ownership")
            record = {
                "kind": "runner",
                "name": request.name,
                "session": request.session_id,
                "network": request.binding.network,
            }
            _append_resource_record(proof, record)
            runner = await super().create_runner(request)
            _append_resource_record(proof, {**record, "created": True})
            return runner

    return CampaignDockerAdapter(**options)


def _recording_docker_exec(proof, dispatch=None):
    # Use the adapter's normal transport; only journal owned creation intents.
    from cayu.egress.docker_adapter import _default_docker_exec

    dispatch = _default_docker_exec if dispatch is None else dispatch

    async def execute(argv):
        kind = (
            "network"
            if argv[:2] == ["network", "create"]
            else "container"
            if argv[:1] == ["run"]
            else None
        )
        if kind is not None:
            labels = [argv[index + 1] for index, item in enumerate(argv[:-1]) if item == "--label"]
            sessions = [
                value.removeprefix("cayu.egress.session=")
                for value in labels
                if value.startswith("cayu.egress.session=")
            ]
            if len(sessions) != 1 or not sessions[0]:
                raise RuntimeError("Campaign resource lacks exact session ownership")
            name = argv[-1] if kind == "network" else argv[argv.index("--name") + 1]
            record = {"kind": kind, "name": name, "session": sessions[0]}
            _append_resource_record(proof, record)
        outcome = await dispatch(argv)
        if kind is not None and outcome[0] == 0:
            _append_resource_record(proof, {**record, "created": True})
        return outcome

    return execute


def _retire_host(docker, name, proof, primary):
    failures = []
    retained = {"controller": name, "networks": [], "unverified_networks": [], "containers": []}
    removable_networks = []

    def attempt(*args):
        try:
            return docker(*args)
        except BaseException as failure:
            failures.append(failure)
            return None

    # Stop further campaign submissions before reading its creation journal.
    attempt("rm", "-f", name)
    try:
        records = []
        for number, line in enumerate(
            (proof / "resource-intents.jsonl").read_bytes().splitlines(), start=1
        ):
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                record = None
            fields = {"kind", "name", "session"}
            if type(record) is dict and record.get("kind") == "runner":
                fields.add("network")
            if (
                type(record) is not dict
                or set(record) not in (fields, fields | {"created"})
                or ("created" in record and record["created"] is not True)
                or record["kind"] not in ("container", "network", "runner")
                or any(
                    type(record[key]) is not str
                    or not 0 < len(record[key]) <= 256
                    or not record[key].isascii()
                    or any(
                        ord(character) < 33 or ord(character) == 127 for character in record[key]
                    )
                    for key in fields - {"kind"}
                )
                or not record["name"][0].isalnum()
                or any(
                    not (character.isalnum() or character in "_.-") for character in record["name"]
                )
            ):
                retained.setdefault("invalid_journal_lines", []).append(number)
                failures.append(
                    RuntimeError(f"Campaign resource journal entry {number} is invalid")
                )
                continue
            records.append(record)
        acknowledgements = {
            (record["kind"], record["name"], record["session"], record.get("network"))
            for record in records
            if record.get("created") is True
        }
        intents = [record for record in records if "created" not in record]
        for record in intents:
            if (
                record["kind"],
                record["name"],
                record["session"],
                record.get("network"),
            ) not in acknowledgements:
                retained.setdefault("unsettled_creations", []).append(record)
                failures.append(
                    RuntimeError("Campaign resource creation has no durable acknowledgement")
                )
        # Containers must retire before networks, independent of creation order.
        for kind in ("runner", "container", "network"):
            for record in intents:
                if record["kind"] != kind:
                    continue
                resource, session = record["name"], record["session"]
                retained["networks" if kind == "network" else "containers"].append(resource)
                if kind == "runner":
                    network = record["network"]
                    runners = attempt("ps", "-a", "--no-trunc", "--format", "{{.ID}} {{.Names}}")
                    matches = [
                        line.split()
                        for line in (runners or "").splitlines()
                        if len(line.split()) == 2 and line.split()[1] == resource
                    ]
                    if not matches:
                        continue
                    networks = attempt(
                        "network",
                        "ls",
                        "--format",
                        "{{.Name}}",
                        "--filter",
                        "label=cayu.egress.session=" + session,
                    )
                    if networks is None or network not in networks.splitlines():
                        failures.append(
                            RuntimeError("Campaign runner network ownership is unverified")
                        )
                        continue
                    scoped = attempt(
                        "ps",
                        "-a",
                        "--no-trunc",
                        "--format",
                        "{{.ID}}",
                        "--filter",
                        "network=" + network,
                    )
                    for identifier, _ in matches:
                        if (
                            len(identifier) != 64
                            or any(char not in "0123456789abcdef" for char in identifier)
                            or identifier not in (scoped or "").splitlines()
                        ):
                            failures.append(RuntimeError("Campaign runner identity is unverified"))
                        else:
                            retained.setdefault("runner_ids", []).append(identifier)
                            attempt("rm", "-f", identifier)
                    continue
                listing = attempt(
                    *(("ps", "-a") if kind == "container" else ("network", "ls")),
                    "--format",
                    "{{.Names}}" if kind == "container" else "{{.Name}}",
                    "--filter",
                    "label=cayu.egress.session=" + session,
                )
                if listing is None:
                    if kind == "network":
                        retained["unverified_networks"].append(resource)
                    continue
                if resource not in listing.splitlines():
                    continue
                if kind == "container":
                    attempt("rm", "-f", resource)
                else:
                    removable_networks.append(resource)
    except BaseException as failure:
        failures.append(failure)
    finally:
        for network in dict.fromkeys(removable_networks):
            attempt("network", "rm", network)
        try:
            (proof / "server.key").unlink(missing_ok=True)
        except BaseException as failure:
            failures.append(failure)
    if failures:
        try:
            (proof / "cleanup-resources.json").write_text(json.dumps(retained))
        except BaseException as failure:
            failures.append(failure)
        print("Cleanup incomplete; proof directory: " + str(proof), flush=True)
    _raise_cleanup_failures(primary, failures)


def host(api_key):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    proof = Path(tempfile.mkdtemp(prefix="cayu-openai-mfa-")).resolve()
    proof.chmod(0o700)
    (proof / "trust").mkdir()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "cayu-control")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("cayu-control"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    (proof / "trust/control.crt").write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    (proof / "server.key").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    (proof / "server.key").chmod(0o600)
    name = "cayu-openai-mfa-" + secrets.token_hex(8)
    (proof / "resource-intents.jsonl").touch(mode=0o600, exist_ok=False)

    def docker(*args, check=True):
        result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=40)
        if check and result.returncode:
            raise RuntimeError("Docker command failed: " + args[0])
        return result.stdout.strip()

    try:
        script = str(Path(__file__).resolve())
        identifier = docker(
            "run",
            "-d",
            "--name",
            name,
            "--user",
            "0",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=32m",
            "--mount",
            f"type=bind,src={REPO},dst={REPO},readonly",
            "--mount",
            f"type=bind,src={proof},dst={proof}",
            "--mount",
            f"type=bind,src={script},dst=/proof.py,readonly",
            "--mount",
            f"type=bind,src={os.environ['CAYU_ACCEPTANCE_DEPS']},dst=/deps,readonly",
            "--mount",
            f"type=bind,src={os.environ['CAYU_ACCEPTANCE_DOCKER_CLI']},dst=/usr/local/bin/docker,readonly",
            "--mount",
            "type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock",
            "--env",
            f"PYTHONPATH={REPO}/src:{REPO}:/deps",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONPYCACHEPREFIX=/tmp/proof-pycache",
            "--env",
            f"TMPDIR={proof}",
            "--env",
            f"CAYU_ACCEPTANCE_PROOF_DIR={proof}",
            "--env",
            f"CAYU_ACCEPTANCE_MAX_USD={PAID_LIMIT}",
            "--workdir",
            str(REPO),
            "--entrypoint",
            "sleep",
            os.environ["CAYU_ACCEPTANCE_CONTROLLER_IMAGE"],
            "600",
        )
        (proof / "controller-id").write_text(identifier)
        completed = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                identifier,
                "python",
                "scripts/run_browser_acceptance.py",
                "examples.browser_acceptance.local_authenticated:setup",
                "--mode",
                "live_authenticated",
                "--authorize-authenticated",
                "--output-directory",
                str(proof / "reports"),
            ],
            input=api_key + "\n",
            text=True,
            capture_output=True,
            timeout=420,
        )
        output = completed.stdout + completed.stderr
        if api_key in output:
            raise RuntimeError("Provider key leaked into captured output")
        # No arbitrary tracebacks/HTTP diagnostic bodies are displayed.
        for line in output.splitlines():
            if line.startswith(("PHASE ", "{", "browser acceptance unavailable")):
                print(line, flush=True)
        print("Proof directory: " + str(proof), flush=True)
        return completed.returncode
    finally:
        _retire_host(docker, name, proof, sys.exception())


if __name__ == "__main__":
    if not Decimal("0") < PAID_LIMIT <= Decimal("1"):
        raise SystemExit("Set an explicitly authorized CAYU_ACCEPTANCE_MAX_USD (up to one dollar)")
    supplied_key = sys.stdin.readline().strip()
    if not supplied_key:
        raise SystemExit("Explicit provider key required on stdin")
    raise SystemExit(host(supplied_key))
