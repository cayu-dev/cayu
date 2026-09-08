"""Opt-in authenticated setup and fail-closed canonical admission, without paid calls."""

import argparse
import asyncio
import io
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from decimal import Decimal

import pytest
from scripts import run_browser_acceptance as command
from tests.evals.test_browser_acceptance_operator_trust import certificate_and_key

from cayu.evals import (
    BrowserAcceptanceAuthenticatedConfigV1,
    live_authenticated_browser_acceptance_manifest,
)
from cayu.evals.browser_acceptance import (
    _authenticated_site_delta,
    _require_authenticated_plan_authority,
    _require_browser_acceptance_egress_authority,
    _require_browser_acceptance_execution_limits,
)


def configuration(**changes):
    return BrowserAcceptanceAuthenticatedConfigV1(
        **{
            "authorized": True,
            "account_scope_revision": "sha256:" + "a" * 64,
            "site_observer_revision": "sha256:" + "d" * 64,
            "profile_authority_fingerprint": "b" * 64,
            "operator_policy_fingerprint": "c" * 64,
            "origin": "https://mfa.example.test",
            "login_path": "/fixture/login",
            "protected_path": "/fixture/member",
            "allowed_endpoints": (("GET", "/fixture/login"), ("GET", "/fixture/member")),
            **changes,
        }
    )


@pytest.mark.parametrize(
    "change",
    [
        {"authorized": False},
        {"authorized": 1},
        {"operator_inputs": True},
        {"origin": "http://mfa.example.test"},
        {"origin": "https://user:private@mfa.example.test"},
        {"protected_path": "/member?token=private"},
        {"max_estimated_cost": "NaN USD"},
        {"max_estimated_cost": "0 USD"},
        {"account_scope_revision": "private"},
        {"allowed_endpoints": (("POST", "/*"),)},
    ],
)
def test_invalid_authenticated_authority_is_rejected_without_echo(change):
    with pytest.raises(ValueError) as error:
        configuration(**change)
    assert "private" not in str(error.value) + repr(error.value)


def test_authenticated_manifest_is_disabled_without_authority_and_content_bound_when_enabled():
    assert not live_authenticated_browser_acceptance_manifest().enabled
    config = configuration()
    manifest = live_authenticated_browser_acceptance_manifest(config)
    assert manifest.enabled and manifest.trial_count == 1
    assert manifest.cases[0].operations == ("navigate", "observe", "close") * 2
    for field in (
        "account_scope_revision",
        "site_observer_revision",
        "profile_authority_fingerprint",
        "operator_policy_fingerprint",
    ):
        altered = configuration(
            **{field: ("sha256:" if field.endswith("revision") else "") + "e" * 64}
        )
        assert live_authenticated_browser_acceptance_manifest(altered).revision != manifest.revision
    corrupted = config.model_copy(update={"authorized": False})
    with pytest.raises(ValueError):
        live_authenticated_browser_acceptance_manifest(corrupted)


def test_command_requires_opt_in_before_loading_any_authenticated_setup(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("setup loaded before authority")

    monkeypatch.setattr(command, "load_target", forbidden)
    with pytest.raises(RuntimeError, match="disabled"):
        asyncio.run(
            command._run(
                argparse.Namespace(
                    mode="live_authenticated",
                    target="site:setup",
                    output_directory=tmp_path,
                )
            )
        )


def test_canonical_example_builds_with_real_authorities_without_model_or_docker_dispatch(
    monkeypatch, tmp_path
):
    import uvicorn
    from examples.browser_acceptance import local_authenticated as example

    cert, key = certificate_and_key()
    (tmp_path / "trust").mkdir()
    (tmp_path / "trust/control.crt").write_bytes(cert)
    (tmp_path / "server.key").write_bytes(key)
    (tmp_path / "controller-id").write_text("a" * 64)
    monkeypatch.setenv("CAYU_ACCEPTANCE_PROOF_DIR", str(tmp_path))
    monkeypatch.setattr(example, "PAID_LIMIT", Decimal("0.50"))
    monkeypatch.setattr(example.sys, "stdin", io.StringIO("not-a-real-provider-key\n"))

    async def serve(server, *args, **kwargs):
        server.started = True
        while not server.should_exit:
            await asyncio.sleep(0.01)

    monkeypatch.setattr(uvicorn.Server, "serve", serve)

    async def scenario():
        retired = False

        @asynccontextmanager
        async def factory():
            nonlocal retired
            try:
                async with example.setup() as plan:
                    yield plan
            finally:
                retired = True

        async def inspect(args, *, plan):
            from cayu import AgentSpec, BrowserProfileBinding, BrowserProfileScope, WebBridge
            from cayu.evals import BrowserAcceptanceAuthenticationCollector, run_browser_acceptance
            from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD

            _require_browser_acceptance_execution_limits(plan)
            _require_authenticated_plan_authority(plan)
            for collector in (
                None,
                BrowserAcceptanceAuthenticationCollector(
                    plan.authenticated_request_count,
                    observer_revision=plan.authenticated.site_observer_revision,
                ),
            ):
                with pytest.raises(ValueError, match="registered phase collector"):
                    _require_authenticated_plan_authority(
                        replace(plan, authentication_collector=collector)
                    )
            assert plan.authenticated_request_count() == 0
            environment = plan.eval_plan.app.get_environment_factory("browser")
            original_profile = plan.bridge.tools[0].browser_profile
            other_profile = BrowserProfileBinding.build(
                scope=BrowserProfileScope.build(
                    application_id="other-app", tenant_id="local", sharing_scope="other"
                ),
                destination_policy=original_profile.authority.destination_policy,
                browser_protocol=plan.bridge.browser_protocol,
                browser_worker_version=plan.bridge.browser_worker_version,
                store=original_profile.store,
                key_authority=original_profile.key_authority,
            )
            for index, profile in enumerate((original_profile, other_profile)):
                override = WebBridge.sandboxed_browser(
                    environment=environment,
                    browser_image=PINNED_BROWSER_SESSION_WORKLOAD.image,
                    interactive=True,
                    browser_profile=profile,
                    interactive_options={
                        "max_operations": 10,
                        "max_sessions": 1,
                        "idle_timeout_seconds": 180,
                        "max_artifact_bytes": 1024 * 1024,
                    },
                )
                name = f"override-{index}"
                override.register_agent(
                    plan.eval_plan.app, AgentSpec(name=name, model=example.MODEL)
                )
                original_case = plan.eval_plan.suite.cases[0]
                altered = replace(
                    plan,
                    case_bridges=((original_case.id, override),),
                    eval_plan=replace(
                        plan.eval_plan,
                        suite=plan.eval_plan.suite.model_copy(
                            update={
                                "cases": [
                                    original_case.model_copy(
                                        update={
                                            "request": original_case.request.model_copy(
                                                update={"agent_name": name}
                                            )
                                        }
                                    )
                                ]
                            }
                        ),
                    ),
                )
                if index == 0:
                    _require_authenticated_plan_authority(altered)
                else:
                    with pytest.raises(ValueError, match="profile authority changed"):
                        await run_browser_acceptance(altered)
                assert plan.eval_plan.app.get_provider("openai").transport.count == 0
                assert plan.authenticated_request_count() == 0
            _require_browser_acceptance_egress_authority(
                plan.manifest, environment.egress_authority_identity, plan.authenticated
            )
            original_policy = environment.egress_authority_identity.policies[0]
            broad = environment.egress_authority_identity.model_copy(
                update={
                    "policies": (
                        original_policy.model_copy(
                            update={
                                "operations": (
                                    *original_policy.operations,
                                    original_policy.operations[0].model_copy(
                                        update={
                                            "method": "POST",
                                            "path": "/extra",
                                            "match": "exact",
                                        }
                                    ),
                                )
                            }
                        ),
                    )
                }
            )
            with pytest.raises(ValueError):
                _require_browser_acceptance_egress_authority(
                    plan.manifest, broad, plan.authenticated
                )
            for count in (True, -1, "1", (1 << 20) + 1):
                invalid = replace(plan, authenticated_request_count=lambda value=count: value)
                with pytest.raises(ValueError):
                    _require_authenticated_plan_authority(invalid)
            with pytest.raises(ValueError, match="reset"):
                _authenticated_site_delta(plan, 1)
            missing_budget = replace(
                plan,
                eval_plan=replace(
                    plan.eval_plan,
                    suite=plan.eval_plan.suite.model_copy(
                        update={
                            "cases": [
                                plan.eval_plan.suite.cases[0].model_copy(
                                    update={
                                        "request": plan.eval_plan.suite.cases[0].request.model_copy(
                                            update={"budget_limits": ()}
                                        )
                                    }
                                )
                            ]
                        }
                    ),
                ),
            )
            with pytest.raises(ValueError, match="cost ceiling"):
                _require_browser_acceptance_execution_limits(missing_budget)
            return 0

        monkeypatch.setattr(command, "load_target", lambda *args, **kwargs: factory)
        monkeypatch.setattr(command, "_run_plan", inspect)
        assert (
            await command._run(
                argparse.Namespace(
                    mode="live_authenticated",
                    authorize_authenticated=True,
                    target="site:setup",
                    output_directory=tmp_path,
                )
            )
            == 0
        )
        assert retired

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_during_close", [False, True])
def test_example_post_yield_cancellation_preserves_signal_and_attempts_all_closes(
    monkeypatch, tmp_path, cancel_during_close
):
    import httpx
    import uvicorn
    from examples.browser_acceptance import local_authenticated as example

    from cayu import ChatCompletionsProvider, SQLiteBrowserProfileStore

    cert, key = certificate_and_key()
    (tmp_path / "trust").mkdir()
    (tmp_path / "trust/control.crt").write_bytes(cert)
    (tmp_path / "server.key").write_bytes(key)
    (tmp_path / "controller-id").write_text("a" * 64)
    monkeypatch.setenv("CAYU_ACCEPTANCE_PROOF_DIR", str(tmp_path))
    monkeypatch.setattr(example, "PAID_LIMIT", Decimal("0.50"))
    monkeypatch.setattr(example.sys, "stdin", io.StringIO("not-a-real-provider-key\n"))
    entered = asyncio.Event()
    closing = asyncio.Event()
    closed = []
    client_failure, provider_failure = RuntimeError("client close"), RuntimeError("provider close")
    primary_failure = RuntimeError("trial failed")
    original_http = httpx.AsyncClient.aclose
    original_profiles = SQLiteBrowserProfileStore.close
    original_provider = ChatCompletionsProvider.aclose

    async def serve(server, *args, **kwargs):
        server.started = True
        while not server.should_exit:
            await asyncio.sleep(0.01)

    async def close_http(client):
        await original_http(client)
        if entered.is_set() and "client" not in closed:
            closed.append("client")
            if cancel_during_close:
                closing.set()
                await asyncio.Event().wait()
            raise client_failure

    async def close_profiles(store):
        await original_profiles(store)
        closed.append("profiles")

    async def close_provider(provider):
        await original_provider(provider)
        closed.append("provider")
        raise provider_failure

    monkeypatch.setattr(uvicorn.Server, "serve", serve)
    monkeypatch.setattr(httpx.AsyncClient, "aclose", close_http)
    monkeypatch.setattr(SQLiteBrowserProfileStore, "close", close_profiles)
    monkeypatch.setattr(ChatCompletionsProvider, "aclose", close_provider)

    async def scenario():
        async def run():
            async with example.setup():
                entered.set()
                if cancel_during_close:
                    raise primary_failure
                await asyncio.Event().wait()

        task = asyncio.create_task(run())
        await entered.wait()
        if cancel_during_close:
            await closing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as error:
            await task
        assert task.cancelled() and task.cancelling() == 1
        assert error.value.__cause__.exceptions == (
            primary_failure if cancel_during_close else client_failure,
            provider_failure,
        )
        assert closed == ["client", "profiles", "provider"]

    asyncio.run(scenario())


def test_host_cleanup_attempts_remaining_resources_and_retains_exact_handles(tmp_path):
    from examples.browser_acceptance import local_authenticated as example

    (tmp_path / "server.key").write_text("private-key-canary")
    calls = []
    primary = RuntimeError("trial failed")
    network_failure, child_failure = RuntimeError("inspect failed"), RuntimeError("remove failed")
    records = [
        {"kind": "network", "name": "first", "session": "first-session"},
        {"kind": "network", "name": "second", "session": "owned-session"},
        {"kind": "container", "name": "child-one", "session": "owned-session"},
        {"kind": "container", "name": "child-two", "session": "owned-session"},
    ]
    records += [{**record, "created": True} for record in records]
    (tmp_path / "resource-intents.jsonl").write_text("\n".join(map(json.dumps, records)))

    def docker(*args):
        calls.append(args)
        if args[:2] == ("network", "ls") and args[-1].endswith("first-session"):
            raise network_failure
        if args[:2] == ("network", "ls"):
            return "second"
        if args[0] == "ps":
            return "child-one\nchild-two"
        if args == ("rm", "-f", "child-one"):
            raise child_failure
        return ""

    with pytest.raises(BaseExceptionGroup) as error:
        example._retire_host(docker, "controller", tmp_path, primary)
    assert error.value.exceptions == (primary, child_failure, network_failure)
    assert ("rm", "-f", "child-two") in calls
    assert ("rm", "-f", "controller") in calls
    assert not (tmp_path / "server.key").exists()
    retained = json.loads((tmp_path / "cleanup-resources.json").read_text())
    assert retained == {
        "controller": "controller",
        "networks": ["first", "second"],
        "unverified_networks": ["first"],
        "containers": ["child-one", "child-two"],
    }


@pytest.mark.parametrize("inspection_fails", [False, True])
@pytest.mark.parametrize("removal_fails", [False, True])
def test_detached_campaign_resources_remain_owned(tmp_path, inspection_fails, removal_fails):
    from examples.browser_acceptance import local_authenticated as example

    records = [
        {"kind": "network", "name": "owned-network", "session": "campaign"},
        {"kind": "container", "name": "owned-sidecar", "session": "campaign"},
    ]
    records += [{**record, "created": True} for record in records]
    (tmp_path / "resource-intents.jsonl").write_text("\n".join(map(json.dumps, records)))
    calls = []
    failure = RuntimeError("removal unavailable")

    def docker(*args):
        calls.append(args)
        if args[0] == "inspect":
            if inspection_fails:
                raise RuntimeError("controller inspection unavailable")
            return json.dumps([{"NetworkSettings": {"Networks": {}}}])
        if args[0] == "ps":
            return "owned-sidecar\nnot-in-inventory"
        if args[:2] == ("network", "ls"):
            return "owned-network\nnot-in-inventory"
        if removal_fails and args in (
            ("rm", "-f", "owned-sidecar"),
            ("network", "rm", "owned-network"),
        ):
            raise failure
        return ""

    if removal_fails:
        with pytest.raises(BaseExceptionGroup):
            example._retire_host(docker, "controller", tmp_path, None)
        retained = json.loads((tmp_path / "cleanup-resources.json").read_text())
        assert retained["networks"] == ["owned-network"]
        assert retained["containers"] == ["owned-sidecar"]
    else:
        example._retire_host(docker, "controller", tmp_path, None)
    assert ("rm", "-f", "owned-sidecar") in calls
    assert ("network", "rm", "owned-network") in calls
    assert ("rm", "-f", "controller") in calls
    assert not any("not-in-inventory" in args for args in calls)
    assert not any(args[0] == "inspect" for args in calls)


@pytest.mark.parametrize(
    "malformed",
    [
        b'{"kind":',
        b"\xff",
        b"null",
        b'{"kind":"network"}',
        b'{"kind":"network","name":false,"session":"campaign"}',
    ],
)
@pytest.mark.parametrize("bad_first", [False, True])
@pytest.mark.parametrize("removal_fails", [False, True])
def test_invalid_journal_entry_does_not_hide_valid_resources(
    tmp_path, malformed, bad_first, removal_fails
):
    from examples.browser_acceptance import local_authenticated as example

    valid = json.dumps({"kind": "network", "name": "owned-network", "session": "campaign"}).encode()
    acknowledged = json.dumps({**json.loads(valid), "created": True}).encode()
    lines = [malformed, valid] if bad_first else [valid, malformed]
    lines.append(acknowledged)
    (tmp_path / "resource-intents.jsonl").write_bytes(b"\n".join(lines))
    calls = []
    removal_failure = RuntimeError("network removal failed")

    def docker(*args):
        calls.append(args)
        if args[:2] == ("network", "ls"):
            return "owned-network"
        if args[:2] == ("network", "rm") and removal_fails:
            raise removal_failure
        return ""

    with pytest.raises((RuntimeError, ExceptionGroup)) as caught:
        example._retire_host(docker, "controller", tmp_path, None)
    if removal_fails:
        assert caught.value.exceptions[1] is removal_failure
    assert ("network", "rm", "owned-network") in calls
    retained = json.loads((tmp_path / "cleanup-resources.json").read_text())
    assert retained["networks"] == ["owned-network"]
    assert retained["invalid_journal_lines"] == [1 if bad_first else 2]
    assert retained["containers"] == []


@pytest.mark.parametrize("write_fails", [False, True])
@pytest.mark.parametrize("kind", ["network", "container"])
def test_resource_intent_is_durable_before_docker_dispatch(
    tmp_path, monkeypatch, write_fails, kind
):
    from examples.browser_acceptance import local_authenticated as example

    calls = []

    async def dispatch(argv):
        record = json.loads((tmp_path / "resource-intents.jsonl").read_text())
        assert record == {"kind": kind, "name": "owned-resource", "session": "campaign"}
        calls.append(argv)
        return 0, ""

    if write_fails:

        def fsync(fd):
            raise OSError("inventory unavailable")

        monkeypatch.setattr(example.os, "fsync", fsync)
    execute = example._recording_docker_exec(tmp_path, dispatch)
    argv = (
        ["network", "create", "--label", "cayu.egress.session=campaign", "owned-resource"]
        if kind == "network"
        else ["run", "--name", "owned-resource", "--label", "cayu.egress.session=campaign", "image"]
    )
    if write_fails:
        with pytest.raises(OSError):
            asyncio.run(execute(argv))
        assert calls == []
    else:
        assert asyncio.run(execute(argv)) == (0, "")
        records = [
            json.loads(line)
            for line in (tmp_path / "resource-intents.jsonl").read_text().splitlines()
        ]
        assert records[1] == {**records[0], "created": True}


def test_host_retains_creation_that_finishes_after_empty_discovery(tmp_path):
    from examples.browser_acceptance import local_authenticated as example

    async def scenario():
        accepted, release = asyncio.Event(), asyncio.Event()
        resources = set()

        async def daemon():
            await release.wait()
            resources.add("owned-network")

        worker = asyncio.create_task(daemon())

        async def dispatch(argv):
            accepted.set()
            await asyncio.shield(worker)
            return 0, ""

        execute = example._recording_docker_exec(tmp_path, dispatch)
        owner = asyncio.create_task(
            execute(
                ["network", "create", "--label", "cayu.egress.session=campaign", "owned-network"]
            )
        )
        await accepted.wait()
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert owner.cancelled() and owner.cancelling() == 1

        def docker(*args):
            if args[:2] == ("network", "ls"):
                return "\n".join(resources)
            return ""

        try:
            with pytest.raises(RuntimeError, match="no durable acknowledgement"):
                example._retire_host(docker, "controller", tmp_path, None)
            retained = json.loads((tmp_path / "cleanup-resources.json").read_text())
            assert retained["unsettled_creations"] == [
                {"kind": "network", "name": "owned-network", "session": "campaign"}
            ]
        finally:
            release.set()
            await worker
        assert resources == {"owned-network"}

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_owner", [False, True])
def test_authenticated_setup_retires_startup_resources_preserving_owner_signal(
    monkeypatch, cancel_owner
):
    from examples.browser_acceptance import local_authenticated as example

    async def scenario():
        entered = asyncio.Event()
        closed = []
        primary = RuntimeError("startup failed")

        @asynccontextmanager
        async def failing_setup(cleanups):
            async def close_first():
                closed.append("first")

            async def close_second():
                closed.append("second")

            cleanups.extend((close_first, close_second))
            entered.set()
            if cancel_owner:
                await asyncio.Event().wait()
            raise primary
            yield  # pragma: no cover - establishes the context-manager protocol

        monkeypatch.setattr(example, "_setup", failing_setup)

        async def run():
            async with example.setup():
                raise AssertionError("Failed startup must never publish a plan")

        task = asyncio.create_task(run())
        await entered.wait()
        if cancel_owner:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled() and task.cancelling() == 1
        else:
            with pytest.raises(RuntimeError) as error:
                await task
            assert error.value is primary
            assert not task.cancelled() and task.cancelling() == 0
        assert closed == ["second", "first"]

    asyncio.run(scenario())
