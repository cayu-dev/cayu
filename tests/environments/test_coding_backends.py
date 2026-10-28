"""Coding backend selection, registration and the named-check helpers built on a profile."""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

import pytest
from pydantic import SecretStr
from tests.sandbox_broker.fake_broker import API_URL, APPLICATION, TOKEN, FakeBroker

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest, ToolContext
from cayu.environments import (
    CLOUD_CODING_BACKEND,
    CLOUD_SANDBOX_VARIABLES,
    CODING_BACKEND_ENV,
    DOCKER_CODING_BACKEND,
    CodingBackend,
    CodingBackendContext,
    CodingBackendRegistry,
    CodingCommandAuthority,
    CodingToolchainProfile,
    DockerCodingBackendOptions,
    DockerCodingEnvironmentFactory,
    LocalWorkspaceEnvironmentFactory,
    default_coding_backends,
    register_coding_environment,
    select_coding_backend,
)
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runners import DockerImageIdentity, ExecCommand
from cayu.runtime import ExecutionProfileMismatchError, check_manifest
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sandbox_broker import (
    BrokeredCodingBackendOptions,
    BrokeredSandboxCodingEnvironmentFactory,
)
from cayu.sandbox_broker.client import (
    BROKER_API_URL_ENV,
    BROKER_APPLICATION_ID_ENV,
    BROKER_TOKEN_ENV,
)
from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
)
from cayu.tools import (
    CommandPolicyDecision,
    CommandRequest,
    ExactNamedCheckCommandPolicy,
    SearchKnowledgeTool,
    named_checks_from_toolchain,
)
from cayu.workspaces import LocalWorkspace

_CLOUD = {
    "CAYU_CLOUD_API_URL": API_URL,
    "CAYU_CLOUD_APPLICATION_ID": APPLICATION,
    "CAYU_CLOUD_SANDBOX_TOKEN": TOKEN,
}
_IMAGE = DockerImageIdentity(reference="coding:local", content_digest="sha256:" + "a" * 64)
_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="test.coding_environment", behavior_version="1", implementation_version="1"
)


def _profile() -> CodingToolchainProfile:
    return CodingToolchainProfile(
        profile_id="backends",
        revision="1",
        command_authorities=(
            CodingCommandAuthority(
                selector="lint",
                revision="1",
                description="Lint.",
                exposure="named_check",
                executable="/opt/tools/ruff",
                fixed_arguments=("check", "."),
                max_arguments=0,
                timeout_seconds=30,
            ),
            CodingCommandAuthority(
                selector="test",
                revision="1",
                description="Test.",
                exposure="named_check",
                executable="/opt/tools/pytest",
                fixed_arguments=("-q",),
                max_arguments=0,
            ),
        ),
    )


def _docker_options() -> dict[str, object]:
    return {
        DOCKER_CODING_BACKEND: DockerCodingBackendOptions(
            image_identity=_IMAGE, platform_architecture="amd64", verify_local_image=False
        )
    }


def test_cloud_broker_variables_match_the_broker_client() -> None:
    assert CLOUD_SANDBOX_VARIABLES == (
        BROKER_API_URL_ENV,
        BROKER_APPLICATION_ID_ENV,
        BROKER_TOKEN_ENV,
    )


def test_selection_follows_the_cloud_variables_and_a_required_name() -> None:
    registry = default_coding_backends()

    assert registry.names() == ("cayu-cloud", "docker")
    assert registry.selected_name({}) == DOCKER_CODING_BACKEND
    assert registry.selected_name(_CLOUD) == CLOUD_CODING_BACKEND
    # Cloud's agent registry and the cayu cloud CLI set these two without a broker.
    registry_only = {k: v for k, v in _CLOUD.items() if k != "CAYU_CLOUD_SANDBOX_TOKEN"}
    assert registry.selected_name(registry_only) == DOCKER_CODING_BACKEND
    assert registry.selected_name({"CAYU_CLOUD_API_URL": API_URL}) == DOCKER_CODING_BACKEND
    # The sandbox token means a broker is intended, so the rest must be there too.
    token_only = {"CAYU_CLOUD_SANDBOX_TOKEN": _CLOUD["CAYU_CLOUD_SANDBOX_TOKEN"]}
    with pytest.raises(ValueError) as incomplete:
        registry.selected_name(token_only)
    assert "CAYU_CLOUD_API_URL, CAYU_CLOUD_APPLICATION_ID are unset" in str(incomplete.value)

    # A deployment that requires Cloud fails at startup instead of falling back to Docker.
    with pytest.raises(ValueError, match="requires the 'cayu-cloud' coding backend, but"):
        registry.selected_name({CODING_BACKEND_ENV: "cayu-cloud"})
    with pytest.raises(ValueError, match="CAYU_CLOUD_SANDBOX_TOKEN is unset"):
        registry.selected_name(
            {k: v for k, v in _CLOUD.items() if k != "CAYU_CLOUD_SANDBOX_TOKEN"},
            required="cayu-cloud",
        )
    assert registry.selected_name(_CLOUD, required="docker") == DOCKER_CODING_BACKEND
    assert registry.selected_name({**_CLOUD, CODING_BACKEND_ENV: "docker"}) == "docker"
    with pytest.raises(ValueError, match="conflicts"):
        registry.selected_name({CODING_BACKEND_ENV: "cayu-cloud"}, required="docker")
    with pytest.raises(ValueError, match="isn't registered"):
        registry.selected_name({CODING_BACKEND_ENV: "lambda-microvm"})


def test_other_backends_register_per_registry_and_select_by_name_only(tmp_path: Path) -> None:
    def local_backend(context: CodingBackendContext) -> CodingBackend:
        def build(source, *, artifact_store=None):
            del artifact_store
            return LocalWorkspaceEnvironmentFactory(allowed_roots=(source.root,))

        return CodingBackend(
            name="local-test",
            candidate="docker",
            toolchain_profile=context.toolchain_profile,
            build_factory=build,
        )

    registry = default_coding_backends()
    registry.register("local-test", local_backend)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("local-test", local_backend)
    registry.register("local-test", local_backend, replace=True)

    assert registry.selected_name({}) == DOCKER_CODING_BACKEND
    backend = registry.build(
        toolchain_profile=_profile(), environ={CODING_BACKEND_ENV: "local-test"}
    )
    assert backend.name == "local-test"
    # Registries are independent: the built-in set doesn't see this one.
    assert "local-test" not in default_coding_backends().names()
    assert "local-test" not in CodingBackendRegistry().names()

    with pytest.raises(ValueError, match="no coding toolchain evidence policy"):
        CodingBackend(
            name="unknown",
            candidate="unknown-sandbox",
            toolchain_profile=_profile(),
            build_factory=lambda source, *, artifact_store=None: None,
        )


def test_docker_backend_runs_the_neutral_profile_at_dockers_bar(tmp_path: Path) -> None:
    profile = _profile()
    backend = select_coding_backend(
        toolchain_profile=profile, options=_docker_options(), environ={}
    )

    factory = backend.factory_for(LocalWorkspace(tmp_path, workspace_id="source"))
    assert isinstance(factory, DockerCodingEnvironmentFactory)
    assert factory.coding_toolchain_profile == profile
    requirements = backend.execution_requirements(real_secret_visibility="non_possession")
    assert requirements.minimum_evidence == "live_verified"
    assert requirements.network_access == "deny_by_default"
    assert requirements.code_trust == "trusted"
    assert requirements.required_executables == profile.required_executables
    assert backend.metadata["image_fingerprint"] == _IMAGE.fingerprint

    with pytest.raises(ValueError, match="DockerCodingBackendOptions"):
        select_coding_backend(toolchain_profile=profile, environ={})


def test_docker_backend_passes_its_cancel_timeout_to_every_factory(tmp_path: Path) -> None:
    options = DockerCodingBackendOptions(
        image_identity=_IMAGE,
        platform_architecture="amd64",
        verify_local_image=False,
        cancel_timeout_s=30.0,
    )
    backend = select_coding_backend(
        toolchain_profile=_profile(), options={DOCKER_CODING_BACKEND: options}, environ={}
    )

    factory = backend.factory_for(LocalWorkspace(tmp_path, workspace_id="source"))
    assert factory.cancel_timeout_s == 30.0
    with pytest.raises(ValueError):
        DockerCodingBackendOptions(image_identity=_IMAGE, cancel_timeout_s=0.0)


def test_cloud_backend_runs_the_neutral_profile_at_the_brokers_bar(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    profile = _profile()
    backend = select_coding_backend(
        toolchain_profile=profile,
        options={
            **_docker_options(),
            CLOUD_CODING_BACKEND: BrokeredCodingBackendOptions(
                image="release", broker=broker.client(), queue_seconds=120
            ),
        },
        environ=_CLOUD,
    )

    factory = backend.factory_for(LocalWorkspace(tmp_path, workspace_id="source"))
    assert isinstance(factory, BrokeredSandboxCodingEnvironmentFactory)
    assert (factory.egress, factory.image) == ("none", "release")
    assert factory.queue_seconds == 120
    assert factory.toolchain_profile == profile
    assert backend.candidate == "cayu-cloud-sandbox"
    assert backend.execution_requirements().minimum_evidence == "available"

    defaults = BrokeredCodingBackendOptions.from_env(
        {**_CLOUD, "CAYU_CLOUD_SANDBOX_IMAGE": "release", "CAYU_CLOUD_SANDBOX_EGRESS": "none"}
    )
    assert defaults.image == "release"
    with pytest.raises(ValueError, match="prohibits the network"):
        BrokeredCodingBackendOptions.from_env({"CAYU_CLOUD_SANDBOX_EGRESS": "public_web"})
    with pytest.raises(ValueError, match="prohibits the network"):
        BrokeredCodingBackendOptions.from_env({"CAYU_CLOUD_SANDBOX_EGRESS": "allowlist"})
    with pytest.raises(ValueError, match="CAYU_CLOUD_SANDBOX_EGRESS_HOSTS"):
        BrokeredCodingBackendOptions.from_env({"CAYU_CLOUD_SANDBOX_EGRESS_HOSTS": "pypi.org"})
    from_env = select_coding_backend(toolchain_profile=profile, environ=_CLOUD)
    assert from_env.metadata["sandbox_image"] == "coding-python"


def test_register_coding_environment_adds_one_environment_per_workspace(tmp_path: Path) -> None:
    backend = select_coding_backend(
        toolchain_profile=_profile(), options=_docker_options(), environ={}
    )
    app = CayuApp(enable_logging=False)
    roots = []
    for index in range(2):
        root = tmp_path / f"slot-{index}"
        root.mkdir()
        roots.append(root)
        register_coding_environment(
            app,
            name=f"slot-{index}",
            backend=backend,
            source_workspace=LocalWorkspace(root, workspace_id=f"slot-{index}"),
            default=index == 0,
            execution_profile_identity=_IDENTITY,
            metadata={"slot": index},
        )

    registrations = app._environment_registry.registrations
    assert set(registrations) == {"slot-0", "slot-1"}
    for index, root in enumerate(roots):
        registered = registrations[f"slot-{index}"]
        assert registered.factory.source_workspace.root == root.resolve()
        metadata = registered.spec.metadata
        assert metadata["coding_backend"] == "docker"
        assert metadata["slot"] == index
        assert metadata["toolchain_profile_fingerprint"] == backend.toolchain_profile.fingerprint
        assert metadata["factory_profile_identity"]["name"] == "cayu.docker_coding_environment"


def test_named_checks_and_their_exact_policy_come_from_the_profile() -> None:
    profile = _profile()
    lint_identity = ExecutionProfileBehaviorIdentity(
        name="app.check.lint", behavior_version="1", implementation_version="1"
    )

    checks = named_checks_from_toolchain(
        profile, identities={"lint": lint_identity}, identity_prefix="app.check"
    )
    assert [check.name for check in checks] == ["lint", "test"]
    assert checks[0].execution_profile_identity == lint_identity
    assert checks[1].execution_profile_identity.name == "app.check.test"
    assert checks[1].execution_profile_identity.implementation_version == (
        profile.command_authority("test").fingerprint
    )
    assert tuple(checks[0].command.argv or ()) == ("/opt/tools/ruff", "check", ".")
    with pytest.raises(ValueError, match="doesn't declare: format"):
        named_checks_from_toolchain(profile, identities={"format": lint_identity})

    policy = ExactNamedCheckCommandPolicy(checks, workspace_path="/src")
    context = ToolContext(session_id="s", agent_name="a", environment_name="coding")

    def decision(argv: tuple[str, ...], *, timeout: int = 30, cwd: str = "/src"):
        request = CommandRequest(
            command=ExecCommand.process(*argv),
            cwd=None,
            canonical_cwd=cwd,
            env=None,
            timeout_s=timeout,
            stdin=None,
        )
        return asyncio.run(policy.evaluate(context, request)).decision

    assert decision(("/opt/tools/ruff", "check", ".")) is CommandPolicyDecision.ALLOW
    assert decision(("/opt/tools/ruff", "check", "--fix", ".")) is CommandPolicyDecision.DENY
    assert decision(("/opt/tools/ruff", "check", "."), timeout=60) is CommandPolicyDecision.DENY
    assert decision(("/opt/tools/ruff", "check", "."), cwd="/workspace") is (
        CommandPolicyDecision.DENY
    )
    assert policy.execution_profile_identity.name == "cayu.exact_named_check_command_policy"
    assert ExactNamedCheckCommandPolicy(checks).execution_profile_identity != (
        policy.execution_profile_identity
    )


def test_a_session_started_on_one_backend_is_not_resumed_on_another(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("project\n", encoding="utf-8")
    profile = CodingToolchainProfile(profile_id="resume", revision="1")
    keyring = PublicAuthorityAliasKeyring(
        active_key_id="test",
        keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
    )

    def app_on(backend: CodingBackend, store: SQLiteSessionStore) -> CayuApp:
        app = CayuApp(
            session_store=store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            ),
            default=True,
        )
        register_coding_environment(
            app,
            name="coding",
            backend=backend,
            source_workspace=LocalWorkspace(
                source, workspace_id="source", excluded_directory_names=(".git",)
            ),
            default=True,
            execution_profile_identity=_IDENTITY,
        )
        app.register_agent(
            AgentSpec(name="coder", model="scripted-model"),
            tools=[],
            execution_requirements=backend.execution_requirements(),
        )
        return app

    cloud = select_coding_backend(
        toolchain_profile=profile,
        options={
            CLOUD_CODING_BACKEND: BrokeredCodingBackendOptions(
                broker=broker.client(), git_baseline=False
            )
        },
        environ=_CLOUD,
    )
    docker = select_coding_backend(toolchain_profile=profile, options=_docker_options(), environ={})

    async def scenario():
        store = SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=PublicAuthorityAliasCodec(keyring),
        )
        try:
            first = [
                event
                async for event in app_on(cloud, store).run(
                    RunRequest(
                        session_id="coding",
                        agent_name="coder",
                        messages=[Message.text("user", "one")],
                    )
                )
            ]
            with pytest.raises(ExecutionProfileMismatchError) as refused:
                async for _ in app_on(docker, store).resume(
                    ResumeRequest(session_id="coding", messages=[Message.text("user", "two")])
                ):
                    pass
            return first, refused.value
        finally:
            await store.close()

    first, refused = asyncio.run(scenario())
    assert first[-1].type is EventType.SESSION_COMPLETED, first[-1].payload
    assert len(broker.sandboxes) == 1
    # The factories' identities differ, so the execution profile no longer matches.
    assert "execution_environment" in str(refused)


def _workspace(tmp_path: Path, name: str) -> LocalWorkspace:
    root = tmp_path / name
    root.mkdir()
    return LocalWorkspace(root, workspace_id=name)


def test_coding_environments_get_knowledge_only_when_given_or_asked_for(
    tmp_path: Path,
) -> None:
    backend = select_coding_backend(
        toolchain_profile=_profile(), options=_docker_options(), environ={}
    )
    store = InMemoryKnowledgeStore()
    scope = KnowledgeAccessScope.for_namespace("project:app")
    narrower = KnowledgeAccessScope.for_namespace("project:app", required_labels={"team": "a"})
    app = CayuApp(enable_logging=False, knowledge_store=store, knowledge_access_scope=scope)

    def register(name: str, **options: object):
        return register_coding_environment(
            app, name=name, backend=backend, source_workspace=_workspace(tmp_path, name), **options
        )

    # The app's scope may span every tenant, so a workspace never gets it by default.
    default = register("default")
    assert default.configured_knowledge_store is None
    assert default.configured_knowledge_access_scope is None
    inherited = register("inherited", inherit_app_knowledge=True)
    assert inherited.configured_knowledge_store is store
    assert inherited.configured_knowledge_access_scope == scope
    own_store = InMemoryKnowledgeStore()
    own = register("own", knowledge_store=own_store, knowledge_access_scope=narrower)
    assert own.configured_knowledge_store is own_store
    assert own.configured_knowledge_access_scope == narrower
    narrowed = register("narrowed", knowledge_store=store, knowledge_access_scope=narrower)
    assert narrowed.configured_knowledge_store is store
    assert narrowed.configured_knowledge_access_scope == narrower
    with pytest.raises(ValueError, match="needs the knowledge_store"):
        register("scope-only", knowledge_access_scope=narrower)
    with pytest.raises(ValueError, match="not both"):
        register("both", knowledge_store=store, inherit_app_knowledge=True)
    with pytest.raises(TypeError, match="inherit_app_knowledge"):
        register("not-a-bool", inherit_app_knowledge="yes")
    assert set(app._environment_registry.registrations) == {
        "default",
        "inherited",
        "own",
        "narrowed",
    }

    # The factory hands out copies, so changing one can't widen later sessions.
    handed_out = narrowed.configured_knowledge_access_scope
    handed_out.required_labels.clear()
    handed_out.allowed_namespaces.append("project:other")
    assert narrowed.configured_knowledge_access_scope == narrower

    with pytest.raises(ValueError, match="needs an app built with knowledge_store"):
        register_coding_environment(
            CayuApp(enable_logging=False),
            name="plain",
            backend=backend,
            source_workspace=_workspace(tmp_path, "plain"),
            inherit_app_knowledge=True,
        )


def test_factory_for_passes_knowledge_to_each_backends_factory(tmp_path: Path) -> None:
    store = InMemoryKnowledgeStore()
    scope = KnowledgeAccessScope.for_namespace("project:app")
    docker = select_coding_backend(
        toolchain_profile=_profile(), options=_docker_options(), environ={}
    )
    cloud = select_coding_backend(
        toolchain_profile=CodingToolchainProfile(profile_id="cloud", revision="1"),
        options={
            CLOUD_CODING_BACKEND: BrokeredCodingBackendOptions(
                broker=FakeBroker(tmp_path / "cloud").client()
            )
        },
        environ=_CLOUD,
    )
    for backend in (docker, cloud):
        workspace = _workspace(tmp_path, backend.name)
        factory = backend.factory_for(
            workspace, knowledge_store=store, knowledge_access_scope=scope
        )
        assert factory.configured_knowledge_store is store
        assert factory.configured_knowledge_access_scope == scope
        assert backend.factory_for(workspace).configured_knowledge_store is None
        with pytest.raises(ValueError, match="needs the knowledge_store"):
            backend.factory_for(workspace, knowledge_access_scope=scope)


def test_a_builder_without_knowledge_keeps_working_and_one_that_drops_it_is_refused(
    tmp_path: Path,
) -> None:
    docker = select_coding_backend(
        toolchain_profile=_profile(), options=_docker_options(), environ={}
    )

    def without_knowledge(source_workspace, *, artifact_store=None):
        return docker.factory_for(source_workspace, artifact_store=artifact_store)

    def dropping_knowledge(source_workspace, *, artifact_store=None, **knowledge):
        del knowledge
        return docker.factory_for(source_workspace, artifact_store=artifact_store)

    workspace = _workspace(tmp_path, "source")
    older = CodingBackend(
        name="older",
        candidate="docker",
        toolchain_profile=_profile(),
        build_factory=without_knowledge,
    )
    assert isinstance(older.factory_for(workspace), DockerCodingEnvironmentFactory)
    dropping = CodingBackend(
        name="dropping",
        candidate="docker",
        toolchain_profile=_profile(),
        build_factory=dropping_knowledge,
    )
    with pytest.raises(TypeError, match="dropped the knowledge_store"):
        dropping.factory_for(workspace, knowledge_store=InMemoryKnowledgeStore())


def test_registering_an_agent_warns_when_its_knowledge_tools_have_no_store(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    backend = select_coding_backend(
        toolchain_profile=_profile(), options=_docker_options(), environ={}
    )

    def app_with(name: str, **options: object) -> CayuApp:
        app = CayuApp(
            enable_logging=False,
            knowledge_store=InMemoryKnowledgeStore(),
            knowledge_access_scope=KnowledgeAccessScope.for_namespace("project:app"),
        )
        app.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
        register_coding_environment(
            app,
            name="coding",
            backend=backend,
            source_workspace=_workspace(tmp_path, name),
            default=True,
            **options,
        )
        app.register_agent(
            AgentSpec(name="coder", model="scripted-model"),
            tools=[SearchKnowledgeTool()],
            execution_requirements=backend.execution_requirements(),
        )
        return app

    with caplog.at_level(logging.WARNING, logger="cayu._application_agent_registry"):
        without = app_with("without")
    warnings = [
        r.getMessage() for r in caplog.records if "missing_knowledge_store" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "Agent coder has search_knowledge" in warnings[0]
    findings = [
        item
        for item in check_manifest(without.describe()).diagnostics
        if item.code == "TOOL_KNOWLEDGE_STORE_UNAVAILABLE"
    ]
    assert [item.subject for item in findings] == ["tool:coder/search_knowledge"]

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="cayu._application_agent_registry"):
        inherited = app_with("inherited", inherit_app_knowledge=True)
    assert not [r for r in caplog.records if "missing_knowledge_store" in r.getMessage()]
    assert "TOOL_KNOWLEDGE_STORE_UNAVAILABLE" not in {
        item.code for item in check_manifest(inherited.describe()).diagnostics
    }


def test_search_knowledge_finds_the_app_entry_in_a_brokered_coding_session(
    tmp_path: Path,
) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("project\n", encoding="utf-8")
    backend = select_coding_backend(
        toolchain_profile=CodingToolchainProfile(profile_id="knowledge", revision="1"),
        options={
            CLOUD_CODING_BACKEND: BrokeredCodingBackendOptions(
                broker=broker.client(), git_baseline=False
            )
        },
        environ=_CLOUD,
    )
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="search", name="search_knowledge", arguments={"query": "releases"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Tagged from main."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )

    async def scenario():
        store = SQLiteKnowledgeStore(
            tmp_path / "knowledge.db",
            access_scope=KnowledgeAccessScope.for_namespace("default"),
        )
        try:
            await store.create_entry(
                KnowledgeEntry(id="release-guide", text="Tag releases from main only."),
                [
                    KnowledgeChunk(
                        id="release-guide-0",
                        entry_id="release-guide",
                        text="Tag releases from main only.",
                        chunk_index=0,
                    )
                ],
            )
            # The app's scope is the one bound to the store.
            app = CayuApp(enable_logging=False, knowledge_store=store)
            app.register_provider(provider, default=True)
            register_coding_environment(
                app,
                name="coding",
                backend=backend,
                source_workspace=LocalWorkspace(
                    source, workspace_id="source", excluded_directory_names=(".git",)
                ),
                default=True,
                execution_profile_identity=_IDENTITY,
                inherit_app_knowledge=True,
            )
            app.register_agent(
                AgentSpec(name="coder", model="scripted-model"),
                tools=[SearchKnowledgeTool()],
                execution_requirements=backend.execution_requirements(),
            )
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="knowledge",
                        agent_name="coder",
                        messages=[Message.text("user", "How are releases tagged?")],
                    )
                )
            ]
        finally:
            await store.close()

    events = asyncio.run(scenario())
    assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
    searched = [
        event
        for event in events
        if event.type is EventType.TOOL_CALL_COMPLETED and event.tool_name == "search_knowledge"
    ]
    assert len(searched) == 1
    assert "Tag releases from main only." in searched[0].payload["result"]["content"]
    assert len(broker.sandboxes) == 1
