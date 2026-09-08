from __future__ import annotations

import asyncio
import importlib
import itertools
import json
import os
import subprocess
import sys
from functools import wraps
from pathlib import Path

import pytest

from cayu import (
    DockerImageIdentity,
    EventType,
    InMemoryKnowledgeStore,
    InMemoryTaskStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeListQuery,
    KnowledgeStatus,
    Message,
    ModelStreamEvent,
    PendingToolApprovalEventView,
    RunRequest,
    ScriptedModelProvider,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
    SQLiteTaskStore,
    ToolApprovalDecision,
    ToolApprovalRequest,
    run_to_completion,
)
from cayu.cli import main
from cayu.cli.project import project_context
from cayu.cli.scaffold import project_files
from cayu.cli.scaffold_check import check_scaffold_capabilities


@pytest.fixture(autouse=True)
def close_constructed_sqlite_stores(monkeypatch):
    """Own stores from direct factories and CLI boots until each test finishes.

    Construction opens SQLite connections even when no session is run. Keeping
    and closing those concrete stores also prevents their delayed GC warnings
    from contaminating an unrelated test's diagnostic assertions.
    """
    owned = []

    def track(store_type):
        original = store_type.__init__

        @wraps(original)
        def initialize(store, *args, **kwargs):
            original(store, *args, **kwargs)
            owned.append(store)

        monkeypatch.setattr(store_type, "__init__", initialize)

    for store_type in (SQLiteSessionStore, SQLiteTaskStore, SQLiteKnowledgeStore):
        track(store_type)
    try:
        yield
    finally:

        async def close():
            for store in reversed(owned):
                await store.close()

        asyncio.run(close())


@pytest.mark.parametrize("dry_run", (False, True))
@pytest.mark.parametrize(
    "preset,execution,excluded",
    (
        ("service", "none", "tasks"),
        ("coding", "docker", "artifacts"),
    ),
)
def test_impossible_exclusions_rejected_before_publication(
    tmp_path,
    capsys,
    dry_run,
    preset,
    execution,
    excluded,
):
    command = [
        "new",
        "impossible",
        "--dir",
        str(tmp_path),
        "--preset",
        preset,
        "--execution",
        execution,
        "--without",
        excluded,
        "--json",
    ]
    if dry_run:
        command.append("--dry-run")
    assert main(command) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "CAPABILITY_REQUIRED"
    assert list(tmp_path.iterdir()) == []


def _write_project(root: Path, **options) -> Path:
    root.mkdir()
    for relative, content in project_files("profile", **options).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    # Explicit temporary repository for the coding factory's real Git checks.
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


@pytest.mark.parametrize("preset", ("agent", "service", "coding"))
@pytest.mark.parametrize("database", ("sqlite", "postgres"))
def test_rendered_capability_owners_are_formatted(preset, database):
    if preset == "service" and database == "postgres":
        pytest.skip("Postgres service is not a supported scaffold profile")
    files = project_files("profile", preset=preset, database=database)
    paths = ["environments/local.py", "configuration/storage.py"]
    if preset == "coding":
        paths += ["configuration/coding_storage.py", "operations/coding.py", "tools/coding.py"]
    for path in paths:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--isolated",
                "--check",
                "--stdin-filename",
                path,
            ],
            input=files[path],
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, path + ": " + result.stdout + result.stderr


_EXCLUSIONS = (
    ("knowledge",),
    ("tasks", "delegation", "human-input", "artifacts", "knowledge"),
    *(
        combination
        for count in range(5)
        for combination in itertools.combinations(
            ("tasks", "delegation", "human-input", "artifacts"), count
        )
    ),
)


@pytest.mark.parametrize("database", ("sqlite", "postgres"))
@pytest.mark.parametrize("excluded", _EXCLUSIONS)
def test_coding_local_capability_matrix(tmp_path, monkeypatch, database, excluded):
    project = _write_project(
        tmp_path / "profile", preset="coding", database=database, without_capabilities=excluded
    )
    with project_context(project):
        coding = importlib.import_module("operations.coding")
        monkeypatch.setattr(coding, "_verify_coding_dependencies", lambda root: None)
        app = importlib.import_module("app").build_app()
        manifest = app.describe(project_root=project)
        diagnostics = check_scaffold_capabilities(project, manifest)
        assert diagnostics == (), [(item.path, dict(item.parameters)) for item in diagnostics]
        assert (manifest.stores.task is None) == ("tasks" in excluded)
        assert (manifest.environments[0].artifact_store is None) == ("artifacts" in excluded)
        assert len(manifest.agents) == (1 if "delegation" in excluded else 2)
        tools = {tool.name for agent in manifest.agents for tool in agent.tools}
        for capability, names in {
            "delegation": {"subagent", "subagent_result"},
            "human-input": {"ask_user"},
            "artifacts": {"list_artifacts"},
        }.items():
            assert names.isdisjoint(tools) if capability in excluded else names <= tools
        if "tasks" in excluded:
            with pytest.raises(ValueError, match="tasks capability"):
                importlib.import_module("app").build_app(task_store=InMemoryTaskStore())


@pytest.mark.parametrize("database", ("sqlite", "postgres"))
@pytest.mark.parametrize(
    "excluded",
    (
        (),
        ("tasks",),
        ("delegation",),
        ("human-input",),
        ("tasks", "delegation", "human-input", "knowledge"),
    ),
)
def test_coding_docker_capability_matrix(tmp_path, monkeypatch, database, excluded):
    project = _write_project(
        tmp_path / "profile",
        preset="coding",
        execution="docker",
        database=database,
        without_capabilities=excluded,
    )
    with project_context(project):
        coding = importlib.import_module("operations.coding")
        monkeypatch.setattr(coding, "_verify_coding_dependencies", lambda root: None)
        monkeypatch.setattr(
            coding,
            "_configured_docker_authority",
            lambda root: (
                DockerImageIdentity(reference="test:profile", content_digest="sha256:" + "a" * 64),
                "/usr/bin/docker",
            ),
        )
        app = importlib.import_module("app").build_app()
        assert check_scaffold_capabilities(project, app.describe()) == ()


@pytest.mark.parametrize("database", ("sqlite", "postgres"))
@pytest.mark.parametrize("preset", ("agent", "coding"))
def test_excluded_knowledge_cannot_be_restored_through_storage_seam(tmp_path, database, preset):
    project = _write_project(
        tmp_path / "profile",
        preset=preset,
        database=database,
        without_capabilities=("memory", "knowledge") if preset == "agent" else ("knowledge",),
    )
    with project_context(project):
        scope = KnowledgeAccessScope.for_namespace("default")
        storage = importlib.import_module(
            "configuration.coding_storage" if preset == "coding" else "configuration.storage"
        )
        with pytest.raises(ValueError, match="knowledge capability"):
            if preset == "coding":
                storage.build_coding_stores(tmp_path / "state", scope)
            else:
                storage.build_stores(knowledge_scope=scope)
        with pytest.raises(ValueError, match="knowledge capability"):
            importlib.import_module("app").build_app(knowledge_store=InMemoryKnowledgeStore())
        if preset == "agent":
            with pytest.raises(ValueError, match="knowledge capability"):
                importlib.import_module("environments.local").build_local_environment(
                    artifact_store=None,
                    knowledge_store=InMemoryKnowledgeStore(),
                    knowledge_scope=scope,
                )
    assert not (tmp_path / "state").exists()
    assert not (project / "data").exists()


@pytest.mark.parametrize(
    "preset,excluded",
    (
        ("agent", ()),
        ("agent", ("tasks",)),
        ("agent", ("memory", "knowledge")),
        ("agent", ("artifacts",)),
        ("agent", ("human-input",)),
        ("agent", ("observability",)),
        ("agent", ("approvals",)),
        ("service", ()),
        ("service", ("approvals", "observability", "evals")),
    ),
)
def test_agent_service_capability_matrix(tmp_path, capsys, monkeypatch, preset, excluded):
    if preset == "service":
        monkeypatch.setenv(
            "PRODUCT_AUTH_TOKENS_JSON",
            '{"test-customer":{"tenant_id":"tenant-a","subject_id":"test-user"}}',
        )
        monkeypatch.setenv("CAYU_OPERATOR_BEARER_TOKEN", "test-operator")
    command = ["new", "profile", "--dir", str(tmp_path), "--preset", preset]
    for name in excluded:
        command.extend(("--without", name))
    assert main(command) == 0
    capsys.readouterr()
    monkeypatch.chdir(tmp_path / "profile")
    assert main(["inspect", "--json"]) == 0
    capsys.readouterr()
    assert main(["check", "--fail-on", "warning", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []


@pytest.mark.parametrize("command", ("inspect", "check"))
@pytest.mark.parametrize("add", (False, True))
@pytest.mark.parametrize("before", (False, True))
def test_cli_preserves_starter_coverage_with_additional_validation(
    tmp_path, capsys, monkeypatch, command, add, before
):
    args = ["new", "profile", "--dir", str(tmp_path)]
    if add:
        args.extend(("--without", "approvals"))
    assert main(args) == 0
    capsys.readouterr()
    project = tmp_path / "profile"
    policy = project / "policies/tools.py"
    source = policy.read_text()
    catch_all = 'DenyPatternRule("text", patterns=(r"(?s).*",))'
    validation = 'RequiredFieldRule("text")'
    assert catch_all in source
    sequence = f"{validation}, {catch_all}" if before else f"{catch_all}, {validation}"
    policy.write_text(source.replace(catch_all, sequence))
    monkeypatch.chdir(project)
    check_args = [command, "--json"]
    if command == "check":
        check_args.extend(("--fail-on", "warning"))
    assert main(check_args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("diagnostics", []) == []
    if command == "inspect":
        proposal = next(
            tool for tool in payload["agents"][0]["tools"] if tool["name"] == "remember_knowledge"
        )
        assert proposal["policy_coverage"] == ("denied" if add else "approval_required")


@pytest.mark.parametrize("command", ("inspect", "check"))
@pytest.mark.parametrize("add", (False, True))
@pytest.mark.parametrize("annotated", (False, True))
def test_cli_checks_starter_rule_coverage(tmp_path, capsys, monkeypatch, command, add, annotated):
    args = ["new", "profile", "--dir", str(tmp_path)]
    if add:
        args.extend(("--without", "approvals"))
    assert main(args) == 0
    capsys.readouterr()
    project = tmp_path / "profile"
    if annotated:
        declaration = project / "agents/agent.py"
        declaration.write_text(declaration.read_text().replace("AGENT =", "AGENT: AgentSpec ="))
    monkeypatch.chdir(project)
    check_args = [command, "--json"]
    if command == "check":
        check_args.extend(("--fail-on", "warning"))
    assert main(check_args) == 0
    capsys.readouterr()
    policy = project / "policies/tools.py"
    source = policy.read_text()
    original = 'DenyPatternRule("text", patterns=(r"(?s).*",))'
    assert original in source
    policy.write_text(source.replace(original, 'RequiredFieldRule("text")'))
    assert main(check_args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert any(
        item["code"] == "SCAFFOLD_CAPABILITY_DRIFT"
        and item["parameters"]["capability"] == "approvals"
        and item["parameters"]["coverage"] == "conditional"
        for item in payload["diagnostics"]
    )


@pytest.mark.parametrize("command", ("inspect", "check"))
@pytest.mark.parametrize("add", (False, True))
def test_cli_reports_starter_approval_decision_drift(tmp_path, capsys, monkeypatch, command, add):
    args = ["new", "profile", "--dir", str(tmp_path)]
    if add:
        args.extend(("--without", "approvals"))
    assert main(args) == 0
    capsys.readouterr()
    project = tmp_path / "profile"
    monkeypatch.chdir(project)
    check_args = [command, "--json"]
    if command == "check":
        check_args.extend(("--fail-on", "warning"))
    assert main(check_args) == 0
    capsys.readouterr()
    policy = project / "policies/tools.py"
    old = "ToolPolicyDecision.DENY" if add else "ToolPolicyDecision.REQUIRE_APPROVAL"
    new = "ToolPolicyDecision.REQUIRE_APPROVAL" if add else "ToolPolicyDecision.DENY"
    source = policy.read_text()
    assert old in source
    policy.write_text(source.replace(old, new))
    assert main(check_args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert any(
        item["code"] == "SCAFFOLD_CAPABILITY_DRIFT"
        and item["parameters"]["capability"] == "approvals"
        and item["parameters"]["observed"] == ("require_approval" if add else "deny")
        for item in payload["diagnostics"]
    )


@pytest.mark.parametrize("command", ("inspect", "check"))
@pytest.mark.parametrize("add", (False, True))
def test_cli_reports_live_capability_drift(tmp_path, capsys, monkeypatch, command, add):
    args = ["new", "profile", "--dir", str(tmp_path)]
    if add:
        args.extend(("--without", "human-input"))
    assert main(args) == 0
    capsys.readouterr()
    project = tmp_path / "profile"
    tools = project / "tools/registration.py"
    source = tools.read_text()
    tools.write_text(
        source.replace(
            "    return tuple(tools)",
            "    tools.append(UserInputTool())\n    return tuple(tools)"
            if add
            else '    return tuple(tool for tool in tools if tool.name != "ask_user")',
        )
    )
    monkeypatch.chdir(project)
    assert main([command, "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert any(
        item["code"] == "SCAFFOLD_CAPABILITY_DRIFT"
        and item["parameters"]["capability"] == "human-input"
        for item in payload["diagnostics"]
    )


@pytest.mark.parametrize("command", ("inspect", "check"))
@pytest.mark.parametrize("add", (False, True))
def test_cli_reports_live_task_store_drift(tmp_path, capsys, monkeypatch, command, add):
    args = ["new", "profile", "--dir", str(tmp_path)]
    if add:
        args.extend(("--without", "tasks"))
    assert main(args) == 0
    capsys.readouterr()
    project = tmp_path / "profile"
    storage = project / "configuration/storage.py"
    source = storage.read_text()
    old = f'SQLiteTaskStore("data/cayu.db") if {not add} else None'
    new = f'SQLiteTaskStore("data/cayu.db") if {add} else None'
    assert old in source
    storage.write_text(source.replace(old, new))
    monkeypatch.chdir(project)
    assert main([command, "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert any(
        item["code"] == "SCAFFOLD_CAPABILITY_DRIFT" and item["path"] == "stores.task"
        for item in payload["diagnostics"]
    )


def test_inspect_does_not_treat_invalid_capability_metadata_as_freeform(
    tmp_path, capsys, monkeypatch
):
    assert main(["new", "profile", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "profile"
    config = project / "pyproject.toml"
    config.write_text(config.read_text().replace('"tasks"]', '"unknown-capability"]'))
    monkeypatch.chdir(project)
    assert main(["inspect", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "SCAFFOLD_CONTRACT_INVALID"


@pytest.mark.parametrize("approvals", (False, True))
@pytest.mark.parametrize("order", ("default", "before", "after"))
def test_generated_sqlite_proposal_runtime_matches_approval_coverage(
    tmp_path, capsys, approvals, order
):
    args = ["new", "profile", "--dir", str(tmp_path)]
    if not approvals:
        args.extend(("--without", "approvals"))
    assert main(args) == 0
    capsys.readouterr()
    project = tmp_path / "profile"
    if order != "default":
        policy = project / "policies/tools.py"
        catch_all = 'DenyPatternRule("text", patterns=(r"(?s).*",))'
        extra = 'RequiredFieldRule("text")'
        sequence = f"{extra}, {catch_all}" if order == "before" else f"{catch_all}, {extra}"
        source = policy.read_text()
        assert catch_all in source
        policy.write_text(source.replace(catch_all, sequence))
    with project_context(project):
        namespace = importlib.import_module("knowledge.retrieval").KNOWLEDGE_NAMESPACE
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="proposal",
                        name="remember_knowledge",
                        arguments={"text": "The atlas project uses blue labels."},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})],
            ]
        )
        app = importlib.import_module("app").build_app(provider=provider)
        proposal = next(
            tool for tool in app.describe().agents[0].tools if tool.name == "remember_knowledge"
        )
        assert proposal.policy_coverage == ("approval_required" if approvals else "denied")

        async def exercise():
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="profile",
                        session_id="approval-smoke",
                        messages=[Message.text("user", "Remember the atlas label convention.")],
                    )
                )
            ]
            reader = SQLiteKnowledgeStore(project / "data/cayu.db")
            query = KnowledgeListQuery(namespace=namespace, statuses=list(KnowledgeStatus))
            scope = KnowledgeAccessScope.privileged()
            assert not (await reader.list_entries(query, access_scope=scope)).entries
            durable = await app.session_store.load_events("approval-smoke")
            requests = [e for e in durable if e.type == EventType.TOOL_CALL_APPROVAL_REQUESTED]
            if approvals:
                assert len(requests) == 1
                pending = PendingToolApprovalEventView.from_event(requests[0])
                events.extend(
                    [
                        event
                        async for event in app.resolve_tool_approval(
                            ToolApprovalRequest(
                                session_id="approval-smoke",
                                approval_id=pending.approval_id,
                                tool_round_id=pending.tool_round_id,
                                tool_call_id=pending.tool_call_id,
                                decision=ToolApprovalDecision.APPROVE,
                            )
                        )
                    ]
                )
                entries = (await reader.list_entries(query, access_scope=scope)).entries
                assert len(entries) == 1
                assert entries[0].entry.status == KnowledgeStatus.PENDING
            else:
                assert not requests
                assert any(
                    e.type == EventType.TOOL_CALL_BLOCKED
                    and e.payload.get("decision") == "deny"
                    and e.payload.get("denied_by") == "tool_policy"
                    for e in durable
                )
                assert not (await reader.list_entries(query, access_scope=scope)).entries
            assert events[-1].type == EventType.SESSION_COMPLETED

        asyncio.run(exercise())


def test_generated_default_search_finds_only_active_scoped_knowledge(tmp_path, capsys):
    assert main(["new", "profile", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "profile"
    with project_context(project):
        namespace = importlib.import_module("knowledge.retrieval").KNOWLEDGE_NAMESPACE
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="search", name="search_knowledge", arguments={"query": "atlas"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = importlib.import_module("app").build_app(provider=provider)

        async def exercise():
            maintenance = SQLiteKnowledgeStore(project / "data/cayu.db")
            for identity, scope, status in (
                ("active", namespace, KnowledgeStatus.ACTIVE),
                ("pending", namespace, KnowledgeStatus.PENDING),
                ("archived", namespace, KnowledgeStatus.ARCHIVED),
                ("outside", "default", KnowledgeStatus.ACTIVE),
            ):
                await maintenance.create_entry(
                    KnowledgeEntry(
                        id=identity,
                        namespace=scope,
                        status=status,
                        text="atlas " + identity,
                    ),
                    access_scope=KnowledgeAccessScope.privileged(),
                )
            outcome = await run_to_completion(
                app,
                RunRequest(
                    agent_name="profile",
                    messages=[Message.text("user", "Search atlas")],
                ),
            )
            assert outcome.ok

        asyncio.run(exercise())
        evidence = json.dumps(
            [
                message.model_dump(mode="json")
                for message in provider.requests[-1].messages
                if message.role == "tool"
            ]
        )
        assert "atlas active" in evidence
        assert "atlas pending" not in evidence
        assert "atlas archived" not in evidence
        assert "atlas outside" not in evidence


@pytest.mark.parametrize(
    "execution,excluded",
    (
        ("none", ("tasks",)),
        ("none", ("delegation", "human-input", "artifacts")),
        ("none", ("knowledge", "tasks", "delegation", "human-input", "artifacts")),
        ("docker", ("knowledge", "tasks", "delegation", "human-input")),
    ),
)
def test_reduced_generated_project_tests_pass(tmp_path, execution, excluded):
    project = _write_project(
        tmp_path / "profile", preset="coding", execution=execution, without_capabilities=excluded
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    suites = ["tests/test_coding_composition.py", "tests/test_architecture.py"]
    if execution != "docker":
        # The ordinary Docker factory test requires a built/admitted image;
        # the reduced Docker suite explicitly proves construction only.
        suites.append("tests/test_application.py")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *suites],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
