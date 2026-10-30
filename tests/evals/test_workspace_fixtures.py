"""Workspace fixtures: each trial of a case runs on its own checkout of the case's revision."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr, ValidationError
from tests.core.test_human_review import identity
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_coding_backend_conformance import _profile as conformance_profile
from tests.sandbox_broker.fake_broker import FakeBroker

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.budgets.usage import SessionUsageSummary, session_usage_summary_payload
from cayu.cli import main
from cayu.environments import (
    CodingAdmissionProbe,
    DockerCodingEnvironmentFactory,
    DockerWorkspaceTransferLimits,
    Environment,
    EnvironmentSpec,
)
from cayu.evals import (
    CorpusTarget,
    CorpusUserMessageSpec,
    EvalCase,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalStatus,
    EvalSuite,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    EvaluationSourceIdentityV1,
    RootStatusAssertionSpec,
    RunInputSpec,
    TrialRequestSpec,
    WorkspaceFileExists,
    WorkspaceFixtureIdentityV1,
    WorkspaceFixtureSpec,
    compare_eval_runs,
    eval_corpus_from_json,
    eval_corpus_to_json,
    run_corpus_suite,
    run_eval_suite,
)
from cayu.evals.capacity import EvalExecutionCapacity
from cayu.evals.models import (
    EvalAssertionResult,
    EvalCaseResult,
    EvalOutcome,
    EvalRun,
    EvalTrialResult,
)
from cayu.evals.reporting import write_eval_run_json
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1
from cayu.messages import MessageRole
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runners import DockerImageIdentity
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sandbox_broker import BrokeredSandboxCodingEnvironmentFactory
from cayu.storage import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workspaces import GitWorkspaceSource, LocalWorkspace, WorkspaceSource

_CLAIM_PATH = "claims/test_claim.py"


# A repository with two commits of calc.py --------------------------------------------


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Eval", "-c", "user.email=eval@example.test", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str, str]:
    """A repository whose first commit subtracts and whose second adds; HEAD is the second."""

    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    (repository / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (repository / "run.sh").write_text("#!/bin/sh\necho ok\n")
    (repository / "run.sh").chmod(0o755)
    _git(repository, "add", "-A")
    _git(repository, "commit", "--quiet", "-m", "first")
    first = _git(repository, "rev-parse", "HEAD")
    (repository / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    _git(repository, "commit", "--quiet", "-am", "second")
    return repository, first, _git(repository, "rev-parse", "HEAD")


# The fixture model ----------------------------------------------------------------------


def test_a_fixture_names_a_full_commit_and_canonical_files() -> None:
    fixture = WorkspaceFixtureSpec(
        repository="target",
        revision="a" * 40,
        files={"z.txt": "last", "claims/a.py": "first"},
    )
    assert list(fixture.files) == ["claims/a.py", "z.txt"]
    assert WorkspaceFixtureSpec(repository="target", revision="b" * 64).files == {}
    for revision in ("a" * 39, "A" * 40, "main", "a" * 41):
        with pytest.raises(ValidationError, match="full lowercase commit id"):
            WorkspaceFixtureSpec(repository="target", revision=revision)
    with pytest.raises(ValidationError, match="repository"):
        WorkspaceFixtureSpec(repository="Target", revision="a" * 40)
    for path, message in (
        ("/abs.txt", "relative"),
        ("../up.txt", "parent traversal"),
        ("a//b.txt", "canonical POSIX"),
        (".git/hooks/post-checkout", ".git"),
        ("sub/.GIT/config", ".git"),
    ):
        with pytest.raises(ValidationError, match=message):
            WorkspaceFixtureSpec(repository="target", revision="a" * 40, files={path: "x"})
    with pytest.raises(ValidationError, match="differ only in case"):
        WorkspaceFixtureSpec(
            repository="target", revision="a" * 40, files={"A.txt": "x", "a.txt": "y"}
        )
    with pytest.raises(ValidationError, match="inside the file"):
        WorkspaceFixtureSpec(
            repository="target", revision="a" * 40, files={"a": "x", "a/b.txt": "y"}
        )
    with pytest.raises(ValidationError, match="UTF-8 bytes in total"):
        WorkspaceFixtureSpec(
            repository="target",
            revision="a" * 40,
            files={"big.txt": "x" * (1 << 20), "more.txt": "x"},
        )


def test_the_identity_covers_the_revision_and_every_written_file() -> None:
    fixture = WorkspaceFixtureSpec(repository="target", revision="a" * 40, files={"a.txt": "one"})
    same = WorkspaceFixtureSpec(repository="target", revision="a" * 40, files={"a.txt": "one"})
    assert fixture.identity() == same.identity()
    assert fixture.identity() == WorkspaceFixtureIdentityV1(
        repository="target", revision="a" * 40, files_digest=fixture.files_digest()
    )
    assert fixture.identity() != fixture.model_copy(update={"revision": "b" * 40}).identity()
    changed = WorkspaceFixtureSpec(repository="target", revision="a" * 40, files={"a.txt": "two"})
    renamed = WorkspaceFixtureSpec(repository="target", revision="a" * 40, files={"b.txt": "one"})
    assert len({fixture.files_digest(), changed.files_digest(), renamed.files_digest()}) == 3


def _case_spec(workspace: WorkspaceFixtureSpec | None) -> EvalCaseSpec:
    return EvalCaseSpec.create(
        id="claim",
        suite_id="verifier",
        name="Claim",
        source=EvaluationSourceIdentityV1(
            application_release_id="captured-release",
            app_manifest_schema_version="7",
            app_manifest_fingerprint="a" * 64,
            evidence_revision="sha256:" + "e" * 64,
        ),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Check the claim."),)),
        assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
        workspace=workspace,
    )


def test_a_corpus_case_revision_covers_its_fixture_and_omits_an_absent_one() -> None:
    without = _case_spec(None)
    first = _case_spec(WorkspaceFixtureSpec(repository="target", revision="a" * 40))
    second = _case_spec(WorkspaceFixtureSpec(repository="target", revision="b" * 40))
    assert "workspace" not in without.model_dump(mode="json")
    assert len({without.revision, first.revision, second.revision}) == 3
    corpus = EvalCorpusDocument.create(
        target_key="verifier",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(EvalSuiteSpec.create(id="verifier", name="Verifier"),),
        cases=(first,),
    )
    assert eval_corpus_from_json(eval_corpus_to_json(corpus)) == corpus
    with pytest.raises(TypeError, match="WorkspaceFixtureSpec"):
        EvalCaseSpec.create(
            id="claim",
            suite_id="verifier",
            name="Claim",
            source=without.source,
            input=without.input,
            assertions=without.assertions,
            workspace={"repository": "target", "revision": "a" * 40},  # type: ignore[arg-type]
        )


# Git sources --------------------------------------------------------------------------


def test_a_git_source_checks_a_revision_out_without_touching_the_repository(
    tmp_path: Path,
) -> None:
    repository, first, second = _repository(tmp_path)
    _git(repository, "tag", "--annotate", "--message", "first", "first", first)
    tag = _git(repository, "rev-parse", "first")
    bare = tmp_path / "mirror.git"
    subprocess.run(
        ["git", "clone", "--quiet", "--bare", str(repository), str(bare)],
        check=True,
        capture_output=True,
    )

    async def scenario() -> None:
        for source in (
            GitWorkspaceSource("target", repository),
            GitWorkspaceSource("mirror", bare),
        ):
            destination = tmp_path / f"{source.name}-checkout"
            destination.mkdir()
            await source.checkout(first, destination)
            assert (destination / "calc.py").read_text().endswith("return a - b\n")
            assert (destination / "run.sh").stat().st_mode & 0o111
            assert not (destination / ".git").exists()
        missing = tmp_path / "missing"
        missing.mkdir()
        with pytest.raises(LookupError, match="no commit"):
            await GitWorkspaceSource("target", repository).checkout("0" * 40, missing)
        with pytest.raises(ValueError, match="empty directory"):
            await GitWorkspaceSource("target", repository).checkout(first, destination)
        tagged = tmp_path / "tagged"
        tagged.mkdir()
        with pytest.raises(LookupError, match=f"is not a commit; name the commit {first}"):
            await GitWorkspaceSource("target", repository).checkout(tag, tagged)

    asyncio.run(scenario())
    assert _git(repository, "rev-parse", "HEAD") == second
    assert _git(repository, "status", "--porcelain") == ""
    assert (repository / "calc.py").read_text().endswith("return a + b\n")


def test_an_app_registers_each_source_name_once(tmp_path: Path) -> None:
    app = CayuApp(enable_logging=False)
    source = app.register_workspace_source(GitWorkspaceSource("target", tmp_path))
    assert app.get_workspace_source("target") is source
    with pytest.raises(ValueError, match="already registered"):
        app.register_workspace_source(GitWorkspaceSource("target", tmp_path))
    with pytest.raises(KeyError, match="not registered"):
        app.get_workspace_source("other")
    with pytest.raises(ValueError, match="lowercase"):
        GitWorkspaceSource("Target", tmp_path)


# Re-sourcing a brokered coding factory ---------------------------------------------------


def _brokered(
    broker: FakeBroker, source: Path, **overrides
) -> BrokeredSandboxCodingEnvironmentFactory:
    options = {
        "broker": broker.client(),
        "source_workspace": LocalWorkspace(
            source, workspace_id="template", excluded_directory_names=("secrets",)
        ),
        "egress": "public_web",
    }
    options.update(overrides)
    return BrokeredSandboxCodingEnvironmentFactory(**options)


def test_a_brokered_factory_takes_another_source_with_the_same_identity(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    broker = FakeBroker(tmp_path / "cloud")
    factory = _brokered(broker, template, labels={"app": "verifier"})
    resourced = factory.with_source_workspace(
        LocalWorkspace(other, workspace_id="trial", excluded_directory_names=("secrets",))
    )
    assert isinstance(resourced, BrokeredSandboxCodingEnvironmentFactory)
    assert resourced.source_workspace.root == other.resolve()
    assert resourced.execution_profile_identity == factory.execution_profile_identity
    assert resourced.labels == factory.labels
    with pytest.raises(ValueError, match="exclude the same paths"):
        factory.with_source_workspace(LocalWorkspace(other, workspace_id="trial"))

    class Custom(BrokeredSandboxCodingEnvironmentFactory):
        pass

    custom = Custom(
        broker=broker.client(), source_workspace=LocalWorkspace(template), egress="public_web"
    )
    assert custom.with_source_workspace(LocalWorkspace(other)) is None


def test_a_docker_factory_takes_another_source_without_its_warm_spares(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(template, excluded_directory_names=("secrets",)),
        toolchain_profile=docker_toolchain_profile(
            image_identity=DockerImageIdentity(reference="python:3.12-slim@sha256:" + "a" * 64)
        ),
        transfer_limits=DockerWorkspaceTransferLimits(max_files=42),
        deferred=True,
        warm_spares=2,
    )
    resourced = factory.with_source_workspace(
        LocalWorkspace(other, workspace_id="trial", excluded_directory_names=("secrets",))
    )
    assert isinstance(resourced, DockerCodingEnvironmentFactory)
    assert resourced.source_workspace.root == other.resolve()
    assert resourced.execution_profile_identity == factory.execution_profile_identity
    assert resourced.transfer_limits == factory.transfer_limits and resourced.deferred
    assert resourced.warm_spares == 0 and resourced.warm_spare_pool is None
    with pytest.raises(ValueError, match="exclude the same paths"):
        factory.with_source_workspace(LocalWorkspace(other))


def test_re_sourcing_ignores_later_changes_to_what_the_caller_passed(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    commands = ["echo prepare"]
    brokered = _brokered(FakeBroker(tmp_path / "cloud"), template, setup_commands=commands)
    probes = [CodingAdmissionProbe(probe_id="extra", argv=("/opt/tools/bin/version",))]
    docker = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(template),
        toolchain_profile=conformance_profile(),
        image_identity=DockerImageIdentity(reference="python:3.12-slim@sha256:" + "a" * 64),
        platform_architecture="amd64",
        extra_admission_probes=probes,
    )
    # The caller reuses its lists for another environment.
    commands.append("echo another")
    probes.append(CodingAdmissionProbe(probe_id="later", argv=("/opt/tools/bin/version",)))

    resourced = brokered.with_source_workspace(
        LocalWorkspace(other, excluded_directory_names=("secrets",))
    )
    assert resourced is not None and resourced.setup_commands == ("echo prepare",)
    assert resourced.execution_profile_identity == brokered.execution_profile_identity
    resourced_docker = docker.with_source_workspace(LocalWorkspace(other))
    assert resourced_docker is not None
    assert resourced_docker.execution_profile_identity == docker.execution_profile_identity


def test_re_sourcing_carries_every_constructor_argument_but_the_source(tmp_path: Path) -> None:
    brokered = _brokered(FakeBroker(tmp_path / "cloud"), tmp_path)
    docker = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        toolchain_profile=docker_toolchain_profile(
            image_identity=DockerImageIdentity(reference="python:3.12-slim@sha256:" + "a" * 64)
        ),
    )
    for factory in (brokered, docker):
        parameters = set(inspect.signature(type(factory).__init__).parameters)
        assert set(factory._source_independent_arguments) == parameters - {
            "self",
            "source_workspace",
        }


def test_a_docker_factory_whose_configuration_changed_takes_no_other_source(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template"
    template.mkdir()
    seccomp = tmp_path / "seccomp.json"
    seccomp.write_text('{"defaultAction": "SCMP_ACT_ERRNO"}')
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(template),
        toolchain_profile=docker_toolchain_profile(
            image_identity=DockerImageIdentity(reference="python:3.12-slim@sha256:" + "a" * 64)
        ),
        seccomp_profile=str(seccomp),
    )
    seccomp.write_text('{"defaultAction": "SCMP_ACT_ALLOW"}')
    with pytest.raises(ValueError, match="configuration changed"):
        factory.with_source_workspace(LocalWorkspace(tmp_path / "template"))


# Running suites -------------------------------------------------------------------------


class _InspectProvider(ModelProvider):
    """Calls the inspect tool once, then finishes; request-aware, so trials can overlap."""

    name = "inspect-provider"

    @property
    def execution_profile_identity(self):
        return identity("inspect-provider")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if any(message.role == MessageRole.TOOL for message in request.messages):
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        yield ModelStreamEvent.tool_call(id="call-inspect", name="inspect_workspace", arguments={})
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class _InspectTool(Tool):
    """Records what the trial's workspace holds at start, then writes a marker into it."""

    spec = ToolSpec(
        name="inspect_workspace",
        parallel_safe=False,
        workspace_mutation=True,
        max_terminal_payload_bytes=1024,
    )

    def __init__(self, seen: list[tuple[str, str, str | None, bool]]) -> None:
        self.seen = seen

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.workspace is not None
        calc = (await ctx.workspace.read_bytes("calc.py")).content.decode()
        try:
            claim = (await ctx.workspace.read_bytes(_CLAIM_PATH)).content.decode()
        except FileNotFoundError:
            claim = None
        try:
            await ctx.workspace.read_bytes("marker.txt")
            marked = True
        except FileNotFoundError:
            marked = False
        self.seen.append((ctx.session_id, calc, claim, marked))
        await ctx.workspace.create_bytes("marker.txt", ctx.session_id.encode())
        # Hold the sandbox so concurrent trials overlap.
        await asyncio.sleep(0.05)
        return ToolResult(content="inspected")


def _app(tmp_path: Path, broker: FakeBroker, template: Path, seen: list) -> CayuApp:
    keyring = PublicAuthorityAliasKeyring(
        active_key_id="test",
        keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
    )
    store = SQLiteSessionStore(
        tmp_path / "sessions.sqlite",
        public_authority_alias_codec=PublicAuthorityAliasCodec(keyring),
    )
    app = CayuApp(session_store=store, enable_logging=False, public_authority_alias_keyring=keyring)
    app.register_provider(_InspectProvider(), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="sandbox", execution_profile_identity=identity("brokered")),
        _brokered(broker, template),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="verifier", model="scripted-model"), tools=[_InspectTool(seen)]
    )
    return app


def _template(tmp_path: Path) -> Path:
    template = tmp_path / "template"
    template.mkdir(parents=True)
    (template / "calc.py").write_text("template\n")
    return template


def _request() -> RunRequest:
    return RunRequest(agent_name="verifier", messages=[Message.text("user", "Check.")])


def _trial_directories() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("cayu-eval-workspace-*"))


def test_each_trial_runs_on_its_own_checkout_of_its_case_revision(tmp_path: Path) -> None:
    repository, first, second = _repository(tmp_path)
    template = _template(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    seen: list[tuple[str, str, str | None, bool]] = []
    app = _app(tmp_path, broker, template, seen)
    app.register_workspace_source(GitWorkspaceSource("target", repository))
    claim = "def test_claim():\n    assert False\n"
    suite = EvalSuite(
        id="verifier",
        cases=[
            EvalCase(
                id=name,
                request=_request(),
                assertions=[WorkspaceFileExists("marker.txt")],
                workspace=WorkspaceFixtureSpec(
                    repository="target", revision=revision, files={_CLAIM_PATH: claim}
                ),
            )
            for name, revision in (("before-fix", first), ("after-fix", second))
        ],
    )
    manifest = app.describe().fingerprint
    directories = _trial_directories()

    run = asyncio.run(run_eval_suite(app, suite, trials=2, max_concurrency=4))

    assert run.status == EvalStatus.PASSED, [
        trial.error for case in run.cases for trial in case.trials
    ]
    calc_by_session = {session_id: calc for session_id, calc, _, _ in seen}
    for case, revision, body in zip(
        run.cases, (first, second), ("return a - b", "return a + b"), strict=True
    ):
        for trial in case.trials:
            assert body in calc_by_session[trial.session_id]
            assert trial.workspace == WorkspaceFixtureIdentityV1(
                repository="target",
                revision=revision,
                files_digest=suite.cases[0].workspace.files_digest(),
            )
    # Every trial saw the claim and no other trial's marker, in a sandbox of its own.
    assert len(seen) == 4 and all(claimed == claim and not marked for _, _, claimed, marked in seen)
    assert len(broker.sandboxes) == 4
    # The application, its template source and the temporary checkouts are as they were.
    assert app.describe().fingerprint == manifest
    assert sorted(path.name for path in template.iterdir()) == ["calc.py"]
    assert _trial_directories() == directories


def test_a_trial_whose_checkout_fails_is_a_setup_failure(tmp_path: Path) -> None:
    repository, _, _ = _repository(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    app = _app(tmp_path, broker, _template(tmp_path), [])
    app.register_workspace_source(GitWorkspaceSource("target", repository))
    fixture = WorkspaceFixtureSpec(repository="target", revision="0" * 40)
    suite = EvalSuite(
        id="verifier",
        cases=[
            EvalCase(
                id="unknown-revision",
                request=_request(),
                assertions=[WorkspaceFileExists("marker.txt")],
                workspace=fixture,
            )
        ],
    )

    run = asyncio.run(run_eval_suite(app, suite))

    (trial,) = run.cases[0].trials
    assert trial.status == EvalStatus.ERROR
    assert trial.execution_status == "failed"
    assert trial.execution_failure_category == "setup_failure"
    assert "no commit" in (trial.error or "")
    assert [assertion.outcome.value for assertion in trial.assertions] == ["unavailable"]
    assert trial.workspace == fixture.identity()
    assert broker.sandboxes == {}


class _StalledSource(WorkspaceSource):
    """A source whose checkout never finishes on its own."""

    def __init__(self) -> None:
        super().__init__("stalled")
        self.started = 0

    async def checkout(self, revision: str, destination: Path) -> None:
        del revision, destination
        self.started += 1
        await asyncio.sleep(30)


def test_the_case_timeout_bounds_workspace_preparation(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    app = _app(tmp_path, broker, _template(tmp_path), [])
    source = app.register_workspace_source(_StalledSource())
    suite = EvalSuite(
        id="verifier",
        cases=[
            EvalCase(
                id="claim",
                request=_request(),
                assertions=[WorkspaceFileExists("marker.txt")],
                workspace=WorkspaceFixtureSpec(repository="stalled", revision="a" * 40),
            )
        ],
    )
    directories = _trial_directories()
    started = time.monotonic()

    run = asyncio.run(run_eval_suite(app, suite, case_timeout_seconds=0.5))

    assert time.monotonic() - started < 10
    (trial,) = run.cases[0].trials
    assert trial.status == EvalStatus.ERROR
    assert trial.error == "Eval case timed out after 0.5 seconds."
    assert trial.session_id is None and trial.workspace is not None
    assert source.started == 1 and broker.sandboxes == {}
    assert _trial_directories() == directories


class _ConflictingEditTool(Tool):
    """Writes in the sandbox and edits the trial's checkout behind its back, so copy-back fails."""

    spec = ToolSpec(
        name="inspect_workspace",
        parallel_safe=False,
        workspace_mutation=True,
        max_terminal_payload_bytes=1024,
    )

    def __init__(self, token: str) -> None:
        self.token = token
        self.original: str | None = None

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del args
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes("marker.txt", b"written")
        (checkout,) = (
            path.parent
            for path in Path(tempfile.gettempdir()).glob("cayu-eval-workspace-*/workspace/token")
            if path.read_text() == self.token
        )
        self.original = (checkout / "calc.py").read_text()
        (checkout / "calc.py").write_text("changed outside the sandbox\n")
        return ToolResult(content="edited")


def test_a_checkout_outlives_a_session_whose_copy_back_waits_for_a_retry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repository, first, _ = _repository(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    app = _app(tmp_path, broker, _template(tmp_path), [])
    editor = _ConflictingEditTool(uuid4().hex)
    app.register_agent(AgentSpec(name="editor", model="scripted-model"), tools=[editor])
    app.register_workspace_source(GitWorkspaceSource("target", repository))
    suite = EvalSuite(
        id="verifier",
        cases=[
            EvalCase(
                id="claim",
                request=_request().model_copy(update={"agent_name": "editor"}),
                workspace=WorkspaceFixtureSpec(
                    repository="target", revision=first, files={"token": editor.token}
                ),
            )
        ],
    )
    directories = _trial_directories()

    async def scenario() -> tuple[EvalRun, Path, bool, bool]:
        run = await run_eval_suite(app, suite)
        (kept,) = _trial_directories() - directories
        lifecycle = app._environment_lifecycle
        (owner,) = lifecycle._active_environment_setups.values()
        bound = owner.registered_environment.bound_workspace
        assert owner.cleanup_requires_finalize_retry and bound is not None
        assert bound.source_workspace is not None
        assert bound.source_workspace.root == (kept / "workspace").resolve()
        retained = lifecycle.binds_source_workspace(bound.source_workspace.id)
        # Once the conflict is gone, the retained retry copies back into the kept checkout.
        assert editor.original is not None
        (kept / "workspace" / "calc.py").write_text(editor.original)
        settled = await app.drain_environment_cleanups()
        return run, kept, retained, settled

    run, kept, retained, settled = asyncio.run(scenario())
    try:
        (trial,) = run.cases[0].trials
        assert "source revision conflict" in (trial.error or "")
        assert retained and "retained for a retry that copies back into it" in caplog.text
        assert settled and not app._environment_lifecycle._active_environment_setups
        assert (kept / "workspace" / "marker.txt").read_text() == "written"
    finally:
        shutil.rmtree(kept)


def test_a_suite_whose_trials_cannot_get_their_own_workspace_is_refused(tmp_path: Path) -> None:
    repository, first, _ = _repository(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    app = _app(tmp_path, broker, _template(tmp_path), [])
    fixture = WorkspaceFixtureSpec(repository="target", revision=first)
    with_fixture = EvalSuite(
        id="verifier", cases=[EvalCase(id="claim", request=_request(), workspace=fixture)]
    )
    with pytest.raises(ValueError, match="hasn't registered"):
        asyncio.run(run_eval_suite(app, with_fixture))
    app.register_workspace_source(GitWorkspaceSource("target", repository))
    app.register_environment(Environment(EnvironmentSpec(name="plain")))
    plain = EvalSuite(
        id="verifier",
        cases=[
            EvalCase(
                id="claim",
                request=_request().model_copy(update={"environment_name": "plain"}),
                workspace=fixture,
            )
        ],
    )
    with pytest.raises(ValueError, match="no factory whose sessions copy in"):
        asyncio.run(run_eval_suite(app, plain))
    # Without a fixture, concurrent trials would share the template's source.
    without = EvalSuite(id="verifier", cases=[EvalCase(id="claim", request=_request())])
    with pytest.raises(ValueError, match="share the source workspace of environment 'sandbox'"):
        asyncio.run(run_eval_suite(app, without, trials=2, max_concurrency=2))
    assert broker.sandboxes == {}


def test_trials_a_shared_capacity_serializes_may_share_a_source(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    app = _app(tmp_path, broker, _template(tmp_path), [])
    suite = EvalSuite(id="verifier", cases=[EvalCase(id="claim", request=_request())])

    run = asyncio.run(
        run_eval_suite(
            app,
            suite,
            trials=2,
            max_concurrency=2,
            execution_capacity=EvalExecutionCapacity(max_active_trials=1),
        )
    )

    assert [trial.error for trial in run.cases[0].trials] == [None, None]
    assert len(broker.sandboxes) == 2


def test_a_fixture_needs_a_factory_that_takes_another_source(tmp_path: Path) -> None:
    repository, first, _ = _repository(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    app = _app(tmp_path, broker, _template(tmp_path), [])
    app.register_workspace_source(GitWorkspaceSource("target", repository))

    class Custom(BrokeredSandboxCodingEnvironmentFactory):
        pass

    app.register_environment_factory(
        EnvironmentSpec(name="custom", execution_profile_identity=identity("custom")),
        Custom(
            broker=broker.client(),
            source_workspace=LocalWorkspace(_template(tmp_path / "custom")),
            egress="public_web",
        ),
    )
    suite = EvalSuite(
        id="verifier",
        cases=[
            EvalCase(
                id="claim",
                request=_request().model_copy(update={"environment_name": "custom"}),
                workspace=WorkspaceFixtureSpec(repository="target", revision=first),
            )
        ],
    )
    with pytest.raises(ValueError, match="can't take another source workspace"):
        asyncio.run(run_eval_suite(app, suite))
    assert broker.sandboxes == {}


def test_a_corpus_case_fixture_reaches_its_trials_and_published_result(tmp_path: Path) -> None:
    repository, first, _ = _repository(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    seen: list[tuple[str, str, str | None, bool]] = []
    app = _app(tmp_path, broker, _template(tmp_path), seen)
    app.register_workspace_source(GitWorkspaceSource("target", repository))
    fixture = WorkspaceFixtureSpec(repository="target", revision=first, files={_CLAIM_PATH: "x"})
    suite = EvalSuiteSpec.create(
        id="verifier",
        name="Verifier",
        trial_request=TrialRequestSpec(
            trials=2,
            timeout_seconds=60,
            trial_policy=EvalSuiteTrialPolicyV1.create(trial_count=2, max_concurrency=2),
        ),
    )
    corpus = EvalCorpusDocument.create(
        target_key="verifier",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(suite,),
        cases=(_case_spec(fixture),),
    )
    target = CorpusTarget(
        key="verifier",
        app=app,
        request_base=RunRequest(agent_name="verifier", messages=[]),
        application_release_id="verifier-release",
    )

    result = asyncio.run(run_corpus_suite(target, corpus, "verifier"))

    (case,) = result.run.cases
    assert case.status == "passed"
    assert [trial.workspace for trial in case.trials] == [fixture.identity()] * 2
    assert all("return a - b" in calc for _, calc, _, _ in seen) and len(seen) == 2


def _run(*, revision: str, status: EvalStatus = EvalStatus.PASSED) -> EvalRun:
    now = datetime.now(UTC)
    passed = status == EvalStatus.PASSED
    score = 1.0 if passed else 0.0
    trial = EvalTrialResult(
        trial_number=1,
        status=status,
        session_id="session-1",
        score=score,
        assertions=(
            EvalAssertionResult(
                name="check",
                outcome=EvalOutcome.PASSED if passed else EvalOutcome.FAILED,
                score=score,
                threshold=0.0 if passed else 1.0,
            ),
        ),
        workspace=WorkspaceFixtureIdentityV1(
            repository="target", revision=revision, files_digest="sha256:" + "f" * 64
        ),
        evidence_complete=True,
        usage_summary=session_usage_summary_payload(SessionUsageSummary(session_id="session-1")),
        started_at=now,
        completed_at=now,
    )
    case = EvalCaseResult.from_trials(case_id="claim", trials=(trial,))
    return EvalRun(
        suite_id="verifier",
        status=status,
        score=score,
        cases=(case,),
        started_at=now,
        completed_at=now,
        duration_ms=0,
    )


def test_runs_whose_case_changed_revision_are_not_compared() -> None:
    baseline = _run(revision="a" * 40)
    regressed = _run(revision="b" * 40, status=EvalStatus.FAILED)

    comparison = compare_eval_runs(baseline, regressed)

    assert comparison.incomparable_cases == ("claim",)
    assert comparison.regressions == ()
    assert comparison.comparable_cases == 0
    assert any("workspace fixture changed" in reason for reason in comparison.inconclusive_reasons)
    same = compare_eval_runs(baseline, _run(revision="a" * 40, status=EvalStatus.FAILED))
    assert same.incomparable_cases == () and same.regressions


def test_cayu_eval_compare_is_inconclusive_when_a_case_changed_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    write_eval_run_json(_run(revision="a" * 40), baseline)
    write_eval_run_json(_run(revision="b" * 40, status=EvalStatus.FAILED), current)

    assert main(["eval", "compare", str(baseline), str(current), "--format", "json"]) == 2

    captured = capsys.readouterr()
    assert json.loads(captured.out)["incomparable_cases"] == ["claim"]
    assert captured.err.splitlines()[-1] == (
        "cayu eval compare: 1 of 1 cases ran on a different workspace fixture, "
        "so the runs are not compared."
    )
