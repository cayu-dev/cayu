"""Recovery evidence retained by the public coding-product runner."""

from __future__ import annotations

import asyncio
import traceback
import warnings
from contextlib import aclosing
from hashlib import sha256

import pytest
from tests.core.test_coding_products import _request

from cayu import CayuApp, Message, RunRequest
from cayu.artifacts import LocalArtifactStore
from cayu.coding_products import (
    CodingProductAdmissionError,
    CodingProductArtifactRepository,
    CodingProductReconstructionRequiredError,
    CodingProductRunner,
    CodingProductState,
    CodingTaskAuthority,
)
from cayu.runtime.sessions import session_input_messages_sha256
from cayu.workspaces import LocalWorkspace
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationLimits,
    observe_deterministic_workspace,
)


@pytest.fixture
def product(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("content\n", encoding="utf-8")
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    messages = [Message.text("user", "repair the project")]
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    request = request.model_copy(
        update={
            "task": CodingTaskAuthority(
                task_id=request.task.task_id,
                instruction_sha256="sha256:" + session_input_messages_sha256(messages),
            )
        }
    )
    repository = CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts"))
    app = CayuApp()

    async def validate_git(expected):
        assert expected == request.source.git_baseline

    runner = CodingProductRunner(
        app,
        source_workspace=workspace,
        repository=repository,
        source_git_authority_validator=validate_git,
    )
    run_request = RunRequest(
        agent_name=request.agent_name,
        session_id=request.session_id,
        environment_name=request.runtime.environment_name,
        messages=messages,
    )
    return runner, request, run_request, observation


def test_runner_retains_initial_manifest_before_session_dispatch(product, monkeypatch):
    runner, request, run_request, observation = product

    async def session_boundary(_request):
        artifacts = await runner.repository.store.list(session_id=request.session_id)
        initial = [
            a for a in artifacts.artifacts if a.filename == "coding-product-source-initial.json"
        ]
        assert not artifacts.truncated
        assert len(initial) == 1
        result = await runner.repository.store.read_bytes(initial[0].id)
        assert WorkspaceRevisionObservation.model_validate_json(result.content) == observation
        receipts, _ = await runner.repository.load_lifecycle(
            request.product_run_id,
            session_id=request.session_id,
            request_fingerprint=request.fingerprint,
        )
        assert receipts[-1].state is CodingProductState.ACTIVE
        assert receipts[-1].evidence_sha256 == "sha256:" + sha256(result.content).hexdigest()
        raise RuntimeError("session boundary reached")
        yield  # pragma: no cover

    monkeypatch.setattr(runner.app, "run", session_boundary)
    with pytest.raises(RuntimeError, match="session boundary reached"):
        asyncio.run(runner.run(request, run_request))


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("receipt_failure", [None, "before", "after"])
def test_runner_preserves_primary_when_real_stream_close_fails(
    product, monkeypatch, cancel, receipt_failure
):
    from tests.core.test_queued_session_messages import RecordingOneShotProvider

    from cayu import AgentSpec, Environment, EnvironmentSpec, SessionStatus

    runner, request, run_request, _ = product
    provider = RecordingOneShotProvider()
    runner.app.register_provider(provider)
    runner.app.register_agent(AgentSpec(name=request.agent_name, model="fake-model"))
    runner.app.register_environment(Environment(EnvironmentSpec(name="coding")))
    original_run = runner.app.run
    cleanup = RuntimeError("controlled stream close failure")
    primary = OSError("controlled anchor retention failure")
    publication_failure = ExceptionGroup(
        "controlled publication failure",
        [OSError("receipt failure"), ExceptionGroup("nested", [RuntimeError("readback failure")])],
    )
    # Historical causal evidence must not turn an ordinary publication failure
    # into a newly delivered cancellation.
    publication_failure.__cause__ = asyncio.CancelledError("historical cancellation")
    original_append = runner.repository.append_lifecycle

    async def append(receipt):
        terminal = receipt.state in {CodingProductState.CANCELLED, CodingProductState.FAILED}
        if terminal and receipt_failure == "before":
            raise publication_failure
        value = await original_append(receipt)
        if terminal and receipt_failure == "after":
            raise publication_failure
        return value

    monkeypatch.setattr(runner.repository, "append_lifecycle", append)

    async def failing_close(request):
        try:
            async with aclosing(original_run(request)) as stream:
                async for event in stream:
                    yield event
        finally:
            raise cleanup

    monkeypatch.setattr(runner.app, "run", failing_close)

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def retain(*args):
            entered.set()
            await release.wait()
            raise primary

        monkeypatch.setattr(runner.repository, "_retain_execution_anchor", retain)
        owner = asyncio.create_task(runner.run(request, run_request))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if cancel:
                owner.cancel("stop-at-anchor")
                with pytest.raises(asyncio.CancelledError) as caught:
                    await owner
                assert owner.cancelled() and owner.cancelling() == 1
            else:
                release.set()
                with pytest.raises(OSError) as caught:
                    await owner
                assert caught.value is primary
            if receipt_failure is None:
                assert caught.value.__cause__ is cleanup
            else:
                cause = caught.value.__cause__
                assert type(cause) is ExceptionGroup
                assert cause.exceptions == (cleanup, publication_failure)
            assert not provider.requests
            session = await runner.app.session_store.load(request.session_id)
            assert session is not None and session.status is SessionStatus.INTERRUPTED
            receipts, _ = await runner.repository.load_lifecycle(
                request.product_run_id,
                session_id=request.session_id,
                request_fingerprint=request.fingerprint,
            )
            expected_state = CodingProductState.CANCELLED if cancel else CodingProductState.FAILED
            if receipt_failure == "before":
                expected_state = CodingProductState.ACTIVE
            assert receipts[-1].state is expected_state
            with pytest.raises(CodingProductReconstructionRequiredError):
                await runner.run(request, run_request)
            assert not provider.requests
        finally:
            release.set()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("initial_cancel", [False, True])
@pytest.mark.parametrize("committed", [False, True])
def test_runner_cancellation_during_failure_receipt(
    product, monkeypatch, initial_cancel, committed
):
    from tests.core.test_queued_session_messages import RecordingOneShotProvider

    from cayu import AgentSpec, Environment, EnvironmentSpec, SessionStatus

    runner, request, run_request, _ = product
    provider = RecordingOneShotProvider()
    runner.app.register_provider(provider)
    runner.app.register_agent(AgentSpec(name=request.agent_name, model="fake-model"))
    runner.app.register_environment(Environment(EnvironmentSpec(name="coding")))
    original_append = runner.repository.append_lifecycle
    primary = OSError("controlled anchor failure")

    async def scenario():
        anchor_entered, receipt_entered = asyncio.Event(), asyncio.Event()
        release_anchor, release_receipt = asyncio.Event(), asyncio.Event()
        cancellations = []

        async def retain(*args):
            anchor_entered.set()
            try:
                await release_anchor.wait()
            except asyncio.CancelledError as failure:
                cancellations.append(failure)
                raise
            raise primary

        async def append(receipt):
            if receipt.state not in {CodingProductState.CANCELLED, CodingProductState.FAILED}:
                return await original_append(receipt)
            artifact = await original_append(receipt) if committed else None
            receipt_entered.set()
            try:
                await release_receipt.wait()
            except asyncio.CancelledError as failure:
                cancellations.append(failure)
                raise
            assert artifact is not None
            return artifact

        monkeypatch.setattr(runner.repository, "_retain_execution_anchor", retain)
        monkeypatch.setattr(runner.repository, "append_lifecycle", append)
        owner = asyncio.create_task(runner.run(request, run_request))
        try:
            await asyncio.wait_for(anchor_entered.wait(), 5)
            if initial_cancel:
                owner.cancel("cancel-execution")
            else:
                release_anchor.set()
            await asyncio.wait_for(receipt_entered.wait(), 5)
            owner.cancel("cancel-receipt")
            with pytest.raises(asyncio.CancelledError) as caught:
                await owner
            assert owner.cancelled()
            assert owner.cancelling() == (2 if initial_cancel else 1)
            assert caught.value is cancellations[0]
            assert caught.value.__cause__ is (cancellations[1] if initial_cancel else primary)
            session = await runner.app.session_store.load(request.session_id)
            assert session is not None and session.status is SessionStatus.INTERRUPTED
            receipts, _ = await runner.repository.load_lifecycle(
                request.product_run_id,
                session_id=request.session_id,
                request_fingerprint=request.fingerprint,
            )
            expected = CodingProductState.CANCELLED if initial_cancel else CodingProductState.FAILED
            assert receipts[-1].state is (expected if committed else CodingProductState.ACTIVE)
            with pytest.raises(CodingProductReconstructionRequiredError):
                await runner.run(request, run_request)
            assert not provider.requests
        finally:
            release_anchor.set()
            release_receipt.set()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("committed", [False, True])
def test_source_validation_preserves_failure_when_receipt_fails(
    product, monkeypatch, capsys, caplog, cancel, committed
):
    runner, request, run_request, _ = product
    original_append = runner.repository.append_lifecycle
    publication_failure = OSError("controlled receipt failure")
    secret = "validator-secret-canary"

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        cancellations = []

        async def validate(expected):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError as failure:
                cancellations.append(failure)
                raise
            raise RuntimeError(secret)

        async def append(receipt):
            if receipt.state not in {
                CodingProductState.CANCELLED,
                CodingProductState.SOURCE_CONFLICT,
            }:
                return await original_append(receipt)
            if committed:
                await original_append(receipt)
            raise publication_failure

        monkeypatch.setattr(runner, "source_git_authority_validator", validate)
        monkeypatch.setattr(runner.repository, "append_lifecycle", append)
        owner = asyncio.create_task(runner.run(request, run_request))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if cancel:
                owner.cancel("cancel-source-validation")
                with pytest.raises(asyncio.CancelledError) as caught:
                    await owner
                assert owner.cancelled() and owner.cancelling() == 1
                assert caught.value is cancellations[0]
            else:
                release.set()
                with pytest.raises(CodingProductAdmissionError) as caught:
                    await owner
            assert caught.value.__cause__ is publication_failure
            assert secret not in "".join(traceback.format_exception(caught.value))
            assert await runner.app.session_store.load(request.session_id) is None
            receipts, _ = await runner.repository.load_lifecycle(
                request.product_run_id,
                session_id=request.session_id,
                request_fingerprint=request.fingerprint,
            )
            expected = (
                CodingProductState.CANCELLED if cancel else CodingProductState.SOURCE_CONFLICT
            )
            assert receipts[-1].state is (
                expected if committed else CodingProductState.PREPARING_WORKSPACE
            )
            with pytest.raises(CodingProductReconstructionRequiredError):
                await runner.run(request, run_request)
        finally:
            release.set()
            await asyncio.gather(owner, return_exceptions=True)

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        asyncio.run(scenario())
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err + caplog.text
    assert all(secret not in str(item.message) for item in captured_warnings)


@pytest.mark.parametrize("receipt_failure", [None, "before", "after"])
def test_final_source_observation_cancellation_preserves_failure_receipt(
    product, monkeypatch, receipt_failure
):
    from tests.core.test_coding_products import _check_event, _git_events, _terminal_events

    import cayu.coding_products as products

    runner, request, run_request, observation = product
    original_observe = products.observe_deterministic_workspace
    original_append = runner.repository.append_lifecycle
    publication_failure = OSError("controlled cancellation receipt failure")

    async def completed_run(_request):
        for event in (
            *(
                _check_event(name, workspace_revision=observation.revision)
                for name in request.settlement.required_checks
            ),
            *_git_events(changed=False),
            *_terminal_events(request=request, workspace_revision=observation.revision),
        ):
            yield event

    monkeypatch.setattr(runner.app, "run", completed_run)

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        observations = 0
        cancellations = []

        async def observe(*args, **kwargs):
            nonlocal observations
            result = await original_observe(*args, **kwargs)
            observations += 1
            if observations == 3:
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError as failure:
                    cancellations.append(failure)
                    raise
            return result

        async def append(receipt):
            cancelled = receipt.state is CodingProductState.CANCELLED
            if cancelled and receipt_failure == "before":
                raise publication_failure
            result = await original_append(receipt)
            if cancelled and receipt_failure == "after":
                raise publication_failure
            return result

        monkeypatch.setattr(products, "observe_deterministic_workspace", observe)
        monkeypatch.setattr(runner.repository, "append_lifecycle", append)
        owner = asyncio.create_task(runner.run(request, run_request))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            owner.cancel("cancel-final-observation")
            with pytest.raises(asyncio.CancelledError) as caught:
                await owner
            assert caught.value is cancellations[0]
            assert caught.value.__cause__ is (
                None if receipt_failure is None else publication_failure
            )
            assert owner.cancelled() and owner.cancelling() == 1
            receipts, _ = await runner.repository.load_lifecycle(
                request.product_run_id,
                session_id=request.session_id,
                request_fingerprint=request.fingerprint,
            )
            assert receipts[-1].state is (
                CodingProductState.PUBLISHING
                if receipt_failure == "before"
                else CodingProductState.CANCELLED
            )
            if receipt_failure != "before":
                with pytest.raises(
                    CodingProductReconstructionRequiredError,
                    match="unreconciled terminal state",
                ):
                    await runner.recover_settled_execution(request)
            artifacts = await runner.repository.store.list(session_id=request.session_id)
            assert not artifacts.truncated
            assert all(a.filename != "coding-product-result.json" for a in artifacts.artifacts)
            with pytest.raises(CodingProductReconstructionRequiredError):
                await runner.run(request, run_request)
            assert observations == 3
        finally:
            release.set()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_phase", ["before_write", "after_write", "readback"])
def test_initial_retention_failure_prevents_dispatch(product, monkeypatch, failure_phase):
    runner, request, run_request, _ = product
    store = runner.repository.store
    original_put = store.put_bytes
    original_read = store.read_bytes
    initial_ids = set()

    async def put(content, **kwargs):
        initial = kwargs.get("filename") == "coding-product-source-initial.json"
        if initial:
            initial_ids.add(kwargs["artifact_id"])
        if initial and failure_phase == "before_write":
            raise OSError("initial retention failed")
        result = await original_put(content, **kwargs)
        if initial and failure_phase == "after_write":
            raise OSError("initial retention failed")
        return result

    async def read(artifact_id, **kwargs):
        if artifact_id in initial_ids and failure_phase == "readback":
            raise OSError("initial retention failed")
        return await original_read(artifact_id, **kwargs)

    async def forbidden_dispatch(_request):
        pytest.fail("session dispatched without durable initial evidence")
        yield  # pragma: no cover

    monkeypatch.setattr(store, "put_bytes", put)
    monkeypatch.setattr(store, "read_bytes", read)
    monkeypatch.setattr(runner.app, "run", forbidden_dispatch)
    with pytest.raises(OSError, match="initial retention failed"):
        asyncio.run(runner.run(request, run_request))


def test_cancellation_during_initial_retention_prevents_dispatch(product, monkeypatch):
    runner, request, run_request, _ = product

    async def scenario():
        entered = asyncio.Event()
        original = runner.repository.store.put_bytes

        async def blocked_put(content, **kwargs):
            if kwargs.get("filename") == "coding-product-source-initial.json":
                entered.set()
                await asyncio.Future()
            return await original(content, **kwargs)

        async def forbidden_dispatch(_request):
            pytest.fail("session dispatched after cancellation")
            yield  # pragma: no cover

        monkeypatch.setattr(runner.repository.store, "put_bytes", blocked_put)
        monkeypatch.setattr(runner.app, "run", forbidden_dispatch)
        task = asyncio.create_task(runner.run(request, run_request))
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        receipts, _ = await runner.repository.load_lifecycle(
            request.product_run_id,
            session_id=request.session_id,
            request_fingerprint=request.fingerprint,
        )
        assert all(receipt.state is not CodingProductState.ACTIVE for receipt in receipts)

    asyncio.run(scenario())


@pytest.fixture
def retained_product(product, monkeypatch):
    runner, request, run_request, observation = product

    async def stopped_session(_request):
        raise RuntimeError("stopped session")
        yield  # pragma: no cover

    monkeypatch.setattr(runner.app, "run", stopped_session)
    with pytest.raises(RuntimeError, match="stopped session"):
        asyncio.run(runner.run(request, run_request))
    return runner, request, observation


def test_initial_source_reconstruction_is_exact_and_read_only(retained_product, monkeypatch):
    runner, request, observation = retained_product
    repository = CodingProductArtifactRepository(runner.repository.store)

    async def forbidden(*args, **kwargs):
        pytest.fail("read-only reconstruction attempted a write or listing")

    monkeypatch.setattr(repository.store, "put_bytes", forbidden)
    monkeypatch.setattr(repository.store, "list", forbidden)
    assert asyncio.run(repository.load_initial_source_observation(request)) == observation


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("source", "origin_id", "other-origin"),
        ("source", "workspace_id", "other-workspace"),
        ("source", "destination_id", "other-destination"),
        ("source", "baseline_revision", "sha256:" + "b" * 64),
        ("task", "task_id", "other-task"),
        ("task", "instruction_sha256", "sha256:" + "b" * 64),
        ("runtime", "environment_name", "other-environment"),
        ("runtime", "image_fingerprint", "sha256:" + "b" * 64),
        ("runtime", "execution_profile_fingerprint", "sha256:" + "b" * 64),
        ("runtime", "tool_policy_fingerprint", "sha256:" + "b" * 64),
        ("runtime", "approval_policy_fingerprint", "sha256:" + "b" * 64),
        ("runtime", "redaction_profile_fingerprint", "sha256:" + "b" * 64),
        ("settlement", "reviewer_required", True),
        ("settlement", "required_checks", ("test",)),
    ],
)
def test_initial_source_rejects_changed_expected_authority(retained_product, section, field, value):
    runner, request, _ = retained_product
    changed = request.model_copy(
        update={section: getattr(request, section).model_copy(update={field: value})}
    )
    with pytest.raises(CodingProductAdmissionError, match="conflicts with admitted authority"):
        asyncio.run(runner.repository.load_initial_source_observation(changed))


def test_initial_source_requires_active_receipt(product):
    runner, request, _, _ = product
    asyncio.run(runner.repository.ensure_request(request))
    with pytest.raises(CodingProductReconstructionRequiredError, match="no exact active receipt"):
        asyncio.run(runner.repository.load_initial_source_observation(request))


@pytest.mark.parametrize("failure", ["missing", "substituted", "truncated"])
def test_initial_source_rejects_unavailable_or_changed_artifact(
    retained_product, monkeypatch, failure
):
    from dataclasses import replace

    runner, request, _ = retained_product
    original = runner.repository.store.read_bytes

    async def read(artifact_id, **kwargs):
        result = await original(artifact_id, **kwargs)
        if result.metadata.filename == "coding-product-source-initial.json":
            if failure == "missing":
                raise FileNotFoundError(artifact_id)
            if kwargs.get("max_bytes", 0) > 1:
                if failure == "substituted":
                    return replace(result, content=b"x" * len(result.content))
                return replace(result, truncated=True)
        return result

    monkeypatch.setattr(runner.repository.store, "read_bytes", read)
    with pytest.raises((FileNotFoundError, ValueError)):
        asyncio.run(runner.repository.load_initial_source_observation(request))


def test_initial_source_readback_preserves_cancellation(retained_product, monkeypatch):
    runner, request, _ = retained_product

    async def scenario():
        entered = asyncio.Event()
        original = runner.repository.store.read_bytes

        async def read(artifact_id, **kwargs):
            result = await original(artifact_id, **kwargs)
            if result.metadata.filename == "coding-product-source-initial.json":
                entered.set()
                await asyncio.Future()
            return result

        monkeypatch.setattr(runner.repository.store, "read_bytes", read)
        task = asyncio.create_task(runner.repository.load_initial_source_observation(request))
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    ("intervening_operation", "failure_phase"),
    [
        (None, "compile"),
        ("resume", "compile"),
        ("recreate", "compile"),
        (None, "ready"),
        (None, "publish"),
        (None, "published"),
        ("ordinary_retry", "publish"),
        ("source_change", "publish"),
        ("lineage", "compile"),
        ("cancelled_after_selection", "ready"),
        ("source_conflict_after_selection", "ready"),
    ],
)
def test_recover_settled_execution_without_provider_redispatch(
    product, monkeypatch, tmp_path, backend, intervening_operation, failure_phase
):
    from tests.core.test_queued_session_messages import RecordingOneShotProvider

    from cayu import AgentSpec, ExecutionProfileBehaviorIdentity, SQLiteSessionStore
    from cayu.environments import Environment, EnvironmentSpec
    from cayu.runtime import InMemorySessionStore, ResumeRequest

    old_runner, request, run_request, _ = product
    if intervening_operation == "lineage":
        lineage = {"parent_session_id": "workflow-root", "causal_budget_id": "workflow-budget"}
        request = request.model_copy(update=lineage)
        run_request = run_request.model_copy(update=lineage)

    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        provider = RecordingOneShotProvider()

        def build_app(selected_store):
            app = CayuApp(session_store=selected_store, enable_logging=False)
            app.register_provider(provider)
            app.register_environment(
                Environment(
                    EnvironmentSpec(
                        name="coding",
                        execution_profile_identity=ExecutionProfileBehaviorIdentity(
                            name="tests:coding-recovery-environment",
                            behavior_version="1",
                            implementation_version="1",
                        ),
                    )
                )
            )
            app.register_agent(AgentSpec(name="coder", model="fake-model"))
            return app

        app = build_app(store)
        if intervening_operation == "lineage":
            from cayu import WorkflowBase, WorkflowSpec

            class RootWorkflow(WorkflowBase):
                spec = WorkflowSpec(name="coding-recovery-root")

                async def run(self, session_id):
                    ctx = self.context(session_id)
                    yield await ctx.start()
                    yield await ctx.completed()

            async for _ in RootWorkflow(app).run("workflow-root"):
                pass
        profile = await app.inspect_run_execution_profile(run_request)
        admitted = request.model_copy(
            update={
                "runtime": request.runtime.model_copy(
                    update={"execution_profile_fingerprint": profile}
                )
            }
        )

        def build_runner(selected_app):
            return CodingProductRunner(
                selected_app,
                source_workspace=old_runner.source_workspace,
                repository=old_runner.repository,
                source_git_authority_validator=old_runner.source_git_authority_validator,
            )

        runner = build_runner(app)

        selected_digests = []
        publish = runner.repository.publish_candidate
        append = runner.repository.append_lifecycle

        async def crash_before_compile(*args, **kwargs):
            raise RuntimeError("injected process loss")

        async def crash_at_publish(candidate):
            selected_digests.append(candidate.digest)
            if failure_phase == "published":
                await publish(candidate)
            raise RuntimeError("injected process loss")

        async def crash_after_ready(receipt):
            result = await append(receipt)
            if receipt.state is CodingProductState.READY_TO_PUBLISH:
                selected_digests.append(receipt.evidence_sha256.removeprefix("sha256:"))
                raise RuntimeError("injected process loss")
            return result

        with monkeypatch.context() as faults:
            if failure_phase == "compile":
                faults.setattr(runner, "_compile_and_publish", crash_before_compile)
            elif failure_phase == "ready":
                faults.setattr(runner.repository, "append_lifecycle", crash_after_ready)
            else:
                faults.setattr(runner.repository, "publish_candidate", crash_at_publish)
            with pytest.raises(RuntimeError, match="injected process loss"):
                await runner.run(admitted, run_request)
        assert len(provider.requests) == 1
        if intervening_operation in {
            "cancelled_after_selection",
            "source_conflict_after_selection",
        }:
            receipts, artifact_ids = await runner.repository.load_lifecycle(
                admitted.product_run_id,
                session_id=admitted.session_id,
                request_fingerprint=admitted.fingerprint,
            )
            trailing_state = (
                CodingProductState.CANCELLED
                if intervening_operation == "cancelled_after_selection"
                else CodingProductState.SOURCE_CONFLICT
            )
            await runner._append_state(
                admitted,
                list(receipts),
                list(artifact_ids),
                trailing_state,
                reason_code="test_trailing_terminal_state",
            )
        if backend == "sqlite":
            assert isinstance(store, SQLiteSessionStore)
            await store.close()
            store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        try:
            recovered_app = build_app(store)
            recovered_runner = build_runner(recovered_app)
            if intervening_operation in {
                "cancelled_after_selection",
                "source_conflict_after_selection",
            }:
                with pytest.raises(
                    CodingProductReconstructionRequiredError,
                    match="unreconciled terminal state",
                ):
                    await recovered_runner.recover_settled_execution(admitted)
                assert len(provider.requests) == 1
                return
            if intervening_operation == "ordinary_retry":
                with pytest.raises(CodingProductReconstructionRequiredError):
                    await recovered_runner.run(admitted, run_request)
            if intervening_operation == "source_change":
                (tmp_path / "source" / "example.py").write_text("changed\n", encoding="utf-8")
                with pytest.raises(
                    CodingProductReconstructionRequiredError, match="candidate conflicts"
                ):
                    await recovered_runner.recover_settled_execution(admitted)
                assert len(provider.requests) == 1
                return
            if intervening_operation == "recreate":
                await store.delete_session(admitted.session_id)
                async for _ in recovered_app.run(run_request):
                    pass
                assert len(provider.requests) == 2
                with pytest.raises(CodingProductReconstructionRequiredError, match="incarnation"):
                    await recovered_runner.recover_settled_execution(admitted)
                assert len(provider.requests) == 2
            elif intervening_operation == "resume":
                async for _ in recovered_app.resume(
                    ResumeRequest(
                        session_id=admitted.session_id,
                        messages=[Message.text("user", "another task")],
                    )
                ):
                    pass
                assert len(provider.requests) == 2
                with pytest.raises(CodingProductReconstructionRequiredError):
                    await recovered_runner.recover_settled_execution(admitted)
                assert len(provider.requests) == 2
            else:
                result = await recovered_runner.recover_settled_execution(admitted)
                if intervening_operation == "lineage":
                    session = await store.load(admitted.session_id)
                    assert session is not None
                    assert session.parent_session_id == "workflow-root"
                    assert session.causal_budget_id == "workflow-budget"
                # This ordinary environment supplies no Docker source-publication
                # receipt. Recovering execution cannot manufacture patch readiness.
                assert result.candidate.state is CodingProductState.RECONSTRUCTION_REQUIRED
                assert result.candidate.request_fingerprint == admitted.fingerprint
                if selected_digests:
                    assert selected_digests == [result.candidate.digest]
                assert await recovered_runner.recover_settled_execution(admitted) == result
                receipts, _ = await runner.repository.load_lifecycle(
                    admitted.product_run_id,
                    session_id=admitted.session_id,
                    request_fingerprint=admitted.fingerprint,
                )
                assert sum(r.state is CodingProductState.READY_TO_PUBLISH for r in receipts) == 1
                assert sum(r.state is CodingProductState.PUBLISHING for r in receipts) <= 1
                assert len(provider.requests) == 1
        finally:
            if backend == "sqlite":
                assert isinstance(store, SQLiteSessionStore)
                await store.close()

    asyncio.run(scenario())
