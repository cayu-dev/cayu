from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from cayu import (
    CayuApp,
    InMemoryTaskStore,
    LocalExecutionAttemptCoordinator,
    LocalExecutionAttemptEffectOutcome,
    LocalExecutionAttemptLimits,
    LocalExecutionAttemptQuiescence,
    LocalExecutionAttemptReceipt,
    LocalExecutionAttemptRequest,
    LocalExecutionAttemptStart,
    LocalExecutionEffectPolicy,
    LocalExecutionProcessIdentity,
    SQLiteTaskStore,
    TaskCreate,
    build_local_execution_attempt_authority,
    local_execution_parent_death_containment_platform_candidate,
)
from cayu.runtime.local_execution_attempts import (
    local_execution_attempt_receipt_sha256,
    local_execution_boot_id,
    local_execution_host_identity,
)
from cayu.vaults import SecretRedactor

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        not local_execution_parent_death_containment_platform_candidate(),
        reason="general local process-tree containment requires supported Linux primitives",
    ),
]

_FIXTURE = Path("tests/fixtures/local_execution_tree.py").resolve()
_OWNER_FIXTURE = Path("tests/fixtures/local_execution_owner.py").resolve()
_RECEIPT_WRITER_FIXTURE = Path("tests/fixtures/local_execution_receipt_writer.py").resolve()
_SUPERVISOR_FAILURE_FIXTURE = Path("tests/fixtures/local_execution_supervisor_failure.py").resolve()
_SOURCE_ROOT = Path.cwd().resolve()
_ROLES = ("root", "child", "grandchild", "background_server")


def _assert_cayu_tracebacks_exclude(error: BaseException, canary: str) -> None:
    pending: list[BaseException] = [error]
    observed: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in observed:
            continue
        observed.add(id(current))
        traceback = current.__traceback__
        while traceback is not None:
            if "/src/cayu/" in traceback.tb_frame.f_code.co_filename:
                assert canary not in repr(traceback.tb_frame.f_locals)
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def _proc_identity(pid: int) -> tuple[int, int, int] | None:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        proc_inode = Path(f"/proc/{pid}").stat().st_ino
    except (OSError, UnicodeError):
        return None
    close = stat_text.rfind(")")
    if close < 0:
        return None
    fields = stat_text[close + 2 :].split()
    if len(fields) <= 19 or fields[0] in {"Z", "X"}:
        return None
    return int(fields[2]), int(fields[19]), proc_inode


def _fixture_identities(state_dir: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads((state_dir / f"{role}.json").read_text(encoding="utf-8")) for role in _ROLES
    )


def _identity_is_live(identity: dict[str, object]) -> bool:
    observed = _proc_identity(int(identity["pid"]))
    return observed == (
        int(identity["process_group"]),
        int(identity["start_tick"]),
        int(identity["proc_inode"]),
    )


async def _wait_for_tree(state_dir: Path, *, timeout: float = 10) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if (state_dir / "tree-ready").is_file() and all(
            (state_dir / f"{role}.json").is_file() for role in _ROLES
        ):
            return
        await asyncio.sleep(0.02)
    raise TimeoutError("local execution fixture tree did not become ready")


async def _wait_for_quiescence(
    identities: tuple[dict[str, object], ...],
    *,
    timeout: float = 10,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if not any(_identity_is_live(identity) for identity in identities):
            return
        await asyncio.sleep(0.02)
    raise AssertionError("a contained local execution process remained live")


async def _claimed(store, task_id: str, *, lease_seconds: int = 300):
    await store.create_task(TaskCreate(task_id=task_id, type="local-execution"))
    task = await store.claim_task("worker-a", lease_seconds=lease_seconds)
    assert task is not None
    return task


def _request(
    tree_state: Path,
    *,
    complete: bool = False,
    deadline_seconds: float | None,
    effect_policy: LocalExecutionEffectPolicy = LocalExecutionEffectPolicy.LOCAL_ONLY,
    signal_process_group: bool = False,
    with_isolated_tool: bool = False,
    with_playwright: bool = False,
) -> LocalExecutionAttemptRequest:
    flags = (
        *(("--complete",) if complete else ()),
        *(("--signal-process-group",) if signal_process_group else ()),
        *(("--with-isolated-tool",) if with_isolated_tool else ()),
        *(("--with-playwright",) if with_playwright else ()),
    )
    environment = {"PYTHONPATH": str(_SOURCE_ROOT / "src")}
    if with_playwright and (browsers_path := os.environ.get("PLAYWRIGHT_BROWSERS_PATH")):
        environment["PLAYWRIGHT_BROWSERS_PATH"] = browsers_path
    return LocalExecutionAttemptRequest(
        effect_lineage_id="tree-effect",
        argv=(sys.executable, str(_FIXTURE), "root", str(tree_state), *flags),
        cwd=str(_SOURCE_ROOT),
        env=environment,
        effect_policy=effect_policy,
        limits=LocalExecutionAttemptLimits(
            deadline_seconds=deadline_seconds,
            term_grace_seconds=0.2,
            kill_grace_seconds=1,
        ),
    )


def test_real_cayu_child_tree_cleans_descendants_after_normal_completion(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "completed-tree")
        tree_state = tmp_path / "tree-state"
        result = await LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        ).run(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(tree_state, complete=True, deadline_seconds=5),
        )
        identities = _fixture_identities(tree_state)
        assert result.attempt.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert result.attempt.effect_outcome is LocalExecutionAttemptEffectOutcome.SUCCEEDED
        assert result.attempt.receipt is not None
        assert result.attempt.receipt.terminal_reason == "root_exit"
        assert (
            LocalExecutionAttemptCoordinator(
                store, state_dir=tmp_path / "other-state"
            ).capability_evidence.state_for("parent_death_containment")
            == "available"
        )
        await _wait_for_quiescence(identities)

    asyncio.run(scenario())


def test_concurrent_recovery_cannot_break_the_live_owner_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        from cayu.runtime import _local_execution_attempt_owner as owner_module

        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "live-owner-receipt-race")
        state_dir = tmp_path / "attempt-state"
        owner_waiting = asyncio.Event()
        allow_owner_read = asyncio.Event()
        original_load = owner_module._load_receipt_or_exact_durable_settlement

        async def gated_load(**kwargs):
            current = asyncio.current_task()
            if current is not None and current.get_name() == "live-owner-receipt-reader":
                owner_waiting.set()
                await allow_owner_read.wait()
            return await original_load(**kwargs)

        monkeypatch.setattr(
            owner_module,
            "_load_receipt_or_exact_durable_settlement",
            gated_load,
        )
        request = LocalExecutionAttemptRequest(
            effect_lineage_id="live-owner-receipt-effect",
            argv=(sys.executable, "-c", "print('live-owner-output')"),
            effect_policy=LocalExecutionEffectPolicy.LOCAL_ONLY,
            limits=LocalExecutionAttemptLimits(deadline_seconds=5),
        )
        owner = asyncio.create_task(
            LocalExecutionAttemptCoordinator(store, state_dir=state_dir).run(
                app=app,
                task=task,
                worker_id="worker-a",
                request=request,
            ),
            name="live-owner-receipt-reader",
        )
        await asyncio.wait_for(owner_waiting.wait(), timeout=10)

        recovered = await LocalExecutionAttemptCoordinator(
            store,
            state_dir=state_dir,
        ).recover(worker_id="recovery-worker")
        assert len(recovered) == 1
        assert recovered[0].quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert not list(state_dir.glob("*.receipt.json"))

        allow_owner_read.set()
        result = await asyncio.wait_for(owner, timeout=10)
        assert result.attempt == recovered[0]
        assert result.stdout == "live-owner-output\n"
        assert result.stdout_truncated is False

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_index", [1, 2, 3])
def test_post_spawn_task_setup_failure_releases_supervisor_ownership(
    failure_index: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        from cayu.runtime import _local_execution_attempt_owner as owner_module

        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, f"post-spawn-setup-{failure_index}")
        state_dir = tmp_path / f"attempt-state-{failure_index}"
        dispatch_marker = tmp_path / f"dispatched-{failure_index}"
        request = LocalExecutionAttemptRequest(
            effect_lineage_id="post-spawn-setup-effect",
            argv=(
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path({str(dispatch_marker)!r}).write_text('dispatched')"
                ),
            ),
            effect_policy=LocalExecutionEffectPolicy.LOCAL_ONLY,
        )
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )
        original_create = owner_module._create_local_execution_task
        create_calls = 0

        def fail_selected_task(operation, *, name=None):
            nonlocal create_calls
            create_calls += 1
            if create_calls == failure_index:
                close = getattr(operation, "close", None)
                if callable(close):
                    close()
                raise RuntimeError("injected post-spawn task creation failure")
            return original_create(operation, name=name)

        monkeypatch.setattr(
            owner_module,
            "_create_local_execution_task",
            fail_selected_task,
        )
        with pytest.raises(BaseException) as captured:
            await LocalExecutionAttemptCoordinator(store, state_dir=state_dir).run(
                app=app,
                task=task,
                worker_id="worker-a",
                request=request,
            )
        assert "injected post-spawn task creation failure" in repr(captured.value)
        assert not dispatch_marker.exists()

        record = await store.load_local_execution_attempt(authority.attempt_id)
        assert record is not None
        assert record.quiescence is LocalExecutionAttemptQuiescence.NOT_DISPATCHED
        assert record.receipt is not None
        assert record.receipt.terminal_reason in {
            "cancelled_before_dispatch",
            "owner_gone_before_dispatch",
        }

    asyncio.run(scenario())


def test_independent_coordinators_do_not_share_a_host_rendezvous(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        stores = (InMemoryTaskStore(), InMemoryTaskStore())
        apps = tuple(CayuApp(task_store=store, enable_logging=False) for store in stores)
        tasks = (
            await _claimed(stores[0], "shared-caller-task"),
            await _claimed(stores[1], "shared-caller-task"),
        )
        release = tmp_path / "release-independent-attempts"
        markers = tuple(tmp_path / f"independent-{index}.ready" for index in range(2))

        async def run_one(index: int):
            command = (
                "from pathlib import Path; import time; "
                f"marker=Path({str(markers[index])!r}); "
                f"release=Path({str(release)!r}); "
                "marker.write_text('ready'); "
                "\nwhile not release.exists(): time.sleep(0.01)"
            )
            return await LocalExecutionAttemptCoordinator(
                stores[index],
                state_dir=tmp_path / f"attempt-state-{index}",
            ).run(
                app=apps[index],
                task=tasks[index],
                worker_id="worker-a",
                request=LocalExecutionAttemptRequest(
                    effect_lineage_id="shared-caller-effect",
                    argv=(sys.executable, "-c", command),
                    effect_policy=LocalExecutionEffectPolicy.LOCAL_ONLY,
                    limits=LocalExecutionAttemptLimits(deadline_seconds=5),
                ),
            )

        owners = tuple(
            asyncio.create_task(run_one(index), name=f"independent-owner-{index}")
            for index in range(2)
        )
        try:
            deadline = asyncio.get_running_loop().time() + 10
            while not all(marker.is_file() for marker in markers):
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("independent local attempts did not both dispatch")
                await asyncio.sleep(0.02)
            assert all(not owner.done() for owner in owners)
            release.write_text("release", encoding="utf-8")
            results = await asyncio.gather(*owners)
        finally:
            release.touch(exist_ok=True)
            for owner in owners:
                if not owner.done():
                    owner.cancel()
            await asyncio.gather(*owners, return_exceptions=True)

        assert all(
            result.attempt.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
            for result in results
        )

    asyncio.run(scenario())


def test_supervisor_reconciles_an_exact_concurrent_receipt_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.runtime import _local_execution_supervisor as supervisor_module

    receipt_path = tmp_path / "settlement.receipt.json"
    payload = {"attempt_id": "exact-promotion", "state": "quiescent"}
    original_replace = supervisor_module.os.replace

    def promote_then_lose_acknowledgement(source, target) -> None:
        original_replace(source, target)
        raise FileNotFoundError("concurrent recovery promoted the exact stage")

    monkeypatch.setattr(
        supervisor_module.os,
        "replace",
        promote_then_lose_acknowledgement,
    )
    supervisor_module._atomic_receipt(receipt_path, payload)

    assert json.loads(receipt_path.read_text(encoding="utf-8")) == payload
    assert not receipt_path.with_name(f"{receipt_path.name}.staging").exists()


@pytest.mark.parametrize(
    ("secret", "prefix", "max_output_bytes", "expected"),
    [
        (
            "local-output-boundary-secret-canary",
            "safe:",
            len("safe:local-output-boundary-secret-canary") - 1,
            "safe:",
        ),
        ("tinykey8", "", len("tinykey8"), ""),
    ],
)
def test_local_attempt_redacts_before_bounding_process_output(
    secret: str,
    prefix: str,
    max_output_bytes: int,
    expected: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        task = await _claimed(store, f"bounded-output-{max_output_bytes}")
        request = LocalExecutionAttemptRequest(
            effect_lineage_id=f"bounded-output-{max_output_bytes}",
            argv=(
                sys.executable,
                "-c",
                (
                    "import os; "
                    "value=(os.environ['OUTPUT_PREFIX']+os.environ['PRIVATE_VALUE']).encode(); "
                    "os.write(1, value); os.write(2, value)"
                ),
            ),
            env={"OUTPUT_PREFIX": prefix, "PRIVATE_VALUE": secret},
            effect_policy=LocalExecutionEffectPolicy.LOCAL_ONLY,
            limits=LocalExecutionAttemptLimits(
                deadline_seconds=5,
                max_output_bytes=max_output_bytes,
            ),
        )
        result = await LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / f"attempt-state-{max_output_bytes}",
        ).run(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )

        assert result.stdout == expected
        assert result.stderr == expected
        assert result.stdout_truncated is True
        assert result.stderr_truncated is True
        assert secret not in result.model_dump_json()

    asyncio.run(scenario())


def test_supervisor_retains_cleanup_authority_until_quiescent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cayu.runtime import _local_execution_supervisor as supervisor

    class SupervisorExit(BaseException):
        pass

    class EmptyListener:
        def accept(self):
            raise BlockingIOError

    receipt: dict[str, object] = {}
    settlement_calls = 0

    def settle_tree(**_kwargs):
        nonlocal settlement_calls
        settlement_calls += 1
        if settlement_calls == 1:
            return 0, False, True, True, 1
        return 0, True, False, True, 1

    monkeypatch.setattr(supervisor, "_reap_children", lambda *_args: (0, False))
    monkeypatch.setattr(supervisor, "_settle_tree", settle_tree)
    monkeypatch.setattr(
        supervisor,
        "_publish_receipt",
        lambda _listener, _path, payload: receipt.update(payload),
    )
    monkeypatch.setattr(supervisor, "_send", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        supervisor,
        "_exit",
        lambda _code: (_ for _ in ()).throw(SupervisorExit()),
    )

    args = SimpleNamespace(
        attempt_id="lex_" + "a" * 32,
        boot_id="boot-a",
        deadline_seconds=None,
        effect_policy="local_only",
        host_identity="host-a",
        kill_grace_seconds=1,
        nonce="b" * 64,
        receipt_path=str(tmp_path / "receipt.json"),
        request_sha256="c" * 64,
        term_grace_seconds=1,
    )
    with pytest.raises(SupervisorExit):
        supervisor._supervise_started_attempt(
            args=args,
            control=SimpleNamespace(),
            listener=EmptyListener(),
            state={},
            control_buffer=bytearray(),
            child=SimpleNamespace(pid=123),
            root_identity={
                "pid": 123,
                "process_group": 123,
                "start_tick": 1,
                "proc_inode": 2,
            },
            detached=False,
        )

    parsed = LocalExecutionAttemptReceipt.model_validate(receipt)
    assert settlement_calls == 2
    assert parsed.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
    assert parsed.effect_outcome is LocalExecutionAttemptEffectOutcome.SUCCEEDED


def test_supervisor_always_attempts_exact_kill_with_positive_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.runtime import _local_execution_supervisor as supervisor

    process = SimpleNamespace(start_tick=1, proc_inode=2)
    observed_signals: list[int] = []
    monkeypatch.setattr(
        supervisor,
        "_freeze_to_closure",
        lambda _pid, _deadline: ({123: process}, False),
    )
    monkeypatch.setattr(
        supervisor,
        "_signal_exact",
        lambda _pid, _stat, signal_number: observed_signals.append(signal_number),
    )
    monkeypatch.setattr(supervisor, "_reap_children", lambda *_args: (None, True))
    monkeypatch.setattr(supervisor, "_descendants", lambda _pid: {123: process})

    _root_exit, quiescent, _term_sent, kill_sent, _observed = supervisor._settle_tree(
        supervisor_pid=100,
        root_pid=123,
        root_exit=None,
        term_grace=1e-12,
        kill_grace=1e-12,
        force=True,
    )

    assert quiescent is False
    assert kill_sent is True
    assert observed_signals == [signal.SIGKILL]


def test_supervisor_internal_failure_after_dispatch_settles_the_complete_tree(
    tmp_path: Path,
) -> None:
    tree_state = tmp_path / "tree-state"
    receipt_path = tmp_path / "supervisor.receipt.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_SOURCE_ROOT / "src")
    owner = subprocess.Popen(
        [
            sys.executable,
            str(_SUPERVISOR_FAILURE_FIXTURE),
            str(_FIXTURE),
            str(tree_state),
            str(receipt_path),
        ],
        cwd=_SOURCE_ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        owner.wait(timeout=10)
        assert owner.returncode == 72
        identities = tuple(
            json.loads((tree_state / f"{role}.json").read_text(encoding="utf-8"))
            for role in ("child", "grandchild", "background_server")
        )
        asyncio.run(_wait_for_quiescence(identities))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["terminal_reason"] == "supervisor_internal_failure"
        assert receipt["quiescence"] == "quiescent"
        assert receipt["effect_outcome"] == "outcome_unknown"
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)


def test_root_process_group_signal_cannot_kill_the_cleanup_supervisor(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "root-process-group-signal")
        tree_state = tmp_path / "tree-state"
        result = await LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        ).run(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(
                tree_state,
                deadline_seconds=10,
                signal_process_group=True,
            ),
        )
        identities = _fixture_identities(tree_state)
        assert result.attempt.start is not None
        assert result.attempt.receipt is not None
        assert result.attempt.receipt.root is not None
        assert (
            result.attempt.start.supervisor.process_group
            != result.attempt.receipt.root.process_group
        )
        assert result.attempt.receipt.terminal_reason == "root_exit"
        assert result.attempt.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        await _wait_for_quiescence(identities)

    asyncio.run(scenario())


def test_quiescent_attempt_allows_same_worker_exact_replacement_execution(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "same-worker-replacement")
        tree_state = tmp_path / "tree-state"
        request = _request(tree_state, complete=True, deadline_seconds=5)
        coordinator = LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        )
        first = await coordinator.run(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )
        await _wait_for_quiescence(_fixture_identities(tree_state))
        shutil.rmtree(tree_state)
        await store.release_task(task.id, "worker-a")
        await asyncio.sleep(0.001)
        replacement_task = await store.claim_task("worker-a", lease_seconds=300)
        assert replacement_task is not None

        replacement = await coordinator.run(
            app=app,
            task=replacement_task,
            worker_id="worker-a",
            request=request,
        )
        assert replacement.attempt.authority.attempt_id != first.attempt.authority.attempt_id
        assert replacement.attempt.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert replacement.attempt.effect_outcome is LocalExecutionAttemptEffectOutcome.SUCCEEDED
        await _wait_for_quiescence(_fixture_identities(tree_state))

    asyncio.run(scenario())


def test_clean_process_exit_does_not_prove_non_idempotent_external_outcome(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "external-completed-tree")
        result = await LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        ).run(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(
                tmp_path / "tree-state",
                complete=True,
                deadline_seconds=5,
                effect_policy=LocalExecutionEffectPolicy.NON_IDEMPOTENT_EXTERNAL,
            ),
        )
        assert result.attempt.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert result.attempt.effect_outcome is LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN
        assert result.attempt.retry_admissible is False

    asyncio.run(scenario())


def test_root_dispatch_is_settled_when_durable_root_publication_fails(
    tmp_path: Path,
) -> None:
    class FailRootPublicationStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.start_publications = 0

        async def start_local_execution_attempt(self, start):
            self.start_publications += 1
            if self.start_publications == 2:
                raise RuntimeError("injected root-publication failure")
            return await super().start_local_execution_attempt(start)

    async def scenario() -> None:
        store = FailRootPublicationStore()
        canary = "local-attempt-start-traceback-secret-canary"
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(canary),
            enable_logging=False,
        )
        task = await _claimed(store, "root-publication-failure")
        request = _request(
            tmp_path / "tree-state",
            deadline_seconds=30,
            effect_policy=LocalExecutionEffectPolicy.NON_IDEMPOTENT_EXTERNAL,
        )
        request = request.model_copy(update={"env": {**request.env, "PRIVATE_VALUE": canary}})
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )

        with pytest.raises(RuntimeError, match="root-publication failure") as captured:
            await LocalExecutionAttemptCoordinator(
                store,
                state_dir=tmp_path / "attempt-state",
            ).run(
                app=app,
                task=task,
                worker_id="worker-a",
                request=request,
            )
        _assert_cayu_tracebacks_exclude(captured.value, canary)

        record = await store.load_local_execution_attempt(authority.attempt_id)
        assert record is not None
        assert record.receipt is not None
        assert record.receipt.root is not None
        assert record.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert record.effect_outcome is LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN
        assert record.retry_admissible is False
        assert not list((tmp_path / "attempt-state").glob("*.receipt.json"))

    asyncio.run(scenario())


def test_pre_dispatch_publication_failure_remains_safely_retryable(
    tmp_path: Path,
) -> None:
    class FailSupervisorPublicationStore(InMemoryTaskStore):
        async def start_local_execution_attempt(self, start):
            raise RuntimeError("injected supervisor-publication failure")

    async def scenario() -> None:
        store = FailSupervisorPublicationStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "supervisor-publication-failure")
        request = _request(
            tmp_path / "tree-state",
            deadline_seconds=30,
            effect_policy=LocalExecutionEffectPolicy.NON_IDEMPOTENT_EXTERNAL,
        )
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )

        with pytest.raises(RuntimeError, match="supervisor-publication failure"):
            await LocalExecutionAttemptCoordinator(
                store,
                state_dir=tmp_path / "attempt-state",
            ).run(
                app=app,
                task=task,
                worker_id="worker-a",
                request=request,
            )

        record = await store.load_local_execution_attempt(authority.attempt_id)
        assert record is not None
        assert record.receipt is not None
        assert record.receipt.root is None
        assert record.quiescence is LocalExecutionAttemptQuiescence.NOT_DISPATCHED
        assert record.effect_outcome is LocalExecutionAttemptEffectOutcome.NOT_STARTED
        assert record.retry_admissible is True

    asyncio.run(scenario())


def test_real_cayu_child_tree_is_quiescent_after_hard_deadline(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "deadline-tree")
        coordinator = LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        )
        tree_state = tmp_path / "tree-state"
        result = await coordinator.run(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(
                tree_state,
                deadline_seconds=3,
                effect_policy=LocalExecutionEffectPolicy.NON_IDEMPOTENT_EXTERNAL,
            ),
        )
        identities = _fixture_identities(tree_state)
        assert result.attempt.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert result.attempt.effect_outcome is LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN
        assert result.attempt.retry_admissible is False
        assert result.attempt.receipt is not None
        assert result.attempt.receipt.terminal_reason == "deadline"
        assert result.attempt.receipt.kill_sent is True
        await _wait_for_quiescence(identities)

        socket_path = tree_state / "background.sock"
        socket_path.unlink()
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.bind(str(socket_path))
        finally:
            probe.close()

    asyncio.run(scenario())


def test_real_cayu_child_tree_is_quiescent_before_cancellation_returns(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        canary = "local-attempt-cancellation-traceback-secret-canary"
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(canary),
            enable_logging=False,
        )
        claimed = await _claimed(store, "cancelled-tree")
        request = _request(tmp_path / "tree-state", deadline_seconds=None)
        request = request.model_copy(update={"env": {**request.env, "PRIVATE_VALUE": canary}})
        coordinator = LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        )
        owner = asyncio.create_task(
            coordinator.run(
                app=app,
                task=claimed,
                worker_id="worker-a",
                request=request,
            )
        )
        await _wait_for_tree(tmp_path / "tree-state")
        identities = _fixture_identities(tmp_path / "tree-state")
        owner.cancel("caller stopped")
        assert owner.cancelling() == 1
        with pytest.raises(asyncio.CancelledError, match="caller stopped") as captured:
            await owner
        _assert_cayu_tracebacks_exclude(captured.value, canary)
        assert owner.cancelled() is True
        assert owner.cancelling() == 1
        await _wait_for_quiescence(identities)
        authority = build_local_execution_attempt_authority(
            app=app,
            task=claimed,
            worker_id="worker-a",
            request=request,
        )
        record = await store.load_local_execution_attempt(authority.attempt_id)
        assert record is not None
        assert record.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert not list((tmp_path / "attempt-state").glob("*.receipt.json"))

    asyncio.run(scenario())


def test_cancellation_during_final_settlement_waits_for_durable_quiescence(
    tmp_path: Path,
) -> None:
    class BlockingSettlementStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.settlement_entered = asyncio.Event()
            self.allow_settlement = asyncio.Event()

        async def settle_local_execution_attempt(self, settlement):
            self.settlement_entered.set()
            await self.allow_settlement.wait()
            return await super().settle_local_execution_attempt(settlement)

    async def scenario() -> None:
        store = BlockingSettlementStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "cancelled-final-settlement")
        tree_state = tmp_path / "tree-state"
        request = _request(tree_state, complete=True, deadline_seconds=5)
        state_dir = tmp_path / "attempt-state"
        owner = asyncio.create_task(
            LocalExecutionAttemptCoordinator(store, state_dir=state_dir).run(
                app=app,
                task=task,
                worker_id="worker-a",
                request=request,
            )
        )
        await asyncio.wait_for(store.settlement_entered.wait(), timeout=10)
        identities = _fixture_identities(tree_state)

        owner.cancel("caller stopped during settlement")
        assert owner.cancelling() == 1
        await asyncio.sleep(0)
        assert owner.done() is False

        store.allow_settlement.set()
        with pytest.raises(
            asyncio.CancelledError,
            match="caller stopped during settlement",
        ):
            await owner
        assert owner.cancelled() is True
        assert owner.cancelling() == 1

        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )
        record = await store.load_local_execution_attempt(authority.attempt_id)
        assert record is not None
        assert record.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert record.retry_admissible is True
        assert not list(state_dir.glob("*.receipt.json"))
        await _wait_for_quiescence(identities)

    asyncio.run(scenario())


def test_exact_replay_clears_receipt_after_settlement_acknowledgement_loss(
    tmp_path: Path,
) -> None:
    class CommitThenRaiseSettlementStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.fail_acknowledgement = True
            self.start_publications = 0

        async def start_local_execution_attempt(self, start):
            self.start_publications += 1
            return await super().start_local_execution_attempt(start)

        async def settle_local_execution_attempt(self, settlement):
            result = await super().settle_local_execution_attempt(settlement)
            if self.fail_acknowledgement:
                self.fail_acknowledgement = False
                raise RuntimeError("injected settlement acknowledgement loss")
            return result

    async def scenario() -> None:
        store = CommitThenRaiseSettlementStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "settlement-acknowledgement-loss")
        tree_state = tmp_path / "tree-state"
        request = _request(tree_state, complete=True, deadline_seconds=5)
        state_dir = tmp_path / "attempt-state"
        coordinator = LocalExecutionAttemptCoordinator(store, state_dir=state_dir)

        with pytest.raises(RuntimeError, match="settlement acknowledgement loss"):
            await coordinator.run(
                app=app,
                task=task,
                worker_id="worker-a",
                request=request,
            )
        starts_after_first_run = store.start_publications
        assert starts_after_first_run == 2
        assert len(list(state_dir.glob("*.receipt.json"))) == 1

        replayed = await coordinator.run(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )
        assert replayed.attempt.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert replayed.stdout == ""
        assert replayed.stderr == ""
        assert replayed.stdout_truncated is True
        assert replayed.stderr_truncated is True
        assert store.start_publications == starts_after_first_run
        assert not list(state_dir.glob("*.receipt.json"))
        await _wait_for_quiescence(_fixture_identities(tree_state))

    asyncio.run(scenario())


def test_expired_task_lease_does_not_release_a_live_tree_for_retry(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "lease-loss-tree", lease_seconds=1)
        coordinator = LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        )
        owner = asyncio.create_task(
            coordinator.run(
                app=app,
                task=task,
                worker_id="worker-a",
                request=_request(tmp_path / "tree-state", deadline_seconds=10),
            )
        )
        await _wait_for_tree(tmp_path / "tree-state")
        await asyncio.sleep(1.05)
        assert await store.reclaim_expired() == []
        assert await store.claim_task("replacement", lease_seconds=300) is None
        result = await owner
        assert result.attempt.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        reclaimed = await store.reclaim_expired()
        assert [item.id for item in reclaimed] == [task.id]

    asyncio.run(scenario())


def test_sigkill_owner_retains_cleanup_and_retry_fence(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = tmp_path / "tasks.sqlite"
        attempt_state = tmp_path / "attempt-state"
        tree_state = tmp_path / "tree-state"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(_SOURCE_ROOT / "src")
        owner = subprocess.Popen(
            [
                sys.executable,
                str(_OWNER_FIXTURE),
                str(database),
                str(attempt_state),
                str(tree_state),
                str(_FIXTURE),
                str(_SOURCE_ROOT),
            ],
            cwd=_SOURCE_ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            await _wait_for_tree(tree_state)
            identities = _fixture_identities(tree_state)
            os.kill(owner.pid, signal.SIGKILL)
            await asyncio.to_thread(owner.wait, 5)
            await asyncio.sleep(1.05)

            store = SQLiteTaskStore(database)
            try:
                assert await store.reclaim_expired() == []
                assert (
                    await store.claim_task(
                        "replacement",
                        lease_seconds=300,
                    )
                    is None
                )
                coordinator = LocalExecutionAttemptCoordinator(
                    store,
                    state_dir=attempt_state,
                )
                recovery_deadline = asyncio.get_running_loop().time() + 10
                recovered = ()
                while not recovered and asyncio.get_running_loop().time() < recovery_deadline:
                    recovered = await coordinator.recover(worker_id="recovery-worker")
                    if not recovered:
                        await asyncio.sleep(0.05)
                assert recovered
                assert recovered[0].quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
                await _wait_for_quiescence(identities)
                reclaimed = await store.reclaim_expired()
                assert [task.id for task in reclaimed] == ["parent-death-task"]
                replacement = await store.claim_task("replacement", lease_seconds=300)
                assert replacement is not None
                assert replacement.id == "parent-death-task"
            finally:
                await store.close()
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait(timeout=5)

    asyncio.run(scenario())


def test_stale_pid_identity_never_targets_the_unrelated_live_process(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "stale-pid")
        request = LocalExecutionAttemptRequest(
            effect_lineage_id="stale-pid-effect",
            argv=("/usr/bin/true",),
            effect_policy=LocalExecutionEffectPolicy.LOCAL_ONLY,
        )
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )
        await store.prepare_local_execution_attempt(authority)
        observed = _proc_identity(os.getpid())
        assert observed is not None
        process_group, start_tick, proc_inode = observed
        await store.start_local_execution_attempt(
            LocalExecutionAttemptStart(
                attempt_id=authority.attempt_id,
                request_sha256=authority.request_sha256,
                host_identity=local_execution_host_identity(),
                boot_id=local_execution_boot_id(),
                supervisor_nonce="a" * 64,
                rendezvous_identity="b" * 64,
                supervisor=LocalExecutionProcessIdentity(
                    pid=os.getpid(),
                    process_group=process_group,
                    start_tick=start_tick + 1,
                    proc_inode=proc_inode,
                ),
                root=None,
                started_at=datetime.now(UTC),
            )
        )
        await store.release_task(task.id, "worker-a")
        recovered = await LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        ).recover(worker_id="recovery-worker")
        assert len(recovered) == 1
        assert recovered[0].quiescence is LocalExecutionAttemptQuiescence.UNAVAILABLE
        assert _proc_identity(os.getpid()) == observed

    asyncio.run(scenario())


def test_process_loss_after_receipt_staging_fsync_recovers_exact_settlement(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "staged-receipt-process-loss")
        request = LocalExecutionAttemptRequest(
            effect_lineage_id="staged-receipt-effect",
            argv=("/usr/bin/true",),
            effect_policy=LocalExecutionEffectPolicy.LOCAL_ONLY,
        )
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id="worker-a",
            request=request,
        )
        await store.prepare_local_execution_attempt(authority)
        started = LocalExecutionAttemptStart(
            attempt_id=authority.attempt_id,
            request_sha256=authority.request_sha256,
            host_identity=local_execution_host_identity(),
            boot_id=local_execution_boot_id(),
            supervisor_nonce="a" * 64,
            rendezvous_identity="b" * 64,
            supervisor=LocalExecutionProcessIdentity(
                pid=2_000_000_000,
                process_group=2_000_000_000,
                start_tick=1,
                proc_inode=1,
            ),
            root=LocalExecutionProcessIdentity(
                pid=2_000_000_001,
                process_group=2_000_000_001,
                start_tick=1,
                proc_inode=1,
            ),
            started_at=datetime.now(UTC),
        )
        await store.start_local_execution_attempt(started)
        await store.release_task(task.id, "worker-a")

        state_dir = tmp_path / "attempt-state"
        state_dir.mkdir(mode=0o700)
        receipt_path = state_dir / f"{authority.request_sha256}.receipt.json"
        payload_path = tmp_path / "receipt-payload.json"
        receipt_payload = {
            "attempt_id": authority.attempt_id,
            "boot_id": started.boot_id,
            "descendants_observed": 2,
            "effect_outcome": LocalExecutionAttemptEffectOutcome.SUCCEEDED.value,
            "exit_code": 0,
            "host_identity": started.host_identity,
            "kill_sent": False,
            "quiescence": LocalExecutionAttemptQuiescence.QUIESCENT.value,
            "request_sha256": authority.request_sha256,
            "root": started.root.model_dump(mode="json") if started.root is not None else None,
            "settled_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "supervisor_nonce": started.supervisor_nonce,
            "term_sent": False,
            "terminal_reason": "test_settlement",
        }
        receipt_payload["receipt_sha256"] = local_execution_attempt_receipt_sha256(receipt_payload)
        receipt = LocalExecutionAttemptReceipt.model_validate(receipt_payload)
        payload_path.write_text(receipt.model_dump_json(), encoding="utf-8")
        writer_ready = tmp_path / "receipt-writer-ready"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(_SOURCE_ROOT / "src")
        writer = subprocess.Popen(
            [
                sys.executable,
                str(_RECEIPT_WRITER_FIXTURE),
                str(payload_path),
                str(receipt_path),
                str(writer_ready),
            ],
            cwd=_SOURCE_ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not writer_ready.is_file():
                if writer.poll() is not None:
                    raise AssertionError("receipt writer exited before staging")
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("receipt writer did not reach the staging boundary")
                await asyncio.sleep(0.01)
            staging_path = receipt_path.with_name(f"{receipt_path.name}.staging")
            assert staging_path.is_file()
            os.kill(writer.pid, signal.SIGKILL)
            await asyncio.to_thread(writer.wait, 5)
            assert not receipt_path.exists()
            recovered = await LocalExecutionAttemptCoordinator(
                store,
                state_dir=state_dir,
            ).recover(worker_id="recovery-worker")
            assert len(recovered) == 1
            assert recovered[0].quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
            assert recovered[0].receipt is not None
            assert recovered[0].receipt.terminal_reason == "test_settlement"
            assert not receipt_path.exists()
            assert not staging_path.exists()
        finally:
            if writer.poll() is None:
                writer.kill()
                writer.wait(timeout=5)

    asyncio.run(scenario())


def test_process_isolated_tool_tree_composes_with_general_attempt_containment(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "isolated-tool-tree")
        tree_state = tmp_path / "tree-state"
        coordinator = LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        )
        owner = asyncio.create_task(
            coordinator.run(
                app=app,
                task=task,
                worker_id="worker-a",
                request=_request(
                    tree_state,
                    deadline_seconds=20,
                    with_isolated_tool=True,
                ),
            )
        )
        await _wait_for_tree(tree_state)
        evidence_deadline = asyncio.get_running_loop().time() + 25
        isolated_pid_path = tree_state / "isolated-grandchild.pid"
        while not (tree_state / "isolated-started").is_file() or not isolated_pid_path.is_file():
            if owner.done():
                result = owner.result()
                raise AssertionError(
                    "outer attempt settled before the nested process-isolated tool started: "
                    f"{result.attempt.model_dump(mode='json')}"
                )
            if asyncio.get_running_loop().time() >= evidence_deadline:
                raise TimeoutError("nested process-isolated tool did not start")
            await asyncio.sleep(0.02)
        isolated_pid = int(isolated_pid_path.read_text(encoding="utf-8"))
        isolated_identity = _proc_identity(isolated_pid)
        assert isolated_identity is not None

        result = await owner
        assert result.attempt.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert result.attempt.receipt is not None
        assert result.attempt.receipt.terminal_reason == "deadline"
        process_group, start_tick, proc_inode = isolated_identity
        await _wait_for_quiescence(
            (
                {
                    "pid": isolated_pid,
                    "process_group": process_group,
                    "start_tick": start_tick,
                    "proc_inode": proc_inode,
                },
            )
        )

    asyncio.run(scenario())


def test_real_playwright_tree_is_contained_when_chromium_is_installed(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await _claimed(store, "playwright-tree")
        tree_state = tmp_path / "tree-state"
        result = await LocalExecutionAttemptCoordinator(
            store,
            state_dir=tmp_path / "attempt-state",
        ).run(
            app=app,
            task=task,
            worker_id="worker-a",
            request=_request(
                tree_state,
                deadline_seconds=20,
                with_playwright=True,
            ),
        )
        if (tree_state / "playwright-unavailable").is_file():
            if os.environ.get("CAYU_REQUIRE_PLAYWRIGHT_CONTAINMENT") == "1":
                pytest.fail("The required Playwright containment tracer could not launch Chromium.")
            pytest.skip("Playwright Chromium is not installed")
        assert (tree_state / "playwright-ready").is_file(), result.attempt.model_dump(mode="json")
        chromium_identities = tuple(
            json.loads((tree_state / "playwright-processes.json").read_text(encoding="utf-8"))
        )
        assert chromium_identities
        assert result.attempt.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
        assert result.attempt.receipt is not None
        assert result.attempt.receipt.descendants_observed >= len(_ROLES)
        await _wait_for_quiescence(chromium_identities)

    asyncio.run(scenario())
