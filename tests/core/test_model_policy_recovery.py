"""Public policy lifecycle, maintenance and deadline recovery boundaries."""

import asyncio
import threading
import time

import pytest
from tests.core.test_model_policy_runtime import Channel, make_app
from tests.core.test_model_policy_runtime import store_factory as store_factory

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.model_policy import ModelPolicy, ModelPolicyController, PolicyContractError
from cayu.runtime._policy_storage import ModelPolicyStore, PolicyStorageCommand, parse_state
from cayu.sessions.outcomes import run_to_completion

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_shutdown_preserves_inflight_sqlite_failure(tmp_path, fails, cancel_caller):
    from cayu.storage.model_policy_sqlite import SQLiteModelPolicyStore

    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    failure = OSError("controlled policy write failure")

    class Store(SQLiteModelPolicyStore):
        armed = False
        writing = False

        async def execute(self, command):
            if self.armed and command.action == "write":
                self.writing = True
            return await super().execute(command)

    native = Store(tmp_path / "shutdown-policy.sqlite")
    connection = native._connection

    class Connection:
        def __getattr__(self, name):
            return getattr(connection, name)

        def __enter__(self):
            connection.__enter__()
            return self

        def __exit__(self, *args):
            return connection.__exit__(*args)

        def execute(self, sql, *args):
            result = connection.execute(sql, *args)
            if native.writing and sql.startswith("INSERT INTO cayu_model_policy_state"):
                native.armed = native.writing = False
                loop.call_soon_threadsafe(entered.set)
                if not release.wait(10):
                    raise TimeoutError("test write barrier expired")
                if fails:
                    raise failure
            return result

    native._connection = Connection()
    app, controller, _ = make_app(native, Channel(), interval=0.02)
    worker = None

    async def serve():
        nonlocal worker
        async with app.model_policy_lifespan():
            worker = app.model_policy._tasks[-1]
            native.armed = True
            await entered.wait()

    caller = asyncio.create_task(serve())
    try:
        await asyncio.wait_for(entered.wait(), 5)

        async def wait_for_shutdown():
            while worker is None or not worker.cancelling():
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_shutdown(), 5)
        assert not caller.done() and not worker.done()
        if cancel_caller:
            caller.cancel()
            await asyncio.sleep(0)
        release.set()
        caught = None
        try:
            await asyncio.wait_for(asyncio.shield(caller), 5)
        except BaseException as exc:
            caught = exc
        assert caller.cancelled() is cancel_caller
        assert caller.cancelling() == int(cancel_caller)
        if cancel_caller:
            assert isinstance(caught, asyncio.CancelledError)
            # Awaiting a cancelled task through shield manufactures a new signal;
            # retrieve its original outcome for the failure-evidence assertion.
            try:
                await caller
            except asyncio.CancelledError as original:
                caught = original
        elif fails:
            assert isinstance(caught, ExceptionGroup)
        else:
            assert caught is None
        seen = set()

        def occurrences(error):
            if error is None or id(error) in seen:
                return 0
            seen.add(id(error))
            return (
                int(error is failure)
                + occurrences(error.__cause__)
                + sum(
                    occurrences(child)
                    for child in (error.exceptions if isinstance(error, BaseExceptionGroup) else ())
                )
            )

        assert occurrences(caught) == int(fails)
        assert worker.cancelled() and worker.cancelling() == 1
        assert controller.status == "stopped"
        assert not app.model_policy._tasks
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        await app.stop_model_policy()
        await native.close()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("phase", ["observation", "acknowledgement"])
async def test_maintenance_publication_does_not_reject_public_run(store_factory, phase):
    native, channel = store_factory(), Channel()
    entered, finish = asyncio.Event(), asyncio.Event()

    class Maintenance(ModelPolicyStore):
        pause = False

        async def execute(self, command):
            if self.pause and command.action == "write":
                before, after = parse_state(command.expected.state), parse_state(command.state)
                matches = (
                    before["observation"] != after["observation"]
                    if phase == "observation"
                    else before["reports"] != after["reports"]
                )
                if matches:
                    self.pause = False
                    entered.set()
                    await finish.wait()
            return await native.execute(command)

    store = Maintenance()
    channel.fail_ack = phase == "acknowledgement"
    app, controller, _ = make_app(store, channel)
    await app.start_model_policy()
    channel.fail_ack = False
    store.pause = True
    task = asyncio.create_task(controller.poll_once())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        result = await run_to_completion(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "during maintenance")],
            ),
        )
        assert result.ok, result.error
        assert (await app.session_store.load(result.session_id)).model == "model-a"
    finally:
        finish.set()
        await task
        await app.stop_model_policy()
        await native.close()


async def test_expired_owner_recovers_but_cannot_displace_replacement(store_factory, monkeypatch):
    import importlib

    native, channel = store_factory(), Channel()
    module = importlib.import_module(type(native).__module__)
    transition = module.transition
    elapsed = 0

    def clocked(row, command, now):
        return transition(row, command, now + elapsed)

    monkeypatch.setattr(module, "transition", clocked)
    app, controller, _ = make_app(native, channel)
    await app.start_model_policy()
    replacement = None
    try:
        elapsed = 61
        await controller.poll_once()
        assert controller.selection().target.model == "model-a"
        elapsed = 122
        replacement, other, _ = make_app(native, channel)
        await replacement.start_model_policy()
        with pytest.raises(PolicyContractError):
            await controller.poll_once()
        with pytest.raises(PolicyContractError):
            controller.selection()
        assert other.selection().target.model == "model-a"
        await app.stop_model_policy()
        await other.poll_once()
        assert other.selection().target.model == "model-a"
    finally:
        await app.stop_model_policy()
        if replacement is not None:
            await replacement.stop_model_policy()
        await native.close()


async def test_background_worker_adopts_changed_default(store_factory):
    native, channel = store_factory(), Channel()
    app, controller, _ = make_app(native, channel, interval=0.02)
    await app.start_model_policy()
    try:
        channel.model, channel.revision = "model-b", 2
        async with asyncio.timeout(5):
            while len(channel.receipts) < 2:
                await asyncio.sleep(0.01)
        assert controller.selection().target.model == "model-b"
        result = await run_to_completion(
            app, RunRequest(agent_name="assistant", messages=[Message.text("user", "updated")])
        )
        assert result.ok, result.error
        assert (await app.session_store.load(result.session_id)).model == "model-b"
    finally:
        await app.stop_model_policy()
        await native.close()


async def test_restart_adopts_uninstalled_cached_snapshot_without_renewing_expiry(store_factory):
    native, channel = store_factory(), Channel()

    class BeforeInstallation(ModelPolicyStore):
        async def execute(self, command):
            if command.action == "write" and parse_state(command.state)["current"] is not None:
                raise OSError("installation unavailable")
            return await native.execute(command)

    app, controller, _ = make_app(BeforeInstallation(), channel)
    await app.start_model_policy()
    await controller._owner.reconcile()
    observation = controller._owner.state()["observation"]
    assert observation is not None and controller.selection() is None
    assert not channel.receipts
    await app.stop_model_policy()
    await native.close()
    reopened = store_factory()
    channel.unavailable = True
    replacement, other, _ = make_app(reopened, channel)
    try:
        await replacement.start_model_policy()
        assert other.selection() is None
        await other.adopt_cached()
        assert other.selection().target.model == "model-a"
        assert other._owner.state()["observation"] == observation
        assert len(other._owner.pending()) == 1
        assert not channel.receipts
    finally:
        await replacement.stop_model_policy()
        await reopened.close()


async def test_native_precommit_deadline_rolls_back_sql_mutation(store_factory, monkeypatch):
    import importlib
    from types import SimpleNamespace

    native, channel = store_factory(), Channel()
    if type(native).__name__ == "InMemoryModelPolicyStore":
        pytest.skip("Memory transition has no awaited SQL/commit boundary")
    app, controller, _ = make_app(native, channel)
    await app.start_model_policy()
    owner = controller._owner
    before = owner._view
    deadline = time.monotonic_ns() + 55_000_000_000
    module = importlib.import_module(type(native).__module__)
    try:
        # The shared transition sees a valid deadline; only the native check
        # after INSERT sees expiry. The real database transaction must roll back.
        with monkeypatch.context() as patch:
            patch.setattr(module, "time", SimpleNamespace(monotonic_ns=lambda: deadline))
            with pytest.raises(PolicyContractError):
                await native.execute(
                    PolicyStorageCommand(
                        "write", owner._binding, owner._owner, before, before.state, deadline
                    )
                )
        await owner.reconcile()
        assert owner._view.revision == before.revision
        assert owner._view.state == before.state
        assert controller.selection().target.model == "model-a"
    finally:
        await app.stop_model_policy()
        await native.close()


async def test_startup_failure_then_real_cancellation_preserves_both(store_factory):
    native = store_factory()
    entered, finish = asyncio.Event(), asyncio.Event()

    class Release(ModelPolicyStore):
        async def execute(self, command):
            if command.action == "release":
                entered.set()
                await finish.wait()
            return await native.execute(command)

    _, first, provider = make_app(Release(), Channel())
    second_channel = Channel()
    second = ModelPolicyController(
        store=Release(),
        agent_name="second",
        provider_name="missing",
        scope=second_channel.scope,
        incarnation=second_channel.incarnation,
        channel=second_channel,
    )
    app = CayuApp(model_policy=ModelPolicy([first, second]), enable_logging=False)
    app.register_provider(provider, default=True)
    for name in ("assistant", "second"):
        app.register_agent(AgentSpec(name=name, model="model-a"))
    task = asyncio.create_task(app.start_model_policy())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert task.cancelled() and task.cancelling() == 1
        assert isinstance(caught.value.__cause__, BaseExceptionGroup)
        assert len(caught.value.__cause__.exceptions) == 1
        assert isinstance(caught.value.__cause__.exceptions[0], KeyError)
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
        await app.stop_model_policy()
        await native.close()


@pytest.mark.parametrize(
    "phase",
    [
        "late_publication",
        "lost_publication_ack",
        "lost_confirmation_ack",
        "cancel_publication",
        "cancel_confirmation",
    ],
)
async def test_deadline_publication_requires_positive_completion_evidence(
    store_factory, phase, monkeypatch
):
    native, channel = store_factory(), Channel()
    real_ns = time.monotonic_ns
    offset = 0
    entered, finish = asyncio.Event(), asyncio.Event()
    # Change only this owner's clock, not asyncio's deadlines or database clocks.
    from types import SimpleNamespace

    import cayu.runtime._policy_installation as installation

    monkeypatch.setattr(
        installation,
        "time",
        SimpleNamespace(monotonic=time.monotonic, monotonic_ns=lambda: real_ns() + offset),
    )

    class Publication(ModelPolicyStore):
        armed = False

        async def execute(self, command):
            nonlocal offset
            result = await native.execute(command)
            if self.armed and command.action == "write":
                publication = parse_state(command.state).get("publication")
                if (
                    publication is not None
                    and parse_state(command.state)["current"]["installed_model"] == "model-b"
                ):
                    confirmed = publication["completed_ns"] is not None
                    if phase.startswith("cancel_"):
                        if confirmed == (phase == "cancel_confirmation"):
                            self.armed = False
                            entered.set()
                            await finish.wait()
                        return result
                    if phase == "lost_confirmation_ack" and confirmed:
                        self.armed = False
                        offset += 60_000_000_000
                        raise OSError("lost confirmation acknowledgement")
                    if phase != "lost_confirmation_ack" and not confirmed:
                        self.armed = False
                        if phase == "late_publication":
                            offset = publication["deadline_ns"] - real_ns()
                        else:
                            raise OSError("lost publication acknowledgement")
            return result

    store = Publication()
    app, controller, _ = make_app(store, channel)
    await app.start_model_policy()
    try:
        channel.model, channel.revision = "model-b", 2
        store.armed = True
        if phase.startswith("cancel_"):
            task = asyncio.create_task(controller.poll_once())
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled() and task.cancelling() == 1
        else:
            with pytest.raises((OSError, PolicyContractError)):
                await controller.poll_once()
        await controller._owner.reconcile()
        committed = phase in ("lost_confirmation_ack", "cancel_confirmation")
        expected = "model-b" if committed else "model-a"
        assert controller.selection().target.model == expected
        pending = controller._owner.pending()
        assert len(pending) == (1 if committed else 0)
        assert len(channel.receipts) == 1
        await app.stop_model_policy()
        await native.close()
        native = store_factory()
        channel.unavailable = True
        app, controller, _ = make_app(native, channel)
        await app.start_model_policy()
        assert controller.selection().target.model == expected
        assert controller._owner.pending() == pending
    finally:
        finish.set()
        await app.stop_model_policy()
        await native.close()
