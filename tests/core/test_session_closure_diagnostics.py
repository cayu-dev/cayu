"""Recursive public reports do not expose extension-owned failure text."""

import asyncio
from uuid import uuid4

import pytest
from tests.core.session_closure_conformance import create_closure_session
from tests.core.test_session_closure import _DependentStore
from tests.core.test_tool_effect_store_conformance import _stores

from cayu import CayuApp
from cayu.runtime.session_closure import SessionClosurePolicy


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("failure", [ValueError, RuntimeError])
@pytest.mark.parametrize("phase", ["child_erase", "progress_read"])
def test_recursive_public_failure_omits_private_diagnostics(
    backend, failure, phase, tmp_path, request, monkeypatch, capsys, caplog, recwarn
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    canary = "unregistered-private-store-configuration"

    async def scenario():
        async with _stores(backend, tmp_path / "diagnostics.db", dsn) as open_store:
            store = open_store()
            root, child = uuid4().hex, uuid4().hex
            await create_closure_session(store, root)
            await create_closure_session(store, child, root)
            policy = SessionClosurePolicy(child_policy="recursive")

            class Adapter(_DependentStore):
                reject = True

                async def erase_session_closure(self, session_id, *, policy, plan_id):
                    if self.reject and phase == "child_erase" and session_id == child:
                        raise failure(canary)
                    return await super().erase_session_closure(
                        session_id, policy=policy, plan_id=plan_id
                    )

            adapter = Adapter()
            app = CayuApp(
                session_store=store, session_closure_stores=(adapter,), enable_logging=False
            )
            if phase == "progress_read":
                original_delete = store.delete_session

                async def fail_root(session_id, **kwargs):
                    if session_id == root:
                        raise RuntimeError("root deletion temporarily unavailable")
                    return await original_delete(session_id, **kwargs)

                monkeypatch.setattr(store, "delete_session", fail_root)
                first = await app.erase_session_closure(root, policy=policy)
                assert not first.complete
                assert await store.load(child) is None
                monkeypatch.setattr(store, "delete_session", original_delete)
                # Reconstruct the coordinator and persistent store before retry.
                store = open_store()
                original_load = store.load_session_closure_receipt

                async def fail_child_receipt(session_id, plan_id):
                    if session_id == child:
                        raise failure(canary)
                    return await original_load(session_id, plan_id)

                monkeypatch.setattr(store, "load_session_closure_receipt", fail_child_receipt)
                app = CayuApp(
                    session_store=store, session_closure_stores=(adapter,), enable_logging=False
                )

            report = await app.erase_session_closure(root, policy=policy)
            assert not report.complete
            expected = (
                "Recursive closure progress could not be validated."
                if phase == "progress_read"
                else "Recursive closure stopped before durable completion."
            )
            assert report.error == expected
            assert canary not in report.model_dump_json()
            assert await store.load(root) is not None
            assert await store.load_session_closure_receipt(root, report.plan_id) is None
            if phase == "child_erase":
                assert await store.load(child) is not None
                assert not adapter.erased
                adapter.reject = False
            else:
                monkeypatch.setattr(store, "load_session_closure_receipt", original_load)
            completed = await app.erase_session_closure(root, policy=policy)
            assert completed.complete
            assert canary not in completed.model_dump_json()
            receipt = await store.load_session_closure_receipt(root, report.plan_id)
            assert receipt is not None and canary not in repr(receipt)
            assert await store.load(root) is None
            assert await store.load(child) is None

    asyncio.run(scenario())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert all(canary not in str(warning.message) for warning in recwarn)
